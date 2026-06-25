"""Online Recursive ICA decomposer (ORICA).

Functional reference
--------------------
``ORICA_final_no_print_quick30.py`` (legacy ORICA receiver).

The update logic mirrors that implementation:

* two-pass RLS whitening then ICA (not interleaved per block)
* whitening on channel-centred data; ICA mixtures use ``sphere @ raw``
* even block partitioning via ``n_splits = n_pts // block_size``
* forgetting factor forced to ``lambda_const`` (quick30 ``if True`` branch)
* initial sample counter ``7681``
* ``scipy.linalg.eigh`` orthogonalisation
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


def _orica_block_ranges(n_pts: int, block_size: int):
    """Yield (start, end) sample ranges for one ORICA update pass.

    Matches legacy ORICA (``ORICA_final_no_print_quick30.py``):

    * ``n_splits = n_pts // block_size`` (floor)
    * partition ``[0, n_pts)`` into ``n_splits`` contiguous segments of
      nearly equal length (e.g. 100 pts / block_size 32 → 33+33+34)
    * when ``n_pts < block_size``, ``n_splits == 0`` and nothing is yielded
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    n_splits = n_pts // block_size
    for bi in range(n_splits):
        start = int(bi * n_pts / n_splits)
        end = min(n_pts, int((bi + 1) * n_pts / n_splits))
        if start < end:
            yield start, end


def _snap_to_kbits(x, k=50):
    """Legacy MATLAB float-snap stub (no-op in quick30)."""
    return x


class ORICAFilter:
    """Online Recursive ICA with RLS whitening (quick30-compatible).

    Parameters
    ----------
    n_components : int
        Number of EEG channels / independent components.
    sfreq : float
        Sampling frequency in Hz.
    ff_profile : {'cooling', 'constant', 'adaptive'}
        Forgetting-factor profile label (see ``force_constant_lambda``).
    block_size_white : int
        Block size for RLS whitening updates (sets ``numsplits``).
    block_size_ica : int
        Block size used to count ICA outer blocks
        (``floor(n_pts / block_size_ica)`` in quick30).
    tau_const : float
        Local stationarity window (seconds).  ``inf`` → ``lambda_const=0.98``.
    gamma : float
        Cooling decay rate.
    lambda_0 : float
        Initial forgetting factor (used when ``force_constant_lambda=False``).
    num_subgaussian : int
        Number of sub-Gaussian sources (0 = all super-Gaussian).
    force_constant_lambda : bool
        When True (default), always use ``lambda_const`` for every sample,
        matching quick30's hard-coded ``if True`` branch.
    time_perm : bool
        Randomly permute sample order during the ICA pass (quick30 option).
    num_pass : int
        Number of passes over the data per ``fit`` / ``update`` call.
    """

    def __init__(
        self,
        n_components: int,
        sfreq: float,
        ff_profile: str = "cooling",
        block_size_white: int = 8,
        block_size_ica: int = 8,
        tau_const: float = 3.0,
        gamma: float = 0.6,
        lambda_0: float = 0.995,
        num_subgaussian: int = 0,
        force_constant_lambda: bool = True,
        time_perm: bool = False,
        num_pass: int = 1,
    ) -> None:
        self.n_components = n_components
        self.sfreq = sfreq
        self.ff_profile = ff_profile
        self.block_size_white = block_size_white
        self.block_size_ica = block_size_ica
        self.tau_const = tau_const
        self.gamma = gamma
        self.lambda_0 = lambda_0
        self.force_constant_lambda = force_constant_lambda
        self.time_perm = time_perm
        self.num_pass = num_pass

        # quick30: lambda_const from tau_const (seconds) × srate
        if np.isfinite(tau_const):
            self._lambda_const = 1.0 - np.exp(-1.0 / (tau_const * sfreq))
        else:
            self._lambda_const = 0.98

        self._kurtosis_sign = np.ones(n_components, dtype=bool)
        if num_subgaussian > 0:
            self._kurtosis_sign[:num_subgaussian] = False

        self._W = np.eye(n_components, dtype=np.float64)
        self._sphere = np.eye(n_components, dtype=np.float64)
        self._counter = 0  # quick30 initial counter
        self._Rn = None
        self._min_non_stat_idx = None
        self._lambda_k = np.zeros(block_size_ica, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public attributes
    # ------------------------------------------------------------------

    @property
    def weights_(self) -> np.ndarray:
        return self._W

    @property
    def sphere_(self) -> np.ndarray:
        return self._sphere

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, calibration_data: np.ndarray) -> "ORICAFilter":
        if calibration_data.shape[0] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} channels, "
                f"got {calibration_data.shape[0]}"
            )
        self._run_orica(calibration_data)
        return self

    def update(self, chunk: np.ndarray) -> None:
        if chunk.shape[0] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} channels, "
                f"got {chunk.shape[0]}"
            )
        self._run_orica(chunk)

    def transform(self, chunk: np.ndarray) -> np.ndarray:
        """Project channels to source space (quick30: no centreing)."""
        return self._W @ (self._sphere @ chunk)

    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        Xw = np.linalg.pinv(self._W) @ sources
        return np.linalg.pinv(self._sphere) @ Xw

    # ------------------------------------------------------------------
    # Forgetting factor (quick30)
    # ------------------------------------------------------------------

    def _gen_cooling_ff(self, t: np.ndarray) -> np.ndarray:
        t_safe = np.maximum(t, 1e-10)
        lam = self.lambda_0 / np.power(t_safe, self.gamma)
        return _snap_to_kbits(lam, k=50)

    def _gen_adaptive_ff(
        self,
        data_range: np.ndarray,
        lambda_vec: np.ndarray,
        ratio_of_norm_rn: float,
        decay_rate_alpha: float = 0.02,
        upper_bound_beta: float = 0.001,
        trans_band_width_gamma: float = 0.05,
        trans_band_center: float = 5.0,
    ) -> np.ndarray:
        """Replicate quick30 ``gen_adaptive_ff``."""
        n_pts = len(data_range)
        lam0 = float(np.asarray(lambda_vec)[-1])
        gain = upper_bound_beta * 0.5 * (
            1.0 + np.tanh(
                (ratio_of_norm_rn - trans_band_center) / trans_band_width_gamma
            )
        )
        n = np.arange(1, n_pts + 1, dtype=np.float64)
        one_plus_g = 1.0 + gain
        term1 = (one_plus_g ** n) * lam0
        eps = 1e-12
        if abs(gain) < eps:
            frac = n
        else:
            frac = (
                (one_plus_g ** (2.0 * n - 1.0)) - (one_plus_g ** (n - 1.0))
            ) / gain
        term2 = decay_rate_alpha * frac * (lam0 ** 2)
        return term1 - term2

    def _forgetting_factor(self, data_range_1idx: np.ndarray) -> np.ndarray:
        """Per-sample λ for the current block (1-indexed ``data_range``)."""
        if self.ff_profile == "adaptive" and self._Rn is not None:
            ratio = 1.0
            if self._min_non_stat_idx is not None and self._min_non_stat_idx > 0:
                ratio = float(
                    np.linalg.norm(self._Rn, "fro") / self._min_non_stat_idx
                )
            lam = self._gen_adaptive_ff(
                data_range_1idx,
                self._lambda_k,
                ratio,
            )
        else:
            lam = self._gen_cooling_ff(self._counter + data_range_1idx)

        # quick30: ``if True`` — always clamp to lambda_const
        if self.force_constant_lambda or self.ff_profile == "constant":
            return np.full(len(data_range_1idx), self._lambda_const)

        if lam[0] < self._lambda_const:
            return np.full(len(data_range_1idx), self._lambda_const)
        return lam

    # ------------------------------------------------------------------
    # Block updates (quick30 dynamic_whitening / dynamic_orica_cooling)
    # ------------------------------------------------------------------

    def _dynamic_whitening(
        self,
        blockdata: np.ndarray,
        data_range_1idx: np.ndarray,
    ) -> None:
        n_pts = blockdata.shape[1]
        lam = self._forgetting_factor(data_range_1idx)
        lam_avg = 1.0 - lam[int(np.ceil(len(lam) / 2)) - 1]

        v = self._sphere @ blockdata
        v = _snap_to_kbits(v, k=38)

        Q_white = lam_avg / (1.0 - lam_avg) + (
            np.linalg.norm(v, "fro") ** 2 / len(data_range_1idx)
        )
        Q_white = _snap_to_kbits(Q_white, k=38)

        update_term = (v @ v.T) / n_pts / Q_white @ self._sphere
        self._sphere = (1.0 / lam_avg) * (self._sphere - update_term)

    def _dynamic_orica(
        self,
        blockdata: np.ndarray,
        data_range_1idx: np.ndarray,
    ) -> None:
        n_chs, n_pts = blockdata.shape
        Y = self._W @ blockdata

        F = np.empty_like(Y)
        F[self._kurtosis_sign] = -2.0 * np.tanh(Y[self._kurtosis_sign])
        F[~self._kurtosis_sign] = (
            np.tanh(Y[~self._kurtosis_sign]) - Y[~self._kurtosis_sign]
        )

        model_fitness = np.eye(n_chs) + (Y @ F.T) / n_pts
        if self._Rn is None:
            self._Rn = model_fitness
        else:
            self._Rn = 0.99 * self._Rn + 0.01 * model_fitness

        non_stat = float(np.linalg.norm(self._Rn, "fro"))
        if self._min_non_stat_idx is None:
            self._min_non_stat_idx = non_stat
        else:
            self._min_non_stat_idx = min(self._min_non_stat_idx, non_stat)

        lam = self._forgetting_factor(data_range_1idx)
        self._counter += n_pts
        self._lambda_k = lam

        lam_prod = np.prod(1.0 / (1.0 - lam))
        Q = 1.0 + lam * (np.sum(F * Y, axis=0) - 1.0)
        F = _snap_to_kbits(F, k=44)

        self._W = lam_prod * (
            self._W - Y @ np.diag(lam / Q) @ F.T @ self._W
        )

        D, V = eigh(self._W @ self._W.T)
        D = np.diag(D)
        D = _snap_to_kbits(D, k=32)
        V = _snap_to_kbits(V, k=32)

        d = np.diag(D)
        inv_sqrt = 1.0 / np.sqrt(d)
        M = V @ np.diag(inv_sqrt) @ V.conj().T
        self._W = _snap_to_kbits(M @ self._W, k=40)

    # ------------------------------------------------------------------
    # Main loop (quick30 orica_rls_whitening)
    # ------------------------------------------------------------------

    def _run_orica(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float64)
        n_chs, n_pts = data.shape
        data_center = data - data.mean(axis=1, keepdims=True)

        numsplits = n_pts // self.block_size_white
        num_block_white = n_pts // self.block_size_white
        num_block_ica = n_pts // self.block_size_ica



        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None

        try:
            # Pass 1 — RLS whitening (centred data; counter not incremented)
            for _ in range(self.num_pass):
                for bi in range(num_block_white):
                    start = int(bi * n_pts / numsplits)
                    end = min(n_pts, int((bi + 1) * n_pts / numsplits))
                    if start >= end:
                        continue
                    data_range_0 = np.arange(start, end)
                    block = data_center[:, data_range_0]
                    self._dynamic_whitening(block, data_range_0 + 1)

            # mixtures from raw (uncentred) data — quick30 convention
            mixtures = self._sphere @ data

            if self.time_perm:
                perm_idx = np.random.permutation(n_pts)
            else:
                perm_idx = np.arange(n_pts)

            # Pass 2 — ICA (mixtures[:, perm_idx]; counter incremented per block)
            for _ in range(self.num_pass):
                for bi in range(num_block_ica):
                    start = int(bi * n_pts / numsplits)
                    end = min(n_pts, int((bi + 1) * n_pts / numsplits))
                    if start >= end:
                        continue
                    data_range_0 = np.arange(start, end)
                    perm_range = perm_idx[data_range_0]
                    self._dynamic_orica(
                        mixtures[:, perm_range],
                        data_range_0 + 1,
                    )
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)


# ------------------------------------------------------------------
# Standalone test entry (shared with ORICA_final_no_print_quick30.py)
# ------------------------------------------------------------------
import numpy as np

np.set_printoptions(
    threshold=np.inf,  # 不省略
    linewidth=np.inf,  # 不换行
)


_DEFAULT_INPUT = Path(__file__).resolve().parent / "input.txt"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output_core.txt"

# Shared runtime parameters for cross-implementation comparison.
_TEST_BLOCK_SIZE_WHITE = 32
_TEST_BLOCK_SIZE_ICA = 32
_TEST_SFREQ = 500.0
_TEST_TAU_CONST = 3.0
_TEST_GAMMA = 0.6
_TEST_LAMBDA_0 = 0.995
_TEST_NUM_PASS = 1


@contextmanager
def _capture_stdout():
    """Mirror stdout to a buffer so prints can be saved to the output file."""
    buffer = io.StringIO()
    stdout = sys.stdout

    class _TeeStdout:
        def write(self, text: str) -> int:
            stdout.write(text)
            buffer.write(text)
            return len(text)

        def flush(self) -> None:
            stdout.flush()

    sys.stdout = _TeeStdout()
    try:
        yield buffer
    finally:
        sys.stdout = stdout


def _load_input_txt(path: Path) -> tuple[np.ndarray, dict[str, float | int]]:
    """Load ``input.txt`` written by the shared test-data generator."""
    meta: dict[str, float | int] = {}
    rows: list[list[float]] = []
    in_data = False
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line == "---":
                in_data = True
                continue
            if not in_data:
                key, value = line.split("=", 1)
                meta[key.strip()] = float(value) if "." in value else int(value)
            else:
                rows.append([float(x) for x in line.split()])
    data = np.asarray(rows, dtype=np.float64)
    return data, meta


def _save_output_txt(
    path: Path,
    *,
    label: str,
    meta: dict[str, float | int],
    sphere: np.ndarray,
    weights: np.ndarray,
    sources: np.ndarray,
    counter: float | int,
    log: str = "",
) -> None:
    lines = [
        f"implementation={label}",
        f"n_ch={meta.get('n_ch', sphere.shape[0])}",
        f"n_pts={meta.get('n_pts', sources.shape[1])}",
        f"sfreq={meta.get('sfreq', _TEST_SFREQ)}",
        f"block_size_white={_TEST_BLOCK_SIZE_WHITE}",
        f"block_size_ica={_TEST_BLOCK_SIZE_ICA}",
        f"counter={counter}",
        f"sphere_fro={float(np.linalg.norm(sphere, 'fro')):.17g}",
        f"weights_fro={float(np.linalg.norm(weights, 'fro')):.17g}",
        f"sources_fro={float(np.linalg.norm(sources, 'fro')):.17g}",
        "--- log ---",
    ]
    if log:
        lines.extend(log.rstrip("\n").splitlines())
    else:
        lines.append("(empty)")
    lines.append("--- sphere ---")
    for row in sphere:
        lines.append(" ".join(f"{v:.17g}" for v in row))
    lines.append("--- weights ---")
    for row in weights:
        lines.append(" ".join(f"{v:.17g}" for v in row))
    lines.append("--- sources (first 8 samples per component) ---")
    preview = sources[:, : min(8, sources.shape[1])]
    for row in preview:
        lines.append(" ".join(f"{v:.17g}" for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(input_path: Path | None = None, output_path: Path | None = None) -> None:
    input_path = Path(input_path or _DEFAULT_INPUT)
    output_path = Path(output_path or _DEFAULT_OUTPUT)

    data, meta = _load_input_txt(input_path)
    n_ch = int(meta.get("n_ch", data.shape[0]))
    sfreq = float(meta.get("sfreq", _TEST_SFREQ))

    with _capture_stdout() as log_buffer:
        print(f"[core.py] input: {input_path}")
        print(f"  data.shape = {data.shape}, sfreq = {sfreq}")
        print(
            f"  block_size_white = {_TEST_BLOCK_SIZE_WHITE}, "
            f"block_size_ica = {_TEST_BLOCK_SIZE_ICA}"
        )

        filt = ORICAFilter(
            n_components=n_ch,
            sfreq=sfreq,
            block_size_white=_TEST_BLOCK_SIZE_WHITE,
            block_size_ica=_TEST_BLOCK_SIZE_ICA,
            tau_const=_TEST_TAU_CONST,
            gamma=_TEST_GAMMA,
            lambda_0=_TEST_LAMBDA_0,
            force_constant_lambda=True,
            time_perm=False,
            num_pass=_TEST_NUM_PASS,
        )
        filt.fit(data)
        sources = filt.transform(data)

        print(f"  counter = {filt._counter}")
        print(f"  ||sphere||_F = {np.linalg.norm(filt.sphere_, 'fro'):.6e}")
        print(f"  ||weights||_F = {np.linalg.norm(filt.weights_, 'fro'):.6e}")
        print(f"  ||sources||_F = {np.linalg.norm(sources, 'fro'):.6e}")
        print(f"  output -> {output_path}")

    _save_output_txt(
        output_path,
        label="pyorica.core.ORICAFilter",
        meta=meta,
        sphere=filt.sphere_,
        weights=filt.weights_,
        sources=sources,
        counter=filt._counter,
        log=log_buffer.getvalue(),
    )


if __name__ == "__main__":
    main()

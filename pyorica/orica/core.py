"""Online Recursive ICA decomposer (ORICA).

Reference
---------
Hsu, S.-H., Mullen, T., Jung, T.-P., & Cauwenberghs, G. (2016).
Real-time adaptive EEG source separation using online recursive independent
component analysis. IEEE Transactions on Neural Systems and Rehabilitation
Engineering, 24(3), 309-319.

Numerics note
-------------
Block partitioning, the two-pass (whiten-then-decompose) update ordering, and
the ``force_constant_lambda`` default are aligned with the quick30 legacy
reference implementation (``ORICA_final_no_print_quick30.py``) rather than a
literal reading of the paper — see ADR-0006 for why.
"""

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


class ORICAFilter:
    """Online Recursive ICA with RLS whitening.

    Parameters
    ----------
    n_components : int
        Number of EEG channels / independent components.
    sfreq : float
        Sampling frequency in Hz.
    ff_profile : {'cooling', 'constant', 'adaptive'}
        Forgetting-factor profile. Only takes effect when
        ``force_constant_lambda=False`` — see below.
    block_size_white : int
        Block size for RLS whitening updates. Independent of
        ``block_size_ica``: it reflects whitening's own stationarity
        assumption, not the real-time chunk size.
    block_size_ica : int
        Block size for ICA weight updates. Independent of
        ``block_size_white`` for the same reason.
    tau_const : float
        Local stationarity window (seconds). ``inf`` → ``lambda_const = 0.98``.
    gamma : float
        Decay rate for the cooling forgetting factor.
    lambda_0 : float
        Initial forgetting factor (cooling profile).
    num_subgaussian : int
        Number of sub-Gaussian sources (default 0; EEG brain sources are
        typically super-Gaussian).
    force_constant_lambda : bool
        When True (default), every update uses ``lambda_const`` regardless of
        ``ff_profile`` — matches quick30's real-time behavior. Set False to
        let ``ff_profile="cooling"`` (stationary validation) or
        ``"adaptive"`` (nonstationary real-time) actually drive the
        forgetting factor.
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

        # steady-state lambda. tau_const=inf means "no steady-state floor" —
        # cooling/adaptive decay toward 0 unconstrained. A nonzero floor here
        # (e.g. quick30's 0.98) makes lam_prod = prod(1/(1-lambda)) diverge
        # within ~100 updates at typical block sizes; see ADR-0006.
        if np.isfinite(tau_const):
            self._lambda_const = 1.0 - np.exp(-1.0 / (tau_const * sfreq))
        else:
            self._lambda_const = 0.0

        # kurtosis sign: True = super-Gaussian, False = sub-Gaussian
        self._kurtosis_sign = np.ones(n_components, dtype=bool)
        if num_subgaussian > 0:
            self._kurtosis_sign[:num_subgaussian] = False

        # state
        self._W = np.eye(n_components, dtype=np.float64)        # ICA weight matrix
        self._sphere = np.eye(n_components, dtype=np.float64)   # whitening matrix
        self._counter = 0
        self._Rn = None                         # leaky average for NSI
        self._min_non_stat_idx = None           # running min ||Rn||_F, for adaptive ff
        self._lambda_k = np.zeros(1)             # most recent lambda block, adaptive ff

    # ------------------------------------------------------------------
    # Public attributes
    # ------------------------------------------------------------------

    @property
    def weights_(self) -> np.ndarray:
        """ICA weight matrix W, shape (n_components, n_components)."""
        return self._W

    @property
    def sphere_(self) -> np.ndarray:
        """Whitening matrix (sphere), shape (n_components, n_components)."""
        return self._sphere

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, calibration_data: np.ndarray) -> "ORICAFilter":
        """Warm-start weights from a calibration recording.

        Parameters
        ----------
        calibration_data : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        self
        """
        if calibration_data.shape[0] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} channels, "
                f"got {calibration_data.shape[0]}"
            )
        self._run_orica(calibration_data)
        return self

    def update(self, chunk: np.ndarray) -> None:
        """Incrementally update W and sphere from one chunk.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_samples)
        """
        if chunk.shape[0] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} channels, "
                f"got {chunk.shape[0]}"
            )
        self._run_orica(chunk)

    def transform(self, chunk: np.ndarray) -> np.ndarray:
        """Project EEG channels to source space.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        np.ndarray, shape (n_components, n_samples)
        """
        return self._W @ (self._sphere @ chunk)

    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        """Reconstruct sensor-space signals from (modified) sources.

        Parameters
        ----------
        sources : np.ndarray, shape (n_components, n_samples)

        Returns
        -------
        np.ndarray, shape (n_channels, n_samples)
        """
        A = np.linalg.pinv(self._W @ self._sphere)
        return A @ sources

    # ------------------------------------------------------------------
    # Forgetting factor
    # ------------------------------------------------------------------

    def _gen_cooling_ff(self, t: np.ndarray) -> np.ndarray:
        return self.lambda_0 / np.power(np.maximum(t, 1e-10), self.gamma)

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
        """Replicate quick30's ``gen_adaptive_ff``."""
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
            lam = self._gen_adaptive_ff(data_range_1idx, self._lambda_k, ratio)
        else:
            lam = self._gen_cooling_ff(self._counter + data_range_1idx)

        if self.force_constant_lambda or self.ff_profile == "constant":
            return np.full(len(data_range_1idx), self._lambda_const)

        if lam[0] < self._lambda_const:
            return np.full(len(data_range_1idx), self._lambda_const)
        return lam

    # ------------------------------------------------------------------
    # Block updates
    # ------------------------------------------------------------------

    def _dynamic_whitening(
        self, blockdata: np.ndarray, data_range_1idx: np.ndarray
    ) -> None:
        n_pts = blockdata.shape[1]
        lam = self._forgetting_factor(data_range_1idx)
        lam_avg = 1.0 - lam[int(np.ceil(len(lam) / 2)) - 1]

        v = self._sphere @ blockdata
        Q = lam_avg / (1.0 - lam_avg) + np.linalg.norm(v, "fro") ** 2 / n_pts
        self._sphere = (1.0 / lam_avg) * (
            self._sphere - (v @ v.T) / n_pts / Q @ self._sphere
        )

    def _dynamic_orica(
        self, blockdata: np.ndarray, data_range_1idx: np.ndarray
    ) -> None:
        n_pts = blockdata.shape[1]
        Y = self._W @ blockdata

        F = np.empty_like(Y)
        F[self._kurtosis_sign] = -2.0 * np.tanh(Y[self._kurtosis_sign])
        F[~self._kurtosis_sign] = (
            np.tanh(Y[~self._kurtosis_sign]) - Y[~self._kurtosis_sign]
        )

        model_fitness = np.eye(self.n_components) + (Y @ F.T) / n_pts
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
        self._W = lam_prod * (self._W - Y @ np.diag(lam / Q) @ F.T @ self._W)

        # orthogonalise
        D, V = eigh(self._W @ self._W.T)
        D = np.maximum(D, 1e-12)
        self._W = (V @ np.diag(1.0 / np.sqrt(D)) @ V.T) @ self._W

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_orica(self, data: np.ndarray) -> None:
        """Run whitening then ICA over data (n_channels, n_samples).

        Two-pass: a full RLS-whitening pass over the (centred) chunk, then a
        full ICA pass over the resulting (uncentred) mixtures — matching
        quick30 rather than the previous per-block interleaving. The two
        passes partition the chunk independently via ``block_size_white`` and
        ``block_size_ica`` respectively; they must not share one partition
        derived from only one of the two sizes.
        """
        data = np.asarray(data, dtype=np.float64)
        n_pts = data.shape[1]
        data_center = data - data.mean(axis=1, keepdims=True)

        # All ORICA matrix ops are on tiny (n_channels × n_channels) matrices.
        # Pin BLAS to 1 thread for the duration of this loop — identical fix as
        # asr_process; prevents OpenBLAS thread-dispatch overhead (~0.2s per eigh
        # call) from dominating when many workers run in parallel.
        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None

        try:
            for _ in range(self.num_pass):
                for start, end in _orica_block_ranges(n_pts, self.block_size_white):
                    data_range = np.arange(start, end) + 1  # 1-indexed, matches MATLAB
                    self._dynamic_whitening(data_center[:, start:end], data_range)

            # mixtures from raw (uncentred) data — quick30 convention
            mixtures = self._sphere @ data

            if self.time_perm:
                perm_idx = np.random.permutation(n_pts)
            else:
                perm_idx = np.arange(n_pts)

            for _ in range(self.num_pass):
                for start, end in _orica_block_ranges(n_pts, self.block_size_ica):
                    data_range = np.arange(start, end) + 1  # 1-indexed, matches MATLAB
                    perm_range = perm_idx[start:end]
                    self._dynamic_orica(mixtures[:, perm_range], data_range)
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)

"""ASRAdapter: unified interface for asrpy (default) and meegkit ASR backends."""

import numpy as np

_VALID_BACKENDS = ("asrpy", "meegkit")


class ASRAdapter:
    """Wraps asrpy or meegkit ASR behind a common fit/transform interface.

    Parameters
    ----------
    backend : {"asrpy", "meegkit"}
        ASR implementation. "asrpy" matches the reference experiments.
    sfreq : float
        Sampling frequency in Hz.
    cutoff : float
        Standard-deviation multiplier for artifact rejection (default 20.0).
    """

    def __init__(self, backend: str = "asrpy", sfreq: float = 256.0,
                 cutoff: float = 20.0):
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend={backend!r} is not valid. Choose from {_VALID_BACKENDS}."
            )
        self._backend = backend
        self._sfreq = sfreq
        self._cutoff = cutoff
        self._fitted = False

        # asrpy stateful transform state
        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None
        self._asr_inst = None   # asrpy or meegkit ASR object

    def fit(self, data: np.ndarray) -> "ASRAdapter":
        """Calibrate ASR on clean data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
        """
        if self._backend == "asrpy":
            self._fit_asrpy(data)
        else:
            self._fit_meegkit(data)
        self._fitted = True
        return self

    def transform(self, chunk: np.ndarray) -> np.ndarray:
        """Apply ASR to a chunk.

        Parameters
        ----------
        chunk : ndarray, shape (n_channels, n_samples)

        Returns
        -------
        ndarray, shape (n_channels, n_samples)
        """
        if not self._fitted:
            return chunk
        if self._backend == "asrpy":
            return self._transform_asrpy(chunk)
        return self._transform_meegkit(chunk)

    # ── asrpy ────────────────────────────────────────────────────────────────

    def _fit_asrpy(self, data: np.ndarray) -> None:
        try:
            import asrpy
        except ImportError as exc:
            raise ImportError(
                "asrpy is required for backend='asrpy'. "
                "Install it with: pip install asrpy"
            ) from exc
        asr = asrpy.ASR(sfreq=float(self._sfreq), cutoff=float(self._cutoff))
        asr.fit(data)
        self._asr_inst = asr
        # reset stateful transform accumulators
        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None

    def _transform_asrpy(self, chunk: np.ndarray) -> np.ndarray:
        from asrpy.asr import asr_process
        asr = self._asr_inst
        n_ch, n_samples = chunk.shape
        lookahead = 0.25          # seconds; matches original ORICA receiver
        stepsize = 32
        maxdims = 0.66
        mem_splits = 1
        ls = int(self._sfreq * lookahead)
        x = np.asarray(chunk, dtype=np.float64)
        # pad end with zeros so the lookahead window is always filled
        X_in = np.concatenate([x, np.zeros((n_ch, ls), dtype=np.float64)], axis=1)
        out, st = asr_process(
            X_in,
            self._sfreq,
            asr.M,
            asr.T,
            asr.win_len,
            float(lookahead),
            int(stepsize),
            float(maxdims),
            (asr.A, asr.B),
            self._asr_R,
            self._asr_Zi,
            self._asr_cov,
            None,
            True,
            asr.method,
            int(mem_splits),
        )
        self._asr_R = st["R"]
        self._asr_Zi = st["Zi"]
        self._asr_cov = st["cov"]
        # strip the lookahead padding and match input length exactly
        out = np.asarray(out[:, ls:], dtype=np.float64)
        if out.shape[1] > n_samples:
            out = out[:, -n_samples:]
        elif out.shape[1] < n_samples:
            out = np.pad(out, ((0, 0), (n_samples - out.shape[1], 0)), mode="edge")
        return out

    # ── meegkit ──────────────────────────────────────────────────────────────

    def _fit_meegkit(self, data: np.ndarray) -> None:
        try:
            from meegkit.asr import ASR
        except ImportError as exc:
            raise ImportError(
                "meegkit is required for backend='meegkit'. "
                "Install it with: pip install meegkit"
            ) from exc
        asr = ASR(sfreq=float(self._sfreq))
        asr.fit(data)
        self._asr_inst = asr

    def _transform_meegkit(self, chunk: np.ndarray) -> np.ndarray:
        return self._asr_inst.transform(chunk)

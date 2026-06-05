"""ASRAdapter: unified interface for asrpy (default) and meegkit ASR backends."""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from pyorica.pipeline.asr_calib import load_asr_calibration_npz
_VALID_BACKENDS = ("asrpy", "meegkit")
_ASRPY_NUMPY2_PATCHED = False


def _patch_asrpy_for_numpy2() -> None:
    """Replace ``asrpy.asr_utils.fit_eeg_distribution`` with a NumPy 2.x-safe copy."""
    global _ASRPY_NUMPY2_PATCHED
    if _ASRPY_NUMPY2_PATCHED:
        return
    try:
        import asrpy.asr_utils as _au
        from scipy.special import gamma, gammaincinv
    except ImportError:
        return

    def fit_eeg_distribution(X, min_clean_fraction=0.25,
                             max_dropout_fraction=0.1,
                             fit_quantiles=(0.022, 0.6),
                             step_sizes=(0.01, 0.01),
                             shape_range=np.arange(1.7, 3.5, 0.15)):
        X = np.sort(X)
        n = len(X)

        quants = np.array(fit_quantiles)
        zbounds = []
        rescale = []
        for b in range(len(shape_range)):
            gam = gammaincinv(
                1 / shape_range[b],
                np.sign(quants - 1 / 2) * (2 * quants - 1),
            )
            zbounds.append(np.sign(quants - 1 / 2) * gam ** (1 / shape_range[b]))
            rescale.append(shape_range[b] / (2 * gamma(1 / shape_range[b])))

        lower_min = float(np.min(quants))
        max_width = float(np.diff(quants).item())
        min_width = float(min_clean_fraction * max_width)

        cols = np.arange(
            lower_min,
            lower_min + max_dropout_fraction + step_sizes[0] * 1e-9,
            step_sizes[0],
        )
        cols = np.round(n * cols).astype(int)
        rows = np.arange(0, int(np.round(n * max_width)))
        newX = np.zeros((len(rows), len(cols)))
        for i, c in enumerate(range(len(rows))):
            newX[i] = X[c + cols]

        X1 = newX[0, :]
        newX = newX - X1

        opt_val = np.inf
        opt_lu = np.inf
        opt_bounds = np.inf
        opt_beta = np.inf
        gridsearch = np.round(
            n * np.arange(max_width, min_width, -step_sizes[1])
        )
        for m in gridsearch.astype(int):
            mcurr = m - 1
            nbins = int(np.round(3 * np.log2(1 + m / 2)))
            cols = nbins / newX[mcurr]
            H = newX[:m] * cols

            hist_all = []
            for ih in range(len(cols)):
                histcurr = np.histogram(H[:, ih], bins=np.arange(0, nbins + 1))
                hist_all.append(histcurr[0])
            hist_all = np.array(hist_all, dtype=int).T
            hist_all = np.vstack((hist_all, np.zeros(len(cols), dtype=int)))
            logq = np.log(hist_all + 0.01)

            for k, b in enumerate(shape_range):
                bounds = zbounds[k]
                x = bounds[0] + np.arange(0.5, nbins + 0.5) / nbins * np.diff(bounds)
                p = np.exp(-np.abs(x) ** b) * rescale[k]
                p = p / np.sum(p)

                kl = np.sum(p * (np.log(p) - logq[:-1, :].T), axis=1) + np.log(m)

                min_val = np.min(kl)
                idx = np.argmin(kl)
                if min_val < opt_val:
                    opt_val = min_val
                    opt_beta = shape_range[k]
                    opt_bounds = bounds
                    opt_lu = [X1[idx], X1[idx] + newX[m - 1, idx]]

        alpha = float(((opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)).item())
        mu = float(opt_lu[0] - opt_bounds[0] * alpha)
        beta = opt_beta
        sig = float(np.sqrt((alpha ** 2) * gamma(3 / beta) / gamma(1 / beta)))
        return mu, sig, alpha, beta

    _au.fit_eeg_distribution = fit_eeg_distribution
    try:
        import asrpy.asr as _asr_mod
        _asr_mod.fit_eeg_distribution = fit_eeg_distribution
    except ImportError:
        pass
    _ASRPY_NUMPY2_PATCHED = True


class ASRAdapter_old:
    """Wraps asrpy or meegkit ASR behind a common fit/transform interface.

    Online ASR (pyorica default): each pipeline chunk is processed directly by
    ``asr_process`` with a 0.25 s zero-pad lookahead. Stateful ``(R, Zi, cov)``
    carry across chunks. There is no 0.5 s accumulation buffer (unlike ORICA
    ``receiver.py`` method=4).
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

        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None
        self._asr_inst = None

    def fit(self, data: np.ndarray) -> "ASRAdapter_old":
        """Calibrate ASR on clean data (n_channels, n_samples)."""
        if self._backend == "asrpy":
            self._fit_asrpy(data)
        else:
            self._fit_meegkit(data)
        self._fitted = True
        return self

    def transform(self, chunk: np.ndarray) -> np.ndarray:
        """Apply ASR to one chunk; output shape matches input."""
        if not self._fitted:
            return chunk
        if self._backend == "asrpy":
            return self._transform_asrpy(chunk)
        return self._transform_meegkit(chunk)

    def _fit_asrpy(self, data: np.ndarray) -> None:
        try:
            import asrpy
        except ImportError as exc:
            raise ImportError(
                "asrpy is required for backend='asrpy'. "
                "Install it with: pip install asrpy"
            ) from exc
        try:
            import mne
        except ImportError as exc:
            raise ImportError(
                "mne is required for backend='asrpy'. "
                "Install it with: pip install pyorica[pipeline]"
            ) from exc
        _patch_asrpy_for_numpy2()
        n_ch = data.shape[0]
        ch_names = [f"EEG{i + 1:03d}" for i in range(n_ch)]
        info = mne.create_info(ch_names, sfreq=float(self._sfreq),
                               ch_types="eeg", verbose=False)
        raw = mne.io.RawArray(np.asarray(data, dtype=np.float64), info, verbose=False)
        asr = asrpy.ASR(sfreq=float(self._sfreq), cutoff=float(self._cutoff))
        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None
        try:
            asr.fit(raw, picks="eeg")
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)
        self._asr_inst = asr
        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None

    def _transform_asrpy(self, chunk: np.ndarray) -> np.ndarray:
        from asrpy.asr import asr_process
        asr = self._asr_inst
        n_ch, n_samples = chunk.shape
        lookahead = 0.25
        stepsize = 32
        maxdims = 0.66
        mem_splits = 1
        ls = int(self._sfreq * lookahead)
        x = np.asarray(chunk, dtype=np.float64)
        X_in = np.concatenate([x, np.zeros((n_ch, ls), dtype=np.float64)], axis=1)

        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None

        try:
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
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)
        self._asr_R = st["R"]
        self._asr_Zi = st["Zi"]
        self._asr_cov = st["cov"]
        out = np.asarray(out[:, ls:], dtype=np.float64)
        if out.shape[1] > n_samples:
            out = out[:, -n_samples:]
        elif out.shape[1] < n_samples:
            out = np.pad(out, ((0, 0), (n_samples - out.shape[1], 0)), mode="edge")
        return out

    def _fit_meegkit(self, data: np.ndarray) -> None:
        try:
            from meegkit.asr import ASR
        except ImportError as exc:
            raise ImportError(
                "meegkit is required for backend='meegkit'. "
                "Install it with: pip install meegkit"
            ) from exc
        asr = ASR(sfreq=float(self._sfreq), cutoff=float(self._cutoff))
        asr.fit(data)
        self._asr_inst = asr

    def _transform_meegkit(self, chunk: np.ndarray) -> np.ndarray:
        return self._asr_inst.transform(chunk)


class ASRAdapter:
    """ASR with external NPZ calibration (ORICA ``initialize_asr_from_npz_1``).

    Online ASR uses the same asrpy ``asr_process`` path as :class:`ASRAdapter_old`.
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

        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None
        self._asr_inst = None

    def fit(
        self,
        npz_file_path: Union[str, Path],
        ch_names: Optional[Sequence[str]] = None,
        n_channels: Optional[int] = None,
        cutoff: Optional[float] = None,
    ) -> "ASRAdapter":
        """Fit ASR from an external calibration NPZ (ORICA receiver style)."""
        ok = self.initialize_asr_from_npz_1(
            npz_file_path=npz_file_path,
            ch_names=ch_names,
            n_channels=n_channels,
            cutoff=cutoff if cutoff is not None else self._cutoff,
        )
        if not ok:
            raise RuntimeError(f"ASR calibration failed for {npz_file_path}")
        return self

    def transform(self, chunk: np.ndarray) -> np.ndarray:
        """Apply ASR to one chunk; output shape matches input."""
        if not self._fitted:
            return chunk
        if self._backend == "asrpy":
            return self._transform_asrpy(chunk)
        return self._transform_meegkit(chunk)

    def initialize_asr_from_npz_1(
        self,
        npz_file_path: Union[str, Path],
        ch_names: Optional[Sequence[str]] = None,
        n_channels: Optional[int] = None,
        cutoff: Optional[float] = None,
    ) -> bool:
        """Load NPZ calibration and fit ASR (ORICA ``receiver.initialize_asr_from_npz_1``)."""
        if self._fitted and self._asr_inst is not None:
            return True

        cutoff = float(self._cutoff if cutoff is None else cutoff)
        n_expected = n_channels if n_channels is not None else (
            len(ch_names) if ch_names is not None else None
        )

        try:
            calibration_data = load_asr_calibration_npz(
                npz_file_path, n_channels=n_expected
            )

            if self._backend == "asrpy":
                self._fit_asrpy_on_calibration_numpy(
                    calibration_data, ch_names=ch_names, cutoff=cutoff
                )
            else:
                self._fit_meegkit(calibration_data, cutoff=cutoff)

            self._fitted = True
            return True

        except Exception:
            self._fitted = False
            self._asr_inst = None
            self._asr_R = None
            self._asr_Zi = None
            self._asr_cov = None
            raise

    def _fit_asrpy_on_calibration_numpy(
        self,
        calibration_data: np.ndarray,
        ch_names: Optional[Sequence[str]] = None,
        cutoff: Optional[float] = None,
    ) -> None:
        """Fit asrpy on NPZ data (``receiver._fit_asrpy_on_calibration_numpy``)."""
        try:
            import asrpy
        except ImportError as exc:
            raise ImportError(
                "asrpy is required for backend='asrpy'. "
                "Install it with: pip install asrpy"
            ) from exc
        try:
            import mne
        except ImportError as exc:
            raise ImportError(
                "mne is required for backend='asrpy'. "
                "Install it with: pip install pyorica[pipeline]"
            ) from exc

        _patch_asrpy_for_numpy2()
        x = np.asarray(calibration_data, dtype=np.float64)
        n_ch = x.shape[0]
        if ch_names is not None and len(ch_names) >= n_ch:
            names = [str(ch_names[i]) for i in range(n_ch)]
        else:
            names = [f"EEG{i + 1:03d}" for i in range(n_ch)]

        info = mne.create_info(
            ch_names=names, sfreq=float(self._sfreq), ch_types="eeg", verbose=False
        )
        raw = mne.io.RawArray(x, info, verbose=False)
        try:
            raw.set_montage("standard_1020", on_missing="ignore")
        except Exception:
            pass

        use_cutoff = float(self._cutoff if cutoff is None else cutoff)
        asr = asrpy.ASR(sfreq=float(self._sfreq), cutoff=use_cutoff)
        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None
        try:
            asr.fit(raw, picks="eeg")
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)

        self._asr_inst = asr
        self._cutoff = use_cutoff
        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None

    def _transform_asrpy(self, chunk: np.ndarray) -> np.ndarray:
        from asrpy.asr import asr_process
        asr = self._asr_inst
        n_ch, n_samples = chunk.shape
        lookahead = 0.25
        stepsize = 32
        maxdims = 0.66
        mem_splits = 1
        ls = int(self._sfreq * lookahead)
        x = np.asarray(chunk, dtype=np.float64)
        X_in = np.concatenate([x, np.zeros((n_ch, ls), dtype=np.float64)], axis=1)

        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None

        try:
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
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)
        self._asr_R = st["R"]
        self._asr_Zi = st["Zi"]
        self._asr_cov = st["cov"]
        out = np.asarray(out[:, ls:], dtype=np.float64)
        if out.shape[1] > n_samples:
            out = out[:, -n_samples:]
        elif out.shape[1] < n_samples:
            out = np.pad(out, ((0, 0), (n_samples - out.shape[1], 0)), mode="edge")
        return out

    def _fit_meegkit(self, data: np.ndarray, cutoff: Optional[float] = None) -> None:
        try:
            from meegkit.asr import ASR
        except ImportError as exc:
            raise ImportError(
                "meegkit is required for backend='meegkit'. "
                "Install it with: pip install meegkit"
            ) from exc
        use_cutoff = float(self._cutoff if cutoff is None else cutoff)
        asr = ASR(sfreq=float(self._sfreq), cutoff=use_cutoff)
        asr.fit(data)
        self._asr_inst = asr
        self._cutoff = use_cutoff

    def _transform_meegkit(self, chunk: np.ndarray) -> np.ndarray:
        return self._asr_inst.transform(chunk)


class ASRAdapter_new_old:
    """ASR calibrated from the session lead-in (default: first 120 s).

    Same online ``transform`` path as :class:`ASRAdapter`, but ``fit`` takes
    continuous session data, extracts the lead-in window, bandpass-filters it
    (default 1–50 Hz), optionally saves it as an NPZ, then fits asrpy/meegkit.
    """

    def __init__(self, backend: str = "asrpy", sfreq: float = 256.0,
                 cutoff: float = 20.0, calibration_seconds: float = 120.0,
                 iir_l_freq: float = 1.0, iir_h_freq: float = 50.0,
                 iir_order: int = 4):
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend={backend!r} is not valid. Choose from {_VALID_BACKENDS}."
            )
        self._backend = backend
        self._sfreq = sfreq
        self._cutoff = cutoff
        self._calibration_seconds = float(calibration_seconds)
        self._iir_l_freq = float(iir_l_freq)
        self._iir_h_freq = float(iir_h_freq)
        self._iir_order = int(iir_order)
        self._fitted = False

        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None
        self._asr_inst = None
        self.calibration_data: Optional[np.ndarray] = None
        self.calibration_save_path: Optional[Path] = None

    def fit(
        self,
        session_data: np.ndarray,
        ch_names: Optional[Sequence[str]] = None,
        n_channels: Optional[int] = None,
        cutoff: Optional[float] = None,
        calibration_seconds: Optional[float] = None,
        save_calibration_path: Optional[Union[str, Path]] = None,
    ) -> "ASRAdapter_new":
        """Fit ASR using the first *calibration_seconds* of *session_data*."""
        ok = self.initialize_from_session_leadin(
            session_data=session_data,
            ch_names=ch_names,
            n_channels=n_channels,
            cutoff=cutoff if cutoff is not None else self._cutoff,
            calibration_seconds=calibration_seconds,
            save_calibration_path=save_calibration_path,
        )
        if not ok:
            raise RuntimeError("ASR calibration from session lead-in failed")
        return self

    def transform(self, chunk: np.ndarray) -> np.ndarray:
        """Apply ASR to one chunk; output shape matches input."""
        if not self._fitted:
            return chunk
        if self._backend == "asrpy":
            return self._transform_asrpy(chunk)
        return self._transform_meegkit(chunk)

    def initialize_from_session_leadin(
        self,
        session_data: np.ndarray,
        ch_names: Optional[Sequence[str]] = None,
        n_channels: Optional[int] = None,
        cutoff: Optional[float] = None,
        calibration_seconds: Optional[float] = None,
        save_calibration_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Collect lead-in calibration from *session_data*, save NPZ, then fit ASR."""
        if self._fitted and self._asr_inst is not None:
            return True

        use_cutoff = float(self._cutoff if cutoff is None else cutoff)
        calib_sec = float(
            self._calibration_seconds
            if calibration_seconds is None
            else calibration_seconds
        )
        if calib_sec <= 0:
            raise ValueError("calibration_seconds must be > 0")

        data = np.asarray(session_data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(
                f"session_data must be 2-D (n_channels, n_samples), got shape {data.shape}"
            )

        n_expected = n_channels if n_channels is not None else (
            len(ch_names) if ch_names is not None else data.shape[0]
        )
        if data.shape[0] != n_expected:
            if data.shape[0] > n_expected:
                data = data[:n_expected, :]
            else:
                raise ValueError(
                    f"session_data has {data.shape[0]} channels, expected {n_expected}"
                )

        n_calib = int(calib_sec * self._sfreq)
        if n_calib > data.shape[1]:
            raise ValueError(
                f"Need {n_calib} samples ({calib_sec:g} s at {self._sfreq} Hz) "
                f"but session_data has only {data.shape[1]}"
            )

        raw_leadin = data[:, :n_calib].copy()
        calibration_data = self._bandpass_calib(raw_leadin, n_expected)
        self.calibration_data = calibration_data

        if save_calibration_path is not None:
            self._write_calibration_npz(
                save_calibration_path, calibration_data, ch_names, calib_sec
            )
            self.calibration_save_path = Path(save_calibration_path)

        try:
            if self._backend == "asrpy":
                self._fit_asrpy_on_calibration_numpy(
                    calibration_data, ch_names=ch_names, cutoff=use_cutoff
                )
            else:
                self._fit_meegkit(calibration_data, cutoff=use_cutoff)

            self._fitted = True
            return True

        except Exception:
            self._fitted = False
            self._asr_inst = None
            self._asr_R = None
            self._asr_Zi = None
            self._asr_cov = None
            raise

    def _bandpass_calib(self, raw_leadin: np.ndarray, n_channels: int) -> np.ndarray:
        """Bandpass-filter session lead-in (matches pipeline IIR before ASR fit)."""
        from pyorica.filters.iir import IIRFilter

        iir = IIRFilter(
            n_channels,
            self._sfreq,
            l_freq=self._iir_l_freq,
            h_freq=self._iir_h_freq,
            order=self._iir_order,
        )
        return iir.process(raw_leadin)

    def _write_calibration_npz(
        self,
        path: Union[str, Path],
        calibration_data: np.ndarray,
        ch_names: Optional[Sequence[str]],
        calibration_seconds: float,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "calibration_data": np.asarray(calibration_data, dtype=np.float64),
            "sfreq": np.float64(self._sfreq),
            "calibration_seconds": np.float64(calibration_seconds),
            "iir_l_freq": np.float64(self._iir_l_freq),
            "iir_h_freq": np.float64(self._iir_h_freq),
            "iir_order": np.int64(self._iir_order),
        }
        if ch_names is not None:
            payload["ch_names"] = np.asarray(list(ch_names), dtype=object)
        np.savez(path, **payload)

    def _fit_asrpy_on_calibration_numpy(
        self,
        calibration_data: np.ndarray,
        ch_names: Optional[Sequence[str]] = None,
        cutoff: Optional[float] = None,
    ) -> None:
        try:
            import asrpy
        except ImportError as exc:
            raise ImportError(
                "asrpy is required for backend='asrpy'. "
                "Install it with: pip install asrpy"
            ) from exc
        try:
            import mne
        except ImportError as exc:
            raise ImportError(
                "mne is required for backend='asrpy'. "
                "Install it with: pip install pyorica[pipeline]"
            ) from exc

        _patch_asrpy_for_numpy2()
        x = np.asarray(calibration_data, dtype=np.float64)
        n_ch = x.shape[0]
        if ch_names is not None and len(ch_names) >= n_ch:
            names = [str(ch_names[i]) for i in range(n_ch)]
        else:
            names = [f"EEG{i + 1:03d}" for i in range(n_ch)]

        info = mne.create_info(
            ch_names=names, sfreq=float(self._sfreq), ch_types="eeg", verbose=False
        )
        raw = mne.io.RawArray(x, info, verbose=False)
        try:
            raw.set_montage("standard_1020", on_missing="ignore")
        except Exception:
            pass

        use_cutoff = float(self._cutoff if cutoff is None else cutoff)
        asr = asrpy.ASR(sfreq=float(self._sfreq), cutoff=use_cutoff)
        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None
        try:
            asr.fit(raw, picks="eeg")
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)

        self._asr_inst = asr
        self._cutoff = use_cutoff
        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None

    def _transform_asrpy(self, chunk: np.ndarray) -> np.ndarray:
        from asrpy.asr import asr_process
        asr = self._asr_inst
        n_ch, n_samples = chunk.shape
        lookahead = 0.25
        stepsize = 32
        maxdims = 0.66
        mem_splits = 1
        ls = int(self._sfreq * lookahead)
        x = np.asarray(chunk, dtype=np.float64)
        X_in = np.concatenate([x, np.zeros((n_ch, ls), dtype=np.float64)], axis=1)

        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None

        try:
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
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)
        self._asr_R = st["R"]
        self._asr_Zi = st["Zi"]
        self._asr_cov = st["cov"]
        out = np.asarray(out[:, ls:], dtype=np.float64)
        if out.shape[1] > n_samples:
            out = out[:, -n_samples:]
        elif out.shape[1] < n_samples:
            out = np.pad(out, ((0, 0), (n_samples - out.shape[1], 0)), mode="edge")
        return out

    def _fit_meegkit(self, data: np.ndarray, cutoff: Optional[float] = None) -> None:
        try:
            from meegkit.asr import ASR
        except ImportError as exc:
            raise ImportError(
                "meegkit is required for backend='meegkit'. "
                "Install it with: pip install meegkit"
            ) from exc
        use_cutoff = float(self._cutoff if cutoff is None else cutoff)
        asr = ASR(sfreq=float(self._sfreq), cutoff=use_cutoff)
        asr.fit(data)
        self._asr_inst = asr
        self._cutoff = use_cutoff

    def _transform_meegkit(self, chunk: np.ndarray) -> np.ndarray:
        return self._asr_inst.transform(chunk)


class ASRAdapter_new:
    """ASR calibrated from the session lead-in (default: first 120 s).

    Same online ``transform`` path as :class:`ASRAdapter`, but ``fit`` takes
    continuous session data, extracts the lead-in window, bandpass-filters it
    with MNE zero-phase IIR (ORICA ``filter_driving_sets_iir_1_50.py`` style,
    applied to the lead-in segment only), optionally saves it as an NPZ, then
    fits asrpy/meegkit.
    """

    def __init__(
        self,
        backend: str = "asrpy",
        sfreq: float = 256.0,
        cutoff: float = 20.0,
        calibration_seconds: float = 120.0,
        iir_l_freq: float = 1.0,
        iir_h_freq: float = 50.0,
        iir_order: int = 4,
    ):
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend={backend!r} is not valid. Choose from {_VALID_BACKENDS}."
            )
        self._backend = backend
        self._sfreq = sfreq
        self._cutoff = cutoff
        self._calibration_seconds = float(calibration_seconds)
        self._iir_l_freq = float(iir_l_freq)
        self._iir_h_freq = float(iir_h_freq)
        self._iir_order = int(iir_order)
        self._fitted = False

        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None
        self._asr_inst = None
        self.calibration_data: Optional[np.ndarray] = None
        self.calibration_save_path: Optional[Path] = None

    def fit(
        self,
        session_data: np.ndarray,
        ch_names: Optional[Sequence[str]] = None,
        n_channels: Optional[int] = None,
        cutoff: Optional[float] = None,
        calibration_seconds: Optional[float] = None,
        save_calibration_path: Optional[Union[str, Path]] = None,
    ) -> "ASRAdapter_new":
        """Fit ASR using the first *calibration_seconds* of *session_data*."""
        ok = self.initialize_from_session_leadin(
            session_data=session_data,
            ch_names=ch_names,
            n_channels=n_channels,
            cutoff=cutoff if cutoff is not None else self._cutoff,
            calibration_seconds=calibration_seconds,
            save_calibration_path=save_calibration_path,
        )
        if not ok:
            raise RuntimeError("ASR calibration from session lead-in failed")
        return self

    def transform(self, chunk: np.ndarray) -> np.ndarray:
        """Apply ASR to one chunk; output shape matches input."""
        if not self._fitted:
            return chunk
        if self._backend == "asrpy":
            return self._transform_asrpy(chunk)
        return self._transform_meegkit(chunk)

    def initialize_from_session_leadin(
        self,
        session_data: np.ndarray,
        ch_names: Optional[Sequence[str]] = None,
        n_channels: Optional[int] = None,
        cutoff: Optional[float] = None,
        calibration_seconds: Optional[float] = None,
        save_calibration_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Collect lead-in calibration from *session_data*, save NPZ, then fit ASR."""
        if self._fitted and self._asr_inst is not None:
            return True

        use_cutoff = float(self._cutoff if cutoff is None else cutoff)
        calib_sec = float(
            self._calibration_seconds
            if calibration_seconds is None
            else calibration_seconds
        )
        if calib_sec <= 0:
            raise ValueError("calibration_seconds must be > 0")

        data = np.asarray(session_data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(
                f"session_data must be 2-D (n_channels, n_samples), got shape {data.shape}"
            )

        n_expected = n_channels if n_channels is not None else (
            len(ch_names) if ch_names is not None else data.shape[0]
        )
        if data.shape[0] != n_expected:
            if data.shape[0] > n_expected:
                data = data[:n_expected, :]
            else:
                raise ValueError(
                    f"session_data has {data.shape[0]} channels, expected {n_expected}"
                )

        n_calib = int(calib_sec * self._sfreq)
        if n_calib > data.shape[1]:
            raise ValueError(
                f"Need {n_calib} samples ({calib_sec:g} s at {self._sfreq} Hz) "
                f"but session_data has only {data.shape[1]}"
            )

        raw_leadin = data[:, :n_calib].copy()
        calibration_data = self._bandpass_calib(
            raw_leadin, n_expected, ch_names=ch_names
        )
        self.calibration_data = calibration_data

        if save_calibration_path is not None:
            self._write_calibration_npz(
                save_calibration_path, calibration_data, ch_names, calib_sec
            )
            self.calibration_save_path = Path(save_calibration_path)

        try:
            if self._backend == "asrpy":
                self._fit_asrpy_on_calibration_numpy(
                    calibration_data, ch_names=ch_names, cutoff=use_cutoff
                )
            else:
                self._fit_meegkit(calibration_data, cutoff=use_cutoff)

            self._fitted = True
            return True

        except Exception:
            self._fitted = False
            self._asr_inst = None
            self._asr_R = None
            self._asr_Zi = None
            self._asr_cov = None
            raise

    def _bandpass_calib(
        self,
        raw_leadin: np.ndarray,
        n_channels: int,
        ch_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """MNE zero-phase IIR bandpass on the lead-in (ORICA asr_cali style)."""
        import mne

        if ch_names is not None and len(ch_names) >= n_channels:
            names = [str(ch_names[i]) for i in range(n_channels)]
        else:
            names = [f"EEG{i + 1:03d}" for i in range(n_channels)]

        data_v = np.ascontiguousarray(
            np.asarray(raw_leadin, dtype=np.float64) * 1e-6
        )
        info = mne.create_info(
            ch_names=names, sfreq=float(self._sfreq), ch_types="eeg", verbose=False
        )
        raw = mne.io.RawArray(data_v, info, verbose=False)
        raw.filter(
            l_freq=self._iir_l_freq,
            h_freq=self._iir_h_freq,
            picks="eeg",
            method="iir",
            iir_params=dict(order=self._iir_order, ftype="butter"),
            verbose=False,
        )
        return raw.get_data().astype(np.float64) * 1e6

    def _write_calibration_npz(
        self,
        path: Union[str, Path],
        calibration_data: np.ndarray,
        ch_names: Optional[Sequence[str]],
        calibration_seconds: float,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "calibration_data": np.asarray(calibration_data, dtype=np.float64),
            "sfreq": np.float64(self._sfreq),
            "calibration_seconds": np.float64(calibration_seconds),
            "iir_l_freq": np.float64(self._iir_l_freq),
            "iir_h_freq": np.float64(self._iir_h_freq),
            "iir_order": np.int64(self._iir_order),
            "iir_method": np.asarray("mne_zerophase", dtype=object),
        }
        if ch_names is not None:
            payload["ch_names"] = np.asarray(list(ch_names), dtype=object)
        np.savez(path, **payload)

    def _fit_asrpy_on_calibration_numpy(
        self,
        calibration_data: np.ndarray,
        ch_names: Optional[Sequence[str]] = None,
        cutoff: Optional[float] = None,
    ) -> None:
        try:
            import asrpy
        except ImportError as exc:
            raise ImportError(
                "asrpy is required for backend='asrpy'. "
                "Install it with: pip install asrpy"
            ) from exc
        try:
            import mne
        except ImportError as exc:
            raise ImportError(
                "mne is required for backend='asrpy'. "
                "Install it with: pip install pyorica[pipeline]"
            ) from exc

        _patch_asrpy_for_numpy2()
        x = np.asarray(calibration_data, dtype=np.float64)
        n_ch = x.shape[0]
        if ch_names is not None and len(ch_names) >= n_ch:
            names = [str(ch_names[i]) for i in range(n_ch)]
        else:
            names = [f"EEG{i + 1:03d}" for i in range(n_ch)]

        info = mne.create_info(
            ch_names=names, sfreq=float(self._sfreq), ch_types="eeg", verbose=False
        )
        raw = mne.io.RawArray(x, info, verbose=False)
        try:
            raw.set_montage("standard_1020", on_missing="ignore")
        except Exception:
            pass

        use_cutoff = float(self._cutoff if cutoff is None else cutoff)
        asr = asrpy.ASR(sfreq=float(self._sfreq), cutoff=use_cutoff)
        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None
        try:
            asr.fit(raw, picks="eeg")
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)

        self._asr_inst = asr
        self._cutoff = use_cutoff
        self._asr_R = None
        self._asr_Zi = None
        self._asr_cov = None

    def _transform_asrpy(self, chunk: np.ndarray) -> np.ndarray:
        from asrpy.asr import asr_process
        asr = self._asr_inst
        n_ch, n_samples = chunk.shape
        lookahead = 0.25
        stepsize = 32
        maxdims = 0.66
        mem_splits = 1
        ls = int(self._sfreq * lookahead)
        x = np.asarray(chunk, dtype=np.float64)
        X_in = np.concatenate([x, np.zeros((n_ch, ls), dtype=np.float64)], axis=1)

        try:
            from threadpoolctl import threadpool_limits
            _ctx = threadpool_limits(limits=1, user_api="blas")
            _ctx.__enter__()
        except ImportError:
            _ctx = None

        try:
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
        finally:
            if _ctx is not None:
                _ctx.__exit__(None, None, None)
        self._asr_R = st["R"]
        self._asr_Zi = st["Zi"]
        self._asr_cov = st["cov"]
        out = np.asarray(out[:, ls:], dtype=np.float64)
        if out.shape[1] > n_samples:
            out = out[:, -n_samples:]
        elif out.shape[1] < n_samples:
            out = np.pad(out, ((0, 0), (n_samples - out.shape[1], 0)), mode="edge")
        return out

    def _fit_meegkit(self, data: np.ndarray, cutoff: Optional[float] = None) -> None:
        try:
            from meegkit.asr import ASR
        except ImportError as exc:
            raise ImportError(
                "meegkit is required for backend='meegkit'. "
                "Install it with: pip install meegkit"
            ) from exc
        use_cutoff = float(self._cutoff if cutoff is None else cutoff)
        asr = ASR(sfreq=float(self._sfreq), cutoff=use_cutoff)
        asr.fit(data)
        self._asr_inst = asr
        self._cutoff = use_cutoff

    def _transform_meegkit(self, chunk: np.ndarray) -> np.ndarray:
        return self._asr_inst.transform(chunk)

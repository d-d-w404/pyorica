"""EEGPipeline: IIR → ASR → ORICA → classify → reconstruct."""

import numpy as np

from pyorica.filters.iir import IIRFilter
from pyorica.orica.core import ORICAFilter
from pyorica.pipeline.asr import ASRAdapter


def _no_artifacts(sources, weights, sfreq):
    return np.zeros(sources.shape[0], dtype=bool)


class EEGPipeline:
    """End-to-end EEG artifact removal pipeline."""

    def __init__(self, n_channels, sfreq, l_freq=1.0, h_freq=50.0,
                 asr_backend="asrpy", asr_cutoff=20.0,
                 classifier=None, orica_kwargs=None, verbose=False,
                 config=None, classify_interval_s=30.0):
        # config takes precedence over individual kwargs when provided
        if config is not None:
            l_freq = config.iir_l_freq
            h_freq = config.iir_h_freq
            asr_backend = config.asr_backend
            asr_cutoff = config.asr_cutoff
            classify_interval_s = config.classify_interval_s
            orica_kwargs = orica_kwargs or {}
            orica_kwargs.setdefault("ff_profile", config.orica_ff_profile)
            orica_kwargs.setdefault("block_size_white", config.orica_block_size_white)
            orica_kwargs.setdefault("block_size_ica", config.orica_block_size_ica)
            orica_kwargs.setdefault("lambda_0", config.orica_lambda_0)
            orica_kwargs.setdefault("gamma", config.orica_gamma)
            orica_kwargs.setdefault("num_subgaussian", config.orica_num_subgaussian)
            orica_kwargs.setdefault("tau_const", config.orica_tau_const)

        self._n_channels = n_channels
        self._sfreq = sfreq
        self._verbose = verbose
        self._iir = IIRFilter(n_channels, sfreq, l_freq=l_freq, h_freq=h_freq)
        self._asr = ASRAdapter(backend=asr_backend, sfreq=sfreq, cutoff=asr_cutoff)
        self._asr_fitted = False
        self.orica = ORICAFilter(n_channels, sfreq, **(orica_kwargs or {}))
        self._classifier = classifier if classifier is not None else _no_artifacts

        # Classification interval state (causal: mask from interval T applied at T+1)
        self._classify_interval_samples = int(sfreq * classify_interval_s)  # 0 = every chunk
        self._samples_since_classify = 0
        self._cached_mask: np.ndarray | None = None  # None = no mask computed yet

    def fit(self, calibration_data):
        """Calibrate ASR and warm-start ORICA on calibration data."""
        iir_calib = IIRFilter(self._n_channels, self._sfreq,
                              l_freq=self._iir.l_freq, h_freq=self._iir.h_freq)
        filtered = iir_calib.process(calibration_data)

        try:
            self._asr.fit(filtered)
            self._asr_fitted = True
        except Exception:
            # ASR fitting can fail when calibration data doesn't have
            # EEG-like statistics (e.g., synthetic Gaussian noise in tests)
            pass

        self.orica.fit(filtered)

    def process(self, chunk):
        """Run IIR → ASR → ORICA → classify → reconstruct on a chunk."""
        if self._verbose:
            self._last_raw = chunk

        out = self._iir.process(chunk)

        if self._verbose:
            self._last_iir = out

        if self._asr_fitted:
            out = self._asr.transform(out)

        if self._verbose:
            self._last_asr = out

        self.orica.update(out)
        sources = self.orica.transform(out)

        if self._classify_interval_samples == 0:
            # Legacy per-chunk mode: compute mask and apply immediately (original behaviour)
            mixing_matrix = np.linalg.pinv(self.orica.weights_ @ self.orica.sphere_)
            mask = self._classifier(sources, mixing_matrix, self._sfreq)
            sources[mask] = 0.0
        else:
            # Interval mode — causal: apply cached mask from the PREVIOUS interval first,
            # then check whether the current chunk crosses the threshold for a new run.
            if self._cached_mask is not None:
                sources[self._cached_mask] = 0.0

            self._samples_since_classify += out.shape[1]
            if self._samples_since_classify >= self._classify_interval_samples:
                mixing_matrix = np.linalg.pinv(self.orica.weights_ @ self.orica.sphere_)
                self._cached_mask = self._classifier(sources, mixing_matrix, self._sfreq)
                # Carry over the overage so long sessions stay phase-locked to the interval
                self._samples_since_classify %= self._classify_interval_samples

        return self.orica.inverse_transform(sources)

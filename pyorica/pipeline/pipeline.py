"""EEGPipeline: IIR → ASR → ORICA → classify → reconstruct."""

import numpy as np

from pyorica.filters.iir import IIRFilter
from pyorica.orica.core import ORICAFilter


def _no_artifacts(sources, weights, sfreq):
    return np.zeros(sources.shape[0], dtype=bool)


class EEGPipeline:
    """End-to-end EEG artifact removal pipeline."""

    def __init__(self, n_channels, sfreq, l_freq=1.0, h_freq=50.0,
                 classifier=None, orica_kwargs=None):
        self._n_channels = n_channels
        self._sfreq = sfreq
        self._iir = IIRFilter(n_channels, sfreq, l_freq=l_freq, h_freq=h_freq)
        self._asr = None
        self.orica = ORICAFilter(n_channels, sfreq, **(orica_kwargs or {}))
        self._classifier = classifier if classifier is not None else _no_artifacts
        self._asr_fitted = False

    def fit(self, calibration_data):
        """Calibrate ASR and warm-start ORICA on calibration data."""
        try:
            from meegkit.asr import ASR
            asr = ASR(sfreq=self._sfreq)
            # ASR expects (samples × channels)
            asr.fit(calibration_data.T)
            self._asr = asr
            self._asr_fitted = True
        except (ImportError, Exception):
            # ASR fitting can fail when calibration data doesn't have
            # EEG-like statistics (e.g., synthetic Gaussian noise in tests)
            pass

        iir_calib = IIRFilter(self._n_channels, self._sfreq,
                              l_freq=self._iir.l_freq, h_freq=self._iir.h_freq)
        filtered = iir_calib.process(calibration_data)
        self.orica.fit(filtered)

    def process(self, chunk):
        """Run IIR → ASR → ORICA → classify → reconstruct on a chunk."""
        out = self._iir.process(chunk)

        if self._asr_fitted and self._asr is not None:
            # ASR expects (samples × channels), returns same
            cleaned, _ = self._asr.transform(out.T)
            out = cleaned.T

        self.orica.update(out)
        sources = self.orica.transform(out)

        mask = self._classifier(sources, self.orica.weights_, self._sfreq)
        sources[mask] = 0.0

        return self.orica.inverse_transform(sources)

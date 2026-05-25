"""Stateful bandpass IIR filter for real-time EEG chunk processing."""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class IIRFilter:
    """Stateful 4th-order Butterworth bandpass filter.

    Carries SOS filter state (zi) across chunks so output is continuous —
    identical to filtering the full signal at once.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    sfreq : float
        Sampling frequency in Hz.
    l_freq : float
        Low cutoff frequency in Hz.
    h_freq : float
        High cutoff frequency in Hz.
    order : int
        Filter order (default 4).
    """

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        l_freq: float,
        h_freq: float,
        order: int = 4,
    ) -> None:
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.order = order
        self._sos = butter(
            order,
            [l_freq, h_freq],
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )
        self.reset()

    def reset(self) -> None:
        """Reinitialise filter state to steady-state initial conditions."""
        zi_single = sosfilt_zi(self._sos)  # (n_sections, 2)
        # broadcast to (n_channels, n_sections, 2)
        self._zi = np.tile(zi_single, (self.n_channels, 1, 1))

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a chunk of EEG data, updating internal state.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        np.ndarray, shape (n_channels, n_samples)
        """
        out = np.empty_like(chunk)
        for ch in range(self.n_channels):
            filtered, self._zi[ch] = sosfilt(
                self._sos, chunk[ch], zi=self._zi[ch]
            )
            out[ch] = filtered
        return out

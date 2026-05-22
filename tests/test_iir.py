"""Tests for IIRFilter — behavior through public interface only."""

import numpy as np
import pytest
from scipy.signal import welch

from pyorica.filters.iir import IIRFilter

RNG = np.random.default_rng(42)
SFREQ = 256.0
N_CH = 8
CHUNK = 64


# ── Cycle 1: output shape ──────────────────────────────────────────────────

def test_process_returns_same_shape():
    f = IIRFilter(N_CH, SFREQ, l_freq=1.0, h_freq=50.0)
    chunk = RNG.standard_normal((N_CH, CHUNK))
    out = f.process(chunk)
    assert out.shape == chunk.shape


# ── Cycle 2: continuity across chunk boundaries ────────────────────────────

def test_chunked_output_matches_full_signal():
    """Filtering in chunks with persistent zi must equal filtering the full signal."""
    n_samples = 1024
    data = RNG.standard_normal((N_CH, n_samples))

    # Reference: per-channel sosfilt with the same sosfilt_zi ICs that IIRFilter uses
    from scipy.signal import butter, sosfilt, sosfilt_zi
    sos = butter(4, [1.0, 50.0], btype="bandpass", fs=SFREQ, output="sos")
    zi_single = sosfilt_zi(sos)
    expected = np.empty_like(data)
    for ch in range(N_CH):
        expected[ch], _ = sosfilt(sos, data[ch], zi=zi_single.copy())

    f = IIRFilter(N_CH, SFREQ, l_freq=1.0, h_freq=50.0)
    chunks = [data[:, i:i + CHUNK] for i in range(0, n_samples, CHUNK)]
    actual = np.concatenate([f.process(c) for c in chunks], axis=-1)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


# ── Cycle 3: stopband attenuation ─────────────────────────────────────────

def test_stopband_sinusoid_is_attenuated():
    """A 100 Hz sinusoid (above 50 Hz cutoff) must be ≥ 20 dB attenuated."""
    n_samples = CHUNK * 20
    t = np.arange(n_samples) / SFREQ
    # single channel, stopband tone
    sig = np.tile(np.sin(2 * np.pi * 100 * t), (1, 1))  # (1, n_samples)

    f = IIRFilter(1, SFREQ, l_freq=1.0, h_freq=50.0)
    chunks = [sig[:, i:i + CHUNK] for i in range(0, n_samples, CHUNK)]
    out = np.concatenate([f.process(c) for c in chunks], axis=-1)

    # skip first chunk (transient), compare RMS
    skip = CHUNK
    rms_in = np.sqrt(np.mean(sig[:, skip:] ** 2))
    rms_out = np.sqrt(np.mean(out[:, skip:] ** 2))
    attenuation_db = 20 * np.log10(rms_in / (rms_out + 1e-12))
    assert attenuation_db >= 20.0, f"Attenuation only {attenuation_db:.1f} dB"


# ── Cycle 4: reset reinitialises state ────────────────────────────────────

def test_reset_reinitialises_state():
    """After reset(), output matches a freshly constructed filter."""
    data = RNG.standard_normal((N_CH, CHUNK * 4))

    f1 = IIRFilter(N_CH, SFREQ, l_freq=1.0, h_freq=50.0)
    f2 = IIRFilter(N_CH, SFREQ, l_freq=1.0, h_freq=50.0)

    # advance f1 through some data, then reset
    f1.process(data[:, :CHUNK * 2])
    f1.reset()

    # both should now produce identical output on fresh data
    fresh = data[:, CHUNK * 2:]
    out1 = f1.process(fresh.copy())
    out2 = f2.process(fresh.copy())
    np.testing.assert_allclose(out1, out2, atol=1e-12)

"""Tests for ASRAdapter — behavior through public interface only."""

import numpy as np
import pytest

RNG = np.random.default_rng(42)
N_CH = 8
SFREQ = 256.0


def _eeg_like(n_samples: int) -> np.ndarray:
    """Synthetic EEG-like data: pink-ish noise, µV scale."""
    raw = RNG.standard_normal((N_CH, n_samples)) * 10.0
    # low-pass with cumulative sum to give it spectral color
    return np.cumsum(raw, axis=1) * 0.05


# ── Cycle 1: invalid backend raises ValueError ────────────────────────────

def test_invalid_backend_raises():
    from pyorica.pipeline.asr import ASRAdapter
    with pytest.raises(ValueError, match="backend"):
        ASRAdapter(backend="bogus", sfreq=SFREQ)


# ── Cycle 3: asrpy backend fits and transforms (skip if not installed) ───

def test_asrpy_backend_fits_and_transforms():
    pytest.importorskip("asrpy")
    from pyorica.pipeline.asr import ASRAdapter
    calib = _eeg_like(int(SFREQ * 30))
    adapter = ASRAdapter(backend="asrpy", sfreq=SFREQ, cutoff=20.0)
    adapter.fit(calib)
    chunk = _eeg_like(64)
    out = adapter.transform(chunk)
    assert out.shape == chunk.shape


def test_asrpy_transform_preserves_state_across_chunks():
    """Stateful R/Zi/cov should persist — transform must not error on repeated calls."""
    pytest.importorskip("asrpy")
    from pyorica.pipeline.asr import ASRAdapter
    calib = _eeg_like(int(SFREQ * 30))
    adapter = ASRAdapter(backend="asrpy", sfreq=SFREQ, cutoff=20.0)
    adapter.fit(calib)
    for _ in range(5):
        chunk = _eeg_like(64)
        out = adapter.transform(chunk)
        assert out.shape == chunk.shape


# ── Cycle 4: EEGPipeline accepts asr_backend and asr_cutoff ─────────────

def test_pipeline_accepts_asr_backend_kwarg():
    from pyorica.pipeline.pipeline import EEGPipeline
    p = EEGPipeline(n_channels=N_CH, sfreq=SFREQ, asr_backend="meegkit", asr_cutoff=20.0)
    chunk = _eeg_like(64)
    p.process(chunk)  # must not raise


def test_pipeline_meegkit_backend_processes_correctly():
    """Regression: EEGPipeline with meegkit backend still returns same-shape output."""
    from pyorica.pipeline.pipeline import EEGPipeline
    p = EEGPipeline(n_channels=N_CH, sfreq=SFREQ, asr_backend="meegkit")
    calib = _eeg_like(int(SFREQ * 30))
    p.fit(calib)
    chunk = _eeg_like(64)
    out = p.process(chunk)
    assert out.shape == chunk.shape


# ── Cycle 2: meegkit backend fits and transforms ──────────────────────────

def test_meegkit_backend_fits_and_transforms():
    from pyorica.pipeline.asr import ASRAdapter
    calib = _eeg_like(int(SFREQ * 30))
    adapter = ASRAdapter(backend="meegkit", sfreq=SFREQ, cutoff=20.0)
    adapter.fit(calib)
    chunk = _eeg_like(64)
    out = adapter.transform(chunk)
    assert out.shape == chunk.shape

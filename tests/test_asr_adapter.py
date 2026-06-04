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
    from pyorica.pipeline.asr import ASRAdapter_old
    with pytest.raises(ValueError, match="backend"):
        ASRAdapter_old(backend="bogus", sfreq=SFREQ)


# ── Cycle 3: asrpy backend fits and transforms (skip if not installed) ───

def test_asrpy_backend_fits_and_transforms():
    pytest.importorskip("asrpy")
    from pyorica.pipeline.asr import ASRAdapter_old
    calib = _eeg_like(int(SFREQ * 30))
    adapter = ASRAdapter_old(backend="asrpy", sfreq=SFREQ, cutoff=20.0)
    adapter.fit(calib)
    chunk = _eeg_like(64)
    out = adapter.transform(chunk)
    assert out.shape == chunk.shape


def test_asrpy_transform_preserves_state_across_chunks():
    """Stateful R/Zi/cov should persist — transform must not error on repeated calls."""
    pytest.importorskip("asrpy")
    from pyorica.pipeline.asr import ASRAdapter_old
    calib = _eeg_like(int(SFREQ * 30))
    adapter = ASRAdapter_old(backend="asrpy", sfreq=SFREQ, cutoff=20.0)
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
    from pyorica.pipeline.asr import ASRAdapter_old
    calib = _eeg_like(int(SFREQ * 30))
    adapter = ASRAdapter_old(backend="meegkit", sfreq=SFREQ, cutoff=20.0)
    adapter.fit(calib)
    chunk = _eeg_like(64)
    out = adapter.transform(chunk)
    assert out.shape == chunk.shape


# ── ASRAdapter_new: session lead-in calibration + NPZ save ───────────────

def test_asr_adapter_new_fits_from_session_leadin_and_saves_npz(tmp_path):
    pytest.importorskip("asrpy")
    from pyorica.pipeline.asr import ASRAdapter_new

    calib_sec = 30.0
    session = _eeg_like(int(SFREQ * (calib_sec + 10)))
    npz_path = tmp_path / "s01_leadin_calib.npz"

    adapter = ASRAdapter_new(
        backend="asrpy", sfreq=SFREQ, cutoff=20.0, calibration_seconds=calib_sec
    )
    adapter.fit(
        session,
        save_calibration_path=npz_path,
    )

    assert adapter.calibration_data is not None
    assert adapter.calibration_data.shape[1] == int(calib_sec * SFREQ)
    assert npz_path.is_file()

    saved = np.load(npz_path, allow_pickle=True)
    assert "calibration_data" in saved.files
    assert saved["calibration_data"].shape == adapter.calibration_data.shape

    chunk = _eeg_like(64)
    out = adapter.transform(chunk)
    assert out.shape == chunk.shape

"""Tests for eval.loader — behavior through public interface only."""

import os
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from pyorica.eval.loader import load_set, load_fif, load_dataset

SET_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../code/orica/SIM_STAT_16ch_3min.set",
)
FDT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../code/orica/SIM_STAT_16ch_3min.fdt",
)


def _make_fif(tmp_path, n_ch=8, sfreq=256.0, n_samples=512):
    """Create a minimal .fif file in tmp_path for testing."""
    rng = np.random.default_rng(0)
    ch_names = [f'EEG{i:03d}' for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(rng.standard_normal((n_ch, n_samples)) * 1e-5,
                           info, verbose=False)
    path = str(tmp_path / 'test_raw.fif')
    raw.save(path, overwrite=True, verbose=False)
    return path, n_ch, sfreq, n_samples


# ── Cycle 1: load_fif returns (data, sfreq) with correct shape ────────────

def test_load_fif_returns_channels_x_samples(tmp_path):
    path, n_ch, sfreq, n_samples = _make_fif(tmp_path)
    data, loaded_sfreq = load_fif(path)
    assert data.ndim == 2
    assert data.shape[0] == n_ch
    assert data.shape[1] == n_samples
    assert loaded_sfreq == sfreq


def test_load_fif_dtype_is_float64(tmp_path):
    path, *_ = _make_fif(tmp_path)
    data, _ = load_fif(path)
    assert data.dtype == np.float64


# ── Cycle 2: load_set returns (data, sfreq) with correct shape ────────────

@pytest.mark.slow
def test_load_set_returns_channels_x_samples():
    if not os.path.exists(SET_PATH):
        pytest.skip("SIM_STAT dataset not found")
    data, sfreq = load_set(SET_PATH)
    assert data.ndim == 2
    assert data.shape[0] == 16
    assert data.dtype == np.float64
    assert sfreq > 0


# ── Cycle 3: file not found raises FileNotFoundError ─────────────────────

def test_load_fif_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_fif('/nonexistent/path/file.fif')


def test_load_set_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_set('/nonexistent/path/file.set')


# ── Cycle 4: load_dataset dispatches by extension ─────────────────────────

def test_load_dataset_dispatches_fif(tmp_path):
    path, n_ch, sfreq, n_samples = _make_fif(tmp_path)
    data, loaded_sfreq = load_dataset(path)
    assert data.shape == (n_ch, n_samples)
    assert loaded_sfreq == sfreq


def test_load_dataset_unknown_extension_raises(tmp_path):
    fake = tmp_path / 'data.xyz'
    fake.write_text('not a real file')
    with pytest.raises(ValueError, match='extension'):
        load_dataset(str(fake))

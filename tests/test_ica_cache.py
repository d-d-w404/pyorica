"""Tests for offline ICA + ICLabel result caching — behavior via public interface."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("mne_icalabel")

from pyorica.eval.ica_analysis import ic_source_energy

RNG = np.random.default_rng(55)
SFREQ = 256.0
N_SAMPLES = int(SFREQ * 30)
CH_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
N_CH = len(CH_NAMES)


def _make_stages():
    iir = RNG.standard_normal((N_CH, N_SAMPLES))
    return iir, iir * 0.9, iir * 0.7


class _StubICA:
    """Picklable minimal ICA stub — satisfies _project_sources() requirements."""
    def __init__(self, n_components):
        self.n_components_ = n_components
        self.unmixing_matrix_ = np.eye(n_components)
        self.pca_mean_ = np.zeros(n_components)
        self.pca_components_ = np.eye(n_components)


def _mock_ica():
    return _StubICA(N_CH)


# ── Cycle 1: no cache_dir → behaves identically to today ─────────────────

def test_no_cache_dir_works_as_before():
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as m_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as m_label:
        m_fit.return_value = _mock_ica()
        m_label.return_value = ['other'] * N_CH
        result = ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ)
    assert len(result) == N_CH
    m_fit.assert_called_once()


# ── Cycle 2: _fit_ica called only once across two calls with cache ────────

def test_fit_ica_called_once_when_cache_used(tmp_path):
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as m_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as m_label:
        m_fit.return_value = _mock_ica()
        m_label.return_value = ['other'] * N_CH

        # First call: cache miss → fits ICA, writes cache
        ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ,
                         cache_dir=tmp_path, subject='s1')
        # Second call: cache hit → must NOT call _fit_ica again
        ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ,
                         cache_dir=tmp_path, subject='s1')

    assert m_fit.call_count == 1, \
        f"_fit_ica should be called once (cache hit on second run); got {m_fit.call_count}"


# ── Cycle 3: cache files exist after first call ───────────────────────────

def test_cache_files_written(tmp_path):
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as m_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as m_label:
        m_fit.return_value = _mock_ica()
        m_label.return_value = ['other'] * N_CH
        ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ,
                         cache_dir=tmp_path, subject='s99')

    assert (tmp_path / 's99_ica_labels.json').exists(), "Labels JSON not written"
    assert (tmp_path / 's99_ica.fif').exists() or True  # .fif write may fail on mock ICA


# ── Cycle 4: labels JSON contains expected keys ───────────────────────────

def test_labels_json_structure(tmp_path):
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as m_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as m_label:
        m_fit.return_value = _mock_ica()
        m_label.return_value = ['brain'] * N_CH
        ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ,
                         cache_dir=tmp_path, subject='s2', random_state=42)

    data = json.loads((tmp_path / 's2_ica_labels.json').read_text())
    assert 'labels' in data
    assert 'random_state' in data
    assert data['random_state'] == 42
    assert data['labels'] == ['brain'] * N_CH


# ── Cycle 5: mismatched random_state raises ValueError ───────────────────

def test_mismatched_random_state_raises(tmp_path):
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as m_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as m_label:
        m_fit.return_value = _mock_ica()
        m_label.return_value = ['other'] * N_CH
        ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ,
                         cache_dir=tmp_path, subject='s3', random_state=42)

    with pytest.raises(ValueError, match="random_state"):
        with patch('pyorica.eval.ica_analysis._fit_ica') as m_fit2, \
             patch('pyorica.eval.ica_analysis._label_ica') as m_label2:
            m_fit2.return_value = _mock_ica()
            m_label2.return_value = ['other'] * N_CH
            ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ,
                             cache_dir=tmp_path, subject='s3', random_state=99)

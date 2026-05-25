"""Tests for eval.ica_analysis — behavior through public interface only."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

mne = pytest.importorskip("mne")
pytest.importorskip("mne_icalabel")

from pyorica.eval.ica_analysis import ic_source_energy

RNG = np.random.default_rng(77)
SFREQ = 256.0
N_SAMPLES = int(SFREQ * 30)   # 30 s — enough for ICA to run

CH_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
N_CH = len(CH_NAMES)

REQUIRED_KEYS = {'ic', 'label', 'ms_iir', 'ms_asr', 'ms_orica', 'pct_asr', 'pct_orica'}


def _make_stages():
    iir  = RNG.standard_normal((N_CH, N_SAMPLES))
    asr  = iir * 0.9
    orica = iir * 0.7
    return iir, asr, orica


def _mock_ica(n_components):
    """Return a minimal MNE ICA stub whose get_sources returns identity-mapped sources."""
    ica = MagicMock()
    ica.n_components_ = n_components
    sphere = np.eye(n_components)
    unmixing = np.eye(n_components)
    ica.unmixing_matrix_ = unmixing
    ica.pca_mean_ = np.zeros(n_components)
    ica.pca_components_ = np.eye(n_components)

    def fake_apply(raw, **kwargs):
        return raw

    def fake_get_sources(raw, **kwargs):
        src = MagicMock()
        src.get_data.return_value = raw.get_data()[:n_components]
        return src

    ica.apply = fake_apply
    ica.get_sources = fake_get_sources
    return ica


# ── Cycle 1: returns a list ──────────────────────────────────────────────

def test_returns_a_list():
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as mock_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as mock_label:
        mock_fit.return_value = _mock_ica(N_CH)
        mock_label.return_value = ['other'] * N_CH
        result = ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ)
    assert isinstance(result, list)


# ── Cycle 2: one dict per IC ─────────────────────────────────────────────

def test_one_dict_per_ic():
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as mock_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as mock_label:
        mock_fit.return_value = _mock_ica(N_CH)
        mock_label.return_value = ['other'] * N_CH
        result = ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ)
    assert len(result) == N_CH


# ── Cycle 3: each dict has all required keys ─────────────────────────────

def test_each_dict_has_required_keys():
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as mock_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as mock_label:
        mock_fit.return_value = _mock_ica(N_CH)
        mock_label.return_value = ['other'] * N_CH
        result = ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ)
    for row in result:
        assert REQUIRED_KEYS == set(row.keys()), f"Missing keys in {row}"


# ── Cycle 4: ms_iir is positive; pct values are floats ───────────────────

def test_ms_iir_positive_and_pct_are_floats():
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as mock_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as mock_label:
        mock_fit.return_value = _mock_ica(N_CH)
        mock_label.return_value = ['other'] * N_CH
        result = ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ)
    for row in result:
        assert row['ms_iir'] > 0
        assert isinstance(row['pct_asr'], float)
        assert isinstance(row['pct_orica'], float)


# ── Cycle 5: ic field is sequential integer index ─────────────────────────

def test_ic_field_is_sequential_index():
    iir, asr, orica = _make_stages()
    with patch('pyorica.eval.ica_analysis._fit_ica') as mock_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as mock_label:
        mock_fit.return_value = _mock_ica(N_CH)
        mock_label.return_value = ['other'] * N_CH
        result = ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ)
    assert [row['ic'] for row in result] == list(range(N_CH))


# ── Cycle 6 (slow): integration with real MNE ICA + ICLabel ──────────────

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning:mne")
@pytest.mark.filterwarnings("ignore::RuntimeWarning:mne_icalabel")
def test_integration_real_ica_and_iclabel():
    """Full run through MNE ICA + ICLabel on synthetic data; checks structure only."""
    iir, asr, orica = _make_stages()
    result = ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ)
    assert isinstance(result, list)
    assert len(result) == N_CH
    for row in result:
        assert REQUIRED_KEYS == set(row.keys())
        assert row['ms_iir'] > 0

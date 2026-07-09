"""Tests for eval.ica_analysis — behavior through public interface only."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

mne = pytest.importorskip("mne")
pytest.importorskip("mne_icalabel")

from pyorica.eval.ica_analysis import _make_raw, ic_source_energy

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


# ── Cycle 6: exclude_lead_seconds skips calibration window in MS stats ───

def _run_with_mock(iir, asr, orica, **kwargs):
    with patch('pyorica.eval.ica_analysis._fit_ica') as mock_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as mock_label:
        mock_fit.return_value = _mock_ica(N_CH)
        mock_label.return_value = ['other'] * N_CH
        return ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ, **kwargs)


def test_exclude_lead_seconds_changes_ms_stats():
    """Excluding lead-in samples must produce different MS values than using all data."""
    iir, asr, orica = _make_stages()
    # inject a large transient in the first 5 s (the "calibration lead-in")
    spike_end = int(SFREQ * 5)
    iir_spike = iir.copy()
    iir_spike[:, :spike_end] += 100.0

    result_all = _run_with_mock(iir_spike, asr, orica)
    result_skip = _run_with_mock(iir_spike, asr, orica, exclude_lead_seconds=5.0)

    # skipping the spike-contaminated lead-in must reduce ms_iir
    ms_all = sum(r['ms_iir'] for r in result_all)
    ms_skip = sum(r['ms_iir'] for r in result_skip)
    assert ms_skip < ms_all


def test_exclude_lead_seconds_zero_matches_default():
    """exclude_lead_seconds=0 must be identical to the default (no exclusion)."""
    iir, asr, orica = _make_stages()
    result_default = _run_with_mock(iir, asr, orica)
    result_zero = _run_with_mock(iir, asr, orica, exclude_lead_seconds=0.0)
    for r1, r2 in zip(result_default, result_zero):
        assert r1['ms_iir'] == r2['ms_iir']
        assert r1['ms_orica'] == r2['ms_orica']


def test_exclude_lead_seconds_too_large_raises():
    """exclude_lead_seconds >= recording length must raise ValueError."""
    iir, asr, orica = _make_stages()
    with pytest.raises(ValueError, match="leaves no samples"):
        _run_with_mock(iir, asr, orica, exclude_lead_seconds=9999.0)


# ── Cycle 7 (slow): integration with real MNE ICA + ICLabel ──────────────

@pytest.mark.slow
@pytest.mark.filterwarnings("error:The data has not been high-pass filtered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:The provided Raw instance does not seem:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:The provided Raw instance is not filtered:RuntimeWarning")
def test_integration_real_ica_and_iclabel():
    """Full run through MNE ICA + ICLabel on synthetic data; checks structure only.

    Errors (rather than ignores) the high-pass warning: ic_source_energy's
    default l_freq=1.0 must reach ICA.fit's info metadata, or this must fail.
    """
    iir, asr, orica = _make_stages()
    result = ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ)
    assert isinstance(result, list)
    assert len(result) == N_CH
    for row in result:
        assert REQUIRED_KEYS == set(row.keys())
        assert row['ms_iir'] > 0


# ── Cycle 8: _make_raw records the upstream filter in info metadata ───────

def test_make_raw_sets_highpass_lowpass_from_l_freq_h_freq():
    """Regression: ICA.fit() warns 'data has not been high-pass filtered' by
    checking raw.info['highpass']. _make_raw must record the bandpass already
    applied upstream (by IIRFilter) so that check reflects reality instead of
    mne.create_info's default of 0.0 (unfiltered)."""
    data = RNG.standard_normal((N_CH, 100))
    raw = _make_raw(data, CH_NAMES, SFREQ, l_freq=1.0, h_freq=50.0)
    assert raw.info['highpass'] == pytest.approx(1.0)
    assert raw.info['lowpass'] == pytest.approx(50.0)


def test_make_raw_leaves_highpass_lowpass_default_when_freqs_omitted():
    data = RNG.standard_normal((N_CH, 100))
    raw = _make_raw(data, CH_NAMES, SFREQ)
    assert raw.info['highpass'] == pytest.approx(0.0)


def test_ic_source_energy_default_l_freq_reaches_make_raw():
    """ic_source_energy's default l_freq/h_freq (matching PipelineConfig's
    iir_l_freq=1.0/iir_h_freq=50.0) must reach _make_raw without the caller
    passing them explicitly."""
    iir, asr, orica = _make_stages()
    captured = {}
    real_make_raw = _make_raw

    def _spy(data, ch_names, sfreq, l_freq=None, h_freq=None):
        captured['l_freq'] = l_freq
        captured['h_freq'] = h_freq
        return real_make_raw(data, ch_names, sfreq, l_freq=l_freq, h_freq=h_freq)

    with patch('pyorica.eval.ica_analysis._fit_ica') as mock_fit, \
         patch('pyorica.eval.ica_analysis._label_ica') as mock_label, \
         patch('pyorica.eval.ica_analysis._make_raw', side_effect=_spy):
        mock_fit.return_value = _mock_ica(N_CH)
        mock_label.return_value = ['other'] * N_CH
        ic_source_energy(iir, asr, orica, CH_NAMES, SFREQ)

    assert captured['l_freq'] == pytest.approx(1.0)
    assert captured['h_freq'] == pytest.approx(50.0)

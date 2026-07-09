"""Tests for ICLabelClassifier — behavior through public interface only."""

import warnings

import numpy as np
import pytest
from unittest.mock import patch

mne = pytest.importorskip("mne")

from pyorica.pipeline.classify import LABEL_NAMES, ICLabelClassifier

RNG = np.random.default_rng(5)
N_CH = 8
SFREQ = 256.0
LONG_N = int(SFREQ * 4)


def _make_info():
    ch_names = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'P3', 'P4']
    info = mne.create_info(ch_names, SFREQ, ch_types='eeg', verbose=False)
    info.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)
    return info


def _chunk(n_channels=N_CH, n_samples=LONG_N):
    return RNG.standard_normal((n_channels, n_samples))


def _all_brain(n):
    """(label_strings, prob_top1) as _run_icalabel would return for all-brain ICs."""
    return np.array(['brain'] * n, dtype=object), np.ones(n)


def _labels_probs(n, overrides):
    """All-brain baseline with per-index (label, prob) overrides."""
    labels = np.array(['brain'] * n, dtype=object)
    probs = np.ones(n)
    for idx, (label, prob) in overrides.items():
        labels[idx] = label
        probs[idx] = prob
    return labels, probs


# ── Cycle 1: returns bool array with correct shape ────────────────────────

def test_returns_bool_array_with_correct_shape():
    clf = ICLabelClassifier(_make_info())
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    with patch.object(clf, '_run_icalabel', return_value=_all_brain(N_CH)):
        mask = clf(data, sources, unmixing, mixing, SFREQ)
    assert mask.dtype == bool
    assert mask.shape == (N_CH,)


# ── Cycle 2 (regression #18): short chunk must not crash ──────────────────

def test_short_chunk_returns_no_artifacts_without_error():
    """Chunks shorter than the ICLabel FIR filter (~825 taps) must not crash."""
    clf = ICLabelClassifier(_make_info())
    data = _chunk(n_samples=40)  # well below the 825-sample minimum at 256 Hz
    sources = _chunk(n_samples=40)
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    with patch.object(clf, '_run_icalabel') as mocked:
        mask = clf(data, sources, unmixing, mixing, SFREQ)  # must not raise
    mocked.assert_not_called()
    assert mask.dtype == bool
    assert mask.shape == (N_CH,)
    assert not mask.any()


# ── Cycle 3: allow-list mode — known artifact label above threshold ─────────

def test_default_mode_marks_known_artifact_label_above_threshold():
    clf = ICLabelClassifier(_make_info())  # default threshold=0.7
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    labels_probs = _labels_probs(N_CH, {2: ('eye', 0.9)})
    with patch.object(clf, '_run_icalabel', return_value=labels_probs):
        mask = clf(data, sources, unmixing, mixing, SFREQ)
    assert mask[2]
    assert not mask[0]


def test_default_mode_does_not_mark_below_threshold():
    clf = ICLabelClassifier(_make_info())  # default threshold=0.7
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    labels_probs = _labels_probs(N_CH, {0: ('muscle', 0.6)})  # below 0.7
    with patch.object(clf, '_run_icalabel', return_value=labels_probs):
        mask = clf(data, sources, unmixing, mixing, SFREQ)
    assert not mask[0]


def test_default_mode_does_not_mark_unrecognized_label():
    """Unrecognized labels are never rejected — the allow-list only rejects
    labels explicitly listed in artifact_labels."""
    clf = ICLabelClassifier(_make_info())
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    labels_probs = _labels_probs(N_CH, {5: ('some_future_class', 0.95)})
    with patch.object(clf, '_run_icalabel', return_value=labels_probs):
        mask = clf(data, sources, unmixing, mixing, SFREQ)
    assert not mask[5]


# ── Cycle 5: label aliasing — old-style spellings and raw mne-icalabel forms ─

def test_custom_artifact_labels_accepts_legacy_alias_spelling():
    """artifact_labels={'eog'} must behave identically to {'eye'}."""
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    labels_probs = _labels_probs(N_CH, {1: ('eye', 0.9)})

    clf = ICLabelClassifier(_make_info(), artifact_labels={'eog'})
    with patch.object(clf, '_run_icalabel', return_value=labels_probs):
        mask = clf(data, sources, unmixing, mixing, SFREQ)
    assert mask[1]


def test_predicted_label_alias_from_run_icalabel_is_canonicalized():
    """A predicted label using an old-style spelling must still match the
    default canonical artifact_labels set."""
    clf = ICLabelClassifier(_make_info())
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    labels_probs = _labels_probs(N_CH, {6: ('ecg', 0.9)})
    with patch.object(clf, '_run_icalabel', return_value=labels_probs):
        mask = clf(data, sources, unmixing, mixing, SFREQ)
    assert mask[6]


# ── Cycle 5b: LABEL_NAMES and real mne_icalabel label strings ───────────────

def test_label_names_matches_mne_icalabel_output_order():
    """LABEL_NAMES must match the raw strings mne_icalabel.label_components()
    returns (see ICLABEL_NUMERICAL_TO_STRING) — visualize.py indexes colors
    and ordering by these exact strings, unmodified, from clf.snapshots."""
    assert LABEL_NAMES == [
        'brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise',
        'channel noise', 'other',
    ]


@pytest.mark.parametrize('raw_label', [
    'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise',
])
def test_default_mode_marks_real_mne_icalabel_label_strings(raw_label):
    """label_components() returns multi-word strings like 'muscle artifact',
    not the short internal names ('muscle'). Regression for a bug where the
    alias table only matched underscore-joined spellings, so no IC was ever
    rejected as an artifact in the default (fail-open) mode."""
    clf = ICLabelClassifier(_make_info())
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    labels_probs = _labels_probs(N_CH, {0: (raw_label, 0.9)})
    with patch.object(clf, '_run_icalabel', return_value=labels_probs):
        mask = clf(data, sources, unmixing, mixing, SFREQ)
    assert mask[0]


# ── Cycle 6: record_snapshots ──────────────────────────────────────────────

def test_record_snapshots_appends_one_entry_per_call():
    clf = ICLabelClassifier(_make_info(), record_snapshots=True)
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    labels_probs = _labels_probs(N_CH, {0: ('eye', 0.9)})

    with patch.object(clf, '_run_icalabel', return_value=labels_probs):
        mask1 = clf(data, sources, unmixing, mixing, SFREQ)
        clf(data, sources, unmixing, mixing, SFREQ)

    assert len(clf.snapshots) == 2
    seq, labels, probs, mask = clf.snapshots[0]
    assert seq == 0
    assert labels[0] == 'eye'
    assert probs[0] == pytest.approx(0.9)
    np.testing.assert_array_equal(mask, mask1)
    assert clf.snapshots[1][0] == 1


def test_record_snapshots_defaults_to_disabled():
    clf = ICLabelClassifier(_make_info())
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    with patch.object(clf, '_run_icalabel', return_value=_all_brain(N_CH)):
        clf(data, sources, unmixing, mixing, SFREQ)
    assert clf.snapshots == []


# ── Cycle 7: _run_icalabel — channel name normalisation to standard_1020 ────

def _fake_label_components(n_components):
    return {
        'labels': ['brain'] * n_components,
        'y_pred_proba': np.ones(n_components),
    }


def test_run_icalabel_normalises_uppercase_channel_names():
    """EEGLAB-style uppercase names (FP1, FZ) must reach mne_icalabel as
    standard_1020 mixed-case spelling (Fp1, Fz)."""
    upper_names = ['FP1', 'FZ', 'CZ', 'PZ', 'F3', 'F4', 'P3', 'P4']
    info = mne.create_info(upper_names, SFREQ, ch_types='eeg', verbose=False)
    clf = ICLabelClassifier(info)
    data = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)

    captured = {}

    def _capture(raw, ica, method):
        captured['ch_names'] = list(raw.ch_names)
        return _fake_label_components(N_CH)

    with patch('mne_icalabel.label_components', side_effect=_capture):
        clf._run_icalabel(data, unmixing, mixing, SFREQ, N_CH)

    assert 'Fp1' in captured['ch_names']
    assert 'Fz' in captured['ch_names']
    assert 'Cz' in captured['ch_names']
    assert 'Pz' in captured['ch_names']


# ── Cycle 8: apply_car_bandpass toggles CAR + 1-100 Hz filtering ────────────

def test_apply_car_bandpass_true_applies_car_and_filter():
    clf = ICLabelClassifier(_make_info(), apply_car_bandpass=True)
    data = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)

    captured = {}

    def _capture(raw, ica, method):
        captured['custom_ref_applied'] = raw.info['custom_ref_applied']
        captured['highpass'] = raw.info['highpass']
        captured['lowpass'] = raw.info['lowpass']
        return _fake_label_components(N_CH)

    with patch('mne_icalabel.label_components', side_effect=_capture):
        clf._run_icalabel(data, unmixing, mixing, SFREQ, N_CH)

    assert captured['custom_ref_applied'] != 0
    assert captured['highpass'] == pytest.approx(1.0)
    assert captured['lowpass'] == pytest.approx(100.0)


def test_apply_car_bandpass_false_skips_car_and_filter():
    clf = ICLabelClassifier(_make_info(), apply_car_bandpass=False)
    data = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)

    captured = {}

    def _capture(raw, ica, method):
        captured['custom_ref_applied'] = raw.info['custom_ref_applied']
        captured['highpass'] = raw.info['highpass']
        return _fake_label_components(N_CH)

    with patch('mne_icalabel.label_components', side_effect=_capture):
        clf._run_icalabel(data, unmixing, mixing, SFREQ, N_CH)

    assert captured['custom_ref_applied'] == 0
    assert captured['highpass'] == 0.0


def test_apply_car_bandpass_true_does_not_mutate_caller_data():
    """Regression: RawArray must not alias `data`'s buffer, since
    set_eeg_reference/filter mutate in place — a caller's ASR-cleaned chunk
    (or anything aliasing it, e.g. EEGPipeline._last_asr) must survive
    classification unchanged."""
    clf = ICLabelClassifier(_make_info(), apply_car_bandpass=True)
    data = _chunk()
    original = data.copy()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)

    with patch('mne_icalabel.label_components',
               side_effect=lambda raw, ica, method: _fake_label_components(N_CH)):
        clf._run_icalabel(data, unmixing, mixing, SFREQ, N_CH)

    np.testing.assert_array_equal(data, original)


# ── Cycle 9: ICA container receives the injected matrices, sliced if needed ─

def test_run_icalabel_injects_unmixing_and_mixing_into_ica_container():
    clf = ICLabelClassifier(_make_info())
    data = _chunk()
    unmixing = RNG.standard_normal((N_CH, N_CH))
    mixing = RNG.standard_normal((N_CH, N_CH))

    captured = {}

    def _capture(raw, ica, method):
        captured['unmixing'] = ica.unmixing_matrix_
        captured['mixing'] = ica.mixing_matrix_
        captured['method'] = ica.method
        return _fake_label_components(N_CH)

    with patch('mne_icalabel.label_components', side_effect=_capture):
        clf._run_icalabel(data, unmixing, mixing, SFREQ, N_CH)

    np.testing.assert_array_equal(captured['unmixing'], unmixing)
    np.testing.assert_array_equal(captured['mixing'], mixing)
    assert captured['method'] == 'picard'


def test_run_icalabel_slices_full_channel_unmixing_to_n_components():
    """When ORICA runs with fewer components than channels, unmixing may
    arrive as a full (n_channels, n_channels) matrix that must be sliced."""
    n_components = 5
    clf = ICLabelClassifier(_make_info())
    data = _chunk()
    unmixing_full = RNG.standard_normal((N_CH, N_CH))  # square, more rows than needed
    mixing = RNG.standard_normal((N_CH, n_components))

    captured = {}

    def _capture(raw, ica, method):
        captured['unmixing_shape'] = ica.unmixing_matrix_.shape
        return _fake_label_components(n_components)

    with patch('mne_icalabel.label_components', side_effect=_capture):
        clf._run_icalabel(data, unmixing_full, mixing, SFREQ, n_components)

    assert captured['unmixing_shape'] == (n_components, N_CH)


# ── Cycle 10: CAR/bandpass warning suppression is scoped to apply_car_bandpass=False

def _warn_then_label(n_components):
    import warnings

    def _side_effect(raw, ica, method):
        warnings.warn(
            'The provided Raw instance does not seem to be referenced to a '
            'common average reference (CAR).',
            RuntimeWarning,
        )
        return _fake_label_components(n_components)
    return _side_effect


def test_car_bandpass_warning_suppressed_when_flag_false():
    clf = ICLabelClassifier(_make_info(), apply_car_bandpass=False)
    data = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)

    with patch('mne_icalabel.label_components', side_effect=_warn_then_label(N_CH)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            clf._run_icalabel(data, unmixing, mixing, SFREQ, N_CH)

    assert not any('common average reference' in str(w.message) for w in caught)


def test_car_bandpass_warning_not_suppressed_when_flag_true():
    clf = ICLabelClassifier(_make_info(), apply_car_bandpass=True)
    data = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)

    with patch('mne_icalabel.label_components', side_effect=_warn_then_label(N_CH)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            clf._run_icalabel(data, unmixing, mixing, SFREQ, N_CH)

    assert any('common average reference' in str(w.message) for w in caught)


# ── Cycle 11 (slow): integration with actual ICLabel network ───────────────

@pytest.mark.slow
def test_iclabel_integration_returns_valid_mask():
    """ICLabelClassifier runs end-to-end without error and returns a valid mask."""
    clf = ICLabelClassifier(_make_info())
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    mask = clf(data, sources, unmixing, mixing, SFREQ)
    assert mask.dtype == bool
    assert mask.shape == (N_CH,)


@pytest.mark.slow
def test_iclabel_integration_respects_apply_car_bandpass_true():
    clf = ICLabelClassifier(_make_info(), apply_car_bandpass=True)
    data = _chunk()
    sources = _chunk()
    unmixing = np.eye(N_CH)
    mixing = np.eye(N_CH)
    mask = clf(data, sources, unmixing, mixing, SFREQ)
    assert mask.dtype == bool
    assert mask.shape == (N_CH,)

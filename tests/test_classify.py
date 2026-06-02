"""Tests for ICLabelClassifier — behavior through public interface only."""

import numpy as np
import pytest
from unittest.mock import patch

mne = pytest.importorskip("mne")

from pyorica.pipeline.classify import ICLabelClassifier

RNG = np.random.default_rng(5)
N_CH = 8
SFREQ = 256.0


def _make_info():
    ch_names = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'P3', 'P4']
    info = mne.create_info(ch_names, SFREQ, ch_types='eeg', verbose=False)
    info.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)
    return info


def _call(clf, sources, unmixing=None, mixing=None, data=None):
    data = RNG.standard_normal((N_CH, int(SFREQ * 4))) if data is None else data
    unmixing = np.eye(N_CH) if unmixing is None else unmixing
    mixing = np.eye(N_CH) if mixing is None else mixing
    return clf(data, sources, unmixing, mixing, SFREQ)


def _labels_probs(n, label_strings, prob_top1):
    return (
        np.asarray(label_strings, dtype=object),
        np.asarray(prob_top1, dtype=np.float64),
    )


# ── Cycle 1: returns bool array with correct shape ────────────────────────

def test_returns_bool_array_with_correct_shape():
    clf = ICLabelClassifier(_make_info())
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    labels = ["brain"] * N_CH
    probs = [1.0] * N_CH
    with patch.object(clf, '_run_icalabel', return_value=_labels_probs(N_CH, labels, probs)):
        mask = _call(clf, sources)
    assert mask.dtype == bool
    assert mask.shape == (N_CH,)


# ── Cycle 2: all-brain prediction → no artifacts ─────────────────────────

def test_all_brain_marks_no_artifacts():
    clf = ICLabelClassifier(_make_info())
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    labels = ["brain"] * N_CH
    probs = [1.0] * N_CH
    with patch.object(clf, '_run_icalabel', return_value=_labels_probs(N_CH, labels, probs)):
        mask = _call(clf, sources)
    assert not mask.any()


# ── Cycle 3: artifact above threshold → marked ───────────────────────────

def test_artifact_above_threshold_is_marked():
    clf = ICLabelClassifier(_make_info(), threshold=0.5)
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    labels = ["brain"] * N_CH
    labels[2] = "eye"
    probs = [1.0] * N_CH
    probs[2] = 0.9
    with patch.object(clf, '_run_icalabel', return_value=_labels_probs(N_CH, labels, probs)):
        mask = _call(clf, sources)
    assert mask[2]
    assert not mask[0]


# ── Cycle 4: below threshold not marked ──────────────────────────────────

def test_below_threshold_not_marked():
    clf = ICLabelClassifier(_make_info(), threshold=0.9)
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    labels = ["muscle"] + ["brain"] * (N_CH - 1)
    probs = [0.7] + [1.0] * (N_CH - 1)
    with patch.object(clf, '_run_icalabel', return_value=_labels_probs(N_CH, labels, probs)):
        mask = _call(clf, sources)
    assert not mask[0]


# ── Cycle 5: legacy protects 'other' even above threshold ────────────────

def test_other_label_never_marked():
    clf = ICLabelClassifier(_make_info(), threshold=0.5)
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    labels = ["other"] + ["brain"] * (N_CH - 1)
    probs = [0.95] + [1.0] * (N_CH - 1)
    with patch.object(clf, '_run_icalabel', return_value=_labels_probs(N_CH, labels, probs)):
        mask = _call(clf, sources)
    assert not mask[0]


# ── Cycle 6 (slow): integration with actual ICLabel network ───────────────

@pytest.mark.slow
def test_iclabel_integration_returns_valid_mask():
    """ICLabelClassifier runs without error and returns a valid bool mask."""
    clf = ICLabelClassifier(_make_info())
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    mask = _call(clf, sources)
    assert mask.dtype == bool
    assert mask.shape == (N_CH,)


# ── Cycle 7 (regression #18): short chunk must not crash ─────────────────

def test_short_chunk_returns_no_artifacts_without_error():
    """Chunks shorter than the ICLabel FIR filter (~825 taps) must not crash."""
    clf = ICLabelClassifier(_make_info())
    sources = RNG.standard_normal((N_CH, 40))
    data = RNG.standard_normal((N_CH, 40))
    mask = clf(data, sources, np.eye(N_CH), np.eye(N_CH), SFREQ)
    assert mask.dtype == bool
    assert mask.shape == (N_CH,)
    assert not mask.any()

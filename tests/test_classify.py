"""Tests for ICLabelClassifier — behavior through public interface only."""

import numpy as np
import pytest
from unittest.mock import patch

mne = pytest.importorskip("mne")

from pyorica.pipeline.classify import ICLabelClassifier

RNG = np.random.default_rng(5)
N_CH = 8
SFREQ = 256.0

LABEL_NAMES = ['brain', 'muscle', 'eog', 'ecg', 'line_noise', 'ch_noise', 'other']


def _make_info():
    ch_names = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'P3', 'P4']
    info = mne.create_info(ch_names, SFREQ, ch_types='eeg', verbose=False)
    info.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)
    return info


def _all_brain_proba(n):
    proba = np.zeros((n, 7))
    proba[:, 0] = 1.0
    return proba


# ── Cycle 1: returns bool array with correct shape ────────────────────────

def test_returns_bool_array_with_correct_shape():
    clf = ICLabelClassifier(_make_info())
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    A = np.eye(N_CH)
    with patch.object(clf, '_get_probabilities', return_value=_all_brain_proba(N_CH)):
        mask = clf(sources, A, SFREQ)
    assert mask.dtype == bool
    assert mask.shape == (N_CH,)


# ── Cycle 2: all-brain prediction → no artifacts ─────────────────────────

def test_all_brain_marks_no_artifacts():
    clf = ICLabelClassifier(_make_info())
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    A = np.eye(N_CH)
    with patch.object(clf, '_get_probabilities', return_value=_all_brain_proba(N_CH)):
        mask = clf(sources, A, SFREQ)
    assert not mask.any()


# ── Cycle 3: artifact above threshold → marked ───────────────────────────

def test_artifact_above_threshold_is_marked():
    clf = ICLabelClassifier(_make_info(), threshold=0.5)
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    A = np.eye(N_CH)
    proba = _all_brain_proba(N_CH)
    proba[2, 0] = 0.1
    proba[2, 2] = 0.9  # eye blink (index 2) on component 2
    with patch.object(clf, '_get_probabilities', return_value=proba):
        mask = clf(sources, A, SFREQ)
    assert mask[2]
    assert not mask[0]


# ── Cycle 4: below threshold not marked ──────────────────────────────────

def test_below_threshold_not_marked():
    clf = ICLabelClassifier(_make_info(), threshold=0.9)
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    A = np.eye(N_CH)
    proba = _all_brain_proba(N_CH)
    proba[0, 0] = 0.3
    proba[0, 1] = 0.7  # muscle at 0.7, below threshold 0.9
    with patch.object(clf, '_get_probabilities', return_value=proba):
        mask = clf(sources, A, SFREQ)
    assert not mask[0]


# ── Cycle 5: custom artifact_labels controls which labels are rejected ────

def test_custom_artifact_labels_excludes_unlisted():
    clf = ICLabelClassifier(_make_info(), artifact_labels={'muscle', 'eog'})
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    A = np.eye(N_CH)
    proba = _all_brain_proba(N_CH)
    proba[0, 0] = 0.2
    proba[0, 6] = 0.8  # 'other' above threshold, but not in artifact_labels
    with patch.object(clf, '_get_probabilities', return_value=proba):
        mask = clf(sources, A, SFREQ)
    assert not mask[0]


# ── Cycle 6 (slow): integration with actual ICLabel network ───────────────

@pytest.mark.slow
def test_iclabel_integration_returns_valid_mask():
    """ICLabelClassifier runs without error and returns a valid bool mask."""
    clf = ICLabelClassifier(_make_info())
    sources = RNG.standard_normal((N_CH, int(SFREQ * 4)))
    A = np.eye(N_CH)
    mask = clf(sources, A, SFREQ)
    assert mask.dtype == bool
    assert mask.shape == (N_CH,)

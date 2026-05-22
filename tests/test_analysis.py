"""Tests for eval.analysis — behavior through public interface only."""

import numpy as np
import pytest

from pyorica.eval.analysis import summarize, rms_reduction_db
from pyorica.eval.runner import RunResult

RNG = np.random.default_rng(17)
N_CH = 8
N_CHUNKS = 20


def _make_result(rms_in=None, rms_out=None):
    if rms_in is None:
        rms_in = RNG.uniform(0.5, 1.5, N_CHUNKS)
    if rms_out is None:
        rms_out = rms_in * RNG.uniform(0.3, 0.7, N_CHUNKS)
    n_samples = N_CHUNKS * 64
    return RunResult(
        output=RNG.standard_normal((N_CH, n_samples)),
        rms_input=np.array(rms_in, dtype=np.float64),
        rms_output=np.array(rms_out, dtype=np.float64),
        chunk_size=64,
        n_channels=N_CH,
        n_samples=n_samples,
    )


# ── Cycle 1: rms_reduction_db computes correct value ─────────────────────

def test_rms_reduction_db_known_value():
    rms_in = np.array([1.0, 1.0])
    rms_out = np.array([0.1, 0.1])  # 20 dB reduction
    db = rms_reduction_db(rms_in, rms_out)
    assert db == pytest.approx(20.0, abs=0.01)


def test_rms_reduction_db_zero_reduction():
    rms = np.array([1.0, 2.0, 0.5])
    db = rms_reduction_db(rms, rms)
    assert db == pytest.approx(0.0, abs=1e-6)


# ── Cycle 2: summarize returns dict with required keys ────────────────────

def test_summarize_has_required_keys():
    result = _make_result()
    summary = summarize(result)
    for key in ('mean_rms_reduction_db', 'mean_rms_input', 'mean_rms_output',
                 'n_chunks', 'n_channels'):
        assert key in summary, f"Missing key: {key}"


def test_summarize_n_chunks_matches_result():
    result = _make_result()
    summary = summarize(result)
    assert summary['n_chunks'] == N_CHUNKS


def test_summarize_n_channels_matches_result():
    result = _make_result()
    summary = summarize(result)
    assert summary['n_channels'] == N_CH


# ── Cycle 3: summarize values are numerically correct ─────────────────────

def test_summarize_mean_rms_values():
    rms_in = np.array([1.0, 2.0])
    rms_out = np.array([0.5, 1.0])
    result = _make_result(rms_in=rms_in, rms_out=rms_out)
    summary = summarize(result)
    assert summary['mean_rms_input'] == pytest.approx(1.5)
    assert summary['mean_rms_output'] == pytest.approx(0.75)


def test_summarize_reduction_db_positive_when_rms_decreases():
    rms_in = np.ones(N_CHUNKS) * 1.0
    rms_out = np.ones(N_CHUNKS) * 0.1  # 20 dB reduction
    result = _make_result(rms_in=rms_in, rms_out=rms_out)
    summary = summarize(result)
    assert summary['mean_rms_reduction_db'] == pytest.approx(20.0, abs=0.1)

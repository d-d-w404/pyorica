"""Tests for eval.runner — behavior through public interface only."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pyorica.eval.runner import run, RunResult
from pyorica.pipeline.pipeline import EEGPipeline

RNG = np.random.default_rng(13)
N_CH = 8
SFREQ = 256.0
CHUNK = 64
N_SAMPLES = CHUNK * 10


def _make_pipeline(**kwargs):
    return EEGPipeline(n_channels=N_CH, sfreq=SFREQ, **kwargs)


def _make_data():
    return RNG.standard_normal((N_CH, N_SAMPLES))


# ── Cycle 1: output shape matches input ──────────────────────────────────

def test_output_shape_matches_input():
    data = _make_data()
    result = run(_make_pipeline(), data, chunk_size=CHUNK)
    assert isinstance(result, RunResult)
    assert result.output.shape == data.shape


# ── Cycle 2: output dtype is float64 ─────────────────────────────────────

def test_output_dtype_is_float64():
    data = _make_data()
    result = run(_make_pipeline(), data, chunk_size=CHUNK)
    assert result.output.dtype == np.float64


# ── Cycle 3: rms metrics have length n_chunks ─────────────────────────────

def test_rms_metrics_have_n_chunks_length():
    data = _make_data()
    result = run(_make_pipeline(), data, chunk_size=CHUNK)
    n_chunks = int(np.ceil(N_SAMPLES / CHUNK))
    assert len(result.rms_input) == n_chunks
    assert len(result.rms_output) == n_chunks


# ── Cycle 4: calibration_data triggers fit ────────────────────────────────

def test_calibration_data_calls_fit():
    p = _make_pipeline()
    spy = MagicMock(wraps=p.fit)
    p.fit = spy

    calibration = RNG.standard_normal((N_CH, int(SFREQ * 5)))
    run(p, _make_data(), chunk_size=CHUNK, calibration_data=calibration)
    spy.assert_called_once()


# ── Cycle 5: no calibration → fit not called ─────────────────────────────

def test_no_calibration_does_not_call_fit():
    p = _make_pipeline()
    spy = MagicMock(wraps=p.fit)
    p.fit = spy

    run(p, _make_data(), chunk_size=CHUNK)
    spy.assert_not_called()


# ── Cycle 6: pass-all classifier → rms_in ≈ rms_out (no zeroing) ─────────

def test_pass_all_rms_ratio_near_one():
    pass_all = lambda sources, A, sfreq: np.zeros(sources.shape[0], dtype=bool)
    p = _make_pipeline(classifier=pass_all)
    p.fit(RNG.standard_normal((N_CH, int(SFREQ * 5))))

    data = _make_data()
    result = run(p, data, chunk_size=CHUNK)
    ratio = result.rms_output / (result.rms_input + 1e-12)
    # ORICA round-trip preserves signal; ratio should be within 10× in either direction
    assert np.all(ratio > 0.1) and np.all(ratio < 10.0)

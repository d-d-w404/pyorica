"""Tests for the ICLabel classification interval — behavior through public interface."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from pyorica.pipeline.pipeline import EEGPipeline

RNG = np.random.default_rng(42)
N_CH = 8
SFREQ = 250.0
CHUNK_SIZE = 1000  # samples per chunk (4 s at 250 Hz)


def _make_counting_classifier():
    """Return a mock classifier that records every call and returns all-False."""
    clf = MagicMock(side_effect=lambda sources, A, sfreq: np.zeros(sources.shape[0], dtype=bool))
    return clf


def _run_pipeline(pipeline: EEGPipeline, total_samples: int, chunk_size: int = CHUNK_SIZE):
    data = RNG.standard_normal((N_CH, total_samples))
    for start in range(0, total_samples, chunk_size):
        chunk = data[:, start: start + chunk_size]
        pipeline.process(chunk)


# ── Cycle 1: classifier called once per interval, not every chunk ─────────

def test_classifier_called_once_per_30s_interval():
    """With classify_interval_s=30 and 120 s of data, classifier runs 4 times."""
    clf = _make_counting_classifier()
    pipeline = EEGPipeline(
        n_channels=N_CH, sfreq=SFREQ,
        classifier=clf,
        classify_interval_s=30.0,
    )
    total_samples = int(SFREQ * 120)  # 120 s
    _run_pipeline(pipeline, total_samples)

    # 4 intervals of 30 s in 120 s
    assert clf.call_count == 4, f"Expected 4 calls, got {clf.call_count}"


# ── Cycle 2: interval=0 preserves per-chunk behaviour ────────────────────

def test_classify_interval_zero_calls_every_chunk():
    """classify_interval_s=0 must call the classifier on every chunk (legacy)."""
    clf = _make_counting_classifier()
    total_samples = int(SFREQ * 4)  # 4 chunks of 1000 samples
    pipeline = EEGPipeline(
        n_channels=N_CH, sfreq=SFREQ,
        classifier=clf,
        classify_interval_s=0,
    )
    _run_pipeline(pipeline, total_samples)

    n_chunks = total_samples // CHUNK_SIZE
    assert clf.call_count == n_chunks, f"Expected {n_chunks}, got {clf.call_count}"


# ── Cycle 3: causal — mask applied to chunk N came from data before N ────

def test_first_chunk_uses_no_artifact_mask():
    """Before 30 s of data have accumulated, no ICs should be zeroed (empty mask)."""
    all_zero = MagicMock(side_effect=lambda sources, A, sfreq: np.ones(sources.shape[0], dtype=bool))

    pipeline = EEGPipeline(
        n_channels=N_CH, sfreq=SFREQ,
        classifier=all_zero,
        classify_interval_s=30.0,
    )

    # Process only 1 chunk (4 s < 30 s interval): classifier should not have fired yet,
    # so the output must NOT be all-zero (i.e. sources were not zeroed).
    chunk = RNG.standard_normal((N_CH, CHUNK_SIZE))
    out = pipeline.process(chunk)

    # If the causal mask was incorrectly applied right away, output would be ~zero.
    # With the correct causal behaviour, the mask hasn't fired yet → output is non-zero.
    assert np.max(np.abs(out)) > 1e-6, \
        "First chunk was incorrectly zeroed; mask should not apply until after first interval"


# ── Cycle 4: mask updates after interval, applied to next chunk ──────────

def test_mask_from_interval_applied_to_subsequent_chunks():
    """After the first 30 s interval fires, the resulting mask is applied to later chunks.

    At 250 Hz with CHUNK_SIZE=1000, the 30 s threshold (7500 samples) is crossed
    during the 8th chunk (cumulative: 8000 ≥ 7500). The cached mask is applied to
    the 9th chunk and later — never to data used to compute it.
    """
    zero_all = MagicMock(side_effect=lambda sources, A, sfreq: np.ones(sources.shape[0], dtype=bool))

    pipeline = EEGPipeline(
        n_channels=N_CH, sfreq=SFREQ,
        classifier=zero_all,
        classify_interval_s=30.0,
    )

    # Feed 8 chunks (8 000 samples > 7 500): interval fires during the 8th chunk;
    # the resulting all-True mask is cached for the next chunk.
    for _ in range(8):
        pipeline.process(RNG.standard_normal((N_CH, CHUNK_SIZE)))

    # 9th chunk: the cached all-True mask is applied → output must be near-zero
    out = pipeline.process(RNG.standard_normal((N_CH, CHUNK_SIZE)))
    assert np.max(np.abs(out)) < 1e-8, \
        "Chunk after interval should have all ICs zeroed"


# ── Cycle 5: classify_interval_s wired from PipelineConfig ───────────────

def test_classify_interval_wired_from_config():
    """PipelineConfig.classify_interval_s is respected by EEGPipeline."""
    from pyorica.config import PipelineConfig
    clf = _make_counting_classifier()
    config = PipelineConfig(classify_interval_s=30.0)
    pipeline = EEGPipeline(n_channels=N_CH, sfreq=SFREQ, classifier=clf, config=config)

    total_samples = int(SFREQ * 60)  # 60 s → 2 intervals
    _run_pipeline(pipeline, total_samples)
    assert clf.call_count == 2

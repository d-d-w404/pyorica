"""Tests for ArrayStream — behavior through public interface only."""

import numpy as np
import pytest

from pyorica.streaming.array import ArrayStream

RNG = np.random.default_rng(7)
N_CH = 6
N_SAMPLES = 300
CHUNK = 64


# ── Cycle 1: chunk shape ──────────────────────────────────────────────────

def test_chunks_have_correct_shape():
    data = RNG.standard_normal((N_CH, N_SAMPLES))
    stream = ArrayStream(data, chunk_size=CHUNK)
    for chunk in stream:
        assert chunk.shape[0] == N_CH
        assert chunk.shape[1] <= CHUNK


# ── Cycle 2: all samples yielded including remainder ──────────────────────

def test_all_samples_yielded():
    data = RNG.standard_normal((N_CH, N_SAMPLES))
    stream = ArrayStream(data, chunk_size=CHUNK)
    reconstructed = np.concatenate(list(stream), axis=-1)
    np.testing.assert_array_equal(reconstructed, data)


def test_remainder_chunk_is_yielded():
    """N_SAMPLES=300, CHUNK=64 → last chunk has 300 % 64 = 44 samples."""
    data = RNG.standard_normal((N_CH, N_SAMPLES))
    stream = ArrayStream(data, chunk_size=CHUNK)
    chunks = list(stream)
    assert chunks[-1].shape[1] == N_SAMPLES % CHUNK


# ── Cycle 3: len and re-iteration ─────────────────────────────────────────

def test_len_returns_number_of_chunks():
    data = RNG.standard_normal((N_CH, N_SAMPLES))
    stream = ArrayStream(data, chunk_size=CHUNK)
    assert len(stream) == int(np.ceil(N_SAMPLES / CHUNK))


def test_reiteration_replays_from_start():
    data = RNG.standard_normal((N_CH, N_SAMPLES))
    stream = ArrayStream(data, chunk_size=CHUNK)
    first_pass = list(stream)
    second_pass = list(stream)
    for a, b in zip(first_pass, second_pass):
        np.testing.assert_array_equal(a, b)


# ── Edge case: 1-D input raises ───────────────────────────────────────────

def test_1d_input_raises():
    with pytest.raises(ValueError):
        ArrayStream(RNG.standard_normal(N_SAMPLES), chunk_size=CHUNK)

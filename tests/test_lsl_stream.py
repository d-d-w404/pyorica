"""Tests for LSLStream — behavior through public interface only."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

pylsl = pytest.importorskip("pylsl")

from pyorica.streaming.lsl import LSLStream

RNG = np.random.default_rng(11)
N_CH = 4
SFREQ = 256.0
CHUNK = 32


def _make_mock_inlet(n_ch=N_CH, sfreq=SFREQ, chunk_data=None):
    """Return a mock pylsl StreamInlet with configurable pull_chunk output."""
    if chunk_data is None:
        chunk_data = RNG.standard_normal((CHUNK, n_ch)).tolist()
    inlet = MagicMock()
    inlet.info.return_value.channel_count.return_value = n_ch
    inlet.info.return_value.nominal_srate.return_value = sfreq
    inlet.pull_chunk.return_value = (chunk_data, [0.0] * len(chunk_data))
    return inlet


def _patched_stream(stream_name='TestEEG', n_ch=N_CH, sfreq=SFREQ, chunk_data=None,
                    chunk_size=CHUNK):
    """Return (LSLStream, mock_inlet) with pylsl mocked."""
    inlet = _make_mock_inlet(n_ch, sfreq, chunk_data)
    with patch('pylsl.resolve_byprop', return_value=[MagicMock()]), \
         patch('pylsl.StreamInlet', return_value=inlet):
        stream = LSLStream(stream_name, chunk_size=chunk_size)
    return stream, inlet


# ── Cycle 1: yields chunk with correct channel count ─────────────────────

def test_yields_correct_channel_count():
    stream, inlet = _patched_stream()
    chunk = next(iter(stream))
    assert chunk.shape[0] == N_CH


# ── Cycle 2: chunk columns ≤ chunk_size ───────────────────────────────────

def test_chunk_has_at_most_chunk_size_samples():
    stream, inlet = _patched_stream()
    chunk = next(iter(stream))
    assert chunk.shape[1] <= CHUNK


# ── Cycle 3: data is transposed from pylsl (n_samples × n_ch) layout ─────

def test_chunk_shape_is_channels_x_samples():
    # pylsl gives (n_samples, n_channels); LSLStream must transpose
    raw_chunk = [[float(i * N_CH + ch) for ch in range(N_CH)] for i in range(CHUNK)]
    stream, _ = _patched_stream(chunk_data=raw_chunk)
    chunk = next(iter(stream))
    assert chunk.shape == (N_CH, CHUNK)
    # first row = channel 0 values across time
    expected_ch0 = np.array([row[0] for row in raw_chunk])
    np.testing.assert_array_equal(chunk[0], expected_ch0)


# ── Cycle 4: raises RuntimeError when no stream found ─────────────────────

def test_raises_if_stream_not_found():
    with patch('pylsl.resolve_byprop', return_value=[]):
        with pytest.raises(RuntimeError, match='not found'):
            LSLStream('NonExistentStream', chunk_size=CHUNK, timeout=0.1)


# ── Cycle 5: exposes n_channels and sfreq ─────────────────────────────────

def test_exposes_n_channels_and_sfreq():
    stream, _ = _patched_stream(n_ch=16, sfreq=512.0)
    assert stream.n_channels == 16
    assert stream.sfreq == pytest.approx(512.0)


# ── Cycle 6: empty pull returns no chunk (skips blank intervals) ──────────

def test_empty_pull_is_skipped():
    inlet = _make_mock_inlet()
    # first call: empty; second call: real data
    full_data = RNG.standard_normal((CHUNK, N_CH)).tolist()
    inlet.pull_chunk.side_effect = [
        ([], []),
        (full_data, [0.0] * CHUNK),
    ]
    with patch('pylsl.resolve_byprop', return_value=[MagicMock()]), \
         patch('pylsl.StreamInlet', return_value=inlet):
        stream = LSLStream('TestEEG', chunk_size=CHUNK)
    chunk = next(iter(stream))
    assert chunk.shape == (N_CH, CHUNK)

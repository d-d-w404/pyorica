"""Tests for ORICAFilter — behavior through public interface only."""

import numpy as np
import pytest
import os

from pyorica.orica.core import ORICAFilter

RNG = np.random.default_rng(0)
N_CH = 8
SFREQ = 256.0
CHUNK = 64

SET_PATH = os.path.join(
    os.path.dirname(__file__),
    "data/SIM_STAT_16ch_3min.set",
)
FDT_PATH = os.path.join(
    os.path.dirname(__file__),
    "data/SIM_STAT_16ch_3min.fdt",
)


# ── Cycle 1: transform output shape ───────────────────────────────────────

def test_transform_returns_components_x_samples():
    orica = ORICAFilter(N_CH, SFREQ)
    chunk = RNG.standard_normal((N_CH, CHUNK))
    sources = orica.transform(chunk)
    assert sources.shape == (N_CH, CHUNK)


# ── Cycle 2: inverse_transform round-trip ─────────────────────────────────

def test_inverse_transform_roundtrip():
    orica = ORICAFilter(N_CH, SFREQ)
    chunk = RNG.standard_normal((N_CH, CHUNK)).astype(np.float64)
    reconstructed = orica.inverse_transform(orica.transform(chunk))
    np.testing.assert_allclose(reconstructed, chunk, atol=1e-6)


# ── Cycle 3: update mutates weights ───────────────────────────────────────

def test_update_mutates_weights():
    orica = ORICAFilter(N_CH, SFREQ)
    W_init = orica.weights_.copy()
    for _ in range(10):
        orica.update(RNG.standard_normal((N_CH, CHUNK)))
    assert not np.allclose(orica.weights_, W_init), \
        "weights_ should change after update()"


# ── Cycle 4: fit warm-start differs from cold start ───────────────────────

def test_fit_warm_start_differs_from_cold_start():
    calibration = RNG.standard_normal((N_CH, int(SFREQ * 10)))  # 10 s

    cold = ORICAFilter(N_CH, SFREQ)
    warm = ORICAFilter(N_CH, SFREQ)
    warm.fit(calibration)

    # after the same subsequent updates, weights should differ
    for _ in range(5):
        chunk = RNG.standard_normal((N_CH, CHUNK))
        cold.update(chunk)
        warm.update(chunk)

    assert not np.allclose(cold.weights_, warm.weights_), \
        "warm-started weights should differ from cold-started weights"


# ── Cycle 5: wrong channel count raises ───────────────────────────────────

def test_update_wrong_channels_raises():
    orica = ORICAFilter(N_CH, SFREQ)
    with pytest.raises(ValueError, match="channels"):
        orica.update(RNG.standard_normal((N_CH + 1, CHUNK)))


def test_fit_wrong_channels_raises():
    orica = ORICAFilter(N_CH, SFREQ)
    with pytest.raises(ValueError, match="channels"):
        orica.fit(RNG.standard_normal((N_CH + 2, 512)))


# ── Cycle 6 (slow): cross-talk error on SIM_STAT_16ch_3min.set ────────────

def _load_sim_dataset():
    """Load SIM_STAT EEG data and ground-truth mixing matrix."""
    import scipy.io

    mat = scipy.io.loadmat(SET_PATH, squeeze_me=True, struct_as_record=False)
    EEG = mat["EEG"]

    # ground-truth mixing matrix: EEG.etc.LFM[0], shape (n_ch, n_ch)
    A_true = np.array(EEG.etc.LFM[0], dtype=np.float64)

    # raw data from external .fdt (channels × samples, float32 column-major)
    n_ch = int(EEG.nbchan)
    n_pts = int(EEG.pnts)
    data = np.fromfile(FDT_PATH, dtype="<f4", count=n_ch * n_pts).reshape(
        (n_ch, n_pts), order="F"
    ).astype(np.float64)

    sfreq = float(EEG.srate)
    return data, A_true, sfreq


def _cross_talk_error(W, sphere, A_true):
    """Cross-talk error from testScript.m: 0 = perfect, 1 = no separation."""
    H = W @ sphere @ A_true
    C = H ** 2
    n = C.shape[0]
    term1 = np.sum(np.max(C, axis=0) / np.sum(C, axis=0))
    term2 = np.sum(np.max(C, axis=1) / np.sum(C, axis=1))
    return (n - term1 / 2 - term2 / 2) / (n - 1)


@pytest.mark.slow
def test_cross_talk_error_on_sim_stat():
    """ORICA recovers sources with cross-talk error < 0.3 on the 3-min simulation."""
    if not os.path.exists(SET_PATH):
        pytest.skip("SIM_STAT dataset not found")

    data, A_true, sfreq = _load_sim_dataset()
    n_ch = data.shape[0]

    # params matching testScript.m: online whitening, block=8, cooling, localstat=Inf
    orica = ORICAFilter(
        n_components=n_ch,
        sfreq=sfreq,
        ff_profile="cooling",
        block_size_white=8,
        block_size_ica=8,
        tau_const=np.inf,
        gamma=0.6,
        lambda_0=0.995,
    )

    chunk_size = 8
    n_pts = data.shape[1]
    for start in range(0, n_pts, chunk_size):
        orica.update(data[:, start:start + chunk_size])

    cte = _cross_talk_error(orica.weights_, orica.sphere_, A_true)
    assert cte < 0.3, f"Cross-talk error {cte:.4f} exceeds threshold 0.3"

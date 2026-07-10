"""Tests for ORICAFilter — behavior through public interface only."""

import numpy as np
import pytest
import os
import warnings

from pyorica.orica.core import ORICAFilter, _orica_block_ranges

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


# ── Cycle 5: block partitioning (legacy ORICA even split) ─────────────────

def test_orica_block_ranges_examples_from_legacy():
    """Spot-check against hand-computed legacy ORICA examples."""
    assert list(_orica_block_ranges(50, 32)) == [(0, 50)]
    assert list(_orica_block_ranges(100, 32)) == [(0, 33), (33, 66), (66, 100)]


def test_orica_block_ranges_covers_all_samples():
    """Every (n_pts, block_size) pair must cover exactly n_pts samples with no gaps."""
    for n_pts in (8, 31, 32, 50, 100, 1000):
        for block_size in (8, 32, 37, 63):
            ranges = list(_orica_block_ranges(n_pts, block_size))
            if n_pts < block_size:
                assert ranges == []
                continue
            assert ranges[0][0] == 0
            assert ranges[-1][1] == n_pts
            covered = sum(end - start for start, end in ranges)
            assert covered == n_pts
            for (s0, e0), (s1, e1) in zip(ranges, ranges[1:]):
                assert e0 == s1


def test_update_processes_all_samples_when_chunk_not_multiple_of_block():
    """update() must process all samples; _counter must advance by exactly n_pts."""
    orica = ORICAFilter(N_CH, SFREQ, block_size_white=8, block_size_ica=8)
    counter_before = orica._counter
    orica.update(RNG.standard_normal((N_CH, 70)))
    assert orica._counter - counter_before == 70


def test_update_processes_all_samples_with_unequal_block_sizes():
    """block_size_white and block_size_ica are independent stationarity knobs;
    the ICA pass must cover every sample even when they differ."""
    orica = ORICAFilter(N_CH, SFREQ, block_size_white=25, block_size_ica=8)
    counter_before = orica._counter
    orica.update(RNG.standard_normal((N_CH, 100)))
    assert orica._counter - counter_before == 100


# ── Cycle 6: ff_profile selects mutually-exclusive λ behavior ────────────

def test_ff_profiles_diverge_from_each_other():
    """constant, cooling, and adaptive are independent paths (no override
    flag) — each must produce distinct weights/sphere over the same data."""
    chunks = [RNG.standard_normal((N_CH, CHUNK)) for _ in range(8)]

    constant = ORICAFilter(N_CH, SFREQ, ff_profile="constant")
    cooling = ORICAFilter(N_CH, SFREQ, ff_profile="cooling")
    adaptive = ORICAFilter(N_CH, SFREQ, ff_profile="adaptive")
    for chunk in chunks:
        constant.update(chunk.copy())
        cooling.update(chunk.copy())
        adaptive.update(chunk.copy())

    assert not np.allclose(constant.weights_, cooling.weights_), \
        "constant and cooling profiles should not produce identical weights"
    assert not np.allclose(constant.weights_, adaptive.weights_), \
        "constant and adaptive profiles should not produce identical weights"
    assert not np.allclose(cooling.weights_, adaptive.weights_), \
        "cooling and adaptive profiles should not produce identical weights"


def test_adaptive_profile_does_not_freeze_at_identity():
    """ff_profile="adaptive" must actually adapt: _lambda_k is seeded from
    lambda_0 (not 0), since _gen_adaptive_ff is multiplicative in that seed
    and a 0 seed is an absorbing fixed point — lambda would stay exactly 0
    forever, freezing weights_ at the identity and dividing by zero in
    _dynamic_whitening's Q computation."""
    orica = ORICAFilter(N_CH, SFREQ, ff_profile="adaptive")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(8):
            orica.update(RNG.standard_normal((N_CH, CHUNK)))

    assert np.all(np.isfinite(orica.weights_))
    assert not np.allclose(orica.weights_, np.eye(N_CH)), \
        "adaptive profile should not stay frozen at the identity matrix"


# ── Cycle 6: wrong channel count raises ───────────────────────────────────

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

    # params matching testScript.m: online whitening, block=8, cooling, localstat=Inf.
    # ff_profile="cooling": this test validates the cooling profile itself
    # (stationary-data assumption) against the MATLAB reference, not the
    # constant-lambda real-time default.
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

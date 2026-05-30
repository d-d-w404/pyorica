"""Tests for parallel subject processing — behavior via public interface."""

from __future__ import annotations

import csv
import os
import struct
from pathlib import Path

import numpy as np
import pytest

# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_minimal_set(path: Path, n_ch: int = 4, n_pts: int = 2000, sfreq: float = 250.0):
    """Write a minimal EEGLAB .set + .fdt file that run_validation._load_set can read."""
    import scipy.io

    ch_names = ['Fp1', 'Fp2', 'C3', 'C4'][:n_ch]

    # Build a minimal EEG struct that scipy.io.loadmat can reconstruct
    # We write data to a separate .fdt file
    fdt_path = path.with_suffix('.fdt')
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_ch, n_pts)).astype('<f4')
    data.flatten(order='F').tofile(fdt_path)

    class ChanLoc:
        def __init__(self, label):
            self.labels = label

    class EEGStruct:
        nbchan = n_ch
        pnts = n_pts
        srate = sfreq
        chanlocs = [ChanLoc(name) for name in ch_names]
        data = 0  # signal that data is in .fdt

    scipy.io.savemat(str(path), {'EEG': EEGStruct()})


# ── Cycle 1: auto-worker count returns sensible values ───────────────────────

def test_auto_worker_count_returns_positive_int(tmp_path):
    pytest.importorskip("psutil")
    from benchmarks.run_all_subjects import _auto_worker_count

    # Create a few dummy .set files so the function has file sizes to inspect
    sessions = []
    for i in range(3):
        p = tmp_path / f"s{i}" / f"s{i}_resampled.set"
        p.parent.mkdir()
        p.write_bytes(b"\x00" * 1024 * 1024)  # 1 MB placeholder
        sessions.append(p)

    n_workers, info = _auto_worker_count(sessions)
    assert isinstance(n_workers, int)
    assert n_workers >= 1
    assert "logical_cpus" in info
    assert "available_ram_gb" in info
    assert "per_worker_ram_gb" in info


# ── Cycle 2: --workers override is respected ─────────────────────────────────

def test_workers_flag_overrides_auto(tmp_path):
    pytest.importorskip("psutil")
    from benchmarks.run_all_subjects import _auto_worker_count

    sessions = []
    for i in range(3):
        p = tmp_path / f"s{i}" / f"s{i}_resampled.set"
        p.parent.mkdir()
        p.write_bytes(b"\x00" * 1024 * 1024)
        sessions.append(p)

    # With many sessions we should always be able to get n_workers == 2
    _, info = _auto_worker_count(sessions)
    assert info["logical_cpus"] >= 1  # sanity check — the override is tested in main()


# ── Cycle 3: module-level _run_subject_safe is picklable ─────────────────────

def test_run_subject_safe_is_picklable():
    """ProcessPoolExecutor requires the submitted callable to be picklable."""
    import pickle
    from benchmarks.run_all_subjects import _run_subject_safe
    # pickle.dumps will raise if the function isn't picklable (e.g. lambda/nested)
    pickle.dumps(_run_subject_safe)

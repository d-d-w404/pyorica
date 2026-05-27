"""Tests for cross-session aggregation logic."""

import csv
import math
import numpy as np
import pytest
from pathlib import Path


IC_LABELS = ["brain", "muscle", "eye", "heart", "line noise", "channel noise", "other"]

FIELDNAMES = ["ic", "label", "ms_iir", "ms_asr", "ms_orica", "pct_asr", "pct_orica"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _make_subject_csv(path: Path, class_pcts: dict[str, tuple[float, float]]) -> None:
    """Write a synthetic per-subject CSV.

    class_pcts maps label → (pct_asr, pct_orica) for one IC of that class.
    """
    rows = []
    for ic_idx, (label, (pct_asr, pct_orica)) in enumerate(class_pcts.items()):
        rows.append({
            "ic": ic_idx,
            "label": label,
            "ms_iir": 1.0,
            "ms_asr": 1.0 - pct_asr / 100,
            "ms_orica": 1.0 - pct_orica / 100,
            "pct_asr": pct_asr,
            "pct_orica": pct_orica,
        })
    _write_csv(path, rows)


# ── Cycle 1: cross-subject mean is computed correctly ────────────────────

def test_cross_subject_mean_is_correct(tmp_path):
    from benchmarks.aggregate_results import aggregate

    # Three subjects, one brain IC each with known pct_asr values
    _make_subject_csv(tmp_path / "s1_ic_source_energy.csv", {"brain": (10.0, 20.0)})
    _make_subject_csv(tmp_path / "s2_ic_source_energy.csv", {"brain": (20.0, 30.0)})
    _make_subject_csv(tmp_path / "s3_ic_source_energy.csv", {"brain": (30.0, 40.0)})

    summary = aggregate(tmp_path)

    brain_row = next(r for r in summary if r["class"] == "brain")
    assert math.isclose(brain_row["mean_pct_asr"], 20.0, abs_tol=1e-6)
    assert math.isclose(brain_row["mean_pct_orica"], 30.0, abs_tol=1e-6)


# ── Cycle 2: all 7 classes always appear in output ───────────────────────

def test_all_seven_classes_always_present(tmp_path):
    from benchmarks.aggregate_results import aggregate

    # Only brain ICs in all subjects
    _make_subject_csv(tmp_path / "s1_ic_source_energy.csv", {"brain": (10.0, 20.0)})

    summary = aggregate(tmp_path)

    classes_in_output = {r["class"] for r in summary}
    assert classes_in_output == set(IC_LABELS)


# ── Cycle 3: within-subject median across multiple ICs of same class ─────

def test_within_subject_median_aggregation(tmp_path):
    from benchmarks.aggregate_results import aggregate

    # Subject 1: two muscle ICs with pct_asr 10 and 30 → median = 20
    rows = [
        {"ic": 0, "label": "muscle", "ms_iir": 1.0, "ms_asr": 0.9,
         "ms_orica": 0.8, "pct_asr": 10.0, "pct_orica": 20.0},
        {"ic": 1, "label": "muscle", "ms_iir": 1.0, "ms_asr": 0.7,
         "ms_orica": 0.6, "pct_asr": 30.0, "pct_orica": 40.0},
    ]
    _write_csv(tmp_path / "s1_ic_source_energy.csv", rows)

    summary = aggregate(tmp_path)

    muscle_row = next(r for r in summary if r["class"] == "muscle")
    assert math.isclose(muscle_row["mean_pct_asr"], 20.0, abs_tol=1e-6)
    assert math.isclose(muscle_row["mean_pct_orica"], 30.0, abs_tol=1e-6)


# ── Cycle 4: SD across subjects is correct ───────────────────────────────

def test_cross_subject_sd_is_correct(tmp_path):
    from benchmarks.aggregate_results import aggregate

    _make_subject_csv(tmp_path / "s1_ic_source_energy.csv", {"brain": (10.0, 0.0)})
    _make_subject_csv(tmp_path / "s2_ic_source_energy.csv", {"brain": (30.0, 0.0)})

    summary = aggregate(tmp_path)

    brain_row = next(r for r in summary if r["class"] == "brain")
    expected_sd = float(np.std([10.0, 30.0], ddof=0))
    assert math.isclose(brain_row["sd_pct_asr"], expected_sd, abs_tol=1e-6)


# ── Cycle 5: n_subjects counts only subjects that have that class ─────────

def test_n_subjects_counts_only_subjects_with_class(tmp_path):
    from benchmarks.aggregate_results import aggregate

    # Only s1 has eye ICs
    _make_subject_csv(tmp_path / "s1_ic_source_energy.csv",
                      {"brain": (10.0, 20.0), "eye": (5.0, 50.0)})
    _make_subject_csv(tmp_path / "s2_ic_source_energy.csv", {"brain": (20.0, 30.0)})

    summary = aggregate(tmp_path)

    eye_row = next(r for r in summary if r["class"] == "eye")
    assert eye_row["n_subjects"] == 1

    brain_row = next(r for r in summary if r["class"] == "brain")
    assert brain_row["n_subjects"] == 2

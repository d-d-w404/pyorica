"""Cross-session aggregation of per-subject IC source energy CSVs.

Reads all ``*_ic_source_energy.csv`` files in a run directory, computes
within-subject median per ICLabel class, then cross-subject mean ± SD, and
produces two bar charts (ASR vs IIR, ORICA vs IIR) across all 7 ICLabel classes.

Usage
-----
    python benchmarks/aggregate_results.py --run-dir benchmarks/results/run_20260527_120000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


IC_LABELS = ["brain", "muscle", "eye", "heart", "line noise", "channel noise", "other"]


def aggregate(run_dir: Path) -> list[dict]:
    """Compute cross-subject mean ± SD of pct reduction per ICLabel class.

    Parameters
    ----------
    run_dir : Path
        Directory containing ``*_ic_source_energy.csv`` files.

    Returns
    -------
    list[dict]
        One dict per ICLabel class (all 7 always present) with keys:
        class, mean_pct_asr, sd_pct_asr, mean_pct_orica, sd_pct_orica, n_subjects
    """
    csv_files = sorted(run_dir.glob("*_ic_source_energy.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_ic_source_energy.csv files found in {run_dir}")

    # per_subject[label] → list of per-subject medians
    per_subject: dict[str, list[float]] = {label: [] for label in IC_LABELS}
    per_subject_orica: dict[str, list[float]] = {label: [] for label in IC_LABELS}

    for csv_path in csv_files:
        # group ICs by label for this subject
        by_label: dict[str, list[float]] = {label: [] for label in IC_LABELS}
        by_label_orica: dict[str, list[float]] = {label: [] for label in IC_LABELS}

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"]
                if label not in by_label:
                    continue
                try:
                    by_label[label].append(float(row["pct_asr"]))
                    by_label_orica[label].append(float(row["pct_orica"]))
                except (ValueError, KeyError):
                    continue

        # within-subject: median across ICs of same class
        for label in IC_LABELS:
            if by_label[label]:
                per_subject[label].append(float(np.median(by_label[label])))
                per_subject_orica[label].append(float(np.median(by_label_orica[label])))

    # cross-subject: mean ± SD (population SD, ddof=0)
    summary = []
    for label in IC_LABELS:
        vals_asr = np.array(per_subject[label], dtype=float)
        vals_orica = np.array(per_subject_orica[label], dtype=float)
        n = len(vals_asr)
        summary.append({
            "class": label,
            "mean_pct_asr": float(np.mean(vals_asr)) if n > 0 else float("nan"),
            "sd_pct_asr": float(np.std(vals_asr)) if n > 0 else float("nan"),
            "mean_pct_orica": float(np.mean(vals_orica)) if n > 0 else float("nan"),
            "sd_pct_orica": float(np.std(vals_orica)) if n > 0 else float("nan"),
            "n_subjects": n,
        })
    return summary


def plot_summary(summary: list[dict], out_path: Path) -> None:
    """Save two side-by-side bar charts (ASR vs IIR, ORICA vs IIR)."""
    import matplotlib.pyplot as plt

    labels = [r["class"] for r in summary]
    means_asr = np.array([r["mean_pct_asr"] for r in summary])
    sds_asr = np.array([r["sd_pct_asr"] for r in summary])
    means_orica = np.array([r["mean_pct_orica"] for r in summary])
    sds_orica = np.array([r["sd_pct_orica"] for r in summary])

    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, means, sds, title in [
        (ax1, means_asr, sds_asr, "ASR vs IIR"),
        (ax2, means_orica, sds_orica, "ORICA vs IIR"),
    ]:
        bars = ax.bar(x, means, yerr=sds, capsize=4, color="steelblue", alpha=0.8,
                      error_kw={"elinewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("% energy reduction vs IIR")
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary_csv(summary: list[dict], out_path: Path) -> None:
    fieldnames = ["class", "mean_pct_asr", "sd_pct_asr",
                  "mean_pct_orica", "sd_pct_orica", "n_subjects"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-subject IC source energy CSVs into cross-session results."
    )
    parser.add_argument(
        "--run-dir", required=True, metavar="DIR",
        help="Directory containing *_ic_source_energy.csv files.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        import sys
        print(f"ERROR: {run_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading CSVs from {run_dir}...")
    summary = aggregate(run_dir)

    n_subjects = max(r["n_subjects"] for r in summary)
    print(f"Aggregated {n_subjects} subject(s) across {len(summary)} ICLabel classes.")

    png_path = run_dir / "cross_session_results.png"
    plot_summary(summary, png_path)
    print(f"Saved plot → {png_path}")

    csv_path = run_dir / "cross_session_summary.csv"
    save_summary_csv(summary, csv_path)
    print(f"Saved summary → {csv_path}")


if __name__ == "__main__":
    main()

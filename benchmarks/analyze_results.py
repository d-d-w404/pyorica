"""Comprehensive analysis of cross-subject IC source energy results.

Reads all ``*_ic_source_energy.csv`` files in a run directory and produces:

When ``*_stages.npz`` files are present (saved by ``run_validation.py``), each file
holds ``raw``, ``iir``, ``asr``, and ``orica`` stage arrays (channels × samples).
Per-subject CSVs are **recomputed** first with the calibration lead-in excluded
from MS / pct stats (``asr_calibration_seconds`` from ``config.yaml``, default 120 s).

1. ``analysis_overview.png``          — 6-panel overview figure
2. ``analysis_stage_comparison.png``  — standalone panel (f): ASR vs ORICA with grid lines
3. ``analysis_all_ics.png``           — per-IC ASR vs ORICA (one file per subject if multiple)
4. ``analysis_summary.csv``           — per-class stats (n, mean, std, median, IQR, min, max)
3. ``analysis_per_subject.csv`` — per-subject median pct_orica matrix
4. ``analysis_ic_counts.csv``   — per-subject IC count per class

Usage
-----
    python benchmarks/analyze_results.py --run-dir benchmarks/result/all/iclabel_1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


IC_LABELS = [
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
]

LABEL_COLORS = {
    "brain":           "#2ca02c",
    "muscle artifact": "#d62728",
    "eye blink":       "#1f77b4",
    "heart beat":      "#e377c2",
    "line noise":      "#ff7f0e",
    "channel noise":   "#9467bd",
    "other":           "#7f7f7f",
}

ARTIFACT_LABELS = {"muscle artifact", "eye blink", "heart beat",
                   "line noise", "channel noise"}

# Legacy ORICA per-IC sort: artifact first, then other, then brain.
_IC_ORDER_RANK = {"artifact": 0, "other": 1, "brain": 2}

# Bar colours aligned with ica_source_energy_analysis_correctly.py
COLOR_ASR = "#ff7f0e"
COLOR_ORICA = "#9467bd"

PCT_REFERENCE_LEVELS = (20, 40, 60, 80, 100)


def _ic_group(label: str) -> str:
    if label in ARTIFACT_LABELS:
        return "artifact"
    if label == "brain":
        return "brain"
    return "other"


def _sort_ics_legacy_order(df: pd.DataFrame) -> pd.DataFrame:
    """Order ICs: artifacts first, then other, then brain (within subject)."""
    sub = df.copy()
    sub["_order_rank"] = sub["label"].map(
        lambda lb: _IC_ORDER_RANK.get(_ic_group(str(lb)), 1)
    )
    return (
        sub.sort_values(["subject", "_order_rank", "ic"])
        .drop(columns="_order_rank")
        .reset_index(drop=True)
    )


def _add_pct_reference_lines(ax) -> None:
    """Horizontal dashed guides at 20/40/60/80/100 for pct_orica comparison."""
    for y in PCT_REFERENCE_LEVELS:
        ax.axhline(y, color="gray", lw=0.7, ls="--", zorder=0)


def plot_orica_reduction(stats: pd.DataFrame, out_path: Path | None = None,
                         ax=None, *, panel_label: str = "(a) "):
    """Bar chart: mean ± SD pct_orica per ICLabel class (overview panel a).

    When *ax* is None, writes a standalone figure to *out_path*.
    """
    classes = stats["class"].tolist()
    means = stats["mean"].to_numpy()
    stds = stats["std"].to_numpy()
    medians = stats["median"].to_numpy()
    n_ics_arr = stats["n_ics"].to_numpy()
    colors = [LABEL_COLORS[c] for c in classes]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure

    x = np.arange(len(classes))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors,
           edgecolor="black", linewidth=0.5, zorder=3,
           error_kw={"linewidth": 1, "ecolor": "black"})
    ax.scatter(x, medians, marker="D", s=40, c="white",
               edgecolors="black", zorder=5, label="median")
    for i, n in enumerate(n_ics_arr):
        ax.text(i, 5, f"n={n}", ha="center", fontsize=9)
    ax.axhline(100, color="gray", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("pct_orica  (% energy vs IIR)")
    ax.set_ylim(0, 115)
    ax.set_title(f"{panel_label}ORICA reduction · mean ± SD per class  (♦ = median)")
    ax.legend(loc="lower right", fontsize=8)

    if standalone:
        if out_path is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_stage_comparison(df: pd.DataFrame, stats: pd.DataFrame,
                          out_path: Path | None = None, ax=None,
                          *, panel_label: str = "(f) "):
    """Grouped bars: ASR vs IIR and ORICA vs IIR per class (overview panel f).

    When *ax* is None, writes a standalone figure to *out_path*.
    """
    classes = stats["class"].tolist()
    means = stats["mean"].to_numpy()
    stds = stats["std"].to_numpy()
    asr_stats = class_stats(df, value_col="pct_asr")
    means_asr = asr_stats["mean"].to_numpy()
    stds_asr = asr_stats["std"].to_numpy()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure

    width = 0.38
    xc = np.arange(len(classes))
    _add_pct_reference_lines(ax)
    ax.bar(xc - width / 2, means_asr, width=width, yerr=stds_asr,
           capsize=3, color=COLOR_ASR, alpha=0.85, zorder=3,
           edgecolor="black", linewidth=0.4, label="ASR vs IIR")
    ax.bar(xc + width / 2, means, width=width, yerr=stds,
           capsize=3, color=COLOR_ORICA, alpha=0.85, zorder=3,
           edgecolor="black", linewidth=0.4, label="ORICA vs IIR")
    medians = stats["median"].to_numpy()
    medians_asr = asr_stats["median"].to_numpy()
    ax.scatter(xc - width / 2, medians_asr, marker="D", s=28, c="white",
               edgecolors="black", zorder=5)
    ax.scatter(xc + width / 2, medians, marker="D", s=28, c="white",
               edgecolors="black", zorder=5, label="median")
    ax.set_xticks(xc)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("% energy vs IIR  (mean ± SD; ♦ = median)")
    ax.set_ylim(0, 115)
    ax.set_title(f"{panel_label}Stage comparison · ASR vs ORICA per class (mean)")
    ax.legend(loc="lower right", fontsize=8)

    if standalone:
        if out_path is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _ic_xlabels(df: pd.DataFrame, multi_subject: bool) -> list[str]:
    """Build x-axis tick labels: IC index + short label (+ subject if needed)."""
    short = {
        "muscle artifact": "muscle",
        "eye blink": "eye",
        "heart beat": "heart",
        "line noise": "line",
        "channel noise": "ch_noise",
    }
    labels = []
    for _, row in df.iterrows():
        tag = short.get(row["label"], row["label"])
        if multi_subject:
            labels.append(f"{row['subject']}\nIC{int(row['ic'])}\n{tag}")
        else:
            labels.append(f"IC{int(row['ic'])}\n{tag}")
    return labels


def plot_all_ics(df: pd.DataFrame, out_path: Path,
                  subject: str | None = None) -> None:
    """Grouped bars: ASR vs IIR and ORICA vs IIR for every IC.

    One bar pair per offline-ICA component, coloured by ICLabel class on the
    x-axis.  Saves to *out_path* (caller picks per-subject filename if needed).
    """
    sub = df if subject is None else df[df["subject"] == subject].copy()
    if sub.empty:
        return

    sub = _sort_ics_legacy_order(sub)
    multi_subject = subject is None and sub["subject"].nunique() > 1
    n = len(sub)
    fig_w = max(12.0, n * 0.48)
    fig, ax = plt.subplots(figsize=(fig_w, 6), constrained_layout=True)

    width = 0.38
    x = np.arange(n)
    _add_pct_reference_lines(ax)

    edge_colors = [LABEL_COLORS.get(lbl, "#7f7f7f") for lbl in sub["label"]]
    ax.bar(x - width / 2, sub["pct_asr"], width=width,
           color=COLOR_ASR, alpha=0.85, zorder=3,
           edgecolor="black", linewidth=0.4, label="ASR vs IIR")
    ax.bar(x + width / 2, sub["pct_orica"], width=width,
           color=COLOR_ORICA, alpha=0.85, zorder=3,
           edgecolor=edge_colors, linewidth=1.8,
           label="ORICA vs IIR")

    for tick, color in zip(ax.get_xticklabels(), edge_colors):
        tick.set_color(color)

    ax.set_xticks(x)
    ax.set_xticklabels(_ic_xlabels(sub, multi_subject), rotation=45, ha="right",
                       fontsize=8)
    ax.set_ylabel("% energy vs IIR")
    ax.set_ylim(0, 115)

    title_subj = subject or (sub["subject"].iloc[0] if not multi_subject else "all subjects")
    ax.set_title(
        f"Per-IC vs IIR (ordered: artifact → other → brain) · {title_subj}, n={n} ICs"
    )
    ax.axhline(100, color="k", ls="--", lw=1.0, alpha=0.65)

    # class colour legend (deduplicated)
    seen = set()
    class_handles = []
    for lbl in IC_LABELS:
        if lbl in seen or lbl not in sub["label"].values:
            continue
        seen.add(lbl)
        class_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=LABEL_COLORS[lbl],
                          edgecolor="black", linewidth=0.5, label=lbl)
        )
    leg1 = ax.legend(loc="upper right", fontsize=8)
    ax.add_artist(leg1)
    if class_handles:
        ax.legend(handles=class_handles, loc="lower right", fontsize=7,
                  title="ICLabel", framealpha=0.9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_ics_for_run(df: pd.DataFrame, run_dir: Path) -> list[Path]:
    """Write per-IC plots — one combined file, or one per subject."""
    paths: list[Path] = []
    subjects = sorted(df["subject"].unique())
    if len(subjects) == 1:
        p = run_dir / "analysis_all_ics.png"
        plot_all_ics(df, p, subject=subjects[0])
        paths.append(p)
    else:
        for subj in subjects:
            p = run_dir / f"analysis_all_ics_{subj}.png"
            plot_all_ics(df, p, subject=subj)
            paths.append(p)
        combined = run_dir / "analysis_all_ics_all_subjects.png"
        plot_all_ics(df, combined)
        paths.append(combined)
    return paths


IC_CSV_FIELDS = ["ic", "label", "ms_iir", "ms_asr", "ms_orica", "pct_asr", "pct_orica"]


def _load_exclude_lead_seconds(run_dir: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        from pyorica.config import PipelineConfig
        return float(PipelineConfig.from_yaml(config_path).asr_calibration_seconds)
    return 120.0


def _write_ic_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=IC_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def backfill_missing_stages_npz(
    run_dir: Path,
    data_root: Path,
    ica_cache_dir: Path | None,
) -> int:
    """Re-run pipeline for subjects that have CSV but no ``*_stages.npz``.

    Old benchmark runs only persisted per-IC summary CSVs; stage time series
    (IIR/ASR/ORICA) lived in memory and were discarded. This one-time backfill
    recovers them from the original ``.set`` / ``.fdt`` files.
    """
    from pyorica.config import PipelineConfig

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from benchmarks.run_validation import run_subject

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        print(f"ERROR: {config_path} not found (needed for backfill).", file=sys.stderr)
        sys.exit(1)
    config = PipelineConfig.from_yaml(config_path)

    n_backfilled = 0
    for csv_path in sorted(run_dir.glob("*_ic_source_energy.csv")):
        subject = csv_path.stem.replace("_ic_source_energy", "")
        npz_path = run_dir / f"{subject}_stages.npz"
        if npz_path.is_file():
            continue
        set_path = data_root / subject / f"{subject}_resampled.set"
        if not set_path.is_file():
            print(
                f"WARNING: skip {subject} — {set_path} not found under {data_root}",
                file=sys.stderr,
            )
            continue
        print(
            f"Backfilling {npz_path.name} for {subject} "
            f"(re-running pipeline to recover stage arrays)..."
        )
        run_subject(set_path, config, run_dir, ica_cache_dir)
        n_backfilled += 1
    return n_backfilled


def refresh_ic_source_csvs(
    run_dir: Path,
    exclude_lead_seconds: float,
    ica_cache_dir: Path | None = None,
    *,
    allow_stale_csv: bool = False,
) -> int:
    """Recompute ``*_ic_source_energy.csv`` from stage NPZ, skipping calib lead-in.

    Returns the number of subjects refreshed. If no ``*_stages.npz`` exist and
    *allow_stale_csv* is False, exits with code 1.
    """
    from pyorica.eval.ica_analysis import ic_source_energy

    npz_files = sorted(run_dir.glob("*_stages.npz"))
    if not npz_files:
        msg = (
            f"No *_stages.npz in {run_dir} — cannot exclude the first "
            f"{exclude_lead_seconds:.0f}s calibration window.\n"
            f"The existing CSVs are per-IC summary numbers (no time axis); "
            f"excluding a lead-in requires the saved IIR/ASR/ORICA stage arrays.\n"
            f"Either pass --rebuild-stages-missing (re-runs pipeline once per "
            f"subject from PYORICA_NCTU_DATA to write {{subject}}_stages.npz), "
            f"or pass --allow-stale-csv to plot the old full-recording stats."
        )
        if allow_stale_csv:
            print(f"WARNING: {msg}", file=sys.stderr)
            return 0
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    n_refreshed = 0
    for npz_path in npz_files:
        subject = npz_path.stem.replace("_stages", "")
        z = np.load(npz_path, allow_pickle=True)
        ch_names = [str(x) for x in np.asarray(z["ch_names"]).ravel()]
        rows = ic_source_energy(
            np.asarray(z["iir"], dtype=np.float64),
            np.asarray(z["asr"], dtype=np.float64),
            np.asarray(z["orica"], dtype=np.float64),
            ch_names,
            float(z["sfreq"]),
            cache_dir=ica_cache_dir,
            subject=subject,
            exclude_lead_seconds=exclude_lead_seconds,
        )
        csv_path = run_dir / f"{subject}_ic_source_energy.csv"
        _write_ic_csv(csv_path, rows)
        print(
            f"Refreshed {csv_path.name} "
            f"(MS stats exclude first {exclude_lead_seconds:.0f}s)"
        )
        n_refreshed += 1
    return n_refreshed


def load_all(run_dir: Path) -> pd.DataFrame:
    csv_files = sorted(run_dir.glob("*_ic_source_energy.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_ic_source_energy.csv files in {run_dir}")
    frames = []
    for path in csv_files:
        subject = path.stem.replace("_ic_source_energy", "")
        df = pd.read_csv(path)
        df["subject"] = subject
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def class_stats(df: pd.DataFrame, value_col: str = "pct_orica") -> pd.DataFrame:
    rows = []
    for label in IC_LABELS:
        sub = df[df["label"] == label]
        n_ics = len(sub)
        n_subj = sub["subject"].nunique()
        if n_ics == 0:
            rows.append({
                "class": label, "n_ics": 0, "n_subjects": 0,
                "mean": np.nan, "std": np.nan, "median": np.nan,
                "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan,
            })
            continue
        vals = sub[value_col].to_numpy()
        rows.append({
            "class":      label,
            "n_ics":      int(n_ics),
            "n_subjects": int(n_subj),
            "mean":       float(np.mean(vals)),
            "std":        float(np.std(vals)),
            "median":     float(np.median(vals)),
            "q25":        float(np.percentile(vals, 25)),
            "q75":        float(np.percentile(vals, 75)),
            "min":        float(np.min(vals)),
            "max":        float(np.max(vals)),
        })
    return pd.DataFrame(rows)


def per_subject_median(df: pd.DataFrame,
                       value_col: str = "pct_orica") -> pd.DataFrame:
    pivot = (df.groupby(["subject", "label"])[value_col]
               .median().unstack("label"))
    return pivot.reindex(columns=IC_LABELS)


def per_subject_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (df.groupby(["subject", "label"]).size().unstack("label")
                .fillna(0).astype(int))
    return counts.reindex(columns=IC_LABELS, fill_value=0)


def plot_overview(df: pd.DataFrame, stats: pd.DataFrame,
                  per_subj: pd.DataFrame, counts: pd.DataFrame,
                  out_path: Path) -> None:
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    classes = stats["class"].tolist()
    colors = [LABEL_COLORS[c] for c in classes]

    # (a) Mean ± SD bars per class (pct_orica)
    ax = fig.add_subplot(gs[0, 0])
    plot_orica_reduction(stats, ax=ax)

    # (b) Box plot pct_orica per class
    ax = fig.add_subplot(gs[0, 1])
    box_data = [df.loc[df["label"] == c, "pct_orica"].to_numpy()
                for c in classes]
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                    widths=0.6, medianprops={"color": "black", "lw": 1.5},
                    flierprops={"marker": ".", "ms": 3, "alpha": 0.4})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(100, color="gray", lw=0.7, ls="--")
    ax.set_xticks(np.arange(1, len(classes) + 1))
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("pct_orica  (% energy vs IIR)")
    ax.set_title("(b) ORICA reduction · per-IC distribution per class")

    # (c) Heatmap subjects × classes (median pct_orica)
    ax = fig.add_subplot(gs[1, 0])
    M = per_subj.to_numpy()
    cmap = LinearSegmentedColormap.from_list(
        "rdyngn", ["#2ca02c", "#ffeb3b", "#d62728"])
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(IC_LABELS)))
    ax.set_xticklabels(IC_LABELS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(per_subj.index)))
    ax.set_yticklabels(per_subj.index, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("median pct_orica", fontsize=9)
    ax.set_title("(c) Per-subject median pct_orica  (green = more reduction)")

    # (d) Stacked bar IC count composition per subject
    ax = fig.add_subplot(gs[1, 1])
    subjects = counts.index.tolist()
    bottom = np.zeros(len(subjects))
    x = np.arange(len(subjects))
    for label in IC_LABELS:
        vals = counts[label].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=LABEL_COLORS[label],
               edgecolor="white", linewidth=0.3, label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=70, fontsize=7)
    ax.set_ylabel("IC count")
    ax.set_title("(d) ICLabel composition per subject")
    ax.legend(loc="upper right", fontsize=7, ncol=2,
              framealpha=0.9, bbox_to_anchor=(1.0, 1.02))

    # (e) Histogram pct_orica · brain vs artifact vs other
    ax = fig.add_subplot(gs[2, 0])
    bins = np.linspace(0, 130, 40)
    brain_vals = df.loc[df["label"] == "brain", "pct_orica"].to_numpy()
    art_vals   = df.loc[df["label"].isin(ARTIFACT_LABELS), "pct_orica"].to_numpy()
    other_vals = df.loc[df["label"] == "other", "pct_orica"].to_numpy()
    ax.hist([brain_vals, art_vals, other_vals], bins=bins,
            stacked=False, alpha=0.7,
            color=[LABEL_COLORS["brain"], LABEL_COLORS["muscle artifact"],
                   LABEL_COLORS["other"]],
            label=[f"brain (n={len(brain_vals)})",
                   f"artifact (n={len(art_vals)})",
                   f"other (n={len(other_vals)})"])
    ax.axvline(100, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("pct_orica  (% energy vs IIR)")
    ax.set_ylabel("IC count")
    ax.set_title("(e) Distribution of pct_orica across all ICs")
    ax.legend(loc="upper left", fontsize=8)

    # (f) Stage-by-stage comparison: ASR vs ORICA per class
    ax = fig.add_subplot(gs[2, 1])
    plot_stage_comparison(df, stats, ax=ax)

    fig.suptitle(
        f"pyorica cross-session analysis  ·  "
        f"{per_subj.shape[0]} subjects · {len(df)} ICs",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of pyorica benchmark results."
    )
    parser.add_argument("--run-dir", required=True, metavar="DIR",
                        help="Directory containing *_ic_source_energy.csv files.")
    parser.add_argument(
        "--exclude-lead-seconds", type=float, default=None, metavar="SEC",
        help="Skip this many seconds at the start of MS / pct stats "
             "(default: asr_calibration_seconds from config.yaml, else 120).",
    )
    parser.add_argument(
        "--ica-cache-dir", metavar="PATH", default=None,
        help="Optional ICA cache directory (same as run_all_subjects --ica-cache-dir).",
    )
    parser.add_argument(
        "--allow-stale-csv", action="store_true",
        help="If no *_stages.npz exist, use existing CSVs without refreshing "
             "(stats may include the calibration lead-in).",
    )
    parser.add_argument(
        "--rebuild-stages-missing", action="store_true",
        help="For subjects with CSV but no *_stages.npz, re-run the pipeline "
             "from the NCTU dataset to write stage NPZ (one-time backfill for "
             "runs done before stages were saved). Requires PYORICA_NCTU_DATA.",
    )
    parser.add_argument(
        "--data-root", metavar="PATH", default=None,
        help="NCTU dataset root (default: PYORICA_NCTU_DATA env var).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    exclude_s = _load_exclude_lead_seconds(run_dir, args.exclude_lead_seconds)
    ica_cache = Path(args.ica_cache_dir) if args.ica_cache_dir else None
    print(f"Run dir: {run_dir}")
    print(f"Excluding first {exclude_s:.0f}s from MS / pct stats when stages NPZ exist.")

    if args.rebuild_stages_missing:
        data_root = Path(args.data_root or os.environ.get("PYORICA_NCTU_DATA", ""))
        if not data_root.is_dir():
            print(
                "ERROR: --rebuild-stages-missing needs a dataset root "
                "(--data-root or PYORICA_NCTU_DATA).",
                file=sys.stderr,
            )
            sys.exit(1)
        n_bf = backfill_missing_stages_npz(run_dir, data_root, ica_cache)
        print(f"Backfilled {n_bf} subject stage NPZ file(s).\n")

    refresh_ic_source_csvs(
        run_dir, exclude_s, ica_cache_dir=ica_cache,
        allow_stale_csv=args.allow_stale_csv,
    )

    print(f"\nReading CSVs from {run_dir}...")
    df = load_all(run_dir)
    n_subjects = df["subject"].nunique()
    n_ics = len(df)
    print(f"Loaded {n_subjects} subjects | {n_ics} ICs total.")

    stats = class_stats(df)
    per_subj = per_subject_median(df)
    counts = per_subject_counts(df)

    stats_path = run_dir / "analysis_summary.csv"
    stats.to_csv(stats_path, index=False, float_format="%.3f")
    print(f"Saved class stats   -> {stats_path}")

    per_subj_path = run_dir / "analysis_per_subject.csv"
    per_subj.to_csv(per_subj_path, float_format="%.3f")
    print(f"Saved per-subject   -> {per_subj_path}")

    counts_path = run_dir / "analysis_ic_counts.csv"
    counts.to_csv(counts_path)
    print(f"Saved IC counts     -> {counts_path}")

    fig_path = run_dir / "analysis_overview.png"
    plot_overview(df, stats, per_subj, counts, fig_path)
    print(f"Saved overview plot -> {fig_path}")

    stage_path = run_dir / "analysis_stage_comparison.png"
    plot_stage_comparison(df, stats, out_path=stage_path)
    print(f"Saved stage comparison plot -> {stage_path}")

    for ic_path in plot_all_ics_for_run(df, run_dir):
        print(f"Saved per-IC plot         -> {ic_path}")

    print("\nClass summary (ORICA vs IIR, mean ± std; median in analysis_summary.csv):")
    print(stats[["class", "n_ics", "mean", "std", "median"]].to_string(index=False))


if __name__ == "__main__":
    main()

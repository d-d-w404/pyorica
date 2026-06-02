"""Visualization utilities for per-session pipeline analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pyorica.pipeline.classify import LABEL_NAMES

# MNE-icalabel color convention (RGB, 0–1)
_LABEL_COLORS: dict[str, tuple[float, float, float]] = {
    'brain':      (0.149, 0.588, 0.898),
    'muscle':     (0.957, 0.263, 0.212),
    'eog':        (0.298, 0.686, 0.314),
    'ecg':        (0.914, 0.118, 0.388),
    'line_noise': (1.000, 0.922, 0.231),
    'ch_noise':   (1.000, 0.596, 0.000),
    'other':      (0.620, 0.620, 0.620),
}

# Legend order used for IC sorting (brain leftmost, other rightmost)
_LABEL_ORDER = {label: i for i, label in enumerate(LABEL_NAMES)}


def plot_ic_class_timeline(
    snapshots: list,
    classify_interval_s: float,
    out_path: str | Path,
) -> None:
    """Save an IC-class timeline plot for one session.

    ICs are sorted left-to-right by their weighted mean class position
    across all snapshots (brain → other). Each cell shows the top-1
    ICLabel class color. ICs classified as artifact at that snapshot are
    marked with a white ×. The x-axis labels show the original ORICA IC
    index so columns can be cross-referenced with the CSV.

    Parameters
    ----------
    snapshots:
        List of ``(seq_num, top1_labels, top1_probs, artifact_mask)`` tuples
        collected by ``ICLabelClassifier`` with ``record_snapshots=True``.
    classify_interval_s:
        Seconds between consecutive classification events.
    out_path:
        Destination path for the saved PNG.
    """
    if not snapshots:
        return

    n_snapshots = len(snapshots)
    n_ics = len(snapshots[0][1])

    # Build label-index matrix (n_snapshots × n_ics) and artifact mask matrix
    label_idx = np.zeros((n_snapshots, n_ics), dtype=np.int32)
    removed = np.zeros((n_snapshots, n_ics), dtype=bool)
    for row, (_seq, labels, _probs, mask) in enumerate(snapshots):
        label_idx[row] = [_LABEL_ORDER[lbl] for lbl in labels]
        removed[row] = mask

    # Sort ICs by weighted mean label index across snapshots (brain=0 → other=6)
    weighted_means = label_idx.mean(axis=0)
    sort_order = np.argsort(weighted_means, kind='stable')

    label_idx_sorted = label_idx[:, sort_order]
    removed_sorted = removed[:, sort_order]

    # Build RGB grid: shape (n_snapshots, n_ics, 3)
    rgb = np.ones((n_snapshots, n_ics, 3), dtype=np.float32)
    for row in range(n_snapshots):
        for col in range(n_ics):
            rgb[row, col] = _LABEL_COLORS[LABEL_NAMES[label_idx_sorted[row, col]]]

    fig_w = max(6.0, n_ics * 0.5)
    fig_h = max(4.0, n_snapshots * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(rgb, aspect='auto', interpolation='nearest', origin='upper')

    # White × markers for artifact ICs (decision mask)
    rows_removed, cols_removed = np.where(removed_sorted)
    if rows_removed.size:
        ax.scatter(cols_removed, rows_removed, marker='x', color='white',
                   s=40, linewidths=1.2, zorder=3)

    # X-axis: original ORICA IC index in sorted order
    ax.set_xticks(np.arange(n_ics))
    ax.set_xticklabels([str(sort_order[i]) for i in range(n_ics)], fontsize=12)
    ax.set_xlabel('IC index (sorted by class)', fontsize=14)

    # Y-axis: compressed — ~6 evenly-spaced tick labels
    tick_step = max(1, round(n_snapshots / 6))
    tick_rows = np.arange(0, n_snapshots, tick_step)
    ax.set_yticks(tick_rows)
    ax.set_yticklabels(
        [f"{int(r * classify_interval_s)}s" for r in tick_rows],
        fontsize=12,
    )
    ax.set_ylabel('Time', fontsize=14)

    ax.set_title('IC class timeline', fontsize=16)

    patches = [
        mpatches.Patch(color=_LABEL_COLORS[label], label=label)
        for label in LABEL_NAMES
    ]
    # Extra legend entry for the artifact marker
    patches.append(plt.scatter([], [], marker='x', color='black', s=40,
                               linewidths=1.2, label='removed (online)'))
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left',
              borderaxespad=0, fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

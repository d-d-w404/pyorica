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


def plot_ic_class_timeline(
    snapshots: list,
    classify_interval_s: float,
    out_path: str | Path,
) -> None:
    """Save an IC-class timeline plot for one session.

    Each cell shows the top-1 ICLabel class (color) and its confidence
    (opacity). IC indices are fixed in their original ORICA order across
    all snapshots.

    Parameters
    ----------
    snapshots:
        List of ``(seq_num, top1_labels, top1_probs)`` tuples collected by
        ``ICLabelClassifier`` with ``record_snapshots=True``.
        ``top1_labels`` is a list of str of length ``n_ics``;
        ``top1_probs`` is a float array of the same length.
    classify_interval_s:
        Seconds between consecutive classification events — used to convert
        sequence numbers to wall-clock seconds on the x-axis.
    out_path:
        Destination path for the saved PNG.
    """
    if not snapshots:
        return

    n_snapshots = len(snapshots)
    n_ics = len(snapshots[0][1])

    # Build RGB grid: shape (n_snapshots, n_ics, 3) — time on Y, IC on X
    rgb = np.ones((n_snapshots, n_ics, 3), dtype=np.float32)
    for row, (_seq, labels, _probs) in enumerate(snapshots):
        for col, label in enumerate(labels):
            rgb[row, col] = _LABEL_COLORS[label]

    fig_w = max(6.0, n_ics * 0.5)
    fig_h = max(5.0, n_snapshots * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(rgb, aspect='auto', interpolation='nearest', origin='upper')

    # X-axis: IC index (fixed ORICA order)
    ax.set_xticks(np.arange(n_ics))
    ax.set_xticklabels([str(i) for i in range(n_ics)], fontsize=12)
    ax.set_xlabel('IC index', fontsize=14)

    # Y-axis: time in seconds
    ax.set_yticks(np.arange(n_snapshots))
    ax.set_yticklabels(
        [f"{int(s * classify_interval_s)}s" for s in range(n_snapshots)],
        fontsize=12,
    )
    ax.set_ylabel('Time', fontsize=14)

    ax.set_title('IC class timeline', fontsize=16)

    patches = [
        mpatches.Patch(color=_LABEL_COLORS[label], label=label)
        for label in LABEL_NAMES
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left',
              borderaxespad=0, fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

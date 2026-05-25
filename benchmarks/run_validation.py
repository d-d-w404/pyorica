"""Per-subject IC source energy validation benchmark.

Loads sessions from the NCTU-LKT dataset, runs the pyorica pipeline in verbose
mode, performs offline ICA analysis, and writes a per-subject CSV.

Usage
-----
    export PYORICA_NCTU_DATA=/path/to/dataset_2019_TBME
    python benchmarks/run_validation.py [--subjects s1 s3 ...] [--output-dir results]

Environment
-----------
PYORICA_NCTU_DATA
    Root directory of the NCTU-LKT dataset. Each subject lives at
    ``{root}/s{N}/s{N}_resampled.set``.

Output
------
One CSV per subject at ``{output_dir}/s{N}_ic_source_energy.csv`` with columns:
    ic, label, ms_iir, ms_asr, ms_orica, pct_asr, pct_orica
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


CALIB_SECONDS = 120   # first 2 min used for ASR calibration
# ICLabel's MNE FIR filter (length ~825 at 250 Hz) needs > 825 samples per chunk.
# 1000 samples = 4 s at 250 Hz gives comfortable headroom.
CHUNK_SIZE = 1000


def _find_sessions(root: Path) -> list[Path]:
    sessions = sorted(root.glob("s*/s*_resampled.set"))
    # exclude any _cleanSec variants (they won't match *_resampled.set without _cleanSec,
    # but guard explicitly)
    return [p for p in sessions if "_cleanSec" not in p.name]


def _load_set(path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Load an EEGLAB .set file. Returns (data, sfreq, ch_names)."""
    import scipy.io

    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    EEG = mat["EEG"]
    n_ch = int(EEG.nbchan)
    n_pts = int(EEG.pnts)
    sfreq = float(EEG.srate)

    ch_names = [str(c.labels) for c in EEG.chanlocs]

    fdt = path.with_suffix(".fdt")
    if fdt.exists():
        data = np.fromfile(fdt, dtype="<f4", count=n_ch * n_pts)
        data = data.reshape((n_ch, n_pts), order="F").astype(np.float64)
    else:
        data = np.array(EEG.data, dtype=np.float64)

    return data, sfreq, ch_names


def _make_mne_info(ch_names: list[str], sfreq: float):
    """Build an MNE Info with standard_1020 montage, normalizing channel name case."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    lookup = {name.lower(): name for name in montage.ch_names}
    normalized = [lookup.get(ch.lower(), ch) for ch in ch_names]

    info = mne.create_info(normalized, sfreq, ch_types="eeg", verbose=False)
    info.set_montage(montage, on_missing="ignore", verbose=False)
    return info


def _run_subject(set_path: Path, output_dir: Path, icalabel_threshold: float) -> None:
    import mne
    from pyorica.eval.ica_analysis import ic_source_energy
    from pyorica.eval.runner import run
    from pyorica.pipeline.classify import ICLabelClassifier
    from pyorica.pipeline.pipeline import EEGPipeline

    subject = set_path.parent.name
    print(f"[{subject}] loading {set_path.name}...")
    data, sfreq, ch_names = _load_set(set_path)
    n_ch, n_samples = data.shape

    calib_samples = int(CALIB_SECONDS * sfreq)
    calibration = data[:, :calib_samples]

    print(f"[{subject}] {n_ch} ch, {sfreq} Hz, {n_samples} samples "
          f"({n_samples/sfreq:.0f} s) — calib {CALIB_SECONDS} s")

    info = _make_mne_info(ch_names, sfreq)
    classifier = ICLabelClassifier(info, threshold=icalabel_threshold)
    pipeline = EEGPipeline(n_channels=n_ch, sfreq=sfreq,
                           classifier=classifier, verbose=True)

    print(f"[{subject}] running pipeline (ICLabel threshold={icalabel_threshold}, "
          f"chunk={CHUNK_SIZE} samples)...")
    result = run(pipeline, data, chunk_size=CHUNK_SIZE,
                 calibration_data=calibration, verbose=True)

    print(f"[{subject}] running offline ICA analysis...")
    rows = ic_source_energy(
        result.iir, result.asr, result.output,
        ch_names, sfreq,
    )

    out_path = output_dir / f"{subject}_ic_source_energy.csv"
    fieldnames = ["ic", "label", "ms_iir", "ms_asr", "ms_orica", "pct_asr", "pct_orica"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{subject}] written → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="pyorica IC source energy benchmark")
    parser.add_argument(
        "--subjects", nargs="*", metavar="SID",
        help="Subject IDs to run (e.g. s1 s3 s5). Default: all found in dataset root.",
    )
    parser.add_argument(
        "--output-dir", default="benchmarks/results",
        help="Directory for per-subject CSVs (default: benchmarks/results).",
    )
    parser.add_argument(
        "--icalabel-threshold", type=float, default=0.7,
        help="ICLabel probability threshold for artifact rejection (default: 0.7).",
    )
    args = parser.parse_args()

    data_root_env = os.environ.get("PYORICA_NCTU_DATA", "")
    if not data_root_env:
        print("ERROR: PYORICA_NCTU_DATA environment variable is not set.", file=sys.stderr)
        print("  export PYORICA_NCTU_DATA=/path/to/dataset_2019_TBME", file=sys.stderr)
        sys.exit(1)

    data_root = Path(data_root_env)
    if not data_root.is_dir():
        print(f"ERROR: PYORICA_NCTU_DATA={data_root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = _find_sessions(data_root)
    if not all_sessions:
        print(f"ERROR: no s*/s*_resampled.set files found under {data_root}", file=sys.stderr)
        sys.exit(1)

    if args.subjects:
        wanted = set(args.subjects)
        sessions = [p for p in all_sessions if p.parent.name in wanted]
        missing = wanted - {p.parent.name for p in sessions}
        if missing:
            print(f"WARNING: subjects not found in dataset: {sorted(missing)}", file=sys.stderr)
    else:
        sessions = all_sessions

    print(f"Running {len(sessions)} subject(s): {[p.parent.name for p in sessions]}")
    print(f"Output → {output_dir.resolve()}\n")

    errors = []
    for set_path in sessions:
        try:
            _run_subject(set_path, output_dir, args.icalabel_threshold)
        except Exception as exc:
            subject = set_path.parent.name
            print(f"[{subject}] ERROR: {exc}", file=sys.stderr)
            errors.append(subject)

    if errors:
        print(f"\nFailed subjects: {errors}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nDone. Results in {output_dir.resolve()}")


if __name__ == "__main__":
    main()

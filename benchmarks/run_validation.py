"""Per-subject IC source energy validation benchmark.

Loads sessions from the NCTU-LKT dataset, runs the pyorica pipeline in verbose
mode, performs offline ICA analysis, and writes a per-subject CSV.

Usage
-----
    export PYORICA_NCTU_DATA=/path/to/dataset_2019_TBME
    python benchmarks/run_validation.py [--subjects s1 s3 ...] [--output-dir results]
                                        [--config config.yaml]

Environment
-----------
PYORICA_NCTU_DATA
    Root directory of the NCTU-LKT dataset. Each subject lives at
    ``{root}/s{N}/s{N}_resampled.set``.

Output
------
One CSV per subject at ``{output_dir}/s{N}_ic_source_energy.csv`` with columns:
    ic, label, ms_iir, ms_asr, ms_orica, pct_asr, pct_orica

Also writes ``{subject}_stages.npz`` with full time-series arrays for each
pipeline stage (raw, IIR, ASR, ORICA) so ``analyze_results.py`` can recompute
MS / pct **excluding** the calibration lead-in without re-running the pipeline.

A ``config.yaml`` capturing all pipeline parameters is also written to output_dir.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

import numpy as np


def _fmt_seconds(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    return f"{m}m{sec:02d}s"

# ICLabel's MNE FIR filter (length ~825 at 250 Hz) needs > 825 samples per chunk
# when classify_interval_s=0.
CHUNK_SIZE = 1000


def _find_sessions(root: Path) -> list[Path]:
    sessions = sorted(root.glob("s*/s*_resampled.set"))
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


def run_subject(set_path: Path, config, out_dir: Path,
                ica_cache_dir: Optional[Path] = None,
                max_seconds: Optional[float] = None) -> Path:
    """Run the full pipeline for one subject and write outputs to out_dir.

    Parameters
    ----------
    set_path : Path
        Path to the subject's .set file.
    config : PipelineConfig
        Pipeline configuration (determines ASR backend, cutoff, ORICA params, etc.).
    out_dir : Path
        Directory to write {subject}_ic_source_energy.csv and config.yaml.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    from pyorica.eval.ica_analysis import ic_source_energy
    from pyorica.eval.runner import run
    from pyorica.pipeline.asr_calib import resolve_asr_calibration_npz
    from pyorica.pipeline.classify import ICLabelClassifier
    from pyorica.pipeline.pipeline import EEGPipeline

    subject = set_path.parent.name
    print(f"[{subject}] loading {set_path.name}...")
    data, sfreq, ch_names = _load_set(set_path)
    n_ch, n_samples = data.shape

    if max_seconds is not None and max_seconds > 0:
        max_samples = int(max_seconds * sfreq)
        if max_samples < n_samples:
            data = data[:, :max_samples]
            n_samples = max_samples
            print(f"[{subject}] truncated to {max_seconds:g} s ({n_samples} samples)")

    chunk_size = int(getattr(config, "chunk_size", CHUNK_SIZE) or CHUNK_SIZE)
    calib_samples = int(config.asr_calibration_seconds * sfreq)
    calibration = data[:, :calib_samples]
    calib_label = f"{config.asr_calibration_seconds:.0f} s session lead-in (ORICA)"

    asr_source = getattr(config, "asr_calibration_source", "npz")
    asr_npz_path = None
    asr_calib_save_path = None
    if asr_source == "session":
        asr_calib_save_path = out_dir / f"{subject}_asr_calib_leadin.npz"
        print(
            f"[{subject}] ASR calib: session lead-in "
            f"({config.asr_calibration_seconds:.0f} s) → {asr_calib_save_path.name}"
        )
    else:
        asr_npz = getattr(config, "asr_calibration_npz", None)
        if not asr_npz:
            raise ValueError(
                f"[{subject}] asr_calibration_npz is required when "
                f"asr_calibration_source='npz'"
            )
        asr_npz_path = resolve_asr_calibration_npz(asr_npz, subject)
        print(f"[{subject}] ASR calib NPZ → {asr_npz_path}")

    print(f"[{subject}] {n_ch} ch, {sfreq} Hz, {n_samples} samples "
          f"({n_samples/sfreq:.0f} s) — calib {calib_label}")

    info = _make_mne_info(ch_names, sfreq)
    classifier = ICLabelClassifier(info, threshold=config.icalabel_threshold)
    pipeline = EEGPipeline(n_channels=n_ch, sfreq=sfreq,
                           classifier=classifier, verbose=True, config=config)

    print(f"[{subject}] running pipeline (ASR={config.asr_backend}, "
          f"source={asr_source}, cutoff={config.asr_cutoff}, "
          f"ICLabel threshold={config.icalabel_threshold}, "
          f"chunk={chunk_size} samples)...")
    t0 = time.monotonic()
    run_kwargs = dict(
        calibration_data=calibration,
        ch_names=ch_names,
        verbose=True,
        label=subject,
    )
    if asr_source == "session":
        run_kwargs["session_data"] = data
        run_kwargs["asr_calibration_save_path"] = asr_calib_save_path
    else:
        run_kwargs["asr_calibration_npz"] = str(asr_npz_path)
    result = run(pipeline, data, chunk_size=chunk_size, **run_kwargs)
    print(f"[{subject}] pipeline done  ({_fmt_seconds(time.monotonic() - t0)})")

    out_dir.mkdir(parents=True, exist_ok=True)
    stages_path = out_dir / f"{subject}_stages.npz"
    np.savez(
        stages_path,
        raw=result.raw,
        iir=result.iir,
        asr=result.asr,
        orica=result.output,
        ch_names=np.asarray(ch_names, dtype=object),
        sfreq=np.float64(sfreq),
        calib_samples=np.int64(calib_samples),
    )
    print(
        f"[{subject}] stages saved → {stages_path} "
        f"(raw/iir/asr/orica, shape {result.raw.shape})"
    )

    print(f"[{subject}] running offline ICA analysis...")
    t0 = time.monotonic()
    rows = ic_source_energy(
        result.iir, result.asr, result.output,
        ch_names, sfreq,
        cache_dir=ica_cache_dir,
        subject=subject,
    )

    print(f"[{subject}] ICA analysis done  ({_fmt_seconds(time.monotonic() - t0)})")

    out_path = out_dir / f"{subject}_ic_source_energy.csv"
    fieldnames = ["ic", "label", "ms_iir", "ms_asr", "ms_orica", "pct_asr", "pct_orica"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    config.to_yaml(out_dir / "config.yaml")
    print(f"[{subject}] written → {out_path}")
    return out_path


def main() -> None:
    from pyorica.config import PipelineConfig

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
        "--config", metavar="YAML",
        help="Path to a PipelineConfig YAML file. Defaults to reference experiment settings.",
    )
    parser.add_argument(
        "--ica-cache-dir", metavar="PATH",
        help="Directory for cached ICA objects (.fif/.pkl) and labels (.json). "
             "If a cache exists for a subject it is reused instead of re-fitting ICA.",
    )
    parser.add_argument(
        "--max-seconds", type=float, metavar="SEC",
        help="Process only the first SEC seconds of each session (for quick tests).",
    )
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    ica_cache_dir = Path(args.ica_cache_dir) if args.ica_cache_dir else None

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
            run_subject(
                set_path, config, output_dir,
                ica_cache_dir=ica_cache_dir,
                max_seconds=args.max_seconds,
            )
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

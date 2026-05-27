"""Batch benchmark runner — processes all subjects in the NCTU-LKT dataset.

Discovers subjects under PYORICA_NCTU_DATA, runs each through the pyorica
pipeline (using run_validation.run_subject), and writes per-subject CSVs to a
timestamped run directory. Resumable: subjects with existing CSVs are skipped.

Recommended workflow
--------------------
1. Generate an annotated config file and edit it:

       python benchmarks/run_all_subjects.py --generate-config config.yaml

2. Run the full benchmark using that config:

       export PYORICA_NCTU_DATA=/path/to/dataset_2019_TBME
       python benchmarks/run_all_subjects.py --config config.yaml

3. Aggregate results:

       python benchmarks/aggregate_results.py --run-dir benchmarks/results/run_YYYYMMDD_HHMMSS

Output
------
    benchmarks/results/run_YYYYMMDD_HHMMSS/
        config.yaml                     exact parameters used (annotated)
        s1_ic_source_energy.csv
        s2_ic_source_energy.csv
        ...
        run_summary.txt                 totals, per-subject status, elapsed time
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def _find_sessions(root: Path) -> list[Path]:
    sessions = sorted(root.glob("s*/s*_resampled.set"))
    return [p for p in sessions if "_cleanSec" not in p.name]


def _format_seconds(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    return f"{m}m{sec:02d}s"


def main() -> None:
    from pyorica.config import PipelineConfig
    from benchmarks.run_validation import run_subject

    parser = argparse.ArgumentParser(
        description="Batch pyorica benchmark — all subjects in PYORICA_NCTU_DATA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--generate-config", metavar="PATH",
        help=(
            "Write an annotated default config YAML to PATH and exit. "
            "Edit the file, then re-run with --config PATH."
        ),
    )
    parser.add_argument(
        "--config", metavar="YAML",
        help="Path to a PipelineConfig YAML file (required to run the benchmark).",
    )
    parser.add_argument(
        "--output-dir", default="benchmarks/results",
        help="Parent directory for run outputs (default: benchmarks/results).",
    )
    parser.add_argument(
        "--subjects", nargs="*", metavar="SID",
        help="Limit to specific subject IDs (e.g. s1 s3). Default: all.",
    )
    args = parser.parse_args()

    if args.generate_config:
        out = Path(args.generate_config)
        PipelineConfig().to_yaml(out)
        print(f"Config written to {out.resolve()}")
        print(f"Review and edit, then run:")
        print(f"  python benchmarks/run_all_subjects.py --config {out}")
        return

    if not args.config:
        parser.error(
            "--config YAML is required. Generate a default config first:\n"
            "  python benchmarks/run_all_subjects.py --generate-config config.yaml"
        )

    config = PipelineConfig.from_yaml(args.config)
    print(f"Config loaded from {Path(args.config).resolve()}")

    data_root_env = os.environ.get("PYORICA_NCTU_DATA", "")
    if not data_root_env:
        print("ERROR: PYORICA_NCTU_DATA environment variable is not set.", file=sys.stderr)
        print("  export PYORICA_NCTU_DATA=/path/to/dataset_2019_TBME", file=sys.stderr)
        sys.exit(1)

    data_root = Path(data_root_env)
    if not data_root.is_dir():
        print(f"ERROR: PYORICA_NCTU_DATA={data_root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    all_sessions = _find_sessions(data_root)
    if not all_sessions:
        print(f"ERROR: no s*/s*_resampled.set files found under {data_root}", file=sys.stderr)
        sys.exit(1)

    if args.subjects:
        wanted = set(args.subjects)
        sessions = [p for p in all_sessions if p.parent.name in wanted]
        missing = wanted - {p.parent.name for p in sessions}
        if missing:
            print(f"WARNING: subjects not found: {sorted(missing)}", file=sys.stderr)
    else:
        sessions = all_sessions

    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    config.to_yaml(run_dir / "config.yaml")

    total = len(sessions)
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped: list[str] = []
    batch_start = time.monotonic()

    print(f"pyorica batch benchmark — {total} subject(s)")
    print(f"ASR backend : {config.asr_backend}  cutoff={config.asr_cutoff}")
    print(f"ICLabel thr : {config.icalabel_threshold}")
    print(f"Output dir  : {run_dir.resolve()}\n")

    for idx, set_path in enumerate(sessions, start=1):
        subject = set_path.parent.name
        out_csv = run_dir / f"{subject}_ic_source_energy.csv"

        elapsed = time.monotonic() - batch_start
        if idx > 1 and succeeded:
            avg_per_subject = elapsed / (len(succeeded) + len(failed))
            remaining_est = avg_per_subject * (total - idx + 1)
            time_info = (f"elapsed {_format_seconds(elapsed)}, "
                         f"est. remaining {_format_seconds(remaining_est)}")
        else:
            time_info = f"elapsed {_format_seconds(elapsed)}"

        print(f"[{idx}/{total}] {subject}  ({time_info})")

        if out_csv.exists():
            print(f"  → skipped (output already exists)")
            skipped.append(subject)
            continue

        try:
            run_subject(set_path, config, run_dir)
            succeeded.append(subject)
        except Exception as exc:
            print(f"  → ERROR: {exc}", file=sys.stderr)
            failed.append((subject, str(exc)))

    total_elapsed = time.monotonic() - batch_start

    summary_lines = [
        f"pyorica batch benchmark — {run_tag}",
        f"Total subjects : {total}",
        f"Succeeded      : {len(succeeded)}",
        f"Skipped        : {len(skipped)}",
        f"Failed         : {len(failed)}",
        f"Total elapsed  : {_format_seconds(total_elapsed)}",
        "",
    ]
    if failed:
        summary_lines.append("Failed subjects:")
        for subj, err in failed:
            summary_lines.append(f"  {subj}: {err}")
        summary_lines.append("")
    if skipped:
        summary_lines.append(f"Skipped: {skipped}")

    summary_text = "\n".join(summary_lines)
    (run_dir / "run_summary.txt").write_text(summary_text)

    print(f"\n{'='*60}")
    print(summary_text)
    print(f"Results in {run_dir.resolve()}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

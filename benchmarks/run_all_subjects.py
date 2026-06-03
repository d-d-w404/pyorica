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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional


def _find_sessions(root: Path) -> list[Path]:
    sessions = sorted(root.glob("s*/s*_resampled.set"))
    return [p for p in sessions if "_cleanSec" not in p.name]


def _format_seconds(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    return f"{m}m{sec:02d}s"


def _auto_worker_count(sessions: list[Path]) -> tuple[int, dict]:
    """Compute a safe worker count from available CPU and RAM.

    Returns (n_workers, info_dict) where info_dict contains the values used.
    Per-worker RAM is estimated as 4× the largest .set file on disk (conservative
    multiplier for raw float64 data + pipeline state buffers).
    """
    import psutil

    logical_cpus = os.cpu_count() or 1
    available_ram_gb = psutil.virtual_memory().available / 1e9

    # Conservative RAM estimate per worker: largest file × 4 (float64 + buffers)
    max_file_bytes = max((p.stat().st_size for p in sessions), default=1)
    per_worker_ram_gb = max(0.5, max_file_bytes * 4 / 1e9)

    cpu_workers = max(1, logical_cpus - 2)
    ram_workers = max(1, int(available_ram_gb * 0.8 / per_worker_ram_gb))
    n_workers = min(cpu_workers, ram_workers, len(sessions))

    info = {
        "logical_cpus": logical_cpus,
        "available_ram_gb": round(available_ram_gb, 1),
        "per_worker_ram_gb": round(per_worker_ram_gb, 2),
        "cpu_workers": cpu_workers,
        "ram_workers": ram_workers,
    }
    return n_workers, info


def _print_worker_banner(n_workers: int, info: dict, override: bool) -> None:
    print("── Worker auto-detection ─────────────────────────────────────────────")
    print(f"  Logical CPUs       : {info['logical_cpus']}")
    print(f"  Available RAM      : {info['available_ram_gb']:.1f} GB")
    print(f"  Est. RAM/worker    : {info['per_worker_ram_gb']:.2f} GB")
    print(f"  CPU-limited workers: {info['cpu_workers']}  "
          f"(logical_cpus - 2)")
    print(f"  RAM-limited workers: {info['ram_workers']}  "
          f"(80% RAM / est. per-worker)")
    if override:
        print(f"  Workers            : {n_workers}  (--workers override)")
    else:
        print(f"  Workers            : {n_workers}  (min of cpu/ram limits)")
    print("──────────────────────────────────────────────────────────────────────")


# Kept alive at module level so it is never GC'd — this is intentional.
# All EEG matrix ops are on tiny (n_channels × n_channels) matrices.
# Multi-threaded BLAS gives zero speedup here and causes severe thread-dispatch
# overhead (~0.2s per tiny eigh/gemm call) when N workers run in parallel.
_BLAS_PIN = None


# Module-level wrapper required for ProcessPoolExecutor (lambdas/nested fns aren't picklable)
def _run_subject_safe(set_path: Path, config, run_dir: Path,
                      ica_cache_dir: Optional[Path]) -> None:
    global _BLAS_PIN
    try:
        from threadpoolctl import threadpool_limits
        _BLAS_PIN = threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        pass

    from benchmarks.run_validation import run_subject
    run_subject(set_path, config, run_dir, ica_cache_dir=ica_cache_dir)


def main() -> None:
    from pyorica.config import PipelineConfig

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
        help="Parent directory for run outputs (default: benchmarks/results). "
             "Ignored if --run-dir is given.",
    )
    parser.add_argument(
        "--run-dir", metavar="PATH",
        help="Full output directory for this run (no timestamp appended). "
             "If it already exists, subjects with existing CSVs are skipped.",
    )
    parser.add_argument(
        "--subjects", nargs="*", metavar="SID",
        help="Limit to specific subject IDs (e.g. s1 s3). Default: all.",
    )
    parser.add_argument(
        "--ica-cache-dir", metavar="PATH",
        help="Directory for cached ICA objects and labels. Shared across runs so "
             "different pipeline parameter sweeps reuse the same fitted ICA.",
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Number of parallel worker processes. Default: auto-detected from CPU/RAM.",
    )
    args = parser.parse_args()
    ica_cache_dir = Path(args.ica_cache_dir) if args.ica_cache_dir else None

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

    # Worker count
    try:
        auto_n, worker_info = _auto_worker_count(sessions)
    except ImportError:
        auto_n, worker_info = 1, {
            "logical_cpus": os.cpu_count() or 1,
            "available_ram_gb": 0.0,
            "per_worker_ram_gb": 0.0,
            "cpu_workers": 1,
            "ram_workers": 1,
        }
    n_workers = args.workers if args.workers is not None else auto_n
    _print_worker_banner(n_workers, worker_info, override=args.workers is not None)

    # All EEG matrix ops are on tiny (n_channels × n_channels) matrices.
    # Set BLAS thread count to 1 in the environment BEFORE spawning workers so
    # each child process inherits this before any BLAS library is initialised.
    # Without this, OpenBLAS spawns 12 threads per tiny eigh/gemm call —
    # ~0.2s overhead per call × 125 calls/chunk × 10 workers = hours of waste.
    for _blas_var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "BLIS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[_blas_var] = "1"

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(run_dir / "config.yaml")

    total = len(sessions)
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped: list[str] = []
    batch_start = time.monotonic()

    print(f"\npyorica batch benchmark — {total} subject(s)")
    print(f"ASR backend : {config.asr_backend}  cutoff={config.asr_cutoff}")
    print(f"ICLabel thr : {config.icalabel_threshold}")
    print(f"Output dir  : {run_dir.resolve()}\n")

    def _subject_complete(subject: str) -> bool:
        return (
            (run_dir / f"{subject}_ic_source_energy.csv").exists()
            and (run_dir / f"{subject}_stages.npz").exists()
        )

    # Separate already-complete subjects before touching the pool
    todo = [p for p in sessions if not _subject_complete(p.parent.name)]
    for p in sessions:
        if p not in set(todo):
            skipped.append(p.parent.name)
            print(f"  → {p.parent.name} skipped (CSV + stages NPZ already exist)")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_subject_safe, p, config, run_dir, ica_cache_dir): p
            for p in todo
        }
        try:
            for fut in as_completed(futures):
                path = futures[fut]
                subject = path.parent.name
                elapsed = time.monotonic() - batch_start
                try:
                    fut.result()
                    succeeded.append(subject)
                    print(f"  [OK] {subject}  (elapsed {_format_seconds(elapsed)})")
                except Exception as exc:
                    failed.append((subject, str(exc)))
                    print(f"  [FAIL] {subject}: {exc}  (elapsed {_format_seconds(elapsed)})",
                          file=sys.stderr)
        except KeyboardInterrupt:
            print("\nInterrupted — cancelling remaining jobs...", file=sys.stderr)
            for fut in futures:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(130)

    total_elapsed = time.monotonic() - batch_start

    summary_lines = [
        f"pyorica batch benchmark — {run_dir.name}",
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

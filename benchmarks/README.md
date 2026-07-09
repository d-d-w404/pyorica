# Benchmarks

Three-step workflow to reproduce the cross-session artifact-reduction results.

---

## Prerequisites

Run the setup script from the repo root to create a `.venv` and install the full dependency set (the default ASR backend, `asrpy`, ships as pyorica's bundled `vendor_asrpy` fork — see [ADR-0005](../docs/adr/0005-vendor-asrpy-fork.md) — so no separate install is needed):

```bash
python setup_env.py
```

Then activate the environment before running any benchmark script:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

To also install dev tools (pytest, ruff) for running the test suite:

```bash
python setup_env.py --dev
```

Set the dataset path:

```bash
export PYORICA_NCTU_DATA=/path/to/dataset_2019_TBME
```

The dataset root must contain subjects in the layout:

```
dataset_2019_TBME/
├── s1/
│   └── s1_resampled.set   (+  s1_resampled.fdt)
├── s2/
│   └── s2_resampled.set
└── ...
```

---

## Step 1 — Choose a config file

All pipeline parameters are captured in a `PipelineConfig` YAML. Use the reference config for results that match the original MATLAB ORICA implementation:

```bash
benchmarks/config/reference.yaml   # reference experiment parameters (recommended)
```

Key parameters in `reference.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `asr_backend` | `asrpy` | matches `SN_Driveasrpy20_2min_70` reference |
| `asr_cutoff` | `20.0` | SD multiplier |
| `asr_calibration_seconds` | `120.0` | first 2 min of session |
| `orica_lambda_0` | `0.00133` | initial forgetting factor ≈ steady-state λ (constant slow adaptation) |
| `orica_tau_const` | `3.0` | steady-state λ = 1 − exp(−1/(3 × sfreq)) |
| `orica_block_size_white/ica` | `32` | samples per ORICA update block |
| `icalabel_threshold` | `0.7` | artifact rejection probability |
| `icalabel_apply_car_bandpass` | `false` | apply CAR + 1–100 Hz bandpass before classifying (see below) |
| `classify_interval_s` | `0` | ICLabel every chunk |
| `chunk_size` | `1000` | samples per simulated-real-time step |

**`icalabel_apply_car_bandpass`:** ICLabel's neural network was trained on data referenced to a common average (CAR) and bandpass filtered between 1–100 Hz. The reference legacy ORICA pipeline classifies directly on ASR-cleaned, IIR-filtered data without adding this preprocessing — `false` (the default) matches that behavior. Set to `true` to apply CAR + 1–100 Hz filtering before classification instead, matching ICLabel's documented training assumptions. Benchmarking confirmed `true` slightly improves classification accuracy (fewer brain ICs reduced as artifacts), at the cost of the extra per-chunk bandpass-filter compute — CAR alone (without the bandpass) was tested and found worse than the combination.

ICs get zeroed once above `icalabel_threshold` if their label is in the artifact set (`muscle`, `eye`, `heart`, `line_noise`, `channel_noise`); `brain`/`other` (or any unrecognized label) are never rejected.

The flag defaults to the legacy-matching setting, so you can A/B a single run:

```bash
python benchmarks/run_all_subjects.py --config benchmarks/config/reference.yaml \
    --ica-cache-dir benchmarks/ica_cache --output-dir benchmarks/results/car_bandpass_on
```

with a copy of `reference.yaml` that only flips `icalabel_apply_car_bandpass: true`, then compare `cross_session_summary.csv` between the two output directories via Step 3.

To create a custom config based on current defaults:

```python
from pyorica.config import PipelineConfig
PipelineConfig().to_yaml("my_config.yaml")
```

---

## Step 2 — Run all subjects

```bash
python benchmarks/run_all_subjects.py --config benchmarks/config/reference.yaml \
                                      --ica-cache-dir benchmarks/ica_cache \
                                      [--output-dir benchmarks/results] \
                                      [--subjects s1 s3 s5]
```

**What it does:**
- Discovers all `s*/s*_resampled.set` files under `PYORICA_NCTU_DATA`
- For each subject: loads data, runs the pipeline in verbose mode, runs offline ICA analysis, writes `{subject}_ic_source_energy.csv`
- Prints progress with elapsed time and estimated remaining time
- **Resumable**: subjects with existing CSVs are skipped — safe to re-run after interruption
- Writes all outputs to a timestamped directory: `{output-dir}/run_YYYYMMDD_HHMMSS/`

**ICA caching (`--ica-cache-dir`):**

Offline ICA fitting is the dominant per-subject cost (~10–20 min each). Use `--ica-cache-dir` to save fitted ICA objects the first time and reuse them in subsequent runs:

```bash
# First run: fits ICA for each subject, writes cache
python benchmarks/run_all_subjects.py --config benchmarks/config/reference.yaml \
    --ica-cache-dir benchmarks/ica_cache

# Subsequent runs with different pipeline parameters: ICA load is instant
python benchmarks/run_all_subjects.py --config my_config.yaml \
    --ica-cache-dir benchmarks/ica_cache   # same cache dir, shared across runs
```

The cache is keyed by subject ID and `random_state` (default 42). If you run with a different seed, the cache is rejected with a clear error. The cache directory can be shared across multiple run directories.

**Output directory layout:**

```
benchmarks/results/run_20260527_120000/
├── config.yaml                    ← exact parameters used
├── s1_ic_source_energy.csv
├── s1_ic_class_timeline.png       ← IC class timeline (see below)
├── s2_ic_source_energy.csv
├── s2_ic_class_timeline.png
│   ...
└── run_summary.txt                ← total/succeeded/failed/elapsed
```

**Per-subject CSV columns:**

| Column | Description |
|--------|-------------|
| `ic` | IC index |
| `label` | ICLabel class (brain, muscle, eye, …) |
| `ms_iir` | Mean-square energy after IIR |
| `ms_asr` | Mean-square energy after ASR |
| `ms_orica` | Mean-square energy after ORICA |
| `pct_asr` | % energy reduction: ASR vs IIR |
| `pct_orica` | % energy reduction: ORICA vs IIR |

**IC class timeline (`{subject}_ic_class_timeline.png`):**

Generated automatically alongside each CSV. Shows how ORICA's online IC classifications evolve across the session:

- **X-axis** — IC index in fixed ORICA unmixing-matrix order (never reordered between snapshots)
- **Y-axis** — time in seconds; one row per ICLabel classification event (every `classify_interval_s` seconds; `0` = every chunk, which is the default in `reference.yaml`)
- **Cell color** — top-1 ICLabel class using the MNE-icalabel color convention (brain=blue, muscle=red, eog=green, ecg=pink, line\_noise=yellow, ch\_noise=orange, other=gray)

The plot reflects the live ORICA weights at each snapshot, not a post-hoc offline ICA.

**Expected runtime:** ~15–30 min per subject (offline ICA is the bottleneck).

---

## Step 3 — Aggregate cross-session results

```bash
python benchmarks/aggregate_results.py --run-dir benchmarks/results/run_20260527_120000
```

**What it does:**
- Reads all `*_ic_source_energy.csv` from the run directory
- Within each subject: takes **median** of `pct_asr` and `pct_orica` across all ICs of the same ICLabel class
- Across subjects: computes **mean ± SD** (population SD) per class
- Always shows all 7 ICLabel classes on the x-axis (classes with no ICs appear as NaN)

**Outputs (written to the same run directory):**

| File | Description |
|------|-------------|
| `cross_session_results.png` | Grouped bar chart: ASR vs IIR and ORICA vs IIR bars side-by-side per IC class |
| `cross_session_summary.csv` | Table: class, mean_pct_asr, sd_pct_asr, mean_pct_orica, sd_pct_orica, n_subjects |

---

## Single-subject quick run

For development or debugging, run one subject directly:

```bash
python benchmarks/run_validation.py --subjects s1 \
                                    --output-dir benchmarks/results/debug \
                                    --config benchmarks/config/reference.yaml
```

Produces `s1_ic_source_energy.csv` and `s1_ic_class_timeline.png` in the output directory.

---

## Known divergences from the original ORICA pipeline

| Aspect | Original | pyorica |
|--------|----------|---------|
| Bad-segment exclusion | Manual per-subject `EXCLUDE_TIME_RANGES_S` | Not implemented — noted as future milestone |
| 60 Hz notch filter | Separate online notch stage | Folded into IIR `h_freq=50.0` (no separate notch) |
| GUI / LSL streaming | Required for online run | Not required for benchmarking (uses `ArrayStream`) |

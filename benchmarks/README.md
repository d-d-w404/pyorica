# Benchmarks

Three-step workflow to reproduce the cross-session artifact-reduction results.

---

## Prerequisites

Install the full dependency set:

```bash
pip install -e ".[full]"
pip install asrpy      # ASR default backend (not yet on PyPI — install from source)
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

## Step 1 — (Optional) Generate a config file

All pipeline parameters are captured in a `PipelineConfig`. The defaults match the reference experiments:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `asr_backend` | `asrpy` | matches `SN_Driveasrpy20_2min_70` reference |
| `asr_cutoff` | `20.0` | SD multiplier |
| `asr_calibration_seconds` | `120.0` | first 2 min of session |
| `icalabel_threshold` | `0.7` | artifact rejection probability |
| `iir_l_freq` / `iir_h_freq` | `1.0` / `50.0` | bandpass |

To generate a config file (and edit before running):

```python
from pyorica.config import PipelineConfig
PipelineConfig().to_yaml("my_config.yaml")
```

---

## Step 2 — Run all subjects

```bash
python benchmarks/run_all_subjects.py [--config my_config.yaml] \
                                      [--output-dir benchmarks/results] \
                                      [--subjects s1 s3 s5]
```

**What it does:**
- Discovers all `s*/s*_resampled.set` files under `PYORICA_NCTU_DATA`
- For each subject: loads data, runs the pipeline in verbose mode, runs offline ICA analysis, writes `{subject}_ic_source_energy.csv`
- Prints progress with elapsed time and estimated remaining time
- **Resumable**: subjects with existing CSVs are skipped — safe to re-run after interruption
- Writes all outputs to a timestamped directory: `{output-dir}/run_YYYYMMDD_HHMMSS/`

**Output directory layout:**

```
benchmarks/results/run_20260527_120000/
├── config.yaml                    ← exact parameters used
├── s1_ic_source_energy.csv
├── s2_ic_source_energy.csv
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
| `cross_session_results.png` | Two bar charts: ASR vs IIR (left) and ORICA vs IIR (right) |
| `cross_session_summary.csv` | Table: class, mean_pct_asr, sd_pct_asr, mean_pct_orica, sd_pct_orica, n_subjects |

---

## Single-subject quick run

For development or debugging, run one subject directly:

```bash
python benchmarks/run_validation.py --subjects s1 \
                                    --output-dir benchmarks/results/debug \
                                    [--config my_config.yaml]
```

---

## Known divergences from the original ORICA pipeline

| Aspect | Original | pyorica |
|--------|----------|---------|
| Bad-segment exclusion | Manual per-subject `EXCLUDE_TIME_RANGES_S` | Not implemented — noted as future milestone |
| 60 Hz notch filter | Separate online notch stage | Folded into IIR `h_freq=50.0` (no separate notch) |
| GUI / LSL streaming | Required for online run | Not required for benchmarking (uses `ArrayStream`) |

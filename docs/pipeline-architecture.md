# Pipeline Architecture

Two diagrams below describe pyorica's architecture. The first covers the real-time processing path; the second covers the evaluation framework used to benchmark results offline.

---

## 1. Real-time pipeline

Each chunk of EEG data flows through five sequential stages. All state (filter coefficients, ASR calibration, ORICA weights) persists across chunks.

```mermaid
flowchart TD
    A["LSL stream / ArrayStream\n(live inlet or file replay)"]
    B["Chunk\n(channels × samples)"]
    C["IIRFilter\nbandpass 1–50 Hz · sosfilt · persistent zi state"]
    D["ASRAdapter\nArtifact Subspace Reconstruction\nbackend: asrpy (default) or meegkit\nstateful R / Zi / cov across chunks"]
    E["ORICAFilter\nOnline Recursive ICA\nupdate() → W and sphere adapted\ntransform() → source space"]
    F["Classifier\nICLabel: (sources, mixing_matrix, sfreq) → artifact_mask\nzero artifact ICs in source space"]
    G["Reconstruction\npinv(W × sphere) · inverse_transform()\nsensor space restored"]
    H["Cleaned chunk\n(channels × samples)"]

    A --> B --> C --> D --> E --> F --> G --> H
```

**Key invariants**

| Stage | Input shape | Output shape | State |
|-------|-------------|--------------|-------|
| IIRFilter | (ch, samples) | (ch, samples) | filter zi per channel |
| ASRAdapter | (ch, samples) | (ch, samples) | R, Zi, cov (asrpy) |
| ORICAFilter | (ch, samples) | (ch, components) | W, sphere |
| Classifier | (comp, samples) | bool mask (comp,) | none |
| Reconstruction | (comp, samples) | (ch, samples) | none |

---

## 2. Evaluation framework

The evaluation framework replays a session file in **simulated real-time** — using the same stateful pipeline code path as a live stream — then performs an offline ICA analysis to measure per-IC artifact energy reduction.

```mermaid
flowchart TD
    A["Session file\n(.set or .fif)"]
    B["loader.load_set() / load_fif()\nreturns data (ch × samples), sfreq, ch_names"]
    C["ArrayStream\nreplays data chunk-by-chunk\npreserves all stateful pipeline behavior"]
    D["EEGPipeline\nverbose=True\nstores _last_raw, _last_iir, _last_asr\nafter each process() call"]
    E["RunResult\nraw · iir · asr · orica\nstage arrays (ch × samples each)"]
    F["ic_source_energy()\noffline MNE extended-Infomax ICA\nfitted on full IIR stage array\nunmixing matrix applied to IIR, ASR, ORICA"]
    G["ICLabel classification\nlabels each IC: brain / muscle / eye\nheart / line noise / ch noise / other"]
    H["per-subject CSV\nic · label · ms_iir · ms_asr · ms_orica\npct_asr · pct_orica"]

    A --> B --> C --> D --> E --> F --> G --> H
```

**Design note**: The offline ICA in `ic_source_energy()` is a separate measurement tool — it uses a stable MNE ICA decomposition fitted on the IIR stage, *not* the online ORICA weights. This ensures the metric is independent of ORICA's convergence state and gives a reproducible ground-truth comparison across pipeline stages.

```mermaid
flowchart LR
    subgraph "Per subject"
        H["per-subject CSV\n(ic · label · ms_* · pct_*)"]
    end
    subgraph "Cross-session aggregation"
        I["aggregate_results.py\nwithin-subject: median per ICLabel class\ncross-subject: mean ± SD"]
        J["cross_session_results.png\ntwo bar charts:\nASR vs IIR · ORICA vs IIR\nall 7 ICLabel classes"]
        K["cross_session_summary.csv"]
    end
    H --> I --> J
    I --> K
```

---

## Correspondence to original ORICA pipeline

| Aspect | Original ORICA (`receiver.py`) | pyorica |
|--------|-------------------------------|---------|
| ASR backend | `EEG_ASR_BACKEND=asrpy` (reference) | `PipelineConfig.asr_backend="asrpy"` |
| ASR cutoff | `EEG_ASR_CUTOFF=20` | `PipelineConfig.asr_cutoff=20.0` |
| Calibration window | first 2 min of session IIR-filtered | `PipelineConfig.asr_calibration_seconds=120.0` |
| ICLabel threshold | `EEG_ICALABEL_THRESHOLD=0.7` | `PipelineConfig.icalabel_threshold=0.7` |
| Pipeline order | IIR → notch → ASR → ORICA | IIR → ASR → ORICA (notch folded into IIR h_freq) |

All reference-experiment parameters are captured in `PipelineConfig` defaults and serialized to `config.yaml` alongside each benchmark output.

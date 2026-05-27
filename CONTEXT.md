# pyorica

An open-source Python package for real-time EEG artifact removal and source decomposition, targeting EEG and BCI research communities.

## Language

### Signal processing

**Chunk**:
A block of EEG data with shape `(channels × samples)` representing one unit of processing — the atomic input and output of every stage in the pipeline.
_Avoid_: frame, window, epoch, block

**Calibration data**:
A segment of clean EEG used to warm-start ORICA weights or fit ASR before real-time processing begins. Optional — both components fall back to default initial conditions if omitted.
_Avoid_: training data, reference data, baseline

**ORICA (Online Recursive Independent Component Analysis)**:
The core adaptive algorithm that incrementally updates an unmixing matrix `W` and whitening matrix `sphere` one chunk at a time without storing a full data history.
_Avoid_: online ICA, adaptive ICA

**ICA weight matrix (W)**:
The unmixing matrix maintained by ORICA; maps whitened EEG channels to independent components. Shape `(n_components × n_components)`.
_Avoid_: weight matrix, ICA weights, icaweights

**Whitening matrix (sphere)**:
The sphering matrix maintained by ORICA's RLS whitening step; decorrelates channels before ICA. Shape `(n_components × n_components)`.
_Avoid_: sphering matrix, icasphere, covariance matrix

**IC (Independent Component)**:
A single source signal extracted from EEG by applying `W × sphere` to the data. Each IC is either brain-origin or an artifact.
_Avoid_: source, component, factor

**Artifact IC**:
An IC classified as non-brain (e.g. eye movement, muscle, line noise) by the classifier. Zeroed in source space before reconstruction.
_Avoid_: bad component, EOG component, noise IC

**Forgetting factor (λ)**:
A scalar in `(0, 1)` that controls how fast ORICA adapts to non-stationary data. Profiles: `cooling` (decreasing), `constant`, `adaptive`.
_Avoid_: learning rate, decay rate

### Pipeline

**Pipeline**:
The ordered chain IIR → ASR → ORICA → classify → reconstruct that transforms a raw EEG chunk into a cleaned EEG chunk.
_Avoid_: processing chain, workflow

**Reconstruction**:
The step that inverse-transforms source-space signals (with artifact ICs zeroed) back to sensor space via `pinv(W × sphere)`.
_Avoid_: back-projection, inverse transform

**ASR (Artifact Subspace Reconstruction)**:
A preprocessing stage that removes gross transient artifacts from EEG before ORICA sees the data. Runs before ORICA in the pipeline.
_Avoid_: artifact removal, cleaning

**ASR backend**:
The library used to implement ASR — either `"asrpy"` (default; matches reference experiments) or `"meegkit"` (alternative for comparison). Recorded in `PipelineConfig.asr_backend`.
_Avoid_: ASR library, ASR implementation

**ASR cutoff**:
The standard-deviation multiplier (e.g. `20`) used by ASR to threshold artifact subspaces. Higher values = less aggressive cleaning.
_Avoid_: ASR threshold, ASR parameter

**PipelineConfig**:
A Python dataclass capturing all parameters needed to reproduce a pipeline run: IIR band edges and order, ASR backend/cutoff/calibration seconds, ORICA forgetting-factor profile and hyperparameters, and ICLabel threshold. Serialized to/from YAML. Saved alongside every benchmark output.
_Avoid_: config, settings, parameters

**Batch runner**:
`benchmarks/run_all_subjects.py` — discovers all sessions in the benchmark dataset, processes each through the pipeline in simulated real-time, and writes per-subject CSVs + a `config.yaml`. Resumable: skips subjects whose output already exists.
_Avoid_: bulk runner, dataset runner

**Cross-session aggregation**:
`benchmarks/aggregate_results.py` — reads all per-subject CSVs, computes within-subject median IC source MS energy per ICLabel class, then cross-subject mean ± SD. Produces two bar charts (ASR vs IIR, ORICA vs IIR) across all 7 ICLabel classes.
_Avoid_: cross-subject aggregation, multi-subject analysis

**Classifier**:
Any callable `(sources, mixing_matrix, sfreq) → artifact_mask` that identifies artifact ICs. `mixing_matrix` is `A = pinv(W × sphere)` — the mixing matrix mapping source space back to channel space. The default implementation uses ICLabel; any callable with this signature is valid.
_Avoid_: IC labeler, artifact detector

### Streaming and evaluation

**Stream**:
A source of EEG chunks. Either a live LSL inlet (`LSLStream`) or a numpy array replayed chunk-by-chunk (`ArrayStream`). Both yield `(channels × samples)` chunks.
_Avoid_: inlet, source, reader

**Session**:
A single recording file (`.set` or `.fif`) treated as one independent run in the evaluation framework.
_Avoid_: recording, file, subject

**Simulated real-time**:
The evaluation mode in which a session file is fed through the pipeline one chunk at a time using `ArrayStream`, preserving all stateful behavior (filter state, ORICA weights, ASR state) across chunks exactly as in a live stream.
_Avoid_: offline processing, batch processing, replay

**Stage arrays**:
The four full-session channel-space arrays captured during a verbose pipeline run: `raw` (pre-IIR), `iir` (post-IIR), `asr` (post-ASR), and `orica` (post-reconstruction). Accumulated by the eval runner when `verbose=True`; all `None` otherwise.
_Avoid_: intermediate outputs, pipeline outputs, saved stages

**Verbose mode**:
A pipeline operating mode (`EEGPipeline(verbose=True)`) in which the pipeline stores the most recent intermediate chunk for each stage as attributes (`_last_raw`, `_last_iir`, `_last_asr`) after each `process()` call, allowing the runner to accumulate stage arrays.
_Avoid_: debug mode, logging mode

**Offline ICA analysis**:
A post-run validation procedure in `eval/ica_analysis.py` that fits MNE extended-Infomax ICA on the full IIR stage array, then projects all three filtered stage arrays (IIR, ASR, ORICA) through the resulting unmixing matrix to compute per-IC source MS energy and percent reduction vs IIR. ICs are classified by ICLabel. Distinct from the online ORICA decomposition used during the pipeline run.
_Avoid_: offline ICA, validation ICA, post-processing ICA

**IC source MS energy**:
The mean-square energy of a single IC's source time series, computed as `mean(source²)` over all samples (or a masked subset excluding bad segments). Used as the primary metric in offline ICA analysis to compare artifact reduction across pipeline stages.
_Avoid_: IC power, source variance, IC energy

**Benchmark**:
A script in `benchmarks/` that loads sessions from a local dataset (path from `PYORICA_NCTU_DATA` env var), runs the pipeline in verbose mode, performs offline ICA analysis per session, and writes per-subject CSV results. Not part of the test suite.
_Avoid_: experiment, validation script, evaluation script

## Relationships

- A **Stream** yields one **Chunk** at a time to the **Pipeline**
- The **Pipeline** passes each **Chunk** through IIR → ASR → **ORICA** → **Classifier** → **Reconstruction**
- **ORICA** maintains `W` and `sphere` across chunks; both are updated by `update()` and used by `transform()` / `inverse_transform()`
- The **Classifier** receives ORICA's sources and returns an artifact mask; it does not modify **ORICA** state
- The eval `runner` wraps an `ArrayStream` + **Pipeline** to process a **Session** in **Simulated real-time**
- **Calibration data** is passed to `fit()` on both ASR and ORICA before streaming begins; it is not a **Session**
- In **verbose mode**, the runner accumulates **stage arrays** (`raw`, `iir`, `asr`, `orica`) alongside the normal `RunResult`
- **Offline ICA analysis** consumes the **stage arrays** from a verbose run; it uses a separate stable ICA decomposition, not the online ORICA weights, to ensure the measurement is independent of convergence state
- **IC source MS energy** is computed per IC per stage inside **offline ICA analysis**; ICLabel class labels come from `mne-icalabel` applied to the ICA fitted on the IIR stage array

## Example dialogue

> **Dev:** "When we get a new chunk from LSL, does ORICA refit from scratch?"
> **Domain expert:** "No — ORICA calls `update()`, which adjusts `W` and `sphere` incrementally using the forgetting factor. A full refit only happens if you explicitly call `fit()` with calibration data."

> **Dev:** "What happens to artifact ICs — are they removed from the data?"
> **Domain expert:** "They're zeroed in source space, then reconstruction projects back to sensor space. The channel count stays the same; artifact energy is suppressed, not channels dropped."

> **Dev:** "Is the eval runner doing real offline ICA?"
> **Domain expert:** "No — it's simulated real-time. It uses ArrayStream to feed the session file chunk-by-chunk through the same stateful pipeline a live stream would use. The weights evolve the same way they would online."

## Flagged ambiguities

- "EOG indices" appears in the source repo (`latest_eog_indices`) but refers to all artifact ICs, not only eye-movement components. In pyorica, the canonical term is **artifact IC** and the mask is called `artifact_mask`.
- "window" was used in validation scripts to mean both a **Chunk** (pipeline input) and an analysis epoch (longer segment for metric computation). In pyorica: **chunk** = pipeline unit; analysis epochs in `eval/analysis.py` are called **windows** only in that context.

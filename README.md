# pyorica

Real-time EEG artifact removal and source decomposition via **Online Recursive ICA (ORICA)**.

> Algorithm: Hsu, S.-H. et al., IEEE TNSRE 2016. [DOI: 10.1109/TNSRE.2015.2508103](https://doi.org/10.1109/TNSRE.2015.2508103)

## What's in the box

| Capability | Module | Extra |
|---|---|---|
| Lightweight ORICA filter — drop-in for real-time LSL pipelines | `pyorica.orica` | *(core)* |
| Stateful IIR bandpass filter with persistent chunk state | `pyorica.filters` | *(core)* |
| LSL stream inlet — yields `(channels × samples)` chunks | `pyorica.streaming.lsl` | `[lsl]` |
| Array replay stream — feeds a numpy array chunk-by-chunk | `pyorica.streaming.array` | *(core)* |
| End-to-end pipeline: IIR → ASR → ORICA → ICLabel → reconstruct | `pyorica.pipeline` | `[pipeline]` |
| ICLabel classifier — pluggable artifact IC labelling | `pyorica.pipeline.classify` | `[pipeline]` |
| Dataset loader — `.set` (EEGLAB) and `.fif` (MNE) | `pyorica.eval.loader` | `[eval]` |
| Simulated real-time runner with per-chunk RMS metrics | `pyorica.eval.runner` | `[eval]` |
| Analysis utilities — RMS reduction dB, run summaries | `pyorica.eval.analysis` | `[eval]` |

## Installation

```bash
# Core ORICA filter (numpy + scipy only)
pip install pyorica

# + LSL streaming
pip install pyorica[lsl]

# + Full pipeline (IIR → ASR → ORICA → ICLabel → reconstruct)
pip install pyorica[pipeline]

# + Evaluation framework (dataset loading, metrics, plots)
pip install pyorica[eval]

# Everything
pip install pyorica[full]
```

## Quick start

### ORICA filter (core)

```python
from pyorica import ORICAFilter

orica = ORICAFilter(n_components=16, sfreq=256)
orica.fit(calibration_data)          # optional warm-start

for chunk in stream:                 # chunk shape: (16, n_samples)
    orica.update(chunk)
    sources = orica.transform(chunk)
    sources[artifact_mask] = 0
    cleaned = orica.inverse_transform(sources)
```

### End-to-end pipeline

```python
from pyorica.pipeline import EEGPipeline, ICLabelClassifier
from pyorica.streaming.lsl import LSLStream

clf = ICLabelClassifier(raw.info)    # needs MNE Info with channel positions
pipeline = EEGPipeline(n_channels=64, sfreq=256, classifier=clf)
pipeline.fit(calibration_data)       # calibrates ASR + warm-starts ORICA

for chunk in LSLStream('MyEEGStream', chunk_size=64):
    cleaned = pipeline.process(chunk)
```

Or with any callable as the classifier:

```python
# zero the highest-variance IC on every chunk
classifier = lambda sources, A, sfreq: (
    mask := np.zeros(sources.shape[0], dtype=bool),
    mask.__setitem__(sources.var(axis=1).argmax(), True),
    mask
)[-1]
pipeline = EEGPipeline(n_channels=16, sfreq=256, classifier=classifier)
```

### Offline evaluation

```python
from pyorica.eval.loader import load_dataset
from pyorica.eval.runner import run
from pyorica.eval.analysis import summarize

data, sfreq = load_dataset('recording.set')          # EEGLAB or MNE .fif
pipeline = EEGPipeline(n_channels=data.shape[0], sfreq=sfreq)
result = run(pipeline, data, chunk_size=64, calibration_data=data[:, :2048])
print(summarize(result))
# {'mean_rms_reduction_db': 4.3, 'mean_rms_input': 0.82, ...}
```

## Project layout

```
pyorica/
  orica/          # ORICAFilter — core adaptive ICA algorithm
  filters/        # IIRFilter — stateful scipy sosfilt wrapper
  streaming/      # ArrayStream, LSLStream
  pipeline/       # EEGPipeline, ICLabelClassifier
  eval/           # loader, runner, analysis
tests/            # pytest suite (53 fast tests + slow integration tests)
docs/
  adr/            # Architecture decision records
CONTEXT.md        # Domain glossary and terminology
CITATION.cff      # BibTeX-ready citation for the ORICA paper
```

## Citation

If you use pyorica, please cite the ORICA algorithm paper (see `CITATION.cff`).

## Documentation

- [Domain glossary and terminology](CONTEXT.md)
- [ADR 0001 — channels-first data convention](docs/adr/0001-channels-first-data-convention.md)
- [ADR 0002 — pylsl over mne-lsl](docs/adr/0002-pylsl-over-mne-lsl.md)
- [ADR 0003 — pipeline order IIR→ASR→ORICA](docs/adr/0003-pipeline-order-iir-asr-orica.md)
- [ADR 0004 — scipy stateful IIR filter](docs/adr/0004-scipy-stateful-iir-filter.md)

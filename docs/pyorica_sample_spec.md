# pyorica — Project Specification

> **For:** Claude Code agent  
> **Task:** Scaffold and implement the `pyorica` Python package  
> **Origin:** Ported and extended from the MATLAB REST (Real-time EEG Source-mapping Toolbox)

---

## Overview

`pyorica` is an open-source Python package for **real-time EEG artifact removal and source decomposition**. It targets the EEG and BCI research community and is designed to be compatible with real-time EEG signal processing tools, particularly [mne-lsl](https://github.com/mne-tools/mne-lsl).

The package implements two core algorithms from the MATLAB REST toolbox:

- **ASR** — Artifact Subspace Reconstruction (artifact removal)
- **ORICA** — Online Recursive ICA (real-time source decomposition)

---

## Package Structure

```
pyorica/
├── __init__.py                  # Public API, version, lazy imports
│
├── asr/                         # Artifact Subspace Reconstruction
│   ├── __init__.py
│   ├── core.py                  # ASRFilter class; processes chunks in-place
│   ├── calibrate.py             # Build clean reference subspace from calibration window
│   └── mixingmat.py             # Mixing/unmixing matrix maintenance, rank-safe updates
│
├── orica/                       # Online Recursive ICA
│   ├── __init__.py
│   ├── core.py                  # ORICADecomposer class with main update loop
│   ├── update.py                # Weight update rules (natural gradient, block, recursive)
│   └── weights.py               # Weight matrix initialization and regularization
│
├── decomposition/               # Higher-level source analysis (ASR + ORICA combined)
│   ├── __init__.py
│   ├── pipeline.py              # RESTpipeline class: chains ASR → ORICA as a single callable
│   ├── dipfit.py                # Dipole fitting for IC source localization (wraps MNE dipole API)
│   └── classify.py              # IC classification: brain vs. artifact (ICLabel-style logic)
│
├── streaming/                   # Real-time streaming interface (mne-lsl compatibility)
│   ├── __init__.py
│   ├── base.py                  # BaseStream abstract class; decouples algorithms from transport
│   ├── lsl.py                   # LSLStream: wraps mne_lsl StreamInlet/StreamOutlet; supports add_callback()
│   └── buffer.py                # Thread-safe ring buffer; chunk assembly and timestamp alignment
│
├── viz/                         # Visualization
│   ├── __init__.py
│   ├── topomap.py               # Real-time IC scalp topographies (wraps MNE plot_topomap)
│   ├── timeseries.py            # Scrolling cleaned vs. raw EEG comparison
│   └── dashboard.py             # Live dashboard (PyQtGraph or matplotlib): ASR state, IC weights, signal quality
│
└── utils/                       # Shared utilities
    ├── __init__.py
    ├── filters.py               # Bandpass/notch helpers for offline MNE Raw and live chunks
    ├── epochs.py                # Adapters to convert streaming chunks to MNE Epochs-like objects
    └── logging.py               # Structured logger matching mne-lsl logging conventions

tests/
├── test_asr.py
├── test_orica.py
└── test_stream.py               # Uses mne-lsl PlayerLSL to mock a live stream

examples/
├── lsl_demo.py                  # Full real-time pipeline via mne-lsl
├── offline_demo.py              # MNE Raw file (offline processing)
└── bci_demo.py                  # P300 / motor imagery closed-loop BCI example

docs/
├── conf.py                      # Sphinx + MyST, styled after MNE-Python docs
├── api.rst                      # Auto-generated API reference
└── tutorials/                   # Narrative tutorials

benchmarks/
├── bench_asr.py                 # ASR latency and throughput profiling
├── bench_orica.py               # ORICA update speed vs. chunk size
└── latency.py                   # End-to-end online processing delay measurement

pyproject.toml                   # Modern packaging with optional dependency groups
README.md
CITATION.cff
.github/workflows/               # CI: lint, test, docs build
```

---

## Module Descriptions

### `asr/core.py` — `ASRFilter`

- Implements the ASR algorithm for online artifact removal
- Accepts EEG chunks as numpy arrays of shape `(n_channels, n_samples)`
- Maintains internal state across chunks (mixing matrix, threshold)
- Key method: `process(chunk: np.ndarray) -> np.ndarray`

### `asr/calibrate.py`

- Computes the clean reference subspace from a calibration recording
- Returns the mixing matrix and threshold used by `ASRFilter`
- Should accept an MNE `Raw` object or a numpy array

### `orica/core.py` — `ORICADecomposer`

- Implements Online Recursive ICA for real-time blind source separation
- Maintains and updates the unmixing matrix `W` incrementally per chunk
- Key method: `update(chunk: np.ndarray) -> np.ndarray` (returns IC activations)

### `orica/update.py`

- Implements weight update rules:
  - Natural gradient update
  - Block update
  - Recursive (sample-by-sample) update
- Configurable learning rate schedule (fixed, adaptive, annealing)

### `decomposition/pipeline.py` — `RESTpipeline`

- High-level entry point combining ASR → ORICA
- Compatible with mne-lsl's callback pattern:

```python
from pyorica.decomposition import RESTpipeline
from pyorica.streaming import LSLStream

pipeline = RESTpipeline(sfreq=256, n_channels=64)
stream = LSLStream(stream_name="EEG")
stream.add_callback(pipeline.process)
stream.connect()
```

### `streaming/base.py` — `BaseStream`

- Abstract base class; any streaming backend must subclass this
- Required methods: `connect()`, `disconnect()`, `add_callback()`, `get_data()`
- Decouples algorithm modules from transport layer (LSL, BrainFlow, custom, etc.)

### `streaming/lsl.py` — `LSLStream`

- Wraps `mne_lsl.lsl.StreamInlet` and `mne_lsl.lsl.StreamOutlet`
- Supports mne-lsl's `add_callback(fn)` pattern; `fn` receives `(data, timestamps)`
- Optionally writes cleaned output back to a new LSL outlet for downstream tools

### `streaming/buffer.py`

- Thread-safe circular ring buffer matching mne-lsl's internal buffer model
- Handles chunk assembly from variable-size pulls
- Maintains accurate timestamps across chunks

---

## Dependency Groups (`pyproject.toml`)

```toml
[project]
name = "pyorica"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "mne>=1.6",
]

[project.optional-dependencies]
lsl   = ["mne-lsl>=1.5"]
viz   = ["matplotlib>=3.7", "pyqtgraph"]
bci   = ["scikit-learn>=1.3", "moabb"]
dev   = ["pytest", "pytest-cov", "ruff", "black", "sphinx", "myst-parser"]
```

---

## mne-lsl Compatibility Notes

- mne-lsl uses a `StreamLSL` (high-level) and `StreamInlet` (low-level) pattern
- Data arrives as numpy arrays of shape `(n_samples, n_channels)` — **note the transpose** vs. MNE's internal `(n_channels, n_times)` convention; handle this at the streaming boundary
- mne-lsl's `add_callback(fn)` passes `(data: np.ndarray, timestamps: np.ndarray)` to `fn` after each acquisition window
- Use `mne_lsl.player.PlayerLSL` in tests to replay `.fif` files as mock live streams
- Target `python >= 3.11` to match mne-lsl's requirement

---

## Testing Strategy

- Unit tests for `asr/` and `orica/` using synthetic EEG (numpy random + known artifact injection)
- Integration tests in `test_stream.py` using `PlayerLSL` to mock a live LSL stream
- All tests runnable offline (no hardware required)
- CI via GitHub Actions: `pytest`, `ruff`, `black`, docs build

---

## Style & Conventions

- Follow MNE-Python API conventions where possible (e.g. `fit()`, `transform()`, `apply()`)
- NumPy-style docstrings on all public classes and methods
- Type hints on all function signatures
- `ruff` for linting, `black` for formatting
- BSD-3-Clause license (matching mne-lsl)
- Include `CITATION.cff` for academic citation support

---

## Reference

- Original MATLAB REST toolbox: artifact subspace reconstruction + ORICA for real-time EEG
- [mne-lsl documentation](https://mne.tools/mne-lsl)
- [MNE-Python documentation](https://mne.tools/stable)

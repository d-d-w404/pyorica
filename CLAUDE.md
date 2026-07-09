# pyorica

Real-time EEG artifact removal and source decomposition (ORICA-based).

## Environment setup

Before running any code, tests, or benchmark scripts in this repo, ensure the `.venv` virtual environment exists and is activated.

Create it (first time only, or if `.venv` is missing):

```bash
python setup_env.py          # base deps
python setup_env.py --dev    # + pytest, ruff for running the test suite
```

Activate it before any Python command:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

The default ASR backend, `asrpy`, ships as pyorica's bundled `vendor_asrpy` fork (see [ADR-0005](docs/adr/0005-vendor-asrpy-fork.md)) — no separate install needed.

See `benchmarks/README.md` for the full benchmark workflow (dataset path, config, running subjects, aggregating results).

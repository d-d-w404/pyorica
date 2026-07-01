# Vendor a patched asrpy fork as vendor_asrpy

`ASRAdapter`'s `"asrpy"` backend depends on `asrpy` 0.0.8 (PyPI), which crashes under NumPy ≥2: `fit_eeg_distribution` calls `int()` on a size-1 ndarray, which NumPy <2 allowed (with a `DeprecationWarning`) but NumPy ≥2 raises as `TypeError`. Upstream (`DiGyt/asrpy`) is unmaintained — its GitHub `main` is byte-identical to the PyPI release, and none of its 19 forks or 2 open PRs address this or several other open issues (block_covariance reshape errors, missing return in `clean_windows`, etc.).

We vendor a patched fork at `vendor_asrpy/` (top-level, alongside `pyorica/`) rather than depending on a git URL or publishing a separate PyPI package. `vendor_asrpy` is bundled into the same wheel as `pyorica`, so `pip install pyorica[pipeline]` remains fully PyPI-installable with no path/git dependencies in the metadata — which matters because pyorica is itself meant to be a publishable, pip-installable package.

The vendored copy carries only source (`asr.py`, `asr_utils.py`, `__init__.py`) plus the upstream BSD-3-Clause `LICENSE` (attributed to Dirk Gütlin) and a provenance header noting the exact upstream commit forked from (`5a99169f`) — not upstream's tests/docs/CI, which pyorica's own test suite supersedes. It also folds in the `block_covariance` off-by-one workaround that `ASRAdapter._fit_asrpy` previously applied externally (trimming a sample before calling `asr.fit`).

The external `asrpy>=0.0.8` PyPI dependency is dropped from `pyproject.toml`'s `pipeline` extra. `ASRAdapter`'s config-facing backend name stays `"asrpy"` (`asr_backend: "asrpy"` in YAML configs) — it names the algorithm/interface, not the package providing it, so no existing config or reference-experiment doc needs to change.

## Considered options

- **Publish a separate forked package to PyPI** (e.g. `asrpy2`). Rejected: means maintaining and releasing a second published package for what is currently a small, targeted fix.
- **Git-URL or local-path dependency.** Rejected: not accepted in a published package's metadata, breaks `pip install pyorica` for anyone outside this repo.
- **Monkeypatch asrpy at runtime from within `pyorica`.** Rejected in favor of vendoring: a runtime patch is harder to test, review, and extend than owning the source directly, and this bug is unlikely to ever be fixed upstream.

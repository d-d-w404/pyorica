# Align ORICAFilter's internals with the quick30 legacy reference

`ORICAFilter` previously implemented the ORICA update loop by interleaving whitening and ICA within a single per-block pass (`min(block_size_white, block_size_ica)`), matching a straightforward reading of Hsu et al. (2016). We're replacing this with a two-pass structure — a full whitening pass over the chunk, then a separate ICA pass over uncentered mixtures — that numerically matches `ORICA_final_no_print_quick30.py`, an existing legacy receiver implementation already validated in prior real-time deployments.

We picked quick30-parity over a from-scratch reading of the paper because the paper under-specifies several details quick30 already resolved through real-world use: exact block partitioning, whether ICA mixtures are computed from centered or raw data, and how the forgetting-factor floor is enforced. Matching quick30 exactly means pyorica's real-time behavior is reproducible against a system with a track record, rather than a fresh reimplementation with unknown edge-case behavior.

The paper (Hsu et al., 2016) remains the algorithmic reference in `core.py`'s docstring alongside the quick30 functional reference — quick30 is itself an implementation of the same algorithm; the citation records *what* ORICA is, quick30 records *which exact numerical choices* pyorica now reproduces.

## Consequences

- `force_constant_lambda` defaulted to `True`, which meant `ff_profile="cooling"` and `"adaptive"` were effectively inert unless a caller explicitly set `force_constant_lambda=False` — this mirrored quick30's own hard-coded behavior, not a pyorica-specific simplification. **Superseded by [ADR-0007](0007-forgetting-factor-profile-consolidation.md)**, which removes the separate flag in favor of `ff_profile="constant"` as the sole (and now default) way to select this behavior.
- `block_size_white` and `block_size_ica` are independent knobs (whitening's per-chunk update granularity vs. ICA's, driven by different stationarity assumptions) and must each drive their own block partitioning — a shared `numsplits` derived from only one of them is a bug, not a simplification, and must not reappear in future edits to `_run_orica`.
- Benchmark results produced before this change are not numerically comparable to results after it; any before/after comparison needs config/version provenance (already captured by `PipelineConfig` serialization).
- **`_lambda_const` stays `0.0` (not `0.98`) when `tau_const=np.inf`.** An early draft of this change carried over `0.98` from the reference PR without validating it: `_dynamic_orica`'s weight update multiplies by `lam_prod = prod(1/(1-lambda))` per block, and a sustained `lambda=0.98` at typical block sizes (8) makes `lam_prod ≈ 50^8` per block — weights diverge to NaN within ~100 updates. `tau_const=np.inf` means "no steady-state floor"; cooling/adaptive must be left to decay toward 0 unconstrained. Caught by `tests/test_orica.py::test_cross_talk_error_on_sim_stat` once it was updated to exercise `force_constant_lambda=False` — the reference PR never updated this test, so it would have hit the same divergence.

## Considered options

- **Keep the paper-based interleaved implementation.** Rejected: no real-world validation history, and the paper doesn't fully specify enough of the numerical details to be confident it matches any known-working system.
- **Support both implementations behind a flag.** Rejected for now: adds a permanent maintenance branch inside `core.py` for two behaviors when only one has field validation; can be revisited if the quick30-aligned version underperforms in broader benchmarking beyond the initial s03 check.

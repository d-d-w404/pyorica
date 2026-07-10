# Consolidate forgetting-factor selection into `ff_profile` alone

ADR-0006 introduced `force_constant_lambda`, a boolean on `ORICAFilter` that, when `True` (its default), overrode `ff_profile` and forced every update to use the constant `lambda_const` floor regardless of whether `"cooling"`, `"constant"`, or `"adaptive"` was selected. This created two switches controlling the same behavior: `ff_profile="constant"` and `force_constant_lambda=True` did the same thing, and `ff_profile="cooling"`/`"adaptive"` were silently inert unless a caller also remembered to pass `force_constant_lambda=False`.

The redundancy had a real consequence: `PipelineConfig` never had a `force_constant_lambda` field, so any `EEGPipeline` built from a `PipelineConfig` always got `ORICAFilter`'s hardcoded `force_constant_lambda=True` default — `PipelineConfig.orica_ff_profile` was silently inert for every config-driven pipeline run, including the reference benchmark. `reference.yaml`'s `orica_ff_profile: "cooling"` never actually ran as cooling; it always ran constant.

We're removing `force_constant_lambda` entirely. `ff_profile` becomes the single source of truth, with three mutually exclusive values and clarified real-world intent:

- **`"constant"`** (new default) — λ pinned to the `tau_const`-derived floor. Assumes data is stationary within that window. The production / real-time setting.
- **`"cooling"`** — λ decreases across the whole session per `lambda_0`/`gamma`, assuming the entire session is stationary. Validation-only, for comparing against offline (batch) ICA — not for real-time deployment.
- **`"adaptive"`** — λ responds to the running non-stationarity index, speeding up adaptation when the data's statistics shift.

## Consequences

- `tau_const` (and the `lambda_const` it derives) remains a floor under **all** profiles, not just `"constant"` — including `"cooling"` and `"adaptive"`. This preserves the numerical-stability guard from ADR-0006 (unconstrained λ decay under `tau_const=inf` risks the `lam_prod` divergence documented there) while still letting cooling/adaptive drive λ above that floor. **Superseded by [ADR-0008](0008-remove-forgetting-factor-floor-from-cooling-adaptive.md)**, which found this floor made cooling/adaptive numerically collapse into constant under everyday near-stationary data, defeating their purpose, and removed it for those two profiles.
- `ORICAFilter.__init__`'s `ff_profile` default changes from `"cooling"` to `"constant"`; `PipelineConfig.orica_ff_profile`'s default changes the same way. Every `PipelineConfig`-driven pipeline now actually runs in the mode its config says, instead of silently ignoring `orica_ff_profile` in favor of `ORICAFilter`'s old hardcoded override.
- `benchmarks/config/reference.yaml` is corrected from `orica_ff_profile: "cooling"` to `"constant"` — this does not change any benchmark's numerical output, since those runs were already executing in constant mode via the old `force_constant_lambda=True` default. It only makes the config honest about what it does. `orica_lambda_0`/`orica_gamma` remain at their tuned values in the file as inert cooling-profile parameters, in case the config is later switched to `"cooling"` for offline-ICA validation.
- `gamma` and `lambda_0` are cooling-only parameters; `PipelineConfig.to_yaml()` and `benchmarks/README.md` now say so explicitly rather than presenting them as always-active.
- The adaptive profile's internal tuning constants (`decay_rate_alpha`, `upper_bound_beta`, `trans_band_width_gamma`, `trans_band_center`) remain hardcoded defaults inside `_gen_adaptive_ff`, not promoted to `ORICAFilter`/`PipelineConfig` fields — out of scope for this change.
- This is a breaking API change: any caller passing `force_constant_lambda=` to `ORICAFilter` will now get a `TypeError`. No backward-compatibility shim is provided, per project convention.

## Considered options

- **Keep both `force_constant_lambda` and `ff_profile`, just change the default profile to `"constant"`.** Rejected: preserves the exact dual-switch redundancy that caused `PipelineConfig.orica_ff_profile` to be silently inert; doesn't fix the underlying bug, only masks it for the default case.
- **Make `tau_const`/`lambda_const` a floor only under `"constant"`, letting `"cooling"`/`"adaptive"` decay unconstrained.** Considered, but rejected at the time in favor of keeping the floor universal, reasoning it preserved the numerical-stability protection from ADR-0006 for all profiles. Revisited and reversed in [ADR-0008](0008-remove-forgetting-factor-floor-from-cooling-adaptive.md) once testing showed the universal floor made cooling/adaptive indistinguishable from constant in practice.

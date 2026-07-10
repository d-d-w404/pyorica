# Remove the tau_const-derived λ floor from cooling and adaptive profiles

ADR-0007 kept `tau_const`'s `lambda_const` floor active under all three `ff_profile` values — `"constant"`, `"cooling"`, and `"adaptive"` — reasoning that it preserved the numerical-stability guard documented in ADR-0006. In practice, this made `"cooling"` and `"adaptive"` fail to do what they're for.

Testing with `benchmarks/config/reference.yaml`'s parameters (`lambda_0=0.00133`, `tau_const=3.0`, `sfreq=256`) surfaced the problem: `lambda_0` had historically been tuned to sit right at `lambda_const` (`≈0.0013`), so cooling's `λ = lambda_0/t^gamma` dropped below the floor within ~10 samples and was clamped there for the rest of any real session — cooling and constant produced numerically indistinguishable output. Restoring `lambda_0` to its original value (`0.995`, matching the `PipelineConfig` default) fixed cooling's divergence from constant, but exposed the deeper issue: **the floor itself defeats the purpose of a decaying/responsive profile.** Any `lambda_0`/`gamma` combination will eventually decay below `lambda_const` given enough samples — that's what "cooling" means — at which point the floor clamps it back to constant behavior regardless of the parameters chosen.

The same investigation showed `"adaptive"` collapsing into constant even more readily: its λ locks onto whatever `lambda_const` floor it last hit and stays there under stationary data, only escaping when a genuine non-stationarity event (a `Rn`-norm ratio approaching the profile's transition-band center) pushes it back up. On stationary or lightly-varying test data — including ordinary synthetic Gaussian noise — this event never fires, so adaptive is floor-locked to constant from essentially the second processed block onward.

We're removing the floor for `"cooling"` and `"adaptive"`; it now applies only to `"constant"`, where it's the entire point (the steady-state λ *is* the floor value, always).

## The zero-seed bug this uncovered

Removing the floor exposed a latent bug: `ORICAFilter.__init__` seeded `_lambda_k` (the adaptive profile's recursive state) to `np.zeros(1)`. `_gen_adaptive_ff` is multiplicative in this seed (`term1 = (1+gain)^n * lam0`, `term2 ∝ lam0²`), so a zero seed is an absorbing fixed point — λ would compute to exactly 0 forever. Previously masked by the floor (λ=0 was always clamped back up to `lambda_const` on the very first call), this became a real failure the instant the floor was removed: weights froze at the identity matrix for the entire session, and `_dynamic_whitening`'s `Q = lam_avg/(1-lam_avg)` computation divided by zero (`lam_avg = 1 - 0 = 1`).

Fixed by seeding `_lambda_k` to `lambda_0` instead of zero — consistent with `lambda_0` already meaning "initial forgetting factor" for the cooling profile, and giving adaptive's recursion a sensible, nonzero starting point.

## Consequences

- `ORICAFilter._forgetting_factor` no longer clamps `"cooling"`/`"adaptive"` output to `lambda_const` under any circumstance; `tau_const` is now read only when `ff_profile="constant"`.
- `_lambda_k` is seeded from `lambda_0` at construction, not zero. This changes `"adaptive"`'s numerical output from any previous run (it was previously floor-clamped from the start; the API contract was already documented as "no backward-compatibility shim" per ADR-0007, so this is treated the same way).
- Cooling's λ now decays genuinely unbounded toward 0 for the life of the filter — verified stable (no NaN/Inf) over a 10-minute simulated stationary session; `lam_prod = prod(1/(1-λ))` doesn't blow up because λ→0 monotonically, never approaching 1.
- `lambda_0` is now meaningful under both `"cooling"` (decay start) and `"adaptive"` (recursion seed) — previously documented as cooling-only. `gamma` remains cooling-only.
- `benchmarks/config/reference.yaml`'s `orica_lambda_0`/`orica_gamma` are labeled "cooling and adaptive" rather than "cooling only, ignored under constant."
- `tests/test_orica.py::test_adaptive_profile_does_not_freeze_at_identity` locks in the zero-seed fix: constructs an adaptive-profile filter under a `RuntimeWarning`-as-error filter (catches the division-by-zero) and asserts weights move away from the identity matrix after several updates.

## Considered options

- **Keep the universal floor (status quo from ADR-0007).** Rejected: directly caused the "all three profiles give identical results" symptom this ADR investigates. A floor that always wins makes the profile selection itself pointless for two of the three options.
- **Keep the floor, but require callers to pass `tau_const=np.inf` to disable it for cooling/adaptive.** Rejected: pushes a footgun onto every caller who wants a working cooling/adaptive profile — the default (`tau_const=3.0`) would silently defeat both non-constant profiles unless the caller knew to override it. Scoping the floor to `"constant"` structurally removes the footgun instead of documenting around it.
- **Seed `_lambda_k` from `lambda_const` instead of `lambda_0`.** Considered — would have been a smaller conceptual change (reusing the value that used to rescue the zero case). Rejected because `lambda_const` may be `0.0` (when `tau_const=inf`), reintroducing the exact same absorbing-zero bug for that configuration; `lambda_0` is guaranteed caller-supplied and already documented as "initial forgetting factor."

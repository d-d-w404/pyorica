# Original ORICA pipeline vs pyorica: known discrepancies

Comparison based on `ORICA/code/receiver.py`, `ORICA_final_no_print_quick30.py`,
`run_two_instances_Driving.py`, and `aa_lsl_npz.py` against the current pyorica
implementation. Reference experiment: `SN_Driveasrpy20_2min_70` (subject s28_resampled).

---

## 1. ORICA block sizes: `block_size_white` and `block_size_ica` are conflated

**Original (`ORICA_final_no_print_quick30.py`)**  
`block_size_white=8` and `block_size_ica=1` are independent loop parameters.
Whitening (sphere) updates every 8 samples; ICA weight (W) updates every 1 sample.

**pyorica (`core.py:211`)**  
```python
block_size = min(self.block_size_white, self.block_size_ica)
```
Both updates run on the same stride. With the reference config (`block_size_white=8,
block_size_ica=1`), `min(8, 1) = 1`, so whitening also updates every sample instead
of every 8. This diverges from the MATLAB reference implementation.

**Risk:** More frequent whitening updates change the covariance estimate dynamics and
may affect convergence rate in ways that are hard to detect from output quality alone.

**Status:** `reference.yaml` uses `block_size_white: 32, block_size_ica: 32`, so `min(32,32)=32`
and the conflation is harmless for that config. The code-level conflation in `_run_orica`
remains; independent block strides are a future improvement.

---

## 2. `orica_tau_const` default: 3 vs inf

**Original:** `tau_const=3` (ORICA_final_new default). Controls the floor of the
cooling forgetting factor: `λ_const = 1 - exp(-1 / (tau_const × sfreq))`.
At 500 Hz this gives `λ_const ≈ 1 - exp(-1/1500) ≈ 0.000667`.

**pyorica:** Code default `tau_const=np.inf` → `λ_const = 0.0`, meaning the forgetting
factor decays all the way to zero with no floor; ORICA never stops discounting old data.

**Status:** Fixed in `reference.yaml` (`orica_tau_const: 3.0`, `orica_lambda_0: 0.00133`).
The code default in `ORICAFilter.__init__` still reads `tau_const=np.inf`; `reference.yaml`
overrides this at runtime.

---

## 3. No notch filter stage in pyorica

**Original (`receiver.py:66–76, 1228–1246`):**  
An online causal IIR notch at 60 Hz (Q=30) is applied immediately after the bandpass,
before ASR. Controlled by `EEG_NOTCH_FREQ` (default 60; set 0 to disable, 50 for EU).

**pyorica:** The pipeline is IIR → ASR → ORICA. There is no notch filter stage or
corresponding config parameter.

**Risk:** 60 Hz (or 50 Hz) power-line noise passes through to ASR and ORICA, which may
increase the number of ICs flagged as line-noise artifacts by ICLabel.

**Status:** Unresolved. A notch filter step and `notch_freq` / `notch_q` config
parameters need to be added if the pipeline is to match the original.

---

## 4. ASR: accumulation buffer vs lookahead zero-padding

**Original (`receiver.py:609–634`):**  
Incoming LSL chunks are accumulated until the buffer reaches `0.5 × srate` samples,
then ASR is called on the full buffer. Only the last `n_in` samples are kept as output.
This guarantees ASR always receives at least ~0.5 s of context per call.

**pyorica (`asr.py:91–125`):**  
No accumulation. Each chunk is padded with `lookahead = 0.25 s` of zeros at the end,
`asr_process` is called, and the padding is stripped. Stateful `(R, Zi, cov)` carry
context across calls.

**Risk:** For chunks below asrpy's internal window length (~0.5 s = 125 samples at
250 Hz), covariance statistics are estimated over fewer real samples than the original
guaranteed. Behavior will be similar on average but may differ for brief bursts.

**Status:** Intentional redesign. The lookahead approach is architecturally cleaner
and avoids the latency introduced by accumulation, but has not been validated against
the original's output on the reference dataset.

---

## 5. ORICA silently drops weight updates for chunks smaller than `block_size`

**pyorica (`core.py:211–212`):**  
```python
block_size = min(self.block_size_white, self.block_size_ica)
n_blocks   = n_pts // block_size   # integer division
```
- If `n_pts < block_size`: `n_blocks = 0`, the loop body never runs, W and sphere are
  not updated. `transform` still applies stale weights — output is produced but the
  model stops learning for that chunk.
- If `n_pts >= block_size` but `n_pts % block_size != 0`: tail samples are silently
  discarded every call.

**Original:** With `block_size_ica=1`, every sample triggered an update and no data
was ever dropped.

**Risk:** With the current config (`block_size_ica=1`), `block_size=1` and this is a
non-issue. If `block_size_ica` is ever set >1 and LSL returns fewer samples than that
value (which can happen at stream start), ORICA freezes silently.

**Status:** Fixed. `_run_orica` now uses `_orica_block_ranges()` (legacy ORICA even-split):
floor division sets the number of blocks, then all samples are distributed evenly. Chunks
smaller than `block_size` still yield zero blocks (consistent with original), but the common
case of chunk ≥ block_size always processes all samples.

---

## 6. No chunk-size parameter in `config.yaml`; benchmark uses inconsistent value

**pyorica:**  
`LSLStream` default: 64 samples. `runner.run()` default: 64 samples.
Benchmark (`run_validation.py`): hardcoded `CHUNK_SIZE = 1000` samples.

**Original:** LSL broadcaster (`aa_lsl_npz.py`) pushed fixed 50-sample chunks;
the receiver processed whatever `pull_chunk` returned (variable, up to 50).

The 1000-sample benchmark chunk is ~15× larger than real-time chunks (64) and ~20×
larger than the original (50). Benchmark results may not reflect real-time behaviour,
particularly for ORICA convergence and ASR windowing.

**Status:** Fixed. `chunk_size` is now a `PipelineConfig` field (default 1000). The
benchmark reads it from the config YAML so experiments are fully reproducible. Value
of 1000 (4 s at 250 Hz) intentionally exceeds LSL chunk sizes to give ICLabel enough
context; documented in `reference.yaml`.

---

## Summary table

| # | Parameter / behaviour | Original | pyorica | Status |
|---|---|---|---|---|
| 1 | whitening vs ICA block stride | independent (8 / 1) | conflated via `min()` | harmless for `reference.yaml` (both=32); independent strides future work |
| 2 | `orica_tau_const` / `lambda_0` | 3 / 0.00133 | code default inf/0.995 | **fixed** in `reference.yaml`; code defaults unchanged |
| 3 | notch filter | 60 Hz IIR notch | absent | unresolved |
| 4 | ASR short-chunk handling | 0.5 s accum buffer | 0.25 s zero-pad | intentional redesign, unvalidated |
| 5 | ORICA update on small chunks | always (block_size=1) | skip if chunk < block_size | **fixed** — even-split covers all samples when chunk ≥ block_size |
| 6 | chunk size | 50 samples (fixed push) | 64 real-time / 1000 benchmark | **fixed** — `chunk_size` in `PipelineConfig`, set to 1000 in `reference.yaml` |

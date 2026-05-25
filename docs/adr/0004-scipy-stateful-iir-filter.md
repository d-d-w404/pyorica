# scipy sosfilt with explicit filter state for IIR filtering

The IIR filter uses `scipy.signal.sosfilt` with persistent `zi` (initial conditions) carried across chunks, rather than `mne.filter.filter_data` as used in the source repo. This keeps `mne` out of the core install — the ORICA algorithm and IIR filter together require only `numpy` and `scipy`.

Stateless per-chunk filtering (calling `sosfilt` without `zi`) introduces a startup transient at every chunk boundary, which is incorrect for a real-time pipeline. Carrying `zi` across chunks produces continuous filtering identical to applying the filter to the full signal — the same guarantee `mne.filter_data` provides internally, without the MNE dependency.

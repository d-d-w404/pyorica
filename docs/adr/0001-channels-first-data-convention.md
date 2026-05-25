# Channels-first (channels × samples) data convention throughout

All numpy arrays representing EEG data in pyorica use shape `(channels × samples)`. This matches the MATLAB `orica.m` original, EEGLAB, MNE's internal layout, and the `pylsl` output after the single boundary transpose. The sklearn convention `(samples × features)` was considered but rejected: the EEG research community universally expects channels on the first axis, and adopting the sklearn convention would require transposing at every algorithm boundary instead of once at the LSL inlet.

The sole exception is the LSL boundary: `pylsl.pull_chunk()` returns `(samples × channels)`, which `LSLStream` transposes immediately before yielding chunks downstream.

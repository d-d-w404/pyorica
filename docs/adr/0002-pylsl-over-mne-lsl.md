# pylsl over mne-lsl as the streaming layer

pyorica uses `pylsl` directly rather than `mne-lsl`. `mne-lsl` is the higher-level MNE-native option and provides `PlayerLSL` for test replay, but it pulls `mne` into the core install — a heavy dependency that the ORICA algorithm and IIR filter do not require. `pylsl` is a thin binding to liblsl with no MNE dependency, matching the source repo's battle-tested usage.

The `mne` dependency is confined to the `[pipeline]` extra (for `mne-icalabel`) and the `[eval]` extra (for file IO). Users who only need the ORICA filter or the streaming layer can install `pyorica[lsl]` without pulling in MNE.

Test replay (the main advantage of `mne-lsl`) is handled by `ArrayStream`, which feeds a numpy array chunk-by-chunk through the same pipeline interface and has no LSL dependency at all.

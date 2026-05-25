# Test Data

`SIM_STAT_16ch_3min.set` and `SIM_STAT_16ch_3min.fdt` are a simulated 16-channel, 3-minute stationary EEG dataset used by the slow integration tests. The dataset was generated using the EEGLAB ORICA simulation scripts from the original MATLAB ORICA repository (Hsu et al., IEEE TNSRE 2016, DOI: 10.1109/TNSRE.2015.2508103). Sampling rate: 128 Hz.

These files are excluded from the pip package (`[tool.hatch.build.targets.sdist] exclude = ["tests/data"]`) and are only needed to run the slow integration tests (`pytest -m slow`).

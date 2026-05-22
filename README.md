# pyorica

Real-time EEG artifact removal and source decomposition via **Online Recursive ICA (ORICA)**.

> Algorithm: Hsu, S.-H. et al., IEEE TNSRE 2016. [DOI: 10.1109/TNSRE.2015.2508103](https://doi.org/10.1109/TNSRE.2015.2508103)

## Installation

```bash
# Core ORICA filter (numpy + scipy only)
pip install pyorica

# + LSL streaming
pip install pyorica[lsl]

# + Full pipeline (IIR → ASR → ORICA → ICLabel → reconstruct)
pip install pyorica[pipeline]

# + Evaluation framework (dataset loading, metrics, plots)
pip install pyorica[eval]

# Everything
pip install pyorica[full]
```

## Quick start

```python
from pyorica import ORICAFilter
import numpy as np

orica = ORICAFilter(n_components=16, sfreq=256)
orica.fit(calibration_data)          # optional warm-start

for chunk in stream:                 # chunk shape: (16, n_samples)
    orica.update(chunk)
    sources = orica.transform(chunk)
    sources[artifact_mask] = 0
    cleaned = orica.inverse_transform(sources)
```

## Citation

If you use pyorica, please cite the ORICA algorithm paper (see `CITATION.cff`).

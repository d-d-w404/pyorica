"""End-to-end EEG artifact removal pipeline (IIR → ASR → ORICA → classify → reconstruct).

Requires: pip install pyorica[pipeline]
"""

from pyorica.pipeline.pipeline import EEGPipeline
from pyorica.pipeline.classify import ICLabelClassifier

__all__ = ["EEGPipeline", "ICLabelClassifier"]

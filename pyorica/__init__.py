"""
pyorica — Real-time EEG artifact removal via Online Recursive ICA.

Install extras:
    pip install pyorica[lsl]       # LSL streaming (pylsl)
    pip install pyorica[pipeline]  # Full pipeline (meegkit, mne, mne-icalabel)
    pip install pyorica[eval]      # Evaluation framework (mne IO, matplotlib)
    pip install pyorica[full]      # Everything
"""

from pyorica.filters.iir import IIRFilter
from pyorica.orica.core import ORICAFilter

__all__ = ["IIRFilter", "ORICAFilter"]
__version__ = "0.1.0"

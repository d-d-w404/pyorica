# -*- coding: utf-8 -*-
# (c) Dirk Gütlin, 2021. <dirk.guetlin@gmail.com>
#
# License: BSD-3-Clause
#
# Vendored into pyorica as `vendor_asrpy` from DiGyt/asrpy @ 5a99169f
# (https://github.com/DiGyt/asrpy), which is unmaintained and incompatible
# with NumPy >=2. See docs/adr/0005-vendor-asrpy-fork.md. Patches applied
# on top of upstream:
#   - asr_utils.fit_eeg_distribution: avoid int() on a size-1 ndarray,
#     which NumPy >=2 raises as TypeError instead of a DeprecationWarning.
#   - asr.asr_calibrate: trim a trailing sample when block_covariance's
#     off-by-one would otherwise misshape the block covariance array.
"""
Welcome to the ASRpy documentation.

ASR (Artefact Subspace Reconstruction) is a widely used automated cleaning
algorithm for EEG data. This version of ASR is implemented to easily
integrate with the MNE-Python toolbox for M/EEG analysis. The original method
was invented by Kothe & Jung (2016) for the EEGLab toolbox.

You find the documentation to all available functions by clicking on
the respective submodules.

"""

__version__ = "0.0.8-pyorica1"

from .asr import ASR, asr_calibrate, asr_process, clean_windows  # noqa: F401

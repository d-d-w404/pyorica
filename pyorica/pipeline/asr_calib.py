"""Load external ASR calibration NPZ files (ORICA receiver format)."""

from pathlib import Path
from typing import Optional, Union

import numpy as np


def load_asr_calibration_npz(
    path: Union[str, Path],
    n_channels: Optional[int] = None,
) -> np.ndarray:
    """Load calibration data from an ORICA-style ``.npz`` file.

    Returns ndarray shaped ``(n_channels, n_samples)`` in float64.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ASR calibration NPZ not found: {path}")

    npz = np.load(path, allow_pickle=True)
    calibration_data = None
    for key in ("calibration_data", "data", "eeg_data"):
        if key in npz.files:
            calibration_data = npz[key]
            break
    if calibration_data is None:
        raise KeyError(
            f"No calibration array in {path}; keys: {list(npz.files)}"
        )

    calibration_data = np.asarray(calibration_data, dtype=np.float64)
    if calibration_data.ndim == 2:
        if calibration_data.shape[0] > calibration_data.shape[1] * 10:
            calibration_data = calibration_data.T

    if n_channels is not None and calibration_data.shape[0] != n_channels:
        if calibration_data.shape[0] > n_channels:
            calibration_data = calibration_data[:n_channels, :]
        else:
            raise ValueError(
                f"Calibration has {calibration_data.shape[0]} channels, "
                f"expected {n_channels}"
            )
    return calibration_data


def resolve_asr_calibration_npz(template: str, subject: str) -> Path:
    """Expand ``{subject}`` in a config path template."""
    return Path(template.replace("{subject}", subject))

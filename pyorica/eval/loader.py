"""Dataset loader for .set (EEGLAB) and .fif (MNE) files."""

import os
import numpy as np


def load_set(path):
    """Load an EEGLAB .set file.

    Parameters
    ----------
    path : str
        Path to the .set file. The companion .fdt file must be in the same directory.

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples), dtype float64
    sfreq : float
    """
    import scipy.io

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    EEG = mat['EEG']
    n_ch = int(EEG.nbchan)
    n_pts = int(EEG.pnts)
    sfreq = float(EEG.srate)

    fdt_path = os.path.splitext(path)[0] + '.fdt'
    if os.path.exists(fdt_path):
        data = np.fromfile(fdt_path, dtype='<f4', count=n_ch * n_pts)
        data = data.reshape((n_ch, n_pts), order='F').astype(np.float64)
    else:
        data = np.array(EEG.data, dtype=np.float64)

    return data, sfreq


def load_fif(path):
    """Load an MNE .fif file.

    Parameters
    ----------
    path : str
        Path to the .fif file.

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples), dtype float64
    sfreq : float
    """
    import mne

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    data = raw.get_data().astype(np.float64)
    return data, raw.info['sfreq']


def load_dataset(path):
    """Load a dataset file, dispatching by extension.

    Supported extensions: ``.set``, ``.fif``.

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples), dtype float64
    sfreq : float

    Raises
    ------
    FileNotFoundError
    ValueError
        If the file extension is not supported.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == '.set':
        return load_set(path)
    elif ext == '.fif':
        return load_fif(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: .set, .fif"
        )

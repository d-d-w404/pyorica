"""Aggregate metrics and publication plots for pipeline evaluation."""

import numpy as np


def rms_reduction_db(rms_input, rms_output):
    """Compute mean RMS reduction in decibels across chunks.

    Parameters
    ----------
    rms_input : ndarray, shape (n_chunks,)
    rms_output : ndarray, shape (n_chunks,)

    Returns
    -------
    float
        Positive values indicate RMS reduction; 0 means no change.
    """
    rms_in = np.asarray(rms_input, dtype=np.float64)
    rms_out = np.asarray(rms_output, dtype=np.float64)
    per_chunk_db = 20.0 * np.log10((rms_in + 1e-12) / (rms_out + 1e-12))
    return float(np.mean(per_chunk_db))


def summarize(result):
    """Compute a summary dict from a ``RunResult``.

    Parameters
    ----------
    result : RunResult

    Returns
    -------
    dict with keys:
        ``mean_rms_reduction_db``, ``mean_rms_input``, ``mean_rms_output``,
        ``n_chunks``, ``n_channels``.
    """
    return {
        'mean_rms_reduction_db': rms_reduction_db(result.rms_input, result.rms_output),
        'mean_rms_input': float(np.mean(result.rms_input)),
        'mean_rms_output': float(np.mean(result.rms_output)),
        'n_chunks': len(result.rms_input),
        'n_channels': result.n_channels,
    }

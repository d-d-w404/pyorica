"""Simulated real-time pipeline runner."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pyorica.streaming.array import ArrayStream


@dataclass
class RunResult:
    """Results from a simulated real-time pipeline run.

    Attributes
    ----------
    output : ndarray, shape (n_channels, n_samples)
        Cleaned EEG data (same shape as input).
    rms_input : ndarray, shape (n_chunks,)
        RMS of each input chunk before the pipeline.
    rms_output : ndarray, shape (n_chunks,)
        RMS of each output chunk after the pipeline.
    chunk_size : int
    n_channels : int
    n_samples : int
    raw : ndarray or None
        Stage array before IIR filtering. Populated only when ``verbose=True``.
    iir : ndarray or None
        Stage array after IIR filtering. Populated only when ``verbose=True``.
    asr : ndarray or None
        Stage array after ASR. Equals ``iir`` when ASR is not fitted.
        Populated only when ``verbose=True``.
    """
    output: np.ndarray
    rms_input: np.ndarray
    rms_output: np.ndarray
    chunk_size: int
    n_channels: int
    n_samples: int
    raw: Optional[np.ndarray] = None
    iir: Optional[np.ndarray] = None
    asr: Optional[np.ndarray] = None


def run(pipeline, data, chunk_size=64, calibration_data=None, verbose=False):
    """Run a pipeline over data in simulated real-time.

    Parameters
    ----------
    pipeline : EEGPipeline
        Configured pipeline instance.
    data : ndarray, shape (n_channels, n_samples)
        EEG data to process.
    chunk_size : int
        Samples per chunk (default 64).
    calibration_data : ndarray, optional
        If provided, ``pipeline.fit()`` is called before processing.
    verbose : bool
        If True, accumulate stage arrays (raw, iir, asr) in the result.

    Returns
    -------
    RunResult
    """
    if calibration_data is not None:
        pipeline.fit(calibration_data)

    n_channels, n_samples = data.shape
    stream = ArrayStream(data, chunk_size=chunk_size)

    chunks_out = []
    rms_in_list = []
    rms_out_list = []

    raw_chunks = [] if verbose else None
    iir_chunks = [] if verbose else None
    asr_chunks = [] if verbose else None

    if verbose:
        pipeline._verbose = True

    for chunk in stream:
        rms_in_list.append(float(np.sqrt(np.mean(chunk ** 2))))
        cleaned = pipeline.process(chunk)
        chunks_out.append(cleaned)
        rms_out_list.append(float(np.sqrt(np.mean(cleaned ** 2))))

        if verbose:
            raw_chunks.append(pipeline._last_raw)
            iir_chunks.append(pipeline._last_iir)
            asr_chunks.append(pipeline._last_asr)

    output = np.concatenate(chunks_out, axis=1)

    return RunResult(
        output=output,
        rms_input=np.array(rms_in_list),
        rms_output=np.array(rms_out_list),
        chunk_size=chunk_size,
        n_channels=n_channels,
        n_samples=n_samples,
        raw=np.concatenate(raw_chunks, axis=1) if verbose else None,
        iir=np.concatenate(iir_chunks, axis=1) if verbose else None,
        asr=np.concatenate(asr_chunks, axis=1) if verbose else None,
    )

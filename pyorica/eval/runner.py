"""Simulated real-time pipeline runner."""

from dataclasses import dataclass

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
    """
    output: np.ndarray
    rms_input: np.ndarray
    rms_output: np.ndarray
    chunk_size: int
    n_channels: int
    n_samples: int


def run(pipeline, data, chunk_size=64, calibration_data=None):
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

    for chunk in stream:
        rms_in_list.append(float(np.sqrt(np.mean(chunk ** 2))))
        cleaned = pipeline.process(chunk)
        chunks_out.append(cleaned)
        rms_out_list.append(float(np.sqrt(np.mean(cleaned ** 2))))

    output = np.concatenate(chunks_out, axis=1)
    return RunResult(
        output=output,
        rms_input=np.array(rms_in_list),
        rms_output=np.array(rms_out_list),
        chunk_size=chunk_size,
        n_channels=n_channels,
        n_samples=n_samples,
    )

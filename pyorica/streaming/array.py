"""ArrayStream: replay a numpy array as fixed-size EEG chunks."""

import numpy as np


class ArrayStream:
    """Replay a (channels × samples) array as fixed-size chunks.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
    chunk_size : int
        Number of samples per yielded chunk.
    """

    def __init__(self, data: np.ndarray, chunk_size: int) -> None:
        if data.ndim != 2:
            raise ValueError("data must be 2-D (channels × samples)")
        self.data = data
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        """Number of chunks that will be yielded (including remainder)."""
        return int(np.ceil(self.data.shape[1] / self.chunk_size))

    def __iter__(self):
        n_samples = self.data.shape[1]
        start = 0
        while start < n_samples:
            end = min(start + self.chunk_size, n_samples)
            yield self.data[:, start:end]
            start = end

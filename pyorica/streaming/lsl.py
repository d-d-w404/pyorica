"""LSLStream: pylsl inlet that yields (channels × samples) chunks."""

import numpy as np


class LSLStream:
    """Real-time EEG stream from a Lab Streaming Layer outlet.

    Iterates indefinitely; break out of the loop to stop.

    Parameters
    ----------
    stream_name : str
        Name of the LSL stream to connect to.
    chunk_size : int
        Number of samples to request per pull (default 64).
    timeout : float
        Seconds to wait for the stream to appear (default 5.0).

    Attributes
    ----------
    n_channels : int
    sfreq : float
    chunk_size : int

    Requires ``pip install pyorica[lsl]``.
    """

    def __init__(self, stream_name, chunk_size=64, timeout=5.0):
        import pylsl

        streams = pylsl.resolve_byprop('name', stream_name, timeout=timeout)
        if not streams:
            raise RuntimeError(
                f"LSL stream '{stream_name}' not found within {timeout:.1f}s"
            )
        self._inlet = pylsl.StreamInlet(streams[0])
        info = self._inlet.info()
        self.n_channels = info.channel_count()
        self.sfreq = info.nominal_srate()
        self.chunk_size = chunk_size

    def __iter__(self):
        return self._generate()

    def _generate(self):
        while True:
            samples, _ = self._inlet.pull_chunk(
                timeout=1.0, max_samples=self.chunk_size
            )
            if samples:
                # pylsl returns list-of-lists (n_samples × n_channels); transpose
                chunk = np.array(samples, dtype=np.float64).T
                yield chunk

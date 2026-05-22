"""EEG data streams. ArrayStream requires no extras; LSLStream requires pyorica[lsl]."""

from pyorica.streaming.array import ArrayStream

__all__ = ["ArrayStream"]

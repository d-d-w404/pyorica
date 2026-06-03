"""Receive an LSL EEG stream and run the pyorica pipeline on it.

Matches the broadcaster in ORICA/code/aa_lsl_npz.py (stream name "mybrain").

Run:
    set PYORICA_ASR_NPZ=D:/path/to/s02_resampled.npz
    .\.venv\Scripts\python.exe lsl_receive.py

    .\.venv\Scripts\python.exe lsl_receive.py --asr-npz D:/path/to/s02_resampled.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from pyorica.pipeline import EEGPipeline
from pyorica.streaming.lsl import LSLStream

STREAM_NAME = "mybrain"
CHUNK_SIZE = 50
CALIBRATION_SECONDS = 120.0


def main() -> None:
    parser = argparse.ArgumentParser(description="LSL receiver with pyorica pipeline")
    parser.add_argument(
        "--asr-npz",
        default=os.environ.get("PYORICA_ASR_NPZ"),
        help="External ASR calibration NPZ (or set PYORICA_ASR_NPZ).",
    )
    args = parser.parse_args()
    if not args.asr_npz:
        print("ERROR: ASR requires external NPZ. Pass --asr-npz or set PYORICA_ASR_NPZ.",
              file=sys.stderr)
        sys.exit(1)

    print(f"connecting to LSL stream {STREAM_NAME!r} ...")
    try:
        stream = LSLStream(STREAM_NAME, chunk_size=CHUNK_SIZE, timeout=15.0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        print("make sure the broadcaster is running first.")
        sys.exit(1)

    n_ch = stream.n_channels
    sfreq = stream.sfreq
    print(f"connected: {n_ch} ch @ {sfreq} Hz, chunk={CHUNK_SIZE}")

    target = int(CALIBRATION_SECONDS * sfreq)
    print(f"collecting {CALIBRATION_SECONDS:.0f}s ({target} samples) for ORICA warm-start...")
    calib_chunks: list[np.ndarray] = []
    collected = 0
    t0 = time.monotonic()
    for chunk in stream:
        calib_chunks.append(chunk)
        collected += chunk.shape[1]
        if collected >= target:
            break
    calib = np.concatenate(calib_chunks, axis=1)
    print(f"calibration data shape: {calib.shape}  (took {time.monotonic() - t0:.1f}s)")

    pipeline = EEGPipeline(n_channels=n_ch, sfreq=sfreq)
    pipeline.fit(calib, asr_calibration_npz=args.asr_npz)
    print(f"pipeline ready. ASR enabled: {pipeline._asr_fitted}")
    print("processing live ... Ctrl+C to stop\n")

    sfreq_int = int(round(sfreq))
    n = 0
    sec_printed = 0
    try:
        for chunk in stream:
            cleaned = pipeline.process(chunk)
            n += chunk.shape[1]
            current_sec = n // sfreq_int
            if current_sec > sec_printed:
                sec_printed = current_sec
                rms_in = float(np.sqrt((chunk ** 2).mean()))
                rms_out = float(np.sqrt((cleaned ** 2).mean()))
                reduction_db = 20.0 * np.log10((rms_in + 1e-12) / (rms_out + 1e-12))
                print(f"  t={sec_printed:4d}s  RMS in={rms_in:7.3f}  "
                      f"out={rms_out:7.3f}  reduction={reduction_db:+5.2f} dB")
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

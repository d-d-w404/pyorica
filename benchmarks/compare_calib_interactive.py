"""Interactive comparison of two ASR calibration NPZ files.

Opens a matplotlib window with channel / time / window sliders. Top panel
overlays both traces; bottom panel shows A − B.

Usage
-----
    python benchmarks/compare_calib_interactive.py

    python benchmarks/compare_calib_interactive.py \\
        --a benchmarks/result/all/s03_iclabel_interval__asr_fit_self/s03_asr_calib_leadin.npz \\
        --b D:/work/Python_Project/ORICA/data/Input_data/asr_cali/Shawn_shared/2min/s03_resampled.npz \\
        --label-a "self2 lead-in (IIR)" \\
        --label-b "2min external NPZ"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_calib_npz(path: Path) -> dict[str, Any]:
    """Load calibration array and metadata from an ORICA-style NPZ."""
    z = np.load(path, allow_pickle=True)
    data = None
    for key in ("calibration_data", "data", "eeg_data"):
        if key in z.files:
            data = np.asarray(z[key], dtype=np.float64)
            break
    if data is None:
        raise KeyError(
            f"No calibration array in {path}; keys: {list(z.files)}"
        )
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D array in {path}, got shape {data.shape}")
    if data.shape[0] > data.shape[1] * 10:
        data = data.T

    ch_names: list[str] = []
    for key in ("ch_names", "channels"):
        if key not in z.files:
            continue
        try:
            ch_names = [str(x) for x in np.asarray(z[key]).ravel()]
            break
        except Exception as exc:
            print(
                f"WARNING: could not read {key!r} from {path} ({exc}); "
                "using generic channel labels.",
                file=sys.stderr,
            )

    sfreq = 250.0
    for key in ("sfreq", "sampling_rate"):
        if key in z.files:
            sfreq = float(z[key])
            break

    calib_sec = data.shape[1] / sfreq
    if "calibration_seconds" in z.files:
        calib_sec = float(z["calibration_seconds"])

    skip = {"calibration_data", "data", "eeg_data", "ch_names", "channels"}
    meta: dict[str, Any] = {}
    for k in z.files:
        if k in skip:
            continue
        try:
            meta[k] = z[k]
        except Exception:
            pass

    return {
        "data": data,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "calib_seconds": calib_sec,
        "path": path,
        "meta": meta,
    }


def align_calib(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, list[str], float, dict]:
    """Truncate to common (n_ch, n_samples)."""
    da, db = a["data"], b["data"]
    n_ch = min(da.shape[0], db.shape[0])
    n_samp = min(da.shape[1], db.shape[1])
    names_a = a["ch_names"]
    names_b = b["ch_names"]
    if names_a and len(names_a) >= n_ch:
        ch_names = names_a[:n_ch]
    elif names_b and len(names_b) >= n_ch:
        ch_names = names_b[:n_ch]
    else:
        ch_names = [f"ch{i}" for i in range(n_ch)]

    sfreq = a["sfreq"]
    if abs(a["sfreq"] - b["sfreq"]) > 0.01:
        print(
            f"WARNING: sfreq mismatch A={a['sfreq']} B={b['sfreq']}; using A.",
            file=sys.stderr,
        )

    info = {
        "n_ch_a": da.shape[0],
        "n_ch_b": db.shape[0],
        "n_samp_a": da.shape[1],
        "n_samp_b": db.shape[1],
        "n_ch_used": n_ch,
        "n_samp_used": n_samp,
        "dur_a_s": da.shape[1] / a["sfreq"],
        "dur_b_s": db.shape[1] / b["sfreq"],
    }
    return da[:n_ch, :n_samp], db[:n_ch, :n_samp], ch_names, sfreq, info


def print_global_stats(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    sfreq: float,
) -> None:
    diff = a - b
    n = a.shape[1]
    print(f"\nGlobal stats ({a.shape[0]} ch × {n} samples, {n / sfreq:.1f} s @ {sfreq} Hz)")
    print(f"  max |diff|: {np.max(np.abs(diff)):.6g}")
    print(f"  RMS(diff):  {np.sqrt(np.mean(diff ** 2)):.6g}")
    print(f"  corr:       {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.6f}")
    print(f"  {label_a}  mean={a.mean():.4f}  std={a.std():.4f}")
    print(f"  {label_b}  mean={b.mean():.4f}  std={b.std():.4f}")


def run_viewer(
    data_a: np.ndarray,
    data_b: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    label_a: str,
    label_b: str,
    align_info: dict,
    path_a: Path,
    path_b: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    n_ch, n_samples = data_a.shape
    duration_s = n_samples / sfreq

    fig, (ax_sig, ax_diff) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.28, top=0.90)

    (line_a,) = ax_sig.plot([], [], color="#1f77b4", lw=0.9, label=label_a)
    (line_b,) = ax_sig.plot([], [], color="#ff7f0e", lw=0.9, alpha=0.85, label=label_b)
    (line_diff,) = ax_diff.plot([], [], color="#d62728", lw=0.9)
    ax_sig.legend(loc="upper right")
    ax_sig.set_ylabel("Amplitude (µV)")
    ax_diff.set_ylabel(f"Δ ({label_a} − {label_b})")
    ax_diff.set_xlabel("Time (s)")
    ax_sig.grid(True, alpha=0.3)
    ax_diff.grid(True, alpha=0.3)

    title = fig.suptitle("", fontsize=11)

    state = {"channel": 0, "time_s": 0.0, "window_s": 4.0}

    def _slice(ch: int, t0_s: float, win_s: float):
        t0 = max(0.0, t0_s)
        t1 = min(duration_s, t0 + win_s)
        i0 = int(t0 * sfreq)
        i1 = max(i0 + 2, int(t1 * sfreq))
        i1 = min(i1, n_samples)
        times = np.arange(i0, i1) / sfreq
        ya = data_a[ch, i0:i1]
        yb = data_b[ch, i0:i1]
        return times, ya, yb, ya - yb

    def refresh(_=None):
        ch = state["channel"]
        ch_label = ch_names[ch] if ch < len(ch_names) else f"ch{ch}"
        times, ya, yb, diff = _slice(ch, state["time_s"], state["window_s"])

        line_a.set_data(times, ya)
        line_b.set_data(times, yb)
        line_diff.set_data(times, diff)

        ax_sig.relim()
        ax_sig.autoscale_view()
        ax_diff.relim()
        ax_diff.autoscale_view()

        rms_diff = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
        corr = float(np.corrcoef(ya, yb)[0, 1]) if diff.size > 1 else float("nan")
        title.set_text(
            f"ch={ch} ({ch_label})  t={state['time_s']:.2f}s  "
            f"window={state['window_s']:.1f}s  "
            f"RMS(Δ)={rms_diff:.4f}  corr={corr:.4f}"
        )
        fig.canvas.draw_idle()

    ax_ch = fig.add_axes([0.30, 0.20, 0.55, 0.03])
    slider_ch = Slider(ax_ch, "Channel", 0, n_ch - 1, valinit=0, valstep=1)
    slider_ch.on_changed(lambda v: (state.update({"channel": int(v)}), refresh()))

    ax_time = fig.add_axes([0.30, 0.14, 0.55, 0.03])
    slider_time = Slider(
        ax_time, "Time (s)", 0.0, max(0.0, duration_s - 0.01), valinit=0.0, valstep=0.1,
    )
    slider_time.on_changed(lambda v: (state.update({"time_s": float(v)}), refresh()))

    ax_win = fig.add_axes([0.30, 0.08, 0.55, 0.03])
    max_win = min(30.0, duration_s)
    slider_win = Slider(
        ax_win, "Window (s)", 0.5, max_win, valinit=min(4.0, max_win), valstep=0.5,
    )
    slider_win.on_changed(lambda v: (state.update({"window_s": float(v)}), refresh()))

    note = (
        f"A: {path_a.name}  ({align_info['n_samp_a']} samples, "
        f"{align_info['dur_a_s']:.1f} s)\n"
        f"B: {path_b.name}  ({align_info['n_samp_b']} samples, "
        f"{align_info['dur_b_s']:.1f} s)\n"
        f"Aligned: {align_info['n_ch_used']} ch × {align_info['n_samp_used']} samples "
        f"({duration_s:.1f} s @ {sfreq:.0f} Hz)"
    )
    if align_info["n_samp_a"] != align_info["n_samp_b"]:
        note += "  | truncated to shorter length"
    fig.text(0.12, 0.02, note, fontsize=9, color="#444")

    refresh()
    plt.show()


def main() -> None:
    root = _repo_root()
    default_a = (
        root / "benchmarks/result/all/s03_iclabel_interval__asr_fit_self3"
        / "s03_asr_calib_leadin.npz"
    )
    default_b = Path(
        r"D:/work/Python_Project/ORICA/data/Input_data/asr_cali"
        r"/Shawn_shared/2min/s03_resampled.npz"
    )

    parser = argparse.ArgumentParser(
        description="Interactive overlay viewer for two ASR calibration NPZ files.",
    )
    parser.add_argument(
        "--a", "--npz-a", dest="path_a", type=Path, default=default_a,
        help="First calibration NPZ (blue trace).",
    )
    parser.add_argument(
        "--b", "--npz-b", dest="path_b", type=Path, default=default_b,
        help="Second calibration NPZ (orange trace).",
    )
    parser.add_argument(
        "--label-a", default="self2 lead-in",
        help="Legend label for --a.",
    )
    parser.add_argument(
        "--label-b", default="2min external NPZ",
        help="Legend label for --b.",
    )
    args = parser.parse_args()

    if not args.path_a.is_file():
        print(f"ERROR: NPZ A not found: {args.path_a}", file=sys.stderr)
        sys.exit(1)
    if not args.path_b.is_file():
        print(f"ERROR: NPZ B not found: {args.path_b}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading A → {args.path_a}")
    cal_a = load_calib_npz(args.path_a)
    print(f"Loading B → {args.path_b}")
    cal_b = load_calib_npz(args.path_b)

    data_a, data_b, ch_names, sfreq, align_info = align_calib(cal_a, cal_b)
    print_global_stats(data_a, data_b, args.label_a, args.label_b, sfreq)

    run_viewer(
        data_a, data_b, ch_names, sfreq,
        args.label_a, args.label_b, align_info,
        args.path_a, args.path_b,
    )


if __name__ == "__main__":
    main()

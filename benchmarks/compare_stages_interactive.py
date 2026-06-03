"""Interactive comparison: pyorica stage arrays vs ORICA receiver NPZ exports.

Loads pyorica ``{subject}_stages.npz`` (keys raw/iir/asr/orica) and the matching
ORICA ``{tag}eeg_{stage}1.npz`` files, then opens a matplotlib window with:

- stage selector (raw / iir / asr / orica)
- channel selector
- time slider (seconds)
- window-length slider (seconds of data shown)

Top panel: pyorica (blue) and ORICA (orange) overlaid.
Bottom panel: difference (pyorica − ORICA).

Usage
-----
    python benchmarks/compare_stages_interactive.py

    python benchmarks/compare_stages_interactive.py \\
        --pyorica-npz benchmarks/result/all/iclabel_2_new2/s02_stages.npz \\
        --orica-dir D:/work/Python_Project/ORICA/data/output_data/SDriveasrpy20_2min_70 \\
        --orica-tag bs2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

STAGES = ("raw", "iir", "asr", "orica")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_pyorica_stages(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    missing = [s for s in STAGES if s not in z.files]
    if missing:
        raise KeyError(f"{path} missing keys: {missing} (have {list(z.files)})")
    ch_names = [str(x) for x in np.asarray(z["ch_names"]).ravel()]
    sfreq = float(z["sfreq"])
    calib = int(z["calib_samples"]) if "calib_samples" in z.files else 0
    stages = {s: np.asarray(z[s], dtype=np.float64) for s in STAGES}
    return {
        "stages": stages,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "calib_samples": calib,
    }


def load_orica_stage(path: Path) -> tuple[np.ndarray, list[str], float]:
    z = np.load(path, allow_pickle=True)
    data = np.asarray(z["data"], dtype=np.float64)
    ch_names = [str(x) for x in np.asarray(z["channels"]).ravel()]
    sfreq = float(z["sampling_rate"]) if "sampling_rate" in z.files else 250.0
    return data, ch_names, sfreq


def load_orica_all(orica_dir: Path, tag: str) -> dict:
    stages = {}
    ch_names = None
    sfreq = None
    for stage in STAGES:
        path = orica_dir / f"{tag}eeg_{stage}1.npz"
        if not path.is_file():
            raise FileNotFoundError(f"ORICA stage file not found: {path}")
        data, names, sr = load_orica_stage(path)
        stages[stage] = data
        ch_names = names
        sfreq = sr
    return {"stages": stages, "ch_names": ch_names or [], "sfreq": sfreq or 250.0}


def align_stages(pyo: dict, ori: dict) -> tuple[dict, dict, int, dict]:
    """Truncate both sides to common (n_ch, n_samples); warn on mismatch."""
    info = {}
    n_ch = min(pyo["stages"]["raw"].shape[0], ori["stages"]["raw"].shape[0])
    n_samp = min(pyo["stages"]["raw"].shape[1], ori["stages"]["raw"].shape[1])
    info["n_ch_pyo"] = pyo["stages"]["raw"].shape[0]
    info["n_ch_ori"] = ori["stages"]["raw"].shape[0]
    info["n_samp_pyo"] = pyo["stages"]["raw"].shape[1]
    info["n_samp_ori"] = ori["stages"]["raw"].shape[1]
    info["n_ch_used"] = n_ch
    info["n_samp_used"] = n_samp

    pyo_a = {s: pyo["stages"][s][:n_ch, :n_samp] for s in STAGES}
    ori_a = {s: ori["stages"][s][:n_ch, :n_samp] for s in STAGES}
    return pyo_a, ori_a, n_samp, info


def run_viewer(
    pyo_stages: dict,
    ori_stages: dict,
    ch_names: list[str],
    sfreq: float,
    n_samples: int,
    align_info: dict,
    calib_samples: int = 0,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider

    n_ch = pyo_stages["raw"].shape[0]
    duration_s = n_samples / sfreq

    fig, (ax_sig, ax_diff) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.28, top=0.92)

    (line_pyo,) = ax_sig.plot([], [], color="#1f77b4", lw=0.9, label="pyorica")
    (line_ori,) = ax_sig.plot([], [], color="#ff7f0e", lw=0.9, alpha=0.85, label="ORICA")
    (line_diff,) = ax_diff.plot([], [], color="#d62728", lw=0.9)
    ax_sig.legend(loc="upper right")
    ax_sig.set_ylabel("Amplitude (µV)")
    ax_diff.set_ylabel("Δ (pyo − ORI)")
    ax_diff.set_xlabel("Time (s)")
    ax_sig.grid(True, alpha=0.3)
    ax_diff.grid(True, alpha=0.3)

    title = fig.suptitle("", fontsize=11)

    state = {
        "stage": "raw",
        "channel": 0,
        "time_s": 0.0,
        "window_s": 4.0,
    }

    def _slice(stage: str, ch: int, t0_s: float, win_s: float):
        t0 = max(0.0, t0_s)
        t1 = min(duration_s, t0 + win_s)
        i0 = int(t0 * sfreq)
        i1 = max(i0 + 2, int(t1 * sfreq))
        i1 = min(i1, n_samples)
        times = np.arange(i0, i1) / sfreq
        pyo = pyo_stages[stage][ch, i0:i1]
        ori = ori_stages[stage][ch, i0:i1]
        return times, pyo, ori, pyo - ori

    def refresh(_=None):
        stage = state["stage"]
        ch = state["channel"]
        ch_label = ch_names[ch] if ch < len(ch_names) else f"ch{ch}"
        times, pyo, ori, diff = _slice(stage, ch, state["time_s"], state["window_s"])

        line_pyo.set_data(times, pyo)
        line_ori.set_data(times, ori)
        line_diff.set_data(times, diff)

        ax_sig.relim()
        ax_sig.autoscale_view()
        ax_diff.relim()
        ax_diff.autoscale_view()

        rms_diff = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
        corr = float(np.corrcoef(pyo, ori)[0, 1]) if diff.size > 1 else float("nan")
        title.set_text(
            f"stage={stage}  ch={ch} ({ch_label})  "
            f"t={state['time_s']:.2f}s  window={state['window_s']:.1f}s  "
            f"RMS(Δ)={rms_diff:.4f}  corr={corr:.4f}"
        )
        fig.canvas.draw_idle()

    # ── widgets ──────────────────────────────────────────────────────────
    ax_stage = fig.add_axes([0.12, 0.16, 0.12, 0.10])
    rb_stage = RadioButtons(ax_stage, STAGES, active=0)
    rb_stage.on_clicked(lambda label: (state.update({"stage": label}), refresh()))

    ax_ch = fig.add_axes([0.30, 0.20, 0.55, 0.03])
    slider_ch = Slider(
        ax_ch, "Channel", 0, n_ch - 1, valinit=0, valstep=1,
    )
    slider_ch.on_changed(lambda v: (state.update({"channel": int(v)}), refresh()))

    ax_time = fig.add_axes([0.30, 0.14, 0.55, 0.03])
    slider_time = Slider(
        ax_time, "Time (s)", 0.0, max(0.0, duration_s - 0.01), valinit=0.0, valstep=0.1,
    )
    slider_time.on_changed(lambda v: (state.update({"time_s": float(v)}), refresh()))

    ax_win = fig.add_axes([0.30, 0.08, 0.55, 0.03])
    slider_win = Slider(
        ax_win, "Window (s)", 0.5, min(30.0, duration_s), valinit=4.0, valstep=0.5,
    )
    slider_win.on_changed(lambda v: (state.update({"window_s": float(v)}), refresh()))

    note = (
        f"Aligned: {align_info['n_ch_used']} ch × {align_info['n_samp_used']} samples "
        f"({duration_s:.1f} s @ {sfreq:.0f} Hz)"
    )
    if align_info["n_samp_pyo"] != align_info["n_samp_ori"]:
        note += (
            f"  | truncated pyo {align_info['n_samp_pyo']} / "
            f"ORI {align_info['n_samp_ori']}"
        )
    if calib_samples:
        note += f"  | pyorica calib lead-in: {calib_samples / sfreq:.0f} s"
    fig.text(0.12, 0.02, note, fontsize=9, color="#444")

    refresh()
    plt.show()


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Interactive pyorica vs ORICA stage comparison viewer.",
    )
    parser.add_argument(
        "--pyorica-npz",
        type=Path,
        default=root / "benchmarks/result/all/t3/s02_stages.npz",
        help="pyorica stages NPZ (raw/iir/asr/orica keys).",
    )
    parser.add_argument(
        "--orica-dir",
        type=Path,
        default=Path(
            r"D:/work/Python_Project/ORICA/data/output_data/SDriveasrpy20_2min_70"
        ),
        help="Directory containing ORICA bs2eeg_*1.npz exports.",
    )
    parser.add_argument(
        "--orica-tag",
        default="bs2",
        help="File tag prefix before 'eeg_raw1.npz' (default: bs2).",
    )
    args = parser.parse_args()

    if not args.pyorica_npz.is_file():
        print(f"ERROR: pyorica NPZ not found: {args.pyorica_npz}", file=sys.stderr)
        sys.exit(1)
    if not args.orica_dir.is_dir():
        print(f"ERROR: ORICA dir not found: {args.orica_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading pyorica → {args.pyorica_npz}")
    pyo = load_pyorica_stages(args.pyorica_npz)
    print(f"Loading ORICA   → {args.orica_dir} ({args.orica_tag}eeg_*1.npz)")
    ori = load_orica_all(args.orica_dir, args.orica_tag)

    pyo_a, ori_a, n_samp, align_info = align_stages(pyo, ori)
    sfreq = pyo["sfreq"]
    ch_names = pyo["ch_names"] or ori["ch_names"]

    print(
        f"Ready: {align_info['n_ch_used']} ch, {n_samp} samples, "
        f"{n_samp / sfreq:.1f} s @ {sfreq} Hz"
    )
    if align_info["n_samp_pyo"] != align_info["n_samp_ori"]:
        print(
            f"NOTE: sample count mismatch — using first {n_samp} samples "
            f"(pyo {align_info['n_samp_pyo']}, ORI {align_info['n_samp_ori']})."
        )

    run_viewer(
        pyo_a, ori_a, ch_names, sfreq, n_samp, align_info,
        calib_samples=pyo.get("calib_samples", 0),
    )


if __name__ == "__main__":
    main()

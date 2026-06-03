"""Quantitative pyorica vs ORICA stage comparison report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STAGES = ("raw", "iir", "asr", "orica")


def load_orica_stage(orica_dir: Path, tag: str, stage: str) -> np.ndarray:
    z = np.load(orica_dir / f"{tag}eeg_{stage}1.npz")
    return np.asarray(z["data"], dtype=np.float64)


def load_pyorica_stages(npz_path: Path) -> dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    return {s: np.asarray(z[s], dtype=np.float64) for s in STAGES}


def align(pyo: dict, ori: dict) -> tuple[dict, dict, int]:
    n = min(*(pyo[s].shape[1] for s in STAGES), *(ori[s].shape[1] for s in STAGES))
    return ({s: pyo[s][:, :n] for s in STAGES}, {s: ori[s][:, :n] for s in STAGES}, n)


def stage_stats(pyo: np.ndarray, ori: np.ndarray) -> dict:
    d = pyo - ori
    rms = float(np.sqrt(np.mean(d**2)))
    corr = float(np.corrcoef(pyo.ravel(), ori.ravel())[0, 1])
    return {"rms": rms, "corr": corr, "max_abs": float(np.max(np.abs(d)))}


def asr_cleanup_rms(asr: np.ndarray, iir: np.ndarray) -> np.ndarray:
    """Per-channel RMS of ASR change relative to IIR, averaged over channels."""
    n = asr.shape[1]
    win = 250
    out = np.zeros(n // win)
    for i in range(len(out)):
        sl = slice(i * win, (i + 1) * win)
        out[i] = float(np.sqrt(np.mean((asr[:, sl] - iir[:, sl]) ** 2)))
    return out


def analyze_asr_windows(pyo_a: dict, ori_a: dict, sfreq: float = 250.0) -> dict:
    win = int(sfreq)
    n = pyo_a["asr"].shape[1]
    n_win = n // win
    d = pyo_a["asr"] - ori_a["asr"]
    pyo_chg = asr_cleanup_rms(pyo_a["asr"], pyo_a["iir"])
    ori_chg = asr_cleanup_rms(ori_a["asr"], ori_a["iir"])

    both_idle = mid = both_clean_diff = only_ori = only_pyo = 0
    low_rms = mid_rms = high_rms = 0
    for i in range(n_win):
        rms_d = float(np.sqrt(np.mean(d[:, i * win : (i + 1) * win] ** 2)))
        if rms_d < 0.5:
            low_rms += 1
        elif rms_d < 2.0:
            mid_rms += 1
        else:
            high_rms += 1

        pyo_did = pyo_chg[i] > 1.0
        ori_did = ori_chg[i] > 1.0
        if not pyo_did and not ori_did:
            both_idle += 1
        elif pyo_did and ori_did and rms_d >= 2.0:
            both_clean_diff += 1
        elif ori_did and not pyo_did:
            only_ori += 1
        elif pyo_did and not ori_did:
            only_pyo += 1
        else:
            mid += 1

    return {
        "n_windows": n_win,
        "low_rms": low_rms,
        "mid_rms": mid_rms,
        "high_rms": high_rms,
        "both_idle": both_idle,
        "both_clean_diff": both_clean_diff,
        "only_ori": only_ori,
        "only_pyo": only_pyo,
        "other": mid,
    }


def print_report(label: str, pyo_path: Path, orica_dir: Path, tag: str) -> None:
    pyo = load_pyorica_stages(pyo_path)
    ori = {s: load_orica_stage(orica_dir, tag, s) for s in STAGES}
    pyo_a, ori_a, n = align(pyo, ori)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  pyorica: {pyo_path}")
    print(f"  ORICA:   {orica_dir}/{tag}eeg_*1.npz  (n={n})")
    print(f"{'=' * 60}")

    print("\nPer-stage global stats:")
    print(f"  {'stage':<8} {'RMS (uV)':>10} {'corr':>8} {'max|d| (uV)':>14}")
    for s in STAGES:
        st = stage_stats(pyo_a[s], ori_a[s])
        print(f"  {s:<8} {st['rms']:10.4f} {st['corr']:8.4f} {st['max_abs']:14.2f}")

    aw = analyze_asr_windows(pyo_a, ori_a)
    print("\nASR 1-second window analysis:")
    print(f"  low diff  (<0.5 uV): {aw['low_rms']} / {aw['n_windows']}")
    print(f"  mid diff  (0.5-2 uV): {aw['mid_rms']} / {aw['n_windows']}")
    print(f"  high diff (>=2 uV):  {aw['high_rms']} / {aw['n_windows']}")
    print(f"  both idle (no cleanup):     {aw['both_idle']}")
    print(f"  both cleaned but differ:    {aw['both_clean_diff']}")
    print(f"  only ORICA cleaned:         {aw['only_ori']}")
    print(f"  only pyorica cleaned:       {aw['only_pyo']}")

    # worst ASR second
    win = 250
    d = pyo_a["asr"] - ori_a["asr"]
    rms_per_s = [float(np.sqrt(np.mean(d[:, i * win : (i + 1) * win] ** 2))) for i in range(n // win)]
    worst = int(np.argmax(rms_per_s))
    print(f"\nWorst ASR second: t={worst}s, RMS={rms_per_s[worst]:.2f} uV")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyorica-npz", type=Path, required=True)
    parser.add_argument(
        "--orica-dir",
        type=Path,
        default=Path(r"D:/work/Python_Project/ORICA/data/output_data/SDriveasrpy20_2min_70"),
    )
    parser.add_argument("--orica-tag", default="bs2")
    parser.add_argument("--label", default="comparison")
    args = parser.parse_args()
    print_report(args.label, args.pyorica_npz, args.orica_dir, args.orica_tag)


if __name__ == "__main__":
    main()

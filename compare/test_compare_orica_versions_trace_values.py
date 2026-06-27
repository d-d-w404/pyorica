"""
Verbose ORICA implementation comparison tool.

Goal
----
Run OLD and NEW ORICA implementations on the same synthetic input and print
old/new values at every important numeric-changing step inside:

1. initialization
2. centering
3. RLS whitening block update
4. mixtures = sphere @ X
5. ICA block update
6. final outputs

PowerShell example:
    python test_compare_orica_versions_trace_values.py `
      --old "D:\work\Python_Project\ORICA\code\ORICA_final_no_print_quick30.py" `
      --new "D:\work\Python_Project\pyorica\pyorica\orica\core.py" `
      --n-ch 30 `
      --n-pts 33 `
      --block-size-white 32 `
      --block-size-ica 32 `
      --seed 7 `
      --max-white-blocks 3 `
      --max-ica-blocks 3

Useful options:
    --print-mode summary     only shape/stat/diff, recommended
    --print-mode head        summary + first few values
    --print-mode full        print full arrays, only use for tiny data
    --max-white-blocks -1    print all whitening blocks
    --max-ica-blocks -1      print all ICA blocks
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg


def load_module(path: str | Path, module_name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if "paths" not in sys.modules:
        dummy_paths = types.ModuleType("paths")
        dummy_paths.DEFAULT_DEMO_SET = Path("dummy.set")
        dummy_paths.TEMP_TXT_ROOT = Path(".")
        sys.modules["paths"] = dummy_paths

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_class(mod, preferred: list[str]):
    for name in preferred:
        if hasattr(mod, name):
            return getattr(mod, name)
    classes = [obj for _, obj in vars(mod).items() if inspect.isclass(obj)]
    raise RuntimeError(
        f"Could not find any of {preferred}. Classes found: {[c.__name__ for c in classes]}"
    )


def make_synthetic_eeg(n_ch: int, n_pts: int, seed: int, sfreq: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_pts, dtype=np.float64) / sfreq
    S = np.zeros((n_ch, n_pts), dtype=np.float64)
    freqs = np.linspace(3.0, 35.0, n_ch)
    for i, f in enumerate(freqs):
        S[i] = np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        S[i] += 0.3 * np.sin(2 * np.pi * (f / 2.0) * t + rng.uniform(0, 2 * np.pi))
        S[i] += 0.05 * rng.standard_normal(n_pts)
    A = rng.standard_normal((n_ch, n_ch))
    X = A @ S
    X *= 20.0
    return X.astype(np.float64, copy=False)


def instantiate_old(old_cls, args):
    kwargs = dict(
        n_components=args.n_ch,
        use_rls_whitening=True,
        block_size_white=args.block_size_white,
        block_size_ica=args.block_size_ica,
        tau_const=args.tau_const,
        gamma=args.gamma,
        lambda_0=args.lambda_0,
        srate=args.sfreq,
        time_perm=False,
    )
    sig = inspect.signature(old_cls)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return old_cls(**kwargs)


def instantiate_new(new_cls, args):
    kwargs = dict(
        n_components=args.n_ch,
        sfreq=args.sfreq,
        block_size_white=args.block_size_white,
        block_size_ica=args.block_size_ica,
        tau_const=args.tau_const,
        gamma=args.gamma,
        lambda_0=args.lambda_0,
        force_constant_lambda=True,
        time_perm=False,
        num_pass=args.num_pass,
    )
    sig = inspect.signature(new_cls)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return new_cls(**kwargs)


def arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def stats(x: Any) -> str:
    a = arr(x)
    if a.size == 0:
        return f"shape={a.shape}, empty"
    return (
        f"shape={a.shape}, min={np.nanmin(a):.6e}, max={np.nanmax(a):.6e}, "
        f"mean={np.nanmean(a):.6e}, fro={np.linalg.norm(a):.6e}"
    )


def diff_stats(old: Any, new: Any, atol: float, rtol: float) -> tuple[bool, str]:
    o = arr(old)
    n = arr(new)
    if o.shape != n.shape:
        return False, f"shape mismatch old={o.shape}, new={n.shape}"
    d = np.abs(o - n)
    ok = bool(np.allclose(o, n, atol=atol, rtol=rtol))
    if d.size == 0:
        return ok, "max_abs=0.000000e+00, mean_abs=0.000000e+00"
    return ok, f"max_abs={d.max():.6e}, mean_abs={d.mean():.6e}, fro_diff={np.linalg.norm(o-n):.6e}"


def head_values(x: Any, n: int = 8) -> str:
    a = arr(x).ravel()
    m = min(n, a.size)
    return np.array2string(a[:m], precision=6, suppress_small=False)


def print_pair(label: str, old: Any, new: Any, args):
    ok, ds = diff_stats(old, new, args.atol, args.rtol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    print(f"  diff: {ds}")
    print(f"  old : {stats(old)}")
    print(f"  new : {stats(new)}")
    if args.print_mode in ("head", "full"):
        print(f"  old head: {head_values(old, args.head_n)}")
        print(f"  new head: {head_values(new, args.head_n)}")
    if args.print_mode == "full":
        print("  old full:")
        print(np.array2string(arr(old), precision=args.precision, suppress_small=False))
        print("  new full:")
        print(np.array2string(arr(new), precision=args.precision, suppress_small=False))
    print()


def should_record(i: int, max_blocks: int) -> bool:
    return max_blocks < 0 or i < max_blocks


def get_old_initial_state(old, X: np.ndarray, args):
    """Reconstruct the exact initial state chosen by old orica_rls_whitening.

    This mirrors the old code's if/else initialization. np.random.seed(args.seed)
    has already been set before object creation/run, so old random Q is reproducible.
    """
    n_chs = X.shape[0]
    if getattr(old, "whitening_matrix", None) is not None and getattr(old, "W", None) is not None:
        return {
            "icasphere": np.array(old.whitening_matrix, dtype=np.float64, copy=True),
            "icaweights": np.array(old.W, dtype=np.float64, copy=True),
            "counter": getattr(old, "counter", np.nan),
        }

    # This follows the old code's native else branch.
    rand_mat = np.random.randn(n_chs, n_chs)
    Q, R = np.linalg.qr(rand_mat)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs

    return {
        "icasphere": Q,
        "icaweights": np.eye(n_chs, dtype=np.float64),
        "counter": 0,
    }


def old_lambda_const(old, args):
    return float(getattr(old, "lambda_const", args.lambda_const))


def old_gen_cooling(old, t, gamma, lambda_0):
    if hasattr(old, "gen_cooling_ff"):
        return arr(old.gen_cooling_ff(t, gamma, lambda_0))
    t_safe = np.maximum(t, 1e-10)
    return lambda_0 / np.power(t_safe, gamma)


def new_gen_cooling(new, t):
    if hasattr(new, "_gen_cooling_ff"):
        return arr(new._gen_cooling_ff(t))
    t_safe = np.maximum(t, 1e-10)
    return new.lambda_0 / np.power(t_safe, new.gamma)


def trace_old_stepwise(old, X: np.ndarray, args):
    trace: dict[str, Any] = {"white": [], "ica": []}
    data = X.astype(np.float64, copy=False)
    n_chs, n_pts = data.shape
    data_center = data - data.mean(axis=1, keepdims=True)
    trace["data_center"] = data_center.copy()

    state = get_old_initial_state(old, data, args)
    trace["initial_sphere"] = arr(state["icasphere"]).copy()
    trace["initial_W"] = arr(state["icaweights"]).copy()
    trace["initial_counter"] = np.array([state.get("counter", np.nan)], dtype=np.float64)

    numsplits = n_pts // args.block_size_white
    lc = old_lambda_const(old, args)

    for _it in range(args.num_pass):
        for bi in range(numsplits):
            start = int(bi * n_pts / numsplits)
            end = min(n_pts, int((bi + 1) * n_pts / numsplits))
            if start >= end:
                continue
            data_range_0 = np.arange(start, end)
            data_range_1 = data_range_0 + 1
            blockdata = data_center[:, data_range_0]

            sphere_before = arr(state["icasphere"]).copy()
            lambda_raw = old_gen_cooling(old, state["counter"] + data_range_1, args.gamma, args.lambda_0)
            lambda_used = np.full(len(data_range_1), lc, dtype=np.float64)
            v = sphere_before @ blockdata
            if hasattr(old, "snap_to_kbits"):
                v = arr(old.snap_to_kbits(v, k=38))
            lambda_avg = 1.0 - lambda_used[int(np.ceil(len(lambda_used) / 2)) - 1]
            QWhite = lambda_avg / (1.0 - lambda_avg) + (np.linalg.norm(v, "fro") ** 2) / len(data_range_1)
            if hasattr(old, "snap_to_kbits"):
                QWhite = float(np.asarray(old.snap_to_kbits(QWhite, k=38)))
            update_term = (v @ v.T) / blockdata.shape[1] / QWhite @ sphere_before
            sphere_after = (1.0 / lambda_avg) * (sphere_before - update_term)
            state["icasphere"] = sphere_after

            if should_record(bi, args.max_white_blocks):
                trace["white"].append({
                    "block": bi,
                    "range": np.array([start, end], dtype=np.float64),
                    "blockdata": blockdata.copy(),
                    "sphere_before": sphere_before,
                    "lambda_raw": lambda_raw.copy(),
                    "lambda_used": lambda_used.copy(),
                    "lambda_avg": np.array([lambda_avg], dtype=np.float64),
                    "v": v.copy(),
                    "QWhite": np.array([QWhite], dtype=np.float64),
                    "update_term": update_term.copy(),
                    "sphere_after": sphere_after.copy(),
                })

    mixtures = arr(state["icasphere"]) @ data
    trace["mixtures"] = mixtures.copy()

    perm_idx = np.random.permutation(n_pts) if getattr(old, "time_perm", False) else np.arange(n_pts)
    trace["perm_idx"] = perm_idx.astype(np.float64)

    block_size_orica = args.block_size_ica
    # Old experimental code often hard-coded 32. Use actual argument here only if caller asks.
    # The simple test's args.block_size_ica default is 32, so this matches quick30.
    num_block_orica = int(np.floor(n_pts / block_size_orica))

    for _it in range(args.num_pass):
        for bi in range(num_block_orica):
            start = int(bi * n_pts / numsplits)
            end = min(n_pts, int((bi + 1) * n_pts / numsplits))
            if start >= end:
                continue
            data_range_0 = np.arange(start, end)
            data_range_1 = data_range_0 + 1
            perm_range = perm_idx[data_range_0]
            blockdata = mixtures[:, perm_range]

            W_before = arr(state["icaweights"]).copy()
            Y = W_before @ blockdata
            kurtsign = state.get("kurtsign", getattr(old, "kurtosis_sign", np.ones(n_chs, dtype=bool)))
            F = np.empty_like(Y)
            F[kurtsign, :] = -2.0 * np.tanh(Y[kurtsign, :])
            F[~kurtsign, :] = np.tanh(Y[~kurtsign, :]) - Y[~kurtsign, :]
            model_fitness = np.eye(n_chs) + (Y @ F.T) / blockdata.shape[1]
            Rn_before = None if getattr(old, "Rn", None) is None else arr(old.Rn).copy()
            if getattr(old, "Rn", None) is None:
                Rn_after = model_fitness
            else:
                Rn_after = 0.99 * arr(old.Rn) + 0.01 * model_fitness
            old.Rn = Rn_after
            non_stat_idx = np.array([np.linalg.norm(Rn_after, "fro")], dtype=np.float64)

            lambda_raw = old_gen_cooling(old, state["counter"] + data_range_1, args.gamma, args.lambda_0)
            state["counter"] += blockdata.shape[1]
            lambda_used = np.full(len(data_range_1), lc, dtype=np.float64)
            lambda_prod = np.prod(1.0 / (1.0 - lambda_used))
            Q = 1.0 + lambda_used * (np.sum(F * Y, axis=0) - 1.0)
            if hasattr(old, "snap_to_kbits"):
                F_for_update = arr(old.snap_to_kbits(F, k=44))
            else:
                F_for_update = F
            W_pre_ortho = lambda_prod * (W_before - Y @ np.diag(lambda_used / Q) @ F_for_update.T @ W_before)
            D, V = scipy.linalg.eigh(W_pre_ortho @ W_pre_ortho.T)
            D_diag = np.diag(D)
            if hasattr(old, "snap_to_kbits"):
                D_diag = arr(old.snap_to_kbits(D_diag, k=32))
                V = arr(old.snap_to_kbits(V, k=32))
            d = np.diag(D_diag)
            M = V @ np.diag(1.0 / np.sqrt(d)) @ V.conj().T
            W_after = M @ W_pre_ortho
            if hasattr(old, "snap_to_kbits"):
                W_after = arr(old.snap_to_kbits(W_after, k=40))
            state["icaweights"] = W_after
            state["lambda_k"] = lambda_used
            state["Rn"] = Rn_after

            if should_record(bi, args.max_ica_blocks):
                trace["ica"].append({
                    "block": bi,
                    "range": np.array([start, end], dtype=np.float64),
                    "blockdata": blockdata.copy(),
                    "W_before": W_before,
                    "Y": Y.copy(),
                    "F": F.copy(),
                    "model_fitness": model_fitness.copy(),
                    "Rn_before": np.array([np.nan]) if Rn_before is None else Rn_before,
                    "Rn_after": Rn_after.copy(),
                    "non_stat_idx": non_stat_idx,
                    "lambda_raw": lambda_raw.copy(),
                    "counter_after": np.array([state["counter"]], dtype=np.float64),
                    "lambda_used": lambda_used.copy(),
                    "lambda_prod": np.array([lambda_prod], dtype=np.float64),
                    "Q": Q.copy(),
                    "F_for_update": F_for_update.copy(),
                    "W_pre_ortho": W_pre_ortho.copy(),
                    "eig_D_diag": D_diag.copy(),
                    "eig_V": V.copy(),
                    "orthogonalizer_M": M.copy(),
                    "W_after": W_after.copy(),
                })

    trace["final_sphere"] = arr(state["icasphere"]).copy()
    trace["final_W"] = arr(state["icaweights"]).copy()
    trace["final_sources"] = trace["final_W"] @ (trace["final_sphere"] @ data)
    trace["final_counter"] = np.array([state.get("counter", np.nan)], dtype=np.float64)
    return trace


def trace_new_stepwise(new, X: np.ndarray, args):
    trace: dict[str, Any] = {"white": [], "ica": []}
    data = X.astype(np.float64, copy=False)
    n_chs, n_pts = data.shape
    data_center = data - data.mean(axis=1, keepdims=True)
    trace["data_center"] = data_center.copy()

    trace["initial_sphere"] = arr(new.sphere_).copy()
    trace["initial_W"] = arr(new.weights_).copy()
    trace["initial_counter"] = np.array([getattr(new, "_counter", np.nan)], dtype=np.float64)

    numsplits = n_pts // args.block_size_white
    num_block_white = n_pts // args.block_size_white
    num_block_ica = n_pts // args.block_size_ica

    for _it in range(args.num_pass):
        for bi in range(num_block_white):
            start = int(bi * n_pts / numsplits)
            end = min(n_pts, int((bi + 1) * n_pts / numsplits))
            if start >= end:
                continue
            data_range_0 = np.arange(start, end)
            data_range_1 = data_range_0 + 1
            blockdata = data_center[:, data_range_0]

            sphere_before = arr(new.sphere_).copy()
            lambda_raw = new_gen_cooling(new, new._counter + data_range_1)
            lambda_used = arr(new._forgetting_factor(data_range_1))
            lambda_avg = 1.0 - lambda_used[int(np.ceil(len(lambda_used) / 2)) - 1]
            v = sphere_before @ blockdata
            if "_snap_to_kbits" in globals():
                pass
            # core.py's _snap_to_kbits is a module-level no-op, so this is fine.
            QWhite = lambda_avg / (1.0 - lambda_avg) + (np.linalg.norm(v, "fro") ** 2) / len(data_range_1)
            update_term = (v @ v.T) / blockdata.shape[1] / QWhite @ sphere_before
            sphere_after = (1.0 / lambda_avg) * (sphere_before - update_term)
            new._sphere = sphere_after

            if should_record(bi, args.max_white_blocks):
                trace["white"].append({
                    "block": bi,
                    "range": np.array([start, end], dtype=np.float64),
                    "blockdata": blockdata.copy(),
                    "sphere_before": sphere_before,
                    "lambda_raw": lambda_raw.copy(),
                    "lambda_used": lambda_used.copy(),
                    "lambda_avg": np.array([lambda_avg], dtype=np.float64),
                    "v": v.copy(),
                    "QWhite": np.array([QWhite], dtype=np.float64),
                    "update_term": update_term.copy(),
                    "sphere_after": sphere_after.copy(),
                })

    mixtures = arr(new.sphere_) @ data
    trace["mixtures"] = mixtures.copy()

    perm_idx = np.random.permutation(n_pts) if getattr(new, "time_perm", False) else np.arange(n_pts)
    trace["perm_idx"] = perm_idx.astype(np.float64)

    for _it in range(args.num_pass):
        for bi in range(num_block_ica):
            start = int(bi * n_pts / numsplits)
            end = min(n_pts, int((bi + 1) * n_pts / numsplits))
            if start >= end:
                continue
            data_range_0 = np.arange(start, end)
            data_range_1 = data_range_0 + 1
            perm_range = perm_idx[data_range_0]
            blockdata = mixtures[:, perm_range]

            W_before = arr(new.weights_).copy()
            Y = W_before @ blockdata
            kurtsign = new._kurtosis_sign
            F = np.empty_like(Y)
            F[kurtsign, :] = -2.0 * np.tanh(Y[kurtsign, :])
            F[~kurtsign, :] = np.tanh(Y[~kurtsign, :]) - Y[~kurtsign, :]
            model_fitness = np.eye(n_chs) + (Y @ F.T) / blockdata.shape[1]
            Rn_before = None if new._Rn is None else arr(new._Rn).copy()
            if new._Rn is None:
                Rn_after = model_fitness
            else:
                Rn_after = 0.99 * new._Rn + 0.01 * model_fitness
            new._Rn = Rn_after
            non_stat = float(np.linalg.norm(new._Rn, "fro"))
            if new._min_non_stat_idx is None:
                new._min_non_stat_idx = non_stat
            else:
                new._min_non_stat_idx = min(new._min_non_stat_idx, non_stat)
            non_stat_idx = np.array([non_stat], dtype=np.float64)

            lambda_raw = new_gen_cooling(new, new._counter + data_range_1)
            lambda_used = arr(new._forgetting_factor(data_range_1))
            new._counter += blockdata.shape[1]
            new._lambda_k = lambda_used
            lambda_prod = np.prod(1.0 / (1.0 - lambda_used))
            Q = 1.0 + lambda_used * (np.sum(F * Y, axis=0) - 1.0)
            F_for_update = F.copy()
            W_pre_ortho = lambda_prod * (W_before - Y @ np.diag(lambda_used / Q) @ F_for_update.T @ W_before)
            D, V = scipy.linalg.eigh(W_pre_ortho @ W_pre_ortho.T)
            D_diag = np.diag(D)
            d = np.diag(D_diag)
            M = V @ np.diag(1.0 / np.sqrt(d)) @ V.conj().T
            W_after = M @ W_pre_ortho
            new._W = W_after

            if should_record(bi, args.max_ica_blocks):
                trace["ica"].append({
                    "block": bi,
                    "range": np.array([start, end], dtype=np.float64),
                    "blockdata": blockdata.copy(),
                    "W_before": W_before,
                    "Y": Y.copy(),
                    "F": F.copy(),
                    "model_fitness": model_fitness.copy(),
                    "Rn_before": np.array([np.nan]) if Rn_before is None else Rn_before,
                    "Rn_after": Rn_after.copy(),
                    "non_stat_idx": non_stat_idx,
                    "lambda_raw": lambda_raw.copy(),
                    "counter_after": np.array([new._counter], dtype=np.float64),
                    "lambda_used": lambda_used.copy(),
                    "lambda_prod": np.array([lambda_prod], dtype=np.float64),
                    "Q": Q.copy(),
                    "F_for_update": F_for_update.copy(),
                    "W_pre_ortho": W_pre_ortho.copy(),
                    "eig_D_diag": D_diag.copy(),
                    "eig_V": V.copy(),
                    "orthogonalizer_M": M.copy(),
                    "W_after": W_after.copy(),
                })

    trace["final_sphere"] = arr(new.sphere_).copy()
    trace["final_W"] = arr(new.weights_).copy()
    trace["final_sources"] = new.weights_ @ (new.sphere_ @ data)
    trace["final_counter"] = np.array([getattr(new, "_counter", np.nan)], dtype=np.float64)
    return trace


def print_trace(old_trace, new_trace, args):
    print("=== Global / initialization ===")
    for key in ["data_center", "initial_sphere", "initial_W", "initial_counter"]:
        print_pair(key, old_trace[key], new_trace[key], args)

    print("=== Whitening trace ===")
    print_pair("number of recorded whitening blocks", np.array([len(old_trace["white"])]), np.array([len(new_trace["white"])]), args)
    for old_ev, new_ev in zip(old_trace["white"], new_trace["white"]):
        bi = int(old_ev["block"])
        print(f"--- Whitening block {bi} ---")
        for key in [
            "range", "blockdata", "sphere_before", "lambda_raw", "lambda_used",
            "lambda_avg", "v", "QWhite", "update_term", "sphere_after"
        ]:
            print_pair(f"white[{bi}].{key}", old_ev[key], new_ev[key], args)

    print("=== Mixtures / permutation ===")
    for key in ["mixtures", "perm_idx"]:
        print_pair(key, old_trace[key], new_trace[key], args)

    print("=== ICA trace ===")
    print_pair("number of recorded ICA blocks", np.array([len(old_trace["ica"])]), np.array([len(new_trace["ica"])]), args)
    for old_ev, new_ev in zip(old_trace["ica"], new_trace["ica"]):
        bi = int(old_ev["block"])
        print(f"--- ICA block {bi} ---")
        for key in [
            "range", "blockdata", "W_before", "Y", "F", "model_fitness",
            "Rn_before", "Rn_after", "non_stat_idx", "lambda_raw", "counter_after",
            "lambda_used", "lambda_prod", "Q", "F_for_update", "W_pre_ortho",
            "eig_D_diag", "eig_V", "orthogonalizer_M", "W_after"
        ]:
            print_pair(f"ica[{bi}].{key}", old_ev[key], new_ev[key], args)

    print("=== Final outputs ===")
    for key in ["final_sphere", "final_W", "final_sources", "final_counter"]:
        print_pair(key, old_trace[key], new_trace[key], args)


def main():
    p = argparse.ArgumentParser(description="Print old/new ORICA values at each numeric-changing step.")
    p.add_argument("--old", required=True, help="Path to old ORICA .py file")
    p.add_argument("--new", required=True, help="Path to new ORICA .py file")
    p.add_argument("--n-ch", type=int, default=8)
    p.add_argument("--n-pts", type=int, default=512)
    p.add_argument("--sfreq", type=float, default=500.0)
    p.add_argument("--block-size-white", type=int, default=8)
    p.add_argument("--block-size-ica", type=int, default=32)
    p.add_argument("--num-pass", type=int, default=1)
    p.add_argument("--tau-const", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=0.6)
    p.add_argument("--lambda-0", type=float, default=0.995)
    p.add_argument("--lambda-const", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument("--max-white-blocks", type=int, default=2, help="-1 means all blocks")
    p.add_argument("--max-ica-blocks", type=int, default=2, help="-1 means all blocks")
    p.add_argument("--print-mode", choices=["summary", "head", "full"], default="summary")
    p.add_argument("--head-n", type=int, default=8)
    p.add_argument("--precision", type=int, default=6)
    args = p.parse_args()

    old_mod = load_module(args.old, "orica_old_module")
    new_mod = load_module(args.new, "orica_new_module")
    old_cls = find_class(old_mod, ["ORICA_final_new", "ORICA_final", "ORICA"])
    new_cls = find_class(new_mod, ["ORICAFilter"])

    # This seed affects both synthetic data and old random native init if old uses np.random.randn.
    np.random.seed(args.seed)
    X = make_synthetic_eeg(args.n_ch, args.n_pts, args.seed, args.sfreq)

    print("Synthetic input")
    print(f"  X.shape={X.shape}, dtype={X.dtype}, seed={args.seed}")
    print(f"  old class={old_cls.__name__}")
    print(f"  new class={new_cls.__name__}")
    print(f"  block_size_white={args.block_size_white}, block_size_ica={args.block_size_ica}")
    print(f"  max_white_blocks={args.max_white_blocks}, max_ica_blocks={args.max_ica_blocks}")
    print()

    old = instantiate_old(old_cls, args)
    new = instantiate_new(new_cls, args)

    # Important: reset seed right before tracing old, so old's random Q is deterministic.
    np.random.seed(args.seed)
    old_trace = trace_old_stepwise(old, X, args)

    # Reset seed before tracing new too, so time_perm would be comparable if enabled later.
    np.random.seed(args.seed)
    new_trace = trace_new_stepwise(new, X, args)

    print_trace(old_trace, new_trace, args)


if __name__ == "__main__":
    main()

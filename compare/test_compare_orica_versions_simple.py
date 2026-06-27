"""
Simple ORICA implementation comparison tool.

This script respects each implementation's own initialization and normal public call path.
You only choose data size and common runtime parameters.

PowerShell example:
    python test_compare_orica_versions_simple.py `
      --old "D:\work\Python_Project\ORICA\code\ORICA_final_no_print_quick30.py" `
      --new "D:\work\Python_Project\pyorica\pyorica\orica\core.py" `
      --n-ch 30 `
      --n-pts 1000 `
      --block-size-white 32 `
      --block-size-ica 32 `
      --seed 7
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class CompareResult:
    name: str
    ok: bool
    max_abs: float
    mean_abs: float
    shape_old: tuple
    shape_new: tuple


def load_module(path: str | Path, module_name: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Some old experimental files import `paths`; provide a harmless dummy.
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


def compare(name: str, old: Any, new: Any, atol: float, rtol: float) -> CompareResult:
    old = np.asarray(old)
    new = np.asarray(new)
    if old.shape != new.shape:
        return CompareResult(name, False, np.inf, np.inf, old.shape, new.shape)
    diff = np.abs(old - new)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    ok = bool(np.allclose(old, new, atol=atol, rtol=rtol))
    return CompareResult(name, ok, max_abs, mean_abs, old.shape, new.shape)


def print_result(r: CompareResult):
    status = "PASS" if r.ok else "FAIL"
    print(f"[{status}] {r.name}")
    print(f"       shape old/new: {r.shape_old} vs {r.shape_new}")
    print(f"       max_abs={r.max_abs:.6e}, mean_abs={r.mean_abs:.6e}")


def get_old_counter(old) -> int | float:
    return getattr(old, "counter", np.nan)


def get_new_counter(new) -> int | float:
    return getattr(new, "_counter", np.nan)


def instantiate_old(old_cls, args):
    # Keep this permissive because old experimental constructors often differ.
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


def run_old_with_trace(old, X: np.ndarray, args):
    trace: dict[str, list[np.ndarray] | np.ndarray | float] = {
        "initial_sphere": [],
        "initial_W": [],
        "initial_counter": [],
        "sphere_after_white": [],
        "W_after_ica": [],
    }

    original_white = old.dynamic_whitening
    original_ica = old.dynamic_orica_cooling

    def white_wrapper(blockdata, data_range, state, lambda_const, gamma, lambda_0):
        if len(trace["initial_sphere"]) == 0:
            trace["initial_sphere"].append(np.array(state["icasphere"], copy=True))
            trace["initial_W"].append(np.array(state["icaweights"], copy=True))
            trace["initial_counter"].append(np.array([state.get("counter", np.nan)], dtype=np.float64))
        out = original_white(blockdata, data_range, state, lambda_const, gamma, lambda_0)
        trace["sphere_after_white"].append(np.array(out["icasphere"], copy=True))
        return out

    def ica_wrapper(blockdata, data_range, state=None, gamma=0.5, lambda_0=1.0):
        out = original_ica(blockdata, data_range, state, gamma, lambda_0)
        trace["W_after_ica"].append(np.array(out["icaweights"], copy=True))
        return out

    old.dynamic_whitening = white_wrapper
    old.dynamic_orica_cooling = ica_wrapper

    fit_kwargs = dict(
        block_size_white=args.block_size_white,
        num_pass=args.num_pass,
        lambda_0=args.lambda_0,
        gamma=args.gamma,
        verbose=False,
    )
    # Use old object's own lambda_const when available, instead of inventing a new one.
    if "lambda_const" in inspect.signature(old.fit).parameters:
        fit_kwargs["lambda_const"] = getattr(old, "lambda_const", args.lambda_const)

    sources, W, sphere = old.fit(X, **fit_kwargs)

    trace["final_sphere"] = np.array(sphere, copy=True)
    trace["final_W"] = np.array(W, copy=True)
    trace["final_sources"] = np.array(sources, copy=True)
    trace["transform_sources"] = np.array(old.transform(X), copy=True)
    trace["final_counter"] = np.array([get_old_counter(old)], dtype=np.float64)
    return trace


def run_new_with_trace(new, X: np.ndarray, args):
    trace: dict[str, list[np.ndarray] | np.ndarray | float] = {
        "initial_sphere": [np.array(new.sphere_, copy=True)],
        "initial_W": [np.array(new.weights_, copy=True)],
        "initial_counter": [np.array([get_new_counter(new)], dtype=np.float64)],
        "sphere_after_white": [],
        "W_after_ica": [],
    }

    original_white = new._dynamic_whitening
    original_ica = new._dynamic_orica

    def white_wrapper(blockdata, data_range):
        out = original_white(blockdata, data_range)
        trace["sphere_after_white"].append(np.array(new.sphere_, copy=True))
        return out

    def ica_wrapper(blockdata, data_range):
        out = original_ica(blockdata, data_range)
        trace["W_after_ica"].append(np.array(new.weights_, copy=True))
        return out

    new._dynamic_whitening = white_wrapper
    new._dynamic_orica = ica_wrapper

    fit_out = new.fit(X)
    sources = new.transform(X)

    trace["final_sphere"] = np.array(new.sphere_, copy=True)
    trace["final_W"] = np.array(new.weights_, copy=True)
    trace["final_sources"] = np.array(sources, copy=True)
    trace["transform_sources"] = np.array(new.transform(X), copy=True)
    trace["final_counter"] = np.array([get_new_counter(new)], dtype=np.float64)
    return trace


def compare_traces(old_trace, new_trace, args) -> list[CompareResult]:
    results: list[CompareResult] = []
    for key, label in [
        ("initial_sphere", "initial sphere"),
        ("initial_W", "initial W"),
        ("initial_counter", "initial counter"),
    ]:
        results.append(compare(label, old_trace[key][0], new_trace[key][0], args.atol, args.rtol))

    nw = min(len(old_trace["sphere_after_white"]), len(new_trace["sphere_after_white"]))
    if len(old_trace["sphere_after_white"]) != len(new_trace["sphere_after_white"]):
        results.append(compare(
            "number of whitening updates",
            np.array([len(old_trace["sphere_after_white"])]),
            np.array([len(new_trace["sphere_after_white"])]),
            args.atol,
            args.rtol,
        ))
    for i in range(nw):
        results.append(compare(
            f"whitening sphere after block {i}",
            old_trace["sphere_after_white"][i],
            new_trace["sphere_after_white"][i],
            args.atol,
            args.rtol,
        ))

    ni = min(len(old_trace["W_after_ica"]), len(new_trace["W_after_ica"]))
    if len(old_trace["W_after_ica"]) != len(new_trace["W_after_ica"]):
        results.append(compare(
            "number of ICA updates",
            np.array([len(old_trace["W_after_ica"])]),
            np.array([len(new_trace["W_after_ica"])]),
            args.atol,
            args.rtol,
        ))
    for i in range(ni):
        results.append(compare(
            f"ICA W after block {i}",
            old_trace["W_after_ica"][i],
            new_trace["W_after_ica"][i],
            args.atol,
            args.rtol,
        ))

    for key, label in [
        ("final_sphere", "final sphere"),
        ("final_W", "final W"),
        ("final_sources", "final sources"),
        ("transform_sources", "transform sources"),
        ("final_counter", "final counter"),
    ]:
        results.append(compare(label, old_trace[key], new_trace[key], args.atol, args.rtol))
    return results


def main():
    p = argparse.ArgumentParser(
        description="Compare old ORICA and new ORICA using each file's own initialization."
    )
    p.add_argument("--old", required=True, help="Path to old ORICA .py file")
    p.add_argument("--new", required=True, help="Path to new ORICA .py file")
    p.add_argument("--n-ch", type=int, default=8, help="Number of synthetic channels")
    p.add_argument("--n-pts", type=int, default=512, help="Number of synthetic samples")
    p.add_argument("--sfreq", type=float, default=500.0, help="Sampling frequency")
    p.add_argument("--block-size-white", type=int, default=8)
    p.add_argument("--block-size-ica", type=int, default=32)
    p.add_argument("--num-pass", type=int, default=1)
    p.add_argument("--tau-const", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=0.6)
    p.add_argument("--lambda-0", type=float, default=0.995)
    p.add_argument("--lambda-const", type=float, default=0.95, help="Fallback only if old object lacks lambda_const")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument("--max-print", type=int, default=80, help="Max detailed checks to print")
    args = p.parse_args()

    old_mod = load_module(args.old, "orica_old_module")
    new_mod = load_module(args.new, "orica_new_module")
    old_cls = find_class(old_mod, ["ORICA_final_new", "ORICA_final", "ORICA"])
    new_cls = find_class(new_mod, ["ORICAFilter"])

    np.random.seed(args.seed)  # affects old code if it uses np.random.randn internally
    X = make_synthetic_eeg(args.n_ch, args.n_pts, args.seed, args.sfreq)

    print("Synthetic input")
    print(f"  X.shape = {X.shape}")
    print(f"  X dtype  = {X.dtype}")
    print(f"  seed     = {args.seed}")
    print(f"  old class = {old_cls.__name__}")
    print(f"  new class = {new_cls.__name__}")
    print()

    old = instantiate_old(old_cls, args)
    new = instantiate_new(new_cls, args)

    old_trace = run_old_with_trace(old, X, args)
    new_trace = run_new_with_trace(new, X, args)

    print("Trace summary")
    print(f"  old whitening updates = {len(old_trace['sphere_after_white'])}")
    print(f"  new whitening updates = {len(new_trace['sphere_after_white'])}")
    print(f"  old ICA updates       = {len(old_trace['W_after_ica'])}")
    print(f"  new ICA updates       = {len(new_trace['W_after_ica'])}")
    print()

    results = compare_traces(old_trace, new_trace, args)
    n_fail = sum(not r.ok for r in results)

    print("Comparison")
    printed = 0
    first_fail_printed = False
    for r in results:
        should_print = printed < args.max_print or (not r.ok and not first_fail_printed)
        if should_print:
            print_result(r)
            printed += 1
            if not r.ok:
                first_fail_printed = True
    if len(results) > printed:
        print(f"... skipped {len(results) - printed} checks. Use --max-print {len(results)} to print all.")
    print()

    if n_fail == 0:
        print("OVERALL: PASS — old and new match using their own actual initialization/call path.")
        sys.exit(0)
    else:
        first = next(r for r in results if not r.ok)
        print(f"OVERALL: FAIL — {n_fail} checks did not match.")
        print(f"First mismatch: {first.name}")
        sys.exit(1)


if __name__ == "__main__":
    main()

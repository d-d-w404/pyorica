"""
Very verbose ORICA old-vs-new comparison tool.

What this script does
---------------------
1. Generates one synthetic EEG-like input X.
2. Prints/saves the original X used for testing.
3. Runs OLD and NEW ORICA implementations on exactly the same X.
4. Monkey-patches their real update functions, so it observes the actual code path.
5. At every important numeric-changing step, prints:
   - old summary
   - new summary
   - difference summary
   - optional full old/new/diff matrices

Recommended small debug run
---------------------------
PowerShell:

python test_compare_orica_versions_verbose_full.py `
  --old "D:\\work\\Python_Project\\ORICA\\code\\ORICA_final_no_print_quick30.py" `
  --new "D:\\work\\Python_Project\\pyorica\\pyorica\\orica\\core.py" `
  --n-ch 3 `
  --n-pts 64 `
  --block-size-white 8 `
  --block-size-ica 32 `
  --seed 7 `
  --print-input full `
  --print-mode full `
  --stop-on-first-fail

For larger data, use:
  --print-mode summary
  --print-fail-full
  --dump-dir debug_dump
  --log-file compare_log.txt
"""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import inspect
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import scipy.linalg


# ---------------------------------------------------------------------
# Log tee (mirror all print output to a file)
# ---------------------------------------------------------------------

class _TeeStdout:
    """Write to multiple text streams (console + log file)."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty() if self._streams else False


def setup_log_file(log_path: Path) -> None:
    """Mirror sys.stdout to *log_path* for the rest of the process."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    log_f.write(f"# ORICA compare log — {datetime.now().isoformat(timespec='seconds')}\n")
    log_f.write(f"# command: {' '.join(sys.argv)}\n\n")
    log_f.flush()

    orig_stdout = sys.stdout
    sys.stdout = _TeeStdout(orig_stdout, log_f)

    def _restore() -> None:
        sys.stdout = orig_stdout
        log_f.close()
        print(f"Log saved → {log_path.resolve()}", file=orig_stdout)

    atexit.register(_restore)


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

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
        f"Could not find any of {preferred}. "
        f"Classes found: {[c.__name__ for c in classes]}"
    )


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def make_synthetic_eeg(n_ch: int, n_pts: int, seed: int, sfreq: float) -> np.ndarray:
    """Generate deterministic EEG-like data: X = A @ S."""
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


# ---------------------------------------------------------------------
# Formatting / printing helpers
# ---------------------------------------------------------------------

def arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def safe_stats(x: Any) -> str:
    a = arr(x)
    if a.size == 0:
        return f"shape={a.shape}, empty"

    finite = np.isfinite(a)
    if not finite.any():
        return f"shape={a.shape}, all non-finite, fro={np.linalg.norm(a):.6e}"

    return (
        f"shape={a.shape}, "
        f"min={np.nanmin(a):.6e}, max={np.nanmax(a):.6e}, "
        f"mean={np.nanmean(a):.6e}, fro={np.linalg.norm(a):.6e}"
    )


def diff_stats(old: Any, new: Any, atol: float, rtol: float) -> tuple[bool, str]:
    o = arr(old)
    n = arr(new)

    if o.shape != n.shape:
        return False, f"shape mismatch old={o.shape}, new={n.shape}"

    # equal_nan=True so two NaNs do not create a false mismatch for Rn_before.
    ok = bool(np.allclose(o, n, atol=atol, rtol=rtol, equal_nan=True))
    d = np.abs(o - n)

    if d.size == 0:
        return ok, "max_abs=0.000000e+00, mean_abs=0.000000e+00, fro_diff=0.000000e+00"

    finite = np.isfinite(d)
    if not finite.any():
        return ok, "diff all non-finite"

    return ok, (
        f"max_abs={np.nanmax(d):.6e}, "
        f"mean_abs={np.nanmean(d):.6e}, "
        f"fro_diff={np.linalg.norm(np.nan_to_num(o - n)):.6e}"
    )


def matrix_to_string(x: Any, precision: int) -> str:
    return np.array2string(
        arr(x),
        precision=precision,
        suppress_small=False,
        threshold=sys.maxsize,
        max_line_width=240,
    )


def head_to_string(x: Any, n: int, precision: int) -> str:
    a = arr(x).ravel()
    m = min(n, a.size)
    return np.array2string(
        a[:m],
        precision=precision,
        suppress_small=False,
        max_line_width=240,
    )


class Printer:
    def __init__(self, args):
        self.args = args
        self.first_fail_seen = False

    def print_array(self, title: str, x: Any):
        print(title)
        print(f"  {safe_stats(x)}")
        if self.args.print_input == "head":
            print(head_to_string(x, self.args.head_n, self.args.precision))
        elif self.args.print_input == "full":
            print(matrix_to_string(x, self.args.precision))
        print()

    def print_pair(self, label: str, old: Any, new: Any):
        ok, ds = diff_stats(old, new, self.args.atol, self.args.rtol)
        status = "PASS" if ok else "FAIL"

        print(f"[{status}] {label}")
        print(f"  diff: {ds}")
        print(f"  old : {safe_stats(old)}")
        print(f"  new : {safe_stats(new)}")

        should_print_head = self.args.print_mode == "head"
        should_print_full = (
            self.args.print_mode == "full"
            or (self.args.print_fail_full and not ok)
        )

        if should_print_head and not should_print_full:
            print(f"  old head: {head_to_string(old, self.args.head_n, self.args.precision)}")
            print(f"  new head: {head_to_string(new, self.args.head_n, self.args.precision)}")
            if arr(old).shape == arr(new).shape:
                print(f"  diff head: {head_to_string(arr(old) - arr(new), self.args.head_n, self.args.precision)}")

        if should_print_full:
            print("  old full:")
            print(matrix_to_string(old, self.args.precision))
            print("  new full:")
            print(matrix_to_string(new, self.args.precision))
            if arr(old).shape == arr(new).shape:
                print("  diff full: old - new")
                print(matrix_to_string(arr(old) - arr(new), self.args.precision))

        print()

        if not ok and self.args.stop_on_first_fail:
            print(f"STOP: first mismatch found at {label}")
            sys.exit(1)


def maybe_dump(dump_dir: Path | None, name: str, value: Any):
    if dump_dir is None:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("[", "_").replace("]", "").replace(".", "_").replace("/", "_")
    np.save(dump_dir / f"{safe_name}.npy", arr(value))


# ---------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Trace OLD implementation by monkey-patching actual methods
# ---------------------------------------------------------------------

def old_lambda_const(old, args) -> float:
    return float(getattr(old, "lambda_const", args.lambda_const))


def old_gen_cooling(old, t, gamma, lambda_0):
    if hasattr(old, "gen_cooling_ff"):
        return arr(old.gen_cooling_ff(t, gamma, lambda_0))
    t_safe = np.maximum(t, 1e-10)
    return lambda_0 / np.power(t_safe, gamma)


def run_old_with_trace(old, X: np.ndarray, args):
    trace: dict[str, Any] = {
        "white": [],
        "ica": [],
        "initial_captured": False,
    }

    original_white = old.dynamic_whitening
    original_ica = old.dynamic_orica_cooling

    white_count = {"i": 0}
    ica_count = {"i": 0}

    def white_wrapper(blockdata, data_range, state, lambda_const, gamma, lambda_0):
        bi = white_count["i"]
        white_count["i"] += 1

        if not trace["initial_captured"]:
            trace["initial_sphere"] = arr(state["icasphere"]).copy()
            trace["initial_W"] = arr(state["icaweights"]).copy()
            trace["initial_counter"] = np.array([state.get("counter", np.nan)], dtype=np.float64)
            trace["initial_captured"] = True

        sphere_before = arr(state["icasphere"]).copy()
        counter_before = state.get("counter", np.nan)
        lambda_raw = old_gen_cooling(old, counter_before + data_range, gamma, lambda_0)

        # Quick30 old code forces lambda_const by `if True`.
        lambda_used_est = np.full(len(data_range), old_lambda_const(old, args), dtype=np.float64)
        lambda_avg_est = 1.0 - lambda_used_est[int(np.ceil(len(lambda_used_est) / 2)) - 1]

        v_est = sphere_before @ blockdata
        if hasattr(old, "snap_to_kbits"):
            v_est = arr(old.snap_to_kbits(v_est, k=38))

        QWhite_est = lambda_avg_est / (1.0 - lambda_avg_est) + (
            np.linalg.norm(v_est, "fro") ** 2 / len(data_range)
        )
        if hasattr(old, "snap_to_kbits"):
            QWhite_est = float(np.asarray(old.snap_to_kbits(QWhite_est, k=38)))

        update_term_est = (v_est @ v_est.T) / blockdata.shape[1] / QWhite_est @ sphere_before

        out = original_white(blockdata, data_range, state, lambda_const, gamma, lambda_0)
        sphere_after = arr(out["icasphere"]).copy()

        if args.max_white_blocks < 0 or bi < args.max_white_blocks:
            trace["white"].append({
                "block": bi,
                "range_1idx": arr(data_range).copy(),
                "blockdata": arr(blockdata).copy(),
                "sphere_before": sphere_before,
                "counter_before": np.array([counter_before], dtype=np.float64),
                "lambda_raw": lambda_raw.copy(),
                "lambda_used_est": lambda_used_est.copy(),
                "lambda_avg_est": np.array([lambda_avg_est], dtype=np.float64),
                "v_est": v_est.copy(),
                "QWhite_est": np.array([QWhite_est], dtype=np.float64),
                "update_term_est": update_term_est.copy(),
                "sphere_after": sphere_after,
            })

        return out

    def ica_wrapper(blockdata, data_range, state=None, gamma=0.5, lambda_0=1.0):
        bi = ica_count["i"]
        ica_count["i"] += 1

        if state is None:
            state = {}

        W_before = arr(state.get("icaweights", np.eye(blockdata.shape[0]))).copy()
        counter_before = state.get("counter", np.nan)
        Rn_before = None if getattr(old, "Rn", None) is None else arr(old.Rn).copy()

        Y = W_before @ blockdata

        kurtsign = state.get(
            "kurtsign",
            getattr(old, "kurtosis_sign", np.ones(blockdata.shape[0], dtype=bool)),
        )
        F = np.empty_like(Y)
        F[kurtsign, :] = -2.0 * np.tanh(Y[kurtsign, :])
        F[~kurtsign, :] = np.tanh(Y[~kurtsign, :]) - Y[~kurtsign, :]

        model_fitness = np.eye(blockdata.shape[0]) + (Y @ F.T) / blockdata.shape[1]
        lambda_raw = old_gen_cooling(old, counter_before + data_range, gamma, lambda_0)

        out = original_ica(blockdata, data_range, state, gamma, lambda_0)

        W_after = arr(out["icaweights"]).copy()
        lambda_used = arr(out.get("lambda_k", np.full(len(data_range), np.nan)))
        counter_after = np.array([out.get("counter", np.nan)], dtype=np.float64)
        Rn_after = np.array([np.nan]) if getattr(old, "Rn", None) is None else arr(old.Rn).copy()

        # Reconstruct some internal values for printing. These are estimates based on formula.
        if lambda_used.shape == (len(data_range),):
            lambda_prod = np.array([np.prod(1.0 / (1.0 - lambda_used))], dtype=np.float64)
            Q = 1.0 + lambda_used * (np.sum(F * Y, axis=0) - 1.0)
            F_for_update = arr(old.snap_to_kbits(F, k=44)) if hasattr(old, "snap_to_kbits") else F
            W_pre_ortho = lambda_prod[0] * (
                W_before - Y @ np.diag(lambda_used / Q) @ F_for_update.T @ W_before
            )
        else:
            lambda_prod = np.array([np.nan], dtype=np.float64)
            Q = np.full(blockdata.shape[1], np.nan)
            F_for_update = F
            W_pre_ortho = np.full_like(W_before, np.nan)

        if args.max_ica_blocks < 0 or bi < args.max_ica_blocks:
            trace["ica"].append({
                "block": bi,
                "range_1idx": arr(data_range).copy(),
                "blockdata": arr(blockdata).copy(),
                "W_before": W_before,
                "counter_before": np.array([counter_before], dtype=np.float64),
                "Y": Y.copy(),
                "F": F.copy(),
                "model_fitness": model_fitness.copy(),
                "Rn_before": np.array([np.nan]) if Rn_before is None else Rn_before,
                "lambda_raw": lambda_raw.copy(),
                "lambda_used": lambda_used.copy(),
                "lambda_prod_est": lambda_prod,
                "Q_est": Q.copy(),
                "F_for_update_est": F_for_update.copy(),
                "W_pre_ortho_est": W_pre_ortho.copy(),
                "W_after": W_after,
                "counter_after": counter_after,
                "Rn_after": Rn_after,
            })

        return out

    old.dynamic_whitening = white_wrapper
    old.dynamic_orica_cooling = ica_wrapper

    data_center = X - X.mean(axis=1, keepdims=True)
    trace["data_center"] = data_center.copy()

    fit_kwargs = dict(
        block_size_white=args.block_size_white,
        num_pass=args.num_pass,
        lambda_0=args.lambda_0,
        gamma=args.gamma,
        verbose=False,
    )
    if "lambda_const" in inspect.signature(old.fit).parameters:
        fit_kwargs["lambda_const"] = old_lambda_const(old, args)

    sources, W, sphere = old.fit(X, **fit_kwargs)

    # If there were zero whitening blocks, initial state was never captured.
    if not trace["initial_captured"]:
        trace["initial_sphere"] = np.full((args.n_ch, args.n_ch), np.nan)
        trace["initial_W"] = np.full((args.n_ch, args.n_ch), np.nan)
        trace["initial_counter"] = np.array([np.nan], dtype=np.float64)

    trace["mixtures"] = arr(sphere) @ X
    trace["final_sphere"] = arr(sphere).copy()
    trace["final_W"] = arr(W).copy()
    trace["final_sources"] = arr(sources).copy()
    trace["transform_sources"] = arr(old.transform(X)).copy()
    trace["final_counter"] = np.array([getattr(old, "counter", np.nan)], dtype=np.float64)
    return trace


# ---------------------------------------------------------------------
# Trace NEW implementation by monkey-patching actual methods
# ---------------------------------------------------------------------

def new_gen_cooling(new, t):
    if hasattr(new, "_gen_cooling_ff"):
        return arr(new._gen_cooling_ff(t))
    t_safe = np.maximum(t, 1e-10)
    return new.lambda_0 / np.power(t_safe, new.gamma)


def run_new_with_trace(new, X: np.ndarray, args):
    trace: dict[str, Any] = {
        "white": [],
        "ica": [],
        "initial_sphere": arr(new.sphere_).copy(),
        "initial_W": arr(new.weights_).copy(),
        "initial_counter": np.array([getattr(new, "_counter", np.nan)], dtype=np.float64),
    }

    original_white = new._dynamic_whitening
    original_ica = new._dynamic_orica

    white_count = {"i": 0}
    ica_count = {"i": 0}

    def white_wrapper(blockdata, data_range):
        bi = white_count["i"]
        white_count["i"] += 1

        sphere_before = arr(new.sphere_).copy()
        counter_before = getattr(new, "_counter", np.nan)
        lambda_raw = new_gen_cooling(new, counter_before + data_range)
        lambda_used_est = arr(new._forgetting_factor(data_range))
        lambda_avg_est = 1.0 - lambda_used_est[int(np.ceil(len(lambda_used_est) / 2)) - 1]

        v_est = sphere_before @ blockdata
        QWhite_est = lambda_avg_est / (1.0 - lambda_avg_est) + (
            np.linalg.norm(v_est, "fro") ** 2 / len(data_range)
        )
        update_term_est = (v_est @ v_est.T) / blockdata.shape[1] / QWhite_est @ sphere_before

        out = original_white(blockdata, data_range)
        sphere_after = arr(new.sphere_).copy()

        if args.max_white_blocks < 0 or bi < args.max_white_blocks:
            trace["white"].append({
                "block": bi,
                "range_1idx": arr(data_range).copy(),
                "blockdata": arr(blockdata).copy(),
                "sphere_before": sphere_before,
                "counter_before": np.array([counter_before], dtype=np.float64),
                "lambda_raw": lambda_raw.copy(),
                "lambda_used_est": lambda_used_est.copy(),
                "lambda_avg_est": np.array([lambda_avg_est], dtype=np.float64),
                "v_est": v_est.copy(),
                "QWhite_est": np.array([QWhite_est], dtype=np.float64),
                "update_term_est": update_term_est.copy(),
                "sphere_after": sphere_after,
            })

        return out

    def ica_wrapper(blockdata, data_range):
        bi = ica_count["i"]
        ica_count["i"] += 1

        W_before = arr(new.weights_).copy()
        counter_before = getattr(new, "_counter", np.nan)
        Rn_before = None if new._Rn is None else arr(new._Rn).copy()

        Y = W_before @ blockdata
        kurtsign = new._kurtosis_sign
        F = np.empty_like(Y)
        F[kurtsign, :] = -2.0 * np.tanh(Y[kurtsign, :])
        F[~kurtsign, :] = np.tanh(Y[~kurtsign, :]) - Y[~kurtsign, :]

        model_fitness = np.eye(blockdata.shape[0]) + (Y @ F.T) / blockdata.shape[1]
        lambda_raw = new_gen_cooling(new, counter_before + data_range)
        lambda_used_est = arr(new._forgetting_factor(data_range))
        lambda_prod_est = np.array([np.prod(1.0 / (1.0 - lambda_used_est))], dtype=np.float64)
        Q_est = 1.0 + lambda_used_est * (np.sum(F * Y, axis=0) - 1.0)
        W_pre_ortho_est = lambda_prod_est[0] * (
            W_before - Y @ np.diag(lambda_used_est / Q_est) @ F.T @ W_before
        )

        out = original_ica(blockdata, data_range)

        W_after = arr(new.weights_).copy()
        counter_after = np.array([getattr(new, "_counter", np.nan)], dtype=np.float64)
        Rn_after = np.array([np.nan]) if new._Rn is None else arr(new._Rn).copy()

        if args.max_ica_blocks < 0 or bi < args.max_ica_blocks:
            trace["ica"].append({
                "block": bi,
                "range_1idx": arr(data_range).copy(),
                "blockdata": arr(blockdata).copy(),
                "W_before": W_before,
                "counter_before": np.array([counter_before], dtype=np.float64),
                "Y": Y.copy(),
                "F": F.copy(),
                "model_fitness": model_fitness.copy(),
                "Rn_before": np.array([np.nan]) if Rn_before is None else Rn_before,
                "lambda_raw": lambda_raw.copy(),
                "lambda_used": lambda_used_est.copy(),
                "lambda_prod_est": lambda_prod_est,
                "Q_est": Q_est.copy(),
                "F_for_update_est": F.copy(),
                "W_pre_ortho_est": W_pre_ortho_est.copy(),
                "W_after": W_after,
                "counter_after": counter_after,
                "Rn_after": Rn_after,
            })

        return out

    new._dynamic_whitening = white_wrapper
    new._dynamic_orica = ica_wrapper

    data_center = X - X.mean(axis=1, keepdims=True)
    trace["data_center"] = data_center.copy()

    new.fit(X)
    sources = new.transform(X)

    trace["mixtures"] = arr(new.sphere_) @ X
    trace["final_sphere"] = arr(new.sphere_).copy()
    trace["final_W"] = arr(new.weights_).copy()
    trace["final_sources"] = arr(sources).copy()
    trace["transform_sources"] = arr(new.transform(X)).copy()
    trace["final_counter"] = np.array([getattr(new, "_counter", np.nan)], dtype=np.float64)
    return trace


# ---------------------------------------------------------------------
# Print trace comparison
# ---------------------------------------------------------------------

def compare_and_print_traces(old_trace, new_trace, X, args):
    printer = Printer(args)
    dump_dir = Path(args.dump_dir) if args.dump_dir else None

    if args.print_input != "none":
        printer.print_array("=== Original synthetic input X ===", X)
    maybe_dump(dump_dir, "X_input", X)

    print("=== Global / initialization ===")
    for key in ["data_center", "initial_sphere", "initial_W", "initial_counter"]:
        printer.print_pair(key, old_trace[key], new_trace[key])
        maybe_dump(dump_dir, f"old_{key}", old_trace[key])
        maybe_dump(dump_dir, f"new_{key}", new_trace[key])

    print("=== Whitening trace ===")
    printer.print_pair(
        "number of recorded whitening blocks",
        np.array([len(old_trace["white"])]),
        np.array([len(new_trace["white"])]),
    )

    for old_ev, new_ev in zip(old_trace["white"], new_trace["white"]):
        bi = int(old_ev["block"])
        print(f"--- Whitening block {bi} ---")
        for key in [
            "range_1idx",
            "blockdata",
            "sphere_before",
            "counter_before",
            "lambda_raw",
            "lambda_used_est",
            "lambda_avg_est",
            "v_est",
            "QWhite_est",
            "update_term_est",
            "sphere_after",
        ]:
            label = f"white[{bi}].{key}"
            printer.print_pair(label, old_ev[key], new_ev[key])
            maybe_dump(dump_dir, f"old_{label}", old_ev[key])
            maybe_dump(dump_dir, f"new_{label}", new_ev[key])

    print("=== Mixtures ===")
    printer.print_pair("mixtures", old_trace["mixtures"], new_trace["mixtures"])
    maybe_dump(dump_dir, "old_mixtures", old_trace["mixtures"])
    maybe_dump(dump_dir, "new_mixtures", new_trace["mixtures"])

    print("=== ICA trace ===")
    printer.print_pair(
        "number of recorded ICA blocks",
        np.array([len(old_trace["ica"])]),
        np.array([len(new_trace["ica"])]),
    )

    for old_ev, new_ev in zip(old_trace["ica"], new_trace["ica"]):
        bi = int(old_ev["block"])
        print(f"--- ICA block {bi} ---")
        for key in [
            "range_1idx",
            "blockdata",
            "W_before",
            "counter_before",
            "Y",
            "F",
            "model_fitness",
            "Rn_before",
            "lambda_raw",
            "lambda_used",
            "lambda_prod_est",
            "Q_est",
            "F_for_update_est",
            "W_pre_ortho_est",
            "W_after",
            "counter_after",
            "Rn_after",
        ]:
            label = f"ica[{bi}].{key}"
            printer.print_pair(label, old_ev[key], new_ev[key])
            maybe_dump(dump_dir, f"old_{label}", old_ev[key])
            maybe_dump(dump_dir, f"new_{label}", new_ev[key])

    print("=== Final outputs ===")
    for key in [
        "final_sphere",
        "final_W",
        "final_sources",
        "transform_sources",
        "final_counter",
    ]:
        printer.print_pair(key, old_trace[key], new_trace[key])
        maybe_dump(dump_dir, f"old_{key}", old_trace[key])
        maybe_dump(dump_dir, f"new_{key}", new_trace[key])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Print full old/new/diff matrices for ORICA comparison."
    )
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

    p.add_argument(
        "--print-mode",
        choices=["summary", "head", "full"],
        default="summary",
        help="How much to print for each old/new variable.",
    )
    p.add_argument(
        "--print-input",
        choices=["none", "head", "full"],
        default="head",
        help="How much of original synthetic X to print.",
    )
    p.add_argument("--print-fail-full", action="store_true", help="Print full matrix whenever a check FAILs")
    p.add_argument("--stop-on-first-fail", action="store_true", help="Exit immediately after first mismatch")
    p.add_argument("--head-n", type=int, default=12)
    p.add_argument("--precision", type=int, default=6)

    p.add_argument(
        "--dump-dir",
        default="",
        help="Optional directory to save all printed arrays as .npy files.",
    )
    p.add_argument(
        "--log-file",
        default="compare_log.txt",
        metavar="PATH",
        help="Mirror all printed output to this file (relative to this script's "
             "directory). Use empty string to disable logging.",
    )

    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = script_dir / log_path
        setup_log_file(log_path)
        print(f"Logging to {log_path.resolve()}\n")

    old_mod = load_module(args.old, "orica_old_module")
    new_mod = load_module(args.new, "orica_new_module")
    old_cls = find_class(old_mod, ["ORICA_final_new", "ORICA_final", "ORICA"])
    new_cls = find_class(new_mod, ["ORICAFilter"])

    # Synthetic X is generated with its own default_rng seed.
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

    # This controls old random Q if old code still uses np.random.randn.
    np.random.seed(args.seed)
    old_trace = run_old_with_trace(old, X, args)

    # This controls new random permutation if time_perm=True in the future.
    np.random.seed(args.seed)
    new_trace = run_new_with_trace(new, X, args)

    compare_and_print_traces(old_trace, new_trace, X, args)


if __name__ == "__main__":
    main()

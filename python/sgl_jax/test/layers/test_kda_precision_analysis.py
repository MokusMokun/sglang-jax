"""KDA per-stage precision analysis using production pipeline with intermediates capture.

Three analysis modes:

  matmul-only   Single Q-projection matmul at three TPU precision levels
                (DEFAULT / HIGH / HIGHEST). Isolates the dominant error source.

  pipeline      Run the production KimiDeltaAttention forward pass with
                capture_intermediates, compare each stage against GPU dump.
                Shows cumulative error propagation through the real pipeline.

  pipeline      With --precision high/highest, uses jax.default_matmul_precision
  (high prec)   context manager to override all matmul precision. Comparing
                DEFAULT vs HIGH shows which stages are matmul-precision dominated.

Usage:
    python -m sgl_jax.test.layers.test_kda_precision_analysis --mode matmul-only
    python -m sgl_jax.test.layers.test_kda_precision_analysis --mode pipeline --layer L22 --dtype fp32
    python -m sgl_jax.test.layers.test_kda_precision_analysis --mode pipeline --precision high
    python -m sgl_jax.test.layers.test_kda_precision_analysis --mode pipeline --all-cases
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np

from sgl_jax.test.layers.test_kda_module import (
    _build_extend_env,
    _build_module,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DUMP_BASE = os.environ.get(
    "KDA_DUMP_DIR", "/models/yuhao/kimi-linear/kda_module"
)

ALL_CASES = [
    "single_T1", "single_T8", "single_T64", "single_T65",
    "single_T128", "single_T256", "single_T1024",
    "varlen_balanced_4x32", "varlen_unbalanced", "varlen_single_T128",
    "single_T128_initstate", "varlen_initstate",
]

# Captured key -> GPU dump key(s)
STAGE_MAP = [
    ("Q projection",         "q_proj",        "intermediates__q_proj"),
    ("K projection",         "k_proj",        "intermediates__k_proj"),
    ("V projection",         "v_proj",        "intermediates__v_proj"),
    ("Q conv+SiLU",          "q_after_conv",  "intermediates__q_after_conv"),
    ("K conv+SiLU",          "k_after_conv",  "intermediates__k_after_conv"),
    ("V conv+SiLU",          "v_after_conv",  "intermediates__v_after_conv"),
    ("Gate (fused_kda_gate)", "g",            "intermediates__g"),
    ("Beta (sigmoid)",       "beta",          "intermediates__beta"),
    ("KDA output",           "o_kda",         "intermediates__o_kda_chunk"),
    ("Recurrent state",      "recurrent_state", "intermediates__recurrent_state_chunk"),
    ("Output gate (g_out)",  "output_gate",   "intermediates__g_out"),
    ("Output norm",          "o_norm",        "intermediates__o_norm"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_case(layer_dir: str, case_name: str) -> dict:
    path = os.path.join(layer_dir, f"case_{case_name}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Case not found: {path}")
    return dict(np.load(path, allow_pickle=True))


# ---------------------------------------------------------------------------
# Metrics / display
# ---------------------------------------------------------------------------

def _dist(arr: np.ndarray) -> dict:
    """Compute distribution stats for an array."""
    a = arr.astype(np.float32).ravel()
    return {
        "mean": float(np.mean(a)),
        "var": float(np.var(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    a = np.asarray(actual, dtype=np.float32).ravel()
    e = np.asarray(expected, dtype=np.float32).ravel()
    if a.size != e.size:
        raise ValueError(f"Size mismatch: actual {a.size} vs expected {e.size}")
    diff = np.abs(a - e)
    nz = np.abs(e) > 1e-8
    return {
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "max_rel": float(np.max(diff[nz] / np.abs(e[nz]))) if np.any(nz) else float("nan"),
        "mean_rel": float(np.mean(diff[nz] / np.abs(e[nz]))) if np.any(nz) else float("nan"),
    }


def compare_stage(label: str, actual, case: dict, dump_key: str) -> dict:
    if dump_key not in case:
        return {"label": label, "skip": True}
    m = metrics(np.asarray(actual), case[dump_key])
    m["label"] = label
    m["tpu_dist"] = _dist(np.asarray(actual))
    m["gpu_dist"] = _dist(case[dump_key])
    return m


def print_error_table(rows: list[dict], title: str):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"  {'Stage':<28} {'max_abs':>10} {'mean_abs':>10} {'max_rel':>10} {'mean_rel':>10}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in rows:
        if r.get("skip"):
            print(f"  {r['label']:<28} {'(no dump key)':>42}")
        else:
            print(
                f"  {r['label']:<28} "
                f"{r['max_abs']:>10.2e} {r['mean_abs']:>10.2e} "
                f"{r['max_rel']:>10.2e} {r['mean_rel']:>10.2e}"
            )
    print()


def print_distribution_table(rows: list[dict], title: str):
    print(f"\n{'=' * 100}")
    print(f"  Distributions: {title}")
    print(f"{'=' * 100}")
    print(f"  {'Stage':<28} {'Src':>4} {'mean':>11} {'var':>11} {'min':>11} {'max':>11}")
    print(f"  {'-' * 28} {'-' * 4} {'-' * 11} {'-' * 11} {'-' * 11} {'-' * 11}")
    for r in rows:
        if r.get("skip"):
            continue
        for src, key in [("TPU", "tpu_dist"), ("GPU", "gpu_dist")]:
            d = r[key]
            prefix = r["label"] if src == "TPU" else ""
            print(
                f"  {prefix:<28} {src:>4} "
                f"{d['mean']:>11.4e} {d['var']:>11.4e} "
                f"{d['min']:>11.4e} {d['max']:>11.4e}"
            )
    print()


# ---------------------------------------------------------------------------
# Mode: matmul-only
# ---------------------------------------------------------------------------

def run_matmul_only(layer_dir: str, dtype: jnp.dtype):
    """Single Q-projection matmul at three precision levels."""
    weights = dict(np.load(os.path.join(layer_dir, "weights.npz"), allow_pickle=True))
    case = load_case(layer_dir, "single_T128")

    W = jnp.asarray(weights["weights__q_proj.weight"].T, dtype=dtype)
    x = jnp.asarray(case["hidden_states"], dtype=dtype)
    if x.ndim == 3:
        x = x[0]
    ref = case["intermediates__q_proj"]
    if ref.ndim == 3:
        ref = ref[0]

    dtype_label = "fp32" if dtype == jnp.float32 else "bf16"

    # Data distributions
    print(f"\n{'=' * 78}")
    print(f"  Data distributions ({dtype_label})")
    print(f"{'=' * 78}")
    for name, arr in [("x (input)", np.asarray(x)), ("W (weight)", np.asarray(W)), ("ref (GPU)", ref)]:
        a = arr.astype(np.float32)
        pcts = np.percentile(np.abs(a), [50, 90, 99, 100])
        print(f"  {name:>12s}  shape={str(arr.shape):<16s}  "
              f"mean={a.mean():+.4e}  std={a.std():.4e}  "
              f"|abs| p50={pcts[0]:.4e} p90={pcts[1]:.4e} p99={pcts[2]:.4e} max={pcts[3]:.4e}")

    # Precision comparison
    layer_name = os.path.basename(layer_dir)
    print(f"\n{'=' * 78}")
    print(f"  Precision comparison ({dtype_label}, {layer_name}, single_T128, Q projection)")
    print(f"{'=' * 78}")
    print(f"  {'Precision':<10s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'p50':>10s}  {'p99':>10s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for name, prec in [("DEFAULT", jax.lax.Precision.DEFAULT),
                        ("HIGH", jax.lax.Precision.HIGH),
                        ("HIGHEST", jax.lax.Precision.HIGHEST)]:
        out = jax.lax.dot(x, W, precision=prec)
        out_f32 = np.asarray(out, dtype=np.float32)
        ref_f32 = ref.astype(np.float32) if ref.ndim == 2 else ref[0].astype(np.float32)
        diff = np.abs(out_f32 - ref_f32)
        s = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "p50": float(np.percentile(diff, 50)),
            "p99": float(np.percentile(diff, 99)),
        }
        print(f"  {name:<10s}  {s['max_abs']:>10.4e}  {s['mean_abs']:>10.4e}  "
              f"{s['p50']:>10.4e}  {s['p99']:>10.4e}")
    print()


# ---------------------------------------------------------------------------
# Mode: pipeline (production forward with intermediates capture)
# ---------------------------------------------------------------------------

def run_pipeline(layer_dir: str, case_name: str, dtype: jnp.dtype, precision: str):
    """Run production KimiDeltaAttention.__call__ with intermediates capture."""
    case = load_case(layer_dir, case_name)
    T = int(case["T"])

    module = _build_module(os.path.join(layer_dir, "weights.npz"), dtype)

    hidden = jnp.asarray(case["hidden_states"], dtype=dtype)
    if hidden.ndim == 3:
        hidden = hidden[0]

    has_init = bool(case.get("has_initial_state", False))
    init_state = (
        jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
        if has_init else None
    )
    has_cu = bool(case.get("has_cu_seqlens", False))
    cu_seqlens = (
        jnp.asarray(case["cu_seqlens"], dtype=jnp.int32)
        if has_cu else None
    )

    fb, pool = _build_extend_env(module, T, init_state, cu_seqlens)

    intermediates = {}
    ctx = (
        jax.default_matmul_precision(precision)
        if precision != "default"
        else contextlib.nullcontext()
    )
    with ctx:
        output, _ = module(None, hidden, fb, pool, intermediates=intermediates)

    # Build comparison rows
    rows = []
    for label, cap_key, dump_key in STAGE_MAP:
        if cap_key in intermediates:
            rows.append(compare_stage(label, intermediates[cap_key], case, dump_key))
        else:
            rows.append({"label": label, "skip": True})

    # Final output
    out_key = "out_fp32" if dtype == jnp.float32 else "out_bf16"
    rows.append(compare_stage("Final output (o_proj)", output, case, out_key))

    dtype_label = "fp32" if dtype == jnp.float32 else "bf16"
    N = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else 1
    title = f"{case_name} ({dtype_label}, T={T}, N={N}, precision={precision.upper()})"
    print_error_table(rows, title)
    print_distribution_table(rows, title)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KDA per-stage precision analysis (matmul-only / pipeline)"
    )
    parser.add_argument(
        "--mode", required=True, choices=["matmul-only", "pipeline"],
        help="Analysis mode",
    )
    parser.add_argument("--layer", default="L22", help="Layer dir (default: L22)")
    parser.add_argument("--case", default="single_T128", help="Case name")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument(
        "--precision", default="default",
        choices=["default", "high", "highest"],
        help="Matmul precision override (pipeline mode only)",
    )
    parser.add_argument("--all-cases", action="store_true", help="Sweep all 12 cases")
    args = parser.parse_args()

    layer_dir = os.path.join(DUMP_BASE, args.layer)
    if not os.path.isdir(layer_dir):
        print(f"ERROR: dump dir not found: {layer_dir}")
        sys.exit(1)

    dtype = jnp.float32 if args.dtype == "fp32" else jnp.bfloat16

    if args.mode == "matmul-only":
        run_matmul_only(layer_dir, dtype)
        return

    cases = ALL_CASES if args.all_cases else [args.case]
    for cn in cases:
        try:
            run_pipeline(layer_dir, cn, dtype, args.precision)
        except FileNotFoundError as e:
            print(f"\n  {cn}: {e} -- skipped")


if __name__ == "__main__":
    main()

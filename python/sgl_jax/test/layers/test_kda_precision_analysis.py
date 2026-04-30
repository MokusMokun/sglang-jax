"""KDA per-stage precision analysis using production pipeline with intermediates capture.

Four analysis modes:

  matmul-only    Single Q-projection matmul at three TPU precision levels
                 (DEFAULT / HIGH / HIGHEST). Isolates the dominant error source.

  accumulated    Run the production KimiDeltaAttention forward pass with
                 capture_intermediates, compare each stage against GPU dump.
                 Shows accumulated error propagation through the real pipeline.
                 With --precision high/highest, uses jax.default_matmul_precision
                 context manager to override all matmul precision.

  isolated       Each stage independently receives the GPU dump intermediate as
                 input and calls the production function. Measures per-stage error
                 without upstream contamination. Single-sequence cases only.

Usage:
    python -m sgl_jax.test.layers.test_kda_precision_analysis --mode matmul-only
    python -m sgl_jax.test.layers.test_kda_precision_analysis --mode accumulated --layer L22 --dtype fp32
    python -m sgl_jax.test.layers.test_kda_precision_analysis --mode accumulated --precision high
    python -m sgl_jax.test.layers.test_kda_precision_analysis --mode accumulated --kernel pallas
    python -m sgl_jax.test.layers.test_kda_precision_analysis --mode isolated --layer L22 --dtype fp32
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

from sgl_jax.srt.kernels.kda import chunk_kda, fused_recurrent_kda
from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sgl_jax.srt.layers.attention.linear.short_convolution import (
    l2_normalize,
    short_convolution,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.test.layers.test_kda_module import (
    _TEST_MESH,
    _build_extend_env,
    _build_module,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DUMP_BASE = os.environ.get(
    "KDA_DUMP_DIR", "/models/yuhao/kimi-linear/kda_module"
)
FULL_MODEL_DUMP_DIR = os.environ.get(
    "KDA_FULL_MODEL_DUMP_DIR", "/models/yuhao/kimi-linear/kda_full_model_dump"
)

ALL_CASES = [
    "single_T1", "single_T8", "single_T64", "single_T65",
    "single_T128", "single_T256", "single_T1024",
    "varlen_balanced_4x32", "varlen_unbalanced", "varlen_single_T128",
    "single_T128_initstate", "varlen_initstate",
]

# Captured key -> GPU dump key (isolated dumps)
STAGE_MAP_ISOLATED = [
    ("Q projection",         "q_proj",        "intermediates__q_proj"),
    ("K projection",         "k_proj",        "intermediates__k_proj"),
    ("V projection",         "v_proj",        "intermediates__v_proj"),
    ("Q conv+SiLU",          "q_after_conv",  "intermediates__q_after_conv"),
    ("K conv+SiLU",          "k_after_conv",  "intermediates__k_after_conv"),
    ("V conv+SiLU",          "v_after_conv",  "intermediates__v_after_conv"),
    ("Gate (fused_kda_gate)", "g",            "intermediates__g"),
    ("Beta (sigmoid)",       "beta",          "intermediates__beta"),
    ("KDA output (fused_rec)", "o_kda",       "intermediates__o_kda_fused_recurrent"),
    ("Recurrent state (fused)", "recurrent_state", "intermediates__recurrent_state_fused_recurrent"),
    ("Output gate (g_out)",  "output_gate",   "intermediates__g_out"),
    ("Output norm",          "o_norm",        "intermediates__o_norm"),
]

# Captured key -> GPU dump key (full-model dumps)
STAGE_MAP_FULL_MODEL = [
    ("Q projection",         "q_proj",        "intermediates__q_proj"),
    ("K projection",         "k_proj",        "intermediates__k_proj"),
    ("V projection",         "v_proj",        "intermediates__v_proj"),
    ("Q conv+SiLU",          "q_after_conv",  "intermediates__q_after_conv"),
    ("K conv+SiLU",          "k_after_conv",  "intermediates__k_after_conv"),
    ("V conv+SiLU",          "v_after_conv",  "intermediates__v_after_conv"),
    ("Gate (fused_kda_gate)", "g",            "intermediates__g"),
    ("Beta (sigmoid)",       "beta",          "intermediates__beta"),
    ("KDA output (fused_rec)", "o_kda",       "intermediates__o_kda"),
    ("Recurrent state (fused)", "recurrent_state", "intermediates__recurrent_state"),
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


def load_full_model_layer(layer_idx: int) -> dict:
    """Load full-model dump and normalize keys to match isolated dump format."""
    path = os.path.join(FULL_MODEL_DUMP_DIR, f"layer_{layer_idx:02d}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Full-model dump not found: {path}")
    raw = dict(np.load(path, allow_pickle=True))
    # Normalize keys: input_hidden_states -> hidden_states, output -> out_fp32
    case = {}
    for k, v in raw.items():
        case[k] = v
    case["hidden_states"] = raw["input_hidden_states"]
    case["out_fp32"] = raw["output"]
    case["T"] = raw["input_hidden_states"].shape[1]
    case["has_initial_state"] = False
    case["has_cu_seqlens"] = False
    return case


# Layer name -> 0-based layer index for full-model dumps
_LAYER_NAME_TO_IDX = {
    "L0": 0, "L1": 1, "L2": 2, "L4": 4, "L5": 5, "L6": 6,
    "L8": 8, "L9": 9, "L10": 10, "L12": 12, "L13": 13, "L14": 14,
    "L16": 16, "L17": 17, "L18": 18, "L20": 20, "L21": 21, "L22": 22,
    "L24": 24, "L25": 25,
}


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

def run_pipeline(layer_dir: str, case_name: str, dtype: jnp.dtype, precision: str,
                 source: str = "isolated", kernel: str = "naive"):
    """Run production KimiDeltaAttention.__call__ with intermediates capture."""
    layer_name = os.path.basename(layer_dir)
    if source == "full-model":
        layer_idx = _LAYER_NAME_TO_IDX[layer_name]
        case = load_full_model_layer(layer_idx)
        stage_map = STAGE_MAP_FULL_MODEL
    else:
        case = load_case(layer_dir, case_name)
        stage_map = STAGE_MAP_ISOLATED
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
    if kernel == "pallas":
        fb.attn_backend.use_pallas_prefill = True

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
    for label, cap_key, dump_key in stage_map:
        if cap_key in intermediates:
            rows.append(compare_stage(label, intermediates[cap_key], case, dump_key))
        else:
            rows.append({"label": label, "skip": True})

    # Final output
    out_key = "out_fp32" if dtype == jnp.float32 else "out_bf16"
    rows.append(compare_stage("Final output (o_proj)", output, case, out_key))

    dtype_label = "fp32" if dtype == jnp.float32 else "bf16"
    N = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else 1
    kernel_label = "pallas" if kernel == "pallas" else "naive"
    title = f"{case_name} ({dtype_label}, T={T}, N={N}, precision={precision.upper()}, kernel={kernel_label})"
    print_error_table(rows, title)
    print_distribution_table(rows, title)
    return rows


# ---------------------------------------------------------------------------
# Mode: isolated (per-stage with GPU dump inputs, single-sequence only)
# ---------------------------------------------------------------------------

def run_isolated(layer_dir: str, case_name: str, dtype: jnp.dtype, precision: str,
                 source: str = "isolated"):
    """Each stage independently receives GPU dump as input. No upstream error."""
    layer_name = os.path.basename(layer_dir)
    if source == "full-model":
        layer_idx = _LAYER_NAME_TO_IDX[layer_name]
        case = load_full_model_layer(layer_idx)
        o_kda_key = "intermediates__o_kda"
        recurrent_key = "intermediates__recurrent_state"
        out_key = "out_fp32"
    else:
        case = load_case(layer_dir, case_name)
        o_kda_key = "intermediates__o_kda_fused_recurrent"
        recurrent_key = "intermediates__recurrent_state_fused_recurrent"
        out_key = "out_fp32" if dtype == jnp.float32 else "out_bf16"
    T = int(case["T"])

    if bool(case.get("has_cu_seqlens", False)):
        print(f"  {case_name}: isolated mode only supports single-sequence cases -- skipped")
        return []

    module = _build_module(os.path.join(layer_dir, "weights.npz"), dtype)
    H, D = module.num_heads, module.head_dim
    proj = H * D

    hidden = jnp.asarray(case["hidden_states"], dtype=dtype)
    if hidden.ndim == 3:
        hidden = hidden[0]

    def gpu(key):
        """Load GPU dump intermediate, squeeze batch dim, cast to dtype."""
        v = jnp.asarray(case[key], dtype=dtype)
        return v[0] if v.ndim > hidden.ndim and v.shape[0] == 1 else v

    def gpu_f32(key):
        """Load GPU dump in float32 (for stages that cast internally)."""
        v = jnp.asarray(case[key], dtype=jnp.float32)
        return v[0] if v.ndim > hidden.ndim and v.shape[0] == 1 else v

    ctx = (
        jax.default_matmul_precision(precision)
        if precision != "default"
        else contextlib.nullcontext()
    )

    rows = []
    with ctx:
        # 1. Projections: hidden -> q/k/v
        for label, proj_fn, key in [
            ("Q projection", module.q_proj, "intermediates__q_proj"),
            ("K projection", module.k_proj, "intermediates__k_proj"),
            ("V projection", module.v_proj, "intermediates__v_proj"),
        ]:
            rows.append(compare_stage(label, proj_fn(hidden)[0], case, key))

        # 2. Conv+SiLU: GPU proj -> conv (uses production short_convolution)
        cache = jnp.zeros((1, proj, module.conv_size), dtype=dtype)
        cu = jnp.array([0, T], dtype=jnp.int32)
        for label, conv_w, proj_key, conv_key in [
            ("Q conv+SiLU", module.q_conv1d.weight.value, "intermediates__q_proj", "intermediates__q_after_conv"),
            ("K conv+SiLU", module.k_conv1d.weight.value, "intermediates__k_proj", "intermediates__k_after_conv"),
            ("V conv+SiLU", module.v_conv1d.weight.value, "intermediates__v_proj", "intermediates__v_after_conv"),
        ]:
            inp = gpu(proj_key).reshape(T, -1)
            out, _ = short_convolution(inp, conv_w, cache, cu, ForwardMode.EXTEND, activation=jax.nn.silu)
            rows.append(compare_stage(label, out.reshape(T, H, D), case, conv_key))

        # 3. Gate: hidden -> f_a_proj -> f_b_proj -> fused_kda_gate
        raw_gate, _ = module.f_b_proj(module.f_a_proj(hidden)[0])
        raw_gate = raw_gate.reshape(T, H, D)
        backend = KDAAttnBackend(mesh=_TEST_MESH)
        g = backend._fused_kda_gate(module.attn, raw_gate)
        rows.append(compare_stage("Gate (fused_kda_gate)", g, case, "intermediates__g"))

        # 4. Beta: hidden -> b_proj -> sigmoid
        beta = jax.nn.sigmoid(module.b_proj(hidden)[0].astype(jnp.float32))
        rows.append(compare_stage("Beta (sigmoid)", beta, case, "intermediates__beta"))

        # 5. KDA kernel: GPU post-conv + gate + beta -> fused_recurrent_kda
        kern_q = l2_normalize(gpu("intermediates__q_after_conv").reshape(T, H, D))
        kern_k = l2_normalize(gpu("intermediates__k_after_conv").reshape(T, H, D))
        kern_v = gpu("intermediates__v_after_conv").reshape(T, H, D)
        kern_g = gpu_f32("intermediates__g").reshape(T, H, D)
        kern_beta = gpu_f32("intermediates__beta")
        if kern_beta.ndim == 3:
            kern_beta = kern_beta.reshape(T, H)

        has_init = bool(case.get("has_initial_state", False))
        init_state = (
            jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
            if has_init else jnp.zeros((1, H, D, D), dtype=jnp.float32)
        )
        o_rec, s_rec = fused_recurrent_kda(
            kern_q[None], kern_k[None], kern_v[None],
            kern_g[None], kern_beta[None],
            scale=D**-0.5, initial_state=init_state, output_final_state=True,
        )
        rows.append(compare_stage("KDA output (fused_rec)", o_rec[0], case, o_kda_key))
        rows.append(compare_stage("Recurrent state (fused)", s_rec, case, recurrent_key))

        # 5b. Pallas chunk_kda: same GPU post-conv inputs -> chunk_kda
        # Pallas kernels don't support Precision.HIGH/HIGHEST, so always run at DEFAULT.
        chunk_o_key = "intermediates__o_kda" if source == "full-model" else "intermediates__o_kda_chunk"
        chunk_s_key = "intermediates__recurrent_state" if source == "full-model" else "intermediates__recurrent_state_chunk"
        chunk_cu = jnp.array([0, T], dtype=jnp.int32)
        with jax.default_matmul_precision("default"):
            o_chunk, s_chunk, *_ = chunk_kda(
                kern_q[None], kern_k[None], kern_v[None],
                kern_g[None],
                kern_beta[None],
                scale=D**-0.5,
                initial_state=init_state,
                output_final_state=True,
                cu_seqlens=chunk_cu,
                use_gate_in_kernel=False,
            )
        rows.append(compare_stage("KDA output (chunk)", o_chunk[0], case, chunk_o_key))
        rows.append(compare_stage("Recurrent state (chunk)", s_chunk, case, chunk_s_key))

        # 6. Output gate: hidden -> g_a_proj -> g_b_proj
        g_out, _ = module.g_b_proj(module.g_a_proj(hidden)[0])
        rows.append(compare_stage("Output gate (g_out)", g_out.reshape(T, H, D), case, "intermediates__g_out"))

        # 7. Output norm: GPU o_kda + GPU g_out -> GatedRMSNorm
        o_kda_dump_key = "intermediates__o_kda" if source == "full-model" else "intermediates__o_kda_chunk"
        o_kda_gpu = gpu(o_kda_dump_key).reshape(T, H, D)
        g_out_gpu = gpu("intermediates__g_out").reshape(T, H, D)
        o_norm = module.o_norm(o_kda_gpu, g_out_gpu)
        rows.append(compare_stage("Output norm", o_norm, case, "intermediates__o_norm"))

        # 8. Final output: GPU o_norm -> o_proj
        o_norm_gpu = gpu("intermediates__o_norm").reshape(T, proj)
        final, _ = module.o_proj(o_norm_gpu)
        rows.append(compare_stage("Final output (o_proj)", final, case, out_key))

    dtype_label = "fp32" if dtype == jnp.float32 else "bf16"
    title = f"{case_name} ({dtype_label}, T={T}, precision={precision.upper()}) [ISOLATED]"
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
        "--mode", required=True, choices=["matmul-only", "accumulated", "isolated"],
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
    parser.add_argument(
        "--source", default="isolated", choices=["isolated", "full-model"],
        help="Dump source: isolated layer dumps or full-model dumps",
    )
    parser.add_argument(
        "--kernel", default="naive", choices=["naive", "pallas"],
        help="Kernel for accumulated mode: naive (fused_recurrent) or pallas (chunk_kda)",
    )
    args = parser.parse_args()

    layer_dir = os.path.join(DUMP_BASE, args.layer)
    if not os.path.isdir(layer_dir):
        print(f"ERROR: dump dir not found: {layer_dir}")
        sys.exit(1)

    dtype = jnp.float32 if args.dtype == "fp32" else jnp.bfloat16

    if args.mode == "matmul-only":
        run_matmul_only(layer_dir, dtype)
        return

    if args.source == "full-model":
        if args.layer not in _LAYER_NAME_TO_IDX:
            print(f"ERROR: {args.layer} not a valid KDA layer for full-model dumps")
            sys.exit(1)
        # Full-model has a single case per layer, ignore --case and --all-cases
        if args.mode == "isolated":
            run_isolated(layer_dir, "full_model", dtype, args.precision, source="full-model")
        else:
            run_pipeline(layer_dir, "full_model", dtype, args.precision,
                         source="full-model", kernel=args.kernel)
    else:
        cases = ALL_CASES if args.all_cases else [args.case]
        if args.mode == "isolated":
            for cn in cases:
                try:
                    run_isolated(layer_dir, cn, dtype, args.precision, source="isolated")
                except FileNotFoundError as e:
                    print(f"\n  {cn}: {e} -- skipped")
        else:
            for cn in cases:
                try:
                    run_pipeline(layer_dir, cn, dtype, args.precision,
                                 source="isolated", kernel=args.kernel)
                except FileNotFoundError as e:
                    print(f"\n  {cn}: {e} -- skipped")


if __name__ == "__main__":
    main()

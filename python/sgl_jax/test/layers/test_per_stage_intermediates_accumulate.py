"""Per-stage intermediate precision comparison: GPU dump vs JAX KDA pipeline.

Runs the full KimiDeltaAttention pipeline step-by-step and compares each
intermediate against the GPU dump reference.  Stages:

  1. Q/K/V projection         (intermediates__q_proj, k_proj, v_proj)
  2. Q/K/V conv + SiLU         (intermediates__q_after_conv, k_after_conv, v_after_conv)
  3. Q/K L2 norm               (no dump — compare against conv output dampening)
  4. Gate (fused_kda_gate)      (intermediates__g)
  5. Beta (sigmoid)             (intermediates__beta)
  6. KDA attention output       (intermediates__o_kda_chunk / o_kda_fused_recurrent)
  7. Recurrent state            (intermediates__recurrent_state_chunk / _fused_recurrent)
  8. Output gate                (intermediates__g_out)
  9. Output norm                (intermediates__o_norm)
 10. Final output               (out_fp32 / out_bf16)

Usage (on TPU with dump access):
    python test_per_stage_intermediates.py                           # L0, single_T128, fp32
    python test_per_stage_intermediates.py --layer L22               # worst-case layer
    python test_per_stage_intermediates.py --dtype bf16              # bf16
    python test_per_stage_intermediates.py --case single_T1024       # long sequence
    python test_per_stage_intermediates.py --all-cases               # sweep all 12 cases
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.linear.short_convolution import (
    l2_normalize,
    short_convolution,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

# ---------------------------------------------------------------------------
# Dump paths & case list
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


# ---------------------------------------------------------------------------
# Weight / case loading
# ---------------------------------------------------------------------------

def load_weights(layer_dir: str) -> dict:
    raw = dict(np.load(os.path.join(layer_dir, "weights.npz"), allow_pickle=True))

    num_heads = int(raw["config__num_heads"])
    head_dim = int(raw["config__head_dim"])
    conv_size = int(raw["config__conv_size"])
    hidden_size = int(raw["config__hidden_size"])
    rms_norm_eps = float(raw.get("config__rms_norm_eps", 1e-6))

    def squeeze_conv(w):
        if w.ndim == 3 and w.shape[1] == 1:
            return w[:, 0, :]
        return w

    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "conv_size": conv_size,
        "hidden_size": hidden_size,
        "rms_norm_eps": rms_norm_eps,
        "proj_size": num_heads * head_dim,
        # Linear projections: GPU [out, in] → JAX [in, out]
        "q_proj_w": raw["weights__q_proj.weight"].T,
        "k_proj_w": raw["weights__k_proj.weight"].T,
        "v_proj_w": raw["weights__v_proj.weight"].T,
        "f_a_proj_w": raw["weights__f_a_proj.weight"].T,
        "f_b_proj_w": raw["weights__f_b_proj.weight"].T,
        "b_proj_w": raw["weights__b_proj.weight"].T,
        "g_a_proj_w": raw["weights__g_a_proj.weight"].T,
        "g_b_proj_w": raw["weights__g_b_proj.weight"].T,
        "o_proj_w": raw["weights__o_proj.weight"].T,
        "o_norm_w": raw["weights__o_norm.weight"],
        # Conv: GPU [D, 1, K] → [D, K]
        "q_conv_w": squeeze_conv(raw["weights__q_conv1d.weight"]),
        "k_conv_w": squeeze_conv(raw["weights__k_conv1d.weight"]),
        "v_conv_w": squeeze_conv(raw["weights__v_conv1d.weight"]),
        # Gate params
        "A_log": raw["weights__A_log"],
        "dt_bias": raw["weights__dt_bias"],
    }


def load_case(layer_dir: str, case_name: str) -> dict:
    path = os.path.join(layer_dir, f"case_{case_name}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Case not found: {path}")
    return dict(np.load(path, allow_pickle=True))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if expected.ndim == actual.ndim + 1 and expected.shape[0] == 1:
        expected = expected[0]

    diff = np.abs(actual - expected)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))

    nonzero = np.abs(expected) > 1e-8
    if np.any(nonzero):
        rel = diff[nonzero] / np.abs(expected[nonzero])
        max_rel = float(np.max(rel))
        mean_rel = float(np.mean(rel))
    else:
        max_rel = mean_rel = float("nan")

    return {"max_abs": max_abs, "mean_abs": mean_abs, "max_rel": max_rel, "mean_rel": mean_rel}


def print_table(rows: list[dict], title: str):
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    print(f"  {'Stage':<35} {'max_abs':>10} {'mean_abs':>10} {'max_rel':>10} {'mean_rel':>10}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in rows:
        if r.get("skip"):
            print(f"  {r['label']:<35} {'(no dump key)':>42}")
            continue
        print(
            f"  {r['label']:<35} "
            f"{r['max_abs']:>10.2e} {r['mean_abs']:>10.2e} "
            f"{r['max_rel']:>10.2e} {r['mean_rel']:>10.2e}"
        )
    print()


# ---------------------------------------------------------------------------
# Per-stage pipeline
# ---------------------------------------------------------------------------

def compare_stage(label: str, actual: jax.Array, case: dict, dump_key: str) -> dict:
    """Compare JAX result against dump intermediate; skip if key absent."""
    if dump_key not in case:
        return {"label": label, "skip": True}
    expected = case[dump_key]
    m = compute_metrics(np.asarray(actual), expected)
    return {"label": label, **m}


def run_one_case(ws: dict, case: dict, case_name: str, dtype: jnp.dtype, precision):
    T = int(case["T"])
    H = ws["num_heads"]
    D = ws["head_dim"]
    K = ws["conv_size"]
    proj = ws["proj_size"]
    eps = ws["rms_norm_eps"]

    hidden = jnp.asarray(case["hidden_states"], dtype=dtype)
    if hidden.ndim == 3:
        hidden = hidden[0]  # [T, hidden_size]

    if bool(case.get("has_cu_seqlens", False)):
        cu_seqlens = jnp.asarray(case["cu_seqlens"], dtype=jnp.int32)
    else:
        cu_seqlens = jnp.array([0, T], dtype=jnp.int32)
    N = cu_seqlens.shape[0] - 1

    def w(name):
        return jnp.asarray(ws[name], dtype=dtype)

    def matmul(a, b):
        return jax.lax.dot(a, b, precision=precision)

    rows = []

    # ------------------------------------------------------------------
    # Stage 1: Linear projections
    # ------------------------------------------------------------------
    q_proj = matmul(hidden, w("q_proj_w"))
    k_proj = matmul(hidden, w("k_proj_w"))
    v_proj = matmul(hidden, w("v_proj_w"))
    rows.append(compare_stage("1. Q projection", q_proj, case, "intermediates__q_proj"))
    rows.append(compare_stage("1. K projection", k_proj, case, "intermediates__k_proj"))
    rows.append(compare_stage("1. V projection", v_proj, case, "intermediates__v_proj"))

    # ------------------------------------------------------------------
    # Stage 2: Conv + SiLU
    # ------------------------------------------------------------------
    cache = jnp.zeros((N, proj, K), dtype=dtype)

    q_conv, _ = short_convolution(q_proj, w("q_conv_w"), cache, cu_seqlens, ForwardMode.EXTEND, activation=jax.nn.silu)
    k_conv, _ = short_convolution(k_proj, w("k_conv_w"), cache, cu_seqlens, ForwardMode.EXTEND, activation=jax.nn.silu)
    v_conv, _ = short_convolution(v_proj, w("v_conv_w"), cache, cu_seqlens, ForwardMode.EXTEND, activation=jax.nn.silu)

    q_heads = q_conv.reshape(T, H, D)
    k_heads = k_conv.reshape(T, H, D)
    v_heads = v_conv.reshape(T, H, D)

    rows.append(compare_stage("2. Q conv+SiLU", q_heads, case, "intermediates__q_after_conv"))
    rows.append(compare_stage("2. K conv+SiLU", k_heads, case, "intermediates__k_after_conv"))
    rows.append(compare_stage("2. V conv+SiLU", v_heads, case, "intermediates__v_after_conv"))

    # ------------------------------------------------------------------
    # Stage 3: L2 normalize (no dump — report value range dampening)
    # ------------------------------------------------------------------
    q_normed = l2_normalize(q_heads)
    k_normed = l2_normalize(k_heads)

    # ------------------------------------------------------------------
    # Stage 4: Gate (fused_kda_gate)
    # ------------------------------------------------------------------
    raw_gate = matmul(matmul(hidden, w("f_a_proj_w")), w("f_b_proj_w"))
    raw_gate = raw_gate.reshape(T, H, D)

    A_log = jnp.asarray(ws["A_log"], dtype=jnp.float32).reshape(H)
    dt_bias = jnp.asarray(ws["dt_bias"], dtype=jnp.float32).reshape(H, D)

    g = -jnp.exp(A_log.reshape(H, 1)) * jax.nn.softplus(
        raw_gate.astype(jnp.float32) + dt_bias
    )
    rows.append(compare_stage("4. Gate (fused_kda_gate)", g, case, "intermediates__g"))

    # ------------------------------------------------------------------
    # Stage 5: Beta (sigmoid)
    # ------------------------------------------------------------------
    beta = jax.nn.sigmoid(matmul(hidden, w("b_proj_w")).astype(jnp.float32))
    rows.append(compare_stage("5. Beta (sigmoid)", beta, case, "intermediates__beta"))

    # ------------------------------------------------------------------
    # Stage 6 & 7: KDA attention + recurrent state
    # ------------------------------------------------------------------
    from sgl_jax.srt.kernels.kda import chunk_kda, fused_recurrent_kda

    scale = D ** -0.5
    has_init = bool(case.get("has_initial_state", False))
    if has_init:
        init_state = jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
    else:
        init_state = jnp.zeros((N, H, D, D), dtype=jnp.float32)

    # Try chunk_kda (Pallas) — may not work on CPU
    try:
        o_chunk, state_chunk, *_ = chunk_kda(
            q_normed[None, ...],
            k_normed[None, ...],
            v_heads[None, ...],
            raw_gate[None, ...],
            beta[None, ...] if beta.ndim == 2 else beta.reshape(T, H)[None, ...],
            scale=scale,
            initial_state=init_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        rows.append(compare_stage("6. KDA output (chunk)", o_chunk[0], case, "intermediates__o_kda_chunk"))
        rows.append(compare_stage("7. Recurrent state (chunk)", state_chunk, case, "intermediates__recurrent_state_chunk"))
    except Exception as e:
        rows.append({"label": f"6. KDA chunk: {type(e).__name__}", "skip": True})

    # Fused recurrent (naive JAX)
    try:
        if cu_seqlens.shape[0] == 2:
            o_fused, state_fused = fused_recurrent_kda(
                q_normed[None, ...],
                k_normed[None, ...],
                v_heads[None, ...],
                g[None, ...],
                beta[None, ...] if beta.ndim == 2 else beta.reshape(T, H)[None, ...],
                scale=scale,
                initial_state=init_state,
                output_final_state=True,
            )
        else:
            # varlen: pad to batch
            from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
            q_b, k_b, v_b, g_b, beta_b = KDAAttnBackend._unpack_varlen(
                q_normed, k_normed, v_heads, g,
                beta if beta.ndim == 2 else beta.reshape(T, H),
                cu_seqlens,
            )
            o_fused, state_fused = fused_recurrent_kda(
                q_b, k_b, v_b, g_b, beta_b,
                scale=scale,
                initial_state=init_state,
                output_final_state=True,
            )
            o_fused_packed = KDAAttnBackend._repack_varlen(o_fused, cu_seqlens, T)
            o_fused = o_fused_packed[None, ...]  # add batch dim for comparison

        rows.append(compare_stage("6. KDA output (fused_rec)", o_fused[0] if o_fused.ndim == 4 else o_fused, case, "intermediates__o_kda_fused_recurrent"))
        rows.append(compare_stage("7. Recurrent state (fused_rec)", state_fused, case, "intermediates__recurrent_state_fused_recurrent"))
    except Exception as e:
        rows.append({"label": f"6. KDA fused_rec: {type(e).__name__}: {e}", "skip": True})

    # ------------------------------------------------------------------
    # Stage 8: Output gate
    # ------------------------------------------------------------------
    g_out = matmul(matmul(hidden, w("g_a_proj_w")), w("g_b_proj_w"))
    g_out = g_out.reshape(T, H, D)
    rows.append(compare_stage("8. Output gate (g_out)", g_out, case, "intermediates__g_out"))

    # ------------------------------------------------------------------
    # Stage 9: Output norm — use JAX's own KDA output (cumulative)
    # ------------------------------------------------------------------
    # Pick whichever JAX kernel output we computed
    o_kda = None
    if 'o_chunk' in dir() and o_chunk is not None:
        o_kda = o_chunk[0]
    elif 'o_fused' in dir() and o_fused is not None:
        o_kda = o_fused[0] if o_fused.ndim == 4 else o_fused

    if o_kda is not None:
        o_norm_w = jnp.asarray(ws["o_norm_w"], dtype=jnp.float32)
        x_f32 = o_kda.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_norm = x_f32 * jax.lax.rsqrt(variance + eps)
        x_norm = x_norm * o_norm_w
        o_normed = (x_norm * jax.nn.sigmoid(g_out.astype(jnp.float32))).astype(dtype)
        rows.append(compare_stage("9. Output norm (o_norm)", o_normed, case, "intermediates__o_norm"))
    else:
        rows.append({"label": "9. Output norm", "skip": True})

    # ------------------------------------------------------------------
    # Stage 10: Final output — use JAX's own o_norm (cumulative)
    # ------------------------------------------------------------------
    dtype_label = "fp32" if dtype == jnp.float32 else "bf16"
    out_key = "out_fp32" if dtype == jnp.float32 else "out_bf16"
    if o_kda is not None:
        final = matmul(o_normed.reshape(T, proj), w("o_proj_w"))
        rows.append(compare_stage("10. Final output (o_proj)", final, case, out_key))
    else:
        rows.append({"label": "10. Final output", "skip": True})

    prec_label = str(precision).split(".")[-1] if precision else "DEFAULT"
    print_table(rows, f"{case_name} ({dtype_label}, T={T}, N={N}, precision={prec_label})")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-stage KDA intermediate comparison")
    parser.add_argument("--layer", default="L0", help="Layer dir (default: L0)")
    parser.add_argument("--case", default="single_T128", help="Case name")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--precision", default="default", choices=["default", "high", "highest"])
    parser.add_argument("--all-cases", action="store_true", help="Sweep all 12 cases")
    args = parser.parse_args()

    layer_dir = os.path.join(DUMP_BASE, args.layer)
    if not os.path.isdir(layer_dir):
        print(f"ERROR: dump dir not found: {layer_dir}")
        sys.exit(1)

    dtype = jnp.float32 if args.dtype == "fp32" else jnp.bfloat16
    precision_map = {
        "default": jax.lax.Precision.DEFAULT,
        "high": jax.lax.Precision.HIGH,
        "highest": jax.lax.Precision.HIGHEST,
    }
    precision = precision_map[args.precision]

    print(f"Loading weights from {layer_dir}/weights.npz ...")
    ws = load_weights(layer_dir)
    print(f"  H={ws['num_heads']}, D={ws['head_dim']}, K={ws['conv_size']}, "
          f"hidden={ws['hidden_size']}, proj={ws['proj_size']}")

    cases = ALL_CASES if args.all_cases else [args.case]
    for cn in cases:
        try:
            case = load_case(layer_dir, cn)
        except FileNotFoundError as e:
            print(f"\n  {cn}: {e} — skipped")
            continue
        run_one_case(ws, case, cn, dtype, precision)


if __name__ == "__main__":
    main()

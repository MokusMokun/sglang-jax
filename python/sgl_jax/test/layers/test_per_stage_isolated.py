"""Isolated per-stage precision comparison: GPU dump vs JAX KDA pipeline.

Unlike the cumulative test, each stage receives the **GPU dump intermediate**
as input, so the measured error is purely from that stage's JAX implementation,
not accumulated from prior stages.

Stages:

  1. Q/K/V projection         hidden_states → JAX proj → vs intermediates__q/k/v_proj
  2. Q/K/V conv + SiLU        GPU q/k/v_proj → JAX conv → vs intermediates__q/k/v_after_conv
  3. Q/K L2 norm               GPU q/k_after_conv → JAX L2norm → (no dump, report range)
  4. Gate (fused_kda_gate)     hidden_states → JAX gate → vs intermediates__g
  5. Beta (sigmoid)            hidden_states → JAX beta → vs intermediates__beta
  6. KDA attention             GPU post-conv (L2-normed) + g + beta → JAX kernel → vs intermediates__o_kda_*
  7. Recurrent state           (same kernel call as 6) → vs intermediates__recurrent_state_*
  8. Output gate               hidden_states → JAX g_out → vs intermediates__g_out
  9. Output norm               GPU o_kda + GPU g_out → JAX norm → vs intermediates__o_norm
 10. Final output              GPU o_norm → JAX o_proj → vs out_fp32

Usage:
    python test_per_stage_isolated.py                                      # L0, single_T128, fp32, default
    python test_per_stage_isolated.py --layer L22 --dtype bf16             # worst-case layer, bf16
    python test_per_stage_isolated.py --layer L22 --precision high         # HIGH precision matmul
    python test_per_stage_isolated.py --all-cases                          # sweep all cases
"""

from __future__ import annotations

import argparse
import os
import sys

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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_weights(layer_dir: str) -> dict:
    raw = dict(np.load(os.path.join(layer_dir, "weights.npz"), allow_pickle=True))

    num_heads = int(raw["config__num_heads"])
    head_dim = int(raw["config__head_dim"])
    conv_size = int(raw["config__conv_size"])
    hidden_size = int(raw["config__hidden_size"])
    rms_norm_eps = float(raw.get("config__rms_norm_eps", 1e-6))

    def squeeze_conv(w):
        return w[:, 0, :] if w.ndim == 3 and w.shape[1] == 1 else w

    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "conv_size": conv_size,
        "hidden_size": hidden_size,
        "rms_norm_eps": rms_norm_eps,
        "proj_size": num_heads * head_dim,
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
        "q_conv_w": squeeze_conv(raw["weights__q_conv1d.weight"]),
        "k_conv_w": squeeze_conv(raw["weights__k_conv1d.weight"]),
        "v_conv_w": squeeze_conv(raw["weights__v_conv1d.weight"]),
        "A_log": raw["weights__A_log"],
        "dt_bias": raw["weights__dt_bias"],
    }


def load_case(layer_dir: str, case_name: str) -> dict:
    path = os.path.join(layer_dir, f"case_{case_name}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


# ---------------------------------------------------------------------------
# Metrics / display
# ---------------------------------------------------------------------------

def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    a = np.asarray(actual, dtype=np.float32)
    e = np.asarray(expected, dtype=np.float32)
    if e.ndim == a.ndim + 1 and e.shape[0] == 1:
        e = e[0]
    diff = np.abs(a - e)
    nz = np.abs(e) > 1e-8
    return {
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "max_rel": float(np.max(diff[nz] / np.abs(e[nz]))) if np.any(nz) else float("nan"),
        "mean_rel": float(np.mean(diff[nz] / np.abs(e[nz]))) if np.any(nz) else float("nan"),
    }


def print_table(rows: list[dict], title: str):
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    print(f"  {'Stage':<35} {'max_abs':>10} {'mean_abs':>10} {'max_rel':>10} {'mean_rel':>10}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in rows:
        if r.get("skip"):
            print(f"  {r['label']:<35} {'(no dump key)':>42}")
        else:
            print(
                f"  {r['label']:<35} "
                f"{r['max_abs']:>10.2e} {r['mean_abs']:>10.2e} "
                f"{r['max_rel']:>10.2e} {r['mean_rel']:>10.2e}"
            )
    print()


def stage(label, actual, case, key):
    if key not in case:
        return {"label": label, "skip": True}
    return {"label": label, **metrics(np.asarray(actual), case[key])}


# ---------------------------------------------------------------------------
# Isolated pipeline
# ---------------------------------------------------------------------------

def run_one_case(ws: dict, case: dict, case_name: str, dtype: jnp.dtype, precision):
    T = int(case["T"])
    H, D, K = ws["num_heads"], ws["head_dim"], ws["conv_size"]
    proj = ws["proj_size"]
    eps = ws["rms_norm_eps"]

    hidden = jnp.asarray(case["hidden_states"], dtype=dtype)
    if hidden.ndim == 3:
        hidden = hidden[0]

    if bool(case.get("has_cu_seqlens", False)):
        cu = jnp.asarray(case["cu_seqlens"], dtype=jnp.int32)
    else:
        cu = jnp.array([0, T], dtype=jnp.int32)
    N = cu.shape[0] - 1

    def w(name):
        return jnp.asarray(ws[name], dtype=dtype)

    def matmul(a, b):
        return jax.lax.dot(a, b, precision=precision)

    rows = []

    # === Stage 1: Projections (input: hidden_states — same for both) ===
    rows.append(stage("1. Q projection", matmul(hidden, w("q_proj_w")), case, "intermediates__q_proj"))
    rows.append(stage("1. K projection", matmul(hidden, w("k_proj_w")), case, "intermediates__k_proj"))
    rows.append(stage("1. V projection", matmul(hidden, w("v_proj_w")), case, "intermediates__v_proj"))

    # === Stage 2: Conv+SiLU (input: GPU dump projection output) ===
    cache = jnp.zeros((N, proj, K), dtype=dtype)
    for stream, conv_name, proj_key, conv_key in [
        ("Q", "q_conv_w", "intermediates__q_proj", "intermediates__q_after_conv"),
        ("K", "k_conv_w", "intermediates__k_proj", "intermediates__k_after_conv"),
        ("V", "v_conv_w", "intermediates__v_proj", "intermediates__v_after_conv"),
    ]:
        if proj_key not in case:
            # Fallback: use JAX projection output if no dump proj available
            if stream == "Q":
                proj_in = hidden @ w("q_proj_w")
            elif stream == "K":
                proj_in = hidden @ w("k_proj_w")
            else:
                proj_in = hidden @ w("v_proj_w")
            label_suffix = " (cumul.)"
        else:
            proj_in = jnp.asarray(case[proj_key], dtype=dtype)
            if proj_in.ndim == 3:
                proj_in = proj_in[0]
            label_suffix = ""

        conv_out, _ = short_convolution(
            proj_in, w(conv_name), cache, cu, ForwardMode.EXTEND, activation=jax.nn.silu,
        )
        heads = conv_out.reshape(T, H, D)
        rows.append(stage(f"2. {stream} conv+SiLU{label_suffix}", heads, case, conv_key))

    # === Stage 3: L2 norm (no dump for normed; show input→output range compression) ===
    for stream, conv_key in [("Q", "intermediates__q_after_conv"), ("K", "intermediates__k_after_conv")]:
        if conv_key in case:
            conv_heads = jnp.asarray(case[conv_key], dtype=dtype)
            if conv_heads.ndim == 4:
                conv_heads = conv_heads[0]
            normed = l2_normalize(conv_heads)
            in_range = float(np.max(np.abs(np.asarray(conv_heads))))
            out_range = float(np.max(np.abs(np.asarray(normed))))
            rows.append({
                "label": f"3. {stream} L2norm range",
                "max_abs": in_range,
                "mean_abs": out_range,
                "max_rel": out_range / max(in_range, 1e-8),
                "mean_rel": float("nan"),
            })

    # === Stage 4: Gate (input: hidden_states — same for both) ===
    raw_gate = matmul(matmul(hidden, w("f_a_proj_w")), w("f_b_proj_w"))
    raw_gate = raw_gate.reshape(T, H, D)

    A_log = jnp.asarray(ws["A_log"], dtype=jnp.float32).reshape(H)
    dt_bias = jnp.asarray(ws["dt_bias"], dtype=jnp.float32).reshape(H, D)
    g = -jnp.exp(A_log.reshape(H, 1)) * jax.nn.softplus(
        raw_gate.astype(jnp.float32) + dt_bias
    )
    rows.append(stage("4. Gate (fused_kda_gate)", g, case, "intermediates__g"))

    # === Stage 5: Beta (input: hidden_states — same for both) ===
    beta = jax.nn.sigmoid((hidden @ w("b_proj_w")).astype(jnp.float32))
    rows.append(stage("5. Beta (sigmoid)", beta, case, "intermediates__beta"))

    # === Stage 6 & 7: KDA kernel (input: GPU dump post-conv + gate + beta) ===
    from sgl_jax.srt.kernels.kda import chunk_kda, fused_recurrent_kda

    scale = D ** -0.5
    has_init = bool(case.get("has_initial_state", False))
    init_state = (
        jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
        if has_init else jnp.zeros((N, H, D, D), dtype=jnp.float32)
    )

    # Get GPU dump inputs for kernel
    gpu_q = gpu_k = gpu_v = gpu_g = gpu_beta = None
    if "intermediates__q_after_conv" in case and "intermediates__k_after_conv" in case:
        gpu_q = jnp.asarray(case["intermediates__q_after_conv"], dtype=dtype)
        gpu_k = jnp.asarray(case["intermediates__k_after_conv"], dtype=dtype)
        gpu_v = jnp.asarray(case["intermediates__v_after_conv"], dtype=dtype)
        if gpu_q.ndim == 4:
            gpu_q, gpu_k, gpu_v = gpu_q[0], gpu_k[0], gpu_v[0]
        # L2 normalize (GPU does this inside kernel)
        gpu_q_normed = l2_normalize(gpu_q)
        gpu_k_normed = l2_normalize(gpu_k)
    if "intermediates__g" in case:
        gpu_g = jnp.asarray(case["intermediates__g"], dtype=jnp.float32)
        if gpu_g.ndim == 4:
            gpu_g = gpu_g[0]
    if "intermediates__beta" in case:
        gpu_beta = jnp.asarray(case["intermediates__beta"], dtype=jnp.float32)
        if gpu_beta.ndim == 3:
            gpu_beta = gpu_beta[0]

    if gpu_q is not None and gpu_g is not None and gpu_beta is not None:
        # Chunk kernel
        try:
            o_chunk, s_chunk, *_ = chunk_kda(
                gpu_q_normed[None], gpu_k_normed[None], gpu_v[None],
                raw_gate[None],  # raw gate — kernel applies gate internally
                gpu_beta[None],
                scale=scale,
                initial_state=init_state,
                output_final_state=True,
                cu_seqlens=cu,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
            )
            rows.append(stage("6. KDA output (chunk)", o_chunk[0], case, "intermediates__o_kda_chunk"))
            rows.append(stage("7. Recurrent state (chunk)", s_chunk, case, "intermediates__recurrent_state_chunk"))
        except Exception as e:
            rows.append({"label": f"6. KDA chunk: {type(e).__name__}", "skip": True})

        # Fused recurrent
        try:
            if cu.shape[0] == 2:
                o_fr, s_fr = fused_recurrent_kda(
                    gpu_q_normed[None], gpu_k_normed[None], gpu_v[None],
                    gpu_g[None], gpu_beta[None],
                    scale=scale, initial_state=init_state, output_final_state=True,
                )
            else:
                from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
                q_b, k_b, v_b, g_b, beta_b = KDAAttnBackend._unpack_varlen(
                    gpu_q_normed, gpu_k_normed, gpu_v, gpu_g, gpu_beta, cu,
                )
                o_fr, s_fr = fused_recurrent_kda(
                    q_b, k_b, v_b, g_b, beta_b,
                    scale=scale, initial_state=init_state, output_final_state=True,
                )
                o_fr = KDAAttnBackend._repack_varlen(o_fr, cu, T)[None]

            rows.append(stage("6. KDA output (fused_rec)", o_fr[0] if o_fr.ndim == 4 else o_fr, case, "intermediates__o_kda_fused_recurrent"))
            rows.append(stage("7. Recurrent state (fused)", s_fr, case, "intermediates__recurrent_state_fused_recurrent"))
        except Exception as e:
            rows.append({"label": f"6. KDA fused: {type(e).__name__}: {e}", "skip": True})
    else:
        rows.append({"label": "6-7. KDA kernel (missing inputs)", "skip": True})

    # === Stage 8: Output gate (input: hidden_states — same for both) ===
    g_out = matmul(matmul(hidden, w("g_a_proj_w")), w("g_b_proj_w")).reshape(T, H, D)
    rows.append(stage("8. Output gate (g_out)", g_out, case, "intermediates__g_out"))

    # === Stage 9: Output norm (input: GPU o_kda + GPU g_out) ===
    gpu_o_kda = None
    if "intermediates__o_kda_chunk" in case:
        gpu_o_kda = jnp.asarray(case["intermediates__o_kda_chunk"], dtype=dtype)
        if gpu_o_kda.ndim == 4:
            gpu_o_kda = gpu_o_kda[0]
    gpu_g_out = None
    if "intermediates__g_out" in case:
        gpu_g_out = jnp.asarray(case["intermediates__g_out"], dtype=dtype)
        if gpu_g_out.ndim == 4:
            gpu_g_out = gpu_g_out[0]

    if gpu_o_kda is not None and gpu_g_out is not None:
        o_norm_w = jnp.asarray(ws["o_norm_w"], dtype=jnp.float32)
        x_f32 = gpu_o_kda.astype(jnp.float32)
        var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_n = x_f32 * jax.lax.rsqrt(var + eps) * o_norm_w
        o_normed = (x_n * jax.nn.sigmoid(gpu_g_out.astype(jnp.float32))).astype(dtype)
        rows.append(stage("9. Output norm (o_norm)", o_normed, case, "intermediates__o_norm"))
    else:
        rows.append({"label": "9. Output norm", "skip": True})

    # === Stage 10: Final output (input: GPU o_norm) ===
    if "intermediates__o_norm" in case:
        gpu_o_norm = jnp.asarray(case["intermediates__o_norm"], dtype=dtype)
        if gpu_o_norm.ndim == 4:
            gpu_o_norm = gpu_o_norm[0]
        final = matmul(gpu_o_norm.reshape(T, proj), w("o_proj_w"))
        rows.append(stage("10. Final output (o_proj)", final, case, "out_fp32"))
    else:
        rows.append({"label": "10. Final output", "skip": True})

    dtype_label = "fp32" if dtype == jnp.float32 else "bf16"
    prec_label = str(precision).split(".")[-1] if precision else "DEFAULT"
    print_table(rows, f"{case_name} ({dtype_label}, T={T}, N={N}, precision={prec_label}) [ISOLATED]")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Isolated per-stage KDA comparison")
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

"""KDA per-stage precision analysis: GPU dump vs JAX/TPU implementation.

Three analysis modes:

  matmul-only   Single Q-projection matmul at three TPU precision levels
                (DEFAULT / HIGH / HIGHEST). Isolates the dominant error source.

  isolated      Each of 10 pipeline stages receives the GPU dump intermediate
                as input, so measured error is purely from that stage's JAX
                implementation — no contamination from prior stages.

  cumulative    Full JAX pipeline run stage-by-stage, each stage feeding its
                output to the next. Shows how errors accumulate end-to-end.

Stages (isolated & cumulative):

   1. Q/K/V projection         hidden_states → matmul → vs intermediates__q/k/v_proj
   2. Q/K/V conv + SiLU        → short_convolution → vs intermediates__q/k/v_after_conv
   3. Q/K L2 norm              → l2_normalize → (no dump, report range)
   4. Gate (fused_kda_gate)     → -exp(A_log) * softplus(g + dt_bias) → vs intermediates__g
   5. Beta (sigmoid)            → sigmoid(b_proj) → vs intermediates__beta
   6. KDA attention             → chunk_kda / fused_recurrent_kda → vs intermediates__o_kda_*
   7. Recurrent state           (same kernel call) → vs intermediates__recurrent_state_*
   8. Output gate               → g_a_proj → g_b_proj → vs intermediates__g_out
   9. Output norm               → GatedRMSNorm → vs intermediates__o_norm
  10. Final output              → o_proj → vs out_fp32

Usage:
    python test_kda_precision_analysis.py --mode matmul-only
    python test_kda_precision_analysis.py --mode isolated --layer L22 --dtype bf16
    python test_kda_precision_analysis.py --mode cumulative --precision high
    python test_kda_precision_analysis.py --mode isolated --all-cases
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

PRECISIONS = {
    "default": jax.lax.Precision.DEFAULT,
    "high": jax.lax.Precision.HIGH,
    "highest": jax.lax.Precision.HIGHEST,
}


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
        raise FileNotFoundError(f"Case not found: {path}")
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


def compare_stage(label: str, actual, case: dict, key: str) -> dict:
    if key not in case:
        return {"label": label, "skip": True}
    return {"label": label, **metrics(np.asarray(actual), case[key])}


# ---------------------------------------------------------------------------
# Mode: matmul-only
# ---------------------------------------------------------------------------

def run_matmul_only(layer_dir: str, dtype: jnp.dtype):
    """Single Q-projection matmul at three precision levels."""
    ws = load_weights(layer_dir)
    case = load_case(layer_dir, "single_T128")

    W = jnp.asarray(ws["q_proj_w"], dtype=dtype)
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
    print(f"\n{'=' * 78}")
    print(f"  Precision comparison ({dtype_label}, L22, single_T128, Q projection)")
    print(f"{'=' * 78}")
    print(f"  {'Precision':<10s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'p50':>10s}  {'p99':>10s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for name, prec in [("DEFAULT", jax.lax.Precision.DEFAULT),
                        ("HIGH", jax.lax.Precision.HIGH),
                        ("HIGHEST", jax.lax.Precision.HIGHEST)]:
        out = jax.lax.dot(x, W, precision=prec)
        out_f32 = np.asarray(out, dtype=np.float32)
        diff = np.abs(out_f32 - ref)
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
# Mode: isolated / cumulative (unified 10-stage pipeline)
# ---------------------------------------------------------------------------

def run_pipeline(ws: dict, case: dict, case_name: str, dtype: jnp.dtype,
                 precision, *, isolated: bool):
    """Run 10-stage KDA pipeline.

    Args:
        isolated: If True, each stage reads GPU dump as input (isolated error).
                  If False, each stage reads JAX output from previous stage (cumulative).
    """
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

    def gpu_or_jax(dump_key, jax_val):
        """In isolated mode, prefer GPU dump; in cumulative mode, use JAX output."""
        if isolated and dump_key in case:
            v = jnp.asarray(case[dump_key], dtype=dtype)
            return v[0] if v.ndim == jax_val.ndim + 1 and v.shape[0] == 1 else v
        return jax_val

    rows = []

    # === Stage 1: Projections ===
    q_proj = matmul(hidden, w("q_proj_w"))
    k_proj = matmul(hidden, w("k_proj_w"))
    v_proj = matmul(hidden, w("v_proj_w"))
    rows.append(compare_stage("1. Q projection", q_proj, case, "intermediates__q_proj"))
    rows.append(compare_stage("1. K projection", k_proj, case, "intermediates__k_proj"))
    rows.append(compare_stage("1. V projection", v_proj, case, "intermediates__v_proj"))

    # === Stage 2: Conv + SiLU ===
    cache = jnp.zeros((N, proj, K), dtype=dtype)
    conv_inputs = {
        "Q": ("q_conv_w", gpu_or_jax("intermediates__q_proj", q_proj), "intermediates__q_after_conv"),
        "K": ("k_conv_w", gpu_or_jax("intermediates__k_proj", k_proj), "intermediates__k_after_conv"),
        "V": ("v_conv_w", gpu_or_jax("intermediates__v_proj", v_proj), "intermediates__v_after_conv"),
    }
    q_heads = k_heads = v_heads = None
    for stream, (conv_name, proj_in, conv_key) in conv_inputs.items():
        if proj_in.ndim == 3:
            proj_in = proj_in[0]
        conv_out, _ = short_convolution(
            proj_in, w(conv_name), cache, cu, ForwardMode.EXTEND, activation=jax.nn.silu,
        )
        heads = conv_out.reshape(T, H, D)
        rows.append(compare_stage(f"2. {stream} conv+SiLU", heads, case, conv_key))
        if stream == "Q":
            q_heads = heads
        elif stream == "K":
            k_heads = heads
        else:
            v_heads = heads

    # === Stage 3: L2 norm (no dump, report range) ===
    q_normed = l2_normalize(q_heads)
    k_normed = l2_normalize(k_heads)

    for stream, heads in [("Q", q_heads), ("K", k_heads)]:
        normed = l2_normalize(heads)
        in_range = float(np.max(np.abs(np.asarray(heads))))
        out_range = float(np.max(np.abs(np.asarray(normed))))
        rows.append({
            "label": f"3. {stream} L2norm range",
            "max_abs": in_range, "mean_abs": out_range,
            "max_rel": out_range / max(in_range, 1e-8), "mean_rel": float("nan"),
        })

    # === Stage 4: Gate ===
    raw_gate = matmul(matmul(hidden, w("f_a_proj_w")), w("f_b_proj_w"))
    raw_gate = raw_gate.reshape(T, H, D)
    A_log = jnp.asarray(ws["A_log"], dtype=jnp.float32).reshape(H)
    dt_bias = jnp.asarray(ws["dt_bias"], dtype=jnp.float32).reshape(H, D)
    g = -jnp.exp(A_log.reshape(H, 1)) * jax.nn.softplus(
        raw_gate.astype(jnp.float32) + dt_bias
    )
    rows.append(compare_stage("4. Gate (fused_kda_gate)", g, case, "intermediates__g"))

    # === Stage 5: Beta ===
    beta = jax.nn.sigmoid(matmul(hidden, w("b_proj_w")).astype(jnp.float32))
    rows.append(compare_stage("5. Beta (sigmoid)", beta, case, "intermediates__beta"))

    # === Stage 6 & 7: KDA kernel ===
    from sgl_jax.srt.kernels.kda import chunk_kda, fused_recurrent_kda

    scale = D ** -0.5
    has_init = bool(case.get("has_initial_state", False))
    init_state = (
        jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
        if has_init else jnp.zeros((N, H, D, D), dtype=jnp.float32)
    )

    # In isolated mode, feed GPU dump post-conv + GPU gate + beta to kernel
    if isolated:
        kern_q = gpu_or_jax("intermediates__q_after_conv", q_heads)
        kern_k = gpu_or_jax("intermediates__k_after_conv", k_heads)
        kern_v = gpu_or_jax("intermediates__v_after_conv", v_heads)
        kern_q = l2_normalize(kern_q)
        kern_k = l2_normalize(kern_k)
        kern_g = gpu_or_jax("intermediates__g", g)
        kern_beta = gpu_or_jax("intermediates__beta", beta)
    else:
        kern_q, kern_k, kern_v = q_normed, k_normed, v_heads
        kern_g, kern_beta = g, beta

    o_kda = None

    # Chunk kernel
    try:
        o_chunk, s_chunk, *_ = chunk_kda(
            kern_q[None], kern_k[None], kern_v[None],
            raw_gate[None], kern_beta[None],
            scale=scale, initial_state=init_state, output_final_state=True,
            cu_seqlens=cu, use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
        )
        rows.append(compare_stage("6. KDA output (chunk)", o_chunk[0], case, "intermediates__o_kda_chunk"))
        rows.append(compare_stage("7. Recurrent state (chunk)", s_chunk, case, "intermediates__recurrent_state_chunk"))
        o_kda = o_chunk[0]
    except Exception as e:
        rows.append({"label": f"6. KDA chunk: {type(e).__name__}", "skip": True})

    # Fused recurrent
    try:
        if cu.shape[0] == 2:
            o_fr, s_fr = fused_recurrent_kda(
                kern_q[None], kern_k[None], kern_v[None],
                kern_g[None], kern_beta[None],
                scale=scale, initial_state=init_state, output_final_state=True,
            )
        else:
            from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
            q_b, k_b, v_b, g_b, beta_b = KDAAttnBackend._unpack_varlen(
                kern_q, kern_k, kern_v, kern_g,
                kern_beta if kern_beta.ndim == 2 else kern_beta.reshape(T, H),
                cu,
            )
            o_fr, s_fr = fused_recurrent_kda(
                q_b, k_b, v_b, g_b, beta_b,
                scale=scale, initial_state=init_state, output_final_state=True,
            )
            o_fr = KDAAttnBackend._repack_varlen(o_fr, cu, T)[None]

        o_fr_cmp = o_fr[0] if o_fr.ndim == 4 else o_fr
        rows.append(compare_stage("6. KDA output (fused_rec)", o_fr_cmp, case, "intermediates__o_kda_fused_recurrent"))
        rows.append(compare_stage("7. Recurrent state (fused)", s_fr, case, "intermediates__recurrent_state_fused_recurrent"))
        if o_kda is None:
            o_kda = o_fr_cmp
    except Exception as e:
        rows.append({"label": f"6. KDA fused: {type(e).__name__}: {e}", "skip": True})

    # === Stage 8: Output gate ===
    g_out = matmul(matmul(hidden, w("g_a_proj_w")), w("g_b_proj_w")).reshape(T, H, D)
    rows.append(compare_stage("8. Output gate (g_out)", g_out, case, "intermediates__g_out"))

    # === Stage 9: Output norm ===
    if isolated:
        norm_o = gpu_or_jax("intermediates__o_kda_chunk", o_kda) if o_kda is not None else None
        norm_g = gpu_or_jax("intermediates__g_out", g_out)
    else:
        norm_o = o_kda
        norm_g = g_out

    o_normed = None
    if norm_o is not None:
        o_norm_w = jnp.asarray(ws["o_norm_w"], dtype=jnp.float32)
        x_f32 = norm_o.astype(jnp.float32)
        var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_n = x_f32 * jax.lax.rsqrt(var + eps) * o_norm_w
        o_normed = (x_n * jax.nn.sigmoid(norm_g.astype(jnp.float32))).astype(dtype)
        rows.append(compare_stage("9. Output norm (o_norm)", o_normed, case, "intermediates__o_norm"))
    else:
        rows.append({"label": "9. Output norm", "skip": True})

    # === Stage 10: Final output ===
    if isolated:
        final_in = gpu_or_jax("intermediates__o_norm", o_normed)
    else:
        final_in = o_normed

    if final_in is not None:
        if final_in.ndim == 4:
            final_in = final_in[0]
        final = matmul(final_in.reshape(T, proj), w("o_proj_w"))
        rows.append(compare_stage("10. Final output (o_proj)", final, case, "out_fp32"))
    else:
        rows.append({"label": "10. Final output", "skip": True})

    dtype_label = "fp32" if dtype == jnp.float32 else "bf16"
    prec_label = str(precision).split(".")[-1] if precision else "DEFAULT"
    mode_label = "ISOLATED" if isolated else "CUMULATIVE"
    print_table(rows, f"{case_name} ({dtype_label}, T={T}, N={N}, precision={prec_label}) [{mode_label}]")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KDA per-stage precision analysis (matmul-only / isolated / cumulative)"
    )
    parser.add_argument(
        "--mode", required=True, choices=["matmul-only", "isolated", "cumulative"],
        help="Analysis mode",
    )
    parser.add_argument("--layer", default="L22", help="Layer dir (default: L22)")
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

    if args.mode == "matmul-only":
        run_matmul_only(layer_dir, dtype)
        return

    precision = PRECISIONS[args.precision]
    isolated = args.mode == "isolated"

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
        run_pipeline(ws, case, cn, dtype, precision, isolated=isolated)


if __name__ == "__main__":
    main()

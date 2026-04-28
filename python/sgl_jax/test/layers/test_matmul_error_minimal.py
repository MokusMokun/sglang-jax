"""Minimal matmul error reproduction: L22 q_proj on TPU vs GPU dump.

One matmul: hidden_states @ q_proj_w → compare against GPU intermediates__q_proj.

Usage:
    python test_matmul_error_minimal.py                # fp32
    python test_matmul_error_minimal.py --dtype bf16   # bf16
"""

import os
import numpy as np
import jax.numpy as jnp

DUMP = os.environ.get("KDA_DUMP_DIR", "/models/yuhao/kimi-linear/kda_module")
LAYER = "L22"
CASE = "single_T128"


def dist(name: str, arr: np.ndarray):
    """Print distribution stats for a tensor."""
    a = arr.astype(np.float32)
    pcts = np.percentile(np.abs(a), [50, 90, 99, 100])
    print(f"  {name:>12s}  shape={str(arr.shape):<16s}  "
          f"mean={a.mean():+.4e}  std={a.std():.4e}  "
          f"|abs| p50={pcts[0]:.4e} p90={pcts[1]:.4e} p99={pcts[2]:.4e} max={pcts[3]:.4e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = parser.parse_args()
    dtype = jnp.float32 if args.dtype == "fp32" else jnp.bfloat16

    layer_dir = os.path.join(DUMP, LAYER)
    weights = np.load(os.path.join(layer_dir, "weights.npz"), allow_pickle=True)
    case = np.load(os.path.join(layer_dir, f"case_{CASE}.npz"), allow_pickle=True)

    # Load inputs
    W = jnp.asarray(weights["weights__q_proj.weight"].T, dtype=dtype)  # [2304, 4096]
    x = jnp.asarray(case["hidden_states"], dtype=dtype)                # [1, 128, 2304]
    if x.ndim == 3:
        x = x[0]                                                       # [128, 2304]
    ref = np.asarray(case["intermediates__q_proj"])                     # GPU output
    if ref.ndim == 3:
        ref = ref[0]

    # --- Data distributions ---
    print(f"{'=' * 78}")
    print(f"  Data distributions ({args.dtype})")
    print(f"{'=' * 78}")
    dist("x (input)", np.asarray(x))
    dist("W (weight)", np.asarray(W))
    dist("ref (GPU)", ref)

    # One matmul
    out = x @ W  # [128, 4096]
    out_f32 = np.asarray(out, dtype=np.float32)

    dist("out (TPU)", out_f32)

    # --- Error distribution ---
    diff = np.abs(out_f32 - ref)
    rel = diff / (np.abs(ref) + 1e-8)

    print(f"\n{'=' * 78}")
    print(f"  Error analysis")
    print(f"{'=' * 78}")
    dist("abs_error", diff)
    dist("rel_error", rel)


if __name__ == "__main__":
    main()

# Pallas KDA Kernel: NaN from Gate Magnitude Overflow

**Date**: 2026-04-26 (fixed 2026-04-27)
**Affected**: `chunk_kda_fwd` in `python/sgl_jax/srt/kernels/kda/kda.py`
**Kernel origin**: PR #964 (`@pathfinder-pf`)
**Fix**: [commit b4c4249](https://github.com/MokusMokun/sglang-jax/commit/b4c4249b2116a00b86324508bba5e4e835520a70)
**Status**: Fixed (with one known edge-case fallback)

---

## Summary

The Pallas chunked KDA kernel produced NaN whenever activated gate values exceeded ~|10| per step. Real HF weights (`moonshotai/Kimi-Linear-48B-A3B-Instruct`) produce gate values in `[-1922, 0]` — far past this threshold. The naive JAX kernel handled the same inputs correctly.

The root cause was **intermediate exp2 overflow** in two places: the intra-chunk attention computation and the inter-chunk state propagation. Both used normalization strategies that broke down when cumulative gate magnitudes exceeded fp32's representable exponent range (~127 for exp2).

The fix replaces both normalization strategies with monotonicity-aware reference points that guarantee exp2 arguments are always non-positive, keeping results in `(0, 1]`.

---

## Root Cause

### Background: gate values in KDA

KDA's gating mechanism produces per-step decay factors. The activated gate values are:

```
g_act = -exp(A_log) * softplus(g + dt_bias)
```

These are always non-positive (decay only). With real weights, individual steps produce values around `-60`. The kernel works in log2-space, converting via `scale = 1/ln(2)`, so each step contributes ~`-86.6` in log2-space.

The chunked kernel accumulates these into cumsums over chunks of 64 steps. Since gates are monotonically non-positive, the cumsum is **monotonically decreasing**:

```
g_cumsum[0] ≥ g_cumsum[1] ≥ ... ≥ g_cumsum[63]
```

### The overflow mechanism

Both broken code paths needed to compute `exp2(g_cumsum[i])` but tried to avoid direct exponentiation of large negative values by using a reference point. The problem was that the chosen reference points produced **positive** exp2 arguments that overflowed:

**Intra-chunk (sub-chunk attention):**

The kernel split each chunk into NC sub-chunks of size BC=16. It picked a midpoint reference `gn = g[BC//2]` and computed:

```python
q_eg  = q * exp2(g[i] - gn)    # query side
k_eng = k * exp2(gn - g[j])    # key side — gn > g[j], so argument is POSITIVE
```

With gate ~ -86.6/step, a 16-step sub-chunk accumulates ~-1386. The midpoint `gn` sits at ~-693, so `gn - g[15]` ≈ +693 — `exp2(693)` overflows to inf, producing NaN in the dot product.

**Inter-chunk (state propagation):**

Similar issue — the code used `g_mid = (g[0] + g[-1]) * 0.5` as reference. With large gates, `exp2(g_mid)` itself could overflow.

### Why the naive kernel was fine

`naive_recurrent_kda` processes one timestep at a time: `S = S * exp(g_t)`. Each `exp(-60)` ≈ `8.7e-27` — a tiny but valid float. No cumsum, no reference point, no overflow.

---

## The Fix

Three changes in the kernel, plus one fallback in the backend. All exploit the same insight: **because gate cumsums are monotonically decreasing, `g[i] - g[j] ≤ 0` for `i ≥ j`, so `exp2(g[i] - g[j])` is always in `(0, 1]`.**

### 1. Intra-chunk: direct pairwise difference (eliminates sub-chunk loops)

**Before:** NC×NC nested loop over sub-chunks with midpoint reference `gn`, producing `exp2(gn - g[j])` that overflows.

**After:** Compute the full `BT × BT` pairwise difference matrix directly:

```python
g_diff = g[i, k] - g[j, k]          # shape [BT, BT, K]
g_diff = where(causal, g_diff, -126)  # mask anti-causal to safe value
decay  = exp2(max(g_diff, -126))      # always in (0, 1] for causal pairs

Aqk = scale * sum_k(q[i,k] * decay[i,j,k] * k[j,k])  # [BT, BT]
L   = sum_k(k[i,k] * decay[i,j,k] * k[j,k]) * beta    # [BT, BT], strict causal
```

The key insight: for causal pairs (`i ≥ j`), `g_cumsum[i] ≤ g_cumsum[j]`, so `g_diff ≤ 0` and `exp2` never exceeds 1. The `max(..., -126)` clamp prevents underflow to denormals but doesn't affect correctness since `exp2(-126) ≈ 1.2e-38` is effectively zero.

This also simplifies the code: the NC×NC sub-chunk double loop is replaced by a single broadcast computation.

### 2. Inter-chunk: first-position reference instead of midpoint

**Before:**

```python
g_mid = (g[0] + g[-1]) * 0.5          # midpoint — can be large negative
qg    = q * exp2(g - g_mid)           # g - g_mid can be positive
h_s   = h * exp2(g_mid)               # exp2(large negative) underflows
```

**After:**

```python
g_ref = g[0]                           # first position = largest cumsum value
qg    = q * exp2(max(g - g_ref, -126)) # g[t] - g[0] ≤ 0 always, safe
h_s   = h * exp2(max(g_ref, -126))     # single reference scaling
```

Since `g[0]` is the maximum of the monotonically decreasing cumsum, `g[t] - g[0] ≤ 0` for all `t`, keeping exp2 in `(0, 1]`.

### 3. Padding fix for `use_gate_in_kernel=True`

An independent bug: `_align_seqs` pads gate values with 0 for variable-length sequence packing. But when gates are activated inside the kernel, `softplus(0 + dt_bias) ≠ 0` — padding positions get non-zero gate activation, corrupting `g_last` (state propagation) and `kg` (state update).

Fix: replace padding value with `-1e4`, making `softplus(-1e4 + dt_bias) ≈ 0` and neutralizing padding positions.

```python
if use_gate_in_kernel:
    valid_mask = ...  # True for real token positions, False for padding
    g = where(valid_mask, g, -1e4)
```

### 4. Backend fallback for short-sequence batches

When total packed tokens ≤ `chunk_size` (64) and there are multiple sequences, every sequence is shorter than one chunk. The kernel pads each to chunk_size, but the padding causes precision loss in the inter-chunk state contribution due to exp2 underflow. The backend falls back to the naive kernel in this case:

```python
BT = 64
if T > BT or N <= 1:
    return self._forward_extend_pallas(...)
return self._forward_extend_naive(...)
```

---

## Why Simpler Fixes Don't Work

Several approaches were tested during diagnosis and all failed:

| Approach | Why it fails |
|----------|-------------|
| **Clamp gate values** to `[-C, 0]` | Even clamp=3 still NaN (cumsum of 64 × 3 = 192 still exceeds exp2 range). Clamping small enough to work (~1.5) changes model semantics. |
| **Per-chunk cumsum normalization** (subtract first position) | Only shifts the range by one step's contribution. Intra-chunk exp2 on 16-step sub-chunks still overflows. |
| **`safe_gate` parameter** | Only changes which position is used as reference (`g[BC//2]` vs `g[0]`), doesn't address the fundamental sign problem of `gn - g[j]`. |
| **`use_gate_in_kernel` toggle** | Both paths feed the same large activated gate values into the same problematic exp2 computations. |

The fix works because it's the only approach that guarantees **all exp2 arguments are non-positive** — a structural invariant, not a magnitude-dependent heuristic.

---

## Additional Bugs Found During Testing

1. **Missing `max_T` in `chunk_local_cumsum_vector`** (line 216): `prepare_chunk_indices(cu_seqlens, BT)` called without `max_T`, causing `TypeError`.
2. **Inverted `chunk_indices` guard in `chunk_kda_fwd`** (line 1206): `if chunk_indices is not None:` should be unconditional — `chunk_indices` must always be computed for varlen operation.

---

## Reproduction

Tested on TPU v6e-4 with JAX 0.8.1 and 0.9.2. Before fix, NaN at gate_scale ≥ 10:

```python
import jax, jax.numpy as jnp
from sgl_jax.srt.kernels.kda import chunk_kda

H, K, V, T = 32, 128, 128, 128
q = jax.random.normal(jax.random.PRNGKey(0), (1, T, H, K), dtype=jnp.float32) * 0.1
k = jax.random.normal(jax.random.PRNGKey(1), (1, T, H, K), dtype=jnp.float32) * 0.1
v = jax.random.normal(jax.random.PRNGKey(2), (1, T, H, V), dtype=jnp.float32) * 0.1
g = -jnp.abs(jax.random.normal(jax.random.PRNGKey(3), (1, T, H, K), dtype=jnp.float32)) * 1000
beta = jax.random.uniform(jax.random.PRNGKey(4), (1, T, H), dtype=jnp.float32)
cu = jnp.array([0, T], dtype=jnp.int32)
init = jnp.zeros((1, H, K, V), dtype=jnp.float32)

o, fs, *_ = chunk_kda(q, k, v, g, beta, scale=K**-0.5,
    initial_state=init, output_final_state=True, cu_seqlens=cu)
print(f"NaN: {jnp.isnan(o).any()}")  # True before fix, False after
```

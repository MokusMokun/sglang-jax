# KDA Phase B: Numerical Alignment

**Updated**: 2026-04-28 | **Branch**: `sub3/layer-tests`

| Item | Value |
|------|-------|
| TPU | v6e-4 (`sky-efe2-yuhao`), JAX 0.10.0, libtpu 0.0.40 |
| GPU reference | H100, fla `chunk_kda` (`force_mode="chunk"`) |
| Model | `moonshotai/Kimi-Linear-48B-A3B-Instruct` |
| Dumps | `/models/yuhao/kimi-linear/kda_module/{L0,L6,L13,L22}/` |
| Prefill kernel | Pallas chunked (`T > 64` or `N ≤ 1`), naive recurrent fallback |
| Decode kernel | `fused_recurrent_kda` (naive recurrent) |
| State pool | `RecurrentStatePool` (production, TP-sharded) |
| Mesh | `Explicit` axis type, TP=1; kernel inputs unsharded via `reshard` |

**Test matrix**: 12 prefill cases × 2 dtypes + 3 decode cases × 2 dtypes = 30 tests/layer, 120 total across 4 layers. All single_T1 cases skipped (GPU chunk kernel outputs zero for T < 64).

**Precision context**: Production runs in **bf16**. FP32 tests verify logic/pipeline correctness; bf16 results represent production-level precision.

**Metrics**: `max_abs = max(|TPU - GPU|)`, `mean_abs = mean(|TPU - GPU|)`, computed elementwise over the full output tensor. Pass/fail uses `np.testing.assert_allclose(atol, rtol)` where `|TPU - GPU| ≤ atol + rtol × |GPU|`.

**Tolerance tiers** (tight first, loose as fallback):

| | Tight | Loose |
|---|---|---|
| Prefill FP32 | atol=2e-3, rtol=5e-3 | atol=3e-2, rtol=2e-2 |
| Prefill BF16 | atol=3e-3, rtol=5e-3 | atol=7e-2, rtol=2e-2 |
| Decode FP32 | atol=1e-3, rtol=1e-3 | atol=1e-2, rtol=1e-2 |
| Decode BF16 | atol=2e-3, rtol=2e-3 | atol=2e-2, rtol=2e-2 |

---

## Results Summary

Matmul of all results below use `Precision.DEFAULT` (single-pass bf16 matmul on TPU MXU).

### Cross-Layer (Prefill)

| Layer | FP32 worst | BF16 worst | Tier |
|-------|-----------|-----------|------|
| L0 | 1.29e-03 | 1.95e-03 | tight |
| L6 | 1.47e-02 | 2.34e-02 | loose |
| L13 | 1.36e-02 | 1.56e-02 | loose |
| L22 | 2.70e-02 | 6.25e-02 | loose |

### Cross-Layer (Decode)

| Layer | FP32 worst | BF16 worst |
|-------|-----------|-----------|
| L0 | < 1e-3 | < 2e-3 |
| L6 | 2.13e-03 | 3.91e-03 |
| L13 | 2.80e-03 | 3.91e-03 |
| L22 | 5.26e-03 | 1.56e-02 |

### Overall

| Layer | Passed | Skipped | Tight | Loose |
|-------|--------|---------|-------|-------|
| L0 | 28 | 2 | 28 | 0 |
| L6 | 28 | 2 | 0 | 28 |
| L13 | 28 | 2 | 0 | 28 |
| L22 | 28 | 2 | 0 | 28 |
| **Total** | **112** | **8** | **28** | **84** |

Error grows ~20x L0→L22 (prefill), ~4x (decode) — deeper layers have larger weight magnitudes, amplifying cross-device matmul precision differences. Decode error is smaller (single token, no sequence accumulation).

---

## Error Source Analysis

### Isolated Per-Stage Comparison

Each stage receives the **GPU dump intermediate** as input, isolating that stage's own JAX-vs-GPU error from upstream accumulation.

**L22, single_T128, FP32**
> Note: 
> `max_abs/mean_abs` columns use `Precision.DEFAULT`; 
> `HIGH` columns use `Precision.HIGH` — 3-pass bf16 matmul, see [Minimal Reproduction](#minimal-reproduction)

| Stage | Input source | max_abs | mean_abs | HIGH max_abs | HIGH mean_abs |
|-------|-------------|---------|----------|-------------|--------------|
| Q projection | hidden_states | <u>2.73e-02</u> | <u>2.60e-03</u> | 8.44e-05 | 7.73e-06 |
| K projection | hidden_states | <u>2.61e-02</u> | <u>2.49e-03</u> | 7.78e-05 | 7.41e-06 |
| V projection | hidden_states | <u>2.23e-02</u> | <u>3.03e-03</u> | 6.72e-05 | 9.00e-06 |
| Q conv+SiLU | GPU q_proj | 1.91e-06 | 5.38e-09 | — | — |
| K conv+SiLU | GPU k_proj | 3.81e-06 | 7.29e-09 | — | — |
| V conv+SiLU | GPU v_proj | 1.43e-06 | 1.58e-08 | — | — |
| Gate (fused_kda_gate) | hidden_states | **1.65e+00** | **5.61e-03** | 2.90e-03 | 1.74e-05 |
| Beta (sigmoid) | hidden_states | 2.84e-03 | 3.64e-04 | — | — |
| KDA output (chunk) | GPU post-conv + g + beta | 1.61e-04 | 1.81e-06 | — | — |
| KDA output (fused_rec) | GPU post-conv + g + beta | 1.19e-07 | 9.50e-10 | — | — |
| Recurrent state (fused) | GPU post-conv + g + beta | 8.34e-07 | 5.96e-09 | — | — |
| Output gate (g_out) | hidden_states | <u>3.46e-02</u> | <u>2.42e-03</u> | 9.97e-05 | 7.11e-06 |
| Output norm | GPU o_kda + GPU g_out | 7.15e-07 | 5.31e-09 | — | — |
| Final output (o_proj) | GPU o_norm | 1.14e-02 | 8.55e-04 | 2.19e-05 | 2.52e-06 |

**L22, single_T128, BF16:**

| Stage | Input source | max_abs | mean_abs |
|-------|-------------|---------|----------|
| Q projection | hidden_states | 3.80e-02 | <u>3.56e-03</u> |
| K projection | hidden_states | <u>5.11e-02</u> | <u>3.42e-03</u> |
| V projection | hidden_states | 4.17e-02 | <u>4.16e-03</u> |
| Q conv+SiLU | GPU q_proj | <u>7.77e-02</u> | 2.46e-04 |
| K conv+SiLU | GPU k_proj | <u>1.24e-01</u> | 3.16e-04 |
| V conv+SiLU | GPU v_proj | 3.76e-02 | 7.06e-04 |
| Gate (fused_kda_gate) | hidden_states | **2.46e+00** | **7.35e-03** |
| Beta (sigmoid) | hidden_states | 2.96e-03 | 4.32e-04 |
| KDA output (chunk) | GPU post-conv + g + beta | 9.42e-04 | 5.20e-06 |
| KDA output (fused_rec) | GPU post-conv + g + beta | 7.92e-04 | 4.12e-06 |
| Recurrent state (fused) | GPU post-conv + g + beta | 6.01e-03 | 3.49e-05 |
| Output gate (g_out) | hidden_states | <u>4.49e-02</u> | <u>2.97e-03</u> |
| Output norm | GPU o_kda + GPU g_out | 1.72e-02 | 1.97e-04 |
| Final output (o_proj) | GPU o_norm | 1.32e-02 | 1.14e-03 |

BF16 conv error (~8e-2) is larger because GPU dump intermediates are fp32 while JAX conv runs in bf16 — the error is bf16 truncation, not algorithmic.

### Error Pipeline Summary

| Path | Stage | Isolated error | Source |
|------|-------|---------------|--------|
| hidden → q/k/v | projection matmul | <u>~2e-2</u> | `Precision.DEFAULT` bf16 truncation |
| q/k/v → heads | conv+SiLU (K=4) | ~2e-6 | negligible |
| heads → normed | L2 norm | — | elementwise |
| hidden → raw_gate | gate projection (2 matmuls) | <u>~2e-2</u> | `Precision.DEFAULT` bf16 truncation |
| raw_gate → g | fused_kda_gate | **~2e+0** | exp(A_log) amplifies matmul error |
| hidden → beta | sigmoid | ~3e-3 | `Precision.DEFAULT` bf16 truncation |
| normed+g+beta → o | fused_recurrent kernel | ~1e-7 | near bit-exact |
| normed+g+beta → o | chunk/Pallas kernel | ~2e-4 | chunked accumulation |
| hidden → g_out | output gate projection | <u>~3e-2</u> | `Precision.DEFAULT` bf16 truncation |
| o+g_out → o_norm | GatedRMSNorm | ~7e-7 | elementwise |
| o_norm → output | o_proj matmul | <u>~1e-2</u> | `Precision.DEFAULT` bf16 truncation |

### Key Findings

1. **All error originates from matmul stages.** Input projections (`[T, 2304] @ [2304, 4096]`, ~2e-2) and output projection (`[T, 4096] @ [4096, 2304]`, ~1e-2) dominate. Conv (~2e-6), fused recurrent (~1e-7), and GatedRMSNorm (~7e-7) are at or near machine epsilon.

2. **Gate max_abs (1.65e+00) is a scaling artifact.** Gate computes `-exp(A_log) * softplus(raw_gate + dt_bias)`. L22's A_log reaches 4.14, giving exp(A_log) = 62.7×. This amplifies the ~2e-2 matmul error: 62.7 × 2.6e-2 ≈ 1.63. Relative error is only ~0.2%.

3. **FP32 matmul error is caused by `Precision.DEFAULT`**, which truncates fp32 inputs to bf16 before multiplication on TPU MXU. `Precision.HIGH` (3-pass bf16) reduces error ~300× to ~8e-5, confirming the JAX implementation is logically correct. This does not affect bf16 production — bf16 inputs are already at native MXU precision, so `precision` has no effect.

### Minimal Reproduction

Single matmul `hidden_states [128, 2304] @ q_proj_w [2304, 4096]` on L22, GPU dump vs TPU (`jax.lax.dot`):

| Precision | FP32 max_abs | FP32 mean_abs | BF16 max_abs | BF16 mean_abs |
|---|---|---|---|---|
| DEFAULT (1-pass) | 2.73e-02 | 2.60e-03 | 3.80e-02 | 3.56e-03 |
| HIGH (3-pass) | 8.44e-05 | 7.73e-06 | 3.80e-02 | 3.56e-03 |
| HIGHEST (6-pass) | 8.58e-06 | 6.64e-07 | 3.80e-02 | 3.56e-03 |

FP32 inputs: `HIGH` improves ~300×, `HIGHEST` ~3000× (near fp32 machine epsilon), confirming pipeline correctness. BF16 inputs: no change — `precision` only affects fp32→bf16 truncation in the multiplier; bf16 inputs are already at native MXU precision, so these results represent the production precision floor.

Script: `test/layers/test_kda_precision_analysis.py --mode matmul-only`.
Relavant Material: https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.Precision

### Cumulative Error (DEFAULT vs HIGH)

Full pipeline cumulative error on L22, single_T128, FP32 — each stage uses JAX output from the previous stage (not GPU dump):

| Stage | DEFAULT max_abs | HIGH max_abs |
|-------|----------------|-------------|
| Q projection | 2.73e-02 | 8.44e-05 |
| Q conv+SiLU | 3.49e-02 | 8.25e-05 |
| Gate (fused_kda_gate) | 1.65e+00 | 2.90e-03 |
| Beta (sigmoid) | 2.84e-03 | 7.09e-06 |
| KDA output (chunk) | 3.75e-04 | 1.53e-04 |
| Output gate (g_out) | 3.46e-02 | 9.97e-05 |
| Output norm | 1.89e-02 | 5.38e-03 |
| **Final output (E2E)** | **1.78e-02** | **5.75e-03** |

HIGH reduces E2E error ~3× (1.78e-02 → 5.75e-03). Under HIGH, matmul stages no longer dominate — the residual E2E error is driven by the chunk kernel (1.53e-04) amplified through gate and norm.

Script: `test/layers/test_kda_precision_analysis.py --mode cumulative --precision high`.

---

## Pallas Kernel

Pallas chunked kernel (`chunk_kda_fwd`) is enabled by default. Shape-based routing: Pallas when `T > 64` or `N ≤ 1`, naive fallback for short multi-sequence batches.

| Case category | Pallas vs Naive |
|---------------|----------------|
| Single seq, T ≤ 256 | tie |
| Single seq, T = 1024 | Pallas ~1.05× better |
| **Varlen (multi-seq packed)** | **Pallas 1.2–1.4× better** |
| With initial state | tie |

Pallas's main advantage is varlen packed scenarios (parallel chunk processing vs naive per-sequence loop).

---

## Per-Layer Details

### L0 — Best Case (all tight)

**FP32** (11 passed, 1 skipped):

| Case | max_abs | mean_abs | Kernel |
|------|---------|----------|--------|
| single_T1 | — | — | SKIP |
| single_T8 | 6.36e-04 | 5.33e-05 | naive |
| single_T64 | 7.09e-04 | 5.85e-05 | naive |
| single_T65 | 7.10e-04 | 5.84e-05 | pallas |
| single_T128 | 7.10e-04 | 5.89e-05 | pallas |
| single_T256 | 8.45e-04 | 5.86e-05 | pallas |
| single_T1024 | 9.47e-04 | 5.90e-05 | pallas |
| varlen_balanced_4x32 | 7.26e-04 | 5.76e-05 | naive |
| varlen_unbalanced | 6.98e-04 | 5.75e-05 | naive |
| varlen_single_T128 | 7.10e-04 | 5.89e-05 | pallas |
| single_T128_initstate | 7.61e-04 | 6.03e-05 | pallas |
| varlen_initstate | 1.29e-03 | 7.99e-05 | pallas |

**BF16** (11 passed, 1 skipped): max_abs clusters at 9.77e-04 (= 1/1024, bf16 ULP near 1.0). Recurrence upcasts to fp32 internally, so bf16 truncation only affects projection/conv.

| Case | max_abs | mean_abs | Kernel |
|------|---------|----------|--------|
| single_T1 | — | — | SKIP |
| single_T8 | 4.88e-04 | 5.95e-05 | naive |
| single_T64 | 9.77e-04 | 6.16e-05 | naive |
| single_T65 | 9.77e-04 | 6.16e-05 | pallas |
| single_T128 | 9.77e-04 | 6.39e-05 | pallas |
| single_T256 | 9.77e-04 | 6.55e-05 | pallas |
| single_T1024 | 9.77e-04 | 6.81e-05 | pallas |
| varlen_balanced_4x32 | 9.77e-04 | 6.74e-05 | naive |
| varlen_unbalanced | 9.77e-04 | 6.69e-05 | naive |
| varlen_single_T128 | 9.77e-04 | 6.39e-05 | pallas |
| single_T128_initstate | 1.46e-03 | 6.73e-05 | pallas |
| varlen_initstate | 1.95e-03 | 8.87e-05 | pallas |

**Decode** (6 passed, all tight):

| Case | FP32 max_abs | BF16 max_abs |
|------|-------------|-------------|
| single_T8 | < 1e-3 | < 2e-3 |
| single_T128 | < 1e-3 | < 2e-3 |
| single_T128_initstate | < 1e-3 | < 2e-3 |

### L22 — Worst Case (all loose)

**FP32** (11 passed, 1 skipped):

| Case | max_abs | mean_abs | Kernel |
|------|---------|----------|--------|
| single_T1 | — | — | SKIP |
| single_T8 | 8.96e-03 | 1.40e-03 | naive |
| single_T64 | 1.78e-02 | 1.85e-03 | naive |
| single_T65 | 1.78e-02 | 1.84e-03 | pallas |
| single_T128 | 1.78e-02 | 1.86e-03 | pallas |
| single_T256 | 2.42e-02 | 1.91e-03 | pallas |
| single_T1024 | 2.70e-02 | 1.94e-03 | pallas |
| varlen_balanced_4x32 | 1.81e-02 | 1.80e-03 | naive |
| varlen_unbalanced | 1.66e-02 | 1.80e-03 | naive |
| varlen_single_T128 | 1.78e-02 | 1.86e-03 | pallas |
| single_T128_initstate | 2.41e-02 | 1.88e-03 | pallas |
| varlen_initstate | 2.39e-02 | 2.17e-03 | pallas |

**BF16** (11 passed, 1 skipped):

| Case | max_abs | mean_abs | Kernel |
|------|---------|----------|--------|
| single_T1 | — | — | SKIP |
| single_T8 | 1.17e-02 | 1.54e-03 | naive |
| single_T64 | 3.13e-02 | 1.99e-03 | naive |
| single_T65 | 3.13e-02 | 1.99e-03 | pallas |
| single_T128 | 3.13e-02 | 2.07e-03 | pallas |
| single_T256 | 3.13e-02 | 2.11e-03 | pallas |
| single_T1024 | 6.25e-02 | 2.12e-03 | pallas |
| varlen_balanced_4x32 | 3.13e-02 | 2.25e-03 | naive |
| varlen_unbalanced | 3.13e-02 | 2.25e-03 | naive |
| varlen_single_T128 | 3.13e-02 | 2.07e-03 | pallas |
| single_T128_initstate | 2.34e-02 | 2.12e-03 | pallas |
| varlen_initstate | 3.13e-02 | 2.43e-03 | pallas |

**Decode** (6 passed, all loose):

| Case | FP32 max_abs | BF16 max_abs |
|------|-------------|-------------|
| single_T8 | 4.78e-03 | 1.17e-02 |
| single_T128 | 5.26e-03 | 1.56e-02 |
| single_T128_initstate | 5.26e-03 | 1.56e-02 |

### L6, L13

All 28 non-skip tests pass at loose tolerance. Per-case tables omitted — see cross-layer summary for worst-case numbers. Full data in `test_kda_backend.py` test output.

---

## Notes

- **T=1 skip**: GPU chunk kernel outputs all zeros for T < chunk_size (64). TPU naive kernel correctly produces non-zero output. Test verifies no NaN + non-zero, then skips comparison.
- **GPU chunk vs fused_recurrent baseline**: Even on GPU the two kernels differ — attention output max_abs_diff = 1.21e-04, recurrent state = 6.33e-04. This sets a floor for cross-kernel comparison.

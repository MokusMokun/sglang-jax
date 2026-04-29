# KDA Phase B: Numerical Alignment

**Updated**: 2026-04-29 | **Branch**: `merge/kda-validation`

| Item | Value |
|------|-------|
| TPU | v6e-4 (`sky-efe2-yuhao`), JAX 0.10.0, libtpu 0.0.40 |
| GPU reference | H100, fla `chunk_kda` (`force_mode="chunk"`) |
| Model | `moonshotai/Kimi-Linear-48B-A3B-Instruct` |
| Dumps (isolated) | `/models/yuhao/kimi-linear/kda_module/{L0,...,L25}/` (20 KDA layers) |
| Dumps (full-model) | `/models/yuhao/kimi-linear/kda_full_model_dump/` (20 layers, real activations) |
| Prefill kernel | Pallas chunked (`T > 64` or `N ≤ 1`), naive recurrent fallback |
| Decode kernel | `fused_recurrent_kda` (naive recurrent) |
| State pool | `RecurrentStatePool` (production, TP-sharded) |
| Mesh | `Explicit` axis type, TP=1; kernel inputs unsharded via `reshard` |

**Test matrix**: Isolated: 12 prefill × 2 dtypes + 3 decode × 2 dtypes = 30 tests/layer, 120 total across 4 layers. Full-model: 20 KDA layers × 2 dtypes = 40 tests. All single_T1 cases skipped (GPU chunk kernel outputs zero for T < 64).

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

Worst case across all 12 cases per layer. `max_abs` and `mean_abs` are from the same worst-case case.

| Layer | FP32 max_abs | FP32 mean_abs | BF16 max_abs | BF16 mean_abs | Tier | Worst case |
|-------|-------------|--------------|-------------|--------------|------|------------|
| L0 | 1.29e-03 | 7.99e-05 | 2.20e-03 | 1.04e-04 | tight | varlen_initstate / single_T128_initstate |
| L6 | 1.44e-02 | 4.83e-04 | 3.12e-02 | 8.14e-04 | loose | single_T1024 |
| L13 | 1.91e-02 | 9.63e-04 | 3.12e-02 | 1.55e-03 | loose | varlen_balanced_4x32 / single_T1024 |
| L22 | 2.84e-02 | 1.94e-03 | 6.25e-02 | 3.39e-03 | loose | single_T1024 |

### Cross-Layer (Decode)

Worst case across 3 decode cases per layer.

| Layer | FP32 max_abs | FP32 mean_abs | BF16 max_abs | BF16 mean_abs |
|-------|-------------|--------------|-------------|--------------|
| L0 | 2.61e-04 | 4.56e-05 | 7.32e-04 | 1.30e-04 |
| L6 | 2.66e-03 | 3.70e-04 | 5.86e-03 | 1.04e-03 |
| L13 | 3.22e-03 | 6.98e-04 | 8.79e-03 | 1.89e-03 |
| L22 | 5.12e-03 | 1.11e-03 | 1.56e-02 | 2.43e-03 |

### Overall

| Layer | Passed | Skipped | Tight | Loose |
|-------|--------|---------|-------|-------|
| L0 | 28 | 2 | 28 | 0 |
| L6 | 28 | 2 | 0 | 28 |
| L13 | 28 | 2 | 0 | 28 |
| L22 | 28 | 2 | 0 | 28 |
| **Total** | **112** | **8** | **28** | **84** |

Error grows ~20x L0→L22 (prefill), ~20x (decode) — deeper layers have larger weight magnitudes, amplifying cross-device matmul precision differences. Decode error is smaller (single token, no sequence accumulation).

---

## Error Source Analysis

Two analysis modes on L22, single_T128. **Isolated**: each stage independently receives GPU dump as input — measures that stage's own error. **Accumulated**: production forward pass — each stage feeds from the previous stage's output, showing real error propagation. Script: `test/layers/test_kda_precision_analysis.py`.

### Per-Stage Isolated Error

Each stage receives the **GPU dump intermediate** as input, isolating that stage's own JAX-vs-GPU error. Script: `--mode isolated`.

**L22, single_T128, FP32 — DEFAULT vs HIGH**

| Stage | Input source | DEFAULT max_abs | DEFAULT mean_abs | HIGH max_abs | HIGH mean_abs |
|-------|-------------|----------------|-----------------|-------------|--------------|
| Q projection | hidden_states | <u>2.73e-02</u> | <u>2.60e-03</u> | 8.44e-05 | 7.73e-06 |
| K projection | hidden_states | <u>2.61e-02</u> | <u>2.49e-03</u> | 7.78e-05 | 7.41e-06 |
| V projection | hidden_states | <u>2.23e-02</u> | <u>3.03e-03</u> | 6.72e-05 | 9.00e-06 |
| Q conv+SiLU | GPU q_proj | 1.91e-06 | 5.38e-09 | 1.91e-06 | 5.38e-09 |
| K conv+SiLU | GPU k_proj | 3.81e-06 | 7.29e-09 | 3.81e-06 | 7.29e-09 |
| V conv+SiLU | GPU v_proj | 1.43e-06 | 1.58e-08 | 1.43e-06 | 1.58e-08 |
| Gate (fused_kda_gate) | hidden_states | **1.65e+00** | **5.61e-03** | 2.90e-03 | 1.74e-05 |
| Beta (sigmoid) | hidden_states | 2.84e-03 | 3.64e-04 | 7.09e-06 | 1.10e-06 |
| KDA output (fused_rec) | GPU post-conv + g + beta | 1.19e-07 | 9.50e-10 | 1.19e-07 | 9.50e-10 |
| Recurrent state (fused) | GPU post-conv + g + beta | 8.34e-07 | 5.96e-09 | 8.34e-07 | 5.96e-09 |
| Output gate (g_out) | hidden_states | <u>3.46e-02</u> | <u>2.42e-03</u> | 9.97e-05 | 7.11e-06 |
| Output norm | GPU o_kda + GPU g_out | 7.15e-07 | 5.31e-09 | 7.15e-07 | 5.31e-09 |
| Final output (o_proj) | GPU o_norm | 1.14e-02 | 8.55e-04 | 2.19e-05 | 2.52e-06 |

Conv (~2e-6), fused recurrent (~1e-7), and GatedRMSNorm (~7e-7) are at or near machine epsilon — unaffected by precision. All error originates from matmul stages; HIGH reduces them ~300×.

**L22, single_T128, BF16:**

| Stage | Input source | max_abs | mean_abs |
|-------|-------------|---------|----------|
| Q projection | hidden_states | 3.80e-02 | <u>3.56e-03</u> |
| K projection | hidden_states | <u>5.11e-02</u> | <u>3.42e-03</u> |
| V projection | hidden_states | 4.17e-02 | <u>4.16e-03</u> |
| Q conv+SiLU | GPU q_proj | <u>7.77e-02</u> | 2.46e-04 |
| K conv+SiLU | GPU k_proj | <u>1.24e-01</u> | 3.16e-04 |
| V conv+SiLU | GPU v_proj | 3.76e-02 | 7.06e-04 |
| Gate (fused_kda_gate) | hidden_states | **1.58e+00** | **8.50e-03** |
| Beta (sigmoid) | hidden_states | 2.96e-03 | 4.32e-04 |
| KDA output (fused_rec) | GPU post-conv + g + beta | 7.92e-04 | 4.12e-06 |
| Recurrent state (fused) | GPU post-conv + g + beta | 6.01e-03 | 3.49e-05 |
| Output gate (g_out) | hidden_states | <u>4.49e-02</u> | <u>2.97e-03</u> |
| Output norm | GPU o_kda + GPU g_out | 1.72e-02 | 1.97e-04 |
| Final output (o_proj) | GPU o_norm | 3.91e-02 | 3.30e-03 |

BF16 conv error (~8e-2) is larger because GPU dump intermediates are fp32 while JAX conv runs in bf16 — the error is bf16 truncation, not algorithmic. BF16 kernel error (~8e-4) is also larger due to bf16 accumulation in the recurrence.

### Per-Stage Accumulated Error (Production Forward Pass)

Production `KimiDeltaAttention.__call__` with `intermediates` capture — each stage feeds from the previous stage's JAX output. Script: `--mode accumulated`.

**L22, single_T128, FP32 — DEFAULT vs HIGH (`jax.default_matmul_precision`)**

| Stage | DEFAULT max_abs | DEFAULT mean_abs | HIGH max_abs | HIGH mean_abs |
|-------|----------------|-----------------|-------------|--------------|
| Q projection | <u>2.73e-02</u> | <u>2.60e-03</u> | 8.44e-05 | 7.73e-06 |
| K projection | <u>2.61e-02</u> | <u>2.49e-03</u> | 7.78e-05 | 7.41e-06 |
| V projection | <u>2.23e-02</u> | <u>3.03e-03</u> | 6.72e-05 | 9.00e-06 |
| Q conv+SiLU | 3.49e-02 | 1.70e-04 | 8.25e-05 | 5.05e-07 |
| K conv+SiLU | 9.25e-02 | 2.13e-04 | 2.68e-04 | 6.34e-07 |
| V conv+SiLU | 1.17e-02 | 4.69e-04 | 3.48e-05 | 1.39e-06 |
| Gate (fused_kda_gate) | **1.65e+00** | **5.61e-03** | 2.90e-03 | 1.74e-05 |
| Beta (sigmoid) | 2.84e-03 | 3.64e-04 | 7.09e-06 | 1.10e-06 |
| KDA output (fused_rec) | 2.93e-04 | 3.94e-06 | 8.85e-07 | 1.18e-08 |
| Recurrent state (fused) | 5.65e-03 | 4.23e-05 | 1.65e-05 | 1.19e-07 |
| Output gate (g_out) | <u>3.46e-02</u> | <u>2.42e-03</u> | 9.97e-05 | 7.11e-06 |
| Output norm | 1.89e-02 | 3.50e-04 | 4.15e-03 | 7.38e-05 |
| **Final output (E2E)** | **1.73e-02** | **1.86e-03** | **5.82e-03** | **3.54e-04** |

**L22, single_T128, BF16:**

| Stage | max_abs | mean_abs |
|-------|---------|----------|
| Q projection | 3.80e-02 | <u>3.56e-03</u> |
| K projection | <u>5.11e-02</u> | <u>3.42e-03</u> |
| V projection | 4.17e-02 | <u>4.16e-03</u> |
| Q conv+SiLU | 7.77e-02 | 3.16e-04 |
| K conv+SiLU | 1.85e-01 | 4.00e-04 |
| V conv+SiLU | 3.82e-02 | 8.91e-04 |
| Gate (fused_kda_gate) | **1.58e+00** | **8.50e-03** |
| Beta (sigmoid) | 2.96e-03 | 4.32e-04 |
| KDA output (fused_rec) | 7.52e-04 | 7.52e-06 |
| Recurrent state (fused) | 1.25e-02 | 7.32e-05 |
| Output gate (g_out) | <u>4.49e-02</u> | <u>2.97e-03</u> |
| Output norm | 4.87e-02 | 6.45e-04 |
| **Final output (E2E)** | **4.69e-02** | **3.30e-03** |

### Value Distributions (L22, single_T128, FP32)

TPU vs GPU distributions for stages with large magnitudes or high error rates:

| Stage | Source | mean | var | min | max |
|-------|--------|------|-----|-----|-----|
| Q projection | TPU | -1.43e-03 | 4.08e+00 | -1.46e+01 | 1.53e+01 |
| | GPU | -1.43e-03 | 4.08e+00 | -1.46e+01 | 1.53e+01 |
| Gate (fused_kda_gate) | TPU | -3.11e+00 | 6.97e+01 | -3.46e+02 | ~0 |
| | GPU | -3.11e+00 | 6.96e+01 | -3.46e+02 | 0 |
| Output gate (g_out) | TPU | 2.89e-04 | 2.05e+00 | -1.24e+01 | 1.18e+01 |
| | GPU | 3.05e-04 | 2.05e+00 | -1.24e+01 | 1.19e+01 |

Gate values span `[-346, 0]` with var=69.6 — the `exp(A_log) ≈ 62.7×` multiplier pushes the min to -346, amplifying upstream matmul error ~60× in the worst case.

### Error Pipeline Summary

| Path | Stage | Pipeline error | HIGH error | Source |
|------|-------|---------------|-----------|--------|
| hidden → q/k/v | projection matmul | <u>~3e-2</u> | ~8e-5 | `Precision.DEFAULT` bf16 truncation |
| q/k/v → heads | conv+SiLU (K=4) | ~3e-2 (cum.) | ~8e-5 | propagated from projection |
| hidden → raw_gate | gate projection (2 matmuls) | <u>~3e-2</u> | ~8e-5 | `Precision.DEFAULT` bf16 truncation |
| raw_gate → g | fused_kda_gate | **~2e+0** | ~3e-3 | exp(A_log) amplifies matmul error |
| hidden → beta | sigmoid | ~3e-3 | ~7e-6 | `Precision.DEFAULT` bf16 truncation |
| normed+g+beta → o | fused_recurrent kernel | ~3e-4 | ~9e-7 | near bit-exact (accumulated upstream error) |
| hidden → g_out | output gate projection | <u>~3e-2</u> | ~1e-4 | `Precision.DEFAULT` bf16 truncation |
| o+g_out → o_norm | GatedRMSNorm | ~2e-2 | ~4e-3 | gate magnitude × kernel error |
| o_norm → output | o_proj matmul | ~2e-2 | ~6e-3 | matmul + upstream accumulation |

### Key Findings

1. **All error originates from matmul stages.** Input projections (`[T, 2304] @ [2304, 4096]`, ~3e-2) and output projection (`[T, 4096] @ [4096, 2304]`, ~2e-2) dominate. Conv propagates upstream error but adds negligible new error. The fused recurrent kernel adds ~3e-4.

2. **Gate max_abs (1.65e+00) is a scaling artifact.** Gate computes `-exp(A_log) * softplus(raw_gate + dt_bias)`. L22's A_log reaches 4.14, giving exp(A_log) = 62.7×. This amplifies the ~2e-2 matmul error: 62.7 × 2.6e-2 ≈ 1.63. Relative error is only ~0.2%.

3. **`Precision.HIGH` reduces E2E error ~3× (1.73e-02 → 5.82e-03).** Under HIGH, matmul stages no longer dominate — the residual E2E error is driven by accumulated upstream error through gate and norm. The fused recurrent kernel itself drops to ~9e-7 under HIGH (near bit-exact). This confirms the JAX implementation is logically correct; all error is from TPU MXU bf16 truncation.

4. **BF16 production is unaffected by `Precision.HIGH`** — bf16 inputs are already at native MXU precision, so the `precision` parameter has no effect. BF16 results represent the production precision floor.

### Minimal Reproduction

Single matmul `hidden_states [128, 2304] @ q_proj_w [2304, 4096]` on L22, GPU dump vs TPU (`jax.lax.dot`):

| Precision | FP32 max_abs | FP32 mean_abs | BF16 max_abs | BF16 mean_abs |
|---|---|---|---|---|
| DEFAULT (1-pass) | 2.73e-02 | 2.60e-03 | 3.80e-02 | 3.56e-03 |
| HIGH (3-pass) | 8.44e-05 | 7.73e-06 | 3.80e-02 | 3.56e-03 |
| HIGHEST (6-pass) | 8.58e-06 | 6.64e-07 | 3.80e-02 | 3.56e-03 |

FP32 inputs: `HIGH` improves ~300×, `HIGHEST` ~3000× (near fp32 machine epsilon), confirming pipeline correctness. BF16 inputs: no change — `precision` only affects fp32→bf16 truncation in the multiplier; bf16 inputs are already at native MXU precision, so these results represent the production precision floor.

Script: `test/layers/test_kda_precision_analysis.py --mode matmul-only`.

---

## Per-Layer Details

### Full-Model End-to-End (All 20 KDA Layers)

Input: `"the capital of France is"` (5 tokens, `fused_recurrent` mode since T=5 ≤ 64). GPU reference: H100, bf16 model weights, intermediates captured as fp32. Weights loaded from isolated dumps (`/models/yuhao/kimi-linear/kda_module/L{N}/weights.npz`).

| Layer | FP32 max_abs | FP32 mean_abs | BF16 max_abs | BF16 mean_abs |
|-------|-------------|---------------|-------------|---------------|
| L0 | 1.52e-03 | 4.99e-05 | 7.32e-04 | 4.50e-05 |
| L1 | 1.74e-03 | 3.70e-05 | 1.95e-03 | 3.98e-05 |
| L2 | 4.42e-03 | 9.81e-05 | 7.81e-03 | 1.08e-04 |
| L4 | 9.68e-03 | 1.57e-04 | 1.56e-02 | 1.39e-04 |
| L5 | 1.71e-02 | 1.92e-04 | 1.56e-02 | 2.16e-04 |
| L6 | 6.42e-03 | 2.68e-04 | 7.81e-03 | 2.79e-04 |
| L8 | 8.84e-03 | 2.69e-04 | 1.56e-02 | 3.07e-04 |
| L9 | 1.16e-02 | 2.76e-04 | 3.12e-02 | 2.95e-04 |
| L10 | <u>4.82e-02</u> | 3.36e-04 | 6.25e-02 | 3.93e-04 |
| L12 | 4.01e-03 | 3.51e-04 | 3.12e-02 | 3.93e-04 |
| L13 | 2.43e-02 | 3.30e-04 | 4.69e-02 | 3.90e-04 |
| L14 | 1.95e-02 | 4.94e-04 | 6.25e-02 | 6.35e-04 |
| L16 | 4.06e-02 | 5.13e-04 | 6.25e-02 | 6.19e-04 |
| L17 | 3.47e-02 | 7.21e-04 | 6.25e-02 | 8.35e-04 |
| L18 | 1.46e-02 | 6.83e-04 | 3.12e-02 | 9.58e-04 |
| L20 | 3.43e-02 | 9.24e-04 | 6.25e-02 | 8.57e-04 |
| L21 | 4.40e-02 | <u>1.08e-03</u> | <u>6.25e-02</u> | <u>1.04e-03</u> |
| L22 | **1.06e-01** | <u>1.12e-03</u> | **1.25e-01** | <u>1.20e-03</u> |
| L24 | <u>8.03e-02</u> | <u>1.45e-03</u> | <u>6.25e-02</u> | <u>1.28e-03</u> |
| L25 | <u>6.82e-02</u> | **1.89e-03** | <u>6.25e-02</u> | **1.99e-03** |

Error grows monotonically with depth: mean_abs rises ~40× from L0 (5e-5) to L25 (2e-3). This is expected — GPU bf16 matmul precision differs from TPU MXU, and deeper layers process larger-magnitude activations. BF16 max_abs clusters at 6.25e-02 = 2⁻⁴ for mid-to-late layers (bf16 quantization effect).

---

## Notes

- **T=1 skip**: GPU chunk kernel outputs all zeros for T < chunk_size (64). TPU naive kernel correctly produces non-zero output. Test verifies no NaN + non-zero, then skips comparison.
- **GPU chunk vs fused_recurrent baseline**: Even on GPU the two kernels differ — attention output max_abs_diff = 1.21e-04, recurrent state = 6.33e-04. This sets a floor for cross-kernel comparison.
- **Test script**: `test/layers/test_kda_module.py` — black-box module-level test at `KimiDeltaAttention.__call__` boundary. Run with `KDA_DUMP_LAYER=L0 python -m pytest ... -v -s` to see per-case metrics.
- **Precision analysis**: `test/layers/test_kda_precision_analysis.py` — uses production forward pass with `intermediates` capture (no manual reimplementation). Modes: `--mode accumulated` (accumulated error), `--mode isolated` (per-stage isolated error), `--mode accumulated --precision high` (matmul precision override via `jax.default_matmul_precision`), `--mode matmul-only` (single matmul at 3 precision levels).

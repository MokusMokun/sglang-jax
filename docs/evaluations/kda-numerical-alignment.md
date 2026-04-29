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

---

## Results Summary (Full-Model, All 20 KDA Layers)

Input: `"the capital of France is"` (5 tokens, `fused_recurrent` mode since T=5 ≤ 64). GPU reference: H100, bf16 model weights, intermediates captured as fp32. Weights loaded from isolated dumps (`/models/yuhao/kimi-linear/kda_module/L{N}/weights.npz`). Matmul uses `Precision.DEFAULT` (single-pass bf16 matmul on TPU MXU).

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

## Error Source Analysis

Per-stage analysis on **L22, full-model dump** (T=5, real activations from `"the capital of France is"`). **Isolated**: each stage independently receives GPU dump as input. **Accumulated**: production forward pass. Script: `test/layers/test_kda_precision_analysis.py --source full-model`.

### Per-Stage Isolated Error (Full-Model, L22)

Each stage receives the **GPU dump intermediate** as input, isolating that stage's own JAX-vs-GPU error. Script: `--mode isolated --source full-model`.

**L22, full_model, FP32 — DEFAULT vs HIGH**

| Stage | Input source | DEFAULT max_abs | DEFAULT mean_abs | HIGH max_abs | HIGH mean_abs |
|-------|-------------|----------------|-----------------|-------------|--------------|
| Q projection | hidden_states | <u>5.41e-02</u> | <u>1.25e-03</u> | 5.41e-02 | 1.25e-03 |
| K projection | hidden_states | <u>5.15e-02</u> | <u>1.38e-03</u> | 5.15e-02 | 1.38e-03 |
| V projection | hidden_states | 3.03e-02 | <u>1.64e-03</u> | 3.03e-02 | 1.64e-03 |
| Q conv+SiLU | GPU q_proj | 2.47e-02 | 9.66e-05 | 2.47e-02 | 9.66e-05 |
| K conv+SiLU | GPU k_proj | 3.06e-02 | 9.73e-05 | 3.06e-02 | 9.73e-05 |
| V conv+SiLU | GPU v_proj | 1.20e-02 | 1.40e-04 | 1.20e-02 | 1.40e-04 |
| Gate (fused_kda_gate) | hidden_states | **1.22e-01** | **1.44e-03** | **2.38e-01** | **2.36e-03** |
| Beta (sigmoid) | hidden_states | 6.79e-04 | 1.73e-04 | 6.79e-04 | 1.73e-04 |
| KDA output (fused_rec) | GPU post-conv + g + beta | 1.06e-04 | 5.64e-07 | 1.06e-04 | 5.64e-07 |
| Recurrent state (fused) | GPU post-conv + g + beta | 3.86e-05 | 6.57e-09 | 3.86e-05 | 6.57e-09 |
| Output gate (g_out) | hidden_states | <u>3.00e-02</u> | <u>1.84e-03</u> | 2.86e-02 | 2.69e-03 |
| Output norm | GPU o_kda + GPU g_out | 7.21e-03 | 5.14e-05 | 7.21e-03 | 5.14e-05 |
| Final output (o_proj) | GPU o_norm | 5.13e-02 | 3.82e-04 | 5.13e-02 | 3.82e-04 |

`Precision.HIGH` behavior by stage category:

- **Single matmul from hidden_states** (projections, beta, final output): DEFAULT = HIGH — input is bf16-precision, truncation is lossless.
- **GPU dump → production function** (conv, kernel, norm): DEFAULT = HIGH — no matmul precision dependence.
- **Two chained matmuls** (gate, output gate): the second matmul's input (`f_a` / `g_a`) is a TPU fp32 intermediate. HIGH computes it more precisely (closer to fp32 truth), but the GPU reference was bf16 matmul. Gate diverges 2× (1.22e-01 → 2.38e-01) because `exp(A_log) ≈ 62.7×` amplifies this; output gate barely changes (3.00e-02 → 2.86e-02) because it has no exponential amplification.

### Per-Stage Accumulated Error (Full-Model, L22)

Production `KimiDeltaAttention.__call__` with `intermediates` capture. Script: `--mode accumulated --source full-model`.

**L22, full_model, FP32 — DEFAULT vs HIGH**

| Stage | DEFAULT max_abs | DEFAULT mean_abs | HIGH max_abs | HIGH mean_abs |
|-------|----------------|-----------------|-------------|--------------|
| Q projection | <u>5.41e-02</u> | <u>1.25e-03</u> | 5.41e-02 | 1.25e-03 |
| K projection | <u>5.15e-02</u> | <u>1.38e-03</u> | 5.15e-02 | 1.38e-03 |
| V projection | 3.03e-02 | <u>1.64e-03</u> | 3.03e-02 | 1.64e-03 |
| Q conv+SiLU | 5.79e-02 | 1.35e-04 | 5.79e-02 | 1.35e-04 |
| K conv+SiLU | 5.17e-02 | 1.32e-04 | 5.17e-02 | 1.32e-04 |
| V conv+SiLU | 1.62e-02 | 1.72e-04 | 1.62e-02 | 1.72e-04 |
| Gate (fused_kda_gate) | **1.22e-01** | **1.44e-03** | 2.38e-01 | 2.36e-03 |
| Beta (sigmoid) | 6.79e-04 | 1.73e-04 | 6.79e-04 | 1.73e-04 |
| KDA output (fused_rec) | 2.42e-04 | 1.35e-06 | 2.42e-04 | 1.36e-06 |
| Recurrent state (fused) | 1.14e-02 | 1.89e-05 | 1.11e-02 | 1.92e-05 |
| Output gate (g_out) | <u>3.00e-02</u> | <u>1.84e-03</u> | 2.86e-02 | 2.69e-03 |
| Output norm | 1.33e-02 | 1.34e-04 | 1.33e-02 | 1.37e-04 |
| **Final output (E2E)** | **1.06e-01** | **1.12e-03** | **6.31e-02** | **1.04e-03** |

Gate under HIGH (2.38e-01) is **larger** than DEFAULT (1.22e-01) — HIGH changes the TPU gate computation but moves it further from the GPU bf16 result. This confirms that the dominant error source for full-model inputs is **cross-device bf16 matmul divergence**, not TPU precision truncation.

### Value Distributions (Full-Model, L22, FP32)

| Stage | Source | mean | var | min | max |
|-------|--------|------|-----|-----|-----|
| Q projection | TPU | -3.55e-03 | 1.66e+00 | -2.06e+01 | 1.04e+01 |
| | GPU | -3.55e-03 | 1.66e+00 | -2.06e+01 | 1.04e+01 |
| Gate (fused_kda_gate) | TPU | -1.23e+00 | 7.26e+00 | -7.97e+01 | ~0 |
| | GPU | -1.23e+00 | 7.26e+00 | -7.96e+01 | ~0 |
| Output gate (g_out) | TPU | -6.08e-01 | 2.84e+00 | -1.01e+01 | 1.14e+01 |
| | GPU | -6.09e-01 | 2.84e+00 | -1.01e+01 | 1.14e+01 |

Gate values span `[-80, 0]` (vs `[-346, 0]` in isolated dump with T=128) — shorter sequence (T=5) produces smaller gate magnitudes.

### Error Pipeline Summary

| Path | Stage | Accumulated error | Source |
|------|-------|------------------|--------|
| hidden → q/k/v | projection matmul | <u>~5e-2</u> | cross-device bf16 matmul divergence |
| q/k/v → heads | conv+SiLU (K=4) | ~5e-2 (cum.) | propagated from projection |
| hidden → raw_gate | gate projection (2 matmuls) | ~5e-2 | cross-device bf16 matmul divergence |
| raw_gate → g | fused_kda_gate | **~1e-1** | exp(A_log) amplifies matmul error |
| hidden → beta | sigmoid | ~7e-4 | small (sigmoid compresses range) |
| normed+g+beta → o | fused_recurrent kernel | ~2e-4 | near bit-exact |
| hidden → g_out | output gate projection | <u>~3e-2</u> | cross-device bf16 matmul divergence |
| o+g_out → o_norm | GatedRMSNorm | ~1e-2 | gate magnitude × kernel error |
| o_norm → output | o_proj matmul | ~1e-1 | matmul + upstream accumulation |

### Key Findings

1. **Error is dominated by cross-device bf16 matmul divergence.** Full-model `hidden_states` are produced by GPU bf16 matmul — the error at projection stages (~5e-2) is from H100 vs TPU MXU bf16 arithmetic differences. `Precision.HIGH` has zero effect on projections (DEFAULT = HIGH), confirming this is not fp32→bf16 truncation.

2. **Gate max_abs (1.22e-01) is smaller than isolated dump (1.65e+00)** because full-model T=5 produces smaller gate magnitudes (range `[-80, 0]` vs `[-346, 0]` for T=128). The exp(A_log) amplification (~62.7×) applies the same way, but the base error is smaller.

3. **`Precision.HIGH` has limited benefit for full-model inputs** — E2E error reduces only ~1.7× (1.06e-01 → 6.31e-02), vs ~3× for isolated dumps. Gate under HIGH actually increases (1.22e-01 → 2.38e-01) because HIGH moves TPU computation further from the GPU bf16 result. The fused recurrent kernel (~2e-4) and GatedRMSNorm (~1e-2) are the main residual contributors.

4. **The fused recurrent kernel is near bit-exact** — isolated error 1.06e-04 (max_abs), 5.64e-07 (mean_abs). This confirms the kernel implementation is correct; all module-level error originates from matmul stages.

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

## Isolated Layer Details (4 Layers, 12 Synthetic Cases)

Dumps: `/models/yuhao/kimi-linear/kda_module/{L0,L6,L13,L22}/`. Matmul uses `Precision.DEFAULT`.

**Tolerance tiers** (tight first, loose as fallback):

| | Tight | Loose |
|---|---|---|
| Prefill FP32 | atol=2e-3, rtol=5e-3 | atol=3e-2, rtol=2e-2 |
| Prefill BF16 | atol=3e-3, rtol=5e-3 | atol=7e-2, rtol=2e-2 |
| Decode FP32 | atol=1e-3, rtol=1e-3 | atol=1e-2, rtol=1e-2 |
| Decode BF16 | atol=2e-3, rtol=2e-3 | atol=2e-2, rtol=2e-2 |

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

## Notes

- **T=1 skip**: GPU chunk kernel outputs all zeros for T < chunk_size (64). TPU naive kernel correctly produces non-zero output. Test verifies no NaN + non-zero, then skips comparison.
- **GPU chunk vs fused_recurrent baseline**: Even on GPU the two kernels differ — attention output max_abs_diff = 1.21e-04, recurrent state = 6.33e-04. This sets a floor for cross-kernel comparison.
- **Test script**: `test/layers/test_kda_module.py` — black-box module-level test at `KimiDeltaAttention.__call__` boundary. Run with `KDA_DUMP_LAYER=L0 python -m pytest ... -v -s` to see per-case metrics.
- **Precision analysis**: `test/layers/test_kda_precision_analysis.py` — uses production forward pass with `intermediates` capture (no manual reimplementation). Modes: `--mode accumulated` (accumulated error), `--mode isolated` (per-stage isolated error), `--mode accumulated --precision high` (matmul precision override via `jax.default_matmul_precision`), `--mode matmul-only` (single matmul at 3 precision levels).

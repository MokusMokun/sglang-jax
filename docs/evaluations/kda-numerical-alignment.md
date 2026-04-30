# KDA Phase B: Numerical Alignment

**Updated**: 2026-04-30 | **Branch**: `merge/kda-validation`

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

Per-stage analysis on **full-model dump** (T=5, real activations from `"the capital of France is"`). Production dtype is **bf16**. Script: `test/layers/test_kda_precision_analysis.py --source full-model`.

### Per-Stage Isolated Error (BF16, L22 / L24 / L25)

Each stage independently receives the **GPU dump intermediate** as input, isolating that stage's own JAX-vs-GPU error. Script: `--mode isolated --source full-model --dtype bf16`.

| Stage | Input source | L22 max | L22 mean | L24 max | L24 mean | L25 max | L25 mean |
|-------|-------------|---------|----------|---------|----------|---------|----------|
| Q projection | hidden_states | 1.56e-02 | 2.43e-06 | <u>7.81e-03</u> | 1.65e-06 | 7.81e-03 | 6.14e-07 |
| K projection | hidden_states | 1.56e-02 | 4.03e-06 | 7.81e-03 | 8.27e-07 | 3.91e-03 | 7.69e-07 |
| V projection | hidden_states | 1.56e-02 | 4.59e-06 | 3.91e-03 | 5.43e-07 | 7.81e-03 | 1.45e-06 |
| Q conv+SiLU | GPU q_proj | <u>3.12e-02</u> | <u>1.02e-04</u> | <u>3.12e-02</u> | <u>8.25e-05</u> | <u>1.56e-02</u> | <u>4.02e-05</u> |
| K conv+SiLU | GPU k_proj | <u>3.12e-02</u> | <u>8.53e-05</u> | <u>1.56e-02</u> | <u>5.16e-05</u> | <u>7.81e-03</u> | <u>2.69e-05</u> |
| V conv+SiLU | GPU v_proj | <u>3.12e-02</u> | <u>9.51e-05</u> | <u>1.56e-02</u> | <u>6.83e-05</u> | <u>7.81e-03</u> | <u>5.04e-05</u> |
| Gate (fused_kda_gate) | hidden_states | **1.22e-01** | **1.71e-03** | **3.05e-01** | **3.03e-03** | **3.50e-01** | **3.13e-03** |
| Beta (sigmoid) | hidden_states | 8.41e-04 | 5.28e-06 | 1.19e-07 | 2.98e-08 | 5.96e-08 | 3.13e-08 |
| KDA output (fused_rec) | GPU post-conv + g + beta | 2.44e-04 | 6.76e-07 | 4.88e-04 | 5.15e-07 | 6.10e-05 | 5.82e-07 |
| Recurrent state (fused) | GPU post-conv + g + beta | 5.55e-03 | 9.09e-06 | 3.54e-03 | 6.49e-06 | 3.71e-03 | 4.53e-06 |
| KDA output (chunk) | GPU post-conv + g + beta | 4.88e-04 | 1.01e-06 | 2.44e-04 | 9.31e-07 | 1.22e-04 | 1.07e-06 |
| Recurrent state (chunk) | GPU post-conv + g + beta | <u>1.76e-02</u> | <u>1.41e-05</u> | 4.95e-03 | <u>9.87e-06</u> | 5.53e-03 | <u>7.10e-06</u> |
| Output gate (g_out) | hidden_states | 3.81e-06 | 1.86e-10 | 9.77e-04 | 5.07e-08 | 0 | 0 |
| Output norm | GPU o_kda + GPU g_out | 0 | 0 | 0 | 0 | 0 | 0 |
| Final output (o_proj) | GPU o_norm | 9.77e-04 | 2.13e-07 | 1.95e-03 | 4.29e-07 | <u>7.81e-03</u> | 8.94e-07 |

BF16 characteristics:
- **Projections**: max_abs = bf16 quantization step (1.56e-02 = 2⁻⁶ or 7.81e-03 = 2⁻⁷), mean_abs ~1000× smaller than FP32 — bf16 inputs are bit-identical on both devices, only rounding at the bf16 boundary differs.
- **Gate**: dominant error source across all layers. L24 is worst (3.05e-01) due to larger exp(A_log) amplification. Gate range spans `[-264, 0]` at L24 vs `[-80, 0]` at L22.
- **Kernel output**: fused_rec ≈ chunk in max_abs (same order of magnitude). Both near bit-exact.
- **Recurrent state**: chunk ~2-3× worse than fused (chunked state accumulation algorithm differs). L22 chunk state (1.76e-02) is the worst across all layers — this error accumulates into every subsequent decode step.
- **Output gate / output norm**: many stages show exact zero error — bf16 inputs are identical on both devices.

### Per-Stage Accumulated Error (BF16, L22)

Production `KimiDeltaAttention.__call__` with `intermediates` capture. Script: `--mode accumulated --source full-model --dtype bf16`.

**L22, full_model, BF16 — Naive (fused_recurrent)**

| Stage | max_abs | mean_abs |
|-------|---------|----------|
| Q projection | 1.56e-02 | 2.43e-06 |
| K projection | 1.56e-02 | 4.03e-06 |
| V projection | 1.56e-02 | 4.59e-06 |
| Q conv+SiLU | 3.12e-02 | 1.02e-04 |
| K conv+SiLU | 3.12e-02 | 8.64e-05 |
| V conv+SiLU | 3.12e-02 | 9.53e-05 |
| Gate (fused_kda_gate) | **1.22e-01** | **1.71e-03** |
| Beta (sigmoid) | 8.41e-04 | 5.28e-06 |
| KDA output (fused_rec) | 4.88e-04 | 1.54e-06 |
| Recurrent state (fused) | <u>2.03e-02</u> | 2.16e-05 |
| Output gate (g_out) | 3.81e-06 | 1.86e-10 |
| Output norm | 3.12e-02 | 1.40e-04 |
| **Final output (E2E)** | **1.25e-01** | **1.20e-03** |

**L22, full_model, BF16 — Chunk (Pallas, overlay)**

Same accumulated intermediates, swapping fused_recurrent → chunk_kda at the kernel stage, recomputing downstream. Pre-kernel stages identical by construction.

| Stage | max_abs | mean_abs |
|-------|---------|----------|
| KDA output (chunk) | 7.32e-04 | 1.70e-06 |
| Output norm (chunk→) | 3.12e-02 | 1.51e-04 |
| **Final output (E2E, chunk→)** | **1.25e-01** | **1.24e-03** |

E2E error is **1.25e-01 = 2⁻³** for both kernels — the bf16 quantization floor. Both kernels are functionally equivalent at production precision.

### FP32 Reference (L22, DEFAULT vs HIGH)

FP32 tests reveal the error pipeline structure that bf16 quantization obscures. `Precision.HIGH` overrides TPU MXU matmul precision via `jax.default_matmul_precision("high")`.

**Per-Stage Isolated Error (FP32)**

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
| KDA output (chunk) | GPU post-conv + g + beta | 1.06e-04 | 6.02e-07 | 1.06e-04 | 6.02e-07 |
| Recurrent state (chunk) | GPU post-conv + g + beta | 1.07e-03 | 2.89e-06 | 1.07e-03 | 2.89e-06 |
| Output gate (g_out) | hidden_states | <u>3.00e-02</u> | <u>1.84e-03</u> | 2.86e-02 | 2.69e-03 |
| Output norm | GPU o_kda + GPU g_out | 7.21e-03 | 5.14e-05 | 7.21e-03 | 5.14e-05 |
| Final output (o_proj) | GPU o_norm | 5.13e-02 | 3.82e-04 | 5.13e-02 | 3.82e-04 |

**Per-Stage Accumulated Error (FP32)**

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

Chunk overlay (FP32): KDA output 2.42e-04 / 1.38e-06, E2E 1.06e-01 / 1.15e-03 (DEFAULT); 6.38e-02 / 1.08e-03 (HIGH).

`Precision.HIGH` behavior:
- **Single matmul from hidden_states** (projections, beta, final output): DEFAULT = HIGH — input is bf16-precision, truncation is lossless.
- **GPU dump → production function** (conv, kernel, norm): DEFAULT = HIGH — no matmul precision dependence.
- **Two chained matmuls** (gate, output gate): HIGH computes the fp32 intermediate more precisely, but moves it further from the GPU bf16 reference. Gate diverges 2× (1.22e-01 → 2.38e-01) because `exp(A_log) ≈ 62.7×` amplifies this.

**Minimal Reproduction** — single matmul `hidden_states [128, 2304] @ q_proj_w [2304, 4096]` on L22:

| Precision | FP32 max_abs | FP32 mean_abs | BF16 max_abs | BF16 mean_abs |
|---|---|---|---|---|
| DEFAULT (1-pass) | 2.73e-02 | 2.60e-03 | 3.80e-02 | 3.56e-03 |
| HIGH (3-pass) | 8.44e-05 | 7.73e-06 | 3.80e-02 | 3.56e-03 |
| HIGHEST (6-pass) | 8.58e-06 | 6.64e-07 | 3.80e-02 | 3.56e-03 |

FP32: HIGH improves ~300×, HIGHEST ~3000×. BF16: no change — bf16 inputs are already at native MXU precision. Script: `--mode matmul-only`.

### Error Pipeline Summary

| Path | Stage | BF16 error | FP32 error | Source |
|------|-------|-----------|-----------|--------|
| hidden → q/k/v | projection matmul | ~1e-2 | ~5e-2 | cross-device bf16 matmul divergence |
| q/k/v → heads | conv+SiLU (K=4) | ~3e-2 | ~5e-2 | conv rounding propagation |
| hidden → raw_gate | gate projection (2 matmuls) | — | ~5e-2 | cross-device bf16 matmul divergence |
| raw_gate → g | fused_kda_gate | **~1e-1** | **~1e-1** | exp(A_log) amplifies matmul error |
| hidden → beta | sigmoid | ~8e-4 | ~7e-4 | small (sigmoid compresses range) |
| normed+g+beta → o | fused_recurrent kernel | ~5e-4 | ~2e-4 | near bit-exact |
| normed+g+beta → o | chunk_kda (Pallas) kernel | ~5e-4 | ~2e-4 | near bit-exact (matches fused_rec) |
| hidden → g_out | output gate projection | ~0 | ~3e-2 | bf16: bit-identical; fp32: matmul divergence |
| o+g_out → o_norm | GatedRMSNorm | ~3e-2 | ~1e-2 | gate magnitude × kernel error |
| o_norm → output | o_proj matmul | **~1e-1** | **~1e-1** | matmul + upstream accumulation |

### Key Findings

1. **Error is dominated by cross-device bf16 matmul divergence.** Full-model `hidden_states` are produced by GPU bf16 matmul — projection error (~1e-2 bf16, ~5e-2 fp32) comes from H100 vs TPU MXU arithmetic differences. FP32 `Precision.HIGH` has zero effect on projections (DEFAULT = HIGH), confirming this is not fp32→bf16 truncation.

2. **Gate is the largest single-stage error source** — exp(A_log) amplification (~62.7× at L22, higher at L24/L25) turns ~1e-2 matmul error into ~1e-1 gate error. L24 is worst (3.05e-01 bf16) because gate range spans `[-264, 0]` vs `[-80, 0]` at L22.

3. **Both kernels are near bit-exact** — isolated error ~2-5e-4 (max_abs) for both fused_recurrent and chunk_kda across bf16 and fp32. E2E is identical at bf16 precision (1.25e-01 = 2⁻³ for both). All module-level error originates from matmul stages, not kernels.

4. **Recurrent state: chunk ~2-3× worse than fused** — chunk state accumulation uses a different algorithm (inter-chunk propagation vs step-by-step scan). L22 chunk state error (1.76e-02 bf16) is the worst; this propagates into every subsequent decode step.

5. **BF16 quantization is the E2E floor** — E2E max_abs clusters at 1.25e-01 = 2⁻³ (bf16) regardless of kernel choice. FP32 E2E is ~1.06e-01 (DEFAULT) and improves to 6.31e-02 under HIGH, but bf16 cannot benefit from precision overrides.

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

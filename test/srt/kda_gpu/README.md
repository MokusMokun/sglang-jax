# `kda_gpu/` — KDA GPU Ground-Truth Dumps

H100 reference dumps of `KimiDeltaAttention.forward` for JAX/TPU alignment.

Uses real HF weights from `moonshotai/Kimi-Linear-48B-A3B-Instruct`
(hidden=2304, heads=32, head_dim=128).

> **Upstream bug**: `modeling_kimi.py:560` has a broken `fused_kda_gate` call.
> We subclass `KimiDeltaAttention` in `fixed_kda_module.py` to fix it.

---

## Environment

Must run on an **NVIDIA GPU** (H100 recommended) with:

| Package | Version | Why |
|---|---|---|
| Python | 3.10 | `fla-core` warns but works |
| `torch` | 2.7.1+cu128 | Driver 535 compatibility |
| `fla-core` | >= 0.4.0, < 0.5 | KDA kernel implementation |
| `transformers` | >= 4.55, < 4.57 | `modeling_kimi.py` imports |
| `triton` | 3.3.1 | fla dependency |
| `einops` | >= 0.8 | Tensor reshaping |
| `safetensors` | any | HF weight loading |

Install:

```bash
pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
pip install 'fla-core>=0.4.0,<0.5' 'transformers>=4.55,<4.57' einops safetensors
```

Also needs `modeling_kimi.py` + `configuration_kimi.py` from
[moonshotai/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
accessible on `sys.path` (parent dir, `~/kda_repro/`, or alongside this script).

---

## Quick start

Two scripts, run in order:

```bash
# Step 1: extract weights → weights.npz
python dump_weights_KDA.py --hf-dir /path/to/Kimi-Linear-48B-A3B-Instruct

# Step 2: run 12 cases → case_*.npz + sanity table
python dump_io_KDAforward.py --weights dumps/weights.npz
```

### `dump_weights_KDA.py`

```
--hf-dir PATH           HF checkpoint dir (required)
--layer-idx N [N ...]   0-based layer index(es) (default: 0)
--all-kda-layers        Dump all KDA layers from config.json
--output-dir PATH       Override output directory
```

Single layer → `weights.npz`. Multiple layers → `weights_L{N}.npz` each.

> **Layer numbering**: `config.json` uses **1-based** `kda_layers` (`[1, 2, 3, ...]`),
> but `--layer-idx` uses **0-based** (matching `model.layers.{N}` in safetensors).
> So `kda_layers=[1]` corresponds to `--layer-idx 0`.

Examples:

```bash
# Single layer (default: layer 0, the first KDA layer)
python dump_weights_KDA.py --hf-dir /path/to/checkpoint

# Specific layers
python dump_weights_KDA.py --hf-dir /path/to/checkpoint --layer-idx 0 6 13 22

# All 20 KDA layers
python dump_weights_KDA.py --hf-dir /path/to/checkpoint --all-kda-layers
```

### `dump_io_KDAforward.py`

```
--weights PATH          Path to weights.npz (required)
--dumps-dir PATH        Override output directory (default: same dir as weights.npz)
```

---

## Tolerances

From `torch.testing.assert_close` defaults:

| Comparison | rtol | atol |
|---|---|---|
| fp32 ↔ fp32 | 1.3e-6 | 1e-5 (5e-5 for long T) |
| bf16 ↔ bf16 | 1.6e-2 | 1e-5 |
| chunk ↔ fused_recurrent | 1.3e-6 | 1e-3 |

> PyTorch `nn.Linear` stores `[out, in]`; Flax `nnx.Linear` uses `[in, out]` — transpose at load.

---

## Case matrix (12 cases)

| File | T | varlen | S0 | Stress |
|---|---|---|---|---|
| `case_single_T1` | 1 | – | – | Degenerate |
| `case_single_T8` | 8 | – | – | T < chunk |
| `case_single_T64` | 64 | – | – | T = 1 chunk |
| `case_single_T65` | 65 | – | – | Chunk boundary +1 |
| `case_single_T128` | 128 | – | – | Multi-chunk |
| `case_single_T256` | 256 | – | – | Accumulation |
| `case_single_T1024` | 1024 | – | – | Long prefill |
| `case_varlen_balanced_4x32` | 128 | [0,32,64,96,128] | – | Equal segments |
| `case_varlen_unbalanced` | 128 | [0,5,22,23,64,128] | – | Unbalanced |
| `case_varlen_single_T128` | 128 | [0,128] | – | Degenerate varlen |
| `case_single_T128_initstate` | 128 | – | Y | S0 != 0 |
| `case_varlen_initstate` | 64 | [0,16,32,48,64] | Y | Per-seg S0 != 0 |

Each case dumps: fp32 chunk intermediates, fp32 fused_recurrent output, bf16 output.

---

## NPZ schema

**`weights.npz`** — weights + config metadata (`config__*`, including `config__layer_idx`) + env snapshot (`env__*`).

Weight keys: `weights__{param_name}` — same names as `module.named_parameters()`:
`q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `{q,k,v}_conv1d.weight`,
`A_log`, `dt_bias`, `f_a_proj.weight`, `f_b_proj.weight`, `b_proj.weight`,
`g_a_proj.weight`, `g_b_proj.weight`, `o_norm.weight`, `o_proj.weight`.

**`case_*.npz`** — per case (33–35 arrays; count varies because optional inputs are only present when applicable):

| Category | Keys | Notes |
|---|---|---|
| Metadata | `case_name`, `T`, `B`, `has_cu_seqlens`, `has_initial_state`, `seed`, `fused_recurrent_skipped`, `fused_recurrent_skip_reason`, `env__*` | Always present |
| Inputs | `hidden_states` `[B, T, hidden]` | Always present |
| Inputs (optional) | `cu_seqlens` `[N+1]` int32 | Only varlen cases |
| Inputs (optional) | `initial_recurrent_state` `[N, H, D, D]` | Only initstate cases |
| Intermediates (proj) | `intermediates__{q,k,v}_proj` `[B, T, proj_size]` | Always present |
| Intermediates (post-conv) | `intermediates__{q,k,v}_after_conv` `[B, T, H, D]`, `intermediates__g` `[B, T, H, D]`, `intermediates__beta` `[B, T, H]` | Always present |
| Intermediates (kernel) | `intermediates__o_kda_chunk` `[B, T, H, D]`, `intermediates__recurrent_state_chunk` `[N, H, D, D]`, `intermediates__o_kda_fused_recurrent`, `intermediates__recurrent_state_fused_recurrent` | Always present; fused_recurrent values are NaN if skipped. N=1 for single seq, N=num_segs for varlen |
| Intermediates (output) | `intermediates__g_out` `[B, T, H, D]`, `intermediates__o_norm` `[B, T, H, D]` | Always present |
| Outputs | `out_fp32` `[B, T, hidden]`, `out_bf16` `[B, T, hidden]` | Always present |

---

## Pre-generated dumps (GCS)

4 KDA layers evenly spaced across model depth, organized per layer:

```
gs://model-storage-sglang/yuhao/kimi-linear/kda_module/
├── L0/           # layer 0  (early)
├── L6/           # layer 6  (early-mid)
├── L13/          # layer 13 (mid)
└── L22/          # layer 22 (late)
    ├── weights.npz              ~151 MiB  (15 params, fp32)
    ├── case_single_T1.npz       4.2 MiB
    ├── case_single_T8.npz
    ├── case_single_T64.npz
    ├── case_single_T65.npz
    ├── case_single_T128.npz
    ├── case_single_T256.npz
    ├── case_single_T1024.npz    ~210 MiB
    ├── case_varlen_balanced_4x32.npz
    ├── case_varlen_unbalanced.npz
    ├── case_varlen_single_T128.npz
    ├── case_single_T128_initstate.npz
    └── case_varlen_initstate.npz
```

All 4 layers: **12/12 cases passed** (chunk vs fused_recurrent max abs diff < 1e-3).

**Access**:
- gsutil: `gsutil ls gs://model-storage-sglang/yuhao/kimi-linear/kda_module/`
- Both H100 and TPU v6e share the same GCS bucket mount at `/models/`, so dumps written on one are immediately visible on the other.

---

## Caveats

1. `modeling_kimi.py:560` is broken upstream — `fixed_kda_module.py` works around it
2. No backward / gradient dumps
3. No incremental decode / cache passing — single forward only
4. `fused_recurrent` may OOM at large T — flagged via `fused_recurrent_skipped`
5. chunk vs fused_recurrent are not bit-equal (~1e-4 to 1e-3 in fp32)

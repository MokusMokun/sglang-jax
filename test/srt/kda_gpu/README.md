# `kda_gpu/` — KDA GPU Ground-Truth Dumps

H100 reference dumps of `KimiDeltaAttention.forward` for JAX/TPU alignment.

Two config profiles:

| Profile | Config | Weights | Output dir | Use case |
|---------|--------|---------|------------|----------|
| `small` (default) | 128h / 4H / 32d | Random init | `dumps/` | Fast local debug |
| `real` | 2304h / 32H / 128d | HF safetensors | `dumps_real/` | Numerical alignment |

> **Upstream bug**: `modeling_kimi.py:560` has a broken `fused_kda_gate` call.
> We subclass `KimiDeltaAttention` in `fixed_kda_module.py` to fix it.
> See `DESIGN.md` for details.

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
| `safetensors` | any | HF weight loading (real config only) |

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
python dump_weights_KDA.py --config small
python dump_weights_KDA.py --config real --hf-dir /path/to/Kimi-Linear-48B-A3B-Instruct

# Step 2: run 12 cases → case_*.npz + sanity table
python dump_io_KDAforward.py --weights dumps/weights.npz
python dump_io_KDAforward.py --weights dumps_real/weights.npz
```

### `dump_weights_KDA.py`

```
--config {small,real}   Config profile (default: small)
--hf-dir PATH           HF checkpoint dir (required for real)
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
python dump_weights_KDA.py --config real --hf-dir /path/to/checkpoint

# Specific layers
python dump_weights_KDA.py --config real --hf-dir /path/to/checkpoint --layer-idx 0 1 2

# All 20 KDA layers
python dump_weights_KDA.py --config real --hf-dir /path/to/checkpoint --all-kda-layers
```

### `dump_io_KDAforward.py`

```
--weights PATH          Path to weights.npz (required)
--dumps-dir PATH        Override output directory (default: same dir as weights.npz)
```

---

## Load & compare (JAX side)

```python
import numpy as np

w = np.load("kda_gpu/dumps/weights.npz")
d = np.load("kda_gpu/dumps/case_single_T128.npz")

# End-to-end check
np.testing.assert_allclose(my_jax_out, d["out_fp32"], atol=1e-5, rtol=1.3e-6)

# Kernel-level check (feed GPU's post-conv tensors, skip projections)
q     = d["intermediates__q_after_conv"]    # [B, T, H, D]
k     = d["intermediates__k_after_conv"]
v     = d["intermediates__v_after_conv"]
g     = d["intermediates__g"]               # fp32, already gate'd
beta  = d["intermediates__beta"]            # fp32, already sigmoid'd
# Note: GPU used use_qk_l2norm_in_kernel=True — L2-norm is fused inside kernel.
# If your JAX kernel doesn't fuse it, normalize q/k yourself before calling.
o_jax, S = my_jax_chunk_kda(q, k, v, g, beta, ...)
np.testing.assert_allclose(o_jax, d["intermediates__o_kda_chunk"], atol=5e-5, rtol=1.3e-6)
```

**Tolerances** (from `torch.testing.assert_close` defaults):

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

**`weights.npz`** (or `weights_L{N}.npz`) — weights + config metadata (`config__*`, including `config__layer_idx`) + env snapshot (`env__*`).

Weight keys: `weights__{param_name}` — same names as `module.named_parameters()`:
`q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `{q,k,v}_conv1d.weight`,
`A_log`, `dt_bias`, `f_a_proj.weight`, `f_b_proj.weight`, `b_proj.weight`,
`g_a_proj.weight`, `g_b_proj.weight`, `o_norm.weight`, `o_proj.weight`.

**`case_*.npz`** — per case:

| Category | Keys |
|---|---|
| Metadata | `case_name`, `T`, `B`, `has_cu_seqlens`, `has_initial_state`, `seed`, `fused_recurrent_skipped` |
| Inputs | `hidden_states`, `cu_seqlens`?, `initial_recurrent_state`? |
| Intermediates | `intermediates__{q,k,v}_after_conv`, `intermediates__g`, `intermediates__beta`, `intermediates__o_kda_chunk`, `intermediates__recurrent_state_chunk`, `intermediates__o_kda_fused_recurrent`, `intermediates__recurrent_state_fused_recurrent`, `intermediates__g_out`, `intermediates__o_norm` |
| Outputs | `out_fp32`, `out_bf16` |

---

## Pre-generated dumps (GCS)

Real-config dumps (HF weights, 4 KDA layers evenly spaced) are on GCS:

```
gs://model-storage-sglang/yuhao/kimi-linear/kda_module/
├── L0/           # layer 0  (early)
├── L6/           # layer 6  (early-mid)
├── L13/          # layer 13 (mid)
└── L22/          # layer 22 (late)
    ├── weights.npz              ~151 MiB
    ├── case_single_T1.npz
    ├── case_single_T8.npz
    ├── case_single_T64.npz
    ├── case_single_T65.npz
    ├── case_single_T128.npz
    ├── case_single_T256.npz
    ├── case_single_T1024.npz    ~160 MiB
    ├── case_varlen_balanced_4x32.npz
    ├── case_varlen_unbalanced.npz
    ├── case_varlen_single_T128.npz
    ├── case_single_T128_initstate.npz
    └── case_varlen_initstate.npz
```

All 4 layers: **12/12 cases passed** (chunk vs fused_recurrent max abs diff < 1e-3).
Total: ~2.2 GiB across 4 layers.

On GCE VMs with GCS FUSE mount, access at `/models/yuhao/kimi-linear/kda_module/`.

---

## Caveats

1. `modeling_kimi.py:560` is broken upstream — `fixed_kda_module.py` works around it
2. No backward / gradient dumps
3. No incremental decode / cache passing — single forward only
4. `fused_recurrent` may OOM at large T — flagged via `fused_recurrent_skipped`
5. chunk vs fused_recurrent are not bit-equal (~1e-4 to 1e-3 in fp32)

"""
dump_full_model_kda.py — Full-model KDA intermediate dump on H100.

Loads the complete Kimi-Linear-48B model via HuggingFace transformers,
runs a single forward pass on user-specified input, and dumps per-KDA-layer
intermediates (input, projections, conv, gate, kernel output, norm, output)
to NPZ files for TPU-side alignment validation.

The KimiDeltaAttention.forward is monkey-patched at class level to:
  1. Fix the upstream bug at modeling_kimi.py:560 (fused_kda_gate kwargs)
  2. Capture all intermediates into a global dict

Usage:
    conda activate sgl_gpu_runtime
    python dump_full_model_kda.py \
        --model-dir /models/Kimi-Linear-48B-A3B-Instruct \
        --input "the capital of France is" \
        --output-dir ~/kda_dump/full_model/run_001

Prerequisites (sgl_gpu_runtime env on H100):
    pip install 'fla-core>=0.4.0,<0.5'
    # torch, transformers, einops, safetensors already installed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Workaround: modeling_kimi.py uses PEP 604 unions (int | None) which crash
# transformers' auto_docstring decorator on Python 3.10. Patch the specific
# function that fails: _process_parameter_type tries to access .annotation.__name__
# on types.UnionType which doesn't have __name__.
import sys
import transformers.utils.auto_docstring  # ensure module is loaded
_ads_mod = sys.modules["transformers.utils.auto_docstring"]
_orig_ppt = _ads_mod._process_parameter_type

def _safe_ppt(param, param_name, func):
    try:
        return _orig_ppt(param, param_name, func)
    except AttributeError:
        return (str(param.annotation), True)

_ads_mod._process_parameter_type = _safe_ppt

from einops import rearrange
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate


# ---------------------------------------------------------------------------
# Global capture storage
# ---------------------------------------------------------------------------

_CAPTURES: dict[int, dict[str, np.ndarray]] = {}


def _to_np(t: torch.Tensor) -> np.ndarray:
    """Detach, upcast to fp32, move to CPU, convert to numpy."""
    return t.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Monkey-patch forward for KimiDeltaAttention
# ---------------------------------------------------------------------------

def _capturing_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache_params=None,
    cu_seqlens: Optional[torch.Tensor] = None,
    initial_recurrent_state: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Fixed KDA forward with intermediate capture.

    Mirrors hf_kda_module.py:FixedKimiDeltaAttention.forward exactly,
    with two differences:
      - attention_mask is silently ignored (full model passes it)
      - intermediates are stored in the global _CAPTURES dict
    """
    B, T, _ = hidden_states.shape
    H, D = self.num_heads, self.head_dim

    # 1) projections
    q_proj_out = self.q_proj(hidden_states)          # [B, T, proj_size]
    k_proj_out = self.k_proj(hidden_states)
    v_proj_out = self.v_proj(hidden_states)

    # 2) short causal conv1d + silu
    q, _ = self.q_conv1d(
        x=q_proj_out, cache=None,
        output_final_state=False, cu_seqlens=cu_seqlens,
    )
    k, _ = self.k_conv1d(
        x=k_proj_out, cache=None,
        output_final_state=False, cu_seqlens=cu_seqlens,
    )
    v, _ = self.v_conv1d(
        x=v_proj_out, cache=None,
        output_final_state=False, cu_seqlens=cu_seqlens,
    )

    # 3) reshape into heads
    q4 = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
    k4 = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
    v4 = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

    # 4) KDA gate — THE FIX (upstream line 560)
    #    Upstream (buggy): fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
    #    Fixed: reshape to [B,T,H,D], pass dt_bias= by name
    g_in = self.f_b_proj(self.f_a_proj(hidden_states))                  # [B, T, H*D]
    g_in_4d = rearrange(g_in, "... (h d) -> ... h d", d=self.head_dim)  # [B, T, H, D]
    g = fused_kda_gate(g_in_4d, self.A_log, dt_bias=self.dt_bias)       # fp32 [B, T, H, D]

    # 5) beta = sigmoid(b_proj(x)) in fp32
    beta = self.b_proj(hidden_states).float().sigmoid()                  # [B, T, H]

    # 6) KDA kernel — auto-select mode (upstream line 523)
    mode = "fused_recurrent" if T <= 64 else self.mode
    if mode == "chunk":
        o, recurrent_state = chunk_kda(
            q=q4, k=k4, v=v4, g=g, beta=beta,
            initial_state=initial_recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    elif mode == "fused_recurrent":
        o, recurrent_state = fused_recurrent_kda(
            q=q4, k=k4, v=v4, g=g, beta=beta,
            initial_state=initial_recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    else:
        raise ValueError(f"unreachable: mode={mode!r}")

    # 7) output gate g_out = g_b(g_a(x)), reshape to [B, T, H, D]
    g_out = self.g_b_proj(self.g_a_proj(hidden_states))
    g_out_4d = rearrange(g_out, "... (h d) -> ... h d", d=self.head_dim)

    # 8) gated RMSNorm
    o_norm = self.o_norm(o, g_out_4d)

    # 9) merge heads + output projection
    o_flat = rearrange(o_norm, "b t h d -> b t (h d)")
    out = self.o_proj(o_flat)

    # 10) capture intermediates → CPU numpy (frees GPU memory immediately)
    layer_idx = self.layer_idx
    _CAPTURES[layer_idx] = {
        "input_hidden_states": _to_np(hidden_states),
        "intermediates__q_proj": _to_np(q_proj_out),
        "intermediates__k_proj": _to_np(k_proj_out),
        "intermediates__v_proj": _to_np(v_proj_out),
        "intermediates__q_after_conv": _to_np(q4),
        "intermediates__k_after_conv": _to_np(k4),
        "intermediates__v_after_conv": _to_np(v4),
        "intermediates__g": _to_np(g),
        "intermediates__beta": _to_np(beta),
        "intermediates__o_kda": _to_np(o),
        "intermediates__recurrent_state": _to_np(recurrent_state),
        "intermediates__g_out": _to_np(g_out_4d),
        "intermediates__o_norm": _to_np(o_norm),
        "output": _to_np(out),
        "mode_used": np.array(mode),
    }
    print(f"  [layer {layer_idx:2d}] captured, mode={mode}, T={T}")

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_kda_layer_indices(config) -> list[int]:
    """Get 0-based KDA layer indices from HF config."""
    lac = getattr(config, "linear_attn_config", None)
    if lac is None:
        raise ValueError("Model config has no linear_attn_config")
    kda_layers_1based = lac.get("kda_layers", [])
    return sorted(k - 1 for k in kda_layers_1based)


def _env_snapshot() -> dict:
    """Capture environment info for metadata."""
    import fla
    import transformers

    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "?",
        "transformers_version": transformers.__version__,
        "fla_version": getattr(fla, "__version__", "?"),
        "device": (
            str(torch.cuda.get_device_name(0))
            if torch.cuda.is_available()
            else "cpu"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dump KDA intermediates from full Kimi-Linear-48B model"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/models/Kimi-Linear-48B-A3B-Instruct",
        help="HF checkpoint directory.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="the capital of France is",
        help="Input text for the forward pass.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for dumps. Default: ~/kda_dump/full_model/",
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path.home() / "kda_dump" / "full_model"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== dump_full_model_kda ===")
    print(f"  model_dir : {args.model_dir}")
    print(f"  input     : {args.input!r}")
    print(f"  output_dir: {output_dir}")

    # --- Load tokenizer ---
    print("\n[1/4] Loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True
    )
    input_ids = tokenizer(args.input, return_tensors="pt")["input_ids"]
    print(f"  tokens: {input_ids.shape[1]} ids = {input_ids[0].tolist()}")

    # --- Load model ---
    print("\n[2/4] Loading model (bf16, device_map='auto')...")
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        # attn_implementation not specified — model uses flash_attention_2 for MLA
    )
    model.eval()
    dt_load = time.time() - t0
    print(f"  loaded in {dt_load:.1f}s")

    # --- Identify KDA layers and patch forward ---
    kda_indices = _get_kda_layer_indices(model.config)
    print(f"  KDA layers (0-based): {kda_indices}")
    print(f"  total: {len(kda_indices)} KDA / "
          f"{model.config.num_hidden_layers} layers")

    print("\n[3/4] Patching KDA forward for intermediate capture...")
    first_kda_attn = model.model.layers[kda_indices[0]].self_attn
    KDAClass = type(first_kda_attn)
    print(f"  class: {KDAClass.__name__}")

    # Verify all KDA layers share the same class
    for idx in kda_indices:
        assert type(model.model.layers[idx].self_attn) is KDAClass, (
            f"layer {idx} self_attn is {type(model.model.layers[idx].self_attn).__name__}, "
            f"expected {KDAClass.__name__}"
        )

    # Patch at class level AND instance level. accelerate's dispatch_model
    # saves the original forward as module._old_forward before our class-level
    # patch, so the hook calls _old_forward (the buggy original). We must
    # replace _old_forward on each instance with our capturing forward.
    import types

    KDAClass.forward = _capturing_forward
    for idx in kda_indices:
        attn = model.model.layers[idx].self_attn
        if hasattr(attn, "_old_forward"):
            attn._old_forward = types.MethodType(_capturing_forward, attn)
    print(f"  patched {len(kda_indices)} KDA layers.")

    # --- Forward pass ---
    T = input_ids.shape[1]
    print(f"\n[4/4] Running forward pass (T={T})...")
    input_ids_dev = input_ids.to(model.device)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids_dev, use_cache=False)
    dt_fwd = time.time() - t0
    print(f"  forward done in {dt_fwd:.1f}s")
    print(f"  captured {len(_CAPTURES)} / {len(kda_indices)} KDA layers")

    # --- Verify ---
    missing = [i for i in kda_indices if i not in _CAPTURES]
    if missing:
        print(f"  WARNING: missing layers: {missing}")

    # --- Save per-layer NPZ ---
    print(f"\nSaving dumps to {output_dir} ...")
    for layer_idx in sorted(_CAPTURES.keys()):
        out_path = output_dir / f"layer_{layer_idx:02d}.npz"
        np.savez(out_path, **_CAPTURES[layer_idx])
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  {out_path.name}  ({size_mb:.1f} MiB)")

    # --- Save metadata ---
    metadata = {
        "input_text": args.input,
        "input_ids": input_ids[0].tolist(),
        "num_tokens": T,
        "model_dir": args.model_dir,
        "kda_layer_indices": kda_indices,
        "num_kda_layers": len(kda_indices),
        "model_config": {
            "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
        },
        "env": _env_snapshot(),
        "load_time_s": round(dt_load, 2),
        "forward_time_s": round(dt_fwd, 2),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  {meta_path.name}")

    print(f"\nDone. {len(_CAPTURES)} layers dumped to {output_dir}")


if __name__ == "__main__":
    main()

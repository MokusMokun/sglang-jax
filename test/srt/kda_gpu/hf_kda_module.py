"""
hf_kda_module.py — Subclass of HuggingFace KimiDeltaAttention that

Reference: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/modeling_kimi.py

  1. Fixes the upstream bug at modeling_kimi.py:560
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
     which raises `TypeError: fused_kda_gate() got an unexpected keyword
     argument 'g_bias'` (verified on H100 + fla-core 0.4.2). The corrected
     call is:
        g = g.view(B, T, self.num_heads, self.head_dim)
        g = fused_kda_gate(g, self.A_log, dt_bias=self.dt_bias)
     This matches both fla's `fused_kda_gate` signature and the math used
     in sglang/srt/models/kimi_linear.py.

  2. Adds two debug-friendly kwargs to forward:
        - return_intermediates: bool
            If True, returns (out, intermediates_dict) instead of just out.
        - force_mode: 'chunk' | 'fused_recurrent' | None
            If None, reproduces the upstream auto-selection
            (`q_len <= 64 -> fused_recurrent`).
            If set, bypasses auto-selection so the same input can be run
            through both kernels and the outputs cross-checked.

The __init__ is inherited unchanged so the parameter tree, weight dtypes,
and the A_log = log(uniform(1, 16)) initialization all match upstream.

Used by run_kda_gpu.py to dump KDA ground-truth tensors for JAX/TPU
alignment. Does not modify modeling_kimi.py.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import torch
from einops import rearrange
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate

# ---------------------------------------------------------------------------
# Locate and import the upstream KimiDeltaAttention without invoking the
# rest of modeling_kimi.py (KimiLinearModel and below trigger
# transformers>=4.55 auto_docstring on PEP 604 unions, which fails). We
# exec only the prefix of modeling_kimi.py up to (but not including)
# `class KimiMoEGate`, which sits right after KimiDeltaAttention ends.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELING_KIMI_CANDIDATES = [
    os.path.join(_HERE, "modeling_kimi.py"),
    os.path.join(os.path.dirname(_HERE), "modeling_kimi.py"),
    os.path.expanduser("~/kda_repro/modeling_kimi.py"),
]


def _find_modeling_kimi() -> str:
    for p in _MODELING_KIMI_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "modeling_kimi.py not found in any of:\n  " +
        "\n  ".join(_MODELING_KIMI_CANDIDATES) +
        "\nMake sure the HF copy is present alongside this file or in the parent dir."
    )


def _load_upstream_kimi_delta_attention():
    """Exec only the file prefix up through KimiDeltaAttention's class body."""
    src_path = _find_modeling_kimi()
    with open(src_path) as f:
        full_src = f.read()
    sentinel = "\nclass KimiMoEGate("
    idx = full_src.index(sentinel)
    prefix_src = full_src[:idx]

    # The file starts with `from .configuration_kimi import KimiLinearConfig`,
    # which fails outside a package. Patch it to absolute import on the fly.
    # Caller is expected to make configuration_kimi.py importable.
    prefix_src = prefix_src.replace(
        "from .configuration_kimi import KimiLinearConfig",
        "from configuration_kimi import KimiLinearConfig",
    )

    mod_globals: dict = {"__name__": "kimi_partial", "__file__": src_path}
    exec(compile(prefix_src, src_path, "exec"), mod_globals)
    return mod_globals["KimiDeltaAttention"]


KimiDeltaAttention = _load_upstream_kimi_delta_attention()


# ---------------------------------------------------------------------------
# FixedKimiDeltaAttention
# ---------------------------------------------------------------------------

class FixedKimiDeltaAttention(KimiDeltaAttention):  # type: ignore[misc, valid-type]
    """Subclass of HF KimiDeltaAttention with the line-560 fix applied.

    See module docstring for the bug explanation. Inherits __init__ as-is.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params=None,
        cu_seqlens: Optional[torch.Tensor] = None,
        initial_recurrent_state: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        force_mode: Optional[str] = None,
        **kwargs,
    ):
        """Forward pass mirroring upstream KimiDeltaAttention.forward, with:
          * the line-560 gate fix
          * captured intermediates
          * optional force_mode bypassing the q_len-based auto-selection.

        Drops support for the upstream `attention_mask` un/repad path
        (we always feed packed inputs) and for `cache_params` (we don't
        do incremental decode in dump). Both are accepted for signature
        compatibility but must be None.
        """
        if attention_mask is not None:
            raise NotImplementedError(
                "FixedKimiDeltaAttention.forward does not support attention_mask; "
                "feed packed inputs with cu_seqlens instead."
            )
        if cache_params is not None:
            raise NotImplementedError(
                "FixedKimiDeltaAttention.forward does not support cache_params; "
                "incremental decode is out of scope for the dump."
            )
        if force_mode is not None and force_mode not in ("chunk", "fused_recurrent"):
            raise ValueError(
                f"force_mode must be 'chunk', 'fused_recurrent', or None; got {force_mode!r}"
            )

        B, T, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        # 1) projections
        q_proj_out = self.q_proj(hidden_states)         # [B, T, proj_size]
        k_proj_out = self.k_proj(hidden_states)
        v_proj_out = self.v_proj(hidden_states)

        # 2) short causal conv + silu, mirroring upstream lines 541-558.
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

        # 3) reshape into heads (and capture as q/k/v_after_conv)
        q4 = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k4 = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v4 = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # 4) KDA gate -- THE FIX
        #    Upstream:  g = self.f_b_proj(self.f_a_proj(hidden_states))           # [B, T, H*D]
        #               g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
        #    Fixed:     reshape to [B, T, H, D] first, pass dt_bias by name.
        g_in = self.f_b_proj(self.f_a_proj(hidden_states))                  # [B, T, H*D]
        g_in_4d = rearrange(g_in, "... (h d) -> ... h d", d=self.head_dim)  # [B, T, H, D]
        g = fused_kda_gate(g_in_4d, self.A_log, dt_bias=self.dt_bias)        # fp32 [B, T, H, D]

        # 5) beta = sigmoid(b_proj(x)) in fp32, matches upstream line 561
        beta = self.b_proj(hidden_states).float().sigmoid()                  # [B, T, H]

        # 6) recurrence -- mode selection
        if force_mode is not None:
            mode = force_mode
        else:
            # Reproduce upstream line 523 selection
            mode = "fused_recurrent" if T <= 64 else self.mode

        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q4,
                k=k4,
                v=v4,
                g=g,
                beta=beta,
                initial_state=initial_recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_kda(
                q=q4,
                k=k4,
                v=v4,
                g=g,
                beta=beta,
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

        if not return_intermediates:
            return out

        intermediates = {
            "q_proj": q_proj_out,                        # [B, T, proj_size] pre-conv
            "k_proj": k_proj_out,
            "v_proj": v_proj_out,
            "q_after_conv": q4,                          # [B, T, H, D]   post-conv+silu
            "k_after_conv": k4,
            "v_after_conv": v4,
            "g": g,                                     # [B, T, H, D]   fp32
            "beta": beta,                               # [B, T, H]      fp32
            "o_kda": o,                                 # [B, T, H, D]   pre o_norm
            "recurrent_state": recurrent_state,         # [N, H, K, V]   fp32 (from kernel)
            "g_out": g_out_4d,                          # [B, T, H, D]   pre-sigmoid
            "o_norm": o_norm,                           # [B, T, H, D]   post gated rmsnorm
            "out": out,                                 # [B, T, hidden]
            "mode_used": mode,                          # 'chunk' | 'fused_recurrent'
        }
        return out, intermediates


__all__ = ["FixedKimiDeltaAttention", "KimiDeltaAttention"]

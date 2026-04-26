"""
dump_io_KDAforward.py — Run 12 KDA forward cases and dump ground truth.

Reads weights.npz (produced by dump_weights_KDA.py) to rebuild the module,
then runs each case through chunk + fused_recurrent + bf16 paths.

Usage:
  python dump_io_KDAforward.py --weights dumps/weights.npz
  python dump_io_KDAforward.py --weights dumps_real/weights.npz
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for cand in (HERE.parent, HERE / "..", Path.home() / "kda_repro"):
    p = Path(cand).resolve()
    if (p / "configuration_kimi.py").exists():
        sys.path.insert(0, str(p))

from fixed_kda_module import FixedKimiDeltaAttention  # noqa: E402


# ---------------------------------------------------------------------------
# Load module from weights.npz
# ---------------------------------------------------------------------------

def _load_module_from_npz(
    weights_path: Path,
    dtype: torch.dtype = torch.float32,
) -> tuple[FixedKimiDeltaAttention, dict]:
    """Rebuild module from weights.npz config metadata + weights.

    Returns (module, config_dict).
    """
    from configuration_kimi import KimiLinearConfig

    w = np.load(weights_path, allow_pickle=True)

    config = {
        "hidden_size": int(w["config__hidden_size"]),
        "num_heads": int(w["config__num_heads"]),
        "head_dim": int(w["config__head_dim"]),
        "conv_size": int(w["config__conv_size"]),
        "rms_norm_eps": float(w["config__rms_norm_eps"]),
        "seed": int(w["config__seed"]),
        "profile": str(w["config__profile"]),
    }

    cfg = KimiLinearConfig(
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_heads"],
        intermediate_size=4 * config["hidden_size"],
        num_hidden_layers=1,
        rms_norm_eps=config["rms_norm_eps"],
        linear_attn_config=dict(
            kda_layers=[1],
            full_attn_layers=[],
            head_dim=config["head_dim"],
            num_heads=config["num_heads"],
            short_conv_kernel_size=config["conv_size"],
        ),
        vocab_size=1000,
    )

    torch.manual_seed(config["seed"])
    m = FixedKimiDeltaAttention(cfg, layer_idx=0).cuda()

    # Load weights from npz
    with torch.no_grad():
        for key in w.files:
            if not key.startswith("weights__"):
                continue
            param_name = key[len("weights__"):]
            parts = param_name.split(".")
            obj = m
            for part in parts[:-1]:
                obj = getattr(obj, part)
            param = getattr(obj, parts[-1])
            param.copy_(torch.from_numpy(w[key].copy()).to(param.device))

    m.eval()
    if dtype != torch.float32:
        m = m.to(dtype)
    return m, config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t2np(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if t is None:
        return None
    return t.detach().float().cpu().numpy()


def _nan_tensor(B: int, T: int, H: int, D: int, device: str) -> torch.Tensor:
    out = torch.empty(B, T, H, D, device=device, dtype=torch.float32)
    out.fill_(float("nan"))
    return out


def env_snapshot() -> dict:
    import fla
    import triton
    import transformers
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "?",
        "triton_version": triton.__version__,
        "fla_version": getattr(fla, "__version__", "?"),
        "transformers_version": transformers.__version__,
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "modeling_kimi_md5": "337ae1fc58c7010db4051e30fa23563e",
        "fix_note": (
            "modeling_kimi.py:560 fused_kda_gate(g, A_log, dt_bias=dt_bias) "
            "with g.view(B,T,H,D) reshape -- see kda_gpu/DESIGN.md"
        ),
    }


# ---------------------------------------------------------------------------
# Per-case execution
# ---------------------------------------------------------------------------

def run_case(
    case_name: str,
    T: int,
    cu_seqlens: Optional[torch.Tensor],
    use_initial_state: bool,
    weights_path: Path,
    dumps_dir: Path,
    config: dict,
    seed: int = 0,
) -> dict:
    print(f"\n===== case: {case_name} =====")
    print(f"  T={T}  cu_seqlens={None if cu_seqlens is None else cu_seqlens.tolist()}  "
          f"use_initial_state={use_initial_state}  seed={seed}")

    B = 1
    H = config["num_heads"]
    D = config["head_dim"]
    hidden_size = config["hidden_size"]

    # Deterministic inputs
    g = torch.Generator(device="cpu").manual_seed(seed + 1000)
    hidden_cpu = torch.empty(B, T, hidden_size)
    hidden_cpu.normal_(generator=g)
    hidden_fp32 = hidden_cpu.cuda()

    cu = None
    if cu_seqlens is not None:
        cu = cu_seqlens.to(dtype=torch.int32, device="cuda")

    init_state = None
    if use_initial_state:
        N = 1 if cu is None else cu.numel() - 1
        init_cpu = torch.empty(N, H, D, D, dtype=torch.float32)
        init_cpu.normal_(generator=g)
        init_cpu.mul_(0.05)
        init_state = init_cpu.cuda()

    # ---------------- fp32, chunk ----------------
    m_fp32, _ = _load_module_from_npz(weights_path, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        out_chunk, inter_chunk = m_fp32(
            hidden_fp32,
            cu_seqlens=cu,
            initial_recurrent_state=init_state,
            return_intermediates=True,
            force_mode="chunk",
        )
    torch.cuda.synchronize()
    t_chunk = time.time() - t0
    print(f"  [fp32 chunk] out shape={tuple(out_chunk.shape)}  "
          f"o_kda shape={tuple(inter_chunk['o_kda'].shape)}  "
          f"S shape={None if inter_chunk['recurrent_state'] is None else tuple(inter_chunk['recurrent_state'].shape)}  "
          f"({t_chunk*1000:.1f} ms)")

    # ---------------- fp32, fused_recurrent (may OOM) ----------------
    fused_recurrent_skipped = False
    fused_recurrent_skip_reason = ""
    try:
        with torch.no_grad():
            out_fr, inter_fr = m_fp32(
                hidden_fp32,
                cu_seqlens=cu,
                initial_recurrent_state=init_state,
                return_intermediates=True,
                force_mode="fused_recurrent",
            )
        torch.cuda.synchronize()
        o_kda_fr = inter_fr["o_kda"]
        S_fr = inter_fr["recurrent_state"]
        print(f"  [fp32 fused_recurrent] o_kda shape={tuple(o_kda_fr.shape)}")
    except Exception as exc:
        fused_recurrent_skipped = True
        fused_recurrent_skip_reason = f"{type(exc).__name__}: {exc}"
        print(f"  [fp32 fused_recurrent] SKIPPED: {fused_recurrent_skip_reason}")
        o_kda_fr = _nan_tensor(B, T, H, D, device="cuda")
        N = 1 if cu is None else cu.numel() - 1
        S_fr = torch.empty(N, H, D, D, device="cuda", dtype=torch.float32)
        S_fr.fill_(float("nan"))
        torch.cuda.empty_cache()

    # ---------------- bf16 chunk ----------------
    m_bf16, _ = _load_module_from_npz(weights_path, dtype=torch.bfloat16)
    hidden_bf16 = hidden_fp32.to(torch.bfloat16)
    with torch.no_grad():
        out_bf16 = m_bf16(
            hidden_bf16,
            cu_seqlens=cu,
            initial_recurrent_state=init_state,
            return_intermediates=False,
            force_mode="chunk",
        )
    torch.cuda.synchronize()
    print(f"  [bf16 chunk] out shape={tuple(out_bf16.shape)}  dtype={out_bf16.dtype}")
    diff = (out_bf16.float() - out_chunk.float()).abs()
    print(f"  bf16 vs fp32 chunk: max={diff.max().item():.4e}  mean={diff.mean().item():.4e}")

    # ---------------- assemble npz ----------------
    payload: dict = {
        "case_name": np.asarray(case_name),
        "T": np.asarray(T),
        "B": np.asarray(B),
        "has_cu_seqlens": np.asarray(cu is not None),
        "has_initial_state": np.asarray(init_state is not None),
        "seed": np.asarray(seed),
        "fused_recurrent_skipped": np.asarray(fused_recurrent_skipped),
        "fused_recurrent_skip_reason": np.asarray(fused_recurrent_skip_reason),
    }
    for k, v in env_snapshot().items():
        payload[f"env__{k}"] = np.asarray(v)

    payload["hidden_states"] = _t2np(hidden_fp32)
    if cu is not None:
        payload["cu_seqlens"] = cu.detach().cpu().numpy().astype(np.int32)
    if init_state is not None:
        payload["initial_recurrent_state"] = _t2np(init_state)

    for k in ("q_after_conv", "k_after_conv", "v_after_conv", "g", "beta",
              "g_out", "o_norm"):
        payload[f"intermediates__{k}"] = _t2np(inter_chunk[k])
    payload["intermediates__o_kda_chunk"] = _t2np(inter_chunk["o_kda"])
    payload["intermediates__recurrent_state_chunk"] = _t2np(inter_chunk["recurrent_state"])
    payload["intermediates__o_kda_fused_recurrent"] = _t2np(o_kda_fr)
    payload["intermediates__recurrent_state_fused_recurrent"] = _t2np(S_fr)

    payload["out_fp32"] = _t2np(out_chunk)
    payload["out_bf16"] = _t2np(out_bf16)

    out_path = dumps_dir / f"case_{case_name}.npz"
    np.savez(out_path, **payload)
    print(f"  -> dumped: {out_path.name}  ({len(payload)} arrays, "
          f"{out_path.stat().st_size / 1024:.1f} KiB)")

    if not fused_recurrent_skipped:
        d = (inter_chunk["o_kda"].float() - o_kda_fr.float()).abs()
        chunk_vs_fr_max = d.max().item()
        chunk_vs_fr_mean = d.mean().item()
    else:
        chunk_vs_fr_max = float("nan")
        chunk_vs_fr_mean = float("nan")

    del m_fp32, m_bf16
    torch.cuda.empty_cache()

    return {
        "name": case_name,
        "T": T,
        "varlen": cu is not None,
        "init_state": init_state is not None,
        "skipped_fr": fused_recurrent_skipped,
        "chunk_vs_fr_max": chunk_vs_fr_max,
        "chunk_vs_fr_mean": chunk_vs_fr_mean,
    }


# ---------------------------------------------------------------------------
# Cases (12 total per DESIGN.md)
# ---------------------------------------------------------------------------

CASES = [
    # name,                       T,    cu_seqlens (CPU long),                 init_state
    ("single_T1",                  1,    None,                                  False),
    ("single_T8",                  8,    None,                                  False),
    ("single_T64",                 64,   None,                                  False),
    ("single_T65",                 65,   None,                                  False),
    ("single_T128",                128,  None,                                  False),
    ("single_T256",                256,  None,                                  False),
    ("single_T1024",               1024, None,                                  False),
    ("varlen_balanced_4x32",       128,  torch.tensor([0, 32, 64, 96, 128]),    False),
    ("varlen_unbalanced",          128,  torch.tensor([0, 5, 22, 23, 64, 128]), False),
    ("varlen_single_T128",         128,  torch.tensor([0, 128]),                False),
    ("single_T128_initstate",      128,  None,                                  True),
    ("varlen_initstate",           64,   torch.tensor([0, 16, 32, 48, 64]),     True),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 12 KDA forward cases and dump ground truth"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to weights.npz (produced by dump_weights.py).",
    )
    parser.add_argument(
        "--dumps-dir", type=str, default=None,
        help="Output directory. Default: same directory as weights.npz.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: {weights_path} not found. Run dump_weights.py first.")
        sys.exit(1)

    if args.dumps_dir:
        dumps_dir = Path(args.dumps_dir)
    else:
        dumps_dir = weights_path.parent
    dumps_dir.mkdir(exist_ok=True, parents=True)

    assert torch.cuda.is_available(), "This script expects a CUDA device"

    # Read config from weights.npz
    w = np.load(weights_path, allow_pickle=True)
    config = {
        "hidden_size": int(w["config__hidden_size"]),
        "num_heads": int(w["config__num_heads"]),
        "head_dim": int(w["config__head_dim"]),
        "conv_size": int(w["config__conv_size"]),
        "rms_norm_eps": float(w["config__rms_norm_eps"]),
        "seed": int(w["config__seed"]),
        "profile": str(w["config__profile"]),
    }

    print(f"=== run_cases: profile={config['profile']} ===")
    print(f"  hidden_size={config['hidden_size']}  num_heads={config['num_heads']}  "
          f"head_dim={config['head_dim']}")
    print(f"  weights={weights_path}")
    print(f"  dumps_dir={dumps_dir}")
    print("=== env ===")
    for k, v in env_snapshot().items():
        print(f"  {k}: {v}")

    summaries = []
    for name, T, cu, init_st in CASES:
        try:
            s = run_case(
                name, T=T, cu_seqlens=cu, use_initial_state=init_st,
                weights_path=weights_path, dumps_dir=dumps_dir,
                config=config, seed=0,
            )
            summaries.append(s)
        except Exception:
            print(f"\n!!! case {name} CRASHED — full traceback follows; continuing\n")
            traceback.print_exc()
            torch.cuda.empty_cache()

    # ---------------- mode-selection sanity ----------------
    print("\n=== mode-selection sanity (T=64 should auto-pick fused_recurrent) ===")
    m_fp32, _ = _load_module_from_npz(weights_path, dtype=torch.float32)
    g = torch.Generator(device="cpu").manual_seed(1000)
    hidden_cpu = torch.empty(1, 64, config["hidden_size"])
    hidden_cpu.normal_(generator=g)
    hidden_cu = hidden_cpu.cuda()
    with torch.no_grad():
        out_auto, inter_auto = m_fp32(hidden_cu, return_intermediates=True, force_mode=None)
        out_fr, inter_fr = m_fp32(hidden_cu, return_intermediates=True, force_mode="fused_recurrent")
    d = (out_auto - out_fr).abs().max().item()
    print(f"  mode_used (auto) = {inter_auto['mode_used']!r} (expected 'fused_recurrent')")
    print(f"  out_auto vs out_force_fr max abs diff = {d:.2e}  -> "
          f"{'OK' if d < 1e-5 else 'MISMATCH'}")
    del m_fp32
    torch.cuda.empty_cache()

    # ---------------- sanity table ----------------
    print("\n=== sanity: o_kda_chunk vs o_kda_fused_recurrent (fp32) ===")
    print(f"{'case':<28s} {'T':>5s} {'varlen':>6s} {'S0':>3s} "
          f"{'fr_skipped':>10s} {'max_abs':>10s} {'mean_abs':>10s}")
    for s in summaries:
        print(f"{s['name']:<28s} {s['T']:>5d} {str(s['varlen']):>6s} "
              f"{('Y' if s['init_state'] else '-'):>3s} "
              f"{('SKIP' if s['skipped_fr'] else '-'):>10s} "
              f"{s['chunk_vs_fr_max']:>10.2e} {s['chunk_vs_fr_mean']:>10.2e}")

    n_pass = sum(1 for s in summaries
                 if not s['skipped_fr'] and s['chunk_vs_fr_max'] < 1e-3)
    print(f"\nAcceptance: {n_pass}/{len(summaries)} cases below 1e-3 "
          f"(DESIGN.md target: >= 10)")
    print(f"\nAll dumps in: {dumps_dir}")


if __name__ == "__main__":
    main()

"""
dump_weights_KDA.py — Extract KDA module weights into weights.npz.

Two modes:
  --config small   Random init (hidden=128, heads=4, head_dim=32)
  --config real    Real weights from HF safetensors (hidden=2304, heads=32, head_dim=128)

Output: weights.npz containing config metadata (config__*), env snapshot
(env__*), and all module parameters (weights__*).

dump_io_KDAforward.py consumes this file to run the 12-case dump.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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
# Config profiles
# ---------------------------------------------------------------------------

@dataclass
class ConfigProfile:
    name: str
    hidden_size: int
    num_heads: int
    head_dim: int
    conv_size: int
    rms_norm_eps: float
    seed: int


SMALL_CONFIG = ConfigProfile(
    name="small",
    hidden_size=128,
    num_heads=4,
    head_dim=32,
    conv_size=4,
    rms_norm_eps=1e-6,
    seed=0,
)

# Real HF config from moonshotai/Kimi-Linear-48B-A3B-Instruct config.json.
# hidden_size=2304 is the model hidden dim; projection_size =
# num_heads * head_dim = 32 * 128 = 4096.
REAL_CONFIG = ConfigProfile(
    name="real",
    hidden_size=2304,
    num_heads=32,
    head_dim=128,
    conv_size=4,
    rms_norm_eps=1e-5,
    seed=0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _build_config(profile: ConfigProfile):
    from configuration_kimi import KimiLinearConfig
    return KimiLinearConfig(
        hidden_size=profile.hidden_size,
        num_attention_heads=profile.num_heads,
        intermediate_size=4 * profile.hidden_size,
        num_hidden_layers=1,
        rms_norm_eps=profile.rms_norm_eps,
        linear_attn_config=dict(
            kda_layers=[1],
            full_attn_layers=[],
            head_dim=profile.head_dim,
            num_heads=profile.num_heads,
            short_conv_kernel_size=profile.conv_size,
        ),
        vocab_size=1000,
    )


def _load_hf_weights(
    module: FixedKimiDeltaAttention,
    hf_dir: str,
    layer_idx: int = 0,
) -> None:
    """Load real HF weights from safetensors into the module.

    Uses safe_open for lazy loading — only extracts
    ``model.layers.{layer_idx}.self_attn.*`` tensors.
    """
    from safetensors import safe_open

    index_path = Path(hf_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Expected {index_path} — provide --hf-dir pointing to the "
            f"HF checkpoint directory."
        )
    with open(index_path) as f:
        index = json.load(f)

    prefix = f"model.layers.{layer_idx}.self_attn."
    shard_to_keys: dict[str, list[str]] = {}
    for full_key, shard in index["weight_map"].items():
        if full_key.startswith(prefix):
            shard_to_keys.setdefault(shard, []).append(full_key)

    if not shard_to_keys:
        raise ValueError(
            f"No weights found with prefix '{prefix}' in {index_path}. "
            f"Is layer_idx={layer_idx} a KDA layer?"
        )

    loaded: dict[str, torch.Tensor] = {}
    for shard, keys in shard_to_keys.items():
        shard_path = Path(hf_dir) / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for full_key in keys:
                param_name = full_key[len(prefix):]
                loaded[param_name] = f.get_tensor(full_key).float()

    state_dict = module.state_dict()
    missing, unexpected = [], []
    for param_name, tensor in loaded.items():
        if param_name in state_dict:
            if state_dict[param_name].shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {param_name}: "
                    f"module={state_dict[param_name].shape}, "
                    f"HF={tensor.shape}"
                )
        else:
            unexpected.append(param_name)

    with torch.no_grad():
        for param_name, tensor in loaded.items():
            if param_name in state_dict:
                parts = param_name.split(".")
                obj = module
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                param = getattr(obj, parts[-1])
                param.copy_(tensor.to(param.device))

    for param_name in state_dict:
        if param_name not in loaded:
            missing.append(param_name)

    print(f"  [HF weights] loaded {len(loaded)} tensors from layer {layer_idx}")
    if missing:
        print(f"  WARNING: module params not in HF: {missing}")
    if unexpected:
        print(f"  WARNING: HF params not in module: {unexpected}")


def _build_module(
    profile: ConfigProfile,
    hf_dir: str | None = None,
    layer_idx: int = 0,
) -> FixedKimiDeltaAttention:
    """Build fp32 module with weights loaded."""
    cfg = _build_config(profile)
    seed = profile.seed
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)

    m = FixedKimiDeltaAttention(cfg, layer_idx=0).cuda()

    if profile.name == "real" and hf_dir is not None:
        _load_hf_weights(m, hf_dir, layer_idx=layer_idx)
    else:
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name == "A_log":
                    continue
                if name == "dt_bias":
                    p.zero_()
                    continue
                if p.dtype.is_floating_point:
                    tmp = torch.empty(p.shape, device="cpu")
                    tmp.normal_(mean=0.0, std=0.02, generator=g)
                    p.copy_(tmp.to(p.device))
            if hasattr(m, "o_norm") and hasattr(m.o_norm, "weight"):
                m.o_norm.weight.fill_(1.0)

    m.eval()
    return m


def _t2np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract KDA module weights into weights.npz"
    )
    parser.add_argument(
        "--config", choices=["small", "real"], default="small",
        help="Config profile (default: small).",
    )
    parser.add_argument(
        "--hf-dir", type=str, default=None,
        help="HF checkpoint directory (required for --config real).",
    )
    parser.add_argument(
        "--layer-idx", type=int, default=0,
        help="HF layer index to extract (default: 0).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path. Default: dumps/weights.npz or dumps_real/weights.npz.",
    )
    args = parser.parse_args()
    if args.config == "real" and args.hf_dir is None:
        parser.error("--hf-dir is required when --config real")
    return args


def main():
    args = parse_args()
    profile = REAL_CONFIG if args.config == "real" else SMALL_CONFIG

    if args.output:
        out_path = Path(args.output)
    else:
        dumps_dir = HERE / ("dumps_real" if args.config == "real" else "dumps")
        dumps_dir.mkdir(exist_ok=True, parents=True)
        out_path = dumps_dir / "weights.npz"

    out_path.parent.mkdir(exist_ok=True, parents=True)

    assert torch.cuda.is_available(), "This script expects a CUDA device"

    print(f"=== dump_weights: config={profile.name} ===")
    print(f"  hidden_size={profile.hidden_size}  num_heads={profile.num_heads}  "
          f"head_dim={profile.head_dim}")
    if args.config == "real":
        print(f"  hf_dir={args.hf_dir}  layer_idx={args.layer_idx}")

    m = _build_module(profile, hf_dir=args.hf_dir, layer_idx=args.layer_idx)

    # Assemble payload
    payload: dict = {
        "config__profile": np.asarray(profile.name),
        "config__hidden_size": np.asarray(profile.hidden_size),
        "config__num_heads": np.asarray(profile.num_heads),
        "config__head_dim": np.asarray(profile.head_dim),
        "config__conv_size": np.asarray(profile.conv_size),
        "config__rms_norm_eps": np.asarray(profile.rms_norm_eps),
        "config__seed": np.asarray(profile.seed),
    }
    for k, v in env_snapshot().items():
        payload[f"env__{k}"] = np.asarray(v)

    weight_keys = []
    for name, p in m.named_parameters():
        payload[f"weights__{name}"] = _t2np(p)
        weight_keys.append((name, tuple(p.shape), str(p.dtype)))

    np.savez(out_path, **payload)
    print(f"\n[weights] -> {out_path}  ({len(payload)} arrays, "
          f"{out_path.stat().st_size / 1024:.1f} KiB)")
    for n, sh, dt in weight_keys:
        print(f"  {n:<28s} {sh} {dt}")


if __name__ == "__main__":
    main()

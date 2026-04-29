"""
dump_weights_KDA.py — Extract KDA module weights from HF safetensors.

Loads real weights from moonshotai/Kimi-Linear-48B-A3B-Instruct checkpoint.

Supports single or multiple layers:
  --layer-idx 0              Single layer → weights.npz
  --layer-idx 0 2 4          Multiple layers → weights_L0.npz, weights_L2.npz, ...
  --all-kda-layers           All KDA layers from config.json → weights_L{N}.npz each

Note: config.json's kda_layers uses 1-based numbering, but --layer-idx
uses 0-based (matching model.layers.{N} in safetensors). The mapping is:
  kda_layers=[1,2,3,...] → layer_idx=0,1,2,...

dump_io_KDAforward.py consumes weights.npz files to run the 12-case dump.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for cand in (HERE.parent, HERE / "..", Path.home() / "kda_repro"):
    p = Path(cand).resolve()
    if (p / "configuration_kimi.py").exists():
        sys.path.insert(0, str(p))

from hf_kda_module import FixedKimiDeltaAttention  # noqa: E402


# ---------------------------------------------------------------------------
# Config (real HF values from moonshotai/Kimi-Linear-48B-A3B-Instruct)
# ---------------------------------------------------------------------------

# hidden_size=2304 is the model hidden dim; projection_size =
# num_heads * head_dim = 32 * 128 = 4096.
HIDDEN_SIZE = 2304
NUM_HEADS = 32
HEAD_DIM = 128
CONV_SIZE = 4
RMS_NORM_EPS = 1e-5
SEED = 0


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
            "with g.view(B,T,H,D) reshape"
        ),
    }


def _build_config():
    from configuration_kimi import KimiLinearConfig
    return KimiLinearConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        intermediate_size=4 * HIDDEN_SIZE,
        num_hidden_layers=1,
        rms_norm_eps=RMS_NORM_EPS,
        linear_attn_config=dict(
            kda_layers=[1],
            full_attn_layers=[],
            head_dim=HEAD_DIM,
            num_heads=NUM_HEADS,
            short_conv_kernel_size=CONV_SIZE,
        ),
        vocab_size=1000,
    )


def _get_kda_layer_indices(hf_dir: str) -> list[int]:
    """Read config.json and return 0-based layer indices for all KDA layers.

    config.json's kda_layers uses 1-based numbering, so we subtract 1.
    """
    config_path = Path(hf_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    kda_layers_1based = cfg.get("linear_attn_config", {}).get("kda_layers", [])
    return sorted(layer_1b - 1 for layer_1b in kda_layers_1based)


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


def _build_module(hf_dir: str, layer_idx: int = 0) -> FixedKimiDeltaAttention:
    """Build fp32 module with HF weights loaded."""
    cfg = _build_config()
    torch.manual_seed(SEED)
    m = FixedKimiDeltaAttention(cfg, layer_idx=0).cuda()
    _load_hf_weights(m, hf_dir, layer_idx=layer_idx)
    m.eval()
    return m


def _t2np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _dump_one_layer(
    layer_idx: int,
    out_path: Path,
    hf_dir: str,
) -> None:
    """Dump weights for a single layer to out_path."""
    print(f"\n--- layer {layer_idx} ---")
    m = _build_module(hf_dir, layer_idx=layer_idx)

    payload: dict = {
        "config__hidden_size": np.asarray(HIDDEN_SIZE),
        "config__num_heads": np.asarray(NUM_HEADS),
        "config__head_dim": np.asarray(HEAD_DIM),
        "config__conv_size": np.asarray(CONV_SIZE),
        "config__rms_norm_eps": np.asarray(RMS_NORM_EPS),
        "config__seed": np.asarray(SEED),
        "config__layer_idx": np.asarray(layer_idx),
    }
    for k, v in env_snapshot().items():
        payload[f"env__{k}"] = np.asarray(v)

    weight_keys = []
    for name, p in m.named_parameters():
        payload[f"weights__{name}"] = _t2np(p)
        weight_keys.append((name, tuple(p.shape), str(p.dtype)))

    np.savez(out_path, **payload)
    print(f"[weights] -> {out_path}  ({len(payload)} arrays, "
          f"{out_path.stat().st_size / 1024:.1f} KiB)")
    for n, sh, dt in weight_keys:
        print(f"  {n:<28s} {sh} {dt}")

    del m
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract KDA module weights from HF safetensors"
    )
    parser.add_argument(
        "--hf-dir", type=str, required=True,
        help="HF checkpoint directory containing model.safetensors.index.json.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--layer-idx", type=int, nargs="+", default=None,
        help="0-based layer index(es) to extract. Note: config.json uses "
             "1-based kda_layers, so kda_layers=[1,2,3] → --layer-idx 0 1 2. "
             "Default: 0.",
    )
    group.add_argument(
        "--all-kda-layers", action="store_true",
        help="Dump all KDA layers (reads kda_layers from config.json).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Default: dumps/",
    )
    args = parser.parse_args()
    if args.layer_idx is None and not args.all_kda_layers:
        args.layer_idx = [0]
    return args


def main():
    args = parse_args()

    dumps_dir = Path(args.output_dir) if args.output_dir else HERE / "dumps"
    dumps_dir.mkdir(exist_ok=True, parents=True)

    assert torch.cuda.is_available(), "This script expects a CUDA device"

    # Resolve layer indices
    if args.all_kda_layers:
        layer_indices = _get_kda_layer_indices(args.hf_dir)
        print(f"=== dump_weights_KDA: all KDA layers ({len(layer_indices)}) ===")
    else:
        layer_indices = args.layer_idx
        print(f"=== dump_weights_KDA: layer(s)={layer_indices} ===")

    print(f"  hidden_size={HIDDEN_SIZE}  num_heads={NUM_HEADS}  head_dim={HEAD_DIM}")
    print(f"  hf_dir={args.hf_dir}")
    print(f"  output_dir={dumps_dir}")

    # Dump each layer
    single = len(layer_indices) == 1
    for layer_idx in layer_indices:
        if single:
            out_path = dumps_dir / "weights.npz"
        else:
            out_path = dumps_dir / f"weights_L{layer_idx}.npz"
        _dump_one_layer(layer_idx, out_path, hf_dir=args.hf_dir)

    print(f"\nDone: {len(layer_indices)} layer(s) dumped to {dumps_dir}")


if __name__ == "__main__":
    main()

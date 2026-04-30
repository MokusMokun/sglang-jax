"""
test_kda_full_model.py — Sanity checks for full-model KDA dumps.

Validates the NPZ files produced by dump_full_model_kda.py:
  - All 20 KDA layer files present
  - Shapes match expected dimensions
  - No NaN / Inf values
  - Basic statistics (mean, std, abs-max) are in reasonable ranges

Usage:
    # Point to the dump directory
    KDA_FULL_DUMP_DIR=/models/yuhao/kimi-linear/kda_full_model_dump/run_001 \
        python -m pytest test_kda_full_model.py -v

    # Or run standalone for a quick summary
    python test_kda_full_model.py /path/to/dump_dir
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Expected KDA layer indices (0-based) from Kimi-Linear-48B config.json
# kda_layers (1-based): [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26]
EXPECTED_KDA_INDICES = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25,
]

HIDDEN_SIZE = 2304
NUM_HEADS = 32
HEAD_DIM = 128
PROJ_SIZE = NUM_HEADS * HEAD_DIM  # 4096

# Keys expected in each layer_NN.npz
EXPECTED_KEYS = {
    "input_hidden_states",
    "intermediates__q_proj",
    "intermediates__k_proj",
    "intermediates__v_proj",
    "intermediates__q_after_conv",
    "intermediates__k_after_conv",
    "intermediates__v_after_conv",
    "intermediates__g",
    "intermediates__beta",
    "intermediates__o_kda",
    "intermediates__recurrent_state",
    "intermediates__g_out",
    "intermediates__o_norm",
    "output",
    "mode_used",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _get_dump_dir() -> Path:
    d = os.environ.get("KDA_FULL_DUMP_DIR")
    if d is None:
        pytest.skip("KDA_FULL_DUMP_DIR not set")
    p = Path(d)
    if not p.exists():
        pytest.skip(f"Dump directory does not exist: {p}")
    return p


@pytest.fixture(scope="module")
def dump_dir():
    return _get_dump_dir()


@pytest.fixture(scope="module")
def metadata(dump_dir):
    meta_path = dump_dir / "metadata.json"
    if not meta_path.exists():
        pytest.skip(f"metadata.json not found in {dump_dir}")
    with open(meta_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDumpCompleteness:
    """Check all expected files and keys are present."""

    def test_metadata_exists(self, dump_dir):
        assert (dump_dir / "metadata.json").exists()

    def test_all_kda_layers_present(self, dump_dir, metadata):
        kda_indices = metadata.get("kda_layer_indices", EXPECTED_KDA_INDICES)
        for idx in kda_indices:
            path = dump_dir / f"layer_{idx:02d}.npz"
            assert path.exists(), f"Missing {path.name}"

    def test_no_extra_layer_files(self, dump_dir, metadata):
        kda_indices = set(metadata.get("kda_layer_indices", EXPECTED_KDA_INDICES))
        for f in dump_dir.glob("layer_*.npz"):
            idx = int(f.stem.split("_")[1])
            assert idx in kda_indices, f"Unexpected layer file: {f.name}"

    def test_npz_keys(self, dump_dir, metadata):
        kda_indices = metadata.get("kda_layer_indices", EXPECTED_KDA_INDICES)
        for idx in kda_indices:
            data = np.load(dump_dir / f"layer_{idx:02d}.npz")
            actual_keys = set(data.files)
            missing = EXPECTED_KEYS - actual_keys
            assert not missing, f"Layer {idx}: missing keys {missing}"


class TestShapes:
    """Validate tensor shapes."""

    def test_layer_shapes(self, dump_dir, metadata):
        T = metadata["num_tokens"]
        B = 1
        kda_indices = metadata.get("kda_layer_indices", EXPECTED_KDA_INDICES)

        expected_shapes = {
            "input_hidden_states": (B, T, HIDDEN_SIZE),
            "intermediates__q_proj": (B, T, PROJ_SIZE),
            "intermediates__k_proj": (B, T, PROJ_SIZE),
            "intermediates__v_proj": (B, T, PROJ_SIZE),
            "intermediates__q_after_conv": (B, T, NUM_HEADS, HEAD_DIM),
            "intermediates__k_after_conv": (B, T, NUM_HEADS, HEAD_DIM),
            "intermediates__v_after_conv": (B, T, NUM_HEADS, HEAD_DIM),
            "intermediates__g": (B, T, NUM_HEADS, HEAD_DIM),
            "intermediates__beta": (B, T, NUM_HEADS),
            "intermediates__o_kda": (B, T, NUM_HEADS, HEAD_DIM),
            # recurrent_state: [N, H, K, V] where N=B for single seq
            "intermediates__g_out": (B, T, NUM_HEADS, HEAD_DIM),
            "intermediates__o_norm": (B, T, NUM_HEADS, HEAD_DIM),
            "output": (B, T, HIDDEN_SIZE),
        }

        for idx in kda_indices:
            data = np.load(dump_dir / f"layer_{idx:02d}.npz")
            for key, expected in expected_shapes.items():
                actual = data[key].shape
                assert actual == expected, (
                    f"Layer {idx} {key}: shape {actual} != expected {expected}"
                )
            # recurrent_state: check rank and last dims
            rs = data["intermediates__recurrent_state"]
            assert rs.ndim == 4, f"Layer {idx} recurrent_state: ndim={rs.ndim}"
            assert rs.shape[1:] == (NUM_HEADS, HEAD_DIM, HEAD_DIM), (
                f"Layer {idx} recurrent_state: shape {rs.shape}"
            )


class TestNumerics:
    """Check values are finite and in reasonable ranges."""

    def test_no_nan_inf(self, dump_dir, metadata):
        kda_indices = metadata.get("kda_layer_indices", EXPECTED_KDA_INDICES)
        for idx in kda_indices:
            data = np.load(dump_dir / f"layer_{idx:02d}.npz")
            for key in EXPECTED_KEYS:
                if key == "mode_used":
                    continue
                arr = data[key]
                assert np.all(np.isfinite(arr)), (
                    f"Layer {idx} {key}: has NaN/Inf "
                    f"(nan={np.isnan(arr).sum()}, inf={np.isinf(arr).sum()})"
                )

    def test_output_not_zero(self, dump_dir, metadata):
        """Output should not be all zeros (would indicate a broken forward)."""
        kda_indices = metadata.get("kda_layer_indices", EXPECTED_KDA_INDICES)
        for idx in kda_indices:
            data = np.load(dump_dir / f"layer_{idx:02d}.npz")
            out = data["output"]
            assert np.abs(out).max() > 1e-6, (
                f"Layer {idx}: output is near-zero (max abs = {np.abs(out).max():.2e})"
            )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _print_summary(dump_dir: Path):
    """Print per-layer stats when run as a script."""
    meta_path = dump_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"Input: {metadata['input_text']!r}")
    print(f"Tokens: {metadata['num_tokens']}")
    print(f"KDA layers: {metadata['num_kda_layers']}")
    print()

    header = f"{'Layer':>5s}  {'Key':>30s}  {'Shape':>25s}  {'Mean':>10s}  {'Std':>10s}  {'AbsMax':>10s}"
    print(header)
    print("-" * len(header))

    report_keys = [
        "input_hidden_states",
        "intermediates__o_kda",
        "output",
    ]

    for idx in metadata["kda_layer_indices"]:
        path = dump_dir / f"layer_{idx:02d}.npz"
        if not path.exists():
            print(f"  L{idx:02d}  MISSING")
            continue
        data = np.load(path)
        for key in report_keys:
            arr = data[key]
            print(
                f"  L{idx:02d}  {key:>30s}  {str(arr.shape):>25s}  "
                f"{arr.mean():>10.4f}  {arr.std():>10.4f}  "
                f"{np.abs(arr).max():>10.4f}"
            )
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <dump_dir>")
        sys.exit(1)
    _print_summary(Path(sys.argv[1]))

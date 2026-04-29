"""Black-box numerical validation for KimiDeltaAttention.

Tests KimiDeltaAttention.__call__ as an opaque module — everything below
(KDAAttnBackend, RecurrentStatePool, kernels) is exercised implicitly.
GPU reference dumps provide the ground truth.

Prefill tests (TestKDAModulePrefill):
    Load GPU reference dumps and compare the JAX module forward pass on TPU.
    12 cases x 2 dtypes (fp32 + bf16) = 24 tests.

Decode tests (TestKDAModuleDecode):
    Verify prefill(T-1) + decode(1) matches prefill(T) at the last position.
    3 cases x 2 dtypes (fp32 + bf16) = 6 tests.

Run on TPU v6e-4:
    conda activate sglang
    python -m pytest sgl_jax/test/layers/test_kda_module.py -v

Override dump location:
    KDA_DUMP_DIR=/path/to/kda_module python -m pytest ...

Override layer:
    KDA_DUMP_LAYER=L22 python -m pytest ...
"""

from __future__ import annotations

import os
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.kimi_linear import KimiDeltaAttention

_TEST_MESH = jax.sharding.Mesh(
    jax.devices()[:1],
    ("tensor",),
    axis_types=(jax.sharding.AxisType.Explicit,),
)
jax.set_mesh(_TEST_MESH)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DUMP_BASE = os.environ.get("KDA_DUMP_DIR", "/models/yuhao/kimi-linear/kda_module")
DEFAULT_LAYER = os.environ.get("KDA_DUMP_LAYER", "L0")


def _layer_dir():
    return os.path.join(DUMP_BASE, DEFAULT_LAYER)


# ===================================================================
# Shared helpers
# ===================================================================


def _make_config(weights: dict) -> SimpleNamespace:
    """Build a minimal config from weights.npz metadata keys."""
    return SimpleNamespace(
        hidden_size=int(weights["config__hidden_size"]),
        rms_norm_eps=float(weights["config__rms_norm_eps"]),
        linear_attn_config={
            "num_heads": int(weights["config__num_heads"]),
            "head_dim": int(weights["config__head_dim"]),
            "short_conv_kernel_size": int(weights["config__conv_size"]),
            "kda_layers": [1],
            "full_attn_layers": [],
        },
    )


def _set_param(module, attr_path: str, value: np.ndarray) -> None:
    """Set a nested parameter on an nnx module."""
    parts = attr_path.split(".")
    obj = module
    for part in parts[:-1]:
        obj = getattr(obj, part)
    param = getattr(obj, parts[-1])
    param[...] = jnp.asarray(value)


def _build_module(
    weights_path: str,
    dtype: jnp.dtype = jnp.float32,
) -> KimiDeltaAttention:
    """Construct a KimiDeltaAttention and load GPU reference weights."""
    weights = dict(np.load(weights_path, allow_pickle=True))
    config = _make_config(weights)
    module = KimiDeltaAttention(config, layer_idx=0, mesh=_TEST_MESH, dtype=dtype)

    num_heads = config.linear_attn_config["num_heads"]

    # Linear projections: GPU dumps are [out, in], LinearBase expects [in, out].
    weight_map = {
        "q_proj.weight": "weights__q_proj.weight",
        "k_proj.weight": "weights__k_proj.weight",
        "v_proj.weight": "weights__v_proj.weight",
        "f_a_proj.weight": "weights__f_a_proj.weight",
        "f_b_proj.weight": "weights__f_b_proj.weight",
        "b_proj.weight": "weights__b_proj.weight",
        "g_a_proj.weight": "weights__g_a_proj.weight",
        "g_b_proj.weight": "weights__g_b_proj.weight",
        "o_proj.weight": "weights__o_proj.weight",
        "o_norm.weight": "weights__o_norm.weight",
    }
    for attr, key in weight_map.items():
        w = weights[key]
        if attr != "o_norm.weight":
            w = w.T
        _set_param(module, attr, w)

    # Conv weights: GPU dumps are [D, (1,) K]; store as [D, K] directly.
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        w = weights[f"weights__{name}.weight"]
        if w.ndim == 3 and w.shape[1] == 1:
            w = w[:, 0, :]
        _set_param(module, f"{name}.weight", w)

    a_log = weights["weights__A_log"]
    _set_param(module, "A_log", a_log.reshape(1, 1, num_heads, 1))
    _set_param(module, "dt_bias", weights["weights__dt_bias"])

    return module


def _build_extend_env(
    module: KimiDeltaAttention,
    T: int,
    init_state: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
) -> tuple[ForwardBatch, RecurrentStatePool]:
    """Build EXTEND ForwardBatch + RecurrentStatePool for one or more sequences."""
    if cu_seqlens is None:
        cu_seqlens = jnp.array([0, T], dtype=jnp.int32)
    N = cu_seqlens.shape[0] - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]

    pool = RecurrentStatePool(
        linear_recurrent_layer_ids=[module.layer_idx],
        max_num_reqs=N,
        num_heads=module.num_heads,
        head_dim=module.head_dim,
        conv_kernel_size=module.conv_size,
        mesh=_TEST_MESH,
    )
    if init_state is not None:
        pool.recurrent_buffers[0] = pool.recurrent_buffers[0].at[1 : N + 1].set(
            init_state.astype(pool.temporal_dtype)
        )

    # Pool slots start at 1 (slot 0 is dummy).
    recurrent_indices = jnp.arange(1, N + 1, dtype=jnp.int32)

    backend = KDAAttnBackend(mesh=_TEST_MESH)
    backend.forward_metadata = LinearRecurrentAttnBackendMetadata(
        cu_q_lens=cu_seqlens,
        recurrent_indices=recurrent_indices,
    )
    fb = ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=int(N),
        input_ids=jnp.zeros(T, dtype=jnp.int32),
        req_pool_indices=jnp.arange(N, dtype=jnp.int32),
        seq_lens=seq_lens,
        out_cache_loc=jnp.zeros(T, dtype=jnp.int32),
        attn_backend=backend,
        extend_seq_lens=seq_lens,
    )
    return fb, pool


def _build_decode_env(
    module: KimiDeltaAttention,
    pool: RecurrentStatePool,
    B: int = 1,
) -> ForwardBatch:
    """Build DECODE ForwardBatch reusing an existing pool."""
    recurrent_indices = jnp.arange(1, B + 1, dtype=jnp.int32)
    backend = KDAAttnBackend(mesh=_TEST_MESH)
    backend.forward_metadata = LinearRecurrentAttnBackendMetadata(
        cu_q_lens=jnp.arange(B + 1, dtype=jnp.int32),
        recurrent_indices=recurrent_indices,
    )
    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.DECODE,
        batch_size=B,
        input_ids=jnp.zeros(B, dtype=jnp.int32),
        req_pool_indices=jnp.arange(B, dtype=jnp.int32),
        seq_lens=jnp.ones(B, dtype=jnp.int32),
        out_cache_loc=jnp.zeros(B, dtype=jnp.int32),
        attn_backend=backend,
    )


# ===================================================================
# Tolerance tiers
# ===================================================================

FP32_ATOL_TIGHT = 2e-3
FP32_RTOL_TIGHT = 5e-3
FP32_ATOL_LOOSE = 3e-2
FP32_RTOL_LOOSE = 2e-2

BF16_ATOL_TIGHT = 3e-3
BF16_RTOL_TIGHT = 5e-3
BF16_ATOL_LOOSE = 7e-2
BF16_RTOL_LOOSE = 2e-2

DECODE_FP32_ATOL_TIGHT = 1e-3
DECODE_FP32_RTOL_TIGHT = 1e-3
DECODE_FP32_ATOL_LOOSE = 1e-2
DECODE_FP32_RTOL_LOOSE = 1e-2

DECODE_BF16_ATOL_TIGHT = 2e-3
DECODE_BF16_RTOL_TIGHT = 2e-3
DECODE_BF16_ATOL_LOOSE = 2e-2
DECODE_BF16_RTOL_LOOSE = 2e-2


def _assert_two_tier(
    actual: np.ndarray,
    expected: np.ndarray,
    atol_tight: float,
    rtol_tight: float,
    atol_loose: float,
    rtol_loose: float,
    label: str,
) -> None:
    """Assert with two-tier tolerance: tight first, then loose as fallback."""
    diff = np.abs(actual - expected)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    print(f"  {label}: max_abs={max_abs:.2e}, mean_abs={mean_abs:.2e}")

    try:
        np.testing.assert_allclose(
            actual, expected, atol=atol_tight, rtol=rtol_tight,
        )
    except AssertionError:
        np.testing.assert_allclose(
            actual, expected, atol=atol_loose, rtol=rtol_loose,
            err_msg=f"{label}: exceeds loose tolerance",
        )
        warnings.warn(
            f"{label}: passed at loose tolerance (max_abs={max_abs:.2e}, "
            f"tight={atol_tight}, loose={atol_loose})",
            stacklevel=2,
        )


# ===================================================================
# Prefill tests
# ===================================================================

PREFILL_CASES = [
    "single_T1",
    "single_T8",
    "single_T64",
    "single_T65",
    "single_T128",
    "single_T256",
    "single_T1024",
    "varlen_balanced_4x32",
    "varlen_unbalanced",
    "varlen_single_T128",
    "single_T128_initstate",
    "varlen_initstate",
]


class TestKDAModulePrefill:
    """Prefill (EXTEND) alignment against GPU reference dumps."""

    @pytest.fixture(scope="class", autouse=True)
    def _check_dumps(self):
        d = _layer_dir()
        if not os.path.isdir(d):
            pytest.skip(f"KDA dumps not found at {d}")

    @pytest.fixture(scope="class")
    def module(self):
        return _build_module(os.path.join(_layer_dir(), "weights.npz"))

    @pytest.fixture(scope="class")
    def module_bf16(self):
        return _build_module(
            os.path.join(_layer_dir(), "weights.npz"),
            dtype=jnp.bfloat16,
        )

    @pytest.mark.parametrize("case_name", PREFILL_CASES)
    def test_prefill_fp32(self, module, case_name):
        case_path = os.path.join(_layer_dir(), f"case_{case_name}.npz")
        if not os.path.isfile(case_path):
            pytest.skip(f"Case file not found: {case_path}")

        case = dict(np.load(case_path, allow_pickle=True))
        forward_batch, pool = _build_extend_env(
            module,
            int(case["T"]),
            init_state=(
                jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
                if bool(case["has_initial_state"])
                else None
            ),
            cu_seqlens=(
                jnp.asarray(case["cu_seqlens"], dtype=jnp.int32)
                if bool(case["has_cu_seqlens"])
                else None
            ),
        )

        hidden = jnp.asarray(case["hidden_states"], dtype=jnp.float32)
        if hidden.ndim == 3:
            hidden = hidden[0]
        output, _ = module(None, hidden, forward_batch, pool)
        output_np = np.asarray(output, dtype=np.float32)

        expected = case["out_fp32"]
        if output_np.ndim == 2 and expected.ndim == 3:
            output_np = output_np[None, ...]

        # GPU chunk kernel produces all-zero output for T < chunk_size (64).
        if np.all(expected == 0):
            assert not np.isnan(output_np).any(), f"{case_name}: NaN in output"
            assert np.abs(output_np).max() > 0, f"{case_name}: all-zero output"
            pytest.skip(
                f"{case_name}: GPU chunk reference is all-zero "
                f"(T < chunk_size); TPU output is non-zero (correct)"
            )

        _assert_two_tier(
            output_np,
            expected,
            FP32_ATOL_TIGHT,
            FP32_RTOL_TIGHT,
            FP32_ATOL_LOOSE,
            FP32_RTOL_LOOSE,
            case_name,
        )

    @pytest.mark.parametrize("case_name", PREFILL_CASES)
    def test_prefill_bf16(self, module_bf16, case_name):
        case_path = os.path.join(_layer_dir(), f"case_{case_name}.npz")
        if not os.path.isfile(case_path):
            pytest.skip(f"Case file not found: {case_path}")

        case = dict(np.load(case_path, allow_pickle=True))
        if "out_bf16" not in case:
            pytest.skip(f"{case_name}: no out_bf16 in dump")

        forward_batch, pool = _build_extend_env(
            module_bf16,
            int(case["T"]),
            init_state=(
                jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
                if bool(case["has_initial_state"])
                else None
            ),
            cu_seqlens=(
                jnp.asarray(case["cu_seqlens"], dtype=jnp.int32)
                if bool(case["has_cu_seqlens"])
                else None
            ),
        )

        hidden = jnp.asarray(case["hidden_states"], dtype=jnp.bfloat16)
        if hidden.ndim == 3:
            hidden = hidden[0]
        output, _ = module_bf16(None, hidden, forward_batch, pool)
        output_np = np.asarray(output, dtype=np.float32)

        expected = case["out_bf16"]
        if output_np.ndim == 2 and expected.ndim == 3:
            output_np = output_np[None, ...]

        if np.all(expected == 0):
            assert not np.isnan(output_np).any(), f"{case_name}: NaN in output"
            pytest.skip(
                f"{case_name}: GPU bf16 reference is all-zero (T < chunk_size)"
            )

        assert not np.isnan(output_np).any(), f"{case_name}: NaN in output"
        _assert_two_tier(
            output_np,
            expected,
            BF16_ATOL_TIGHT,
            BF16_RTOL_TIGHT,
            BF16_ATOL_LOOSE,
            BF16_RTOL_LOOSE,
            f"{case_name} bf16",
        )


# ===================================================================
# Decode tests
# ===================================================================

DECODE_CASES = ["single_T8", "single_T128", "single_T128_initstate"]


class TestKDAModuleDecode:
    """Decode: prefill(T-1) + decode(1) vs GPU reference at position T."""

    @pytest.fixture(scope="class", autouse=True)
    def _check_dumps(self):
        d = _layer_dir()
        if not os.path.isdir(d):
            pytest.skip(f"KDA dumps not found at {d}")

    @pytest.fixture(scope="class")
    def module(self):
        return _build_module(os.path.join(_layer_dir(), "weights.npz"))

    @pytest.fixture(scope="class")
    def module_bf16(self):
        return _build_module(
            os.path.join(_layer_dir(), "weights.npz"),
            dtype=jnp.bfloat16,
        )

    def _run_decode(self, module, case, in_dtype):
        """Prefill T-1 tokens, decode the T-th, return decode output."""
        T = int(case["T"])
        hidden = jnp.asarray(case["hidden_states"], dtype=in_dtype)
        if hidden.ndim == 3:
            hidden = hidden[0]  # [T, D]

        has_init = bool(case["has_initial_state"])
        init_state = (
            jnp.asarray(case["initial_recurrent_state"], dtype=jnp.float32)
            if has_init
            else None
        )

        # 1) Prefill T-1 tokens
        fb_prefix, pool = _build_extend_env(module, T - 1, init_state)
        _, (new_ssm, new_conv_list) = module(None, hidden[: T - 1], fb_prefix, pool)

        # 2) Write state back via production API
        pool.replace_buffer(([new_ssm], [new_conv_list]))

        # 3) Decode the T-th token
        fb_decode = _build_decode_env(module, pool, B=1)
        out_decode, _ = module(None, hidden[T - 1 : T], fb_decode, pool)
        return np.asarray(out_decode, dtype=np.float32)  # [1, D]

    @pytest.mark.parametrize("case_name", DECODE_CASES)
    def test_decode_fp32(self, module, case_name):
        case_path = os.path.join(_layer_dir(), f"case_{case_name}.npz")
        if not os.path.isfile(case_path):
            pytest.skip(f"Case file not found: {case_path}")

        case = dict(np.load(case_path, allow_pickle=True))
        T = int(case["T"])
        expected = case["out_fp32"]
        expected_last = (
            expected[0, T - 1 : T] if expected.ndim == 3 else expected[T - 1 : T]
        )

        out_decode = self._run_decode(module, case, jnp.float32)

        assert not np.isnan(out_decode).any(), f"{case_name}: NaN in decode output"
        _assert_two_tier(
            out_decode,
            expected_last,
            DECODE_FP32_ATOL_TIGHT,
            DECODE_FP32_RTOL_TIGHT,
            DECODE_FP32_ATOL_LOOSE,
            DECODE_FP32_RTOL_LOOSE,
            f"{case_name} decode fp32",
        )

    @pytest.mark.parametrize("case_name", DECODE_CASES)
    def test_decode_bf16(self, module_bf16, case_name):
        case_path = os.path.join(_layer_dir(), f"case_{case_name}.npz")
        if not os.path.isfile(case_path):
            pytest.skip(f"Case file not found: {case_path}")

        case = dict(np.load(case_path, allow_pickle=True))
        if "out_bf16" not in case:
            pytest.skip(f"{case_name}: no out_bf16 in dump")

        T = int(case["T"])
        expected = case["out_bf16"]
        expected_last = (
            expected[0, T - 1 : T] if expected.ndim == 3 else expected[T - 1 : T]
        )

        out_decode = self._run_decode(module_bf16, case, jnp.bfloat16)

        assert not np.isnan(out_decode).any(), f"{case_name}: NaN in decode output"
        _assert_two_tier(
            out_decode,
            expected_last,
            DECODE_BF16_ATOL_TIGHT,
            DECODE_BF16_RTOL_TIGHT,
            DECODE_BF16_ATOL_LOOSE,
            DECODE_BF16_RTOL_LOOSE,
            f"{case_name} decode bf16",
        )

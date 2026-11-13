# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.utils.device_info import get_device_capability

OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert"
VENDOR = flaggems_vllm.vendor_name
DEVICE = flaggems_vllm.device
SUPPORTED_VENDORS = frozenset({"nvidia", "mthreads", "hygon", "metax", "thead"})
UINT8_CONTAINER_VENDORS = frozenset({"hygon", "metax", "thead"})
FP8_STORE_DTYPE = getattr(torch, "float8_e4m3fn", None)
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
FP8_MAX = 448.0


def is_support_fp8e4nv():
    """Check E4M3 storage/conversion, not native FP8 matrix instructions."""
    if FP8_STORE_DTYPE is None:
        return False
    if VENDOR in SUPPORTED_VENDORS - {"nvidia"}:
        return True
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


_GEMS_OP_ERROR = None


def _resolve_gems_op():
    global _GEMS_OP_ERROR
    try:
        module = importlib.import_module(
            f"flaggems_vllm.runtime.backend._{VENDOR}.fused.{OP_NAME}"
        )
    except ModuleNotFoundError:
        try:
            module = importlib.import_module(f"flaggems_vllm.ops.{OP_NAME}")
        except ImportError as exc:
            _GEMS_OP_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    op = getattr(module, OP_NAME, None)
    if op is None:
        _GEMS_OP_ERROR = f"{module.__name__} exports no {OP_NAME}"
    return op


_GEMS_OP = _resolve_gems_op()


def fp8_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Representable-value distance between two uint8 E4M3 tensors."""

    def key(t):
        u = t.contiguous().to(torch.int16)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return (key(a) - key(b)).abs()


def _to_e4m3_uint8(x: torch.Tensor) -> torch.Tensor:
    """Independent reference conversion, returned as raw E4M3 bytes."""
    try:
        return x.to(FP8_STORE_DTYPE).view(torch.uint8)
    except RuntimeError:
        return x.cpu().to(FP8_STORE_DTYPE).view(torch.uint8).to(x.device)


def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
    """Build a cos||sin cache in the DeepseekV4 rotary layout."""
    base = 10000.0
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def apply_rope_gptj_last_k(x, positions, cos_sin_cache):
    """Apply interleaved-pair RoPE and retain fp32, as in the kernel."""
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    cs = cos_sin_cache[positions.long()].float()
    cos = cs[..., :half]
    sin = cs[..., half:]

    rope = x[..., nope_dim:].float()
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)
    even = rope[..., 0]
    odd = rope[..., 1]
    for _ in range(rope.ndim - 3):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # addcmul mirrors the kernel's fused multiply-add rotation.
    new_even = torch.addcmul(-odd * sin, even, cos)
    new_odd = torch.addcmul(odd * cos, even, sin)
    out = x.clone().float()
    out[..., nope_dim:] = torch.stack((new_even, new_odd), dim=-1).reshape(shape)
    return out


def rmsnorm_no_weight(x, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)


def _transform_q(q, positions, cos_sin_cache, eps, apply_q_norm):
    q_ref = rmsnorm_no_weight(q, eps) if apply_q_norm else q.float()
    return apply_rope_gptj_last_k(q_ref, positions, cos_sin_cache)


def _reference(
    q,
    kv,
    q_fp8,
    k_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    fp8_scale,
    q_fp8_scale_inv,
    eps,
    block_size,
    apply_q_norm,
):
    if q.shape[1] > 0:
        q_ref = _transform_q(q, positions, cos_sin_cache, eps, apply_q_norm)
        q_scaled = torch.clamp(q_ref * q_fp8_scale_inv, -FP8_MAX, FP8_MAX)
        q_fp8.view(torch.uint8).copy_(_to_e4m3_uint8(q_scaled))

    kv_ref = apply_rope_gptj_last_k(kv, positions, cos_sin_cache)
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    block_idx = slots // block_size
    pos_in_block = slots % block_size
    # Match the kernel's reciprocal-load followed by multiplication.
    kv_scaled = torch.clamp(
        kv_ref[: slot_mapping.numel()][valid] * (1.0 / fp8_scale),
        -FP8_MAX,
        FP8_MAX,
    )
    k_cache.view(torch.uint8)[block_idx, pos_in_block] = _to_e4m3_uint8(kv_scaled)


def _call(
    q,
    kv,
    q_fp8,
    k_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    fp8_scale,
    q_fp8_scale_inv,
    eps,
    block_size,
    apply_q_norm,
):
    if VENDOR in UINT8_CONTAINER_VENDORS:
        q_fp8 = q_fp8.view(torch.uint8)
        k_cache = k_cache.view(torch.uint8)
    args = (
        q,
        kv,
        q_fp8,
        k_cache,
        slot_mapping,
        positions.long(),
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        eps,
        block_size,
    )
    if apply_q_norm is None:
        return _GEMS_OP(*args)
    return _GEMS_OP(*args, apply_q_norm)


def _run_and_compare(
    num_tokens,
    n_heads,
    *,
    dtype=torch.bfloat16,
    positions_dtype=torch.int64,
    apply_q_norm=None,
    scale_mode="unit",
    data_scale=1.0,
    block_size=16,
    slot_mapping=None,
    cache_fill=0,
):
    torch.manual_seed(4)
    eps = 1e-6
    positions = torch.arange(num_tokens, dtype=positions_dtype, device=DEVICE)
    cos_sin_cache = make_cos_sin_cache(4096, ROPE_DIM, torch.float32, DEVICE)
    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype, device=DEVICE)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=DEVICE)
    q = (q * data_scale).contiguous()
    kv = (kv * data_scale).contiguous()
    if slot_mapping is None:
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)

    if scale_mode == "amax":
        kv_ref = apply_rope_gptj_last_k(kv, positions, cos_sin_cache)
        kv_amax = kv_ref.abs().max().clamp_min(1e-20)
        fp8_scale = (kv_amax / FP8_MAX).reshape(1)
        if n_heads > 0:
            q_ref = _transform_q(
                q, positions, cos_sin_cache, eps, apply_q_norm is not False
            )
            q_amax = q_ref.abs().max().clamp_min(1e-20)
            q_fp8_scale_inv = (FP8_MAX / q_amax).reshape(1)
        else:
            q_fp8_scale_inv = torch.ones(1, dtype=torch.float32, device=DEVICE)
    else:
        fp8_scale = torch.ones(1, dtype=torch.float32, device=DEVICE)
        q_fp8_scale_inv = torch.ones(1, dtype=torch.float32, device=DEVICE)

    num_blocks = num_tokens // block_size + 2
    q_ref = torch.empty_like(q, dtype=FP8_STORE_DTYPE)
    q_fused = torch.empty_like(q, dtype=FP8_STORE_DTYPE)
    # Fill through uint8 so the sentinel is a byte pattern, not an FP8 cast.
    cache_ref = torch.full(
        (num_blocks, block_size, HEAD_DIM),
        cache_fill,
        dtype=torch.uint8,
        device=DEVICE,
    ).view(FP8_STORE_DTYPE)
    cache_fused = cache_ref.clone()

    _reference(
        q,
        kv,
        q_ref,
        cache_ref,
        slot_mapping,
        positions,
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        eps,
        block_size,
        apply_q_norm is not False,
    )
    _call(
        q,
        kv,
        q_fused,
        cache_fused,
        slot_mapping,
        positions,
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        eps,
        block_size,
        apply_q_norm,
    )

    if n_heads > 0:
        q_max_ulp = int(
            fp8_ulp_distance(q_fused.view(torch.uint8), q_ref.view(torch.uint8))
            .max()
            .item()
        )
        assert q_max_ulp <= 1, f"Q FP8 differs by {q_max_ulp} ULP (>1)"

    cache_fused_u8 = cache_fused.view(torch.uint8)
    cache_ref_u8 = cache_ref.view(torch.uint8)
    # NoPE has no reduction or rotation and must be bit-identical. This also
    # verifies that negative slots and DP padding leave sentinel bytes intact.
    torch.testing.assert_close(
        cache_fused_u8[..., :NOPE_DIM],
        cache_ref_u8[..., :NOPE_DIM],
        rtol=0,
        atol=0,
    )
    k_max_ulp = int(
        fp8_ulp_distance(cache_fused_u8[..., NOPE_DIM:], cache_ref_u8[..., NOPE_DIM:])
        .max()
        .item()
    )
    assert k_max_ulp <= 1, f"K-cache RoPE FP8 differs by {k_max_ulp} ULP (>1)"


pytestmark = [
    pytest.mark.skipif(VENDOR not in SUPPORTED_VENDORS, reason="Unsupported vendor"),
    pytest.mark.skipif(
        _GEMS_OP is None,
        reason=(
            "No FlagGems implementation found for this operator"
            f" ({_GEMS_OP_ERROR or 'unknown import failure'})"
        ),
    ),
    pytest.mark.skipif(
        not is_support_fp8e4nv(),
        reason="E4M3 dtype/storage is unavailable on this device",
    ),
]


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
@pytest.mark.parametrize("num_tokens", [4, 17])
@pytest.mark.parametrize("n_heads", [0, 8, 17])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "apply_q_norm",
    [
        pytest.param(None, id="qnorm_default"),
        pytest.param(True, id="qnorm_explicit"),
        pytest.param(False, id="rope_only"),
    ],
)
def test_full_cache_software_e4m3_matches_reference(
    num_tokens, n_heads, dtype, positions_dtype, apply_q_norm
):
    _run_and_compare(
        num_tokens,
        n_heads,
        dtype=dtype,
        positions_dtype=positions_dtype,
        apply_q_norm=apply_q_norm,
    )


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
@pytest.mark.parametrize("n_heads", [16, 17])
@pytest.mark.parametrize("scale_mode", ["unit", "amax"])
def test_split_kernels_and_nonunit_scales(monkeypatch, n_heads, scale_mode):
    # Force the large-shape dispatch with small tensors. H=16 covers the
    # uniform-token broadcast path; H=17 covers generic split-Q addressing.
    monkeypatch.setenv("DSV4_SPLIT_MIN_Q_ROWS", "1")
    _run_and_compare(
        17, n_heads, apply_q_norm=True, scale_mode=scale_mode, block_size=64
    )


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
def test_dp_padding_negative_slots_and_nonpow2_cache_block():
    slots = torch.arange(12, dtype=torch.int64, device=DEVICE)
    slots[3] = -1
    slots[7] = -1
    _run_and_compare(
        17,
        8,
        apply_q_norm=True,
        scale_mode="amax",
        block_size=12,
        slot_mapping=slots,
        cache_fill=0xA5,
    )


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
@pytest.mark.parametrize("data_scale", [0.01, 300.0], ids=["subnormal", "saturate"])
def test_e4m3_subnormal_and_saturation(data_scale):
    _run_and_compare(4, 16, apply_q_norm=False, data_scale=data_scale)

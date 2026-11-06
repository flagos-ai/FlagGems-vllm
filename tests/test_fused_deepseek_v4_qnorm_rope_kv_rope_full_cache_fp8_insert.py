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

import pytest
import torch
from vllm.platforms import current_platform

from flaggems_vllm.ops.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert import (
    fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert as _triton_op,
)


def fp8_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Representable-step distance between two 8-bit fp8 tensors.

    Reinterprets the fp8 bytes under a sign-magnitude total ordering so that
    adjacent representable values differ by exactly 1. Inputs must already share
    the same fp8 encoding (e.g. both FP8_STORE_DTYPE).
    """

    def key(t: torch.Tensor) -> torch.Tensor:
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return (key(a) - key(b)).abs()


def get_fp8_min_max() -> tuple[float, float]:
    """Get the min and max values for FP8 quantization."""
    # Using the default value (240.0) from pytorch will cause accuracy
    # issue on dynamic quantization models on ROCm. Here, use 224.0 for fnuz
    # on ROCm platforms that use the torch.float8_e4m3fnuz dtype.
    if current_platform.is_fp8_fnuz():
        return -224.0, 224.0
    finfo = torch.finfo(current_platform.fp8_dtype())
    return finfo.min, finfo.max


# ── Constants matching the kernel ────────────────────────────────────────────
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448
# FNUZ on gfx942, OCP elsewhere.
USE_FNUZ = current_platform.is_fp8_fnuz()
_, FP8_MAX = get_fp8_min_max()
# The kernel emits FNUZ-encoded fp8 bytes on gfx942 (rocm_cvt_float_to_fp8_e4m3)
# but stores them into float8_e4m3fn-typed tensors, matching vLLM's ROCm cache
# convention. References must encode under the same scheme and the kernel's
# e4m3fn-typed outputs must be reinterpreted under it before decoding.
FP8_STORE_DTYPE = torch.float8_e4m3fnuz if USE_FNUZ else torch.float8_e4m3fn


# ── PyTorch reference implementations ────────────────────────────────────────


def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
    """Build a cos||sin cache matching DeepseekV4ScalingRotaryEmbedding layout.
    cos_sin_cache[pos, :rope_dim/2] = cos(theta), [rope_dim/2:] = sin(theta).
    """
    base = 10000.0
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rope_dim/2]
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rope_dim]
    return cache.to(dtype)


def apply_rope_gptj_last_k(
    x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """GPT-J-style (interleaved-pair) RoPE on the LAST rope_dim elements.

    x: [..., head_dim] float32
    positions: [num_tokens] int64 (positions[i] corresponds to x[i, ...])
    cos_sin_cache: [max_pos, rope_dim] float (cos|sin layout)

    Returns rotated x (same shape/dtype).
    """
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim

    cs = cos_sin_cache[positions.long()].to(torch.float32)
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

    # Use addcmul (an FMA) for the 2x2 rotation to mirror the kernel's
    # `e*c - o*s` fused form. This keeps the reference close to the kernel, but
    # the fp32 reference and the fp32 GPU kernel can still round to bf16 on
    # opposite sides of a round-to-nearest tie for a tiny number of elements at
    # high positions, so callers allow for bf16 rounding in the RoPE region.
    new_even = torch.addcmul(-odd * sin, even, cos)
    new_odd = torch.addcmul(odd * cos, even, sin)
    rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)

    out = x.clone().float()
    out[..., nope_dim:] = rope_rotated
    return out.to(x.dtype)


def rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with no learnable weight, matching
    `RMSNorm(head_dim, has_weight=False)`.

    Returns fp32 so callers can chain RoPE without an intermediate bf16 round
    (the kernel keeps the whole RMSNorm→RoPE pipeline in fp32 and rounds once
    at the final store).
    """
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason=("CUDA not available")
)


def _call_full_cache_fp8_fused(
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
    bs,
    apply_q_norm=None,
):
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
        bs,
    )
    if apply_q_norm is None:
        _triton_op(*args)
    else:
        _triton_op(*args, apply_q_norm)


def _as_stored_fp8(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a float8_e4m3fn-typed kernel output under the real (FNUZ on
    gfx942) encoding the kernel actually wrote, without touching the bytes."""
    return t.contiguous().view(torch.uint8).view(FP8_STORE_DTYPE)


def _fp8_full_cache_reference(
    q,
    kv,
    k_cache,
    q_fp8,
    slot_mapping,
    positions,
    cos_sin_cache,
    eps,
    block_size,
    fp8_scale,
    q_fp8_scale_inv,
    apply_q_norm,
):
    q_ref = rmsnorm_no_weight(q, eps) if apply_q_norm else q.float()
    q_ref = apply_rope_gptj_last_k(q_ref, positions, cos_sin_cache)
    q_fp8.copy_(
        torch.clamp(q_ref.float() * q_fp8_scale_inv, -FP8_MAX, FP8_MAX).to(
            FP8_STORE_DTYPE
        )
    )

    kv_ref = apply_rope_gptj_last_k(kv, positions, cos_sin_cache)
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    block_idx = slots // block_size
    pos_in_block = slots % block_size
    k_cache[block_idx, pos_in_block] = torch.clamp(
        kv_ref[valid].float() / fp8_scale, -FP8_MAX, FP8_MAX
    ).to(FP8_STORE_DTYPE)


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
def test_full_cache_per_tensor_fp8_matches_reference(
    num_tokens: int,
    n_heads: int,
    dtype: torch.dtype,
    positions_dtype: torch.dtype,
    apply_q_norm: bool | None,
):
    torch.manual_seed(4)
    device = "cuda"
    eps = 1e-6
    block_size = 16
    max_pos = 4096

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=positions_dtype, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    fp8_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    q_fp8_scale_inv = torch.tensor([1.0], dtype=torch.float32, device=device)

    # References are encoded under the scheme the kernel actually writes
    # (FNUZ on gfx942); the kernel's own outputs must stay float8_e4m3fn-typed
    # because the op asserts that dtype.
    q_fp8_ref = torch.empty_like(q, dtype=FP8_STORE_DTYPE)
    q_fp8_fused = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_cache_ref = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=FP8_STORE_DTYPE, device=device
    )
    k_cache_fused = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )

    _fp8_full_cache_reference(
        q,
        kv,
        k_cache_ref,
        q_fp8_ref,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps,
        block_size,
        fp8_scale,
        q_fp8_scale_inv,
        apply_q_norm is not False,
    )
    _call_full_cache_fp8_fused(
        q.clone(),
        kv,
        q_fp8_fused,
        k_cache_fused,
        slot_mapping,
        positions,
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        eps,
        block_size,
        apply_q_norm,
    )

    # Q uses optional RMSNorm followed by RoPE in fp32 before fp8 quant. The
    # kernel and torch reference can land on opposite sides of an fp8
    # round-to-nearest tie, so allow <=1 fp8 ULP. With n_heads == 0 the Q side
    # never runs and q_fp8 is empty, so there is nothing to compare.
    if n_heads > 0:
        q_fused = _as_stored_fp8(q_fp8_fused)
        q_max_ulp = int(fp8_ulp_distance(q_fused, q_fp8_ref).max().item())
        assert q_max_ulp <= 1, f"Q fp8 differs by {q_max_ulp} ULP (>1)"

    # K-cache NoPE region [0, NOPE_DIM) is a deterministic per-tensor fp8 quant
    # of the (un-rotated) KV input, so it must be bit-identical. The RoPE region
    # [NOPE_DIM, HEAD_DIM) is rotated in fp32 and may differ by <=1 fp8 ULP.
    k_fused = _as_stored_fp8(k_cache_fused)
    torch.testing.assert_close(
        k_fused[..., :NOPE_DIM].float(),
        k_cache_ref[..., :NOPE_DIM].float(),
        rtol=0,
        atol=0,
    )
    k_max_ulp = int(
        fp8_ulp_distance(k_fused[..., NOPE_DIM:], k_cache_ref[..., NOPE_DIM:])
        .max()
        .item()
    )
    assert k_max_ulp <= 1, f"K-cache RoPE fp8 differs by {k_max_ulp} ULP (>1)"

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

import inspect
import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils.device_info import kernel_supports_fp8_e4m3

try:
    import triton.experimental.tle.language as tle

    _HAS_LOAD_REORDER = "reorder" in inspect.signature(tle.range.__init__).parameters
except (ImportError, AttributeError, TypeError, ValueError):
    tle = tl
    _HAS_LOAD_REORDER = False

if kernel_supports_fp8_e4m3():
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


def _get_tma_aligned_size(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


@triton.jit
def _round_scale_to_power_of_two(scale):
    # Round positive finite scales upward; preserve Inf/NaN classification.
    scale = tl.maximum(tl.abs(scale), 1e-10)
    bits = scale.to(tl.uint32, bitcast=True)
    return tl.where(
        bits >= 0x7F800000,
        scale,
        ((bits + 0x007FFFFF) & 0xFF800000).to(tl.float32, bitcast=True),
    )


@triton.jit
def _inverse_rounded_scale(scale):
    # Finite rounded scales here are powers of two in [2**-33, 2**120].
    # Reflect the f32 exponent around its bias; preserve Inf/NaN division behavior.
    bits = scale.to(tl.uint32, bitcast=True)
    inverse = ((254 << 23) - bits).to(tl.float32, bitcast=True)
    return tl.where(
        bits == 0x7F800000, 0.0, tl.where(bits > 0x7F800000, scale, inverse)
    )


@triton.jit
def _apply_inverse_rope(x, partner, cos, sin, rope_local, is_rope):
    rotated = tl.where(
        (rope_local & 1) == 0, x * cos + partner * sin, x * cos - partner * sin
    )
    return tl.where(is_rope, rotated, x)


@triton.jit
def _compute_scale(x, axis: tl.constexpr, eps: tl.constexpr, fp8_max: tl.constexpr):
    return tl.maximum(tl.max(tl.abs(x), axis=axis), eps) * (1.0 / fp8_max)


@triton.jit
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD

    if pid_token >= num_tokens:
        if TMA_ALIGNED_SCALES:
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x = _apply_inverse_rope(x, x_partner, cos_v, sin_v, rope_local, is_rope)
    scales = _compute_scale(
        tl.reshape(x, (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE)), 1, eps, fp8_max
    )
    if TMA_ALIGNED_SCALES:
        scales = tl.math.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scales), 1e-10))))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


@triton.jit
def _fused_inv_rope_fp8_quant_window(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    HEAD_TILE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
    REORDER: tl.constexpr,
    o_stride_token: tl.constexpr,
    o_stride_head: tl.constexpr,
    cache_stride_pos: tl.constexpr,
    fp8_stride_group: tl.constexpr,
    fp8_stride_token: tl.constexpr,
    scale_stride_group: tl.constexpr,
    scale_stride_k: tl.constexpr,
    heads_per_group: tl.constexpr,
):
    token = tl.program_id(0)
    head_base = tl.program_id(1) * (HEAD_TILE * 2)
    heads = tl.arange(0, HEAD_TILE)
    a = tl.arange(0, 256)
    b = tl.arange(0, 128)
    scale_idx = tl.arange(0, 2)
    local = b - 64
    mask = b >= 64
    pos = tl.load(positions_ptr + token)
    idx = tl.maximum(local >> 1, 0)
    cos = tl.load(
        cos_sin_cache_ptr + pos * cache_stride_pos + idx, mask=mask, other=1.0
    )
    sin = tl.load(
        cos_sin_cache_ptr + pos * cache_stride_pos + 32 + idx, mask=mask, other=0.0
    )
    # Split the 512-element head into 256 + 128 NOPE elements and a
    # final 128-element block containing 64 NOPE and 64 RoPE elements.
    # The two slots retain both loop iterations until the stores below.
    a0 = tl.full((HEAD_TILE, 256), 0, tl.uint8)
    b0 = tl.full((HEAD_TILE, 128), 0, tl.uint8)
    c0 = tl.full((HEAD_TILE, 128), 0, tl.uint8)
    sa0 = tl.full((HEAD_TILE, 2), 0, tl.float32)
    sb0 = tl.full((HEAD_TILE,), 0, tl.float32)
    sc0 = tl.full((HEAD_TILE,), 0, tl.float32)
    a1 = tl.full((HEAD_TILE, 256), 0, tl.uint8)
    b1 = tl.full((HEAD_TILE, 128), 0, tl.uint8)
    c1 = tl.full((HEAD_TILE, 128), 0, tl.uint8)
    sa1 = tl.full((HEAD_TILE, 2), 0, tl.float32)
    sb1 = tl.full((HEAD_TILE,), 0, tl.float32)
    sc1 = tl.full((HEAD_TILE,), 0, tl.float32)
    for i in tle.range(0, 2, loop_unroll_factor=2, reorder=REORDER):
        h = head_base + i * HEAD_TILE + heads
        base = o_ptr + token * o_stride_token + h[:, None] * o_stride_head
        xa = tl.load(base + a[None, :]).to(tl.float32)
        xb = tl.load(base + 256 + b[None, :]).to(tl.float32)
        xc = tl.load(base + 384 + b[None, :]).to(tl.float32)
        xe, xo = tl.split(tl.reshape(xc, (HEAD_TILE, 64, 2)))
        partner = tl.reshape(tl.join(xo, xe), (HEAD_TILE, 128))
        xc = _apply_inverse_rope(
            xc, partner, cos[None, :], sin[None, :], local[None, :], mask[None, :]
        )
        sa = _compute_scale(tl.reshape(xa, (HEAD_TILE, 2, 128)), 2, 1e-10, 448.0)
        sb = _compute_scale(xb, 1, 1e-10, 448.0)
        sc = _compute_scale(xc, 1, 1e-10, 448.0)
        if TMA_ALIGNED_SCALES:
            sa = _round_scale_to_power_of_two(sa)
            sb = _round_scale_to_power_of_two(sb)
            sc = _round_scale_to_power_of_two(sc)
        if TMA_ALIGNED_SCALES:
            inv_sa = _inverse_rounded_scale(sa)
            inv_sb = _inverse_rounded_scale(sb)
            inv_sc = _inverse_rounded_scale(sc)
            va = xa * tl.reshape(
                tl.broadcast_to(inv_sa[:, :, None], (HEAD_TILE, 2, 128)),
                (HEAD_TILE, 256),
            )
            vb = xb * inv_sb[:, None]
            vc = xc * inv_sc[:, None]
        else:
            va = xa / tl.reshape(
                tl.broadcast_to(sa[:, :, None], (HEAD_TILE, 2, 128)), (HEAD_TILE, 256)
            )
            vb = xb / sb[:, None]
            vc = xc / sc[:, None]
        va = tl.clamp(va, -448.0, 448.0)
        vb = tl.clamp(vb, -448.0, 448.0)
        vc = tl.clamp(vc, -448.0, 448.0)
        qa = va.to(fp8_ptr.dtype.element_ty).to(tl.uint8, bitcast=True)
        qb = vb.to(fp8_ptr.dtype.element_ty).to(tl.uint8, bitcast=True)
        qc = vc.to(fp8_ptr.dtype.element_ty).to(tl.uint8, bitcast=True)
        a0, a1 = a1, qa
        b0, b1 = b1, qb
        c0, c1 = c1, qc
        sa0, sa1 = sa1, sa
        sb0, sb1 = sb1, sb
        sc0, sc1 = sc1, sc
    # Keep stores after both independent head tiles so they do not block
    # load motion across the unrolled compute window.
    for tile in tl.static_range(2):
        qa, qb, qc = (a0, a1)[tile], (b0, b1)[tile], (c0, c1)[tile]
        sa, sb, sc = (sa0, sa1)[tile], (sb0, sb1)[tile], (sc0, sc1)[tile]
        h = head_base + tile * HEAD_TILE + heads
        group = h // heads_per_group
        within = h % heads_per_group
        out = (
            fp8_ptr
            + group[:, None] * fp8_stride_group
            + token * fp8_stride_token
            + within[:, None] * 512
        )
        tl.store(out + a[None, :], qa.to(fp8_ptr.dtype.element_ty, bitcast=True))
        tl.store(out + 256 + b[None, :], qb.to(fp8_ptr.dtype.element_ty, bitcast=True))
        tl.store(out + 384 + b[None, :], qc.to(fp8_ptr.dtype.element_ty, bitcast=True))
        if TMA_ALIGNED_SCALES:
            aa = (sa.to(tl.uint32, bitcast=True) >> 23) & 0xFF
            bb = (sb.to(tl.uint32, bitcast=True) >> 23) & 0xFF
            cc = (sc.to(tl.uint32, bitcast=True) >> 23) & 0xFF
            packed = (
                tl.sum(aa << (scale_idx[None, :] * 8), axis=1) | (bb << 16) | (cc << 24)
            )
            tl.store(
                scale_ptr
                + group * scale_stride_group
                + token
                + within * scale_stride_k,
                packed.to(tl.int32),
            )
        else:
            base_s = (
                scale_ptr
                + group * scale_stride_group
                + token
                + within * 4 * scale_stride_k
            )
            tl.store(base_s[:, None] + scale_idx[None, :] * scale_stride_k, sa)
            tl.store(base_s + 2 * scale_stride_k, sb)
            tl.store(base_s + 3 * scale_stride_k, sc)


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    tma_aligned_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton draft of DeepSeek-V4 fused inverse-RoPE + FP8 group quant.

    Args:
        o: [num_tokens, num_heads, head_dim]
        positions: [num_tokens]
        cos_sin_cache: [max_position, rope_dim] laid out as cos || sin

    Returns:
        o_fp8: [num_tokens, n_groups, heads_per_group * head_dim]
        o_scale: [num_tokens, n_groups, num_scale_blocks] or packed UE8M0 view
    """
    logger.debug("GEMS FUSED INV ROPE FP8 QUANT")

    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert fp8_dtype == torch.float8_e4m3fn, "only torch.float8_e4m3fn is supported"
    assert o.ndim == 3, "`o` must be [num_tokens, num_heads, head_dim]"
    assert positions.ndim == 1, "`positions` must be 1D"
    assert cos_sin_cache.ndim == 2, "`cos_sin_cache` must be 2D"
    assert o.stride(-1) == 1, "head_dim must be contiguous"
    assert positions.shape[0] == o.shape[0], "positions and o token count mismatch"

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    chunks_per_head = head_dim // quant_group_size
    if tma_aligned_scales:
        assert (
            chunks_per_head <= 4
        ), "packed UE8M0 path currently expects at most 4 scale blocks per head"

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    tma_aligned_t = _get_tma_aligned_size(num_tokens, 4)

    if tma_aligned_scales:
        scale_inner = (num_scale_blocks + 3) // 4
        scale_dtype = torch.int32
    else:
        scale_inner = num_scale_blocks
        scale_dtype = torch.float32

    finfo = torch.finfo(fp8_dtype)
    fp8_q = torch.empty((n_groups, num_tokens, d), dtype=fp8_dtype, device=o.device)
    scale = torch.empty(
        n_groups * scale_inner * tma_aligned_t,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )

    # Two independent head tiles expose a load window to the optional pass.
    # Keep the incumbent kernel for other geometries and unsupported compilers.
    if (
        _HAS_LOAD_REORDER
        and o.dtype == torch.bfloat16
        and o.is_contiguous()
        and positions.is_contiguous()
        and cos_sin_cache.is_contiguous()
        and (nope_dim, rope_dim, quant_group_size) == (448, 64, 128)
        and eps == 1e-10
        and num_heads % 8 == 0
        and num_tokens >= 256
        and num_tokens % 4 == 0
    ):
        head_tile = 4
        _fused_inv_rope_fp8_quant_window[(num_tokens, num_heads // (2 * head_tile))](
            o,
            positions,
            cos_sin_cache,
            fp8_q,
            scale,
            head_tile,
            tma_aligned_scales,
            True,
            o.stride(0),
            o.stride(1),
            cos_sin_cache.stride(0),
            fp8_q.stride(0),
            fp8_q.stride(1),
            scale.stride(0),
            scale.stride(2),
            heads_per_group,
            num_warps=2,
            num_stages=1,
        )
    else:
        grid = (tma_aligned_t, n_groups * heads_per_group)
        _fused_inv_rope_fp8_quant_per_head[grid](
            o,
            positions,
            cos_sin_cache,
            fp8_q,
            scale,
            num_tokens,
            heads_per_group=heads_per_group,
            o_stride_token=o.stride(0),
            o_stride_head=o.stride(1),
            cache_stride_pos=cos_sin_cache.stride(0),
            fp8_stride_group=fp8_q.stride(0),
            fp8_stride_token=fp8_q.stride(1),
            scale_stride_group=scale.stride(0),
            scale_stride_k=scale.stride(2),
            fp8_max=finfo.max,
            eps=eps,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=nope_dim % quant_group_size,
            HALF_ROPE=rope_dim // 2,
            TMA_ALIGNED_SCALES=tma_aligned_scales,
            num_warps=1,
            num_stages=1,
        )

    return fp8_q.transpose(0, 1), scale.transpose(0, 1)

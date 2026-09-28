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

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _get_tma_aligned_size(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


@triton.jit
def _float_to_e4m3fn_bits(x):
    """Convert f32 values to e4m3fn bits (0..255).

    FlagTree on PPU has no working native fp8e4m3fn conversion, so the byte
    is built with integer ops directly from the f32 bit pattern. Rounding is
    round-to-nearest-even with saturation above 448, matching torch's
    float8_e4m3fn cast bit-for-bit (single rounding, no f16 intermediate;
    verified by an exhaustive probe over all 65536 f16 patterns and a wide
    f32 sweep). NaN inputs are not handled: production values are finite.

    The |x| <= 448 clamp of the generic kernel is folded into the saturating
    conversion: any magnitude above the fp8 maximum maps to 0x7E, so no
    explicit clamp is needed before calling this.
    """
    xb = x.to(tl.int32, bitcast=True)
    s = (xb >> 24) & 0x80
    a = xb & 0x7FFFFFFF
    # fp8-normal region (|x| >= 2^-6): rebased field with RNE; mantissa
    # overflow carries into the exponent, saturate at 0x7E.
    c = a - 0x3C000000
    q_norm = (c + 0x7FFFF + ((c >> 20) & 1)) >> 20
    q_norm = tl.minimum(q_norm, 0x7E)
    # fp8-subnormal region (|x| < 2^-6): values are multiples of 2^-9; a
    # magic-add (1.5 * 2^23) performs the RNE-to-integer rounding in one f32
    # add instead of an integer shift chain.
    aj = a.to(tl.float32, bitcast=True)
    q_sub = (aj * 512.0 + 12582912.0).to(tl.int32, bitcast=True) & 0xFF
    q = tl.where(a >= 0x3C800000, q_norm, q_sub)
    return s | (q & 0x7F)


@triton.jit
def _pow2_ue8m0_exponent(absmax):
    """k = ceil(log2(absmax / fp8_max)) with fp8_max = 448, from f32 bits.

    absmax is guaranteed > 0 (clamped to eps) and normal. log2(448) splits as
    8 + log2(1.75), so k = e - 8 + (mantissa > 0.75 * 2^23). Integer ops only
    (no log2/ceil/exp2): scale = 2^k and 2^-k are then built directly from
    exponent bits.
    """
    abits = absmax.to(tl.int32, bitcast=True)
    e_b = (abits >> 23) - 127
    mant = abits & 0x7FFFFF
    return e_b - 8 + (mant > 0x600000).to(tl.int32)


@triton.jit
def _fused_inv_rope_fp8_quant_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_u8_ptr,
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
    BLOCK_T: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    # Each program handles BLOCK_T tokens for one (group, head) pair. Chunk
    # (quant group) tiles are processed separately: the generic kernel runs
    # the rope math over the full head-dim tile even though only the last
    # QUANT_GROUP_SIZE - ROPE_START lanes rotate, which wastes 8x lane work;
    # the rope rotation here runs on a narrow (BLOCK_T, ROPE_WIDTH) tile.
    # Head axis first: consecutive programs cover consecutive heads and thus
    # consecutive 512-element rows of the same token, which keeps L2-locality
    # for both the o reads and the fp8 writes (measured ~2-8% faster than
    # the token-block-first order across all sweep shapes).
    pid_gh = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1).to(tl.int64)

    o_stride_token = o_stride_token.to(tl.int64)
    o_stride_head = o_stride_head.to(tl.int64)
    cache_stride_pos = cache_stride_pos.to(tl.int64)
    fp8_stride_group = fp8_stride_group.to(tl.int64)
    fp8_stride_token = fp8_stride_token.to(tl.int64)
    scale_stride_group = scale_stride_group.to(tl.int64)
    scale_stride_k = scale_stride_k.to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh

    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rows_valid = rows < num_tokens
    mask2d = rows_valid[:, None]
    input_base = o_ptr + global_head * o_stride_head
    row_off = rows[:, None] * o_stride_token
    chunk_cols = tl.arange(0, QUANT_GROUP_SIZE)
    fp8_base = (
        fp8_u8_ptr
        + g * fp8_stride_group
        + rows[:, None] * fp8_stride_token
        + head_in_group * CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    )
    scale_base = scale_ptr + g * scale_stride_group + rows

    # Whole chunks without rope.
    packed: tl.tensor = tl.zeros((BLOCK_T,), dtype=tl.int32)
    for ci in tl.static_range(CHUNKS_PER_HEAD - 1):
        xi = tl.load(
            input_base + row_off + ci * QUANT_GROUP_SIZE + chunk_cols[None, :],
            mask=mask2d,
            other=0.0,
        ).to(tl.float32)
        absmax_i = tl.maximum(tl.max(tl.abs(xi), axis=1), eps)
        if TMA_ALIGNED_SCALES:
            ki = _pow2_ue8m0_exponent(tl.maximum(absmax_i, eps * fp8_max))
            rscale_i = ((127 - ki) << 23).to(tl.float32, bitcast=True)
            scale_bits_i = (127 + ki) << 23
            byte_i = (scale_bits_i >> 23) & 0xFF
            byte_i = tl.where(rows_valid, byte_i, 0)
            packed |= byte_i << (ci * 8)
        else:
            scale_i = absmax_i * (1.0 / fp8_max)
            rscale_i = fp8_max / absmax_i
            tl.store(
                scale_base + (head_in_group * CHUNKS_PER_HEAD + ci) * scale_stride_k,
                tl.where(rows_valid, scale_i, 0.0),
            )
        qi = _float_to_e4m3fn_bits(xi * rscale_i[:, None]).to(tl.uint8)
        tl.store(
            fp8_base + ci * QUANT_GROUP_SIZE + chunk_cols[None, :],
            qi,
            mask=mask2d,
        )

    # Last chunk: first ROPE_START columns are pass-through, the trailing
    # ROPE_WIDTH columns are inverse-rotated (on a narrow tile).
    ROPE_WIDTH: tl.constexpr = QUANT_GROUP_SIZE - ROPE_START
    last_base = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE
    lo_cols = tl.arange(0, ROPE_START)
    hi_cols = tl.arange(0, ROPE_WIDTH)
    lo = tl.load(
        input_base + row_off + last_base + lo_cols[None, :], mask=mask2d, other=0.0
    ).to(tl.float32)
    pos = tl.load(positions_ptr + rows, mask=rows_valid, other=0)
    hi = tl.load(
        input_base + row_off + last_base + ROPE_START + hi_cols[None, :],
        mask=mask2d,
        other=0.0,
    ).to(tl.float32)
    # ROPE_START is even, so (col ^ 1) stays inside the rope tile.
    partner = tl.load(
        input_base + row_off + last_base + ROPE_START + (hi_cols ^ 1)[None, :],
        mask=mask2d,
        other=0.0,
    ).to(tl.float32)
    cs_idx = hi_cols >> 1
    cache_off = pos[:, None] * cache_stride_pos + cs_idx[None, :]
    cos_v = tl.load(cos_sin_cache_ptr + cache_off, mask=mask2d, other=1.0)
    sin_v = tl.load(cos_sin_cache_ptr + HALF_ROPE + cache_off, mask=mask2d, other=0.0)
    x_add = hi * cos_v + partner * sin_v
    x_sub = hi * cos_v - partner * sin_v
    hi_rot = tl.where(((hi_cols & 1) == 0)[None, :], x_add, x_sub)

    absmax_last = tl.maximum(
        tl.maximum(tl.max(tl.abs(lo), axis=1), tl.max(tl.abs(hi_rot), axis=1)), eps
    )
    if TMA_ALIGNED_SCALES:
        k_last = _pow2_ue8m0_exponent(tl.maximum(absmax_last, eps * fp8_max))
        rscale_last = ((127 - k_last) << 23).to(tl.float32, bitcast=True)
        scale_bits_last = (127 + k_last) << 23
        byte_last = (scale_bits_last >> 23) & 0xFF
        byte_last = tl.where(rows_valid, byte_last, 0)
        packed |= byte_last << ((CHUNKS_PER_HEAD - 1) * 8)
        tl.store(scale_base + head_in_group * scale_stride_k, packed)
    else:
        scale_last = absmax_last * (1.0 / fp8_max)
        rscale_last = fp8_max / absmax_last
        tl.store(
            scale_base
            + (head_in_group * CHUNKS_PER_HEAD + CHUNKS_PER_HEAD - 1) * scale_stride_k,
            tl.where(rows_valid, scale_last, 0.0),
        )

    tl.store(
        fp8_base + last_base + lo_cols[None, :],
        _float_to_e4m3fn_bits(lo * rscale_last[:, None]).to(tl.uint8),
        mask=mask2d,
    )
    tl.store(
        fp8_base + last_base + ROPE_START + hi_cols[None, :],
        _float_to_e4m3fn_bits(hi_rot * rscale_last[:, None]).to(tl.uint8),
        mask=mask2d,
    )


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
    DeepSeek-V4 fused inverse-RoPE + FP8 group quant (PPU/thead backend).

    Identical contract to the generic implementation; only the FP8 E4M3
    conversion is done manually in integer ops because FlagTree on PPU has no
    working native fp8e4m3fn conversion.

    Args:
        o: [num_tokens, num_heads, head_dim]
        positions: [num_tokens]
        cos_sin_cache: [max_position, rope_dim] laid out as cos || sin

    Returns:
        o_fp8: [num_tokens, n_groups, heads_per_group * head_dim]
        o_scale: [num_tokens, n_groups, num_scale_blocks] or packed UE8M0 view
    """
    logger.debug("GEMS THEAD FUSED INV ROPE FP8 QUANT")

    fp8_dtype = torch.float8_e4m3fn if dtype is None else dtype
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

    rope_start = nope_dim % quant_group_size
    rope_width = quant_group_size - rope_start
    # The split-tile kernel needs power-of-two tile widths for the last chunk.
    if (
        chunks_per_head < 2
        or rope_start & (rope_start - 1) != 0
        or rope_width & (rope_width - 1) != 0
    ):
        raise NotImplementedError(
            "thead fused_inv_rope_fp8_quant requires rope region boundaries to "
            "be powers of two inside the last quant chunk"
        )

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
    # FlagTree cannot take a float8_e4m3fn pointer; store through a uint8 view.
    fp8_q_u8 = fp8_q.view(torch.uint8)
    scale = torch.empty(
        n_groups * scale_inner * tma_aligned_t,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )

    # Static dispatch: single token per program is launch-bound for small
    # batches; BLOCK_T=2 wins once enough programs fill the device.
    block_t = 1 if tma_aligned_t < 768 else 2
    grid = (n_groups * heads_per_group, triton.cdiv(tma_aligned_t, block_t))
    _fused_inv_rope_fp8_quant_kernel[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_q_u8,
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
        ROPE_START=rope_start,
        HALF_ROPE=rope_dim // 2,
        BLOCK_T=block_t,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        num_warps=1,
        num_stages=1,
    )

    return fp8_q.transpose(0, 1), scale.transpose(0, 1)

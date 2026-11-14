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

"""MetaX ``deep_gemm_fp8_o_proj``: inverse RoPE + FP8 quant + block FP8 einsum.

The generic implementation cannot run here: it composes ``fused_inv_rope_fp8_quant``
with ``fp8_einsum``, and ``fp8_einsum`` only accepts NVIDIA Hopper / MThreads.

MetaX (MACA 3.8, FlagTree 3.6) has no FP8 ``tl.dot`` lowering -- FP8 tiles fail in
``ConvertTritonGPUToLLVM`` -- and the FP8-to-BF16 cast is a software sequence that
drops the same GEMM from 130 to 11-18 TFLOPS. INT8 ``tl.dot`` with INT32
accumulation, on the other hand, is a first-class path here (it backs
``w8a8_block_int8_bmm``), runs at ~2x the BF16 rate, and halves the weight bytes.

So this backend keeps the operator's FP8 block-wise semantics but transports the
data as INT8 fixed point:

1. The activation kernel rounds on the E4M3 grid (four integer ops on the FP32
   bits, so the FP8 contract is preserved), then rescales those grid values onto
   INT8 (``x * 448/127``) and stores them with ``amax / 127`` block scales.
2. ``wo_a`` is converted once per weight tensor to INT8 (``round(w_fp8 * 127/448)``)
   with the same ``amax / 127`` scales and cached; it is a model parameter, and
   per-call conversion would cost more than the GEMM.
3. ``w8a8_block_int8_bmm`` runs the grouped, block-scaled product on INT8 tensor
   cores and accumulates in FP32.

Re-quantizing the exact FP8 values onto the INT8 grid adds at most half an INT8
step (<= 0.4% of the block amax), which is far below the FP8 grid's own error and
measures as ~1.2% peak-relative against the exact FP8 reference on a C550.

vLLM >= 0.32 may pass a ``QuantizedActivation`` (attention already rotated and
cast); its FP8 payload is converted to INT8 the same way, skipping RoPE.
"""

import logging
import weakref
from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.deep_gemm_fp8_o_proj import (
    _SUPPORTED_RECIPE,
    _is_quantized_activation,
    _weight_and_scale,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._metax.ops.w8a8_block_int8_bmm import (
    w8a8_block_int8_bmm,
)

logger = logging.getLogger(__name__)

_QUANT_BLOCK = 128
_FP8_MAX = 448.0
_INT8_MAX = 127.0
_INT8_TO_FP8 = _INT8_MAX / _FP8_MAX
_FP8_TO_INT8 = _FP8_MAX / _INT8_MAX
_EPS = 1e-10


@triton.jit
def _round_to_e4m3_grid(v):
    """Round FP32 magnitudes onto the E4M3 grid (RNE, 3 mantissa bits).

    Adding the RNE bias to the FP32 bit pattern and masking the low 20 mantissa
    bits is equivalent to a hardware FP8 cast for normal values, and costs a few
    integer ops instead of the software ``.to(tl.float8e4nv)`` sequence.
    """
    bits = v.to(tl.int32, bitcast=True)
    bits = bits + 0x0007FFFF + ((bits >> 20) & 1)
    # ~0xFFFFF as a signed literal; keeps the top three mantissa bits.
    bits = bits & -1048576
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _inv_rope_quant_int8_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    s_stride_group,
    s_stride_token,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    NBLOCKS: tl.constexpr,
    HEADS_PER_CTA: tl.constexpr,
):
    """``HEADS_PER_CTA`` (token, head) pairs per CTA: inverse RoPE + INT8 quant.

    One head per CTA gives each 1-warp CTA a 1 KiB load and a 512 B store, so at
    prefill sizes the pass measures CTA throughput rather than bandwidth. Taking
    several heads of the same token per CTA keeps the tile one coalesced
    row-major load and raises the work per CTA by ``HEADS_PER_CTA``.
    """
    pid_token = tl.program_id(0).to(tl.int64)
    pid_head = tl.program_id(1).to(tl.int64)
    if pid_token >= num_tokens:
        return

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    head0 = pid_head * HEADS_PER_CTA
    group = head0 // heads_per_group
    head_in_group0 = head0 % heads_per_group

    rows = tl.arange(0, HEADS_PER_CTA)
    cols = tl.arange(0, HEAD_DIM)
    row_off = (
        o_ptr
        + pid_token * o_stride_token
        + head0 * o_stride_head
        + rows[:, None] * o_stride_head
    )
    x = tl.load(row_off + cols[None, :]).to(tl.float32)

    # --- compact inverse RoPE on the trailing rope segment ---
    # The 64-wide segment is loaded again rather than sliced out of ``x``: a
    # reduction over the block axis to recover it costs more than the 1/8 extra
    # read, which is served from L2 anyway.
    rope_abs_start: tl.constexpr = HEAD_DIM - ROPE_DIM
    rope_cols = tl.arange(0, ROPE_DIM)
    xr = tl.load(row_off + rope_abs_start + rope_cols[None, :]).to(tl.float32)
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    j = tl.arange(0, HALF_ROPE)
    cos_v = tl.load(cache_base + j)
    sin_v = tl.load(cache_base + HALF_ROPE + j)

    # Pairs are interleaved (even, odd); the partner of a lane is the other lane
    # of its pair, recovered as pair_sum - x without a second load.
    pairs = tl.reshape(xr, (HEADS_PER_CTA, HALF_ROPE, 2))
    pair_sum = tl.sum(pairs, axis=2)
    partner = pair_sum[:, :, None] - pairs
    sgn = tl.where(tl.arange(0, 2)[None, None, :] == 0, 1.0, -1.0)
    rope_out = pairs * cos_v[None, :, None] + partner * sin_v[None, :, None] * sgn
    rope_out = tl.reshape(rope_out, (HEADS_PER_CTA, ROPE_DIM))
    # splice: [H, HEAD_DIM] -> [H, NBLOCKS, ROPE_DIM]; rope is the last block
    xb = tl.reshape(x, (HEADS_PER_CTA, NBLOCKS, ROPE_DIM))
    xb = tl.where(
        tl.arange(0, NBLOCKS)[None, :, None] == NBLOCKS - 1,
        rope_out[:, None, :],
        xb,
    )
    x = tl.reshape(xb, (HEADS_PER_CTA, HEAD_DIM))

    # --- per-128 block quant: FP8 grid, then rescaled onto INT8 ---
    nblocks: tl.constexpr = HEADS_PER_CTA * CHUNKS_PER_HEAD
    x_2d = tl.reshape(tl.abs(x), (nblocks, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    recip = fp8_max / block_absmax
    recip_exp = tl.reshape(
        tl.broadcast_to(recip[:, None], (nblocks, QUANT_GROUP_SIZE)),
        (HEADS_PER_CTA, HEAD_DIM),
    )
    # 448/127 inlined: Triton kernels see plain literals as constexpr.
    y = _round_to_e4m3_grid(x * recip_exp) * 0.28348214285714285
    q_int8 = tl.where(y >= 0, y + 0.5, y - 0.5).to(tl.int32).to(tl.int8)

    out_off = (
        out_ptr
        + group * out_stride_group
        + pid_token * out_stride_token
        + (head_in_group0 + rows[:, None]) * HEAD_DIM
        + cols[None, :]
    )
    tl.store(out_off, q_int8)

    scale_off = (
        scale_ptr
        + group * s_stride_group
        + pid_token * s_stride_token
        + (head_in_group0 + rows[:, None]) * CHUNKS_PER_HEAD
        + tl.arange(0, CHUNKS_PER_HEAD)[None, :]
    )
    tl.store(
        scale_off,
        tl.reshape(block_absmax, (HEADS_PER_CTA, CHUNKS_PER_HEAD))
        * 0.007874015748031496,  # 1/127
    )


def _compact_rope_supported(head_dim: int, rope_dim: int) -> bool:
    if head_dim & (head_dim - 1) != 0:
        return False
    if rope_dim & (rope_dim - 1) != 0:
        return False
    return head_dim % rope_dim == 0


def _heads_per_cta(heads_per_group: int) -> int:
    """Largest head grouping that still splits a group evenly (CTA must not straddle)."""
    for candidate in (4, 2):
        if heads_per_group % candidate == 0:
            return candidate
    return 1


def _launch_quant_int8(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out: torch.Tensor,
    scale: torch.Tensor,
    g0: int,
    g1: int,
    heads_per_group: int,
    rope_dim: int,
) -> None:
    """Quantize groups ``[g0, g1)`` of ``o`` into the preallocated buffers."""
    num_tokens, _, head_dim = o.shape
    heads_per_cta = _heads_per_cta(heads_per_group)
    chunk_out = out[g0:g1]
    chunk_scale = scale[g0:g1]
    grid = (num_tokens, (g1 - g0) * heads_per_group // heads_per_cta)
    _inv_rope_quant_int8_kernel[grid](
        o[:, g0 * heads_per_group : g1 * heads_per_group, :],
        positions,
        cos_sin_cache,
        chunk_out,
        chunk_scale,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        out_stride_group=chunk_out.stride(0),
        out_stride_token=chunk_out.stride(1),
        s_stride_group=chunk_scale.stride(0),
        s_stride_token=chunk_scale.stride(1),
        fp8_max=_FP8_MAX,
        eps=_EPS,
        QUANT_GROUP_SIZE=_QUANT_BLOCK,
        CHUNKS_PER_HEAD=head_dim // _QUANT_BLOCK,
        ROPE_DIM=rope_dim,
        HALF_ROPE=rope_dim // 2,
        NBLOCKS=head_dim // rope_dim,
        HEADS_PER_CTA=heads_per_cta,
        num_warps=heads_per_cta,
        num_stages=1,
    )


def _quant_buffers(
    o: torch.Tensor, n_groups: int, heads_per_group: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the ``[G, T, K]`` INT8 activation and its ``[G, T, K/128]`` scales."""
    num_tokens, _, head_dim = o.shape
    group_dim = heads_per_group * head_dim
    out = torch.empty(
        (n_groups, num_tokens, group_dim), dtype=torch.int8, device=o.device
    )
    scale = torch.empty(
        (n_groups, num_tokens, group_dim // _QUANT_BLOCK),
        dtype=torch.float32,
        device=o.device,
    )
    return out, scale


# ---------------------------------------------------------------------------
# INT8 weight cache
# ---------------------------------------------------------------------------

_MODULE_CACHE = weakref.WeakKeyDictionary()
_TENSOR_CACHE = OrderedDict()
_TENSOR_CACHE_BYTES = 0
_TENSOR_CACHE_LIMIT = 512 << 20


def _quantize_weight_int8(
    weight: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rescale an FP8 block-wise ``wo_a`` onto INT8 with ``amax / 127`` scales."""
    groups, columns, reduction = weight.shape
    out = torch.empty(
        (groups, columns, reduction), dtype=torch.int8, device=weight.device
    )
    for group in range(groups):
        out[group].copy_(
            torch.round(weight[group].float() * _INT8_TO_FP8).to(torch.int8)
        )
    return out, (scale * _FP8_TO_INT8).contiguous()


def _cache_key(weight: torch.Tensor, scale: torch.Tensor):
    return (
        weight.data_ptr(),
        weight._version,
        tuple(weight.shape),
        scale.data_ptr(),
        scale._version,
        tuple(scale.shape),
    )


def _cached_weight(owner, weight: torch.Tensor, scale: torch.Tensor):
    """Return the cached INT8 ``(weight, scale)``; weights are model parameters."""
    global _TENSOR_CACHE_BYTES
    key = _cache_key(weight, scale)

    if isinstance(owner, torch.nn.Module):
        entry = _MODULE_CACHE.get(owner)
        if entry is not None and entry[0] == key:
            return entry[1]
        quantized = _quantize_weight_int8(weight, scale)
        try:
            _MODULE_CACHE[owner] = (key, quantized)
        except TypeError:  # pragma: no cover - module without weakref support
            pass
        return quantized

    entry = _TENSOR_CACHE.get(key)
    if entry is not None:
        _TENSOR_CACHE.move_to_end(key)
        return entry

    quantized = _quantize_weight_int8(weight, scale)
    nbytes = quantized[0].numel() + quantized[1].numel() * 4
    while _TENSOR_CACHE and _TENSOR_CACHE_BYTES + nbytes > _TENSOR_CACHE_LIMIT:
        _, evicted = _TENSOR_CACHE.popitem(last=False)
        _TENSOR_CACHE_BYTES -= evicted[0].numel() + evicted[1].numel() * 4
    _TENSOR_CACHE[key] = quantized
    _TENSOR_CACHE_BYTES += nbytes
    return quantized


# ---------------------------------------------------------------------------
# Pre-quantized activation (vLLM >= 0.32): FP8 payload onto INT8
# ---------------------------------------------------------------------------


@triton.jit
def _e4m3_bits_to_f32(bits):
    """Decode E4M3 bytes into the exact FP32 value without a hardware FP8 cast."""
    b = bits.to(tl.int32) & 0xFF
    sign = (b & 0x80) << 8
    exp = (b >> 3) & 0xF
    man = b & 0x7
    packed = sign | ((exp + 120) << 7) | (man << 4)
    normal = (packed << 16).to(tl.float32, bitcast=True)
    sub = man.to(tl.float32) * 0.001953125  # mantissa * 2^-9
    sub = tl.where(sign != 0, -sub, sub)
    return tl.where(exp == 0, sub, normal)


@triton.jit
def _activation_to_int8_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    out_scale_ptr,
    num_tokens,
    K: tl.constexpr,
    x_stride_t,
    x_stride_g,
    s_stride_t,
    s_stride_g,
    o_stride_g,
    o_stride_t,
    os_stride_g,
    os_stride_t,
    QUANT_BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    if token >= num_tokens:
        return
    offsets = tl.arange(0, K)
    values = _e4m3_bits_to_f32(
        tl.load(x_ptr + token * x_stride_t + group * x_stride_g + offsets)
    )
    y = values * 0.28348214285714285  # 448/127
    q_int8 = tl.where(y >= 0, y + 0.5, y - 0.5).to(tl.int32).to(tl.int8)
    tl.store(out_ptr + group * o_stride_g + token * o_stride_t + offsets, q_int8)

    blocks = tl.arange(0, K // QUANT_BLOCK)
    scales = tl.load(scale_ptr + token * s_stride_t + group * s_stride_g + blocks)
    tl.store(
        out_scale_ptr + group * os_stride_g + token * os_stride_t + blocks,
        scales * 3.5275590551181102,  # 448/127: fp8 scale -> int8 scale
    )


def _activation_to_int8(x: torch.Tensor, x_scale: torch.Tensor):
    """Return a ``[G, T, K]`` INT8 activation and its ``[G, T, K/128]`` scales."""
    num_tokens, n_groups, reduction = x.shape
    out = torch.empty(
        (n_groups, num_tokens, reduction), dtype=torch.int8, device=x.device
    )
    out_scale = torch.empty(
        (n_groups, num_tokens, reduction // _QUANT_BLOCK),
        dtype=torch.float32,
        device=x.device,
    )
    _activation_to_int8_kernel[(num_tokens, n_groups)](
        x.view(torch.int8),
        x_scale,
        out,
        out_scale,
        num_tokens,
        reduction,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        out.stride(0),
        out.stride(1),
        out_scale.stride(0),
        out_scale.stride(1),
        QUANT_BLOCK=_QUANT_BLOCK,
        num_warps=4,
        num_stages=1,
    )
    return out, out_scale


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _weight_supported(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_groups: int,
    o_lora_rank: int,
    reduction: int,
) -> bool:
    if weight.shape != (n_groups, o_lora_rank, reduction):
        return False
    if reduction % _QUANT_BLOCK or o_lora_rank % _QUANT_BLOCK:
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.stride(-1) != 1:
        return False
    if weight_scale.dtype != torch.float32:
        return False
    return weight_scale.shape == (
        n_groups,
        o_lora_rank // _QUANT_BLOCK,
        reduction // _QUANT_BLOCK,
    )


def deep_gemm_fp8_o_proj(
    o,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    wo_b: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: Optional[Sequence[int]] = _SUPPORTED_RECIPE,
    tma_aligned_scales: bool = False,
) -> torch.Tensor:
    """DeepSeek-V4 output projection on MetaX (INT8 tensor-core transport)."""
    logger.debug("GEMS DEEP GEMM FP8 O PROJ (METAX)")

    if tuple(einsum_recipe) != _SUPPORTED_RECIPE or tma_aligned_scales:
        raise NotImplementedError(
            "the MetaX implementation only supports the Hopper FP32 block-scale "
            f"recipe (1, 128, 128) with 128-wide K blocks, got "
            f"einsum_recipe={tuple(einsum_recipe)}, tma_aligned_scales={tma_aligned_scales}"
        )
    assert o_lora_rank % _QUANT_BLOCK == 0, "o_lora_rank must be a multiple of 128"

    head_dim = nope_dim + rope_dim
    reduction = heads_per_group * head_dim
    weight, weight_scale = _weight_and_scale(wo_a)
    prequantized = _is_quantized_activation(o)

    if prequantized:
        act, act_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
        assert act.ndim == 3, "`o.data` must be [num_tokens, n_groups, k]"
        assert act.dtype == torch.float8_e4m3fn, "`o.data` must be float8_e4m3fn"
        assert act_scale.dtype == torch.float32, "`o.scale` must be float32"
        num_tokens = act.shape[0]
        device = act.device
        activation_ok = act.shape[1:] == (n_groups, reduction) and act.stride(-1) == 1
        activation_ok = activation_ok and act_scale.shape == (
            num_tokens,
            n_groups,
            reduction // _QUANT_BLOCK,
        )
    else:
        assert o.ndim == 3, "`o` must be [num_tokens, num_heads, head_dim]"
        assert positions.ndim == 1 and positions.shape[0] == o.shape[0]
        assert cos_sin_cache.ndim == 2, "`cos_sin_cache` must be 2D"
        assert o.dtype == torch.bfloat16, "`o` must be bfloat16"
        assert (
            o.shape[1] == n_groups * heads_per_group
        ), "num_heads must equal n_groups * heads_per_group"
        assert o.shape[-1] == head_dim, "head_dim must equal nope_dim + rope_dim"
        num_tokens = o.shape[0]
        device = o.device
        activation_ok = (
            o.stride(-1) == 1
            and head_dim % _QUANT_BLOCK == 0
            and nope_dim % _QUANT_BLOCK == _QUANT_BLOCK - rope_dim
            and _compact_rope_supported(head_dim, rope_dim)
        )

    if not (
        activation_ok
        and _weight_supported(weight, weight_scale, n_groups, o_lora_rank, reduction)
    ):
        raise NotImplementedError(
            "unsupported shape for the MetaX deep_gemm_fp8_o_proj: expected "
            f"activation [tokens, {n_groups}, {reduction}] (FP8 pre-quantized or "
            "BF16 attention output), 128x128 block scales and an FP8 wo_a"
        )

    owner = wo_a if isinstance(wo_a, torch.nn.Module) else None
    with torch_device_fn.device(device):
        if num_tokens == 0:
            empty = torch.empty(
                (0, n_groups * o_lora_rank), dtype=torch.bfloat16, device=device
            )
            return wo_b(empty)
        if prequantized:
            act_i8, act_scale_i8 = _activation_to_int8(act, act_scale)
        else:
            act_i8, act_scale_i8 = _quant_buffers(o, n_groups, heads_per_group)
        weight_i8, weight_scale_i8 = _cached_weight(owner, weight, weight_scale)
        out = torch.empty(
            (num_tokens, n_groups, o_lora_rank), dtype=torch.bfloat16, device=device
        )
        z = out.permute(1, 0, 2)
        if not prequantized:
            _launch_quant_int8(
                o,
                positions,
                cos_sin_cache,
                act_i8,
                act_scale_i8,
                0,
                n_groups,
                heads_per_group,
                rope_dim,
            )
        w8a8_block_int8_bmm(
            act_i8,
            weight_i8,
            act_scale_i8,
            weight_scale_i8,
            block_size=(_QUANT_BLOCK, _QUANT_BLOCK),
            z=z,
            output_dtype=torch.bfloat16,
        )

    assert callable(wo_b), "`wo_b` must be a callable linear layer"
    return wo_b(out.flatten(1))

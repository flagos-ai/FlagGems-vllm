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

"""PPU ``deep_gemm_fp8_o_proj``: inverse RoPE + FP8-grid quant + block FP8 einsum.

The generic implementation composes ``fused_inv_rope_fp8_quant`` with
``fp8_einsum``; the latter only accepts NVIDIA Hopper / MThreads, and this
backend has no FP8 ``tl.dot`` lowering either (``fp8e4nv`` is rejected at
compile time). INT8 ``tl.dot`` with INT32 accumulation is the tuned tensor-core
path here (it backs ``w8a8_block_int8_bmm``), so this backend keeps the
operator's FP8 block-wise semantics but transports the values as INT8:

1. The activation kernel rotates the trailing RoPE columns and rounds on the
   E4M3 grid (integer ops on the FP32 bits, preserving the FP8 contract), then
   rescales those grid values onto INT8 (``* 127/448``) with ``amax / 127``
   block scales.
2. ``wo_a`` is converted once per weight tensor to INT8
   (``round(w_fp8 * 127/448)``) with the same ``amax / 127`` scales and cached;
   it is a model parameter, so per-call conversion would cost more than the
   GEMM itself.
3. ``w8a8_block_int8_bmm`` runs the grouped, block-scaled product on INT8 tensor
   cores with FP32 accumulation.

Re-quantizing the exact FP8 values onto the INT8 grid adds at most half an INT8
step (<= 0.4% of the block amax), far below the FP8 grid's own error.

vLLM may pass a ``QuantizedActivation`` (attention already rotated and cast);
its FP8 payload is converted to INT8 the same way, skipping RoPE.
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
from flaggems_vllm.runtime.backend._thead.ops.w8a8_block_int8_bmm import (
    _int8_block_bmm_kernel,
    w8a8_block_int8_bmm,
)

logger = logging.getLogger(__name__)

_QUANT_BLOCK = 128
_FP8_MAX = 448.0
_INT8_MAX = 127.0
_EPS = 1e-10
# Decode GEMM settings (see `_decode_int8_kernel` for the measurements).
_DECODE_MAX_T = 16
_DECODE_BLOCK_N = 128
_DECODE_K_BLOCKS = 2
_DECODE_WARPS = 4
# G=16 decode is weight-read bound with only 128 CTAs (2 waves); splitting K
# gives 512 CTAs and measured 80.6 -> 77.4 us at T=1 (neutral at T>=16).
_DECODE_SPLIT = 4
_DECODE_SPLIT_BN = 64
_DECODE_SPLIT_MAX_T = 8


@triton.jit
def _round_to_e4m3_grid(v):
    """Round FP32 magnitudes onto the E4M3 grid (RNE, 3 mantissa bits).

    Adding the RNE bias to the FP32 bit pattern and masking the low 20 mantissa
    bits is equivalent to a hardware FP8 cast for normal values, and costs a few
    integer ops instead of the emulated FP8 conversion this backend would
    otherwise emit.
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
    HEADS_PER_CTA: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    scale_stride_group,
    scale_stride_token,
    scale_stride_k,
    fp8_max: tl.constexpr,
    int8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One program covers ``BLOCK_T`` tokens of one (group, head) pair.

    Whole 128-wide chunks are quantized from a compact tile, and only the
    trailing chunk runs the narrow rope rotation (``ROPE_START`` pass-through
    columns plus the rotated remainder). The E4M3 grid values are rescaled onto
    INT8, so the block scale is ``amax / 127``.
    """
    pid_gh = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1).to(tl.int64)

    pairs_per_group: tl.constexpr = heads_per_group // HEADS_PER_CTA
    g = pid_gh // pairs_per_group
    head0 = (pid_gh % pairs_per_group) * HEADS_PER_CTA

    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rows_valid = rows < num_tokens
    mask2d = rows_valid[:, None]
    row_off = rows[:, None] * o_stride_token
    chunk_cols = tl.arange(0, QUANT_GROUP_SIZE)

    # Last chunk: first ROPE_START columns are pass-through, the trailing
    # ROPE_WIDTH columns are inverse-rotated on a narrow tile.
    ROPE_WIDTH: tl.constexpr = QUANT_GROUP_SIZE - ROPE_START
    last_base = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE
    lo_cols = tl.arange(0, ROPE_START)
    hi_cols = tl.arange(0, ROPE_WIDTH)
    # The rope tables depend on the token only, so one load serves every head
    # of the CTA (and each head's 512 columns stay contiguous on the store).
    pos = tl.load(positions_ptr + rows, mask=rows_valid, other=0)
    cs_idx = hi_cols >> 1
    cache_off = pos[:, None] * cache_stride_pos + cs_idx[None, :]
    cos_v = tl.load(cos_sin_cache_ptr + cache_off, mask=mask2d, other=1.0)
    sin_v = tl.load(cos_sin_cache_ptr + HALF_ROPE + cache_off, mask=mask2d, other=0.0)

    for h in tl.static_range(HEADS_PER_CTA):
        head_in_group = head0 + h
        input_base = (
            o_ptr + (g * heads_per_group + head_in_group) * o_stride_head
        )
        out_base = (
            out_ptr
            + g * out_stride_group
            + rows[:, None] * out_stride_token
            + head_in_group * CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
        )
        scale_base = scale_ptr + g * scale_stride_group + rows * scale_stride_token

        # Whole chunks without rope.
        for ci in tl.static_range(CHUNKS_PER_HEAD - 1):
            xi = tl.load(
                input_base + row_off + ci * QUANT_GROUP_SIZE + chunk_cols[None, :],
                mask=mask2d,
                other=0.0,
            ).to(tl.float32)
            absmax_i = tl.maximum(tl.max(tl.abs(xi), axis=1), eps)
            scale_i = absmax_i * (1.0 / int8_max)
            rscale_i = fp8_max / absmax_i
            qi = _round_to_e4m3_grid(xi * rscale_i[:, None]) * (int8_max / fp8_max)
            qi = tl.where(qi >= 0, qi + 0.5, qi - 0.5).to(tl.int32).to(tl.int8)
            tl.store(
                scale_base + (head_in_group * CHUNKS_PER_HEAD + ci) * scale_stride_k,
                scale_i,
                mask=rows_valid,
            )
            tl.store(
                out_base + ci * QUANT_GROUP_SIZE + chunk_cols[None, :],
                qi,
                mask=mask2d,
            )

        lo = tl.load(
            input_base + row_off + last_base + lo_cols[None, :],
            mask=mask2d,
            other=0.0,
        ).to(tl.float32)
        hi = tl.load(
            input_base + row_off + last_base + ROPE_START + hi_cols[None, :],
            mask=mask2d,
            other=0.0,
        ).to(tl.float32)
        # ROPE_START is even, so (col ^ 1) stays inside the rope tile.
        partner = tl.load(
            input_base
            + row_off
            + last_base
            + ROPE_START
            + (hi_cols ^ 1)[None, :],
            mask=mask2d,
            other=0.0,
        ).to(tl.float32)
        x_add = hi * cos_v + partner * sin_v
        x_sub = hi * cos_v - partner * sin_v
        hi_rot = tl.where(((hi_cols & 1) == 0)[None, :], x_add, x_sub)

        absmax_last = tl.maximum(
            tl.maximum(tl.max(tl.abs(lo), axis=1), tl.max(tl.abs(hi_rot), axis=1)),
            eps,
        )
        scale_last = absmax_last * (1.0 / int8_max)
        rscale_last = fp8_max / absmax_last
        tl.store(
            scale_base
            + (head_in_group * CHUNKS_PER_HEAD + CHUNKS_PER_HEAD - 1) * scale_stride_k,
            scale_last,
            mask=rows_valid,
        )

        q_lo = _round_to_e4m3_grid(lo * rscale_last[:, None]) * (int8_max / fp8_max)
        q_lo = tl.where(q_lo >= 0, q_lo + 0.5, q_lo - 0.5).to(tl.int32).to(tl.int8)
        q_hi = _round_to_e4m3_grid(hi_rot * rscale_last[:, None]) * (int8_max / fp8_max)
        q_hi = tl.where(q_hi >= 0, q_hi + 0.5, q_hi - 0.5).to(tl.int32).to(tl.int8)
        tl.store(
            out_base + last_base + lo_cols[None, :],
            q_lo,
            mask=mask2d,
        )
        tl.store(
            out_base + last_base + ROPE_START + hi_cols[None, :],
            q_hi,
            mask=mask2d,
        )


def _compact_rope_supported(head_dim: int, rope_dim: int) -> bool:
    if head_dim & (head_dim - 1) != 0:
        return False
    if rope_dim & (rope_dim - 1) != 0:
        return False
    return head_dim % rope_dim == 0


def _launch_quant_int8(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out: torch.Tensor,
    scale: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    block_t: int,
    heads_per_cta: int = 1,
) -> None:
    """Quantize ``o`` into the preallocated INT8 buffers.

    ``heads_per_cta`` folds several heads of the same group into one program:
    the rope tables are then loaded once per token instead of once per head,
    and each head's contiguously stored 512 columns stay adjacent.
    """
    num_tokens, _, head_dim = o.shape
    heads_per_cta = max(1, min(heads_per_cta, heads_per_group))
    while heads_per_group % heads_per_cta:
        heads_per_cta //= 2
    grid = (
        n_groups * heads_per_group // heads_per_cta,
        triton.cdiv(max(num_tokens, 1), block_t),
    )
    _inv_rope_quant_int8_kernel[grid](
        o,
        positions,
        cos_sin_cache,
        out,
        scale,
        num_tokens,
        heads_per_group,
        heads_per_cta,
        o.stride(0),
        o.stride(1),
        cos_sin_cache.stride(0),
        out.stride(0),
        out.stride(1),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        _FP8_MAX,
        _INT8_MAX,
        _EPS,
        _QUANT_BLOCK,
        head_dim // _QUANT_BLOCK,
        nope_dim % _QUANT_BLOCK,
        rope_dim // 2,
        block_t,
        num_warps=1,
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
# FP8 -> INT8 conversion kernels
# ---------------------------------------------------------------------------


@triton.jit
def _e4m3_byte_to_f32(bits):
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
def _fp8_payload_to_int8_kernel(
    src_ptr,
    dst_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    """``round(e4m3_value * 127/448)`` over a flat E4M3 byte payload."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    value = _e4m3_byte_to_f32(tl.load(src_ptr + offsets, mask=mask, other=0))
    value = value * 0.28348214285714285  # 127/448
    quant = tl.where(value >= 0, value + 0.5, value - 0.5).to(tl.int32)
    tl.store(dst_ptr + offsets, quant.to(tl.int8), mask=mask)


@triton.jit
def _fp8_scale_to_int8_kernel(
    src_ptr,
    dst_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    """Rescale FP8 block scales onto the INT8 grid (``* 448/127``)."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    value = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, value * 3.5275590551181102, mask=mask)


def _fp8_payload_to_int8(src: torch.Tensor, dst: torch.Tensor) -> None:
    numel = src.numel()
    if numel == 0:
        return
    _fp8_payload_to_int8_kernel[(triton.cdiv(numel, 4096),)](
        src.view(torch.uint8).reshape(-1),
        dst.reshape(-1),
        numel,
        BLOCK=4096,
        num_warps=4,
        num_stages=1,
    )


def _fp8_scale_to_int8(src: torch.Tensor, dst: torch.Tensor) -> None:
    numel = src.numel()
    if numel == 0:
        return
    _fp8_scale_to_int8_kernel[(triton.cdiv(numel, 1024),)](
        src.reshape(-1),
        dst.reshape(-1),
        numel,
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )


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
    weight = weight.contiguous()
    out = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    _fp8_payload_to_int8(weight, out)
    out_scale = torch.empty(scale.shape, dtype=torch.float32, device=scale.device)
    _fp8_scale_to_int8(scale.contiguous(), out_scale)
    return out, out_scale


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
# Pre-quantized activation (vLLM ``QuantizedActivation``): FP8 payload onto INT8
# ---------------------------------------------------------------------------


@triton.jit
def _activation_to_int8_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    out_scale_ptr,
    num_tokens,
    CHUNKS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    x_stride_t,
    x_stride_g,
    s_stride_t,
    s_stride_g,
    o_stride_g,
    o_stride_t,
    os_stride_g,
    os_stride_t,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    if token >= num_tokens:
        return
    cols = tl.arange(0, BLOCK)
    for chunk in tl.static_range(CHUNKS // (BLOCK // 128)):
        offsets = chunk * BLOCK + cols
        value = _e4m3_byte_to_f32(
            tl.load(x_ptr + token * x_stride_t + group * x_stride_g + offsets)
        )
        value = value * 0.28348214285714285  # 127/448
        quant = tl.where(value >= 0, value + 0.5, value - 0.5).to(tl.int32)
        tl.store(
            out_ptr + group * o_stride_g + token * o_stride_t + offsets,
            quant.to(tl.int8),
        )

    blocks = tl.arange(0, SCALE_BLOCK)
    mask = blocks < CHUNKS
    scales = tl.load(
        scale_ptr + token * s_stride_t + group * s_stride_g + blocks,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_scale_ptr + group * os_stride_g + token * os_stride_t + blocks,
        scales * 3.5275590551181102,  # 448/127: fp8 scale -> int8 scale
        mask=mask,
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
    chunks = reduction // _QUANT_BLOCK
    # One tile covers a whole number of 128-wide scale blocks; keep it at or
    # below 1024 elements so the program stays register-resident.
    chunks_per_tile = 8
    while chunks_per_tile > 1 and chunks % chunks_per_tile:
        chunks_per_tile //= 2
    _activation_to_int8_kernel[(num_tokens, n_groups)](
        x.view(torch.uint8),
        x_scale,
        out,
        out_scale,
        num_tokens,
        CHUNKS=chunks,
        SCALE_BLOCK=triton.next_power_of_2(chunks),
        BLOCK=chunks_per_tile * _QUANT_BLOCK,
        x_stride_t=x.stride(0),
        x_stride_g=x.stride(1),
        s_stride_t=x_scale.stride(0),
        s_stride_g=x_scale.stride(1),
        o_stride_g=out.stride(0),
        o_stride_t=out.stride(1),
        os_stride_g=out_scale.stride(0),
        os_stride_t=out_scale.stride(1),
        num_warps=4,
        num_stages=1,
    )
    return out, out_scale


# ---------------------------------------------------------------------------
# INT8 GEMM
# ---------------------------------------------------------------------------


@triton.jit
def _decode_int8_kernel(
    A,
    W,
    AS,
    WS,
    O,
    T,
    sa_g,
    sas_g,
    sas_t,
    sw_g,
    sw_n,
    sws_g,
    sws_n,
    so_g,
    so_t,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TB: tl.constexpr,
    KB: tl.constexpr,
):
    """Decode GEMM (``T <= 16``): one (group, n-block) per program.

    The shared ``_int8_block_bmm_kernel`` pays ~1.25 us of fixed cost per K
    iteration (AIU block-pointer copy + barrier), which dominates the decode
    latencies. This kernel keeps the same math and tile shape but uses plain
    row-contiguous loads, batches ``KB`` 128-wide K blocks per loop iteration
    and pads the token axis to ``TB = 16``; measured 6% faster cold and 28%
    faster with a warm L2 (weights circa L2-resident) than the shared kernel
    for ``g = 8``. The ``g = 16`` bucket is faster on the shared kernel, so only
    the g=8 decode shapes dispatch here.
    """
    pid = tl.program_id(0)
    n_blocks: tl.constexpr = N // BN
    n_k: tl.constexpr = K // 128
    g = pid // n_blocks
    nb = pid % n_blocks
    rn = nb * BN + tl.arange(0, BN)
    rt = tl.arange(0, TB)
    rk = tl.arange(0, 128)
    tmask = rt < T

    acc = tl.zeros((BN, TB), tl.float32)
    a_base = A + g * sa_g + rt[:, None] * K
    as_base = AS + g * sas_g + rt * sas_t
    w_base = W + g * sw_g + rn[:, None] * sw_n
    ws_base = WS + g * sws_g + ((nb * BN) // 128) * sws_n

    for kb0 in tl.range(0, n_k, KB):
        for u in tl.static_range(KB):
            kb = kb0 + u
            a = tl.load(a_base + kb * 128 + rk[None, :], mask=tmask[:, None], other=0)
            w = tl.load(w_base + kb * 128 + rk[None, :])
            part = tl.dot(w, tl.trans(a), out_dtype=tl.int32)
            asc = tl.load(as_base + kb, mask=tmask, other=0.0)
            wsc = tl.load(ws_base + kb)
            acc = tl.fma(part.to(tl.float32), asc[None, :] * wsc, acc)

    op = O + g * so_g + rt[:, None] * so_t + rn[None, :]
    tl.store(op, tl.trans(acc).to(tl.bfloat16), mask=tmask[:, None])


def _decode_bmm(act, weight, act_scale, weight_scale, out) -> None:
    """Launch the decode-specialized INT8 GEMM (``T <= 16``, ``g = 8``)."""
    batch, m, k = act.shape
    n = weight.shape[1]
    _decode_int8_kernel[(batch * (n // _DECODE_BLOCK_N),)](
        act,
        weight,
        act_scale,
        weight_scale,
        out,
        m,
        act.stride(0),
        act_scale.stride(0),
        act_scale.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        out.stride(0),
        out.stride(1),
        N=n,
        K=k,
        BN=_DECODE_BLOCK_N,
        TB=_DECODE_MAX_T,
        KB=_DECODE_K_BLOCKS,
        num_warps=_DECODE_WARPS,
        num_stages=2,
    )


@triton.jit
def _decode_splitk_kernel(
    A,
    W,
    AS,
    WS,
    P,
    T,
    sa_g,
    sas_g,
    sas_t,
    sw_g,
    sw_n,
    sws_g,
    sws_n,
    sp_s,
    sp_t,
    sp_g,
    sp_n,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    TB: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Split-K decode GEMM (``g = 16``): each program owns a K slice.

    The g=16 weight is 67 MB and the plain decode kernel only gets 128 CTAs
    (two waves of 64 SMs), which reaches 0.83 TB/s. Splitting K by ``SPLIT``
    raises the CTA count to ``g * N/BN * SPLIT`` and measured 80.6 -> 77.4 us
    (T=1) / 78.5 us (T=4); partial sums are fp32 and reduced by
    ``_decode_reduce_kernel``.
    """
    pid = tl.program_id(0)
    n_blocks: tl.constexpr = N // BN
    g = pid // (n_blocks * SPLIT)
    rem = pid % (n_blocks * SPLIT)
    s = rem // n_blocks
    nb = rem % n_blocks

    rn = nb * BN + tl.arange(0, BN)
    rt = tl.arange(0, TB)
    rk = tl.arange(0, 128)
    t_mask = rt < T
    acc = tl.zeros((BN, TB), tl.float32)

    k_per_split: tl.constexpr = K // SPLIT
    kb0 = s * (k_per_split // 128)
    w_base = W + g * sw_g + rn[:, None] * sw_n + kb0 * 128
    a_base = A + g * sa_g + rt[:, None] * K + kb0 * 128
    as_base = AS + g * sas_g + rt * sas_t + kb0
    ws_base = WS + g * sws_g + ((nb * BN) // 128) * sws_n + kb0

    for kb in range(k_per_split // 128):
        w = tl.load(w_base + kb * 128 + rk[None, :])
        a = tl.load(a_base + kb * 128 + rk[None, :], mask=t_mask[:, None], other=0)
        part = tl.dot(w, tl.trans(a), out_dtype=tl.int32).to(tl.float32)
        asc = tl.load(as_base + kb, mask=t_mask, other=0.0)
        wsc = tl.load(ws_base + kb)
        acc = tl.fma(part, asc[None, :] * wsc, acc)

    op = P + s * sp_s + rt[:, None] * sp_t + g * sp_g + rn[None, :]
    tl.store(op, tl.trans(acc), mask=t_mask[:, None])


@triton.jit
def _decode_reduce_kernel(
    P,
    O,
    total,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum the ``SPLIT`` fp32 partials of ``_decode_splitk_kernel`` into bf16."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(P + s * total + offs, mask=mask, other=0.0)
    tl.store(O + offs, acc.to(tl.bfloat16), mask=mask)


def _decode_splitk_bmm(act, weight, act_scale, weight_scale, out) -> None:
    """Split-K decode GEMM into a contiguous ``[T, G, N]`` output."""
    batch, m, k = act.shape
    n = weight.shape[1]
    split = _DECODE_SPLIT
    partial = torch.empty(
        (split, m, batch, n), dtype=torch.float32, device=act.device
    )
    grid = (batch * (n // _DECODE_SPLIT_BN) * split,)
    _decode_splitk_kernel[grid](
        act,
        weight,
        act_scale,
        weight_scale,
        partial,
        m,
        act.stride(0),
        act_scale.stride(0),
        act_scale.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        partial.stride(3),
        N=n,
        K=k,
        BN=_DECODE_SPLIT_BN,
        TB=_DECODE_MAX_T,
        SPLIT=split,
        num_warps=4,
        num_stages=3,
    )
    total = m * batch * n
    _decode_reduce_kernel[(triton.cdiv(total, 1024),)](
        partial,
        out,
        total,
        SPLIT=split,
        BLOCK=1024,
        num_warps=4,
    )


def _tile(m: int):
    """Tiles for the operator's ``n = 1024`` projection, measured on PPU-ZW810E.

    The shared tuner only has exact records for the ``k = 4096, g = 8`` decode
    shapes (and ``g = 16`` only for ``k = 7168``), so the ``g = 16, k = 4096``
    buckets fell back to a generic tile. Rows: (BLOCK_M, BLOCK_N, num_warps,
    num_stages, GROUP_M, ACC_FP32); ``SWAP_AB`` stays off and the async AIU
    loads are used, which measured best across the sweep.
    """
    if m <= 16:
        return 16, 128, 4, 2, 1, True
    if m <= 64:
        return 64, 128, 4, 2, 1, False
    if m <= 1024:
        # m=128 measured 73.4 us (117 TFLOPS) with this tile against 99.5 us
        # (86 TFLOPS) for the 128-row tile: 128 CTAs keep the load side busy
        # while the 64-row tile only touches two waves of SMs.
        return 64, 128, 4, 3, 32, False
    return 64, 128, 4, 3, 8, False


def _aiu_supported(act: torch.Tensor, weight_t: torch.Tensor, n: int) -> bool:
    """Mirror the shared entry's AIU (async copy) eligibility check."""
    batch, m, k = act.shape
    return (
        n % 128 == 0
        and k % _QUANT_BLOCK == 0
        and act.stride(2) == 1
        and act.stride(1) < 2**31
        and (
            weight_t.stride(2) == 1
            or (
                weight_t.stride(1) == 1
                and weight_t.data_ptr() % 128 == 0
                and weight_t.stride(0) == n * k
                and weight_t.stride(2) == k
            )
        )
        and act.data_ptr() % 128 == 0
        and (
            (act.stride(1) == k and act.stride(0) == m * k)
            or (act.stride(0) == k and act.stride(1) == batch * k)
        )
        and act.stride(0) % 128 == 0
        and act.stride(1) % 128 == 0
    )


def _lean_bmm(act, weight, act_scale, weight_scale, out) -> None:
    """Launch the tuned INT8 block BMM kernel without the shared host wrapper.

    ``w8a8_block_int8_bmm`` re-validates every argument and re-derives the tuned
    config on each call (~20 us on this CPU), which is a third of the decode
    latency; the kernel itself is unchanged.
    """
    batch, m, k = act.shape
    n = weight.shape[1]
    block_m, block_n, warps, stages, group, acc_fp32 = _tile(m)
    tile_k = _QUANT_BLOCK
    aligned = m % block_m == 0 and n % block_n == 0 and k % tile_k == 0
    weight_t = weight.transpose(1, 2)
    scale_t = weight_scale.transpose(1, 2)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), batch)
    aiu = _aiu_supported(act, weight_t, n)
    _int8_block_bmm_kernel[grid](
        act,
        weight_t,
        act_scale,
        scale_t,
        out,
        m,
        n,
        k,
        triton.cdiv(k, tile_k),
        act.stride(),
        weight_t.stride(),
        act_scale.stride(),
        scale_t.stride(),
        out.stride(),
        1,
        _QUANT_BLOCK,
        _QUANT_BLOCK,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_M=group,
        ALIGNED=aligned,
        USE_AIU=aiu,
        USE_AIU_W=aiu,
        TILE_K=tile_k,
        ACC_FP32=acc_fp32,
        num_warps=warps,
        num_stages=stages,
    )


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
    """DeepSeek-V4 output projection on PPU (INT8 tensor-core transport)."""
    logger.debug("GEMS DEEP GEMM FP8 O PROJ (THEAD)")

    if tuple(einsum_recipe) != _SUPPORTED_RECIPE or tma_aligned_scales:
        raise NotImplementedError(
            "the PPU implementation only supports the Hopper FP32 block-scale "
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
            "unsupported shape for the PPU deep_gemm_fp8_o_proj: expected "
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
            block_t = 2 if num_tokens < 64 else 4
            # Folding two heads per program saves one rope-table load per head
            # and paid off only once the token count makes the kernel fully
            # bandwidth bound (T>=2048: 260.5->254.5 us, 499.3->483.8 us);
            # below that the extra loop just adds latency.
            heads_per_cta = 2 if num_tokens >= 2048 else 1
            _launch_quant_int8(
                o,
                positions,
                cos_sin_cache,
                act_i8,
                act_scale_i8,
                n_groups,
                heads_per_group,
                nope_dim,
                rope_dim,
                block_t,
                heads_per_cta,
            )
        weight_i8, weight_scale_i8 = _cached_weight(owner, weight, weight_scale)
        # The kernel stores one group's tile at a time; writing straight into
        # the caller's ``[tokens, groups, o_lora_rank]`` buffer costs one extra
        # launch-free ``reshape`` (the view is already contiguous) and keeps the
        # store rows spread over the whole output.
        out = torch.empty(
            (num_tokens, n_groups, o_lora_rank),
            dtype=torch.bfloat16,
            device=device,
        )
        z = out.permute(1, 0, 2)
        if (
            o_lora_rank == 1024
            and num_tokens <= _DECODE_SPLIT_MAX_T
            and n_groups > 8
            and reduction % (_QUANT_BLOCK * _DECODE_SPLIT) == 0
        ):
            _decode_splitk_bmm(act_i8, weight_i8, act_scale_i8, weight_scale_i8, out)
        elif (
            o_lora_rank == 1024
            and num_tokens <= _DECODE_MAX_T
            and n_groups <= 8
            and reduction % (_QUANT_BLOCK * _DECODE_K_BLOCKS) == 0
        ):
            _decode_bmm(act_i8, weight_i8, act_scale_i8, weight_scale_i8, z)
        elif o_lora_rank == 1024 and num_tokens <= 1024:
            _lean_bmm(act_i8, weight_i8, act_scale_i8, weight_scale_i8, z)
        else:
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
    return wo_b(out.reshape(num_tokens, n_groups * o_lora_rank))

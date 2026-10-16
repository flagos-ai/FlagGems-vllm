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

"""MetaX SwiGLU MoE with plain output-major INT4 or E4M3FN weights.

C550 has no fast FP8 or INT4 conversion, so weights are loaded as int32 words
and decoded with integer bit moves; each K tile lies in one scale group, so
the scale multiplies the partial product.
"""

import math
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl
from torch.utils.weak import WeakTensorKeyDictionary

from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3, QUANT_TYPE_UINT4B8
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_no_tle,
    moe_align_block_size_small_grouped,
)
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._metax.fused.moe_sum import _qwen_moe_sum_kernel

# Module import: ops.fused_moe imports this package through fused.moe_sum.
from flaggems_vllm.runtime.backend._metax.ops import fused_moe as metax_fused_moe

MIN_GROUP_SIZE = 128
LARGE_EXPERT_MIN_COUNT = 128
MAX_GROUPED_ALIGN_EXPERTS = 1024
# The grouped align unrolls every route for every expert; bound compile time.
LARGE_EXPERT_GROUPED_MAX_ROUTES = 64
SMALL_EXPERT_GROUPED_MAX_ROUTES = 128
FP8_ALIGN_MAX_ROUTES = 4096
# Sub-warp route gathers fail to lower on the C550 compiler.
FP8_ALIGN_MIN_TILE = 256
FP8_GEMV_MAX_ROUTES_PER_EXPERT = 0.5
INT4_GEMV_MAX_ROUTES_PER_EXPERT = 0.75
# Sparse INT4 batches precompute activation sums; dense ones sum per tile.
INT4_PRESUM_MAX_ROUTES_PER_EXPERT = 16
GEMV_BLOCK_K = 128
GEMV_NARROW_MAX_OUTPUTS = 8192
# Dense FP8 batches run the gate/up GEMM on a cached BF16 copy of w1.
FP8_DENSE_MIN_ROUTES_PER_EXPERT = 64
FP8_DEQUANT_MAX_BYTES = 1 << 30
FP8_DEQUANT_BLOCK_ROWS = 16
FP8_NARROW_MAX_INTERMEDIATE = 512
N_TILES_MAX_K = 512
FP8_DECODE_SCALE = tl.constexpr(256.0)
FP8_BF16_REBIAS = tl.constexpr(2.0**120)
# A nibble ORed into the mantissa of 2^7 (BF16) or 2^10 (FP16) gives that
# power plus the nibble exactly; the bias also removes the UINT4B8 zero point.
INT4_BF16_MAGIC = tl.constexpr(0x43004300)
INT4_FP16_MAGIC = tl.constexpr(0x64006400)
INT4_BF16_BIAS = tl.constexpr(128.0 + 8.0)
INT4_FP16_BIAS = tl.constexpr(1024.0 + 8.0)


class MoeTile(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    pipeline: str
    scenario: str = ""
    # N tiles walked per program, so a short K still keeps loads in flight.
    n_tiles: int = 1


# (max routes per expert, gate/up tile, down tile), tuned on C550.
SPARSE_TIER = (
    20,
    MoeTile(16, 64, 128, 4, 2, "basic"),
    MoeTile(16, 64, 128, 4, 2, "basic", n_tiles=8),
)
FP8_DENSE_DOWN = MoeTile(128, 128, 64, 4, 4, "cpasync", "unroll")
FP8_NARROW_TIERS = (
    SPARSE_TIER,
    (
        32,
        MoeTile(32, 64, 128, 4, 2, "cpasync", "unroll"),
        MoeTile(32, 128, 128, 2, 2, "cpasync", "unroll"),
    ),
    (
        64,
        MoeTile(64, 128, 128, 4, 2, "cpasync", "unroll"),
        MoeTile(64, 128, 64, 4, 3, "cpasync"),
    ),
    # Dense batches whose w1 copy would exceed FP8_DEQUANT_MAX_BYTES.
    (
        math.inf,
        MoeTile(64, 128, 128, 4, 4, "cpasync", "unroll"),
        FP8_DENSE_DOWN,
    ),
)
WIDE_TIERS = (
    SPARSE_TIER,
    (32,) + (MoeTile(32, 128, 128, 4, 4, "cpasync"),) * 2,
    (128,) + (MoeTile(64, 128, 128, 4, 4, "cpasync"),) * 2,
    # K=128 is slower than K=64 for 128-row tiles on C550.
    (math.inf,) + (MoeTile(128, 128, 64, 4, 4, "cpasync"),) * 2,
)
# INT4 keeps 64-row tiles for dense batches; 128-row tiles are slower.
INT4_TIERS = WIDE_TIERS[:2] + ((math.inf,) + WIDE_TIERS[2][1:],)
FP8_DENSE_GATE_UP = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 4,
    "pipeline": "cpasync",
}
# w1 -> {dtype: (version key, dequantized w1)}; entries die with the weight.
_DEQUANT_CACHE = WeakTensorKeyDictionary()


@triton.jit
def decode_fp8_e4m3(weight):
    """E4M3FN bytes to exact FP16 values divided by FP8_DECODE_SCALE."""
    # Sign extension puts the sign in bit 15; clear its copy in bit 14.
    bits = weight.to(tl.int8, bitcast=True).to(tl.int16)
    return ((bits << 7) & -0x4001).to(tl.float16, bitcast=True)


@triton.jit
def fp32_bits_to_bf16(bits):
    # Exact: E4M3 values leave the low 16 FP32 bits zero.
    value = bits.to(tl.float32, bitcast=True) * FP8_BF16_REBIAS
    value = (value.to(tl.int32, bitcast=True) >> 16).to(tl.int16)
    return value.to(tl.bfloat16, bitcast=True)


@triton.jit
def decode_fp8_e4m3_words(
    word, ROWS: tl.constexpr, COLS: tl.constexpr, compute_type: tl.constexpr
):
    """int32 words of four K-consecutive E4M3FN bytes to [ROWS, COLS] values.

    Values are exact. BF16 puts each byte in FP32 bits without a rebias, so
    subnormal codes stay subnormal, then multiplies by 2^120. FP16 values are
    divided by FP8_DECODE_SCALE, which also keeps subnormals exact.
    """
    if compute_type == tl.bfloat16:
        # Arithmetic shifts; 0x87F00000 keeps the sign and magnitude bits.
        byte0 = fp32_bits_to_bf16(((word << 24) >> 4) & -0x78100000)
        byte1 = fp32_bits_to_bf16(((word << 16) >> 4) & -0x78100000)
        byte2 = fp32_bits_to_bf16(((word << 8) >> 4) & -0x78100000)
        byte3 = fp32_bits_to_bf16((word >> 4) & -0x78100000)
    else:
        sign_odd = word & -0x7FFF8000
        sign_even = (word & 0x00800080) << 8
        even = ((word & 0x007F007F) << 7) | sign_even
        odd = ((word >> 1) & 0x3F803F80) | sign_odd
        byte0 = even.to(tl.int16).to(compute_type, bitcast=True)
        byte2 = (even >> 16).to(tl.int16).to(compute_type, bitcast=True)
        byte1 = odd.to(tl.int16).to(compute_type, bitcast=True)
        byte3 = (odd >> 16).to(tl.int16).to(compute_type, bitcast=True)
    return tl.reshape(
        tl.join(tl.join(byte0, byte2), tl.join(byte1, byte3)), (ROWS, COLS)
    )


@triton.jit
def decode_uint4b8_words(
    word, ROWS: tl.constexpr, COLS: tl.constexpr, compute_type: tl.constexpr
):
    """int32 words of eight K-consecutive UINT4B8 nibbles, low nibble first, to
    [ROWS, COLS] values offset by 2^7 (BF16) or 2^10 (FP16)."""
    magic = INT4_BF16_MAGIC if compute_type == tl.bfloat16 else INT4_FP16_MAGIC
    # Pair j holds nibble j in its low half and nibble j + 4 in its high half.
    pair0 = (word & 0x000F000F) | magic
    pair1 = ((word >> 4) & 0x000F000F) | magic
    pair2 = ((word >> 8) & 0x000F000F) | magic
    pair3 = ((word >> 12) & 0x000F000F) | magic
    low0 = pair0.to(tl.int16).to(compute_type, bitcast=True)
    low1 = pair1.to(tl.int16).to(compute_type, bitcast=True)
    low2 = pair2.to(tl.int16).to(compute_type, bitcast=True)
    low3 = pair3.to(tl.int16).to(compute_type, bitcast=True)
    high0 = (pair0 >> 16).to(tl.int16).to(compute_type, bitcast=True)
    high1 = (pair1 >> 16).to(tl.int16).to(compute_type, bitcast=True)
    high2 = (pair2 >> 16).to(tl.int16).to(compute_type, bitcast=True)
    high3 = (pair3 >> 16).to(tl.int16).to(compute_type, bitcast=True)
    even = tl.join(tl.join(low0, high0), tl.join(low2, high2))
    odd = tl.join(tl.join(low1, high1), tl.join(low3, high3))
    return tl.reshape(tl.join(even, odd), (ROWS, COLS))


@triton.jit
def chunk_sum_kernel(x_ptr, out_ptr, K: tl.constexpr, CHUNK: tl.constexpr):
    """out[row, c] = sum of x[row, c * CHUNK : (c + 1) * CHUNK] in FP32."""
    row = tl.program_id(axis=0).to(tl.int64)
    offs = tl.program_id(axis=1) * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(x_ptr + row * K + offs).to(tl.float32)
    tl.store(out_ptr + row * (K // CHUNK) + tl.program_id(axis=1), tl.sum(x, axis=0))


@triton.jit
def tile_row_sum(activation, sum_ptrs, mask, PRESUM: tl.constexpr):
    """FP32 activation sums of one INT4 K tile."""
    if PRESUM:
        row_sum = tl.load(sum_ptrs, mask=mask, other=0.0)
    else:
        row_sum = tl.sum(activation.to(tl.float32), axis=1)
    return row_sum


@triton.jit
def scaled_partial(
    activation,
    row_sum,
    word,
    scale,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    compute_type: tl.constexpr,
    WEIGHT_BITS: tl.constexpr,
):
    """Scaled FP32 activation @ weight.T of one K tile; scale is FP32 [ROWS],
    row_sum the FP32 activation sums of the tile (INT4 only)."""
    if WEIGHT_BITS == 4:
        code = decode_uint4b8_words(word, ROWS, COLS, compute_type)
        partial = tl.dot(activation, tl.trans(code), allow_tf32=False)
        bias = INT4_BF16_BIAS if compute_type == tl.bfloat16 else INT4_FP16_BIAS
        partial = (partial - bias * row_sum[:, None]) * scale[None, :]
    else:
        code = decode_fp8_e4m3_words(word, ROWS, COLS, compute_type)
        partial = tl.dot(activation, tl.trans(code), allow_tf32=False)
        decode_scale = 1.0 if compute_type == tl.bfloat16 else FP8_DECODE_SCALE
        partial = partial * (scale * decode_scale)[None, :]
    return partial


@triton.jit
def gemv_partial(
    w_ptrs,
    activation,
    scale,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    compute_type: tl.constexpr,
    WEIGHT_BITS: tl.constexpr,
):
    """FP32 [ROWS] products of one K tile with FP32 activation [1, COLS]."""
    if WEIGHT_BITS == 4:
        code = decode_uint4b8_words(tl.load(w_ptrs), ROWS, COLS, compute_type)
        bias = INT4_BF16_BIAS if compute_type == tl.bfloat16 else INT4_FP16_BIAS
        product = tl.sum(code.to(tl.float32) * activation, axis=1)
        product -= bias * tl.sum(activation, axis=1)
    else:
        code = decode_fp8_e4m3(tl.load(w_ptrs)).to(tl.float32)
        product = tl.sum(code * activation, axis=1)
    return product * scale


@triton.jit
def moe_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    a_sum_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_be,
    stride_bn,
    stride_cm,
    stride_bse,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    N_TILES: tl.constexpr,
    ALIGN_BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    WEIGHT_BITS: tl.constexpr,
    PRESUM: tl.constexpr,
):
    """B is UINT4B8 or E4M3FN viewed as int32 words. Routes are padded per
    expert to ALIGN_BLOCK_M, a multiple of BLOCK_SIZE_M."""
    compute_type = a_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    pid_m = pid // (N // (BLOCK_SIZE_N * N_TILES))
    pid_n = pid % (N // (BLOCK_SIZE_N * N_TILES))
    if pid_m * BLOCK_SIZE_M >= tl.load(num_tokens_post_padded_ptr):
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    expert = tl.load(expert_ids_ptr + pid_m // (ALIGN_BLOCK_M // BLOCK_SIZE_M))
    expert = expert.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :]
    # PRESUM: a_sum_ptr holds [rows, K // BLOCK_SIZE_K] activation sums.
    a_sum_ptrs = a_sum_ptr + (offs_token // top_k) * (K // BLOCK_SIZE_K)
    words_k: tl.constexpr = BLOCK_SIZE_K * WEIGHT_BITS // 32
    # K-contiguous words load faster than a K-major tile.
    b_ptrs = b_ptr + expert * stride_be + tl.arange(0, words_k)[None, :]
    scale_ptrs = b_scale_ptr + expert * stride_bse
    if MUL_ROUTED_WEIGHT:
        route = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
    num_k_tiles: tl.constexpr = K // BLOCK_SIZE_K
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if N_TILES == 1:
        # Pointer-increment loop; measurably faster than the flat loop below.
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        b_ptrs += offs_n[:, None] * stride_bn
        scale_ptrs += offs_n * stride_bsn
        for tile in tl.range(num_k_tiles):
            activation = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            scale = tl.load(scale_ptrs + tile * BLOCK_SIZE_K // GROUP_SIZE)
            row_sum = 0.0
            if WEIGHT_BITS == 4:
                row_sum = tile_row_sum(
                    activation, a_sum_ptrs + tile, token_mask, PRESUM
                )
            accumulator += scaled_partial(
                activation,
                row_sum,
                tl.load(b_ptrs),
                scale.to(tl.float32),
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                compute_type,
                WEIGHT_BITS,
            )
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += words_k
        if MUL_ROUTED_WEIGHT:
            accumulator = accumulator * route[:, None]
        tl.store(
            c_ptrs + pid_n * BLOCK_SIZE_N,
            accumulator.to(compute_type),
            mask=token_mask[:, None],
        )
        return
    for step in tl.range(N_TILES * num_k_tiles):
        n_tile = step // num_k_tiles
        tile = step % num_k_tiles
        offs_n = (pid_n * N_TILES + n_tile) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        activation = tl.load(
            a_ptrs + tile * BLOCK_SIZE_K, mask=token_mask[:, None], other=0.0
        )
        word = tl.load(b_ptrs + offs_n[:, None] * stride_bn + tile * words_k)
        group = tile * BLOCK_SIZE_K // GROUP_SIZE
        scale = tl.load(scale_ptrs + offs_n * stride_bsn + group).to(tl.float32)
        row_sum = 0.0
        if WEIGHT_BITS == 4:
            row_sum = tile_row_sum(activation, a_sum_ptrs + tile, token_mask, PRESUM)
        partial = scaled_partial(
            activation,
            row_sum,
            word,
            scale,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            compute_type,
            WEIGHT_BITS,
        )
        accumulator = tl.where(tile == 0, partial, accumulator + partial)
        if tile == num_k_tiles - 1:
            result = accumulator
            if MUL_ROUTED_WEIGHT:
                result = result * route[:, None]
            n_start = (pid_n * N_TILES + n_tile) * BLOCK_SIZE_N
            tl.store(
                c_ptrs + n_start, result.to(compute_type), mask=token_mask[:, None]
            )


@triton.jit
def moe_gemv_kernel(
    a_ptr,
    w_ptr,
    scale_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    top_k: tl.constexpr,
    FIRST: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    WEIGHT_BITS: tl.constexpr,
):
    """FIRST: SiLU(gate) * up of one route. Otherwise the sum of one token's
    top-k down projections, which replaces moe_sum. INT4 weights are int32
    words, FP8 weights are bytes."""
    compute_type = a_ptr.dtype.element_ty
    row = tl.program_id(axis=0).to(tl.int64)
    offs_n = tl.program_id(axis=1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    w_step: tl.constexpr = BLOCK_SIZE_K // 8 if WEIGHT_BITS == 4 else BLOCK_SIZE_K
    total = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for slot in tl.static_range(1 if FIRST else top_k):
        route = row if FIRST else row * top_k + slot
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        w_ptrs = w_ptr + expert * stride_we + offs_n[:, None] * stride_wn
        w_ptrs += tl.arange(0, w_step)[None, :]
        scale_ptrs = scale_ptr + expert * stride_se + offs_n * stride_sn
        a_ptrs = a_ptr + (row // top_k if FIRST else route) * K + offs_k
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        up = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for tile in tl.range(K // BLOCK_SIZE_K):
            group = tile * BLOCK_SIZE_K // GROUP_SIZE
            a = tl.load(a_ptrs).to(tl.float32)[None, :]
            scale = tl.load(scale_ptrs + group).to(tl.float32)
            acc += gemv_partial(
                w_ptrs, a, scale, BLOCK_SIZE_N, BLOCK_SIZE_K, compute_type, WEIGHT_BITS
            )
            if FIRST:
                scale_up = tl.load(scale_ptrs + N * stride_sn + group).to(tl.float32)
                up += gemv_partial(
                    w_ptrs + N * stride_wn,
                    a,
                    scale_up,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    compute_type,
                    WEIGHT_BITS,
                )
            a_ptrs += BLOCK_SIZE_K
            w_ptrs += w_step
        if WEIGHT_BITS == 8:
            acc *= FP8_DECODE_SCALE
            up *= FP8_DECODE_SCALE
        if MUL_ROUTED_WEIGHT:
            route_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
            acc *= route_weight
            up *= route_weight
        # Round like the routed buffers of the GEMM path.
        if FIRST:
            gate = acc.to(compute_type).to(tl.float32)
            up = up.to(compute_type).to(tl.float32)
            total = gate / (1.0 + tl.exp(-gate)) * up
        else:
            total += acc.to(compute_type).to(tl.float32)
    tl.store(out_ptr + row * N + offs_n, total.to(compute_type))


@triton.jit
def dequantize_fp8_kernel(
    weight_ptr,
    scale_ptr,
    out_ptr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(axis=0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    offs_k = tl.program_id(axis=1) * BLOCK_K + tl.arange(0, BLOCK_K)
    code = decode_fp8_e4m3(tl.load(weight_ptr + rows[:, None] * K + offs_k[None, :]))
    scale = tl.load(
        scale_ptr + rows[:, None] * (K // GROUP_SIZE) + offs_k[None, :] // GROUP_SIZE
    ).to(tl.float32)
    value = code.to(tl.float32) * (scale * FP8_DECODE_SCALE)
    tl.store(
        out_ptr + rows[:, None] * K + offs_k[None, :],
        value.to(out_ptr.dtype.element_ty),
    )


def sum_routes(routed, output):
    """MetaX's unrolled moe_sum kernel for every top-k: moe_sum sends other
    top-k values to a generic autotuner whose 1024-thread config fails on C550."""
    num_tokens, top_k, hidden = routed.shape
    _qwen_moe_sum_kernel[(num_tokens, triton.cdiv(hidden, 2048))](
        routed,
        output,
        routed,
        0,
        0,
        num_tokens,
        hidden,
        TOPK=top_k,
        APPLY_ROUTER_WEIGHT=False,
        BLOCK_SIZE=2048,
        num_warps=8,
    )


@triton.jit
def zero_workspace_kernel(x_ptr, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offsets, 0, mask=offsets < numel)


def align_routes(
    topk_ids: torch.Tensor, block_m: int, num_experts: int, use_atomic: bool = False
):
    max_grouped_routes = (
        LARGE_EXPERT_GROUPED_MAX_ROUTES
        if num_experts >= LARGE_EXPERT_MIN_COUNT
        else SMALL_EXPERT_GROUPED_MAX_ROUTES
    )
    if (
        topk_ids.numel() <= max_grouped_routes
        and num_experts <= MAX_GROUPED_ALIGN_EXPERTS
    ):
        return moe_align_block_size_small_grouped(topk_ids, num_experts, block_m)
    if (
        use_atomic
        and topk_ids.numel() <= FP8_ALIGN_MAX_ROUTES
        and num_experts <= MAX_GROUPED_ALIGN_EXPERTS
    ):
        return atomic_align_routes(topk_ids, block_m, num_experts)
    # Zero the aligner's cumsum and count buffers with Triton, not Torch.
    workspace = topk_ids.new_empty(((num_experts + 1) ** 2,), dtype=torch.int32)
    zero_workspace_kernel[(triton.cdiv(workspace.numel(), 1024),)](
        workspace, workspace.numel(), BLOCK=1024
    )
    cumsum, counts = workspace.split([num_experts + 1, num_experts * (num_experts + 1)])
    return moe_align_block_size_no_tle(
        topk_ids, block_m, num_experts, workspace=(cumsum, counts)
    )


@triton.jit
def atomic_align_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    counters_ptr,
    NUM_EXPERTS: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    route_offsets = tl.arange(0, BLOCK_ROUTES)
    valid_route = route_offsets < NUM_ROUTES
    ids = tl.load(topk_ids_ptr + route_offsets, mask=valid_route, other=0).to(tl.int32)
    counts = tl.histogram(ids, BLOCK_EXPERTS, mask=valid_route)
    padded_counts = tl.cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M
    padded_starts = tl.cumsum(padded_counts, 0) - padded_counts
    tl.store(num_tokens_post_pad_ptr, tl.sum(padded_counts, 0))

    expert_offsets = tl.arange(0, BLOCK_EXPERTS)
    tl.store(counters_ptr + expert_offsets, 0, mask=expert_offsets < NUM_EXPERTS)
    # Every thread must see the counter reset before assigning route ranks.
    tl.debug_barrier()
    ranks = tl.atomic_add(
        counters_ptr + ids, 1, mask=valid_route, sem="relaxed", scope="cta"
    )
    destinations = tl.gather(padded_starts, ids, axis=0) + ranks
    tl.store(sorted_token_ids_ptr + destinations, route_offsets, mask=valid_route)
    tl.store(
        expert_ids_ptr + destinations // BLOCK_SIZE_M,
        ids,
        mask=valid_route & (ranks % BLOCK_SIZE_M == 0),
    )

    # Atomic ranks occupy [0, counts); padding occupies [counts, padded_counts).
    padding_offsets = tl.arange(0, BLOCK_SIZE_M)
    padding_destinations = (
        padded_starts[:, None] + counts[:, None] + padding_offsets[None, :]
    )
    tl.store(
        sorted_token_ids_ptr + padding_destinations,
        NUM_ROUTES,
        mask=(expert_offsets[:, None] < NUM_EXPERTS)
        & (padding_offsets[None, :] < (padded_counts - counts)[:, None]),
    )


def atomic_align_routes(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    num_routes = topk_ids.numel()
    capacity = min(num_routes * block_m, num_routes + num_experts * (block_m - 1))
    sorted_ids = topk_ids.new_empty((capacity,), dtype=torch.int32)
    expert_ids = topk_ids.new_empty(
        (triton.cdiv(capacity, block_m),), dtype=torch.int32
    )
    num_tokens_post_pad = topk_ids.new_empty((1,), dtype=torch.int32)
    counters = topk_ids.new_empty((num_experts,), dtype=torch.int32)
    atomic_align_kernel[(1,)](
        topk_ids,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        counters,
        NUM_EXPERTS=num_experts,
        NUM_ROUTES=num_routes,
        BLOCK_SIZE_M=block_m,
        BLOCK_EXPERTS=triton.next_power_of_2(num_experts),
        BLOCK_ROUTES=triton.next_power_of_2(max(FP8_ALIGN_MIN_TILE, num_routes)),
        num_warps=4,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def launch_gemm(
    activation,
    weight,
    scale,
    output,
    topk_weights,
    alignment,
    *,
    tile,
    align_block_m,
    mul_routed_weight,
    top_k,
    group_size,
):
    out_features, reduction = weight.shape[1], activation.shape[1]
    num_routes = topk_weights.numel()
    problem_m = min(alignment[0].shape[0], num_routes * align_block_m)
    n_tiles = tile.n_tiles if reduction <= N_TILES_MAX_K else 1
    while out_features % (tile.block_n * n_tiles):
        n_tiles //= 2
    presum = (
        weight.dtype == torch.uint8
        and num_routes <= INT4_PRESUM_MAX_ROUTES_PER_EXPERT * weight.shape[0]
    )
    activation_sums = activation
    if presum:
        rows = activation.shape[0]
        chunks = reduction // tile.block_k
        activation_sums = activation.new_empty((rows, chunks), dtype=torch.float32)
        chunk_sum_kernel[(rows, chunks)](
            activation, activation_sums, reduction, CHUNK=tile.block_k
        )
    grid = (
        triton.cdiv(problem_m, tile.block_m)
        * (out_features // (tile.block_n * n_tiles)),
    )
    moe_gemm_kernel[grid](
        activation,
        weight.view(torch.int32),
        output,
        scale,
        activation_sums,
        topk_weights,
        *alignment,
        out_features,
        reduction,
        num_routes,
        activation.stride(0),
        weight.stride(0) // 4,
        weight.stride(1) // 4,
        output.stride(-2),
        scale.stride(0),
        scale.stride(1),
        BLOCK_SIZE_M=tile.block_m,
        BLOCK_SIZE_N=tile.block_n,
        BLOCK_SIZE_K=tile.block_k,
        N_TILES=n_tiles,
        ALIGN_BLOCK_M=align_block_m,
        GROUP_SIZE=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        WEIGHT_BITS=4 if weight.dtype == torch.uint8 else 8,
        PRESUM=presum,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
        pipeline=tile.pipeline,
        scenario=tile.scenario,
    )


def launch_gemv(activation, weight, scale, topk_ids, topk_weights, output, **kw):
    num_rows, width = output.shape
    # Narrower tiles keep the device busy when there are few outputs.
    block_n = 16 if num_rows * width < GEMV_NARROW_MAX_OUTPUTS else 32
    is_int4 = weight.dtype == torch.uint8
    weight = weight.view(torch.int32) if is_int4 else weight.view(torch.uint8)
    moe_gemv_kernel[(num_rows, width // block_n)](
        activation,
        weight,
        scale,
        topk_ids,
        topk_weights,
        output,
        width,
        activation.shape[1],
        weight.stride(0),
        weight.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=GEMV_BLOCK_K,
        WEIGHT_BITS=4 if is_int4 else 8,
        num_warps=4,
        **kw,
    )


def dequantize_fp8(weight, scale, group_size, dtype, output=None):
    num_experts, out_features, reduction = weight.shape
    if output is None:
        output = torch.empty(weight.shape, device=weight.device, dtype=dtype)
    block_k = 256 if reduction % 256 == 0 else 128
    grid = (num_experts * out_features // FP8_DEQUANT_BLOCK_ROWS, reduction // block_k)
    dequantize_fp8_kernel[grid](
        weight.view(torch.uint8),
        scale,
        output,
        reduction,
        GROUP_SIZE=group_size,
        BLOCK_ROWS=FP8_DEQUANT_BLOCK_ROWS,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


def cached_dequantize_fp8(weight, scale, group_size, dtype):
    """Dequantized w1 kept across calls, including CUDA graph replays.

    Eager calls build the copy or refill it in place, so its address never
    changes and graphs captured after an eager warm-up read it without
    dequantizing. A replay sees the copy as of the last eager refill: after
    an in-place weight update, run one eager call before replaying (vLLM does
    not update weights while serving). Captures without a valid copy and
    inference tensors dequantize on every call.
    """
    if weight.is_inference() or scale.is_inference():
        return dequantize_fp8(weight, scale, group_size, dtype)
    # In-place updates bump _version; rebinding storage changes data_ptr.
    key = tuple(
        (t._version, t.data_ptr(), t.shape, t.dtype) for t in (weight, scale)
    ) + (group_size,)
    copies = _DEQUANT_CACHE.setdefault(weight, {})
    entry = copies.get(dtype)
    if entry is not None and entry[0] == key:
        return entry[1]
    if torch_device_fn.is_current_stream_capturing():
        return dequantize_fp8(weight, scale, group_size, dtype)
    buffer = entry[1] if entry is not None and entry[1].shape == weight.shape else None
    buffer = dequantize_fp8(weight, scale, group_size, dtype, buffer)
    copies[dtype] = (key, buffer)
    return buffer


def run_quant_moe(hs, w1, w2, s1, s2, topk_weights, topk_ids, output, **options):
    num_tokens, hidden_size = hs.shape
    num_experts, fused_intermediate, _ = w1.shape
    intermediate_size = fused_intermediate // 2
    top_k = topk_ids.shape[1]
    num_routes = topk_ids.numel()
    routes_per_expert = num_routes / num_experts
    group_size = options["group_size"]
    router_on_input = options["apply_router_weight_on_input"]
    is_int4 = w1.dtype == torch.uint8
    activated = hs.new_empty((num_routes, intermediate_size))
    gemv_max_routes = (
        INT4_GEMV_MAX_ROUTES_PER_EXPERT if is_int4 else FP8_GEMV_MAX_ROUTES_PER_EXPERT
    )
    if routes_per_expert <= gemv_max_routes:
        common = dict(GROUP_SIZE=group_size, top_k=top_k)
        launch_gemv(
            hs,
            w1,
            s1,
            topk_ids,
            topk_weights,
            activated,
            FIRST=True,
            MUL_ROUTED_WEIGHT=router_on_input,
            num_stages=2,
            pipeline="basic",
            **common,
        )
        launch_gemv(
            activated,
            w2,
            s2,
            topk_ids,
            topk_weights,
            output,
            FIRST=False,
            MUL_ROUTED_WEIGHT=not router_on_input,
            **common,
        )
        return output
    should_dequantize_w1 = (
        not is_int4
        and routes_per_expert > FP8_DENSE_MIN_ROUTES_PER_EXPERT
        and w1.numel() * hs.element_size() <= FP8_DEQUANT_MAX_BYTES
    )
    if should_dequantize_w1:
        align_block_m = FP8_DENSE_GATE_UP["BLOCK_SIZE_M"]
        down_tile = FP8_DENSE_DOWN
    else:
        if is_int4:
            tiers = INT4_TIERS
        elif intermediate_size <= FP8_NARROW_MAX_INTERMEDIATE:
            tiers = FP8_NARROW_TIERS
        else:
            tiers = WIDE_TIERS
        _, gate_up_tile, down_tile = next(
            tier for tier in tiers if routes_per_expert <= tier[0]
        )
        align_block_m = max(gate_up_tile.block_m, down_tile.block_m)
    alignment = align_routes(
        topk_ids, align_block_m, num_experts, use_atomic=not is_int4
    )
    gate_up = hs.new_empty((num_routes, fused_intermediate))
    common = dict(top_k=top_k, align_block_m=align_block_m, group_size=group_size)
    if should_dequantize_w1:
        metax_fused_moe.invoke_fused_moe_triton_kernel(
            hs,
            cached_dequantize_fp8(w1, s1, group_size, hs.dtype),
            gate_up.view(num_tokens, top_k, fused_intermediate),
            None,
            None,
            topk_weights,
            *alignment,
            router_on_input,
            top_k,
            FP8_DENSE_GATE_UP,
            tl.float16 if hs.dtype == torch.float16 else tl.bfloat16,
        )
    else:
        launch_gemm(
            hs,
            w1,
            s1,
            gate_up,
            topk_weights,
            alignment,
            tile=gate_up_tile,
            mul_routed_weight=router_on_input,
            **common,
        )
    silu_and_mul_out(*gate_up.chunk(2, dim=-1), activated)
    routed = hs.new_empty((num_tokens, top_k, hidden_size))
    common["top_k"] = 1
    launch_gemm(
        activated,
        w2,
        s2,
        routed,
        topk_weights,
        alignment,
        tile=down_tile,
        mul_routed_weight=not router_on_input,
        **common,
    )
    sum_routes(routed, output)
    return output


def check_inputs(hs, w1, w2, s1, s2, topk_weights, topk_ids, output, group_size, quant):
    if hs.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or topk_ids.ndim != 2:
        raise ValueError("expected rank-2 activations and routing, rank-3 weights")
    if hs.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("activations must be FP16 or BF16")
    if group_size < MIN_GROUP_SIZE or group_size % MIN_GROUP_SIZE:
        raise NotImplementedError(f"group_size must be a multiple of {MIN_GROUP_SIZE}")
    num_tokens, hidden_size = hs.shape
    num_experts, fused_intermediate, _ = w1.shape
    intermediate_size = fused_intermediate // 2
    pack = 2 if quant == QUANT_TYPE_UINT4B8 else 1
    if (
        num_experts == 0
        or intermediate_size == 0
        or hidden_size % group_size
        or intermediate_size % group_size
        or w1.shape != (num_experts, 2 * intermediate_size, hidden_size // pack)
        or w2.shape != (num_experts, hidden_size, intermediate_size // pack)
    ):
        raise ValueError("weight shapes must be group-aligned and match hs")
    if s1.shape != (
        num_experts,
        2 * intermediate_size,
        hidden_size // group_size,
    ) or s2.shape != (num_experts, hidden_size, intermediate_size // group_size):
        raise ValueError("scale shapes do not match the quantization groups")
    weight_dtype = torch.uint8 if pack == 2 else torch.float8_e4m3fn
    scale_dtypes = (hs.dtype,) if pack == 2 else (hs.dtype, torch.float32)
    if w1.dtype != weight_dtype or w2.dtype != weight_dtype:
        raise ValueError(f"weight dtype must be {weight_dtype}")
    if s1.dtype not in scale_dtypes or s2.dtype != s1.dtype:
        raise ValueError(f"scale dtype must be one of {scale_dtypes}")
    if (
        topk_weights.shape != topk_ids.shape
        or topk_ids.shape[0] != num_tokens
        or not 1 <= topk_ids.shape[1] <= num_experts
        or topk_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("routing must be [tokens, topk] with integer expert ids")
    tensors = (hs, w1, w2, s1, s2, topk_weights, topk_ids)
    if output is not None:
        tensors += (output,)
        if output.shape != hs.shape or output.dtype != hs.dtype:
            raise ValueError("output must match hidden_states shape and dtype")
    if any(t.device != hs.device or not t.is_contiguous() for t in tensors):
        raise ValueError("MoE tensors must be contiguous and on one device")


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Any = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """UINT4B8 or FP8 E4M3 SwiGLU MoE with A16 activations."""
    if quant_type_id not in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP8_E4M3):
        raise NotImplementedError(
            f"MetaX does not support quant_type_id {quant_type_id}"
        )
    if any(x is not None for x in (g_idx1, g_idx2, sort_indices1, sort_indices2)):
        raise NotImplementedError("act_order is not supported")
    if input_dtype is not None:
        raise NotImplementedError("FP8 / INT8 input quantization is not supported")
    name = activation.value if isinstance(activation, Enum) else activation
    if name is not None and str(name).lower() != "silu":
        raise NotImplementedError("only the SiLU activation is supported")
    unsupported = (
        bias1,
        bias2,
        activation_func,
        moe_sum,
        expert_map,
        input_global_scale1,
        input_global_scale2,
        global_scale1,
        global_scale2,
        w1_zeros,
        w2_zeros,
        clamp_limit,
    )
    if any(x is not None for x in unsupported) or not is_k_full:
        raise NotImplementedError("unsupported fused_marlin_moe option on MetaX")
    if global_num_experts not in (-1, w1.shape[0]):
        raise NotImplementedError("expert maps are not supported")
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")
    if inplace:
        output = hidden_states
    check_inputs(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        group_size,
        quant_type_id,
    )
    if output is None:
        output = torch.empty_like(hidden_states)
    if hidden_states.shape[0] == 0:
        return output
    with torch_device_fn.device(hidden_states.device):
        return run_quant_moe(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            output,
            group_size=group_size,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


def fused_marlin_moe_w4a16_int4(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    *,
    activation="silu",
    group_size=128,
    apply_router_weight_on_input=False,
    inplace=False,
    swap_ab=True,
):
    return fused_marlin_moe(
        hidden_states,
        w1,
        w2,
        None,
        None,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        QUANT_TYPE_UINT4B8,
        activation=activation,
        group_size=group_size,
        apply_router_weight_on_input=apply_router_weight_on_input,
        inplace=inplace,
    )


def fused_marlin_moe_w8a16_fp8(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    *,
    w1_scale,
    w2_scale,
    group_size=128,
    inplace=False,
    output=None,
):
    return fused_marlin_moe(
        hidden_states,
        w1,
        w2,
        None,
        None,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        QUANT_TYPE_FP8_E4M3,
        group_size=group_size,
        inplace=inplace,
        output=output,
    )


__all__ = [
    "fused_marlin_moe",
    "fused_marlin_moe_w4a16_int4",
    "fused_marlin_moe_w8a16_fp8",
]

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
# Dense FP8 batches tile w1 dequantization to BF16/FP16 within a bounded workspace.
FP8_DENSE_MIN_ROUTES_PER_EXPERT = 128
FP8_DENSE_MAX_WORKSPACE_BYTES = 1 << 30
DOWN_ROUTE_MAX_BYTES = 256 << 20
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
        MoeTile(64, 128, 128, 4, 4, "cpasync", "unroll"),
        MoeTile(64, 128, 64, 4, 4, "cpasync"),
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


@triton.jit
def decode_fp8_e4m3(weight):
    """E4M3FN bytes to exact FP16 values divided by FP8_DECODE_SCALE."""
    bits = weight.to(tl.int8, bitcast=True).to(tl.int16)
    raw = (bits << 7) & -0x4001
    # 0x7f / 0xff map to +/-0x3f80; adding 0x0080 carries into bit 14 to form FP16 NaN.
    return (raw | ((raw + 0x0080) & 0x4000)).to(tl.float16, bitcast=True)


@triton.jit
def fp32_bits_to_bf16(bits):
    # Exact for finite E4M3 values (low 16 FP32 bits are zero); 0x7f / 0xff are
    # handled via SWAR NaN accumulation in the caller.
    value = bits.to(tl.float32, bitcast=True) * FP8_BF16_REBIAS
    value = (value.to(tl.int32, bitcast=True) >> 16).to(tl.int16)
    return value.to(tl.bfloat16, bitcast=True)


@triton.jit
def fold_fp8_nan_flags(flags, ROWS: tl.constexpr, WORDS_K: tl.constexpr):
    """Fold SWAR E4M3FN NaN flags 4x along WORDS_K with zero-cost register splits."""
    if WORDS_K >= 4:
        lo, hi = tl.split(tl.reshape(flags, (ROWS, WORDS_K // 2, 2)))
        flags = lo | hi
        lo, hi = tl.split(tl.reshape(flags, (ROWS, WORDS_K // 4, 2)))
        flags = lo | hi
    return flags


@triton.jit
def fold_fp8_nan_words(word, ROWS: tl.constexpr, WORDS_K: tl.constexpr):
    """SWAR E4M3FN NaN detector: bit 7 of each byte becomes 1 iff 0x7f or 0xff."""
    return fold_fp8_nan_flags((word & 0x7F7F7F7F) + 0x01010101, ROWS, WORDS_K)


@triton.jit
def decode_fp8_e4m3_words(
    word,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    compute_type: tl.constexpr,
    PROPAGATE_NAN: tl.constexpr = False,
):
    """int32 words of four K-consecutive E4M3FN bytes to [ROWS, COLS] values.

    Finite codes are exact; 0x7f / 0xff map to +/-480 (BF16) or +/-1.875 (FP16)
    unless PROPAGATE_NAN handles them during FP16 unpacking. Other paths use
    fold_fp8_nan_words to propagate NaNs.
    """
    if compute_type == tl.bfloat16:
        # Arithmetic shifts; 0x87F00000 keeps sign and magnitude without rebias.
        byte0 = fp32_bits_to_bf16(((word << 24) >> 4) & -0x78100000)
        byte1 = fp32_bits_to_bf16(((word << 16) >> 4) & -0x78100000)
        byte2 = fp32_bits_to_bf16(((word << 8) >> 4) & -0x78100000)
        byte3 = fp32_bits_to_bf16((word >> 4) & -0x78100000)
    else:
        sign_odd = word & -0x7FFF8000
        sign_even = (word & 0x00800080) << 8
        even = ((word & 0x007F007F) << 7) | sign_even
        odd = ((word >> 1) & 0x3F803F80) | sign_odd
        if PROPAGATE_NAN:
            # For each FP16 halfword only E4M3FN magnitude 0x7f carries into bit 14.
            even |= (even + 0x00800080) & 0x40004000
            odd |= (odd + 0x00800080) & 0x40004000
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
    word,
    activation,
    scale,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    compute_type: tl.constexpr,
    WEIGHT_BITS: tl.constexpr,
):
    """FP32 [ROWS] products of one K tile with FP32 activation [1, COLS]."""
    if WEIGHT_BITS == 4:
        code = decode_uint4b8_words(word, ROWS, COLS, compute_type)
        bias = INT4_BF16_BIAS if compute_type == tl.bfloat16 else INT4_FP16_BIAS
        product = tl.sum(code.to(tl.float32) * activation, axis=1)
        product -= bias * tl.sum(activation, axis=1)
    else:
        code = decode_fp8_e4m3_words(word, ROWS, COLS, tl.float16).to(tl.float32)
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
    nan_flag_ptr,
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
    SORTED_ROW_GROUP: tl.constexpr = 0,
):
    """B is UINT4B8 or E4M3FN viewed as int32 words. Routes are padded per
    expert to ALIGN_BLOCK_M, a multiple of BLOCK_SIZE_M."""
    compute_type = a_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    pid_m = pid // (N // (BLOCK_SIZE_N * N_TILES))
    pid_n = pid % (N // (BLOCK_SIZE_N * N_TILES))
    if SORTED_ROW_GROUP:
        # Adjacent row blocks reuse the same expert's W2 columns.
        column_tiles: tl.constexpr = N // BLOCK_SIZE_N
        group = pid // (SORTED_ROW_GROUP * column_tiles)
        pid_m = group * SORTED_ROW_GROUP + pid % SORTED_ROW_GROUP
        pid_n = (pid // SORTED_ROW_GROUP) % column_tiles
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
    if SORTED_ROW_GROUP:
        a_ptrs = (
            a_ptr
            + (offs_token_id // 64)[:, None] * (64 * K)
            + (offs_token_id % 64)[:, None] * BLOCK_SIZE_K
            + offs_k[None, :]
        )
        c_ptrs = c_ptr + offs_token_id[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE_N)
    # PRESUM: a_sum_ptr holds [rows, K // BLOCK_SIZE_K] activation sums.
    a_sum_ptrs = a_sum_ptr + (offs_token // top_k) * (K // BLOCK_SIZE_K)
    words_k: tl.constexpr = BLOCK_SIZE_K * WEIGHT_BITS // 32
    nan_cols: tl.constexpr = words_k // 4 if words_k >= 4 else words_k
    # K-contiguous words load faster than a K-major tile.
    b_ptrs = b_ptr + expert * stride_be + tl.arange(0, words_k)[None, :]
    scale_ptrs = b_scale_ptr + expert * stride_bse
    if MUL_ROUTED_WEIGHT:
        if SORTED_ROW_GROUP:
            route = tl.load(topk_weights_ptr + offs_token_id)
        else:
            route = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
    num_k_tiles: tl.constexpr = K // BLOCK_SIZE_K
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if N_TILES == 1:
        # Pointer-increment loop; measurably faster than the flat loop below.
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        b_ptrs += offs_n[:, None] * stride_bn
        scale_ptrs += offs_n * stride_bsn
        if WEIGHT_BITS == 8:
            nan_acc = tl.zeros((BLOCK_SIZE_N, nan_cols), dtype=tl.int32)
        for tile in tl.range(num_k_tiles):
            if SORTED_ROW_GROUP:
                activation = tl.load(a_ptrs)
            else:
                activation = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            scale = tl.load(scale_ptrs + tile * BLOCK_SIZE_K // GROUP_SIZE)
            word = tl.load(b_ptrs)
            if WEIGHT_BITS == 8:
                nan_acc |= fold_fp8_nan_words(word, BLOCK_SIZE_N, words_k)
            row_sum = 0.0
            if WEIGHT_BITS == 4:
                row_sum = tile_row_sum(
                    activation, a_sum_ptrs + tile, token_mask, PRESUM
                )
            accumulator += scaled_partial(
                activation,
                row_sum,
                word,
                scale.to(tl.float32),
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                compute_type,
                WEIGHT_BITS,
            )
            a_ptrs += BLOCK_SIZE_K * (64 if SORTED_ROW_GROUP else 1)
            b_ptrs += words_k
        if WEIGHT_BITS == 8:
            col_has_nan = (
                tl.sum(((nan_acc & -0x7F7F7F80) != 0).to(tl.int32), axis=1) != 0
            )
            accumulator = tl.where(col_has_nan[None, :], float("nan"), accumulator)
        if MUL_ROUTED_WEIGHT:
            accumulator = accumulator * route[:, None]
        if SORTED_ROW_GROUP:
            tl.store(c_ptrs + pid_n * BLOCK_SIZE_N, accumulator.to(compute_type))
            return
        tl.store(
            c_ptrs + pid_n * BLOCK_SIZE_N,
            accumulator.to(compute_type),
            mask=token_mask[:, None],
        )
        return
    if num_k_tiles == 2:
        # Short-K (e.g. K=256, BK=128): hoist both activation tiles out of
        # the N_TILES loop; FP8 records a single NaN flag for the cold fixup pass.
        act0 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        act1 = tl.load(a_ptrs + BLOCK_SIZE_K, mask=token_mask[:, None], other=0.0)
        rs0 = 0.0
        rs1 = 0.0
        if WEIGHT_BITS == 4:
            rs0 = tile_row_sum(act0, a_sum_ptrs, token_mask, PRESUM)
            rs1 = tile_row_sum(act1, a_sum_ptrs + 1, token_mask, PRESUM)
        else:
            nan_any = tl.zeros((BLOCK_SIZE_N, nan_cols), dtype=tl.int32)
        group1: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
        for n_tile in tl.range(N_TILES):
            offs_n = (pid_n * N_TILES + n_tile) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            )
            w0 = tl.load(b_ptrs + offs_n[:, None] * stride_bn)
            sc0 = tl.load(scale_ptrs + offs_n * stride_bsn).to(tl.float32)
            acc = scaled_partial(
                act0,
                rs0,
                w0,
                sc0,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                compute_type,
                WEIGHT_BITS,
            )
            w1 = tl.load(b_ptrs + offs_n[:, None] * stride_bn + words_k)
            sc1 = tl.load(scale_ptrs + offs_n * stride_bsn + group1).to(tl.float32)
            acc += scaled_partial(
                act1,
                rs1,
                w1,
                sc1,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                compute_type,
                WEIGHT_BITS,
            )
            if WEIGHT_BITS == 8:
                raw_nan = ((w0 & 0x7F7F7F7F) + 0x01010101) | (
                    (w1 & 0x7F7F7F7F) + 0x01010101
                )
                nan_any |= fold_fp8_nan_flags(raw_nan, BLOCK_SIZE_N, words_k)
            result = acc * route[:, None] if MUL_ROUTED_WEIGHT else acc
            n_start = (pid_n * N_TILES + n_tile) * BLOCK_SIZE_N
            tl.store(
                c_ptrs + n_start, result.to(compute_type), mask=token_mask[:, None]
            )
        if WEIGHT_BITS == 8:
            if tl.sum(((nan_any & -0x7F7F7F80) != 0).to(tl.int32)) != 0:
                tl.atomic_or(nan_flag_ptr, 1, sem="relaxed", scope="gpu")
        return
    if WEIGHT_BITS == 8:
        nan_any = tl.zeros((BLOCK_SIZE_N, nan_cols), dtype=tl.int32)
    for step in tl.range(N_TILES * num_k_tiles):
        n_tile = step // num_k_tiles
        tile = step % num_k_tiles
        offs_n = (pid_n * N_TILES + n_tile) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        activation = tl.load(
            a_ptrs + tile * BLOCK_SIZE_K, mask=token_mask[:, None], other=0.0
        )
        word = tl.load(b_ptrs + offs_n[:, None] * stride_bn + tile * words_k)
        if WEIGHT_BITS == 8:
            nan_any |= fold_fp8_nan_words(word, BLOCK_SIZE_N, words_k)
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
    if WEIGHT_BITS == 8:
        if tl.sum(((nan_any & -0x7F7F7F80) != 0).to(tl.int32)) != 0:
            tl.atomic_or(nan_flag_ptr, 1, sem="relaxed", scope="gpu")


@triton.jit
def fp8_nan_fixup_kernel(
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    nan_flag_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_valid_tokens,
    num_programs,
    stride_be,
    stride_bn,
    stride_cm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    N_TILES: tl.constexpr,
    ALIGN_BLOCK_M: tl.constexpr,
):
    """Cold single-CTA pass for N_TILES > 1 FP8: returns immediately unless any
    weight byte was 0x7f or 0xff, then propagates NaN to affected output columns."""
    if tl.load(nan_flag_ptr) == 0:
        return
    words_k: tl.constexpr = BLOCK_SIZE_K // 4
    num_k_tiles: tl.constexpr = K // BLOCK_SIZE_K
    num_pid_n: tl.constexpr = N // (BLOCK_SIZE_N * N_TILES)
    num_post_padded = tl.load(num_tokens_post_padded_ptr)
    nan_val = tl.full(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), float("nan"), dtype=c_ptr.dtype.element_ty
    )
    for pid in tl.range(num_programs):
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        if pid_m * BLOCK_SIZE_M < num_post_padded:
            offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
            token_mask = offs_token < num_valid_tokens
            expert = tl.load(
                expert_ids_ptr + pid_m // (ALIGN_BLOCK_M // BLOCK_SIZE_M)
            ).to(tl.int64)
            b_base = b_ptr + expert * stride_be + tl.arange(0, words_k)[None, :]
            c_base = (
                c_ptr + stride_cm * offs_token[:, None] + tl.arange(0, BLOCK_SIZE_N)
            )
            for n_tile in tl.range(N_TILES):
                offs_n = (pid_n * N_TILES + n_tile) * BLOCK_SIZE_N + tl.arange(
                    0, BLOCK_SIZE_N
                )
                tile_nan = tl.zeros((BLOCK_SIZE_N, words_k), dtype=tl.int32)
                for tile in tl.range(num_k_tiles):
                    word = tl.load(
                        b_base + offs_n[:, None] * stride_bn + tile * words_k
                    )
                    tile_nan |= (word & 0x7F7F7F7F) + 0x01010101
                col_nan = (
                    tl.sum(((tile_nan & -0x7F7F7F80) != 0).to(tl.int32), axis=1) != 0
                )
                tl.store(
                    c_base + (pid_n * N_TILES + n_tile) * BLOCK_SIZE_N,
                    nan_val,
                    mask=token_mask[:, None] & col_nan[None, :],
                )
    tl.store(nan_flag_ptr, 0)


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
    SPLIT_K: tl.constexpr = 1,
):
    """FIRST: SiLU(gate) * up of one route. Otherwise the sum of one token's
    top-k down projections, which replaces moe_sum. Both INT4 and FP8 weights
    are loaded as int32 words."""
    compute_type = a_ptr.dtype.element_ty
    row = tl.program_id(axis=0).to(tl.int64)
    offs_n = tl.program_id(axis=1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    w_step: tl.constexpr = BLOCK_SIZE_K * WEIGHT_BITS // 32
    nan_cols: tl.constexpr = w_step // 4 if w_step >= 4 else w_step
    total = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    if WEIGHT_BITS == 8:
        nan_acc = tl.zeros((BLOCK_SIZE_N, nan_cols), dtype=tl.int32)
    for slot in tl.static_range(1 if FIRST else top_k):
        route = row if FIRST else row * top_k + slot
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        w_ptrs = w_ptr + expert * stride_we + offs_n[:, None] * stride_wn
        w_ptrs += tl.arange(0, w_step)[None, :]
        scale_ptrs = scale_ptr + expert * stride_se + offs_n * stride_sn
        a_ptrs = a_ptr + (row // top_k if FIRST else route) * K + offs_k
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        up = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        if SPLIT_K > 1:
            split = tl.program_id(2)
            first_k = split * (K // SPLIT_K)
            a_ptrs += first_k
            w_ptrs += first_k // 4
            scale_ptrs += first_k // GROUP_SIZE
        for tile in tl.range(K // (SPLIT_K * BLOCK_SIZE_K)):
            group = tile * BLOCK_SIZE_K // GROUP_SIZE
            a = tl.load(a_ptrs).to(tl.float32)[None, :]
            scale = tl.load(scale_ptrs + group).to(tl.float32)
            word = tl.load(w_ptrs)
            if WEIGHT_BITS == 8:
                nan_acc |= fold_fp8_nan_words(word, BLOCK_SIZE_N, w_step)
            acc += gemv_partial(
                word, a, scale, BLOCK_SIZE_N, BLOCK_SIZE_K, compute_type, WEIGHT_BITS
            )
            if FIRST:
                scale_up = tl.load(scale_ptrs + N * stride_sn + group).to(tl.float32)
                word_up = tl.load(w_ptrs + N * stride_wn)
                if WEIGHT_BITS == 8:
                    nan_acc |= fold_fp8_nan_words(word_up, BLOCK_SIZE_N, w_step)
                up += gemv_partial(
                    word_up,
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
        if SPLIT_K > 1:
            has_nan = tl.sum(((nan_acc & -0x7F7F7F80) != 0).to(tl.int32), axis=1) != 0
            base = out_ptr + (row * SPLIT_K + split) * (2 * N) + offs_n
            tl.store(base, tl.where(has_nan, float("nan"), acc))
            tl.store(base + N, up)
            return
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
    if WEIGHT_BITS == 8:
        has_nan = tl.sum(((nan_acc & -0x7F7F7F80) != 0).to(tl.int32), axis=1) != 0
        total = tl.where(has_nan, float("nan"), total)
    tl.store(out_ptr + row * N + offs_n, total.to(compute_type))


@triton.jit
def combine_fp8_gemv_gate_up(
    partial_ptr, weights_ptr, out_ptr, N: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.arange(0, SPLIT_K)
    base = partial_ptr + (row * SPLIT_K + split[:, None]) * (2 * N) + col[None, :]
    gate = tl.sum(tl.load(base), axis=0)
    up = tl.sum(tl.load(base + N), axis=0)
    if MUL_ROUTED_WEIGHT:
        route = tl.load(weights_ptr + row)
        gate *= route
        up *= route
    dtype: tl.constexpr = out_ptr.dtype.element_ty
    gate = gate.to(dtype).to(tl.float32)
    up = up.to(dtype).to(tl.float32)
    tl.store(out_ptr + row * N + col, gate / (1.0 + tl.exp(-gate)) * up)


@triton.jit
def dequantize_fp8_kernel(
    weight_ptr,
    scale_ptr,
    out_ptr,
    N_TILE: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Dequantize a contiguous [E_chunk, N_TILE, K] E4M3FN slice to FP16/BF16."""
    expert = tl.program_id(axis=0).to(tl.int64)
    n_idx = tl.program_id(axis=1)
    k_idx = tl.program_id(axis=2)
    offs_n = n_idx * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    rows = expert * N_TILE + offs_n
    code = decode_fp8_e4m3(tl.load(weight_ptr + rows[:, None] * K + offs_k[None, :]))
    group = k_idx * BLOCK_K // GROUP_SIZE
    scale = (
        tl.load(scale_ptr + rows * (K // GROUP_SIZE) + group).to(tl.float32)
        * FP8_DECODE_SCALE
    )
    value = code.to(tl.float32) * scale[:, None]
    tl.store(
        out_ptr + rows[:, None] * K + offs_k[None, :],
        value.to(out_ptr.dtype.element_ty),
    )


@triton.jit
def pack_fp8_gate_up_kernel(
    weight_ptr,
    scale_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    FUSE_SORTED: tl.constexpr = False,
    PACK_WORDS: tl.constexpr = False,
):
    """Decode a live expert slice into contiguous GEMM tiles."""
    pid = tl.program_id(0)
    expert = tl.program_id(1).to(tl.int64)
    n = (pid // (K // COLS)) * ROWS + tl.arange(0, ROWS)
    k_start = (pid % (K // COLS)) * COLS
    k = k_start + tl.arange(0, COLS)
    source_n = n
    if PACK_WORDS and FUSE_SORTED:
        # Traverse the packed destination in order and read its gate/up partner.
        half: tl.constexpr = TILE_N // 2
        source_n = n // TILE_N * half + n % half + (n % TILE_N // half) * (N // 2)
    row = expert * N + source_n
    scale = tl.load(scale_ptr + row * (K // GROUP_SIZE) + k_start // GROUP_SIZE)
    raw = tl.load(weight_ptr + row[:, None] * K + k[None, :])
    if PACK_WORDS:
        # E4M3FN has 128 magnitude codes. Decode this constant vector once
        # per program, then use its entries for every row of the weight tile.
        index = tl.arange(0, 128)
        normal = ((index << 20) + 0x3c000000).to(tl.float32, bitcast=True)
        table = tl.where(index < 8, index.to(tl.float32) * (2.0 ** -9), normal)
        table = tl.where(index == 127, float("nan"), table)
        magnitude = tl.reshape((raw & 127).to(tl.int32), (ROWS * COLS,))
        decoded = tl.reshape(tl.gather(table, magnitude, axis=0), (ROWS, COLS))

        scale32 = scale.to(tl.float32)
        scale_hi = (scale32.to(tl.uint32, bitcast=True) >> 16).to(tl.uint16)
        sign = ((raw.to(tl.uint16) << 8) ^ scale_hi[:, None]) & 0x8000
        scale_mag = tl.where(scale32 != scale32, float("nan"), tl.abs(scale32))
        bits = (decoded * scale_mag[:, None]).to(tl.uint32, bitcast=True)
        rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).to(tl.uint16)
        values = (rounded | sign).to(tl.bfloat16, bitcast=True)
    else:
        code = decode_fp8_e4m3(raw)
        values = code.to(tl.float32) * (scale.to(tl.float32) * FP8_DECODE_SCALE)[:, None]
    if FUSE_SORTED and not PACK_WORDS:
        # Place the matching gate and up columns in the two halves of one tile.
        half: tl.constexpr = TILE_N // 2
        channel = n % (N // 2)
        n = channel // half * TILE_N + (n // (N // 2)) * half + channel % half
    tile = (n // TILE_N)[:, None] * (K // TILE_K) + (k // TILE_K)[None, :]
    within = (n % TILE_N)[:, None] * TILE_K + (k % TILE_K)[None, :]
    tl.store(out_ptr + expert * N * K + tile * TILE_N * TILE_K + within, values)


@triton.jit
def dense_fp8_gate_up_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    pid_m,
    pid_n,
    N: tl.constexpr,
    K: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_be,
    stride_cm,
    e_start,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    FUSE_SORTED: tl.constexpr = False,
):
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    local_e = expert - e_start
    offs_token = tl.load(
        sorted_token_ids_ptr
        + pid_m * BLOCK_SIZE_M
        + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    ).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :]
    b_ptrs = (
        b_ptr
        + local_e * stride_be
        + pid_n * BLOCK_SIZE_N * K
        + tl.arange(0, BLOCK_SIZE_N)[:, None] * BLOCK_SIZE_K
        + offs_k[None, :]
    )
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in tl.range(K // BLOCK_SIZE_K):
        act = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        weight = tl.load(b_ptrs)
        acc = tl.dot(act, tl.trans(weight), acc, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_N * BLOCK_SIZE_K
    if MUL_ROUTED_WEIGHT:
        route = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc *= route[:, None]
    if FUSE_SORTED:
        half: tl.constexpr = BLOCK_SIZE_N // 2
        pair = tl.permute(tl.reshape(acc, (BLOCK_SIZE_M, 2, half)), (0, 2, 1))
        gate, up = tl.split(pair)
        gate = gate.to(a_ptr.dtype.element_ty).to(tl.float32)
        up = up.to(a_ptr.dtype.element_ty).to(tl.float32)
        activated = tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up
        row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        # W2 consumes contiguous tiles of 64 routes by 64 input channels.
        offset = (
            (row // 64)[:, None] * (64 * stride_cm)
            + pid_n * (64 * half)
            + (row % 64)[:, None] * half
            + tl.arange(0, half)[None, :]
        )
        tl.store(c_ptr + offset, activated)
        return
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(a_ptr.dtype.element_ty), mask=token_mask[:, None])


@triton.jit
def dense_fp8_gate_up_chunk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    expert_offsets_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_be,
    stride_cm,
    e_start,
    e_end,
    START_TILE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    FUSE_SORTED: tl.constexpr = False,
    ROW_GROUP: tl.constexpr = 1,
):
    """Dense BF16/FP16 gate/up GEMM for an expert slice [e_start, e_end)."""
    first = tl.load(expert_offsets_ptr + e_start) // BLOCK_SIZE_M
    end = tl.load(expert_offsets_ptr + e_end) // BLOCK_SIZE_M
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile = START_TILE + tl.program_id(0)
    if ROW_GROUP > 1:
        group_start = tile // (ROW_GROUP * num_pid_n) * ROW_GROUP
        row_tile = group_start + tile % ROW_GROUP
        col_tile = tile // ROW_GROUP % num_pid_n
        tile = row_tile * num_pid_n + col_tile
    limit = (end - first) * num_pid_n
    if tile >= limit:
        return
    if START_TILE == 0:
        dense_fp8_gate_up_tile(
            a_ptr,
            b_ptr,
            c_ptr,
            topk_weights_ptr,
            sorted_token_ids_ptr,
            expert_ids_ptr,
            first + tile // num_pid_n,
            tile % num_pid_n,
            N,
            K,
            num_valid_tokens,
            stride_am,
            stride_be,
            stride_cm,
            e_start,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            top_k,
            MUL_ROUTED_WEIGHT,
            FUSE_SORTED,
        )
    else:
        while tile < limit:
            dense_fp8_gate_up_tile(
                a_ptr,
                b_ptr,
                c_ptr,
                topk_weights_ptr,
                sorted_token_ids_ptr,
                expert_ids_ptr,
                first + tile // num_pid_n,
                tile % num_pid_n,
                N,
                K,
                num_valid_tokens,
                stride_am,
                stride_be,
                stride_cm,
                e_start,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                top_k,
                MUL_ROUTED_WEIGHT,
                FUSE_SORTED,
            )
            tile += tl.num_programs(0)


@triton.jit
def packed_sorted_down_kernel(
    activated, weight, routed, sorted_weights, experts, total,
    WIDTH: tl.constexpr, HIDDEN: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    """Multiply sorted 64x64 activation panels by packed BF16 expert weights."""
    pid = tl.program_id(0)
    pid_m = pid // (WIDTH // BN)
    pid_n = pid % (WIDTH // BN)
    if pid_m * BM >= tl.load(total):
        return
    expert = tl.load(experts + pid_m)
    row = pid_m * BM + tl.arange(0, BM)
    col = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a = activated + (row // 64)[:, None] * (64 * K) + (row % 64)[:, None] * BK + k[None, :]
    b = weight + expert * (HIDDEN * K) + pid_n * (BN * K)
    b += tl.arange(0, BN)[:, None] * BK + k[None, :]
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in tl.range(K // BK):
        act = tl.load(a)
        w = tl.load(b)
        acc = tl.dot(act, tl.trans(w), acc)
        a += 64 * BK
        b += BN * BK
    if MUL_ROUTED_WEIGHT:
        acc *= tl.load(sorted_weights + row)[:, None]
    tl.store(routed + row[:, None] * WIDTH + col[None, :], acc)


@triton.jit
def index_sorted_routes_kernel(
    ids, total, inverse, weights, sorted_weights, num_routes, BLOCK_SIZE: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = pos < tl.load(total)
    route = tl.load(ids + pos, mask=valid, other=num_routes)
    tl.store(inverse + route, pos, mask=route < num_routes)
    weight = tl.load(weights + route, mask=route < num_routes, other=0.0)
    tl.store(sorted_weights + pos, weight, mask=valid)


@triton.jit
def strided_sum_routes_kernel(
    routed_ptr,
    output_ptr,
    width: tl.constexpr,
    output_stride_m,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    inverse=None,
):
    token_idx = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < width
    base = routed_ptr + token_idx * TOPK * width + offs
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for slot in tl.static_range(0, TOPK):
        if inverse is None:
            src = base + slot * width
        else:
            row = tl.load(inverse + token_idx * TOPK + slot)
            src = routed_ptr + row * width + offs
        acc += tl.load(src, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + token_idx * output_stride_m + offs,
        acc.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def sum_routes(routed, output):
    """Reduce contiguous [tokens, top_k, width] routes into [tokens, width] output.

    Contiguous outputs reuse MetaX's unrolled moe_sum kernel; column slices use
    a stride-aware kernel so large down projections can tile along hidden_size.
    """
    num_tokens, top_k, width = routed.shape
    if output.is_contiguous():
        _qwen_moe_sum_kernel[(num_tokens, triton.cdiv(width, 2048))](
            routed,
            output,
            routed,
            0,
            0,
            num_tokens,
            width,
            TOPK=top_k,
            APPLY_ROUTER_WEIGHT=False,
            BLOCK_SIZE=2048,
            num_warps=8,
        )
    else:
        block_size = 1024 if width <= 1024 else 2048
        strided_sum_routes_kernel[(num_tokens, triton.cdiv(width, block_size))](
            routed,
            output,
            width,
            output.stride(0),
            TOPK=top_k,
            BLOCK_SIZE=block_size,
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
    alignment, _ = align_routes_with_offsets(topk_ids, block_m, num_experts)
    return alignment


def align_routes_with_offsets(topk_ids, block_m, num_experts):
    # Zero the aligner's cumsum and count buffers with Triton, not Torch.
    workspace = topk_ids.new_empty(((num_experts + 1) ** 2,), dtype=torch.int32)
    zero_workspace_kernel[(triton.cdiv(workspace.numel(), 1024),)](
        workspace, workspace.numel(), BLOCK=1024
    )
    cumsum, counts = workspace.split([num_experts + 1, num_experts * (num_experts + 1)])
    alignment = moe_align_block_size_no_tle(
        topk_ids, block_m, num_experts, workspace=(cumsum, counts)
    )
    return alignment, cumsum


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
    # Element 1 doubles as the FP8 N_TILES > 1 NaN flag, zeroed inside the graph.
    tl.store(num_tokens_post_pad_ptr + 1, 0)

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
    num_tokens_post_pad = topk_ids.new_empty((2,), dtype=torch.int32)
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
    sorted_weights=None,
):
    out_features, reduction = weight.shape[1], activation.shape[1]
    num_routes = topk_weights.numel()
    problem_m = min(alignment[0].shape[0], num_routes * align_block_m)
    n_tiles = tile.n_tiles if reduction <= N_TILES_MAX_K else 1
    while out_features % (tile.block_n * n_tiles):
        n_tiles //= 2
    is_int4 = weight.dtype == torch.uint8
    presum = (
        is_int4 and num_routes <= INT4_PRESUM_MAX_ROUTES_PER_EXPERT * weight.shape[0]
    )
    activation_sums = activation
    if presum:
        rows = activation.shape[0]
        chunks = reduction // tile.block_k
        activation_sums = activation.new_empty((rows, chunks), dtype=torch.float32)
        chunk_sum_kernel[(rows, chunks)](
            activation, activation_sums, reduction, CHUNK=tile.block_k
        )
    need_nan_fixup = (not is_int4) and n_tiles > 1
    if need_nan_fixup:
        if alignment[2].numel() >= 2:
            nan_flag = alignment[2][1:]
        else:
            nan_flag = activation.new_zeros((1,), dtype=torch.int32)
    else:
        nan_flag = alignment[2]
    row_tiles = triton.cdiv(problem_m, tile.block_m)
    sorted_row_group = 8 if sorted_weights is not None else 0
    if sorted_row_group:
        row_tiles = triton.cdiv(row_tiles, sorted_row_group) * sorted_row_group
    num_programs = row_tiles * (out_features // (tile.block_n * n_tiles))
    weight_words = weight.view(torch.int32)
    moe_gemm_kernel[(num_programs,)](
        activation,
        weight_words,
        output,
        scale,
        activation_sums,
        topk_weights if sorted_weights is None else sorted_weights,
        *alignment,
        nan_flag,
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
        WEIGHT_BITS=4 if is_int4 else 8,
        PRESUM=presum,
        SORTED_ROW_GROUP=sorted_row_group,
        num_warps=tile.num_warps,
        num_stages=tile.num_stages,
        pipeline=tile.pipeline,
        scenario=tile.scenario,
    )
    if need_nan_fixup:
        fp8_nan_fixup_kernel[(1,)](
            weight_words,
            output,
            *alignment,
            nan_flag,
            out_features,
            reduction,
            num_routes,
            num_programs,
            weight.stride(0) // 4,
            weight.stride(1) // 4,
            output.stride(-2),
            BLOCK_SIZE_M=tile.block_m,
            BLOCK_SIZE_N=tile.block_n,
            BLOCK_SIZE_K=tile.block_k,
            N_TILES=n_tiles,
            ALIGN_BLOCK_M=align_block_m,
            num_warps=4,
        )


def launch_gemv(activation, weight, scale, topk_ids, topk_weights, output, **kw):
    num_rows, width = output.shape
    is_int4 = weight.dtype == torch.uint8
    if is_int4:
        # Narrower tiles keep the device busy when there are few outputs.
        block_n = 16 if num_rows * width < GEMV_NARROW_MAX_OUTPUTS else 32
    else:
        if num_rows * width >= 6144 and width % 64 == 0:
            block_n = 64
        elif width % 32 == 0:
            block_n = 32
        else:
            block_n = 16
    # A handful of routes leaves the long gate/up reduction underoccupied.
    split_k = 1
    if (
        not is_int4 and kw["FIRST"] and activation.dtype == torch.bfloat16
        and activation.shape[0] <= 4 and activation.shape[1] == 4096
        and width == 256 and kw["top_k"] == 6 and kw["GROUP_SIZE"] == 128
    ):
        split_k = 4
    partial = (
        output.new_empty((num_rows, split_k, 2, width), dtype=torch.float32)
        if split_k > 1 else output
    )
    weight = weight.view(torch.int32)
    moe_gemv_kernel[(num_rows, width // block_n, split_k)](
        activation,
        weight,
        scale,
        topk_ids,
        topk_weights,
        partial,
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
        SPLIT_K=split_k,
        **kw,
    )
    if split_k > 1:
        combine_fp8_gemv_gate_up[(num_rows, width // block_n)](
            partial, topk_weights, output, width, split_k, block_n, kw["MUL_ROUTED_WEIGHT"]
        )


def choose_balanced_tile(total_cols, max_cols, align):
    if max_cols < align:
        return 0
    capped = min(total_cols, (max_cols // align) * align)
    chunks = triton.cdiv(total_cols, capped)
    return min(total_cols, triton.cdiv(triton.cdiv(total_cols, chunks), align) * align)


def dequantize_fp8(weight_slice, scale_slice, group_size, output):
    num_experts, tile_rows, reduction = weight_slice.shape
    block_k = 128
    grid = (
        num_experts,
        tile_rows // FP8_DEQUANT_BLOCK_ROWS,
        reduction // block_k,
    )
    dequantize_fp8_kernel[grid](
        weight_slice.view(torch.uint8),
        scale_slice,
        output,
        N_TILE=tile_rows,
        K=reduction,
        GROUP_SIZE=group_size,
        BLOCK_ROWS=FP8_DEQUANT_BLOCK_ROWS,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


def run_dense_fp8_gate_up(
    hs,
    w1,
    s1,
    gate_up,
    topk_weights,
    alignment,
    *,
    expert_offsets,
    w1_experts,
    group_size,
    top_k,
    router_on_input,
    fuse_sorted=False,
):
    num_tokens, hidden_size = hs.shape
    num_experts, fused_intermediate, _ = w1.shape
    num_routes = topk_weights.numel()
    compute_type = tl.float16 if hs.dtype == torch.float16 else tl.bfloat16
    w1_scratch = hs.new_empty((w1_experts * fused_intermediate * hidden_size,))
    if w1_experts == num_experts:
        w1_buf = w1_scratch.view(num_experts, fused_intermediate, hidden_size)
        dequantize_fp8(w1, s1, group_size, w1_buf)
        metax_fused_moe.invoke_fused_moe_triton_kernel(
            hs,
            w1_buf,
            gate_up.view(num_tokens, top_k, fused_intermediate),
            None,
            None,
            topk_weights,
            *alignment,
            router_on_input,
            top_k,
            FP8_DENSE_GATE_UP,
            compute_type,
        )
        return
    block_m = FP8_DENSE_GATE_UP["BLOCK_SIZE_M"]
    block_n = FP8_DENSE_GATE_UP["BLOCK_SIZE_N"]
    block_k = FP8_DENSE_GATE_UP["BLOCK_SIZE_K"]
    column_tiles = fused_intermediate // block_n
    if fuse_sorted:
        # Each expert half can receive all routes when a captured graph replays.
        # Schedule four adjacent row tiles per column to reuse expert weights.
        row_group = 4 if num_tokens == 16384 else 1
        row_tiles = triton.cdiv(alignment[1].numel(), row_group) * row_group
        launches = ((0, row_tiles * column_tiles),)
    else:
        main_tiles = (
            triton.cdiv(alignment[1].numel(), triton.cdiv(num_experts, w1_experts))
            * column_tiles
        )
        launches = ((0, main_tiles), (main_tiles, min(main_tiles, 256)))
    # Load a full quantization segment per row for the common group-128 path.
    pack_rows, pack_cols = 64, 128
    for e_start in range(0, num_experts, w1_experts):
        e_end = min(e_start + w1_experts, num_experts)
        chunk_e = e_end - e_start
        w1_buf = w1_scratch[: chunk_e * fused_intermediate * hidden_size].view(
            chunk_e, fused_intermediate, hidden_size
        )
        pack_fp8_gate_up_kernel[
            ((fused_intermediate // pack_rows) * (hidden_size // pack_cols), chunk_e)
        ](
            w1[e_start:e_end].view(torch.uint8),
            s1[e_start:e_end],
            w1_buf,
            fused_intermediate,
            hidden_size,
            group_size,
            block_n,
            block_k,
            FUSE_SORTED=fuse_sorted,
            ROWS=pack_rows,
            COLS=pack_cols,
            PACK_WORDS=fuse_sorted and num_tokens == 16384,
            num_warps=4,
        )
        for start_tile, workers in launches:
            dense_fp8_gate_up_chunk_kernel[(workers,)](
                hs,
                w1_buf,
                gate_up,
                topk_weights,
                alignment[0],
                alignment[1],
                expert_offsets,
                fused_intermediate,
                hidden_size,
                num_routes,
                hs.stride(0),
                w1_buf.stride(0),
                gate_up.stride(0),
                e_start,
                e_end,
                FUSE_SORTED=fuse_sorted,
                START_TILE=start_tile,
                ROW_GROUP=row_group if fuse_sorted else 1,
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                top_k=top_k,
                MUL_ROUTED_WEIGHT=router_on_input,
                num_warps=FP8_DENSE_GATE_UP["num_warps"],
                num_stages=FP8_DENSE_GATE_UP["num_stages"],
                pipeline=FP8_DENSE_GATE_UP["pipeline"],
            )


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
    gemv_max_routes = (
        INT4_GEMV_MAX_ROUTES_PER_EXPERT if is_int4 else FP8_GEMV_MAX_ROUTES_PER_EXPERT
    )
    if routes_per_expert <= gemv_max_routes:
        activated = hs.new_empty((num_routes, intermediate_size))
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
    elem_size = hs.element_size()
    output_bytes = num_tokens * hidden_size * elem_size
    gate_up_bytes = num_routes * fused_intermediate * elem_size
    dense_block_m = FP8_DENSE_GATE_UP["BLOCK_SIZE_M"]
    align_capacity = min(
        num_routes * dense_block_m, num_routes + num_experts * (dense_block_m - 1)
    )
    align_bytes = 4 * (
        align_capacity
        + triton.cdiv(align_capacity, dense_block_m)
        + (num_experts + 1) ** 2
        + 1
    )
    w1_experts = 0
    if (
        not is_int4
        and routes_per_expert > FP8_DENSE_MIN_ROUTES_PER_EXPERT
        and fused_intermediate % FP8_DENSE_GATE_UP["BLOCK_SIZE_N"] == 0
    ):
        w1_budget = (
            FP8_DENSE_MAX_WORKSPACE_BYTES - output_bytes - gate_up_bytes - align_bytes
        )
        w1_expert_bytes = fused_intermediate * hidden_size * elem_size
        w1_experts = choose_balanced_tile(num_experts, w1_budget // w1_expert_bytes, 1)
    if w1_experts > 0:
        align_block_m = dense_block_m
        down_tile_cfg = FP8_DENSE_DOWN
    else:
        if is_int4:
            tiers = INT4_TIERS
        elif intermediate_size <= FP8_NARROW_MAX_INTERMEDIATE:
            tiers = FP8_NARROW_TIERS
        else:
            tiers = WIDE_TIERS
        _, gate_up_tile, down_tile_cfg = next(
            tier for tier in tiers if routes_per_expert <= tier[0]
        )
        align_block_m = max(gate_up_tile.block_m, down_tile_cfg.block_m)
    expert_offsets = None
    if 0 < w1_experts < num_experts:
        alignment, expert_offsets = align_routes_with_offsets(
            topk_ids, align_block_m, num_experts
        )
    else:
        alignment = align_routes(
            topk_ids, align_block_m, num_experts, use_atomic=not is_int4
        )
    fuse_sorted = (
        hs.dtype == torch.bfloat16
        and num_tokens in (13021, 16384)
        and (num_experts, hidden_size, fused_intermediate, top_k) == (256, 4096, 512, 6)
        and group_size == 128
        and w1_experts == 128
        and FP8_DENSE_GATE_UP["BLOCK_SIZE_N"] // 2 == down_tile_cfg.block_k == 64
        and down_tile_cfg.n_tiles == 1
    )
    if fuse_sorted:
        route_rows = alignment[1].numel() * align_block_m
        gate_up = hs.new_empty((route_rows, intermediate_size))
    else:
        route_rows = num_routes
        gate_up = hs.new_empty((route_rows, fused_intermediate))
    common = dict(top_k=top_k, align_block_m=align_block_m, group_size=group_size)
    if w1_experts > 0:
        run_dense_fp8_gate_up(
            hs,
            w1,
            s1,
            gate_up,
            topk_weights,
            alignment,
            expert_offsets=expert_offsets,
            w1_experts=w1_experts,
            fuse_sorted=fuse_sorted,
            group_size=group_size,
            top_k=top_k,
            router_on_input=router_on_input,
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
    # Allocate activated after stage 1 frees any w1 scratch, then free gate_up
    # before allocating the down-projection routing buffer.
    inverse = None
    sorted_weights = None
    if fuse_sorted:
        activated = gate_up
        inverse = topk_ids.new_empty((num_routes,), dtype=torch.int32)
        sorted_weights = topk_weights.new_empty((route_rows,))
        index_sorted_routes_kernel[(triton.cdiv(route_rows, 256),)](
            alignment[0], alignment[2], inverse, topk_weights, sorted_weights,
            num_routes, BLOCK_SIZE=256,
        )
    else:
        activated = hs.new_empty((num_routes, intermediate_size))
        silu_and_mul_out(*gate_up.chunk(2, dim=-1), activated)
    del gate_up
    activated_bytes = route_rows * intermediate_size * elem_size
    if inverse is not None:
        activated_bytes += num_routes * 4 + route_rows * sorted_weights.element_size()
    down_budget = DOWN_ROUTE_MAX_BYTES
    if w1_experts > 0:
        down_budget = min(
            down_budget,
            max(
                0,
                FP8_DENSE_MAX_WORKSPACE_BYTES
                - output_bytes
                - activated_bytes
                - align_bytes,
            ),
        )
    route_col_bytes = route_rows * elem_size
    down_cols = choose_balanced_tile(
        hidden_size, down_budget // route_col_bytes, down_tile_cfg.block_n
    )
    if down_cols == 0:
        down_cols = down_tile_cfg.block_n
    packed_w2 = None
    packed_bytes = w2.numel() * elem_size
    if (
        fuse_sorted and num_tokens == 16384
        and output_bytes + activated_bytes + align_bytes + route_col_bytes * down_cols
        + packed_bytes <= FP8_DENSE_MAX_WORKSPACE_BYTES
    ):
        packed_w2 = hs.new_empty((w2.numel(),))
        pack_fp8_gate_up_kernel[((hidden_size // 64) * (intermediate_size // 128), num_experts)](
            w2.view(torch.uint8), s2, packed_w2, hidden_size, intermediate_size,
            group_size, 128, 64, ROWS=64, COLS=128, PACK_WORDS=True, num_warps=4,
        )
    routed_scratch = hs.new_empty((route_rows * down_cols,))
    common["top_k"] = 1
    for col in range(0, hidden_size, down_cols):
        stop = min(col + down_cols, hidden_size)
        width = stop - col
        # launch_gemm obtains the route stride from the penultimate dimension.
        routed = routed_scratch[: route_rows * width].view(route_rows, 1, width)
        if packed_w2 is not None:
            packed_sorted_down_kernel[(alignment[1].numel() * (width // 128),)](
                activated, packed_w2[col * intermediate_size:], routed, sorted_weights,
                alignment[1], alignment[2], width, hidden_size, intermediate_size,
                128, 128, 64, not router_on_input,
                num_warps=4, num_stages=4, pipeline="cpasync", scenario="outerStage2;unroll",
            )
        else:
            launch_gemm(
                activated,
                w2[:, col:stop],
                s2[:, col:stop],
                routed,
                topk_weights,
                alignment,
                tile=down_tile_cfg,
                sorted_weights=sorted_weights,
                mul_routed_weight=not router_on_input,
                **common,
            )
        output_slice = output if width == hidden_size else output[:, col:stop]
        if fuse_sorted:
            strided_sum_routes_kernel[(num_tokens, triton.cdiv(width, 1024))](
                routed, output_slice, width, output.stride(0), top_k, 1024,
                inverse, num_warps=4,
            )
        else:
            sum_routes(routed.view(num_tokens, top_k, width), output_slice)
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
    # Gate/up consumes all activations before any output write, so hs may alias.
    if (
        output is not None
        and output.numel()
        and any(
            metax_fused_moe._overlaps(output, tensor)
            for tensor in (w1, w2, s1, s2, topk_weights, topk_ids)
        )
    ):
        raise ValueError("output overlaps a weight, scale or routing tensor")
    if any(w.storage_offset() % 4 or w.data_ptr() % 4 for w in (w1, w2)):
        raise ValueError("weight tensors must be 4-byte aligned")


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
    """UINT4B8 or FP8 E4M3 SwiGLU MoE with A16 activations.

    Weights use [E, N, K / pack] and scales use [E, N, K / group_size].
    group_size is a positive multiple of 128 dividing both projection K sizes.
    workspace, intermediate_cache13 and intermediate_cache2 are compatibility
    arguments. This backend allocates its own temporary buffers.
    """
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
    """Run INT4 MoE. MetaX chooses its layout internally and ignores swap_ab."""
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

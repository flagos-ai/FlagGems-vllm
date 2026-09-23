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


import hashlib
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language import core as tl_core

# ============================================================================
# Format constants and execution parameters
# MODEL1 format constants and measured MUSA execution parameters.
# ============================================================================

# The one shape this implementation has been validated for. No kernel reads
# these: the attention and combine kernels receive the query's actual head
# count as a constexpr. The entry uses them only to reject shapes that were
# never tested, because the SQMMA instruction shape and the warp row
# distribution have only been checked at this head count.
VALIDATED_HEADS = 64
HEAD_DIM = tl.constexpr(512)
NOPE_DIM = tl.constexpr(448)
ROPE_DIM = HEAD_DIM - NOPE_DIM
QUANT_GROUP = tl.constexpr(64)
KV_DATA_BYTES = NOPE_DIM + 2 * ROPE_DIM
KV_SCALE_BYTES = tl.constexpr(8)  # Seven exponent bytes and one padding byte.
KV_TOKEN_BYTES = KV_DATA_BYTES + KV_SCALE_BYTES
FP32_EXPONENT_SHIFT = tl.constexpr(23)
EXPONENT_BITS = tl.constexpr(8)
EXPONENT_MASK = tl.constexpr(255)
TOKEN_TILE = tl.constexpr(64)
PIPELINE_HALVES = tl.constexpr(2)
HALF_DIM = HEAD_DIM // PIPELINE_HALVES
PV_TILE = tl.constexpr(128)
# Packed bytes of one token per half. The first half is FP8 only; the second
# holds the remaining FP8 bytes followed by the BF16 RoPE tail.
FIRST_HALF_DATA_BYTES = tl.constexpr(256)
SECOND_HALF_DATA_BYTES = NOPE_DIM - FIRST_HALF_DATA_BYTES
ROPE_BYTES = 2 * ROPE_DIM
# One uint32 carries four FP8 values, and one scale word carries four exponents.
FP8_PER_WORD = tl.constexpr(4)
EXPONENTS_PER_WORD = tl.constexpr(4)
WORDS_PER_HALF = HALF_DIM // FP8_PER_WORD
# Bytes each lane copies per global-to-shared instruction.
DATA_COPY_BYTES = tl.constexpr(8)
ROPE_COPY_BYTES = tl.constexpr(16)
SCALE_COPY_BYTES = tl.constexpr(4)
METADATA_STRIDE = tl.constexpr(8)  # Five fields in a 32-byte record.
LN_2 = tl.constexpr(math.log(2.0))
# A finite sentinel avoids -inf - -inf on empty tiles.
EMPTY_SCORE = tl.constexpr(-1.0e30)

# Selected by the trace component measurements; unrelated to batch or seed.
COLLECT_WARPS = 8
ATTENTION_WARPS = 8
# Warp roles inside the attention CTA: two consumers and one producer.
CONSUMER_WARPS = tl.constexpr(8)
PRODUCER_WARPS = tl.constexpr(4)
COMBINE_HEADS_PER_CTA = tl.constexpr(8)
# The schedule kernel is one short vector pass; four warps finish it fastest.
SCHEDULE_WARPS = 4


# ============================================================================
# Native vector primitives: FP8 decode, LMA wait
# Native MODEL1 vector primitives linked through Triton's extern_libs API.
# ============================================================================

_IR_model1_vectors = r"""
target triple = "musa"
declare <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8>)
declare <4 x bfloat> @llvm.musa.mul.bhf.bst4.vv(<4 x half>, <4 x float>)
declare void @llvm.musa.lma.wait()
define i32 @model1_lma_complete(i32 %token) alwaysinline nounwind {
  call void @llvm.musa.lma.wait()
  ret i32 %token
}
define i64 @model1_decode_four(i32 %bytes, float %scale) alwaysinline nounwind {
  %v = bitcast i32 %bytes to <4 x i8>
  %h = call <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8> %v)
  %s0 = insertelement <4 x float> poison, float %scale, i32 0
  %s = shufflevector <4 x float> %s0, <4 x float> poison, <4 x i32> zeroinitializer
  %b = call <4 x bfloat> @llvm.musa.mul.bhf.bst4.vv(<4 x half> %h, <4 x float> %s)
  %result = bitcast <4 x bfloat> %b to i64
  ret i64 %result
}
"""


@lru_cache(maxsize=8)
def materialize_library(name, ir, cache_dir):
    digest = hashlib.sha256(ir.encode()).hexdigest()[:16]
    path = Path(cache_dir) / (name + "-" + digest + ".ll")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as tmp:
            tmp.write(ir)
        os.replace(tmp.name, path)
    return str(path)


def extern_libs_model1_vectors():
    # Content-addressed text IR is accepted by link_extern_libs. No native
    # compilation, kernel launch, or import of the baseline is required.
    cache_dir = os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton/cache"))
    path = materialize_library("model1-vectors", _IR_model1_vectors, cache_dir)
    return {"model1_vectors": str(path)}


@tl_core.extern
def decode_four(bytes, scale, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [bytes, scale],
        {(tl.uint32, tl.float32): ("model1_decode_four", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def lma_wait(token, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [token],
        {(tl.int32,): ("model1_lma_complete", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# Native outer-loop hint
# Isolated native request-loop hint, using the supported extern_libs path.
# ============================================================================

_IR_model1_loop_hint = """
declare void @llvm.musa.loop.transparent.outermost()
define i32 @model1_outer_loop(i32 %token) alwaysinline nounwind {
  call void @llvm.musa.loop.transparent.outermost()
  ret i32 %token
}
"""


def extern_libs_model1_loop_hint():
    result = extern_libs_model1_vectors()
    cache_dir = os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton/cache"))
    result["model1_loop_hint"] = materialize_library(
        "model1-loop-hint", _IR_model1_loop_hint, cache_dir
    )
    return result


@tl_core.extern
def outer_loop(token, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [token],
        {(tl.int32,): ("model1_outer_loop", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# Bounded async global-to-shared copy
# MODEL1's native bounded async G2S, with hardware out-of-range zero fill.
# ============================================================================

_IR_model1_robust_copy = """
declare void @llvm.musa.memcpy.g2s.robust.v4(ptr addrspace(3), ptr addrspace(1), i32, <4 x i32>, i32)
declare void @llvm.musa.memcpy.g2s.wait()
define i32 @model1_copy_complete(i32 %token) alwaysinline nounwind {
  call void @llvm.musa.memcpy.g2s.wait()
  ret i32 %token
}
"""
for width in (4, 8, 16):
    _IR_model1_robust_copy += f"""
define i32 @model1_robust_copy_{width}(i32 %dst, i64 %src, i64 %base, i64 %size) alwaysinline nounwind {{
  %d = inttoptr i32 %dst to ptr addrspace(3)
  %s = inttoptr i64 %src to ptr addrspace(1)
  %b = bitcast i64 %base to <2 x i32>
  %z = bitcast i64 %size to <2 x i32>
  %descriptor = shufflevector <2 x i32> %b, <2 x i32> %z, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void @llvm.musa.memcpy.g2s.robust.v4(ptr addrspace(3) %d, ptr addrspace(1) %s,
                                           i32 {width}, <4 x i32> %descriptor, i32 0)
  ret i32 %dst
}}
"""


def extern_libs_model1_robust_copy():
    result = extern_libs_model1_loop_hint()
    cache_dir = os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton/cache"))
    result["model1_robust_copy"] = materialize_library(
        "model1-robust-copy", _IR_model1_robust_copy, cache_dir
    )
    return result


@tl_core.extern
def copy_four(dst, src, base, size, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [dst, src, base, size],
        {
            (tl.uint32, tl.uint64, tl.uint64, tl.uint64): (
                "model1_robust_copy_4",
                tl.int32,
            )
        },
        is_pure=False,
        _semantic=_semantic,
    )


@tl_core.extern
def copy_eight(dst, src, base, size, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [dst, src, base, size],
        {
            (tl.uint32, tl.uint64, tl.uint64, tl.uint64): (
                "model1_robust_copy_8",
                tl.int32,
            )
        },
        is_pure=False,
        _semantic=_semantic,
    )


@tl_core.extern
def copy_sixteen(dst, src, base, size, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [dst, src, base, size],
        {
            (tl.uint32, tl.uint64, tl.uint64, tl.uint64): (
                "model1_robust_copy_16",
                tl.int32,
            )
        },
        is_pure=False,
        _semantic=_semantic,
    )


@tl_core.extern
def copy_wait(token, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [token],
        {(tl.int32,): ("model1_copy_complete", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# Within-thread vector reductions
# Native within-thread reduction helpers for MODEL1.
# ============================================================================

_IR_model1_reduce = """
declare <4 x float> @llvm.musa.max.f.bst4.src02(<4 x float>, <4 x float>)
declare <2 x float> @llvm.musa.max.f.bst2.src02(<2 x float>, <2 x float>)
declare <2 x float> @llvm.musa.add.f.bst2.src01(<2 x float>, <2 x float>)
declare float @llvm.maxnum.f32(float, float)
define float @model1_max8(float %a0, float %a1, float %a2, float %a3,
                          float %a4, float %a5, float %a6, float %a7)
    alwaysinline nounwind readnone {
  %l0 = insertelement <4 x float> poison, float %a0, i32 0
  %l1 = insertelement <4 x float> %l0, float %a1, i32 1
  %l2 = insertelement <4 x float> %l1, float %a2, i32 2
  %l3 = insertelement <4 x float> %l2, float %a3, i32 3
  %r0 = insertelement <4 x float> poison, float %a4, i32 0
  %r1 = insertelement <4 x float> %r0, float %a5, i32 1
  %r2 = insertelement <4 x float> %r1, float %a6, i32 2
  %r3 = insertelement <4 x float> %r2, float %a7, i32 3
  %v = call <4 x float> @llvm.musa.max.f.bst4.src02(<4 x float> %l3, <4 x float> %r3)
  %lo = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hi = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %p = call <2 x float> @llvm.musa.max.f.bst2.src02(<2 x float> %lo, <2 x float> %hi)
  %x = extractelement <2 x float> %p, i32 0
  %y = extractelement <2 x float> %p, i32 1
  %z = call float @llvm.maxnum.f32(float %x, float %y)
  ret float %z
}
define float @model1_sum8(float %a0, float %a1, float %a2, float %a3,
                          float %a4, float %a5, float %a6, float %a7)
    alwaysinline nounwind readnone {
  %l0 = insertelement <4 x float> poison, float %a0, i32 0
  %l1 = insertelement <4 x float> %l0, float %a1, i32 1
  %l2 = insertelement <4 x float> %l1, float %a2, i32 2
  %l3 = insertelement <4 x float> %l2, float %a3, i32 3
  %r0 = insertelement <4 x float> poison, float %a4, i32 0
  %r1 = insertelement <4 x float> %r0, float %a5, i32 1
  %r2 = insertelement <4 x float> %r1, float %a6, i32 2
  %r3 = insertelement <4 x float> %r2, float %a7, i32 3
  %v = fadd <4 x float> %l3, %r3
  %lo = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hi = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %p = call <2 x float> @llvm.musa.add.f.bst2.src01(<2 x float> %lo, <2 x float> %hi)
  %x = extractelement <2 x float> %p, i32 0
  %y = extractelement <2 x float> %p, i32 1
  %z = fadd float %x, %y
  ret float %z
}
"""


def extern_libs_model1_reduce():
    result = extern_libs_model1_robust_copy()
    cache_dir = os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton/cache"))
    result["model1_reduce"] = materialize_library(
        "model1-reduce", _IR_model1_reduce, cache_dir
    )
    return result


@tl_core.extern
def max8(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("model1_max8", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def sum8(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("model1_sum8", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


# ============================================================================
# Two-wide burst exp2
# Two-wide burst exp2 for PH1, linked through Triton's extern_libs API.
# ============================================================================

_IR_model1_exp2 = r"""
target triple = "musa"
declare <2 x float> @llvm.musa.exp2.f.bst2(<2 x float>)
define i64 @model1_exp2_two(float %a, float %b) alwaysinline nounwind readnone {
  %v0 = insertelement <2 x float> poison, float %a, i32 0
  %v1 = insertelement <2 x float> %v0, float %b, i32 1
  %r = call <2 x float> @llvm.musa.exp2.f.bst2(<2 x float> %v1)
  %i = bitcast <2 x float> %r to i64
  ret i64 %i
}
"""


def extern_libs_model1_exp2():
    result = extern_libs_model1_reduce()
    cache_dir = os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton/cache"))
    result["model1_exp2"] = materialize_library(
        "model1-exp2", _IR_model1_exp2, cache_dir
    )
    return result


@tl_core.extern
def exp2_two(a, b, _semantic=None):
    """exp2 of two values in one instruction; the pair returns packed in a uint64."""
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b],
        {(tl.float32, tl.float32): ("model1_exp2_two", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


# ============================================================================
# Kernel 1: effective sparse lengths
# Effective sparse lengths: how far into each index array the work really is.
# ============================================================================


@triton.jit
def collect_effective_lengths(
    indices,
    lengths,
    extra_indices,
    extra_lengths,
    effective_lengths,
    effective_extra_lengths,
    TOPK: tl.constexpr,
    TOKENS: tl.constexpr,
    EXTRA: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    EXTRA_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
    EXTRA_BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    position = tl.arange(0, BLOCK)
    token = tl.load(indices + request * TOPK + position, mask=position < TOPK, other=-1)
    valid = (position < tl.load(lengths + request)) & (token >= 0) & (token < TOKENS)
    tl.store(effective_lengths + request, tl.max(tl.where(valid, position + 1, 0), 0))
    if EXTRA:
        declared = tl.minimum(
            tl.maximum(tl.load(extra_lengths + request), 0), EXTRA_TOPK
        )
        last = tl.load(
            extra_indices + request * EXTRA_TOPK + tl.maximum(declared - 1, 0),
            mask=declared > 0,
            other=-1,
        )
        # A valid final index proves there is no invalid suffix, which is the
        # common case and avoids scanning thousands of entries for it.
        if (declared > 0) & ((last < 0) | (last >= EXTRA_TOKENS)):
            extra_position = tl.arange(0, EXTRA_BLOCK)
            extra_token = tl.load(
                extra_indices + request * EXTRA_TOPK + extra_position,
                mask=extra_position < EXTRA_TOPK,
                other=-1,
            )
            extra_valid = (
                (extra_position < declared)
                & (extra_token >= 0)
                & (extra_token < EXTRA_TOKENS)
            )
            tl.store(
                effective_extra_lengths + request,
                tl.max(tl.where(extra_valid, extra_position + 1, 0), 0),
            )
        else:
            tl.store(effective_extra_lengths + request, declared)


def prepare_effective_lengths(
    q,
    cache,
    *,
    indices,
    topk_length,
    effective_lengths,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    extra_topk_length=None,
    effective_extra_lengths=None,
):
    batch = q.shape[0]
    extra = extra_k_cache is not None
    prepare_effective_lengths.last_compiled = collect_effective_lengths[(batch,)](
        indices,
        topk_length,
        extra_indices_in_kvcache if extra else indices,
        extra_topk_length if extra else topk_length,
        effective_lengths,
        effective_extra_lengths if extra else effective_lengths,
        indices.shape[-1],
        cache.shape[0] * cache.shape[1],
        extra,
        extra_indices_in_kvcache.shape[-1] if extra else 0,
        extra_k_cache.shape[0] * extra_k_cache.shape[1] if extra else 0,
        triton.next_power_of_2(indices.shape[-1]),
        triton.next_power_of_2(extra_indices_in_kvcache.shape[-1]) if extra else 1,
        num_warps=COLLECT_WARPS,
    )


# ============================================================================
# Kernel 2: balanced tile intervals
# Build balanced tile intervals with one CTA per attention partition.
# ============================================================================


@triton.jit
def make_schedule(
    lengths,
    extra_lengths,
    metadata,
    split_prefix,
    BATCH: tl.constexpr,
    PARTS: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    part = tl.program_id(0)
    request = tl.arange(0, BLOCK)
    length = tl.load(lengths + request, request < BATCH, other=0)
    tiles = (tl.maximum(length, 1) + TOKEN_TILE - 1) // TOKEN_TILE
    if HAS_EXTRA:
        extra_length = tl.load(extra_lengths + request, request < BATCH, other=0)
        tiles += (tl.maximum(extra_length, 0) + TOKEN_TILE - 1) // TOKEN_TILE
    tiles = tl.where(request < BATCH, tiles, 0)
    end = tl.cumsum(tiles, 0)
    start = end - tiles
    total = tl.sum(tiles, 0)
    payload = (total + PARTS - 1) // PARTS
    begin = part * payload
    limit = tl.minimum(begin + payload, total)
    first = tl.sum(((request < BATCH) & (end <= begin)).to(tl.int32), 0)
    last = tl.sum(((request < BATCH) & (end < limit)).to(tl.int32), 0)
    first_start = tl.sum(tl.where(request < first, tiles, 0), 0)
    last_start = tl.sum(tl.where(request < last, tiles, 0), 0)
    tl.store(metadata + part * METADATA_STRIDE, first)
    tl.store(metadata + part * METADATA_STRIDE + 1, begin - first_start)
    tl.store(metadata + part * METADATA_STRIDE + 2, last)
    tl.store(metadata + part * METADATA_STRIDE + 3, limit - last_start)
    tl.store(metadata + part * METADATA_STRIDE + 4, part - first_start // payload)
    if part == 0:
        splits = (end - 1) // payload - start // payload + 1
        splits = tl.where(request < BATCH, splits, 0)
        tl.store(split_prefix, 0)
        tl.store(split_prefix + request + 1, tl.cumsum(splits, 0), request < BATCH)


# ============================================================================
# Kernel 4: split combine
# Combine split requests, eight heads per CTA as in the MODEL1 baseline.
# ============================================================================

_LAYOUT = tl.constexpr(tle.gpu.BlockEncoding([1, 4], [1, 32], [8, 1], [1, 0]))
_HEADS = tl.constexpr(tle.gpu.SlicedEncoding(1, _LAYOUT.value))
_DIMS = tl.constexpr(tle.gpu.SlicedEncoding(0, _LAYOUT.value))


@triton.jit
def combine_splits(
    partial,
    partial_lse,
    prefix,
    sink,
    out,
    lse,
    HAS_SINK: tl.constexpr,
    HEADS: tl.constexpr,
):
    request = tl.program_id(0)
    row = HEADS * HEAD_DIM
    first = tl.load(prefix + request)
    end = tl.load(prefix + request + 1)
    if end - first > 1:
        h = tl.program_id(1) * COMBINE_HEADS_PER_CTA + tle.gpu.set_layout(
            tl.arange(0, COMBINE_HEADS_PER_CTA), _HEADS
        )
        d = tle.gpu.set_layout(tl.arange(0, HEAD_DIM), _DIMS)
        maximum = tl.full((COMBINE_HEADS_PER_CTA,), float("-inf"), tl.float32)
        denominator = tl.zeros((COMBINE_HEADS_PER_CTA,), tl.float32)
        result = tl.zeros((COMBINE_HEADS_PER_CTA, HEAD_DIM), tl.float32)
        for slot in range(first, end):
            value = tl.load(partial_lse + slot * HEADS + h)
            new_maximum = tl.maximum(maximum, value)
            safe_maximum = tl.where(new_maximum == float("-inf"), 0.0, new_maximum)
            alpha = tl.exp2(maximum - safe_maximum)
            weight = tl.exp2(value - safe_maximum)
            result = (
                result * alpha[:, None]
                + tl.load(partial + slot * row + h[:, None] * HEAD_DIM + d[None, :])
                * weight[:, None]
            )
            denominator = denominator * alpha + weight
            maximum = new_maximum
        valid = denominator > 0
        natural_lse = tl.where(
            valid, (tl.log2(denominator) + maximum) * LN_2, float("inf")
        )
        factor = tl.where(valid, 1.0 / denominator, 0.0)
        if HAS_SINK:
            sink_value = tl.load(sink + h)
            factor *= tl.where(
                sink_value == float("inf"),
                0.0,
                1.0 / (1.0 + tl.exp(sink_value - natural_lse)),
            )
        tl.store(
            out + request * row + h[:, None] * HEAD_DIM + d[None, :],
            result * factor[:, None],
        )
        tl.store(lse + request * HEADS + h, natural_lse)


# ============================================================================
# Kernel 3: warp-specialised attention
# MODEL1 8/8/4-warp pipeline: producer gathers KV, two consumers split the head dim.
# ============================================================================

# Elements of the shared K buffer that one lane's copy or load covers.
DATA_COPY_ELEMENTS = DATA_COPY_BYTES // 2
# Lanes per row in the packed read: the first half fills a whole staging row.
PACKED_LANES = FIRST_HALF_DATA_BYTES // DATA_COPY_BYTES
# Packed words that still hold FP8 data in the second half; the rest are RoPE.
SECOND_HALF_WORDS = SECOND_HALF_DATA_BYTES // FP8_PER_WORD
# Where the BF16 RoPE tail starts inside a second-half K row, in elements.
ROPE_ELEMENT_BASE = HALF_DIM - ROPE_BYTES // 2
# Lanes the producer uses for the wider RoPE copy.
ROPE_LANES = ROPE_BYTES // ROPE_COPY_BYTES
# Barriers both consumers arrive at.
BOTH_CONSUMER_WARPS = CONSUMER_WARPS * 2


_C2 = tl.constexpr(tle.gpu.BlockEncoding([1, 8], [4, 8], [8, 1], [1, 0]))
_CQ2 = tl.constexpr(tle.gpu.BlockEncoding([1, 1], [4, 8], [8, 1], [1, 0]))
_CV4 = tl.constexpr(tle.gpu.BlockEncoding([1, 2], [4, 8], [8, 1], [1, 0]))
_PC = tl.constexpr(tle.gpu.BlockEncoding([1, 1], [4, 8], [4, 1], [1, 0]))


@tl_core.builtin
def _inline(fn, args, _semantic=None, _generator=None):
    """Use the same frontend inlining as WS for nested, role-local helpers.

    Normal nested JIT calls survive the stock inliner in isolated WS regions
    and lose the caller's explicit layout. This stays within the operator.
    """
    fn = tl_core._unwrap_if_constexpr(fn)
    args = tl_core._unwrap_if_constexpr(args)
    if isinstance(args, tl_core.tuple):
        args = tuple(args.values)
    return _generator.inline_JitFunction(fn, args, kwargs={})


@triton.jit
def _decode(k, scales, HALF: tl.constexpr):
    # Match MATE's 32 rows x 64 columns per CTA distribution: each lane owns
    # two rows and four groups of eight columns. PV extracts stay in registers.
    # Every index vector below is carved from its own arange window: layout
    # propagation merges identical arange domains, which would tie these to
    # softmax's own arange over the heads. Only the distinctness matters.
    # Each lane fetches eight packed FP8 bytes in one shared-memory word.
    # A volatile integer carrier preserves this access through LLVM's byte
    # extraction optimizations and observes each producer publication.
    pr = tle.gpu.set_layout(
        tl.broadcast_to(
            (tl.arange(256, 320) - 256)[:, None], (TOKEN_TILE, PACKED_LANES)
        ),
        _CQ2,
    )
    pc = tle.gpu.set_layout(
        tl.broadcast_to(
            (tl.arange(HEAD_DIM, 544) - HEAD_DIM)[None, :], (TOKEN_TILE, PACKED_LANES)
        ),
        _CQ2,
    )
    raw_ptr = tle.gpu.local_ptr(k, (pr, pc * DATA_COPY_ELEMENTS)).to(
        tl.pointer_type(tl.uint64, 3)
    )
    raw = tl.load(raw_ptr, volatile=True)
    words = tl.reshape(
        tl.join(raw.to(tl.uint32), (raw >> 32).to(tl.uint32)),
        (TOKEN_TILE, WORDS_PER_HALF),
    )
    vr = tle.gpu.set_layout(
        tl.broadcast_to(
            (tl.arange(320, 384) - 320)[:, None], (TOKEN_TILE, WORDS_PER_HALF)
        ),
        _CV4,
    )
    vc = tle.gpu.set_layout(
        tl.broadcast_to(
            (tl.arange(544, 608) - 544)[None, :], (TOKEN_TILE, WORDS_PER_HALF)
        ),
        _CV4,
    )
    words = tle.gpu.set_layout(words, _CV4)
    scale_word = tl.load(tle.gpu.local_ptr(scales, (vr * PIPELINE_HALVES + HALF,)))
    exponent = (
        scale_word >> (vc // (WORDS_PER_HALF // EXPONENTS_PER_WORD) * EXPONENT_BITS)
    ) & EXPONENT_MASK
    if HALF == 1:
        exponent = tl.where(vc < SECOND_HALF_WORDS, exponent, 0)
    scale = (exponent << FP32_EXPONENT_SHIFT).to(tl.float32, bitcast=True)
    packed = decode_four(words, scale)
    if HALF == 1:
        rope_ptr = tle.gpu.local_ptr(k, (vr, vc * DATA_COPY_ELEMENTS)).to(
            tl.pointer_type(tl.uint64, 3)
        )
        rope = tl.load(rope_ptr, mask=vc >= SECOND_HALF_WORDS, other=0, volatile=True)
        packed = tl.where(vc < SECOND_HALF_WORDS, packed, rope)
    pairs = tl.reshape(
        tl.join(packed.to(tl.uint32), (packed >> 32).to(tl.uint32)),
        (TOKEN_TILE, HALF_DIM // 2),
    )
    values = tl.reshape(
        tl.join(pairs.to(tl.uint16), (pairs >> 16).to(tl.uint16)),
        (TOKEN_TILE, HALF_DIM),
    )
    return tle.gpu.set_layout(values.to(tl.bfloat16, bitcast=True), _C2)


@triton.jit
def _exp2_pairs(x):
    """exp2 over the score tile, two elements per hardware instruction.

    Pairs must be thread-local. SQMMA puts the low three token bits across
    lanes, so neighbours along the last axis live in different threads; the
    same permute the row reduction uses brings the register-resident bits to
    the end, and the pairs are taken there.
    """
    x = tl.permute(tl.reshape(x, (64, 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    x0, x1 = tl.split(x)
    x00, x01 = tl.split(x0)
    x10, x11 = tl.split(x1)
    a0, a4 = tl.split(x00)
    a2, a6 = tl.split(x01)
    a1, a5 = tl.split(x10)
    a3, a7 = tl.split(x11)

    p04 = exp2_two(a0, a4)
    p26 = exp2_two(a2, a6)
    p15 = exp2_two(a1, a5)
    p37 = exp2_two(a3, a7)
    b0 = (p04 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b4 = (p04 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b2 = (p26 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b6 = (p26 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b1 = (p15 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b5 = (p15 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b3 = (p37 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b7 = (p37 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    y = tl.join(
        tl.join(tl.join(b0, b4), tl.join(b2, b6)),
        tl.join(tl.join(b1, b5), tl.join(b3, b7)),
    )
    return tl.reshape(tl.permute(y, (0, 2, 3, 4, 1)), (64, 64))


@triton.jit
def _pack_kv(value):
    # Preserve two BF16 values per uint32 until PV staging. The tied empty asm
    # keeps LLVM from propagating unpacked scalar lifetimes through softmax;
    # it preserves every bit and does not provide any memory synchronization.
    lo, hi = tl.split(
        tl.reshape(value.to(tl.uint16, bitcast=True), (TOKEN_TILE, HALF_DIM // 2, 2))
    )
    packed = lo.to(tl.uint32) | (hi.to(tl.uint32) << 16)
    return tl.inline_asm_elementwise(
        "", constraints="=R,0", args=[packed], dtype=tl.uint32, is_pure=False, pack=1
    )


@triton.jit
def _unpack_piece(packed, PIECE: tl.constexpr):
    # Unpack only the 128-column V tile whose shared buffer is now available.
    part = tle.extract_tile(packed, index=PIECE, tile_shape=(TOKEN_TILE, PV_TILE // 2))
    part = tl.inline_asm_elementwise(
        "", constraints="=R,0", args=[part], dtype=tl.uint32, is_pure=False, pack=1
    )
    values = tl.reshape(
        tl.join(part.to(tl.uint16), (part >> 16).to(tl.uint16)), (TOKEN_TILE, PV_TILE)
    )
    return tle.gpu.set_layout(values.to(tl.bfloat16, bitcast=True), _C2)


@triton.jit
def _produce(
    cache,
    indices,
    lengths,
    extra_cache,
    extra_indices,
    extra_lengths,
    metadata,
    kstage,
    kb,
    scales,
    mask_shared,
    full0,
    full1,
    empty0,
    empty1,
    sync,
    BATCH: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    PAGES: tl.constexpr,
    EXTRA: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    EXTRA_PAGE: tl.constexpr,
    EXTRA_STRIDE: tl.constexpr,
    EXTRA_PAGES: tl.constexpr,
    PAGE_LOG2: tl.constexpr,
    EXTRA_PAGE_LOG2: tl.constexpr,
):
    part = tl.program_id(0)
    first = tl.load(metadata + part * METADATA_STRIDE)
    first_block = tl.load(metadata + part * METADATA_STRIDE + 1)
    last = tl.load(metadata + part * METADATA_STRIDE + 2)
    last_block = tl.load(metadata + part * METADATA_STRIDE + 3)
    row = tle.gpu.set_layout(tl.arange(0, TOKEN_TILE), tle.gpu.SlicedEncoding(1, _PC))
    # Transpose the row order within each group of eight so the eight lanes of
    # one copy instruction address eight consecutive tokens.
    source_row = row % 8 * 8 + row // 8
    rr = tle.gpu.set_layout(
        tl.broadcast_to((tl.arange(64, 128) - 64)[:, None], (TOKEN_TILE, PACKED_LANES)),
        _PC,
    )
    cc = tle.gpu.set_layout(
        tl.broadcast_to(
            (tl.arange(256, 288) - 256)[None, :], (TOKEN_TILE, PACKED_LANES)
        ),
        _PC,
    )
    rrope = tle.gpu.set_layout(
        tl.broadcast_to((tl.arange(128, 192) - 128)[:, None], (TOKEN_TILE, ROPE_LANES)),
        _PC,
    )
    crope = tle.gpu.set_layout(
        tl.broadcast_to((tl.arange(384, 392) - 384)[None, :], (TOKEN_TILE, ROPE_LANES)),
        _PC,
    )
    step = 0
    for request in tl.range(first, tl.minimum(last + 1, BATCH), num_stages=1):
        outer_loop(tl.full((), 0, tl.int32))
        main_length = tl.load(lengths + request)
        main_blocks = (tl.maximum(main_length, 1) + TOKEN_TILE - 1) // TOKEN_TILE
        total_blocks = main_blocks
        if EXTRA:
            extra_length = tl.load(extra_lengths + request)
            total_blocks += (tl.maximum(extra_length, 0) + TOKEN_TILE - 1) // TOKEN_TILE
        begin = tl.where(request == first, first_block, 0)
        end = tl.where(request == last, last_block, total_blocks)
        for block in tl.range(begin, end, num_stages=1):
            if EXTRA:
                extra = block >= main_blocks
                block_offset = tl.where(extra, block - main_blocks, block) * TOKEN_TILE
                ids = tl.where(
                    extra,
                    extra_indices + request * EXTRA_TOPK,
                    indices + request * TOPK,
                )
                valid_length = tl.where(extra, extra_length, main_length)
                topk = tl.where(extra, EXTRA_TOPK, TOPK)
                selected_cache = tl.where(extra, extra_cache, cache)
                page = tl.where(extra, EXTRA_PAGE, PAGE)
                shift = tl.where(extra, EXTRA_PAGE_LOG2, PAGE_LOG2)
                stride = tl.where(extra, EXTRA_STRIDE, STRIDE)
                pages = tl.where(extra, EXTRA_PAGES, PAGES)
            else:
                block_offset = block * TOKEN_TILE
                ids = indices + request * TOPK
                valid_length = main_length
                topk = TOPK
                selected_cache = cache
                page = PAGE
                shift = PAGE_LOG2
                stride = STRIDE
                pages = PAGES
            token = tl.load(
                ids + block_offset + source_row,
                mask=block_offset + source_row < topk,
                other=-1,
            )
            good = (
                (block_offset + source_row < valid_length)
                & (token >= 0)
                & (token < pages * page)
            )
            token = tl.where(good, token, pages * page)
            # A token index fits int32 while its byte offset can exceed 2 GiB.
            # Widen before multiplying, including the out-of-range sentinel.
            page_base = selected_cache + (token >> shift).to(tl.int64) * stride
            base = page_base + (token & (page - 1)) * KV_DATA_BYTES
            scale_base = (
                page_base + page * KV_DATA_BYTES + (token & (page - 1)) * KV_SCALE_BYTES
            )
            bound = tl.full((), 1, tl.uint64) * pages * stride
            for half in tl.static_range(2):
                if half == 0:
                    tle.gpu.barrier_wait(empty0, phaseIdx=step & 1)
                    # The first half lands in staging, never in ka. While the
                    # producer wrote ka directly, both forms of the intermittent
                    # wrong result appeared; separating the buffers removed them
                    # (0 in 100000 per-call checks on each extra family).
                    k = kstage
                    tl.store(tle.gpu.local_ptr(mask_shared, (row,)), good.to(tl.int32))
                else:
                    tle.gpu.barrier_wait(empty1, phaseIdx=step & 1)
                    k = kb
                src = (
                    base[:, None] + half * FIRST_HALF_DATA_BYTES + cc * DATA_COPY_BYTES
                )
                if half == 1:
                    # The second half holds fewer FP8 bytes; point the surplus
                    # lanes past the cache so the bounded copy zero-fills them.
                    src = tl.where(
                        cc < SECOND_HALF_DATA_BYTES // DATA_COPY_BYTES,
                        src,
                        selected_cache + bound,
                    )
                copy_eight(
                    tle.gpu.local_ptr(k, (rr, cc * DATA_COPY_ELEMENTS))
                    .to(tl.uint64)
                    .to(tl.uint32),
                    src.to(tl.uint64),
                    selected_cache.to(tl.uint64),
                    bound,
                )
                if half == 1:
                    copy_sixteen(
                        tle.gpu.local_ptr(
                            k,
                            (rrope, ROPE_ELEMENT_BASE + crope * (ROPE_COPY_BYTES // 2)),
                        )
                        .to(tl.uint64)
                        .to(tl.uint32),
                        (base[:, None] + NOPE_DIM + crope * ROPE_COPY_BYTES).to(
                            tl.uint64
                        ),
                        selected_cache.to(tl.uint64),
                        bound,
                    )
                copy_four(
                    tle.gpu.local_ptr(scales, (row * PIPELINE_HALVES + half,))
                    .to(tl.uint64)
                    .to(tl.uint32),
                    (scale_base + half * SCALE_COPY_BYTES).to(tl.uint64),
                    selected_cache.to(tl.uint64),
                    bound,
                )
                copy_wait(tl.full((), 0, tl.int32))
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(sync, phaseIdx=half)
                tle.gpu.barrier_wait(sync, phaseIdx=half)
                lma_wait(tl.full((), 0, tl.int32))
                if half == 0:
                    tle.gpu.barrier_arrive(full0, phaseIdx=step & 1)
                else:
                    tle.gpu.barrier_arrive(full1, phaseIdx=step & 1)
            step += 1


@triton.jit
def _row_reduce(x, MAXIMUM: tl.constexpr):
    # SQMMA distributes low three K bits across lanes; high bits are registers.
    x = tl.permute(tl.reshape(x, (64, 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    x0, x1 = tl.split(x)
    x00, x01 = tl.split(x0)
    x10, x11 = tl.split(x1)
    a0, a4 = tl.split(x00)
    a2, a6 = tl.split(x01)
    a1, a5 = tl.split(x10)
    a3, a7 = tl.split(x11)
    if MAXIMUM:
        partial = max8(a0, a1, a2, a3, a4, a5, a6, a7)
        result = tl.max(partial, 1)
    else:
        partial = sum8(a0, a1, a2, a3, a4, a5, a6, a7)
        result = tl.sum(partial, 1)
    return result


@triton.jit
def _consume(
    q,
    lengths,
    extra_lengths,
    metadata,
    prefix,
    partial,
    partial_lse,
    qa,
    qb,
    ka,
    kb,
    kstage,
    scales,
    mask_shared,
    ps,
    v0,
    v1,
    vl0_free,
    vl1_free,
    alpha_shared,
    inverse_shared,
    qready,
    full0,
    full1,
    empty0,
    empty1,
    quant_ready,
    pfull,
    pempty,
    final,
    done,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    SCORE_TO_LOG2: tl.constexpr,
    EXTRA: tl.constexpr,
    HALF: tl.constexpr,
    out,
    lse,
    sink,
    HAS_SINK: tl.constexpr,
):
    part = tl.program_id(0)
    first = tl.load(metadata + part * METADATA_STRIDE)
    first_block = tl.load(metadata + part * METADATA_STRIDE + 1)
    last = tl.load(metadata + part * METADATA_STRIDE + 2)
    last_block = tl.load(metadata + part * METADATA_STRIDE + 3)
    first_split = tl.load(metadata + part * METADATA_STRIDE + 4)
    h = tl.arange(0, HEADS)
    row = HEADS * HEAD_DIM
    vcol = tl.arange(0, PV_TILE)
    step = 0
    for request in tl.range(first, tl.minimum(last + 1, BATCH), num_stages=1):
        outer_loop(tl.full((), 0, tl.int32))
        main_blocks = (
            tl.maximum(tl.load(lengths + request), 1) + TOKEN_TILE - 1
        ) // TOKEN_TILE
        total_blocks = main_blocks
        if EXTRA:
            total_blocks += (
                tl.maximum(tl.load(extra_lengths + request), 0) + TOKEN_TILE - 1
            ) // TOKEN_TILE
        begin = tl.where(request == first, first_block, 0)
        end = tl.where(request == last, last_block, total_blocks)
        slot = tl.load(prefix + request) + tl.where(request == first, first_split, 0)
        unsplit = tl.load(prefix + request + 1) - tl.load(prefix + request) == 1
        qt = tle.load(
            q
            + request * row
            + h[:, None] * HEAD_DIM
            + HALF * HALF_DIM
            + tl.arange(0, HALF_DIM)[None, :],
            is_async=True,
        )
        if HALF == 0:
            tl.store(tle.gpu.local_ptr(qa), qt)
        else:
            tl.store(tle.gpu.local_ptr(qb), qt)
        lma_wait(tl.full((), 0, tl.int32))
        tle.gpu.barrier_arrive(qready, phaseIdx=(request - first) & 1)
        tle.gpu.barrier_wait(qready, phaseIdx=(request - first) & 1)
        maximum = tl.full((HEADS,), EMPTY_SCORE, tl.float32)
        denominator = tl.zeros((HEADS,), tl.float32)
        o0 = tl.zeros((HEADS, PV_TILE), tl.float32)
        o1 = tl.zeros((HEADS, PV_TILE), tl.float32)
        # The consumer only needs the tile count; the producer owns the
        # block index and turns it into cache addresses.
        for _block in tl.range(begin, end, num_stages=1):
            if HALF == 0:
                tle.gpu.barrier_wait(full0, phaseIdx=step & 1)
            else:
                tle.gpu.barrier_wait(full1, phaseIdx=step & 1)
            if HALF == 0:
                kv = _inline(_decode, (kstage, scales, 0))
                valid = tl.load(tle.gpu.local_ptr(mask_shared, (h,))) != 0
            else:
                kv = _inline(_decode, (kb, scales, 1))
            # For the second half the raw read and the expanded store alias the
            # same buffer; the compiler inserts a partition LMA barrier for that
            # dependency.
            if HALF == 0:
                tl.store(tle.gpu.set_layout(tle.gpu.local_ptr(ka), _C2), kv)
            else:
                tl.store(tle.gpu.set_layout(tle.gpu.local_ptr(kb), _C2), kv)
            packed_kv = _inline(_pack_kv, (kv,))
            if HALF == 0:
                # Mask the initial accumulator as in the native pipeline. A finite
                # maximum sentinel makes an empty tile yield exp2(-inf) == 0.
                initial_score = tl.broadcast_to(
                    tl.where(valid[None, :], 0.0, float("-inf")), (HEADS, TOKEN_TILE)
                )
                score = tle.gpu.wgmma(qa, ka, initial_score, trans_b=True)
                score = tle.gpu.wgmma_wait(0, score)
                tle.gpu.barrier_wait(quant_ready, phaseIdx=step & 1)
                score = tle.gpu.wgmma(qb, kb, score, trans_b=True)
                score = tle.gpu.wgmma_wait(0, score)
                # Release the first K half only after the second product.
                # Signalling right after its own wait left the severe form in
                # place; this position removed it. The mechanism behind the
                # window is not established, so do not move this earlier
                # without re-running the per-call checks.
                tle.gpu.barrier_arrive(empty0, phaseIdx=step & 1)
                tle.gpu.barrier_arrive(empty1, phaseIdx=step & 1)
                new_maximum = tl.maximum(maximum, _inline(_row_reduce, (score, True)))
                alpha = tl.exp2((maximum - new_maximum) * SCORE_TO_LOG2)
                p = _inline(
                    _exp2_pairs, ((score - new_maximum[:, None]) * SCORE_TO_LOG2,)
                )
                maximum = new_maximum
                denominator = denominator * alpha + _inline(_row_reduce, (p, False))
                tle.gpu.barrier_wait(pempty, phaseIdx=step & 1)
                tl.store(tle.gpu.local_ptr(ps), p.to(tl.bfloat16))
                tl.store(tle.gpu.local_ptr(alpha_shared, (h,)), alpha)
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(pfull, phaseIdx=step & 1)
            else:
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(quant_ready, phaseIdx=step & 1)
                tle.gpu.barrier_wait(pfull, phaseIdx=step & 1)
                alpha = tl.load(tle.gpu.local_ptr(alpha_shared, (h,)))
            if HALF == 1:
                tle.gpu.barrier_wait(vl0_free, phaseIdx=step & 1)
            value0 = _inline(_unpack_piece, (packed_kv, 0))
            tl.store(tle.gpu.set_layout(tle.gpu.local_ptr(v0), _C2), value0)
            acc0 = tle.gpu.wgmma(ps, v0, o0 * alpha[:, None])
            if HALF == 1:
                tle.gpu.barrier_wait(vl1_free, phaseIdx=step & 1)
            value1 = _inline(_unpack_piece, (packed_kv, 1))
            tl.store(tle.gpu.set_layout(tle.gpu.local_ptr(v1), _C2), value1)
            o0 = tle.gpu.wgmma_wait(0, acc0)
            if HALF == 0:
                tle.gpu.barrier_arrive(vl0_free, phaseIdx=step & 1)
            acc1 = tle.gpu.wgmma(ps, v1, o1 * alpha[:, None])
            o1 = tle.gpu.wgmma_wait(0, acc1)
            if HALF == 0:
                tle.gpu.barrier_arrive(vl1_free, phaseIdx=step & 1)
            if HALF == 1:
                tle.gpu.barrier_arrive(pempty, phaseIdx=step & 1)
            step += 1
        if HALF == 0:
            inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
            lse2 = tl.where(
                denominator > 0,
                tl.log2(denominator) + maximum * SCORE_TO_LOG2,
                float("-inf"),
            )
            if unsplit:
                natural_lse = tl.where(denominator > 0, lse2 * LN_2, float("inf"))
                tl.store(lse + request * HEADS + h, natural_lse)
                if HAS_SINK:
                    sink_value = tl.load(sink + h)
                    inverse *= tl.where(
                        sink_value == float("inf"),
                        0.0,
                        1.0 / (1.0 + tl.exp(sink_value - natural_lse)),
                    )
            else:
                tl.store(partial_lse + slot * HEADS + h, lse2)
            tl.store(tle.gpu.local_ptr(inverse_shared, (h,)), inverse)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(final, phaseIdx=(request - first) & 1)
        else:
            tle.gpu.barrier_wait(final, phaseIdx=(request - first) & 1)
            inverse = tl.load(tle.gpu.local_ptr(inverse_shared, (h,)))
        offset = h[:, None] * HEAD_DIM + HALF * HALF_DIM + vcol[None, :]
        if unsplit:
            out_addr = out + request * row + offset
            tl.store(out_addr, o0 * inverse[:, None])
            tl.store(out_addr + PV_TILE, o1 * inverse[:, None])
        else:
            partial_addr = partial + slot * row + offset
            tl.store(partial_addr, o0 * inverse[:, None])
            tl.store(partial_addr + PV_TILE, o1 * inverse[:, None])
        tle.gpu.barrier_arrive(done, phaseIdx=(request - first) & 1)
        tle.gpu.barrier_wait(done, phaseIdx=(request - first) & 1)


@triton.jit
def attention_scheduled_ws(
    q,
    cache,
    indices,
    lengths,
    extra_cache,
    extra_indices,
    extra_lengths,
    metadata,
    prefix,
    partial,
    partial_lse,
    BATCH: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    PAGES: tl.constexpr,
    EXTRA: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    EXTRA_PAGE: tl.constexpr,
    EXTRA_STRIDE: tl.constexpr,
    EXTRA_PAGES: tl.constexpr,
    out,
    lse,
    sink,
    HAS_SINK: tl.constexpr,
    PAGE_LOG2: tl.constexpr,
    EXTRA_PAGE_LOG2: tl.constexpr,
    HEADS: tl.constexpr,
    SCORE_TO_LOG2: tl.constexpr,
):
    # Each consumer owns one half of the head dimension; Q and K are staged
    # per half, V is staged in PV_TILE columns shared by both consumers.
    qa = tle.gpu.alloc((HEADS, HALF_DIM), tl.bfloat16)
    qb = tle.gpu.alloc((HEADS, HALF_DIM), tl.bfloat16)
    ka = tle.gpu.alloc((TOKEN_TILE, HALF_DIM), tl.bfloat16)
    kb = tle.gpu.alloc((TOKEN_TILE, HALF_DIM), tl.bfloat16)
    # Landing buffer for the first half's packed bytes: the producer must never
    # write the buffer the matrix unit reads as an operand.
    kstage = tle.gpu.alloc((TOKEN_TILE, FIRST_HALF_DATA_BYTES // 2), tl.bfloat16)
    ps = tle.gpu.alloc((HEADS, TOKEN_TILE), tl.bfloat16)
    v0 = tle.gpu.alloc((TOKEN_TILE, PV_TILE), tl.bfloat16)
    v1 = tle.gpu.alloc((TOKEN_TILE, PV_TILE), tl.bfloat16)
    vl0_free = tle.gpu.alloc_barrier(arrive_count=CONSUMER_WARPS)
    vl1_free = tle.gpu.alloc_barrier(arrive_count=CONSUMER_WARPS)
    scales = tle.gpu.alloc(
        (TOKEN_TILE * PIPELINE_HALVES,), tl.uint32, nv_mma_shared_layout=False
    )
    mask_shared = tle.gpu.alloc((TOKEN_TILE,), tl.int32, nv_mma_shared_layout=False)
    alpha = tle.gpu.alloc((HEADS,), tl.float32, nv_mma_shared_layout=False)
    inverse = tle.gpu.alloc((HEADS,), tl.float32, nv_mma_shared_layout=False)
    # Barriers name who arrives: one consumer, both consumers, or the producer.
    qready = tle.gpu.alloc_barrier(arrive_count=BOTH_CONSUMER_WARPS)
    full0 = tle.gpu.alloc_barrier(arrive_count=PRODUCER_WARPS)
    full1 = tle.gpu.alloc_barrier(arrive_count=PRODUCER_WARPS)
    empty0 = tle.gpu.alloc_barrier(arrive_count=CONSUMER_WARPS, init=tle.gpu.READY)
    empty1 = tle.gpu.alloc_barrier(arrive_count=CONSUMER_WARPS, init=tle.gpu.READY)
    quant_ready = tle.gpu.alloc_barrier(arrive_count=CONSUMER_WARPS)
    pfull = tle.gpu.alloc_barrier(arrive_count=CONSUMER_WARPS)
    pempty = tle.gpu.alloc_barrier(arrive_count=CONSUMER_WARPS, init=tle.gpu.READY)
    producer_sync = tle.gpu.alloc_barrier(arrive_count=PRODUCER_WARPS)
    final = tle.gpu.alloc_barrier(arrive_count=CONSUMER_WARPS)
    done = tle.gpu.alloc_barrier(arrive_count=BOTH_CONSUMER_WARPS)
    tle.gpu.warp_specialize(
        [
            (
                _consume,
                (
                    q,
                    lengths,
                    extra_lengths,
                    metadata,
                    prefix,
                    partial,
                    partial_lse,
                    qa,
                    qb,
                    ka,
                    kb,
                    kstage,
                    scales,
                    mask_shared,
                    ps,
                    v0,
                    v1,
                    vl0_free,
                    vl1_free,
                    alpha,
                    inverse,
                    qready,
                    full0,
                    full1,
                    empty0,
                    empty1,
                    quant_ready,
                    pfull,
                    pempty,
                    final,
                    done,
                    BATCH,
                    HEADS,
                    SCORE_TO_LOG2,
                    EXTRA,
                    0,
                    out,
                    lse,
                    sink,
                    HAS_SINK,
                ),
            ),
            (
                _consume,
                (
                    q,
                    lengths,
                    extra_lengths,
                    metadata,
                    prefix,
                    partial,
                    partial_lse,
                    qa,
                    qb,
                    ka,
                    kb,
                    kstage,
                    scales,
                    mask_shared,
                    ps,
                    v0,
                    v1,
                    vl0_free,
                    vl1_free,
                    alpha,
                    inverse,
                    qready,
                    full0,
                    full1,
                    empty0,
                    empty1,
                    quant_ready,
                    pfull,
                    pempty,
                    final,
                    done,
                    BATCH,
                    HEADS,
                    SCORE_TO_LOG2,
                    EXTRA,
                    1,
                    out,
                    lse,
                    sink,
                    HAS_SINK,
                ),
            ),
            (
                _produce,
                (
                    cache,
                    indices,
                    lengths,
                    extra_cache,
                    extra_indices,
                    extra_lengths,
                    metadata,
                    kstage,
                    kb,
                    scales,
                    mask_shared,
                    full0,
                    full1,
                    empty0,
                    empty1,
                    producer_sync,
                    BATCH,
                    TOPK,
                    PAGE,
                    STRIDE,
                    PAGES,
                    EXTRA,
                    EXTRA_TOPK,
                    EXTRA_PAGE,
                    EXTRA_STRIDE,
                    EXTRA_PAGES,
                    PAGE_LOG2,
                    EXTRA_PAGE_LOG2,
                ),
            ),
        ],
        worker_num_warps=[8, 4],
        worker_num_regs=[224, 64],
    )


# ============================================================================
# Scratch allocation and kernel sequence
# Allocate scratch and compose the public MODEL1 GPU pipeline.
# ============================================================================


def _page_log2(cache):
    page = cache.shape[1]
    if cache.dtype != torch.uint8 or cache.shape[2:] != (1, KV_TOKEN_BYTES.value):
        raise ValueError(
            "MODEL1 requires packed uint8 KV pages with 584 bytes per token"
        )
    if page <= 0 or page & (page - 1):
        raise ValueError("KV page size must be a positive power of two")
    return page.bit_length() - 1


def flash_mla_model1(
    q,
    cache,
    *,
    indices,
    topk_length,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    extra_topk_length=None,
    out=None,
    lse=None,
    softmax_scale=None,
    parts=None,
):
    """Compute attention; every call regenerates its GPU schedule.

    One precision policy and one pipeline: every request goes through the same
    warp-specialised BF16 path. ``parts`` is a diagnostic override; the public
    entry uses one partition per multiprocessor.
    """
    batch = q.shape[0]
    expected_shape = (batch, 1, VALIDATED_HEADS, HEAD_DIM.value)
    if (
        tuple(q.shape) != expected_shape
        or not q.is_contiguous()
        or q.dtype != torch.bfloat16
    ):
        raise ValueError("MODEL1 requires contiguous BF16 [B, 1, 64, 512] queries")
    if not indices.is_contiguous() or indices.shape[:2] != (batch, 1):
        raise ValueError("Sparse indices must be contiguous [B, 1, topk]")
    page_log2 = _page_log2(cache)
    extra = extra_k_cache is not None
    extra_page_log2 = _page_log2(extra_k_cache) if extra else 0
    if parts is None:
        parts = torch.musa.get_device_properties(q.device).multi_processor_count
    if parts <= 0 or batch <= 0:
        raise ValueError("Batch and partition counts must be positive")
    _, _, heads, dimension = q.shape
    output = torch.empty_like(q) if out is None else out
    # Both buffers may come from the caller; vLLM-side integrations pass their
    # own so the result lands where the rest of the model expects it.
    if lse is None:
        lse = torch.empty((batch, heads, 1), dtype=torch.float32, device=q.device)
    elif (
        lse.numel() != batch * heads
        or lse.dtype != torch.float32
        or not lse.is_contiguous()
    ):
        raise ValueError("lse must be contiguous FP32 with batch * heads elements")
    # softmax_scale is the value before the change of base, matching the
    # vLLM and FlagGems-vllm convention (`qk * (sm_scale * LOG2E)`); the log2(e)
    # factor is applied here so the kernel's exp2 takes it directly. The default
    # comes from the query's own head dimension, as it does upstream.
    scale = dimension**-0.5 if softmax_scale is None else float(softmax_scale)
    score_to_log2 = scale * math.log2(math.e)
    metadata = torch.empty(
        (parts, METADATA_STRIDE.value), dtype=torch.int32, device=q.device
    )
    prefix = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
    # Each partition boundary can introduce at most one extra partial result.
    partial = torch.empty(
        (batch + parts, heads, dimension), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty(
        (batch + parts, heads), dtype=torch.float32, device=q.device
    )
    effective_main = torch.empty_like(topk_length)
    effective_extra = torch.empty_like(extra_topk_length) if extra else None
    prepare_effective_lengths(
        q,
        cache,
        indices=indices,
        topk_length=topk_length,
        effective_lengths=effective_main,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length,
        effective_extra_lengths=effective_extra,
    )
    topk_length, extra_topk_length = effective_main, effective_extra
    make_schedule[(parts,)](
        topk_length,
        extra_topk_length if extra else topk_length,
        metadata,
        prefix,
        batch,
        parts,
        extra,
        triton.next_power_of_2(batch),
        num_warps=SCHEDULE_WARPS,
    )
    sink = attn_sink if attn_sink is not None else lse
    compiled = attention_scheduled_ws[(parts,)](
        q,
        cache,
        indices,
        topk_length,
        extra_k_cache if extra else cache,
        extra_indices_in_kvcache if extra else indices,
        extra_topk_length if extra else topk_length,
        metadata,
        prefix,
        partial,
        partial_lse,
        batch,
        indices.shape[-1],
        cache.shape[1],
        cache.stride(0),
        cache.shape[0],
        extra,
        extra_indices_in_kvcache.shape[-1] if extra else 0,
        extra_k_cache.shape[1] if extra else 0,
        extra_k_cache.stride(0) if extra else 0,
        extra_k_cache.shape[0] if extra else 0,
        output,
        lse,
        sink,
        attn_sink is not None,
        PAGE_LOG2=page_log2,
        EXTRA_PAGE_LOG2=extra_page_log2,
        HEADS=heads,
        SCORE_TO_LOG2=score_to_log2,
        num_warps=ATTENTION_WARPS,
        num_stages=1,
        extern_libs=extern_libs_model1_exp2(),
    )
    flash_mla_model1.last_compiled = compiled
    combine_splits[(batch, heads // COMBINE_HEADS_PER_CTA.value)](
        partial,
        partial_lse,
        prefix,
        sink,
        output,
        lse,
        attn_sink is not None,
        HEADS=heads,
        num_warps=ATTENTION_WARPS,
    )
    return output, lse


# ============================================================================
# FlagGems entry points
# Gate on PH1 and the shapes this pipeline is built for, then dispatch.
# ============================================================================

PAGE_TOKEN_BYTES = 584
MTHREADS_CAPABILITY = (3, 1)


def _cache_is_supported(cache):
    """Packed MODEL1 pages, 16B-aligned as the gathers require."""
    if cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        return False
    if cache.shape[-1] != PAGE_TOKEN_BYTES or cache.ndim != 4 or cache.shape[2] != 1:
        return False
    u8 = cache.view(torch.uint8)
    return (
        u8.stride(-1) == 1 and u8.stride(0) % 16 == 0 and u8.storage_offset() % 16 == 0
    )


def can_use_model1_mthreads(
    q, kv, indices, out, lse, extra_kv=None, extra_indices=None
):
    """Whether this pipeline can serve the call; callers fall back if not."""
    try:
        capability = torch.musa.get_device_capability(q.device.index or 0)
    except Exception:
        return False
    if tuple(capability) != MTHREADS_CAPABILITY:
        return False
    if q.dtype != torch.bfloat16 or q.ndim != 4:
        return False
    # Addressing assumes one query token per request, and the SQMMA tile plus
    # the shared-memory budget assume the full 64-head block.
    if q.shape[1] != 1 or q.shape[2] != VALIDATED_HEADS or q.shape[3] != HEAD_DIM.value:
        return False
    if indices is None or indices.shape[-1] % TOKEN_TILE.value != 0:
        return False
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous()
    ):
        return False
    if lse is not None and (lse.dtype != torch.float32 or not lse.is_contiguous()):
        return False
    if not _cache_is_supported(kv):
        return False
    if extra_kv is not None:
        if not _cache_is_supported(extra_kv):
            return False
        if extra_indices is None or extra_indices.shape[-1] % TOKEN_TILE.value != 0:
            return False
    return True


def sparse_decode_model1_mthreads(
    q,
    kv,
    indices,
    attn_sink=None,
    topk_length=None,
    extra_kv=None,
    extra_indices=None,
    extra_topk_length=None,
    out=None,
    lse=None,
    sm_scale=None,
):
    """Run the PH1 pipeline; the schedule is rebuilt on the GPU every call.

    ``sm_scale`` is the value before the change of base, matching the vLLM and
    FlagGems convention of ``qk * (sm_scale * LOG2E)``.
    """
    if topk_length is None:
        raise ValueError("MODEL1 sparse decode requires topk_length")
    if extra_kv is not None and (extra_indices is None or extra_topk_length is None):
        raise ValueError("An extra cache requires extra_indices and extra_topk_length")
    return flash_mla_model1(
        q,
        kv,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_k_cache=extra_kv,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
        out=out,
        lse=lse,
        softmax_scale=sm_scale,
    )

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MThreads FP8-state GDN with optional MP31 TLE kernels."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language import core
from triton.runtime.cache import get_cache_manager

from flaggems_vllm.ops.gdn_w8a16 import _gdn_step, _prepare_qk, _store_dynamic_fp8_state
from flaggems_vllm.ops.gdn_w8a16 import gdn_w8a16 as _gdn_w8a16
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

HAS_TLE = has_triton_tle(3, 6, 0)
if HAS_TLE:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

GDN_VECTOR_IR = r"""; Copyright 2026 FlagOS Contributors
; SPDX-License-Identifier: Apache-2.0
; MP31 four-lane FP32 primitives. Each i64 holds two FP32 register values.

declare <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8>)

declare <4 x i8> @llvm.musa.f2e4m3.rn.bst4(<4 x float>)

declare <4 x float> @llvm.musa.fma.f.bst4.src12(float, <4 x float>, <4 x float>)

declare <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float>, <4 x float>)

declare <4 x float> @llvm.musa.mul.f.bst4.src1(float, <4 x float>)

declare <4 x float> @llvm.musa.mul.fhf.bst4.src01(<4 x half>, <4 x float>)

define protected float @gdn_norm4(i64 %0, i32 %1) alwaysinline nounwind readnone {
  %3 = icmp eq i32 %1, 0
  %4 = bitcast i64 %0 to <4 x bfloat>
  %5 = fpext <4 x bfloat> %4 to <4 x float>
  %6 = bitcast i64 %0 to <4 x half>
  %7 = fpext <4 x half> %6 to <4 x float>
  %8 = select i1 %3, <4 x float> %7, <4 x float> %5
  %9 = tail call contract <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float> %8, <4 x float> %8)
  %10 = shufflevector <4 x float> %9, <4 x float> undef, <2 x i32> <i32 1, i32 2>
  %11 = shufflevector <4 x float> %9, <4 x float> undef, <2 x i32> <i32 0, i32 3>
  %12 = fadd contract <2 x float> %10, %11
  %13 = shufflevector <2 x float> %12, <2 x float> poison, <2 x i32> <i32 1, i32 undef>
  %14 = fadd contract <2 x float> %12, %13
  %15 = extractelement <2 x float> %14, i64 0
  ret float %15
}

define i64 @gdn_normalized_pair(i64 %a0, float %a1, i32 %a2, i32 %a3) alwaysinline nounwind readnone
  {
%h = bitcast i64 %a0 to <4 x half>
  %b = bitcast i64 %a0 to <4 x bfloat>
  %isbf = icmp ne i32 %a2, 0
  %hv = fpext <4 x half> %h to <4 x float>
  %bv = fpext <4 x bfloat> %b to <4 x float>
  %q = select i1 %isbf,<4 x float> %bv,<4 x float> %hv
  %v=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %a1, <4 x float> %q)
  %vv = bitcast <4 x float> %v to <2 x i64>
  %pair = extractelement <2 x i64> %vv, i32 %a3
  ret i64 %pair
}

define i64 @gdn_state_pair(i32 %a0, float %a1, float %a2, i32 %a3) alwaysinline nounwind readnone {
%raw=bitcast i32 %a0 to <4 x i8>
  %h=call <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8> %raw)
  %v= fpext <4 x half> %h to <4 x float>
  %scaled=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %a1,<4 x float> %v)
  %prior=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %a2,<4 x float> %scaled)
  %vv = bitcast <4 x float> %prior to <2 x i64>
  %pair = extractelement <2 x i64> %vv, i32 %a3
  ret i64 %pair
}

define float @gdn_dot4(i64 %a0, i64 %a1, i64 %a2, i64 %a3) alwaysinline nounwind readnone {
%xl = bitcast i64 %a0 to <2 x float>
  %xh = bitcast i64 %a1 to <2 x float>
  %x = shufflevector <2 x float> %xl, <2 x float> %xh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %yl = bitcast i64 %a2 to <2 x float>
  %yh = bitcast i64 %a3 to <2 x float>
  %y = shufflevector <2 x float> %yl, <2 x float> %yh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %z=call <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float> %x,<4 x float> %y)
  %lo=shufflevector <4 x float> %z,<4 x float> undef,<2 x i32> <i32 0,i32 2>
  %hi=shufflevector <4 x float> %z,<4 x float> undef,<2 x i32> <i32 1,i32 3>
  %t=fadd <2 x float> %lo,%hi
  %a=extractelement <2 x float> %t,i32 0
  %b=extractelement <2 x float> %t,i32 1
  %c=fadd float %a,%b
  ret float %c
}

define i64 @gdn_updated_pair(i64 %a0, i64 %a1, i64 %a2, i64 %a3, float %a4, i32 %a5) alwaysinline
  nounwind readnone {
%sl = bitcast i64 %a0 to <2 x float>
  %sh = bitcast i64 %a1 to <2 x float>
  %s = shufflevector <2 x float> %sl, <2 x float> %sh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %kl = bitcast i64 %a2 to <2 x float>
  %kh = bitcast i64 %a3 to <2 x float>
  %k = shufflevector <2 x float> %kl, <2 x float> %kh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %v=call <4 x float> @llvm.musa.fma.f.bst4.src12(float %a4,<4 x float> %k,<4 x float> %s)
  %vv = bitcast <4 x float> %v to <2 x i64>
  %pair = extractelement <2 x i64> %vv, i32 %a5
  ret i64 %pair
}

define float @gdn_absmax4(i64 %a0, i64 %a1) alwaysinline nounwind readnone {
%vl = bitcast i64 %a0 to <2 x float>
  %vh = bitcast i64 %a1 to <2 x float>
  %v = shufflevector <2 x float> %vl, <2 x float> %vh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %vi=bitcast <4 x float> %v to <4 x i32>
  %mag=and <4 x i32> %vi,<i32 2147483647,i32 2147483647,i32 2147483647,i32 2147483647>
  %lo=shufflevector <4 x i32> %mag,<4 x i32> undef,<2 x i32> <i32 0,i32 2>
  %hi=shufflevector <4 x i32> %mag,<4 x i32> undef,<2 x i32> <i32 1,i32 3>
  %test=icmp ugt <2 x i32> %lo,%hi
  %t=select <2 x i1> %test,<2 x i32> %lo,<2 x i32> %hi
  %a=extractelement <2 x i32> %t,i32 0
  %b=extractelement <2 x i32> %t,i32 1
  %test2=icmp ugt i32 %a,%b
  %m=select i1 %test2,i32 %a,i32 %b
  %r=bitcast i32 %m to float
  ret float %r
}

define i32 @gdn_encode4(i64 %a0, i64 %a1, float %a2) alwaysinline nounwind readnone {
%sl = bitcast i64 %a0 to <2 x float>
  %sh = bitcast i64 %a1 to <2 x float>
  %s = shufflevector <2 x float> %sl, <2 x float> %sh, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %v=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %a2,<4 x float> %s)
  %q=call <4 x i8> @llvm.musa.f2e4m3.rn.bst4(<4 x float> %v)
  %i=bitcast <4 x i8> %q to i32
  ret i32 %i
}

define i64 @gdn_scaled_query_pair(i64 %word,float %iq,float %scale,i32 %bf,i32 %which) alwaysinline
  nounwind readnone {
  %b=bitcast i64 %word to <4 x bfloat>
  %h=bitcast i64 %word to <4 x half>
  %bv=fpext <4 x bfloat> %b to <4 x float>
  %hv=fpext <4 x half> %h to <4 x float>
  %use_bf=icmp ne i32 %bf,0
  %v=select i1 %use_bf,<4 x float> %bv,<4 x float> %hv
  %norm=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %iq,<4 x float> %v)
  %scaled=call <4 x float> @llvm.musa.mul.f.bst4.src1(float %scale,<4 x float> %norm)
  %parts=bitcast <4 x float> %scaled to <2 x i64>
  %r=extractelement <2 x i64> %parts,i32 %which
  ret i64 %r
}

; Decode four contiguous E4M3 values without applying a common channel scale.
define i64 @gdn_decode_half4(i32 %bits) alwaysinline nounwind readnone {
  %v = bitcast i32 %bits to <4 x i8>
  %f = call <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8> %v)
  %r = bitcast <4 x half> %f to i64
  ret i64 %r
}
"""


@lru_cache(maxsize=1)
def vector_library() -> str:
    # The compiler expects a file. Its atomic cache keeps the package read-only
    # and gives concurrent workers the same immutable LLVM source.
    cache = get_cache_manager(hashlib.sha256(GDN_VECTOR_IR.encode()).hexdigest())
    filename = "gdn_w8a16_mp31.ll"
    path = cache.get_file(filename)
    if path is None:
        path = cache.put(GDN_VECTOR_IR, filename, binary=False)
    return path


@triton.jit
def store_scaled_state(values, State, Scales, offset, scale_offset, mask, valid_v):
    magnitude = tl.max(tl.abs(values), 1)
    safe = tl.where(magnitude > 0.0, magnitude, 448.0)
    scale = safe * (1.0 / 448.0)
    quantized = (values * (448.0 / safe)[:, None]).to(State.dtype.element_ty)
    tl.store(State + offset, quantized, mask)
    tl.store(Scales + scale_offset, scale, valid_v)


@libentry()
@triton.jit
def decode_single_tle(
    Q, Key, Value, State, Scales, Alog, A, Bias, B, Out, Index, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    HV: tl.constexpr = META[1]
    Q0: tl.constexpr = META[2]
    Q2: tl.constexpr = META[3]
    Q3: tl.constexpr = META[4]
    K0: tl.constexpr = META[5]
    K2: tl.constexpr = META[6]
    K3: tl.constexpr = META[7]
    V0: tl.constexpr = META[8]
    V2: tl.constexpr = META[9]
    V3: tl.constexpr = META[10]
    A0: tl.constexpr = META[11]
    A2: tl.constexpr = META[12]
    B0: tl.constexpr = META[13]
    B2: tl.constexpr = META[14]
    S0: tl.constexpr = META[15]
    S1: tl.constexpr = META[16]
    S2: tl.constexpr = META[17]
    S3: tl.constexpr = META[18]
    C0: tl.constexpr = META[19]
    C1: tl.constexpr = META[20]
    C2: tl.constexpr = META[21]
    O0: tl.constexpr = META[22]
    O2: tl.constexpr = META[23]
    O3: tl.constexpr = META[24]
    IX: tl.constexpr = META[25]
    SCALE: tl.constexpr = META[26]
    INDEXED: tl.constexpr = META[27]
    BV: tl.constexpr = META[28]
    chunks: tl.constexpr = 128 // BV
    sequence_head = tl.program_id(0) // chunks
    value_chunk = tl.program_id(0) % chunks
    sequence, head = sequence_head // HV, sequence_head % HV
    query_head = head // (HV // H)
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [1, 32], [4, 1], [1, 0])
    columns = value_chunk * BV + tle.gpu.set_layout(
        tl.arange(0, BV), tle.gpu.SlicedEncoding(1, layout)
    )
    reduction = tle.gpu.set_layout(tl.arange(0, 128), tle.gpu.SlicedEncoding(0, layout))
    valid_v = columns < 128
    mask = valid_v[:, None] & (reduction[None, :] < 128)
    slot = (
        tl.load(Index + sequence * IX).to(tl.int64)
        if INDEXED
        else sequence.to(tl.int64)
    )
    output_offset = sequence * O0 + head * O2 + columns * O3
    if slot < 0:
        tl.store(Out + output_offset, 0.0, valid_v)
        return
    state_offset = (
        slot * S0 + head * S1 + columns[:, None] * S2 + reduction[None, :] * S3
    )
    scale_offset = slot * C0 + head * C1 + columns * C2
    state = tl.load(State + state_offset, mask, 0.0).to(tl.float32)
    state *= tl.load(Scales + scale_offset, valid_v, 0.0)[:, None]
    query = tl.load(Q + sequence * Q0 + query_head * Q2 + reduction[None, :] * Q3).to(
        tl.float32
    )
    key = tl.load(Key + sequence * K0 + query_head * K2 + reduction[None, :] * K3).to(
        tl.float32
    )
    query *= tl.exp2(-0.5 * tl.log2(tl.sum(query * query, 1) + 1.0e-6))[:, None]
    key *= tl.exp2(-0.5 * tl.log2(tl.sum(key * key, 1) + 1.0e-6))[:, None]
    query *= SCALE
    alpha = tl.exp2(tl.load(Alog + head).to(tl.float32) * 1.4426950408889634)
    x = tl.load(A + sequence * A0 + head * A2).to(tl.float32) + tl.load(Bias + head).to(
        tl.float32
    )
    softplus = tl.where(
        x > 20.0,
        x,
        tl.log2(1.0 + tl.exp2(x * 1.4426950408889634)) * 0.6931471805599453,
    )
    decay = tl.exp2(-alpha * softplus * 1.4426950408889634)
    beta = tl.sigmoid(tl.load(B + sequence * B0 + head * B2).to(tl.float32))
    state *= decay
    value = tl.load(Value + sequence * V0 + head * V2 + columns * V3, valid_v, 0.0).to(
        tl.float32
    )
    delta = (value - tl.sum(state * key, 1)) * beta
    state = tl.fma(delta[:, None], key, state)
    result = tl.sum(state * query, 1)
    tl.store(Out + output_offset, result, valid_v)
    store_scaled_state(state, State, Scales, state_offset, scale_offset, mask, valid_v)


@core.extern
def norm4(x0, x1, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1],
        {
            (
                tl.uint64,
                tl.int32,
            ): ("gdn_norm4", tl.float32)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def normalized_pair(x0, x1, x2, x3, _semantic=None):
    """Four normalized FP32 K/Q elements; select the low or high register pair."""
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3],
        {
            (
                tl.uint64,
                tl.float32,
                tl.int32,
                tl.int32,
            ): ("gdn_normalized_pair", tl.uint64)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def scaled_query_pair(x0, x1, x2, x3, x4, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3, x4],
        {
            (tl.uint64, tl.float32, tl.float32, tl.int32, tl.int32): (
                "gdn_scaled_query_pair",
                tl.uint64,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def state_pair(x0, x1, x2, x3, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3],
        {
            (
                tl.uint32,
                tl.float32,
                tl.float32,
                tl.int32,
            ): ("gdn_state_pair", tl.uint64)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def dot4(x0, x1, x2, x3, _semantic=None):
    """Four FP32 products with pairwise summation."""
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3],
        {
            (
                tl.uint64,
                tl.uint64,
                tl.uint64,
                tl.uint64,
            ): ("gdn_dot4", tl.float32)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def updated_pair(x0, x1, x2, x3, x4, x5, _semantic=None):
    """Updated FP32 values are retained for both projection and quantization."""
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2, x3, x4, x5],
        {
            (
                tl.uint64,
                tl.uint64,
                tl.uint64,
                tl.uint64,
                tl.float32,
                tl.int32,
            ): ("gdn_updated_pair", tl.uint64)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def absmax4(x0, x1, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1],
        {
            (
                tl.uint64,
                tl.uint64,
            ): ("gdn_absmax4", tl.float32)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def encode4(x0, x1, x2, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x0, x1, x2],
        {
            (
                tl.uint64,
                tl.uint64,
                tl.float32,
            ): ("gdn_encode4", tl.uint32)
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def _rsqrt(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.float32,): ("llvm.musa.rsq", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def _raw_rcp(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.float32,): ("llvm.musa.rcp", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _reciprocal(x):
    return tl.where(x == float("inf"), 0.0, _raw_rcp(x))


@triton.jit
def decode_quart(
    Q,
    Key,
    Value,
    State,
    Scales,
    Alog,
    A,
    Bias,
    B,
    Out,
    Index,
    Desc,
    META: tl.constexpr,
    TME: tl.constexpr,
    LANES: tl.constexpr,
    INTERLEAVE_HEADS: tl.constexpr,
):
    # A lane owns four contiguous K values. A subgroup reduces one value row.
    H: tl.constexpr = META[0]
    HV: tl.constexpr = META[1]
    Q0: tl.constexpr = META[2]
    Q2: tl.constexpr = META[3]
    K0: tl.constexpr = META[5]
    K2: tl.constexpr = META[6]
    V0: tl.constexpr = META[8]
    V2: tl.constexpr = META[9]
    V3: tl.constexpr = META[10]
    A0: tl.constexpr = META[11]
    A2: tl.constexpr = META[12]
    B0: tl.constexpr = META[13]
    B2: tl.constexpr = META[14]
    S0: tl.constexpr = META[15]
    S1: tl.constexpr = META[16]
    S2: tl.constexpr = META[17]
    C0: tl.constexpr = META[19]
    C1: tl.constexpr = META[20]
    C2: tl.constexpr = META[21]
    O0: tl.constexpr = META[22]
    O2: tl.constexpr = META[23]
    O3: tl.constexpr = META[24]
    IX: tl.constexpr = META[25]
    SCALE: tl.constexpr = META[26]
    INDEXED: tl.constexpr = META[27]
    BV: tl.constexpr = META[28]
    if TME:
        # Describe the contiguous FP8 pool as words so each lane receives four codes.
        buf = tle.gpu.alloc((BV, 32), dtype=tl.float32, nv_mma_shared_layout=False)
        full = tle.gpu.alloc_barrier(expect_bytes=BV * 128)
    chunks: tl.constexpr = 128 // BV
    if INTERLEAVE_HEADS:
        head = tl.program_id(0) % HV
        value_chunk = (tl.program_id(0) // HV) % chunks
        sequence = tl.program_id(0) // (HV * chunks)
    else:
        sequence_head = tl.program_id(0) // chunks
        value_chunk = tl.program_id(0) % chunks
        sequence, head = sequence_head // HV, sequence_head % HV
    query_head = head // (HV // H)
    layout: tl.constexpr = tle.gpu.BlockEncoding(
        [1, 1], [32 // LANES, LANES], [4, 1], [1, 0]
    )
    columns = value_chunk * BV + tle.gpu.set_layout(
        tl.arange(0, BV), tle.gpu.SlicedEncoding(1, layout)
    )
    reduction = tle.gpu.set_layout(tl.arange(0, 32), tle.gpu.SlicedEncoding(0, layout))
    valid_v = columns < 128
    mask = valid_v[:, None] & (reduction[None, :] < 32)
    slot = (
        tl.load(Index + sequence * IX).to(tl.int64)
        if INDEXED
        else sequence.to(tl.int64)
    )
    al = tl.load(Alog + head).to(tl.float32)
    ga = tl.load(A + sequence * A0 + head * A2).to(tl.float32)
    gb = tl.load(B + sequence * B0 + head * B2).to(tl.float32)
    bi = tl.load(Bias + head).to(tl.float32)
    pwq = tl.load(
        (Q + sequence * Q0 + query_head * Q2).to(tl.pointer_type(tl.uint64))
        + reduction[None, :]
    )
    pwk = tl.load(
        (Key + sequence * K0 + query_head * K2).to(tl.pointer_type(tl.uint64))
        + reduction[None, :]
    )
    value = tl.load(Value + sequence * V0 + head * V2 + columns * V3, valid_v, 0.0).to(
        tl.float32
    )
    output_offset = sequence * O0 + head * O2 + columns * O3
    live = slot >= 0
    if TME and not live:
        tl.store(Out + output_offset, 0.0, valid_v)
        return
    valid_v = valid_v & live
    mask = mask & live
    slot = tl.maximum(slot, 0)
    state_offset = (
        slot * (S0 // 4)
        + head * (S1 // 4)
        + columns[:, None] * (S2 // 4)
        + reduction[None, :]
    )
    scale_offset = slot * C0 + head * C1 + columns * C2
    if TME:
        tile_row = ((slot * HV + head) * 128 + value_chunk * BV).to(tl.int32)
        tle.gpu.copy(Desc, buf, (BV, 32), (tile_row, 0), barrier=full)
        tle.gpu.barrier_wait(full, phaseIdx=0)
        r = tl.broadcast_to((columns - value_chunk * BV)[:, None], (BV, 32))
        c = tl.broadcast_to(reduction[None, :], (BV, 32))
        packed = tl.load(tle.gpu.local_ptr(buf, (r, c))).to(tl.uint32, bitcast=True)
    else:
        packed = tl.load(
            State.to(tl.pointer_type(tl.uint32)) + state_offset,
            mask,
            0,
        )
    # The device helpers unpack four activation/state elements per lane.
    bf = tl.full((), 1 if Q.dtype.element_ty == tl.bfloat16 else 0, tl.int32)
    nq = tl.reshape(tl.sum(norm4(pwq, bf), 1), ()) + 1.0e-6
    nk = tl.reshape(tl.sum(norm4(pwk, bf), 1), ()) + 1.0e-6
    iq = tl.where(nq == float("inf"), 0.0, _rsqrt(nq))
    ik = tl.where(nk == float("inf"), 0.0, _rsqrt(nk))
    alpha = tl.exp2(al * 1.4426950408889634)
    x = ga + bi
    softplus = tl.where(
        x > 20.0, x, tl.log2(1.0 + tl.exp2(x * 1.4426950408889634)) * 0.6931471805599453
    )
    decay = tl.exp2(-alpha * softplus * 1.4426950408889634)
    beta = _reciprocal(1.0 + tl.exp2(-gb * 1.4426950408889634))
    channel = tl.load(Scales + scale_offset, valid_v, 0.0)
    k0 = normalized_pair(pwk, ik, bf, 0)
    k1 = normalized_pair(pwk, ik, bf, 1)
    s0 = state_pair(packed, channel[:, None], decay, 0)
    s1 = state_pair(packed, channel[:, None], decay, 1)
    delta = (value - tl.sum(dot4(s0, s1, k0, k1), 1)) * beta
    t0 = updated_pair(s0, s1, k0, k1, delta[:, None], 0)
    t1 = updated_pair(s0, s1, k0, k1, delta[:, None], 1)
    if SCALE == 128**-0.5:
        q0 = normalized_pair(pwq, iq * SCALE, bf, 0)
        q1 = normalized_pair(pwq, iq * SCALE, bf, 1)
    else:
        q0 = scaled_query_pair(pwq, iq, tl.full((), SCALE, tl.float32), bf, 0)
        q1 = scaled_query_pair(pwq, iq, tl.full((), SCALE, tl.float32), bf, 1)
    result = tl.sum(dot4(t0, t1, q0, q1), 1)
    tl.store(Out + output_offset, tl.where(valid_v, result, 0.0), columns < 128)
    mag = tl.max(absmax4(t0, t1), 1)
    safe = tl.where(mag > 0.0, mag, 448.0)
    scale = safe * (1.0 / 448.0)
    quantized = encode4(t0, t1, (448.0 / safe)[:, None])
    tl.store(State.to(tl.pointer_type(tl.uint32)) + state_offset, quantized, mask)
    tl.store(Scales + scale_offset, scale, valid_v)


@core.extern
def decode_half4(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.uint32,): ("gdn_decode_half4", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def load_fp8_values4(
    S, slot, h, k, v4, SL0: tl.constexpr, SL1: tl.constexpr, SL2: tl.constexpr
):
    x = tl.load(
        S.to(tl.pointer_type(tl.uint32))
        + (slot * SL0 + h * SL1 + k[:, None] * SL2) // 4
        + v4[None, :]
    )
    d = decode_half4(x)
    a = (d & 65535).to(tl.uint16).to(tl.float16, bitcast=True)
    b = ((d >> 16) & 65535).to(tl.uint16).to(tl.float16, bitcast=True)
    c = ((d >> 32) & 65535).to(tl.uint16).to(tl.float16, bitcast=True)
    e = ((d >> 48) & 65535).to(tl.uint16).to(tl.float16, bitcast=True)
    return tl.interleave(tl.interleave(a, c), tl.interleave(b, e)).to(tl.float32)


@triton.jit
def store_fp8_values4(
    S,
    values,
    slot,
    h,
    k,
    v4,
    SL0: tl.constexpr,
    SL1: tl.constexpr,
    SL2: tl.constexpr,
    BV: tl.constexpr,
):
    ev, od = tl.split(tl.reshape(values, (128, BV // 2, 2)))
    a, c = tl.split(tl.reshape(ev, (128, BV // 4, 2)))
    b, d = tl.split(tl.reshape(od, (128, BV // 4, 2)))
    lo = a.to(tl.uint32, bitcast=True).to(tl.uint64) | (
        b.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    )
    hi = c.to(tl.uint32, bitcast=True).to(tl.uint64) | (
        d.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    )
    q = encode4(lo, hi, tl.full((), 1.0, tl.float32))
    tl.store(
        S.to(tl.pointer_type(tl.uint32))
        + (slot * SL0 + h * SL1 + k[:, None] * SL2) // 4
        + v4[None, :],
        q,
    )


@triton.jit
def _chunk_map(
    Cu, Map, T: tl.constexpr, NS: tl.constexpr, BLOCK: tl.constexpr, COUNT: tl.constexpr
):
    c = tl.program_id(0) * 32 + tl.arange(0, 32)
    si = tl.arange(0, triton.next_power_of_2(NS))
    starts = tl.load(Cu + si, si < NS, 0).to(tl.int32)
    n = tl.max(
        tl.where(
            (si < NS)[None, :] & ((starts // BLOCK + si)[None, :] <= c[:, None]),
            si[None, :],
            0,
        ),
        1,
    )
    bos = tl.load(Cu + n).to(tl.int32)
    end = tl.load(Cu + n + 1).to(tl.int32)
    beg = bos + (c - (bos // BLOCK + n)) * BLOCK
    tl.store(Map + c * 4, n, c < COUNT)
    tl.store(Map + c * 4 + 1, bos, c < COUNT)
    tl.store(Map + c * 4 + 2, end, c < COUNT)
    tl.store(Map + c * 4 + 3, beg, c < COUNT)


@triton.jit
def _window_output(
    Q,
    K,
    V,
    G,
    B,
    S,
    Sc,
    Cu,
    Index,
    O,
    Map,
    Dirty,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    NS: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK: tl.constexpr,
    SL0: tl.constexpr,
    SL1: tl.constexpr,
    SL2: tl.constexpr,
    SL3: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    SCALE: tl.constexpr,
    VAR: tl.constexpr,
    INDEXED: tl.constexpr,
    WITH_INITIAL: tl.constexpr,
    QDesc,
    Bounds,
    RAW: tl.constexpr,
):
    if RAW and not WITH_INITIAL:
        ready = tle.gpu.alloc_barrier(expect_bytes=BLOCK * 128 * 2)
    (c, h, tv) = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    if WITH_INITIAL:
        n = c
        if VAR:
            bos = tl.load(Cu + n).to(tl.int32)
            end = tl.load(Cu + n + 1).to(tl.int32)
        else:
            bos = n * T
            end = bos + T
        beg = bos
    elif VAR and NS != 1:
        n = tl.load(Map + c * 4)
        bos = tl.load(Map + c * 4 + 1)
        end = tl.load(Map + c * 4 + 2)
        beg = tl.load(Map + c * 4 + 3)
    elif VAR:
        bos = tl.load(Cu).to(tl.int32)
        end = tl.load(Cu + 1).to(tl.int32)
        n = 0
        beg = bos + c * BLOCK
    else:
        n = c // tl.cdiv(T, BLOCK)
        bos = n * T
        end = bos + T
        beg = bos + c % tl.cdiv(T, BLOCK) * BLOCK
    if beg >= end:
        return
    if not WITH_INITIAL and beg < bos + 64:
        return
    slot = tl.load(Index + n).to(tl.int64) if INDEXED else n.to(tl.int64)
    seqstart = beg - (BT - BLOCK)
    if seqstart < bos:
        seqstart = bos
    i = tl.arange(0, BT)
    j = tl.arange(0, BT if WITH_INITIAL else BLOCK)
    k = tl.arange(0, 128)
    v = tv * BV + tl.arange(0, BV)
    t = seqstart + i
    to = beg + j
    qh = h // (HV // H)
    if slot < 0:
        tl.store(O + (to[:, None] * HV + h) * 128 + v[None, :], 0.0, to[:, None] < end)
        return
    kk = tl.load(K + (t[:, None] * H + qh) * 128 + k[None, :], t[:, None] < end, 0.0)
    if not RAW or WITH_INITIAL:
        qq = tl.load(
            Q + (to[:, None] * H + qh) * 128 + k[None, :], to[:, None] < end, 0.0
        )
        vv = tl.load(
            V + (t[:, None] * HV + h) * 128 + v[None, :], t[:, None] < end, 0.0
        )
    else:
        value_max = tl.load(Bounds + t * HV + h, t < end, 0.0)
    gv = tl.load(G + t * HV + h, t < end, 0.0).to(tl.float32)
    bv = tl.load(B + t * HV + h, t < end, 0.0).to(tl.float32)
    pref = tl.cumsum(gv, 0)
    # Center exponents to form exp(prefix[i]-prefix[j]) as an outer
    # product without exponentiating a whole matrix.
    bottom = tl.min(pref, 0)
    center = 0.5 * bottom
    left = tl.exp(tl.minimum(80.0, pref - center))
    right = tl.exp(tl.minimum(80.0, center - pref))
    tri = i[:, None] > i[None, :]
    if RAW and not WITH_INITIAL:
        # Reuse operand A after its K products complete. The descriptor copy
        # preserves the SQMMA swizzle; raw shared-memory pointers do not.
        key_slots = tle.gpu.alloc(
            (2, BLOCK, 128),
            K.dtype.element_ty,
            init_value=tl.reshape(kk, (2, BLOCK, 128)),
            nv_mma_shared_layout=True,
        )
        key_a = key_slots.slot(0)
        key_b = tle.gpu.alloc(
            (BT, 128), K.dtype.element_ty, init_value=kk, nv_mma_shared_layout=True
        )
        gram_first = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(key_a, key_b, trans_b=True))
        gram_last = tle.gpu.wgmma_wait(
            0, tle.gpu.wgmma(key_slots.slot(1), key_b, trans_b=True)
        )
        tle.gpu.copy(QDesc, key_a, (BLOCK, 128), (beg, qh * 128), barrier=ready)
        tle.gpu.barrier_wait(ready, phaseIdx=0)
        qgram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(key_a, key_b, trans_b=True))
        gram = tl.reshape(
            tl.permute(tl.join(gram_first, gram_last), (2, 0, 1)), (BT, BT)
        )
    else:
        kk0 = tle.gpu.alloc(
            (BT, 128), K.dtype.element_ty, init_value=kk, nv_mma_shared_layout=True
        )
        kk1 = tle.gpu.alloc(
            (BT, 128), K.dtype.element_ty, init_value=kk, nv_mma_shared_layout=True
        )
        qq0 = tle.gpu.alloc(
            (BT if WITH_INITIAL else BLOCK, 128),
            Q.dtype.element_ty,
            init_value=qq,
            nv_mma_shared_layout=True,
        )
        a = tle.gpu.wgmma(kk0, kk1, trans_b=True)
        gram = tle.gpu.wgmma_wait(0, a)
        cmat = tle.gpu.wgmma(qq0, kk1, trans_b=True)
        qgram = tle.gpu.wgmma_wait(0, cmat)
    L = tl.where(tri, -gram * (left[:, None] * right[None, :]) * bv[:, None], 0.0).to(
        tl.float16
    )
    causal = to[:, None] >= t[None, :]
    po = tl.gather(pref, tl.minimum(j + (beg - seqstart), BT - 1), 0)
    qk = tl.where(
        causal,
        qgram
        * (
            tl.gather(left, tl.minimum(j + (beg - seqstart), BT - 1), 0)[:, None]
            * right[None, :]
        ),
        0.0,
    ).to(tl.float16)
    if RAW:
        ggood = 1
    else:
        ggood = tl.min(
            ((gv >= -80.0) & (gv <= 0.0) & (bv >= 0.0) & (bv <= 1.0)).to(tl.int32), 0
        )
    if RAW and not WITH_INITIAL:
        value_ok = tl.max(value_max, 0) <= 1.0
    else:
        value_ok = tl.min((tl.abs(vv.to(tl.float32)) <= 1.0).to(tl.int32))
    # With normalized keys and beta in [0,1], the state transition is
    # non-expansive. Each accepted full block attenuates prior state by
    # exp(-14) or less, before adding at most 32 bounded updates.
    block_log = tl.sum(tl.where((t >= beg) & (t < beg + BLOCK), gv, 0.0), 0)
    bad = (
        (ggood == 0)
        | (value_ok == 0)
        | (beg + BLOCK <= end) & (block_log > -14.0)
        | (tl.abs(SCALE) > 0.1)
    )
    # Use the measured row norm in the Neumann remainder bound;
    # the accepted output residual budget is 1.2e-4.
    lnorm = tl.max(tl.sum(tl.abs(L).to(tl.float32), 1), 0)
    bad = bad | ~(lnorm <= 0.5) | ~(bottom >= -140.0)
    if WITH_INITIAL:
        bad = bad | (bos + 64 <= end) & (tl.sum(tl.where(i >= 32, gv, 0.0), 0) > -14.0)
    if not RAW or WITH_INITIAL:
        rhs = vv.to(tl.float32)
    base = tl.full((BT if WITH_INITIAL else BLOCK, BV), 0.0, tl.float32)
    if WITH_INITIAL:
        packed: tl.constexpr = (
            SL3 == 1 and SL0 % 4 == 0 and SL1 % 4 == 0 and SL2 % 4 == 0
        )
        if packed:
            s = (
                load_fp8_values4(
                    S, slot, h, k, tv * (BV // 4) + tl.arange(0, BV // 4), SL0, SL1, SL2
                )
                * tl.load(Sc + slot * C0 + h * C1 + v * C2)[None, :]
            )
        else:
            s = (
                tl.load(
                    S + slot * SL0 + h * SL1 + k[:, None] * SL2 + v[None, :] * SL3
                ).to(tl.float32)
                * tl.load(Sc + slot * C0 + h * C1 + v * C2)[None, :]
            )
        bad = bad | ~(tl.max(tl.abs(s)) <= 4.0)
        sf = s.to(tl.float16)
        rhs -= tl.dot(kk.to(tl.float16), sf) * tl.exp(pref)[:, None]
        base = tl.dot(qq.to(tl.float16), sf) * tl.exp(po)[:, None]
    if not RAW or WITH_INITIAL:
        val = (rhs * bv[:, None]).to(tl.float16)
    q0 = qk.to(tl.float16)
    e1 = tl.dot(q0, L)
    row_norm = tl.sum(tl.abs(L).to(tl.float32), 1)
    if RAW and not WITH_INITIAL:
        magnitude = tl.max(value_max * tl.abs(bv), 0) * tl.abs(SCALE)
    else:
        magnitude = tl.max(tl.abs(val).to(tl.float32)) * tl.abs(SCALE)
    residual_budget = 1.2e-4 * (1.0 - lnorm)
    if (
        tl.max(tl.sum(tl.abs(e1) * row_norm[None, :], 1), 0) * magnitude
        <= residual_budget
    ):
        qk = (q0 + e1).to(tl.float16)
    else:
        e2 = tl.dot(e1.to(tl.float16), L)
        residual = tl.max(tl.sum(tl.abs(e2) * row_norm[None, :], 1), 0)
        bad = bad | ~(residual * magnitude <= residual_budget)
        qk = (q0 + e1 + e2).to(tl.float16)
    if RAW and not WITH_INITIAL:
        vv = tl.load(
            V + (t[:, None] * HV + h) * 128 + v[None, :], t[:, None] < end, 0.0
        )
        val = (vv.to(tl.float32) * bv[:, None]).to(tl.float16)
    x = val.to(tl.float32)
    # Each CTA only marks a head when its window is unsafe. The next
    # kernel observes all marks after the predictor launches complete.
    if bad:
        tl.atomic_or(Dirty + n * HV + h, 1, sem="relaxed")
    out = base + tl.dot(qk, x.to(tl.float16))
    tl.store(
        O + (to[:, None] * HV + h) * 128 + v[None, :], out * SCALE, to[:, None] < end
    )


@triton.jit
def _clear_dirty(Dirty, COUNT: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Dirty + i, 0, i < COUNT)


@triton.jit
def _try_final(
    K,
    V,
    G,
    B,
    S,
    Sc,
    Cu,
    Index,
    Bad,
    Retry,
    Final,
    FinalScale,
    FM: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BV: tl.constexpr,
    SL: tl.constexpr,
    CS: tl.constexpr,
    VAR: tl.constexpr,
    IDX: tl.constexpr,
):
    n, h, tv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(Cu + n).to(tl.int32) if VAR else n * T
    end = tl.load(Cu + n + 1).to(tl.int32) if VAR else bos + T
    slot = tl.load(Index + n).to(tl.int64) if IDX else n.to(tl.int64)
    flag = Retry + (n * HV + h) * (128 // BV) + tv
    if slot < 0 or bos >= end or tl.load(Bad + n * HV + h) != 0:
        tl.store(flag, 1)
        return
    z = tl.arange(0, 64)
    k = tl.arange(0, 128)
    v = tv * BV + tl.arange(0, BV)
    t = end - 64 + z
    g = tl.load(G + t * HV + h, t >= bos, 0.0).to(tl.float32)
    b = tl.load(B + t * HV + h, t >= bos, 0.0).to(tl.float32)
    kv = tl.load(
        K + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :], t[:, None] >= bos, 0.0
    ).to(tl.float16)
    values = tl.load(
        V + (t[:, None] * HV + h) * 128 + v[None, :], t[:, None] >= bos, 0.0
    ).to(tl.float32)
    pref = tl.cumsum(g, 0)
    total = tl.sum(g, 0)
    rhs = values * b[:, None]
    initial = tl.full((128, BV), 0.0, tl.float32)
    if end - bos <= 64:
        initial = (
            tl.load(
                S + slot * SL[0] + h * SL[1] + k[:, None] * SL[2] + v[None, :] * SL[3]
            ).to(tl.float32)
            * tl.load(Sc + slot * CS[0] + h * CS[1] + v * CS[2])[None, :]
        )
        sh = initial.to(tl.float16)
        rhs -= (
            b[:, None]
            * tl.exp(pref)[:, None]
            * (
                tl.dot(kv, sh)
                + tl.dot(kv, (initial - sh.to(tl.float32)).to(tl.float16))
            )
        )
    tri = z[:, None] > z[None, :]
    L = tl.where(
        tri,
        -tl.dot(kv, tl.trans(kv))
        * b[:, None]
        * tl.exp(tl.where(tri, pref[:, None] - pref[None, :], 0.0)),
        0.0,
    )
    lh = L.to(tl.float16)
    y = rhs
    for j in tl.static_range(2):
        y = rhs + tl.dot(lh, y.to(tl.float16))
    yh = y.to(tl.float16)
    refined = (
        rhs
        + tl.dot(lh, yh)
        + tl.dot((L - lh.to(tl.float32)).to(tl.float16), yh)
        + tl.dot(lh, (y - yh.to(tl.float32)).to(tl.float16))
    )
    weights = tl.exp(total - pref)
    w = refined * (64.0 * weights)[:, None]
    wh = w.to(tl.float16)
    st = initial * tl.exp(total) + (
        tl.dot(tl.trans(kv), wh)
        + tl.dot(tl.trans(kv), (w - wh.to(tl.float32)).to(tl.float16))
    ) * (1.0 / 64.0)
    mag = tl.max(tl.abs(st), 0)
    # Residuals act as additive rank-one updates. The normalized-key recurrent
    # transition is nonexpansive; weight every residual by its remaining decay
    # instead of applying the largest early residual to the entire tail. This
    # bound does not require the triangular matrix to contract in infinity norm.
    residual_weights = tl.sum(weights[:, None] * tl.abs(L), 0)
    err = 1.01 * tl.sum(residual_weights[:, None] * tl.abs(refined - y), 0)
    discarded = tl.where(end - bos > 64, 130.0 * tl.exp(total), 0.0)
    error = err + discarded
    accepted = tl.min((error * 4096.0 <= mag).to(tl.int32), 0) != 0
    if not accepted:
        safe_bound = tl.where(mag > 0.0, mag, 448.0)
        rounded = (st * (448.0 / safe_bound)[None, :]).to(S.dtype.element_ty).to(
            tl.float32
        ) * (safe_bound / 448.0)[None, :]
        quant_error = tl.max(tl.abs(rounded - st), 0)
        # Use actual FP8 rounding distance. This is stricter than the existing
        # channel-normalized bound, including uncertainty of reference amax.
        tau = 16.0 / 448.0 + 4.0e-4
        accepted = (
            tl.min((quant_error + (1.0 + tau) * error <= tau * mag).to(tl.int32), 0)
            != 0
        )
    if not accepted:
        tl.store(flag, 1)
        return
    tl.store(flag, 0)
    safe = tl.where(mag > 0.0, mag, 448.0)
    packed: tl.constexpr = (
        FM[3] == 1 and FM[0] % 4 == 0 and FM[1] % 4 == 0 and FM[2] % 4 == 0
    )
    if packed:
        store_fp8_values4(
            Final,
            st * (448.0 / safe)[None, :],
            slot,
            h,
            k,
            tv * (BV // 4) + tl.arange(0, BV // 4),
            FM[0],
            FM[1],
            FM[2],
            BV,
        )
    else:
        tl.store(
            Final + slot * FM[0] + h * FM[1] + k[:, None] * FM[2] + v[None, :] * FM[3],
            (st * (448.0 / safe)[None, :]).to(S.dtype.element_ty),
        )
    tl.store(FinalScale + slot * FM[4] + h * FM[5] + v * FM[6], safe / 448.0)


@triton.jit
def _finish(
    K,
    V,
    G,
    B,
    S,
    Sc,
    Cu,
    Index,
    Q,
    O,
    Bad,
    Retry,
    Final,
    FinalScale,
    FMETA: tl.constexpr,
    COPY_EMPTY: tl.constexpr,
    Scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    NS: tl.constexpr,
    BV: tl.constexpr,
    SL0: tl.constexpr,
    SL1: tl.constexpr,
    SL2: tl.constexpr,
    SL3: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    VAR: tl.constexpr,
    INDEXED: tl.constexpr,
    WINDOW: tl.constexpr,
    FINAL_BV: tl.constexpr = 32,
    PARTIAL: tl.constexpr = False,
    INDEX_STRIDE: tl.constexpr = 1,
):
    n, h, tv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(Cu + n).to(tl.int32) if VAR else n * T
    end = tl.load(Cu + n + 1).to(tl.int32) if VAR else bos + T
    slot = tl.load(Index + n * INDEX_STRIDE).to(tl.int64) if INDEXED else n.to(tl.int64)
    if slot < 0 or (bos >= end and not COPY_EMPTY):
        return
    if tl.load(Retry + (n * HV + h) * (128 // FINAL_BV) + tv // (FINAL_BV // BV)) == 0:
        return
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [1, 32], [4, 1], [1, 0])
    k = tle.gpu.set_layout(tl.arange(0, 128), tle.gpu.SlicedEncoding(0, layout))
    v = tv * BV + tle.gpu.set_layout(
        tl.arange(0, BV), tle.gpu.SlicedEncoding(1, layout)
    )
    hq = h // (HV // H)
    so = slot * SL0 + h * SL1 + k[None, :] * SL2 + v[:, None] * SL3
    co = slot * C0 + h * C1 + v * C2
    if COPY_EMPTY and bos == end:
        dst = (
            slot * FMETA[0]
            + h * FMETA[1]
            + k[None, :] * FMETA[2]
            + v[:, None] * FMETA[3]
        )
        tl.store(Final + dst, tl.load(S + so))
        tl.store(
            FinalScale + slot * FMETA[4] + h * FMETA[5] + v * FMETA[6], tl.load(Sc + co)
        )
        return
    if PARTIAL:
        fallback = (
            tl.load(Bad + (n * HV + h) * (128 // FINAL_BV) + tv // (FINAL_BV // BV))
            != 0
        )
    else:
        fallback = tl.load(Bad + n * HV + h) != 0
    start = tl.where(fallback, bos, tl.maximum(bos, end - WINDOW))
    s = tl.full((BV, 128), 0.0, tl.float32)
    if start == bos:
        s = tl.load(S + so).to(tl.float32) * tl.load(Sc + co)[:, None]
    for t in range(start, end):
        kk = tl.load(K + (t * H + hq) * 128 + k).to(tl.float32)
        vv = tl.load(V + (t * HV + h) * 128 + v).to(tl.float32)
        decay = tl.exp(tl.load(G + t * HV + h).to(tl.float32))
        beta = tl.load(B + t * HV + h).to(tl.float32)
        s *= decay
        dif = (vv - tl.sum(s * kk[None, :], 1)) * beta
        s = tl.fma(dif[:, None], kk[None, :], s)
        if fallback:
            qq = tl.load(Q + (t * H + hq) * 128 + k).to(tl.float32)
            tl.store(O + (t * HV + h) * 128 + v, tl.sum(s * (qq * Scale)[None, :], 1))
    m = tl.max(tl.abs(s), 1)
    if not fallback and start > bos:
        z = tl.arange(0, WINDOW)
        ll = tl.sum(tl.load(G + (start + z) * HV + h).to(tl.float32), 0)
        # Re-evaluate rows whose admissible truncation error is not negligible
        # against the final channel scale. A 64-token tail spans two guarded blocks.
        if (
            tl.min(tl.broadcast_to(m[:, None], (BV, 128)), axis=None)
            < 130.0 * tl.exp(ll) * 4096.0
        ):
            s = tl.load(S + so).to(tl.float32) * tl.load(Sc + co)[:, None]
            for ti in range(bos, end):
                kv = tl.load(K + (ti * H + hq) * 128 + k).to(tl.float32)
                vv = tl.load(V + (ti * HV + h) * 128 + v).to(tl.float32)
                s *= tl.exp(tl.load(G + ti * HV + h).to(tl.float32))
                dif = (vv - tl.sum(s * kv[None, :], 1)) * tl.load(B + ti * HV + h).to(
                    tl.float32
                )
                s = tl.fma(dif[:, None], kv[None, :], s)
            m = tl.max(tl.abs(s), 1)
    safe = tl.where(m > 0.0, m, 448.0)
    tl.store(
        Final
        + slot * FMETA[0]
        + h * FMETA[1]
        + k[None, :] * FMETA[2]
        + v[:, None] * FMETA[3],
        (s * (448.0 / safe)[:, None]).to(S.dtype.element_ty),
    )
    tl.store(
        FinalScale + slot * FMETA[4] + h * FMETA[5] + v * FMETA[6], safe * (1.0 / 448.0)
    )


@triton.jit
def _split_dot(a, b):
    ah = a.to(tl.float16)
    bh = b.to(tl.float16)
    z = tl.dot(ah, bh)
    if a.dtype == tl.float32:
        z += tl.dot((a - ah.to(tl.float32)).to(tl.float16), bh)
    if b.dtype == tl.float32:
        z += tl.dot(ah, (b - bh.to(tl.float32)).to(tl.float16))
    return z


@triton.jit
def _short_factors(
    Q,
    K,
    G,
    B,
    F,
    Cu,
    VAR: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    NC: tl.constexpr = 1,
    MAX_LEN: tl.constexpr = 0,
    EXTENDED: tl.constexpr = False,
):
    bid = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2) if EXTENDED else 0
    bos = tl.load(Cu + n).to(tl.int32) if VAR else n * T
    beg = bos + bid * BT
    end = tl.load(Cu + n + 1).to(tl.int32) if VAR else bos + T
    if EXTENDED and (end <= bos or end - bos > MAX_LEN):
        return
    i = tl.arange(0, BT)
    t = beg + i
    k = tl.arange(0, 128)
    kk = tl.load(
        K + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :], t[:, None] < end, 0.0
    ).to(tl.float16)
    qq = tl.load(
        Q + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :], t[:, None] < end, 0.0
    ).to(tl.float16)
    g = tl.load(G + t * HV + h, t < end, 0.0).to(tl.float32)
    b = tl.load(B + t * HV + h, t < end, 0.0).to(tl.float32)
    pref = tl.cumsum(g, 0)
    tri = i[:, None] >= i[None, :]
    d = tl.exp(tl.where(tri, pref[:, None] - pref[None, :], 0.0))
    ka = tle.gpu.alloc((BT, 128), tl.float16, init_value=kk, nv_mma_shared_layout=True)
    kb = tle.gpu.alloc((BT, 128), tl.float16, init_value=kk, nv_mma_shared_layout=True)
    qa = tle.gpu.alloc((BT, 128), tl.float16, init_value=qq, nv_mma_shared_layout=True)
    gram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(ka, kb, trans_b=True))
    qgram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(qa, kb, trans_b=True))
    L = tl.where(i[:, None] > i[None, :], -gram * d * b[:, None], 0.0)
    qk = tl.where(tri, qgram * d, 0.0)
    eye = (i[:, None] == i[None, :]).to(tl.float32)
    # Use nilpotence: (I-L)^-1 is the product of I+L^(2^j).
    P = L.to(tl.float16)
    pow2 = tl.dot(P, P).to(tl.float16)
    A = eye + L + tl.dot((eye + L).to(tl.float16), pow2)
    pow4 = tl.dot(pow2, pow2).to(tl.float16)
    A += tl.dot(A.to(tl.float16), pow4)
    if EXTENDED:
        power8 = tl.dot(pow4, pow4).to(tl.float16)
        A += tl.dot(A.to(tl.float16), power8)
        # After I+...+L^15, the omitted series starts at L^16. Bound its
        # row norm by ||A|| * ||L^8||^2 / (1-||L^8||).
        valid = (tl.max(tl.sum(tl.abs(power8).to(tl.float32), 1), 0) <= 1e-4) & (
            tl.max(tl.sum(tl.abs(A), 1), 0) <= 8.0
        )
    else:
        valid = tl.max(tl.sum(tl.abs(pow4).to(tl.float32), 1), 0) <= 0.0009
    if not valid:
        A = tl.full((BT, BT), float("nan"), tl.float32)
    xf = A * b[None, :]
    off = (
        F
        + ((n * NC + bid) * HV + h) * (2 * BT * BT + 2 * BT)
        + i[:, None] * BT
        + i[None, :]
    )
    tl.store(off, xf)
    tl.store(off + BT * BT, qk)
    tl.store(
        F + ((n * NC + bid) * HV + h) * (2 * BT * BT + 2 * BT) + 2 * BT * BT + i,
        tl.exp(pref),
    )
    tl.store(
        F + ((n * NC + bid) * HV + h) * (2 * BT * BT + 2 * BT) + 2 * BT * BT + BT + i,
        tl.exp(tl.sum(g, 0) - pref),
    )


@triton.jit
def _short_apply(
    Q,
    K,
    V,
    F,
    G,
    B,
    FUSED: tl.constexpr,
    S,
    Sc,
    I,
    Cu,
    VAR: tl.constexpr,
    O,
    Final,
    FinalScale,
    FM: tl.constexpr,
    IDX: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    SL: tl.constexpr,
    CS: tl.constexpr,
    Scale: tl.constexpr,
    COPY_EMPTY: tl.constexpr,
    Retry,
    NC: tl.constexpr = 0,
    MAX_LEN: tl.constexpr = 0,
    IX: tl.constexpr = 1,
    EXTENDED: tl.constexpr = False,
):
    CHUNKS: tl.constexpr = NC if EXTENDED else triton.cdiv(T, BT)
    LENGTH: tl.constexpr = MAX_LEN if EXTENDED else T
    n = tl.program_id(0)
    h = tl.program_id(1)
    vv = tl.program_id(2) * BV + tl.arange(0, BV)
    sidx = tl.load(I + n * IX).to(tl.int64) if IDX else n.to(tl.int64)
    beg = tl.load(Cu + n).to(tl.int32) if VAR else n * T
    end = tl.load(Cu + n + 1).to(tl.int32) if VAR else beg + T
    i = tl.arange(0, BT)
    t = beg + i
    k = tl.arange(0, 128)
    if sidx < 0:
        for z in range(tl.cdiv(end - beg, BT)):
            tok = beg + z * BT + i
            tl.store(
                O + (tok[:, None] * HV + h) * 128 + vv[None, :], 0.0, tok[:, None] < end
            )
        return

    so = S + sidx * SL[0] + h * SL[1] + k[:, None] * SL[2] + vv[None, :] * SL[3]
    co = Sc + sidx * CS[0] + h * CS[1] + vv * CS[2]
    if end <= beg:
        tl.store(Retry + (n * HV + h) * (128 // BV) + tl.program_id(2), 0)
        if COPY_EMPTY:
            tl.store(
                Final
                + sidx * FM[0]
                + h * FM[1]
                + k[:, None] * FM[2]
                + vv[None, :] * FM[3],
                tl.load(so),
            )
            tl.store(FinalScale + sidx * FM[4] + h * FM[5] + vv * FM[6], tl.load(co))
        return
    if EXTENDED and end - beg > MAX_LEN:
        tl.store(Retry + (n * HV + h) * (128 // BV) + tl.program_id(2), 1)
        return
    packed_input: tl.constexpr = (
        (T > 64 or FUSED)
        and SL[3] == 1
        and SL[0] % 4 == 0
        and SL[1] % 4 == 0
        and SL[2] % 4 == 0
    )
    v4 = tl.program_id(2) * (BV // 4) + tl.arange(0, BV // 4)
    if packed_input:
        st = (
            load_fp8_values4(S, sidx, h, k, v4, SL[0], SL[1], SL[2])
            * tl.load(co)[None, :]
        )
    else:
        st = tl.load(so).to(tl.float32) * tl.load(co)[None, :]
    if EXTENDED:
        initial_mag = tl.max(tl.abs(st))
    for z in tl.static_range(CHUNKS):
        t = beg + z * BT + i
        kk = tl.load(
            K + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :],
            t[:, None] < end,
            0.0,
        ).to(tl.float16)
        qq = tl.load(
            Q + (t[:, None] * H + h // (HV // H)) * 128 + k[None, :],
            t[:, None] < end,
            0.0,
        ).to(tl.float16)
        value = tl.load(
            V + (t[:, None] * HV + h) * 128 + vv[None, :], t[:, None] < end, 0.0
        ).to(tl.float32)
        if FUSED:
            g = tl.load(G + t * HV + h, t < end, 0.0).to(tl.float32)
            b = tl.load(B + t * HV + h, t < end, 0.0).to(tl.float32)
            pref = tl.cumsum(g, 0)
            tri = i[:, None] >= i[None, :]
            d = tl.exp(tl.where(tri, pref[:, None] - pref[None, :], 0.0))
            ka = tle.gpu.alloc(
                (BT, 128), tl.float16, init_value=kk, nv_mma_shared_layout=True
            )
            kb = tle.gpu.alloc(
                (BT, 128), tl.float16, init_value=kk, nv_mma_shared_layout=True
            )
            qa = tle.gpu.alloc(
                (BT, 128), tl.float16, init_value=qq, nv_mma_shared_layout=True
            )
            gram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(ka, kb, trans_b=True))
            qgram = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(qa, kb, trans_b=True))
            L = tl.where(i[:, None] > i[None, :], -gram * d * b[:, None], 0.0)
            qk = tl.where(tri, qgram * d, 0.0)
            eye = (i[:, None] == i[None, :]).to(tl.float32)
            # Use nilpotence: (I-L)^-1 is the product of I+L^(2^j).
            P = L.to(tl.float16)
            pow2 = tl.dot(P, P).to(tl.float16)
            A = eye + L + tl.dot((eye + L).to(tl.float16), pow2)
            pow4 = tl.dot(pow2, pow2).to(tl.float16)
            A += tl.dot(A.to(tl.float16), pow4)
            rn = tl.max(tl.sum(tl.abs(pow4).to(tl.float32), 1), 0)
            # Large triangular powers can cancel catastrophically even when the
            # finite inverse exists. A NaN sentinel preserves state for FP32 replay.
            if ~(rn <= 0.0009):
                A = tl.full((BT, BT), float("nan"), tl.float32)
            xf = A * b[None, :]
            c = xf
            qmat = qk
            pre = tl.exp(pref)
            rev = tl.exp(tl.sum(g, 0) - pref)
        else:
            f = F + ((n * CHUNKS + z) * HV + h) * (2 * BT * BT + 2 * BT)
            pre = tl.load(f + 2 * BT * BT + i)
            rev = tl.load(f + 2 * BT * BT + BT + i)
        if EXTENDED:
            if z == 0 and initial_mag == 0.0:
                rhs = value
            else:
                rhs = value - pre[:, None] * _split_dot(kk, st)
        else:
            rhs = value - pre[:, None] * tl.dot(kk, st.to(tl.float16))
        if not FUSED:
            c = tl.load(f + i[:, None] * BT + i[None, :])
        x = tl.dot(c.to(tl.float16), rhs.to(tl.float16))
        if not FUSED:
            qmat = tl.load(f + BT * BT + i[:, None] * BT + i[None, :])
        if EXTENDED:
            o = tl.dot(qmat.to(tl.float16), x.to(tl.float16))
            if z != 0 or initial_mag != 0.0:
                o += pre[:, None] * tl.dot(qq, st.to(tl.float16))
        else:
            o = pre[:, None] * tl.dot(qq, st.to(tl.float16)) + tl.dot(
                qmat.to(tl.float16), x.to(tl.float16)
            )
        tl.store(
            O + (t[:, None] * HV + h) * 128 + vv[None, :], o * Scale, t[:, None] < end
        )
        w = x * (64.0 * rev)[:, None]
        st = st * tl.sum(tl.where(i == BT - 1, pre, 0.0), 0) + _split_dot(
            tl.trans(kk), w
        ) * (1.0 / 64.0)
    m = tl.max(tl.abs(st), 0)
    check_t = beg + tl.arange(0, triton.next_power_of_2(LENGTH))
    check_g = tl.load(G + check_t * HV + h, check_t < end, 0.0)
    check_b = tl.load(B + check_t * HV + h, check_t < end, 0.0)
    valid_g = (
        tl.min(
            (
                (check_g >= -80.0)
                & (check_g <= 0.0)
                & (check_b >= 0.0)
                & (check_b <= 1.0)
            ).to(tl.int32),
            0,
        )
        != 0
    )
    if EXTENDED:
        valid_s = initial_mag <= 4.0
    else:
        valid_s = tl.max(tl.abs(tl.load(co)), 0) <= 4.0 / 448.0
    finite = tl.min(((m == m) & (m < float("inf"))).to(tl.int32), 0) != 0
    # Commit a complete V tile only after its range checks pass. Rejected
    # tiles keep their original input state for the subsequent FP32 replay.
    bad = ~valid_g | ~valid_s | ~finite
    tl.store(Retry + (n * HV + h) * (128 // BV) + tl.program_id(2), bad)
    if bad:
        return
    safe = tl.where(m > 0.0, m, 448.0)
    packed_output: tl.constexpr = (
        (T > 64 or FUSED)
        and FM[3] == 1
        and FM[0] % 4 == 0
        and FM[1] % 4 == 0
        and FM[2] % 4 == 0
    )
    if packed_output:
        store_fp8_values4(
            Final, st * (448.0 / safe)[None, :], sidx, h, k, v4, FM[0], FM[1], FM[2], BV
        )
    else:
        tl.store(
            Final + sidx * FM[0] + h * FM[1] + k[:, None] * FM[2] + vv[None, :] * FM[3],
            (st * (448.0 / safe)[None, :]).to(S.dtype.element_ty),
        )
    tl.store(
        FinalScale + sidx * FM[4] + h * FM[5] + vv * FM[6],
        tl.where((m == m) & (m < float("inf")), safe / 448.0, float("nan")),
    )


def chunk_prefill(
    q,
    k,
    v,
    g,
    beta,
    scale,
    state,
    sc,
    cu,
    idx,
    out,
    final_state=None,
    final_scale=None,
    copy_empty=False,
    value_bounds=None,
    sequence_bound=None,
):
    batch, tokens, heads, _ = q.shape
    hv = v.shape[2]
    ns = batch if cu is None else cu.numel() - 1
    bounded = sequence_bound is not None
    if bounded or (ns == 1 and 32 < tokens <= 128):
        real_final = state if final_state is None else final_state
        real_scale = sc if final_scale is None else final_scale
        final_state, final_scale = real_final, real_scale
        bt = 64
        length = sequence_bound if bounded else tokens
        fused = not bounded and tokens <= 64
        bv = (32 if ns * hv < 64 else 64) if bounded else (128 if fused else 64)
        nc = triton.cdiv(length, bt)
        retry = torch.empty((ns, hv, 128 // bv), device=q.device, dtype=torch.int32)
        coeff = (
            None
            if fused
            else torch.empty(
                (ns, nc, hv, 2 * bt * bt + 2 * bt), device=q.device, dtype=torch.float32
            )
        )
        fm = (*final_state.stride(), *final_scale.stride())
        if not fused:
            _short_factors[(nc, hv, ns)](
                q,
                k,
                g,
                beta,
                coeff,
                cu,
                cu is not None,
                heads,
                hv,
                tokens,
                bt,
                nc,
                length,
                bounded,
                num_warps=(8 if ns * hv < 64 else 16) if bounded else 16,
                num_stages=1,
                enable_backend_opt=True,
            )
        _short_apply[(ns, hv, 128 // bv)](
            q,
            k,
            v,
            coeff,
            g,
            beta,
            fused,
            state,
            sc,
            idx,
            cu,
            cu is not None,
            out,
            final_state,
            final_scale,
            fm,
            idx is not None,
            heads,
            hv,
            tokens,
            bt,
            bv,
            state.stride(),
            sc.stride(),
            scale,
            copy_empty,
            retry,
            nc,
            length,
            idx.stride(0) if idx is not None else 1,
            bounded,
            num_warps=16 if fused else 8,
            num_stages=1,
            enable_backend_opt=True,
            extern_libs={"": vector_library()},
        )
        rfm = (*real_final.stride(), *real_scale.stride())
        _finish[(ns, hv, 16)](
            k,
            v,
            g,
            beta,
            state,
            sc,
            cu,
            idx,
            q,
            out,
            retry,
            retry,
            real_final,
            real_scale,
            rfm,
            copy_empty,
            scale,
            tokens,
            heads,
            hv,
            ns,
            8,
            *state.stride(),
            *sc.stride(),
            cu is not None,
            idx is not None,
            64,
            bv,
            True,
            idx.stride(0) if idx is not None else 1,
            num_warps=4,
            num_stages=1,
        )
        return out, real_final, real_scale
    capacity = batch * triton.cdiv(tokens, 32) if cu is None else tokens // 32 + ns
    table = None
    if cu is not None and ns > 1:
        table = torch.empty((capacity, 4), device=q.device, dtype=torch.int32)
        _chunk_map[(triton.cdiv(capacity, 32),)](
            cu, table, tokens, ns, 32, capacity, num_warps=4
        )
    dirty = torch.empty((ns, hv), device=q.device, dtype=torch.int32)
    _clear_dirty[(triton.cdiv(ns * hv, 256),)](dirty, ns * hv, num_warps=2)
    query_desc = None
    if value_bounds is not None:
        query_desc = TensorDescriptor(
            q, [batch * tokens, heads * 128], [heads * 128, 1], [32, 128]
        )
    tensors = (q, k, v, g, beta, state, sc, cu, idx, out, table, dirty)
    strides = (*state.stride(), *sc.stride(), scale, cu is not None, idx is not None)
    # Most CTAs need only 32 output rows. The first 64 rows use the
    # actual initial state, so split V to expose enough parallelism.
    _window_output[(capacity, hv, 1)](
        *tensors,
        tokens,
        heads,
        hv,
        ns,
        64,
        128,
        32,
        *strides,
        False,
        query_desc,
        value_bounds,
        value_bounds is not None,
        num_warps=4,
        num_stages=1,
        enable_backend_opt=True,
    )
    prefix_v = 64
    _window_output[(ns, hv, 128 // prefix_v)](
        *tensors,
        tokens,
        heads,
        hv,
        ns,
        64,
        prefix_v,
        32,
        *strides,
        True,
        query_desc,
        value_bounds,
        value_bounds is not None,
        num_warps=8,
        num_stages=1,
        enable_backend_opt=True,
        extern_libs={"": vector_library()},
    )
    if final_state is None:
        final_state, final_scale = state, sc
    fm = (*final_state.stride(), *final_scale.stride())
    final_v = 32
    retry = torch.empty((ns, hv, 128 // final_v), device=q.device, dtype=torch.int32)
    _try_final[(ns, hv, 128 // final_v)](
        k,
        v,
        g,
        beta,
        state,
        sc,
        cu,
        idx,
        dirty,
        retry,
        final_state,
        final_scale,
        fm,
        tokens,
        heads,
        hv,
        final_v,
        state.stride(),
        sc.stride(),
        cu is not None,
        idx is not None,
        num_warps=8,
        num_stages=1,
        enable_backend_opt=True,
        extern_libs={"": vector_library()},
    )
    _finish[(ns, hv, 16)](
        k,
        v,
        g,
        beta,
        state,
        sc,
        cu,
        idx,
        q,
        out,
        dirty,
        retry,
        final_state,
        final_scale,
        fm,
        copy_empty,
        scale,
        tokens,
        heads,
        hv,
        ns,
        8,
        *state.stride(),
        *sc.stride(),
        cu is not None,
        idx is not None,
        64,
        final_v,
        num_warps=4,
        num_stages=1,
    )
    return out, final_state, final_scale


@libentry()
@triton.jit
def decode_vk_kernel(
    Q,
    Key,
    Value,
    State,
    Scales,
    Alog,
    A,
    Bias,
    B,
    Out,
    Index,
    History,
    HistScale,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    T: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    V3: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    O3: tl.constexpr,
    IX: tl.constexpr,
    HIST_T: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    INDEXED: tl.constexpr,
    SAVE_STATE: tl.constexpr,
    SAVE_HISTORY: tl.constexpr,
    Cu,
    VARLEN: tl.constexpr,
    RAW_GATES: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FinalState=None,
    FinalScale=None,
    HasInitial=None,
    PREFILL: tl.constexpr = False,
    HAS_INITIAL_MASK: tl.constexpr = False,
    HAS_INITIAL_STATE: tl.constexpr = False,
    HAS_FORGET_GATE: tl.constexpr = False,
    HAS_UPDATE_GATE: tl.constexpr = False,
    LOG_SPACE: tl.constexpr = True,
    KEY_HEADS: tl.constexpr = 1,
    VALUE_HEADS: tl.constexpr = 1,
    HAS_INITIAL_STRIDE: tl.constexpr = 0,
    FS0: tl.constexpr = 0,
    FS1: tl.constexpr = 0,
    FS2: tl.constexpr = 0,
    FS3: tl.constexpr = 0,
    FC0: tl.constexpr = 0,
    FC1: tl.constexpr = 0,
    FC2: tl.constexpr = 0,
    TLE_PREFILL: tl.constexpr = False,
):
    sequence_head = tl.program_id(0)
    sequence, head = sequence_head // HV, sequence_head % HV
    query_head = head // (HV // H)
    if VARLEN:
        batch = 0
        begin = tl.load(Cu + sequence)
        length = tl.load(Cu + sequence + 1) - begin
    else:
        batch, begin, length = sequence, 0, T
    if not PREFILL and length <= 0:
        return
    if TLE_PREFILL:
        # One warp reduces a complete 128-key row without exchanging partial sums.
        layout: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [1, 32], [4, 1], [1, 0])
        row = tle.gpu.set_layout(tl.arange(0, BV), tle.gpu.SlicedEncoding(1, layout))
        reduction = tle.gpu.set_layout(
            tl.arange(0, BK), tle.gpu.SlicedEncoding(0, layout)
        )
    else:
        row = tl.arange(0, BV)
        reduction = tl.arange(0, BK)
    columns = tl.program_id(1) * BV + row
    slot = (
        tl.load(Index + sequence * IX).to(tl.int64)
        if INDEXED
        else sequence.to(tl.int64)
    )
    valid_v = columns < DV
    mask = valid_v[:, None] & (reduction[None, :] < DK)
    if slot < 0:
        for step in range(length):
            token = begin + step
            tl.store(
                Out + batch * O0 + token * O1 + head * O2 + columns * O3,
                0.0,
                valid_v,
            )
        return
    state_offset = (
        slot * S0 + head * S1 + columns[:, None] * S2 + reduction[None, :] * S3
    )
    scale_offset = slot * C0 + head * C1 + columns * C2
    if PREFILL:
        key_head = head // (HV // KEY_HEADS)
        value_head = head // (HV // VALUE_HEADS)
        state = tl.full((BV, BK), 0.0, tl.float32)
        if HAS_INITIAL_STATE:
            read_initial = (
                tl.load(HasInitial + sequence * HAS_INITIAL_STRIDE)
                if HAS_INITIAL_MASK
                else True
            )
            state = tl.load(State + state_offset, mask & read_initial, 0.0).to(
                tl.float32
            )
            state *= tl.load(Scales + scale_offset, valid_v & read_initial, 0.0)[
                :, None
            ]
    else:
        key_head, value_head = query_head, head
        state = tl.load(State + state_offset, mask, 0.0).to(tl.float32)
        state *= tl.load(Scales + scale_offset, valid_v, 0.0)[:, None]
    if RAW_GATES:
        alpha = tl.exp2(tl.load(Alog + head).to(tl.float32) * 1.4426950408889634)
        bias = tl.load(Bias + head).to(tl.float32)
    for step in range(length):
        token = begin + step
        query = tl.load(
            Q + batch * Q0 + token * Q1 + query_head * Q2 + reduction * Q3,
            reduction < DK,
            0.0,
        ).to(tl.float32)
        key = tl.load(
            Key + batch * K0 + token * K1 + key_head * K2 + reduction * K3,
            reduction < DK,
            0.0,
        ).to(tl.float32)
        if NORMALIZE:
            # The stock MP31 rsqrt can return NaN for +inf. Finite BF16
            # components with an overflowing squared norm must normalize to zero.
            qnorm = tl.sum(query * query, 0) + 1.0e-6
            knorm = tl.sum(key * key, 0) + 1.0e-6
            query = tl.where(qnorm == float("inf"), 0.0, query * tl.rsqrt(qnorm))
            key = tl.where(knorm == float("inf"), 0.0, key * tl.rsqrt(knorm))
        query *= SCALE
        if RAW_GATES:
            gate = (
                tl.load(A + batch * A0 + token * A1 + head * A2).to(tl.float32) + bias
            )
            softplus = tl.where(
                gate > 20.0,
                gate,
                tl.log2(1.0 + tl.exp2(gate * 1.4426950408889634)) * 0.6931471805599453,
            )
            decay = tl.exp2(-alpha * softplus * 1.4426950408889634)
            beta = tl.sigmoid(
                tl.load(B + batch * B0 + token * B1 + head * B2).to(tl.float32)
            )
        else:
            if PREFILL:
                decay = 1.0
                beta = 1.0
                if HAS_FORGET_GATE:
                    gate = tl.load(A + batch * A0 + token * A1 + head * A2).to(
                        tl.float32
                    )
                    decay = tl.exp2(gate * 1.4426950408889634) if LOG_SPACE else gate
                if HAS_UPDATE_GATE:
                    beta = tl.load(B + batch * B0 + token * B1 + head * B2).to(
                        tl.float32
                    )
            else:
                decay = tl.exp2(
                    tl.load(A + batch * A0 + token * A1 + head * A2).to(tl.float32)
                    * 1.4426950408889634
                )
                beta = tl.load(B + batch * B0 + token * B1 + head * B2).to(tl.float32)
        state *= decay
        value = tl.load(
            Value + batch * V0 + token * V1 + value_head * V2 + columns * V3,
            valid_v,
            0.0,
        ).to(tl.float32)
        delta = (value - tl.sum(state * key[None, :], 1)) * beta
        state = tl.fma(delta[:, None], key[None, :], state)
        result = tl.sum(state * query[None, :], 1)
        tl.store(
            Out + batch * O0 + token * O1 + head * O2 + columns * O3, result, valid_v
        )
        if SAVE_HISTORY:
            history_group = (sequence * HIST_T + step) * HV + head
            history_offset = (history_group * DV + columns[:, None]) * DK + reduction[
                None, :
            ]
            store_scaled_state(
                state,
                History,
                HistScale,
                history_offset,
                history_group * DV + columns,
                mask,
                valid_v,
            )
    if SAVE_STATE:
        if PREFILL:
            final_offset = (
                slot * FS0
                + head * FS1
                + columns[:, None] * FS2
                + reduction[None, :] * FS3
            )
            final_scale_offset = slot * FC0 + head * FC1 + columns * FC2
            store_scaled_state(
                state,
                FinalState,
                FinalScale,
                final_offset,
                final_scale_offset,
                mask,
                valid_v,
            )
        else:
            store_scaled_state(
                state, State, Scales, state_offset, scale_offset, mask, valid_v
            )


def gated_delta_rule_decode_w8a16_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    state_layout: str = "VK",
    state_indices: torch.Tensor | None = None,
    scale: float | None = None,
    output: torch.Tensor | None = None,
    intermediate_states_buffer: torch.Tensor | None = None,
    disable_state_update: bool = False,
    use_qk_l2norm: bool = True,
    *,
    state_scale: torch.Tensor,
    intermediate_state_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MATE decode/MTP contract with FP8 VK state and explicit channel scales."""
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("expected Q/K[B,T,H,K] and V[B,T,HV,V]")
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if (
        min(heads, key_dim, value_heads, value_dim) < 1
        or value_heads % heads
        or v.shape[:2] != q.shape[:2]
    ):
        raise ValueError("invalid QKV dimensions")
    if state_layout != "VK":
        raise NotImplementedError("MThreads native GDN supports VK state only")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("QKV must share BF16 or FP16 dtype")
    if state.ndim != 4 or state.shape[1:] != (value_heads, value_dim, key_dim):
        raise ValueError("state must have shape [pool,HV,V,K]")
    if state.dtype != torch.float8_e4m3fn or state_scale.dtype != torch.float32:
        raise TypeError("expected E4M3 state and FP32 scales")
    if state_scale.shape != state.shape[:3]:
        raise ValueError("state_scale must have shape [pool,HV,V]")
    if (
        a.shape != (batch, tokens, value_heads)
        or b.shape != a.shape
        or a.dtype != q.dtype
        or b.dtype != q.dtype
    ):
        raise ValueError("a/b must have shape [B,T,HV] and match activations")
    if (
        A_log.shape != (value_heads,)
        or dt_bias.shape != A_log.shape
        or A_log.dtype != torch.float32
        or dt_bias.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError("invalid A_log or dt_bias")
    if state_indices is not None:
        if state_indices.shape != (batch,) or state_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("state_indices must be int32/int64 [B]")
    elif state.shape[0] != batch:
        raise ValueError("identity mapping requires one state per sequence")
    if scale is not None and type(scale) not in (float, int):
        raise TypeError("scale must be a Python number")
    if type(use_qk_l2norm) is not bool or type(disable_state_update) is not bool:
        raise TypeError("controls must be Python booleans")
    if tokens == 1 and (disable_state_update or intermediate_states_buffer is not None):
        raise NotImplementedError(
            "MATE T=1 does not support rollback or read-only state"
        )
    if tokens > 1 and (key_dim != 128 or value_dim != 128):
        raise NotImplementedError("MATE MTP supports K=V=128")
    history = intermediate_states_buffer
    tensors = [q, k, v, state, state_scale, A_log, a, dt_bias, b]
    if state_indices is not None:
        tensors.append(state_indices)
    if history is not None:
        if (
            history.ndim != 5
            or history.shape[0] < batch
            or history.shape[1] < tokens
            or history.shape[2:] != state.shape[1:]
            or history.dtype != state.dtype
            or not history.is_contiguous()
        ):
            raise ValueError(
                "rollback state must be contiguous FP8 [buffer,steps,HV,V,K]"
            )
        if (
            intermediate_state_scale is None
            or intermediate_state_scale.shape != history.shape[:4]
            or intermediate_state_scale.dtype != torch.float32
            or not intermediate_state_scale.is_contiguous()
        ):
            raise ValueError(
                "rollback FP8 state requires FP32 [buffer,steps,HV,V] scales"
            )
        tensors.extend([history, intermediate_state_scale])
    elif intermediate_state_scale is not None:
        raise ValueError("rollback scales require a rollback state buffer")
    if output is not None:
        if output.shape != (
            batch,
            tokens,
            value_heads,
            value_dim,
        ) or output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("output must be [B,T,HV,V] FP16/BF16/FP32")
        tensors.append(output)
    if q.device.type != "musa" or any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on one MUSA device")
    if any(t.requires_grad for t in tensors):
        raise NotImplementedError("GDN is forward-only")
    if any(stride < 0 for t in tensors for stride in t.stride()):
        raise ValueError("negative strides are unsupported")
    if output is None:
        output = torch.empty(
            (
                (batch, value_heads, value_dim)
                if tokens == 1
                else (batch, tokens, value_heads, value_dim)
            ),
            device=q.device,
            dtype=q.dtype,
        )
    out_view = output.unsqueeze(1) if output.ndim == 3 else output
    if not batch or not tokens:
        return output, state
    single = HAS_TLE and tokens == 1 and key_dim == value_dim == 128 and use_qk_l2norm
    packed = (
        single
        and torch.musa.get_device_capability(q.device) == (3, 1)
        and state.is_contiguous()
        and state.data_ptr() % 16 == 0
        and all(
            t.stride(3) == 1
            and t.data_ptr() % 8 == 0
            and t.stride(0) % 4 == 0
            and t.stride(2) % 4 == 0
            for t in (q, k)
        )
    )
    use_tme = (
        packed
        and os.environ.get("FLAGGEMS_MTHREADS_GDN_TME") == "1"
        and state.numel() // 128 < 2**31
    )
    if single and (packed or batch <= 3):
        value_tile = 8 if batch == 1 else 16
        args = (
            q,
            k,
            v,
            state,
            state_scale,
            A_log,
            a,
            dt_bias,
            b,
            out_view,
            state_indices,
            (
                heads,
                value_heads,
                q.stride(0),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(2),
                v.stride(3),
                a.stride(0),
                a.stride(2),
                b.stride(0),
                b.stride(2),
                *state.stride(),
                *state_scale.stride(),
                out_view.stride(0),
                out_view.stride(2),
                out_view.stride(3),
                0 if state_indices is None else state_indices.stride(0),
                key_dim**-0.5 if scale is None else float(scale),
                state_indices is not None,
                value_tile,
            ),
        )
        grid = (batch * value_heads * (128 // value_tile),)
        if packed:
            desc = None
            if use_tme:
                desc = TensorDescriptor(
                    state.view(torch.float32),
                    [state.numel() // 128, 32],
                    [32, 1],
                    [value_tile, 32],
                )
            interleave_heads = batch <= 2 or 4 <= batch <= 7
            decode_quart[grid](
                *args[:-1],
                desc,
                args[-1],
                use_tme,
                16 if interleave_heads else 8,
                interleave_heads,
                num_warps=4,
                num_stages=1,
                enable_backend_opt=True,
                extern_libs={"gdn_fp8_quart": vector_library()},
            )
        else:
            decode_single_tle[grid](*args, num_warps=4, num_stages=1)
        return output, state
    # Live recurrence cannot be replayed for runtime autotuning.
    value_tile, warps = (8 if batch == 1 else (16 if batch <= 3 else 32)), 4
    decode_vk_kernel[(batch * value_heads, triton.cdiv(value_dim, value_tile))](
        q,
        k,
        v,
        state,
        state_scale,
        A_log,
        a,
        dt_bias,
        b,
        out_view,
        state_indices,
        history,
        intermediate_state_scale,
        heads,
        value_heads,
        key_dim,
        value_dim,
        tokens,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *a.stride(),
        *b.stride(),
        *state.stride(),
        *state_scale.stride(),
        *out_view.stride(),
        0 if state_indices is None else state_indices.stride(0),
        0 if history is None else history.shape[1],
        key_dim**-0.5 if scale is None else float(scale),
        use_qk_l2norm,
        state_indices is not None,
        not disable_state_update,
        history is not None,
        None,
        False,
        True,
        triton.next_power_of_2(key_dim),
        value_tile,
        num_warps=warps,
        num_stages=1,
    )
    return output, state


@triton.jit
def normalize_query_key_kernel(
    Q,
    Key,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
    NORMALIZE_KEY: tl.constexpr = True,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    reduction = tl.arange(0, BK)
    head = rows % H
    token = (rows // H) % TOKENS
    batch = rows // (TOKENS * H)
    qo = (
        batch[:, None] * Q0
        + token[:, None] * Q1
        + head[:, None] * Q2
        + reduction[None, :] * Q3
    )
    ko = (
        batch[:, None] * K0
        + token[:, None] * K1
        + head[:, None] * K2
        + reduction[None, :] * K3
    )
    mask = (rows[:, None] < ROWS) & (reduction[None, :] < DK)
    query = tl.load(Q + qo, mask, 0.0).to(tl.float32)
    if NORMALIZE_KEY:
        key = tl.load(Key + ko, mask, 0.0).to(tl.float32)
    query *= tl.rsqrt(tl.sum(query * query, 1) + 1.0e-6)[:, None]
    if NORMALIZE_KEY:
        key *= tl.rsqrt(tl.sum(key * key, 1) + 1.0e-6)[:, None]
    tl.store(Q + qo, query, mask)
    if NORMALIZE_KEY:
        tl.store(Key + ko, key, mask)


def _can_chunk_prefill(q, k, v, g, beta, cu, indices, normalize, scale, bounded=False):
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    sequences = batch if cu is None else cu.numel() - 1
    # Short calls avoid multiple launches. The matrix path assumes unit-length K,
    # a contiguous activation stream and bounded 32-bit activation offsets.
    return (
        HAS_TLE
        and normalize
        and key_dim == value_dim == 128
        and k.shape == q.shape
        and value_heads >= heads
        and value_heads % heads == 0
        and (bounded or tokens >= 128 or (sequences == 1 and tokens > 32))
        and 0 < sequences <= 128
        and abs(scale) <= 0.1
        and (cu is None or (batch == 1 and cu.is_contiguous()))
        and (indices is None or indices.is_contiguous() or bounded)
        and all(x.is_contiguous() for x in (q, k, v))
        and g is not None
        and beta is not None
        and g.is_contiguous()
        and beta.is_contiguous()
        and q.dtype in (torch.bfloat16, torch.float16)
        and batch * tokens * value_heads * 128 < 2**31
        and torch.musa.get_device_capability(q.device) == (3, 1)
    )


@triton.jit
def prepare_raw_prefill(
    Q,
    K,
    V,
    Bounds,
    Alog,
    A,
    B,
    Bias,
    G,
    Beta,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BR: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BOUND: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, 128)
    off = row[:, None] * 128 + col[None, :]
    if NORMALIZE:
        q = tl.load(Q + off, row[:, None] < T * H, 0.0).to(tl.float32)
        k = tl.load(K + off, row[:, None] < T * H, 0.0).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, 1) + 1e-6)[:, None]
        k *= tl.rsqrt(tl.sum(k * k, 1) + 1e-6)[:, None]
        tl.store(Q + off, q, row[:, None] < T * H)
        tl.store(K + off, k, row[:, None] < T * H)
    if BOUND:
        vr = tl.program_id(0) * BR * (HV // H) + tl.arange(
            0, triton.next_power_of_2(BR * (HV // H))
        )
        value = tl.load(
            V + vr[:, None] * 128 + col[None, :],
            (vr[:, None] < T * HV)
            & (
                tl.arange(0, triton.next_power_of_2(BR * (HV // H)))[:, None]
                < BR * (HV // H)
            ),
            0.0,
        ).to(tl.float32)
        tl.store(
            Bounds + vr,
            tl.max(tl.abs(value), 1),
            (vr < T * HV)
            & (tl.arange(0, triton.next_power_of_2(BR * (HV // H))) < BR * (HV // H)),
        )
    group = tl.arange(0, triton.next_power_of_2(HV // H))
    head = row[:, None] % H * (HV // H) + group[None, :]
    go = row[:, None] // H * HV + head
    alpha = tl.load(Alog + head, group[None, :] < HV // H, 0.0).to(tl.float32)
    bias = tl.load(Bias + head, group[None, :] < HV // H, 0.0).to(tl.float32)
    aa = tl.load(A + go, (row[:, None] < T * H) & (group[None, :] < HV // H), 0.0).to(
        tl.float32
    )
    bb = tl.load(B + go, (row[:, None] < T * H) & (group[None, :] < HV // H), 0.0).to(
        tl.float32
    )
    x = aa + bias
    sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    tl.store(
        G + go, -tl.exp(alpha) * sp, (row[:, None] < T * H) & (group[None, :] < HV // H)
    )
    tl.store(
        Beta + go, tl.sigmoid(bb), (row[:, None] < T * H) & (group[None, :] < HV // H)
    )


def raw_prefill_w8a16_fp8(
    q, k, v, scale, state, sc, cu, indices, normalize, A_log, a, bias, b
):
    if any(tensor is None for tensor in (a, bias, b)):
        raise ValueError("raw GDN gating requires A_log, a, dt_bias, and b")
    batch, tokens, heads, key_dim = q.shape
    hv, value_dim = v.shape[-2:]
    if (
        key_dim != 128
        or value_dim != 128
        or k.shape != q.shape
        or v.shape[:2] != q.shape[:2]
    ):
        raise NotImplementedError(
            "raw MThreads GDN prefill requires K=V=128 and matching QKV"
        )
    if heads <= 0 or hv < heads or hv % heads:
        raise ValueError("GDN value heads must be a positive multiple of query heads")
    if (
        a.shape != (batch, tokens, hv)
        or b.shape != a.shape
        or A_log.shape != (hv,)
        or bias.shape != (hv,)
    ):
        raise ValueError("raw GDN gate shapes must match [B,T,HV] and [HV]")
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise NotImplementedError(
            "raw MThreads GDN prefill requires matching FP16/BF16 QKV"
        )
    if not all(x.is_contiguous() for x in (q, k, v, A_log, a, bias, b)):
        raise NotImplementedError(
            "raw MThreads GDN prefill requires contiguous activations and gates"
        )
    if not all(x.device == q.device for x in (k, v, A_log, a, bias, b, state, sc)):
        raise ValueError("GDN inputs must be on the same device")
    if (
        state.ndim != 4
        or state.shape[1:] != (hv, 128, 128)
        or sc.shape != (state.shape[0], hv, 128)
    ):
        raise ValueError(
            "GDN state and scales must have shapes [S,HV,K,V] and [S,HV,V]"
        )
    if state.dtype != torch.float8_e4m3fn or sc.dtype != torch.float32:
        raise NotImplementedError("raw GDN prefill requires E4M3 state and FP32 scales")
    if any(
        t.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        for t in (A_log, a, bias, b)
    ):
        raise TypeError("raw GDN gates must be floating-point tensors")
    for metadata in (cu, indices):
        if metadata is not None and (
            metadata.device != q.device
            or metadata.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "GDN sequence metadata must contain device integer indices"
            )
    ns = batch if cu is None else cu.numel() - 1
    if cu is not None and (batch != 1 or cu.ndim != 1 or not cu.is_contiguous()):
        raise NotImplementedError(
            "packed GDN prefill requires batch one and contiguous boundaries"
        )
    if indices is not None and (indices.shape != (ns,) or not indices.is_contiguous()):
        raise ValueError("prefill state indices must have shape [num_sequences]")
    g = torch.empty((batch, tokens, hv), device=q.device, dtype=torch.float32)
    beta = torch.empty_like(g)
    use_chunk = _can_chunk_prefill(q, k, v, g, beta, cu, indices, normalize, scale)
    bounds = (
        torch.empty((batch * tokens * hv,), device=q.device, dtype=torch.float32)
        if use_chunk and tokens >= 128
        else None
    )
    if tokens:
        prepare_raw_prefill[(triton.cdiv(batch * tokens * heads, 16),)](
            q,
            k,
            v,
            bounds,
            A_log,
            a,
            b,
            bias,
            g,
            beta,
            batch * tokens,
            heads,
            hv,
            16,
            normalize,
            bounds is not None,
            num_warps=4,
            num_stages=1,
        )
    if use_chunk:

        return chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale,
            state,
            sc,
            cu,
            indices,
            torch.empty_like(v),
            value_bounds=bounds,
        )
    return recurrent_prefill_w8a16_fp8(
        q, k, v, g, beta, scale, state, sc, cu, indices, False
    )


def recurrent_prefill_w8a16_fp8(
    q,
    k,
    v,
    g,
    beta,
    scale,
    state_fp8,
    state_scale,
    cu_seqlens,
    state_indices,
    normalize,
    max_sequence_length=None,
):
    """FP32 recurrence without intermediate activation rounding."""
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    sequence_count = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    state_vk = state_fp8.transpose(-1, -2)
    output = torch.empty(v.shape, device=q.device, dtype=q.dtype)
    if not sequence_count or not tokens:
        return output, state_fp8, state_scale
    sequence_bound = tokens if max_sequence_length is None else max_sequence_length
    bounded = (
        32 < sequence_bound <= 256
        and (sequence_count > 1 or sequence_bound > 128)
        and _can_chunk_prefill(
            q, k, v, g, beta, cu_seqlens, state_indices, normalize, scale, True
        )
    )
    norm_rows = 8 if bounded and batch * tokens * heads <= 8192 else 16
    if normalize:
        normalize_query_key_kernel[(triton.cdiv(batch * tokens * heads, norm_rows),)](
            q,
            k,
            batch * tokens * heads,
            tokens,
            heads,
            key_dim,
            *q.stride(),
            *k.stride(),
            triton.next_power_of_2(key_dim),
            norm_rows,
            num_warps=4,
            num_stages=1,
        )
    if bounded:
        return chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale,
            state_fp8,
            state_scale,
            cu_seqlens,
            state_indices,
            output,
            sequence_bound=sequence_bound,
        )
    if _can_chunk_prefill(
        q, k, v, g, beta, cu_seqlens, state_indices, normalize, scale
    ):

        return chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale,
            state_fp8,
            state_scale,
            cu_seqlens,
            state_indices,
            output,
        )
    tle_prefill = HAS_TLE and key_dim == 128
    value_tile = 8 if tle_prefill else 16
    decode_vk_kernel[
        (sequence_count * value_heads, triton.cdiv(value_dim, value_tile))
    ](
        q,
        k,
        v,
        state_vk,
        state_scale,
        None,
        g,
        None,
        beta,
        output,
        state_indices,
        None,
        None,
        heads,
        value_heads,
        key_dim,
        value_dim,
        tokens,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *g.stride(),
        *beta.stride(),
        *state_vk.stride(),
        *state_scale.stride(),
        *output.stride(),
        0 if state_indices is None else state_indices.stride(0),
        0,
        scale,
        False,
        state_indices is not None,
        True,
        False,
        cu_seqlens,
        cu_seqlens is not None,
        False,
        triton.next_power_of_2(key_dim),
        value_tile,
        TLE_PREFILL=tle_prefill,
        num_warps=4,
        num_stages=1,
    )
    return output, state_fp8, state_scale


def chunk_gated_delta_rule_w8a16_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    is_log_space: bool = True,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    *,
    initial_state_scale: torch.Tensor | None = None,
    output_state_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """MATE MP31 prefill with FP8 VK state; return output, final state and scales.

    Initial state is read-only. Supplied output state/scales are returned by
    identity; both final values are None when output_final_state is false.
    Indexed active slots must be in bounds and unique, as required by MATE.
    """
    if scale is not None and type(scale) not in (float, int):
        raise TypeError("scale must be a Python number")
    if chunk_size != 64 or checkpoint_every_n_tokens:
        raise NotImplementedError(
            "MP31 prefill requires chunk_size=64 and no checkpoints"
        )
    unbatched = q.ndim == 3
    if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("QKV must all be rank 3 or rank 4")
    if unbatched:
        if cu_seqlens is None:
            raise ValueError("rank-3 QKV requires cu_seqlens")
        query, key, value = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    else:
        query, key, value = q, k, v
    batch, tokens, heads, key_dim = query.shape
    key_heads, value_heads, value_dim = key.shape[2], value.shape[2], value.shape[3]
    state_heads = max(heads, value_heads)
    if (
        min(heads, key_heads, value_heads, key_dim, value_dim) < 1
        or key.shape[:2] != query.shape[:2]
        or value.shape[:2] != query.shape[:2]
        or key.shape[-1] != key_dim
        or not (
            (heads == key_heads and value_heads % heads == 0)
            or (value_heads == key_heads and heads % key_heads == 0)
        )
    ):
        raise ValueError("invalid QKV dimensions or GQA/GVA head groups")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("QKV must share BF16 or FP16 dtype")
    tensors = [q, k, v]
    if cu_seqlens is not None:
        if (
            batch != 1
            or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("cu_seqlens requires packed B=1 and an integer vector")
        # The vendor kernel addresses cumulative lengths contiguously.
        if not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous")
        sequences = cu_seqlens.numel() - 1
        tensors.append(cu_seqlens)
    else:
        sequences = batch
    if state_indices is not None:
        if state_indices.shape != (sequences,) or state_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("state_indices must be int32/int64 [N]")
        tensors.append(state_indices)
    if has_initial_state is not None:
        if (
            initial_state is None
            or has_initial_state.shape != (sequences,)
            or has_initial_state.dtype != torch.bool
        ):
            raise ValueError("has_initial_state requires an initial state and bool [N]")
        tensors.append(has_initial_state)
    for state, scales in (
        (initial_state, initial_state_scale),
        (output_state, output_state_scale),
    ):
        if state is None:
            if scales is not None:
                raise ValueError("state scales require the corresponding state")
        else:
            if (
                state.ndim != 4
                or state.shape[1:] != (state_heads, value_dim, key_dim)
                or state.dtype != torch.float8_e4m3fn
            ):
                raise ValueError("state must be FP8 E4M3 [pool,H,V,K]")
            if state_indices is None and state.shape[0] != sequences:
                raise ValueError("sequence-order state requires N slots")
            if (
                scales is None
                or scales.shape != state.shape[:3]
                or scales.dtype != torch.float32
            ):
                raise ValueError("FP8 state requires FP32 [pool,H,V] scales")
            tensors.extend((state, scales))
    gate_shape = (*q.shape[:-2], state_heads)
    for gate in (g, beta):
        if gate is not None:
            if gate.shape != gate_shape or gate.dtype not in (q.dtype, torch.float32):
                raise ValueError("g/beta must match token dimensions and state heads")
            tensors.append(gate)
    output_shape = (*q.shape[:-2], state_heads, value_dim)
    if output is not None:
        if output.shape != output_shape or output.dtype != q.dtype:
            raise ValueError("output must match Q dtype and [B,T,H,V] or [T,H,V]")
        tensors.append(output)
    if q.device.type != "musa" or any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all tensors must be on one MUSA device")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("GDN is forward-only")
    if output_final_state and state_indices is not None and output_state is None:
        raise ValueError(
            "indexed final state requires an explicit output pool and scales"
        )
    if output is None:
        output = torch.empty(output_shape, device=q.device, dtype=q.dtype)
    if output_final_state:
        if output_state is None:
            output_state = torch.empty(
                (sequences, state_heads, value_dim, key_dim),
                device=q.device,
                dtype=torch.float8_e4m3fn,
            )
            output_state_scale = torch.empty(
                (sequences, state_heads, value_dim),
                device=q.device,
                dtype=torch.float32,
            )
        final_state, final_scale = output_state, output_state_scale
    else:
        final_state, final_scale = None, None
    if sequences:
        if use_qk_l2norm_in_kernel and tokens:
            for tensor in (query, key):
                rows = batch * tokens * tensor.shape[2]
                normalize_query_key_kernel[(triton.cdiv(rows, 16),)](
                    tensor,
                    None,
                    rows,
                    tokens,
                    tensor.shape[2],
                    key_dim,
                    *tensor.stride(),
                    0,
                    0,
                    0,
                    0,
                    triton.next_power_of_2(key_dim),
                    16,
                    False,
                    num_warps=4,
                    num_stages=1,
                )
        gate = None if g is None else (g.unsqueeze(0) if unbatched else g)
        update = None if beta is None else (beta.unsqueeze(0) if unbatched else beta)
        out = output.unsqueeze(0) if unbatched else output
        # Empty sequences only initialize final state; no K reduction is needed.
        attention_scale = key_dim**-0.5 if scale is None else float(scale)
        if (
            initial_state is not None
            and has_initial_state is None
            and output_final_state
            and is_log_space
            and out.is_contiguous()
            and _can_chunk_prefill(
                query,
                key,
                value,
                gate,
                update,
                cu_seqlens,
                state_indices,
                use_qk_l2norm_in_kernel,
                attention_scale,
            )
        ):

            chunk_prefill(
                query,
                key,
                value,
                gate,
                update,
                attention_scale,
                initial_state.transpose(-2, -1),
                initial_state_scale,
                cu_seqlens,
                state_indices,
                out,
                final_state.transpose(-2, -1),
                final_scale,
                copy_empty=True,
            )
            return output, final_state, final_scale
        tle_prefill = HAS_TLE and key_dim == 128 and tokens > 0
        value_tile = 8 if tle_prefill else 16
        decode_vk_kernel[(sequences * state_heads, triton.cdiv(value_dim, value_tile))](
            query,
            key,
            value,
            initial_state,
            initial_state_scale,
            None,
            gate,
            None,
            update,
            out,
            state_indices,
            None,
            None,
            heads,
            state_heads,
            key_dim,
            value_dim,
            tokens,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *((0, 0, 0) if gate is None else gate.stride()),
            *((0, 0, 0) if update is None else update.stride()),
            *((0, 0, 0, 0) if initial_state is None else initial_state.stride()),
            *(
                (0, 0, 0)
                if initial_state_scale is None
                else initial_state_scale.stride()
            ),
            *out.stride(),
            0 if state_indices is None else state_indices.stride(0),
            0,
            key_dim**-0.5 if scale is None else float(scale),
            False,
            state_indices is not None,
            output_final_state,
            False,
            cu_seqlens,
            cu_seqlens is not None,
            False,
            triton.next_power_of_2(key_dim),
            value_tile,
            final_state,
            final_scale,
            has_initial_state,
            True,
            *(
                has_initial_state is not None,
                initial_state is not None,
                g is not None,
                beta is not None,
                is_log_space,
                key_heads,
                value_heads,
                0 if has_initial_state is None else has_initial_state.stride(0),
                *((0, 0, 0, 0) if final_state is None else final_state.stride()),
                *((0, 0, 0) if final_scale is None else final_scale.stride()),
            ),
            TLE_PREFILL=tle_prefill,
            num_warps=4,
            num_stages=1,
        )
    return output, final_state, final_scale


@libentry()
@triton.jit
def _mthreads_gdn_decode_w8a16_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    raw_a,
    dt_bias,
    raw_b,
    output,
    state_fp8,
    state_scale,
    cu_seqlens,
    state_indices,
    num_accepted_tokens,
    scale,
    stride_q_b: tl.constexpr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_k: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_k: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_v: tl.constexpr,
    stride_g_b: tl.constexpr,
    stride_g_t: tl.constexpr,
    stride_g_h: tl.constexpr,
    stride_beta_b: tl.constexpr,
    stride_beta_t: tl.constexpr,
    stride_beta_h: tl.constexpr,
    stride_raw_a_b: tl.constexpr,
    stride_raw_a_t: tl.constexpr,
    stride_raw_a_h: tl.constexpr,
    stride_raw_b_b: tl.constexpr,
    stride_raw_b_t: tl.constexpr,
    stride_raw_b_h: tl.constexpr,
    USE_RAW_GATES: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_v: tl.constexpr,
    stride_state_s: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_scale_s: tl.constexpr,
    stride_scale_h: tl.constexpr,
    stride_scale_v: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_PACKED_INPUT: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_SPEC_DECODING: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    LOW_PRECISION_AMAX: tl.constexpr,
):
    value_tile = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence = sequence_head // HV
    value_head = sequence_head % HV
    query_head = value_head // (HV // H)

    if USE_PACKED_INPUT:
        batch = 0
        token = tl.load(cu_seqlens + sequence).to(tl.int64)
    else:
        batch = sequence
        token = 0

    if USE_STATE_INDICES:
        index_offset = sequence * stride_indices_seq
        initial_offset = index_offset
        if USE_SPEC_DECODING:
            accepted = tl.load(num_accepted_tokens + sequence).to(tl.int64)
            initial_offset += (accepted - 1) * stride_indices_tok
        initial_slot = tl.load(state_indices + initial_offset).to(tl.int64)
        output_slot = tl.load(state_indices + index_offset).to(tl.int64)
        if initial_slot < 0:
            return
        output_valid = output_slot >= 0
    else:
        initial_slot = sequence
        output_slot = sequence
        output_valid = True

    keys = tl.arange(0, BK)
    columns = value_tile * BV + tl.arange(0, BV)
    mask_k = keys < K
    mask_v = columns < V
    mask_state = mask_k[:, None] & mask_v[None, :]
    state_offsets = (
        initial_slot * stride_state_s
        + value_head * stride_state_h
        + keys[:, None] * stride_state_k
        + columns[None, :] * stride_state_v
    )
    scale_offsets = (
        initial_slot * stride_scale_s
        + value_head * stride_scale_h
        + columns * stride_scale_v
    )
    state_acc = _load_state(
        state_fp8,
        state_scale,
        state_offsets,
        scale_offsets,
        mask_state,
        mask_v,
        ACTIVATION_TYPE=q.dtype.element_ty,
    )

    q_offsets = (
        batch * stride_q_b
        + token * stride_q_t
        + query_head * stride_q_h
        + keys * stride_q_k
    )
    k_offsets = (
        batch * stride_k_b
        + token * stride_k_t
        + query_head * stride_k_h
        + keys * stride_k_k
    )
    q_values = tl.load(q + q_offsets, mask=mask_k, other=0.0)
    k_values = tl.load(k + k_offsets, mask=mask_k, other=0.0)
    q_low, k_low = _prepare_qk(
        q_values,
        k_values,
        scale,
        ACTIVATION_TYPE=q.dtype.element_ty,
        USE_QK_L2NORM=USE_QK_L2NORM,
    )
    value_offsets = (
        batch * stride_v_b
        + token * stride_v_t
        + value_head * stride_v_h
        + columns * stride_v_v
    )
    value = tl.load(v + value_offsets, mask=mask_v, other=0.0)
    gate_offset = batch * stride_g_b + token * stride_g_t + value_head * stride_g_h
    beta_offset = (
        batch * stride_beta_b + token * stride_beta_t + value_head * stride_beta_h
    )
    if USE_RAW_GATES:
        raw_a_offset = (
            batch * stride_raw_a_b
            + token * stride_raw_a_t
            + value_head * stride_raw_a_h
        )
        raw_b_offset = (
            batch * stride_raw_b_b
            + token * stride_raw_b_t
            + value_head * stride_raw_b_h
        )
        gate_input = tl.load(raw_a + raw_a_offset).to(tl.float32)
        gate_input += tl.load(dt_bias + value_head).to(tl.float32)
        softplus = tl.where(
            gate_input <= 20.0, tl.log(1.0 + tl.exp(gate_input)), gate_input
        )
        log_decay = -tl.exp(tl.load(A_log + value_head).to(tl.float32)) * softplus
        decay = tl.exp(log_decay)
        update = tl.sigmoid(tl.load(raw_b + raw_b_offset).to(tl.float32))
    else:
        decay = tl.exp(tl.load(g + gate_offset).to(tl.float32))
        update = tl.load(beta + beta_offset)
    state_acc, result = _gdn_step(
        state_acc,
        q_low,
        k_low,
        value,
        decay,
        update,
        ACTIVATION_TYPE=q.dtype.element_ty,
        FP32_STATE_PRODUCTS=True,
        FP32_UPDATE_PRODUCTS=True,
    )
    output_offsets = (
        batch * stride_o_b
        + token * stride_o_t
        + value_head * stride_o_h
        + columns * stride_o_v
    )
    tl.store(output + output_offsets, result.to(output.dtype.element_ty), mask=mask_v)

    safe_output_slot = tl.where(output_valid, output_slot, 0)
    output_state_offsets = (
        safe_output_slot * stride_state_s
        + value_head * stride_state_h
        + keys[:, None] * stride_state_k
        + columns[None, :] * stride_state_v
    )
    output_scale_offsets = (
        safe_output_slot * stride_scale_s
        + value_head * stride_scale_h
        + columns * stride_scale_v
    )
    _store_dynamic_fp8_state(
        state_acc,
        state_fp8,
        state_scale,
        output_state_offsets,
        output_scale_offsets,
        mask_state & output_valid,
        mask_v & output_valid,
        ACTIVATION_TYPE=q.dtype.element_ty,
        LOW_PRECISION_AMAX=LOW_PRECISION_AMAX,
    )


@triton.jit
def _load_state(
    state_fp8,
    state_scale,
    state_offsets,
    scale_offsets,
    mask_state,
    mask_v,
    ACTIVATION_TYPE: tl.constexpr,
):
    quantized = tl.load(state_fp8 + state_offsets, mask_state, 0.0).to(ACTIVATION_TYPE)
    scales = tl.load(state_scale + scale_offsets, mask_v, 0.0).to(ACTIVATION_TYPE)
    return (quantized * scales[None, :]).to(tl.float32)


def _mthreads_gdn_decode_w8a16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    scale: float,
    state_fp8: torch.Tensor,
    state_scale: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    A_log: torch.Tensor | None,
    raw_a: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    raw_b: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, query_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    sequence_count = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    if cu_seqlens is not None and tokens != sequence_count:
        raise NotImplementedError(
            "packed MThreads GDN decode requires one token per sequence"
        )
    if key_dim != 128 or value_dim != 128:
        raise NotImplementedError("MThreads GDN decode currently supports K=V=128")
    if state_indices is not None and state_indices.ndim not in (1, 2):
        raise ValueError("state indices must be one- or two-dimensional")
    if num_accepted_tokens is not None:
        if state_indices is None or state_indices.ndim != 2:
            raise ValueError("accepted-token metadata requires per-token state indices")
        if num_accepted_tokens.shape != (sequence_count,):
            raise ValueError("accepted-token metadata must have shape [num_sequences]")
    use_raw_gates = A_log is not None
    if use_raw_gates and any(tensor is None for tensor in (raw_a, dt_bias, raw_b)):
        raise ValueError("raw GDN gating requires A_log, a, dt_bias, and b")
    if not use_raw_gates and (g is None or beta is None):
        raise ValueError("precomputed GDN gating requires g and beta")
    gate_tensor = g if g is not None else A_log
    beta_tensor = beta if beta is not None else raw_b
    output = torch.empty(
        batch, tokens, value_heads, value_dim, device=q.device, dtype=q.dtype
    )
    if not sequence_count:
        return output, state_fp8, state_scale
    block_k = triton.next_power_of_2(key_dim)
    block_v = 32
    with torch_device_fn.device(q.device):
        _mthreads_gdn_decode_w8a16_kernel[
            (triton.cdiv(value_dim, block_v), sequence_count * value_heads)
        ](
            q,
            k,
            v,
            gate_tensor,
            beta_tensor,
            A_log if A_log is not None else gate_tensor,
            raw_a if raw_a is not None else q,
            dt_bias if dt_bias is not None else gate_tensor,
            raw_b if raw_b is not None else beta_tensor,
            output,
            state_fp8,
            state_scale,
            cu_seqlens,
            state_indices,
            num_accepted_tokens,
            scale,
            stride_q_b=q.stride(0),
            stride_q_t=q.stride(1),
            stride_q_h=q.stride(2),
            stride_q_k=q.stride(3),
            stride_k_b=k.stride(0),
            stride_k_t=k.stride(1),
            stride_k_h=k.stride(2),
            stride_k_k=k.stride(3),
            stride_v_b=v.stride(0),
            stride_v_t=v.stride(1),
            stride_v_h=v.stride(2),
            stride_v_v=v.stride(3),
            stride_g_b=gate_tensor.stride(0),
            stride_g_t=gate_tensor.stride(1) if gate_tensor.ndim == 3 else 0,
            stride_g_h=gate_tensor.stride(2) if gate_tensor.ndim == 3 else 0,
            stride_beta_b=beta_tensor.stride(0),
            stride_beta_t=beta_tensor.stride(1),
            stride_beta_h=beta_tensor.stride(2) if beta_tensor.ndim == 3 else 0,
            stride_raw_a_b=raw_a.stride(0) if raw_a is not None else 0,
            stride_raw_a_t=raw_a.stride(1) if raw_a is not None else 0,
            stride_raw_a_h=raw_a.stride(2) if raw_a is not None else 0,
            stride_raw_b_b=raw_b.stride(0) if raw_b is not None else 0,
            stride_raw_b_t=raw_b.stride(1) if raw_b is not None else 0,
            stride_raw_b_h=raw_b.stride(2) if raw_b is not None else 0,
            USE_RAW_GATES=use_raw_gates,
            stride_o_b=output.stride(0),
            stride_o_t=output.stride(1),
            stride_o_h=output.stride(2),
            stride_o_v=output.stride(3),
            stride_state_s=state_fp8.stride(0),
            stride_state_h=state_fp8.stride(1),
            stride_state_k=state_fp8.stride(2),
            stride_state_v=state_fp8.stride(3),
            stride_scale_s=state_scale.stride(0),
            stride_scale_h=state_scale.stride(1),
            stride_scale_v=state_scale.stride(2),
            stride_indices_seq=1 if state_indices is None else state_indices.stride(0),
            stride_indices_tok=(
                0
                if state_indices is None or state_indices.ndim == 1
                else state_indices.stride(1)
            ),
            H=query_heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            S=state_fp8.shape[0],
            BK=block_k,
            BV=block_v,
            USE_PACKED_INPUT=cu_seqlens is not None,
            USE_STATE_INDICES=state_indices is not None,
            USE_SPEC_DECODING=num_accepted_tokens is not None,
            USE_QK_L2NORM=use_qk_l2norm_in_kernel,
            LOW_PRECISION_AMAX=True,
            num_warps=1 if block_v == 4 else 4,
            num_stages=1,
        )
    return output, state_fp8, state_scale


def gdn_w8a16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    scale: float,
    state_fp8: torch.Tensor,
    state_scale: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    max_sequence_length: int | None = None,
    A_log: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run GDN with MThreads-native raw or precomputed gates."""
    single_token_decode = (
        max_sequence_length == 1 if cu_seqlens is not None else q.shape[1] == 1
    )
    raw_gates = A_log is not None
    if single_token_decode:
        return _mthreads_gdn_decode_w8a16(
            q,
            k,
            v,
            g,
            beta,
            scale,
            state_fp8,
            state_scale,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
            use_qk_l2norm_in_kernel,
            A_log,
            a,
            dt_bias,
            b,
        )
    if raw_gates:
        if num_accepted_tokens is not None:
            raise NotImplementedError(
                "raw GDN prefill does not produce speculative checkpoints"
            )

        return raw_prefill_w8a16_fp8(
            q,
            k,
            v,
            scale,
            state_fp8,
            state_scale,
            cu_seqlens,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
            A_log,
            a,
            dt_bias,
            b,
        )
    if num_accepted_tokens is None and (
        ssm_state_indices is None or ssm_state_indices.ndim == 1
    ):

        return recurrent_prefill_w8a16_fp8(
            q,
            k,
            v,
            g,
            beta,
            scale,
            state_fp8,
            state_scale,
            cu_seqlens,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
            max_sequence_length,
        )
    return _gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        scale,
        state_fp8,
        state_scale,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        max_sequence_length=max_sequence_length,
    )

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from functools import lru_cache

import triton
import triton.language as tl
from triton.language import core
from triton.runtime.cache import get_cache_manager

NATIVE_IR = r"""
declare void @llvm.musa.lma.wait()
define i32 @sparse_lma_wait(i32 %token) alwaysinline nounwind {
 call void @llvm.musa.lma.wait()
 ret i32 %token
}
declare void @llvm.musa.memcpy.g2s.robust.v4(ptr addrspace(3),ptr addrspace(1),i32,<4 x i32>,i32)
declare void @llvm.musa.memcpy.g2s.wait()
define i32 @bounded_copy16(i32 %dst,i64 %src,i64 %base,i64 %size) alwaysinline nounwind {
  %d = inttoptr i32 %dst to ptr addrspace(3)
  %s = inttoptr i64 %src to ptr addrspace(1)
  %b = bitcast i64 %base to <2 x i32>
  %z = bitcast i64 %size to <2 x i32>
  %desc = shufflevector <2 x i32> %b,<2 x i32> %z,<4 x i32> <i32 0,i32 1,i32 2,i32 3>
  call void @llvm.musa.memcpy.g2s.robust.v4(ptr addrspace(3) %d,ptr addrspace(1) %s,i32 16,<4 x i32> %desc,i32 0)
  ret i32 %dst
}
define i32 @bounded_copy_wait(i32 %token) alwaysinline nounwind {
  call void @llvm.musa.memcpy.g2s.wait()
  ret i32 %token
}

declare <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8>)
define i64 @sparse_promote_fp8_four(i32 %bytes) alwaysinline nounwind readnone {
  %v = bitcast i32 %bytes to <4 x i8>
  %h = call <4 x half> @llvm.musa.e4m32f16.rn.bst4(<4 x i8> %v)
  %result = bitcast <4 x half> %h to i64
  ret i64 %result
}

declare <2 x float> @llvm.musa.exp2.f.bst2(<2 x float>)
define i64 @sparse_exp2_pair(float %a, float %b) alwaysinline nounwind readnone {
  %a2 = fmul float %a, 0x4007154760000000
  %b2 = fmul float %b, 0x4007154760000000
  %v0 = insertelement <2 x float> poison, float %a2, i32 0
  %v1 = insertelement <2 x float> %v0, float %b2, i32 1
  %r = call <2 x float> @llvm.musa.exp2.f.bst2(<2 x float> %v1)
  %i = bitcast <2 x float> %r to i64
  ret i64 %i
}
"""


@lru_cache(maxsize=2)
def native_library(ir: str = NATIVE_IR) -> str:
    cache = get_cache_manager(hashlib.sha256(ir.encode()).hexdigest())
    filename = "sparse-fp8-pipeline.ll"
    return cache.get_file(filename) or cache.put(ir, filename, binary=False)


@core.extern
def lma_wait(token, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [token],
        {(tl.int32,): ("sparse_lma_wait", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def bounded_copy16(dst, src, base, size, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [dst, src, base, size],
        {(tl.uint32, tl.uint64, tl.uint64, tl.uint64): ("bounded_copy16", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def bounded_copy_wait(token, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [token],
        {(tl.int32,): ("bounded_copy_wait", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def promote_fp8_four(word, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [word],
        {(tl.uint32,): ("sparse_promote_fp8_four", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def exp2_pair(first, second, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [first, second],
        {(tl.float32, tl.float32): ("sparse_exp2_pair", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


QUERY_NORM_IR = (
    NATIVE_IR
    + r"""
define float @sparse_query_abs4(i32 %word) alwaysinline nounwind readnone {
  %masked = and i32 %word, 2139062143
  %promoted = call i64 @sparse_promote_fp8_four(i32 %masked)
  %h = bitcast i64 %promoted to <4 x half>
  %h0 = extractelement <4 x half> %h, i32 0
  %h1 = extractelement <4 x half> %h, i32 1
  %h2 = extractelement <4 x half> %h, i32 2
  %h3 = extractelement <4 x half> %h, i32 3
  %s0 = fadd half %h0, %h1
  %s1 = fadd half %h2, %h3
  %sum = fadd half %s0, %s1
  %result = fpext half %sum to float
  ret float %result
}
"""
)


@core.extern
def query_abs4(word, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [word],
        {(tl.uint32,): ("sparse_query_abs4", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


VECTOR_EXP2_IR = (
    NATIVE_IR
    + r"""
declare <4 x float> @llvm.musa.exp2.f.bst4(<4 x float>) nounwind readnone
define <4 x float> @sparse_exp2_four(float %a,float %b,float %c,float %d) alwaysinline nounwind readnone {
  %v0 = insertelement <4 x float> poison, float %a, i32 0
  %v1 = insertelement <4 x float> %v0, float %b, i32 1
  %v2 = insertelement <4 x float> %v1, float %c, i32 2
  %v3 = insertelement <4 x float> %v2, float %d, i32 3
  %value = call <4 x float> @llvm.musa.exp2.f.bst4(<4 x float> %v3) nounwind readnone
  ret <4 x float> %value
}

define i64 @sparse_exp2_four_first(float %a,float %b,float %c,float %d) alwaysinline nounwind readnone {
  %value = call <4 x float> @sparse_exp2_four(float %a,float %b,float %c,float %d)
  %packed = bitcast <4 x float> %value to <2 x i64>
  %pair = extractelement <2 x i64> %packed, i32 0
  ret i64 %pair
}

define i64 @sparse_exp2_four_second(float %a,float %b,float %c,float %d) alwaysinline nounwind readnone {
  %value = call <4 x float> @sparse_exp2_four(float %a,float %b,float %c,float %d)
  %packed = bitcast <4 x float> %value to <2 x i64>
  %pair = extractelement <2 x i64> %packed, i32 1
  ret i64 %pair
}

"""
)


@core.extern
def exp2_four_first(a, b, c, d, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [a, b, c, d],
        {(tl.float32,) * 4: ("sparse_exp2_four_first", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def exp2_four_second(a, b, c, d, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [a, b, c, d],
        {(tl.float32,) * 4: ("sparse_exp2_four_second", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def vector_exp2(scaled, HEADS: tl.constexpr, BN: tl.constexpr):
    # Token bits 3 and 4 stay in one SQMMA thread's registers for this tile.
    shape = tl.reshape(scaled, (HEADS, BN // 32, 2, 1, 2, 8))
    order = tl.permute(shape, (0, 1, 3, 5, 2, 4))
    left, right = tl.split(order)
    exp_a, exp_c = tl.split(left)
    exp_b, exp_d = tl.split(right)
    first = exp2_four_first(exp_a, exp_b, exp_c, exp_d)
    second = exp2_four_second(exp_a, exp_b, exp_c, exp_d)
    fa = first.to(tl.uint32).to(tl.float32, bitcast=True)
    fb = (first >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    fc = second.to(tl.uint32).to(tl.float32, bitcast=True)
    fd = (second >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    back = tl.join(tl.join(fa, fc), tl.join(fb, fd))
    return tl.reshape(tl.permute(back, (0, 1, 4, 2, 5, 3)), (HEADS, BN))

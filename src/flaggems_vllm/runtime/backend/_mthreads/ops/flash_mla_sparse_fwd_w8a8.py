# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import math
import re
from functools import lru_cache
from typing import Callable, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError
from triton.language import core
from triton.runtime import driver
from triton.runtime.cache import get_cache_manager
from triton.runtime.errors import OutOfResources
from triton.runtime.jit import JITFunction

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import has_triton_tle_attrs

# The general Triton kernel also runs with compilers that lack MUSA TLE.
HAS_TLE = has_triton_tle_attrs(("gpu",))
if HAS_TLE:
    from triton.experimental.tle import language as tle
else:
    tle = None

try:
    from triton._C.libtriton import mthreads as musa_compiler
    from triton.experimental.tle.language.gpu.mthreads.types import MusaSqmmaEncoding
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    HAS_MODEL1_PIPELINE = False
else:
    HAS_MODEL1_PIPELINE = HAS_TLE and has_triton_tle_attrs(("gpu.SlicedEncoding",))


# MUSA instructions and layouts


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


_V32_DIRECT_COPY_COMMON_IR = r"""
declare i32 @llvm.musa.read.ptx.sreg.tid.x()
declare void @llvm.musa.memcpy.g2s(ptr addrspace(3), ptr addrspace(1), i32, i32)

; MP31 maps each 256-byte line by XORing its 16-byte sector index.
; The operation is its own inverse. It also works when a view starts
; at a sector other than the first sector in a line.
define internal i32 @v32_swizzle(i32 %address) alwaysinline nounwind readnone {
  %line = lshr i32 %address, 8
  %sector_mask = and i32 %line, 15
  %mask = shl i32 %sector_mask, 4
  %swizzled = xor i32 %address, %mask
  ret i32 %swizzled
}

define internal void @v32_copy_sector(i64 %address, i32 %logical) alwaysinline nounwind {
  %physical = call i32 @v32_swizzle(i32 %logical)
  %dst = inttoptr i32 %physical to ptr addrspace(3)
  %src = inttoptr i64 %address to ptr addrspace(1)
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %dst, ptr addrspace(1) %src, i32 16, i32 0)
  ret void
}

define internal void @v32_copy_page_general(
    i32 %dst, i64 %kv, i64 %indices, i32 %tokens, i32 %rope_dst, i64 %kr
) alwaysinline nounwind {
entry:
  %tid = call i32 @llvm.musa.read.ptx.sreg.tid.x()
  %loader_lane = and i32 %tid, 127
  %first_row = lshr i32 %loader_lane, 2
  %sector = and i32 %loader_lane, 3
  %col = shl i32 %sector, 4
  %col64 = zext i32 %col to i64
  %base = call i32 @v32_swizzle(i32 %dst)
  %rope_base = call i32 @v32_swizzle(i32 %rope_dst)
  %index_ptr = inttoptr i64 %indices to ptr addrspace(1)
  %copy_rope = icmp ne i64 %kr, 0
  br label %loop

loop:
  %row = phi i32 [ %first_row, %entry ], [ %next, %advance ]
  %row64 = zext i32 %row to i64
  %index_address = getelementptr i32, ptr addrspace(1) %index_ptr, i64 %row64
  %index = load i32, ptr addrspace(1) %index_address, align 4
  %valid = icmp ult i32 %index, %tokens
  %selected = select i1 %valid, i32 %index, i32 0
  %key = zext i32 %selected to i64
  %key_bytes = shl i64 %key, 9
  %input_row = add i64 %kv, %key_bytes
  %src0 = add i64 %input_row, %col64
  %row_bytes = shl i32 %row, 8
  %logical_row = add i32 %base, %row_bytes
  %dst0 = add i32 %logical_row, %col
  call void @v32_copy_sector(i64 %src0, i32 %dst0)
  %src1 = add i64 %src0, 64
  %dst1 = add i32 %dst0, 64
  call void @v32_copy_sector(i64 %src1, i32 %dst1)
  %src2 = add i64 %src0, 128
  %dst2 = add i32 %dst0, 128
  call void @v32_copy_sector(i64 %src2, i32 %dst2)
  %src3 = add i64 %src0, 192
  %dst3 = add i32 %dst0, 192
  call void @v32_copy_sector(i64 %src3, i32 %dst3)
  br i1 %copy_rope, label %rope, label %advance

rope:
  %rkey_bytes = shl i64 %key, 7
  %rinput_row = add i64 %kr, %rkey_bytes
  %rsrc0 = add i64 %rinput_row, %col64
  %rrow_bytes = shl i32 %row, 7
  %rlogical_row = add i32 %rope_base, %rrow_bytes
  %rdst0 = add i32 %rlogical_row, %col
  call void @v32_copy_sector(i64 %rsrc0, i32 %rdst0)
  %rsrc1 = add i64 %rsrc0, 64
  %rdst1 = add i32 %rdst0, 64
  call void @v32_copy_sector(i64 %rsrc1, i32 %rdst1)
  br label %advance

advance:
  %next = add nuw nsw i32 %row, 32
  %again = icmp ult i32 %next, 128
  br i1 %again, label %loop, label %done, !llvm.loop !901

done:
  ret void
}

define internal void @v32_copy_aligned_kv(i32 %first_dst, i64 %values, i64 %indices,
                                      i32 %limit, i32 %tokens) alwaysinline nounwind {
entry:
  %tid = call i32 @llvm.musa.read.ptx.sreg.tid.x()
  %physical_lane = and i32 %tid, 127
  %col = and i32 %physical_lane, 3
  %start = lshr i32 %physical_lane, 2
  %base = and i32 %first_dst, -256
  %base_row = lshr i32 %base, 8
  %value_ptr = inttoptr i64 %values to ptr addrspace(1)
  %index_ptr = inttoptr i64 %indices to ptr addrspace(1)
  br label %loop

loop:
  %row = phi i32 [ %start, %entry ], [ %next, %copy ]
  %in_span = icmp slt i32 %row, %limit
  br i1 %in_span, label %read, label %copy

read:
  %offset = zext i32 %row to i64
  %address = getelementptr i32, ptr addrspace(1) %index_ptr, i64 %offset
  %index = load i32, ptr addrspace(1) %address, align 4
  %valid = icmp ult i32 %index, %tokens
  %selected = select i1 %valid, i32 %index, i32 0
  br label %copy

copy:
  %safe = phi i32 [ %selected, %read ], [ 0, %loop ]
  %safe64 = zext i32 %safe to i64
  %row_bytes = shl i64 %safe64, 9
  %col32 = shl i32 %col, 4
  %col64 = zext i32 %col32 to i64
  %index64 = or i64 %row_bytes, %col64
  %input = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %index64
  %row_in_smem = add i32 %row, %base_row
  %phase = and i32 %row_in_smem, 15
  %column = xor i32 %phase, %col
  %local_col = shl i32 %column, 4
  %local_row = shl i32 %row, 8
  %local_offset = or i32 %local_row, %local_col
  %destination = add i32 %base, %local_offset
  %output = inttoptr i32 %destination to ptr addrspace(3)
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %output, ptr addrspace(1) %input, i32 16, i32 0)

  %input1 = getelementptr i8, ptr addrspace(1) %input, i64 64
  %permuted1 = xor i32 %local_offset, 64
  %destination1 = add i32 %base, %permuted1
  %output1 = inttoptr i32 %destination1 to ptr addrspace(3)
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %output1, ptr addrspace(1) %input1, i32 16, i32 0)

  %input2 = getelementptr i8, ptr addrspace(1) %input, i64 128
  %permuted2 = xor i32 %local_offset, 128
  %destination2 = add i32 %base, %permuted2
  %output2 = inttoptr i32 %destination2 to ptr addrspace(3)
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %output2, ptr addrspace(1) %input2, i32 16, i32 0)

  %input3 = getelementptr i8, ptr addrspace(1) %input, i64 192
  %permuted3 = xor i32 %local_offset, 192
  %destination3 = add i32 %base, %permuted3
  %output3 = inttoptr i32 %destination3 to ptr addrspace(3)
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %output3, ptr addrspace(1) %input3, i32 16, i32 0)
  %next = add nuw nsw i32 %row, 32
  %again = icmp slt i32 %next, 128
  br i1 %again, label %loop, label %done, !llvm.loop !911

done:
  ret void
}

!911 = distinct !{!911, !902}
"""
_V32_FOUR_LANE_ROPE_IR = r"""define internal void @v32_copy_aligned_rope(i32 %first_dst, i64 %values, i64 %indices,
                                      i32 %limit, i32 %tokens, i32 %rope_dest, i64 %rope_values) alwaysinline nounwind {
entry:
  %tid = call i32 @llvm.musa.read.ptx.sreg.tid.x()
  %physical_lane = and i32 %tid, 127
  %col = and i32 %physical_lane, 3
  %start = lshr i32 %physical_lane, 2
  %base = and i32 %first_dst, -256
  %base_row = lshr i32 %base, 8
  %value_ptr = inttoptr i64 %values to ptr addrspace(1)
  %index_ptr = inttoptr i64 %indices to ptr addrspace(1)
  br label %loop

loop:
  %row = phi i32 [ %start, %entry ], [ %next, %copy ]
  %in_span = icmp slt i32 %row, %limit
  br i1 %in_span, label %read, label %copy

read:
  %offset = zext i32 %row to i64
  %address = getelementptr i32, ptr addrspace(1) %index_ptr, i64 %offset
  %index = load i32, ptr addrspace(1) %address, align 4
  %valid = icmp ult i32 %index, %tokens
  %selected = select i1 %valid, i32 %index, i32 0
  br label %copy

copy:
  %safe = phi i32 [ %selected, %read ], [ 0, %loop ]
  %safe64 = zext i32 %safe to i64
  %row_bytes = shl i64 %safe64, 9
  %col32 = shl i32 %col, 4
  %col64 = zext i32 %col32 to i64
  %index64 = or i64 %row_bytes, %col64
  %input = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %index64
  %row_in_smem = add i32 %row, %base_row
  %phase = and i32 %row_in_smem, 15
  %column = xor i32 %phase, %col
  %local_col = shl i32 %column, 4
  %local_row = shl i32 %row, 8
  %local_offset = or i32 %local_row, %local_col
  %destination = add i32 %base, %local_offset
  %output = inttoptr i32 %destination to ptr addrspace(3)
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %output, ptr addrspace(1) %input, i32 16, i32 0)

  %input1 = getelementptr i8, ptr addrspace(1) %input, i64 64
  %permuted1 = xor i32 %local_offset, 64
  %destination1 = add i32 %base, %permuted1
  %output1 = inttoptr i32 %destination1 to ptr addrspace(3)
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %output1, ptr addrspace(1) %input1, i32 16, i32 0)

  %input2 = getelementptr i8, ptr addrspace(1) %input, i64 128
  %permuted2 = xor i32 %local_offset, 128
  %destination2 = add i32 %base, %permuted2
  %output2 = inttoptr i32 %destination2 to ptr addrspace(3)
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %output2, ptr addrspace(1) %input2, i32 16, i32 0)

  %input3 = getelementptr i8, ptr addrspace(1) %input, i64 192
  %permuted3 = xor i32 %local_offset, 192
  %destination3 = add i32 %base, %permuted3
  %output3 = inttoptr i32 %destination3 to ptr addrspace(3)
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %output3, ptr addrspace(1) %input3, i32 16, i32 0)

  %rope_base = and i32 %rope_dest, -256
  %key_source = shl i64 %safe64, 7
  %rope_row = shl i32 %row, 7
  %rope_pointer = inttoptr i64 %rope_values to ptr addrspace(1)

  %rcol0 = add i32 %col32, 0
  %rlinear0 = add i32 %rope_row, %rcol0
  %rabs0 = add i32 %rope_base, %rlinear0
  %rphase0 = lshr i32 %rabs0, 8
  %rphase40 = shl i32 %rphase0, 4
  %rmask0 = and i32 %rphase40, 240
  %rphysical0 = xor i32 %rabs0, %rmask0
  %routput0 = inttoptr i32 %rphysical0 to ptr addrspace(3)
  %rcol640 = zext i32 %rcol0 to i64
  %rsource0 = add i64 %key_source, %rcol640
  %rinput0 = getelementptr i8, ptr addrspace(1) %rope_pointer, i64 %rsource0
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %routput0, ptr addrspace(1) %rinput0, i32 16, i32 0)

  %rcol1 = add i32 %col32, 64
  %rlinear1 = add i32 %rope_row, %rcol1
  %rabs1 = add i32 %rope_base, %rlinear1
  %rphase1 = lshr i32 %rabs1, 8
  %rphase41 = shl i32 %rphase1, 4
  %rmask1 = and i32 %rphase41, 240
  %rphysical1 = xor i32 %rabs1, %rmask1
  %routput1 = inttoptr i32 %rphysical1 to ptr addrspace(3)
  %rcol641 = zext i32 %rcol1 to i64
  %rsource1 = add i64 %key_source, %rcol641
  %rinput1 = getelementptr i8, ptr addrspace(1) %rope_pointer, i64 %rsource1
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %routput1, ptr addrspace(1) %rinput1, i32 16, i32 0)
  %next = add nuw nsw i32 %row, 32
  %again = icmp slt i32 %next, 128
  br i1 %again, label %loop, label %done, !llvm.loop !921

done:
  ret void
}

!921 = distinct !{!921, !902}

"""
# Two lanes per token issue the same sectors over two groups of 64 rows.
_V32_TWO_LANE_ROPE_IR = r"""define internal void @v32_copy_aligned_rope(i32 %first_dst, i64 %values, i64 %indices,
                                      i32 %limit, i32 %tokens, i32 %rope_dest, i64 %rope_values) alwaysinline nounwind {
entry:
  %tid = call i32 @llvm.musa.read.ptx.sreg.tid.x()
  %physical_lane = and i32 %tid, 127
  %col = and i32 %physical_lane, 1
  %start = lshr i32 %physical_lane, 1
  %base = and i32 %first_dst, -256
  %value_ptr = inttoptr i64 %values to ptr addrspace(1)
  %index_ptr = inttoptr i64 %indices to ptr addrspace(1)
  br label %loop

loop:
  %row = phi i32 [ %start, %entry ], [ %next, %loop ]
  %row64 = zext i32 %row to i64
  %read = getelementptr i32, ptr addrspace(1) %index_ptr, i64 %row64
  %idx = load i32, ptr addrspace(1) %read, align 4
  %valid = icmp ult i32 %idx, %tokens
  %safe = select i1 %valid, i32 %idx, i32 0
  %safe64 = zext i32 %safe to i64
  %colb = shl i32 %col, 4
  %col64 = zext i32 %colb to i64
  %rope_base = and i32 %rope_dest, -256
  %rope_ptr = inttoptr i64 %rope_values to ptr addrspace(1)
  %kvrow = shl i32 %row, 8
  %kvrowcol = or i32 %kvrow, %colb
  %kvlinear = add i32 %base, %kvrowcol
  %kvshift = lshr i32 %kvlinear, 8
  %kvphase = and i32 %kvshift, 15
  %kvsw = shl i32 %kvphase, 4
  %kvphysical = xor i32 %kvlinear, %kvsw
  %kvsrcrow = shl i64 %safe64, 9
  %kvsource = or i64 %kvsrcrow, %col64
  %kvdst0 = xor i32 %kvphysical, 0
  %kvd0 = inttoptr i32 %kvdst0 to ptr addrspace(3)
  %kvidx0 = add i64 %kvsource, 0
  %kvp0 = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %kvidx0
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %kvd0, ptr addrspace(1) %kvp0, i32 16, i32 0)
  %kvdst1 = xor i32 %kvphysical, 32
  %kvd1 = inttoptr i32 %kvdst1 to ptr addrspace(3)
  %kvidx1 = add i64 %kvsource, 32
  %kvp1 = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %kvidx1
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %kvd1, ptr addrspace(1) %kvp1, i32 16, i32 0)
  %kvdst2 = xor i32 %kvphysical, 64
  %kvd2 = inttoptr i32 %kvdst2 to ptr addrspace(3)
  %kvidx2 = add i64 %kvsource, 64
  %kvp2 = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %kvidx2
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %kvd2, ptr addrspace(1) %kvp2, i32 16, i32 0)
  %kvdst3 = xor i32 %kvphysical, 96
  %kvd3 = inttoptr i32 %kvdst3 to ptr addrspace(3)
  %kvidx3 = add i64 %kvsource, 96
  %kvp3 = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %kvidx3
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %kvd3, ptr addrspace(1) %kvp3, i32 16, i32 0)
  %kvdst4 = xor i32 %kvphysical, 128
  %kvd4 = inttoptr i32 %kvdst4 to ptr addrspace(3)
  %kvidx4 = add i64 %kvsource, 128
  %kvp4 = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %kvidx4
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %kvd4, ptr addrspace(1) %kvp4, i32 16, i32 0)
  %kvdst5 = xor i32 %kvphysical, 160
  %kvd5 = inttoptr i32 %kvdst5 to ptr addrspace(3)
  %kvidx5 = add i64 %kvsource, 160
  %kvp5 = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %kvidx5
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %kvd5, ptr addrspace(1) %kvp5, i32 16, i32 0)
  %kvdst6 = xor i32 %kvphysical, 192
  %kvd6 = inttoptr i32 %kvdst6 to ptr addrspace(3)
  %kvidx6 = add i64 %kvsource, 192
  %kvp6 = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %kvidx6
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %kvd6, ptr addrspace(1) %kvp6, i32 16, i32 0)
  %kvdst7 = xor i32 %kvphysical, 224
  %kvd7 = inttoptr i32 %kvdst7 to ptr addrspace(3)
  %kvidx7 = add i64 %kvsource, 224
  %kvp7 = getelementptr i8, ptr addrspace(1) %value_ptr, i64 %kvidx7
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %kvd7, ptr addrspace(1) %kvp7, i32 16, i32 0)
  %krrow = shl i32 %row, 7
  %krrowcol = or i32 %krrow, %colb
  %krlinear = add i32 %rope_base, %krrowcol
  %krshift = lshr i32 %krlinear, 8
  %krphase = and i32 %krshift, 15
  %krsw = shl i32 %krphase, 4
  %krphysical = xor i32 %krlinear, %krsw
  %krsrcrow = shl i64 %safe64, 7
  %krsource = or i64 %krsrcrow, %col64
  %krdst0 = xor i32 %krphysical, 0
  %krd0 = inttoptr i32 %krdst0 to ptr addrspace(3)
  %kridx0 = add i64 %krsource, 0
  %krp0 = getelementptr i8, ptr addrspace(1) %rope_ptr, i64 %kridx0
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %krd0, ptr addrspace(1) %krp0, i32 16, i32 0)
  %krdst1 = xor i32 %krphysical, 32
  %krd1 = inttoptr i32 %krdst1 to ptr addrspace(3)
  %kridx1 = add i64 %krsource, 32
  %krp1 = getelementptr i8, ptr addrspace(1) %rope_ptr, i64 %kridx1
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %krd1, ptr addrspace(1) %krp1, i32 16, i32 0)
  %krdst2 = xor i32 %krphysical, 64
  %krd2 = inttoptr i32 %krdst2 to ptr addrspace(3)
  %kridx2 = add i64 %krsource, 64
  %krp2 = getelementptr i8, ptr addrspace(1) %rope_ptr, i64 %kridx2
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %krd2, ptr addrspace(1) %krp2, i32 16, i32 0)
  %krdst3 = xor i32 %krphysical, 96
  %krd3 = inttoptr i32 %krdst3 to ptr addrspace(3)
  %kridx3 = add i64 %krsource, 96
  %krp3 = getelementptr i8, ptr addrspace(1) %rope_ptr, i64 %kridx3
  call void @llvm.musa.memcpy.g2s(ptr addrspace(3) %krd3, ptr addrspace(1) %krp3, i32 16, i32 0)
  %next = add nuw nsw i32 %row, 64
  %again = icmp slt i32 %next, %limit
  br i1 %again, label %loop, label %done, !llvm.loop !931
done:
  ret void
}

!931 = distinct !{!931, !932}
!932 = !{!"llvm.loop.unroll.count", i32 2}

"""
_V32_DIRECT_COPY_ENTRY_IR = r"""define i32 @v32_copy_page(
    i32 %dst, i64 %kv, i64 %indices, i32 %tokens, i32 %rope_dst, i64 %kr
) alwaysinline nounwind {
entry:
  %kv_base = call i32 @v32_swizzle(i32 %dst)
  %rope_base = call i32 @v32_swizzle(i32 %rope_dst)
  %bases = or i32 %kv_base, %rope_base
  %remainder = and i32 %bases, 255
  %aligned = icmp eq i32 %remainder, 0
  br i1 %aligned, label %choose, label %general

choose:
  %with_rope = icmp ne i64 %kr, 0
  br i1 %with_rope, label %fused, label %kv_only

fused:
  call void @v32_copy_aligned_rope(i32 %dst, i64 %kv, i64 %indices, i32 128, i32 %tokens, i32 %rope_dst, i64 %kr)
  br label %done

kv_only:
  call void @v32_copy_aligned_kv(i32 %dst, i64 %kv, i64 %indices, i32 128, i32 %tokens)
  br label %done

general:
  call void @v32_copy_page_general(i32 %dst, i64 %kv, i64 %indices, i32 %tokens, i32 %rope_dst, i64 %kr)
  br label %done

done:
  ; Publish copied tiles only after the G2S operations complete.
  call void @llvm.musa.memcpy.g2s.wait()
  ret i32 %dst
}

!901 = distinct !{!901, !902}
!902 = !{!"llvm.loop.unroll.count", i32 4}
"""

V32_DIRECT_COPY_IR = (
    _V32_DIRECT_COPY_COMMON_IR + _V32_FOUR_LANE_ROPE_IR + _V32_DIRECT_COPY_ENTRY_IR
)
V32_DIRECT_COPY_TWO_LANE_IR = (
    _V32_DIRECT_COPY_COMMON_IR + _V32_TWO_LANE_ROPE_IR + _V32_DIRECT_COPY_ENTRY_IR
)


@core.extern
def v32_copy_page(dst, kv, indices, tokens, rope_dst, kr, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [dst, kv, indices, tokens, rope_dst, kr],
        {
            (tl.uint32, tl.uint64, tl.uint64, tl.int32, tl.uint32, tl.uint64): (
                "v32_copy_page",
                tl.int32,
            )
        },
        is_pure=False,
        _semantic=_semantic,
    )


# Selected KV metadata and precision checks


KV_TOKEN_BYTES = tl.constexpr(512)
KV_OFFSET_TOKEN_LIMIT = (1 << 32) // KV_TOKEN_BYTES.value
BOUNDED_EXP_PV_BUDGET = tl.constexpr(8.1e-5)
FAST_NATIVE_EXP = tl.constexpr(1)
FAST_BOUNDED_EXP = tl.constexpr(2)

PACKED_METADATA_TOKEN_LIMIT = 1 << 17

# Packet bits 0..7 contain a positive normal float32 scale exponent;
# bits 9..25 contain the token's 512-byte aligned offset. Invalid indices
# encode the cache-end offset and a zero scale. All cache scales are checked
# on every call. Other formats retain exact float32 scale bits and raw indices.
# Fast flags retain the original exponent mode in bits 0..1; bit 2 selects
# this packet format. The original error bound still selects FP32 repair.


@triton.jit
def encode_kv_byte_offset(token, valid, TOKENS: tl.constexpr):
    # Clamp invalid token IDs before the 32-bit multiplication can wrap.
    return (tl.where(valid, token, TOKENS).to(tl.uint32) * KV_TOKEN_BYTES).to(tl.int32)


@triton.jit
def gather_scales(
    Indices,
    KS,
    Selected,
    Offsets,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED_META: tl.constexpr = False,
    Global=None,
    CacheBits=None,
):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(Indices + x, x < ROWS * K, -1)
    valid = (x < ROWS * K) & (ids >= 0) & (ids < TOKENS)
    if PACKED_META:
        minimum = tl.load(Global + 1).to(tl.int32)
        if minimum > 0:
            safe_ids = tl.where(valid, ids, 0).to(tl.uint32)
            token_bits = tl.load(CacheBits + (safe_ids >> 5), valid, 0).to(tl.uint32)
            parity = (token_bits >> (safe_ids & 31)) & 1
            exponent = tl.where(
                valid, minimum.to(tl.uint32) + (parity ^ (minimum & 1).to(tl.uint32)), 0
            )
            location = encode_kv_byte_offset(ids, valid, TOKENS).to(tl.uint32)
            word = exponent | location
        else:
            scale = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
            word = scale.to(tl.uint32, bitcast=True)
        tl.store(Selected + x, word, x < ROWS * K)
    else:
        scale = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        tl.store(Selected + x, scale, x < ROWS * K)
        if Offsets is not None:
            tl.store(
                Offsets + x, encode_kv_byte_offset(ids, valid, TOKENS), x < ROWS * K
            )


@triton.jit
def cache_scale_bounds(
    KS,
    Indices,
    Selected,
    Offsets,
    ScaleParts,
    BadCount,
    TOKENS: tl.constexpr,
    SELECTED: tl.constexpr,
    BT: tl.constexpr,
    PACKED_META: tl.constexpr = False,
    CacheBits=None,
):
    if tl.program_id(0) == 0:
        tl.store(BadCount, 0)
    offset = tl.program_id(0) * BT + tl.arange(0, BT)
    if SELECTED:
        token = tl.load(Indices + offset, offset < SELECTED, -1)
        valid = (offset < SELECTED) & (token >= 0) & (token < TOKENS)
    else:
        token = offset
        valid = token < TOKENS
    scale = tl.load(KS + tl.where(valid, token, 0), valid, 0.0)
    if SELECTED:
        tl.store(Selected + offset, scale, offset < SELECTED)
        if Offsets is not None:
            tl.store(
                Offsets + offset,
                encode_kv_byte_offset(token, valid, TOKENS),
                offset < SELECTED,
            )
    tl.store(ScaleParts + tl.program_id(0), tl.max(scale, 0))
    if PACKED_META:
        tl.static_assert(SELECTED == 0)
        PARTS: tl.constexpr = triton.cdiv(TOKENS, BT)
        bits = scale.to(tl.uint32, bitcast=True)
        exponent = (bits >> 23) & 255
        exact = ((bits & 0x807FFFFF) == 0) & (exponent > 0) & (exponent < 255)
        bad = tl.max((valid & ~exact).to(tl.int32), 0)
        minimum = tl.min(tl.where(valid, exponent, 255), 0)
        parity = tl.reshape(tl.where(valid, exponent & 1, 0), (BT // 32, 32))
        word = tl.sum(parity << tl.arange(0, 32)[None, :], 1)
        pos = tl.program_id(0) * (BT // 32) + tl.arange(0, BT // 32)
        tl.store(CacheBits + pos, word, pos < triton.cdiv(TOKENS, 32))
        tl.store(ScaleParts + PARTS + tl.program_id(0), minimum.to(tl.float32))
        tl.store(ScaleParts + 2 * PARTS + tl.program_id(0), bad.to(tl.float32))


@triton.jit
def reduce_scales(
    ScaleParts,
    Global,
    PARTS: tl.constexpr,
    BP: tl.constexpr,
    PACKED_META: tl.constexpr = False,
):
    part = tl.arange(0, BP)
    maximum = tl.max(tl.load(ScaleParts + part, part < PARTS, 0.0), 0)
    tl.store(Global, maximum)
    if PACKED_META:
        minimum = tl.min(tl.load(ScaleParts + PARTS + part, part < PARTS, 255.0), 0).to(
            tl.int32
        )
        bad = tl.max(tl.load(ScaleParts + 2 * PARTS + part, part < PARTS, 0.0), 0)
        maximum_exponent = ((maximum.to(tl.uint32, bitcast=True) >> 23) & 255).to(
            tl.int32
        )
        codec_ok = (
            (bad == 0)
            & (maximum_exponent >= minimum)
            & (maximum_exponent - minimum <= 1)
        )
        tl.store(Global + 1, tl.where(codec_ok, minimum, 0).to(tl.float32))


@triton.jit(noinline=True)
def repair_k128(
    Q,
    QS,
    KV,
    KS,
    Indices,
    Output,
    LSE,
    row,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
):
    selected = tl.arange(0, 128)
    indices = tl.load(Indices + row * 128 + selected)
    valid = (indices >= 0) & (indices < TOKENS)
    if tl.max(valid.to(tl.int32), 0) == 0:
        lanes = tl.arange(0, 1024)
        for start in range(0, 64 * 512, 1024):
            tl.store(Output + row * (64 * 512) + start + lanes, 0.0)
        tl.store(LSE + row * 64 + tl.arange(0, 64), float("inf"))
    else:
        HEADS: tl.constexpr = 8
        FEATURES: tl.constexpr = 8
        tokens = tl.where(valid, indices, 0).to(tl.int64)
        gains = tl.load(KS + tokens, valid, 0.0)
        columns = tl.arange(0, FEATURES)
        # Reuse each key tile across a head group while bounding register use.
        for group in range(64 // HEADS):
            h = row * 64 + group * HEADS + tl.arange(0, HEADS)
            q_scale = tl.load(QS + h)
            scores = tl.full((HEADS, 128), 0.0, tl.float32)
            for start in range(0, 512, FEATURES):
                dims = start + columns
                query = tl.load(Q + h[:, None] * 512 + dims[None, :]).to(tl.float32)
                query *= q_scale[:, None]
                key = tl.load(
                    KV + tokens[:, None] * 512 + dims[None, :], valid[:, None], 0.0
                ).to(tl.float32)
                key *= gains[:, None]
                scores += tl.sum(query[:, None, :] * key[None, :, :], 2)
            logits = tl.where(valid[None, :], scores * SCALE, float("-inf"))
            maximum = tl.max(logits, 1)
            center = tl.where(maximum == float("-inf"), 0.0, maximum)
            weights = tl.where(valid[None, :], tl.exp(logits - center[:, None]), 0.0)
            total = tl.sum(weights, 1)
            normal = tl.where(total > 0.0, total, 1.0)
            tl.store(
                LSE + h, tl.where(total > 0.0, center + tl.log(total), float("inf"))
            )
            weights /= normal[:, None]
            for start in range(0, 512, FEATURES):
                dims = start + columns
                value = tl.load(
                    KV + tokens[:, None] * 512 + dims[None, :], valid[:, None], 0.0
                ).to(tl.float32)
                value *= gains[:, None]
                acc = tl.sum(value[None, :, :] * weights[:, :, None], 1)
                tl.store(Output + h[:, None] * 512 + dims[None, :], acc)


@triton.jit
def query_bounds_global(
    Q,
    QS,
    Global,
    PScale,
    Fast,
    BadRows,
    BadCount,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    PACKED: tl.constexpr = False,
    Indices=None,
    KS=None,
    Selected=None,
    Offsets=None,
    TOKENS: tl.constexpr = 0,
    PACKED_META: tl.constexpr = False,
    KV=None,
    Output=None,
    LSE=None,
):
    row = tl.program_id(0).to(tl.int64)
    if Indices is None:
        kscale = tl.load(Global)
    else:
        col = tl.arange(0, triton.next_power_of_2(K))
        ids = tl.load(Indices + row * K + col, col < K, -1)
        valid = (col < K) & (ids >= 0) & (ids < TOKENS)
        scales = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        tl.store(Selected + row * K + col, scales, col < K)
        if Offsets is not None:
            byte_offset = encode_kv_byte_offset(ids, valid, TOKENS)
            tl.store(Offsets + row * K + col, byte_offset, col < K)
        # Each query only uses its own selected KV rows.
        kscale = tl.max(scales, 0)
    h = tl.arange(0, 64)
    qs = tl.load(QS + row * 64 + h)
    if PACKED:
        d = tl.arange(0, 128)
        words = tl.load(
            Q.to(tl.pointer_type(tl.uint32))
            + (row * 64 + h[:, None]) * 128
            + d[None, :]
        )
        q_parts = query_abs4(words)
        # Cover both FP16 addition levels while retaining the original L1 margin.
        bound = tl.sum(q_parts, 1) * qs * (448.0 * kscale) * SCALE * 1.00154 + 1e-5
    else:
        d = tl.arange(0, 512)
        q = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :]).to(tl.float32)
        bound = tl.sum(tl.abs(q), 1) * qs * (448.0 * kscale) * SCALE * 1.0005 + 1e-5
    safe_bound = tl.minimum(bound, 20.0)
    exp_bound = tl.exp(safe_bound)
    scale = kscale * exp_bound * (1.0001 / 448.0)
    if K == 640 and PACKED and Indices is not None:
        # For this common range, the original score and PV budgets are
        # bounded by 7.2e-5 and 0.0024 respectively (including rounding).
        head_certified = (
            (qs >= 0)
            & (bound >= 0)
            & (bound <= 2.0)
            & (exp_bound <= 8.0)
            & (scale >= 2.0**-126)
        )
        certified = (
            (kscale <= 2.0**-9)
            & (kscale > 0)
            & (tl.min(head_certified.to(tl.int32), 0) != 0)
        )
    else:
        certified = False
    if certified:
        use_poly = True
        eligible = True
    else:
        U: tl.constexpr = 2.0**-24
        GAMMA_PV: tl.constexpr = 2 * K * U / (1 - 2 * K * U)
        DENOM_STEPS: tl.constexpr = triton.cdiv(K, 128) + 10
        GAMMA_DENOM: tl.constexpr = (
            DENOM_STEPS * 2.0**-23 / (1 - DENOM_STEPS * 2.0**-23)
        )
        # Two FP8 terms have a relative error plus a subnormal rounding floor.
        coefficient = (
            2.0**-9 + 1.25 * GAMMA_PV + GAMMA_DENOM + 4 * 2.0**-19 + 8 * U
        ) / (1 - GAMMA_DENOM)
        score_error = bound * (512 * U / (1 - 512 * U))
        floor_error = (
            448 * scale * exp_bound * 2.0**-10 * (1 + 4 * GAMMA_PV) / (1 - GAMMA_DENOM)
        )
        pv_error = (
            448 * kscale * (coefficient + tl.exp(4.1 * score_error) - 1) + floor_error
        )
        if K > 128:
            use_poly = tl.min(((qs >= 0) & (bound <= 2.0)).to(tl.int32), 0) != 0
            pv_error += tl.where(use_poly, 448 * kscale * BOUNDED_EXP_PV_BUDGET, 0.0)
        else:
            use_poly = False
        eligible = (
            tl.min(
                (
                    (score_error <= 1e-4)
                    & (pv_error < 0.0038)
                    & (bound < 20.0)
                    & (scale >= 2.0**-126)
                ).to(tl.int32),
                0,
            )
            != 0
        )
    tl.store(PScale + row * 64 + h, scale)
    fast_mode = tl.where(use_poly, FAST_BOUNDED_EXP, FAST_NATIVE_EXP)
    if PACKED_META:
        exponent_base = tl.load(Global + 1).to(tl.int32)
        format_tag = tl.where(exponent_base > 0, 4, 0)
        fast_mode = fast_mode | format_tag
    tl.store(Fast + row, tl.where(eligible, fast_mode, 0))
    if BadCount is not None and not eligible:
        index = tl.atomic_add(BadCount, 1)
        tl.store(BadRows + index, row)
    if KV is not None:
        tl.static_assert(K == 128 and Indices is not None)
        if not eligible:
            repair_k128(Q, QS, KV, KS, Indices, Output, LSE, row, TOKENS, SCALE)


# Split scheduling and denominator reduction


@triton.jit
def collect_v32_tail_denominator(Denominators, Stats, split, write=True):
    heads = tl.arange(0, 64)
    columns = tl.arange(0, 4)
    row = split * 64 + heads
    values = tl.load(Denominators + row[:, None] * 4 + columns[None, :])
    total = tl.sum(values, 1)
    bits = total.to(tl.uint32, bitcast=True)
    nonnegative_finite = (bits < 0x7F800000) | (bits == 0x80000000)
    valid = tl.min(nonnegative_finite.to(tl.int32), 0) != 0
    if valid and write:
        center = tl.where((bits & 0x7FFFFFFF) != 0, 0, 0xFF800000).to(tl.uint64)
        packet = center | (bits.to(tl.uint64) << 32)
        tl.store(Stats.to(tl.pointer_type(tl.uint64)) + row, packet)
    return valid


@triton.jit
def _mixed_v32_tail_span(
    tail_slot, SPLITS: tl.constexpr, CHUNK: tl.constexpr, FULL_ROWS: tl.constexpr
):
    row = FULL_ROWS + tail_slot // SPLITS
    start = (tail_slot % SPLITS) * CHUNK
    return row, start, start + CHUNK


@triton.jit
def mixed_v32_span(
    slot,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    FULL_ROWS: tl.constexpr,
):
    full = slot < FULL_ROWS
    row, start, end = _mixed_v32_tail_span(
        tl.maximum(slot - FULL_ROWS, 0), SPLITS, CHUNK, FULL_ROWS
    )
    return (
        tl.where(full, slot, row),
        full,
        tl.where(full, 0, start),
        tl.where(full, TOPK, end),
    )


def group_short_pages(
    rows: int, pages: int, programs: int
) -> Optional[Tuple[int, int, int, int]]:
    groups = math.gcd(rows, programs)
    group_rows, group_programs = rows // groups, programs // groups
    if group_rows == 1 or pages < group_programs:
        return None
    max_pages = triton.cdiv(group_rows * pages, group_programs)
    row_programs = triton.cdiv(pages, max_pages)
    row_slack = row_programs * max_pages - pages
    if row_slack == 0:
        return None
    # Use spare page capacity to align queries without lengthening the busiest CTA.
    spare_pages = group_programs * max_pages - group_rows * pages
    prefix_rows = min(group_rows - 1, spare_pages // row_slack)
    return (
        (group_rows, group_programs, prefix_rows, row_programs) if prefix_rows else None
    )


def prepare_persistent_combine(
    q,
    kv,
    qs,
    ks,
    indices,
    partial,
    stats,
    partial_flags,
    output,
    lse,
    scale,
    splits,
    programs,
    groups,
    repair,
    head_groups,
):

    rows, topk = q.shape[0], indices.shape[-1]
    tokens = kv.shape[0] * 64
    args = (
        partial,
        stats,
        partial_flags,
        output,
        lse,
        rows,
        topk,
        splits,
        programs,
        triton.next_power_of_2(splits * head_groups),
        groups,
        head_groups,
    )
    combine_heads, combine_warps = 16, 8
    if rows == 16 and topk >= 4096:
        combine_heads, combine_warps = 32, 16
    elif (rows <= 4 and topk >= 4096) or (rows == 4 and topk == 640):
        combine_heads, combine_warps = 8, 4
    options = dict(
        num_warps=combine_warps,
        num_stages=1,
        enable_backend_opt=True,
        enable_llc_opt=True,
    )
    grid = (rows, 64 // combine_heads, 1)
    combine_args = args + (
        combine_heads,
        None,
        q,
        kv,
        qs,
        ks,
        indices,
        tokens,
        scale,
        repair,
    )
    try:
        kernel = combine_persistent_sparse_fp8.warmup(
            *combine_args, grid=grid, **options
        )
        return kernel[grid], combine_args, None
    except (CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug(
            "Sparse FP8 fused combine unavailable: %s", error
        )
    flags = torch.empty((rows, 1), dtype=torch.int32, device=q.device)
    grid = (rows, 8, 1)
    options["num_warps"] = 8
    combine_args = args + (8, flags, None, None, None, None, None, 0, 0.0, None)
    kernel = combine_persistent_sparse_fp8.warmup(*combine_args, grid=grid, **options)
    return kernel[grid], combine_args, flags


# MODEL1 producer and consumer pipelines


# The mixed path reuses FP16 operands and crosses SQMMA accumulator layouts.
# Older FlagTree builds compile it but model1_produce incorrect values.
if HAS_MODEL1_PIPELINE:
    _MIXED_PV_SUPPORTED = tl.constexpr(
        all(
            getattr(musa_compiler, feature, lambda: False)()
            for feature in (
                "supports_tle_sqmma_m16_m_split",
                "supports_tle_barrier_memory_ordering",
            )
        )
    )

    PV = tl.constexpr(MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))
    OROW = tl.constexpr(
        tle.gpu.SlicedEncoding(1, MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))
    )
    OCOL = tl.constexpr(
        tle.gpu.SlicedEncoding(0, MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))
    )
else:
    _MIXED_PV_SUPPORTED = tl.constexpr(False)
    PV = OROW = OCOL = None


@triton.jit
def model1_consume(
    Output,
    PScale,
    ph,
    pl,
    sk,
    FinalInv,
    Full,
    PEmpty,
    NormReady,
    Active,
    KV,
    Locations,
    CopyReady,
    TOKENS: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    DOUBLE_P: tl.constexpr,
    CONSUMER_COPY: tl.constexpr,
    PACKED_META: tl.constexpr,
    MetaInfo,
    Selected,
):
    if Active:
        for meta_kind in tl.static_range(2 if PACKED_META else 1):
            if not PACKED_META or (meta_kind == 1) == ((MetaInfo & 1) != 0):
                row = tl.program_id(0).to(tl.int64)
                LOCATION_BYTES: tl.constexpr = (
                    (meta_kind == 1) if PACKED_META else USE_BYTE_OFFSETS
                )
                LOCATION_PACKED: tl.constexpr = PACKED_META and meta_kind == 1
                LocationPtr = Selected if LOCATION_PACKED else Locations
                if CONSUMER_COPY:
                    origin = tl.inline_asm_elementwise(
                        "",
                        constraints="=R,0",
                        args=[tl.full((), 128, tl.int32)],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                    copy_rows = tl.arange(128, 128 + BN) - origin
                    columns = tl.arange(0, 32) * 16
                    copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
                        [1, 1], [8, 4], [16, 1], [1, 0]
                    )
                    rr = tle.gpu.set_layout(
                        tl.broadcast_to(copy_rows[:, None], (BN, 32)), copy_layout
                    )
                    cc = tle.gpu.set_layout(
                        tl.broadcast_to(columns[None, :], (BN, 32)), copy_layout
                    )
                    sentinel = (
                        tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint32).to(tl.int32)
                        if LOCATION_BYTES
                        else TOKENS
                    )
                    tle.gpu.barrier_wait(PEmpty[0], phaseIdx=0)
                    init_location = tl.load(
                        LocationPtr + row * K + rr, rr < K, sentinel, volatile=True
                    )
                    init_source = kv_copy_address(
                        KV, init_location, cc, TOKENS, LOCATION_BYTES, LOCATION_PACKED
                    )
                    init_destination = (
                        tle.gpu.local_ptr(sk.slot(0), (rr, cc))
                        .to(tl.uint64)
                        .to(tl.uint32)
                    )
                    bounded_copy16(
                        init_destination,
                        init_source,
                        KV.to(tl.uint64),
                        tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint64),
                    )
                    bounded_copy_wait(tl.full((), 0, tl.int32))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(CopyReady[0], phaseIdx=0)
                accum = tle.gpu.set_layout(tl.full((64, 512), 0.0, tl.float32), PV)
                PAGES: tl.constexpr = triton.cdiv(K, BN)
                # Four pages use both slots at both barrier phases.
                PERIOD: tl.constexpr = 4 if CONSUMER_COPY and K == 8320 else 1
                for group in range(PAGES // PERIOD):
                    for step in tl.static_range(PERIOD):
                        page = group * PERIOD + step
                        phase_step = step if PERIOD == 4 else page
                        slot = phase_step & 1
                        if CONSUMER_COPY and page + 1 < tl.cdiv(K, BN):
                            next_slot = (phase_step + 1) & 1
                            next_phase = ((phase_step + 1) // 2) & 1
                            tle.gpu.barrier_wait(PEmpty[next_slot], phaseIdx=next_phase)
                            offsets = (page + 1) * BN + rr
                            next_location = tl.load(
                                LocationPtr + row * K + offsets,
                                offsets < K,
                                sentinel,
                                volatile=True,
                            )
                            next_source = kv_copy_address(
                                KV,
                                next_location,
                                cc,
                                TOKENS,
                                LOCATION_BYTES,
                                LOCATION_PACKED,
                            )
                            next_destination = (
                                tle.gpu.local_ptr(sk.slot(next_slot), (rr, cc))
                                .to(tl.uint64)
                                .to(tl.uint32)
                            )
                            bounded_copy16(
                                next_destination,
                                next_source,
                                KV.to(tl.uint64),
                                tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint64),
                            )
                            bounded_copy_wait(tl.full((), 0, tl.int32))
                            lma_wait(tl.full((), 0, tl.int32))
                            tle.gpu.barrier_arrive(
                                CopyReady[next_slot], phaseIdx=next_phase
                            )
                        if DOUBLE_P:
                            phase = (phase_step // 2) & 1
                            tle.gpu.barrier_wait(Full[slot], phaseIdx=phase)
                            high, low = ph.slot(slot), pl.slot(slot)
                        else:
                            tle.gpu.barrier_wait(Full, phaseIdx=slot)
                            high, low = ph, pl
                        values = sk.slot(slot)
                        accum = tle.gpu.wgmma(high, values, acc=accum)
                        accum = tle.gpu.wgmma(low, values, acc=accum)
                        accum = tle.gpu.wgmma_wait(0, accum)
                        if DOUBLE_P:
                            tle.gpu.barrier_arrive(PEmpty[slot], phaseIdx=phase)
                        else:
                            tle.gpu.barrier_arrive(PEmpty, phaseIdx=slot)
                if PERIOD == 4:
                    # Page 64 has no successor and starts a fresh barrier cycle.
                    tl.static_assert(PAGES == 65)
                    tle.gpu.barrier_wait(Full[0], phaseIdx=0)
                    accum = tle.gpu.wgmma(ph.slot(0), sk.slot(0), acc=accum)
                    accum = tle.gpu.wgmma(pl.slot(0), sk.slot(0), acc=accum)
                    accum = tle.gpu.wgmma_wait(0, accum)
                    tle.gpu.barrier_arrive(PEmpty[0], phaseIdx=0)
                oh = tl.inline_asm_elementwise(
                    "",
                    constraints="=R,0",
                    args=[tl.full((), 256, tl.int32)],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                od = tl.inline_asm_elementwise(
                    "",
                    constraints="=R,0",
                    args=[tl.full((), 1024, tl.int32)],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                h = tle.gpu.set_layout(tl.arange(256, 320) - oh, OROW)
                d = tle.gpu.set_layout(tl.arange(1024, 1536) - od, OCOL)
                tle.gpu.barrier_wait(NormReady, phaseIdx=0)
                inverse = tl.load(tle.gpu.local_ptr(FinalInv, (h,)), volatile=True)
                scale = tl.load(PScale + row * 64 + h, volatile=True)
                if K > 128 and K <= 1024:
                    # Keep the per-head factor normal at the lower certified scale.
                    factor = (scale * (2.0**32)) * inverse
                    result = (accum * factor[:, None]) * (2.0**-32)
                else:
                    result = accum * scale[:, None] * inverse[:, None]
                tl.store(Output + (row * 64 + h[:, None]) * 512 + d[None, :], result)


@triton.jit
def kv_copy_address(
    KV,
    location,
    dim,
    TOKENS: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    PACKED_META: tl.constexpr = False,
):
    if PACKED_META:
        location = (location.to(tl.uint32) & 0xFFFFFE00).to(tl.int32)
    if USE_BYTE_OFFSETS:
        offset = location.to(tl.uint32).to(tl.uint64)
        if TOKENS * KV_TOKEN_BYTES >= 2**31:
            # Pointer indexing can lower the high bit as a signed offset.
            return KV.to(tl.uint64) + offset + dim.to(tl.uint64)
    else:
        offset = location.to(tl.int64) * KV_TOKEN_BYTES
    return (KV + offset + dim).to(tl.uint64)


@triton.jit
def bounded_exp_from_half(half_score):
    # For a half score in [-1, 1], the squared result has relative error below 2.7e-5.
    value = tl.fma(half_score, 0.001439274288713932, 0.008686820976436138)
    value = tl.fma(half_score, value, 0.04163501039147377)
    value = tl.fma(half_score, value, 0.1664888709783554)
    value = tl.fma(half_score, value, 0.5000063180923462)
    value = tl.fma(half_score, value, 1.000022292137146)
    value = tl.fma(half_score, value, 0.9999998211860657)
    return value * value


@triton.jit
def model1_produce(
    Q,
    SharedQ,
    QReady,
    KV,
    QS,
    Locations,
    PScale,
    LSE,
    ph,
    pl,
    sk,
    FinalInv,
    Full,
    PEmpty,
    NormReady,
    Sync,
    SumParts,
    SumSync,
    CopySync,
    CopyReady,
    Selected,
    Active,
    UsePoly,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    DOUBLE_P: tl.constexpr,
    CONSUMER_COPY: tl.constexpr,
    PACKED_META: tl.constexpr,
    MetaInfo,
):
    if Active:
        row = tl.program_id(0).to(tl.int64)
        h = tl.arange(0, 64)
        col = tl.arange(0, BN)
        if DOUBLE_P:
            q = SharedQ
        else:
            q = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
        qs = tl.load(QS + row * 64 + h)
        ds = tl.load(PScale + row * 64 + h)
        reciprocal = tl.div_rn(1.0, ds)
        if CONSUMER_COPY:
            cd = tl.arange(0, 32) * 16
            copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [2, 16], [16, 1], [1, 0]
            )
        else:
            copy_origin = tl.inline_asm_elementwise(
                "",
                constraints="=R,0",
                args=[tl.full((), 128, tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            cr = tl.arange(128, 128 + BN) - copy_origin
            cd = tl.arange(0, 32) * 16
            copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [2, 16], [16, 1], [1, 0]
            )
            kv_lane_rows: tl.constexpr = 8 if K > 128 else 2
            kv_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [kv_lane_rows, 32 // kv_lane_rows], [16, 1], [1, 0]
            )
            crr = tle.gpu.set_layout(tl.broadcast_to(cr[:, None], (BN, 32)), kv_layout)
            cdd = tle.gpu.set_layout(tl.broadcast_to(cd[None, :], (BN, 32)), kv_layout)
        if isinstance(Q, tl.tensor_descriptor):
            tle.gpu.copy(
                Q, q, (64, 512), offsets=(row.to(tl.int32) * 64, 0), barrier=QReady
            )
        else:
            # Stage raw Q bytes without materializing the MMA-distributed tensor.
            q_origin = tl.inline_asm_elementwise(
                "",
                constraints="=R,0",
                args=[tl.full((), 4096, tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            q_rows = tl.arange(4096, 4160) - q_origin
            qrr = tle.gpu.set_layout(
                tl.broadcast_to(q_rows[:, None], (64, 32)), copy_layout
            )
            qcc = tle.gpu.set_layout(
                tl.broadcast_to(cd[None, :], (64, 32)), copy_layout
            )
            q_dst = tle.gpu.local_ptr(q, (qrr, qcc)).to(tl.uint64).to(tl.uint32)
            q_base = Q + row * 64 * 512
            q_src = q_base + qrr.to(tl.int64) * 512 + qcc
            bounded_copy16(
                q_dst,
                q_src.to(tl.uint64),
                q_base.to(tl.uint64),
                tl.full((), 64 * 512, tl.uint64),
            )
        if not CONSUMER_COPY:
            if USE_BYTE_OFFSETS:
                sentinel = tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint32).to(tl.int32)
            else:
                sentinel = TOKENS
            init_location = tl.load(
                Locations + row * K + crr, crr < K, sentinel, volatile=True
            )
            init_src = kv_copy_address(
                KV, init_location, cdd, TOKENS, USE_BYTE_OFFSETS, PACKED_META
            )
            init_dst0 = (
                tle.gpu.local_ptr(sk.slot(0), (crr, cdd)).to(tl.uint64).to(tl.uint32)
            )
            bounded_copy16(
                init_dst0,
                init_src,
                KV.to(tl.uint64),
                tl.full((), TOKENS * 512, tl.uint64),
            )
        bounded_copy_wait(tl.full((), 0, tl.int32))
        if isinstance(Q, tl.tensor_descriptor):
            tle.gpu.barrier_wait(QReady, phaseIdx=0)
        lma_wait(tl.full((), 0, tl.int32))
        tle.gpu.barrier_arrive(Sync, phaseIdx=0)
        tle.gpu.barrier_wait(Sync, phaseIdx=0)
        for meta_kind in tl.static_range(2 if PACKED_META else 1):
            if not PACKED_META or (meta_kind == 1) == ((MetaInfo & 1) != 0):
                # Select the exponent mode per query, outside its hot page loop.
                SPLIT_EXP: tl.constexpr = CONSUMER_COPY and K == 8320
                for mode in tl.static_range(2 if SPLIT_EXP else 1):
                    if not SPLIT_EXP or (mode == 1) == UsePoly:
                        if K > 1024:
                            total_matrix = tl.full((64, BN // 16, 8), 0.0, tl.float32)
                        else:
                            total_matrix = tl.full((64, BN), 0.0, tl.float32)
                        for page in range(tl.cdiv(K, BN)):
                            slot = page & 1
                            if CONSUMER_COPY:
                                tle.gpu.barrier_wait(
                                    CopyReady[slot], phaseIdx=(page // 2) & 1
                                )
                            offsets = page * BN + col
                            if PACKED_META:
                                word = tl.load(
                                    Selected + row * K + offsets, offsets < K, 0
                                ).to(tl.uint32)
                                if meta_kind == 1:
                                    ks = ((word & 255) << 23).to(
                                        tl.float32, bitcast=True
                                    )
                                else:
                                    ks = word.to(tl.float32, bitcast=True)
                            else:
                                ks = tl.load(
                                    Selected + row * K + offsets, offsets < K, 0.0
                                )
                            valid = (offsets < K) & (ks > 0.0)
                            dot = tle.gpu.wgmma(q, sk.slot(slot), trans_b=True)
                            dot = tle.gpu.wgmma_wait(0, dot).to(tl.float32)
                            if DOUBLE_P:
                                phase = (page // 2) & 1
                                high_tile, low_tile = ph.slot(slot), pl.slot(slot)
                            else:
                                tle.gpu.barrier_wait(PEmpty, phaseIdx=slot)
                                high_tile, low_tile = ph, pl
                            shifted = dot * qs[:, None] * ks[None, :] * SCALE
                            if K > 128 and (mode == 1 if SPLIT_EXP else UsePoly):
                                if K > 512:
                                    # Pair token bit 3 within each SQMMA thread's registers.
                                    pairs = tl.permute(
                                        tl.reshape(shifted * 0.5, (64, BN // 16, 2, 8)),
                                        (0, 1, 3, 2),
                                    )
                                    first, second = tl.split(pairs)
                                    packed = exp2_pair(first, second)
                                    low = packed.to(tl.uint32).to(
                                        tl.float32, bitcast=True
                                    )
                                    high = (
                                        (packed >> 32)
                                        .to(tl.uint32)
                                        .to(tl.float32, bitcast=True)
                                    )
                                    exp_value = tl.reshape(
                                        tl.permute(tl.join(low, high), (0, 1, 3, 2)),
                                        (64, BN),
                                    )
                                else:
                                    exp_value = bounded_exp_from_half(shifted * 0.5)
                            else:
                                exp_value = tl.exp(shifted)
                            probability = tl.where(valid[None, :], exp_value, 0.0)
                            if K > 1024:
                                # Token bit 3 is register-local; the summation depth is unchanged.
                                sum_pairs = tl.permute(
                                    tl.reshape(probability, (64, BN // 16, 2, 8)),
                                    (0, 1, 3, 2),
                                )
                                sum_first, sum_second = tl.split(sum_pairs)
                                total_matrix += sum_first + sum_second
                            else:
                                total_matrix += probability
                            normalized = probability * ks[None, :] * reciprocal[:, None]
                            high = normalized.to(tl.float8e4nv)
                            low = (normalized - high.to(tl.float32)).to(tl.float8e4nv)
                            phh = tl.broadcast_to(h[:, None], (64, BN))
                            phc = tl.broadcast_to(col[None, :], (64, BN))
                            tl.store(tle.gpu.local_ptr(high_tile, (phh, phc)), high)
                            tl.store(tle.gpu.local_ptr(low_tile, (phh, phc)), low)
                            lma_wait(tl.full((), 0, tl.int32))
                            # Full already orders all producer writes before the consumer's PV.
                            if DOUBLE_P:
                                tle.gpu.barrier_arrive(Full[slot], phaseIdx=phase)
                            else:
                                tle.gpu.barrier_arrive(Full, phaseIdx=slot)
                            if page + 1 == tl.cdiv(K, BN):
                                if K > 1024:
                                    partial = tl.sum(
                                        tl.reshape(total_matrix, (64, 4, 16)), 2
                                    )
                                else:
                                    partial = tl.sum(
                                        tl.reshape(total_matrix, (64, 4, 32)), 2
                                    )
                                # Preserve the partials' layout to avoid conversion scratch.
                                tl.store(tle.gpu.local_ptr(SumParts), partial)
                                lma_wait(tl.full((), 0, tl.int32))
                                # This barrier is used once, regardless of the number of tiles.
                                tle.gpu.barrier_arrive(SumSync, phaseIdx=0)
                                tle.gpu.barrier_wait(SumSync, phaseIdx=0)
                                s0 = tl.load(
                                    tle.gpu.local_ptr(
                                        SumParts, (h, tl.full((64,), 0, tl.int32))
                                    ),
                                    volatile=True,
                                )
                                s1 = tl.load(
                                    tle.gpu.local_ptr(
                                        SumParts, (h, tl.full((64,), 1, tl.int32))
                                    ),
                                    volatile=True,
                                )
                                s2 = tl.load(
                                    tle.gpu.local_ptr(
                                        SumParts, (h, tl.full((64,), 2, tl.int32))
                                    ),
                                    volatile=True,
                                )
                                s3 = tl.load(
                                    tle.gpu.local_ptr(
                                        SumParts, (h, tl.full((64,), 3, tl.int32))
                                    ),
                                    volatile=True,
                                )
                                total = (s0 + s1) + (s2 + s3)
                                inverse = tl.where(
                                    total > 0,
                                    1.0 / tl.where(total > 0, total, 1.0),
                                    0.0,
                                )
                                tl.store(tle.gpu.local_ptr(FinalInv, (h,)), inverse)
                                lse = tl.where(total > 0, tl.log(total), float("inf"))
                                tl.store(LSE + row * 64 + h, lse)
                                lma_wait(tl.full((), 0, tl.int32))
                                tle.gpu.barrier_arrive(NormReady, phaseIdx=0)
                            if not CONSUMER_COPY and page + 1 < tl.cdiv(K, BN):
                                ns = (page + 1) & 1
                                copy_offsets = (page + 1) * BN + crr
                                copy_location = tl.load(
                                    Locations + row * K + copy_offsets,
                                    copy_offsets < K,
                                    sentinel,
                                    volatile=True,
                                )
                                copy_src = kv_copy_address(
                                    KV,
                                    copy_location,
                                    cdd,
                                    TOKENS,
                                    USE_BYTE_OFFSETS,
                                    PACKED_META,
                                )
                                copy_dst0 = (
                                    tle.gpu.local_ptr(sk.slot(ns), (crr, cdd))
                                    .to(tl.uint64)
                                    .to(tl.uint32)
                                )
                                if DOUBLE_P:
                                    # Wait once before overwriting the KV and later P in this slot.
                                    tle.gpu.barrier_wait(
                                        PEmpty[ns], phaseIdx=((page + 1) // 2) & 1
                                    )
                                bounded_copy16(
                                    copy_dst0,
                                    copy_src,
                                    KV.to(tl.uint64),
                                    tl.full((), TOKENS * 512, tl.uint64),
                                )
                                bounded_copy_wait(tl.full((), 0, tl.int32))
                                lma_wait(tl.full((), 0, tl.int32))
                                tle.gpu.barrier_arrive(CopySync, phaseIdx=page & 1)
                                tle.gpu.barrier_wait(CopySync, phaseIdx=page & 1)


@triton.jit
def fused_fp8_guarded(
    Q,
    KV,
    QS,
    Locations,
    PScale,
    Fast,
    Output,
    LSE,
    Selected,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    DOUBLE_P: tl.constexpr,
    PACKED_META: tl.constexpr,
):
    BN: tl.constexpr = 128
    consumer_copy: tl.constexpr = DOUBLE_P and K > 1024 and _MIXED_PV_SUPPORTED
    sk = tle.gpu.alloc((2, BN, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    if DOUBLE_P:
        ph = tle.gpu.alloc((2, 64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
        pl = tle.gpu.alloc((2, 64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
        q = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
        # The final normalization uses this storage only after the last QK wait.
        inverse = tle.gpu.alloc((64,), tl.float32, alias=q, nv_mma_shared_layout=False)
        sp = tle.gpu.alloc(
            (64, 4),
            tl.float32,
            alias=q,
            alias_offset_bytes=256,
            nv_mma_shared_layout=False,
        )
        full = tle.gpu.alloc_barriers(2, arrive_count=16)
        pe = tle.gpu.alloc_barriers(2, arrive_count=16, init=tle.gpu.READY)
    else:
        ph = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
        pl = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
        # Only the double-buffered specialization uses this argument.
        q = sk
        inverse = tle.gpu.alloc((64,), tl.float32, nv_mma_shared_layout=False)
        sp = tle.gpu.alloc((64, 4), tl.float32, nv_mma_shared_layout=False)
        full = tle.gpu.alloc_barrier(arrive_count=16)
        pe = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.READY)
    q_ready = (
        tle.gpu.alloc_barrier(arrive_count=1, expect_bytes=64 * 512)
        if isinstance(Q, tl.tensor_descriptor)
        else full
    )
    norm_ready = tle.gpu.alloc_barrier(arrive_count=16)
    sync = tle.gpu.alloc_barrier(arrive_count=16)
    sb = tle.gpu.alloc_barrier(arrive_count=16)
    if consumer_copy:
        copy_ready = tle.gpu.alloc_barriers(2, arrive_count=16)
        cs = sync
    else:
        copy_ready = full
        cs = tle.gpu.alloc_barrier(arrive_count=16)
    mode = tl.load(Fast + tl.program_id(0), volatile=True)
    active = mode != 0
    use_poly = ((mode & 3) if PACKED_META else mode) == FAST_BOUNDED_EXP
    meta_info = mode >> 2 if PACKED_META else 0
    tl.debug_barrier()
    tle.gpu.warp_specialize(
        [
            (
                model1_produce,
                (
                    Q,
                    q,
                    q_ready,
                    KV,
                    QS,
                    Locations,
                    PScale,
                    LSE,
                    ph,
                    pl,
                    sk,
                    inverse,
                    full,
                    pe,
                    norm_ready,
                    sync,
                    sp,
                    sb,
                    cs,
                    copy_ready,
                    Selected,
                    active,
                    use_poly,
                    K,
                    TOKENS,
                    BN,
                    SCALE,
                    USE_BYTE_OFFSETS,
                    DOUBLE_P,
                    consumer_copy,
                    PACKED_META,
                    meta_info,
                ),
            ),
            (
                model1_consume,
                (
                    Output,
                    PScale,
                    ph,
                    pl,
                    sk,
                    inverse,
                    full,
                    pe,
                    norm_ready,
                    active,
                    KV,
                    Locations,
                    copy_ready,
                    TOKENS,
                    USE_BYTE_OFFSETS,
                    K,
                    BN,
                    DOUBLE_P,
                    consumer_copy,
                    PACKED_META,
                    meta_info,
                    Selected,
                ),
            ),
        ],
        worker_num_warps=[16],
        worker_num_regs=[128],
    )


@lru_cache(maxsize=64)
def compiled_pipeline(
    topk,
    tokens,
    scale,
    output_dtype,
    device_index,
    target,
    use_byte_offsets,
    use_tme_q=False,
    packed_meta=False,
):
    """Cache compiled code by metadata; never retain caller Tensor storage."""
    fn = fused_fp8_guarded
    signature = dict(
        zip(
            fn.arg_names[:9],
            [
                "*fp8e4nv",
                "*fp8e4nv",
                "*fp32",
                "*i32",
                "*fp32",
                "*i32",
                output_dtype,
                "*fp32",
                "*fp32",
            ],
        )
    )
    if packed_meta:
        signature["Selected"] = "*i32"
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(9)}
    if use_tme_q:
        signature["Q"] = "tensordesc<fp8e4nv[64,512]>"
        attrs[(0,)] = [["musa.tme_tail_divisibility", 4]]
    constants = dict(
        K=topk,
        TOKENS=tokens,
        SCALE=scale,
        USE_BYTE_OFFSETS=use_byte_offsets,
        PACKED_META=packed_meta,
    )
    library = native_library()
    buffering = (
        (True,)
        if (use_tme_q or packed_meta)
        else ((True, False) if topk > 128 else (False,))
    )
    for double_p in buffering:
        try:
            return triton.compile(
                ASTSource(
                    fn,
                    signature,
                    dict(constants, DOUBLE_P=double_p),
                    attrs=attrs,
                ),
                target=target,
                options=dict(
                    num_warps=16,
                    num_stages=1,
                    maxnreg=128,
                    enable_backend_opt=True,
                    enable_llc_opt=True,
                    extern_libs={"native": library},
                ),
            )
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "Sparse FP8 pipeline compile failed (double_p=%s): %s", double_p, error
            )
    return None


def prepare_model1_pipeline(q, kv, qs, ks, indices, output, lse, scale):
    rows = q.shape[0]
    topk = indices.shape[-1]
    tokens = kv.shape[0] * 64
    # Only selected tokens contribute to QK/PV; their maxima give the same
    # error guarantees without scanning a larger, mostly unused cache.
    selected_count = rows * topk if rows * topk < tokens else 0
    parts = triton.cdiv(selected_count or tokens, 1024)
    selected_bytes = 4 * rows * topk
    workspace_bytes = selected_bytes + rows * (64 * 4 + 8) + 4 * parts + 8
    workspace_limit = 8 * 1024**3
    if workspace_bytes > workspace_limit:
        return None
    if any(t.data_ptr() % 16 for t in (q, kv, qs, ks, indices, output, lse)):
        return None
    target = driver.active.get_current_target()
    dtype = "*bf16" if output.dtype == torch.bfloat16 else "*fp16"
    use_byte_offsets = (
        tokens < KV_OFFSET_TOKEN_LIMIT
        and workspace_bytes + selected_bytes <= workspace_limit
    )
    packed_meta = (
        topk == 8320
        and rows >= 128
        and tokens < PACKED_METADATA_TOKEN_LIMIT
        and use_byte_offsets
        and _MIXED_PV_SUPPORTED.value
        and driver.active.utils.get_device_properties(
            driver.active.get_current_device()
        )["max_shared_mem"]
        >= 192 * 1024
    )
    compile_args = (topk, tokens, scale, dtype, q.device.index, target)
    # Large K640 queries amortize descriptor expansion in the MUSA launcher.
    use_tme_q = topk == 640 and rows >= 4096 and _MIXED_PV_SUPPORTED.value
    programs, kernel = 0, None
    if topk == 128 and rows >= 128 and _MIXED_PV_SUPPORTED.value:

        properties = driver.active.utils.get_device_properties(
            driver.active.get_current_device()
        )
        programs = min(rows, properties["multiprocessor_count"])
        use_tme_q = properties["max_shared_mem"] >= 192 * 1024
        kernel = compiled_kernel(*compile_args, use_byte_offsets, programs, use_tme_q)
        if use_tme_q and kernel is None:
            use_tme_q = False
            kernel = compiled_kernel(*compile_args, use_byte_offsets, programs)
        if kernel is not None and kernel.metadata.shared > properties["max_shared_mem"]:
            kernel = None
    if kernel is None:
        if programs:
            use_tme_q = False
        programs = 0
        kernel = compiled_pipeline(
            *compile_args, use_byte_offsets, use_tme_q, packed_meta
        )
    if use_tme_q and kernel is None:
        use_tme_q = False
        kernel = compiled_pipeline(*compile_args, use_byte_offsets)
    if kernel is None:
        return None
    probability_scale = torch.empty((rows, 64), dtype=torch.float32, device=q.device)
    flags = torch.empty((rows,), dtype=torch.int32, device=q.device)
    bad_rows = torch.empty_like(flags)
    bad_count = torch.empty((1,), dtype=torch.int32, device=q.device)
    selected = torch.empty(
        (rows, topk),
        dtype=torch.int32 if packed_meta else torch.float32,
        device=q.device,
    )
    byte_offsets = selected if packed_meta else None
    cache_bits = None
    if packed_meta:
        try:
            cache_bits = torch.empty(
                (triton.cdiv(tokens, 32),), dtype=torch.int32, device=q.device
            )
        except torch.OutOfMemoryError:
            packed_meta = False
            use_byte_offsets = False
            byte_offsets = None
            selected = selected.view(torch.float32)
            kernel = compiled_pipeline(*compile_args, False, False, False)
            if kernel is None:
                return None
    if use_byte_offsets and not packed_meta:
        try:
            byte_offsets = torch.empty_like(indices)
        except torch.OutOfMemoryError:
            # Offset staging is optional when the caller has limited device memory.
            use_tme_q = False
            if programs:
                kernel = compiled_kernel(*compile_args, False, programs)
            if not programs or kernel is None:
                programs = 0
                kernel = compiled_pipeline(*compile_args, False)
            if kernel is None:
                return None
    a2 = (
        q,
        qs,
        None,
        probability_scale,
        flags,
        bad_rows,
        bad_count,
        topk,
        scale,
        topk == 640 and rows >= 4096,
    )
    bounds_options = dict(
        num_warps=16,
        grid=(rows,),
        extern_libs={"native": native_library(QUERY_NORM_IR)},
    )
    k2 = None
    repair_in_bounds = False
    no_repair = (None, None, None)
    if (programs and use_tme_q) or (
        topk == 640 and rows >= 128 and _MIXED_PV_SUPPORTED.value
    ):
        active_query_args = (*a2[:5], None, None, topk, scale)
        row_data = (indices, ks, selected, byte_offsets)
        row_args = (*active_query_args, True, *row_data, tokens, False)
        row_options = dict(bounds_options, num_warps=8) if programs else bounds_options
        repair_buffers = ((kv, output, lse), no_repair) if programs else (no_repair,)
        for buffers in repair_buffers:
            fused_args = (*row_args, *buffers)
            try:
                k2 = query_bounds_global.warmup(*fused_args, **row_options)
            except (CompilationError, OutOfResources, RuntimeError) as error:
                logging.getLogger(__name__).debug(
                    "Sparse FP8 row bounds failed: %s", error
                )
            else:
                a2 = fused_args
                repair_in_bounds = buffers[0] is not None
                break
    row_scales = k2 is not None
    if not row_scales:
        scales = torch.empty(
            (parts * (3 if packed_meta else 1),), dtype=torch.float32, device=q.device
        )
        global_scale = torch.empty(
            (2 if packed_meta else 1,), dtype=torch.float32, device=q.device
        )
        a0 = (
            ks,
            indices,
            selected,
            byte_offsets,
            scales,
            bad_count,
            tokens,
            selected_count,
            1024,
            packed_meta,
            cache_bits,
        )
        a1 = (scales, global_scale, parts, triton.next_power_of_2(parts), packed_meta)
        r0 = cache_scale_bounds.warmup(*a0, num_warps=4, grid=(parts,))[(parts, 1, 1)]
        r1 = reduce_scales.warmup(*a1, num_warps=4, grid=(1,))[(1, 1, 1)]
        a2 = (
            *a2[:2],
            global_scale,
            *a2[3:],
            None,
            None,
            None,
            None,
            0,
            packed_meta,
            *no_repair,
        )
        k2 = query_bounds_global.warmup(*a2, **bounds_options)
    a3 = (
        indices,
        ks,
        selected,
        byte_offsets,
        rows,
        topk,
        tokens,
        1024,
        packed_meta,
        global_scale if packed_meta else None,
        cache_bits,
    )
    g3 = triton.cdiv(rows * topk, 1024)
    if selected_count or row_scales:
        r3 = None
    else:
        k3 = gather_scales.warmup(*a3, num_warps=4, grid=(g3,))
        r3 = k3[(g3, 1, 1)]
    r2 = k2[(rows, 1, 1)]
    r4 = kernel[(programs or rows, 1, 1)]
    locations = indices if packed_meta or byte_offsets is None else byte_offsets
    query_arg = (
        TensorDescriptor.from_tensor(q.view(-1, 512), [64, 512]) if use_tme_q else q
    )
    a4 = (query_arg, kv, qs, locations, probability_scale, flags, output, lse, selected)
    if programs:
        a4 += (rows,)

    def run_pipeline():
        if not row_scales:
            r0(*a0)
            r1(*a1)
        r2(*a2)
        if r3 is not None:
            r3(*a3)
        r4(*a4)

    if repair_in_bounds:
        return run_pipeline, None, None
    if row_scales:
        return run_pipeline, flags, None
    return run_pipeline, bad_rows, bad_count


def prepare_short_pipeline(q, kv, qs, ks, indices, output, lse, scale, combine, repair):
    rows, topk = q.shape[0], indices.shape[-1]
    if any(t.data_ptr() % 16 for t in (q, kv, qs, ks, indices, output, lse)):
        return None
    fixed_pv = _MIXED_PV_SUPPORTED.value and topk >= 256
    if (
        fixed_pv
        and rows == 16
        and topk == 8320
        and kv.shape[0] * 64 < KV_OFFSET_TOKEN_LIMIT
        and 2.0**-125 <= scale <= 2.0**16
    ):

        staged = prepare_staged_pipeline(
            q, kv, qs, ks, indices, output, lse, scale, repair
        )
        if staged is not None:
            return staged
    pages = triton.cdiv(topk, 128)
    direct = _MIXED_PV_SUPPORTED.value and 64 < topk <= 128
    persistent = direct or (fixed_pv and topk >= 512)
    single_split = persistent and rows == 16 and topk == 640
    can_prefetch = False
    if direct:
        if topk == 128 and rows < 16:
            heads, warps = 16, 8
        else:
            heads, warps = 32, 16 if topk == 128 else 8
        splits = 1
        programs, groups = rows, None
    elif persistent:
        small_grid = topk <= 1024 and rows * pages < 40
        # Keep all K=640 pages in one CTA when query parallelism is sufficient.
        if topk == 640 and (rows <= 4 or single_split):
            heads, warps = 32, 16
        else:
            heads, warps = (32, 8) if small_grid else (64, 16)
        properties = driver.active.utils.get_device_properties(
            driver.active.get_current_device()
        )
        can_prefetch = (
            single_split
            and properties["max_shared_mem"] >= 192 * 1024
            and kv.shape[0] * 64 < KV_OFFSET_TOKEN_LIMIT
        )
        programs = min(properties["multiprocessor_count"], rows * pages)
        groups = group_short_pages(rows, pages, programs) if topk > 1024 else None
        if single_split:
            programs, splits, groups = rows, 1, None
        elif topk <= 1024 and 2 * rows <= programs < rows * pages:
            # Keep short fragments inside one query to avoid extra Q staging.
            splits = programs // rows
            programs = rows * splits
        elif groups is not None:
            group_rows, group_programs, prefix_rows, row_programs = groups
            tail_rows = group_rows - prefix_rows
            tail_programs = group_programs - prefix_rows * row_programs
            splits = max(row_programs, triton.cdiv(tail_programs, tail_rows) + 1)
        elif small_grid and programs % rows == 0 and programs > rows:
            splits = programs // rows
        else:
            splits = triton.cdiv(programs, rows) + 1
    elif fixed_pv:
        heads, warps = 32, 8
        splits = min(8, triton.cdiv(topk, 64), triton.cdiv(64, rows * 2))
    else:
        heads = 16 if topk <= 1024 else 64
        warps = 4 if heads == 16 else 16
        splits = (
            1
            if topk <= 1024
            else min(32, triton.cdiv(topk, 64), triton.cdiv(256, rows))
        )
    partial_flags = torch.empty(
        (rows, splits * (64 // heads)), dtype=torch.int32, device=q.device
    )
    if splits == 1:
        partial, stats = output, lse
    else:
        partial = torch.empty(
            (rows, splits, 64, 512), dtype=torch.float32, device=q.device
        )
        stats = torch.empty((rows, splits, 64, 2), dtype=torch.float32, device=q.device)
    arguments = (
        q,
        kv,
        qs,
        ks,
        indices,
        partial,
        stats,
        partial_flags,
        topk,
        kv.shape[0] * 64,
        scale,
        heads,
        splits,
    )
    vector_log2 = can_prefetch or (persistent and rows == 4 and topk == 640)
    library = native_library(VECTOR_EXP2_IR) if vector_log2 else native_library()
    if persistent:
        arguments = arguments[:8] + (
            rows,
            topk,
            kv.shape[0] * 64,
            scale,
            programs,
            splits,
            groups,
            heads,
            warps,
        )
        kernel_function = persistent_sparse_fp8
        grid = (programs, 64 // heads, 1)
    else:
        kernel_function = fixed_scale_sparse_fp8 if fixed_pv else short_sparse_fp8
        grid = (rows, 64 // heads, splits)
    # The 16-head K128 tile needs a separate launch for deterministic FP32 repair.
    repairs = (repair, None) if (direct and heads == 32) or single_split else (None,)
    for direct_repair in repairs:
        kernel_args = (
            arguments + (direct_repair, can_prefetch) if persistent else arguments
        )
        try:
            kernel = kernel_function.warmup(
                *kernel_args,
                num_warps=warps,
                num_stages=1,
                enable_backend_opt=persistent or fixed_pv or topk > 1024,
                enable_llc_opt=persistent or fixed_pv or topk > 1024,
                extern_libs={"native": library},
                grid=grid,
            )
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "Sparse FP8 short path compile failed (fused_repair=%s): %s",
                direct_repair is not None,
                error,
            )
        else:
            break
    else:
        return None
    runner = kernel[grid]
    if persistent and splits > 1:
        combine_runner, combine_args, flags = prepare_persistent_combine(
            q,
            kv,
            qs,
            ks,
            indices,
            partial,
            stats,
            partial_flags,
            output,
            lse,
            scale,
            splits,
            programs,
            groups,
            repair,
            64 // heads,
        )
    elif splits > 1:
        flags = partial_flags
        combine_args = (
            partial,
            stats,
            None,
            output,
            lse,
            rows,
            64,
            1,
            splits,
            False,
            triton.next_power_of_2(splits),
            512,
        )
        combined = combine.warmup(
            *combine_args, num_warps=4, num_stages=1, grid=(rows, 64, 1)
        )
        combine_runner = combined[(rows, 64, 1)]
    else:
        flags = None if direct_repair is not None else partial_flags

    def run_short():
        runner(*kernel_args)
        if splits > 1:
            combine_runner(*combine_args)

    return run_short, flags


@triton.jit
def short_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Output,
    LSE,
    Flags,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(2)
    begin = tl.cdiv(K, 64) * split // SPLITS
    end = tl.cdiv(K, 64) * (split + 1) // SPLITS
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    col = tl.arange(0, 64)
    qv = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :])
    qs = tl.load(QS + row * 64 + h)
    q_l1 = tl.sum(tl.abs(qv.to(tl.float32)), 1) * qs
    q = tle.gpu.alloc(
        (BH, 512), tl.float8e4nv, init_value=qv, nv_mma_shared_layout=True
    )
    sk = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((BH, 64), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((BH, 64), tl.float8e4nv, nv_mma_shared_layout=True)
    lma_wait(tl.full((), 0, tl.int32))
    tl.debug_barrier()
    cr = tl.arange(128, 192) - 128
    cd = tl.arange(1024, 1056) * 16 - 16384
    crr = tl.broadcast_to(cr[:, None], (64, 32))
    cdd = tl.broadcast_to(cd[None, :], (64, 32))
    dst = tle.gpu.local_ptr(sk, (crr, cdd)).to(tl.uint64).to(tl.uint32)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.full((BH,), 0.0, tl.float32)
    running = tl.full((BH,), 0.0, tl.float32)
    accum = tl.full((BH, 512), 0.0, tl.float32)
    needs_repair = tl.full((), 0, tl.int32)
    for page in range(begin, end):
        ids_copy = tl.load(
            Indices + row * K + page * 64 + crr, page * 64 + crr < K, TOKENS
        )
        src = KV + ids_copy.to(tl.int64) * 512 + cdd
        bounded_copy16(
            dst,
            src.to(tl.uint64),
            KV.to(tl.uint64),
            tl.full((), TOKENS * 512, tl.uint64),
        )
        bounded_copy_wait(tl.full((), 0, tl.int32))
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        dot = tle.gpu.wgmma(q, sk, trans_b=True)
        dot = tle.gpu.wgmma_wait(0, dot)
        ids = tl.load(Indices + row * K + page * 64 + col, page * 64 + col < K, -1)
        valid = (page * 64 + col < K) & (ids >= 0) & (ids < TOKENS)
        ks = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        error = (
            q_l1
            * (448.0 * tl.max(ks, 0))
            * SCALE
            * ((512.0 * 2.0**-24) / (1.0 - 512.0 * 2.0**-24))
        )
        needs_repair = tl.maximum(
            needs_repair, (tl.max(error, 0) > 1.0e-4).to(tl.int32)
        )
        scores = tl.where(
            valid[None, :],
            dot * qs[:, None] * ks[None, :] * (SCALE * 1.4426950408889634),
            -float("inf"),
        )
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(new_max == -float("inf"), 0.0, new_max)
        p = tl.exp2(scores - safe[:, None])
        correction = tl.exp2(maximum - safe)
        denominator = denominator * correction + tl.sum(p, 1)
        weights = p * ks[None, :]
        ds = tl.maximum(tl.max(weights, 1) * (1.0 / 448.0), 1.0e-30)
        inv = 1.0 / ds
        norm = weights * inv[:, None]
        high = norm.to(tl.float8e4nv)
        low = ((norm - high.to(tl.float32)) * 16.0).to(tl.float8e4nv)
        tl.store(tle.gpu.local_ptr(ph), high)
        tl.store(tle.gpu.local_ptr(pl), low)
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        accum = tle.gpu.wgmma(ph, sk, acc=accum * (running * correction * inv)[:, None])
        accum = tle.gpu.wgmma_wait(0, accum)
        accum = tle.gpu.wgmma(pl, sk, acc=accum * 16.0)
        accum = tle.gpu.wgmma_wait(0, accum) * (1.0 / 16.0)
        tl.debug_barrier()
        running = ds
        maximum = new_max
    tl.store(
        Flags + (row * SPLITS + split) * (64 // BH) + tl.program_id(1), needs_repair
    )
    factor = running
    if SPLITS == 1:
        factor *= tl.div_rn(1.0, tl.where(denominator > 0, denominator, 1.0))
    base = (row * SPLITS + split) * 64 + h
    tl.store(Output + base[:, None] * 512 + d[None, :], accum * factor[:, None])
    logsum = maximum * 0.6931471805599453 + tl.log(
        tl.where(denominator > 0, denominator, 1.0)
    )
    if SPLITS == 1:
        tl.store(LSE + row * 64 + h, tl.where(denominator > 0, logsum, float("inf")))
    else:
        tl.store(LSE + base * 2, maximum * 0.6931471805599453)
        tl.store(LSE + base * 2 + 1, denominator)


@triton.jit
def fixed_scale_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Output,
    LSE,
    Flags,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    SPLITS: tl.constexpr,
):
    BN: tl.constexpr = 64
    V_TILE: tl.constexpr = 256
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(2)
    begin = tl.cdiv(K, BN) * split // SPLITS
    end = tl.cdiv(K, BN) * (split + 1) // SPLITS
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    col = tl.arange(0, BN)
    qv = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :])
    qs = tl.load(QS + row * 64 + h)
    q_l1 = tl.sum(tl.abs(qv.to(tl.float32)), 1) * qs
    q = tle.gpu.alloc(
        (BH, 512), tl.float8e4nv, init_value=qv, nv_mma_shared_layout=True
    )
    sk = tle.gpu.alloc((BN, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((BH, BN), tl.float16, nv_mma_shared_layout=True)
    value = tle.gpu.alloc((BN, V_TILE), tl.float16, nv_mma_shared_layout=True)
    lma_wait(tl.full((), 0, tl.int32))
    tl.debug_barrier()
    cr = tl.arange(128, 128 + BN) - 128
    cd = tl.arange(1024, 1056) * 16 - 16384
    crr = tl.broadcast_to(cr[:, None], (BN, 32))
    cdd = tl.broadcast_to(cd[None, :], (BN, 32))
    dst = tle.gpu.local_ptr(sk, (crr, cdd)).to(tl.uint64).to(tl.uint32)
    denominator = tl.full((BH,), 0.0, tl.float32)
    running = tl.full((BH,), 0.0, tl.float32)
    if V_TILE == 256:
        accum0 = tl.full((BH, 256), 0.0, tl.float32)
        accum1 = tl.full((BH, 256), 0.0, tl.float32)
    else:
        accum = tl.full((BH, 512), 0.0, tl.float32)
    needs_repair = tl.full((), 0, tl.int32)
    PRE: tl.constexpr = triton.next_power_of_2(
        triton.cdiv(triton.cdiv(K, BN), SPLITS) * BN
    )
    pre_offsets = begin * BN + tl.arange(0, PRE)
    pre_ids = tl.load(
        Indices + row * K + pre_offsets,
        (pre_offsets < K) & (pre_offsets < end * BN),
        -1,
    )
    pre_valid = (
        (pre_offsets < K)
        & (pre_offsets < end * BN)
        & (pre_ids >= 0)
        & (pre_ids < TOKENS)
    )
    pre_scales = tl.load(KS + tl.where(pre_valid, pre_ids, 0), pre_valid, 0.0)
    fixed_scale = tl.maximum(tl.max(pre_scales, 0), 1e-30)
    minimum_scale = tl.min(tl.where(pre_valid, pre_scales, float("inf")), 0)
    original_score_error = (
        q_l1
        * (448.0 * tl.max(pre_scales, 0))
        * SCALE
        * (512.0 * 2.0 ** (-24) / (1.0 - 512.0 * 2.0 ** (-24)))
    )
    PART_TOKENS: tl.constexpr = triton.cdiv(triton.cdiv(K, BN), SPLITS) * BN
    ROUND_UNIT: tl.constexpr = 2.0 ** (-23)
    SUM_GAMMA: tl.constexpr = (
        PART_TOKENS * ROUND_UNIT / (1.0 - PART_TOKENS * ROUND_UNIT)
    )
    normalization_error = (
        2.0 ** (-11) + 2.0 * SUM_GAMMA + 2.0 ** (-19) + 4.0 * SPLITS * ROUND_UNIT
    ) / (1.0 - SUM_GAMMA)
    score_sensitivity = tl.exp(4.1 * original_score_error) - 1.0
    pv_error_bound = 448.0 * fixed_scale * (normalization_error + score_sensitivity)
    eligible = (
        tl.min(
            ((original_score_error <= 0.0001) & (pv_error_bound < 0.0018)).to(tl.int32),
            0,
        )
        != 0
    )
    eligible = eligible & (minimum_scale >= fixed_scale * (1.0 / 256.0))
    eligible = eligible & (0.5 * tl.max(pre_scales, 0) <= 0.0019)
    needs_repair = (~eligible).to(tl.int32)
    for page in range(begin, end):
        ids_copy = tl.load(
            Indices + row * K + page * BN + crr, page * BN + crr < K, TOKENS
        )
        src = KV + ids_copy.to(tl.int64) * 512 + cdd
        bounded_copy16(
            dst,
            src.to(tl.uint64),
            KV.to(tl.uint64),
            tl.full((), TOKENS * 512, tl.uint64),
        )
        bounded_copy_wait(tl.full((), 0, tl.int32))
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        dot = tle.gpu.wgmma(q, sk, trans_b=True)
        dot = tle.gpu.wgmma_wait(0, dot)
        ids = tl.load(Indices + row * K + page * BN + col, page * BN + col < K, -1)
        valid = (page * BN + col < K) & (ids >= 0) & (ids < TOKENS)
        ks = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        scores = tl.where(
            valid[None, :],
            dot * qs[:, None] * ks[None, :] * (SCALE * 1.4426950408889634),
            -float("inf"),
        )
        p = tl.where(valid[None, :], tl.exp2(tl.where(eligible, scores, 0.0)), 0.0)
        denominator += tl.sum(p, 1)
        ds = fixed_scale + tl.zeros((BH,), tl.float32)
        inv = 1.0 / ds
        tl.store(
            tle.gpu.local_ptr(ph), (p * (ks[None, :] * inv[:, None])).to(tl.float16)
        )
        vr = tl.arange(256, 256 + BN) - 256
        vc = tl.arange(4096, 4096 + V_TILE // 4) * 4 - 16384
        vrr = tl.broadcast_to(vr[:, None], (BN, V_TILE // 4))
        vcc = tl.broadcast_to(vc[None, :], (BN, V_TILE // 4))
        for half in tl.static_range(512 // V_TILE):
            raw = tl.load(
                tle.gpu.local_ptr(sk, (vrr, vcc + half * V_TILE)).to(
                    tl.pointer_type(tl.uint32, 3)
                ),
                volatile=True,
            )
            tl.store(
                tle.gpu.local_ptr(value, (vrr, vcc)).to(tl.pointer_type(tl.uint64, 3)),
                promote_fp8_four(raw),
            )
            lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
            if V_TILE == 512:
                accum = tle.gpu.wgmma(ph, value, acc=accum)
                accum = tle.gpu.wgmma_wait(0, accum)
            elif half == 0:
                accum0 = tle.gpu.wgmma(ph, value, acc=accum0)
                accum0 = tle.gpu.wgmma_wait(0, accum0)
            else:
                accum1 = tle.gpu.wgmma(ph, value, acc=accum1)
                accum1 = tle.gpu.wgmma_wait(0, accum1)
            tl.debug_barrier()
        running = ds
    tl.store(
        Flags + (row * SPLITS + split) * (64 // BH) + tl.program_id(1), needs_repair
    )
    factor = running
    if SPLITS == 1:
        factor *= tl.div_rn(1.0, tl.where(denominator > 0, denominator, 1.0))
    base = (row * SPLITS + split) * 64 + h
    if V_TILE == 256:
        output_dims = tl.arange(0, 256)
        tl.store(
            Output + base[:, None] * 512 + output_dims[None, :],
            accum0 * factor[:, None],
        )
        tl.store(
            Output + base[:, None] * 512 + 256 + output_dims[None, :],
            accum1 * factor[:, None],
        )
    else:
        tl.store(Output + base[:, None] * 512 + d[None, :], accum * factor[:, None])
    logsum = tl.log(tl.where(denominator > 0, denominator, 1.0))
    if SPLITS == 1:
        tl.store(LSE + row * 64 + h, tl.where(denominator > 0, logsum, float("inf")))
    else:
        tl.store(LSE + base * 2, 0.0)
        tl.store(LSE + base * 2 + 1, denominator)


@triton.jit
def sparse_fp8_program_range(
    row,
    ROWS: tl.constexpr,
    PAGES: tl.constexpr,
    PROGRAMS: tl.constexpr,
    GROUPS: tl.constexpr,
):
    if GROUPS is None:
        first = ((row * PAGES + 1) * PROGRAMS - 1) // (ROWS * PAGES)
        last = ((row + 1) * PAGES * PROGRAMS - 1) // (ROWS * PAGES)
    else:
        GROUP_ROWS: tl.constexpr = GROUPS[0]
        GROUP_PROGRAMS: tl.constexpr = GROUPS[1]
        PREFIX_ROWS: tl.constexpr = GROUPS[2]
        ROW_PROGRAMS: tl.constexpr = GROUPS[3]
        PREFIX_PROGRAMS: tl.constexpr = PREFIX_ROWS * ROW_PROGRAMS
        TAIL_PROGRAMS: tl.constexpr = GROUP_PROGRAMS - PREFIX_PROGRAMS
        TAIL_PAGES: tl.constexpr = (GROUP_ROWS - PREFIX_ROWS) * PAGES
        group_start = (row // GROUP_ROWS) * GROUP_PROGRAMS
        local_row = row % GROUP_ROWS
        tail_row = local_row - PREFIX_ROWS
        first = group_start + tl.where(
            local_row < PREFIX_ROWS,
            local_row * ROW_PROGRAMS,
            PREFIX_PROGRAMS
            + ((tail_row * PAGES + 1) * TAIL_PROGRAMS - 1) // TAIL_PAGES,
        )
        last = group_start + tl.where(
            local_row < PREFIX_ROWS,
            (local_row + 1) * ROW_PROGRAMS - 1,
            PREFIX_PROGRAMS
            + ((tail_row + 1) * PAGES * TAIL_PROGRAMS - 1) // TAIL_PAGES,
        )
    return first, last


@triton.jit
def persistent_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Partial,
    Stats,
    PartialFlags,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    SLOTS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS: tl.constexpr,
    WARPS: tl.constexpr,
    REPAIR: tl.constexpr,
    PREFETCH: tl.constexpr = False,
):
    BN: tl.constexpr = 128
    PAGES: tl.constexpr = triton.cdiv(K, BN)
    PART_TOKENS: tl.constexpr = triton.cdiv(ROWS * PAGES, PROGRAMS) * BN
    PRE: tl.constexpr = triton.next_power_of_2(PART_TOKENS)
    CACHE_METADATA: tl.constexpr = PRE <= 4096 and TOKENS * KV_TOKEN_BYTES < 2**32
    # Keep the exponent multipliers normal and limit amplification of underflow
    # in the hoisted query factor.
    PREPARE_SCALES: tl.constexpr = (
        (K > 1024 or (ROWS == 16 and K == 640))
        and CACHE_METADATA
        and SCALE >= 2.0**-125
        and SCALE <= 2.0**16
    )
    if CACHE_METADATA:
        offset_cache = tle.gpu.alloc((PRE,), tl.int32, nv_mma_shared_layout=False)
        scale_cache = tle.gpu.alloc((PRE,), tl.float32, nv_mma_shared_layout=False)
    program = tl.program_id(0)
    ALIGNED: tl.constexpr = GROUPS is None and PROGRAMS % ROWS == 0
    if ALIGNED:
        QUERY_SPLITS: tl.constexpr = PROGRAMS // ROWS
        query = program // QUERY_SPLITS
        query_split = program % QUERY_SPLITS
    elif GROUPS is None:
        first = ROWS * PAGES * program // PROGRAMS
        stop = ROWS * PAGES * (program + 1) // PROGRAMS
    else:
        GROUP_ROWS: tl.constexpr = GROUPS[0]
        GROUP_PROGRAMS: tl.constexpr = GROUPS[1]
        PREFIX_ROWS: tl.constexpr = GROUPS[2]
        ROW_PROGRAMS: tl.constexpr = GROUPS[3]
        PREFIX_PROGRAMS: tl.constexpr = PREFIX_ROWS * ROW_PROGRAMS
        TAIL_PROGRAMS: tl.constexpr = GROUP_PROGRAMS - PREFIX_PROGRAMS
        TAIL_PAGES: tl.constexpr = (GROUP_ROWS - PREFIX_ROWS) * PAGES
        group_row = (program // GROUP_PROGRAMS) * GROUP_ROWS
        local_program = program % GROUP_PROGRAMS
        if local_program < PREFIX_PROGRAMS:
            query = group_row + local_program // ROW_PROGRAMS
            part = local_program % ROW_PROGRAMS
            first = query * PAGES + PAGES * part // ROW_PROGRAMS
            stop = query * PAGES + PAGES * (part + 1) // ROW_PROGRAMS
        else:
            base = (group_row + PREFIX_ROWS) * PAGES
            part = local_program - PREFIX_PROGRAMS
            first = base + TAIL_PAGES * part // TAIL_PROGRAMS
            stop = base + TAIL_PAGES * (part + 1) // TAIL_PROGRAMS
    head_group = tl.program_id(1) if HEADS < 64 else 0
    h = head_group * HEADS + tl.arange(0, HEADS)
    d = tl.arange(0, 512)
    col = tl.arange(0, BN)
    q = tle.gpu.alloc((HEADS, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    if PREFETCH:
        sk = tle.gpu.alloc((2, BN, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    else:
        sk = tle.gpu.alloc((BN, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((HEADS, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((HEADS, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    # Keep the copy indices separate from MMA indices to avoid layout reuse.
    origin = tl.inline_asm_elementwise(
        "",
        constraints="=R,0",
        args=[tl.full((), 128, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    copy_rows = tl.arange(128, 128 + BN) - origin
    cd = tl.arange(1024, 1056) * 16 - 16384
    copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
        [1, 1], [2, 16], [WARPS, 1], [1, 0]
    )
    rr = tle.gpu.set_layout(tl.broadcast_to(copy_rows[:, None], (BN, 32)), copy_layout)
    cc = tle.gpu.set_layout(tl.broadcast_to(cd[None, :], (BN, 32)), copy_layout)
    if not PREFETCH:
        destination = tle.gpu.local_ptr(sk, (rr, cc)).to(tl.uint64).to(tl.uint32)
    first_request = query if ALIGNED else first // PAGES
    fragments = 1 if ALIGNED else (stop - 1) // PAGES - first_request + 1
    for fragment in range(fragments):
        request = first_request + fragment
        row = request.to(tl.int64)
        if ALIGNED:
            begin = query_split * PAGES // QUERY_SPLITS
            end = (query_split + 1) * PAGES // QUERY_SPLITS
            slot = query_split
        else:
            begin = tl.maximum(first - request * PAGES, 0)
            end = tl.minimum(stop - request * PAGES, PAGES)
            first_program, _ = sparse_fp8_program_range(
                request, ROWS, PAGES, PROGRAMS, GROUPS
            )
            slot = program - first_program
        qv = tl.load(Q + (row * 64 + h[:, None]) * 512 + d[None, :])
        qs = tl.load(QS + row * 64 + h)
        if K >= 4096 or (PREFETCH and ALIGNED and ROWS == 16 and K == 640):
            # Four finite FP8 magnitudes fit in FP16. Inflate the subtotal
            # before applying QS to cover rounding in both reductions.
            q_abs = tl.reshape(tl.abs(qv.to(tl.float16)), (HEADS, 128, 4))
            q_parts = tl.sum(q_abs, 2, dtype=tl.float16).to(tl.float32)
            q_l1 = tl.sum(q_parts, 1) * 1.00154 * qs
        else:
            q_l1 = tl.sum(tl.abs(qv.to(tl.float32)), 1) * qs
        tl.store(tle.gpu.local_ptr(q), qv)
        meta_pos = tl.arange(0, PRE)
        offsets = begin * BN + meta_pos
        ids = tl.load(
            Indices + row * K + offsets, (offsets < K) & (offsets < end * BN), -1
        )
        valid = (offsets < K) & (offsets < end * BN) & (ids >= 0) & (ids < TOKENS)
        scales = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        if CACHE_METADATA:
            # Reuse the guard's gathers instead of reloading them for each page.
            tl.store(
                tle.gpu.local_ptr(offset_cache, (meta_pos,)),
                encode_kv_byte_offset(ids, valid, TOKENS),
            )
            if not PREPARE_SCALES:
                tl.store(tle.gpu.local_ptr(scale_cache, (meta_pos,)), scales)
        maximum = tl.max(scales, 0)
        fixed_scale = tl.maximum(maximum, 1e-30)
        minimum = tl.min(tl.where(valid, scales, float("inf")), 0)
        score_error = (
            q_l1 * (448 * maximum) * SCALE * (512 * 2**-24 / (1 - 512 * 2**-24))
        )
        U: tl.constexpr = 2**-24
        GP: tl.constexpr = 2 * PART_TOKENS * U / (1 - 2 * PART_TOKENS * U)
        DS: tl.constexpr = triton.cdiv(PART_TOKENS, BN) + 8
        GD: tl.constexpr = DS * 2**-23 / (1 - DS * 2**-23)
        coefficient = (
            2**-9 + 1.25 * GP + GD + 4 * 2**-19 + 8 * U + 4 * SLOTS * 2**-23
        ) / (1 - GD)
        bound = q_l1 * (448 * fixed_scale) * SCALE * 1.0005 + 1e-5
        if PREPARE_SCALES:
            # Budget rounding in the reordered FP32 scale products and quotient.
            score_error += bound * (16 * U)
        floor = 2**-14 * tl.exp(tl.minimum(bound, 20.0)) * (1 + 4 * GP) / (1 - GD)
        pv_error = (
            448 * fixed_scale * (coefficient + floor + tl.exp(4.1 * score_error) - 1)
        )
        if K > 1024 or (ROWS == 16 and K == 640):
            use_poly = tl.min(((qs >= 0) & (bound <= 2.0)).to(tl.int32), 0) != 0
            # A row can mix exponential paths across its fragments.
            pv_error += 448 * fixed_scale * BOUNDED_EXP_PV_BUDGET
        else:
            use_poly = False
        if PREPARE_SCALES:
            scale = fixed_scale * (1.0 / 16.0)
            # Zero queries do not need a potentially large scale product.
            q_scaled = tl.where(q_l1 == 0.0, 0.0, qs) * scale
            VECTOR_LOG2: tl.constexpr = PREFETCH and ALIGNED and ROWS == 16 and K == 640
            # The vector exponential takes the native log2 argument directly.
            if VECTOR_LOG2:
                multiplier = SCALE * 1.4426950408889634
            else:
                multiplier = tl.where(use_poly, SCALE * 0.5, SCALE * 1.4426950408889634)
            exponent_scale = q_scaled * multiplier
        eligible = (
            tl.min(
                ((score_error <= 1e-4) & (pv_error < 0.0038)).to(tl.int32),
                0,
            )
            != 0
        )
        eligible = eligible & (minimum >= fixed_scale / 256) & (0.5 * maximum <= 0.0019)
        accumulator = tl.full((HEADS, 512), 0.0, tl.float32)
        # Defer the cross-warp row reduction until the fragment is complete.
        probability_sums = tl.full((HEADS, BN), 0.0, tl.float32)
        if PREPARE_SCALES:
            inverse = tl.div_rn(16.0, fixed_scale)
            tl.store(tle.gpu.local_ptr(scale_cache, (meta_pos,)), scales * inverse)
            scale = scale + tl.zeros((HEADS,), tl.float32)
        else:
            scale = fixed_scale / 16 + tl.zeros((HEADS,), tl.float32)
            inverse = 1.0 / scale
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        if PREFETCH:
            # Keep shared views local; older TLE cannot lower them across calls.
            init_location = tl.load(tle.gpu.local_ptr(offset_cache, (rr,)))
            init_source = kv_copy_address(KV, init_location, cc, TOKENS, True)
            init_destination = (
                tle.gpu.local_ptr(sk.slot(0), (rr, cc)).to(tl.uint64).to(tl.uint32)
            )
            bounded_copy16(
                init_destination,
                init_source,
                KV.to(tl.uint64),
                tl.full((), TOKENS * 512, tl.uint64),
            )
            bounded_copy_wait(tl.full((), 0, tl.int32))
            lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
        for page in range(begin, end):
            if PREFETCH:
                pipe_slot = (page - begin) & 1
                has_next = page + 1 < end
                if has_next:
                    next_position = (page + 1 - begin) * BN + rr
                    next_location = tl.load(
                        tle.gpu.local_ptr(offset_cache, (next_position,))
                    )
                    next_source = kv_copy_address(KV, next_location, cc, TOKENS, True)
                    next_destination = (
                        tle.gpu.local_ptr(sk.slot(1 - pipe_slot), (rr, cc))
                        .to(tl.uint64)
                        .to(tl.uint32)
                    )
                    bounded_copy16(
                        next_destination,
                        next_source,
                        KV.to(tl.uint64),
                        tl.full((), TOKENS * 512, tl.uint64),
                    )
                current_kv = sk.slot(pipe_slot)
            else:
                if CACHE_METADATA:
                    copy_pos = (page - begin) * BN + rr
                    copy_location = tl.load(
                        tle.gpu.local_ptr(offset_cache, (copy_pos,))
                    )
                    source = kv_copy_address(KV, copy_location, cc, TOKENS, True)
                else:
                    copy_offsets = page * BN + rr
                    copy_ids = tl.load(
                        Indices + row * K + copy_offsets, copy_offsets < K, TOKENS
                    )
                    source = kv_copy_address(KV, copy_ids, cc, TOKENS, False)
                bounded_copy16(
                    destination,
                    source,
                    KV.to(tl.uint64),
                    tl.full((), TOKENS * 512, tl.uint64),
                )
                bounded_copy_wait(tl.full((), 0, tl.int32))
                lma_wait(tl.full((), 0, tl.int32))
                tl.debug_barrier()
                current_kv = sk
            scores = tle.gpu.wgmma(q, current_kv, trans_b=True)
            scores = tle.gpu.wgmma_wait(0, scores)
            if CACHE_METADATA:
                page_pos = (page - begin) * BN + col
                ks = tl.load(tle.gpu.local_ptr(scale_cache, (page_pos,)))
                if PREPARE_SCALES:
                    present = ks > 0.0
                else:
                    page_location = tl.load(
                        tle.gpu.local_ptr(offset_cache, (page_pos,))
                    )
                    present = page_location.to(tl.uint32) < TOKENS * KV_TOKEN_BYTES
            else:
                indices = tl.load(
                    Indices + row * K + page * BN + col, page * BN + col < K, -1
                )
                present = (page * BN + col < K) & (indices >= 0) & (indices < TOKENS)
                ks = tl.load(KS + tl.where(present, indices, 0), present, 0.0)
            if PREPARE_SCALES:
                scaled = scores * exponent_scale[:, None] * ks[None, :]
                scaled = tl.where(eligible, scaled, 0.0)
                if VECTOR_LOG2:
                    probability = vector_exp2(scaled, HEADS, BN)
                elif use_poly:
                    probability = bounded_exp_from_half(scaled)
                else:
                    probability = tl.exp2(scaled)
            else:
                scores = scores * qs[:, None] * ks[None, :]
                if use_poly:
                    probability = bounded_exp_from_half(
                        tl.where(eligible, scores * (SCALE * 0.5), 0.0)
                    )
                else:
                    scores = scores * (SCALE * 1.4426950408889634)
                    log2_score = tl.where(eligible, scores, 0.0)
                    if ROWS == 4 and K == 640 and HEADS == 32 and WARPS == 16:
                        probability = vector_exp2(log2_score, HEADS, BN)
                    else:
                        probability = tl.exp2(log2_score)
            p = tl.where(present[None, :], probability, 0.0)
            probability_sums += p
            if PREPARE_SCALES:
                normalized = p * ks[None, :]
            else:
                normalized = p * (ks[None, :] * inverse[:, None])
            high = normalized.to(tl.float8e4nv)
            low = (normalized - high.to(tl.float32)).to(tl.float8e4nv)
            tl.store(tle.gpu.local_ptr(ph), high)
            tl.store(tle.gpu.local_ptr(pl), low)
            lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
            accumulator = tle.gpu.wgmma(ph, current_kv, acc=accumulator)
            accumulator = tle.gpu.wgmma(pl, current_kv, acc=accumulator)
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
            if PREFETCH:
                if has_next:
                    bounded_copy_wait(tl.full((), 0, tl.int32))
                    lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
        denominator = tl.sum(probability_sums, 1)
        base = (row * SLOTS + slot) * 64 + h
        if SLOTS == 1:
            if REPAIR is not None and not eligible:
                REPAIR(
                    Q,
                    None,
                    KV,
                    None,
                    QS,
                    KS,
                    Indices,
                    None,
                    None,
                    None,
                    Partial,
                    Stats,
                    None,
                    row,
                    64,
                    1,
                    K,
                    TOKENS,
                    0,
                    SCALE,
                    False,
                    False,
                    1,
                    triton.cdiv(K, 64) * 64,
                    HEADS,
                    32,
                )
            else:
                norm = scale / tl.where(denominator > 0, denominator, 1.0)
                tl.store(
                    Partial + base[:, None] * 512 + d[None, :],
                    accumulator * norm[:, None],
                )
                tl.store(
                    Stats + base,
                    tl.where(denominator > 0, tl.log(denominator), float("inf")),
                )
        else:
            tl.store(
                Partial + base[:, None] * 512 + d[None, :], accumulator * scale[:, None]
            )
            tl.store(Stats + base * 2, 0.0)
            tl.store(Stats + base * 2 + 1, denominator)
        flag_offset = (row * SLOTS + slot) * (64 // HEADS) + head_group
        if REPAIR is None:
            tl.store(PartialFlags + flag_offset, (~eligible).to(tl.int32))
        tl.debug_barrier()


@triton.jit
def combine_persistent_sparse_fp8(
    Partial,
    Stats,
    PartialFlags,
    Output,
    LSE,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK_FLAGS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    COMBINE_HEADS: tl.constexpr,
    Flags,
    Q,
    KV,
    QS,
    KS,
    Indices,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    REPAIR: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    PAGES: tl.constexpr = triton.cdiv(K, 128)
    ALIGNED: tl.constexpr = GROUPS is None and PROGRAMS % ROWS == 0
    if ALIGNED:
        splits = tl.full((), PROGRAMS // ROWS, tl.int32)
    else:
        first_program, last_program = sparse_fp8_program_range(
            row, ROWS, PAGES, PROGRAMS, GROUPS
        )
        splits = last_program - first_program + 1
    part = tl.arange(0, BLOCK_FLAGS)
    flag_pos = row * SLOTS * HEAD_GROUPS + part
    flag_valid = part < splits * HEAD_GROUPS
    if REPAIR is not None:
        bad = tl.max(tl.load(PartialFlags + flag_pos, flag_valid, 0), 0)
    else:
        bad = 0
    if REPAIR is not None and bad != 0:
        REPAIR_HEADS: tl.constexpr = max(16, COMBINE_HEADS)
        if tile < 64 // REPAIR_HEADS:
            REPAIR(
                Q,
                None,
                KV,
                None,
                QS,
                KS,
                Indices,
                None,
                None,
                None,
                Output,
                LSE,
                None,
                row,
                64,
                1,
                K,
                TOKENS,
                0,
                SCALE,
                False,
                False,
                1,
                triton.cdiv(K, 64) * 64,
                REPAIR_HEADS,
                32,
            )
    else:
        HEADS: tl.constexpr = COMBINE_HEADS
        h = tile * HEADS + tl.arange(0, HEADS)
        d = tl.arange(0, 512)
        numerator = tl.full((HEADS, 512), 0.0, tl.float32)
        denominator = tl.full((HEADS,), 0.0, tl.float32)
        iterations = SLOTS if REPAIR is None else splits
        UNROLL: tl.constexpr = SLOTS if ALIGNED and SLOTS <= 6 else 1
        for split in tl.range(0, iterations, loop_unroll_factor=UNROLL):
            valid = split < splits if REPAIR is None else True
            partial = tl.load(
                Partial + ((row * SLOTS + split) * 64 + h[:, None]) * 512 + d[None, :],
                valid,
                0.0,
            )
            weight = tl.load(
                Stats + ((row * SLOTS + split) * 64 + h) * 2 + 1, valid, 0.0
            )
            numerator += partial
            denominator += weight
        inverse = 1.0 / tl.where(denominator > 0, denominator, 1.0)
        output = numerator * inverse[:, None]
        tl.store(Output + (row * 64 + h[:, None]) * 512 + d[None, :], output)
        tl.store(
            LSE + row * 64 + h,
            tl.where(denominator > 0, tl.log(denominator), float("inf")),
        )
        if REPAIR is None and tile == 0:
            bad = tl.max(tl.load(PartialFlags + flag_pos, flag_valid, 0), 0)
            tl.store(Flags + row, bad)


# Persistent MODEL1 K128 pipeline


@triton.jit
def k128_consume(
    Output,
    PScale,
    ph,
    pl,
    sk,
    FinalInv,
    Full,
    PEmpty,
    NormReady,
    Fast,
    Rows,
    PROGRAMS: tl.constexpr,
    QueryDone,
    TME_QUERY: tl.constexpr,
):
    # Each CTA owns complete queries; skipped queries do not advance phases.
    query_start = Rows * tl.program_id(0) // PROGRAMS
    query_stop = Rows * (tl.program_id(0) + 1) // PROGRAMS
    epoch = 0
    for request in range(query_start, query_stop):
        row = request.to(tl.int64)
        mode = tl.load(Fast + row, volatile=True)
        if mode != 0:
            accum = tle.gpu.set_layout(tl.full((64, 512), 0.0, tl.float32), PV)
            slot = epoch & 1
            tle.gpu.barrier_wait(Full, phaseIdx=slot)
            (high, low) = (ph, pl)
            values = sk.slot(slot)
            accum = tle.gpu.wgmma(high, values, acc=accum)
            accum = tle.gpu.wgmma(low, values, acc=accum)
            accum = tle.gpu.wgmma_wait(0, accum)
            tle.gpu.barrier_arrive(PEmpty, phaseIdx=slot)
            oh = tl.inline_asm_elementwise(
                "",
                constraints="=R,0",
                args=[tl.full((), 256, tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            od = tl.inline_asm_elementwise(
                "",
                constraints="=R,0",
                args=[tl.full((), 1024, tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            h = tle.gpu.set_layout(tl.arange(256, 320) - oh, OROW)
            d = tle.gpu.set_layout(tl.arange(1024, 1536) - od, OCOL)
            tle.gpu.barrier_wait(NormReady, phaseIdx=epoch & 1)
            inverse = tl.load(tle.gpu.local_ptr(FinalInv, (h,)), volatile=True)
            if TME_QUERY:
                result = accum * inverse[:, None]
            else:
                scale = tl.load(PScale + row * 64 + h, volatile=True)
                result = accum * scale[:, None] * inverse[:, None]
            tl.store(Output + (row * 64 + h[:, None]) * 512 + d[None, :], result)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(QueryDone, phaseIdx=epoch & 1)
            epoch += 1


@triton.jit
def k128_produce(
    Q,
    KV,
    QS,
    Locations,
    PScale,
    LSE,
    ph,
    pl,
    sk,
    FinalInv,
    Full,
    PEmpty,
    NormReady,
    Sync,
    SumSync,
    Selected,
    Fast,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    Rows,
    PROGRAMS: tl.constexpr,
    QueryDone,
    QReady,
):
    q = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    # Each CTA owns complete queries; skipped queries do not advance phases.
    query_start = Rows * tl.program_id(0) // PROGRAMS
    query_stop = Rows * (tl.program_id(0) + 1) // PROGRAMS
    epoch = 0
    for request in range(query_start, query_stop):
        row = request.to(tl.int64)
        mode = tl.load(Fast + row, volatile=True)
        if mode != 0:
            h = tl.arange(0, 64)
            col = tl.arange(0, 128)
            qs = tl.load(QS + row * 64 + h)
            ds = tl.load(PScale + row * 64 + h)
            reciprocal = tl.div_rn(1.0, ds)
            copy_origin = tl.inline_asm_elementwise(
                "",
                constraints="=R,0",
                args=[tl.full((), 128, tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            cr = tl.arange(128, 256) - copy_origin
            cd = tl.arange(0, 32) * 16
            copy_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [2, 16], [16, 1], [1, 0]
            )
            kv_lane_rows: tl.constexpr = 8 if isinstance(Q, tl.tensor_descriptor) else 2
            kv_layout: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 1], [kv_lane_rows, 32 // kv_lane_rows], [16, 1], [1, 0]
            )
            crr = tle.gpu.set_layout(tl.broadcast_to(cr[:, None], (128, 32)), kv_layout)
            cdd = tle.gpu.set_layout(tl.broadcast_to(cd[None, :], (128, 32)), kv_layout)
            if isinstance(Q, tl.tensor_descriptor):
                tle.gpu.copy(Q, q, (64, 512), offsets=(request * 64, 0), barrier=QReady)
            else:
                q_origin = tl.inline_asm_elementwise(
                    "",
                    constraints="=R,0",
                    args=[tl.full((), 4096, tl.int32)],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                q_rows = tl.arange(4096, 4160) - q_origin
                qrr = tle.gpu.set_layout(
                    tl.broadcast_to(q_rows[:, None], (64, 32)), copy_layout
                )
                qcc = tle.gpu.set_layout(
                    tl.broadcast_to(cd[None, :], (64, 32)), copy_layout
                )
                q_dst = tle.gpu.local_ptr(q, (qrr, qcc)).to(tl.uint64).to(tl.uint32)
                q_base = Q + row * 64 * 512
                q_src = q_base + qrr.to(tl.int64) * 512 + qcc
                bounded_copy16(
                    q_dst,
                    q_src.to(tl.uint64),
                    q_base.to(tl.uint64),
                    tl.full((), 64 * 512, tl.uint64),
                )
            if USE_BYTE_OFFSETS:
                sentinel = tl.full((), TOKENS * KV_TOKEN_BYTES, tl.uint32).to(tl.int32)
            else:
                sentinel = TOKENS
            init_location = tl.load(
                Locations + row * 128 + crr, crr < 128, sentinel, volatile=True
            )
            init_src = kv_copy_address(KV, init_location, cdd, TOKENS, USE_BYTE_OFFSETS)
            init_dst0 = (
                tle.gpu.local_ptr(sk.slot(epoch & 1), (crr, cdd))
                .to(tl.uint64)
                .to(tl.uint32)
            )
            bounded_copy16(
                init_dst0,
                init_src,
                KV.to(tl.uint64),
                tl.full((), TOKENS * 512, tl.uint64),
            )
            bounded_copy_wait(tl.full((), 0, tl.int32))
            if isinstance(Q, tl.tensor_descriptor):
                tle.gpu.barrier_wait(QReady, phaseIdx=epoch & 1)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Sync, phaseIdx=epoch & 1)
            tle.gpu.barrier_wait(Sync, phaseIdx=epoch & 1)
            tle.gpu.barrier_wait(QueryDone, phaseIdx=epoch & 1)
            total_matrix = tl.full((64, 128), 0.0, tl.float32)
            slot = epoch & 1
            offsets = col
            ks = tl.load(Selected + row * 128 + offsets, offsets < 128, 0.0)
            valid = (offsets < 128) & (ks > 0.0)
            dot = tle.gpu.wgmma(q, sk.slot(slot), trans_b=True)
            dot = tle.gpu.wgmma_wait(0, dot).to(tl.float32)
            tle.gpu.barrier_wait(PEmpty, phaseIdx=slot)
            (high_tile, low_tile) = (ph, pl)
            shifted = dot * qs[:, None] * ks[None, :] * SCALE
            if isinstance(Q, tl.tensor_descriptor):
                exp_value = vector_exp2(shifted * 1.4426950408889634, 64, 128)
            else:
                exp_value = tl.exp(shifted)
            probability = tl.where(valid[None, :], exp_value, 0.0)
            total_matrix += probability
            normalized = probability * ks[None, :] * reciprocal[:, None]
            high = normalized.to(tl.float8e4nv)
            low = (normalized - high.to(tl.float32)).to(tl.float8e4nv)
            phh = tl.broadcast_to(h[:, None], (64, 128))
            phc = tl.broadcast_to(col[None, :], (64, 128))
            tl.store(tle.gpu.local_ptr(high_tile, (phh, phc)), high)
            tl.store(tle.gpu.local_ptr(low_tile, (phh, phc)), low)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Full, phaseIdx=slot)
            total = tl.sum(tl.sum(tl.reshape(total_matrix, (64, 4, 32)), 2), 1)
            inverse = tl.where(total > 0, 1.0 / tl.where(total > 0, total, 1.0), 0.0)
            if isinstance(Q, tl.tensor_descriptor):
                inverse *= ds
            tl.store(tle.gpu.local_ptr(FinalInv, (h,)), inverse)
            lse = tl.where(total > 0, tl.log(total), float("inf"))
            tl.store(LSE + row * 64 + h, lse)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(NormReady, phaseIdx=epoch & 1)
            # Complete all producer LSE scratch reads before the next query.
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(SumSync, phaseIdx=epoch & 1)
            tle.gpu.barrier_wait(SumSync, phaseIdx=epoch & 1)
            epoch += 1


@triton.jit
def persistent_k128(
    Q,
    KV,
    QS,
    Locations,
    PScale,
    Fast,
    Output,
    LSE,
    Selected,
    Rows,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    USE_BYTE_OFFSETS: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    sk = tle.gpu.alloc((2, 128, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((64, 128), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((64, 128), tl.float8e4nv, nv_mma_shared_layout=True)
    inverse = tle.gpu.alloc((64,), tl.float32, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barrier(arrive_count=16)
    pe = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.READY)
    norm_ready = tle.gpu.alloc_barrier(arrive_count=16)
    sync = tle.gpu.alloc_barrier(arrive_count=16)
    sb = tle.gpu.alloc_barrier(arrive_count=16)
    query_done = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.READY)
    q_ready = (
        tle.gpu.alloc_barrier(arrive_count=1, expect_bytes=64 * 512)
        if isinstance(Q, tl.tensor_descriptor)
        else full
    )
    tl.debug_barrier()
    tle.gpu.warp_specialize(
        [
            (
                k128_produce,
                (
                    Q,
                    KV,
                    QS,
                    Locations,
                    PScale,
                    LSE,
                    ph,
                    pl,
                    sk,
                    inverse,
                    full,
                    pe,
                    norm_ready,
                    sync,
                    sb,
                    Selected,
                    Fast,
                    TOKENS,
                    SCALE,
                    USE_BYTE_OFFSETS,
                    Rows,
                    PROGRAMS,
                    query_done,
                    q_ready,
                ),
            ),
            (
                k128_consume,
                (
                    Output,
                    PScale,
                    ph,
                    pl,
                    sk,
                    inverse,
                    full,
                    pe,
                    norm_ready,
                    Fast,
                    Rows,
                    PROGRAMS,
                    query_done,
                    isinstance(Q, tl.tensor_descriptor),
                ),
            ),
        ],
        worker_num_warps=[16],
        worker_num_regs=[128],
    )


@lru_cache(maxsize=64)
def compiled_kernel(
    topk,
    tokens,
    scale,
    output_dtype,
    device_index,
    target,
    use_byte_offsets,
    programs,
    use_tme_q=False,
):
    assert topk == 128 and programs > 0
    fn = persistent_k128
    signature = dict(
        zip(
            fn.arg_names[:10],
            [
                "*fp8e4nv",
                "*fp8e4nv",
                "*fp32",
                "*i32",
                "*fp32",
                "*i32",
                output_dtype,
                "*fp32",
                "*fp32",
                "i32",
            ],
        )
    )
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(9)}
    if use_tme_q:
        signature["Q"] = "tensordesc<fp8e4nv[64,512]>"
        attrs[(0,)] = [["musa.tme_tail_divisibility", 4]]
    constants = dict(
        TOKENS=tokens, SCALE=scale, USE_BYTE_OFFSETS=use_byte_offsets, PROGRAMS=programs
    )
    try:
        return triton.compile(
            ASTSource(
                fn,
                signature,
                constants,
                attrs=attrs,
            ),
            target=target,
            options=dict(
                num_warps=16,
                num_stages=1,
                maxnreg=128,
                enable_backend_opt=True,
                enable_llc_opt=True,
                extern_libs={
                    "native": (
                        native_library(VECTOR_EXP2_IR)
                        if use_tme_q
                        else native_library()
                    )
                },
            ),
        )
    except (CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug(
            "Sparse FP8 K128 pipeline unavailable: %s", error
        )
        return None


# Staged MODEL1 pipeline for short batches


@triton.jit
def staged_guard(
    Q,
    QS,
    KS,
    Indices,
    PartialFlags,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    SLOTS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS: tl.constexpr,
    MetaOffset,
    MetaScale,
    Setup,
    Modes,
    First,
    Stop,
):
    BN: tl.constexpr = 128
    PAGES: tl.constexpr = triton.cdiv(K, BN)
    PART_TOKENS: tl.constexpr = triton.cdiv(ROWS * PAGES, PROGRAMS) * BN
    PRE: tl.constexpr = triton.next_power_of_2(PART_TOKENS)
    CACHE_METADATA: tl.constexpr = PRE <= 4096 and TOKENS * KV_TOKEN_BYTES < 2**32
    PREPARE_SCALES: tl.constexpr = (
        (K > 1024 or (ROWS == 16 and K == 640))
        and CACHE_METADATA
        and (SCALE >= 2.0 ** (-125))
        and (SCALE <= 2.0**16)
    )
    tl.static_assert(ROWS == 16 and K == 8320 and (HEADS == 64) and PREPARE_SCALES)
    program = tl.program_id(0)
    h = tl.arange(0, HEADS)
    first_request = First // PAGES
    fragments = (Stop - 1) // PAGES - first_request + 1
    for fragment in range(fragments):
        request = first_request + fragment
        row = request.to(tl.int64)
        begin = tl.maximum(First - request * PAGES, 0)
        end = tl.minimum(Stop - request * PAGES, PAGES)
        (first_program, _) = sparse_fp8_program_range(
            request, ROWS, PAGES, PROGRAMS, GROUPS
        )
        slot = program - first_program
        word_col = tl.arange(0, 128)
        words = tl.load(
            Q.to(tl.pointer_type(tl.uint32))
            + (row * 64 + h[:, None]) * 128
            + word_col[None, :]
        )
        qs = tl.load(QS + row * 64 + h)
        q_parts = query_abs4(words)
        # Four finite E4M3 magnitudes fit in FP16; retain the original L1 inflation.
        q_l1 = tl.sum(q_parts, 1) * 1.00154 * qs
        meta_pos = tl.arange(0, PRE)
        offsets = begin * BN + meta_pos
        ids = tl.load(
            Indices + row * K + offsets, (offsets < K) & (offsets < end * BN), -1
        )
        valid = (offsets < K) & (offsets < end * BN) & (ids >= 0) & (ids < TOKENS)
        scales = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
        maximum = tl.max(scales, 0)
        fixed_scale = tl.maximum(maximum, 1e-30)
        minimum = tl.min(tl.where(valid, scales, float("inf")), 0)
        score_error = (
            q_l1 * (448 * maximum) * SCALE * (512 * 2 ** (-24) / (1 - 512 * 2 ** (-24)))
        )
        U: tl.constexpr = 2 ** (-24)
        GP: tl.constexpr = 2 * PART_TOKENS * U / (1 - 2 * PART_TOKENS * U)
        DS: tl.constexpr = triton.cdiv(PART_TOKENS, BN) + 8
        GD: tl.constexpr = DS * 2 ** (-23) / (1 - DS * 2 ** (-23))
        coefficient = (
            2 ** (-9) + 1.25 * GP + GD + 4 * 2 ** (-19) + 8 * U + 4 * SLOTS * 2 ** (-23)
        ) / (1 - GD)
        bound = q_l1 * (448 * fixed_scale) * SCALE * 1.0005 + 1e-05
        if PREPARE_SCALES:
            score_error += bound * (16 * U)
        floor = 2 ** (-14) * tl.exp(tl.minimum(bound, 20.0)) * (1 + 4 * GP) / (1 - GD)
        pv_error = (
            448 * fixed_scale * (coefficient + floor + tl.exp(4.1 * score_error) - 1)
        )
        if K > 1024 or (ROWS == 16 and K == 640):
            use_poly = tl.min(((qs >= 0) & (bound <= 2.0)).to(tl.int32), 0) != 0
            pv_error += 448 * fixed_scale * BOUNDED_EXP_PV_BUDGET
        else:
            use_poly = False
        if PREPARE_SCALES:
            scale = fixed_scale * (1.0 / 16.0)
            q_scaled = tl.where(q_l1 == 0.0, 0.0, qs) * scale
            multiplier = tl.where(use_poly, SCALE * 0.5, SCALE * 1.4426950408889634)
            exponent_scale = q_scaled * multiplier
        eligible = (
            tl.min(((score_error <= 0.0001) & (pv_error < 0.0038)).to(tl.int32), 0) != 0
        )
        eligible = eligible & (minimum >= fixed_scale / 256) & (0.5 * maximum <= 0.0019)
        inverse = tl.div_rn(16.0, fixed_scale)
        meta_base = (row * SLOTS + slot) * PRE
        tl.store(
            MetaOffset + meta_base + meta_pos, encode_kv_byte_offset(ids, valid, TOKENS)
        )
        tl.store(MetaScale + meta_base + meta_pos, scales * inverse)
        setup_base = ((row * SLOTS + slot) * 64 + h) * 2
        tl.store(Setup + setup_base, exponent_scale)
        tl.store(Setup + setup_base + 1, scale)
        tl.store(
            Modes + row * SLOTS + slot, tl.where(eligible, tl.where(use_poly, 2, 1), 0)
        )
        tl.store(PartialFlags + row * SLOTS + slot, (~eligible).to(tl.int32))


@triton.jit
def staged_produce(
    Q,
    MetaScale,
    Setup,
    Modes,
    Stats,
    First,
    Stop,
    q,
    sk,
    ph,
    pl,
    Full,
    Ready,
    Sync,
    SumParts,
    SumSync,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    PROGRAMS: tl.constexpr,
    SLOTS: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
):
    BN: tl.constexpr = 128
    PAGES: tl.constexpr = triton.cdiv(K, BN)
    PRE: tl.constexpr = triton.next_power_of_2(triton.cdiv(ROWS * PAGES, PROGRAMS) * BN)
    first_request = First // PAGES
    fragments = (Stop - 1) // PAGES - first_request + 1
    h = tl.arange(0, 64)
    col = tl.arange(0, BN)
    copy_layout: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [2, 16], [16, 1], [1, 0])
    q_origin = tl.inline_asm_elementwise(
        "",
        constraints="=R,0",
        args=[tl.full((), 4096, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    q_rows = tl.arange(4096, 4160) - q_origin
    qrr = tle.gpu.set_layout(tl.broadcast_to(q_rows[:, None], (64, 32)), copy_layout)
    qcc = tle.gpu.set_layout(
        tl.broadcast_to((tl.arange(0, 32) * 16)[None, :], (64, 32)), copy_layout
    )
    q_dst = tle.gpu.local_ptr(q, (qrr, qcc)).to(tl.uint64).to(tl.uint32)
    epoch = 0
    fragment_phase = 0
    for fragment in range(fragments):
        row = (first_request + fragment).to(tl.int64)
        begin = tl.maximum(First - row * PAGES, 0)
        end = tl.minimum(Stop - row * PAGES, PAGES)
        (first_program, _) = sparse_fp8_program_range(
            row, ROWS, PAGES, PROGRAMS, (G0, G1, G2, G3)
        )
        part = tl.program_id(0) - first_program
        base = row * SLOTS + part
        mode = tl.load(Modes + base)
        if mode != 0:
            q_base = Q + row * 64 * 512
            q_src = q_base + qrr * 512 + qcc
            bounded_copy16(
                q_dst,
                q_src.to(tl.uint64),
                q_base.to(tl.uint64),
                tl.full((), 64 * 512, tl.uint64),
            )
            exponent_scale = tl.load(Setup + (base * 64 + h) * 2)
            bounded_copy_wait(tl.full((), 0, tl.int32))
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Sync, phaseIdx=0)
            tle.gpu.barrier_wait(Sync, phaseIdx=0)
            probability_sums = tl.full((64, BN), 0.0, tl.float32)
            for page in range(begin, end):
                slot = epoch & 1
                phase = epoch // 2 & 1
                tle.gpu.barrier_wait(Ready[slot], phaseIdx=phase)
                ks = tl.load(MetaScale + base * PRE + (page - begin) * BN + col)
                present = ks > 0.0
                scores = tle.gpu.wgmma(q, sk.slot(slot), trans_b=True)
                scores = tle.gpu.wgmma_wait(0, scores)
                scaled = scores * exponent_scale[:, None] * ks[None, :]
                if mode == 2:
                    probability = bounded_exp_from_half(scaled)
                else:
                    probability = tl.exp2(scaled)
                probability = tl.where(present[None, :], probability, 0.0)
                probability_sums += probability
                normalized = probability * ks[None, :]
                high = normalized.to(tl.float8e4nv)
                low = (normalized - high.to(tl.float32)).to(tl.float8e4nv)
                phh = tl.broadcast_to(h[:, None], (64, BN))
                phc = tl.broadcast_to(col[None, :], (64, BN))
                tl.store(tle.gpu.local_ptr(ph.slot(slot), (phh, phc)), high)
                tl.store(tle.gpu.local_ptr(pl.slot(slot), (phh, phc)), low)
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Full[slot], phaseIdx=phase)
                epoch += 1
            # The last QK wait releases Q storage for the denominator reduction.
            partial_sum = tl.sum(tl.reshape(probability_sums, (64, 4, 32)), 2)
            tl.store(tle.gpu.local_ptr(SumParts), partial_sum)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(SumSync, phaseIdx=fragment_phase)
            tle.gpu.barrier_wait(SumSync, phaseIdx=fragment_phase)
            s0 = tl.load(
                tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 0, tl.int32))),
                volatile=True,
            )
            s1 = tl.load(
                tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 1, tl.int32))),
                volatile=True,
            )
            s2 = tl.load(
                tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 2, tl.int32))),
                volatile=True,
            )
            s3 = tl.load(
                tle.gpu.local_ptr(SumParts, (h, tl.full((64,), 3, tl.int32))),
                volatile=True,
            )
            denominator = s0 + s1 + (s2 + s3)
            tl.store(Stats + (base * 64 + h) * 2, 0.0)
            tl.store(Stats + (base * 64 + h) * 2 + 1, denominator)
            # Finish all alias reads before any producer overwrites Q.
            tle.gpu.barrier_arrive(Sync, phaseIdx=1)
            tle.gpu.barrier_wait(Sync, phaseIdx=1)
            fragment_phase ^= 1


@triton.jit
def staged_consume(
    KV,
    MetaOffset,
    Setup,
    Modes,
    Partial,
    First,
    Stop,
    sk,
    ph,
    pl,
    Full,
    Empty,
    Ready,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    SLOTS: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
):
    BN: tl.constexpr = 128
    PAGES: tl.constexpr = triton.cdiv(K, BN)
    PRE: tl.constexpr = triton.next_power_of_2(triton.cdiv(ROWS * PAGES, PROGRAMS) * BN)
    first_request = First // PAGES
    fragments = (Stop - 1) // PAGES - first_request + 1
    origin = tl.inline_asm_elementwise(
        "",
        constraints="=R,0",
        args=[tl.full((), 128, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    copy_rows = tl.arange(128, 128 + BN) - origin
    copy_layout: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [8, 4], [16, 1], [1, 0])
    rr = tle.gpu.set_layout(tl.broadcast_to(copy_rows[:, None], (BN, 32)), copy_layout)
    cc = tle.gpu.set_layout(
        tl.broadcast_to((tl.arange(0, 32) * 16)[None, :], (BN, 32)), copy_layout
    )
    oh = tl.inline_asm_elementwise(
        "",
        constraints="=R,0",
        args=[tl.full((), 256, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    od = tl.inline_asm_elementwise(
        "",
        constraints="=R,0",
        args=[tl.full((), 1024, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    h = tle.gpu.set_layout(tl.arange(256, 320) - oh, OROW)
    d = tle.gpu.set_layout(tl.arange(1024, 1536) - od, OCOL)
    epoch = 0
    for fragment in range(fragments):
        row = (first_request + fragment).to(tl.int64)
        begin = tl.maximum(First - row * PAGES, 0)
        end = tl.minimum(Stop - row * PAGES, PAGES)
        (first_program, _) = sparse_fp8_program_range(
            row, ROWS, PAGES, PROGRAMS, (G0, G1, G2, G3)
        )
        part = tl.program_id(0) - first_program
        base = row * SLOTS + part
        mode = tl.load(Modes + base)
        if mode != 0:
            first_slot = epoch & 1
            first_phase = epoch // 2 & 1
            tle.gpu.barrier_wait(Empty[first_slot], phaseIdx=first_phase)
            location = tl.load(MetaOffset + base * PRE + rr)
            source = kv_copy_address(KV, location, cc, TOKENS, True)
            destination = (
                tle.gpu.local_ptr(sk.slot(first_slot), (rr, cc))
                .to(tl.uint64)
                .to(tl.uint32)
            )
            bounded_copy16(
                destination,
                source,
                KV.to(tl.uint64),
                tl.full((), TOKENS * 512, tl.uint64),
            )
            bounded_copy_wait(tl.full((), 0, tl.int32))
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Ready[first_slot], phaseIdx=first_phase)
            accum = tle.gpu.set_layout(tl.full((64, 512), 0.0, tl.float32), PV)
            for page in range(begin, end):
                slot = epoch & 1
                phase = epoch // 2 & 1
                if page + 1 < end:
                    next_slot = epoch + 1 & 1
                    next_phase = (epoch + 1) // 2 & 1
                    tle.gpu.barrier_wait(Empty[next_slot], phaseIdx=next_phase)
                    next_location = tl.load(
                        MetaOffset + base * PRE + (page + 1 - begin) * BN + rr
                    )
                    next_source = kv_copy_address(KV, next_location, cc, TOKENS, True)
                    next_destination = (
                        tle.gpu.local_ptr(sk.slot(next_slot), (rr, cc))
                        .to(tl.uint64)
                        .to(tl.uint32)
                    )
                    bounded_copy16(
                        next_destination,
                        next_source,
                        KV.to(tl.uint64),
                        tl.full((), TOKENS * 512, tl.uint64),
                    )
                    bounded_copy_wait(tl.full((), 0, tl.int32))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(Ready[next_slot], phaseIdx=next_phase)
                tle.gpu.barrier_wait(Full[slot], phaseIdx=phase)
                accum = tle.gpu.wgmma(ph.slot(slot), sk.slot(slot), acc=accum)
                accum = tle.gpu.wgmma(pl.slot(slot), sk.slot(slot), acc=accum)
                accum = tle.gpu.wgmma_wait(0, accum)
                tle.gpu.barrier_arrive(Empty[slot], phaseIdx=phase)
                epoch += 1
            scale = tl.load(Setup + (base * 64 + h) * 2 + 1)
            tl.store(
                Partial + (base * 64 + h[:, None]) * 512 + d[None, :],
                accum * scale[:, None],
            )


@triton.jit
def staged_main(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Partial,
    Stats,
    PartialFlags,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    SLOTS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS: tl.constexpr,
    WARPS: tl.constexpr,
    MetaOffset,
    MetaScale,
    Setup,
    Modes,
):
    tl.static_assert(
        ROWS == 16 and K == 8320 and (HEADS == 64) and (GROUPS is not None)
    )
    tl.static_assert(WARPS == 16)
    PAGES: tl.constexpr = triton.cdiv(K, 128)
    G0: tl.constexpr = GROUPS[0]
    G1: tl.constexpr = GROUPS[1]
    G2: tl.constexpr = GROUPS[2]
    G3: tl.constexpr = GROUPS[3]
    PREFIX_PROGRAMS: tl.constexpr = G2 * G3
    TAIL_PROGRAMS: tl.constexpr = G1 - PREFIX_PROGRAMS
    TAIL_PAGES: tl.constexpr = (G0 - G2) * PAGES
    program = tl.program_id(0)
    group_row = program // G1 * G0
    local_program = program % G1
    if local_program < PREFIX_PROGRAMS:
        query = group_row + local_program // G3
        part = local_program % G3
        first = query * PAGES + PAGES * part // G3
        stop = query * PAGES + PAGES * (part + 1) // G3
    else:
        base = (group_row + G2) * PAGES
        part = local_program - PREFIX_PROGRAMS
        first = base + TAIL_PAGES * part // TAIL_PROGRAMS
        stop = base + TAIL_PAGES * (part + 1) // TAIL_PROGRAMS
    staged_guard(
        Q,
        QS,
        KS,
        Indices,
        PartialFlags,
        ROWS,
        K,
        TOKENS,
        SCALE,
        PROGRAMS,
        SLOTS,
        GROUPS,
        HEADS,
        MetaOffset,
        MetaScale,
        Setup,
        Modes,
        first,
        stop,
    )
    q = tle.gpu.alloc((64, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    sk = tle.gpu.alloc((2, 128, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((2, 64, 128), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((2, 64, 128), tl.float8e4nv, nv_mma_shared_layout=True)
    sums = tle.gpu.alloc((64, 4), tl.float32, alias=q, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barriers(2, arrive_count=16)
    empty = tle.gpu.alloc_barriers(2, arrive_count=16, init=tle.gpu.READY)
    ready = tle.gpu.alloc_barriers(2, arrive_count=16)
    sync = tle.gpu.alloc_barrier(arrive_count=16)
    sum_sync = tle.gpu.alloc_barrier(arrive_count=16)
    tl.debug_barrier()
    tle.gpu.warp_specialize(
        [
            (
                staged_produce,
                (
                    Q,
                    MetaScale,
                    Setup,
                    Modes,
                    Stats,
                    first,
                    stop,
                    q,
                    sk,
                    ph,
                    pl,
                    full,
                    ready,
                    sync,
                    sums,
                    sum_sync,
                    ROWS,
                    K,
                    PROGRAMS,
                    SLOTS,
                    G0,
                    G1,
                    G2,
                    G3,
                ),
            ),
            (
                staged_consume,
                (
                    KV,
                    MetaOffset,
                    Setup,
                    Modes,
                    Partial,
                    first,
                    stop,
                    sk,
                    ph,
                    pl,
                    full,
                    empty,
                    ready,
                    ROWS,
                    K,
                    TOKENS,
                    PROGRAMS,
                    SLOTS,
                    G0,
                    G1,
                    G2,
                    G3,
                ),
            ),
        ],
        worker_num_warps=[16],
        worker_num_regs=[128],
    )


def prepare_staged_pipeline(
    q: torch.Tensor,
    kv: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    repair: JITFunction,
) -> Optional[Tuple[Callable[[], None], Optional[torch.Tensor]]]:
    rows, topk = q.shape[0], indices.shape[-1]
    pages = triton.cdiv(topk, 128)
    properties = driver.active.utils.get_device_properties(
        driver.active.get_current_device()
    )
    programs = min(properties["multiprocessor_count"], rows * pages)
    groups = group_short_pages(rows, pages, programs)
    if groups is None or properties["max_shared_mem"] < 192 * 1024:
        return None
    group_rows, group_programs, prefix_rows, row_programs = groups
    splits = max(
        row_programs,
        triton.cdiv(
            group_programs - prefix_rows * row_programs, group_rows - prefix_rows
        )
        + 1,
    )
    pre = triton.next_power_of_2(triton.cdiv(rows * pages, programs) * 128)
    # This optional specialization needs metadata beyond the ordinary short path.
    try:
        partial_flags = torch.empty((rows, splits), dtype=torch.int32, device=q.device)
        partial = torch.empty(
            (rows, splits, 64, 512), dtype=torch.float32, device=q.device
        )
        stats = torch.empty((rows, splits, 64, 2), dtype=torch.float32, device=q.device)
        meta_offset = torch.empty(
            (rows, splits, pre), dtype=torch.int32, device=q.device
        )
        meta_scale = torch.empty_like(meta_offset, dtype=torch.float32)
        setup = torch.empty((rows, splits, 64, 2), dtype=torch.float32, device=q.device)
        modes = torch.empty((rows, splits), dtype=torch.int32, device=q.device)
    except torch.OutOfMemoryError:
        return None
    kernel_args = (
        q,
        kv,
        qs,
        ks,
        indices,
        partial,
        stats,
        partial_flags,
        rows,
        topk,
        kv.shape[0] * 64,
        scale,
        programs,
        splits,
        groups,
        64,
        16,
        meta_offset,
        meta_scale,
        setup,
        modes,
    )
    grid = (programs, 1, 1)
    try:
        kernel = staged_main.warmup(
            *kernel_args,
            grid=grid,
            num_warps=16,
            num_stages=1,
            maxnreg=128,
            enable_backend_opt=True,
            enable_llc_opt=True,
            extern_libs={"native": native_library(QUERY_NORM_IR)},
        )
    except (CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug(
            "Sparse short pipeline unavailable: %s", error
        )
        return None
    runner = kernel[grid]
    combine_runner, combine_args, flags = prepare_persistent_combine(
        q,
        kv,
        qs,
        ks,
        indices,
        partial,
        stats,
        partial_flags,
        output,
        lse,
        scale,
        splits,
        programs,
        groups,
        repair,
        1,
    )

    def run_short():
        runner(*kernel_args)
        combine_runner(*combine_args)

    return run_short, flags


# V32 producer and consumer pipelines


@triton.jit
def observed_query_bounds(q_lower, q_upper, q_rope, query_scale, SCALE: tl.constexpr):
    query_valid = (query_scale >= 2.0**-20) & (query_scale <= 2.0**-6)
    q_scale = tl.where(query_valid, query_scale, 0.0) * SCALE
    q_bound = tl.max(
        tl.sum(tl.abs(q_lower.to(tl.float32)) + tl.abs(q_upper.to(tl.float32)), 1)
        * q_scale,
        0,
    )
    qr_bound = tl.max(tl.sum(tl.abs(q_rope.to(tl.float32)), 1) * q_scale, 0)
    return q_scale, query_valid, q_bound, qr_bound


@triton.jit
def observed_sentinel_ok(
    KV,
    Indices,
    row,
    start,
    end,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # Zero probability cannot mask a nonfinite placeholder value in PV.
    sentinel = tl.load(KV + tl.arange(0, 512)).to(tl.float32)
    sentinel_ok = tl.min((tl.abs(sentinel) <= 448.0).to(tl.int32), 0) != 0
    if not sentinel_ok:
        offsets = start + tl.arange(0, triton.next_power_of_2(CHUNK))
        within_split = offsets < end
        idx = tl.load(Indices + row * TOPK + offsets, within_split, 0)
        selected = (idx >= 0) & (idx < TOKENS)
        sentinel_ok = tl.min((~within_split | selected).to(tl.int32), 0) != 0
    return sentinel_ok


@triton.jit
def observed_score_error(q_bound, qr_bound, k_max, rope_max):
    # Include FP32 accumulation and scaling in the logit error bound.
    gamma: tl.constexpr = (580.0 * 2.0**-24) / (1.0 - 580.0 * 2.0**-24)
    return gamma * (448.0 * q_bound * tl.max(k_max, 0) + qr_bound * rope_max)


@triton.jit
def preloaded_active(
    Skip, Cutoff, Again, split_index, OBSERVED: tl.constexpr, MODE: tl.constexpr
):
    if MODE == 1:
        return tl.load(Cutoff + split_index) > 0.0
    elif MODE == 2:
        return tl.load(Again + split_index)
    else:
        return True if OBSERVED else not tl.load(Skip + split_index)


@triton.jit
def observed_eligible(
    sentinel_ok, query_valid, q_bound, qr_bound, k_max, rope_max, weighted, total
):
    score_error = observed_score_error(q_bound, qr_bound, k_max, rope_max)
    energy = tl.sum(weighted, 1) * (1.00001 / 16384.0)
    return (
        sentinel_ok
        & (tl.sum((~query_valid).to(tl.int32), 0) == 0)
        & (1.001 * score_error <= 8.0e-5)
        & (tl.max(energy - 0.00105 * total, 0) <= 0.0)
    )


@triton.jit
def v32_produce(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    LSE,
    Skip,
    QL,
    QU,
    QRope,
    KL,
    KU,
    KRope,
    PH,
    PL,
    Inv,
    Full,
    Empty,
    UpperEmpty,
    Local,
    Norm,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    OBSERVED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(2)
    split_index = row * SPLITS + split
    run = True if OBSERVED else not tl.load(Skip + split_index)
    if run:
        start = 0 if SPLITS == 1 else split * CHUNK
        end = tl.minimum(start + CHUNK, TOPK)
        pages = (
            tl.cdiv(CHUNK, BN)
            if SPLITS * CHUNK == TOPK
            else tl.cdiv(tl.maximum(end - start, 0), BN)
        )
        heads = tl.arange(0, 64)
        dims = tl.arange(0, 256)
        rope_dims = tl.arange(0, 64)
        cols = tl.arange(0, BN)
        qr_offset = (row * 64 + heads[:, None]) * 64 + rope_dims[None, :]
        q_offset = (row * 64 + heads[:, None]) * 512 + dims[None, :]
        if OBSERVED:
            q_lower = tl.load(Q + q_offset)
            q_upper = tl.load(Q + q_offset + 256)
            q_rope = tl.load(QR + qr_offset)
            tl.store(tle.gpu.local_ptr(QL), q_lower)
            tl.store(tle.gpu.local_ptr(QU), q_upper)
            tl.store(tle.gpu.local_ptr(QRope), q_rope)
            query_scale = tl.load(QS + row * 64 + heads)
            q_scale, query_valid, q_bound, qr_bound = observed_query_bounds(
                q_lower, q_upper, q_rope, query_scale, SCALE
            )
            sentinel_ok = observed_sentinel_ok(
                KV, Indices, row, start, end, TOKENS, TOPK, CHUNK
            )
            k_max = tl.full((BN,), 0.0, tl.float32)
            kr_max = tl.full((BN, 64), 0.0, tl.float32)
            p_sum = tl.full((64, BN), 0.0, tl.float32)
            weighted = tl.full((64, BN), 0.0, tl.float32)
        else:
            tle.gpu.copy(Q + q_offset, QL, (64, 256))
            tle.gpu.copy(Q + q_offset + 256, QU, (64, 256))
            tle.gpu.copy(QR + qr_offset, QRope, (64, 64))
            q_scale = tl.load(QS + row * 64 + heads) * SCALE
            # A separate check has already bounded the scores for this path.
            q_scale *= 1.4426950408889634
            p_sum = tl.full((64, BN), 0.0, tl.float32)
        for page in range(pages):
            slot = page & 1
            phase = (page // 2) & 1
            # The consumer owns this slot until both PV halves finish.
            tle.gpu.barrier_wait(Empty[slot], phaseIdx=phase)
            offsets = start + page * BN + cols
            index_ptr = Indices + row * TOPK + offsets
            if SPLITS * CHUNK == TOPK and CHUNK % BN == 0:
                idx = tl.load(index_ptr)
            else:
                idx = tl.load(index_ptr, offsets < end, -1)
            valid = (idx >= 0) & (idx < TOKENS)
            safe = tl.where(valid, idx, 0).to(tl.int64)
            kv = KV + safe[:, None] * 512 + dims[None, :]
            rope = KR + safe[:, None] * 64 + rope_dims[None, :]
            tle.gpu.copy(kv, KL.slot(slot), (BN, 256))
            if OBSERVED:
                k_scale = tl.where(valid, tl.load(KS + safe), 0.0)
                scale_valid = (k_scale >= 2.0**-20) & (k_scale <= 2.0**-9)
                k_scale = tl.where(scale_valid, k_scale, 0.0)
                k_max = tl.maximum(k_max, k_scale)
                key_rope = tl.load(rope, valid[:, None], 0.0)
                kr_max = tl.maximum(
                    kr_max, tl.abs(key_rope.to(tl.float32)) * k_scale[:, None]
                )
                tl.store(tle.gpu.local_ptr(KRope.slot(slot)), key_rope)
            else:
                tle.gpu.copy(rope, KRope, (BN, 64))
                k_scale = tl.where(valid, tl.load(KS + safe), 0.0)
            if OBSERVED:
                tle.gpu.barrier_wait(UpperEmpty, phaseIdx=slot)
                tle.gpu.copy(kv + 256, KU, (BN, 256))
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Local, phaseIdx=slot)
                tle.gpu.barrier_wait(Local, phaseIdx=slot)
                scores = tle.gpu.wgmma(QL, KL.slot(slot), trans_b=True)
                scores = tle.gpu.wgmma(
                    QRope, KRope.slot(slot), acc=scores, trans_b=True
                )
                scores = tle.gpu.wgmma(QU, KU, trans_b=True, acc=scores)
                scores = tle.gpu.wgmma_wait(0, scores)
            else:
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Local, phaseIdx=0)
                tle.gpu.barrier_wait(Local, phaseIdx=0)
                scores = tle.gpu.wgmma(QL, KL.slot(slot), trans_b=True)
                scores = tle.gpu.wgmma(QRope, KRope, acc=scores, trans_b=True)
                scores = tle.gpu.wgmma_wait(0, scores)
                # The other role may still be using the previous upper half.
                tle.gpu.barrier_wait(UpperEmpty, phaseIdx=slot)
                tle.gpu.copy(kv + 256, KU, (BN, 256))
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Local, phaseIdx=1)
                tle.gpu.barrier_wait(Local, phaseIdx=1)
                scores = tle.gpu.wgmma(QU, KU, trans_b=True, acc=scores)
                scores = tle.gpu.wgmma_wait(0, scores)
            scores *= q_scale[:, None] * k_scale[None, :]
            if OBSERVED:
                logit_valid = tl.abs(scores) <= 0.625
                accepted = logit_valid & scale_valid[None, :]
            else:
                accepted = True
            exponent = tl.where(valid[None, :] & accepted, scores, float("-inf"))
            if OBSERVED:
                exponent *= 1.4426950408889634
            p = vector_exp2(exponent, 64, BN)
            p_sum += p
            scaled = p * (k_scale[None, :] * 16384.0)
            if OBSERVED:
                weighted += tl.where(valid[None, :] & ~accepted, float("inf"), scaled)
            high = scaled.to(tl.float8e4nv)
            low = (scaled - high.to(tl.float32)).to(tl.float8e4nv)
            tl.store(tle.gpu.local_ptr(PH), high)
            tl.store(tle.gpu.local_ptr(PL), low)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Full[slot], phaseIdx=phase)
        total = tl.sum(p_sum, 1)
        if OBSERVED:
            eligible = observed_eligible(
                sentinel_ok,
                query_valid,
                q_bound,
                qr_bound,
                k_max,
                tl.max(tl.max(kr_max, 0), 0),
                weighted,
                total,
            )
            tl.store(Skip + split_index, ~eligible)
        if SPLITS == 1:
            inverse = tl.div_rn(1.0 / 16384.0, tl.where(total > 0.0, total, 1.0))
            tl.store(
                LSE + row * 64 + heads,
                tl.where(total > 0.0, tl.log(total), float("inf")),
            )
            tl.store(tle.gpu.local_ptr(Inv), inverse)
            lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Norm, phaseIdx=0)
        else:
            # The caller passes its split stats in place of LSE.
            stats = (split_index * 64 + heads) * 2
            tl.store(LSE + stats, tl.where(total > 0.0, 0.0, float("-inf")))
            tl.store(LSE + stats + 1, total)


@triton.jit
def pipeline_plan(
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BN: tl.constexpr,
    FULL_ROWS: tl.constexpr,
):
    if FULL_ROWS:
        # Long programs occupy the first wave. The short programs fill the tail.
        slot = tl.program_id(0).to(tl.int64)
        row, full, start, end = mixed_v32_span(slot, TOPK, SPLITS, CHUNK, FULL_ROWS)
        pages = tl.where(full, TOPK // BN, CHUNK // BN)
    else:
        row = tl.program_id(0).to(tl.int64)
        split = tl.program_id(2)
        slot = row * SPLITS + split
        full = SPLITS == 1
        start = 0 if SPLITS == 1 else split * CHUNK
        end = tl.minimum(start + CHUNK, TOPK)
        pages = (
            tl.cdiv(CHUNK, BN)
            if SPLITS * CHUNK == TOPK
            else tl.cdiv(tl.maximum(end - start, 0), BN)
        )
    return slot, row, full, start, end, pages


@triton.jit
def qualify_v32_scales(
    KV,
    KS,
    Indices,
    Cutoff,
    Skip,
    Again,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    FULL_ROWS: tl.constexpr,
):
    if FULL_ROWS:
        slot, row, _, start, end, _ = pipeline_plan(TOPK, SPLITS, CHUNK, 128, FULL_ROWS)
    else:
        row = tl.program_id(0).to(tl.int64)
        split = tl.program_id(1)
        slot = row * SPLITS + split
        start = split * CHUNK
        end = tl.minimum(start + CHUNK, TOPK)
    span: tl.constexpr = TOPK if FULL_ROWS else CHUNK
    pos = start + tl.arange(0, triton.next_power_of_2(span))
    idx = tl.load(Indices + row * TOPK + pos, pos < end, -1)
    valid = (idx >= 0) & (idx < TOKENS)
    safe = tl.where(valid, idx, 0)
    scale = tl.load(KS + safe, valid, 0.0)
    allowed = (~valid) | ((scale >= 2.0**-20) & (scale <= 2.0**-9))
    scales_ok = tl.min(allowed.to(tl.int32), 0) != 0
    dev = tl.where(valid, scale - 0.0010499, 0.0)
    excess = tl.sum(tl.maximum(dev, 0.0), 0)
    slack = tl.sum(tl.maximum(-dev, 0.0), 0)
    # A cap c gives a weighted-scale certificate when exp(2c) * excess <= slack.
    # Keep room for exponent and reduction rounding as well as the final PV margin.
    cutoff = tl.where(
        excess * (1.00001 * 3.4903429574618414) <= slack,
        0.625,
        tl.where(
            excess * (1.00001 * 2.718281828459045) <= slack,
            0.5,
            tl.where(excess * (1.00001 * 1.6487212707001282) <= slack, 0.25, 0.0),
        ),
    )
    safe_placeholder = observed_sentinel_ok(
        KV, Indices, row, start, end, TOKENS, TOPK, span
    )
    cutoff = tl.where(scales_ok & safe_placeholder, cutoff, 0.0)
    tl.store(Cutoff + slot, cutoff)
    tl.store(Skip + slot, cutoff == 0.0)
    tl.store(Again + slot, cutoff == 0.0)


@triton.jit
def v32_consume(
    Y,
    TailY,
    Skip,
    KL,
    KU,
    PH,
    PL,
    Inv,
    Full,
    Empty,
    UpperEmpty,
    Norm,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BN: tl.constexpr,
    OBSERVED: tl.constexpr,
    Cutoff,
    Again,
    SCALE_MODE: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    STATIC_SPLITS: tl.constexpr,
    HighP=None,
    PEEL_LAST: tl.constexpr = False,
):
    # Both mixed schedules get constant page counts. Only one runs per CTA.
    for part in tl.static_range(2 if STATIC_SPLITS else 1):
        if not STATIC_SPLITS or (tl.program_id(0) < FULL_ROWS) == (part == 0):
            if STATIC_SPLITS:
                split_index = tl.program_id(0).to(tl.int64)
                full_row: tl.constexpr = part == 0
                if full_row:
                    row = split_index
                    start: tl.constexpr = 0
                    end: tl.constexpr = TOPK
                else:
                    row, start, end = _mixed_v32_tail_span(
                        split_index - FULL_ROWS, SPLITS, CHUNK, FULL_ROWS
                    )
                pages: tl.constexpr = (TOPK if full_row else CHUNK) // BN
            else:
                split_index, row, full_row, start, end, pages = pipeline_plan(
                    TOPK, SPLITS, CHUNK, BN, FULL_ROWS
                )
            if preloaded_active(Skip, Cutoff, Again, split_index, OBSERVED, SCALE_MODE):
                heads = tl.arange(0, 64)
                dims = tl.arange(0, 256)
                y_l = tl.full((64, 256), 0.0, tl.float32)
                y_u = tl.full((64, 256), 0.0, tl.float32)
                for is_last in tl.static_range(2 if PEEL_LAST else 1):
                    first = tl.maximum(pages - 1, 0) if is_last else 0
                    stop = (
                        pages if is_last or not PEEL_LAST else tl.maximum(pages - 1, 0)
                    )
                    for page in range(first, stop):
                        slot = page & 1
                        phase = (page // 2) & 1
                        lower = KL.slot(slot)
                        if HighP is not None:
                            tle.gpu.barrier_wait(HighP[slot], phaseIdx=phase)
                            y_l = tle.gpu.wgmma(PH, lower, acc=y_l)
                            y_u = tle.gpu.wgmma(PH, KU, acc=y_u)
                            y_l = tle.gpu.wgmma_wait(0, y_l)
                            y_u = tle.gpu.wgmma_wait(0, y_u)
                            tle.gpu.barrier_wait(Full[slot], phaseIdx=phase)
                            y_l = tle.gpu.wgmma(PL, lower, acc=y_l)
                            y_u = tle.gpu.wgmma(PL, KU, acc=y_u)
                        else:
                            tle.gpu.barrier_wait(Full[slot], phaseIdx=phase)
                            y_l = tle.gpu.wgmma(PH, lower, acc=y_l)
                            y_l = tle.gpu.wgmma(PL, lower, acc=y_l)
                            y_u = tle.gpu.wgmma(PH, KU, acc=y_u)
                            y_u = tle.gpu.wgmma(PL, KU, acc=y_u)
                        y_l = tle.gpu.wgmma_wait(0, y_l)
                        y_u = tle.gpu.wgmma_wait(0, y_u)
                        # The loader only needs the release signals when more pages follow.
                        if not PEEL_LAST or not is_last:
                            tle.gpu.barrier_arrive(UpperEmpty, phaseIdx=slot)
                            tle.gpu.barrier_arrive(Empty[slot], phaseIdx=phase)
                if full_row:
                    y_offset = (row * 64 + heads[:, None]) * 512 + dims[None, :]
                    tle.gpu.barrier_wait(Norm, phaseIdx=0)
                    inv = tl.load(tle.gpu.local_ptr(Inv), volatile=True)
                    tl.store(Y + y_offset, y_l * inv[:, None])
                    tl.store(Y + y_offset + 256, y_u * inv[:, None])
                else:
                    y_offset = (
                        (split_index - FULL_ROWS) * 64 + heads[:, None]
                    ) * 512 + dims[None, :]
                    partial = TailY if FULL_ROWS else Y
                    # Partial outputs use the same gain as the full normalization.
                    tl.store(partial + y_offset, y_l * (1.0 / 16384.0))
                    tl.store(partial + y_offset + 256, y_u * (1.0 / 16384.0))


@triton.jit
def scale_bound_terms(scale, valid, factor):
    limit: tl.constexpr = 0.00105
    scale_ok = (scale >= 2.0**-20) & (scale <= 2.0**-9)
    value = tl.where(valid, scale, limit)
    excess = tl.maximum(value - limit, 0.0)
    weighted = tl.fma(excess, factor - 1.0, value)
    return tl.where(~valid | scale_ok, weighted, float("inf"))


@triton.jit
def load_pages(
    KV,
    KR,
    KS,
    Indices,
    Skip,
    KL,
    KU,
    KRope,
    Cache,
    Present,
    RopeBound,
    Peak,
    Empty,
    UpperEmpty,
    LowReady,
    HighReady,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BN: tl.constexpr,
    OBSERVED: tl.constexpr,
    PEAKS: tl.constexpr,
    Cutoff,
    Again,
    SCALE_MODE: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    STATIC_SPLITS: tl.constexpr,
    DIRECT_COPY: tl.constexpr,
    ScaleOK,
    ScaleReady,
    SMALL_SCALE_CHECK: tl.constexpr,
    LARGE_SCALE_CHECK: tl.constexpr,
    SINGLE_ROPE: tl.constexpr,
    RopeEmpty,
):
    PREFETCH_UPPER: tl.constexpr = DIRECT_COPY == 2 and (
        LARGE_SCALE_CHECK or SMALL_SCALE_CHECK
    )
    # Both mixed schedules get constant page counts. Only one runs per CTA.
    for part in tl.static_range(2 if STATIC_SPLITS else 1):
        if not STATIC_SPLITS or (tl.program_id(0) < FULL_ROWS) == (part == 0):
            if STATIC_SPLITS:
                split_index = tl.program_id(0).to(tl.int64)
                full_row: tl.constexpr = part == 0
                if full_row:
                    row = split_index
                    start: tl.constexpr = 0
                    end: tl.constexpr = TOPK
                else:
                    row, start, end = _mixed_v32_tail_span(
                        split_index - FULL_ROWS, SPLITS, CHUNK, FULL_ROWS
                    )
                pages: tl.constexpr = (TOPK if full_row else CHUNK) // BN
            else:
                split_index, row, full_row, start, end, pages = pipeline_plan(
                    TOPK, SPLITS, CHUNK, BN, FULL_ROWS
                )
            if preloaded_active(Skip, Cutoff, Again, split_index, OBSERVED, SCALE_MODE):
                cols = tl.arange(0, BN)
                dims = tl.arange(0, 256)
                rd = tl.arange(0, 64)
                if SMALL_SCALE_CHECK:
                    scale_bound = tl.full((BN,), 0.0, tl.float32)
                if LARGE_SCALE_CHECK:
                    scale_sum = tl.full((BN,), 0.0, tl.float32)
                    excess_sum = tl.full((BN,), 0.0, tl.float32)
                for page in range(pages):
                    slot = page & 1
                    phase = (page // 2) & 1
                    rope_tile = KRope if SINGLE_ROPE else KRope.slot(slot)
                    tle.gpu.barrier_wait(Empty[slot], phaseIdx=phase)
                    pos = start + page * BN + cols
                    if FULL_ROWS or (
                        OBSERVED and SPLITS * CHUNK == TOPK and CHUNK % BN == 0
                    ):
                        idx = tl.load(Indices + row * TOPK + pos)
                    else:
                        idx = tl.load(Indices + row * TOPK + pos, pos < end, -1)
                    valid = (idx >= 0) & (idx < TOKENS)
                    safe = tl.where(valid, idx, 0).to(tl.int64)
                    if SMALL_SCALE_CHECK:
                        # Fetch the scale before scheduling the two bulk copies.
                        raw_scale = tl.load(KS + safe)
                    kv = KV + safe[:, None] * 512 + dims[None, :]
                    rope = KR + safe[:, None] * 64 + rd[None, :]
                    if DIRECT_COPY:
                        tl.static_assert(
                            BN == 128 and TOPK % BN == 0 and CHUNK % BN == 0
                        )
                        zero = tl.full((), 0, tl.int32)
                        low_addr = tle.gpu.local_ptr(KL.slot(slot), (zero, zero))
                        ids = (Indices + row * TOPK + start + page * BN).to(tl.uint64)
                        if DIRECT_COPY == 2:
                            rope_addr = (
                                tle.gpu.local_ptr(rope_tile, (zero, zero))
                                .to(tl.uint64)
                                .to(tl.uint32)
                            )
                            rope_ptr = KR.to(tl.uint64)
                        else:
                            rope_addr = tl.full((), 0, tl.uint32)
                            rope_ptr = tl.full((), 0, tl.uint64)
                        v32_copy_page(
                            low_addr.to(tl.uint64).to(tl.uint32),
                            KV.to(tl.uint64),
                            ids,
                            TOKENS,
                            rope_addr,
                            rope_ptr,
                        )
                    else:
                        tle.gpu.copy(kv, KL.slot(slot), (BN, 256))
                    if not SMALL_SCALE_CHECK:
                        raw_scale = tl.load(KS + safe)
                    if LARGE_SCALE_CHECK:
                        scale_ok = (raw_scale >= 2.0**-20) & (raw_scale <= 2.0**-9)
                        marked = tl.where(scale_ok, raw_scale, float("inf"))
                        ks = tl.where(valid, marked, -1.0)
                    else:
                        ks = tl.where(valid, raw_scale, 0.0)
                    if SMALL_SCALE_CHECK:
                        # Covers |logit| <= 0.15625 and the relative polynomial error.
                        scale_bound += scale_bound_terms(ks, valid, 1.3672)
                    tl.store(tle.gpu.local_ptr(Cache.slot(slot)), ks)
                    if OBSERVED and SCALE_MODE != 1:
                        tl.store(
                            tle.gpu.local_ptr(Present.slot(slot)), valid.to(tl.int8)
                        )
                    if DIRECT_COPY != 2:
                        if SINGLE_ROPE:
                            tle.gpu.barrier_wait(RopeEmpty, phaseIdx=slot)
                        tle.gpu.copy(rope, rope_tile, (BN, 64))
                    if PEAKS:
                        peak = tl.load(Peak + safe, valid, 0.0)
                        tl.store(tle.gpu.local_ptr(RopeBound.slot(slot)), peak)
                    if PREFETCH_UPPER:
                        upper_data = tl.load(
                            kv + 256, cache_modifier=".ca", volatile=True
                        )
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(LowReady[slot], phaseIdx=phase)
                    tle.gpu.barrier_wait(UpperEmpty, phaseIdx=slot)
                    if DIRECT_COPY == 1:
                        high_addr = tle.gpu.local_ptr(KU, (zero, zero))
                        v32_copy_page(
                            high_addr.to(tl.uint64).to(tl.uint32),
                            (KV + 256).to(tl.uint64),
                            ids,
                            TOKENS,
                            rope_addr,
                            rope_ptr,
                        )
                    elif PREFETCH_UPPER:
                        tl.store(tle.gpu.local_ptr(KU), upper_data)
                    else:
                        tle.gpu.copy(kv + 256, KU, (BN, 256))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(HighReady, phaseIdx=slot)
                    if LARGE_SCALE_CHECK:
                        scale_terms = tl.where(valid, ks, 0.00105)
                        scale_sum += scale_terms
                        excess_sum += tl.maximum(scale_terms - 0.00105, 0.0)
                if SMALL_SCALE_CHECK:
                    allowed = tl.sum(scale_bound, 0) * 1.0002 <= pages * BN * 0.00105
                    addr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 0, tl.int32),))
                    tl.store(addr, allowed.to(tl.int32))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(ScaleReady, phaseIdx=0)
                if LARGE_SCALE_CHECK:
                    base_ptr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 0, tl.int32),))
                    extra_ptr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 1, tl.int32),))
                    tl.store(base_ptr, tl.sum(scale_sum, 0))
                    tl.store(extra_ptr, tl.sum(excess_sum, 0))
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(ScaleReady, phaseIdx=0)


@triton.jit
def fold_probability_pairs(values, BN: tl.constexpr):
    # The elements differing in token bit 3 are in the same SQMMA lane.
    view = tl.reshape(values, (64, BN // 32, 2, 1, 2, 8))
    lanes = tl.permute(view, (0, 1, 3, 5, 2, 4))
    low_group, high_group = tl.split(lanes)
    return tl.reshape(low_group + high_group, (64, BN // 2))


@triton.jit
def produce_preloaded(
    Q,
    QR,
    KV,
    QS,
    Indices,
    LSE,
    TailStats,
    Skip,
    QL,
    QU,
    QRope,
    KL,
    KU,
    KRope,
    PH,
    PL,
    Inv,
    Full,
    Local,
    Norm,
    Cache,
    Present,
    RopeBound,
    LowReady,
    HighReady,
    HighP,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    OBSERVED: tl.constexpr,
    PEAKS: tl.constexpr,
    Cutoff,
    Again,
    SCALE_MODE: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    STATIC_SPLITS: tl.constexpr,
    CERTIFIED_POLY: tl.constexpr,
    ScaleOK,
    ScaleReady,
    UpperEmpty,
    Coeff,
    BadRows,
    BadCount,
    SMALL_SCALE_CHECK: tl.constexpr,
    LARGE_SCALE_CHECK: tl.constexpr,
    LOCAL_QUERY_NORM: tl.constexpr,
    TailDenominators,
    HAS_TAIL_DENOMINATORS: tl.constexpr,
    STATIC_SQ64: tl.constexpr,
    SINGLE_ROPE: tl.constexpr,
    RopeEmpty,
):
    # Both mixed schedules get constant page counts. Only one runs per CTA.
    for part in tl.static_range(2 if STATIC_SPLITS else 1):
        if not STATIC_SPLITS or (tl.program_id(0) < FULL_ROWS) == (part == 0):
            if STATIC_SPLITS:
                split_index = tl.program_id(0).to(tl.int64)
                full_row: tl.constexpr = part == 0
                if full_row:
                    row = split_index
                    start: tl.constexpr = 0
                    end: tl.constexpr = TOPK
                else:
                    row, start, end = _mixed_v32_tail_span(
                        split_index - FULL_ROWS, SPLITS, CHUNK, FULL_ROWS
                    )
                pages: tl.constexpr = (TOPK if full_row else CHUNK) // BN
            else:
                split_index, row, full_row, start, end, pages = pipeline_plan(
                    TOPK, SPLITS, CHUNK, BN, FULL_ROWS
                )
            if preloaded_active(Skip, Cutoff, Again, split_index, OBSERVED, SCALE_MODE):
                SUM_COLS: tl.constexpr = BN // 2 if FULL_ROWS else BN
                DEFER_TAIL_SUM: tl.constexpr = HAS_TAIL_DENOMINATORS and part == 1
                if DEFER_TAIL_SUM:
                    tl.static_assert(SMALL_SCALE_CHECK and SUM_COLS == 64)

                DEFER_DENOMINATOR: tl.constexpr = SMALL_SCALE_CHECK or LARGE_SCALE_CHECK
                # Keep the original exponent for the short branch of the static split plan.
                observed_poly: tl.constexpr = (
                    TOPK <= 2048
                    and SCALE_MODE != 2
                    and (not STATIC_SPLITS or part == 0)
                )
                DECOUPLE_LOGIT_CHECK: tl.constexpr = SMALL_SCALE_CHECK and observed_poly
                logit_cap = tl.load(Cutoff + split_index) if SCALE_MODE == 1 else 0.625
                heads = tl.arange(0, 64)
                dims = tl.arange(0, 256)
                rope_dims = tl.arange(0, 64)
                qr_offset = (row * 64 + heads[:, None]) * 64 + rope_dims[None, :]
                q_offset = (row * 64 + heads[:, None]) * 512 + dims[None, :]
                if OBSERVED:
                    q_lower = tl.load(Q + q_offset)
                    q_upper = tl.load(Q + q_offset + 256)
                    q_rope = tl.load(QR + qr_offset)
                    tl.store(tle.gpu.local_ptr(QL), q_lower)
                    tl.store(tle.gpu.local_ptr(QU), q_upper)
                    tl.store(tle.gpu.local_ptr(QRope), q_rope)
                    query_scale = tl.load(QS + row * 64 + heads)
                    q_scale, query_valid, q_bound, qr_bound = observed_query_bounds(
                        q_lower, q_upper, q_rope, query_scale, SCALE
                    )
                    k_max = tl.full((BN,), 0.0, tl.float32)
                    kr_max = tl.full((BN,), 0.0, tl.float32)
                    p_sum = tl.full((64, SUM_COLS), 0.0, tl.float32)
                    if SCALE_MODE != 1 and not SMALL_SCALE_CHECK:
                        weighted = tl.full((64, SUM_COLS), 0.0, tl.float32)
                else:
                    query_scale = tl.load(QS + row * 64 + heads)
                    if LOCAL_QUERY_NORM:
                        k_max = tl.load(Coeff)
                        use_local_norm = k_max <= 1024.0
                        lower = tl.load(Q + q_offset)
                        upper = tl.load(Q + q_offset + 256)
                        rope = tl.load(QR + qr_offset)
                        tl.store(tle.gpu.local_ptr(QL), lower)
                        tl.store(tle.gpu.local_ptr(QU), upper)
                        tl.store(tle.gpu.local_ptr(QRope), rope)
                    else:
                        tle.gpu.copy(Q + q_offset, QL, (64, 256))
                        tle.gpu.copy(Q + q_offset + 256, QU, (64, 256))
                        tle.gpu.copy(QR + qr_offset, QRope, (64, 64))
                    q_scale = query_scale * SCALE
                    # A separate check has already bounded the scores for this path.
                    if not CERTIFIED_POLY:
                        q_scale *= 1.4426950408889634
                    p_sum = tl.full((64, SUM_COLS), 0.0, tl.float32)
                lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(Local, phaseIdx=0)
                tle.gpu.barrier_wait(Local, phaseIdx=0)
                if LARGE_SCALE_CHECK:
                    if LOCAL_QUERY_NORM:
                        norm_ok = tl.full((), True, tl.int1)
                        factor = tl.full((), 0.0, tl.float32)
                        if use_local_norm:
                            lower_fp = lower.to(tl.float32)
                            upper_fp = upper.to(tl.float32)
                            rope_fp = rope.to(tl.float32)
                            content_square = tl.sum(
                                tl.fma(lower_fp, lower_fp, upper_fp * upper_fp), 1
                            )
                            sq = (content_square + tl.sum(rope_fp * rope_fp, 1)) * (
                                query_scale * query_scale
                            )
                            valid_q = (
                                (sq >= 0.0)
                                & (query_scale >= 2.0**-20)
                                & (query_scale <= 2.0**-6)
                            )
                            q_max = tl.max(tl.where(valid_q, sq, float("inf")), 0)
                            cap = tl.sqrt(q_max) * tl.sqrt(k_max) * (1.001 * SCALE)
                            norm_ok = (cap <= 0.625) & (q_max <= 1024.0)
                            factor = tl.where(norm_ok, 1.00001 * tl.exp(2.0 * cap), 0.0)
                    else:
                        norm_ok = tl.full((), True, tl.int1)
                        factor = tl.load(Coeff + row)
                for page in range(pages):
                    slot = page & 1
                    phase = (page // 2) & 1
                    rope_tile = KRope if SINGLE_ROPE else KRope.slot(slot)
                    tle.gpu.barrier_wait(LowReady[slot], phaseIdx=phase)
                    ks = tl.load(tle.gpu.local_ptr(Cache.slot(slot)), volatile=True)
                    if LARGE_SCALE_CHECK:
                        scale_ok = (ks >= 2.0**-20) & (ks <= 2.0**-9)
                        ks = tl.where(scale_ok, ks, 0.0)
                    if OBSERVED:
                        if SCALE_MODE == 1:
                            valid = ks > 0.0
                            k_scale = ks
                        else:
                            valid = (
                                tl.load(
                                    tle.gpu.local_ptr(Present.slot(slot)), volatile=True
                                )
                                != 0
                            )
                            scale_valid = (ks >= 2.0**-20) & (ks <= 2.0**-9)
                            k_scale = tl.where(scale_valid, ks, 0.0)
                        k_max = tl.maximum(k_max, k_scale)
                    else:
                        k_scale = ks
                    scores = tle.gpu.wgmma(QL, KL.slot(slot), trans_b=True)
                    scores = tle.gpu.wgmma(QRope, rope_tile, acc=scores, trans_b=True)
                    if OBSERVED:
                        if PEAKS:
                            peak = tl.load(
                                tle.gpu.local_ptr(RopeBound.slot(slot)), volatile=True
                            )
                            if SCALE_MODE != 1:
                                peak = tl.where(scale_valid, peak, 0.0)
                        else:
                            key_rope = tl.load(
                                tle.gpu.local_ptr(rope_tile), volatile=True
                            )
                            if SCALE_MODE == 1:
                                # Unsigned magnitude bits preserve the order for finite FP16/BF16.
                                magnitude_bits = (
                                    key_rope.to(tl.uint16, bitcast=True) & 0x7FFF
                                )
                                peak_bits = tl.max(magnitude_bits, 1).to(tl.uint16)
                                magnitude = peak_bits.to(
                                    key_rope.dtype, bitcast=True
                                ).to(tl.float32)
                                magnitude = tl.where(
                                    magnitude < float("inf"), magnitude, float("inf")
                                )
                                peak = tl.where(valid, magnitude * k_scale, 0.0)
                            else:
                                magnitude = tl.where(
                                    valid[:, None], tl.abs(key_rope.to(tl.float32)), 0.0
                                )
                                peak = tl.max(magnitude, 1) * k_scale
                        kr_max = tl.maximum(kr_max, peak)
                    scores = tle.gpu.wgmma_wait(0, scores)
                    if SINGLE_ROPE:
                        tle.gpu.barrier_arrive(RopeEmpty, phaseIdx=slot)
                    tle.gpu.barrier_wait(HighReady, phaseIdx=slot)
                    scores = tle.gpu.wgmma(QU, KU, trans_b=True, acc=scores)
                    scores = tle.gpu.wgmma_wait(0, scores)
                    if OBSERVED:
                        scores *= q_scale[:, None] * k_scale[None, :]
                        if not DECOUPLE_LOGIT_CHECK:
                            # The observed path rejects logits outside the polynomial interval.
                            # The second pass keeps the original exponent and acceptance logic.
                            if observed_poly:
                                accepted = tl.abs(scores) <= tl.minimum(
                                    logit_cap, 0.1015625
                                )
                            else:
                                if SMALL_SCALE_CHECK:
                                    accepted = tl.abs(scores) <= tl.minimum(
                                        logit_cap, 0.15625
                                    )
                                else:
                                    accepted = tl.abs(scores) <= logit_cap
                            if SCALE_MODE != 1:
                                accepted &= scale_valid[None, :]
                        if observed_poly:
                            p = tl.fma(
                                scores,
                                tl.fma(0.5004297892252604, scores, 1.0012893676757812),
                                0.9999994458436655,
                            )
                            if DECOUPLE_LOGIT_CHECK:
                                p = tl.where(valid[None, :], p, 0.0)
                            else:
                                p = tl.where(valid[None, :] & accepted, p, 0.0)
                        else:
                            exponent = tl.where(
                                valid[None, :] & accepted, scores, float("-inf")
                            )
                            p = vector_exp2(exponent * 1.4426950408889634, 64, BN)
                    else:
                        # The norm certificate covers [-0.625, 0.625].
                        if CERTIFIED_POLY:
                            scores *= q_scale[:, None] * ks[None, :]
                            p = tl.fma(
                                scores,
                                tl.fma(
                                    scores,
                                    tl.fma(
                                        scores,
                                        tl.fma(
                                            scores,
                                            0.0424871223799997,
                                            0.17077562984900999,
                                        ),
                                        0.49987949041777147,
                                    ),
                                    0.9995974299900329,
                                ),
                                1.0000026187636524,
                            )
                            p = tl.where(ks[None, :] > 0.0, p, 0.0)
                        else:
                            exponent = tl.where(
                                ks[None, :] > 0.0,
                                scores * (q_scale[:, None] * ks[None, :]),
                                float("-inf"),
                            )
                            p = vector_exp2(exponent, 64, BN)
                    if not DEFER_DENOMINATOR:
                        if SCALE_MODE == 1:
                            p_terms = tl.where(
                                valid[None, :] & ~accepted, float("inf"), p
                            )
                        else:
                            p_terms = p
                        p_sum += (
                            fold_probability_pairs(p_terms, BN)
                            if FULL_ROWS
                            else p_terms
                        )
                    scaled = p * (k_scale[None, :] * 16384.0)
                    if OBSERVED and SCALE_MODE != 1 and not SMALL_SCALE_CHECK:
                        weighted_terms = tl.where(
                            valid[None, :] & ~accepted, float("inf"), scaled
                        )
                        weighted += (
                            fold_probability_pairs(weighted_terms, BN)
                            if FULL_ROWS
                            else weighted_terms
                        )
                    if DECOUPLE_LOGIT_CHECK:
                        # Rejected logits set an infinite denominator and replay.
                        accepted = scale_valid[None, :] & (
                            tl.abs(scores) <= tl.minimum(logit_cap, 0.1015625)
                        )
                    high = scaled.to(tl.float8e4nv)
                    tl.store(tle.gpu.local_ptr(PH), high)
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(HighP[slot], phaseIdx=phase)
                    low = (scaled - high.to(tl.float32)).to(tl.float8e4nv)
                    tl.store(tle.gpu.local_ptr(PL), low)
                    lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(Full[slot], phaseIdx=phase)
                    # Keep these reductions off the probability publication path.
                    if DEFER_DENOMINATOR:
                        terms = p
                        if SMALL_SCALE_CHECK:
                            terms = tl.where(
                                valid[None, :] & ~accepted, float("inf"), p
                            )
                        p_sum += (
                            fold_probability_pairs(terms, BN) if FULL_ROWS else terms
                        )
                if not DEFER_TAIL_SUM:
                    total = tl.sum(p_sum, 1)
                if SMALL_SCALE_CHECK:
                    if full_row:
                        inverse = tl.div_rn(
                            1.0 / 16384.0, tl.where(total > 0.0, total, 1.0)
                        )
                        tl.store(tle.gpu.local_ptr(Inv), inverse)
                        lma_wait(tl.full((), 0, tl.int32))
                        tle.gpu.barrier_arrive(Norm, phaseIdx=0)
                if OBSERVED and SCALE_MODE != 1:
                    sentinel_ok = observed_sentinel_ok(
                        KV,
                        Indices,
                        row,
                        start,
                        end,
                        TOKENS,
                        TOPK,
                        TOPK if FULL_ROWS else CHUNK,
                    )
                if SMALL_SCALE_CHECK:
                    tle.gpu.barrier_wait(ScaleReady, phaseIdx=0)
                    addr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 0, tl.int32),))
                    scale_ok = tl.load(addr, volatile=True) != 0
                    if DEFER_TAIL_SUM:
                        # The next kernel checks the complete probability sum.
                        denominator_ok = scale_ok
                    else:
                        p_max = tl.max(total, 0)
                        denominator_ok = (p_max < float("inf")) & (
                            scale_ok | (p_max == 0.0)
                        )
                    error = observed_score_error(
                        q_bound, qr_bound, k_max, tl.max(kr_max, 0)
                    )
                    eligible = (
                        sentinel_ok
                        & (tl.min(query_valid.to(tl.int32), 0) != 0)
                        & (1.001 * error <= 8.0e-5)
                        & denominator_ok
                    )
                    tl.store(Skip + split_index, ~eligible)
                elif OBSERVED:
                    if SCALE_MODE == 1:
                        error = observed_score_error(
                            q_bound, qr_bound, k_max, tl.max(kr_max, 0)
                        )
                        eligible = (
                            (tl.max(total, 0) < float("inf"))
                            & (tl.min(query_valid.to(tl.int32), 0) != 0)
                            & (1.001 * error <= 8.0e-5)
                        )
                        tl.store(Again + split_index, ~eligible)
                    else:
                        eligible = observed_eligible(
                            sentinel_ok,
                            query_valid,
                            q_bound,
                            qr_bound,
                            k_max,
                            tl.max(kr_max, 0),
                            weighted,
                            total,
                        )
                    tl.store(Skip + split_index, ~eligible)
                if full_row:
                    if not SMALL_SCALE_CHECK:
                        inverse = tl.div_rn(
                            1.0 / 16384.0, tl.where(total > 0.0, total, 1.0)
                        )
                        tl.store(tle.gpu.local_ptr(Inv), inverse)
                        lma_wait(tl.full((), 0, tl.int32))
                        tle.gpu.barrier_arrive(Norm, phaseIdx=0)
                    tl.store(
                        LSE + row * 64 + heads,
                        tl.where(total > 0.0, tl.log(total), float("inf")),
                    )
                else:
                    if DEFER_TAIL_SUM:
                        columns = tl.arange(0, 4)
                        row_base = (split_index - FULL_ROWS) * 64 + heads
                        offset = row_base[:, None] * 4 + columns[None, :]
                        group_sum = tl.sum(tl.reshape(p_sum, (64, 4, 16)), 2)
                        tl.store(TailDenominators + offset, group_sum)
                    else:
                        stats = (split_index - FULL_ROWS) * 128 + heads * 2
                        split_stats = TailStats if FULL_ROWS else LSE
                        tl.store(
                            split_stats + stats,
                            tl.where(total > 0.0, 0.0, float("-inf")),
                        )
                        tl.store(split_stats + stats + 1, total)
                if LARGE_SCALE_CHECK:
                    tle.gpu.barrier_wait(ScaleReady, phaseIdx=0)
                    base_ptr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 0, tl.int32),))
                    extra_ptr = tle.gpu.local_ptr(ScaleOK, (tl.full((), 1, tl.int32),))
                    bias = tl.load(base_ptr, volatile=True)
                    excess = tl.load(extra_ptr, volatile=True)
                    weighted_scale = tl.fma(excess, factor - 1.0, bias)
                    if not norm_ok or (
                        factor > 0.0
                        and not (weighted_scale * 1.00021 <= TOPK * 0.00105)
                    ):
                        # PV also reads Skip on entry. Wait for its last release before changing it.
                        tle.gpu.barrier_wait(UpperEmpty, phaseIdx=pages & 1)
                        tl.store(Skip + split_index, True)
                        position = tl.atomic_add(BadCount, 1, sem="relaxed")
                        tl.store(BadRows + position, row)


@triton.jit
def sparse_fp8_v32_pipeline(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Y,
    LSE,
    Skip,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    WP: tl.constexpr,
    WC: tl.constexpr,
    OBSERVED: tl.constexpr,
    Peak=None,
    PEAKS: tl.constexpr = False,
    PRELOAD: tl.constexpr = False,
    Cutoff=None,
    Again=None,
    SCALE_MODE: tl.constexpr = 0,
    TailY=None,
    TailStats=None,
    FULL_ROWS: tl.constexpr = 0,
    STATIC_SPLITS: tl.constexpr = False,
    DIRECT_COPY: tl.constexpr = 0,
    CERTIFIED_POLY: tl.constexpr = False,
    STATIC_SQ64: tl.constexpr = False,
    Coeff=None,
    BadRows=None,
    BadCount=None,
    LOCAL_QUERY_NORM: tl.constexpr = False,
    TailDenominators=None,
    LOADER_SCALE_CHECK: tl.constexpr = False,
):
    SMALL_SCALE_CHECK: tl.constexpr = (
        (LOADER_SCALE_CHECK or (STATIC_SQ64 and STATIC_SPLITS))
        and OBSERVED
        and SCALE_MODE == 0
        and TOPK == 2048
    )
    LARGE_SCALE_CHECK: tl.constexpr = Coeff is not None
    SINGLE_ROPE: tl.constexpr = (
        STATIC_SQ64 and STATIC_SPLITS and PRELOAD and DIRECT_COPY == 0
    )
    HAS_TAIL_DENOMINATORS: tl.constexpr = TailDenominators is not None
    if HAS_TAIL_DENOMINATORS:
        tl.static_assert(SMALL_SCALE_CHECK and FULL_ROWS > 0)
    if LOCAL_QUERY_NORM:
        tl.static_assert(LARGE_SCALE_CHECK)
    if LARGE_SCALE_CHECK:
        tl.static_assert(PRELOAD and not OBSERVED and SPLITS == 1 and TOPK == 2048)
    dtype: tl.constexpr = QR.dtype.element_ty
    ql = tle.gpu.alloc((64, 256), tl.float8e4nv, nv_mma_shared_layout=True)
    qu = tle.gpu.alloc((64, 256), tl.float8e4nv, nv_mma_shared_layout=True)
    qr = tle.gpu.alloc((64, 64), dtype, nv_mma_shared_layout=True)
    kl = tle.gpu.alloc((2, BN, 256), tl.float8e4nv, nv_mma_shared_layout=True)
    ku = tle.gpu.alloc((BN, 256), tl.float8e4nv, nv_mma_shared_layout=True)
    rope_shape: tl.constexpr = (
        (2, BN, 64) if (OBSERVED or PRELOAD) and not SINGLE_ROPE else (BN, 64)
    )
    kr = tle.gpu.alloc(rope_shape, dtype, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((64, BN), tl.float8e4nv, nv_mma_shared_layout=True)
    inv = tle.gpu.alloc((64,), tl.float32, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barriers(2, arrive_count=WP)
    empty = tle.gpu.alloc_barriers(2, arrive_count=WC, init=tle.gpu.READY)
    upper_empty = tle.gpu.alloc_barrier(arrive_count=WC, init=tle.gpu.READY)
    local = tle.gpu.alloc_barrier(arrive_count=WP)
    norm = tle.gpu.alloc_barrier(arrive_count=WP)
    if PRELOAD:
        cache = tle.gpu.alloc((2, BN), tl.float32, nv_mma_shared_layout=False)
        if OBSERVED and SCALE_MODE != 1:
            present = tle.gpu.alloc((2, BN), tl.int8, nv_mma_shared_layout=False)
        else:
            present = cache
        if PEAKS:
            rope_bound = tle.gpu.alloc((2, BN), tl.float32, nv_mma_shared_layout=False)
        else:
            rope_bound = cache
        rope_empty = (
            tle.gpu.alloc_barrier(arrive_count=WP, init=tle.gpu.READY)
            if SINGLE_ROPE
            else local
        )
        low_ready = tle.gpu.alloc_barriers(2, arrive_count=4)
        high_ready = tle.gpu.alloc_barrier(arrive_count=4)
        high_p = tle.gpu.alloc_barriers(2, arrive_count=WP)
        if SMALL_SCALE_CHECK:
            scale_ok = tle.gpu.alloc((1,), tl.int32, nv_mma_shared_layout=False)
            scale_ready = tle.gpu.alloc_barrier(arrive_count=4)
        elif LARGE_SCALE_CHECK:
            scale_ok = tle.gpu.alloc((2,), tl.float32, nv_mma_shared_layout=False)
            scale_ready = tle.gpu.alloc_barrier(arrive_count=4)
        else:
            scale_ok = cache
            scale_ready = local
        tl.debug_barrier()
        tle.gpu.warp_specialize(
            [
                (
                    produce_preloaded,
                    (
                        Q,
                        QR,
                        KV,
                        QS,
                        Indices,
                        LSE,
                        TailStats if FULL_ROWS else LSE,
                        Skip,
                        ql,
                        qu,
                        qr,
                        kl,
                        ku,
                        kr,
                        ph,
                        pl,
                        inv,
                        full,
                        local,
                        norm,
                        cache,
                        present,
                        rope_bound,
                        low_ready,
                        high_ready,
                        high_p,
                        TOKENS,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        SCALE,
                        BN,
                        OBSERVED,
                        PEAKS,
                        Cutoff if SCALE_MODE != 0 else Skip,
                        Again if SCALE_MODE != 0 else Skip,
                        SCALE_MODE,
                        FULL_ROWS,
                        STATIC_SPLITS,
                        CERTIFIED_POLY,
                        scale_ok,
                        scale_ready,
                        upper_empty,
                        Coeff if LARGE_SCALE_CHECK else Skip,
                        BadRows if LARGE_SCALE_CHECK else Skip,
                        BadCount if LARGE_SCALE_CHECK else Skip,
                        SMALL_SCALE_CHECK,
                        LARGE_SCALE_CHECK,
                        LOCAL_QUERY_NORM,
                        TailDenominators if HAS_TAIL_DENOMINATORS else Skip,
                        HAS_TAIL_DENOMINATORS,
                        STATIC_SQ64,
                        SINGLE_ROPE,
                        rope_empty,
                    ),
                ),
                (
                    v32_consume,
                    (
                        Y,
                        TailY if FULL_ROWS else Y,
                        Skip,
                        kl,
                        ku,
                        ph,
                        pl,
                        inv,
                        full,
                        empty,
                        upper_empty,
                        norm,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        BN,
                        OBSERVED,
                        Cutoff if SCALE_MODE != 0 else Skip,
                        Again if SCALE_MODE != 0 else Skip,
                        SCALE_MODE,
                        FULL_ROWS,
                        STATIC_SPLITS,
                        high_p,
                        STATIC_SQ64 and STATIC_SPLITS,
                    ),
                ),
                (
                    load_pages,
                    (
                        KV,
                        KR,
                        KS,
                        Indices,
                        Skip,
                        kl,
                        ku,
                        kr,
                        cache,
                        present,
                        rope_bound,
                        Peak,
                        empty,
                        upper_empty,
                        low_ready,
                        high_ready,
                        TOKENS,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        BN,
                        OBSERVED,
                        PEAKS,
                        Cutoff if SCALE_MODE != 0 else Skip,
                        Again if SCALE_MODE != 0 else Skip,
                        SCALE_MODE,
                        FULL_ROWS,
                        STATIC_SPLITS,
                        DIRECT_COPY,
                        scale_ok,
                        scale_ready,
                        SMALL_SCALE_CHECK,
                        LARGE_SCALE_CHECK,
                        SINGLE_ROPE,
                        rope_empty,
                    ),
                ),
            ],
            worker_num_warps=[WC, 4],
            worker_num_regs=[128, 64],
        )
    else:
        tl.debug_barrier()
        tle.gpu.warp_specialize(
            [
                (
                    v32_produce,
                    (
                        Q,
                        QR,
                        KV,
                        KR,
                        QS,
                        KS,
                        Indices,
                        LSE,
                        Skip,
                        ql,
                        qu,
                        qr,
                        kl,
                        ku,
                        kr,
                        ph,
                        pl,
                        inv,
                        full,
                        empty,
                        upper_empty,
                        local,
                        norm,
                        TOKENS,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        SCALE,
                        BN,
                        OBSERVED,
                    ),
                ),
                (
                    v32_consume,
                    (
                        Y,
                        Y,
                        Skip,
                        kl,
                        ku,
                        ph,
                        pl,
                        inv,
                        full,
                        empty,
                        upper_empty,
                        norm,
                        TOPK,
                        SPLITS,
                        CHUNK,
                        BN,
                        OBSERVED,
                        Skip,
                        Skip,
                        0,
                        0,
                        False,
                    ),
                ),
            ],
            worker_num_warps=[WC],
            worker_num_regs=[128],
        )


# V32 path selection and recomputation


@triton.jit
def v32_norm_bound(q_max, k_max, excess, slack, SCALE: tl.constexpr):
    # Cauchy-Schwarz, padded for the floating-point norm reductions.
    score_cap = tl.sqrt(q_max) * tl.sqrt(k_max) * (1.001 * SCALE)
    return (
        (score_cap <= 0.625)
        & (q_max <= 1024.0)
        & (k_max <= 1024.0)
        & (excess * (1.00001 * tl.exp(2.0 * score_cap)) <= slack)
    )


@triton.jit
def v32_norm_eligible(
    q_square,
    Indices,
    KS,
    KeyNorm,
    row,
    split,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    CHECK_SENTINEL: tl.constexpr,
    GlobalNorm=None,
):
    check_begin = split * CHUNK
    check_offsets = check_begin + tl.arange(0, triton.next_power_of_2(CHUNK))
    chosen = tl.load(
        Indices + row * TOPK + check_offsets,
        check_offsets < tl.minimum(check_begin + CHUNK, TOPK),
        -1,
    )
    check_valid = (chosen >= 0) & (chosen < TOKENS)
    safe = tl.where(check_valid, chosen, 0)
    key_scale = tl.load(KS + safe, check_valid, 0.0)
    q_max = tl.max(q_square, 0)
    # Under the score cap, exp(x) is between exp(-cap) and exp(cap).
    # For theta, sum exp(x) * (k_scale - theta) <= 0 certifies
    # a bound on the weighted mean scale without tracking weights in the loop.
    # Limit the probability-weighted mean scale to keep the PV error below
    # the output's absolute tolerance after two E4M3 roundings.
    scale_limit: tl.constexpr = 0.00105
    deviation = tl.where(check_valid, key_scale - scale_limit, 0.0)
    excess = tl.sum(tl.maximum(deviation, 0.0), 0)
    slack = tl.sum(tl.maximum(-deviation, 0.0), 0)
    scale_ok = (tl.max(key_scale, 0) <= 2.0**-9) & (
        tl.min(tl.where(check_valid, key_scale, 1.0), 0) >= 2.0**-20
    )
    eligible = tl.full((), False, tl.int1)
    if GlobalNorm is not None:
        # A proof over all keys also covers any selected subset and key zero.
        eligible = scale_ok & v32_norm_bound(
            q_max, tl.load(GlobalNorm), excess, slack, SCALE
        )
    if GlobalNorm is None or (scale_ok & ~eligible):
        key_square = tl.load(KeyNorm + safe, check_valid, 0.0)
        eligible = scale_ok & v32_norm_bound(
            q_max, tl.max(key_square, 0), excess, slack, SCALE
        )
        if CHECK_SENTINEL:
            # The unmasked cache copies use key zero for invalid indices.
            if not (tl.load(KeyNorm) < float("inf")):
                eligible &= tl.min(check_valid.to(tl.int32), 0) != 0
    return eligible


@triton.jit
def v32_centered_eligible(
    q_data,
    qr_data,
    qscale,
    Indices,
    KS,
    KeyNorm,
    row,
    split,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    CHECK_SENTINEL: tl.constexpr,
):
    # The squared norm includes the physical FP32 scales.
    q_square = (
        tl.sum(q_data.to(tl.float32) * q_data.to(tl.float32), 1)
        + tl.sum(qr_data.to(tl.float32) * qr_data.to(tl.float32), 1)
    ) * (qscale * qscale)
    valid_scale = (qscale >= 2.0**-20) & (qscale <= 2.0**-6)
    q_square = tl.where(valid_scale & (q_square >= 0.0), q_square, float("inf"))
    return v32_norm_eligible(
        q_square,
        Indices,
        KS,
        KeyNorm,
        row,
        split,
        TOKENS,
        TOPK,
        CHUNK,
        SCALE,
        CHECK_SENTINEL,
    )


@triton.jit
def prequalify_v32_local_query(
    Q,
    QR,
    QS,
    KS,
    Indices,
    Norms,
    Flags,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BadRows,
    BadCount,
    GlobalNorm,
):
    row = tl.program_id(0).to(tl.int64)
    eligible = tl.load(GlobalNorm) <= 1024.0
    if not eligible:
        # With no usable global bound, check the selected keys before starting QK.
        head = tl.arange(0, 16)
        dim = tl.arange(0, 256)
        rope_dim = tl.arange(0, 64)
        q_max = tl.full((), 0.0, tl.float32)
        for group in range(4):
            h = row * 64 + group * 16 + head
            offset = h[:, None] * 512 + dim[None, :]
            lower = tl.load(Q + offset).to(tl.float32)
            upper = tl.load(Q + offset + 256).to(tl.float32)
            rope = tl.load(QR + h[:, None] * 64 + rope_dim[None, :]).to(tl.float32)
            gain = tl.load(QS + h)
            content_square = tl.sum(tl.fma(lower, lower, upper * upper), 1)
            sq = (content_square + tl.sum(rope * rope, 1)) * (gain * gain)
            valid = (sq >= 0.0) & (gain >= 2.0**-20) & (gain <= 2.0**-6)
            q_max = tl.maximum(q_max, tl.max(tl.where(valid, sq, float("inf")), 0))
        eligible = v32_norm_eligible(
            tl.full((1,), q_max, tl.float32),
            Indices,
            KS,
            Norms,
            row,
            0,
            TOKENS,
            TOPK,
            TOPK,
            SCALE,
            True,
            None,
        )
    tl.store(Flags + row, ~eligible)
    if not eligible:
        pos = tl.atomic_add(BadCount, 1, sem="relaxed")
        tl.store(BadRows + pos, row)


@triton.jit
def prequalify_v32_scale_rows(
    QueryMax,
    KS,
    Indices,
    Norms,
    Flags,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    GROUPS: tl.constexpr,
    GlobalNorm,
    BadRows,
    BadCount,
    Coeff,
):
    row = tl.program_id(0).to(tl.int64)
    sq = tl.load(QueryMax + row * GROUPS + tl.arange(0, GROUPS))
    qsq = tl.max(sq, 0)
    ksq = tl.load(GlobalNorm)
    cap = tl.sqrt(qsq) * tl.sqrt(ksq) * (1.001 * SCALE)
    use_global = (cap <= 0.625) & (qsq <= 1024.0) & (ksq <= 1024.0)
    eligible = use_global
    if not use_global:
        eligible = v32_norm_eligible(
            sq, Indices, KS, Norms, row, 0, TOKENS, TOPK, TOPK, SCALE, True, None
        )
    tl.store(Flags + row, ~eligible)
    tl.store(Coeff + row, tl.where(use_global, 1.00001 * tl.exp(2.0 * cap), 0.0))
    if not eligible:
        slot = tl.atomic_add(BadCount, 1, sem="relaxed")
        tl.store(BadRows + slot, row)


@triton.jit
def certify_v32_rows(
    QueryMax,
    KS,
    Indices,
    Norms,
    Flags,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    GROUPS: tl.constexpr,
    GlobalNorm=None,
    BadRows=None,
    BadCount=None,
):
    row = tl.program_id(0).to(tl.int64)
    q_square = tl.load(QueryMax + row * GROUPS + tl.arange(0, GROUPS))
    eligible = v32_norm_eligible(
        q_square,
        Indices,
        KS,
        Norms,
        row,
        0,
        TOKENS,
        TOPK,
        TOPK,
        SCALE,
        True,
        GlobalNorm,
    )
    tl.store(Flags + row, ~eligible)
    if BadCount is not None:
        if not eligible:
            position = tl.atomic_add(BadCount, 1, sem="relaxed")
            tl.store(BadRows + position, row)


@triton.jit
def _sparse_fp8_v32_partition(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Partial,
    Stats,
    Output,
    LSE,
    row,
    split,
    flag,
    partial_slot,
    start,
    end,
    full,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    KeyNorm,
    CENTERED: tl.constexpr,
    FULL_ROWS: tl.constexpr,
):
    H: tl.constexpr = 64
    N: tl.constexpr = 128
    h = tl.arange(0, H)
    n = tl.arange(0, N)
    dims = tl.arange(0, 512)
    rdims = tl.arange(0, 64)
    rope_dtype: tl.constexpr = QR.dtype.element_ty
    q = tle.gpu.alloc((H, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    qr = tle.gpu.alloc((H, 64), rope_dtype, nv_mma_shared_layout=True)
    sk = tle.gpu.alloc((N, 512), tl.float8e4nv, nv_mma_shared_layout=True)
    skr = tle.gpu.alloc((N, 64), rope_dtype, nv_mma_shared_layout=True)
    ph = tle.gpu.alloc((H, N), tl.float8e4nv, nv_mma_shared_layout=True)
    pl = tle.gpu.alloc((H, N), tl.float8e4nv, nv_mma_shared_layout=True)
    q_data = tl.load(Q + (row * H + h[:, None]) * 512 + dims[None, :])
    qr_data = tl.load(QR + (row * H + h[:, None]) * 64 + rdims[None, :])
    qscale = tl.load(QS + row * H + h)
    if CENTERED:
        eligible = v32_centered_eligible(
            q_data,
            qr_data,
            qscale,
            Indices,
            KS,
            KeyNorm,
            row,
            split,
            TOKENS,
            TOPK,
            CHUNK,
            SCALE,
            False,
        )
        tl.store(flag, ~eligible)
    else:
        eligible = True
        q_l1 = tl.sum(tl.abs(q_data.to(tl.float32)), 1) * qscale
        qr_l1 = tl.sum(tl.abs(qr_data.to(tl.float32)), 1) * qscale
    if eligible:
        tl.store(tle.gpu.local_ptr(q), q_data)
        tl.store(tle.gpu.local_ptr(qr), qr_data)
        lma_wait(tl.full((), 0, tl.int32))
        tl.debug_barrier()
        accum = tl.full((H, 512), 0.0, tl.float32)
        if CENTERED:
            sums = tl.full((H, N), 0.0, tl.float32)
        else:
            total = tl.full((H,), 0.0, tl.float32)
            m = tl.full((H,), float("-inf"), tl.float32)
        needs_repair = tl.full((), False, tl.int1)
        for begin in range(start, end, N):
            offsets = begin + n
            idx = tl.load(Indices + row * TOPK + offsets, offsets < end, -1)
            valid = (offsets < end) & (idx >= 0) & (idx < TOKENS)
            idx_safe = tl.where(valid, idx, 0)
            tle.gpu.copy(
                KV + idx_safe[:, None].to(tl.int64) * 512 + dims[None, :],
                sk,
                (N, 512),
                mask=tl.broadcast_to(valid[:, None], (N, 512)),
            )
            rk = tl.load(
                KR + idx_safe[:, None].to(tl.int64) * 64 + rdims[None, :],
                valid[:, None],
                0.0,
            )
            ks = tl.load(KS + idx_safe, valid, 0)
            if not CENTERED:
                max_ks = tl.max(ks, 0)
                max_kr = tl.max(tl.max(tl.abs(rk.to(tl.float32)), 1) * ks, 0)
                # Apply the general kernel's QK error bound to each split.
                error_bound = q_l1 * (448.0 * max_ks) * SCALE * (
                    (512.0 * 2.0**-24) / (1.0 - 512.0 * 2.0**-24)
                ) + qr_l1 * max_kr * SCALE * (
                    (64.0 * 2.0**-24) / (1.0 - 64.0 * 2.0**-24)
                )
                needs_repair |= tl.max(error_bound, 0) > 1.0e-4
            tl.store(tle.gpu.local_ptr(skr), rk)
            lma_wait(tl.full((), 0, tl.int32))
            tl.debug_barrier()
            dot = tle.gpu.wgmma(q, sk, trans_b=True)
            dot = tle.gpu.wgmma_wait(0, dot)
            rope = tle.gpu.wgmma(qr, skr, trans_b=True)
            rope = tle.gpu.wgmma_wait(0, rope)
            scores = (dot + rope) * (qscale[:, None] * ks[None, :] * SCALE)
            if CENTERED:
                probability = tl.exp(tl.where(valid[None, :], scores, float("-inf")))
                sums += probability
                norm = probability * (ks[None, :] * 16384.0)
                high = norm.to(tl.float8e4nv)
                low = (norm - high.to(tl.float32)).to(tl.float8e4nv)
                tl.store(tle.gpu.local_ptr(ph), high)
                tl.store(tle.gpu.local_ptr(pl), low)
                lma_wait(tl.full((), 0, tl.int32))
                tl.debug_barrier()
                accum = tle.gpu.wgmma(ph, sk, acc=accum)
                accum = tle.gpu.wgmma(pl, sk, acc=accum)
                accum = tle.gpu.wgmma_wait(0, accum)
            else:
                scores = tl.where(valid[None, :], scores, float("-inf"))
                next_max = tl.maximum(m, tl.max(scores, 1))
                safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
                correction = tl.exp(m - safe_max)
                p = tl.exp(scores - safe_max[:, None])
                total = total * correction + tl.sum(p, 1)
                w = p * ks[None, :]
                pscale = tl.maximum(tl.max(w, 1) / 448.0, 1.0e-30)
                norm = w / pscale[:, None]
                hi = norm.to(tl.float8e4nv)
                lo = ((norm - hi.to(tl.float32)) * 16.0).to(tl.float8e4nv)
                tl.store(tle.gpu.local_ptr(ph), hi)
                tl.store(tle.gpu.local_ptr(pl), lo)
                lma_wait(tl.full((), 0, tl.int32))
                tl.debug_barrier()
                prod = tle.gpu.wgmma(ph, sk)
                prod = tle.gpu.wgmma_wait(0, prod)
                prod = tle.gpu.wgmma(pl, sk, acc=prod * 16.0)
                prod = tle.gpu.wgmma_wait(0, prod)
                accum = accum * correction[:, None] + prod * (
                    pscale[:, None] * (1.0 / 16.0)
                )
                m = next_max
            tl.debug_barrier()
        if CENTERED:
            total = tl.sum(sums, 1)
            accum *= 1.0 / 16384.0
            m = tl.where(total > 0.0, 0.0, float("-inf"))
        tl.store(flag, needs_repair)
        if full if FULL_ROWS else SPLITS == 1:
            inverse = tl.div_rn(1.0, tl.where(total > 0, total, 1.0))
            tl.store(
                Output + (row * H + h[:, None]) * 512 + dims[None, :],
                accum * inverse[:, None],
            )
            tl.store(
                LSE + row * H + h,
                tl.where(total > 0, m + tl.log(total), float("inf")),
            )
        else:
            base = partial_slot * H + h
            tl.store(Partial + base[:, None] * 512 + dims[None, :], accum)
            tl.store(Stats + base * 2, m)
            tl.store(Stats + base * 2 + 1, total)


@triton.jit
def sparse_fp8_v32(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Partial,
    Stats,
    Output,
    LSE,
    RepairFlags,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    KeyNorm,
    CENTERED: tl.constexpr,
    REPLAY: tl.constexpr,
    FULL_ROWS: tl.constexpr = 0,
    TailDenominators=None,
):
    pid = tl.program_id(0).to(tl.int64)
    split = tl.program_id(2)
    if FULL_ROWS:
        row, full, start, end = mixed_v32_span(pid, TOPK, SPLITS, CHUNK, FULL_ROWS)
        flag = RepairFlags + pid
        partial_slot = pid - FULL_ROWS
    else:
        row = pid
        start = split * CHUNK
        end = tl.minimum(start + CHUNK, TOPK)
        partial_slot = row * SPLITS + split
        flag = RepairFlags + partial_slot
    if REPLAY:
        active = tl.load(flag)
        if TailDenominators is not None:
            tl.static_assert(FULL_ROWS > 0 and not CENTERED)
            if not full and not active:
                active = not collect_v32_tail_denominator(
                    TailDenominators, Stats, partial_slot
                )
    else:
        active = True
    if active:
        _sparse_fp8_v32_partition(
            Q,
            QR,
            KV,
            KR,
            QS,
            KS,
            Indices,
            Partial,
            Stats,
            Output,
            LSE,
            row,
            split,
            flag,
            partial_slot,
            start,
            end,
            full if FULL_ROWS else None,
            TOKENS,
            TOPK,
            SPLITS,
            CHUNK,
            SCALE,
            KeyNorm,
            CENTERED,
            FULL_ROWS,
        )


@triton.jit
def compact_v32_stable(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Output,
    LSE,
    RepairFlags,
    BadRows,
    BadCount,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
):
    count = tl.load(BadCount)
    for position in range(tl.program_id(0), count, tl.num_programs(0)):
        row = tl.load(BadRows + position).to(tl.int64)
        _sparse_fp8_v32_partition(
            Q,
            QR,
            KV,
            KR,
            QS,
            KS,
            Indices,
            None,
            None,
            Output,
            LSE,
            row,
            0,
            RepairFlags + row,
            row,
            0,
            TOPK,
            None,
            TOKENS,
            TOPK,
            1,
            TOPK,
            SCALE,
            None,
            False,
            0,
        )


@triton.jit
def prepare_v32_key_norm(KV, KR, KS, KNorm, TOKENS: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) * B + tl.arange(0, B)
    dims = tl.arange(0, 512)
    rdims = tl.arange(0, 64)
    k = tl.load(
        KV + token[:, None].to(tl.int64) * 512 + dims[None, :],
        token[:, None] < TOKENS,
        0.0,
    ).to(tl.float32)
    rope = tl.load(
        KR + token[:, None].to(tl.int64) * 64 + rdims[None, :],
        token[:, None] < TOKENS,
        0.0,
    ).to(tl.float32)
    scale = tl.load(KS + token, token < TOKENS, 0.0)
    squared = (tl.sum(k * k, 1) + tl.sum(rope * rope, 1)) * (scale * scale)
    squared = tl.where(squared >= 0.0, squared, float("inf"))
    tl.store(KNorm + token, squared, token < TOKENS)


@triton.jit
def prepare_v32_qk_norm(
    Q,
    QR,
    QS,
    KV,
    KR,
    KS,
    QueryMax,
    KeyNorm,
    TOKENS: tl.constexpr,
    B: tl.constexpr,
    GroupNorm=None,
):
    block = tl.program_id(0)
    is_key = block < TOKENS // B
    if is_key:
        content, rope, scale = KV, KR, KS
        group = block
    else:
        content, rope, scale = Q, QR, QS
        group = block - TOKENS // B
    row = group * B + tl.arange(0, B)
    offset = row[:, None].to(tl.int64)
    address = content + offset * 512 + tl.arange(0, 256)[None, :]
    lower = tl.load(address).to(tl.float32)
    upper = tl.load(address + 256).to(tl.float32)
    content_square = tl.sum(tl.fma(lower, lower, upper * upper), 1)
    rope_values = tl.load(rope + offset * 64 + tl.arange(0, 64)[None, :]).to(tl.float32)
    gain = tl.load(scale + row)
    square = (content_square + tl.sum(rope_values * rope_values, 1)) * (gain * gain)
    if is_key:
        square = tl.where(square >= 0.0, square, float("inf"))
        tl.store(KeyNorm + row, square)
        if GroupNorm is not None:
            tl.store(GroupNorm + group, tl.max(square, 0))
    else:
        # Encode any invalid head so the whole query fails the later certificate.
        valid = (square >= 0.0) & (gain >= 2.0**-20) & (gain <= 2.0**-6)
        tl.store(QueryMax + group, tl.max(tl.where(valid, square, float("inf")), 0))


@triton.jit
def reduce_v32_key_norm(
    GroupNorm, GlobalNorm, GROUPS: tl.constexpr, BLOCK: tl.constexpr, BadCount=None
):
    group = tl.arange(0, BLOCK)
    maximum = tl.max(tl.load(GroupNorm + group, group < GROUPS, 0.0), 0)
    tl.store(GlobalNorm, maximum)
    if BadCount is not None:
        tl.store(BadCount, 0)


@triton.jit
def prepare_v32_rope_peak(KR, KS, Peak, TOKENS: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) * B + tl.arange(0, B)
    channel = tl.arange(0, 64)
    rope = tl.load(
        KR + token[:, None] * 64 + channel[None, :], token[:, None] < TOKENS, 0.0
    ).to(tl.float32)
    magnitude = tl.abs(rope)
    magnitude = tl.where(magnitude < float("inf"), magnitude, float("inf"))
    scale = tl.load(KS + token, token < TOKENS, 0.0)
    scaled = tl.max(magnitude, 1) * scale
    scaled = tl.where(scaled >= 0.0, scaled, float("inf"))
    tl.store(Peak + token, scaled, token < TOKENS)


@lru_cache(maxsize=64)
def _v32_direct_layout_matches(kernel, copy_mode):
    # The native copy inverts the MP31 shared-memory swizzle at (0, 0).
    # Validate the actual allocated tile encodings before launching it.
    ir = kernel.asm.get("ttgir", "")
    encodings = dict(
        re.findall(r"^(#[a-zA-Z0-9_]+) = (#ttg\.swizzled_shared<.+>)$", ir, re.M)
    )
    kv = ("vec = 16", "perPhase = 1", "maxPhase = 16")
    expected = [
        ("2x128x256xf8E4M3FN", (*kv, "order = [2, 1, 0]")),
        (
            "128x256xf8E4M3FN" if copy_mode == 1 else r"2x128x64x(?:bf16|f16)",
            (
                (*kv, "order = [1, 0]")
                if copy_mode == 1
                else ("vec = 8", "perPhase = 2", "maxPhase = 16", "order = [2, 1, 0]")
            ),
        ),
    ]
    for shape, fields in expected:
        allocations = re.findall(
            r"ttg\.local_alloc[^\n]*?!ttg\.memdesc<"
            + shape
            + r", (#[a-zA-Z0-9_]+), #smem, mutable>",
            ir,
        )
        if len(allocations) != 1:
            return False
        encoding = encodings.get(allocations[0], "")
        if not all(field in encoding for field in fields):
            return False
    return True


def _warmup_v32_pipeline(pipeline, args, options, copy_mode=0, scale_check_args=()):
    options = dict(options)
    certified_poly = options.pop("CERTIFIED_POLY", False)
    static_sq64 = options.pop("STATIC_SQ64", False)
    local_query_norm = options.pop("LOCAL_QUERY_NORM", False)
    tail_denominators = options.pop("TAIL_DENOMINATORS", None)
    loader_scale_check = options.pop("LOADER_SCALE_CHECK", False)
    trailing = (
        certified_poly,
        static_sq64,
        *(scale_check_args or (None, None, None)),
        local_query_norm,
        tail_denominators,
        loader_scale_check,
    )

    def bind(compiled, mode):
        launch = compiled[options["grid"]]

        def run(*launch_args):
            launch(
                *launch_args,
                mode,
                certified_poly,
                static_sq64,
                *(scale_check_args or (None, None, None)),
                local_query_norm,
                tail_denominators,
                loader_scale_check,
            )

        return run

    if copy_mode:
        target = triton.runtime.driver.active.get_current_target()
        inputs_aligned = args[2].data_ptr() % 16 == 0 and (
            copy_mode == 1 or args[3].data_ptr() % 16 == 0
        )
        if (
            target.backend == "musa"
            and str(target.arch) == "31"
            and torch.musa.get_device_capability(args[0].device) == (3, 1)
            and args[11] % 128 == 0
            and args[13] % 128 == 0
            and inputs_aligned
        ):
            try:
                copy_ir = (
                    V32_DIRECT_COPY_TWO_LANE_IR
                    if copy_mode == 2 and local_query_norm
                    else V32_DIRECT_COPY_IR
                )
                direct_options = dict(
                    options,
                    extern_libs={"native": native_library(VECTOR_EXP2_IR + copy_ir)},
                )
                compiled = pipeline.warmup(
                    *args, copy_mode, *trailing, **direct_options
                )
                if _v32_direct_layout_matches(compiled, copy_mode):
                    return bind(compiled, copy_mode)
            except (CompilationError, OutOfResources, RuntimeError) as error:
                logging.getLogger(__name__).debug(
                    "V32 direct copy unavailable: %s", error
                )
    return bind(pipeline.warmup(*args, 0, *trailing, **options), 0)


@triton.jit
def finalize_v32_tail_denominator(Denominators, Stats, Flags, FULL_ROWS: tl.constexpr):
    split = tl.program_id(0)
    flag_ptr = Flags + FULL_ROWS + split
    if not tl.load(flag_ptr):
        if not collect_v32_tail_denominator(Denominators, Stats, split):
            tl.store(flag_ptr, True)


def prepare_mixed_v32(
    inputs,
    output,
    lse,
    scale,
    repair,
    mixed_repair,
    combine,
    finish_mixed_kernel,
    *,
    loader_scale_check=None,
):
    rows, topk = inputs[0].shape[0], inputs[-1].shape[-1]
    if loader_scale_check is None:
        loader_scale_check = rows in (132, 193) and topk == 2048
    tokens = inputs[2].shape[0] * 64
    # Fill whole waves before distributing the final rows across short splits.
    full_rows = (rows // 60) * 60
    splits = {64: 8, 132: 4, 193: 4, 449: 2}[rows]
    tail_rows = rows - full_rows
    chunk = topk // splits
    flags = torch.empty(
        (full_rows + tail_rows * splits,), dtype=torch.bool, device=output.device
    )
    partial = torch.empty(
        (tail_rows, splits, 64, 512), dtype=torch.float32, device=output.device
    )
    stats = torch.empty(
        (tail_rows, splits, 64, 2), dtype=torch.float32, device=output.device
    )

    def prepare_replay(lo, hi, count, partial, stats, flags, mixed_rows=0):
        q, qr, kv, kr, qs, ks, indices = inputs
        local_inputs = (q[lo:hi], qr[lo:hi], kv, kr, qs[lo:hi], ks, indices[lo:hi])
        y, local_lse = output[lo:hi], lse[lo:hi]
        n = hi - lo
        chunk = topk // count
        args = (
            *local_inputs,
            partial,
            stats,
            y,
            local_lse,
            flags,
            tokens,
            topk,
            count,
            chunk,
            scale,
            None,
            False,
            True,
            mixed_rows,
            tail_denominators if mixed_rows else None,
        )
        grid = (flags.numel(), 1, 1) if mixed_rows else (n, 1, count)
        safe = sparse_fp8_v32.warmup(
            *args,
            grid=grid,
            num_warps=16,
            num_stages=1,
            enable_backend_opt=True,
            extern_libs={"native": native_library()},
        )[grid]
        repair_args = (
            *local_inputs,
            None,
            partial,
            stats,
            y,
            local_lse,
            None,
            n,
            64,
            1,
            topk,
            tokens,
            64,
            scale,
            False,
            False,
            count,
            chunk,
            32,
            64,
            flags,
            mixed_rows,
        )
        grid = (flags.numel(), 2, 1) if mixed_rows else (n, 2, count)
        exact = repair.warmup(*repair_args, grid=grid, num_warps=4, num_stages=1)[grid]
        return (safe, args), (exact, repair_args)

    try:
        tail_denominators = None
        finalize_tail = None
        if topk == 2048 and (rows == 64 or loader_scale_check):
            tail_denominators = torch.empty(
                (tail_rows, splits, 64, 4), dtype=torch.float32, device=output.device
            )

        preload_args = (
            *inputs,
            output,
            lse,
            flags,
            tokens,
            topk,
            splits,
            chunk,
            scale,
            128,
            16,
            8,
            True,
            inputs[5],
            False,
            True,
        )
        grid = (flags.numel(), 1, 1)
        pipeline_options = dict(
            grid=grid,
            num_warps=16,
            num_stages=1,
            enable_backend_opt=True,
            STATIC_SQ64=rows == 64,
            TAIL_DENOMINATORS=tail_denominators,
            LOADER_SCALE_CHECK=loader_scale_check,
            extern_libs={"native": native_library(VECTOR_EXP2_IR)},
        )
        check = None
        cutoff = again = flags
        if rows != 64 and not loader_scale_check:
            cutoff = torch.empty_like(flags, dtype=torch.float32)
            again = torch.empty_like(flags)
            check_args = (
                inputs[2],
                inputs[5],
                inputs[6],
                cutoff,
                flags,
                again,
                tokens,
                topk,
                splits,
                chunk,
                full_rows,
            )
            check = qualify_v32_scales.warmup(*check_args, grid=grid, num_warps=4)[grid]

        def prepare_pipeline(static_splits):
            args = (*preload_args, cutoff, again)
            tail = (partial, stats, full_rows, static_splits)
            pipeline_args = (*args, 0 if check is None else 1, *tail)
            pipeline = _warmup_v32_pipeline(
                sparse_fp8_v32_pipeline, pipeline_args, pipeline_options
            )
            scale_precheck = None
            if check is not None:
                repeat_args = (*args, 2, *tail)
                repeat = _warmup_v32_pipeline(
                    sparse_fp8_v32_pipeline, repeat_args, pipeline_options
                )
                scale_precheck = check, check_args, repeat, repeat_args
            return pipeline, pipeline_args, scale_precheck

        try:
            pipeline, pipeline_args, scale_precheck = prepare_pipeline(True)
        except (CompilationError, OutOfResources, RuntimeError):
            if rows == 449 or tail_denominators is not None:
                # Use the uniform pipeline if static specialization is unavailable.
                raise
            pipeline, pipeline_args, scale_precheck = prepare_pipeline(False)
        if check is None and tail_denominators is not None:
            heads = 16 if rows == 193 else 8
            finish_args = (
                *inputs,
                partial,
                output,
                lse,
                flags,
                tail_denominators,
                tokens,
                topk,
                splits,
                scale,
                full_rows,
                heads,
            )
            finish_grid = (rows, 64 // heads, 1)
            try:
                finish = finish_mixed_kernel.warmup(
                    *finish_args, grid=finish_grid, num_warps=4, num_stages=1
                )[finish_grid]
            except (CompilationError, OutOfResources, RuntimeError) as error:
                logging.getLogger(__name__).debug(
                    "V32 combined finish unavailable: %s", error
                )
            else:

                def finish_mixed():
                    pipeline(*pipeline_args)
                    finish(*finish_args)

                return finish_mixed
        mixed_args = (
            *inputs,
            partial,
            stats,
            output,
            lse,
            flags,
            tail_denominators,
            tokens,
            topk,
            splits,
            chunk,
            scale,
            full_rows,
        )
        repair_grid = (flags.numel(), 2, 1)
        try:
            fused_repair = mixed_repair.warmup(
                *mixed_args, grid=repair_grid, num_warps=4, num_stages=1
            )[repair_grid]
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "V32 fused exact replay unavailable: %s", error
            )
            try:
                replays = prepare_replay(
                    0, rows, splits, partial, stats, flags, full_rows
                )
            except (CompilationError, OutOfResources, RuntimeError) as error:
                logging.getLogger(__name__).debug(
                    "V32 combined replay unavailable: %s", error
                )
                replays = (
                    *prepare_replay(0, full_rows, 1, None, None, flags[:full_rows]),
                    *prepare_replay(
                        full_rows, rows, splits, partial, stats, flags[full_rows:]
                    ),
                )
                if tail_denominators is not None:
                    finalize_args = (tail_denominators, stats, flags, full_rows)
                    tail_grid = (tail_rows * splits, 1, 1)
                    finalize_tail = finalize_v32_tail_denominator.warmup(
                        *finalize_args, grid=tail_grid, num_warps=4
                    )[tail_grid]
        else:
            replays = ((fused_repair, mixed_args),)
        merge_args = (
            partial,
            stats,
            None,
            output[full_rows:],
            lse[full_rows:],
            tail_rows,
            64,
            1,
            splits,
            False,
            splits,
            512,
        )
        grid = (tail_rows, 64, 1)
        merge = combine.warmup(*merge_args, grid=grid, num_warps=4)[grid]
    except (ImportError, CompilationError, OutOfResources, RuntimeError) as error:
        logging.getLogger(__name__).debug("V32 mixed splits unavailable: %s", error)
        if loader_scale_check:
            return prepare_mixed_v32(
                inputs,
                output,
                lse,
                scale,
                repair,
                mixed_repair,
                combine,
                finish_mixed_kernel,
                loader_scale_check=False,
            )
        return None

    def run():
        if scale_precheck is not None:
            check, check_args, repeat, repeat_args = scale_precheck
            check(*check_args)
        pipeline(*pipeline_args)
        if scale_precheck is not None:
            repeat(*repeat_args)
        if finalize_tail is not None:
            finalize_tail(*finalize_args)
        for replay, args in replays:
            replay(*args)
        merge(*merge_args)

    return run


def prepare_sparse_fp8_v32(
    inputs,
    partial,
    stats,
    output,
    lse,
    scale,
    splits,
    chunk,
    repair,
    repair_args,
    compact_repair,
):
    rows, topk = inputs[0].shape[0], inputs[-1].shape[-1]
    tokens = inputs[2].shape[0] * 64
    flags = torch.empty((rows, splits), dtype=torch.bool, device=output.device)
    grid = (rows, 1, splits)
    repair_grid = (rows, 2, splits)
    args = (
        *inputs,
        partial,
        stats,
        output,
        lse,
        flags,
        tokens,
        topk,
        splits,
        chunk,
        scale,
    )
    repair_args = (*repair_args, 32, 64, flags, 0)
    options = dict(
        grid=grid,
        num_warps=16,
        num_stages=1,
        enable_backend_opt=True,
        extern_libs={"native": native_library()},
    )
    centered = pipeline = prepare_peak = bound_reduce = None
    scale_precheck = compact_replay = None
    observed = 64 <= rows <= 512
    use_norms = rows > 512 and splits == 1 and tokens <= 1024 * rows
    if (observed or use_norms) and 0 < tokens <= (1 << 20) and scale <= 1.0:
        try:
            if use_norms:
                norms = torch.empty(
                    (tokens,), dtype=torch.float32, device=output.device
                )
            try:
                base_pipeline_args = (
                    *inputs,
                    output if splits == 1 else partial,
                    lse if splits == 1 else stats,
                    flags,
                    tokens,
                    topk,
                    splits,
                    chunk,
                    scale,
                    128,
                    16,
                    8,
                    observed,
                )
                if use_norms:
                    norm_block = 16
                    groups = 64 // norm_block
                    query_max = torch.empty(
                        (rows, groups), dtype=torch.float32, device=output.device
                    )
                    norm_args = (
                        inputs[0],
                        inputs[1],
                        inputs[4],
                        inputs[2],
                        inputs[3],
                        inputs[5],
                        query_max,
                        norms,
                        tokens,
                        norm_block,
                    )
                    # Both inputs have a whole number of 64-row blocks.
                    norm_grid = ((tokens + rows * 64) // norm_block, 1, 1)
                    certify_args = (
                        query_max,
                        inputs[5],
                        inputs[6],
                        norms,
                        flags,
                        tokens,
                        topk,
                        scale,
                        groups,
                    )
                    key_groups = tokens // norm_block
                    group_norm = torch.empty(
                        (key_groups,), dtype=torch.float32, device=output.device
                    )
                    global_norm = torch.empty(
                        (), dtype=torch.float32, device=output.device
                    )
                    bound_grid = (1, 1, 1)
                    bound_args = (
                        group_norm,
                        global_norm,
                        key_groups,
                        triton.next_power_of_2(key_groups),
                    )
                    try:
                        prepare_norm = prepare_v32_qk_norm.warmup(
                            *norm_args, group_norm, grid=norm_grid, num_warps=4
                        )[norm_grid]
                        reduce_bound = reduce_v32_key_norm.warmup(
                            *bound_args, None, grid=bound_grid, num_warps=8
                        )[bound_grid]
                        certify = certify_v32_rows.warmup(
                            *certify_args,
                            global_norm,
                            None,
                            None,
                            grid=grid,
                            num_warps=4,
                        )[grid]
                    except (CompilationError, OutOfResources, RuntimeError) as error:
                        logging.getLogger(__name__).debug(
                            "V32 global norm check unavailable: %s", error
                        )
                        norm_args = (*norm_args, None)
                        certify_args = (*certify_args, None, None, None)
                        prepare_norm = prepare_v32_qk_norm.warmup(
                            *norm_args, grid=norm_grid, num_warps=4
                        )[norm_grid]
                        certify = certify_v32_rows.warmup(
                            *certify_args, grid=grid, num_warps=4
                        )[grid]
                    else:
                        norm_args = (*norm_args, group_norm)
                        certify_args = (*certify_args, global_norm, None, None)
                        bound_reduce = reduce_bound, (*bound_args, None)
                pipeline_options = dict(
                    options,
                    extern_libs={"native": native_library(VECTOR_EXP2_IR)},
                    CERTIFIED_POLY=use_norms and topk <= 2048,
                )
                use_peaks = observed and rows >= 384 and tokens <= 1024 * rows
                try:
                    if use_peaks:
                        peak = torch.empty(
                            (tokens,), dtype=torch.float32, device=output.device
                        )
                        peak_grid = (triton.cdiv(tokens, 16), 1, 1)
                        peak_args = (inputs[3], inputs[5], peak, tokens, 16)
                        prepare_peak = prepare_v32_rope_peak.warmup(
                            *peak_args, grid=peak_grid, num_warps=4
                        )[peak_grid]
                    else:
                        peak = inputs[5]
                    preload_args = (*base_pipeline_args, peak, use_peaks, True)
                    copy_mode = 2 if rows == 512 or use_norms else 0
                    if rows == 512 and observed and splits == 1 and topk == 2048:
                        loader_args = (
                            *preload_args,
                            flags,
                            flags,
                            0,
                            None,
                            None,
                            0,
                            False,
                        )
                        try:
                            pipeline = _warmup_v32_pipeline(
                                sparse_fp8_v32_pipeline,
                                loader_args,
                                dict(pipeline_options, LOADER_SCALE_CHECK=True),
                                copy_mode,
                            )
                        except (
                            CompilationError,
                            OutOfResources,
                            RuntimeError,
                        ) as error:
                            logging.getLogger(__name__).debug(
                                "V32 loader scale check unavailable: %s", error
                            )
                        else:
                            pipeline_args = loader_args
                    if pipeline is None and observed and splits == 1:
                        try:
                            cutoff = torch.empty(
                                (rows, splits),
                                dtype=torch.float32,
                                device=output.device,
                            )
                            again = torch.empty(
                                (rows, splits), dtype=torch.bool, device=output.device
                            )
                            check_grid = (rows, splits, 1)
                            check_args = (
                                inputs[2],
                                inputs[5],
                                inputs[6],
                                cutoff,
                                flags,
                                again,
                                tokens,
                                topk,
                                splits,
                                chunk,
                                0,
                            )
                            check = qualify_v32_scales.warmup(
                                *check_args, grid=check_grid, num_warps=4
                            )[check_grid]
                            fast_args = (
                                *preload_args,
                                cutoff,
                                again,
                                1,
                                None,
                                None,
                                0,
                                False,
                            )
                            again_args = (
                                *preload_args,
                                cutoff,
                                again,
                                2,
                                None,
                                None,
                                0,
                                False,
                            )
                            fast = _warmup_v32_pipeline(
                                sparse_fp8_v32_pipeline,
                                fast_args,
                                pipeline_options,
                                copy_mode,
                            )
                            observed_again = _warmup_v32_pipeline(
                                sparse_fp8_v32_pipeline,
                                again_args,
                                pipeline_options,
                                copy_mode,
                            )
                        except (
                            CompilationError,
                            OutOfResources,
                            RuntimeError,
                        ) as error:
                            logging.getLogger(__name__).debug(
                                "V32 scale precheck unavailable: %s", error
                            )
                        else:
                            scale_precheck = (
                                check,
                                check_args,
                                observed_again,
                                again_args,
                            )
                            pipeline, pipeline_args = fast, fast_args
                    if pipeline is None:
                        pipeline_args = (
                            *preload_args,
                            flags,
                            flags,
                            0,
                            None,
                            None,
                            0,
                            False,
                        )
                        pipeline = _warmup_v32_pipeline(
                            sparse_fp8_v32_pipeline,
                            pipeline_args,
                            pipeline_options,
                            copy_mode,
                        )
                except (CompilationError, OutOfResources, RuntimeError):
                    # Keep the two-partition pipeline available on resource-constrained compilers.
                    prepare_peak = None
                    pipeline_args = (
                        *base_pipeline_args,
                        inputs[5],
                        False,
                        False,
                        flags,
                        flags,
                        0,
                        None,
                        None,
                        0,
                        False,
                    )
                    pipeline = _warmup_v32_pipeline(
                        sparse_fp8_v32_pipeline, pipeline_args, pipeline_options
                    )
            except (
                ImportError,
                CompilationError,
                OutOfResources,
                RuntimeError,
            ) as error:
                logging.getLogger(__name__).debug("V32 pipeline unavailable: %s", error)
            if pipeline is None and use_norms and rows >= 512:
                norm_block = 16
                norm_args = (*inputs[2:4], inputs[5], norms, tokens, norm_block)
                norm_grid = (triton.cdiv(tokens, norm_block), 1, 1)
                prepare_norm = prepare_v32_key_norm.warmup(
                    *norm_args, grid=norm_grid, num_warps=4
                )[norm_grid]
                centered = sparse_fp8_v32.warmup(
                    *args, norms, True, False, 0, None, **options
                )[grid]
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "Centered V32 sparse FP8 tile unavailable: %s", error
            )
            centered = pipeline = None

    if pipeline is not None and bound_reduce is not None:
        bad_rows = torch.empty((rows,), dtype=torch.int32, device=output.device)
        bad_count = torch.empty((), dtype=torch.int32, device=output.device)
        queued_bound_args = (*bound_reduce[1][:-1], bad_count)
        queued_certify_args = (*certify_args[:-2], bad_rows, bad_count)
        replay_args = (
            *inputs,
            output,
            lse,
            flags,
            bad_rows,
            bad_count,
            tokens,
            topk,
            scale,
        )
        replay_grid = (120, 1, 1)
        exact_grid = (120, 2, 1)
        try:
            queued_reduce = reduce_v32_key_norm.warmup(
                *queued_bound_args, grid=bound_grid, num_warps=8
            )[bound_grid]
            queued_certify = certify_v32_rows.warmup(
                *queued_certify_args, grid=grid, num_warps=4
            )[grid]
            safe = compact_v32_stable.warmup(
                *replay_args, **dict(options, grid=replay_grid)
            )[replay_grid]
            # Four warps can trigger conflicting LLVM fences for FP16 RoPE.
            exact_warps = 8 if inputs[1].dtype == torch.float16 else 4
            exact = compact_repair.warmup(
                *replay_args, grid=exact_grid, num_warps=exact_warps, num_stages=1
            )[exact_grid]
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "V32 compact replay unavailable: %s", error
            )
        else:
            bound_reduce = queued_reduce, queued_bound_args
            certify, certify_args = queued_certify, queued_certify_args
            compact_replay = safe, exact, replay_args
            if topk == 2048:
                coeff = torch.empty((rows,), dtype=torch.float32, device=output.device)
                fused_certify_args = (*queued_certify_args, coeff)
                try:
                    fused_certify = prequalify_v32_scale_rows.warmup(
                        *fused_certify_args, grid=grid, num_warps=4
                    )[grid]
                    fused_pipeline = _warmup_v32_pipeline(
                        sparse_fp8_v32_pipeline,
                        pipeline_args,
                        pipeline_options,
                        copy_mode,
                        scale_check_args=(coeff, bad_rows, bad_count),
                    )
                except (CompilationError, OutOfResources, RuntimeError) as error:
                    logging.getLogger(__name__).debug(
                        "Fused V32 scale check unavailable: %s", error
                    )
                else:
                    pipeline = fused_pipeline
                    certify, certify_args = fused_certify, fused_certify_args
                    key_grid = (tokens // norm_block, 1, 1)
                    local_certify_args = (
                        inputs[0],
                        inputs[1],
                        inputs[4],
                        inputs[5],
                        inputs[6],
                        norms,
                        flags,
                        tokens,
                        topk,
                        scale,
                        bad_rows,
                        bad_count,
                        global_norm,
                    )
                    try:
                        key_norm = prepare_v32_qk_norm.warmup(
                            *norm_args, grid=key_grid, num_warps=4
                        )[key_grid]
                        local_certify = prequalify_v32_local_query.warmup(
                            *local_certify_args, grid=grid, num_warps=4
                        )[grid]
                        local_pipeline = _warmup_v32_pipeline(
                            sparse_fp8_v32_pipeline,
                            pipeline_args,
                            dict(pipeline_options, LOCAL_QUERY_NORM=True),
                            copy_mode,
                            scale_check_args=(global_norm, bad_rows, bad_count),
                        )
                    except (CompilationError, OutOfResources, RuntimeError) as error:
                        logging.getLogger(__name__).debug(
                            "Local V32 query norm unavailable: %s", error
                        )
                    else:
                        prepare_norm = key_norm
                        pipeline = local_pipeline
                        certify, certify_args = local_certify, local_certify_args

    if compact_replay is None:
        try:
            main = sparse_fp8_v32.warmup(
                *args,
                None,
                False,
                centered is not None or pipeline is not None,
                0,
                None,
                **options,
            )[grid]
            repair = repair.warmup(
                *repair_args, grid=repair_grid, num_warps=4, num_stages=1
            )[repair_grid]
        except (CompilationError, OutOfResources, RuntimeError) as error:
            logging.getLogger(__name__).debug(
                "V32 sparse FP8 tile unavailable: %s", error
            )
            return None

    def run():
        if pipeline is not None:
            if use_norms:
                prepare_norm(*norm_args)
                if bound_reduce is not None:
                    reduce_bound, bound_args = bound_reduce
                    reduce_bound(*bound_args)
                certify(*certify_args)
            elif prepare_peak is not None:
                prepare_peak(*peak_args)
            if scale_precheck is not None:
                check, check_args, observed_again, again_args = scale_precheck
                check(*check_args)
                pipeline(*pipeline_args)
                observed_again(*again_args)
            else:
                pipeline(*pipeline_args)
        elif centered is not None:
            prepare_norm(*norm_args)
            centered(*args, norms, True, False, 0, None)
        if compact_replay is None:
            main(
                *args,
                None,
                False,
                centered is not None or pipeline is not None,
                0,
                None,
            )
            repair(*repair_args)
        else:
            safe, exact, replay_args = compact_replay
            safe(*replay_args)
            exact(*replay_args)

    return run


# General Triton kernels and public dispatch


@triton.jit
def _sparse_fp8_row(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Lengths,
    Partial,
    Stats,
    Output,
    LSE,
    Sink,
    row,
    H: tl.constexpr,
    SQ: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    FULL_ROWS: tl.constexpr = 0,
):
    if FULL_ROWS:
        slot = row
        row, full, split_start, split_end = mixed_v32_span(
            slot, K, SPLITS, CHUNK, FULL_ROWS
        )
        partial_slot = slot - FULL_ROWS
    else:
        split = tl.program_id(2)
        length = (
            tl.minimum(tl.maximum(tl.load(Lengths + row), 0), K) if HAS_LENGTH else K
        )
        split_start = split * CHUNK
        split_end = tl.minimum(split_start + CHUNK, length)
        partial_slot = row * SPLITS + split
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    dims = tl.arange(0, 512)
    columns = tl.arange(0, BN)
    query = tl.load(
        Q + (row * H + heads[:, None]) * 512 + dims[None, :], heads[:, None] < H, 0.0
    )
    qscale = tl.load(QS + row * H + heads, heads < H, 0)
    if ROPE:
        rdims = tl.arange(0, 64)
        qrope = tl.load(
            QR + (row * H + heads[:, None]) * 64 + rdims[None, :],
            heads[:, None] < H,
            0.0,
        )
    q_l1 = tl.sum(tl.abs(query.to(tl.float32)), 1) * qscale
    if ROPE:
        qr_l1 = tl.sum(tl.abs(qrope.to(tl.float32)), 1) * qscale
    maximum = tl.full((BH,), float("-inf"), tl.float32)
    denominator = tl.full((BH,), 0.0, tl.float32)
    accumulator = tl.full((BH, 512), 0.0, tl.float32)
    for begin in range(split_start, split_end, BN):
        offsets = begin + columns
        tokens = tl.load(Indices + row * K + offsets, offsets < split_end, -1)
        valid = (offsets < split_end) & (tokens >= 0) & (tokens < TOKENS)
        safe_tokens = tl.where(valid, tokens, 0)
        keys = tl.load(
            KV + safe_tokens[None, :].to(tl.int64) * 512 + dims[:, None],
            valid[None, :],
            0.0,
        )
        kscale = tl.load(KS + safe_tokens, valid, 0)
        scores = tl.dot(query, keys).to(tl.float32)
        if ROPE:
            krope = tl.load(
                KR + safe_tokens[None, :].to(tl.int64) * 64 + rdims[:, None],
                valid[None, :],
                0,
            )
            scores += tl.dot(qrope, krope).to(tl.float32)
        scores *= qscale[:, None] * kscale[None, :] * SCALE
        # Large dequantized inputs can amplify FP32 accumulation and scale
        # rounding beyond the LSE budget. Recompute those tiles in IEEE FP32.
        max_ks = tl.max(kscale, 0)
        error_bound = (
            q_l1
            * (448.0 * max_ks)
            * SCALE
            * ((512.0 * 2.0**-24) / (1.0 - 512.0 * 2.0**-24))
        )
        if ROPE:
            max_kr = tl.max(tl.max(tl.abs(krope.to(tl.float32)), 0) * kscale, 0)
            error_bound += (
                qr_l1 * max_kr * SCALE * ((64.0 * 2.0**-24) / (1.0 - 64.0 * 2.0**-24))
            )
        if tl.max(error_bound, 0) > 1.0e-4:
            scores = tl.full((BH, BN), 0.0, tl.float32)
            repair_dims = tl.arange(0, 64)
            for feature in range(0, 512, 64):
                feature_dims = feature + repair_dims
                exact_q = (
                    tl.load(
                        Q + (row * H + heads[:, None]) * 512 + feature_dims[None, :],
                        heads[:, None] < H,
                        0.0,
                        volatile=True,
                    ).to(tl.float32)
                    * qscale[:, None]
                )
                exact_k = (
                    tl.load(
                        KV
                        + safe_tokens[None, :].to(tl.int64) * 512
                        + feature_dims[:, None],
                        valid[None, :],
                        0.0,
                        volatile=True,
                    ).to(tl.float32)
                    * kscale[None, :]
                )
                scores += tl.dot(exact_q, exact_k, input_precision="ieee")
            scores *= SCALE
            if ROPE:
                exact_qr = qrope.to(tl.float32) * qscale[:, None]
                exact_kr = krope.to(tl.float32) * kscale[None, :]
                scores += tl.dot(exact_qr, exact_kr, input_precision="ieee") * SCALE
        scores = tl.where(valid[None, :], scores, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        correction = tl.exp(maximum - safe_max)
        probability = tl.exp(scores - safe_max[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        weights = probability * kscale[None, :]
        pscale = tl.maximum(tl.max(weights, 1) / 448.0, 1.0e-30)
        normalized = weights / pscale[:, None]
        high = normalized.to(tl.float8e4nv)
        low = ((normalized - high.to(tl.float32)) * 16.0).to(tl.float8e4nv)
        values = tl.trans(keys)
        product = tl.dot(high, values).to(tl.float32)
        product = tl.dot(low, values, product * 16.0).to(tl.float32) * (1.0 / 16.0)
        accumulator = accumulator * correction[:, None] + product * pscale[:, None]
        maximum = next_max
    if full if FULL_ROWS else SPLITS == 1:
        logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
        inverse = tl.div_rn(1.0, tl.where(denominator > 0, denominator, 1.0))
        if HAS_SINK:
            sink = tl.load(Sink + heads, heads < H, 0.0)
            inverse *= tl.where(
                denominator > 0, 1.0 / (1.0 + tl.exp(sink - logsum)), 0.0
            )
        tl.store(
            Output + (row * H + heads[:, None]) * 512 + dims[None, :],
            accumulator * inverse[:, None],
            heads[:, None] < H,
        )
        tl.store(LSE + (row // SQ * H + heads) * SQ + row % SQ, logsum, heads < H)
    else:
        base = partial_slot * H + heads
        tl.store(
            Partial + base[:, None] * 512 + dims[None, :],
            accumulator,
            heads[:, None] < H,
        )
        tl.store(Stats + base * 2, maximum, heads < H)
        tl.store(Stats + base * 2 + 1, denominator, heads < H)


# Keep the rare repair path out of the fused combine kernel's fast path.
_sparse_fp8_row_outlined = triton.jit(_sparse_fp8_row.fn, noinline=True)


_SPARSE_MLA_CONFIGS = runtime.get_tuned_config("mthreads_sparse_fp8_mla")


@libentry()
@libtuner(
    configs=_SPARSE_MLA_CONFIGS,
    key=["ROWS", "H", "K", "ROPE", "SPLITS"],
)
@triton.jit
def sparse_fp8_mla(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Lengths,
    Partial,
    Stats,
    Output,
    LSE,
    Sink,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    RepairFlags=None,
    FULL_ROWS: tl.constexpr = 0,
):
    row = tl.program_id(0).to(tl.int64)
    slot = row if FULL_ROWS else row * SPLITS + tl.program_id(2)
    if RepairFlags is None or tl.load(RepairFlags + slot):
        _sparse_fp8_row(
            Q,
            QR,
            KV,
            KR,
            QS,
            KS,
            Indices,
            Lengths,
            Partial,
            Stats,
            Output,
            LSE,
            Sink,
            row,
            H,
            SQ,
            K,
            TOKENS,
            ROPE,
            SCALE,
            HAS_LENGTH,
            HAS_SINK,
            SPLITS,
            CHUNK,
            BH,
            BN,
            FULL_ROWS,
        )


@triton.jit
def mixed_v32_exact_repair(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Partial,
    Stats,
    Output,
    LSE,
    Flags,
    TailDenominators,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    FULL_ROWS: tl.constexpr,
):
    slot = tl.program_id(0).to(tl.int64)
    needs_repair = tl.load(Flags + slot)
    if TailDenominators is not None:
        if slot >= FULL_ROWS and not needs_repair:
            # Both head blocks validate the split. Only one writes its stats.
            needs_repair = not collect_v32_tail_denominator(
                TailDenominators, Stats, slot - FULL_ROWS, tl.program_id(1) == 0
            )
    if needs_repair:
        _sparse_fp8_row(
            Q,
            QR,
            KV,
            KR,
            QS,
            KS,
            Indices,
            None,
            Partial,
            Stats,
            Output,
            LSE,
            None,
            slot,
            64,
            1,
            TOPK,
            TOKENS,
            64,
            SCALE,
            False,
            False,
            SPLITS,
            CHUNK,
            32,
            64,
            FULL_ROWS,
        )


@triton.jit
def finish_v32_mixed(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Partial,
    Output,
    LSE,
    Flags,
    TailDenominators,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    BH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    if row < FULL_ROWS:
        needs_repair = tl.load(Flags + row)
    else:
        splits = (row - FULL_ROWS) * SPLITS + tl.arange(0, SPLITS)
        needs_repair = tl.sum(tl.load(Flags + FULL_ROWS + splits).to(tl.int32), 0) != 0
        if not needs_repair:
            item = splits[:, None] * 64 + heads[None, :]
            parts = tl.arange(0, 4)
            denoms = tl.sum(
                tl.load(TailDenominators + item[:, :, None] * 4 + parts[None, None, :]),
                2,
            )
            bits = denoms.to(tl.uint32, bitcast=True)
            valid = (bits < 0x7F800000) | (bits == 0x80000000)
            total = tl.sum(denoms, 0)
            needs_repair = tl.min(tl.min(valid.to(tl.int32), 0), 0) == 0
            needs_repair |= tl.max(total, 0) == float("inf")
            if not needs_repair:
                # Accepted splits share the same zero logit center.
                dims = tl.arange(0, 512)
                values = tl.load(Partial + item[:, :, None] * 512 + dims[None, None, :])
                combined = tl.sum(values, 0)
                inverse = tl.div_rn(1.0, tl.where(total > 0.0, total, 1.0))
                tl.store(
                    Output + (row * 64 + heads[:, None]) * 512 + dims[None, :],
                    combined * inverse[:, None],
                )
                tl.store(
                    LSE + row * 64 + heads,
                    tl.where(total > 0.0, tl.log(total), float("inf")),
                )
    if needs_repair:
        _sparse_fp8_row(
            Q,
            QR,
            KV,
            KR,
            QS,
            KS,
            Indices,
            None,
            None,
            None,
            Output,
            LSE,
            None,
            row,
            64,
            1,
            TOPK,
            TOKENS,
            64,
            SCALE,
            False,
            False,
            1,
            TOPK,
            BH,
            64,
        )


@triton.jit
def _compact_v32_repair(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Output,
    LSE,
    Flags,
    BadRows,
    BadCount,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
):
    count = tl.load(BadCount)
    for position in range(tl.program_id(0), count, tl.num_programs(0)):
        row = tl.load(BadRows + position).to(tl.int64)
        if tl.load(Flags + row):
            _sparse_fp8_row(
                Q,
                QR,
                KV,
                KR,
                QS,
                KS,
                Indices,
                None,
                None,
                None,
                Output,
                LSE,
                None,
                row,
                64,
                1,
                TOPK,
                TOKENS,
                64,
                SCALE,
                False,
                False,
                1,
                TOPK,
                32,
                64,
            )


@triton.jit
def _repair_model1_row(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Output,
    LSE,
    row,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
):
    _sparse_fp8_row(
        Q,
        None,
        KV,
        None,
        QS,
        KS,
        Indices,
        None,
        None,
        None,
        Output,
        LSE,
        None,
        row,
        64,
        1,
        K,
        TOKENS,
        0,
        SCALE,
        False,
        False,
        1,
        tl.cdiv(K, 64) * 64,
        32,
        64,
    )


@triton.jit
def compact_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    BadRows,
    BadCount,
    Output,
    LSE,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    ROWS: tl.constexpr,
    SCAN_ROWS: tl.constexpr,
):
    if ROWS:
        positions = tl.program_id(0) * SCAN_ROWS + tl.arange(0, SCAN_ROWS)
        active = (positions < ROWS) & (
            tl.load(BadRows + positions, positions < ROWS, 1) == 0
        )
        first = tl.min(tl.where(active, positions, ROWS), 0)
        while first < ROWS:
            row = first.to(tl.int64)
            _repair_model1_row(
                Q, KV, QS, KS, Indices, Output, LSE, row, K, TOKENS, SCALE
            )
            active = active & (positions > first)
            first = tl.min(tl.where(active, positions, ROWS), 0)
    else:
        count = tl.load(BadCount)
        for request in range(tl.program_id(0), count, tl.num_programs(0)):
            row = tl.load(BadRows + request).to(tl.int64)
            _repair_model1_row(
                Q, KV, QS, KS, Indices, Output, LSE, row, K, TOKENS, SCALE
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_sparse_fp8_combine"),
    key=["ROWS", "H", "SQ", "SPLITS"],
)
@triton.jit
def combine_sparse_fp8(
    Partial,
    Stats,
    Sink,
    Output,
    LSE,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BV: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    splits = tl.arange(0, BLOCK_SPLITS)
    dims = tile * BV + tl.arange(0, BV)
    base = (row * SPLITS + splits) * H + head
    maxima = tl.load(Stats + base * 2, splits < SPLITS, float("-inf"))
    sums = tl.load(Stats + base * 2 + 1, splits < SPLITS, 0.0)
    maximum = tl.max(maxima, 0)
    safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.exp(maxima - safe_max)
    denominator = tl.sum(weights * sums, 0)
    logsum = tl.where(denominator > 0, safe_max + tl.log(denominator), float("inf"))
    partial = tl.load(
        Partial + base[:, None] * 512 + dims[None, :], splits[:, None] < SPLITS, 0.0
    )
    values = tl.sum(partial * weights[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    if HAS_SINK:
        sink = tl.load(Sink + head)
        values *= tl.where(denominator > 0, 1.0 / (1.0 + tl.exp(sink - logsum)), 0.0)
    tl.store(Output + (row * H + head) * 512 + dims, values)
    if tile == 0:
        tl.store(LSE + (row // SQ * H + head) * SQ + row % SQ, logsum)


@triton.jit
def combine_v32_two_splits(Partial, Stats, Output, LSE):
    row = tl.program_id(0)
    head = tl.program_id(1) * 2 + tl.arange(0, 2)
    dim = tl.arange(0, 512)
    first = row * 128 + head
    second = first + 64
    max0 = tl.load(Stats + first * 2)
    max1 = tl.load(Stats + second * 2)
    sum0 = tl.load(Stats + first * 2 + 1)
    sum1 = tl.load(Stats + second * 2 + 1)
    maximum = tl.maximum(max0, max1)
    safe = tl.where(maximum == float("-inf"), 0.0, maximum)
    weight0 = tl.exp(max0 - safe)
    weight1 = tl.exp(max1 - safe)
    denominator = weight0 * sum0 + weight1 * sum1
    divisor = tl.where(denominator > 0, denominator, 1.0)
    x = tl.load(Partial + first[:, None] * 512 + dim[None, :])
    y = tl.load(Partial + second[:, None] * 512 + dim[None, :])
    value = (x * weight0[:, None] + y * weight1[:, None]) / divisor[:, None]
    tl.store(Output + (row * 64 + head[:, None]) * 512 + dim[None, :], value)
    tl.store(
        LSE + row * 64 + head,
        tl.where(denominator > 0, safe + tl.log(denominator), float("inf")),
    )


@triton.jit
def flagged_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Flags,
    Output,
    LSE,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    FLAGS_PER_ROW: tl.constexpr,
    BLOCK_FLAGS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_FLAGS)
    flags = tl.load(Flags + row * FLAGS_PER_ROW + columns, columns < FLAGS_PER_ROW, 0)
    if tl.max(flags, 0) != 0:
        _repair_model1_row(Q, KV, QS, KS, Indices, Output, LSE, row, K, TOKENS, SCALE)


def prepare_flash_mla_sparse_fwd_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[
    Callable[[], tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]
]:
    """Prepare contiguous page64 E4M3 FP8 MLA; scales must be positive and finite.

    RoPE is divided by content scales. Calls through a prepared handle must be
    serialized on one stream; tensor contents may change between executions.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if heads not in (0, 16, 32) and (heads < 64 or heads % 64):
        raise NotImplementedError("requires 16, 32, or a multiple of 64 query heads")
    if (
        k_nope.ndim not in (3, 4)
        or k_nope.shape[1] != 64
        or k_nope.shape[-1] != 512
        or (k_nope.ndim == 4 and k_nope.shape[2] != 1)
    ):
        raise NotImplementedError("requires KV [pages,64,512] or [pages,64,1,512]")
    if q_nope.dtype != torch.float8_e4m3fn or k_nope.dtype != torch.float8_e4m3fn:
        raise TypeError("Q and KV content must be E4M3 FP8")
    if (
        q_rope.dtype not in (torch.bfloat16, torch.float16)
        or k_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE must have matching BF16 or FP16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if (
        q_rope.ndim != 4
        or q_rope.shape[-1] not in (0, 64)
        or q_rope.shape[:-1] != q_nope.shape[:-1]
        or k_rope.shape != (*k_nope.shape[:-1], q_rope.shape[-1])
        or q_scale.shape != (*q_nope.shape[:-1], 1)
        or k_scale.shape != (*k_nope.shape[:-1], 1)
    ):
        raise ValueError("RoPE/scales must match content rows with width 0 or 64 / 1")
    if (
        indices.dtype != torch.int32
        or indices.ndim != 3
        or indices.shape[:2] != (batch, queries)
        or indices.shape[-1] > 65536
    ):
        raise ValueError("indices must be INT32 [batch, queries, topk <= 65536]")
    if topk_length is not None and topk_length.shape == (batch,) and queries == 1:
        topk_length = topk_length[:, None]
    if topk_length is not None and (
        topk_length.shape != (batch, queries) or topk_length.dtype != torch.int32
    ):
        raise ValueError("topk_length must be INT32 [batch, queries]")
    if attn_sink is not None and (
        attn_sink.shape != (heads,) or attn_sink.dtype != torch.float32
    ):
        raise ValueError("attn_sink must be FP32 [heads]")
    if out is not None and (
        out.shape != (batch, queries, heads, 512) or out.dtype != q_rope.dtype
    ):
        raise ValueError("out must match output shape and RoPE dtype")
    inputs = (q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, indices)
    optional = tuple(t for t in (topk_length, attn_sink) if t is not None)
    tensors = inputs + optional + ((out,) if out is not None else ())
    if q_nope.device.type != "musa" or any(t.device != q_nope.device for t in tensors):
        raise ValueError("all inputs must share a MUSA device")
    if any(not t.is_contiguous() for t in tensors):
        raise NotImplementedError("requires contiguous inputs and output")
    if (
        out is not None
        and out.numel()
        and any(
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in inputs + optional
        )
    ):
        raise ValueError("out must not alias an input")
    if softmax_scale is not None and (
        isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (float, int))
    ):
        raise TypeError("softmax_scale must be a Python number")
    scale = (
        (512 + q_rope.shape[-1]) ** -0.5
        if softmax_scale is None
        else float(softmax_scale)
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be positive and finite")
    rows, topk = batch * queries, indices.shape[-1]
    splits = max(
        1,
        min(
            32,
            triton.cdiv(topk, 256),
            triton.cdiv(256, max(1, rows * triton.cdiv(heads, 32))),
        ),
    )
    use_v32 = (
        rows
        and queries == 1
        and heads == 64
        and q_rope.shape[-1] == 64
        and topk == 2048
        and topk_length is None
        and attn_sink is None
    )
    if use_v32 and 96 <= rows < 128:
        # One split gives the observed pipeline more pages to amortize startup.
        splits = 1
    chunk = triton.cdiv(topk, splits * 64) * 64
    if rows * splits * heads * 2056 > 8 * 1024**3 or k_nope.shape[0] * 64 >= 2**31:
        raise NotImplementedError(
            "workspace exceeds 8 GiB or cache exceeds INT32 indices"
        )
    output = (
        torch.empty(q_nope.shape, dtype=q_rope.dtype, device=q_nope.device)
        if out is None
        else out
    )
    lse = torch.empty(
        (batch, heads, queries), dtype=torch.float32, device=q_nope.device
    )
    if (
        HAS_MODEL1_PIPELINE
        and 0 < rows <= 16
        and queries == 1
        and heads == 64
        and q_rope.shape[-1] == 0
        and topk > 0
        and k_nope.shape[0] > 0
        and topk_length is None
        and attn_sink is None
    ):
        with runtime.torch_device_fn.device(q_nope.device):
            short = prepare_short_pipeline(
                q_nope,
                k_nope,
                q_scale,
                k_scale,
                indices,
                output,
                lse,
                scale,
                combine_sparse_fp8.jit_function,
                _sparse_fp8_row_outlined,
            )
        if short is not None:
            run_short, flags = short
            if flags is not None:
                fallback_args = (
                    q_nope,
                    k_nope,
                    q_scale,
                    k_scale,
                    indices,
                    flags,
                    output,
                    lse,
                    topk,
                    k_nope.shape[0] * 64,
                    scale,
                    flags.shape[1],
                    triton.next_power_of_2(flags.shape[1]),
                )
                with runtime.torch_device_fn.device(q_nope.device):
                    grid = (rows, 2, 1)
                    fallback = flagged_sparse_fp8.warmup(
                        *fallback_args, num_warps=4, num_stages=1, grid=grid
                    )
                    fallback_runner = fallback[grid]

            def run_fast():
                with runtime.torch_device_fn.device(q_nope.device):
                    run_short()
                    if flags is not None:
                        fallback_runner(*fallback_args)
                return output, lse

            return run_fast, run_fast()

    if (
        HAS_MODEL1_PIPELINE
        and rows >= 128
        and queries == 1
        and heads == 64
        and q_rope.shape[-1] == 0
        and topk >= 128
        and k_nope.shape[0] > 0
        and topk_length is None
        and attn_sink is None
    ):
        with runtime.torch_device_fn.device(q_nope.device):
            pipeline = prepare_model1_pipeline(
                q_nope, k_nope, q_scale, k_scale, indices, output, lse, scale
            )
        if pipeline is not None:
            run_pipeline, bad_rows, bad_count = pipeline
            if bad_rows is None:

                def run_repaired_pipeline():
                    with runtime.torch_device_fn.device(q_nope.device):
                        run_pipeline()
                    return output, lse

                return run_repaired_pipeline, run_repaired_pipeline()
            programs = min(
                rows,
                runtime.torch_device_fn.get_device_properties(
                    q_nope.device
                ).multi_processor_count,
            )
            scan_rows = (
                triton.next_power_of_2(triton.cdiv(rows, programs))
                if bad_count is None
                else 0
            )
            fallback_args = (
                q_nope,
                k_nope,
                q_scale,
                k_scale,
                indices,
                bad_rows,
                bad_count,
                output,
                lse,
                topk,
                k_nope.shape[0] * 64,
                scale,
                rows if scan_rows else 0,
                scan_rows,
            )
            with runtime.torch_device_fn.device(q_nope.device):
                fallback = compact_sparse_fp8.warmup(
                    *fallback_args, num_warps=4, num_stages=1, grid=(programs, 2, 1)
                )
                fallback_runner = fallback[(programs, 2, 1)]

            def run_fast():
                with runtime.torch_device_fn.device(q_nope.device):
                    run_pipeline()
                    fallback_runner(*fallback_args)
                return output, lse

            return run_fast, run_fast()

    if (
        HAS_TLE
        and use_v32
        and rows in (64, 132, 193, 449)
        and 0 < k_nope.shape[0] <= (1 << 14)
        and scale <= 1.0
        and runtime.torch_device_fn.get_device_properties(
            q_nope.device
        ).multi_processor_count
        == 60
    ):
        with runtime.torch_device_fn.device(q_nope.device):
            mixed = prepare_mixed_v32(
                inputs,
                output,
                lse,
                scale,
                sparse_fp8_mla.jit_function,
                mixed_v32_exact_repair,
                combine_sparse_fp8.jit_function,
                finish_v32_mixed,
            )
        if mixed is not None:

            def run_mixed():
                with runtime.torch_device_fn.device(q_nope.device):
                    mixed()
                return output, lse

            return run_mixed, run_mixed()

    if splits == 1:
        partial, stats = None, None
    else:
        partial = torch.empty(
            (rows, splits, heads, 512), dtype=torch.float32, device=q_nope.device
        )
        stats = torch.empty(
            (rows, splits, heads, 2), dtype=torch.float32, device=q_nope.device
        )

    launch_mla = sparse_fp8_mla[
        lambda meta: (rows, triton.cdiv(heads, meta["BH"]), splits)
    ]
    launch_options = {}
    mla_args = (
        *inputs[:6],
        indices,
        topk_length,
        partial,
        stats,
        output,
        lse,
        attn_sink,
        rows,
        heads,
        queries,
        topk,
        k_nope.shape[0] * 64,
        q_rope.shape[-1],
        scale,
        topk_length is not None,
        attn_sink is not None,
        splits,
        chunk,
    )

    run_v32 = None
    if HAS_TLE and use_v32:
        with runtime.torch_device_fn.device(q_nope.device):
            run_v32 = prepare_sparse_fp8_v32(
                inputs,
                partial,
                stats,
                output,
                lse,
                scale,
                splits,
                chunk,
                sparse_fp8_mla.jit_function,
                mla_args,
                _compact_v32_repair,
            )

    combine_v32_pair = use_v32 and 64 <= rows < 96 and splits == 2

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal launch_mla, launch_options
        if rows == 0 or heads == 0:
            return output, lse
        with runtime.torch_device_fn.device(q_nope.device):
            if run_v32 is not None:
                run_v32()
            else:
                try:
                    launch_mla(*mla_args, **launch_options)
                except OutOfResources:
                    # The cached tile may exceed shared memory on another compiler.
                    config = min(
                        _SPARSE_MLA_CONFIGS,
                        key=lambda c: (c.kwargs["BH"] * c.kwargs["BN"], c.num_warps),
                    )
                    launch_options = config.all_kwargs()
                    grid = (rows, triton.cdiv(heads, launch_options["BH"]), splits)
                    launch_mla = sparse_fp8_mla.jit_function[grid]
                    launch_mla(*mla_args, **launch_options)
            if combine_v32_pair:
                combine_v32_two_splits[(rows, 32)](
                    partial, stats, output, lse, num_warps=4
                )
            elif splits > 1:
                combine_sparse_fp8[lambda meta: (rows, heads, 512 // meta["BV"])](
                    partial,
                    stats,
                    attn_sink,
                    output,
                    lse,
                    rows,
                    heads,
                    queries,
                    splits,
                    attn_sink is not None,
                    triton.next_power_of_2(splits),
                )
        return output, lse

    return run, run()


def flash_mla_sparse_fwd_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, output = prepare_flash_mla_sparse_fwd_w8a8(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
    )
    return output

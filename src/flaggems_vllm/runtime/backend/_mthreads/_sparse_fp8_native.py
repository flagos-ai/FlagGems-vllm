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

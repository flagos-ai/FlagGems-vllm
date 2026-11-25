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

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
import triton.language.core as tlc
from triton.experimental.tle.language.gpu import types as tle_types

ACCUMULATOR_SCALE_FLOOR = tl.constexpr(2.0**-32)


QK_RECOMPUTE_THRESHOLD = tl.constexpr(2.0**-14)


@tlc.builtin
def sparse_smem_subslice(buf, offsets, shape, _semantic=None):
    offsets = [int(tlc._unwrap_if_constexpr(value)) for value in offsets]
    shape = [int(tlc._unwrap_if_constexpr(value)) for value in shape]
    view_type = tle_types.buffered_tensor_type(
        buf.dtype,
        shape,
        buf.type.storage,
        buf.type.layout,
        _semantic,
        alloc_shape=buf.type.alloc_shape,
    )
    handle = _semantic.builder.create_memdesc_subslice(
        view_type.to_ir(_semantic.builder), buf.handle, offsets
    )
    return tle_types.buffered_tensor(
        handle,
        buf.dtype,
        shape,
        buf.type.storage,
        buf.type.layout,
        _semantic,
        alloc_shape=buf.type.alloc_shape,
    )


@tlc.builtin
def sparse_named_barriers(count, threads, base, _semantic=None):
    barriers = tle.gpu.alloc_barriers(count, arrive_count=threads, _semantic=_semantic)
    # Lazy IDs are not unique across JIT helpers; reserve them before capture.
    base = int(tlc._unwrap_if_constexpr(base))
    barriers.named_base_id = base
    barriers.type.named_base_id = base
    return barriers


@triton.jit
def sparse_fp8_accumulate(logits, contribution):
    # Keep FP32 additions separate from tensor-core accumulation.
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [logits, contribution],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def sparse_fp8_transpose_values(
    s_src,
    s_dst,
    dst_row: tl.constexpr,
    warp_offset: tl.constexpr,
):
    """Transpose with coupled K permutation and bank-distributed output rows."""
    carrier = tl.arange(0, 128).to(tl.uint32)
    src_base = tle.gpu.local_ptr(s_src, (0, 0))
    dst_base = tle.gpu.local_ptr(s_dst, (dst_row, 0))
    return tl.inline_asm_elementwise(
        asm=(
            "{\n"
            ".reg .b32 tid, lane, warp, src_row, tmp, tmp2;\n"
            ".reg .b32 src_log, src_phys, src_addr0, src_addr1;\n"
            ".reg .b32 dst_row_r, dst_col, dst_log0, dst_log1;\n"
            ".reg .b32 dst_phys0, dst_phys1, dst_addr0, dst_addr1;\n"
            ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
            ".reg .b32 c0, c1, c2, c3, d0, d1, d2, d3;\n"
            "mov.u32 tid, %tid.x;\n"
            "and.b32 tid, tid, 127;\n"
            "and.b32 lane, tid, 31;\n"
            "shr.u32 warp, tid, 5;\n"
            "add.u32 warp, warp, $4;\n"
            # The coupled P permutation cancels the source-row bit permutation.
            "mov.u32 src_row, lane;\n"
            "shl.b32 src_log, src_row, 7;\n"
            "shl.b32 tmp, warp, 4;\n"
            "add.u32 src_log, src_log, tmp;\n"
            "shr.u32 tmp, src_log, 7;\n"
            "and.b32 tmp, tmp, 7;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 src_phys, src_log, tmp;\n"
            "add.u32 src_addr0, $2, src_phys;\n"
            "add.u32 src_addr1, src_addr0, 4096;\n"
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
            "{a0, a1, a2, a3}, [src_addr0];\n"
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
            "{b0, b1, b2, b3}, [src_addr1];\n"
            "prmt.b32 c0, a0, a1, 0x6420;\n"
            "prmt.b32 c1, a0, a1, 0x7531;\n"
            "prmt.b32 c2, a2, a3, 0x6420;\n"
            "prmt.b32 c3, a2, a3, 0x7531;\n"
            "prmt.b32 d0, b0, b1, 0x6420;\n"
            "prmt.b32 d1, b0, b1, 0x7531;\n"
            "prmt.b32 d2, b2, b3, 0x6420;\n"
            "prmt.b32 d3, b2, b3, 0x7531;\n"
            # Consecutive rows distribute each matrix store over all shared banks.
            "and.b32 dst_row_r, lane, 15;\n"
            "shl.b32 tmp, warp, 4;\n"
            "add.u32 dst_row_r, dst_row_r, tmp;\n"
            "shr.u32 dst_col, lane, 4;\n"
            "and.b32 dst_col, dst_col, 1;\n"
            "shl.b32 dst_col, dst_col, 4;\n"
            "shl.b32 dst_log0, dst_row_r, 6;\n"
            "add.u32 dst_log0, dst_log0, dst_col;\n"
            "add.u32 dst_log1, dst_log0, 32;\n"
            "shr.u32 tmp, dst_log0, 7;\n"
            "and.b32 tmp, tmp, 3;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 dst_phys0, dst_log0, tmp;\n"
            "shr.u32 tmp2, dst_log1, 7;\n"
            "and.b32 tmp2, tmp2, 3;\n"
            "shl.b32 tmp2, tmp2, 4;\n"
            "xor.b32 dst_phys1, dst_log1, tmp2;\n"
            "add.u32 dst_addr0, $3, dst_phys0;\n"
            "add.u32 dst_addr1, $3, dst_phys1;\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "[dst_addr0], {c0, c1, c2, c3};\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "[dst_addr1], {d0, d1, d2, d3};\n"
            "mov.u32 $0, $1;\n"
            "}"
        ),
        constraints="=r,r,r,r,r",
        args=[carrier, src_base, dst_base, tl.full((), warp_offset, tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def sparse_fp8_split_blocks(length, SPLITS: tl.constexpr):
    blocks = tl.cdiv(length, 64)
    if SPLITS == 1:
        first_block = 0
        split_blocks = blocks
    else:
        blocks_per_split = tl.cdiv(blocks, SPLITS)
        first_block = tl.program_id(2) * blocks_per_split
        split_blocks = tl.maximum(0, tl.minimum(blocks_per_split, blocks - first_block))
    return first_block, split_blocks


@triton.jit
def sparse_fp8_publish_output(
    acc,
    inverse,
    Output,
    batch,
    head_block,
    H: tl.constexpr,
    OFFSET: tl.constexpr,
    STREAMING: tl.constexpr = False,
):
    if Output.dtype.element_ty == tl.float32:
        rows = head_block * 64 + tl.arange(0, 64)
        columns = tl.arange(0, 128)
        columns = (columns // 16) * 16 + (columns % 8) * 2 + (columns % 16) // 8
        tl.store(
            Output + (batch * H + rows[:, None]) * 512 + OFFSET + columns[None, :],
            acc * inverse[:, None],
        )
        return
        # Pair permuted accumulator columns into adjacent BF16 output elements.
    base_u64 = (Output + (batch * H + head_block * 64) * 512 + OFFSET).to(tl.uint64)
    st_op: tl.constexpr = "st.global.cs.v2.b32" if STREAMING else "st.global.v2.b32"
    tl.inline_asm_elementwise(
        asm=(
            "{\n"
            ".reg .b32 tid, row, col, offset, a, b, c, d;\n"
            ".reg .b64 addr;\n"
            "mov.u32 tid, %tid.x;\n"
            "and.b32 tid, tid, 127;\n"
            "shr.u32 row, tid, 5;\n"
            "shl.b32 row, row, 4;\n"
            "and.b32 col, tid, 31;\n"
            "shr.u32 col, col, 2;\n"
            "add.u32 row, row, col;\n"
            "and.b32 col, tid, 3;\n"
            "shl.b32 col, col, 3;\n"
            "shl.b32 row, row, 10;\n"
            "add.u32 offset, row, col;\n"
            "cvt.u64.u32 addr, offset;\n"
            "add.u64 addr, addr, $128;\n"
            "cvt.rn.bf16x2.f32 a, $68, $64;\n"
            "cvt.rn.bf16x2.f32 b, $69, $65;\n"
            "cvt.rn.bf16x2.f32 c, $70, $66;\n"
            "cvt.rn.bf16x2.f32 d, $71, $67;\n"
            + st_op
            + " [addr+0], {a, b};\n"
            + st_op
            + " [addr+8192], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $76, $72;\n"
            "cvt.rn.bf16x2.f32 b, $77, $73;\n"
            "cvt.rn.bf16x2.f32 c, $78, $74;\n"
            "cvt.rn.bf16x2.f32 d, $79, $75;\n"
            + st_op
            + " [addr+32], {a, b};\n"
            + st_op
            + " [addr+8224], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $84, $80;\n"
            "cvt.rn.bf16x2.f32 b, $85, $81;\n"
            "cvt.rn.bf16x2.f32 c, $86, $82;\n"
            "cvt.rn.bf16x2.f32 d, $87, $83;\n"
            + st_op
            + " [addr+64], {a, b};\n"
            + st_op
            + " [addr+8256], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $92, $88;\n"
            "cvt.rn.bf16x2.f32 b, $93, $89;\n"
            "cvt.rn.bf16x2.f32 c, $94, $90;\n"
            "cvt.rn.bf16x2.f32 d, $95, $91;\n"
            + st_op
            + " [addr+96], {a, b};\n"
            + st_op
            + " [addr+8288], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $100, $96;\n"
            "cvt.rn.bf16x2.f32 b, $101, $97;\n"
            "cvt.rn.bf16x2.f32 c, $102, $98;\n"
            "cvt.rn.bf16x2.f32 d, $103, $99;\n"
            + st_op
            + " [addr+128], {a, b};\n"
            + st_op
            + " [addr+8320], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $108, $104;\n"
            "cvt.rn.bf16x2.f32 b, $109, $105;\n"
            "cvt.rn.bf16x2.f32 c, $110, $106;\n"
            "cvt.rn.bf16x2.f32 d, $111, $107;\n"
            + st_op
            + " [addr+160], {a, b};\n"
            + st_op
            + " [addr+8352], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $116, $112;\n"
            "cvt.rn.bf16x2.f32 b, $117, $113;\n"
            "cvt.rn.bf16x2.f32 c, $118, $114;\n"
            "cvt.rn.bf16x2.f32 d, $119, $115;\n"
            + st_op
            + " [addr+192], {a, b};\n"
            + st_op
            + " [addr+8384], {c, d};\n"
            "cvt.rn.bf16x2.f32 a, $124, $120;\n"
            "cvt.rn.bf16x2.f32 b, $125, $121;\n"
            "cvt.rn.bf16x2.f32 c, $126, $122;\n"
            "cvt.rn.bf16x2.f32 d, $127, $123;\n"
            + st_op
            + " [addr+224], {a, b};\n"
            + st_op
            + " [addr+8416], {c, d};\n"
            "mov.u32 $0, 0;\n"
            "}\n"
        ),
        constraints=(
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,"
            "l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l"
        ),
        args=[acc * inverse[:, None], base_u64],
        dtype=tl.uint32,
        is_pure=False,
        pack=64,
    )


@triton.constexpr_function
def p128_publication_asm(asm, prequantized):
    input_registers = 8 if prequantized else 32
    first_base = 64 + input_registers * 2
    halves = []
    for half in range(2):
        block = asm.replace(f"${32 + input_registers}", f"${first_base + half * 64}")
        for operand in range(31 + input_registers, 31, -1):
            block = block.replace(
                f"${operand}", f"${operand + 32 + half * input_registers}"
            )
        halves.append(block)
    return "".join(halves)


@triton.jit
def publish_sparse_p128_fp8(s_p, p, s_p1, PREQUANTIZED: tl.constexpr = False):
    base = tle.gpu.local_ptr(s_p, (0, 0))
    base_u32 = tl.inline_asm_elementwise(
        "mov.u32 $0, $1;", "=r,r", [base], dtype=tl.uint32, is_pure=True, pack=1
    )
    if PREQUANTIZED:
        declaration: tl.constexpr = ""
        conversion: tl.constexpr = (
            "prmt.b32 a0, $32, $33, 0x5410;\n"
            "prmt.b32 a1, $32, $33, 0x7632;\n"
            "prmt.b32 a2, $34, $35, 0x5410;\n"
            "prmt.b32 a3, $34, $35, 0x7632;\n"
            "prmt.b32 b0, $36, $37, 0x5410;\n"
            "prmt.b32 b1, $36, $37, 0x7632;\n"
            "prmt.b32 b2, $38, $39, 0x5410;\n"
            "prmt.b32 b3, $38, $39, 0x7632;\n"
        )
        publication: tl.constexpr = (
            "shr.u32 tmp, common, 7;\n"
            "and.b32 tmp, tmp, 3;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 phys0, common, tmp;\n"
            "add.u32 common, common, 32;\n"
            "shr.u32 tmp, common, 7;\n"
            "and.b32 tmp, tmp, 3;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 phys1, common, tmp;\n"
            "add.u32 addr0, $40, phys0;\n"
            "add.u32 addr1, $40, phys1;\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr0], {a0, a1, a2, a3};\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr1], {b0, b1, b2, b3};\n"
            "fence.proxy.async.shared::cta;\n"
            "mov.u32 $0, $40;\n"
            "}"
        )
    else:
        declaration: tl.constexpr = (
            ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15;\n"
        )
        conversion: tl.constexpr = (
            "cvt.rn.satfinite.e4m3x2.f32 h0, $33, $32;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h1, $35, $34;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h2, $37, $36;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h3, $39, $38;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h4, $41, $40;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h5, $43, $42;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h6, $45, $44;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h7, $47, $46;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h8, $49, $48;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h9, $51, $50;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h10, $53, $52;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h11, $55, $54;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h12, $57, $56;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h13, $59, $58;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h14, $61, $60;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h15, $63, $62;\n"
            "mov.b32 a0, {h0, h2};\n"
            "mov.b32 a1, {h1, h3};\n"
            "mov.b32 a2, {h4, h6};\n"
            "mov.b32 a3, {h5, h7};\n"
            "mov.b32 b0, {h8, h10};\n"
            "mov.b32 b1, {h9, h11};\n"
            "mov.b32 b2, {h12, h14};\n"
            "mov.b32 b3, {h13, h15};\n"
        )
        publication: tl.constexpr = (
            "shr.u32 tmp, common, 7;\n"
            "and.b32 tmp, tmp, 3;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 phys0, common, tmp;\n"
            "add.u32 common, common, 32;\n"
            "shr.u32 tmp, common, 7;\n"
            "and.b32 tmp, tmp, 3;\n"
            "shl.b32 tmp, tmp, 4;\n"
            "xor.b32 phys1, common, tmp;\n"
            "add.u32 addr0, $64, phys0;\n"
            "add.u32 addr1, $64, phys1;\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr0], {a0, a1, a2, a3};\n"
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [addr1], {b0, b1, b2, b3};\n"
            "fence.proxy.async.shared::cta;\n"
            "mov.u32 $0, $64;\n"
            "}"
        )
    asm: tl.constexpr = (
        "{\n"
        + declaration
        + (
            (
                ".reg .b32 tid, warp_off, row_off, common, tmp, phys0, phys1, addr0, addr1;\n"
                ".reg .b32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 warp_off, tid, 96;\n"
                "shl.b32 warp_off, warp_off, 5;\n"
                "and.b32 row_off, tid, 15;\n"
                "shl.b32 row_off, row_off, 6;\n"
                "or.b32 common, warp_off, row_off;\n"
                "and.b32 tmp, tid, 16;\n"
                "or.b32 common, common, tmp;\n"
            )
        )
        + conversion
        + publication
    )
    base1 = tle.gpu.local_ptr(s_p1, (0, 0))
    if PREQUANTIZED:
        constraints128: tl.constexpr = (
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,"
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,"
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,"
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,"
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r"
        )
    else:
        constraints128: tl.constexpr = (
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,"
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,"
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,"
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r"
        )
    return tl.inline_asm_elementwise(
        p128_publication_asm(asm, PREQUANTIZED),
        constraints128,
        [p, base_u32, base1.to(tl.uint64).to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=64,
    )


@triton.jit
def quantize_p_fp8_before_ready(p):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }",
        "=r,f,f,f,f",
        [p],
        dtype=tl.float8e4nv,
        is_pure=False,
        pack=4,
    )

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

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import HAS_TLE

__all__ = ["flash_mla_sparse_fwd_w8a8"]

if not HAS_TLE:
    sparse_attention = None
else:
    import triton
    import triton.experimental.tle.language as tle
    import triton.language as tl
    import triton.language.core as tlc
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language import (
        BlockedLayout,
        NVMMADistributedLayout,
        NVMMASharedLayout,
        SliceLayout,
    )
    from triton.experimental.gluon.language.nvidia import hopper
    from triton.experimental.gluon.language.nvidia.hopper import async_copy

    from flaggems_vllm import runtime
    from flaggems_vllm.ops.flash_mla import _get_num_sms as get_num_sms
    from flaggems_vllm.ops.flash_mla_with_kvcache_model1 import (
        _cache_views as cache_views,
    )
    from flaggems_vllm.ops.flash_mla_with_kvcache_model1 import tle_subslice
    from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import (
        publish_p_fp8_sw64_coupled_stmatrix,
        vtranspose_fp8_64x128_kperm,
    )
    from flaggems_vllm.utils import libentry, libtuner
    from flaggems_vllm.utils.libentry import LibEntry

    ACCUMULATOR_SCALE_FLOOR = tl.constexpr(2.0**-32)

    QK_RECOMPUTE_THRESHOLD = tl.constexpr(2.0**-14)

    @tlc.builtin
    def sparse_named_barriers(count, threads, base, _semantic=None):
        barriers = tle.gpu.alloc_barriers(
            count, arrive_count=threads, _semantic=_semantic
        )
        # Lazy IDs are not unique across JIT helpers; reserve them before capture.
        base = int(tlc._unwrap_if_constexpr(base))
        barriers.named_base_id = base
        barriers.type.named_base_id = base
        return barriers

    @triton.jit
    def sink_normalization(inverse, sink, logsum):
        return inverse * tl.where(
            sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
        )

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
            split_blocks = tl.maximum(
                0, tl.minimum(blocks_per_split, blocks - first_block)
            )
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
            block = asm.replace(
                f"${32 + input_registers}", f"${first_base + half * 64}"
            )
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

    @triton.jit
    def sparse_fp8_precise_scores(
        Q,
        KV,
        Indices,
        batch,
        heads,
        base,
        length,
        seed,
        qb: tl.constexpr,
        qh: tl.constexpr,
        kp: tl.constexpr,
        kt: tl.constexpr,
        ib: tl.constexpr,
        ik: tl.constexpr,
        N: tl.constexpr,
    ):
        positions = base + tl.arange(0, 64)
        ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
        valid = (positions < length) & (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0).to(tl.uint64)
        # Preserve the incoming MMA layout instead of allocating a shared conversion tile.
        scores = tl.inline_asm_elementwise(
            "mov.b32 $0, 0;",
            "=f,f",
            [seed],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
        )
        for feature in range(512):
            query = tl.load(Q + batch * qb + heads * qh + feature).to(tl.float32)
            key = tl.load(
                KV + (ids // 64) * kp + (ids % 64) * kt + feature, valid, 0.0
            ).to(tl.float32)
            product = query[:, None] * key[None, :]
            scores = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [scores, product],
                dtype=tl.float32,
                is_pure=False,
                pack=1,
            )
        return scores

    @triton.jit
    def sparse_fp8_load_tile(
        KV,
        KR,
        KS,
        Indices,
        sk,
        skr,
        scales,
        masks,
        batch,
        base,
        length,
        slot: tl.constexpr,
        N: tl.constexpr,
        kp: tl.constexpr,
        kt: tl.constexpr,
        krp: tl.constexpr,
        krt: tl.constexpr,
        ksp: tl.constexpr,
        kst: tl.constexpr,
        ib: tl.constexpr,
        ik: tl.constexpr,
        CAN_ASYNC: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_ROPE: tl.constexpr = True,
    ):
        keys = tl.arange(0, BLOCK_K)
        features = tl.arange(0, 512)
        ropes = tl.arange(0, 64)
        positions = base + keys
        ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
        valid = (positions < length) & (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0).to(tl.uint64)
        pages, slots = ids // 64, ids % 64
        cache = tl.load(
            KV + pages[:, None] * kp + slots[:, None] * kt + features[None, :],
            valid[:, None],
            0.0,
            volatile=not CAN_ASYNC,
        )
        scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
        tl.store(tle.gpu.local_ptr(sk.slot(slot)), cache)
        if HAS_ROPE:
            rope = tl.load(
                KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
                valid[:, None],
                0.0,
                volatile=not CAN_ASYNC,
            )
            tl.store(tle.gpu.local_ptr(skr.slot(slot)), rope)
        else:
            pass
        tl.store(tle.gpu.local_ptr(scales.slot(slot)), scale)
        tl.store(tle.gpu.local_ptr(masks.slot(slot)), valid.to(tl.int32))

    @triton.jit
    def sparse_fp8_compact_leader(
        Q,
        QR,
        KV,
        KR,
        QS,
        KS,
        Indices,
        Length,
        Sink,
        Output,
        Stats,
        RepairFlags,
        qb: tl.constexpr,
        qh: tl.constexpr,
        qrb: tl.constexpr,
        qrh: tl.constexpr,
        kp: tl.constexpr,
        kt: tl.constexpr,
        krp: tl.constexpr,
        krt: tl.constexpr,
        qsb: tl.constexpr,
        qsh: tl.constexpr,
        ksp: tl.constexpr,
        kst: tl.constexpr,
        ib: tl.constexpr,
        ik: tl.constexpr,
        sq,
        sr,
        sk,
        skr,
        scales,
        masks,
        sp,
        sv0,
        sv1,
        alpha,
        factor,
        full,
        empty,
        done,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        HAS_SINK: tl.constexpr,
        CAN_ASYNC: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_ROPE: tl.constexpr = True,
    ):
        batch = tl.program_id(0)
        head_group = tl.program_id(1)
        heads = head_group * 64 + tl.arange(0, 64)
        split = tl.program_id(2)
        features = tl.arange(0, 512)
        ropes = tl.arange(0, 64)
        # Wider scores need more registers; move another 128 PV columns to the follower.
        LEADER_D: tl.constexpr = 128 if BLOCK_K == 128 else 256
        columns = tl.arange(0, LEADER_D)
        query = tl.load(
            Q + batch * qb + heads[:, None] * qh + features[None, :],
            volatile=not CAN_ASYNC,
        )
        tl.store(tle.gpu.local_ptr(sq), query)
        if HAS_ROPE:
            query_rope = tl.load(
                QR + batch * qrb + heads[:, None] * qrh + ropes[None, :],
                volatile=not CAN_ASYNC,
            )
            tl.store(tle.gpu.local_ptr(sr), query_rope)
        else:
            pass
        query_scale = tl.load(QS + batch * qsb + heads * qsh) * (
            SCALE * 1.4426950408889634
        )
        amplification = tl.max(tl.abs(query_scale), 0)
        needs_repair = tl.full((), False, tl.int1)
        maximum_weight_scale = tl.zeros((64,), tl.float32)
        maximum = tl.full((64,), -float("inf"), tl.float32)
        denominator = tl.zeros((64,), tl.float32)
        value = tl.zeros((64, LEADER_D), tl.float32)
        previous_scale = tl.full((64,), 1.0, tl.float32)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
        start = split * blocks_per_split * BLOCK_K
        stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
        block_count = tl.cdiv(tl.maximum(stop - start, 0), BLOCK_K)
        if BLOCK_K == 64 and block_count > 0:
            sparse_fp8_load_tile(
                KV,
                KR,
                KS,
                Indices,
                sk,
                skr,
                scales,
                masks,
                batch,
                start,
                length,
                0,
                N,
                kp,
                kt,
                krp,
                krt,
                ksp,
                kst,
                ib,
                ik,
                CAN_ASYNC=CAN_ASYNC,
                BLOCK_K=BLOCK_K,
                HAS_ROPE=HAS_ROPE,
            )
        else:
            pass
        SLOTS: tl.constexpr = 2 if BLOCK_K == 64 else 1
        for pair in range(tl.cdiv(block_count, SLOTS)):
            for slot in tl.static_range(SLOTS):
                step = pair * SLOTS + slot
                if step < block_count:
                    if BLOCK_K == 128:
                        sparse_fp8_load_tile(
                            KV,
                            KR,
                            KS,
                            Indices,
                            sk,
                            skr,
                            scales,
                            masks,
                            batch,
                            start + step * BLOCK_K,
                            length,
                            slot,
                            N,
                            kp,
                            kt,
                            krp,
                            krt,
                            ksp,
                            kst,
                            ib,
                            ik,
                            CAN_ASYNC=CAN_ASYNC,
                            BLOCK_K=BLOCK_K,
                            HAS_ROPE=HAS_ROPE,
                        )
                    else:
                        pass
                    logits = tle.gpu.wgmma(
                        sq, sk.slot(slot), out_dtype=tl.float32, trans_b=True
                    )
                    if HAS_ROPE:
                        logits = tle.gpu.wgmma(sr, skr.slot(slot), logits, trans_b=True)
                    else:
                        pass
                    if BLOCK_K == 64 and step + 1 < block_count:
                        sparse_fp8_load_tile(
                            KV,
                            KR,
                            KS,
                            Indices,
                            sk,
                            skr,
                            scales,
                            masks,
                            batch,
                            start + (step + 1) * BLOCK_K,
                            length,
                            1 - slot,
                            N,
                            kp,
                            kt,
                            krp,
                            krt,
                            ksp,
                            kst,
                            ib,
                            ik,
                            CAN_ASYNC=CAN_ASYNC,
                            BLOCK_K=BLOCK_K,
                            HAS_ROPE=HAS_ROPE,
                        )
                    else:
                        pass
                    logits = tle.gpu.wgmma_wait(0, logits)
                    kv_scale = tl.load(tle.gpu.local_ptr(scales.slot(slot)))
                    valid = tl.load(tle.gpu.local_ptr(masks.slot(slot))) != 0
                    needs_repair |= (
                        amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD
                    )
                    logits = logits * query_scale[:, None] * kv_scale[None, :]
                    logits = tl.where(valid[None, :], logits, -float("inf"))
                    next_maximum = tl.maximum(maximum, tl.max(logits, 1))
                    safe_maximum = tl.where(
                        next_maximum == -float("inf"), 0.0, next_maximum
                    )
                    correction = tl.exp2(maximum - safe_maximum)
                    probabilities = tl.exp2(logits - safe_maximum[:, None])
                    denominator = denominator * correction + tl.sum(probabilities, 1)
                    weighted = probabilities * kv_scale[None, :]
                    probability_scale = tl.max(weighted, 1) / 448.0
                    maximum_weight_scale = tl.maximum(
                        maximum_weight_scale * correction, probability_scale
                    )
                    probability_scale = tl.where(
                        probability_scale > 0, probability_scale, 1.0
                    )
                    needs_repair |= (
                        tl.max(
                            (
                                maximum_weight_scale * ACCUMULATOR_SCALE_FLOOR
                                > probability_scale
                            ).to(tl.int32),
                            0,
                        )
                        != 0
                    )
                    probability_values = weighted / probability_scale[:, None]
                    if BLOCK_K == 128:
                        tl.store(
                            tle.gpu.local_ptr(sp), probability_values.to(tl.float8e4nv)
                        )
                        vp = tle.gpu.local_ptr(sk.slot(slot))
                        left, right = (
                            vp.reshape(BLOCK_K, 2, 256).permute(0, 2, 1).split()
                        )
                        tl.store(tle.gpu.local_ptr(sv0), tl.trans(tl.load(left)))
                        tl.store(tle.gpu.local_ptr(sv1), tl.trans(tl.load(right)))
                    else:
                        publish_p_fp8_sw64_coupled_stmatrix(sp, probability_values)
                        for tile in tl.static_range(4):
                            source = tle_subslice(
                                sk.slot(slot), [0, tile * 128], [64, 128]
                            )
                            if tile < 2:
                                vtranspose_fp8_64x128_kperm(source, sv0, tile * 128)
                            else:
                                vtranspose_fp8_64x128_kperm(
                                    source, sv1, (tile - 2) * 128
                                )
                    tl.inline_asm_elementwise(
                        "{ fence.proxy.async.shared::cta; mov.b32 $0, $1; }",
                        "=r,r",
                        [tl.arange(0, 128)],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                    tl.debug_barrier()
                    correction = (correction * previous_scale) / probability_scale
                    tl.store(tle.gpu.local_ptr(alpha), correction)
                    tle.gpu.barrier_arrive(full[0])
                    value *= correction[:, None]
                    value = tle.gpu.wgmma(
                        sp,
                        tle_subslice(sv0, [0, 0], [LEADER_D, BLOCK_K]),
                        value,
                        trans_b=True,
                    )
                    value = tle.gpu.wgmma_wait(0, value)
                    previous_scale = probability_scale
                    maximum = next_maximum
                    tle.gpu.barrier_wait(empty[0])
                else:
                    pass
        output_batch = batch * SPLITS + split
        if SPLITS == 1:
            logsum = tl.where(
                denominator > 0,
                maximum * 0.6931471805599453 + tl.log(denominator),
                float("inf"),
            )
            inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
            if HAS_SINK:
                sink = tl.load(Sink + heads)
                inverse *= tl.where(
                    sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
                )
            else:
                pass
            output_scale = inverse * previous_scale
        else:
            output_scale = previous_scale
        tl.store(tle.gpu.local_ptr(factor), output_scale)
        tle.gpu.barrier_arrive(done[0], phaseIdx=0)
        value *= output_scale[:, None]
        tl.store(
            Output + (output_batch * H + heads[:, None]) * 512 + columns[None, :], value
        )
        if SPLITS == 1:
            tl.store(Stats + batch * H + heads, logsum)
        else:
            stats = Stats + (output_batch * H + heads) * 2
            tl.store(stats, maximum * 0.6931471805599453)
            tl.store(stats + 1, denominator)
        tl.store(
            RepairFlags + (batch * (H // 64) + head_group) * SPLITS + split,
            needs_repair.to(tl.int32),
        )

    @triton.jit
    def sparse_fp8_compact_follower(
        Length,
        Output,
        sp,
        sv0,
        sv1,
        alpha,
        factor,
        full,
        empty,
        done,
        H: tl.constexpr,
        TOPK: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch = tl.program_id(0)
        split = tl.program_id(2)
        heads = tl.program_id(1) * 64 + tl.arange(0, 64)
        columns = 256 + tl.arange(0, 256)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
        start = split * blocks_per_split * BLOCK_K
        stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
        value = tl.zeros((64, 256), tl.float32)
        if BLOCK_K == 128:
            extra = tl.zeros((64, 128), tl.float32)
        else:
            pass
        for base in range(start, stop, BLOCK_K):
            tle.gpu.barrier_wait(full[0])
            correction = tl.load(tle.gpu.local_ptr(alpha))
            value *= correction[:, None]
            if BLOCK_K == 128:
                extra *= correction[:, None]
                extra = tle.gpu.wgmma(
                    sp,
                    tle_subslice(sv0, [128, 0], [128, BLOCK_K]),
                    extra,
                    trans_b=True,
                )
            else:
                pass
            value = tle.gpu.wgmma(sp, sv1, value, trans_b=True)
            value = tle.gpu.wgmma_wait(0, value)
            if BLOCK_K == 128:
                extra = tle.gpu.wgmma_wait(0, extra)
            else:
                pass
            tle.gpu.barrier_arrive(empty[0])
        tle.gpu.barrier_wait(done[0], phaseIdx=0)
        output_scale = tl.load(tle.gpu.local_ptr(factor))
        tl.store(
            Output
            + ((batch * SPLITS + split) * H + heads[:, None]) * 512
            + columns[None, :],
            value * output_scale[:, None],
        )
        if BLOCK_K == 128:
            tl.store(
                Output
                + ((batch * SPLITS + split) * H + heads[:, None]) * 512
                + 128
                + tl.arange(0, 128)[None, :],
                extra * output_scale[:, None],
            )
        else:
            pass

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("sparse_fp8_compact"),
        key=["B", "H", "TOPK", "SPLITS", "CAN_ASYNC", "HAS_ROPE"],
        use_cuda_graph=True,
    )
    @triton.jit
    def sparse_fp8_compact(
        Q,
        QR,
        KV,
        KR,
        QS,
        KS,
        Indices,
        Length,
        Sink,
        Output,
        Stats,
        RepairFlags,
        qb: tl.constexpr,
        qh: tl.constexpr,
        qrb: tl.constexpr,
        qrh: tl.constexpr,
        kp: tl.constexpr,
        kt: tl.constexpr,
        krp: tl.constexpr,
        krt: tl.constexpr,
        qsb: tl.constexpr,
        qsh: tl.constexpr,
        ksp: tl.constexpr,
        kst: tl.constexpr,
        ib: tl.constexpr,
        ik: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        HAS_SINK: tl.constexpr,
        FOLLOWER_REGS: tl.constexpr,
        CAN_ASYNC: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_ROPE: tl.constexpr = True,
    ):
        sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
        # A double-buffered 128-key tile would exceed Hopper shared-memory capacity.
        SLOTS: tl.constexpr = 2 if BLOCK_K == 64 else 1
        sk = tle.gpu.alloc([SLOTS, BLOCK_K, 512], tl.float8e4nv, scope=tle.gpu.smem)
        skr = tle.gpu.alloc([SLOTS, BLOCK_K, 64], tl.bfloat16, scope=tle.gpu.smem)
        scales = tle.gpu.alloc(
            [SLOTS, BLOCK_K], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        masks = tle.gpu.alloc(
            [SLOTS, BLOCK_K], tl.int32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        sp = tle.gpu.alloc([64, BLOCK_K], tl.float8e4nv, scope=tle.gpu.smem)
        sv0 = tle.gpu.alloc([256, BLOCK_K], tl.float8e4nv, scope=tle.gpu.smem)
        sv1 = tle.gpu.alloc([256, BLOCK_K], tl.float8e4nv, scope=tle.gpu.smem)
        alpha = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        factor = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        full = sparse_named_barriers(1, 256, 16)
        empty = sparse_named_barriers(1, 256, 17)
        done = tle.gpu.alloc_barriers(1, arrive_count=1)
        tle.gpu.warp_specialize(
            [
                (
                    sparse_fp8_compact_leader,
                    (
                        Q,
                        QR,
                        KV,
                        KR,
                        QS,
                        KS,
                        Indices,
                        Length,
                        Sink,
                        Output,
                        Stats,
                        RepairFlags,
                        qb,
                        qh,
                        qrb,
                        qrh,
                        kp,
                        kt,
                        krp,
                        krt,
                        qsb,
                        qsh,
                        ksp,
                        kst,
                        ib,
                        ik,
                        sq,
                        sr,
                        sk,
                        skr,
                        scales,
                        masks,
                        sp,
                        sv0,
                        sv1,
                        alpha,
                        factor,
                        full,
                        empty,
                        done,
                        H,
                        N,
                        TOPK,
                        SCALE,
                        SPLITS,
                        HAS_LENGTH,
                        HAS_SINK,
                        CAN_ASYNC,
                        BLOCK_K,
                        HAS_ROPE,
                    ),
                ),
                (
                    sparse_fp8_compact_follower,
                    (
                        Length,
                        Output,
                        sp,
                        sv0,
                        sv1,
                        alpha,
                        factor,
                        full,
                        empty,
                        done,
                        H,
                        TOPK,
                        SPLITS,
                        HAS_LENGTH,
                        BLOCK_K,
                    ),
                ),
            ],
            [4],
            [FOLLOWER_REGS],
        )

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("sparse_fp8_tile"),
        key=["B", "H", "TOPK", "SPLITS", "CAN_ASYNC", "HAS_ROPE"],
        use_cuda_graph=True,
    )
    @triton.jit
    def sparse_fp8_tile(
        Q,
        QR,
        KV,
        KR,
        QS,
        KS,
        Indices,
        Length,
        Sink,
        Output,
        Stats,
        qb: tl.constexpr,
        qh: tl.constexpr,
        qrb: tl.constexpr,
        qrh: tl.constexpr,
        kp: tl.constexpr,
        kt: tl.constexpr,
        krp: tl.constexpr,
        krt: tl.constexpr,
        qsb: tl.constexpr,
        qsh: tl.constexpr,
        ksp: tl.constexpr,
        kst: tl.constexpr,
        ib: tl.constexpr,
        ik: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        HAS_SINK: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAN_ASYNC: tl.constexpr,
        HAS_ROPE: tl.constexpr = True,
    ):
        tl.static_assert(TOPK <= 64 * SPLITS)
        batch = tl.program_id(0)
        head_group = tl.program_id(1)
        heads = head_group * 64 + tl.arange(0, 64)
        value_group = tl.program_id(2) % (512 // BLOCK_D)
        split = tl.program_id(2) // (512 // BLOCK_D)
        features = tl.arange(0, 512)
        ropes = tl.arange(0, 64)
        keys = tl.arange(0, BLOCK_K)
        columns = value_group * BLOCK_D + tl.arange(0, BLOCK_D)
        sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
        sk = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        skr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
        sp = tle.gpu.alloc([64, 64], tl.float8e4nv, scope=tle.gpu.smem)
        sv = tle.gpu.alloc([BLOCK_D, 64], tl.float8e4nv, scope=tle.gpu.smem)
        query = tl.load(
            Q + batch * qb + heads[:, None] * qh + features[None, :],
            volatile=not CAN_ASYNC,
        )
        tl.store(tle.gpu.local_ptr(sq), query)
        if HAS_ROPE:
            query_rope = tl.load(
                QR + batch * qrb + heads[:, None] * qrh + ropes[None, :],
                volatile=not CAN_ASYNC,
            )
            tl.store(tle.gpu.local_ptr(sr), query_rope)
        else:
            pass
        query_scale = tl.load(QS + batch * qsb + heads * qsh) * (
            SCALE * 1.4426950408889634
        )
        amplification = tl.max(tl.abs(query_scale), 0)
        maximum = tl.full((64,), -float("inf"), tl.float32)
        denominator = tl.zeros((64,), tl.float32)
        value = tl.zeros((64, BLOCK_D), tl.float32)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
        start = split * blocks_per_split * BLOCK_K
        stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
        for base in range(start, stop, BLOCK_K):
            positions = base + keys
            ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
            valid = (positions < length) & (ids >= 0) & (ids < N)
            ids = tl.where(valid, ids, 0).to(tl.uint64)
            pages, slots = ids // 64, ids % 64
            cache = tl.load(
                KV + pages[:, None] * kp + slots[:, None] * kt + features[None, :],
                valid[:, None],
                0.0,
                volatile=not CAN_ASYNC,
            )
            tl.store(tle.gpu.local_ptr(sk), cache)
            if HAS_ROPE:
                cache_rope = tl.load(
                    KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
                    valid[:, None],
                    0.0,
                    volatile=not CAN_ASYNC,
                )
                tl.store(tle.gpu.local_ptr(skr), cache_rope)
            else:
                pass
            tl.debug_barrier()
            content_logits = tle.gpu.wgmma(sq, sk, out_dtype=tl.float32, trans_b=True)
            if HAS_ROPE:
                logits = tle.gpu.wgmma(sr, skr, content_logits, trans_b=True)
            else:
                logits = content_logits
            logits = tle.gpu.wgmma_wait(0, logits)
            kv_scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
            if amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD:
                exact = sparse_fp8_precise_scores(
                    Q,
                    KV,
                    Indices,
                    batch,
                    heads,
                    base,
                    length,
                    logits,
                    qb,
                    qh,
                    kp,
                    kt,
                    ib,
                    ik,
                    N,
                )
                if HAS_ROPE:
                    rope_scores = tle.gpu.wgmma(
                        sr, skr, out_dtype=tl.float32, trans_b=True
                    )
                    rope_scores = tle.gpu.wgmma_wait(0, rope_scores)
                    logits = exact + rope_scores
                else:
                    logits = exact
            else:
                pass
            logits = logits * query_scale[:, None] * kv_scale[None, :]
            logits = tl.where(valid[None, :], logits, -float("inf"))
            next_maximum = tl.max(logits, 1)
            safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
            probabilities = tl.exp2(logits - safe_maximum[:, None])
            denominator = tl.sum(probabilities, 1)
            weighted = probabilities * kv_scale[None, :]
            probability_scale = tl.max(weighted, 1) / 448.0
            probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
            publish_p_fp8_sw64_coupled_stmatrix(
                sp, weighted / probability_scale[:, None]
            )
            for candidate_group in tl.static_range(512 // BLOCK_D):
                if value_group == candidate_group:
                    for tile in tl.static_range(BLOCK_D // 128):
                        source = tle_subslice(
                            sk, [0, candidate_group * BLOCK_D + tile * 128], [64, 128]
                        )
                        vtranspose_fp8_64x128_kperm(source, sv, tile * 128)
                else:
                    pass
            tl.inline_asm_elementwise(
                "{ fence.proxy.async.shared::cta; mov.b32 $0, $1; }",
                "=r,r",
                [tl.arange(0, 128)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            tl.debug_barrier()
            value = tle.gpu.wgmma(sp, sv, out_dtype=tl.float32, trans_b=True)
            value = tle.gpu.wgmma_wait(0, value)
            value *= probability_scale[:, None]
            maximum = next_maximum
        output_batch = batch * SPLITS + split
        if SPLITS == 1:
            logsum = tl.where(
                denominator > 0,
                maximum * 0.6931471805599453 + tl.log(denominator),
                float("inf"),
            )
            inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
            if HAS_SINK:
                sink = tl.load(Sink + heads)
                inverse *= tl.where(
                    sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
                )
            else:
                pass
            value *= inverse[:, None]
        else:
            pass
        tl.store(
            Output + (output_batch * H + heads[:, None]) * 512 + columns[None, :], value
        )
        if value_group == 0:
            if SPLITS == 1:
                tl.store(Stats + batch * H + heads, logsum)
            else:
                stats = Stats + (output_batch * H + heads) * 2
                tl.store(stats, maximum * 0.6931471805599453)
                tl.store(stats + 1, denominator)
        else:
            pass

    @triton.jit
    def sparse_fp8_producer(
        Q,
        QR,
        KV,
        KR,
        KS,
        Indices,
        Length,
        qb,
        qh,
        qrb,
        qrh,
        kp,
        kt,
        krp,
        krt,
        ksp,
        kst,
        ib,
        ik,
        sq,
        sr,
        sk,
        skr,
        scales,
        mask,
        cached_indices,
        qfull,
        kfull,
        kfree,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_ROPE: tl.constexpr = True,
        PRODUCER_WARPS: tl.constexpr = 4,
        CAN_ASYNC_NOPE: tl.constexpr = False,
        CAN_ASYNC_ROPE: tl.constexpr = False,
    ):
        batch = tl.program_id(0) // (H // 64)
        heads = (tl.program_id(0) % (H // 64)) * 64 + tl.arange(0, 64)
        dims = tl.arange(0, 512)
        ropes = tl.arange(0, 64)
        CACHE_INDICES: tl.constexpr = not HAS_ROPE and SPLITS == 1 and TOPK <= 1024
        ASYNC_KR: tl.constexpr = (
            HAS_ROPE and CAN_ASYNC_ROPE and CAN_ASYNC_NOPE and PRODUCER_WARPS == 8
        )
        if CACHE_INDICES:
            CACHE_SLOTS: tl.constexpr = triton.next_power_of_2(max(TOPK, 128)) // 128
            cache_length = (
                tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
                if HAS_LENGTH
                else TOPK
            )
            cache_positions = (
                tl.arange(0, CACHE_SLOTS)[:, None] * 128 + tl.arange(0, 128)[None, :]
            )
            index_data = tl.load(
                Indices + batch * ib + cache_positions * ik,
                cache_positions < cache_length,
                -1,
            )
            tl.store(tle.gpu.local_ptr(cached_indices), index_data)
        else:
            pass
        query = tl.load(
            Q + batch * qb + heads[:, None] * qh + dims[None, :],
            eviction_policy="evict_first",
        )
        tl.store(tle.gpu.local_ptr(sq), query)
        if HAS_ROPE:
            query_rope = tl.load(
                QR + batch * qrb + heads[:, None] * qrh + ropes[None, :],
                eviction_policy="evict_first",
            )
            tl.store(tle.gpu.local_ptr(sr), query_rope)
        else:
            pass
        tle.gpu.barrier_arrive(qfull[0])
        length = (
            cache_length
            if CACHE_INDICES
            else (
                tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
                if HAS_LENGTH
                else TOPK
            )
        )
        first_block, key_blocks = sparse_fp8_split_blocks(length, SPLITS)
        split_blocks = tl.cdiv(key_blocks, 2)
        split_end = tl.minimum(length, (first_block + key_blocks) * 64)
        for step in range(split_blocks):
            buf = 0
            meta_buf = step % 2
            positions = first_block * 64 + step * 128 + tl.arange(0, 128)
            if CACHE_INDICES:
                ids = tl.load(tle.gpu.local_ptr(cached_indices.slot(step)))
            else:
                ids = tl.load(
                    Indices + batch * ib + positions * ik,
                    positions < split_end,
                    -1,
                    eviction_policy="evict_first",
                )
            valid = (positions < split_end) & (ids >= 0) & (ids < N)
            ids = tl.where(valid, ids, 0).to(tl.uint64)
            pages, slots = (ids // 64, ids % 64)
            if CAN_ASYNC_NOPE:
                copy_rows = tl.broadcast_to(tl.arange(0, 128)[:, None], (128, 8))
                copy_columns = tl.broadcast_to(
                    (tl.arange(0, 8) * 16)[None, :], (128, 8)
                )
                for key_half in tl.static_range(2):
                    if step >= 1:
                        tle.gpu.barrier_wait(kfree[key_half])
                    else:
                        pass
                    for group in tl.static_range(4):
                        dst = tle.gpu.local_ptr(
                            tle_subslice(sk.slot(buf), [0, group * 128], [128, 128]),
                            (copy_rows, copy_columns),
                        )
                        src = (
                            KV
                            + pages[:, None] * kp
                            + slots[:, None] * kt
                            + group * 128
                            + copy_columns
                        )
                        # Keep the first K64 copies pending across the second free wait.
                        tl.inline_asm_elementwise(
                            "{ .reg .pred active; setp.ne.u32 active, $4, 0; "
                            "@active cp.async.cg.shared.global [$1], [$2], 16, $3; "
                            "mov.u32 $0, 0; }",
                            constraints="=r,r,l,r,r",
                            args=[
                                dst.to(tl.uint64).to(tl.uint32),
                                src.to(tl.uint64),
                                tl.where(valid[:, None], 16, 0),
                                ((copy_rows // 64) == key_half).to(tl.uint32),
                            ],
                            dtype=tl.uint32,
                            is_pure=False,
                            pack=1,
                        )
                    # Consumer 0 releases all RoPE rows with kfree[0].
                    if key_half == 0 and ASYNC_KR:
                        rope_columns = tl.broadcast_to(
                            (tl.arange(0, 8) * 8)[None, :], (128, 8)
                        )
                        dst_r = tle.gpu.local_ptr(
                            skr.slot(buf), (copy_rows, rope_columns)
                        )
                        src_r = (
                            KR
                            + pages[:, None] * krp
                            + slots[:, None] * krt
                            + rope_columns
                        )
                        tl.inline_asm_elementwise(
                            "{ cp.async.cg.shared.global [$1], [$2], 16, $3; "
                            "mov.u32 $0, 0; }",
                            constraints="=r,r,l,r",
                            args=[
                                dst_r.to(tl.uint64).to(tl.uint32),
                                src_r.to(tl.uint64),
                                tl.where(valid[:, None], 16, 0).to(tl.uint32),
                            ],
                            dtype=tl.uint32,
                            is_pure=False,
                            pack=1,
                        )
                    else:
                        pass
            else:
                if step >= 1:
                    tle.gpu.barrier_wait(kfree[0])
                    tle.gpu.barrier_wait(kfree[1])
                else:
                    pass
                columns = tl.arange(0, 128)
                for group in tl.static_range(4):
                    content = tl.load(
                        KV
                        + pages[:, None] * kp
                        + slots[:, None] * kt
                        + group * 128
                        + columns[None, :],
                        valid[:, None],
                        0.0,
                    )
                    tl.store(
                        tle.gpu.local_ptr(
                            tle_subslice(sk.slot(buf), [0, group * 128], [128, 128])
                        ),
                        content,
                    )
            if HAS_ROPE and not ASYNC_KR:
                rope = tl.load(
                    KR + pages[:, None] * krp + slots[:, None] * krt + ropes[None, :],
                    valid[:, None],
                    0.0,
                )
                tl.store(tle.gpu.local_ptr(skr.slot(buf)), rope)
            else:
                pass
            kv_scale = tl.load(KS + pages * ksp + slots * kst, valid, 0.0)
            tl.store(tle.gpu.local_ptr(scales.slot(meta_buf)), kv_scale)
            if HAS_ROPE or TOPK > 512:
                tl.store(
                    tle.gpu.local_ptr(mask.slot(meta_buf)),
                    tl.where(valid, 0.0, -float("inf")),
                )
            else:
                pass
            if CAN_ASYNC_NOPE:
                tl.inline_asm_elementwise(
                    "{ cp.async.commit_group; cp.async.wait_group 0; mov.u32 $0, $1; }",
                    constraints="=r,r",
                    args=[tl.arange(0, PRODUCER_WARPS * 32)],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
            else:
                pass
            # Matrix transpose avoids byte stores and their shared-memory bank conflicts.
            tl.debug_barrier()
            tl.inline_asm_elementwise(
                "{ fence.proxy.async.shared::cta; mov.u32 $0, $1; }",
                constraints="=r,r",
                args=[tl.arange(0, 128)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            tle.gpu.barrier_arrive(kfull[0])
            tle.gpu.barrier_arrive(kfull[1])

    @triton.jit
    def sparse_fp8_consumer0(
        QS,
        Length,
        Sink,
        Output,
        LSE,
        qsb,
        qsh,
        sq,
        sr,
        sk,
        skr,
        sv0,
        sp,
        alpha,
        scales,
        mask,
        factor,
        qfull,
        kfull,
        kfree,
        vfull,
        vfree,
        pfull,
        ofull,
        H: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_SINK: tl.constexpr,
        RepairFlags,
        HAS_ROPE: tl.constexpr = True,
        PREQUANTIZE_P: tl.constexpr = False,
    ):
        batch = tl.program_id(0) // (H // 64)
        heads = (tl.program_id(0) % (H // 64)) * 64 + tl.arange(0, 64)
        # Keep online softmax in base-2 units and fold row-constant scaling once.
        query_scale = tl.load(
            QS + batch * qsb + heads * qsh, eviction_policy="evict_first"
        ) * (SCALE * 1.4426950408889634)
        amplification = tl.max(tl.abs(query_scale), 0)
        key_repair = tl.full((128,), False, tl.int1)
        head_repair = tl.full((64,), False, tl.int1)
        maximum_weight_scale = tl.zeros((64,), tl.float32)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        maximum = tl.full((64,), -float("inf"), tl.float32)
        denominator = tl.zeros((64,), tl.float32)
        acc0 = tl.zeros((64, 128), tl.float32)
        previous_scale = tl.full((64,), 1.0, tl.float32)
        tle.gpu.barrier_wait(qfull[0])
        first_block, key_blocks = sparse_fp8_split_blocks(length, SPLITS)
        split_blocks = tl.cdiv(key_blocks, 2)
        for step in range(split_blocks):
            meta_buf = step % 2
            tle.gpu.barrier_wait(kfull[0])
            content_logits = tle.gpu.wgmma(
                sq, sk.slot(0), out_dtype=tl.float32, trans_b=True
            )
            if HAS_ROPE:
                logits = tle.gpu.wgmma(sr, skr.slot(0), content_logits, trans_b=True)
            else:
                logits = content_logits
            logits = tle.gpu.wgmma_wait(0, logits)
            tle.gpu.barrier_arrive(kfree[0])
            tle.gpu.barrier_arrive(kfree[1])
            kv_scale = tl.load(tle.gpu.local_ptr(scales.slot(meta_buf)))
            if HAS_ROPE or TOPK > 512:
                add_mask = tl.load(tle.gpu.local_ptr(mask.slot(meta_buf)))
            else:
                add_mask = tl.where(
                    kv_scale.to(tl.uint32, bitcast=True) != 0,
                    0.0,
                    -float("inf"),
                )
            key_repair |= amplification * kv_scale > QK_RECOMPUTE_THRESHOLD
            scores = (
                logits * query_scale[:, None] * kv_scale[None, :] + add_mask[None, :]
            )
            next_maximum = tl.maximum(maximum, tl.max(scores, 1))
            safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
            correction = tl.exp2(maximum - safe_maximum)
            probabilities = tl.exp2(scores - safe_maximum[:, None])
            denominator = denominator * correction + tl.sum(probabilities, 1)
            weighted = probabilities * kv_scale[None, :]
            probability_scale = tl.max(weighted, 1) / 448.0
            next_weight_scale = tl.maximum(
                maximum_weight_scale * correction, probability_scale
            )
            probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
            head_repair |= (
                next_weight_scale * ACCUMULATOR_SCALE_FLOOR > probability_scale
            )
            if PREQUANTIZE_P:
                p = quantize_p_fp8_before_ready(weighted / probability_scale[:, None])
            else:
                p = weighted / probability_scale[:, None]
            beta = correction * previous_scale / probability_scale
            tle.gpu.barrier_wait(vfull[0])
            tle.gpu.barrier_wait(vfull[1])
            publish_sparse_p128_fp8(
                sp.slot(0), p, sp.slot(1), PREQUANTIZED=PREQUANTIZE_P
            )
            tl.store(tle.gpu.local_ptr(alpha.slot(0)), beta)
            tl.debug_barrier()
            tle.gpu.barrier_arrive(pfull[0])
            tle.gpu.barrier_arrive(pfull[1])
            acc0 *= beta[:, None]
            for half in tl.static_range(2):
                acc0 = tle.gpu.wgmma(
                    sp.slot(half),
                    tle_subslice(sv0.slot(half), [0, 0], [128, 64]),
                    acc0,
                    trans_b=True,
                )
            acc0 = tle.gpu.wgmma_wait(0, acc0)
            maximum = next_maximum
            maximum_weight_scale = next_weight_scale
            previous_scale = probability_scale
            tle.gpu.barrier_arrive(vfree[0])
        if SPLITS == 1:
            logsum = tl.where(
                denominator > 0,
                (maximum + tl.log2(denominator)) * 0.6931471805599453,
                float("inf"),
            )
            denominator_inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
            if HAS_SINK:
                sink = tl.load(Sink + heads)
                inverse = sink_normalization(denominator_inverse, sink, logsum)
            else:
                inverse = denominator_inverse
            inverse *= previous_scale
        else:
            # Keep partial values in physical units; merge applies the sink only once.
            inverse = previous_scale
        output_batch = batch if SPLITS == 1 else batch * SPLITS + tl.program_id(2)
        tl.store(tle.gpu.local_ptr(factor), inverse)
        tle.gpu.barrier_arrive(ofull[0], phaseIdx=0)
        sparse_fp8_publish_output(
            acc0,
            inverse,
            Output,
            output_batch,
            (tl.program_id(0) % (H // 64)),
            H,
            0,
            STREAMING=SPLITS == 1,
        )
        if SPLITS == 1:
            tl.store(LSE + batch * H + heads, logsum, eviction_policy="evict_first")
        else:
            stats = LSE + ((batch * SPLITS + tl.program_id(2)) * H + heads) * 2
            tl.store(stats, maximum * 0.6931471805599453)
            tl.store(stats + 1, denominator)
        needs_repair = (tl.max(key_repair.to(tl.int32), 0) != 0) | (
            tl.max(head_repair.to(tl.int32), 0) != 0
        )
        tl.store(
            RepairFlags
            + (batch * (H // 64) + (tl.program_id(0) % (H // 64))) * SPLITS
            + tl.program_id(2),
            needs_repair.to(tl.int32),
        )

    @triton.jit
    def sparse_fp8_consumer1(
        Length,
        Output,
        sk,
        sv0,
        sv1,
        sv_tail,
        sp,
        alpha,
        factor,
        kfull,
        kfree,
        vfull,
        vfree,
        pfull,
        ofull,
        H: tl.constexpr,
        TOPK: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        batch = tl.program_id(0) // (H // 64)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        acc0 = tl.zeros((64, 128), tl.float32)
        acc1 = tl.zeros((64, 128), tl.float32)
        acc2 = tl.zeros((64, 128), tl.float32)
        first_block, key_blocks = sparse_fp8_split_blocks(length, SPLITS)
        split_blocks = tl.cdiv(key_blocks, 2)
        for step in range(split_blocks):
            buf = 0
            tle.gpu.barrier_wait(kfull[1])
            # Complete K64 transposes permit the next half-copy to start early.
            for outer in tl.static_range(2):
                for inner in tl.static_range(2):
                    if inner == 0:
                        source = tle_subslice(
                            sk.slot(buf),
                            [outer * 64, 0],
                            [64, 128],
                        )
                        for warp_half in tl.static_range(2):
                            sparse_fp8_transpose_values(
                                source,
                                sv0.slot(outer),
                                0,
                                warp_half * 4,
                            )
                    else:
                        for group in tl.static_range(1, 4):
                            source = tle_subslice(
                                sk.slot(buf),
                                [outer * 64, group * 128],
                                [64, 128],
                            )
                            for warp_half in tl.static_range(2):
                                if group == 3:
                                    sparse_fp8_transpose_values(
                                        source,
                                        sv_tail.slot(outer),
                                        0,
                                        warp_half * 4,
                                    )
                                else:
                                    sparse_fp8_transpose_values(
                                        source,
                                        sv1.slot(outer),
                                        (group - 1) * 128,
                                        warp_half * 4,
                                    )
                    tl.inline_asm_elementwise(
                        "{ fence.proxy.async.shared::cta; mov.u32 $0, $1; }",
                        constraints="=r,r",
                        args=[tl.arange(0, 128)],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                    if inner == 0:
                        tle.gpu.barrier_arrive(vfull[outer])
                    else:
                        tle.gpu.barrier_arrive(kfree[outer])
            for half in tl.static_range(2):
                tle.gpu.barrier_wait(pfull[half])
                if half == 0:
                    correction = tl.load(tle.gpu.local_ptr(alpha.slot(half)))
                    acc0 *= correction[:, None]
                    acc1 *= correction[:, None]
                    acc2 *= correction[:, None]
                else:
                    pass
                acc0 = tle.gpu.wgmma(
                    sp.slot(half),
                    tle_subslice(sv1.slot(half), [0, 0], [128, 64]),
                    acc0,
                    trans_b=True,
                )
                acc1 = tle.gpu.wgmma(
                    sp.slot(half),
                    tle_subslice(sv1.slot(half), [128, 0], [128, 64]),
                    acc1,
                    trans_b=True,
                )
                acc2 = tle.gpu.wgmma(
                    sp.slot(half),
                    sv_tail.slot(half),
                    acc2,
                    trans_b=True,
                )
                if half == 1:
                    acc0 = tle.gpu.wgmma_wait(0, acc0)
                    acc1 = tle.gpu.wgmma_wait(0, acc1)
                    acc2 = tle.gpu.wgmma_wait(0, acc2)
                else:
                    pass
            tle.gpu.barrier_wait(vfree[0])
        tle.gpu.barrier_wait(ofull[0], phaseIdx=0)
        inverse = tl.load(tle.gpu.local_ptr(factor))
        output_batch = batch if SPLITS == 1 else batch * SPLITS + tl.program_id(2)
        sparse_fp8_publish_output(
            acc0,
            inverse,
            Output,
            output_batch,
            (tl.program_id(0) % (H // 64)),
            H,
            128,
            STREAMING=SPLITS == 1,
        )
        sparse_fp8_publish_output(
            acc1,
            inverse,
            Output,
            output_batch,
            (tl.program_id(0) % (H // 64)),
            H,
            256,
            STREAMING=SPLITS == 1,
        )

        sparse_fp8_publish_output(
            acc2,
            inverse,
            Output,
            output_batch,
            (tl.program_id(0) % (H // 64)),
            H,
            384,
            STREAMING=SPLITS == 1,
        )

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("sparse_fp8_warp_specialized"),
        key=[
            "B_BUCKET",
            "H",
            "TOPK",
            "SPLITS",
            "HAS_ROPE",
            "N",
            "CAN_ASYNC_NOPE",
            "CAN_ASYNC_ROPE",
        ],
    )
    @triton.jit
    def sparse_fp8_warp_specialized(
        Q,
        QR,
        KV,
        KR,
        QS,
        KS,
        Indices,
        Length,
        Sink,
        Output,
        LSE,
        qb: tl.constexpr,
        qh: tl.constexpr,
        qrb: tl.constexpr,
        qrh: tl.constexpr,
        kp: tl.constexpr,
        kt: tl.constexpr,
        krp: tl.constexpr,
        krt: tl.constexpr,
        qsb: tl.constexpr,
        qsh: tl.constexpr,
        ksp: tl.constexpr,
        kst: tl.constexpr,
        ib: tl.constexpr,
        ik: tl.constexpr,
        B_BUCKET: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_SINK: tl.constexpr,
        RepairFlags,
        SMALL_REGS: tl.constexpr,
        SMALL_WARPS: tl.constexpr,
        LARGE_REGS: tl.constexpr,
        LARGE_WARPS: tl.constexpr,
        HAS_ROPE: tl.constexpr = True,
        CAN_ASYNC_NOPE: tl.constexpr = False,
        CAN_ASYNC_ROPE: tl.constexpr = False,
    ):
        PRODUCER_WARPS: tl.constexpr = SMALL_WARPS if B_BUCKET == 0 else LARGE_WARPS
        PRODUCER_REGS: tl.constexpr = SMALL_REGS if B_BUCKET == 0 else LARGE_REGS
        # H800 has 50 MiB L2; large RoPE KV sets benefit from prepacking P.
        PREQUANTIZE_P: tl.constexpr = (
            HAS_ROPE and N * (512 + 64 * 2 + 4) > 50 * 1024 * 1024
        )
        COPY_REGS: tl.constexpr = (
            40
            if PREQUANTIZE_P and PRODUCER_WARPS == 8 and PRODUCER_REGS == 32
            else PRODUCER_REGS
        )
        sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
        # Paired copies overlap NoPE buffer reuse with consumer compute.
        sk = tle.gpu.alloc([1, 128, 512], tl.float8e4nv, scope=tle.gpu.smem)
        skr = tle.gpu.alloc([1, 128, 64], tl.bfloat16, scope=tle.gpu.smem)
        sv0 = tle.gpu.alloc([2, 128, 64], tl.float8e4nv, scope=tle.gpu.smem)
        sv1 = tle.gpu.alloc([2, 256, 64], tl.float8e4nv, scope=tle.gpu.smem)
        sv_tail = tle.gpu.alloc([2, 128, 64], tl.float8e4nv, scope=tle.gpu.smem)
        sp = tle.gpu.alloc([2, 64, 64], tl.float8e4nv, scope=tle.gpu.smem)
        alpha = tle.gpu.alloc(
            [2, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        # Metadata survives early K reuse, including compiler-rematerialized loads.
        scales = tle.gpu.alloc(
            [2, 128], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if HAS_ROPE or TOPK > 512:
            mask = tle.gpu.alloc(
                [2, 128], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
        else:
            # Preserve private helper arguments without unused mask storage.
            mask = scales
        CACHE_INDICES: tl.constexpr = not HAS_ROPE and SPLITS == 1 and TOPK <= 1024
        if CACHE_INDICES:
            CACHE_SLOTS: tl.constexpr = triton.next_power_of_2(max(TOPK, 128)) // 128
            cached_indices = tle.gpu.alloc(
                [CACHE_SLOTS, 128],
                tl.int32,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            cached_indices = mask
        factor = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        # TLE maps these virtual IDs to physical IDs outside the WS reserved set.
        qfull = sparse_named_barriers(1, PRODUCER_WARPS * 32 + 128, 16)
        kfull = sparse_named_barriers(2, PRODUCER_WARPS * 32 + 128, 17)
        kfree = sparse_named_barriers(2, PRODUCER_WARPS * 32 + 256, 19)
        vfull = sparse_named_barriers(2, 256, 24)
        vfree = sparse_named_barriers(1, 256, 23)
        pfull = sparse_named_barriers(2, 256, 21)
        ofull = tle.gpu.alloc_barriers(1, arrive_count=1)
        tle.gpu.warp_specialize(
            [
                (
                    sparse_fp8_consumer0,
                    (
                        QS,
                        Length,
                        Sink,
                        Output,
                        LSE,
                        qsb,
                        qsh,
                        sq,
                        sr,
                        sk,
                        skr,
                        sv0,
                        sp,
                        alpha,
                        scales,
                        mask,
                        factor,
                        qfull,
                        kfull,
                        kfree,
                        vfull,
                        vfree,
                        pfull,
                        ofull,
                        H,
                        TOPK,
                        SCALE,
                        HAS_LENGTH,
                        SPLITS,
                        HAS_SINK,
                        RepairFlags,
                        HAS_ROPE,
                        PREQUANTIZE_P,
                    ),
                ),
                (
                    sparse_fp8_producer,
                    (
                        Q,
                        QR,
                        KV,
                        KR,
                        KS,
                        Indices,
                        Length,
                        qb,
                        qh,
                        qrb,
                        qrh,
                        kp,
                        kt,
                        krp,
                        krt,
                        ksp,
                        kst,
                        ib,
                        ik,
                        sq,
                        sr,
                        sk,
                        skr,
                        scales,
                        mask,
                        cached_indices,
                        qfull,
                        kfull,
                        kfree,
                        H,
                        N,
                        TOPK,
                        HAS_LENGTH,
                        SPLITS,
                        HAS_ROPE,
                        PRODUCER_WARPS,
                        CAN_ASYNC_NOPE,
                        CAN_ASYNC_ROPE,
                    ),
                ),
                (
                    sparse_fp8_consumer1,
                    (
                        Length,
                        Output,
                        sk,
                        sv0,
                        sv1,
                        sv_tail,
                        sp,
                        alpha,
                        factor,
                        kfull,
                        kfree,
                        vfull,
                        vfree,
                        pfull,
                        ofull,
                        H,
                        TOPK,
                        HAS_LENGTH,
                        SPLITS,
                    ),
                ),
            ],
            [PRODUCER_WARPS, 4],
            [
                COPY_REGS,
                (240),
            ],
        )

    @libentry()
    @triton.jit
    def sparse_fp8_empty(Output, LSE, ROWS: tl.constexpr):
        # Fixed write-only geometry; no attention tiling or reduction is performed.
        offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
        tl.store(Output + offsets, 0.0, offsets < ROWS * 512)
        tl.store(LSE + offsets, float("inf"), offsets < ROWS)

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("sparse_fp8_repair"),
        key=["H", "TOPK", "SPLITS", "HAS_ROPE", "REPAIR_GROUP"],
    )
    @triton.jit
    def sparse_fp8_repair(
        Q,
        QRope,
        KV,
        KVRope,
        Indices,
        QScale,
        KVScale,
        Sink,
        Length,
        Output,
        LSE,
        Partial,
        Stats,
        stride_qb,
        stride_qh,
        stride_qrb,
        stride_qrh,
        stride_kvp,
        stride_kvt,
        stride_krp,
        stride_krt,
        stride_ib,
        stride_ik,
        stride_qsb,
        stride_qsh,
        stride_ksp,
        stride_kst,
        B,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SM_SCALE: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_SINK: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        RepairFlags,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        REPAIR_SPLITS: tl.constexpr = 1,
        HAS_ROPE: tl.constexpr = True,
        REPAIR_GROUP: tl.constexpr = 256,
    ):
        # Coarsen single-split flag scan to keep empty repair <= 1 wave.
        GROUP: tl.constexpr = REPAIR_GROUP if SPLITS == 1 else 1
        base_batch = tl.program_id(0) * GROUP
        head_group = tl.program_id(1)
        split = tl.program_id(2)
        batches = base_batch + tl.arange(0, GROUP)
        if SPLITS == 1 and REPAIR_SPLITS == 1:
            group_flags = tl.load(
                RepairFlags + batches * (H // 64) + head_group,
                batches < B,
                0,
            )
            if tl.max((group_flags != 0).to(tl.int32), 0) == 0:
                return
            else:
                pass
        else:
            flag_splits = tl.arange(0, REPAIR_SPLITS)
            group_flags = tl.load(
                RepairFlags
                + (batches[:, None] * (H // 64) + head_group) * REPAIR_SPLITS
                + flag_splits[None, :],
                batches[:, None] < B,
                0,
            )
            active_rows = (tl.max(group_flags, 1) != 0) & (batches < B)
            if tl.max(active_rows.to(tl.int32), 0) == 0:
                return
            else:
                pass
        for offset in range(GROUP):
            batch = base_batch + offset
            if GROUP == 1:
                # The initial check already established this sole task is active.
                active = True
            elif SPLITS == 1 and REPAIR_SPLITS == 1:
                flag = tl.load(
                    RepairFlags + batch * (H // 64) + head_group,
                    batch < B,
                    0,
                )
                active = (batch < B) & (flag != 0)
            else:
                flag_splits = tl.arange(0, REPAIR_SPLITS)
                # Do not retain the group flag vector across attention computation.
                flags = tl.load(
                    RepairFlags
                    + (batch * (H // 64) + head_group) * REPAIR_SPLITS
                    + flag_splits,
                    batch < B,
                    0,
                )
                active = (batch < B) & (tl.max(flags, 0) != 0)
            if active:
                heads = head_group * BLOCK_H + tl.arange(0, BLOCK_H)
                dims = tl.arange(0, 256)
                keys = tl.arange(0, BLOCK_K)
                q_ptr = (
                    Q + batch * stride_qb + heads[:, None] * stride_qh + dims[None, :]
                )
                query0 = tl.load(q_ptr, heads[:, None] < H, 0.0)
                query1 = tl.load(q_ptr + 256, heads[:, None] < H, 0.0)
                rope_dims = tl.arange(0, 64)
                if HAS_ROPE:
                    query_rope = tl.load(
                        QRope
                        + batch * stride_qrb
                        + heads[:, None] * stride_qrh
                        + rope_dims[None, :],
                        heads[:, None] < H,
                        0,
                    )
                else:
                    pass
                query_scale = tl.load(
                    QScale + batch * stride_qsb + heads * stride_qsh, heads < H, 0
                )
                amplification = tl.max(
                    tl.abs(query_scale * (SM_SCALE * 1.4426950408889634)), 0
                )
                maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
                denominator = tl.full((BLOCK_H,), 0, tl.float32)
                value0 = tl.full((BLOCK_H, 256), 0, tl.float32)
                value1 = tl.full((BLOCK_H, 256), 0, tl.float32)
                length = (
                    tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
                    if HAS_LENGTH
                    else TOPK
                )
                blocks_per_split: tl.constexpr = triton.cdiv(TOPK, BLOCK_K * SPLITS)
                start = split * blocks_per_split * BLOCK_K
                stop = tl.minimum(start + blocks_per_split * BLOCK_K, length)
                for base in range(start, stop, BLOCK_K):
                    positions = base + keys
                    ids = tl.load(
                        Indices + batch * stride_ib + positions * stride_ik,
                        positions < length,
                        -1,
                    )
                    valid = (positions < length) & (ids >= 0) & (ids < N)
                    ids = tl.where(valid, ids, 0).to(tl.int64)
                    pages = ids // 64
                    tokens = ids % 64
                    kv_ptr = (
                        KV
                        + pages[None, :] * stride_kvp
                        + tokens[None, :] * stride_kvt
                        + dims[:, None]
                    )
                    key0 = tl.load(kv_ptr, valid[None, :], 0.0)
                    key1 = tl.load(kv_ptr + 256, valid[None, :], 0.0)
                    kv_scale = tl.load(
                        KVScale + pages * stride_ksp + tokens * stride_kst, valid, 0
                    )
                    if amplification * tl.max(kv_scale, 0) > QK_RECOMPUTE_THRESHOLD:
                        logits = tl.zeros((BLOCK_H, BLOCK_K), tl.float32)
                        pad = tl.arange(0, 32)
                        for feature in range(512):
                            query_column = tl.load(
                                Q + batch * stride_qb + heads * stride_qh + feature,
                                heads < H,
                                0.0,
                            )
                            key_column = tl.load(
                                KV + pages * stride_kvp + tokens * stride_kvt + feature,
                                valid,
                                0.0,
                            )
                            query_feature = tl.where(
                                pad[None, :] == 0, query_column[:, None], 0.0
                            ).to(tl.float8e4nv)
                            key_feature = tl.where(
                                pad[:, None] == 0, key_column[None, :], 0.0
                            ).to(tl.float8e4nv)
                            contribution = tl.dot(
                                query_feature, key_feature, out_dtype=tl.float32
                            )
                            logits = sparse_fp8_accumulate(logits, contribution)
                    else:
                        logits = tl.dot(query0, key0, out_dtype=tl.float32)
                        logits = tl.dot(query1, key1, logits)
                    if HAS_ROPE:
                        key_rope = tl.load(
                            KVRope
                            + pages[None, :] * stride_krp
                            + tokens[None, :] * stride_krt
                            + rope_dims[:, None],
                            valid[None, :],
                            0,
                        )
                        rope_logits = tl.dot(query_rope, key_rope, out_dtype=tl.float32)
                        logits += rope_logits
                    else:
                        pass
                    logits = (
                        logits * query_scale[:, None] * kv_scale[None, :] * SM_SCALE
                    )
                    logits = tl.where(valid[None, :], logits, -float("inf"))
                    next_maximum = tl.maximum(maximum, tl.max(logits, 1))
                    safe_maximum = tl.where(
                        next_maximum == -float("inf"), 0.0, next_maximum
                    )
                    correction = tl.exp(maximum - safe_maximum)
                    probabilities = tl.exp(logits - safe_maximum[:, None])
                    denominator = denominator * correction + tl.sum(probabilities, 1)
                    # Fold token-dependent V scales into P so PV still uses FP8 operands.
                    weighted = probabilities * kv_scale[None, :]
                    probability_scale = tl.max(weighted, 1) / 448.0
                    probability_scale = tl.where(
                        probability_scale > 0, probability_scale, 1.0
                    )
                    probability_fp8 = (weighted / probability_scale[:, None]).to(
                        tl.float8e4nv
                    )
                    contribution0 = tl.dot(
                        probability_fp8, tl.trans(key0), out_dtype=tl.float32
                    )
                    contribution1 = tl.dot(
                        probability_fp8, tl.trans(key1), out_dtype=tl.float32
                    )
                    value0 = (
                        value0 * correction[:, None]
                        + contribution0 * probability_scale[:, None]
                    )
                    value1 = (
                        value1 * correction[:, None]
                        + contribution1 * probability_scale[:, None]
                    )
                    maximum = next_maximum
                if SPLITS == 1:
                    logsum = tl.where(
                        denominator > 0, maximum + tl.log(denominator), float("inf")
                    )
                    denominator_inverse = tl.where(
                        denominator > 0, 1.0 / denominator, 0.0
                    )
                    if HAS_SINK:
                        sink = tl.load(Sink + heads, heads < H, 0)
                        inverse = sink_normalization(denominator_inverse, sink, logsum)
                    else:
                        inverse = denominator_inverse
                    output_ptr = (
                        Output + (batch * H + heads[:, None]) * 512 + dims[None, :]
                    )
                    tl.store(output_ptr, value0 * inverse[:, None], heads[:, None] < H)
                    tl.store(
                        output_ptr + 256, value1 * inverse[:, None], heads[:, None] < H
                    )
                    tl.store(LSE + batch * H + heads, logsum, heads < H)
                else:
                    partial_ptr = (
                        Partial
                        + ((batch * SPLITS + split) * H + heads[:, None]) * 512
                        + dims[None, :]
                    )
                    tl.store(partial_ptr, value0, heads[:, None] < H)
                    tl.store(partial_ptr + 256, value1, heads[:, None] < H)
                    stats_ptr = Stats + ((batch * SPLITS + split) * H + heads) * 2
                    tl.store(stats_ptr, maximum, heads < H)
                    tl.store(stats_ptr + 1, denominator, heads < H)
            else:
                pass

    @triton.jit
    def sparse_fp8_fast_merge(
        Q,
        QR,
        KV,
        KR,
        QS,
        KS,
        Indices,
        Length,
        Sink,
        Flags,
        Partial,
        Stats,
        Output,
        LSE,
        qb: tl.constexpr,
        qh: tl.constexpr,
        qrb: tl.constexpr,
        qrh: tl.constexpr,
        kp: tl.constexpr,
        kt: tl.constexpr,
        krp: tl.constexpr,
        krt: tl.constexpr,
        qsb: tl.constexpr,
        qsh: tl.constexpr,
        ksp: tl.constexpr,
        kst: tl.constexpr,
        ib: tl.constexpr,
        ik: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        SPLITS: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        HAS_SINK: tl.constexpr,
        CHECK_REPAIR: tl.constexpr,
        HEAD_TILE: tl.constexpr = 4,
        HAS_ROPE: tl.constexpr = True,
    ):
        batch = tl.program_id(0)
        head_base = tl.program_id(1) * HEAD_TILE
        dims = tl.arange(0, 512)
        if CHECK_REPAIR:
            head_group = head_base // 64
            flag_base = Flags + (batch * (H // 64) + head_group) * SPLITS
            flag_acc = tl.full((), 0, tl.int32)
            for s in tl.static_range(SPLITS):
                flag_acc = flag_acc | tl.load(flag_base + s)
            need_repair = flag_acc != 0
        else:
            need_repair = False
        if need_repair:
            ropes = tl.arange(0, 64)
            length = (
                tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
                if HAS_LENGTH
                else TOPK
            )
            for h_off in range(HEAD_TILE):
                head = head_base + h_off
                row = batch * H + head
                query = tl.load(Q + batch * qb + head * qh + dims).to(tl.float32)
                if HAS_ROPE:
                    query_rope = tl.load(QR + batch * qrb + head * qrh + ropes).to(
                        tl.float32
                    )
                else:
                    pass
                query_scale = tl.load(QS + batch * qsb + head * qsh)
                maximum = -float("inf")
                denominator = 0.0
                values = tl.zeros((512,), tl.float32)
                for position in range(length):
                    token = tl.load(Indices + batch * ib + position * ik)
                    if (token >= 0) & (token < N):
                        physical_token = token.to(tl.uint64)
                        page, slot = physical_token // 64, physical_token % 64
                        key = tl.load(KV + page * kp + slot * kt + dims).to(tl.float32)
                        if HAS_ROPE:
                            key_rope = tl.load(KR + page * krp + slot * krt + ropes).to(
                                tl.float32
                            )
                        else:
                            pass
                        key_scale = tl.load(KS + page * ksp + slot * kst)
                        if HAS_ROPE:
                            rope_score = tl.sum(query_rope * key_rope, 0)
                        else:
                            rope_score = 0.0
                        score = (
                            (tl.sum(query * key, 0) + rope_score)
                            * query_scale
                            * key_scale
                            * (SCALE * 1.4426950408889634)
                        )
                        new_maximum = tl.maximum(maximum, score)
                        old_delta = tl.inline_asm_elementwise(
                            "sub.rn.f32 $0, $1, $2;",
                            "=f,f,f",
                            [maximum, new_maximum],
                            dtype=tl.float32,
                            is_pure=False,
                            pack=1,
                        )
                        score_delta = tl.inline_asm_elementwise(
                            "sub.rn.f32 $0, $1, $2;",
                            "=f,f,f",
                            [score, new_maximum],
                            dtype=tl.float32,
                            is_pure=False,
                            pack=1,
                        )
                        correction = tl.exp2(old_delta)
                        probability = tl.exp2(score_delta)
                        denominator = denominator * correction + probability
                        values = values * correction + probability * (key * key_scale)
                        maximum = new_maximum
                    else:
                        pass
                logsum = tl.where(
                    denominator > 0,
                    (maximum + tl.log2(denominator)) * 0.6931471805599453,
                    float("inf"),
                )
                inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
                if HAS_SINK:
                    sink = tl.load(Sink + head)
                    inverse *= tl.where(
                        sink == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - logsum))
                    )
                else:
                    pass
                tl.store(Output + row * 512 + dims, (values * inverse).to(tl.bfloat16))
                tl.store(LSE + row, logsum)
        else:
            if SPLITS == 2 and HEAD_TILE % 2 == 0:
                base_s0 = (batch * 2) * H
                base_s1 = (batch * 2 + 1) * H
                for pair in tl.static_range(HEAD_TILE // 2):
                    h0 = head_base + pair * 2
                    h1 = h0 + 1
                    st0_a = Stats + (base_s0 + h0) * 2
                    st1_a = Stats + (base_s1 + h0) * 2
                    st0_b = Stats + (base_s0 + h1) * 2
                    st1_b = Stats + (base_s1 + h1) * 2
                    m0_a = tl.load(st0_a)
                    d0_a = tl.load(st0_a + 1)
                    m1_a = tl.load(st1_a)
                    d1_a = tl.load(st1_a + 1)
                    m0_b = tl.load(st0_b)
                    d0_b = tl.load(st0_b + 1)
                    m1_b = tl.load(st1_b)
                    d1_b = tl.load(st1_b + 1)

                    p0_a = tl.load(
                        Partial + (base_s0 + h0) * 512 + dims,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    p1_a = tl.load(
                        Partial + (base_s1 + h0) * 512 + dims,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    p0_b = tl.load(
                        Partial + (base_s0 + h1) * 512 + dims,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    p1_b = tl.load(
                        Partial + (base_s1 + h1) * 512 + dims,
                        eviction_policy="evict_first",
                    ).to(tl.float32)

                    gm_a = tl.maximum(m0_a, m1_a)
                    sm_a = tl.where(gm_a == -float("inf"), 0.0, gm_a)
                    w0_a = tl.exp(m0_a - sm_a)
                    w1_a = tl.exp(m1_a - sm_a)
                    tot_a = d0_a * w0_a + d1_a * w1_a
                    lse_a = tl.where(tot_a > 0.0, gm_a + tl.log(tot_a), float("inf"))
                    denominator_inverse = tl.where(tot_a > 0.0, 1.0 / tot_a, 0.0)
                    if HAS_SINK:
                        sk_a = tl.load(Sink + h0)
                        inv_a = sink_normalization(denominator_inverse, sk_a, lse_a)
                    else:
                        inv_a = denominator_inverse
                    s0_a = w0_a * inv_a
                    s1_a = w1_a * inv_a

                    gm_b = tl.maximum(m0_b, m1_b)
                    sm_b = tl.where(gm_b == -float("inf"), 0.0, gm_b)
                    w0_b = tl.exp(m0_b - sm_b)
                    w1_b = tl.exp(m1_b - sm_b)
                    tot_b = d0_b * w0_b + d1_b * w1_b
                    lse_b = tl.where(tot_b > 0.0, gm_b + tl.log(tot_b), float("inf"))
                    denominator_inverse = tl.where(tot_b > 0.0, 1.0 / tot_b, 0.0)
                    if HAS_SINK:
                        sk_b = tl.load(Sink + h1)
                        inv_b = sink_normalization(denominator_inverse, sk_b, lse_b)
                    else:
                        inv_b = denominator_inverse
                    s0_b = w0_b * inv_b
                    s1_b = w1_b * inv_b

                    out_a = (p0_a * s0_a + p1_a * s1_a).to(tl.bfloat16)
                    out_b = (p0_b * s0_b + p1_b * s1_b).to(tl.bfloat16)
                    row_a = batch * H + h0
                    row_b = batch * H + h1
                    tl.store(
                        Output + row_a * 512 + dims,
                        out_a,
                        eviction_policy="evict_first",
                    )
                    tl.store(
                        Output + row_b * 512 + dims,
                        out_b,
                        eviction_policy="evict_first",
                    )
                    tl.store(LSE + row_a, lse_a)
                    tl.store(LSE + row_b, lse_b)
            else:
                for h_off in tl.static_range(HEAD_TILE):
                    head = head_base + h_off
                    row = batch * H + head
                    gmax = tl.full((), -float("inf"), tl.float32)
                    for s in tl.static_range(SPLITS):
                        m_s = tl.load(Stats + ((batch * SPLITS + s) * H + head) * 2)
                        gmax = tl.maximum(gmax, m_s)
                    safe_max = tl.where(gmax == -float("inf"), 0.0, gmax)
                    total = tl.full((), 0.0, tl.float32)
                    for s in tl.static_range(SPLITS):
                        st_ptr = Stats + ((batch * SPLITS + s) * H + head) * 2
                        m_s = tl.load(st_ptr)
                        d_s = tl.load(st_ptr + 1)
                        total += d_s * tl.exp(m_s - safe_max)
                    logsum = tl.where(total > 0.0, gmax + tl.log(total), float("inf"))
                    denominator_inverse = tl.where(total > 0.0, 1.0 / total, 0.0)
                    if HAS_SINK:
                        sink = tl.load(Sink + head)
                        inv = sink_normalization(denominator_inverse, sink, logsum)
                    else:
                        inv = denominator_inverse
                    acc = tl.zeros((512,), tl.float32)
                    for s in tl.static_range(SPLITS):
                        m_s = tl.load(Stats + ((batch * SPLITS + s) * H + head) * 2)
                        w_s = tl.exp(m_s - safe_max) * inv
                        p_s = tl.load(
                            Partial + ((batch * SPLITS + s) * H + head) * 512 + dims,
                            eviction_policy="evict_first",
                        ).to(tl.float32)
                        acc += p_s * w_s
                    tl.store(
                        Output + row * 512 + dims,
                        acc.to(tl.bfloat16),
                        eviction_policy="evict_first",
                    )
                    tl.store(LSE + row, logsum)

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("sparse_fp8_short"),
        key=["B", "H", "N", "TOPK", "HAS_LENGTH", "HAS_SINK"],
        use_cuda_graph=True,
    )
    @triton.jit
    def sparse_fp8_short(
        Q,
        KV,
        QS,
        KS,
        Indices,
        Length,
        Sink,
        Output,
        LSE,
        qb: tl.constexpr,
        qh: tl.constexpr,
        kp: tl.constexpr,
        kt: tl.constexpr,
        qsb: tl.constexpr,
        qsh: tl.constexpr,
        ksp: tl.constexpr,
        kst: tl.constexpr,
        ib: tl.constexpr,
        ik: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        TOPK: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_LENGTH: tl.constexpr,
        HAS_SINK: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        VALUE_DIM: tl.constexpr,
    ):
        batch = tl.program_id(0)
        heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        columns = tl.program_id(2) * VALUE_DIM + tl.arange(0, VALUE_DIM)
        features = tl.arange(0, 512)
        positions = tl.arange(0, BLOCK_K)
        length = (
            tl.minimum(tl.maximum(tl.load(Length + batch), 0), TOPK)
            if HAS_LENGTH
            else TOPK
        )
        ids = tl.load(Indices + batch * ib + positions * ik, positions < length, -1)
        valid = (positions < length) & (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0).to(tl.int64)
        pages, tokens = ids // 64, ids % 64
        query = tl.load(Q + batch * qb + heads[:, None] * qh + features[None, :])
        cache = tl.load(
            KV + pages[:, None] * kp + tokens[:, None] * kt + features[None, :],
            valid[:, None],
            0.0,
        )
        query_scale = tl.load(QS + batch * qsb + heads * qsh)
        cache_scale = tl.load(KS + pages * ksp + tokens * kst, valid, 0.0)
        # Merge FP8 QK partial accumulators in FP32 at each MMA K step.
        scores = tl.dot(
            query, tl.trans(cache), out_dtype=tl.float32, max_num_imprecise_acc=0
        )
        scores *= query_scale[:, None] * cache_scale[None, :] * SCALE
        scores = tl.where(valid[None, :], scores, -float("inf"))
        maximum = tl.max(scores, 1)
        safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        probabilities = tl.exp(scores - safe_maximum[:, None])
        denominator = tl.sum(probabilities, 1)
        weighted = probabilities * cache_scale[None, :]
        probability_scale = tl.max(weighted, 1) / 448.0
        probability_scale = tl.where(probability_scale > 0, probability_scale, 1.0)
        probability_fp8 = (weighted / probability_scale[:, None]).to(tl.float8e4nv)
        values = tl.load(
            KV + pages[:, None] * kp + tokens[:, None] * kt + columns[None, :],
            valid[:, None],
            0.0,
        )
        value = tl.dot(probability_fp8, values, out_dtype=tl.float32)
        logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
        denominator_inverse = tl.where(denominator > 0, 1.0 / denominator, 0.0)
        if HAS_SINK:
            sink = tl.load(Sink + heads)
            inverse = sink_normalization(denominator_inverse, sink, logsum)
        else:
            inverse = denominator_inverse
        value *= (probability_scale * inverse)[:, None]
        tl.store(Output + (batch * H + heads[:, None]) * 512 + columns[None, :], value)
        tl.store(LSE + batch * H + heads, logsum, tl.program_id(2) == 0)

    @triton.jit
    def split_block_range(split, NUM_BLOCKS, SPLITS: tl.constexpr):
        if SPLITS == 1:
            return 0, NUM_BLOCKS
        base = NUM_BLOCKS // SPLITS
        rem = NUM_BLOCKS - base * SPLITS
        first = split * base + tl.minimum(split, rem)
        count = base + tl.where(split < rem, 1, 0)
        return first, count

    @triton.jit
    def scaled_bytes(quantized, ratios, R: tl.constexpr):
        factors = tl.broadcast_to(ratios[:, :, None], (R, 8, 64)).reshape(R, 512)
        return (quantized.to(tl.float32) * factors).to(tl.float8e4nv)

    @triton.jit
    def normalize_query(Q, QS, Norm, Magnitude, H: tl.constexpr):
        b, tile = tl.program_id(0), tl.program_id(1)
        rows = b * H + tile * 4 + tl.arange(0, 4)
        g, d = tl.arange(0, 8), tl.arange(0, 512)
        scales = tl.load(QS + rows[:, None] * 7 + g[None, :], g[None, :] < 7, 0.0)
        mx = tl.max(scales, 1)
        mn = tl.min(tl.where(scales > 0.0, scales, float("inf")), 1)
        q = tl.load(Q + rows[:, None] * 448 + d[None, :], d[None, :] < 448, 0.0)
        qnorm = scaled_bytes(q, scales / tl.where(mx > 0.0, mx, 1.0)[:, None], 4)
        tl.store(Norm + rows[:, None] * 448 + d[None, :], qnorm, d[None, :] < 448)
        tl.store(Magnitude + rows, tl.where(mx > 4.0 * mn, -mx, mx))

    @triton.jit
    def normalize_cache(
        K,
        E,
        Normalized,
        Scale,
        PackedRope,
        N,
        STRIDE: tl.constexpr,
    ):
        # Each block processes consecutive physical slots, including a partial final page.
        ids = tl.program_id(0) * 32 + tl.arange(0, 32)
        page = (ids >> 6).to(tl.int64) * STRIDE
        slot = ids & 63
        g = tl.arange(0, 8)
        word = tl.load(
            E.to(tl.pointer_type(tl.uint64)) + page // 8 + 4608 + slot, ids < N, 0
        )
        exps = ((word[:, None] >> (g * 8)[None, :]) & 255).to(tl.int32)
        hi = tl.max(tl.where(g[None, :] < 7, exps, 0), 1)
        lo = tl.min(tl.where(g[None, :] < 7, exps, 255), 1)
        d = tl.arange(0, 512)
        content = tl.load(
            K + (page + slot * 576)[:, None] + d[None, :],
            (ids[:, None] < N) & (d[None, :] < 448),
            0.0,
        )
        ratio = (tl.maximum(127 + tl.minimum(exps - hi[:, None], 0), 0) << 23).to(
            tl.float32, bitcast=True
        )
        norm = scaled_bytes(content, ratio, 32)
        tl.store(
            Normalized + ids.to(tl.int64)[:, None] * 448 + d[None, :],
            norm,
            (ids[:, None] < N) & (d[None, :] < 448),
        )
        r = tl.arange(0, 64)
        src = (page + slot * 576 + 448) // 2
        rope = tl.load(
            E.to(tl.pointer_type(tl.bfloat16)) + src[:, None] + r[None, :],
            ids[:, None] < N,
            0.0,
        )
        tl.store(PackedRope + ids[:, None] * 64 + r[None, :], rope, ids[:, None] < N)
        scale = tl.where(hi == 0, 0x00400000, hi << 23).to(tl.float32, bitcast=True)
        tl.store(Scale + ids, tl.where(hi - lo > 2, -scale, scale), ids < N)

    @triton.constexpr_function
    def gather_asm(block, page_stride, panel_rows, warps):
        # A subgroup of two or four lanes owns one selected row. Every lane keeps
        # that row's address and exponent word through both copying and rescaling.
        rows = block // warps
        lanes = 32 // rows
        asm = f"""
    {{
    .reg .pred ok, first, diff, any_diff, more, lower, wide, skip;
    .reg .b32 tid,lane,warp,row,chunk,pos,id,page,slot,e0,e1,hi,lo,ex,sz,flag;
    .reg .b32 off,sw,kbase,rbase,kdst,rdst,sdst,tmp,u,group,word,shift,delta,fac;
    .reg .b32 a,b,c,d;
    .reg .b16 xlo,xhi;
    .reg .b64 address,ptr,src,pg;
    mov.u32 tid,%tid.x;
    and.b32 tid,tid,{warps*32-1};
    shr.u32 warp,tid,5;
    and.b32 lane,tid,31;
    shr.u32 row,lane,{lanes.bit_length()-1};
    mad.lo.u32 row,warp,{rows},row;
    and.b32 chunk,lane,{lanes-1};
    add.u32 pos,$8,row;
    setp.lt.u32 ok,pos,$9;
    mov.u32 id,-1;
    mul.wide.u32 address,pos,4;
    add.u64 ptr,$3,address;
    @ok ld.global.L1::evict_first.s32 id,[ptr];
    setp.lt.u32 ok,id,$7;
    @!ok mov.u32 id,0;
    shr.u32 page,id,6;
    and.b32 slot,id,63;
    mul.wide.u32 address,page,{page_stride};
    add.u64 pg,$2,address;
    mul.wide.u32 address,slot,8;
    add.u64 ptr,pg,address;
    ld.global.v2.b32 {{e0,e1}},[ptr+36864];
    mul.wide.u32 address,slot,576;
    add.u64 src,pg,address;
    shl.b32 off,chunk,4;
    cvt.u64.u32 address,off;
    add.u64 src,src,address;
    selp.u32 sz,16,0,ok;
    and.b32 sw,row,7;
    shl.b32 sw,sw,4;
    xor.b32 off,off,sw;
    mad.lo.u32 tmp,row,128,$4;
    add.u32 kbase,tmp,off;
    mad.lo.u32 tmp,row,128,$5;
    add.u32 rbase,tmp,off;
    """
        for g in range(9):
            for seg in range(4 // lanes):
                column_offset = (g % 2) * 64 + seg * lanes * 16
                adr = g * 64 + seg * lanes * 16
                if g < 7:
                    asm += f"xor.b32 kdst,kbase,{column_offset};\n"
                    asm += f"cp.async.cg.shared.global [kdst+{(g//2)*panel_rows*128}],[src+{adr}],16,sz;\n"
                else:
                    column_offset = (g - 7) * 64 + seg * lanes * 16
                    asm += f"xor.b32 rdst,rbase,{column_offset};\ncp.async.cg.shared.global [rdst],[src+{adr}],16,sz;\n"
        asm += """
    cp.async.commit_group;
    and.b32 hi,e0,255;
    mov.b32 lo,hi;
    """
        for g in range(1, 7):
            asm += f"bfe.u32 ex,{'e0' if g<4 else 'e1'},{(g%4)*8},8; min.u32 lo,lo,ex; max.u32 hi,hi,ex;\n"
        asm += """
    shl.b32 tmp,hi,23;
    setp.eq.u32 first,hi,0;
    @first mov.b32 tmp,0x00400000;
    @!ok mov.b32 tmp,0xbf800000;
    setp.eq.u32 first,chunk,0;
    mad.lo.u32 sdst,row,4,$6;
    @first st.shared.b32 [sdst],tmp;
    sub.u32 delta,hi,lo;
    setp.gt.and.u32 wide,delta,2,ok;
    selp.u32 flag,1,0,wide;
    setp.ne.and.u32 diff,delta,0,ok;
    vote.sync.any.pred any_diff,diff,0xffffffff;
    cp.async.wait_group 0;
    @!any_diff bra DONE;
    mov.u32 group,0;
    RESCALE:
    setp.lt.u32 lower,group,4;
    selp.u32 word,e0,e1,lower;
    and.b32 shift,group,3;
    shl.b32 shift,shift,3;
    bfe.u32 ex,word,shift,8;
    setp.eq.u32 skip,hi,ex;
    @skip bra NEXT;
    sub.u32 delta,hi,ex;
    min.u32 delta,delta,15;
    sub.u32 fac,15,delta;
    shl.b32 fac,fac,10;
    prmt.b32 fac,fac,fac,0x1010;
    and.b32 tmp,group,1;
    shl.b32 tmp,tmp,6;
    xor.b32 kdst,kbase,tmp;
    shr.u32 tmp,group,1;
    """
        asm += f"mad.lo.u32 kdst,tmp,{panel_rows*128},kdst;\n"
        for seg in range(4 // lanes):
            asm += f"xor.b32 rdst,kdst,{seg*lanes*16};\nld.shared.v4.b32 {{a,b,c,d}},[rdst];\n"
            for reg in ("a", "b", "c", "d"):
                asm += f"""
            mov.b32 {{xlo,xhi}},{reg};
            cvt.rn.f16x2.e4m3x2 tmp,xlo;
            cvt.rn.f16x2.e4m3x2 u,xhi;
            mul.f16x2 tmp,tmp,fac;
            mul.f16x2 u,u,fac;
            cvt.rn.satfinite.e4m3x2.f16x2 xlo,tmp;
            cvt.rn.satfinite.e4m3x2.f16x2 xhi,u;
            mov.b32 {reg},{{xlo,xhi}};
            """
            asm += "st.shared.v4.b32 [rdst],{a,b,c,d};\n"
        return (
            asm
            + """
    NEXT:
    add.u32 group,group,1;
    setp.lt.u32 more,group,7;
    @more bra RESCALE;
    DONE:
    mov.b32 $0,flag;
    }
    """
        )

    @triton.constexpr_function
    def warp_gather_asm(block, page_stride, panel_rows, warps):
        # Each warp resolves its selected rows before issuing 16-byte copies.
        # The 8 (or 16) metadata lanes broadcast one physical slot at a time.
        asm = """
    {
    .reg .pred ok,active,first,rope,loop,real,fromlo,wide,rescale,row_diff,any_diff,fast;
    .reg .b32 tid,lane,warp,row,pos,id,page,slot,e0,e1,ex,emin,emax,flag,sz;
    .reg .b32 src0,src1,peer0,peer1,peer_sz,peer_hi,peer_e0,peer_e1,which;
    .reg .b32 panel,off,sw,kdst,rdst,sdst,fac,a,b,c,d,t,u,delta,iter;
    .reg .b16 xlo,xhi;
    .reg .b64 address,pgaddr,src,peer,ptr;
    mov.u32 tid,%tid.x;
    and.b32 tid,tid,TIDMASK;
    shr.u32 warp,tid,5;
    and.b32 lane,tid,31;
    and.b32 row,lane,ROWMASK;
    mad.lo.u32 row,row,WARPS,warp;
    add.u32 pos,$8,row;
    setp.lt.u32 ok,pos,$9;
    mul.wide.u32 address,pos,4;
    add.u64 ptr,$3,address;
    mov.b32 id,-1;
    @ok ld.global.L1::evict_first.s32 id,[ptr];
    setp.lt.u32 ok,id,$7;
    @!ok mov.b32 id,0;
    shr.u32 page,id,6;
    and.b32 slot,id,63;
    mul.wide.u32 address,page,PAGESTRIDE;
    add.u64 pgaddr,$2,address;
    mul.wide.u32 address,slot,8;
    add.u64 ptr,pgaddr,address;
    ld.global.v2.b32 {e0,e1},[ptr+36864];
    mul.wide.u32 address,slot,576;
    add.u64 src,pgaddr,address;
    mov.b64 {src0,src1},src;
    selp.u32 sz,16,0,ok;
    shr.u32 panel,lane,3;
    mul.lo.u32 panel,panel,PANEL;
    setp.lt.u32 real,lane,28;
    setp.lt.u32 rope,lane,8;
    mov.u32 iter,0;
    COPY:
    shfl.sync.idx.b32 peer0,src0,iter,31,0xffffffff;
    shfl.sync.idx.b32 peer1,src1,iter,31,0xffffffff;
    shfl.sync.idx.b32 peer_sz,sz,iter,31,0xffffffff;
    mov.b64 peer,{peer0,peer1};
    mad.lo.u32 row,iter,WARPS,warp;
    and.b32 sw,row,7;
    shl.b32 sw,sw,4;
    and.b32 off,lane,7;
    shl.b32 off,off,4;
    cvt.u64.u32 address,off;
    add.u64 address,peer,address;
    add.u64 address,address,448;
    mad.lo.u32 off,row,128,off;
    xor.b32 off,off,sw;
    add.u32 rdst,$5,off;
    @rope cp.async.cg.shared.global [rdst],[address],16,peer_sz;
    add.u32 kdst,$4,off;
    add.u32 kdst,kdst,panel;
    shl.b32 t,lane,4;
    cvt.u64.u32 address,t;
    add.u64 address,peer,address;
    @real cp.async.cg.shared.global [kdst],[address],16,peer_sz;
    add.u32 iter,iter,1;
    setp.le.u32 loop,iter,ROWMASK;
    @loop bra COPY;
    cp.async.commit_group;
    and.b32 emin,e0,255;
    mov.b32 emax,emin;
    """
        for group in range(1, 7):
            asm += f"bfe.u32 ex,{'e0' if group < 4 else 'e1'},{(group % 4) * 8},8; "
            asm += "min.u32 emin,emin,ex; max.u32 emax,emax,ex; "
        asm += """
    sub.u32 delta,emax,emin;
    setp.gt.u32 wide,delta,2;
    and.pred wide,wide,ok;
    selp.u32 flag,1,0,wide;
    // One metadata lane owns the SMEM scale for this row.
    shl.b32 t,emax,23;
    setp.eq.u32 first,emax,0;
    @first mov.b32 t,0x00400000;
    @!ok mov.b32 t,0xbf800000;
    and.b32 row,lane,ROWMASK;
    mad.lo.u32 row,row,WARPS,warp;
    mad.lo.u32 sdst,row,4,$6;
    setp.le.u32 first,lane,ROWMASK;
    @first st.shared.b32 [sdst],t;
    cp.async.wait_group 0;
    mov.u32 iter,0;
    setp.ne.u32 row_diff,emax,emin;
    and.pred row_diff,row_diff,first;
    and.pred row_diff,row_diff,ok;
    vote.sync.any.pred any_diff,row_diff,0xffffffff;
    @!any_diff bra DONE_RESCALE;
    RESCALE:
    shfl.sync.idx.b32 peer_e0,e0,iter,31,0xffffffff;
    shfl.sync.idx.b32 peer_e1,e1,iter,31,0xffffffff;
    shfl.sync.idx.b32 peer_hi,emax,iter,31,0xffffffff;
    setp.lt.u32 fromlo,lane,16;
    selp.u32 ex,peer_e0,peer_e1,fromlo;
    shr.u32 which,lane,2;
    and.b32 which,which,3;
    shl.b32 which,which,3;
    bfe.u32 ex,ex,which,8;
    setp.ne.u32 rescale,peer_hi,ex;
    and.pred rescale,rescale,real;
    @!rescale bra NEXT;
    mad.lo.u32 row,iter,WARPS,warp;
    and.b32 sw,row,7;
    shl.b32 sw,sw,4;
    and.b32 off,lane,7;
    shl.b32 off,off,4;
    mad.lo.u32 off,row,128,off;
    xor.b32 off,off,sw;
    add.u32 kdst,$4,off;
    add.u32 kdst,kdst,panel;
    ld.shared.v4.b32 {a,b,c,d},[kdst];
    sub.s32 delta,peer_hi,ex;
    // Subtracting from each exponent is exact while every merged_value stays normal.
    // The half2 conversion below handles zeros, subnormals and wide spreads.
    min.u32 fac,delta,14;
    shl.b32 fac,fac,3;
    mul.lo.u32 fac,fac,0x01010101;
    add.u32 which,fac,0x08080808;
    or.b32 t,a,0x80808080;
    sub.u32 t,t,which;
    or.b32 u,b,0x80808080;
    sub.u32 u,u,which;
    and.b32 t,t,u;
    or.b32 u,c,0x80808080;
    sub.u32 u,u,which;
    and.b32 t,t,u;
    or.b32 u,d,0x80808080;
    sub.u32 u,u,which;
    and.b32 t,t,u;
    and.b32 t,t,0x80808080;
    and.b32 u,a,0x7f7f7f7f;
    add.u32 u,u,0x01010101;
    and.b32 which,b,0x7f7f7f7f;
    add.u32 which,which,0x01010101;
    or.b32 u,u,which;
    and.b32 which,c,0x7f7f7f7f;
    add.u32 which,which,0x01010101;
    or.b32 u,u,which;
    and.b32 which,d,0x7f7f7f7f;
    add.u32 which,which,0x01010101;
    or.b32 u,u,which;
    not.b32 u,u;
    and.b32 t,t,u;
    and.b32 t,t,0x80808080;
    setp.eq.u32 fast,t,0x80808080;
    setp.lt.u32.and fast,delta,15,fast;
    @!fast bra SLOW_RESCALE;
    sub.u32 a,a,fac;
    sub.u32 b,b,fac;
    sub.u32 c,c,fac;
    sub.u32 d,d,fac;
    bra WRITE_RESCALE;
    SLOW_RESCALE:
    min.u32 delta,delta,15;
    sub.s32 fac,15,delta;
    shl.b32 fac,fac,10;
    prmt.b32 fac,fac,fac,0x1010;
    """
        for reg in ("a", "b", "c", "d"):
            asm += """
        mov.b32 {xlo,xhi},WORD;
        cvt.rn.f16x2.e4m3x2 t,xlo;
        cvt.rn.f16x2.e4m3x2 u,xhi;
        mul.f16x2 t,t,fac;
        mul.f16x2 u,u,fac;
        cvt.rn.satfinite.e4m3x2.f16x2 xlo,t;
        cvt.rn.satfinite.e4m3x2.f16x2 xhi,u;
        mov.b32 WORD,{xlo,xhi};
        """.replace(
                "WORD", reg
            )
        asm += """
    WRITE_RESCALE:
    st.shared.v4.b32 [kdst],{a,b,c,d};
    NEXT:
    add.u32 iter,iter,1;
    setp.le.u32 loop,iter,ROWMASK;
    @loop bra RESCALE;
    DONE_RESCALE:
    mov.b32 $0,flag;
    }
    """
        return (
            asm.replace("PAGESTRIDE", str(page_stride))
            .replace("PANEL", str(panel_rows * 128))
            .replace("TIDMASK", str(warps * 32 - 1))
            .replace("WARPS", str(warps))
            .replace("ROWMASK", str(block // warps - 1))
        )

    @triton.jit
    def gather_tile(
        K,
        Idx,
        sk,
        skr,
        ks,
        b,
        first_pos,
        N,
        T: tl.constexpr,
        BK: tl.constexpr,
        STRIDE: tl.constexpr,
    ):
        return tl.inline_asm_elementwise(
            (
                warp_gather_asm(BK, STRIDE, BK, 8)
                if BK == 128
                else gather_asm(BK, STRIDE, BK, 8)
            ),
            "=r,r,l,l,r,r,r,r,r,r",
            [
                tl.arange(0, 256),
                K.to(tl.uint64),
                (Idx + b.to(tl.int64) * T).to(tl.uint64),
                tle.gpu.local_ptr(sk, (0, 0)).to(tl.uint64).to(tl.uint32),
                tle.gpu.local_ptr(skr, (0, 0)).to(tl.uint64).to(tl.uint32),
                tle.gpu.local_ptr(ks, (0,)).to(tl.uint64).to(tl.uint32),
                N,
                first_pos,
                tl.full((), T, tl.uint32),
            ],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def producer(
        K,
        KR,
        Idx,
        Flags,
        RepairState,
        Normalized,
        Scale,
        sk,
        skr,
        ks,
        kfull,
        kfree,
        N,
        T: tl.constexpr,
        STRIDE: tl.constexpr,
        S: tl.constexpr,
        BK: tl.constexpr,
        BUFS: tl.constexpr,
        PREPACK: tl.constexpr,
    ):
        b = tl.program_id(0)
        first, count = split_block_range(tl.program_id(2), tl.cdiv(T, BK), S)
        repair = tl.full((), 0, tl.int32)
        for step in range(count):
            buf, cycle = step % BUFS, step // BUFS
            if BUFS == 1:
                if step > 0:
                    tle.gpu.barrier_wait(kfree[0])
                else:
                    pass
            else:
                tle.gpu.barrier_wait(kfree[buf], phaseIdx=cycle)
            if PREPACK:
                k = tl.arange(0, BK)
                g = tl.arange(0, 8)
                pos = (first + step) * BK + k
                ids = tl.load(
                    Idx + b.to(tl.int64) * T + pos,
                    pos < T,
                    -1,
                    eviction_policy="evict_first",
                )
                valid = (ids >= 0) & (ids < N)
                safe = tl.where(valid, ids, 0)
                signed_scale = tl.load(Scale + safe)
                repair |= tl.max((valid & (signed_scale < 0.0)).to(tl.int32), 0)
                tl.store(
                    tle.gpu.local_ptr(ks.slot(buf)),
                    tl.where(valid, tl.abs(signed_scale), -1.0),
                )
                rows = tl.broadcast_to(k[:, None], (BK, 8))
                cols = tl.broadcast_to((g * 8)[None, :], (BK, 8))
                dst = tle.gpu.local_ptr(skr.slot(buf), (rows, cols))
                src = KR + safe.to(tl.int64)[:, None] * 64 + cols
                tl.inline_asm_elementwise(
                    "{ cp.async.cg.shared.global [$1], [$2], 16, $3; mov.u32 $0, 0; }",
                    "=r,r,l,r",
                    [
                        dst.to(tl.uint64).to(tl.uint32),
                        src.to(tl.uint64),
                        tl.where(valid[:, None], 16, 0),
                    ],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
                byte_cols = cols * 2
                for tile in tl.static_range(4):
                    dst_k = tle.gpu.local_ptr(
                        tle_subslice(sk.slot(buf), [0, tile * 128], [BK, 128]),
                        (rows, byte_cols),
                    )
                    src_k = (
                        Normalized
                        + safe.to(tl.int64)[:, None] * 448
                        + tile * 128
                        + byte_cols
                    )
                    tl.inline_asm_elementwise(
                        "{ .reg .pred p; setp.ne.b32 p,$4,0; "
                        "@p cp.async.cg.shared.global [$1], [$2], 16, $3; mov.u32 $0, 0; }",
                        "=r,r,l,r,r",
                        [
                            dst_k.to(tl.uint64).to(tl.uint32),
                            src_k.to(tl.uint64),
                            tl.where(valid[:, None], 16, 0),
                            ((tile < 3) | (byte_cols < 64)).to(tl.int32),
                        ],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
            else:
                flag = gather_tile(
                    K,
                    Idx,
                    sk.slot(buf),
                    skr.slot(step % 4 if BUFS == 2 else 0),
                    ks.slot(buf),
                    b,
                    (first + step) * BK,
                    N,
                    T,
                    BK,
                    STRIDE,
                )
                repair |= tl.max(flag, 0).to(tl.int32)
            tl.inline_asm_elementwise(
                (
                    "{ cp.async.commit_group; cp.async.wait_group 0; fence.proxy.async.shared::cta; mov.u32 $0, $1; }"
                    if PREPACK
                    else "{ fence.proxy.async.shared::cta; mov.u32 $0, $1; }"
                ),
                "=r,r",
                [tl.arange(0, 256)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            tl.debug_barrier()
            if BUFS == 1:
                tle.gpu.barrier_arrive(kfull[0])
            else:
                tle.gpu.barrier_arrive(kfull[buf], phaseIdx=cycle)
        tl.atomic_or(Flags + b, 1, mask=repair != 0, sem="relaxed")
        tl.atomic_or(RepairState, 1, mask=repair != 0, sem="relaxed")

    @gluon.jit
    def publish(
        Output,
        row0,
        acc,
        scale,
        OFFSET: gl.constexpr,
        WIDTH: gl.constexpr,
        STREAM: gl.constexpr,
    ):
        layout: gl.constexpr = acc.type.layout
        row = row0 + gl.arange(0, 64, layout=gl.constexpr(SliceLayout(1, layout)))
        col = OFFSET + gl.arange(0, WIDTH, layout=gl.constexpr(SliceLayout(0, layout)))
        factor = gl.convert_layout(scale, gl.constexpr(SliceLayout(1, layout)))
        gl.store(
            Output + row[:, None] * 512 + col[None, :],
            (acc * factor[:, None]).to(Output.dtype.element_ty),
            cache_modifier=".cs" if STREAM else "",
        )

    @gluon.jit
    def bf_band(
        Q,
        QS,
        K,
        q,
        kv,
        h,
        offset,
        valid,
        scales,
        layout: gl.constexpr,
        policy: gl.constexpr,
        start: gl.constexpr,
        width: gl.constexpr,
    ):
        col = start + gl.arange(0, width, layout=gl.constexpr(SliceLayout(0, layout)))
        qv = gl.load(
            Q + (h * 448)[:, None] + col[None, :], eviction_policy="evict_first"
        )
        qs = gl.load(
            QS + (h * 7)[:, None] + (col // 64)[None, :], eviction_policy="evict_first"
        )
        q.slice(start, width, 1).store((qv.to(gl.float32) * qs).to(gl.bfloat16))
        quantized_values = gl.load(
            K + offset[:, None] + col[None, :],
            valid[:, None],
            0.0,
            eviction_policy=policy,
        )
        e = ((scales[:, None] >> ((col // 64) * 8)[None, :]) & 255).to(gl.uint16)
        power = (e << 7).to(gl.bfloat16, bitcast=True)
        kv.slice(start, width, 1).store(
            gl.inline_asm_elementwise(
                "mul.rn.bf16x2 $0,$1,$2;",
                "=r,r,r",
                [quantized_values.to(gl.bfloat16), power],
                gl.bfloat16,
                is_pure=True,
                pack=2,
            )
        )

    @gluon.jit
    def one_block(
        Q,
        QR,
        QS,
        K,
        KS,
        KR,
        Indices,
        Output,
        L,
        scale,
        N: gl.constexpr,
        TOPK: gl.constexpr,
        STRIDE: gl.constexpr,
        H: gl.constexpr,
        SPLITS: gl.constexpr,
        BK: gl.constexpr,
    ):
        # All eight warps cooperate when a split is exactly one block. There
        # is no running softmax state, inter-partition handoff or KV recycling.
        layout: gl.constexpr = BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0])
        qkl: gl.constexpr = NVMMADistributedLayout([3, 0], [4, 2], [16, BK // 2, 16])
        pvl: gl.constexpr = NVMMADistributedLayout([3, 0], [4, 2], [16, 128, 16])
        shared: gl.constexpr = NVMMASharedLayout(128, 16)
        q = gl.allocate_shared_memory(gl.bfloat16, (64, 512), shared)
        kv = gl.allocate_shared_memory(gl.bfloat16, (BK, 512), shared)
        p = gl.allocate_shared_memory(gl.bfloat16, (64, BK), shared)
        batch = gl.program_id(0).to(gl.int64)
        head = gl.program_id(1) * 64
        block = gl.program_id(2)
        row0 = batch * H + head
        row_out = (batch * SPLITS + block) * H + head
        r = gl.arange(0, BK, layout=gl.constexpr(SliceLayout(1, layout)))
        h = row0 + gl.arange(0, 64, layout=gl.constexpr(SliceLayout(1, layout)))
        d = gl.arange(0, 64, layout=gl.constexpr(SliceLayout(0, layout)))
        pos = block * BK + r
        idx = gl.load(
            Indices + batch * TOPK + pos, pos < TOPK, -1, eviction_policy="evict_first"
        )
        valid = (idx >= 0) & (idx < N)
        idx = gl.where(valid, idx, 0)
        page = (idx // 64).to(gl.int64) * STRIDE
        slot = idx % 64
        offset = page + slot * 576
        policy: gl.constexpr = "evict_first" if N > 131072 else ""
        scales = gl.load(KS + page // 8 + 4608 + slot, valid, 0, eviction_policy=policy)
        async_copy.async_copy_global_to_shared(
            kv.slice(448, 64, 1),
            KR + (offset // 2 + 224)[:, None] + d[None, :],
            valid[:, None],
            cache_modifier=".cg",
        )
        async_copy.async_copy_global_to_shared(
            q.slice(448, 64, 1), QR + h[:, None] * 64 + d[None, :], cache_modifier=".cg"
        )
        async_copy.commit_group()
        bf_band(Q, QS, K, q, kv, h, offset, valid, scales, layout, policy, 0, 256)
        bf_band(Q, QS, K, q, kv, h, offset, valid, scales, layout, policy, 256, 128)
        bf_band(Q, QS, K, q, kv, h, offset, valid, scales, layout, policy, 384, 64)
        async_copy.wait_group(0)
        hopper.fence_async_shared()
        gl.thread_barrier()
        zero_scores = gl.full((64, BK), 0, gl.float32, layout=gl.constexpr(qkl))
        score = hopper.warpgroup_mma(
            q, kv.permute((1, 0)), zero_scores, use_acc=False
        ) * (scale * 1.4426950408889634)
        score = gl.where(
            gl.convert_layout(valid, gl.constexpr(SliceLayout(0, qkl)))[None, :],
            score,
            -float("inf"),
        )
        m = gl.max(score, 1)
        prob = gl.exp2(score - gl.where(m == -float("inf"), 0.0, m)[:, None])
        denom = gl.sum(prob, 1)
        safe = gl.where(denom == 0.0, 1.0, denom)
        p.store(prob.to(gl.bfloat16))
        hopper.fence_async_shared()
        gl.thread_barrier()
        zo = gl.full((64, 512), 0, gl.float32, layout=gl.constexpr(pvl))
        output = hopper.warpgroup_mma(p, kv, zo, use_acc=False)
        publish(Output, row_out, output, 1.0 / safe, 0, 512, SPLITS == 1)
        rows = row_out + gl.arange(0, 64, layout=gl.constexpr(SliceLayout(1, qkl)))
        gl.store(
            L + rows,
            gl.where(
                denom > 0.0,
                m * 0.6931471805599453 + gl.log(safe),
                float("inf") if SPLITS == 1 else -float("inf"),
            ),
        )

    @triton.jit
    def merge_two(V, L, Output, Lout, H: tl.constexpr):
        batch = tl.program_id(0).to(tl.int64)
        dims = tl.arange(0, 512)
        for off in tl.static_range(2):
            h = tl.program_id(1) * 2 + off
            row = batch * H + h
            base = batch * (2 * H) + h
            a = tl.load(L + base)
            b = tl.load(L + base + H)
            va = (a != -float("inf")) & (a != float("inf"))
            vb = (b != -float("inf")) & (b != float("inf"))
            if va & vb:
                m = tl.maximum(a, b)
                wa = tl.exp(a - m)
                wb = tl.exp(b - m)
                den = wa + wb
                value = (
                    wa * tl.load(V + base * 512 + dims).to(tl.float32)
                    + wb * tl.load(V + (base + H) * 512 + dims).to(tl.float32)
                ) / den
                stat = m + tl.log(den)
            else:
                ix = tl.where(va, base, base + H)
                value = tl.load(V + ix * 512 + dims, va | vb, 0.0).to(tl.float32)
                stat = tl.where(va, a, tl.where(vb, b, float("inf")))
            tl.store(Output + row * 512 + dims, value)
            tl.store(Lout + row, stat)

    @triton.jit
    def leader(
        Output,
        L,
        sq,
        sr,
        qs,
        sk,
        skr,
        ks,
        sv64,
        sp,
        alpha,
        factor,
        qfull,
        kfull,
        kfree,
        pfull,
        vfull,
        vfree,
        ofull,
        SCALE: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        BK: tl.constexpr,
        BUFS: tl.constexpr,
        RING: tl.constexpr,
    ):
        b, hb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rows = (b * S + split) * H + hb * 64 + tl.arange(0, 64)
        _, count = split_block_range(split, tl.cdiv(T, BK), S)
        maximum = tl.full((64,), -float("inf"), tl.float32)
        denom = tl.full((64,), 0.0, tl.float32)
        prev = tl.full((64,), 1.0, tl.float32)
        acc = tl.full((64, 64), 0.0, tl.float32)
        tail = tl.full((64, 64), 0.0, tl.float32)
        tle.gpu.barrier_wait(qfull[0])
        qscale = tl.load(tle.gpu.local_ptr(qs))
        for step in range(count):
            buf, cycle = step % BUFS, step // BUFS
            if BUFS == 1:
                tle.gpu.barrier_wait(kfull[0])
            else:
                tle.gpu.barrier_wait(kfull[buf], phaseIdx=cycle)
            logits = tle.gpu.wgmma(
                tle_subslice(sq, [0, 0], [64, 256]),
                tle_subslice(sk.slot(buf), [0, 0], [BK, 256]),
                out_dtype=tl.float32,
                trans_b=True,
            )
            logits = tle.gpu.wgmma(
                tle_subslice(sq, [0, 256], [64, 128]),
                tle_subslice(sk.slot(buf), [0, 256], [BK, 128]),
                logits,
                trans_b=True,
            )
            logits = tle.gpu.wgmma(
                tle_subslice(sq, [0, 384], [64, 64]),
                tle_subslice(sk.slot(buf), [0, 384], [BK, 64]),
                logits,
                trans_b=True,
            )
            rope = tle.gpu.wgmma(
                sr, skr.slot(step % RING), out_dtype=tl.float32, trans_b=True
            )
            logits = tle.gpu.wgmma_wait(0, logits)
            rope = tle.gpu.wgmma_wait(0, rope)
            sc = tl.load(tle.gpu.local_ptr(ks.slot(buf)))
            if RING > BUFS:
                tle.gpu.barrier_arrive(kfree[buf], phaseIdx=cycle)
            else:
                pass
            scores = tl.where(
                sc[None, :] >= 0.0,
                (logits * qscale[:, None] * sc[None, :] + rope)
                * (SCALE * 1.4426950408889634),
                -float("inf"),
            )
            maximum_new = tl.maximum(maximum, tl.max(scores, 1))
            safe = tl.where(maximum_new == -float("inf"), 0.0, maximum_new)
            corr = tl.exp2(maximum - safe)
            p = tl.exp2(scores - safe[:, None])
            denom = denom * corr + tl.sum(p, 1)
            maximum = maximum_new
            weighted = p * tl.maximum(sc, 0.0)[None, :]
            w = tl.max(weighted, 1) / 448.0
            w = tl.where(w > 0.0, w, 1.0)
            normalized = weighted / w[:, None]
            beta = prev * corr / w
            prev = w
            if BK == 64:
                publish_p_fp8_sw64_coupled_stmatrix(sp.slot(buf), normalized)
            else:
                publish_sparse_p128_fp8(sp.slot(0), normalized, sp.slot(1))
            tl.store(tle.gpu.local_ptr(alpha.slot(buf)), beta)
            tle.gpu.barrier_arrive(pfull[0])
            tail = tle.gpu.wgmma(
                p.to(tl.bfloat16), skr.slot(step % RING), tail * corr[:, None]
            )
            if RING == BUFS:
                tail = tle.gpu.wgmma_wait(0, tail)
                if BUFS == 1:
                    tle.gpu.barrier_arrive(kfree[0])
                else:
                    tle.gpu.barrier_arrive(kfree[buf], phaseIdx=cycle)
            else:
                pass
            tle.gpu.barrier_wait(vfull[0])
            acc *= beta[:, None]
            for half in tl.static_range(BK // 64):
                vbuf = buf * (BK // 64) + half
                acc = tle.gpu.wgmma(
                    sp.slot(vbuf),
                    sv64.slot(vbuf),
                    acc,
                    trans_b=True,
                )
            acc = tle.gpu.wgmma_wait(0, acc)
            if RING > BUFS:
                tail = tle.gpu.wgmma_wait(0, tail)
            else:
                pass
            if BUFS == 1 and tl.cdiv(T, BK) > S:
                tle.gpu.barrier_wait(vfree[0])
            else:
                pass
        inv = tl.where(denom > 0.0, 1.0 / denom, 0.0)
        f = inv * prev
        tl.store(tle.gpu.local_ptr(factor), f)
        tle.gpu.barrier_arrive(ofull[0])
        d = tl.arange(0, 64)
        cols = (d // 16) * 16 + (d % 8) * 2 + (d % 16) // 8
        tl.store(
            Output + rows[:, None] * 512 + 384 + cols[None, :],
            acc * f[:, None],
            cache_modifier=".cs" if S == 1 else "",
        )
        tl.store(
            Output + rows[:, None] * 512 + 448 + tl.arange(0, 64)[None, :],
            tail * inv[:, None],
            cache_modifier=".cs" if S == 1 else "",
        )
        tl.store(
            L + rows,
            tl.where(
                denom > 0.0,
                maximum * 0.6931471805599453 + tl.log(denom),
                float("inf") if S == 1 else -float("inf"),
            ),
        )

    @triton.jit
    def follower(
        Q,
        QR,
        QS,
        Flags,
        RepairState,
        sq,
        sr,
        qs,
        qfull,
        Output,
        sk,
        sv,
        sv128,
        sv64,
        sp,
        alpha,
        factor,
        kfull,
        kfree,
        pfull,
        vfull,
        vfree,
        ofull,
        T: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        BK: tl.constexpr,
        BUFS: tl.constexpr,
    ):
        b, hb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        h = (b * H + tl.program_id(1) * 64) + tl.arange(0, 64)
        d = tl.arange(0, 512)
        if S > 1 and T > 640:
            signed_scale = tl.load(QS + h)
            mx = tl.abs(signed_scale)
            qrepair = tl.max((signed_scale < 0.0).to(tl.int32), 0)
            tl.store(
                tle.gpu.local_ptr(sq),
                tl.load(
                    Q + h[:, None] * 448 + d[None, :],
                    d[None, :] < 448,
                    0.0,
                    eviction_policy="evict_first",
                ),
            )
        else:
            g = tl.arange(0, 8)
            scales = tl.load(QS + h[:, None] * 7 + g[None, :], g[None, :] < 7, 0.0)
            mx = tl.max(scales, 1)
            mn = tl.min(tl.where(scales > 0.0, scales, float("inf")), 1)
            qrepair = tl.max((mx > 4.0 * mn).to(tl.int32), 0)
            cols128 = tl.arange(0, 128)
            inv = 1.0 / tl.where(mx > 0.0, mx, 1.0)
            for pair in tl.static_range(4):
                dims = pair * 128 + cols128
                qpart = tl.load(
                    Q + h[:, None] * 448 + dims[None, :],
                    dims[None, :] < 448,
                    0.0,
                    eviction_policy="evict_first",
                )
                qleft = tl.load(QS + h * 7 + 2 * pair) * inv
                if pair < 3:
                    qright = tl.load(QS + h * 7 + 2 * pair + 1) * inv
                else:
                    qright = 0.0
                ratio = tl.where(cols128[None, :] < 64, qleft[:, None], qright[:, None])
                qnorm = (qpart.to(tl.float32) * ratio).to(tl.float8e4nv)
                tl.store(
                    tle.gpu.local_ptr(tle_subslice(sq, [0, pair * 128], [64, 128])),
                    qnorm,
                )
        tl.store(tle.gpu.local_ptr(qs), mx)
        tl.store(
            tle.gpu.local_ptr(sr),
            tl.load(
                QR + h[:, None] * 64 + tl.arange(0, 64)[None, :],
                eviction_policy="evict_first",
            ),
        )
        tle.gpu.barrier_arrive(qfull[0])
        tl.atomic_or(Flags + b, 1, mask=qrepair != 0, sem="relaxed")
        tl.atomic_or(RepairState, 1, mask=qrepair != 0, sem="relaxed")
        _, count = split_block_range(split, tl.cdiv(T, BK), S)
        a1 = tl.full((64, 128), 0.0, tl.float32)
        a = tl.full((64, 128), 0.0, tl.float32)
        c = tl.full((64, 128), 0.0, tl.float32)
        for step in range(count):
            buf, cycle = step % BUFS, step // BUFS
            if BUFS == 1:
                tle.gpu.barrier_wait(kfull[0])
            else:
                tle.gpu.barrier_wait(kfull[buf], phaseIdx=cycle)
            for half in tl.static_range(BK // 64):
                for group in tl.static_range(4):
                    src = tle_subslice(
                        sk.slot(buf), [half * 64, group * 128], [64, 128]
                    )
                    for warp_half in tl.static_range(1 if group == 3 else 2):
                        sparse_fp8_transpose_values(
                            src,
                            (sv if group < 2 else sv128 if group == 2 else sv64).slot(
                                buf * (BK // 64) + half
                            ),
                            group * 128 if group < 2 else 0,
                            warp_half * 4,
                        )
            tl.inline_asm_elementwise(
                "{ fence.proxy.async.shared::cta; mov.b32 $0, $1; }",
                "=r,r",
                [tl.arange(0, 128)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            tle.gpu.barrier_arrive(vfull[0])
            if BUFS == 1:
                tle.gpu.barrier_arrive(kfree[0])
            else:
                tle.gpu.barrier_arrive(kfree[buf], phaseIdx=cycle)
            tle.gpu.barrier_wait(pfull[0])
            beta = tl.load(tle.gpu.local_ptr(alpha.slot(buf)))
            a *= beta[:, None]
            a1 *= beta[:, None]
            c *= beta[:, None]
            for half in tl.static_range(BK // 64):
                vbuf = buf * (BK // 64) + half
                a = tle.gpu.wgmma(
                    sp.slot(vbuf),
                    tle_subslice(sv.slot(vbuf), [0, 0], [128, 64]),
                    a,
                    trans_b=True,
                )
                a1 = tle.gpu.wgmma(
                    sp.slot(vbuf),
                    tle_subslice(sv.slot(vbuf), [128, 0], [128, 64]),
                    a1,
                    trans_b=True,
                )
                c = tle.gpu.wgmma(
                    sp.slot(vbuf),
                    sv128.slot(vbuf),
                    c,
                    trans_b=True,
                )
            a1 = tle.gpu.wgmma_wait(0, a1)
            a = tle.gpu.wgmma_wait(0, a)
            c = tle.gpu.wgmma_wait(0, c)
            if BUFS == 1 and tl.cdiv(T, BK) > S:
                tle.gpu.barrier_wait(vfree[0])
            else:
                pass
        tle.gpu.barrier_wait(ofull[0])
        f = tl.load(tle.gpu.local_ptr(factor))
        sparse_fp8_publish_output(a, f, Output, b * S + split, hb, H, 0, S == 1)
        sparse_fp8_publish_output(a1, f, Output, b * S + split, hb, H, 128, S == 1)
        sparse_fp8_publish_output(c, f, Output, b * S + split, hb, H, 256, S == 1)

    @triton.jit
    def model1_fp8(
        PackedRope,
        Q,
        QR,
        QS,
        E,
        Idx,
        Output,
        L,
        Flags,
        RepairState,
        Normalized,
        Scale,
        N,
        T: tl.constexpr,
        H: tl.constexpr,
        STRIDE: tl.constexpr,
        SCALE: tl.constexpr,
        S: tl.constexpr,
        BK: tl.constexpr,
        BUFS: tl.constexpr,
        PREPACK: tl.constexpr,
    ):
        K = E.to(tl.pointer_type(tl.float8e4nv))
        KR = PackedRope if PREPACK else E.to(tl.pointer_type(tl.bfloat16))
        sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
        sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
        qs = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        sk = tle.gpu.alloc([BUFS, BK, 512], tl.float8e4nv, scope=tle.gpu.smem)
        RING: tl.constexpr = 4 if BUFS == 2 and not PREPACK else BUFS
        skr = tle.gpu.alloc([RING, BK, 64], tl.bfloat16, scope=tle.gpu.smem)
        ks = tle.gpu.alloc(
            [BUFS, BK], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        sv = tle.gpu.alloc(
            [BUFS * (BK // 64), 256, 64], tl.float8e4nv, scope=tle.gpu.smem
        )
        sv128 = tle.gpu.alloc(
            [BUFS * (BK // 64), 128, 64], tl.float8e4nv, scope=tle.gpu.smem
        )
        sv64 = tle.gpu.alloc(
            [BUFS * (BK // 64), 64, 64], tl.float8e4nv, scope=tle.gpu.smem
        )
        sp = tle.gpu.alloc(
            [BUFS * (BK // 64), 64, 64], tl.float8e4nv, scope=tle.gpu.smem
        )
        alpha = tle.gpu.alloc(
            [BUFS, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        factor = tle.gpu.alloc(
            [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        qfull = sparse_named_barriers(1, 256, 16)
        if BUFS == 1:
            kfull = sparse_named_barriers(1, 512, 18)
            kfree = sparse_named_barriers(1, 512, 19)
        else:
            kfull = tle.gpu.alloc_barriers(BUFS, arrive_count=1)
            kfree = tle.gpu.alloc_barriers(BUFS, arrive_count=2, init=tle.gpu.READY)
        vfull = sparse_named_barriers(1, 256, 20)
        pfull = sparse_named_barriers(1, 256, 21)
        vfree = sparse_named_barriers(1, 256, 23)
        ofull = sparse_named_barriers(1, 256, 17)
        tle.gpu.warp_specialize(
            [
                (
                    leader,
                    (
                        Output,
                        L,
                        sq,
                        sr,
                        qs,
                        sk,
                        skr,
                        ks,
                        sv64,
                        sp,
                        alpha,
                        factor,
                        qfull,
                        kfull,
                        kfree,
                        pfull,
                        vfull,
                        vfree,
                        ofull,
                        SCALE,
                        T,
                        H,
                        S,
                        BK,
                        BUFS,
                        RING,
                    ),
                ),
                (
                    producer,
                    (
                        K,
                        KR,
                        Idx,
                        Flags,
                        RepairState,
                        Normalized,
                        Scale,
                        sk,
                        skr,
                        ks,
                        kfull,
                        kfree,
                        N,
                        T,
                        STRIDE,
                        S,
                        BK,
                        BUFS,
                        PREPACK,
                    ),
                ),
                (
                    follower,
                    (
                        Q,
                        QR,
                        QS,
                        Flags,
                        RepairState,
                        sq,
                        sr,
                        qs,
                        qfull,
                        Output,
                        sk,
                        sv,
                        sv128,
                        sv64,
                        sp,
                        alpha,
                        factor,
                        kfull,
                        kfree,
                        pfull,
                        vfull,
                        vfree,
                        ofull,
                        T,
                        H,
                        S,
                        BK,
                        BUFS,
                    ),
                ),
            ],
            [8, 4],
            [40 if S > 1 and BK == 64 else 48, 248],
        )

    @triton.jit
    def repair_heads(
        Q,
        QR,
        QS,
        K,
        E,
        KR,
        Idx,
        Output,
        L,
        b,
        group,
        N,
        T: tl.constexpr,
        H: tl.constexpr,
        STRIDE: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        rows = b * H + group * 16 + tl.arange(0, 16)
        d = tl.arange(0, 512)
        q = tl.load(Q + rows[:, None] * 448 + d[None, :], d[None, :] < 448, 0.0)
        qs = tl.load(QS + rows[:, None] * 7 + d[None, :] // 64, d[None, :] < 448, 0.0)
        qr = tl.load(
            QR + rows[:, None] * 64 + (d % 64)[None, :], d[None, :] >= 448, 0.0
        )
        qbf = tl.where(d[None, :] < 448, q.to(tl.float32) * qs, qr).to(tl.bfloat16)
        m = tl.full((16,), -float("inf"), tl.float32)
        denominator = tl.full((16,), 0.0, tl.float32)
        acc = tl.full((16, 512), 0.0, tl.float32)
        for it in range(tl.cdiv(T, 32)):
            pos = it * 32 + tl.arange(0, 32)
            ids = tl.load(
                Idx + b.to(tl.int64) * T + pos,
                pos < T,
                -1,
                eviction_policy="evict_first",
            )
            valid = (ids >= 0) & (ids < N)
            ids = tl.where(valid, ids, 0)
            page = (ids >> 6).to(tl.int64) * STRIDE
            slot = ids & 63
            base = page + slot * 576
            e = tl.load(E + (page + 36864 + slot * 8)[:, None] + (d // 64)[None, :]).to(
                tl.int32
            )
            ks = tl.exp2((e - 127).to(tl.float32))
            raw = tl.load(
                K + base[:, None] + d[None, :], valid[:, None] & (d[None, :] < 448), 0.0
            )
            r = tl.load(
                KR + (base // 2 + 224)[:, None] + (d % 64)[None, :],
                valid[:, None] & (d[None, :] >= 448),
                0.0,
            )
            kv = tl.where(d[None, :] < 448, raw.to(tl.float32) * ks, r).to(tl.bfloat16)
            scores = tl.where(
                valid[None, :],
                tl.dot(qbf, tl.trans(kv)) * (SCALE * 1.4426950408889634),
                -float("inf"),
            )
            nm = tl.maximum(m, tl.max(scores, 1))
            safe = tl.where(nm == -float("inf"), 0.0, nm)
            p = tl.exp2(scores - safe[:, None])
            corr = tl.exp2(m - safe)
            acc = tl.dot(p.to(tl.bfloat16), kv, acc * corr[:, None])
            denominator = denominator * corr + tl.sum(p, 1)
            m = nm
        tl.store(
            Output + rows[:, None] * 512 + d[None, :],
            acc * tl.where(denominator > 0.0, 1.0 / denominator, 0.0)[:, None],
        )
        tl.store(
            L + rows,
            tl.where(
                denominator > 0.0,
                m * 0.6931471805599453 + tl.log(denominator),
                float("inf"),
            ),
        )

    @triton.jit
    def repair(
        Q,
        QR,
        QS,
        K,
        E,
        KR,
        Idx,
        Output,
        L,
        Flags,
        RepairState,
        B,
        N,
        T: tl.constexpr,
        H: tl.constexpr,
        STRIDE: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        # The usual scale range needs no second attention pass.
        if tl.load(RepairState) == 0:
            return
        grid = tl.num_programs(0)
        for b in range(tl.program_id(0), B, grid):
            if tl.load(Flags + b):
                for group in range(H // 16):
                    repair_heads(
                        Q,
                        QR,
                        QS,
                        K,
                        E,
                        KR,
                        Idx,
                        Output,
                        L,
                        b,
                        group,
                        N,
                        T,
                        H,
                        STRIDE,
                        SCALE,
                    )
            else:
                pass

    @gluon.jit
    def merge(
        o_part,
        lse_part,
        out,
        lse,
        H: gl.constexpr,
        SPLITS: gl.constexpr,
        D: gl.constexpr,
    ):
        # A warp covers four splits and eight column vectors. Reductions use
        # warp shuffles, including for padded and completely empty splits.
        layout: gl.constexpr = BlockedLayout([1, 8], [4, 8], [1, 2], [1, 0])
        ss = gl.arange(
            0,
            triton.next_power_of_2(SPLITS),
            layout=gl.constexpr(SliceLayout(1, layout)),
        )
        d = gl.program_id(2) * D + gl.arange(
            0, D, layout=gl.constexpr(SliceLayout(0, layout))
        )
        batch = gl.program_id(0).to(gl.int64)
        h = gl.program_id(1)
        rows = (batch * SPLITS + ss) * H + h
        stats = gl.load(lse_part + rows, ss < SPLITS, -float("inf"))
        stats = gl.where(stats == float("inf"), -float("inf"), stats)
        m = gl.max(stats, 0)
        empty = m == -float("inf")
        w = gl.exp(stats - gl.where(empty, 0.0, m))
        denom = gl.sum(w, 0)
        v = gl.load(
            o_part + rows[:, None] * 512 + d[None, :],
            ss[:, None] < SPLITS,
            0.0,
            eviction_policy="evict_first",
        )
        merged_value = gl.sum(v.to(gl.float32) * w[:, None], 0) / gl.where(
            empty, 1.0, denom
        )
        gl.store(out + (batch * H + h) * 512 + d, merged_value.to(gl.bfloat16))
        gl.store(
            lse + batch * H + h,
            gl.where(empty, float("inf"), m + gl.log(denom)),
            mask=gl.program_id(2) == 0,
        )

    def model1_attention(
        q: torch.Tensor,
        q_rope: torch.Tensor,
        q_scale: torch.Tensor,
        packed_cache: torch.Tensor,
        indices: torch.Tensor,
        *,
        tokens: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, heads, width = q.shape
        pages = packed_cache.shape[0]
        topk = indices.shape[-1]
        if batch == 0 or heads not in (64, 128) or width != 448 or topk == 0:
            raise ValueError(
                "Model1 requires nonempty queries, 64/128 heads and 448 FP8 dims"
            )
        if q.dtype != torch.float8_e4m3fn or not q.is_contiguous():
            raise ValueError("Query content must be contiguous E4M3")
        if (
            q_rope.shape != (batch, heads, 64)
            or q_rope.dtype != torch.bfloat16
            or not q_rope.is_contiguous()
        ):
            raise ValueError("Query RoPE must be contiguous BF16 [batch, heads, 64]")
        if (
            q_scale.shape != (batch, heads, 7)
            or q_scale.dtype != torch.float32
            or not q_scale.is_contiguous()
        ):
            raise ValueError("Query scales must be contiguous FP32 [batch, heads, 7]")
        if (
            packed_cache.shape != (pages, 64, 1, 584)
            or packed_cache.dtype != torch.uint8
        ):
            raise ValueError("Expected native V4 cache pages [pages, 64, 1, 584]")
        if (
            packed_cache.stride()[1:] != (584, 584, 1)
            or packed_cache.stride(0) % 16 != 0
            or packed_cache.data_ptr() % 16 != 0
            or packed_cache.stride(0) < 64 * 584
        ):
            raise ValueError(
                "V4 cache page data must be 16B-aligned and contiguous within a page"
            )
        if (
            indices.shape != (batch, 1, topk)
            or indices.dtype != torch.int32
            or not indices.is_contiguous()
        ):
            raise ValueError("Expected contiguous int32 indices [batch, 1, topk]")
        if not 0 < tokens <= pages * 64:
            raise ValueError("Token count is outside the allocated cache")
        if q.device.type != "cuda" or any(
            quantized_values.device != q.device
            for quantized_values in (q_rope, q_scale, packed_cache, indices)
        ):
            raise ValueError("Model1 tensors must share a CUDA device")

        fp8_keys, exponent_words, bf16_keys, page_stride, _, _ = cache_views(
            packed_cache
        )
        sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        work = batch * (heads // 64)
        sm_limit = sm // work
        block = (
            128 if 128 < topk <= 640 and 1 < triton.cdiv(topk, 128) <= sm_limit else 64
        )
        num_blocks = triton.cdiv(topk, block)
        target = max(1, sm_limit)
        exact = topk <= 128 and work <= 64
        if exact:
            block = 64 if topk <= 64 else 128
            splits = 1
        elif num_blocks >= 32 and sm < work < 4 * sm and 0 < work % sm <= sm // 4:
            splits = 2
        else:
            splits = (
                1
                if topk <= 128
                else triton.cdiv(num_blocks, triton.cdiv(num_blocks, target))
            )
        # Bound FP8 MMA accumulation before the FP32 split reduction.
        splits = max(splits, triton.cdiv(topk, 4096))
        out = torch.empty((batch, heads, 512), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((batch, heads), dtype=torch.float32, device=q.device)
        part = out if splits == 1 else out.new_empty((batch, splits, heads, 512))
        stats = lse if splits == 1 else lse.new_empty((batch, splits, heads))
        if exact:
            one_block[(batch, heads // 64, splits)](
                q,
                q_rope,
                q_scale,
                fp8_keys,
                exponent_words,
                bf16_keys,
                indices,
                part,
                stats,
                scale,
                tokens,
                topk,
                page_stride,
                heads,
                splits,
                block,
                num_warps=8,
                num_stages=1,
            )
            return out, lse
        repair_flags = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
        flags, repair_state = repair_flags[1:], repair_flags[:1]
        exponent_bytes = exponent_words.view(torch.uint8)
        prepack = topk > 128 and batch * topk >= 10 * tokens
        if splits > 1 and topk > 640:
            query = torch.empty_like(q)
            query_magnitude = q_scale.new_empty((batch, heads))
            normalize_query[(batch, heads // 4)](
                q, q_scale, query, query_magnitude, heads, num_warps=4
            )
        else:
            query, query_magnitude = q, q_scale
        if prepack:
            normalized = q.new_empty((tokens, 448))
            token_scale = q_scale.new_empty((tokens,))
            rope_cache = q_rope.new_empty((tokens, 64))
            normalize_cache[(triton.cdiv(tokens, 32),)](
                fp8_keys,
                exponent_bytes,
                normalized,
                token_scale,
                rope_cache,
                tokens,
                page_stride,
                num_warps=8,
            )
        else:
            normalized, token_scale, rope_cache = fp8_keys, q_scale, bf16_keys
        model1_fp8[(batch, heads // 64, splits)](
            rope_cache,
            query,
            q_rope,
            query_magnitude,
            exponent_bytes,
            indices,
            part,
            stats,
            flags,
            repair_state,
            normalized,
            token_scale,
            tokens,
            topk,
            heads,
            page_stride,
            scale,
            splits,
            block,
            128 // block,
            prepack,
            ptx_options="--register-usage-level=2",
            num_warps=4,
            num_stages=1,
        )
        if splits == 1:
            pass  # The attention kernel wrote the final output directly.
        elif splits == 2:
            merge_two[(batch, heads // 2)](part, stats, out, lse, heads, num_warps=2)
        else:
            d = 128 if batch <= 4 else 512
            merge[(batch, heads, 512 // d)](
                part,
                stats,
                out,
                lse,
                heads,
                splits,
                d,
                num_warps=2,
            )
        repair[(min(batch, 64),)](
            q,
            q_rope,
            q_scale,
            fp8_keys,
            exponent_bytes,
            bf16_keys,
            indices,
            out,
            lse,
            flags,
            repair_state,
            batch,
            tokens,
            topk,
            heads,
            page_stride,
            scale,
            num_warps=8,
            num_stages=1,
        )
        return out, lse

    SPARSE_LAUNCH_CACHE_LIMIT = 128

    SPARSE_LAUNCH_CACHE: dict[
        tuple, tuple[Callable[..., None], tuple[int | bool, ...]]
    ] = {}

    def launch_sparse_entry(
        entry: LibEntry,
        grid: tuple[int, ...] | Callable[[dict[str, object]], tuple[int, ...]],
        *arguments: torch.Tensor | int | float | bool | None,
        **launch_kwargs: int | bool,
    ) -> None:
        # Retain dynamic FlagTune hooks; ordinary entries tune once per exact ABI.
        if entry._has_flagtune_tuner:
            entry[grid](*arguments, **launch_kwargs)
            return
        signature = tuple(
            (
                (argument.dtype, argument.data_ptr() % entry.divisibility == 0)
                if isinstance(argument, torch.Tensor)
                else (type(argument), argument)
            )
            for argument in arguments
        )
        static_grid = None if callable(grid) else grid
        key = (
            entry,
            torch.cuda.current_device(),
            static_grid,
            signature,
            tuple(launch_kwargs.items()),
        )
        prepared = SPARSE_LAUNCH_CACHE.get(key)
        if prepared is None:
            kernel, constants = entry[grid](*arguments, **launch_kwargs)
            trailing = tuple(
                (
                    launch_kwargs[param.name]
                    if param.name in launch_kwargs
                    else constants[param.name]
                )
                for param in entry.jit_function.params[len(arguments) :]
            )
            if callable(grid):
                resolved_grid = grid(
                    {
                        **dict(zip(entry.arg_names, arguments)),
                        **launch_kwargs,
                        **constants,
                    }
                )
            else:
                resolved_grid = grid
            while len(SPARSE_LAUNCH_CACHE) >= SPARSE_LAUNCH_CACHE_LIMIT:
                SPARSE_LAUNCH_CACHE.clear()
            # Retain launch metadata without holding input or output Tensor storage.
            SPARSE_LAUNCH_CACHE[key] = (kernel[resolved_grid], trailing)
            return
        runner, trailing = prepared
        runner(*arguments, *trailing)

    def sparse_attention(
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_cache_lora: torch.Tensor,
        k_cache_rope: Optional[torch.Tensor],
        q_scale: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        indices: torch.Tensor,
        softmax_scale: Optional[float] = None,
        attn_sink: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        *,
        cache_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            q_nope.device.type != "cuda"
            or torch.cuda.get_device_capability(q_nope.device)[0] != 9
        ):
            raise NotImplementedError("FP8 sparse MLA requires NVIDIA Hopper CUDA")
        if k_cache_lora.dtype == torch.uint8:
            if k_cache_rope is not None or k_scale is not None:
                raise ValueError(
                    "Native V4 cache pages already contain RoPE and scales"
                )
            if attn_sink is not None or topk_length is not None:
                raise NotImplementedError("Native V4 uses the full supplied index row")
            if (
                q_nope.ndim != 4
                or q_nope.shape[1] != 1
                or q_rope.ndim != 4
                or q_scale.ndim != 4
            ):
                raise ValueError(
                    "Expected query components with singleton query length"
                )
            if q_rope.shape[1] != 1 or q_scale.shape[1] != 1:
                raise ValueError("Expected singleton RoPE and scale query length")
            output, lse = model1_attention(
                q_nope[:, 0],
                q_rope[:, 0],
                q_scale[:, 0],
                k_cache_lora,
                indices,
                tokens=(
                    k_cache_lora.shape[0] * 64 if cache_tokens is None else cache_tokens
                ),
                scale=512**-0.5 if softmax_scale is None else float(softmax_scale),
            )
            return output[:, None], lse[:, :, None]
        if cache_tokens is not None or k_cache_rope is None or k_scale is None:
            raise ValueError("Separate V32 pages require explicit RoPE and scales")
        if (
            q_nope.dtype != torch.float8_e4m3fn
            or k_cache_lora.dtype != torch.float8_e4m3fn
        ):
            raise TypeError("NoPE tensors must have dtype float8_e4m3fn")
        if q_rope.dtype != torch.bfloat16 or k_cache_rope.dtype != torch.bfloat16:
            raise TypeError("RoPE tensors must have dtype bfloat16")
        if q_nope.ndim != 4 or k_cache_lora.ndim != 3 or indices.ndim != 3:
            raise ValueError("q_nope, cache and indices must have ranks 4, 3 and 3")
        batch, query_length, heads, dim = q_nope.shape
        pages = k_cache_lora.shape[0]
        topk = indices.shape[-1]
        if query_length != 1 or heads not in (64, 128) or dim != 512:
            raise NotImplementedError(
                "Requires one query, 64/128 heads and 512 NoPE dimensions"
            )
        if (
            q_rope.ndim != 4
            or q_rope.shape[:3] != (batch, 1, heads)
            or q_rope.shape[-1] not in (0, 64)
        ):
            raise ValueError("q_rope must have shape [batch, 1, heads, 0 or 64]")
        rope_dim = q_rope.shape[-1]
        has_rope = rope_dim != 0
        if k_cache_lora.shape != (pages, 64, 512) or k_cache_rope.shape != (
            pages,
            64,
            rope_dim,
        ):
            raise ValueError(
                "Caches require page size 64 and the matching query RoPE dimension"
            )
        if indices.shape != (batch, 1, topk) or indices.dtype != torch.int32:
            raise ValueError("indices must be int32 [batch, 1, topk]")
        if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
            raise TypeError("q_scale and k_scale must have dtype float32")
        if q_scale.shape != (batch, 1, heads, 1) or k_scale.shape != (pages, 64, 1):
            raise ValueError(
                "Scale shapes must be [batch, 1, heads, 1] and [pages, 64, 1]"
            )
        for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope):
            if tensor.stride(-1) != 1:
                raise ValueError(
                    "NoPE and RoPE must be contiguous in the last dimension"
                )
        for tensor in (
            q_rope,
            k_cache_lora,
            k_cache_rope,
            indices,
            q_scale,
            k_scale,
            attn_sink,
            topk_length,
        ):
            if tensor is not None and tensor.device != q_nope.device:
                raise ValueError("All tensors must be on the same CUDA device")
        if attn_sink is not None and (
            attn_sink.shape != (heads,)
            or attn_sink.dtype != torch.float32
            or not attn_sink.is_contiguous()
        ):
            raise ValueError("attn_sink must be contiguous float32 [heads]")
        if topk_length is not None and (
            topk_length.shape != (batch,)
            or topk_length.dtype != torch.int32
            or not topk_length.is_contiguous()
        ):
            raise ValueError("topk_length must be contiguous int32 [batch]")
        output = torch.empty(
            (batch, 1, heads, 512), device=q_nope.device, dtype=torch.bfloat16
        )
        lse = torch.empty((batch, heads, 1), device=q_nope.device, dtype=torch.float32)
        if batch == 0:
            return output, lse
        if pages == 0 or topk == 0:
            rows = batch * heads
            sparse_fp8_empty[(triton.cdiv(rows * 512, 1024),)](
                output, lse, rows, num_warps=4
            )
            return output, lse
        softmax_scale = (
            (512 + rope_dim) ** -0.5 if softmax_scale is None else float(softmax_scale)
        )
        if batch <= 16 and not has_rope and topk <= 128:
            launch_sparse_entry(
                sparse_fp8_short,
                lambda meta: (
                    batch,
                    triton.cdiv(heads, meta["BLOCK_H"]),
                    512 // meta["VALUE_DIM"],
                ),
                q_nope,
                k_cache_lora,
                q_scale,
                k_scale,
                indices,
                indices if topk_length is None else topk_length,
                q_scale if attn_sink is None else attn_sink,
                output,
                lse,
                q_nope.stride(0),
                q_nope.stride(2),
                k_cache_lora.stride(0),
                k_cache_lora.stride(1),
                q_scale.stride(0),
                q_scale.stride(2),
                k_scale.stride(0),
                k_scale.stride(1),
                indices.stride(0),
                indices.stride(2),
                batch,
                heads,
                pages * 64,
                topk,
                softmax_scale,
                topk_length is not None,
                attn_sink is not None,
            )
            return output, lse
        # TLE does not yet lower the FP8-only compact shared-memory layout.
        use_partitioned = batch <= 16 and has_rope
        num_sms = get_num_sms(q_nope.device)
        head_groups = batch * (heads // 64)
        desired_splits = max(1, num_sms // head_groups)
        keys_per_split = 64 if batch <= 16 else 256
        max_splits = min(desired_splits, 32, max(1, topk // keys_per_split))
        splits = 1 << (max_splits.bit_length() - 1)
        if use_partitioned:
            can_async_copy = all(
                tensor.data_ptr() % 16 == 0
                and tensor.stride(0) * tensor.element_size() % 16 == 0
                and tensor.stride(-2) * tensor.element_size() % 16 == 0
                for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope)
            )
            tile_splits = triton.next_power_of_2(max(1, triton.cdiv(topk, 64)))
            # Keep value tiling in its validated workset; short rows use the compact CTA.
            use_tile = topk >= 512 and head_groups <= 8 and tile_splits <= 32
            if use_tile:
                splits = tile_splits
            else:
                # Checked merge consumes partial statistics, including for empty/short rows.
                splits = max(splits, 2)
        else:
            use_tile = False
        # Bound FP8 tensor-core accumulation; split outputs are merged in FP32.
        max_accumulation_keys = 4096
        splits = max(
            splits, triton.next_power_of_2(triton.cdiv(topk, max_accumulation_keys))
        )
        flag_heads = heads // 64
        repair_flags = (
            None
            if use_tile
            else torch.empty(
                (batch, flag_heads, splits), device=q_nope.device, dtype=torch.int32
            )
        )
        if splits > 1:
            partial = torch.empty(
                (batch, splits, heads, 512),
                device=q_nope.device,
                dtype=torch.bfloat16,
            )
            stats = torch.empty(
                (batch, splits, heads, 2), device=q_nope.device, dtype=torch.float32
            )
        else:
            partial, stats = output, lse
        if use_tile:
            launch_sparse_entry(
                sparse_fp8_tile,
                lambda meta: (batch, heads // 64, splits * (512 // meta["BLOCK_D"])),
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                q_scale,
                k_scale,
                indices,
                indices if topk_length is None else topk_length,
                q_scale if attn_sink is None else attn_sink,
                partial,
                stats,
                q_nope.stride(0),
                q_nope.stride(2),
                q_rope.stride(0),
                q_rope.stride(2),
                k_cache_lora.stride(0),
                k_cache_lora.stride(1),
                k_cache_rope.stride(0),
                k_cache_rope.stride(1),
                q_scale.stride(0),
                q_scale.stride(2),
                k_scale.stride(0),
                k_scale.stride(1),
                indices.stride(0),
                indices.stride(2),
                batch,
                heads,
                pages * 64,
                topk,
                softmax_scale,
                splits,
                topk_length is not None,
                attn_sink is not None,
                CAN_ASYNC=can_async_copy,
                HAS_ROPE=has_rope,
            )
        elif use_partitioned:
            launch_sparse_entry(
                sparse_fp8_compact,
                (batch, heads // 64, splits),
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                q_scale,
                k_scale,
                indices,
                indices if topk_length is None else topk_length,
                q_scale if attn_sink is None else attn_sink,
                partial,
                stats,
                repair_flags,
                q_nope.stride(0),
                q_nope.stride(2),
                q_rope.stride(0),
                q_rope.stride(2),
                k_cache_lora.stride(0),
                k_cache_lora.stride(1),
                k_cache_rope.stride(0),
                k_cache_rope.stride(1),
                q_scale.stride(0),
                q_scale.stride(2),
                k_scale.stride(0),
                k_scale.stride(1),
                indices.stride(0),
                indices.stride(2),
                batch,
                heads,
                pages * 64,
                topk,
                softmax_scale,
                splits,
                topk_length is not None,
                attn_sink is not None,
                CAN_ASYNC=can_async_copy,
                HAS_ROPE=has_rope,
            )
        else:
            b_bucket = 0 if head_groups * splits <= 128 else 1
            launch_sparse_entry(
                sparse_fp8_warp_specialized,
                (batch * (heads // 64), 1, splits),
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                q_scale,
                k_scale,
                indices,
                indices if topk_length is None else topk_length,
                q_scale if attn_sink is None else attn_sink,
                partial,
                stats,
                q_nope.stride(0),
                q_nope.stride(2),
                q_rope.stride(0),
                q_rope.stride(2),
                k_cache_lora.stride(0),
                k_cache_lora.stride(1),
                k_cache_rope.stride(0),
                k_cache_rope.stride(1),
                q_scale.stride(0),
                q_scale.stride(2),
                k_scale.stride(0),
                k_scale.stride(1),
                indices.stride(0),
                indices.stride(2),
                b_bucket,
                heads,
                pages * 64,
                topk,
                softmax_scale,
                topk_length is not None,
                splits,
                attn_sink is not None,
                repair_flags,
                HAS_ROPE=has_rope,
                CAN_ASYNC_ROPE=(
                    has_rope
                    and k_cache_rope.data_ptr() % 16 == 0
                    and (k_cache_rope.stride(0) * 2) % 16 == 0
                    and (k_cache_rope.stride(1) * 2) % 16 == 0
                ),
                CAN_ASYNC_NOPE=(
                    k_cache_lora.data_ptr() % 16 == 0
                    and k_cache_lora.stride(0) % 16 == 0
                    and k_cache_lora.stride(1) % 16 == 0
                ),
            )
        if splits == 1:
            if use_partitioned:
                return output, lse
            launch_sparse_entry(
                sparse_fp8_repair,
                lambda meta: (
                    triton.cdiv(batch, meta["REPAIR_GROUP"]),
                    triton.cdiv(heads, meta["BLOCK_H"]),
                    splits,
                ),
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                indices,
                q_scale,
                k_scale,
                attn_sink,
                topk_length,
                output,
                lse,
                partial,
                stats,
                q_nope.stride(0),
                q_nope.stride(2),
                q_rope.stride(0),
                q_rope.stride(2),
                k_cache_lora.stride(0),
                k_cache_lora.stride(1),
                k_cache_rope.stride(0),
                k_cache_rope.stride(1),
                indices.stride(0),
                indices.stride(2),
                q_scale.stride(0),
                q_scale.stride(2),
                k_scale.stride(0),
                k_scale.stride(1),
                batch,
                heads,
                pages * 64,
                topk,
                softmax_scale,
                splits,
                attn_sink is not None,
                topk_length is not None,
                repair_flags,
                REPAIR_SPLITS=splits,
                HAS_ROPE=has_rope,
            )
        else:
            head_tile = 2 if splits == 2 else 4
            sparse_fp8_fast_merge[(batch, heads // head_tile)](
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                q_scale,
                k_scale,
                indices,
                indices if topk_length is None else topk_length,
                q_scale if attn_sink is None else attn_sink,
                stats if repair_flags is None else repair_flags,
                partial,
                stats,
                output,
                lse,
                q_nope.stride(0),
                q_nope.stride(2),
                q_rope.stride(0),
                q_rope.stride(2),
                k_cache_lora.stride(0),
                k_cache_lora.stride(1),
                k_cache_rope.stride(0),
                k_cache_rope.stride(1),
                q_scale.stride(0),
                q_scale.stride(2),
                k_scale.stride(0),
                k_scale.stride(1),
                indices.stride(0),
                indices.stride(2),
                heads,
                pages * 64,
                topk,
                softmax_scale,
                splits,
                topk_length is not None,
                attn_sink is not None,
                CHECK_REPAIR=not use_tile,
                HEAD_TILE=head_tile,
                HAS_ROPE=has_rope,
                num_warps=2,
            )
        return output, lse


def flash_mla_sparse_fwd_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: Optional[torch.Tensor],
    q_scale: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    cache_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode using the separate per-token cache format of dense MLA.

    q_nope [B, 1, H, 512] and k_cache_lora [P, 64, 512] are FP8 e4m3fn;
    q_rope [B, 1, H, R] and k_cache_rope [P, 64, R] are BF16.
    R is 0 for inputs without RoPE, or 64 for the RoPE component.
    Both NoPE and RoPE store values divided by the corresponding FP32 scale:
    q_scale [B, 1, H, 1] and k_scale [P, 64, 1]. This directly accepts
    quantize_q_ckv_per_token / quantize_k_ckv_per_token outputs from dense MLA.
    V is the dequantized 512-dimensional NoPE cache. H must be 64 or 128.

    indices [B, 1, topk] contains int32 physical token IDs (page * 64 + slot).
    Negative and out-of-range IDs are ignored. topk_length [B] optionally
    limits the number of entries read per request. attn_sink [H] affects
    output only. Inputs must be finite, with positive finite scales.

    Native V4 takes 448 contiguous FP8 query channels, seven FP32 scales and
    64 unscaled BF16 RoPE channels. Pass the uint8 [P,64,1,584] page view
    as k_cache_lora with k_cache_rope and k_scale set to None. cache_tokens
    optionally excludes padding in the last page.

    Returns BF16 output [B, 1, H, 512] and natural-log FP32 LSE [B, H, 1].
    Empty attention produces zero output and +inf LSE. Forward only.
    Sensitive rows use FP32 recomputation inline, during merge, or in the
    existing separate repair kernel, depending on the selected path.
    Requires FlagTree cross-dtype WGMMA support (PR #1001).
    """
    if sparse_attention is None:
        raise NotImplementedError(
            "FP8 sparse MLA requires Hopper and FlagTree GPU extensions"
        )

    return sparse_attention(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        softmax_scale,
        attn_sink,
        topk_length,
        cache_tokens=cache_tokens,
    )

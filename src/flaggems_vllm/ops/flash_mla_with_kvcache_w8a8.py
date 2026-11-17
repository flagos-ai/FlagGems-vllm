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

import math
from itertools import accumulate
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_mla import (
    _ensure_triton_descriptor_allocator,
    _get_num_sms,
    _get_tensor_descriptor_cls,
)
from flaggems_vllm.utils import has_triton_tle_attrs, libentry, libtuner

HAS_TLE = runtime.device.vendor_name == "nvidia" and has_triton_tle_attrs(
    ("gpu.warp_specialize", "gpu.wgmma", "gpu.copy"), 3, 6, 0
)
if HAS_TLE:
    import triton.experimental.tle.language as tle


class FlashMLAFp8SplitKSchedMeta:
    """Reusable Split-K scheduling metadata."""

    def __init__(self) -> None:
        self.have_initialized = False
        self.num_splits = None
        self.split_batch = None
        self.split_page_begin = None
        self.split_num_pages = None
        self.split_order = None
        self.total_split_capacity = 0
        self.adaptive_fixed_pages = None
        self.adaptive_fixed_pairs = None
        self.capacity_splits = ()
        self.padded_pages = 0
        # Shape a GPU length plan was built for (get_mla_fp8_metadata).
        self.batch_size = 0
        self.query_heads = 0


# Fixed64-head/64-token tiles, two WGs, four base warps and one stage are
# coupled to PTX register/shared-memory permutations and named-barrier IDs.
# These structural parameters retain their validated tuning exemption.
D_CKV = 512  # content / V head dim


D_ROPE = 64  # rope tail dim


PAGE_SIZE = 64  # paged KV cache page size (= BK)


FP8_MAX = 448.0  # E4M3 dynamic range upper bound


LOG2E = 1.4426950408889634


LN2 = 0.6931471805599453


P_AMAX_FLOOR = 1e-26


TLE_FP8_BH = 64  # heads per iteration


K_CONTENT_TILE_HOST = 128


K_CONTENT_TILE = tl.constexpr(K_CONTENT_TILE_HOST)


DEFAULT_PAGES_PER_SPLIT = 2


MAX_SEQUENCE_LENGTH = 65536


TLE_LOG2E = tl.constexpr(LOG2E)


TLE_LN2 = tl.constexpr(LN2)


TLE_FP8_MAX = tl.constexpr(FP8_MAX)


TLE_P_AMAX_FLOOR = tl.constexpr(P_AMAX_FLOOR)


TLE_NEG_INF = tl.constexpr(float("-inf"))


ADAPTIVE_MODEL_MIN_PAGES = 69


ADAPTIVE_MIN_FIXED_PAGES = 4


ADAPTIVE_MAX_FIXED_PAGES = 32


ADAPTIVE_TAIL_WAVE_MAX_FIXED_PAGES = 34


CUDA_REF_FIXED_OVERHEAD_PAGES = 5


TLE_PAGE_SIZE = tl.constexpr(64)


# Per-split setup and combine cost in KV pages for the length planner.
TLE_SPLIT_OVERHEAD_PAGES = tl.constexpr(4)


# Split capacity per SM partition, beyond one split per batch row.
LENGTH_SPLIT_CAPACITY_PER_PART = 2


TLE_POS_INF = tl.constexpr(float("inf"))


D_QK = 576  # Q/K head dim (content 512 + rope 64)


TLE_FP8_BK = 64  # KV tokens per iteration (= PAGE_SIZE)


TLE_FP8_DPH = 256  # output 512 dim split into left/right halves of 256


COMBINE_BLOCK_SPLITS = 8


COMBINE_BLOCK_D = 128


CUDA_COARSE_COMBINE_BLOCK_SPLITS = 32


CUDA_COARSE_COMBINE_BLOCK_ROWS = 8


CUDA_COARSE_COMBINE_MIN_BATCH = 4


LSE_FINALIZE_BLOCK = 256


if HAS_TLE:

    # Saturating FP8 publication also supplies the bound for scaled probabilities.
    @triton.jit
    def publish_p_fp8_sw64_coupled_stmatrix(s_p, p):
        """CUDA-native P publication; V repack carries the matching K permutation."""
        base = tle.gpu.local_ptr(s_p, (0, 0))
        base_u32 = tl.inline_asm_elementwise(
            asm="mov.u32 $0, $1;",
            constraints="=r,r",
            args=[base],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15;\n"
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
            ),
            constraints=(
                "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,"
                "=r,=r,=r,=r,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,r,r,r,r,"
                "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r"
            ),
            args=[p, base_u32],
            dtype=tl.uint32,
            is_pure=False,
            pack=32,
        )


if HAS_TLE:

    @triton.jit
    def vtranspose_fp8_64x128_plain(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
    ):
        """CUDA-authority SW128 -> SW64 FP8 transpose for one 64x128 tile."""
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
                # CUDA's LDSM/STSM register order presents source-row bits as
                # [b1,b3,b2,b0] to TLE's logical SW64 view.  Apply the inverse
                # [b3,b1,b2,b0] mapping at the load boundary so PV observes
                # the same logical transpose as the tensor path.
                "and.b32 src_row, lane, 17;\n"
                "and.b32 tmp, lane, 8;\n"
                "shr.u32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 2;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 4;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
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
                "and.b32 dst_row_r, lane, 7;\n"
                "shl.b32 dst_row_r, dst_row_r, 1;\n"
                "shr.u32 tmp, lane, 3;\n"
                "and.b32 tmp, tmp, 1;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
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
                "add.u32 src_log, src_log, 64;\n"
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
                "add.u32 dst_log0, dst_log0, 4096;\n"
                "add.u32 dst_log1, dst_log1, 4096;\n"
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
            constraints="=r,r,r,r",
            args=[carrier, src_base, dst_base],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )


if HAS_TLE:

    @triton.jit
    def vtranspose_fp8_64x128_kperm(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
    ):
        """CUDA-authority SW128 -> SW64 FP8 transpose for one 64x128 tile."""
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
                # CUDA's LDSM/STSM register order presents source-row bits as
                # [b1,b3,b2,b0] to TLE's logical SW64 view.  Apply the inverse
                # [b3,b1,b2,b0] mapping at the load boundary so PV observes
                # the same logical transpose as the tensor path.
                "and.b32 src_row, lane, 17;\n"
                "and.b32 tmp, lane, 8;\n"
                "shr.u32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 2;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, lane, 4;\n"
                "shl.b32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                # Direct CUDA STSM presents P to TLE as dest <- source pi,
                # pi=[0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] per K16.
                # Load V from pi(dest) as well, preserving the dot product
                # while removing the publication-side cross-lane shuffle.
                "mov.u32 tmp2, src_row;\n"
                "and.b32 src_row, tmp2, 17;\n"
                "and.b32 tmp, tmp2, 4;\n"
                "shr.u32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, tmp2, 8;\n"
                "shr.u32 tmp, tmp, 1;\n"
                "or.b32 src_row, src_row, tmp;\n"
                "and.b32 tmp, tmp2, 2;\n"
                "shl.b32 tmp, tmp, 2;\n"
                "or.b32 src_row, src_row, tmp;\n"
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
                "and.b32 dst_row_r, lane, 7;\n"
                "shl.b32 dst_row_r, dst_row_r, 1;\n"
                "shr.u32 tmp, lane, 3;\n"
                "and.b32 tmp, tmp, 1;\n"
                "add.u32 dst_row_r, dst_row_r, tmp;\n"
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
                "add.u32 src_log, src_log, 64;\n"
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
                "add.u32 dst_log0, dst_log0, 4096;\n"
                "add.u32 dst_log1, dst_log1, 4096;\n"
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
            constraints="=r,r,r,r",
            args=[carrier, src_base, dst_base],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )


if HAS_TLE:

    @triton.jit
    def vtranspose_fp8_64x128(
        s_src,
        s_dst,
        dst_row: tl.constexpr,
        permute_k: tl.constexpr,
    ):
        if permute_k:
            return vtranspose_fp8_64x128_kperm(s_src, s_dst, dst_row)
        return vtranspose_fp8_64x128_plain(s_src, s_dst, dst_row)


if HAS_TLE:

    @triton.jit
    def zero_invalid_fp8_rows_sw128_x4(
        s_src0,
        s_src1,
        s_src2,
        s_src3,
        valid_tokens,
    ):
        """Zero the same invalid rows in four SW128 64x128 FP8 tiles.

        The four content tiles share row validity and SW128 addressing.  Keep
        the four 16B stores per tile, but compute the predicate and swizzled
        byte offset only once.
        """
        carrier = tl.arange(0, 128).to(tl.uint32)
        src0_base = tle.gpu.local_ptr(s_src0, (0, 0))
        src1_base = tle.gpu.local_ptr(s_src1, (0, 0))
        src2_base = tle.gpu.local_ptr(s_src2, (0, 0))
        src3_base = tle.gpu.local_ptr(s_src3, (0, 0))
        return tl.inline_asm_elementwise(
            asm=(
                "{\n"
                ".reg .pred invalid;\n"
                ".reg .b32 tid, lane, warp, row, col, logical, swz, off, addr, z;\n"
                "mov.u32 tid, %tid.x;\n"
                "and.b32 tid, tid, 127;\n"
                "and.b32 lane, tid, 31;\n"
                "shr.u32 warp, tid, 5;\n"
                "mov.u32 row, lane;\n"
                "shl.b32 col, warp, 4;\n"
                "setp.ge.u32 invalid, row, $6;\n"
                "mov.u32 z, 0;\n"
                "shl.b32 logical, row, 7;\n"
                "add.u32 logical, logical, col;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 64;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 row, row, 32;\n"
                "setp.ge.u32 invalid, row, $6;\n"
                "shl.b32 logical, row, 7;\n"
                "add.u32 logical, logical, col;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 logical, logical, 64;\n"
                "shr.u32 swz, logical, 7;\n"
                "and.b32 swz, swz, 7;\n"
                "shl.b32 swz, swz, 4;\n"
                "xor.b32 off, logical, swz;\n"
                "add.u32 addr, $2, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $3, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $4, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "add.u32 addr, $5, off;\n"
                "@invalid st.shared.v4.b32 [addr], {z, z, z, z};\n"
                "mov.u32 $0, $1;\n"
                "}"
            ),
            constraints="=r,r,r,r,r,r,r",
            args=[
                carrier,
                src0_base,
                src1_base,
                src2_base,
                src3_base,
                valid_tokens,
            ],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )


if HAS_TLE:

    @triton.jit
    def fp8_mla_wg0(
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        block_table,
        stride_bt_pg,
        row0,
        num_pages,
        q_ckv_full,
        q_rope_full,
        q_scale_full,
        k_content_full,
        k_rope_full,
        k_scale_full,
        state0_ready,
        state1_ready,
        p0_ready,
        p1_ready,
        v0_ready,
        v1_ready,
        slot0_empty,
        slot1_empty,
        tail0_zero_ready,
        s_q,
        s_qr,
        s_kc_a0,
        s_kc_a1,
        s_kc_a2,
        s_kc_a3,
        s_kc_b0,
        s_kc_b1,
        s_kc_b2,
        s_kc_b3,
        s_kr_a,
        s_vt0_a,
        s_vt1_a,
        s_vt0_b,
        s_p_a,
        s_p_b,
        s_beta_a,
        s_beta_b,
        s_ks_a,
        s_ks_b,
        s_state0_m,
        s_state0_s,
        s_state0_l,
        s_state0_valid,
        s_state1_m,
        s_state1_s,
        s_state1_l,
        s_state1_valid,
        split_cache_seqlen,
        causal_cache_seqlen,
        out_ptr,
        lse2_ptr,
        stride_po_h,
        stride_pl_h,
        h_base,
        softmax_scale,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        DP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        QUERY_HEADS: tl.constexpr,
        QUERY_COUNT: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
        KNOWN_NUM_PAGES: tl.constexpr,
        final_out_ptr,
        stride_fo_h,
        final_lse_ptr,
        stride_fl_h,
        stride_fl_q,
        is_single_split,
        has_output,
        DYNAMIC_DIRECT: tl.constexpr,
        TMA_DIRECT: tl.constexpr,
        LSE_HEADS: tl.constexpr,
    ):
        """WG0: Q owner, even-page math, and the left output half."""
        if KNOWN_NUM_PAGES > 0:
            # The immutable host plan proves this logical page count.
            # Keep it opaque to layout propagation and specialize in LLVM.
            tl.assume(num_pages == KNOWN_NUM_PAGES)
            # The three CUDA-aligned Q payloads are one-shot TMA transactions.  The
            # scale temporarily occupies state1_m; WG1 cannot overwrite that field
            # until state0_ready, after both workers have consumed Q scale.
        s_state1_m_row = s_state1_m.slot(0)
        s_beta_a_row = s_beta_a.slot(0)
        s_beta_b_row = s_beta_b.slot(0)
        s_ks_a_row = s_ks_a.slot(0)
        state_idx = tl.arange(0, BH)
        tle.gpu.copy(q_desc, s_q, [BH, CKV], [row0, 0], barrier=q_ckv_full)
        tle.gpu.copy(qr_desc, s_qr, [BH, ROPE], [row0, 0], barrier=q_rope_full)
        tle.gpu.copy(
            qs_desc,
            s_state1_m,
            [1, BH],
            [row0 // HQ, h_base],
            barrier=q_scale_full,
        )
        tle.gpu.barrier_wait(q_ckv_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_rope_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_scale_full, phaseIdx=0)

        offs_t = tl.arange(0, BK)
        offs_h = h_base + tl.arange(0, BH)
        mask_h = offs_h < HQ
        query_end = causal_cache_seqlen
        if QUERY_COUNT > 1:
            query_end = causal_cache_seqlen - (QUERY_COUNT - 1 - offs_h // QUERY_HEADS)
        qs = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), volatile=True)

        acc_left = tl.zeros((BH, DP), dtype=tl.float32)
        state_m = tl.full((BH,), float("-inf"), tl.float32)
        state_s = tl.full((BH,), 1.0, tl.float32)
        state_l = tl.zeros((BH,), dtype=tl.float32)
        state_valid = tl.zeros((BH,), dtype=tl.int32) != 0

        q_rows_d128 = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, K_CONTENT_TILE))
        q_c0_cols = tl.broadcast_to(
            tl.arange(0, K_CONTENT_TILE)[None, :], (BH, K_CONTENT_TILE)
        )
        q_c1_cols = tl.broadcast_to(
            (K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c2_cols = tl.broadcast_to(
            (2 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c3_cols = tl.broadcast_to(
            (3 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c0 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c0_cols)))
        q_c1 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c1_cols)))
        q_c2 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c2_cols)))
        q_c3 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c3_cols)))
        k_a_c0 = s_kc_a0
        k_a_c1 = s_kc_a1
        k_a_c2 = s_kc_a2
        k_a_c3 = s_kc_a3
        k_b_c0 = s_kc_b0
        k_b_c1 = s_kc_b1
        kv_rows_d128 = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DP // 2))
        kv_c0_cols = tl.broadcast_to(tl.arange(0, DP // 2)[None, :], (BK, DP // 2))
        vt_c0_rows = tl.broadcast_to(tl.arange(0, DP // 2)[:, None], (DP // 2, BK))
        vt_c1_rows = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[:, None], (DP // 2, BK)
        )
        vt_cols = tl.arange(0, BK)
        # The masked transpose uses the same K permutation as coupled STSM.
        vt_cols = (
            (vt_cols & ~14)
            | ((vt_cols & 2) << 1)
            | ((vt_cols & 4) << 1)
            | ((vt_cols & 8) >> 2)
        )
        vt_cols_d128 = tl.broadcast_to(vt_cols[None, :], (DP // 2, BK))

        num_pairs = (num_pages + 1) // 2
        # Fixed writer ownership applies to cold prime and steady state: WG0
        # issues content tiles 0/1 for both physical slots.
        if num_pages > 0:
            first_phys = tl.load(block_table)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c0,
                [BK, K_CONTENT_TILE],
                [first_base, 0],
                barrier=k_content_full[0],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c1,
                [BK, K_CONTENT_TILE],
                [first_base, K_CONTENT_TILE],
                barrier=k_content_full[1],
            )
        if num_pages > 1:
            first_phys = tl.load(block_table + stride_bt_pg)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_b_c0,
                [BK, K_CONTENT_TILE],
                [first_base, 0],
                barrier=k_content_full[4],
            )
            tle.gpu.copy(
                k_desc,
                k_b_c1,
                [BK, K_CONTENT_TILE],
                [first_base, K_CONTENT_TILE],
                barrier=k_content_full[5],
            )

            # Cold prime: page 0 QK, scale, and V are steady-loop live-ins. Rope
            # accumulates after content tile 3, matching the CUDA rP0 sequence.
        qk = tl.zeros((BH, BK), dtype=tl.float32)
        ks = tl.zeros((BK,), dtype=tl.float32)
        if num_pages > 0:
            tle.gpu.barrier_wait(k_content_full[0], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c0, k_a_c0, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[1], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c1, k_a_c1, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[2], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c2, k_a_c2, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[3], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c3, k_a_c3, qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[0], phaseIdx=0)
            qk = tle.gpu.wgmma(s_qr, s_kr_a, qk, trans_b=True)
            qk = tle.gpu.wgmma_wait(0, qk)

            tle.gpu.barrier_wait(k_scale_full[0], phaseIdx=0)
            prime_valid = offs_t < split_cache_seqlen
            ks_raw = tl.load(tle.gpu.local_ptr(s_ks_a_row, (offs_t,)))
            ks = ks_raw if FULL_TAIL else tl.where(prime_valid, ks_raw, 0.0)

        steady_pairs = tl.maximum(num_pairs - 1, 0)
        for pair in tl.range(steady_pairs, disable_licm=True):
            page = pair * 2
            valid = tl.full((BK,), True, tl.int1)
            valid_row = valid[None, :]
            score = qk * qs[:, None] * ks[None, :] * softmax_scale
            score_safe = score
            x = score_safe * TLE_LOG2E
            page_m = tl.max(x, axis=1)
            old_m = tl.where(state_valid, state_m, TLE_NEG_INF)
            old_s = tl.where(state_valid, state_s, 1.0)
            old_l = tl.where(state_valid, state_l, 0.0)
            m_new = tl.maximum(old_m, page_m)
            m_safe = tl.where(m_new == TLE_NEG_INF, 0.0, m_new)
            e = tl.exp2(x - m_safe[:, None])
            f = e * ks[None, :]
            amax = tl.max(tl.abs(f), axis=1)
            s_new = tl.where(
                amax == 0.0,
                1.0,
                tl.maximum(amax, TLE_P_AMAX_FLOOR) / TLE_FP8_MAX,
            )
            page_valid = True
            if USE_HOTLOOP_RECIP:
                inv_s_new = 1.0 / s_new
                p_scaled = f * inv_s_new[:, None]
            else:
                p_scaled = f / s_new[:, None]
            p_new = p_scaled
            p0 = p_new
            publish_p_fp8_sw64_coupled_stmatrix(s_p_a, p0)
            old_m_finite = tl.where(state_valid, old_m, 0.0)
            alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
            if USE_HOTLOOP_RECIP:
                beta = alpha * old_s * inv_s_new
                l_new = old_l * beta + tl.sum(e, axis=1) * inv_s_new
            else:
                beta = alpha * old_s / s_new
                l_new = old_l * beta + tl.sum(e, axis=1) / s_new
            state_m = tl.where(page_valid, m_new, old_m)
            state_s = tl.where(page_valid, s_new, old_s)
            state_l = tl.where(page_valid, l_new, old_l)
            beta = tl.where(page_valid, beta, 1.0)
            state_valid = state_valid | page_valid

            tl.store(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)), beta)
            tl.store(tle.gpu.local_ptr(s_state0_m, (state_idx,)), state_m)
            tl.store(tle.gpu.local_ptr(s_state0_s, (state_idx,)), state_s)
            tl.store(tle.gpu.local_ptr(s_state0_l, (state_idx,)), state_l)
            tl.store(
                tle.gpu.local_ptr(s_state0_valid, (state_idx,)),
                state_valid.to(tl.int32),
            )

            # CUDA publishes the completed online-softmax state at its last
            # shared write. Do not serialize WG1 softmax behind the unrelated
            # V repack that follows in WG0.
            if not MERGE_STATE_V:
                tle.gpu.barrier_arrive(state0_ready)

                # This loop excludes the final pair, so its even page is always a
                # complete logical page.  Match CUDA's compile-time steady-state
                # specialization and keep the masked tensor fallback in the
                # epilogue only.
            vtranspose_fp8_64x128(s_kc_a0, s_vt0_a, 0, True)
            vtranspose_fp8_64x128(s_kc_a1, s_vt0_a, DP // 2, True)
            vtranspose_fp8_64x128(s_kc_a2, s_vt1_a, 0, True)
            vtranspose_fp8_64x128(s_kc_a3, s_vt1_a, DP // 2, True)

            tle.gpu.barrier_arrive(v0_ready)

            # CUDA local-P wait point: finish current local PV, then launch
            # slot-A generation pair+1 content0/1 for p+2.
            acc_left *= beta[:, None]
            acc_left = tle.gpu.wgmma(s_p_a, s_vt0_a, acc_left, trans_b=True)
            # Slot A content was consumed by the V transpose; refill it while
            # the local PV runs.
            next_even_page = page + 2
            next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
            next_even_base = (next_even_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c0,
                [BK, K_CONTENT_TILE],
                [next_even_base, 0],
                barrier=k_content_full[0],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c1,
                [BK, K_CONTENT_TILE],
                [next_even_base, K_CONTENT_TILE],
                barrier=k_content_full[1],
            )
            acc_left = tle.gpu.wgmma_wait(0, acc_left)

            next_generation = pair + 1
            next_qk = tl.zeros((BH, BK), dtype=tl.float32)
            next_ks = tl.zeros((BK,), dtype=tl.float32)

            odd_page = page + 1
            tle.gpu.barrier_wait(v1_ready)
            beta1 = tl.load(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)))
            acc_left *= beta1[:, None]
            acc_left = tle.gpu.wgmma(s_p_b, s_vt0_b, acc_left, trans_b=True)

            # Keep the async rP0 chain inside one real-p+2 branch:
            # TLE permits loop-carried accumulators but not an async value
            # yielded through an intermediate scf.if.
            tle.gpu.barrier_wait(k_content_full[0], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(q_c0, k_a_c0, next_qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[1], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(q_c1, k_a_c1, next_qk, trans_b=True)
            tle.gpu.wgmma_wait(2, next_qk)
            tle.gpu.barrier_arrive(slot1_empty)

            # CUDA wait2 point starts p+3 content0/1 before p+2
            # phase-2.
            next_odd_page = odd_page + 2
            if next_odd_page < num_pages:
                next_odd_phys = tl.load(block_table + next_odd_page * stride_bt_pg)
                next_odd_base = (next_odd_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_b_c0,
                    [BK, K_CONTENT_TILE],
                    [next_odd_base, 0],
                    barrier=k_content_full[4],
                )
                tle.gpu.copy(
                    k_desc,
                    k_b_c1,
                    [BK, K_CONTENT_TILE],
                    [next_odd_base, K_CONTENT_TILE],
                    barrier=k_content_full[5],
                )

                # CUDA QK phase-2 completes p+2 in the current pair.
            tle.gpu.barrier_wait(k_content_full[2], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(q_c2, k_a_c2, next_qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[3], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(q_c3, k_a_c3, next_qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[0], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(s_qr, s_kr_a, next_qk, trans_b=True)
            next_qk = tle.gpu.wgmma_wait(0, next_qk)
            # The wait is global in hardware, but TLE also requires
            # the remote-P SSA value itself to pass through a wait.
            acc_left = tle.gpu.wgmma_wait(0, acc_left)

            tle.gpu.barrier_wait(k_scale_full[0], phaseIdx=next_generation)
            next_valid = next_even_page * PAGE_SIZE + offs_t < split_cache_seqlen
            next_ks_raw = tl.load(tle.gpu.local_ptr(s_ks_a_row, (offs_t,)))
            next_ks = (
                next_ks_raw if FULL_TAIL else tl.where(next_valid, next_ks_raw, 0.0)
            )

            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(state1_ready)
            state_m = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state1_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state1_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state1_valid, (state_idx,))) != 0

            # WG1 publishes this only after its remote P0/V0 wait0.
            tle.gpu.barrier_wait(slot0_empty)
            qk = next_qk
            ks = next_ks

            # CUDA-style epilogue: the final pair never creates a younger QK
            # accumulator, so every PV dependency is retired inside this tail.
        if num_pairs > 0:
            pair = steady_pairs
            page = pair * 2
            valid = page * PAGE_SIZE + offs_t < split_cache_seqlen
            valid_row = valid[None, :]
            if QUERY_COUNT > 1:
                valid_row = valid_row & (
                    page * PAGE_SIZE + offs_t[None, :] < query_end[:, None]
                )
            tail_ks_raw = tl.load(tle.gpu.local_ptr(s_ks_a_row, (offs_t,)))
            tail_ks = tail_ks_raw if FULL_TAIL else tl.where(valid, tail_ks_raw, 0.0)
            score = qk * qs[:, None] * tail_ks[None, :] * softmax_scale
            score_safe = (
                score
                if FULL_TAIL and QUERY_COUNT == 1
                else tl.where(valid_row, score, 0.0)
            )
            x = score_safe * TLE_LOG2E
            page_m = tl.max(
                (
                    x
                    if FULL_TAIL and QUERY_COUNT == 1
                    else tl.where(valid_row, x, TLE_NEG_INF)
                ),
                axis=1,
            )
            old_m = tl.where(state_valid, state_m, TLE_NEG_INF)
            old_s = tl.where(state_valid, state_s, 1.0)
            old_l = tl.where(state_valid, state_l, 0.0)
            m_new = tl.maximum(old_m, page_m)
            m_safe = tl.where(m_new == TLE_NEG_INF, 0.0, m_new)
            e = (
                tl.exp2(x - m_safe[:, None])
                if FULL_TAIL and QUERY_COUNT == 1
                else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
            )
            f = e * tail_ks[None, :]
            amax = tl.max(tl.abs(f), axis=1)
            s_new = tl.where(
                amax == 0.0,
                1.0,
                tl.maximum(amax, TLE_P_AMAX_FLOOR) / TLE_FP8_MAX,
            )
            page_valid = True if FULL_TAIL else page * PAGE_SIZE < split_cache_seqlen
            if QUERY_COUNT > 1:
                page_valid = page_valid & (page * PAGE_SIZE < query_end)
            inv_s_new = 1.0 / s_new
            p_new = f * inv_s_new[:, None]
            p0 = p_new
            publish_p_fp8_sw64_coupled_stmatrix(s_p_a, p0)
            old_m_finite = tl.where(state_valid, old_m, 0.0)
            alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
            beta = alpha * old_s * inv_s_new
            l_new = old_l * beta + tl.sum(e, axis=1) * inv_s_new
            state_m = tl.where(page_valid, m_new, old_m)
            state_s = tl.where(page_valid, s_new, old_s)
            state_l = tl.where(page_valid, l_new, old_l)
            beta = tl.where(page_valid, beta, 1.0)
            state_valid = state_valid | page_valid

            tl.store(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)), beta)
            tl.store(tle.gpu.local_ptr(s_state0_m, (state_idx,)), state_m)
            tl.store(tle.gpu.local_ptr(s_state0_s, (state_idx,)), state_s)
            tl.store(tle.gpu.local_ptr(s_state0_l, (state_idx,)), state_l)
            tl.store(
                tle.gpu.local_ptr(s_state0_valid, (state_idx,)),
                state_valid.to(tl.int32),
            )

            # Tail generation follows the same last-write publication rule.
            if not MERGE_STATE_V:
                tle.gpu.barrier_arrive(state0_ready)

                # Invalid probability columns are already exact FP8 zero after
                # the masked softmax above, so their V values cannot contribute
                # to PV.  Reuse the CUDA-aligned vectorized transpose for a
                # partial physical page instead of materializing a masked tensor
                # transpose in registers.
            if PAGE_GRAIN_TAIL_ZERO:
                if not FULL_TAIL:
                    valid_tokens = tl.minimum(split_cache_seqlen - page * PAGE_SIZE, BK)
                    if valid_tokens < BK:
                        zero_invalid_fp8_rows_sw128_x4(
                            s_kc_a0,
                            s_kc_a1,
                            s_kc_a2,
                            s_kc_a3,
                            valid_tokens,
                        )
                        tle.gpu.barrier_arrive(tail0_zero_ready, phaseIdx=0)
                        tle.gpu.barrier_wait(tail0_zero_ready, phaseIdx=0)
                vtranspose_fp8_64x128(s_kc_a0, s_vt0_a, 0, True)
                vtranspose_fp8_64x128(s_kc_a1, s_vt0_a, DP // 2, True)
                vtranspose_fp8_64x128(s_kc_a2, s_vt1_a, 0, True)
                vtranspose_fp8_64x128(s_kc_a3, s_vt1_a, DP // 2, True)
            elif FULL_TAIL or (page + 1) * PAGE_SIZE <= split_cache_seqlen:
                vtranspose_fp8_64x128(s_kc_a0, s_vt0_a, 0, True)
                vtranspose_fp8_64x128(s_kc_a1, s_vt0_a, DP // 2, True)
                vtranspose_fp8_64x128(s_kc_a2, s_vt1_a, 0, True)
                vtranspose_fp8_64x128(s_kc_a3, s_vt1_a, DP // 2, True)
            else:
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a0, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt0_a, (vt_c0_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a1, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt0_a, (vt_c1_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a2, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt1_a, (vt_c0_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )
                kc_tile = tl.load(
                    tle.gpu.local_ptr(s_kc_a3, (kv_rows_d128, kv_c0_cols))
                )
                kc_tile = tl.where(valid[:, None], kc_tile, tl.zeros_like(kc_tile))
                tl.store(
                    tle.gpu.local_ptr(s_vt1_a, (vt_c1_rows, vt_cols_d128)),
                    tl.trans(kc_tile),
                )

            tle.gpu.barrier_arrive(v0_ready)

            acc_left *= beta[:, None]
            acc_left = tle.gpu.wgmma(s_p_a, s_vt0_a, acc_left, trans_b=True)
            acc_left = tle.gpu.wgmma_wait(0, acc_left)

            odd_page = page + 1
            if odd_page < num_pages:
                tle.gpu.barrier_wait(v1_ready)
                beta1 = tl.load(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)))
                acc_left *= beta1[:, None]
                acc_left = tle.gpu.wgmma(s_p_b, s_vt0_b, acc_left, trans_b=True)
                acc_left = tle.gpu.wgmma_wait(0, acc_left)
                tle.gpu.barrier_arrive(slot1_empty)

                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(state1_ready)
                state_m = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state1_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state1_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state1_valid, (state_idx,))) != 0
                )

            tle.gpu.barrier_wait(slot0_empty)

            # CUDA-aligned programmatic dependency trigger.  Only the B>=4
            # coarse-combine specialization receives ENABLE_PDL=True.
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

        # Empty slots of the fixed split budget publish nothing.
        if has_output:
            offs_d = tl.arange(0, DP)
            l_div = tl.where(state_l > 0.0, state_l, 1.0)
            inv_l_div = 1.0 / l_div
            out_left = tl.where(
                state_valid[:, None], acc_left * inv_l_div[:, None], 0.0
            )
            if TMA_DIRECT:
                use_tma_output = is_single_split
            else:
                use_tma_output = USE_TMA_OUTPUT
            if use_tma_output:
                # The final K-rope read has retired before the loop exits, so its
                # existing 64x64 BF16 buffer can stage four output chunks without
                # adding shared memory. Each copy is a complete TMA S2G group; the
                # TLE store scheduler inserts the reuse-safe commit/wait sequence.
                out_left_lo, out_left_hi = tl.split(
                    tl.permute(tl.reshape(out_left, (BH, 2, DP // 2)), (0, 2, 1))
                )
                out_left_0, out_left_1 = tl.split(
                    tl.permute(tl.reshape(out_left_lo, (BH, 2, ROPE)), (0, 2, 1))
                )
                out_left_2, out_left_3 = tl.split(
                    tl.permute(tl.reshape(out_left_hi, (BH, 2, ROPE)), (0, 2, 1))
                )
                tile_rows = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, ROPE))
                tile_cols = tl.broadcast_to(tl.arange(0, ROPE)[None, :], (BH, ROPE))
                tl.store(
                    tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                    out_left_0.to(tl.bfloat16),
                )
                tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, 0])
                tl.store(
                    tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                    out_left_1.to(tl.bfloat16),
                )
                tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, ROPE])
                tl.store(
                    tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                    out_left_2.to(tl.bfloat16),
                )
                tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, 2 * ROPE])
                tl.store(
                    tle.gpu.local_ptr(s_kr_a, (tile_rows, tile_cols)),
                    out_left_3.to(tl.bfloat16),
                )
                tle.gpu.copy(s_kr_a, out_desc, [BH, ROPE], [row0, 3 * ROPE])
            elif DYNAMIC_DIRECT and is_single_split:
                # A row planned as one split needs no combine pass.
                tl.store(
                    final_out_ptr + offs_h[:, None] * stride_fo_h + offs_d[None, :],
                    out_left.to(tl.bfloat16),
                    mask=mask_h[:, None],
                )
            else:
                tl.store(
                    out_ptr + offs_h[:, None] * stride_po_h + offs_d[None, :],
                    out_left,
                    mask=mask_h[:, None],
                )
            lse_arg = state_l * state_s
            lse_ok = state_valid & (lse_arg > 0.0)
            lse2_value = tl.where(
                lse_ok,
                state_m + tl.log(tl.where(lse_arg > 0.0, lse_arg, 1.0)) * TLE_LOG2E,
                TLE_NEG_INF,
            )
            if DYNAMIC_DIRECT and is_single_split:
                # Merged head q * LSE_HEADS + h lands at LSE[b, h, q].
                tl.store(
                    final_lse_ptr
                    + (offs_h % LSE_HEADS) * stride_fl_h
                    + (offs_h // LSE_HEADS) * stride_fl_q,
                    lse2_value * TLE_LN2,
                    mask=mask_h,
                )
            else:
                tl.store(
                    lse2_ptr + offs_h * stride_pl_h,
                    lse2_value * TLE_LN2 if DIRECT_LSE else lse2_value,
                    mask=mask_h,
                )


if HAS_TLE:

    @triton.jit
    def fp8_mla_wg1(
        k_desc,
        kr_desc,
        ks_desc,
        out_desc,
        block_table,
        stride_bt_pg,
        row0,
        num_pages,
        q_ckv_full,
        q_rope_full,
        q_scale_full,
        k_content_full,
        k_rope_full,
        k_scale_full,
        state0_ready,
        state1_ready,
        p0_ready,
        p1_ready,
        v0_ready,
        v1_ready,
        slot0_empty,
        slot1_empty,
        tail1_zero_ready,
        s_q,
        s_qr,
        s_kc_a0,
        s_kc_a1,
        s_kc_a2,
        s_kc_a3,
        s_kr_a,
        s_kc_b0,
        s_kc_b1,
        s_kc_b2,
        s_kc_b3,
        s_kr_b,
        s_vt0_b,
        s_vt1_b,
        s_vt1_a,
        s_p_a,
        s_p_b,
        s_beta_a,
        s_beta_b,
        s_ks_a,
        s_ks_b,
        s_state0_m,
        s_state0_s,
        s_state0_l,
        s_state0_valid,
        s_state1_m,
        s_state1_s,
        s_state1_l,
        s_state1_valid,
        split_cache_seqlen,
        causal_cache_seqlen,
        out_ptr,
        stride_po_h,
        h_base,
        softmax_scale,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        DP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        QUERY_HEADS: tl.constexpr,
        QUERY_COUNT: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        KNOWN_NUM_PAGES: tl.constexpr,
        PRETRANSPOSE_V1: tl.constexpr,
        final_out_ptr,
        stride_fo_h,
        is_single_split,
        has_output,
        DYNAMIC_DIRECT: tl.constexpr,
        TMA_DIRECT: tl.constexpr,
    ):
        """WG1 with compile-time V-repacking and prefetch schedule."""
        if KNOWN_NUM_PAGES > 0:
            # This is a host-certified logical page count, not a token mask.
            tl.assume(num_pages == KNOWN_NUM_PAGES)
        s_state1_m_row = s_state1_m.slot(0)
        s_beta_a_row = s_beta_a.slot(0)
        s_beta_b_row = s_beta_b.slot(0)
        s_ks_b_row = s_ks_b.slot(0)
        tle.gpu.barrier_wait(q_ckv_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_rope_full, phaseIdx=0)
        tle.gpu.barrier_wait(q_scale_full, phaseIdx=0)
        offs_t = tl.arange(0, BK)
        offs_h = h_base + tl.arange(0, BH)
        mask_h = offs_h < HQ
        query_end = causal_cache_seqlen
        if QUERY_COUNT > 1:
            query_end = causal_cache_seqlen - (QUERY_COUNT - 1 - offs_h // QUERY_HEADS)
        state_idx = tl.arange(0, BH)
        qs = tl.load(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), volatile=True)
        acc_right = tl.zeros((BH, DP), dtype=tl.float32)
        state_m = tl.full((BH,), float("-inf"), tl.float32)
        state_s = tl.full((BH,), 1.0, tl.float32)
        state_l = tl.zeros((BH,), dtype=tl.float32)
        state_valid = tl.zeros((BH,), dtype=tl.int32) != 0
        q_rows_d128 = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, K_CONTENT_TILE))
        q_c0_cols = tl.broadcast_to(
            tl.arange(0, K_CONTENT_TILE)[None, :], (BH, K_CONTENT_TILE)
        )
        q_c1_cols = tl.broadcast_to(
            (K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c2_cols = tl.broadcast_to(
            (2 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c3_cols = tl.broadcast_to(
            (3 * K_CONTENT_TILE + tl.arange(0, K_CONTENT_TILE))[None, :],
            (BH, K_CONTENT_TILE),
        )
        q_c0 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c0_cols)))
        q_c1 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c1_cols)))
        q_c2 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c2_cols)))
        q_c3 = tl.load(tle.gpu.local_ptr(s_q, (q_rows_d128, q_c3_cols)))
        k_a_c2 = s_kc_a2
        k_a_c3 = s_kc_a3
        k_b_c0 = s_kc_b0
        k_b_c1 = s_kc_b1
        k_b_c2 = s_kc_b2
        k_b_c3 = s_kc_b3
        kv_rows_d128 = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DP // 2))
        kv_c0_cols = tl.broadcast_to(tl.arange(0, DP // 2)[None, :], (BK, DP // 2))
        vt_c0_rows = tl.broadcast_to(tl.arange(0, DP // 2)[:, None], (DP // 2, BK))
        vt_c1_rows = tl.broadcast_to(
            (DP // 2 + tl.arange(0, DP // 2))[:, None], (DP // 2, BK)
        )
        vt_cols = tl.arange(0, BK)
        # The masked transpose uses the same K permutation as coupled STSM.
        vt_cols = (
            (vt_cols & ~14)
            | ((vt_cols & 2) << 1)
            | ((vt_cols & 4) << 1)
            | ((vt_cols & 8) >> 2)
        )
        vt_cols_d128 = tl.broadcast_to(vt_cols[None, :], (DP // 2, BK))
        # WG1 completes generation zero for both slots. The writer groups use
        # disjoint slices and independent completion barriers.
        if num_pages > 0:
            first_phys = tl.load(block_table)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[2],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[3],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_a,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[0],
            )
            tle.gpu.copy(
                ks_desc,
                s_ks_a,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[0],
            )
        if num_pages > 1:
            first_phys = tl.load(block_table + stride_bt_pg)
            first_base = (first_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_b_c2,
                [BK, K_CONTENT_TILE],
                [first_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[6],
            )
            tle.gpu.copy(
                k_desc,
                k_b_c3,
                [BK, K_CONTENT_TILE],
                [first_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[7],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_b,
                [BK, ROPE],
                [first_base, 0],
                barrier=k_rope_full[1],
            )
            tle.gpu.copy(
                ks_desc,
                s_ks_b,
                [1, BK],
                [first_phys, 0],
                barrier=k_scale_full[1],
            )
            # Cold prime: page 1 QK, scale, and V become loop live-ins. No page-1
            # QK is repeated in pair zero.
        qk = tl.zeros((BH, BK), dtype=tl.float32)
        ks = tl.zeros((BK,), dtype=tl.float32)
        if num_pages > 1:
            tle.gpu.barrier_wait(k_content_full[4], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c0, k_b_c0, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[5], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c1, k_b_c1, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[6], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c2, k_b_c2, qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[7], phaseIdx=0)
            qk = tle.gpu.wgmma(q_c3, k_b_c3, qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=0)
            qk = tle.gpu.wgmma(s_qr, s_kr_b, qk, trans_b=True)
            qk = tle.gpu.wgmma_wait(0, qk)

            tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=0)
            prime_valid = PAGE_SIZE + offs_t < split_cache_seqlen
            ks_raw = tl.load(tle.gpu.local_ptr(s_ks_b_row, (offs_t,)))
            ks = ks_raw if FULL_TAIL else tl.where(prime_valid, ks_raw, 0.0)
        full_pairs = tl.maximum(num_pages // 2 - 1, 0)
        for pair in tl.range(full_pairs, disable_licm=True):
            even_page = pair * 2
            odd_page = even_page + 1
            if PRETRANSPOSE_V1:
                # V1 is independent of WG0's state payload.  Execute useful
                # transpose work while WG0 completes state0; keep publication after
                # P1 so the v1_ready payload/happens-before edge is unchanged.
                vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, True)
                vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, True)
                vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, True)
                vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, True)
            else:
                pass
            if MERGE_STATE_V:
                if PRETRANSPOSE_V1:
                    pass
                else:
                    # The odd-page V repack reads only
                    # this WG's already-waited K content (k_content_full[4..7]
                    # retired by the QK chain that produced the resident qk) and
                    # its prior-generation readers retired through slot1_empty
                    # (WG0 PV, waited last iteration) and this WG's own
                    # wgmma_wait.  It does not depend on WG0's incoming state, P,
                    # or V, so issue it before the merged completion wait and
                    # remove it from the wait->v1_ready critical path.  The
                    # v1_ready arrive below still follows every one of these
                    # shared writes in program order.
                    vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, True)
                    vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, True)
                    vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, True)
                    vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, True)
                    # The merged completion is intentionally later than the old
                    # state-only publication.  Hide part of that wait with the
                    # page-local score work, which depends only on the resident
                    # QK accumulator and scales, not on WG0's incoming state.
                    valid = tl.full((BK,), True, tl.int1)
                    valid_row = valid[None, :]
                    score = qk * qs[:, None] * ks[None, :] * softmax_scale
                    score_safe = score
                    x = score_safe * TLE_LOG2E
                    page_m = tl.max(
                        x,
                        axis=1,
                    )
                tle.gpu.barrier_wait(v0_ready)
            else:
                tle.gpu.barrier_wait(state0_ready)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0
            beta1 = tl.full((BH,), 1.0, tl.float32)
            if PRETRANSPOSE_V1:
                valid = tl.full((BK,), True, tl.int1)
                valid_row = valid[None, :]
                score = qk * qs[:, None] * ks[None, :] * softmax_scale
                score_safe = score
                x = score_safe * TLE_LOG2E
                page_m = tl.max(x, axis=1)
            else:
                # MERGE_STATE_V is constexpr, so this schedule retains only
                # the selected page-local chain after lowering.
                if not MERGE_STATE_V:
                    valid = tl.full((BK,), True, tl.int1)
                    valid_row = valid[None, :]
                    score = qk * qs[:, None] * ks[None, :] * softmax_scale
                    score_safe = score
                    x = score_safe * TLE_LOG2E
                    page_m = tl.max(
                        x,
                        axis=1,
                    )
            old_m = tl.where(state_valid, state_m, TLE_NEG_INF)
            old_s = tl.where(state_valid, state_s, 1.0)
            old_l = tl.where(state_valid, state_l, 0.0)
            m_new = tl.maximum(old_m, page_m)
            m_safe = tl.where(m_new == TLE_NEG_INF, 0.0, m_new)
            e = tl.exp2(x - m_safe[:, None])
            f = e * ks[None, :]
            amax = tl.max(tl.abs(f), axis=1)
            s_new = tl.where(
                amax == 0.0,
                1.0,
                tl.maximum(amax, TLE_P_AMAX_FLOOR) / TLE_FP8_MAX,
            )
            page_valid = True
            if USE_HOTLOOP_RECIP:
                inv_s_new = 1.0 / s_new
                p_scaled = f * inv_s_new[:, None]
            else:
                p_scaled = f / s_new[:, None]
            p_new = p_scaled
            p1 = p_new
            publish_p_fp8_sw64_coupled_stmatrix(s_p_b, p1)
            old_m_finite = tl.where(state_valid, old_m, 0.0)
            alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
            if USE_HOTLOOP_RECIP:
                beta1 = alpha * old_s * inv_s_new
                l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
            else:
                beta1 = alpha * old_s / s_new
                l_new = old_l * beta1 + tl.sum(e, axis=1) / s_new
            state_m = tl.where(page_valid, m_new, old_m)
            state_s = tl.where(page_valid, s_new, old_s)
            state_l = tl.where(page_valid, l_new, old_l)
            beta1 = tl.where(page_valid, beta1, 1.0)
            state_valid = state_valid | page_valid
            tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
            tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
            tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
            tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
            tl.store(
                tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                state_valid.to(tl.int32),
            )
            # Publish WG1 state before V repack/PV/next-QK, matching the
            # CUDA scale/state hand-off rather than delaying the consumer
            # behind unrelated work.
            if not MERGE_STATE_V:
                tle.gpu.barrier_arrive(state1_ready)
            if PRETRANSPOSE_V1:
                pass
            else:
                # full_pairs excludes the residual/tail pair.  The steady odd
                # page is therefore complete and can use CUDA's single
                # LDSM/PRMT/STSM path without a runtime fallback branch.
                # The merged-state specialization moves this repack before its
                # completion wait; all other specializations keep it here.
                if not MERGE_STATE_V:
                    vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, True)
                    vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, True)
                    vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, True)
                    vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, True)
            tle.gpu.barrier_arrive(v1_ready)
            # CUDA remote-P wait point for the current even page.
            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
            if PRETRANSPOSE_V1:
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
            else:
                pass
                # These K/RoPE/scale reads have retired; PV uses distinct P/V
                # buffers. Issue p+2 transfers now, then drain PV before release.
            next_even_page = even_page + 2
            next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
            next_even_base = (next_even_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_a_c2,
                [BK, K_CONTENT_TILE],
                [next_even_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[2],
            )
            tle.gpu.copy(
                k_desc,
                k_a_c3,
                [BK, K_CONTENT_TILE],
                [next_even_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[3],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_a,
                [BK, ROPE],
                [next_even_base, 0],
                barrier=k_rope_full[0],
            )
            tle.gpu.copy(
                ks_desc,
                s_ks_a,
                [1, BK],
                [next_even_phys, 0],
                barrier=k_scale_full[0],
            )
            if PRETRANSPOSE_V1:
                pass
            else:
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
            tle.gpu.barrier_arrive(slot0_empty)
            next_qk = tl.zeros((BH, BK), dtype=tl.float32)
            next_ks = tl.zeros((BK,), dtype=tl.float32)
            acc_right *= beta1[:, None]
            acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
            acc_right = tle.gpu.wgmma_wait(0, acc_right)

            next_odd_page = odd_page + 2
            next_generation = pair + 1
            next_odd_phys = tl.load(block_table + next_odd_page * stride_bt_pg)
            next_odd_base = (next_odd_phys * BK).to(tl.int32)
            tle.gpu.copy(
                k_desc,
                k_b_c2,
                [BK, K_CONTENT_TILE],
                [next_odd_base, 2 * K_CONTENT_TILE],
                barrier=k_content_full[6],
            )
            tle.gpu.copy(
                k_desc,
                k_b_c3,
                [BK, K_CONTENT_TILE],
                [next_odd_base, 3 * K_CONTENT_TILE],
                barrier=k_content_full[7],
            )
            tle.gpu.copy(
                kr_desc,
                s_kr_b,
                [BK, ROPE],
                [next_odd_base, 0],
                barrier=k_rope_full[1],
            )
            tle.gpu.copy(
                ks_desc,
                s_ks_b,
                [1, BK],
                [next_odd_phys, 0],
                barrier=k_scale_full[1],
            )

            # CUDA QK phase-1 completes p+3 in this pair.
            tle.gpu.barrier_wait(k_content_full[4], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(q_c0, k_b_c0, next_qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[5], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(q_c1, k_b_c1, next_qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[6], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(q_c2, k_b_c2, next_qk, trans_b=True)
            tle.gpu.barrier_wait(k_content_full[7], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(q_c3, k_b_c3, next_qk, trans_b=True)
            tle.gpu.barrier_wait(k_rope_full[1], phaseIdx=next_generation)
            next_qk = tle.gpu.wgmma(s_qr, s_kr_b, next_qk, trans_b=True)
            next_qk = tle.gpu.wgmma_wait(0, next_qk)

            tle.gpu.barrier_wait(slot1_empty)
            tle.gpu.barrier_wait(k_scale_full[1], phaseIdx=next_generation)
            next_valid = next_odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
            next_ks_raw = tl.load(tle.gpu.local_ptr(s_ks_b_row, (offs_t,)))
            next_ks = (
                next_ks_raw if FULL_TAIL else tl.where(next_valid, next_ks_raw, 0.0)
            )
            qk = next_qk
            ks = next_ks
            # CUDA-style WG1 epilogue. The first residual pair is either the last
            # full pair or the 3-page transition; an odd transition has one final
            # even-only pair after it.
        if num_pages > 0:
            pair = full_pairs
            even_page = pair * 2
            odd_page = even_page + 1
            if MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            else:
                tle.gpu.barrier_wait(state0_ready)
            state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
            state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
            state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
            state_valid = tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0
            beta1 = tl.full((BH,), 1.0, tl.float32)
            if odd_page < num_pages:
                valid = odd_page * PAGE_SIZE + offs_t < split_cache_seqlen
                valid_row = valid[None, :]
                if QUERY_COUNT > 1:
                    valid_row = valid_row & (
                        odd_page * PAGE_SIZE + offs_t[None, :] < query_end[:, None]
                    )
                tail_ks_raw = tl.load(tle.gpu.local_ptr(s_ks_b_row, (offs_t,)))
                tail_ks = (
                    tail_ks_raw if FULL_TAIL else tl.where(valid, tail_ks_raw, 0.0)
                )
                score = qk * qs[:, None] * tail_ks[None, :] * softmax_scale
                score_safe = (
                    score
                    if FULL_TAIL and QUERY_COUNT == 1
                    else tl.where(valid_row, score, 0.0)
                )
                x = score_safe * TLE_LOG2E
                page_m = tl.max(
                    (
                        x
                        if FULL_TAIL and QUERY_COUNT == 1
                        else tl.where(valid_row, x, TLE_NEG_INF)
                    ),
                    axis=1,
                )
                old_m = tl.where(state_valid, state_m, TLE_NEG_INF)
                old_s = tl.where(state_valid, state_s, 1.0)
                old_l = tl.where(state_valid, state_l, 0.0)
                m_new = tl.maximum(old_m, page_m)
                m_safe = tl.where(m_new == TLE_NEG_INF, 0.0, m_new)
                e = (
                    tl.exp2(x - m_safe[:, None])
                    if FULL_TAIL and QUERY_COUNT == 1
                    else tl.where(valid_row, tl.exp2(x - m_safe[:, None]), 0.0)
                )
                f = e * tail_ks[None, :]
                amax = tl.max(tl.abs(f), axis=1)
                s_new = tl.where(
                    amax == 0.0,
                    1.0,
                    tl.maximum(amax, TLE_P_AMAX_FLOOR) / TLE_FP8_MAX,
                )
                page_valid = (
                    True if FULL_TAIL else odd_page * PAGE_SIZE < split_cache_seqlen
                )
                if QUERY_COUNT > 1:
                    page_valid = page_valid & (odd_page * PAGE_SIZE < query_end)
                inv_s_new = 1.0 / s_new
                p_new = f * inv_s_new[:, None]
                p1 = p_new
                publish_p_fp8_sw64_coupled_stmatrix(s_p_b, p1)
                old_m_finite = tl.where(state_valid, old_m, 0.0)
                alpha = tl.where(state_valid, tl.exp2(old_m_finite - m_safe), 0.0)
                beta1 = alpha * old_s * inv_s_new
                l_new = old_l * beta1 + tl.sum(e, axis=1) * inv_s_new
                state_m = tl.where(page_valid, m_new, old_m)
                state_s = tl.where(page_valid, s_new, old_s)
                state_l = tl.where(page_valid, l_new, old_l)
                beta1 = tl.where(page_valid, beta1, 1.0)
                state_valid = state_valid | page_valid
                tl.store(tle.gpu.local_ptr(s_beta_b_row, (state_idx,)), beta1)
                tl.store(tle.gpu.local_ptr(s_state1_m_row, (state_idx,)), state_m)
                tl.store(tle.gpu.local_ptr(s_state1_s, (state_idx,)), state_s)
                tl.store(tle.gpu.local_ptr(s_state1_l, (state_idx,)), state_l)
                tl.store(
                    tle.gpu.local_ptr(s_state1_valid, (state_idx,)),
                    state_valid.to(tl.int32),
                )
                # Tail generation follows the same last-write publication rule.
                if not MERGE_STATE_V:
                    tle.gpu.barrier_arrive(state1_ready)
                if PRETRANSPOSE_V1:
                    if FULL_TAIL or (odd_page + 1) * PAGE_SIZE <= split_cache_seqlen:
                        vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, True)
                        vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, True)
                        vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, True)
                        vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, True)
                    else:
                        kc_tile = tl.load(
                            tle.gpu.local_ptr(s_kc_b0, (kv_rows_d128, kv_c0_cols))
                        )
                        kc_tile = tl.where(
                            valid[:, None], kc_tile, tl.zeros_like(kc_tile)
                        )
                        tl.store(
                            tle.gpu.local_ptr(s_vt0_b, (vt_c0_rows, vt_cols_d128)),
                            tl.trans(kc_tile),
                        )
                        kc_tile = tl.load(
                            tle.gpu.local_ptr(s_kc_b1, (kv_rows_d128, kv_c0_cols))
                        )
                        kc_tile = tl.where(
                            valid[:, None], kc_tile, tl.zeros_like(kc_tile)
                        )
                        tl.store(
                            tle.gpu.local_ptr(s_vt0_b, (vt_c1_rows, vt_cols_d128)),
                            tl.trans(kc_tile),
                        )
                        kc_tile = tl.load(
                            tle.gpu.local_ptr(s_kc_b2, (kv_rows_d128, kv_c0_cols))
                        )
                        kc_tile = tl.where(
                            valid[:, None], kc_tile, tl.zeros_like(kc_tile)
                        )
                        tl.store(
                            tle.gpu.local_ptr(s_vt1_b, (vt_c0_rows, vt_cols_d128)),
                            tl.trans(kc_tile),
                        )
                        kc_tile = tl.load(
                            tle.gpu.local_ptr(s_kc_b3, (kv_rows_d128, kv_c0_cols))
                        )
                        kc_tile = tl.where(
                            valid[:, None], kc_tile, tl.zeros_like(kc_tile)
                        )
                        tl.store(
                            tle.gpu.local_ptr(s_vt1_b, (vt_c1_rows, vt_cols_d128)),
                            tl.trans(kc_tile),
                        )
                else:
                    # As on the even-page owner, invalid P columns are exact zero,
                    # so a masked V transpose is unnecessary for PV correctness.
                    if PAGE_GRAIN_TAIL_ZERO:
                        if not FULL_TAIL:
                            valid_tokens = tl.minimum(
                                split_cache_seqlen - odd_page * PAGE_SIZE,
                                BK,
                            )
                            valid_tokens = tl.maximum(valid_tokens, 0)
                            if valid_tokens < BK:
                                zero_invalid_fp8_rows_sw128_x4(
                                    s_kc_b0,
                                    s_kc_b1,
                                    s_kc_b2,
                                    s_kc_b3,
                                    valid_tokens,
                                )
                                tle.gpu.barrier_arrive(tail1_zero_ready, phaseIdx=0)
                                tle.gpu.barrier_wait(tail1_zero_ready, phaseIdx=0)
                        vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, True)
                        vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, True)
                        vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, True)
                        vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, True)
                    elif FULL_TAIL or (odd_page + 1) * PAGE_SIZE <= split_cache_seqlen:
                        vtranspose_fp8_64x128(s_kc_b0, s_vt0_b, 0, True)
                        vtranspose_fp8_64x128(s_kc_b1, s_vt0_b, DP // 2, True)
                        vtranspose_fp8_64x128(s_kc_b2, s_vt1_b, 0, True)
                        vtranspose_fp8_64x128(s_kc_b3, s_vt1_b, DP // 2, True)
                    else:
                        kc_tile = tl.load(
                            tle.gpu.local_ptr(s_kc_b0, (kv_rows_d128, kv_c0_cols))
                        )
                        kc_tile = tl.where(
                            valid[:, None], kc_tile, tl.zeros_like(kc_tile)
                        )
                        tl.store(
                            tle.gpu.local_ptr(s_vt0_b, (vt_c0_rows, vt_cols_d128)),
                            tl.trans(kc_tile),
                        )
                        kc_tile = tl.load(
                            tle.gpu.local_ptr(s_kc_b1, (kv_rows_d128, kv_c0_cols))
                        )
                        kc_tile = tl.where(
                            valid[:, None], kc_tile, tl.zeros_like(kc_tile)
                        )
                        tl.store(
                            tle.gpu.local_ptr(s_vt0_b, (vt_c1_rows, vt_cols_d128)),
                            tl.trans(kc_tile),
                        )
                        kc_tile = tl.load(
                            tle.gpu.local_ptr(s_kc_b2, (kv_rows_d128, kv_c0_cols))
                        )
                        kc_tile = tl.where(
                            valid[:, None], kc_tile, tl.zeros_like(kc_tile)
                        )
                        tl.store(
                            tle.gpu.local_ptr(s_vt1_b, (vt_c0_rows, vt_cols_d128)),
                            tl.trans(kc_tile),
                        )
                        kc_tile = tl.load(
                            tle.gpu.local_ptr(s_kc_b3, (kv_rows_d128, kv_c0_cols))
                        )
                        kc_tile = tl.where(
                            valid[:, None], kc_tile, tl.zeros_like(kc_tile)
                        )
                        tl.store(
                            tle.gpu.local_ptr(s_vt1_b, (vt_c1_rows, vt_cols_d128)),
                            tl.trans(kc_tile),
                        )
                tle.gpu.barrier_arrive(v1_ready)
            else:
                pass
            if not MERGE_STATE_V:
                tle.gpu.barrier_wait(v0_ready)
            beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
            acc_right *= beta0[:, None]
            acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
            acc_right = tle.gpu.wgmma_wait(0, acc_right)
            next_even_page = even_page + 2
            if next_even_page < num_pages:
                next_even_phys = tl.load(block_table + next_even_page * stride_bt_pg)
                next_even_base = (next_even_phys * BK).to(tl.int32)
                tle.gpu.copy(
                    k_desc,
                    k_a_c2,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 2 * K_CONTENT_TILE],
                    barrier=k_content_full[2],
                )
                tle.gpu.copy(
                    k_desc,
                    k_a_c3,
                    [BK, K_CONTENT_TILE],
                    [next_even_base, 3 * K_CONTENT_TILE],
                    barrier=k_content_full[3],
                )
                tle.gpu.copy(
                    kr_desc,
                    s_kr_a,
                    [BK, ROPE],
                    [next_even_base, 0],
                    barrier=k_rope_full[0],
                )
                tle.gpu.copy(
                    ks_desc,
                    s_ks_a,
                    [1, BK],
                    [next_even_phys, 0],
                    barrier=k_scale_full[0],
                )
            tle.gpu.barrier_arrive(slot0_empty)
            if odd_page < num_pages:
                acc_right *= beta1[:, None]
                acc_right = tle.gpu.wgmma(s_p_b, s_vt1_b, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_wait(slot1_empty)
            if next_even_page < num_pages:
                if MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready)
                else:
                    tle.gpu.barrier_wait(state0_ready)
                state_m = tl.load(tle.gpu.local_ptr(s_state0_m, (state_idx,)))
                state_s = tl.load(tle.gpu.local_ptr(s_state0_s, (state_idx,)))
                state_l = tl.load(tle.gpu.local_ptr(s_state0_l, (state_idx,)))
                state_valid = (
                    tl.load(tle.gpu.local_ptr(s_state0_valid, (state_idx,))) != 0
                )
                if not MERGE_STATE_V:
                    tle.gpu.barrier_wait(v0_ready)
                beta0 = tl.load(tle.gpu.local_ptr(s_beta_a_row, (state_idx,)))
                acc_right *= beta0[:, None]
                acc_right = tle.gpu.wgmma(s_p_a, s_vt1_a, acc_right, trans_b=True)
                acc_right = tle.gpu.wgmma_wait(0, acc_right)
                tle.gpu.barrier_arrive(slot0_empty)
        else:
            pass
        # Empty slots of the fixed split budget publish nothing.
        if has_output:
            offs_d = tl.arange(0, DP)
            l_div = tl.where(state_l > 0.0, state_l, 1.0)
            inv_l_div = 1.0 / l_div
            out_right = tl.where(
                state_valid[:, None], acc_right * inv_l_div[:, None], 0.0
            )
            if TMA_DIRECT:
                use_tma_output = is_single_split
            else:
                use_tma_output = USE_TMA_OUTPUT
            if use_tma_output:
                out_right_lo, out_right_hi = tl.split(
                    tl.permute(tl.reshape(out_right, (BH, 2, DP // 2)), (0, 2, 1))
                )
                out_right_0, out_right_1 = tl.split(
                    tl.permute(tl.reshape(out_right_lo, (BH, 2, ROPE)), (0, 2, 1))
                )
                out_right_2, out_right_3 = tl.split(
                    tl.permute(tl.reshape(out_right_hi, (BH, 2, ROPE)), (0, 2, 1))
                )
                tile_rows = tl.broadcast_to(tl.arange(0, BH)[:, None], (BH, ROPE))
                tile_cols = tl.broadcast_to(tl.arange(0, ROPE)[None, :], (BH, ROPE))
                tl.store(
                    tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                    out_right_0.to(tl.bfloat16),
                )
                tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP])
                tl.store(
                    tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                    out_right_1.to(tl.bfloat16),
                )
                tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + ROPE])
                tl.store(
                    tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                    out_right_2.to(tl.bfloat16),
                )
                tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 2 * ROPE])
                tl.store(
                    tle.gpu.local_ptr(s_kr_b, (tile_rows, tile_cols)),
                    out_right_3.to(tl.bfloat16),
                )
                tle.gpu.copy(s_kr_b, out_desc, [BH, ROPE], [row0, DP + 3 * ROPE])
            elif DYNAMIC_DIRECT and is_single_split:
                tl.store(
                    final_out_ptr
                    + offs_h[:, None] * stride_fo_h
                    + DP
                    + offs_d[None, :],
                    out_right.to(tl.bfloat16),
                    mask=mask_h[:, None],
                )
            else:
                tl.store(
                    out_ptr + offs_h[:, None] * stride_po_h + DP + offs_d[None, :],
                    out_right,
                    mask=mask_h[:, None],
                )


if HAS_TLE:

    @triton.jit
    def fp8_dense_mla_splitk_partial(
        qc_ptr,
        qr_ptr,
        qs_ptr,
        kc_ptr,
        kr_ptr,
        ks_ptr,
        block_table,
        cache_seqlens,
        split_batch_ptr,
        split_page_begin_ptr,
        split_num_pages_ptr,
        split_order_ptr,
        partial_out_ptr,
        partial_lse2_ptr,
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        kr_desc,
        ks_desc,
        stride_qc_b: tl.constexpr,
        stride_qc_h: tl.constexpr,
        stride_qr_b: tl.constexpr,
        stride_qr_h: tl.constexpr,
        stride_qs_b: tl.constexpr,
        stride_qs_h: tl.constexpr,
        stride_kc_blk: tl.constexpr,
        stride_kc_pg: tl.constexpr,
        stride_kr_blk: tl.constexpr,
        stride_kr_pg: tl.constexpr,
        stride_ks_blk: tl.constexpr,
        stride_ks_pg: tl.constexpr,
        stride_bt_b: tl.constexpr,
        stride_bt_pg: tl.constexpr,
        stride_seqlen: tl.constexpr,
        stride_split_batch: tl.constexpr,
        stride_split_begin: tl.constexpr,
        stride_split_num_pages: tl.constexpr,
        stride_po_split: tl.constexpr,
        stride_po_h: tl.constexpr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_CKV_BYTES: tl.constexpr,
        Q_ROPE_BYTES: tl.constexpr,
        Q_SCALE_BYTES: tl.constexpr,
        K_CONTENT_TILE_BYTES: tl.constexpr,
        K_ROPE_BYTES: tl.constexpr,
        K_SCALE_BYTES: tl.constexpr,
        CKV: tl.constexpr,
        ROPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        HQ: tl.constexpr,
        RH: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DP: tl.constexpr,
        USE_HOTLOOP_RECIP: tl.constexpr,
        FULL_TAIL: tl.constexpr,
        QUERY_HEADS: tl.constexpr,
        QUERY_COUNT: tl.constexpr,
        PAGE_GRAIN_TAIL_ZERO: tl.constexpr,
        MERGE_STATE_V: tl.constexpr,
        USE_TMA_OUTPUT: tl.constexpr,
        FIXED_NUM_PAGES: tl.constexpr,
        DIRECT_LSE: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
        PRETRANSPOSE_V1: tl.constexpr,
        row_prefix_ptr=None,
        final_out_ptr=None,
        final_lse_ptr=None,
        stride_fo_b=0,
        stride_fo_h=0,
        stride_fl_b=0,
        stride_fl_h=0,
        stride_fl_q=0,
        DYNAMIC_DIRECT: tl.constexpr = False,
        TMA_DIRECT: tl.constexpr = False,
        LSE_HEADS: tl.constexpr = 1,
    ):
        """One strict-2WG CTA per (split, head block)."""
        pid = tl.program_id(0)
        global_split = tl.load(split_order_ptr + pid // RH)
        h_base = (pid % RH) * BH
        global_split64 = global_split.to(tl.int64)
        batch_idx = tl.load(split_batch_ptr + global_split64 * stride_split_batch)
        batch_idx64 = batch_idx.to(tl.int64)
        page_begin = tl.load(split_page_begin_ptr + global_split64 * stride_split_begin)
        split_num_pages_runtime = tl.load(
            split_num_pages_ptr + global_split64 * stride_split_num_pages
        )
        split_num_pages = (
            FIXED_NUM_PAGES
            if USE_TMA_OUTPUT and FIXED_NUM_PAGES > 0 and FIXED_NUM_PAGES <= 10
            else split_num_pages_runtime
        )
        page_end = page_begin + split_num_pages
        full_cache_seqlen = tl.load(cache_seqlens + batch_idx64 * stride_seqlen)
        token_begin = page_begin * PAGE_SIZE
        token_end = tl.minimum(page_end * PAGE_SIZE, full_cache_seqlen)
        split_cache_seqlen = tl.maximum(token_end - token_begin, 0)
        causal_cache_seqlen = full_cache_seqlen - token_begin
        if not FULL_TAIL:
            # Capacity pages beyond the actual tail may contain NaNs.
            split_num_pages = tl.cdiv(split_cache_seqlen, PAGE_SIZE)

        block_table_ptr = (
            block_table
            + batch_idx64 * stride_bt_b
            + page_begin.to(tl.int64) * stride_bt_pg
        )
        out_split_ptr = partial_out_ptr + global_split64 * stride_po_split
        lse2_split_ptr = partial_lse2_ptr + global_split64 * stride_pl_split
        if DYNAMIC_DIRECT:
            # Unused slots of the fixed split budget carry page_begin=-1 and
            # zero pages. They cannot exit early: worker warps must still join.
            has_output = page_begin >= 0
            is_single_split = (
                tl.load(row_prefix_ptr + batch_idx64 + 1)
                - tl.load(row_prefix_ptr + batch_idx64)
            ) == 1
            final_out_row = final_out_ptr + batch_idx64 * stride_fo_b
            final_lse_row = final_lse_ptr + batch_idx64 * stride_fl_b
        else:
            has_output = True
            is_single_split = False
            final_out_row = out_split_ptr
            final_lse_row = lse2_split_ptr

        s_q = tle.gpu.alloc([BH, CKV], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_qr = tle.gpu.alloc([BH, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)

        s_kc_a0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_a3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_a = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_a = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_kc_b0 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b1 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b2 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kc_b3 = tle.gpu.alloc(
            [BK, K_CONTENT_TILE], dtype=tl.float8e4nv, scope=tle.gpu.smem
        )
        s_kr_b = tle.gpu.alloc([BK, ROPE], dtype=tl.bfloat16, scope=tle.gpu.smem)
        s_vt0_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_vt1_b = tle.gpu.alloc([DP, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)

        s_p_a = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_p_b = tle.gpu.alloc([BH, BK], dtype=tl.float8e4nv, scope=tle.gpu.smem)
        s_beta_a = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_beta_b = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        # K scales land in their own buffers so their loads need not wait
        # for the beta handoff that reuses s_beta_*.
        s_ks_a = tle.gpu.alloc([1, BK], dtype=tl.float32, scope=tle.gpu.smem)
        s_ks_b = tle.gpu.alloc([1, BK], dtype=tl.float32, scope=tle.gpu.smem)

        s_state0_m = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state0_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)
        s_state1_m = tle.gpu.alloc([1, BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_s = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_l = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem)
        s_state1_valid = tle.gpu.alloc([BH], dtype=tl.int32, scope=tle.gpu.smem)

        # One TMA copy == one completion-barrier generation.
        q_ckv_full = tle.gpu.alloc_barrier(expect_bytes=Q_CKV_BYTES)
        q_rope_full = tle.gpu.alloc_barrier(expect_bytes=Q_ROPE_BYTES)
        q_scale_full = tle.gpu.alloc_barrier(expect_bytes=Q_SCALE_BYTES)
        k_content_full = tle.gpu.alloc_barriers(8, expect_bytes=K_CONTENT_TILE_BYTES)
        k_rope_full = tle.gpu.alloc_barriers(2, expect_bytes=K_ROPE_BYTES)
        k_scale_full = tle.gpu.alloc_barriers(2, expect_bytes=K_SCALE_BYTES)

        # Cross-WG handoffs use named barriers: 128 producer arrivals plus
        # 128 consumer waiters complete each handshake. Per-WG tail-zero
        # synchronization retains its separate phaseful mbarriers.
        control_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=1)
        handoff_barriers = tle.gpu.alloc_barriers(num_barriers=8, arrive_count=256)
        state0_ready = handoff_barriers[0]
        state1_ready = handoff_barriers[1]
        # P is stored before V repack.  Publishing v*_ready after the repack
        # therefore certifies both P and V visibility to the remote PV owner.
        p0_ready = handoff_barriers[2]
        p1_ready = handoff_barriers[3]
        v0_ready = handoff_barriers[2]
        v1_ready = handoff_barriers[3]
        slot0_empty = handoff_barriers[4]
        slot1_empty = handoff_barriers[5]

        # CUDA fill_oob_V publishes shared zeros before the LDSM transpose.
        # Reuse one previously idle control mbarrier per compute warp-group;
        # each has one fixed elected writer and is used only once per CTA.
        tail0_zero_ready = control_barriers[6]
        tail1_zero_ready = control_barriers[7]

        row0 = (batch_idx * HQ + h_base).to(tl.int32)

        # Barrier initialization still owns temporary shared storage that the
        # warp-specialize captures reuse. A CTA barrier (not an mbarrier that
        # could itself be overwritten while being polled) orders the two.
        tl.debug_barrier()
        tle.gpu.warp_specialize(
            [
                (
                    fp8_mla_wg0,
                    (
                        q_desc,
                        qr_desc,
                        qs_desc,
                        out_desc,
                        k_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail0_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_a,
                        s_vt0_a,
                        s_vt1_a,
                        s_vt0_b,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_ks_a,
                        s_ks_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        causal_cache_seqlen,
                        out_split_ptr,
                        lse2_split_ptr,
                        stride_po_h,
                        stride_pl_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        QUERY_HEADS,
                        QUERY_COUNT,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        ENABLE_PDL,
                        USE_TMA_OUTPUT,
                        DIRECT_LSE,
                        FIXED_NUM_PAGES,
                        final_out_row,
                        stride_fo_h,
                        final_lse_row,
                        stride_fl_h,
                        stride_fl_q,
                        is_single_split,
                        has_output,
                        DYNAMIC_DIRECT,
                        TMA_DIRECT,
                        LSE_HEADS,
                    ),
                ),
                (
                    fp8_mla_wg1,
                    (
                        k_desc,
                        kr_desc,
                        ks_desc,
                        out_desc,
                        block_table_ptr,
                        stride_bt_pg,
                        row0,
                        split_num_pages,
                        q_ckv_full,
                        q_rope_full,
                        q_scale_full,
                        k_content_full,
                        k_rope_full,
                        k_scale_full,
                        state0_ready,
                        state1_ready,
                        p0_ready,
                        p1_ready,
                        v0_ready,
                        v1_ready,
                        slot0_empty,
                        slot1_empty,
                        tail1_zero_ready,
                        s_q,
                        s_qr,
                        s_kc_a0,
                        s_kc_a1,
                        s_kc_a2,
                        s_kc_a3,
                        s_kr_a,
                        s_kc_b0,
                        s_kc_b1,
                        s_kc_b2,
                        s_kc_b3,
                        s_kr_b,
                        s_vt0_b,
                        s_vt1_b,
                        s_vt1_a,
                        s_p_a,
                        s_p_b,
                        s_beta_a,
                        s_beta_b,
                        s_ks_a,
                        s_ks_b,
                        s_state0_m,
                        s_state0_s,
                        s_state0_l,
                        s_state0_valid,
                        s_state1_m,
                        s_state1_s,
                        s_state1_l,
                        s_state1_valid,
                        split_cache_seqlen,
                        causal_cache_seqlen,
                        out_split_ptr,
                        stride_po_h,
                        h_base,
                        softmax_scale,
                        CKV,
                        ROPE,
                        BK,
                        BH,
                        HQ,
                        DP,
                        PAGE_SIZE,
                        USE_HOTLOOP_RECIP,
                        FULL_TAIL,
                        QUERY_HEADS,
                        QUERY_COUNT,
                        PAGE_GRAIN_TAIL_ZERO,
                        MERGE_STATE_V,
                        USE_TMA_OUTPUT,
                        FIXED_NUM_PAGES,
                        PRETRANSPOSE_V1,
                        final_out_row,
                        stride_fo_h,
                        is_single_split,
                        has_output,
                        DYNAMIC_DIRECT,
                        TMA_DIRECT,
                    ),
                ),
            ],
            [4],
            [255],
        )


if HAS_TLE:

    @triton.jit
    def triton_fp8_splitk_combine_kernel(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SKIP_SINGLE_SPLIT: tl.constexpr = False,
        stride_lse_q=0,
        LSE_HEADS: tl.constexpr = 0,
    ):
        row = tl.program_id(0)
        d_block = tl.program_id(1)
        batch_idx = row // HQ
        head_idx = row % HQ
        split_begin = tl.load(num_splits_ptr + batch_idx * stride_ns)
        split_end = tl.load(num_splits_ptr + (batch_idx + 1) * stride_ns)
        split_count = split_end - split_begin
        # Single-split rows were written by the partial kernel.
        is_skipped = SKIP_SINGLE_SPLIT and split_count == 1
        split_count = tl.where(is_skipped, 0, split_count)

        split_lanes = tl.arange(0, BLOCK_SPLITS)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < DV

        max_lse2 = TLE_NEG_INF
        for split_base in tl.range(0, split_count, BLOCK_SPLITS):
            local_split = split_base + split_lanes
            mask_split = local_split < split_count
            global_split = split_begin + local_split
            local_lse2 = tl.load(
                partial_lse2_ptr
                + global_split * stride_pl_split
                + head_idx * stride_pl_h,
                mask=mask_split,
                other=TLE_NEG_INF,
            )
            finite_lse = (
                mask_split & (local_lse2 > TLE_NEG_INF) & (local_lse2 < TLE_POS_INF)
            )
            local_lse2 = tl.where(finite_lse, local_lse2, TLE_NEG_INF)
            max_lse2 = tl.maximum(max_lse2, tl.max(local_lse2, axis=0))

        finite_max = max_lse2 != TLE_NEG_INF
        safe_max = tl.where(finite_max, max_lse2, 0.0)
        denom = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for split_base in tl.range(0, split_count, BLOCK_SPLITS):
            local_split = split_base + split_lanes
            mask_split = local_split < split_count
            global_split = split_begin + local_split
            local_lse2 = tl.load(
                partial_lse2_ptr
                + global_split * stride_pl_split
                + head_idx * stride_pl_h,
                mask=mask_split,
                other=TLE_NEG_INF,
            )
            finite_lse = (
                mask_split & (local_lse2 > TLE_NEG_INF) & (local_lse2 < TLE_POS_INF)
            )
            local_lse2 = tl.where(finite_lse, local_lse2, TLE_NEG_INF)
            weights = tl.where(
                finite_lse,
                tl.exp2(local_lse2 - safe_max),
                0.0,
            )
            partial = tl.load(
                partial_out_ptr
                + global_split[:, None] * stride_po_split
                + head_idx * stride_po_h
                + offs_d[None, :],
                mask=mask_split[:, None] & mask_d[None, :],
                other=0.0,
            )
            partial = tl.where(finite_lse[:, None], partial, 0.0)
            denom += tl.sum(weights, axis=0)
            acc += tl.sum(partial * weights[:, None], axis=0)

        valid = finite_max & (denom > 0.0)
        safe_denom = tl.where(valid, denom, 1.0)
        result = tl.where(valid, acc / safe_denom, 0.0)
        tl.store(
            out_ptr + batch_idx * stride_out_b + head_idx * stride_out_h + offs_d,
            result,
            mask=mask_d & ~is_skipped,
        )

        global_lse = tl.where(
            valid,
            (safe_max + tl.log(safe_denom) * TLE_LOG2E) * TLE_LN2,
            TLE_NEG_INF,
        )
        tl.store(
            lse_ptr
            + batch_idx * stride_lse_b
            + (
                (head_idx % LSE_HEADS) * stride_lse_h
                + (head_idx // LSE_HEADS) * stride_lse_q
                if LSE_HEADS > 0
                else head_idx * stride_lse_h
            ),
            global_lse,
            mask=(d_block == 0) & ~is_skipped,
        )


if HAS_TLE:

    @triton.jit
    def triton_fp8_coarse_combine_kernel(
        partial_out_ptr,
        partial_lse2_ptr,
        num_splits_ptr,
        out_ptr,
        lse_ptr,
        stride_po_split,
        stride_po_h,
        stride_pl_split,
        stride_pl_h,
        stride_ns,
        stride_out_b,
        stride_out_h,
        stride_lse_b,
        stride_lse_h,
        HQ: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
        SKIP_SINGLE_SPLIT: tl.constexpr = False,
        stride_lse_q=0,
        LSE_HEADS: tl.constexpr = 0,
    ):
        """CUDA-aligned combine: one warp owns one row, eight rows per CTA."""
        # A PDL consumer may become resident before the partial grid retires.
        # No workspace or split metadata read is legal before this wait.
        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()

        batch_idx = tl.program_id(0)
        row_block = tl.program_id(1)
        heads = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        mask_h = heads < HQ

        split_begin = tl.load(num_splits_ptr + batch_idx * stride_ns)
        split_end = tl.load(num_splits_ptr + (batch_idx + 1) * stride_ns)
        split_count = split_end - split_begin
        # Single-split rows were written by the partial kernel.
        is_skipped = SKIP_SINGLE_SPLIT and split_count == 1
        split_count = tl.where(is_skipped, 0, split_count)

        split_lanes = tl.arange(0, BLOCK_SPLITS)
        max_lse2 = tl.full((BLOCK_ROWS,), TLE_NEG_INF, tl.float32)
        for split_base in tl.range(0, split_count, BLOCK_SPLITS):
            local_splits = split_base + split_lanes
            mask_split = local_splits < split_count
            global_splits = split_begin + local_splits
            local_lse2 = tl.load(
                partial_lse2_ptr
                + global_splits[None, :] * stride_pl_split
                + heads[:, None] * stride_pl_h,
                mask=mask_h[:, None] & mask_split[None, :],
                other=TLE_NEG_INF,
            )
            finite_lse = (
                mask_h[:, None]
                & mask_split[None, :]
                & (local_lse2 > TLE_NEG_INF)
                & (local_lse2 < TLE_POS_INF)
            )
            local_lse2 = tl.where(finite_lse, local_lse2, TLE_NEG_INF)
            max_lse2 = tl.maximum(max_lse2, tl.max(local_lse2, axis=1))

        finite_max = mask_h & (max_lse2 != TLE_NEG_INF)
        safe_max = tl.where(finite_max, max_lse2, 0.0)
        denom = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
        offs_d = tl.arange(0, DV)
        acc = tl.zeros((BLOCK_ROWS, DV), dtype=tl.float32)

        # Match CUDA split-order accumulation. One warp owns one logical row
        # and each lane owns DV/32 output columns.
        for local_split in tl.range(0, split_count):
            global_split = split_begin + local_split
            local_lse2 = tl.load(
                partial_lse2_ptr + global_split * stride_pl_split + heads * stride_pl_h,
                mask=mask_h,
                other=TLE_NEG_INF,
            )
            finite_lse = (
                mask_h & (local_lse2 > TLE_NEG_INF) & (local_lse2 < TLE_POS_INF)
            )
            weights = tl.where(
                finite_lse,
                tl.exp2(local_lse2 - safe_max),
                0.0,
            )
            partial = tl.load(
                partial_out_ptr
                + global_split * stride_po_split
                + heads[:, None] * stride_po_h
                + offs_d[None, :],
                mask=mask_h[:, None],
                other=0.0,
            )
            denom += weights
            acc += partial * weights[:, None]

        valid = finite_max & (denom > 0.0)
        safe_denom = tl.where(valid, denom, 1.0)
        result = tl.where(valid[:, None], acc / safe_denom[:, None], 0.0)
        tl.store(
            out_ptr
            + batch_idx * stride_out_b
            + heads[:, None] * stride_out_h
            + offs_d[None, :],
            result,
            mask=mask_h[:, None] & ~is_skipped,
        )

        global_lse = tl.where(
            valid,
            (safe_max + tl.log(safe_denom) * TLE_LOG2E) * TLE_LN2,
            TLE_NEG_INF,
        )
        tl.store(
            lse_ptr
            + batch_idx * stride_lse_b
            + (
                (heads % LSE_HEADS) * stride_lse_h + (heads // LSE_HEADS) * stride_lse_q
                if LSE_HEADS > 0
                else heads * stride_lse_h
            ),
            global_lse,
            mask=mask_h & ~is_skipped,
        )


if HAS_TLE:

    @triton.jit
    def triton_fp8_single_split_lse_finalize_kernel(
        partial_lse2_ptr,
        lse_ptr,
        stride_pl_split: tl.constexpr,
        stride_pl_h: tl.constexpr,
        stride_lse_b: tl.constexpr,
        stride_lse_h: tl.constexpr,
        HQ: tl.constexpr,
        TOTAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Convert one-split log2 LSE to the public natural-log convention."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        batch_idx = offsets // HQ
        head_idx = offsets - batch_idx * HQ
        lse2 = tl.load(
            partial_lse2_ptr
            + batch_idx.to(tl.int64) * stride_pl_split
            + head_idx.to(tl.int64) * stride_pl_h,
            mask=mask,
            other=0.0,
        )
        tl.store(
            lse_ptr
            + batch_idx.to(tl.int64) * stride_lse_b
            + head_idx.to(tl.int64) * stride_lse_h,
            lse2 * TLE_LN2,
            mask=mask,
        )


def tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError:
        return -1


def host_certificate_lengths(
    value,
    label: str,
    *,
    batch_size: Optional[int] = None,
):
    """Parse the host-only vectors used by the prepared decode contract."""
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be host integers, not a Tensor")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be an iterable of host integers")
    try:
        iterator = iter(value)
    except TypeError:
        lengths = (int(value),)
    else:
        lengths = tuple(int(item) for item in iterator)
    if not lengths:
        raise ValueError(f"{label} must not be empty")
    if batch_size is not None and len(lengths) != batch_size:
        raise ValueError(f"{label} must have {batch_size} entries")
    if any(length <= 0 or length > MAX_SEQUENCE_LENGTH for length in lengths):
        raise ValueError(f"{label} entries must be in [1, {MAX_SEQUENCE_LENGTH}]")
    return lengths


def length_page_state(lengths, pages_per_split: int):
    pages = tuple(math.ceil(int(length) / PAGE_SIZE) for length in lengths)
    splits = tuple(math.ceil(page_count / pages_per_split) for page_count in pages)
    return pages, splits


def adaptive_fixed_pages(max_pages: int) -> int:
    safety_splits = math.ceil(max_pages / 16)
    required_pages = math.ceil(max_pages / safety_splits)
    return min(16, max(2, 2 * math.ceil(required_pages / 2)))


def wave_grain_selection(
    max_cache_seqlens: tuple[int, ...],
    h_q: int,
    sm_count: int,
):
    capacity_pages = tuple(
        math.ceil(int(length) / PAGE_SIZE) for length in max_cache_seqlens
    )
    max_pages = max(capacity_pages, default=0)
    if max_pages < ADAPTIVE_MODEL_MIN_PAGES:
        return adaptive_fixed_pages(max_pages)

    if h_q <= 0 or (h_q not in (16, 32) and h_q % TLE_FP8_BH):
        raise ValueError("HQ must be 16, 32 or a positive multiple of 64")
    if sm_count <= 0:
        raise ValueError("SM count must be positive")
    rh = math.ceil(h_q / TLE_FP8_BH)

    # CUDA authority (`get_mla_metadata.cu`) assigns each SM partition a
    # payload that includes five fixed-overhead page blocks.  Preserve this
    # implementation's fixed even-pair routing, but derive its grain from the
    # same payload model. For 32 active heads per block at high batch, align
    # down to two page pairs; other routes round up to a whole pair.
    num_sm_parts = max(1, sm_count // rh)
    total_num_blocks = sum(
        pages + CUDA_REF_FIXED_OVERHEAD_PAGES for pages in capacity_pages
    )
    payload_blocks = max(
        math.ceil(total_num_blocks / num_sm_parts) + CUDA_REF_FIXED_OVERHEAD_PAGES,
        2 * CUDA_REF_FIXED_OVERHEAD_PAGES,
    )
    usable_pages = payload_blocks - CUDA_REF_FIXED_OVERHEAD_PAGES
    # Longest-first CTA order keeps coarse variable-length fragments balanced.
    # At high batch, use the CUDA payload estimate without the small-batch cap.
    grain_limit = (
        max_pages
        if len(capacity_pages) * rh >= sm_count // 2
        else ADAPTIVE_MAX_FIXED_PAGES
    )
    selected_pages = min(
        grain_limit,
        max(
            ADAPTIVE_MIN_FIXED_PAGES,
            (
                4 * (usable_pages // 4)
                if h_q == 32 and len(capacity_pages) * rh >= sm_count // 2
                else 2 * math.ceil(usable_pages / 2)
            ),
        ),
    )

    # A uniform per-row grain can leave only a handful of CTAs in a second
    # wave.  Keep the fixed even-pair contract, but allow the smallest larger
    # even grain when it collapses that sparse tail back into one H800 wave.
    # This is deliberately capped at 34 pages: it changes B8/L33280 from
    # 8 * ceil(520 / 32) = 136 CTAs to 8 * ceil(520 / 34) = 128 CTAs while
    # leaving the other formal routing points unchanged.
    initial_counts = tuple(
        max(1, math.ceil(pages / selected_pages)) for pages in capacity_pages
    )
    initial_total_ctas = sum(initial_counts) * rh
    if sm_count < initial_total_ctas <= 2 * sm_count:
        for candidate_pages in range(
            selected_pages + 2,
            ADAPTIVE_TAIL_WAVE_MAX_FIXED_PAGES + 1,
            2,
        ):
            candidate_counts = tuple(
                max(1, math.ceil(pages / candidate_pages)) for pages in capacity_pages
            )
            if sum(candidate_counts) * rh <= sm_count:
                selected_pages = candidate_pages
                break

    return int(selected_pages)


def build_adaptive_execution_meta(
    max_cache_seqlens,
    h_q: int,
    device: torch.device,
    short_pages_per_split: int,
):
    capacity_pages = tuple(
        math.ceil(int(length) / PAGE_SIZE) for length in max_cache_seqlens
    )
    max_pages = max(capacity_pages, default=0)
    if max_pages <= 2:
        fixed_pages = int(short_pages_per_split)
    elif (
        max_pages >= 64
        and len(capacity_pages) >= 128
        and all(pages == max_pages for pages in capacity_pages)
    ):
        # The query-row grid fills the device; avoid redundant split CTAs and Q loads.
        fixed_pages = max_pages
    elif (
        max_pages in (4, 8, 16, 32, 64)
        and h_q == 128
        and len(capacity_pages) == 128
        and all(pages == max_pages for pages in capacity_pages)
    ):
        # The measured high-batch workload already fills two CTA waves without
        # splitting; finer grains add redundant query loads and merge traffic.
        fixed_pages = max_pages
    elif 3 <= max_pages <= 8:
        # Short-K route: expose one physical-page CTA at a time instead of
        # serializing the complete 3-8 page row in a single CTA.  This is
        # host scheduling only; the strict-2WG kernel and pair pipeline are
        # unchanged.
        fixed_pages = 1
    elif (
        max_pages == 10
        and len(capacity_pages) >= 32
        and all(pages == 10 for pages in capacity_pages)
    ):
        # At high batch, five two-page split CTAs per row over-subscribe the
        # short ten-page workload and require a combine kernel. Use one
        # direct-output CTA per (batch, 64-head group) and only finalize LSE.
        # Keep this eligibility exact for heterogeneous rows and adjacent lengths.
        fixed_pages = max_pages
    elif (
        max_pages == 10
        and h_q == 128
        and len(capacity_pages) == 16
        and all(pages == 10 for pages in capacity_pages)
    ):
        # For B16/L640, reduce the partial grid from 160 CTAs (five
        # two-page splits per row) to 96 CTAs (three four-page-capacity
        # splits per row and two head groups).
        fixed_pages = 4
    elif 9 <= max_pages <= 10:
        # A ten-page direct-single CTA is not the best short-sequence route.  Use the finest
        # legal two-page pair grain to expose five split CTAs, mirroring the
        # CUDA reference's short-workload parallel split behavior.  Keep the
        # policy deliberately narrow until adjacent page ranges are measured.
        fixed_pages = 2
    elif (
        max_pages == 128
        and h_q == 64
        and len(capacity_pages) >= 64
        and all(pages == 128 for pages in capacity_pages)
    ):
        # Choose an even per-row grain that targets one H800
        # partial-CTA wave for a regular high-batch 8192-token workload.
        # This changes only host scheduling metadata; the partial kernel,
        # TMA/WGMMA/barrier structure, math, and route contracts are reused.
        sm_count = int(_get_num_sms(device))
        target_splits_per_row = max(1, sm_count // len(capacity_pages))
        fixed_pages = min(
            max_pages,
            2 * math.ceil(math.ceil(max_pages / target_splits_per_row) / 2),
        )
    elif (
        max_pages == 520
        and h_q == 64
        and len(capacity_pages) == 16
        and all(pages == 520 for pages in capacity_pages)
    ):
        fixed_pages = 65
    elif (
        max_pages == 520
        and h_q == 64
        and len(capacity_pages) >= 16
        and all(pages == 520 for pages in capacity_pages)
    ):
        # Choose the minimum even grain that caps the regular high-batch
        # L33280 workload at one H800 partial-CTA wave.
        sm_count = int(_get_num_sms(device))
        target_splits_per_row = max(1, sm_count // len(capacity_pages))
        fixed_pages = min(
            max_pages,
            2 * math.ceil(math.ceil(max_pages / target_splits_per_row) / 2),
        )
    elif (
        h_q == 64
        and len(capacity_pages) == 16
        and all(pages == 128 for pages in capacity_pages)
    ):
        fixed_pages = 16
    else:
        sm_count = int(_get_num_sms(device))
        fixed_pages = wave_grain_selection(tuple(max_cache_seqlens), h_q, sm_count)
    # Host counts size the workspace; split ranges are generated only on the GPU.
    counts = tuple(max(1, math.ceil(pages / fixed_pages)) for pages in capacity_pages)
    prefix = (0, *accumulate(counts))
    total_splits = prefix[-1]
    padded_pages = max(1, max(counts, default=1) * fixed_pages)

    meta = FlashMLAFp8SplitKSchedMeta()
    meta.have_initialized = True
    meta.num_splits = torch.empty((len(prefix),), dtype=torch.int32, device=device)
    meta.split_batch = torch.empty((total_splits,), dtype=torch.int32, device=device)
    meta.split_page_begin = torch.empty_like(meta.split_batch)
    meta.split_num_pages = torch.empty_like(meta.split_batch)
    initialize_capacity_plan[(len(capacity_pages),)](
        meta.num_splits,
        meta.split_batch,
        meta.split_page_begin,
        meta.split_num_pages,
        capacity_pages,
        prefix,
        BATCH=len(capacity_pages),
        GRAIN=fixed_pages,
        BLOCK=triton.next_power_of_2(max(counts, default=1)),
        num_warps=4,
    )
    meta.split_order = torch.empty_like(meta.split_batch)
    order_dense_splits[(1,)](
        meta.split_num_pages,
        meta.split_order,
        total_splits,
        triton.next_power_of_2(total_splits),
    )
    meta.total_split_capacity = total_splits
    meta.adaptive_fixed_pages = fixed_pages
    meta.adaptive_fixed_pairs = math.ceil(fixed_pages / 2)
    meta.capacity_splits = counts
    meta.padded_pages = padded_pages
    return meta


if HAS_TLE:

    # One metadata-only sorting block; size is fixed by the prepared split count.
    @triton.jit
    def order_dense_splits(Pages, Order, COUNT: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        pages = tl.load(Pages + offsets, offsets < COUNT, 0)
        keys = tl.where(offsets < COUNT, -pages * BLOCK + offsets, 2147483647)
        keys = tl.sort(keys, descending=False)
        order = keys & (BLOCK - 1)
        tl.store(Order + offsets, order, offsets < COUNT)

    @triton.jit
    def initialize_capacity_plan(
        Prefix,
        Batch,
        Begin,
        Pages,
        CapacityPages,
        PrefixValues,
        BATCH: tl.constexpr,
        GRAIN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0)
        capacity = 0
        start = 0
        stop = 0
        for row in tl.static_range(BATCH):
            capacity = tl.where(batch == row, CapacityPages[row], capacity)
            start = tl.where(batch == row, PrefixValues[row], start)
            stop = tl.where(batch == row, PrefixValues[row + 1], stop)
        count = stop - start
        local = tl.arange(0, BLOCK)
        page_begin = local * GRAIN
        num_pages = tl.maximum(0, tl.minimum(GRAIN, capacity - page_begin))
        if GRAIN == 16:
            short_before = local * 4 // count
            short_through = (local + 1) * 4 // count
            balanced = (capacity == 520) & (count == 33)
            page_begin = tl.where(balanced, local * 16 - 2 * short_before, page_begin)
            num_pages = tl.where(
                balanced, tl.where(short_through != short_before, 14, 16), num_pages
            )
        elif GRAIN == 34:
            balanced = (capacity == 520) & (count == 16)
            begin = capacity * local // count
            end = capacity * (local + 1) // count
            page_begin = tl.where(balanced, begin, page_begin)
            num_pages = tl.where(balanced, end - begin, num_pages)
        else:
            pass
        mask = local < count
        tl.store(Prefix + batch, start)
        if batch == BATCH - 1:
            tl.store(Prefix + BATCH, stop)
        tl.store(Batch + start + local, batch, mask)
        tl.store(Begin + start + local, page_begin, mask)
        tl.store(Pages + start + local, num_pages, mask)


if HAS_TLE:

    @triton.jit
    def plan_length_splits(
        Lengths,
        Prefix,
        Batch,
        Begin,
        Pages,
        BATCH: tl.constexpr,
        NUM_PARTS: tl.constexpr,
        CAPACITY: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_S: tl.constexpr,
        CHUNK: tl.constexpr,
        GRAIN_CHUNK: tl.constexpr,
    ):
        """Split each row by its GPU length over a fixed CTA budget."""
        rows = tl.arange(0, BLOCK_B)
        valid = rows < BATCH
        lengths = tl.load(Lengths + rows, valid, 0)
        pages = tl.where(valid, tl.cdiv(tl.maximum(lengths, 0), TLE_PAGE_SIZE), 0)
        max_pages = tl.maximum(tl.max(pages, 0), 1)
        total_pages = tl.sum(pages, 0)
        # Try every grain up to the longest row. Each row is cut into equal
        # pieces no larger than the grain. The estimated makespan is the larger
        # of the biggest piece and the per-SM work rounded up to whole waves;
        # every split also pays a fixed setup/combine overhead.
        best_cost = tl.full((), float("inf"), tl.float32)
        grain = max_pages
        for grain_base in tl.range(0, max_pages, GRAIN_CHUNK):
            grains = grain_base + 1 + tl.arange(0, GRAIN_CHUNK)
            pieces = tl.where(
                valid[None, :],
                tl.maximum(tl.cdiv(pages[None, :], grains[:, None]), 1),
                1,
            )
            split_counts = tl.sum(tl.where(valid[None, :], pieces, 0), 1)
            largest = tl.max(tl.cdiv(pages[None, :], pieces), 1)
            waves = tl.cdiv(split_counts, NUM_PARTS)
            work = (total_pages + TLE_SPLIT_OVERHEAD_PAGES * split_counts).to(
                tl.float32
            )
            cost = tl.maximum(
                (largest + TLE_SPLIT_OVERHEAD_PAGES).to(tl.float32),
                work * waves.to(tl.float32) / split_counts.to(tl.float32),
            )
            usable = (grains <= max_pages) & (split_counts <= CAPACITY)
            cost = tl.where(usable, cost, float("inf"))
            chunk_cost = tl.min(cost, 0)
            chunk_grain = grain_base + 1 + tl.argmin(cost, 0)
            better = chunk_cost < best_cost
            grain = tl.where(better, chunk_grain, grain)
            best_cost = tl.where(better, chunk_cost, best_cost)
        counts = tl.where(valid, tl.maximum(tl.cdiv(pages, grain), 1), 0)
        ends = tl.cumsum(counts, 0)
        starts = ends - counts
        tl.store(Prefix + rows, starts, valid)
        tl.store(Prefix + BATCH, tl.sum(counts, 0))
        for chunk in tl.range(0, BLOCK_S, CHUNK):
            splits = chunk + tl.arange(0, CHUNK)
            owner = tl.sum(
                tl.where(valid[None, :] & (ends[None, :] <= splits[:, None]), 1, 0), 1
            )
            is_owner = rows[None, :] == owner[:, None]
            row_pages = tl.sum(tl.where(is_owner, pages[None, :], 0), 1)
            row_start = tl.sum(tl.where(is_owner, starts[None, :], 0), 1)
            row_count = tl.maximum(tl.sum(tl.where(is_owner, counts[None, :], 0), 1), 1)
            local = splits - row_start
            active = owner < BATCH
            # Unused budget slots get page_begin=-1 and no pages.
            begin = tl.where(active, row_pages * local // row_count, -1)
            end = tl.where(active, row_pages * (local + 1) // row_count, -1)
            mask = splits < CAPACITY
            tl.store(Batch + splits, tl.where(active, owner, 0), mask)
            tl.store(Begin + splits, begin, mask)
            tl.store(Pages + splits, end - begin, mask)


if HAS_TLE:

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("flash_mla_fp8_copy_table"),
        key=["BATCH", "COLS", "PADDED"],
        use_cuda_graph=True,
    )
    @triton.jit
    def copy_block_table(
        Source,
        Destination,
        BATCH: tl.constexpr,
        COLS: tl.constexpr,
        PADDED: tl.constexpr,
        ROW_STRIDE: tl.constexpr,
        COL_STRIDE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        rows, cols = offsets // PADDED, offsets % PADDED
        values = tl.load(
            Source + rows * ROW_STRIDE + cols * COL_STRIDE,
            (rows < BATCH) & (cols < COLS),
            0,
        )
        tl.store(Destination + offsets, values, rows < BATCH)


def pad_block_table(block_table: torch.Tensor, padded_pages: int):
    if int(block_table.shape[1]) >= padded_pages:
        return block_table
    padded = torch.empty(
        (block_table.shape[0], padded_pages),
        dtype=block_table.dtype,
        device=block_table.device,
    )
    copy_block_table[lambda meta: (triton.cdiv(padded.numel(), meta["BLOCK"]),)](
        block_table,
        padded,
        block_table.shape[0],
        block_table.shape[1],
        padded_pages,
        *block_table.stride(),
    )
    return padded


def get_mla_fp8_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int = 1,
) -> Tuple[FlashMLAFp8SplitKSchedMeta, torch.Tensor]:
    """Plan length-balanced splits on the current stream from GPU lengths."""
    if cache_seqlens.ndim != 1 or cache_seqlens.dtype != torch.int32:
        raise ValueError("cache_seqlens must be a 1-D int32 tensor")
    if cache_seqlens.device.type != "cuda" or cache_seqlens.stride(0) != 1:
        raise NotImplementedError("metadata requires contiguous CUDA lengths")
    if num_heads_k != 1:
        raise NotImplementedError("metadata requires one KV head")
    batch = cache_seqlens.numel()
    heads = int(num_q_tokens_per_head_k)
    if batch <= 0 or heads <= 0:
        raise ValueError("metadata requires a nonempty batch and positive heads")
    device = cache_seqlens.device
    num_parts = max(1, int(_get_num_sms(device)) // math.ceil(heads / TLE_FP8_BH))
    capacity = num_parts * LENGTH_SPLIT_CAPACITY_PER_PART + batch
    meta = FlashMLAFp8SplitKSchedMeta()
    meta.num_splits = torch.empty((batch + 1,), dtype=torch.int32, device=device)
    meta.split_batch = torch.empty((capacity,), dtype=torch.int32, device=device)
    meta.split_page_begin = torch.empty_like(meta.split_batch)
    meta.split_num_pages = torch.empty_like(meta.split_batch)
    meta.split_order = torch.empty_like(meta.split_batch)
    block_rows = triton.next_power_of_2(batch)
    block_splits = triton.next_power_of_2(capacity)
    plan_length_splits[(1,)](
        cache_seqlens,
        meta.num_splits,
        meta.split_batch,
        meta.split_page_begin,
        meta.split_num_pages,
        batch,
        num_parts,
        capacity,
        block_rows,
        block_splits,
        min(block_splits, max(16, 8192 // block_rows)),
        max(1, 8192 // block_rows),
        num_warps=8,
    )
    order_dense_splits[(1,)](
        meta.split_num_pages, meta.split_order, capacity, block_splits
    )
    meta.total_split_capacity = capacity
    meta.batch_size = batch
    meta.query_heads = heads
    meta.have_initialized = True
    return meta, meta.num_splits


def launch_length_scheduled_decode(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    meta: FlashMLAFp8SplitKSchedMeta,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    query_heads: int,
    query_count: int,
) -> None:
    """Run split partials and the variable-split combine on the current stream.

    ``out`` is the merged-head view; ``lse`` keeps the public (batch, heads,
    queries) layout and receives merged head ``q * heads + h`` at ``[b, h, q]``.
    """
    batch, _, heads, _ = q_nope.shape
    head_blocks = math.ceil(heads / TLE_FP8_BH)
    capacity = meta.total_split_capacity
    device = q_nope.device
    partial_out = torch.empty(
        (capacity, heads, D_CKV), dtype=torch.float32, device=device
    )
    partial_lse2 = torch.empty((capacity, heads), dtype=torch.float32, device=device)
    _ensure_triton_descriptor_allocator(device)
    descriptor = _get_tensor_descriptor_cls()
    q_desc = descriptor.from_tensor(
        q_nope.view(-1, D_CKV), block_shape=[TLE_FP8_BH, D_CKV]
    )
    qr_desc = descriptor.from_tensor(
        q_rope.view(-1, D_ROPE), block_shape=[TLE_FP8_BH, D_ROPE]
    )
    qs_desc = descriptor.from_tensor(
        q_scale.view(-1, heads), block_shape=[1, TLE_FP8_BH]
    )
    k_desc = descriptor.from_tensor(
        k_cache_lora.view(-1, D_CKV), block_shape=[TLE_FP8_BK, K_CONTENT_TILE_HOST]
    )
    kr_desc = descriptor.from_tensor(
        k_cache_rope.view(-1, D_ROPE), block_shape=[TLE_FP8_BK, D_ROPE]
    )
    ks_desc = descriptor.from_tensor(
        k_scale.view(-1, TLE_FP8_BK), block_shape=[1, TLE_FP8_BK]
    )
    # Single-split rows store whole 64-head tiles through TMA when every tile
    # stays inside one batch row and the output meets TMA alignment.
    tma_direct = heads % TLE_FP8_BH == 0 and out.data_ptr() % 16 == 0
    out_desc = (
        descriptor.from_tensor(out.view(-1, D_CKV), block_shape=[TLE_FP8_BH, D_ROPE])
        if tma_direct
        else q_desc
    )
    use_pdl = batch >= CUDA_COARSE_COMBINE_MIN_BATCH
    fp8_dense_mla_splitk_partial[(capacity * head_blocks,)](
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        cache_seqlens,
        meta.split_batch,
        meta.split_page_begin,
        meta.split_num_pages,
        meta.split_order,
        partial_out,
        partial_lse2,
        q_desc,
        qr_desc,
        qs_desc,
        out_desc,
        k_desc,
        kr_desc,
        ks_desc,
        q_nope.stride(0),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(2),
        q_scale.stride(0),
        q_scale.stride(2),
        k_cache_lora.stride(0),
        k_cache_lora.stride(1),
        k_cache_rope.stride(0),
        k_cache_rope.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        cache_seqlens.stride(0),
        1,
        1,
        1,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_lse2.stride(0),
        partial_lse2.stride(1),
        softmax_scale,
        TLE_FP8_BH * D_CKV,
        TLE_FP8_BH * D_ROPE * 2,
        TLE_FP8_BH * 4,
        TLE_FP8_BK * K_CONTENT_TILE_HOST,
        TLE_FP8_BK * D_ROPE * 2,
        TLE_FP8_BK * 4,
        D_CKV,
        D_ROPE,
        TLE_FP8_BK,
        TLE_FP8_BH,
        heads,
        head_blocks,
        PAGE_SIZE,
        TLE_FP8_DPH,
        USE_HOTLOOP_RECIP=True,
        FULL_TAIL=False,
        QUERY_HEADS=query_heads,
        QUERY_COUNT=query_count,
        PAGE_GRAIN_TAIL_ZERO=True,
        MERGE_STATE_V=False,
        USE_TMA_OUTPUT=False,
        FIXED_NUM_PAGES=0,
        DIRECT_LSE=False,
        ENABLE_PDL=use_pdl,
        PRETRANSPOSE_V1=False,
        row_prefix_ptr=meta.num_splits,
        final_out_ptr=out,
        final_lse_ptr=lse,
        stride_fo_b=out.stride(0),
        stride_fo_h=out.stride(2),
        stride_fl_b=lse.stride(0),
        stride_fl_h=lse.stride(1),
        stride_fl_q=lse.stride(2),
        DYNAMIC_DIRECT=True,
        TMA_DIRECT=tma_direct,
        LSE_HEADS=lse.shape[1],
        num_warps=4,
        num_stages=1,
    )
    combine_args = (
        partial_out,
        partial_lse2,
        meta.num_splits,
        out,
        lse,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_lse2.stride(0),
        partial_lse2.stride(1),
        meta.num_splits.stride(0),
        out.stride(0),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        heads,
        D_CKV,
    )
    if batch >= CUDA_COARSE_COMBINE_MIN_BATCH:
        triton_fp8_coarse_combine_kernel[
            (batch, math.ceil(heads / CUDA_COARSE_COMBINE_BLOCK_ROWS))
        ](
            *combine_args,
            CUDA_COARSE_COMBINE_BLOCK_SPLITS,
            CUDA_COARSE_COMBINE_BLOCK_ROWS,
            True,
            SKIP_SINGLE_SPLIT=True,
            stride_lse_q=lse.stride(2),
            LSE_HEADS=lse.shape[1],
            num_warps=8,
            launch_pdl=True,
        )
    else:
        triton_fp8_splitk_combine_kernel[
            (batch * heads, math.ceil(D_CKV / COMBINE_BLOCK_D))
        ](
            *combine_args,
            COMBINE_BLOCK_SPLITS,
            COMBINE_BLOCK_D,
            SKIP_SINGLE_SPLIT=True,
            stride_lse_q=lse.stride(2),
            LSE_HEADS=lse.shape[1],
            num_warps=4,
        )


if HAS_TLE:

    @triton.jit
    def write_cache_lengths(Lengths, Values, BATCH: tl.constexpr):
        row = tl.program_id(0)
        value = 0
        for index in tl.static_range(BATCH):
            value = tl.where(row == index, Values[index], value)
        tl.store(Lengths + row, value)


def validate_dense_inputs(
    q_nope,
    q_rope,
    k_cache_lora,
    k_cache_rope,
    q_scale,
    k_scale,
    block_table,
    cache_seqlens,
    head_dim_v,
):
    if (
        runtime.device.vendor_name != "nvidia"
        or q_nope.device.type != "cuda"
        or torch.cuda.get_device_capability(q_nope.device)[0] != 9
    ):
        raise NotImplementedError("requires NVIDIA Hopper tensors")
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != D_CKV:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if batch <= 0 or heads <= 0 or (heads not in (16, 32) and heads % TLE_FP8_BH):
        raise NotImplementedError(
            "requires positive batch and 16, 32 or a multiple of 64 query heads"
        )
    if (
        k_cache_lora.ndim not in (3, 4)
        or k_cache_lora.shape[1] != PAGE_SIZE
        or k_cache_lora.shape[-1] != D_CKV
    ):
        raise NotImplementedError(
            "requires 64-token KV pages with 512 content dimensions"
        )
    if k_cache_lora.ndim == 4 and k_cache_lora.shape[2] != 1:
        raise NotImplementedError("requires one KV head")
    if head_dim_v != D_CKV:
        raise NotImplementedError("requires head_dim_v=512")
    expected = (
        (batch, queries, heads, D_ROPE),
        (*k_cache_lora.shape[:-1], D_ROPE),
        (batch, queries, heads, 1),
        (*k_cache_lora.shape[:-1], 1),
        (batch,),
    )
    for tensor, shape in zip(
        (q_rope, k_cache_rope, q_scale, k_scale, cache_seqlens), expected
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"expected tensor shape {shape}, got {tuple(tensor.shape)}"
            )
    if q_nope.dtype != torch.float8_e4m3fn or k_cache_lora.dtype != torch.float8_e4m3fn:
        raise TypeError("NoPE tensors must be float8_e4m3fn")
    if q_rope.dtype != torch.bfloat16 or k_cache_rope.dtype != torch.bfloat16:
        raise TypeError("RoPE tensors must be bfloat16")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be float32")
    if cache_seqlens.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise TypeError("lengths and block table must be int32")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] <= 0
    ):
        raise ValueError(
            "block table must have positive page capacity for every request"
        )
    descriptor_inputs = (q_nope, q_rope, q_scale, k_cache_lora, k_cache_rope, k_scale)
    for tensor in (*descriptor_inputs, block_table, cache_seqlens):
        if tensor.device != q_nope.device:
            raise ValueError("all inputs must share the query device")
    for tensor in descriptor_inputs:
        if not tensor.is_contiguous():
            raise NotImplementedError("dense TMA inputs require contiguous storage")
        if tensor.data_ptr() % 16:
            raise NotImplementedError("dense TMA inputs require 16-byte alignment")
    if cache_seqlens.stride(0) != 1:
        raise NotImplementedError("length storage must be contiguous")
    if min(q_nope.shape[0], k_cache_lora.shape[0]) <= 0:
        raise ValueError("query and KV storage must be nonempty")


def prepare_compiled_runner(
    jit_function,
    args,
    grid,
    *,
    num_warps: int = 4,
    launch_pdl: bool = False,
):
    """Bind one already-specialized Triton kernel to its direct launcher."""
    from triton.runtime._async_compile import FutureKernel
    from triton.runtime.driver import driver

    if jit_function.pre_run_hooks:
        raise RuntimeError(
            "prepared compiled launch does not support JIT pre-run hooks"
        )
    device = driver.active.get_current_device()
    _, _, _, _, binder = jit_function.device_caches[device]
    bound_args, _, _ = binder(
        *args,
        num_warps=num_warps,
        num_stages=1,
        launch_pdl=launch_pdl,
    )
    kernel = jit_function.warmup(
        *args,
        grid=grid,
        num_warps=num_warps,
        num_stages=1,
        launch_pdl=launch_pdl,
    )
    if isinstance(kernel, FutureKernel):
        kernel = kernel.result()
    grid3 = tuple(grid) + (1,) * (3 - len(grid))
    return kernel[grid3], tuple(bound_args.values())


def prepare_dense_decode_core(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    softmax_scale: Optional[float] = None,
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
    *,
    initial_cache_seqlens: Sequence[int],
    max_cache_seqlens: Sequence[int],
    has_length_certificate: bool = True,
    query_heads: int = 0,
    query_count: int = 1,
) -> tuple[FlashMLAFp8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    """Build an adaptive split plan and a CUDA Graph-compatible replay handle."""
    if not HAS_TLE:
        raise NotImplementedError("FP8 MLA requires Hopper and FlagTree GPU extensions")
    validate_dense_inputs(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
    )
    batch_size = int(q_nope.shape[0])
    if batch_size <= 0 or int(cache_seqlens.numel()) != batch_size:
        raise ValueError("batch dimensions do not match")
    if int(q_nope.shape[1]) != 1:
        raise ValueError("SQ=1 decode only")
    h_q = int(q_nope.shape[2])
    if h_q <= 0 or (h_q not in (16, 32) and h_q % TLE_FP8_BH):
        raise ValueError("HQ must be 16, 32 or a positive multiple of 64")
    initial = host_certificate_lengths(
        initial_cache_seqlens,
        "initial_cache_seqlens",
        batch_size=batch_size,
    )
    capacity = host_certificate_lengths(
        max_cache_seqlens,
        "max_cache_seqlens",
        batch_size=batch_size,
    )
    if any(current > maximum for current, maximum in zip(initial, capacity)):
        raise ValueError("initial_cache_seqlens cannot exceed max_cache_seqlens")
    required_pages = max(math.ceil(length / PAGE_SIZE) for length in capacity)
    if block_table.ndim != 2 or int(block_table.shape[0]) != batch_size:
        raise ValueError("block_table must be a two-dimensional batch table")
    if int(block_table.shape[1]) < required_pages:
        raise ValueError(
            "block_table does not cover the prepared max_cache_seqlens capacity"
        )
    meta = build_adaptive_execution_meta(
        capacity,
        h_q,
        q_nope.device,
        pages_per_split,
    )
    if max_splits is not None and int(max_splits) < max(meta.capacity_splits):
        raise ValueError("max_splits cannot be below an adaptive row capacity")
    execution_block_table = pad_block_table(block_table, meta.padded_pages)
    total_splits = int(meta.split_batch.numel())
    scale = float(softmax_scale if softmax_scale is not None else D_QK**-0.5)

    direct_single_output = bool(meta.capacity_splits) and all(
        count == 1 for count in meta.capacity_splits
    )
    if direct_single_output:
        partial_out = torch.empty((0,), dtype=torch.float32, device=q_nope.device)
    else:
        partial_out = torch.empty(
            (total_splits, h_q, D_CKV),
            dtype=torch.float32,
            device=q_nope.device,
        )
    partial_lse2 = torch.empty(
        (total_splits, h_q), dtype=torch.float32, device=q_nope.device
    )
    out = torch.empty(
        (batch_size, 1, h_q, head_dim_v), dtype=q_rope.dtype, device=q_nope.device
    )
    lse = torch.empty((batch_size, h_q, 1), dtype=torch.float32, device=q_nope.device)

    handle = FlashMLAFp8PreparedHandle(
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        execution_block_table,
        cache_seqlens,
        meta,
        partial_out,
        partial_lse2,
        out,
        lse,
        h_q,
        scale,
        head_dim_v,
        initial,
        capacity,
        has_length_certificate=has_length_certificate,
        query_heads=query_heads,
        query_count=query_count,
    )
    first_result = handle()
    return handle, first_result


class FlashMLAFp8PreparedHandle:
    """Callable prepared decode handle with stable descriptors and workspace."""

    __slots__ = (
        "q_nope",
        "q_rope",
        "q_scale",
        "k_cache_lora",
        "k_cache_rope",
        "k_scale",
        "block_table",
        "execution_block_table",
        "cache_seqlens",
        "meta",
        "partial_out",
        "partial_lse2",
        "out",
        "lse",
        "h_q",
        "scale",
        "head_dim_v",
        "initial_cache_seqlens",
        "max_cache_seqlens",
        "cache_seqlens_host",
        "cache_version",
        "num_pages",
        "logical_active_splits",
        "direct_single_output",
        "in_use",
        "q_desc",
        "qr_desc",
        "qs_desc",
        "out_desc",
        "k_desc",
        "kr_desc",
        "ks_desc",
        "launch_pack_key",
        "partial_compiled_runner",
        "partial_compiled_args",
        "aux_compiled_runner",
        "aux_compiled_args",
        "launch_pack_reuses",
        "cuda_graph_key",
        "cuda_graph",
        "cuda_graph_capture_stream",
        "cuda_graph_eligible",
        "has_length_certificate",
        "query_heads",
        "query_count",
    )

    def __init__(
        self,
        q_nope,
        q_rope,
        q_scale,
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        execution_block_table,
        cache_seqlens,
        meta,
        partial_out,
        partial_lse2,
        out,
        lse,
        h_q,
        scale,
        head_dim_v,
        initial_cache_seqlens,
        max_cache_seqlens,
        has_length_certificate=True,
        query_heads=0,
        query_count=1,
    ) -> None:
        self.has_length_certificate = has_length_certificate
        self.query_heads = query_heads
        self.query_count = query_count
        self.q_nope = q_nope
        self.q_rope = q_rope
        self.q_scale = q_scale
        self.k_cache_lora = k_cache_lora
        self.k_cache_rope = k_cache_rope
        self.k_scale = k_scale
        self.block_table = block_table
        self.execution_block_table = execution_block_table
        self.cache_seqlens = cache_seqlens
        self.meta = meta
        self.partial_out = partial_out
        self.partial_lse2 = partial_lse2
        self.out = out
        self.lse = lse
        self.h_q = h_q
        self.scale = scale
        self.head_dim_v = head_dim_v
        self.direct_single_output = bool(meta.capacity_splits) and all(
            count == 1 for count in meta.capacity_splits
        )
        # Prepared replay keeps the bound Q/K tensors and their storage
        # addresses stable.  Match CUDA's launch-parameter lifetime by
        # materializing the six immutable TMA descriptors once instead of
        # rebuilding them on every decode step.
        _ensure_triton_descriptor_allocator(q_nope.device)
        self.q_desc = _get_tensor_descriptor_cls().from_tensor(
            q_nope.view(-1, D_CKV), block_shape=[TLE_FP8_BH, D_CKV]
        )
        self.qr_desc = _get_tensor_descriptor_cls().from_tensor(
            q_rope.view(-1, D_ROPE), block_shape=[TLE_FP8_BH, D_ROPE]
        )
        self.qs_desc = _get_tensor_descriptor_cls().from_tensor(
            q_scale.view(-1, h_q), block_shape=[1, TLE_FP8_BH]
        )
        # Rebound in _partial_launch_args when caller-provided output storage
        # changes; non-direct routes never consume this placeholder.
        self.out_desc = self.q_desc
        self.k_desc = _get_tensor_descriptor_cls().from_tensor(
            k_cache_lora.view(-1, D_CKV),
            block_shape=[TLE_FP8_BK, K_CONTENT_TILE_HOST],
        )
        self.kr_desc = _get_tensor_descriptor_cls().from_tensor(
            k_cache_rope.view(-1, D_ROPE), block_shape=[TLE_FP8_BK, D_ROPE]
        )
        self.ks_desc = _get_tensor_descriptor_cls().from_tensor(
            k_scale.view(-1, TLE_FP8_BK), block_shape=[1, TLE_FP8_BK]
        )
        self.initial_cache_seqlens = tuple(initial_cache_seqlens)
        self.max_cache_seqlens = tuple(max_cache_seqlens)
        self.cache_seqlens_host = tuple(initial_cache_seqlens)
        self.cache_version = tensor_version(cache_seqlens)
        self.num_pages, self.logical_active_splits = length_page_state(
            self.cache_seqlens_host,
            int(meta.adaptive_fixed_pages),
        )
        if self.direct_single_output:
            if int(meta.split_batch.numel()) != len(self.max_cache_seqlens):
                raise AssertionError(
                    "direct-single schedule must contain one split per batch row"
                )
            if self.partial_out.numel() != 0:
                raise AssertionError(
                    "direct-single schedule must not allocate an FP32 output workspace"
                )
        self.in_use = False
        self.launch_pack_key = None
        self.partial_compiled_runner = None
        self.partial_compiled_args = None
        self.aux_compiled_runner = None
        self.aux_compiled_args = None
        self.launch_pack_reuses = 0
        self.cuda_graph_key = None
        self.cuda_graph = None
        self.cuda_graph_capture_stream = None
        self.cuda_graph_eligible = False

    def programmatic_dependency_capacity(self):
        batch_size = int(self.out.shape[0])
        partial_ctas = int(self.meta.split_batch.numel()) * math.ceil(
            self.h_q / TLE_FP8_BH
        )
        consumer_ctas = batch_size * math.ceil(
            self.h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS
        )
        sm_count = int(_get_num_sms(self.out.device))
        return partial_ctas, consumer_ctas, sm_count

    def use_programmatic_dependent_launch(self) -> bool:
        partial_ctas, consumer_ctas, sm_count = self.programmatic_dependency_capacity()
        # Keep one full consumer grid of scheduling headroom beyond the
        # producer and consumer fit.
        return (
            not self.direct_single_output
            and int(self.out.shape[0]) >= CUDA_COARSE_COMBINE_MIN_BATCH
            and partial_ctas + 2 * consumer_ctas <= sm_count + 1
        )

    def use_full_tail_specialization(self) -> bool:
        # Every scheduled capacity page must be real and complete.  A handle
        # whose current lengths have not reached prepared capacity keeps the
        # masked kernel even when the current token count is 64-aligned.
        # Causal query suffix masks are applied independently in the epilogue.
        return (
            self.has_length_certificate
            and self.cache_seqlens_host == self.max_cache_seqlens
            and all(length % PAGE_SIZE == 0 for length in self.cache_seqlens_host)
            and tuple(self.logical_active_splits) == tuple(self.meta.capacity_splits)
        )

    def use_merged_state_v_completion(self) -> bool:
        return (
            self.h_q == 64
            # Admit B8 full-tail shapes to the split-structure-agnostic
            # merged-completion schedule.
            and int(self.out.shape[0]) >= 8
            # L8192 also satisfies the full-tail certificate required below.
            and all(length in (33280, 8192) for length in self.max_cache_seqlens)
            and self.use_full_tail_specialization()
        )

    def use_pretranspose_v1(self) -> bool:
        return (
            self.h_q == 128
            and int(self.out.shape[0]) in (16, 32)
            and all(length == 640 for length in self.max_cache_seqlens)
        )

    def use_fixed_ten_page_v1(self) -> bool:
        return (
            self.h_q == 128
            and int(self.out.shape[0]) in (32, 64, 128)
            and all(length == 640 for length in self.max_cache_seqlens)
            and self.use_full_tail_specialization()
        )

    def use_fixed_two_page_v1(self) -> bool:
        # The direct one-pair family has one CTA per batch row.  With every
        # certified length in (64, 128], each CTA has exactly two real pages,
        # although page one may be partial.  Constant-fold only num_pages;
        # retain the masked math and worker schedule unchanged.
        return (
            self.has_length_certificate
            and self.direct_single_output
            and self.cache_seqlens_host == self.max_cache_seqlens
            and all(
                PAGE_SIZE < length <= 2 * PAGE_SIZE
                for length in self.cache_seqlens_host
            )
        )

    def use_direct_lse_v2(self) -> bool:
        # A direct-single route has exactly one partial CTA for each output
        # row/head block, so its WG0 LSE store has no cross-split reduction.
        # Every direct-output route can write natural-log LSE here and omit
        # the separate conversion kernel.
        return self.direct_single_output

    def partial_launch_args(self, target_out, target_lse):
        h_q = self.h_q
        rh = math.ceil(h_q / TLE_FP8_BH)
        # Short head blocks use masked stores; a 64-row TMA store would overlap batches.
        if self.direct_single_output and h_q >= TLE_FP8_BH:
            self.out_desc = _get_tensor_descriptor_cls().from_tensor(
                target_out.view(-1, D_CKV), block_shape=[TLE_FP8_BH, D_ROPE]
            )
        return (
            self.q_nope,
            self.q_rope,
            self.q_scale,
            self.k_cache_lora,
            self.k_cache_rope,
            self.k_scale,
            self.execution_block_table,
            self.cache_seqlens,
            self.meta.split_batch,
            self.meta.split_page_begin,
            self.meta.split_num_pages,
            self.meta.split_order,
            target_out,
            target_lse,
            self.q_desc,
            self.qr_desc,
            self.qs_desc,
            self.out_desc,
            self.k_desc,
            self.kr_desc,
            self.ks_desc,
            self.q_nope.stride(0),
            self.q_nope.stride(2),
            self.q_rope.stride(0),
            self.q_rope.stride(2),
            self.q_scale.stride(0),
            self.q_scale.stride(2),
            self.k_cache_lora.stride(0),
            self.k_cache_lora.stride(1),
            self.k_cache_rope.stride(0),
            self.k_cache_rope.stride(1),
            self.k_scale.stride(0),
            self.k_scale.stride(1),
            self.execution_block_table.stride(0),
            self.execution_block_table.stride(1),
            self.cache_seqlens.stride(0),
            self.meta.split_batch.stride(0),
            self.meta.split_page_begin.stride(0),
            self.meta.split_num_pages.stride(0),
            target_out.stride(0),
            target_out.stride(2 if self.direct_single_output else 1),
            target_lse.stride(0),
            target_lse.stride(1),
            self.scale,
            64 * 512,
            64 * 64 * 2,
            64 * 4,
            64 * K_CONTENT_TILE_HOST,
            64 * 64 * 2,
            64 * 4,
            D_CKV,
            D_ROPE,
            TLE_FP8_BK,
            TLE_FP8_BH,
            h_q,
            rh,
            PAGE_SIZE,
            TLE_FP8_DPH,
            int(self.meta.adaptive_fixed_pairs) >= 2,
            self.use_full_tail_specialization(),
            self.query_heads,
            self.query_count,
            True,  # Zero invalid rows in shared memory for every split grain.
            self.use_merged_state_v_completion(),
            self.direct_single_output and h_q >= TLE_FP8_BH,
            (
                10
                if self.use_fixed_ten_page_v1()
                else (
                    2
                    if self.use_fixed_two_page_v1()
                    else (
                        int(self.meta.adaptive_fixed_pages)
                        if self.use_full_tail_specialization()
                        and all(
                            (length // PAGE_SIZE) % int(self.meta.adaptive_fixed_pages)
                            == 0
                            for length in self.max_cache_seqlens
                        )
                        else 0
                    )
                )
            ),
            self.use_direct_lse_v2(),
        )

    def aux_launch_spec(self, out, lse):
        if self.direct_single_output:
            total = int(lse.shape[0]) * self.h_q
            return (
                triton_fp8_single_split_lse_finalize_kernel,
                (
                    self.partial_lse2,
                    lse,
                    self.partial_lse2.stride(0),
                    self.partial_lse2.stride(1),
                    lse.stride(0),
                    lse.stride(1),
                    self.h_q,
                    total,
                    LSE_FINALIZE_BLOCK,
                ),
                (triton.cdiv(total, LSE_FINALIZE_BLOCK),),
            )

        common_args = (
            self.partial_out,
            self.partial_lse2,
            self.meta.num_splits,
            out,
            lse,
            self.partial_out.stride(0),
            self.partial_out.stride(1),
            self.partial_lse2.stride(0),
            self.partial_lse2.stride(1),
            self.meta.num_splits.stride(0),
            out.stride(0),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            self.h_q,
            D_CKV,
        )
        batch_size = int(out.shape[0])
        if batch_size >= CUDA_COARSE_COMBINE_MIN_BATCH:

            return (
                triton_fp8_coarse_combine_kernel,
                common_args
                + (
                    CUDA_COARSE_COMBINE_BLOCK_SPLITS,
                    CUDA_COARSE_COMBINE_BLOCK_ROWS,
                    self.use_programmatic_dependent_launch(),
                ),
                (
                    batch_size,
                    math.ceil(self.h_q / CUDA_COARSE_COMBINE_BLOCK_ROWS),
                ),
            )
        return (
            triton_fp8_splitk_combine_kernel,
            common_args + (COMBINE_BLOCK_SPLITS, COMBINE_BLOCK_D),
            (
                batch_size * self.h_q,
                math.ceil(D_CKV / COMBINE_BLOCK_D),
            ),
        )

    def ensure_compiled_launch_pack(self, out, lse) -> None:
        # Exact pointer identity preserves every alignment specialization that
        # the public caller-provided output contract previously admitted. Keep
        # only the most recent pack so workloads that rotate output buffers do
        # not grow an unbounded launcher cache.
        key = (
            int(out.data_ptr()),
            int(lse.data_ptr()),
            self.use_full_tail_specialization(),
            self.use_merged_state_v_completion(),
            self.use_pretranspose_v1(),
            self.use_fixed_ten_page_v1(),
            self.use_fixed_two_page_v1(),
        )
        if key == self.launch_pack_key:
            self.launch_pack_reuses += 1
            return

        target_out = out if self.direct_single_output else self.partial_out
        direct_lse = self.use_direct_lse_v2()
        target_lse = lse if direct_lse else self.partial_lse2
        use_pdl = self.use_programmatic_dependent_launch()
        partial_grid = (
            int(self.meta.split_batch.numel()) * math.ceil(self.h_q / TLE_FP8_BH),
        )
        pretranspose_v1 = not use_pdl and self.use_pretranspose_v1()
        partial_runner, partial_args = prepare_compiled_runner(
            fp8_dense_mla_splitk_partial,
            self.partial_launch_args(target_out, target_lse)
            + (use_pdl, pretranspose_v1),
            partial_grid,
            launch_pdl=False,
        )
        if direct_lse:
            aux_runner, aux_bound_args = None, None
        else:
            aux_jit, aux_args, aux_grid = self.aux_launch_spec(out, lse)
            aux_runner, aux_bound_args = prepare_compiled_runner(
                aux_jit,
                aux_args,
                aux_grid,
                num_warps=(8 if aux_jit is triton_fp8_coarse_combine_kernel else 4),
                launch_pdl=use_pdl,
            )
        self.partial_compiled_runner = partial_runner
        self.partial_compiled_args = partial_args
        self.aux_compiled_runner = aux_runner
        self.aux_compiled_args = aux_bound_args
        self.launch_pack_key = key
        self.launch_pack_reuses = 0
        self.cuda_graph_key = None
        self.cuda_graph = None
        self.cuda_graph_capture_stream = None
        self.cuda_graph_eligible = not use_pdl

    def ensure_graph_replay(self):
        """Capture the stable two-kernel prepared replay after one pointer hit."""
        # An outer graph must capture the launches, not replay an inner graph.
        if torch.cuda.is_current_stream_capturing():
            return None
        key = self.launch_pack_key
        if key is None or not self.cuda_graph_eligible or self.launch_pack_reuses < 1:
            return None
        if self.cuda_graph_key == key and self.cuda_graph is not None:
            return self.cuda_graph

        current_stream = torch.cuda.current_stream(self.out.device)
        capture_stream = torch.cuda.Stream(device=self.out.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            self.partial_compiled_runner(*self.partial_compiled_args)
            if self.aux_compiled_runner is not None:
                self.aux_compiled_runner(*self.aux_compiled_args)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            self.partial_compiled_runner(*self.partial_compiled_args)
            if self.aux_compiled_runner is not None:
                self.aux_compiled_runner(*self.aux_compiled_args)
        current_stream.wait_stream(capture_stream)
        self.cuda_graph_key = key
        self.cuda_graph = graph
        self.cuda_graph_capture_stream = capture_stream
        return graph

    def claim(self) -> None:
        if self.in_use:
            raise RuntimeError("prepared handle is already in use")
        self.in_use = True

    def validate_length_bounds(self, values) -> None:
        for batch_index, (value, previous, capacity) in enumerate(
            zip(values, self.cache_seqlens_host, self.max_cache_seqlens)
        ):
            if value < previous:
                raise RuntimeError(f"cache_seqlens[{batch_index}] must be monotonic")
            if value > capacity:
                raise RuntimeError(
                    f"cache_seqlens[{batch_index}] {value} exceeds prepared "
                    f"capacity {capacity}"
                )

    def apply_length_certificate(
        self,
        cache_seqlens,
        *,
        require_version_change: bool,
    ) -> None:
        values = host_certificate_lengths(
            cache_seqlens,
            "cache_seqlens",
            batch_size=len(self.cache_seqlens_host),
        )
        self.validate_length_bounds(values)

        version = tensor_version(self.cache_seqlens)
        changed = values != self.cache_seqlens_host
        if changed and require_version_change and version == self.cache_version:
            raise RuntimeError(
                "cache_seqlens storage did not receive an observable in-place "
                "PyTorch update before the new host certificate"
            )
        if not changed and version != self.cache_version:
            raise RuntimeError(
                "cache_seqlens changed without a new host length certificate"
            )

        pages, logical_active_splits = length_page_state(
            values,
            int(self.meta.adaptive_fixed_pages),
        )
        for batch_index, (logical, capacity) in enumerate(
            zip(logical_active_splits, self.meta.capacity_splits)
        ):
            if logical > capacity:
                raise RuntimeError(
                    f"logical split count for batch {batch_index} exceeds capacity"
                )

        self.cache_seqlens_host = values
        self.num_pages = pages
        self.logical_active_splits = logical_active_splits
        self.cache_version = version

    def set_cache_seqlens_(self, cache_seqlens: Sequence[int]) -> None:
        """Own a monotonic in-place length update on the bound CUDA stream."""
        values = host_certificate_lengths(
            cache_seqlens,
            "cache_seqlens",
            batch_size=len(self.cache_seqlens_host),
        )
        self.validate_length_bounds(values)
        self.claim()
        try:
            if values != self.cache_seqlens_host:
                write_cache_lengths[(len(values),)](
                    self.cache_seqlens,
                    values,
                    BATCH=len(values),
                    num_warps=4,
                )
            self.apply_length_certificate(
                values,
                require_version_change=False,
            )
        finally:
            self.in_use = False

    def validate_output(self, tensor: torch.Tensor, *, lse: bool) -> None:
        template = self.lse if lse else self.out
        label = "lse" if lse else "out"
        if tuple(tensor.shape) != tuple(template.shape):
            raise RuntimeError(
                f"{label} shape must be {tuple(template.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != template.dtype or tensor.device != template.device:
            raise RuntimeError(f"{label} dtype/device mismatch")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{label} must be contiguous")

    def launch(
        self,
        *,
        cache_seqlens: Sequence[int] | None = None,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Submit one prepared decode step using a host length certificate."""
        self.claim()
        try:
            if cache_seqlens is None:
                if tensor_version(self.cache_seqlens) != self.cache_version:
                    raise RuntimeError(
                        "cache_seqlens changed without a host length certificate"
                    )
            else:
                self.apply_length_certificate(
                    cache_seqlens,
                    require_version_change=True,
                )

            if (out is None) != (lse is None):
                raise RuntimeError(
                    "out and lse must either both be supplied or both omitted"
                )
            if out is None:
                out = torch.empty_like(self.out)
                lse = torch.empty_like(self.lse)
            else:
                self.validate_output(out, lse=False)
                self.validate_output(lse, lse=True)

            self.ensure_compiled_launch_pack(out, lse)
            graph = self.ensure_graph_replay()
            if graph is None:
                self.partial_compiled_runner(*self.partial_compiled_args)
                if self.aux_compiled_runner is not None:
                    self.aux_compiled_runner(*self.aux_compiled_args)
            else:
                graph.replay()
            return out, lse
        finally:
            self.in_use = False

    __call__ = launch


PACK_CONTENT_DIM = tl.constexpr(D_CKV)

if HAS_TLE:

    @triton.jit
    def unpack_dense_output(
        PackedOutput,
        PackedLSE,
        Output,
        LSE,
        QUERIES: tl.constexpr,
        HEADS: tl.constexpr,
        PADDED: tl.constexpr,
        TOTAL: tl.constexpr,
        COPY_OUTPUT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = offsets // PACK_CONTENT_DIM if COPY_OUTPUT else offsets
        group = row // HEADS
        head = row % HEADS
        batch = group // QUERIES
        query = group % QUERIES
        source = batch * PADDED + query * HEADS + head
        lse_mask = row < TOTAL
        if COPY_OUTPUT:
            dims = offsets % PACK_CONTENT_DIM
            values = tl.load(PackedOutput + source * PACK_CONTENT_DIM + dims)
            tl.store(Output + offsets, values)
            lse_mask = lse_mask & (dims == 0)
        lse = tl.load(PackedLSE + source, lse_mask, 0.0)
        tl.store(LSE + (batch * HEADS + head) * QUERIES + query, lse, lse_mask)


def validate_dense_query_layout(query: torch.Tensor) -> None:
    if query.ndim != 4 or query.shape[1] not in (1, 2):
        raise NotImplementedError("dense MLA supports one or two query tokens")
    if query.shape[2] not in (16, 32, 64, 128):
        raise NotImplementedError("dense MLA supports 16/32/64/128 query heads")


class FlashMLAFp8PackedHandle:
    """Reuse the 64-head kernel for small-head and multi-token upstream decode."""

    def __init__(
        self,
        query: torch.Tensor,
        rope: torch.Tensor,
        scale: torch.Tensor,
        table: torch.Tensor,
        lengths: torch.Tensor,
        core: FlashMLAFp8PreparedHandle,
        packed: tuple[torch.Tensor, ...],
        causal: bool,
    ) -> None:
        self.query, self.rope, self.scale = query, rope, scale
        self.table, self.lengths = table, lengths
        self.length_version = tensor_version(lengths)
        self.core, self.packed = core, packed
        self.batch, self.queries, self.heads, _ = query.shape
        self.merged_heads = packed[0].shape[2]
        self.causal = causal
        self.out = torch.empty(
            (self.batch, self.queries, self.heads, D_CKV),
            dtype=torch.bfloat16,
            device=query.device,
        )
        self.lse = torch.empty(
            (self.batch, self.heads, self.queries),
            dtype=torch.float32,
            device=query.device,
        )
        self.graph_key = None
        self.graph = None
        self.unpack_compiled_runner = None
        self.unpack_compiled_args = None

    def launch_compiled(self):
        self.core.partial_compiled_runner(*self.core.partial_compiled_args)
        if self.core.aux_compiled_runner is not None:
            self.core.aux_compiled_runner(*self.core.aux_compiled_args)
        self.unpack_compiled_runner(*self.unpack_compiled_args)

    def launch(
        self,
        *,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (out is None) != (lse is None):
            raise ValueError("out and lse must be supplied together")
        if out is None:
            out = torch.empty_like(self.out)
            lse = torch.empty_like(self.lse)
        else:
            FlashMLAFp8PreparedHandle.validate_output(self, out, lse=False)
            FlashMLAFp8PreparedHandle.validate_output(self, lse, lse=True)
        if (
            self.length_version < 0
            or tensor_version(self.lengths) != self.length_version
        ):
            self.core.has_length_certificate = False
        key = (out.data_ptr(), lse.data_ptr(), self.core.has_length_certificate)
        if torch.cuda.is_current_stream_capturing():
            if key != self.graph_key:
                raise RuntimeError(
                    "Warm the prepared handle with these output buffers before capture"
                )
            self.launch_compiled()
            return out, lse
        if key != self.graph_key:
            copy_output = out.data_ptr() % 16 != 0
            core_out = (
                self.core.out
                if copy_output
                else out.view(self.batch, 1, self.merged_heads, D_CKV)
            )
            output_rows = self.batch * self.queries * self.heads
            unpack_runner, unpack_args = prepare_compiled_runner(
                unpack_dense_output,
                (
                    core_out,
                    self.core.lse,
                    out,
                    lse,
                    self.queries,
                    self.heads,
                    self.merged_heads,
                    output_rows,
                    copy_output,
                    4 * D_CKV if copy_output else D_CKV,
                ),
                (output_rows // 4 if copy_output else triton.cdiv(output_rows, D_CKV),),
            )
            self.unpack_compiled_runner = unpack_runner
            self.unpack_compiled_args = unpack_args
            self.core.ensure_compiled_launch_pack(core_out, self.core.lse)
            stream = torch.cuda.Stream(device=out.device)
            stream.wait_stream(torch.cuda.current_stream(out.device))
            with torch.cuda.stream(stream):
                self.launch_compiled()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.launch_compiled()
            torch.cuda.current_stream(out.device).wait_stream(stream)
            self.graph, self.graph_key = graph, key
        self.graph.replay()
        return out, lse

    __call__ = launch


def prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    pages_per_split: int = DEFAULT_PAGES_PER_SPLIT,
    max_splits: Optional[int] = None,
    *,
    initial_cache_seqlens: Sequence[int],
    max_cache_seqlens: Sequence[int],
    has_length_certificate: bool = True,
) -> tuple[
    FlashMLAFp8PreparedHandle | FlashMLAFp8PackedHandle,
    tuple[torch.Tensor, torch.Tensor],
]:
    if not HAS_TLE:
        raise NotImplementedError("FP8 MLA requires NVIDIA Hopper and TLE")
    validate_dense_query_layout(q_nope)
    validate_dense_inputs(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
    )
    batch, queries, heads, _ = q_nope.shape
    if queries == 1 and heads >= TLE_FP8_BH:
        return prepare_dense_decode_core(
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            block_table,
            cache_seqlens,
            head_dim_v,
            softmax_scale=softmax_scale,
            pages_per_split=pages_per_split,
            max_splits=max_splits,
            initial_cache_seqlens=initial_cache_seqlens,
            max_cache_seqlens=max_cache_seqlens,
            has_length_certificate=has_length_certificate,
        )
    if not block_table.is_contiguous():
        raise NotImplementedError("packed decode requires a contiguous block table")
    initial_cache_seqlens = host_certificate_lengths(
        initial_cache_seqlens,
        "initial_cache_seqlens",
        batch_size=batch,
    )
    max_cache_seqlens = host_certificate_lengths(
        max_cache_seqlens,
        "max_cache_seqlens",
        batch_size=batch,
    )
    merged_heads = queries * heads
    packed_query = q_nope.view(batch, 1, merged_heads, D_CKV)
    packed_rope = q_rope.view(batch, 1, merged_heads, D_ROPE)
    packed_scale = q_scale.view(batch, 1, merged_heads, 1)
    packed_table, packed_lengths = block_table, cache_seqlens
    packed = (packed_query, packed_rope, packed_scale, packed_table, packed_lengths)
    initial = initial_cache_seqlens
    capacity = max_cache_seqlens
    core, _ = prepare_dense_decode_core(
        packed_query,
        packed_rope,
        k_cache_lora,
        k_cache_rope,
        packed_scale,
        k_scale,
        packed_table,
        packed_lengths,
        head_dim_v,
        softmax_scale=softmax_scale,
        pages_per_split=pages_per_split,
        max_splits=max_splits,
        initial_cache_seqlens=initial,
        max_cache_seqlens=capacity,
        has_length_certificate=has_length_certificate,
        query_heads=heads,
        query_count=queries if causal else 1,
    )
    handle = FlashMLAFp8PackedHandle(
        q_nope, q_rope, q_scale, block_table, cache_seqlens, core, packed, causal
    )
    return handle, handle(out=handle.out, lse=handle.lse)


def flash_mla_with_kvcache_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLAFp8SplitKSchedMeta | None = None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TLE:
        raise NotImplementedError("FP8 MLA requires NVIDIA Hopper and TLE")
    validate_dense_query_layout(q_nope)
    validate_dense_inputs(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
    )
    # Fold the query tokens into the head axis so the 64-head kernel serves
    # small-head and multi-token decode; causal masks use QUERY_HEADS/COUNT.
    batch, queries, heads, _ = q_nope.shape
    merged_heads = queries * heads
    if tile_scheduler_metadata is None or not tile_scheduler_metadata.have_initialized:
        tile_scheduler_metadata, num_splits = get_mla_fp8_metadata(
            cache_seqlens, merged_heads
        )
    elif (
        tile_scheduler_metadata.batch_size != batch
        or tile_scheduler_metadata.query_heads != merged_heads
    ):
        raise ValueError("tile_scheduler_metadata was planned for another shape")
    else:
        pass
    if num_splits is not None and num_splits is not tile_scheduler_metadata.num_splits:
        raise ValueError("num_splits must come from tile_scheduler_metadata")
    if out is None:
        out = torch.empty(
            (batch, queries, heads, head_dim_v),
            dtype=q_rope.dtype,
            device=q_nope.device,
        )
    if lse is None:
        lse = torch.empty(
            (batch, heads, queries), dtype=torch.float32, device=q_nope.device
        )
    for tensor, shape, dtype, label in (
        (out, (batch, queries, heads, head_dim_v), q_rope.dtype, "out"),
        (lse, (batch, heads, queries), torch.float32, "lse"),
    ):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{label} must have shape {shape} and dtype {dtype}")
        if tensor.device != q_nope.device or not tensor.is_contiguous():
            raise ValueError(f"{label} must be contiguous on the query device")
    launch_length_scheduled_decode(
        q_nope.view(batch, 1, merged_heads, D_CKV),
        q_rope.view(batch, 1, merged_heads, D_ROPE),
        q_scale.view(batch, 1, merged_heads, 1),
        k_cache_lora,
        k_cache_rope,
        k_scale,
        block_table,
        cache_seqlens,
        tile_scheduler_metadata,
        out.view(batch, 1, merged_heads, head_dim_v),
        lse,
        float(softmax_scale if softmax_scale is not None else D_QK**-0.5),
        heads,
        queries if causal else 1,
    )
    return out, lse

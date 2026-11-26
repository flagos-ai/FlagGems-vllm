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
from typing import Optional

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language import core

HEAD_DIM_NOPE = 512
HEAD_DIM_ROPE = 64
HEAD_DIM_TOTAL = HEAD_DIM_NOPE + HEAD_DIM_ROPE
PAGE_BLOCK_SIZE = 64
LOG2_E = tl.constexpr(1.4426950408889634)
LN_2 = tl.constexpr(0.6931471805599453)
# Leave headroom below 127.0 so round-to-nearest cannot overflow signed int8.
P7_INV_MAX_QUANT = tl.constexpr(1.0 / 126.99)
MIN_PROBABILITY_SCALE = tl.constexpr(1.0e-30)
MAX_GPU_SORT_BATCH = 2048


@core.extern
def native_rcpf(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(core.float32,): ("__builtin_ppu_rcpf", core.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


# PPU inline assembly rounds 4 FP32 probabilities in [0, 127] to signed int8
# and packs them into one i32 via ppu.prmt.b32 without scalar extraction loops.
P7_NORMALIZED_PACKED_ASM = tl.constexpr(
    """{
.reg .s32 q<4>;
.reg .f32 r<4>;
.reg .b32 pair0, pair1;
ppu.add.rn.f32 r0, $1, 0f3F000000;
ppu.cvt.rmi.s32.f32 q0, r0;
ppu.add.rn.f32 r1, $2, 0f3F000000;
ppu.cvt.rmi.s32.f32 q1, r1;
ppu.add.rn.f32 r2, $3, 0f3F000000;
ppu.cvt.rmi.s32.f32 q2, r2;
ppu.add.rn.f32 r3, $4, 0f3F000000;
ppu.cvt.rmi.s32.f32 q3, r3;
ppu.prmt.b32 pair0, q0, q1, 0x5410;
ppu.prmt.b32 pair1, q2, q3, 0x5410;
ppu.prmt.b32 $0, pair0, pair1, 0x6420;
}"""
)


@triton.jit
def quantize_mla_probabilities_p7(normalized_probabilities):
    return tl.inline_asm_elementwise(
        P7_NORMALIZED_PACKED_ASM,
        "=r,f,f,f,f",
        [normalized_probabilities],
        dtype=tl.int8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def sum_and_max(sum_left, max_left, sum_right, max_right):
    return sum_left + sum_right, tl.maximum(max_left, max_right)


@triton.constexpr_function
def swizzled_2d(vec, max_phase):
    return tle.gpu.swizzled_shared_layout(
        vec, 1, max_phase, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]
    )


@triton.jit
def sort_requests_kernel(Lengths, Order, batch_size, N: tl.constexpr):
    offs = tl.arange(0, N)
    mask = offs < batch_size
    lens = tl.load(Lengths + offs, mask=mask, other=-1)
    first = tl.load(Lengths)
    # Uniform batches skip bitonic sort to avoid unnecessary register traffic.
    all_eq = tl.sum(tl.where(mask & (lens != first), 1, 0)) == 0
    if all_eq:
        tl.store(Order + offs, offs.to(tl.int32), mask=mask)
    else:
        keys = (lens.to(tl.int64) << 32) | (N - 1 - offs).to(tl.int64)
        sorted_keys = tl.sort(keys, descending=True)
        order = (N - 1 - (sorted_keys & 0xFFFFFFFF)).to(tl.int32)
        tl.store(Order + offs, order, mask=mask)


@triton.jit
def prep_flat_q2_kernel(
    Lengths,
    FlatOrder,
    EffLengths,
    batch_size,
    CAUSAL: tl.constexpr,
    SORT_REQUESTS: tl.constexpr,
    N: tl.constexpr,
):
    offs = tl.arange(0, N)
    mask = offs < batch_size
    lens = tl.load(Lengths + offs, mask=mask, other=-1)
    if SORT_REQUESTS:
        first = tl.load(Lengths)
        all_eq = tl.sum(tl.where(mask & (lens != first), 1, 0)) == 0
        if all_eq:
            order = offs.to(tl.int32)
        else:
            keys = (lens.to(tl.int64) << 32) | (N - 1 - offs).to(tl.int64)
            sorted_keys = tl.sort(keys, descending=True)
            order = (N - 1 - (sorted_keys & 0xFFFFFFFF)).to(tl.int32)
    else:
        order = offs.to(tl.int32)
    # Place q=1 and q=0 of the same request adjacently so their KV pages reuse L2.
    tl.store(FlatOrder + offs * 2, order * 2 + 1, mask=mask)
    tl.store(FlatOrder + offs * 2 + 1, order * 2, mask=mask)
    len_q0 = tl.maximum(lens - 1, 0) if CAUSAL else lens
    tl.store(EffLengths + offs * 2, len_q0, mask=mask)
    tl.store(EffLengths + offs * 2 + 1, lens, mask=mask)


@triton.jit
def dense_int8_mla_tle_kernel(
    Q,
    QR,
    KV,
    KR,
    KS,
    QS,
    Table,
    Lengths,
    BatchOrder,
    Out,
    Lse,
    Q_STRIDE0: tl.constexpr,
    Q_STRIDE1: tl.constexpr,
    Q_STRIDE2: tl.constexpr,
    QR_STRIDE0: tl.constexpr,
    QR_STRIDE1: tl.constexpr,
    QR_STRIDE2: tl.constexpr,
    KV_STRIDE0: tl.constexpr,
    KV_STRIDE1: tl.constexpr,
    KR_STRIDE0: tl.constexpr,
    KR_STRIDE1: tl.constexpr,
    QS_STRIDE0: tl.constexpr,
    QS_STRIDE1: tl.constexpr,
    QS_STRIDE2: tl.constexpr,
    KS_STRIDE0: tl.constexpr,
    KS_STRIDE1: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    batch_size,
    total_heads,
    heads_per_query,
    softmax_scale,
    NUM_SPLITS: tl.constexpr,
    USE_BATCH_ORDER: tl.constexpr,
    CAUSAL: tl.constexpr,
    SINGLE_QUERY: tl.constexpr,
    HEADS_ALIGNED: tl.constexpr,
    FLAT_Q2: tl.constexpr,
    CONTIG_Q2_H16: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    BH: tl.constexpr = 32
    # Grid = (head_blocks, num_splits, batch) schedules longest requests first
    # while keeping head tiles of the same (split, batch) co-resident for L2 reuse.
    head_block = tl.program_id(0)
    split_idx = tl.program_id(1)
    raw_batch = tl.program_id(2)
    batch = (
        tl.load(BatchOrder + raw_batch).to(tl.int32) if USE_BATCH_ORDER else raw_batch
    )
    tbl_batch = (batch >> 1) if FLAT_Q2 else batch
    head_base = head_block * BH

    length = tl.load(Lengths + batch)
    total_pages = (length + 63) // 64
    if NUM_SPLITS > 1:
        total_tiles = (total_pages + 1) // 2
        tiles_per_split = (total_tiles + NUM_SPLITS - 1) // NUM_SPLITS
        start_page = split_idx * tiles_per_split * 2
        end_page = tl.minimum(start_page + tiles_per_split * 2, total_pages)
    else:
        start_page = 0
        end_page = total_pages

    # 104 KiB SMEM keeps 2 eight-warp CTAs resident per SM; maxPhase=32 matches
    # the 512-byte row stride of Q and KV to avoid TSM bank conflicts.
    q_buf = tle.gpu.alloc(
        [BH, 512], dtype=tl.int8, scope=tle.gpu.smem, layout=swizzled_2d(16, 32)
    )
    q_pe_buf = tle.gpu.alloc(
        [BH, 64], dtype=OUT_DTYPE, scope=tle.gpu.smem, layout=swizzled_2d(8, 8)
    )
    kv_buf = tle.gpu.alloc(
        [128, 512], dtype=tl.int8, scope=tle.gpu.smem, layout=swizzled_2d(16, 32)
    )
    kr_buf = tle.gpu.alloc(
        [128, 64], dtype=OUT_DTYPE, scope=tle.gpu.smem, layout=swizzled_2d(8, 8)
    )
    p_buf = tle.gpu.alloc(
        [BH, 128], dtype=tl.int8, scope=tle.gpu.smem, layout=swizzled_2d(16, 8)
    )
    rescale_buf = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem, layout=None)
    cscale_buf = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem, layout=None)

    flat_h_1d = head_base + tl.arange(0, BH)
    if HEADS_ALIGNED:
        safe_flat_1d = flat_h_1d
    else:
        head_mask_1d = flat_h_1d < total_heads
        safe_flat_1d = tl.where(head_mask_1d, flat_h_1d, 0)
    if SINGLE_QUERY or CONTIG_Q2_H16:
        q_idx_1d = 0
        h_idx_1d = safe_flat_1d
        if HEADS_ALIGNED:
            qs = tl.load(QS + batch * QS_STRIDE0 + h_idx_1d * QS_STRIDE2)
        else:
            qs = tl.load(
                QS + batch * QS_STRIDE0 + h_idx_1d * QS_STRIDE2,
                mask=head_mask_1d,
                other=0.0,
            )
    else:
        q_idx_1d = safe_flat_1d // heads_per_query
        h_idx_1d = safe_flat_1d % heads_per_query
        if HEADS_ALIGNED:
            qs = tl.load(
                QS + batch * QS_STRIDE0 + q_idx_1d * QS_STRIDE1 + h_idx_1d * QS_STRIDE2
            )
        else:
            qs = tl.load(
                QS + batch * QS_STRIDE0 + q_idx_1d * QS_STRIDE1 + h_idx_1d * QS_STRIDE2,
                mask=head_mask_1d,
                other=0.0,
            )
        q_len = total_heads // heads_per_query
        query_limit = (
            (length - (q_len - 1 - q_idx_1d))
            if CAUSAL
            else tl.full([BH], 0, dtype=tl.int32) + length
        )
    qs_log2 = qs * (softmax_scale * LOG2_E)

    flat_h = head_base + tl.arange(0, BH)[:, None]
    d512 = tl.arange(0, 512)[None, :]
    d256_l = tl.arange(0, 256)[None, :]
    d256_r = tl.arange(256, 512)[None, :]
    d64 = tl.arange(0, 64)[None, :]
    if HEADS_ALIGNED:
        if SINGLE_QUERY or CONTIG_Q2_H16:
            q_row = Q + batch * Q_STRIDE0 + flat_h * Q_STRIDE2
            qr_row = QR + batch * QR_STRIDE0 + flat_h * QR_STRIDE2
        else:
            q_idx = flat_h // heads_per_query
            h_idx = flat_h % heads_per_query
            q_row = Q + batch * Q_STRIDE0 + q_idx * Q_STRIDE1 + h_idx * Q_STRIDE2
            qr_row = QR + batch * QR_STRIDE0 + q_idx * QR_STRIDE1 + h_idx * QR_STRIDE2
        tle.gpu.copy(q_row + d512, q_buf, [BH, 512], is_async=True)
        tle.gpu.copy(qr_row + d64, q_pe_buf, [BH, 64], is_async=True)
    else:
        head_valid = flat_h < total_heads
        safe_flat = tl.where(head_valid, flat_h, 0)
        if SINGLE_QUERY:
            q_row = Q + batch * Q_STRIDE0 + safe_flat * Q_STRIDE2
            qr_row = QR + batch * QR_STRIDE0 + safe_flat * QR_STRIDE2
        else:
            q_idx = safe_flat // heads_per_query
            h_idx = safe_flat % heads_per_query
            q_row = Q + batch * Q_STRIDE0 + q_idx * Q_STRIDE1 + h_idx * Q_STRIDE2
            qr_row = QR + batch * QR_STRIDE0 + q_idx * QR_STRIDE1 + h_idx * QR_STRIDE2
        tle.gpu.copy(
            q_row + d512, q_buf, [BH, 512], mask=head_valid, other=0, is_async=True
        )
        tle.gpu.copy(
            qr_row + d64, q_pe_buf, [BH, 64], mask=head_valid, other=0, is_async=True
        )
    tle.gpu.async_commit_group()

    rows128 = tl.arange(0, 128)[:, None]
    is_p0_128 = rows128 < 64
    tok64_128 = rows128 & 63
    tokens = tl.arange(0, 128)
    is_p0_1d = tokens < 64
    tok64_1d = tokens & 63

    # Clamp to 0 so empty splits and zero-length requests skip the loop safely.
    n_tiles = tl.maximum((end_page - start_page + 1) // 2, 0)
    end_tok = tl.minimum(length, end_page * 64)
    maximum = tl.full([BH], -float("inf"), dtype=tl.float32)
    denominator = tl.zeros([BH], dtype=tl.float32)
    acc_l = tl.zeros([BH, 256], dtype=tl.float32)
    acc_r = tl.zeros([BH, 256], dtype=tl.float32)

    p0 = tl.load(
        Table + tbl_batch * TABLE_STRIDE + start_page,
        mask=start_page < end_page,
        other=0,
    ).to(tl.int64)
    p1 = tl.load(
        Table + tbl_batch * TABLE_STRIDE + start_page + 1,
        mask=(start_page + 1) < end_page,
        other=0,
    ).to(tl.int64)
    if n_tiles > 0:
        kv_base0 = (
            tl.where(is_p0_128, KV + p0 * KV_STRIDE0, KV + p1 * KV_STRIDE0)
            + tok64_128 * KV_STRIDE1
            + d512
        )
        v0 = (start_page * 64 + rows128) < end_tok
        tle.gpu.copy(kv_base0, kv_buf, [128, 512], mask=v0, other=0, is_async=True)
        tle.gpu.async_commit_group()

    if SINGLE_QUERY or CONTIG_Q2_H16:
        # Peel at most one partially valid trailing tile so the hot loop runs
        # without KR/KS masks or softmax tail branches while staying NaN-safe.
        eff_end_tok = (
            end_tok
            if (SINGLE_QUERY or (not CAUSAL))
            else tl.minimum(length - 1, end_page * 64)
        )
        n_full = tl.maximum((eff_end_tok - start_page * 64) // 128, 0)

        for t in range(n_full):
            page_idx = start_page + t * 2
            nxt_page = page_idx + 2
            rem_nxt = end_tok - nxt_page * 64

            tle.gpu.async_wait_group(0)
            kr_ptrs = (
                tl.where(is_p0_128, KR + p0 * KR_STRIDE0, KR + p1 * KR_STRIDE0)
                + tok64_128 * KR_STRIDE1
                + d64
            )
            ks_ptrs = (
                tl.where(is_p0_1d, KS + p0 * KS_STRIDE0, KS + p1 * KS_STRIDE0)
                + tok64_1d * KS_STRIDE1
            )
            tle.gpu.copy(kr_ptrs, kr_buf, [128, 64], is_async=True)
            tle.gpu.async_commit_group()
            ks_cur = tl.load(ks_ptrs)

            np0 = tl.load(
                Table + tbl_batch * TABLE_STRIDE + nxt_page,
                mask=nxt_page < end_page,
                other=0,
            ).to(tl.int64)
            np1 = tl.load(
                Table + tbl_batch * TABLE_STRIDE + nxt_page + 1,
                mask=(nxt_page + 1) < end_page,
                other=0,
            ).to(tl.int64)

            content_scores = tl.dot(
                tl.load(tle.gpu.local_ptr(q_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kv_buf))),
                out_dtype=tl.int32,
            ).to(tl.float32)

            tle.gpu.async_wait_group(0)
            qk = tl.dot(
                tl.load(tle.gpu.local_ptr(q_pe_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kr_buf))),
                content_scores,
                out_dtype=tl.float32,
            )

            scores = tl.trans(qk) * qs_log2[None, :] * ks_cur[:, None]
            new_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
            probabilities = tl.exp2(scores - new_maximum[None, :])
            rescale = tl.exp2(maximum - new_maximum)
            scaled_probabilities = probabilities * ks_cur[:, None]
            probability_sum, scaled_probability_max = tl.reduce(
                (probabilities, scaled_probabilities), 0, sum_and_max
            )
            denominator = denominator * rescale + probability_sum
            probability_scale = tl.maximum(
                scaled_probability_max * P7_INV_MAX_QUANT, MIN_PROBABILITY_SCALE
            )
            norm_p = tl.trans(
                scaled_probabilities * native_rcpf(probability_scale)[None, :]
            )
            p_q = quantize_mla_probabilities_p7(norm_p)
            tl.store(tle.gpu.local_ptr(p_buf), p_q)
            tl.store(tle.gpu.local_ptr(rescale_buf), rescale)
            tl.store(tle.gpu.local_ptr(cscale_buf), probability_scale)

            kvl = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows128, d256_l), shape=(128, 256))
            )
            pq_l = tl.load(tle.gpu.local_ptr(p_buf))
            rs_pv = tl.load(tle.gpu.local_ptr(rescale_buf))
            cs_pv = tl.load(tle.gpu.local_ptr(cscale_buf))
            hl = tl.dot(pq_l, kvl, out_dtype=tl.int32)
            acc_l = acc_l * rs_pv[:, None] + hl.to(tl.float32) * cs_pv[:, None]

            # Complete all kvl reads before launching the next async KV overwrite.
            tl.debug_barrier()

            kvr = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows128, d256_r), shape=(128, 256))
            )
            if nxt_page < end_page:
                nkv_ptrs = (
                    tl.where(is_p0_128, KV + np0 * KV_STRIDE0, KV + np1 * KV_STRIDE0)
                    + tok64_128 * KV_STRIDE1
                    + d512
                )
                tle.gpu.copy(
                    nkv_ptrs,
                    kv_buf,
                    [128, 512],
                    mask=rows128 < rem_nxt,
                    other=0,
                    is_async=True,
                )
                tle.gpu.async_commit_group()

            hr = tl.dot(pq_l, kvr, out_dtype=tl.int32)
            acc_r = acc_r * rs_pv[:, None] + hr.to(tl.float32) * cs_pv[:, None]

            maximum = new_maximum
            p0, p1 = np0, np1

        if n_full < n_tiles:
            tail_page = start_page + n_full * 2
            rem_cur = end_tok - tail_page * 64

            tle.gpu.async_wait_group(0)
            kr_ptrs = (
                tl.where(is_p0_128, KR + p0 * KR_STRIDE0, KR + p1 * KR_STRIDE0)
                + tok64_128 * KR_STRIDE1
                + d64
            )
            ks_ptrs = (
                tl.where(is_p0_1d, KS + p0 * KS_STRIDE0, KS + p1 * KS_STRIDE0)
                + tok64_1d * KS_STRIDE1
            )
            tle.gpu.copy(
                kr_ptrs,
                kr_buf,
                [128, 64],
                mask=rows128 < rem_cur,
                other=0,
                is_async=True,
            )
            tle.gpu.async_commit_group()
            ks_cur = tl.load(ks_ptrs, mask=tokens < rem_cur, other=0.0)

            content_scores = tl.dot(
                tl.load(tle.gpu.local_ptr(q_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kv_buf))),
                out_dtype=tl.int32,
            ).to(tl.float32)

            tle.gpu.async_wait_group(0)
            qk = tl.dot(
                tl.load(tle.gpu.local_ptr(q_pe_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kr_buf))),
                content_scores,
                out_dtype=tl.float32,
            )

            scores = tl.trans(qk) * qs_log2[None, :] * ks_cur[:, None]
            if SINGLE_QUERY or (not CAUSAL):
                visible = tokens[:, None] < rem_cur
            else:
                q_rem = (
                    tl.minimum(length - (flat_h_1d < 16).to(tl.int32), end_page * 64)
                    - tail_page * 64
                )
                visible = tokens[:, None] < q_rem[None, :]
            scores = tl.where(visible, scores, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
            safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
            probabilities = tl.exp2(scores - safe_maximum[None, :])
            rescale = tl.exp2(maximum - safe_maximum)
            scaled_probabilities = probabilities * ks_cur[:, None]
            probability_sum, scaled_probability_max = tl.reduce(
                (probabilities, scaled_probabilities), 0, sum_and_max
            )
            denominator = denominator * rescale + probability_sum
            probability_scale = tl.maximum(
                scaled_probability_max * P7_INV_MAX_QUANT, MIN_PROBABILITY_SCALE
            )
            norm_p = tl.trans(
                scaled_probabilities * native_rcpf(probability_scale)[None, :]
            )
            p_q = quantize_mla_probabilities_p7(norm_p)
            tl.store(tle.gpu.local_ptr(p_buf), p_q)
            tl.store(tle.gpu.local_ptr(rescale_buf), rescale)
            tl.store(tle.gpu.local_ptr(cscale_buf), probability_scale)

            kvl = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows128, d256_l), shape=(128, 256))
            )
            pq_l = tl.load(tle.gpu.local_ptr(p_buf))
            rs_pv = tl.load(tle.gpu.local_ptr(rescale_buf))
            cs_pv = tl.load(tle.gpu.local_ptr(cscale_buf))
            hl = tl.dot(pq_l, kvl, out_dtype=tl.int32)
            acc_l = acc_l * rs_pv[:, None] + hl.to(tl.float32) * cs_pv[:, None]

            tl.debug_barrier()

            kvr = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows128, d256_r), shape=(128, 256))
            )
            hr = tl.dot(pq_l, kvr, out_dtype=tl.int32)
            acc_r = acc_r * rs_pv[:, None] + hr.to(tl.float32) * cs_pv[:, None]

            maximum = new_maximum
    else:
        for t in range(n_tiles):
            page_idx = start_page + t * 2
            nxt_page = page_idx + 2
            rem_cur = end_tok - page_idx * 64

            tle.gpu.async_wait_group(0)
            kr_ptrs = (
                tl.where(is_p0_128, KR + p0 * KR_STRIDE0, KR + p1 * KR_STRIDE0)
                + tok64_128 * KR_STRIDE1
                + d64
            )
            ks_ptrs = (
                tl.where(is_p0_1d, KS + p0 * KS_STRIDE0, KS + p1 * KS_STRIDE0)
                + tok64_1d * KS_STRIDE1
            )
            tle.gpu.copy(
                kr_ptrs,
                kr_buf,
                [128, 64],
                mask=rows128 < rem_cur,
                other=0,
                is_async=True,
            )
            tle.gpu.async_commit_group()
            ks_cur = tl.load(ks_ptrs, mask=tokens < rem_cur, other=0.0)

            np0 = tl.load(
                Table + tbl_batch * TABLE_STRIDE + nxt_page,
                mask=nxt_page < end_page,
                other=0,
            ).to(tl.int64)
            np1 = tl.load(
                Table + tbl_batch * TABLE_STRIDE + nxt_page + 1,
                mask=(nxt_page + 1) < end_page,
                other=0,
            ).to(tl.int64)

            content_scores = tl.dot(
                tl.load(tle.gpu.local_ptr(q_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kv_buf))),
                out_dtype=tl.int32,
            ).to(tl.float32)

            tle.gpu.async_wait_group(0)
            qk = tl.dot(
                tl.load(tle.gpu.local_ptr(q_pe_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kr_buf))),
                content_scores,
                out_dtype=tl.float32,
            )

            scores = tl.trans(qk) * qs_log2[None, :] * ks_cur[:, None]
            if page_idx + 2 >= end_page - 1:
                page_offsets = page_idx + tokens // 64
                visible = (page_idx * 64 + tokens[:, None] < query_limit[None, :]) & (
                    page_offsets[:, None] < end_page
                )
                scores = tl.where(visible, scores, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
            safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
            probabilities = tl.exp2(scores - safe_maximum[None, :])
            rescale = tl.exp2(maximum - safe_maximum)
            scaled_probabilities = probabilities * ks_cur[:, None]
            probability_sum, scaled_probability_max = tl.reduce(
                (probabilities, scaled_probabilities), 0, sum_and_max
            )
            denominator = denominator * rescale + probability_sum
            probability_scale = tl.maximum(
                scaled_probability_max * P7_INV_MAX_QUANT, MIN_PROBABILITY_SCALE
            )
            norm_p = tl.trans(
                scaled_probabilities * native_rcpf(probability_scale)[None, :]
            )
            p_q = quantize_mla_probabilities_p7(norm_p)
            tl.store(tle.gpu.local_ptr(p_buf), p_q)
            tl.store(tle.gpu.local_ptr(rescale_buf), rescale)
            tl.store(tle.gpu.local_ptr(cscale_buf), probability_scale)

            kvl = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows128, d256_l), shape=(128, 256))
            )
            pq_l = tl.load(tle.gpu.local_ptr(p_buf))
            rs_pv = tl.load(tle.gpu.local_ptr(rescale_buf))
            cs_pv = tl.load(tle.gpu.local_ptr(cscale_buf))
            hl = tl.dot(pq_l, kvl, out_dtype=tl.int32)
            acc_l = acc_l * rs_pv[:, None] + hl.to(tl.float32) * cs_pv[:, None]

            tl.debug_barrier()

            kvr = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows128, d256_r), shape=(128, 256))
            )
            if nxt_page < end_page:
                nkv_ptrs = (
                    tl.where(is_p0_128, KV + np0 * KV_STRIDE0, KV + np1 * KV_STRIDE0)
                    + tok64_128 * KV_STRIDE1
                    + d512
                )
                tle.gpu.copy(
                    nkv_ptrs,
                    kv_buf,
                    [128, 512],
                    mask=rows128 < (rem_cur - 128),
                    other=0,
                    is_async=True,
                )
                tle.gpu.async_commit_group()

            hr = tl.dot(pq_l, kvr, out_dtype=tl.int32)
            acc_r = acc_r * rs_pv[:, None] + hr.to(tl.float32) * cs_pv[:, None]

            maximum = new_maximum
            p0, p1 = np0, np1

    # Zero-visible rows have acc == 0, so replacing denom == 0 with 1.0 yields 0.0.
    valid_row = (
        (denominator > 0.0) if HEADS_ALIGNED else ((denominator > 0.0) & head_mask_1d)
    )
    safe_d = tl.where(valid_row, denominator, 1.0)
    inv_d = 1.0 / safe_d
    if NUM_SPLITS > 1:
        # -inf marks empty splits so merge_splits_kernel skips them.
        lse = tl.where(valid_row, maximum * LN_2 + tl.log(safe_d), -float("inf"))
        if FLAT_Q2:
            q_idx_s = batch & 1
            eff_flat_1d = q_idx_s * heads_per_query + flat_h_1d
            out_base = (
                Out
                + (
                    (tbl_batch * NUM_SPLITS + split_idx) * total_heads
                    + eff_flat_1d[:, None]
                )
                * 512
            )
            lse_ptrs = (
                Lse + (tbl_batch * NUM_SPLITS + split_idx) * total_heads + eff_flat_1d
            )
        else:
            out_base = (
                Out
                + ((batch * NUM_SPLITS + split_idx) * total_heads + flat_h_1d[:, None])
                * 512
            )
            lse_ptrs = Lse + (batch * NUM_SPLITS + split_idx) * total_heads + flat_h_1d
        if HEADS_ALIGNED:
            tl.store(out_base + d256_l, (acc_l * inv_d[:, None]).to(tl.float32))
            tl.store(out_base + d256_r, (acc_r * inv_d[:, None]).to(tl.float32))
            tl.store(lse_ptrs, lse)
        else:
            tl.store(
                out_base + d256_l,
                (acc_l * inv_d[:, None]).to(tl.float32),
                mask=head_mask_1d[:, None],
            )
            tl.store(
                out_base + d256_r,
                (acc_r * inv_d[:, None]).to(tl.float32),
                mask=head_mask_1d[:, None],
            )
            tl.store(lse_ptrs, lse, mask=head_mask_1d)
    else:
        # Match FlashMLA convention: +inf LSE when a row sees no valid tokens.
        lse = tl.where(valid_row, maximum * LN_2 + tl.log(safe_d), float("inf"))
        if FLAT_Q2:
            q_idx_s = batch & 1
            out_base = Out + (batch * heads_per_query + flat_h_1d[:, None]) * 512
            lse_ptrs = (
                Lse
                + tbl_batch
                + flat_h_1d * batch_size
                + q_idx_s * (batch_size * heads_per_query)
            )
        else:
            out_base = Out + (batch * total_heads + flat_h_1d[:, None]) * 512
            if SINGLE_QUERY:
                lse_ptrs = Lse + batch + h_idx_1d * batch_size
            elif CONTIG_Q2_H16:
                q_h16 = (flat_h_1d >= 16).to(tl.int32)
                h_h16 = flat_h_1d & 15
                lse_ptrs = Lse + batch + h_h16 * batch_size + q_h16 * (batch_size * 16)
            else:
                lse_ptrs = (
                    Lse
                    + batch
                    + h_idx_1d * batch_size
                    + q_idx_1d * (batch_size * heads_per_query)
                )
        if HEADS_ALIGNED:
            tl.store(out_base + d256_l, (acc_l * inv_d[:, None]).to(OUT_DTYPE))
            tl.store(out_base + d256_r, (acc_r * inv_d[:, None]).to(OUT_DTYPE))
            tl.store(lse_ptrs, lse)
        else:
            tl.store(
                out_base + d256_l,
                (acc_l * inv_d[:, None]).to(OUT_DTYPE),
                mask=head_mask_1d[:, None],
            )
            tl.store(
                out_base + d256_r,
                (acc_r * inv_d[:, None]).to(OUT_DTYPE),
                mask=head_mask_1d[:, None],
            )
            tl.store(lse_ptrs, lse, mask=head_mask_1d)


@triton.jit
def dense_int8_mla_tle_bh16_bk64_kernel(
    Q,
    QR,
    KV,
    KR,
    KS,
    QS,
    Table,
    Lengths,
    BatchOrder,
    Out,
    Lse,
    Q_STRIDE0: tl.constexpr,
    Q_STRIDE1: tl.constexpr,
    Q_STRIDE2: tl.constexpr,
    QR_STRIDE0: tl.constexpr,
    QR_STRIDE1: tl.constexpr,
    QR_STRIDE2: tl.constexpr,
    KV_STRIDE0: tl.constexpr,
    KV_STRIDE1: tl.constexpr,
    KR_STRIDE0: tl.constexpr,
    KR_STRIDE1: tl.constexpr,
    QS_STRIDE0: tl.constexpr,
    QS_STRIDE1: tl.constexpr,
    QS_STRIDE2: tl.constexpr,
    KS_STRIDE0: tl.constexpr,
    KS_STRIDE1: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    batch_size,
    total_heads,
    heads_per_query,
    softmax_scale,
    NUM_SPLITS: tl.constexpr,
    USE_BATCH_ORDER: tl.constexpr,
    CAUSAL: tl.constexpr,
    SINGLE_QUERY: tl.constexpr,
    HEADS_ALIGNED: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """High-occupancy 4-warp BK=64 kernel for BH=16 (51.7 KiB SMEM, 4 CTAs/SM)."""
    BH: tl.constexpr = 16
    head_block = tl.program_id(0)
    split_idx = tl.program_id(1)
    raw_batch = tl.program_id(2)
    batch = (
        tl.load(BatchOrder + raw_batch).to(tl.int32) if USE_BATCH_ORDER else raw_batch
    )
    head_base = head_block * BH

    length = tl.load(Lengths + batch)
    total_pages = (length + 63) // 64
    if NUM_SPLITS > 1:
        pages_per_split = (total_pages + NUM_SPLITS - 1) // NUM_SPLITS
        start_page = split_idx * pages_per_split
        end_page = tl.minimum(start_page + pages_per_split, total_pages)
    else:
        start_page = 0
        end_page = total_pages
    n_pages = tl.maximum(end_page - start_page, 0)
    end_tok = tl.minimum(length, end_page * 64)

    q_buf = tle.gpu.alloc(
        [BH, 512], dtype=tl.int8, scope=tle.gpu.smem, layout=swizzled_2d(16, 32)
    )
    q_pe_buf = tle.gpu.alloc(
        [BH, 64], dtype=OUT_DTYPE, scope=tle.gpu.smem, layout=swizzled_2d(8, 8)
    )
    kv_buf = tle.gpu.alloc(
        [64, 512], dtype=tl.int8, scope=tle.gpu.smem, layout=swizzled_2d(16, 32)
    )
    kr_buf = tle.gpu.alloc(
        [64, 64], dtype=OUT_DTYPE, scope=tle.gpu.smem, layout=swizzled_2d(8, 8)
    )
    p_buf = tle.gpu.alloc(
        [BH, 64], dtype=tl.int8, scope=tle.gpu.smem, layout=swizzled_2d(16, 4)
    )
    rescale_buf = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem, layout=None)
    cscale_buf = tle.gpu.alloc([BH], dtype=tl.float32, scope=tle.gpu.smem, layout=None)

    flat_h_1d = head_base + tl.arange(0, BH)
    if HEADS_ALIGNED:
        safe_flat_1d = flat_h_1d
    else:
        head_mask_1d = flat_h_1d < total_heads
        safe_flat_1d = tl.where(head_mask_1d, flat_h_1d, 0)
    if SINGLE_QUERY:
        q_idx_1d = 0
        h_idx_1d = safe_flat_1d
        if HEADS_ALIGNED:
            qs = tl.load(QS + batch * QS_STRIDE0 + h_idx_1d * QS_STRIDE2)
        else:
            qs = tl.load(
                QS + batch * QS_STRIDE0 + h_idx_1d * QS_STRIDE2,
                mask=head_mask_1d,
                other=0.0,
            )
    else:
        q_idx_1d = safe_flat_1d // heads_per_query
        h_idx_1d = safe_flat_1d % heads_per_query
        if HEADS_ALIGNED:
            qs = tl.load(
                QS + batch * QS_STRIDE0 + q_idx_1d * QS_STRIDE1 + h_idx_1d * QS_STRIDE2
            )
        else:
            qs = tl.load(
                QS + batch * QS_STRIDE0 + q_idx_1d * QS_STRIDE1 + h_idx_1d * QS_STRIDE2,
                mask=head_mask_1d,
                other=0.0,
            )
        q_len = total_heads // heads_per_query
        query_limit = (
            (length - (q_len - 1 - q_idx_1d))
            if CAUSAL
            else tl.full([BH], 0, dtype=tl.int32) + length
        )
    qs_log2 = qs * (softmax_scale * LOG2_E)

    flat_h = head_base + tl.arange(0, BH)[:, None]
    d512 = tl.arange(0, 512)[None, :]
    d256_l = tl.arange(0, 256)[None, :]
    d256_r = tl.arange(256, 512)[None, :]
    d64 = tl.arange(0, 64)[None, :]
    if HEADS_ALIGNED:
        if SINGLE_QUERY:
            q_row = Q + batch * Q_STRIDE0 + flat_h * Q_STRIDE2
            qr_row = QR + batch * QR_STRIDE0 + flat_h * QR_STRIDE2
        else:
            q_idx = flat_h // heads_per_query
            h_idx = flat_h % heads_per_query
            q_row = Q + batch * Q_STRIDE0 + q_idx * Q_STRIDE1 + h_idx * Q_STRIDE2
            qr_row = QR + batch * QR_STRIDE0 + q_idx * QR_STRIDE1 + h_idx * QR_STRIDE2
        tle.gpu.copy(q_row + d512, q_buf, [BH, 512], is_async=True)
        tle.gpu.copy(qr_row + d64, q_pe_buf, [BH, 64], is_async=True)
    else:
        head_valid = flat_h < total_heads
        safe_flat = tl.where(head_valid, flat_h, 0)
        if SINGLE_QUERY:
            q_row = Q + batch * Q_STRIDE0 + safe_flat * Q_STRIDE2
            qr_row = QR + batch * QR_STRIDE0 + safe_flat * QR_STRIDE2
        else:
            q_idx = safe_flat // heads_per_query
            h_idx = safe_flat % heads_per_query
            q_row = Q + batch * Q_STRIDE0 + q_idx * Q_STRIDE1 + h_idx * Q_STRIDE2
            qr_row = QR + batch * QR_STRIDE0 + q_idx * QR_STRIDE1 + h_idx * QR_STRIDE2
        tle.gpu.copy(
            q_row + d512, q_buf, [BH, 512], mask=head_valid, other=0, is_async=True
        )
        tle.gpu.copy(
            qr_row + d64, q_pe_buf, [BH, 64], mask=head_valid, other=0, is_async=True
        )
    tle.gpu.async_commit_group()

    rows64 = tl.arange(0, 64)[:, None]
    tok64 = tl.arange(0, 64)
    maximum = tl.full([BH], -float("inf"), dtype=tl.float32)
    denominator = tl.zeros([BH], dtype=tl.float32)
    acc_l = tl.zeros([BH, 256], dtype=tl.float32)
    acc_r = tl.zeros([BH, 256], dtype=tl.float32)

    p0 = tl.load(
        Table + batch * TABLE_STRIDE + start_page,
        mask=start_page < end_page,
        other=0,
    ).to(tl.int64)
    if n_pages > 0:
        v0 = (start_page * 64 + rows64) < end_tok
        tle.gpu.copy(
            KV + p0 * KV_STRIDE0 + rows64 * KV_STRIDE1 + d512,
            kv_buf,
            [64, 512],
            mask=v0,
            other=0,
            is_async=True,
        )
        tle.gpu.async_commit_group()

    if SINGLE_QUERY:
        n_full = tl.maximum((end_tok - start_page * 64) // 64, 0)
        for t in range(n_full):
            page_idx = start_page + t
            nxt_page = page_idx + 1
            rem_nxt = end_tok - nxt_page * 64
            tle.gpu.async_wait_group(0)
            tle.gpu.copy(
                KR + p0 * KR_STRIDE0 + rows64 * KR_STRIDE1 + d64,
                kr_buf,
                [64, 64],
                is_async=True,
            )
            tle.gpu.async_commit_group()
            ks_cur = tl.load(KS + p0 * KS_STRIDE0 + tok64 * KS_STRIDE1)
            np0 = tl.load(
                Table + batch * TABLE_STRIDE + nxt_page,
                mask=nxt_page < end_page,
                other=0,
            ).to(tl.int64)

            content_scores = tl.dot(
                tl.load(tle.gpu.local_ptr(q_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kv_buf))),
                out_dtype=tl.int32,
            ).to(tl.float32)
            tle.gpu.async_wait_group(0)
            qk = tl.dot(
                tl.load(tle.gpu.local_ptr(q_pe_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kr_buf))),
                content_scores,
                out_dtype=tl.float32,
            )

            scores = qk * qs_log2[:, None] * ks_cur[None, :]
            new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            probabilities = tl.exp2(scores - new_maximum[:, None])
            rescale = tl.exp2(maximum - new_maximum)
            scaled_probabilities = probabilities * ks_cur[None, :]
            probability_sum, scaled_probability_max = tl.reduce(
                (probabilities, scaled_probabilities), 1, sum_and_max
            )
            denominator = denominator * rescale + probability_sum
            probability_scale = tl.maximum(
                scaled_probability_max * P7_INV_MAX_QUANT, MIN_PROBABILITY_SCALE
            )
            norm_p = scaled_probabilities * native_rcpf(probability_scale)[:, None]
            p_q = quantize_mla_probabilities_p7(norm_p)
            tl.store(tle.gpu.local_ptr(p_buf), p_q)
            tl.store(tle.gpu.local_ptr(rescale_buf), rescale)
            tl.store(tle.gpu.local_ptr(cscale_buf), probability_scale)

            # Split PV into two 256-col subslices so all kv_buf loads finish
            # before the next async KV overwrite starts.
            kvl = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows64, d256_l), shape=(64, 256))
            )
            pq = tl.load(tle.gpu.local_ptr(p_buf))
            rs_pv = tl.load(tle.gpu.local_ptr(rescale_buf))
            cs_pv = tl.load(tle.gpu.local_ptr(cscale_buf))
            hl = tl.dot(pq, kvl, out_dtype=tl.int32)
            acc_l = acc_l * rs_pv[:, None] + hl.to(tl.float32) * cs_pv[:, None]
            kvr = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows64, d256_r), shape=(64, 256))
            )
            if nxt_page < end_page:
                tle.gpu.copy(
                    KV + np0 * KV_STRIDE0 + rows64 * KV_STRIDE1 + d512,
                    kv_buf,
                    [64, 512],
                    mask=rows64 < rem_nxt,
                    other=0,
                    is_async=True,
                )
                tle.gpu.async_commit_group()
            hr = tl.dot(pq, kvr, out_dtype=tl.int32)
            acc_r = acc_r * rs_pv[:, None] + hr.to(tl.float32) * cs_pv[:, None]

            maximum = new_maximum
            p0 = np0

        if n_full < n_pages:
            rem_cur = end_tok - (start_page + n_full) * 64
            tle.gpu.async_wait_group(0)
            tle.gpu.copy(
                KR + p0 * KR_STRIDE0 + rows64 * KR_STRIDE1 + d64,
                kr_buf,
                [64, 64],
                mask=rows64 < rem_cur,
                other=0,
                is_async=True,
            )
            tle.gpu.async_commit_group()
            ks_cur = tl.load(
                KS + p0 * KS_STRIDE0 + tok64 * KS_STRIDE1,
                mask=tok64 < rem_cur,
                other=0.0,
            )
            content_scores = tl.dot(
                tl.load(tle.gpu.local_ptr(q_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kv_buf))),
                out_dtype=tl.int32,
            ).to(tl.float32)
            tle.gpu.async_wait_group(0)
            qk = tl.dot(
                tl.load(tle.gpu.local_ptr(q_pe_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kr_buf))),
                content_scores,
                out_dtype=tl.float32,
            )
            scores = qk * qs_log2[:, None] * ks_cur[None, :]
            scores = tl.where(tok64[None, :] < rem_cur, scores, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
            probabilities = tl.exp2(scores - safe_maximum[:, None])
            rescale = tl.exp2(maximum - safe_maximum)
            scaled_probabilities = probabilities * ks_cur[None, :]
            probability_sum, scaled_probability_max = tl.reduce(
                (probabilities, scaled_probabilities), 1, sum_and_max
            )
            denominator = denominator * rescale + probability_sum
            probability_scale = tl.maximum(
                scaled_probability_max * P7_INV_MAX_QUANT, MIN_PROBABILITY_SCALE
            )
            norm_p = scaled_probabilities * native_rcpf(probability_scale)[:, None]
            p_q = quantize_mla_probabilities_p7(norm_p)
            tl.store(tle.gpu.local_ptr(p_buf), p_q)
            tl.store(tle.gpu.local_ptr(rescale_buf), rescale)
            tl.store(tle.gpu.local_ptr(cscale_buf), probability_scale)

            kvl = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows64, d256_l), shape=(64, 256))
            )
            pq = tl.load(tle.gpu.local_ptr(p_buf))
            rs_pv = tl.load(tle.gpu.local_ptr(rescale_buf))
            cs_pv = tl.load(tle.gpu.local_ptr(cscale_buf))
            hl = tl.dot(pq, kvl, out_dtype=tl.int32)
            acc_l = acc_l * rs_pv[:, None] + hl.to(tl.float32) * cs_pv[:, None]
            kvr = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows64, d256_r), shape=(64, 256))
            )
            hr = tl.dot(pq, kvr, out_dtype=tl.int32)
            acc_r = acc_r * rs_pv[:, None] + hr.to(tl.float32) * cs_pv[:, None]
            maximum = new_maximum
    else:
        for t in range(n_pages):
            page_idx = start_page + t
            nxt_page = page_idx + 1
            rem_cur = end_tok - page_idx * 64
            tle.gpu.async_wait_group(0)
            tle.gpu.copy(
                KR + p0 * KR_STRIDE0 + rows64 * KR_STRIDE1 + d64,
                kr_buf,
                [64, 64],
                mask=rows64 < rem_cur,
                other=0,
                is_async=True,
            )
            tle.gpu.async_commit_group()
            ks_cur = tl.load(
                KS + p0 * KS_STRIDE0 + tok64 * KS_STRIDE1,
                mask=tok64 < rem_cur,
                other=0.0,
            )
            np0 = tl.load(
                Table + batch * TABLE_STRIDE + nxt_page,
                mask=nxt_page < end_page,
                other=0,
            ).to(tl.int64)

            content_scores = tl.dot(
                tl.load(tle.gpu.local_ptr(q_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kv_buf))),
                out_dtype=tl.int32,
            ).to(tl.float32)
            tle.gpu.async_wait_group(0)
            qk = tl.dot(
                tl.load(tle.gpu.local_ptr(q_pe_buf)),
                tl.trans(tl.load(tle.gpu.local_ptr(kr_buf))),
                content_scores,
                out_dtype=tl.float32,
            )

            scores = qk * qs_log2[:, None] * ks_cur[None, :]
            if nxt_page >= end_page - 1:
                visible = (page_idx * 64 + tok64[None, :]) < query_limit[:, None]
                scores = tl.where(visible, scores, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
            probabilities = tl.exp2(scores - safe_maximum[:, None])
            rescale = tl.exp2(maximum - safe_maximum)
            scaled_probabilities = probabilities * ks_cur[None, :]
            probability_sum, scaled_probability_max = tl.reduce(
                (probabilities, scaled_probabilities), 1, sum_and_max
            )
            denominator = denominator * rescale + probability_sum
            probability_scale = tl.maximum(
                scaled_probability_max * P7_INV_MAX_QUANT, MIN_PROBABILITY_SCALE
            )
            norm_p = scaled_probabilities * native_rcpf(probability_scale)[:, None]
            p_q = quantize_mla_probabilities_p7(norm_p)
            tl.store(tle.gpu.local_ptr(p_buf), p_q)
            tl.store(tle.gpu.local_ptr(rescale_buf), rescale)
            tl.store(tle.gpu.local_ptr(cscale_buf), probability_scale)

            kvl = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows64, d256_l), shape=(64, 256))
            )
            pq = tl.load(tle.gpu.local_ptr(p_buf))
            rs_pv = tl.load(tle.gpu.local_ptr(rescale_buf))
            cs_pv = tl.load(tle.gpu.local_ptr(cscale_buf))
            hl = tl.dot(pq, kvl, out_dtype=tl.int32)
            acc_l = acc_l * rs_pv[:, None] + hl.to(tl.float32) * cs_pv[:, None]
            kvr = tl.load(
                tle.gpu.local_ptr(kv_buf, indices=(rows64, d256_r), shape=(64, 256))
            )
            if nxt_page < end_page:
                tle.gpu.copy(
                    KV + np0 * KV_STRIDE0 + rows64 * KV_STRIDE1 + d512,
                    kv_buf,
                    [64, 512],
                    mask=rows64 < (rem_cur - 64),
                    other=0,
                    is_async=True,
                )
                tle.gpu.async_commit_group()
            hr = tl.dot(pq, kvr, out_dtype=tl.int32)
            acc_r = acc_r * rs_pv[:, None] + hr.to(tl.float32) * cs_pv[:, None]

            maximum = new_maximum
            p0 = np0

    valid_row = (
        (denominator > 0.0) if HEADS_ALIGNED else ((denominator > 0.0) & head_mask_1d)
    )
    safe_d = tl.where(valid_row, denominator, 1.0)
    inv_d = 1.0 / safe_d
    if NUM_SPLITS > 1:
        lse = tl.where(valid_row, maximum * LN_2 + tl.log(safe_d), -float("inf"))
        out_base = (
            Out
            + ((batch * NUM_SPLITS + split_idx) * total_heads + flat_h_1d[:, None])
            * 512
        )
        lse_ptrs = Lse + (batch * NUM_SPLITS + split_idx) * total_heads + flat_h_1d
        if HEADS_ALIGNED:
            tl.store(out_base + d256_l, (acc_l * inv_d[:, None]).to(tl.float32))
            tl.store(out_base + d256_r, (acc_r * inv_d[:, None]).to(tl.float32))
            tl.store(lse_ptrs, lse)
        else:
            tl.store(
                out_base + d256_l,
                (acc_l * inv_d[:, None]).to(tl.float32),
                mask=head_mask_1d[:, None],
            )
            tl.store(
                out_base + d256_r,
                (acc_r * inv_d[:, None]).to(tl.float32),
                mask=head_mask_1d[:, None],
            )
            tl.store(lse_ptrs, lse, mask=head_mask_1d)
    else:
        lse = tl.where(valid_row, maximum * LN_2 + tl.log(safe_d), float("inf"))
        out_base = Out + (batch * total_heads + flat_h_1d[:, None]) * 512
        if SINGLE_QUERY:
            lse_ptrs = Lse + batch + h_idx_1d * batch_size
        else:
            lse_ptrs = (
                Lse
                + batch
                + h_idx_1d * batch_size
                + q_idx_1d * (batch_size * heads_per_query)
            )
        if HEADS_ALIGNED:
            tl.store(out_base + d256_l, (acc_l * inv_d[:, None]).to(OUT_DTYPE))
            tl.store(out_base + d256_r, (acc_r * inv_d[:, None]).to(OUT_DTYPE))
            tl.store(lse_ptrs, lse)
        else:
            tl.store(
                out_base + d256_l,
                (acc_l * inv_d[:, None]).to(OUT_DTYPE),
                mask=head_mask_1d[:, None],
            )
            tl.store(
                out_base + d256_r,
                (acc_r * inv_d[:, None]).to(OUT_DTYPE),
                mask=head_mask_1d[:, None],
            )
            tl.store(lse_ptrs, lse, mask=head_mask_1d)


@triton.jit
def merge_splits_kernel(
    PartOut,
    PartLse,
    Out,
    Lse,
    batch_size,
    total_heads,
    heads_per_query,
    NUM_SPLITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // total_heads
    flat_h = pid % total_heads
    q_idx = flat_h // heads_per_query
    h_idx = flat_h % heads_per_query
    d = tl.arange(0, 512)
    m = -float("inf")
    for s in tl.static_range(NUM_SPLITS):
        ls = tl.load(PartLse + (batch * NUM_SPLITS + s) * total_heads + flat_h)
        m = tl.maximum(m, ls)
    safe_m = tl.where(m == -float("inf"), 0.0, m)
    acc = tl.zeros([512], dtype=tl.float32)
    denom = 0.0
    for s in tl.static_range(NUM_SPLITS):
        ls = tl.load(PartLse + (batch * NUM_SPLITS + s) * total_heads + flat_h)
        w = tl.where(ls == -float("inf"), 0.0, tl.exp(ls - safe_m))
        po = tl.load(
            PartOut + ((batch * NUM_SPLITS + s) * total_heads + flat_h) * 512 + d
        )
        acc += po * w
        denom += w
    inv_d = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    lse = tl.where(denom > 0.0, safe_m + tl.log(denom), float("inf"))
    tl.store(
        Out + (batch * total_heads + flat_h) * 512 + d, (acc * inv_d).to(OUT_DTYPE)
    )
    tl.store(
        Lse + batch + h_idx * batch_size + q_idx * (batch_size * heads_per_query), lse
    )


def select_launch_config(
    batch: int,
    total_heads: int,
    table_pages: int,
    num_splits_override: Optional[int],
    is_varlen: bool,
    mean_pages: int,
    queries: int,
) -> tuple[int, int]:
    """Select (head_tile, num_splits) for 64-SM PPU-ZW810E occupancy."""
    bh = 16 if total_heads <= 16 else 32
    if num_splits_override is not None:
        return bh, num_splits_override

    head_blocks = triton.cdiv(total_heads, bh)
    base_ctas = head_blocks * batch

    if is_varlen:
        if bh == 32:
            if base_ctas <= 32 and table_pages >= 32:
                num_splits = 4
            elif base_ctas <= 64 and table_pages >= 16:
                num_splits = 2
            elif (
                base_ctas <= 128
                and queries == 2
                and total_heads == 32
                and mean_pages >= 96
            ):
                num_splits = 2
            else:
                num_splits = 1
        else:
            # 4-warp BK=64 fits 4 CTAs/SM (256 CTAs/wave), so S=4/8 fills 2-4 waves.
            if base_ctas <= 32 and table_pages >= 32:
                num_splits = 8
            elif base_ctas <= 128 and mean_pages >= 192:
                num_splits = 8
            elif base_ctas <= 128 and mean_pages >= 16:
                num_splits = 4
            else:
                num_splits = 1
    else:
        if bh == 16:
            if base_ctas <= 64 and table_pages >= 32:
                num_splits = 8
            elif base_ctas <= 128 and table_pages >= 192:
                num_splits = 8
            elif base_ctas <= 128 and table_pages >= 16:
                num_splits = 4
            else:
                num_splits = 1
        else:
            if base_ctas <= 32 and table_pages >= 32:
                num_splits = 8
            elif base_ctas <= 128 and table_pages >= 96:
                num_splits = 4
            elif base_ctas <= 128 and table_pages >= 16:
                num_splits = 2
            else:
                num_splits = 1

    min_pages_per_split = 2 if bh == 16 else 4
    while num_splits > 1 and table_pages < num_splits * min_pages_per_split:
        num_splits //= 2
    return bh, num_splits


class FlashMLAInt8PreparedHandle:
    """Zero-reallocation replay handle for PPU dense INT8 MLA decode."""

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        splits: int,
        bh: int,
        softmax_scale: float,
        causal: bool,
        sort_requests: bool,
    ):
        q, qr, kv, kr, qs, ks, table, lengths = tensors
        self.tensors = tensors
        self.splits = splits
        self.bh = bh
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.batch, self.queries, self.heads, _ = q.shape
        self.total_heads = self.queries * self.heads
        self.out_dtype_tl = tl.float16 if qr.dtype == torch.float16 else tl.bfloat16
        self.use_order = bool(sort_requests and 16 <= self.batch <= MAX_GPU_SORT_BATCH)

        self.out = torch.empty(
            (self.batch, self.queries, self.heads, HEAD_DIM_NOPE),
            dtype=qr.dtype,
            device=q.device,
        )
        self.lse = torch.empty_strided(
            (self.batch, self.heads, self.queries),
            (1, self.batch, self.batch * self.heads),
            dtype=torch.float32,
            device=q.device,
        )
        # When Q==2 and H is a multiple of BH=32 with contiguous layout, flatten
        # (B, 2) -> 2*B slots so Q=2 uses the single-query peeled hot loop.
        self.flat_q2 = bool(
            self.queries == 2
            and self.bh == 32
            and self.heads >= 32
            and (self.heads % 32) == 0
            and 0 < self.batch <= 1024
            and q.stride(0) == 2 * q.stride(1)
            and q.stride(1) == self.heads * q.stride(2)
            and qr.stride(0) == 2 * qr.stride(1)
            and qr.stride(1) == self.heads * qr.stride(2)
            and qs.stride(0) == 2 * qs.stride(1)
            and qs.stride(1) == self.heads * qs.stride(2)
            and self.out.stride(0) == 2 * self.out.stride(1)
            and self.out.stride(1) == self.heads * self.out.stride(2)
            and self.out.stride(2) == HEAD_DIM_NOPE
            and self.lse.stride(0) == 1
            and self.lse.stride(1) == self.batch
            and self.lse.stride(2) == self.batch * self.heads
        )
        self.contig_q2_h16 = bool(
            not self.flat_q2
            and self.queries == 2
            and self.heads == 16
            and self.bh == 32
            and q.stride(1) == 16 * q.stride(2)
            and qr.stride(1) == 16 * qr.stride(2)
            and qs.stride(1) == 16 * qs.stride(2)
        )
        if self.flat_q2:
            self.flat_order = torch.empty(
                (self.batch * 2,), dtype=torch.int32, device=q.device
            )
            self.eff_lengths = torch.empty(
                (self.batch * 2,), dtype=torch.int32, device=q.device
            )
            self.batch_order = None
        elif self.batch > 0 and self.total_heads > 0 and self.use_order:
            self.batch_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=q.device
            )
            self.flat_order = None
            self.eff_lengths = None
        else:
            self.batch_order = None
            self.flat_order = None
            self.eff_lengths = None

        if self.batch > 0 and self.total_heads > 0 and self.splits > 1:
            self.part_out = torch.empty(
                (self.batch, self.splits, self.total_heads, HEAD_DIM_NOPE),
                dtype=torch.float32,
                device=q.device,
            )
            self.part_lse = torch.empty(
                (self.batch, self.splits, self.total_heads),
                dtype=torch.float32,
                device=q.device,
            )
        else:
            self.part_out = None
            self.part_lse = None

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch == 0 or self.total_heads == 0:
            return self.out, self.lse
        q, qr, kv, kr, qs, ks, block_table, cache_seqlens = self.tensors
        if self.flat_q2:
            n_pow2 = triton.next_power_of_2(max(self.batch, 16))
            prep_flat_q2_kernel[(1,)](
                cache_seqlens,
                self.flat_order,
                self.eff_lengths,
                self.batch,
                self.causal,
                self.use_order,
                n_pow2,
                num_warps=4,
            )
            batch_order = self.flat_order
            lengths_arg = self.eff_lengths
            use_batch_order = True
            grid = (self.heads // self.bh, self.splits, self.batch * 2)
            q_s0, qr_s0, qs_s0 = q.stride(1), qr.stride(1), qs.stride(1)
            single_query = True
            heads_aligned = True
        else:
            if self.use_order and self.batch_order is not None:
                n_pow2 = triton.next_power_of_2(self.batch)
                sort_requests_kernel[(1,)](
                    cache_seqlens, self.batch_order, self.batch, n_pow2, num_warps=4
                )
                batch_order = self.batch_order
            else:
                batch_order = cache_seqlens
            lengths_arg = cache_seqlens
            use_batch_order = self.use_order and self.batch_order is not None
            grid = (triton.cdiv(self.total_heads, self.bh), self.splits, self.batch)
            q_s0, qr_s0, qs_s0 = q.stride(0), qr.stride(0), qs.stride(0)
            single_query = self.queries == 1
            heads_aligned = (self.total_heads % self.bh) == 0

        if self.splits > 1:
            tgt_out, tgt_lse = self.part_out, self.part_lse
        else:
            tgt_out, tgt_lse = self.out, self.lse

        if self.bh == 16:
            dense_int8_mla_tle_bh16_bk64_kernel[grid](
                q,
                qr,
                kv,
                kr,
                ks,
                qs,
                block_table,
                lengths_arg,
                batch_order,
                tgt_out,
                tgt_lse,
                q_s0,
                q.stride(1),
                q.stride(2),
                qr_s0,
                qr.stride(1),
                qr.stride(2),
                kv.stride(0),
                kv.stride(1),
                kr.stride(0),
                kr.stride(1),
                qs_s0,
                qs.stride(1),
                qs.stride(2),
                ks.stride(0),
                ks.stride(1),
                block_table.stride(0),
                self.batch,
                self.total_heads,
                self.heads,
                self.softmax_scale,
                self.splits,
                use_batch_order,
                self.causal,
                single_query,
                heads_aligned,
                self.out_dtype_tl,
                num_warps=4,
                num_stages=1,
            )
        else:
            dense_int8_mla_tle_kernel[grid](
                q,
                qr,
                kv,
                kr,
                ks,
                qs,
                block_table,
                lengths_arg,
                batch_order,
                tgt_out,
                tgt_lse,
                q_s0,
                q.stride(1),
                q.stride(2),
                qr_s0,
                qr.stride(1),
                qr.stride(2),
                kv.stride(0),
                kv.stride(1),
                kr.stride(0),
                kr.stride(1),
                qs_s0,
                qs.stride(1),
                qs.stride(2),
                ks.stride(0),
                ks.stride(1),
                block_table.stride(0),
                self.batch,
                self.total_heads,
                self.heads,
                self.softmax_scale,
                self.splits,
                use_batch_order,
                self.causal,
                single_query,
                heads_aligned,
                self.flat_q2,
                self.contig_q2_h16,
                self.out_dtype_tl,
                num_warps=8,
                num_stages=1,
            )
        if self.splits > 1:
            merge_splits_kernel[(self.batch * self.total_heads,)](
                self.part_out,
                self.part_lse,
                self.out,
                self.lse,
                self.batch,
                self.total_heads,
                self.heads,
                self.splits,
                self.out_dtype_tl,
                num_warps=4,
            )
        return self.out, self.lse


def prepare_flash_mla_with_kvcache_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = HEAD_DIM_NOPE,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    num_splits: Optional[int] = None,
    sort_requests: Optional[bool] = None,
) -> tuple[FlashMLAInt8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    """Validate inputs, prepare a zero-reallocation replay handle, and execute once.

    RoPE inputs are divided by the content scale. Active page IDs and lengths
    must be valid; active scales must be finite and positive. Unused cache
    slots may contain NaNs. Queries correspond to the final one or two cache
    tokens, and causal masks preserve their order.
    """
    if k_cache_lora.ndim == 4 and k_cache_lora.shape[2] == 1:
        k_cache_lora = k_cache_lora.squeeze(2)
        k_cache_rope = k_cache_rope.squeeze(2)
        k_scale = k_scale.squeeze(2)
    if (
        q_nope.ndim != 4
        or q_nope.shape[1] not in (1, 2)
        or q_nope.shape[-1] != HEAD_DIM_NOPE
    ):
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if (
        k_cache_lora.ndim != 3
        or k_cache_lora.shape[1:] != (PAGE_BLOCK_SIZE, HEAD_DIM_NOPE)
        or head_dim_v != HEAD_DIM_NOPE
    ):
        raise NotImplementedError("requires KV [pages, 64, 512] and head_dim_v=512")
    expected_shapes = (
        (batch, queries, heads, HEAD_DIM_ROPE),
        (k_cache_lora.shape[0], PAGE_BLOCK_SIZE, HEAD_DIM_ROPE),
        (batch, queries, heads, 1),
        (k_cache_lora.shape[0], PAGE_BLOCK_SIZE, 1),
        (batch,),
    )
    for tensor, shape in zip(
        (q_rope, k_cache_rope, q_scale, k_scale, cache_seqlens), expected_shapes
    ):
        if tensor.shape != shape:
            raise ValueError(f"expected shape {shape}, got {tuple(tensor.shape)}")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] == 0
    ):
        raise ValueError("block_table must have shape [batch, positive page capacity]")
    if q_nope.dtype != torch.int8 or k_cache_lora.dtype != torch.int8:
        raise TypeError("Q and KV content must be INT8")
    if (
        q_rope.dtype not in (torch.float16, torch.bfloat16)
        or k_cache_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE tensors must have the same FP16 or BF16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise TypeError("block_table and cache_seqlens must be INT32")
    tensors = (
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
    )
    for tensor in tensors:
        if tensor.device != q_nope.device or (
            tensor.numel() != 0 and tensor.stride(-1) != 1
        ):
            raise ValueError(
                "all tensors must share a device and contiguous last dimension"
            )
    scale = HEAD_DIM_TOTAL**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if num_splits is not None and num_splits not in (1, 2, 4, 8, 16):
        raise ValueError("num_splits must be one of 1, 2, 4, 8, 16")

    is_varlen = bool(batch > 1 and (cache_seqlens != cache_seqlens[0]).any().item())
    mean_pages = (
        int(round(cache_seqlens.float().mean().item() / PAGE_BLOCK_SIZE))
        if (num_splits is None and is_varlen)
        else block_table.shape[1]
    )
    bh, splits = select_launch_config(
        batch,
        queries * heads,
        block_table.shape[1],
        num_splits_override=num_splits,
        is_varlen=is_varlen,
        mean_pages=mean_pages,
        queries=queries,
    )
    should_sort = is_varlen if sort_requests is None else bool(sort_requests)
    handle = FlashMLAInt8PreparedHandle(
        tensors,
        splits,
        bh,
        scale,
        causal,
        sort_requests=should_sort,
    )
    return handle, handle()


def flash_mla_with_kvcache_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = HEAD_DIM_NOPE,
    tile_scheduler_metadata: None = None,
    num_splits: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tile_scheduler_metadata is not None:
        raise NotImplementedError("PPU MLA uses its own split scheduling")
    if out is not None or lse is not None:
        raise NotImplementedError(
            "PPU MLA outputs are allocated by its prepared handle"
        )
    if num_splits is not None and not isinstance(num_splits, int):
        raise TypeError("PPU MLA num_splits must be an integer or None")
    _, outputs = prepare_flash_mla_with_kvcache_w8a8(
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
        causal=causal,
        num_splits=num_splits,
    )
    return outputs

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""DSV4 TP8 decode: padding-aware blockdiag and split-topk orchestration."""

import importlib
import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._metax.ops import sparse_mla_policy as _tuning
from flaggems_vllm.runtime.backend._metax.ops.sparse_mla_policy import local_launch

logger = logging.getLogger(__name__)


_native = importlib.import_module("flaggems_vllm.ops.flash_mla_with_kvcache")
FlashMLASchedMeta = _native.FlashMLASchedMeta
_sparse_decode_model1_kernel = local_launch(
    _native._sparse_decode_model1_kernel,
    {"BK": 16, "BH": 16, "num_warps": 2, "num_stages": 1},
)


@triton.jit
def _partial_model1_kernel_halfd(
    q,
    kv,
    indices,
    extra_kv,
    extra_indices,
    topk_length,
    extra_topk_length,
    sm_scale: tl.constexpr,
    part_max,
    part_sum,
    part_acc,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kv_block,
    stride_ib,
    stride_isq,
    stride_extra_kv_block,
    stride_eib,
    stride_eisq,
    SQ: tl.constexpr,
    HQc: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    EXTRA_NUM_BLOCKS: tl.constexpr,
    SPLITS: tl.constexpr,
    BKc: tl.constexpr,
    BHc: tl.constexpr,
    V_START: tl.constexpr,
    WRITE_STATS: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQc + BHc - 1) // BHc
    pid = tl.program_id(0)
    i_split = pid % SPLITS
    base_pid = pid // SPLITS
    i_b = base_pid // (SQ * num_head_blocks)
    rem = base_pid % (SQ * num_head_blocks)
    i_sq = (rem // num_head_blocks).to(tl.int64)
    i_gbh = rem % num_head_blocks
    gbh_base = i_gbh * BHc
    NOPE: tl.constexpr = 448
    ROPE: tl.constexpr = 64
    BDP: tl.constexpr = 256
    TOKEN_DATA_BYTES: tl.constexpr = 576
    SCALE_BYTES: tl.constexpr = 8
    offs_h = tl.arange(0, BHc)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BKc)
    offs_rope = tl.arange(0, ROPE)
    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    et_base = extra_indices + i_b * stride_eib + i_sq * stride_eisq
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1_nope = tl.load(
        q_ptr + BDP,
        mask=offs_d[None, :] < (NOPE - BDP),
        other=0.0,
        eviction_policy="evict_first",
    )
    q_rope = tl.load(
        q_base + offs_h[:, None] * stride_qh + (NOPE + offs_rope[None, :]),
        eviction_policy="evict_first",
    )
    max_log = tl.full([BHc], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BHc], 0.0, dtype=tl.float32)
    acc = tl.zeros([BHc, BDP], dtype=tl.float32)

    topk_len = tl.load(topk_length + i_b)
    start = (topk_len * i_split) // SPLITS
    end = (topk_len * (i_split + 1)) // SPLITS
    seg_len = end - start
    NK = tl.cdiv(seg_len, BKc)
    for ck in range(NK):
        t_offs = start + BKc * ck + offs_t
        t_msk = t_offs < end
        kv_ids = tl.load(t_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // PAGE_SIZE
        rel_ids = kv_ids - block_ids * PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            kv + block_ids.to(tl.int64) * stride_kv_block + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            kv
            + block_ids.to(tl.int64) * stride_kv_block
            + PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < (NOPE - BDP),
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        if V_START == 0:
            acc = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk0.trans(),
                acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        else:
            acc = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk1.trans(),
                acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        max_log = new_max

    extra_len = tl.load(extra_topk_length + i_b)
    estart = (extra_len * i_split) // SPLITS
    eend = (extra_len * (i_split + 1)) // SPLITS
    eseg = eend - estart
    ENK = tl.cdiv(eseg, BKc)
    for ck in range(ENK):
        t_offs = estart + BKc * ck + offs_t
        t_msk = t_offs < eend
        kv_ids = tl.load(et_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // EXTRA_PAGE_SIZE
        rel_ids = kv_ids - block_ids * EXTRA_PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < EXTRA_NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + EXTRA_PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < (NOPE - BDP),
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        if V_START == 0:
            acc = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk0.trans(),
                acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        else:
            acc = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv_blk1.trans(),
                acc * alpha[:, None],
                out_dtype=tl.float32,
            )
        max_log = new_max

    if WRITE_STATS:
        base = (i_b * SPLITS + i_split) * HQc + gbh_base
        tl.store(part_max + base + offs_h, max_log)
        tl.store(part_sum + base + offs_h, sum_exp)
    acc_base = ((i_b * SPLITS + i_split) * HQc + gbh_base) * 512
    tl.store(
        part_acc + acc_base + offs_h[:, None] * 512 + (V_START + offs_d[None, :]), acc
    )


@triton.jit
def _partial_model1_kernel_halfd_qkfold_blockdiag(
    q,
    kv,
    indices,
    extra_kv,
    extra_indices,
    topk_length,
    extra_topk_length,
    sm_scale: tl.constexpr,
    part_max,
    part_sum,
    part_acc,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kv_block,
    stride_ib,
    stride_isq,
    stride_extra_kv_block,
    stride_eib,
    stride_eisq,
    SQ: tl.constexpr,
    HQc: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    EXTRA_NUM_BLOCKS: tl.constexpr,
    SPLITS: tl.constexpr,
    BKc: tl.constexpr,
    BHc: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQc + BHc - 1) // BHc
    pid = tl.program_id(0)
    i_split = pid % SPLITS
    base_pid = pid // SPLITS
    i_b = base_pid // (SQ * num_head_blocks)
    rem = base_pid % (SQ * num_head_blocks)
    i_sq = (rem // num_head_blocks).to(tl.int64)
    i_gbh = rem % num_head_blocks
    gbh_base = i_gbh * BHc
    NOPE: tl.constexpr = 448
    ROPE: tl.constexpr = 64
    BDP: tl.constexpr = 256
    TOKEN_DATA_BYTES: tl.constexpr = 576
    SCALE_BYTES: tl.constexpr = 8
    offs_h = tl.arange(0, BHc)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BKc)
    offs_rope = tl.arange(0, ROPE)
    real_h = offs_h % 8
    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    et_base = extra_indices + i_b * stride_eib + i_sq * stride_eisq
    q_ptr = q_base + real_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1_nope = tl.load(
        q_ptr + BDP,
        mask=offs_d[None, :] < (NOPE - BDP),
        other=0.0,
        eviction_policy="evict_first",
    )
    q_rope = tl.load(
        q_base + real_h[:, None] * stride_qh + (NOPE + offs_rope[None, :]),
        eviction_policy="evict_first",
    )
    max_log = tl.full([BHc], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BHc], 0.0, dtype=tl.float32)
    acc = tl.zeros([BHc, BDP], dtype=tl.float32)

    topk_len = tl.load(topk_length + i_b)
    start = (topk_len * i_split) // SPLITS
    end = (topk_len * (i_split + 1)) // SPLITS
    seg_len = end - start
    NK = tl.cdiv(seg_len, BKc)
    for ck in range(NK):
        t_offs = start + BKc * ck + offs_t
        t_msk = t_offs < end
        kv_ids = tl.load(t_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // PAGE_SIZE
        rel_ids = kv_ids - block_ids * PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            kv + block_ids.to(tl.int64) * stride_kv_block + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            kv
            + block_ids.to(tl.int64) * stride_kv_block
            + PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < (NOPE - BDP),
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        prob = exp_qk.to(tl.bfloat16)
        zero_prob = tl.zeros([BHc, BKc], dtype=tl.bfloat16)
        prob_lo = tl.where(offs_h[:, None] < 8, prob, zero_prob)
        prob_hi = tl.where(offs_h[:, None] >= 8, prob, zero_prob)
        prob_pair = tl.reshape(tl.join(prob_lo, prob_hi), [BHc, BKc * 2])
        kv_pair = tl.reshape(tl.join(kv_blk0, kv_blk1), [BDP, BKc * 2])
        acc = tl.dot(
            prob_pair, kv_pair.trans(), acc * alpha[:, None], out_dtype=tl.float32
        )
        max_log = new_max

    extra_len = tl.load(extra_topk_length + i_b)
    estart = (extra_len * i_split) // SPLITS
    eend = (extra_len * (i_split + 1)) // SPLITS
    eseg = eend - estart
    ENK = tl.cdiv(eseg, BKc)
    for ck in range(ENK):
        t_offs = estart + BKc * ck + offs_t
        t_msk = t_offs < eend
        kv_ids = tl.load(et_base + t_offs, t_msk, other=-1)
        block_ids = kv_ids // EXTRA_PAGE_SIZE
        rel_ids = kv_ids - block_ids * EXTRA_PAGE_SIZE
        valid_ids = t_msk & (kv_ids >= 0) & (block_ids < EXTRA_NUM_BLOCKS)
        block_ids = tl.where(valid_ids, block_ids, 0)
        rel_ids = tl.where(valid_ids, rel_ids, 0)
        token_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + rel_ids * TOKEN_DATA_BYTES
        )
        scale_base = (
            extra_kv
            + block_ids.to(tl.int64) * stride_extra_kv_block
            + EXTRA_PAGE_SIZE * TOKEN_DATA_BYTES
            + rel_ids * SCALE_BYTES
        )
        kv_fp8_0_u8 = tl.load(
            token_base[None, :] + offs_d[:, None],
            mask=valid_ids[None, :],
            other=0,
            cache_modifier=".cg",
        )
        kv_fp8_1_u8 = tl.load(
            token_base[None, :] + (BDP + offs_d[:, None]),
            mask=valid_ids[None, :] & (offs_d[:, None] < (NOPE - BDP)),
            other=0,
            cache_modifier=".cg",
        )
        scale0_u8 = tl.load(scale_base + 0, mask=valid_ids, other=127)
        scale1_u8 = tl.load(scale_base + 1, mask=valid_ids, other=127)
        scale2_u8 = tl.load(scale_base + 2, mask=valid_ids, other=127)
        scale3_u8 = tl.load(scale_base + 3, mask=valid_ids, other=127)
        scale4_u8 = tl.load(scale_base + 4, mask=valid_ids, other=127)
        scale5_u8 = tl.load(scale_base + 5, mask=valid_ids, other=127)
        scale6_u8 = tl.load(scale_base + 6, mask=valid_ids, other=127)
        scale0 = (scale0_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale1 = (scale1_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale2 = (scale2_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale3 = (scale3_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale4 = (scale4_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale5 = (scale5_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale6 = (scale6_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        kv_fp8_0 = kv_fp8_0_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_0 = tl.where(
            offs_d[:, None] < 64,
            scale0[None, :],
            tl.where(
                offs_d[:, None] < 128,
                scale1[None, :],
                tl.where(offs_d[:, None] < 192, scale2[None, :], scale3[None, :]),
            ),
        )
        kv_blk0 = (kv_fp8_0 * scale_0).to(tl.bfloat16)
        kv_fp8_1 = kv_fp8_1_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_1 = tl.where(
            offs_d[:, None] < 64,
            scale4[None, :],
            tl.where(offs_d[:, None] < 128, scale5[None, :], scale6[None, :]),
        )
        nope_tail = (kv_fp8_1 * scale_1).to(tl.bfloat16)
        rope_ptr = (token_base + NOPE).to(tl.pointer_type(tl.bfloat16))
        rope_blk = tl.load(
            rope_ptr[None, :] + offs_rope[:, None],
            mask=valid_ids[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv_blk1 = tl.where(
            offs_d[:, None] < (NOPE - BDP),
            nope_tail,
            tl.load(
                rope_ptr[None, :] + (offs_d[:, None] - (NOPE - BDP)),
                mask=valid_ids[None, :] & (offs_d[:, None] >= (NOPE - BDP)),
                other=0.0,
                cache_modifier=".cg",
            ),
        )
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1_nope, nope_tail, qk, out_dtype=tl.float32)
        qk = tl.dot(q_rope, rope_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(valid_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        prob = exp_qk.to(tl.bfloat16)
        zero_prob = tl.zeros([BHc, BKc], dtype=tl.bfloat16)
        prob_lo = tl.where(offs_h[:, None] < 8, prob, zero_prob)
        prob_hi = tl.where(offs_h[:, None] >= 8, prob, zero_prob)
        prob_pair = tl.reshape(tl.join(prob_lo, prob_hi), [BHc, BKc * 2])
        kv_pair = tl.reshape(tl.join(kv_blk0, kv_blk1), [BDP, BKc * 2])
        acc = tl.dot(
            prob_pair, kv_pair.trans(), acc * alpha[:, None], out_dtype=tl.float32
        )
        max_log = new_max

    base = (i_b * SPLITS + i_split) * HQc + gbh_base
    tl.store(part_max + base + offs_h, max_log)
    tl.store(part_sum + base + offs_h, sum_exp)
    acc_base = ((i_b * SPLITS + i_split) * HQc + gbh_base) * 512
    v_half = (offs_h >= 8).to(tl.int64) * BDP
    tl.store(
        part_acc + acc_base + real_h[:, None] * 512 + v_half[:, None] + offs_d[None, :],
        acc,
    )


@triton.jit(do_not_specialize=["Bsz"])
def _reduce_split_kernel(
    part_max,
    part_sum,
    part_acc,
    attn_sink,
    out,
    lse,
    stride_ob,
    stride_osq,
    stride_oh,
    stride_lseb,
    stride_lseh,
    Bsz,
    HQc: tl.constexpr,
    SPLITS: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    dblock = tl.program_id(1)
    h = pid % HQc
    b = pid // HQc
    offs_d = tl.arange(0, D_BLOCK)
    d0 = dblock * D_BLOCK + offs_d
    m = tl.full((), float("-inf"), dtype=tl.float32)
    for s in range(SPLITS):
        v = tl.load(part_max + (b * SPLITS + s) * HQc + h)
        m = tl.maximum(m, v)
    gsum = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([D_BLOCK], dtype=tl.float32)
    for s in range(SPLITS):
        pm = tl.load(part_max + (b * SPLITS + s) * HQc + h)
        ps = tl.load(part_sum + (b * SPLITS + s) * HQc + h)
        w = tl.math.exp(pm - m)
        gsum += ps * w
        pa = tl.load(part_acc + (((b * SPLITS + s) * HQc + h) * 512) + d0)
        acc += pa * w
    orig_lse = m + tl.math.log(gsum)
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + h)
        factor = tl.math.exp(m) / (tl.math.exp(orig_lse) + tl.math.exp(sink))
    else:
        factor = 1.0 / gsum
    vals = acc * factor
    tl.store(
        out + b * stride_ob + 0 * stride_osq + h * stride_oh + d0, vals.to(tl.bfloat16)
    )
    if dblock == 0:
        tl.store(lse + b * stride_lseb + h * stride_lseh + 0, orig_lse)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    *,
    real_num_heads: Optional[int] = None,
    split_workspace: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MODEL1 sparse decode with caller-owned output and optional split workspace.
    Dense and V32 paths delegate the existing upstream public operator.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v)
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32
    """
    logger.debug("GEMS_METAX FLASH_MLA_WITH_KVCACHE")
    sched_meta = tile_scheduler_metadata
    if not (isinstance(sched_meta, FlashMLASchedMeta)):
        raise ValueError(
            "Invalid operator argument: isinstance(sched_meta, FlashMLASchedMeta)"
        )
    if not (num_splits is None):
        raise ValueError("Invalid operator argument: num_splits is None")
    if not (q.ndim == 4):
        raise ValueError("Invalid operator argument: q.ndim == 4")
    if not (k_cache.ndim == 4):
        raise ValueError("Invalid operator argument: k_cache.ndim == 4")

    # Non-MODEL1 paths use the existing public operator before state writes.
    if indices is None or q.shape[-1] != 512:
        return _native.flash_mla_with_kvcache(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            head_dim_v,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale,
            causal,
            is_fp8_kvcache,
            indices,
            attn_sink,
            extra_k_cache,
            extra_indices_in_kvcache,
            topk_length,
            extra_topk_length,
            out,
        )

    topk = indices.shape[-1] if indices is not None else None
    extra_k_page_block_size = (
        extra_k_cache.shape[1] if extra_k_cache is not None else None
    )
    extra_topk_val = (
        extra_indices_in_kvcache.shape[-1]
        if extra_indices_in_kvcache is not None
        else None
    )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if not sched_meta.have_initialized:
        if not (not causal):
            raise ValueError("causal must be False when sparse attention is enabled")
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
            k_cache.shape[2],
            causal,
            is_fp8_kvcache,
            topk,
            extra_k_page_block_size,
            extra_topk_val,
        )
    else:
        helper_msg = (
            " Your input arguments are inconsistent with sched_meta. Please make "
            "sure the input arguments are consistent across different invocations "
            "of flash_mla_with_kvcache on the same sched_meta."
        )
        if not (sched_meta.config is not None):
            raise ValueError("Invalid operator argument: sched_meta.config is not None")
        if not (sched_meta.config.b == q.shape[0]):
            raise ValueError(
                "sched_meta.config.b must be equal to batch_size." + helper_msg
            )
        if not (sched_meta.config.s_q == q.shape[1]):
            raise ValueError(
                "sched_meta.config.s_q must be equal to seq_len_q." + helper_msg
            )
        if not (sched_meta.config.h_q == q.shape[2]):
            raise ValueError(
                "sched_meta.config.h_q must be equal to num_heads_q." + helper_msg
            )
        if not (sched_meta.config.page_block_size == k_cache.shape[1]):
            raise ValueError(
                "sched_meta.config.page_block_size must be equal to page_block_size."
                + helper_msg
            )
        if not (sched_meta.config.h_k == k_cache.shape[2]):
            raise ValueError(
                "sched_meta.config.h_k must be equal to num_heads_k." + helper_msg
            )
        if not (sched_meta.config.causal == causal):
            raise ValueError(
                "sched_meta.config.causal must be equal to causal." + helper_msg
            )
        if not (sched_meta.config.is_fp8_kvcache == is_fp8_kvcache):
            raise ValueError(
                "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache."
                + helper_msg
            )
        if not (sched_meta.config.topk == topk):
            raise ValueError(
                "sched_meta.config.topk must be equal to the last dim of indices."
                + helper_msg
            )
        if not (sched_meta.config.extra_page_block_size == extra_k_page_block_size):
            raise ValueError(
                "sched_meta.config.extra_page_block_size must be equal to the page_"
                "block_size of extra_k_cache." + helper_msg
            )
        if not (sched_meta.config.extra_topk == extra_topk_val):
            raise ValueError(
                "sched_meta.config.extra_topk must be equal to the last dim of extr"
                "a_indices_in_kvcache." + helper_msg
            )

    batch_size, seq_q, num_heads_q, head_dim_k = q.shape
    num_heads_k = k_cache.shape[2]

    if out is None:
        out = torch.empty(
            (batch_size, seq_q, num_heads_q, head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )
    else:
        if not (out.shape == (batch_size, seq_q, num_heads_q, head_dim_v)):
            raise ValueError(
                "Invalid operator argument: out.shape == (batch_size, seq_q, num_he"
                "ads_q, head_dim_v)"
            )
        if not (out.dtype == q.dtype):
            raise ValueError("Invalid operator argument: out.dtype == q.dtype")
        if not (out.device == q.device):
            raise ValueError("Invalid operator argument: out.device == q.device")
        if not (out.stride(-1) == 1):
            raise ValueError("Invalid operator argument: out.stride(-1) == 1")
    lse = torch.empty(
        (batch_size, num_heads_q, seq_q),
        dtype=torch.float32,
        device=q.device,
    )

    if not (not causal):
        raise ValueError("causal must be False when sparse attention is enabled")
    if not (is_fp8_kvcache):
        raise ValueError("is_fp8_kvcache must be True for sparse attention")
    if not (num_heads_k == 1):
        raise ValueError(
            "Currently only MQA (h_kv == 1) is supported for sparse decoding"
        )
    if not (head_dim_v == 512):
        raise ValueError("Only head_size_v == 512 is supported")
    if not (num_heads_q in (16, 64, 128)):
        raise ValueError("Only h_q == 16, 64 or 128 is supported")
    if not (head_dim_k in (512, 576)):
        raise ValueError(
            "Only head_size_k == 512 or 576 is supported for sparse decoding"
        )
    if not (q.dtype == torch.bfloat16):
        raise ValueError("Invalid operator argument: q.dtype == torch.bfloat16")
    if not (k_cache.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)):
        raise ValueError(
            "Invalid operator argument: k_cache.dtype in (torch.float8_e4m3fn, torc"
            "h.int8, torch.uint8)"
        )
    if not (topk is not None and topk > 0):
        raise ValueError("Invalid operator argument: topk is not None and topk > 0")
    if not (topk % 64 == 0):
        raise ValueError("topk must be divisible by 64")
    if not (indices.ndim == 3 and indices.shape[:2] == (batch_size, seq_q)):
        raise ValueError(
            "Invalid operator argument: indices.ndim == 3 and indices.shape[:2] == "
            "(batch_size, seq_q)"
        )
    if not (indices.dtype == torch.int32):
        raise ValueError("Invalid operator argument: indices.dtype == torch.int32")
    if not (indices.stride(-1) == 1):
        raise ValueError("Invalid operator argument: indices.stride(-1) == 1")
    if topk_length is not None:
        if not (topk_length.shape == (batch_size,)):
            raise ValueError(
                "Invalid operator argument: topk_length.shape == (batch_size,)"
            )
        if not (topk_length.dtype == torch.int32):
            raise ValueError(
                "Invalid operator argument: topk_length.dtype == torch.int32"
            )
        if not (topk_length.is_contiguous()):
            raise ValueError("Invalid operator argument: topk_length.is_contiguous()")
    if attn_sink is not None:
        if not (attn_sink.shape == (num_heads_q,)):
            raise ValueError(
                "Invalid operator argument: attn_sink.shape == (num_heads_q,)"
            )
        if not (attn_sink.dtype == torch.float32):
            raise ValueError(
                "Invalid operator argument: attn_sink.dtype == torch.float32"
            )
    if extra_k_cache is not None:
        if not (extra_indices_in_kvcache is not None):
            raise ValueError(
                "extra_indices_in_kvcache must be provided when extra_k_cache is pr"
                "ovided"
            )
        if not (extra_k_cache.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)):
            raise ValueError(
                "Invalid operator argument: extra_k_cache.dtype in (torch.float8_e4"
                "m3fn, "
                "torch.int8, torch.uint8)"
            )
    else:
        if not (extra_indices_in_kvcache is None):
            raise ValueError(
                "extra_indices_in_kvcache must not be provided when extra_k_cache i"
                "s not provided"
            )
        if not (extra_topk_length is None):
            raise ValueError(
                "extra_topk_length must not be provided when extra_k_cache is not p"
                "rovided"
            )
    if extra_indices_in_kvcache is not None:
        if not (extra_indices_in_kvcache.ndim == 3):
            raise ValueError(
                "Invalid operator argument: extra_indices_in_kvcache.ndim == 3"
            )
        if not (extra_indices_in_kvcache.shape[:2] == (batch_size, seq_q)):
            raise ValueError(
                "Invalid operator argument: extra_indices_in_kvcache.shape[:2] == ("
                "batch_size, seq_q)"
            )
        if not (extra_indices_in_kvcache.dtype == torch.int32):
            raise ValueError(
                "Invalid operator argument: extra_indices_in_kvcache.dtype == torch"
                ".int32"
            )
        if not (extra_indices_in_kvcache.stride(-1) == 1):
            raise ValueError(
                "Invalid operator argument: extra_indices_in_kvcache.stride(-1) == 1"
            )
        if not (extra_indices_in_kvcache.shape[-1] % 64 == 0):
            raise ValueError(
                "Invalid operator argument: extra_indices_in_kvcache.shape[-1] % 64"
                " == 0"
            )
    if extra_topk_length is not None:
        if not (extra_topk_length.shape == (batch_size,)):
            raise ValueError(
                "Invalid operator argument: extra_topk_length.shape == (batch_size,)"
            )
        if not (extra_topk_length.dtype == torch.int32):
            raise ValueError(
                "Invalid operator argument: extra_topk_length.dtype == torch.int32"
            )
        if not (extra_topk_length.is_contiguous()):
            raise ValueError(
                "Invalid operator argument: extra_topk_length.is_contiguous()"
            )
    if not (k_cache.shape[-1] == 584):
        raise ValueError("MODEL1 sparse FP8 cache must use 584 bytes per token")
    if not (k_cache.stride(1) == 584):
        raise ValueError("The whole block must be contiguous for MODEL1 KV cache")
    if extra_k_cache is not None:
        if not (extra_k_cache.ndim == 4):
            raise ValueError("Invalid operator argument: extra_k_cache.ndim == 4")
        if not (extra_k_cache.shape[2] == 1):
            raise ValueError("Invalid operator argument: extra_k_cache.shape[2] == 1")
        if not (extra_k_cache.shape[-1] == 584):
            raise ValueError(
                "Invalid operator argument: extra_k_cache.shape[-1] == 584"
            )
        if not (extra_k_cache.stride(1) == 584):
            raise ValueError(
                "Invalid operator argument: extra_k_cache.stride(1) == 584"
            )
    _sparse_decode_dispatch(
        q,
        k_cache,
        indices,
        out,
        lse,
        attn_sink,
        topk_length,
        extra_k_cache,
        extra_indices_in_kvcache,
        extra_topk_length,
        batch_size,
        seq_q,
        num_heads_q,
        head_dim_k,
        head_dim_v,
        topk,
        k_cache.shape[1],
        softmax_scale,
        is_fp8_kvcache,
        real_num_heads=real_num_heads,
        split_workspace=split_workspace,
    )

    return out, lse


def _split_topk_enabled() -> bool:
    return _tuning.DEFAULT_POLICY.decode_split_topk


def _try_sparse_decode_model1_split(
    q,
    kv,
    indices,
    out,
    lse,
    attn_sink,
    topk_length,
    extra_kv,
    extra_indices,
    extra_topk_length,
    batch_size,
    seq_q,
    num_heads_q,
    head_dim_k,
    head_dim_v,
    topk,
    page_block_size,
    softmax_scale,
    real_num_heads: Optional[int] = None,
    split_workspace: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """Launch graph-safe split-topk prototype for MODEL1 decode when explicitly enabled.

    Returns True if the split path launched. Unsupported shapes use the original
    semantic kernel path; this is not a reference/no-op fallback.
    """
    if not _split_topk_enabled():
        return False
    if head_dim_k != 512 or head_dim_v != 512:
        return False
    if seq_q != 1 or num_heads_q != 16:
        return False
    if topk_length is None or extra_topk_length is None:
        return False
    if extra_kv is None or extra_indices is None:
        return False
    if attn_sink is None:
        return False
    splits = _tuning.DEFAULT_POLICY.decode_splits
    bk = _tuning.DEFAULT_POLICY.decode_block_k
    bh = _tuning.DEFAULT_POLICY.decode_block_h
    warps = _tuning.DEFAULT_POLICY.decode_warps
    stages = _tuning.DEFAULT_POLICY.decode_stages
    max_b = _tuning.DEFAULT_POLICY.decode_max_batch
    if splits not in (2, 4, 8, 16) or bk not in (8, 16, 32) or bh != 16:
        return False
    if batch_size > max_b:
        return False
    if split_workspace is None:
        raise ValueError("split-topk requires caller-owned workspace")
    part_max, part_sum, part_acc = split_workspace
    expected = (
        (max_b, splits, num_heads_q),
        (max_b, splits, num_heads_q),
        (max_b, splits, num_heads_q, head_dim_v),
    )
    for buffer, shape in zip(split_workspace, expected):
        if (
            buffer.shape != shape
            or buffer.dtype != torch.float32
            or buffer.device != q.device
            or not buffer.is_contiguous()
        ):
            raise ValueError("Invalid split-topk workspace")
    pm = part_max[:batch_size]
    ps = part_sum[:batch_size]
    pa = part_acc[:batch_size]
    num_head_blocks = (num_heads_q + bh - 1) // bh
    qkfold = (
        _tuning.DEFAULT_POLICY.decode_qkfold
        and real_num_heads == 8
        and num_heads_q == 16
        and bh == 16
        and head_dim_k == 512
        and head_dim_v == 512
        and num_head_blocks == 1
    )
    if qkfold:
        _partial_model1_kernel_halfd_qkfold_blockdiag[
            (batch_size * seq_q * num_head_blocks * splits,)
        ](
            q,
            kv,
            indices,
            extra_kv,
            extra_indices,
            topk_length,
            extra_topk_length,
            softmax_scale,
            pm,
            ps,
            pa,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            indices.stride(0),
            indices.stride(1),
            extra_kv.stride(0),
            extra_indices.stride(0),
            extra_indices.stride(1),
            seq_q,
            num_heads_q,
            page_block_size,
            extra_kv.shape[1],
            kv.shape[0],
            extra_kv.shape[0],
            splits,
            bk,
            bh,
            num_warps=warps,
            num_stages=stages,
        )
    else:
        for _v_start, _write_stats in ((0, True), (256, False)):
            _partial_model1_kernel_halfd[
                (batch_size * seq_q * num_head_blocks * splits,)
            ](
                q,
                kv,
                indices,
                extra_kv,
                extra_indices,
                topk_length,
                extra_topk_length,
                softmax_scale,
                pm,
                ps,
                pa,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                kv.stride(0),
                indices.stride(0),
                indices.stride(1),
                extra_kv.stride(0),
                extra_indices.stride(0),
                extra_indices.stride(1),
                seq_q,
                num_heads_q,
                page_block_size,
                extra_kv.shape[1],
                kv.shape[0],
                extra_kv.shape[0],
                splits,
                bk,
                bh,
                V_START=_v_start,
                WRITE_STATS=_write_stats,
                num_warps=warps,
                num_stages=stages,
            )
    _reduce_split_kernel[(batch_size * num_heads_q, 2)](
        pm,
        ps,
        pa,
        attn_sink,
        out,
        lse,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        batch_size,
        num_heads_q,
        splits,
        True,
        256,
        num_warps=4,
        num_stages=1,
    )
    return True


def _sparse_decode_dispatch(
    q,
    kv,
    indices,
    out,
    lse,
    attn_sink,
    topk_length,
    extra_kv,
    extra_indices,
    extra_topk_length,
    batch_size,
    seq_q,
    num_heads_q,
    head_dim_k,
    head_dim_v,
    topk,
    page_block_size,
    softmax_scale,
    is_fp8_kvcache,
    real_num_heads: Optional[int] = None,
    split_workspace: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """Launch sparse decode kernel."""
    BH = 16
    num_head_blocks = (num_heads_q + BH - 1) // BH
    grid = (batch_size * seq_q * num_head_blocks,)

    if _try_sparse_decode_model1_split(
        q,
        kv,
        indices,
        out,
        lse,
        attn_sink,
        topk_length,
        extra_kv,
        extra_indices,
        extra_topk_length,
        batch_size,
        seq_q,
        num_heads_q,
        head_dim_k,
        head_dim_v,
        topk,
        page_block_size,
        softmax_scale,
        real_num_heads=real_num_heads,
        split_workspace=split_workspace,
    ):
        return
    _sparse_decode_model1_kernel[grid](
        q,
        kv,
        indices,
        extra_kv if extra_kv is not None else kv,
        extra_indices if extra_indices is not None else indices,
        attn_sink if attn_sink is not None else None,
        topk_length if topk_length is not None else None,
        extra_topk_length if extra_topk_length is not None else None,
        softmax_scale,
        out,
        lse,
        # Q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        # KV and indices strides
        kv.stride(0),
        indices.stride(0),
        indices.stride(1),
        extra_kv.stride(0) if extra_kv is not None else kv.stride(0),
        extra_indices.stride(0) if extra_indices is not None else indices.stride(0),
        extra_indices.stride(1) if extra_indices is not None else indices.stride(1),
        # Output strides
        out.stride(0),
        out.stride(1),
        out.stride(2),
        # LSE strides
        lse.stride(0),
        lse.stride(1),
        # Scalar args
        seq_q,
        num_heads_q,
        page_block_size,
        extra_kv.shape[1] if extra_kv is not None else 1,
        kv.shape[0],
        extra_kv.shape[0] if extra_kv is not None else 0,
        topk,
        extra_indices.shape[-1] if extra_indices is not None else 0,
        attn_sink is not None,
        topk_length is not None,
        extra_kv is not None,
        extra_topk_length is not None,
    )
    return

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

import os
from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE_FLASHMLA_SPARSE = True
    except ImportError:
        tle = None
        HAS_TLE_FLASHMLA_SPARSE = False
else:
    tle = None
    HAS_TLE_FLASHMLA_SPARSE = False


TLE_FLASHMLA_PREFILL_BK = 64
TLE_FLASHMLA_PREFILL_BH = 64
TLE_FLASHMLA_PREFILL_PAIR_BLOCKS = 2
TLE_FLASHMLA_PREFILL_WORKER_NUM_WARPS = 4


@lru_cache(maxsize=None)
def _use_quad_gather_pipeline(device: torch.device) -> bool:
    # Resource/performance validation covers this compiler/device (215040 B),
    # not an installation requirement. Other versions keep the original BK64.
    if device.type != "cuda" or device.index is None or triton.__version__ != "3.7.1":
        return False
    props = torch.cuda.get_device_properties(device)
    return (
        getattr(props, "name", None) == "NVIDIA H20"
        and getattr(props, "major", None) == 9
        and getattr(props, "minor", None) == 0
        and getattr(props, "shared_memory_per_block_optin", 0) >= 215040
    )


# Autotune exemption: this explicitly gated port preserves the source PR's
# local Triton candidate set for the HQ4 single-row kernel. Pair/quad kernels
# use the source PR's fixed H20 schedules below, so none of these choices can
# alter the default HQ64/HQ128 path.
@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BH": 4}, num_warps=4, num_stages=2),
        triton.Config({"BK": 64, "BH": 8}, num_warps=4, num_stages=3),
        triton.Config({"BK": 128, "BH": 4}, num_warps=8, num_stages=2),
        triton.Config({"BK": 128, "BH": 8}, num_warps=8, num_stages=3),
    ],
    key=[
        "SQ",
        "SKV",
        "TOPK",
        "HAVE_ATTN_SINK",
        "HAVE_TOPK_LENGTH",
        "RETURN_STATS",
    ],
)
@triton.jit
def triton_flash_mla_sparse_fwd_hq4(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvn,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    RETURN_STATS: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    """Four-head specialization with heads on the MMA N dimension."""

    i_sq = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, BH)
    mask_h = offs_h < 4
    offs_d = tl.arange(0, 256)
    offs_t = tl.arange(0, BK)

    q_base = q + i_sq * stride_qm
    indices_base = indices + i_sq * stride_tm
    output_base = output + i_sq * stride_om
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(
        q_ptr, mask=mask_h[:, None], other=0.0, eviction_policy="evict_first"
    )
    q_blk1 = tl.load(
        q_ptr + 256,
        mask=mask_h[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([BH], dtype=tl.float32)
    acc0 = tl.zeros([256, BH], dtype=tl.float32)
    acc1 = tl.zeros([256, BH], dtype=tl.float32)

    topk_len = tl.load(topk_length + i_sq) if HAVE_TOPK_LENGTH else TOPK
    topk_len = tl.minimum(tl.maximum(topk_len, 0), TOPK)
    num_k_blocks = tl.cdiv(topk_len, BK)
    for block_index in range(num_k_blocks):
        topk_offsets = block_index * BK + offs_t
        topk_mask = topk_offsets < topk_len
        kv_ids = tl.load(indices_base + topk_offsets, mask=topk_mask, other=-1)
        valid_ids = (kv_ids >= 0) & (kv_ids < SKV)
        safe_kv_ids = tl.where(valid_ids, kv_ids, 0)

        kv_ptr = kv + offs_d[:, None] + safe_kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
        kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".cg")

        # Keeping heads on N wastes only the hardware N=8 minimum.  Putting
        # four heads on M would execute the M=16 minimum in both dot products.
        qk_transposed = tl.dot(kv_blk0.trans(), q_blk0.trans(), out_dtype=tl.float32)
        qk_transposed = tl.dot(
            kv_blk1.trans(),
            q_blk1.trans(),
            qk_transposed,
            out_dtype=tl.float32,
        )
        qk_transposed *= sm_scale
        qk_transposed = tl.where(
            valid_ids[:, None] & mask_h[None, :],
            qk_transposed,
            float("-inf"),
        )

        new_max = tl.maximum(max_log, tl.max(qk_transposed, axis=0))
        max_for_exp = tl.where(new_max == float("-inf"), 0.0, new_max)
        exp_qk_transposed = tl.math.exp(qk_transposed - max_for_exp[None, :])
        block_sum = tl.sum(exp_qk_transposed, axis=0)
        alpha = tl.math.exp(max_log - max_for_exp)
        sum_exp = sum_exp * alpha + block_sum

        weights = exp_qk_transposed.to(tl.bfloat16)
        acc0 = tl.dot(
            kv_blk0,
            weights,
            acc0 * alpha[None, :],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            kv_blk1,
            weights,
            acc1 * alpha[None, :],
            out_dtype=tl.float32,
        )
        max_log = new_max

    valid_row = max_log != float("-inf")
    if RETURN_STATS:
        max_logits_base = max_logits + i_sq * stride_mm
        lse_base = lse + i_sq * stride_lm
        tl.store(max_logits_base + offs_h, max_log, mask=mask_h)
        original_lse = max_log + tl.math.log(sum_exp)
        lse_values = tl.where(valid_row, original_lse, float("inf"))
        tl.store(lse_base + offs_h, lse_values, mask=mask_h)

    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + offs_h, mask=mask_h, other=float("-inf"))
        denominator = sum_exp + tl.math.exp(sink - max_log)
    else:
        denominator = sum_exp
    denominator = tl.where(valid_row, denominator, 1.0)
    factor = 1.0 / denominator
    out_vals0 = tl.where(valid_row[None, :], acc0 * factor[None, :], 0.0)
    out_vals1 = tl.where(valid_row[None, :], acc1 * factor[None, :], 0.0)
    output_ptr = output_base + offs_h[:, None] * stride_oh + offs_d[None, :]
    tl.store(output_ptr, out_vals0.trans().to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(
        output_ptr + 256,
        out_vals1.trans().to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.jit
def _flash_mla_sparse_hq4_pair(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    output,
    first_row,
    pair_mode,
    first_topk_length,
    stride_qh,
    stride_qm,
    stride_kvn,
    stride_tm,
    stride_oh,
    stride_om,
    SKV,
    sm_scale: tl.constexpr,
    WINDOW: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    PAIR_CACHE_CA: tl.constexpr,
    BK: tl.constexpr,
):
    """Evaluate two four-head rows from one small union of their KV IDs."""

    first_length = tl.load(topk_length + first_row)
    second_length = tl.load(topk_length + first_row + 1)
    offsets_n = tl.arange(0, 8)
    query_in_pair = offsets_n // 4
    head = offsets_n % 4
    query_row = first_row + query_in_pair
    column_length = tl.where(query_in_pair == 0, first_length, second_length)
    is_mode2 = pair_mode == 2
    is_mode3 = pair_mode == 3
    is_mode4 = pair_mode == 4
    second_topk_length = first_topk_length + 1
    union_length = second_length + tl.where(is_mode2 | is_mode4, 1, 0)

    offsets_d = tl.arange(0, 256)
    q_ptr = (
        q
        + query_row[:, None] * stride_qm
        + head[:, None] * stride_qh
        + offsets_d[None, :]
    )
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + 256, eviction_policy="evict_first")

    offsets_t = tl.arange(0, BK)
    max_log = tl.full([8], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([8], dtype=tl.float32)
    acc0 = tl.zeros([256, 8], dtype=tl.float32)
    acc1 = tl.zeros([256, 8], dtype=tl.float32)

    for block_index in range(tl.cdiv(union_length, BK)):
        union_position = block_index * BK + offsets_t
        union_mask = union_position < union_length

        # Modes 1 and 3 use row 1 directly.  A saturated, unchanged top-k
        # (mode 2) uses row 0 plus row 1's new window tail.  A saturated,
        # growing top-k (mode 4) uses row 1 plus row 0's expired window head.
        mode2_source_row = tl.where(
            union_position < first_length, first_row, first_row + 1
        )
        mode2_source_position = tl.where(
            union_position < first_length,
            union_position,
            first_length - 1,
        )
        mode4_source_row = tl.where(
            union_position == second_topk_length, first_row, first_row + 1
        )
        mode4_source_position = tl.where(
            union_position < second_topk_length,
            union_position,
            tl.where(
                union_position == second_topk_length,
                first_topk_length,
                union_position - 1,
            ),
        )
        source_row = tl.where(
            is_mode2,
            mode2_source_row,
            tl.where(is_mode4, mode4_source_row, first_row + 1),
        )
        source_position = tl.where(
            is_mode2,
            mode2_source_position,
            tl.where(is_mode4, mode4_source_position, union_position),
        )
        kv_ids = tl.load(
            indices + source_row * stride_tm + source_position,
            mask=union_mask,
            other=-1,
        )
        valid_ids = union_mask & (kv_ids >= 0) & (kv_ids < SKV)
        safe_kv_ids = tl.where(valid_ids, kv_ids, 0)
        kv_ptr = kv + offsets_d[:, None] + safe_kv_ids[None, :] * stride_kvn
        if PAIR_CACHE_CA:
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".ca")
            kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".ca")
        else:
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
            kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".cg")

        scores = tl.dot(kv_blk0.trans(), q_blk0.trans(), out_dtype=tl.float32)
        scores = tl.dot(kv_blk1.trans(), q_blk1.trans(), scores, out_dtype=tl.float32)
        scores *= sm_scale

        prefix_mask = union_position[:, None] < column_length[None, :]
        mode2_mask = tl.where(
            query_in_pair[None, :] == 0,
            union_position[:, None] < first_length,
            union_position[:, None] != first_topk_length,
        )
        mode3_first_mask = (
            (union_position[:, None] < second_length)
            & (union_position[:, None] != first_topk_length)
            & (union_position[:, None] != second_length - 1)
        )
        mode3_mask = tl.where(
            query_in_pair[None, :] == 0,
            mode3_first_mask,
            union_position[:, None] < second_length,
        )
        mode4_first_mask = (union_position[:, None] < first_topk_length) | (
            (union_position[:, None] >= second_topk_length)
            & (union_position[:, None] < second_topk_length + WINDOW)
        )
        mode4_mask = tl.where(
            query_in_pair[None, :] == 0,
            mode4_first_mask,
            union_position[:, None] != second_topk_length,
        )
        pair_mask = tl.where(
            is_mode2,
            mode2_mask,
            tl.where(
                is_mode3,
                mode3_mask,
                tl.where(is_mode4, mode4_mask, prefix_mask),
            ),
        )
        scores = tl.where(valid_ids[:, None] & pair_mask, scores, float("-inf"))

        new_max = tl.maximum(max_log, tl.max(scores, axis=0))
        max_for_exp = tl.where(new_max == float("-inf"), 0.0, new_max)
        exp_scores = tl.math.exp(scores - max_for_exp[None, :])
        alpha = tl.math.exp(max_log - max_for_exp)
        sum_exp = sum_exp * alpha + tl.sum(exp_scores, axis=0)
        weights = exp_scores.to(tl.bfloat16)
        acc0 = tl.dot(
            kv_blk0,
            weights,
            acc0 * alpha[None, :],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            kv_blk1,
            weights,
            acc1 * alpha[None, :],
            out_dtype=tl.float32,
        )
        max_log = new_max

    valid_row = max_log != float("-inf")
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + head)
        denominator = sum_exp + tl.math.exp(sink - max_log)
    else:
        denominator = sum_exp
    denominator = tl.where(valid_row, denominator, 1.0)
    factor = 1.0 / denominator
    out_vals0 = tl.where(valid_row[None, :], acc0 * factor[None, :], 0.0)
    out_vals1 = tl.where(valid_row[None, :], acc1 * factor[None, :], 0.0)
    output_ptr = (
        output
        + query_row[:, None] * stride_om
        + head[:, None] * stride_oh
        + offsets_d[None, :]
    )
    tl.store(output_ptr, out_vals0.trans().to(tl.bfloat16))
    tl.store(output_ptr + 256, out_vals1.trans().to(tl.bfloat16))


@triton.jit
def _flash_mla_sparse_hq4_single(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    output,
    row,
    stride_qh,
    stride_qm,
    stride_kvn,
    stride_tm,
    stride_oh,
    stride_om,
    SKV,
    sm_scale: tl.constexpr,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    SINGLE_CACHE_CA: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    """Run the regular four-head path for a work item that cannot be paired."""

    offsets_h = tl.arange(0, BH)
    mask_h = offsets_h < 4
    offsets_d = tl.arange(0, 256)
    q_ptr = q + row * stride_qm + offsets_h[:, None] * stride_qh + offsets_d[None, :]
    q_blk0 = tl.load(q_ptr, mask=mask_h[:, None], other=0.0)
    q_blk1 = tl.load(q_ptr + 256, mask=mask_h[:, None], other=0.0)

    raw_length = tl.load(topk_length + row)
    length = tl.minimum(tl.maximum(raw_length, 0), TOPK)
    offsets_t = tl.arange(0, BK)
    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([BH], dtype=tl.float32)
    acc0 = tl.zeros([256, BH], dtype=tl.float32)
    acc1 = tl.zeros([256, BH], dtype=tl.float32)

    for block_index in range(tl.cdiv(length, BK)):
        positions = block_index * BK + offsets_t
        token_mask = positions < length
        kv_ids = tl.load(
            indices + row * stride_tm + positions,
            mask=token_mask,
            other=-1,
        )
        valid_ids = token_mask & (kv_ids >= 0) & (kv_ids < SKV)
        safe_kv_ids = tl.where(valid_ids, kv_ids, 0)
        kv_ptr = kv + offsets_d[:, None] + safe_kv_ids[None, :] * stride_kvn
        if SINGLE_CACHE_CA:
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".ca")
            kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".ca")
        else:
            kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
            kv_blk1 = tl.load(kv_ptr + 256, cache_modifier=".cg")

        scores = tl.dot(kv_blk0.trans(), q_blk0.trans(), out_dtype=tl.float32)
        scores = tl.dot(kv_blk1.trans(), q_blk1.trans(), scores, out_dtype=tl.float32)
        scores *= sm_scale
        scores = tl.where(valid_ids[:, None] & mask_h[None, :], scores, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(scores, axis=0))
        max_for_exp = tl.where(new_max == float("-inf"), 0.0, new_max)
        exp_scores = tl.math.exp(scores - max_for_exp[None, :])
        alpha = tl.math.exp(max_log - max_for_exp)
        sum_exp = sum_exp * alpha + tl.sum(exp_scores, axis=0)
        weights = exp_scores.to(tl.bfloat16)
        acc0 = tl.dot(
            kv_blk0,
            weights,
            acc0 * alpha[None, :],
            out_dtype=tl.float32,
        )
        acc1 = tl.dot(
            kv_blk1,
            weights,
            acc1 * alpha[None, :],
            out_dtype=tl.float32,
        )
        max_log = new_max

    valid_row = max_log != float("-inf")
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + offsets_h, mask=mask_h, other=float("-inf"))
        denominator = sum_exp + tl.math.exp(sink - max_log)
    else:
        denominator = sum_exp
    denominator = tl.where(valid_row, denominator, 1.0)
    factor = 1.0 / denominator
    out_vals0 = tl.where(valid_row[None, :], acc0 * factor[None, :], 0.0)
    out_vals1 = tl.where(valid_row[None, :], acc1 * factor[None, :], 0.0)
    output_ptr = (
        output + row * stride_om + offsets_h[:, None] * stride_oh + offsets_d[None, :]
    )
    tl.store(output_ptr, out_vals0.trans().to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(
        output_ptr + 256,
        out_vals1.trans().to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.jit
def triton_flash_mla_sparse_fwd_hq4_pair_work_items(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    pair_metadata,
    output,
    stride_qh,
    stride_qm,
    stride_kvn,
    stride_tm,
    stride_oh,
    stride_om,
    SQ,
    SKV,
    sm_scale: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    PAIR_CACHE_CA: tl.constexpr,
    SINGLE_CACHE_CA: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    work_id_override=None,
):
    """Select an N=8 pair or an N=4 single path from producer metadata."""

    work_id = tl.program_id(0) if work_id_override is None else work_id_override
    pair_id = work_id // 2
    slot = work_id % 2
    first_row = (pair_id * 2).to(tl.int64)
    has_second = first_row + 1 < SQ
    packed_metadata = tl.load(pair_metadata + pair_id)
    pair_mode = packed_metadata & 7
    first_topk_length = packed_metadata >> 3
    first_length = tl.load(topk_length + first_row)
    second_length = tl.load(topk_length + first_row + 1, mask=has_second, other=0)

    valid_mode1 = (
        (pair_mode == 1)
        & has_second
        & (first_length > 0)
        & (second_length == first_length + 1)
        & (second_length <= TOPK)
        & (first_topk_length >= 0)
        & (first_length - first_topk_length > 0)
        & (first_length - first_topk_length < WINDOW)
    )
    valid_mode2 = (
        (pair_mode == 2)
        & has_second
        & (first_length >= WINDOW)
        & (first_length == second_length)
        & (first_length <= TOPK)
        & (first_length - first_topk_length == WINDOW)
    )
    valid_mode3 = (
        (pair_mode == 3)
        & has_second
        & (first_length > 0)
        & (second_length == first_length + 2)
        & (second_length <= TOPK)
        & (first_topk_length >= 0)
        & (first_length - first_topk_length > 0)
        & (first_length - first_topk_length < WINDOW)
    )
    valid_mode4 = (
        (pair_mode == 4)
        & has_second
        & (first_length > 0)
        & (second_length == first_length + 1)
        & (second_length <= TOPK)
        & (first_topk_length >= 0)
        & (first_length - first_topk_length == WINDOW)
    )
    is_pair = valid_mode1 | valid_mode2 | valid_mode3 | valid_mode4

    if is_pair:
        if slot == 0:
            _flash_mla_sparse_hq4_pair(
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                output,
                first_row,
                pair_mode,
                first_topk_length,
                stride_qh,
                stride_qm,
                stride_kvn,
                stride_tm,
                stride_oh,
                stride_om,
                SKV,
                sm_scale,
                WINDOW,
                HAVE_ATTN_SINK,
                PAIR_CACHE_CA,
                BK,
            )
    else:
        row = first_row + slot
        if row < SQ:
            _flash_mla_sparse_hq4_single(
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                output,
                row,
                stride_qh,
                stride_qm,
                stride_kvn,
                stride_tm,
                stride_oh,
                stride_om,
                SKV,
                sm_scale,
                TOPK,
                HAVE_ATTN_SINK,
                SINGLE_CACHE_CA,
                BK,
                BH,
            )


@triton.jit
def _flash_mla_sparse_hq4_quad(
    q,
    kv,
    indices,
    sink,
    lens,
    pairs,
    out,
    row,
    stride_qh,
    stride_qm,
    stride_kv,
    stride_i,
    stride_oh,
    stride_om,
    SKV,
    SCALE: tl.constexpr,
    W: tl.constexpr,
    SINK: tl.constexpr,
    BK: tl.constexpr,
    GATHER_PIPELINE: tl.constexpr = False,
):
    """Reuse one KV union for four queries certified by the index producer."""
    m0 = tl.load(pairs + row // 2)
    m1 = tl.load(pairs + row // 2 + 1)
    p0 = m0 >> 3
    p1 = p0 + (((m0 & 7) == 3) | ((m0 & 7) == 4)).to(tl.int32)
    p2 = m1 >> 3
    p3 = p2 + (((m1 & 7) == 3) | ((m1 & 7) == 4)).to(tl.int32)
    s0 = tl.load(lens + row) - p0
    union_len = p3 + s0 + 3
    swa_base = tl.load(indices + row * stride_i + p0)
    col = tl.arange(0, 16)
    qid = col // 4
    head = col % 4
    plen = tl.where(qid == 0, p0, tl.where(qid == 1, p1, tl.where(qid == 2, p2, p3)))
    wstart = tl.maximum(0, s0 + qid - W)
    wend = s0 + qid
    # Four 128-wide fragments keep the MMA operands below the register and
    # shared-memory footprint of two 256-wide fragments at N=16.
    d = tl.arange(0, 128)
    t = tl.arange(0, BK)
    qp = q + (row + qid[:, None]) * stride_qm + head[:, None] * stride_qh + d[None, :]
    q0 = tl.load(qp)
    q1 = tl.load(qp + 128)
    q2 = tl.load(qp + 256)
    q3 = tl.load(qp + 384)
    maximum = tl.full((16,), float("-inf"), tl.float32)
    total = tl.zeros((16,), tl.float32)
    acc0 = tl.zeros((128, 16), tl.float32)
    acc1 = tl.zeros((128, 16), tl.float32)
    acc2 = tl.zeros((128, 16), tl.float32)
    acc3 = tl.zeros((128, 16), tl.float32)
    if GATHER_PIPELINE:
        # Carry only integer addresses, leaving FP32 online-softmax unchanged.
        pos = t
        cid = tl.load(
            indices + (row + 3) * stride_i + pos,
            mask=(pos < p3) & (union_len > 0),
            other=0,
        )
        kid = tl.where(pos < p3, cid, swa_base + pos - p3)
        valid = (pos < union_len) & (kid >= 0) & (kid < SKV)
        kid = tl.where(valid, kid, 0)
    for block in range(tl.cdiv(union_len, BK)):
        pos = block * BK + t
        if not GATHER_PIPELINE:
            cid = tl.load(indices + (row + 3) * stride_i + pos, mask=pos < p3, other=0)
            kid = tl.where(pos < p3, cid, swa_base + pos - p3)
            valid = (pos < union_len) & (kid >= 0) & (kid < SKV)
            kid = tl.where(valid, kid, 0)
        kp = kv + d[:, None] + kid[None, :] * stride_kv
        k0 = tl.load(kp, cache_modifier=".cg")
        k1 = tl.load(kp + 128, cache_modifier=".cg")
        k2 = tl.load(kp + 256, cache_modifier=".cg")
        k3 = tl.load(kp + 384, cache_modifier=".cg")
        scores = tl.dot(k0.trans(), q0.trans(), out_dtype=tl.float32)
        scores = tl.dot(k1.trans(), q1.trans(), scores, out_dtype=tl.float32)
        scores = tl.dot(k2.trans(), q2.trans(), scores, out_dtype=tl.float32)
        scores = tl.dot(k3.trans(), q3.trans(), scores, out_dtype=tl.float32) * SCALE
        attend = (pos[:, None] < plen[None, :]) | (
            (pos[:, None] >= p3 + wstart[None, :]) & (pos[:, None] < p3 + wend[None, :])
        )
        scores = tl.where(valid[:, None] & attend, scores, float("-inf"))
        if GATHER_PIPELINE:
            # Decode the next gather before this block's max/exp/PV work.
            next_pos = (block + 1) * BK + t
            has_next = block + 1 < tl.cdiv(union_len, BK)
            next_cid = tl.load(
                indices + (row + 3) * stride_i + next_pos,
                mask=(next_pos < p3) & has_next,
                other=0,
            )
            next_kid = tl.where(next_pos < p3, next_cid, swa_base + next_pos - p3)
            next_valid = (next_pos < union_len) & (next_kid >= 0) & (next_kid < SKV)
            next_kid = tl.where(next_valid, next_kid, 0)
        updated = tl.maximum(maximum, tl.max(scores, 0))
        safe = tl.where(updated == float("-inf"), 0.0, updated)
        weights = tl.exp(scores - safe[None, :])
        alpha = tl.exp(maximum - safe)
        total = total * alpha + tl.sum(weights, 0)
        w = weights.to(tl.bfloat16)
        acc0 = tl.dot(k0, w, acc0 * alpha[None, :], out_dtype=tl.float32)
        acc1 = tl.dot(k1, w, acc1 * alpha[None, :], out_dtype=tl.float32)
        acc2 = tl.dot(k2, w, acc2 * alpha[None, :], out_dtype=tl.float32)
        acc3 = tl.dot(k3, w, acc3 * alpha[None, :], out_dtype=tl.float32)
        maximum = updated
        if GATHER_PIPELINE:
            kid = next_kid
            valid = next_valid
    valid = maximum != float("-inf")
    if SINK:
        sink_value = tl.load(sink + head)
        denom = total + tl.exp(sink_value - maximum)
    else:
        denom = total
    denom = tl.where(valid, denom, 1.0)
    factor = 1.0 / denom
    o0 = tl.where(valid[None, :], acc0 * factor[None, :], 0.0)
    o1 = tl.where(valid[None, :], acc1 * factor[None, :], 0.0)
    o2 = tl.where(valid[None, :], acc2 * factor[None, :], 0.0)
    o3 = tl.where(valid[None, :], acc3 * factor[None, :], 0.0)
    op = out + (row + qid[:, None]) * stride_om + head[:, None] * stride_oh + d[None, :]
    tl.store(op, o0.trans().to(tl.bfloat16))
    tl.store(op + 128, o1.trans().to(tl.bfloat16))
    tl.store(op + 256, o2.trans().to(tl.bfloat16))
    tl.store(op + 384, o3.trans().to(tl.bfloat16))


@triton.jit
def _hq4_pair_descriptor_is_valid(lens, pairs, pair_id, SQ, TOPK, W):
    first_row = pair_id * 2
    has_first = first_row < SQ
    has_second = first_row + 1 < SQ
    packed = tl.load(pairs + pair_id, mask=has_first, other=0)
    mode = packed & 7
    first_topk = packed >> 3
    first_length = tl.load(lens + first_row, mask=has_first, other=0)
    second_length = tl.load(lens + first_row + 1, mask=has_second, other=0)
    common = (
        has_second
        & (first_topk >= 0)
        & (first_topk <= first_length)
        & (first_length <= TOPK)
        & (second_length <= TOPK)
    )
    valid_mode1 = (
        (mode == 1)
        & (first_length > 0)
        & (second_length == first_length + 1)
        & (first_length - first_topk > 0)
        & (first_length - first_topk < W)
    )
    valid_mode2 = (
        (mode == 2)
        & (first_length >= W)
        & (second_length == first_length)
        & (first_length - first_topk == W)
    )
    valid_mode3 = (
        (mode == 3)
        & (first_length > 0)
        & (second_length == first_length + 2)
        & (first_length - first_topk > 0)
        & (first_length - first_topk < W)
    )
    valid_mode4 = (
        (mode == 4)
        & (first_length > 0)
        & (second_length == first_length + 1)
        & (first_length - first_topk == W)
    )
    return common & (valid_mode1 | valid_mode2 | valid_mode3 | valid_mode4)


@triton.jit
def _hq4_quad_descriptor_is_valid(lens, pairs, quads, group, SQ, TOPK, W):
    row = group * 4
    enabled = tl.load(quads + group)
    pair0_valid = _hq4_pair_descriptor_is_valid(lens, pairs, group * 2, SQ, TOPK, W)
    pair1_valid = _hq4_pair_descriptor_is_valid(lens, pairs, group * 2 + 1, SQ, TOPK, W)
    m0 = tl.load(pairs + group * 2, mask=row < SQ, other=0)
    m1 = tl.load(pairs + group * 2 + 1, mask=row + 2 < SQ, other=0)
    p0 = m0 >> 3
    p1 = p0 + (((m0 & 7) == 3) | ((m0 & 7) == 4)).to(tl.int32)
    p2 = m1 >> 3
    p3 = p2 + (((m1 & 7) == 3) | ((m1 & 7) == 4)).to(tl.int32)
    l0 = tl.load(lens + row, mask=row < SQ, other=0)
    s0 = l0 - p0
    union_length = p3 + s0 + 3
    boundary_valid = (p2 >= p1) & (p2 <= p1 + 1)
    return (
        (enabled == 1)
        & (row + 3 < SQ)
        & pair0_valid
        & pair1_valid
        & boundary_valid
        & (s0 > 0)
        & (s0 <= W)
        # Four sliding windows can extend three positions past one row's
        # logical width; those IDs are generated arithmetically, not loaded
        # from the row.
        & (union_length <= TOPK + 3)
    )


@triton.jit
def triton_flash_mla_sparse_fwd_hq4_quad(
    q,
    kv,
    indices,
    sink,
    lens,
    pairs,
    quad_metadata,
    out,
    stride_qh,
    stride_qm,
    stride_kv,
    stride_i,
    stride_oh,
    stride_om,
    SQ,
    SKV,
    SCALE: tl.constexpr,
    TOPK: tl.constexpr,
    W: tl.constexpr,
    SINK: tl.constexpr,
    BK: tl.constexpr,
    GATHER_PIPELINE: tl.constexpr = False,
):
    group = tl.program_id(0)
    row = (group * 4).to(tl.int64)
    descriptor_is_valid = _hq4_quad_descriptor_is_valid(
        lens, pairs, quad_metadata, group, SQ, TOPK, W
    )
    if descriptor_is_valid:
        _flash_mla_sparse_hq4_quad(
            q,
            kv,
            indices,
            sink,
            lens,
            pairs,
            out,
            row,
            stride_qh,
            stride_qm,
            stride_kv,
            stride_i,
            stride_oh,
            stride_om,
            SKV,
            SCALE,
            W,
            SINK,
            BK,
            GATHER_PIPELINE,
        )


@triton.jit
def triton_flash_mla_sparse_fwd_hq4_quad_fallback(
    q,
    kv,
    indices,
    sink,
    lens,
    pairs,
    quad_metadata,
    out,
    stride_qh,
    stride_qm,
    stride_kv,
    stride_i,
    stride_oh,
    stride_om,
    SQ,
    SKV,
    SCALE: tl.constexpr,
    TOPK: tl.constexpr,
    W: tl.constexpr,
    SINK: tl.constexpr,
    BK: tl.constexpr,
):
    work_id = tl.program_id(0)
    group = work_id // 4
    descriptor_is_valid = _hq4_quad_descriptor_is_valid(
        lens, pairs, quad_metadata, group, SQ, TOPK, W
    )
    if descriptor_is_valid == 0:
        triton_flash_mla_sparse_fwd_hq4_pair_work_items(
            q,
            kv,
            indices,
            sink,
            lens,
            pairs,
            out,
            stride_qh,
            stride_qm,
            stride_kv,
            stride_i,
            stride_oh,
            stride_om,
            SQ,
            SKV,
            SCALE,
            TOPK,
            W,
            SINK,
            not SINK,
            SINK,
            BK,
            4,
            work_id_override=work_id,
        )


# The pair/quad schedules above are intentionally fixed. They are the
# H20-validated schedules from the source optimization and are kept out of the
# default path by ``enable_hq4_sparse_prefill``.


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BH": 64}, num_warps=8, num_stages=2),
        triton.Config({"BK": 64, "BH": 64}, num_warps=8, num_stages=4),
    ],
    key=["SQ", "HQ", "DQK", "SKV", "TOPK", "HAVE_ATTN_SINK", "HAVE_TOPK_LENGTH"],
)
@triton.jit
def triton_flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvg,
    stride_kvn,
    stride_tg,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,  # s_q
    HQ: tl.constexpr,  # h_q=64 or 128
    DQK: tl.constexpr,  # d_qk=512 or 576
    SKV,  # s_kv
    TOPK: tl.constexpr,  # topk
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)  # prevent mul overflow
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256

    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    tkv_base = kv + DP
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    max_log_base = max_logits + i_sq * stride_mm + gbh_base
    l_base = lse + i_sq * stride_lm + gbh_base

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)

    # `[BH, 256] x 2` delivers better performance than `[BH, 512]` when BH=64
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + BDP, eviction_policy="evict_first")
    if DQK == 576:
        tq_ptr = q_base + DP + offs_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(tq_ptr, eviction_policy="evict_first")

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BH, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BH, BDP], dtype=tl.float32)

    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        # step1: load indices
        t_ptr = BK * ck + offs_t  # [BK]
        t_msk = t_ptr < topk_len
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        # filter invalid index that may cause overflow in mul
        kv_ids = tl.where(mask_ids, kv_ids, 0)

        # step2: gather kv with indices
        kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")  # [BDP, BK]
        kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")  # [BDP, BK]
        # step3: (q @ kv) * sm_scale
        qk = tl.dot(
            q_blk0, kv_blk0, out_dtype=tl.float32
        )  # [BH, BDP]@[BDP, BK] -> [BH, BK]
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            tkv_ptr = tkv_base + offs_td[:, None] + kv_ids[None, :] * stride_kvn
            tkv_blk = tl.load(tkv_ptr, cache_modifier=".cg")  # [TDP, BK]
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale

        # step4: preprocess for logsumexp
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))  # [BH, BK]
        # step5: lse=logsumexp(qk), loop part
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))  # [BH]
        exp_qk = tl.math.exp(qk - new_max[:, None])  # [BH, BK]
        sum_qk = tl.sum(exp_qk, axis=1)  # [BH]
        alpha = tl.math.exp(max_log - new_max)  # [BH]
        sum_exp = sum_exp * alpha + sum_qk  # [BH]
        # step6: exp(qk-lse) @ gathered_kv.trans(), loop part
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )  # [BH, BK]@[BK, BDP]->[BH, BDP]
        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )  # [BH, BK]@[BK, BDP]->[BH, BDP]
        max_log = new_max

    # step7: store max_logits
    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    tl.store(max_log_base + offs_h, max_log)  # [BH], float32

    # step8: lse=logsumexp(qk) final part, store lse
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h, lse_out)  # [BH], float32

    # step9: exp(qk-lse) @ gathered_kv.trans(), final part
    if HAVE_ATTN_SINK:
        # step10: attn_sink
        sink = tl.load(attn_sink_ptr + offs_h)  # [BH]
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp

    out_vals0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)
    # step11: store output
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]  # [BH, BDP]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16))


if HAS_TLE_FLASHMLA_SPARSE:

    @triton.jit
    def _tle_flashmla_prefill_producer(
        k0_l_writer,
        k0_r_writer,
        k1_l_writer,
        k1_r_writer,
        valid_writer,
        kv_base,
        tkv_base,
        t_base,
        topk_len_ptr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        VG: tl.constexpr,
        SKV,
        TOPK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
    ):
        topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
        max_col = SKV - 1
        stride_kvn: tl.constexpr = VG * (TD + D)
        NK = tl.cdiv(topk_len, BK)
        NPAIRS = tl.cdiv(NK, 2)
        offs_t = tl.arange(0, BK)
        offs_tile = tl.arange(0, 64)
        kv_tile_rows = tl.broadcast_to(offs_t[:, None], (BK, 64))
        for pair in tl.range(NPAIRS):
            ck0 = pair * 2
            ck1 = ck0 + 1
            t_offs0 = BK * ck0 + offs_t
            t_msk0 = t_offs0 < topk_len
            kv_ids0 = tl.load(t_base + t_offs0, t_msk0, other=-1)
            valid0 = t_msk0 & (kv_ids0 <= max_col) & (kv_ids0 >= 0)
            kv_offsets0 = tl.where(valid0, kv_ids0, 0).to(tl.int64) * stride_kvn

            t_offs1 = BK * ck1 + offs_t
            t_msk1 = t_offs1 < topk_len
            kv_ids1 = tl.load(t_base + t_offs1, t_msk1, other=-1)
            valid1 = t_msk1 & (kv_ids1 <= max_col) & (kv_ids1 >= 0)
            kv_offsets1 = tl.where(valid1, kv_ids1, 0).to(tl.int64) * stride_kvn

            k0_l_slot = k0_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k0_l_ptr = kv_base + kv_offsets0[:, None] + k_cols[None, :]
                k0_l_msk = valid0[:, None] & (k_cols < D)[None, :]
                k0_l_blk = tl.load(
                    k0_l_ptr,
                    mask=k0_l_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k0_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k0_l_blk,
                    mask=k0_l_msk,
                )
            k0_l_writer.commit(pair)

            k1_r_slot = k1_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k1_r_ptr = kv_base + kv_offsets1[:, None] + k_cols[None, :]
                k1_r_msk = valid1[:, None] & (k_cols < D)[None, :]
                k1_r_blk = tl.load(
                    k1_r_ptr,
                    mask=k1_r_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k1_r_blk,
                    mask=k1_r_msk,
                )
            if HAVE_TAIL:
                offs_td = tl.arange(0, TDP)
                k1_r_tail_ptr = tkv_base + kv_offsets1[:, None] + offs_td[None, :]
                k1_r_tail_msk = valid1[:, None] & (offs_td < TD)[None, :]
                k1_r_tail_blk = tl.load(
                    k1_r_tail_ptr,
                    mask=k1_r_tail_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK_tail),
                    k1_r_tail_blk,
                    mask=k1_r_tail_msk,
                )
            k1_r_writer.commit(pair)

            k0_r_slot = k0_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k0_r_ptr = kv_base + kv_offsets0[:, None] + k_cols[None, :]
                k0_r_msk = valid0[:, None] & (k_cols < D)[None, :]
                k0_r_blk = tl.load(
                    k0_r_ptr,
                    mask=k0_r_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k0_r_blk,
                    mask=k0_r_msk,
                )
            if HAVE_TAIL:
                offs_td = tl.arange(0, TDP)
                k0_r_tail_ptr = tkv_base + kv_offsets0[:, None] + offs_td[None, :]
                k0_r_tail_msk = valid0[:, None] & (offs_td < TD)[None, :]
                k0_r_tail_blk = tl.load(
                    k0_r_tail_ptr,
                    mask=k0_r_tail_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK_tail),
                    k0_r_tail_blk,
                    mask=k0_r_tail_msk,
                )
            k0_r_writer.commit(pair)

            k1_l_slot = k1_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k1_l_ptr = kv_base + kv_offsets1[:, None] + k_cols[None, :]
                k1_l_msk = valid1[:, None] & (k_cols < D)[None, :]
                k1_l_blk = tl.load(
                    k1_l_ptr,
                    mask=k1_l_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k1_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k1_l_blk,
                    mask=k1_l_msk,
                )
            k1_l_writer.commit(pair)

            valid_slot = valid_writer.acquire(pair)
            valid_row0 = tl.full([BK], 0, dtype=tl.int32)
            valid_row1 = tl.full([BK], 1, dtype=tl.int32)
            valid_ptr0 = tle.gpu.local_ptr(valid_slot.is_kv_valid, (valid_row0, offs_t))
            valid_ptr1 = tle.gpu.local_ptr(valid_slot.is_kv_valid, (valid_row1, offs_t))
            tl.store(valid_ptr0, valid0.to(tl.int8))
            tl.store(valid_ptr1, valid1.to(tl.int8))
            valid_writer.commit(pair)

    @triton.jit
    def _tle_flashmla_prefill_consumer0(
        q_writer,
        q_reader,
        q_desc,
        tq_desc,
        k0_l_reader,
        k0_r_qk_reader,
        k1_l_remote_reader,
        valid_reader,
        sM_wg0_writer,
        sM_wg1_reader,
        sS0_writer,
        sS1_reader,
        sL_wg0_writer,
        sL_wg1_reader,
        output_desc,
        output_row,
        h_base,
        topk_len_ptr,
        attn_sink_base,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        TOPK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
    ):
        topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
        offs_h = tl.arange(0, BH)
        offs_dh = tl.arange(0, DPH)
        mask_h = h_base + offs_h < G
        mask_od_l = offs_dh < D
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))

        q_write_slot = q_writer.acquire(0)
        tle.gpu.copy(q_desc, q_write_slot.sQ_l, [BH, DPH], [output_row, 0])
        tle.gpu.copy(q_desc, q_write_slot.sQ_r, [BH, DPH], [output_row, DPH])
        if HAVE_TAIL:
            tle.gpu.copy(tq_desc, q_write_slot.sQ_tail, [BH, TDP], [output_row, D])
        q_writer.commit(0)

        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_l = tl.zeros([BH, DPH], dtype=tl.float32)

        NK = tl.cdiv(topk_len, BK)
        NPAIRS = tl.cdiv(NK, 2)

        for pair in tl.range(NPAIRS):
            k0_l_wait = k0_l_reader.wait(pair)
            k0_l_slot = k0_l_wait.slot

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))

            qk0 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk0 = tl.dot(q_l_blk, tl.trans(k0_l_blk), qk0, out_dtype=tl.float32)

            k0_r_wait = k0_r_qk_reader.wait(pair)
            k0_r_slot = k0_r_wait.slot
            k0_r_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols_r)))
            qk0 = tl.dot(q_r_blk, tl.trans(k0_r_blk), qk0, out_dtype=tl.float32)
            if HAVE_TAIL:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k0_t_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK_tail))
                qk0 = tl.dot(q_tail_blk, tl.trans(k0_t_blk), qk0, out_dtype=tl.float32)

            valid_wait = valid_reader.wait(pair)
            row0 = tl.full([BK], 0, dtype=tl.int32)
            valid0 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_kv_valid, (row0, tl.arange(0, BK))
                    )
                )
                != 0
            )
            qk0 = tl.where(valid0[None, :], qk0, float("-inf"))
            valid_reader.release(pair)

            local_max = tl.maximum(max_prev, tl.max(qk0, axis=1))
            alpha = tl.math.exp2((max_prev - local_max) * log_scale)
            prob0 = tl.math.exp2(qk0 * log_scale - local_max[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob0, axis=1)
            acc_l = acc_l * alpha[:, None]
            prob0_b = prob0.to(OUT_DTYPE)

            sM_wg0_slot = sM_wg0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg0_slot.sM), local_max)
            sM_wg0_writer.commit(pair)

            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))
            acc_l = tl.dot(prob0_b, k0_l_blk, acc_l, out_dtype=tl.float32)
            k0_l_reader.release(pair)
            k0_r_qk_reader.release(pair)

            sM_wg1_wait = sM_wg1_reader.wait(pair)
            max_next = tl.load(tle.gpu.local_ptr(sM_wg1_wait.slot.sM))
            sM_wg1_reader.release(pair)

            final_scale = tl.math.exp2((local_max - max_next) * log_scale)
            sum_exp = sum_exp * final_scale
            acc_l = acc_l * final_scale[:, None]

            prob0_scaled = prob0 * final_scale[:, None]
            sS0_slot = sS0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS0_slot.sS0), prob0_scaled.to(OUT_DTYPE))
            sS0_writer.commit(pair)

            sS1_wait = sS1_reader.wait(pair)
            prob1 = tl.load(tle.gpu.local_ptr(sS1_wait.slot.sS1))
            k1_l_wait = k1_l_remote_reader.wait(pair)
            k1_l_blk = tl.load(
                tle.gpu.local_ptr(k1_l_wait.slot.sK, (kv_rows, kv_cols_l))
            )
            acc_l = tl.dot(prob1, k1_l_blk, acc_l, out_dtype=tl.float32)
            sS1_reader.release(pair)
            k1_l_remote_reader.release(pair)

            max_prev = max_next

        sL_wg0_slot = sL_wg0_writer.acquire(0)
        tl.store(tle.gpu.local_ptr(sL_wg0_slot.sL), sum_exp)
        sL_wg0_writer.commit(0)
        sL_wg1_wait = sL_wg1_reader.wait(1)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg1_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg1_reader.release(1)

        is_no_valid_tokens = total_sum == 0.0
        inv_total_sum = tl.fdiv(1.0, total_sum)
        out_l_vals = acc_l * inv_total_sum[:, None]
        if HAVE_ATTN_SINK:
            fin_log = (
                max_prev * log_scale + tl.math.log2(total_sum)
            ) * 0.6931471805599453
            sink = tl.load(attn_sink_base + h_base + offs_h, mask_h, other=0.0)
            sink_scale = tl.fdiv(1.0, 1.0 + tl.math.exp(sink - fin_log))
            out_l_vals = out_l_vals * sink_scale[:, None]
        out_l_vals = tl.where(is_no_valid_tokens[:, None], 0.0, out_l_vals)
        o_l_msk = mask_h[:, None] & mask_od_l[None, :]
        tl.store(q_l_smem_ptr, out_l_vals.to(OUT_DTYPE), o_l_msk)
        tle.gpu.copy(q_slot.sQ_l, output_desc, [BH, DPH], [output_row, 0])

    @triton.jit
    def _tle_flashmla_prefill_consumer1(
        q_reader,
        k1_r_reader,
        k1_l_qk_reader,
        k0_r_remote_reader,
        valid_reader,
        sM_wg1_writer,
        sM_wg0_reader,
        sS1_writer,
        sS0_reader,
        sL_wg1_writer,
        sL_wg0_reader,
        final_max_logits_smem,
        final_lse_smem,
        output_desc,
        output_row,
        max_logits_base,
        l_base,
        h_base,
        topk_len_ptr,
        attn_sink_base,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        TOPK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
    ):
        topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
        offs_h = tl.arange(0, BH)
        offs_dh = tl.arange(0, DPH)
        mask_h = h_base + offs_h < G
        mask_od_r = DPH + offs_dh < D
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))
        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_r = tl.zeros([BH, DPH], dtype=tl.float32)

        NK = tl.cdiv(topk_len, BK)
        NPAIRS = tl.cdiv(NK, 2)
        for pair in tl.range(NPAIRS):
            k1_r_wait = k1_r_reader.wait(pair)
            k1_r_slot = k1_r_wait.slot

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k1_r_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols_r)))

            qk1 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk1 = tl.dot(q_r_blk, tl.trans(k1_r_blk), qk1, out_dtype=tl.float32)
            if HAVE_TAIL:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k1_t_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK_tail))
                qk1 = tl.dot(q_tail_blk, tl.trans(k1_t_blk), qk1, out_dtype=tl.float32)
            k1_l_wait = k1_l_qk_reader.wait(pair)
            k1_l_slot = k1_l_wait.slot
            k1_l_blk = tl.load(tle.gpu.local_ptr(k1_l_slot.sK, (kv_rows, kv_cols_l)))
            qk1 = tl.dot(q_l_blk, tl.trans(k1_l_blk), qk1, out_dtype=tl.float32)

            valid_wait = valid_reader.wait(pair)
            row1 = tl.full([BK], 1, dtype=tl.int32)
            valid1 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_kv_valid, (row1, tl.arange(0, BK))
                    )
                )
                != 0
            )
            qk1 = tl.where(valid1[None, :], qk1, float("-inf"))
            valid_reader.release(pair)

            sM_wg0_wait = sM_wg0_reader.wait(pair)
            candidate0 = tl.load(tle.gpu.local_ptr(sM_wg0_wait.slot.sM))
            sM_wg0_reader.release(pair)

            candidate1 = tl.maximum(max_prev, tl.max(qk1, axis=1))
            max_next = tl.maximum(candidate1, candidate0)
            sM_wg1_slot = sM_wg1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg1_slot.sM), max_next)
            sM_wg1_writer.commit(pair)

            alpha = tl.math.exp2((max_prev - max_next) * log_scale)
            prob1 = tl.math.exp2(qk1 * log_scale - max_next[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob1, axis=1)
            acc_r = acc_r * alpha[:, None]
            prob1_b = prob1.to(OUT_DTYPE)

            k1_l_qk_reader.release(pair)

            acc_r = tl.dot(prob1_b, k1_r_blk, acc_r, out_dtype=tl.float32)

            sS1_slot = sS1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS1_slot.sS1), prob1_b)
            sS1_writer.commit(pair)

            sS0_wait = sS0_reader.wait(pair)
            prob0 = tl.load(tle.gpu.local_ptr(sS0_wait.slot.sS0))
            k0_r_wait = k0_r_remote_reader.wait(pair)
            k0_r_blk = tl.load(
                tle.gpu.local_ptr(k0_r_wait.slot.sK, (kv_rows, kv_cols_r))
            )
            acc_r = tl.dot(prob0, k0_r_blk, acc_r, out_dtype=tl.float32)
            k1_r_reader.release(pair)
            sS0_reader.release(pair)
            k0_r_remote_reader.release(pair)
            max_prev = max_next

        sL_wg1_slot = sL_wg1_writer.acquire(1)
        tl.store(tle.gpu.local_ptr(sL_wg1_slot.sL), sum_exp)
        sL_wg1_writer.commit(1)
        sL_wg0_wait = sL_wg0_reader.wait(0)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg0_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg0_reader.release(0)

        is_no_valid_tokens = total_sum == 0.0
        inv_total_sum = tl.fdiv(1.0, total_sum)
        out_r_vals = acc_r * inv_total_sum[:, None]
        final_max_logits_log2 = max_prev * log_scale
        final_max_logits = final_max_logits_log2 * 0.6931471805599453
        fin_log = (final_max_logits_log2 + tl.math.log2(total_sum)) * 0.6931471805599453
        if HAVE_ATTN_SINK:
            sink = tl.load(attn_sink_base + h_base + offs_h, mask_h, other=0.0)
            sink_scale = tl.fdiv(1.0, 1.0 + tl.math.exp(sink - fin_log))
            out_r_vals = out_r_vals * sink_scale[:, None]
        out_r_vals = tl.where(is_no_valid_tokens[:, None], 0.0, out_r_vals)
        o_r_msk = mask_h[:, None] & mask_od_r[None, :]
        tl.store(q_r_smem_ptr, out_r_vals.to(OUT_DTYPE), o_r_msk)
        tle.gpu.copy(q_slot.sQ_r, output_desc, [BH, DPH], [output_row, DPH])

        final_max_logits = tl.where(is_no_valid_tokens, float("-inf"), final_max_logits)
        fin_log = tl.where(is_no_valid_tokens, float("inf"), fin_log)
        tl.store(tle.gpu.local_ptr(final_max_logits_smem), final_max_logits, mask_h)
        tl.store(tle.gpu.local_ptr(final_lse_smem), fin_log, mask_h)
        final_max_logits = tl.load(
            tle.gpu.local_ptr(final_max_logits_smem), mask_h, other=float("-inf")
        )
        fin_log = tl.load(tle.gpu.local_ptr(final_lse_smem), mask_h, other=float("inf"))
        tl.store(max_logits_base + offs_h, final_max_logits, mask_h)
        tl.store(l_base + offs_h, fin_log, mask_h)

    @triton.jit
    def _tle_flashmla_prefill_fwd(
        q_desc,
        tq_desc,
        output_desc,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale: tl.constexpr,
        output,
        max_logits,
        lse,
        SQ,
        H: tl.constexpr,
        DQK: tl.constexpr,
        SKV,
        TOPK: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DP: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
        VG: tl.constexpr,
        RH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        PAIR_BLOCKS: tl.constexpr,
    ):
        DPH: tl.constexpr = DP // 2
        stride_kvg: tl.constexpr = TD + D
        stride_tg = TOPK
        stride_tm = VG * stride_tg
        stride_lm = H
        stride_mm = H

        pid = tl.program_id(0)
        programs_per_q: tl.constexpr = VG * RH
        i_sq = pid // programs_per_q
        i_grh = pid % programs_per_q
        i_g = i_grh // RH
        i_rh = i_grh % RH
        h_base = i_rh * BH
        q_head_base = i_g * G + h_base
        i_sq64 = i_sq.to(tl.int64)
        i_g64 = i_g.to(tl.int64)
        q_head_base64 = q_head_base.to(tl.int64)
        kv_base = kv + i_g64 * stride_kvg
        tkv_base = kv_base + D
        t_base = indices + i_sq64 * stride_tm + i_g64 * stride_tg
        topk_len_ptr = topk_length + i_sq64 if HAVE_TOPK_LENGTH else indices
        attn_sink_base = attn_sink if HAVE_ATTN_SINK else max_logits
        max_logits_base = max_logits + i_sq64 * stride_mm + q_head_base64
        l_base = lse + i_sq64 * stride_lm + q_head_base64
        q_row = i_sq * H + q_head_base
        _ = output
        _ = SQ
        _ = DQK

        sQ_l_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sQ_r_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        if HAVE_TAIL:
            sQ_tail_smem = tle.gpu.alloc(
                [1, BH, TDP],
                dtype=kv.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
                sQ_tail=sQ_tail_smem,
            )
        else:
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
            )

        sK0_smem = tle.gpu.alloc(
            [1, BK, DP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sK1_smem = tle.gpu.alloc(
            [1, BK, DP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        if HAVE_TAIL:
            sK0_tail_smem = tle.gpu.alloc(
                [1, BK, TDP],
                dtype=kv.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
            sK1_tail_smem = tle.gpu.alloc(
                [1, BK, TDP],
                dtype=kv.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
            sS0_smem = sK0_tail_smem
        else:
            sS0_smem = tle.gpu.alloc(
                [1, BH, BK],
                dtype=kv.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
        is_kv_valid_smem = tle.gpu.alloc(
            [1, PAIR_BLOCKS, BK],
            dtype=tl.int8,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        k0_l_pipe = tle.pipe(
            capacity=1, scope="cta", name="flashmla_sK0_l", sK=sK0_smem
        )
        if HAVE_TAIL:
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sK0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
                sK_tail=sK0_tail_smem,
            )
        else:
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sK0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
            )
        k1_l_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flashmla_sK1_l",
            readers=("qk", "remote"),
            sK=sK1_smem,
        )
        if HAVE_TAIL:
            k1_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sK1_r",
                sK=sK1_smem,
                sK_tail=sK1_tail_smem,
            )
        else:
            k1_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sK1_r",
                sK=sK1_smem,
            )
        is_kv_valid_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flashmla_is_kv_valid_ready",
            readers=("wg0", "wg1"),
            is_kv_valid=is_kv_valid_smem,
        )

        sM_smem = tle.gpu.alloc(
            [1, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sS1_smem = tle.gpu.alloc(
            [1, BH, BK], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sL_smem = tle.gpu.alloc(
            [2, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        final_max_logits_smem = tle.gpu.alloc(
            [BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        final_lse_smem = tle.gpu.alloc(
            [BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sM_wg0_pipe = tle.pipe(
            capacity=1, scope="cta", name="flashmla_wg0_bunch_0_ready", sM=sM_smem
        )
        sM_wg1_pipe = tle.pipe(
            capacity=1, scope="cta", name="flashmla_wg1_bunch_0_ready", sM=sM_smem
        )
        sS0_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sS0", sS0=sS0_smem)
        sS1_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sS1", sS1=sS1_smem)
        sL_wg0_pipe = tle.pipe(
            capacity=2, scope="cta", name="flashmla_sL_wg0", sL=sL_smem
        )
        sL_wg1_pipe = tle.pipe(
            capacity=2, scope="cta", name="flashmla_sL_wg1", sL=sL_smem
        )

        log_scale: tl.constexpr = sm_scale * 1.4426950408889634

        tle.gpu.warp_specialize(
            [
                (
                    _tle_flashmla_prefill_consumer0,
                    (
                        q_pipe.writer(),
                        q_pipe.reader("wg0"),
                        q_desc,
                        tq_desc,
                        k0_l_pipe.reader(),
                        k0_r_pipe.reader("qk"),
                        k1_l_pipe.reader("remote", fields=("sK",)),
                        is_kv_valid_pipe.reader("wg0"),
                        sM_wg0_pipe.writer(),
                        sM_wg1_pipe.reader(),
                        sS0_pipe.writer(),
                        sS1_pipe.reader(),
                        sL_wg0_pipe.writer(),
                        sL_wg1_pipe.reader(),
                        output_desc,
                        q_row,
                        h_base,
                        topk_len_ptr,
                        attn_sink_base,
                        log_scale,
                        D,
                        TD,
                        kv.dtype.element_ty,
                        HAVE_ATTN_SINK,
                        TOPK,
                        HAVE_TOPK_LENGTH,
                        HAVE_TAIL,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        G,
                    ),
                ),
                (
                    _tle_flashmla_prefill_consumer1,
                    (
                        q_pipe.reader("wg1"),
                        k1_r_pipe.reader(),
                        k1_l_pipe.reader("qk"),
                        k0_r_pipe.reader("remote", fields=("sK",)),
                        is_kv_valid_pipe.reader("wg1"),
                        sM_wg1_pipe.writer(),
                        sM_wg0_pipe.reader(),
                        sS1_pipe.writer(),
                        sS0_pipe.reader(),
                        sL_wg1_pipe.writer(),
                        sL_wg0_pipe.reader(),
                        final_max_logits_smem,
                        final_lse_smem,
                        output_desc,
                        q_row,
                        max_logits_base,
                        l_base,
                        h_base,
                        topk_len_ptr,
                        attn_sink_base,
                        log_scale,
                        D,
                        TD,
                        kv.dtype.element_ty,
                        HAVE_ATTN_SINK,
                        TOPK,
                        HAVE_TOPK_LENGTH,
                        HAVE_TAIL,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        G,
                    ),
                ),
                (
                    _tle_flashmla_prefill_producer,
                    (
                        k0_l_pipe.writer(),
                        k0_r_pipe.writer(),
                        k1_l_pipe.writer(),
                        k1_r_pipe.writer(),
                        is_kv_valid_pipe.writer(),
                        kv_base,
                        tkv_base,
                        t_base,
                        topk_len_ptr,
                        D,
                        TD,
                        DPH,
                        TDP,
                        VG,
                        SKV,
                        TOPK,
                        HAVE_TOPK_LENGTH,
                        HAVE_TAIL,
                        BK,
                    ),
                ),
            ],
            [4, 4],
            [216, 72],
        )


def _flash_mla_sparse_tle_enabled() -> bool:
    value = os.environ.get("FLAGGEMS_FLASHMLA_SPARSE_TLE", "1").lower()
    return value not in {"0", "false", "off", "no"}


def _can_use_tle_flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    d_v: int,
    topk_length: Optional[torch.Tensor] = None,
) -> bool:
    if not (HAS_TLE_FLASHMLA_SPARSE and _flash_mla_sparse_tle_enabled()):
        return False
    if q.device.type != "cuda":
        return False
    SQ, HQ, DQK = q.shape
    _ = SQ
    HKV = kv.shape[1]
    TOPK = indices.shape[-1]
    return (
        d_v == 512
        and HKV == 1
        and DQK in (512, 576)
        and HQ % TLE_FLASHMLA_PREFILL_BH == 0
        and TOPK > 0
        and TOPK % 128 == 0
    )


def _set_triton_descriptor_allocator(device: torch.device) -> None:
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)


def _flash_mla_sparse_fwd_default(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by
            exp(lse) / (exp(lse) + exp(attn_sink)). +-inf in attn_sink will be handled normally (i.e., -inf has no
            effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32.
            If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]],
            ignoring later k/v tokens (even if provided in indices). In extremely rare cases (topk_length provided,
            there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token
            containing NaN), operator output will contain NaN, so please avoid this situation.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
    )
    SQ, HQ, DQK = q.shape
    SKV, HKV, _ = kv.shape

    assert d_v == 512, "Unsupported d_v"
    DV = d_v

    assert kv.shape[-1] == DQK
    _, _, TOPK = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (SQ,), "topk_length error shape"

    # check from FlashMLA
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 64 or HQ == 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"

    _ = SKV
    D = DV
    TD = DQK - D
    DP = triton.next_power_of_2(D)
    HAVE_TAIL = TD > 0
    TDP = triton.next_power_of_2(TD) if HAVE_TAIL else 1
    G = HQ // HKV
    BH = TLE_FLASHMLA_PREFILL_BH
    RH = G // BH
    BK = TLE_FLASHMLA_PREFILL_BK
    output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)

    def triton_grid(META):
        return (triton.cdiv(HQ, META["BH"]) * SQ,)

    if _can_use_tle_flash_mla_sparse_fwd(q, kv, indices, d_v, topk_length):
        from triton.tools.tensor_descriptor import TensorDescriptor

        _set_triton_descriptor_allocator(q.device)
        q_desc = TensorDescriptor(
            q, shape=[SQ * HQ, DQK], strides=[DQK, 1], block_shape=[BH, DP // 2]
        )
        if HAVE_TAIL:
            tq_desc = TensorDescriptor(
                q, shape=[SQ * HQ, DQK], strides=[DQK, 1], block_shape=[BH, TDP]
            )
        else:
            tq_desc = q_desc
        output_desc = TensorDescriptor(
            output, shape=[SQ * HQ, D], strides=[D, 1], block_shape=[BH, DP // 2]
        )
        _tle_flashmla_prefill_fwd[triton_grid](
            q_desc,
            tq_desc,
            output_desc,
            kv,
            indices,
            attn_sink,
            topk_length,
            sm_scale,
            output,
            max_logits,
            lse,
            SQ,
            HQ,
            DQK,
            SKV,
            TOPK,
            attn_sink is not None,
            topk_length is not None,
            D,
            TD,
            DP,
            TDP,
            G,
            HKV,
            RH,
            HAVE_TAIL,
            BK,
            BH,
            TLE_FLASHMLA_PREFILL_PAIR_BLOCKS,
            num_warps=TLE_FLASHMLA_PREFILL_WORKER_NUM_WARPS,
            num_stages=1,
        )
        return output, max_logits, lse

    triton_flash_mla_sparse_fwd[triton_grid](
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale,
        output,
        max_logits,
        lse,
        q.stride(1),
        q.stride(0),
        kv.stride(1),
        kv.stride(0),
        indices.stride(1),
        indices.stride(0),
        output.stride(1),
        output.stride(0),
        max_logits.stride(0),
        lse.stride(0),
        SQ,
        HQ,
        DQK,
        SKV,
        TOPK,
        attn_sink is not None,
        topk_length is not None,
    )
    return output, max_logits, lse


def _flash_mla_sparse_fwd_hq4_optimized(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_stats: bool = True,
    *,
    pair_metadata: Optional[torch.Tensor] = None,
    pair_window_size: int = 0,
    quad_metadata: Optional[torch.Tensor] = None,
    max_kv_length: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by
            exp(lse) / (exp(lse) + exp(attn_sink)). +-inf in attn_sink will be handled normally (i.e., -inf has no
            effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32.
            If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]],
            ignoring later k/v tokens (even if provided in indices). In extremely rare cases (topk_length provided,
            there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token
            containing NaN), operator output will contain NaN, so please avoid this situation.
        out: optional pre-allocated output tensor, [s_q, h_q, d_v], bfloat16.
        return_stats: whether to return max_logits and lse. Set this to False
            when only the attention output is consumed.
        pair_metadata: optional packed int32 descriptors emitted by
            combine_topk_swa_indices. One descriptor covers two adjacent rows;
            structurally invalid or unsupported descriptors use the regular
            path. Descriptors must come from the same indices and topk_length
            tensors and must not be reused after either tensor is changed.
        pair_window_size: sliding-window width used to create pair_metadata.
        quad_metadata: optional int32 flags emitted together with pair_metadata
            by combine_topk_swa_indices(return_quad_metadata=True). Each flag
            certifies four adjacent queries from one request with exactly
            matching compressed prefixes and a contiguous sliding window.
            It must describe the same unmodified indices and topk_length.
        max_kv_length: optional CPU length hint including compressed and SWA
            entries. It only selects a quad launch configuration; attention
            masks and loop bounds always come from the device lengths.

    Returns:
        (output, max_logits, lse). max_logits and lse are None when
        return_stats is False.
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    assert q.stride(-1) == 1 and kv.stride(-1) == 1 and indices.stride(-1) == 1
    if type(return_stats) is not bool:
        raise TypeError("return_stats must be a bool")
    if type(pair_window_size) is not int:
        raise TypeError("pair_window_size must be an int")
    if max_kv_length is not None:
        if type(max_kv_length) is not int:
            raise TypeError("max_kv_length must be an int or None")
        if max_kv_length < 0:
            raise ValueError("max_kv_length must be non-negative")
    if pair_metadata is None:
        if pair_window_size != 0:
            raise ValueError("pair_window_size requires pair_metadata")
    elif not isinstance(pair_metadata, torch.Tensor):
        raise TypeError("pair_metadata must be a torch.Tensor or None")
    elif pair_window_size <= 0:
        raise ValueError("pair_window_size must be positive with pair_metadata")
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
    )
    assert q.device == kv.device and q.device == indices.device
    SQ, HQ, DQK = q.shape
    SKV, HKV, _ = kv.shape
    assert SKV > 0, "kv must contain at least one row"

    assert d_v == 512, "Unsupported d_v"
    DV = d_v

    assert kv.shape[-1] == DQK
    _, _, TOPK = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.device == q.device
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.device == q.device
        assert topk_length.shape == (SQ,), "topk_length error shape"
    if pair_metadata is not None:
        if topk_length is None:
            raise ValueError("pair_metadata requires topk_length")
        assert pair_metadata.is_contiguous()
        assert pair_metadata.dtype == torch.int32
        assert pair_metadata.device == q.device
        assert pair_metadata.shape == (triton.cdiv(SQ, 2),), "pair_metadata error shape"
    if quad_metadata is not None:
        if not isinstance(quad_metadata, torch.Tensor):
            raise TypeError("quad_metadata must be a torch.Tensor or None")
        if pair_metadata is None:
            raise ValueError("quad_metadata requires pair_metadata")
        assert quad_metadata.is_contiguous()
        assert quad_metadata.dtype == torch.int32
        assert quad_metadata.device == q.device
        assert quad_metadata.shape == (triton.cdiv(SQ, 4),), "quad_metadata error shape"

    # check from FlashMLA
    assert HKV == 1, "h_kv is expected to be 1"
    assert 0 < HQ <= 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"

    _ = SKV
    if out is None:
        output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    else:
        assert out.shape == (SQ, HQ, DV), "out error shape"
        assert out.dtype == q.dtype, "out error dtype"
        assert out.device == q.device, "out error device"
        assert out.stride(-1) == 1, "out must have a contiguous last dimension"
        output = out
    if return_stats:
        max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
        lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
        max_logits_ptr = max_logits
        lse_ptr = lse
        stride_mm = max_logits.stride(0)
        stride_lm = lse.stride(0)
    else:
        max_logits = None
        lse = None
        # Triton specializes RETURN_STATS at compile time, so these pointers
        # and strides are never dereferenced in the no-statistics variant.
        max_logits_ptr = output
        lse_ptr = output
        stride_mm = 0
        stride_lm = 0

    if (
        pair_metadata is not None
        and HQ == 4
        and DQK == 512
        and not return_stats
        and SQ >= 2
    ):
        if quad_metadata is not None and SQ >= 4:
            # Keeping fallback in a separate launch avoids forcing the
            # four-query kernel to reserve the pair kernel's register layout.
            # Flags come from the same producer launch as indices and lengths.
            common = (
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                pair_metadata,
                quad_metadata,
                output,
                q.stride(1),
                q.stride(0),
                kv.stride(0),
                indices.stride(0),
                output.stride(1),
                output.stride(0),
                SQ,
                SKV,
                sm_scale,
            )
            block_k = (
                64
                if SQ <= 2048 and max_kv_length is not None and max_kv_length > 512
                else 32
            )
            triton_flash_mla_sparse_fwd_hq4_quad[(triton.cdiv(SQ, 4),)](
                *common,
                TOPK,
                pair_window_size,
                attn_sink is not None,
                BK=block_k,
                GATHER_PIPELINE=block_k == 64 and _use_quad_gather_pipeline(q.device),
                num_warps=4,
                num_stages=3,
            )
            triton_flash_mla_sparse_fwd_hq4_quad_fallback[(SQ,)](
                *common,
                TOPK,
                pair_window_size,
                attn_sink is not None,
                BK=64,
                num_warps=4,
                num_stages=3,
            )
            return output, max_logits, lse
        pair_cache_ca = attn_sink is None
        single_cache_ca = attn_sink is not None
        triton_flash_mla_sparse_fwd_hq4_pair_work_items[(SQ,)](
            q,
            kv,
            indices,
            attn_sink,
            topk_length,
            pair_metadata,
            output,
            q.stride(1),
            q.stride(0),
            kv.stride(0),
            indices.stride(0),
            output.stride(1),
            output.stride(0),
            SQ,
            SKV,
            sm_scale,
            TOPK,
            pair_window_size,
            attn_sink is not None,
            pair_cache_ca,
            single_cache_ca,
            BK=64,
            BH=4,
            num_warps=4,
            num_stages=3,
        )
        return output, max_logits, lse

    if HQ == 4 and DQK == 512:
        triton_flash_mla_sparse_fwd_hq4[(SQ,)](
            q,
            kv,
            indices,
            attn_sink,
            topk_length,
            sm_scale,
            output,
            max_logits_ptr,
            lse_ptr,
            q.stride(1),
            q.stride(0),
            kv.stride(0),
            indices.stride(0),
            output.stride(1),
            output.stride(0),
            stride_mm,
            stride_lm,
            SQ,
            SKV,
            TOPK,
            attn_sink is not None,
            topk_length is not None,
            return_stats,
        )
        return output, max_logits, lse

    raise NotImplementedError("the optimized helper only supports HQ4/DQK512")


def _tensors_share_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Conservatively detect shared storage without launching a device kernel."""
    overlaps = getattr(torch._C, "_overlaps", None)
    if overlaps is not None:
        return bool(overlaps(first, second))
    return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    enable_hq4_sparse_prefill: bool = False,
    logical_num_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    return_stats: bool = True,
    pair_metadata: Optional[torch.Tensor] = None,
    pair_window_size: int = 0,
    quad_metadata: Optional[torch.Tensor] = None,
    max_kv_length: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run sparse MLA attention, with an explicitly enabled HQ4 specialization.

    The default call is intentionally identical to the existing HQ64/HQ128
    implementation, including its optional TLE dispatch. The HQ4 path is
    standard Triton and is selected only when enable_hq4_sparse_prefill is
    true.

    logical_num_heads=4 permits q to be a view backed by a head-padded
    allocation, or a tensor whose physical head dimension is larger than
    four. Only the first four heads are read, and out always has logical shape
    [SQ, 4, 512]; a strided view into padded output storage is supported.

    Pair and quad descriptors are certificates, not standalone indices. They
    must be produced by combine_topk_swa_indices for the exact indices and
    topk_length tensors, on the same stream, and consumed before either tensor
    is modified. Structurally invalid descriptors fall back to the
    single/pair Triton path. A stale but structurally valid certificate cannot
    be detected without repeating the producer comparisons.

    The explicit path is inference-only. Its output storage must not alias any
    input or metadata tensor because independently scheduled Triton programs
    may still be reading those tensors while another program writes output.
    """
    if type(enable_hq4_sparse_prefill) is not bool:
        raise TypeError("enable_hq4_sparse_prefill must be a bool")

    uses_hq4_only_argument = (
        logical_num_heads is not None
        or out is not None
        or return_stats is not True
        or pair_metadata is not None
        or pair_window_size != 0
        or quad_metadata is not None
        or max_kv_length is not None
    )
    if not enable_hq4_sparse_prefill:
        if uses_hq4_only_argument:
            raise ValueError(
                "HQ4-only arguments require " "enable_hq4_sparse_prefill=True"
            )
        return _flash_mla_sparse_fwd_default(
            q,
            kv,
            indices,
            sm_scale,
            d_v,
            attn_sink,
            topk_length,
        )

    if not all(isinstance(value, torch.Tensor) for value in (q, kv, indices)):
        raise TypeError("q, kv, and indices must be torch.Tensor instances")
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("q, kv, and indices must all be 3D")
    if logical_num_heads is not None and type(logical_num_heads) is not int:
        raise TypeError("logical_num_heads must be an int or None")
    logical_hq = q.shape[1] if logical_num_heads is None else logical_num_heads
    if logical_hq != 4:
        raise NotImplementedError(
            "enable_hq4_sparse_prefill supports exactly four logical heads"
        )
    if q.shape[1] < logical_hq:
        raise ValueError("q has fewer physical heads than logical_num_heads")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise NotImplementedError("the HQ4 sparse prefill path requires BF16")
    if indices.dtype != torch.int32:
        raise ValueError("indices must have dtype torch.int32")
    if q.stride(-1) != 1 or kv.stride(-1) != 1 or indices.stride(-1) != 1:
        raise ValueError("q, kv, and indices must have a contiguous last dimension")
    if d_v != 512 or q.shape[2] != 512 or kv.shape[2] != 512:
        raise NotImplementedError("the HQ4 sparse prefill path requires DQK=DV=512")
    if kv.shape[1] != 1:
        raise NotImplementedError("the HQ4 sparse prefill path requires one KV head")
    if q.shape[0] <= 0 or kv.shape[0] <= 0 or indices.shape[2] <= 0:
        raise ValueError("SQ, SKV, and TOPK must all be positive")
    if indices.shape[:2] != (q.shape[0], 1):
        raise ValueError("indices must have shape [SQ, 1, TOPK]")
    for name, value in (
        ("attn_sink", attn_sink),
        ("topk_length", topk_length),
        ("pair_metadata", pair_metadata),
        ("quad_metadata", quad_metadata),
    ):
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor or None")
        if out.shape != (q.shape[0], logical_hq, 512):
            raise ValueError("out must have logical shape [SQ, 4, 512]")
        if out.dtype != q.dtype or out.device != q.device:
            raise ValueError("out must match q's dtype and device")
        if out.stride(-1) != 1:
            raise ValueError("out must have a contiguous last dimension")
    for name, value in (
        ("q", q),
        ("kv", kv),
        ("attn_sink", attn_sink),
        ("out", out),
    ):
        if isinstance(value, torch.Tensor) and value.requires_grad:
            raise RuntimeError(
                "enable_hq4_sparse_prefill is inference-only; "
                f"{name}.requires_grad must be False"
            )
    if out is not None:
        for name, value in (
            ("q", q),
            ("kv", kv),
            ("indices", indices),
            ("attn_sink", attn_sink),
            ("topk_length", topk_length),
            ("pair_metadata", pair_metadata),
            ("quad_metadata", quad_metadata),
        ):
            if isinstance(value, torch.Tensor) and _tensors_share_storage(out, value):
                raise ValueError(f"out must not share storage with {name}")
    if pair_metadata is not None:
        if pair_window_size != 128 or indices.shape[2] != 2176:
            raise NotImplementedError(
                "metadata reuse requires W=128 and a 2048+128 index row"
            )
        if return_stats:
            raise ValueError("metadata reuse requires return_stats=False")
    if quad_metadata is not None and pair_metadata is None:
        raise ValueError("quad_metadata requires pair_metadata")
    if max_kv_length is not None and quad_metadata is None:
        raise ValueError("max_kv_length requires quad_metadata")
    if q.device.type != "cuda":
        raise NotImplementedError("the HQ4 sparse prefill path requires CUDA")
    if q.device != kv.device or q.device != indices.device:
        raise ValueError("q, kv, and indices must be on the same device")

    q_logical = q[:, :logical_hq, :]
    return _flash_mla_sparse_fwd_hq4_optimized(
        q_logical,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
        out,
        return_stats,
        pair_metadata=pair_metadata,
        pair_window_size=pair_window_size,
        quad_metadata=quad_metadata,
        max_kv_length=max_kv_length,
    )

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

"""Device-adaptive direct decode with in-kernel Split-KV reduction."""

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.common import (
    _paged_blockwise_cache_indices,
    _softmax_online_deferred,
    _split_n_block_range,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.config import (
    block_k,
    direct_configs,
    prune_direct_configs,
)
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def _decode_step(
    q,
    k_base,
    v_base,
    page_table_ptr,
    bid,
    block,
    k_len,
    k_row_stride,
    v_row_stride,
    k_batch_stride,
    page_table_batch_stride,
    acc,
    rowmax,
    rowsum,
    d: tl.constexpr,
    block_size: tl.constexpr,
    scale_log2: tl.constexpr,
    IS_PAGED: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_K: tl.constexpr,
    INDEX_INT64: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_K)
    indices = block * BLOCK_N + cols
    if IS_PAGED:
        physical = _paged_blockwise_cache_indices(
            block * BLOCK_N,
            cols,
            k_len,
            page_table_ptr + bid * page_table_batch_stride,
            block_size,
            k_batch_stride // k_row_stride,
            BLOCK_N,
            2,
            BOUNDARY_CHECK=MASK_K,
        )
    else:
        physical = indices.to(tl.int64) if INDEX_INT64 else indices
    k = tl.load(
        k_base + physical[None, :] * k_row_stride + dims[:, None],
        mask=(dims[:, None] < d) & ((indices[None, :] < k_len) | (not MASK_K)),
        other=0.0,
    )
    v = tl.load(
        v_base + physical[:, None] * v_row_stride + dims[None, :],
        mask=((indices[:, None] < k_len) | (not MASK_K)) & (dims[None, :] < d),
        other=0.0,
    )
    scores = tl.dot(q, k, out_dtype=tl.float32)
    if MASK_K:
        scores = tl.where(indices[None, :] < k_len, scores, float("-inf"))
    alpha, prob, rowmax, rowsum = _softmax_online_deferred(
        scores, rowmax, rowsum, softmax_scale_log2e=scale_log2, IS_BORDER=MASK_K
    )
    acc *= alpha[:, None]
    acc = tl.dot(prob.to(v_base.dtype.element_ty), v, acc, out_dtype=tl.float32)
    return acc, rowmax, rowsum


@triton.jit
def _decode_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    page_table_ptr,
    q_bos,
    k_bos,
    k_len,
    bid,
    head_pid,
    n_min,
    n_max,
    q_row_stride,
    q_head_stride,
    k_row_stride,
    k_head_stride,
    v_row_stride,
    v_head_stride,
    k_batch_stride,
    page_table_batch_stride,
    ratio: tl.constexpr,
    d: tl.constexpr,
    block_size: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_log2: tl.constexpr,
    IS_PAGED: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_LSE: tl.constexpr = True,
    SINGLE_TILE: tl.constexpr = False,
    INDEX_INT64: tl.constexpr = False,
):
    rows = tl.arange(0, 16)
    dims = tl.arange(0, BLOCK_K)
    heads = head_pid * ratio + rows
    q = tl.load(
        q_ptr + q_bos * q_row_stride + heads[:, None] * q_head_stride + dims[None, :],
        mask=(rows[:, None] < ratio) & (dims[None, :] < d),
        other=0.0,
    )
    if IS_PAGED:
        k_base = k_ptr + head_pid * k_head_stride
        v_base = v_ptr + head_pid * v_head_stride
    else:
        # Dense K/V larger than 2**31 elements overflow int32 row offsets.
        k_row = tl.cast(k_bos, tl.int64) if INDEX_INT64 else k_bos
        k_base = k_ptr + k_row * k_row_stride + head_pid * k_head_stride
        v_base = v_ptr + k_row * v_row_stride + head_pid * v_head_stride
    rowmax = tl.full([16], float("-inf"), tl.float32)
    rowsum = tl.zeros([16], tl.float32)
    acc = tl.zeros([16, BLOCK_K], tl.float32)
    if SINGLE_TILE:
        acc, rowmax, rowsum = _decode_step(
            q,
            k_base,
            v_base,
            page_table_ptr,
            bid,
            0,
            k_len,
            k_row_stride,
            v_row_stride,
            k_batch_stride,
            page_table_batch_stride,
            acc,
            rowmax,
            rowsum,
            d,
            block_size,
            scale_log2,
            IS_PAGED,
            BLOCK_K,
            BLOCK_N,
            True,
            INDEX_INT64,
        )
    else:
        # Process the possible boundary once; full blocks need no KV/page mask.
        dense_end = tl.minimum(n_max, k_len // BLOCK_N)
        if n_max > n_min and n_max > dense_end:
            acc, rowmax, rowsum = _decode_step(
                q,
                k_base,
                v_base,
                page_table_ptr,
                bid,
                n_max - 1,
                k_len,
                k_row_stride,
                v_row_stride,
                k_batch_stride,
                page_table_batch_stride,
                acc,
                rowmax,
                rowsum,
                d,
                block_size,
                scale_log2,
                IS_PAGED,
                BLOCK_K,
                BLOCK_N,
                True,
                INDEX_INT64,
            )
        # Dense D256 loads need one stage to stay within shared-memory capacity.
        for block in tl.range(
            dense_end - 1, n_min - 1, -1, num_stages=3 if IS_PAGED else 1
        ):
            acc, rowmax, rowsum = _decode_step(
                q,
                k_base,
                v_base,
                page_table_ptr,
                bid,
                block,
                k_len,
                k_row_stride,
                v_row_stride,
                k_batch_stride,
                page_table_batch_stride,
                acc,
                rowmax,
                rowsum,
                d,
                block_size,
                scale_log2,
                IS_PAGED,
                BLOCK_K,
                BLOCK_N,
                False,
                INDEX_INT64,
            )
    invalid = (rowsum == 0.0) | (rowsum != rowsum)
    acc *= tl.where(invalid, 1.0, 1.0 / rowsum)[:, None]
    lse = tl.full([16], 0.0, tl.float32)
    if COMPUTE_LSE:
        lse = tl.where(invalid, float("-inf"), rowmax * scale_softmax + tl.log(rowsum))
    return acc, lse


@triton.jit
def _combine_decode_splits(
    o_ptr,
    lse_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    q_bos,
    head_pid,
    o_row_stride,
    o_head_stride,
    total_q,
    split_count,
    h: tl.constexpr,
    ratio: tl.constexpr,
    d: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    # Four partials at a time bound register use independently of the split cap.
    heads = head_pid * ratio + tl.arange(0, triton.next_power_of_2(ratio))
    valid_heads = tl.arange(0, triton.next_power_of_2(ratio)) < ratio
    splits = tl.arange(0, triton.next_power_of_2(MAX_SPLITS))
    rows = (splits[None, :] * h + heads[:, None]) * total_q + q_bos
    lses = tl.load(
        partial_lse_ptr + rows,
        mask=valid_heads[:, None] & (splits[None, :] < split_count),
        other=float("-inf"),
        cache_modifier=".cg",
    )
    lse_max = tl.max(lses, 1)
    lse_safe = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    weights = tl.where(lses != float("-inf"), tl.exp(lses - lse_safe[:, None]), 0.0)
    weight_sum = tl.sum(weights, 1)
    cols = tl.arange(0, BLOCK_K)
    acc = tl.zeros([triton.next_power_of_2(ratio), BLOCK_K], tl.float32)
    for group in range(tl.cdiv(split_count, 4)):
        part_ids = group * 4 + tl.arange(0, 4)
        part_rows = (part_ids[None, :] * h + heads[:, None]) * total_q + q_bos
        valid_parts = valid_heads[:, None] & (part_ids[None, :] < split_count)
        part_lses = tl.load(
            partial_lse_ptr + part_rows,
            mask=valid_parts,
            other=float("-inf"),
            cache_modifier=".cg",
        )
        weight = tl.where(
            part_lses != float("-inf"), tl.exp(part_lses - lse_safe[:, None]), 0.0
        )
        parts = tl.load(
            partial_out_ptr + part_rows[:, :, None] * d + cols[None, None, :],
            mask=valid_parts[:, :, None] & (cols[None, None, :] < d),
            other=0.0,
            cache_modifier=".cg",
        )
        acc += tl.sum(parts * weight[:, :, None], 1)
    invalid = (weight_sum == 0.0) | (weight_sum != weight_sum)
    acc *= tl.where(invalid, 1.0, 1.0 / weight_sum)[:, None]
    output = (
        o_ptr + q_bos * o_row_stride + heads[:, None] * o_head_stride + cols[None, :]
    )
    tl.store(
        output,
        acc.to(o_ptr.dtype.element_ty),
        mask=valid_heads[:, None] & (cols[None, :] < d),
    )
    if STORE_LSE:
        tl.store(
            lse_ptr + heads * total_q + q_bos,
            tl.where(invalid, float("-inf"), lse_max + tl.log(weight_sum)),
            mask=valid_heads,
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("flash_attn_varlen_fa3_direct")
    or direct_configs(),
    prune_configs_by={"early_config_prune": prune_direct_configs},
    use_cuda_graph=True,
    rep=20,
    key=[
        "h",
        "hk",
        "d",
        "block_size",
        "is_paged",
        "is_causal",
        "is_local",
        "is_alibi",
        "is_s_aux",
        "seqlen_q",
        "TUNING_BATCH",
        "TUNING_K",
        "TUNING_Q",
        "DIRECT_SHAPE_BUCKET",
        "PACK_GQA",
        "BATCH_FIRST_GRID",
        "SINGLE_KV_TILE",
        "STORE_LSE",
        "RAGGED_SCHEDULER",
        "PAGED_GATHER_MODE",
        "PAGED_KV_NON_TMA",
        "MAX_SPLITS",
        "DECODE_POLICY_VERSION",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_K": block_k,
    }
)
@triton.jit(
    do_not_specialize=[
        "k_batch_stride",
        "b",
        "bk",
        "seqlen_q",
        "seqlen_k",
        "total_q",
        "page_table_batch_stride",
        "TUNING_BATCH",
        "TUNING_K",
        "TUNING_Q",
    ]
)
def flash_varlen_fwd_v3_tle_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    k_batch_stride,
    cu_seqlens_q_ptr,
    is_seqused_k: tl.constexpr,
    cu_seqlens_k_ptr,
    seqused_k_ptr,
    b,
    bk,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    d: tl.constexpr,
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    is_paged: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    is_s_aux: tl.constexpr,
    s_aux_ptr,
    total_q,
    page_table_ptr,
    page_table_batch_stride,
    block_size: tl.constexpr,
    partial_out_ptr,
    partial_lse_ptr,
    split_counter_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIRECT_SHAPE_BUCKET: tl.constexpr,
    PACK_GQA: tl.constexpr,
    BATCH_FIRST_GRID: tl.constexpr,
    SINGLE_KV_TILE: tl.constexpr,
    STORE_LSE: tl.constexpr,
    PAGED_KV_NON_TMA: tl.constexpr,
    RAGGED_SCHEDULER: tl.constexpr,
    HEADS_IN_L2: tl.constexpr,
    PAGED_GATHER_MODE: tl.constexpr = 2,
    MAX_SPLITS: tl.constexpr = 1,
    TUNING_BATCH=0,
    TUNING_K=0,
    TUNING_Q=0,
    DECODE_SHAPE_BUCKET=(),
    DECODE_POLICY_VERSION: tl.constexpr = 0,
    INDEX_INT64: tl.constexpr = False,
):
    tl.static_assert(PACK_GQA and BLOCK_M == 16 and h_hk_ratio <= 16 and MAX_SPLITS > 1)
    bid = tl.program_id(1)
    head_pid = tl.program_id(2) // MAX_SPLITS
    split_id = tl.program_id(2) % MAX_SPLITS
    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
        k_bos = 0
    else:
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32) - k_bos
    split_count = tl.where(k_len < 1024, 1, tl.minimum(MAX_SPLITS, tl.cdiv(k_len, 128)))
    if q_bos < q_eos and split_id < split_count:
        rows = tl.arange(0, 16)
        dims = tl.arange(0, BLOCK_K)
        heads = head_pid * h_hk_ratio + rows
        valid = (rows[:, None] < h_hk_ratio) & (dims[None, :] < d)
        if k_len <= 1:
            # A graph's maximum can be a capacity, including during warmup.
            q = tl.load(
                q_ptr
                + q_bos * q_row_stride
                + heads[:, None] * q_head_stride
                + dims[None, :],
                mask=valid,
                other=0.0,
            )
            if is_paged:
                page = tl.load(
                    page_table_ptr + bid * page_table_batch_stride,
                    mask=k_len > 0,
                    other=0,
                ).to(tl.int64)
                k_base = k_ptr + page * k_batch_stride + head_pid * k_head_stride
                v_base = v_ptr + page * k_batch_stride + head_pid * v_head_stride
            else:
                k_row = tl.cast(k_bos, tl.int64) if INDEX_INT64 else k_bos
                k_base = k_ptr + k_row * k_row_stride + head_pid * k_head_stride
                v_base = v_ptr + k_row * v_row_stride + head_pid * v_head_stride
            kval = tl.load(k_base + dims, mask=(dims < d) & (k_len > 0), other=0.0)
            vval = tl.load(v_base + dims, mask=(dims < d) & (k_len > 0), other=0.0)
            score = tl.sum(q.to(tl.float32) * kval[None, :].to(tl.float32), 1)
            score_safe = tl.where(score == float("-inf"), 0.0, score)
            prob = tl.where(
                k_len > 0,
                tl.exp2(score * scale_softmax_log2 - score_safe * scale_softmax_log2),
                0.0,
            )
            invalid = (prob == 0.0) | (prob != prob)
            acc = prob.to(v_ptr.dtype.element_ty)[:, None].to(tl.float32) * vval[
                None, :
            ].to(tl.float32)
            acc *= tl.where(invalid, 1.0, 1.0 / prob)[:, None]
            lse = tl.where(invalid, float("-inf"), score * scale_softmax + tl.log(prob))
        elif k_len <= 64:
            acc, lse = _decode_attention(
                q_ptr,
                k_ptr,
                v_ptr,
                page_table_ptr,
                q_bos,
                k_bos,
                k_len,
                bid,
                head_pid,
                0,
                1,
                q_row_stride,
                q_head_stride,
                k_row_stride,
                k_head_stride,
                v_row_stride,
                v_head_stride,
                k_batch_stride,
                page_table_batch_stride,
                h_hk_ratio,
                d,
                block_size,
                scale_softmax,
                scale_softmax_log2,
                is_paged,
                BLOCK_K,
                64,
                STORE_LSE,
                True,
                INDEX_INT64=INDEX_INT64,
            )
        elif k_len <= 128:
            acc, lse = _decode_attention(
                q_ptr,
                k_ptr,
                v_ptr,
                page_table_ptr,
                q_bos,
                k_bos,
                k_len,
                bid,
                head_pid,
                0,
                1,
                q_row_stride,
                q_head_stride,
                k_row_stride,
                k_head_stride,
                v_row_stride,
                v_head_stride,
                k_batch_stride,
                page_table_batch_stride,
                h_hk_ratio,
                d,
                block_size,
                scale_softmax,
                scale_softmax_log2,
                is_paged,
                BLOCK_K,
                128,
                STORE_LSE,
                True,
                INDEX_INT64=INDEX_INT64,
            )
        elif k_len < 1024:
            acc, lse = _decode_attention(
                q_ptr,
                k_ptr,
                v_ptr,
                page_table_ptr,
                q_bos,
                k_bos,
                k_len,
                bid,
                head_pid,
                0,
                tl.cdiv(k_len, 128),
                q_row_stride,
                q_head_stride,
                k_row_stride,
                k_head_stride,
                v_row_stride,
                v_head_stride,
                k_batch_stride,
                page_table_batch_stride,
                h_hk_ratio,
                d,
                block_size,
                scale_softmax,
                scale_softmax_log2,
                is_paged,
                BLOCK_K,
                128,
                STORE_LSE,
                INDEX_INT64=INDEX_INT64,
            )
        else:
            n_min, n_max = _split_n_block_range(
                0, tl.cdiv(k_len, BLOCK_N), split_id, split_count
            )
            acc, lse = _decode_attention(
                q_ptr,
                k_ptr,
                v_ptr,
                page_table_ptr,
                q_bos,
                k_bos,
                k_len,
                bid,
                head_pid,
                n_min,
                n_max,
                q_row_stride,
                q_head_stride,
                k_row_stride,
                k_head_stride,
                v_row_stride,
                v_head_stride,
                k_batch_stride,
                page_table_batch_stride,
                h_hk_ratio,
                d,
                block_size,
                scale_softmax,
                scale_softmax_log2,
                is_paged,
                BLOCK_K,
                BLOCK_N,
                INDEX_INT64=INDEX_INT64,
            )
        if split_count == 1:
            output = (
                o_ptr
                + q_bos * o_row_stride
                + heads[:, None] * o_head_stride
                + dims[None, :]
            )
            tl.store(output, acc.to(o_ptr.dtype.element_ty), mask=valid)
            if STORE_LSE:
                tl.store(
                    softmax_lse_ptr + heads * total_q + q_bos,
                    lse,
                    mask=rows < h_hk_ratio,
                )
        else:
            part_rows = (split_id * h + heads) * total_q + q_bos
            tl.store(
                partial_out_ptr + part_rows[:, None] * d + dims[None, :],
                acc,
                mask=valid,
            )
            tl.store(partial_lse_ptr + part_rows, lse, mask=rows < h_hk_ratio)
            # All producer threads publish before the CTA's GPU-scope arrival.
            tl.debug_barrier()
            counter = split_counter_ptr + bid * hk + head_pid
            ticket = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
            if ticket == split_count - 1:
                _combine_decode_splits(
                    o_ptr,
                    softmax_lse_ptr,
                    partial_out_ptr,
                    partial_lse_ptr,
                    q_bos,
                    head_pid,
                    o_row_stride,
                    o_head_stride,
                    total_q,
                    split_count,
                    h,
                    h_hk_ratio,
                    d,
                    BLOCK_K,
                    MAX_SPLITS,
                    STORE_LSE,
                )
                tl.debug_barrier()
                tl.store(counter, 0)


@triton.jit(do_not_specialize=["numel"])
def _zero_decode_split_counters(counter_ptr, numel):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(counter_ptr + offsets, 0, mask=offsets < numel)

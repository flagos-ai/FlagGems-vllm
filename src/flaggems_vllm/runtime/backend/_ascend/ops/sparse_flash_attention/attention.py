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


"""General and split-K SFA attention kernels and output reductions."""

import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops.sparse_flash_attention.common import (
    TL_COMBINE_HEADS_PER_PROGRAM,
    TL_COMBINE_PROGRAMS_PER_QUERY,
    TL_COMBINE_TWO_HEADS_PER_PROGRAM,
    TL_COMBINE_TWO_PROGRAMS_PER_QUERY,
    TL_COMBINE_TWO_SPLITS,
    TL_COMBINE_TWO_VALUE_COLUMNS,
    TL_COMBINE_TWO_VALUE_TILES,
    TL_FAST_QUERY_HEADS,
    TL_FAST_SPARSE_ENTRIES,
    TL_GATHERED_HEADS_PER_GROUP,
    TL_GATHERED_KV_BLOCK_ROWS,
    TL_KV_DIM,
    TL_ROPE_DIM,
    TL_VLLM_FINITE_MASK_MAX_BLOCK_SIZE,
    TL_VLLM_FINITE_SOFTMAX_SENTINEL,
    _sfa_load_cache,
    _sfa_query_meta,
    _sfa_sparse_cache_rows,
)


@triton.jit
def _sfa_empty_row_stats(
    kv_end, NUM_SPARSE_BLOCKS: tl.constexpr, SPARSE_BLOCK_SIZE: tl.constexpr
):
    empty_max = 0.0
    empty_sum = 0.0
    if SPARSE_BLOCK_SIZE <= TL_VLLM_FINITE_MASK_MAX_BLOCK_SIZE:
        empty_max = tl.where(kv_end > 0, TL_VLLM_FINITE_SOFTMAX_SENTINEL, 0.0)
        empty_sum = tl.minimum(NUM_SPARSE_BLOCKS * SPARSE_BLOCK_SIZE, kv_end).to(
            tl.float32
        )
    return empty_max, empty_sum


@triton.jit
def _sfa_general(
    QUERY,
    QUERY_ROPE,
    KV,
    KEY_ROPE,
    SPARSE_INDICES,
    BLOCK_TABLE,
    QUERY_LENGTHS,
    KV_LENGTHS,
    OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    KV_END,
    NUM_REQUESTS: tl.constexpr,
    NUM_QUERY_TOKENS: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    KV_SEQ_CAPACITY: tl.constexpr,
    NUM_SPARSE_BLOCKS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
    SPARSE_BLOCK_SIZE: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    KV_IS_TND: tl.constexpr,
    PAGED: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    HAS_KV_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    RETURN_SOFTMAX_STATS: tl.constexpr,
    HEAD_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_ROWS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    KV_ACCESS_SPEC: tl.constexpr = None,
    KEY_ROPE_ACCESS_SPEC: tl.constexpr = None,
):
    HEAD_GROUPS: tl.constexpr = triton.cdiv(NUM_QUERY_HEADS, HEAD_BLOCK_SIZE)
    KV_BLOCKS: tl.constexpr = triton.cdiv(
        NUM_SPARSE_BLOCKS * SPARSE_BLOCK_SIZE, KV_BLOCK_ROWS
    )
    for task_idx in range(
        tl.program_id(0), NUM_QUERY_TOKENS * HEAD_GROUPS * NUM_SPLITS, NUM_CORES
    ):
        split = task_idx % NUM_SPLITS
        head_task = task_idx // NUM_SPLITS
        query_token_idx = (head_task // HEAD_GROUPS).to(tl.int64)
        head_group = head_task % HEAD_GROUPS
        head_indices = head_group * HEAD_BLOCK_SIZE + tl.arange(0, HEAD_BLOCK_SIZE)
        kv_columns = tl.arange(0, TL_KV_DIM)
        rope_columns = tl.arange(0, TL_ROPE_DIM)
        request_idx, kv_end_exclusive, kv_start_offset = _sfa_query_meta(
            query_token_idx,
            QUERY_LENGTHS,
            KV_LENGTHS,
            NUM_REQUESTS,
            QUERY_SEQ_CAPACITY,
            KV_SEQ_CAPACITY,
            QUERY_IS_TND,
            KV_IS_TND,
            HAS_QUERY_LENGTHS,
            HAS_KV_LENGTHS,
            CAUSAL,
        )
        if not KV_IS_TND and not PAGED:
            kv_start_offset = request_idx * KV_SEQ_CAPACITY
        query_tile = tl.load(
            QUERY
            + (query_token_idx * NUM_QUERY_HEADS + head_indices[:, None]) * TL_KV_DIM
            + kv_columns[None, :],
            head_indices[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        query_rope_tile = tl.load(
            QUERY_ROPE
            + (query_token_idx * NUM_QUERY_HEADS + head_indices[:, None]) * TL_ROPE_DIM
            + rope_columns[None, :],
            head_indices[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        softmax_max = tl.full((HEAD_BLOCK_SIZE,), float("-inf"), tl.float32)
        softmax_sum = tl.zeros((HEAD_BLOCK_SIZE,), tl.float32)
        weighted_value_sum = tl.zeros((HEAD_BLOCK_SIZE, TL_KV_DIM), tl.float32)
        # Balanced, disjoint ranges preserve global sparse positions and avoid
        # empty splits when the number of KV blocks is not divisible by splits.
        for kv_block_idx in range(
            split * KV_BLOCKS // NUM_SPLITS,
            (split + 1) * KV_BLOCKS // NUM_SPLITS,
        ):
            sparse_positions = kv_block_idx * KV_BLOCK_ROWS + tl.arange(
                0, KV_BLOCK_ROWS
            )
            cache_row_indices, valid_kv_mask = _sfa_sparse_cache_rows(
                SPARSE_INDICES,
                BLOCK_TABLE,
                query_token_idx,
                request_idx,
                kv_end_exclusive,
                kv_start_offset,
                sparse_positions,
                NUM_SPARSE_BLOCKS,
                SPARSE_BLOCK_SIZE,
                PAGED,
                PAGE_SIZE,
                PAGES_PER_REQUEST,
            )
            kv_tile = _sfa_load_cache(
                KV,
                cache_row_indices,
                kv_columns,
                TL_KV_DIM,
                KV_ACCESS_SPEC,
                valid_kv_mask,
            )
            key_rope_tile = _sfa_load_cache(
                KEY_ROPE,
                cache_row_indices,
                rope_columns,
                TL_ROPE_DIM,
                KEY_ROPE_ACCESS_SPEC,
                valid_kv_mask,
            )
            attention_scores = tl.dot(query_tile, tl.trans(kv_tile))
            attention_scores = (
                tl.dot(query_rope_tile, tl.trans(key_rope_tile), attention_scores)
                * SOFTMAX_SCALE
            )
            attention_scores = tl.where(
                valid_kv_mask[None, :], attention_scores, float("-inf")
            )
            updated_softmax_max = tl.maximum(softmax_max, tl.max(attention_scores, 1))
            softmax_exp_base = tl.where(
                updated_softmax_max == float("-inf"), 0.0, updated_softmax_max
            )
            previous_state_scale = tl.where(
                softmax_sum > 0, tl.exp(softmax_max - softmax_exp_base), 0.0
            )
            unnormalized_weights = tl.exp(attention_scores - softmax_exp_base[:, None])
            softmax_sum = softmax_sum * previous_state_scale + tl.sum(
                unnormalized_weights, 1
            )
            weighted_value_sum = weighted_value_sum * previous_state_scale[
                :, None
            ] + tl.dot(unnormalized_weights.to(KV.dtype.element_ty), kv_tile)
            softmax_max = updated_softmax_max
        if NUM_SPLITS > 1:
            row = (
                query_token_idx * NUM_SPLITS + split
            ) * NUM_QUERY_HEADS + head_indices
            tl.store(
                OUT + row[:, None] * TL_KV_DIM + kv_columns[None, :],
                weighted_value_sum,
                head_indices[:, None] < NUM_QUERY_HEADS,
            )
            tl.store(SOFTMAX_MAX + row, softmax_max, head_indices < NUM_QUERY_HEADS)
            tl.store(SOFTMAX_SUM + row, softmax_sum, head_indices < NUM_QUERY_HEADS)
            if RETURN_SOFTMAX_STATS:
                if head_group == 0 and split == 0:
                    tl.store(KV_END + query_token_idx, kv_end_exclusive)
        else:
            tl.store(
                OUT
                + (query_token_idx * NUM_QUERY_HEADS + head_indices[:, None])
                * TL_KV_DIM
                + kv_columns[None, :],
                weighted_value_sum
                / tl.where(softmax_sum > 0, softmax_sum, 1.0)[:, None],
                head_indices[:, None] < NUM_QUERY_HEADS,
            )
            if RETURN_SOFTMAX_STATS:
                # The installed vLLM template for sparse blocks of up to four tokens
                # uses a finite mask. Preserve its all-masked max/sum convention.
                empty_row_softmax_max, empty_row_softmax_sum = _sfa_empty_row_stats(
                    kv_end_exclusive, NUM_SPARSE_BLOCKS, SPARSE_BLOCK_SIZE
                )
                tl.store(
                    SOFTMAX_MAX + query_token_idx * NUM_QUERY_HEADS + head_indices,
                    tl.where(softmax_sum > 0, softmax_max, empty_row_softmax_max),
                    head_indices < NUM_QUERY_HEADS,
                )
                tl.store(
                    SOFTMAX_SUM + query_token_idx * NUM_QUERY_HEADS + head_indices,
                    tl.where(softmax_sum > 0, softmax_sum, empty_row_softmax_sum),
                    head_indices < NUM_QUERY_HEADS,
                )


@triton.jit
def _sfa_tiled_attention(
    QUERY,
    QUERY_ROPE,
    GATHERED_KV,
    GATHERED_KEY_ROPE,
    MASK,
    PARTIAL_OUT,
    PARTIAL_MAX,
    PARTIAL_SUM,
    NUM_QUERY_TOKENS: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    PADDED_SPARSE_BLOCKS: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    GENERAL: tl.constexpr = False,
    KV_END=None,
    NUM_SPARSE_BLOCKS: tl.constexpr = 0,
    SPARSE_BLOCK_SIZE: tl.constexpr = 1,
    RETURN_STATS: tl.constexpr = False,
):
    HEAD_GROUPS: tl.constexpr = triton.cdiv(NUM_QUERY_HEADS, HEAD_BLOCK)
    NUM_KV_BLOCKS: tl.constexpr = PADDED_SPARSE_BLOCKS // KV_BLOCK
    INITIAL_MAX: tl.constexpr = (
        float("-inf") if GENERAL else TL_VLLM_FINITE_SOFTMAX_SENTINEL
    )
    for task in range(
        tl.program_id(0), NUM_QUERY_TOKENS * HEAD_GROUPS * NUM_SPLITS, NUM_CORES
    ):
        split = task % NUM_SPLITS
        hg = (task // NUM_SPLITS) % HEAD_GROUPS
        token = (task // (NUM_SPLITS * HEAD_GROUPS)).to(tl.int64)
        heads = hg * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
        cols = tl.arange(0, TL_KV_DIM)
        q = tl.load(
            QUERY
            + (token * NUM_QUERY_HEADS + heads[:, None]) * TL_KV_DIM
            + cols[None, :],
            heads[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        qr = tl.load(
            QUERY_ROPE
            + (token * NUM_QUERY_HEADS + heads[:, None]) * TL_ROPE_DIM
            + tl.arange(0, TL_ROPE_DIM)[None, :],
            heads[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        sm = tl.full((HEAD_BLOCK,), INITIAL_MAX, tl.float32)
        ss = tl.zeros((HEAD_BLOCK,), tl.float32)
        acc = tl.zeros((HEAD_BLOCK, TL_KV_DIM), tl.float32)
        for block in range(split, NUM_KV_BLOCKS, NUM_SPLITS):
            k = tl.load(
                tl.make_block_ptr(
                    GATHERED_KV + token * PADDED_SPARSE_BLOCKS * TL_KV_DIM,
                    (PADDED_SPARSE_BLOCKS, TL_KV_DIM),
                    (TL_KV_DIM, 1),
                    (block * KV_BLOCK, 0),
                    (KV_BLOCK, TL_KV_DIM),
                    (1, 0),
                )
            )
            kr = tl.load(
                tl.make_block_ptr(
                    GATHERED_KEY_ROPE + token * PADDED_SPARSE_BLOCKS * TL_ROPE_DIM,
                    (PADDED_SPARSE_BLOCKS, TL_ROPE_DIM),
                    (TL_ROPE_DIM, 1),
                    (block * KV_BLOCK, 0),
                    (KV_BLOCK, TL_ROPE_DIM),
                    (1, 0),
                )
            )
            score = tl.dot(q, tl.trans(k))
            score = tl.dot(qr, tl.trans(kr), score) * SOFTMAX_SCALE
            valid = (
                tl.load(
                    MASK
                    + token * PADDED_SPARSE_BLOCKS
                    + block * KV_BLOCK
                    + tl.arange(0, KV_BLOCK)
                )
                == 0
            )
            score = tl.where(valid[None, :], score, float("-inf"))
            sm_next = tl.maximum(sm, tl.max(score, 1))
            if GENERAL:
                exp_base = tl.where(sm_next == float("-inf"), 0.0, sm_next)
                alpha = tl.where(ss > 0, tl.exp(sm - exp_base), 0.0)
                p = tl.exp(score - exp_base[:, None])
            else:
                alpha = tl.exp(sm - sm_next)
                p = tl.exp(score - sm_next[:, None])
            ss = ss * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(GATHERED_KV.dtype.element_ty), k)
            sm = sm_next
        partial_row = (token * NUM_SPLITS + split) * NUM_QUERY_HEADS + heads
        if GENERAL and NUM_SPLITS == 1:
            tl.store(
                PARTIAL_OUT + partial_row[:, None] * TL_KV_DIM + cols[None, :],
                acc / tl.where(ss > 0, ss, 1.0)[:, None],
                heads[:, None] < NUM_QUERY_HEADS,
            )
            if RETURN_STATS:
                kv_end = tl.load(KV_END + token)
                empty_max, empty_sum = _sfa_empty_row_stats(
                    kv_end, NUM_SPARSE_BLOCKS, SPARSE_BLOCK_SIZE
                )
                tl.store(
                    PARTIAL_MAX + partial_row,
                    tl.where(ss > 0, sm, empty_max),
                    heads < NUM_QUERY_HEADS,
                )
                tl.store(
                    PARTIAL_SUM + partial_row,
                    tl.where(ss > 0, ss, empty_sum),
                    heads < NUM_QUERY_HEADS,
                )
        else:
            tl.store(
                PARTIAL_OUT + partial_row[:, None] * TL_KV_DIM + cols[None, :],
                acc,
                heads[:, None] < NUM_QUERY_HEADS,
            )
            tl.store(PARTIAL_MAX + partial_row, sm, heads < NUM_QUERY_HEADS)
            tl.store(PARTIAL_SUM + partial_row, ss, heads < NUM_QUERY_HEADS)


@triton.jit
def _sfa_split_combine(
    PARTIAL_OUT,
    PARTIAL_MAX,
    PARTIAL_SUM,
    OUT,
    MAX_OUT,
    SUM_OUT,
    KV_END,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPARSE_BLOCKS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    RETURN_STATS: tl.constexpr,
    GENERAL: tl.constexpr = False,
    SPARSE_BLOCK_SIZE: tl.constexpr = 1,
):
    GROUPS: tl.constexpr = triton.cdiv(NUM_QUERY_HEADS, HEAD_BLOCK)
    INITIAL_MAX: tl.constexpr = (
        float("-inf") if GENERAL else TL_VLLM_FINITE_SOFTMAX_SENTINEL
    )
    token = tl.program_id(0) // GROUPS
    hg = tl.program_id(0) % GROUPS
    heads = hg * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    cols = tl.arange(0, TL_KV_DIM)
    sm = tl.full((HEAD_BLOCK,), INITIAL_MAX, tl.float32)
    for split in range(NUM_SPLITS):
        row = (token * NUM_SPLITS + split) * NUM_QUERY_HEADS + heads
        part_max = tl.load(
            PARTIAL_MAX + row,
            heads < NUM_QUERY_HEADS,
            other=TL_VLLM_FINITE_SOFTMAX_SENTINEL,
        )
        sm = tl.maximum(sm, part_max)
    acc = tl.zeros((HEAD_BLOCK, TL_KV_DIM), tl.float32)
    ss = tl.zeros((HEAD_BLOCK,), tl.float32)
    for split in range(NUM_SPLITS):
        row = (token * NUM_SPLITS + split) * NUM_QUERY_HEADS + heads
        part_max = tl.load(
            PARTIAL_MAX + row,
            heads < NUM_QUERY_HEADS,
            other=TL_VLLM_FINITE_SOFTMAX_SENTINEL,
        )
        part_sum = tl.load(PARTIAL_SUM + row, heads < NUM_QUERY_HEADS, other=0)
        part_out = tl.load(
            PARTIAL_OUT + row[:, None] * TL_KV_DIM + cols[None, :],
            heads[:, None] < NUM_QUERY_HEADS,
            other=0,
        )
        if GENERAL:
            exp_base = tl.where(sm == float("-inf"), 0.0, sm)
            alpha = tl.where(part_sum > 0, tl.exp(part_max - exp_base), 0.0)
        else:
            alpha = tl.exp(part_max - sm)
        ss += part_sum * alpha
        acc += part_out * alpha[:, None]
    tl.store(
        OUT + (token * NUM_QUERY_HEADS + heads[:, None]) * TL_KV_DIM + cols[None, :],
        acc / tl.where(ss > 0, ss, 1.0)[:, None],
        heads[:, None] < NUM_QUERY_HEADS,
    )
    if RETURN_STATS:
        kv_end = tl.load(KV_END + token)
        empty_max, empty_sum = _sfa_empty_row_stats(
            kv_end, NUM_SPARSE_BLOCKS, SPARSE_BLOCK_SIZE
        )
        tl.store(
            MAX_OUT + token * NUM_QUERY_HEADS + heads,
            tl.where(ss > 0, sm, empty_max),
            heads < NUM_QUERY_HEADS,
        )
        tl.store(
            SUM_OUT + token * NUM_QUERY_HEADS + heads,
            tl.where(ss > 0, ss, empty_sum),
            heads < NUM_QUERY_HEADS,
        )


@triton.jit
def _sfa_gathered_attention(
    QUERY,
    QUERY_ROPE,
    GATHERED_KV,
    GATHERED_KEY_ROPE,
    SPLIT_WEIGHTED_VALUE_SUM,
    SPLIT_SOFTMAX_MAX,
    SPLIT_SOFTMAX_SUM,
    OUT,
    ADDITIVE_ATTENTION_MASK,
    KV_END_EXCLUSIVE,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    NUM_QUERY_TOKENS: tl.constexpr,
    RETURN_SOFTMAX_STATS: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
):
    # Fixed specialization; head groups and KV blocks are compute tiles.
    NUM_QUERY_HEADS: tl.constexpr = TL_FAST_QUERY_HEADS
    KV_DIM: tl.constexpr = TL_KV_DIM
    ROPE_DIM: tl.constexpr = TL_ROPE_DIM
    HEADS_PER_GROUP: tl.constexpr = TL_GATHERED_HEADS_PER_GROUP
    NUM_HEAD_GROUPS: tl.constexpr = NUM_QUERY_HEADS // HEADS_PER_GROUP
    NUM_GATHERED_KV_ROWS: tl.constexpr = TL_FAST_SPARSE_ENTRIES
    KV_BLOCK_ROWS: tl.constexpr = TL_GATHERED_KV_BLOCK_ROWS
    NUM_KV_BLOCKS: tl.constexpr = NUM_GATHERED_KV_ROWS // KV_BLOCK_ROWS
    program_id = tl.program_id(0)
    for task_idx in range(
        program_id, NUM_QUERY_TOKENS * NUM_HEAD_GROUPS * NUM_KV_SPLITS, NUM_CORES
    ):
        kv_split_idx = task_idx % NUM_KV_SPLITS
        head_group_idx = (task_idx // NUM_KV_SPLITS) % NUM_HEAD_GROUPS
        query_token_idx = (task_idx // (NUM_HEAD_GROUPS * NUM_KV_SPLITS)).to(tl.int64)
        if RETURN_SOFTMAX_STATS:
            kv_end_exclusive = tl.load(KV_END_EXCLUSIVE + query_token_idx)
        query_tile = tl.load(
            tl.make_block_ptr(
                QUERY + query_token_idx * NUM_QUERY_HEADS * KV_DIM,
                (NUM_QUERY_HEADS, KV_DIM),
                (KV_DIM, 1),
                (head_group_idx * HEADS_PER_GROUP, 0),
                (HEADS_PER_GROUP, KV_DIM),
                (1, 0),
            )
        )
        query_rope_tile = tl.load(
            tl.make_block_ptr(
                QUERY_ROPE + query_token_idx * NUM_QUERY_HEADS * ROPE_DIM,
                (NUM_QUERY_HEADS, ROPE_DIM),
                (ROPE_DIM, 1),
                (head_group_idx * HEADS_PER_GROUP, 0),
                (HEADS_PER_GROUP, ROPE_DIM),
                (1, 0),
            )
        )
        softmax_max = tl.full(
            (HEADS_PER_GROUP,), TL_VLLM_FINITE_SOFTMAX_SENTINEL, tl.float32
        )
        softmax_sum = tl.zeros((HEADS_PER_GROUP,), tl.float32)
        weighted_value_sum = tl.zeros((HEADS_PER_GROUP, KV_DIM), tl.float32)
        KV_BLOCKS_PER_SPLIT: tl.constexpr = triton.cdiv(NUM_KV_BLOCKS, NUM_KV_SPLITS)
        for kv_block_idx in range(
            kv_split_idx * KV_BLOCKS_PER_SPLIT,
            tl.minimum((kv_split_idx + 1) * KV_BLOCKS_PER_SPLIT, NUM_KV_BLOCKS),
        ):
            kv_tile = tl.load(
                tl.make_block_ptr(
                    GATHERED_KV + query_token_idx * NUM_GATHERED_KV_ROWS * KV_DIM,
                    (NUM_GATHERED_KV_ROWS, KV_DIM),
                    (KV_DIM, 1),
                    (kv_block_idx * KV_BLOCK_ROWS, 0),
                    (KV_BLOCK_ROWS, KV_DIM),
                    (1, 0),
                )
            )
            key_rope_tile = tl.load(
                tl.make_block_ptr(
                    GATHERED_KEY_ROPE
                    + query_token_idx * NUM_GATHERED_KV_ROWS * ROPE_DIM,
                    (NUM_GATHERED_KV_ROWS, ROPE_DIM),
                    (ROPE_DIM, 1),
                    (kv_block_idx * KV_BLOCK_ROWS, 0),
                    (KV_BLOCK_ROWS, ROPE_DIM),
                    (1, 0),
                )
            )
            attention_scores = tl.dot(query_tile, tl.trans(kv_tile))
            attention_scores = (
                tl.dot(query_rope_tile, tl.trans(key_rope_tile), attention_scores)
                * SOFTMAX_SCALE
            )
            sparse_positions = kv_block_idx * KV_BLOCK_ROWS + tl.arange(
                0, KV_BLOCK_ROWS
            )
            additive_mask = tl.load(
                ADDITIVE_ATTENTION_MASK
                + query_token_idx * NUM_GATHERED_KV_ROWS
                + sparse_positions
            )
            attention_scores = attention_scores + additive_mask[None, :]
            updated_softmax_max = tl.maximum(softmax_max, tl.max(attention_scores, 1))
            softmax_exp_base = updated_softmax_max
            previous_state_scale = tl.exp(softmax_max - softmax_exp_base)
            unnormalized_weights = tl.exp(attention_scores - softmax_exp_base[:, None])
            softmax_sum = softmax_sum * previous_state_scale + tl.sum(
                unnormalized_weights, 1
            )
            weighted_value_sum = weighted_value_sum * previous_state_scale[
                :, None
            ] + tl.dot(unnormalized_weights.to(GATHERED_KV.dtype.element_ty), kv_tile)
            softmax_max = updated_softmax_max
        head_indices = head_group_idx * HEADS_PER_GROUP + tl.arange(0, HEADS_PER_GROUP)
        kv_columns = tl.arange(0, KV_DIM)
        if NUM_KV_SPLITS == 1:
            tl.store(
                OUT
                + query_token_idx * NUM_QUERY_HEADS * KV_DIM
                + head_indices[:, None] * KV_DIM
                + kv_columns[None, :],
                weighted_value_sum
                / tl.where(softmax_sum > 0, softmax_sum, 1.0)[:, None],
            )
            if RETURN_SOFTMAX_STATS:
                tl.store(
                    SOFTMAX_MAX + query_token_idx * NUM_QUERY_HEADS + head_indices,
                    tl.where(
                        softmax_sum > 0,
                        softmax_max,
                        tl.where(
                            kv_end_exclusive > 0, TL_VLLM_FINITE_SOFTMAX_SENTINEL, 0.0
                        ),
                    ),
                )
                tl.store(
                    SOFTMAX_SUM + query_token_idx * NUM_QUERY_HEADS + head_indices,
                    tl.where(
                        softmax_sum > 0,
                        softmax_sum,
                        tl.minimum(NUM_GATHERED_KV_ROWS, kv_end_exclusive).to(
                            tl.float32
                        ),
                    ),
                )
        else:
            tl.store(
                SPLIT_WEIGHTED_VALUE_SUM
                + (query_token_idx * NUM_KV_SPLITS + kv_split_idx)
                * NUM_QUERY_HEADS
                * KV_DIM
                + head_indices[:, None] * KV_DIM
                + kv_columns[None, :],
                weighted_value_sum,
            )
            tl.store(
                SPLIT_SOFTMAX_MAX
                + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * NUM_QUERY_HEADS
                + head_indices,
                softmax_max,
            )
            tl.store(
                SPLIT_SOFTMAX_SUM
                + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * NUM_QUERY_HEADS
                + head_indices,
                softmax_sum,
            )


@triton.jit
def _sfa_combine(
    SPLIT_WEIGHTED_VALUE_SUM,
    SPLIT_SOFTMAX_MAX,
    SPLIT_SOFTMAX_SUM,
    OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    QUERY_LENGTHS,
    KV_LENGTHS,
    NUM_KV_SPLITS: tl.constexpr,
    RETURN_SOFTMAX_STATS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    task_idx = tl.program_id(0)
    query_token_idx = task_idx // TL_COMBINE_PROGRAMS_PER_QUERY
    head_indices = (
        task_idx % TL_COMBINE_PROGRAMS_PER_QUERY * TL_COMBINE_HEADS_PER_PROGRAM
        + tl.arange(0, TL_COMBINE_HEADS_PER_PROGRAM)
    )
    kv_columns = tl.arange(0, TL_KV_DIM)
    softmax_max = tl.full((TL_COMBINE_HEADS_PER_PROGRAM,), float("-inf"), tl.float32)
    for kv_split_idx in range(NUM_KV_SPLITS):
        split_softmax_max = tl.load(
            SPLIT_SOFTMAX_MAX
            + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * TL_FAST_QUERY_HEADS
            + head_indices
        )
        softmax_max = tl.maximum(softmax_max, split_softmax_max)
    combined_weighted_value_sum = tl.zeros(
        (TL_COMBINE_HEADS_PER_PROGRAM, TL_KV_DIM), tl.float32
    )
    softmax_sum = tl.zeros((TL_COMBINE_HEADS_PER_PROGRAM,), tl.float32)
    for kv_split_idx in range(NUM_KV_SPLITS):
        split_softmax_max = tl.load(
            SPLIT_SOFTMAX_MAX
            + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * TL_FAST_QUERY_HEADS
            + head_indices
        )
        split_softmax_sum = tl.load(
            SPLIT_SOFTMAX_SUM
            + (query_token_idx * NUM_KV_SPLITS + kv_split_idx) * TL_FAST_QUERY_HEADS
            + head_indices
        )
        split_weighted_value_sum = tl.load(
            SPLIT_WEIGHTED_VALUE_SUM
            + (query_token_idx * NUM_KV_SPLITS + kv_split_idx)
            * TL_FAST_QUERY_HEADS
            * TL_KV_DIM
            + head_indices[:, None] * TL_KV_DIM
            + kv_columns[None, :]
        )
        split_rescale_factor = tl.exp(split_softmax_max - softmax_max)
        combined_weighted_value_sum += (
            split_weighted_value_sum * split_rescale_factor[:, None]
        )
        softmax_sum += split_softmax_sum * split_rescale_factor
    tl.store(
        OUT
        + query_token_idx * TL_FAST_QUERY_HEADS * TL_KV_DIM
        + head_indices[:, None] * TL_KV_DIM
        + kv_columns[None, :],
        combined_weighted_value_sum
        / tl.where(softmax_sum > 0, softmax_sum, 1.0)[:, None],
    )

    if RETURN_SOFTMAX_STATS:
        _, kv_end_exclusive, _ = _sfa_query_meta(
            query_token_idx,
            QUERY_LENGTHS,
            KV_LENGTHS,
            NUM_REQUESTS,
            QUERY_SEQ_CAPACITY,
            0,
            QUERY_IS_TND,
            False,
            HAS_QUERY_LENGTHS,
            True,
            CAUSAL,
        )
        tl.store(
            SOFTMAX_MAX + query_token_idx * TL_FAST_QUERY_HEADS + head_indices,
            tl.where(
                softmax_sum > 0,
                softmax_max,
                tl.where(kv_end_exclusive > 0, TL_VLLM_FINITE_SOFTMAX_SENTINEL, 0.0),
            ),
        )
        tl.store(
            SOFTMAX_SUM + query_token_idx * TL_FAST_QUERY_HEADS + head_indices,
            tl.where(
                softmax_sum > 0,
                softmax_sum,
                tl.minimum(TL_FAST_SPARSE_ENTRIES, kv_end_exclusive).to(tl.float32),
            ),
        )


@triton.jit
def _sfa_combine_two(
    SPLIT_WEIGHTED_VALUE_SUM,
    SPLIT_SOFTMAX_MAX,
    SPLIT_SOFTMAX_SUM,
    OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    QUERY_LENGTHS,
    KV_LENGTHS,
    NUM_KV_SPLITS: tl.constexpr,
    RETURN_SOFTMAX_STATS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    QUERY_SEQ_CAPACITY: tl.constexpr,
    QUERY_IS_TND: tl.constexpr,
    HAS_QUERY_LENGTHS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    task_idx = tl.program_id(0)
    query_token_idx = task_idx // TL_COMBINE_TWO_PROGRAMS_PER_QUERY
    head_indices = (
        task_idx % TL_COMBINE_TWO_PROGRAMS_PER_QUERY * TL_COMBINE_TWO_HEADS_PER_PROGRAM
        + tl.arange(0, TL_COMBINE_TWO_HEADS_PER_PROGRAM)
    )
    split_max_0 = tl.load(
        SPLIT_SOFTMAX_MAX
        + query_token_idx * TL_COMBINE_TWO_SPLITS * TL_FAST_QUERY_HEADS
        + head_indices
    )
    split_max_1 = tl.load(
        SPLIT_SOFTMAX_MAX
        + (query_token_idx * TL_COMBINE_TWO_SPLITS + 1) * TL_FAST_QUERY_HEADS
        + head_indices
    )
    softmax_max = tl.maximum(split_max_0, split_max_1)
    split_weight_0 = tl.exp(split_max_0 - softmax_max)
    split_weight_1 = tl.exp(split_max_1 - softmax_max)
    split_sum_0 = tl.load(
        SPLIT_SOFTMAX_SUM
        + query_token_idx * TL_COMBINE_TWO_SPLITS * TL_FAST_QUERY_HEADS
        + head_indices
    )
    split_sum_1 = tl.load(
        SPLIT_SOFTMAX_SUM
        + (query_token_idx * TL_COMBINE_TWO_SPLITS + 1) * TL_FAST_QUERY_HEADS
        + head_indices
    )
    softmax_sum = split_sum_0 * split_weight_0 + split_sum_1 * split_weight_1
    inverse_softmax_sum = 1.0 / tl.where(softmax_sum > 0, softmax_sum, 1.0)
    split_weight_0 = split_weight_0 * inverse_softmax_sum
    split_weight_1 = split_weight_1 * inverse_softmax_sum
    for value_block_idx in range(TL_COMBINE_TWO_VALUE_TILES):
        kv_columns = value_block_idx * TL_COMBINE_TWO_VALUE_COLUMNS + tl.arange(
            0, TL_COMBINE_TWO_VALUE_COLUMNS
        )
        weighted_value_sum_0 = tl.load(
            SPLIT_WEIGHTED_VALUE_SUM
            + query_token_idx * TL_COMBINE_TWO_SPLITS * TL_FAST_QUERY_HEADS * TL_KV_DIM
            + head_indices[:, None] * TL_KV_DIM
            + kv_columns[None, :]
        )
        weighted_value_sum_1 = tl.load(
            SPLIT_WEIGHTED_VALUE_SUM
            + (query_token_idx * TL_COMBINE_TWO_SPLITS + 1)
            * TL_FAST_QUERY_HEADS
            * TL_KV_DIM
            + head_indices[:, None] * TL_KV_DIM
            + kv_columns[None, :]
        )
        output_tile = (
            weighted_value_sum_0 * split_weight_0[:, None]
            + weighted_value_sum_1 * split_weight_1[:, None]
        )
        tl.store(
            OUT
            + query_token_idx * TL_FAST_QUERY_HEADS * TL_KV_DIM
            + head_indices[:, None] * TL_KV_DIM
            + kv_columns[None, :],
            output_tile,
        )
    if RETURN_SOFTMAX_STATS:
        _, kv_end_exclusive, _ = _sfa_query_meta(
            query_token_idx,
            QUERY_LENGTHS,
            KV_LENGTHS,
            NUM_REQUESTS,
            QUERY_SEQ_CAPACITY,
            0,
            QUERY_IS_TND,
            False,
            HAS_QUERY_LENGTHS,
            True,
            CAUSAL,
        )
        tl.store(
            SOFTMAX_MAX + query_token_idx * TL_FAST_QUERY_HEADS + head_indices,
            tl.where(
                softmax_sum > 0,
                softmax_max,
                tl.where(kv_end_exclusive > 0, TL_VLLM_FINITE_SOFTMAX_SENTINEL, 0.0),
            ),
        )
        tl.store(
            SOFTMAX_SUM + query_token_idx * TL_FAST_QUERY_HEADS + head_indices,
            tl.where(
                softmax_sum > 0,
                softmax_sum,
                tl.minimum(TL_FAST_SPARSE_ENTRIES, kv_end_exclusive).to(tl.float32),
            ),
        )

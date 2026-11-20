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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

# Minimum kv chunks for the launch decomposition. Only the chunk->page mapping
# changes; the set of (head, query, block) entries written and their values are
# independent of this value.
_MIN_KV_CHUNKS = 8


@triton.jit
def _decode_qk_score_kernel(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    block_table_ptr,
    seq_lens_ptr,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    decode_query_len: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    # the prologue is worth more than any restructuring of the body (all seven
    # workloads improved, including the four that perform no key load at all).
    stride_q_n: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_ik_blk: tl.constexpr,
    stride_ik_pos: tl.constexpr,
    stride_ik_d: tl.constexpr,
    stride_s_h: tl.constexpr,
    stride_s_n: tl.constexpr,
    stride_s_k: tl.constexpr,
    stride_bt_b: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    # Chunk index in the fastest grid dimension: every chunk of one query is
    # contiguous in linear program order.
    chunk_id = tl.program_id(0)
    query_id = tl.program_id(1)

    request_id = query_id // decode_query_len
    query_offset = query_id - request_id * decode_query_len

    # Page id of this chunk's first block, issued before the visible-range scalars
    # resolve. block_table_row[chunk_id] is in bounds by construction (NUM_CHUNKS
    # is clamped to the block_table row stride) and is only consumed when
    # chunk_size_blocks == 1, i.e. when block chunk_id is visible in this chunk.
    block_table_row = block_table_ptr + request_id * stride_bt_b
    first_page_id = tl.load(block_table_row + chunk_id).to(tl.int64)

    seq_len = tl.load(seq_lens_ptr + request_id)
    kv_length = tl.maximum(
        seq_len - decode_query_len + query_offset + 1,
        0,
    )
    valid_blocks = (kv_length + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    local_start = tl.maximum(valid_blocks - local_blocks, 0)

    # Split only the visible range. Empty chunks return before loading Q.
    chunk_size_blocks = (valid_blocks + NUM_CHUNKS - 1) // NUM_CHUNKS
    chunk_start = chunk_id * chunk_size_blocks
    chunk_end = tl.minimum(chunk_start + chunk_size_blocks, valid_blocks)
    if chunk_start >= chunk_end:
        return

    head_offsets = tl.arange(0, num_idx_heads)
    score_ptrs = score_ptr + head_offsets * stride_s_h + query_id * stride_s_n

    # Every block at or beyond local_start stores exactly 1e29 (local override last),
    # so it needs no page, no dot and no max.
    computed_end = tl.minimum(chunk_end, local_start)
    forced_start = tl.maximum(chunk_start, local_start)

    if chunk_start < computed_end:
        dim_offsets = tl.arange(0, head_dim)
        key_offsets = tl.arange(0, BLOCK_SIZE_K)

        query = tl.load(
            q_ptr
            + query_id * stride_q_n
            + head_offsets[:, None] * stride_q_h
            + dim_offsets[None, :] * stride_q_d,
        )

        if chunk_size_blocks == 1:
            page_id = first_page_id
        else:
            page_id = tl.load(block_table_row + chunk_start).to(tl.int64)

        for block_id in tl.range(chunk_start, computed_end):
            if block_id != chunk_start:
                page_id = tl.load(block_table_row + block_id).to(tl.int64)
            key = tl.load(
                ik_cache_ptr
                + page_id * stride_ik_blk
                + key_offsets[None, :] * stride_ik_pos
                + dim_offsets[:, None] * stride_ik_d,
            )
            query_key = tl.dot(query, key, out_dtype=tl.float32)

            # A fully visible block has every position below kv_length, so the
            # select is a local identity there and is only emitted for the tail.
            if (block_id + 1) * BLOCK_SIZE_K > kv_length:
                key_positions = block_id * BLOCK_SIZE_K + key_offsets
                query_key = tl.where(
                    key_positions[None, :] < kv_length,
                    query_key,
                    float("-inf"),
                )
            block_score = tl.max(query_key, axis=1)

            # Only reachable when init_blocks > 0; constant-folded otherwise.
            if init_blocks > 0:
                block_score = tl.where(block_id < init_blocks, 1e30, block_score)

            tl.store(score_ptrs + block_id * stride_s_k, block_score)

    if forced_start < chunk_end:
        local_value = tl.full((num_idx_heads,), 1e29, tl.float32)
        for block_id in tl.range(forced_start, chunk_end):
            tl.store(score_ptrs + block_id * stride_s_k, local_value)


@torch.inference_mode()
def _launch(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    block_table_ptr,
    seq_lens_ptr,
    num_idx_heads,
    head_dim,
    decode_query_len,
    init_blocks,
    local_blocks,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_K=128,
    NUM_CHUNKS=4,
    *,
    grid=None,
    warmup=False,
):
    total_q = q_ptr.shape[0]
    # Decomposition-only choice (pure function of the call arguments): more kv
    # chunks shorten the per-program serial page depth, and chunk_id is the
    # fastest grid dimension. chunk_size = cdiv(valid_blocks, num_chunks) and empty
    # chunks return before loading Q, so every num_chunks writes exactly the same
    # score entries.
    num_chunks = NUM_CHUNKS if NUM_CHUNKS >= _MIN_KV_CHUNKS else _MIN_KV_CHUNKS
    # The prefetched block_table slot is chunk_id, so keep chunk_id inside the row.
    block_stride = int(stride_bt_b)
    if block_stride > 0 and num_chunks > block_stride:
        num_chunks = block_stride
    launch_grid = (num_chunks, total_q)
    launch_args = (
        q_ptr,
        ik_cache_ptr,
        score_ptr,
        block_table_ptr,
        seq_lens_ptr,
        num_idx_heads,
        head_dim,
        int(decode_query_len),
        init_blocks,
        local_blocks,
        int(stride_q_n),
        int(stride_q_h),
        int(stride_q_d),
        int(stride_ik_blk),
        int(stride_ik_pos),
        int(stride_ik_d),
        int(stride_s_h),
        int(stride_s_n),
        int(stride_s_k),
        int(stride_bt_b),
    )
    # A program owns one query row and one or two key pages, so the default warp
    # count cannot add arithmetic parallelism; num_warps=1 is the minimal launch
    # configuration and tests whether the fixed per-task cost is proportional to the
    # resources reserved per program.
    meta = {"BLOCK_SIZE_K": BLOCK_SIZE_K, "NUM_CHUNKS": num_chunks}
    launch_options = {"num_warps": 1}
    # kernel[grid](...) already forwards warmup=False itself, so the compile-only
    # path has to go through JITFunction.run directly.
    if warmup:
        _decode_qk_score_kernel.run(
            *launch_args, grid=launch_grid, warmup=True, **meta, **launch_options
        )
        return None
    _decode_qk_score_kernel[launch_grid](*launch_args, **meta, **launch_options)
    return None


def minimax_m3_decode_qk_score(
    q,
    index_cache,
    block_table,
    seq_lens,
    decode_query_len=1,
    init_blocks=0,
    local_blocks=1,
):
    """Return FP32 [heads, tokens, page-table width] scores before top-k.

    Unused scores are -inf. Lengths count tokens and stay within each logical
    page table; its visible entries contain valid physical cache page IDs.
    """
    if (
        q.ndim != 3
        or q.shape[1:] != (1, 128)
        or index_cache.ndim != 3
        or index_cache.shape[1:] != (128, 128)
    ):
        raise ValueError("expected one 128-wide query head and a 128-token KV page")
    if (
        q.dtype != torch.bfloat16
        or index_cache.dtype != q.dtype
        or block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
    ):
        raise ValueError("expected BF16 query/cache and int32 metadata")
    if (
        block_table.ndim != 2
        or seq_lens.shape != (block_table.shape[0],)
        or decode_query_len < 1
        or q.shape[0] != block_table.shape[0] * decode_query_len
    ):
        raise ValueError("batch sizes and decode_query_len must match")
    if q.device.type != "npu" or any(
        t.device != q.device or not t.is_contiguous()
        for t in (q, index_cache, block_table, seq_lens)
    ):
        raise ValueError("expected contiguous tensors on one NPU")
    if (
        init_blocks < 0
        or local_blocks < 0
        or block_table.shape[1] < 1
        or index_cache.shape[0] < 1
    ):
        raise ValueError("expected a nonempty cache/table and nonnegative block counts")
    tokens, heads, dim = q.shape
    scores = torch.full(
        (heads, tokens, block_table.shape[1]),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    if tokens:
        with torch_device_fn.device(q.device):
            _launch(
                q,
                index_cache,
                scores,
                block_table,
                seq_lens,
                heads,
                dim,
                decode_query_len,
                init_blocks,
                local_blocks,
                *q.stride(),
                *index_cache.stride(),
                *scores.stride(),
                block_table.stride(0),
            )
    return scores

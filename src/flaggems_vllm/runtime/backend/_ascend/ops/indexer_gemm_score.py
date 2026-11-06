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

"""GEMM-form scorer for the GLM lightning indexer.

The per-token matvec scorer re-reads the whole pool cache once per token row
(~96x logical redundancy on big prefill shapes).  This GEMM form gathers each
pool tile once per request and multiplies it with an [BM, 128] query block via
tl.dot: 14063 -> 777 us at the production 8192-token/6220-pool shape (18.1x),
5343.6 -> 1130.2 ms per rank end-to-end (-67.4s/16rank).  Rows below
``_GEMM_SCORE_MIN_ROWS`` (decode/draft steps) fall back to the matvec kernel,
mirroring the production launcher contract.

Input layout (semantics only, no code dependency):
  qbar [T, 4, 128] fp32        head-weighted query rows
  cache [max_pool, 96, 1, 128] bf16 pool cache (96 tokens per pool page)
  bt [n_req, n_pages] int32    logical-page -> physical-block table
  cum_q / seq_lens / positions varlen batch description
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

TRITON_POOL_CHUNK_SIZE = 2048
TRITON_POOL_SUBTILE_SIZE = 128
_GEMM_SCORE_BN = 64
_GEMM_SCORE_BM = 64
_GEMM_SCORE_MIN_ROWS = 64


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit(
    do_not_specialize=[
        "token_offset",
        "max_pool_seq_len",
        "num_reqs",
        "num_cache_blocks",
    ]
)
def _glm5_next_lightning_indexer_score_kernel(
    qbar_ptr,
    indexer_cache_ptr,
    cum_query_lens_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    scores_ptr,
    token_offset,
    max_pool_seq_len,
    num_reqs,
    num_cache_blocks,
    cache_stride_block: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_d: tl.constexpr,
    block_table_stride_req: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    pool_block_size: tl.constexpr,
    REQ_POW2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    SUB_POOL: tl.constexpr,
):
    # qbar/scores rows are chunk-local; positions/query-boundary lookups use
    # the batch-global token index.
    local_token_idx = tl.program_id(0)
    token_idx = local_token_idx + token_offset
    chunk = tl.program_id(1)

    req_offsets = tl.arange(0, REQ_POW2)
    query_ends = tl.load(
        cum_query_lens_ptr + req_offsets, mask=req_offsets < num_reqs, other=2147483647
    )
    req_id = tl.sum(tl.where(token_idx >= query_ends, 1, 0))
    # Full ACL graphs keep padded rows beyond the last request; keep their
    # pointer arithmetic in bounds even though their outputs are unused.
    req_id = tl.minimum(req_id, num_reqs - 1)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    request_pool_len = tl.load(indexer_seq_lens_ptr + req_id).to(tl.int32)
    causal_pool_len = (pos + 1) // INDEX_KPOOL
    visible_pool_len = tl.minimum(causal_pool_len, request_pool_len)

    dim_offsets = tl.arange(0, HEAD_DIM)
    qbar = tl.load(qbar_ptr + local_token_idx * HEAD_DIM + dim_offsets)

    chunk_start = chunk * BLOCK_POOL
    # Dynamic trip count: requests shorter than the static max pool length
    # skip their out-of-range sub-tiles even inside captured graphs. Cells
    # beyond ``visible_pool_len`` keep the -inf the wrapper initialized.
    chunk_visible = tl.maximum(
        tl.minimum(visible_pool_len, chunk_start + BLOCK_POOL) - chunk_start, 0
    )
    num_subs = tl.cdiv(chunk_visible, SUB_POOL)
    for sub in tl.range(num_subs):
        pool_offsets = chunk_start + sub * SUB_POOL + tl.arange(0, SUB_POOL)
        in_range = pool_offsets < max_pool_seq_len
        valid_pool = in_range & (pool_offsets < visible_pool_len)
        logical_pages = pool_offsets // pool_block_size
        page_offsets = pool_offsets % pool_block_size
        physical_blocks = tl.load(
            indexer_block_table_ptr
            + req_id * block_table_stride_req
            + logical_pages * block_table_stride_page,
            mask=in_range,
            other=0,
        ).to(tl.int64)
        # Clamp both sides: padded/stale block-table entries must never form
        # an out-of-range cache address, even though their loads are masked.
        physical_blocks = tl.minimum(
            tl.maximum(physical_blocks, 0), num_cache_blocks - 1
        )
        k_addrs = (
            physical_blocks[:, None] * cache_stride_block
            + page_offsets[:, None] * cache_stride_offset
            + dim_offsets[None, :] * cache_stride_d
        )
        k_tile = tl.load(
            indexer_cache_ptr + k_addrs, mask=valid_pool[:, None], other=0.0
        ).to(tl.float32)
        scores = tl.sum(k_tile * qbar[None, :], axis=1)
        scores = tl.where(valid_pool, scores, float("-inf"))
        tl.store(
            scores_ptr + local_token_idx * max_pool_seq_len + pool_offsets,
            scores,
            mask=in_range,
        )


@triton.jit(do_not_specialize=["num_reqs", "num_tokens", "max_pool", "num_blocks"])
def _glm5_indexer_gemm_score_kernel(
    qbar_ptr,
    indexer_cache_ptr,
    cum_query_lens_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    scores_ptr,
    token_start,
    max_pool_seq_len,
    num_reqs,
    num_cache_blocks,
    num_rows,
    cache_stride_block: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_d: tl.constexpr,
    block_table_stride_req: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    pool_block_size: tl.constexpr,
    REQ_POW2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
):
    req = tl.program_id(0)
    if req >= num_reqs:
        return
    sub_tile = tl.program_id(1)

    # chunk-local row range of this request; cum_query_lens is the per-request
    # END offset array (same layout as indexer_seq_lens), not a cu array.
    r_hi = tl.load(cum_query_lens_ptr + req)
    if req == 0:
        r_lo = r_hi * 0  # keep int64 type consistent across branches
    else:
        r_lo = tl.load(cum_query_lens_ptr + req - 1)
    lo = tl.maximum(r_lo - token_start, 0)
    hi = tl.minimum(r_hi - token_start, num_rows)

    # gather this pool tile once for the whole request
    pool_offsets = sub_tile * BN + tl.arange(0, BN)
    in_range = pool_offsets < max_pool_seq_len
    logical_pages = pool_offsets // pool_block_size
    page_offsets = pool_offsets % pool_block_size
    physical_blocks = tl.load(
        indexer_block_table_ptr
        + req * block_table_stride_req
        + logical_pages * block_table_stride_page,
        mask=in_range,
        other=0,
    ).to(tl.int64)
    physical_blocks = tl.minimum(tl.maximum(physical_blocks, 0), num_cache_blocks - 1)
    dim_offsets = tl.arange(0, HEAD_DIM)
    k_addrs = (
        physical_blocks[:, None] * cache_stride_block
        + page_offsets[:, None] * cache_stride_offset
        + dim_offsets[None, :] * cache_stride_d
    )
    b_tile = tl.load(indexer_cache_ptr + k_addrs, mask=in_range[:, None], other=0.0).to(
        tl.float32
    )  # [BN, D]
    b_tile_t = tl.trans(b_tile)  # [D, BN]
    b_tile_t = tl.where(in_range[None, :], b_tile_t, 0.0)

    req_pool_len = tl.load(indexer_seq_lens_ptr + req).to(tl.int32)

    for m0 in tl.range(lo, hi, BM):
        offs_m = m0 + tl.arange(0, BM)
        m_m = offs_m < hi
        a = tl.load(
            qbar_ptr + offs_m[:, None] * HEAD_DIM + dim_offsets[None, :],
            mask=m_m[:, None],
            other=0.0,
        )  # [BM, D]
        c = tl.dot(a, b_tile_t, input_precision="ieee")  # [BM, BN] fp32
        pos = tl.load(positions_ptr + token_start + offs_m, mask=m_m, other=0).to(
            tl.int32
        )
        vis = tl.minimum((pos + 1) // INDEX_KPOOL, req_pool_len)
        c = tl.where(pool_offsets[None, :] < vis[:, None], c, float("-inf"))
        tl.store(
            scores_ptr + offs_m[:, None] * max_pool_seq_len + pool_offsets[None, :],
            c,
            mask=m_m[:, None] & in_range[None, :],
        )


def indexer_gemm_score(
    qbar: torch.Tensor,
    cache: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    token_start: int = 0,
) -> torch.Tensor:
    """Score every query row against its visible pool window.

    ``scores[i, p] = <qbar[i], cache[pool p]>`` for pools visible to row i
    (causal + per-request length bound), ``-inf`` elsewhere.  Prefill-sized
    row counts take the GEMM form; small batches fall back to the per-token
    matvec kernel exactly like the production launcher.
    """
    logger.debug("GEMS_ASCEND INDEXER_GEMM_SCORE")
    num_tokens = qbar.shape[0]
    max_pool = int(seq_lens.max().item()) if num_tokens else 0
    num_reqs = cum_query_lens.shape[0]
    out = torch.full(
        (num_tokens, max_pool), float("-inf"), dtype=torch.float32, device=qbar.device
    )
    if num_tokens == 0 or max_pool == 0:
        return out
    req_pow2 = _next_power_of_2(max(1, num_reqs))
    head_dim = qbar.shape[-1]
    index_kpool = 4
    if num_tokens >= _GEMM_SCORE_MIN_ROWS:
        grid = (num_reqs, (max_pool + _GEMM_SCORE_BN - 1) // _GEMM_SCORE_BN)
        _glm5_indexer_gemm_score_kernel[grid](
            qbar,
            cache,
            cum_query_lens,
            seq_lens,
            block_table,
            positions,
            out,
            token_start,
            max_pool,
            num_reqs,
            cache.shape[0],
            num_tokens,
            cache.stride(0),
            cache.stride(1),
            cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            cache.shape[1],
            req_pow2,
            head_dim,
            index_kpool,
            _GEMM_SCORE_BN,
            _GEMM_SCORE_BM,
            num_warps=4,
            num_stages=1,
        )
    else:
        num_chunks = (max_pool + TRITON_POOL_CHUNK_SIZE - 1) // TRITON_POOL_CHUNK_SIZE
        _glm5_next_lightning_indexer_score_kernel[(num_tokens, num_chunks)](
            qbar,
            cache,
            cum_query_lens,
            seq_lens,
            block_table,
            positions,
            out,
            token_start,
            max_pool,
            num_reqs,
            cache.shape[0],
            cache.stride(0),
            cache.stride(1),
            cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            cache.shape[1],
            req_pow2,
            head_dim,
            index_kpool,
            TRITON_POOL_CHUNK_SIZE,
            TRITON_POOL_SUBTILE_SIZE,
        )
    return out

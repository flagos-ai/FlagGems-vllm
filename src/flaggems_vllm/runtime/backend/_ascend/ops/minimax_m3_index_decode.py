# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-M3 decode index top-k, ported from the vendor runtime's kernels.

Why this exists
---------------
The upstream kernels in ``vllm/models/minimax_m3/common/ops/index_topk.py`` and
the vendor's own split differ in ways that only show up once a DP rank serves
more than one request. Measured on A3 (DP2xTP8+EP, FULL_DECODE_ONLY), the
upstream ``_decode_index_score_kernel`` cost 8.9 us per call when each rank ran
one decode request but 129 us when it ran two, with identical grid and
constexpr arguments -- a 27x jump in ``aic_total_cycles`` (154k -> 4245k).
vLLM-Ascend's fork stayed at 10.5 us in both cases.

The vendor version differs structurally rather than by tuning:

* its split-K factor comes from ``@triton.autotune`` over
  ``num_kv_chunks`` (1..256) with a request-count-aware prune, so it adapts to
  the batch instead of relying on a fixed formula;
* the init/local forcing is precomputed once per step into boolean masks by
  ``_prepare_decode_score_masks_kernel`` (scalar loads, no per-block recompute)
  and consumed by simple ``tl.load`` of those masks inside the block loop;
* the unwritten score tail is filled by a separate small kernel, so the score
  buffer can stay ``torch.empty`` rather than being fully initialised.

The kernels are taken verbatim; only the module imports and the surrounding
plumbing are FL's. Accuracy is checked against the previous implementation by
comparing the selected block ids, because downstream attention sums over the
selected blocks.
"""

from __future__ import annotations

import torch
import flag_gems
import triton
import triton.language as tl


def round_up(value, multiple):
    return triton.cdiv(value, multiple) * multiple


# Data-layout constants.
SPARSE_BLOCK_SIZE = 128
SCORE_BLOCK_STRIDE_ALIGNMENT = 16


def _is_arch_support_pdl():
    # This operator family is intentionally Ascend-only.
    return False


def _as_triton_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Normalizes the index-key cache to [num_blocks, 128, head_dim]."""
    if isinstance(index_kv_cache, (tuple, list)):
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 5 and index_kv_cache.shape[0] == 2:
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 4:
        if index_kv_cache.shape[2] != 1:
            raise ValueError(
                f"Unexpected index cache head dim: {tuple(index_kv_cache.shape)}"
            )
        index_kv_cache = index_kv_cache.squeeze(2)
    if index_kv_cache.ndim != 3:
        raise ValueError(f"Unexpected index cache ndim: {index_kv_cache.ndim}")
    return index_kv_cache


def _prune_decode_score_configs(configs, named_args, **_):
    """Keeps decode split-K launches within the configured program budget."""
    request_count = max(1, named_args["num_reqs"])
    chunk_limit = max(1, 512 // request_count)
    chunk_limit = 1 << (chunk_limit.bit_length() - 1)
    valid_configs = [
        config for config in configs if config.kwargs["num_kv_chunks"] <= chunk_limit
    ]
    return valid_configs or configs[:1]


@triton.autotune(
    configs=[
        triton.Config(
            {"num_kv_chunks": chunk_count},
            num_stages=stage_count,
        )
        for chunk_count in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        for stage_count in (1, 2)
    ],
    key=["num_idx_heads", "BLOCK_SIZE_Q", "head_dim", "num_reqs"],
    prune_configs_by={"early_config_prune": _prune_decode_score_configs},
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    init_mask_ptr,  # [total_q, score_block_stride] bool
    local_mask_ptr,  # [total_q, score_block_stride] bool
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_reqs: tl.constexpr,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_mask_q,
    stride_mask_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_Q: tl.constexpr,
    num_kv_chunks,
    USE_PDL: tl.constexpr,
):
    BLOCK_SIZE_HQ: tl.constexpr = num_idx_heads * BLOCK_SIZE_Q
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    hq_offsets = tl.arange(0, BLOCK_SIZE_HQ)
    h_offsets = hq_offsets // BLOCK_SIZE_Q
    q_offsets = hq_offsets % BLOCK_SIZE_Q
    q_mask = q_offsets < decode_query_len
    q_ids = pid_r * decode_query_len + q_offsets

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + pid_r)
    query_pos = seq_len - decode_query_len + q_offsets
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    kv_len_max = tl.max(tl.where(q_mask, kv_len, 0), axis=0)
    num_blocks = (kv_len_max + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # block-aligned fixed-count split: grid independent of seq_len (captured graph).
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)  # positions within a 128-block
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_r * stride_bt_b
    # Query vectors for all index heads in a small spec-decode block.
    q = tl.load(
        q_ptr
        + q_ids[:, None] * stride_q_n
        + h_offsets[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=q_mask[:, None],
        other=0.0,
    )  # [HQ,D]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos[None, :] < kv_len[:, None]
        # index-K for this page: [D,N] (transposed), same layout as prefill.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )  # [D,N]
        # fp32 accumulation is required for the fp8 (e4m3) index cache: q/k are
        # loaded in their stored dtype (bf16 or e4m3) and the MMA accumulates in
        # fp32 so the per-block max score is exact for the fp8 indexer too.
        qk = tl.dot(q, k, out_dtype=tl.float32)  # [HQ,N]
        qk = tl.where(pos_mask & q_mask[:, None], qk, float("-inf"))
        score = tl.max(qk, axis=1)  # [HQ]
        mask_off = q_ids * stride_mask_q + blk * stride_mask_k
        is_init = tl.load(init_mask_ptr + mask_off) != 0
        is_local = tl.load(local_mask_ptr + mask_off) != 0
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr + h_offsets * stride_s_h + q_ids * stride_s_n + blk * stride_s_k,
            score,
            mask=q_mask,
        )


# ---------------------------------------------------------------------------
# Pad unwritten score tail with -inf so torch.topk ignores [num_blocks, max_block).
# _decode_index_score_kernel only writes [0, row_num_blocks); torch.empty leaves
# the rest as garbage. Per-token num_blocks matches the top-k invalid mask.
# Split-K over max_block with a shape-constant chunk count (captured graph-safe).


@triton.jit(do_not_specialize=["decode_query_len", "max_block", "chunk_blocks"])
def _fill_decode_score_tail_kernel(
    score_ptr,  # [num_idx_heads, total_q, score_block_stride] fp32
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    max_block,
    decode_query_len,
    chunk_blocks,  # max_block split count per chunk (shape-constant)
    stride_s_h,
    stride_s_b,
    stride_s_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, max_block)
    fill_start = tl.maximum(chunk_start, num_blocks)
    if fill_start >= chunk_end:
        return

    num_to_fill = chunk_end - fill_start
    off_k = tl.arange(0, BLOCK_SIZE_K)
    for i in tl.range(0, num_to_fill, BLOCK_SIZE_K):
        blk = fill_start + i + off_k
        store_mask = (i + off_k) < num_to_fill
        s_ptrs = score_ptr + pid_h * stride_s_h + pid_b * stride_s_b + blk * stride_s_k
        tl.store(s_ptrs, float("-inf"), mask=store_mask)


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize=["decode_query_len"])
def _mask_decode_topk_indices_kernel(
    ti_ptr,  # [num_idx_heads, total_q, topk] int32 in/out
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    decode_query_len,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    off_t = tl.arange(0, BLOCK_SIZE_T)
    ti_ptrs = ti_ptr + pid_h * stride_ti_h + pid_b * stride_ti_b + off_t * stride_ti_t
    store_mask = off_t < topk
    idx = tl.load(ti_ptrs, mask=store_mask, other=0)
    valid_slot = off_t < tl.minimum(topk, num_blocks)
    valid_idx = (idx >= 0) & (idx < num_blocks)
    masked_idx = tl.where(valid_slot & valid_idx, idx, -1)
    tl.store(ti_ptrs, masked_idx.to(ti_ptr.dtype.element_ty), mask=store_mask)


# ---------------------------------------------------------------------------
# Prefill score finalization before torch.topk.
#
# A program handles one query tile and one index head. BLOCK_SIZE_Q is selected
# on the host from {8, 16, 32, 64} so the number of programs stays near a
# configurable target. Since BLOCK_SIZE_Q <= 64 and one sparse block contains
# 128 tokens, a tile can contain at most two distinct valid-block counts.
#
# Init blocks use one regular [Q, K] store. Local blocks are split into the two
# possible valid-block groups, so every store still has a scalar block base and
# regular vector addresses. The invalid tail is also written with a common
# column vector and a row mask; no per-query dynamic scatter is used.
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["decode_query_len", "max_block", "chunk_blocks"])
def _prepare_decode_score_masks_kernel(
    init_mask_ptr,  # [total_q, score_block_stride] bool out
    local_mask_ptr,  # [total_q, score_block_stride] bool out
    seq_lens,  # [num_reqs] int32
    block_size: tl.constexpr,  # sparse block size (128)
    max_block,
    decode_query_len,
    chunk_blocks,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_mask_q,
    stride_mask_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    req_id = pid_q // decode_query_len
    q_offset = pid_q - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id).to(tl.float32)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1.0, 0.0)
    valid_blocks = tl.floor((query_pos + block_size * 1.0) / (block_size * 1.0))
    local_start = tl.maximum(
        tl.floor((kv_len + (block_size - 1) * 1.0) / (block_size * 1.0))
        - local_blocks * 1.0,
        0.0,
    )

    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, max_block)
    if chunk_start >= chunk_end:
        return

    num_blks = chunk_end - chunk_start
    off_k = tl.arange(0, BLOCK_SIZE_K)
    for i in tl.range(0, num_blks, BLOCK_SIZE_K):
        blk = chunk_start + i + off_k
        store_mask = (i + off_k) < num_blks
        blk_f = blk * 1.0
        blk_valid = blk_f < valid_blocks
        is_init = (blk_f < init_blocks * 1.0) & blk_valid
        is_local = (blk_f >= local_start) & blk_valid
        mask_ptrs = init_mask_ptr + pid_q * stride_mask_q + blk * stride_mask_k
        tl.store(mask_ptrs, is_init, mask=store_mask)
        tl.store(
            local_mask_ptr + pid_q * stride_mask_q + blk * stride_mask_k,
            is_local,
            mask=store_mask,
        )


def _copy_topk_indices(
    raw_indices: torch.Tensor,
    requested_topk: int,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Copies top-k indices into an int32 result and pads missing slots."""
    head_count, total_query_tokens, selected_count = raw_indices.shape
    if output is None:
        result = torch.empty(
            (head_count, total_query_tokens, requested_topk),
            dtype=torch.int32,
            device=raw_indices.device,
        )
    else:
        result = output[:, :total_query_tokens, :requested_topk]

    if selected_count < requested_topk:
        result.copy_(
            torch.full(result.shape, -1, dtype=result.dtype, device=result.device)
        )
    result[..., :selected_count].copy_(raw_indices)
    return result


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int | None = None,
    out: torch.Tensor | None = None,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Computes decode block scores and returns zero-based top-k block IDs.

    ``sm_scale`` is accepted for API compatibility and intentionally omitted
    because this score-only path consumes block ordering.
    """
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    assert topk > 0
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    assert index_head_count == num_kv_heads, "M3 requires num_idx_heads == num_kv_heads"

    if max_decode_query_len is None:
        max_decode_query_len = decode_query_len
    assert decode_query_len <= max_decode_query_len

    request_count = seq_lens.shape[0]
    assert total_query_tokens == request_count * decode_query_len

    max_block_count = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = round_up(
        max_block_count,
        SCORE_BLOCK_STRIDE_ALIGNMENT,
    )
    score = torch.empty(
        (index_head_count, total_query_tokens, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )

    init_mask = torch.zeros(
        (total_query_tokens, score_block_stride),
        dtype=torch.bool,
        device=seq_lens.device,
    )
    local_mask = torch.zeros_like(init_mask)
    mask_chunk_count = max(
        1,
        min(16, 64 // max(1, total_query_tokens)),
    )
    mask_chunk_blocks = triton.cdiv(max_block_count, mask_chunk_count)
    _prepare_decode_score_masks_kernel[(total_query_tokens, mask_chunk_count)](
        init_mask,
        local_mask,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        max_block_count,
        decode_query_len,
        mask_chunk_blocks,
        init_blocks,
        local_blocks,
        init_mask.stride(0),
        init_mask.stride(1),
        BLOCK_SIZE_K=2048,
    )

    use_pdl = _is_arch_support_pdl()
    launch_kwargs = {"launch_pdl": True} if use_pdl else {}
    decode_query_tile_size = triton.next_power_of_2(max_decode_query_len)
    decode_score_grid = lambda metadata: (
        request_count,
        metadata["num_kv_chunks"],
    )
    _decode_index_score_kernel[decode_score_grid](
        idx_q,
        index_kv_cache,
        score,
        init_mask,
        local_mask,
        block_table,
        seq_lens,
        index_head_count,
        head_dim,
        request_count,
        decode_query_len,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        init_mask.stride(0),
        init_mask.stride(1),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=decode_query_tile_size,
        USE_PDL=use_pdl,
        **launch_kwargs,
    )

    tail_chunk_count = max(
        1,
        min(
            16,
            64 // max(1, total_query_tokens * index_head_count),
        ),
    )
    tail_chunk_blocks = triton.cdiv(max_block_count, tail_chunk_count)
    _fill_decode_score_tail_kernel[
        (total_query_tokens, index_head_count, tail_chunk_count)
    ](
        score,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        max_block_count,
        decode_query_len,
        tail_chunk_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        BLOCK_SIZE_K=2048,
    )

    selected_count = min(topk, max_block_count)
    score_rows = score[:, :total_query_tokens, :max_block_count]
    raw_indices = flag_gems.topk(
        score_rows,
        k=selected_count,
        dim=-1,
    )[1]
    topk_indices = _copy_topk_indices(raw_indices, topk, out)

    _mask_decode_topk_indices_kernel[(total_query_tokens, index_head_count)](
        topk_indices,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
    )
    return topk_indices

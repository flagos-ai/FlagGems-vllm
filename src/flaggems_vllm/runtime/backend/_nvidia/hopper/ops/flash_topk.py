# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

"""SM90 paged FP8 scoring for the vLLM sparse attention indexer."""

import logging

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flaggems_vllm import runtime
from flaggems_vllm.ops.cooperative_topk import _cooperative_topk_impl, _memory_interval
from flaggems_vllm.ops.flash_topk import (
    HEAD_DIM,
    HEADS,
    MAX_ROWS,
    PAGE_SIZE,
    TOP_K,
    flash_topk_workspace_specs,
)

logger = logging.getLogger(__name__)


_TOP_K = tl.constexpr(TOP_K)
_HEADS = tl.constexpr(HEADS)
_HEAD_DIM = tl.constexpr(HEAD_DIM)
_PAGE_SIZE = tl.constexpr(PAGE_SIZE)


@triton.jit
def _prefill_length(
    Starts,
    Ends,
    row,
    mask,
    MAX_CONTEXT: tl.constexpr,
    TOTAL_K: tl.constexpr,
):
    start = tl.load(Starts + row, mask, other=0)
    end = tl.load(Ends + row, mask, other=0)
    length = end - start
    valid = (start >= 0) & (end >= start) & (end <= TOTAL_K) & (length <= MAX_CONTEXT)
    return start, tl.where(valid, length, 0)


@triton.jit
def _direct_ids(
    Starts,
    Ends,
    Out,
    PREFILL: tl.constexpr,
    NEXT_N: tl.constexpr,
    SEQ_STRIDE_B: tl.constexpr,
    SEQ_STRIDE_N: tl.constexpr,
    MAX_CONTEXT: tl.constexpr,
    TOTAL_K: tl.constexpr,
):
    row = tl.program_id(0)
    ids = tl.arange(0, _TOP_K)
    if PREFILL:
        _, length = _prefill_length(Starts, Ends, row, True, MAX_CONTEXT, TOTAL_K)
    else:
        length = tl.load(
            Ends + row // NEXT_N * SEQ_STRIDE_B + row % NEXT_N * SEQ_STRIDE_N
        )
    tl.store(
        Out + row.to(tl.int64) * _TOP_K + ids,
        tl.where((ids < length) & (length <= MAX_CONTEXT), ids, -1),
    )


@triton.jit
def _prepare_prefill_lengths(
    Starts,
    Ends,
    TokenToSeq,
    Lengths,
    ROWS: tl.constexpr,
    MAX_CONTEXT: tl.constexpr,
    TOTAL_K: tl.constexpr,
    REQUESTS: tl.constexpr,
):
    row = tl.program_id(0) * 256 + tl.arange(0, 256)
    start, length = _prefill_length(Starts, Ends, row, row < ROWS, MAX_CONTEXT, TOTAL_K)
    request = tl.load(TokenToSeq + start, length > 0, other=-1)
    length = tl.where((request >= 0) & (request < REQUESTS), length, 0)
    tl.store(Lengths + row, length, row < ROWS)


@triton.jit
def _score_pages(
    Q,
    KeyPages,
    Scales,
    Weights,
    Tables,
    Scores,
    first,
    request,
    lengths,
    first_page,
    ROWS: tl.constexpr,
    QUERIES: tl.constexpr,
    PAGES_PER_CTA: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    STREAM_STORES: tl.constexpr,
):
    row = first + tl.arange(0, QUERIES)
    heads = tl.arange(0, QUERIES * _HEADS)
    dims = tl.arange(0, _HEAD_DIM)
    tokens = tl.arange(0, _PAGE_SIZE)
    query_mask = first + heads // _HEADS < ROWS
    query_pos = (first.to(tl.int64) * _HEADS + heads[:, None]) * _HEAD_DIM + dims[
        None, :
    ]
    query = tl.load(Q + query_pos, query_mask[:, None], other=0).to(
        tl.float8e4nv, bitcast=True
    )
    shared_query = tle.gpu.alloc(
        [QUERIES * _HEADS, _HEAD_DIM],
        dtype=tl.float8e4nv,
        init_value=query,
        scope=tle.gpu.smem,
    )
    weight = tl.load(
        Weights
        + (first.to(tl.int64) + heads // _HEADS) * WEIGHT_STRIDE
        + heads % _HEADS,
        query_mask,
        other=0.0,
    )
    max_length = tl.max(lengths, 0)
    end_page = tl.minimum(first_page + PAGES_PER_CTA, tl.cdiv(max_length, _PAGE_SIZE))
    for page in tl.range(first_page, end_page, num_stages=2):
        physical = tl.load(Tables + request.to(tl.int64) * TABLE_STRIDE + page)
        valid_page = (physical >= 0) & (physical < NUM_PAGES)
        # An invalid page cannot be addressed by a tensor descriptor.
        backing_page = tl.where(valid_page, physical, 0)
        keys = KeyPages.load([backing_page * 66, 0])
        pending = tle.gpu.wgmma(keys, shared_query, trans_b=True, out_dtype=tl.float32)
        dots = tle.gpu.wgmma_wait(0, pending)
        relu = tl.maximum(dots, 0.0, propagate_nan=tl.PropagateNan.ALL)
        score = tl.trans(
            tl.sum(tl.reshape(relu * weight[None, :], (_PAGE_SIZE, QUERIES, _HEADS)), 2)
        )
        scales = tl.load(Scales + backing_page.to(tl.int64) * 2112 + 2048 + tokens)
        scores = tl.where(valid_page, score * scales[None, :], -float("inf"))
        col = page * _PAGE_SIZE + tokens
        dest = Scores + row[:, None].to(tl.int64) * SCORE_STRIDE + col[None, :]
        mask = (
            (row[:, None] < ROWS)
            & (lengths[:, None] > _TOP_K)
            & (col[None, :] < lengths[:, None])
        )
        if STREAM_STORES:
            tl.store(dest, scores, mask, cache_modifier=".cs")
        else:
            tl.store(dest, scores, mask)


@triton.jit
def _paged_flash_topk_scores(
    Q,
    KeyPages,
    Scales,
    Weights,
    Tables,
    Starts,
    Ends,
    TokenToSeq,
    Lengths,
    Scores,
    ROWS: tl.constexpr,
    REQUESTS: tl.constexpr,
    TOTAL_K: tl.constexpr,
    NEXT_N: tl.constexpr,
    SEQ_STRIDE_B: tl.constexpr,
    SEQ_STRIDE_N: tl.constexpr,
    MAX_CONTEXT: tl.constexpr,
    PREFILL: tl.constexpr,
    PAGES_PER_CTA: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    PAGE_GROUPS: tl.constexpr,
    QUERY_GROUP: tl.constexpr,
    STREAM_STORES: tl.constexpr,
):
    page_tile = tl.program_id(0)
    row_tile = tl.program_id(1)
    if PREFILL:
        if QUERY_GROUP > 1:
            linear = row_tile * PAGE_GROUPS + page_tile
            group = linear // (QUERY_GROUP * PAGE_GROUPS)
            width = tl.minimum(QUERY_GROUP, tl.cdiv(ROWS, 2) - group * QUERY_GROUP)
            local = linear % (QUERY_GROUP * PAGE_GROUPS)
            page_tile = local // width
            row_tile = group * QUERY_GROUP + local % width
        first = row_tile * 2
        row = first + tl.arange(0, 2)
        lengths = tl.load(Lengths + row, row < ROWS, other=0)
        starts = tl.load(Starts + row, row < ROWS, other=0)
        reqs = tl.load(
            TokenToSeq + starts,
            (lengths > _TOP_K) & (starts >= 0) & (starts < TOTAL_K),
            other=-1,
        )
        first_page = page_tile * PAGES_PER_CTA
        eligible = (
            (lengths > _TOP_K)
            & (first_page * _PAGE_SIZE < lengths)
            & (reqs >= 0)
            & (reqs < REQUESTS)
        )
        if tl.sum(eligible.to(tl.int32), 0) == 2 and tl.min(reqs, 0) == tl.max(reqs, 0):
            _score_pages(
                Q,
                KeyPages,
                Scales,
                Weights,
                Tables,
                Scores,
                first,
                tl.min(reqs, 0),
                lengths,
                first_page,
                ROWS,
                2,
                PAGES_PER_CTA,
                TABLE_STRIDE,
                SCORE_STRIDE,
                WEIGHT_STRIDE,
                NUM_PAGES,
                STREAM_STORES,
            )
        else:
            # Pairs crossing a request boundary never share K tiles.
            for i in tl.static_range(2):
                active = tl.sum(tl.where(tl.arange(0, 2) == i, eligible, 0), 0)
                if active:
                    request = tl.sum(tl.where(tl.arange(0, 2) == i, reqs, 0), 0)
                    length = tl.sum(tl.where(tl.arange(0, 2) == i, lengths, 0), 0)
                    _score_pages(
                        Q,
                        KeyPages,
                        Scales,
                        Weights,
                        Tables,
                        Scores,
                        first + i,
                        request,
                        length + tl.full((1,), 0, tl.int32),
                        first_page,
                        ROWS,
                        1,
                        PAGES_PER_CTA,
                        TABLE_STRIDE,
                        SCORE_STRIDE,
                        WEIGHT_STRIDE,
                        NUM_PAGES,
                        STREAM_STORES,
                    )
    else:
        request = row_tile // NEXT_N
        position = row_tile % NEXT_N
        length = tl.load(Ends + request * SEQ_STRIDE_B + position * SEQ_STRIDE_N)
        length = tl.where((length >= 0) & (length <= MAX_CONTEXT), length, 0)
        if page_tile == 0:
            tl.store(Lengths + row_tile, length)
        first_page = page_tile * PAGES_PER_CTA
        if length > _TOP_K and first_page * _PAGE_SIZE < length:
            _score_pages(
                Q,
                KeyPages,
                Scales,
                Weights,
                Tables,
                Scores,
                row_tile,
                request,
                length + tl.full((1,), 0, tl.int32),
                first_page,
                ROWS,
                1,
                PAGES_PER_CTA,
                TABLE_STRIDE,
                SCORE_STRIDE,
                WEIGHT_STRIDE,
                NUM_PAGES,
                STREAM_STORES,
            )


def _matrix_contract(q, kv_cache, weights, block_table, max_context, out, prefill):
    if not isinstance(q, tuple) or len(q) != 2 or q[1] is not None:
        raise NotImplementedError("FlashTopK supports the (FP8 Q, None) input only")
    values = q[0]
    if values.device.type != "cuda" or torch.cuda.get_device_capability(
        values.device
    ) != (9, 0):
        raise NotImplementedError("FlashTopK requires NVIDIA SM90")
    if (
        values.ndim != (3 if prefill else 4)
        or values.shape[-2:] != (HEADS, HEAD_DIM)
        or values.dtype != torch.float8_e4m3fn
        or not values.is_contiguous()
    ):
        raise NotImplementedError("FlashTopK requires contiguous FP8 E4M3FN H64 D128 Q")
    rows = values.numel() // (HEADS * HEAD_DIM)
    specs = flash_topk_workspace_specs(rows, max_context)
    if (
        kv_cache.shape[1:]
        not in ((PAGE_SIZE, HEAD_DIM + 4), (PAGE_SIZE, 1, HEAD_DIM + 4))
        or kv_cache.dtype != torch.uint8
        or not kv_cache.is_contiguous()
        or kv_cache.data_ptr() % 16
        or kv_cache.shape[0] >= 2**31 // 66
    ):
        raise NotImplementedError("FlashTopK requires aligned 64-token FP8 index pages")
    if (
        weights.dtype != torch.float32
        or weights.shape != (rows, HEADS)
        or weights.stride(1) != 1
        or weights.stride(0) < HEADS
    ):
        raise ValueError("weights must be nonoverlapping FP32 [rows, 64]")
    if (
        block_table.ndim != 2
        or block_table.dtype != torch.int32
        or block_table.stride(1) != 1
        or block_table.stride(0) < block_table.shape[1]
        or max_context > block_table.shape[1] * PAGE_SIZE
        or (not prefill and block_table.shape[0] != values.shape[0])
        or (max_context > TOP_K and rows and not kv_cache.shape[0])
    ):
        raise ValueError("invalid vLLM block table or context capacity")
    if (
        out.dtype != torch.int32
        or out.shape != (rows, TOP_K)
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous INT32 [rows, 2048]")
    return values, rows, specs


def _check_buffers(device, reads, writes):
    if any(t.device != device or t.layout != torch.strided for t in (*reads, *writes)):
        raise ValueError("FlashTopK buffers must be strided tensors on the Q device")
    before = [_memory_interval(t) for t in reads]
    for tensor in writes:
        start, end = _memory_interval(tensor)
        if any(start < b and a < end for a, b in before):
            raise ValueError("FlashTopK output and workspace buffers must not overlap")
        before.append((start, end))


def _workspace(specs, workspace, device):
    if not specs:
        return ()
    if workspace is None:
        return tuple(
            torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs
        )
    if len(workspace) != len(specs):
        raise ValueError("workspace must follow flash_topk_workspace_specs")
    for tensor, (shape, dtype) in zip(workspace, specs):
        if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError("workspace must follow flash_topk_workspace_specs")
    if workspace[-1].data_ptr() % 4:
        raise ValueError("FlashTopK radix workspace requires four-byte alignment")
    return workspace


def _run(
    values,
    kv_cache,
    weights,
    block_table,
    starts,
    ends,
    token_to_seq,
    max_context,
    out,
    workspace,
    specs,
    *,
    prefill,
    next_n,
    stride_b,
    stride_n,
):
    rows = out.shape[0]
    device = values.device
    scratch = _workspace(specs, workspace, device)
    _check_buffers(
        device,
        (values, kv_cache, weights, block_table, starts, ends, token_to_seq),
        (out, *scratch),
    )
    if not rows:
        return out
    with runtime.torch_device_fn.device(device):
        if max_context <= TOP_K:
            # Fixed shape for the read-free scoring path and CUDA Graph replay.
            _direct_ids[(rows,)](
                starts,
                ends,
                out,
                prefill,
                next_n,
                stride_b,
                stride_n,
                max_context,
                token_to_seq.numel(),
                num_warps=4,
            )
            return out

        logits, lengths, radix_workspace = scratch
        if prefill:
            _prepare_prefill_lengths[(triton.cdiv(rows, 256),)](
                starts,
                ends,
                token_to_seq,
                lengths,
                rows,
                max_context,
                token_to_seq.numel(),
                block_table.shape[0],
                num_warps=4,
            )

        # Keep the measured scan tiling fixed so capture does not execute a
        # search on real output/workspace buffers. Selection reuses its existing
        # tuned cooperative radix kernels.
        if prefill:
            pages = 4 if rows <= 32 else 16
            stream_stores = rows >= 2048 and max_context >= 131072
            query_group = 2 if stream_stores else 32 if block_table.shape[0] > 1 else 1
            if stream_stores or (block_table.shape[0] > 1 and max_context >= 65536):
                pages = 32
        else:
            pages = 1 if rows <= 8 and max_context <= 8192 else 4
            if max_context == 4096 and 9 <= rows <= 16:
                pages = 2
            if max_context == 8192 and rows >= 48:
                pages = 8
            stream_stores = False
            query_group = 1
        page_groups = triton.cdiv(max_context, PAGE_SIZE * pages)
        cache = kv_cache.view(torch.float8_e4m3fn).view(-1, HEAD_DIM)
        keys = TensorDescriptor.from_tensor(cache, [PAGE_SIZE, HEAD_DIM])
        _paged_flash_topk_scores[
            (page_groups, triton.cdiv(rows, 2) if prefill else rows)
        ](
            values.view(torch.uint8),
            keys,
            kv_cache.view(torch.float32),
            weights,
            block_table,
            starts,
            ends,
            token_to_seq,
            lengths,
            logits,
            rows,
            block_table.shape[0],
            token_to_seq.numel(),
            next_n,
            stride_b,
            stride_n,
            max_context,
            prefill,
            pages,
            block_table.stride(0),
            logits.stride(0),
            weights.stride(0),
            kv_cache.shape[0],
            page_groups,
            query_group,
            stream_stores,
            num_warps=4,
            enable_fp_fusion=False,
        )
        # Narrow the visible width to the caller's real per-launch bound.
        _cooperative_topk_impl(
            logits[:, :max_context],
            lengths,
            out,
            radix_workspace,
            TOP_K,
            max_context,
            max_rows=MAX_ROWS,
        )
    return out


def flash_topk_prefill(
    q,
    kv_cache,
    weights,
    block_table,
    cu_seqlen_ks,
    cu_seqlen_ke,
    token_to_seq,
    max_context,
    out,
    *,
    workspace=None,
):
    """Consume a single rank-local vLLM prefill chunk and write its row slice.

    ``cu_seqlen_ks`` and ``cu_seqlen_ke`` are the chunk's global flattened K
    bounds, where each K start is the first token of its request.
    ``token_to_seq`` maps flattened K tokens, not query rows, to the
    corresponding row of ``block_table``. Indices written to ``out`` are relative
    to each query's own K origin. See `flash_topk_workspace_specs` for scratch.
    """
    logger.debug("GEMS_NVIDIA FLASH_TOPK_PREFILL")
    values, rows, specs = _matrix_contract(
        q, kv_cache, weights, block_table, max_context, out, True
    )
    if (
        any(
            t.dtype != torch.int32 or t.shape != (rows,) or not t.is_contiguous()
            for t in (cu_seqlen_ks, cu_seqlen_ke)
        )
        or token_to_seq.dtype != torch.int32
        or token_to_seq.ndim != 1
        or not token_to_seq.is_contiguous()
    ):
        raise ValueError("invalid vLLM prefill range or K-token-to-request metadata")
    return _run(
        values,
        kv_cache,
        weights,
        block_table,
        cu_seqlen_ks,
        cu_seqlen_ke,
        token_to_seq,
        max_context,
        out,
        workspace,
        specs,
        prefill=True,
        next_n=1,
        stride_b=1,
        stride_n=0,
    )


def flash_topk_decode(
    q,
    kv_cache,
    weights,
    block_table,
    seq_lens,
    max_context,
    out,
    *,
    workspace=None,
):
    """Consume already packed vLLM decode queries and write flat rows to out.

    Real context lengths are in ``seq_lens[B, next_n]`` (or ``[B]`` for
    next_n=1). The batch dimension alone selects the block-table row.
    """
    logger.debug("GEMS_NVIDIA FLASH_TOPK_DECODE")
    values, _, specs = _matrix_contract(
        q, kv_cache, weights, block_table, max_context, out, False
    )
    batch, next_n = values.shape[:2]
    if (
        next_n < 1
        or seq_lens.dtype != torch.int32
        or (
            seq_lens.shape != (batch, next_n)
            and not (next_n == 1 and seq_lens.shape == (batch,))
        )
    ):
        raise ValueError("seq_lens must give one INT32 effective length per decode row")
    return _run(
        values,
        kv_cache,
        weights,
        block_table,
        seq_lens,
        seq_lens,
        seq_lens,
        max_context,
        out,
        workspace,
        specs,
        prefill=False,
        next_n=next_n,
        stride_b=seq_lens.stride(0),
        stride_n=seq_lens.stride(1) if seq_lens.ndim == 2 else 0,
    )


# The caller warms the chosen shape and owns mutable graph buffers.
flash_topk_prefill.supports_cuda_graph = True
flash_topk_decode.supports_cuda_graph = True

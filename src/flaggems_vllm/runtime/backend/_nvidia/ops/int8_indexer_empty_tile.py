# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Skip wholly invalid INT8 logits tiles; all output lanes are still stored.

Original valid-tile math/cache layout/schedule remain exact. Selection is
constructor-frozen in the plugin; this numerical entry has no Torch fallback.
"""

import logging

import torch
import triton
import triton.language as tl
from flag_gems.fused.int8_sparse_indexer import int8_indexer_cache_views

from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _paged_logits_empty_tile(
    Q,
    W,
    V,
    S,
    Table,
    RowBatch,
    Lengths,
    Out,
    QT: tl.constexpr,
    QH: tl.constexpr,
    WT: tl.constexpr,
    WH: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    TS: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    MAX_K: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    keys = tl.program_id(1) * BK + tl.arange(0, BK)
    batch = tl.load(RowBatch + row)
    length = tl.load(Lengths + row)
    # Device-uniform branch. Every invalid output lane is still written.
    # Includes negative RowBatch; avoids reading Q/cache for those rows too.
    if (length <= tl.program_id(1) * BK) | (batch < 0):
        tl.store(Out + row * MAX_K + keys, float("-inf"), keys < MAX_K)
    else:
        valid = (keys < length) & (keys < MAX_K) & (batch >= 0)
        pages = tl.load(Table + batch * TS + keys // P, valid, 0)
        valid = valid & (pages >= 0)
        d = tl.arange(0, 128)
        heads = tl.arange(0, BH)
        q = tl.load(
            Q + row * QT + heads[:, None] * QH + d[None, :], heads[:, None] < H, 0
        )
        k = tl.load(
            V + pages[None, :] * VS + (keys[None, :] % P) * 128 + d[:, None],
            valid[None, :],
            0,
        )
        dots = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
        weights = tl.load(W + row * WT + heads * WH, heads < H, 0).to(tl.float32)
        scale = tl.load(S + pages * SS + keys % P, valid, 0)
        logits = tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0) * scale
        logits = tl.where(valid, logits, float("-inf"))
        tl.store(Out + row * MAX_K + keys, logits, keys < MAX_K)


@libentry()
@triton.jit
def _paged_logits_coarse32(
    Q,
    W,
    V,
    S,
    Table,
    RowBatch,
    Lengths,
    Out,
    QT: tl.constexpr,
    QH: tl.constexpr,
    WT: tl.constexpr,
    WH: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    TS: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    MAX_K: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    first = tl.program_id(1) * (BK * GROUP)
    batch = tl.load(RowBatch + row)
    length = tl.load(Lengths + row)
    if (length <= first) | (batch < 0):
        keys = first + tl.arange(0, BK * GROUP)
        tl.store(Out + row * MAX_K + keys, float("-inf"), keys < MAX_K)
    else:
        d = tl.arange(0, 128)
        heads = tl.arange(0, BH)
        q = tl.load(
            Q + row * QT + heads[:, None] * QH + d[None, :], heads[:, None] < H, 0
        )
        weights = tl.load(W + row * WT + heads * WH, heads < H, 0).to(tl.float32)
        # Keep every original 32-column dot and F32 head reduction intact.
        # Four sequential original tiles reuse Q/weights, not one BK128 dot.
        for subtile in range(GROUP):
            keys = first + subtile * BK + tl.arange(0, BK)
            valid = (keys < length) & (keys < MAX_K) & (batch >= 0)
            pages = tl.load(Table + batch * TS + keys // P, valid, 0)
            valid = valid & (pages >= 0)
            k = tl.load(
                V + pages[None, :] * VS + (keys[None, :] % P) * 128 + d[:, None],
                valid[None, :],
                0,
            )
            dots = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
            scale = tl.load(S + pages * SS + keys % P, valid, 0)
            logits = tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0) * scale
            logits = tl.where(valid, logits, float("-inf"))
            tl.store(Out + row * MAX_K + keys, logits, keys < MAX_K)


def int8_indexer_paged_logits_empty_tile(
    q,
    cache,
    weights,
    block_table,
    row_batch,
    lengths,
    max_model_len,
    page_size,
    *,
    output_buffer=None,
    pure_prefill=False,
):
    """Compute paged scores for the first ``max_model_len`` columns.

    The output is float32 ``[tokens, max_model_len]``. Scores outside each
    row's valid length or on invalid pages are negative infinity. If callers
    select a width shorter than their table, they must prove the current
    effective lengths fit that width before using the result as a full score
    set; otherwise this API intentionally computes only the requested prefix.
    ``pure_prefill`` is caller-validated execution metadata. Input cache bytes
    are not mutated. When ``output_buffer`` is provided, the caller owns its
    lifetime, disjointness and graph/stream synchronization.
    """
    logger.debug("GEMS_NVIDIA INT8_INDEXER_PAGED_LOGITS_EMPTY_TILE")
    tensors = (q, cache, weights, block_table, row_batch, lengths)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("tensor inputs required")
    if (
        q.device.type != "cuda"
        or torch.version.hip is not None
        or any(value.device != q.device for value in tensors)
    ):
        raise NotImplementedError("same-device NVIDIA CUDA tensors required")
    if any(
        value.layout != torch.strided
        or value.is_conj()
        or value.is_neg()
        or value.requires_grad
        for value in tensors
    ):
        raise NotImplementedError("ordinary strided inference tensors required")
    if type(page_size) is not int or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if type(max_model_len) is not int:
        raise ValueError("max_model_len must be an integer score capacity")
    if (
        weights.dtype != torch.float32
        or block_table.dtype not in (torch.int32, torch.int64)
        or row_batch.dtype not in (torch.int32, torch.int64)
        or lengths.dtype not in (torch.int32, torch.int64)
    ):
        raise NotImplementedError("FP32 head weights and INT32/INT64 metadata required")
    if (
        q.ndim != 3
        or q.shape[1] <= 0
        or q.stride(-1) != 1
        or row_batch.ndim != 1
        or lengths.ndim != 1
        or row_batch.stride(0) != 1
        or lengths.stride(0) != 1
    ):
        raise NotImplementedError(
            "D128 unit-stride heads and contiguous row/length vectors required"
        )
    if q.dtype != torch.int8 or q.ndim != 3 or q.shape[-1] != 128:
        raise ValueError("INT8 DSA queries must have shape [tokens, heads, 128]")
    if (
        weights.shape != q.shape[:2]
        or row_batch.numel() != q.shape[0]
        or lengths.numel() != q.shape[0]
    ):
        raise ValueError("INT8 DSA query metadata shape mismatch")
    if block_table.ndim != 2 or block_table.stride(1) != 1:
        raise ValueError("INT8 DSA block table must have contiguous rows")
    if (
        max_model_len < 0
        or triton.cdiv(max_model_len, page_size) > block_table.shape[1]
    ):
        raise ValueError("INT8 DSA maximum context exceeds the block table")
    if type(pure_prefill) is not bool:
        raise ValueError("pure_prefill must be a framework-validated bool")
    q2048 = (
        pure_prefill
        and q.shape == (2048, 32, 128)
        and max_model_len == 34816
        and page_size == 64
        and q.stride() == (4096, 128, 1)
        and weights.dtype == torch.float32
        and weights.stride() == (32, 1)
        and block_table.dtype == torch.int32
        and block_table.shape == (1, 544)
        and row_batch.dtype == torch.int64
        and row_batch.stride() == (1,)
        and lengths.dtype == torch.int32
        and lengths.stride() == (1,)
    )
    if output_buffer is not None:
        if (
            not q2048
            or output_buffer.dtype != torch.float32
            or output_buffer.shape != (2048, 34816)
            or output_buffer.device != q.device
            or not output_buffer.is_contiguous()
        ):
            raise ValueError(
                "output_buffer requires exact pureP F32[2048,34816] on query device"
            )
        for value in (q, cache, weights, block_table, row_batch, lengths):
            if value.device != q.device or torch._C._overlaps(output_buffer, value):
                raise ValueError(
                    "output_buffer must be independent of active inputs on the same device"
                )
    values, scales = int8_indexer_cache_views(cache, page_size)
    out = (
        output_buffer
        if output_buffer is not None
        else torch.empty(
            (q.shape[0], max_model_len), dtype=torch.float32, device=q.device
        )
    )
    bounded_16k = (
        pure_prefill
        and output_buffer is None
        and q.shape[0] in (256, 1024, 2048)
        and q.shape[1] == 32
        and max_model_len == 16384
        and page_size == 64
        and q.is_cuda
        and torch.version.hip is None
        and q.stride() == (4096, 128, 1)
        and weights.dtype == torch.float32
        and weights.stride() == (32, 1)
        and block_table.dtype == torch.int32
        and block_table.shape[0] == 1
        and row_batch.dtype == torch.int64
        and row_batch.stride() == (1,)
        and lengths.dtype == torch.int32
        and lengths.stride() == (1,)
        and all(
            x.device == q.device
            for x in (cache, weights, block_table, row_batch, lengths)
        )
        and torch.cuda.get_device_capability(q.device) == (9, 0)
    )
    with torch.cuda.device(q.device):
        if (
            q2048
            or bounded_16k
            or (
                q.shape[0] in (256, 1004, 1024)
                and q.shape[1] == 32
                and (q.shape[0] != 256 or row_batch.dtype == torch.int64)
                and max_model_len == 34816
                and page_size == 64
                and q.stride() == (4096, 128, 1)
                and weights.dtype == torch.float32
                and weights.stride() == (32, 1)
                and block_table.dtype == torch.int32
                and row_batch.dtype in (torch.int32, torch.int64)
                and row_batch.stride() == (1,)
                and lengths.dtype == torch.int32
                and lengths.stride() == (1,)
            )
        ):
            _paged_logits_coarse32[(q.shape[0], triton.cdiv(max_model_len, 128))](
                q,
                weights,
                values,
                scales,
                block_table,
                row_batch,
                lengths,
                out,
                q.stride(0),
                q.stride(1),
                weights.stride(0),
                weights.stride(1),
                values.stride(0),
                scales.stride(0),
                block_table.stride(0),
                q.shape[1],
                page_size,
                max_model_len,
                triton.next_power_of_2(max(q.shape[1], 16)),
                32,
                4,
                num_warps=4,
            )
        elif q.shape[0] and max_model_len:
            _paged_logits_empty_tile[(q.shape[0], triton.cdiv(max_model_len, 32))](
                q,
                weights,
                values,
                scales,
                block_table,
                row_batch,
                lengths,
                out,
                q.stride(0),
                q.stride(1),
                weights.stride(0),
                weights.stride(1),
                values.stride(0),
                scales.stride(0),
                block_table.stride(0),
                q.shape[1],
                page_size,
                max_model_len,
                triton.next_power_of_2(max(q.shape[1], 16)),
                32,
                num_warps=4,
            )
    return out


__all__ = ["int8_indexer_paged_logits_empty_tile"]

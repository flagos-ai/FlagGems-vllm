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

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl

# ---- scheduling ----


@triton.jit
def effective_kv_splits(
    length, SPLITS: tl.constexpr, STACK: tl.constexpr, MIN_LENGTH: tl.constexpr = 32768
):
    """Keep aligned long-KV spans, and omit empty or short-KV splits."""
    if SPLITS > 1:
        span = tl.maximum(STACK, tl.cdiv(length, SPLITS * STACK) * STACK)
        active = tl.maximum(1, tl.cdiv(length, span))
        active = tl.where(length >= MIN_LENGTH, active, 1)
        span = tl.where(
            active > 1, span, tl.maximum(STACK, tl.cdiv(length, STACK) * STACK)
        )
    else:
        active = 1
        span = tl.maximum(STACK, tl.cdiv(length, STACK) * STACK)
    return active, span


# ---- scheduling ----


@dataclass(frozen=True)
class AscendAttentionPlan:
    # Keep the FA3 512-key probability-rounding boundary independently of the
    # smaller Cube tiles. Conservative first-version tiles fit the 910B UB.
    block_m: int = 16
    block_n: int = 32
    kv_stack: int = 512
    short_decode: bool = False
    kv_splits: int = 1


# ---- scheduling ----


def build_plan(params, num_splits):
    """Select the initial unsplit paged forward path after shared preparation."""
    if type(num_splits) is not int or num_splits not in (0, 1):
        raise NotImplementedError("Ascend attention supports num_splits=0 or 1")
    if (
        params.d not in (128, 192, 256)
        or not params.is_paged
        or params.block_size not in (16, 32)
    ):
        raise NotImplementedError(
            "Ascend attention requires D128/D192/D256 and page16/page32 KV"
        )
    if not params.is_seqused_k:
        raise NotImplementedError("Ascend paged attention requires seqused_k")
    if not params.is_cu_seqlens_q and not params.seqlenq_ngroups_swapped:
        raise NotImplementedError(
            "Ascend attention requires varlen or swapped decode Q"
        )
    if any(
        (
            params.is_dropout,
            params.return_softmax,
            params.is_local,
            params.is_alibi,
            params.is_softcap,
        )
    ):
        raise NotImplementedError(
            "Ascend attention does not support dropout, probabilities, windows, ALiBi or softcap"
        )
    splits = 1
    if (
        num_splits == 0
        and params.is_cu_seqlens_q
        and params.seqlen_q <= 64
        and params.seqlen_k >= 32768
        and params.total_q * params.h <= 4096
    ):
        workers = torch.npu.get_device_limit(params.q_ptr.device.index)["cube_core_num"]
        # Fill an otherwise sparse Cube wave without creating extra waves.
        tasks = max(1, (params.total_q * params.h + 15) // 16)
        splits = min(16, workers // tasks) if tasks <= workers else 4
    return AscendAttentionPlan(
        short_decode=(
            params.is_cu_seqlens_q
            and params.seqlen_q == 1
            and 0 < params.seqlen_k <= 32
            and params.h // params.hk <= 32
            and not params.is_causal
        ),
        kv_splits=splits,
    )


# ---- scheduling ----


@triton.jit
def _classify_long_requests(
    CUQ, USED, IDS, COUNT, B: tl.constexpr, BLOCK: tl.constexpr
):
    rows = tl.arange(0, BLOCK)
    first = tl.load(CUQ + rows, rows < B, 0)
    end = tl.load(CUQ + rows + 1, rows < B, 0)
    length = tl.load(USED + rows, rows < B, 0)
    qlen = end - first
    keep = (rows < B) & ((qlen > 1) | ((qlen == 1) & (length > 32)))
    positions = tl.cumsum(keep.to(tl.int32), 0) - 1
    tl.store(IDS + positions, rows, keep)
    tl.store(COUNT, tl.sum(keep.to(tl.int32), 0))


# ---- scheduling ----


@triton.jit
def classify_long_requests(CUQ, USED, IDS, COUNT, B: tl.constexpr, BLOCK: tl.constexpr):
    _classify_long_requests(CUQ, USED, IDS, COUNT, B, BLOCK)


# ---- scheduling ----


def allocate_long_request_list(params):
    ids = torch.empty((params.b,), dtype=torch.int32, device=params.q_ptr.device)
    count = torch.empty((), dtype=torch.int32, device=params.q_ptr.device)
    return ids, count


# ---- scheduling ----


def build_long_request_list(params):
    """Keep original batch IDs for requests not handled by actual-short Decode."""
    ids, count = allocate_long_request_list(params)
    classify_long_requests[(1,)](
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        ids,
        count,
        params.b,
        triton.next_power_of_2(params.b),
    )
    return ids, count


# ---- scheduling ----


@triton.jit
def classify_mixed_requests(
    CUQ, DECODE_IDS, OTHER_IDS, COUNTS, B: tl.constexpr, BLOCK: tl.constexpr
):
    rows = tl.arange(0, BLOCK)
    first = tl.load(CUQ + rows, rows < B, 0)
    end = tl.load(CUQ + rows + 1, rows < B, 0)
    qlen = end - first
    decode = (rows < B) & (qlen == 1)
    other = (rows < B) & (qlen > 1)
    decode_pos = tl.cumsum(decode.to(tl.int32), 0) - 1
    other_pos = tl.cumsum(other.to(tl.int32), 0) - 1
    tl.store(DECODE_IDS + decode_pos, rows, decode)
    tl.store(OTHER_IDS + other_pos, rows, other)
    tl.store(COUNTS, tl.sum(decode.to(tl.int32), 0))
    tl.store(COUNTS + 1, tl.sum(other.to(tl.int32), 0))


# ---- scheduling ----


def build_mixed_request_lists(params, *, defer=False):
    decode_ids = torch.empty((params.b,), dtype=torch.int32, device=params.q_ptr.device)
    other_ids = torch.empty_like(decode_ids)
    counts = torch.empty((2,), dtype=torch.int32, device=params.q_ptr.device)
    if not defer:
        classify_mixed_requests[(1,)](
            params.cu_seqlens_q_ptr,
            decode_ids,
            other_ids,
            counts,
            params.b,
            triton.next_power_of_2(params.b),
        )
    return decode_ids, other_ids, counts


# ---- scheduling ----


def build_mixed_decode_plan(params, plan, num_splits, workers):
    """Allow low-parallelism mixed Q1 requests to split independently."""
    if (
        num_splits == 0
        and params.b <= 8
        and params.seqlen_k >= 32768
        and params.b * params.hk < workers
    ):
        return replace(plan, kv_splits=4)
    return plan


# ---- splitkv ----


@triton.jit
def _merge_request(
    PARTIAL,
    PARTIAL_LSE,
    O,
    LSE,
    CUQ,
    USED,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    STACK: tl.constexpr,
    batch,
    head,
    SKIP_SHORT_KV: tl.constexpr = 0,
    PARTIAL_Q: tl.constexpr = 0,
    BYPASS_SHORT: tl.constexpr = False,
    MIN_SPLIT_K: tl.constexpr = 32768,
):
    PARTIAL_ROWS: tl.constexpr = PARTIAL_Q if PARTIAL_Q else TOTAL_Q
    first = tl.load(CUQ + batch)
    end = tl.load(CUQ + batch + 1)
    length = tl.load(USED + batch)
    handled = (SKIP_SHORT_KV > 0) & (end - first == 1) & (length <= SKIP_SHORT_KV)
    active, _ = effective_kv_splits(length, SPLITS, STACK, MIN_SPLIT_K)
    if BYPASS_SHORT:
        handled = handled | (active == 1)
    if not handled:
        parts = tl.arange(0, BLOCK_SPLITS)
        dims = tl.arange(0, D)
        for index in range(first, end):
            row = index.to(tl.int64)
            partial_row = batch.to(tl.int64) if PARTIAL_Q else row
            if active == 1:
                # Unused partial slots are intentionally never written or read.
                result = tl.load(PARTIAL + (partial_row * HEADS + head) * D + dims)
                merged_lse = tl.load(PARTIAL_LSE + head * PARTIAL_ROWS + partial_row)
            else:
                ls = tl.load(
                    PARTIAL_LSE
                    + parts * HEADS * PARTIAL_ROWS
                    + head * PARTIAL_ROWS
                    + partial_row,
                    parts < active,
                    -float("inf"),
                )
                maximum = tl.max(ls, 0)
                valid = maximum > -float("inf")
                safe_max = tl.where(valid, maximum, 0.0)
                weights = tl.exp(ls - safe_max)
                denom = tl.sum(weights, 0)
                safe_denom = tl.where(valid, denom, 1.0)
                values = tl.load(
                    PARTIAL
                    + parts[:, None].to(tl.int64) * PARTIAL_ROWS * HEADS * D
                    + (partial_row * HEADS + head) * D
                    + dims[None, :],
                    parts[:, None] < active,
                    0.0,
                )
                result = tl.sum(values * (weights / safe_denom)[:, None], 0)
                merged_lse = tl.where(
                    valid, safe_max + tl.log(safe_denom), -float("inf")
                )
            tl.store(O + row * OR + head * OH + dims, result)
            tl.store(LSE + head * TOTAL_Q + row, merged_lse)


# ---- splitkv ----


@triton.jit
def merge_request_list(
    PARTIAL,
    PARTIAL_LSE,
    O,
    LSE,
    CUQ,
    USED,
    REQUEST_IDS,
    REQUEST_COUNT,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    STACK: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr,
    PARTIAL_Q: tl.constexpr,
    BYPASS_SHORT: tl.constexpr,
    worker,
    workers,
    MIN_SPLIT_K: tl.constexpr = 32768,
):
    count = tl.load(REQUEST_COUNT)
    for task in range(worker, count * HEADS, workers):
        batch = tl.load(REQUEST_IDS + task // HEADS)
        head = task % HEADS
        _merge_request(
            PARTIAL,
            PARTIAL_LSE,
            O,
            LSE,
            CUQ,
            USED,
            OR,
            OH,
            TOTAL_Q,
            HEADS,
            D,
            SPLITS,
            BLOCK_SPLITS,
            STACK,
            batch,
            head,
            SKIP_SHORT_KV,
            PARTIAL_Q,
            BYPASS_SHORT,
            MIN_SPLIT_K,
        )


# ---- splitkv ----


@triton.jit
def merge_q_rows(
    PARTIAL,
    PARTIAL_LSE,
    O,
    LSE,
    CUQ,
    USED,
    REQUEST_IDS,
    REQUEST_COUNT,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    STACK: tl.constexpr,
    B: tl.constexpr,
    REQUEST_LIST: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr,
    BYPASS_SHORT: tl.constexpr,
    MIN_SPLIT_K: tl.constexpr,
    CLIP_TAIL: tl.constexpr = False,
):
    # Spread actual Q/head rows across Vector workers, without host length reads.
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    count = B
    if REQUEST_LIST:
        count = tl.load(REQUEST_COUNT)
    cursor = 0
    parts = tl.arange(0, BLOCK_SPLITS)
    dims = tl.arange(0, D)
    for request in range(count):
        batch = request
        if REQUEST_LIST:
            batch = tl.load(REQUEST_IDS + request)
        first = tl.load(CUQ + batch)
        end = tl.load(CUQ + batch + 1)
        qlen = end - first
        if CLIP_TAIL:
            tail = qlen % 128
            end -= tl.where((qlen > 128) & (tail > 0) & (tail <= 4), tail, 0)
        length = tl.load(USED + batch)
        active, _ = effective_kv_splits(length, SPLITS, STACK, MIN_SPLIT_K)
        handled = (SKIP_SHORT_KV > 0) & (qlen == 1) & (length <= SKIP_SHORT_KV)
        if BYPASS_SHORT:
            handled = handled | (active == 1)
        if not handled:
            ntasks = (end - first) * HEADS
            begin = (worker + workers - cursor) % workers
            for task in range(begin, ntasks, workers):
                row = (first + task // HEADS).to(tl.int64)
                head = task % HEADS
                if active == 1:
                    result = tl.load(PARTIAL + (row * HEADS + head) * D + dims)
                    merged_lse = tl.load(PARTIAL_LSE + head * TOTAL_Q + row)
                else:
                    ls = tl.load(
                        PARTIAL_LSE + parts * HEADS * TOTAL_Q + head * TOTAL_Q + row,
                        parts < active,
                        -float("inf"),
                    )
                    maximum = tl.max(ls, 0)
                    valid = maximum > -float("inf")
                    safe_max = tl.where(valid, maximum, 0.0)
                    weights = tl.exp(ls - safe_max)
                    denom = tl.sum(weights, 0)
                    safe_denom = tl.where(valid, denom, 1.0)
                    values = tl.load(
                        PARTIAL
                        + parts[:, None].to(tl.int64) * TOTAL_Q * HEADS * D
                        + (row * HEADS + head) * D
                        + dims[None, :],
                        parts[:, None] < active,
                        0.0,
                    )
                    result = tl.sum(values * (weights / safe_denom)[:, None], 0)
                    merged_lse = tl.where(
                        valid, safe_max + tl.log(safe_denom), -float("inf")
                    )
                tl.store(O + row * OR + head * OH + dims, result)
                tl.store(LSE + head * TOTAL_Q + row, merged_lse)
            cursor = (cursor + ntasks) % workers


# ---- splitkv ----


@triton.jit
def merge_splits(
    PARTIAL,
    PARTIAL_LSE,
    O,
    LSE,
    CUQ,
    USED,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    STACK: tl.constexpr,
    SKIP_SHORT_KV: tl.constexpr = 0,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
    PARTIAL_Q: tl.constexpr = 0,
    BYPASS_SHORT: tl.constexpr = False,
    MIN_SPLIT_K: tl.constexpr = 32768,
    PARALLEL_Q: tl.constexpr = False,
    B: tl.constexpr = 0,
):
    if PARALLEL_Q:
        tl.static_assert(PARTIAL_Q == 0)
        merge_q_rows(
            PARTIAL,
            PARTIAL_LSE,
            O,
            LSE,
            CUQ,
            USED,
            REQUEST_IDS,
            REQUEST_COUNT,
            OR,
            OH,
            TOTAL_Q,
            HEADS,
            D,
            SPLITS,
            BLOCK_SPLITS,
            STACK,
            B,
            REQUEST_LIST,
            SKIP_SHORT_KV,
            BYPASS_SHORT,
            MIN_SPLIT_K,
        )
    elif REQUEST_LIST:
        merge_request_list(
            PARTIAL,
            PARTIAL_LSE,
            O,
            LSE,
            CUQ,
            USED,
            REQUEST_IDS,
            REQUEST_COUNT,
            OR,
            OH,
            TOTAL_Q,
            HEADS,
            D,
            SPLITS,
            BLOCK_SPLITS,
            STACK,
            SKIP_SHORT_KV,
            PARTIAL_Q,
            BYPASS_SHORT,
            tl.program_id(0),
            tl.num_programs(0),
            MIN_SPLIT_K,
        )
    else:
        batch = tl.program_id(0)
        head = tl.program_id(1)
        _merge_request(
            PARTIAL,
            PARTIAL_LSE,
            O,
            LSE,
            CUQ,
            USED,
            OR,
            OH,
            TOTAL_Q,
            HEADS,
            D,
            SPLITS,
            BLOCK_SPLITS,
            STACK,
            batch,
            head,
            SKIP_SHORT_KV,
            PARTIAL_Q,
            BYPASS_SHORT,
            MIN_SPLIT_K,
        )

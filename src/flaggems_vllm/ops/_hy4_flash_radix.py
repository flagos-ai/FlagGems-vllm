# SPDX-License-Identifier: Apache-2.0
"""Flash sorted group selection using ping-pong radix histograms.

Each next pass reduces the preceding immutable histogram in its CTAs. Separate
histogram slots prevent read/write races, while kernel boundaries order passes.
The final scatter reduces the last histogram and writes ascending IDs directly.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.cooperative_topk import (
    _key_greater,
    _masked_histogram,
    _ordered_key,
    _radix_digit,
)
from flaggems_vllm.utils import libentry


@triton.jit
def _threshold(
    W,
    ROW,
    PARTS: tl.constexpr,
    WORDS: tl.constexpr,
    PASS: tl.constexpr,
    K: tl.constexpr,
):
    base = W + ROW * WORDS
    parts = tl.arange(0, PARTS)
    bins = tl.arange(0, 256)
    hist = tl.load(
        base + (PASS % 2) * PARTS * 256 + parts[:, None] * 256 + bins[None, :]
    )
    state = base + 2 * PARTS * 256
    prefix = tl.full((), 0, tl.uint32)
    remaining = tl.full((), K, tl.int32)
    greater = tl.full((PARTS,), 0, tl.int32)
    if PASS > 0:
        prev = state + (PASS - 1) * (PARTS + 2)
        prefix = tl.load(prev).to(tl.uint32, bitcast=True)
        remaining = tl.load(prev + 1)
        greater = tl.load(prev + 2 + parts)
    digit, remaining = _radix_digit(tl.sum(hist, 0), remaining, 256)
    greater += tl.sum(tl.where(bins[None, :] > digit.to(tl.int32), hist, 0), 1)
    ties = tl.sum(tl.where(bins[None, :] == digit.to(tl.int32), hist, 0), 1)
    prefix = prefix | (digit << ((3 - PASS) * 8))
    return prefix, remaining, greater, ties


@libentry()
@triton.jit
def _histogram(
    Logits,
    U,
    W,
    WIDTH: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    PARTS: tl.constexpr,
    WORDS: tl.constexpr,
    PASS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    part = tl.program_id(0).to(tl.int64)
    row = tl.program_id(1).to(tl.int64)
    prefix = tl.full((), 0, tl.uint32)
    if PASS > 0:
        prefix, remaining, greater, ties = _threshold(W, row, PARTS, WORDS, PASS - 1, K)
        if part == 0:
            state = W + row * WORDS + 2 * PARTS * 256 + (PASS - 1) * (PARTS + 2)
            tl.store(state, prefix.to(tl.int32, bitcast=True))
            tl.store(state + 1, remaining)
            tl.store(state + 2 + tl.arange(0, PARTS), greater)
    length = tl.minimum(tl.maximum(tl.load(U + row), 0), WIDTH)
    hist = tl.full((256,), 0, tl.int32)
    if length > K:
        chunk: tl.constexpr = triton.cdiv(WIDTH, PARTS)
        end = tl.minimum((part + 1) * chunk, length)
        for start in range(part * chunk, end, BLOCK):
            j = start + tl.arange(0, BLOCK)
            valid = j < end
            x = tl.load(Logits + row * S0 + j * S1, valid, other=-float("inf"))
            valid = valid & (x != -float("inf"))
            keys = _ordered_key(x)
            if PASS > 0:
                valid = valid & (
                    (keys >> ((4 - PASS) * 8)) == (prefix >> ((4 - PASS) * 8))
                )
            digits = ((keys >> ((3 - PASS) * 8)) & 255).to(tl.int32)
            hist += _masked_histogram(digits, valid, 256)
    tl.store(
        W + row * WORDS + (PASS % 2) * PARTS * 256 + part * 256 + tl.arange(0, 256),
        hist,
    )


@libentry()
@triton.jit
def _scatter(
    Logits,
    U,
    W,
    Output,
    WIDTH: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    PARTS: tl.constexpr,
    WORDS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    part = tl.program_id(0).to(tl.int64)
    row = tl.program_id(1).to(tl.int64)
    length = tl.minimum(tl.maximum(tl.load(U + row), 0), WIDTH)
    if length <= K:
        if part == 0:
            j = tl.arange(0, triton.next_power_of_2(K))
            x = tl.load(Logits + row * S0 + j * S1, j < length, other=-float("inf"))
            valid = (j < length) & (x != -float("inf"))
            rank = tl.cumsum(valid.to(tl.int32), 0) - 1
            tl.store(Output + row * K + rank, j, valid)
            tail = tl.sum(valid.to(tl.int32), 0) + j
            tl.store(Output + row * K + tail, -1, tail < K)
    else:
        threshold, remaining, greater, ties = _threshold(W, row, PARTS, WORDS, 3, K)
        parts = tl.arange(0, PARTS)
        before_greater = tl.sum(tl.where(parts < part, greater, 0), 0)
        tie_offset = tl.sum(tl.where(parts < part, ties, 0), 0)
        before = before_greater + tl.minimum(tie_offset, remaining)
        selected_before = tl.full((), 0, tl.int32)
        chunk: tl.constexpr = triton.cdiv(WIDTH, PARTS)
        end = tl.minimum((part + 1) * chunk, length)
        for start in range(part * chunk, end, BLOCK):
            j = start + tl.arange(0, BLOCK)
            valid = j < end
            x = tl.load(Logits + row * S0 + j * S1, valid, other=-float("inf"))
            valid = valid & (x != -float("inf"))
            keys = _ordered_key(x)
            gt = valid & _key_greater(keys, threshold)
            eq = valid & (keys == threshold)
            tie_rank = tie_offset + tl.cumsum(eq.to(tl.int32), 0)
            chosen = gt | (eq & (tie_rank <= remaining))
            rank = tl.cumsum(chosen.to(tl.int32), 0) - 1
            tl.store(Output + row * K + before + selected_before + rank, j, chosen)
            selected_before += tl.sum(chosen.to(tl.int32), 0)
            tie_offset += tl.sum(eq.to(tl.int32), 0)
        if part == 0:
            total = tl.sum(greater, 0) + tl.minimum(tl.sum(ties, 0), remaining)
            tail = total + tl.arange(0, triton.next_power_of_2(K))
            tl.store(Output + row * K + tail, -1, tail < K)


def select_wide(logits, upper, out, parts=None):
    n, width = logits.shape
    k = out.shape[1]
    parts = (
        (2 if n >= 1024 else 4 if width > 8192 and n >= 128 else 8)
        if parts is None
        else parts
    )
    if parts not in (2, 4, 8, 16):
        raise ValueError("Unsupported Flash radix partition count")
    words = triton.next_power_of_2(2 * parts * 256 + 4 * (parts + 2))
    workspace = torch.empty((n * words,), dtype=torch.int32, device=logits.device)
    with torch.cuda.device(logits.device):
        for step in range(4):
            _histogram[(parts, n)](
                logits,
                upper,
                workspace,
                width,
                *logits.stride(),
                parts,
                words,
                step,
                k,
                1024,
                num_warps=4,
            )
        _scatter[(parts, n)](
            logits,
            upper,
            workspace,
            out,
            width,
            *logits.stride(),
            parts,
            words,
            k,
            1024,
            num_warps=4,
        )
    return out

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compact GPU tile mapping, adapted from the portable MetaX worklist idea."""

import triton
import triton.language as tl


@triton.jit
def compact_tile_coords(
    task,
    cuq,
    BATCH: tl.constexpr,
    BB: tl.constexpr,
    BM: tl.constexpr,
    HEAD_PACK: tl.constexpr,
):
    bids = tl.arange(0, BB)
    starts = tl.load(cuq + bids, bids < BATCH, 0)
    ends = tl.load(cuq + bids + 1, bids < BATCH, 0)
    counts = tl.where(bids < BATCH, tl.cdiv((ends - starts) * HEAD_PACK, BM), 0)
    prefix = tl.cumsum(counts)
    before = prefix - counts
    hit = (bids < BATCH) & (task >= before) & (task < prefix)
    batch = tl.sum(tl.where(hit, bids, 0), 0)
    tile = tl.sum(tl.where(hit, task - before, 0), 0)
    valid = task < tl.sum(counts, 0)
    return batch, tile, valid

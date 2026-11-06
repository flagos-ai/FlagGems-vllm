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

"""Ascend SparseAttnSharedKV decode over an FP8 packed KV cache.

`sparse_attn_sharedkv` serves DSV4 shared-KV attention with a BF16 or FP16
cache. This file adds the decode step over the packed FP8 cache, which costs
584 bytes per token against 1024 for BF16: per page, `PAGE` tokens of a
576-byte record (448 e4m3 NoPE bytes, then 64 BF16 RoPE values) followed by the
page's scale block, eight bytes per token carrying the seven ue8m0 scales of
its seven 64-wide NoPE groups. 910B4 has no hardware FP8, so the cache is
widened to BF16 in software before the cube reads it.

A request attends to the last `window` positions of the original KV through
`ori_block_table`, then, of its `floor(seqused / cmp_ratio)` compressed
positions, to those `cmp_sparse_indices` names at ratio 4 or to the whole
prefix at ratio 128, through `cmp_block_table`; the per-head sink adds
denominator mass without a value.

One call builds the schedule -- the (page, slot) pair of every position, each
request's token count and the tiles' assignment to the programs -- then runs
one attention kernel, two for the split-K shape. The attention kernel
dequantises a tile with a vector device function, hands it to the cube through
global memory -- c220 has no unified-buffer-to-L1 path -- and keeps the output
accumulator in L0C across a request's tiles.

The three device functions are FlagTree custom ops --
`dequant_e4m3_ue8m0_rows`, `mma_m64_tile` and `mma_m64_resident` -- so their
Ascend C and bitcode ship with the toolchain and this file only names them and
passes this cache's page geometry.

They reach the cube through `hivm.hir.custom`, which CANN 9.0.0's bishengir-opt
rejects; building this file needs the 9.1.0 toolchain (AscendNPU-IR 1.2.0) on
PATH, and the kernels it emits run on either runtime.
"""

# No `from __future__ import annotations` here, unlike its BF16 sibling: the
# custom-op registry reads the device functions' parameter annotations to type
# their scalar operands, and PEP 563 would turn those into strings.
from typing import Optional

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

HEADS = 64
HEAD_DIM = 512
NOPE_BYTES = 448
ROPE_DIM = HEAD_DIM - NOPE_BYTES
SCALE_GROUPS = 7
PAGE = 128
TOKEN_RECORD = 576
PAGE_STRIDE = (
    (PAGE * (TOKEN_RECORD + 8) + TOKEN_RECORD - 1) // TOKEN_RECORD * TOKEN_RECORD
)


# The same geometry in the 8-byte words the device functions address with.
PAGE_STRIDE_WORDS = PAGE_STRIDE // 8
PAGE_BODY_WORDS = PAGE * TOKEN_RECORD // 8


def _compressed_extent(cmp_ratio, cmp_block_table, max_model_len):
    """Compressed positions a CFA call has to address, rounded to a tile.

    CFA takes no index list: a request attends its whole `floor(seqused /
    ratio)` prefix, so the extent follows the longest sequence the caller can
    present -- the block table's capacity, narrowed by `max_model_len` when it
    is given -- and not an index list's width. The BF16 entry sizes its mode-1
    workspace the same way.
    """
    if cmp_ratio <= 1 or cmp_block_table is None:
        return 0
    extent = cmp_block_table.shape[1] * PAGE
    if max_model_len:
        extent = min(extent, -(-int(max_model_len) // int(cmp_ratio)))
    return max(PAGE, -(-extent // PAGE) * PAGE)


# Unified-buffer scratch the dequantisation device functions index, in BF16
# columns per row, and the rows one vector sub-block converts per call. Both
# are part of their ABI.
DEQUANT_SCRATCH_COLS = 832
DEQUANT_ROWS = 64


# Tokens per attention tile, and the output width the cube keeps resident in
# L0C across a request's tiles.
BLOCK_N = 128
PERSISTENT_COLS = 384


# Programs a pipeline spreads its tiles over: 910B4-1 has 20 cube cores and a
# program owns one.
PROGRAM_LIMIT = 20


# The device functions live in FlagTree as registered custom ops, so this file
# only names them. The page geometry they take as arguments is this cache's:
# `PAGE_STRIDE` bytes per page, `TOKEN_RECORD` per token record, and the page's
# scale block at `SCALE_BLOCK`.
TOKEN_RECORD = 576
SCALE_BLOCK = PAGE * TOKEN_RECORD
PAGE_STRIDE = (
    (PAGE * (TOKEN_RECORD + 8) + TOKEN_RECORD - 1) // TOKEN_RECORD * TOKEN_RECORD
)

# `tle.dsa.ascend.raw` rather than `tle_raw.call`: the two emit the same IR,
# and the older entry also works on toolchains whose `call` predates its
# variadic signature.
DEQUANT_OP = "dequant_e4m3_ue8m0_rows"
MMA_TILE_OP = "mma_m64_tile"
MMA_RESIDENT_OP = "mma_m64_resident"

# A kernel cannot read a plain module global, so the values it passes to those
# ops are mirrored as constexpr; the plain ones above stay for the host.
_PAGE_STRIDE = tl.constexpr(PAGE_STRIDE)
_TOKEN_RECORD = tl.constexpr(TOKEN_RECORD)
_SCALE_BLOCK = tl.constexpr(SCALE_BLOCK)
# Records per page, and the logical rows the window region spans in the map.
# `_check` accepts only window == PAGE, so the two coincide.
_PAGE_SLOTS = tl.constexpr(PAGE)
_WINDOW_EXTENT = tl.constexpr(PAGE)
_TILE_TOKENS = tl.constexpr(BLOCK_N)
_RESIDENT_COLS = tl.constexpr(PERSISTENT_COLS)
# The window pipeline names its cube op directly rather than taking it
# as a parameter, so that name is a constexpr too.
_MMA_TILE_OP = tl.constexpr(MMA_TILE_OP)


@triton.jit
def _window_dequant_tile(
    request,
    tile,
    slot,
    sub,
    cache_words,
    Pages,
    Slots,
    count,
    win_count,
    Tile,
    BN: tl.constexpr,
    SB: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
    scratch,
):
    """Dequantise tile `tile` of `request` into staging slot `slot`.

    `request` selects the (page, slot) map row and the counts; `slot` selects
    the destination, which is a staging slot owned by the calling program and
    not generally the request index. The two sub-vector cores split the tile,
    each taking SB rows of every 2 * SB.
    """
    pool = tl.make_block_ptr(cache_words, (1 << 40, 1), (1, 1), (0, 0), (SB, 1), (1, 0))
    pp = tl.make_block_ptr(
        Pages + request * MAP_STRIDE,
        (MAP_STRIDE, 1),
        (1, 1),
        (0, 0),
        (MAP_STRIDE, 1),
        (1, 0),
    )
    sp = tl.make_block_ptr(
        Slots + request * MAP_STRIDE,
        (MAP_STRIDE, 1),
        (1, 1),
        (0, 0),
        (MAP_STRIDE, 1),
        (1, 0),
    )
    for piece in range(0, BN // (2 * SB)):
        first = tile * BN + piece * 2 * SB + sub * SB
        target = slot * BN + piece * 2 * SB + sub * SB
        dst = tl.make_block_ptr(
            Tile + target * 512,
            (SB, 512),
            (512, 1),
            (0, 0),
            (SB, 512),
            (1, 0),
        )
        tle.dsa.ascend.raw(
            DEQUANT_OP,
            pool,
            pool,
            pp,
            sp,
            first,
            win_count,
            count,
            _PAGE_STRIDE,
            _TOKEN_RECORD,
            _SCALE_BLOCK,
            _PAGE_SLOTS,
            _WINDOW_EXTENT,
            dst,
            scratch,
        )


@triton.jit
def _sparse_attn_sharedkv_fp8_window_kernel(
    Q,
    cache_words,
    Pages,
    Slots,
    Count,
    WinCount,
    Sink,
    Tile,
    Scores,
    Weights,
    Partial,
    Out,
    BATCH: tl.constexpr,
    PROGRAMS: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    SCRATCH_COLS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
):
    """Window-only decode: one program walks requests pid, pid + PROGRAMS, ...

    Every request's window is at most 128 tokens, so one staged tile covers it
    and the attention needs no running maximum. Vector and cube work on the
    same program are handed to each other through global memory, so each handoff
    carries a block sync:

      flag 3  vector -> cube, a staged KV tile in `Tile` is ready for QK;
              set once per dequantised tile, waited once per QK.
      flag 4  cube -> vector, scores are ready; flags 4 and 8 alternate by
              loop step so the next request's QK can already be in flight
              while this request's scores are being read.
      flag 8  the odd-step counterpart of flag 4.
      flag 5  vector -> cube, the softmax numerators in `Weights` are ready;
              it sits between the store and the PV because `Weights` is the
              global buffer the two cores share.
      flag 6  cube -> vector, the PV accumulator in `Partial` is ready to be
              normalised and written out.
    """
    pid = tl.program_id(0)
    sub = tle.dsa.ascend.sub_vec_id()
    head = sub * (64 // 2) + tl.arange(0, 64 // 2)
    inner = tl.arange(0, 128)
    col = tl.arange(0, 512)
    storage = tle.dsa.alloc(
        (DEQUANT_ROWS, SCRATCH_COLS),
        dtype=tl.bfloat16,
        mem_addr_space=tle.dsa.ascend.UB,
    )
    scratch = tle.dsa.to_tensor(storage)
    count = tl.load(Count + pid)
    win_count = tl.load(WinCount + pid)
    _window_dequant_tile(
        pid,
        0,
        pid * 2,
        sub,
        cache_words,
        Pages,
        Slots,
        count,
        win_count,
        Tile,
        128,
        DEQUANT_ROWS,
        MAP_STRIDE,
        DEQUANT_OP,
        scratch,
    )
    if pid + PROGRAMS < BATCH:
        next_request = pid + PROGRAMS
        _window_dequant_tile(
            next_request,
            0,
            pid * 2 + 1,
            sub,
            cache_words,
            Pages,
            Slots,
            tl.load(Count + next_request),
            tl.load(WinCount + next_request),
            Tile,
            128,
            DEQUANT_ROWS,
            MAP_STRIDE,
            DEQUANT_OP,
            scratch,
        )
    tle.dsa.ascend.sync_block_set("vector", "cube", 3)
    tle.dsa.ascend.sync_block_wait("vector", "cube", 3)
    q = tl.make_block_ptr(
        Q + pid * 64 * 512,
        (64, 512),
        (512, 1),
        (0, 0),
        (64, 512),
        (1, 0),
    )
    kv = tl.make_block_ptr(
        Tile + pid * 2 * 128 * 512,
        (128, 512),
        (512, 1),
        (0, 0),
        (128, 512),
        (1, 0),
    )
    scores = tl.make_block_ptr(
        Scores + pid * 2 * 64 * 128,
        (64, 128),
        (128, 1),
        (0, 0),
        (64, 128),
        (1, 0),
    )
    tle.dsa.ascend.raw(
        _MMA_TILE_OP,
        q,
        kv,
        _TILE_TOKENS,
        0,
        0,
        0,
        scores,
    )
    tle.dsa.ascend.sync_block_set("cube", "vector", 4)
    if pid + PROGRAMS < BATCH:
        tle.dsa.ascend.sync_block_set("vector", "cube", 3)
    for request in range(pid, BATCH, PROGRAMS):
        step = (request - pid) // PROGRAMS
        if step % 2 == 0:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 4)
        else:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 8)
        if request + PROGRAMS < BATCH:
            next_request = request + PROGRAMS
            tle.dsa.ascend.sync_block_wait("vector", "cube", 3)
            q_next = tl.make_block_ptr(
                Q + next_request * 64 * 512,
                (64, 512),
                (512, 1),
                (0, 0),
                (64, 512),
                (1, 0),
            )
            kv_next = tl.make_block_ptr(
                Tile + (pid * 2 + (step + 1) % 2) * 128 * 512,
                (128, 512),
                (512, 1),
                (0, 0),
                (128, 512),
                (1, 0),
            )
            scores_next = tl.make_block_ptr(
                Scores + (pid * 2 + (step + 1) % 2) * 64 * 128,
                (64, 128),
                (128, 1),
                (0, 0),
                (64, 128),
                (1, 0),
            )
            tle.dsa.ascend.raw(
                _MMA_TILE_OP,
                q_next,
                kv_next,
                _TILE_TOKENS,
                0,
                (step + 1).to(tl.int64),
                0,
                scores_next,
            )
            if step % 2 == 0:
                tle.dsa.ascend.sync_block_set("cube", "vector", 8)
            else:
                tle.dsa.ascend.sync_block_set("cube", "vector", 4)
        count = tl.load(Count + request)
        score = tl.load(
            Scores
            + (pid * 2 + step % 2) * 64 * 128
            + head[:, None] * 128
            + inner[None, :]
        )
        weight = tl.where(inner[None, :] < count, tl.exp(score * (512.0**-0.5)), 0.0)
        total = tl.sum(weight, 1)
        tl.store(
            Weights + pid * 64 * 128 + head[:, None] * 128 + inner[None, :],
            weight.to(tl.bfloat16),
        )
        tle.dsa.ascend.sync_block_set("vector", "cube", 5)
        tle.dsa.ascend.sync_block_wait("vector", "cube", 5)
        p = tl.make_block_ptr(
            Weights + pid * 64 * 128,
            (64, 128),
            (128, 1),
            (0, 0),
            (64, 128),
            (1, 0),
        )
        kv = tl.make_block_ptr(
            Tile + (pid * 2 + step % 2) * 128 * 512,
            (128, 512),
            (512, 1),
            (0, 0),
            (128, 512),
            (1, 0),
        )
        partial = tl.make_block_ptr(
            Partial + pid * 64 * 512,
            (64, 512),
            (512, 1),
            (0, 0),
            (64, 512),
            (1, 0),
        )
        tle.dsa.ascend.raw(
            _MMA_TILE_OP,
            p,
            kv,
            _TILE_TOKENS,
            1,
            step.to(tl.int64),
            (request + PROGRAMS >= BATCH).to(tl.int64),
            partial,
        )
        if request + 2 * PROGRAMS < BATCH:
            following = request + 2 * PROGRAMS
            _window_dequant_tile(
                following,
                0,
                pid * 2 + (step + 2) % 2,
                sub,
                cache_words,
                Pages,
                Slots,
                tl.load(Count + following),
                tl.load(WinCount + following),
                Tile,
                128,
                DEQUANT_ROWS,
                MAP_STRIDE,
                DEQUANT_OP,
                scratch,
            )
        if request + 2 * PROGRAMS < BATCH:
            tle.dsa.ascend.sync_block_set("vector", "cube", 3)
        tle.dsa.ascend.sync_block_set("cube", "vector", 6)
        tle.dsa.ascend.sync_block_wait("cube", "vector", 6)
        value = tl.load(Partial + pid * 64 * 512 + head[:, None] * 512 + col[None, :])
        sink = tl.load(Sink + head)
        tl.store(
            Out + request * 64 * 512 + head[:, None] * 512 + col[None, :],
            (value / (total + sink)[:, None]).to(tl.bfloat16),
        )


def _launch_window(q, win_cache, win_block_table, sinks=None, schedule=None):
    """Window pipeline host side. Returns [B, 64, 512] BF16."""
    pages, slots = device_map(schedule, win_block_table)
    out = torch.empty_like(q)
    _sparse_attn_sharedkv_fp8_window_kernel[(schedule.programs,)](
        q,
        win_cache.view(torch.int64),
        pages,
        slots,
        schedule.count,
        schedule.win_count,
        schedule.sink_exp(sinks),
        schedule.staged,
        schedule.scores,
        schedule.weights,
        schedule.partial,
        out,
        BATCH=schedule.batch,
        PROGRAMS=schedule.programs,
        MAP_STRIDE=pages.stride(0),
        DEQUANT_ROWS=DEQUANT_ROWS,
        SCRATCH_COLS=DEQUANT_SCRATCH_COLS,
        DEQUANT_OP=schedule.dequant_op,
        enable_auto_bind_sub_block=False,
        disable_auto_cv_work_space_manage=True,
        disable_auto_inject_block_sync=True,
    )
    return out


# --- schedule on device ---
#
# The host build below is the reference; these kernels reproduce its arrays
# bit for bit so the attention kernels see the same bytes either way. The
# point is to stop reading device values back to the host, which the graph-safe
# BF16 entry also avoids -- it builds its plans in Triton kernels.
@triton.jit
def _schedule_counts_kernel(
    Seqused,
    WinCount,
    WinStart,
    Count,
    Tiles,
    WINDOW: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
    BATCH: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
):
    """Per-request token counts, window columns first then compressed."""
    rows = tl.arange(0, BATCH_BLOCK)
    valid = rows < BATCH
    used = tl.load(Seqused + rows, mask=valid, other=0)
    win = tl.minimum(used, WINDOW)
    cmp = tl.zeros((BATCH_BLOCK,), tl.int32)
    if CMP_RATIO > 1:
        cmp = tl.minimum(used // CMP_RATIO, TOPK)
    count = win + cmp
    tl.store(WinCount + rows, win, mask=valid)
    tl.store(WinStart + rows, used - win, mask=valid)
    tl.store(Count + rows, count, mask=valid)
    # A request with no tokens still owns one tile, as the host build does.
    tl.store(Tiles + rows, tl.maximum(1, tl.cdiv(count, TILE)), mask=valid)


@triton.jit
def _schedule_maps_kernel(
    WinStart,
    OriTable,
    CmpTable,
    CmpIndices,
    Pages,
    Slots,
    ORI_COLUMNS: tl.constexpr,
    CMP_COLUMNS: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    HAS_INDICES: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LOG2_PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """(page, slot) of every column: index list -> block table -> pool.

    The two levels are resolved here because a doubly indirect access faults
    the memory engine. PAGE is a power of two, so the floor division and the
    positive modulo the host uses are a shift and a mask -- which is what a -1
    padding column has to resolve to as well, hence the clamp on both sides.
    """
    request = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = columns < MAP_STRIDE
    window = columns < WINDOW

    position = tl.load(WinStart + request) + columns
    if HAS_INDICES:
        picked = tl.load(
            CmpIndices + request * TOPK + (columns - WINDOW),
            mask=inside & (columns >= WINDOW),
            other=0,
        )
    else:
        picked = columns - WINDOW
    logical = tl.where(window, position, picked)

    ori = tl.minimum(tl.maximum(position >> LOG2_PAGE_SIZE, 0), ORI_COLUMNS - 1)
    cmp = tl.minimum(tl.maximum(picked >> LOG2_PAGE_SIZE, 0), CMP_COLUMNS - 1)
    ori_page = tl.load(
        OriTable + request * ORI_COLUMNS + ori, mask=inside & window, other=0
    )
    cmp_page = tl.load(
        CmpTable + request * CMP_COLUMNS + cmp,
        mask=inside & (columns >= WINDOW),
        other=0,
    )
    tl.store(
        Pages + request * MAP_STRIDE + columns,
        tl.where(window, ori_page, cmp_page),
        mask=inside,
    )
    tl.store(
        Slots + request * MAP_STRIDE + columns,
        logical & (PAGE_SIZE - 1),
        mask=inside,
    )


@triton.jit
def _schedule_links_kernel(
    Pages,
    Slots,
    Reach,
    COLUMNS: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Where each column's physically contiguous run ends, as a column index.

    A run shares a page and has consecutive slots, so `Reach` starts as the
    column itself where the run ends there and as a value past every column
    otherwise; the doubling pass below then turns it into the end of the run.
    A run never crosses the window-to-compressed boundary.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = offsets < COLUMNS
    column = offsets % MAP_STRIDE
    page = tl.load(Pages + offsets, mask=inside, other=-1)
    slot = tl.load(Slots + offsets, mask=inside, other=0)
    last = (column + 1 == MAP_STRIDE) | (column == WINDOW - 1)
    page_after = tl.load(Pages + offsets + 1, mask=inside & ~last, other=-2)
    slot_after = tl.load(Slots + offsets + 1, mask=inside & ~last, other=-2)
    ends = last | (page != page_after) | (slot + 1 != slot_after)
    tl.store(Reach + offsets, tl.where(ends, offsets, COLUMNS), mask=inside)


@triton.jit
def _schedule_reach_kernel(
    Source,
    Target,
    STEP: tl.constexpr,
    COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One doubling step of the run's end: fold in the value STEP columns on."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = offsets < COLUMNS
    here = tl.load(Source + offsets, mask=inside, other=COLUMNS)
    ahead = tl.load(
        Source + offsets + STEP, mask=offsets + STEP < COLUMNS, other=COLUMNS
    )
    tl.store(Target + offsets, tl.minimum(here, ahead), mask=inside)


@triton.jit
def _schedule_run_slots_kernel(
    Slots,
    Reach,
    RunSlots,
    COLUMNS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold the run length into each slot's high bits, as the host build did."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = offsets < COLUMNS
    run = tl.load(Reach + offsets, mask=inside, other=0) - offsets + 1
    tl.store(
        RunSlots + offsets,
        tl.load(Slots + offsets, mask=inside, other=0) + run * PAGE_SIZE,
        mask=inside,
    )


@triton.jit
def _schedule_assign_kernel(
    Tiles,
    Program,
    Offset,
    Loads,
    PROGRAMS: tl.constexpr,
    BATCH: tl.constexpr,
):
    """Give each request's tiles to the least loaded program, in request order.

    One program runs this: the state is the PROGRAMS running loads, and the
    choice for one request depends on every earlier one. What it writes is per
    request -- the program that takes it and the offset its tiles start at --
    so filling the task table afterwards is parallel again. Ties go to the
    lowest program, as the host build's `min` does.
    """
    lanes = tl.arange(0, PROGRAMS)
    loads = tl.zeros((PROGRAMS,), tl.int32)
    for request in range(BATCH):
        count = tl.load(Tiles + request)
        least = tl.min(loads)
        chosen = tl.min(tl.where(loads == least, lanes, PROGRAMS))
        tl.store(Program + request, chosen)
        tl.store(Offset + request, least)
        loads = tl.where(lanes == chosen, loads + count, loads)
    tl.store(Loads + lanes, loads)


@triton.jit
def _schedule_tasks_kernel(
    Tiles,
    Program,
    Offset,
    TaskRequests,
    TaskTiles,
    TaskFlags,
    TASK_STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
):
    """Write one request's tiles into the queue its program owns.

    The flags mark where a work unit starts and ends, which is what lets the
    cube keep the accumulator resident across a request's tiles.
    """
    request = tl.program_id(0)
    tiles = tl.arange(0, TILE_BLOCK)
    count = tl.load(Tiles + request)
    mine = tiles < count
    slot = tl.load(Program + request) * TASK_STRIDE + tl.load(Offset + request) + tiles
    flags = tl.where(tiles == 0, 2, 0) + tl.where(tiles + 1 == count, 4, 0)
    tl.store(TaskRequests + slot, request, mask=mine)
    tl.store(TaskTiles + slot, tiles, mask=mine)
    tl.store(TaskFlags + slot, flags, mask=mine)


@triton.jit
def _schedule_clear_kernel(Rows, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    """Zero one row of a task table."""
    column = tl.arange(0, BLOCK)
    tl.store(Rows + tl.program_id(0) * STRIDE + column, 0, mask=column < STRIDE)


def _zero_table(rows, columns, device):
    """A zero-filled int32 table; a kernel writes only a program's own load."""
    table = torch.empty((rows, columns), dtype=torch.int32, device=device)
    _schedule_clear_kernel[(rows,)](
        table, STRIDE=columns, BLOCK=triton.next_power_of_2(columns)
    )
    return table


@triton.jit
def _sink_exp_kernel(Sinks, Out, HEAD_BLOCK: tl.constexpr):
    """exp(sink) per head, which the kernels add to the softmax denominator."""
    head = tl.arange(0, HEAD_BLOCK)
    tl.store(Out + head, tl.exp(tl.load(Sinks + head)))


# Columns one program resolves in the map pass, and in the run passes, which
# are one element per column and so can take a wider block.
_SCHEDULE_MAP_BLOCK = 128
_SCHEDULE_RUN_BLOCK = 1024


@triton.jit
def _schedule_partitions_kernel(
    Tiles,
    TaskRequests,
    TaskTiles,
    TaskFlags,
    TaskPartitions,
    TaskCounts,
    PartitionIds,
    PartitionCounts,
    PROGRAMS: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
    PARTITION_STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
):
    """Split K: every tile is its own partition, dealt round robin.

    One tile per work unit means the request pipelines' greedy assignment
    degenerates to dealing in order, so no running loads are needed. Each
    partition is both the first and the last tile of its unit, hence flag 6.
    """
    tiles = tl.arange(0, TILE_BLOCK)
    taken = 0
    for request in range(BATCH):
        count = tl.load(Tiles + request)
        mine = tiles < count
        tl.store(PartitionCounts + request, count)
        partition = taken + tl.cumsum(mine.to(tl.int32), 0) - 1
        slot = (partition % PROGRAMS) * TASK_STRIDE + partition // PROGRAMS
        tl.store(TaskRequests + slot, request, mask=mine)
        tl.store(TaskTiles + slot, tiles, mask=mine)
        tl.store(TaskFlags + slot, 6, mask=mine)
        tl.store(TaskPartitions + slot, partition, mask=mine)
        tl.store(
            PartitionIds + request * PARTITION_STRIDE + tiles, partition, mask=mine
        )
        taken += count
    lanes = tl.arange(0, PROGRAMS)
    tl.store(
        TaskCounts + lanes,
        taken // PROGRAMS + tl.where(lanes < taken % PROGRAMS, 1, 0),
    )


def task_stride(batch, max_tokens, programs):
    """Queue length to allocate, from the shape alone.

    The host build takes the longest queue the greedy assignment produced,
    which means reading it back. Greedy list scheduling cannot leave a queue
    longer than the even share plus one whole work unit, and a work unit is at
    most `tiles` long, so this bound needs no device value. The kernels walk
    `task_counts`, so the columns past a program's own load are never read.
    """
    tiles = max(1, -(-max_tokens // BLOCK_N))
    return -(-batch * tiles // programs) + tiles


def device_tasks(schedule, tiles):
    """The task table and the per-program loads, built by the two kernels."""
    device = tiles.device
    batch, programs = schedule.batch, schedule.programs
    stride = task_stride(batch, schedule.max_tokens, programs)
    program = torch.empty((batch,), dtype=torch.int32, device=device)
    offset = torch.empty((batch,), dtype=torch.int32, device=device)
    counts = torch.empty((programs,), dtype=torch.int32, device=device)
    arrays = [_zero_table(programs, stride, device) for _ in range(3)]
    _schedule_assign_kernel[(1,)](
        tiles, program, offset, counts, PROGRAMS=programs, BATCH=batch
    )
    _schedule_tasks_kernel[(batch,)](
        tiles,
        program,
        offset,
        *arrays,
        TASK_STRIDE=stride,
        BATCH=batch,
        TILE_BLOCK=triton.next_power_of_2(max(1, -(-schedule.max_tokens // BLOCK_N))),
    )
    return arrays, counts, stride


def device_map(
    schedule,
    ori_block_table,
    cmp_block_table=None,
    cmp_sparse_indices=None,
):
    """`maps()` and `run_slots()` as kernel launches, reading nothing back.

    Returns pages and slots, with each position's physical run length folded
    into its slot's high bits. Cached on the schedule: it belongs to a decode
    step, whose tables and index list do not change, and the attention kernels
    run once per layer of that step -- and once per graph replay, which must
    not rebuild it.
    """
    if schedule._map is not None:
        return schedule._map
    device = ori_block_table.device
    batch, stride = schedule.batch, schedule.max_tokens
    pages = torch.empty((batch, stride), dtype=torch.int32, device=device)
    slots = torch.empty((batch, stride), dtype=torch.int32, device=device)
    indices = cmp_sparse_indices
    if indices is not None:
        indices = indices.reshape(batch, -1)
        if not indices.is_contiguous():
            raise ValueError("cmp_sparse_indices must be contiguous.")
    # The unused table and index arguments still need a pointer; the masks keep
    # every lane that would read them switched off.
    _schedule_maps_kernel[(batch, triton.cdiv(stride, _SCHEDULE_MAP_BLOCK))](
        schedule.win_start,
        ori_block_table,
        ori_block_table if cmp_block_table is None else cmp_block_table,
        ori_block_table if indices is None else indices,
        pages,
        slots,
        ORI_COLUMNS=ori_block_table.shape[1],
        CMP_COLUMNS=1 if cmp_block_table is None else cmp_block_table.shape[1],
        MAP_STRIDE=stride,
        WINDOW=schedule.window,
        HAS_INDICES=indices is not None,
        TOPK=schedule.topk,
        PAGE_SIZE=PAGE,
        LOG2_PAGE_SIZE=PAGE.bit_length() - 1,
        BLOCK=_SCHEDULE_MAP_BLOCK,
    )
    # Where each run ends, then seven doubling steps to carry that end back
    # over the PAGE columns a run can span. Every pass is one contiguous
    # element per column, which the [BLOCK, PAGE] tile it replaces was not.
    columns = batch * stride
    grid = (triton.cdiv(columns, _SCHEDULE_RUN_BLOCK),)
    reach = torch.empty((columns,), dtype=torch.int32, device=device)
    folded = torch.empty((columns,), dtype=torch.int32, device=device)
    run_slots = torch.empty((batch, stride), dtype=torch.int32, device=device)
    _schedule_links_kernel[grid](
        pages,
        slots,
        reach,
        COLUMNS=columns,
        MAP_STRIDE=stride,
        WINDOW=schedule.window,
        BLOCK=_SCHEDULE_RUN_BLOCK,
    )
    step = 1
    while step < PAGE:
        _schedule_reach_kernel[grid](
            reach, folded, STEP=step, COLUMNS=columns, BLOCK=_SCHEDULE_RUN_BLOCK
        )
        reach, folded = folded, reach
        step *= 2
    _schedule_run_slots_kernel[grid](
        slots,
        reach,
        run_slots,
        COLUMNS=columns,
        PAGE_SIZE=PAGE,
        BLOCK=_SCHEDULE_RUN_BLOCK,
    )
    schedule._map = (pages, run_slots)
    return schedule._map


# --- request queues ---
@triton.jit
def _score_weight(score, valid_columns):
    """exp(score * softmax_scale), zero past the valid columns.

    Every pipeline uses this form. A device function that fused the three
    steps over the score's unified buffer was 2.6% faster on the dense shape
    and level everywhere else (final7 against final6), which is not worth a
    registration."""
    return tl.where(
        tl.arange(0, 128)[None, :] < valid_columns,
        tl.exp(score * 512.0**-0.5),
        0.0,
    )


SUB_HEADS = HEADS // 2  # one sub-vector core per half of the heads
TAIL_COLS = HEAD_DIM - PERSISTENT_COLS  # columns the cube returns via Partial
STAGES = 3  # depth of the staged KV ring
SCORE_BANKS = 2  # consecutive tasks alternate between the two score banks
SOFTMAX_SCALE = HEAD_DIM**-0.5

# Task flag bits. The host sets bits 1 and 2; the kernels add bit 0.
FLAG_LAST_TASK = 1  # last task of this program's queue
FLAG_FIRST_TILE = 2  # first tile of the work unit, so the cube starts fresh
FLAG_LAST_TILE = 4  # last tile of the work unit, so the result is complete


@triton.jit
def _full_block(base, ROWS: tl.constexpr, COLS: tl.constexpr):
    """A block pointer over one whole contiguous ROWS x COLS tile at `base`."""
    return tl.make_block_ptr(
        base, (ROWS, COLS), (COLS, 1), (0, 0), (ROWS, COLS), (1, 0)
    )


@triton.jit
def _cache_pool_block(base, ROWS: tl.constexpr):
    """A ROWS x 1 window onto an int64-viewed cache pool.

    The dequant device function walks the pool itself from the page and slot
    maps, so the pointer only carries the base address; the 1 << 40 extent
    stands in for "as large as the pool happens to be".
    """
    return tl.make_block_ptr(base, (1 << 40, 1), (1, 1), (0, 0), (ROWS, 1), (1, 0))


@triton.jit
def _staged_slot_shared(pid, task):
    """Ring slot when one ring is shared by every task of a program."""
    return pid * 3 + task % 3


@triton.jit
def _staged_slot_per_request(request, tile):
    """Ring slot when every request owns a ring of its own."""
    return request * 3 + tile % 3


@triton.jit
def _stage_kv_tile(
    request,
    tile,
    slot,
    sub,
    WindowCacheI64,
    CompressedCacheI64,
    Pages,
    Slots,
    Count,
    WinCount,
    staged,
    scratch,
    MAP_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
):
    """Dequantise one BLOCK_N-row KV tile into `slot` of the staged ring.

    The two sub-vector cores split every piece, so a core writes DEQUANT_ROWS
    rows at a time starting at `sub * DEQUANT_ROWS`.
    """
    count = tl.load(Count + request)
    win_count = tl.load(WinCount + request)
    pool_w = _cache_pool_block(WindowCacheI64, DEQUANT_ROWS)
    pool_c = _cache_pool_block(CompressedCacheI64, DEQUANT_ROWS)
    pp = _full_block(Pages + request * MAP_STRIDE, MAP_STRIDE, 1)
    sp = _full_block(Slots + request * MAP_STRIDE, MAP_STRIDE, 1)
    for piece in range(0, 128 // (2 * DEQUANT_ROWS)):
        first = tile * 128 + piece * 2 * DEQUANT_ROWS + sub * DEQUANT_ROWS
        target = slot * 128 + piece * 2 * DEQUANT_ROWS + sub * DEQUANT_ROWS
        dst = _full_block(staged + target * 512, DEQUANT_ROWS, 512)
        tle.dsa.ascend.raw(
            DEQUANT_OP,
            pool_w,
            pool_c,
            pp,
            sp,
            first,
            win_count,
            count,
            _PAGE_STRIDE,
            _TOKEN_RECORD,
            _SCALE_BLOCK,
            _PAGE_SLOTS,
            _WINDOW_EXTENT,
            dst,
            scratch,
        )


@triton.jit
def _stage_task_shared_ring(
    pid,
    task,
    sub,
    WindowCacheI64,
    CompressedCacheI64,
    Pages,
    Slots,
    Count,
    WinCount,
    TaskRequests,
    TaskTiles,
    staged,
    scratch,
    MAP_STRIDE: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
):
    """Stage the KV tile of one queued task into the program's shared ring."""
    request = tl.load(TaskRequests + pid * TASK_STRIDE + task)
    tile = tl.load(TaskTiles + pid * TASK_STRIDE + task)
    _stage_kv_tile(
        request,
        tile,
        _staged_slot_shared(pid, task),
        sub,
        WindowCacheI64,
        CompressedCacheI64,
        Pages,
        Slots,
        Count,
        WinCount,
        staged,
        scratch,
        MAP_STRIDE,
        DEQUANT_ROWS,
        DEQUANT_OP,
    )


@triton.jit
def _stage_task_per_request(
    pid,
    task,
    sub,
    WindowCacheI64,
    CompressedCacheI64,
    Pages,
    Slots,
    Count,
    WinCount,
    TaskRequests,
    TaskTiles,
    staged,
    scratch,
    MAP_STRIDE: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
):
    """Stage the KV tile of one queued task into the request's own ring."""
    request = tl.load(TaskRequests + pid * TASK_STRIDE + task)
    tile = tl.load(TaskTiles + pid * TASK_STRIDE + task)
    _stage_kv_tile(
        request,
        tile,
        _staged_slot_per_request(request, tile),
        sub,
        WindowCacheI64,
        CompressedCacheI64,
        Pages,
        Slots,
        Count,
        WinCount,
        staged,
        scratch,
        MAP_STRIDE,
        DEQUANT_ROWS,
        DEQUANT_OP,
    )


@triton.jit
def _prologue_shared_ring(
    pid,
    sub,
    tasks,
    Q,
    WindowCacheI64,
    CompressedCacheI64,
    Pages,
    Slots,
    Count,
    WinCount,
    TaskRequests,
    TaskTiles,
    staged,
    Scores,
    scratch,
    MAP_STRIDE: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
    MMA_OP: tl.constexpr,
):
    """Fill the ring and issue the QK of task 0, for a program-owned ring.

    Tasks 0 and 1 are staged before the first QK and task 2 after it, so the
    cube and the vector core overlap from the first tile onwards. Task 0 is
    always the first tile of its work unit, hence the FLAG_FIRST_TILE literal.
    """
    first_request = tl.load(TaskRequests + pid * TASK_STRIDE)
    first_tile = tl.load(TaskTiles + pid * TASK_STRIDE)
    _stage_kv_tile(
        first_request,
        first_tile,
        _staged_slot_shared(pid, 0),
        sub,
        WindowCacheI64,
        CompressedCacheI64,
        Pages,
        Slots,
        Count,
        WinCount,
        staged,
        scratch,
        MAP_STRIDE,
        DEQUANT_ROWS,
        DEQUANT_OP,
    )
    if tasks > 1:
        _stage_task_shared_ring(
            pid,
            1,
            sub,
            WindowCacheI64,
            CompressedCacheI64,
            Pages,
            Slots,
            Count,
            WinCount,
            TaskRequests,
            TaskTiles,
            staged,
            scratch,
            MAP_STRIDE,
            TASK_STRIDE,
            DEQUANT_ROWS,
            DEQUANT_OP,
        )
    # The cube may not read the ring until both sub-vector cores have written it.
    tle.dsa.ascend.sync_block_set("vector", "cube", 3)
    tle.dsa.ascend.sync_block_wait("vector", "cube", 3)
    q = _full_block(Q + first_request * 64 * 512, 64, 512)
    first_kv = _full_block(staged + _staged_slot_shared(pid, 0) * 128 * 512, 128, 512)
    scores = _full_block(Scores + pid * 2 * 64 * 128, 64, 128)
    tle.dsa.ascend.raw(
        MMA_OP,
        q,
        first_kv,
        first_kv,
        _TILE_TOKENS,
        0,
        0,
        2 | (tasks == 1).to(tl.int64),
        _RESIDENT_COLS,
        scores,
    )
    tle.dsa.ascend.sync_block_set("cube", "vector", 4)
    if tasks > 2:
        _stage_task_shared_ring(
            pid,
            2,
            sub,
            WindowCacheI64,
            CompressedCacheI64,
            Pages,
            Slots,
            Count,
            WinCount,
            TaskRequests,
            TaskTiles,
            staged,
            scratch,
            MAP_STRIDE,
            TASK_STRIDE,
            DEQUANT_ROWS,
            DEQUANT_OP,
        )


@triton.jit
def _prologue_per_request(
    pid,
    sub,
    tasks,
    Q,
    WindowCacheI64,
    CompressedCacheI64,
    Pages,
    Slots,
    Count,
    WinCount,
    TaskRequests,
    TaskTiles,
    staged,
    Scores,
    scratch,
    MAP_STRIDE: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
    MMA_OP: tl.constexpr,
):
    """Fill the ring and issue the QK of task 0, for per-request rings.

    Same order as `_prologue_shared_ring`; only the ring slot and the score bank
    are indexed by request rather than by program.
    """
    first_request = tl.load(TaskRequests + pid * TASK_STRIDE)
    first_tile = tl.load(TaskTiles + pid * TASK_STRIDE)
    _stage_kv_tile(
        first_request,
        first_tile,
        _staged_slot_per_request(first_request, first_tile),
        sub,
        WindowCacheI64,
        CompressedCacheI64,
        Pages,
        Slots,
        Count,
        WinCount,
        staged,
        scratch,
        MAP_STRIDE,
        DEQUANT_ROWS,
        DEQUANT_OP,
    )
    if tasks > 1:
        _stage_task_per_request(
            pid,
            1,
            sub,
            WindowCacheI64,
            CompressedCacheI64,
            Pages,
            Slots,
            Count,
            WinCount,
            TaskRequests,
            TaskTiles,
            staged,
            scratch,
            MAP_STRIDE,
            TASK_STRIDE,
            DEQUANT_ROWS,
            DEQUANT_OP,
        )
    # The cube may not read the ring until both sub-vector cores have written it.
    tle.dsa.ascend.sync_block_set("vector", "cube", 3)
    tle.dsa.ascend.sync_block_wait("vector", "cube", 3)
    q = _full_block(Q + first_request * 64 * 512, 64, 512)
    first_kv = _full_block(
        staged + _staged_slot_per_request(first_request, first_tile) * 128 * 512,
        128,
        512,
    )
    # Clamped so a single-task program still hands the cube a mapped address.
    next_task = tl.minimum(1, tasks - 1)
    next_request = tl.load(TaskRequests + pid * TASK_STRIDE + next_task)
    next_tile = tl.load(TaskTiles + pid * TASK_STRIDE + next_task)
    next_kv = _full_block(
        staged + _staged_slot_per_request(next_request, next_tile) * 128 * 512,
        128,
        512,
    )
    scores = _full_block(Scores + first_request * 2 * 64 * 128, 64, 128)
    tle.dsa.ascend.raw(
        MMA_OP,
        q,
        first_kv,
        next_kv,
        _TILE_TOKENS,
        0,
        0,
        8 | 2 | (tasks == 1).to(tl.int64),
        _RESIDENT_COLS,
        scores,
    )
    tle.dsa.ascend.sync_block_set("cube", "vector", 4)
    if tasks > 2:
        _stage_task_per_request(
            pid,
            2,
            sub,
            WindowCacheI64,
            CompressedCacheI64,
            Pages,
            Slots,
            Count,
            WinCount,
            TaskRequests,
            TaskTiles,
            staged,
            scratch,
            MAP_STRIDE,
            TASK_STRIDE,
            DEQUANT_ROWS,
            DEQUANT_OP,
        )


@triton.jit
def _fence_partial():
    """Release Partial for the cube after this core has read it.

    The variable shape reads the persistent columns on the vector core, where
    the cube has just written them; this fences the cube's next fixpipe into
    the same buffer against that read.
    """
    tle.dsa.ascend.sync_block_set(
        "vector",
        "cube",
        7,
        tle.dsa.ascend.PIPE.PIPE_MTE2,
        tle.dsa.ascend.PIPE.PIPE_FIX,
    )
    tle.dsa.ascend.sync_block_wait(
        "vector",
        "cube",
        7,
        tle.dsa.ascend.PIPE.PIPE_MTE2,
        tle.dsa.ascend.PIPE.PIPE_FIX,
    )


@triton.jit
def _store_output(Partial, Out, row, request, head, tail, acc, denom):
    """Normalise one core's half of a request's output and store it.

    The PERSISTENT_COLS columns come from Partial, the TAIL_COLS tail from the
    accumulator the vector core has been carrying: two stores rather than one,
    because the halves live in different buffers. Only the variable shape uses
    it: dense and split-K have no unified buffer left for the Triton form, and
    on pull it cost cmp4 2.5% (final7), so those three keep the device
    function.
    """
    cols = tl.arange(0, 384)
    nope = tl.load(Partial + row * 64 * 512 + head[:, None] * 512 + cols[None, :])
    tl.store(
        Out + request * 64 * 512 + head[:, None] * 512 + cols[None, :],
        (nope / denom[:, None]).to(tl.bfloat16),
    )
    tl.store(
        Out + request * 64 * 512 + head[:, None] * 512 + 384 + tail[None, :],
        (acc / denom[:, None]).to(tl.bfloat16),
    )


@triton.jit
def _store_output_chunked(Partial, Out, row, request, head, tail, acc, denom):
    """`_store_output` without materialising the persistent half at once.

    `_store_output` keeps a [32, 384] FP32 tile live, 48 KiB, plus
    its BF16 result. Beside the decoder's scratch -- which bishengir treats as
    live for the whole kernel -- that is what makes dense and split-K fail to
    compile with `Ub overflow detected`. Storing the persistent columns in
    128-wide chunks keeps one [32, 128] FP32 tile live instead, the width the
    tail already uses. The arithmetic is the same division, so
    the result is the one the variable shape already produces.
    """
    for chunk in range(3):  # 384 persistent columns in 128-wide chunks
        cols = chunk * 128 + tail
        value = tl.load(Partial + row * 64 * 512 + head[:, None] * 512 + cols[None, :])
        tl.store(
            Out + request * 64 * 512 + head[:, None] * 512 + cols[None, :],
            (value / denom[:, None]).to(tl.bfloat16),
        )
    tl.store(
        Out + request * 64 * 512 + head[:, None] * 512 + 384 + tail[None, :],
        (acc / denom[:, None]).to(tl.bfloat16),
    )


@triton.jit
def _requests_finish_partial(
    Partial,
    Out,
    Sink,
    TaskRequests,
    TaskFlags,
    pid,
    task,
    head,
    tail,
    acc,
    total,
    TASK_STRIDE: tl.constexpr,
):
    """Fold one task's PV tail into `acc`; on its last tile also write the output.

    Only the TAIL_COLS columns the cube hands back live in `acc`. The device
    function reads the PERSISTENT_COLS columns straight out of `Partial`,
    divides both halves by `denom` and casts them to BF16 on the way to `Out`,
    all inside the decoder's dead scratch -- which is why this shape keeps it
    while the others use the Triton epilogue.
    """
    request = tl.load(TaskRequests + pid * TASK_STRIDE + task)
    acc += tl.load(Partial + pid * 64 * 512 + head[:, None] * 512 + 384 + tail[None, :])
    if tl.load(TaskFlags + pid * TASK_STRIDE + task) & 4:
        denom = total + tl.load(Sink + head)
        _store_output_chunked(Partial, Out, pid, request, head, tail, acc, denom)
    return acc


@triton.jit
def _sparse_attn_sharedkv_fp8_requests_dense_kernel(
    Q,
    WindowCacheI64,
    CompressedCacheI64,
    Pages,
    Slots,
    Count,
    WinCount,
    Sink,
    TaskRequests,
    TaskTiles,
    TaskFlags,
    TaskCounts,
    staged,
    Scores,
    Weights,
    Partial,
    Out,
    MAP_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    SCRATCH_COLS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
    MMA_OP: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    sub = tle.dsa.ascend.sub_vec_id()
    head = sub * 32 + tl.arange(0, 32)
    inner = tl.arange(0, 128)
    tail = tl.arange(0, 128)
    tasks = tl.load(TaskCounts + pid)
    scratch_ub = tle.dsa.alloc(
        (DEQUANT_ROWS, SCRATCH_COLS),
        dtype=tl.bfloat16,
        mem_addr_space=tle.dsa.ascend.UB,
    )
    scratch = tle.dsa.to_tensor(scratch_ub)
    _prologue_shared_ring(
        pid,
        sub,
        tasks,
        Q,
        WindowCacheI64,
        CompressedCacheI64,
        Pages,
        Slots,
        Count,
        WinCount,
        TaskRequests,
        TaskTiles,
        staged,
        Scores,
        scratch,
        MAP_STRIDE,
        TASK_STRIDE,
        DEQUANT_ROWS,
        DEQUANT_OP,
        MMA_OP,
    )
    acc = tl.zeros((32, 128), tl.float32)
    total = tl.zeros((32,), tl.float32)
    for task in range(0, tasks):
        request = tl.load(TaskRequests + pid * TASK_STRIDE + task)
        tile = tl.load(TaskTiles + pid * TASK_STRIDE + task)
        # The cube signals on event 4 after an even task and on event 8 after an
        # odd one, so a wait can never be satisfied by the set two tasks back.
        if task % 2 == 0:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 4)
        else:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 8)
        # A task's PV tail is collected one task later, which keeps the vector
        # core off the cube's critical path.
        if task > 0:
            acc = _requests_finish_partial(
                Partial,
                Out,
                Sink,
                TaskRequests,
                TaskFlags,
                pid,
                task - 1,
                head,
                tail,
                acc,
                total,
                TASK_STRIDE,
            )
        # Reset by multiplication rather than by branching on the first tile: the
        # branch becomes a memref select with unknown strides, which propagates
        # an eightfold last-axis alignment onto every other op in the kernel.
        keep = (tile != 0).to(tl.float32)
        acc = acc * keep
        total = total * keep
        if task + 1 < tasks:
            future_request = tl.load(TaskRequests + pid * TASK_STRIDE + task + 1)
            q_next = _full_block(Q + future_request * 64 * 512, 64, 512)
            future = _full_block(
                staged + _staged_slot_shared(pid, task + 1) * 128 * 512,
                128,
                512,
            )
            next_scores = _full_block(
                Scores + (pid * 2 + (task + 1) % 2) * 64 * 128,
                64,
                128,
            )
            flags = (task + 2 == tasks).to(tl.int64) | tl.load(
                TaskFlags + pid * TASK_STRIDE + task + 1
            ).to(tl.int64)
            tle.dsa.ascend.raw(
                MMA_OP,
                q_next,
                future,
                future,
                _TILE_TOKENS,
                0,
                (task + 1).to(tl.int64),
                flags,
                _RESIDENT_COLS,
                next_scores,
            )
        count = tl.load(Count + request)
        score = tl.load(
            Scores
            + (pid * 2 + task % 2) * 64 * 128
            + head[:, None] * 128
            + inner[None, :]
        )
        weight = _score_weight(score, count - tile * 128)
        total += tl.sum(weight, 1)
        tl.store(
            Weights + pid * 64 * 128 + head[:, None] * 128 + inner[None, :],
            weight.to(tl.bfloat16),
        )
        tle.dsa.ascend.sync_block_set("vector", "cube", 5)
        tle.dsa.ascend.sync_block_wait("vector", "cube", 5)
        p = _full_block(Weights + pid * 64 * 128, 64, 128)
        kv = _full_block(
            staged + _staged_slot_shared(pid, task) * 128 * 512,
            128,
            512,
        )
        partial = _full_block(Partial + pid * 64 * 512, 64, 512)
        flags = (task + 1 == tasks).to(tl.int64) | tl.load(
            TaskFlags + pid * TASK_STRIDE + task
        ).to(tl.int64)
        tle.dsa.ascend.raw(
            MMA_OP,
            p,
            kv,
            kv,
            _TILE_TOKENS,
            1,
            task.to(tl.int64),
            flags,
            _RESIDENT_COLS,
            partial,
        )
        if task % 2 == 0:
            tle.dsa.ascend.sync_block_set("cube", "vector", 8)
        else:
            tle.dsa.ascend.sync_block_set("cube", "vector", 4)
        # Refill the ring slot this task has just consumed, STAGES tasks ahead.
        if task + 3 < tasks:
            _stage_task_shared_ring(
                pid,
                task + 3,
                sub,
                WindowCacheI64,
                CompressedCacheI64,
                Pages,
                Slots,
                Count,
                WinCount,
                TaskRequests,
                TaskTiles,
                staged,
                scratch,
                MAP_STRIDE,
                TASK_STRIDE,
                DEQUANT_ROWS,
                DEQUANT_OP,
            )

    # Deferring by one task leaves the final task outstanding; its event id
    # follows the same even/odd alternation as the loop head.
    if tasks % 2 == 0:
        tle.dsa.ascend.sync_block_wait("cube", "vector", 4)
    else:
        tle.dsa.ascend.sync_block_wait("cube", "vector", 8)
    _requests_finish_partial(
        Partial,
        Out,
        Sink,
        TaskRequests,
        TaskFlags,
        pid,
        tasks - 1,
        head,
        tail,
        acc,
        total,
        TASK_STRIDE,
    )


@triton.jit
def _sparse_attn_sharedkv_fp8_requests_pull_kernel(
    Q,
    WindowCacheI64,
    CompressedCacheI64,
    Pages,
    Slots,
    Count,
    WinCount,
    Sink,
    TaskRequests,
    TaskTiles,
    TaskFlags,
    TaskCounts,
    staged,
    Scores,
    Weights,
    Partial,
    Out,
    MAP_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    SCRATCH_COLS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
    MMA_OP: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    sub = tle.dsa.ascend.sub_vec_id()
    head = sub * 32 + tl.arange(0, 32)
    inner = tl.arange(0, 128)
    tail = tl.arange(0, 128)
    tasks = tl.load(TaskCounts + pid)
    scratch_ub = tle.dsa.alloc(
        (DEQUANT_ROWS, SCRATCH_COLS),
        dtype=tl.bfloat16,
        mem_addr_space=tle.dsa.ascend.UB,
    )
    scratch = tle.dsa.to_tensor(scratch_ub)
    _prologue_shared_ring(
        pid,
        sub,
        tasks,
        Q,
        WindowCacheI64,
        CompressedCacheI64,
        Pages,
        Slots,
        Count,
        WinCount,
        TaskRequests,
        TaskTiles,
        staged,
        Scores,
        scratch,
        MAP_STRIDE,
        TASK_STRIDE,
        DEQUANT_ROWS,
        DEQUANT_OP,
        MMA_OP,
    )
    acc = tl.zeros((32, 128), tl.float32)
    total = tl.zeros((32,), tl.float32)
    for task in range(0, tasks):
        request = tl.load(TaskRequests + pid * TASK_STRIDE + task)
        tile = tl.load(TaskTiles + pid * TASK_STRIDE + task)
        # Reset by multiplication rather than by branching on the first tile: the
        # branch becomes a memref select with unknown strides, which propagates
        # an eightfold last-axis alignment onto every other op in the kernel.
        keep = (tile != 0).to(tl.float32)
        acc = acc * keep
        total = total * keep
        # The cube signals on event 4 after an even task and on event 8 after an
        # odd one, so a wait can never be satisfied by the set two tasks back.
        if task % 2 == 0:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 4)
        else:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 8)
        if task + 1 < tasks:
            future_request = tl.load(TaskRequests + pid * TASK_STRIDE + task + 1)
            q_next = _full_block(Q + future_request * 64 * 512, 64, 512)
            future = _full_block(
                staged + _staged_slot_shared(pid, task + 1) * 128 * 512,
                128,
                512,
            )
            next_scores = _full_block(
                Scores + (pid * 2 + (task + 1) % 2) * 64 * 128,
                64,
                128,
            )
            flags = (task + 2 == tasks).to(tl.int64) | tl.load(
                TaskFlags + pid * TASK_STRIDE + task + 1
            ).to(tl.int64)
            tle.dsa.ascend.raw(
                MMA_OP,
                q_next,
                future,
                future,
                _TILE_TOKENS,
                0,
                (task + 1).to(tl.int64),
                flags,
                _RESIDENT_COLS,
                next_scores,
            )
            if task % 2 == 0:
                tle.dsa.ascend.sync_block_set("cube", "vector", 8)
            else:
                tle.dsa.ascend.sync_block_set("cube", "vector", 4)
        count = tl.load(Count + request)
        score = tl.load(
            Scores
            + (pid * 2 + task % 2) * 64 * 128
            + head[:, None] * 128
            + inner[None, :]
        )
        weight = _score_weight(score, count - tile * 128)
        total += tl.sum(weight, 1)
        tl.store(
            Weights + pid * 64 * 128 + head[:, None] * 128 + inner[None, :],
            weight.to(tl.bfloat16),
        )
        tle.dsa.ascend.sync_block_set("vector", "cube", 5)
        tle.dsa.ascend.sync_block_wait("vector", "cube", 5)
        p = _full_block(Weights + pid * 64 * 128, 64, 128)
        kv = _full_block(
            staged + _staged_slot_shared(pid, task) * 128 * 512,
            128,
            512,
        )
        partial = _full_block(Partial + pid * 64 * 512, 64, 512)
        flags = (task + 1 == tasks).to(tl.int64) | tl.load(
            TaskFlags + pid * TASK_STRIDE + task
        ).to(tl.int64)
        tle.dsa.ascend.raw(
            MMA_OP,
            p,
            kv,
            kv,
            _TILE_TOKENS,
            1,
            task.to(tl.int64),
            flags,
            _RESIDENT_COLS,
            partial,
        )
        # Refill the ring slot this task has just consumed, STAGES tasks ahead.
        if task + 3 < tasks:
            _stage_task_shared_ring(
                pid,
                task + 3,
                sub,
                WindowCacheI64,
                CompressedCacheI64,
                Pages,
                Slots,
                Count,
                WinCount,
                TaskRequests,
                TaskTiles,
                staged,
                scratch,
                MAP_STRIDE,
                TASK_STRIDE,
                DEQUANT_ROWS,
                DEQUANT_OP,
            )
        tle.dsa.ascend.sync_block_set("cube", "vector", 6)
        tle.dsa.ascend.sync_block_wait("cube", "vector", 6)
        acc += tl.load(
            Partial + pid * 64 * 512 + head[:, None] * 512 + 384 + tail[None, :]
        )
        if tl.load(TaskFlags + pid * TASK_STRIDE + task) & 4:
            denom = total + tl.load(Sink + head)
            _store_output_chunked(Partial, Out, pid, request, head, tail, acc, denom)


@triton.jit
def _sparse_attn_sharedkv_fp8_requests_variable_kernel(
    Q,
    WindowCacheI64,
    CompressedCacheI64,
    Pages,
    Slots,
    Count,
    WinCount,
    Sink,
    TaskRequests,
    TaskTiles,
    TaskFlags,
    TaskCounts,
    staged,
    Scores,
    Weights,
    Partial,
    Out,
    MAP_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    SCRATCH_COLS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
    MMA_OP: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    sub = tle.dsa.ascend.sub_vec_id()
    head = sub * 32 + tl.arange(0, 32)
    inner = tl.arange(0, 128)
    tail = tl.arange(0, 128)
    tasks = tl.load(TaskCounts + pid)
    scratch_ub = tle.dsa.alloc(
        (DEQUANT_ROWS, SCRATCH_COLS),
        dtype=tl.bfloat16,
        mem_addr_space=tle.dsa.ascend.UB,
    )
    scratch = tle.dsa.to_tensor(scratch_ub)
    _prologue_per_request(
        pid,
        sub,
        tasks,
        Q,
        WindowCacheI64,
        CompressedCacheI64,
        Pages,
        Slots,
        Count,
        WinCount,
        TaskRequests,
        TaskTiles,
        staged,
        Scores,
        scratch,
        MAP_STRIDE,
        TASK_STRIDE,
        DEQUANT_ROWS,
        DEQUANT_OP,
        MMA_OP,
    )
    acc = tl.zeros((32, 128), tl.float32)
    total = tl.zeros((32,), tl.float32)
    for task in range(0, tasks):
        request = tl.load(TaskRequests + pid * TASK_STRIDE + task)
        tile = tl.load(TaskTiles + pid * TASK_STRIDE + task)
        # Reset by multiplication rather than by branching on the first tile: the
        # branch becomes a memref select with unknown strides, which propagates
        # an eightfold last-axis alignment onto every other op in the kernel.
        keep = (tile != 0).to(tl.float32)
        acc = acc * keep
        total = total * keep
        # This MMA op does not pull the current KV tile itself, so the staging
        # stores issued at the tail of the previous task are fenced here.
        tle.dsa.ascend.sync_block_set("vector", "cube", 3)
        tle.dsa.ascend.sync_block_wait("vector", "cube", 3)
        # The cube signals on event 4 after an even task and on event 8 after an
        # odd one, so a wait can never be satisfied by the set two tasks back.
        if task % 2 == 0:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 4)
        else:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 8)
        if task + 1 < tasks:
            future_request = tl.load(TaskRequests + pid * TASK_STRIDE + task + 1)
            future_tile = tl.load(TaskTiles + pid * TASK_STRIDE + task + 1)
            # Clamped so the cube always receives a mapped address to prefetch.
            following_task = tl.minimum(task + 2, tasks - 1)
            following_request = tl.load(
                TaskRequests + pid * TASK_STRIDE + following_task
            )
            following_tile = tl.load(TaskTiles + pid * TASK_STRIDE + following_task)
            q_next = _full_block(Q + future_request * 64 * 512, 64, 512)
            future = _full_block(
                staged
                + _staged_slot_per_request(future_request, future_tile) * 128 * 512,
                128,
                512,
            )
            following = _full_block(
                staged
                + _staged_slot_per_request(following_request, following_tile)
                * 128
                * 512,
                128,
                512,
            )
            next_scores = _full_block(
                Scores + (future_request * 2 + (task + 1) % 2) * 64 * 128,
                64,
                128,
            )
            # Bit 3: this pipeline prefetches the next tile's KV, which the
            # others load where they consume it.
            flags = (
                8
                | (task + 2 == tasks).to(tl.int64)
                | tl.load(TaskFlags + pid * TASK_STRIDE + task + 1).to(tl.int64)
            )
            tle.dsa.ascend.raw(
                MMA_OP,
                q_next,
                future,
                following,
                _TILE_TOKENS,
                0,
                (task + 1).to(tl.int64),
                flags,
                _RESIDENT_COLS,
                next_scores,
            )
            if task % 2 == 0:
                tle.dsa.ascend.sync_block_set("cube", "vector", 8)
            else:
                tle.dsa.ascend.sync_block_set("cube", "vector", 4)
        count = tl.load(Count + request)
        score = tl.load(
            Scores
            + (request * 2 + task % 2) * 64 * 128
            + head[:, None] * 128
            + inner[None, :]
        )
        weight = tl.where(
            tile * 128 + inner[None, :] < count,
            tl.exp(score * 512.0**-0.5),
            0.0,
        )
        total += tl.sum(weight, 1)
        tl.store(
            Weights + request * 64 * 128 + head[:, None] * 128 + inner[None, :],
            weight.to(tl.bfloat16),
        )
        tle.dsa.ascend.sync_block_set("vector", "cube", 5)
        tle.dsa.ascend.sync_block_wait("vector", "cube", 5)
        p = _full_block(Weights + request * 64 * 128, 64, 128)
        kv = _full_block(
            staged + _staged_slot_per_request(request, tile) * 128 * 512,
            128,
            512,
        )
        partial = _full_block(Partial + request * 64 * 512, 64, 512)
        flags = (
            8
            | (task + 1 == tasks).to(tl.int64)
            | tl.load(TaskFlags + pid * TASK_STRIDE + task).to(tl.int64)
        )
        tle.dsa.ascend.raw(
            MMA_OP,
            p,
            kv,
            kv,
            _TILE_TOKENS,
            1,
            task.to(tl.int64),
            flags,
            _RESIDENT_COLS,
            partial,
        )
        # Refill the ring slot this task has just consumed, STAGES tasks ahead.
        if task + 3 < tasks:
            _stage_task_per_request(
                pid,
                task + 3,
                sub,
                WindowCacheI64,
                CompressedCacheI64,
                Pages,
                Slots,
                Count,
                WinCount,
                TaskRequests,
                TaskTiles,
                staged,
                scratch,
                MAP_STRIDE,
                TASK_STRIDE,
                DEQUANT_ROWS,
                DEQUANT_OP,
            )
        tle.dsa.ascend.sync_block_set("cube", "vector", 6)
        tle.dsa.ascend.sync_block_wait("cube", "vector", 6)
        acc += tl.load(
            Partial + request * 64 * 512 + head[:, None] * 512 + 384 + tail[None, :]
        )
        if tl.load(TaskFlags + pid * TASK_STRIDE + task) & 4:
            denom = total + tl.load(Sink + head)
            _store_output(Partial, Out, request, request, head, tail, acc, denom)
        _fence_partial()


def _launch_requests_dense(q, out, ori_kv, cache_cmp, pages, slots, sink_exp, schedule):
    """Launch the dense kernel over `schedule.programs` programs."""
    return _sparse_attn_sharedkv_fp8_requests_dense_kernel[(schedule.programs,)](
        q,
        ori_kv.view(torch.int64),
        cache_cmp.view(torch.int64),
        pages,
        slots,
        schedule.count,
        schedule.win_count,
        sink_exp,
        schedule.task_requests,
        schedule.task_tiles,
        schedule.task_flags,
        schedule.task_counts,
        schedule.staged,
        schedule.scores,
        schedule.weights,
        schedule.partial,
        out,
        MAP_STRIDE=pages.stride(0),
        DEQUANT_ROWS=DEQUANT_ROWS,
        SCRATCH_COLS=DEQUANT_SCRATCH_COLS,
        DEQUANT_OP=schedule.dequant_op,
        MMA_OP=schedule.mma_op,
        TASK_STRIDE=schedule.task_stride,
        enable_auto_bind_sub_block=False,
        disable_auto_cv_work_space_manage=True,
        disable_auto_inject_block_sync=True,
    )


def _launch_requests_pull(q, out, ori_kv, cache_cmp, pages, slots, sink_exp, schedule):
    """Launch the pull kernel over `schedule.programs` programs."""
    return _sparse_attn_sharedkv_fp8_requests_pull_kernel[(schedule.programs,)](
        q,
        ori_kv.view(torch.int64),
        cache_cmp.view(torch.int64),
        pages,
        slots,
        schedule.count,
        schedule.win_count,
        sink_exp,
        schedule.task_requests,
        schedule.task_tiles,
        schedule.task_flags,
        schedule.task_counts,
        schedule.staged,
        schedule.scores,
        schedule.weights,
        schedule.partial,
        out,
        MAP_STRIDE=pages.stride(0),
        DEQUANT_ROWS=DEQUANT_ROWS,
        SCRATCH_COLS=DEQUANT_SCRATCH_COLS,
        DEQUANT_OP=schedule.dequant_op,
        MMA_OP=schedule.mma_op,
        TASK_STRIDE=schedule.task_stride,
        enable_auto_bind_sub_block=False,
        disable_auto_cv_work_space_manage=True,
        disable_auto_inject_block_sync=True,
    )


def _launch_requests_variable(
    q, out, ori_kv, cache_cmp, pages, slots, sink_exp, schedule
):
    """Launch the variable kernel over `schedule.programs` programs."""
    return _sparse_attn_sharedkv_fp8_requests_variable_kernel[(schedule.programs,)](
        q,
        ori_kv.view(torch.int64),
        cache_cmp.view(torch.int64),
        pages,
        slots,
        schedule.count,
        schedule.win_count,
        sink_exp,
        schedule.task_requests,
        schedule.task_tiles,
        schedule.task_flags,
        schedule.task_counts,
        schedule.staged,
        schedule.scores,
        schedule.weights,
        schedule.partial,
        out,
        MAP_STRIDE=pages.stride(0),
        DEQUANT_ROWS=DEQUANT_ROWS,
        SCRATCH_COLS=DEQUANT_SCRATCH_COLS,
        DEQUANT_OP=schedule.dequant_op,
        MMA_OP=schedule.mma_op,
        TASK_STRIDE=schedule.task_stride,
        enable_auto_bind_sub_block=False,
        disable_auto_cv_work_space_manage=True,
        disable_auto_inject_block_sync=True,
    )


# --- split K ---
@triton.jit
def _partition_dequant_tile(
    pid,
    task,
    sub,
    window_kv_words,
    compressed_kv_words,
    pages,
    slots,
    counts,
    window_counts,
    task_requests,
    task_tiles,
    staged,
    scratch,
    MAP_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
):
    """Dequantise the task's KV tile from both FP8 pools into its staged slot."""
    request = tl.load(task_requests + pid * TASK_STRIDE + task)
    tile = tl.load(task_tiles + pid * TASK_STRIDE + task)
    count = tl.load(counts + request)
    window_count = tl.load(window_counts + request)
    pool_window = tl.make_block_ptr(
        window_kv_words, (1 << 40, 1), (1, 1), (0, 0), (DEQUANT_ROWS, 1), (1, 0)
    )
    pool_compressed = tl.make_block_ptr(
        compressed_kv_words, (1 << 40, 1), (1, 1), (0, 0), (DEQUANT_ROWS, 1), (1, 0)
    )
    page_map = tl.make_block_ptr(
        pages + request * MAP_STRIDE,
        (MAP_STRIDE, 1),
        (1, 1),
        (0, 0),
        (MAP_STRIDE, 1),
        (1, 0),
    )
    slot_map = tl.make_block_ptr(
        slots + request * MAP_STRIDE,
        (MAP_STRIDE, 1),
        (1, 1),
        (0, 0),
        (MAP_STRIDE, 1),
        (1, 0),
    )
    for piece in range(0, 128 // (2 * DEQUANT_ROWS)):
        first = tile * 128 + piece * 2 * DEQUANT_ROWS + sub * DEQUANT_ROWS
        target = (
            _staged_slot_shared(pid, task) * 128
            + piece * 2 * DEQUANT_ROWS
            + sub * DEQUANT_ROWS
        )
        dst = tl.make_block_ptr(
            staged + target * 512,
            (DEQUANT_ROWS, 512),
            (512, 1),
            (0, 0),
            (DEQUANT_ROWS, 512),
            (1, 0),
        )
        tle.dsa.ascend.raw(
            DEQUANT_OP,
            pool_window,
            pool_compressed,
            page_map,
            slot_map,
            first,
            window_count,
            count,
            _PAGE_STRIDE,
            _TOKEN_RECORD,
            _SCALE_BLOCK,
            _PAGE_SLOTS,
            _WINDOW_EXTENT,
            dst,
            scratch,
        )


@triton.jit
def _partition_finish_partial(
    partials,
    out,
    task_flags,
    task_partitions,
    partition_totals,
    pid,
    task,
    head,
    tail,
    acc,
    total,
    TASK_STRIDE: tl.constexpr,
):
    """Pull in the trailing output columns and, on a last tile, store the partition.

    The MMA op keeps the leading PERSISTENT_COLS columns in its own accumulator
    and leaves the rest in the partial buffer, so the vector side adds them
    here. A partition is stored unnormalised, with its row sums; the merge
    kernel divides once it has every partition of the request.

    The persistent columns move in 128-wide chunks for the same reason
    `_store_output_chunked` does: a [32, 384] FP32 tile does not fit beside
    the decoder's scratch, which bishengir keeps live.
    """
    acc += tl.load(
        partials + pid * 64 * 512 + head[:, None] * 512 + 384 + tail[None, :]
    )
    if tl.load(task_flags + pid * TASK_STRIDE + task) & 4:
        partition = tl.load(task_partitions + pid * TASK_STRIDE + task)
        for chunk in range(3):  # 384 persistent columns in 128-wide chunks
            cols = chunk * 128 + tail
            tl.store(
                out + partition * 64 * 512 + head[:, None] * 512 + cols[None, :],
                tl.load(
                    partials + pid * 64 * 512 + head[:, None] * 512 + cols[None, :]
                ),
            )
        tl.store(
            out + partition * 64 * 512 + head[:, None] * 512 + 384 + tail[None, :],
            acc,
        )
        tl.store(partition_totals + partition * 64 + head, total)
    return acc


@triton.jit
def _sparse_attn_sharedkv_fp8_partition_attention_kernel(
    q,
    window_kv_words,
    compressed_kv_words,
    pages,
    slots,
    counts,
    window_counts,
    task_requests,
    task_tiles,
    task_flags,
    task_counts,
    task_partitions,
    partition_totals,
    staged,
    scores,
    weights,
    partials,
    out,
    MAP_STRIDE: tl.constexpr,
    DEQUANT_ROWS: tl.constexpr,
    SCRATCH_COLS: tl.constexpr,
    DEQUANT_OP: tl.constexpr,
    MMA_OP: tl.constexpr,
    TASK_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    sub = tle.dsa.ascend.sub_vec_id()
    head = sub * (64 // 2) + tl.arange(0, 64 // 2)
    inner = tl.arange(0, 128)
    tail = tl.arange(0, 512 - 384)
    tasks = tl.load(task_counts + pid)
    scratch_buffer = tle.dsa.alloc(
        (DEQUANT_ROWS, SCRATCH_COLS),
        dtype=tl.bfloat16,
        mem_addr_space=tle.dsa.ascend.UB,
    )
    scratch = tle.dsa.to_tensor(scratch_buffer)
    _partition_dequant_tile(
        pid,
        0,
        sub,
        window_kv_words,
        compressed_kv_words,
        pages,
        slots,
        counts,
        window_counts,
        task_requests,
        task_tiles,
        staged,
        scratch,
        MAP_STRIDE,
        DEQUANT_ROWS,
        DEQUANT_OP,
        TASK_STRIDE,
    )
    if tasks > 1:
        _partition_dequant_tile(
            pid,
            1,
            sub,
            window_kv_words,
            compressed_kv_words,
            pages,
            slots,
            counts,
            window_counts,
            task_requests,
            task_tiles,
            staged,
            scratch,
            MAP_STRIDE,
            DEQUANT_ROWS,
            DEQUANT_OP,
            TASK_STRIDE,
        )
    tle.dsa.ascend.sync_block_set("vector", "cube", 3)
    tle.dsa.ascend.sync_block_wait("vector", "cube", 3)
    first_request = tl.load(task_requests + pid * TASK_STRIDE)
    q_block = tl.make_block_ptr(
        q + first_request * 64 * 512,
        (64, 512),
        (512, 1),
        (0, 0),
        (64, 512),
        (1, 0),
    )
    first_kv = tl.make_block_ptr(
        staged + _staged_slot_shared(pid, 0) * 128 * 512,
        (128, 512),
        (512, 1),
        (0, 0),
        (128, 512),
        (1, 0),
    )
    first_scores = tl.make_block_ptr(
        scores + pid * 2 * 64 * 128,
        (64, 128),
        (128, 1),
        (0, 0),
        (64, 128),
        (1, 0),
    )
    tle.dsa.ascend.raw(
        MMA_OP,
        q_block,
        first_kv,
        first_kv,
        _TILE_TOKENS,
        0,
        0,
        2 | (tasks == 1).to(tl.int64),
        _RESIDENT_COLS,
        first_scores,
    )
    tle.dsa.ascend.sync_block_set("cube", "vector", 4)
    if tasks > 2:
        _partition_dequant_tile(
            pid,
            2,
            sub,
            window_kv_words,
            compressed_kv_words,
            pages,
            slots,
            counts,
            window_counts,
            task_requests,
            task_tiles,
            staged,
            scratch,
            MAP_STRIDE,
            DEQUANT_ROWS,
            DEQUANT_OP,
            TASK_STRIDE,
        )
    acc = tl.zeros((64 // 2, 512 - 384), tl.float32)
    total = tl.zeros((64 // 2,), tl.float32)
    for task in range(0, tasks):
        request = tl.load(task_requests + pid * TASK_STRIDE + task)
        tile = tl.load(task_tiles + pid * TASK_STRIDE + task)
        if task % 2 == 0:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 4)
        else:
            tle.dsa.ascend.sync_block_wait("cube", "vector", 8)
        if task > 0:
            acc = _partition_finish_partial(
                partials,
                out,
                task_flags,
                task_partitions,
                partition_totals,
                pid,
                task - 1,
                head,
                tail,
                acc,
                total,
                TASK_STRIDE,
            )
        # The accumulator restarts on the first tile of a partition, which flag
        # bit1 marks; tile != 0 would not do, because a partition can start in
        # the middle of a request. Scaling by a 0/1 factor instead of branching
        # avoids a memref select with unknown strides, which otherwise
        # propagates an eightfold last-axis alignment to all ops.
        keep = ((tl.load(task_flags + pid * TASK_STRIDE + task) & 2) == 0).to(
            tl.float32
        )
        acc = acc * keep
        total = total * keep
        if task + 1 < tasks:
            future_request = tl.load(task_requests + pid * TASK_STRIDE + task + 1)
            q_next = tl.make_block_ptr(
                q + future_request * 64 * 512,
                (64, 512),
                (512, 1),
                (0, 0),
                (64, 512),
                (1, 0),
            )
            future = tl.make_block_ptr(
                staged + _staged_slot_shared(pid, task + 1) * 128 * 512,
                (128, 512),
                (512, 1),
                (0, 0),
                (128, 512),
                (1, 0),
            )
            next_scores = tl.make_block_ptr(
                scores + (pid * 2 + (task + 1) % 2) * 64 * 128,
                (64, 128),
                (128, 1),
                (0, 0),
                (64, 128),
                (1, 0),
            )
            flags = (task + 2 == tasks).to(tl.int64) | tl.load(
                task_flags + pid * TASK_STRIDE + task + 1
            ).to(tl.int64)
            tle.dsa.ascend.raw(
                MMA_OP,
                q_next,
                future,
                future,
                _TILE_TOKENS,
                0,
                (task + 1).to(tl.int64),
                flags,
                _RESIDENT_COLS,
                next_scores,
            )
        count = tl.load(counts + request)
        score = tl.load(
            scores
            + (pid * 2 + task % 2) * 64 * 128
            + head[:, None] * 128
            + inner[None, :]
        )
        weight = _score_weight(score, count - tile * 128)
        total += tl.sum(weight, 1)
        tl.store(
            weights + pid * 64 * 128 + head[:, None] * 128 + inner[None, :],
            weight.to(tl.bfloat16),
        )
        tle.dsa.ascend.sync_block_set("vector", "cube", 5)
        tle.dsa.ascend.sync_block_wait("vector", "cube", 5)
        p = tl.make_block_ptr(
            weights + pid * 64 * 128,
            (64, 128),
            (128, 1),
            (0, 0),
            (64, 128),
            (1, 0),
        )
        kv = tl.make_block_ptr(
            staged + _staged_slot_shared(pid, task) * 128 * 512,
            (128, 512),
            (512, 1),
            (0, 0),
            (128, 512),
            (1, 0),
        )
        partial = tl.make_block_ptr(
            partials + pid * 64 * 512,
            (64, 512),
            (512, 1),
            (0, 0),
            (64, 512),
            (1, 0),
        )
        flags = (task + 1 == tasks).to(tl.int64) | tl.load(
            task_flags + pid * TASK_STRIDE + task
        ).to(tl.int64)
        tle.dsa.ascend.raw(
            MMA_OP,
            p,
            kv,
            kv,
            _TILE_TOKENS,
            1,
            task.to(tl.int64),
            flags,
            _RESIDENT_COLS,
            partial,
        )
        if task % 2 == 0:
            tle.dsa.ascend.sync_block_set("cube", "vector", 8)
        else:
            tle.dsa.ascend.sync_block_set("cube", "vector", 4)
        if task + 3 < tasks:
            _partition_dequant_tile(
                pid,
                task + 3,
                sub,
                window_kv_words,
                compressed_kv_words,
                pages,
                slots,
                counts,
                window_counts,
                task_requests,
                task_tiles,
                staged,
                scratch,
                MAP_STRIDE,
                DEQUANT_ROWS,
                DEQUANT_OP,
                TASK_STRIDE,
            )

    if tasks % 2 == 0:
        tle.dsa.ascend.sync_block_wait("cube", "vector", 4)
    else:
        tle.dsa.ascend.sync_block_wait("cube", "vector", 8)
    acc = _partition_finish_partial(
        partials,
        out,
        task_flags,
        task_partitions,
        partition_totals,
        pid,
        tasks - 1,
        head,
        tail,
        acc,
        total,
        TASK_STRIDE,
    )


@triton.jit
def _sparse_attn_sharedkv_fp8_merge_partitions_kernel(
    partition_values,
    partition_totals,
    partition_ids,
    partition_counts,
    sink_exp,
    out,
    PARTITION_STRIDE: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1) * 8 + tl.arange(0, 8)
    column = tl.arange(0, 512)
    acc = tl.zeros((8, 512), tl.float32)
    total = tl.zeros((8,), tl.float32)
    count = tl.load(partition_counts + request)
    for index in range(0, count):
        partition = tl.load(partition_ids + request * PARTITION_STRIDE + index)
        acc += tl.load(
            partition_values
            + partition * 64 * 512
            + head[:, None] * 512
            + column[None, :]
        )
        total += tl.load(partition_totals + partition * 64 + head)
    denominator = total + tl.load(sink_exp + head)
    # A head can see no valid token at all once the sink exponent is zero, and
    # dividing by that sum would store NaN into the output; the guard keeps it 0.
    value = tl.where(denominator[:, None] > 0, acc / denominator[:, None], 0)
    tl.store(
        out + request * 64 * 512 + head[:, None] * 512 + column[None, :],
        value.to(tl.bfloat16),
    )


def _launch_partition(q, window_kv, compressed_kv, pages, slots, sink_exp, schedule):
    """Run both split-K launches and return the merged BF16 output."""
    schedule.kernel = _sparse_attn_sharedkv_fp8_partition_attention_kernel[
        (schedule.programs,)
    ](
        q,
        window_kv.view(torch.int64),
        compressed_kv.view(torch.int64),
        pages,
        slots,
        schedule.count,
        schedule.win_count,
        schedule.task_requests,
        schedule.task_tiles,
        schedule.task_flags,
        schedule.task_counts,
        schedule.task_partitions,
        schedule.partition_totals,
        schedule.staged,
        schedule.scores,
        schedule.weights,
        schedule.partial,
        schedule.partition_values,
        MAP_STRIDE=pages.stride(0),
        DEQUANT_ROWS=DEQUANT_ROWS,
        SCRATCH_COLS=DEQUANT_SCRATCH_COLS,
        DEQUANT_OP=DEQUANT_OP,
        MMA_OP=MMA_RESIDENT_OP,
        TASK_STRIDE=schedule.task_stride,
        enable_auto_bind_sub_block=False,
        disable_auto_cv_work_space_manage=True,
        disable_auto_inject_block_sync=True,
    )
    merged = torch.empty_like(q)
    schedule.merge_kernel = _sparse_attn_sharedkv_fp8_merge_partitions_kernel[
        (schedule.batch, HEADS // 8)
    ](
        schedule.partition_values,
        schedule.partition_totals,
        schedule.partition_ids,
        schedule.partition_counts,
        sink_exp,
        merged,
        PARTITION_STRIDE=schedule.partition_stride,
        enable_auto_bind_sub_block=False,
        disable_auto_cv_work_space_manage=True,
    )
    return merged


# Batches from which the dense pipeline serves a compressed shape, and below
# which a request's tiles become split-K partitions.
DENSE_MIN_BATCH = 32
SPLIT_K_BATCHES = 4


class _Schedule:
    """Per-call metadata, reused by every kernel of the chosen branch."""

    def __init__(self, batch, seqused_kv, cmp_ratio, window, topk, selected):
        device = seqused_kv.device
        self.batch = batch
        self.cmp_ratio = cmp_ratio
        self.window = window
        self.topk = topk
        self.max_tokens = window + (topk if cmp_ratio > 1 else 0)
        # Window tokens first, then compressed, so one count and one staged
        # buffer drive the attention pass. Clamping the compressed prefix is
        # the selection when an index list names the tokens; without one the
        # extent comes from the caller's own block table and max_model_len, and
        # both bound the prefix, so the clamp cannot cut one short unless the
        # caller contradicts itself.
        fields = [
            torch.empty((batch,), dtype=torch.int32, device=device) for _ in range(4)
        ]
        _schedule_counts_kernel[(1,)](
            seqused_kv,
            *fields,
            WINDOW=window,
            CMP_RATIO=cmp_ratio,
            TOPK=topk,
            TILE=BLOCK_N,
            BATCH=batch,
            BATCH_BLOCK=triton.next_power_of_2(batch),
        )
        self.win_count, self.win_start, self.count, self.tiles = fields
        self.device = device
        self._map = None
        self._sink_source = None
        self._sink_exp = None

    def sink_exp(self, sinks):
        """exp(sink) per head; the kernels add it to the softmax denominator."""
        if self._sink_exp is None or self._sink_source is not sinks:
            self._sink_source = sinks
            self._sink_exp = torch.empty_like(sinks)
            _sink_exp_kernel[(1,)](sinks, self._sink_exp, HEAD_BLOCK=HEADS)
        return self._sink_exp


def _window_programs(batch):
    """Programs for the window pipeline: whole waves of at most PROGRAM_LIMIT."""
    waves = -(-batch // PROGRAM_LIMIT)
    return min(batch, -(-batch // waves))


def _setup_window(schedule):
    programs = _window_programs(schedule.batch)
    schedule.programs = programs
    schedule.staged = torch.empty(
        (programs * 2, BLOCK_N, HEAD_DIM), dtype=torch.bfloat16, device=schedule.device
    )
    schedule.scores = torch.empty(
        (programs * 2, HEADS, BLOCK_N), dtype=torch.float32, device=schedule.device
    )
    schedule.weights = torch.empty(
        (programs, HEADS, BLOCK_N), dtype=torch.bfloat16, device=schedule.device
    )
    schedule.partial = torch.empty(
        (programs, HEADS, HEAD_DIM), dtype=torch.float32, device=schedule.device
    )
    schedule.dequant_op = DEQUANT_OP
    schedule.branch = "window"


def _setup_requests(schedule, pipeline, mma_op):
    """The request pipelines: one program walks the tiles queued on it."""
    programs = min(schedule.batch, PROGRAM_LIMIT)
    schedule.programs = programs
    # The variable pipeline indexes its buffers by request, the others by
    # program, which is what lets a program reuse one set across its queue.
    per_program = pipeline != "variable"
    owners = programs if per_program else schedule.batch
    schedule.staged = torch.empty(
        (owners, 3 * BLOCK_N, HEAD_DIM), dtype=torch.bfloat16, device=schedule.device
    )
    schedule.scores = torch.empty(
        (owners, 2, HEADS, BLOCK_N), dtype=torch.float32, device=schedule.device
    )
    schedule.weights = torch.empty(
        (owners, HEADS, BLOCK_N), dtype=torch.bfloat16, device=schedule.device
    )
    schedule.partial = torch.empty(
        (owners, HEADS, HEAD_DIM), dtype=torch.float32, device=schedule.device
    )
    # One work unit per request: its tiles share the resident accumulator, so
    # they stay on one program and in order.
    arrays, counts, stride = device_tasks(schedule, schedule.tiles)
    schedule.task_requests, schedule.task_tiles, schedule.task_flags = arrays
    schedule.task_counts = counts
    schedule.task_stride = stride
    schedule.dequant_op = DEQUANT_OP
    schedule.mma_op = mma_op
    schedule.per_program = per_program
    schedule.pipeline = pipeline
    schedule.branch = "requests"


def _setup_partition(schedule):
    """Split K: every tile becomes its own partition so small batches fill the cores."""
    # Partitions, programs and the per-request partition row all follow the
    # shape rather than the lengths: a request owns at most `tiles` tiles, so
    # the bound is exact enough to allocate from and a program whose share is
    # empty leaves its queue at zero and exits.
    tiles = max(1, -(-schedule.max_tokens // BLOCK_N))
    partitions = schedule.batch * tiles
    programs = min(PROGRAM_LIMIT, partitions)
    schedule.programs = programs
    stride = -(-partitions // programs)
    arrays = [_zero_table(programs, stride, schedule.device) for _ in range(4)]
    schedule.task_counts = torch.empty(
        (programs,), dtype=torch.int32, device=schedule.device
    )
    schedule.partition_stride = tiles
    schedule.partition_ids = _zero_table(schedule.batch, tiles, schedule.device)
    schedule.partition_counts = torch.empty(
        (schedule.batch,), dtype=torch.int32, device=schedule.device
    )
    _schedule_partitions_kernel[(1,)](
        schedule.tiles,
        *arrays,
        schedule.task_counts,
        schedule.partition_ids,
        schedule.partition_counts,
        PROGRAMS=programs,
        TASK_STRIDE=stride,
        PARTITION_STRIDE=tiles,
        BATCH=schedule.batch,
        TILE_BLOCK=triton.next_power_of_2(tiles),
    )
    (
        schedule.task_requests,
        schedule.task_tiles,
        schedule.task_flags,
        schedule.task_partitions,
    ) = arrays
    schedule.task_stride = stride
    schedule.staged = torch.empty(
        (programs, 3 * BLOCK_N, HEAD_DIM), dtype=torch.bfloat16, device=schedule.device
    )
    schedule.scores = torch.empty(
        (programs, 2, HEADS, BLOCK_N), dtype=torch.float32, device=schedule.device
    )
    schedule.weights = torch.empty(
        (programs, HEADS, BLOCK_N), dtype=torch.bfloat16, device=schedule.device
    )
    schedule.partial = torch.empty(
        (programs, HEADS, HEAD_DIM), dtype=torch.float32, device=schedule.device
    )
    schedule.partition_values = torch.empty(
        (partitions, HEADS, HEAD_DIM), dtype=torch.float32, device=schedule.device
    )
    schedule.partition_totals = torch.empty(
        (partitions, HEADS), dtype=torch.float32, device=schedule.device
    )
    schedule.dequant_op = DEQUANT_OP
    schedule.mma_op = MMA_RESIDENT_OP
    schedule.per_program = True
    schedule.branch = "partition"


def plan(
    batch,
    seqused_kv,
    cmp_ratio,
    window,
    cmp_sparse_indices=None,
    cmp_block_table=None,
    max_model_len=None,
):
    """Metadata and buffers for the shape this (batch, ratio) selects.

    Takes the arguments the entry has rather than a width, so the compressed
    extent -- an index list's own width, or the whole prefix's when there is no
    index list -- is derived in one place.

    Window-only decode is one tile per request, so the window pipeline serves
    every batch. Compressed decode walks a queue of tiles per program, with a
    kernel per pipelining shape; below batch four at ratio four a request does
    not have enough tiles to fill the cores, so its tiles become split-K
    partitions that a second kernel merges. The ratio is 1, 4 or 128, which is
    what the entry accepts.
    """
    selected = cmp_sparse_indices is not None
    topk = (
        cmp_sparse_indices.shape[-1]
        if selected
        else _compressed_extent(cmp_ratio, cmp_block_table, max_model_len)
    )
    schedule = _Schedule(batch, seqused_kv, cmp_ratio, window, topk, selected)
    if cmp_ratio <= 1:
        _setup_window(schedule)
    elif batch >= DENSE_MIN_BATCH:
        _setup_requests(
            schedule,
            "dense",
            MMA_RESIDENT_OP,
        )
    elif cmp_ratio == 128:
        _setup_requests(
            schedule,
            "variable",
            MMA_RESIDENT_OP,
        )
    elif batch < SPLIT_K_BATCHES:
        _setup_partition(schedule)
    else:
        _setup_requests(
            schedule,
            "pull",
            MMA_RESIDENT_OP,
        )
    return schedule


def _dispatch(
    q,
    ori_kv,
    ori_block_table,
    cmp_kv,
    cmp_block_table,
    cmp_sparse_indices,
    sinks,
    schedule,
):
    """Resolve the token maps, then run the branch the plan selected."""
    if schedule.branch == "window":
        # The window decoder reads plain slots: its tokens are consecutive
        # logical positions, so it derives a run from the page boundary.
        return _launch_window(q, ori_kv, ori_block_table, sinks, schedule)
    # The compressed decoders take the run length in the slot's high bits.
    pages, slots = device_map(
        schedule, ori_block_table, cmp_block_table, cmp_sparse_indices
    )
    sink_exp = schedule.sink_exp(sinks)
    compressed = cmp_kv if cmp_kv is not None else ori_kv
    if schedule.branch == "partition":
        return _launch_partition(
            q, ori_kv, compressed, pages, slots, sink_exp, schedule
        )
    out = torch.empty_like(q)
    launch = {
        "dense": _launch_requests_dense,
        "pull": _launch_requests_pull,
        "variable": _launch_requests_variable,
    }[schedule.pipeline]
    launch(q, out, ori_kv, compressed, pages, slots, sink_exp, schedule)
    return out


def is_packed_fp8_cache(cache) -> bool:
    """Whether a cache argument is the packed FP8 layout this path serves."""
    return (
        isinstance(cache, torch.Tensor)
        and cache.dtype == torch.uint8
        and cache.dim() == 2
        and cache.shape[1] == PAGE_STRIDE
    )


def _check(q, caches, tensors, cmp_ratio, mask_modes, window, layouts, lse):
    if q.device.type != "npu":
        raise NotImplementedError(
            "SparseAttnSharedKV FP8 is implemented only for Ascend NPU."
        )
    if lse:
        raise NotImplementedError("SparseAttnSharedKV FP8 does not return softmax LSE.")
    if layouts != ("TND", "PA_ND"):
        raise NotImplementedError("SparseAttnSharedKV FP8 requires TND Q and PA_ND KV.")
    if q.dtype != torch.bfloat16:
        raise TypeError("Q must be BF16 for the FP8 cache path.")
    if q.dim() != 3 or tuple(q.shape[1:]) != (HEADS, HEAD_DIM):
        raise ValueError(f"Q must have shape [T, {HEADS}, {HEAD_DIM}].")
    if mask_modes != (4, 3) or window != (PAGE - 1, 0):
        raise ValueError(
            "Only ori/cmp mask modes 4/3 and window [127, 0] are supported."
        )
    for name, cache in caches.items():
        if not is_packed_fp8_cache(cache):
            raise TypeError(
                f"{name} must be the packed FP8 cache: uint8 [pages, {PAGE_STRIDE}]."
            )
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as Q.")
        if not tensor.is_contiguous():
            raise NotImplementedError(
                f"{name} must be contiguous; no implicit copy is performed."
            )
    if cmp_ratio > 1 and "cmp_kv" not in caches:
        raise ValueError("A compress ratio above one needs cmp_kv.")
    if cmp_ratio <= 1 and "cmp_kv" in caches:
        raise ValueError("cmp_kv needs a compress ratio above one.")
    # The same two compressed modes the BF16 entry defines: an index list
    # selects tokens at ratio 4, its absence keeps the prefix at ratio 128.
    if tensors["cmp_sparse_indices"] is not None:
        if cmp_ratio != 4:
            raise ValueError("A sparse index list requires cmp_ratio 4.")
    elif cmp_ratio > 1 and cmp_ratio != 128:
        raise ValueError("Without a sparse index list, cmp_ratio must be 128.")


def sparse_attn_sharedkv_fp8(
    q: torch.Tensor,
    *,
    ori_kv: torch.Tensor,
    ori_block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    cmp_kv: Optional[torch.Tensor] = None,
    cmp_block_table: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    softmax_scale: float = 0,
    cmp_ratio: int = 0,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = PAGE - 1,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_ND",
    return_softmax_lse: bool = False,
    max_model_len: Optional[int] = None,
    schedule=None,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode over the packed FP8 KV cache; returns (attention, empty lse).

    One query token per request: `q` is `[batch, 64, 512]` and `cu_seqlens_q`
    is `arange(batch + 1)`. Prefill stays on the BF16 paths.

    `schedule` is this operator's tiling metadata, as `plan()` returns it. It
    depends on the request lengths and the index list, so it belongs to a
    decode step and can be reused by that step's layers; passing None builds it
    for this call.

    `unused` absorbs the arguments the BF16 entry accepts and this path derives
    itself (`metadata`, the cumulative KV lengths, `seqused_q`) or does not
    support (`ori_sparse_indices`, host lengths); a value that would change the
    result is rejected rather than ignored.
    """
    for name in ("ori_sparse_indices", "host_q_lens", "host_kv_lens"):
        if unused.get(name) is not None:
            raise NotImplementedError(f"{name} is not supported by the FP8 path.")
    caches = {"ori_kv": ori_kv}
    if cmp_kv is not None:
        caches["cmp_kv"] = cmp_kv
    _check(
        q,
        caches,
        {
            "q": q,
            "ori_block_table": ori_block_table,
            "cmp_block_table": cmp_block_table,
            "cmp_sparse_indices": cmp_sparse_indices,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_kv": seqused_kv,
            "sinks": sinks,
        },
        cmp_ratio,
        (ori_mask_mode, cmp_mask_mode),
        (ori_win_left, ori_win_right),
        (layout_q, layout_kv),
        return_softmax_lse,
    )
    batch = cu_seqlens_q.numel() - 1
    if q.shape[0] != batch:
        raise NotImplementedError(
            "The FP8 path serves decode only: one query token per request."
        )
    if sinks.dtype != torch.float32 or sinks.shape != (HEADS,):
        raise TypeError(f"sinks must be float32 [{HEADS}].")
    # The softmax scale is a literal operand of the score device function.
    if softmax_scale not in (0, None) and abs(softmax_scale - HEAD_DIM**-0.5) > 1e-9:
        raise NotImplementedError(
            f"The FP8 path implements softmax_scale = {HEAD_DIM}**-0.5 only."
        )
    if cmp_ratio > 1 and cmp_sparse_indices is not None:
        if cmp_sparse_indices.dtype != torch.int32:
            raise TypeError("cmp_sparse_indices must be int32.")
        if cmp_sparse_indices.shape[0] != batch:
            raise ValueError("cmp_sparse_indices must have one row per request.")
    if schedule is None:
        schedule = plan(
            batch,
            seqused_kv,
            cmp_ratio,
            ori_win_left + 1,
            cmp_sparse_indices,
            cmp_block_table,
            max_model_len,
        )
    out = _dispatch(
        q,
        ori_kv,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        sinks,
        schedule,
    )
    return out, torch.empty((0,), dtype=torch.float32, device=q.device)

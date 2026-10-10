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

import hashlib
import os
import statistics
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language import core as tl_core
from triton.tools.tensor_descriptor import TensorDescriptor

# ============================================================================
# Constants
# ============================================================================

# The scale block is 128 in N and K; 128 is also the e4m3 SQMMA K.
SCALE_BLOCK = tl.constexpr(128)
BLOCK_K = tl.constexpr(128)
PRODUCER_WARPS = tl.constexpr(4)
SHARED_BYTES_PER_MP = 196608
GEMV_BLOCK_N = 4
GEMV_K_BLOCKS = 32
# The device backend option ``mcc -Od3`` passes; FlagTree's llc call omits it.
LLC_OPTIONS = "-mtgpu-opt-level=1"


# ============================================================================
# Thread index
# TLE has no warp index; tid.x comes from an LLVM IR library linked through
# extern_libs (the MTGPU backend has no inline-asm register constraints).
# ============================================================================

_IR_THREAD_INDEX = """
declare i32 @llvm.musa.read.ptx.sreg.tid.x()
define i32 @fp8_einsum_tid_x(i32 %token) alwaysinline nounwind {
  %t = call i32 @llvm.musa.read.ptx.sreg.tid.x()
  ret i32 %t
}
"""


@lru_cache(maxsize=8)
def materialize_library(name, ir, cache_dir):
    digest = hashlib.sha256(ir.encode()).hexdigest()[:16]
    path = Path(cache_dir) / (name + "-" + digest + ".ll")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as tmp:
            tmp.write(ir)
        os.replace(tmp.name, path)
    return str(path)


def _cache_dir():
    return os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton/cache"))


class _ExternLibs(dict):
    # FlagTree's autotuner hashes every launch keyword to deduplicate configs.
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def extern_libs_thread_index():
    return _ExternLibs(
        fp8_einsum_thread_index=materialize_library(
            "fp8-einsum-thread-index", _IR_THREAD_INDEX, _cache_dir()
        )
    )


@tl_core.extern
def tid_x(token, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [token],
        {(tl.int32,): ("fp8_einsum_tid_x", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# Warp-specialized GEMM
# ============================================================================


@triton.jit
def _unit_coords(
    unit,
    num_units,
    num_m_tiles,
    NUM_N_TILES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BATCH: tl.constexpr,
):
    """Work unit -> (m_tile, n_tile, split, batch): batches slowest, then splits,
    then groups of GROUP_M m-tiles walked m-fastest."""
    if BATCH == 1:
        batch = 0
    else:
        units_per_batch = num_units // BATCH
        batch = unit // units_per_batch
        unit = unit % units_per_batch
        num_units = units_per_batch
    tiles = num_units // SPLIT_K
    split = unit // tiles
    tile = unit % tiles
    tiles_per_group: tl.constexpr = GROUP_M * NUM_N_TILES
    first_m = tile // tiles_per_group * GROUP_M
    group_m = min(num_m_tiles - first_m, GROUP_M)
    in_group = tile % tiles_per_group
    return first_m + in_group % group_m, in_group // group_m, split, batch


@triton.jit
def _block_scales(
    x_scale,
    y_scale_ptrs,
    staged_row_scales,
    k_block,
    unit_k_block,
    row_scale_offsets,
    x_scale_stride_k,
    y_scale_stride_k,
    N_SLICES: tl.constexpr,
    STAGE_ROW_SCALES: tl.constexpr,
):
    """Row scales and per-slice weight scales of one K block.  Holds no
    musa_tle op, since MLIR does not inline a function that does."""
    if STAGE_ROW_SCALES:
        row = tl.load(staged_row_scales + unit_k_block)
    else:
        # Rows past M read row 0's scale and are masked at the store.
        row = tl.load(x_scale + k_block * x_scale_stride_k + row_scale_offsets)
    weights = ()
    for p in tl.static_range(0, N_SLICES):
        weights = weights + (tl.load(y_scale_ptrs[p] + k_block * y_scale_stride_k),)
    return row, weights


@triton.jit
def _pipe_step(step):
    # The pipe derives the stage with a signed remainder; a known non-negative
    # step lets LLVM fold ((group + i) * STAGES + s) % STAGES to the constant s,
    # so barrier ids and slot addresses stay constants.
    return step & 0x7FFFFFFF


@triton.jit
def _produce(
    x_desc,
    y_desc,
    x_scale_desc,
    operands,
    row_scales,
    row_scales_full,
    row_scales_free,
    producer_sync,
    m,
    num_units,
    num_m_tiles,
    NUM_N_TILES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    UNIT_K_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    STAGES: tl.constexpr,
    N_SLICES: tl.constexpr,
    STAGE_ROW_SCALES: tl.constexpr,
    ROW_SCALE_PAD: tl.constexpr,
    BATCH: tl.constexpr,
    X_BATCH_MAJOR: tl.constexpr,
):
    # y is read through its [B*N, K] view.  x (and its scales) through [B*M, K]
    # when batch-major, else through [M, B*K].  One warp issues the copies,
    # except WIDE: all four warps run and meet between consecutive copies.
    WIDE: tl.constexpr = N_SLICES > 1
    if WIDE:
        active = tl.program_id(0) >= 0
    else:
        active = tid_x(tl.program_id(0)) // 32 % 4 == 0
    if active:
        SLICE_N: tl.constexpr = BLOCK_N // N_SLICES
        STEPS_PER_ITER: tl.constexpr = 1 if WIDE else STAGES
        K_BLOCKS: tl.constexpr = SPLIT_K * UNIT_K_BLOCKS
        group = 0
        row_scales_phase = 0
        sync_phase = 0
        for unit in tl.range(tl.program_id(0), num_units, NUM_CTAS, num_stages=1):
            m_tile, n_tile, split, batch = _unit_coords(
                unit, num_units, num_m_tiles, NUM_N_TILES, SPLIT_K, GROUP_M, BATCH
            )
            col0 = batch * (NUM_N_TILES * BLOCK_N) + n_tile * BLOCK_N
            k_block0 = split * UNIT_K_BLOCKS
            if X_BATCH_MAJOR:
                row0 = batch * m + m_tile * BLOCK_M
                x_k_block0 = k_block0
            else:
                row0 = m_tile * BLOCK_M
                x_k_block0 = batch * K_BLOCKS + k_block0
            if STAGE_ROW_SCALES:
                tle.gpu.barrier_wait(row_scales_free[0], phaseIdx=row_scales_phase)
                if WIDE:
                    tle.gpu.barrier_arrive(producer_sync[0], phaseIdx=sync_phase)
                    tle.gpu.barrier_wait(producer_sync[0], phaseIdx=sync_phase)
                    sync_phase = 1 - sync_phase
                tle.gpu.copy(
                    x_scale_desc,
                    row_scales,
                    (BLOCK_M, ROW_SCALE_PAD),
                    (row0, x_k_block0),
                    barrier=row_scales_full[0],
                )
                row_scales_phase = 1 - row_scales_phase
            for it in tl.range(0, UNIT_K_BLOCKS // STEPS_PER_ITER, num_stages=1):
                for s in tl.static_range(0, STEPS_PER_ITER):
                    k_block = it * STEPS_PER_ITER + s
                    if WIDE:
                        step = _pipe_step(group * STAGES + it)
                    else:
                        step = _pipe_step((group + it) * STAGES + s)
                    slot = operands.acquire(step)
                    tle.gpu.copy(
                        x_desc,
                        slot.x,
                        (BLOCK_M, BLOCK_K),
                        (row0, (x_k_block0 + k_block) * BLOCK_K),
                    )
                    for p in tl.static_range(0, N_SLICES):
                        if WIDE:
                            tle.gpu.barrier_arrive(
                                producer_sync[0], phaseIdx=sync_phase
                            )
                            tle.gpu.barrier_wait(producer_sync[0], phaseIdx=sync_phase)
                            sync_phase = 1 - sync_phase
                        if p == 0:
                            y_slot = slot.y0
                        else:
                            y_slot = slot.y1
                        tle.gpu.copy(
                            y_desc,
                            y_slot,
                            (SLICE_N, BLOCK_K),
                            (col0 + p * SLICE_N, (k_block0 + k_block) * BLOCK_K),
                        )
                    operands.commit(step)
            group += UNIT_K_BLOCKS // STAGES


@triton.jit
def _consume(
    x_scale,
    y_scale,
    z,
    partial,
    operands,
    row_scales,
    issue_turn,
    row_scales_full,
    row_scales_free,
    m,
    num_units,
    num_m_tiles,
    x_scale_stride_m,
    x_scale_stride_b,
    x_scale_stride_k,
    y_scale_stride_b,
    y_scale_stride_n,
    y_scale_stride_k,
    z_stride_m,
    z_stride_b,
    z_stride_n,
    partial_plane,
    NUM_N_TILES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    UNIT_K_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    STAGES: tl.constexpr,
    MMA_LAYOUT: tl.constexpr,
    ROW_LAYOUT: tl.constexpr,
    NUM_SQUADS: tl.constexpr,
    N_SLICES: tl.constexpr,
    STAGE_ROW_SCALES: tl.constexpr,
    Z_STRIDE_M: tl.constexpr,
    Z_STRIDE_B: tl.constexpr,
    PROMOTE: tl.constexpr,
    BATCH: tl.constexpr,
):
    squad = tid_x(tl.program_id(0)) // 128
    SLICE_N: tl.constexpr = BLOCK_N // N_SLICES
    WIDE: tl.constexpr = N_SLICES > 1
    # Squads issue their SQMMA in a fixed order; the last one hands squad 0 its
    # first turn.
    ORDERED: tl.constexpr = not PROMOTE and NUM_SQUADS > 1
    if ORDERED:
        if squad == NUM_SQUADS - 1:
            tle.gpu.barrier_arrive(issue_turn[0], phaseIdx=0)
    group = 0
    row_scales_phase = 0
    for unit in tl.range(tl.program_id(0), num_units, NUM_CTAS, num_stages=1):
        m_tile, n_tile, split, batch = _unit_coords(
            unit, num_units, num_m_tiles, NUM_N_TILES, SPLIT_K, GROUP_M, BATCH
        )
        batch_x_scale = x_scale + batch * x_scale_stride_b
        rows = tle.gpu.set_layout(tl.arange(0, BLOCK_M), ROW_LAYOUT) + m_tile * BLOCK_M
        row_scale_offsets = tl.where(rows < m, rows, 0) * x_scale_stride_m
        k_block0 = split * UNIT_K_BLOCKS
        zero = tle.gpu.set_layout(
            tl.zeros((BLOCK_M, SLICE_N), dtype=tl.float32), MMA_LAYOUT
        )
        # A slice is at most one scale block wide, so its weight scale is a scalar.
        accs = ()
        y_scale_ptrs = ()
        for p in tl.static_range(0, N_SLICES):
            accs = accs + (zero,)
            y_scale_ptrs = y_scale_ptrs + (
                y_scale
                + batch * y_scale_stride_b
                + (n_tile * BLOCK_N + p * SLICE_N) // SCALE_BLOCK * y_scale_stride_n,
            )
        staged_row_scales = batch_x_scale
        if STAGE_ROW_SCALES:
            local_rows = rows - m_tile * BLOCK_M
            staged_row_scales = tle.gpu.local_ptr(
                row_scales, (local_rows, local_rows * 0)
            )
            tle.gpu.barrier_wait(row_scales_full[0], phaseIdx=row_scales_phase)

        if PROMOTE:
            acc = zero
            for it in tl.range(0, UNIT_K_BLOCKS // STAGES, num_stages=1):
                prev = zero
                prev_scale = (rows * 0).to(tl.float32)
                for s in tl.static_range(0, STAGES):
                    k_block = k_block0 + it * STAGES + s
                    scale = tl.load(
                        batch_x_scale + k_block * x_scale_stride_k + row_scale_offsets
                    ) * tl.load(y_scale_ptrs[0] + k_block * y_scale_stride_k)
                    stage = operands.wait(_pipe_step((group + it) * STAGES + s))
                    cur = tle.gpu.wgmma(stage.slot.x, stage.slot.y0, zero, trans_b=True)
                    if s > 0:
                        done = tle.gpu.wgmma_wait(1, prev)
                        operands.release(_pipe_step((group + it) * STAGES + s - 1))
                        acc = tl.fma(done, prev_scale[:, None], acc)
                    prev = cur
                    prev_scale = scale
                # The last step of a group is drained before the next group.
                done = tle.gpu.wgmma_wait(0, prev)
                operands.release(_pipe_step((group + it) * STAGES + STAGES - 1))
                acc = tl.fma(done, prev_scale[:, None], acc)
            results = (acc,)
        else:
            row, weights = _block_scales(
                batch_x_scale,
                y_scale_ptrs,
                staged_row_scales,
                k_block0,
                0,
                row_scale_offsets,
                x_scale_stride_k,
                y_scale_stride_k,
                N_SLICES,
                STAGE_ROW_SCALES,
            )
            acc_scales = ()
            for p in tl.static_range(0, N_SLICES):
                acc_scales = acc_scales + (row * weights[p],)
            # WIDE takes one K step per iteration and reads the next scales
            # after handing the stage back; the rest unroll over the stages
            # and read them first.
            STEPS_PER_ITER: tl.constexpr = 1 if WIDE else STAGES
            for it in tl.range(0, UNIT_K_BLOCKS // STEPS_PER_ITER, num_stages=1):
                for s in tl.static_range(0, STEPS_PER_ITER):
                    k_block = it * STEPS_PER_ITER + s
                    next_k_block = tl.minimum(k_block + 1, UNIT_K_BLOCKS - 1)
                    if not WIDE:
                        row, weights = _block_scales(
                            batch_x_scale,
                            y_scale_ptrs,
                            staged_row_scales,
                            k_block0 + next_k_block,
                            next_k_block,
                            row_scale_offsets,
                            x_scale_stride_k,
                            y_scale_stride_k,
                            N_SLICES,
                            STAGE_ROW_SCALES,
                        )
                    if WIDE:
                        step = _pipe_step(group * STAGES + it)
                    else:
                        step = _pipe_step((group + it) * STAGES + s)
                    stage = operands.wait(step)
                    if ORDERED:
                        tle.gpu.barrier_wait(issue_turn[squad], phaseIdx=k_block % 2)
                    pending = ()
                    for p in tl.static_range(0, N_SLICES):
                        if p == 0:
                            y_slot = stage.slot.y0
                        else:
                            y_slot = stage.slot.y1
                        pending = pending + (
                            tle.gpu.wgmma(stage.slot.x, y_slot, accs[p], trans_b=True),
                        )
                    if ORDERED:
                        tle.gpu.barrier_arrive(
                            issue_turn[(squad + 1) % NUM_SQUADS], phaseIdx=k_block % 2
                        )
                    accs = ()
                    for p in tl.static_range(0, N_SLICES):
                        accs = accs + (tle.gpu.wgmma_wait(0, pending[p]),)
                    operands.release(step)
                    if WIDE:
                        row, weights = _block_scales(
                            batch_x_scale,
                            y_scale_ptrs,
                            staged_row_scales,
                            k_block0 + next_k_block,
                            next_k_block,
                            row_scale_offsets,
                            x_scale_stride_k,
                            y_scale_stride_k,
                            N_SLICES,
                            STAGE_ROW_SCALES,
                        )
                    rescaled = ()
                    next_scales = ()
                    for p in tl.static_range(0, N_SLICES):
                        next_scale = row * weights[p]
                        # The ratio of two UE8M0 products is a power of two:
                        # subtract the FP32 bit patterns and re-add the bias.
                        ratio = (
                            acc_scales[p].to(tl.int32, bitcast=True)
                            - next_scale.to(tl.int32, bitcast=True)
                            + 0x3F800000
                        ).to(tl.float32, bitcast=True)
                        rescaled = rescaled + (accs[p] * ratio[:, None],)
                        next_scales = next_scales + (next_scale,)
                    accs = rescaled
                    acc_scales = next_scales
            results = ()
            for p in tl.static_range(0, N_SLICES):
                results = results + (accs[p] * acc_scales[p][:, None],)
        group += UNIT_K_BLOCKS // STAGES
        if STAGE_ROW_SCALES:
            tle.gpu.barrier_arrive(row_scales_free[0], phaseIdx=row_scales_phase)
            row_scales_phase = 1 - row_scales_phase

        # Built after the K loop so they are not live next to the accumulators.
        tile_rows = tle.gpu.set_layout(
            tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, SLICE_N)),
            MMA_LAYOUT,
        )
        tile_cols = tle.gpu.set_layout(
            tl.broadcast_to(tl.arange(0, SLICE_N)[None, :], (BLOCK_M, SLICE_N)),
            MMA_LAYOUT,
        )
        z_rows = m_tile * BLOCK_M + tile_rows
        N: tl.constexpr = NUM_N_TILES * BLOCK_N
        for p in tl.static_range(0, N_SLICES):
            z_cols = n_tile * BLOCK_N + p * SLICE_N + tile_cols
            if SPLIT_K > 1:
                tl.store(
                    partial
                    + (batch * SPLIT_K + split) * partial_plane
                    + z_rows * N
                    + z_cols,
                    results[p].to(partial.dtype.element_ty),
                    mask=z_rows < m,
                )
            elif Z_STRIDE_M > 0:
                # Constexpr strides, and full tiles store unpredicated.
                ptrs = (
                    z
                    + (
                        m_tile * BLOCK_M * Z_STRIDE_M
                        + batch * Z_STRIDE_B
                        + n_tile * BLOCK_N
                        + p * SLICE_N
                    )
                    + (tile_rows * Z_STRIDE_M + tile_cols)
                )
                value = results[p].to(z.dtype.element_ty)
                if m_tile * BLOCK_M + BLOCK_M <= m:
                    tl.store(ptrs, value)
                else:
                    tl.store(ptrs, value, mask=z_rows < m)
            else:
                tl.store(
                    z + batch * z_stride_b + z_rows * z_stride_m + z_cols * z_stride_n,
                    results[p].to(z.dtype.element_ty),
                    mask=z_rows < m,
                )


@triton.jit
def _ws_gemm_fp8(
    x_desc,
    y_desc,
    x_scale_desc,
    x_scale,
    y_scale,
    z,
    partial,
    m,
    n,
    k_blocks,
    x_scale_stride_m,
    x_scale_stride_b,
    x_scale_stride_k,
    y_scale_stride_b,
    y_scale_stride_n,
    y_scale_stride_k,
    z_stride_m,
    z_stride_b,
    z_stride_n,
    num_units,
    num_m_tiles,
    partial_plane,
    BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CONSUMER_WARPS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PROMOTE: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    UNIT_K_BLOCKS: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    STAGES: tl.constexpr,
    MMA_LAYOUT: tl.constexpr,
    ROW_LAYOUT: tl.constexpr,
    N_SLICES: tl.constexpr,
    STAGE_ROW_SCALES: tl.constexpr,
    ROW_SCALE_PAD: tl.constexpr,
    Z_STRIDE_M: tl.constexpr,
    Z_STRIDE_B: tl.constexpr,
    X_BATCH_MAJOR: tl.constexpr,
):
    tl.static_assert(
        STAGES == 2 or STAGES == 4 or STAGES == 8,
        "alloc_barriers takes power-of-two counts",
    )
    tl.static_assert(
        UNIT_K_BLOCKS % STAGES == 0, "a work unit covers whole groups of STAGES"
    )
    # One pipe stage holds x and every N slice of y; each slice is its own
    # SQMMA operand.
    x_bufs = tle.gpu.alloc((STAGES, BLOCK_M, BLOCK_K), tl.float8e4nv)
    y0_bufs = tle.gpu.alloc((STAGES, BLOCK_N // N_SLICES, BLOCK_K), tl.float8e4nv)
    if N_SLICES == 1:
        operands = tle.pipe(capacity=STAGES, name="operands", x=x_bufs, y0=y0_bufs)
    else:
        y1_bufs = tle.gpu.alloc((STAGES, BLOCK_N // N_SLICES, BLOCK_K), tl.float8e4nv)
        operands = tle.pipe(
            capacity=STAGES, name="operands", x=x_bufs, y0=y0_bufs, y1=y1_bufs
        )
    issue_turn = tle.gpu.alloc_barriers(4, arrive_count=4, init=tle.gpu.PENDING)
    producer_sync = tle.gpu.alloc_barriers(
        1, arrive_count=PRODUCER_WARPS, init=tle.gpu.PENDING
    )
    if STAGE_ROW_SCALES:
        # The work unit's whole block of row scales, padded to ROW_SCALE_PAD.
        row_scales = tle.gpu.alloc(
            (BLOCK_M, ROW_SCALE_PAD), tl.float32, nv_mma_shared_layout=False
        )
        row_scales_full = tle.gpu.alloc_barriers(
            1,
            arrive_count=1,
            init=tle.gpu.PENDING,
            expect_bytes=BLOCK_M * ROW_SCALE_PAD * 4,
        )
        row_scales_free = tle.gpu.alloc_barriers(
            1, arrive_count=CONSUMER_WARPS, init=tle.gpu.READY
        )
    else:
        row_scales = x_bufs
        row_scales_full = issue_turn
        row_scales_free = issue_turn
    tle.gpu.warp_specialize(
        [
            (
                _consume,
                (
                    x_scale,
                    y_scale,
                    z,
                    partial,
                    operands.reader(),
                    row_scales,
                    issue_turn,
                    row_scales_full,
                    row_scales_free,
                    m,
                    num_units,
                    num_m_tiles,
                    x_scale_stride_m,
                    x_scale_stride_b,
                    x_scale_stride_k,
                    y_scale_stride_b,
                    y_scale_stride_n,
                    y_scale_stride_k,
                    z_stride_m,
                    z_stride_b,
                    z_stride_n,
                    partial_plane,
                    NUM_N_TILES,
                    SPLIT_K,
                    UNIT_K_BLOCKS,
                    BLOCK_M,
                    BLOCK_N,
                    GROUP_M,
                    NUM_CTAS,
                    STAGES,
                    MMA_LAYOUT,
                    ROW_LAYOUT,
                    CONSUMER_WARPS // 4,
                    N_SLICES,
                    STAGE_ROW_SCALES,
                    Z_STRIDE_M,
                    Z_STRIDE_B,
                    PROMOTE,
                    BATCH,
                ),
            ),
            (
                _produce,
                (
                    x_desc,
                    y_desc,
                    x_scale_desc,
                    operands.writer(),
                    row_scales,
                    row_scales_full,
                    row_scales_free,
                    producer_sync,
                    m,
                    num_units,
                    num_m_tiles,
                    NUM_N_TILES,
                    SPLIT_K,
                    UNIT_K_BLOCKS,
                    BLOCK_M,
                    BLOCK_N,
                    GROUP_M,
                    NUM_CTAS,
                    STAGES,
                    N_SLICES,
                    STAGE_ROW_SCALES,
                    ROW_SCALE_PAD,
                    BATCH,
                    X_BATCH_MAJOR,
                ),
            ),
        ],
        worker_num_warps=[PRODUCER_WARPS],
        worker_num_regs=[64],
    )


@triton.jit
def combine_splits(
    partial,
    z,
    m,
    partial_plane,
    z_stride_m,
    z_stride_b,
    z_stride_n,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum the split-K BF16 partials in FP32, in a fixed order."""
    row = tl.program_id(0)
    batch = tl.program_id(2)
    if row < m:
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for split in range(SPLIT_K):
            acc += tl.load(
                partial + (batch * SPLIT_K + split) * partial_plane + row * N + cols
            ).to(tl.float32)
        tl.store(
            z + row * z_stride_m + batch * z_stride_b + cols * z_stride_n,
            acc.to(z.dtype.element_ty),
        )


# ============================================================================
# GEMV on the CUDA cores: M=1, and shapes no tile configuration covers
# ============================================================================


@triton.jit
def gemv_fp8(
    x,
    x_scale,
    y,
    y_scale,
    z,
    x_stride_m,
    x_stride_b,
    x_scale_stride_m,
    x_scale_stride_b,
    y_stride_b,
    y_scale_stride_b,
    y_scale_stride_n,
    y_scale_stride_k,
    z_stride_m,
    z_stride_b,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_BLOCKS: tl.constexpr,
):
    """BLOCK_N columns of one row and batch per program, K in spans of K_BLOCKS
    scale blocks."""
    batch = tl.program_id(1)
    row = tl.program_id(2)
    x += row * x_stride_m + batch * x_stride_b
    x_scale += row * x_scale_stride_m + batch * x_scale_stride_b
    y += batch * y_stride_b
    y_scale += batch * y_scale_stride_b
    z += row * z_stride_m + batch * z_stride_b
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    NUM_K_BLOCKS: tl.constexpr = K // SCALE_BLOCK
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for span in range(0, tl.cdiv(NUM_K_BLOCKS, K_BLOCKS)):
        k = span * K_BLOCKS * SCALE_BLOCK + tl.arange(0, K_BLOCKS * SCALE_BLOCK)
        xk = tl.load(x + k, mask=k < K, other=0.0).to(tl.float32)
        w = tl.load(
            y + cols[:, None] * K + k[None, :], mask=k[None, :] < K, other=0.0
        ).to(tl.float32)
        block_dots = tl.sum(
            tl.reshape(w * xk[None, :], (BLOCK_N, K_BLOCKS, SCALE_BLOCK)), axis=2
        )
        k_block = span * K_BLOCKS + tl.arange(0, K_BLOCKS)
        x_block_scale = tl.load(
            x_scale + k_block, mask=k_block < NUM_K_BLOCKS, other=0.0
        )
        y_block_scale = tl.load(
            y_scale
            + cols[:, None] // SCALE_BLOCK * y_scale_stride_n
            + k_block[None, :] * y_scale_stride_k,
            mask=k_block[None, :] < NUM_K_BLOCKS,
            other=0.0,
        )
        acc += tl.sum(block_dots * (x_block_scale[None, :] * y_block_scale), axis=1)
    tl.store(z + cols, acc.to(z.dtype.element_ty))


# ============================================================================
# Launch configuration and autotuning
# ============================================================================

# The accumulator must be in the SQMMA layout the backend picks for the tile
# (selectSqmmaConfig, AccelerateMUSAMatmul.cpp); any other costs a conversion
# of the FP32 tile through shared memory.  Its choice: fewest instructions,
# largest instruction, fewest M then N repeats, then a pure M split.
_SQMMA_CANDIDATE_MN = (128, 64, 32, 16)
_SQMMA_CANDIDATE_K = (128, 64, 32)
_SQMMA_INSTR_MN = frozenset(
    {
        (32, 32),
        (32, 64),
        (32, 128),
        (16, 64),
        (64, 16),
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 32),
        (128, 64),
        (128, 128),
    }
)


@lru_cache(maxsize=None)
def sqmma_layout(m, n, k, num_warps):
    best = None  # (key, instruction, warps_per_cta)
    for inst_m in _SQMMA_CANDIDATE_MN:
        for inst_n in _SQMMA_CANDIDATE_MN:
            if m % inst_m or n % inst_n or (inst_m, inst_n) not in _SQMMA_INSTR_MN:
                continue
            for inst_k in _SQMMA_CANDIDATE_K:
                if k % inst_k:
                    continue
                warps_m = 4
                while warps_m <= num_warps:
                    warps_n = num_warps // warps_m
                    tile_m, tile_n = inst_m * (warps_m // 4), inst_n * warps_n
                    if num_warps % warps_m == 0 and m % tile_m == 0 and n % tile_n == 0:
                        rep_m, rep_n = m // tile_m, n // tile_n
                        key = (
                            rep_m * rep_n * (k // inst_k),
                            -inst_m * inst_n * inst_k,
                            rep_m,
                            rep_n,
                        )
                        if (
                            best is None
                            or key < best[0]
                            or (
                                key == best[0]
                                and warps_n == 1
                                and inst_m >= 32
                                and best[2][1] != 1
                            )
                        ):
                            best = (key, [inst_m, inst_n, inst_k], [warps_m, warps_n])
                    warps_m *= 2
    if best is None:
        raise ValueError(f"no SQMMA configuration for {m}x{n}x{k} on {num_warps} warps")
    return tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], best[2], best[1])


# (BLOCK_M, BLOCK_N, CONSUMER_WARPS).  256x256 keeps N in two accumulators: one
# FP32 256x256 tile does not fit the registers.
TILES = (
    (16, 128, 8),
    (32, 128, 8),
    (32, 128, 4),
    (64, 64, 8),
    (64, 128, 8),
    (128, 128, 16),
    (256, 256, 16),
)
# Every K split that divides 32 or 56 K blocks (r = 4096, 7168).
SPLITS = (1, 2, 4, 7, 8, 14, 16, 28)


def stages_for(block_m, block_n, consumer_warps):
    # Each CTA takes more than a third of the MP's shared memory: with three
    # resident CTAs kernels faulted or hung.  Tiles that promote own the MP.
    if block_m == 256:
        return 2
    if block_n == 64 or consumer_warps == 4:
        return 8
    return 4


@lru_cache(maxsize=None)
def _num_sms(device_index):
    return torch.musa.get_device_properties(device_index).multi_processor_count


def _ctas_per_mp(block_m, block_n, consumer_warps):
    return SHARED_BYTES_PER_MP // (
        stages_for(block_m, block_n, consumer_warps)
        * (block_m + block_n)
        * BLOCK_K.value
    )


def _row_scale_pad(unit_k_blocks):
    # At least 16 bytes, the narrowest TME box; only staged units use it.
    return max(4, triton.next_power_of_2(unit_k_blocks))


def legal_configs(configs, named_args, **kwargs):
    """Drop configurations that do not fit the shape or are unsafe.  Promote
    runs only with one CTA per MP, at most two squads and one work unit per
    CTA; a K split keeps whole stage groups per unit and fits one wave."""
    m, n, k_blocks, batch = (
        named_args["m"],
        named_args["n"],
        named_args["k_blocks"],
        kwargs["BATCH"],
    )
    num_sms = _num_sms(torch.musa.current_device())
    legal = []
    for config in configs:
        block_m, block_n, warps, split_k, promote = (
            config.kwargs[name]
            for name in ("BLOCK_M", "BLOCK_N", "CONSUMER_WARPS", "SPLIT_K", "PROMOTE")
        )
        stages = stages_for(block_m, block_n, warps)
        ctas_per_mp = _ctas_per_mp(block_m, block_n, warps)
        tiles = batch * triton.cdiv(m, block_m) * (n // block_n)
        units = tiles * split_k
        if (
            n % block_n
            or block_m > max(16, triton.next_power_of_2(m))
            or k_blocks % split_k
            or (k_blocks // split_k) % stages
            or tiles > 8 * num_sms * ctas_per_mp
        ):
            continue
        if split_k > 1 and units > num_sms * ctas_per_mp:
            continue
        if promote and not (ctas_per_mp == 1 and warps <= 8 and units <= num_sms):
            continue
        legal.append(config)
    return legal


def _set_block_shapes(args):
    """The TME descriptors' boxes follow the configuration."""
    block_m, block_n, split_k = args["BLOCK_M"], args["BLOCK_N"], args["SPLIT_K"]
    args["x_desc"].block_shape = [block_m, BLOCK_K.value]
    args["y_desc"].block_shape = [
        block_n // (2 if block_m == 256 else 1),
        BLOCK_K.value,
    ]
    args["x_scale_desc"].block_shape = [
        block_m,
        _row_scale_pad(args["k_blocks"] // split_k),
    ]


def _combine(args, exception=None):
    if exception is None and args["SPLIT_K"] > 1:
        z, m, n = args["z"], args["m"], args["n"]
        block_n = min(n, 1024)
        rows_per_batch = triton.cdiv(m, args["BLOCK_M"]) * args["BLOCK_M"]
        combine_splits[(rows_per_batch, n // block_n, args["BATCH"])](
            args["partial"],
            z,
            m,
            rows_per_batch * n,
            z.stride(1),
            z.stride(0),
            z.stride(2),
            N=n,
            SPLIT_K=args["SPLIT_K"],
            BLOCK_N=block_n,
            num_warps=8,
        )


def graph_time_us(fn, quantiles=None, rep_ms=100, replays=5):
    """Time of one call in microseconds, from replays of a CUDA graph of about
    ``rep_ms`` of calls.  A replay costs 100-200 us on its own, and the number
    of calls is sized from a short graph, not from eager calls, so neither
    that cost nor the host overhead of ``fn`` reaches the result."""

    def per_call_us(calls, samples):
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            for _ in range(calls):
                fn()
        graph.replay()
        times = []
        for _ in range(samples):
            start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(
                enable_timing=True
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1e3 / calls)
        return statistics.median(times)

    fn()
    torch.musa.synchronize()
    calls = max(10, min(5000, int(rep_ms * 1e3 / per_call_us(10, 1))))
    time_us = per_call_us(calls, replays)
    return [time_us] * len(quantiles) if quantiles else time_us


@lru_cache(maxsize=4096)
def _derived_values(
    block_m, block_n, warps, split_k, m, n, k_blocks, batch, device_index, z_strides
):
    wide = block_m == 256
    stages = stages_for(block_m, block_n, warps)
    num_m_tiles = triton.cdiv(m, block_m)
    num_units = batch * num_m_tiles * (n // block_n) * split_k
    unit_k_blocks = k_blocks // split_k
    pad = _row_scale_pad(unit_k_blocks)
    mma_layout = sqmma_layout(
        block_m, block_n // (2 if wide else 1), BLOCK_K.value, warps
    )
    return dict(
        num_units=num_units,
        num_m_tiles=num_m_tiles,
        partial_plane=num_m_tiles * block_m * n,
        NUM_N_TILES=n // block_n,
        UNIT_K_BLOCKS=unit_k_blocks,
        # n-fastest for 256x256: the two MPs of an MPX then read the same rows.
        GROUP_M=1 if wide else 8,
        NUM_CTAS=min(
            _num_sms(device_index) * _ctas_per_mp(block_m, block_n, warps), num_units
        ),
        STAGES=stages,
        MMA_LAYOUT=mma_layout,
        ROW_LAYOUT=tle.gpu.SlicedEncoding(1, mma_layout),
        N_SLICES=2 if wide else 1,
        STAGE_ROW_SCALES=int(
            wide
            and unit_k_blocks > 8
            and stages * (block_m + block_n) * BLOCK_K.value + block_m * pad * 4
            <= SHARED_BYTES_PER_MP
        ),
        ROW_SCALE_PAD=pad,
        # 256x256 stores through constexpr z strides (0: runtime strides).
        Z_STRIDE_M=z_strides[1] if wide else 0,
        Z_STRIDE_B=z_strides[0] if wide else 0,
    )


def _derived(args):
    """The constexprs and loop bounds that follow from the configuration."""
    z = args["z"]
    return _derived_values(
        args["BLOCK_M"],
        args["BLOCK_N"],
        args["CONSUMER_WARPS"],
        args["SPLIT_K"],
        args["m"],
        args["n"],
        args["k_blocks"],
        args["BATCH"],
        z.device.index,
        z.stride(),
    )


_DERIVED = (
    "num_units",
    "num_m_tiles",
    "partial_plane",
    "NUM_N_TILES",
    "UNIT_K_BLOCKS",
    "GROUP_M",
    "NUM_CTAS",
    "STAGES",
    "MMA_LAYOUT",
    "ROW_LAYOUT",
    "N_SLICES",
    "STAGE_ROW_SCALES",
    "ROW_SCALE_PAD",
    "Z_STRIDE_M",
    "Z_STRIDE_B",
)

ws_gemm_fp8 = triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "CONSUMER_WARPS": warps,
                "SPLIT_K": split_k,
                "PROMOTE": promote,
            },
            num_warps=warps,
            pre_hook=_set_block_shapes,
        )
        for bm, bn, warps in TILES
        for split_k in SPLITS
        for promote in (0, 1)
    ],
    key=["m", "n", "k_blocks", "BATCH"],
    prune_configs_by={"early_config_prune": legal_configs},
    post_hook=_combine,
    do_bench=graph_time_us,
)(
    triton.heuristics(
        {
            name: (lambda name: lambda args: _derived(args)[name])(name)
            for name in _DERIVED
        }
    )(_ws_gemm_fp8)
)


@lru_cache(maxsize=None)
def _has_tile_config(m, n, k_blocks, batch):
    # A work unit covers whole groups of stages, so a short K may have none.
    return bool(
        legal_configs(
            ws_gemm_fp8.configs, {"m": m, "n": n, "k_blocks": k_blocks}, BATCH=batch
        )
    )


@lru_cache(maxsize=None)
def _partial_rows(m, n, k_blocks, batch):
    """Rows per batch of the split-K partials, for the largest split any legal
    configuration of the shape takes."""
    named = {"m": m, "n": n, "k_blocks": k_blocks}
    return max(
        (
            c.kwargs["SPLIT_K"]
            * triton.cdiv(m, c.kwargs["BLOCK_M"])
            * c.kwargs["BLOCK_M"]
            for c in legal_configs(ws_gemm_fp8.configs, named, BATCH=batch)
            if c.kwargs["SPLIT_K"] > 1
        ),
        default=1,
    )


def _tme_ready(t):
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.contiguous()


def w8a8_block_fp8_bmm(
    x, y, xs, ys, block_size=[128, 128], z=None, output_dtype=torch.bfloat16
):
    """``z[B,M,N] = x[B,M,K] @ y[B,N,K]^T`` with 128x128 block scales
    ``xs[B,M,K/128]`` and ``ys[B,N/128,K/128]``.

    The TME descriptors address y as [B*N, K], and x (with xs) either as
    [B*M, K] when batch-major or as [M, B*K] when it is an [M,B,K] tensor seen
    through ``transpose(0, 1)``, as ``fp8_einsum`` passes it.  Any other layout
    is copied first.  ``z`` may be strided along B and M."""
    assert tuple(block_size) == (
        128,
        128,
    ), "this kernel assumes 128x128 block-wise FP8 scales"
    batch, m, k = x.shape
    n = y.shape[1]
    assert (
        y.shape == (batch, n, k)
        and xs.shape[:2] == (batch, m)
        and k % BLOCK_K.value == 0
        and n % 128 == 0
    )
    if z is None:
        z = torch.empty((batch, m, n), device=x.device, dtype=output_dtype)
    assert z.shape == (batch, m, n) and z.dtype == output_dtype and z.stride(-1) == 1
    if z.numel() == 0:
        return z
    if k == 0:
        return z.zero_()
    k_blocks = k // BLOCK_K.value
    y = _tme_ready(y)
    if m == 1 or not _has_tile_config(m, n, k_blocks, batch):
        x = x if x.stride(-1) == 1 else x.contiguous()
        xs = xs if xs.stride(-1) == 1 else xs.contiguous()
        gemv_fp8[(n // GEMV_BLOCK_N, batch, m)](
            x,
            xs,
            y,
            ys,
            z,
            x.stride(1),
            x.stride(0),
            xs.stride(1),
            xs.stride(0),
            y.stride(0),
            ys.stride(0),
            ys.stride(1),
            ys.stride(2),
            z.stride(1),
            z.stride(0),
            K=k,
            BLOCK_N=GEMV_BLOCK_N,
            K_BLOCKS=GEMV_K_BLOCKS,
            num_warps=8,
        )
        return z
    interleaved = (
        batch > 1 and x.transpose(0, 1).is_contiguous() and x.data_ptr() % 16 == 0
    )
    if interleaved:
        xs = xs.transpose(0, 1).contiguous().transpose(0, 1)
        x_view, xs_view = x.transpose(0, 1).reshape(m, batch * k), xs.transpose(
            0, 1
        ).reshape(m, batch * k_blocks)
    else:
        x, xs = _tme_ready(x), _tme_ready(xs)
        x_view, xs_view = x.view(batch * m, k), xs.view(batch * m, k_blocks)
    partial = torch.empty(
        (batch * _partial_rows(m, n, k_blocks, batch), n),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # Placeholder boxes; each configuration's pre_hook sets its own.
    x_desc = TensorDescriptor.from_tensor(x_view, block_shape=[16, BLOCK_K.value])
    y_desc = TensorDescriptor.from_tensor(
        y.view(batch * n, k), block_shape=[128, BLOCK_K.value]
    )
    x_scale_desc = TensorDescriptor.from_tensor(xs_view, block_shape=[16, 4])
    ws_gemm_fp8[lambda meta: (meta["NUM_CTAS"],)](
        x_desc,
        y_desc,
        x_scale_desc,
        xs,
        ys,
        z,
        partial,
        m,
        n,
        k_blocks,
        xs.stride(1),
        xs.stride(0),
        xs.stride(2),
        ys.stride(0),
        ys.stride(1),
        ys.stride(2),
        z.stride(1),
        z.stride(0),
        z.stride(2),
        BATCH=batch,
        X_BATCH_MAJOR=int(not interleaved),
        llc_options=LLC_OPTIONS,
        extern_libs=extern_libs_thread_index(),
    )
    # The autotuner's post_hook only runs while benchmarking.
    _combine(
        {
            **ws_gemm_fp8.best_config.kwargs,
            "z": z,
            "partial": partial,
            "m": m,
            "n": n,
            "BATCH": batch,
        }
    )
    return z

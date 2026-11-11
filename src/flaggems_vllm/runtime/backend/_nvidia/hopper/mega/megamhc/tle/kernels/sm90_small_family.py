"""TLE SM90 MegaMHC bring-up kernel and cross-CTA protocol probe.

This module is deliberately *not* a complete MegaMHC implementation.  It is
the first migration checkpoint for the Hopper TLE kernel: prove that split-K
producers can publish the values needed by the MegaMHC tail and that exactly
one last-arriving CTA can safely reduce them in the same kernel launch.

The probe mirrors the real MegaMHC split workspace at a smaller granularity.
For every ``(token, split)`` producer it accepts:

* 24 FP32 projection partials (four Pre, four Post, and sixteen residual-mix
  logits); and
* one FP32 sum-of-squares partial.

Each producer writes one private 32-float global-workspace row.  A TLE named
barrier orders all CTA threads' stores before a device-scope ``acq_rel``
atomic increment.  Ticket generation ``g`` owns the range
``[g * num_splits, (g + 1) * num_splits)``.  The CTA for which
``(ticket + 1) % num_splits == 0`` is the owner: after another CTA barrier
propagates the acquire to its worker threads, it reduces all rows and writes
the two outputs.  The uint64 ticket remains monotonic, so replay needs neither
a separate reset nor a memset.

The public launch path performs no allocation and no counter memset, so a
captured invocation remains exactly one CUDA Graph kernel node.  Allocate the
workspace before capture with
``allocate_tle_sm90_mega_mhc_atomic_workspace``.  One workspace instance may
be reused by ordered launches on one stream because each replay advances to a
new ticket generation.  It must never be shared by overlapping launches or
streams.

The module also contains deliberately narrow complete MegaMHC schedules for
the first seven DeepSeek-V4-Flash Normal BF16 migration cases,
``T in {1, 2, 4, 8, 16, 24, 32}, H=4096, R=4, P=24``.  Every schedule launches
exactly one kernel.  T=1 uses split-16/tile-4 with routing-worker overlap;
T=2 uses the promoted split-8/tile-3 16-warp path; T in {4, 8} uses
split-8/tile-6 with four output tiles per token; and T in {16, 24, 32} uses
split-8/tile-12.  T=32 additionally exposes default-off split-8/tile-24
candidates that halve its CTA count: the original generic 8-warp reduction
and shared25 reductions in both 8-warp pair-folded and 16-warp direct forms,
selected explicitly for A/B experiments.  The 8-warp shared25 form also has a
default-off three-output projection-ILP probe.  The explicit single-CTA
correctness fallback remains T=1-only.  A separate default-off T=32 token-pair
kernel halves the grid to 128 CTAs and reuses every Fn weight load across two
adjacent tokens while retaining independent token workspaces and handoffs; the
other experimental schedules remain available only for their admitted buckets.
The default-off T=32 packed-sqsum probe stores its statistic in projection
workspace padding lane 24 so producer and tail each access one packed row.  A
second default-off probe replaces four Pre-mix selection reductions with four
direct lane-0..3 warp broadcasts on that exact packed path.  An optional
shared-scalar bridge publishes those four warp-uniform values through a private
``[8, 4]`` FP32 shared table per physical warp, avoiding layout conversion when
the hidden tail consumes them.  A third default-off probe narrows the
producer-publish barrier to warp 0, whose lanes
0..24 are the only threads that store the packed partial row, before lane 0
performs the existing release atomic handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import os
from typing import Iterable, Mapping

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl


NUM_MHC_OUTPUTS = 24
NUM_SQRSUM_PARTIALS = 1
WORKSPACE_ROW_ELEMENTS = 32
MAX_NUM_SPLITS = 132
NUM_WARPS = 4
CTA_THREADS = NUM_WARPS * 32
FULL_TOKENS = 1
SUPPORTED_FULL_TOKENS = (1, 2, 4, 8, 16, 24, 32)
FULL_HIDDEN = 4096
FULL_HC_MULT = 4
FULL_PROJECTION_OUTPUTS = 24
FULL_FLAT_HIDDEN = FULL_HIDDEN * FULL_HC_MULT
FULL_NUM_WARPS = 8
FULL_CTA_THREADS = FULL_NUM_WARPS * 32
FULL_OVERLAP_ROUTING_WARPS = 1
FULL_OVERLAP_LOGICAL_THREADS = (
    FULL_CTA_THREADS + FULL_OVERLAP_ROUTING_WARPS * 32
)
# SM90 TLE warp-specialize partitions are allocated in four-warp groups.  A
# one-warp worker therefore makes the physical CTA 8 + 4 = 12 warps, while
# only 8 + 1 warps participate in the two named producer/consumer barriers.
FULL_OVERLAP_PHYSICAL_WARPS = FULL_NUM_WARPS + 4
FULL_OVERLAP_PHYSICAL_THREADS = FULL_OVERLAP_PHYSICAL_WARPS * 32
FULL_SINKHORN_ITERS = 20
FULL_NUM_SPLITS = 16
FULL_HIDDEN_PER_SPLIT = FULL_HIDDEN // FULL_NUM_SPLITS
FULL_OUTPUT_TILE = 4
FULL_NUM_OUTPUT_TILES = FULL_PROJECTION_OUTPUTS // FULL_OUTPUT_TILE
FULL_SPLIT_CTAS = FULL_NUM_SPLITS * FULL_NUM_OUTPUT_TILES
FULL_LOCAL_FLAT_HIDDEN = FULL_HC_MULT * FULL_HIDDEN_PER_SPLIT
T8_NUM_SPLITS = 8
T8_HIDDEN_PER_SPLIT = FULL_HIDDEN // T8_NUM_SPLITS
T8_OUTPUTS_PER_TILE = 12
T8_OUTPUT_TILE_BLOCK = 16
T8_NUM_OUTPUT_TILES = FULL_PROJECTION_OUTPUTS // T8_OUTPUTS_PER_TILE
T8_CTAS_PER_TOKEN = T8_NUM_SPLITS * T8_NUM_OUTPUT_TILES
T8_GRID_CTAS = 8 * T8_CTAS_PER_TOKEN
T8_REDUCTION_THREADS = FULL_CTA_THREADS
T8_REDUCTION_WARPS = FULL_NUM_WARPS
T8_REDUCTION_CHANNELS = T8_OUTPUTS_PER_TILE + 1
T8_REDUCTION_CHANNEL_BLOCK = 16
T2_TILE3_OUTPUTS_PER_TILE = 3
T2_TILE3_OUTPUT_BLOCK = 4
T2_TILE3_NUM_OUTPUT_TILES = (
    FULL_PROJECTION_OUTPUTS // T2_TILE3_OUTPUTS_PER_TILE
)
T2_TILE3_REDUCTION_CHANNELS = T2_TILE3_OUTPUTS_PER_TILE + 1
T2_TILE3_REDUCTION_CHANNEL_BLOCK = 4
T8_TILE6_OUTPUTS_PER_TILE = 6
T8_TILE6_OUTPUT_BLOCK = 8
T8_TILE6_NUM_OUTPUT_TILES = FULL_PROJECTION_OUTPUTS // T8_TILE6_OUTPUTS_PER_TILE
T8_TILE6_REDUCTION_CHANNELS = T8_TILE6_OUTPUTS_PER_TILE + 1
T8_TILE6_REDUCTION_CHANNEL_BLOCK = 8
T8_TILE8_OUTPUTS_PER_TILE = 8
T8_TILE8_OUTPUT_BLOCK = 8
T8_TILE8_NUM_OUTPUT_TILES = FULL_PROJECTION_OUTPUTS // T8_TILE8_OUTPUTS_PER_TILE
T8_TILE8_REDUCTION_CHANNELS = T8_TILE8_OUTPUTS_PER_TILE + 1
T8_TILE8_REDUCTION_CHANNEL_BLOCK = 16
T32_TILE24_REDUCTION_CHANNELS = FULL_PROJECTION_OUTPUTS + 1
T32_TILE24_REDUCTION_CHANNEL_BLOCK = 32
COUNTER_CACHE_LINE_BYTES = 128
COUNTER_U32_STRIDE = COUNTER_CACHE_LINE_BYTES // 4
JIT_NUM_MHC_OUTPUTS = tl.constexpr(NUM_MHC_OUTPUTS)
JIT_NUM_SQRSUM_PARTIALS = tl.constexpr(NUM_SQRSUM_PARTIALS)
JIT_WORKSPACE_ROW_ELEMENTS = tl.constexpr(WORKSPACE_ROW_ELEMENTS)
JIT_CTA_THREADS = tl.constexpr(CTA_THREADS)
JIT_FULL_HIDDEN = tl.constexpr(FULL_HIDDEN)
JIT_FULL_HC_MULT = tl.constexpr(FULL_HC_MULT)
JIT_FULL_PROJECTION_OUTPUTS = tl.constexpr(FULL_PROJECTION_OUTPUTS)
JIT_FULL_FLAT_HIDDEN = tl.constexpr(FULL_FLAT_HIDDEN)
JIT_FULL_CTA_THREADS = tl.constexpr(FULL_CTA_THREADS)
JIT_FULL_OVERLAP_ROUTING_WARPS = tl.constexpr(
    FULL_OVERLAP_ROUTING_WARPS
)
JIT_FULL_OVERLAP_LOGICAL_THREADS = tl.constexpr(
    FULL_OVERLAP_LOGICAL_THREADS
)
JIT_FULL_SINKHORN_ITERS = tl.constexpr(FULL_SINKHORN_ITERS)
JIT_FULL_NUM_SPLITS = tl.constexpr(FULL_NUM_SPLITS)
JIT_FULL_HIDDEN_PER_SPLIT = tl.constexpr(FULL_HIDDEN_PER_SPLIT)
JIT_FULL_OUTPUT_TILE = tl.constexpr(FULL_OUTPUT_TILE)
JIT_FULL_NUM_OUTPUT_TILES = tl.constexpr(FULL_NUM_OUTPUT_TILES)
JIT_FULL_SPLIT_CTAS = tl.constexpr(FULL_SPLIT_CTAS)
JIT_FULL_LOCAL_FLAT_HIDDEN = tl.constexpr(FULL_LOCAL_FLAT_HIDDEN)
JIT_T8_NUM_SPLITS = tl.constexpr(T8_NUM_SPLITS)
JIT_T8_REDUCTION_THREADS = tl.constexpr(T8_REDUCTION_THREADS)
JIT_T8_REDUCTION_WARPS = tl.constexpr(T8_REDUCTION_WARPS)
JIT_T8_REDUCTION_CHANNELS = tl.constexpr(T8_REDUCTION_CHANNELS)
JIT_T8_REDUCTION_CHANNEL_BLOCK = tl.constexpr(T8_REDUCTION_CHANNEL_BLOCK)
JIT_T2_TILE3_REDUCTION_CHANNELS = tl.constexpr(
    T2_TILE3_REDUCTION_CHANNELS
)
JIT_T2_TILE3_REDUCTION_CHANNEL_BLOCK = tl.constexpr(
    T2_TILE3_REDUCTION_CHANNEL_BLOCK
)
JIT_T2_TILE3_NUM_OUTPUT_TILES = tl.constexpr(T2_TILE3_NUM_OUTPUT_TILES)
JIT_T8_TILE6_REDUCTION_CHANNELS = tl.constexpr(T8_TILE6_REDUCTION_CHANNELS)
JIT_T8_TILE6_REDUCTION_CHANNEL_BLOCK = tl.constexpr(
    T8_TILE6_REDUCTION_CHANNEL_BLOCK
)
JIT_T8_TILE6_NUM_OUTPUT_TILES = tl.constexpr(T8_TILE6_NUM_OUTPUT_TILES)
JIT_T8_TILE8_REDUCTION_CHANNELS = tl.constexpr(T8_TILE8_REDUCTION_CHANNELS)
JIT_T8_TILE8_REDUCTION_CHANNEL_BLOCK = tl.constexpr(
    T8_TILE8_REDUCTION_CHANNEL_BLOCK
)
JIT_T32_TILE24_REDUCTION_CHANNELS = tl.constexpr(
    T32_TILE24_REDUCTION_CHANNELS
)
JIT_T32_TILE24_REDUCTION_CHANNEL_BLOCK = tl.constexpr(
    T32_TILE24_REDUCTION_CHANNEL_BLOCK
)
JIT_T8_TILE8_NUM_OUTPUT_TILES = tl.constexpr(T8_TILE8_NUM_OUTPUT_TILES)


@triton.jit
def _warp_sum_32(value):
    """Return a full-warp FP32 sum in lane zero without a CTA barrier."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t;
          mov.f32 $0, $1;
          shfl.sync.down.b32 t, $0, 16, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 8, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 4, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 2, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 1, 31, 0xffffffff;
          add.f32 $0, $0, t;
        }
        """,
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_sum_broadcast_32(value):
    """Return the lane-zero FP32 warp sum broadcast to every lane."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t;
          mov.f32 $0, $1;
          shfl.sync.down.b32 t, $0, 16, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 8, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 4, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 2, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 1, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.idx.b32 t, $0, 0, 31, 0xffffffff;
          mov.f32 $0, t;
        }
        """,
        # The output is written before the input is no longer live.
        constraints="=&f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_broadcast_lanes0_3_f32(value):
    """Broadcast lanes 0..3 as four independent FP32 warp scalars."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          shfl.sync.idx.b32 $0, $4, 0, 31, 0xffffffff;
          shfl.sync.idx.b32 $1, $4, 1, 31, 0xffffffff;
          shfl.sync.idx.b32 $2, $4, 2, 31, 0xffffffff;
          shfl.sync.idx.b32 $3, $4, 3, 31, 0xffffffff;
        }
        """,
        # All four outputs are written before the shared input is dead.
        constraints="=&f,=&f,=&f,=&f,f",
        args=[value],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_sync_full_mask():
    """Order one physical warp's shared stores before its scalar loads."""
    return tl.inline_asm_elementwise(
        asm="bar.warp.sync 0xffffffff; mov.u32 $0, 0;",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _warp_sum3_32(value0, value1, value2):
    """Reduce three independent FP32 values after all three are available."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t;
          mov.f32 $0, $3;
          shfl.sync.down.b32 t, $0, 16, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 8, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 4, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 2, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 1, 31, 0xffffffff;
          add.f32 $0, $0, t;
          mov.f32 $1, $4;
          shfl.sync.down.b32 t, $1, 16, 31, 0xffffffff;
          add.f32 $1, $1, t;
          shfl.sync.down.b32 t, $1, 8, 31, 0xffffffff;
          add.f32 $1, $1, t;
          shfl.sync.down.b32 t, $1, 4, 31, 0xffffffff;
          add.f32 $1, $1, t;
          shfl.sync.down.b32 t, $1, 2, 31, 0xffffffff;
          add.f32 $1, $1, t;
          shfl.sync.down.b32 t, $1, 1, 31, 0xffffffff;
          add.f32 $1, $1, t;
          mov.f32 $2, $5;
          shfl.sync.down.b32 t, $2, 16, 31, 0xffffffff;
          add.f32 $2, $2, t;
          shfl.sync.down.b32 t, $2, 8, 31, 0xffffffff;
          add.f32 $2, $2, t;
          shfl.sync.down.b32 t, $2, 4, 31, 0xffffffff;
          add.f32 $2, $2, t;
          shfl.sync.down.b32 t, $2, 2, 31, 0xffffffff;
          add.f32 $2, $2, t;
          shfl.sync.down.b32 t, $2, 1, 31, 0xffffffff;
          add.f32 $2, $2, t;
        }
        """,
        # Every output is written before all three inputs have been consumed.
        # Mark them early-clobber so the allocator cannot alias $0/$1/$2 with
        # a still-live $3/$4/$5 input and silently corrupt a later channel.
        constraints="=&f,=&f,=&f,f,f,f",
        args=[value0, value1, value2],
        dtype=(tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_sum2_32(value0, value1):
    """Reduce two independent token accumulators in the same warp tree."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t;
          mov.f32 $0, $2;
          shfl.sync.down.b32 t, $0, 16, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 8, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 4, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 2, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.down.b32 t, $0, 1, 31, 0xffffffff;
          add.f32 $0, $0, t;
          mov.f32 $1, $3;
          shfl.sync.down.b32 t, $1, 16, 31, 0xffffffff;
          add.f32 $1, $1, t;
          shfl.sync.down.b32 t, $1, 8, 31, 0xffffffff;
          add.f32 $1, $1, t;
          shfl.sync.down.b32 t, $1, 4, 31, 0xffffffff;
          add.f32 $1, $1, t;
          shfl.sync.down.b32 t, $1, 2, 31, 0xffffffff;
          add.f32 $1, $1, t;
          shfl.sync.down.b32 t, $1, 1, 31, 0xffffffff;
          add.f32 $1, $1, t;
        }
        """,
        constraints="=&f,=&f,f,f",
        args=[value0, value1],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_sum2_interleaved_32(value0, value1):
    """Reduce two FP32 values while interleaving each shuffle dependency."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t0, t1;
          mov.f32 $0, $2;
          mov.f32 $1, $3;
          shfl.sync.down.b32 t0, $0, 16, 31, 0xffffffff;
          shfl.sync.down.b32 t1, $1, 16, 31, 0xffffffff;
          add.f32 $0, $0, t0;
          add.f32 $1, $1, t1;
          shfl.sync.down.b32 t0, $0, 8, 31, 0xffffffff;
          shfl.sync.down.b32 t1, $1, 8, 31, 0xffffffff;
          add.f32 $0, $0, t0;
          add.f32 $1, $1, t1;
          shfl.sync.down.b32 t0, $0, 4, 31, 0xffffffff;
          shfl.sync.down.b32 t1, $1, 4, 31, 0xffffffff;
          add.f32 $0, $0, t0;
          add.f32 $1, $1, t1;
          shfl.sync.down.b32 t0, $0, 2, 31, 0xffffffff;
          shfl.sync.down.b32 t1, $1, 2, 31, 0xffffffff;
          add.f32 $0, $0, t0;
          add.f32 $1, $1, t1;
          shfl.sync.down.b32 t0, $0, 1, 31, 0xffffffff;
          shfl.sync.down.b32 t1, $1, 1, 31, 0xffffffff;
          add.f32 $0, $0, t0;
          add.f32 $1, $1, t1;
        }
        """,
        constraints="=&f,=&f,f,f",
        args=[value0, value1],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pair_sum_512_to_256(value):
    """Add each contiguous pair owned by one lane in the split-8 v2 layout."""
    pairs = tl.reshape(value, (JIT_T8_REDUCTION_THREADS, 2))
    first, second = tl.split(pairs)
    return first + second


@triton.jit
def _load_shared5_f32(p0, p1, p2, p3, ps):
    """Load four projection partials and sqsum in one TLE IR operation."""
    return tl.inline_asm_elementwise(
        asm="""
        ld.shared.f32 $0, [$5];
        ld.shared.f32 $1, [$6];
        ld.shared.f32 $2, [$7];
        ld.shared.f32 $3, [$8];
        ld.shared.f32 $4, [$9];
        """,
        constraints="=f,=f,=f,=f,=f,r,r,r,r,r",
        args=[p0, p1, p2, p3, ps],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _shared_address_u32(pointer):
    """Expose a TLE shared pointer as a PTX shared address."""
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _store_shared5_f32(p0, p1, p2, p3, ps, v0, v1, v2, v3, vs, lane):
    """Publish five warp partials with one compiler-visible operation."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader;
          setp.eq.u32 leader, $11, 0;
          @leader st.shared.f32 [$1], $6;
          @leader st.shared.f32 [$2], $7;
          @leader st.shared.f32 [$3], $8;
          @leader st.shared.f32 [$4], $9;
          @leader st.shared.f32 [$5], $10;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,r,r,r,r,f,f,f,f,f,r",
        args=[p0, p1, p2, p3, ps, v0, v1, v2, v3, vs, lane],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_shared4_f32(p0, p1, p2, p3, v0, v1, v2, v3, lane):
    """Publish four warp-uniform FP32 values from physical lane zero."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader;
          setp.eq.u32 leader, $9, 0;
          @leader st.shared.f32 [$1], $5;
          @leader st.shared.f32 [$2], $6;
          @leader st.shared.f32 [$3], $7;
          @leader st.shared.f32 [$4], $8;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,r,r,r,f,f,f,f,r",
        args=[p0, p1, p2, p3, v0, v1, v2, v3, lane],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_shared3_f32(p0, p1, p2, v0, v1, v2, lane):
    """Publish three projection warp partials with one side-effecting op."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader;
          setp.eq.u32 leader, $7, 0;
          @leader st.shared.f32 [$1], $4;
          @leader st.shared.f32 [$2], $5;
          @leader st.shared.f32 [$3], $6;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,r,r,f,f,f,r",
        args=[p0, p1, p2, v0, v1, v2, lane],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_shared2_f32(p0, p1, v0, v1, lane):
    """Publish the two token-local warp partials with one side effect."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader;
          setp.eq.u32 leader, $5, 0;
          @leader st.shared.f32 [$1], $3;
          @leader st.shared.f32 [$2], $4;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,r,f,f,r",
        args=[p0, p1, v0, v1, lane],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_shared1_f32(pointer, value, lane):
    """Publish one warp partial with an inline predicated shared store."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader;
          setp.eq.u32 leader, $3, 0;
          @leader st.shared.f32 [$1], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,f,r",
        args=[pointer, value, lane],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reduce_shared8_f32(p0, p1, p2, p3, p4, p5, p6, p7, thread):
    """Reduce eight warp partials in the same tree order as shfl.down."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          setp.lt.u32 active, $9, 5;
          mov.f32 v0, 0f00000000;
          mov.f32 v1, 0f00000000;
          mov.f32 v2, 0f00000000;
          mov.f32 v3, 0f00000000;
          mov.f32 v4, 0f00000000;
          mov.f32 v5, 0f00000000;
          mov.f32 v6, 0f00000000;
          mov.f32 v7, 0f00000000;
          @active ld.shared.f32 v0, [$1];
          @active ld.shared.f32 v1, [$2];
          @active ld.shared.f32 v2, [$3];
          @active ld.shared.f32 v3, [$4];
          @active ld.shared.f32 v4, [$5];
          @active ld.shared.f32 v5, [$6];
          @active ld.shared.f32 v6, [$7];
          @active ld.shared.f32 v7, [$8];
          add.f32 v0, v0, v4;
          add.f32 v1, v1, v5;
          add.f32 v2, v2, v6;
          add.f32 v3, v3, v7;
          add.f32 v0, v0, v2;
          add.f32 v1, v1, v3;
          add.f32 $0, v0, v1;
        }
        """,
        constraints="=f,r,r,r,r,r,r,r,r,r",
        args=[p0, p1, p2, p3, p4, p5, p6, p7, thread],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reduce_shared8_tid4_f32(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    thread,
    output_tile,
):
    """Let tid 0..2 reduce projection rows and tid 3 reduce sqsum."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq, active;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          setp.lt.u32 projection, $9, 3;
          setp.eq.u32 sq_thread, $9, 3;
          setp.eq.u32 tile_zero, $10, 0;
          and.pred sq, sq_thread, tile_zero;
          or.pred active, projection, sq;
          mov.f32 v0, 0f00000000;
          mov.f32 v1, 0f00000000;
          mov.f32 v2, 0f00000000;
          mov.f32 v3, 0f00000000;
          mov.f32 v4, 0f00000000;
          mov.f32 v5, 0f00000000;
          mov.f32 v6, 0f00000000;
          mov.f32 v7, 0f00000000;
          @active ld.shared.f32 v0, [$1];
          @active ld.shared.f32 v1, [$2];
          @active ld.shared.f32 v2, [$3];
          @active ld.shared.f32 v3, [$4];
          @active ld.shared.f32 v4, [$5];
          @active ld.shared.f32 v5, [$6];
          @active ld.shared.f32 v6, [$7];
          @active ld.shared.f32 v7, [$8];
          add.f32 v0, v0, v4;
          add.f32 v1, v1, v5;
          add.f32 v2, v2, v6;
          add.f32 v3, v3, v7;
          add.f32 v0, v0, v2;
          add.f32 v1, v1, v3;
          add.f32 $0, v0, v1;
        }
        """,
        constraints="=f,r,r,r,r,r,r,r,r,r,r",
        args=[
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            thread,
            output_tile,
        ],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reduce_shared16_tid4_f32(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    thread,
    output_tile,
):
    """Let tid 0..3 reduce one channel across sixteen producer warps."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq, active;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          .reg .f32 v8, v9, v10, v11, v12, v13, v14, v15;
          setp.lt.u32 projection, $17, 3;
          setp.eq.u32 sq_thread, $17, 3;
          setp.eq.u32 tile_zero, $18, 0;
          and.pred sq, sq_thread, tile_zero;
          or.pred active, projection, sq;
          mov.f32 v0, 0f00000000;
          mov.f32 v1, 0f00000000;
          mov.f32 v2, 0f00000000;
          mov.f32 v3, 0f00000000;
          mov.f32 v4, 0f00000000;
          mov.f32 v5, 0f00000000;
          mov.f32 v6, 0f00000000;
          mov.f32 v7, 0f00000000;
          mov.f32 v8, 0f00000000;
          mov.f32 v9, 0f00000000;
          mov.f32 v10, 0f00000000;
          mov.f32 v11, 0f00000000;
          mov.f32 v12, 0f00000000;
          mov.f32 v13, 0f00000000;
          mov.f32 v14, 0f00000000;
          mov.f32 v15, 0f00000000;
          @active ld.shared.f32 v0, [$1];
          @active ld.shared.f32 v1, [$2];
          @active ld.shared.f32 v2, [$3];
          @active ld.shared.f32 v3, [$4];
          @active ld.shared.f32 v4, [$5];
          @active ld.shared.f32 v5, [$6];
          @active ld.shared.f32 v6, [$7];
          @active ld.shared.f32 v7, [$8];
          @active ld.shared.f32 v8, [$9];
          @active ld.shared.f32 v9, [$10];
          @active ld.shared.f32 v10, [$11];
          @active ld.shared.f32 v11, [$12];
          @active ld.shared.f32 v12, [$13];
          @active ld.shared.f32 v13, [$14];
          @active ld.shared.f32 v14, [$15];
          @active ld.shared.f32 v15, [$16];
          add.f32 v0, v0, v8;
          add.f32 v1, v1, v9;
          add.f32 v2, v2, v10;
          add.f32 v3, v3, v11;
          add.f32 v4, v4, v12;
          add.f32 v5, v5, v13;
          add.f32 v6, v6, v14;
          add.f32 v7, v7, v15;
          add.f32 v0, v0, v4;
          add.f32 v1, v1, v5;
          add.f32 v2, v2, v6;
          add.f32 v3, v3, v7;
          add.f32 v0, v0, v2;
          add.f32 v1, v1, v3;
          add.f32 $0, v0, v1;
        }
        """,
        constraints="=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
        args=[
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p8,
            p9,
            p10,
            p11,
            p12,
            p13,
            p14,
            p15,
            thread,
            output_tile,
        ],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reduce_shared16_tid25_f32(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    thread,
    output_tile,
):
    """Let tid 0..23 reduce projections and tid 24 reduce sqsum."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq, active;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          .reg .f32 v8, v9, v10, v11, v12, v13, v14, v15;
          setp.lt.u32 projection, $17, 24;
          setp.eq.u32 sq_thread, $17, 24;
          setp.eq.u32 tile_zero, $18, 0;
          and.pred sq, sq_thread, tile_zero;
          or.pred active, projection, sq;
          mov.f32 v0, 0f00000000;
          mov.f32 v1, 0f00000000;
          mov.f32 v2, 0f00000000;
          mov.f32 v3, 0f00000000;
          mov.f32 v4, 0f00000000;
          mov.f32 v5, 0f00000000;
          mov.f32 v6, 0f00000000;
          mov.f32 v7, 0f00000000;
          mov.f32 v8, 0f00000000;
          mov.f32 v9, 0f00000000;
          mov.f32 v10, 0f00000000;
          mov.f32 v11, 0f00000000;
          mov.f32 v12, 0f00000000;
          mov.f32 v13, 0f00000000;
          mov.f32 v14, 0f00000000;
          mov.f32 v15, 0f00000000;
          @active ld.shared.f32 v0, [$1];
          @active ld.shared.f32 v1, [$2];
          @active ld.shared.f32 v2, [$3];
          @active ld.shared.f32 v3, [$4];
          @active ld.shared.f32 v4, [$5];
          @active ld.shared.f32 v5, [$6];
          @active ld.shared.f32 v6, [$7];
          @active ld.shared.f32 v7, [$8];
          @active ld.shared.f32 v8, [$9];
          @active ld.shared.f32 v9, [$10];
          @active ld.shared.f32 v10, [$11];
          @active ld.shared.f32 v11, [$12];
          @active ld.shared.f32 v12, [$13];
          @active ld.shared.f32 v13, [$14];
          @active ld.shared.f32 v14, [$15];
          @active ld.shared.f32 v15, [$16];
          add.f32 v0, v0, v8;
          add.f32 v1, v1, v9;
          add.f32 v2, v2, v10;
          add.f32 v3, v3, v11;
          add.f32 v4, v4, v12;
          add.f32 v5, v5, v13;
          add.f32 v6, v6, v14;
          add.f32 v7, v7, v15;
          add.f32 v0, v0, v4;
          add.f32 v1, v1, v5;
          add.f32 v2, v2, v6;
          add.f32 v3, v3, v7;
          add.f32 v0, v0, v2;
          add.f32 v1, v1, v3;
          add.f32 $0, v0, v1;
        }
        """,
        constraints="=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
        args=[
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p8,
            p9,
            p10,
            p11,
            p12,
            p13,
            p14,
            p15,
            thread,
            output_tile,
        ],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reduce_shared8_tid25_f32(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    thread,
    output_tile,
):
    """Let tid 0..23 reduce projections and tid 24 reduce sqsum."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq, active;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          setp.lt.u32 projection, $9, 24;
          setp.eq.u32 sq_thread, $9, 24;
          setp.eq.u32 tile_zero, $10, 0;
          and.pred sq, sq_thread, tile_zero;
          or.pred active, projection, sq;
          mov.f32 v0, 0f00000000;
          mov.f32 v1, 0f00000000;
          mov.f32 v2, 0f00000000;
          mov.f32 v3, 0f00000000;
          mov.f32 v4, 0f00000000;
          mov.f32 v5, 0f00000000;
          mov.f32 v6, 0f00000000;
          mov.f32 v7, 0f00000000;
          @active ld.shared.f32 v0, [$1];
          @active ld.shared.f32 v1, [$2];
          @active ld.shared.f32 v2, [$3];
          @active ld.shared.f32 v3, [$4];
          @active ld.shared.f32 v4, [$5];
          @active ld.shared.f32 v5, [$6];
          @active ld.shared.f32 v6, [$7];
          @active ld.shared.f32 v7, [$8];
          add.f32 v0, v0, v4;
          add.f32 v1, v1, v5;
          add.f32 v2, v2, v6;
          add.f32 v3, v3, v7;
          add.f32 v0, v0, v2;
          add.f32 v1, v1, v3;
          add.f32 $0, v0, v1;
        }
        """,
        constraints="=f,r,r,r,r,r,r,r,r,r,r",
        args=[
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            thread,
            output_tile,
        ],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reduce_shared8_tid13_f32(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    thread,
    output_tile,
):
    """Let tid 0..11 reduce projection rows and tid 12 reduce sqsum."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq, active;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          setp.lt.u32 projection, $9, 12;
          setp.eq.u32 sq_thread, $9, 12;
          setp.eq.u32 tile_zero, $10, 0;
          and.pred sq, sq_thread, tile_zero;
          or.pred active, projection, sq;
          mov.f32 v0, 0f00000000;
          mov.f32 v1, 0f00000000;
          mov.f32 v2, 0f00000000;
          mov.f32 v3, 0f00000000;
          mov.f32 v4, 0f00000000;
          mov.f32 v5, 0f00000000;
          mov.f32 v6, 0f00000000;
          mov.f32 v7, 0f00000000;
          @active ld.shared.f32 v0, [$1];
          @active ld.shared.f32 v1, [$2];
          @active ld.shared.f32 v2, [$3];
          @active ld.shared.f32 v3, [$4];
          @active ld.shared.f32 v4, [$5];
          @active ld.shared.f32 v5, [$6];
          @active ld.shared.f32 v6, [$7];
          @active ld.shared.f32 v7, [$8];
          add.f32 v0, v0, v4;
          add.f32 v1, v1, v5;
          add.f32 v2, v2, v6;
          add.f32 v3, v3, v7;
          add.f32 v0, v0, v2;
          add.f32 v1, v1, v3;
          add.f32 $0, v0, v1;
        }
        """,
        constraints="=f,r,r,r,r,r,r,r,r,r,r",
        args=[
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            thread,
            output_tile,
        ],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reduce_shared8_tid7_f32(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    thread,
    output_tile,
):
    """Let tid 0..5 reduce projection rows and tid 6 reduce sqsum."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq, active;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          setp.lt.u32 projection, $9, 6;
          setp.eq.u32 sq_thread, $9, 6;
          setp.eq.u32 tile_zero, $10, 0;
          and.pred sq, sq_thread, tile_zero;
          or.pred active, projection, sq;
          mov.f32 v0, 0f00000000;
          mov.f32 v1, 0f00000000;
          mov.f32 v2, 0f00000000;
          mov.f32 v3, 0f00000000;
          mov.f32 v4, 0f00000000;
          mov.f32 v5, 0f00000000;
          mov.f32 v6, 0f00000000;
          mov.f32 v7, 0f00000000;
          @active ld.shared.f32 v0, [$1];
          @active ld.shared.f32 v1, [$2];
          @active ld.shared.f32 v2, [$3];
          @active ld.shared.f32 v3, [$4];
          @active ld.shared.f32 v4, [$5];
          @active ld.shared.f32 v5, [$6];
          @active ld.shared.f32 v6, [$7];
          @active ld.shared.f32 v7, [$8];
          add.f32 v0, v0, v4;
          add.f32 v1, v1, v5;
          add.f32 v2, v2, v6;
          add.f32 v3, v3, v7;
          add.f32 v0, v0, v2;
          add.f32 v1, v1, v3;
          add.f32 $0, v0, v1;
        }
        """,
        constraints="=f,r,r,r,r,r,r,r,r,r,r",
        args=[
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            thread,
            output_tile,
        ],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reduce_shared8_tid9_f32(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    thread,
    output_tile,
):
    """Let tid 0..7 reduce projection rows and tid 8 reduce sqsum."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq, active;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          setp.lt.u32 projection, $9, 8;
          setp.eq.u32 sq_thread, $9, 8;
          setp.eq.u32 tile_zero, $10, 0;
          and.pred sq, sq_thread, tile_zero;
          or.pred active, projection, sq;
          mov.f32 v0, 0f00000000;
          mov.f32 v1, 0f00000000;
          mov.f32 v2, 0f00000000;
          mov.f32 v3, 0f00000000;
          mov.f32 v4, 0f00000000;
          mov.f32 v5, 0f00000000;
          mov.f32 v6, 0f00000000;
          mov.f32 v7, 0f00000000;
          @active ld.shared.f32 v0, [$1];
          @active ld.shared.f32 v1, [$2];
          @active ld.shared.f32 v2, [$3];
          @active ld.shared.f32 v3, [$4];
          @active ld.shared.f32 v4, [$5];
          @active ld.shared.f32 v5, [$6];
          @active ld.shared.f32 v6, [$7];
          @active ld.shared.f32 v7, [$8];
          add.f32 v0, v0, v4;
          add.f32 v1, v1, v5;
          add.f32 v2, v2, v6;
          add.f32 v3, v3, v7;
          add.f32 v0, v0, v2;
          add.f32 v1, v1, v3;
          add.f32 $0, v0, v1;
        }
        """,
        constraints="=f,r,r,r,r,r,r,r,r,r,r",
        args=[
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            thread,
            output_tile,
        ],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_tid4_partials_f32(
    projection_base,
    sqsum_base,
    value,
    thread,
    output_tile,
):
    """Store three projection values and the tile-zero sqsum from tid 0..3."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq;
          .reg .u64 byte_offset, address;
          setp.lt.u32 projection, $4, 3;
          mul.wide.u32 byte_offset, $4, 4;
          add.u64 address, $1, byte_offset;
          @projection st.global.f32 [address], $3;
          setp.eq.u32 sq_thread, $4, 3;
          setp.eq.u32 tile_zero, $5, 0;
          and.pred sq, sq_thread, tile_zero;
          @sq st.global.f32 [$2], $3;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,f,r,r",
        args=[projection_base, sqsum_base, value, thread, output_tile],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_tid25_partials_f32(
    projection_base,
    sqsum_base,
    value,
    thread,
    output_tile,
):
    """Store 24 projection values and tile-zero sqsum from tid 0..24."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq;
          .reg .u64 byte_offset, address;
          setp.lt.u32 projection, $4, 24;
          mul.wide.u32 byte_offset, $4, 4;
          add.u64 address, $1, byte_offset;
          @projection st.global.f32 [address], $3;
          setp.eq.u32 sq_thread, $4, 24;
          setp.eq.u32 tile_zero, $5, 0;
          and.pred sq, sq_thread, tile_zero;
          @sq st.global.f32 [$2], $3;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,f,r,r",
        args=[projection_base, sqsum_base, value, thread, output_tile],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_tid25_packed_partials_f32(
    projection_base,
    value,
    thread,
):
    """Store projection lanes 0..23 and sqsum lane 24 in one instruction."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active;
          .reg .u64 byte_offset, address;
          setp.lt.u32 active, $3, 25;
          mul.wide.u32 byte_offset, $3, 4;
          add.u64 address, $1, byte_offset;
          @active st.global.f32 [address], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r",
        args=[projection_base, value, thread],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_tid5_partials_f32(
    projection_base,
    sqsum_base,
    value,
    thread,
    output_tile,
):
    """Store four projection values and the tile-zero sqsum from tid 0..4."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq;
          .reg .u64 byte_offset, address;
          setp.lt.u32 projection, $4, 4;
          mul.wide.u32 byte_offset, $4, 4;
          add.u64 address, $1, byte_offset;
          @projection st.global.f32 [address], $3;
          setp.eq.u32 sq_thread, $4, 4;
          setp.eq.u32 tile_zero, $5, 0;
          and.pred sq, sq_thread, tile_zero;
          @sq st.global.f32 [$2], $3;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,f,r,r",
        args=[projection_base, sqsum_base, value, thread, output_tile],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_tid13_partials_f32(
    projection_base,
    sqsum_base,
    value,
    thread,
    output_tile,
):
    """Store 12 projection values and tile-zero sqsum from tid 0..12."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq;
          .reg .u64 byte_offset, address;
          setp.lt.u32 projection, $4, 12;
          mul.wide.u32 byte_offset, $4, 4;
          add.u64 address, $1, byte_offset;
          @projection st.global.f32 [address], $3;
          setp.eq.u32 sq_thread, $4, 12;
          setp.eq.u32 tile_zero, $5, 0;
          and.pred sq, sq_thread, tile_zero;
          @sq st.global.f32 [$2], $3;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,f,r,r",
        args=[projection_base, sqsum_base, value, thread, output_tile],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_tid7_partials_f32(
    projection_base,
    sqsum_base,
    value,
    thread,
    output_tile,
):
    """Store six projection values and tile-zero sqsum from tid 0..6."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq;
          .reg .u64 byte_offset, address;
          setp.lt.u32 projection, $4, 6;
          mul.wide.u32 byte_offset, $4, 4;
          add.u64 address, $1, byte_offset;
          @projection st.global.f32 [address], $3;
          setp.eq.u32 sq_thread, $4, 6;
          setp.eq.u32 tile_zero, $5, 0;
          and.pred sq, sq_thread, tile_zero;
          @sq st.global.f32 [$2], $3;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,f,r,r",
        args=[projection_base, sqsum_base, value, thread, output_tile],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_tid9_partials_f32(
    projection_base,
    sqsum_base,
    value,
    thread,
    output_tile,
):
    """Store eight projection values and tile-zero sqsum from tid 0..8."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred projection, sq_thread, tile_zero, sq;
          .reg .u64 byte_offset, address;
          setp.lt.u32 projection, $4, 8;
          mul.wide.u32 byte_offset, $4, 4;
          add.u64 address, $1, byte_offset;
          @projection st.global.f32 [address], $3;
          setp.eq.u32 sq_thread, $4, 8;
          setp.eq.u32 tile_zero, $5, 0;
          and.pred sq, sq_thread, tile_zero;
          @sq st.global.f32 [$2], $3;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,f,r,r",
        args=[projection_base, sqsum_base, value, thread, output_tile],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _warp_reduce4_max(value):
    """C++-ordered max over each adjacent four-lane routing row."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t;
          mov.f32 $0, $1;
          shfl.sync.bfly.b32 t, $0, 2, 31, 0xffffffff;
          max.f32 $0, $0, t;
          shfl.sync.bfly.b32 t, $0, 1, 31, 0xffffffff;
          max.f32 $0, $0, t;
        }
        """,
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_reduce4_sum(value):
    """C++-ordered sum over each adjacent four-lane routing row."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t;
          mov.f32 $0, $1;
          shfl.sync.bfly.b32 t, $0, 2, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.bfly.b32 t, $0, 1, 31, 0xffffffff;
          add.f32 $0, $0, t;
        }
        """,
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_reduce4x4_column_sum(value):
    """Collect a 4x4 column with the C++ xor-4/xor-24 order."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t;
          mov.f32 $0, $1;
          shfl.sync.bfly.b32 t, $0, 4, 31, 0xffffffff;
          add.f32 $0, $0, t;
          shfl.sync.bfly.b32 t, $0, 24, 31, 0xffffffff;
          add.f32 $0, $0, t;
        }
        """,
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fast_rcp_ftz_f32(value):
    """Match the SM90 CUDA path's approximate FP32 reciprocal."""
    return tl.inline_asm_elementwise(
        asm="rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _physical_thread_id_x():
    """Return the physical CTA thread id as a per-thread scalar."""
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, %tid.x;",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _release_inc_u32_last_arrival(counter_base, thread, limit):
    """Release-publish one CTA and elect the configured final arrival."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, winner;
          .reg .u32 old;
          setp.eq.u32 leader, $2, 0;
          mov.u32 old, 0;
          @leader atom.release.gpu.global.inc.u32 old, [$1], $3;
          setp.eq.u32 winner, old, $3;
          and.pred winner, winner, leader;
          @winner fence.acquire.gpu;
          selp.u32 $0, 1, 0, winner;
        }
        """,
        constraints="=r,l,r,r",
        args=[counter_base, thread, limit],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _warp0_publish_release_inc_u32_last_arrival(
    counter_base,
    thread,
    limit,
    partial_store_token,
):
    """Publish warp-0 stores, then let lane 0 release-increment the ticket."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred warp0, published, participant, leader, winner;
          .reg .u32 old;
          setp.lt.u32 warp0, $2, 32;
          setp.eq.u32 published, $4, 0;
          and.pred participant, warp0, published;
          @participant barrier.cta.sync 14, 32;
          setp.eq.u32 leader, $2, 0;
          and.pred leader, leader, published;
          mov.u32 old, 0;
          @leader atom.release.gpu.global.inc.u32 old, [$1], $3;
          setp.eq.u32 winner, old, $3;
          and.pred winner, winner, leader;
          @winner fence.acquire.gpu;
          selp.u32 $0, 1, 0, winner;
        }
        """,
        constraints="=r,l,r,r,r",
        args=[counter_base, thread, limit, partial_store_token],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _bar_red_or_pred_u32_256(predicate):
    """Broadcast one winner predicate across an eight-warp CTA."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred input, result;
          setp.ne.u32 input, $1, 0;
          bar.red.or.pred result, 15, 256, input;
          selp.u32 $0, 1, 0, result;
        }
        """,
        constraints="=r,r",
        args=[predicate],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _bar_red_or_pred_u32_512(predicate):
    """Broadcast one winner predicate across a sixteen-warp CTA."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred input, result;
          setp.ne.u32 input, $1, 0;
          bar.red.or.pred result, 15, 512, input;
          selp.u32 $0, 1, 0, result;
        }
        """,
        constraints="=r,r",
        args=[predicate],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _tle_mhc_overlap_hidden_tail(
    mixes_ready,
    pre_ready,
    pre_shared,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Eight-warp default partition: publish Mix, then run hidden tail."""
    # The default partition owns 256 participating threads.  Its non-blocking
    # arrival publishes the Mix shared stores performed before entering the
    # warp-specialized region.  The routing worker's 32-thread sync completes
    # this 288-thread logical barrier.
    tle.gpu.barrier_arrive(mixes_ready)

    # Conversely, the routing worker publishes four Pre coefficients with a
    # 32-thread arrival while these 256 threads wait.  Once released, the
    # hidden path is independent of the worker's long Sinkhorn loop.
    tle.gpu.barrier_wait(pre_ready)
    h = tl.arange(0, JIT_FULL_HIDDEN)
    norm_input = tl.zeros((JIT_FULL_HIDDEN,), tl.float32)
    for channel in tl.static_range(0, JIT_FULL_HC_MULT):
        value = tl.load(
            new_residual_ptr + channel * JIT_FULL_HIDDEN + h
        ).to(tl.float32)
        pre_scalar = tl.load(tle.gpu.local_ptr(pre_shared, (channel,)))
        norm_input += value * pre_scalar
    norm_input_bf16 = norm_input.to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=0)
        / JIT_FULL_HIDDEN
        + rmsnorm_eps
    )
    rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + h,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
    )


@triton.jit
def _tle_mhc_overlap_routing_tail(
    mixes_ready,
    pre_ready,
    mixes_shared,
    pre_shared,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    USE_FAST_RCP: tl.constexpr,
):
    """One logical worker warp: publish Pre, then finish routing."""
    lane = tl.inline_asm_elementwise(
        asm="mov.u32 $0, %laneid;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    projection_mask = lane < JIT_FULL_PROJECTION_OUTPUTS
    tle.gpu.barrier_wait(mixes_ready)
    mixes = tl.load(
        tle.gpu.local_ptr(mixes_shared, (lane,)),
        mask=projection_mask,
        other=0.0,
    )
    pre_mix = tl.sigmoid(mixes) + hc_pre_eps
    post_mix = tl.sigmoid(mixes) * hc_post_scale

    # Scalar TLE stores are lane-zero elected.  The worker deliberately uses
    # scalar values to avoid a root-layout block tensor, so issue the two
    # per-lane stores in PTX and predicate them with the physical lane id.
    pre_pointer = tle.gpu.local_ptr(pre_shared, (lane,))
    pre_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[pre_pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred p;
          setp.lt.u32 p, $3, 4;
          @p st.shared.f32 [$1], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,f,r",
        args=[pre_address, pre_mix, lane],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    post_base = tl.cast(new_post_mix_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred ge4, lt8, p;
          .reg .u32 index;
          .reg .u64 offset, address;
          setp.ge.u32 ge4, $3, 4;
          setp.lt.u32 lt8, $3, 8;
          and.pred p, ge4, lt8;
          sub.u32 index, $3, 4;
          mul.wide.u32 offset, index, 4;
          add.u64 address, $1, offset;
          @p st.global.f32 [address], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r",
        args=[post_base, post_mix, lane],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tle.gpu.barrier_arrive(pre_ready)

    routing_mask = (
        (lane >= 2 * JIT_FULL_HC_MULT)
        & (lane < JIT_FULL_PROJECTION_OUTPUTS)
    )
    sinkhorn = tl.where(routing_mask, mixes, 0.0)
    row_max = _warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _warp_reduce4_sum(sinkhorn)
    if USE_FAST_RCP:
        sinkhorn = sinkhorn * _fast_rcp_ftz_f32(row_sum)
        sinkhorn += sinkhorn_eps
        column_sum = (
            _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
        )
        sinkhorn *= _fast_rcp_ftz_f32(column_sum)
    else:
        sinkhorn = sinkhorn / row_sum + sinkhorn_eps
        sinkhorn = sinkhorn / (
            _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
        )
    for _ in tl.range(1, JIT_FULL_SINKHORN_ITERS):
        row_sum = _warp_reduce4_sum(sinkhorn) + sinkhorn_eps
        if USE_FAST_RCP:
            sinkhorn *= _fast_rcp_ftz_f32(row_sum)
            column_sum = (
                _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
            )
            sinkhorn *= _fast_rcp_ftz_f32(column_sum)
        else:
            sinkhorn = sinkhorn / row_sum
            sinkhorn = sinkhorn / (
                _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
            )
    comb_base = tl.cast(new_comb_res_mix_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred ge8, lt24, p;
          .reg .u32 index;
          .reg .u64 offset, address;
          setp.ge.u32 ge8, $3, 8;
          setp.lt.u32 lt24, $3, 24;
          and.pred p, ge8, lt24;
          sub.u32 index, $3, 8;
          mul.wide.u32 offset, index, 4;
          add.u64 address, $1, offset;
          @p st.global.f32 [address], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r",
        args=[comb_base, sinkhorn, lane],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _read_globaltimer_ns():
    """Read the GPU-wide nanosecond timer for diagnostic specializations."""
    return tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@dataclass(frozen=True)
class MegaMHCAtomicProtocolWorkspace:
    """Preallocated storage owned by one ordered launch/replay sequence.

    ``partials_workspace`` is FP32 ``[T, S, 32]``.  Slots 0:24 hold projection
    partials, slot 24 holds the sum-of-squares value, and the final seven slots
    are deterministic zero padding.  ``counters`` is uint64 ``[T]`` and must
    be zero before first use.  A successful launch advances each token's
    counter by exactly ``S``; it does not reset or wrap the generation.
    """

    partials_workspace: torch.Tensor
    counters: torch.Tensor


@dataclass(frozen=True)
class TLESM90MegaMHCPlan:
    """Prepared exact-one-launch plan for a supported focused migration."""

    x: torch.Tensor
    residual: torch.Tensor
    post_mix: torch.Tensor
    comb_res_mix: torch.Tensor
    fn: torch.Tensor
    mix_scales: torch.Tensor
    mix_bases: torch.Tensor
    rmsnorm_weight: torch.Tensor
    new_residual: torch.Tensor
    new_post_mix: torch.Tensor
    new_comb_res_mix: torch.Tensor
    y_bf16: torch.Tensor
    hc_norm_eps: float
    hc_pre_eps: float
    hc_post_scale: float
    sinkhorn_eps: float
    rmsnorm_eps: float
    rmsnorm_scale: float
    schedule: str
    tokens: int
    num_splits: int
    hidden_per_split: int
    ctas_per_token: int
    counter_stride_elements: int
    projection_partials: torch.Tensor | None
    sqsum_partials: torch.Tensor | None
    arrival_counter: torch.Tensor | None
    phase_profile: torch.Tensor | None
    outputs_per_tile: int
    output_tile_block: int
    num_output_tiles: int
    grid_ctas: int
    split_num_warps: int
    use_hierarchical_reduction: bool
    use_tid4_reduction: bool
    use_tid5_reduction: bool
    use_tid7_reduction: bool
    use_tid9_reduction: bool
    use_tid13_reduction: bool
    use_tid25_reduction: bool
    use_warp_sinkhorn: bool
    use_fast_rcp: bool
    use_warp_overlap: bool
    use_custom_handoff: bool
    use_contiguous_partials: bool
    use_t2_projection_ilp: bool
    use_t2_sqsum_tree: bool
    use_t32_projection_ilp: bool
    use_t32_projection_ilp2: bool
    use_t32_custom_store1: bool
    use_t32_pack_sqsum: bool
    use_t32_skip_epilogue_barrier: bool
    use_t32_premix_direct_bcast: bool
    use_t32_premix_shared_bridge: bool
    use_t32_warp0_handoff_barrier: bool
    use_t32_fn_cache_cg: bool
    use_t32_token_pair: bool
    phase_profiling: bool
    workspace_tensors: dict[str, torch.Tensor]
    counter_tensors: dict[str, torch.Tensor]
    metadata: dict[str, object]

    def run(self) -> None:
        """Enqueue exactly one already-specialized kernel launch."""
        if self.use_t32_token_pair:
            _tle_sm90_mega_mhc_t32_pair_kernel[(self.grid_ctas,)](
                self.x,
                self.residual,
                self.post_mix,
                self.comb_res_mix,
                self.fn,
                self.mix_scales,
                self.mix_bases,
                self.rmsnorm_weight,
                self.new_residual,
                self.new_post_mix,
                self.new_comb_res_mix,
                self.y_bf16,
                self.projection_partials,
                self.sqsum_partials,
                self.arrival_counter,
                self.hc_norm_eps,
                self.hc_pre_eps,
                self.hc_post_scale,
                self.sinkhorn_eps,
                self.rmsnorm_eps,
                self.rmsnorm_scale,
                COUNTER_STRIDE_ELEMENTS=self.counter_stride_elements,
                num_warps=FULL_NUM_WARPS,
                num_stages=1,
                num_ctas=1,
            )
            return
        if self.schedule.startswith("split") and "_tile" in self.schedule:
            _tle_sm90_mega_mhc_t1_split_kernel[(self.grid_ctas,)](
                self.x,
                self.residual,
                self.post_mix,
                self.comb_res_mix,
                self.fn,
                self.mix_scales,
                self.mix_bases,
                self.rmsnorm_weight,
                self.new_residual,
                self.new_post_mix,
                self.new_comb_res_mix,
                self.y_bf16,
                self.projection_partials,
                self.sqsum_partials,
                self.arrival_counter,
                (
                    self.phase_profile
                    if self.phase_profile is not None
                    else self.arrival_counter
                ),
                self.hc_norm_eps,
                self.hc_pre_eps,
                self.hc_post_scale,
                self.sinkhorn_eps,
                self.rmsnorm_eps,
                self.rmsnorm_scale,
                OUTPUTS_PER_TILE=self.outputs_per_tile,
                OUTPUT_TILE_BLOCK=self.output_tile_block,
                NUM_OUTPUT_TILES=self.num_output_tiles,
                TOKENS=self.tokens,
                NUM_SPLITS=self.num_splits,
                HIDDEN_PER_SPLIT=self.hidden_per_split,
                CTAS_PER_TOKEN=self.ctas_per_token,
                TOTAL_CTAS=self.grid_ctas,
                COUNTER_STRIDE_ELEMENTS=self.counter_stride_elements,
                CTA_THREADS_PER_BLOCK=self.split_num_warps * 32,
                USE_HIERARCHICAL_REDUCTION=self.use_hierarchical_reduction,
                USE_TID4_REDUCTION=self.use_tid4_reduction,
                USE_TID5_REDUCTION=self.use_tid5_reduction,
                USE_TID7_REDUCTION=self.use_tid7_reduction,
                USE_TID9_REDUCTION=self.use_tid9_reduction,
                USE_TID13_REDUCTION=self.use_tid13_reduction,
                USE_TID25_REDUCTION=self.use_tid25_reduction,
                USE_WARP_SINKHORN=self.use_warp_sinkhorn,
                USE_FAST_RCP=self.use_fast_rcp,
                USE_WARP_OVERLAP=self.use_warp_overlap,
                USE_CUSTOM_HANDOFF=self.use_custom_handoff,
                USE_CONTIGUOUS_PARTIALS=self.use_contiguous_partials,
                USE_T2_PROJECTION_ILP=self.use_t2_projection_ilp,
                USE_T2_SQSUM_TREE=self.use_t2_sqsum_tree,
                USE_T32_PROJECTION_ILP=self.use_t32_projection_ilp,
                USE_T32_PROJECTION_ILP2=self.use_t32_projection_ilp2,
                USE_T32_CUSTOM_STORE1=self.use_t32_custom_store1,
                USE_T32_PACK_SQSUM=self.use_t32_pack_sqsum,
                USE_T32_SKIP_EPILOGUE_BARRIER=(
                    self.use_t32_skip_epilogue_barrier
                ),
                USE_T32_PREMIX_DIRECT_BCAST=(
                    self.use_t32_premix_direct_bcast
                ),
                USE_T32_PREMIX_SHARED_BRIDGE=(
                    self.use_t32_premix_shared_bridge
                ),
                USE_T32_WARP0_HANDOFF_BARRIER=(
                    self.use_t32_warp0_handoff_barrier
                ),
                USE_T32_FN_CACHE_CG=self.use_t32_fn_cache_cg,
                ENABLE_PHASE_PROFILING=self.phase_profiling,
                num_warps=self.split_num_warps,
                num_stages=1,
                num_ctas=1,
            )
            return
        _tle_sm90_mega_mhc_t1_single_cta_kernel[(1,)](
            self.x,
            self.residual,
            self.post_mix,
            self.comb_res_mix,
            self.fn,
            self.mix_scales,
            self.mix_bases,
            self.rmsnorm_weight,
            self.new_residual,
            self.new_post_mix,
            self.new_comb_res_mix,
            self.y_bf16,
            self.hc_norm_eps,
            self.hc_pre_eps,
            self.hc_post_scale,
            self.sinkhorn_eps,
            self.rmsnorm_eps,
            self.rmsnorm_scale,
            num_warps=FULL_NUM_WARPS,
            num_stages=1,
            num_ctas=1,
        )


@triton.jit
def _tle_sm90_mega_mhc_t1_split_kernel(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    rmsnorm_weight_ptr,
    new_residual_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    y_bf16_ptr,
    projection_partials_ptr,
    sqsum_partials_ptr,
    arrival_counter_ptr,
    phase_profile_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
    OUTPUTS_PER_TILE: tl.constexpr,
    OUTPUT_TILE_BLOCK: tl.constexpr,
    NUM_OUTPUT_TILES: tl.constexpr,
    TOKENS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HIDDEN_PER_SPLIT: tl.constexpr,
    CTAS_PER_TOKEN: tl.constexpr,
    TOTAL_CTAS: tl.constexpr,
    COUNTER_STRIDE_ELEMENTS: tl.constexpr,
    CTA_THREADS_PER_BLOCK: tl.constexpr,
    USE_HIERARCHICAL_REDUCTION: tl.constexpr,
    USE_TID4_REDUCTION: tl.constexpr,
    USE_TID5_REDUCTION: tl.constexpr,
    USE_TID7_REDUCTION: tl.constexpr,
    USE_TID9_REDUCTION: tl.constexpr,
    USE_TID13_REDUCTION: tl.constexpr,
    USE_TID25_REDUCTION: tl.constexpr,
    USE_WARP_SINKHORN: tl.constexpr,
    USE_FAST_RCP: tl.constexpr,
    USE_WARP_OVERLAP: tl.constexpr,
    USE_CUSTOM_HANDOFF: tl.constexpr,
    USE_CONTIGUOUS_PARTIALS: tl.constexpr,
    USE_T2_PROJECTION_ILP: tl.constexpr,
    USE_T2_SQSUM_TREE: tl.constexpr,
    USE_T32_PROJECTION_ILP: tl.constexpr,
    USE_T32_PROJECTION_ILP2: tl.constexpr,
    USE_T32_CUSTOM_STORE1: tl.constexpr,
    USE_T32_PACK_SQSUM: tl.constexpr,
    USE_T32_SKIP_EPILOGUE_BARRIER: tl.constexpr,
    USE_T32_PREMIX_DIRECT_BCAST: tl.constexpr,
    USE_T32_PREMIX_SHARED_BRIDGE: tl.constexpr,
    USE_T32_WARP0_HANDOFF_BARRIER: tl.constexpr,
    USE_T32_FN_CACHE_CG: tl.constexpr,
    ENABLE_PHASE_PROFILING: tl.constexpr,
):
    """Token-local split producers plus in-kernel last-arrival epilogues."""
    pid = tl.program_id(0)
    if ENABLE_PHASE_PROFILING:
        tl.store(phase_profile_ptr + pid * 2, _read_globaltimer_ns())
    if TOKENS == 1:
        token = 0
        token_cta = pid
    else:
        token = pid // CTAS_PER_TOKEN
        token_cta = pid % CTAS_PER_TOKEN
    split = token_cta // NUM_OUTPUT_TILES
    output_tile = token_cta % NUM_OUTPUT_TILES
    local_hidden = tl.arange(0, HIDDEN_PER_SPLIT)
    hidden = split * HIDDEN_PER_SPLIT + local_hidden
    token_hidden_base = token * JIT_FULL_HIDDEN
    token_residual_base = token * JIT_FULL_FLAT_HIDDEN
    token_state_base = token * JIT_FULL_HC_MULT
    token_comb_base = token * JIT_FULL_HC_MULT * JIT_FULL_HC_MULT
    x = tl.load(x_ptr + token_hidden_base + hidden).to(tl.float32)
    r0 = tl.load(residual_ptr + token_residual_base + hidden).to(tl.float32)
    r1 = tl.load(
        residual_ptr + token_residual_base + JIT_FULL_HIDDEN + hidden
    ).to(tl.float32)
    r2 = tl.load(
        residual_ptr + token_residual_base + 2 * JIT_FULL_HIDDEN + hidden
    ).to(tl.float32)
    r3 = tl.load(
        residual_ptr + token_residual_base + 3 * JIT_FULL_HIDDEN + hidden
    ).to(tl.float32)
    q0 = (
        x * tl.load(post_mix_ptr + token_state_base).to(tl.float32)
        + r0 * tl.load(comb_res_mix_ptr + token_comb_base).to(tl.float32)
        + r1 * tl.load(comb_res_mix_ptr + token_comb_base + 4).to(tl.float32)
        + r2 * tl.load(comb_res_mix_ptr + token_comb_base + 8).to(tl.float32)
        + r3 * tl.load(comb_res_mix_ptr + token_comb_base + 12).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q1 = (
        x * tl.load(post_mix_ptr + token_state_base + 1).to(tl.float32)
        + r0 * tl.load(comb_res_mix_ptr + token_comb_base + 1).to(tl.float32)
        + r1 * tl.load(comb_res_mix_ptr + token_comb_base + 5).to(tl.float32)
        + r2 * tl.load(comb_res_mix_ptr + token_comb_base + 9).to(tl.float32)
        + r3 * tl.load(comb_res_mix_ptr + token_comb_base + 13).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q2 = (
        x * tl.load(post_mix_ptr + token_state_base + 2).to(tl.float32)
        + r0 * tl.load(comb_res_mix_ptr + token_comb_base + 2).to(tl.float32)
        + r1 * tl.load(comb_res_mix_ptr + token_comb_base + 6).to(tl.float32)
        + r2 * tl.load(comb_res_mix_ptr + token_comb_base + 10).to(tl.float32)
        + r3 * tl.load(comb_res_mix_ptr + token_comb_base + 14).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q3 = (
        x * tl.load(post_mix_ptr + token_state_base + 3).to(tl.float32)
        + r0 * tl.load(comb_res_mix_ptr + token_comb_base + 3).to(tl.float32)
        + r1 * tl.load(comb_res_mix_ptr + token_comb_base + 7).to(tl.float32)
        + r2 * tl.load(comb_res_mix_ptr + token_comb_base + 11).to(tl.float32)
        + r3 * tl.load(comb_res_mix_ptr + token_comb_base + 15).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q0f = q0.to(tl.float32)
    q1f = q1.to(tl.float32)
    q2f = q2.to(tl.float32)
    q3f = q3.to(tl.float32)
    if output_tile == 0:
        tl.store(new_residual_ptr + token_residual_base + hidden, q0)
        tl.store(
            new_residual_ptr + token_residual_base + JIT_FULL_HIDDEN + hidden,
            q1,
        )
        tl.store(
            new_residual_ptr
            + token_residual_base
            + 2 * JIT_FULL_HIDDEN
            + hidden,
            q2,
        )
        tl.store(
            new_residual_ptr
            + token_residual_base
            + 3 * JIT_FULL_HIDDEN
            + hidden,
            q3,
        )

    if USE_TID25_REDUCTION:
        tl.static_assert(TOKENS == 32)
        tl.static_assert(NUM_SPLITS == 8)
        tl.static_assert(
            CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS
            or CTA_THREADS_PER_BLOCK == 2 * JIT_T8_REDUCTION_THREADS
        )
        tl.static_assert(HIDDEN_PER_SPLIT == 512)
        tl.static_assert(
            OUTPUTS_PER_TILE == JIT_T32_TILE24_REDUCTION_CHANNELS - 1
        )
        tl.static_assert(OUTPUT_TILE_BLOCK == 32)
        tl.static_assert(NUM_OUTPUT_TILES == 1)
        tl.static_assert(CTAS_PER_TOKEN == 8)
        if USE_T32_CUSTOM_STORE1:
            tl.static_assert(
                CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS
            )
            tl.static_assert(not USE_T32_PROJECTION_ILP)
        if USE_T32_PROJECTION_ILP2:
            tl.static_assert(USE_T32_CUSTOM_STORE1)
            tl.static_assert(not USE_T32_PROJECTION_ILP)
        if USE_T32_PACK_SQSUM:
            tl.static_assert(USE_T32_CUSTOM_STORE1)
            tl.static_assert(not USE_T32_PROJECTION_ILP)
            tl.static_assert(not USE_T32_PROJECTION_ILP2)
            tl.static_assert(not USE_T32_FN_CACHE_CG)
        if USE_T32_SKIP_EPILOGUE_BARRIER:
            tl.static_assert(USE_T32_CUSTOM_STORE1)
            tl.static_assert(USE_T32_PACK_SQSUM)
            tl.static_assert(USE_FAST_RCP)
            tl.static_assert(USE_CUSTOM_HANDOFF)
            tl.static_assert(not USE_T32_PROJECTION_ILP)
            tl.static_assert(not USE_T32_PROJECTION_ILP2)
            tl.static_assert(not USE_T32_FN_CACHE_CG)
            tl.static_assert(not USE_WARP_OVERLAP)
            tl.static_assert(not ENABLE_PHASE_PROFILING)
        if USE_T32_PREMIX_DIRECT_BCAST:
            tl.static_assert(USE_T32_CUSTOM_STORE1)
            tl.static_assert(USE_T32_PACK_SQSUM)
            tl.static_assert(USE_FAST_RCP)
            tl.static_assert(USE_CUSTOM_HANDOFF)
            tl.static_assert(not USE_T32_SKIP_EPILOGUE_BARRIER)
            tl.static_assert(not USE_T32_PROJECTION_ILP)
            tl.static_assert(not USE_T32_PROJECTION_ILP2)
            tl.static_assert(not USE_T32_FN_CACHE_CG)
            tl.static_assert(not USE_WARP_OVERLAP)
            tl.static_assert(not ENABLE_PHASE_PROFILING)
        if USE_T32_PREMIX_SHARED_BRIDGE:
            tl.static_assert(USE_T32_PREMIX_DIRECT_BCAST)
            tl.static_assert(CTA_THREADS_PER_BLOCK == 256)
        if USE_T32_WARP0_HANDOFF_BARRIER:
            tl.static_assert(USE_T32_CUSTOM_STORE1)
            tl.static_assert(USE_T32_PACK_SQSUM)
            tl.static_assert(USE_FAST_RCP)
            tl.static_assert(USE_CUSTOM_HANDOFF)
            tl.static_assert(not USE_T32_SKIP_EPILOGUE_BARRIER)
            tl.static_assert(not USE_T32_PREMIX_DIRECT_BCAST)
            tl.static_assert(not USE_T32_PROJECTION_ILP)
            tl.static_assert(not USE_T32_PROJECTION_ILP2)
            tl.static_assert(not USE_T32_FN_CACHE_CG)
            tl.static_assert(not USE_WARP_OVERLAP)
            tl.static_assert(not ENABLE_PHASE_PROFILING)
        if USE_T32_FN_CACHE_CG:
            tl.static_assert(USE_T32_CUSTOM_STORE1)
            tl.static_assert(not USE_T32_PROJECTION_ILP)
            tl.static_assert(not USE_T32_PROJECTION_ILP2)
        # The split-8/tile-24 T=32 candidate maps either two adjacent hidden
        # elements to each of 256 CTA threads or one hidden element to each
        # of 512 threads.  The 8-warp form folds each pair before its warp
        # tree.  Both forms publish all 24 outputs plus sqsum to one shared
        # table and cross exactly one CTA barrier before tid 0..24 reduce.
        reduction_thread = tl.arange(0, CTA_THREADS_PER_BLOCK)
        reduction_lane = reduction_thread % 32
        reduction_warp = reduction_thread // 32
        channel0 = reduction_thread * 0
        scratch25 = tle.gpu.alloc(
            [
                CTA_THREADS_PER_BLOCK // 32,
                JIT_T32_TILE24_REDUCTION_CHANNEL_BLOCK,
            ],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        if USE_T32_PROJECTION_ILP:
            tl.static_assert(
                CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS
            )
            tl.static_assert(OUTPUTS_PER_TILE % 3 == 0)
            tl.static_assert(OUTPUTS_PER_TILE // 3 == 8)
            # Batch three independent projection chains before reducing them.
            # The 8-warp shared25 layout owns two adjacent hidden values per
            # thread, so every channel is pair-folded from 512 values to the
            # 256 physical threads before the three-way warp reduction.
            for output_group in tl.static_range(0, OUTPUTS_PER_TILE, 3):
                output_base = output_tile * OUTPUTS_PER_TILE + output_group
                weight_row0 = fn_ptr + output_base * JIT_FULL_FLAT_HIDDEN
                weight_row1 = (
                    fn_ptr + (output_base + 1) * JIT_FULL_FLAT_HIDDEN
                )
                weight_row2 = (
                    fn_ptr + (output_base + 2) * JIT_FULL_FLAT_HIDDEN
                )

                lane_partial0 = q0f * tl.load(
                    weight_row0 + hidden
                ).to(tl.float32)
                lane_partial1 = q0f * tl.load(
                    weight_row1 + hidden
                ).to(tl.float32)
                lane_partial2 = q0f * tl.load(
                    weight_row2 + hidden
                ).to(tl.float32)
                lane_partial0 += q1f * tl.load(
                    weight_row0 + JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial1 += q1f * tl.load(
                    weight_row1 + JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial2 += q1f * tl.load(
                    weight_row2 + JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial0 += q2f * tl.load(
                    weight_row0 + 2 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial1 += q2f * tl.load(
                    weight_row1 + 2 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial2 += q2f * tl.load(
                    weight_row2 + 2 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial0 += q3f * tl.load(
                    weight_row0 + 3 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial1 += q3f * tl.load(
                    weight_row1 + 3 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial2 += q3f * tl.load(
                    weight_row2 + 3 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial0 = _pair_sum_512_to_256(lane_partial0)
                lane_partial1 = _pair_sum_512_to_256(lane_partial1)
                lane_partial2 = _pair_sum_512_to_256(lane_partial2)
                warp_partial0, warp_partial1, warp_partial2 = _warp_sum3_32(
                    lane_partial0,
                    lane_partial1,
                    lane_partial2,
                )
                shared_store_token = _store_shared3_f32(
                    _shared_address_u32(
                        tle.gpu.local_ptr(
                            scratch25,
                            (reduction_warp, channel0 + output_group + 0),
                        )
                    ),
                    _shared_address_u32(
                        tle.gpu.local_ptr(
                            scratch25,
                            (reduction_warp, channel0 + output_group + 1),
                        )
                    ),
                    _shared_address_u32(
                        tle.gpu.local_ptr(
                            scratch25,
                            (reduction_warp, channel0 + output_group + 2),
                        )
                    ),
                    warp_partial0,
                    warp_partial1,
                    warp_partial2,
                    reduction_lane,
                )
        elif USE_T32_PROJECTION_ILP2:
            tl.static_assert(
                CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS
            )
            tl.static_assert(USE_T32_CUSTOM_STORE1)
            tl.static_assert(OUTPUTS_PER_TILE % 2 == 0)
            tl.static_assert(OUTPUTS_PER_TILE // 2 == 12)
            # Preserve each output's q0 -> q1 -> q2 -> q3 accumulation and
            # pair fold, but interleave the two independent warp trees so
            # Hopper can issue the second shuffle while the first is pending.
            for output_group in tl.static_range(0, OUTPUTS_PER_TILE, 2):
                output_base = output_tile * OUTPUTS_PER_TILE + output_group
                weight_row0 = fn_ptr + output_base * JIT_FULL_FLAT_HIDDEN
                weight_row1 = (
                    fn_ptr + (output_base + 1) * JIT_FULL_FLAT_HIDDEN
                )

                lane_partial0 = q0f * tl.load(
                    weight_row0 + hidden
                ).to(tl.float32)
                lane_partial1 = q0f * tl.load(
                    weight_row1 + hidden
                ).to(tl.float32)
                lane_partial0 += q1f * tl.load(
                    weight_row0 + JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial1 += q1f * tl.load(
                    weight_row1 + JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial0 += q2f * tl.load(
                    weight_row0 + 2 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial1 += q2f * tl.load(
                    weight_row1 + 2 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial0 += q3f * tl.load(
                    weight_row0 + 3 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial1 += q3f * tl.load(
                    weight_row1 + 3 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial0 = _pair_sum_512_to_256(lane_partial0)
                lane_partial1 = _pair_sum_512_to_256(lane_partial1)
                warp_partial0, warp_partial1 = _warp_sum2_interleaved_32(
                    lane_partial0,
                    lane_partial1,
                )
                shared_store_token0 = _store_shared1_f32(
                    _shared_address_u32(
                        tle.gpu.local_ptr(
                            scratch25,
                            (reduction_warp, channel0 + output_group),
                        )
                    ),
                    warp_partial0,
                    reduction_lane,
                )
                shared_store_token1 = _store_shared1_f32(
                    _shared_address_u32(
                        tle.gpu.local_ptr(
                            scratch25,
                            (reduction_warp, channel0 + output_group + 1),
                        )
                    ),
                    warp_partial1,
                    reduction_lane,
                )
        else:
            for output_in_tile in tl.static_range(0, OUTPUTS_PER_TILE):
                output_index = output_tile * OUTPUTS_PER_TILE + output_in_tile
                weight_row = fn_ptr + output_index * JIT_FULL_FLAT_HIDDEN
                if USE_T32_FN_CACHE_CG:
                    w0 = tl.load(
                        weight_row + hidden, cache_modifier=".cg"
                    ).to(tl.float32)
                    w1 = tl.load(
                        weight_row + JIT_FULL_HIDDEN + hidden,
                        cache_modifier=".cg",
                    ).to(tl.float32)
                    w2 = tl.load(
                        weight_row + 2 * JIT_FULL_HIDDEN + hidden,
                        cache_modifier=".cg",
                    ).to(tl.float32)
                    w3 = tl.load(
                        weight_row + 3 * JIT_FULL_HIDDEN + hidden,
                        cache_modifier=".cg",
                    ).to(tl.float32)
                else:
                    w0 = tl.load(weight_row + hidden).to(tl.float32)
                    w1 = tl.load(
                        weight_row + JIT_FULL_HIDDEN + hidden
                    ).to(tl.float32)
                    w2 = tl.load(
                        weight_row + 2 * JIT_FULL_HIDDEN + hidden
                    ).to(tl.float32)
                    w3 = tl.load(
                        weight_row + 3 * JIT_FULL_HIDDEN + hidden
                    ).to(tl.float32)
                lane_partial = (
                    q0f * w0 + q1f * w1 + q2f * w2 + q3f * w3
                )
                if CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS:
                    lane_partial = _pair_sum_512_to_256(lane_partial)
                warp_partial = _warp_sum_32(lane_partial)
                if USE_T32_CUSTOM_STORE1:
                    shared_store_token = _store_shared1_f32(
                        _shared_address_u32(
                            tle.gpu.local_ptr(
                                scratch25,
                                (
                                    reduction_warp,
                                    channel0 + output_in_tile,
                                ),
                            )
                        ),
                        warp_partial,
                        reduction_lane,
                    )
                else:
                    tl.store(
                        tle.gpu.local_ptr(
                            scratch25,
                            (reduction_warp, channel0 + output_in_tile),
                        ),
                        warp_partial,
                        mask=reduction_lane == 0,
                    )
        if output_tile == 0:
            lane_sqsum = q0f * q0f + q1f * q1f + q2f * q2f + q3f * q3f
            if CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS:
                lane_sqsum = _pair_sum_512_to_256(lane_sqsum)
            warp_sqsum = _warp_sum_32(lane_sqsum)
            if USE_T32_CUSTOM_STORE1:
                shared_store_token = _store_shared1_f32(
                    _shared_address_u32(
                        tle.gpu.local_ptr(
                            scratch25,
                            (
                                reduction_warp,
                                channel0 + OUTPUTS_PER_TILE,
                            ),
                        )
                    ),
                    warp_sqsum,
                    reduction_lane,
                )
            else:
                tl.store(
                    tle.gpu.local_ptr(
                        scratch25,
                        (reduction_warp, channel0 + OUTPUTS_PER_TILE),
                    ),
                    warp_sqsum,
                    mask=reduction_lane == 0,
                )

        shared25_ready = tle.gpu.alloc_barrier(
            arrive_count=CTA_THREADS_PER_BLOCK
        )
        tle.gpu.barrier_wait(shared25_ready)
        safe_channel = tl.minimum(
            reduction_thread, JIT_T32_TILE24_REDUCTION_CHANNELS - 1
        )
        scratch25_p0 = _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 0, safe_channel))
        )
        scratch25_p1 = _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 1, safe_channel))
        )
        scratch25_p2 = _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 2, safe_channel))
        )
        scratch25_p3 = _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 3, safe_channel))
        )
        scratch25_p4 = _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 4, safe_channel))
        )
        scratch25_p5 = _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 5, safe_channel))
        )
        scratch25_p6 = _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 6, safe_channel))
        )
        scratch25_p7 = _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 7, safe_channel))
        )
        if CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS:
            reduced = _reduce_shared8_tid25_f32(
                scratch25_p0,
                scratch25_p1,
                scratch25_p2,
                scratch25_p3,
                scratch25_p4,
                scratch25_p5,
                scratch25_p6,
                scratch25_p7,
                reduction_thread,
                output_tile,
            )
        else:
            reduced = _reduce_shared16_tid25_f32(
                scratch25_p0,
                scratch25_p1,
                scratch25_p2,
                scratch25_p3,
                scratch25_p4,
                scratch25_p5,
                scratch25_p6,
                scratch25_p7,
                _shared_address_u32(
                    tle.gpu.local_ptr(
                        scratch25, (channel0 + 8, safe_channel)
                    )
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(
                        scratch25, (channel0 + 9, safe_channel)
                    )
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(
                        scratch25, (channel0 + 10, safe_channel)
                    )
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(
                        scratch25, (channel0 + 11, safe_channel)
                    )
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(
                        scratch25, (channel0 + 12, safe_channel)
                    )
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(
                        scratch25, (channel0 + 13, safe_channel)
                    )
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(
                        scratch25, (channel0 + 14, safe_channel)
                    )
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(
                        scratch25, (channel0 + 15, safe_channel)
                    )
                ),
                reduction_thread,
                output_tile,
            )
        projection_offset = (
            token * NUM_OUTPUT_TILES * NUM_SPLITS
            + output_tile * NUM_SPLITS
            + split
        ) * OUTPUT_TILE_BLOCK
        if USE_T32_PACK_SQSUM:
            # Lane 24 is otherwise padding in the 32-float projection row.
            # Packing it here turns the 24 projection stores and the sqsum
            # store into one predicated global-store instruction and removes
            # all kernel accesses to the separate sqsum ABI allocation.
            partial_store_token = _store_tid25_packed_partials_f32(
                tl.cast(
                    projection_partials_ptr + projection_offset,
                    tl.uint64,
                ),
                reduced,
                reduction_thread,
            )
        else:
            partial_store_token = _store_tid25_partials_f32(
                tl.cast(
                    projection_partials_ptr + projection_offset,
                    tl.uint64,
                ),
                tl.cast(
                    sqsum_partials_ptr + token * NUM_SPLITS + split,
                    tl.uint64,
                ),
                reduced,
                reduction_thread,
                output_tile,
            )
    elif USE_TID4_REDUCTION:
        tl.static_assert(
            CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS
            or CTA_THREADS_PER_BLOCK == 2 * JIT_T8_REDUCTION_THREADS
        )
        tl.static_assert(HIDDEN_PER_SPLIT == 2 * JIT_T8_REDUCTION_THREADS)
        tl.static_assert(
            OUTPUTS_PER_TILE == JIT_T2_TILE3_REDUCTION_CHANNELS - 1
        )
        tl.static_assert(NUM_OUTPUT_TILES == JIT_T2_TILE3_NUM_OUTPUT_TILES)
        # The 8-warp producer folds 512 adjacent hidden values onto 256 lanes;
        # the 16-warp producer maps one hidden value to each of 512 lanes.
        # Both preserve a fixed warp tree followed by an explicit balanced
        # shared-memory tree.  Only four reducer threads are active (three
        # projection channels plus tile-zero sqsum).
        reduction_thread = tl.arange(0, CTA_THREADS_PER_BLOCK)
        reduction_lane = reduction_thread % 32
        reduction_warp = reduction_thread // 32
        channel0 = reduction_thread * 0
        scratch4 = tle.gpu.alloc(
            [
                CTA_THREADS_PER_BLOCK // 32,
                JIT_T2_TILE3_REDUCTION_CHANNEL_BLOCK,
            ],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        if USE_T2_PROJECTION_ILP:
            tl.static_assert(CTA_THREADS_PER_BLOCK == 512)
            output_base = output_tile * OUTPUTS_PER_TILE
            weight_row0 = fn_ptr + (output_base + 0) * JIT_FULL_FLAT_HIDDEN
            weight_row1 = fn_ptr + (output_base + 1) * JIT_FULL_FLAT_HIDDEN
            weight_row2 = fn_ptr + (output_base + 2) * JIT_FULL_FLAT_HIDDEN

            # Keep three independent projection chains live until one inline
            # PTX reduction consumes all of them.  This gives ptxas three
            # independent global-load/FFMA streams to schedule before the
            # first shuffle without retaining twelve weight temporaries.
            lane_partial0 = q0f * tl.load(weight_row0 + hidden).to(tl.float32)
            lane_partial1 = q0f * tl.load(weight_row1 + hidden).to(tl.float32)
            lane_partial2 = q0f * tl.load(weight_row2 + hidden).to(tl.float32)
            lane_partial0 += q1f * tl.load(
                weight_row0 + JIT_FULL_HIDDEN + hidden
            ).to(tl.float32)
            lane_partial1 += q1f * tl.load(
                weight_row1 + JIT_FULL_HIDDEN + hidden
            ).to(tl.float32)
            lane_partial2 += q1f * tl.load(
                weight_row2 + JIT_FULL_HIDDEN + hidden
            ).to(tl.float32)
            lane_partial0 += q2f * tl.load(
                weight_row0 + 2 * JIT_FULL_HIDDEN + hidden
            ).to(tl.float32)
            lane_partial1 += q2f * tl.load(
                weight_row1 + 2 * JIT_FULL_HIDDEN + hidden
            ).to(tl.float32)
            lane_partial2 += q2f * tl.load(
                weight_row2 + 2 * JIT_FULL_HIDDEN + hidden
            ).to(tl.float32)
            lane_partial0 += q3f * tl.load(
                weight_row0 + 3 * JIT_FULL_HIDDEN + hidden
            ).to(tl.float32)
            lane_partial1 += q3f * tl.load(
                weight_row1 + 3 * JIT_FULL_HIDDEN + hidden
            ).to(tl.float32)
            lane_partial2 += q3f * tl.load(
                weight_row2 + 3 * JIT_FULL_HIDDEN + hidden
            ).to(tl.float32)
            warp_partial0, warp_partial1, warp_partial2 = _warp_sum3_32(
                lane_partial0,
                lane_partial1,
                lane_partial2,
            )
            shared_store_token = _store_shared3_f32(
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (reduction_warp, channel0 + 0))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (reduction_warp, channel0 + 1))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (reduction_warp, channel0 + 2))
                ),
                warp_partial0,
                warp_partial1,
                warp_partial2,
                reduction_lane,
            )
        else:
            for output_in_tile in tl.static_range(0, OUTPUTS_PER_TILE):
                output_index = output_tile * OUTPUTS_PER_TILE + output_in_tile
                weight_row = fn_ptr + output_index * JIT_FULL_FLAT_HIDDEN
                w0 = tl.load(weight_row + hidden).to(tl.float32)
                w1 = tl.load(weight_row + JIT_FULL_HIDDEN + hidden).to(tl.float32)
                w2 = tl.load(
                    weight_row + 2 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                w3 = tl.load(
                    weight_row + 3 * JIT_FULL_HIDDEN + hidden
                ).to(tl.float32)
                lane_partial = q0f * w0 + q1f * w1 + q2f * w2 + q3f * w3
                if CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS:
                    lane_partial = _pair_sum_512_to_256(lane_partial)
                warp_partial = _warp_sum_32(lane_partial)
                tl.store(
                    tle.gpu.local_ptr(
                        scratch4,
                        (reduction_warp, channel0 + output_in_tile),
                    ),
                    warp_partial,
                    mask=reduction_lane == 0,
                )
        if output_tile == 0:
            lane_sqsum = q0f * q0f + q1f * q1f + q2f * q2f + q3f * q3f
            if CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS:
                lane_sqsum = _pair_sum_512_to_256(lane_sqsum)
            warp_sqsum = _warp_sum_32(lane_sqsum)
            tl.store(
                tle.gpu.local_ptr(
                    scratch4,
                    (reduction_warp, channel0 + OUTPUTS_PER_TILE),
                ),
                warp_sqsum,
                mask=reduction_lane == 0,
            )

        shared4_ready = tle.gpu.alloc_barrier(
            arrive_count=CTA_THREADS_PER_BLOCK
        )
        tle.gpu.barrier_wait(shared4_ready)
        safe_channel = tl.minimum(
            reduction_thread, JIT_T2_TILE3_REDUCTION_CHANNELS - 1
        )
        if CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS:
            reduced = _reduce_shared8_tid4_f32(
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 0, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 1, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 2, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 3, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 4, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 5, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 6, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 7, safe_channel))
                ),
                reduction_thread,
                output_tile,
            )
        else:
            reduced = _reduce_shared16_tid4_f32(
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 0, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 1, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 2, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 3, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 4, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 5, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 6, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 7, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 8, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 9, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 10, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 11, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 12, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 13, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 14, safe_channel))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch4, (channel0 + 15, safe_channel))
                ),
                reduction_thread,
                output_tile,
            )
        if USE_CONTIGUOUS_PARTIALS:
            tl.static_assert(CTA_THREADS_PER_BLOCK == 512)
            tl.static_assert(NUM_SPLITS == 8)
            projection_offset = (
                (token * NUM_SPLITS + split) * JIT_FULL_PROJECTION_OUTPUTS
                + output_tile * OUTPUTS_PER_TILE
            )
        else:
            projection_offset = (
                token * NUM_OUTPUT_TILES * NUM_SPLITS
                + output_tile * NUM_SPLITS
                + split
            ) * OUTPUT_TILE_BLOCK
        partial_store_token = _store_tid4_partials_f32(
            tl.cast(projection_partials_ptr + projection_offset, tl.uint64),
            tl.cast(
                sqsum_partials_ptr + token * NUM_SPLITS + split,
                tl.uint64,
            ),
            reduced,
            reduction_thread,
            output_tile,
        )
    elif USE_TID13_REDUCTION:
        tl.static_assert(CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS)
        tl.static_assert(
            HIDDEN_PER_SPLIT == JIT_T8_REDUCTION_THREADS
            or HIDDEN_PER_SPLIT == 2 * JIT_T8_REDUCTION_THREADS
        )
        tl.static_assert(OUTPUTS_PER_TILE == JIT_T8_REDUCTION_CHANNELS - 1)
        tl.static_assert(NUM_OUTPUT_TILES == 2)
        # Split-8 owns 512 hidden values and folds each adjacent pair onto one
        # of the 256 physical lanes.  Split-16 owns 256 hidden values, so each
        # lane directly contributes one value.  Both geometries then use the
        # same 32-lane reduction tree.  All 13 channels publish their eight
        # warp partials before one shared CTA phase boundary; tid 0..12 then
        # independently reduce one channel.  The physical [warp, channel]
        # transpose keeps those reducer threads on consecutive shared-memory
        # banks while retaining the logical 13x8 channel-by-warp table.
        reduction_thread = tl.arange(0, JIT_T8_REDUCTION_THREADS)
        reduction_lane = reduction_thread % 32
        reduction_warp = reduction_thread // 32
        channel0 = reduction_thread * 0
        scratch13 = tle.gpu.alloc(
            [JIT_T8_REDUCTION_WARPS, JIT_T8_REDUCTION_CHANNEL_BLOCK],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        for output_in_tile in tl.static_range(0, OUTPUTS_PER_TILE):
            output_index = output_tile * OUTPUTS_PER_TILE + output_in_tile
            weight_row = fn_ptr + output_index * JIT_FULL_FLAT_HIDDEN
            w0 = tl.load(weight_row + hidden).to(tl.float32)
            w1 = tl.load(weight_row + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            w2 = tl.load(weight_row + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            w3 = tl.load(weight_row + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            lane_partial = q0f * w0 + q1f * w1 + q2f * w2 + q3f * w3
            if HIDDEN_PER_SPLIT == 2 * JIT_T8_REDUCTION_THREADS:
                lane_partial = _pair_sum_512_to_256(lane_partial)
            warp_partial = _warp_sum_32(lane_partial)
            tl.store(
                tle.gpu.local_ptr(
                    scratch13,
                    (reduction_warp, channel0 + output_in_tile),
                ),
                warp_partial,
                mask=reduction_lane == 0,
            )
        if output_tile == 0:
            lane_sqsum = q0f * q0f + q1f * q1f + q2f * q2f + q3f * q3f
            if HIDDEN_PER_SPLIT == 2 * JIT_T8_REDUCTION_THREADS:
                lane_sqsum = _pair_sum_512_to_256(lane_sqsum)
            warp_sqsum = _warp_sum_32(lane_sqsum)
            tl.store(
                tle.gpu.local_ptr(
                    scratch13,
                    (reduction_warp, channel0 + OUTPUTS_PER_TILE),
                ),
                warp_sqsum,
                mask=reduction_lane == 0,
            )

        shared13_ready = tle.gpu.alloc_barrier(
            arrive_count=CTA_THREADS_PER_BLOCK
        )
        tle.gpu.barrier_wait(shared13_ready)
        safe_channel = tl.minimum(
            reduction_thread, JIT_T8_REDUCTION_CHANNELS - 1
        )
        reduced = _reduce_shared8_tid13_f32(
            _shared_address_u32(
                tle.gpu.local_ptr(scratch13, (channel0 + 0, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch13, (channel0 + 1, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch13, (channel0 + 2, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch13, (channel0 + 3, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch13, (channel0 + 4, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch13, (channel0 + 5, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch13, (channel0 + 6, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch13, (channel0 + 7, safe_channel))
            ),
            reduction_thread,
            output_tile,
        )
        projection_offset = (
            token * NUM_OUTPUT_TILES * NUM_SPLITS
            + output_tile * NUM_SPLITS
            + split
        ) * OUTPUT_TILE_BLOCK
        partial_store_token = _store_tid13_partials_f32(
            tl.cast(projection_partials_ptr + projection_offset, tl.uint64),
            tl.cast(
                sqsum_partials_ptr + token * NUM_SPLITS + split,
                tl.uint64,
            ),
            reduced,
            reduction_thread,
            output_tile,
        )
    elif USE_TID9_REDUCTION:
        tl.static_assert(CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS)
        tl.static_assert(HIDDEN_PER_SPLIT == 2 * JIT_T8_REDUCTION_THREADS)
        tl.static_assert(
            OUTPUTS_PER_TILE == JIT_T8_TILE8_REDUCTION_CHANNELS - 1
        )
        tl.static_assert(NUM_OUTPUT_TILES == JIT_T8_TILE8_NUM_OUTPUT_TILES)
        # The split-8/tile-8 producer owns 512 adjacent hidden values on 256
        # physical lanes.  Fold each adjacent pair, reduce within each warp,
        # and publish the resulting [8 warps, 9 channels] table in one
        # physical [8, 16] shared-memory tile.  A single explicit CTA barrier
        # makes all ordinary shared stores visible before tid 0..8 each reduce
        # one channel across the eight warp partials.
        reduction_thread = tl.arange(0, JIT_T8_REDUCTION_THREADS)
        reduction_lane = reduction_thread % 32
        reduction_warp = reduction_thread // 32
        channel0 = reduction_thread * 0
        scratch9 = tle.gpu.alloc(
            [JIT_T8_REDUCTION_WARPS, JIT_T8_TILE8_REDUCTION_CHANNEL_BLOCK],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        for output_in_tile in tl.static_range(0, OUTPUTS_PER_TILE):
            output_index = output_tile * OUTPUTS_PER_TILE + output_in_tile
            weight_row = fn_ptr + output_index * JIT_FULL_FLAT_HIDDEN
            w0 = tl.load(weight_row + hidden).to(tl.float32)
            w1 = tl.load(weight_row + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            w2 = tl.load(weight_row + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            w3 = tl.load(weight_row + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            pair_partial = _pair_sum_512_to_256(
                q0f * w0 + q1f * w1 + q2f * w2 + q3f * w3
            )
            warp_partial = _warp_sum_32(pair_partial)
            tl.store(
                tle.gpu.local_ptr(
                    scratch9,
                    (reduction_warp, channel0 + output_in_tile),
                ),
                warp_partial,
                mask=reduction_lane == 0,
            )
        if output_tile == 0:
            pair_sqsum = _pair_sum_512_to_256(
                q0f * q0f + q1f * q1f + q2f * q2f + q3f * q3f
            )
            warp_sqsum = _warp_sum_32(pair_sqsum)
            tl.store(
                tle.gpu.local_ptr(
                    scratch9,
                    (reduction_warp, channel0 + OUTPUTS_PER_TILE),
                ),
                warp_sqsum,
                mask=reduction_lane == 0,
            )

        shared9_ready = tle.gpu.alloc_barrier(
            arrive_count=CTA_THREADS_PER_BLOCK
        )
        tle.gpu.barrier_wait(shared9_ready)
        safe_channel = tl.minimum(
            reduction_thread, JIT_T8_TILE8_REDUCTION_CHANNELS - 1
        )
        reduced = _reduce_shared8_tid9_f32(
            _shared_address_u32(
                tle.gpu.local_ptr(scratch9, (channel0 + 0, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch9, (channel0 + 1, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch9, (channel0 + 2, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch9, (channel0 + 3, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch9, (channel0 + 4, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch9, (channel0 + 5, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch9, (channel0 + 6, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch9, (channel0 + 7, safe_channel))
            ),
            reduction_thread,
            output_tile,
        )
        projection_offset = (
            token * NUM_OUTPUT_TILES * NUM_SPLITS
            + output_tile * NUM_SPLITS
            + split
        ) * OUTPUT_TILE_BLOCK
        partial_store_token = _store_tid9_partials_f32(
            tl.cast(projection_partials_ptr + projection_offset, tl.uint64),
            tl.cast(
                sqsum_partials_ptr + token * NUM_SPLITS + split,
                tl.uint64,
            ),
            reduced,
            reduction_thread,
            output_tile,
        )
    elif USE_TID7_REDUCTION:
        tl.static_assert(CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS)
        tl.static_assert(HIDDEN_PER_SPLIT == 2 * JIT_T8_REDUCTION_THREADS)
        tl.static_assert(
            OUTPUTS_PER_TILE == JIT_T8_TILE6_REDUCTION_CHANNELS - 1
        )
        tl.static_assert(NUM_OUTPUT_TILES == JIT_T8_TILE6_NUM_OUTPUT_TILES)
        # The split-8/tile-6 producer owns 512 adjacent hidden values on 256
        # physical lanes.  Fold each adjacent pair, reduce within each warp,
        # and publish the resulting [8 warps, 7 channels] table in one
        # physical [8, 8] shared-memory tile.  A single explicit CTA barrier
        # makes all ordinary shared stores visible before tid 0..6 each reduce
        # one channel across the eight warp partials.
        reduction_thread = tl.arange(0, JIT_T8_REDUCTION_THREADS)
        reduction_lane = reduction_thread % 32
        reduction_warp = reduction_thread // 32
        channel0 = reduction_thread * 0
        scratch7 = tle.gpu.alloc(
            [JIT_T8_REDUCTION_WARPS, JIT_T8_TILE6_REDUCTION_CHANNEL_BLOCK],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        for output_in_tile in tl.static_range(0, OUTPUTS_PER_TILE):
            output_index = output_tile * OUTPUTS_PER_TILE + output_in_tile
            weight_row = fn_ptr + output_index * JIT_FULL_FLAT_HIDDEN
            w0 = tl.load(weight_row + hidden).to(tl.float32)
            w1 = tl.load(weight_row + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            w2 = tl.load(weight_row + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            w3 = tl.load(weight_row + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            pair_partial = _pair_sum_512_to_256(
                q0f * w0 + q1f * w1 + q2f * w2 + q3f * w3
            )
            warp_partial = _warp_sum_32(pair_partial)
            tl.store(
                tle.gpu.local_ptr(
                    scratch7,
                    (reduction_warp, channel0 + output_in_tile),
                ),
                warp_partial,
                mask=reduction_lane == 0,
            )
        if output_tile == 0:
            pair_sqsum = _pair_sum_512_to_256(
                q0f * q0f + q1f * q1f + q2f * q2f + q3f * q3f
            )
            warp_sqsum = _warp_sum_32(pair_sqsum)
            tl.store(
                tle.gpu.local_ptr(
                    scratch7,
                    (reduction_warp, channel0 + OUTPUTS_PER_TILE),
                ),
                warp_sqsum,
                mask=reduction_lane == 0,
            )

        shared7_ready = tle.gpu.alloc_barrier(
            arrive_count=CTA_THREADS_PER_BLOCK
        )
        tle.gpu.barrier_wait(shared7_ready)
        safe_channel = tl.minimum(
            reduction_thread, JIT_T8_TILE6_REDUCTION_CHANNELS - 1
        )
        reduced = _reduce_shared8_tid7_f32(
            _shared_address_u32(
                tle.gpu.local_ptr(scratch7, (channel0 + 0, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch7, (channel0 + 1, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch7, (channel0 + 2, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch7, (channel0 + 3, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch7, (channel0 + 4, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch7, (channel0 + 5, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch7, (channel0 + 6, safe_channel))
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(scratch7, (channel0 + 7, safe_channel))
            ),
            reduction_thread,
            output_tile,
        )
        projection_offset = (
            token * NUM_OUTPUT_TILES * NUM_SPLITS
            + output_tile * NUM_SPLITS
            + split
        ) * OUTPUT_TILE_BLOCK
        partial_store_token = _store_tid7_partials_f32(
            tl.cast(projection_partials_ptr + projection_offset, tl.uint64),
            tl.cast(
                sqsum_partials_ptr + token * NUM_SPLITS + split,
                tl.uint64,
            ),
            reduced,
            reduction_thread,
            output_tile,
        )
    elif USE_HIERARCHICAL_REDUCTION:
        output_base = output_tile * 4
        weight_row0 = fn_ptr + (output_base + 0) * JIT_FULL_FLAT_HIDDEN
        weight_row1 = fn_ptr + (output_base + 1) * JIT_FULL_FLAT_HIDDEN
        weight_row2 = fn_ptr + (output_base + 2) * JIT_FULL_FLAT_HIDDEN
        weight_row3 = fn_ptr + (output_base + 3) * JIT_FULL_FLAT_HIDDEN
        v0 = (
            q0f * tl.load(weight_row0 + hidden).to(tl.float32)
            + q1f
            * tl.load(weight_row0 + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q2f
            * tl.load(weight_row0 + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q3f
            * tl.load(weight_row0 + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
        )
        v1 = (
            q0f * tl.load(weight_row1 + hidden).to(tl.float32)
            + q1f
            * tl.load(weight_row1 + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q2f
            * tl.load(weight_row1 + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q3f
            * tl.load(weight_row1 + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
        )
        v2 = (
            q0f * tl.load(weight_row2 + hidden).to(tl.float32)
            + q1f
            * tl.load(weight_row2 + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q2f
            * tl.load(weight_row2 + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q3f
            * tl.load(weight_row2 + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
        )
        v3 = (
            q0f * tl.load(weight_row3 + hidden).to(tl.float32)
            + q1f
            * tl.load(weight_row3 + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q2f
            * tl.load(weight_row3 + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q3f
            * tl.load(weight_row3 + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
        )
        vs = q0f * q0f + q1f * q1f + q2f * q2f + q3f * q3f
        s0 = _warp_sum_32(v0)
        s1 = _warp_sum_32(v1)
        s2 = _warp_sum_32(v2)
        s3 = _warp_sum_32(v3)
        ss = _warp_sum_32(vs)

        lane = local_hidden % 32
        warp = local_hidden // 32
        scratch = tle.gpu.alloc(
            [8, 8],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        channel0 = local_hidden * 0
        if USE_TID5_REDUCTION:
            shared_store_token = _store_shared5_f32(
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (channel0 + 0, warp))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (channel0 + 1, warp))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (channel0 + 2, warp))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (channel0 + 3, warp))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (channel0 + 4, warp))
                ),
                s0,
                s1,
                s2,
                s3,
                ss,
                lane,
            )
        else:
            tl.store(
                tle.gpu.local_ptr(scratch, (channel0 + 0, warp)),
                s0,
                mask=lane == 0,
            )
            tl.store(
                tle.gpu.local_ptr(scratch, (channel0 + 1, warp)),
                s1,
                mask=lane == 0,
            )
            tl.store(
                tle.gpu.local_ptr(scratch, (channel0 + 2, warp)),
                s2,
                mask=lane == 0,
            )
            tl.store(
                tle.gpu.local_ptr(scratch, (channel0 + 3, warp)),
                s3,
                mask=lane == 0,
            )
            tl.store(
                tle.gpu.local_ptr(scratch, (channel0 + 4, warp)),
                ss,
                mask=lane == 0,
            )

        # The five shared loads below are emitted through inline PTX, so the
        # TLE local-memory dependence pass cannot be relied on to discover
        # their RAW relationship with these stores.  Keep one explicit CTA
        # phase boundary before warp 0 consumes all eight warp partials.
        hierarchical_ready = tle.gpu.alloc_barrier(
            arrive_count=CTA_THREADS_PER_BLOCK
        )
        tle.gpu.barrier_wait(hierarchical_ready)

        projection_offset = (
            (
                token * NUM_OUTPUT_TILES * NUM_SPLITS
                + output_tile * NUM_SPLITS
                + split
            )
        ) * 4
        if USE_TID5_REDUCTION:
            thread = local_hidden
            reduced = _reduce_shared8_f32(
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (thread, channel0 + 0))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (thread, channel0 + 1))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (thread, channel0 + 2))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (thread, channel0 + 3))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (thread, channel0 + 4))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (thread, channel0 + 5))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (thread, channel0 + 6))
                ),
                _shared_address_u32(
                    tle.gpu.local_ptr(scratch, (thread, channel0 + 7))
                ),
                thread,
            )
            partial_store_token = _store_tid5_partials_f32(
                tl.cast(
                    projection_partials_ptr + projection_offset,
                    tl.uint64,
                ),
                tl.cast(
                    sqsum_partials_ptr + token * NUM_SPLITS + split,
                    tl.uint64,
                ),
                reduced,
                thread,
                output_tile,
            )
        else:
            read_col = local_hidden % 8
            read_mask = local_hidden < 8
            r0, r1, r2, r3, rs = _load_shared5_f32(
                tle.gpu.local_ptr(scratch, (channel0 + 0, read_col)),
                tle.gpu.local_ptr(scratch, (channel0 + 1, read_col)),
                tle.gpu.local_ptr(scratch, (channel0 + 2, read_col)),
                tle.gpu.local_ptr(scratch, (channel0 + 3, read_col)),
                tle.gpu.local_ptr(scratch, (channel0 + 4, read_col)),
            )
            r0 = tl.where(read_mask, r0, 0.0)
            r1 = tl.where(read_mask, r1, 0.0)
            r2 = tl.where(read_mask, r2, 0.0)
            r3 = tl.where(read_mask, r3, 0.0)
            rs = tl.where(read_mask, rs, 0.0)
            o0 = _warp_sum_32(r0)
            o1 = _warp_sum_32(r1)
            o2 = _warp_sum_32(r2)
            o3 = _warp_sum_32(r3)
            os = _warp_sum_32(rs)
            write_mask = local_hidden == 0
            tl.store(
                projection_partials_ptr
                + projection_offset
                + local_hidden
                + 0,
                o0,
                mask=write_mask,
            )
            tl.store(
                projection_partials_ptr
                + projection_offset
                + local_hidden
                + 1,
                o1,
                mask=write_mask,
            )
            tl.store(
                projection_partials_ptr
                + projection_offset
                + local_hidden
                + 2,
                o2,
                mask=write_mask,
            )
            tl.store(
                projection_partials_ptr
                + projection_offset
                + local_hidden
                + 3,
                o3,
                mask=write_mask,
            )
            if output_tile == 0:
                tl.store(
                    sqsum_partials_ptr
                    + token * NUM_SPLITS
                    + split
                    + local_hidden,
                    os,
                    mask=write_mask,
                )
    else:
        partial_lane = tl.arange(0, OUTPUT_TILE_BLOCK)
        projection_partial = tl.zeros((OUTPUT_TILE_BLOCK,), tl.float32)
        for output_in_tile in tl.static_range(0, OUTPUTS_PER_TILE):
            output_index = output_tile * OUTPUTS_PER_TILE + output_in_tile
            weight_row = fn_ptr + output_index * JIT_FULL_FLAT_HIDDEN
            w0 = tl.load(weight_row + hidden).to(tl.float32)
            w1 = tl.load(weight_row + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            w2 = tl.load(weight_row + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            w3 = tl.load(weight_row + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            dot = tl.sum(
                q0f * w0 + q1f * w1 + q2f * w2 + q3f * w3,
                axis=0,
            )
            projection_partial += tl.where(
                partial_lane == output_in_tile, dot, 0.0
            )
        projection_offset = (
            (
                token * NUM_OUTPUT_TILES * NUM_SPLITS
                + output_tile * NUM_SPLITS
                + split
            )
            * OUTPUT_TILE_BLOCK
            + partial_lane
        )
        tl.store(
            projection_partials_ptr + projection_offset,
            projection_partial,
        )
        if output_tile == 0:
            split_sqsum = tl.sum(
                q0f * q0f + q1f * q1f + q2f * q2f + q3f * q3f,
                axis=0,
            )
            tl.store(
                sqsum_partials_ptr + token * NUM_SPLITS + split,
                split_sqsum,
            )

    if not USE_T32_WARP0_HANDOFF_BARRIER:
        producer_ready = tle.gpu.alloc_barrier(
            arrive_count=CTA_THREADS_PER_BLOCK
        )
        tle.gpu.barrier_wait(producer_ready)
    if ENABLE_PHASE_PROFILING:
        tl.store(phase_profile_ptr + pid * 2 + 1, _read_globaltimer_ns())
    if USE_CUSTOM_HANDOFF:
        thread = _physical_thread_id_x()
        token_counter_ptr = (
            arrival_counter_ptr + token * COUNTER_STRIDE_ELEMENTS
        )
        if USE_T32_WARP0_HANDOFF_BARRIER:
            # The packed store helper returns one dependency token per
            # logical CTA thread.  Fold them to a scalar before entering the
            # scalar handoff helper; otherwise its result becomes a 256-lane
            # tensor and cannot guard the scalar last-arrival tail.
            partial_store_dependency = tl.sum(
                partial_store_token,
                axis=0,
            )
            leader_won = _warp0_publish_release_inc_u32_last_arrival(
                tl.cast(token_counter_ptr, tl.uint64),
                thread,
                CTAS_PER_TOKEN - 1,
                partial_store_dependency,
            )
        else:
            leader_won = _release_inc_u32_last_arrival(
                tl.cast(token_counter_ptr, tl.uint64),
                thread,
                CTAS_PER_TOKEN - 1,
            )
        if CTA_THREADS_PER_BLOCK == 256:
            is_last = _bar_red_or_pred_u32_256(leader_won) != 0
        else:
            is_last = _bar_red_or_pred_u32_512(leader_won) != 0
    else:
        old_ticket = tl.atomic_add(
            arrival_counter_ptr + token * COUNTER_STRIDE_ELEMENTS,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        is_last = old_ticket % CTAS_PER_TOKEN == CTAS_PER_TOKEN - 1
    if is_last:
        if not USE_CUSTOM_HANDOFF:
            acquire_ready = tle.gpu.alloc_barrier(
                arrive_count=CTA_THREADS_PER_BLOCK
            )
            tle.gpu.barrier_wait(acquire_ready)
        if ENABLE_PHASE_PROFILING:
            tl.store(
                phase_profile_ptr + TOTAL_CTAS * 2 + token * 4,
                _read_globaltimer_ns(),
            )

        projection_lane = tl.arange(0, 32)
        projection_mask = projection_lane < JIT_FULL_PROJECTION_OUTPUTS
        tile_for_output = projection_lane // OUTPUTS_PER_TILE
        lane_in_output_tile = projection_lane % OUTPUTS_PER_TILE
        projected = tl.zeros((32,), tl.float32)
        squared_sum = 0.0
        if USE_CONTIGUOUS_PARTIALS:
            for source_split in tl.static_range(0, NUM_SPLITS):
                partial_offset = (
                    (token * NUM_SPLITS + source_split)
                    * JIT_FULL_PROJECTION_OUTPUTS
                    + projection_lane
                )
                projected += tl.load(
                    projection_partials_ptr + partial_offset,
                    mask=projection_mask,
                    other=0.0,
                )
                if not USE_T2_SQSUM_TREE:
                    squared_sum += tl.load(
                        sqsum_partials_ptr + token * NUM_SPLITS + source_split
                    )
        else:
            if USE_T32_PACK_SQSUM:
                tl.static_assert(TOKENS == 32)
                tl.static_assert(NUM_SPLITS == 8)
                tl.static_assert(NUM_OUTPUT_TILES == 1)
                tl.static_assert(OUTPUTS_PER_TILE == 24)
                tl.static_assert(OUTPUT_TILE_BLOCK == 32)
                # Projection and sqsum now share one global row.  Keep the
                # proven scalar sqsum load, which avoids the code expansion
                # of an explicit lane-24 shuffle while preserving locality
                # and the original split-0 -> split-7 sum order.
                for source_split in tl.range(0, NUM_SPLITS):
                    packed_row_offset = (
                        (token * NUM_SPLITS + source_split)
                        * OUTPUT_TILE_BLOCK
                    )
                    packed_partial = tl.load(
                        projection_partials_ptr
                        + packed_row_offset
                        + projection_lane,
                        mask=projection_mask,
                        other=0.0,
                    )
                    projected += packed_partial
                    squared_sum += tl.load(
                        projection_partials_ptr
                        + packed_row_offset
                        + JIT_FULL_PROJECTION_OUTPUTS
                    )
            else:
                for source_split in tl.range(0, NUM_SPLITS):
                    partial_offset = (
                        (
                            token * NUM_OUTPUT_TILES * NUM_SPLITS
                            + tile_for_output * NUM_SPLITS
                            + source_split
                        )
                        * OUTPUT_TILE_BLOCK
                        + lane_in_output_tile
                    )
                    projected += tl.load(
                        projection_partials_ptr + partial_offset,
                        mask=projection_mask,
                        other=0.0,
                    )
                    if not USE_T2_SQSUM_TREE:
                        squared_sum += tl.load(
                            sqsum_partials_ptr
                            + token * NUM_SPLITS
                            + source_split
                        )

        if USE_T2_SQSUM_TREE:
            tl.static_assert(TOKENS == 2)
            tl.static_assert(NUM_SPLITS == 8)
            tl.static_assert(CTA_THREADS_PER_BLOCK == 512)
            lane_sqsum_partial = tl.load(
                sqsum_partials_ptr + token * NUM_SPLITS + projection_lane,
                mask=projection_lane < NUM_SPLITS,
                other=0.0,
            )
            squared_sum = _warp_sum_broadcast_32(lane_sqsum_partial)

        projection_norm = tl.rsqrt(
            squared_sum / JIT_FULL_FLAT_HIDDEN + hc_norm_eps
        )
        category_scale = tl.where(
            projection_lane < JIT_FULL_HC_MULT,
            tl.load(mix_scales_ptr),
            tl.where(
                projection_lane < 2 * JIT_FULL_HC_MULT,
                tl.load(mix_scales_ptr + 1),
                tl.load(mix_scales_ptr + 2),
            ),
        )
        base = tl.load(
            mix_bases_ptr + projection_lane,
            mask=projection_mask,
            other=0.0,
        ).to(tl.float32)
        mixes = projected * projection_norm * category_scale + base
        if USE_WARP_OVERLAP:
            mixes_shared = tle.gpu.alloc(
                [32],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            pre_shared = tle.gpu.alloc(
                [4],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            tl.store(
                tle.gpu.local_ptr(mixes_shared, (projection_lane,)),
                mixes,
                mask=projection_mask,
            )
            # TLE lowers a one-warp SM90 worker partition as one physical
            # four-warp group.  Only the requested worker warp executes the
            # worker function; the named barriers therefore count the 256
            # default threads plus 32 logical worker threads, exactly as the
            # independently verified 8+[1] microprobe does.
            mixes_ready = tle.gpu.alloc_barrier(
                arrive_count=JIT_FULL_OVERLAP_LOGICAL_THREADS
            )
            pre_ready = tle.gpu.alloc_barrier(
                arrive_count=JIT_FULL_OVERLAP_LOGICAL_THREADS
            )
            tle.gpu.warp_specialize(
                [
                    (
                        _tle_mhc_overlap_hidden_tail,
                        (
                            mixes_ready,
                            pre_ready,
                            pre_shared,
                            new_residual_ptr + token_residual_base,
                            rmsnorm_weight_ptr,
                            y_bf16_ptr + token_hidden_base,
                            rmsnorm_eps,
                            rmsnorm_scale,
                        ),
                    ),
                    (
                        _tle_mhc_overlap_routing_tail,
                        (
                            mixes_ready,
                            pre_ready,
                            mixes_shared,
                            pre_shared,
                            new_post_mix_ptr + token_state_base,
                            new_comb_res_mix_ptr + token_comb_base,
                            hc_pre_eps,
                            hc_post_scale,
                            sinkhorn_eps,
                            USE_FAST_RCP,
                        ),
                    ),
                ],
                [JIT_FULL_OVERLAP_ROUTING_WARPS],
                [64],
            )
            if not USE_CUSTOM_HANDOFF:
                tl.atomic_xchg(
                    arrival_counter_ptr + token * COUNTER_STRIDE_ELEMENTS,
                    0,
                    sem="release",
                    scope="gpu",
                )
            return
        pre_mix = tl.sigmoid(mixes) + hc_pre_eps
        post_mix = tl.sigmoid(mixes) * hc_post_scale
        post_mask = (
            (projection_lane >= JIT_FULL_HC_MULT)
            & (projection_lane < 2 * JIT_FULL_HC_MULT)
        )
        tl.store(
            new_post_mix_ptr
            + token_state_base
            + projection_lane
            - JIT_FULL_HC_MULT,
            post_mix,
            mask=post_mask,
        )
        if ENABLE_PHASE_PROFILING:
            profile_head_ready = tle.gpu.alloc_barrier(
                arrive_count=CTA_THREADS_PER_BLOCK
            )
            tle.gpu.barrier_wait(profile_head_ready)
            tl.store(
                phase_profile_ptr + TOTAL_CTAS * 2 + token * 4 + 1,
                _read_globaltimer_ns(),
            )

        routing_mask = (
            (projection_lane >= 2 * JIT_FULL_HC_MULT)
            & (projection_lane < JIT_FULL_PROJECTION_OUTPUTS)
        )
        routing = mixes
        if USE_WARP_SINKHORN:
            sinkhorn = tl.where(routing_mask, routing, 0.0)
            row_max = _warp_reduce4_max(sinkhorn)
            sinkhorn = tl.exp(sinkhorn - row_max)
            row_sum = _warp_reduce4_sum(sinkhorn)
            if USE_FAST_RCP:
                sinkhorn = sinkhorn * _fast_rcp_ftz_f32(row_sum)
                sinkhorn += sinkhorn_eps
                column_sum = (
                    _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
                )
                sinkhorn *= _fast_rcp_ftz_f32(column_sum)
            else:
                sinkhorn = sinkhorn / row_sum + sinkhorn_eps
                sinkhorn = sinkhorn / (
                    _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
                )
            for _ in tl.range(1, JIT_FULL_SINKHORN_ITERS):
                row_sum = _warp_reduce4_sum(sinkhorn) + sinkhorn_eps
                if USE_FAST_RCP:
                    sinkhorn *= _fast_rcp_ftz_f32(row_sum)
                    column_sum = (
                        _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
                    )
                    sinkhorn *= _fast_rcp_ftz_f32(column_sum)
                else:
                    sinkhorn = sinkhorn / row_sum
                    sinkhorn = sinkhorn / (
                        _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
                    )
        else:
            sinkhorn = tl.zeros((32,), tl.float32)
            for row in tl.static_range(0, JIT_FULL_HC_MULT):
                row_mask = routing_mask & (
                    (projection_lane - 2 * JIT_FULL_HC_MULT)
                    // JIT_FULL_HC_MULT
                    == row
                )
                row_max = tl.max(
                    tl.where(row_mask, routing, -float("inf")), axis=0
                )
                exponent = tl.where(
                    row_mask, tl.exp(routing - row_max), 0.0
                )
                row_sum = tl.sum(exponent, axis=0)
                sinkhorn += tl.where(
                    row_mask, exponent / row_sum + sinkhorn_eps, 0.0
                )

            normalized = tl.zeros((32,), tl.float32)
            for column in tl.static_range(0, JIT_FULL_HC_MULT):
                column_mask = routing_mask & (
                    (projection_lane - 2 * JIT_FULL_HC_MULT)
                    % JIT_FULL_HC_MULT
                    == column
                )
                column_sum = tl.sum(
                    tl.where(column_mask, sinkhorn, 0.0), axis=0
                )
                normalized += tl.where(
                    column_mask,
                    sinkhorn / (column_sum + sinkhorn_eps),
                    0.0,
                )
            sinkhorn = normalized
            for _ in tl.range(0, JIT_FULL_SINKHORN_ITERS - 1):
                normalized = tl.zeros((32,), tl.float32)
                for row in tl.static_range(0, JIT_FULL_HC_MULT):
                    row_mask = routing_mask & (
                        (projection_lane - 2 * JIT_FULL_HC_MULT)
                        // JIT_FULL_HC_MULT
                        == row
                    )
                    row_sum = tl.sum(
                        tl.where(row_mask, sinkhorn, 0.0), axis=0
                    )
                    normalized += tl.where(
                        row_mask,
                        sinkhorn / (row_sum + sinkhorn_eps),
                        0.0,
                    )
                sinkhorn = normalized
                normalized = tl.zeros((32,), tl.float32)
                for column in tl.static_range(0, JIT_FULL_HC_MULT):
                    column_mask = routing_mask & (
                        (projection_lane - 2 * JIT_FULL_HC_MULT)
                        % JIT_FULL_HC_MULT
                        == column
                    )
                    column_sum = tl.sum(
                        tl.where(column_mask, sinkhorn, 0.0), axis=0
                    )
                    normalized += tl.where(
                        column_mask,
                        sinkhorn / (column_sum + sinkhorn_eps),
                        0.0,
                    )
                sinkhorn = normalized
        tl.store(
            new_comb_res_mix_ptr
            + token_comb_base
            + projection_lane
            - 2 * JIT_FULL_HC_MULT,
            sinkhorn,
            mask=routing_mask,
        )
        if ENABLE_PHASE_PROFILING:
            profile_sinkhorn_ready = tle.gpu.alloc_barrier(
                arrive_count=CTA_THREADS_PER_BLOCK
            )
            tle.gpu.barrier_wait(profile_sinkhorn_ready)
            tl.store(
                phase_profile_ptr + TOTAL_CTAS * 2 + token * 4 + 2,
                _read_globaltimer_ns(),
            )

        h = tl.arange(0, JIT_FULL_HIDDEN)
        norm_input = tl.zeros((JIT_FULL_HIDDEN,), tl.float32)
        if USE_T32_PREMIX_DIRECT_BCAST:
            # The tail's 32-element projection vector is warp-replicated.
            # Broadcast its four Pre lanes directly, then expose each
            # warp-uniform value across the 128 x 32 hidden-value view.  Keep
            # the residual loads and dependent accumulations in the original
            # channel-0 -> channel-3 order: spelling all four products as one
            # expression lets LLVM reassociate the FP32 tree and can move a
            # value across the following BF16 rounding boundary.
            pre0, pre1, pre2, pre3 = _warp_broadcast_lanes0_3_f32(pre_mix)
            if USE_T32_PREMIX_SHARED_BRIDGE:
                # Direct broadcasts naturally carry the 32-lane projection
                # layout.  Bridging each warp's four uniform values through a
                # private shared row turns the consumers into true per-thread
                # scalars, so the 4096-element hidden layout can reuse them
                # without a cross-layout conversion.  A warp barrier is enough:
                # no other warp ever reads this row.
                pre_mix_shared = tle.gpu.alloc(
                    [8, 4],
                    dtype=tl.float32,
                    layout=None,
                    scope=tle.gpu.smem,
                    nv_mma_shared_layout=False,
                )
                physical_thread = _physical_thread_id_x()
                physical_warp = physical_thread // 32
                physical_lane = physical_thread % 32
                pre0_ptr = tle.gpu.local_ptr(
                    pre_mix_shared, (physical_warp, 0)
                )
                pre1_ptr = tle.gpu.local_ptr(
                    pre_mix_shared, (physical_warp, 1)
                )
                pre2_ptr = tle.gpu.local_ptr(
                    pre_mix_shared, (physical_warp, 2)
                )
                pre3_ptr = tle.gpu.local_ptr(
                    pre_mix_shared, (physical_warp, 3)
                )
                _store_shared4_f32(
                    _shared_address_u32(pre0_ptr),
                    _shared_address_u32(pre1_ptr),
                    _shared_address_u32(pre2_ptr),
                    _shared_address_u32(pre3_ptr),
                    pre0,
                    pre1,
                    pre2,
                    pre3,
                    physical_lane,
                )
                _warp_sync_full_mask()
                pre0_scalar = tl.load(pre0_ptr)
                pre1_scalar = tl.load(pre1_ptr)
                pre2_scalar = tl.load(pre2_ptr)
                pre3_scalar = tl.load(pre3_ptr)
                value = tl.load(
                    new_residual_ptr + token_residual_base + h
                ).to(tl.float32)
                norm_input += value * pre0_scalar
                value = tl.load(
                    new_residual_ptr
                    + token_residual_base
                    + JIT_FULL_HIDDEN
                    + h
                ).to(tl.float32)
                norm_input += value * pre1_scalar
                value = tl.load(
                    new_residual_ptr
                    + token_residual_base
                    + 2 * JIT_FULL_HIDDEN
                    + h
                ).to(tl.float32)
                norm_input += value * pre2_scalar
                value = tl.load(
                    new_residual_ptr
                    + token_residual_base
                    + 3 * JIT_FULL_HIDDEN
                    + h
                ).to(tl.float32)
                norm_input += value * pre3_scalar
            else:
                pre0 = tl.reshape(pre0, (1, 32))
                pre1 = tl.reshape(pre1, (1, 32))
                pre2 = tl.reshape(pre2, (1, 32))
                pre3 = tl.reshape(pre3, (1, 32))
                norm_input_2d = tl.reshape(norm_input, (128, 32))
                value = tl.reshape(
                    tl.load(new_residual_ptr + token_residual_base + h).to(
                        tl.float32
                    ),
                    (128, 32),
                )
                norm_input_2d += value * pre0
                value = tl.reshape(
                    tl.load(
                        new_residual_ptr
                        + token_residual_base
                        + JIT_FULL_HIDDEN
                        + h
                    ).to(tl.float32),
                    (128, 32),
                )
                norm_input_2d += value * pre1
                value = tl.reshape(
                    tl.load(
                        new_residual_ptr
                        + token_residual_base
                        + 2 * JIT_FULL_HIDDEN
                        + h
                    ).to(tl.float32),
                    (128, 32),
                )
                norm_input_2d += value * pre2
                value = tl.reshape(
                    tl.load(
                        new_residual_ptr
                        + token_residual_base
                        + 3 * JIT_FULL_HIDDEN
                        + h
                    ).to(tl.float32),
                    (128, 32),
                )
                norm_input_2d += value * pre3
                norm_input = tl.reshape(norm_input_2d, (JIT_FULL_HIDDEN,))
        else:
            for channel in tl.static_range(0, JIT_FULL_HC_MULT):
                value = tl.load(
                    new_residual_ptr
                    + token_residual_base
                    + channel * JIT_FULL_HIDDEN
                    + h
                ).to(tl.float32)
                pre_scalar = tl.sum(
                    tl.where(
                        projection_lane == channel, pre_mix, 0.0
                    ),
                    axis=0,
                )
                norm_input += value * pre_scalar
        norm_input_bf16 = norm_input.to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        norm_input_fp32 = norm_input_bf16.to(tl.float32)
        rms = tl.rsqrt(
            tl.sum(norm_input_fp32 * norm_input_fp32, axis=0)
            / JIT_FULL_HIDDEN
            + rmsnorm_eps
        )
        rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
        y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
        tl.store(
            y_bf16_ptr + token_hidden_base + h,
            y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        )

        if not USE_T32_SKIP_EPILOGUE_BARRIER:
            epilogue_ready = tle.gpu.alloc_barrier(
                arrive_count=CTA_THREADS_PER_BLOCK
            )
            tle.gpu.barrier_wait(epilogue_ready)
        # The strict T=32 packed-sqsum probe has no in-kernel consumer after
        # these public-output stores.  Its release atomic_inc(limit=7)
        # self-resets before this tail, profiling is disabled, and kernel (or
        # captured-Graph) completion supplies inter-launch ordering.  Thus its
        # final no-consumer full-CTA barrier can be compiled away safely.
        if ENABLE_PHASE_PROFILING:
            tl.store(
                phase_profile_ptr + TOTAL_CTAS * 2 + token * 4 + 3,
                _read_globaltimer_ns(),
            )
        if not USE_CUSTOM_HANDOFF:
            tl.atomic_xchg(
                arrival_counter_ptr + token * COUNTER_STRIDE_ELEMENTS,
                0,
                sem="release",
                scope="gpu",
            )


@triton.jit
def _tle_sm90_mega_mhc_t32_pair_tail(
    token,
    projection_partials_ptr,
    sqsum_partials_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    rmsnorm_weight_ptr,
    new_residual_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    y_bf16_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Finish one acquired token for the strict T=32 token-pair kernel."""
    token_hidden_base = token * JIT_FULL_HIDDEN
    token_residual_base = token * JIT_FULL_FLAT_HIDDEN
    token_state_base = token * JIT_FULL_HC_MULT
    token_comb_base = token * JIT_FULL_HC_MULT * JIT_FULL_HC_MULT
    projection_lane = tl.arange(0, 32)
    projection_mask = projection_lane < JIT_FULL_PROJECTION_OUTPUTS
    projected = tl.zeros((32,), tl.float32)
    squared_sum = 0.0
    # Keep the established split-0..7 accumulation order for each token.
    for source_split in tl.range(0, JIT_T8_NUM_SPLITS):
        partial_offset = (
            (token * JIT_T8_NUM_SPLITS + source_split) * 32
            + projection_lane
        )
        projected += tl.load(
            projection_partials_ptr + partial_offset,
            mask=projection_mask,
            other=0.0,
        )
        squared_sum += tl.load(
            sqsum_partials_ptr + token * JIT_T8_NUM_SPLITS + source_split
        )

    projection_norm = tl.rsqrt(
        squared_sum / JIT_FULL_FLAT_HIDDEN + hc_norm_eps
    )
    category_scale = tl.where(
        projection_lane < JIT_FULL_HC_MULT,
        tl.load(mix_scales_ptr),
        tl.where(
            projection_lane < 2 * JIT_FULL_HC_MULT,
            tl.load(mix_scales_ptr + 1),
            tl.load(mix_scales_ptr + 2),
        ),
    )
    base = tl.load(
        mix_bases_ptr + projection_lane,
        mask=projection_mask,
        other=0.0,
    ).to(tl.float32)
    mixes = projected * projection_norm * category_scale + base
    pre_mix = tl.sigmoid(mixes) + hc_pre_eps
    post_mix = tl.sigmoid(mixes) * hc_post_scale
    post_mask = (
        (projection_lane >= JIT_FULL_HC_MULT)
        & (projection_lane < 2 * JIT_FULL_HC_MULT)
    )
    tl.store(
        new_post_mix_ptr
        + token_state_base
        + projection_lane
        - JIT_FULL_HC_MULT,
        post_mix,
        mask=post_mask,
    )

    routing_mask = (
        (projection_lane >= 2 * JIT_FULL_HC_MULT)
        & (projection_lane < JIT_FULL_PROJECTION_OUTPUTS)
    )
    sinkhorn = tl.where(routing_mask, mixes, 0.0)
    row_max = _warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _warp_reduce4_sum(sinkhorn)
    sinkhorn = sinkhorn * _fast_rcp_ftz_f32(row_sum)
    sinkhorn += sinkhorn_eps
    column_sum = _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
    sinkhorn *= _fast_rcp_ftz_f32(column_sum)
    for _ in tl.range(1, JIT_FULL_SINKHORN_ITERS):
        row_sum = _warp_reduce4_sum(sinkhorn) + sinkhorn_eps
        sinkhorn *= _fast_rcp_ftz_f32(row_sum)
        column_sum = _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
        sinkhorn *= _fast_rcp_ftz_f32(column_sum)
    tl.store(
        new_comb_res_mix_ptr
        + token_comb_base
        + projection_lane
        - 2 * JIT_FULL_HC_MULT,
        sinkhorn,
        mask=routing_mask,
    )

    h = tl.arange(0, JIT_FULL_HIDDEN)
    norm_input = tl.zeros((JIT_FULL_HIDDEN,), tl.float32)
    for channel in tl.static_range(0, JIT_FULL_HC_MULT):
        value = tl.load(
            new_residual_ptr
            + token_residual_base
            + channel * JIT_FULL_HIDDEN
            + h
        ).to(tl.float32)
        pre_scalar = tl.sum(
            tl.where(projection_lane == channel, pre_mix, 0.0), axis=0
        )
        norm_input += value * pre_scalar
    norm_input_bf16 = norm_input.to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=0)
        / JIT_FULL_HIDDEN
        + rmsnorm_eps
    )
    rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + token_hidden_base + h,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
    )
    # Preserve the admitted tail's full-CTA completion point.  When one CTA
    # owns both tokens, this also makes the two sequential tail calls explicit.
    epilogue_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_T8_REDUCTION_THREADS
    )
    tle.gpu.barrier_wait(epilogue_ready)


@triton.jit
def _tle_sm90_mega_mhc_t32_pair_kernel(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    rmsnorm_weight_ptr,
    new_residual_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    y_bf16_ptr,
    projection_partials_ptr,
    sqsum_partials_ptr,
    arrival_counter_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
    COUNTER_STRIDE_ELEMENTS: tl.constexpr,
):
    """Strict T=32 split-8 kernel that shares each Fn load across two tokens."""
    pid = tl.program_id(0)
    pair = pid // JIT_T8_NUM_SPLITS
    split = pid % JIT_T8_NUM_SPLITS
    token0 = 2 * pair
    token1 = token0 + 1
    local_hidden = tl.arange(0, 512)
    hidden = split * 512 + local_hidden

    token0_hidden_base = token0 * JIT_FULL_HIDDEN
    token1_hidden_base = token1 * JIT_FULL_HIDDEN
    token0_residual_base = token0 * JIT_FULL_FLAT_HIDDEN
    token1_residual_base = token1 * JIT_FULL_FLAT_HIDDEN
    token0_state_base = token0 * JIT_FULL_HC_MULT
    token1_state_base = token1 * JIT_FULL_HC_MULT
    token0_comb_base = token0 * JIT_FULL_HC_MULT * JIT_FULL_HC_MULT
    token1_comb_base = token1 * JIT_FULL_HC_MULT * JIT_FULL_HC_MULT

    x0 = tl.load(x_ptr + token0_hidden_base + hidden).to(tl.float32)
    x1 = tl.load(x_ptr + token1_hidden_base + hidden).to(tl.float32)
    r00 = tl.load(residual_ptr + token0_residual_base + hidden).to(tl.float32)
    r01 = tl.load(residual_ptr + token1_residual_base + hidden).to(tl.float32)
    r10 = tl.load(
        residual_ptr + token0_residual_base + JIT_FULL_HIDDEN + hidden
    ).to(tl.float32)
    r11 = tl.load(
        residual_ptr + token1_residual_base + JIT_FULL_HIDDEN + hidden
    ).to(tl.float32)
    r20 = tl.load(
        residual_ptr + token0_residual_base + 2 * JIT_FULL_HIDDEN + hidden
    ).to(tl.float32)
    r21 = tl.load(
        residual_ptr + token1_residual_base + 2 * JIT_FULL_HIDDEN + hidden
    ).to(tl.float32)
    r30 = tl.load(
        residual_ptr + token0_residual_base + 3 * JIT_FULL_HIDDEN + hidden
    ).to(tl.float32)
    r31 = tl.load(
        residual_ptr + token1_residual_base + 3 * JIT_FULL_HIDDEN + hidden
    ).to(tl.float32)

    # Compute each token's channels independently and retain every BF16 RTNE
    # boundary from the admitted one-token-per-CTA producer.
    q00 = (
        x0 * tl.load(post_mix_ptr + token0_state_base).to(tl.float32)
        + r00 * tl.load(comb_res_mix_ptr + token0_comb_base).to(tl.float32)
        + r10 * tl.load(comb_res_mix_ptr + token0_comb_base + 4).to(tl.float32)
        + r20 * tl.load(comb_res_mix_ptr + token0_comb_base + 8).to(tl.float32)
        + r30 * tl.load(comb_res_mix_ptr + token0_comb_base + 12).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q01 = (
        x0 * tl.load(post_mix_ptr + token0_state_base + 1).to(tl.float32)
        + r00 * tl.load(comb_res_mix_ptr + token0_comb_base + 1).to(tl.float32)
        + r10 * tl.load(comb_res_mix_ptr + token0_comb_base + 5).to(tl.float32)
        + r20 * tl.load(comb_res_mix_ptr + token0_comb_base + 9).to(tl.float32)
        + r30 * tl.load(comb_res_mix_ptr + token0_comb_base + 13).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q02 = (
        x0 * tl.load(post_mix_ptr + token0_state_base + 2).to(tl.float32)
        + r00 * tl.load(comb_res_mix_ptr + token0_comb_base + 2).to(tl.float32)
        + r10 * tl.load(comb_res_mix_ptr + token0_comb_base + 6).to(tl.float32)
        + r20 * tl.load(comb_res_mix_ptr + token0_comb_base + 10).to(tl.float32)
        + r30 * tl.load(comb_res_mix_ptr + token0_comb_base + 14).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q03 = (
        x0 * tl.load(post_mix_ptr + token0_state_base + 3).to(tl.float32)
        + r00 * tl.load(comb_res_mix_ptr + token0_comb_base + 3).to(tl.float32)
        + r10 * tl.load(comb_res_mix_ptr + token0_comb_base + 7).to(tl.float32)
        + r20 * tl.load(comb_res_mix_ptr + token0_comb_base + 11).to(tl.float32)
        + r30 * tl.load(comb_res_mix_ptr + token0_comb_base + 15).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q10 = (
        x1 * tl.load(post_mix_ptr + token1_state_base).to(tl.float32)
        + r01 * tl.load(comb_res_mix_ptr + token1_comb_base).to(tl.float32)
        + r11 * tl.load(comb_res_mix_ptr + token1_comb_base + 4).to(tl.float32)
        + r21 * tl.load(comb_res_mix_ptr + token1_comb_base + 8).to(tl.float32)
        + r31 * tl.load(comb_res_mix_ptr + token1_comb_base + 12).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q11 = (
        x1 * tl.load(post_mix_ptr + token1_state_base + 1).to(tl.float32)
        + r01 * tl.load(comb_res_mix_ptr + token1_comb_base + 1).to(tl.float32)
        + r11 * tl.load(comb_res_mix_ptr + token1_comb_base + 5).to(tl.float32)
        + r21 * tl.load(comb_res_mix_ptr + token1_comb_base + 9).to(tl.float32)
        + r31 * tl.load(comb_res_mix_ptr + token1_comb_base + 13).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q12 = (
        x1 * tl.load(post_mix_ptr + token1_state_base + 2).to(tl.float32)
        + r01 * tl.load(comb_res_mix_ptr + token1_comb_base + 2).to(tl.float32)
        + r11 * tl.load(comb_res_mix_ptr + token1_comb_base + 6).to(tl.float32)
        + r21 * tl.load(comb_res_mix_ptr + token1_comb_base + 10).to(tl.float32)
        + r31 * tl.load(comb_res_mix_ptr + token1_comb_base + 14).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q13 = (
        x1 * tl.load(post_mix_ptr + token1_state_base + 3).to(tl.float32)
        + r01 * tl.load(comb_res_mix_ptr + token1_comb_base + 3).to(tl.float32)
        + r11 * tl.load(comb_res_mix_ptr + token1_comb_base + 7).to(tl.float32)
        + r21 * tl.load(comb_res_mix_ptr + token1_comb_base + 11).to(tl.float32)
        + r31 * tl.load(comb_res_mix_ptr + token1_comb_base + 15).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    q00f, q01f, q02f, q03f = (
        q00.to(tl.float32),
        q01.to(tl.float32),
        q02.to(tl.float32),
        q03.to(tl.float32),
    )
    q10f, q11f, q12f, q13f = (
        q10.to(tl.float32),
        q11.to(tl.float32),
        q12.to(tl.float32),
        q13.to(tl.float32),
    )
    tl.store(new_residual_ptr + token0_residual_base + hidden, q00)
    tl.store(new_residual_ptr + token1_residual_base + hidden, q10)
    tl.store(
        new_residual_ptr
        + token0_residual_base
        + JIT_FULL_HIDDEN
        + hidden,
        q01,
    )
    tl.store(
        new_residual_ptr
        + token1_residual_base
        + JIT_FULL_HIDDEN
        + hidden,
        q11,
    )
    tl.store(
        new_residual_ptr
        + token0_residual_base
        + 2 * JIT_FULL_HIDDEN
        + hidden,
        q02,
    )
    tl.store(
        new_residual_ptr
        + token1_residual_base
        + 2 * JIT_FULL_HIDDEN
        + hidden,
        q12,
    )
    tl.store(
        new_residual_ptr
        + token0_residual_base
        + 3 * JIT_FULL_HIDDEN
        + hidden,
        q03,
    )
    tl.store(
        new_residual_ptr
        + token1_residual_base
        + 3 * JIT_FULL_HIDDEN
        + hidden,
        q13,
    )

    reduction_thread = tl.arange(0, JIT_T8_REDUCTION_THREADS)
    reduction_lane = reduction_thread % 32
    reduction_warp = reduction_thread // 32
    channel0 = reduction_thread * 0
    scratch25 = tle.gpu.alloc(
        [16, JIT_T32_TILE24_REDUCTION_CHANNEL_BLOCK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    for output_index in tl.static_range(0, JIT_FULL_PROJECTION_OUTPUTS):
        weight_row = fn_ptr + output_index * JIT_FULL_FLAT_HIDDEN
        # The four Fn vectors are loaded once and consumed by both tokens.
        w0 = tl.load(weight_row + hidden).to(tl.float32)
        w1 = tl.load(weight_row + JIT_FULL_HIDDEN + hidden).to(tl.float32)
        w2 = tl.load(weight_row + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
        w3 = tl.load(weight_row + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
        lane_partial0 = q00f * w0 + q01f * w1 + q02f * w2 + q03f * w3
        lane_partial1 = q10f * w0 + q11f * w1 + q12f * w2 + q13f * w3
        lane_partial0 = _pair_sum_512_to_256(lane_partial0)
        lane_partial1 = _pair_sum_512_to_256(lane_partial1)
        warp_partial0, warp_partial1 = _warp_sum2_32(
            lane_partial0, lane_partial1
        )
        shared_store_token = _store_shared2_f32(
            _shared_address_u32(
                tle.gpu.local_ptr(
                    scratch25, (reduction_warp, channel0 + output_index)
                )
            ),
            _shared_address_u32(
                tle.gpu.local_ptr(
                    scratch25,
                    (reduction_warp + 8, channel0 + output_index),
                )
            ),
            warp_partial0,
            warp_partial1,
            reduction_lane,
        )

    lane_sqsum0 = q00f * q00f + q01f * q01f + q02f * q02f + q03f * q03f
    lane_sqsum1 = q10f * q10f + q11f * q11f + q12f * q12f + q13f * q13f
    lane_sqsum0 = _pair_sum_512_to_256(lane_sqsum0)
    lane_sqsum1 = _pair_sum_512_to_256(lane_sqsum1)
    warp_sqsum0, warp_sqsum1 = _warp_sum2_32(lane_sqsum0, lane_sqsum1)
    shared_sqsum_store_token = _store_shared2_f32(
        _shared_address_u32(
            tle.gpu.local_ptr(
                scratch25,
                (
                    reduction_warp,
                    channel0 + JIT_FULL_PROJECTION_OUTPUTS,
                ),
            )
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(
                scratch25,
                (
                    reduction_warp + 8,
                    channel0 + JIT_FULL_PROJECTION_OUTPUTS,
                ),
            )
        ),
        warp_sqsum0,
        warp_sqsum1,
        reduction_lane,
    )

    shared25_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_T8_REDUCTION_THREADS
    )
    tle.gpu.barrier_wait(shared25_ready)
    safe_channel = tl.minimum(
        reduction_thread, JIT_T32_TILE24_REDUCTION_CHANNELS - 1
    )
    reduced0 = _reduce_shared8_tid25_f32(
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 0, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 1, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 2, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 3, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 4, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 5, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 6, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 7, safe_channel))
        ),
        reduction_thread,
        0,
    )
    reduced1 = _reduce_shared8_tid25_f32(
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 8, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 9, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 10, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 11, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 12, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 13, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 14, safe_channel))
        ),
        _shared_address_u32(
            tle.gpu.local_ptr(scratch25, (channel0 + 15, safe_channel))
        ),
        reduction_thread,
        0,
    )
    projection_offset0 = (token0 * JIT_T8_NUM_SPLITS + split) * 32
    projection_offset1 = (token1 * JIT_T8_NUM_SPLITS + split) * 32
    partial_store_token0 = _store_tid25_partials_f32(
        tl.cast(projection_partials_ptr + projection_offset0, tl.uint64),
        tl.cast(
            sqsum_partials_ptr + token0 * JIT_T8_NUM_SPLITS + split,
            tl.uint64,
        ),
        reduced0,
        reduction_thread,
        0,
    )
    partial_store_token1 = _store_tid25_partials_f32(
        tl.cast(projection_partials_ptr + projection_offset1, tl.uint64),
        tl.cast(
            sqsum_partials_ptr + token1 * JIT_T8_NUM_SPLITS + split,
            tl.uint64,
        ),
        reduced1,
        reduction_thread,
        0,
    )

    # Publish every thread's two-token output before either release increment.
    producer_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_T8_REDUCTION_THREADS
    )
    tle.gpu.barrier_wait(producer_ready)
    thread = _physical_thread_id_x()
    token0_counter_ptr = (
        arrival_counter_ptr + token0 * COUNTER_STRIDE_ELEMENTS
    )
    token1_counter_ptr = (
        arrival_counter_ptr + token1 * COUNTER_STRIDE_ELEMENTS
    )
    leader_won0 = _release_inc_u32_last_arrival(
        tl.cast(token0_counter_ptr, tl.uint64), thread, JIT_T8_NUM_SPLITS - 1
    )
    leader_won1 = _release_inc_u32_last_arrival(
        tl.cast(token1_counter_ptr, tl.uint64), thread, JIT_T8_NUM_SPLITS - 1
    )
    # Both atomics must execute before either token can enter its tail.  Each
    # reduction broadcasts its independently acquired winner flag CTA-wide.
    is_last0 = _bar_red_or_pred_u32_256(leader_won0) != 0
    is_last1 = _bar_red_or_pred_u32_256(leader_won1) != 0
    if is_last0:
        _tle_sm90_mega_mhc_t32_pair_tail(
            token0,
            projection_partials_ptr,
            sqsum_partials_ptr,
            mix_scales_ptr,
            mix_bases_ptr,
            rmsnorm_weight_ptr,
            new_residual_ptr,
            new_post_mix_ptr,
            new_comb_res_mix_ptr,
            y_bf16_ptr,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sinkhorn_eps,
            rmsnorm_eps,
            rmsnorm_scale,
        )
    if is_last1:
        _tle_sm90_mega_mhc_t32_pair_tail(
            token1,
            projection_partials_ptr,
            sqsum_partials_ptr,
            mix_scales_ptr,
            mix_bases_ptr,
            rmsnorm_weight_ptr,
            new_residual_ptr,
            new_post_mix_ptr,
            new_comb_res_mix_ptr,
            y_bf16_ptr,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sinkhorn_eps,
            rmsnorm_eps,
            rmsnorm_scale,
        )


@triton.jit
def _tle_sm90_mega_mhc_t1_single_cta_kernel(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    rmsnorm_weight_ptr,
    new_residual_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    y_bf16_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Complete T=1 Normal-BF16 MegaMHC in one correctness-first CTA.

    The two explicit TLE barriers document the phase boundaries and make the
    single-CTA fallback exercise the same TLE synchronization surface as the
    split-CTA migration protocol.  Values consumed by later phases remain in
    SSA registers; no helper kernel or hidden workspace is involved.
    """
    flat = tl.arange(0, JIT_FULL_FLAT_HIDDEN)
    hidden = flat % JIT_FULL_HIDDEN
    residual_channel = flat // JIT_FULL_HIDDEN

    x = tl.load(x_ptr + hidden).to(tl.float32)
    post = tl.load(post_mix_ptr + residual_channel).to(tl.float32)
    new_residual_fp32 = x * post
    for source_channel in tl.static_range(0, JIT_FULL_HC_MULT):
        residual = tl.load(
            residual_ptr + source_channel * JIT_FULL_HIDDEN + hidden
        ).to(tl.float32)
        coefficient = tl.load(
            comb_res_mix_ptr
            + source_channel * JIT_FULL_HC_MULT
            + residual_channel
        ).to(tl.float32)
        new_residual_fp32 += residual * coefficient

    # This BF16 boundary is part of the public operator semantics.  The
    # projection and RMS path both consume the rounded value, not the FP32
    # accumulator above.
    new_residual_bf16 = new_residual_fp32.to(tl.bfloat16)
    tl.store(new_residual_ptr + flat, new_residual_bf16)
    rounded_residual = new_residual_bf16.to(tl.float32)

    residual_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_FULL_CTA_THREADS
    )
    tle.gpu.barrier_wait(residual_ready)

    projection_lane = tl.arange(0, 32)
    projected = tl.zeros((32,), tl.float32)
    for output_index in tl.static_range(0, JIT_FULL_PROJECTION_OUTPUTS):
        weight = tl.load(
            fn_ptr + output_index * JIT_FULL_FLAT_HIDDEN + flat
        ).to(tl.float32)
        dot = tl.sum(rounded_residual * weight, axis=0)
        projected += tl.where(projection_lane == output_index, dot, 0.0)

    squared_sum = tl.sum(rounded_residual * rounded_residual, axis=0)
    projection_norm = tl.rsqrt(
        squared_sum / JIT_FULL_FLAT_HIDDEN + hc_norm_eps
    )
    category_scale = tl.where(
        projection_lane < JIT_FULL_HC_MULT,
        tl.load(mix_scales_ptr),
        tl.where(
            projection_lane < 2 * JIT_FULL_HC_MULT,
            tl.load(mix_scales_ptr + 1),
            tl.load(mix_scales_ptr + 2),
        ),
    )
    base = tl.load(
        mix_bases_ptr + projection_lane,
        mask=projection_lane < JIT_FULL_PROJECTION_OUTPUTS,
        other=0.0,
    ).to(tl.float32)
    mixes = projected * projection_norm * category_scale + base

    pre_mix = tl.sigmoid(mixes) + hc_pre_eps
    post_mix = tl.sigmoid(mixes) * hc_post_scale
    post_mask = (
        (projection_lane >= JIT_FULL_HC_MULT)
        & (projection_lane < 2 * JIT_FULL_HC_MULT)
    )
    tl.store(
        new_post_mix_ptr + projection_lane - JIT_FULL_HC_MULT,
        post_mix,
        mask=post_mask,
    )

    routing_lane = projection_lane
    routing = mixes
    sinkhorn = tl.zeros((32,), tl.float32)
    for row in tl.static_range(0, JIT_FULL_HC_MULT):
        row_mask = (
            routing_lane >= 2 * JIT_FULL_HC_MULT
        ) & (
            routing_lane < JIT_FULL_PROJECTION_OUTPUTS
        ) & (
            (routing_lane - 2 * JIT_FULL_HC_MULT)
            // JIT_FULL_HC_MULT
            == row
        )
        row_max = tl.max(
            tl.where(row_mask, routing, -float("inf")), axis=0
        )
        exponent = tl.where(row_mask, tl.exp(routing - row_max), 0.0)
        row_sum = tl.sum(exponent, axis=0)
        sinkhorn += tl.where(row_mask, exponent / row_sum + sinkhorn_eps, 0.0)

    # Initial softmax is followed by one column normalization, then nineteen
    # row/column pairs for the revision-pinned 20-iteration contract.
    normalized = tl.zeros((32,), tl.float32)
    for column in tl.static_range(0, JIT_FULL_HC_MULT):
        column_mask = (
            routing_lane >= 2 * JIT_FULL_HC_MULT
        ) & (
            routing_lane < JIT_FULL_PROJECTION_OUTPUTS
        ) & (
            (routing_lane - 2 * JIT_FULL_HC_MULT)
            % JIT_FULL_HC_MULT
            == column
        )
        column_sum = tl.sum(tl.where(column_mask, sinkhorn, 0.0), axis=0)
        normalized += tl.where(
            column_mask, sinkhorn / (column_sum + sinkhorn_eps), 0.0
        )
    sinkhorn = normalized
    for _ in tl.static_range(0, JIT_FULL_SINKHORN_ITERS - 1):
        normalized = tl.zeros((32,), tl.float32)
        for row in tl.static_range(0, JIT_FULL_HC_MULT):
            row_mask = (
                routing_lane >= 2 * JIT_FULL_HC_MULT
            ) & (
                routing_lane < JIT_FULL_PROJECTION_OUTPUTS
            ) & (
                (routing_lane - 2 * JIT_FULL_HC_MULT)
                // JIT_FULL_HC_MULT
                == row
            )
            row_sum = tl.sum(tl.where(row_mask, sinkhorn, 0.0), axis=0)
            normalized += tl.where(
                row_mask, sinkhorn / (row_sum + sinkhorn_eps), 0.0
            )
        sinkhorn = normalized
        normalized = tl.zeros((32,), tl.float32)
        for column in tl.static_range(0, JIT_FULL_HC_MULT):
            column_mask = (
                routing_lane >= 2 * JIT_FULL_HC_MULT
            ) & (
                routing_lane < JIT_FULL_PROJECTION_OUTPUTS
            ) & (
                (routing_lane - 2 * JIT_FULL_HC_MULT)
                % JIT_FULL_HC_MULT
                == column
            )
            column_sum = tl.sum(
                tl.where(column_mask, sinkhorn, 0.0), axis=0
            )
            normalized += tl.where(
                column_mask, sinkhorn / (column_sum + sinkhorn_eps), 0.0
            )
        sinkhorn = normalized
    routing_mask = (
        (routing_lane >= 2 * JIT_FULL_HC_MULT)
        & (routing_lane < JIT_FULL_PROJECTION_OUTPUTS)
    )
    tl.store(
        new_comb_res_mix_ptr + routing_lane - 2 * JIT_FULL_HC_MULT,
        sinkhorn,
        mask=routing_mask,
    )

    projection_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_FULL_CTA_THREADS
    )
    tle.gpu.barrier_wait(projection_ready)

    # Each hidden lane is represented four times in ``flat``.  Only the first
    # 4096 lanes compute the channel reduction and RMSNorm output.
    h = tl.arange(0, JIT_FULL_HIDDEN)
    norm_input = tl.zeros((JIT_FULL_HIDDEN,), tl.float32)
    for channel in tl.static_range(0, JIT_FULL_HC_MULT):
        value = tl.load(
            new_residual_ptr + channel * JIT_FULL_HIDDEN + h
        ).to(tl.float32)
        pre_scalar = tl.sum(
            tl.where(projection_lane == channel, pre_mix, 0.0), axis=0
        )
        norm_input += value * pre_scalar
    norm_input_bf16 = norm_input.to(tl.bfloat16)
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=0)
        / JIT_FULL_HIDDEN
        + rmsnorm_eps
    )
    rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
    tl.store(y_bf16_ptr + h, y.to(tl.bfloat16))


def workspace_shapes(
    tokens: int, num_splits: int
) -> tuple[tuple[int, int, int], tuple[int]]:
    """Return the exact workspace and counter shapes for the probe."""
    _validate_geometry(tokens, num_splits)
    return (
        (tokens, num_splits, WORKSPACE_ROW_ELEMENTS),
        (tokens,),
    )


def workspace_nbytes(tokens: int, num_splits: int) -> int:
    """Return total bytes required by workspace plus uint64 tickets."""
    partial_shape, counter_shape = workspace_shapes(tokens, num_splits)
    return prod(partial_shape) * 4 + prod(counter_shape) * 8


def allocate_tle_sm90_mega_mhc_atomic_workspace(
    tokens: int,
    num_splits: int,
    *,
    device: torch.device | str,
) -> MegaMHCAtomicProtocolWorkspace:
    """Allocate and initialize protocol storage outside CUDA Graph capture.

    The partial workspace does not need initialization because every producer
    overwrites its complete padded row before publishing.  Counters do require
    one initial zeroing.  Every complete launch monotonically advances them by
    ``num_splits``, including every graph replay.
    """
    partial_shape, counter_shape = workspace_shapes(tokens, num_splits)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MegaMHC TLE protocol workspace must be allocated before CUDA "
            "Graph capture"
        )
    return MegaMHCAtomicProtocolWorkspace(
        partials_workspace=torch.empty(
            partial_shape, dtype=torch.float32, device=device
        ),
        counters=torch.zeros(counter_shape, dtype=torch.uint64, device=device),
    )


@triton.jit
def _tle_sm90_mega_mhc_atomic_protocol_kernel(
    mix_partials_ptr,
    sqsum_partials_ptr,
    partials_workspace_ptr,
    counters_ptr,
    reduced_mix_ptr,
    reduced_sqsum_ptr,
    NUM_SPLITS: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    """Publish one split and let the last producer reduce all split rows."""
    token = tl.program_id(0)
    split = tl.program_id(1)
    lane = tl.arange(0, JIT_WORKSPACE_ROW_ELEMENTS)

    if LAUNCH_PDL:
        # No input or workspace access may move above a graph dependency.
        tl.extra.cuda.gdc_wait()

    mix_mask = lane < JIT_NUM_MHC_OUTPUTS
    mix_offset = (
        (token * NUM_SPLITS + split) * JIT_NUM_MHC_OUTPUTS + lane
    )
    mix_value = tl.load(
        mix_partials_ptr + mix_offset,
        mask=mix_mask,
        other=0.0,
    ).to(tl.float32)

    sqsum_mask = lane == JIT_NUM_MHC_OUTPUTS
    sqsum_offset = token * NUM_SPLITS + split
    sqsum_scalar = tl.load(sqsum_partials_ptr + sqsum_offset).to(tl.float32)
    sqsum_value = tl.where(sqsum_mask, sqsum_scalar, 0.0)

    packed_partial = tl.where(mix_mask, mix_value, sqsum_value)
    packed_partial = tl.where(
        lane < JIT_NUM_MHC_OUTPUTS + JIT_NUM_SQRSUM_PARTIALS,
        packed_partial,
        0.0,
    )
    workspace_offset = (
        (token * NUM_SPLITS + split) * JIT_WORKSPACE_ROW_ELEMENTS + lane
    )
    tl.store(partials_workspace_ptr + workspace_offset, packed_partial)

    # A scalar Triton atomic is issued once per program.  This TLE barrier is
    # still required: it makes every worker thread's preceding global store
    # happen-before the one thread that publishes the CTA's arrival.
    publish_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_CTA_THREADS)
    tle.gpu.barrier_wait(publish_barrier)
    ticket = tl.atomic_add(
        counters_ptr + token,
        1,
        sem="acq_rel",
        scope="gpu",
    )

    is_last_generation_arrival = (ticket + 1) % NUM_SPLITS == 0
    if is_last_generation_arrival:
        # The acquiring atomic is scalar.  Propagate that acquire through a
        # CTA barrier before vector lanes read rows written by peer CTAs.
        acquire_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_CTA_THREADS)
        tle.gpu.barrier_wait(acquire_barrier)

        total = tl.zeros((JIT_WORKSPACE_ROW_ELEMENTS,), tl.float32)
        for source_split in tl.range(0, NUM_SPLITS):
            source_offset = (
                (token * NUM_SPLITS + source_split)
                * JIT_WORKSPACE_ROW_ELEMENTS
                + lane
            )
            total += tl.load(partials_workspace_ptr + source_offset)

        tl.store(
            reduced_mix_ptr + token * JIT_NUM_MHC_OUTPUTS + lane,
            total,
            mask=mix_mask,
        )
        reduced_sqsum = tl.sum(tl.where(sqsum_mask, total, 0.0), axis=0)
        tl.store(reduced_sqsum_ptr + token, reduced_sqsum)

        # PDL may release a dependent kernel, so make all vector stores visible
        # before the last-arriving CTA signals completion.  The monotonic
        # ticket itself is intentionally not reset here.
        finish_barrier = tle.gpu.alloc_barrier(arrive_count=JIT_CTA_THREADS)
        tle.gpu.barrier_wait(finish_barrier)
        if LAUNCH_PDL:
            tl.extra.cuda.gdc_launch_dependents()


def tle_sm90_mega_mhc_atomic_protocol_probe(
    mix_partials: torch.Tensor,
    sqsum_partials: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor],
    *,
    workspace: MegaMHCAtomicProtocolWorkspace,
    launch_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the one-kernel SM90 cross-CTA publication/reduction probe.

    Args:
        mix_partials: Contiguous FP32 ``[T, S, 24]`` producer values.
        sqsum_partials: Contiguous FP32 ``[T, S]`` producer statistics.
        out: Preallocated ``(reduced_mix[T, 24], reduced_sqsum[T])``.
        workspace: Preallocated private protocol workspace.  Its counters must
            be zero before first use.  Their generation must fit uint64, and
            the allocation must not be shared by overlapping launches or
            streams.
        launch_pdl: Enable programmatic dependent launch.  Keep false unless
            the surrounding graph intentionally uses PDL.

    The wrapper allocates nothing and launches exactly one kernel.  It cannot
    cheaply inspect a CUDA counter without adding synchronization, so the
    initial-zero, no-overflow, and non-overlap rules are caller-owned
    preconditions.
    """
    tokens, num_splits = _validate_host_contract(
        mix_partials, sqsum_partials, out, workspace
    )
    if not isinstance(launch_pdl, bool):
        raise TypeError("launch_pdl must be bool")

    reduced_mix, reduced_sqsum = out
    _tle_sm90_mega_mhc_atomic_protocol_kernel[(tokens, num_splits)](
        mix_partials,
        sqsum_partials,
        workspace.partials_workspace,
        workspace.counters,
        reduced_mix,
        reduced_sqsum,
        NUM_SPLITS=num_splits,
        LAUNCH_PDL=launch_pdl,
        num_warps=NUM_WARPS,
        num_stages=1,
        num_ctas=1,
        launch_pdl=launch_pdl,
    )
    return out


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> TLESM90MegaMHCPlan:
    """Prepare one supported DeepSeek-V4-Flash focused migration case.

    This entry point intentionally accepts only the revision-pinned Normal
    BF16 contract at T in {1, 2, 4, 8, 16, 24, 32}.  Preparation owns all
    workspace allocation; the returned plan's :meth:`run` contains one and
    only one Triton/TLE launch.
    """
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping")
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must be a mapping")

    required_inputs = {
        "x",
        "residual",
        "post_mix",
        "comb_res_mix",
        "shifted_prev_mix",
        "fn",
        "mix_scales",
        "mix_bases",
        "rmsnorm_weight",
        "hc_mult",
        "hc_norm_eps",
        "hc_pre_eps",
        "hc_post_scale",
        "sinkhorn_eps",
        "num_sinkhorn_iters",
        "rmsnorm_eps",
        "rmsnorm_scale",
    }
    missing_inputs = required_inputs - set(inputs)
    if missing_inputs:
        raise ValueError(f"inputs are missing {sorted(missing_inputs)}")
    required_outputs = {
        "new_residual",
        "new_prev_mix",
        "new_post_mix",
        "new_comb_res_mix",
        "y_bf16",
    }
    missing_outputs = required_outputs - set(outputs)
    if missing_outputs:
        raise ValueError(f"outputs are missing {sorted(missing_outputs)}")
    x_tensor = inputs["x"]
    if not isinstance(x_tensor, torch.Tensor):
        raise TypeError("inputs.x must be a torch.Tensor")
    if x_tensor.ndim != 2:
        raise ValueError("inputs.x must have rank 2")
    tokens = int(x_tensor.shape[0])
    if tokens not in SUPPORTED_FULL_TOKENS:
        raise ValueError(
            f"T={tokens} is unsupported; focused migrations support "
            f"T in {SUPPORTED_FULL_TOKENS}"
        )
    if inputs["shifted_prev_mix"] is not None:
        raise ValueError("the focused migration supports Normal mode only")
    if outputs["new_prev_mix"] is not None:
        raise ValueError("Normal mode requires outputs['new_prev_mix'] to be None")

    tensor_specs = (
        (inputs["x"], "inputs.x", (tokens, FULL_HIDDEN), torch.bfloat16),
        (
            inputs["residual"],
            "inputs.residual",
            (tokens, FULL_HC_MULT, FULL_HIDDEN),
            torch.bfloat16,
        ),
        (
            inputs["post_mix"],
            "inputs.post_mix",
            (tokens, FULL_HC_MULT, 1),
            torch.float32,
        ),
        (
            inputs["comb_res_mix"],
            "inputs.comb_res_mix",
            (tokens, FULL_HC_MULT, FULL_HC_MULT),
            torch.float32,
        ),
        (
            inputs["fn"],
            "inputs.fn",
            (FULL_PROJECTION_OUTPUTS, FULL_FLAT_HIDDEN),
            torch.float32,
        ),
        (inputs["mix_scales"], "inputs.mix_scales", (3,), torch.float32),
        (
            inputs["mix_bases"],
            "inputs.mix_bases",
            (FULL_PROJECTION_OUTPUTS,),
            torch.float32,
        ),
        (
            inputs["rmsnorm_weight"],
            "inputs.rmsnorm_weight",
            (FULL_HIDDEN,),
            torch.bfloat16,
        ),
        (
            outputs["new_residual"],
            "outputs.new_residual",
            (tokens, FULL_HC_MULT, FULL_HIDDEN),
            torch.bfloat16,
        ),
        (
            outputs["new_post_mix"],
            "outputs.new_post_mix",
            (tokens, FULL_HC_MULT, 1),
            torch.float32,
        ),
        (
            outputs["new_comb_res_mix"],
            "outputs.new_comb_res_mix",
            (tokens, FULL_HC_MULT, FULL_HC_MULT),
            torch.float32,
        ),
        (
            outputs["y_bf16"],
            "outputs.y_bf16",
            (tokens, FULL_HIDDEN),
            torch.bfloat16,
        ),
    )
    for tensor, name, shape, dtype in tensor_specs:
        _validate_tensor(tensor, name=name, shape=shape, dtype=dtype)
    devices = {tensor.device for tensor, _, _, _ in tensor_specs}
    if len(devices) != 1:
        raise ValueError("all MegaMHC inputs and outputs must share a device")
    _validate_no_overlap(
        (name, tensor) for tensor, name, _, _ in tensor_specs
    )

    expected_scalars = {
        "hc_mult": FULL_HC_MULT,
        "hc_norm_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_post_scale": 2.0,
        "sinkhorn_eps": 1e-6,
        "num_sinkhorn_iters": FULL_SINKHORN_ITERS,
        "rmsnorm_eps": 1e-6,
        "rmsnorm_scale": 1.0,
    }
    for name, expected in expected_scalars.items():
        actual = inputs[name]
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise TypeError(f"inputs.{name} must be a Python scalar")
        if actual != expected:
            raise ValueError(
                f"inputs.{name}={actual!r}, expected {expected!r} for "
                "DeepSeek-V4-Flash Normal BF16"
            )

    default_schedule = (
        "split16_tile4_last_arrival"
        if tokens == 1
        else "split8_tile3_last_arrival"
        if tokens == 2
        else "split8_tile6_last_arrival"
        if tokens in (4, 8)
        else "split8_tile24_last_arrival"
        if tokens == 32
        else "split8_tile12_last_arrival"
    )
    schedule_override = os.environ.get("DG_TLE_MHC_SCHEDULE")
    requested_schedule = (
        schedule_override if schedule_override is not None else default_schedule
    )
    reduction_override = os.environ.get("DG_TLE_MHC_REDUCTION")
    sinkhorn_override = os.environ.get("DG_TLE_MHC_SINKHORN")
    valid_sinkhorns = (
        "triton_masked_reductions",
        "warp_shuffle_4x4",
        "warp_shuffle_4x4_fast_rcp",
    )
    if sinkhorn_override is not None and sinkhorn_override not in valid_sinkhorns:
        raise ValueError(
            "DG_TLE_MHC_SINKHORN must be triton_masked_reductions or "
            "warp_shuffle_4x4 or warp_shuffle_4x4_fast_rcp"
        )
    overlap_override = os.environ.get("DG_TLE_MHC_OVERLAP")
    valid_overlaps = ("none", "routing_worker_1wg")
    if overlap_override is not None and overlap_override not in valid_overlaps:
        raise ValueError(
            "DG_TLE_MHC_OVERLAP must be none or routing_worker_1wg"
        )
    handoff_override = os.environ.get("DG_TLE_MHC_HANDOFF")
    valid_handoffs = (
        "triton_atomic_u64",
        "ptx_release_inc_u32_bar_red",
    )
    if handoff_override is not None and handoff_override not in valid_handoffs:
        raise ValueError(
            "DG_TLE_MHC_HANDOFF must be triton_atomic_u64 or "
            "ptx_release_inc_u32_bar_red"
        )
    phase_profile_mode = os.environ.get("DG_TLE_MHC_PHASE_PROFILE", "0")
    if phase_profile_mode not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_PHASE_PROFILE must be 0 or 1")
    phase_profiling = phase_profile_mode == "1"
    contiguous_partials_override = os.environ.get(
        "DG_TLE_MHC_T2_CONTIGUOUS_PARTIALS"
    )
    if contiguous_partials_override not in (None, "0", "1"):
        raise ValueError(
            "DG_TLE_MHC_T2_CONTIGUOUS_PARTIALS must be 0 or 1"
        )
    use_contiguous_partials = contiguous_partials_override == "1"
    projection_ilp_override = os.environ.get(
        "DG_TLE_MHC_T2_PROJECTION_ILP"
    )
    if projection_ilp_override not in (None, "0", "1"):
        raise ValueError("DG_TLE_MHC_T2_PROJECTION_ILP must be 0 or 1")
    use_t2_projection_ilp = projection_ilp_override == "1"
    sqsum_tree_override = os.environ.get("DG_TLE_MHC_T2_SQSUM_TREE")
    if sqsum_tree_override not in (None, "0", "1"):
        raise ValueError("DG_TLE_MHC_T2_SQSUM_TREE must be 0 or 1")
    use_t2_sqsum_tree = sqsum_tree_override == "1"
    t32_projection_ilp_override = os.environ.get(
        "DG_TLE_MHC_T32_PROJECTION_ILP"
    )
    if t32_projection_ilp_override not in (None, "0", "1"):
        raise ValueError("DG_TLE_MHC_T32_PROJECTION_ILP must be 0 or 1")
    use_t32_projection_ilp = t32_projection_ilp_override == "1"
    t32_projection_ilp2_override = os.environ.get(
        "DG_TLE_MHC_T32_PROJECTION_ILP2"
    )
    if t32_projection_ilp2_override not in (None, "0", "1"):
        raise ValueError("DG_TLE_MHC_T32_PROJECTION_ILP2 must be 0 or 1")
    use_t32_projection_ilp2 = t32_projection_ilp2_override == "1"
    t32_custom_store1_override = os.environ.get(
        "DG_TLE_MHC_T32_CUSTOM_STORE1"
    )
    if t32_custom_store1_override not in (None, "0", "1"):
        raise ValueError("DG_TLE_MHC_T32_CUSTOM_STORE1 must be 0 or 1")
    use_t32_custom_store1 = t32_custom_store1_override == "1"
    t32_pack_sqsum_override = os.environ.get(
        "DG_TLE_MHC_T32_PACK_SQSUM"
    )
    if t32_pack_sqsum_override not in (None, "0", "1"):
        raise ValueError("DG_TLE_MHC_T32_PACK_SQSUM must be 0 or 1")
    use_t32_pack_sqsum = t32_pack_sqsum_override == "1"
    t32_skip_epilogue_barrier_override = os.environ.get(
        "DG_TLE_MHC_T32_SKIP_EPILOGUE_BARRIER"
    )
    if t32_skip_epilogue_barrier_override not in (None, "0", "1"):
        raise ValueError(
            "DG_TLE_MHC_T32_SKIP_EPILOGUE_BARRIER must be 0 or 1"
        )
    use_t32_skip_epilogue_barrier = (
        t32_skip_epilogue_barrier_override == "1"
    )
    t32_premix_direct_bcast_override = os.environ.get(
        "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST"
    )
    if t32_premix_direct_bcast_override not in (None, "0", "1"):
        raise ValueError(
            "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST must be 0 or 1"
        )
    use_t32_premix_direct_bcast = (
        t32_premix_direct_bcast_override == "1"
    )
    t32_premix_shared_bridge_override = os.environ.get(
        "DG_TLE_MHC_T32_PREMIX_SHARED_BRIDGE"
    )
    if t32_premix_shared_bridge_override not in (None, "0", "1"):
        raise ValueError(
            "DG_TLE_MHC_T32_PREMIX_SHARED_BRIDGE must be 0 or 1"
        )
    use_t32_premix_shared_bridge = (
        t32_premix_shared_bridge_override == "1"
    )
    if (
        use_t32_premix_shared_bridge
        and not use_t32_premix_direct_bcast
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_PREMIX_SHARED_BRIDGE=1 requires "
            "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST=1"
        )
    t32_warp0_handoff_barrier_override = os.environ.get(
        "DG_TLE_MHC_T32_WARP0_HANDOFF_BARRIER"
    )
    if t32_warp0_handoff_barrier_override not in (None, "0", "1"):
        raise ValueError(
            "DG_TLE_MHC_T32_WARP0_HANDOFF_BARRIER must be 0 or 1"
        )
    use_t32_warp0_handoff_barrier = (
        t32_warp0_handoff_barrier_override == "1"
    )
    t32_fn_cache_cg_override = os.environ.get(
        "DG_TLE_MHC_T32_FN_CACHE_CG"
    )
    if t32_fn_cache_cg_override not in (None, "0", "1"):
        raise ValueError("DG_TLE_MHC_T32_FN_CACHE_CG must be 0 or 1")
    use_t32_fn_cache_cg = t32_fn_cache_cg_override == "1"
    t32_token_pair_override = os.environ.get(
        "DG_TLE_MHC_T32_TOKEN_PAIR"
    )
    if t32_token_pair_override not in (None, "0", "1"):
        raise ValueError("DG_TLE_MHC_T32_TOKEN_PAIR must be 0 or 1")
    use_t32_token_pair = t32_token_pair_override == "1"
    if (
        t32_pack_sqsum_override == "1"
        and t32_custom_store1_override == "0"
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_PACK_SQSUM=1 requires "
            "DG_TLE_MHC_T32_CUSTOM_STORE1!=0"
        )
    if t32_premix_direct_bcast_override == "1" and (
        t32_custom_store1_override == "0"
        or t32_pack_sqsum_override == "0"
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST=1 requires "
            "DG_TLE_MHC_T32_CUSTOM_STORE1!=0 and "
            "DG_TLE_MHC_T32_PACK_SQSUM!=0"
        )
    if use_t32_custom_store1 and use_t32_projection_ilp:
        raise ValueError(
            "DG_TLE_MHC_T32_CUSTOM_STORE1=1 and "
            "DG_TLE_MHC_T32_PROJECTION_ILP=1 are mutually exclusive"
        )
    if use_t32_projection_ilp2 and (
        use_t32_projection_ilp
        or use_t32_token_pair
        or use_t32_fn_cache_cg
        or use_t32_pack_sqsum
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_PROJECTION_ILP2=1 is mutually exclusive with "
            "DG_TLE_MHC_T32_PROJECTION_ILP, "
            "DG_TLE_MHC_T32_TOKEN_PAIR, and "
            "DG_TLE_MHC_T32_FN_CACHE_CG or DG_TLE_MHC_T32_PACK_SQSUM"
        )
    if use_t32_pack_sqsum and (
        use_t32_projection_ilp
        or use_t32_projection_ilp2
        or use_t32_token_pair
        or use_t32_fn_cache_cg
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_PACK_SQSUM=1 is mutually exclusive with "
            "DG_TLE_MHC_T32_PROJECTION_ILP, "
            "DG_TLE_MHC_T32_PROJECTION_ILP2, "
            "DG_TLE_MHC_T32_TOKEN_PAIR, and "
            "DG_TLE_MHC_T32_FN_CACHE_CG"
        )
    if use_t32_skip_epilogue_barrier and (
        use_t32_projection_ilp
        or use_t32_projection_ilp2
        or use_t32_token_pair
        or use_t32_fn_cache_cg
        or use_t32_premix_direct_bcast
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_SKIP_EPILOGUE_BARRIER=1 is mutually exclusive "
            "with DG_TLE_MHC_T32_PROJECTION_ILP, "
            "DG_TLE_MHC_T32_PROJECTION_ILP2, "
            "DG_TLE_MHC_T32_TOKEN_PAIR, DG_TLE_MHC_T32_FN_CACHE_CG, and "
            "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST"
        )
    if use_t32_premix_direct_bcast and (
        use_t32_projection_ilp
        or use_t32_projection_ilp2
        or use_t32_skip_epilogue_barrier
        or use_t32_token_pair
        or use_t32_fn_cache_cg
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST=1 is mutually exclusive "
            "with DG_TLE_MHC_T32_PROJECTION_ILP, "
            "DG_TLE_MHC_T32_PROJECTION_ILP2, "
            "DG_TLE_MHC_T32_SKIP_EPILOGUE_BARRIER, "
            "DG_TLE_MHC_T32_TOKEN_PAIR, and DG_TLE_MHC_T32_FN_CACHE_CG"
        )
    if use_t32_warp0_handoff_barrier and (
        use_t32_projection_ilp
        or use_t32_projection_ilp2
        or use_t32_skip_epilogue_barrier
        or use_t32_premix_direct_bcast
        or use_t32_token_pair
        or use_t32_fn_cache_cg
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_WARP0_HANDOFF_BARRIER=1 is mutually exclusive "
            "with DG_TLE_MHC_T32_PROJECTION_ILP, "
            "DG_TLE_MHC_T32_PROJECTION_ILP2, "
            "DG_TLE_MHC_T32_SKIP_EPILOGUE_BARRIER, "
            "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST, "
            "DG_TLE_MHC_T32_TOKEN_PAIR, and DG_TLE_MHC_T32_FN_CACHE_CG"
        )
    if use_t32_token_pair and (
        use_t32_projection_ilp
        or use_t32_projection_ilp2
        or use_t32_custom_store1
        or use_t32_fn_cache_cg
        or use_t32_pack_sqsum
        or use_t32_premix_direct_bcast
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_TOKEN_PAIR=1 is mutually exclusive with "
            "DG_TLE_MHC_T32_PROJECTION_ILP, "
            "DG_TLE_MHC_T32_PROJECTION_ILP2, "
            "DG_TLE_MHC_T32_CUSTOM_STORE1, and "
            "DG_TLE_MHC_T32_FN_CACHE_CG, DG_TLE_MHC_T32_PACK_SQSUM, or "
            "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST"
        )
    t32_experimental_probe_requested = any(
        (
            use_t32_projection_ilp,
            use_t32_projection_ilp2,
            use_t32_skip_epilogue_barrier,
            use_t32_premix_shared_bridge,
            use_t32_warp0_handoff_barrier,
            use_t32_fn_cache_cg,
            use_t32_token_pair,
        )
    )
    t32_split8_tile24_default_selected = False
    t32_shared25_default_selected = False
    t32_fast_rcp_default_selected = False
    t32_custom_handoff_default_selected = False
    t32_custom_store1_default_selected = False
    t32_pack_sqsum_default_selected = False
    t32_premix_direct_bcast_default_selected = False
    t2_contiguous_partials_default_selected = False
    t2_projection_ilp_default_selected = False
    valid_reductions = (
        "triton_cta_sum",
        "warp_shuffle_shared4_tid4",
        "warp_shuffle_shared5",
        "warp_shuffle_shared5_tid5",
        "warp_shuffle_shared7_tid7",
        "warp_shuffle_shared9_tid9",
        "warp_shuffle_shared13_tid13",
        "warp_shuffle_shared25_tid25",
    )
    if (
        reduction_override is not None
        and reduction_override not in valid_reductions
    ):
        raise ValueError(
            "DG_TLE_MHC_REDUCTION must be triton_cta_sum or "
            "warp_shuffle_shared4_tid4 or warp_shuffle_shared5 or "
            "warp_shuffle_shared5_tid5 or "
            "warp_shuffle_shared7_tid7 or warp_shuffle_shared9_tid9 or "
            "warp_shuffle_shared13_tid13 or warp_shuffle_shared25_tid25"
        )
    split_schedule_tiles = {
        "split16_tile3_last_arrival": (3, 4),
        "split16_tile4_last_arrival": (4, 4),
        "split16_tile6_last_arrival": (6, 8),
        "split16_tile8_last_arrival": (8, 8),
        "split16_tile12_last_arrival": (12, 16),
        "split16_tile24_last_arrival": (24, 32),
        "split8_tile3_last_arrival": (3, 4),
        "split8_tile6_last_arrival": (6, 8),
        "split8_tile8_last_arrival": (8, 8),
        "split8_tile12_last_arrival": (12, 16),
        "split8_tile24_last_arrival": (24, 32),
    }
    if requested_schedule not in (
        *split_schedule_tiles,
        "single_cta_t1_correctness_fallback",
    ):
        raise ValueError(
            "DG_TLE_MHC_SCHEDULE must select split16_tile{3,4,6,8,12,24}_"
            "last_arrival, split8_tile{3,6,8,12,24}_last_arrival, or "
            "single_cta_t1_correctness_fallback"
        )
    if tokens == 1 and requested_schedule.startswith("split8_tile"):
        raise ValueError(
            "split8 schedules require T in {2, 4, 8, 16, 24, 32}"
        )
    if tokens == 2 and requested_schedule not in (
        "split16_tile4_last_arrival",
        "split8_tile3_last_arrival",
    ):
        raise ValueError(
            "T=2 focused migration currently requires "
            "split16_tile4_last_arrival or split8_tile3_last_arrival"
        )
    if tokens == 4 and requested_schedule != "split8_tile6_last_arrival":
        raise ValueError(
            "T=4 focused migration currently requires "
            "split8_tile6_last_arrival"
        )
    if tokens == 8 and requested_schedule not in (
        "split8_tile6_last_arrival",
        "split8_tile8_last_arrival",
        "split8_tile12_last_arrival",
        "split16_tile12_last_arrival",
    ):
        raise ValueError(
            "T=8 focused migration requires split8_tile6_last_arrival, "
            "split8_tile8_last_arrival, split8_tile12_last_arrival, or "
            "split16_tile12_last_arrival"
        )
    if tokens == 16 and requested_schedule not in (
        "split8_tile6_last_arrival",
        "split8_tile12_last_arrival",
    ):
        raise ValueError(
            "T=16 focused migration currently requires "
            "split8_tile6_last_arrival or split8_tile12_last_arrival"
        )
    if tokens == 24 and requested_schedule != "split8_tile12_last_arrival":
        raise ValueError(
            "T=24 focused migration currently requires "
            "split8_tile12_last_arrival"
        )
    if tokens == 32 and requested_schedule not in (
        "split8_tile12_last_arrival",
        "split8_tile24_last_arrival",
    ):
        raise ValueError(
            "T=32 focused migration currently requires "
            "split8_tile12_last_arrival or split8_tile24_last_arrival"
        )
    if requested_schedule in split_schedule_tiles:
        default_split_num_warps = (
            16
            if tokens == 2
            and requested_schedule == "split8_tile3_last_arrival"
            else FULL_NUM_WARPS
        )
        try:
            split_num_warps = int(
                os.environ.get(
                    "DG_TLE_MHC_NUM_WARPS", str(default_split_num_warps)
                )
            )
        except ValueError as exc:
            raise ValueError(
                "DG_TLE_MHC_NUM_WARPS must be 4, 8, or 16"
            ) from exc
        if split_num_warps not in (4, 8, 16):
            raise ValueError("DG_TLE_MHC_NUM_WARPS must be 4, 8, or 16")
        if split_num_warps == 16 and not (
            (
                tokens == 2
                and requested_schedule == "split8_tile3_last_arrival"
            )
            or (
                tokens == 8
                and requested_schedule == "split8_tile12_last_arrival"
            )
            or (
                tokens == 32
                and requested_schedule == "split8_tile24_last_arrival"
                and reduction_override == "warp_shuffle_shared25_tid25"
            )
        ):
            raise ValueError(
                "DG_TLE_MHC_NUM_WARPS=16 is supported only for the T=2 "
                "split8_tile3_last_arrival or T=8 "
                "split8_tile12_last_arrival, or T=32 "
                "split8_tile24_last_arrival shared25 candidate"
            )
        if (
            tokens == 2
            and requested_schedule == "split8_tile3_last_arrival"
            and split_num_warps not in (FULL_NUM_WARPS, 16)
        ):
            raise ValueError(
                "the T=2 split8_tile3_last_arrival candidate requires "
                "DG_TLE_MHC_NUM_WARPS=8 or 16"
            )
        requires_eight_warps = (
            tokens in (4, 8)
            and requested_schedule in (
                "split8_tile6_last_arrival",
                "split8_tile8_last_arrival",
                "split16_tile12_last_arrival",
            )
        ) or (
            tokens in (16, 24, 32)
            and requested_schedule in (
                "split8_tile6_last_arrival",
                "split8_tile12_last_arrival",
            )
        ) or (
            tokens == 32
            and requested_schedule == "split8_tile24_last_arrival"
            and not (
                split_num_warps == 16
                and reduction_override == "warp_shuffle_shared25_tid25"
            )
        )
        if requires_eight_warps and split_num_warps != FULL_NUM_WARPS:
            raise ValueError(
                f"the T={tokens} {requested_schedule} candidate requires "
                "DG_TLE_MHC_NUM_WARPS=8"
            )
        outputs_per_tile, output_tile_block = split_schedule_tiles[
            requested_schedule
        ]
        num_splits = (
            T8_NUM_SPLITS
            if requested_schedule.startswith("split8_tile")
            else FULL_NUM_SPLITS
        )
        hidden_per_split = FULL_HIDDEN // num_splits
        num_output_tiles = FULL_PROJECTION_OUTPUTS // outputs_per_tile
        ctas_per_token = num_splits * num_output_tiles
        grid_ctas = tokens * ctas_per_token
        hierarchical_geometry = (
            tokens in (1, 2)
            and num_splits == FULL_NUM_SPLITS
            and outputs_per_tile == 4
            and output_tile_block == 4
            and split_num_warps == FULL_NUM_WARPS
        )
        t2_tile3_geometry = (
            tokens == 2
            and num_splits == T8_NUM_SPLITS
            and outputs_per_tile == T2_TILE3_OUTPUTS_PER_TILE
            and output_tile_block == T2_TILE3_OUTPUT_BLOCK
            and split_num_warps in (FULL_NUM_WARPS, 16)
        )
        t2_tile3_16warp_geometry = (
            t2_tile3_geometry and split_num_warps == 16
        )
        t8_migration_geometry = (
            tokens == 8
            and num_splits == T8_NUM_SPLITS
            and outputs_per_tile == T8_OUTPUTS_PER_TILE
            and output_tile_block == T8_OUTPUT_TILE_BLOCK
            and split_num_warps in (FULL_NUM_WARPS, 16)
        )
        t8_16warp_candidate_geometry = (
            t8_migration_geometry and split_num_warps == 16
        )
        t8_8warp_geometry = (
            t8_migration_geometry and split_num_warps == FULL_NUM_WARPS
        )
        t8_split16_8warp_geometry = (
            tokens == 8
            and num_splits == FULL_NUM_SPLITS
            and outputs_per_tile == T8_OUTPUTS_PER_TILE
            and output_tile_block == T8_OUTPUT_TILE_BLOCK
            and split_num_warps == FULL_NUM_WARPS
        )
        t8_shared13_geometry = (
            t8_8warp_geometry or t8_split16_8warp_geometry
        )
        t16plus_split8_tile12_8warp_geometry = (
            tokens in (16, 24, 32)
            and num_splits == T8_NUM_SPLITS
            and outputs_per_tile == T8_OUTPUTS_PER_TILE
            and output_tile_block == T8_OUTPUT_TILE_BLOCK
            and split_num_warps == FULL_NUM_WARPS
        )
        t32_split8_tile24_8warp_geometry = (
            tokens == 32
            and num_splits == T8_NUM_SPLITS
            and outputs_per_tile == FULL_PROJECTION_OUTPUTS
            and output_tile_block == 32
            and num_output_tiles == 1
            and split_num_warps == FULL_NUM_WARPS
        )
        t32_split8_tile24_16warp_geometry = (
            tokens == 32
            and num_splits == T8_NUM_SPLITS
            and outputs_per_tile == FULL_PROJECTION_OUTPUTS
            and output_tile_block == 32
            and num_output_tiles == 1
            and split_num_warps == 16
        )
        shared13_geometry = (
            t8_shared13_geometry or t16plus_split8_tile12_8warp_geometry
        )
        split8_tile6_8warp_geometry = (
            tokens in (4, 8, 16)
            and num_splits == T8_NUM_SPLITS
            and outputs_per_tile == T8_TILE6_OUTPUTS_PER_TILE
            and output_tile_block == T8_TILE6_OUTPUT_BLOCK
            and split_num_warps == FULL_NUM_WARPS
        )
        t8_tile8_8warp_geometry = (
            tokens == 8
            and num_splits == T8_NUM_SPLITS
            and outputs_per_tile == T8_TILE8_OUTPUTS_PER_TILE
            and output_tile_block == T8_TILE8_OUTPUT_BLOCK
            and split_num_warps == FULL_NUM_WARPS
        )
        shared13_reduction_requested = (
            reduction_override == "warp_shuffle_shared13_tid13"
        )
        t2_shared4_reduction_requested = (
            reduction_override == "warp_shuffle_shared4_tid4"
        )
        t8_shared7_reduction_requested = (
            reduction_override == "warp_shuffle_shared7_tid7"
        )
        t8_shared9_reduction_requested = (
            reduction_override == "warp_shuffle_shared9_tid9"
        )
        t32_shared25_reduction_requested = (
            reduction_override == "warp_shuffle_shared25_tid25"
        )
        fast_reduction_requested = (
            t2_shared4_reduction_requested and t2_tile3_geometry
        ) or (
            shared13_reduction_requested and shared13_geometry
        ) or (
            t8_shared7_reduction_requested and split8_tile6_8warp_geometry
        ) or (
            t8_shared9_reduction_requested and t8_tile8_8warp_geometry
        ) or (
            t32_shared25_reduction_requested
            and (
                t32_split8_tile24_8warp_geometry
                or t32_split8_tile24_16warp_geometry
            )
        )
        tuning_override_present = any(
            override is not None
            for override in (
                reduction_override,
                sinkhorn_override,
                overlap_override,
                handoff_override,
            )
        )
        t2_fast_path_auto_selected = (
            t2_tile3_16warp_geometry
            and not phase_profiling
            and not tuning_override_present
        )
        if contiguous_partials_override is None:
            use_contiguous_partials = t2_fast_path_auto_selected
            t2_contiguous_partials_default_selected = (
                use_contiguous_partials
            )
        if projection_ilp_override is None:
            use_t2_projection_ilp = t2_fast_path_auto_selected
            t2_projection_ilp_default_selected = use_t2_projection_ilp
        t8_16warp_candidate_selected = (
            t8_16warp_candidate_geometry
            and not phase_profiling
            and not tuning_override_present
        )
        t2_tile3_candidate_selected = (
            t2_tile3_geometry
            and not phase_profiling
            and not tuning_override_present
        )
        t2_tile3_16warp_candidate_selected = (
            t2_tile3_16warp_geometry
            and not phase_profiling
            and not tuning_override_present
        )
        t32_split8_tile24_default_selected = (
            t32_split8_tile24_8warp_geometry
            and schedule_override is None
            and not phase_profiling
            and not tuning_override_present
        )
        t32_tile24_candidate_selected = (
            t32_split8_tile24_8warp_geometry
            and not t32_split8_tile24_default_selected
            and not phase_profiling
            and not tuning_override_present
        )
        production_default_selected = (
            tokens == 1
            and requested_schedule == "split16_tile4_last_arrival"
            and split_num_warps == FULL_NUM_WARPS
            and not phase_profiling
            and not tuning_override_present
        )
        t2_overlap_default_selected = (
            tokens == 2
            and requested_schedule == "split16_tile4_last_arrival"
            and split_num_warps == FULL_NUM_WARPS
            and not phase_profiling
            and not tuning_override_present
        )
        focused_migration_default_selected = (
            (
                t2_overlap_default_selected
                or t2_tile3_16warp_geometry
                or t8_migration_geometry
                or split8_tile6_8warp_geometry
                or t16plus_split8_tile12_8warp_geometry
                or t32_split8_tile24_default_selected
            )
            and (
                t2_tile3_16warp_geometry
                or split_num_warps == FULL_NUM_WARPS
            )
            and not phase_profiling
            and not tuning_override_present
        )
        if production_default_selected or t2_overlap_default_selected:
            requested_reduction = "warp_shuffle_shared5_tid5"
            requested_sinkhorn = "warp_shuffle_4x4_fast_rcp"
            requested_overlap = "routing_worker_1wg"
            requested_handoff = "ptx_release_inc_u32_bar_red"
        elif t2_tile3_candidate_selected:
            requested_reduction = "warp_shuffle_shared4_tid4"
            requested_sinkhorn = "warp_shuffle_4x4_fast_rcp"
            requested_overlap = "none"
            requested_handoff = "ptx_release_inc_u32_bar_red"
        elif t32_split8_tile24_default_selected:
            requested_reduction = "warp_shuffle_shared25_tid25"
            requested_sinkhorn = "warp_shuffle_4x4_fast_rcp"
            requested_overlap = "none"
            requested_handoff = "ptx_release_inc_u32_bar_red"
            t32_shared25_default_selected = True
            t32_fast_rcp_default_selected = True
            t32_custom_handoff_default_selected = True
        elif t32_tile24_candidate_selected:
            # Keep the explicit schedule-only candidate available for
            # historical A/B harnesses.  Production-like no-override T=32
            # dispatch is selected by the promoted branch above.
            requested_reduction = "triton_cta_sum"
            requested_sinkhorn = "warp_shuffle_4x4_fast_rcp"
            requested_overlap = "none"
            requested_handoff = "ptx_release_inc_u32_bar_red"
        elif (
            focused_migration_default_selected
            or t8_16warp_candidate_selected
        ):
            requested_reduction = (
                "warp_shuffle_shared7_tid7"
                if split8_tile6_8warp_geometry
                else "warp_shuffle_shared13_tid13"
                if t16plus_split8_tile12_8warp_geometry
                else "triton_cta_sum"
            )
            requested_sinkhorn = "warp_shuffle_4x4_fast_rcp"
            requested_overlap = "none"
            requested_handoff = "ptx_release_inc_u32_bar_red"
        else:
            requested_reduction = (
                reduction_override
                if reduction_override is not None
                else "warp_shuffle_shared5"
                if hierarchical_geometry
                else "triton_cta_sum"
            )
            requested_sinkhorn = (
                sinkhorn_override
                if sinkhorn_override is not None
                else "warp_shuffle_4x4_fast_rcp"
                if fast_reduction_requested
                else "triton_masked_reductions"
            )
            requested_overlap = (
                overlap_override if overlap_override is not None else "none"
            )
            requested_handoff = (
                handoff_override
                if handoff_override is not None
                else "ptx_release_inc_u32_bar_red"
                if fast_reduction_requested
                else "triton_atomic_u64"
            )
        t32_shared25_candidate_selected = (
            (
                t32_split8_tile24_8warp_geometry
                or t32_split8_tile24_16warp_geometry
            )
            and requested_reduction == "warp_shuffle_shared25_tid25"
            and not phase_profiling
            and sinkhorn_override is None
            and overlap_override is None
            and handoff_override is None
        )
        t32_shared25_8warp_candidate_selected = (
            t32_shared25_candidate_selected
            and t32_split8_tile24_8warp_geometry
        )
        if (
            t32_split8_tile24_default_selected
            and not t32_experimental_probe_requested
        ):
            use_t32_custom_store1 = t32_custom_store1_override != "0"
            use_t32_pack_sqsum = (
                use_t32_custom_store1
                and t32_pack_sqsum_override != "0"
            )
            use_t32_premix_direct_bcast = (
                use_t32_pack_sqsum
                and t32_premix_direct_bcast_override != "0"
            )
            t32_custom_store1_default_selected = (
                use_t32_custom_store1
                and t32_custom_store1_override is None
            )
            t32_pack_sqsum_default_selected = (
                use_t32_pack_sqsum and t32_pack_sqsum_override is None
            )
            t32_premix_direct_bcast_default_selected = (
                use_t32_premix_direct_bcast
                and t32_premix_direct_bcast_override is None
            )
        use_warp_sinkhorn = requested_sinkhorn in (
            "warp_shuffle_4x4",
            "warp_shuffle_4x4_fast_rcp",
        )
        use_fast_rcp = requested_sinkhorn == "warp_shuffle_4x4_fast_rcp"
        use_hierarchical_reduction = (
            requested_reduction
            in ("warp_shuffle_shared5", "warp_shuffle_shared5_tid5")
        )
        use_tid4_reduction = (
            requested_reduction == "warp_shuffle_shared4_tid4"
        )
        use_tid5_reduction = (
            requested_reduction == "warp_shuffle_shared5_tid5"
        )
        use_tid7_reduction = (
            requested_reduction == "warp_shuffle_shared7_tid7"
        )
        use_tid9_reduction = (
            requested_reduction == "warp_shuffle_shared9_tid9"
        )
        use_tid13_reduction = (
            requested_reduction == "warp_shuffle_shared13_tid13"
        )
        use_tid25_reduction = (
            requested_reduction == "warp_shuffle_shared25_tid25"
        )
        if use_tid4_reduction and not t2_tile3_geometry:
            raise ValueError(
                "warp_shuffle_shared4_tid4 requires the T=2 "
                "split8/tile3 8- or 16-warp schedule"
            )
        if use_tid7_reduction and not split8_tile6_8warp_geometry:
            raise ValueError(
                "warp_shuffle_shared7_tid7 requires the T in {4, 8, 16} "
                "split8/tile6 8-warp schedule"
            )
        if use_tid9_reduction and not t8_tile8_8warp_geometry:
            raise ValueError(
                "warp_shuffle_shared9_tid9 requires the T=8 "
                "split8/tile8 8-warp schedule"
            )
        if use_tid13_reduction and not shared13_geometry:
            raise ValueError(
                "warp_shuffle_shared13_tid13 requires a T=8 "
                "split8-or-split16/tile12 schedule or a T in {16, 24, 32} "
                "split8/tile12 schedule, all with an 8-warp schedule"
            )
        if use_tid25_reduction and not (
            t32_split8_tile24_8warp_geometry
            or t32_split8_tile24_16warp_geometry
        ):
            raise ValueError(
                "warp_shuffle_shared25_tid25 requires the T=32 "
                "split8/tile24 8- or 16-warp schedule"
            )
        use_custom_handoff = (
            requested_handoff == "ptx_release_inc_u32_bar_red"
        )
        if use_hierarchical_reduction and not (
            hierarchical_geometry
        ):
            raise ValueError(
                "warp-shuffle reductions currently require the "
                "tile4/8-warp split schedule"
            )
        use_warp_overlap = requested_overlap == "routing_worker_1wg"
        if use_warp_overlap and not (
            hierarchical_geometry
            and use_hierarchical_reduction
            and requested_sinkhorn == "warp_shuffle_4x4_fast_rcp"
            and not phase_profiling
        ):
            raise ValueError(
                "routing_worker_1wg requires split16_tile4_last_arrival, "
                "8 default warps, warp_shuffle_shared5, "
                "warp_shuffle_4x4_fast_rcp, and phase profiling disabled"
            )
        if use_tid25_reduction and not (
            use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and not phase_profiling
        ):
            raise ValueError(
                "warp_shuffle_shared25_tid25 requires the fast-rcp, "
                "no-overlap, custom-u32, phase-profiling-disabled tail"
            )
        if use_custom_handoff and not (
            (
                (hierarchical_geometry and use_tid5_reduction)
                or (
                    (
                        t8_migration_geometry
                        or t2_tile3_geometry
                        or t8_split16_8warp_geometry
                        or t16plus_split8_tile12_8warp_geometry
                        or t32_split8_tile24_8warp_geometry
                        or t32_split8_tile24_16warp_geometry
                        or split8_tile6_8warp_geometry
                        or t8_tile8_8warp_geometry
                    )
                    and not use_hierarchical_reduction
                    and requested_sinkhorn == "warp_shuffle_4x4_fast_rcp"
                    and requested_overlap == "none"
                )
            )
            and not phase_profiling
        ):
            raise ValueError(
                "ptx_release_inc_u32_bar_red requires either the T in {1, 2} "
                "tile4/8-warp warp_shuffle_shared5_tid5 schedule, the "
                "T=2 split8/tile3 8- or 16-warp schedule, the T in "
                "{4, 8, 16} split8/tile6 schedule, a T in {16, 24, 32} "
                "split8/tile12 schedule, the T=32 split8/tile24 schedule, "
                "or a supported T=8 split8/tile8, "
                "split8/tile12, or split16/tile12 reduction schedule, with "
                "phase profiling disabled"
            )
        if use_contiguous_partials and not (
            t2_tile3_16warp_geometry
            and use_tid4_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T2_CONTIGUOUS_PARTIALS=1 requires the T=2 "
                "split8_tile3_last_arrival 16-warp shared4/fast-rcp/"
                "no-overlap/custom-u32 candidate"
            )
        if use_t2_projection_ilp and not (
            t2_tile3_16warp_geometry
            and use_tid4_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T2_PROJECTION_ILP=1 requires the T=2 "
                "split8_tile3_last_arrival 16-warp shared4/fast-rcp/"
                "no-overlap/custom-u32 candidate"
            )
        if use_t2_sqsum_tree and not (
            t2_tile3_16warp_geometry
            and use_tid4_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and use_contiguous_partials
            and use_t2_projection_ilp
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T2_SQSUM_TREE=1 requires the promoted T=2 "
                "split8_tile3_last_arrival 16-warp shared4/fast-rcp/"
                "no-overlap/custom-u32/contiguous-partials/projection-ILP "
                "path"
            )
        if use_t32_projection_ilp and not (
            t32_split8_tile24_8warp_geometry
            and use_tid25_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_PROJECTION_ILP=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/non-profiling path"
            )
        if use_t32_projection_ilp2 and not (
            t32_split8_tile24_8warp_geometry
            and use_tid25_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and use_t32_custom_store1
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_PROJECTION_ILP2=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/custom-store1/non-profiling path"
            )
        if use_t32_warp0_handoff_barrier and not (
            t32_split8_tile24_8warp_geometry
            and use_tid25_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and use_t32_custom_store1
            and use_t32_pack_sqsum
            and not use_t32_skip_epilogue_barrier
            and not use_t32_premix_direct_bcast
            and not use_t32_projection_ilp
            and not use_t32_projection_ilp2
            and not use_t32_fn_cache_cg
            and not use_t32_token_pair
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_WARP0_HANDOFF_BARRIER=1 requires the "
                "T=32 split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/custom-store1/packed-sqsum/"
                "retained-epilogue-barrier/non-profiling serial path"
            )
        if (
            use_t32_pack_sqsum
            and not use_t32_premix_direct_bcast
            and not (
                t32_split8_tile24_8warp_geometry
                and use_tid25_reduction
                and use_fast_rcp
                and not use_warp_overlap
                and use_custom_handoff
                and use_t32_custom_store1
                and not use_t32_projection_ilp
                and not use_t32_projection_ilp2
                and not use_t32_fn_cache_cg
                and not use_t32_token_pair
                and not phase_profiling
            )
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_PACK_SQSUM=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/custom-store1/non-profiling "
                "serial path"
            )
        if use_t32_skip_epilogue_barrier and not (
            t32_split8_tile24_8warp_geometry
            and use_tid25_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and use_t32_custom_store1
            and use_t32_pack_sqsum
            and not use_t32_projection_ilp
            and not use_t32_projection_ilp2
            and not use_t32_fn_cache_cg
            and not use_t32_token_pair
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_SKIP_EPILOGUE_BARRIER=1 requires the "
                "T=32 split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/custom-store1/packed-sqsum/"
                "non-profiling serial path"
            )
        if use_t32_premix_direct_bcast and not (
            t32_split8_tile24_8warp_geometry
            and use_tid25_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and use_t32_custom_store1
            and use_t32_pack_sqsum
            and not use_t32_skip_epilogue_barrier
            and not use_t32_projection_ilp
            and not use_t32_projection_ilp2
            and not use_t32_fn_cache_cg
            and not use_t32_token_pair
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST=1 requires the "
                "T=32 split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/custom-store1/packed-sqsum/"
                "retained-epilogue-barrier/non-profiling serial path"
            )
        if use_t32_premix_shared_bridge and not (
            use_t32_premix_direct_bcast
            and t32_split8_tile24_8warp_geometry
            and use_tid25_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and use_t32_custom_store1
            and use_t32_pack_sqsum
            and not use_t32_skip_epilogue_barrier
            and not use_t32_projection_ilp
            and not use_t32_projection_ilp2
            and not use_t32_fn_cache_cg
            and not use_t32_token_pair
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_PREMIX_SHARED_BRIDGE=1 requires the "
                "T=32 direct-broadcast split8_tile24_last_arrival 8-warp "
                "shared25/fast-rcp/no-overlap/custom-u32/custom-store1/"
                "packed-sqsum/retained-epilogue-barrier/non-profiling path"
            )
        if use_t32_custom_store1 and not (
            t32_split8_tile24_8warp_geometry
            and use_tid25_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_CUSTOM_STORE1=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/non-profiling serial path"
            )
        if use_t32_fn_cache_cg and not (
            t32_split8_tile24_8warp_geometry
            and use_tid25_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and use_t32_custom_store1
            and not use_t32_projection_ilp
            and not use_t32_projection_ilp2
            and not use_t32_pack_sqsum
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_FN_CACHE_CG=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/custom-store1/non-profiling "
                "serial path"
            )
        if use_t32_token_pair and not (
            t32_split8_tile24_8warp_geometry
            and use_tid25_reduction
            and use_fast_rcp
            and not use_warp_overlap
            and use_custom_handoff
            and not phase_profiling
        ):
            raise ValueError(
                "DG_TLE_MHC_T32_TOKEN_PAIR=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/non-profiling path"
            )
        if use_t32_token_pair:
            # One CTA produces the same split for two adjacent tokens.
            grid_ctas = (tokens // 2) * num_splits
        projection_partials = torch.empty(
            (
                (tokens, num_splits, FULL_PROJECTION_OUTPUTS)
                if use_contiguous_partials
                else (num_output_tiles, num_splits, output_tile_block)
                if tokens == 1
                else (
                    tokens,
                    num_output_tiles,
                    num_splits,
                    output_tile_block,
                )
            ),
            dtype=torch.float32,
            device=inputs["x"].device,
        )
        sqsum_partials = torch.empty(
            (num_splits,) if tokens == 1 else (tokens, num_splits),
            dtype=torch.float32,
            device=inputs["x"].device,
        )
        counter_stride_elements = (
            COUNTER_U32_STRIDE if use_custom_handoff and tokens > 1 else 1
        )
        arrival_counter = torch.zeros(
            (
                (tokens, counter_stride_elements)
                if counter_stride_elements > 1
                else (tokens,)
            ),
            dtype=torch.uint32 if use_custom_handoff else torch.uint64,
            device=inputs["x"].device,
        )
        phase_profile = (
            torch.zeros(
                (grid_ctas * 2 + tokens * 4,),
                dtype=torch.int64,
                device=inputs["x"].device,
            )
            if phase_profiling
            else None
        )
        workspace_tensors = {
            "projection_partials": projection_partials,
            "sqsum_partials": sqsum_partials,
        }
        if phase_profile is not None:
            workspace_tensors["phase_profile"] = phase_profile
        counter_tensors = {"arrival_counter": arrival_counter}
        counter_increment = ctas_per_token
        performance_schedule = True
        cta_count = grid_ctas
        split_count = num_splits
    else:
        production_default_selected = False
        focused_migration_default_selected = False
        t8_16warp_candidate_selected = False
        t2_tile3_candidate_selected = False
        t2_tile3_16warp_candidate_selected = False
        t32_tile24_candidate_selected = False
        t32_shared25_candidate_selected = False
        t32_shared25_8warp_candidate_selected = False
        requested_reduction = reduction_override
        requested_sinkhorn = (
            sinkhorn_override
            if sinkhorn_override is not None
            else "triton_masked_reductions"
        )
        requested_overlap = (
            overlap_override if overlap_override is not None else "none"
        )
        requested_handoff = (
            handoff_override
            if handoff_override is not None
            else "triton_atomic_u64"
        )
        if requested_overlap != "none":
            raise ValueError(
                "DG_TLE_MHC_OVERLAP applies only to a split schedule"
            )
        if requested_handoff != "triton_atomic_u64":
            raise ValueError(
                "DG_TLE_MHC_HANDOFF applies only to a split schedule"
            )
        if requested_sinkhorn != "triton_masked_reductions":
            raise ValueError(
                "DG_TLE_MHC_SINKHORN applies only to a split schedule"
            )
        use_warp_sinkhorn = False
        use_fast_rcp = False
        use_warp_overlap = False
        use_custom_handoff = False
        if use_contiguous_partials:
            raise ValueError(
                "DG_TLE_MHC_T2_CONTIGUOUS_PARTIALS=1 requires the T=2 "
                "split8_tile3_last_arrival 16-warp split schedule"
            )
        if use_t2_projection_ilp:
            raise ValueError(
                "DG_TLE_MHC_T2_PROJECTION_ILP=1 requires the T=2 "
                "split8_tile3_last_arrival 16-warp split schedule"
            )
        if use_t2_sqsum_tree:
            raise ValueError(
                "DG_TLE_MHC_T2_SQSUM_TREE=1 requires the promoted T=2 "
                "split8_tile3_last_arrival 16-warp split schedule"
            )
        if use_t32_projection_ilp:
            raise ValueError(
                "DG_TLE_MHC_T32_PROJECTION_ILP=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25 split schedule"
            )
        if use_t32_projection_ilp2:
            raise ValueError(
                "DG_TLE_MHC_T32_PROJECTION_ILP2=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/"
                "custom-store1 split schedule"
            )
        if use_t32_custom_store1:
            raise ValueError(
                "DG_TLE_MHC_T32_CUSTOM_STORE1=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25 split schedule"
            )
        if use_t32_pack_sqsum:
            raise ValueError(
                "DG_TLE_MHC_T32_PACK_SQSUM=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/"
                "custom-store1 split schedule"
            )
        if use_t32_skip_epilogue_barrier:
            raise ValueError(
                "DG_TLE_MHC_T32_SKIP_EPILOGUE_BARRIER=1 requires the "
                "T=32 split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/custom-store1/packed-sqsum/"
                "non-profiling serial path"
            )
        if use_t32_premix_direct_bcast:
            raise ValueError(
                "DG_TLE_MHC_T32_PREMIX_DIRECT_BCAST=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/custom-store1/packed-sqsum/"
                "retained-epilogue-barrier/non-profiling serial path"
            )
        if use_t32_premix_shared_bridge:
            raise ValueError(
                "DG_TLE_MHC_T32_PREMIX_SHARED_BRIDGE=1 requires the T=32 "
                "direct-broadcast split8_tile24_last_arrival 8-warp "
                "shared25/fast-rcp/no-overlap/custom-u32/custom-store1/"
                "packed-sqsum/retained-epilogue-barrier/non-profiling path"
            )
        if use_t32_warp0_handoff_barrier:
            raise ValueError(
                "DG_TLE_MHC_T32_WARP0_HANDOFF_BARRIER=1 requires the "
                "T=32 split8_tile24_last_arrival 8-warp shared25/fast-rcp/"
                "no-overlap/custom-u32/custom-store1/packed-sqsum/"
                "retained-epilogue-barrier/non-profiling serial path"
            )
        if use_t32_fn_cache_cg:
            raise ValueError(
                "DG_TLE_MHC_T32_FN_CACHE_CG=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25/"
                "custom-store1 split schedule"
            )
        if use_t32_token_pair:
            raise ValueError(
                "DG_TLE_MHC_T32_TOKEN_PAIR=1 requires the T=32 "
                "split8_tile24_last_arrival 8-warp shared25 split schedule"
            )
        if phase_profiling:
            raise ValueError(
                "DG_TLE_MHC_PHASE_PROFILE requires a split schedule"
            )
        if requested_reduction not in (None, "triton_cta_sum"):
            raise ValueError(
                "DG_TLE_MHC_REDUCTION applies only to a split schedule"
            )
        use_hierarchical_reduction = False
        use_tid4_reduction = False
        use_tid5_reduction = False
        use_tid7_reduction = False
        use_tid9_reduction = False
        use_tid13_reduction = False
        use_tid25_reduction = False
        split_num_warps = FULL_NUM_WARPS
        outputs_per_tile = FULL_PROJECTION_OUTPUTS
        output_tile_block = 32
        num_output_tiles = 1
        num_splits = 1
        hidden_per_split = FULL_HIDDEN
        ctas_per_token = 1
        counter_stride_elements = 1
        grid_ctas = 1
        projection_partials = None
        sqsum_partials = None
        arrival_counter = None
        phase_profile = None
        workspace_tensors = {}
        counter_tensors = {}
        counter_increment = 0
        performance_schedule = False
        cta_count = 1
        split_count = 1
    metadata: dict[str, object] = {
        "T": tokens,
        "H": FULL_HIDDEN,
        "R": FULL_HC_MULT,
        "P": FULL_PROJECTION_OUTPUTS,
        "mode": "normal",
        "dtype": "bfloat16",
        "kernel_count": 1,
        "CTA": cta_count,
        "split": split_count,
        "ctas_per_token": ctas_per_token,
        "hidden_per_split": hidden_per_split,
        "outputs_per_tile": outputs_per_tile,
        "output_tile_block": output_tile_block,
        "num_warps": split_num_warps,
        "producer_reduction": (
            "warp_shuffle_shared4_tid4"
            if use_tid4_reduction
            else "warp_shuffle_shared7_tid7"
            if use_tid7_reduction
            else "warp_shuffle_shared9_tid9"
            if use_tid9_reduction
            else "warp_shuffle_shared13_tid13"
            if use_tid13_reduction
            else "warp_shuffle_shared25_tid25"
            if use_tid25_reduction
            else "warp_shuffle_shared5_tid5"
            if use_tid5_reduction
            else "warp_shuffle_shared5"
            if use_hierarchical_reduction
            else "triton_cta_sum"
        ),
        "sinkhorn_implementation": requested_sinkhorn,
        "sinkhorn_reciprocal": (
            "rcp.approx.ftz.f32"
            if use_fast_rcp
            else "triton_full_division"
        ),
        "overlap_implementation": requested_overlap,
        "handoff_implementation": requested_handoff,
        "counter_dtype": (
            "uint32" if use_custom_handoff else "uint64"
        ),
        "counter_stride_elements": counter_stride_elements,
        "counter_stride_bytes": (
            counter_stride_elements * (4 if use_custom_handoff else 8)
        ),
        "atomic_inc_limit": (
            ctas_per_token - 1 if use_custom_handoff else None
        ),
        "handoff_barrier_participants": (
            split_num_warps * 32 if use_custom_handoff else None
        ),
        "t32_warp0_handoff_barrier": use_t32_warp0_handoff_barrier,
        "producer_publish_barrier_implementation": (
            "inline_ptx_barrier_cta_sync_14_32_release_inc"
            if use_t32_warp0_handoff_barrier
            else "tle_full_cta_named_barrier"
            if requested_schedule.startswith("split")
            else None
        ),
        "producer_publish_barrier_id": (
            14 if use_t32_warp0_handoff_barrier else None
        ),
        "producer_publish_barrier_participants": (
            32
            if use_t32_warp0_handoff_barrier
            else split_num_warps * 32
            if requested_schedule.startswith("split")
            else None
        ),
        "handoff_broadcast_barrier_implementation": (
            "inline_ptx_bar_red_or_pred_id15"
            if use_custom_handoff
            else None
        ),
        "handoff_broadcast_barrier_id": 15 if use_custom_handoff else None,
        "default_partition_warps": split_num_warps,
        "routing_worker_logical_warps": (
            FULL_OVERLAP_ROUTING_WARPS if use_warp_overlap else 0
        ),
        "physical_cta_warps": (
            FULL_OVERLAP_PHYSICAL_WARPS
            if use_warp_overlap
            else split_num_warps
        ),
        "physical_cta_threads": (
            FULL_OVERLAP_PHYSICAL_THREADS
            if use_warp_overlap
            else split_num_warps * 32
        ),
        "named_barrier_participants": (
            FULL_OVERLAP_LOGICAL_THREADS if use_warp_overlap else None
        ),
        "schedule": requested_schedule,
        "dispatch_policy": (
            "production_auto"
            if production_default_selected
            else "focused_migration_auto"
            if focused_migration_default_selected
            else "focused_migration_16warp_candidate"
            if (
                t8_16warp_candidate_selected
                or t2_tile3_16warp_candidate_selected
                or (
                    t32_shared25_candidate_selected
                    and split_num_warps == 16
                )
            )
            else "focused_migration_explicit_candidate"
            if (
                t2_tile3_candidate_selected
                or t32_tile24_candidate_selected
                or t32_shared25_8warp_candidate_selected
            )
            else "explicit_or_compatible_fallback"
        ),
        "production_default_selected": production_default_selected,
        "focused_migration_default_selected": (
            focused_migration_default_selected
        ),
        "t8_16warp_candidate_selected": t8_16warp_candidate_selected,
        "t2_tile3_candidate_selected": t2_tile3_candidate_selected,
        "t2_tile3_16warp_candidate_selected": (
            t2_tile3_16warp_candidate_selected
        ),
        "t32_tile24_candidate_selected": t32_tile24_candidate_selected,
        "t32_split8_tile24_default_selected": (
            t32_split8_tile24_default_selected
        ),
        "t32_shared25_candidate_selected": (
            t32_shared25_candidate_selected
        ),
        "t32_shared25_default_selected": t32_shared25_default_selected,
        "t32_fast_rcp_default_selected": t32_fast_rcp_default_selected,
        "t32_custom_handoff_default_selected": (
            t32_custom_handoff_default_selected
        ),
        "t32_shared25_8warp_candidate_selected": (
            t32_shared25_8warp_candidate_selected
        ),
        "t32_shared25_warp_mode": (
            split_num_warps if use_tid25_reduction else None
        ),
        "t2_contiguous_partials": use_contiguous_partials,
        "t2_contiguous_partials_default_selected": (
            t2_contiguous_partials_default_selected
        ),
        "t2_projection_ilp": use_t2_projection_ilp,
        "t2_projection_ilp_default_selected": (
            t2_projection_ilp_default_selected
        ),
        "t2_sqsum_tree": use_t2_sqsum_tree,
        "t32_projection_ilp": use_t32_projection_ilp,
        "t32_projection_ilp2": use_t32_projection_ilp2,
        "t32_custom_store1": use_t32_custom_store1,
        "t32_custom_store1_default_selected": (
            t32_custom_store1_default_selected
        ),
        "t32_pack_sqsum": use_t32_pack_sqsum,
        "t32_pack_sqsum_default_selected": (
            t32_pack_sqsum_default_selected
        ),
        "t32_skip_epilogue_barrier": use_t32_skip_epilogue_barrier,
        "t32_premix_direct_bcast": use_t32_premix_direct_bcast,
        "t32_premix_direct_bcast_default_selected": (
            t32_premix_direct_bcast_default_selected
        ),
        "t32_premix_shared_bridge": use_t32_premix_shared_bridge,
        "pre_mix_broadcast_schedule": (
            "inline_ptx_shfl_idx_lanes0_3_warp_local_shared_scalar_bridge"
            if use_t32_premix_shared_bridge
            else "inline_ptx_shfl_idx_lanes0_3"
            if use_t32_premix_direct_bcast
            else "warp_specialized_shared_pre_mix"
            if use_warp_overlap
            else "triton_sum_select_per_channel"
        ),
        "pre_mix_bridge_shared_shape": (
            [8, 4] if use_t32_premix_shared_bridge else None
        ),
        "pre_mix_bridge_shared_bytes": (
            8 * 4 * 4 if use_t32_premix_shared_bridge else 0
        ),
        "pre_mix_bridge_sync": (
            "inline_ptx_bar_warp_sync_full_mask"
            if use_t32_premix_shared_bridge
            else None
        ),
        "epilogue_barrier_policy": (
            "elided_no_in_kernel_consumer"
            if use_t32_skip_epilogue_barrier
            else "retained_existing_path"
        ),
        "t32_fn_cache_cg": use_t32_fn_cache_cg,
        "t32_token_pair": use_t32_token_pair,
        "tokens_per_producer_cta": 2 if use_t32_token_pair else 1,
        "token_pair_count": tokens // 2 if use_t32_token_pair else None,
        "t32_fn_load_cache_modifier": (
            ".cg" if use_t32_fn_cache_cg else "default"
        ),
        "t32_shared25_store_implementation": (
            "inline_asm_predicated_lane0_store2_all25"
            if use_t32_token_pair
            else "inline_asm_predicated_lane0_store1_all25"
            if use_t32_custom_store1 or use_t32_projection_ilp2
            else "inline_asm_store3_projection_masked_store_sqsum"
            if use_t32_projection_ilp
            else "triton_masked_store_all25"
            if use_tid25_reduction
            else None
        ),
        "sqsum_reduction_schedule": (
            "packed_projection_lane24_serial_split_scalar_load"
            if use_t32_pack_sqsum
            else "lane8_warp_tree_broadcast"
            if use_t2_sqsum_tree
            else "serial_split_scalar_loads"
        ),
        "projection_accumulation_schedule": (
            "token_pair_weight_reuse_serial_output"
            if use_t32_token_pair
            else "batch2_interleaved_warp_reduce_pairfold_store1"
            if use_t32_projection_ilp2
            else "batch3_pairfold_before_reduce"
            if use_t32_projection_ilp
            else "batch3_before_reduce"
            if use_t2_projection_ilp
            else "serial_output_reduce"
        ),
        "projection_partials_layout": (
            "token_split_output_contiguous"
            if use_contiguous_partials
            else "token_tile_split_padded_lane24_sqsum"
            if use_t32_pack_sqsum
            else "token_tile_split_padded"
            if projection_partials is not None and tokens > 1
            else "tile_split_padded"
            if projection_partials is not None
            else None
        ),
        "projection_partials_shape": (
            list(projection_partials.shape)
            if projection_partials is not None
            else None
        ),
        "projection_partials_bytes": (
            projection_partials.numel() * projection_partials.element_size()
            if projection_partials is not None
            else 0
        ),
        "t32_partial_store_implementation": (
            "inline_asm_predicated_tid25_packed_global_store"
            if use_t32_pack_sqsum
            else "separate_projection_and_sqsum_global_stores"
            if use_tid25_reduction
            else None
        ),
        "sqsum_partials_kernel_usage": (
            "unused_abi_allocation"
            if use_t32_pack_sqsum
            else "producer_and_tail_workspace"
            if sqsum_partials is not None
            else None
        ),
        "sqsum_partials_shape": (
            list(sqsum_partials.shape)
            if sqsum_partials is not None
            else None
        ),
        "sqsum_partials_allocated_bytes": (
            sqsum_partials.numel() * sqsum_partials.element_size()
            if sqsum_partials is not None
            else 0
        ),
        "sqsum_partials_effective_kernel_bytes": (
            0
            if use_t32_pack_sqsum
            else sqsum_partials.numel() * sqsum_partials.element_size()
            if sqsum_partials is not None
            else 0
        ),
        "report_eligible": False,
        "result_scope": "focused_tle_migration",
        "programmatic_dependent_launch": False,
        "performance_schedule": performance_schedule,
        "phase_profiling": phase_profiling,
        "phase_profile_layout": (
            {
                "cta_entry_and_publish": [grid_ctas, 2],
                "tail_markers": [
                    "tail_start",
                    "projection_head_end",
                    "sinkhorn_end",
                    "collapse_rms_end",
                ],
                "tail_markers_per_token": 4,
                "token_count": tokens,
                "units": "globaltimer_nanoseconds",
            }
            if phase_profiling
            else None
        ),
        "workspace_tensor_count": len(workspace_tensors),
        "counter_tensor_count": len(counter_tensors),
        "counter_policy": "self_reset",
        "counter_increment_per_launch": counter_increment,
        "bf16_rounding_boundaries": ["new_residual", "norm_input", "y_bf16"],
    }
    return TLESM90MegaMHCPlan(
        x=inputs["x"],
        residual=inputs["residual"],
        post_mix=inputs["post_mix"],
        comb_res_mix=inputs["comb_res_mix"],
        fn=inputs["fn"],
        mix_scales=inputs["mix_scales"],
        mix_bases=inputs["mix_bases"],
        rmsnorm_weight=inputs["rmsnorm_weight"],
        new_residual=outputs["new_residual"],
        new_post_mix=outputs["new_post_mix"],
        new_comb_res_mix=outputs["new_comb_res_mix"],
        y_bf16=outputs["y_bf16"],
        hc_norm_eps=float(inputs["hc_norm_eps"]),
        hc_pre_eps=float(inputs["hc_pre_eps"]),
        hc_post_scale=float(inputs["hc_post_scale"]),
        sinkhorn_eps=float(inputs["sinkhorn_eps"]),
        rmsnorm_eps=float(inputs["rmsnorm_eps"]),
        rmsnorm_scale=float(inputs["rmsnorm_scale"]),
        schedule=requested_schedule,
        tokens=tokens,
        num_splits=num_splits,
        hidden_per_split=hidden_per_split,
        ctas_per_token=ctas_per_token,
        counter_stride_elements=counter_stride_elements,
        projection_partials=projection_partials,
        sqsum_partials=sqsum_partials,
        arrival_counter=arrival_counter,
        phase_profile=phase_profile,
        outputs_per_tile=outputs_per_tile,
        output_tile_block=output_tile_block,
        num_output_tiles=num_output_tiles,
        grid_ctas=grid_ctas,
        split_num_warps=split_num_warps,
        use_hierarchical_reduction=use_hierarchical_reduction,
        use_tid4_reduction=use_tid4_reduction,
        use_tid5_reduction=use_tid5_reduction,
        use_tid7_reduction=use_tid7_reduction,
        use_tid9_reduction=use_tid9_reduction,
        use_tid13_reduction=use_tid13_reduction,
        use_tid25_reduction=use_tid25_reduction,
        use_warp_sinkhorn=use_warp_sinkhorn,
        use_fast_rcp=use_fast_rcp,
        use_warp_overlap=use_warp_overlap,
        use_custom_handoff=use_custom_handoff,
        use_contiguous_partials=use_contiguous_partials,
        use_t2_projection_ilp=use_t2_projection_ilp,
        use_t2_sqsum_tree=use_t2_sqsum_tree,
        use_t32_projection_ilp=use_t32_projection_ilp,
        use_t32_projection_ilp2=use_t32_projection_ilp2,
        use_t32_custom_store1=use_t32_custom_store1,
        use_t32_pack_sqsum=use_t32_pack_sqsum,
        use_t32_skip_epilogue_barrier=(
            use_t32_skip_epilogue_barrier
        ),
        use_t32_premix_direct_bcast=use_t32_premix_direct_bcast,
        use_t32_premix_shared_bridge=use_t32_premix_shared_bridge,
        use_t32_warp0_handoff_barrier=use_t32_warp0_handoff_barrier,
        use_t32_fn_cache_cg=use_t32_fn_cache_cg,
        use_t32_token_pair=use_t32_token_pair,
        phase_profiling=phase_profiling,
        workspace_tensors=workspace_tensors,
        counter_tensors=counter_tensors,
        metadata=metadata,
    )


def _validate_geometry(tokens: int, num_splits: int) -> None:
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise TypeError("tokens must be an integer")
    if isinstance(num_splits, bool) or not isinstance(num_splits, int):
        raise TypeError("num_splits must be an integer")
    if tokens < 1:
        raise ValueError("tokens must be positive")
    if not 1 <= num_splits <= MAX_NUM_SPLITS:
        raise ValueError(
            f"num_splits must be in [1, {MAX_NUM_SPLITS}]"
        )


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.requires_grad:
        raise ValueError(f"{name} must not require gradients")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    begin = tensor.data_ptr()
    return begin, begin + tensor.numel() * tensor.element_size()


def _validate_no_overlap(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> None:
    intervals = [
        (*_byte_interval(tensor), name)
        for name, tensor in named_tensors
    ]
    intervals.sort()
    for (_, previous_end, previous_name), (
        current_begin,
        _,
        current_name,
    ) in zip(intervals, intervals[1:]):
        if current_begin < previous_end:
            raise ValueError(
                f"{previous_name} and {current_name} must not overlap"
            )


def _validate_host_contract(
    mix_partials: torch.Tensor,
    sqsum_partials: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor],
    workspace: MegaMHCAtomicProtocolWorkspace,
) -> tuple[int, int]:
    if not isinstance(mix_partials, torch.Tensor):
        raise TypeError("mix_partials must be a torch.Tensor")
    if mix_partials.ndim != 3:
        raise ValueError("mix_partials must be 3D [T, S, 24]")
    tokens, num_splits, outputs = tuple(mix_partials.shape)
    _validate_geometry(tokens, num_splits)
    if outputs != NUM_MHC_OUTPUTS:
        raise ValueError("mix_partials must have shape [T, S, 24]")
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError("out must be (reduced_mix, reduced_sqsum)")
    if not isinstance(workspace, MegaMHCAtomicProtocolWorkspace):
        raise TypeError(
            "workspace must be MegaMHCAtomicProtocolWorkspace"
        )

    reduced_mix, reduced_sqsum = out
    partial_shape, counter_shape = workspace_shapes(tokens, num_splits)
    specs = (
        (mix_partials, "mix_partials", (tokens, num_splits, 24), torch.float32),
        (
            sqsum_partials,
            "sqsum_partials",
            (tokens, num_splits),
            torch.float32,
        ),
        (reduced_mix, "reduced_mix", (tokens, 24), torch.float32),
        (reduced_sqsum, "reduced_sqsum", (tokens,), torch.float32),
        (
            workspace.partials_workspace,
            "workspace.partials_workspace",
            partial_shape,
            torch.float32,
        ),
        (
            workspace.counters,
            "workspace.counters",
            counter_shape,
            torch.uint64,
        ),
    )
    for tensor, name, shape, dtype in specs:
        _validate_tensor(tensor, name=name, shape=shape, dtype=dtype)

    devices = {tensor.device for tensor, _, _, _ in specs}
    if len(devices) != 1:
        raise ValueError("all inputs, outputs, and workspace must share a device")
    _validate_no_overlap((name, tensor) for tensor, name, _, _ in specs)
    return tokens, num_splits


__all__ = (
    "FULL_HC_MULT",
    "FULL_HIDDEN",
    "FULL_PROJECTION_OUTPUTS",
    "FULL_SINKHORN_ITERS",
    "FULL_TOKENS",
    "SUPPORTED_FULL_TOKENS",
    "T8_CTAS_PER_TOKEN",
    "T8_GRID_CTAS",
    "T8_NUM_SPLITS",
    "T8_OUTPUTS_PER_TILE",
    "T8_OUTPUT_TILE_BLOCK",
    "COUNTER_CACHE_LINE_BYTES",
    "COUNTER_U32_STRIDE",
    "MAX_NUM_SPLITS",
    "MegaMHCAtomicProtocolWorkspace",
    "NUM_MHC_OUTPUTS",
    "NUM_SQRSUM_PARTIALS",
    "WORKSPACE_ROW_ELEMENTS",
    "TLESM90MegaMHCPlan",
    "allocate_tle_sm90_mega_mhc_atomic_workspace",
    "prepare_tle_sm90_mega_mhc",
    "tle_sm90_mega_mhc_atomic_protocol_probe",
    "workspace_nbytes",
    "workspace_shapes",
)

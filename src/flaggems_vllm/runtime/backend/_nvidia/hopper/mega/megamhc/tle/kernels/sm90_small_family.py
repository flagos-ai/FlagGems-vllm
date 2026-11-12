"""Production SM90 MegaMHC kernels for the small CUDA-Graph buckets.

The public factory admits only the four production-dispatched DeepSeek-V4-
Flash Normal BF16 buckets, ``T in {1, 2, 4, 8}``, at ``H=4096, R=4, P=24``.
Each prepared plan owns its workspaces and enqueues exactly one specialized
Triton/TLE kernel. Historical bring-up code and non-production schedules are
not part of this production module.

The fixed production schedules are: T1 split16/tile4 with eight producer
warps and warp overlap; T2 split8/tile3 with sixteen producer warps,
contiguous partials, and projection ILP; and T4/T8 split8/tile6 with eight
producer warps.  The table near ``_SMALL_PRODUCTION_CONFIGS`` is the complete
admitted configuration set rather than an autotuning search space.
"""

from __future__ import annotations

from dataclasses import dataclass
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
SUPPORTED_FULL_TOKENS = (1, 2, 4, 8)
FULL_HIDDEN = 4096
FULL_HC_MULT = 4
FULL_PROJECTION_OUTPUTS = 24
FULL_FLAT_HIDDEN = FULL_HIDDEN * FULL_HC_MULT
FULL_NUM_WARPS = 8
FULL_CTA_THREADS = FULL_NUM_WARPS * 32
FULL_OVERLAP_ROUTING_WARPS = 1
FULL_OVERLAP_LOGICAL_THREADS = FULL_CTA_THREADS + FULL_OVERLAP_ROUTING_WARPS * 32
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
T2_TILE3_NUM_OUTPUT_TILES = FULL_PROJECTION_OUTPUTS // T2_TILE3_OUTPUTS_PER_TILE
T2_TILE3_REDUCTION_CHANNELS = T2_TILE3_OUTPUTS_PER_TILE + 1
T2_TILE3_REDUCTION_CHANNEL_BLOCK = 4
T8_TILE6_OUTPUTS_PER_TILE = 6
T8_TILE6_OUTPUT_BLOCK = 8
T8_TILE6_NUM_OUTPUT_TILES = FULL_PROJECTION_OUTPUTS // T8_TILE6_OUTPUTS_PER_TILE
T8_TILE6_REDUCTION_CHANNELS = T8_TILE6_OUTPUTS_PER_TILE + 1
T8_TILE6_REDUCTION_CHANNEL_BLOCK = 8
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
JIT_FULL_OVERLAP_ROUTING_WARPS = tl.constexpr(FULL_OVERLAP_ROUTING_WARPS)
JIT_FULL_OVERLAP_LOGICAL_THREADS = tl.constexpr(FULL_OVERLAP_LOGICAL_THREADS)
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
JIT_T2_TILE3_REDUCTION_CHANNELS = tl.constexpr(T2_TILE3_REDUCTION_CHANNELS)
JIT_T2_TILE3_REDUCTION_CHANNEL_BLOCK = tl.constexpr(T2_TILE3_REDUCTION_CHANNEL_BLOCK)
JIT_T2_TILE3_NUM_OUTPUT_TILES = tl.constexpr(T2_TILE3_NUM_OUTPUT_TILES)
JIT_T8_TILE6_REDUCTION_CHANNELS = tl.constexpr(T8_TILE6_REDUCTION_CHANNELS)
JIT_T8_TILE6_REDUCTION_CHANNEL_BLOCK = tl.constexpr(T8_TILE6_REDUCTION_CHANNEL_BLOCK)
JIT_T8_TILE6_NUM_OUTPUT_TILES = tl.constexpr(T8_TILE6_NUM_OUTPUT_TILES)


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
        value = tl.load(new_residual_ptr + channel * JIT_FULL_HIDDEN + h).to(tl.float32)
        pre_scalar = tl.load(tle.gpu.local_ptr(pre_shared, (channel,)))
        norm_input += value * pre_scalar
    norm_input_bf16 = norm_input.to(tl.bfloat16, fp_downcast_rounding="rtne")
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=0) / JIT_FULL_HIDDEN
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

    routing_mask = (lane >= 2 * JIT_FULL_HC_MULT) & (lane < JIT_FULL_PROJECTION_OUTPUTS)
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
    outputs_per_tile: int
    output_tile_block: int
    num_output_tiles: int
    grid_ctas: int
    split_num_warps: int
    use_hierarchical_reduction: bool
    use_tid4_reduction: bool
    use_tid5_reduction: bool
    use_tid7_reduction: bool
    use_warp_overlap: bool
    use_contiguous_partials: bool
    use_t2_projection_ilp: bool
    workspace_tensors: dict[str, torch.Tensor]
    counter_tensors: dict[str, torch.Tensor]
    metadata: dict[str, object]

    def run(self) -> None:
        """Enqueue the one production small-bucket kernel launch."""
        if not (self.schedule.startswith("split") and "_tile" in self.schedule):
            raise RuntimeError("non-production small-family plan")
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
            USE_WARP_OVERLAP=self.use_warp_overlap,
            USE_CONTIGUOUS_PARTIALS=self.use_contiguous_partials,
            USE_T2_PROJECTION_ILP=self.use_t2_projection_ilp,
            num_warps=self.split_num_warps,
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
    USE_WARP_OVERLAP: tl.constexpr,
    USE_CONTIGUOUS_PARTIALS: tl.constexpr,
    USE_T2_PROJECTION_ILP: tl.constexpr,
):
    """Token-local split producers plus in-kernel last-arrival epilogues."""
    pid = tl.program_id(0)
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
    r1 = tl.load(residual_ptr + token_residual_base + JIT_FULL_HIDDEN + hidden).to(
        tl.float32
    )
    r2 = tl.load(residual_ptr + token_residual_base + 2 * JIT_FULL_HIDDEN + hidden).to(
        tl.float32
    )
    r3 = tl.load(residual_ptr + token_residual_base + 3 * JIT_FULL_HIDDEN + hidden).to(
        tl.float32
    )
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
            new_residual_ptr + token_residual_base + 2 * JIT_FULL_HIDDEN + hidden,
            q2,
        )
        tl.store(
            new_residual_ptr + token_residual_base + 3 * JIT_FULL_HIDDEN + hidden,
            q3,
        )

    if USE_TID4_REDUCTION:
        tl.static_assert(
            CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS
            or CTA_THREADS_PER_BLOCK == 2 * JIT_T8_REDUCTION_THREADS
        )
        tl.static_assert(HIDDEN_PER_SPLIT == 2 * JIT_T8_REDUCTION_THREADS)
        tl.static_assert(OUTPUTS_PER_TILE == JIT_T2_TILE3_REDUCTION_CHANNELS - 1)
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
            lane_partial0 += q1f * tl.load(weight_row0 + JIT_FULL_HIDDEN + hidden).to(
                tl.float32
            )
            lane_partial1 += q1f * tl.load(weight_row1 + JIT_FULL_HIDDEN + hidden).to(
                tl.float32
            )
            lane_partial2 += q1f * tl.load(weight_row2 + JIT_FULL_HIDDEN + hidden).to(
                tl.float32
            )
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
                w2 = tl.load(weight_row + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
                w3 = tl.load(weight_row + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
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

        shared4_ready = tle.gpu.alloc_barrier(arrive_count=CTA_THREADS_PER_BLOCK)
        tle.gpu.barrier_wait(shared4_ready)
        safe_channel = tl.minimum(reduction_thread, JIT_T2_TILE3_REDUCTION_CHANNELS - 1)
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
                token * NUM_SPLITS + split
            ) * JIT_FULL_PROJECTION_OUTPUTS + output_tile * OUTPUTS_PER_TILE
        else:
            projection_offset = (
                token * NUM_OUTPUT_TILES * NUM_SPLITS + output_tile * NUM_SPLITS + split
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
    elif USE_TID7_REDUCTION:
        tl.static_assert(CTA_THREADS_PER_BLOCK == JIT_T8_REDUCTION_THREADS)
        tl.static_assert(HIDDEN_PER_SPLIT == 2 * JIT_T8_REDUCTION_THREADS)
        tl.static_assert(OUTPUTS_PER_TILE == JIT_T8_TILE6_REDUCTION_CHANNELS - 1)
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

        shared7_ready = tle.gpu.alloc_barrier(arrive_count=CTA_THREADS_PER_BLOCK)
        tle.gpu.barrier_wait(shared7_ready)
        safe_channel = tl.minimum(reduction_thread, JIT_T8_TILE6_REDUCTION_CHANNELS - 1)
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
            token * NUM_OUTPUT_TILES * NUM_SPLITS + output_tile * NUM_SPLITS + split
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
            + q1f * tl.load(weight_row0 + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q2f * tl.load(weight_row0 + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q3f * tl.load(weight_row0 + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
        )
        v1 = (
            q0f * tl.load(weight_row1 + hidden).to(tl.float32)
            + q1f * tl.load(weight_row1 + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q2f * tl.load(weight_row1 + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q3f * tl.load(weight_row1 + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
        )
        v2 = (
            q0f * tl.load(weight_row2 + hidden).to(tl.float32)
            + q1f * tl.load(weight_row2 + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q2f * tl.load(weight_row2 + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q3f * tl.load(weight_row2 + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
        )
        v3 = (
            q0f * tl.load(weight_row3 + hidden).to(tl.float32)
            + q1f * tl.load(weight_row3 + JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q2f * tl.load(weight_row3 + 2 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
            + q3f * tl.load(weight_row3 + 3 * JIT_FULL_HIDDEN + hidden).to(tl.float32)
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
            _ = _store_shared5_f32(
                _shared_address_u32(tle.gpu.local_ptr(scratch, (channel0 + 0, warp))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (channel0 + 1, warp))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (channel0 + 2, warp))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (channel0 + 3, warp))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (channel0 + 4, warp))),
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
        hierarchical_ready = tle.gpu.alloc_barrier(arrive_count=CTA_THREADS_PER_BLOCK)
        tle.gpu.barrier_wait(hierarchical_ready)

        projection_offset = (
            (token * NUM_OUTPUT_TILES * NUM_SPLITS + output_tile * NUM_SPLITS + split)
        ) * 4
        if USE_TID5_REDUCTION:
            thread = local_hidden
            reduced = _reduce_shared8_f32(
                _shared_address_u32(tle.gpu.local_ptr(scratch, (thread, channel0 + 0))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (thread, channel0 + 1))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (thread, channel0 + 2))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (thread, channel0 + 3))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (thread, channel0 + 4))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (thread, channel0 + 5))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (thread, channel0 + 6))),
                _shared_address_u32(tle.gpu.local_ptr(scratch, (thread, channel0 + 7))),
                thread,
            )
            _ = _store_tid5_partials_f32(
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
                projection_partials_ptr + projection_offset + local_hidden + 0,
                o0,
                mask=write_mask,
            )
            tl.store(
                projection_partials_ptr + projection_offset + local_hidden + 1,
                o1,
                mask=write_mask,
            )
            tl.store(
                projection_partials_ptr + projection_offset + local_hidden + 2,
                o2,
                mask=write_mask,
            )
            tl.store(
                projection_partials_ptr + projection_offset + local_hidden + 3,
                o3,
                mask=write_mask,
            )
            if output_tile == 0:
                tl.store(
                    sqsum_partials_ptr + token * NUM_SPLITS + split + local_hidden,
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
            projection_partial += tl.where(partial_lane == output_in_tile, dot, 0.0)
        projection_offset = (
            token * NUM_OUTPUT_TILES * NUM_SPLITS + output_tile * NUM_SPLITS + split
        ) * OUTPUT_TILE_BLOCK + partial_lane
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

    producer_ready = tle.gpu.alloc_barrier(arrive_count=CTA_THREADS_PER_BLOCK)
    tle.gpu.barrier_wait(producer_ready)
    thread = _physical_thread_id_x()
    token_counter_ptr = arrival_counter_ptr + token * COUNTER_STRIDE_ELEMENTS
    leader_won = _release_inc_u32_last_arrival(
        tl.cast(token_counter_ptr, tl.uint64),
        thread,
        CTAS_PER_TOKEN - 1,
    )
    if CTA_THREADS_PER_BLOCK == 256:
        is_last = _bar_red_or_pred_u32_256(leader_won) != 0
    else:
        is_last = _bar_red_or_pred_u32_512(leader_won) != 0
    if is_last:

        projection_lane = tl.arange(0, 32)
        projection_mask = projection_lane < JIT_FULL_PROJECTION_OUTPUTS
        tile_for_output = projection_lane // OUTPUTS_PER_TILE
        lane_in_output_tile = projection_lane % OUTPUTS_PER_TILE
        projected = tl.zeros((32,), tl.float32)
        squared_sum = 0.0
        if USE_CONTIGUOUS_PARTIALS:
            for source_split in tl.static_range(0, NUM_SPLITS):
                partial_offset = (
                    token * NUM_SPLITS + source_split
                ) * JIT_FULL_PROJECTION_OUTPUTS + projection_lane
                projected += tl.load(
                    projection_partials_ptr + partial_offset,
                    mask=projection_mask,
                    other=0.0,
                )
                squared_sum += tl.load(
                    sqsum_partials_ptr + token * NUM_SPLITS + source_split
                )
        else:
            for source_split in tl.range(0, NUM_SPLITS):
                partial_offset = (
                    token * NUM_OUTPUT_TILES * NUM_SPLITS
                    + tile_for_output * NUM_SPLITS
                    + source_split
                ) * OUTPUT_TILE_BLOCK + lane_in_output_tile
                projected += tl.load(
                    projection_partials_ptr + partial_offset,
                    mask=projection_mask,
                    other=0.0,
                )
                squared_sum += tl.load(
                    sqsum_partials_ptr + token * NUM_SPLITS + source_split
                )

        projection_norm = tl.rsqrt(squared_sum / JIT_FULL_FLAT_HIDDEN + hc_norm_eps)
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
            # independently verified 8+[1] microbenchmark does.
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
                        ),
                    ),
                ],
                [JIT_FULL_OVERLAP_ROUTING_WARPS],
                [64],
            )
            return
        pre_mix = tl.sigmoid(mixes) + hc_pre_eps
        post_mix = tl.sigmoid(mixes) * hc_post_scale
        post_mask = (projection_lane >= JIT_FULL_HC_MULT) & (
            projection_lane < 2 * JIT_FULL_HC_MULT
        )
        tl.store(
            new_post_mix_ptr + token_state_base + projection_lane - JIT_FULL_HC_MULT,
            post_mix,
            mask=post_mask,
        )

        routing_mask = (projection_lane >= 2 * JIT_FULL_HC_MULT) & (
            projection_lane < JIT_FULL_PROJECTION_OUTPUTS
        )
        routing = mixes
        sinkhorn = tl.where(routing_mask, routing, 0.0)
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
                new_residual_ptr + token_residual_base + channel * JIT_FULL_HIDDEN + h
            ).to(tl.float32)
            pre_scalar = tl.sum(
                tl.where(projection_lane == channel, pre_mix, 0.0),
                axis=0,
            )
            norm_input += value * pre_scalar
        norm_input_bf16 = norm_input.to(tl.bfloat16, fp_downcast_rounding="rtne")
        norm_input_fp32 = norm_input_bf16.to(tl.float32)
        rms = tl.rsqrt(
            tl.sum(norm_input_fp32 * norm_input_fp32, axis=0) / JIT_FULL_HIDDEN
            + rmsnorm_eps
        )
        rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
        y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
        tl.store(
            y_bf16_ptr + token_hidden_base + h,
            y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        )

        epilogue_ready = tle.gpu.alloc_barrier(arrive_count=CTA_THREADS_PER_BLOCK)
        tle.gpu.barrier_wait(epilogue_ready)


_SMALL_PRODUCTION_CONFIGS: dict[int, dict[str, object]] = {
    1: {
        "schedule": "split16_tile4_last_arrival",
        "num_splits": 16,
        "outputs_per_tile": 4,
        "output_tile_block": 4,
        "num_output_tiles": 6,
        "split_num_warps": 8,
        "use_hierarchical_reduction": True,
        "use_tid4_reduction": False,
        "use_tid5_reduction": True,
        "use_tid7_reduction": False,
        "use_warp_overlap": True,
        "use_contiguous_partials": False,
        "use_t2_projection_ilp": False,
    },
    2: {
        "schedule": "split8_tile3_last_arrival",
        "num_splits": 8,
        "outputs_per_tile": 3,
        "output_tile_block": 4,
        "num_output_tiles": 8,
        "split_num_warps": 16,
        "use_hierarchical_reduction": False,
        "use_tid4_reduction": True,
        "use_tid5_reduction": False,
        "use_tid7_reduction": False,
        "use_warp_overlap": False,
        "use_contiguous_partials": True,
        "use_t2_projection_ilp": True,
    },
    4: {
        "schedule": "split8_tile6_last_arrival",
        "num_splits": 8,
        "outputs_per_tile": 6,
        "output_tile_block": 8,
        "num_output_tiles": 4,
        "split_num_warps": 8,
        "use_hierarchical_reduction": False,
        "use_tid4_reduction": False,
        "use_tid5_reduction": False,
        "use_tid7_reduction": True,
        "use_warp_overlap": False,
        "use_contiguous_partials": False,
        "use_t2_projection_ilp": False,
    },
    8: {
        "schedule": "split8_tile6_last_arrival",
        "num_splits": 8,
        "outputs_per_tile": 6,
        "output_tile_block": 8,
        "num_output_tiles": 4,
        "split_num_warps": 8,
        "use_hierarchical_reduction": False,
        "use_tid4_reduction": False,
        "use_tid5_reduction": False,
        "use_tid7_reduction": True,
        "use_warp_overlap": False,
        "use_contiguous_partials": False,
        "use_t2_projection_ilp": False,
    },
}


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> TLESM90MegaMHCPlan:
    """Prepare one of the four production-dispatched small buckets."""
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
            f"T={tokens} is unsupported; production small buckets support "
            f"T in {SUPPORTED_FULL_TOKENS}"
        )
    if inputs["shifted_prev_mix"] is not None:
        raise ValueError("the production small family supports Normal mode only")
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
    if len({tensor.device for tensor, _, _, _ in tensor_specs}) != 1:
        raise ValueError("all MegaMHC inputs and outputs must share a device")
    _validate_no_overlap((name, tensor) for tensor, name, _, _ in tensor_specs)

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

    config = _SMALL_PRODUCTION_CONFIGS[tokens]
    schedule = str(config["schedule"])
    num_splits = int(config["num_splits"])
    outputs_per_tile = int(config["outputs_per_tile"])
    output_tile_block = int(config["output_tile_block"])
    num_output_tiles = int(config["num_output_tiles"])
    split_num_warps = int(config["split_num_warps"])
    hidden_per_split = FULL_HIDDEN // num_splits
    ctas_per_token = num_splits * num_output_tiles
    grid_ctas = tokens * ctas_per_token
    use_hierarchical_reduction = bool(config["use_hierarchical_reduction"])
    use_tid4_reduction = bool(config["use_tid4_reduction"])
    use_tid5_reduction = bool(config["use_tid5_reduction"])
    use_tid7_reduction = bool(config["use_tid7_reduction"])
    use_warp_overlap = bool(config["use_warp_overlap"])
    use_contiguous_partials = bool(config["use_contiguous_partials"])
    use_t2_projection_ilp = bool(config["use_t2_projection_ilp"])

    projection_partials = torch.empty(
        (
            (tokens, num_splits, FULL_PROJECTION_OUTPUTS)
            if use_contiguous_partials
            else (
                (num_output_tiles, num_splits, output_tile_block)
                if tokens == 1
                else (tokens, num_output_tiles, num_splits, output_tile_block)
            )
        ),
        dtype=torch.float32,
        device=x_tensor.device,
    )
    sqsum_partials = torch.empty(
        (num_splits,) if tokens == 1 else (tokens, num_splits),
        dtype=torch.float32,
        device=x_tensor.device,
    )
    counter_stride_elements = COUNTER_U32_STRIDE if tokens > 1 else 1
    arrival_counter = torch.zeros(
        (
            (tokens, counter_stride_elements)
            if counter_stride_elements > 1
            else (tokens,)
        ),
        dtype=torch.uint32,
        device=x_tensor.device,
    )
    workspace_tensors = {
        "projection_partials": projection_partials,
        "sqsum_partials": sqsum_partials,
    }
    counter_tensors = {"arrival_counter": arrival_counter}
    requested_reduction = (
        "warp_shuffle_shared5_tid5"
        if use_tid5_reduction
        else (
            "warp_shuffle_shared4_tid4"
            if use_tid4_reduction
            else "warp_shuffle_shared7_tid7"
        )
    )
    metadata: dict[str, object] = {
        "T": tokens,
        "H": FULL_HIDDEN,
        "R": FULL_HC_MULT,
        "P": FULL_PROJECTION_OUTPUTS,
        "mode": "normal",
        "dtype": "bfloat16",
        "kernel_count": 1,
        "CTA": grid_ctas,
        "split": num_splits,
        "ctas_per_token": ctas_per_token,
        "hidden_per_split": hidden_per_split,
        "outputs_per_tile": outputs_per_tile,
        "output_tile_block": output_tile_block,
        "num_warps": split_num_warps,
        "producer_reduction": requested_reduction,
        "sinkhorn_implementation": "warp_shuffle_4x4_fast_rcp",
        "sinkhorn_reciprocal": "rcp.approx.ftz.f32",
        "overlap_implementation": (
            "routing_worker_1wg" if use_warp_overlap else "none"
        ),
        "handoff_implementation": "ptx_release_inc_u32_bar_red",
        "counter_dtype": "uint32",
        "counter_stride_elements": counter_stride_elements,
        "counter_stride_bytes": counter_stride_elements * 4,
        "atomic_inc_limit": ctas_per_token - 1,
        "handoff_barrier_participants": split_num_warps * 32,
        "producer_publish_barrier_implementation": "tle_full_cta_named_barrier",
        "producer_publish_barrier_id": None,
        "producer_publish_barrier_participants": split_num_warps * 32,
        "handoff_broadcast_barrier_implementation": "inline_ptx_bar_red_or_pred_id15",
        "handoff_broadcast_barrier_id": 15,
        "default_partition_warps": split_num_warps,
        "routing_worker_logical_warps": (
            FULL_OVERLAP_ROUTING_WARPS if use_warp_overlap else 0
        ),
        "physical_cta_warps": (
            FULL_OVERLAP_PHYSICAL_WARPS if use_warp_overlap else split_num_warps
        ),
        "physical_cta_threads": (
            FULL_OVERLAP_PHYSICAL_THREADS if use_warp_overlap else split_num_warps * 32
        ),
        "named_barrier_participants": (
            FULL_OVERLAP_LOGICAL_THREADS if use_warp_overlap else None
        ),
        "schedule": schedule,
        "dispatch_policy": "production_auto",
        "dispatch_enabled": True,
        "production_default_selected": True,
        "t2_contiguous_partials": use_contiguous_partials,
        "t2_projection_ilp": use_t2_projection_ilp,
        "pre_mix_broadcast_schedule": (
            "warp_specialized_shared_pre_mix"
            if use_warp_overlap
            else "triton_sum_select_per_channel"
        ),
        "epilogue_barrier_policy": "retained_existing_path",
        "sqsum_reduction_schedule": "serial_split_scalar_loads",
        "projection_accumulation_schedule": (
            "batch3_before_reduce" if use_t2_projection_ilp else "serial_output_reduce"
        ),
        "projection_partials_layout": (
            "token_split_output_contiguous"
            if use_contiguous_partials
            else "token_tile_split_padded" if tokens > 1 else "tile_split_padded"
        ),
        "projection_partials_shape": list(projection_partials.shape),
        "projection_partials_bytes": (
            projection_partials.numel() * projection_partials.element_size()
        ),
        "sqsum_partials_kernel_usage": "producer_and_tail_workspace",
        "sqsum_partials_shape": list(sqsum_partials.shape),
        "sqsum_partials_allocated_bytes": (
            sqsum_partials.numel() * sqsum_partials.element_size()
        ),
        "sqsum_partials_effective_kernel_bytes": (
            sqsum_partials.numel() * sqsum_partials.element_size()
        ),
        "report_eligible": True,
        "result_scope": "small_graph_production",
        "programmatic_dependent_launch": False,
        "performance_schedule": True,
        "workspace_tensor_count": len(workspace_tensors),
        "counter_tensor_count": len(counter_tensors),
        "counter_policy": "self_reset",
        "counter_increment_per_launch": ctas_per_token,
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
        schedule=schedule,
        tokens=tokens,
        num_splits=num_splits,
        hidden_per_split=hidden_per_split,
        ctas_per_token=ctas_per_token,
        counter_stride_elements=counter_stride_elements,
        projection_partials=projection_partials,
        sqsum_partials=sqsum_partials,
        arrival_counter=arrival_counter,
        outputs_per_tile=outputs_per_tile,
        output_tile_block=output_tile_block,
        num_output_tiles=num_output_tiles,
        grid_ctas=grid_ctas,
        split_num_warps=split_num_warps,
        use_hierarchical_reduction=use_hierarchical_reduction,
        use_tid4_reduction=use_tid4_reduction,
        use_tid5_reduction=use_tid5_reduction,
        use_tid7_reduction=use_tid7_reduction,
        use_warp_overlap=use_warp_overlap,
        use_contiguous_partials=use_contiguous_partials,
        use_t2_projection_ilp=use_t2_projection_ilp,
        workspace_tensors=workspace_tensors,
        counter_tensors=counter_tensors,
        metadata=metadata,
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
    intervals = [(*_byte_interval(tensor), name) for name, tensor in named_tensors]
    intervals.sort()
    for (_, previous_end, previous_name), (
        current_begin,
        _,
        current_name,
    ) in zip(intervals, intervals[1:]):
        if current_begin < previous_end:
            raise ValueError(f"{previous_name} and {current_name} must not overlap")


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
    "TLESM90MegaMHCPlan",
    "prepare_tle_sm90_mega_mhc",
)

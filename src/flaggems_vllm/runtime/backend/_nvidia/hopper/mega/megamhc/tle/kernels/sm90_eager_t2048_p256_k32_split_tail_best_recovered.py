"""Independent BM64/split15 TLE eager prototype for MegaMHC ``T=1028``.

This module follows the producer geometry of the CUDA eager p256 path:
17 M64 tiles times 15 uneven hidden splits, or 255 producer CTAs.  Every CTA
produces projection and squared-sum partials for one tile/split.  A per-tile
device-scope release ``atom.inc`` elects the fifteenth (last) arrival.  The
last arrival always joins a tile queue; earlier arrivals perform only 256
finite acquire polls and join when ready.  Ready CTAs atomically claim rows,
so the last arrival drains any work left by timed-out helpers.  There is no
full-grid barrier, unbounded spin loop, cooperative launch, or full-grid
residency requirement.

The tail advances the per-tile queue head by three.  Its validation schedule
maps the default eight warps onto two H4096 rows and the math four warps onto
the third row after three routing finalizations.
The module is explicit opt-in, does not participate in production dispatch,
allocates all workspace in prepare, and :meth:`EagerT1028P256Plan.run`
contains exactly one TLE launch.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from .. import fused_post_gemm_t64 as _t64
from . import sm90_small_family as _small


PROFILE_T2048 = (
    os.environ.get("DG_TLE_MHC_EAGER_P256_SPLIT_T2048", "0") == "1"
)
TOKENS = 2048 if PROFILE_T2048 else 1028
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = HIDDEN * HC_MULT
SINKHORN_ITERS = 20
BLOCK_M = 64
BLOCK_N = 32
HIDDEN_BLOCK = 32
NUM_HIDDEN_BLOCKS = HIDDEN // HIDDEN_BLOCK
NUM_SPLITS = 8 if PROFILE_T2048 else 15
BASE_BLOCKS_PER_SPLIT = NUM_HIDDEN_BLOCKS // NUM_SPLITS
WIDE_SPLITS = 0 if PROFILE_T2048 else 4
EXTRA_BLOCKS_PER_WIDE_SPLIT = 0 if PROFILE_T2048 else 2
MAX_BLOCKS_PER_SPLIT = (
    BASE_BLOCKS_PER_SPLIT + EXTRA_BLOCKS_PER_WIDE_SPLIT
)
SPLIT_HIDDEN_BLOCKS = (
    (16,) * 8
    if PROFILE_T2048
    else (10, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8)
)
NUM_M_TILES = (TOKENS + BLOCK_M - 1) // BLOCK_M
GRID_CTAS = NUM_M_TILES * NUM_SPLITS
NUM_WARPS = 8
MATH_WARPS = 4
MATH_REGS = 80
PIPE_STAGES = 8
BOUNDED_HELPER_POLLS_TARGET = 256

JIT_TOKENS = tl.constexpr(TOKENS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_HIDDEN_BLOCK = tl.constexpr(HIDDEN_BLOCK)
JIT_NUM_SPLITS = tl.constexpr(NUM_SPLITS)
JIT_BASE_BLOCKS_PER_SPLIT = tl.constexpr(BASE_BLOCKS_PER_SPLIT)
JIT_WIDE_SPLITS = tl.constexpr(WIDE_SPLITS)
JIT_EXTRA_BLOCKS_PER_WIDE_SPLIT = tl.constexpr(
    EXTRA_BLOCKS_PER_WIDE_SPLIT
)
JIT_MAX_BLOCKS_PER_SPLIT = tl.constexpr(MAX_BLOCKS_PER_SPLIT)
JIT_NUM_M_TILES = tl.constexpr(NUM_M_TILES)
JIT_GRID_CTAS = tl.constexpr(GRID_CTAS)
JIT_MATH_WARPS = tl.constexpr(MATH_WARPS)
JIT_MATH_REGS = tl.constexpr(MATH_REGS)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_SINKHORN_ITERS = tl.constexpr(SINKHORN_ITERS)
JIT_PRODUCER_THREADS = tl.constexpr(NUM_WARPS * 32)
JIT_LOGICAL_THREADS = tl.constexpr((NUM_WARPS + MATH_WARPS) * 32)


@triton.jit
def _bounded_tile_ready(state_base, physical_thread):
    """Publish one split and finite-poll the tile phase from thread zero."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, last, changed, keep_polling;
          .reg .u32 old_phase, phase, prior, poll;
          .reg .u64 phase_address, head_address;
          setp.eq.u32 leader, $2, 0;
          mov.u32 $0, 0;
          @!leader bra DONE;
          add.u64 phase_address, $1, 4;
          add.u64 head_address, $1, 8;
          ld.acquire.gpu.global.u32 old_phase, [phase_address];
          atom.release.gpu.global.inc.u32 prior, [$1], 14;
          setp.eq.u32 last, prior, 14;
          @!last bra POLL_INIT;
          fence.acquire.gpu;
          st.global.u32 [head_address], 0;
          atom.release.gpu.global.add.u32 phase, [phase_address], 1;
          mov.u32 $0, 1;
          bra DONE;
        POLL_INIT:
          mov.u32 poll, 0;
        POLL:
          ld.acquire.gpu.global.u32 phase, [phase_address];
          setp.ne.u32 changed, phase, old_phase;
          @changed mov.u32 $0, 1;
          @changed bra DONE;
          nanosleep.u32 32;
          add.u32 poll, poll, 1;
          setp.lt.u32 keep_polling, poll, 256;
          @keep_polling bra POLL;
        DONE:
        }
        """,
        constraints="=r,l,r",
        args=[state_base, physical_thread],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _claim_tile_row(state_base, physical_thread):
    """Claim three consecutive rows; only physical thread zero acts."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader;
          .reg .u64 head_address;
          setp.eq.u32 leader, $2, 0;
          mov.u32 $0, 4294967295;
          add.u64 head_address, $1, 8;
          @leader atom.global.add.u32 $0, [head_address], 3;
        }
        """,
        constraints="=r,l,r",
        args=[state_base, physical_thread],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _tile_phase_wait(
    state_base, phase_byte_offset, physical_thread, arrival_limit
):
    """Release-arrive and wait for all fifteen split CTAs of one tile."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, last, changed;
          .reg .u32 old_phase, phase, prior;
          .reg .u64 arrivals, phase_address, byte_offset;
          setp.eq.u32 leader, $2, 0;
          @!leader bra DONE;
          cvt.u64.u32 byte_offset, $3;
          add.u64 arrivals, $1, byte_offset;
          add.u64 phase_address, arrivals, 4;
          ld.acquire.gpu.global.u32 old_phase, [phase_address];
          atom.release.gpu.global.inc.u32 prior, [arrivals], $4;
          setp.eq.u32 last, prior, $4;
          @!last bra WAIT;
          fence.acquire.gpu;
          atom.release.gpu.global.add.u32 phase, [phase_address], 1;
          bra DONE;
        WAIT:
          ld.acquire.gpu.global.u32 phase, [phase_address];
          setp.ne.u32 changed, phase, old_phase;
          @changed bra DONE;
          nanosleep.u32 32;
          bra WAIT;
        DONE:
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,r,r,r",
        args=[
            state_base, physical_thread, phase_byte_offset, arrival_limit
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _profile_phase_timestamp(state_base, byte_offset, physical_thread):
    """Atomically retain the latest SM global-timer value for one tile."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader;
          .reg .u32 discard;
          .reg .u64 address, byte_offset, timestamp, prior;
          setp.eq.u32 leader, $3, 0;
          @!leader bra DONE;
          cvt.u64.u32 byte_offset, $2;
          add.u64 address, $1, byte_offset;
          mov.u64 timestamp, %globaltimer;
          atom.relaxed.gpu.global.max.u64 prior, [address], timestamp;
        DONE:
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,r,r",
        args=[state_base, byte_offset, physical_thread],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _split_hidden_combine_ptx(
    pre_base,
    residual_base,
    y_base,
    square_base,
    local_sums_shared,
    physical_thread,
    hidden_quad_offset,
    hidden_quad_count,
    live_rows,
):
    """Compute one split's packed BF16 boundary and local RMS sums."""
    sums_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(local_sums_shared, (0,))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active, row_live, lane0;
          .reg .u32 lane, warp, row, quad, sums_addr, shared_address;
          .reg .b32 r00, r01, r10, r11, r20, r21, r30, r31;
          .reg .b32 packed0, packed1;
          .reg .b16 h0, h1, h2, h3;
          .reg .f32 p0, p1, p2, p3;
          .reg .f32 a0, a1, a2, a3;
          .reg .f32 b0, b1, b2, b3;
          .reg .f32 c0, c1, c2, c3;
          .reg .f32 d0, d1, d2, d3;
          .reg .f32 v0, v1, v2, v3, sum, tmp;
          .reg .u64 pre_row, residual_row, y_row, address, offset;

          mov.u32 sums_addr, $5;
          and.b32 lane, $6, 31;
          shr.u32 warp, $6, 5;
          setp.lt.u32 active, $6, $8;
          setp.lt.u32 row_live, $6, 64;
          mul.lo.u32 quad, $6, 4;
          add.u32 quad, quad, $7;
          mul.wide.u32 offset, quad, 2;
          mad.lo.u32 shared_address, $6, 4, sums_addr;
          @row_live st.shared.u32 [shared_address], 0;
          barrier.cta.sync 13, 384;
          mov.u32 row, 0;
        DG_TLE_MHC_SPLIT_COMBINE_ROW:
          setp.lt.u32 row_live, row, $9;
          @!row_live bra DG_TLE_MHC_SPLIT_COMBINE_DONE;
          mul.wide.u32 pre_row, row, 128;
          add.u64 pre_row, $1, pre_row;
          ld.global.f32 p0, [pre_row];
          ld.global.f32 p1, [pre_row+4];
          ld.global.f32 p2, [pre_row+8];
          ld.global.f32 p3, [pre_row+12];
          mul.wide.u32 residual_row, row, 32768;
          add.u64 residual_row, $2, residual_row;
          add.u64 residual_row, residual_row, offset;
          mov.b32 r00, 0;
          mov.b32 r01, 0;
          mov.b32 r10, 0;
          mov.b32 r11, 0;
          mov.b32 r20, 0;
          mov.b32 r21, 0;
          mov.b32 r30, 0;
          mov.b32 r31, 0;
          @active ld.global.v2.b32 {r00, r01}, [residual_row];
          add.u64 address, residual_row, 8192;
          @active ld.global.v2.b32 {r10, r11}, [address];
          add.u64 address, address, 8192;
          @active ld.global.v2.b32 {r20, r21}, [address];
          add.u64 address, address, 8192;
          @active ld.global.v2.b32 {r30, r31}, [address];
          mov.b32 {h0, h1}, r00;
          cvt.f32.bf16 a0, h0;
          cvt.f32.bf16 a1, h1;
          mov.b32 {h0, h1}, r01;
          cvt.f32.bf16 a2, h0;
          cvt.f32.bf16 a3, h1;
          mov.b32 {h0, h1}, r10;
          cvt.f32.bf16 b0, h0;
          cvt.f32.bf16 b1, h1;
          mov.b32 {h0, h1}, r11;
          cvt.f32.bf16 b2, h0;
          cvt.f32.bf16 b3, h1;
          mov.b32 {h0, h1}, r20;
          cvt.f32.bf16 c0, h0;
          cvt.f32.bf16 c1, h1;
          mov.b32 {h0, h1}, r21;
          cvt.f32.bf16 c2, h0;
          cvt.f32.bf16 c3, h1;
          mov.b32 {h0, h1}, r30;
          cvt.f32.bf16 d0, h0;
          cvt.f32.bf16 d1, h1;
          mov.b32 {h0, h1}, r31;
          cvt.f32.bf16 d2, h0;
          cvt.f32.bf16 d3, h1;
          mul.f32 v0, a0, p0;
          mul.f32 v1, a1, p0;
          mul.f32 v2, a2, p0;
          mul.f32 v3, a3, p0;
          fma.rn.f32 v0, b0, p1, v0;
          fma.rn.f32 v1, b1, p1, v1;
          fma.rn.f32 v2, b2, p1, v2;
          fma.rn.f32 v3, b3, p1, v3;
          fma.rn.f32 v0, c0, p2, v0;
          fma.rn.f32 v1, c1, p2, v1;
          fma.rn.f32 v2, c2, p2, v2;
          fma.rn.f32 v3, c3, p2, v3;
          fma.rn.f32 v0, d0, p3, v0;
          fma.rn.f32 v1, d1, p3, v1;
          fma.rn.f32 v2, d2, p3, v2;
          fma.rn.f32 v3, d3, p3, v3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          mul.wide.u32 y_row, row, 8192;
          add.u64 y_row, $3, y_row;
          add.u64 y_row, y_row, offset;
          @active st.global.v2.b32 [y_row], {packed0, packed1};
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          mul.f32 sum, v0, v0;
          fma.rn.f32 sum, v1, v1, sum;
          fma.rn.f32 sum, v2, v2, sum;
          fma.rn.f32 sum, v3, v3, sum;
          shfl.sync.bfly.b32 tmp, sum, 16, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 8, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 4, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 2, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 1, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          setp.eq.u32 lane0, lane, 0;
          mad.lo.u32 shared_address, row, 4, sums_addr;
          @lane0 atom.shared.add.f32 tmp, [shared_address], sum;
          add.u32 row, row, 1;
          bra DG_TLE_MHC_SPLIT_COMBINE_ROW;
        DG_TLE_MHC_SPLIT_COMBINE_DONE:
          barrier.cta.sync 13, 384;
          setp.lt.u32 row_live, $6, $9;
          @!row_live bra DG_TLE_MHC_SPLIT_COMBINE_EXIT;
          mul.wide.u32 offset, $6, 128;
          add.u64 address, $4, offset;
          mad.lo.u32 shared_address, $6, 4, sums_addr;
          ld.shared.f32 sum, [shared_address];
          st.global.f32 [address], sum;
        DG_TLE_MHC_SPLIT_COMBINE_EXIT:
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,l,l,r,r,r,r,r",
        args=[
            pre_base,
            residual_base,
            y_base,
            square_base,
            sums_address,
            physical_thread,
            hidden_quad_offset,
            hidden_quad_count,
            live_rows,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _split_hidden_normalize_ptx(
    y_base,
    rmsnorm_weight_ptr,
    norm_shared,
    physical_thread,
    hidden_quad_offset,
    hidden_quad_count,
    live_rows,
):
    """Normalize the BF16 scratch row for this split in place."""
    norm_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(norm_shared, (0,))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active, more;
          .reg .u32 row, quad, shared_address;
          .reg .b32 packed0, packed1, weight0, weight1;
          .reg .b16 h0, h1, h2, h3, w0, w1, w2, w3;
          .reg .f32 v0, v1, v2, v3, f0, f1, f2, f3, norm;
          .reg .u64 y_row, address, offset;
          setp.lt.u32 active, $4, $6;
          mul.lo.u32 quad, $4, 4;
          add.u32 quad, quad, $5;
          mul.wide.u32 offset, quad, 2;
          add.u64 address, $2, offset;
          ld.global.v2.b32 {weight0, weight1}, [address];
          mov.u32 row, 0;
        DG_TLE_MHC_SPLIT_NORM_ROW:
          setp.lt.u32 more, row, $7;
          @!more bra DG_TLE_MHC_SPLIT_NORM_DONE;
          mad.lo.u32 shared_address, row, 4, $3;
          ld.shared.f32 norm, [shared_address];
          mul.wide.u32 y_row, row, 8192;
          add.u64 y_row, $1, y_row;
          add.u64 y_row, y_row, offset;
          mov.b32 packed0, 0;
          mov.b32 packed1, 0;
          @active ld.global.v2.b32 {packed0, packed1}, [y_row];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          mov.b32 {w0, w1}, weight0;
          mov.b32 {w2, w3}, weight1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          cvt.f32.bf16 f0, w0;
          cvt.f32.bf16 f1, w1;
          cvt.f32.bf16 f2, w2;
          cvt.f32.bf16 f3, w3;
          mul.f32 v0, v0, norm;
          mul.f32 v1, v1, norm;
          mul.f32 v2, v2, norm;
          mul.f32 v3, v3, norm;
          mul.f32 v0, v0, f0;
          mul.f32 v1, v1, f1;
          mul.f32 v2, v2, f2;
          mul.f32 v3, v3, f3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          @active st.global.v2.b32 [y_row], {packed0, packed1};
          add.u32 row, row, 1;
          bra DG_TLE_MHC_SPLIT_NORM_ROW;
        DG_TLE_MHC_SPLIT_NORM_DONE:
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,r,r,r,r,r",
        args=[
            y_base,
            rmsnorm_weight_ptr,
            norm_address,
            physical_thread,
            hidden_quad_offset,
            hidden_quad_count,
            live_rows,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _split_hidden_combine_wave_ptx(
    pre_base,
    residual_base,
    tail_scratch,
    square_base,
    physical_thread,
    hidden_quad_offset,
    hidden_quad_count,
    live_rows,
):
    """Use all twelve warps: one row per warp, six row waves."""
    pre_address = tl.cast(pre_base, tl.uint64)
    residual_address = tl.cast(residual_base, tl.uint64)
    scratch_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(tail_scratch, (0, 0, 0))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    square_address = tl.cast(square_base, tl.uint64)
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred row_live, quad_live, lane0, store_sum, more;
          .reg .u32 warp, lane, wave, row, qiter, quad_iters, local_quad, quad;
          .reg .u32 scratch_row;
          .reg .b32 r00, r01, r10, r11, r20, r21, r30, r31;
          .reg .b32 packed0, packed1;
          .reg .b16 h0, h1, h2, h3;
          .reg .f32 p0, p1, p2, p3;
          .reg .f32 a0, a1, a2, a3, b0, b1, b2, b3;
          .reg .f32 c0, c1, c2, c3, d0, d1, d2, d3;
          .reg .f32 v0, v1, v2, v3, sum, tmp;
          .reg .u64 pre_row, residual_row, address, offset;
          shr.u32 warp, $5, 5;
          and.b32 lane, $5, 31;
          add.u32 quad_iters, $7, 31;
          shr.u32 quad_iters, quad_iters, 5;
          mov.u32 wave, 0;
        DG_TLE_MHC_SPLIT_WAVE:
          mad.lo.u32 row, wave, 12, warp;
          setp.lt.u32 row_live, row, $8;
          mov.f32 p0, 0f00000000;
          mov.f32 p1, 0f00000000;
          mov.f32 p2, 0f00000000;
          mov.f32 p3, 0f00000000;
          mul.wide.u32 pre_row, row, 128;
          add.u64 pre_row, $1, pre_row;
          @row_live ld.global.f32 p0, [pre_row];
          @row_live ld.global.f32 p1, [pre_row+4];
          @row_live ld.global.f32 p2, [pre_row+8];
          @row_live ld.global.f32 p3, [pre_row+12];
          mov.f32 sum, 0f00000000;
          mov.u32 qiter, 0;
        DG_TLE_MHC_SPLIT_QUAD:
          mad.lo.u32 local_quad, qiter, 32, lane;
          setp.lt.u32 quad_live, local_quad, $7;
          and.pred quad_live, quad_live, row_live;
          add.u32 quad, local_quad, $6;
          mul.wide.u32 offset, quad, 8;
          mul.wide.u32 residual_row, row, 32768;
          add.u64 residual_row, $2, residual_row;
          add.u64 residual_row, residual_row, offset;
          mov.b32 r00, 0;
          mov.b32 r01, 0;
          mov.b32 r10, 0;
          mov.b32 r11, 0;
          mov.b32 r20, 0;
          mov.b32 r21, 0;
          mov.b32 r30, 0;
          mov.b32 r31, 0;
          @quad_live ld.global.v2.b32 {r00, r01}, [residual_row];
          add.u64 address, residual_row, 8192;
          @quad_live ld.global.v2.b32 {r10, r11}, [address];
          add.u64 address, address, 8192;
          @quad_live ld.global.v2.b32 {r20, r21}, [address];
          add.u64 address, address, 8192;
          @quad_live ld.global.v2.b32 {r30, r31}, [address];
          mov.b32 {h0, h1}, r00;
          cvt.f32.bf16 a0, h0;
          cvt.f32.bf16 a1, h1;
          mov.b32 {h0, h1}, r01;
          cvt.f32.bf16 a2, h0;
          cvt.f32.bf16 a3, h1;
          mov.b32 {h0, h1}, r10;
          cvt.f32.bf16 b0, h0;
          cvt.f32.bf16 b1, h1;
          mov.b32 {h0, h1}, r11;
          cvt.f32.bf16 b2, h0;
          cvt.f32.bf16 b3, h1;
          mov.b32 {h0, h1}, r20;
          cvt.f32.bf16 c0, h0;
          cvt.f32.bf16 c1, h1;
          mov.b32 {h0, h1}, r21;
          cvt.f32.bf16 c2, h0;
          cvt.f32.bf16 c3, h1;
          mov.b32 {h0, h1}, r30;
          cvt.f32.bf16 d0, h0;
          cvt.f32.bf16 d1, h1;
          mov.b32 {h0, h1}, r31;
          cvt.f32.bf16 d2, h0;
          cvt.f32.bf16 d3, h1;
          mul.f32 v0, a0, p0;
          mul.f32 v1, a1, p0;
          mul.f32 v2, a2, p0;
          mul.f32 v3, a3, p0;
          fma.rn.f32 v0, b0, p1, v0;
          fma.rn.f32 v1, b1, p1, v1;
          fma.rn.f32 v2, b2, p1, v2;
          fma.rn.f32 v3, b3, p1, v3;
          fma.rn.f32 v0, c0, p2, v0;
          fma.rn.f32 v1, c1, p2, v1;
          fma.rn.f32 v2, c2, p2, v2;
          fma.rn.f32 v3, c3, p2, v3;
          fma.rn.f32 v0, d0, p3, v0;
          fma.rn.f32 v1, d1, p3, v1;
          fma.rn.f32 v2, d2, p3, v2;
          fma.rn.f32 v3, d3, p3, v3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          mad.lo.u32 scratch_row, row, 640, $3;
          mad.lo.u32 scratch_row, local_quad, 8, scratch_row;
          @quad_live st.shared.v2.b32 [scratch_row], {packed0, packed1};
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 sum, v0, v0, sum;
          fma.rn.f32 sum, v1, v1, sum;
          fma.rn.f32 sum, v2, v2, sum;
          fma.rn.f32 sum, v3, v3, sum;
          add.u32 qiter, qiter, 1;
          setp.lt.u32 more, qiter, quad_iters;
          @more bra.uni DG_TLE_MHC_SPLIT_QUAD;
          shfl.sync.bfly.b32 tmp, sum, 16, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 8, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 4, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 2, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 1, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          setp.eq.u32 lane0, lane, 0;
          and.pred store_sum, lane0, row_live;
          mul.wide.u32 offset, row, 128;
          add.u64 address, $4, offset;
          @store_sum st.global.f32 [address], sum;
          add.u32 wave, wave, 1;
          setp.lt.u32 more, wave, 6;
          @more bra.uni DG_TLE_MHC_SPLIT_WAVE;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,r,l,r,r,r,r",
        args=[
            pre_address,
            residual_address,
            scratch_address,
            square_address,
            physical_thread,
            hidden_quad_offset,
            hidden_quad_count,
            live_rows,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _split_hidden_combine_bf16x8_ptx(
    pre_base,
    residual_base,
    tail_scratch,
    square_base,
    physical_thread,
    hidden_quad_offset,
    hidden_quad_count,
    live_rows,
    scratch_stride_bytes,
    scratch_chunk_bytes,
    accumulate_square,
):
    """Combine 256 values as BF16x8 plus one optional 64-value tail."""
    pre_address = tl.cast(pre_base, tl.uint64)
    residual_address = tl.cast(residual_base, tl.uint64)
    scratch_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(tail_scratch, (0, 0, 0))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    square_address = tl.cast(square_base, tl.uint64)
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred row_live, wide, tail_lane, tail_live, lane0, store_sum, more;
          .reg .pred accumulate, store_accumulate;
          .reg .u32 warp, lane, wave, row, quad, scratch_row, scratch_tail;
          .reg .b32 a00, a01, a02, a03, b00, b01, b02, b03;
          .reg .b32 c00, c01, c02, c03, d00, d01, d02, d03;
          .reg .b32 packed0, packed1;
          .reg .b16 h0, h1, h2, h3;
          .reg .f32 p0, p1, p2, p3;
          .reg .f32 a0, a1, a2, a3, b0, b1, b2, b3;
          .reg .f32 c0, c1, c2, c3, d0, d1, d2, d3;
          .reg .f32 v0, v1, v2, v3, sum, tmp;
          .reg .u64 pre_row, residual_row, address, offset;
          shr.u32 warp, $5, 5;
          and.b32 lane, $5, 31;
          setp.gt.u32 wide, $7, 64;
          setp.lt.u32 tail_lane, lane, 16;
          mov.u32 wave, 0;
        DG_TLE_MHC_SPLIT_COMBINE8_WAVE:
          mad.lo.u32 row, wave, 12, warp;
          setp.lt.u32 row_live, row, $8;
          mov.f32 p0, 0f00000000;
          mov.f32 p1, 0f00000000;
          mov.f32 p2, 0f00000000;
          mov.f32 p3, 0f00000000;
          mul.wide.u32 pre_row, row, 128;
          add.u64 pre_row, $1, pre_row;
          @row_live ld.global.f32 p0, [pre_row];
          @row_live ld.global.f32 p1, [pre_row+4];
          @row_live ld.global.f32 p2, [pre_row+8];
          @row_live ld.global.f32 p3, [pre_row+12];
          mov.f32 sum, 0f00000000;

          mad.lo.u32 quad, lane, 2, $6;
          mul.wide.u32 offset, quad, 8;
          mul.wide.u32 residual_row, row, 32768;
          add.u64 residual_row, $2, residual_row;
          add.u64 residual_row, residual_row, offset;
          mov.b32 a00, 0; mov.b32 a01, 0; mov.b32 a02, 0; mov.b32 a03, 0;
          mov.b32 b00, 0; mov.b32 b01, 0; mov.b32 b02, 0; mov.b32 b03, 0;
          mov.b32 c00, 0; mov.b32 c01, 0; mov.b32 c02, 0; mov.b32 c03, 0;
          mov.b32 d00, 0; mov.b32 d01, 0; mov.b32 d02, 0; mov.b32 d03, 0;
          @row_live ld.global.v4.b32 {a00, a01, a02, a03}, [residual_row];
          add.u64 address, residual_row, 8192;
          @row_live ld.global.v4.b32 {b00, b01, b02, b03}, [address];
          add.u64 address, address, 8192;
          @row_live ld.global.v4.b32 {c00, c01, c02, c03}, [address];
          add.u64 address, address, 8192;
          @row_live ld.global.v4.b32 {d00, d01, d02, d03}, [address];
          mad.lo.u32 scratch_row, row, $9, $3;
          add.u32 scratch_row, scratch_row, $10;
          mad.lo.u32 scratch_row, lane, 16, scratch_row;

          mov.b32 {h0, h1}, a00; mov.b32 {h2, h3}, a01;
          cvt.f32.bf16 a0, h0; cvt.f32.bf16 a1, h1;
          cvt.f32.bf16 a2, h2; cvt.f32.bf16 a3, h3;
          mov.b32 {h0, h1}, b00; mov.b32 {h2, h3}, b01;
          cvt.f32.bf16 b0, h0; cvt.f32.bf16 b1, h1;
          cvt.f32.bf16 b2, h2; cvt.f32.bf16 b3, h3;
          mov.b32 {h0, h1}, c00; mov.b32 {h2, h3}, c01;
          cvt.f32.bf16 c0, h0; cvt.f32.bf16 c1, h1;
          cvt.f32.bf16 c2, h2; cvt.f32.bf16 c3, h3;
          mov.b32 {h0, h1}, d00; mov.b32 {h2, h3}, d01;
          cvt.f32.bf16 d0, h0; cvt.f32.bf16 d1, h1;
          cvt.f32.bf16 d2, h2; cvt.f32.bf16 d3, h3;
          mul.f32 v0, a0, p0; mul.f32 v1, a1, p0;
          mul.f32 v2, a2, p0; mul.f32 v3, a3, p0;
          fma.rn.f32 v0, b0, p1, v0; fma.rn.f32 v1, b1, p1, v1;
          fma.rn.f32 v2, b2, p1, v2; fma.rn.f32 v3, b3, p1, v3;
          fma.rn.f32 v0, c0, p2, v0; fma.rn.f32 v1, c1, p2, v1;
          fma.rn.f32 v2, c2, p2, v2; fma.rn.f32 v3, c3, p2, v3;
          fma.rn.f32 v0, d0, p3, v0; fma.rn.f32 v1, d1, p3, v1;
          fma.rn.f32 v2, d2, p3, v2; fma.rn.f32 v3, d3, p3, v3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          @row_live st.shared.v2.b32 [scratch_row], {packed0, packed1};
          mov.b32 {h0, h1}, packed0; mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0; cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2; cvt.f32.bf16 v3, h3;
          fma.rn.f32 sum, v0, v0, sum; fma.rn.f32 sum, v1, v1, sum;
          fma.rn.f32 sum, v2, v2, sum; fma.rn.f32 sum, v3, v3, sum;

          mov.b32 {h0, h1}, a02; mov.b32 {h2, h3}, a03;
          cvt.f32.bf16 a0, h0; cvt.f32.bf16 a1, h1;
          cvt.f32.bf16 a2, h2; cvt.f32.bf16 a3, h3;
          mov.b32 {h0, h1}, b02; mov.b32 {h2, h3}, b03;
          cvt.f32.bf16 b0, h0; cvt.f32.bf16 b1, h1;
          cvt.f32.bf16 b2, h2; cvt.f32.bf16 b3, h3;
          mov.b32 {h0, h1}, c02; mov.b32 {h2, h3}, c03;
          cvt.f32.bf16 c0, h0; cvt.f32.bf16 c1, h1;
          cvt.f32.bf16 c2, h2; cvt.f32.bf16 c3, h3;
          mov.b32 {h0, h1}, d02; mov.b32 {h2, h3}, d03;
          cvt.f32.bf16 d0, h0; cvt.f32.bf16 d1, h1;
          cvt.f32.bf16 d2, h2; cvt.f32.bf16 d3, h3;
          mul.f32 v0, a0, p0; mul.f32 v1, a1, p0;
          mul.f32 v2, a2, p0; mul.f32 v3, a3, p0;
          fma.rn.f32 v0, b0, p1, v0; fma.rn.f32 v1, b1, p1, v1;
          fma.rn.f32 v2, b2, p1, v2; fma.rn.f32 v3, b3, p1, v3;
          fma.rn.f32 v0, c0, p2, v0; fma.rn.f32 v1, c1, p2, v1;
          fma.rn.f32 v2, c2, p2, v2; fma.rn.f32 v3, c3, p2, v3;
          fma.rn.f32 v0, d0, p3, v0; fma.rn.f32 v1, d1, p3, v1;
          fma.rn.f32 v2, d2, p3, v2; fma.rn.f32 v3, d3, p3, v3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          @row_live st.shared.v2.b32 [scratch_row+8], {packed0, packed1};
          mov.b32 {h0, h1}, packed0; mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0; cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2; cvt.f32.bf16 v3, h3;
          fma.rn.f32 sum, v0, v0, sum; fma.rn.f32 sum, v1, v1, sum;
          fma.rn.f32 sum, v2, v2, sum; fma.rn.f32 sum, v3, v3, sum;

          and.pred tail_live, wide, tail_lane;
          and.pred tail_live, tail_live, row_live;
          add.u32 quad, $6, 64;
          add.u32 quad, quad, lane;
          mul.wide.u32 offset, quad, 8;
          mul.wide.u32 residual_row, row, 32768;
          add.u64 residual_row, $2, residual_row;
          add.u64 residual_row, residual_row, offset;
          mov.b32 a00, 0; mov.b32 a01, 0;
          mov.b32 b00, 0; mov.b32 b01, 0;
          mov.b32 c00, 0; mov.b32 c01, 0;
          mov.b32 d00, 0; mov.b32 d01, 0;
          @tail_live ld.global.v2.b32 {a00, a01}, [residual_row];
          add.u64 address, residual_row, 8192;
          @tail_live ld.global.v2.b32 {b00, b01}, [address];
          add.u64 address, address, 8192;
          @tail_live ld.global.v2.b32 {c00, c01}, [address];
          add.u64 address, address, 8192;
          @tail_live ld.global.v2.b32 {d00, d01}, [address];
          mov.b32 {h0, h1}, a00; mov.b32 {h2, h3}, a01;
          cvt.f32.bf16 a0, h0; cvt.f32.bf16 a1, h1;
          cvt.f32.bf16 a2, h2; cvt.f32.bf16 a3, h3;
          mov.b32 {h0, h1}, b00; mov.b32 {h2, h3}, b01;
          cvt.f32.bf16 b0, h0; cvt.f32.bf16 b1, h1;
          cvt.f32.bf16 b2, h2; cvt.f32.bf16 b3, h3;
          mov.b32 {h0, h1}, c00; mov.b32 {h2, h3}, c01;
          cvt.f32.bf16 c0, h0; cvt.f32.bf16 c1, h1;
          cvt.f32.bf16 c2, h2; cvt.f32.bf16 c3, h3;
          mov.b32 {h0, h1}, d00; mov.b32 {h2, h3}, d01;
          cvt.f32.bf16 d0, h0; cvt.f32.bf16 d1, h1;
          cvt.f32.bf16 d2, h2; cvt.f32.bf16 d3, h3;
          mul.f32 v0, a0, p0; mul.f32 v1, a1, p0;
          mul.f32 v2, a2, p0; mul.f32 v3, a3, p0;
          fma.rn.f32 v0, b0, p1, v0; fma.rn.f32 v1, b1, p1, v1;
          fma.rn.f32 v2, b2, p1, v2; fma.rn.f32 v3, b3, p1, v3;
          fma.rn.f32 v0, c0, p2, v0; fma.rn.f32 v1, c1, p2, v1;
          fma.rn.f32 v2, c2, p2, v2; fma.rn.f32 v3, c3, p2, v3;
          fma.rn.f32 v0, d0, p3, v0; fma.rn.f32 v1, d1, p3, v1;
          fma.rn.f32 v2, d2, p3, v2; fma.rn.f32 v3, d3, p3, v3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          mad.lo.u32 scratch_tail, row, $9, $3;
          add.u32 scratch_tail, scratch_tail, $10;
          mad.lo.u32 scratch_tail, lane, 8, scratch_tail;
          add.u32 scratch_tail, scratch_tail, 512;
          @tail_live st.shared.v2.b32 [scratch_tail], {packed0, packed1};
          mov.b32 {h0, h1}, packed0; mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0; cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2; cvt.f32.bf16 v3, h3;
          fma.rn.f32 sum, v0, v0, sum; fma.rn.f32 sum, v1, v1, sum;
          fma.rn.f32 sum, v2, v2, sum; fma.rn.f32 sum, v3, v3, sum;

          shfl.sync.bfly.b32 tmp, sum, 16, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 8, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 4, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 2, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 1, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          setp.eq.u32 lane0, lane, 0;
          and.pred store_sum, lane0, row_live;
          mul.wide.u32 offset, row, 128;
          add.u64 address, $4, offset;
          setp.ne.u32 accumulate, $11, 0;
          and.pred store_accumulate, store_sum, accumulate;
          @store_accumulate ld.global.f32 tmp, [address];
          @store_accumulate add.rn.f32 sum, sum, tmp;
          @store_sum st.global.f32 [address], sum;
          add.u32 wave, wave, 1;
          setp.lt.u32 more, wave, 6;
          @more bra.uni DG_TLE_MHC_SPLIT_COMBINE8_WAVE;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,r,l,r,r,r,r,r,r,r",
        args=[
            pre_address,
            residual_address,
            scratch_address,
            square_address,
            physical_thread,
            hidden_quad_offset,
            hidden_quad_count,
            live_rows,
            scratch_stride_bytes,
            scratch_chunk_bytes,
            accumulate_square,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _split_hidden_normalize_wave_ptx(
    tail_scratch,
    y_base,
    rmsnorm_weight_ptr,
    norm_shared,
    physical_thread,
    hidden_quad_offset,
    hidden_quad_count,
    live_rows,
):
    """Normalize the same six row waves with all twelve warps."""
    scratch_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(tail_scratch, (0, 0, 0))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    norm_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(norm_shared, (0,))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    y_address = tl.cast(y_base, tl.uint64)
    weight_address = tl.cast(rmsnorm_weight_ptr, tl.uint64)
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred row_live, quad_valid, quad_live, more;
          .reg .u32 warp, lane, wave, row, qiter, quad_iters, local_quad, quad;
          .reg .u32 shared_address, scratch_row;
          .reg .b32 packed0, packed1, weight0, weight1;
          .reg .b16 h0, h1, h2, h3, w0, w1, w2, w3;
          .reg .f32 v0, v1, v2, v3, f0, f1, f2, f3, norm;
          .reg .u64 y_row, address, offset;
          shr.u32 warp, $5, 5;
          and.b32 lane, $5, 31;
          add.u32 quad_iters, $7, 31;
          shr.u32 quad_iters, quad_iters, 5;
          mov.u32 qiter, 0;
        DG_TLE_MHC_SPLIT_NORM_QUAD:
          mad.lo.u32 local_quad, qiter, 32, lane;
          setp.lt.u32 quad_valid, local_quad, $7;
          add.u32 quad, local_quad, $6;
          mul.wide.u32 offset, quad, 8;
          mov.b32 weight0, 0;
          mov.b32 weight1, 0;
          add.u64 address, $3, offset;
          @quad_valid ld.global.v2.b32 {weight0, weight1}, [address];
          mov.b32 {w0, w1}, weight0;
          mov.b32 {w2, w3}, weight1;
          cvt.f32.bf16 f0, w0;
          cvt.f32.bf16 f1, w1;
          cvt.f32.bf16 f2, w2;
          cvt.f32.bf16 f3, w3;
          mov.u32 wave, 0;
        DG_TLE_MHC_SPLIT_NORM_WAVE:
          mad.lo.u32 row, wave, 12, warp;
          setp.lt.u32 row_live, row, $8;
          and.pred quad_live, quad_valid, row_live;
          mad.lo.u32 shared_address, row, 4, $4;
          mov.f32 norm, 0f00000000;
          @row_live ld.shared.f32 norm, [shared_address];
          mul.wide.u32 y_row, row, 8192;
          add.u64 y_row, $2, y_row;
          add.u64 y_row, y_row, offset;
          mov.b32 packed0, 0;
          mov.b32 packed1, 0;
          mad.lo.u32 scratch_row, row, 640, $1;
          mad.lo.u32 scratch_row, local_quad, 8, scratch_row;
          @quad_live ld.shared.v2.b32 {packed0, packed1}, [scratch_row];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          mul.f32 v0, v0, norm;
          mul.f32 v1, v1, norm;
          mul.f32 v2, v2, norm;
          mul.f32 v3, v3, norm;
          mul.f32 v0, v0, f0;
          mul.f32 v1, v1, f1;
          mul.f32 v2, v2, f2;
          mul.f32 v3, v3, f3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          @quad_live st.global.v2.b32 [y_row], {packed0, packed1};
          add.u32 wave, wave, 1;
          setp.lt.u32 more, wave, 6;
          @more bra.uni DG_TLE_MHC_SPLIT_NORM_WAVE;
          add.u32 qiter, qiter, 1;
          setp.lt.u32 more, qiter, quad_iters;
          @more bra.uni DG_TLE_MHC_SPLIT_NORM_QUAD;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,l,l,r,r,r,r,r",
        args=[
            scratch_address,
            y_address,
            weight_address,
            norm_address,
            physical_thread,
            hidden_quad_offset,
            hidden_quad_count,
            live_rows,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _split_hidden_normalize_bf16x8_ptx(
    tail_scratch,
    y_base,
    rmsnorm_weight_ptr,
    norm_shared,
    physical_thread,
    hidden_quad_offset,
    hidden_quad_count,
    live_rows,
    scratch_stride_bytes,
    scratch_chunk_bytes,
):
    """Normalize 256 values as BF16x8 plus one optional 64-value tail."""
    scratch_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(tail_scratch, (0, 0, 0))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    norm_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(norm_shared, (0,))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    y_address = tl.cast(y_base, tl.uint64)
    weight_address = tl.cast(rmsnorm_weight_ptr, tl.uint64)
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred row_live, wide, tail_lane, tail_live, more;
          .reg .u32 warp, lane, wave, row, quad, shared_norm;
          .reg .u32 scratch_row, scratch_tail;
          .reg .b32 packed0, packed1, packed2, packed3;
          .reg .b32 weight0, weight1, weight2, weight3;
          .reg .b32 tail_weight0, tail_weight1;
          .reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;
          .reg .b16 w0, w1, w2, w3, w4, w5, w6, w7;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
          .reg .f32 tf0, tf1, tf2, tf3, norm;
          .reg .u64 y_row, address, main_offset, tail_offset;
          shr.u32 warp, $5, 5;
          and.b32 lane, $5, 31;

          mad.lo.u32 quad, lane, 2, $6;
          mul.wide.u32 main_offset, quad, 8;
          mov.b32 weight0, 0;
          mov.b32 weight1, 0;
          mov.b32 weight2, 0;
          mov.b32 weight3, 0;
          add.u64 address, $3, main_offset;
          ld.global.v4.b32 {weight0, weight1, weight2, weight3}, [address];
          mov.b32 {w0, w1}, weight0;
          mov.b32 {w2, w3}, weight1;
          mov.b32 {w4, w5}, weight2;
          mov.b32 {w6, w7}, weight3;
          cvt.f32.bf16 f0, w0;
          cvt.f32.bf16 f1, w1;
          cvt.f32.bf16 f2, w2;
          cvt.f32.bf16 f3, w3;
          cvt.f32.bf16 f4, w4;
          cvt.f32.bf16 f5, w5;
          cvt.f32.bf16 f6, w6;
          cvt.f32.bf16 f7, w7;

          setp.gt.u32 wide, $7, 64;
          setp.lt.u32 tail_lane, lane, 16;
          and.pred tail_live, wide, tail_lane;
          add.u32 quad, $6, 64;
          add.u32 quad, quad, lane;
          mul.wide.u32 tail_offset, quad, 8;
          mov.b32 tail_weight0, 0;
          mov.b32 tail_weight1, 0;
          add.u64 address, $3, tail_offset;
          @tail_live ld.global.v2.b32 {tail_weight0, tail_weight1}, [address];
          mov.b32 {w0, w1}, tail_weight0;
          mov.b32 {w2, w3}, tail_weight1;
          cvt.f32.bf16 tf0, w0;
          cvt.f32.bf16 tf1, w1;
          cvt.f32.bf16 tf2, w2;
          cvt.f32.bf16 tf3, w3;

          mov.u32 wave, 0;
        DG_TLE_MHC_SPLIT_NORM8_WAVE:
          mad.lo.u32 row, wave, 12, warp;
          setp.lt.u32 row_live, row, $8;
          mad.lo.u32 shared_norm, row, 4, $4;
          mov.f32 norm, 0f00000000;
          @row_live ld.shared.f32 norm, [shared_norm];
          mad.lo.u32 scratch_row, row, $9, $1;
          add.u32 scratch_row, scratch_row, $10;
          mad.lo.u32 scratch_row, lane, 16, scratch_row;
          mov.b32 packed0, 0;
          mov.b32 packed1, 0;
          mov.b32 packed2, 0;
          mov.b32 packed3, 0;
          @row_live ld.shared.v4.b32 {packed0, packed1, packed2, packed3}, [scratch_row];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          mov.b32 {h4, h5}, packed2;
          mov.b32 {h6, h7}, packed3;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          cvt.f32.bf16 v4, h4;
          cvt.f32.bf16 v5, h5;
          cvt.f32.bf16 v6, h6;
          cvt.f32.bf16 v7, h7;
          mul.f32 v0, v0, norm;
          mul.f32 v1, v1, norm;
          mul.f32 v2, v2, norm;
          mul.f32 v3, v3, norm;
          mul.f32 v4, v4, norm;
          mul.f32 v5, v5, norm;
          mul.f32 v6, v6, norm;
          mul.f32 v7, v7, norm;
          mul.f32 v0, v0, f0;
          mul.f32 v1, v1, f1;
          mul.f32 v2, v2, f2;
          mul.f32 v3, v3, f3;
          mul.f32 v4, v4, f4;
          mul.f32 v5, v5, f5;
          mul.f32 v6, v6, f6;
          mul.f32 v7, v7, f7;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          cvt.rn.bf16x2.f32 packed2, v5, v4;
          cvt.rn.bf16x2.f32 packed3, v7, v6;
          mul.wide.u32 y_row, row, 8192;
          add.u64 y_row, $2, y_row;
          add.u64 address, y_row, main_offset;
          @row_live st.global.v4.b32 [address], {packed0, packed1, packed2, packed3};

          and.pred tail_live, wide, tail_lane;
          and.pred tail_live, tail_live, row_live;
          mad.lo.u32 scratch_tail, row, $9, $1;
          add.u32 scratch_tail, scratch_tail, $10;
          mad.lo.u32 scratch_tail, lane, 8, scratch_tail;
          add.u32 scratch_tail, scratch_tail, 512;
          mov.b32 packed0, 0;
          mov.b32 packed1, 0;
          @tail_live ld.shared.v2.b32 {packed0, packed1}, [scratch_tail];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          mul.f32 v0, v0, norm;
          mul.f32 v1, v1, norm;
          mul.f32 v2, v2, norm;
          mul.f32 v3, v3, norm;
          mul.f32 v0, v0, tf0;
          mul.f32 v1, v1, tf1;
          mul.f32 v2, v2, tf2;
          mul.f32 v3, v3, tf3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          add.u64 address, y_row, tail_offset;
          @tail_live st.global.v2.b32 [address], {packed0, packed1};

          add.u32 wave, wave, 1;
          setp.lt.u32 more, wave, 6;
          @more bra.uni DG_TLE_MHC_SPLIT_NORM8_WAVE;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,l,l,r,r,r,r,r,r,r",
        args=[
            scratch_address,
            y_address,
            weight_address,
            norm_address,
            physical_thread,
            hidden_quad_offset,
            hidden_quad_count,
            live_rows,
            scratch_stride_bytes,
            scratch_chunk_bytes,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _split_norms_ptx(
    square_tile_base,
    norm_shared,
    physical_thread,
    live_rows,
    rmsnorm_eps,
    rmsnorm_scale,
    num_splits,
):
    """Reduce fifteen split-local sums and publish 64 RMS scales."""
    square_address = tl.cast(square_tile_base, tl.uint64)
    norm_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(norm_shared, (0,))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active, more;
          .reg .u32 split, shared_address;
          .reg .f32 sum, value, norm;
          .reg .u64 address, offset;
          setp.lt.u32 active, $3, $4;
          mov.f32 sum, 0f00000000;
          mov.u32 split, 0;
        DG_TLE_MHC_SPLIT_NORM_REDUCE:
          mul.wide.u32 offset, split, 8192;
          add.u64 address, $1, offset;
          mul.wide.u32 offset, $3, 128;
          add.u64 address, address, offset;
          mov.f32 value, 0f00000000;
          @active ld.global.f32 value, [address];
          add.rn.f32 sum, sum, value;
          add.u32 split, split, 1;
          setp.lt.u32 more, split, $7;
          @more bra.uni DG_TLE_MHC_SPLIT_NORM_REDUCE;
          mul.f32 sum, sum, 0f39800000;
          add.rn.f32 sum, sum, $5;
          rsqrt.approx.f32 norm, sum;
          mul.f32 norm, norm, $6;
          mad.lo.u32 shared_address, $3, 4, $2;
          @active st.shared.f32 [shared_address], norm;
          barrier.cta.sync 14, 384;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,r,r,r,f,f,r",
        args=[
            square_address,
            norm_address,
            physical_thread,
            live_rows,
            rmsnorm_eps,
            rmsnorm_scale,
            num_splits,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _hidden_tail_no_wait(
    pre_shared,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Finish one H4096 row after its Pre values are already visible."""
    h = tl.arange(0, JIT_HIDDEN)
    norm_input = tl.zeros((JIT_HIDDEN,), tl.float32)
    for channel in tl.static_range(0, JIT_HC_MULT):
        value = tl.load(
            new_residual_ptr + channel * JIT_HIDDEN + h
        ).to(tl.float32)
        pre_scalar = tl.load(tle.gpu.local_ptr(pre_shared, (channel,)))
        norm_input += value * pre_scalar
    norm_input_bf16 = norm_input.to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=0)
        / JIT_HIDDEN
        + rmsnorm_eps
    )
    rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + h,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
    )


@triton.jit
def _hidden_tail_batch2_no_wait(
    pre_shared0,
    pre_shared1,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    live_rows,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Map the default eight warps to two independent H4096 rows."""
    batch = tl.arange(0, 2)[:, None]
    h = tl.arange(0, JIT_HIDDEN)[None, :]
    live_mask = batch < live_rows
    norm_input = tl.zeros((2, JIT_HIDDEN), tl.float32)
    for channel in tl.static_range(0, JIT_HC_MULT):
        value = tl.load(
            new_residual_ptr
            + batch * JIT_FLAT_HIDDEN
            + channel * JIT_HIDDEN
            + h,
            mask=live_mask,
            other=0.0,
        ).to(tl.float32)
        pre0 = tl.load(tle.gpu.local_ptr(pre_shared0, (channel,)))
        pre1 = tl.load(tle.gpu.local_ptr(pre_shared1, (channel,)))
        pre = tl.where(batch == 0, pre0, pre1)
        norm_input += value * pre
    norm_input_bf16 = norm_input.to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=1)
        / JIT_HIDDEN
        + rmsnorm_eps
    )
    rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms[:, None] * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + batch * JIT_HIDDEN + h,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=live_mask,
    )


@triton.jit
def _routing_finalize_no_barrier(
    pre_global_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    packed_sum,
    lane,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    """Normal-MHC routing body with its rendezvous owned by the caller."""
    projection_mask = lane < JIT_PROJECTION_OUTPUTS
    token_squared_sum = _t64._warp_broadcast_lane24_f32(packed_sum)
    projection_norm = tl.rsqrt(
        token_squared_sum / JIT_FLAT_HIDDEN + hc_norm_eps
    )
    category_scale = tl.where(
        lane < JIT_HC_MULT,
        tl.load(mix_scales_ptr),
        tl.where(
            lane < 2 * JIT_HC_MULT,
            tl.load(mix_scales_ptr + 1),
            tl.load(mix_scales_ptr + 2),
        ),
    )
    base = tl.load(
        mix_bases_ptr + lane,
        mask=projection_mask,
        other=0.0,
    ).to(tl.float32)
    mixes = tl.where(
        projection_mask,
        packed_sum * projection_norm * category_scale + base,
        0.0,
    )
    sigmoid = tl.sigmoid(mixes)
    pre_mix = sigmoid + hc_pre_eps
    post_mix = sigmoid * hc_post_scale
    pre_address = tl.cast(pre_global_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred p;
          .reg .u64 offset, address;
          setp.lt.u32 p, $3, 4;
          mul.wide.u32 offset, $3, 4;
          add.u64 address, $1, offset;
          @p st.global.f32 [address], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r",
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

    routing_mask = (
        lane >= 2 * JIT_HC_MULT
    ) & (lane < JIT_PROJECTION_OUTPUTS)
    sinkhorn = tl.where(routing_mask, mixes, 0.0)
    row_max = _t64._warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _t64._warp_reduce4_sum(sinkhorn)
    sinkhorn *= _t64._fast_rcp_ftz_f32(row_sum)
    sinkhorn += sinkhorn_eps
    column_sum = (
        _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
    )
    sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
    # A routing owner is one physical warp.  Keep the loop fully unrolled:
    # the pinned compiler otherwise lowers a dynamic loop backedge to a
    # four-warp worker-group rendezvous and deadlocks the bypassing warps.
    for _ in tl.static_range(1, JIT_SINKHORN_ITERS):
        row_sum = _t64._warp_reduce4_sum(sinkhorn) + sinkhorn_eps
        sinkhorn *= _t64._fast_rcp_ftz_f32(row_sum)
        column_sum = (
            _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
        )
        sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
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
def _p256_producer(
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
    partials_ptr,
    tile_states_ptr,
    q_fn_writer,
    tail_scratch,
    coeff_shared,
    winner_shared,
    task_shared,
    pre_shared,
    norm_shared,
    pre_ready,
    producer_ready,
    handoff_ready,
    task_ready,
    task_done,
    coeff_ready,
    tile,
    split,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
    ENABLE_TAIL: tl.constexpr,
    PARALLEL_TAIL: tl.constexpr,
    PROFILE_PHASES: tl.constexpr,
):
    """Eight producer warps stream 40 route-by-H32 stages."""
    physical_thread = _small._physical_thread_id_x()
    state_base = tl.cast(tile_states_ptr + tile * 32, tl.uint64)
    if PROFILE_PHASES:
        _profile_phase_timestamp(state_base, 24, physical_thread)
    m = tile * JIT_BLOCK_M + tl.arange(0, JIT_BLOCK_M)
    split_blocks = JIT_BASE_BLOCKS_PER_SPLIT + tl.where(
        split < JIT_WIDE_SPLITS,
        JIT_EXTRA_BLOCKS_PER_WIDE_SPLIT,
        0,
    )
    hidden_block_offset = (
        split * JIT_BASE_BLOCKS_PER_SPLIT
        + tl.minimum(split, JIT_WIDE_SPLITS)
        * JIT_EXTRA_BLOCKS_PER_WIDE_SPLIT
    )
    token_mask = m < JIT_TOKENS
    squared_sum = tl.zeros((JIT_BLOCK_M,), tl.float32)
    n = tl.arange(0, JIT_BLOCK_N)
    local_hidden = tl.arange(0, JIT_HIDDEN_BLOCK)

    # Post[4] and Comb[4x4] are invariant across all forty K32 stages.
    # Cache one padded 64x32 tile once instead of reloading 20 FP32 values
    # for every hidden block.
    local_m = tl.arange(0, JIT_BLOCK_M)
    coeff_column = tl.arange(0, JIT_BLOCK_N)
    coeff_token = tile * JIT_BLOCK_M + local_m
    post_value = tl.load(
        post_mix_ptr
        + coeff_token[:, None] * JIT_HC_MULT
        + coeff_column[None, :],
        mask=(coeff_token[:, None] < JIT_TOKENS)
        & (coeff_column[None, :] < JIT_HC_MULT),
        other=0.0,
    ).to(tl.float32)
    comb_column = coeff_column - JIT_HC_MULT
    comb_value = tl.load(
        comb_res_mix_ptr
        + coeff_token[:, None] * JIT_HC_MULT * JIT_HC_MULT
        + comb_column[None, :],
        mask=(coeff_token[:, None] < JIT_TOKENS)
        & (comb_column[None, :] >= 0)
        & (comb_column[None, :] < JIT_HC_MULT * JIT_HC_MULT),
        other=0.0,
    ).to(tl.float32)
    coeff_value = tl.where(
        coeff_column[None, :] < JIT_HC_MULT,
        post_value,
        comb_value,
    )
    tl.store(tle.gpu.local_ptr(coeff_shared), coeff_value)
    tle.gpu.barrier_wait(coeff_ready)

    # Preserve CUDA's 320/256-hidden split boundaries using K32: the first
    # four splits own ten blocks and the remaining eleven own eight.  The
    # fixed ten-by-four schedule publishes zero stages for inactive blocks.
    for block_in_split in tl.static_range(0, JIT_MAX_BLOCKS_PER_SPLIT):
        active_block = block_in_split < split_blocks
        hidden = (
            hidden_block_offset + block_in_split
        ) * JIT_HIDDEN_BLOCK + local_hidden
        matrix_mask = token_mask[:, None] & active_block
        token_hidden = m[:, None] * JIT_HIDDEN + hidden[None, :]
        token_residual = m[:, None] * JIT_FLAT_HIDDEN + hidden[None, :]
        x = tl.load(
            x_ptr + token_hidden, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        r0 = tl.load(
            residual_ptr + token_residual, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        r1 = tl.load(
            residual_ptr + token_residual + JIT_HIDDEN,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        r2 = tl.load(
            residual_ptr + token_residual + 2 * JIT_HIDDEN,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        r3 = tl.load(
            residual_ptr + token_residual + 3 * JIT_HIDDEN,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        for channel in tl.static_range(0, JIT_HC_MULT):
            stage_index = block_in_split * JIT_HC_MULT + channel
            coeff_channel = tl.full(
                (JIT_BLOCK_M,), channel, tl.int32
            )
            post = tl.load(
                tle.gpu.local_ptr(
                    coeff_shared, (local_m, coeff_channel)
                )
            )[:, None]
            q = (
                x * post
                + r0
                * tl.load(
                    tle.gpu.local_ptr(
                        coeff_shared,
                        (
                            local_m,
                            coeff_channel + JIT_HC_MULT,
                        ),
                    )
                )[:, None]
                + r1
                * tl.load(
                    tle.gpu.local_ptr(
                        coeff_shared,
                        (
                            local_m,
                            coeff_channel + 2 * JIT_HC_MULT,
                        ),
                    )
                )[:, None]
                + r2
                * tl.load(
                    tle.gpu.local_ptr(
                        coeff_shared,
                        (
                            local_m,
                            coeff_channel + 3 * JIT_HC_MULT,
                        ),
                    )
                )[:, None]
                + r3
                * tl.load(
                    tle.gpu.local_ptr(
                        coeff_shared,
                        (
                            local_m,
                            coeff_channel + 4 * JIT_HC_MULT,
                        ),
                    )
                )[:, None]
            )
            q = tl.where(matrix_mask, q, 0.0)
            q_bf16 = q.to(tl.bfloat16, fp_downcast_rounding="rtne")
            q_fp32 = q_bf16.to(tl.float32)
            flat_hidden = channel * JIT_HIDDEN + hidden
            tl.store(
                new_residual_ptr
                + m[:, None] * JIT_FLAT_HIDDEN
                + flat_hidden[None, :],
                q_bf16,
                mask=matrix_mask,
            )

            fn_offsets = (
                n[:, None] * JIT_FLAT_HIDDEN + flat_hidden[None, :]
            )
            fn_tile = tl.load(
                fn_ptr + fn_offsets,
                mask=(n[:, None] < JIT_PROJECTION_OUTPUTS)
                & active_block,
                other=0.0,
            ).to(tl.float32)
            squared_sum += tl.sum(q_fp32 * q_fp32, axis=1)
            slot = q_fn_writer.acquire(stage_index)
            tl.store(tle.gpu.local_ptr(slot.q), q_fp32)
            tl.store(tle.gpu.local_ptr(slot.fn), fn_tile)
            q_fn_writer.commit(stage_index)

    sq_offset = (
        ((tile * JIT_NUM_SPLITS + split) * JIT_BLOCK_M)
        + tl.arange(0, JIT_BLOCK_M)
    ) * JIT_BLOCK_N + JIT_PROJECTION_OUTPUTS
    tl.store(partials_ptr + sq_offset, squared_sum, mask=token_mask)
    if PROFILE_PHASES:
        _profile_phase_timestamp(state_base, 32, physical_thread)

    if ENABLE_TAIL:
        # Phase 1: all fifteen split CTAs have published projection and Post.
        tle.gpu.barrier_wait(producer_ready)
        _tile_phase_wait(
            state_base, 0, physical_thread,
            tl.full((), JIT_NUM_SPLITS - 1, tl.uint32),
        )
        tle.gpu.barrier_wait(handoff_ready)
        if PROFILE_PHASES:
            _profile_phase_timestamp(state_base, 40, physical_thread)
        # Fixed routing ownership: split i owns rows 4i..4i+3; splits 0..3
        # additionally own rows 60..63.  Five producer warps are sufficient.
        producer_warp = physical_thread // 32
        lane = physical_thread % 32
        if JIT_NUM_SPLITS == 8:
            route_row = split * 8 + producer_warp
            route_active = producer_warp < 8
        else:
            extra_route = (split < 4) & (producer_warp == 4)
            route_row = tl.where(
                extra_route, 60 + split, split * 4 + producer_warp
            )
            route_active = (producer_warp < 4) | extra_route
        token = tile * JIT_BLOCK_M + route_row
        route_active = route_active & (token < JIT_TOKENS)
        if route_active:
            packed_sum = 0.0
            for source_split in tl.static_range(0, JIT_NUM_SPLITS):
                source_offset = (
                    ((tile * JIT_NUM_SPLITS + source_split)
                     * JIT_BLOCK_M + route_row)
                    * JIT_BLOCK_N + lane
                )
                packed_sum += tl.load(
                    partials_ptr + source_offset,
                    mask=lane <= JIT_PROJECTION_OUTPUTS,
                    other=0.0,
                )
            pre_pointer = (
                partials_ptr
                + ((tile * JIT_NUM_SPLITS) * JIT_BLOCK_M + route_row)
                * JIT_BLOCK_N
                + 25
            )
            _routing_finalize_no_barrier(
                pre_pointer,
                mix_scales_ptr,
                mix_bases_ptr,
                new_post_mix_ptr + token * JIT_HC_MULT,
                new_comb_res_mix_ptr
                + token * JIT_HC_MULT * JIT_HC_MULT,
                packed_sum,
                lane,
                hc_norm_eps,
                hc_pre_eps,
                hc_post_scale,
                sinkhorn_eps,
            )

        tle.gpu.barrier_wait(task_ready)
        if PROFILE_PHASES:
            _profile_phase_timestamp(state_base, 48, physical_thread)
        _tile_phase_wait(
            state_base, 8, physical_thread,
            tl.full((), JIT_NUM_SPLITS - 1, tl.uint32),
        )
        tle.gpu.barrier_wait(task_done)
        if PROFILE_PHASES:
            _profile_phase_timestamp(state_base, 56, physical_thread)

        live_rows = tl.minimum(
            JIT_BLOCK_M, JIT_TOKENS - tile * JIT_BLOCK_M
        )
        if JIT_NUM_SPLITS == 8:
            hidden_offset = split * 512
            hidden_width = 512
        else:
            hidden_offset = tl.where(
                split < 4, split * 320, 1280 + (split - 4) * 256
            )
            hidden_width = tl.where(split < 4, 320, 256)
        hidden_quad_offset = hidden_offset // 4
        hidden_quad_count = hidden_width // 4
        pre_base = (
            partials_ptr
            + (tile * JIT_NUM_SPLITS * JIT_BLOCK_M) * JIT_BLOCK_N
            + 25
        )
        residual_base = (
            new_residual_ptr
            + tile * JIT_BLOCK_M * JIT_FLAT_HIDDEN
        )
        y_base = y_bf16_ptr + tile * JIT_BLOCK_M * JIT_HIDDEN
        square_base = (
            partials_ptr
            + ((tile * JIT_NUM_SPLITS + split) * JIT_BLOCK_M)
            * JIT_BLOCK_N
            + JIT_PROJECTION_OUTPUTS
        )
        if JIT_NUM_SPLITS == 8:
            _split_hidden_combine_bf16x8_ptx(
                pre_base, residual_base, tail_scratch, square_base,
                physical_thread, hidden_quad_offset, 64, live_rows,
                1024, 0, 0,
            )
            _split_hidden_combine_bf16x8_ptx(
                pre_base, residual_base, tail_scratch, square_base,
                physical_thread, hidden_quad_offset + 64, 64, live_rows,
                1024, 512, 1,
            )
        else:
            _split_hidden_combine_bf16x8_ptx(
                pre_base, residual_base, tail_scratch, square_base,
                physical_thread, hidden_quad_offset, hidden_quad_count,
                live_rows, 640, 0, 0,
            )
        tle.gpu.barrier_wait(pre_ready)
        if PROFILE_PHASES:
            _profile_phase_timestamp(state_base, 64, physical_thread)
        _tile_phase_wait(
            state_base, 16, physical_thread,
            tl.full((), JIT_NUM_SPLITS - 1, tl.uint32),
        )
        tle.gpu.barrier_wait(handoff_ready)
        if PROFILE_PHASES:
            _profile_phase_timestamp(state_base, 72, physical_thread)

        square_tile_base = (
            partials_ptr
            + (tile * JIT_NUM_SPLITS * JIT_BLOCK_M) * JIT_BLOCK_N
            + JIT_PROJECTION_OUTPUTS
        )
        _split_norms_ptx(
            square_tile_base,
            norm_shared,
            physical_thread,
            live_rows,
            rmsnorm_eps,
            rmsnorm_scale,
            tl.full((), JIT_NUM_SPLITS, tl.uint32),
        )
        if JIT_NUM_SPLITS == 8:
            _split_hidden_normalize_bf16x8_ptx(
                tail_scratch, y_base, rmsnorm_weight_ptr, norm_shared,
                physical_thread, hidden_quad_offset, 64, live_rows,
                1024, 0,
            )
            _split_hidden_normalize_bf16x8_ptx(
                tail_scratch, y_base, rmsnorm_weight_ptr, norm_shared,
                physical_thread, hidden_quad_offset + 64, 64, live_rows,
                1024, 512,
            )
        else:
            _split_hidden_normalize_bf16x8_ptx(
                tail_scratch, y_base, rmsnorm_weight_ptr, norm_shared,
                physical_thread, hidden_quad_offset, hidden_quad_count,
                live_rows, 640, 0,
            )
        if PROFILE_PHASES:
            _profile_phase_timestamp(state_base, 80, physical_thread)


@triton.jit
def _p256_math_and_routing(
    partials_ptr,
    rmsnorm_weight_ptr,
    new_residual_ptr,
    y_bf16_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    q_fn_reader,
    tail_scratch,
    route_shared,
    winner_shared,
    task_shared,
    pre_shared,
    norm_shared,
    pre_ready,
    producer_ready,
    handoff_ready,
    task_ready,
    task_done,
    tile,
    split,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
    ENABLE_TAIL: tl.constexpr,
    PARALLEL_TAIL: tl.constexpr,
):
    """Four WGMMA warps produce partials and the winner routes its tile."""
    accumulator = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), tl.float32)
    for stage_index in tl.static_range(
        0, JIT_MAX_BLOCKS_PER_SPLIT * JIT_HC_MULT
    ):
        wait = q_fn_reader.wait(stage_index)
        accumulator = tle.gpu.wgmma(
            wait.slot.q,
            wait.slot.fn,
            accumulator,
            input_precision="tf32",
            trans_b=True,
        )
        if stage_index > 0:
            accumulator = tle.gpu.wgmma_wait(1, accumulator)
            q_fn_reader.release(stage_index - 1)
    accumulator = tle.gpu.wgmma_wait(0, accumulator)
    q_fn_reader.release(JIT_MAX_BLOCKS_PER_SPLIT * JIT_HC_MULT - 1)

    m = tile * JIT_BLOCK_M + tl.arange(0, JIT_BLOCK_M)
    n = tl.arange(0, JIT_BLOCK_N)
    partial_offset = (
        ((tile * JIT_NUM_SPLITS + split) * JIT_BLOCK_M)
        + tl.arange(0, JIT_BLOCK_M)[:, None]
    ) * JIT_BLOCK_N + n[None, :]
    tl.store(
        partials_ptr + partial_offset,
        accumulator,
        mask=(m[:, None] < JIT_TOKENS)
        & (n[None, :] < JIT_PROJECTION_OUTPUTS),
    )

    if ENABLE_TAIL:
        tle.gpu.barrier_wait(producer_ready)
        tle.gpu.barrier_wait(handoff_ready)
        tle.gpu.barrier_wait(task_ready)
        tle.gpu.barrier_wait(task_done)
        physical_thread = _small._physical_thread_id_x()
        live_rows = tl.minimum(
            JIT_BLOCK_M, JIT_TOKENS - tile * JIT_BLOCK_M
        )
        if JIT_NUM_SPLITS == 8:
            hidden_offset = split * 512
            hidden_width = 512
        else:
            hidden_offset = tl.where(
                split < 4, split * 320, 1280 + (split - 4) * 256
            )
            hidden_width = tl.where(split < 4, 320, 256)
        hidden_quad_offset = hidden_offset // 4
        hidden_quad_count = hidden_width // 4
        pre_base = (
            partials_ptr
            + (tile * JIT_NUM_SPLITS * JIT_BLOCK_M) * JIT_BLOCK_N
            + 25
        )
        residual_base = (
            new_residual_ptr
            + tile * JIT_BLOCK_M * JIT_FLAT_HIDDEN
        )
        y_base = y_bf16_ptr + tile * JIT_BLOCK_M * JIT_HIDDEN
        square_base = (
            partials_ptr
            + ((tile * JIT_NUM_SPLITS + split) * JIT_BLOCK_M)
            * JIT_BLOCK_N
            + JIT_PROJECTION_OUTPUTS
        )
        if JIT_NUM_SPLITS == 8:
            _split_hidden_combine_bf16x8_ptx(
                pre_base, residual_base, tail_scratch, square_base,
                physical_thread, hidden_quad_offset, 64, live_rows,
                1024, 0, 0,
            )
            _split_hidden_combine_bf16x8_ptx(
                pre_base, residual_base, tail_scratch, square_base,
                physical_thread, hidden_quad_offset + 64, 64, live_rows,
                1024, 512, 1,
            )
        else:
            _split_hidden_combine_bf16x8_ptx(
                pre_base, residual_base, tail_scratch, square_base,
                physical_thread, hidden_quad_offset, hidden_quad_count,
                live_rows, 640, 0, 0,
            )
        tle.gpu.barrier_wait(pre_ready)
        tle.gpu.barrier_wait(handoff_ready)
        square_tile_base = (
            partials_ptr
            + (tile * JIT_NUM_SPLITS * JIT_BLOCK_M) * JIT_BLOCK_N
            + JIT_PROJECTION_OUTPUTS
        )
        _split_norms_ptx(
            square_tile_base,
            norm_shared,
            physical_thread,
            live_rows,
            rmsnorm_eps,
            rmsnorm_scale,
            tl.full((), JIT_NUM_SPLITS, tl.uint32),
        )
        if JIT_NUM_SPLITS == 8:
            _split_hidden_normalize_bf16x8_ptx(
                tail_scratch, y_base, rmsnorm_weight_ptr, norm_shared,
                physical_thread, hidden_quad_offset, 64, live_rows,
                1024, 0,
            )
            _split_hidden_normalize_bf16x8_ptx(
                tail_scratch, y_base, rmsnorm_weight_ptr, norm_shared,
                physical_thread, hidden_quad_offset + 64, 64, live_rows,
                1024, 512,
            )
        else:
            _split_hidden_normalize_bf16x8_ptx(
                tail_scratch, y_base, rmsnorm_weight_ptr, norm_shared,
                physical_thread, hidden_quad_offset, hidden_quad_count,
                live_rows, 640, 0,
            )


@triton.jit
def _sm90_eager_t1028_p256_wgmma_kernel(
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
    partials_ptr,
    tile_states_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
    ENABLE_TAIL: tl.constexpr,
    PARALLEL_TAIL: tl.constexpr,
    PROFILE_PHASES: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid // JIT_NUM_SPLITS
    split = pid % JIT_NUM_SPLITS
    tl.static_assert(
        (JIT_NUM_M_TILES == 17) | (JIT_NUM_M_TILES == 32)
    )
    tl.static_assert(
        (JIT_NUM_SPLITS == 15) | (JIT_NUM_SPLITS == 8)
    )
    tl.static_assert(JIT_GRID_CTAS == JIT_NUM_M_TILES * JIT_NUM_SPLITS)
    tl.static_assert(JIT_HIDDEN_BLOCK == 32)

    q_shared = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_BLOCK_M, JIT_HIDDEN_BLOCK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    fn_shared = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_BLOCK_N, JIT_HIDDEN_BLOCK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    coeff_shared = tle.gpu.alloc(
        [JIT_BLOCK_M, JIT_BLOCK_N],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    route_shared = tle.gpu.alloc(
        [JIT_MATH_WARPS, JIT_BLOCK_N],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    winner_shared = tle.gpu.alloc(
        [1], dtype=tl.uint32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    task_shared = tle.gpu.alloc(
        [1], dtype=tl.uint32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pre_shared = tle.gpu.alloc(
        [4, JIT_HC_MULT], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    norm_shared = tle.gpu.alloc(
        [JIT_BLOCK_M], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    q_fn_pipe = tle.pipe(
        capacity=JIT_PIPE_STAGES,
        name="eager_t1028_p256_q_fn",
        q=q_shared,
        fn=fn_shared,
    )
    producer_ready = tle.gpu.alloc_barrier(arrive_count=JIT_LOGICAL_THREADS)
    handoff_ready = tle.gpu.alloc_barrier(arrive_count=JIT_LOGICAL_THREADS)
    task_ready = tle.gpu.alloc_barrier(arrive_count=JIT_LOGICAL_THREADS)
    task_done = tle.gpu.alloc_barrier(arrive_count=JIT_LOGICAL_THREADS)
    coeff_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_PRODUCER_THREADS
    )
    pre_ready = tle.gpu.alloc_barrier(arrive_count=JIT_LOGICAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _p256_producer,
                (
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
                    partials_ptr,
                    tile_states_ptr,
                    q_fn_pipe.writer(),
                    q_shared,
                    coeff_shared,
                    winner_shared,
                    task_shared,
                    pre_shared,
                    norm_shared,
                    pre_ready,
                    producer_ready,
                    handoff_ready,
                    task_ready,
                    task_done,
                    coeff_ready,
                    tile,
                    split,
                    hc_norm_eps,
                    hc_pre_eps,
                    hc_post_scale,
                    sinkhorn_eps,
                    rmsnorm_eps,
                    rmsnorm_scale,
                    ENABLE_TAIL,
                    PARALLEL_TAIL,
                    PROFILE_PHASES,
                ),
            ),
            (
                _p256_math_and_routing,
                (
                    partials_ptr,
                    rmsnorm_weight_ptr,
                    new_residual_ptr,
                    y_bf16_ptr,
                    mix_scales_ptr,
                    mix_bases_ptr,
                    new_post_mix_ptr,
                    new_comb_res_mix_ptr,
                    q_fn_pipe.reader(),
                    q_shared,
                    route_shared,
                    winner_shared,
                    task_shared,
                    pre_shared,
                    norm_shared,
                    pre_ready,
                    producer_ready,
                    handoff_ready,
                    task_ready,
                    task_done,
                    tile,
                    split,
                    hc_norm_eps,
                    hc_pre_eps,
                    hc_post_scale,
                    sinkhorn_eps,
                    rmsnorm_eps,
                    rmsnorm_scale,
                    ENABLE_TAIL,
                    PARALLEL_TAIL,
                ),
            ),
        ],
        [JIT_MATH_WARPS],
        [JIT_MATH_REGS],
    )


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name}.shape={tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name}.dtype={value.dtype}, expected {dtype}")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


@dataclass(frozen=True)
class EagerT1028P256Plan:
    """Prepared exact-one-launch plan; ordered reuse on one stream only."""

    tensors: tuple[torch.Tensor, ...]
    partials: torch.Tensor
    tile_states: torch.Tensor
    scalars: tuple[float, ...]
    workspace_tensors: dict[str, torch.Tensor]
    counter_tensors: dict[str, torch.Tensor]
    metadata: dict[str, object]
    maxnreg: int
    tail_enabled: bool
    parallel_tail: bool
    profile_phases: bool

    @property
    def grid_ctas(self) -> int:
        return GRID_CTAS

    def _kernel_args(self) -> tuple[object, ...]:
        return (
            *self.tensors,
            self.partials,
            self.tile_states,
            *self.scalars,
            self.tail_enabled,
            self.parallel_tail,
            self.profile_phases,
        )

    def compile(self):
        """Compile and load the CUfunction without launching it."""
        return _sm90_eager_t1028_p256_wgmma_kernel.warmup(
            *self._kernel_args(),
            grid=(GRID_CTAS,),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            maxnreg=self.maxnreg,
        )

    def run(self) -> None:
        """Enqueue exactly one non-cooperative TLE kernel."""
        _sm90_eager_t1028_p256_wgmma_kernel[(GRID_CTAS,)](
            *self._kernel_args(),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            maxnreg=self.maxnreg,
        )


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT1028P256Plan:
    """Validate the pinned profile and allocate stable eager workspace."""
    if os.environ.get("DG_TLE_MHC_EAGER_T1028_P256", "0") != "1":
        raise RuntimeError(
            "set DG_TLE_MHC_EAGER_T1028_P256=1 to enable this experiment"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the eager p256 plan outside capture")
    if inputs.get("shifted_prev_mix") is not None:
        raise ValueError("only Normal MHC is supported")
    if outputs.get("new_prev_mix") is not None:
        raise ValueError("Normal MHC requires new_prev_mix=None")

    specs = (
        (inputs.get("x"), "inputs.x", (TOKENS, HIDDEN), torch.bfloat16),
        (
            inputs.get("residual"), "inputs.residual",
            (TOKENS, HC_MULT, HIDDEN), torch.bfloat16,
        ),
        (
            inputs.get("post_mix"), "inputs.post_mix",
            (TOKENS, HC_MULT, 1), torch.float32,
        ),
        (
            inputs.get("comb_res_mix"), "inputs.comb_res_mix",
            (TOKENS, HC_MULT, HC_MULT), torch.float32,
        ),
        (
            inputs.get("fn"), "inputs.fn",
            (PROJECTION_OUTPUTS, FLAT_HIDDEN), torch.float32,
        ),
        (inputs.get("mix_scales"), "inputs.mix_scales", (3,), torch.float32),
        (
            inputs.get("mix_bases"), "inputs.mix_bases",
            (PROJECTION_OUTPUTS,), torch.float32,
        ),
        (
            inputs.get("rmsnorm_weight"), "inputs.rmsnorm_weight",
            (HIDDEN,), torch.bfloat16,
        ),
        (
            outputs.get("new_residual"), "outputs.new_residual",
            (TOKENS, HC_MULT, HIDDEN), torch.bfloat16,
        ),
        (
            outputs.get("new_post_mix"), "outputs.new_post_mix",
            (TOKENS, HC_MULT, 1), torch.float32,
        ),
        (
            outputs.get("new_comb_res_mix"), "outputs.new_comb_res_mix",
            (TOKENS, HC_MULT, HC_MULT), torch.float32,
        ),
        (
            outputs.get("y_bf16"), "outputs.y_bf16",
            (TOKENS, HIDDEN), torch.bfloat16,
        ),
    )
    tensors = tuple(
        _require_tensor(value, name=name, shape=shape, dtype=dtype)
        for value, name, shape, dtype in specs
    )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must be on one CUDA device")

    expected_scalars = {
        "hc_mult": HC_MULT,
        "hc_norm_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_post_scale": 2.0,
        "sinkhorn_eps": 1e-6,
        "num_sinkhorn_iters": SINKHORN_ITERS,
        "rmsnorm_eps": 1e-6,
        "rmsnorm_scale": 1.0,
    }
    for name, expected in expected_scalars.items():
        if inputs.get(name) != expected:
            raise ValueError(
                f"inputs.{name}={inputs.get(name)!r}, expected {expected!r}"
            )

    device = tensors[0].device
    maxnreg = int(
        os.environ.get("DG_TLE_MHC_EAGER_T1028_MAXNREG", str(MATH_REGS))
    )
    if maxnreg not in (80, 84, 96, 112):
        raise ValueError(
            "DG_TLE_MHC_EAGER_T1028_MAXNREG must be 80/84/96/112"
        )
    tail_enabled = (
        os.environ.get("DG_TLE_MHC_EAGER_T1028_PRODUCER_ONLY", "0") != "1"
    )
    parallel_tail = (
        os.environ.get("DG_TLE_MHC_EAGER_T1028_PARALLEL_TAIL", "1") == "1"
    )
    profile_phases = (
        os.environ.get("DG_TLE_MHC_EAGER_T1028_PROFILE_PHASES", "0") == "1"
    )
    partials = torch.empty(
        (NUM_M_TILES, NUM_SPLITS, BLOCK_M, BLOCK_N),
        dtype=torch.float32,
        device=device,
    )
    tile_states = torch.zeros(
        (NUM_M_TILES, 32), dtype=torch.uint32, device=device
    )
    metadata: dict[str, object] = {
        "model": "DeepSeek-V4-Flash@60d8d707",
        "execution_mode": "eager",
        "mode": "normal",
        "T": TOKENS,
        "H": HIDDEN,
        "R": HC_MULT,
        "P": PROJECTION_OUTPUTS,
        "kernel_count": 1,
        "expected_cuda_kernel_activities": 1,
        "kernel_symbol": "_sm90_eager_t1028_p256_wgmma_kernel",
        "CTA": GRID_CTAS,
        "grid_ctas": GRID_CTAS,
        "block_m": BLOCK_M,
        "block_n": BLOCK_N,
        "block_k": HIDDEN_BLOCK,
        "num_m_tiles": NUM_M_TILES,
        "num_splits": NUM_SPLITS,
        "split_hidden_blocks": list(SPLIT_HIDDEN_BLOCKS),
        "split_hidden_widths": [
            blocks * HIDDEN_BLOCK for blocks in SPLIT_HIDDEN_BLOCKS
        ],
        "active_route_stages_per_split": [
            blocks * HC_MULT for blocks in SPLIT_HIDDEN_BLOCKS
        ],
        "fixed_route_stages_per_cta": MAX_BLOCKS_PER_SPLIT * HC_MULT,
        "cta_mapping": (
            f"tile=pid//{NUM_SPLITS},split=pid%{NUM_SPLITS}"
        ),
        "producer_warps": NUM_WARPS,
        "math_warps": MATH_WARPS,
        "physical_cta_threads": (NUM_WARPS + MATH_WARPS) * 32,
        "target_registers_per_thread": maxnreg,
        "producer_pipeline_stages": PIPE_STAGES,
        "producer_coeff_cache_bytes": BLOCK_M * BLOCK_N * 4,
        "wgmma_wait_policy": "wait1_then_final_wait0",
        "synchronization": "per_tile_three_phase_arrival_barriers",
        "counter_policy": "three_arrival_phase_pairs_128b_state",
        "full_grid_barrier": False,
        "full_grid_polling": False,
        "cooperative_launch": False,
        "full_grid_residency_required": True,
        "bounded_helper_queue_migrated": False,
        "bounded_helper_polls_target": BOUNDED_HELPER_POLLS_TARGET,
        "tail_schedule": "split_sharded_three_phase",
        "tail_hidden_workers": "all_15_or_8_split_ctas",
        "tail_routing_workers": "fixed_rows_across_producer_warps",
        "tail_enabled": tail_enabled,
        "parallel_tail": parallel_tail,
        "profile_phases": profile_phases,
        "tile_state_shape": list(tile_states.shape),
        "tile_state_bytes": tile_states.numel() * tile_states.element_size(),
        "workspace_shape": list(partials.shape),
        "workspace_bytes": partials.numel() * partials.element_size(),
        "experimental": True,
        "dispatch_enabled": False,
    }
    # Kernel pointer order is inputs[0:8], then outputs[0:4].
    return EagerT1028P256Plan(
        tensors=tensors,
        partials=partials,
        tile_states=tile_states,
        scalars=(
            float(inputs["hc_norm_eps"]),
            float(inputs["hc_pre_eps"]),
            float(inputs["hc_post_scale"]),
            float(inputs["sinkhorn_eps"]),
            float(inputs["rmsnorm_eps"]),
            float(inputs["rmsnorm_scale"]),
        ),
        workspace_tensors={"projection_and_sqsum_partials": partials},
        counter_tensors={"tile_bounded_helper_state": tile_states},
        metadata=metadata,
        maxnreg=maxnreg,
        tail_enabled=tail_enabled,
        parallel_tail=parallel_tail,
        profile_phases=profile_phases,
    )


__all__ = [
    "EagerT1028P256Plan",
    "GRID_CTAS",
    "NUM_M_TILES",
    "NUM_SPLITS",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
]

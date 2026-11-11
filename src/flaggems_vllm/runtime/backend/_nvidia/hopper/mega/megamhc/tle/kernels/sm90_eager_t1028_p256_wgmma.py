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

The default tail advances the per-tile queue head by three.  A compile-time
research switch also permits two-row claims, mapping the default eight warps
onto two H4096 rows while the math four warps only finalize routing.  This
isolates the instruction-cache cost of the separate one-row math hidden body.
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
from . import sm90_eager_quad_layout_probe as _quad


TOKENS = 1028
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = HIDDEN * HC_MULT
SINKHORN_ITERS = 20
BLOCK_M = 64
BLOCK_N = 32
HIDDEN_BLOCK = 64
WGMMA_K_FRAGMENT = int(
    os.environ.get("DG_TLE_MHC_EAGER_T1028_WGMMA_K_FRAGMENT", "32")
)
if WGMMA_K_FRAGMENT not in (8, 16, 32):
    raise ValueError(
        "DG_TLE_MHC_EAGER_T1028_WGMMA_K_FRAGMENT must be 8/16/32"
    )
WGMMA_K_FRAGMENTS = HIDDEN_BLOCK // WGMMA_K_FRAGMENT
NUM_HIDDEN_BLOCKS = HIDDEN // HIDDEN_BLOCK
NUM_SPLITS = 15
BASE_BLOCKS_PER_SPLIT = NUM_HIDDEN_BLOCKS // NUM_SPLITS
WIDE_SPLITS = 4
EXTRA_BLOCKS_PER_WIDE_SPLIT = 1
MAX_BLOCKS_PER_SPLIT = (
    BASE_BLOCKS_PER_SPLIT + EXTRA_BLOCKS_PER_WIDE_SPLIT
)
SPLIT_HIDDEN_BLOCKS = (
    5, 5, 5, 5,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
)
NUM_M_TILES = (TOKENS + BLOCK_M - 1) // BLOCK_M
GRID_CTAS = NUM_M_TILES * NUM_SPLITS
NUM_WARPS = 8
MATH_WARPS = 4
MATH_REGS = int(
    os.environ.get("DG_TLE_MHC_EAGER_T1028_MATH_REGS", "80")
)
if MATH_REGS not in (80, 96, 112, 128, 144, 160):
    raise ValueError(
        "DG_TLE_MHC_EAGER_T1028_MATH_REGS must be "
        "80/96/112/128/144/160"
    )
PIPE_STAGES = 4
BOUNDED_HELPER_POLLS_TARGET = 256
TAIL_BATCH = int(os.environ.get("DG_TLE_MHC_EAGER_T1028_TAIL_BATCH", "3"))
if TAIL_BATCH not in (2, 3):
    raise ValueError("DG_TLE_MHC_EAGER_T1028_TAIL_BATCH must be 2 or 3")
USE_RMS_WEIGHT_CACHE = (
    os.environ.get("DG_TLE_MHC_EAGER_T1028_RMS_WEIGHT_CACHE", "0") == "1"
)

JIT_TOKENS = tl.constexpr(TOKENS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_HIDDEN_BLOCK = tl.constexpr(HIDDEN_BLOCK)
JIT_WGMMA_K_FRAGMENT = tl.constexpr(WGMMA_K_FRAGMENT)
JIT_WGMMA_K_FRAGMENTS = tl.constexpr(WGMMA_K_FRAGMENTS)
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
JIT_TAIL_BATCH = tl.constexpr(TAIL_BATCH)
JIT_USE_RMS_WEIGHT_CACHE = tl.constexpr(USE_RMS_WEIGHT_CACHE)


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
def _claim_tile_row(state_base, physical_thread, batch_rows):
    """Claim a compile-time-sized row batch; only thread zero acts."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader;
          .reg .u64 head_address;
          setp.eq.u32 leader, $2, 0;
          mov.u32 $0, 4294967295;
          add.u64 head_address, $1, 8;
          @leader atom.global.add.u32 $0, [head_address], $3;
        }
        """,
        constraints="=r,l,r,r",
        args=[state_base, physical_thread, batch_rows],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _sinkhorn_nineteen_warp_iterations(value, epsilon):
    """Run the repeated 4x4 Sinkhorn body as a warp-local PTX loop."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred again;
          .reg .u32 iteration;
          .reg .f32 x, t, total, inverse;
          mov.f32 x, $1;
          mov.u32 iteration, 19;
        DG_TLE_MHC_SINKHORN_LOOP:
          mov.f32 total, x;
          shfl.sync.bfly.b32 t, total, 2, 31, 0xffffffff;
          add.rn.f32 total, total, t;
          shfl.sync.bfly.b32 t, total, 1, 31, 0xffffffff;
          add.rn.f32 total, total, t;
          add.rn.f32 total, total, $2;
          rcp.approx.ftz.f32 inverse, total;
          mul.rn.f32 x, x, inverse;
          mov.f32 total, x;
          shfl.sync.bfly.b32 t, total, 4, 31, 0xffffffff;
          add.rn.f32 total, total, t;
          shfl.sync.bfly.b32 t, total, 24, 31, 0xffffffff;
          add.rn.f32 total, total, t;
          add.rn.f32 total, total, $2;
          rcp.approx.ftz.f32 inverse, total;
          mul.rn.f32 x, x, inverse;
          add.u32 iteration, iteration, 0xffffffff;
          setp.ne.u32 again, iteration, 0;
          @again bra.uni DG_TLE_MHC_SINKHORN_LOOP;
          mov.f32 $0, x;
        }
        """,
        constraints="=f,f,f",
        args=[value, epsilon],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _hidden_tail_no_wait(
    pre_shared,
    rms_weight_shared,
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
    if JIT_USE_RMS_WEIGHT_CACHE:
        rms_weight = tl.load(
            tle.gpu.local_ptr(rms_weight_shared, (h,))
        ).to(tl.float32)
    else:
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
    rms_weight_shared,
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
    if JIT_USE_RMS_WEIGHT_CACHE:
        rms_weight = tl.load(
            tle.gpu.local_ptr(rms_weight_shared, (h,))
        ).to(tl.float32)
    else:
        rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms[:, None] * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + batch * JIT_HIDDEN + h,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=live_mask,
    )


@triton.jit
def _routing_finalize_no_barrier(
    pre_shared,
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
    # A Triton dynamic loop inserts a partition rendezvous and deadlocks the
    # bypassing sibling warps.  The PTX loop is explicitly warp-local and
    # keeps one compact copy instead of unrolling nineteen iterations.
    sinkhorn = _sinkhorn_nineteen_warp_iterations(
        sinkhorn, sinkhorn_eps
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
def _load_global_bf16x4_masked(address, valid):
    """Load one packed BF16 quad, returning zeros for an invalid token."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred p;
          .reg .b32 packed0, packed1;
          .reg .b16 h0, h1, h2, h3;
          mov.b32 packed0, 0;
          mov.b32 packed1, 0;
          setp.ne.u32 p, $5, 0;
          @p ld.global.v2.b32 {packed0, packed1}, [$4];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 $0, h0;
          cvt.f32.bf16 $1, h1;
          cvt.f32.bf16 $2, h2;
          cvt.f32.bf16 $3, h3;
        }
        """,
        constraints="=&f,=&f,=&f,=&f,l,r",
        args=[address, tl.cast(valid, tl.uint32)],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _serial_quad_post_ptx(
    x_address,
    residual_address,
    new_residual_address,
    post_shared_address,
    comb_shared_address,
    q_shared_address0,
    q_shared_address1,
    q_shared_address2,
    q_shared_address3,
    valid,
):
    """Execute the complete four-route BF16x4 Post body in one PTX scope.

    The CUDA p256 producer keeps the five packed source quads and its route
    temporaries inside one per-thread serial-quad loop.  Expressing that body
    as several Triton tensors extends all 20 FP32 source values across the
    surrounding SSA graph and causes local-memory spills at an 80-register
    launch bound.  This helper matches the CUDA lexical lifetime: it loads
    five BF16x4 quads, computes/stores all four routes, and exposes only the
    BF16-boundary square sum to Triton.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred p, ch1, ch2, ch3, again;
          .reg .u32 channel, coeff_offset, post_address, comb_address;
          .reg .u32 q_address;
          .reg .u64 out_offset, out_address;
          .reg .b32 xp0, xp1;
          .reg .b32 rp00, rp01, rp10, rp11, rp20, rp21, rp30, rp31;
          .reg .b32 packed0, packed1;
          .reg .b16 xh0, xh1, xh2, xh3;
          .reg .b16 r0h0, r0h1, r0h2, r0h3;
          .reg .b16 r1h0, r1h1, r1h2, r1h3;
          .reg .b16 r2h0, r2h1, r2h2, r2h3;
          .reg .b16 r3h0, r3h1, r3h2, r3h3;
          .reg .b16 qh0, qh1, qh2, qh3;
          .reg .f32 x0, x1, x2, x3;
          .reg .f32 r00, r01, r02, r03;
          .reg .f32 r10, r11, r12, r13;
          .reg .f32 r20, r21, r22, r23;
          .reg .f32 r30, r31, r32, r33;
          .reg .f32 post, comb0, comb1, comb2, comb3;
          .reg .f32 q0, q1, q2, q3;
          .reg .f32 rounded0, rounded1, rounded2, rounded3;
          .reg .f32 route_square;

          mov.b32 xp0, 0;
          mov.b32 xp1, 0;
          mov.b32 rp00, 0;
          mov.b32 rp01, 0;
          mov.b32 rp10, 0;
          mov.b32 rp11, 0;
          mov.b32 rp20, 0;
          mov.b32 rp21, 0;
          mov.b32 rp30, 0;
          mov.b32 rp31, 0;
          setp.ne.u32 p, $10, 0;
          @p ld.global.v2.b32 {xp0, xp1}, [$1];
          @p ld.global.v2.b32 {rp00, rp01}, [$2];
          @p ld.global.v2.b32 {rp10, rp11}, [$2+8192];
          @p ld.global.v2.b32 {rp20, rp21}, [$2+16384];
          @p ld.global.v2.b32 {rp30, rp31}, [$2+24576];

          mov.b32 {xh0, xh1}, xp0;
          mov.b32 {xh2, xh3}, xp1;
          mov.b32 {r0h0, r0h1}, rp00;
          mov.b32 {r0h2, r0h3}, rp01;
          mov.b32 {r1h0, r1h1}, rp10;
          mov.b32 {r1h2, r1h3}, rp11;
          mov.b32 {r2h0, r2h1}, rp20;
          mov.b32 {r2h2, r2h3}, rp21;
          mov.b32 {r3h0, r3h1}, rp30;
          mov.b32 {r3h2, r3h3}, rp31;
          cvt.f32.bf16 x0, xh0;
          cvt.f32.bf16 x1, xh1;
          cvt.f32.bf16 x2, xh2;
          cvt.f32.bf16 x3, xh3;
          cvt.f32.bf16 r00, r0h0;
          cvt.f32.bf16 r01, r0h1;
          cvt.f32.bf16 r02, r0h2;
          cvt.f32.bf16 r03, r0h3;
          cvt.f32.bf16 r10, r1h0;
          cvt.f32.bf16 r11, r1h1;
          cvt.f32.bf16 r12, r1h2;
          cvt.f32.bf16 r13, r1h3;
          cvt.f32.bf16 r20, r2h0;
          cvt.f32.bf16 r21, r2h1;
          cvt.f32.bf16 r22, r2h2;
          cvt.f32.bf16 r23, r2h3;
          cvt.f32.bf16 r30, r3h0;
          cvt.f32.bf16 r31, r3h1;
          cvt.f32.bf16 r32, r3h2;
          cvt.f32.bf16 r33, r3h3;

          mov.u32 channel, 0;
          mov.f32 $0, 0f00000000;
        DG_TLE_MHC_POST_ROUTE_LOOP:
          mul.lo.u32 coeff_offset, channel, 4;
          add.u32 post_address, $4, coeff_offset;
          add.u32 comb_address, $5, coeff_offset;
          ld.shared.f32 post, [post_address];
          ld.shared.f32 comb0, [comb_address];
          ld.shared.f32 comb1, [comb_address+16];
          ld.shared.f32 comb2, [comb_address+32];
          ld.shared.f32 comb3, [comb_address+48];

          mul.f32 q0, x0, post;
          fma.rn.f32 q0, r00, comb0, q0;
          fma.rn.f32 q0, r10, comb1, q0;
          fma.rn.f32 q0, r20, comb2, q0;
          fma.rn.f32 q0, r30, comb3, q0;
          mul.f32 q1, x1, post;
          fma.rn.f32 q1, r01, comb0, q1;
          fma.rn.f32 q1, r11, comb1, q1;
          fma.rn.f32 q1, r21, comb2, q1;
          fma.rn.f32 q1, r31, comb3, q1;
          mul.f32 q2, x2, post;
          fma.rn.f32 q2, r02, comb0, q2;
          fma.rn.f32 q2, r12, comb1, q2;
          fma.rn.f32 q2, r22, comb2, q2;
          fma.rn.f32 q2, r32, comb3, q2;
          mul.f32 q3, x3, post;
          fma.rn.f32 q3, r03, comb0, q3;
          fma.rn.f32 q3, r13, comb1, q3;
          fma.rn.f32 q3, r23, comb2, q3;
          fma.rn.f32 q3, r33, comb3, q3;
          cvt.rn.bf16x2.f32 packed0, q1, q0;
          cvt.rn.bf16x2.f32 packed1, q3, q2;

          mul.wide.u32 out_offset, channel, 8192;
          add.u64 out_address, $3, out_offset;
          @p st.global.v2.b32 [out_address], {packed0, packed1};
          mov.u32 q_address, $6;
          setp.eq.u32 ch1, channel, 1;
          setp.eq.u32 ch2, channel, 2;
          setp.eq.u32 ch3, channel, 3;
          @ch1 mov.u32 q_address, $7;
          @ch2 mov.u32 q_address, $8;
          @ch3 mov.u32 q_address, $9;
          st.shared.v2.b32 [q_address], {packed0, packed1};

          mov.b32 {qh0, qh1}, packed0;
          mov.b32 {qh2, qh3}, packed1;
          cvt.f32.bf16 rounded0, qh0;
          cvt.f32.bf16 rounded1, qh1;
          cvt.f32.bf16 rounded2, qh2;
          cvt.f32.bf16 rounded3, qh3;
          mul.f32 route_square, rounded0, rounded0;
          fma.rn.f32 route_square, rounded1, rounded1, route_square;
          fma.rn.f32 route_square, rounded2, rounded2, route_square;
          fma.rn.f32 route_square, rounded3, rounded3, route_square;
          add.rn.f32 $0, $0, route_square;

          add.u32 channel, channel, 1;
          setp.lt.u32 again, channel, 4;
          @again bra.uni DG_TLE_MHC_POST_ROUTE_LOOP;
        }
        """,
        constraints="=&f,l,l,l,r,r,r,r,r,r,r",
        args=[
            x_address,
            residual_address,
            new_residual_address,
            post_shared_address,
            comb_shared_address,
            q_shared_address0,
            q_shared_address1,
            q_shared_address2,
            q_shared_address3,
            tl.cast(valid, tl.uint32),
        ],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _round_store_global_shared_bf16x4(
    global_address,
    shared_address,
    value0,
    value1,
    value2,
    value3,
    valid,
):
    """Round one quad once, publish it globally and to the K64 A stage."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred p;
          .reg .b32 packed0, packed1;
          .reg .b16 h0, h1, h2, h3;
          cvt.rn.bf16x2.f32 packed0, $7, $6;
          cvt.rn.bf16x2.f32 packed1, $9, $8;
          setp.ne.u32 p, $10, 0;
          @p st.global.v2.b32 [$4], {packed0, packed1};
          st.shared.v2.b32 [$5], {packed0, packed1};
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 $0, h0;
          cvt.f32.bf16 $1, h1;
          cvt.f32.bf16 $2, h2;
          cvt.f32.bf16 $3, h3;
        }
        """,
        constraints="=&f,=&f,=&f,=&f,l,r,f,f,f,f,r",
        args=[
            global_address,
            shared_address,
            value0,
            value1,
            value2,
            value3,
            tl.cast(valid, tl.uint32),
        ],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _half_warp_reduce16_sum(value):
    """Reduce FP32 independently in lanes 0..15 and 16..31."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t;
          mov.f32 $0, $1;
          shfl.sync.bfly.b32 t, $0, 8, 0x100f, 0xffffffff;
          add.rn.f32 $0, $0, t;
          shfl.sync.bfly.b32 t, $0, 4, 0x100f, 0xffffffff;
          add.rn.f32 $0, $0, t;
          shfl.sync.bfly.b32 t, $0, 2, 0x100f, 0xffffffff;
          add.rn.f32 $0, $0, t;
          shfl.sync.bfly.b32 t, $0, 1, 0x100f, 0xffffffff;
          add.rn.f32 $0, $0, t;
        }
        """,
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _accumulate_global_row_square(
    address, value, half_lane, valid, initialize
):
    """Let one stable half-warp leader update its row's global partial."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred lane0, row_valid, leader, init, need_load;
          .reg .f32 current;
          setp.eq.u32 lane0, $3, 0;
          setp.ne.u32 row_valid, $4, 0;
          and.pred leader, lane0, row_valid;
          setp.ne.u32 init, $5, 0;
          not.pred init, init;
          and.pred need_load, leader, init;
          mov.f32 current, 0f00000000;
          @need_load ld.global.f32 current, [$1];
          @leader add.rn.f32 current, current, $2;
          @leader st.global.f32 [$1], current;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r,r,r",
        args=[
            address,
            value,
            half_lane,
            tl.cast(valid, tl.uint32),
            tl.cast(initialize, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _route_stage_full_arrive(channel: tl.constexpr):
    """Publish the producer's 256 arrivals to route stage ``channel``."""
    barrier_id: tl.constexpr = 8 + channel
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.arrive $1, 384; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[barrier_id],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _route_stage_full_sync(channel: tl.constexpr):
    """Add/wait for the math partition on route stage ``channel``."""
    barrier_id: tl.constexpr = 8 + channel
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.sync $1, 384; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[barrier_id],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _route_stage_empty_arrive(channel: tl.constexpr):
    """Publish the math partition's 128 route-stage retirements."""
    barrier_id: tl.constexpr = 12 + channel
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.arrive $1, 384; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[barrier_id],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _route_stage_empty_sync(channel: tl.constexpr):
    """Wait until math retires a route stage before producer reuse."""
    barrier_id: tl.constexpr = 12 + channel
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.sync $1, 384; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[barrier_id],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _stage_fn_k32(fn_ptr, fn_shared, flat_hidden_base):
    """Stage one FP32 ``[N32, K32]`` projection-weight fragment."""
    n = tl.arange(0, JIT_BLOCK_N)
    k = tl.arange(0, JIT_WGMMA_K_FRAGMENT)
    values = tl.load(
        fn_ptr
        + n[:, None] * JIT_FLAT_HIDDEN
        + flat_hidden_base
        + k[None, :],
        mask=n[:, None] < JIT_PROJECTION_OUTPUTS,
        other=0.0,
    ).to(tl.float32)
    tl.store(tle.gpu.local_ptr(fn_shared), values)


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
    q_shared,
    fn_shared,
    post_shared,
    comb_shared,
    winner_shared,
    task_shared,
    pre_shared,
    rms_weight_shared,
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
    ENABLE_SQSUM: tl.constexpr,
    ENABLE_TAIL: tl.constexpr,
    PARALLEL_TAIL: tl.constexpr,
):
    """Eight producer warps stream 20/16 route-by-H64 stages."""
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

    # Post[4] and Comb[4x4] are invariant across all K64 stages.
    # Keep exact 64x4 and 64x16 allocations (5 KiB total) so the later RMS
    # weight cache still leaves enough SMEM for two resident CTAs.
    local_m = tl.arange(0, JIT_BLOCK_M)
    coeff_token = tile * JIT_BLOCK_M + local_m
    post_column = tl.arange(0, JIT_HC_MULT)
    post_value = tl.load(
        post_mix_ptr
        + coeff_token[:, None] * JIT_HC_MULT
        + post_column[None, :],
        mask=(coeff_token[:, None] < JIT_TOKENS)
        & (post_column[None, :] < JIT_HC_MULT),
        other=0.0,
    ).to(tl.float32)
    comb_column = tl.arange(0, JIT_HC_MULT * JIT_HC_MULT)
    comb_value = tl.load(
        comb_res_mix_ptr
        + coeff_token[:, None] * JIT_HC_MULT * JIT_HC_MULT
        + comb_column[None, :],
        mask=(coeff_token[:, None] < JIT_TOKENS)
        & (comb_column[None, :] < JIT_HC_MULT * JIT_HC_MULT),
        other=0.0,
    ).to(tl.float32)
    tl.store(tle.gpu.local_ptr(post_shared), post_value)
    tl.store(tle.gpu.local_ptr(comb_shared), comb_value)
    tle.gpu.barrier_wait(coeff_ready)

    # Match the CUDA p256 producer exactly: the first four splits own five
    # K64 blocks and the remaining eleven own four.  Each producer thread
    # serially owns four BF16x4 quads, keeping only one x/residual quad live.
    physical_thread = _small._physical_thread_id_x()
    # The producer carries no tensor accumulator across K64 blocks, so this
    # CTA-uniform runtime loop keeps one copy of the large Post body.  The
    # math partition remains statically unrolled because this pinned TLE
    # cannot carry its WGMMA accumulator through a worker runtime loop.
    for block_in_split in tl.range(0, JIT_MAX_BLOCKS_PER_SPLIT):
        active_block = block_in_split < split_blocks
        if active_block:
            hidden_base = (
                hidden_block_offset + block_in_split
            ) * JIT_HIDDEN_BLOCK
            if block_in_split > 0:
                for channel in tl.static_range(0, JIT_HC_MULT):
                    _route_stage_empty_sync(channel)
            for channel in tl.static_range(0, JIT_HC_MULT):
                for half in tl.static_range(0, JIT_WGMMA_K_FRAGMENTS):
                    _stage_fn_k32(
                        fn_ptr,
                        fn_shared.slot(channel).slot(half),
                        channel * JIT_HIDDEN
                        + hidden_base
                        + half * JIT_WGMMA_K_FRAGMENT,
                    )

            # Keep one Post body per K64 block.  All eight producer warps
            # execute this four-quad loop uniformly, so a runtime loop avoids
            # four copies of the large inline-PTX helper without introducing
            # subgroup divergence.
            for iteration in tl.range(0, 4):
                linear_quad = physical_thread + iteration * 256
                row = linear_quad // 16
                column = (linear_quad % 16) * 4
                token = tile * JIT_BLOCK_M + row
                token_valid = token < JIT_TOKENS
                hidden_quad = hidden_base + column
                x_address = tl.cast(
                    x_ptr + token * JIT_HIDDEN + hidden_quad, tl.uint64
                )
                residual_address = tl.cast(
                    residual_ptr
                    + token * JIT_FLAT_HIDDEN
                    + hidden_quad,
                    tl.uint64,
                )
                new_residual_address = tl.cast(
                    new_residual_ptr
                    + token * JIT_FLAT_HIDDEN
                    + hidden_quad,
                    tl.uint64,
                )
                post_address = _quad._shared_address(
                    tle.gpu.local_ptr(post_shared, (row, 0))
                )
                comb_address = _quad._shared_address(
                    tle.gpu.local_ptr(comb_shared, (row, 0))
                )
                q_address0 = _quad._shared_address(
                    tle.gpu.local_ptr(
                        q_shared.slot(0),
                        (
                            column // JIT_WGMMA_K_FRAGMENT,
                            row,
                            column % JIT_WGMMA_K_FRAGMENT,
                        ),
                    )
                )
                q_address1 = _quad._shared_address(
                    tle.gpu.local_ptr(
                        q_shared.slot(1),
                        (
                            column // JIT_WGMMA_K_FRAGMENT,
                            row,
                            column % JIT_WGMMA_K_FRAGMENT,
                        ),
                    )
                )
                q_address2 = _quad._shared_address(
                    tle.gpu.local_ptr(
                        q_shared.slot(2),
                        (
                            column // JIT_WGMMA_K_FRAGMENT,
                            row,
                            column % JIT_WGMMA_K_FRAGMENT,
                        ),
                    )
                )
                q_address3 = _quad._shared_address(
                    tle.gpu.local_ptr(
                        q_shared.slot(3),
                        (
                            column // JIT_WGMMA_K_FRAGMENT,
                            row,
                            column % JIT_WGMMA_K_FRAGMENT,
                        ),
                    )
                )
                quad_square = _serial_quad_post_ptx(
                    x_address,
                    residual_address,
                    new_residual_address,
                    post_address,
                    comb_address,
                    q_address0,
                    q_address1,
                    q_address2,
                    q_address3,
                    token_valid,
                )

                if ENABLE_SQSUM:
                    row_square = _half_warp_reduce16_sum(quad_square)
                    half_lane = physical_thread % 16
                    sq_offset = (
                        (
                            (tile * JIT_NUM_SPLITS + split)
                            * JIT_BLOCK_M
                            + row
                        )
                        * JIT_BLOCK_N
                        + JIT_PROJECTION_OUTPUTS
                    )
                    _accumulate_global_row_square(
                        tl.cast(partials_ptr + sq_offset, tl.uint64),
                        row_square,
                        half_lane,
                        token_valid,
                        block_in_split == 0,
                    )

            # Four non-aligned CTA named barriers publish the route stages
            # independently.  IDs 8..15 are disjoint from TLE allocations.
            for channel in tl.static_range(0, JIT_HC_MULT):
                _route_stage_full_arrive(channel)

    if ENABLE_TAIL:
        if JIT_USE_RMS_WEIGHT_CACHE:
            weight_h = tl.arange(0, JIT_HIDDEN)
            weight = tl.load(rmsnorm_weight_ptr + weight_h)
            tl.store(
                tle.gpu.local_ptr(rms_weight_shared, (weight_h,)),
                weight,
            )
        # Both partitions publish their global partial stores before one
        # producer thread release-increments this tile's split counter.
        tle.gpu.barrier_wait(producer_ready)
        physical_thread = _small._physical_thread_id_x()
        state_base = tl.cast(tile_states_ptr + tile * 32, tl.uint64)
        leader_ready = _bounded_tile_ready(
            state_base,
            physical_thread,
        )
        tl.store(
            tle.gpu.local_ptr(winner_shared, (0,)),
            leader_ready,
            mask=physical_thread == 0,
        )
        tle.gpu.barrier_wait(handoff_ready)
        helper_ready = tl.load(tle.gpu.local_ptr(winner_shared, (0,)))
        if helper_ready != 0:
            queue_active = True
            for queue_iteration in tl.range(0, JIT_BLOCK_M):
                if queue_active:
                    claimed_row = _claim_tile_row(
                        state_base, physical_thread, JIT_TAIL_BATCH
                    )
                    tl.store(
                        tle.gpu.local_ptr(task_shared, (0,)),
                        claimed_row,
                        mask=physical_thread == 0,
                    )
                    tle.gpu.barrier_wait(task_ready)
                    row = tl.cast(
                        tl.load(tle.gpu.local_ptr(task_shared, (0,))),
                        tl.int32,
                    )
                    token = tl.cast(tile * JIT_BLOCK_M, tl.int32) + row
                    queue_active = (
                        (row >= 0)
                        & (row < JIT_BLOCK_M)
                        & (token < JIT_TOKENS)
                    )
                    if queue_active:
                        live_rows = tl.minimum(
                            JIT_TAIL_BATCH,
                            tl.minimum(
                                JIT_BLOCK_M - row, JIT_TOKENS - token
                            ),
                        )
                        if PARALLEL_TAIL:
                            # Producer warps 0 and 4 are independent routing
                            # owners for rows zero and one.  The remaining
                            # three warps in each 128-thread worker bypass
                            # routing and arrive once at the full-CTA Pre
                            # barrier below.
                            producer_warp = physical_thread // 32
                            route_owner = producer_warp // 4
                            route_active = (
                                (producer_warp % 4 == 0)
                                & (route_owner < live_rows)
                                & (route_owner < 2)
                            )
                            if route_active:
                                lane = physical_thread % 32
                                packed_sum = 0.0
                                for source_split in tl.static_range(
                                    0, JIT_NUM_SPLITS
                                ):
                                    source_offset = (
                                        ((tile * JIT_NUM_SPLITS
                                          + source_split)
                                         * JIT_BLOCK_M
                                         + row + route_owner)
                                        * JIT_BLOCK_N
                                        + lane
                                    )
                                    packed_sum += tl.load(
                                        partials_ptr + source_offset,
                                        mask=(
                                            lane
                                            <= JIT_PROJECTION_OUTPUTS
                                        ),
                                        other=0.0,
                                    )
                                _routing_finalize_no_barrier(
                                    pre_shared.slot(
                                        tl.cast(route_owner, tl.int32)
                                    ),
                                    mix_scales_ptr,
                                    mix_bases_ptr,
                                    new_post_mix_ptr
                                    + (token + route_owner)
                                    * JIT_HC_MULT,
                                    new_comb_res_mix_ptr
                                    + (token + route_owner)
                                    * JIT_HC_MULT * JIT_HC_MULT,
                                    packed_sum,
                                    lane,
                                    hc_norm_eps,
                                    hc_pre_eps,
                                    hc_post_scale,
                                    sinkhorn_eps,
                                )
                        # Math and producer partitions each execute this
                        # full-CTA rendezvous exactly once per batch.
                        tle.gpu.barrier_wait(pre_ready)
                        _hidden_tail_batch2_no_wait(
                            pre_shared.slot(tl.cast(0, tl.int32)),
                            pre_shared.slot(tl.cast(1, tl.int32)),
                            rms_weight_shared,
                            new_residual_ptr + token * JIT_FLAT_HIDDEN,
                            rmsnorm_weight_ptr,
                            y_bf16_ptr + token * JIT_HIDDEN,
                            tl.minimum(live_rows, 2),
                            rmsnorm_eps,
                            rmsnorm_scale,
                        )
                        tle.gpu.barrier_wait(task_done)


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
    q_shared,
    fn_shared,
    route_shared,
    winner_shared,
    task_shared,
    pre_shared,
    rms_weight_shared,
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
    split_blocks = JIT_BASE_BLOCKS_PER_SPLIT + tl.where(
        split < JIT_WIDE_SPLITS,
        JIT_EXTRA_BLOCKS_PER_WIDE_SPLIT,
        0,
    )
    for block_in_split in tl.static_range(0, JIT_MAX_BLOCKS_PER_SPLIT):
        active_block = block_in_split < split_blocks
        if active_block:
            for channel in tl.static_range(0, JIT_HC_MULT):
                _route_stage_full_sync(channel)
                # The BF16 A tile is loaded into the math warpgroup's
                # registers and converted there, matching CUDA's RS-WGMMA
                # path.  Producer half-warps separately publish the matching
                # BF16-boundary square-sum scalar for each row.
                for half in tl.static_range(0, JIT_WGMMA_K_FRAGMENTS):
                    q_registers = tl.load(
                        tle.gpu.local_ptr(
                            q_shared.slot(channel).slot(half)
                        )
                    ).to(tl.float32)
                    accumulator = tle.gpu.wgmma(
                        q_registers,
                        fn_shared.slot(channel).slot(half),
                        accumulator,
                        input_precision="tf32",
                        trans_b=True,
                    )
                    accumulator = tle.gpu.wgmma_wait(0, accumulator)
                _route_stage_empty_arrive(channel)

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
        helper_ready = tl.load(tle.gpu.local_ptr(winner_shared, (0,)))
        if helper_ready != 0:
            physical_thread = _small._physical_thread_id_x()
            lane = physical_thread % 32
            queue_active = True
            for queue_iteration in tl.range(0, JIT_BLOCK_M):
                if queue_active:
                    tle.gpu.barrier_wait(task_ready)
                    row = tl.cast(
                        tl.load(tle.gpu.local_ptr(task_shared, (0,))),
                        tl.int32,
                    )
                    token = tl.cast(tile * JIT_BLOCK_M, tl.int32) + row
                    queue_active = (
                        (row >= 0)
                        & (row < JIT_BLOCK_M)
                        & (token < JIT_TOKENS)
                    )
                    if queue_active:
                        live_rows = tl.minimum(
                            JIT_TAIL_BATCH,
                            tl.minimum(
                                JIT_BLOCK_M - row, JIT_TOKENS - token
                            ),
                        )
                        if PARALLEL_TAIL:
                            # Math warp zero is the third routing owner.  Its
                            # sibling warps bypass this static Sinkhorn body
                            # and rendezvous exactly once at pre_ready.
                            worker_warp = (
                                physical_thread - JIT_PRODUCER_THREADS
                            ) // 32
                            if (worker_warp == 0) & (live_rows > 2):
                                packed_sum = 0.0
                                for source_split in tl.static_range(
                                    0, JIT_NUM_SPLITS
                                ):
                                    source_offset = (
                                        ((tile * JIT_NUM_SPLITS
                                          + source_split)
                                         * JIT_BLOCK_M
                                         + row + 2)
                                        * JIT_BLOCK_N
                                        + lane
                                    )
                                    packed_sum += tl.load(
                                        partials_ptr + source_offset,
                                        mask=(
                                            lane
                                            <= JIT_PROJECTION_OUTPUTS
                                        ),
                                        other=0.0,
                                    )
                                _routing_finalize_no_barrier(
                                    pre_shared.slot(tl.cast(2, tl.int32)),
                                    mix_scales_ptr,
                                    mix_bases_ptr,
                                    new_post_mix_ptr
                                    + (token + 2) * JIT_HC_MULT,
                                    new_comb_res_mix_ptr
                                    + (token + 2)
                                    * JIT_HC_MULT * JIT_HC_MULT,
                                    packed_sum,
                                    lane,
                                    hc_norm_eps,
                                    hc_pre_eps,
                                    hc_post_scale,
                                    sinkhorn_eps,
                                )
                        else:
                            for batch_offset in tl.range(0, live_rows):
                                packed_sum = 0.0
                                for source_split in tl.static_range(
                                    0, JIT_NUM_SPLITS
                                ):
                                    source_offset = (
                                        ((tile * JIT_NUM_SPLITS
                                          + source_split)
                                         * JIT_BLOCK_M
                                         + row + batch_offset)
                                        * JIT_BLOCK_N
                                        + lane
                                    )
                                    packed_sum += tl.load(
                                        partials_ptr + source_offset,
                                        mask=(
                                            lane
                                            <= JIT_PROJECTION_OUTPUTS
                                        ),
                                        other=0.0,
                                    )
                                _routing_finalize_no_barrier(
                                    pre_shared.slot(
                                        tl.cast(batch_offset, tl.int32)
                                    ),
                                    mix_scales_ptr,
                                    mix_bases_ptr,
                                    new_post_mix_ptr
                                    + (token + batch_offset)
                                    * JIT_HC_MULT,
                                    new_comb_res_mix_ptr
                                    + (token + batch_offset)
                                    * JIT_HC_MULT * JIT_HC_MULT,
                                    packed_sum,
                                    lane,
                                    hc_norm_eps,
                                    hc_pre_eps,
                                    hc_post_scale,
                                    sinkhorn_eps,
                                )
                        tle.gpu.barrier_wait(pre_ready)
                        if live_rows > 2:
                            _small._warp_sync_full_mask()
                            _hidden_tail_no_wait(
                                pre_shared.slot(tl.cast(2, tl.int32)),
                                rms_weight_shared,
                                new_residual_ptr
                                + (token + 2) * JIT_FLAT_HIDDEN,
                                rmsnorm_weight_ptr,
                                y_bf16_ptr
                                + (token + 2) * JIT_HIDDEN,
                                rmsnorm_eps,
                                rmsnorm_scale,
                            )
                        tle.gpu.barrier_wait(task_done)


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
    ENABLE_SQSUM: tl.constexpr,
    ENABLE_TAIL: tl.constexpr,
    PARALLEL_TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid // JIT_NUM_SPLITS
    split = pid % JIT_NUM_SPLITS
    tl.static_assert(JIT_NUM_M_TILES == 17)
    tl.static_assert(JIT_NUM_SPLITS == 15)
    tl.static_assert(JIT_GRID_CTAS == 255)
    tl.static_assert(JIT_HIDDEN_BLOCK == 64)

    q_shared = tle.gpu.alloc(
        [
            JIT_PIPE_STAGES,
            JIT_WGMMA_K_FRAGMENTS,
            JIT_BLOCK_M,
            JIT_WGMMA_K_FRAGMENT,
        ],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    fn_shared = tle.gpu.alloc(
        [
            JIT_PIPE_STAGES,
            JIT_WGMMA_K_FRAGMENTS,
            JIT_BLOCK_N,
            JIT_WGMMA_K_FRAGMENT,
        ],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    post_shared = tle.gpu.alloc(
        [JIT_BLOCK_M, JIT_HC_MULT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    comb_shared = tle.gpu.alloc(
        [JIT_BLOCK_M, JIT_HC_MULT * JIT_HC_MULT],
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
    rms_weight_shared = tle.gpu.alloc(
        [JIT_HIDDEN], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
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
                    q_shared,
                    fn_shared,
                    post_shared,
                    comb_shared,
                    winner_shared,
                    task_shared,
                    pre_shared,
                    rms_weight_shared,
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
                    ENABLE_SQSUM,
                    ENABLE_TAIL,
                    PARALLEL_TAIL,
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
                    q_shared,
                    fn_shared,
                    route_shared,
                    winner_shared,
                    task_shared,
                    pre_shared,
                    rms_weight_shared,
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
    sqsum_enabled: bool
    tail_enabled: bool
    parallel_tail: bool

    @property
    def grid_ctas(self) -> int:
        return GRID_CTAS

    def _kernel_args(self) -> tuple[object, ...]:
        return (
            *self.tensors,
            self.partials,
            self.tile_states,
            *self.scalars,
            self.sqsum_enabled,
            self.tail_enabled,
            self.parallel_tail,
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
    if maxnreg not in (80, 84, 96, 112, 128, 144, 160):
        raise ValueError(
            "DG_TLE_MHC_EAGER_T1028_MAXNREG must be "
            "80/84/96/112/128/144/160"
        )
    tail_enabled = (
        os.environ.get("DG_TLE_MHC_EAGER_T1028_PRODUCER_ONLY", "0") != "1"
    )
    sqsum_enabled = (
        os.environ.get("DG_TLE_MHC_EAGER_T1028_DISABLE_SQSUM", "0") != "1"
    )
    parallel_tail = (
        os.environ.get("DG_TLE_MHC_EAGER_T1028_PARALLEL_TAIL", "1") == "1"
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
        "cta_mapping": "tile=pid//15,split=pid%15",
        "producer_warps": NUM_WARPS,
        "math_warps": MATH_WARPS,
        "physical_cta_threads": (NUM_WARPS + MATH_WARPS) * 32,
        "target_registers_per_thread": maxnreg,
        "producer_pipeline_stages": PIPE_STAGES,
        "producer_coeff_cache_bytes": (
            BLOCK_M * (HC_MULT + HC_MULT * HC_MULT) * 4
        ),
        "rms_weight_cache": USE_RMS_WEIGHT_CACHE,
        "rms_weight_cache_bytes": HIDDEN * 2 if USE_RMS_WEIGHT_CACHE else 0,
        "wgmma_wait_policy": "register_a_wait0_per_stage",
        "synchronization": "per_tile_bounded_helper_phase_queue",
        "counter_policy": "arrival_phase_head_128b_state",
        "full_grid_barrier": False,
        "full_grid_polling": False,
        "cooperative_launch": False,
        "full_grid_residency_required": False,
        "bounded_helper_queue_migrated": True,
        "bounded_helper_polls_target": BOUNDED_HELPER_POLLS_TARGET,
        "tail_batch": TAIL_BATCH,
        "tail_schedule": (
            f"finite_poll_ready_ctas_atomic_head_batch{TAIL_BATCH}"
        ),
        "tail_hidden_workers": (
            "default8_batch2_plus_math4_batch1"
            if TAIL_BATCH == 3
            else "default8_batch2_only"
        ),
        "tail_routing_workers": (
            "producer_warp0_plus_producer_warp4_plus_math_warp0"
            if parallel_tail
            else f"math_warp0_serial{TAIL_BATCH}"
        ),
        "tail_enabled": tail_enabled,
        "sqsum_enabled": sqsum_enabled,
        "parallel_tail": parallel_tail,
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
        sqsum_enabled=sqsum_enabled,
        tail_enabled=tail_enabled,
        parallel_tail=parallel_tail,
    )


__all__ = [
    "EagerT1028P256Plan",
    "GRID_CTAS",
    "NUM_M_TILES",
    "NUM_SPLITS",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
]

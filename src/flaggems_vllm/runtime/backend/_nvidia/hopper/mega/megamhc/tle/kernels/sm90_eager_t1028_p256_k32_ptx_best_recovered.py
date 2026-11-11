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


TOKENS = 1028
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = HIDDEN * HC_MULT
SINKHORN_ITERS = 20
BLOCK_M = 64
BLOCK_N = 32
HIDDEN_BLOCK = 32
NUM_HIDDEN_BLOCKS = HIDDEN // HIDDEN_BLOCK
NUM_SPLITS = 15
BASE_BLOCKS_PER_SPLIT = NUM_HIDDEN_BLOCKS // NUM_SPLITS
WIDE_SPLITS = 4
EXTRA_BLOCKS_PER_WIDE_SPLIT = 2
MAX_BLOCKS_PER_SPLIT = (
    BASE_BLOCKS_PER_SPLIT + EXTRA_BLOCKS_PER_WIDE_SPLIT
)
SPLIT_HIDDEN_BLOCKS = (
    10, 10, 10, 10,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
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
    post_shared,
    comb_shared,
    winner_shared,
    task_shared,
    pre_shared,
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
):
    """Eight producer warps stream 40 route-by-H32 stages."""
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
                    post_shared, (local_m, coeff_channel)
                )
            )[:, None]
            q = (
                x * post
                + r0
                * tl.load(
                    tle.gpu.local_ptr(
                        comb_shared,
                        (
                            local_m,
                            coeff_channel,
                        ),
                    )
                )[:, None]
                + r1
                * tl.load(
                    tle.gpu.local_ptr(
                        comb_shared,
                        (
                            local_m,
                            coeff_channel + JIT_HC_MULT,
                        ),
                    )
                )[:, None]
                + r2
                * tl.load(
                    tle.gpu.local_ptr(
                        comb_shared,
                        (
                            local_m,
                            coeff_channel + 2 * JIT_HC_MULT,
                        ),
                    )
                )[:, None]
                + r3
                * tl.load(
                    tle.gpu.local_ptr(
                        comb_shared,
                        (
                            local_m,
                            coeff_channel + 3 * JIT_HC_MULT,
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

    if ENABLE_TAIL:
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
                        state_base, physical_thread
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
                            3,
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
    q_fn_reader,
    route_shared,
    winner_shared,
    task_shared,
    pre_shared,
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
                            3,
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
    ENABLE_TAIL: tl.constexpr,
    PARALLEL_TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid // JIT_NUM_SPLITS
    split = pid % JIT_NUM_SPLITS
    tl.static_assert(JIT_NUM_M_TILES == 17)
    tl.static_assert(JIT_NUM_SPLITS == 15)
    tl.static_assert(JIT_GRID_CTAS == 255)
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
                    post_shared,
                    comb_shared,
                    winner_shared,
                    task_shared,
                    pre_shared,
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
                    route_shared,
                    winner_shared,
                    task_shared,
                    pre_shared,
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
        "wgmma_wait_policy": "wait1_then_final_wait0",
        "synchronization": "per_tile_bounded_helper_phase_queue",
        "counter_policy": "arrival_phase_head_128b_state",
        "full_grid_barrier": False,
        "full_grid_polling": False,
        "cooperative_launch": False,
        "full_grid_residency_required": False,
        "bounded_helper_queue_migrated": True,
        "bounded_helper_polls_target": BOUNDED_HELPER_POLLS_TARGET,
        "tail_schedule": "finite_poll_ready_ctas_atomic_head_batch3",
        "tail_hidden_workers": "default8_batch2_plus_math4_batch1",
        "tail_routing_workers": (
            "producer_warp0_plus_producer_warp4_plus_math_warp0"
            if parallel_tail
            else "math_warp0_serial3"
        ),
        "tail_enabled": tail_enabled,
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

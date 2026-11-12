"""Clean role2 kernel with a compact warp-local PTX Sinkhorn loop.

This follows the production grid geometry for the pinned DeepSeek-V4-Flash
Normal BF16 profile at logical ``T=136..256``.  It does not participate in
production dispatch unless its explicit experiment switch is set.  The
cooperative grid is mapped as::

    tile  = program_id // 32
    split = program_id % 32

Consequently every CTA produces exactly one M64 tile for one H128 hidden
split.  T=136..192 launches the real 3x32=96 producer grid; T=200..256
launches 4x32=128.  CTAs with the same split write disjoint token rows in a
shared ``[32, 256, 32]`` FP32 workspace.
The final tile masks input loads, public stores, and all tails beyond the
selected logical token count.

After the sole grid-wide release/acquire rendezvous CTA ``i`` owns tokens
``i + round * gridDim.x`` when active.  Its producer partition runs the
hidden/RMSNorm tail while its WGMMA worker partition reduces the 32
projection partials and runs routing.  A
prepared plan enqueues exactly one kernel and therefore captures as one CUDA
Graph kernel node.  Full-grid residency must be proved from the compiled
CUfunction before launch; otherwise the software rendezvous can deadlock.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from .. import fused_post_gemm_t64 as _t64

PADDED_TOKENS = 256
LOGICAL_TOKEN_BUCKETS = (
    136,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
    216,
    224,
    232,
    240,
    248,
    256,
)
_TOKENS_TEXT = os.environ.get("DG_TLE_MHC_MID_GRID_PIPE4_ROLE2_LOGICAL_TOKENS", "192")
try:
    LOGICAL_TOKENS = int(_TOKENS_TEXT)
except ValueError as exc:  # pragma: no cover - import-time contract
    raise ValueError(
        "DG_TLE_MHC_MID_GRID_PIPE4_ROLE2_LOGICAL_TOKENS must be an integer"
    ) from exc
if LOGICAL_TOKENS not in LOGICAL_TOKEN_BUCKETS:
    raise ValueError(
        "DG_TLE_MHC_MID_GRID_PIPE4_ROLE2_LOGICAL_TOKENS must be one of "
        f"{LOGICAL_TOKEN_BUCKETS}, got {LOGICAL_TOKENS}"
    )
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = HIDDEN * HC_MULT
SINKHORN_ITERS = 20
BLOCK_M = 64
BLOCK_N = 32
NUM_M_TILES = (LOGICAL_TOKENS + BLOCK_M - 1) // BLOCK_M
NUM_SPLITS = 32
HIDDEN_PER_SPLIT = HIDDEN // NUM_SPLITS
HIDDEN_CHUNK = 64
NUM_HIDDEN_CHUNKS = HIDDEN_PER_SPLIT // HIDDEN_CHUNK
GRID_CTAS = NUM_M_TILES * NUM_SPLITS
TAIL_OWNER_CTAS = GRID_CTAS
TOKENS_PER_OWNER = (LOGICAL_TOKENS + TAIL_OWNER_CTAS - 1) // TAIL_OWNER_CTAS
NUM_WARPS = 16
STAGED_MATH_WARPS = 4
STAGED_MATH_REGS = 128
PIPE_STAGES = 4
ROLE2_TOKENS = 2
ROLE2_ROOT_WARPS_PER_TOKEN = NUM_WARPS // ROLE2_TOKENS
ROLE2_ROUTING_WARPS_PER_TOKEN = 1
ROLE2_PHASE = "early_pre_ilp4_ptx_sinkhorn"

JIT_LOGICAL_TOKENS = tl.constexpr(LOGICAL_TOKENS)
JIT_PADDED_TOKENS = tl.constexpr(PADDED_TOKENS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)
JIT_SINKHORN_ITERS = tl.constexpr(SINKHORN_ITERS)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_NUM_M_TILES = tl.constexpr(NUM_M_TILES)
JIT_NUM_SPLITS = tl.constexpr(NUM_SPLITS)
JIT_HIDDEN_PER_SPLIT = tl.constexpr(HIDDEN_PER_SPLIT)
JIT_HIDDEN_CHUNK = tl.constexpr(HIDDEN_CHUNK)
JIT_NUM_HIDDEN_CHUNKS = tl.constexpr(NUM_HIDDEN_CHUNKS)
JIT_GRID_CTAS = tl.constexpr(GRID_CTAS)
JIT_TAIL_OWNER_CTAS = tl.constexpr(TAIL_OWNER_CTAS)
JIT_TOKENS_PER_OWNER = tl.constexpr(TOKENS_PER_OWNER)
JIT_CTA_THREADS = tl.constexpr(NUM_WARPS * 32)
JIT_STAGED_MATH_WARPS = tl.constexpr(STAGED_MATH_WARPS)
JIT_STAGED_MATH_THREADS = tl.constexpr(STAGED_MATH_WARPS * 32)
JIT_STAGED_MATH_REGS = tl.constexpr(STAGED_MATH_REGS)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_STAGED_LOGICAL_THREADS = tl.constexpr(NUM_WARPS * 32 + STAGED_MATH_WARPS * 32)
JIT_ROLE2_TOKENS = tl.constexpr(ROLE2_TOKENS)
JIT_ROLE2_ROOT_THREADS = tl.constexpr(ROLE2_ROOT_WARPS_PER_TOKEN * 32)
JIT_ROLE2_RUN_ROUTE = tl.constexpr(True)
JIT_ROLE2_RUN_SYNC = tl.constexpr(False)
JIT_ROLE2_RUN_HIDDEN = tl.constexpr(True)
JIT_ROLE2_EARLY_PRE = tl.constexpr(True)


@triton.jit
def _monotonic_grid96_arrive_and_wait(counter_base, thread):
    """Release/acquire rendezvous for ordered 96-CTA Graph replays."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, pending;
          .reg .u64 old, one, divisor, generation, target, current;
          setp.eq.u32 leader, $2, 0;
          setp.ne.u32 pending, $2, $2;
          mov.u64 old, 0;
          mov.u64 one, 1;
          mov.u64 divisor, 96;
          @leader atom.release.gpu.global.add.u64 old, [$1], one;
          @leader div.u64 generation, old, divisor;
          @leader add.u64 generation, generation, 1;
          @leader mul.lo.u64 target, generation, divisor;
        DG_TLE_MHC_GRID96_WAIT:
          @leader ld.acquire.gpu.global.u64 current, [$1];
          @leader setp.lt.u64 pending, current, target;
          @pending bra DG_TLE_MHC_GRID96_WAIT;
          mov.u64 $0, old;
        }
        """,
        constraints="=l,l,r",
        args=[counter_base, thread],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _monotonic_actual_grid_arrive_and_wait(counter_base, thread):
    if JIT_GRID_CTAS == 96:
        return _monotonic_grid96_arrive_and_wait(counter_base, thread)
    else:
        tl.static_assert(JIT_GRID_CTAS == 128)
        return _t64._monotonic_grid128_arrive_and_wait(counter_base, thread)


@triton.jit
def _role2_store_pre_and_optionally_arrive(
    pre_address,
    pre_mix,
    lane,
    ROLE: tl.constexpr,
):
    """Publish four Pre scalars and optionally release one root subgroup."""
    if JIT_ROLE2_EARLY_PRE:
        if ROLE == 0:
            _ = tl.inline_asm_elementwise(
                asm="""
                {
                  .reg .pred p;
                  setp.lt.u32 p, $3, 4;
                  @p st.shared.f32 [$1], $2;
                  barrier.cta.arrive 14, 288;
                  mov.u32 $0, 0;
                }
                """,
                constraints="=r,r,f,r",
                args=[pre_address, pre_mix, lane],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
        else:
            tl.static_assert(ROLE == 1)
            _ = tl.inline_asm_elementwise(
                asm="""
                {
                  .reg .pred p;
                  setp.lt.u32 p, $3, 4;
                  @p st.shared.f32 [$1], $2;
                  barrier.cta.arrive 15, 288;
                  mov.u32 $0, 0;
                }
                """,
                constraints="=r,r,f,r",
                args=[pre_address, pre_mix, lane],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
    else:
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


@triton.jit
def _role2_wait_early_pre(physical_thread):
    if physical_thread < JIT_ROLE2_ROOT_THREADS:
        _ = tl.inline_asm_elementwise(
            asm="{ barrier.cta.sync 14, 288; mov.u32 $0, 0; }",
            constraints="=r",
            args=[],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
    else:
        _ = tl.inline_asm_elementwise(
            asm="{ barrier.cta.sync 15, 288; mov.u32 $0, 0; }",
            constraints="=r",
            args=[],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _role2_hidden_tail(
    pre_shared0,
    pre_shared1,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    pid,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Run two H4096 tails concurrently on the sixteen root warps."""
    token0 = pid
    token1 = pid + JIT_TAIL_OWNER_CTAS
    token0_active = token0 < JIT_LOGICAL_TOKENS
    token1_active = token1 < JIT_LOGICAL_TOKENS
    # The pinned TLE runtime has no public BlockEncoding API.  Its default
    # H8192 ownership is logical = tid + item * 512.  This permutation makes
    # root warps 0..7 own token0/H4096 and 8..15 own token1/H4096 exactly
    # once; sm90_role2_subgroup_probe.py audits that contract and canaries.
    logical = tl.arange(0, JIT_ROLE2_TOKENS * JIT_HIDDEN)
    layout_thread = logical % JIT_CTA_THREADS
    item = logical // JIT_CTA_THREADS
    token_slot = layout_thread // JIT_ROLE2_ROOT_THREADS
    hidden = item * JIT_ROLE2_ROOT_THREADS + (layout_thread % JIT_ROLE2_ROOT_THREADS)
    token = tl.where(token_slot == 0, token0, token1)
    token_active = tl.where(token_slot == 0, token0_active, token1_active)
    norm_input = tl.zeros((JIT_ROLE2_TOKENS * JIT_HIDDEN,), tl.float32)
    for channel in tl.static_range(0, JIT_HC_MULT):
        value = tl.load(
            new_residual_ptr + token * JIT_FLAT_HIDDEN + channel * JIT_HIDDEN + hidden,
            mask=token_active,
            other=0.0,
        ).to(tl.float32)
        pre0 = tl.load(
            tle.gpu.local_ptr(pre_shared0, (channel,)),
            mask=token0_active,
            other=0.0,
        )
        pre1 = tl.load(
            tle.gpu.local_ptr(pre_shared1, (channel,)),
            mask=token1_active,
            other=0.0,
        )
        pre = tl.where(token_slot == 0, pre0, pre1)
        norm_input += value * pre

    norm_input_bf16 = norm_input.to(tl.bfloat16, fp_downcast_rounding="rtne")
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    squares = norm_input_fp32 * norm_input_fp32
    sum0 = tl.sum(tl.where(token_slot == 0, squares, 0.0), axis=0)
    sum1 = tl.sum(tl.where(token_slot == 1, squares, 0.0), axis=0)
    rms0 = tl.rsqrt(sum0 / JIT_HIDDEN + rmsnorm_eps)
    rms1 = tl.rsqrt(sum1 / JIT_HIDDEN + rmsnorm_eps)
    rms = tl.where(token_slot == 0, rms0, rms1)
    rms_weight = tl.load(rmsnorm_weight_ptr + hidden).to(tl.float32)
    y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + token * JIT_HIDDEN + hidden,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=token_active,
    )


@triton.jit
def _tile_split32_producer(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    new_residual_ptr,
    partials_ptr,
    pipe_writer,
    producer_ready,
    grid_acquire_ready,
    pre_shared0,
    pre_shared1,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    grid_counter_ptr,
    tile,
    split,
    pid,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Sixteen producer warps form one M64 x H128 tile."""
    m = tile * JIT_BLOCK_M + tl.arange(0, JIT_BLOCK_M)
    token_active = m < JIT_LOGICAL_TOKENS
    n = tl.arange(0, JIT_BLOCK_N)
    squared_sum = tl.zeros((JIT_BLOCK_M,), tl.float32)

    for hidden_chunk in tl.static_range(0, JIT_NUM_HIDDEN_CHUNKS):
        local_hidden = tl.arange(0, JIT_HIDDEN_CHUNK)
        hidden = (
            split * JIT_HIDDEN_PER_SPLIT
            + hidden_chunk * JIT_HIDDEN_CHUNK
            + local_hidden
        )
        token_hidden = m[:, None] * JIT_HIDDEN + hidden[None, :]
        token_residual = m[:, None] * JIT_FLAT_HIDDEN + hidden[None, :]
        # X and all four residual routes are invariant across output channel.
        # Retain one H64 slice through four stages instead of reloading it for
        # every channel as the first migration did.
        x = tl.load(
            x_ptr + token_hidden,
            mask=token_active[:, None],
            other=0.0,
        ).to(tl.float32)
        r0 = tl.load(
            residual_ptr + token_residual,
            mask=token_active[:, None],
            other=0.0,
        ).to(tl.float32)
        r1 = tl.load(
            residual_ptr + token_residual + JIT_HIDDEN,
            mask=token_active[:, None],
            other=0.0,
        ).to(tl.float32)
        r2 = tl.load(
            residual_ptr + token_residual + 2 * JIT_HIDDEN,
            mask=token_active[:, None],
            other=0.0,
        ).to(tl.float32)
        r3 = tl.load(
            residual_ptr + token_residual + 3 * JIT_HIDDEN,
            mask=token_active[:, None],
            other=0.0,
        ).to(tl.float32)
        for channel in tl.static_range(0, JIT_HC_MULT):
            iteration = hidden_chunk * JIT_HC_MULT + channel
            slot = pipe_writer.acquire(iteration)
            post = tl.load(
                post_mix_ptr + m[:, None] * JIT_HC_MULT + channel,
                mask=token_active[:, None],
                other=0.0,
            ).to(tl.float32)
            comb_base = m[:, None] * JIT_HC_MULT * JIT_HC_MULT + channel
            q = (
                x * post
                + r0
                * tl.load(
                    comb_res_mix_ptr + comb_base,
                    mask=token_active[:, None],
                    other=0.0,
                ).to(tl.float32)
                + r1
                * tl.load(
                    comb_res_mix_ptr + comb_base + JIT_HC_MULT,
                    mask=token_active[:, None],
                    other=0.0,
                ).to(tl.float32)
                + r2
                * tl.load(
                    comb_res_mix_ptr + comb_base + 2 * JIT_HC_MULT,
                    mask=token_active[:, None],
                    other=0.0,
                ).to(tl.float32)
                + r3
                * tl.load(
                    comb_res_mix_ptr + comb_base + 3 * JIT_HC_MULT,
                    mask=token_active[:, None],
                    other=0.0,
                ).to(tl.float32)
            )
            q_bf16 = q.to(tl.bfloat16, fp_downcast_rounding="rtne")
            q_fp32 = q_bf16.to(tl.float32)
            flat_hidden = channel * JIT_HIDDEN + hidden
            tl.store(
                new_residual_ptr + m[:, None] * JIT_FLAT_HIDDEN + flat_hidden[None, :],
                q_bf16,
                mask=token_active[:, None],
            )
            fn_offsets = n[:, None] * JIT_FLAT_HIDDEN + flat_hidden[None, :]
            fn_tile = tl.load(
                fn_ptr + fn_offsets,
                mask=n[:, None] < JIT_PROJECTION_OUTPUTS,
                other=0.0,
            ).to(tl.float32)
            squared_sum += tl.sum(q_fp32 * q_fp32, axis=1)
            tl.store(tle.gpu.local_ptr(slot.q), q_fp32)
            tl.store(tle.gpu.local_ptr(slot.fn), fn_tile)
            pipe_writer.commit(iteration)

    sq_offsets = (
        split * JIT_PADDED_TOKENS * JIT_BLOCK_N
        + m * JIT_BLOCK_N
        + JIT_PROJECTION_OUTPUTS
    )
    tl.store(partials_ptr + sq_offsets, squared_sum, mask=token_active)

    # Publish both specialization partitions before the grid leader performs
    # the release atomic.  The acquire is propagated back to the worker group
    # through grid_acquire_ready.
    tle.gpu.barrier_wait(producer_ready)
    physical_thread = _t64._physical_thread_id_x()
    _ = _monotonic_actual_grid_arrive_and_wait(
        tl.cast(grid_counter_ptr, tl.uint64), physical_thread
    )
    tle.gpu.barrier_wait(grid_acquire_ready)

    _role2_wait_early_pre(physical_thread)
    _role2_hidden_tail(
        pre_shared0,
        pre_shared1,
        new_residual_ptr,
        rmsnorm_weight_ptr,
        y_bf16_ptr,
        pid,
        rmsnorm_eps,
        rmsnorm_scale,
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
        DG_TLE_MHC_MID_SINKHORN_LOOP:
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
          @again bra.uni DG_TLE_MHC_MID_SINKHORN_LOOP;
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
def _role2_routing_finalize_explicit(
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
    token_active,
    ROLE: tl.constexpr,
):
    """Finalize one routing token and release its hidden subgroup."""
    projection_mask = lane < JIT_PROJECTION_OUTPUTS
    token_squared_sum = _t64._warp_broadcast_lane24_f32(packed_sum)
    projection_norm = tl.rsqrt(token_squared_sum / JIT_FLAT_HIDDEN + hc_norm_eps)
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
    # A normal TLE local_ptr store introduces a 128-thread worker-group
    # rendezvous in this compiler.  Only one physical warp owns each role2
    # token, so issue the lane-masked shared store directly instead.
    pre_pointer = tle.gpu.local_ptr(pre_shared, (lane,))
    pre_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[pre_pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    _role2_store_pre_and_optionally_arrive(pre_address, pre_mix, lane, ROLE)
    post_base = tl.cast(new_post_mix_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active, ge4, lt8, p;
          .reg .u32 index;
          .reg .u64 offset, address;
          setp.ge.u32 ge4, $3, 4;
          setp.lt.u32 lt8, $3, 8;
          and.pred p, ge4, lt8;
          setp.ne.u32 active, $4, 0;
          and.pred p, p, active;
          sub.u32 index, $3, 4;
          mul.wide.u32 offset, index, 4;
          add.u64 address, $1, offset;
          @p st.global.f32 [address], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r,r",
        args=[
            post_base,
            post_mix,
            lane,
            tl.cast(token_active, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    routing_mask = (lane >= 2 * JIT_HC_MULT) & (lane < JIT_PROJECTION_OUTPUTS)
    sinkhorn = tl.where(routing_mask, mixes, 0.0)
    row_max = _t64._warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _t64._warp_reduce4_sum(sinkhorn)
    sinkhorn = sinkhorn * _t64._fast_rcp_ftz_f32(row_sum)
    sinkhorn += sinkhorn_eps
    column_sum = _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
    sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
    sinkhorn = _sinkhorn_nineteen_warp_iterations(sinkhorn, sinkhorn_eps)
    comb_base = tl.cast(new_comb_res_mix_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active, ge8, lt24, p;
          .reg .u32 index;
          .reg .u64 offset, address;
          setp.ge.u32 ge8, $3, 8;
          setp.lt.u32 lt24, $3, 24;
          and.pred p, ge8, lt24;
          setp.ne.u32 active, $4, 0;
          and.pred p, p, active;
          sub.u32 index, $3, 8;
          mul.wide.u32 offset, index, 4;
          add.u64 address, $1, offset;
          @p st.global.f32 [address], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r,r",
        args=[
            comb_base,
            sinkhorn,
            lane,
            tl.cast(token_active, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _role2_route_one_token(
    partials_ptr,
    pre_shared,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    token,
    token_active,
    lane,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    ROLE: tl.constexpr,
):
    safe_token = tl.where(token_active, token, 0)
    # Production fixes the routing reduction to ILP4.
    sum0 = 0.0
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    for group in tl.static_range(0, JIT_NUM_SPLITS // 4):
        base_split = group * 4
        common_offset = safe_token * JIT_BLOCK_N + lane
        mask = (lane <= JIT_PROJECTION_OUTPUTS) & token_active
        sum0 += tl.load(
            partials_ptr
            + (base_split + 0) * JIT_PADDED_TOKENS * JIT_BLOCK_N
            + common_offset,
            mask=mask,
            other=0.0,
        )
        sum1 += tl.load(
            partials_ptr
            + (base_split + 1) * JIT_PADDED_TOKENS * JIT_BLOCK_N
            + common_offset,
            mask=mask,
            other=0.0,
        )
        sum2 += tl.load(
            partials_ptr
            + (base_split + 2) * JIT_PADDED_TOKENS * JIT_BLOCK_N
            + common_offset,
            mask=mask,
            other=0.0,
        )
        sum3 += tl.load(
            partials_ptr
            + (base_split + 3) * JIT_PADDED_TOKENS * JIT_BLOCK_N
            + common_offset,
            mask=mask,
            other=0.0,
        )
    packed_sum = (sum0 + sum1) + (sum2 + sum3)
    _role2_routing_finalize_explicit(
        pre_shared,
        mix_scales_ptr,
        mix_bases_ptr,
        new_post_mix_ptr + safe_token * JIT_HC_MULT,
        new_comb_res_mix_ptr + safe_token * JIT_HC_MULT * JIT_HC_MULT,
        packed_sum,
        lane,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        token_active,
        ROLE,
    )


@triton.jit
def _tile_split32_math_and_routing(
    partials_ptr,
    pipe_reader,
    producer_ready,
    grid_acquire_ready,
    pre_shared0,
    pre_shared1,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    split,
    pid,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    """Four WGMMA warps produce one tile, then reduce 32 partials."""
    accumulator = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), dtype=tl.float32)
    first = pipe_reader.wait(0)
    accumulator = tle.gpu.wgmma(
        first.slot.q,
        first.slot.fn,
        accumulator,
        input_precision="tf32",
        trans_b=True,
    )
    for iteration in tl.static_range(1, JIT_NUM_HIDDEN_CHUNKS * JIT_HC_MULT):
        wait = pipe_reader.wait(iteration)
        accumulator = tle.gpu.wgmma(
            wait.slot.q,
            wait.slot.fn,
            accumulator,
            input_precision="tf32",
            trans_b=True,
        )
        accumulator = tle.gpu.wgmma_wait(1, accumulator)
        pipe_reader.release(iteration - 1)
    accumulator = tle.gpu.wgmma_wait(0, accumulator)
    pipe_reader.release(JIT_NUM_HIDDEN_CHUNKS * JIT_HC_MULT - 1)

    tile = pid // JIT_NUM_SPLITS
    m = tile * JIT_BLOCK_M + tl.arange(0, JIT_BLOCK_M)
    token_active = m < JIT_LOGICAL_TOKENS
    n = tl.arange(0, JIT_BLOCK_N)
    workspace_offsets = (
        split * JIT_PADDED_TOKENS * JIT_BLOCK_N + m[:, None] * JIT_BLOCK_N + n[None, :]
    )
    tl.store(
        partials_ptr + workspace_offsets,
        accumulator,
        mask=token_active[:, None] & (n[None, :] < JIT_PROJECTION_OUTPUTS),
    )

    tle.gpu.barrier_wait(producer_ready)
    tle.gpu.barrier_wait(grid_acquire_ready)

    lane = tl.inline_asm_elementwise(
        asm="mov.u32 $0, %laneid;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    physical_thread = _t64._physical_thread_id_x()
    worker_warp = (physical_thread - JIT_CTA_THREADS) // 32
    if worker_warp == 0:
        token = pid
        _role2_route_one_token(
            partials_ptr,
            pre_shared0,
            mix_scales_ptr,
            mix_bases_ptr,
            new_post_mix_ptr,
            new_comb_res_mix_ptr,
            token,
            token < JIT_LOGICAL_TOKENS,
            lane,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sinkhorn_eps,
            0,
        )
    elif worker_warp == 1:
        token = pid + JIT_TAIL_OWNER_CTAS
        _role2_route_one_token(
            partials_ptr,
            pre_shared1,
            mix_scales_ptr,
            mix_bases_ptr,
            new_post_mix_ptr,
            new_comb_res_mix_ptr,
            token,
            token < JIT_LOGICAL_TOKENS,
            lane,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sinkhorn_eps,
            1,
        )


@triton.jit
def _sm90_mid_grid_pipe4_role2_wgmma_kernel(
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
    grid_counter_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
):
    pid = tl.program_id(0)
    tile = pid // JIT_NUM_SPLITS
    split = pid % JIT_NUM_SPLITS
    tl.static_assert(JIT_PADDED_TOKENS == 256)
    tl.static_assert(JIT_LOGICAL_TOKENS > 128)
    tl.static_assert(JIT_LOGICAL_TOKENS <= JIT_PADDED_TOKENS)
    tl.static_assert(JIT_GRID_CTAS == JIT_NUM_M_TILES * JIT_NUM_SPLITS)
    tl.static_assert(JIT_NUM_M_TILES == 3 or JIT_NUM_M_TILES == 4)
    tl.static_assert(JIT_NUM_SPLITS == 32)
    tl.static_assert(JIT_HIDDEN_PER_SPLIT == 128)
    tl.static_assert(JIT_HIDDEN_CHUNK == 64)
    tl.static_assert(JIT_NUM_HIDDEN_CHUNKS == 2)
    tl.static_assert(JIT_TOKENS_PER_OWNER == 2)
    tl.static_assert(JIT_ROLE2_ROOT_THREADS == 256)

    q_shared = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_BLOCK_M, JIT_HIDDEN_CHUNK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    fn_shared = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_BLOCK_N, JIT_HIDDEN_CHUNK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    projection_pipe = tle.pipe(
        capacity=JIT_PIPE_STAGES,
        scope="cta",
        name="mhc_projection_pipe4_role2_clean",
        q=q_shared,
        fn=fn_shared,
    )
    pre_shared0 = tle.gpu.alloc(
        [JIT_HC_MULT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pre_shared1 = tle.gpu.alloc(
        [JIT_HC_MULT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    producer_ready = tle.gpu.alloc_barrier(arrive_count=JIT_STAGED_LOGICAL_THREADS)
    grid_acquire_ready = tle.gpu.alloc_barrier(arrive_count=JIT_STAGED_LOGICAL_THREADS)
    tle.gpu.warp_specialize(
        [
            (
                _tile_split32_producer,
                (
                    x_ptr,
                    residual_ptr,
                    post_mix_ptr,
                    comb_res_mix_ptr,
                    fn_ptr,
                    new_residual_ptr,
                    partials_ptr,
                    projection_pipe.writer(),
                    producer_ready,
                    grid_acquire_ready,
                    pre_shared0,
                    pre_shared1,
                    rmsnorm_weight_ptr,
                    y_bf16_ptr,
                    grid_counter_ptr,
                    tile,
                    split,
                    pid,
                    rmsnorm_eps,
                    rmsnorm_scale,
                ),
            ),
            (
                _tile_split32_math_and_routing,
                (
                    partials_ptr,
                    projection_pipe.reader(),
                    producer_ready,
                    grid_acquire_ready,
                    pre_shared0,
                    pre_shared1,
                    mix_scales_ptr,
                    mix_bases_ptr,
                    new_post_mix_ptr,
                    new_comb_res_mix_ptr,
                    split,
                    pid,
                    hc_norm_eps,
                    hc_pre_eps,
                    hc_post_scale,
                    sinkhorn_eps,
                ),
            ),
        ],
        [JIT_STAGED_MATH_WARPS],
        [JIT_STAGED_MATH_REGS],
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
class MidGridPipe4Role2Plan:
    """Prepared one-launch plan; do not share it across streams."""

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
    partials: torch.Tensor
    grid_counter: torch.Tensor
    hc_norm_eps: float
    hc_pre_eps: float
    hc_post_scale: float
    sinkhorn_eps: float
    rmsnorm_eps: float
    rmsnorm_scale: float
    workspace_tensors: dict[str, torch.Tensor]
    counter_tensors: dict[str, torch.Tensor]
    metadata: dict[str, object]

    def _kernel_args(self) -> tuple[object, ...]:
        return (
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
            self.partials,
            self.grid_counter,
            self.hc_norm_eps,
            self.hc_pre_eps,
            self.hc_post_scale,
            self.sinkhorn_eps,
            self.rmsnorm_eps,
            self.rmsnorm_scale,
        )

    def compile(self):
        return _sm90_mid_grid_pipe4_role2_wgmma_kernel.warmup(
            *self._kernel_args(),
            grid=(GRID_CTAS,),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )

    def run(self) -> None:
        _sm90_mid_grid_pipe4_role2_wgmma_kernel[(GRID_CTAS,)](
            *self._kernel_args(),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> MidGridPipe4Role2Plan:
    """Validate the fixed profile and allocate capture-stable state."""
    if os.environ.get("DG_TLE_MHC_MID_GRID_PIPE4_ROLE2", "0") != "1":
        raise RuntimeError(
            "set DG_TLE_MHC_MID_GRID_PIPE4_ROLE2=1 to enable this experiment"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the plan before CUDA Graph capture")
    if inputs.get("shifted_prev_mix") is not None:
        raise ValueError("only Normal MHC (shifted_prev_mix=None) is supported")

    specs = (
        (
            inputs.get("x"),
            "inputs.x",
            (LOGICAL_TOKENS, HIDDEN),
            torch.bfloat16,
        ),
        (
            inputs.get("residual"),
            "inputs.residual",
            (LOGICAL_TOKENS, HC_MULT, HIDDEN),
            torch.bfloat16,
        ),
        (
            inputs.get("post_mix"),
            "inputs.post_mix",
            (LOGICAL_TOKENS, HC_MULT, 1),
            torch.float32,
        ),
        (
            inputs.get("comb_res_mix"),
            "inputs.comb_res_mix",
            (LOGICAL_TOKENS, HC_MULT, HC_MULT),
            torch.float32,
        ),
        (
            inputs.get("fn"),
            "inputs.fn",
            (PROJECTION_OUTPUTS, FLAT_HIDDEN),
            torch.float32,
        ),
        (inputs.get("mix_scales"), "inputs.mix_scales", (3,), torch.float32),
        (
            inputs.get("mix_bases"),
            "inputs.mix_bases",
            (PROJECTION_OUTPUTS,),
            torch.float32,
        ),
        (
            inputs.get("rmsnorm_weight"),
            "inputs.rmsnorm_weight",
            (HIDDEN,),
            torch.bfloat16,
        ),
        (
            outputs.get("new_residual"),
            "outputs.new_residual",
            (LOGICAL_TOKENS, HC_MULT, HIDDEN),
            torch.bfloat16,
        ),
        (
            outputs.get("new_post_mix"),
            "outputs.new_post_mix",
            (LOGICAL_TOKENS, HC_MULT, 1),
            torch.float32,
        ),
        (
            outputs.get("new_comb_res_mix"),
            "outputs.new_comb_res_mix",
            (LOGICAL_TOKENS, HC_MULT, HC_MULT),
            torch.float32,
        ),
        (
            outputs.get("y_bf16"),
            "outputs.y_bf16",
            (LOGICAL_TOKENS, HIDDEN),
            torch.bfloat16,
        ),
    )
    tensors = [
        _require_tensor(value, name=name, shape=shape, dtype=dtype)
        for value, name, shape, dtype in specs
    ]
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must be on the same CUDA device")

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
    partials = torch.empty(
        (NUM_SPLITS, PADDED_TOKENS, BLOCK_N),
        dtype=torch.float32,
        device=device,
    )
    grid_counter = torch.zeros((1,), dtype=torch.uint64, device=device)
    metadata: dict[str, object] = {
        "model": "DeepSeek-V4-Flash@60d8d707",
        "mode": "normal",
        "T": LOGICAL_TOKENS,
        "padded_T": PADDED_TOKENS,
        "H": HIDDEN,
        "R": HC_MULT,
        "P": PROJECTION_OUTPUTS,
        "kernel_count": 1,
        "expected_cuda_graph_nodes": 1,
        "kernel_symbol": "_sm90_mid_grid_pipe4_role2_wgmma_kernel",
        "CTA": GRID_CTAS,
        "grid_ctas": GRID_CTAS,
        "tail_owner_ctas": TAIL_OWNER_CTAS,
        "tokens_per_owner": TOKENS_PER_OWNER,
        "num_m_tiles": NUM_M_TILES,
        "num_splits": NUM_SPLITS,
        "block_m": BLOCK_M,
        "block_n": BLOCK_N,
        "hidden_per_split": HIDDEN_PER_SPLIT,
        "hidden_chunk": HIDDEN_CHUNK,
        "num_hidden_chunks": NUM_HIDDEN_CHUNKS,
        "projection_bk": HIDDEN_CHUNK,
        "projection_pipe8": PIPE_STAGES == 8,
        "wgmma_pipeline_slots": PIPE_STAGES,
        "producer_reuses_x_residual_across_routes": True,
        "tail_role2": True,
        "tail_role2_phase": ROLE2_PHASE,
        "tail_root_warps_per_token": ROLE2_ROOT_WARPS_PER_TOKEN,
        "tail_routing_warps_per_token": (2 if ROLE2_PHASE == "early_pre_pair2" else 1),
        "tail_full_sync_threads": (NUM_WARPS + STAGED_MATH_WARPS) * 32,
        "tail_early_pre_overlap": ROLE2_PHASE.startswith("early_pre"),
        "tail_layout_contract": ("default_h8192_permuted_to_two_h4096_subgroups"),
        "staged_math_regs": STAGED_MATH_REGS,
        "cta_mapping": "tile=pid//32,split=pid%32",
        "threads_per_cta_expected": (NUM_WARPS + STAGED_MATH_WARPS) * 32,
        "physical_cta_threads": (NUM_WARPS + STAGED_MATH_WARPS) * 32,
        "cooperative_launch": True,
        "full_grid_residency_required": True,
        "required_total_resident_ctas": GRID_CTAS,
        "occupancy_query_required_before_first_launch": True,
        "workspace_shape": list(partials.shape),
        "workspace_bytes": partials.numel() * partials.element_size(),
        "padding_requires_extra_kernel": False,
        "experimental": True,
        "dispatch_enabled": False,
    }
    return MidGridPipe4Role2Plan(
        x=tensors[0],
        residual=tensors[1],
        post_mix=tensors[2],
        comb_res_mix=tensors[3],
        fn=tensors[4],
        mix_scales=tensors[5],
        mix_bases=tensors[6],
        rmsnorm_weight=tensors[7],
        new_residual=tensors[8],
        new_post_mix=tensors[9],
        new_comb_res_mix=tensors[10],
        y_bf16=tensors[11],
        partials=partials,
        grid_counter=grid_counter,
        hc_norm_eps=float(inputs["hc_norm_eps"]),
        hc_pre_eps=float(inputs["hc_pre_eps"]),
        hc_post_scale=float(inputs["hc_post_scale"]),
        sinkhorn_eps=float(inputs["sinkhorn_eps"]),
        rmsnorm_eps=float(inputs["rmsnorm_eps"]),
        rmsnorm_scale=float(inputs["rmsnorm_scale"]),
        workspace_tensors={"projection_and_sqsum_partials": partials},
        counter_tensors={"monotonic_grid_barrier": grid_counter},
        metadata=metadata,
    )

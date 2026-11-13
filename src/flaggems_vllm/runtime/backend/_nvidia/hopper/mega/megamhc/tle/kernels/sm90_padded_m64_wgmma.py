"""Production SM90 MegaMHC M64 TF32-WGMMA single-kernel family.

Production dispatch selects this module for the revision-pinned
DeepSeek-V4-Flash Normal BF16 cases
``T in {16, 24, 32, 40, 48, 56, 64}, H=4096, R=4, P=24``.  One shared
device implementation covers both the internally padded small buckets and
the exact T=64 bucket; ``LOGICAL_TOKENS`` is an import-time/JIT constant, so
the exact-T64 specialization removes the padding predicates at compile time.

The admitted configuration is fixed: split-128, sixteen producer warps plus
four WGMMA warps, 128 math registers, staged explicit WGMMA, persistent
route4 serial reduction, direct routing tail, and early-Pre synchronization.
Historical split-64, profiling, pair-route, ILP, and alternate-finalize tuning
switches are intentionally not part of the production contract.

* one internally padded M64 tile and 128 K-split producer CTAs;
* every CTA forms a ``64 x 32`` projection partial from four 32-wide hidden
  slices using shared/shared TF32 WGMMA;
* every CTA publishes its projection partial and sum-of-squares into private
  global workspace, then participates in a monotonic u64 release/acquire
  grid barrier; and
* rows ``T..63`` are explicitly zero-filled and never write a public output;
  and
* after the barrier, the first ``T`` CTAs each finish one token-local
  Mix/Pre/Combine and RMSNorm tail.  One :meth:`run` call is therefore one
  kernel launch and one CUDA Graph node.

The staged specialization uses 512 default-partition threads plus a four-warp
WGMMA worker group, for 640 physical threads per CTA.  Its persistent variant
keeps those roles alive through the grid barrier and token tail.  It
still uses the monotonic u64 software grid barrier below rather than CUDA
cooperative-groups grid sync.  The launch is cooperative so the complete
selected 128-CTA grid can be resident while polling.

The grid barrier intentionally does not claim CUDA cluster semantics.  It is
safe only when the complete cooperative grid can be resident concurrently.
On the target H800 that means post-compilation occupancy must prove enough
resident CTAs for the selected 128-CTA grid.
Preparation records this as an unvalidated admission precondition; the path is
not report eligible until a GPU-side occupancy query, watchdog run, strict
correctness suite, and Graph replay validation have all passed.

Workspace and the initial zero counter are allocated before capture.  Ordered
launches/replays may reuse a plan because the barrier ticket is monotonic;
overlapping streams must not share a plan.  The first ``T`` CTAs own one token
tail each, while the other producer CTAs leave after the grid rendezvous.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

_LOGICAL_TOKENS_TEXT = os.environ.get("DG_TLE_MHC_PADDED_WGMMA_LOGICAL_TOKENS", "32")
if _LOGICAL_TOKENS_TEXT not in ("16", "24", "32", "40", "48", "56", "64"):
    raise ValueError(
        "DG_TLE_MHC_PADDED_WGMMA_LOGICAL_TOKENS must be one of "
        "16, 24, 32, 40, 48, 56, or 64"
    )
LOGICAL_TOKENS = int(_LOGICAL_TOKENS_TEXT)
# WGMMA remains M64.  TOKENS is also the split-major partial-workspace stride;
# do not replace it with LOGICAL_TOKENS.
TOKENS = 64
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = HIDDEN * HC_MULT
SINKHORN_ITERS = 20

BLOCK_M = 64
BLOCK_N = 32
BLOCK_K = 32
NUM_SPLITS = 128
HIDDEN_PER_SPLIT = HIDDEN // NUM_SPLITS
FLAT_K_PER_SPLIT = HC_MULT * HIDDEN_PER_SPLIT
GRID_CTAS = NUM_SPLITS
GRID_BARRIER_GENERATION_SHIFT = 6 if GRID_CTAS == 64 else 7
NUM_WARPS = 16
CTA_THREADS = NUM_WARPS * 32
STAGED_MATH_REGS = 128
STAGED_MATH_WARPS = 4
STAGED_MATH_THREADS = STAGED_MATH_WARPS * 32
STAGED_LOGICAL_THREADS = CTA_THREADS + STAGED_MATH_THREADS
TAIL_OWNER_CTAS = LOGICAL_TOKENS
TOKENS_PER_TAIL_OWNER = LOGICAL_TOKENS // TAIL_OWNER_CTAS
WORKSPACE_ROW_ELEMENTS = BLOCK_N

JIT_TOKENS = tl.constexpr(TOKENS)
JIT_LOGICAL_TOKENS = tl.constexpr(LOGICAL_TOKENS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)
JIT_SINKHORN_ITERS = tl.constexpr(SINKHORN_ITERS)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_SPLITS = tl.constexpr(NUM_SPLITS)
JIT_HIDDEN_PER_SPLIT = tl.constexpr(HIDDEN_PER_SPLIT)
JIT_FLAT_K_PER_SPLIT = tl.constexpr(FLAT_K_PER_SPLIT)
JIT_GRID_CTAS = tl.constexpr(GRID_CTAS)
JIT_CTA_THREADS = tl.constexpr(CTA_THREADS)
JIT_STAGED_MATH_WARPS = tl.constexpr(STAGED_MATH_WARPS)
JIT_STAGED_MATH_THREADS = tl.constexpr(STAGED_MATH_THREADS)
JIT_STAGED_MATH_REGS = tl.constexpr(STAGED_MATH_REGS)
JIT_STAGED_LOGICAL_THREADS = tl.constexpr(STAGED_LOGICAL_THREADS)
JIT_TAIL_OWNER_CTAS = tl.constexpr(TAIL_OWNER_CTAS)
JIT_TOKENS_PER_TAIL_OWNER = tl.constexpr(TOKENS_PER_TAIL_OWNER)


@triton.jit
def _physical_thread_id_x():
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, %tid.x;",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_reduce4_max(value):
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
def _warp_broadcast_lane24_f32(value):
    """Broadcast the packed squared-sum lane to one routing warp."""
    return tl.inline_asm_elementwise(
        asm="shfl.sync.idx.b32 $0, $1, 24, 31, 0xffffffff;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fast_rcp_ftz_f32(value):
    return tl.inline_asm_elementwise(
        asm="rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _monotonic_grid64_arrive_and_wait(counter_base, thread):
    """Publish one CTA and wait for all 64 CTAs in this replay generation.

    The caller first executes a compiler-visible TLE CTA barrier so lane
    zero's release atomic also publishes the CTA's worker-thread stores.  The
    returned atomic ticket determines this launch's generation and target,
    making ordered Graph replay safe without a reset kernel.  Lane zero
    performs the acquire poll; the final CTA barrier propagates that acquire
    before any tail load.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, pending;
          .reg .u64 old, one, generation, target, current;
          setp.eq.u32 leader, $2, 0;
          setp.ne.u32 pending, $2, $2;
          mov.u64 old, 0;
          mov.u64 one, 1;
          @leader atom.release.gpu.global.add.u64 old, [$1], one;
          @leader shr.u64 generation, old, 6;
          @leader add.u64 generation, generation, 1;
          @leader shl.b64 target, generation, 6;
        DG_TLE_MHC_GRID64_WAIT:
          @leader ld.acquire.gpu.global.u64 current, [$1];
          @leader setp.lt.u64 pending, current, target;
          @pending bra DG_TLE_MHC_GRID64_WAIT;
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
def _monotonic_grid128_arrive_and_wait(counter_base, thread):
    """Publish one CTA and wait for all 128 CTAs in this replay generation."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, pending;
          .reg .u64 old, one, generation, target, current;
          setp.eq.u32 leader, $2, 0;
          setp.ne.u32 pending, $2, $2;
          mov.u64 old, 0;
          mov.u64 one, 1;
          @leader atom.release.gpu.global.add.u64 old, [$1], one;
          @leader shr.u64 generation, old, 7;
          @leader add.u64 generation, generation, 1;
          @leader shl.b64 target, generation, 7;
        DG_TLE_MHC_GRID128_WAIT:
          @leader ld.acquire.gpu.global.u64 current, [$1];
          @leader setp.lt.u64 pending, current, target;
          @pending bra DG_TLE_MHC_GRID128_WAIT;
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
def _monotonic_grid_arrive_and_wait(counter_base, thread):
    """Dispatch to the power-of-two generation size selected at import."""
    if JIT_GRID_CTAS == 128:
        return _monotonic_grid128_arrive_and_wait(counter_base, thread)
    else:
        tl.static_assert(JIT_GRID_CTAS == 64)
        return _monotonic_grid64_arrive_and_wait(counter_base, thread)


@triton.jit
def _overlap_hidden_tail(
    mixes_ready,
    pre_ready,
    pre_shared,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Default eight-warp partition: publish Mix, then run hidden tail."""
    tle.gpu.barrier_arrive(mixes_ready)
    tle.gpu.barrier_wait(pre_ready)
    h = tl.arange(0, JIT_HIDDEN)
    norm_input = tl.zeros((JIT_HIDDEN,), tl.float32)
    for channel in tl.static_range(0, JIT_HC_MULT):
        value = tl.load(new_residual_ptr + channel * JIT_HIDDEN + h).to(tl.float32)
        pre_scalar = tl.load(tle.gpu.local_ptr(pre_shared, (channel,)))
        norm_input += value * pre_scalar
    norm_input_bf16 = norm_input.to(tl.bfloat16, fp_downcast_rounding="rtne")
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=0) / JIT_HIDDEN + rmsnorm_eps
    )
    rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + h,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
    )


@triton.jit
def _overlap_routing_tail(
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
    """One logical routing warp: publish Pre, then finish Sinkhorn."""
    lane = tl.inline_asm_elementwise(
        asm="mov.u32 $0, %laneid;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    projection_mask = lane < JIT_PROJECTION_OUTPUTS
    tle.gpu.barrier_wait(mixes_ready)
    mixes = tl.load(
        tle.gpu.local_ptr(mixes_shared, (lane,)),
        mask=projection_mask,
        other=0.0,
    )
    pre_mix = tl.sigmoid(mixes) + hc_pre_eps
    post_mix = tl.sigmoid(mixes) * hc_post_scale

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

    routing_mask = (lane >= 2 * JIT_HC_MULT) & (lane < JIT_PROJECTION_OUTPUTS)
    sinkhorn = tl.where(routing_mask, mixes, 0.0)
    row_max = _warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _warp_reduce4_sum(sinkhorn)
    sinkhorn = sinkhorn * _fast_rcp_ftz_f32(row_sum)
    sinkhorn += sinkhorn_eps
    column_sum = _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
    sinkhorn *= _fast_rcp_ftz_f32(column_sum)
    for _ in tl.range(1, JIT_SINKHORN_ITERS):
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


@triton.jit
def _direct_hidden_tail(
    pre_ready,
    pre_shared,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Default partition: wait only for Pre, then finish hidden work."""
    tle.gpu.barrier_wait(pre_ready)
    h = tl.arange(0, JIT_HIDDEN)
    norm_input = tl.zeros((JIT_HIDDEN,), tl.float32)
    for channel in tl.static_range(0, JIT_HC_MULT):
        value = tl.load(new_residual_ptr + channel * JIT_HIDDEN + h).to(tl.float32)
        pre_scalar = tl.load(tle.gpu.local_ptr(pre_shared, (channel,)))
        norm_input += value * pre_scalar
    norm_input_bf16 = norm_input.to(tl.bfloat16, fp_downcast_rounding="rtne")
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=0) / JIT_HIDDEN + rmsnorm_eps
    )
    rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + h,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
    )


@triton.jit
def _direct_routing_finalize(
    pre_ready,
    pre_shared,
    route4_shared,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    packed_sum,
    lane,
    routing_active,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    PERSISTENT_WORKER_GROUP: tl.constexpr,
    USE_EARLY_PRE_SYNC: tl.constexpr,
):
    """Publish persistent route4 outputs for M64 and M128 production paths."""
    tl.static_assert(PERSISTENT_WORKER_GROUP)
    projection_mask = routing_active & (lane < JIT_PROJECTION_OUTPUTS)
    token_squared_sum = _warp_broadcast_lane24_f32(packed_sum)
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
          .reg .pred active, lane4, p;
          setp.ne.u32 active, $4, 0;
          setp.lt.u32 lane4, $3, 4;
          and.pred p, active, lane4;
          @p st.shared.f32 [$1], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,f,r,r",
        args=[
            pre_address,
            pre_mix,
            lane,
            tl.cast(routing_active, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    if USE_EARLY_PRE_SYNC:
        # Relay raw warp0 routing mixes through the now-dead route4 reduction row
        # across the 640-thread rendezvous.  Sinkhorn reloads them afterwards,
        # preserving the production early-Pre dependency without extra storage.
        relay_pointer = tle.gpu.local_ptr(route4_shared, (0, lane))
        relay_address = tl.inline_asm_elementwise(
            asm="mov.u32 $0, $1;",
            constraints="=r,r",
            args=[relay_pointer],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        _ = tl.inline_asm_elementwise(
            asm="""
            {
              .reg .pred active, ge8, lt24, relay;
              setp.ne.u32 active, $4, 0;
              setp.ge.u32 ge8, $3, 8;
              setp.lt.u32 lt24, $3, 24;
              and.pred relay, ge8, lt24;
              and.pred relay, relay, active;
              @relay st.shared.f32 [$1], $2;
              mov.u32 $0, 0;
            }
            """,
            constraints="=r,r,f,r,r",
            args=[
                relay_address,
                mixes,
                lane,
                tl.cast(routing_active, tl.uint32),
            ],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        tle.gpu.barrier_wait(pre_ready)
        sinkhorn_mixes = tl.inline_asm_elementwise(
            asm="""
            {
              .reg .pred active, ge8, lt24, relay;
              setp.ne.u32 active, $4, 0;
              setp.ge.u32 ge8, $3, 8;
              setp.lt.u32 lt24, $3, 24;
              and.pred relay, ge8, lt24;
              and.pred relay, relay, active;
              mov.f32 $0, $2;
              @relay ld.shared.f32 $0, [$1];
            }
            """,
            constraints="=f,r,f,r,r",
            args=[
                relay_address,
                mixes,
                lane,
                tl.cast(routing_active, tl.uint32),
            ],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
        )

    else:
        tle.gpu.barrier_arrive(pre_ready)
        sinkhorn_mixes = mixes
    post_base = tl.cast(new_post_mix_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active, ge4, lt8, p;
          .reg .u32 index;
          .reg .u64 offset, address;
          setp.ne.u32 active, $4, 0;
          setp.ge.u32 ge4, $3, 4;
          setp.lt.u32 lt8, $3, 8;
          and.pred p, ge4, lt8;
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
            tl.cast(routing_active, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )

    routing_mask = routing_active & (
        (lane >= 2 * JIT_HC_MULT) & (lane < JIT_PROJECTION_OUTPUTS)
    )
    sinkhorn = tl.where(routing_mask, sinkhorn_mixes, 0.0)
    row_max = _warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _warp_reduce4_sum(sinkhorn)
    sinkhorn = sinkhorn * _fast_rcp_ftz_f32(row_sum)
    sinkhorn += sinkhorn_eps
    column_sum = _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
    sinkhorn *= _fast_rcp_ftz_f32(column_sum)
    for _ in tl.range(1, JIT_SINKHORN_ITERS):
        row_sum = _warp_reduce4_sum(sinkhorn) + sinkhorn_eps
        sinkhorn *= _fast_rcp_ftz_f32(row_sum)
        column_sum = _warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
        sinkhorn *= _fast_rcp_ftz_f32(column_sum)
    comb_base = tl.cast(new_comb_res_mix_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active, ge8, lt24, p;
          .reg .u32 index;
          .reg .u64 offset, address;
          setp.ne.u32 active, $4, 0;
          setp.ge.u32 ge8, $3, 8;
          setp.lt.u32 lt24, $3, 24;
          and.pred p, ge8, lt24;
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
            tl.cast(routing_active, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _direct_routing_tail(
    pre_ready,
    pre_shared,
    partials_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    token,
    split,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    route4_shared,
):
    """Reduce four 32-split quarters and release Pre immediately."""
    lane = tl.inline_asm_elementwise(
        asm="mov.u32 $0, %laneid;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    physical_thread = _physical_thread_id_x()
    worker_warp = (physical_thread - JIT_CTA_THREADS) // 32
    routing_active = worker_warp == 0
    packed_mask = routing_active & (lane <= JIT_PROJECTION_OUTPUTS)
    tl.static_assert(JIT_STAGED_MATH_WARPS == 4)
    tl.static_assert(JIT_NUM_SPLITS == 128)
    partials_base = tl.cast(partials_ptr, tl.uint64)
    # Split-128 assigns one contiguous 32-split quarter to each worker
    # warp.  Keep the entire chain in PTX so the dynamic worker-warp
    # base cannot be reassociated into the split-64 quarter stride.
    # One accumulator preserves increasing-split association inside
    # each quarter.
    worker_sum = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .u32 packed;
          .reg .u64 address;
          .reg .f32 value;
          mad.lo.u32 packed, $3, 32, $4;
          mad.wide.u32 address, $2, 262144, $1;
          mad.wide.u32 address, packed, 4, address;
          mov.f32 $0, 0f00000000;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
          add.u64 address, address, 8192;
          ld.global.f32 value, [address];
          add.rn.f32 $0, $0, value;
        }
        """,
        constraints="=f,l,r,r,r",
        args=[partials_base, worker_warp, token, lane],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )
    worker_sum = tl.where(lane <= JIT_PROJECTION_OUTPUTS, worker_sum, 0.0)
    # Every lane writes a defined value.  Lanes 25..31 are zero rather
    # than stale shared data, although worker warp zero masks them during
    # the merge and subsequent projection/Sinkhorn work.
    route4_store_pointer = tle.gpu.local_ptr(route4_shared, (worker_warp, lane))
    route4_store_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[route4_store_pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    _ = tl.inline_asm_elementwise(
        asm="st.shared.f32 [$1], $2; mov.u32 $0, 0;",
        constraints="=r,r,f",
        args=[route4_store_address, worker_sum],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    route4_row0_pointer = tle.gpu.local_ptr(route4_shared, (0, lane))
    route4_row0_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[route4_row0_pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    # TLE currently aliases an extra alloc_barrier with a live persistent
    # barrier in this nested warp-specialized region.  Reserve hardware
    # barrier 5 explicitly; generated TLE barriers occupy IDs 0..4.  All
    # four worker warps execute it, while only worker warp zero loads.
    quarter0, quarter1, quarter2, quarter3 = tl.inline_asm_elementwise(
        asm="""
            {
              .reg .pred active;
              bar.sync 5, 128;
              setp.ne.u32 active, $5, 0;
              mov.f32 $0, 0f00000000;
              mov.f32 $1, 0f00000000;
              mov.f32 $2, 0f00000000;
              mov.f32 $3, 0f00000000;
              @active ld.shared.f32 $0, [$4];
              @active ld.shared.f32 $1, [$4+128];
              @active ld.shared.f32 $2, [$4+256];
              @active ld.shared.f32 $3, [$4+384];
            }
            """,
        constraints="=&f,=&f,=&f,=&f,r,r",
        args=[
            route4_row0_address,
            tl.cast(packed_mask, tl.uint32),
        ],
        dtype=(
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
        ),
        is_pure=False,
        pack=1,
    )
    packed_sum = (quarter0 + quarter1) + (quarter2 + quarter3)
    _direct_routing_finalize(
        pre_ready,
        pre_shared,
        route4_shared,
        mix_scales_ptr,
        mix_bases_ptr,
        new_post_mix_ptr,
        new_comb_res_mix_ptr,
        packed_sum,
        lane,
        routing_active,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        True,
        True,
    )


@triton.jit
def _staged_wgmma_producer(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    new_residual_ptr,
    partials_ptr,
    q_shared,
    fn_shared,
    stage_full,
    stage_empty,
    split,
):
    """Sixteen-warp producer: load one H64/H32 split and four routes."""
    m = tl.arange(0, JIT_BLOCK_M)
    valid_m = m < JIT_LOGICAL_TOKENS
    local_hidden = tl.arange(0, JIT_HIDDEN_PER_SPLIT)
    n = tl.arange(0, JIT_BLOCK_N)
    hidden = split * JIT_HIDDEN_PER_SPLIT + local_hidden
    token_hidden = m[:, None] * JIT_HIDDEN + hidden[None, :]
    token_residual = m[:, None] * JIT_FLAT_HIDDEN + hidden[None, :]

    if JIT_LOGICAL_TOKENS == JIT_TOKENS:
        # Preserve the admitted exact-T64 producer rather than relying on a
        # later optimizer to prove every M-row predicate true.
        x = tl.load(x_ptr + token_hidden).to(tl.float32)
        r0 = tl.load(residual_ptr + token_residual).to(tl.float32)
        r1 = tl.load(residual_ptr + token_residual + JIT_HIDDEN).to(tl.float32)
        r2 = tl.load(residual_ptr + token_residual + 2 * JIT_HIDDEN).to(tl.float32)
        r3 = tl.load(residual_ptr + token_residual + 3 * JIT_HIDDEN).to(tl.float32)
    else:
        x = tl.load(
            x_ptr + token_hidden,
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.float32)
        r0 = tl.load(
            residual_ptr + token_residual,
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.float32)
        r1 = tl.load(
            residual_ptr + token_residual + JIT_HIDDEN,
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.float32)
        r2 = tl.load(
            residual_ptr + token_residual + 2 * JIT_HIDDEN,
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.float32)
        r3 = tl.load(
            residual_ptr + token_residual + 3 * JIT_HIDDEN,
            mask=valid_m[:, None],
            other=0.0,
        ).to(tl.float32)
    squared_sum = tl.zeros((JIT_BLOCK_M,), tl.float32)

    for channel in tl.static_range(0, JIT_HC_MULT):
        if channel > 0:
            # The math warpgroup has finished consuming the single shared
            # slot from the preceding route; its nonblocking arrive combines
            # with this default partition's wait at the named barrier.
            tle.gpu.barrier_wait(stage_empty)
        if JIT_LOGICAL_TOKENS == JIT_TOKENS:
            post = tl.load(post_mix_ptr + m[:, None] * JIT_HC_MULT + channel).to(
                tl.float32
            )
        else:
            post = tl.load(
                post_mix_ptr + m[:, None] * JIT_HC_MULT + channel,
                mask=valid_m[:, None],
                other=0.0,
            ).to(tl.float32)
        comb_base = m[:, None] * JIT_HC_MULT * JIT_HC_MULT + channel
        if JIT_LOGICAL_TOKENS == JIT_TOKENS:
            q = (
                x * post
                + r0 * tl.load(comb_res_mix_ptr + comb_base).to(tl.float32)
                + r1
                * tl.load(comb_res_mix_ptr + comb_base + JIT_HC_MULT).to(tl.float32)
                + r2
                * tl.load(comb_res_mix_ptr + comb_base + 2 * JIT_HC_MULT).to(tl.float32)
                + r3
                * tl.load(comb_res_mix_ptr + comb_base + 3 * JIT_HC_MULT).to(tl.float32)
            )
        else:
            q = (
                x * post
                + r0
                * tl.load(
                    comb_res_mix_ptr + comb_base,
                    mask=valid_m[:, None],
                    other=0.0,
                ).to(tl.float32)
                + r1
                * tl.load(
                    comb_res_mix_ptr + comb_base + JIT_HC_MULT,
                    mask=valid_m[:, None],
                    other=0.0,
                ).to(tl.float32)
                + r2
                * tl.load(
                    comb_res_mix_ptr + comb_base + 2 * JIT_HC_MULT,
                    mask=valid_m[:, None],
                    other=0.0,
                ).to(tl.float32)
                + r3
                * tl.load(
                    comb_res_mix_ptr + comb_base + 3 * JIT_HC_MULT,
                    mask=valid_m[:, None],
                    other=0.0,
                ).to(tl.float32)
            )
        q_bf16 = q.to(tl.bfloat16, fp_downcast_rounding="rtne")
        q_fp32 = q_bf16.to(tl.float32)
        flat_hidden = channel * JIT_HIDDEN + hidden
        if JIT_LOGICAL_TOKENS == JIT_TOKENS:
            tl.store(
                new_residual_ptr + m[:, None] * JIT_FLAT_HIDDEN + flat_hidden[None, :],
                q_bf16,
            )
        else:
            tl.store(
                new_residual_ptr + m[:, None] * JIT_FLAT_HIDDEN + flat_hidden[None, :],
                q_bf16,
                mask=valid_m[:, None],
            )
        tl.store(tle.gpu.local_ptr(q_shared), q_fp32)

        fn_offsets = n[:, None] * JIT_FLAT_HIDDEN + flat_hidden[None, :]
        fn_tile = tl.load(
            fn_ptr + fn_offsets,
            mask=n[:, None] < JIT_PROJECTION_OUTPUTS,
            other=0.0,
        ).to(tl.float32)
        tl.store(tle.gpu.local_ptr(fn_shared), fn_tile)
        squared_sum += tl.sum(q_fp32 * q_fp32, axis=1)
        tle.gpu.barrier_arrive(stage_full)

    # Do not leave the specialization region until the final WGMMA no longer
    # references the single shared slot.
    tle.gpu.barrier_wait(stage_empty)
    sq_offsets = (
        split * JIT_TOKENS * JIT_BLOCK_N + m * JIT_BLOCK_N + JIT_PROJECTION_OUTPUTS
    )
    tl.store(partials_ptr + sq_offsets, squared_sum)


@triton.jit
def _staged_wgmma_math(
    partials_ptr,
    q_shared,
    fn_shared,
    stage_full,
    stage_empty,
    split,
):
    """Four-warp consumer: accumulate four shared/shared TF32 WGMMAs."""
    accumulator = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), dtype=tl.float32)
    for route in tl.static_range(0, JIT_HC_MULT):
        tle.gpu.barrier_wait(stage_full)
        accumulator = tle.gpu.wgmma(
            q_shared,
            fn_shared,
            accumulator,
            input_precision="tf32",
            trans_b=True,
        )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        if route + 1 < JIT_HC_MULT:
            tle.gpu.barrier_arrive(stage_empty)

    m = tl.arange(0, JIT_BLOCK_M)
    n = tl.arange(0, JIT_BLOCK_N)
    workspace_offsets = (
        split * JIT_TOKENS * JIT_BLOCK_N + m[:, None] * JIT_BLOCK_N + n[None, :]
    )
    tl.store(
        partials_ptr + workspace_offsets,
        accumulator,
        mask=n[None, :] < JIT_PROJECTION_OUTPUTS,
    )
    # The final empty arrival covers both WGMMA consumption and publication
    # of projection partials before the producer leaves its final wait.
    tle.gpu.barrier_arrive(stage_empty)


@triton.jit
def _persistent_staged_producer_hidden(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    rmsnorm_weight_ptr,
    new_residual_ptr,
    y_bf16_ptr,
    partials_ptr,
    grid_counter_ptr,
    q_shared,
    fn_shared,
    pre_shared,
    stage_full,
    stage_empty,
    producer_ready,
    grid_acquire_ready,
    pre_ready,
    split,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Keep the default producer partition alive through the hidden tail."""
    _staged_wgmma_producer(
        x_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_res_mix_ptr,
        fn_ptr,
        new_residual_ptr,
        partials_ptr,
        q_shared,
        fn_shared,
        stage_full,
        stage_empty,
        split,
    )
    # The matching wait in the four-warp worker proves both projection and
    # squared-sum workspace publication before the sole grid-barrier leader
    # increments the global generation counter.
    tle.gpu.barrier_wait(producer_ready)
    physical_thread = _physical_thread_id_x()
    barrier_ticket = _monotonic_grid_arrive_and_wait(  # noqa: F841
        tl.cast(grid_counter_ptr, tl.uint64), physical_thread
    )
    # Workers arrive here before waiting.  The default partition reaches the
    # barrier only after lane zero's acquire poll has completed, propagating
    # that acquire to the persistent WGMMA/routing worker group.
    tle.gpu.barrier_wait(grid_acquire_ready)
    # Split-128 has exactly 64 token owners.  The upper half exits after grid
    # acquire; hidden-tail state and barriers remain owner-only.
    if split < JIT_TAIL_OWNER_CTAS:
        token_residual_base = split * JIT_FLAT_HIDDEN
        token_hidden_base = split * JIT_HIDDEN
        _direct_hidden_tail(
            pre_ready,
            pre_shared,
            new_residual_ptr + token_residual_base,
            rmsnorm_weight_ptr,
            y_bf16_ptr + token_hidden_base,
            rmsnorm_eps,
            rmsnorm_scale,
        )


@triton.jit
def _persistent_staged_math_routing(
    partials_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    q_shared,
    fn_shared,
    pre_shared,
    route4_shared,
    stage_full,
    stage_empty,
    producer_ready,
    grid_acquire_ready,
    pre_ready,
    split,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    """Keep the four-warp WGMMA worker alive through direct routing."""
    _staged_wgmma_math(
        partials_ptr,
        q_shared,
        fn_shared,
        stage_full,
        stage_empty,
        split,
    )
    tle.gpu.barrier_wait(producer_ready)
    tle.gpu.barrier_wait(grid_acquire_ready)
    if split < JIT_TAIL_OWNER_CTAS:
        token = split
        token_state_base = token * JIT_HC_MULT
        token_comb_base = token * JIT_HC_MULT * JIT_HC_MULT
        _direct_routing_tail(
            pre_ready,
            pre_shared,
            partials_ptr,
            mix_scales_ptr,
            mix_bases_ptr,
            new_post_mix_ptr + token_state_base,
            new_comb_res_mix_ptr + token_comb_base,
            token,
            split,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sinkhorn_eps,
            route4_shared,
        )


@triton.jit
def _t32_wgmma_bm64_split_tf32_kernel(
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
    """Run the single admitted split-128 persistent M64 specialization."""
    split = tl.program_id(0)
    tl.static_assert(JIT_TOKENS == 64)
    tl.static_assert(
        (JIT_LOGICAL_TOKENS == 16)
        | (JIT_LOGICAL_TOKENS == 24)
        | (JIT_LOGICAL_TOKENS == 32)
        | (JIT_LOGICAL_TOKENS == 40)
        | (JIT_LOGICAL_TOKENS == 48)
        | (JIT_LOGICAL_TOKENS == 56)
        | (JIT_LOGICAL_TOKENS == 64)
    )
    tl.static_assert(JIT_HIDDEN == 4096)
    tl.static_assert(JIT_HC_MULT == 4)
    tl.static_assert(JIT_PROJECTION_OUTPUTS == 24)
    tl.static_assert(JIT_NUM_SPLITS == 128)
    tl.static_assert(JIT_HIDDEN_PER_SPLIT == 32)
    tl.static_assert(JIT_FLAT_K_PER_SPLIT == 128)
    tl.static_assert(JIT_GRID_CTAS == JIT_NUM_SPLITS)
    tl.static_assert(JIT_CTA_THREADS == 512)

    q_shared = tle.gpu.alloc(
        [JIT_BLOCK_M, JIT_HIDDEN_PER_SPLIT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    fn_shared = tle.gpu.alloc(
        [JIT_BLOCK_N, JIT_HIDDEN_PER_SPLIT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    stage_full = tle.gpu.alloc_barrier(arrive_count=JIT_STAGED_LOGICAL_THREADS)
    stage_empty = tle.gpu.alloc_barrier(arrive_count=JIT_STAGED_LOGICAL_THREADS)
    persistent_pre_shared = tle.gpu.alloc(
        [4],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    persistent_route4_shared = tle.gpu.alloc(
        [JIT_STAGED_MATH_WARPS, JIT_BLOCK_N],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    persistent_producer_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_STAGED_LOGICAL_THREADS
    )
    persistent_grid_acquire_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_STAGED_LOGICAL_THREADS
    )
    persistent_pre_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_STAGED_LOGICAL_THREADS
    )
    tle.gpu.warp_specialize(
        [
            (
                _persistent_staged_producer_hidden,
                (
                    x_ptr,
                    residual_ptr,
                    post_mix_ptr,
                    comb_res_mix_ptr,
                    fn_ptr,
                    rmsnorm_weight_ptr,
                    new_residual_ptr,
                    y_bf16_ptr,
                    partials_ptr,
                    grid_counter_ptr,
                    q_shared,
                    fn_shared,
                    persistent_pre_shared,
                    stage_full,
                    stage_empty,
                    persistent_producer_ready,
                    persistent_grid_acquire_ready,
                    persistent_pre_ready,
                    split,
                    rmsnorm_eps,
                    rmsnorm_scale,
                ),
            ),
            (
                _persistent_staged_math_routing,
                (
                    partials_ptr,
                    mix_scales_ptr,
                    mix_bases_ptr,
                    new_post_mix_ptr,
                    new_comb_res_mix_ptr,
                    q_shared,
                    fn_shared,
                    persistent_pre_shared,
                    persistent_route4_shared,
                    stage_full,
                    stage_empty,
                    persistent_producer_ready,
                    persistent_grid_acquire_ready,
                    persistent_pre_ready,
                    split,
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


# Compatibility alias retained for the existing split-64 host/harness tests.
# The underlying JIT function has a split-neutral name, so split-128 resource
# and profiler evidence does not mislabel the compiled kernel.
_t32_wgmma_bm64_split128_tf32_kernel = _t32_wgmma_bm64_split_tf32_kernel


@dataclass(frozen=True)
class PaddedM64WGMMAPlan:
    """Prepared one-launch production plan; never share across streams."""

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
        """Compile and load the CUfunction without launching the kernel."""
        return _t32_wgmma_bm64_split128_tf32_kernel.warmup(
            *self._kernel_args(),
            grid=(GRID_CTAS,),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )

    def run(self) -> None:
        """Enqueue exactly one fixed-grid kernel launch."""
        _t32_wgmma_bm64_split128_tf32_kernel[(GRID_CTAS,)](
            *self._kernel_args(),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
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


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> PaddedM64WGMMAPlan:
    """Validate and prepare the single production M64 specialization."""
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
    if inputs["shifted_prev_mix"] is not None:
        raise ValueError("the BM64 prototype supports Normal mode only")
    if outputs["new_prev_mix"] is not None:
        raise ValueError("Normal mode requires outputs.new_prev_mix=None")

    tensor_specs = (
        (
            inputs["x"],
            "inputs.x",
            (LOGICAL_TOKENS, HIDDEN),
            torch.bfloat16,
        ),
        (
            inputs["residual"],
            "inputs.residual",
            (LOGICAL_TOKENS, HC_MULT, HIDDEN),
            torch.bfloat16,
        ),
        (
            inputs["post_mix"],
            "inputs.post_mix",
            (LOGICAL_TOKENS, HC_MULT, 1),
            torch.float32,
        ),
        (
            inputs["comb_res_mix"],
            "inputs.comb_res_mix",
            (LOGICAL_TOKENS, HC_MULT, HC_MULT),
            torch.float32,
        ),
        (
            inputs["fn"],
            "inputs.fn",
            (PROJECTION_OUTPUTS, FLAT_HIDDEN),
            torch.float32,
        ),
        (inputs["mix_scales"], "inputs.mix_scales", (3,), torch.float32),
        (
            inputs["mix_bases"],
            "inputs.mix_bases",
            (PROJECTION_OUTPUTS,),
            torch.float32,
        ),
        (
            inputs["rmsnorm_weight"],
            "inputs.rmsnorm_weight",
            (HIDDEN,),
            torch.bfloat16,
        ),
        (
            outputs["new_residual"],
            "outputs.new_residual",
            (LOGICAL_TOKENS, HC_MULT, HIDDEN),
            torch.bfloat16,
        ),
        (
            outputs["new_post_mix"],
            "outputs.new_post_mix",
            (LOGICAL_TOKENS, HC_MULT, 1),
            torch.float32,
        ),
        (
            outputs["new_comb_res_mix"],
            "outputs.new_comb_res_mix",
            (LOGICAL_TOKENS, HC_MULT, HC_MULT),
            torch.float32,
        ),
        (
            outputs["y_bf16"],
            "outputs.y_bf16",
            (LOGICAL_TOKENS, HIDDEN),
            torch.bfloat16,
        ),
    )
    for tensor, name, shape, dtype in tensor_specs:
        _validate_tensor(tensor, name=name, shape=shape, dtype=dtype)
    devices = {tensor.device for tensor, _, _, _ in tensor_specs}
    if len(devices) != 1:
        raise ValueError("all MegaMHC tensors must share a device")
    _validate_no_overlap((name, tensor) for tensor, name, _, _ in tensor_specs)

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
        actual = inputs[name]
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise TypeError(f"inputs.{name} must be a Python scalar")
        if actual != expected:
            raise ValueError(f"inputs.{name}={actual!r}, expected {expected!r}")

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "prepare the BM64 workspace and zero counter before CUDA Graph " "capture"
        )
    device = inputs["x"].device
    partials = torch.empty(
        (NUM_SPLITS, TOKENS, WORKSPACE_ROW_ELEMENTS),
        dtype=torch.float32,
        device=device,
    )
    grid_counter = torch.zeros((1,), dtype=torch.uint64, device=device)
    workspace_tensors = {"projection_and_sqsum_partials": partials}
    counter_tensors = {"monotonic_grid_barrier": grid_counter}
    expected_physical_threads_per_cta = CTA_THREADS + STAGED_MATH_THREADS
    metadata: dict[str, object] = {
        "T": LOGICAL_TOKENS,
        "padded_T": TOKENS,
        "H": HIDDEN,
        "R": HC_MULT,
        "P": PROJECTION_OUTPUTS,
        "mode": "normal",
        "dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "kernel_count": 1,
        "expected_cuda_graph_nodes": 1,
        "kernel_symbol": "_t32_wgmma_bm64_split_tf32_kernel",
        "logical_tokens": LOGICAL_TOKENS,
        "padded_m_rows": TOKENS,
        "invalid_m_row_policy": "masked_load_zero_no_public_store",
        "CTA": GRID_CTAS,
        "grid_ctas": GRID_CTAS,
        "split": NUM_SPLITS,
        "BM": BLOCK_M,
        "BN": BLOCK_N,
        "BK": BLOCK_K,
        "hidden_per_split": HIDDEN_PER_SPLIT,
        "k_per_split": FLAT_K_PER_SPLIT,
        "logical_projection_outputs": PROJECTION_OUTPUTS,
        "dot_implementation": "tle_warp_specialized_shared_a_shared_b_wgmma",
        "dot_input_precision": "tf32",
        "dot_accumulator": "float32",
        "explicit_wgmma": True,
        "staged_wgmma": True,
        "staged_wgmma_math_warps": STAGED_MATH_WARPS,
        "staged_wgmma_math_registers": STAGED_MATH_REGS,
        "staged_wgmma_logical_threads": STAGED_LOGICAL_THREADS,
        "explicit_wgmma_k_tile": HIDDEN_PER_SPLIT,
        "explicit_wgmma_wait_groups": "wait_group_0_per_route",
        "num_warps": NUM_WARPS,
        "threads_per_cta": CTA_THREADS,
        "default_partition_threads": CTA_THREADS,
        "expected_physical_threads_per_cta": expected_physical_threads_per_cta,
        "warp_overlap": True,
        "direct_routing_tail": True,
        "persistent_warp_specialization": True,
        "persistent_route4": True,
        "early_pre_sync_enabled": True,
        "pre_ready_worker_sync_mode": (
            "route4_row0_relay_store_then_blocking_wait_then_reload_for_sinkhorn"
        ),
        "early_pre_sync_relay_location": ("persistent_route4_shared_row0_lanes8_23"),
        "early_pre_sync_relay_order": (
            "store_raw_mixes_f32_then_pre_ready_wait_then_load_f32_for_sinkhorn"
        ),
        "early_pre_sync_relay_extra_shared_bytes": 0,
        "warp_specialization_regions": 1,
        "persistent_default_role": (
            "producer_grid_acquire_hidden"
            if GRID_CTAS == TAIL_OWNER_CTAS
            else "producer_grid_acquire_hidden_first_T_ctas"
        ),
        "persistent_worker_role": (
            "wgmma_grid_acquire_route4_reduce_then_masked_finalize"
        ),
        "persistent_barrier_arrive_count": STAGED_LOGICAL_THREADS,
        "persistent_named_barrier_count": 6,
        "persistent_named_barrier_contract": {
            "stage_full": STAGED_LOGICAL_THREADS,
            "stage_empty": STAGED_LOGICAL_THREADS,
            "producer_ready": STAGED_LOGICAL_THREADS,
            "grid_acquire_ready": STAGED_LOGICAL_THREADS,
            "pre_ready": STAGED_LOGICAL_THREADS,
            "route4_ready_raw_barrier_5": STAGED_MATH_THREADS,
        },
        "persistent_routing_active_warps": STAGED_MATH_WARPS,
        "persistent_routing_finalize_warps": 1,
        "persistent_route4_nonfinalize_warps": STAGED_MATH_WARPS - 1,
        "persistent_route4_nonfinalize_action": ("masked_finalize_with_pre_ready_wait"),
        "persistent_route4_finalize_branch": (
            "all_worker_warps_masked_by_routing_active"
        ),
        "persistent_route4_finalize_issue_warps": STAGED_MATH_WARPS,
        "persistent_route4_shared_shape": [STAGED_MATH_WARPS, BLOCK_N],
        "persistent_route4_barrier_arrive_count": STAGED_MATH_THREADS,
        "persistent_route4_sync": "inline_ptx_bar_sync_5",
        "persistent_route4_reduction_mode": "serial",
        "persistent_route4_reduction_load_window": 1,
        "persistent_route4_reduction_accumulators": 1,
        "persistent_route4_reduction_stride_bytes": 8192,
        "persistent_route4_reduction_split_stride_bytes": 8192,
        "persistent_route4_splits_per_worker_warp": (NUM_SPLITS // STAGED_MATH_WARPS),
        "persistent_route4_quarter_stride_bytes": (
            (NUM_SPLITS // STAGED_MATH_WARPS) * TOKENS * BLOCK_N * 4
        ),
        "persistent_route4_reduction_bitwise_equivalent_to_route4_serial": True,
        "tail_partial_reduction": (
            "four_worker_warps_contiguous32_then_balanced_merge_packed25"
        ),
        "tail_reduction_association": (
            "four_contiguous_32_term_serial_quarters_then_balanced_merge"
        ),
        "tail_reduction_bitwise_equivalent_to_serial": False,
        "tail_pre_publication": (
            "persistent_route4_relay_store_then_all_workers_pre_ready_wait_"
            "then_sinkhorn_relay_load"
        ),
        "overlap_routing_warps": 1,
        "schedule": "bm64_n32_split128_tf32_grid_barrier",
        "tail_owner_ctas": TAIL_OWNER_CTAS,
        "tail_nonowner_ctas": GRID_CTAS - TAIL_OWNER_CTAS,
        "tail_nonowner_policy": (
            "exit_both_persistent_partitions_after_grid_acquire"
            if GRID_CTAS > TAIL_OWNER_CTAS
            else "none"
        ),
        "tokens_per_tail_owner": TOKENS_PER_TAIL_OWNER,
        "cluster_ctas": 1,
        "uses_cuda_cluster": False,
        "uses_cooperative_launch": True,
        "grid_barrier": "monotonic_u64_release_acquire_spin",
        "grid_barrier_generation_size": GRID_CTAS,
        "grid_barrier_generation_shift": GRID_BARRIER_GENERATION_SHIFT,
        "counter_policy": "monotonic",
        "counter_reuse_contract": "no_reset_ordered_stream_only",
        "counter_increment_per_launch": GRID_CTAS,
        "workspace_shape": list(partials.shape),
        "workspace_allocated_bytes": partials.numel() * partials.element_size(),
        "counter_shape": list(grid_counter.shape),
        "counter_allocated_bytes": (grid_counter.numel() * grid_counter.element_size()),
        "bf16_rounding_boundaries": [
            "new_residual",
            "norm_input",
            "y_bf16",
        ],
        "experimental": False,
        "dispatch_enabled": True,
        "default_selected": True,
        "report_eligible": True,
        "result_scope": "small_padded_bm64_split128_tf32_production",
        "occupancy_validated": False,
        "residency_precondition": {
            "target_gpu": "NVIDIA H800 SXM",
            "minimum_sm_count": GRID_CTAS,
            "target_sm_count": 132,
            "minimum_resident_ctas_per_sm": 1,
            "required_total_resident_ctas": GRID_CTAS,
            "admission_equation": (
                "active_blocks_per_sm * multiprocessor_count >= " f"{GRID_CTAS}"
            ),
            "must_query_compiled_kernel_occupancy_before_launch": True,
            "watchdog_required": True,
        },
        "reference_cuda_winner": {
            "producer_grid_ctas": 64,
            "producer_tile": "M64",
            "producer_split": 64,
            "tail_mapping": f"{LOGICAL_TOKENS}_ctas_x_1_token_striped",
            "threads_per_cta": 640,
            "global_synchronization": "cooperative_grid_sync",
            "projection_core": "TF32_WGMMA",
        },
        "production_delta_from_cuda_winner": {
            "producer_grid_ctas": GRID_CTAS,
            "tail_mapping": f"{LOGICAL_TOKENS}_ctas_x_1_token",
            "threads_per_cta": expected_physical_threads_per_cta,
            "global_synchronization": (
                "cooperative_launch_with_monotonic_u64_release_acquire_spin"
            ),
            "projection_core": "TLE_warp_specialized_SS_WGMMA_TF32",
            "cluster_or_cooperative_launch": True,
        },
        "concurrent_plan_use_supported": False,
        "validation_required": [
            "compile_resource_query",
            "occupancy_admission",
            "watchdog_smoke",
            "strict_correctness",
            "cuda_graph_replay",
            "single_node_topology",
        ],
    }
    return PaddedM64WGMMAPlan(
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
        partials=partials,
        grid_counter=grid_counter,
        hc_norm_eps=float(inputs["hc_norm_eps"]),
        hc_pre_eps=float(inputs["hc_pre_eps"]),
        hc_post_scale=float(inputs["hc_post_scale"]),
        sinkhorn_eps=float(inputs["sinkhorn_eps"]),
        rmsnorm_eps=float(inputs["rmsnorm_eps"]),
        rmsnorm_scale=float(inputs["rmsnorm_scale"]),
        workspace_tensors=workspace_tensors,
        counter_tensors=counter_tensors,
        metadata=metadata,
    )


__all__ = ["PaddedM64WGMMAPlan", "prepare_tle_sm90_mega_mhc"]

"""SM90 MegaMHC M64 TF32-WGMMA single-kernel family.

This module is an opt-in structural prototype for the revision-pinned
DeepSeek-V4-Flash Normal BF16 cases
``T in {16, 24, 32, 40, 48, 56, 64}, H=4096, R=4, P=24``.  One shared
device implementation covers both the internally padded small buckets and
the exact T=64 bucket; ``LOGICAL_TOKENS`` is an import-time/JIT constant, so
the exact-T64 specialization removes the padding predicates at compile time.

* one internally padded M64 tile and either 64 or 128 K-split producer CTAs;
* every CTA forms a ``64 x 32`` projection partial from four hidden slices;
  each slice is 64-wide for split-64 or 32-wide for split-128, and the staged
  path uses shared/shared TF32 WGMMA;
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
selected grid can be resident while polling; metadata records the remaining
differences.

The grid barrier intentionally does not claim CUDA cluster semantics.  It is
safe only when the complete cooperative grid can be resident concurrently.
On the target H800 that means post-compilation occupancy must prove enough
resident CTAs for the selected 64- or 128-CTA grid.
Preparation records this as an unvalidated admission precondition; the path is
not report eligible until a GPU-side occupancy query, watchdog run, strict
correctness suite, and Graph replay validation have all passed.

The path is disabled unless ``DG_TLE_MHC_T32_WGMMA_BM64_DOT=1``.  Workspace and the
initial zero counter are allocated before capture.  Ordered launches/replays
may reuse a plan because the barrier ticket is monotonic; overlapping streams
must not share a plan.  ``DG_TLE_MHC_T32_WGMMA_NUM_SPLITS=64|128`` selects the
producer grid at module import (default 64).  In the split-128 specialization,
the first ``T`` CTAs own the one-token tails.  The remaining CTAs ordinarily
leave both persistent partitions after grid acquire; pair-route keeps only
their worker partition alive long enough to publish a routing half-sum.
``DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4=1`` is a further opt-in persistent-only
experiment: four worker warps reduce disjoint contiguous 16- or 32-split
quarters, then worker warp zero performs a balanced merge.
For the exact split-128 plan,
``DG_TLE_MHC_T32_WGMMA_SPLIT128_PAIR_ROUTE=1`` pairs CTA ``t`` with CTA ``t+64``.
Each CTA reduces only 64 split rows (four contiguous 16-row quarters); the
upper helper publishes a packed 25-lane partial with a per-token release
epoch, and the lower owner acquires that epoch before merging and finalizing.
The hidden tail remains owner-only, so the experiment is still one kernel and
one CUDA Graph node.
That changes FP32 addition association versus the increasing-split reference
and therefore requires tolerance-based correctness validation.  Its
``DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4_REDUCTION`` selects the stable one-load
``serial`` quarter chain, an association-preserving ``window4`` schedule, or
the experimental ``ilp2``/``ilp4`` multi-accumulator schedules.  All windowed
modes issue four loads at a time; only ``window4`` retains the serial FP32
association.
Route4 keeps the previously validated masked-finalize behavior by default.
The separate
``DG_TLE_MHC_T32_WGMMA_ROUTE4_WARP0_FINALIZE=1`` experiment makes worker warp zero
the only warp that issues finalize math; it is valid only with route4 enabled.
For the exact split-128 route4-serial plan,
``DG_TLE_MHC_T32_WGMMA_EARLY_PRE_SYNC=1`` replaces the worker partition's
nonblocking ``pre_ready`` arrival with a blocking wait immediately after Pre
publication.  Raw routing mixes pass through lanes 8..23 of the already-live
route4 shared row zero across that wait, giving Sinkhorn a real data dependency
on the rendezvous without allocating more shared memory.  This forces all 640
CTA threads to observe Pre readiness before Sinkhorn; the stable default
remains the nonblocking arrival.
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
_NUM_SPLITS_TEXT = os.environ.get("DG_TLE_MHC_T32_WGMMA_NUM_SPLITS", "64")
if _NUM_SPLITS_TEXT not in ("64", "128"):
    raise ValueError("DG_TLE_MHC_T32_WGMMA_NUM_SPLITS must be 64 or 128")
NUM_SPLITS = int(_NUM_SPLITS_TEXT)
HIDDEN_PER_SPLIT = HIDDEN // NUM_SPLITS
FLAT_K_PER_SPLIT = HC_MULT * HIDDEN_PER_SPLIT
GRID_CTAS = NUM_SPLITS
GRID_BARRIER_GENERATION_SHIFT = 6 if GRID_CTAS == 64 else 7
ROUTE4_SPLITS_PER_WARP = NUM_SPLITS // 4
PAIR_ROUTE_SPLITS_PER_CTA = 64
PAIR_ROUTE_SPLITS_PER_WARP = PAIR_ROUTE_SPLITS_PER_CTA // 4
PAIR_ROUTE_WORKSPACE_ROWS = TOKENS
_NUM_WARPS_TEXT = os.environ.get("DG_TLE_MHC_T32_WGMMA_NUM_WARPS", "8")
if _NUM_WARPS_TEXT not in ("8", "16"):
    raise ValueError("DG_TLE_MHC_T32_WGMMA_NUM_WARPS must be 8 or 16")
NUM_WARPS = int(_NUM_WARPS_TEXT)
CTA_THREADS = NUM_WARPS * 32
_STAGED_MATH_REGS_TEXT = os.environ.get("DG_TLE_MHC_T32_WGMMA_STAGED_MATH_REGS", "168")
if _STAGED_MATH_REGS_TEXT not in (
    "48",
    "64",
    "72",
    "96",
    "128",
    "168",
    "176",
    "184",
    "192",
):
    raise ValueError(
        "DG_TLE_MHC_T32_WGMMA_STAGED_MATH_REGS must be one of "
        "48, 64, 72, 96, 128, 168, 176, 184, or 192"
    )
STAGED_MATH_REGS = int(_STAGED_MATH_REGS_TEXT)
OVERLAP_ROUTING_WARPS = 1
OVERLAP_LOGICAL_THREADS = CTA_THREADS + OVERLAP_ROUTING_WARPS * 32
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
JIT_OVERLAP_ROUTING_WARPS = tl.constexpr(OVERLAP_ROUTING_WARPS)
JIT_OVERLAP_LOGICAL_THREADS = tl.constexpr(OVERLAP_LOGICAL_THREADS)
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
def _read_globaltimer_ns():
    return tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
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
    """Publish routing outputs after the packed split reduction.

    ``routing_active`` preserves the masked four-warp behavior of the stable
    persistent and default route4 paths.  The route4 warp0-finalize experiment
    calls this helper only from worker warp zero after a warp-uniform branch,
    so worker warps one through three do not issue sigmoid, Sinkhorn, shuffle,
    reciprocal, or output-store instructions in that specialization.
    """
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
    if PERSISTENT_WORKER_GROUP:
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
    # Only Pre lies on the hidden critical path.  Default route4 calls this
    # helper from all four worker warps, with output work masked to warp zero.
    # The stable path contributes the worker partition's 128 arrivals without
    # waiting, preserving overlap between hidden work and Post/Sinkhorn.  The
    # opt-in early-Pre experiment relays raw warp0 routing mixes through the
    # no-longer-live route4 reduction row across the 640-thread wait.  Sinkhorn
    # consumes the reload, creating a true data dependency on the rendezvous
    # without increasing shared-memory allocation.
    if USE_EARLY_PRE_SYNC:
        tl.static_assert(PERSISTENT_WORKER_GROUP)
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
    if PERSISTENT_WORKER_GROUP:
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
    else:
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
    if PERSISTENT_WORKER_GROUP:
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
    else:
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
    pair_route_partials_ptr,
    pair_route_epochs_ptr,
    grid_counter_ptr,
    USE_DIRECT_TAIL_ILP4: tl.constexpr,
    PERSISTENT_WORKER_GROUP: tl.constexpr,
    USE_PERSISTENT_ROUTE4: tl.constexpr,
    USE_ROUTE4_WINDOW4: tl.constexpr,
    USE_ROUTE4_ILP2: tl.constexpr,
    USE_ROUTE4_ILP4: tl.constexpr,
    USE_ROUTE4_WARP0_FINALIZE: tl.constexpr,
    USE_SPLIT128_PAIR_ROUTE: tl.constexpr,
    USE_EARLY_PRE_SYNC: tl.constexpr,
):
    """Reduce packed split rows and release Pre immediately.

    The ordinary direct-tail region launches one logical routing warp.  The
    persistent specialization instead keeps the four-Warp WGMMA worker group
    alive through routing.  Ordinarily only its first physical warp owns the
    reduction and stores.  The opt-in route4 variant lets every worker warp
    reduce one contiguous quarter into shared memory, synchronizes
    the 128-thread worker group, and lets worker warp zero merge the four
    quarter sums.  By default all four worker warps then execute the historical
    masked finalize helper: only warp zero owns outputs, while all four warps
    arrive at ``pre_ready``.  The independent warp0-finalize experiment makes
    worker warps one through three arrive and exit before finalize math.  Both
    modes retain the same uniform 640-thread named-barrier contract.
    Split-128 pair-route keeps the same intra-CTA four-warp merge, but assigns
    one 64-split half to each CTA in an owner/helper pair.  Helpers publish
    their packed half-sum and return; owners acquire, add the helper half, and
    alone execute finalize.
    """
    lane = tl.inline_asm_elementwise(
        asm="mov.u32 $0, %laneid;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    if PERSISTENT_WORKER_GROUP:
        physical_thread = _physical_thread_id_x()
        worker_warp = (physical_thread - JIT_CTA_THREADS) // 32
        routing_active = worker_warp == 0
    else:
        routing_active = True
    packed_mask = routing_active & (lane <= JIT_PROJECTION_OUTPUTS)
    # Production admits only split128 persistent route4-serial.
    tl.static_assert(PERSISTENT_WORKER_GROUP)
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
        PERSISTENT_WORKER_GROUP,
        USE_EARLY_PRE_SYNC,
    )


@triton.jit
def _direct_routing_tail_single_warp(
    pre_ready,
    pre_shared,
    partials_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    token,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    """Compile-only fallback for the non-persistent structural branch.

    Production dispatch always selects the persistent route4 specialization,
    but FlagTree still compiles this sibling warp-specialized region.  Keep
    its original increasing-split reduction without reintroducing the removed
    research variants.
    """
    lane = tl.inline_asm_elementwise(
        asm="mov.u32 $0, %laneid;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    packed_mask = lane <= JIT_PROJECTION_OUTPUTS
    packed_sum = 0.0
    for source_split in tl.static_range(0, JIT_NUM_SPLITS):
        packed_sum += tl.load(
            partials_ptr
            + source_split * JIT_TOKENS * JIT_BLOCK_N
            + token * JIT_BLOCK_N
            + lane,
            mask=packed_mask,
            other=0.0,
        )
    _direct_routing_finalize(
        pre_ready,
        pre_shared,
        pre_shared,
        mix_scales_ptr,
        mix_bases_ptr,
        new_post_mix_ptr,
        new_comb_res_mix_ptr,
        packed_sum,
        lane,
        True,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        False,
        False,
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
    phase_profile_ptr,
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
    ENABLE_PHASE_PROFILE: tl.constexpr,
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
    if ENABLE_PHASE_PROFILE:
        tl.store(
            phase_profile_ptr + split * 4 + 1,
            _read_globaltimer_ns(),
        )
    physical_thread = _physical_thread_id_x()
    barrier_ticket = _monotonic_grid_arrive_and_wait(  # noqa: F841
        tl.cast(grid_counter_ptr, tl.uint64), physical_thread
    )
    # Workers arrive here before waiting.  The default partition reaches the
    # barrier only after lane zero's acquire poll has completed, propagating
    # that acquire to the persistent WGMMA/routing worker group.
    tle.gpu.barrier_wait(grid_acquire_ready)
    if ENABLE_PHASE_PROFILE:
        tl.store(
            phase_profile_ptr + split * 4 + 2,
            _read_globaltimer_ns(),
        )
    # Split-128 still has exactly 64 token owners.  The default partition in
    # the upper half always exits after grid acquire.  Pair-route may keep the
    # matching worker partition alive to publish routing data, but hidden-tail
    # state and barriers remain owner-only.
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
    grid_counter_ptr,
    pair_route_partials_ptr,
    pair_route_epochs_ptr,
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
    USE_DIRECT_TAIL_ILP4: tl.constexpr,
    USE_PERSISTENT_ROUTE4: tl.constexpr,
    USE_ROUTE4_WINDOW4: tl.constexpr,
    USE_ROUTE4_ILP2: tl.constexpr,
    USE_ROUTE4_ILP4: tl.constexpr,
    USE_ROUTE4_WARP0_FINALIZE: tl.constexpr,
    USE_SPLIT128_PAIR_ROUTE: tl.constexpr,
    USE_EARLY_PRE_SYNC: tl.constexpr,
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
    if (split < JIT_TAIL_OWNER_CTAS) | USE_SPLIT128_PAIR_ROUTE:
        token = split % JIT_TAIL_OWNER_CTAS
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
            pair_route_partials_ptr,
            pair_route_epochs_ptr,
            grid_counter_ptr,
            USE_DIRECT_TAIL_ILP4,
            True,
            USE_PERSISTENT_ROUTE4,
            USE_ROUTE4_WINDOW4,
            USE_ROUTE4_ILP2,
            USE_ROUTE4_ILP4,
            USE_ROUTE4_WARP0_FINALIZE,
            USE_SPLIT128_PAIR_ROUTE,
            USE_EARLY_PRE_SYNC,
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
    pair_route_partials_ptr,
    pair_route_epochs_ptr,
    phase_profile_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
    USE_STAGED_WGMMA: tl.constexpr,
    USE_EXPLICIT_WGMMA: tl.constexpr,
    USE_WARP_OVERLAP: tl.constexpr,
    USE_DIRECT_ROUTING_TAIL: tl.constexpr,
    USE_DIRECT_TAIL_ILP4: tl.constexpr,
    USE_PERSISTENT_WS: tl.constexpr,
    USE_PERSISTENT_ROUTE4: tl.constexpr,
    USE_ROUTE4_WINDOW4: tl.constexpr,
    USE_ROUTE4_ILP2: tl.constexpr,
    USE_ROUTE4_ILP4: tl.constexpr,
    USE_ROUTE4_WARP0_FINALIZE: tl.constexpr,
    USE_SPLIT128_PAIR_ROUTE: tl.constexpr,
    USE_EARLY_PRE_SYNC: tl.constexpr,
    ENABLE_PHASE_PROFILE: tl.constexpr,
):
    """One padded M64 tile, 64/128 TF32 producers, and T token tails."""
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
    tl.static_assert((JIT_NUM_SPLITS == 64) | (JIT_NUM_SPLITS == 128))
    tl.static_assert((JIT_HIDDEN_PER_SPLIT == 64) | (JIT_HIDDEN_PER_SPLIT == 32))
    tl.static_assert((JIT_FLAT_K_PER_SPLIT == 256) | (JIT_FLAT_K_PER_SPLIT == 128))
    tl.static_assert(JIT_GRID_CTAS == JIT_NUM_SPLITS)
    tl.static_assert((JIT_CTA_THREADS == 256) | (JIT_CTA_THREADS == 512))
    if USE_EARLY_PRE_SYNC:
        tl.static_assert(JIT_NUM_SPLITS == 128)
        tl.static_assert(USE_STAGED_WGMMA)
        tl.static_assert(USE_EXPLICIT_WGMMA)
        tl.static_assert(USE_WARP_OVERLAP)
        tl.static_assert(USE_DIRECT_ROUTING_TAIL)
        tl.static_assert(not USE_DIRECT_TAIL_ILP4)
        tl.static_assert(USE_PERSISTENT_WS)
        tl.static_assert(USE_PERSISTENT_ROUTE4)
        tl.static_assert(not USE_ROUTE4_WINDOW4)
        tl.static_assert(not USE_ROUTE4_ILP2)
        tl.static_assert(not USE_ROUTE4_ILP4)
        tl.static_assert(not USE_ROUTE4_WARP0_FINALIZE)
        tl.static_assert(not USE_SPLIT128_PAIR_ROUTE)
    if ENABLE_PHASE_PROFILE:
        tl.store(
            phase_profile_ptr + split * 4,
            _read_globaltimer_ns(),
        )

    m = tl.arange(0, JIT_BLOCK_M)
    n = tl.arange(0, JIT_BLOCK_N)
    accumulator = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), tl.float32)
    squared_sum = tl.zeros((JIT_BLOCK_M,), tl.float32)

    if USE_STAGED_WGMMA:
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
        if USE_PERSISTENT_WS:
            tl.static_assert(USE_DIRECT_ROUTING_TAIL)
            tl.static_assert(USE_WARP_OVERLAP)
            persistent_pre_shared = tle.gpu.alloc(
                [4],
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
            if USE_PERSISTENT_ROUTE4:
                persistent_route4_shared = tle.gpu.alloc(
                    [JIT_STAGED_MATH_WARPS, JIT_BLOCK_N],
                    dtype=tl.float32,
                    layout=None,
                    scope=tle.gpu.smem,
                    nv_mma_shared_layout=False,
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
                                phase_profile_ptr,
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
                                ENABLE_PHASE_PROFILE,
                            ),
                        ),
                        (
                            _persistent_staged_math_routing,
                            (
                                partials_ptr,
                                grid_counter_ptr,
                                pair_route_partials_ptr,
                                pair_route_epochs_ptr,
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
                                USE_DIRECT_TAIL_ILP4,
                                True,
                                USE_ROUTE4_WINDOW4,
                                USE_ROUTE4_ILP2,
                                USE_ROUTE4_ILP4,
                                USE_ROUTE4_WARP0_FINALIZE,
                                USE_SPLIT128_PAIR_ROUTE,
                                USE_EARLY_PRE_SYNC,
                            ),
                        ),
                    ],
                    [JIT_STAGED_MATH_WARPS],
                    [JIT_STAGED_MATH_REGS],
                )
            else:
                # Keep route4 resources out of the stable specialization.  The
                # calls are intentionally split instead of assigning route4
                # aliases: the TLE barrier allocator otherwise coalesces the
                # worker-local barrier with ``persistent_pre_ready``, causing
                # concurrent 128- and 640-thread uses of one hardware barrier.
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
                                phase_profile_ptr,
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
                                ENABLE_PHASE_PROFILE,
                            ),
                        ),
                        (
                            _persistent_staged_math_routing,
                            (
                                partials_ptr,
                                grid_counter_ptr,
                                pair_route_partials_ptr,
                                pair_route_epochs_ptr,
                                mix_scales_ptr,
                                mix_bases_ptr,
                                new_post_mix_ptr,
                                new_comb_res_mix_ptr,
                                q_shared,
                                fn_shared,
                                persistent_pre_shared,
                                persistent_pre_shared,
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
                                USE_DIRECT_TAIL_ILP4,
                                False,
                                False,
                                False,
                                False,
                                False,
                                False,
                                False,
                            ),
                        ),
                    ],
                    [JIT_STAGED_MATH_WARPS],
                    [JIT_STAGED_MATH_REGS],
                )
            if ENABLE_PHASE_PROFILE:
                tl.store(
                    phase_profile_ptr + split * 4 + 3,
                    _read_globaltimer_ns(),
                )
            return
        tle.gpu.warp_specialize(
            [
                (
                    _staged_wgmma_producer,
                    (
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
                    ),
                ),
                (
                    _staged_wgmma_math,
                    (
                        partials_ptr,
                        q_shared,
                        fn_shared,
                        stage_full,
                        stage_empty,
                        split,
                    ),
                ),
            ],
            [JIT_STAGED_MATH_WARPS],
            [JIT_STAGED_MATH_REGS],
        )
    elif USE_EXPLICIT_WGMMA:
        # Keep the real P=24 panel in its public [N,K] orientation.  The
        # default MMA shared layout plus trans_b=True exposes it as [K,N] to
        # WGMMA.  Reusing a single slot bounds dynamic SMEM while wait(0)
        # makes each overwrite safe in this first correctness implementation.
        fn_shared = tle.gpu.alloc(
            [JIT_BLOCK_N, JIT_HIDDEN_PER_SPLIT],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
        fn_ready = tle.gpu.alloc_barrier(arrive_count=JIT_CTA_THREADS)
        local_hidden = tl.arange(0, JIT_HIDDEN_PER_SPLIT)
        hidden = split * JIT_HIDDEN_PER_SPLIT + local_hidden
        token_hidden = m[:, None] * JIT_HIDDEN + hidden[None, :]
        token_residual = m[:, None] * JIT_FLAT_HIDDEN + hidden[None, :]
        for channel in tl.range(0, JIT_HC_MULT):
            # Reload per route in the first integrated version.  Keeping all
            # five M64xK64 source tiles live across the four WGMMA issues
            # caused 224 spills in the complete kernel.  This still halves
            # the old K32 setup count (four K64 routes instead of eight K32
            # fragments); a later producer partition will stage once without
            # extending the WGMMA partition's register lifetime.
            x = tl.load(x_ptr + token_hidden).to(tl.float32)
            r0 = tl.load(residual_ptr + token_residual).to(tl.float32)
            r1 = tl.load(residual_ptr + token_residual + JIT_HIDDEN).to(tl.float32)
            r2 = tl.load(residual_ptr + token_residual + 2 * JIT_HIDDEN).to(tl.float32)
            r3 = tl.load(residual_ptr + token_residual + 3 * JIT_HIDDEN).to(tl.float32)
            post = tl.load(post_mix_ptr + m[:, None] * JIT_HC_MULT + channel).to(
                tl.float32
            )
            comb_base = m[:, None] * JIT_HC_MULT * JIT_HC_MULT + channel
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
            q_bf16 = q.to(tl.bfloat16, fp_downcast_rounding="rtne")
            q_fp32 = q_bf16.to(tl.float32)
            flat_hidden = channel * JIT_HIDDEN + hidden
            tl.store(
                new_residual_ptr + m[:, None] * JIT_FLAT_HIDDEN + flat_hidden[None, :],
                q_bf16,
            )

            fn_offsets = n[:, None] * JIT_FLAT_HIDDEN + flat_hidden[None, :]
            fn_tile = tl.load(
                fn_ptr + fn_offsets,
                mask=n[:, None] < JIT_PROJECTION_OUTPUTS,
                other=0.0,
            ).to(tl.float32)
            tl.store(tle.gpu.local_ptr(fn_shared), fn_tile)
            tle.gpu.barrier_wait(fn_ready)
            accumulator = tle.gpu.wgmma(
                q_fp32,
                fn_shared,
                accumulator,
                input_precision="tf32",
                trans_b=True,
            )
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
            squared_sum += tl.sum(q_fp32 * q_fp32, axis=1)
    else:
        for k_base in tl.static_range(0, JIT_FLAT_K_PER_SPLIT, JIT_BLOCK_K):
            local_flat = k_base + tl.arange(0, JIT_BLOCK_K)
            channel = local_flat // JIT_HIDDEN_PER_SPLIT
            hidden = split * JIT_HIDDEN_PER_SPLIT + local_flat % JIT_HIDDEN_PER_SPLIT
            token_hidden = m[:, None] * JIT_HIDDEN + hidden[None, :]
            token_residual = m[:, None] * JIT_FLAT_HIDDEN + hidden[None, :]
            token_state = m[:, None] * JIT_HC_MULT + channel[None, :]

            x = tl.load(x_ptr + token_hidden).to(tl.float32)
            r0 = tl.load(residual_ptr + token_residual).to(tl.float32)
            r1 = tl.load(residual_ptr + token_residual + JIT_HIDDEN).to(tl.float32)
            r2 = tl.load(residual_ptr + token_residual + 2 * JIT_HIDDEN).to(tl.float32)
            r3 = tl.load(residual_ptr + token_residual + 3 * JIT_HIDDEN).to(tl.float32)
            post = tl.load(post_mix_ptr + token_state).to(tl.float32)
            comb_base = m[:, None] * JIT_HC_MULT * JIT_HC_MULT + channel[None, :]
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
            q_bf16 = q.to(tl.bfloat16, fp_downcast_rounding="rtne")
            q_fp32 = q_bf16.to(tl.float32)
            flat_hidden = channel * JIT_HIDDEN + hidden
            tl.store(
                new_residual_ptr + m[:, None] * JIT_FLAT_HIDDEN + flat_hidden[None, :],
                q_bf16,
            )

            fn_offsets = flat_hidden[:, None] + n[None, :] * JIT_FLAT_HIDDEN
            fn_tile = tl.load(
                fn_ptr + fn_offsets,
                mask=n[None, :] < JIT_PROJECTION_OUTPUTS,
                other=0.0,
            ).to(tl.float32)
            accumulator = tl.dot(
                q_fp32,
                fn_tile,
                accumulator,
                input_precision="tf32",
                out_dtype=tl.float32,
            )
            squared_sum += tl.sum(q_fp32 * q_fp32, axis=1)

    if not USE_STAGED_WGMMA:
        workspace_offsets = (
            split * JIT_TOKENS * JIT_BLOCK_N + m[:, None] * JIT_BLOCK_N + n[None, :]
        )
        workspace_values = tl.where(
            n[None, :] < JIT_PROJECTION_OUTPUTS,
            accumulator,
            tl.where(
                n[None, :] == JIT_PROJECTION_OUTPUTS,
                squared_sum[:, None],
                0.0,
            ),
        )
        tl.store(partials_ptr + workspace_offsets, workspace_values)

    producer_ready = tle.gpu.alloc_barrier(arrive_count=JIT_CTA_THREADS)
    tle.gpu.barrier_wait(producer_ready)
    if ENABLE_PHASE_PROFILE:
        tl.store(
            phase_profile_ptr + split * 4 + 1,
            _read_globaltimer_ns(),
        )
    physical_thread = _physical_thread_id_x()
    barrier_ticket = _monotonic_grid_arrive_and_wait(  # noqa: F841
        tl.cast(grid_counter_ptr, tl.uint64), physical_thread
    )
    # The leader performs the acquire poll above.  Propagate that acquire to
    # every worker thread before any CTA starts loading peer partials.
    acquire_ready = tle.gpu.alloc_barrier(arrive_count=JIT_CTA_THREADS)
    tle.gpu.barrier_wait(acquire_ready)
    if ENABLE_PHASE_PROFILE:
        tl.store(
            phase_profile_ptr + split * 4 + 2,
            _read_globaltimer_ns(),
        )

    if split < JIT_TAIL_OWNER_CTAS:
        projection_lane = tl.arange(0, JIT_BLOCK_N)
        projection_mask = projection_lane < JIT_PROJECTION_OUTPUTS
        h = tl.arange(0, JIT_HIDDEN)
        source_split = tl.arange(0, JIT_NUM_SPLITS)

        for token_in_owner in tl.static_range(0, JIT_TOKENS_PER_TAIL_OWNER):
            token = split * JIT_TOKENS_PER_TAIL_OWNER + token_in_owner
            token_hidden_base = token * JIT_HIDDEN
            token_residual_base = token * JIT_FLAT_HIDDEN
            token_state_base = token * JIT_HC_MULT
            token_comb_base = token * JIT_HC_MULT * JIT_HC_MULT

            if USE_DIRECT_ROUTING_TAIL:
                tl.static_assert(JIT_CTA_THREADS == 512)
                tl.static_assert(JIT_TOKENS_PER_TAIL_OWNER == 1)
                pre_shared = tle.gpu.alloc(
                    [4],
                    dtype=tl.float32,
                    layout=None,
                    scope=tle.gpu.smem,
                    nv_mma_shared_layout=False,
                )
                pre_ready = tle.gpu.alloc_barrier(
                    arrive_count=JIT_OVERLAP_LOGICAL_THREADS
                )
                tle.gpu.warp_specialize(
                    [
                        (
                            _direct_hidden_tail,
                            (
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
                            _direct_routing_tail_single_warp,
                            (
                                pre_ready,
                                pre_shared,
                                partials_ptr,
                                mix_scales_ptr,
                                mix_bases_ptr,
                                new_post_mix_ptr + token_state_base,
                                new_comb_res_mix_ptr + token_comb_base,
                                token,
                                hc_norm_eps,
                                hc_pre_eps,
                                hc_post_scale,
                                sinkhorn_eps,
                            ),
                        ),
                    ],
                    [JIT_OVERLAP_ROUTING_WARPS],
                    [64],
                )
                if ENABLE_PHASE_PROFILE:
                    tl.store(
                        phase_profile_ptr + split * 4 + 3,
                        _read_globaltimer_ns(),
                    )
                return

            partial_offsets = (
                source_split[:, None] * JIT_TOKENS * JIT_BLOCK_N
                + token * JIT_BLOCK_N
                + projection_lane[None, :]
            )
            packed_partials = tl.load(partials_ptr + partial_offsets)
            projected = tl.sum(
                tl.where(projection_mask[None, :], packed_partials, 0.0),
                axis=0,
            )
            token_squared_sum = tl.sum(
                tl.load(
                    partials_ptr
                    + source_split * JIT_TOKENS * JIT_BLOCK_N
                    + token * JIT_BLOCK_N
                    + JIT_PROJECTION_OUTPUTS
                ),
                axis=0,
            )
            projection_norm = tl.rsqrt(
                token_squared_sum / JIT_FLAT_HIDDEN + hc_norm_eps
            )
            category_scale = tl.where(
                projection_lane < JIT_HC_MULT,
                tl.load(mix_scales_ptr),
                tl.where(
                    projection_lane < 2 * JIT_HC_MULT,
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
                mixes_ready = tle.gpu.alloc_barrier(
                    arrive_count=JIT_OVERLAP_LOGICAL_THREADS
                )
                pre_ready = tle.gpu.alloc_barrier(
                    arrive_count=JIT_OVERLAP_LOGICAL_THREADS
                )
                tle.gpu.warp_specialize(
                    [
                        (
                            _overlap_hidden_tail,
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
                            _overlap_routing_tail,
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
                    [JIT_OVERLAP_ROUTING_WARPS],
                    [64],
                )
                if ENABLE_PHASE_PROFILE:
                    tl.store(
                        phase_profile_ptr + split * 4 + 3,
                        _read_globaltimer_ns(),
                    )
                return
            pre_mix = tl.sigmoid(mixes) + hc_pre_eps
            next_post_mix = tl.sigmoid(mixes) * hc_post_scale
            post_mask = (projection_lane >= JIT_HC_MULT) & (
                projection_lane < 2 * JIT_HC_MULT
            )
            tl.store(
                new_post_mix_ptr + token_state_base + projection_lane - JIT_HC_MULT,
                next_post_mix,
                mask=post_mask,
            )

            routing_mask = (projection_lane >= 2 * JIT_HC_MULT) & (
                projection_lane < JIT_PROJECTION_OUTPUTS
            )
            # The first structural smoke showed that the warp-shuffle 4x4
            # mapping is not preserved once this mixed tl.dot/tail kernel is
            # lowered with eight warps.  Use the already validated masked
            # reduction formulation until the tail is split into explicit
            # routing/hidden warp roles.
            sinkhorn = tl.zeros((JIT_BLOCK_N,), tl.float32)
            for row in tl.static_range(0, JIT_HC_MULT):
                row_mask = routing_mask & (
                    (projection_lane - 2 * JIT_HC_MULT) // JIT_HC_MULT == row
                )
                row_max = tl.max(tl.where(row_mask, mixes, -float("inf")), axis=0)
                exponent = tl.where(row_mask, tl.exp(mixes - row_max), 0.0)
                row_sum = tl.sum(exponent, axis=0)
                sinkhorn += tl.where(row_mask, exponent / row_sum + sinkhorn_eps, 0.0)
            normalized = tl.zeros((JIT_BLOCK_N,), tl.float32)
            for column in tl.static_range(0, JIT_HC_MULT):
                column_mask = routing_mask & (
                    (projection_lane - 2 * JIT_HC_MULT) % JIT_HC_MULT == column
                )
                column_sum = tl.sum(tl.where(column_mask, sinkhorn, 0.0), axis=0)
                normalized += tl.where(
                    column_mask,
                    sinkhorn / (column_sum + sinkhorn_eps),
                    0.0,
                )
            sinkhorn = normalized
            for _ in tl.range(0, JIT_SINKHORN_ITERS - 1):
                normalized = tl.zeros((JIT_BLOCK_N,), tl.float32)
                for row in tl.static_range(0, JIT_HC_MULT):
                    row_mask = routing_mask & (
                        (projection_lane - 2 * JIT_HC_MULT) // JIT_HC_MULT == row
                    )
                    row_sum = tl.sum(tl.where(row_mask, sinkhorn, 0.0), axis=0)
                    normalized += tl.where(
                        row_mask,
                        sinkhorn / (row_sum + sinkhorn_eps),
                        0.0,
                    )
                sinkhorn = normalized
                normalized = tl.zeros((JIT_BLOCK_N,), tl.float32)
                for column in tl.static_range(0, JIT_HC_MULT):
                    column_mask = routing_mask & (
                        (projection_lane - 2 * JIT_HC_MULT) % JIT_HC_MULT == column
                    )
                    column_sum = tl.sum(tl.where(column_mask, sinkhorn, 0.0), axis=0)
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
                - 2 * JIT_HC_MULT,
                sinkhorn,
                mask=routing_mask,
            )

            norm_input = tl.zeros((JIT_HIDDEN,), tl.float32)
            for channel_idx in tl.static_range(0, JIT_HC_MULT):
                pre_scalar = tl.sum(
                    tl.where(
                        projection_lane == channel_idx,
                        pre_mix,
                        0.0,
                    ),
                    axis=0,
                )
                residual_value = tl.load(
                    new_residual_ptr
                    + token_residual_base
                    + channel_idx * JIT_HIDDEN
                    + h
                ).to(tl.float32)
                norm_input += residual_value * pre_scalar
            norm_input_bf16 = norm_input.to(tl.bfloat16, fp_downcast_rounding="rtne")
            norm_input_fp32 = norm_input_bf16.to(tl.float32)
            rms = tl.rsqrt(
                tl.sum(norm_input_fp32 * norm_input_fp32, axis=0) / JIT_HIDDEN
                + rmsnorm_eps
            )
            rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
            y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
            tl.store(
                y_bf16_ptr + token_hidden_base + h,
                y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            )
            if ENABLE_PHASE_PROFILE:
                tl.store(
                    phase_profile_ptr + split * 4 + 3,
                    _read_globaltimer_ns(),
                )


# Compatibility alias retained for the existing split-64 host/harness tests.
# The underlying JIT function has a split-neutral name, so split-128 resource
# and profiler evidence does not mislabel the compiled kernel.
_t32_wgmma_bm64_split128_tf32_kernel = _t32_wgmma_bm64_split_tf32_kernel


@dataclass(frozen=True)
class PaddedM64WGMMAPlan:
    """Prepared one-launch experimental plan; never share across streams."""

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
    pair_route_partials: torch.Tensor
    pair_route_epochs: torch.Tensor
    phase_profile: torch.Tensor
    hc_norm_eps: float
    hc_pre_eps: float
    hc_post_scale: float
    sinkhorn_eps: float
    rmsnorm_eps: float
    rmsnorm_scale: float
    use_staged_wgmma: bool
    use_explicit_wgmma: bool
    use_warp_overlap: bool
    use_direct_routing_tail: bool
    use_direct_tail_ilp4: bool
    use_persistent_ws: bool
    use_persistent_route4: bool
    persistent_route4_reduction: str
    use_route4_warp0_finalize: bool
    use_split128_pair_route: bool
    use_early_pre_sync: bool
    enable_phase_profile: bool
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
            self.pair_route_partials,
            self.pair_route_epochs,
            self.phase_profile,
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
            USE_STAGED_WGMMA=self.use_staged_wgmma,
            USE_EXPLICIT_WGMMA=self.use_explicit_wgmma,
            USE_WARP_OVERLAP=self.use_warp_overlap,
            USE_DIRECT_ROUTING_TAIL=self.use_direct_routing_tail,
            USE_DIRECT_TAIL_ILP4=self.use_direct_tail_ilp4,
            USE_PERSISTENT_WS=self.use_persistent_ws,
            USE_PERSISTENT_ROUTE4=self.use_persistent_route4,
            USE_ROUTE4_WINDOW4=(self.persistent_route4_reduction == "window4"),
            USE_ROUTE4_ILP2=(self.persistent_route4_reduction == "ilp2"),
            USE_ROUTE4_ILP4=(self.persistent_route4_reduction == "ilp4"),
            USE_ROUTE4_WARP0_FINALIZE=self.use_route4_warp0_finalize,
            USE_SPLIT128_PAIR_ROUTE=self.use_split128_pair_route,
            USE_EARLY_PRE_SYNC=self.use_early_pre_sync,
            ENABLE_PHASE_PROFILE=self.enable_phase_profile,
        )

    def run(self) -> None:
        """Enqueue exactly one fixed-grid kernel launch."""
        _t32_wgmma_bm64_split128_tf32_kernel[(GRID_CTAS,)](
            *self._kernel_args(),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
            USE_STAGED_WGMMA=self.use_staged_wgmma,
            USE_EXPLICIT_WGMMA=self.use_explicit_wgmma,
            USE_WARP_OVERLAP=self.use_warp_overlap,
            USE_DIRECT_ROUTING_TAIL=self.use_direct_routing_tail,
            USE_DIRECT_TAIL_ILP4=self.use_direct_tail_ilp4,
            USE_PERSISTENT_WS=self.use_persistent_ws,
            USE_PERSISTENT_ROUTE4=self.use_persistent_route4,
            USE_ROUTE4_WINDOW4=(self.persistent_route4_reduction == "window4"),
            USE_ROUTE4_ILP2=(self.persistent_route4_reduction == "ilp2"),
            USE_ROUTE4_ILP4=(self.persistent_route4_reduction == "ilp4"),
            USE_ROUTE4_WARP0_FINALIZE=self.use_route4_warp0_finalize,
            USE_SPLIT128_PAIR_ROUTE=self.use_split128_pair_route,
            USE_EARLY_PRE_SYNC=self.use_early_pre_sync,
            ENABLE_PHASE_PROFILE=self.enable_phase_profile,
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
    """Validate one admitted small-bucket padded-BM64 path."""
    enabled = os.environ.get("DG_TLE_MHC_T32_WGMMA_BM64_DOT")
    if enabled not in (None, "0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_BM64_DOT must be 0 or 1")
    if enabled != "1":
        raise RuntimeError(
            "the experimental BM64 TF32 path is disabled; set "
            "DG_TLE_MHC_T32_WGMMA_BM64_DOT=1 explicitly"
        )
    explicit_wgmma_text = os.environ.get("DG_TLE_MHC_T32_WGMMA_EXPLICIT_WGMMA", "0")
    if explicit_wgmma_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_EXPLICIT_WGMMA must be 0 or 1")
    use_explicit_wgmma = explicit_wgmma_text == "1"
    staged_wgmma_text = os.environ.get("DG_TLE_MHC_T32_WGMMA_STAGED_WGMMA", "0")
    if staged_wgmma_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_STAGED_WGMMA must be 0 or 1")
    use_staged_wgmma = staged_wgmma_text == "1"
    if use_staged_wgmma and not use_explicit_wgmma:
        raise ValueError("staged WGMMA requires DG_TLE_MHC_T32_WGMMA_EXPLICIT_WGMMA=1")
    if use_staged_wgmma and NUM_WARPS != 16:
        raise ValueError("staged WGMMA requires 16 default warps")
    overlap_text = os.environ.get("DG_TLE_MHC_T32_WGMMA_WARP_OVERLAP", "0")
    if overlap_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_WARP_OVERLAP must be 0 or 1")
    use_warp_overlap = overlap_text == "1"
    if use_warp_overlap and not (
        NUM_WARPS == 8 or (use_staged_wgmma and NUM_WARPS == 16)
    ):
        raise ValueError(
            "warp-overlap T32 WGMMA path requires 8 default warps, or 16 "
            "with staged WGMMA"
        )
    direct_routing_tail_text = os.environ.get(
        "DG_TLE_MHC_T32_WGMMA_DIRECT_ROUTING_TAIL", "0"
    )
    if direct_routing_tail_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_DIRECT_ROUTING_TAIL must be 0 or 1")
    use_direct_routing_tail = direct_routing_tail_text == "1"
    if use_direct_routing_tail and not (
        use_staged_wgmma and use_warp_overlap and NUM_WARPS == 16
    ):
        raise ValueError(
            "direct routing tail requires staged WGMMA, warp overlap, and "
            "16 default warps"
        )
    direct_tail_ilp4_text = os.environ.get("DG_TLE_MHC_T32_WGMMA_DIRECT_TAIL_ILP4", "0")
    if direct_tail_ilp4_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_DIRECT_TAIL_ILP4 must be 0 or 1")
    use_direct_tail_ilp4 = direct_tail_ilp4_text == "1"
    if use_direct_tail_ilp4 and not use_direct_routing_tail:
        raise ValueError(
            "direct-tail ILP4 requires DG_TLE_MHC_T32_WGMMA_DIRECT_ROUTING_TAIL=1"
        )
    persistent_ws_text = os.environ.get("DG_TLE_MHC_T32_WGMMA_PERSISTENT_WS", "0")
    if persistent_ws_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_PERSISTENT_WS must be 0 or 1")
    use_persistent_ws = persistent_ws_text == "1"
    if use_persistent_ws and not (
        use_staged_wgmma
        and use_warp_overlap
        and use_direct_routing_tail
        and NUM_WARPS == 16
    ):
        raise ValueError(
            "persistent warp specialization requires staged WGMMA, warp "
            "overlap, direct routing tail, and 16 default warps"
        )
    persistent_route4_text = os.environ.get(
        "DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4", "0"
    )
    if persistent_route4_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4 must be 0 or 1")
    use_persistent_route4 = persistent_route4_text == "1"
    if use_persistent_route4 and not use_persistent_ws:
        raise ValueError(
            "persistent route4 requires DG_TLE_MHC_T32_WGMMA_PERSISTENT_WS=1"
        )
    if use_persistent_route4 and use_direct_tail_ilp4:
        raise ValueError(
            "persistent route4 and direct-tail ILP4 are mutually exclusive"
        )
    persistent_route4_reduction = os.environ.get(
        "DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4_REDUCTION", "serial"
    )
    if persistent_route4_reduction not in (
        "serial",
        "window4",
        "ilp2",
        "ilp4",
    ):
        raise ValueError(
            "DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4_REDUCTION must be serial, "
            "window4, ilp2, or ilp4"
        )
    if persistent_route4_reduction != "serial" and not use_persistent_route4:
        raise ValueError(
            "non-serial route4 reduction requires "
            "DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4=1"
        )
    if (
        NUM_SPLITS == 128
        and use_persistent_route4
        and persistent_route4_reduction not in ("serial", "window4")
    ):
        raise ValueError(
            "split128 persistent route4 currently requires the serial or "
            "window4 32-split quarter reduction"
        )
    route4_warp0_finalize_text = os.environ.get(
        "DG_TLE_MHC_T32_WGMMA_ROUTE4_WARP0_FINALIZE", "0"
    )
    if route4_warp0_finalize_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_ROUTE4_WARP0_FINALIZE must be 0 or 1")
    use_route4_warp0_finalize = route4_warp0_finalize_text == "1"
    if use_route4_warp0_finalize and not use_persistent_route4:
        raise ValueError(
            "route4 warp0 finalize requires " "DG_TLE_MHC_T32_WGMMA_PERSISTENT_ROUTE4=1"
        )
    split128_pair_route_text = os.environ.get(
        "DG_TLE_MHC_T32_WGMMA_SPLIT128_PAIR_ROUTE", "0"
    )
    if split128_pair_route_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_SPLIT128_PAIR_ROUTE must be 0 or 1")
    use_split128_pair_route = split128_pair_route_text == "1"
    if use_split128_pair_route and not (
        NUM_SPLITS == 128
        and use_staged_wgmma
        and use_explicit_wgmma
        and use_warp_overlap
        and use_direct_routing_tail
        and not use_direct_tail_ilp4
        and use_persistent_ws
        and use_persistent_route4
        and persistent_route4_reduction == "serial"
        and not use_route4_warp0_finalize
        and NUM_WARPS == 16
        and STAGED_MATH_REGS == 128
    ):
        raise ValueError(
            "split128 pair-route is admitted only for the exact validated "
            "split128 staged-WGMMA persistent route4-serial plan"
        )
    early_pre_sync_text = os.environ.get("DG_TLE_MHC_T32_WGMMA_EARLY_PRE_SYNC", "0")
    if early_pre_sync_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_EARLY_PRE_SYNC must be 0 or 1")
    use_early_pre_sync = early_pre_sync_text == "1"
    if use_early_pre_sync and not (
        NUM_SPLITS == 128
        and use_staged_wgmma
        and use_explicit_wgmma
        and use_warp_overlap
        and use_direct_routing_tail
        and not use_direct_tail_ilp4
        and use_persistent_ws
        and use_persistent_route4
        and persistent_route4_reduction == "serial"
        and not use_route4_warp0_finalize
        and not use_split128_pair_route
        and NUM_WARPS == 16
        and STAGED_MATH_REGS == 128
    ):
        raise ValueError(
            "early Pre sync is admitted only for the exact split128 "
            "staged-WGMMA persistent route4-serial plan with pair-route and "
            "warp0-finalize disabled"
        )
    phase_profile_text = os.environ.get("DG_TLE_MHC_T32_WGMMA_PHASE_PROFILE", "0")
    if phase_profile_text not in ("0", "1"):
        raise ValueError("DG_TLE_MHC_T32_WGMMA_PHASE_PROFILE must be 0 or 1")
    enable_phase_profile = phase_profile_text == "1"
    if NUM_SPLITS == 128 and not (
        use_staged_wgmma
        and use_explicit_wgmma
        and use_warp_overlap
        and use_direct_routing_tail
        and not use_direct_tail_ilp4
        and use_persistent_ws
        and use_persistent_route4
        and persistent_route4_reduction in ("serial", "window4")
        and not use_route4_warp0_finalize
        and NUM_WARPS == 16
        and STAGED_MATH_REGS == 128
    ):
        raise ValueError(
            "split128 is currently admitted only for the validated staged "
            "WGMMA + persistent route4 serial/window4 plan with warp overlap, "
            "direct routing tail, 16 default warps, 128 math registers, and "
            "masked four-warp finalize"
        )
    stable_t32_wgmma = (
        NUM_SPLITS == 128
        and NUM_WARPS == 16
        and STAGED_MATH_REGS == 128
        and use_staged_wgmma
        and use_explicit_wgmma
        and use_warp_overlap
        and use_direct_routing_tail
        and not use_direct_tail_ilp4
        and use_persistent_ws
        and use_persistent_route4
        and persistent_route4_reduction == "serial"
        and not use_route4_warp0_finalize
        and not use_split128_pair_route
        and use_early_pre_sync
        and not enable_phase_profile
    )
    if not stable_t32_wgmma:
        raise ValueError(
            "the padded-BM64 path currently admits only the validated "
            "split128/16-warp/staged-WGMMA/persistent-route4-serial/"
            "early-Pre-sync configuration"
        )
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
    if use_split128_pair_route:
        pair_route_partials = torch.empty(
            (PAIR_ROUTE_WORKSPACE_ROWS, WORKSPACE_ROW_ELEMENTS),
            dtype=torch.float32,
            device=device,
        )
        pair_route_epochs = torch.zeros(
            (PAIR_ROUTE_WORKSPACE_ROWS,),
            dtype=torch.uint64,
            device=device,
        )
    else:
        # Compile-time-disabled kernel arguments alias existing allocations;
        # the baseline plan exposes no extra workspace or counter mapping.
        pair_route_partials = partials
        pair_route_epochs = grid_counter
    phase_profile = torch.zeros(
        (GRID_CTAS, 4) if enable_phase_profile else (1,),
        dtype=torch.uint64,
        device=device,
    )
    workspace_tensors = {"projection_and_sqsum_partials": partials}
    if use_split128_pair_route:
        workspace_tensors["split128_pair_route_partials"] = pair_route_partials
    if enable_phase_profile:
        workspace_tensors["phase_profile_ns"] = phase_profile
    counter_tensors = {"monotonic_grid_barrier": grid_counter}
    if use_split128_pair_route:
        counter_tensors["split128_pair_route_epochs"] = pair_route_epochs
    expected_physical_threads_per_cta = CTA_THREADS
    if use_staged_wgmma:
        expected_physical_threads_per_cta += STAGED_MATH_THREADS
    elif use_warp_overlap:
        # TLE reserves worker warps in four-warp specialization groups even
        # when the routing worker has only one logical warp.
        expected_physical_threads_per_cta += 4 * 32
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
        "dot_implementation": (
            "tle_warp_specialized_shared_a_shared_b_wgmma"
            if use_staged_wgmma
            else (
                "tle_explicit_register_a_shared_b_wgmma"
                if use_explicit_wgmma
                else "triton_tl_dot"
            )
        ),
        "dot_input_precision": "tf32",
        "dot_accumulator": "float32",
        "explicit_wgmma": use_explicit_wgmma,
        "staged_wgmma": use_staged_wgmma,
        "staged_wgmma_math_warps": (STAGED_MATH_WARPS if use_staged_wgmma else 0),
        "staged_wgmma_math_registers": (STAGED_MATH_REGS if use_staged_wgmma else 0),
        "staged_wgmma_logical_threads": (
            STAGED_LOGICAL_THREADS if use_staged_wgmma else 0
        ),
        "explicit_wgmma_k_tile": (HIDDEN_PER_SPLIT if use_explicit_wgmma else None),
        "explicit_wgmma_wait_groups": (
            "wait_group_0_per_route" if use_explicit_wgmma else None
        ),
        "num_warps": NUM_WARPS,
        "threads_per_cta": CTA_THREADS,
        "default_partition_threads": CTA_THREADS,
        "expected_physical_threads_per_cta": (expected_physical_threads_per_cta),
        "warp_overlap": use_warp_overlap,
        "direct_routing_tail": use_direct_routing_tail,
        "direct_tail_ilp4": use_direct_tail_ilp4,
        "persistent_warp_specialization": use_persistent_ws,
        "persistent_route4": use_persistent_route4,
        "persistent_route4_warp0_finalize": (
            use_route4_warp0_finalize if use_persistent_route4 else False
        ),
        "split128_pair_route_enabled": use_split128_pair_route,
        "early_pre_sync_enabled": use_early_pre_sync,
        "pre_ready_worker_sync_mode": (
            "route4_row0_relay_store_then_blocking_wait_then_" "reload_for_sinkhorn"
            if use_early_pre_sync
            else "nonblocking_arrive_after_pre_store"
        ),
        "early_pre_sync_intent": (
            "force_sinkhorn_data_dependency_through_640_thread_"
            "pre_ready_rendezvous_to_test_hidden_tail_critical_path"
            if use_early_pre_sync
            else None
        ),
        "early_pre_sync_relay_location": (
            "persistent_route4_shared_row0_lanes8_23" if use_early_pre_sync else None
        ),
        "early_pre_sync_relay_order": (
            "store_raw_mixes_f32_then_pre_ready_wait_then_load_f32_" "for_sinkhorn"
            if use_early_pre_sync
            else None
        ),
        "early_pre_sync_relay_extra_shared_bytes": 0,
        "warp_specialization_regions": (
            1
            if use_persistent_ws
            else (
                2
                if use_staged_wgmma and use_direct_routing_tail
                else 1 if use_staged_wgmma or use_warp_overlap else 0
            )
        ),
        "persistent_default_role": (
            (
                "producer_grid_acquire_hidden"
                if GRID_CTAS == TAIL_OWNER_CTAS
                else "producer_grid_acquire_hidden_first_T_ctas"
            )
            if use_persistent_ws
            else None
        ),
        "persistent_worker_role": (
            "wgmma_grid_acquire_pair_reduce_helper_publish_owner_finalize"
            if use_split128_pair_route
            else (
                "wgmma_grid_acquire_route4_reduce_then_warp0_finalize"
                if use_route4_warp0_finalize
                else (
                    "wgmma_grid_acquire_route4_reduce_then_masked_finalize"
                    if use_persistent_route4
                    else (
                        "wgmma_grid_acquire_routing_warp0"
                        if use_persistent_ws
                        else None
                    )
                )
            )
        ),
        "persistent_barrier_arrive_count": (
            STAGED_LOGICAL_THREADS if use_persistent_ws else None
        ),
        "persistent_named_barrier_count": (
            6 if use_persistent_route4 else 5 if use_persistent_ws else 0
        ),
        "persistent_named_barrier_contract": (
            {
                "stage_full": STAGED_LOGICAL_THREADS,
                "stage_empty": STAGED_LOGICAL_THREADS,
                "producer_ready": STAGED_LOGICAL_THREADS,
                "grid_acquire_ready": STAGED_LOGICAL_THREADS,
                "pre_ready": STAGED_LOGICAL_THREADS,
                **(
                    {"route4_ready_raw_barrier_5": STAGED_MATH_THREADS}
                    if use_persistent_route4
                    else {}
                ),
            }
            if use_persistent_ws
            else None
        ),
        "persistent_routing_active_warps": (
            4 if use_persistent_route4 else 1 if use_persistent_ws else None
        ),
        "persistent_routing_finalize_warps": (1 if use_persistent_ws else None),
        "persistent_route4_nonfinalize_warps": (
            STAGED_MATH_WARPS - 1 if use_persistent_route4 else None
        ),
        "persistent_route4_nonfinalize_action": (
            "owner_masked_finalize_arrive_or_helper_pair_publish"
            if use_split128_pair_route
            else (
                "masked_finalize_with_pre_ready_wait"
                if use_early_pre_sync
                else (
                    "pre_ready_arrive_only"
                    if use_route4_warp0_finalize
                    else (
                        "masked_finalize_with_pre_ready_arrive"
                        if use_persistent_route4
                        else None
                    )
                )
            )
        ),
        "persistent_route4_finalize_branch": (
            "cta_uniform_helper_publish_or_owner_masked_finalize"
            if use_split128_pair_route
            else (
                "warp_uniform_worker_warp0"
                if use_route4_warp0_finalize
                else (
                    "all_worker_warps_masked_by_routing_active"
                    if use_persistent_route4
                    else None
                )
            )
        ),
        "persistent_route4_finalize_issue_warps": (
            1
            if use_route4_warp0_finalize
            else STAGED_MATH_WARPS if use_persistent_route4 else None
        ),
        "persistent_route4_shared_shape": (
            [STAGED_MATH_WARPS, BLOCK_N] if use_persistent_route4 else None
        ),
        "persistent_route4_barrier_arrive_count": (
            STAGED_MATH_THREADS if use_persistent_route4 else None
        ),
        "persistent_route4_sync": (
            "inline_ptx_bar_sync_5_plus_helper_bar_warp_sync"
            if use_split128_pair_route
            else "inline_ptx_bar_sync_5" if use_persistent_route4 else None
        ),
        "persistent_route4_reduction_mode": (
            persistent_route4_reduction if use_persistent_route4 else None
        ),
        "persistent_route4_reduction_load_window": (
            4
            if use_persistent_route4 and persistent_route4_reduction != "serial"
            else 1 if use_persistent_route4 else None
        ),
        "persistent_route4_reduction_accumulators": (
            4
            if use_persistent_route4 and persistent_route4_reduction == "ilp4"
            else (
                2
                if use_persistent_route4 and persistent_route4_reduction == "ilp2"
                else 1 if use_persistent_route4 else None
            )
        ),
        "persistent_route4_reduction_stride_bytes": (
            32768
            if use_persistent_route4 and persistent_route4_reduction != "serial"
            else 8192 if use_persistent_route4 else None
        ),
        "persistent_route4_reduction_split_stride_bytes": (
            8192 if use_persistent_route4 else None
        ),
        "persistent_route4_splits_per_worker_warp": (
            PAIR_ROUTE_SPLITS_PER_WARP
            if use_split128_pair_route
            else ROUTE4_SPLITS_PER_WARP if use_persistent_route4 else None
        ),
        "persistent_route4_quarter_stride_bytes": (
            PAIR_ROUTE_SPLITS_PER_WARP * TOKENS * BLOCK_N * 4
            if use_split128_pair_route
            else (
                ROUTE4_SPLITS_PER_WARP * TOKENS * BLOCK_N * 4
                if use_persistent_route4
                else None
            )
        ),
        "persistent_route4_reduction_bitwise_equivalent_to_route4_serial": (
            False
            if use_split128_pair_route
            else (
                persistent_route4_reduction in ("serial", "window4")
                if use_persistent_route4
                else None
            )
        ),
        "tail_partial_reduction": (
            "paired_ctas_each_four_worker_warps_contiguous16_then_"
            "balanced_half_merge_then_owner_add_packed25"
            if use_split128_pair_route
            else (
                f"four_worker_warps_contiguous{ROUTE4_SPLITS_PER_WARP}_"
                "ilp4_then_balanced_merge_packed25"
                if use_persistent_route4 and persistent_route4_reduction == "ilp4"
                else (
                    f"four_worker_warps_contiguous{ROUTE4_SPLITS_PER_WARP}_"
                    "ilp2_then_balanced_merge_packed25"
                    if use_persistent_route4 and persistent_route4_reduction == "ilp2"
                    else (
                        f"four_worker_warps_contiguous{ROUTE4_SPLITS_PER_WARP}_"
                        "window4_then_balanced_merge_packed25"
                        if use_persistent_route4
                        and persistent_route4_reduction == "window4"
                        else (
                            f"four_worker_warps_contiguous{ROUTE4_SPLITS_PER_WARP}_"
                            "then_balanced_merge_packed25"
                            if use_persistent_route4
                            else (
                                "single_routing_warp_mod4_ilp_packed25"
                                if use_direct_tail_ilp4
                                else (
                                    "single_routing_warp_increasing_split_packed25"
                                    if use_direct_routing_tail
                                    else "default_partition_triton_sum"
                                )
                            )
                        )
                    )
                )
            )
        ),
        "tail_reduction_association": (
            "two_64_split_halves_each_four_contiguous_16_term_serial_"
            "quarters_then_balanced_merge_then_owner_add"
            if use_split128_pair_route
            else (
                f"four_contiguous_{ROUTE4_SPLITS_PER_WARP}_term_mod4_"
                "quarters_then_balanced_merge"
                if use_persistent_route4 and persistent_route4_reduction == "ilp4"
                else (
                    f"four_contiguous_{ROUTE4_SPLITS_PER_WARP}_term_mod2_"
                    "quarters_then_balanced_merge"
                    if use_persistent_route4 and persistent_route4_reduction == "ilp2"
                    else (
                        f"four_contiguous_{ROUTE4_SPLITS_PER_WARP}_term_serial_"
                        "quarters_then_balanced_merge"
                        if use_persistent_route4
                        else (
                            "mod4_16_terms_then_balanced_2_level_merge"
                            if use_direct_tail_ilp4
                            else (
                                "increasing_split_serial"
                                if use_direct_routing_tail
                                else "compiler_defined"
                            )
                        )
                    )
                )
            )
        ),
        "tail_reduction_bitwise_equivalent_to_serial": (
            not (use_direct_tail_ilp4 or use_persistent_route4)
        ),
        "tail_pre_publication": (
            "pair_owner_acquire_then_all_workers_masked_finalize_arrive"
            if use_split128_pair_route
            else (
                "persistent_route4_relay_store_then_all_workers_pre_ready_"
                "wait_then_sinkhorn_relay_load"
                if use_early_pre_sync
                else (
                    "persistent_route4_worker_warp0_finalize_" "nonowners_arrive_only"
                    if use_route4_warp0_finalize
                    else (
                        "persistent_route4_all_workers_masked_finalize_arrive"
                        if use_persistent_route4
                        else (
                            "persistent_worker_warp0_shared4_group_arrive"
                            if use_persistent_ws
                            else (
                                "routing_warp_shared4_nonblocking_arrive"
                                if use_warp_overlap
                                else "default_partition"
                            )
                        )
                    )
                )
            )
        ),
        "phase_profile_enabled": enable_phase_profile,
        "overlap_routing_warps": (OVERLAP_ROUTING_WARPS if use_warp_overlap else 0),
        "schedule": (
            "bm64_n32_split128_pair_route_tf32_grid_barrier"
            if use_split128_pair_route
            else f"bm64_n32_split{NUM_SPLITS}_tf32_grid_barrier"
        ),
        "tail_owner_ctas": TAIL_OWNER_CTAS,
        "tail_nonowner_ctas": GRID_CTAS - TAIL_OWNER_CTAS,
        "tail_nonowner_policy": (
            "producer_exit_worker_publish_pair_partial_after_grid_acquire"
            if use_split128_pair_route
            else (
                "exit_both_persistent_partitions_after_grid_acquire"
                if GRID_CTAS > TAIL_OWNER_CTAS
                else "none"
            )
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
        "split128_pair_route_owner_ctas": (
            TAIL_OWNER_CTAS if use_split128_pair_route else None
        ),
        "split128_pair_route_helper_ctas": (
            TAIL_OWNER_CTAS if use_split128_pair_route else None
        ),
        "split128_pair_route_cta_distance": (
            TAIL_OWNER_CTAS if use_split128_pair_route else None
        ),
        "split128_pair_route_splits_per_cta": (
            PAIR_ROUTE_SPLITS_PER_CTA if use_split128_pair_route else None
        ),
        "split128_pair_route_splits_per_worker_warp": (
            PAIR_ROUTE_SPLITS_PER_WARP if use_split128_pair_route else None
        ),
        "split128_pair_route_workspace_key": (
            "split128_pair_route_partials" if use_split128_pair_route else None
        ),
        "split128_pair_route_workspace_shape": (
            list(pair_route_partials.shape) if use_split128_pair_route else None
        ),
        "split128_pair_route_workspace_allocated_bytes": (
            pair_route_partials.numel() * pair_route_partials.element_size()
            if use_split128_pair_route
            else 0
        ),
        "split128_pair_route_counter_key": (
            "split128_pair_route_epochs" if use_split128_pair_route else None
        ),
        "split128_pair_route_counter_shape": (
            list(pair_route_epochs.shape) if use_split128_pair_route else None
        ),
        "split128_pair_route_counter_allocated_bytes": (
            pair_route_epochs.numel() * pair_route_epochs.element_size()
            if use_split128_pair_route
            else 0
        ),
        "split128_pair_route_epoch_increment_per_launch": (
            GRID_CTAS if use_split128_pair_route else None
        ),
        "split128_pair_route_counter_policy": (
            "publish_current_grid_target_monotonic_epoch"
            if use_split128_pair_route
            else None
        ),
        "split128_pair_route_counter_reuse_contract": (
            "zero_initialized_no_reset_ordered_stream_only"
            if use_split128_pair_route
            else None
        ),
        "split128_pair_route_epoch_publication": (
            "warp0_store_25_lanes_bar_warp_sync_lane0_st_release_gpu"
            if use_split128_pair_route
            else None
        ),
        "split128_pair_route_epoch_acquire": (
            "owner_warp0_lanes0_24_ld_acquire_gpu_then_partial_load"
            if use_split128_pair_route
            else None
        ),
        "workspace_shape": list(partials.shape),
        "workspace_allocated_bytes": partials.numel() * partials.element_size(),
        "counter_shape": list(grid_counter.shape),
        "counter_allocated_bytes": (grid_counter.numel() * grid_counter.element_size()),
        "bf16_rounding_boundaries": [
            "new_residual",
            "norm_input",
            "y_bf16",
        ],
        "experimental": True,
        "default_selected": False,
        "report_eligible": False,
        "result_scope": (
            "small_padded_bm64_split128_pair_route_tf32_structural_prototype"
            if use_split128_pair_route
            else (
                "small_padded_bm64_tf32_structural_prototype"
                if NUM_SPLITS == 64
                else "small_padded_bm64_split128_tf32_structural_prototype"
            )
        ),
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
        "prototype_delta_from_cuda_winner": {
            "producer_grid_ctas": GRID_CTAS,
            "tail_mapping": (
                "32_owner_helper_cta_pairs_x_1_token"
                if use_split128_pair_route
                else f"{LOGICAL_TOKENS}_ctas_x_1_token"
            ),
            "threads_per_cta": expected_physical_threads_per_cta,
            "global_synchronization": (
                "cooperative_launch_with_monotonic_u64_" "release_acquire_spin"
            ),
            "projection_core": (
                "TLE_warp_specialized_SS_WGMMA_TF32"
                if use_staged_wgmma
                else (
                    "TLE_explicit_WGMMA_TF32"
                    if use_explicit_wgmma
                    else "Triton_tl_dot_TF32"
                )
            ),
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
        pair_route_partials=pair_route_partials,
        pair_route_epochs=pair_route_epochs,
        phase_profile=phase_profile,
        hc_norm_eps=float(inputs["hc_norm_eps"]),
        hc_pre_eps=float(inputs["hc_pre_eps"]),
        hc_post_scale=float(inputs["hc_post_scale"]),
        sinkhorn_eps=float(inputs["sinkhorn_eps"]),
        rmsnorm_eps=float(inputs["rmsnorm_eps"]),
        rmsnorm_scale=float(inputs["rmsnorm_scale"]),
        use_staged_wgmma=use_staged_wgmma,
        use_explicit_wgmma=use_explicit_wgmma,
        use_warp_overlap=use_warp_overlap,
        use_direct_routing_tail=use_direct_routing_tail,
        use_direct_tail_ilp4=use_direct_tail_ilp4,
        use_persistent_ws=use_persistent_ws,
        use_persistent_route4=use_persistent_route4,
        persistent_route4_reduction=persistent_route4_reduction,
        use_route4_warp0_finalize=use_route4_warp0_finalize,
        use_split128_pair_route=use_split128_pair_route,
        use_early_pre_sync=use_early_pre_sync,
        enable_phase_profile=enable_phase_profile,
        workspace_tensors=workspace_tensors,
        counter_tensors=counter_tensors,
        metadata=metadata,
    )


__all__ = ["PaddedM64WGMMAPlan", "prepare_tle_sm90_mega_mhc"]

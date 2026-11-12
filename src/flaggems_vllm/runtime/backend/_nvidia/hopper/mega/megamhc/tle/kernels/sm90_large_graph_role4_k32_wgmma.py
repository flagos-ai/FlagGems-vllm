"""SM90 large-Graph MegaMHC role-4 K32 pipeline experiment.

This is a shape-strict TileLang/TLE migration of the production CUDA direct
cooperative path for the DeepSeek-V4-Flash Normal-BF16 profile.  It covers the
16 CUDA-Graph buckets from T=272 through T=512.  The launch geometry mirrors
the CUDA winner instead of padding every shape to one synthetic topology:

* T=272..320: five M64 tiles, split-26, grid 130;
* T=336..384: six M64 tiles, split-22, grid 132;
* T=400: seven M64 tiles, split-16, 112 producers plus 20 consumer CTAs;
* T=416..448: seven M64 tiles, split-16, grid 112;
* T=464..512: eight M64 tiles, split-16, grid 128.

Phase one assigns the default four-warp partition to WGMMA and four additional
four-warp partitions to disjoint M16 producer stripes.  Eight K32 stages keep
the BF16-rounded Post values and FP32 Fn slices below 100 KiB of shared memory.
After the cooperative release/acquire rendezvous, the four math warps route
four striped tokens while the four producer partitions independently run the
matching hidden/RMSNorm work.  The prepared plan enqueues exactly one kernel.

The module is experimental and is never selected without its explicit switch.
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
from . import sm90_fn_cp_async_stage_probe as _fn_cp

LOGICAL_TOKEN_BUCKETS = (
    272,
    288,
    304,
    320,
    336,
    352,
    368,
    384,
    400,
    416,
    432,
    448,
    464,
    480,
    496,
    512,
)
_TOKENS_TEXT = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_ROLE4_LOGICAL_TOKENS", "272")
try:
    LOGICAL_TOKENS = int(_TOKENS_TEXT)
except ValueError as exc:  # pragma: no cover - import-time contract
    raise ValueError(
        "DG_TLE_MHC_LARGE_GRAPH_ROLE4_LOGICAL_TOKENS must be an integer"
    ) from exc
if LOGICAL_TOKENS not in LOGICAL_TOKEN_BUCKETS:
    raise ValueError(
        "DG_TLE_MHC_LARGE_GRAPH_ROLE4_LOGICAL_TOKENS must be one of "
        f"{LOGICAL_TOKEN_BUCKETS}, got {LOGICAL_TOKENS}"
    )

# Research-only phase seam.  The default value is the complete operator.
# Values 1..3 compile progressively larger prefixes so that a Graph replay
# can attribute the critical path without adding clock instrumentation to the
# full candidate:
#   1 = Post/Pre + projection producer/WGMMA + partial stores
#   2 = phase 1 + cooperative grid rendezvous
#   3 = phase 2 + routing reduction/finalize
#   4 = complete operator, including the hidden/RMSNorm tail
_PROBE_PHASE_TEXT = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_PROBE_PHASE", "4")
try:
    PROBE_PHASE = int(_PROBE_PHASE_TEXT)
except ValueError as exc:  # pragma: no cover - import-time contract
    raise ValueError("DG_TLE_MHC_LARGE_GRAPH_PROBE_PHASE must be an integer") from exc
if PROBE_PHASE not in (1, 2, 3, 4):
    raise ValueError("DG_TLE_MHC_LARGE_GRAPH_PROBE_PHASE must be 1, 2, 3, or 4")
USE_WEIGHT_CACHE = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_WEIGHT_CACHE", "1") == "1"
USE_ROLE_OVERLAP = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_ROLE_OVERLAP", "0") == "1"
USE_ALLOCATED_ROLE_BARRIERS = (
    os.environ.get("DG_TLE_MHC_LARGE_GRAPH_ALLOCATED_ROLE_BARRIERS", "0") == "1"
)
if USE_ALLOCATED_ROLE_BARRIERS and not USE_ROLE_OVERLAP:
    raise ValueError("allocated role barriers require role overlap")
USE_SHARED_ROLE_FLAGS = (
    os.environ.get("DG_TLE_MHC_LARGE_GRAPH_SHARED_ROLE_FLAGS", "0") == "1"
)
if USE_SHARED_ROLE_FLAGS and not USE_ROLE_OVERLAP:
    raise ValueError("shared role flags require role overlap")
if USE_SHARED_ROLE_FLAGS and USE_ALLOCATED_ROLE_BARRIERS:
    raise ValueError("shared flags and allocated role barriers are exclusive")
USE_EARLY_PRE_SYNC = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_EARLY_PRE_SYNC", "1") == "1"
USE_TRUE_K64_STAGE = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_TRUE_K64_STAGE", "0") == "1"
USE_K128_PIPE4 = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_K128_PIPE4", "0") == "1"
if USE_K128_PIPE4 and LOGICAL_TOKENS < 400:
    raise ValueError("K128/pipe4 requires the split-16 T>=400 family")
USE_FN_CP_ASYNC = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_FN_CP_ASYNC", "0") == "1"
if USE_FN_CP_ASYNC and not USE_TRUE_K64_STAGE:
    raise ValueError("Fn cp.async staging requires the true-K64 pipeline")
if USE_K128_PIPE4 and USE_FN_CP_ASYNC:
    raise ValueError("K128/pipe4 does not support the K64 Fn cp.async probe")
USE_BF16_RS_PIPE = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_BF16_RS_PIPE", "0") == "1"
USE_MINIMAL_SCRATCH = (
    os.environ.get("DG_TLE_MHC_LARGE_GRAPH_MINIMAL_SCRATCH", "0") == "1"
)
USE_COEFF_CACHE = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_COEFF_CACHE", "0") == "1"
USE_PTX_SINKHORN = os.environ.get("DG_TLE_MHC_LARGE_GRAPH_PTX_SINKHORN", "0") == "1"
USE_NON_STRIPED_TAIL = (
    os.environ.get(
        "DG_TLE_MHC_LARGE_GRAPH_NON_STRIPED_TAIL",
        "1" if LOGICAL_TOKENS >= 464 else "0",
    )
    == "1"
)
USE_EXACT_T512 = (
    os.environ.get(
        "DG_TLE_MHC_LARGE_GRAPH_EXACT_T512",
        "0",
    )
    == "1"
)
if USE_EXACT_T512 and LOGICAL_TOKENS != 512:
    raise ValueError("exact-T512 specialization requires T=512")
USE_EXACT_T448 = (
    os.environ.get(
        "DG_TLE_MHC_LARGE_GRAPH_EXACT_T448",
        "0",
    )
    == "1"
)
if USE_EXACT_T448 and LOGICAL_TOKENS != 448:
    raise ValueError("exact-T448 specialization requires T=448")
if USE_EXACT_T448 and not USE_NON_STRIPED_TAIL:
    raise ValueError("exact-T448 specialization requires non-striped owners")
USE_EXACT_FULL_TILE = USE_EXACT_T448 or USE_EXACT_T512
if USE_MINIMAL_SCRATCH and not USE_EXACT_FULL_TILE:
    raise ValueError("minimal scratch requires an exact full-tile specialization")
USE_MANUAL_ROLE4_TAIL = (
    os.environ.get("DG_TLE_MHC_LARGE_GRAPH_MANUAL_ROLE4_TAIL", "0") == "1"
)
USE_PACKED_MANUAL_ROLE4_TAIL = (
    os.environ.get("DG_TLE_MHC_LARGE_GRAPH_PACKED_MANUAL_TAIL", "0") == "1"
)
if USE_PACKED_MANUAL_ROLE4_TAIL and not USE_MANUAL_ROLE4_TAIL:
    raise ValueError("packed manual tail requires manual role-4 tail")
USE_FAST_GRID_BARRIER = (
    os.environ.get("DG_TLE_MHC_LARGE_GRAPH_FAST_GRID_BARRIER", "0") == "1"
)
USE_TILE_LOCAL_BARRIER = (
    os.environ.get("DG_TLE_MHC_LARGE_GRAPH_TILE_LOCAL_BARRIER", "0") == "1"
)
USE_TILE_LOCAL_FAST_BARRIER = (
    os.environ.get("DG_TLE_MHC_LARGE_GRAPH_TILE_LOCAL_FAST_BARRIER", "0") == "1"
)
if USE_TILE_LOCAL_FAST_BARRIER and not USE_TILE_LOCAL_BARRIER:
    raise ValueError("tile-local fast barrier requires tile-local barrier")
if USE_TILE_LOCAL_BARRIER and USE_FAST_GRID_BARRIER:
    raise ValueError("tile-local and fast global barriers are exclusive")
USE_PTX_BF16X8_TAIL = (
    os.environ.get("DG_TLE_MHC_LARGE_GRAPH_PTX_BF16X8_TAIL", "0") == "1"
)
if USE_PTX_BF16X8_TAIL and USE_MANUAL_ROLE4_TAIL:
    raise ValueError("PTX BF16x8 and tensor-manual tails are exclusive")
if USE_MINIMAL_SCRATCH and (
    USE_WEIGHT_CACHE or USE_COEFF_CACHE or USE_MANUAL_ROLE4_TAIL or USE_PTX_BF16X8_TAIL
):
    raise ValueError("minimal scratch requires uncached default hidden-tail storage")

HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = HIDDEN * HC_MULT
SINKHORN_ITERS = 20
BLOCK_M = 64
BLOCK_N = 32
K_STAGE = 128 if USE_K128_PIPE4 else 64
K_HALF = K_STAGE if (USE_TRUE_K64_STAGE or USE_K128_PIPE4) else 32
NUM_K_HALVES = K_STAGE // K_HALF
NUM_HIDDEN_BLOCKS = HIDDEN // K_STAGE
NUM_M_TILES = (LOGICAL_TOKENS + BLOCK_M - 1) // BLOCK_M
NUM_SPLITS = 26 if LOGICAL_TOKENS <= 320 else 22 if LOGICAL_TOKENS <= 384 else 16
BASE_HIDDEN_BLOCKS = NUM_HIDDEN_BLOCKS // NUM_SPLITS
REMAIN_HIDDEN_BLOCKS = NUM_HIDDEN_BLOCKS % NUM_SPLITS
MAX_HIDDEN_BLOCKS_PER_SPLIT = BASE_HIDDEN_BLOCKS + (1 if REMAIN_HIDDEN_BLOCKS else 0)
PRODUCER_CTAS = NUM_M_TILES * NUM_SPLITS
_DEFAULT_GRID_CTAS = 132 if LOGICAL_TOKENS == 400 else PRODUCER_CTAS
_GRID_CTAS_TEXT = os.environ.get(
    "DG_TLE_MHC_LARGE_GRAPH_GRID_CTAS", str(_DEFAULT_GRID_CTAS)
)
try:
    GRID_CTAS = int(_GRID_CTAS_TEXT)
except ValueError as exc:  # pragma: no cover - import-time contract
    raise ValueError("DG_TLE_MHC_LARGE_GRAPH_GRID_CTAS must be an integer") from exc
if not (PRODUCER_CTAS <= GRID_CTAS <= 132):
    raise ValueError(
        "DG_TLE_MHC_LARGE_GRAPH_GRID_CTAS must keep every producer and fit "
        f"the 132-SM cooperative grid, got {GRID_CTAS} for {PRODUCER_CTAS=}"
    )
if not USE_NON_STRIPED_TAIL and GRID_CTAS * 4 < LOGICAL_TOKENS:
    raise ValueError(
        "striped role-4 ownership requires 4 * GRID_CTAS >= LOGICAL_TOKENS"
    )
if USE_TILE_LOCAL_BARRIER and (not USE_NON_STRIPED_TAIL or GRID_CTAS != PRODUCER_CTAS):
    raise ValueError(
        "tile-local barrier requires non-striped owners and an all-producer grid"
    )
PADDED_TOKENS = LOGICAL_TOKENS
NUM_PIPE_STAGES = 4 if USE_K128_PIPE4 else 8
ROOT_WARPS = 16
MATH_WARPS = 4
MATH_REGS = int(os.environ.get("DG_TLE_MHC_LARGE_GRAPH_MATH_REGS", "128"))
if MATH_REGS not in (80, 96, 112, 128):
    raise ValueError("DG_TLE_MHC_LARGE_GRAPH_MATH_REGS must be 80, 96, 112, or 128")
ROLE_WARPS = 4
NUM_ROLES = 4
PHYSICAL_WARPS = MATH_WARPS + NUM_ROLES * ROLE_WARPS
PHYSICAL_THREADS = PHYSICAL_WARPS * 32
ROLE_THREADS = ROLE_WARPS * 32
TAIL_HIDDEN_CHUNK = 1024
TAIL_ITEMS = HIDDEN // TAIL_HIDDEN_CHUNK

JIT_LOGICAL_TOKENS = tl.constexpr(LOGICAL_TOKENS)
JIT_PADDED_TOKENS = tl.constexpr(PADDED_TOKENS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_K_HALF = tl.constexpr(K_HALF)
JIT_K_STAGE = tl.constexpr(K_STAGE)
JIT_NUM_K_HALVES = tl.constexpr(NUM_K_HALVES)
JIT_NUM_M_TILES = tl.constexpr(NUM_M_TILES)
JIT_NUM_SPLITS = tl.constexpr(NUM_SPLITS)
JIT_BASE_HIDDEN_BLOCKS = tl.constexpr(BASE_HIDDEN_BLOCKS)
JIT_REMAIN_HIDDEN_BLOCKS = tl.constexpr(REMAIN_HIDDEN_BLOCKS)
JIT_MAX_HIDDEN_BLOCKS = tl.constexpr(MAX_HIDDEN_BLOCKS_PER_SPLIT)
JIT_PRODUCER_CTAS = tl.constexpr(PRODUCER_CTAS)
JIT_GRID_CTAS = tl.constexpr(GRID_CTAS)
JIT_NUM_PIPE_STAGES = tl.constexpr(NUM_PIPE_STAGES)
JIT_PHYSICAL_THREADS = tl.constexpr(PHYSICAL_THREADS)
JIT_ROOT_THREADS = tl.constexpr(ROOT_WARPS * 32)
JIT_MATH_WARPS = tl.constexpr(MATH_WARPS)
JIT_MATH_REGS = tl.constexpr(MATH_REGS)
JIT_NUM_ROLES = tl.constexpr(NUM_ROLES)
JIT_SINKHORN_ITERS = tl.constexpr(SINKHORN_ITERS)
JIT_ROLE_WARPS = tl.constexpr(ROLE_WARPS)
JIT_ROLE_THREADS = tl.constexpr(ROLE_THREADS)
JIT_TAIL_HIDDEN_CHUNK = tl.constexpr(TAIL_HIDDEN_CHUNK)
JIT_TAIL_ITEMS = tl.constexpr(TAIL_ITEMS)
JIT_PROBE_PHASE = tl.constexpr(PROBE_PHASE)
JIT_USE_WEIGHT_CACHE = tl.constexpr(USE_WEIGHT_CACHE)
JIT_USE_ROLE_OVERLAP = tl.constexpr(USE_ROLE_OVERLAP)
JIT_USE_ALLOCATED_ROLE_BARRIERS = tl.constexpr(USE_ALLOCATED_ROLE_BARRIERS)
JIT_USE_SHARED_ROLE_FLAGS = tl.constexpr(USE_SHARED_ROLE_FLAGS)
JIT_USE_EARLY_PRE_SYNC = tl.constexpr(USE_EARLY_PRE_SYNC)
JIT_USE_K128_PIPE4 = tl.constexpr(USE_K128_PIPE4)
JIT_USE_FN_CP_ASYNC = tl.constexpr(USE_FN_CP_ASYNC)
JIT_USE_BF16_RS_PIPE = tl.constexpr(USE_BF16_RS_PIPE)
JIT_USE_MINIMAL_SCRATCH = tl.constexpr(USE_MINIMAL_SCRATCH)
JIT_USE_COEFF_CACHE = tl.constexpr(USE_COEFF_CACHE)
JIT_USE_PTX_SINKHORN = tl.constexpr(USE_PTX_SINKHORN)
JIT_USE_NON_STRIPED_TAIL = tl.constexpr(USE_NON_STRIPED_TAIL)
JIT_USE_EXACT_T512 = tl.constexpr(USE_EXACT_T512)
JIT_USE_EXACT_T448 = tl.constexpr(USE_EXACT_T448)
JIT_USE_EXACT_FULL_TILE = tl.constexpr(USE_EXACT_FULL_TILE)
JIT_USE_MANUAL_ROLE4_TAIL = tl.constexpr(USE_MANUAL_ROLE4_TAIL)
JIT_USE_PACKED_MANUAL_ROLE4_TAIL = tl.constexpr(USE_PACKED_MANUAL_ROLE4_TAIL)
JIT_USE_FAST_GRID_BARRIER = tl.constexpr(USE_FAST_GRID_BARRIER)
JIT_USE_TILE_LOCAL_BARRIER = tl.constexpr(USE_TILE_LOCAL_BARRIER)
JIT_USE_TILE_LOCAL_FAST_BARRIER = tl.constexpr(USE_TILE_LOCAL_FAST_BARRIER)
JIT_USE_PTX_BF16X8_TAIL = tl.constexpr(USE_PTX_BF16X8_TAIL)


@triton.jit
def _sinkhorn_nineteen_warp_iterations(value, epsilon):
    """Run the repeated 4x4 Sinkhorn body as one warp-local PTX loop."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred again;
          .reg .u32 iteration;
          .reg .f32 x, t, total, inverse;
          mov.f32 x, $1;
          mov.u32 iteration, 19;
        DG_TLE_MHC_LARGE_SINKHORN_LOOP:
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
          @again bra.uni DG_TLE_MHC_LARGE_SINKHORN_LOOP;
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
def _role_barrier_arrive(role):
    """Publish one routing warp to its four root warps without waiting.

    PTX ``barrier.cta.arrive`` is intentionally the non-``.aligned`` form:
    only the five warps in this 160-thread role execute a given barrier.
    Barrier IDs 12..15 are disjoint from TLE's generated IDs 0..5.
    """
    barrier_id = role + 12
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.arrive $1, 160; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[barrier_id],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _role_barrier_sync(role):
    """Wait for one routing warp plus its four 128-thread root warps."""
    barrier_id = role + 12
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.sync $1, 160; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[barrier_id],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _allocated_role_barrier_arrive(role, b0, b1, b2, b3):
    """Publish one routing warp through a TLE-allocated 160-thread barrier."""
    if role == 0:
        tle.gpu.barrier_arrive(b0, phaseIdx=0)
    elif role == 1:
        tle.gpu.barrier_arrive(b1, phaseIdx=0)
    elif role == 2:
        tle.gpu.barrier_arrive(b2, phaseIdx=0)
    else:
        tle.gpu.barrier_arrive(b3, phaseIdx=0)


@triton.jit
def _allocated_role_barrier_wait(role, b0, b1, b2, b3):
    """Join four root warps with their matching routing warp."""
    if role == 0:
        tle.gpu.barrier_arrive(b0, phaseIdx=0)
        tle.gpu.barrier_wait(b0, phaseIdx=0)
    elif role == 1:
        tle.gpu.barrier_arrive(b1, phaseIdx=0)
        tle.gpu.barrier_wait(b1, phaseIdx=0)
    elif role == 2:
        tle.gpu.barrier_arrive(b2, phaseIdx=0)
        tle.gpu.barrier_wait(b2, phaseIdx=0)
    else:
        tle.gpu.barrier_arrive(b3, phaseIdx=0)
        tle.gpu.barrier_wait(b3, phaseIdx=0)


@triton.jit
def _shared_role_flag_publish(role_flags, role, lane):
    flag_pointer = tle.gpu.local_ptr(role_flags, (0,)) + role
    flag_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[flag_pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader;
          bar.warp.sync 0xffffffff;
          setp.eq.u32 leader, $2, 0;
          @leader st.release.cta.shared.u32 [$1], 1;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,r",
        args=[flag_address, lane],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _shared_role_flags_reset(role_flags, thread):
    base_pointer = tle.gpu.local_ptr(role_flags, (0,))
    base_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[base_pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred active;
          .reg .u32 offset, address;
          setp.lt.u32 active, $2, 4;
          shl.b32 offset, $2, 2;
          add.u32 address, $1, offset;
          @active st.shared.u32 [address], 0;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,r",
        args=[base_address, thread],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _shared_role_flag_wait(role_flags, role):
    flag_pointer = tle.gpu.local_ptr(role_flags, (0,)) + role
    flag_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[flag_pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred pending;
          .reg .u32 value;
        DG_TLE_MHC_LARGE_ROLE_FLAG_WAIT:
          ld.acquire.cta.shared.u32 value, [$1];
          setp.eq.u32 pending, value, 0;
          @pending bra DG_TLE_MHC_LARGE_ROLE_FLAG_WAIT;
          mov.u32 $0, value;
        }
        """,
        constraints="=r,r",
        args=[flag_address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _root_barrier_sync():
    """Make the shared BF16 weight row visible across sixteen root warps."""
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.sync 11, 512; mov.u32 $0, 0; }",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _manual_role4_hidden_sum_sync(physical_thread):
    """Synchronize each independent four-warp hidden worker."""
    worker = physical_thread // JIT_ROLE_THREADS
    barrier_id = worker + 12
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.sync $1, 128; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[barrier_id],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _warp_reduce32_sum(value):
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
def _full_tail_barrier_arrive():
    """Publish Pre from the four math warps without waiting for root."""
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.arrive 10, 640; mov.u32 $0, 0; }",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _full_tail_barrier_sync():
    """Wait for all 640 physical threads after Pre publication."""
    _ = tl.inline_asm_elementwise(
        asm="{ barrier.cta.sync 10, 640; mov.u32 $0, 0; }",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _barrier_wait_slot(
    slot: tl.constexpr,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
):
    if slot == 0:
        tle.gpu.barrier_wait(b0)
    elif slot == 1:
        tle.gpu.barrier_wait(b1)
    elif slot == 2:
        tle.gpu.barrier_wait(b2)
    elif slot == 3:
        tle.gpu.barrier_wait(b3)
    elif slot == 4:
        tle.gpu.barrier_wait(b4)
    elif slot == 5:
        tle.gpu.barrier_wait(b5)
    elif slot == 6:
        tle.gpu.barrier_wait(b6)
    else:
        tle.gpu.barrier_wait(b7)


@triton.jit
def _barrier_arrive_slot(
    slot: tl.constexpr,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
):
    if slot == 0:
        tle.gpu.barrier_arrive(b0)
    elif slot == 1:
        tle.gpu.barrier_arrive(b1)
    elif slot == 2:
        tle.gpu.barrier_arrive(b2)
    elif slot == 3:
        tle.gpu.barrier_arrive(b3)
    elif slot == 4:
        tle.gpu.barrier_arrive(b4)
    elif slot == 5:
        tle.gpu.barrier_arrive(b5)
    elif slot == 6:
        tle.gpu.barrier_arrive(b6)
    else:
        tle.gpu.barrier_arrive(b7)


@triton.jit
def _issue_fn_cp_async_k64(
    fn_ptr,
    fn_slot,
    flat_hidden_base,
):
    """Stage one 32x64 Fn tile with one aligned 16-byte copy per root thread.

    Rows 24..31 are hardware-zero-filled through ``src-size=0``.  The caller
    commits the per-thread async group immediately, overlaps the copy with the
    Post calculation, waits on that same issuing thread, and only then
    publishes the pipe slot to the math partition.
    """
    physical_thread = _t64._physical_thread_id_x()
    vectors_per_row: tl.constexpr = JIT_K_STAGE // 4
    row = physical_thread // vectors_per_row
    k_base = (physical_thread % vectors_per_row) * 4
    valid_row = row < JIT_PROJECTION_OUTPUTS
    safe_row = tl.where(valid_row, row, 0)
    global_address = tl.cast(
        fn_ptr + safe_row * JIT_FLAT_HIDDEN + flat_hidden_base + k_base,
        tl.uint64,
    )
    shared_address = _fn_cp._shared_address(tle.gpu.local_ptr(fn_slot, (row, k_base)))
    _ = _fn_cp._cp_async_cg_16b_zero_fill(
        shared_address,
        global_address,
        tl.where(valid_row, 16, 0),
    )


@triton.jit
def _cp_async_commit_group():
    return tl.inline_asm_elementwise(
        asm="{ cp.async.commit_group; mov.u32 $0, 0; }",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _cp_async_wait_group_zero():
    return tl.inline_asm_elementwise(
        asm="{ cp.async.wait_group 0; mov.u32 $0, 0; }",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _monotonic_grid_arrive_and_wait(counter_base, thread):
    """Generation barrier for the compile-time physical cooperative grid."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, pending;
          .reg .u64 old, one, divisor, generation, target, current;
          setp.eq.u32 leader, $2, 0;
          setp.ne.u32 pending, $2, $2;
          mov.u64 old, 0;
          mov.u64 one, 1;
          cvt.u64.u32 divisor, $3;
          @leader atom.release.gpu.global.add.u64 old, [$1], one;
          @leader div.u64 generation, old, divisor;
          @leader add.u64 generation, generation, 1;
          @leader mul.lo.u64 target, generation, divisor;
        DG_TLE_MHC_LARGE_GRID_WAIT:
          @leader ld.acquire.gpu.global.u64 current, [$1];
          @leader setp.lt.u64 pending, current, target;
          @pending bra DG_TLE_MHC_LARGE_GRID_WAIT;
          mov.u64 $0, old;
        }
        """,
        constraints="=l,l,r,r",
        args=[counter_base, thread, JIT_GRID_CTAS],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _tile_monotonic_arrive_and_wait(counter_base, tile, thread):
    """Release/acquire only the splits that produce one M64 token tile.

    Non-striped ownership never consumes another tile's partials.  Each tile
    therefore owns one 128-byte-spaced monotonic counter and only its
    ``NUM_SPLITS`` resident CTAs participate in this rendezvous.
    """
    tile_counter = counter_base + tile * 16
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, pending;
          .reg .u64 old, one, divisor, generation, target, current;
          setp.eq.u32 leader, $2, 0;
          setp.ne.u32 pending, $2, $2;
          mov.u64 old, 0;
          mov.u64 one, 1;
          cvt.u64.u32 divisor, $3;
          @leader atom.release.gpu.global.add.u64 old, [$1], one;
          @leader div.u64 generation, old, divisor;
          @leader add.u64 generation, generation, 1;
          @leader mul.lo.u64 target, generation, divisor;
        DG_TLE_MHC_LARGE_TILE_WAIT:
          @leader ld.acquire.gpu.global.u64 current, [$1];
          @leader setp.lt.u64 pending, current, target;
          @pending bra DG_TLE_MHC_LARGE_TILE_WAIT;
          mov.u64 $0, old;
        }
        """,
        constraints="=l,l,r,r",
        args=[tl.cast(tile_counter, tl.uint64), thread, JIT_NUM_SPLITS],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _tile_fast_arrive_and_wait(counter_base, tile, thread):
    """Two-word reusable barrier for one tile's resident split CTAs."""
    tile_state = counter_base + tile * 32
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, pending, last;
          .reg .u32 old, generation, next_generation, current;
          .reg .u64 generation_address;
          setp.eq.u32 leader, $2, 0;
          setp.ne.u32 pending, $2, $2;
          mov.u32 old, 0;
          mov.u32 generation, 0;
          add.u64 generation_address, $1, 4;
          @leader ld.acquire.gpu.global.u32 generation, [generation_address];
          @leader atom.acq_rel.gpu.global.inc.u32 old, [$1], $3;
          @leader setp.eq.u32 last, old, $3;
          @leader add.u32 next_generation, generation, 1;
          @last st.release.gpu.global.u32 [generation_address], next_generation;
        DG_TLE_MHC_LARGE_TILE_FAST_WAIT:
          @leader ld.acquire.gpu.global.u32 current, [generation_address];
          @leader setp.eq.u32 pending, current, generation;
          @pending bra DG_TLE_MHC_LARGE_TILE_FAST_WAIT;
          mov.u32 $0, old;
        }
        """,
        constraints="=r,l,r,r",
        args=[
            tl.cast(tile_state, tl.uint64),
            thread,
            JIT_NUM_SPLITS - 1,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _fast_grid_arrive_and_wait(counter_base, thread):
    """Reusable 32-bit arrival/generation barrier for Graph replay.

    Word zero is an ``atom.inc`` arrival counter that wraps at the physical
    grid size.  Word one is a monotonically increasing generation.  The
    acquire-release RMW chain publishes every CTA's partial stores to the last
    arrival; its release generation store then publishes that chain to every
    waiter.  Sequential CUDA Graph replays cannot enter the next generation
    until every CTA has returned from the current kernel.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, pending, last;
          .reg .u32 old, generation, next_generation, current, limit;
          .reg .u64 generation_address;
          setp.eq.u32 leader, $2, 0;
          setp.ne.u32 pending, $2, $2;
          mov.u32 old, 0;
          mov.u32 generation, 0;
          add.u64 generation_address, $1, 4;
          add.u32 limit, $3, 0xffffffff;
          @leader ld.acquire.gpu.global.u32 generation, [generation_address];
          @leader atom.acq_rel.gpu.global.inc.u32 old, [$1], limit;
          @leader setp.eq.u32 last, old, limit;
          @leader add.u32 next_generation, generation, 1;
          @last st.release.gpu.global.u32 [generation_address], next_generation;
        DG_TLE_MHC_LARGE_FAST_GRID_WAIT:
          @leader ld.acquire.gpu.global.u32 current, [generation_address];
          @leader setp.eq.u32 pending, current, generation;
          @pending bra DG_TLE_MHC_LARGE_FAST_GRID_WAIT;
          mov.u32 $0, old;
        }
        """,
        constraints="=r,l,r,r",
        args=[counter_base, thread, JIT_GRID_CTAS],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _role4_routing_finalize(
    pre_ready,
    pre_shared,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    packed_sum,
    token,
    role,
    lane,
    token_active,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    projection_mask = token_active & (lane < JIT_PROJECTION_OUTPUTS)
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
    tl.store(
        tle.gpu.local_ptr(pre_shared, (role, lane)),
        pre_mix,
        mask=token_active & (lane < JIT_HC_MULT),
    )
    # All four routing warps and all four hidden partitions share one
    # rendezvous.  This releases hidden work only after every live Pre vector
    # has been published, while Post/Sinkhorn continue after the nonblocking
    # math-partition arrival.
    tle.gpu.barrier_arrive(pre_ready)
    tl.store(
        new_post_mix_ptr + token * JIT_HC_MULT + (lane - JIT_HC_MULT),
        post_mix,
        mask=token_active & (lane >= JIT_HC_MULT) & (lane < 2 * JIT_HC_MULT),
    )

    routing_mask = token_active & (
        (lane >= 2 * JIT_HC_MULT) & (lane < JIT_PROJECTION_OUTPUTS)
    )
    sinkhorn = tl.where(routing_mask, mixes, 0.0)
    row_max = _t64._warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _t64._warp_reduce4_sum(sinkhorn)
    sinkhorn = sinkhorn * _t64._fast_rcp_ftz_f32(row_sum)
    sinkhorn += sinkhorn_eps
    column_sum = _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
    sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
    for _ in tl.range(1, JIT_SINKHORN_ITERS):
        row_sum = _t64._warp_reduce4_sum(sinkhorn) + sinkhorn_eps
        sinkhorn *= _t64._fast_rcp_ftz_f32(row_sum)
        column_sum = _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
        sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
    tl.store(
        new_comb_res_mix_ptr
        + token * JIT_HC_MULT * JIT_HC_MULT
        + (lane - 2 * JIT_HC_MULT),
        sinkhorn,
        mask=routing_mask,
    )


@triton.jit
def _pipe_role4_routing_finalize(
    pre_shared,
    role_flags,
    role_pre0,
    role_pre1,
    role_pre2,
    role_pre3,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    packed_sum,
    token,
    role,
    lane,
    token_active,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    projection_mask = token_active & (lane < JIT_PROJECTION_OUTPUTS)
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
    # Keep the same scalar lane-store lowering as the proven direct-routing
    # implementation.  A dynamic ``tl.store(local_ptr + role * 4 + lane)``
    # was observed to publish zeros for this warp-specialized layout even
    # though the projection reduction itself was numerically correct.
    pre_pointer = tle.gpu.local_ptr(pre_shared, (role, lane))
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
            tl.cast(token_active, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    if JIT_USE_ROLE_OVERLAP:
        if JIT_USE_SHARED_ROLE_FLAGS:
            _shared_role_flag_publish(role_flags, role, lane)
        elif JIT_USE_ALLOCATED_ROLE_BARRIERS:
            _allocated_role_barrier_arrive(
                role, role_pre0, role_pre1, role_pre2, role_pre3
            )
        else:
            _role_barrier_arrive(role)
    elif JIT_USE_EARLY_PRE_SYNC:
        # The four routing warps publish Pre and arrive without waiting.  The
        # sixteen root warps wait on the same 640-thread TLE barrier, then run
        # hidden/RMSNorm while routing continues through Post and Sinkhorn.
        _full_tail_barrier_arrive()
    post_base = tl.cast(
        new_post_mix_ptr + token * JIT_HC_MULT,
        tl.uint64,
    )
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
            tl.cast(token_active, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    routing_mask = token_active & (
        (lane >= 2 * JIT_HC_MULT) & (lane < JIT_PROJECTION_OUTPUTS)
    )
    sinkhorn = tl.where(routing_mask, mixes, 0.0)
    row_max = _t64._warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _t64._warp_reduce4_sum(sinkhorn)
    sinkhorn *= _t64._fast_rcp_ftz_f32(row_sum)
    sinkhorn += sinkhorn_eps
    column_sum = _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
    sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
    if JIT_USE_PTX_SINKHORN:
        sinkhorn = _sinkhorn_nineteen_warp_iterations(sinkhorn, sinkhorn_eps)
    else:
        for _ in tl.range(1, JIT_SINKHORN_ITERS):
            row_sum = _t64._warp_reduce4_sum(sinkhorn) + sinkhorn_eps
            sinkhorn *= _t64._fast_rcp_ftz_f32(row_sum)
            column_sum = _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
            sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
    comb_base = tl.cast(
        new_comb_res_mix_ptr + token * JIT_HC_MULT * JIT_HC_MULT,
        tl.uint64,
    )
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
            tl.cast(token_active, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _pipe_role4_hidden_tail(
    weight_shared,
    pre_shared,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    pid,
    tile,
    split,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Run four H4096 tails with the rounded BF16 values kept in registers.

    The 512-thread root partition is reinterpreted as four 128-thread token
    workers.  Each physical thread owns eight hidden values for each of its
    worker's four logical 1024-wide stripes (32 values total), matching the
    production CUDA role-overlap geometry.  Retaining the rounded values
    eliminates the former 32-KiB shared write/read round trip and collapses
    sixteen chunk reductions into four whole-token reductions.
    """
    role = tl.arange(0, JIT_NUM_ROLES)[:, None]
    hidden = tl.arange(0, JIT_HIDDEN)[None, :]
    if JIT_USE_NON_STRIPED_TAIL:
        token = tile * JIT_BLOCK_M + split * JIT_NUM_ROLES + role
    else:
        token = pid + role * JIT_GRID_CTAS
    if JIT_USE_EXACT_FULL_TILE:
        token_active = tl.full((JIT_NUM_ROLES, 1), True, tl.int1)
    else:
        token_active = token < JIT_LOGICAL_TOKENS
    pre_base = tle.gpu.local_ptr(pre_shared, (0, 0))
    norm_input = tl.zeros((JIT_NUM_ROLES, JIT_HIDDEN), tl.float32)
    for channel in tl.static_range(0, JIT_HC_MULT):
        value = tl.load(
            new_residual_ptr + token * JIT_FLAT_HIDDEN + channel * JIT_HIDDEN + hidden,
            mask=token_active,
            other=0.0,
        ).to(tl.float32)
        pre0 = tl.load(pre_base + channel)
        pre1 = tl.load(pre_base + JIT_HC_MULT + channel)
        pre2 = tl.load(pre_base + 2 * JIT_HC_MULT + channel)
        pre3 = tl.load(pre_base + 3 * JIT_HC_MULT + channel)
        pre = tl.where(
            role == 0,
            pre0,
            tl.where(role == 1, pre1, tl.where(role == 2, pre2, pre3)),
        )
        norm_input += value * pre
    rounded = norm_input.to(tl.bfloat16, fp_downcast_rounding="rtne")
    rounded_f32 = rounded.to(tl.float32)
    squares = tl.where(token_active, rounded_f32 * rounded_f32, 0.0)
    rms = tl.rsqrt(tl.sum(squares, axis=1) / JIT_HIDDEN + rmsnorm_eps)
    if JIT_USE_WEIGHT_CACHE:
        weight = tl.load(tle.gpu.local_ptr(weight_shared, (0,)) + hidden).to(tl.float32)
    else:
        weight = tl.load(rmsnorm_weight_ptr + hidden).to(tl.float32)
    y = rounded_f32 * rms[:, None] * weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + token * JIT_HIDDEN + hidden,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=token_active,
    )


@triton.jit
def _k128_compute_channel_half(
    post_mix_ptr,
    comb_res_mix_ptr,
    new_residual_ptr,
    m,
    hidden,
    token_active,
    x,
    r0,
    r1,
    r2,
    r3,
    CHANNEL: tl.constexpr,
):
    """Compute/store one channel K64 half before K128 pipe staging."""
    post = tl.load(
        post_mix_ptr + m[:, None] * JIT_HC_MULT + CHANNEL,
        mask=token_active[:, None],
        other=0.0,
    ).to(tl.float32)
    comb_base = m[:, None] * JIT_HC_MULT * JIT_HC_MULT + CHANNEL
    c0 = tl.load(
        comb_res_mix_ptr + comb_base,
        mask=token_active[:, None],
        other=0.0,
    ).to(tl.float32)
    c1 = tl.load(
        comb_res_mix_ptr + comb_base + JIT_HC_MULT,
        mask=token_active[:, None],
        other=0.0,
    ).to(tl.float32)
    c2 = tl.load(
        comb_res_mix_ptr + comb_base + 2 * JIT_HC_MULT,
        mask=token_active[:, None],
        other=0.0,
    ).to(tl.float32)
    c3 = tl.load(
        comb_res_mix_ptr + comb_base + 3 * JIT_HC_MULT,
        mask=token_active[:, None],
        other=0.0,
    ).to(tl.float32)
    q = x * post + r0 * c0 + r1 * c1 + r2 * c2 + r3 * c3
    q_bf16 = q.to(tl.bfloat16, fp_downcast_rounding="rtne")
    q_fp32 = q_bf16.to(tl.float32)
    tl.store(
        new_residual_ptr
        + m[:, None] * JIT_FLAT_HIDDEN
        + CHANNEL * JIT_HIDDEN
        + hidden[None, :],
        q_bf16,
        mask=token_active[:, None],
    )
    return tl.sum(q_fp32 * q_fp32, axis=1)


@triton.jit
def _pipe8_root_producer(
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
    tail_ready,
    role_flags,
    role_pre0,
    role_pre1,
    role_pre2,
    role_pre3,
    coeff_ready,
    post_coeff_shared,
    comb_coeff_shared,
    weight_shared,
    pre_shared,
    warp_sqr_sums,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    grid_counter_ptr,
    tile,
    split,
    pid,
    is_producer,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Sixteen root warps produce one M64 tile through an eight-slot pipe."""
    if JIT_USE_SHARED_ROLE_FLAGS:
        init_thread = _t64._physical_thread_id_x()
        _shared_role_flags_reset(role_flags, init_thread)
    m = tile * JIT_BLOCK_M + tl.arange(0, JIT_BLOCK_M)
    if JIT_USE_EXACT_FULL_TILE:
        token_active = tl.full((JIT_BLOCK_M,), True, tl.int1)
    else:
        token_active = is_producer & (m < JIT_LOGICAL_TOKENS)
    hidden_blocks = JIT_BASE_HIDDEN_BLOCKS + (split < JIT_REMAIN_HIDDEN_BLOCKS)
    hidden_block_offset = split * JIT_BASE_HIDDEN_BLOCKS + tl.minimum(
        split, JIT_REMAIN_HIDDEN_BLOCKS
    )
    squared_sum = tl.zeros((JIT_BLOCK_M,), tl.float32)
    n = tl.arange(0, JIT_BLOCK_N)

    if JIT_USE_COEFF_CACHE:
        local_m = tl.arange(0, JIT_BLOCK_M)[:, None]
        post_channel = tl.arange(0, JIT_HC_MULT)[None, :]
        comb_channel = tl.arange(0, JIT_HC_MULT * JIT_HC_MULT)[None, :]
        tl.store(
            tle.gpu.local_ptr(post_coeff_shared, (0, 0))
            + local_m * JIT_HC_MULT
            + post_channel,
            tl.load(
                post_mix_ptr + m[:, None] * JIT_HC_MULT + post_channel,
                mask=token_active[:, None],
                other=0.0,
            ),
        )
        tl.store(
            tle.gpu.local_ptr(comb_coeff_shared, (0, 0))
            + local_m * JIT_HC_MULT * JIT_HC_MULT
            + comb_channel,
            tl.load(
                comb_res_mix_ptr
                + m[:, None] * JIT_HC_MULT * JIT_HC_MULT
                + comb_channel,
                mask=token_active[:, None],
                other=0.0,
            ),
        )
        tle.gpu.barrier_wait(coeff_ready)

    # A K128 stage cuts pipe handoffs in half.  Compute every channel from one
    # K64 x/residual load and write the BF16 rounding boundary first.  After a
    # root-partition barrier, each K128 pipe slot is staged by reloading those
    # rounded values.  This keeps the same global x/residual traffic as K64,
    # avoids K128 register live ranges, and preserves the pipe verifier's
    # acquire->payload copies->commit lifecycle.
    for hb in tl.static_range(0, JIT_MAX_HIDDEN_BLOCKS):
        if JIT_USE_K128_PIPE4 and is_producer & (hb < hidden_blocks):
            for subhalf in tl.static_range(0, 2):
                hidden = (
                    (hidden_block_offset + hb) * JIT_K_STAGE
                    + subhalf * 64
                    + tl.arange(0, 64)
                )
                token_hidden = m[:, None] * JIT_HIDDEN + hidden[None, :]
                token_residual = m[:, None] * JIT_FLAT_HIDDEN + hidden[None, :]
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
                    squared_sum += _k128_compute_channel_half(
                        post_mix_ptr,
                        comb_res_mix_ptr,
                        new_residual_ptr,
                        m,
                        hidden,
                        token_active,
                        x,
                        r0,
                        r1,
                        r2,
                        r3,
                        CHANNEL=channel,
                    )

    if JIT_USE_K128_PIPE4:
        tle.gpu.barrier_wait(coeff_ready)
        for hb in tl.static_range(0, JIT_MAX_HIDDEN_BLOCKS):
            if is_producer & (hb < hidden_blocks):
                hidden = (hidden_block_offset + hb) * JIT_K_STAGE + tl.arange(
                    0, JIT_K_HALF
                )
                for channel in tl.static_range(0, JIT_HC_MULT):
                    iteration = hb * JIT_HC_MULT + channel
                    slot = pipe_writer.acquire(iteration)
                    rounded = tl.load(
                        new_residual_ptr
                        + m[:, None] * JIT_FLAT_HIDDEN
                        + channel * JIT_HIDDEN
                        + hidden[None, :],
                        mask=token_active[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    fn_tile = tl.load(
                        fn_ptr
                        + n[:, None] * JIT_FLAT_HIDDEN
                        + channel * JIT_HIDDEN
                        + hidden[None, :],
                        mask=n[:, None] < JIT_PROJECTION_OUTPUTS,
                        other=0.0,
                    ).to(tl.float32)
                    tl.store(tle.gpu.local_ptr(slot.q), rounded)
                    tl.store(tle.gpu.local_ptr(slot.fn), fn_tile)
                    pipe_writer.commit(iteration)

    for hb in tl.static_range(0, JIT_MAX_HIDDEN_BLOCKS):
        if (not JIT_USE_K128_PIPE4) and is_producer & (hb < hidden_blocks):
            # K32 uses two physical stages per H64; the experimental true-K64
            # mode uses one.  Pipe endpoint operations stay inline in both
            # worker partitions because helper specialization can corrupt the
            # stage/phase state in this TLE revision.
            for half in tl.static_range(0, JIT_NUM_K_HALVES):
                hidden = (
                    (hidden_block_offset + hb) * JIT_K_STAGE
                    + half * JIT_K_HALF
                    + tl.arange(0, JIT_K_HALF)
                )
                token_hidden = m[:, None] * JIT_HIDDEN + hidden[None, :]
                token_residual = m[:, None] * JIT_FLAT_HIDDEN + hidden[None, :]
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
                    iteration = (
                        hb * JIT_NUM_K_HALVES * JIT_HC_MULT
                        + half * JIT_HC_MULT
                        + channel
                    )
                    slot = pipe_writer.acquire(iteration)
                    if JIT_USE_FN_CP_ASYNC:
                        flat_hidden_base = (
                            channel * JIT_HIDDEN
                            + (hidden_block_offset + hb) * JIT_K_STAGE
                            + half * JIT_K_HALF
                        )
                        _issue_fn_cp_async_k64(
                            fn_ptr,
                            slot.fn,
                            flat_hidden_base,
                        )
                        _ = _cp_async_commit_group()
                    if JIT_USE_COEFF_CACHE:
                        post = tl.load(
                            tle.gpu.local_ptr(post_coeff_shared, (0, 0))
                            + tl.arange(0, JIT_BLOCK_M)[:, None] * JIT_HC_MULT
                            + channel
                        ).to(tl.float32)
                    else:
                        post = tl.load(
                            post_mix_ptr + m[:, None] * JIT_HC_MULT + channel,
                            mask=token_active[:, None],
                            other=0.0,
                        ).to(tl.float32)
                    comb_base = m[:, None] * JIT_HC_MULT * JIT_HC_MULT + channel
                    if JIT_USE_COEFF_CACHE:
                        local_comb_base = (
                            tl.arange(0, JIT_BLOCK_M)[:, None]
                            * JIT_HC_MULT
                            * JIT_HC_MULT
                            + channel
                        )
                        comb_ptr = tle.gpu.local_ptr(comb_coeff_shared, (0, 0))
                        c0 = tl.load(comb_ptr + local_comb_base).to(tl.float32)
                        c1 = tl.load(comb_ptr + local_comb_base + JIT_HC_MULT).to(
                            tl.float32
                        )
                        c2 = tl.load(comb_ptr + local_comb_base + 2 * JIT_HC_MULT).to(
                            tl.float32
                        )
                        c3 = tl.load(comb_ptr + local_comb_base + 3 * JIT_HC_MULT).to(
                            tl.float32
                        )
                    else:
                        c0 = tl.load(
                            comb_res_mix_ptr + comb_base,
                            mask=token_active[:, None],
                            other=0.0,
                        ).to(tl.float32)
                        c1 = tl.load(
                            comb_res_mix_ptr + comb_base + JIT_HC_MULT,
                            mask=token_active[:, None],
                            other=0.0,
                        ).to(tl.float32)
                        c2 = tl.load(
                            comb_res_mix_ptr + comb_base + 2 * JIT_HC_MULT,
                            mask=token_active[:, None],
                            other=0.0,
                        ).to(tl.float32)
                        c3 = tl.load(
                            comb_res_mix_ptr + comb_base + 3 * JIT_HC_MULT,
                            mask=token_active[:, None],
                            other=0.0,
                        ).to(tl.float32)
                    q = x * post + r0 * c0 + r1 * c1 + r2 * c2 + r3 * c3
                    q_bf16 = q.to(tl.bfloat16, fp_downcast_rounding="rtne")
                    q_fp32 = q_bf16.to(tl.float32)
                    tl.store(
                        new_residual_ptr
                        + m[:, None] * JIT_FLAT_HIDDEN
                        + channel * JIT_HIDDEN
                        + hidden[None, :],
                        q_bf16,
                        mask=token_active[:, None],
                    )
                    squared_sum += tl.sum(q_fp32 * q_fp32, axis=1)
                    if JIT_USE_FN_CP_ASYNC:
                        _ = _cp_async_wait_group_zero()
                    else:
                        fn_tile = tl.load(
                            fn_ptr
                            + n[:, None] * JIT_FLAT_HIDDEN
                            + channel * JIT_HIDDEN
                            + hidden[None, :],
                            mask=n[:, None] < JIT_PROJECTION_OUTPUTS,
                            other=0.0,
                        ).to(tl.float32)
                        tl.store(tle.gpu.local_ptr(slot.fn), fn_tile)
                    if JIT_USE_BF16_RS_PIPE:
                        tl.store(tle.gpu.local_ptr(slot.q), q_bf16)
                    else:
                        tl.store(tle.gpu.local_ptr(slot.q), q_fp32)
                    pipe_writer.commit(iteration)

    if is_producer:
        sqsum_ptr = (
            partials_ptr
            + split * JIT_PADDED_TOKENS * JIT_BLOCK_N
            + m * JIT_BLOCK_N
            + JIT_PROJECTION_OUTPUTS
        )
        if JIT_USE_EXACT_FULL_TILE:
            tl.store(sqsum_ptr, squared_sum)
        else:
            tl.store(
                sqsum_ptr,
                squared_sum,
                mask=m < JIT_LOGICAL_TOKENS,
            )
    tle.gpu.barrier_wait(producer_ready)
    if JIT_PROBE_PHASE >= 2:
        physical_thread = _t64._physical_thread_id_x()
        if JIT_USE_TILE_LOCAL_BARRIER:
            if JIT_USE_TILE_LOCAL_FAST_BARRIER:
                _ = _tile_fast_arrive_and_wait(grid_counter_ptr, tile, physical_thread)
            else:
                _ = _tile_monotonic_arrive_and_wait(
                    grid_counter_ptr, tile, physical_thread
                )
        elif JIT_USE_FAST_GRID_BARRIER:
            _ = _fast_grid_arrive_and_wait(
                tl.cast(grid_counter_ptr, tl.uint64), physical_thread
            )
        else:
            _ = _monotonic_grid_arrive_and_wait(
                tl.cast(grid_counter_ptr, tl.uint64), physical_thread
            )
        tle.gpu.barrier_wait(grid_acquire_ready)
        if JIT_PROBE_PHASE >= 3:
            if JIT_PROBE_PHASE >= 4 and JIT_USE_WEIGHT_CACHE:
                weight_hidden = tl.arange(0, JIT_HIDDEN)
                tl.store(
                    tle.gpu.local_ptr(weight_shared, (0,)) + weight_hidden,
                    tl.load(rmsnorm_weight_ptr + weight_hidden),
                )
                if JIT_USE_ROLE_OVERLAP:
                    _root_barrier_sync()
            if JIT_USE_ROLE_OVERLAP:
                root_role = physical_thread // JIT_ROLE_THREADS
                if JIT_USE_SHARED_ROLE_FLAGS:
                    _shared_role_flag_wait(role_flags, root_role)
                elif JIT_USE_ALLOCATED_ROLE_BARRIERS:
                    _allocated_role_barrier_wait(
                        root_role,
                        role_pre0,
                        role_pre1,
                        role_pre2,
                        role_pre3,
                    )
                else:
                    _role_barrier_sync(root_role)
            elif JIT_USE_EARLY_PRE_SYNC:
                _full_tail_barrier_sync()
            else:
                tle.gpu.barrier_wait(tail_ready)
            if JIT_PROBE_PHASE >= 4:
                _pipe_role4_hidden_tail(
                    weight_shared,
                    pre_shared,
                    new_residual_ptr,
                    rmsnorm_weight_ptr,
                    y_bf16_ptr,
                    pid,
                    tile,
                    split,
                    rmsnorm_eps,
                    rmsnorm_scale,
                )


@triton.jit
def _pipe8_k32_math_body(pipe_reader, NUM_HIDDEN_BLOCKS: tl.constexpr):
    """Consume K32 stages, draining once per H64 block."""
    accumulator = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), tl.float32)
    for hb in tl.static_range(0, NUM_HIDDEN_BLOCKS):
        for stage in tl.static_range(0, 2 * JIT_HC_MULT):
            iteration = hb * 2 * JIT_HC_MULT + stage
            wait = pipe_reader.wait(iteration)
            accumulator = tle.gpu.wgmma(
                wait.slot.q,
                wait.slot.fn,
                accumulator,
                input_precision="tf32",
                trans_b=True,
            )
            if stage > 0:
                accumulator = tle.gpu.wgmma_wait(1, accumulator)
                pipe_reader.release(iteration - 1)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        pipe_reader.release((hb + 1) * 2 * JIT_HC_MULT - 1)
    return accumulator


@triton.jit
def _pipe8_math_and_routing(
    partials_ptr,
    pipe_reader,
    producer_ready,
    grid_acquire_ready,
    tail_ready,
    role_flags,
    role_pre0,
    role_pre1,
    role_pre2,
    role_pre3,
    pre_shared,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    tile,
    split,
    pid,
    is_producer,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    accumulator = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), tl.float32)
    hidden_blocks = JIT_BASE_HIDDEN_BLOCKS + (split < JIT_REMAIN_HIDDEN_BLOCKS)
    # Keep pipe endpoint operations directly in the worker body.  This is the
    # correctness-proven form for this TLE revision; passing a runtime pipe
    # endpoint through mutually exclusive static helper specializations can
    # miscompile its stage/phase state.
    for hb in tl.static_range(0, JIT_MAX_HIDDEN_BLOCKS):
        if is_producer & (hb < hidden_blocks):
            for half in tl.static_range(0, JIT_NUM_K_HALVES):
                for channel in tl.static_range(0, JIT_HC_MULT):
                    stage = half * JIT_HC_MULT + channel
                    iteration = hb * JIT_NUM_K_HALVES * JIT_HC_MULT + stage
                    wait = pipe_reader.wait(iteration)
                    if JIT_USE_BF16_RS_PIPE:
                        q_operand = tl.load(tle.gpu.local_ptr(wait.slot.q)).to(
                            tl.float32
                        )
                    else:
                        q_operand = wait.slot.q
                    accumulator = tle.gpu.wgmma(
                        q_operand,
                        wait.slot.fn,
                        accumulator,
                        input_precision="tf32",
                        trans_b=True,
                    )
                    if stage > 0:
                        accumulator = tle.gpu.wgmma_wait(1, accumulator)
                        pipe_reader.release(iteration - 1)
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
            pipe_reader.release((hb + 1) * JIT_NUM_K_HALVES * JIT_HC_MULT - 1)

    if is_producer:
        m = tile * JIT_BLOCK_M + tl.arange(0, JIT_BLOCK_M)
        n = tl.arange(0, JIT_BLOCK_N)
        projection_ptr = (
            partials_ptr
            + split * JIT_PADDED_TOKENS * JIT_BLOCK_N
            + m[:, None] * JIT_BLOCK_N
            + n[None, :]
        )
        if JIT_USE_EXACT_FULL_TILE:
            tl.store(
                projection_ptr,
                accumulator,
                mask=n[None, :] < JIT_PROJECTION_OUTPUTS,
            )
        else:
            tl.store(
                projection_ptr,
                accumulator,
                mask=(m[:, None] < JIT_LOGICAL_TOKENS)
                & (n[None, :] < JIT_PROJECTION_OUTPUTS),
            )

    tle.gpu.barrier_wait(producer_ready)
    if JIT_PROBE_PHASE >= 2:
        tle.gpu.barrier_wait(grid_acquire_ready)
        if JIT_PROBE_PHASE >= 3:
            physical_thread = _t64._physical_thread_id_x()
            lane = tl.inline_asm_elementwise(
                asm="mov.u32 $0, %laneid;",
                constraints="=r",
                args=[],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            role = (physical_thread - JIT_ROOT_THREADS) // 32
            if JIT_USE_NON_STRIPED_TAIL:
                token = tile * JIT_BLOCK_M + split * JIT_NUM_ROLES + role
            else:
                token = pid + role * JIT_GRID_CTAS
            if JIT_USE_EXACT_FULL_TILE:
                token_active = tl.constexpr(True)
            else:
                token_active = token < JIT_LOGICAL_TOKENS
            packed_sum = 0.0
            for source_split in tl.static_range(0, JIT_NUM_SPLITS):
                packed_sum += tl.load(
                    partials_ptr
                    + source_split * JIT_PADDED_TOKENS * JIT_BLOCK_N
                    + token * JIT_BLOCK_N
                    + lane,
                    mask=token_active & (lane <= JIT_PROJECTION_OUTPUTS),
                    other=0.0,
                )
            _pipe_role4_routing_finalize(
                pre_shared,
                role_flags,
                role_pre0,
                role_pre1,
                role_pre2,
                role_pre3,
                mix_scales_ptr,
                mix_bases_ptr,
                new_post_mix_ptr,
                new_comb_res_mix_ptr,
                packed_sum,
                token,
                role,
                lane,
                token_active,
                hc_norm_eps,
                hc_pre_eps,
                hc_post_scale,
                sinkhorn_eps,
            )
            if not JIT_USE_ROLE_OVERLAP and not JIT_USE_EARLY_PRE_SYNC:
                tle.gpu.barrier_wait(tail_ready)


@triton.jit
def _sm90_large_graph_role4_k32_wgmma_kernel(
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
    if JIT_USE_EXACT_FULL_TILE:
        is_producer = tl.constexpr(True)
        producer_pid = pid
    else:
        is_producer = pid < JIT_PRODUCER_CTAS
        producer_pid = tl.where(is_producer, pid, 0)
    tile = producer_pid // JIT_NUM_SPLITS
    split = producer_pid % JIT_NUM_SPLITS
    tl.static_assert(
        (JIT_USE_K128_PIPE4 and JIT_NUM_PIPE_STAGES == 4)
        or (not JIT_USE_K128_PIPE4 and JIT_NUM_PIPE_STAGES == 8)
    )
    tl.static_assert(JIT_GRID_CTAS <= 132)
    tl.static_assert(JIT_PRODUCER_CTAS <= JIT_GRID_CTAS)
    tl.static_assert(JIT_PHYSICAL_THREADS == 640)

    if JIT_USE_BF16_RS_PIPE:
        q_stages = tle.gpu.alloc(
            [JIT_NUM_PIPE_STAGES, JIT_BLOCK_M, JIT_K_HALF],
            dtype=tl.bfloat16,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
    else:
        q_stages = tle.gpu.alloc(
            [JIT_NUM_PIPE_STAGES, JIT_BLOCK_M, JIT_K_HALF],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
    fn_stages = tle.gpu.alloc(
        [JIT_NUM_PIPE_STAGES, JIT_BLOCK_N, JIT_K_HALF],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    projection_pipe = tle.pipe(
        capacity=JIT_NUM_PIPE_STAGES,
        scope="cta",
        name="mhc_large_role4_projection_pipe",
        q=q_stages,
        fn=fn_stages,
    )
    pre_shared = tle.gpu.alloc(
        [JIT_NUM_ROLES, JIT_HC_MULT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    role_flags = tle.gpu.alloc(
        [JIT_NUM_ROLES],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    if JIT_USE_MINIMAL_SCRATCH:
        weight_shared = tle.gpu.alloc(
            [1],
            dtype=tl.bfloat16,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        post_coeff_shared = tle.gpu.alloc(
            [1, 1],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        comb_coeff_shared = tle.gpu.alloc(
            [1, 1],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        warp_sqr_sums = tle.gpu.alloc(
            [1, 1],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    else:
        weight_shared = tle.gpu.alloc(
            [JIT_HIDDEN],
            dtype=tl.bfloat16,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        post_coeff_shared = tle.gpu.alloc(
            [JIT_BLOCK_M, JIT_HC_MULT],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        comb_coeff_shared = tle.gpu.alloc(
            [JIT_BLOCK_M, JIT_HC_MULT * JIT_HC_MULT],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        warp_sqr_sums = tle.gpu.alloc(
            [JIT_NUM_ROLES, JIT_ROLE_WARPS],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    producer_ready = tle.gpu.alloc_barrier(arrive_count=JIT_PHYSICAL_THREADS)
    grid_acquire_ready = tle.gpu.alloc_barrier(arrive_count=JIT_PHYSICAL_THREADS)
    if JIT_USE_MINIMAL_SCRATCH:
        tail_ready = producer_ready
        coeff_ready = producer_ready
    else:
        tail_ready = tle.gpu.alloc_barrier(arrive_count=JIT_PHYSICAL_THREADS)
        coeff_ready = tle.gpu.alloc_barrier(arrive_count=JIT_ROOT_THREADS)
    if JIT_USE_ALLOCATED_ROLE_BARRIERS:
        role_pre0 = tle.gpu.alloc_barrier(arrive_count=160)
        role_pre1 = tle.gpu.alloc_barrier(arrive_count=160)
        role_pre2 = tle.gpu.alloc_barrier(arrive_count=160)
        role_pre3 = tle.gpu.alloc_barrier(arrive_count=160)
    else:
        role_pre0 = tail_ready
        role_pre1 = tail_ready
        role_pre2 = tail_ready
        role_pre3 = tail_ready
    tle.gpu.warp_specialize(
        [
            (
                _pipe8_root_producer,
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
                    tail_ready,
                    role_flags,
                    role_pre0,
                    role_pre1,
                    role_pre2,
                    role_pre3,
                    coeff_ready,
                    post_coeff_shared,
                    comb_coeff_shared,
                    weight_shared,
                    pre_shared,
                    warp_sqr_sums,
                    rmsnorm_weight_ptr,
                    y_bf16_ptr,
                    grid_counter_ptr,
                    tile,
                    split,
                    pid,
                    is_producer,
                    rmsnorm_eps,
                    rmsnorm_scale,
                ),
            ),
            (
                _pipe8_math_and_routing,
                (
                    partials_ptr,
                    projection_pipe.reader(),
                    producer_ready,
                    grid_acquire_ready,
                    tail_ready,
                    role_flags,
                    role_pre0,
                    role_pre1,
                    role_pre2,
                    role_pre3,
                    pre_shared,
                    mix_scales_ptr,
                    mix_bases_ptr,
                    new_post_mix_ptr,
                    new_comb_res_mix_ptr,
                    tile,
                    split,
                    pid,
                    is_producer,
                    hc_norm_eps,
                    hc_pre_eps,
                    hc_post_scale,
                    sinkhorn_eps,
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
class LargeGraphRole4Plan:
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
        return _sm90_large_graph_role4_k32_wgmma_kernel.warmup(
            *self._kernel_args(),
            grid=(GRID_CTAS,),
            num_warps=ROOT_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )

    def run(self) -> None:
        _sm90_large_graph_role4_k32_wgmma_kernel[(GRID_CTAS,)](
            *self._kernel_args(),
            num_warps=ROOT_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> LargeGraphRole4Plan:
    if os.environ.get("DG_TLE_MHC_LARGE_GRAPH_ROLE4", "0") != "1":
        raise RuntimeError(
            "set DG_TLE_MHC_LARGE_GRAPH_ROLE4=1 to enable this experiment"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the plan before CUDA Graph capture")
    if inputs.get("shifted_prev_mix") is not None:
        raise ValueError("only Normal MHC (shifted_prev_mix=None) is supported")

    specs = (
        (inputs.get("x"), "inputs.x", (LOGICAL_TOKENS, HIDDEN), torch.bfloat16),
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
    if USE_TILE_LOCAL_BARRIER:
        grid_counter = torch.zeros(
            (
                NUM_M_TILES,
                32 if USE_TILE_LOCAL_FAST_BARRIER else 16,
            ),
            dtype=(torch.uint32 if USE_TILE_LOCAL_FAST_BARRIER else torch.uint64),
            device=device,
        )
    else:
        grid_counter = torch.zeros(
            (2,) if USE_FAST_GRID_BARRIER else (1,),
            dtype=torch.uint32 if USE_FAST_GRID_BARRIER else torch.uint64,
            device=device,
        )
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
        "kernel_symbol": "_sm90_large_graph_role4_k32_wgmma_kernel",
        "CTA": GRID_CTAS,
        "grid_ctas": GRID_CTAS,
        "producer_ctas": PRODUCER_CTAS,
        "consumer_only_ctas": GRID_CTAS - PRODUCER_CTAS,
        "num_m_tiles": NUM_M_TILES,
        "num_splits": NUM_SPLITS,
        "base_hidden_blocks_per_split": BASE_HIDDEN_BLOCKS,
        "remain_hidden_blocks": REMAIN_HIDDEN_BLOCKS,
        "pipeline_k": K_HALF,
        "pipeline_slots": NUM_PIPE_STAGES,
        "true_k64_stage": USE_TRUE_K64_STAGE,
        "k128_pipe4": USE_K128_PIPE4,
        "fn_cp_async_stage": USE_FN_CP_ASYNC,
        "bf16_rs_pipe": USE_BF16_RS_PIPE,
        "minimal_scratch": USE_MINIMAL_SCRATCH,
        "coefficient_cache_bytes": (
            BLOCK_M * (HC_MULT + HC_MULT * HC_MULT) * 4 if USE_COEFF_CACHE else 0
        ),
        "ptx_sinkhorn_loop": USE_PTX_SINKHORN,
        "non_striped_tail": USE_NON_STRIPED_TAIL,
        "exact_t512": USE_EXACT_T512,
        "exact_t448": USE_EXACT_T448,
        "exact_full_tile": USE_EXACT_FULL_TILE,
        "manual_role4_tail": USE_MANUAL_ROLE4_TAIL,
        "packed_manual_role4_tail": USE_PACKED_MANUAL_ROLE4_TAIL,
        "fast_grid_barrier": USE_FAST_GRID_BARRIER,
        "tile_local_barrier": USE_TILE_LOCAL_BARRIER,
        "tile_local_fast_barrier": USE_TILE_LOCAL_FAST_BARRIER,
        "tile_counter_stride_bytes": 128 if USE_TILE_LOCAL_BARRIER else 0,
        "ptx_bf16x8_tail": USE_PTX_BF16X8_TAIL,
        "cross_h64_wgmma_pipeline": False,
        "pipeline_implementation": "tle.pipe",
        "root_warps": ROOT_WARPS,
        "math_warps": MATH_WARPS,
        "math_regs": MATH_REGS,
        "tail_roles": NUM_ROLES,
        "tail_hidden_chunk": TAIL_HIDDEN_CHUNK,
        "tail_shared_bytes": 0,
        "rms_weight_cache_bytes": HIDDEN * 2 if USE_WEIGHT_CACHE else 0,
        "role_overlap": USE_ROLE_OVERLAP,
        "allocated_role_barriers": USE_ALLOCATED_ROLE_BARRIERS,
        "shared_role_flags": USE_SHARED_ROLE_FLAGS,
        "role_barrier_ids": (
            [12, 13, 14, 15]
            if USE_ROLE_OVERLAP and not USE_ALLOCATED_ROLE_BARRIERS
            else []
        ),
        "early_pre_sync": USE_EARLY_PRE_SYNC,
        "research_probe_phase": PROBE_PHASE,
        "physical_cta_threads": PHYSICAL_THREADS,
        "cooperative_launch": True,
        "full_grid_residency_required": True,
        "required_total_resident_ctas": GRID_CTAS,
        "occupancy_query_required_before_first_launch": True,
        "workspace_shape": list(partials.shape),
        "workspace_bytes": partials.numel() * partials.element_size(),
        "experimental": True,
        "dispatch_enabled": False,
    }
    return LargeGraphRole4Plan(
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
        counter_tensors={
            (
                "tile_local_monotonic_barriers"
                if USE_TILE_LOCAL_BARRIER
                else "monotonic_grid_barrier"
            ): grid_counter
        },
        metadata=metadata,
    )


__all__ = [
    "GRID_CTAS",
    "LOGICAL_TOKENS",
    "NUM_SPLITS",
    "prepare_tle_sm90_mega_mhc",
]

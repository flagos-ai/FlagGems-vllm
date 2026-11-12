"""Production SM90 eager MegaMHC base for ``T=3076, 3805, 4100``.

All admitted cases use the DeepSeek-V4-Flash Normal-BF16 profile
``H=4096, R=4, P=24``, sixteen physical warps (12 producer + 4 RS-WGMMA),
the complete phase-3 x8 tail, a 128-element tail chunk, cp.async Fn staging,
and ``maxnreg=64``.  The exact shape facades select one of two real launch
topologies:

* T=3076: cluster16 with H256 splits and a release/acquire cluster barrier;
* T=3805: cluster4 with H1024 splits and a cluster barrier; and
* T=4100: four H1024 splits with a resident global tile barrier.

Projection, routing, and the BF16-boundary RMSNorm tail execute in one kernel.
Only the topology selector remains import-time configurable because it changes
the launch contract; historical tail, TMA, and register experiments are fixed
to their admitted values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

USE_GLOBAL_TILE_BARRIER = (
    os.environ.get(
        "DG_TLE_MHC_EAGER_T3076_P256_CLUSTER16_MANUAL512_GLOBAL_TILE_BARRIER",
        "0",
    )
    == "1"
)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.experimental.tle.language as tle  # noqa: E402
import triton.language as tl  # noqa: E402

from .sm90_eager_p256_manual512_projection_split_ptx import (  # noqa: E402
    _manual512_p256_projection_split_ptx,
)
from .sm90_eager_p256_manual512_token_tail_exact1024_ptx import (  # noqa: E402
    _token_shard_combine_exact1024_bf16x8_ptx as _token_shard_combine_bf16x8_ptx,
)
from .sm90_eager_p256_manual512_token_tail_exact1024_ptx import (  # noqa: E402
    _token_shard_normalize_exact1024_bf16x8_ptx as _token_shard_normalize_bf16x8_ptx,
)
from .sm90_eager_p256_routing import _routing_finalize_no_barrier  # noqa: E402
from .sm90_small_family import _physical_thread_id_x  # noqa: E402

TOKENS = 3076
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
BLOCK_M = 64
BLOCK_N = 32
NUM_M_TILES = (TOKENS + BLOCK_M - 1) // BLOCK_M
CLUSTER_CTAS = 16
NUM_SPLITS = CLUSTER_CTAS
SPLIT_HIDDEN = HIDDEN // NUM_SPLITS
MAX_SPLITS = NUM_SPLITS
GRID_CTAS = NUM_M_TILES * NUM_SPLITS
NUM_WARPS = 16
TAIL_PHASE = 3
HIDDEN_PHASE = 3
TAIL_BATCHES = 32
TAIL_MODE = "x8"
TAIL_CHUNK = 128
MAXNREG = 64

JIT_TOKENS = tl.constexpr(TOKENS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_NUM_M_TILES = tl.constexpr(NUM_M_TILES)
JIT_NUM_SPLITS = tl.constexpr(NUM_SPLITS)
JIT_SPLIT_HIDDEN = tl.constexpr(SPLIT_HIDDEN)
JIT_MAX_SPLITS = tl.constexpr(MAX_SPLITS)
JIT_TAIL_PHASE = tl.constexpr(TAIL_PHASE)
JIT_HIDDEN_PHASE = tl.constexpr(HIDDEN_PHASE)
JIT_TAIL_BATCHES = tl.constexpr(TAIL_BATCHES)
JIT_X8_TAIL = tl.constexpr(True)
JIT_TAIL_CHUNK = tl.constexpr(TAIL_CHUNK)
JIT_USE_GLOBAL_TILE_BARRIER = tl.constexpr(USE_GLOBAL_TILE_BARRIER)


@triton.jit
def _manual512_cta_sync():
    """Full-CTA rendezvous for the physical sixteen-warp block."""
    _ = tl.inline_asm_elementwise(
        asm="{ bar.sync 0, 512; mov.u32 $0, 0; }",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _shared_address(pointer):
    """Expose a TLE shared pointer as a 32-bit PTX shared address."""
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cluster_split_barrier():
    """Publish split partials across the sixteen-CTA Hopper cluster."""
    physical_thread = _physical_thread_id_x()
    _ = tl.inline_asm_elementwise(
        asm=(
            "{ barrier.cluster.arrive.release.aligned; "
            "barrier.cluster.wait.acquire.aligned; mov.u32 $0, $1; }"
        ),
        constraints="=r,r",
        args=[physical_thread],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _tile_split_barrier(tile_state_ptr, participant_last):
    """Reusable resident-grid release/acquire barrier for one M64 tile."""
    physical_thread = _physical_thread_id_x()
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, last, changed;
          .reg .u32 old_phase, phase, prior;
          .reg .u64 phase_address;
          setp.eq.u32 leader, $2, 0;
          @!leader bra.uni DG_TLE_MHC_CLUSTER4_TILE_CTA_SYNC;
          add.u64 phase_address, $1, 4;
          ld.acquire.gpu.global.u32 old_phase, [phase_address];
          atom.acq_rel.gpu.global.inc.u32 prior, [$1], $3;
          setp.eq.u32 last, prior, $3;
          @last bra.uni DG_TLE_MHC_CLUSTER4_TILE_LAST;
        DG_TLE_MHC_CLUSTER4_TILE_WAIT:
          ld.acquire.gpu.global.u32 phase, [phase_address];
          setp.ne.u32 changed, phase, old_phase;
          @!changed bra.uni DG_TLE_MHC_CLUSTER4_TILE_WAIT;
          bra.uni DG_TLE_MHC_CLUSTER4_TILE_CTA_SYNC;
        DG_TLE_MHC_CLUSTER4_TILE_LAST:
          atom.release.gpu.global.add.u32 phase, [phase_address], 1;
        DG_TLE_MHC_CLUSTER4_TILE_CTA_SYNC:
          bar.sync 0, 512;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,r,r",
        args=[
            tl.cast(tile_state_ptr, tl.uint64),
            physical_thread,
            participant_last,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _raw_shared_store_bf16(pointer, value):
    """Store through an nv_mma shared pointer without a layout conversion."""
    address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    _ = tl.inline_asm_elementwise(
        asm="{ st.shared.b16 [$2], $1; mov.u32 $0, 0; }",
        constraints="=r,h,r",
        args=[value, address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _raw_shared_load_bf16(pointer):
    """Load BF16 bits from an nv_mma shared pointer."""
    address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        asm="ld.shared.b16 $0, [$1];",
        constraints="=h,r",
        args=[address],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _run_extra_four_row_hidden_tail(
    owned_pre_base,
    residual_base,
    y_base,
    square_base,
    q_scratch,
    fn_scratch,
    rmsnorm_weight_ptr,
    physical_thread,
    owned_rows,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Run one additional four-row H4096 tail batch for cluster4."""
    _token_shard_combine_bf16x8_ptx(
        owned_pre_base,
        residual_base,
        q_scratch,
        square_base,
        physical_thread,
        0,
        64,
        owned_rows,
        4096,
        256,
        0,
        8,
        1,
        128,
        64,
    )
    _token_shard_combine_bf16x8_ptx(
        owned_pre_base,
        residual_base,
        q_scratch,
        square_base,
        physical_thread,
        256,
        64,
        owned_rows,
        4096,
        2304,
        4,
        8,
        1,
        128,
        64,
    )
    _token_shard_combine_bf16x8_ptx(
        owned_pre_base,
        residual_base,
        fn_scratch,
        square_base,
        physical_thread,
        512,
        64,
        owned_rows,
        4096,
        256,
        8,
        8,
        1,
        128,
        64,
    )
    _token_shard_combine_bf16x8_ptx(
        owned_pre_base,
        residual_base,
        fn_scratch,
        square_base,
        physical_thread,
        768,
        64,
        owned_rows,
        4096,
        2304,
        12,
        8,
        1,
        128,
        64,
    )
    _manual512_cta_sync()
    tail_rows = tl.arange(0, 4)
    square_columns = tl.arange(0, 16)
    square_values = tl.load(
        square_base + tail_rows[:, None] * 16 + square_columns[None, :],
        mask=tail_rows[:, None] < owned_rows,
        other=0.0,
    )
    total_square = tl.sum(square_values, axis=1)
    row_norm = tl.rsqrt(total_square / 4096.0 + rmsnorm_eps) * rmsnorm_scale
    tl.store(
        square_base + tail_rows,
        row_norm,
        mask=tail_rows < owned_rows,
    )
    _manual512_cta_sync()
    _token_shard_normalize_bf16x8_ptx(
        q_scratch,
        y_base,
        rmsnorm_weight_ptr,
        square_base,
        physical_thread,
        0,
        64,
        owned_rows,
        4096,
        256,
        8,
        1,
    )
    _token_shard_normalize_bf16x8_ptx(
        q_scratch,
        y_base,
        rmsnorm_weight_ptr,
        square_base,
        physical_thread,
        256,
        64,
        owned_rows,
        4096,
        2304,
        8,
        1,
    )
    _token_shard_normalize_bf16x8_ptx(
        fn_scratch,
        y_base,
        rmsnorm_weight_ptr,
        square_base,
        physical_thread,
        512,
        64,
        owned_rows,
        4096,
        256,
        8,
        1,
    )
    _token_shard_normalize_bf16x8_ptx(
        fn_scratch,
        y_base,
        rmsnorm_weight_ptr,
        square_base,
        physical_thread,
        768,
        64,
        owned_rows,
        4096,
        2304,
        8,
        1,
    )
    _manual512_cta_sync()


@triton.jit
def _sm90_eager_t3076_p256_cluster16_manual512_cpasync_kernel(
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
    debug_partials_ptr,
    partials_ptr,
    tile_states_ptr,
    tail_squares_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
):
    # FlagTree expands each logical tile into sixteen physical programs.
    pid = tl.program_id(0)
    tile = pid // JIT_NUM_SPLITS
    split = pid % JIT_NUM_SPLITS
    num_splits = JIT_NUM_SPLITS
    split_hidden = JIT_SPLIT_HIDDEN
    active_k64_blocks = split_hidden // 64
    rows_per_split = JIT_BLOCK_M // JIT_NUM_SPLITS

    # Four route slots.  A slot is K64 BF16 and Fn is K64 FP32.  The imported
    # PTX internally consumes each K64 slot as two persistent K32 halves.
    q_stages = tle.gpu.alloc(
        [4, 2, JIT_BLOCK_M, 32],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    fn_stages = tle.gpu.alloc(
        [4, 2, JIT_BLOCK_N, 32],
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
    projection_shared = tle.gpu.alloc(
        [JIT_BLOCK_M, JIT_BLOCK_N],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    # The PTX body starts with a 512-thread rendezvous that publishes these
    # coefficient caches before role split.  The last tile has four live rows.
    local_m = tl.arange(0, JIT_BLOCK_M)
    m = tile * JIT_BLOCK_M + local_m
    token_mask = m < JIT_TOKENS
    live_rows = tl.minimum(JIT_BLOCK_M, JIT_TOKENS - tile * JIT_BLOCK_M)
    post_column = tl.arange(0, JIT_HC_MULT)
    comb_column = tl.arange(0, JIT_HC_MULT * JIT_HC_MULT)
    tl.store(
        tle.gpu.local_ptr(post_shared),
        tl.load(
            post_mix_ptr + m[:, None] * JIT_HC_MULT + post_column[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ),
    )
    tl.store(
        tle.gpu.local_ptr(comb_shared),
        tl.load(
            comb_res_mix_ptr
            + m[:, None] * JIT_HC_MULT * JIT_HC_MULT
            + comb_column[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ),
    )

    hidden_base = split * split_hidden
    compact_partial_base = (
        partials_ptr + (tile * JIT_MAX_SPLITS + split) * JIT_BLOCK_M * JIT_BLOCK_N
    )
    _ = _manual512_p256_projection_split_ptx(
        x_ptr + hidden_base,
        residual_ptr + hidden_base,
        fn_ptr + hidden_base,
        new_residual_ptr + hidden_base,
        compact_partial_base,
        _shared_address(tle.gpu.local_ptr(post_shared, (0, 0))),
        _shared_address(tle.gpu.local_ptr(comb_shared, (0, 0))),
        _shared_address(tle.gpu.local_ptr(q_stages.slot(0).slot(0), (0, 0))),
        _shared_address(tle.gpu.local_ptr(q_stages.slot(1).slot(0), (0, 0))),
        _shared_address(tle.gpu.local_ptr(q_stages.slot(2).slot(0), (0, 0))),
        _shared_address(tle.gpu.local_ptr(q_stages.slot(3).slot(0), (0, 0))),
        _shared_address(tle.gpu.local_ptr(fn_stages.slot(0).slot(0), (0, 0))),
        _shared_address(tle.gpu.local_ptr(fn_stages.slot(1).slot(0), (0, 0))),
        _shared_address(tle.gpu.local_ptr(fn_stages.slot(2).slot(0), (0, 0))),
        _shared_address(tle.gpu.local_ptr(fn_stages.slot(3).slot(0), (0, 0))),
        _shared_address(tle.gpu.local_ptr(projection_shared, (0, 0))),
        tile,
        active_k64_blocks,
        JIT_TOKENS,
    )

    # The isolated projection PTX has already converged all sixteen warps
    # and written columns 0..24 directly into the compact split plane.
    # The hardware cluster barrier publishes all sixteen split partials
    # without requiring full-grid residency.
    if JIT_USE_GLOBAL_TILE_BARRIER:
        _tile_split_barrier(tile_states_ptr + tile * 2, num_splits - 1)
    else:
        _cluster_split_barrier()
    if JIT_X8_TAIL:
        physical_thread = _physical_thread_id_x()
        physical_warp = physical_thread // 32
        lane = physical_thread % 32
        pre_base = (
            partials_ptr + (tile * JIT_MAX_SPLITS * JIT_BLOCK_M) * JIT_BLOCK_N + 28
        )
        if JIT_TAIL_PHASE >= 2:
            # Eight warp owners per split route disjoint rows.  The eight
            # split CTAs therefore cover all 64 tokens concurrently.
            if physical_warp < rows_per_split:
                route_row = split * rows_per_split + physical_warp
                packed_route = 0.0
                for source_split in tl.static_range(0, JIT_MAX_SPLITS):
                    source_offset = (
                        (tile * JIT_MAX_SPLITS + source_split) * JIT_BLOCK_M + route_row
                    ) * JIT_BLOCK_N + lane
                    packed_route += tl.load(
                        partials_ptr + source_offset,
                        mask=(lane <= 24) & (source_split < num_splits),
                        other=0.0,
                    )
                route_token = tile * JIT_BLOCK_M + route_row
                if route_token < JIT_TOKENS:
                    _routing_finalize_no_barrier(
                        pre_base + route_row * JIT_BLOCK_N,
                        mix_scales_ptr,
                        mix_bases_ptr,
                        new_post_mix_ptr + route_token * JIT_HC_MULT,
                        new_comb_res_mix_ptr + route_token * JIT_HC_MULT * JIT_HC_MULT,
                        packed_route,
                        lane,
                        hc_norm_eps,
                        hc_pre_eps,
                        hc_post_scale,
                        sinkhorn_eps,
                    )
            _manual512_cta_sync()

        if JIT_TAIL_PHASE >= 3:
            # Each routing owner keeps at most four rows and processes all
            # H4096 columns.  The retired 64KiB projection ring stores the
            # complete BF16 boundary, so no second cross-CTA barrier or
            # sixteen-way square reduction is needed.
            owned_row_base = split * rows_per_split
            owned_rows = tl.maximum(
                0, tl.minimum(rows_per_split, live_rows - owned_row_base)
            )
            owned_token_base = tile * JIT_BLOCK_M + owned_row_base
            owned_pre_base = pre_base + owned_row_base * JIT_BLOCK_N
            residual_base = (
                new_residual_ptr + owned_token_base * JIT_HC_MULT * JIT_HIDDEN
            )
            y_base = y_bf16_ptr + owned_token_base * JIT_HIDDEN
            square_base = tail_squares_ptr + pid * 4 * 16
            _token_shard_combine_bf16x8_ptx(
                owned_pre_base,
                residual_base,
                q_stages.slot(0),
                square_base,
                physical_thread,
                0,
                64,
                owned_rows,
                4096,
                256,
                0,
                8,
                1,
                128,
                64,
            )
            _token_shard_combine_bf16x8_ptx(
                owned_pre_base,
                residual_base,
                q_stages.slot(0),
                square_base,
                physical_thread,
                256,
                64,
                owned_rows,
                4096,
                2304,
                4,
                8,
                1,
                128,
                64,
            )
            _token_shard_combine_bf16x8_ptx(
                owned_pre_base,
                residual_base,
                fn_stages.slot(0),
                square_base,
                physical_thread,
                512,
                64,
                owned_rows,
                4096,
                256,
                8,
                8,
                1,
                128,
                64,
            )
            _token_shard_combine_bf16x8_ptx(
                owned_pre_base,
                residual_base,
                fn_stages.slot(0),
                square_base,
                physical_thread,
                768,
                64,
                owned_rows,
                4096,
                2304,
                12,
                8,
                1,
                128,
                64,
            )
            _manual512_cta_sync()
            if JIT_HIDDEN_PHASE >= 2:
                tail_rows = tl.arange(0, 4)
                square_columns = tl.arange(0, 16)
                square_values = tl.load(
                    square_base + tail_rows[:, None] * 16 + square_columns[None, :],
                    mask=tail_rows[:, None] < owned_rows,
                    other=0.0,
                )
                total_square = tl.sum(square_values, axis=1)
                row_norm = (
                    tl.rsqrt(total_square / JIT_HIDDEN + rmsnorm_eps) * rmsnorm_scale
                )
                tl.store(
                    square_base + tail_rows,
                    row_norm,
                    mask=tail_rows < owned_rows,
                )
                _manual512_cta_sync()
            if JIT_HIDDEN_PHASE >= 3:
                _token_shard_normalize_bf16x8_ptx(
                    q_stages.slot(0),
                    y_base,
                    rmsnorm_weight_ptr,
                    square_base,
                    physical_thread,
                    0,
                    64,
                    owned_rows,
                    4096,
                    256,
                    8,
                    1,
                )
                _token_shard_normalize_bf16x8_ptx(
                    q_stages.slot(0),
                    y_base,
                    rmsnorm_weight_ptr,
                    square_base,
                    physical_thread,
                    256,
                    64,
                    owned_rows,
                    4096,
                    2304,
                    8,
                    1,
                )
                _token_shard_normalize_bf16x8_ptx(
                    fn_stages.slot(0),
                    y_base,
                    rmsnorm_weight_ptr,
                    square_base,
                    physical_thread,
                    512,
                    64,
                    owned_rows,
                    4096,
                    256,
                    8,
                    1,
                )
                _token_shard_normalize_bf16x8_ptx(
                    fn_stages.slot(0),
                    y_base,
                    rmsnorm_weight_ptr,
                    square_base,
                    physical_thread,
                    768,
                    64,
                    owned_rows,
                    4096,
                    2304,
                    8,
                    1,
                )
                if JIT_NUM_SPLITS == 4:
                    _manual512_cta_sync()
                    for extra_batch in tl.static_range(1, 4):
                        extra_row_base = owned_row_base + extra_batch * 4
                        extra_rows = tl.maximum(
                            0,
                            tl.minimum(4, live_rows - extra_row_base),
                        )
                        _run_extra_four_row_hidden_tail(
                            owned_pre_base + extra_batch * 4 * JIT_BLOCK_N,
                            residual_base + extra_batch * 4 * JIT_HC_MULT * JIT_HIDDEN,
                            y_base + extra_batch * 4 * JIT_HIDDEN,
                            square_base,
                            q_stages.slot(0),
                            fn_stages.slot(0),
                            rmsnorm_weight_ptr,
                            physical_thread,
                            extra_rows,
                            rmsnorm_eps,
                            rmsnorm_scale,
                        )


def _require_tensor(value, *, name, shape, dtype):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name}.shape={tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name}.dtype={value.dtype}, expected {dtype}")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


@dataclass(frozen=True)
class EagerT3076P256Cluster16Manual512ProjectionPlan:
    tensors: tuple[torch.Tensor, ...]
    debug_partials: torch.Tensor
    partials: torch.Tensor
    tile_states: torch.Tensor
    tail_squares: torch.Tensor
    scalars: tuple[float, ...]
    metadata: dict[str, object]
    maxnreg: int

    @property
    def grid_ctas(self) -> int:
        return GRID_CTAS

    def _kernel_args(self) -> tuple[object, ...]:
        return (
            *self.tensors,
            self.debug_partials,
            self.partials,
            self.tile_states,
            self.tail_squares,
            *self.scalars,
        )

    def compile(self):
        if USE_GLOBAL_TILE_BARRIER:
            return _sm90_eager_t3076_p256_cluster16_manual512_cpasync_kernel.warmup(
                *self._kernel_args(),
                grid=(GRID_CTAS,),
                num_warps=NUM_WARPS,
                num_stages=1,
                num_ctas=1,
                maxnreg=self.maxnreg,
            )
        return _sm90_eager_t3076_p256_cluster16_manual512_cpasync_kernel.warmup(
            *self._kernel_args(),
            grid=(NUM_M_TILES,),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            cluster_dims=(CLUSTER_CTAS, 1, 1),
            maxnreg=self.maxnreg,
        )

    def run(self) -> None:
        if USE_GLOBAL_TILE_BARRIER:
            _sm90_eager_t3076_p256_cluster16_manual512_cpasync_kernel[(GRID_CTAS,)](
                *self._kernel_args(),
                num_warps=NUM_WARPS,
                num_stages=1,
                num_ctas=1,
                maxnreg=self.maxnreg,
            )
            return
        _sm90_eager_t3076_p256_cluster16_manual512_cpasync_kernel[(NUM_M_TILES,)](
            *self._kernel_args(),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            cluster_dims=(CLUSTER_CTAS, 1, 1),
            maxnreg=self.maxnreg,
        )


def prepare_tle_sm90_projection(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT3076P256Cluster16Manual512ProjectionPlan:
    """Prepare one admitted exact-shape P256 eager implementation."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the eager projection plan outside capture")

    specs = (
        (inputs.get("x"), "inputs.x", (TOKENS, HIDDEN), torch.bfloat16),
        (
            inputs.get("residual"),
            "inputs.residual",
            (TOKENS, HC_MULT, HIDDEN),
            torch.bfloat16,
        ),
        (
            inputs.get("post_mix"),
            "inputs.post_mix",
            (TOKENS, HC_MULT, 1),
            torch.float32,
        ),
        (
            inputs.get("comb_res_mix"),
            "inputs.comb_res_mix",
            (TOKENS, HC_MULT, HC_MULT),
            torch.float32,
        ),
        (
            inputs.get("fn"),
            "inputs.fn",
            (PROJECTION_OUTPUTS, HC_MULT * HIDDEN),
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
            (TOKENS, HC_MULT, HIDDEN),
            torch.bfloat16,
        ),
        (
            outputs.get("new_post_mix"),
            "outputs.new_post_mix",
            (TOKENS, HC_MULT, 1),
            torch.float32,
        ),
        (
            outputs.get("new_comb_res_mix"),
            "outputs.new_comb_res_mix",
            (TOKENS, HC_MULT, HC_MULT),
            torch.float32,
        ),
        (
            outputs.get("y_bf16"),
            "outputs.y_bf16",
            (TOKENS, HIDDEN),
            torch.bfloat16,
        ),
    )
    tensors = tuple(
        _require_tensor(value, name=name, shape=shape, dtype=dtype)
        for value, name, shape, dtype in specs
    )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must be on one CUDA device")

    maxnreg = MAXNREG
    device = tensors[0].device
    debug_partials = torch.empty((1,), dtype=torch.float32, device=device)
    partials = torch.empty(
        (NUM_M_TILES, MAX_SPLITS, BLOCK_M, BLOCK_N),
        dtype=torch.float32,
        device=device,
    )
    tile_states = torch.zeros((NUM_M_TILES, 2), dtype=torch.uint32, device=device)
    tail_squares = torch.empty((GRID_CTAS, 4, 16), dtype=torch.float32, device=device)
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
        "kernel_symbol": ("_sm90_eager_t3076_p256_cluster16_manual512_cpasync_kernel"),
        "grid_ctas": GRID_CTAS,
        "num_m_tiles": NUM_M_TILES,
        "logical_grid_tiles": NUM_M_TILES,
        "cluster_ctas": CLUSTER_CTAS,
        "nonportable_cluster_size": not USE_GLOBAL_TILE_BARRIER,
        "uses_cuda_cluster": not USE_GLOBAL_TILE_BARRIER,
        "uses_global_tile_barrier": USE_GLOBAL_TILE_BARRIER,
        "requires_full_grid_residency": USE_GLOBAL_TILE_BARRIER,
        "num_splits": NUM_SPLITS,
        "split_hidden": SPLIT_HIDDEN,
        "active_k64_blocks": SPLIT_HIDDEN // 64,
        "cta_mapping": "tile=pid//16, split=pid%16",
        "physical_cta_threads": 512,
        "producer_warps": 12,
        "math_warps": 4,
        "target_registers_per_thread": maxnreg,
        "a_storage": "K64 BF16 128B-swizzled shared",
        "fn_storage": "four-route cp.async FP32 ring",
        "wgmma": "persistent TF32 RS M64N24, internal K32 halves",
        "projection_only": False,
        "tail_enabled": True,
        "tail_schedule": "token-sharded x8 PTX",
        "tail_mode": TAIL_MODE,
        "tail_chunk": TAIL_CHUNK,
        "tail_phase": TAIL_PHASE,
        "hidden_phase": HIDDEN_PHASE,
        "row_barrier": False,
        "tail_batches": TAIL_BATCHES,
        "experimental": False,
        "dispatch_enabled": True,
    }
    return EagerT3076P256Cluster16Manual512ProjectionPlan(
        tensors=tensors,
        debug_partials=debug_partials,
        partials=partials,
        tile_states=tile_states,
        tail_squares=tail_squares,
        scalars=(
            float(inputs["hc_norm_eps"]),
            float(inputs["hc_pre_eps"]),
            float(inputs["hc_post_scale"]),
            float(inputs["sinkhorn_eps"]),
            float(inputs["rmsnorm_eps"]),
            float(inputs["rmsnorm_scale"]),
        ),
        metadata=metadata,
        maxnreg=maxnreg,
    )


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT3076P256Cluster16Manual512ProjectionPlan:
    """Shape-exact production seam for the formally admitted full tail."""
    return prepare_tle_sm90_projection(inputs, outputs)


__all__ = [
    "EagerT3076P256Cluster16Manual512ProjectionPlan",
    "CLUSTER_CTAS",
    "GRID_CTAS",
    "MAX_SPLITS",
    "NUM_M_TILES",
    "NUM_SPLITS",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
    "prepare_tle_sm90_projection",
]

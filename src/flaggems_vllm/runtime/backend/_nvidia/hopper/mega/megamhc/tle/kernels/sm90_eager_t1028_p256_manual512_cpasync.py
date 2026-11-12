"""T=1028 p256 projection probe using the proven manual512 cp.async pipe.

This module is deliberately independent from the production dispatch and the
older p256 manual-ring experiments.  One ordinary 16-warp CTA owns one
``M64 x H512`` split: physical warps 0..11 produce BF16 A and four FP32 Fn
routes while physical warps 12..15 execute the persistent K32-half RS-WGMMA
loop.  The grid contains ``32 token tiles x 8 hidden splits`` CTAs and enqueues
exactly one CUDA kernel.

The role-aware projection and tail PTX bodies are local to this module family.
The admitted implementation intentionally exposes only the validated x8 tail;
it has no runtime dependency on the still-experimental p384 source.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Mapping

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from .sm90_eager_p256_manual512_projection_n32_ptx import (
    _manual512_p256_projection_n32_ptx,
)
from .sm90_eager_p256_routing import (
    _routing_finalize_no_barrier,
)
from .sm90_eager_t1028_p256_manual512_token_tail_ptx import (
    _token_shard_combine_bf16x8_ptx,
    _token_shard_normalize_bf16x8_ptx,
)
from .sm90_small_family import _physical_thread_id_x

TOKENS = 1028
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
BLOCK_M = 64
BLOCK_N = 32
NUM_M_TILES = (TOKENS + BLOCK_M - 1) // BLOCK_M
FULL_M_TILES = NUM_M_TILES - 1
FULL_TILE_SPLITS = 16
TAIL_TILE_SPLITS = 8
MAX_SPLITS = FULL_TILE_SPLITS
GRID_CTAS = FULL_M_TILES * FULL_TILE_SPLITS + TAIL_TILE_SPLITS
NUM_WARPS = 16
TAIL_PHASE = int(
    os.environ.get("DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_TAIL_PHASE", "3")
)
if TAIL_PHASE not in (1, 2, 3):
    raise ValueError("manual512 tail phase must be 1/2/3")
TAIL_BATCHES = 32
TAIL_MODE = os.environ.get("DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_TAIL_MODE", "x8")
if TAIL_MODE != "x8":
    raise ValueError("the admitted manual512 implementation requires x8 tail")
TAIL_CHUNK = int(
    os.environ.get("DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_TAIL_CHUNK", "128")
)
if TAIL_CHUNK not in (32, 64, 128, 256):
    raise ValueError("manual512 tail chunk must be 32/64/128/256")
USE_FN_TMA = os.environ.get("DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_FN_TMA", "0") == "1"

JIT_TOKENS = tl.constexpr(TOKENS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_FULL_M_TILES = tl.constexpr(FULL_M_TILES)
JIT_FULL_TILE_SPLITS = tl.constexpr(FULL_TILE_SPLITS)
JIT_TAIL_TILE_SPLITS = tl.constexpr(TAIL_TILE_SPLITS)
JIT_MAX_SPLITS = tl.constexpr(MAX_SPLITS)
JIT_TAIL_PHASE = tl.constexpr(TAIL_PHASE)
JIT_TAIL_BATCHES = tl.constexpr(TAIL_BATCHES)
JIT_X8_TAIL = tl.constexpr(True)
JIT_TAIL_CHUNK = tl.constexpr(TAIL_CHUNK)
JIT_USE_FN_TMA = tl.constexpr(USE_FN_TMA)


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
def _tile_split_barrier(tile_state_ptr, participant_last):
    """Reusable eight-CTA release/acquire barrier followed by CTA sync.

    The full T=1028 grid has 136 blocks and the compiled kernel admits two
    blocks on each of 132 SMs (capacity 264), so all participants are
    resident.  Only thread zero updates/spins on the tile-local epoch; the
    remaining 511 threads wait at the CTA barrier.  The caller must enter
    after a full-CTA publish rendezvous: projection PTX ends with bar.sync,
    while routing/combine call ``_manual512_cta_sync`` explicitly.  Keeping
    that precondition at the producer boundary avoids a duplicate CTA barrier
    inside this helper without weakening the release/acquire chain.
    """
    physical_thread = _physical_thread_id_x()
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, last, changed;
          .reg .u32 old_phase, phase, prior;
          .reg .u64 phase_address;
          setp.eq.u32 leader, $2, 0;
          @!leader bra.uni DG_TLE_MHC_T1028_TILE_CTA_SYNC;
          add.u64 phase_address, $1, 4;
          ld.acquire.gpu.global.u32 old_phase, [phase_address];
          // acq_rel RMWs form the transitive chain across all eight CTA
          // arrivals; the last leader then publishes that chain through the
          // phase release.  Waiter leaders acquire the phase, and the final
          // CTA barrier transfers their acquired visibility to every thread.
          atom.acq_rel.gpu.global.inc.u32 prior, [$1], $3;
          setp.eq.u32 last, prior, $3;
          @last bra.uni DG_TLE_MHC_T1028_TILE_LAST;
        DG_TLE_MHC_T1028_TILE_WAIT:
          ld.acquire.gpu.global.u32 phase, [phase_address];
          setp.ne.u32 changed, phase, old_phase;
          @!changed bra.uni DG_TLE_MHC_T1028_TILE_WAIT;
          bra.uni DG_TLE_MHC_T1028_TILE_CTA_SYNC;
        DG_TLE_MHC_T1028_TILE_LAST:
          atom.release.gpu.global.add.u32 phase, [phase_address], 1;
        DG_TLE_MHC_T1028_TILE_CTA_SYNC:
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
def _sm90_eager_t1028_p256_manual512_cpasync_kernel(
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
    fn_tensormap_ptr,
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
    ENABLE_TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    full_grid = JIT_FULL_M_TILES * JIT_FULL_TILE_SPLITS
    full_tile = pid < full_grid
    tile = tl.where(full_tile, pid // JIT_FULL_TILE_SPLITS, JIT_FULL_M_TILES)
    split = tl.where(full_tile, pid % JIT_FULL_TILE_SPLITS, pid - full_grid)
    num_splits = tl.where(full_tile, JIT_FULL_TILE_SPLITS, JIT_TAIL_TILE_SPLITS)
    split_hidden = tl.where(full_tile, 256, 512)
    active_k64_blocks = split_hidden // 64
    rows_per_split = tl.where(full_tile, 4, 8)

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
    fn_tma_barriers = tle.gpu.alloc(
        [8],
        dtype=tl.uint64,
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
    _ = _manual512_p256_projection_n32_ptx(
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
        tl.cast(fn_tensormap_ptr, tl.uint64),
        _shared_address(tle.gpu.local_ptr(fn_tma_barriers, (0,))),
        JIT_USE_FN_TMA,
        JIT_TOKENS,
    )

    if ENABLE_TAIL:
        # The isolated projection PTX has already converged all sixteen warps
        # and written columns 0..24 directly into the compact split plane.
        rows = tl.arange(0, JIT_BLOCK_M)
        columns = tl.arange(0, JIT_BLOCK_N)
        # All 136 CTAs fit concurrently (2 blocks/SM x 132 SMs = 264).
        # The acquire/release barrier makes all eight split partials visible
        # before split zero performs this tile's uniform manual512 tail.
        _tile_split_barrier(tile_states_ptr + tile * 2, num_splits - 1)
        if JIT_X8_TAIL:
            physical_thread = _physical_thread_id_x()
            physical_warp = physical_thread // 32
            lane = physical_thread % 32
            pre_base = (
                partials_ptr + (tile * JIT_MAX_SPLITS * JIT_BLOCK_M) * JIT_BLOCK_N + 25
            )
            if JIT_TAIL_PHASE >= 2:
                # Eight warp owners per split route disjoint rows.  The eight
                # split CTAs therefore cover all 64 tokens concurrently.
                if physical_warp < rows_per_split:
                    route_row = split * rows_per_split + physical_warp
                    packed_route = 0.0
                    for source_split in tl.static_range(0, JIT_MAX_SPLITS):
                        source_offset = (
                            (tile * JIT_MAX_SPLITS + source_split) * JIT_BLOCK_M
                            + route_row
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
                            new_comb_res_mix_ptr
                            + route_token * JIT_HC_MULT * JIT_HC_MULT,
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
                # eight-way square reduction is needed.
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


def _encode_global_tensor_map(fn: torch.Tensor) -> torch.Tensor:
    """Encode the validated 128B-swizzled FP32 N32xK32 Fn map."""
    import triton.runtime.driver as driver_module

    tensor_map = driver_module.active.utils.fill_tma_descriptor(
        fn.data_ptr(),
        3,
        4,
        7,
        [32, 32],
        list(fn.shape),
        list(fn.stride()),
        0,
    )
    object_address = id(tensor_map)
    if object_address % 128 != 0:
        raise RuntimeError("PyCUtensorMap lost its 128-byte alignment")
    raw = ctypes.string_at(object_address + 128, 128)
    host = torch.tensor(list(raw), dtype=torch.uint8)
    device = host.to(device=fn.device)
    if device.data_ptr() % 128 != 0:
        raise RuntimeError("device tensor-map storage must be 128-byte aligned")
    return device


@dataclass(frozen=True)
class EagerT1028P256Manual512ProjectionPlan:
    tensors: tuple[torch.Tensor, ...]
    debug_partials: torch.Tensor
    partials: torch.Tensor
    tile_states: torch.Tensor
    tail_squares: torch.Tensor
    scalars: tuple[float, ...]
    metadata: dict[str, object]
    maxnreg: int
    tail_enabled: bool

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
            self.tail_enabled,
        )

    def compile(self):
        return _sm90_eager_t1028_p256_manual512_cpasync_kernel.warmup(
            *self._kernel_args(),
            grid=(GRID_CTAS,),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            maxnreg=self.maxnreg,
        )

    def run(self) -> None:
        _sm90_eager_t1028_p256_manual512_cpasync_kernel[(GRID_CTAS,)](
            *self._kernel_args(),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            maxnreg=self.maxnreg,
        )


def prepare_tle_sm90_projection(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT1028P256Manual512ProjectionPlan:
    """Prepare the opt-in, exact-one-kernel projection experiment."""
    if os.environ.get("DG_TLE_MHC_EAGER_T1028_P256_MANUAL512", "0") != "1":
        raise RuntimeError(
            "set DG_TLE_MHC_EAGER_T1028_P256_MANUAL512=1 to enable this probe"
        )
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

    fn_tensormap_storage = (
        _encode_global_tensor_map(tensors[4])
        if USE_FN_TMA
        else torch.empty((1,), dtype=torch.uint8, device=tensors[0].device)
    )
    kernel_tensors = (*tensors, fn_tensormap_storage)
    maxnreg = int(os.environ.get("DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_MAXNREG", "64"))
    if maxnreg not in (48, 56, 64, 72, 80):
        raise ValueError("manual512 maxnreg must be 48/56/64/72/80")
    device = tensors[0].device
    debug_partials = torch.empty(
        (MAX_SPLITS, TOKENS, 2048), dtype=torch.float32, device=device
    )
    partials = torch.empty(
        (NUM_M_TILES, MAX_SPLITS, BLOCK_M, BLOCK_N),
        dtype=torch.float32,
        device=device,
    )
    tile_states = torch.zeros((NUM_M_TILES, 2), dtype=torch.uint32, device=device)
    tail_squares = torch.empty((GRID_CTAS, 4, 16), dtype=torch.float32, device=device)
    tail_enabled = (
        os.environ.get("DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_TAIL", "0") == "1"
    )
    dispatch_enabled = (
        os.environ.get("DG_TLE_MHC_EAGER_T1028_P256_MANUAL512_ADMITTED", "0") == "1"
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
        "kernel_symbol": ("_sm90_eager_t1028_p256_manual512_cpasync_kernel"),
        "grid_ctas": GRID_CTAS,
        "num_m_tiles": NUM_M_TILES,
        "num_splits": "16 full tiles / 8 partial tile",
        "split_hidden": "H256 full tiles / H512 partial tile",
        "active_k64_blocks": "4 full tiles / 8 partial tile",
        "cta_mapping": (
            "pid<256: tile=pid//16 split=pid%16; " "pid>=256: tile=16 split=pid-256"
        ),
        "physical_cta_threads": 512,
        "producer_warps": 12,
        "math_warps": 4,
        "target_registers_per_thread": maxnreg,
        "a_storage": "K64 BF16 128B-swizzled shared",
        "fn_storage": (
            "four-route raw-global TMA FP32 ring"
            if USE_FN_TMA
            else "four-route cp.async FP32 ring"
        ),
        "wgmma": "persistent TF32 RS M64N32, internal K32 halves",
        "projection_only": not tail_enabled,
        "tail_enabled": tail_enabled,
        "tail_schedule": "token-sharded x8 PTX",
        "tail_mode": TAIL_MODE,
        "tail_chunk": TAIL_CHUNK,
        "tail_phase": TAIL_PHASE,
        "tail_batches": TAIL_BATCHES,
        "experimental": not dispatch_enabled,
        "dispatch_enabled": dispatch_enabled,
    }
    return EagerT1028P256Manual512ProjectionPlan(
        tensors=kernel_tensors,
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
        tail_enabled=tail_enabled,
    )


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
) -> EagerT1028P256Manual512ProjectionPlan:
    """Shape-exact production seam for the formally admitted full tail."""
    plan = prepare_tle_sm90_projection(inputs, outputs)
    if not plan.tail_enabled:
        raise RuntimeError("the admitted T=1028 dispatch requires full tail")
    return plan


__all__ = [
    "EagerT1028P256Manual512ProjectionPlan",
    "FULL_TILE_SPLITS",
    "GRID_CTAS",
    "MAX_SPLITS",
    "NUM_M_TILES",
    "TAIL_TILE_SPLITS",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
    "prepare_tle_sm90_projection",
]

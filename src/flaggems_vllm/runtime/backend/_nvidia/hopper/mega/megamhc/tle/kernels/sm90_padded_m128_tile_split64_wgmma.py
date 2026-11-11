"""Experimental SM90 padded-M128 tile-by-split MegaMHC Graph kernel.

This generalizes the fixed-T128 winner to the pinned DeepSeek-V4-Flash
Normal BF16 profile for logical ``T=72..128``.  It does not participate in
production dispatch unless its explicit experiment switch is set.  The
128-CTA cooperative grid is mapped as::

    tile  = program_id // 64
    split = program_id % 64

Consequently every CTA produces exactly one M64 tile for one H64 hidden
split.  The two CTAs with the same split write disjoint token rows in a
shared ``[64, 128, 32]`` FP32 workspace.  This is half the workspace of the
earlier 128-split multi-tile control and removes its serial two-tile loop.
The second tile masks input loads, public stores, and all tails beyond the
selected logical token count.

After the sole grid-wide release/acquire rendezvous CTA ``i`` owns token
``i``.  Its producer partition runs the hidden/RMSNorm tail while its WGMMA
worker partition reduces the 64 projection partials and runs routing.  A
prepared plan enqueues exactly one kernel and therefore captures as one CUDA
Graph kernel node.  Full-grid residency must be proved from the compiled
CUfunction before launch; otherwise the software rendezvous can deadlock.
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


PADDED_TOKENS = 128
LOGICAL_TOKEN_BUCKETS = (72, 80, 88, 96, 104, 112, 120, 128)
_TOKENS_TEXT = os.environ.get(
    "DG_TLE_MHC_PADDED_M128_TILE_SPLIT64_LOGICAL_TOKENS", "128"
)
try:
    LOGICAL_TOKENS = int(_TOKENS_TEXT)
except ValueError as exc:  # pragma: no cover - import-time contract
    raise ValueError(
        "DG_TLE_MHC_PADDED_M128_TILE_SPLIT64_LOGICAL_TOKENS must be an integer"
    ) from exc
if LOGICAL_TOKENS not in LOGICAL_TOKEN_BUCKETS:
    raise ValueError(
        "DG_TLE_MHC_PADDED_M128_TILE_SPLIT64_LOGICAL_TOKENS must be one of "
        f"{LOGICAL_TOKEN_BUCKETS}, got {LOGICAL_TOKENS}"
    )
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = HIDDEN * HC_MULT
SINKHORN_ITERS = 20
BLOCK_M = 64
BLOCK_N = 32
NUM_M_TILES = 2
NUM_SPLITS = 64
HIDDEN_PER_SPLIT = HIDDEN // NUM_SPLITS
GRID_CTAS = NUM_M_TILES * NUM_SPLITS
NUM_WARPS = 16
STAGED_MATH_WARPS = 4
STAGED_MATH_REGS = 128

JIT_LOGICAL_TOKENS = tl.constexpr(LOGICAL_TOKENS)
JIT_PADDED_TOKENS = tl.constexpr(PADDED_TOKENS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_NUM_M_TILES = tl.constexpr(NUM_M_TILES)
JIT_NUM_SPLITS = tl.constexpr(NUM_SPLITS)
JIT_HIDDEN_PER_SPLIT = tl.constexpr(HIDDEN_PER_SPLIT)
JIT_GRID_CTAS = tl.constexpr(GRID_CTAS)
JIT_CTA_THREADS = tl.constexpr(NUM_WARPS * 32)
JIT_STAGED_MATH_WARPS = tl.constexpr(STAGED_MATH_WARPS)
JIT_STAGED_MATH_THREADS = tl.constexpr(STAGED_MATH_WARPS * 32)
JIT_STAGED_MATH_REGS = tl.constexpr(STAGED_MATH_REGS)
JIT_STAGED_LOGICAL_THREADS = tl.constexpr(
    NUM_WARPS * 32 + STAGED_MATH_WARPS * 32
)


@triton.jit
def _tile_split64_producer(
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
    producer_ready,
    grid_acquire_ready,
    pre_ready,
    pre_shared,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    grid_counter_ptr,
    tile,
    split,
    pid,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Sixteen producer warps form one M64 x H64 tile."""
    m = tile * JIT_BLOCK_M + tl.arange(0, JIT_BLOCK_M)
    token_active = m < JIT_LOGICAL_TOKENS
    local_hidden = tl.arange(0, JIT_HIDDEN_PER_SPLIT)
    hidden = split * JIT_HIDDEN_PER_SPLIT + local_hidden
    n = tl.arange(0, JIT_BLOCK_N)
    token_hidden = m[:, None] * JIT_HIDDEN + hidden[None, :]
    token_residual = m[:, None] * JIT_FLAT_HIDDEN + hidden[None, :]

    x = tl.load(
        x_ptr + token_hidden, mask=token_active[:, None], other=0.0
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
    squared_sum = tl.zeros((JIT_BLOCK_M,), tl.float32)

    for channel in tl.static_range(0, JIT_HC_MULT):
        if channel > 0:
            tle.gpu.barrier_wait(stage_empty)
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
            new_residual_ptr
            + m[:, None] * JIT_FLAT_HIDDEN
            + flat_hidden[None, :],
            q_bf16,
            mask=token_active[:, None],
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

    tle.gpu.barrier_wait(stage_empty)
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
    _ = _t64._monotonic_grid128_arrive_and_wait(
        tl.cast(grid_counter_ptr, tl.uint64), physical_thread
    )
    tle.gpu.barrier_wait(grid_acquire_ready)

    if pid < JIT_LOGICAL_TOKENS:
        _t64._direct_hidden_tail(
            pre_ready,
            pre_shared,
            new_residual_ptr + pid * JIT_FLAT_HIDDEN,
            rmsnorm_weight_ptr,
            y_bf16_ptr + pid * JIT_HIDDEN,
            rmsnorm_eps,
            rmsnorm_scale,
        )


@triton.jit
def _tile_split64_math_and_routing(
    partials_ptr,
    q_shared,
    fn_shared,
    route_shared,
    stage_full,
    stage_empty,
    producer_ready,
    grid_acquire_ready,
    pre_ready,
    pre_shared,
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
    """Four WGMMA warps produce one tile, then reduce 64 partials."""
    accumulator = tl.zeros(
        (JIT_BLOCK_M, JIT_BLOCK_N), dtype=tl.float32
    )
    for _ in tl.static_range(0, JIT_HC_MULT):
        tle.gpu.barrier_wait(stage_full)
        accumulator = tle.gpu.wgmma(
            q_shared,
            fn_shared,
            accumulator,
            input_precision="tf32",
            trans_b=True,
        )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        tle.gpu.barrier_arrive(stage_empty)

    tile = pid // JIT_NUM_SPLITS
    m = tile * JIT_BLOCK_M + tl.arange(0, JIT_BLOCK_M)
    token_active = m < JIT_LOGICAL_TOKENS
    n = tl.arange(0, JIT_BLOCK_N)
    workspace_offsets = (
        split * JIT_PADDED_TOKENS * JIT_BLOCK_N
        + m[:, None] * JIT_BLOCK_N
        + n[None, :]
    )
    tl.store(
        partials_ptr + workspace_offsets,
        accumulator,
        mask=token_active[:, None]
        & (n[None, :] < JIT_PROJECTION_OUTPUTS),
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
    if pid < JIT_LOGICAL_TOKENS:
        packed_sum = 0.0
        for source_split in tl.static_range(0, JIT_NUM_SPLITS):
            route_offset = (
                source_split * JIT_PADDED_TOKENS * JIT_BLOCK_N
                + pid * JIT_BLOCK_N
                + lane
            )
            packed_sum += tl.load(
                partials_ptr + route_offset,
                mask=lane <= JIT_PROJECTION_OUTPUTS,
                other=0.0,
            )
        _t64._direct_routing_finalize(
            pre_ready,
            pre_shared,
            route_shared,
            mix_scales_ptr,
            mix_bases_ptr,
            new_post_mix_ptr + pid * JIT_HC_MULT,
            new_comb_res_mix_ptr + pid * JIT_HC_MULT * JIT_HC_MULT,
            packed_sum,
            lane,
            True,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sinkhorn_eps,
            True,
            False,
        )


@triton.jit
def _sm90_padded_m128_tile_split64_wgmma_kernel(
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
    tl.static_assert(JIT_PADDED_TOKENS == 128)
    tl.static_assert(JIT_LOGICAL_TOKENS > 64)
    tl.static_assert(JIT_LOGICAL_TOKENS <= JIT_PADDED_TOKENS)
    tl.static_assert(JIT_GRID_CTAS == 128)
    tl.static_assert(JIT_NUM_M_TILES == 2)
    tl.static_assert(JIT_NUM_SPLITS == 64)
    tl.static_assert(JIT_HIDDEN_PER_SPLIT == 64)

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
    route_shared = tle.gpu.alloc(
        [JIT_STAGED_MATH_WARPS, JIT_BLOCK_N],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pre_shared = tle.gpu.alloc(
        [JIT_HC_MULT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    stage_full = tle.gpu.alloc_barrier(arrive_count=JIT_STAGED_LOGICAL_THREADS)
    stage_empty = tle.gpu.alloc_barrier(arrive_count=JIT_STAGED_LOGICAL_THREADS)
    producer_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_STAGED_LOGICAL_THREADS
    )
    grid_acquire_ready = tle.gpu.alloc_barrier(
        arrive_count=JIT_STAGED_LOGICAL_THREADS
    )
    pre_ready = tle.gpu.alloc_barrier(arrive_count=JIT_STAGED_LOGICAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _tile_split64_producer,
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
                    producer_ready,
                    grid_acquire_ready,
                    pre_ready,
                    pre_shared,
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
                _tile_split64_math_and_routing,
                (
                    partials_ptr,
                    q_shared,
                    fn_shared,
                    route_shared,
                    stage_full,
                    stage_empty,
                    producer_ready,
                    grid_acquire_ready,
                    pre_ready,
                    pre_shared,
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
class PaddedM128TileSplit64Plan:
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
        return _sm90_padded_m128_tile_split64_wgmma_kernel.warmup(
            *self._kernel_args(),
            grid=(GRID_CTAS,),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )

    def run(self) -> None:
        _sm90_padded_m128_tile_split64_wgmma_kernel[(GRID_CTAS,)](
            *self._kernel_args(),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> PaddedM128TileSplit64Plan:
    """Validate the fixed profile and allocate capture-stable state."""
    if os.environ.get("DG_TLE_MHC_PADDED_M128_TILE_SPLIT64", "0") != "1":
        raise RuntimeError(
            "set DG_TLE_MHC_PADDED_M128_TILE_SPLIT64=1 to enable this experiment"
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
        "kernel_symbol": "_sm90_padded_m128_tile_split64_wgmma_kernel",
        "CTA": GRID_CTAS,
        "grid_ctas": GRID_CTAS,
        "num_m_tiles": NUM_M_TILES,
        "num_splits": NUM_SPLITS,
        "block_m": BLOCK_M,
        "block_n": BLOCK_N,
        "hidden_per_split": HIDDEN_PER_SPLIT,
        "cta_mapping": "tile=pid//64,split=pid%64",
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
    return PaddedM128TileSplit64Plan(
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

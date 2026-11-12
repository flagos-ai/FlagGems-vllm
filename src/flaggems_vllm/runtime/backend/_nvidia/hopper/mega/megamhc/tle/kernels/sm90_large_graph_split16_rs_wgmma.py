"""Production SM90 split-16 large-Graph MegaMHC shared kernel body.

This shape-strict family covers the H4096 DeepSeek-V4-Flash Graph buckets
``T in {400, 416, 432, 448, 464, 496, 512}`` at ``R=4, P=24``.  T432 and
T448 use this body directly; the other admitted buckets wrap it with their
exact non-striped tail ownership.  It keeps the proven one-kernel cooperative
tail but replaces its FP32 shared-A projection with the CUDA winner's producer
geometry:

* 512 root threads each produce two naturally aligned BF16x4 quads per K64;
* the BF16 boundary is written once to both ``new_residual`` and shared A;
* four math warps load BF16 shared A into registers and issue RS-TF32 WGMMA;
* an explicit eight-slot ring holds two complete four-route K64 bundles in
  flight.  Two full-CTA ready/done barrier pairs avoid the TLE pipe pass's
  restriction on inline-PTX BF16x4 stores while preserving producer/math
  overlap across adjacent K64 bundles.

All admitted variants fix split-16, 16 producer warps, four RS-WGMMA warps,
eight pipeline stages, and 128 math registers.  The complete grid contains
only producer CTAs; historical consumer-only and register-sweep variants are
not part of the production contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from .. import fused_post_gemm_t64 as _t64
from . import sm90_cp_async_helpers as _fn_cp
from . import sm90_eager_t1028_p256_wgmma as _eager
from . import sm90_large_graph_role4_k32_wgmma as _base
from . import sm90_shared_memory_helpers as _quad

if _base.LOGICAL_TOKENS < 400:
    raise ValueError("split16 RS family requires a T=400..512 bucket")
if _base.NUM_SPLITS != 16:
    raise ValueError("split16 RS family requires exactly 16 K splits")
TOKENS = _base.LOGICAL_TOKENS
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = HIDDEN * HC_MULT
BLOCK_M = 64
BLOCK_N = 32
BLOCK_K = 64
NUM_SPLITS = 16
NUM_M_TILES = (TOKENS + BLOCK_M - 1) // BLOCK_M
PRODUCER_CTAS = NUM_M_TILES * NUM_SPLITS
GRID_CTAS = PRODUCER_CTAS
PIPE_STAGES = 8
ROOT_WARPS = 16
ROOT_THREADS = ROOT_WARPS * 32
MATH_WARPS = 4
# Keep enough registers for the four RS-WGMMA warps; the scalar producer is
# kept below the complementary 96-register budget by incrementally publishing
# its row-square partials instead of carrying them across all K64 bundles.
MATH_REGS = 128
PHYSICAL_THREADS = ROOT_THREADS + MATH_WARPS * 32
HIDDEN_BLOCKS_PER_SPLIT = HIDDEN // BLOCK_K // NUM_SPLITS

JIT_TOKENS = tl.constexpr(TOKENS)
JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_NUM_SPLITS = tl.constexpr(NUM_SPLITS)
JIT_PRODUCER_CTAS = tl.constexpr(PRODUCER_CTAS)
JIT_GRID_CTAS = tl.constexpr(GRID_CTAS)
JIT_PIPE_STAGES = tl.constexpr(PIPE_STAGES)
JIT_ROOT_THREADS = tl.constexpr(ROOT_THREADS)
JIT_MATH_WARPS = tl.constexpr(MATH_WARPS)
JIT_MATH_REGS = tl.constexpr(MATH_REGS)
JIT_PHYSICAL_THREADS = tl.constexpr(PHYSICAL_THREADS)
JIT_HIDDEN_BLOCKS_PER_SPLIT = tl.constexpr(HIDDEN_BLOCKS_PER_SPLIT)


@triton.jit
def _round_store_global_shared_bf16x4_square(
    global_address,
    shared_address,
    value0,
    value1,
    value2,
    value3,
    valid,
):
    """Round/store a quad and expose only its BF16-boundary square sum.

    Returning four rounded FP32 values makes all four values live in Triton
    IR across the call and pushes the 512-thread producer over its 96-register
    partition budget.  Keep the unpack and four FMAs inside one PTX block and
    return the sole scalar consumed by the row reduction.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred p;
          .reg .b32 packed0, packed1;
          .reg .b16 h0, h1, h2, h3;
          .reg .f32 f0, f1, f2, f3;
          cvt.rn.bf16x2.f32 packed0, $4, $3;
          cvt.rn.bf16x2.f32 packed1, $6, $5;
          setp.ne.u32 p, $7, 0;
          @p st.global.v2.b32 [$1], {packed0, packed1};
          st.shared.v2.b32 [$2], {packed0, packed1};
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 f0, h0;
          cvt.f32.bf16 f1, h1;
          cvt.f32.bf16 f2, h2;
          cvt.f32.bf16 f3, h3;
          mul.rn.f32 $0, f0, f0;
          fma.rn.f32 $0, f1, f1, $0;
          fma.rn.f32 $0, f2, f2, $0;
          fma.rn.f32 $0, f3, f3, $0;
        }
        """,
        constraints="=&f,l,r,f,f,f,f,r",
        args=[
            global_address,
            shared_address,
            value0,
            value1,
            value2,
            value3,
            tl.cast(valid, tl.uint32),
        ],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _issue_fn_cp_async_k64(
    fn_ptr, fn_stages, stage: tl.constexpr, flat_hidden_base, thread
):
    row = thread // 16
    column = (thread % 16) * 4
    valid_row = row < JIT_PROJECTION_OUTPUTS
    safe_row = tl.where(valid_row, row, 0)
    global_address = tl.cast(
        fn_ptr + safe_row * JIT_FLAT_HIDDEN + flat_hidden_base + column,
        tl.uint64,
    )
    shared_address = _quad._shared_address(
        tle.gpu.local_ptr(fn_stages.slot(stage), (row, column))
    )
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
def _cp_async_wait_all():
    return tl.inline_asm_elementwise(
        asm="{ cp.async.wait_group 0; mov.u32 $0, 0; }",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _split16_rs_root(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    new_residual_ptr,
    partials_ptr,
    q_stages,
    fn_stages,
    ready0,
    ready1,
    done0,
    done1,
    producer_ready,
    grid_acquire_ready,
    coeff_ready,
    post_shared,
    comb_shared,
    pre_shared,
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
    """Produce four K64 route stages with two BF16x4 quads per thread."""
    physical_thread = _t64._physical_thread_id_x()
    half_lane = physical_thread % 16

    if is_producer:
        local_m = tl.arange(0, JIT_BLOCK_M)
        tokens = tile * JIT_BLOCK_M + local_m
        post_column = tl.arange(0, JIT_HC_MULT)
        comb_column = tl.arange(0, JIT_HC_MULT * JIT_HC_MULT)
        tl.store(
            tle.gpu.local_ptr(post_shared),
            tl.load(
                post_mix_ptr + tokens[:, None] * JIT_HC_MULT + post_column[None, :],
                mask=tokens[:, None] < JIT_TOKENS,
                other=0.0,
            ),
        )
        tl.store(
            tle.gpu.local_ptr(comb_shared),
            tl.load(
                comb_res_mix_ptr
                + tokens[:, None] * JIT_HC_MULT * JIT_HC_MULT
                + comb_column[None, :],
                mask=tokens[:, None] < JIT_TOKENS,
                other=0.0,
            ),
        )
        tle.gpu.barrier_wait(coeff_ready)

        for hb in tl.static_range(0, JIT_HIDDEN_BLOCKS_PER_SPLIT):
            # With two independent bundles the producer may stage bundle N
            # while math consumes N-1.  After staging N, first retire N-1,
            # then publish N.  This ordering is essential: publishing N
            # before joining done(N-1) would leave root waiting at ready(N)
            # while math waits at done(N-1).
            hidden_base = (split * JIT_HIDDEN_BLOCKS_PER_SPLIT + hb) * JIT_BLOCK_K
            _issue_fn_cp_async_k64(
                fn_ptr,
                fn_stages,
                (hb % 2) * JIT_HC_MULT,
                hidden_base,
                physical_thread,
            )
            _issue_fn_cp_async_k64(
                fn_ptr,
                fn_stages,
                (hb % 2) * JIT_HC_MULT + 1,
                JIT_HIDDEN + hidden_base,
                physical_thread,
            )
            _issue_fn_cp_async_k64(
                fn_ptr,
                fn_stages,
                (hb % 2) * JIT_HC_MULT + 2,
                2 * JIT_HIDDEN + hidden_base,
                physical_thread,
            )
            _issue_fn_cp_async_k64(
                fn_ptr,
                fn_stages,
                (hb % 2) * JIT_HC_MULT + 3,
                3 * JIT_HIDDEN + hidden_base,
                physical_thread,
            )
            _ = _cp_async_commit_group()

            for quad_iteration in tl.static_range(0, 2):
                linear_quad = physical_thread + quad_iteration * JIT_ROOT_THREADS
                row = linear_quad // 16
                column = (linear_quad % 16) * 4
                token = tile * JIT_BLOCK_M + row
                token_active = token < JIT_TOKENS
                hidden_quad = hidden_base + column
                x0, x1, x2, x3 = _eager._load_global_bf16x4_masked(
                    tl.cast(x_ptr + token * JIT_HIDDEN + hidden_quad, tl.uint64),
                    token_active,
                )
                r00, r01, r02, r03 = _eager._load_global_bf16x4_masked(
                    tl.cast(
                        residual_ptr + token * JIT_FLAT_HIDDEN + hidden_quad,
                        tl.uint64,
                    ),
                    token_active,
                )
                r10, r11, r12, r13 = _eager._load_global_bf16x4_masked(
                    tl.cast(
                        residual_ptr
                        + token * JIT_FLAT_HIDDEN
                        + JIT_HIDDEN
                        + hidden_quad,
                        tl.uint64,
                    ),
                    token_active,
                )
                r20, r21, r22, r23 = _eager._load_global_bf16x4_masked(
                    tl.cast(
                        residual_ptr
                        + token * JIT_FLAT_HIDDEN
                        + 2 * JIT_HIDDEN
                        + hidden_quad,
                        tl.uint64,
                    ),
                    token_active,
                )
                r30, r31, r32, r33 = _eager._load_global_bf16x4_masked(
                    tl.cast(
                        residual_ptr
                        + token * JIT_FLAT_HIDDEN
                        + 3 * JIT_HIDDEN
                        + hidden_quad,
                        tl.uint64,
                    ),
                    token_active,
                )
                quad_square = 0.0
                for channel in tl.static_range(0, JIT_HC_MULT):
                    post = tl.load(tle.gpu.local_ptr(post_shared, (row, channel)))
                    comb0 = tl.load(tle.gpu.local_ptr(comb_shared, (row, channel)))
                    comb1 = tl.load(
                        tle.gpu.local_ptr(comb_shared, (row, channel + JIT_HC_MULT))
                    )
                    comb2 = tl.load(
                        tle.gpu.local_ptr(comb_shared, (row, channel + 2 * JIT_HC_MULT))
                    )
                    comb3 = tl.load(
                        tle.gpu.local_ptr(comb_shared, (row, channel + 3 * JIT_HC_MULT))
                    )
                    q0 = (
                        x0 * post
                        + r00 * comb0
                        + r10 * comb1
                        + r20 * comb2
                        + r30 * comb3
                    )
                    q1 = (
                        x1 * post
                        + r01 * comb0
                        + r11 * comb1
                        + r21 * comb2
                        + r31 * comb3
                    )
                    q2 = (
                        x2 * post
                        + r02 * comb0
                        + r12 * comb1
                        + r22 * comb2
                        + r32 * comb3
                    )
                    q3 = (
                        x3 * post
                        + r03 * comb0
                        + r13 * comb1
                        + r23 * comb2
                        + r33 * comb3
                    )
                    quad_square += _round_store_global_shared_bf16x4_square(
                        tl.cast(
                            new_residual_ptr
                            + token * JIT_FLAT_HIDDEN
                            + channel * JIT_HIDDEN
                            + hidden_quad,
                            tl.uint64,
                        ),
                        _quad._shared_address(
                            tle.gpu.local_ptr(
                                q_stages.slot((hb % 2) * JIT_HC_MULT + channel),
                                (row, column),
                            )
                        ),
                        q0,
                        q1,
                        q2,
                        q3,
                        token_active,
                    )
                reduced_square = _eager._half_warp_reduce16_sum(quad_square)
                _ = _eager._accumulate_global_row_square(
                    tl.cast(
                        partials_ptr
                        + split * JIT_TOKENS * JIT_BLOCK_N
                        + token * JIT_BLOCK_N
                        + JIT_PROJECTION_OUTPUTS,
                        tl.uint64,
                    ),
                    reduced_square,
                    half_lane,
                    token_active,
                    hb == 0,
                )

            # The issuing root threads must retire their own async group.
            # The following full-CTA ready barrier then publishes both Fn and
            # the BF16 A stage to the RS-WGMMA partition.
            _ = _cp_async_wait_all()
            if hb > 0:
                if (hb - 1) % 2 == 0:
                    tle.gpu.barrier_wait(done0)
                else:
                    tle.gpu.barrier_wait(done1)
            if hb % 2 == 0:
                tle.gpu.barrier_wait(ready0)
            else:
                tle.gpu.barrier_wait(ready1)

        # All earlier bundles were retired before their successor was
        # published; only the final bundle remains outstanding.
        tle.gpu.barrier_wait(done1)

    tle.gpu.barrier_wait(producer_ready)
    _ = _base._monotonic_grid_arrive_and_wait(
        tl.cast(grid_counter_ptr, tl.uint64), physical_thread
    )
    tle.gpu.barrier_wait(grid_acquire_ready)
    _base._full_tail_barrier_sync()
    _base._pipe_role4_hidden_tail(
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
def _split16_rs_math(
    partials_ptr,
    q_stages,
    fn_stages,
    ready0,
    ready1,
    done0,
    done1,
    producer_ready,
    grid_acquire_ready,
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
    if is_producer:
        for hb in tl.static_range(0, JIT_HIDDEN_BLOCKS_PER_SPLIT):
            if hb % 2 == 0:
                tle.gpu.barrier_wait(ready0)
            else:
                tle.gpu.barrier_wait(ready1)
            for channel in tl.static_range(0, JIT_HC_MULT):
                q_registers = tl.load(
                    tle.gpu.local_ptr(q_stages.slot((hb % 2) * JIT_HC_MULT + channel))
                ).to(tl.float32)
                accumulator = tle.gpu.wgmma(
                    q_registers,
                    fn_stages.slot((hb % 2) * JIT_HC_MULT + channel),
                    accumulator,
                    input_precision="tf32",
                    trans_b=True,
                )
                accumulator = tle.gpu.wgmma_wait(0, accumulator)
            if hb % 2 == 0:
                tle.gpu.barrier_wait(done0)
            else:
                tle.gpu.barrier_wait(done1)

        m = tile * JIT_BLOCK_M + tl.arange(0, JIT_BLOCK_M)
        n = tl.arange(0, JIT_BLOCK_N)
        tl.store(
            partials_ptr
            + split * JIT_TOKENS * JIT_BLOCK_N
            + m[:, None] * JIT_BLOCK_N
            + n[None, :],
            accumulator,
            mask=(m[:, None] < JIT_TOKENS) & (n[None, :] < JIT_PROJECTION_OUTPUTS),
        )

    tle.gpu.barrier_wait(producer_ready)
    tle.gpu.barrier_wait(grid_acquire_ready)
    physical_thread = _t64._physical_thread_id_x()
    lane = physical_thread % 32
    role = (physical_thread - JIT_ROOT_THREADS) // 32
    token = pid + role * JIT_GRID_CTAS
    token_active = token < JIT_TOKENS
    packed_sum = 0.0
    for source_split in tl.static_range(0, JIT_NUM_SPLITS):
        packed_sum += tl.load(
            partials_ptr
            + source_split * JIT_TOKENS * JIT_BLOCK_N
            + token * JIT_BLOCK_N
            + lane,
            mask=token_active & (lane <= JIT_PROJECTION_OUTPUTS),
            other=0.0,
        )
    _base._pipe_role4_routing_finalize(
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
    )


@triton.jit
def _sm90_large_graph_split16_rs_wgmma_kernel(
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
    is_producer = pid < JIT_PRODUCER_CTAS
    producer_pid = tl.where(is_producer, pid, 0)
    tile = producer_pid // JIT_NUM_SPLITS
    split = producer_pid % JIT_NUM_SPLITS
    tl.static_assert(JIT_PHYSICAL_THREADS == 640)
    tl.static_assert(JIT_GRID_CTAS <= 132)

    q_stages = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_BLOCK_M, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    fn_stages = tle.gpu.alloc(
        [JIT_PIPE_STAGES, JIT_BLOCK_N, JIT_BLOCK_K],
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
    pre_shared = tle.gpu.alloc(
        [4, JIT_HC_MULT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    producer_ready = tle.gpu.alloc_barrier(arrive_count=JIT_PHYSICAL_THREADS)
    grid_acquire_ready = tle.gpu.alloc_barrier(arrive_count=JIT_PHYSICAL_THREADS)
    coeff_ready = tle.gpu.alloc_barrier(arrive_count=JIT_ROOT_THREADS)
    ready0 = tle.gpu.alloc_barrier(arrive_count=JIT_PHYSICAL_THREADS)
    ready1 = tle.gpu.alloc_barrier(arrive_count=JIT_PHYSICAL_THREADS)
    done0 = tle.gpu.alloc_barrier(arrive_count=JIT_PHYSICAL_THREADS)
    done1 = tle.gpu.alloc_barrier(arrive_count=JIT_PHYSICAL_THREADS)

    tle.gpu.warp_specialize(
        [
            (
                _split16_rs_root,
                (
                    x_ptr,
                    residual_ptr,
                    post_mix_ptr,
                    comb_res_mix_ptr,
                    fn_ptr,
                    new_residual_ptr,
                    partials_ptr,
                    q_stages,
                    fn_stages,
                    ready0,
                    ready1,
                    done0,
                    done1,
                    producer_ready,
                    grid_acquire_ready,
                    coeff_ready,
                    post_shared,
                    comb_shared,
                    pre_shared,
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
                _split16_rs_math,
                (
                    partials_ptr,
                    q_stages,
                    fn_stages,
                    ready0,
                    ready1,
                    done0,
                    done1,
                    producer_ready,
                    grid_acquire_ready,
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


@dataclass(frozen=True)
class LargeGraphSplit16RSPlan:
    base: _base.LargeGraphRole4Plan
    metadata: dict[str, object]

    @property
    def grid_ctas(self) -> int:
        return GRID_CTAS

    @property
    def partials(self) -> torch.Tensor:
        return self.base.partials

    @property
    def workspace_tensors(self) -> dict[str, torch.Tensor]:
        return self.base.workspace_tensors

    @property
    def counter_tensors(self) -> dict[str, torch.Tensor]:
        return self.base.counter_tensors

    def _kernel_args(self) -> tuple[object, ...]:
        return self.base._kernel_args()

    def compile(self):
        return _sm90_large_graph_split16_rs_wgmma_kernel.warmup(
            *self._kernel_args(),
            grid=(GRID_CTAS,),
            num_warps=ROOT_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )

    def run(self) -> None:
        _sm90_large_graph_split16_rs_wgmma_kernel[(GRID_CTAS,)](
            *self._kernel_args(),
            num_warps=ROOT_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> LargeGraphSplit16RSPlan:
    base = _base.prepare_tle_sm90_mega_mhc(inputs, outputs)
    metadata = dict(base.metadata)
    metadata.update(
        {
            "kernel_symbol": "_sm90_large_graph_split16_rs_wgmma_kernel",
            "pipeline_implementation": (
                "explicit BF16-A ring + Fn cp.async + RS-WGMMA"
            ),
            "pipeline_k": BLOCK_K,
            "pipeline_slots": PIPE_STAGES,
            "producer_quad_width": 4,
            "producer_quads_per_thread_per_k64": 2,
            "projection_wgmma_operand_mode": "register-A/shared-B TF32",
            "coefficient_cache_bytes": BLOCK_M * (HC_MULT + HC_MULT * HC_MULT) * 4,
            "audited_spill_admission": TOKENS == 432,
            "expected_spills": 2 if TOKENS == 432 else 0,
            "expected_local_bytes_per_thread": 8 if TOKENS == 432 else 0,
            "experimental": False,
            "dispatch_enabled": TOKENS == 432,
        }
    )
    return LargeGraphSplit16RSPlan(base=base, metadata=metadata)


__all__ = [
    "GRID_CTAS",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
]

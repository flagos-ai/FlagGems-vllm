"""Shared exact-shape SM90 TLE MegaMHC kernel for eager P384 buckets.

This is the production migration of the CUDA P384 split-1 family for exact
``T in {12991, 13005, 13021, 16384}`` at ``H=4096, R=4, P=24``.
One M64 CTA owns the complete H4096 reduction for its token tile, so the
kernel has no grid-wide barrier, counter, cooperative launch, or auxiliary
workspace.  The admitted CTA contains twelve producer warps and four WGMMA
math warps, matching the CUDA p384 family's 12+4-warp geometry.  The producer
partition streams BF16
M64xK64 A fragments and FP32
N32xK64 weights through a four-slot named-barrier ring; the worker partition converts
A to FP32 registers and executes TF32 RS-WGMMA.  Projection, Normal-MHC
routing, and the BF16-boundary RMSNorm tail are then completed inside the
same CTA and the prepared plan enqueues exactly one kernel.  The hidden tail
keeps two consecutive route loads in flight so the HBM load-use latency is
overlapped without adding a second kernel or workspace.

The fixed production configuration is manual512 with twelve producer warps,
four math warps, a four-stage K64 ring, TMA Fn staging, route12 tail scheduling,
shared-round hidden processing, and ``maxnreg=64``.  The implementation is
shared by the four admitted eager token buckets.  The
token count remains a ``tl.constexpr`` kernel argument, so every bucket keeps
an independent compiled specialization without copying this source file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from .. import fused_post_gemm_t64 as _t64
from . import sm90_eager_t1028_p256_k32_ptx_best_recovered as _common
from . import sm90_small_family as _small

SUPPORTED_TOKENS = frozenset((12991, 13005, 13021, 16384))
HIDDEN = 4096
HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = HIDDEN * HC_MULT
SINKHORN_ITERS = 20
BLOCK_M = 64
BLOCK_N = 32
BLOCK_K = 64
NUM_K64_BLOCKS = HIDDEN // 64


@dataclass(frozen=True)
class P384ShapeConfig:
    """Immutable host/JIT identity for one exact eager token bucket."""

    tokens: int

    def __post_init__(self) -> None:
        if self.tokens not in SUPPORTED_TOKENS:
            raise ValueError(f"unsupported P384 token bucket {self.tokens}")

    @property
    def num_m_tiles(self) -> int:
        return (self.tokens + BLOCK_M - 1) // BLOCK_M

    @property
    def kernel_symbol(self) -> str:
        return f"_sm90_eager_t{self.tokens}_p384_rs_wgmma_kernel"


@dataclass(frozen=True)
class P384CompileConfig:
    """The single admitted Hopper P384 implementation configuration."""

    cta_mode: str = "manual512"
    num_warps: int = 16
    math_warps: int = 4
    pipe_stages: int = 4
    wgmma_k: int = 64
    active_k64_blocks: int = 64
    accum_mode: str = "ptx-persistent"
    producer_mode: str = "ptx-quad"
    fn_stage: str = "tma"
    tail_schedule: str = "route12"
    manual_hidden_mode: str = "shared-round"
    maxnreg: int = 64


COMPILE_CONFIG = P384CompileConfig()
CTA_MODE = COMPILE_CONFIG.cta_mode
NUM_WARPS = COMPILE_CONFIG.num_warps
MATH_WARPS = COMPILE_CONFIG.math_warps
PIPE_STAGES = COMPILE_CONFIG.pipe_stages
WGMMA_K = COMPILE_CONFIG.wgmma_k
WGMMA_K_FRAGMENTS = BLOCK_K // WGMMA_K
PRODUCER_WARPS = 12
TOTAL_LOGICAL_THREADS = 512
ACTIVE_PRODUCER_WARPS = PRODUCER_WARPS
ACTIVE_K64_BLOCKS = COMPILE_CONFIG.active_k64_blocks
# Compatibility constants consumed by the existing exact-shape launch adapters.
TAIL_BATCHES = 22
TAIL_WORK_PHASE = 3
ACCUM_MODE = COMPILE_CONFIG.accum_mode
PRODUCER_MODE = COMPILE_CONFIG.producer_mode
TAIL_SCHEDULE = COMPILE_CONFIG.tail_schedule
MANUAL_HIDDEN_MODE = COMPILE_CONFIG.manual_hidden_mode

JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_ACTIVE_K64_BLOCKS = tl.constexpr(ACTIVE_K64_BLOCKS)
JIT_PRE_ROWS = tl.constexpr(BLOCK_M)
JIT_RMS_WEIGHT_ELEMENTS = tl.constexpr(HIDDEN)
JIT_WGMMA_K = tl.constexpr(WGMMA_K)
JIT_WGMMA_K_FRAGMENTS = tl.constexpr(WGMMA_K_FRAGMENTS)


@triton.jit
def _manual512_cta_sync():
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
def _manual512_b1_projection_ptx(
    x_ptr,
    residual_ptr,
    fn_ptr,
    new_residual_ptr,
    debug_projection_ptr,
    post_shared_address,
    comb_shared_address,
    q_shared_address0,
    q_shared_address1,
    q_shared_address2,
    q_shared_address3,
    fn_shared_address0,
    fn_shared_address1,
    fn_shared_address2,
    fn_shared_address3,
    projection_shared_address,
    tile,
    active_k64_blocks,
    fn_tensormap_address,
    fn_tma_barrier_address,
    use_fn_tma,
    q_handoff_barrier_address,
    use_k32_handoff,
    write_debug_mirror,
    token_limit: tl.constexpr,
):
    """Production K64/TMA role-aware 12+4-warp projection.

    All sixteen physical warps enter one lexical PTX scope.  This is
    intentional: a high-level Triton branch around the producer inserts
    implicit ``bar.sync 0`` instructions for shared tensor stores, which can
    never be reached by the four math warps.  The admitted call fixes Fn to
    TMA staging, K64 handoff, and disables the legacy debug mirror.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred producer, math, active, valid, n_valid, again;
          .reg .pred ch0, ch1, ch2, ch3, more_channel, more_half, leader;
          .reg .pred use_tma, init_tma, k32_handoff, init_q, complete;
          .reg .pred half_waiter;
          .reg .pred more_q_half, mirror_enabled, mirror_row0, mirror_row1;
          .reg .pred first_stage, more_stage;
          .reg .pred row0_valid, row1_valid;
          .reg .u32 tid, mtid, lane, warp, linear, channel, rem, n, k;
          .reg .u32 half, k_low, n_row, n_swizzle, k_swizzle, smem_offset;
          .reg .u32 safe_n, source_bytes;
          .reg .u32 fn_base, q_base, quad, row, column, load_idx, bank;
          .reg .u32 elem, byte_offset, post_address, comb_address;
          .reg .u32 row0, row1, row_base, half4, address, encoded_fn;
          .reg .u32 half_desc32, odd_pair, pair_in_bank, out_column;
          .reg .u32 lane4, token, token0, token1, stage, barrier_id;
          .reg .u32 tma_bar, tma_coord, tma_smem1, tma_phase;
          .reg .u32 q_bar, q_half;
          .reg .u64 global_address, global_offset, desc_base;
          .reg .u64 desc0, desc1, desc2, desc3, half_desc64;
          .reg .u64 x_address, residual_address, new_residual_address;
          .reg .u64 out_address0, out_address1;
          .reg .b32 value, xp0, xp1;
          .reg .b32 rp00, rp01, rp10, rp11, rp20, rp21, rp30, rp31;
          .reg .b32 packed0, packed1;
          .reg .b16 xh0, xh1, xh2, xh3;
          .reg .b16 r0h0, r0h1, r0h2, r0h3;
          .reg .b16 r1h0, r1h1, r1h2, r1h3;
          .reg .b16 r2h0, r2h1, r2h2, r2h3;
          .reg .b16 r3h0, r3h1, r3h2, r3h3;
          .reg .b16 bh0, bh1, bh2, bh3;
          .reg .b32 a<16>;
          .reg .f32 x0, x1, x2, x3;
          .reg .f32 r00, r01, r02, r03;
          .reg .f32 r10, r11, r12, r13;
          .reg .f32 r20, r21, r22, r23;
          .reg .f32 r30, r31, r32, r33;
          .reg .f32 post, comb0, comb1, comb2, comb3;
          .reg .f32 q0, q1, q2, q3;
          .reg .f32 c<16>;
          .reg .f32 square0, square1, shuffle_tmp;

          mov.u32 tid, %tid.x;
          setp.ne.u32 use_tma, $21, 0;
          setp.ne.u32 k32_handoff, $23, 0;
          setp.ne.u32 mirror_enabled, $24, 0;
          setp.eq.u32 init_tma, tid, 0;
          and.pred init_tma, init_tma, use_tma;
          @init_tma mbarrier.init.shared::cta.b64 [$20], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+8], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+16], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+24], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+32], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+40], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+48], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+56], 1;
          setp.eq.u32 init_q, tid, 0;
          and.pred init_q, init_q, k32_handoff;
          @init_q mbarrier.init.shared::cta.b64 [$22], 384;
          @init_q mbarrier.init.shared::cta.b64 [$22+8], 384;
          @init_q mbarrier.init.shared::cta.b64 [$22+16], 384;
          @init_q mbarrier.init.shared::cta.b64 [$22+24], 384;
          @init_q mbarrier.init.shared::cta.b64 [$22+32], 384;
          @init_q mbarrier.init.shared::cta.b64 [$22+40], 384;
          @init_q mbarrier.init.shared::cta.b64 [$22+48], 384;
          @init_q mbarrier.init.shared::cta.b64 [$22+56], 384;
          // A host-encoded tensor map is copied to device global before the
          // launch.  CUDA kernel entry does not acquire the tensormap proxy,
          // so the sole TMA issuer performs that acquire exactly once.
          @init_tma fence.proxy.tensormap::generic.acquire.gpu [$19], 128;
          // Every warp executes this rendezvous.  It publishes the uniform
          // coefficient fill and TMA-barrier initialization before the role
          // split below.
          bar.sync 0, 512;
          setp.lt.u32 producer, tid, 384;
          @!producer bra.uni DG_TLE_MHC_P384_MANUAL_B1_MATH;

          mov.u32 stage, 0;
        DG_TLE_MHC_P384_MANUAL_B1_PRODUCER_STAGE:
          setp.eq.u32 first_stage, stage, 0;
          mov.u32 channel, 0;
        DG_TLE_MHC_P384_MANUAL_B1_FN_CHANNEL:
          @first_stage bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_COPY;
          add.u32 barrier_id, channel, 12;
          barrier.cta.sync barrier_id, 512;
        DG_TLE_MHC_P384_MANUAL_B1_FN_COPY:
          @use_tma bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_COPY;
          // One route owns 512 naturally aligned FP32x4 vectors.  All twelve
          // producer warps issue one vector; the first four warps issue a
          // second.  Each route is a distinct cp.async group so its full
          // barrier can be published as soon as wait_group retires it.
          mov.u32 linear, tid;
        DG_TLE_MHC_P384_MANUAL_B1_FN_LOOP:
          setp.lt.u32 active, linear, 512;
          @!active bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_ROUTE_DONE;
          shr.u32 n, linear, 4;
          and.b32 k, linear, 15;
          shl.b32 k, k, 2;
          setp.lt.u32 n_valid, n, 24;
          selp.u32 safe_n, n, 0, n_valid;
          selp.u32 source_bytes, 16, 0, n_valid;
          mul.wide.u32 global_offset, safe_n, 65536;
          cvt.u64.u32 global_address, channel;
          shl.b64 global_address, global_address, 14;
          add.u64 global_offset, global_offset, global_address;
          cvt.u64.u32 global_address, k;
          shl.b64 global_address, global_address, 2;
          add.u64 global_offset, global_offset, global_address;
          cvt.u64.u32 global_address, stage;
          shl.b64 global_address, global_address, 8;
          add.u64 global_offset, global_offset, global_address;
          add.u64 global_address, $3, global_offset;

          mov.u32 fn_base, $12;
          setp.eq.u32 ch1, channel, 1;
          setp.eq.u32 ch2, channel, 2;
          setp.eq.u32 ch3, channel, 3;
          @ch1 mov.u32 fn_base, $13;
          @ch2 mov.u32 fn_base, $14;
          @ch3 mov.u32 fn_base, $15;
          and.b32 k_low, k, 31;
          and.b32 half, k, 32;
          or.b32 n_row, n, half;
          shl.b32 n_row, n_row, 7;
          and.b32 n_swizzle, n, 7;
          shl.b32 n_swizzle, n_swizzle, 4;
          shl.b32 k_swizzle, k_low, 2;
          xor.b32 n_swizzle, n_swizzle, k_swizzle;
          or.b32 smem_offset, n_row, n_swizzle;
          add.u32 address, fn_base, smem_offset;
          cp.async.cg.shared.global [address], [global_address], 16, source_bytes;
          add.u32 linear, linear, 384;
          bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_LOOP;
        DG_TLE_MHC_P384_MANUAL_B1_FN_ROUTE_DONE:
          cp.async.commit_group;
          bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_CHANNEL_DONE;

        DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_COPY:
          // The 128B-swizzled K64 tile is two legal N32xK32 TMA boxes.
          // Only physical thread zero issues them; Post arithmetic below
          // overlaps their transfer exactly as in the CUDA producer.
          setp.eq.u32 leader, tid, 0;
          @!leader bra DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_DONE;
          mov.u32 fn_base, $12;
          setp.eq.u32 ch1, channel, 1;
          setp.eq.u32 ch2, channel, 2;
          setp.eq.u32 ch3, channel, 3;
          @ch1 mov.u32 fn_base, $13;
          @ch2 mov.u32 fn_base, $14;
          @ch3 mov.u32 fn_base, $15;
          shl.b32 tma_coord, channel, 12;
          shl.b32 byte_offset, stage, 6;
          add.u32 tma_coord, tma_coord, byte_offset;
          @k32_handoff bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_K32;
          shl.b32 byte_offset, channel, 3;
          add.u32 tma_bar, $20, byte_offset;
          mbarrier.arrive.expect_tx.shared::cta.b64
            _, [tma_bar], 8192;
          cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [fn_base], [$19, {tma_coord, 0}], [tma_bar];
          add.u32 tma_coord, tma_coord, 32;
          add.u32 tma_smem1, fn_base, 4096;
          cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [tma_smem1], [$19, {tma_coord, 0}], [tma_bar];
          bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_DONE;

        DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_K32:
          shl.b32 byte_offset, channel, 4;
          add.u32 tma_bar, $20, byte_offset;
          mbarrier.arrive.expect_tx.shared::cta.b64
            _, [tma_bar], 4096;
          cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [fn_base], [$19, {tma_coord, 0}], [tma_bar];
          add.u32 tma_coord, tma_coord, 32;
          add.u32 tma_smem1, fn_base, 4096;
          add.u32 tma_bar, tma_bar, 8;
          mbarrier.arrive.expect_tx.shared::cta.b64
            _, [tma_bar], 4096;
          cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [tma_smem1], [$19, {tma_coord, 0}], [tma_bar];
        DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_DONE:

        DG_TLE_MHC_P384_MANUAL_B1_FN_CHANNEL_DONE:
          add.u32 channel, channel, 1;
          setp.lt.u32 more_channel, channel, 4;
          @more_channel bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_CHANNEL;

          // Each producer thread owns up to three BF16x4 source quads.  Five
          // packed input loads are reused across all four output routes.
          mov.u32 q_half, 0;
        DG_TLE_MHC_P384_MANUAL_B1_QUAD_HALF:
          mov.u32 quad, tid;
        DG_TLE_MHC_P384_MANUAL_B1_QUAD_LOOP:
          @k32_handoff bra.uni DG_TLE_MHC_P384_MANUAL_B1_QUAD_MAP_K32;
          setp.lt.u32 active, quad, 1024;
          shr.u32 row, quad, 4;
          and.b32 column, quad, 15;
          shl.b32 column, column, 2;
          bra.uni DG_TLE_MHC_P384_MANUAL_B1_QUAD_MAP_DONE;
        DG_TLE_MHC_P384_MANUAL_B1_QUAD_MAP_K32:
          setp.lt.u32 active, quad, 512;
          shr.u32 row, quad, 3;
          and.b32 column, quad, 7;
          shl.b32 column, column, 2;
          shl.b32 byte_offset, q_half, 5;
          add.u32 column, column, byte_offset;
        DG_TLE_MHC_P384_MANUAL_B1_QUAD_MAP_DONE:
          @!active bra.uni DG_TLE_MHC_P384_MANUAL_B1_QUAD_DONE;
          mad.lo.u32 token, $17, 64, row;
          setp.lt.u32 valid, token, $25;

          mul.wide.u32 global_offset, token, 8192;
          cvt.u64.u32 global_address, column;
          shl.b64 global_address, global_address, 1;
          add.u64 global_offset, global_offset, global_address;
          cvt.u64.u32 global_address, stage;
          shl.b64 global_address, global_address, 7;
          add.u64 global_offset, global_offset, global_address;
          add.u64 x_address, $1, global_offset;
          mul.wide.u32 global_offset, token, 32768;
          cvt.u64.u32 global_address, column;
          shl.b64 global_address, global_address, 1;
          add.u64 global_offset, global_offset, global_address;
          cvt.u64.u32 global_address, stage;
          shl.b64 global_address, global_address, 7;
          add.u64 global_offset, global_offset, global_address;
          add.u64 residual_address, $2, global_offset;
          add.u64 new_residual_address, $4, global_offset;

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
          @valid ld.global.v2.b32 {xp0, xp1}, [x_address];
          @valid ld.global.v2.b32 {rp00, rp01}, [residual_address];
          @valid ld.global.v2.b32 {rp10, rp11}, [residual_address+8192];
          @valid ld.global.v2.b32 {rp20, rp21}, [residual_address+16384];
          @valid ld.global.v2.b32 {rp30, rp31}, [residual_address+24576];
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

          // The route count is the compile-time constant four.  Hoist the
          // channel-invariant row and swizzled-Q address work, then spell out
          // each route with fixed shared/global offsets.  This removes the
          // hottest IMAD/IADD/ISETP/BRA loop in the NCU instruction profile.
          mad.lo.u32 post_address, row, 16, $6;
          mad.lo.u32 comb_address, row, 64, $7;
          shr.u32 load_idx, column, 3;
          and.b32 elem, column, 7;
          xor.b32 bank, row, load_idx;
          and.b32 bank, bank, 7;
          mul.lo.u32 smem_offset, row, 128;
          shl.b32 byte_offset, bank, 4;
          add.u32 smem_offset, smem_offset, byte_offset;
          shl.b32 byte_offset, elem, 1;
          add.u32 smem_offset, smem_offset, byte_offset;

          // Route 0.
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
          @valid st.global.v2.b32
            [new_residual_address], {packed0, packed1};
          add.u32 address, $8, smem_offset;
          st.shared.v2.b32 [address], {packed0, packed1};

          // Route 1.
          ld.shared.f32 post, [post_address+4];
          ld.shared.f32 comb0, [comb_address+4];
          ld.shared.f32 comb1, [comb_address+20];
          ld.shared.f32 comb2, [comb_address+36];
          ld.shared.f32 comb3, [comb_address+52];
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
          @valid st.global.v2.b32
            [new_residual_address+8192], {packed0, packed1};
          add.u32 address, $9, smem_offset;
          st.shared.v2.b32 [address], {packed0, packed1};

          // Route 2.
          ld.shared.f32 post, [post_address+8];
          ld.shared.f32 comb0, [comb_address+8];
          ld.shared.f32 comb1, [comb_address+24];
          ld.shared.f32 comb2, [comb_address+40];
          ld.shared.f32 comb3, [comb_address+56];
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
          @valid st.global.v2.b32
            [new_residual_address+16384], {packed0, packed1};
          add.u32 address, $10, smem_offset;
          st.shared.v2.b32 [address], {packed0, packed1};

          // Route 3.
          ld.shared.f32 post, [post_address+12];
          ld.shared.f32 comb0, [comb_address+12];
          ld.shared.f32 comb1, [comb_address+28];
          ld.shared.f32 comb2, [comb_address+44];
          ld.shared.f32 comb3, [comb_address+60];
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
          @valid st.global.v2.b32
            [new_residual_address+24576], {packed0, packed1};
          add.u32 address, $11, smem_offset;
          st.shared.v2.b32 [address], {packed0, packed1};
          add.u32 quad, quad, 384;
          bra.uni DG_TLE_MHC_P384_MANUAL_B1_QUAD_LOOP;
        DG_TLE_MHC_P384_MANUAL_B1_QUAD_DONE:
          @k32_handoff bra.uni DG_TLE_MHC_P384_MANUAL_B1_Q_K32_PUBLISH;
          @use_tma bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_RETIRE;
          // Retire and publish the four route groups independently.  This
          // allows math to consume route 0 while later groups are still in
          // flight instead of imposing one all-route wait per hidden block.
          cp.async.wait_group 3;
          fence.proxy.async.shared::cta;
          barrier.cta.arrive 8, 512;
          cp.async.wait_group 2;
          fence.proxy.async.shared::cta;
          barrier.cta.arrive 9, 512;
          cp.async.wait_group 1;
          fence.proxy.async.shared::cta;
          barrier.cta.arrive 10, 512;
          cp.async.wait_group 0;
          fence.proxy.async.shared::cta;
          barrier.cta.arrive 11, 512;
          bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_RETIRE_DONE;

        DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_RETIRE:
          // Fold Q and Fn readiness into the existing full named barrier.
          // The 383 non-leader producers can arrive immediately after their
          // Q stores.  Thread zero first acquires the route's TMA mbarrier,
          // then contributes the final producer arrival.  Math therefore
          // needs only its full-barrier sync, not 128 redundant TMA polls.
          setp.eq.u32 leader, tid, 0;
          mov.u32 channel, 0;
        DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_PUBLISH_LOOP:
          @!leader bra DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_PUBLISH_ARRIVE;
          shl.b32 byte_offset, channel, 3;
          add.u32 tma_bar, $20, byte_offset;
          and.b32 tma_phase, stage, 1;
        DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_PUBLISH_WAIT:
          mbarrier.try_wait.parity.shared::cta.b64
            complete, [tma_bar], tma_phase;
          @!complete bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_PUBLISH_WAIT;
        DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_PUBLISH_ARRIVE:
          add.u32 barrier_id, channel, 8;
          barrier.cta.arrive barrier_id, 512;
          add.u32 channel, channel, 1;
          setp.lt.u32 more_channel, channel, 4;
          @more_channel bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_TMA_PUBLISH_LOOP;

          bra.uni DG_TLE_MHC_P384_MANUAL_B1_FN_RETIRE_DONE;

        DG_TLE_MHC_P384_MANUAL_B1_Q_K32_PUBLISH:
          // Each half has an independent 384-arrival Q barrier, allowing the
          // math WG to consume K0:32 while producers generate K32:64.
          shl.b32 byte_offset, q_half, 3;
          add.u32 q_bar, $22, byte_offset;
          mbarrier.arrive.release.cta.shared::cta.b64 _, [q_bar];
          add.u32 q_bar, q_bar, 16;
          mbarrier.arrive.release.cta.shared::cta.b64 _, [q_bar];
          add.u32 q_bar, q_bar, 16;
          mbarrier.arrive.release.cta.shared::cta.b64 _, [q_bar];
          add.u32 q_bar, q_bar, 16;
          mbarrier.arrive.release.cta.shared::cta.b64 _, [q_bar];
          add.u32 q_half, q_half, 1;
          setp.lt.u32 more_q_half, q_half, 2;
          @more_q_half bra.uni DG_TLE_MHC_P384_MANUAL_B1_QUAD_HALF;

        DG_TLE_MHC_P384_MANUAL_B1_FN_RETIRE_DONE:
          add.u32 stage, stage, 1;
          setp.lt.u32 more_stage, stage, $18;
          @more_stage bra.uni DG_TLE_MHC_P384_MANUAL_B1_PRODUCER_STAGE;
          bra.uni DG_TLE_MHC_P384_MANUAL_B1_DONE;

        DG_TLE_MHC_P384_MANUAL_B1_MATH:
          sub.u32 mtid, tid, 384;
          and.b32 lane, mtid, 31;
          shr.u32 warp, mtid, 5;
          shr.u32 row0, lane, 2;
          mad.lo.u32 row0, warp, 16, row0;
          add.u32 row1, row0, 8;
          and.b32 elem, lane, 3;
          mov.f32 c0, 0f00000000;
          mov.f32 c1, 0f00000000;
          mov.f32 c2, 0f00000000;
          mov.f32 c3, 0f00000000;
          mov.f32 c4, 0f00000000;
          mov.f32 c5, 0f00000000;
          mov.f32 c6, 0f00000000;
          mov.f32 c7, 0f00000000;
          mov.f32 c8, 0f00000000;
          mov.f32 c9, 0f00000000;
          mov.f32 c10, 0f00000000;
          mov.f32 c11, 0f00000000;
          mov.f32 c12, 0f00000000;
          mov.f32 c13, 0f00000000;
          mov.f32 c14, 0f00000000;
          mov.f32 c15, 0f00000000;
          mov.f32 square0, 0f00000000;
          mov.f32 square1, 0f00000000;
          mov.u32 stage, 0;
        DG_TLE_MHC_P384_MANUAL_B1_MATH_STAGE:
          mov.u32 channel, 0;

        DG_TLE_MHC_P384_MANUAL_B1_MATH_CHANNEL:
          @k32_handoff bra.uni DG_TLE_MHC_P384_MANUAL_B1_MATH_BASE_SETUP;
          add.u32 barrier_id, channel, 8;
          barrier.cta.sync barrier_id, 512;
        DG_TLE_MHC_P384_MANUAL_B1_MATH_TMA_READY:
          fence.proxy.async.shared::cta;
        DG_TLE_MHC_P384_MANUAL_B1_MATH_BASE_SETUP:
          mov.u32 q_base, $8;
          mov.u32 fn_base, $12;
          setp.eq.u32 ch1, channel, 1;
          setp.eq.u32 ch2, channel, 2;
          setp.eq.u32 ch3, channel, 3;
          @ch1 mov.u32 q_base, $9;
          @ch2 mov.u32 q_base, $10;
          @ch3 mov.u32 q_base, $11;
          @ch1 mov.u32 fn_base, $13;
          @ch2 mov.u32 fn_base, $14;
          @ch3 mov.u32 fn_base, $15;
          mad.lo.u32 row_base, row0, 128, q_base;
          shl.b32 byte_offset, elem, 1;
          add.u32 row_base, row_base, byte_offset;
          bfe.u32 encoded_fn, fn_base, 4, 14;
          cvt.u64.u32 desc_base, encoded_fn;
          add.u64 desc_base, desc_base, 0x4000004000000000;
          mov.u32 half, 0;

        DG_TLE_MHC_P384_MANUAL_B1_MATH_HALF:
          @!k32_handoff bra.uni DG_TLE_MHC_P384_MANUAL_B1_MATH_HALF_READY;
          // Q and Fn halves have independent mbarriers.  The acquire waits
          // publish generic Q stores and TMA writes; the proxy fence then
          // orders both before WGMMA's async-proxy shared reads.  One lane
          // per math warp polls the two mbarriers and a warp barrier fans the
          // completion out.  A warpgroup-wide named barrier here costs about
          // as much as the K32 overlap we are trying to expose.
          shl.b32 byte_offset, channel, 4;
          shl.b32 address, half, 3;
          add.u32 byte_offset, byte_offset, address;
          add.u32 q_bar, $22, byte_offset;
          add.u32 tma_bar, $20, byte_offset;
          and.b32 tma_phase, stage, 1;
          setp.eq.u32 half_waiter, lane, 0;
          @!half_waiter bra DG_TLE_MHC_P384_MANUAL_B1_MATH_HALF_WAIT_DONE;
        DG_TLE_MHC_P384_MANUAL_B1_MATH_Q_HALF_WAIT:
          mbarrier.try_wait.parity.shared::cta.b64
            complete, [q_bar], tma_phase;
          @!complete bra.uni DG_TLE_MHC_P384_MANUAL_B1_MATH_Q_HALF_WAIT;
        DG_TLE_MHC_P384_MANUAL_B1_MATH_FN_HALF_WAIT:
          mbarrier.try_wait.parity.shared::cta.b64
            complete, [tma_bar], tma_phase;
          @!complete bra.uni DG_TLE_MHC_P384_MANUAL_B1_MATH_FN_HALF_WAIT;
        DG_TLE_MHC_P384_MANUAL_B1_MATH_HALF_WAIT_DONE:
          bar.warp.sync 0xffffffff;
          fence.proxy.async.shared::cta;
        DG_TLE_MHC_P384_MANUAL_B1_MATH_HALF_READY:
          shl.b32 half4, half, 2;
          add.u32 load_idx, half4, 0;
          xor.b32 bank, row0, load_idx;
          and.b32 bank, bank, 7;
          shl.b32 byte_offset, bank, 4;
          add.u32 address, row_base, byte_offset;
          ld.shared.b16 bh0, [address];
          ld.shared.b16 bh2, [address+8];
          ld.shared.b16 bh1, [address+1024];
          ld.shared.b16 bh3, [address+1032];
          cvt.f32.bf16 a0, bh0;
          cvt.f32.bf16 a1, bh1;
          cvt.f32.bf16 a2, bh2;
          cvt.f32.bf16 a3, bh3;
          fma.rn.f32 square0, a0, a0, square0;
          fma.rn.f32 square0, a2, a2, square0;
          fma.rn.f32 square1, a1, a1, square1;
          fma.rn.f32 square1, a3, a3, square1;
          add.u32 load_idx, half4, 1;
          xor.b32 bank, row0, load_idx;
          and.b32 bank, bank, 7;
          shl.b32 byte_offset, bank, 4;
          add.u32 address, row_base, byte_offset;
          ld.shared.b16 bh0, [address];
          ld.shared.b16 bh2, [address+8];
          ld.shared.b16 bh1, [address+1024];
          ld.shared.b16 bh3, [address+1032];
          cvt.f32.bf16 a4, bh0;
          cvt.f32.bf16 a5, bh1;
          cvt.f32.bf16 a6, bh2;
          cvt.f32.bf16 a7, bh3;
          fma.rn.f32 square0, a4, a4, square0;
          fma.rn.f32 square0, a6, a6, square0;
          fma.rn.f32 square1, a5, a5, square1;
          fma.rn.f32 square1, a7, a7, square1;
          add.u32 load_idx, half4, 2;
          xor.b32 bank, row0, load_idx;
          and.b32 bank, bank, 7;
          shl.b32 byte_offset, bank, 4;
          add.u32 address, row_base, byte_offset;
          ld.shared.b16 bh0, [address];
          ld.shared.b16 bh2, [address+8];
          ld.shared.b16 bh1, [address+1024];
          ld.shared.b16 bh3, [address+1032];
          cvt.f32.bf16 a8, bh0;
          cvt.f32.bf16 a9, bh1;
          cvt.f32.bf16 a10, bh2;
          cvt.f32.bf16 a11, bh3;
          fma.rn.f32 square0, a8, a8, square0;
          fma.rn.f32 square0, a10, a10, square0;
          fma.rn.f32 square1, a9, a9, square1;
          fma.rn.f32 square1, a11, a11, square1;
          add.u32 load_idx, half4, 3;
          xor.b32 bank, row0, load_idx;
          and.b32 bank, bank, 7;
          shl.b32 byte_offset, bank, 4;
          add.u32 address, row_base, byte_offset;
          ld.shared.b16 bh0, [address];
          ld.shared.b16 bh2, [address+8];
          ld.shared.b16 bh1, [address+1024];
          ld.shared.b16 bh3, [address+1032];
          cvt.f32.bf16 a12, bh0;
          cvt.f32.bf16 a13, bh1;
          cvt.f32.bf16 a14, bh2;
          cvt.f32.bf16 a15, bh3;
          fma.rn.f32 square0, a12, a12, square0;
          fma.rn.f32 square0, a14, a14, square0;
          fma.rn.f32 square1, a13, a13, square1;
          fma.rn.f32 square1, a15, a15, square1;
          shl.b32 half_desc32, half, 8;
          cvt.u64.u32 half_desc64, half_desc32;
          add.u64 desc0, desc_base, half_desc64;
          add.u64 desc1, desc0, 2;
          add.u64 desc2, desc0, 4;
          add.u64 desc3, desc0, 6;
          wgmma.fence.sync.aligned;
          wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32
            {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15},
            {a0,a1,a2,a3}, desc0, 1, 1, 1;
          wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32
            {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15},
            {a4,a5,a6,a7}, desc1, 1, 1, 1;
          wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32
            {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15},
            {a8,a9,a10,a11}, desc2, 1, 1, 1;
          wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32
            {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15},
            {a12,a13,a14,a15}, desc3, 1, 1, 1;
          add.u32 half, half, 1;
          setp.lt.u32 more_half, half, 2;
          @more_half bra.uni DG_TLE_MHC_P384_MANUAL_B1_MATH_HALF;
          // RS source registers are consumed when each WGMMA is issued, so
          // the two K32 halves can reuse a0..a15 exactly like the retained
          // K64 path.  Retire one eight-MMA group per route, not one group
          // per half (which would add 256 completion waits per CTA).
          wgmma.commit_group.sync.aligned;
          wgmma.wait_group.sync.aligned 0;
          add.u32 barrier_id, channel, 12;
          barrier.cta.arrive barrier_id, 512;
          add.u32 channel, channel, 1;
          setp.lt.u32 more_channel, channel, 4;
          @more_channel bra.uni DG_TLE_MHC_P384_MANUAL_B1_MATH_CHANNEL;
          add.u32 stage, stage, 1;
          setp.lt.u32 more_stage, stage, $18;
          @more_stage bra.uni DG_TLE_MHC_P384_MANUAL_B1_MATH_STAGE;

          shfl.sync.bfly.b32 shuffle_tmp, square0, 1, 31, -1;
          add.f32 square0, square0, shuffle_tmp;
          shfl.sync.bfly.b32 shuffle_tmp, square0, 2, 31, -1;
          add.f32 square0, square0, shuffle_tmp;
          shfl.sync.bfly.b32 shuffle_tmp, square1, 1, 31, -1;
          add.f32 square1, square1, shuffle_tmp;
          shfl.sync.bfly.b32 shuffle_tmp, square1, 2, 31, -1;
          add.f32 square1, square1, shuffle_tmp;
          shr.u32 odd_pair, lane, 1;
          and.b32 odd_pair, odd_pair, 1;
          and.b32 pair_in_bank, lane, 1;
          shl.b32 out_column, odd_pair, 2;
          shl.b32 byte_offset, pair_in_bank, 1;
          add.u32 out_column, out_column, byte_offset;
          shl.b32 byte_offset, out_column, 2;
          mad.lo.u32 address, row0, 128, $16;
          add.u32 address, address, byte_offset;
          st.shared.v2.f32 [address], {c0,c1};
          st.shared.v2.f32 [address+1024], {c2,c3};
          st.shared.v2.f32 [address+32], {c4,c5};
          st.shared.v2.f32 [address+1056], {c6,c7};
          st.shared.v2.f32 [address+64], {c8,c9};
          st.shared.v2.f32 [address+1088], {c10,c11};
          // Routing reads the BF16-derived square sum from projection
          // column 24.  The global mirror below is validation-only and does
          // not publish this value to the in-kernel tail.
          and.b32 lane4, lane, 3;
          setp.eq.u32 leader, lane4, 0;
          @leader st.shared.f32 [address+96], square0;
          @leader st.shared.f32 [address+1120], square1;

          mad.lo.u32 token0, $17, 64, row0;
          mad.lo.u32 token1, $17, 64, row1;
          setp.lt.u32 row0_valid, token0, $25;
          setp.lt.u32 row1_valid, token1, $25;
          and.pred mirror_row0, row0_valid, mirror_enabled;
          and.pred mirror_row1, row1_valid, mirror_enabled;
          mul.wide.u32 global_offset, token0, 8192;
          cvt.u64.u32 global_address, out_column;
          shl.b64 global_address, global_address, 2;
          add.u64 global_offset, global_offset, global_address;
          add.u64 out_address0, $5, global_offset;
          mul.wide.u32 global_offset, token1, 8192;
          cvt.u64.u32 global_address, out_column;
          shl.b64 global_address, global_address, 2;
          add.u64 global_offset, global_offset, global_address;
          add.u64 out_address1, $5, global_offset;
          @mirror_row0 st.global.v2.f32 [out_address0], {c0,c1};
          @mirror_row1 st.global.v2.f32 [out_address1], {c2,c3};
          @mirror_row0 st.global.v2.f32 [out_address0+32], {c4,c5};
          @mirror_row1 st.global.v2.f32 [out_address1+32], {c6,c7};
          @mirror_row0 st.global.v2.f32 [out_address0+64], {c8,c9};
          @mirror_row1 st.global.v2.f32 [out_address1+64], {c10,c11};
          and.b32 lane4, lane, 3;
          setp.eq.u32 leader, lane4, 0;
          and.pred leader, leader, mirror_row0;
          @leader st.global.f32 [out_address0+96], square0;
          setp.eq.u32 leader, lane4, 0;
          and.pred leader, leader, mirror_row1;
          @leader st.global.f32 [out_address1+96], square1;

        DG_TLE_MHC_P384_MANUAL_B1_DONE:
          // Producer warps may reach DONE before the math warpgroup stores C.
          // Publish the complete projection before the unified 16-warp tail.
          bar.sync 0, 512;
          mov.u32 $0, 0;
        }
        """,
        constraints=("=r,l,l,l,l,l,r,r,r,r,r,r,r,r,r,r,r,r,r,l,r,r,r,r,r,r"),
        args=[
            tl.cast(x_ptr, tl.uint64),
            tl.cast(residual_ptr, tl.uint64),
            tl.cast(fn_ptr, tl.uint64),
            tl.cast(new_residual_ptr, tl.uint64),
            tl.cast(debug_projection_ptr, tl.uint64),
            post_shared_address,
            comb_shared_address,
            q_shared_address0,
            q_shared_address1,
            q_shared_address2,
            q_shared_address3,
            fn_shared_address0,
            fn_shared_address1,
            fn_shared_address2,
            fn_shared_address3,
            projection_shared_address,
            tile,
            active_k64_blocks,
            fn_tensormap_address,
            fn_tma_barrier_address,
            tl.cast(use_fn_tma, tl.uint32),
            q_handoff_barrier_address,
            tl.cast(use_k32_handoff, tl.uint32),
            tl.cast(write_debug_mirror, tl.uint32),
            tl.cast(token_limit, tl.uint32),
        ],
        dtype=tl.uint32,
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
    stage_active,
):
    """Compute all four routes for one BF16x4 source quad.

    Keeping the five packed input quads and route temporaries inside one PTX
    lexical scope mirrors the CUDA p384 producer.  Triton therefore sees no
    20-value FP32 tensor live range spanning the four route stores.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred p, stage, ch1, ch2, ch3, again;
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
          .reg .f32 x0, x1, x2, x3;
          .reg .f32 r00, r01, r02, r03;
          .reg .f32 r10, r11, r12, r13;
          .reg .f32 r20, r21, r22, r23;
          .reg .f32 r30, r31, r32, r33;
          .reg .f32 post, comb0, comb1, comb2, comb3;
          .reg .f32 q0, q1, q2, q3;

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
          setp.ne.u32 stage, $11, 0;
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
        DG_TLE_MHC_P384_POST_ROUTE_LOOP:
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
          @stage st.shared.v2.b32 [q_address], {packed0, packed1};

          add.u32 channel, channel, 1;
          setp.lt.u32 again, channel, 4;
          @again bra.uni DG_TLE_MHC_P384_POST_ROUTE_LOOP;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,l,r,r,r,r,r,r,r,r",
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
            tl.cast(stage_active, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _hidden_shared_round_ptx(
    residual_address,
    weight_shared_address,
    output_address,
    pre_shared_address,
    rounded_shared_address,
    warp_sums_shared_address,
    logical_thread,
    lane,
    active,
    barrier_id,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """One 128-thread token worker with BF16x8 ILP and shared retention.

    Each thread owns four pairs of naturally aligned hidden quads.  Both
    independent route loads in a pair are issued before either is consumed,
    hiding long-scoreboard latency while keeping the exact BF16 boundary in
    a retired producer stage.  Four warp partials are reduced through a
    worker-local named barrier, then each pair is normalized and stored.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred enabled, lane0, lane_lt4, again_quad;
          .reg .u32 quad_iter, hidden_quad0, hidden_quad1;
          .reg .u32 byte_offset0, byte_offset1;
          .reg .u32 rounded_address0, rounded_address1;
          .reg .u32 weight_address0, weight_address1;
          .reg .u32 sum_address, worker_warp;
          .reg .u64 global_offset0, global_offset1;
          .reg .u64 residual_ptr0, residual_ptr1, output_ptr0, output_ptr1;
          .reg .b32 packed0, packed1, packed2, packed3;
          .reg .b32 ahead0, ahead1, ahead2, ahead3;
          .reg .b32 weight0, weight1;
          .reg .b16 h0, h1, h2, h3, w0, w1, w2, w3;
          .reg .f32 v0, v1, v2, v3;
          .reg .f32 coeff<4>;
          .reg .f32 q0, q1, q2, q3, q4, q5, q6, q7;
          .reg .f32 s0, s1, s2, s3, total, temp, norm;

          setp.ne.u32 enabled, $9, 0;
          mov.f32 s0, 0f00000000;
          mov.f32 s1, 0f00000000;
          mov.f32 s2, 0f00000000;
          mov.f32 s3, 0f00000000;
          // Pre coefficients are token-invariant across all eight hidden
          // quads owned by this lane.  Load them once instead of repeating
          // the same four shared loads in every quad-pair iteration.
          ld.shared.v4.f32 {coeff0, coeff1, coeff2, coeff3}, [$4];
          mov.u32 quad_iter, 0;
        DG_TLE_MHC_P384_HIDDEN_COMBINE_LOOP:
          mad.lo.u32 hidden_quad0, quad_iter, 256, $7;
          add.u32 hidden_quad1, hidden_quad0, 128;
          mul.lo.u32 byte_offset0, hidden_quad0, 8;
          mul.lo.u32 byte_offset1, hidden_quad1, 8;
          mul.wide.u32 global_offset0, hidden_quad0, 8;
          mul.wide.u32 global_offset1, hidden_quad1, 8;
          add.u32 rounded_address0, $5, byte_offset0;
          add.u32 rounded_address1, $5, byte_offset1;
          mov.f32 q0, 0f00000000;
          mov.f32 q1, 0f00000000;
          mov.f32 q2, 0f00000000;
          mov.f32 q3, 0f00000000;
          mov.f32 q4, 0f00000000;
          mov.f32 q5, 0f00000000;
          mov.f32 q6, 0f00000000;
          mov.f32 q7, 0f00000000;
          add.u64 residual_ptr0, $1, global_offset0;
          add.u64 residual_ptr1, $1, global_offset1;
          mov.b32 packed0, 0;
          mov.b32 packed1, 0;
          mov.b32 packed2, 0;
          mov.b32 packed3, 0;
          mov.b32 ahead0, 0;
          mov.b32 ahead1, 0;
          mov.b32 ahead2, 0;
          mov.b32 ahead3, 0;
          // Two-route lookahead.  Issue route 0 and route 1 before consuming
          // either; each consumed buffer is then immediately reused to
          // prefetch route 2 or route 3.  This breaks the four serial
          // LDG-to-PRMT dependency chains identified by SourceCounters.
          @enabled ld.global.v2.b32 {packed0, packed1}, [residual_ptr0];
          @enabled ld.global.v2.b32 {packed2, packed3}, [residual_ptr1];
          @enabled ld.global.v2.b32
              {ahead0, ahead1}, [residual_ptr0+8192];
          @enabled ld.global.v2.b32
              {ahead2, ahead3}, [residual_ptr1+8192];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 q0, v0, coeff0, q0;
          fma.rn.f32 q1, v1, coeff0, q1;
          fma.rn.f32 q2, v2, coeff0, q2;
          fma.rn.f32 q3, v3, coeff0, q3;
          mov.b32 {h0, h1}, packed2;
          mov.b32 {h2, h3}, packed3;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 q4, v0, coeff0, q4;
          fma.rn.f32 q5, v1, coeff0, q5;
          fma.rn.f32 q6, v2, coeff0, q6;
          fma.rn.f32 q7, v3, coeff0, q7;

          // Route 1.
          @enabled ld.global.v2.b32
              {packed0, packed1}, [residual_ptr0+16384];
          @enabled ld.global.v2.b32
              {packed2, packed3}, [residual_ptr1+16384];
          mov.b32 {h0, h1}, ahead0;
          mov.b32 {h2, h3}, ahead1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 q0, v0, coeff1, q0;
          fma.rn.f32 q1, v1, coeff1, q1;
          fma.rn.f32 q2, v2, coeff1, q2;
          fma.rn.f32 q3, v3, coeff1, q3;
          mov.b32 {h0, h1}, ahead2;
          mov.b32 {h2, h3}, ahead3;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 q4, v0, coeff1, q4;
          fma.rn.f32 q5, v1, coeff1, q5;
          fma.rn.f32 q6, v2, coeff1, q6;
          fma.rn.f32 q7, v3, coeff1, q7;

          // Route 2.
          @enabled ld.global.v2.b32
              {ahead0, ahead1}, [residual_ptr0+24576];
          @enabled ld.global.v2.b32
              {ahead2, ahead3}, [residual_ptr1+24576];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 q0, v0, coeff2, q0;
          fma.rn.f32 q1, v1, coeff2, q1;
          fma.rn.f32 q2, v2, coeff2, q2;
          fma.rn.f32 q3, v3, coeff2, q3;
          mov.b32 {h0, h1}, packed2;
          mov.b32 {h2, h3}, packed3;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 q4, v0, coeff2, q4;
          fma.rn.f32 q5, v1, coeff2, q5;
          fma.rn.f32 q6, v2, coeff2, q6;
          fma.rn.f32 q7, v3, coeff2, q7;

          // Route 3.
          mov.b32 {h0, h1}, ahead0;
          mov.b32 {h2, h3}, ahead1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 q0, v0, coeff3, q0;
          fma.rn.f32 q1, v1, coeff3, q1;
          fma.rn.f32 q2, v2, coeff3, q2;
          fma.rn.f32 q3, v3, coeff3, q3;
          mov.b32 {h0, h1}, ahead2;
          mov.b32 {h2, h3}, ahead3;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 q4, v0, coeff3, q4;
          fma.rn.f32 q5, v1, coeff3, q5;
          fma.rn.f32 q6, v2, coeff3, q6;
          fma.rn.f32 q7, v3, coeff3, q7;

          cvt.rn.bf16x2.f32 packed0, q1, q0;
          cvt.rn.bf16x2.f32 packed1, q3, q2;
          cvt.rn.bf16x2.f32 packed2, q5, q4;
          cvt.rn.bf16x2.f32 packed3, q7, q6;
          st.shared.v2.b32 [rounded_address0], {packed0, packed1};
          st.shared.v2.b32 [rounded_address1], {packed2, packed3};
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 s0, v0, v0, s0;
          fma.rn.f32 s1, v1, v1, s1;
          fma.rn.f32 s2, v2, v2, s2;
          fma.rn.f32 s3, v3, v3, s3;
          mov.b32 {h0, h1}, packed2;
          mov.b32 {h2, h3}, packed3;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          fma.rn.f32 s0, v0, v0, s0;
          fma.rn.f32 s1, v1, v1, s1;
          fma.rn.f32 s2, v2, v2, s2;
          fma.rn.f32 s3, v3, v3, s3;
          add.u32 quad_iter, quad_iter, 1;
          setp.lt.u32 again_quad, quad_iter, 4;
          @again_quad bra.uni DG_TLE_MHC_P384_HIDDEN_COMBINE_LOOP;


          add.rn.f32 total, s0, s1;
          add.rn.f32 temp, s2, s3;
          add.rn.f32 total, total, temp;
          shfl.sync.down.b32 temp, total, 16, 31, 0xffffffff;
          add.rn.f32 total, total, temp;
          shfl.sync.down.b32 temp, total, 8, 31, 0xffffffff;
          add.rn.f32 total, total, temp;
          shfl.sync.down.b32 temp, total, 4, 31, 0xffffffff;
          add.rn.f32 total, total, temp;
          shfl.sync.down.b32 temp, total, 2, 31, 0xffffffff;
          add.rn.f32 total, total, temp;
          shfl.sync.down.b32 temp, total, 1, 31, 0xffffffff;
          add.rn.f32 total, total, temp;
          setp.eq.u32 lane0, $8, 0;
          shr.u32 worker_warp, $7, 5;
          mul.lo.u32 byte_offset0, worker_warp, 4;
          add.u32 sum_address, $6, byte_offset0;
          @lane0 st.shared.f32 [sum_address], total;
          barrier.cta.sync $10, 128;

          setp.lt.u32 lane_lt4, $8, 4;
          mov.f32 total, 0f00000000;
          mul.lo.u32 byte_offset0, $8, 4;
          add.u32 sum_address, $6, byte_offset0;
          @lane_lt4 ld.shared.f32 total, [sum_address];
          shfl.sync.bfly.b32 temp, total, 2, 31, 0xffffffff;
          add.rn.f32 total, total, temp;
          shfl.sync.bfly.b32 temp, total, 1, 31, 0xffffffff;
          add.rn.f32 total, total, temp;
          shfl.sync.idx.b32 total, total, 0, 31, 0xffffffff;
          mul.rn.f32 total, total, 0f39800000;
          add.rn.f32 total, total, $11;
          rsqrt.approx.f32 norm, total;
          mul.rn.f32 norm, norm, $12;

          mov.u32 quad_iter, 0;
        DG_TLE_MHC_P384_HIDDEN_NORMALIZE_LOOP:
          mad.lo.u32 hidden_quad0, quad_iter, 256, $7;
          add.u32 hidden_quad1, hidden_quad0, 128;
          mul.lo.u32 byte_offset0, hidden_quad0, 8;
          mul.lo.u32 byte_offset1, hidden_quad1, 8;
          mul.wide.u32 global_offset0, hidden_quad0, 8;
          mul.wide.u32 global_offset1, hidden_quad1, 8;
          add.u32 rounded_address0, $5, byte_offset0;
          add.u32 rounded_address1, $5, byte_offset1;
          add.u32 weight_address0, $2, byte_offset0;
          add.u32 weight_address1, $2, byte_offset1;
          ld.shared.v2.b32 {packed0, packed1}, [rounded_address0];
          ld.shared.v2.b32 {packed2, packed3}, [rounded_address1];
          ld.shared.v2.b32 {weight0, weight1}, [weight_address0];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          mov.b32 {w0, w1}, weight0;
          mov.b32 {w2, w3}, weight1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          cvt.f32.bf16 q0, w0;
          cvt.f32.bf16 q1, w1;
          cvt.f32.bf16 q2, w2;
          cvt.f32.bf16 q3, w3;
          mul.rn.f32 v0, v0, norm;
          mul.rn.f32 v1, v1, norm;
          mul.rn.f32 v2, v2, norm;
          mul.rn.f32 v3, v3, norm;
          mul.rn.f32 v0, v0, q0;
          mul.rn.f32 v1, v1, q1;
          mul.rn.f32 v2, v2, q2;
          mul.rn.f32 v3, v3, q3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          add.u64 output_ptr0, $3, global_offset0;
          @enabled st.global.v2.b32 [output_ptr0], {packed0, packed1};

          ld.shared.v2.b32 {weight0, weight1}, [weight_address1];
          mov.b32 {h0, h1}, packed2;
          mov.b32 {h2, h3}, packed3;
          mov.b32 {w0, w1}, weight0;
          mov.b32 {w2, w3}, weight1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          cvt.f32.bf16 q0, w0;
          cvt.f32.bf16 q1, w1;
          cvt.f32.bf16 q2, w2;
          cvt.f32.bf16 q3, w3;
          mul.rn.f32 v0, v0, norm;
          mul.rn.f32 v1, v1, norm;
          mul.rn.f32 v2, v2, norm;
          mul.rn.f32 v3, v3, norm;
          mul.rn.f32 v0, v0, q0;
          mul.rn.f32 v1, v1, q1;
          mul.rn.f32 v2, v2, q2;
          mul.rn.f32 v3, v3, q3;
          cvt.rn.bf16x2.f32 packed2, v1, v0;
          cvt.rn.bf16x2.f32 packed3, v3, v2;
          add.u64 output_ptr1, $3, global_offset1;
          @enabled st.global.v2.b32 [output_ptr1], {packed2, packed3};
          add.u32 quad_iter, quad_iter, 1;
          setp.lt.u32 again_quad, quad_iter, 4;
          @again_quad bra.uni DG_TLE_MHC_P384_HIDDEN_NORMALIZE_LOOP;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,r,l,r,r,r,r,r,r,r,f,f",
        args=[
            residual_address,
            weight_shared_address,
            output_address,
            pre_shared_address,
            rounded_shared_address,
            warp_sums_shared_address,
            logical_thread,
            lane,
            tl.cast(active, tl.uint32),
            barrier_id,
            rmsnorm_eps,
            rmsnorm_scale,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _routing_finalize_predicated(
    pre_shared,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    packed_sum,
    lane,
    active,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    """Normal-MHC routing with uniform execution and predicated stores."""
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
    base = tl.load(mix_bases_ptr + lane, mask=projection_mask, other=0.0).to(tl.float32)
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
          .reg .pred lane_ok, enabled, p;
          setp.lt.u32 lane_ok, $3, 4;
          setp.ne.u32 enabled, $4, 0;
          and.pred p, lane_ok, enabled;
          @p st.shared.f32 [$1], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,f,r,r",
        args=[pre_address, pre_mix, lane, tl.cast(active, tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    post_base = tl.cast(new_post_mix_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred ge4, lt8, enabled, p;
          .reg .u32 index;
          .reg .u64 offset, address;
          setp.ge.u32 ge4, $3, 4;
          setp.lt.u32 lt8, $3, 8;
          and.pred p, ge4, lt8;
          setp.ne.u32 enabled, $4, 0;
          and.pred p, p, enabled;
          sub.u32 index, $3, 4;
          mul.wide.u32 offset, index, 4;
          add.u64 address, $1, offset;
          @p st.global.f32 [address], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r,r",
        args=[post_base, post_mix, lane, tl.cast(active, tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )

    routing_mask = (lane >= 2 * JIT_HC_MULT) & (lane < JIT_PROJECTION_OUTPUTS)
    sinkhorn = tl.where(routing_mask, mixes, 0.0)
    row_max = _t64._warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _t64._warp_reduce4_sum(sinkhorn)
    sinkhorn *= _t64._fast_rcp_ftz_f32(row_sum)
    sinkhorn += sinkhorn_eps
    column_sum = _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
    sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
    sinkhorn = _common._sinkhorn_nineteen_warp_iterations(sinkhorn, sinkhorn_eps)
    comb_base = tl.cast(new_comb_res_mix_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred ge8, lt24, enabled, p;
          .reg .u32 index;
          .reg .u64 offset, address;
          setp.ge.u32 ge8, $3, 8;
          setp.lt.u32 lt24, $3, 24;
          and.pred p, ge8, lt24;
          setp.ne.u32 enabled, $4, 0;
          and.pred p, p, enabled;
          sub.u32 index, $3, 8;
          mul.wide.u32 offset, index, 4;
          add.u64 address, $1, offset;
          @p st.global.f32 [address], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r,r",
        args=[comb_base, sinkhorn, lane, tl.cast(active, tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _sm90_eager_p384_rs_wgmma_kernel(
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
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
    token_limit: tl.constexpr,
):
    tile = tl.program_id(0)
    tl.static_assert(token_limit > 0)
    tl.static_assert(JIT_BLOCK_M == 64)
    tl.static_assert(JIT_BLOCK_K == 64)

    q_stages = tle.gpu.alloc(
        [4, JIT_WGMMA_K_FRAGMENTS, JIT_BLOCK_M, JIT_WGMMA_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    fn_stages = tle.gpu.alloc(
        [4, JIT_WGMMA_K_FRAGMENTS, JIT_BLOCK_N, JIT_WGMMA_K],
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
    pre_shared = tle.gpu.alloc(
        [JIT_PRE_ROWS, JIT_HC_MULT],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    rms_weight_shared = tle.gpu.alloc(
        [JIT_RMS_WEIGHT_ELEMENTS],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hidden_warp_sums = tle.gpu.alloc(
        [4, 4],
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
    q_handoff_barriers = tle.gpu.alloc(
        [8],
        dtype=tl.uint64,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    # Every physical warp cooperates in the coefficient cache fill.  The
    # role-aware PTX scope publishes it with a full-CTA barrier before
    # assigning warps 0..11 to Post/Fn and warps 12..15 to WGMMA.
    local_m = tl.arange(0, JIT_BLOCK_M)
    m = tile * JIT_BLOCK_M + local_m
    token_mask = m < token_limit
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
    _ = _manual512_b1_projection_ptx(
        x_ptr,
        residual_ptr,
        fn_ptr,
        new_residual_ptr,
        y_bf16_ptr,
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
        JIT_ACTIVE_K64_BLOCKS,
        tl.cast(fn_tensormap_ptr, tl.uint64),
        _shared_address(tle.gpu.local_ptr(fn_tma_barriers, (0,))),
        1,
        _shared_address(tle.gpu.local_ptr(q_handoff_barriers, (0,))),
        0,
        tl.cast(0, tl.uint32),
        token_limit,
    )
    # All sixteen warps are now converged.  Cache the BF16 RMS row, route
    # four tokens per warp, then assign each four-warp group one hidden worker.
    physical_thread = _small._physical_thread_id_x()
    physical_warp = physical_thread // 32
    lane = physical_thread % 32
    for route_batch in tl.static_range(0, 4):
        route_row = route_batch * 16 + physical_warp
        token = tile * JIT_BLOCK_M + route_row
        route_active = token < token_limit
        packed_sum = tl.load(
            tle.gpu.local_ptr(projection_shared, (route_row, lane)),
            mask=(lane <= JIT_PROJECTION_OUTPUTS) & route_active,
            other=0.0,
        )
        _routing_finalize_predicated(
            pre_shared.slot(tl.cast(route_row, tl.int32)),
            mix_scales_ptr,
            mix_bases_ptr,
            new_post_mix_ptr + token * JIT_HC_MULT,
            new_comb_res_mix_ptr + token * JIT_HC_MULT * JIT_HC_MULT,
            packed_sum,
            lane,
            route_active,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sinkhorn_eps,
        )
    _manual512_cta_sync()
    # Projection is dead after routing.  Fill the cache only now so shared
    # liveness reuse cannot overwrite packed C/square values before lane 24
    # broadcasts the projection norm.
    weight_h = tl.arange(0, JIT_HIDDEN)
    tl.store(
        tle.gpu.local_ptr(rms_weight_shared, (weight_h,)),
        tl.load(rmsnorm_weight_ptr + weight_h),
    )
    _manual512_cta_sync()
    worker_idx = physical_warp // 4
    worker_thread = (physical_warp % 4) * 32 + lane
    rounded_address = _shared_address(
        tle.gpu.local_ptr(
            q_stages.slot(tl.cast(worker_idx, tl.int32)).slot(0),
            (0, 0),
        )
    )
    weight_address = _shared_address(tle.gpu.local_ptr(rms_weight_shared, (0,)))
    warp_sums_address = _shared_address(
        tle.gpu.local_ptr(
            hidden_warp_sums.slot(tl.cast(worker_idx, tl.int32)),
            (0,),
        )
    )
    for hidden_batch in tl.range(0, 16):
        route_row = hidden_batch * 4 + worker_idx
        token = tile * JIT_BLOCK_M + route_row
        active = token < token_limit
        _ = _hidden_shared_round_ptx(
            tl.cast(new_residual_ptr + token * JIT_FLAT_HIDDEN, tl.uint64),
            weight_address,
            tl.cast(y_bf16_ptr + token * JIT_HIDDEN, tl.uint64),
            _shared_address(
                tle.gpu.local_ptr(
                    pre_shared.slot(tl.cast(route_row, tl.int32)),
                    (0,),
                )
            ),
            rounded_address,
            warp_sums_address,
            worker_thread,
            lane,
            active,
            1 + worker_idx,
            rmsnorm_eps,
            rmsnorm_scale,
        )


@dataclass(frozen=True)
class P384Plan:
    """Prepared exact-one-launch plan for one immutable P384 shape."""

    shape_config: P384ShapeConfig
    tensors: tuple[torch.Tensor, ...]
    scalars: tuple[float, ...]
    workspace_tensors: dict[str, torch.Tensor]
    counter_tensors: dict[str, torch.Tensor]
    metadata: dict[str, object]
    maxnreg: int

    @property
    def grid_ctas(self) -> int:
        return self.shape_config.num_m_tiles

    def _kernel_args(self) -> tuple[object, ...]:
        return (*self.tensors, *self.scalars, self.shape_config.tokens)

    def compile(self):
        return _sm90_eager_p384_rs_wgmma_kernel.warmup(
            *self._kernel_args(),
            grid=(self.grid_ctas,),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            maxnreg=self.maxnreg,
        )

    def run(self) -> None:
        """Launch the complete production kernel."""
        _sm90_eager_p384_rs_wgmma_kernel[(self.grid_ctas,)](
            *self._kernel_args(),
            num_warps=NUM_WARPS,
            num_stages=1,
            num_ctas=1,
            maxnreg=self.maxnreg,
        )


def prepare_p384_plan(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
    *,
    shape_config: P384ShapeConfig,
    plan_type: type[P384Plan] = P384Plan,
) -> P384Plan:
    """Validate the complete pinned profile; allocate no device workspace."""
    maxnreg = COMPILE_CONFIG.maxnreg
    tokens = shape_config.tokens
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the eager p384 plan outside capture")
    if inputs.get("shifted_prev_mix") is not None:
        raise ValueError("only Normal MHC is supported")
    if outputs.get("new_prev_mix") is not None:
        raise ValueError("Normal MHC requires new_prev_mix=None")

    specs = (
        (inputs.get("x"), "inputs.x", (tokens, HIDDEN), torch.bfloat16),
        (
            inputs.get("residual"),
            "inputs.residual",
            (tokens, HC_MULT, HIDDEN),
            torch.bfloat16,
        ),
        (
            inputs.get("post_mix"),
            "inputs.post_mix",
            (tokens, HC_MULT, 1),
            torch.float32,
        ),
        (
            inputs.get("comb_res_mix"),
            "inputs.comb_res_mix",
            (tokens, HC_MULT, HC_MULT),
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
            (tokens, HC_MULT, HIDDEN),
            torch.bfloat16,
        ),
        (
            outputs.get("new_post_mix"),
            "outputs.new_post_mix",
            (tokens, HC_MULT, 1),
            torch.float32,
        ),
        (
            outputs.get("new_comb_res_mix"),
            "outputs.new_comb_res_mix",
            (tokens, HC_MULT, HC_MULT),
            torch.float32,
        ),
        (
            outputs.get("y_bf16"),
            "outputs.y_bf16",
            (tokens, HIDDEN),
            torch.bfloat16,
        ),
    )
    tensors = tuple(
        _common._require_tensor(value, name=name, shape=shape, dtype=dtype)
        for value, name, shape, dtype in specs
    )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must be on one CUDA device")
    # Encoding and the 128-byte H2D copy happen during plan preparation,
    # never in the timed one-kernel interval.
    from .sm90_global_tensormap_helpers import _encode_global_tensor_map

    fn_tensormap_storage = _encode_global_tensor_map(tensors[4])
    kernel_tensors = (*tensors, fn_tensormap_storage)

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

    metadata: dict[str, object] = {
        "model": "DeepSeek-V4-Flash@60d8d707",
        "execution_mode": "eager",
        "mode": "normal",
        "T": tokens,
        "H": HIDDEN,
        "R": HC_MULT,
        "P": PROJECTION_OUTPUTS,
        "input_dtype": "bf16",
        "weight_dtype": "fp32",
        "accumulator_dtype": "fp32",
        "output_dtype": "bf16/fp32",
        "kernel_count": 1,
        "expected_cuda_kernel_activities": 1,
        "kernel_symbol": shape_config.kernel_symbol,
        "grid_ctas": shape_config.num_m_tiles,
        "block_m": BLOCK_M,
        "block_n": BLOCK_N,
        "block_k": BLOCK_K,
        "num_m_tiles": shape_config.num_m_tiles,
        "num_splits": 1,
        "cta_mapping": "tile=pid; each CTA owns full H4096",
        "producer_warps": PRODUCER_WARPS,
        "active_producer_warps": ACTIVE_PRODUCER_WARPS,
        "math_warps": MATH_WARPS,
        "cta_mode": CTA_MODE,
        "physical_cta_threads": TOTAL_LOGICAL_THREADS,
        "target_registers_per_thread": maxnreg,
        "producer_pipeline_stages": PIPE_STAGES,
        "producer_mode": PRODUCER_MODE,
        "fn_stage_mode": COMPILE_CONFIG.fn_stage,
        "fn_tensormap_storage_bytes": 128,
        "active_k64_blocks": ACTIVE_K64_BLOCKS,
        "accum_mode": ACCUM_MODE,
        "a_storage": "bf16_shared_to_fp32_register",
        "wgmma": f"RS TF32 M64N32K{WGMMA_K} serial wait0",
        "synchronization": "cta_local_named_full_empty_ring",
        "workspace_bytes": 0,
        "counter_bytes": 0,
        "full_grid_barrier": False,
        "cooperative_launch": False,
        "full_grid_residency_required": False,
        "bf16_rounding_boundary": (
            "new_residual before projection and norm_input before rmsnorm"
        ),
        "tail_schedule": TAIL_SCHEDULE,
        "manual_hidden_mode": MANUAL_HIDDEN_MODE,
        "manual_hidden_quad_ilp": (2 if MANUAL_HIDDEN_MODE == "shared-round" else None),
        "experimental": False,
        "dispatch_enabled": True,
        "shape_specialization": (
            "shared P384 source with an exact constexpr token limit"
        ),
    }
    return plan_type(
        shape_config=shape_config,
        tensors=kernel_tensors,
        scalars=(
            float(inputs["hc_norm_eps"]),
            float(inputs["hc_pre_eps"]),
            float(inputs["hc_post_scale"]),
            float(inputs["sinkhorn_eps"]),
            float(inputs["rmsnorm_eps"]),
            float(inputs["rmsnorm_scale"]),
        ),
        workspace_tensors={},
        counter_tensors={},
        metadata=metadata,
        maxnreg=maxnreg,
    )


__all__ = [
    "ACTIVE_K64_BLOCKS",
    "COMPILE_CONFIG",
    "NUM_K64_BLOCKS",
    "NUM_WARPS",
    "P384Plan",
    "P384ShapeConfig",
    "SUPPORTED_TOKENS",
    "TAIL_BATCHES",
    "TAIL_WORK_PHASE",
    "_serial_quad_post_ptx",
    "_sm90_eager_p384_rs_wgmma_kernel",
    "prepare_p384_plan",
]

"""P256 manual512 projection PTX with split-aware Fn TMA coordinates.

This copy keeps the hash-pinned projection body intact except for adding the
CTA hidden-base coordinate to TensorMap loads.  The original global-load path
is byte-for-byte equivalent; the TMA path now addresses each H256/H512 split
instead of repeatedly loading split zero.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _manual512_p256_projection_split_tma_ptx(
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
    token_limit,
    fn_hidden_base,
):
    """Role-aware 12+4-warp projection with compact M64xN32 output.

    All sixteen physical warps enter one lexical PTX scope.  This is
    intentional: a high-level Triton branch around the producer inserts
    implicit ``bar.sync 0`` instructions for shared tensor stores, which can
    never be reached by the four math warps.  The prototype first establishes
    correctness and resource safety for the CUDA migration geometry.  It uses
    ordinary global loads for Fn; later versions can replace those loads with
    the production TMA ring without changing the role protocol.

    The first 24 FP32 projection values and the BF16-derived square sum are
    mirrored into the byte storage backing ``debug_projection_ptr``.  That
    write is probe-only and lets the host validate the otherwise CTA-local
    projection before the full Normal-MHC tail is attached.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred producer, math, active, valid, n_valid, again;
          .reg .pred ch0, ch1, ch2, ch3, more_channel, more_half, leader;
          .reg .pred use_tma, init_tma, complete;
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
          .reg .f32 c<12>;
          .reg .f32 square0, square1, shuffle_tmp;

          mov.u32 tid, %tid.x;
          setp.ne.u32 use_tma, $21, 0;
          setp.eq.u32 init_tma, tid, 0;
          and.pred init_tma, init_tma, use_tma;
          @init_tma mbarrier.init.shared::cta.b64 [$20], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+8], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+16], 1;
          @init_tma mbarrier.init.shared::cta.b64 [$20+24], 1;
          // A host-encoded tensor map is copied to device global before the
          // launch.  CUDA kernel entry does not acquire the tensormap proxy,
          // so the sole TMA issuer performs that acquire exactly once.
          @init_tma fence.proxy.tensormap::generic.acquire.gpu [$19], 128;
          // Every warp executes this rendezvous.  It publishes the uniform
          // coefficient fill and TMA-barrier initialization before the role
          // split below.
          bar.sync 0, 512;
          setp.lt.u32 producer, tid, 384;
          @!producer bra.uni DG_TLE_MHC_T12991_MANUAL_B1_MATH;

          mov.u32 stage, 0;
        DG_TLE_MHC_T12991_MANUAL_B1_PRODUCER_STAGE:
          setp.eq.u32 first_stage, stage, 0;
          mov.u32 channel, 0;
        DG_TLE_MHC_T12991_MANUAL_B1_FN_CHANNEL:
          @first_stage bra.uni DG_TLE_MHC_T12991_MANUAL_B1_FN_COPY;
          add.u32 barrier_id, channel, 12;
          barrier.cta.sync barrier_id, 512;
        DG_TLE_MHC_T12991_MANUAL_B1_FN_COPY:
          @use_tma bra.uni DG_TLE_MHC_T12991_MANUAL_B1_FN_TMA_COPY;
          // N24 needs exactly 384 naturally aligned FP32x4 vectors.  All
          // twelve producer warps issue one vector; unlike the old N32 path,
          // no warp emits zero-fill copies for unused rows 24..31.
          mov.u32 linear, tid;
          shr.u32 n, linear, 4;
          and.b32 k, linear, 15;
          shl.b32 k, k, 2;
          mul.wide.u32 global_offset, n, 65536;
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
          cp.async.cg.shared.global [address], [global_address], 16;
          cp.async.commit_group;
          bra.uni DG_TLE_MHC_T12991_MANUAL_B1_FN_CHANNEL_DONE;

        DG_TLE_MHC_T12991_MANUAL_B1_FN_TMA_COPY:
          // The 128B-swizzled K64 tile is two legal N32xK32 TMA boxes.
          // Only physical thread zero issues them; Post arithmetic below
          // overlaps their transfer exactly as in the CUDA producer.
          setp.eq.u32 leader, tid, 0;
          @!leader bra DG_TLE_MHC_T12991_MANUAL_B1_FN_TMA_DONE;
          shl.b32 byte_offset, channel, 3;
          add.u32 tma_bar, $20, byte_offset;
          mbarrier.arrive.expect_tx.shared::cta.b64
            _, [tma_bar], 8192;
          mov.u32 fn_base, $12;
          setp.eq.u32 ch0, channel, 0;
          setp.eq.u32 ch1, channel, 1;
          setp.eq.u32 ch2, channel, 2;
          setp.eq.u32 ch3, channel, 3;
          @ch1 mov.u32 fn_base, $13;
          @ch2 mov.u32 fn_base, $14;
          @ch3 mov.u32 fn_base, $15;
          shl.b32 tma_coord, channel, 12;
          shl.b32 byte_offset, stage, 6;
          add.u32 tma_coord, tma_coord, byte_offset;
          add.u32 tma_coord, tma_coord, $23;
          cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [fn_base], [$19, {tma_coord, 0}], [tma_bar];
          add.u32 tma_coord, tma_coord, 32;
          add.u32 tma_smem1, fn_base, 4096;
          cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [tma_smem1], [$19, {tma_coord, 0}], [tma_bar];
        DG_TLE_MHC_T12991_MANUAL_B1_FN_TMA_DONE:

        DG_TLE_MHC_T12991_MANUAL_B1_FN_CHANNEL_DONE:
          add.u32 channel, channel, 1;
          setp.lt.u32 more_channel, channel, 4;
          @more_channel bra.uni DG_TLE_MHC_T12991_MANUAL_B1_FN_CHANNEL;

          // Each producer thread owns up to three BF16x4 source quads.  Five
          // packed input loads are reused across all four output routes.
          mov.u32 quad, tid;
        DG_TLE_MHC_T12991_MANUAL_B1_QUAD_LOOP:
          setp.lt.u32 active, quad, 1024;
          @!active bra.uni DG_TLE_MHC_T12991_MANUAL_B1_QUAD_DONE;
          shr.u32 row, quad, 4;
          and.b32 column, quad, 15;
          shl.b32 column, column, 2;
          mad.lo.u32 token, $17, 64, row;
          setp.lt.u32 valid, token, $22;

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

          mov.u32 channel, 0;
        DG_TLE_MHC_T12991_MANUAL_B1_ROUTE_LOOP:
          shl.b32 byte_offset, channel, 2;
          mad.lo.u32 post_address, row, 16, $6;
          add.u32 post_address, post_address, byte_offset;
          mad.lo.u32 comb_address, row, 64, $7;
          add.u32 comb_address, comb_address, byte_offset;
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
          cvt.u64.u32 global_address, channel;
          shl.b64 global_address, global_address, 13;
          add.u64 out_address0, new_residual_address, global_address;
          @valid st.global.v2.b32 [out_address0], {packed0, packed1};

          mov.u32 q_base, $8;
          setp.eq.u32 ch1, channel, 1;
          setp.eq.u32 ch2, channel, 2;
          setp.eq.u32 ch3, channel, 3;
          @ch1 mov.u32 q_base, $9;
          @ch2 mov.u32 q_base, $10;
          @ch3 mov.u32 q_base, $11;
          shr.u32 load_idx, column, 3;
          and.b32 elem, column, 7;
          xor.b32 bank, row, load_idx;
          and.b32 bank, bank, 7;
          mad.lo.u32 address, row, 128, q_base;
          shl.b32 byte_offset, bank, 4;
          add.u32 address, address, byte_offset;
          shl.b32 byte_offset, elem, 1;
          add.u32 address, address, byte_offset;
          st.shared.v2.b32 [address], {packed0, packed1};
          add.u32 channel, channel, 1;
          setp.lt.u32 again, channel, 4;
          @again bra.uni DG_TLE_MHC_T12991_MANUAL_B1_ROUTE_LOOP;
          add.u32 quad, quad, 384;
          bra.uni DG_TLE_MHC_T12991_MANUAL_B1_QUAD_LOOP;

        DG_TLE_MHC_T12991_MANUAL_B1_QUAD_DONE:
          @use_tma bra.uni DG_TLE_MHC_T12991_MANUAL_B1_FN_TMA_RETIRE;
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
          bra.uni DG_TLE_MHC_T12991_MANUAL_B1_FN_RETIRE_DONE;

        DG_TLE_MHC_T12991_MANUAL_B1_FN_TMA_RETIRE:
          // Full barriers publish producer-written Q only.  Math observes Fn
          // completion directly on each TMA mbarrier before WGMMA, so no
          // producer-side TMA wait or 384-thread rendezvous is required.
          barrier.cta.arrive 8, 512;
          barrier.cta.arrive 9, 512;
          barrier.cta.arrive 10, 512;
          barrier.cta.arrive 11, 512;

        DG_TLE_MHC_T12991_MANUAL_B1_FN_RETIRE_DONE:
          add.u32 stage, stage, 1;
          setp.lt.u32 more_stage, stage, $18;
          @more_stage bra.uni DG_TLE_MHC_T12991_MANUAL_B1_PRODUCER_STAGE;
          bra.uni DG_TLE_MHC_T12991_MANUAL_B1_DONE;

        DG_TLE_MHC_T12991_MANUAL_B1_MATH:
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
          mov.f32 square0, 0f00000000;
          mov.f32 square1, 0f00000000;
          mov.u32 stage, 0;
        DG_TLE_MHC_T12991_MANUAL_B1_MATH_STAGE:
          mov.u32 channel, 0;

        DG_TLE_MHC_T12991_MANUAL_B1_MATH_CHANNEL:
          add.u32 barrier_id, channel, 8;
          barrier.cta.sync barrier_id, 512;
          @!use_tma bra.uni DG_TLE_MHC_T12991_MANUAL_B1_MATH_TMA_READY;
          shl.b32 byte_offset, channel, 3;
          add.u32 tma_bar, $20, byte_offset;
          and.b32 tma_phase, stage, 1;
        DG_TLE_MHC_T12991_MANUAL_B1_MATH_TMA_WAIT:
          mbarrier.try_wait.parity.shared::cta.b64
            complete, [tma_bar], tma_phase;
          @!complete bra.uni DG_TLE_MHC_T12991_MANUAL_B1_MATH_TMA_WAIT;
        DG_TLE_MHC_T12991_MANUAL_B1_MATH_TMA_READY:
          fence.proxy.async.shared::cta;
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

        DG_TLE_MHC_T12991_MANUAL_B1_MATH_HALF:
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
          wgmma.mma_async.sync.aligned.m64n24k8.f32.tf32.tf32
            {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11},
            {a0,a1,a2,a3}, desc0, 1, 1, 1;
          wgmma.mma_async.sync.aligned.m64n24k8.f32.tf32.tf32
            {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11},
            {a4,a5,a6,a7}, desc1, 1, 1, 1;
          wgmma.mma_async.sync.aligned.m64n24k8.f32.tf32.tf32
            {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11},
            {a8,a9,a10,a11}, desc2, 1, 1, 1;
          wgmma.mma_async.sync.aligned.m64n24k8.f32.tf32.tf32
            {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11},
            {a12,a13,a14,a15}, desc3, 1, 1, 1;
          add.u32 half, half, 1;
          setp.lt.u32 more_half, half, 2;
          @more_half bra.uni DG_TLE_MHC_T12991_MANUAL_B1_MATH_HALF;
          // Both K32 halves update the same persistent accumulator.  Keep
          // all eight K8 MMAs in one group and pay one completion wait per
          // K64 route instead of one wait per half.
          wgmma.commit_group.sync.aligned;
          wgmma.wait_group.sync.aligned 0;
          add.u32 barrier_id, channel, 12;
          barrier.cta.arrive barrier_id, 512;
          add.u32 channel, channel, 1;
          setp.lt.u32 more_channel, channel, 4;
          @more_channel bra.uni DG_TLE_MHC_T12991_MANUAL_B1_MATH_CHANNEL;
          add.u32 stage, stage, 1;
          setp.lt.u32 more_stage, stage, $18;
          @more_stage bra.uni DG_TLE_MHC_T12991_MANUAL_B1_MATH_STAGE;

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

          // P256-local compact workspace: the caller passes this CTA's
          // M64xN32 base, so row stride is 128B instead of the historical
          // T-wide 8192B debug plane.
          setp.lt.u32 row0_valid, row0, 64;
          setp.lt.u32 row1_valid, row1, 64;
          mul.wide.u32 global_offset, row0, 128;
          cvt.u64.u32 global_address, out_column;
          shl.b64 global_address, global_address, 2;
          add.u64 global_offset, global_offset, global_address;
          add.u64 out_address0, $5, global_offset;
          mul.wide.u32 global_offset, row1, 128;
          cvt.u64.u32 global_address, out_column;
          shl.b64 global_address, global_address, 2;
          add.u64 global_offset, global_offset, global_address;
          add.u64 out_address1, $5, global_offset;
          @row0_valid st.global.v2.f32 [out_address0], {c0,c1};
          @row1_valid st.global.v2.f32 [out_address1], {c2,c3};
          @row0_valid st.global.v2.f32 [out_address0+32], {c4,c5};
          @row1_valid st.global.v2.f32 [out_address1+32], {c6,c7};
          @row0_valid st.global.v2.f32 [out_address0+64], {c8,c9};
          @row1_valid st.global.v2.f32 [out_address1+64], {c10,c11};
          and.b32 lane4, lane, 3;
          setp.eq.u32 leader, lane4, 0;
          and.pred leader, leader, row0_valid;
          @leader st.global.f32 [out_address0+96], square0;
          setp.eq.u32 leader, lane4, 0;
          and.pred leader, leader, row1_valid;
          @leader st.global.f32 [out_address1+96], square1;

        DG_TLE_MHC_T12991_MANUAL_B1_DONE:
          // Producer warps may reach DONE before the math warpgroup stores C.
          // Publish the complete projection before the unified 16-warp tail.
          bar.sync 0, 512;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,l,l,l,r,r,r,r,r,r,r,r,r,r,r,r,r,l,r,r,r,r",
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
            token_limit,
            fn_hidden_base,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )

"""Production token-sharded manual512 PTX tail for exact eager ``T=1028``.

The admitted caller fixes the x8 schedule, phase 3, and a 128-element chunk.
The helper remains isolated from the P384 and Graph families because T=1028
has a real four-row final tile and a distinct row-wave mapping.
"""

from __future__ import annotations

import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _token_shard_combine_bf16x8_ptx(
    pre_base,
    residual_base,
    tail_scratch,
    square_base,
    physical_thread,
    hidden_quad_offset,
    hidden_quad_count,
    live_rows,
    scratch_stride_bytes,
    scratch_chunk_bytes,
    square_chunk_base,
    row_warps: tl.constexpr = 12,
    row_waves: tl.constexpr = 6,
    pre_stride_bytes: tl.constexpr = 128,
    square_stride_bytes: tl.constexpr = 128,
):
    """Combine 256 values as BF16x8 plus one optional 64-value tail."""
    pre_address = tl.cast(pre_base, tl.uint64)
    residual_address = tl.cast(residual_base, tl.uint64)
    scratch_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(tail_scratch, (0, 0, 0))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    square_address = tl.cast(square_base, tl.uint64)
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred row_live, wide, tail_lane, tail_live, lane0, store_sum, more;
          .reg .u32 warp, lane, wave, row, chunk, chunk_index, quad;
          .reg .u32 scratch_row, scratch_tail;
          .reg .b32 a00, a01, a02, a03, b00, b01, b02, b03;
          .reg .b32 c00, c01, c02, c03, d00, d01, d02, d03;
          .reg .b32 packed0, packed1;
          .reg .b16 h0, h1, h2, h3;
          .reg .f32 p0, p1, p2, p3;
          .reg .f32 a0, a1, a2, a3, b0, b1, b2, b3;
          .reg .f32 c0, c1, c2, c3, d0, d1, d2, d3;
          .reg .f32 v0, v1, v2, v3, sum, tmp;
          .reg .u64 pre_row, residual_row, address, offset;
          shr.u32 warp, $5, 5;
          and.b32 lane, $5, 31;
          and.b32 row, warp, 3;
          shr.u32 chunk, warp, 2;
          setp.gt.u32 wide, $7, 64;
          setp.lt.u32 tail_lane, lane, 16;
          mov.u32 wave, 0;
        DG_TLE_MHC_SPLIT_COMBINE8_WAVE:
          setp.lt.u32 row_live, row, $8;
          mov.f32 p0, 0f00000000;
          mov.f32 p1, 0f00000000;
          mov.f32 p2, 0f00000000;
          mov.f32 p3, 0f00000000;
          mul.wide.u32 pre_row, row, $14;
          add.u64 pre_row, $1, pre_row;
          @row_live ld.global.f32 p0, [pre_row];
          @row_live ld.global.f32 p1, [pre_row+4];
          @row_live ld.global.f32 p2, [pre_row+8];
          @row_live ld.global.f32 p3, [pre_row+12];
          mov.f32 sum, 0f00000000;

          mad.lo.u32 quad, chunk, 64, $6;
          mad.lo.u32 quad, lane, 2, quad;
          mul.wide.u32 offset, quad, 8;
          mul.wide.u32 residual_row, row, 32768;
          add.u64 residual_row, $2, residual_row;
          add.u64 residual_row, residual_row, offset;
          mov.b32 a00, 0; mov.b32 a01, 0; mov.b32 a02, 0; mov.b32 a03, 0;
          mov.b32 b00, 0; mov.b32 b01, 0; mov.b32 b02, 0; mov.b32 b03, 0;
          mov.b32 c00, 0; mov.b32 c01, 0; mov.b32 c02, 0; mov.b32 c03, 0;
          mov.b32 d00, 0; mov.b32 d01, 0; mov.b32 d02, 0; mov.b32 d03, 0;
          @row_live ld.global.v4.b32 {a00, a01, a02, a03}, [residual_row];
          add.u64 address, residual_row, 8192;
          @row_live ld.global.v4.b32 {b00, b01, b02, b03}, [address];
          add.u64 address, address, 8192;
          @row_live ld.global.v4.b32 {c00, c01, c02, c03}, [address];
          add.u64 address, address, 8192;
          @row_live ld.global.v4.b32 {d00, d01, d02, d03}, [address];
          mad.lo.u32 scratch_row, row, $9, $3;
          add.u32 scratch_row, scratch_row, $10;
          mad.lo.u32 scratch_row, chunk, 512, scratch_row;
          mad.lo.u32 scratch_row, lane, 16, scratch_row;

          mov.b32 {h0, h1}, a00; mov.b32 {h2, h3}, a01;
          cvt.f32.bf16 a0, h0; cvt.f32.bf16 a1, h1;
          cvt.f32.bf16 a2, h2; cvt.f32.bf16 a3, h3;
          mov.b32 {h0, h1}, b00; mov.b32 {h2, h3}, b01;
          cvt.f32.bf16 b0, h0; cvt.f32.bf16 b1, h1;
          cvt.f32.bf16 b2, h2; cvt.f32.bf16 b3, h3;
          mov.b32 {h0, h1}, c00; mov.b32 {h2, h3}, c01;
          cvt.f32.bf16 c0, h0; cvt.f32.bf16 c1, h1;
          cvt.f32.bf16 c2, h2; cvt.f32.bf16 c3, h3;
          mov.b32 {h0, h1}, d00; mov.b32 {h2, h3}, d01;
          cvt.f32.bf16 d0, h0; cvt.f32.bf16 d1, h1;
          cvt.f32.bf16 d2, h2; cvt.f32.bf16 d3, h3;
          mul.f32 v0, a0, p0; mul.f32 v1, a1, p0;
          mul.f32 v2, a2, p0; mul.f32 v3, a3, p0;
          fma.rn.f32 v0, b0, p1, v0; fma.rn.f32 v1, b1, p1, v1;
          fma.rn.f32 v2, b2, p1, v2; fma.rn.f32 v3, b3, p1, v3;
          fma.rn.f32 v0, c0, p2, v0; fma.rn.f32 v1, c1, p2, v1;
          fma.rn.f32 v2, c2, p2, v2; fma.rn.f32 v3, c3, p2, v3;
          fma.rn.f32 v0, d0, p3, v0; fma.rn.f32 v1, d1, p3, v1;
          fma.rn.f32 v2, d2, p3, v2; fma.rn.f32 v3, d3, p3, v3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          @row_live st.shared.v2.b32 [scratch_row], {packed0, packed1};
          mov.b32 {h0, h1}, packed0; mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0; cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2; cvt.f32.bf16 v3, h3;
          fma.rn.f32 sum, v0, v0, sum; fma.rn.f32 sum, v1, v1, sum;
          fma.rn.f32 sum, v2, v2, sum; fma.rn.f32 sum, v3, v3, sum;

          mov.b32 {h0, h1}, a02; mov.b32 {h2, h3}, a03;
          cvt.f32.bf16 a0, h0; cvt.f32.bf16 a1, h1;
          cvt.f32.bf16 a2, h2; cvt.f32.bf16 a3, h3;
          mov.b32 {h0, h1}, b02; mov.b32 {h2, h3}, b03;
          cvt.f32.bf16 b0, h0; cvt.f32.bf16 b1, h1;
          cvt.f32.bf16 b2, h2; cvt.f32.bf16 b3, h3;
          mov.b32 {h0, h1}, c02; mov.b32 {h2, h3}, c03;
          cvt.f32.bf16 c0, h0; cvt.f32.bf16 c1, h1;
          cvt.f32.bf16 c2, h2; cvt.f32.bf16 c3, h3;
          mov.b32 {h0, h1}, d02; mov.b32 {h2, h3}, d03;
          cvt.f32.bf16 d0, h0; cvt.f32.bf16 d1, h1;
          cvt.f32.bf16 d2, h2; cvt.f32.bf16 d3, h3;
          mul.f32 v0, a0, p0; mul.f32 v1, a1, p0;
          mul.f32 v2, a2, p0; mul.f32 v3, a3, p0;
          fma.rn.f32 v0, b0, p1, v0; fma.rn.f32 v1, b1, p1, v1;
          fma.rn.f32 v2, b2, p1, v2; fma.rn.f32 v3, b3, p1, v3;
          fma.rn.f32 v0, c0, p2, v0; fma.rn.f32 v1, c1, p2, v1;
          fma.rn.f32 v2, c2, p2, v2; fma.rn.f32 v3, c3, p2, v3;
          fma.rn.f32 v0, d0, p3, v0; fma.rn.f32 v1, d1, p3, v1;
          fma.rn.f32 v2, d2, p3, v2; fma.rn.f32 v3, d3, p3, v3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          @row_live st.shared.v2.b32 [scratch_row+8], {packed0, packed1};
          mov.b32 {h0, h1}, packed0; mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0; cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2; cvt.f32.bf16 v3, h3;
          fma.rn.f32 sum, v0, v0, sum; fma.rn.f32 sum, v1, v1, sum;
          fma.rn.f32 sum, v2, v2, sum; fma.rn.f32 sum, v3, v3, sum;

          and.pred tail_live, wide, tail_lane;
          and.pred tail_live, tail_live, row_live;
          mad.lo.u32 quad, chunk, 64, $6;
          add.u32 quad, quad, 64;
          add.u32 quad, quad, lane;
          mul.wide.u32 offset, quad, 8;
          mul.wide.u32 residual_row, row, 32768;
          add.u64 residual_row, $2, residual_row;
          add.u64 residual_row, residual_row, offset;
          mov.b32 a00, 0; mov.b32 a01, 0;
          mov.b32 b00, 0; mov.b32 b01, 0;
          mov.b32 c00, 0; mov.b32 c01, 0;
          mov.b32 d00, 0; mov.b32 d01, 0;
          @tail_live ld.global.v2.b32 {a00, a01}, [residual_row];
          add.u64 address, residual_row, 8192;
          @tail_live ld.global.v2.b32 {b00, b01}, [address];
          add.u64 address, address, 8192;
          @tail_live ld.global.v2.b32 {c00, c01}, [address];
          add.u64 address, address, 8192;
          @tail_live ld.global.v2.b32 {d00, d01}, [address];
          mov.b32 {h0, h1}, a00; mov.b32 {h2, h3}, a01;
          cvt.f32.bf16 a0, h0; cvt.f32.bf16 a1, h1;
          cvt.f32.bf16 a2, h2; cvt.f32.bf16 a3, h3;
          mov.b32 {h0, h1}, b00; mov.b32 {h2, h3}, b01;
          cvt.f32.bf16 b0, h0; cvt.f32.bf16 b1, h1;
          cvt.f32.bf16 b2, h2; cvt.f32.bf16 b3, h3;
          mov.b32 {h0, h1}, c00; mov.b32 {h2, h3}, c01;
          cvt.f32.bf16 c0, h0; cvt.f32.bf16 c1, h1;
          cvt.f32.bf16 c2, h2; cvt.f32.bf16 c3, h3;
          mov.b32 {h0, h1}, d00; mov.b32 {h2, h3}, d01;
          cvt.f32.bf16 d0, h0; cvt.f32.bf16 d1, h1;
          cvt.f32.bf16 d2, h2; cvt.f32.bf16 d3, h3;
          mul.f32 v0, a0, p0; mul.f32 v1, a1, p0;
          mul.f32 v2, a2, p0; mul.f32 v3, a3, p0;
          fma.rn.f32 v0, b0, p1, v0; fma.rn.f32 v1, b1, p1, v1;
          fma.rn.f32 v2, b2, p1, v2; fma.rn.f32 v3, b3, p1, v3;
          fma.rn.f32 v0, c0, p2, v0; fma.rn.f32 v1, c1, p2, v1;
          fma.rn.f32 v2, c2, p2, v2; fma.rn.f32 v3, c3, p2, v3;
          fma.rn.f32 v0, d0, p3, v0; fma.rn.f32 v1, d1, p3, v1;
          fma.rn.f32 v2, d2, p3, v2; fma.rn.f32 v3, d3, p3, v3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          mad.lo.u32 scratch_tail, row, $9, $3;
          add.u32 scratch_tail, scratch_tail, $10;
          mad.lo.u32 scratch_tail, chunk, 512, scratch_tail;
          mad.lo.u32 scratch_tail, lane, 8, scratch_tail;
          add.u32 scratch_tail, scratch_tail, 512;
          @tail_live st.shared.v2.b32 [scratch_tail], {packed0, packed1};
          mov.b32 {h0, h1}, packed0; mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0; cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2; cvt.f32.bf16 v3, h3;
          fma.rn.f32 sum, v0, v0, sum; fma.rn.f32 sum, v1, v1, sum;
          fma.rn.f32 sum, v2, v2, sum; fma.rn.f32 sum, v3, v3, sum;

          shfl.sync.bfly.b32 tmp, sum, 16, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 8, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 4, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 2, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          shfl.sync.bfly.b32 tmp, sum, 1, 0x1f, 0xffffffff;
          add.rn.f32 sum, sum, tmp;
          setp.eq.u32 lane0, lane, 0;
          and.pred store_sum, lane0, row_live;
          mul.wide.u32 offset, row, $15;
          add.u64 address, $4, offset;
          add.u32 chunk_index, $11, chunk;
          mul.wide.u32 offset, chunk_index, 4;
          add.u64 address, address, offset;
          @store_sum st.global.f32 [address], sum;
          add.u32 wave, wave, 1;
          setp.lt.u32 more, wave, $13;
          @more bra.uni DG_TLE_MHC_SPLIT_COMBINE8_WAVE;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,r,l,r,r,r,r,r,r,r,r,r,r,r",
        args=[
            pre_address,
            residual_address,
            scratch_address,
            square_address,
            physical_thread,
            hidden_quad_offset,
            hidden_quad_count,
            live_rows,
            scratch_stride_bytes,
            scratch_chunk_bytes,
            square_chunk_base,
            row_warps,
            row_waves,
            pre_stride_bytes,
            square_stride_bytes,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _token_shard_normalize_bf16x8_ptx(
    tail_scratch,
    y_base,
    rmsnorm_weight_ptr,
    norm_global,
    physical_thread,
    hidden_quad_offset,
    hidden_quad_count,
    live_rows,
    scratch_stride_bytes,
    scratch_chunk_bytes,
    row_warps: tl.constexpr = 12,
    row_waves: tl.constexpr = 6,
):
    """Normalize 256 values as BF16x8 plus one optional 64-value tail."""
    scratch_address = tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[tle.gpu.local_ptr(tail_scratch, (0, 0, 0))],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    norm_address = tl.cast(norm_global, tl.uint64)
    y_address = tl.cast(y_base, tl.uint64)
    weight_address = tl.cast(rmsnorm_weight_ptr, tl.uint64)
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred row_live, wide, tail_lane, tail_live, more;
          .reg .u32 warp, lane, wave, row, chunk, quad;
          .reg .u32 scratch_row, scratch_tail;
          .reg .b32 packed0, packed1, packed2, packed3;
          .reg .b32 weight0, weight1, weight2, weight3;
          .reg .b32 tail_weight0, tail_weight1;
          .reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;
          .reg .b16 w0, w1, w2, w3, w4, w5, w6, w7;
          .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
          .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
          .reg .f32 tf0, tf1, tf2, tf3, norm;
          .reg .u64 y_row, address, main_offset, tail_offset;
          shr.u32 warp, $5, 5;
          and.b32 lane, $5, 31;
          and.b32 row, warp, 3;
          shr.u32 chunk, warp, 2;

          mad.lo.u32 quad, chunk, 64, $6;
          mad.lo.u32 quad, lane, 2, quad;
          mul.wide.u32 main_offset, quad, 8;
          mov.b32 weight0, 0;
          mov.b32 weight1, 0;
          mov.b32 weight2, 0;
          mov.b32 weight3, 0;
          add.u64 address, $3, main_offset;
          ld.global.v4.b32 {weight0, weight1, weight2, weight3}, [address];
          mov.b32 {w0, w1}, weight0;
          mov.b32 {w2, w3}, weight1;
          mov.b32 {w4, w5}, weight2;
          mov.b32 {w6, w7}, weight3;
          cvt.f32.bf16 f0, w0;
          cvt.f32.bf16 f1, w1;
          cvt.f32.bf16 f2, w2;
          cvt.f32.bf16 f3, w3;
          cvt.f32.bf16 f4, w4;
          cvt.f32.bf16 f5, w5;
          cvt.f32.bf16 f6, w6;
          cvt.f32.bf16 f7, w7;

          setp.gt.u32 wide, $7, 64;
          setp.lt.u32 tail_lane, lane, 16;
          and.pred tail_live, wide, tail_lane;
          mad.lo.u32 quad, chunk, 64, $6;
          add.u32 quad, quad, 64;
          add.u32 quad, quad, lane;
          mul.wide.u32 tail_offset, quad, 8;
          mov.b32 tail_weight0, 0;
          mov.b32 tail_weight1, 0;
          add.u64 address, $3, tail_offset;
          @tail_live ld.global.v2.b32 {tail_weight0, tail_weight1}, [address];
          mov.b32 {w0, w1}, tail_weight0;
          mov.b32 {w2, w3}, tail_weight1;
          cvt.f32.bf16 tf0, w0;
          cvt.f32.bf16 tf1, w1;
          cvt.f32.bf16 tf2, w2;
          cvt.f32.bf16 tf3, w3;

          mov.u32 wave, 0;
        DG_TLE_MHC_SPLIT_NORM8_WAVE:
          setp.lt.u32 row_live, row, $8;
          mov.f32 norm, 0f00000000;
          mul.wide.u32 y_row, row, 4;
          add.u64 y_row, $4, y_row;
          @row_live ld.global.f32 norm, [y_row];
          mad.lo.u32 scratch_row, row, $9, $1;
          add.u32 scratch_row, scratch_row, $10;
          mad.lo.u32 scratch_row, chunk, 512, scratch_row;
          mad.lo.u32 scratch_row, lane, 16, scratch_row;
          mov.b32 packed0, 0;
          mov.b32 packed1, 0;
          mov.b32 packed2, 0;
          mov.b32 packed3, 0;
          @row_live ld.shared.v4.b32 {packed0, packed1, packed2, packed3}, [scratch_row];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          mov.b32 {h4, h5}, packed2;
          mov.b32 {h6, h7}, packed3;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          cvt.f32.bf16 v4, h4;
          cvt.f32.bf16 v5, h5;
          cvt.f32.bf16 v6, h6;
          cvt.f32.bf16 v7, h7;
          mul.f32 v0, v0, norm;
          mul.f32 v1, v1, norm;
          mul.f32 v2, v2, norm;
          mul.f32 v3, v3, norm;
          mul.f32 v4, v4, norm;
          mul.f32 v5, v5, norm;
          mul.f32 v6, v6, norm;
          mul.f32 v7, v7, norm;
          mul.f32 v0, v0, f0;
          mul.f32 v1, v1, f1;
          mul.f32 v2, v2, f2;
          mul.f32 v3, v3, f3;
          mul.f32 v4, v4, f4;
          mul.f32 v5, v5, f5;
          mul.f32 v6, v6, f6;
          mul.f32 v7, v7, f7;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          cvt.rn.bf16x2.f32 packed2, v5, v4;
          cvt.rn.bf16x2.f32 packed3, v7, v6;
          mul.wide.u32 y_row, row, 8192;
          add.u64 y_row, $2, y_row;
          add.u64 address, y_row, main_offset;
          @row_live st.global.v4.b32 [address], {packed0, packed1, packed2, packed3};

          and.pred tail_live, wide, tail_lane;
          and.pred tail_live, tail_live, row_live;
          mad.lo.u32 scratch_tail, row, $9, $1;
          add.u32 scratch_tail, scratch_tail, $10;
          mad.lo.u32 scratch_tail, chunk, 512, scratch_tail;
          mad.lo.u32 scratch_tail, lane, 8, scratch_tail;
          add.u32 scratch_tail, scratch_tail, 512;
          mov.b32 packed0, 0;
          mov.b32 packed1, 0;
          @tail_live ld.shared.v2.b32 {packed0, packed1}, [scratch_tail];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 v0, h0;
          cvt.f32.bf16 v1, h1;
          cvt.f32.bf16 v2, h2;
          cvt.f32.bf16 v3, h3;
          mul.f32 v0, v0, norm;
          mul.f32 v1, v1, norm;
          mul.f32 v2, v2, norm;
          mul.f32 v3, v3, norm;
          mul.f32 v0, v0, tf0;
          mul.f32 v1, v1, tf1;
          mul.f32 v2, v2, tf2;
          mul.f32 v3, v3, tf3;
          cvt.rn.bf16x2.f32 packed0, v1, v0;
          cvt.rn.bf16x2.f32 packed1, v3, v2;
          add.u64 address, y_row, tail_offset;
          @tail_live st.global.v2.b32 [address], {packed0, packed1};

          add.u32 wave, wave, 1;
          setp.lt.u32 more, wave, $12;
          @more bra.uni DG_TLE_MHC_SPLIT_NORM8_WAVE;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,l,l,l,r,r,r,r,r,r,r,r",
        args=[
            scratch_address,
            y_address,
            weight_address,
            norm_address,
            physical_thread,
            hidden_quad_offset,
            hidden_quad_count,
            live_rows,
            scratch_stride_bytes,
            scratch_chunk_bytes,
            row_warps,
            row_waves,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )

"""Shared Normal-MHC routing tail for the eager Hopper P256 kernels.

The admitted P256 kernels need only this routing fragment from the historical
split-tail research kernels.  Keeping it here avoids importing two complete
shape-exact prototypes into the production path.
"""

from __future__ import annotations

import triton
import triton.language as tl

from .. import fused_post_gemm_t64 as _t64

HC_MULT = 4
PROJECTION_OUTPUTS = 24
FLAT_HIDDEN = 4096 * HC_MULT
SINKHORN_ITERS = 20

JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)
JIT_SINKHORN_ITERS = tl.constexpr(SINKHORN_ITERS)


@triton.jit
def _routing_finalize_no_barrier(
    pre_global_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    packed_sum,
    lane,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    """Normal-MHC routing body with its rendezvous owned by the caller."""
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
    pre_address = tl.cast(pre_global_ptr, tl.uint64)
    _ = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred p;
          .reg .u64 offset, address;
          setp.lt.u32 p, $3, 4;
          mul.wide.u32 offset, $3, 4;
          add.u64 address, $1, offset;
          @p st.global.f32 [address], $2;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r",
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

    routing_mask = (lane >= 2 * JIT_HC_MULT) & (lane < JIT_PROJECTION_OUTPUTS)
    sinkhorn = tl.where(routing_mask, mixes, 0.0)
    row_max = _t64._warp_reduce4_max(sinkhorn)
    sinkhorn = tl.exp(sinkhorn - row_max)
    row_sum = _t64._warp_reduce4_sum(sinkhorn)
    sinkhorn *= _t64._fast_rcp_ftz_f32(row_sum)
    sinkhorn += sinkhorn_eps
    column_sum = _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
    sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
    # Keep the loop fully unrolled.  A dynamic loop backedge is lowered to a
    # worker-group rendezvous by the pinned compiler and deadlocks bypassing
    # warps in these role-specialized kernels.
    for _ in tl.static_range(1, JIT_SINKHORN_ITERS):
        row_sum = _t64._warp_reduce4_sum(sinkhorn) + sinkhorn_eps
        sinkhorn *= _t64._fast_rcp_ftz_f32(row_sum)
        column_sum = _t64._warp_reduce4x4_column_sum(sinkhorn) + sinkhorn_eps
        sinkhorn *= _t64._fast_rcp_ftz_f32(column_sum)
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


__all__ = ["_routing_finalize_no_barrier"]

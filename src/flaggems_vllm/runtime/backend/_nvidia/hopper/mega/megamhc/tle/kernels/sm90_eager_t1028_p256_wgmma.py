"""Shared Hopper PTX helpers for the production split16 Graph family.

Formal callers cover ``T in {400, 416, 432, 448, 464, 496, 512}`` at
``H=4096, R=4, P=24`` with split-16, 16 producer warps, and four RS-WGMMA
warps.  The prototype kernel and host preparation path were unreachable and
are omitted; only the three production-used leaf helpers keep this import
path.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _load_global_bf16x4_masked(address, valid):
    """Load one packed BF16 quad, returning zeros for an invalid token."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred p;
          .reg .b32 packed0, packed1;
          .reg .b16 h0, h1, h2, h3;
          mov.b32 packed0, 0;
          mov.b32 packed1, 0;
          setp.ne.u32 p, $5, 0;
          @p ld.global.v2.b32 {packed0, packed1}, [$4];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 $0, h0;
          cvt.f32.bf16 $1, h1;
          cvt.f32.bf16 $2, h2;
          cvt.f32.bf16 $3, h3;
        }
        """,
        constraints="=&f,=&f,=&f,=&f,l,r",
        args=[address, tl.cast(valid, tl.uint32)],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _half_warp_reduce16_sum(value):
    """Reduce FP32 independently in lanes 0..15 and 16..31."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .f32 t;
          mov.f32 $0, $1;
          shfl.sync.bfly.b32 t, $0, 8, 0x100f, 0xffffffff;
          add.rn.f32 $0, $0, t;
          shfl.sync.bfly.b32 t, $0, 4, 0x100f, 0xffffffff;
          add.rn.f32 $0, $0, t;
          shfl.sync.bfly.b32 t, $0, 2, 0x100f, 0xffffffff;
          add.rn.f32 $0, $0, t;
          shfl.sync.bfly.b32 t, $0, 1, 0x100f, 0xffffffff;
          add.rn.f32 $0, $0, t;
        }
        """,
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _accumulate_global_row_square(address, value, half_lane, valid, initialize):
    """Let one stable half-warp leader update its row's global partial."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred lane0, row_valid, leader, init, need_load;
          .reg .f32 current;
          setp.eq.u32 lane0, $3, 0;
          setp.ne.u32 row_valid, $4, 0;
          and.pred leader, lane0, row_valid;
          setp.ne.u32 init, $5, 0;
          not.pred init, init;
          and.pred need_load, leader, init;
          mov.f32 current, 0f00000000;
          @need_load ld.global.f32 current, [$1];
          @leader add.rn.f32 current, current, $2;
          @leader st.global.f32 [$1], current;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,f,r,r,r",
        args=[
            address,
            value,
            half_lane,
            tl.cast(valid, tl.uint32),
            tl.cast(initialize, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


__all__ = [
    "_accumulate_global_row_square",
    "_half_warp_reduce16_sum",
    "_load_global_bf16x4_masked",
]

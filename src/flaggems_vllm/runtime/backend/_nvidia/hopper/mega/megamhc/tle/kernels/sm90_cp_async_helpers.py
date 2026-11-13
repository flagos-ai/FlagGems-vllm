"""Classic ``cp.async`` helpers for the production large-Graph kernels.

Formal callers cover ``T=272..512`` at ``H=4096, R=4, P=24`` and use these
leaf helpers for K64 Fn staging in the 16-producer/4-RS-WGMMA topology.  This
module has no independently dispatchable specialization or tuning switch.
"""

from __future__ import annotations

import triton
import triton.language as tl

from .sm90_shared_memory_helpers import _shared_address


@triton.jit
def _cp_async_cg_16b_zero_fill(shared_address, global_address, src_size):
    """Issue one 16-byte global-to-shared copy with hardware zero fill."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          cp.async.cg.shared.global [$1], [$2], 16, $3;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,l,r",
        args=[shared_address, global_address, src_size],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


__all__ = ["_cp_async_cg_16b_zero_fill", "_shared_address"]

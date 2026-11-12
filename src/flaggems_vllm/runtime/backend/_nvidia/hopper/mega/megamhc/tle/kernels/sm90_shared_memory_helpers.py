"""Shared-address helpers for production large-Graph MegaMHC kernels.

Formal callers are the ``T=272..512`` K64/cp.async Graph families at
``H=4096, R=4, P=24``.  The helper only exposes a shared pointer as its PTX
address and has no dispatchable shape or experimental configuration.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _shared_address(pointer):
    """Expose a TLE shared pointer as the 32-bit PTX shared address."""
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


__all__ = ["_shared_address"]

"""Shared-address helper used by the production Hopper MegaMHC kernels."""

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

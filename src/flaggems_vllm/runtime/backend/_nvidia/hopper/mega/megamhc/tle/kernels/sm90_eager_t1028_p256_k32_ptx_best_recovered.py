"""Shared routing/hidden helpers for the production eager P384 family.

Formal callers are ``T in {12991, 13005, 13021, 16384}`` at
``H=4096, R=4, P=24`` using the manual512, 12-producer/4-math P384 topology.
The original bring-up kernel, plan, and workspace factory were not reachable
from production dispatch and are omitted; only numerically validated leaf
helpers remain under the compatibility import path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sinkhorn_nineteen_warp_iterations(value, epsilon):
    """Run the repeated 4x4 Sinkhorn body as a warp-local PTX loop."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred again;
          .reg .u32 iteration;
          .reg .f32 x, t, total, inverse;
          mov.f32 x, $1;
          mov.u32 iteration, 19;
        DG_TLE_MHC_SINKHORN_LOOP:
          mov.f32 total, x;
          shfl.sync.bfly.b32 t, total, 2, 31, 0xffffffff;
          add.rn.f32 total, total, t;
          shfl.sync.bfly.b32 t, total, 1, 31, 0xffffffff;
          add.rn.f32 total, total, t;
          add.rn.f32 total, total, $2;
          rcp.approx.ftz.f32 inverse, total;
          mul.rn.f32 x, x, inverse;
          mov.f32 total, x;
          shfl.sync.bfly.b32 t, total, 4, 31, 0xffffffff;
          add.rn.f32 total, total, t;
          shfl.sync.bfly.b32 t, total, 24, 31, 0xffffffff;
          add.rn.f32 total, total, t;
          add.rn.f32 total, total, $2;
          rcp.approx.ftz.f32 inverse, total;
          mul.rn.f32 x, x, inverse;
          add.u32 iteration, iteration, 0xffffffff;
          setp.ne.u32 again, iteration, 0;
          @again bra.uni DG_TLE_MHC_SINKHORN_LOOP;
          mov.f32 $0, x;
        }
        """,
        constraints="=f,f,f",
        args=[value, epsilon],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
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


__all__ = [
    "_require_tensor",
    "_sinkhorn_nineteen_warp_iterations",
]

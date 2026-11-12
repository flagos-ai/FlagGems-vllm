"""Shared Hopper helpers retained from the recovered P256 prototype.

The original module also contained an opt-in research kernel, plan, and
workspace factory that were never reachable from production dispatch.  P384
still imports these numerically validated routing and hidden-tail helpers, so
the compatibility module name remains stable while the dead prototype is
omitted.
"""

from __future__ import annotations

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

HIDDEN = 4096
HC_MULT = 4
FLAT_HIDDEN = HIDDEN * HC_MULT

JIT_HIDDEN = tl.constexpr(HIDDEN)
JIT_HC_MULT = tl.constexpr(HC_MULT)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)


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


@triton.jit
def _hidden_tail_no_wait(
    pre_shared,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Finish one H4096 row after its Pre values are already visible."""
    h = tl.arange(0, JIT_HIDDEN)
    norm_input = tl.zeros((JIT_HIDDEN,), tl.float32)
    for channel in tl.static_range(0, JIT_HC_MULT):
        value = tl.load(new_residual_ptr + channel * JIT_HIDDEN + h).to(tl.float32)
        pre_scalar = tl.load(tle.gpu.local_ptr(pre_shared, (channel,)))
        norm_input += value * pre_scalar
    norm_input_bf16 = norm_input.to(tl.bfloat16, fp_downcast_rounding="rtne")
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=0) / JIT_HIDDEN + rmsnorm_eps
    )
    rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + h,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
    )


@triton.jit
def _hidden_tail_batch2_no_wait(
    pre_shared0,
    pre_shared1,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    live_rows,
    rmsnorm_eps,
    rmsnorm_scale,
):
    """Map the default eight warps to two independent H4096 rows."""
    batch = tl.arange(0, 2)[:, None]
    h = tl.arange(0, JIT_HIDDEN)[None, :]
    live_mask = batch < live_rows
    norm_input = tl.zeros((2, JIT_HIDDEN), tl.float32)
    for channel in tl.static_range(0, JIT_HC_MULT):
        value = tl.load(
            new_residual_ptr + batch * JIT_FLAT_HIDDEN + channel * JIT_HIDDEN + h,
            mask=live_mask,
            other=0.0,
        ).to(tl.float32)
        pre0 = tl.load(tle.gpu.local_ptr(pre_shared0, (channel,)))
        pre1 = tl.load(tle.gpu.local_ptr(pre_shared1, (channel,)))
        pre = tl.where(batch == 0, pre0, pre1)
        norm_input += value * pre
    norm_input_bf16 = norm_input.to(tl.bfloat16, fp_downcast_rounding="rtne")
    norm_input_fp32 = norm_input_bf16.to(tl.float32)
    rms = tl.rsqrt(
        tl.sum(norm_input_fp32 * norm_input_fp32, axis=1) / JIT_HIDDEN + rmsnorm_eps
    )
    rms_weight = tl.load(rmsnorm_weight_ptr + h).to(tl.float32)
    y = norm_input_fp32 * rms[:, None] * rms_weight * rmsnorm_scale
    tl.store(
        y_bf16_ptr + batch * JIT_HIDDEN + h,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=live_mask,
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
    "_hidden_tail_batch2_no_wait",
    "_hidden_tail_no_wait",
    "_require_tensor",
    "_sinkhorn_nineteen_warp_iterations",
]

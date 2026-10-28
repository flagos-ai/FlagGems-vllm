# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Inference-only BF16 RMSNorm plus native-CUDA-compatible dynamic INT8.

Returns the original BF16 normalized value as well as its INT8 representation.
No registration, serving policy, training, or GEMM is part of this operator.
"""
import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit(
    do_not_specialize=[
        "eps",
        "y_stride_r",
        "q_stride_r",
        "scale_stride_r",
        "x_stride_r",
    ]
)
def _rms_scaled_int8_quant_kernel(
    y_ptr,
    q_ptr,
    scale_ptr,
    x_ptr,
    weight_ptr,
    y_stride_r,
    q_stride_r,
    scale_stride_r,
    x_stride_r,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # M is only the launch grid, never a JIT argument. Row strides are runtime
    # inputs excluded from LibEntry exact-value keys; N/BLOCK_SIZE retain the
    # original norm width/tile specialization (real q_a has one width: 2048).
    # Same row, reduction tree, FP32 norm, and pre-weight BF16 cast as the
    # selected FlagGems rms_norm_kernel. Keep its default FP-fusion policy.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * x_stride_r + cols, mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    x_normed = (x * rrms).to(x_ptr.dtype.element_ty)
    # This SECOND BF16 cast is the original RMSNorm output-store boundary.
    # Quantization must read this value, never the unrounded FP32 product.
    normalized = (x_normed * w).to(tl.bfloat16)
    tl.store(y_ptr + row * y_stride_r + cols, normalized, mask=mask)
    z = normalized.to(tl.float32)
    # Reuse the already-qualified mul_scaled_int8_quant native-CUDA quant
    # semantics: fmax ignores NaN, scale/inverse use div.rn, zero rows use 0.
    absmax = tl.max(tl.where(mask & (z == z), tl.abs(z), 0.0), 0)
    scale = tl.div_rn(absmax, 127.0)
    inverse = tl.where(absmax == 0.0, 0.0, tl.div_rn(127.0, absmax))
    tl.store(scale_ptr + row * scale_stride_r, scale)
    q32 = tl.inline_asm_elementwise(
        "cvt.rni.sat.s8.f32 $0, $1;",
        constraints="=r,f",
        args=[z * inverse],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(q_ptr + row * q_stride_r + cols, q32.to(tl.int8), mask=mask)


def supports_rms_scaled_int8_quant(x, weight):
    """Metadata-only support; last-axis contiguous, non-overlapping rows."""
    return (
        isinstance(x, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and x.layout == weight.layout == torch.strided
        and x.ndim == 2
        and weight.ndim == 1
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.device == x.device
        and torch.version.cuda is not None
        and torch.version.hip is None
        and 0 < x.shape[1] <= 4096
        and weight.shape == (x.shape[1],)
        and x.stride(1) == 1
        and x.stride(0) >= x.shape[1]
        and weight.stride(0) == 1
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and not x.is_conj()
        and not weight.is_conj()
        and not x.is_neg()
        and not weight.is_neg()
    )


def _validate_output(value, shape, dtype, device):
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.shape != shape
        or value.dtype != dtype
        or value.device != device
        or value.stride(1) != 1
        or value.stride(0) < shape[1]
        or value.requires_grad
        or value.is_conj()
        or value.is_neg()
    ):
        raise NotImplementedError("unsupported output dtype/layout/device")


def rms_scaled_int8_quant(
    x, weight, eps=1e-5, *, norm_out=None, out=None, scale_out=None
):
    """Return (BF16 norm, INT8 norm, F32 [M,1] scale), with separate storage.

    Must run with grad disabled (native norm Parameters can retain requires_grad).
    Padding and storage offsets are supported; overlap, training and alternate
    dtypes decline. A caller must retain the original path when unsupported.
    """
    if not supports_rms_scaled_int8_quant(x, weight):
        raise NotImplementedError("BF16 2D row-major CUDA inference inputs required")
    m, n = x.shape
    if norm_out is None:
        norm_out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if out is None:
        out = torch.empty((m, n), dtype=torch.int8, device=x.device)
    if scale_out is None:
        scale_out = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    _validate_output(norm_out, (m, n), torch.bfloat16, x.device)
    _validate_output(out, (m, n), torch.int8, x.device)
    _validate_output(scale_out, (m, 1), torch.float32, x.device)
    outputs = [norm_out, out, scale_out]
    for i, value in enumerate(outputs):
        if any(torch._C._is_alias_of(value, other) for other in (x, weight)):
            raise NotImplementedError("outputs may not alias inputs")
        if any(torch._C._is_alias_of(value, other) for other in outputs[:i]):
            raise NotImplementedError("outputs must have disjoint storage")
    if m:
        with runtime.torch_device_fn.device(x.device):
            _rms_scaled_int8_quant_kernel[(m,)](
                norm_out,
                out,
                scale_out,
                x,
                weight,
                norm_out.stride(0),
                out.stride(0),
                scale_out.stride(0),
                x.stride(0),
                n,
                eps,
                BLOCK_SIZE=triton.next_power_of_2(n),
                num_warps=4,
                num_stages=1,
            )
    return norm_out, out, scale_out


__all__ = ["supports_rms_scaled_int8_quant", "rms_scaled_int8_quant"]

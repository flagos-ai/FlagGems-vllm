# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 product boundary followed by native CUDA symmetric INT8 quantization."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


def _semantic_configs():
    # Keep the existing dynamic-row tuning family, with one initial semantic
    # configuration. Wider tuning is deliberately not part of this candidate.
    configs = [
        c
        for c in runtime.get_tuned_config("scaled_int8_quant_dynamic_single")
        if c.num_warps == 4 and c.num_stages == 1
    ]
    if len(configs) != 1:
        raise NotImplementedError("existing NVIDIA dynamic-row config changed")
    return configs


@libentry()
@libtuner(
    configs=_semantic_configs(),
    key=["num_tokens", "hidden_size"],
    strategy=["align32", "default"],
)
@triton.jit
def _mul_scaled_int8_quant_kernel(
    a_ptr,
    b_ptr,
    q_ptr,
    scale_ptr,
    product_ptr,
    hidden_size,
    num_tokens,
    stride_a: tl.constexpr,
    stride_b: tl.constexpr,
    stride_q: tl.constexpr,
    stride_scale: tl.constexpr,
    stride_product: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    WRITE_PRODUCT: tl.constexpr,
    PLACEHOLDER_UNUSED: tl.constexpr = 1,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, ROW_BLOCK)
    valid = col < hidden_size
    a = tl.load(a_ptr + row * stride_a + col, valid, other=0).to(tl.float32)
    b = tl.load(b_ptr + row * stride_b + col, valid, other=0).to(tl.float32)
    # Original FlagGems mul stores BF16 before native quant reads FP32.
    product = (a * b).to(tl.bfloat16)
    z = product.to(tl.float32)
    if WRITE_PRODUCT:
        tl.store(product_ptr + row * stride_product + col, product, valid)
    # The native CUDA row reduction starts at +0 and uses fmaxf, which ignores
    # NaNs. All-NaN rows consequently have absmax +0, not NaN.
    absmax = tl.max(tl.where(valid & (z == z), tl.abs(z), 0.0), 0)
    scale = tl.div_rn(absmax, 127.0)
    inverse = tl.where(absmax == 0.0, 0.0, tl.div_rn(127.0, absmax))
    tl.store(scale_ptr + row * stride_scale, scale)
    # Use precisely the native NVIDIA saturating nearest-even conversion,
    # including its NaN/Inf behavior; no FP32->INT32 approximation is inserted.
    q32 = tl.inline_asm_elementwise(
        "cvt.rni.sat.s8.f32 $0, $1;",
        constraints="=r,f",
        args=[z * inverse],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(q_ptr + row * stride_q + col, q32.to(tl.int8), valid)


def supports_mul_scaled_int8_quant(a, b):
    """Metadata-only eligibility; no device query, cast, copy or computation."""
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.layout == b.layout == torch.strided
        and a.ndim == b.ndim == 2
        and a.shape == b.shape
        and a.dtype == b.dtype == torch.bfloat16
        and a.is_cuda
        and b.device == a.device
        and torch.version.hip is None
        and torch.version.cuda is not None
        and 0 < a.shape[1] <= 16384
        and a.stride(1) == b.stride(1) == 1
        and a.stride(0) >= a.shape[1]
        and b.stride(0) >= b.shape[1]
        and not a.requires_grad
        and not b.requires_grad
        and not a.is_conj()
        and not b.is_conj()
        and not a.is_neg()
        and not b.is_neg()
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


def mul_scaled_int8_quant(a, b, *, out=None, scale_out=None, product_out=None):
    """Return INT8 product and FP32 [M,1] native symmetric per-token scales.

    Optional outputs are disjoint row-major views. ``product_out`` is a
    diagnostic BF16 materialization; serving does not request or allocate it.
    Unsupported inputs raise instead of using a Torch compute fallback.
    """
    if not supports_mul_scaled_int8_quant(a, b):
        raise NotImplementedError("equal 2D row-major BF16 CUDA inputs required")
    m, n = a.shape
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.int8)
    if scale_out is None:
        scale_out = torch.empty((m, 1), device=a.device, dtype=torch.float32)
    _validate_output(out, (m, n), torch.int8, a.device)
    _validate_output(scale_out, (m, 1), torch.float32, a.device)
    outputs = [out, scale_out]
    if product_out is not None:
        _validate_output(product_out, (m, n), torch.bfloat16, a.device)
        outputs.append(product_out)
    for i, value in enumerate(outputs):
        if any(torch._C._is_alias_of(value, other) for other in (a, b)):
            raise NotImplementedError("outputs may not alias inputs")
        if any(torch._C._is_alias_of(value, other) for other in outputs[:i]):
            raise NotImplementedError("outputs must have disjoint storage")
    if m:
        with runtime.torch_device_fn.device(a.device):
            _mul_scaled_int8_quant_kernel[(m,)](
                a,
                b,
                out,
                scale_out,
                a if product_out is None else product_out,
                n,
                m,
                a.stride(0),
                b.stride(0),
                out.stride(0),
                scale_out.stride(0),
                0 if product_out is None else product_out.stride(0),
                ROW_BLOCK=triton.next_power_of_2(n),
                WRITE_PRODUCT=product_out is not None,
                enable_fp_fusion=False,
            )
    return out, scale_out


__all__ = ["mul_scaled_int8_quant", "supports_mul_scaled_int8_quant"]

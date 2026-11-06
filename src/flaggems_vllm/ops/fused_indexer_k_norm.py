# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""FP32 LayerNorm and cast-back for the GLM indexer key projection."""

import math
import numbers
import struct

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_MAX_DIM = 4096
_MAX_OFFSET = (1 << 63) - 1
_SUPPORTED_VENDORS = frozenset(("nvidia", "ascend", "thead", "metax", "iluvatar"))


@triton.jit
def _rsqrt_preserving_subnormals(value):
    # Some backends lower rsqrt to an instruction that flushes subnormal
    # inputs to zero. An exact power-of-two scale makes every positive FP32
    # subnormal normal before rsqrt, then restores the reciprocal-root scale.
    is_subnormal = (value > 0.0) & (value < 1.1754943508222875e-38)
    scaled = tl.where(is_subnormal, value * 16777216.0, value)
    result = tl.rsqrt(scaled)
    return tl.where(is_subnormal, result * 4096.0, result)


@triton.jit(do_not_specialize=["EPS"])
def _fused_indexer_k_norm_kernel(
    X,
    W,
    B,
    Y,
    M,
    D: tl.constexpr,
    STRIDE_XM,
    STRIDE_XD,
    STRIDE_W,
    STRIDE_B,
    EPS,
    BLOCK_M: tl.constexpr,
):
    # This is a semantic extent, not a tuning parameter. Keeping it inside
    # the JIT also prevents FlagTree AABS from adding host metadata to configs.
    BLOCK_D: tl.constexpr = triton.next_power_of_2(D)
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    valid = (rows[:, None] < M) & (cols[None, :] < D)
    values = tl.load(
        X + rows[:, None] * STRIDE_XM + cols[None, :] * STRIDE_XD,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    # Divide before summing so a finite mean does not overflow the FP32 sum.
    mean = tl.sum(values / D, axis=1)
    # Padding must also be zero after centering, not merely before the load.
    centered = tl.where(cols[None, :] < D, values - mean[:, None], 0.0)
    variance = tl.sum(centered * centered, axis=1) / D
    rstd = _rsqrt_preserving_subnormals(variance + EPS)

    weight = tl.load(W + cols * STRIDE_W, mask=cols < D, other=0.0)
    bias = tl.load(B + cols * STRIDE_B, mask=cols < D, other=0.0)
    result = (values - mean[:, None]) * rstd[:, None]
    result = result * weight[None, :] + bias[None, :]
    tl.store(Y + rows[:, None] * D + cols[None, :], result, mask=valid)


def _prune_norm_configs(configs, named_args, **kwargs):
    dim = named_args["D"]
    if dim > 256:
        return [config for config in configs if config.kwargs["BLOCK_M"] == 1]
    return configs


if runtime.device.vendor_name == "nvidia":
    _norm_kernel = libentry()(
        libtuner(
            configs=runtime.get_tuned_config("fused_indexer_k_norm"),
            key=["M", "D", "STRIDE_XM", "STRIDE_XD", "STRIDE_W", "STRIDE_B"],
            prune_configs_by={"early_config_prune": _prune_norm_configs},
            use_cuda_graph=False,
        )(_fused_indexer_k_norm_kernel)
    )
else:
    # A single conservative launch reuses exactly the same algorithm. Vendor
    # compilers must validate it before expanding their tuning candidates.
    _norm_kernel = libentry()(_fused_indexer_k_norm_kernel)


def _validate_inputs(x, weight, bias, dim, eps):
    for name, tensor in (("x", x), ("weight", weight), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"fused_indexer_k_norm {name} must be a Tensor")
        if tensor.layout != torch.strided:
            raise NotImplementedError("fused_indexer_k_norm requires strided tensors")

    if x.ndim != 2:
        raise NotImplementedError("fused_indexer_k_norm requires rank-2 x")
    if isinstance(dim, bool) or not isinstance(dim, numbers.Integral):
        raise TypeError("fused_indexer_k_norm dim must be an integer")
    if not 1 <= dim <= _MAX_DIM:
        raise NotImplementedError("fused_indexer_k_norm supports dimensions 1..4096")
    if x.shape[1] != dim:
        raise ValueError("fused_indexer_k_norm dim must match x.shape[1]")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("fused_indexer_k_norm requires FP16, BF16, or FP32 x")
    for name, parameter in (("weight", weight), ("bias", bias)):
        if parameter.shape != (dim,):
            raise ValueError(f"fused_indexer_k_norm {name} must have shape (dim,)")
        if parameter.dtype != torch.float32:
            raise NotImplementedError(f"fused_indexer_k_norm {name} must be FP32")
        if parameter.device != x.device:
            raise ValueError("fused_indexer_k_norm tensors must share a device")

    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise TypeError("fused_indexer_k_norm eps must be a real scalar")
    try:
        eps = struct.unpack("f", struct.pack("f", float(eps)))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(
            "fused_indexer_k_norm eps must be finite positive FP32"
        ) from exc
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("fused_indexer_k_norm eps must be finite positive FP32")

    for tensor in (x, weight, bias):
        if any(stride < 0 for stride in tensor.stride()):
            raise NotImplementedError(
                "fused_indexer_k_norm does not support negative strides"
            )
        if (
            tensor.numel()
            and sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride())
            )
            > _MAX_OFFSET
        ):
            raise NotImplementedError(
                "fused_indexer_k_norm tensor offsets exceed int64"
            )
    if x.numel() > _MAX_OFFSET:
        raise NotImplementedError("fused_indexer_k_norm output size exceeds int64")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, bias)):
        raise NotImplementedError("fused_indexer_k_norm supports inference only")
    if (
        runtime.device.vendor_name not in _SUPPORTED_VENDORS
        or x.device.type in ("cpu", "meta")
        or x.device.type != runtime.device.name
    ):
        raise NotImplementedError(
            "fused_indexer_k_norm requires a supported accelerator"
        )
    return eps


def _launch_fused_indexer_k_norm(x, weight, bias, out, eps):
    rows, dim = x.shape
    args = (
        x,
        weight,
        bias,
        out,
        rows,
        dim,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        bias.stride(0),
        eps,
    )
    with runtime.torch_device_fn.device(x.device):
        if runtime.device.vendor_name == "nvidia":
            _norm_kernel[lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)](*args)
        else:
            _norm_kernel[(rows,)](
                *args,
                BLOCK_M=1,
                num_warps=4,
                num_stages=1,
            )


def fused_indexer_k_norm(x, weight, bias, dim, eps):
    """Normalize indexer keys in FP32 and return a fresh cast-back tensor.

    Supports inference on rank-2 FP16/BF16/FP32 input with width 1..4096,
    FP32 weight/bias of shape ``(dim,)``, and nonnegative read strides.
    ``eps`` must remain finite and positive after conversion to FP32.
    """
    eps = _validate_inputs(x, weight, bias, dim, eps)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.shape[0]:
        _launch_fused_indexer_k_norm(x, weight, bias, out, eps)
    return out

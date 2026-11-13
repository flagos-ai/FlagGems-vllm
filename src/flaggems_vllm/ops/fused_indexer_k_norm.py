# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""FP32 LayerNorm and cast-back for the GLM indexer key projection."""

import math
import numbers
import struct
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_MAX_DIM = 4096
_MAX_OFFSET = (1 << 63) - 1
_H20Z_D128_ROW_STRIDES = frozenset((128, 144, 192))
_H20Z_D128_MAX_ROWS = 8192
_H20Z_D128_TRITON_VERSION = "3.6.0"
_IS_NVIDIA = runtime.device.vendor_name == "nvidia"
_SUPPORTED_VENDORS = frozenset(
    ("nvidia", "ascend", "thead", "metax", "iluvatar", "mthreads", "hygon")
)


@triton.jit
def _rsqrt_preserving_subnormals(value, SUBNORMAL_METHOD: tl.constexpr = 1):
    # Some backends lower rsqrt to an instruction that flushes subnormal
    # inputs to zero. An exact power-of-two scale makes every positive FP32
    # subnormal normal before rsqrt, then restores the reciprocal-root scale.
    # Two lowering-safe variants are kept in the same kernel body; Triton
    # traces only the active branch because SUBNORMAL_METHOD is constexpr.
    if SUBNORMAL_METHOD == 1:
        # bitcast path: safe for backends where llvm.is.fpclass is missing.
        bits = value.to(tl.uint32, bitcast=True)
        is_subnormal = (bits > 0) & (bits < 0x00800000)
    else:
        # interval path: safe for backends where uint32 bitcast lowering fails.
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
    SUBNORMAL_METHOD: tl.constexpr = 1,
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
    rstd = _rsqrt_preserving_subnormals(variance + EPS, SUBNORMAL_METHOD)

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


def _h20z_d128_config(rows):
    # NOTE: block_m thresholds must stay in sync with
    # test_h20z_d128_config_thresholds.
    if rows == 1:
        return 1, 4
    if rows >= 4096:
        return 8, 4
    return 4, 4


@lru_cache(maxsize=None)
def _is_h20z(device_index=None):
    if not _IS_NVIDIA:
        return False
    try:
        name = runtime.torch_device_fn.get_device_name(device_index)
    except (AttributeError, RuntimeError):
        return False
    return "H20Z" in name.replace(" ", "_").upper().split("_")


def _use_h20z_d128_fast_path(
    rows, dim, stride_xm, stride_xd, stride_w, stride_b, device_index=None
):
    return (
        0 < rows <= _H20Z_D128_MAX_ROWS
        and dim == 128
        and stride_xm in _H20Z_D128_ROW_STRIDES
        and stride_xd == 1
        and stride_w == 1
        and stride_b == 1
        and triton.__version__ == _H20Z_D128_TRITON_VERSION
        and _is_h20z(device_index)
    )


if _IS_NVIDIA:
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


def _tensor_signature(tensor):
    return (
        tensor.shape,
        tensor.stride(),
        str(tensor.dtype),
        str(tensor.device),
        tensor.requires_grad,
        str(tensor.layout),
    )


@lru_cache(maxsize=256)
def _validate_inputs_cached(x_sig, w_sig, b_sig, dim, eps):
    eps = _validate_inputs_full(x_sig, w_sig, b_sig, dim, eps)
    return eps


def _validate_inputs_full(x_sig, w_sig, b_sig, dim, eps):
    if (
        x_sig[5] != "torch.strided"
        or w_sig[5] != "torch.strided"
        or b_sig[5] != "torch.strided"
    ):
        raise NotImplementedError("fused_indexer_k_norm requires strided tensors")

    if len(x_sig[0]) != 2:
        raise NotImplementedError("fused_indexer_k_norm requires rank-2 x")
    if isinstance(dim, bool) or not isinstance(dim, numbers.Integral):
        raise TypeError("fused_indexer_k_norm dim must be an integer")
    if not 1 <= dim <= _MAX_DIM:
        raise NotImplementedError("fused_indexer_k_norm supports dimensions 1..4096")
    if x_sig[0][1] != dim:
        raise ValueError("fused_indexer_k_norm dim must match x.shape[1]")
    if x_sig[2] not in ("torch.float16", "torch.bfloat16", "torch.float32"):
        raise NotImplementedError("fused_indexer_k_norm requires FP16, BF16, or FP32 x")
    for name, sig in (("weight", w_sig), ("bias", b_sig)):
        if sig[0] != (dim,):
            raise ValueError(f"fused_indexer_k_norm {name} must have shape (dim,)")
        if sig[2] != "torch.float32":
            raise NotImplementedError(f"fused_indexer_k_norm {name} must be FP32")
        if sig[3] != x_sig[3]:
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

    for sig in (x_sig, w_sig, b_sig):
        if any(stride < 0 for stride in sig[1]):
            raise NotImplementedError(
                "fused_indexer_k_norm does not support negative strides"
            )
        if (
            sig[0]
            and sum((size - 1) * stride for size, stride in zip(sig[0], sig[1]))
            > _MAX_OFFSET
        ):
            raise NotImplementedError(
                "fused_indexer_k_norm tensor offsets exceed int64"
            )
    if x_sig[0] and x_sig[0][0] * x_sig[0][1] > _MAX_OFFSET:
        raise NotImplementedError("fused_indexer_k_norm output size exceeds int64")
    if torch.is_grad_enabled() and (x_sig[4] or w_sig[4] or b_sig[4]):
        raise NotImplementedError("fused_indexer_k_norm supports inference only")
    if (
        runtime.device.vendor_name not in _SUPPORTED_VENDORS
        or x_sig[3].startswith("cpu")
        or x_sig[3].startswith("meta")
    ):
        raise NotImplementedError(
            "fused_indexer_k_norm requires a supported accelerator"
        )
    return eps


def _validate_inputs(x, weight, bias, dim, eps):
    for name, tensor in (("x", x), ("weight", weight), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"fused_indexer_k_norm {name} must be a Tensor")
    x_sig = _tensor_signature(x)
    w_sig = _tensor_signature(weight)
    b_sig = _tensor_signature(bias)
    eps = _validate_inputs_cached(x_sig, w_sig, b_sig, dim, eps)
    if torch.is_grad_enabled() and (x_sig[4] or w_sig[4] or b_sig[4]):
        raise NotImplementedError("fused_indexer_k_norm supports inference only")
    return eps


def _launch_fused_indexer_k_norm(x, weight, bias, out, eps):
    rows, dim = x.shape
    stride_xm, stride_xd = x.stride()
    stride_w = weight.stride(0)
    stride_b = bias.stride(0)
    args = (
        x,
        weight,
        bias,
        out,
        rows,
        dim,
        stride_xm,
        stride_xd,
        stride_w,
        stride_b,
        eps,
    )
    with runtime.torch_device_fn.device(x.device):
        if _IS_NVIDIA:
            if _use_h20z_d128_fast_path(
                rows,
                dim,
                stride_xm,
                stride_xd,
                stride_w,
                stride_b,
                x.device.index,
            ):
                block_m, num_warps = _h20z_d128_config(rows)
                _fused_indexer_k_norm_kernel[(triton.cdiv(rows, block_m),)](
                    *args,
                    BLOCK_M=block_m,
                    num_warps=num_warps,
                    num_stages=1,
                )
            else:
                grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)
                _norm_kernel[grid](*args)
        else:
            # Ascend backend lowers uint32 bitcast to an unsupported vcast;
            # switch to the interval-based subnormal classification path.
            if runtime.device.vendor_name == "ascend":
                # Dedicated NPU kernel lives in backend-specific module so
                # the shared operator file stays vendor-agnostic.
                from flaggems_vllm.runtime.backend._ascend.ops.fused_indexer_k_norm import (
                    _launch_ascend,
                )

                _launch_ascend(x, weight, bias, out, eps)
            elif runtime.device.vendor_name == "mthreads":
                from flaggems_vllm.runtime.backend._mthreads.ops.fused_indexer_k_norm import (
                    _launch_musa,
                )

                _launch_musa(x, weight, bias, out, eps)
            elif runtime.device.vendor_name == "metax":
                # MetaX inline path is handled in fused_indexer_k_norm() to
                # avoid _launch_fused_indexer_k_norm dispatcher overhead.
                _norm_kernel[(rows,)](
                    *args,
                    BLOCK_M=1,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                # Generic thin launcher for remaining backends (PPU, etc.)
                # on D=128 aligned fast path.
                if dim == 128 and stride_xd == 1 and stride_w == 1 and stride_b == 1:
                    _fused_indexer_k_norm_kernel[(rows,)](
                        *args,
                        BLOCK_M=1,
                        num_warps=4,
                        num_stages=1,
                    )
                else:
                    _norm_kernel[(rows,)](
                        *args,
                        BLOCK_M=1,
                        num_warps=4,
                        num_stages=1,
                    )


def fused_indexer_k_norm(x, weight, bias, dim, eps, out=None):
    """Normalize indexer keys in FP32 and return a fresh cast-back tensor.

    Supports inference on rank-2 FP16/BF16/FP32 input with width 1..4096,
    FP32 weight/bias of shape ``(dim,)``, and nonnegative read strides.
    ``eps`` must remain finite and positive after conversion to FP32.

    If ``out`` is provided it must be a strided tensor with the same shape,
    dtype and device as ``x``; otherwise a fresh output is allocated.
    """
    eps = _validate_inputs(x, weight, bias, dim, eps)
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    elif out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError(
            "fused_indexer_k_norm out must match x in shape, dtype and device"
        )
    if x.shape[0]:
        if runtime.device.vendor_name == "ascend":
            # Inline fast path: avoid _launch_fused_indexer_k_norm dispatcher
            # overhead on NPU where host launch cost is significant.
            from flaggems_vllm.runtime.backend._ascend.ops.fused_indexer_k_norm import (
                _launch_ascend,
            )

            _launch_ascend(x, weight, bias, out, eps)
        elif runtime.device.vendor_name == "metax":
            # Inline fast path to avoid _launch_fused_indexer_k_norm overhead.
            from flaggems_vllm.runtime.backend._metax.ops.fused_indexer_k_norm import (
                _launch_metax_fast,
            )

            _launch_metax_fast(x, weight, bias, out, eps)
        elif _IS_NVIDIA and _use_h20z_d128_fast_path(
            x.shape[0],
            dim,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            bias.stride(0),
            x.device.index,
        ):
            # Inline fast path for H20Z to avoid _launch_fused_indexer_k_norm
            # dispatcher overhead (~0.004-0.006ms per call).
            rows, _ = x.shape
            stride_xm, stride_xd = x.stride()
            stride_w = weight.stride(0)
            stride_b = bias.stride(0)
            args = (
                x,
                weight,
                bias,
                out,
                rows,
                dim,
                stride_xm,
                stride_xd,
                stride_w,
                stride_b,
                eps,
            )
            block_m, num_warps = _h20z_d128_config(rows)
            with runtime.torch_device_fn.device(x.device):
                _fused_indexer_k_norm_kernel[(triton.cdiv(rows, block_m),)](
                    *args,
                    BLOCK_M=block_m,
                    num_warps=num_warps,
                    num_stages=1,
                )
        else:
            _launch_fused_indexer_k_norm(x, weight, bias, out, eps)
    return out

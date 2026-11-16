# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend launch selection with a retained compatibility kernel."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import triton
import triton.language as tl

from flaggems_vllm import runtime

try:
    _FLAGTREE_VERSION = version("flagtree")
except PackageNotFoundError:
    _FLAGTREE_VERSION = None


@lru_cache(maxsize=None)
def _shared_tile_validated_device(device_index):
    if triton.__version__ != "3.5.1" or _FLAGTREE_VERSION != "0.7.0+ascend3.5":
        return False
    try:
        props = runtime.torch_device_fn.get_device_properties(device_index)
        return props.name in {"Ascend910_9392", "Ascend910B1"}
    except (AttributeError, RuntimeError):
        return False


def _shared_tile_block_m(rows):
    if rows <= 8:
        return 4
    if rows <= 128:
        return 16
    if rows <= 1024:
        return 32
    if rows < 4096:
        return 64
    return 128


# ------------------------------------------------------------------
# Kernel
# ------------------------------------------------------------------
@triton.jit(do_not_specialize=["EPS"])
def _ascend_fused_indexer_k_norm_kernel(
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
    ROWS_PER_BLOCK: tl.constexpr,
    NO_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,  # dummy; NPU backend requires this parameter
):
    BLOCK_D: tl.constexpr = triton.next_power_of_2(D)
    row_start = tl.program_id(0).to(tl.int64) * ROWS_PER_BLOCK
    cols = tl.arange(0, BLOCK_D).to(tl.int64)

    if NO_MASK:
        weight = tl.load(W + cols * STRIDE_W).to(tl.float32)
        bias = tl.load(B + cols * STRIDE_B).to(tl.float32)
    else:
        weight = tl.load(W + cols * STRIDE_W, mask=cols < D, other=0.0)
        bias = tl.load(B + cols * STRIDE_B, mask=cols < D, other=0.0)

    for r in range(ROWS_PER_BLOCK):
        row = row_start + r
        if row < M:
            if NO_MASK:
                values = tl.load(X + row * STRIDE_XM + cols * STRIDE_XD).to(tl.float32)
            else:
                values = tl.load(
                    X + row * STRIDE_XM + cols * STRIDE_XD,
                    mask=cols < D,
                    other=0.0,
                ).to(tl.float32)

            mean = tl.sum(values / D)
            if NO_MASK:
                centered = values - mean
            else:
                centered = tl.where(cols < D, values - mean, 0.0)
            variance = tl.sum(centered * centered) / D
            rstd = tl.rsqrt(variance + EPS)
            result = centered * rstd
            result = result * weight + bias
            if NO_MASK:
                tl.store(Y + row * D + cols, result)
            else:
                tl.store(Y + row * D + cols, result, mask=cols < D)


# ------------------------------------------------------------------
# Launcher
# ------------------------------------------------------------------
def _launch_ascend(x, weight, bias, out, eps):
    """Select a validated shared tile or the compatibility row-loop kernel."""
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
    if (
        rows > 1
        and dim == 128
        and stride_xm in (128, 144)
        and stride_xd == stride_w == stride_b == 1
        and _shared_tile_validated_device(x.device.index)
    ):
        from flaggems_vllm.ops.fused_indexer_k_norm import _fused_indexer_k_norm_kernel

        block_m = _shared_tile_block_m(rows)
        with runtime.torch_device_fn.device(x.device):
            _fused_indexer_k_norm_kernel[(triton.cdiv(rows, block_m),)](
                *args,
                BLOCK_M=block_m,
                SUBNORMAL_METHOD=0,
                num_warps=2,
                num_stages=1,
            )
        return
    # Retained compatibility path for unvalidated devices and layouts.
    rpb = 16 if rows > 1024 else 8
    no_mask = 1 if dim == triton.next_power_of_2(dim) else 0
    _ascend_fused_indexer_k_norm_kernel[(triton.cdiv(rows, rpb),)](
        *args,
        ROWS_PER_BLOCK=rpb,
        NO_MASK=no_mask,
        BLOCK_M=1,
        num_warps=2,
        num_stages=1,
    )

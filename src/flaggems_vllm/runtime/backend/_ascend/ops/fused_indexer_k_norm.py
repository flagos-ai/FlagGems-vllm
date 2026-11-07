# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend-specific fused_indexer_k_norm kernel and launcher.

The Ascend backend (Triton 3.5 + CANN 9.0) does not support BLOCK_M>1
as a grid parameter; raw kernel launch fails with
  "Keyword argument BLOCK_M was passed but not declared"

This module provides a dedicated kernel that keeps BLOCK_M=1 and
amortises launch overhead via an in-kernel ROWS_PER_BLOCK loop.
Welford single-pass formulation keeps the math identical to the
shared kernel while reducing redundant reductions.
"""

import triton
import triton.language as tl


# ------------------------------------------------------------------
# Kernel
# ------------------------------------------------------------------
@triton.jit(do_not_specialize=["EPS"])
def _ascend_fused_indexer_k_norm_kernel(
    X, W, B, Y,
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
                values = tl.load(
                    X + row * STRIDE_XM + cols * STRIDE_XD
                ).to(tl.float32)
            else:
                values = tl.load(
                    X + row * STRIDE_XM + cols * STRIDE_XD,
                    mask=cols < D, other=0.0,
                ).to(tl.float32)

            mean = tl.sum(values) / D
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
    """Launch the Ascend-specific kernel with dynamic RPB tuning."""
    rows, dim = x.shape
    stride_xm, stride_xd = x.stride()
    stride_w = weight.stride(0)
    stride_b = bias.stride(0)
    args = (
        x, weight, bias, out,
        rows, dim,
        stride_xm, stride_xd, stride_w, stride_b,
        eps,
    )
    rpb = 16 if rows > 1024 else 8
    no_mask = 1 if dim == triton.next_power_of_2(dim) else 0
    _ascend_fused_indexer_k_norm_kernel[(triton.cdiv(rows, rpb),)](
        *args,
        ROWS_PER_BLOCK=rpb,
        NO_MASK=no_mask,
        BLOCK_M=1,
        num_warps=4,
        num_stages=1,
    )

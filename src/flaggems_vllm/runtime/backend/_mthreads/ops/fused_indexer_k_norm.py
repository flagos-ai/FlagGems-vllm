# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific fused_indexer_k_norm launcher.

MUSA benefits from BLOCK_M=4 for large M (>1024) on the D=128 aligned
fast path, reducing launch overhead by ~30%.
"""

import triton


def _launch_musa(x, weight, bias, out, eps):
    """Launch the MUSA-optimised kernel with dynamic BLOCK_M."""
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
    # Fast path: D=128 with unit strides
    if dim == 128 and stride_xd == 1 and stride_w == 1 and stride_b == 1:
        block_m = 1
        if rows > 1024:
            block_m = 4
        from flaggems_vllm.ops.fused_indexer_k_norm import _fused_indexer_k_norm_kernel

        _fused_indexer_k_norm_kernel[(triton.cdiv(rows, block_m),)](
            *args,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
    else:
        from flaggems_vllm.ops.fused_indexer_k_norm import _norm_kernel

        block_m = 1
        if rows > 1024:
            block_m = 4
        _norm_kernel[(triton.cdiv(rows, block_m),)](
            *args,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )

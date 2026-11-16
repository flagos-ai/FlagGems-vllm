# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX-specific fused_indexer_k_norm launcher.

MetaX suffers from high Python wrapper overhead (~0.019ms per call).
The public fast path is inlined directly in the shared operator to
avoid the _launch_fused_indexer_k_norm dispatcher layer.
"""


def _launch_metax_fast(x, weight, bias, out, eps):
    """MetaX inlined fast path: call raw kernel directly, no dispatcher."""
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
    if dim == 128 and stride_xd == 1 and stride_w == 1 and stride_b == 1:
        from flaggems_vllm.ops.fused_indexer_k_norm import _fused_indexer_k_norm_kernel

        _fused_indexer_k_norm_kernel[(rows,)](
            *args,
            BLOCK_M=1,
            num_warps=4,
            num_stages=1,
        )
    else:
        from flaggems_vllm.ops.fused_indexer_k_norm import _norm_kernel

        _norm_kernel[(rows,)](
            *args,
            BLOCK_M=1,
            num_warps=4,
            num_stages=1,
        )

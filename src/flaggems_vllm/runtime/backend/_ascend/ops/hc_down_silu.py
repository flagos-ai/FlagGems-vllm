# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ascend launch configurations for the Qwen4 HC down projection."""

import logging

import torch
import triton

from flaggems_vllm.ops.qwen4.hc_down_silu import (
    _hc_down_silu,
    _hc_down_silu_gemm,
    _hc_down_silu_gemv,
    _hc_down_silu_wide_gemv,
)
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)

# Reuse the projection kernels with explicit tiles and no online tile search.
_gemm = libentry()(_hc_down_silu_gemm.jit_function)
_gemv = libentry()(_hc_down_silu_gemv.jit_function)
_wide_gemv = libentry()(_hc_down_silu_wide_gemv.jit_function)


def _gemm_tile(m, k, width, sx, sw):
    # A larger output tile reuses weights across more rows at prefill sizes.
    # On 910B1 this wins for both dense and row-padded model projections.
    if m >= 512 and k == 10240 and width == 324 and sx >= k and sw >= k:
        return 64, 64, 128
    return 16, 32, 128


def _launch(x, weight, output, dimensions, wide):
    m, k, n, rank, hc, sx, sw = dimensions
    width = rank + hc
    if wide or m <= 4 or k < 2048:
        # GEMV reads the row from the grid. Keep M out of its compilation key.
        kernel = _wide_gemv if wide else _gemv
        shape = (0, k, n, rank, hc, sx, sw, 2 if wide else 1)
        grid = (m, triton.cdiv(width, 4))
        kernel[grid](x, weight, output, shape, 4, 1024, num_warps=4, num_stages=1)
    else:
        bm, bn, bk = _gemm_tile(m, k, width, sx, sw)
        grid = (triton.cdiv(m, bm), triton.cdiv(width, bn))
        _gemm[grid](
            x, weight, output, (*dimensions, 0), bm, bn, bk, num_warps=4, num_stages=2
        )


def hc_down_silu(
    x: torch.Tensor, weight: torch.Tensor, rank: int, hc: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return activation and injection views of the packed BF16 projection."""
    logger.debug("GEMS_ASCEND HC_DOWN_SILU")
    return _hc_down_silu(x, weight, rank, hc, _launch)

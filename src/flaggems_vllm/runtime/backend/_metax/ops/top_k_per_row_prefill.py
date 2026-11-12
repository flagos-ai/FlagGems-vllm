# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Reuse the portable selector while preserving the deployed pitched-view ABI."""
import logging

import torch

from flaggems_vllm.ops.top_k_per_row_prefill import top_k_per_row_prefill as _generic

logger = logging.getLogger(__name__)


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    logger.debug("GEMS_METAX TOP_K_PER_ROW_PREFILL")
    if num_rows != logits.shape[0] or indices.shape != (num_rows, top_k):
        raise ValueError("Invalid prefill Top-k row/output shape")
    if logits.dtype != torch.float32 or indices.dtype != torch.int32:
        raise TypeError("Prefill Top-k requires FP32 scores and INT32 output")
    if any(x.device != logits.device for x in (row_starts, row_ends, indices)):
        raise ValueError("Prefill Top-k inputs must share a device")
    if num_rows == 0:
        return
    # Keep row pitch when columns are contiguous; there is no per-width copy.
    if logits.stride(1) != 1 or logits.stride(0) < logits.shape[1]:
        logits = logits.contiguous()
    row_starts, row_ends = row_starts.contiguous(), row_ends.contiguous()
    output = (
        indices
        if indices.is_contiguous()
        else torch.empty((num_rows, top_k), dtype=indices.dtype, device=indices.device)
    )
    _generic(
        logits,
        row_starts,
        row_ends,
        output,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )
    if output is not indices:
        indices.copy_(output)

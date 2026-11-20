# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

"""Paged FP8 FlashTopK interfaces for the vLLM sparse indexer."""

import torch
import triton

from flaggems_vllm.ops.cooperative_topk import _WORKSPACE_BYTES

PAGE_SIZE = 64
HEADS = 64
HEAD_DIM = 128
TOP_K = 2048
MAX_ROWS = 65535
MAX_CONTEXT = 163840


def flash_topk_workspace_specs(rows, max_context):
    """Return the (shape, dtype) tuples for the caller's workspace manager.

    These buffers hold request-local scores, effective row lengths, and selector
    scratch. An all-IDs call (max_context <= 2048) uses no workspace.
    """
    if not isinstance(rows, int) or not 0 <= rows <= MAX_ROWS:
        raise ValueError("FlashTopK requires 0 <= rows <= 65535")
    if not isinstance(max_context, int) or not 0 <= max_context <= MAX_CONTEXT:
        raise ValueError("FlashTopK requires 0 <= max_context <= 163840")
    if not rows or max_context <= TOP_K:
        return ()
    return (
        ((rows, triton.cdiv(max_context, 256) * 256), torch.float32),
        ((rows,), torch.int32),
        ((rows * _WORKSPACE_BYTES,), torch.uint8),
    )


def flash_topk_prefill(
    q,
    kv_cache,
    weights,
    block_table,
    cu_seqlen_ks,
    cu_seqlen_ke,
    token_to_seq,
    max_context,
    out,
    *,
    workspace=None,
):
    """Write request-local INT32 IDs for one vLLM prefill chunk.

    q follows the (FP8 values, None) convention. Read the chunk's 64-token
    physical pages and its flattened K ranges directly. The token_to_seq table
    maps K positions, so the first visible K position identifies the page-table
    row for each query. No cache insertion or distributed KV handling is done.
    """
    raise NotImplementedError(
        "flash_topk_prefill requires NVIDIA SM90 and FlagTree TLE"
    )


def flash_topk_decode(
    q,
    kv_cache,
    weights,
    block_table,
    seq_lens,
    max_context,
    out,
    *,
    workspace=None,
):
    """Write request-local INT32 IDs for vLLM's already packed decode rows.

    q follows the (FP8 [B, next_n, 64, 128] values, None) convention.
    seq_lens contains each row's effective context. Padding and unpacking
    remain the caller's responsibility.
    """
    raise NotImplementedError("flash_topk_decode requires NVIDIA SM90 and FlagTree TLE")

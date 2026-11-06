# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""PPU sparse QSA dispatch with the required minimum 16-row dot tile."""

from __future__ import annotations

import torch

from flaggems_vllm.ops.qwen4.qsa_attention import _qsa_sparse_paged_attention


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    split_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Use M>=16 on PPU for both single-CTA and split sparse attention.

    The PPU matrix-dot path requires this floor. Larger query-head groups
    retain their power-of-two tile, and all validation/ABI semantics remain
    shared with the generic implementation.
    """
    return _qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        softmax_scale,
        out,
        gate,
        split_workspace,
        min_block_m=16,
    )

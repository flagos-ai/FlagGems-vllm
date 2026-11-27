# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch


def minimax_qknorm_rope_cache(
    packed: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    num_actual_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
    index_packed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Q/index-Q and update separate BF16 K, V and index-K caches.

    Caller supplies valid positions and unique nonnegative slots per cache.
    Negative slots and rows beyond num_actual_tokens skip cache writes.
    Input and cache storage must not alias. Main and index projections may
    share the same packed input allocation.
    """
    if packed.device.type != "npu":
        raise NotImplementedError("minimax_qknorm_rope_cache requires Ascend")
    try:
        from flaggems_vllm.runtime.backend._ascend.ops.minimax_qknorm_rope_cache import (
            minimax_qknorm_rope_cache as ascend_minimax,
        )
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax requires FlagTree Ascend TLE and slice operations"
        ) from exc

    return ascend_minimax(
        packed,
        cos_sin_cache,
        positions,
        q_weight,
        k_weight,
        index_q_weight,
        index_k_weight,
        key_cache,
        value_cache,
        index_cache,
        slot_mapping,
        index_slot_mapping,
        num_actual_tokens,
        num_q_heads,
        num_kv_heads,
        num_index_heads,
        eps,
        index_packed,
    )

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch


def flash_mla_with_kvcache_sparse_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA with INT8 content, FP16/BF16 RoPE, and FP32 per-row scales.

    Indices address physical cache tokens. Invalid indices are ignored and
    duplicates count repeatedly. Return output and natural-log LSE; an optional
    sink affects output normalization only. RoPE may have zero width for D512.
    """
    raise NotImplementedError("No registered sparse W8A8 INT8 KVCache MLA backend")


def prepare_flash_mla_with_kvcache_sparse_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[
    Callable[[], tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]
]:
    """Prepare reusable workspace and compilation before graph capture.

    Calls through the returned handle must be serialized on one stream.
    """
    raise NotImplementedError("No registered sparse W8A8 INT8 KVCache MLA backend")

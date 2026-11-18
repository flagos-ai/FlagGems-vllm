# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch


def flash_mla_with_kvcache_sparse_w8a8_fp8(
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
    """Sparse MLA with E4M3 FP8 content, FP16/BF16 RoPE, and FP32 per-row scales.

    Indices address physical cache tokens. Invalid indices are ignored and
    duplicates count repeatedly. Return output and natural-log LSE; an optional
    sink affects output normalization only. RoPE may have zero width for D512.
    Q is [batch, query_tokens, heads, 512], KV is [pages, 64, 1, 512]
    (also [pages, 64, 512]), scales have trailing width one, and RoPE has
    width zero or 64. RoPE is divided by the associated content scale.
    The MThreads backend accepts contiguous tensors, one or two query tokens,
    and 16/32 or a multiple of 64 query heads. Scales must be finite and
    positive; RoPE values must be finite. Sink values may be +/-inf.
    Output matches the RoPE dtype.
    Rows without valid tokens return zero output and +inf LSE. Forward only.
    """
    raise NotImplementedError("No registered sparse W8A8 E4M3 FP8 KVCache MLA backend")

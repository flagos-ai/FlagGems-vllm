# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch

from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import HAS_TLE


def flash_mla_sparse_fwd_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: Optional[torch.Tensor],
    q_scale: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    cache_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode using the separate per-token cache format of dense MLA.

    q_nope [B, 1, H, 512] and k_cache_lora [P, 64, 512] are FP8 e4m3fn;
    q_rope [B, 1, H, R] and k_cache_rope [P, 64, R] are BF16.
    R is 0 for inputs without RoPE, or 64 for the RoPE component.
    Both NoPE and RoPE store values divided by the corresponding FP32 scale:
    q_scale [B, 1, H, 1] and k_scale [P, 64, 1]. This directly accepts
    quantize_q_ckv_per_token / quantize_k_ckv_per_token outputs from dense MLA.
    V is the dequantized 512-dimensional NoPE cache. H must be 64 or 128.

    indices [B, 1, topk] contains int32 physical token IDs (page * 64 + slot).
    Negative and out-of-range IDs are ignored. topk_length [B] optionally
    limits the number of entries read per request. attn_sink [H] affects
    output only. Inputs must be finite, with positive finite scales.

    Native V4 takes 448 contiguous FP8 query channels, seven FP32 scales and
    64 unscaled BF16 RoPE channels. Pass the uint8 [P,64,1,584] page view
    as k_cache_lora with k_cache_rope and k_scale set to None. cache_tokens
    optionally excludes padding in the last page.

    Returns BF16 output [B, 1, H, 512] and natural-log FP32 LSE [B, H, 1].
    Empty attention produces zero output and +inf LSE. Forward only.
    Sensitive rows use FP32 recomputation inline, during merge, or in the
    existing separate repair kernel, depending on the selected path.
    Requires FlagTree cross-dtype WGMMA support (PR #1001).
    """
    if not HAS_TLE:
        raise NotImplementedError(
            "FP8 sparse MLA requires Hopper and FlagTree GPU extensions"
        )
    from flaggems_vllm.runtime.backend._nvidia.sparse_mla import sparse_attention

    return sparse_attention(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        softmax_scale,
        attn_sink,
        topk_length,
        cache_tokens=cache_tokens,
    )

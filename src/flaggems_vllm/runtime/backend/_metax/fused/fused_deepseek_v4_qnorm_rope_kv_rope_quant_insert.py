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
"""MetaX C550 override: token-tiled fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.

The generic kernel gives each (token, head) slot its own program, so KV slots'
quant blocks straggle behind Q slots and 64-thread blocks under-fill the SM. Each
program here takes TPP tokens of one slot, uniformly all-Q or all-KV.

Below 512 tokens this dispatches to the generic kernel, which launches cheaper
where TPP=8 would mask off most of every program. Output is bit-identical.
"""

import torch

# Measured on MetaX C550; see module docstring.
_TILED_MIN_TOKENS = 512
_TPP = 8
_NUM_WARPS = 4


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_head_padded: int,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
    kv_mxfp8: bool = False,
    apply_q_rope: bool = True,
    is_q_interleaved: bool = False,
):
    """See the generic implementation for the layout contract."""
    assert q.is_contiguous() and kv.is_contiguous()
    num_tokens, num_heads, head_dims = q.shape

    if num_tokens < _TILED_MIN_TOKENS:
        # Small shapes are launch-bound; the narrower generic kernel wins.
        from flaggems_vllm.ops.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
            fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert as _generic,
        )

        return _generic(
            q,
            kv,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            q_head_padded,
            eps,
            cache_block_size,
            apply_q_norm,
            kv_mxfp8,
            apply_q_rope,
            is_q_interleaved,
        )

    from flaggems_vllm.ops.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
        _fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_impl as _tiled_impl,
    )

    return _tiled_impl(
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        q_head_padded,
        eps,
        cache_block_size,
        apply_q_norm,
        kv_mxfp8,
        apply_q_rope,
        is_q_interleaved,
        TPP=_TPP,
        num_warps=_NUM_WARPS,
    )

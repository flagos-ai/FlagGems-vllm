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
"""Moore Threads S5000 override: token-tiled, and only at 64 heads.

REQUIRES FLAGTREE >= 0.6.1+mthreads3.6, which bundles a working `llc` at
`triton/backends/mthreads/bin/llc` (md5 cec9ff66714e311670b9412ec760e4aa). With
the `llc` from MUSA toolkit 4.3.5, which older wheels fall back to, the 2-D tile
trips an instruction-selection defect and does not compile.

S5000 has a 32-lane warp, so its generic launch already gives each lane the work
the 64-lane parts need TPP=8/num_warps=4 for. At 128 heads no tiling beats the
generic kernel, and below 512 tokens the difference is within noise. The FP8
cache is bit-identical; q differs by at most one bf16 ULP.
"""

import torch

# See the module docstring: 64 heads only, and 512 tokens and up.
_TILED_MAX_HEADS = 64
_TILED_MIN_TOKENS = 512
_TPP = 4
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

    if num_heads > _TILED_MAX_HEADS or num_tokens < _TILED_MIN_TOKENS:
        # 128 heads: no tiling configuration beats the generic kernel. Small shapes:
        # launch-bound, and too noisy to claim a win in.
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

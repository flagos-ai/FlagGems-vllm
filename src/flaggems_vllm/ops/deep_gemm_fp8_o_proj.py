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

import logging
from typing import Optional, Sequence, Tuple, Union

import torch

from flaggems_vllm.ops.fp8_einsum import fp8_einsum
from flaggems_vllm.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

logger = logging.getLogger(__name__)

_BLOCK_SIZE = (128, 128)
# Only the Hopper recipe (1, 128, 128) with FP32 block scales is supported by the
# generic path; the SM100 recipe (1, 1, 128) needs packed UE8M0 scales.
_SUPPORTED_RECIPE = (1, 128, 128)


def _weight_and_scale(param: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
    """Accept either a quantized linear module or an explicit (weight, scale)."""
    if isinstance(param, (tuple, list)) and len(param) == 2:
        return param[0], param[1]
    return param.weight, param.weight_scale_inv


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    wo_b: torch.nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: Optional[Sequence[int]] = _SUPPORTED_RECIPE,
    tma_aligned_scales: bool = False,
) -> torch.Tensor:
    """DeepSeek-V4 output projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Composes the generic operators in this package, so no DeepGEMM dependency is
    needed: ``fused_inv_rope_fp8_quant`` produces the block-wise FP8 attention
    output, ``fp8_einsum`` performs the ``bhr,hdr->bhd`` projection with the
    block-wise FP8 ``wo_a`` weights, and ``wo_b`` finishes the projection.

    Args:
        o: attention output, ``[num_tokens, n_groups * heads_per_group, head_dim]``
            where ``head_dim == nope_dim + rope_dim``.
        positions: token positions, ``[num_tokens]``.
        cos_sin_cache: RoPE cache laid out as ``cos || sin``, ``[max_pos, rope_dim]``.
        wo_a: FP8 block-wise weight with shape
            ``[n_groups, o_lora_rank, heads_per_group * head_dim]``; either a
            module exposing ``weight`` / ``weight_scale_inv`` or a
            ``(weight, weight_scale_inv)`` tuple.
        wo_b: linear layer mapping ``n_groups * o_lora_rank`` to the hidden size.

    Returns:
        ``[num_tokens, hidden]`` in bfloat16.
    """
    logger.debug("GEMS DEEP GEMM FP8 O PROJ")

    assert o.ndim == 3, "`o` must be [num_tokens, num_heads, head_dim]"
    assert positions.ndim == 1 and positions.shape[0] == o.shape[0]
    assert cos_sin_cache.ndim == 2, "`cos_sin_cache` must be 2D"
    assert o.dtype == torch.bfloat16, "`o` must be bfloat16"
    assert (
        o.shape[1] == n_groups * heads_per_group
    ), "num_heads must equal n_groups * heads_per_group"
    assert o.shape[-1] == nope_dim + rope_dim, "head_dim must equal nope_dim + rope_dim"
    assert o_lora_rank % _BLOCK_SIZE[0] == 0, "o_lora_rank must be a multiple of 128"

    weight_a, scale_a = _weight_and_scale(wo_a)
    reduction = heads_per_group * o.shape[-1]
    assert weight_a.shape == (n_groups, o_lora_rank, reduction), (
        "wo_a weight must be [n_groups, o_lora_rank, heads_per_group * head_dim], "
        f"got {tuple(weight_a.shape)}"
    )

    # The generic path does not implement the SM100 packed-UE8M0 scale layout.
    if tuple(einsum_recipe) != _SUPPORTED_RECIPE or tma_aligned_scales:
        raise NotImplementedError(
            "only the Hopper FP32 block-scale recipe (1, 128, 128) is supported, "
            f"got einsum_recipe={tuple(einsum_recipe)}, "
            f"tma_aligned_scales={tma_aligned_scales}"
        )

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=_BLOCK_SIZE[1],
        tma_aligned_scales=False,
    )

    out = torch.empty(
        (o.shape[0], n_groups, o_lora_rank), dtype=torch.bfloat16, device=o.device
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        o_fp8,
        o_scale,
        weight_a,
        scale_a,
        block_size=_BLOCK_SIZE,
        output_dtype=out.dtype,
        out=out,
    )

    assert callable(wo_b), "`wo_b` must be a callable linear layer"
    return wo_b(out.flatten(1))

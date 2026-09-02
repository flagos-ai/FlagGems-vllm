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

"""PPU (thead) optimized moe_align_block_size.

The generic implementation tries a TLE cooperative kernel
(``moe_align_block_size_tle_atomic_fused_coop``) whose ``tle.distributed_barrier``
feature is unsupported on the PPU backend; the failed JIT attempt costs ~30ms on
every call. This module runs the generic non-TLE 4-stage pipeline directly,
which measures ~70-110us on PPU-ZW810E.
"""

from typing import Optional

import torch
import triton

from flaggems_vllm.ops.moe_align_block_size import (
    ceil_div,
    moe_align_block_size_stage1,
    moe_align_block_size_stage2,
    moe_align_block_size_stage2_vec,
    moe_align_block_size_stage3,
    moe_align_block_size_stage4,
)


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Same semantics as the generic ``moe_align_block_size``, PPU fast path.

    Avoids the TLE cooperative attempt that fails to compile on PPU.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    numel = topk_ids.numel()
    numel_sorted_token_ids = sorted_ids.numel()
    numel_expert_ids = expert_ids.numel()
    grid = (num_experts,)
    tokens_per_thread = triton.next_power_of_2(ceil_div(numel, num_experts))
    block_size_sorted = triton.next_power_of_2(
        ceil_div(numel_sorted_token_ids, num_experts)
    )
    block_size_expert = triton.next_power_of_2(ceil_div(numel_expert_ids, num_experts))

    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_experts_next_power_of_2 = triton.next_power_of_2(num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
        sorted_ids,
        expert_ids,
        numel_sorted_token_ids,
        numel_expert_ids,
        block_size_sorted,
        block_size_expert,
    )
    if num_experts == triton.next_power_of_2(num_experts):
        moe_align_block_size_stage2_vec[grid](tokens_cnts, num_experts)
    else:
        moe_align_block_size_stage2[grid](tokens_cnts, num_experts)
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        num_experts_next_power_of_2,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )

    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad

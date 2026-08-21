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

"""Hygon (HCU) optimized moe_align_block_size.

The generic implementation tries TLE cooperative kernels
(``moe_align_block_size_tle_atomic_fused_coop`` /
``moe_align_block_size_tle_cluster_fused``) whose ``tle.distributed_barrier``
feature cannot be legalized on the Hygon DCU (gfx936) backend; the failed JIT
attempt costs ~290ms on every call. This module runs the generic non-TLE
4-stage pipeline directly (same kernels as the generic triton fallback),
following the pattern of the PPU (thead) backend.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.moe_align_block_size import (
    ceil_div,
    moe_align_block_size_stage1,
    moe_align_block_size_stage2,
    moe_align_block_size_stage2_vec,
    moe_align_block_size_stage3,
    moe_align_block_size_stage4,
)


@triton.jit
def moe_align_block_size_stage4_hygon(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    total_tokens_post_pad_ptr,
    num_experts: tl.constexpr,
    num_experts_next_power_of_2: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
):
    """stage3+stage4 fused (Hygon): each block recomputes the expert prefix
    sum locally and gathers its own experts' starts via a register gather,
    never reading a cross-block-written buffer (unsafe on this backend).
    Saves one kernel launch; on Hygon DCU launches cost ~24us each and
    dominate small-batch align latency (measured 13.9us total for the
    4-stage pipeline at M=16)."""
    pid = tl.program_id(0)
    off_cnt = num_experts * num_experts
    expert_offsets = tl.arange(0, num_experts_next_power_of_2)
    mask = expert_offsets < num_experts
    token_cnts = tl.load(tokens_cnts_ptr + off_cnt + expert_offsets, mask=mask, other=0)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size
    cumsum_values = tl.cumsum(aligned_cnts, axis=0)
    if pid == 0:
        tl.store(total_tokens_post_pad_ptr, tl.sum(aligned_cnts, axis=0))

    start_idx = tl.sum(tl.where(expert_offsets == pid - 1, cumsum_values, 0), axis=0)
    end_idx = tl.sum(tl.where(expert_offsets == pid, cumsum_values, 0), axis=0)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    offset = tl.arange(0, tokens_per_thread) + start_idx
    mask = offset < numel
    expert_id = tl.load(topk_ids_ptr + offset, mask=mask)
    token_idx_in_expert = tl.atomic_add(
        tokens_cnts_ptr + off_t + expert_id, 1, mask=mask
    )
    start_ex = tl.sum(
        tl.where(
            expert_offsets[None, :] == expert_id[:, None],
            cumsum_values[None, :] - aligned_cnts[None, :],
            0,
        ),
        axis=1,
    )
    rank_post_pad = token_idx_in_expert + start_ex
    tl.store(sorted_token_ids_ptr + rank_post_pad, offset, mask=mask)


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Same semantics as the generic ``moe_align_block_size``, Hygon fast path.

    Avoids the TLE cooperative attempt that fails to compile on Hygon DCU.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        # Small-batch tightening (same as vLLM): otherwise the (numel +
        # E*(block_size-1)) bound allocates a huge sorted buffer and the
        # kernel's EM (= sorted.size(0)) inflates the grid with empty blocks.
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    numel = topk_ids.numel()
    expert_p2 = triton.next_power_of_2(num_experts)
    numel_sorted_token_ids = sorted_ids.numel()
    numel_expert_ids = expert_ids.numel()
    grid = (num_experts,)
    tokens_per_thread = triton.next_power_of_2(ceil_div(numel, num_experts))
    block_size_sorted = triton.next_power_of_2(
        ceil_div(numel_sorted_token_ids, num_experts)
    )
    block_size_expert = triton.next_power_of_2(ceil_div(numel_expert_ids, num_experts))

    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
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
    # Small batches (M <= 16 with topk=8): fuse the stage3+stage4 scatter
    # into one kernel that recomputes the expert prefix locally. The
    # single-block scatter variant is unreliable on the HCU backend
    # (atomics dropped for wide aranges).
    if numel <= 128 and expert_p2 <= 512:
        moe_align_block_size_stage4_hygon[grid](
            topk_ids,
            sorted_ids,
            expert_ids,
            tokens_cnts,
            num_tokens_post_pad,
            num_experts,
            num_experts_next_power_of_2,
            block_size,
            numel,
            tokens_per_thread,
        )
    else:
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

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

"""MetaX non-TLE routing alignment, with the same allocating API as generic."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_no_tle as _generic_align,
)
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_stage2_vec,
    moe_align_block_size_stage3,
    moe_align_block_size_stage4,
)
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def _count_routes(Ids, Counts, N: tl.constexpr, R: tl.constexpr):
    offsets = tl.arange(0, R)
    ids = tl.load(Ids + offsets, offsets < N, other=-1)
    expert = tl.program_id(0)
    count = tl.sum(((offsets < N) & (ids == expert)).to(tl.int32), 0)
    tl.store(Counts + expert, count)


@libentry()
@triton.jit
def _scatter_routes(
    Ids,
    Counts,
    Sorted,
    Experts,
    Total,
    N: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
    CAP: tl.constexpr,
    ECAP: tl.constexpr,
    R: tl.constexpr,
):
    expert = tl.program_id(0)
    es = tl.arange(0, E)
    counts = tl.load(Counts + es)
    padded = tl.cdiv(counts, B) * B
    start = tl.sum(tl.where(es < expert, padded, 0), 0)
    count = tl.sum(tl.where(es == expert, counts, 0), 0)
    end = start + tl.cdiv(count, B) * B
    total = tl.sum(padded, 0)

    offsets = tl.arange(0, R)
    ids = tl.load(Ids + offsets, offsets < N, other=-1)
    match = (offsets < N) & (ids == expert)
    rank = tl.cumsum(match.to(tl.int32), 0) - 1
    tl.store(Sorted + start + rank, offsets, match)
    # Padding writes are disjoint from valid-route writes, including between warps.
    for base in range(tl.cdiv(N + B - 1, 256)):
        off = base * 256 + tl.arange(0, 256)
        tl.store(Sorted + start + off, N, (off >= count) & (start + off < end))
    blocks = tl.arange(0, triton.next_power_of_2(triton.cdiv(N, B) + 1))
    tl.store(Experts + start // B + blocks, expert, start // B + blocks < end // B)
    if expert == 0:
        tl.store(Total, total)
    for base in range(tl.cdiv(CAP, E * 256)):
        off = total + (base * E + expert) * 256 + tl.arange(0, 256)
        tl.store(Sorted + off, N, off < CAP)
    for base in range(tl.cdiv(ECAP, E)):
        off = total // B + base * E + expert
        tl.store(Experts + off, -1, off < ECAP)


@libentry()
@triton.jit
def _stage1_with_init(
    Ids,
    Counts,
    Cumsum,
    Sorted,
    Experts,
    E: tl.constexpr,
    N: tl.constexpr,
    TOKENS: tl.constexpr,
    CAP: tl.constexpr,
    ECAP: tl.constexpr,
    SORTED_TILE: tl.constexpr,
    EXPERT_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    es = tl.arange(0, E)
    # Each CTA owns one count row; row zero is the exclusive-prefix identity.
    tl.store(Counts + (pid + 1) * E + es, 0)
    if pid == 0:
        tl.store(Counts + es, 0)
        tl.store(Cumsum, 0)
    tl.debug_barrier()
    offsets = pid * SORTED_TILE + tl.arange(0, SORTED_TILE)
    tl.store(Sorted + offsets, N, offsets < CAP)
    offsets = pid * EXPERT_TILE + tl.arange(0, EXPERT_TILE)
    tl.store(Experts + offsets, -1, offsets < ECAP)
    offsets = pid * TOKENS + tl.arange(0, TOKENS)
    ids = tl.load(Ids + offsets, offsets < N, other=0)
    tl.atomic_add(Counts + (pid + 1) * E + ids, 1, mask=offsets < N)


def moe_align_block_size_no_tle(
    topk_ids,
    block_size,
    num_experts,
    expert_map=None,
    pad_sorted_ids=False,
    workspace=None,
):
    """Align valid contiguous routes; retain the generic path outside E=256.

    Fixed four-warp launches are MetaX-only and bounded by the C550 shared-memory
    budget. No process-wide compiler hooks or compiled-kernel cache are needed.
    Caller-owned workspaces and expert maps keep the generic contract/path.
    """
    if (
        num_experts != 256
        or expert_map is not None
        or workspace is not None
        or not topk_ids.is_contiguous()
    ):
        return _generic_align(
            topk_ids, block_size, num_experts, expert_map, pad_sorted_ids, workspace
        )
    n = topk_ids.numel()
    capacity = n + num_experts * (block_size - 1)
    if pad_sorted_ids:
        capacity = triton.cdiv(capacity, block_size) * block_size
    if n < num_experts:
        capacity = min(capacity, n * block_size)
    sorted_ids = torch.empty(capacity, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(
        triton.cdiv(capacity, block_size), dtype=torch.int32, device=topk_ids.device
    )
    total = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    if n <= 512:
        counts = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
        r = triton.next_power_of_2(max(n, 1))
        _count_routes[(num_experts,)](topk_ids, counts, n, r, num_warps=4, num_stages=1)
        _scatter_routes[(num_experts,)](
            topk_ids,
            counts,
            sorted_ids,
            expert_ids,
            total,
            n,
            num_experts,
            block_size,
            capacity,
            expert_ids.numel(),
            r,
            num_warps=4,
            num_stages=1,
        )
    else:
        cumsum = torch.empty(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
        counts = torch.empty(
            (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
        )
        tokens = triton.next_power_of_2(triton.cdiv(n, num_experts))
        _stage1_with_init[(num_experts,)](
            topk_ids,
            counts,
            cumsum,
            sorted_ids,
            expert_ids,
            num_experts,
            n,
            tokens,
            capacity,
            expert_ids.numel(),
            triton.next_power_of_2(triton.cdiv(capacity, num_experts)),
            triton.next_power_of_2(triton.cdiv(expert_ids.numel(), num_experts)),
            num_warps=4,
            num_stages=1,
        )
        moe_align_block_size_stage2_vec[(num_experts,)](
            counts, num_experts, num_warps=4, num_stages=1
        )
        moe_align_block_size_stage3[(1,)](
            total,
            counts,
            cumsum,
            num_experts,
            num_experts,
            block_size,
            num_warps=4,
            num_stages=1,
        )
        moe_align_block_size_stage4[(num_experts,)](
            topk_ids,
            sorted_ids,
            expert_ids,
            counts,
            cumsum,
            num_experts,
            block_size,
            n,
            tokens,
            num_warps=4,
            num_stages=1,
        )
    return sorted_ids, expert_ids, total

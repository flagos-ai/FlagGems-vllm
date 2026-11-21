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

"""Expert alignment and routed FP8 scale checks."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from flag_gems import zeros

from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle
from flaggems_vllm.runtime.backend._thead.fused.marlin.formats import (
    SHORT_TILE_FLAG,
    write_fp8_expert_scale_safety,
)
from flaggems_vllm.utils import libentry

LARGE_SCALE_WARP_THRESHOLD = 8 * 1024


@libentry()
@triton.jit
def w8_small_routes_kernel(
    ids_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    post_ptr,
    num_routes: tl.constexpr,
    block_m: tl.constexpr,
    block_routes: tl.constexpr,
):
    routes = tl.arange(0, block_routes)
    ids = tl.load(ids_ptr + routes, routes < num_routes, other=-1)
    same_expert = (
        (ids[:, None] == ids[None, :])
        & (routes[:, None] < num_routes)
        & (routes[None, :] < num_routes)
    )
    route_rank = tl.sum(
        (same_expert & (routes[None, :] < routes[:, None])).to(tl.int32), 1
    )
    expert_count = tl.sum(same_expert.to(tl.int32), 1)
    first_route = (route_rank == 0) & (routes < num_routes)
    padded_count = tl.where(first_route, tl.cdiv(expert_count, block_m) * block_m, 0)
    starts = tl.cumsum(padded_count) - padded_count
    first_route_index = tl.min(tl.where(same_expert, routes[None, :], block_routes), 1)
    first_route_index = tl.where(routes < num_routes, first_route_index, 0)
    expert_start = tl.gather(starts, first_route_index, axis=0)
    tl.store(sorted_ids_ptr + expert_start + route_rank, routes, routes < num_routes)
    slots = tl.arange(0, block_routes)
    tl.store(
        sorted_ids_ptr + starts[:, None] + slots[None, :],
        num_routes,
        first_route[:, None]
        & (slots[None, :] >= expert_count[:, None])
        & (slots[None, :] < padded_count[:, None]),
    )
    tl.store(
        expert_ids_ptr + starts[:, None] // block_m + slots[None, :],
        ids[:, None],
        first_route[:, None]
        & (slots[None, :] < tl.cdiv(expert_count[:, None], block_m)),
    )
    tl.store(post_ptr, tl.sum(padded_count))


@libentry()
@triton.jit
def init_route_workspace(
    sorted_routes, ranks, num_routes, capacity, num_experts, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(sorted_routes + offsets, num_routes, offsets < capacity)
    tl.store(ranks + offsets, 0, offsets < num_experts)


@triton.jit
def w8_one_cta_routes(
    route_ids,
    sorted_routes,
    block_experts,
    padded_count,
    expert_ranks,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    PACK_WIDTH: tl.constexpr = False,
):
    route = tl.arange(0, BLOCK_ROUTES)
    is_valid = route < NUM_ROUTES
    expert = tl.load(route_ids + route, is_valid, other=0).to(tl.int32)
    count = tl.histogram(expert, BLOCK_EXPERTS, mask=is_valid)
    padded = tl.cdiv(count, BM) * BM
    start = tl.cumsum(padded, 0) - padded
    rank = tl.atomic_add(expert_ranks + expert, 1, mask=is_valid, sem="relaxed")
    if PACK_WIDTH:
        tl.static_assert(BM == 16)
        offset = tl.gather(start, expert, 0) + rank
        valid_left = tl.gather(count, expert, 0) - rank
        # Record when the 16-row tile only needs its first eight rows.
        expert_word = expert | tl.where(valid_left <= 8, SHORT_TILE_FLAG, 0)
    else:
        offset = tl.gather(start, expert, 0) + rank
        expert_word = expert
    tl.store(sorted_routes + offset, route, is_valid)
    tl.store(block_experts + offset // BM, expert_word, is_valid & (rank % BM == 0))
    tl.store(padded_count, tl.sum(padded, 0))


@libentry()
@triton.jit
def fp8_safety_and_routes(
    gate_scales,
    down_scales,
    gate_scale_flags,
    down_scale_flags,
    route_ids,
    sorted_routes,
    block_experts,
    padded_count,
    expert_ranks,
    NUM_EXPERTS: tl.constexpr,
    N1: tl.constexpr,
    G1: tl.constexpr,
    N2: tl.constexpr,
    G2: tl.constexpr,
    SE1: tl.constexpr,
    SN1: tl.constexpr,
    SG1: tl.constexpr,
    SE2: tl.constexpr,
    SN2: tl.constexpr,
    SG2: tl.constexpr,
    BLOCK1: tl.constexpr,
    BLOCK2: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    PACK_WIDTH: tl.constexpr = False,
):
    if tl.program_id(0) == 0:
        # Routing never reads other CTAs' flags; GEMMs consume both after this launch.
        w8_one_cta_routes(
            route_ids,
            sorted_routes,
            block_experts,
            padded_count,
            expert_ranks,
            NUM_ROUTES,
            NUM_EXPERTS,
            BM,
            BLOCK_ROUTES,
            BLOCK_EXPERTS,
            PACK_WIDTH=PACK_WIDTH,
        )
        return
    safety_pid = tl.program_id(0) - 1
    if NUM_ROUTES <= NUM_EXPERTS:
        # Only the first route for each expert writes its scale flags.
        route = safety_pid
        expert = tl.load(route_ids + route).to(tl.int64)
        prior = tl.arange(0, BLOCK_ROUTES)
        prior_ids = tl.load(route_ids + prior, prior < route, other=-1).to(tl.int64)
        is_first = (
            (expert >= 0)
            & (expert < NUM_EXPERTS)
            & (tl.sum(prior_ids == expert, 0) == 0)
        )
        if is_first:
            write_fp8_expert_scale_safety(
                gate_scales, gate_scale_flags, expert, N1, G1, SE1, SN1, SG1, BLOCK1
            )
            write_fp8_expert_scale_safety(
                down_scales, down_scale_flags, expert, N2, G2, SE2, SN2, SG2, BLOCK2
            )
    else:
        # An expert with no routed tokens has no GEMM consuming its flags.
        expert = safety_pid.to(tl.int64)
        row = tl.arange(0, BLOCK_ROUTES)
        routed = tl.load(route_ids + row, row < NUM_ROUTES, other=-1)
        is_active = tl.sum((routed == expert).to(tl.int32), 0) != 0
        if is_active:
            write_fp8_expert_scale_safety(
                gate_scales, gate_scale_flags, expert, N1, G1, SE1, SN1, SG1, BLOCK1
            )
            write_fp8_expert_scale_safety(
                down_scales, down_scale_flags, expert, N2, G2, SE2, SN2, SG2, BLOCK2
            )


def align_grouped_tokens(
    topk_ids: torch.Tensor,
    block_m: int,
    num_experts: int,
    fp8_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    fp8_safety: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    is_one_cta: bool = False,
    packed_width: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_one_cta:
        routes = topk_ids.numel()
        max_padded_routes = min(routes * block_m, routes + num_experts * (block_m - 1))
        sorted_ids = torch.empty(
            (max_padded_routes,), device=topk_ids.device, dtype=torch.int32
        )
        experts = torch.empty(
            (triton.cdiv(max_padded_routes, block_m),),
            device=topk_ids.device,
            dtype=torch.int32,
        )
        padded = torch.empty((1,), device=topk_ids.device, dtype=torch.int32)
        rank = torch.empty((num_experts,), device=topk_ids.device, dtype=torch.int32)
        # Stream launch order makes the global writes visible before scatter and atomics.
        init_route_workspace[(triton.cdiv(max(max_padded_routes, num_experts), 256),)](
            sorted_ids, rank, routes, max_padded_routes, num_experts, 256
        )
        warps = 8 if routes <= 128 else 16
        if fp8_scales is None:
            w8_one_cta_routes[(1,)](
                topk_ids,
                sorted_ids,
                experts,
                padded,
                rank,
                routes,
                num_experts,
                block_m,
                triton.next_power_of_2(max(routes, warps * 32)),
                triton.next_power_of_2(num_experts),
                num_warps=warps,
                num_stages=1,
            )
        else:
            s1, s2 = fp8_scales
            b1 = triton.next_power_of_2(s1.size(1) * s1.size(2))
            b2 = triton.next_power_of_2(s2.size(1) * s2.size(2))
            warps = max(warps, 16 if max(b1, b2) >= LARGE_SCALE_WARP_THRESHOLD else 4)
            safety_ctas = routes if routes <= num_experts else num_experts
            fp8_safety_and_routes[(safety_ctas + 1,)](
                s1,
                s2,
                *fp8_safety,
                topk_ids,
                sorted_ids,
                experts,
                padded,
                rank,
                num_experts,
                s1.size(1),
                s1.size(2),
                s2.size(1),
                s2.size(2),
                *s1.stride(),
                *s2.stride(),
                b1,
                b2,
                routes,
                block_m,
                triton.next_power_of_2(max(routes, warps * 32)),
                triton.next_power_of_2(num_experts),
                PACK_WIDTH=packed_width,
                num_warps=warps,
                num_stages=1,
            )
        return sorted_ids, experts, padded
    workspace = (
        zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device),
        zeros(
            (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
        ),
    )
    aligned = moe_align_block_size_no_tle(
        topk_ids, block_m, num_experts, workspace=workspace
    )
    if fp8_scales is not None:
        s1, s2 = fp8_scales
        b1 = triton.next_power_of_2(s1.size(1) * s1.size(2))
        b2 = triton.next_power_of_2(s2.size(1) * s2.size(2))
        check_routed_fp8_scales[(num_experts,)](
            s1,
            s2,
            *fp8_safety,
            workspace[0],
            *s1.shape[1:],
            *s2.shape[1:],
            *s1.stride(),
            *s2.stride(),
            b1,
            b2,
            num_warps=16 if max(b1, b2) >= LARGE_SCALE_WARP_THRESHOLD else 4,
        )
    return aligned


@libentry()
@triton.jit
def check_routed_fp8_scales(
    gate_scales,
    down_scales,
    gate_scale_flags,
    down_scale_flags,
    Prefix,
    N1: tl.constexpr,
    G1: tl.constexpr,
    N2: tl.constexpr,
    G2: tl.constexpr,
    SE1: tl.constexpr,
    SN1: tl.constexpr,
    SG1: tl.constexpr,
    SE2: tl.constexpr,
    SN2: tl.constexpr,
    SG2: tl.constexpr,
    BLOCK1: tl.constexpr,
    BLOCK2: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    if tl.load(Prefix + expert) != tl.load(Prefix + expert + 1):
        write_fp8_expert_scale_safety(
            gate_scales, gate_scale_flags, expert, N1, G1, SE1, SN1, SG1, BLOCK1
        )
        write_fp8_expert_scale_safety(
            down_scales, down_scale_flags, expert, N2, G2, SE2, SN2, SG2, BLOCK2
        )

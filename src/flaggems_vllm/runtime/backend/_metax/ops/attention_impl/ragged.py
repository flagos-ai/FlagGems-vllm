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

"""CW1 materialized compact-ragged worklist adapter."""

import torch
import triton
import triton.language as tl

from .common import compact_ragged_tile_coords
from .direct import launch_direct
from .tn_direct import launch_d256_tn_direct


def compact_worklist_task_upper_bound(total_q, batch_size, block_m):
    """Safe host-visible upper bound for ``sum(ceil(q_len / block_m))``."""

    return triton.cdiv(total_q, block_m) + batch_size - 1


@triton.jit
def build_compact_worklist_kernel(
    cu_seqlens_q_ptr,
    worklist_ptr,
    batch_size,
    BLOCK_M: tl.constexpr,
):
    """Materialize one AoS ``(bid, m_block)`` descriptor per Q tile."""

    task_pid = tl.program_id(0)
    m_block, bid, work_valid = compact_ragged_tile_coords(
        task_pid,
        cu_seqlens_q_ptr,
        batch_size,
        BLOCK_M,
    )
    descriptor_offset = task_pid * 2
    tl.store(worklist_ptr + descriptor_offset, tl.where(work_valid, bid, -1))
    tl.store(worklist_ptr + descriptor_offset + 1, tl.where(work_valid, m_block, 0))


@triton.jit
def build_two_request_cost_ordered_worklist_kernel(
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    seqused_k_ptr,
    worklist_ptr,
    IS_CU_SEQLENS_K: tl.constexpr,
    IS_SEQUSED_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Put high-cost request/tile descriptors first for a two-request batch."""

    task_pid = tl.program_id(0)
    q0_bos = tl.load(cu_seqlens_q_ptr).to(tl.int32)
    q1_bos = tl.load(cu_seqlens_q_ptr + 1).to(tl.int32)
    q1_eos = tl.load(cu_seqlens_q_ptr + 2).to(tl.int32)
    q0_len = q1_bos - q0_bos
    q1_len = q1_eos - q1_bos
    q0_tiles = tl.cdiv(q0_len, BLOCK_M)
    q1_tiles = tl.cdiv(q1_len, BLOCK_M)

    if IS_SEQUSED_K:
        k0_len = tl.load(seqused_k_ptr).to(tl.int32)
        k1_len = tl.load(seqused_k_ptr + 1).to(tl.int32)
    else:
        tl.static_assert(IS_CU_SEQLENS_K, "TN2 requires a KV length source")
        k0_bos = tl.load(cu_seqlens_k_ptr).to(tl.int32)
        k1_bos = tl.load(cu_seqlens_k_ptr + 1).to(tl.int32)
        k1_eos = tl.load(cu_seqlens_k_ptr + 2).to(tl.int32)
        k0_len = k1_bos - k0_bos
        k1_len = k1_eos - k1_bos

    # The maximum causal work of a request grows with K length.  Schedule the
    # request with larger K first, then issue its Q tiles from high to low so
    # the final hardware wave contains the lightest causal prefixes.
    first_is_one = k1_len > k0_len
    first_bid = tl.where(first_is_one, 1, 0)
    first_tiles = tl.where(first_is_one, q1_tiles, q0_tiles)
    second_tiles = tl.where(first_is_one, q0_tiles, q1_tiles)
    in_first = task_pid < first_tiles
    local_task = tl.where(in_first, task_pid, task_pid - first_tiles)
    selected_tiles = tl.where(in_first, first_tiles, second_tiles)
    bid = tl.where(in_first, first_bid, 1 - first_bid)
    m_block = selected_tiles - 1 - local_task
    work_valid = task_pid < first_tiles + second_tiles

    descriptor_offset = task_pid * 2
    tl.store(worklist_ptr + descriptor_offset, tl.where(work_valid, bid, -1))
    tl.store(worklist_ptr + descriptor_offset + 1, tl.where(work_valid, m_block, 0))


def launch_compact_worklist(
    params,
    *,
    max_seqlen_q,
    batch_size,
    num_heads,
    total_q,
    head_size,
    is_paged,
    grid_order,
    task_upper,
    block_m,
):
    """Build the graph-captured descriptor list, then launch Direct."""

    expected_upper = compact_worklist_task_upper_bound(total_q, batch_size, block_m)
    if task_upper != expected_upper or task_upper <= 0:
        raise ValueError("CW1 worklist task upper bound is inconsistent")
    worklist = torch.empty(
        (task_upper, 2),
        dtype=torch.int32,
        device=params.q_ptr.device,
    )
    build_compact_worklist_kernel[(task_upper,)](
        params.cu_seqlens_q_ptr,
        worklist,
        batch_size,
        BLOCK_M=block_m,
        num_warps=1,
        num_stages=1,
    )

    return launch_direct(
        params,
        max_seqlen_q=max_seqlen_q,
        batch_size=batch_size,
        num_heads=num_heads,
        total_q=total_q,
        head_size=head_size,
        is_paged=is_paged,
        compact_worklist=worklist,
        worklist_block_m=block_m,
        grid_order=grid_order,
    )


def launch_d256_tn_compact_worklist(
    params,
    *,
    max_seqlen_q,
    max_seqlen_k,
    batch_size,
    num_heads,
    total_q,
    grid_order,
    task_upper,
    block_m,
    block_n,
):
    """Build PackGQA descriptors and launch the TN2 async-TN kernel."""

    expected_upper = compact_worklist_task_upper_bound(total_q, batch_size, block_m)
    if task_upper != expected_upper or task_upper <= 0:
        raise ValueError("TN1 worklist task upper bound is inconsistent")
    if block_m % params.h_hk_ratio != 0:
        raise ValueError("TN2 PackGQA requires BLOCK_M divisible by the GQA ratio")
    if params.block_size == 16 and params.h_hk_ratio == 4:
        from .tn_direct import _d256_tn_bulk_policy

        use_bulk, _, _ = _d256_tn_bulk_policy(
            params,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            batch_size=batch_size,
            total_q=total_q,
        )
        if use_bulk:
            return launch_d256_tn_direct(
                params,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                num_heads=num_heads,
                total_q=total_q,
                compact_worklist=None,
                block_m=block_m,
                block_n=block_n,
                grid_order=grid_order,
            )
    packed_block_m = block_m // params.h_hk_ratio
    packed_task_upper = compact_worklist_task_upper_bound(
        total_q, batch_size, packed_block_m
    )
    worklist = torch.empty(
        (packed_task_upper, 2),
        dtype=torch.int32,
        device=params.q_ptr.device,
    )
    # This narrow D256 TN route has a fixed two-request specialization.  Put
    # the more expensive request/tile work first to reduce the final CTA wave.
    cost_ordered_batch2 = batch_size == 2
    if cost_ordered_batch2:
        build_two_request_cost_ordered_worklist_kernel[(packed_task_upper,)](
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
            worklist,
            IS_CU_SEQLENS_K=params.is_cu_seqlens_k,
            IS_SEQUSED_K=params.is_seqused_k,
            BLOCK_M=packed_block_m,
            num_warps=1,
            num_stages=1,
        )
    else:
        build_compact_worklist_kernel[(packed_task_upper,)](
            params.cu_seqlens_q_ptr,
            worklist,
            batch_size,
            BLOCK_M=packed_block_m,
            num_warps=1,
            num_stages=1,
        )
    return launch_d256_tn_direct(
        params,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        batch_size=batch_size,
        num_heads=num_heads,
        total_q=total_q,
        compact_worklist=worklist,
        block_m=block_m,
        block_n=block_n,
        grid_order=grid_order,
    )


__all__ = [
    "build_compact_worklist_kernel",
    "build_two_request_cost_ordered_worklist_kernel",
    "compact_worklist_task_upper_bound",
    "launch_compact_worklist",
    "launch_d256_tn_compact_worklist",
]

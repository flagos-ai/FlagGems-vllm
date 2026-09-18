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

import copy

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

from .common import compact_ragged_tile_coords, tn_compile_scenario
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
        tl.static_assert(IS_CU_SEQLENS_K, "Async-TN requires a KV length source")
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
        raise ValueError("Compact worklist task upper bound is inconsistent")
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
    """Build PackGQA descriptors and launch the async-TN kernel."""

    expected_upper = compact_worklist_task_upper_bound(total_q, batch_size, block_m)
    if task_upper != expected_upper or task_upper <= 0:
        raise ValueError("Async-TN worklist task upper bound is inconsistent")
    if block_m % params.h_hk_ratio != 0:
        raise ValueError("Async-TN PackGQA requires BLOCK_M divisible by the GQA ratio")
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


@libentry()
@triton.jit
def _repack_pages(
    k,
    v,
    table,
    used,
    cu,
    cu_q,
    compact_k,
    compact_v,
    compact_table,
    TABLE_STRIDE: tl.constexpr,
    PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    K_PAGE: tl.constexpr,
    V_PAGE: tl.constexpr,
    K_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_HEAD: tl.constexpr,
    USED: tl.constexpr,
    BLOCK: tl.constexpr,
    BULK_ONLY: tl.constexpr,
):
    page = tl.program_id(0)
    bid = tl.program_id(1)
    destination = bid * PAGES + page
    tl.store(compact_table + destination, destination)
    if USED:
        length = tl.load(used + bid)
    else:
        length = tl.load(cu + bid + 1) - tl.load(cu + bid)
    if BULK_ONLY:
        q_len = tl.load(cu_q + bid + 1) - tl.load(cu_q + bid)
        prefix_q = min(q_len, (q_len - length) & 15)
        if q_len < 16 and q_len == prefix_q:
            return
    if page * PAGE_SIZE >= length:
        return
    physical = tl.load(table + bid * TABLE_STRIDE + page).to(tl.int64)
    elements = tl.arange(0, BLOCK)
    rows = elements // (HEADS * DIM)
    heads = elements // DIM % HEADS
    dims = elements % DIM
    valid = (elements < PAGE_SIZE * HEADS * DIM) & (page * PAGE_SIZE + rows < length)
    kval = tl.load(
        k + physical * K_PAGE + rows * K_ROW + heads * K_HEAD + dims,
        mask=valid,
        other=0,
    )
    vval = tl.load(
        v + physical * V_PAGE + rows * V_ROW + heads * V_HEAD + dims,
        mask=valid,
        other=0,
    )
    dst = destination * (PAGE_SIZE * HEADS * DIM) + elements
    tl.store(compact_k + dst, kval, mask=elements < PAGE_SIZE * HEADS * DIM)
    tl.store(compact_v + dst, vval, mask=elements < PAGE_SIZE * HEADS * DIM)


def maybe_repack(
    params, *, max_seqlen_q, max_seqlen_k, batch_size, total_q, bulk_only=False
):
    if not (
        params.block_size == 16
        and params.h_hk_ratio == 4
        and params.d == 256
        and params.q_ptr.dtype == torch.bfloat16
        and max_seqlen_q >= 1024
        and total_q >= 8192
    ):
        return params
    if "noaddropt" not in tn_compile_scenario(params.k_ptr, params.v_ptr):
        return params
    pages = triton.cdiv(max_seqlen_k, params.block_size)
    if pages <= 0 or pages > params.page_table_ptr.shape[1]:
        return params
    elements = batch_size * pages * params.block_size * params.hk * params.d
    if elements * params.k_ptr.element_size() > (1 << 32):
        return params
    shape = (batch_size * pages, params.block_size, params.hk, params.d)
    k = torch.empty(shape, dtype=params.k_ptr.dtype, device=params.k_ptr.device)
    v = torch.empty_like(k)
    table = torch.empty((batch_size, pages), dtype=torch.int32, device=k.device)
    _repack_pages[(pages, batch_size)](
        params.k_ptr,
        params.v_ptr,
        params.page_table_ptr,
        params.seqused_k_ptr,
        params.cu_seqlens_k_ptr,
        params.cu_seqlens_q_ptr,
        k,
        v,
        table,
        TABLE_STRIDE=params.page_table_batch_stride,
        PAGES=pages,
        PAGE_SIZE=params.block_size,
        HEADS=params.hk,
        DIM=params.d,
        K_PAGE=params.k_page_stride,
        V_PAGE=params.v_ptr.stride(0),
        K_ROW=params.k_row_stride,
        V_ROW=params.v_row_stride,
        K_HEAD=params.k_head_stride,
        V_HEAD=params.v_head_stride,
        USED=params.is_seqused_k,
        BULK_ONLY=bulk_only,
        BLOCK=triton.next_power_of_2(params.block_size * params.hk * params.d),
        num_warps=4,
        num_stages=1,
        pipeline="basic",
    )
    result = copy.copy(params)
    result.k_ptr, result.v_ptr = k, v
    result.k_page_stride = k.stride(0)
    result.k_batch_stride, result.v_batch_stride = k.stride(0), v.stride(0)
    result.k_row_stride, result.v_row_stride = k.stride(1), v.stride(1)
    result.k_head_stride, result.v_head_stride = k.stride(2), v.stride(2)
    result.page_table_ptr = table
    result.page_table_batch_stride = table.stride(0)
    return result

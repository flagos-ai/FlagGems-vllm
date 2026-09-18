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

"""Stage active page16 KV into a bounded paged cache, including graph replay."""
import copy

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

from .tn_addressing import tn_compile_scenario


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

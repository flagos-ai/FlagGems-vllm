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

from typing import Optional, Tuple, Union

import torch
import triton

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._thead.fused.attention import (
    PACK_TILE,
    _flash_int8_fwd,
    _flash_int8_merge_splits,
    _flash_int8_pack_kv,
)


def launch_paged_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    window_size: Tuple[int, int],
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    return_softmax_lse: bool,
    out: Optional[torch.Tensor],
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    batch = cu_seqlens_q.numel() - 1
    total, heads, dim = q.shape
    kv_heads = k.shape[-2]
    group = heads // kv_heads
    paged = block_table is not None
    page_size = k.shape[1] if paged else 0
    # Bound packing by the physical cache size, including shared-page layouts.
    logical_kv = (
        paged
        and max_seqlen_q >= 128
        and max_seqlen_k > 0
        and k.shape[0] > 0
        and batch * max_seqlen_k <= k.shape[0] * page_size
    )
    left, right = window_size
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = (
        torch.empty((heads, total), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )
    # Folding short queries reuses each KV tile across its grouped query heads.
    fold = max_seqlen_q <= 16 and 1 < group <= 16 and group & (group - 1) == 0
    block_m = 32 if fold and max_seqlen_q > 4 else 16
    if max_seqlen_q >= 128:
        block_m = 64
    block_n = 128 if max_seqlen_k >= 512 else 64
    num_warps = 4
    if dim == 128 and logical_kv:
        block_m, block_n, num_warps = 32, 64, 2
    query_tile = block_m // group if fold else block_m
    grid_heads = kv_heads if fold else heads
    parallel_heads = batch * grid_heads
    if left < 0 and right < 0 and max_seqlen_q <= 4 and max_seqlen_k >= 512:
        splits = 4 if parallel_heads <= 128 else 2
    elif left < 0 and right < 0 and 4 < max_seqlen_q <= 16 and max_seqlen_k >= 2048:
        splits = 2
    else:
        splits = 1
    split_kv = splits > 1
    # The compact mapper reads CUQ on device, including empty and padded requests.
    compact = max_seqlen_q > 16 and total * 2 < batch * max_seqlen_q
    tiles = (
        triton.cdiv(total, query_tile) + batch - 1
        if compact
        else triton.cdiv(max_seqlen_q, query_tile)
    )
    grid = (tiles * splits, 1 if compact else batch, grid_heads)
    with torch_device_fn.device(q.device):
        if logical_kv:
            packed_shape = (batch, max_seqlen_k, kv_heads, dim)
            packed_stride = (kv_heads * max_seqlen_k * dim, dim, max_seqlen_k * dim, 1)
            packed_key = torch.empty_strided(
                packed_shape, packed_stride, dtype=torch.int8, device=k.device
            )
            packed_value = torch.empty_strided(
                packed_shape, packed_stride, dtype=torch.float16, device=v.device
            )
            _flash_int8_pack_kv[
                (triton.cdiv(max_seqlen_k, PACK_TILE.value), batch, kv_heads)
            ](
                k,
                v,
                packed_key,
                packed_value,
                block_table,
                seqused_k,
                cu_seqlens_q,
                max_seqlen_k,
                dim,
                kv_heads,
                page_size,
                block_table.stride(0),
                *k.stride()[:3],
                *v.stride()[:3],
                LOGICAL=True,
                SKIP_SHORT=False,
            )
            k, v = packed_key, packed_value
        if split_kv:
            kernel_out = torch.empty(
                (splits, total, heads, dim), dtype=torch.float32, device=q.device
            )
            kernel_stats = torch.empty(
                (splits, 2, heads, total), dtype=torch.float32, device=q.device
            )
        else:
            kernel_out, kernel_stats = out, lse
        _flash_int8_fwd[grid](
            q,
            k,
            v,
            kernel_out,
            kernel_stats,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            block_table,
            None,
            q_descale,
            k_descale,
            v_descale,
            alibi_slopes,
            q.stride(0),
            q.stride(1),
            k.stride(-3),
            k.stride(-2),
            k.stride(0) if paged else 0,
            v.stride(-3),
            v.stride(-2),
            v.stride(0) if paged else 0,
            kernel_out.stride(-3),
            kernel_out.stride(-2),
            *q_descale.stride(),
            *k_descale.stride(),
            *v_descale.stride(),
            block_table.stride(0) if paged else 0,
            (
                alibi_slopes.stride(0)
                if alibi_slopes is not None and alibi_slopes.ndim == 2
                else 0
            ),
            total,
            group,
            dim,
            page_size,
            paged,
            causal,
            left,
            right,
            softcap,
            softmax_scale,
            alibi_slopes is not None,
            return_softmax_lse or split_kv,
            HALF_PV=True,
            PRECISE_PV=False,
            ASYNC_K=False,
            REORDER_CAUSAL=False,
            TAIL_N=block_n,
            KV_SPLITS=splits,
            OUT_SPLIT_STRIDE=kernel_out.stride(0) if split_kv else 0,
            LSE_SPLIT_STRIDE=kernel_stats.stride(0) if split_kv else 0,
            SEPARATE_MASK=True,
            LOGICAL_KV=logical_kv,
            BATCH=batch,
            COMPACT=compact,
            USE_WORKLIST=False,
            WORK_CAPACITY=0,
            TILE_PITCH=1,
            SHORT_ONLY=False,
            FOLD=fold,
            BM=block_m,
            BN=block_n,
            TRANSPOSE_DOT=True,
            num_warps=num_warps,
            num_stages=1,
            pipeline="basic",
            scenario="attention-query-warps" if dim == 128 and logical_kv else "",
        )
        if split_kv:
            _flash_int8_merge_splits[(total, heads)](
                kernel_out,
                kernel_stats,
                out,
                lse,
                cu_seqlens_q,
                total,
                heads,
                dim,
                batch,
                splits,
                out.stride(0),
                out.stride(1),
                return_softmax_lse,
                num_warps=1,
            )
    return (out, lse) if return_softmax_lse else out

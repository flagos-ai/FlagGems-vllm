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

"""Materialize paged KV blocks in block-table order for attention consumers."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def _paged_kv_block_gather_kernel(
    cache,
    block_table,
    out,
    num_blocks,
    table_stride_0,
    table_stride_1,
    cache_stride_0,
    cache_stride_1,
    cache_stride_2,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
    TILE: tl.constexpr,
):
    seq = tl.program_id(0)
    slot = tl.program_id(1)
    physical = tl.load(block_table + seq * table_stride_0 + slot * table_stride_1)
    valid = (physical >= 0) & (physical < num_blocks)
    dst_base = (seq * tl.num_programs(1) + slot) * BLOCK_ELEMS
    for offset in range(0, BLOCK_ELEMS, TILE):
        idx = offset + tl.arange(0, TILE)
        row = idx // HEAD_DIM
        col = idx % HEAD_DIM
        values = tl.load(
            cache
            + physical * cache_stride_0
            + row * cache_stride_1
            + col * cache_stride_2,
            mask=(idx < BLOCK_ELEMS) & valid,
            other=0,
        )
        # Invalid physical block ids produce deterministic zero-filled output.
        tl.store(out + dst_base + idx, values, mask=idx < BLOCK_ELEMS)


def paged_kv_block_gather(
    cache: torch.Tensor, block_table: torch.Tensor
) -> torch.Tensor:
    """Return [seqs, blocks/seq, block_size, head_dim]; invalid ids yield zeros."""
    if cache.ndim != 3 or block_table.ndim != 2:
        raise ValueError("Expected cache [blocks, size, dim], table [seqs, blocks/seq]")
    if cache.device != block_table.device:
        raise ValueError("cache and block_table must use the same device")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_table must contain int32 or int64 block indices")
    if cache.layout != torch.strided or block_table.layout != torch.strided:
        raise NotImplementedError("paged_kv_block_gather requires strided tensors")
    num_blocks, block_size, head_dim = cache.shape
    num_seqs, blocks_per_seq = block_table.shape
    out = torch.empty(
        (num_seqs, blocks_per_seq, block_size, head_dim),
        dtype=cache.dtype,
        device=cache.device,
    )
    if out.numel() == 0:
        return out
    if num_blocks == 0:
        raise ValueError("cannot gather from an empty cache")
    block_elems = block_size * head_dim
    tile = min(triton.next_power_of_2(block_elems), 8192)
    with torch_device_fn.device(cache.device):
        _paged_kv_block_gather_kernel[(num_seqs, blocks_per_seq)](
            cache,
            block_table,
            out,
            num_blocks,
            block_table.stride(0),
            block_table.stride(1),
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            BLOCK_ELEMS=block_elems,
            TILE=tile,
        )
    return out

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

"""Ascend TensorView block-DMA path for packed paged KV block gather."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.paged_kv_block_gather import (
    paged_kv_block_gather as _generic,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    UB_BYTES,
    tensorview_call,
)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle
    from triton.experimental.tle import tile_load, tile_store

    @triton.jit
    def _paged_kv_block_gather_tv_kernel(
        cache,
        block_table,
        out,
        num_blocks,
        num_seqs,
        table_stride,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCKS_PER_SEQ: tl.constexpr,
    ):
        seq = tl.program_id(0)
        slots = tl.arange(0, BLOCKS_PER_SEQ)
        physical = tl.load(block_table + seq * table_stride + slots)
        physical = tl.where((physical >= 0) & (physical < num_blocks), physical, -1).to(
            tl.int32
        )
        cache_view = tle.make_gather_scatter_view(
            cache,
            shape=[num_blocks, BLOCK_SIZE, HEAD_DIM],
            strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCKS_PER_SEQ, BLOCK_SIZE, HEAD_DIM],
            sparse_dim=[0],
            bounds_check=True,
        )
        gathered = tile_load(
            cache_view,
            index=(physical, 0, 0),
            dst_space=tle.language.dsa.ascend.UB,
        )
        out_view = tle.make_partition_view(
            out,
            shape=[num_seqs * BLOCKS_PER_SEQ, BLOCK_SIZE, HEAD_DIM],
            strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCKS_PER_SEQ, BLOCK_SIZE, HEAD_DIM],
        )
        tile_store(
            out_view,
            value=gathered,
            index=(seq, 0, 0),
            src_space=tle.language.dsa.ascend.UB,
        )

    def _tensorview(cache, block_table):
        num_blocks, block_size, head_dim = cache.shape
        num_seqs, blocks_per_seq = block_table.shape
        out = torch.empty(
            (num_seqs, blocks_per_seq, block_size, head_dim),
            dtype=cache.dtype,
            device=cache.device,
        )
        with torch_device_fn.device(cache.device):
            _paged_kv_block_gather_tv_kernel[(num_seqs,)](
                cache,
                block_table,
                out,
                num_blocks,
                num_seqs,
                block_table.stride(0),
                BLOCK_SIZE=block_size,
                HEAD_DIM=head_dim,
                BLOCKS_PER_SEQ=blocks_per_seq,
            )
        return out


def _tv_eligible(cache, block_table):
    if not HAS_TENSORVIEW or cache.ndim != 3 or block_table.ndim != 2:
        return False
    if cache.device != block_table.device or cache.numel() == 0:
        return False
    if block_table.numel() == 0 or block_table.dtype not in (torch.int32, torch.int64):
        return False
    blocks_per_seq = block_table.shape[1]
    if triton.next_power_of_2(blocks_per_seq) != blocks_per_seq:
        return False
    if (
        blocks_per_seq * cache.shape[1] * cache.shape[2] * cache.element_size()
        > UB_BYTES // 2
    ):
        return False
    return cache.is_contiguous() and block_table.stride(1) == 1


def paged_kv_block_gather(cache, block_table):
    """Select block DMA only when a full sequence fits in the Unified Buffer."""
    if _tv_eligible(cache, block_table):
        return tensorview_call(
            "PAGED_KV_BLOCK_GATHER", _tensorview, _generic, cache, block_table
        )
    return _generic(cache, block_table)

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

import torch
import triton

from flaggems_vllm.ops.cp_gather_indexer_k_quant_cache import (
    _cp_gather_indexer_quant_cache_kernel,
)


def cp_gather_indexer_k_quant_cache(
    k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen
):
    """MetaX launch policy for the byte gather kernel.

    MetaX measurements show one warp with a larger token tile is faster than the
    generic 32-token/implicit-warp launch.  Keep the kernel itself shared with
    the generic implementation so layout and semantics stay identical.
    """
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    quant_block_size = head_dim * 4 // k_fp8_scale.size(1)
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size

    k_cache_flat = k_cache.view(num_blocks, -1)
    k_cache_value = k_cache_flat[:, : block_size * head_dim]
    k_cache_scale = k_cache_flat[:, block_size * head_dim :].view(torch.float32)
    k_fp8 = k_fp8.view(torch.uint8)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    batch_size = block_table.shape[0]

    # Use the launch policy validated by the MetaX sweep and benchmark.
    # Reduce the tile for workloads exceeding 128K tokens.
    token_block = 64 if num_tokens > 131072 else 128
    batch_scan_size = triton.next_power_of_2(batch_size) if batch_size <= 16 else 1
    search_steps = (batch_size + 1).bit_length()
    grid = (triton.cdiv(num_tokens, token_block), num_quant_blocks)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        k_fp8.stride(0),
        num_quant_blocks,
        batch_size,
        head_dim,
        quant_block_size,
        token_block,
        batch_scan_size,
        search_steps,
        num_warps=1,
    )

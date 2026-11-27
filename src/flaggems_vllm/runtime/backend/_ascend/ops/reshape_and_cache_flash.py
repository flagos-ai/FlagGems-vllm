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

"""Ascend block-DMA scatter for packed non-quantized flash KV caches."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.reshape_and_cache_flash import (
    reshape_and_cache_flash as _generic,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    tile_fits_ub,
    tile_rows_for_ub,
)

TV_BLOCK_T = 32

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle
    from triton.experimental.tle import tile_load, tile_store

    @triton.jit
    def _reshape_and_cache_flash_tv_kernel(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_tokens,
        num_slots,
        key_stride,
        value_stride,
        ROW: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
    ):
        block = tl.program_id(0)
        tokens = block * BLOCK_T + tl.arange(0, BLOCK_T)
        slots = tl.load(slot_mapping + tokens, tokens < num_tokens, other=-1)
        slots = tl.where((slots >= 0) & (slots < num_slots), slots, -1).to(tl.int32)
        key_view = tle.make_partition_view(
            key,
            shape=[num_tokens, ROW],
            strides=[key_stride, 1],
            tile=[BLOCK_T, BLOCK_ROW],
        )
        value_view = tle.make_partition_view(
            value,
            shape=[num_tokens, ROW],
            strides=[value_stride, 1],
            tile=[BLOCK_T, BLOCK_ROW],
        )
        key_tile = tile_load(
            key_view, index=(block, 0), dst_space=tle.language.dsa.ascend.UB
        )
        value_tile = tile_load(
            value_view, index=(block, 0), dst_space=tle.language.dsa.ascend.UB
        )
        key_cache_view = tle.make_gather_scatter_view(
            key_cache,
            shape=[num_slots, ROW],
            strides=[ROW, 1],
            tile=[BLOCK_T, BLOCK_ROW],
            sparse_dim=[0],
            bounds_check=True,
        )
        value_cache_view = tle.make_gather_scatter_view(
            value_cache,
            shape=[num_slots, ROW],
            strides=[ROW, 1],
            tile=[BLOCK_T, BLOCK_ROW],
            sparse_dim=[0],
            bounds_check=True,
        )
        tile_store(
            key_cache_view,
            value=key_tile,
            index=(slots, 0),
            src_space=tle.language.dsa.ascend.UB,
        )
        tile_store(
            value_cache_view,
            value=value_tile,
            index=(slots, 0),
            src_space=tle.language.dsa.ascend.UB,
        )

    def _tensorview(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    ):
        num_tokens, heads, dim = key.shape
        row = heads * dim
        block_row = triton.next_power_of_2(row)
        block_t = tile_rows_for_ub(block_row * key.element_size(), 2, TV_BLOCK_T)
        with torch_device_fn.device(key.device):
            _reshape_and_cache_flash_tv_kernel[(triton.cdiv(num_tokens, block_t),)](
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                num_tokens,
                key_cache.shape[0] * key_cache.shape[1],
                key.stride(0),
                value.stride(0),
                ROW=row,
                BLOCK_T=block_t,
                BLOCK_ROW=block_row,
            )
        return None


def _tv_eligible(
    key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale
):
    if not HAS_TENSORVIEW or kv_cache_dtype != "auto":
        return False
    if key.ndim != 3 or value.ndim != 3:
        return False
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return False
    if key.shape != value.shape or key_cache.shape != value_cache.shape:
        return False
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != key.shape[0]:
        return False
    if key_cache.shape[2:] != key.shape[1:]:
        return False
    if key.numel() == 0 or slot_mapping.dtype not in (torch.int32, torch.int64):
        return False
    if key.dtype != value.dtype or key_cache.dtype != key.dtype:
        return False
    if value_cache.dtype != value.dtype:
        return False
    if any(
        t.device != key.device for t in (value, key_cache, value_cache, slot_mapping)
    ):
        return False
    row = key.shape[1] * key.shape[2]
    if key_cache.shape[1] == 0 or not tile_fits_ub(
        triton.next_power_of_2(row) * key.element_size(), 2
    ):
        return False
    if slot_mapping.stride(0) != 1:
        return False
    if any(t.stride(2) != 1 or t.stride(1) != key.shape[2] for t in (key, value)):
        return False
    for cache in (key_cache, value_cache):
        if cache.stride(3) != 1 or cache.stride(2) != key.shape[2]:
            return False
        if cache.stride(1) != row or cache.stride(0) != cache.shape[1] * row:
            return False
    return True


def reshape_and_cache_flash(
    key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale
):
    """Write packed flash KV cache; preserve the generic kernel for other layouts."""
    args = (
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    if _tv_eligible(*args):
        return tensorview_call("RESHAPE_AND_CACHE_FLASH", _tensorview, _generic, *args)
    return _generic(*args)

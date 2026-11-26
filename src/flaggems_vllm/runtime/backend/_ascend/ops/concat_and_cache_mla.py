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

"""Ascend block-DMA scatter for the non-quantized MLA cache."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.concat_and_cache_mla import (
    concat_and_cache_mla as _generic,
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
    def _concat_and_cache_mla_tv_kernel(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        num_tokens,
        num_slots,
        kv_stride,
        pe_stride,
        RANK: tl.constexpr,
        ROPE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        block = tl.program_id(0)
        tokens = block * BLOCK_T + tl.arange(0, BLOCK_T)
        slots = tl.load(slot_mapping + tokens, tokens < num_tokens, other=-1)
        slots = tl.where((slots >= 0) & (slots < num_slots), slots, -1).to(tl.int32)
        kv_view = tle.make_partition_view(
            kv_c,
            shape=[num_tokens, RANK],
            strides=[kv_stride, 1],
            tile=[BLOCK_T, RANK],
        )
        pe_view = tle.make_partition_view(
            k_pe,
            shape=[num_tokens, ROPE],
            strides=[pe_stride, 1],
            tile=[BLOCK_T, ROPE],
        )
        kv_tile = tile_load(
            kv_view, index=(block, 0), dst_space=tle.language.dsa.ascend.UB
        )
        pe_tile = tile_load(
            pe_view, index=(block, 0), dst_space=tle.language.dsa.ascend.UB
        )
        kv_cache_view = tle.make_gather_scatter_view(
            kv_cache,
            shape=[num_slots, RANK + ROPE],
            strides=[RANK + ROPE, 1],
            tile=[BLOCK_T, RANK],
            sparse_dim=[0],
            bounds_check=True,
        )
        pe_cache_view = tle.make_gather_scatter_view(
            kv_cache,
            shape=[num_slots, RANK + ROPE],
            strides=[RANK + ROPE, 1],
            tile=[BLOCK_T, ROPE],
            sparse_dim=[0],
            bounds_check=True,
        )
        tile_store(
            kv_cache_view,
            value=kv_tile,
            index=(slots, 0),
            src_space=tle.language.dsa.ascend.UB,
        )
        tile_store(
            pe_cache_view,
            value=pe_tile,
            index=(slots, RANK // ROPE),
            src_space=tle.language.dsa.ascend.UB,
        )

    def _tensorview(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale):
        num_tokens, rank = kv_c.shape
        rope = k_pe.shape[1]
        block_t = tile_rows_for_ub((rank + rope) * kv_c.element_size(), 1, TV_BLOCK_T)
        with torch_device_fn.device(kv_c.device):
            _concat_and_cache_mla_tv_kernel[(triton.cdiv(num_tokens, block_t),)](
                kv_c,
                k_pe,
                kv_cache,
                slot_mapping,
                num_tokens,
                kv_cache.shape[0] * kv_cache.shape[1],
                kv_c.stride(0),
                k_pe.stride(0),
                RANK=rank,
                ROPE=rope,
                BLOCK_T=block_t,
            )
        return None


def _tv_eligible(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale):
    if not HAS_TENSORVIEW or kv_cache_dtype != "auto":
        return False
    if kv_c.ndim != 2 or k_pe.ndim != 2 or kv_cache.ndim != 3:
        return False
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != kv_c.shape[0]:
        return False
    if k_pe.shape[0] != kv_c.shape[0] or slot_mapping.numel() == 0:
        return False
    if scale.numel() != 1 or slot_mapping.dtype not in (torch.int32, torch.int64):
        return False
    if kv_c.dtype != k_pe.dtype or kv_cache.dtype != kv_c.dtype:
        return False
    if any(t.device != kv_c.device for t in (k_pe, kv_cache, slot_mapping, scale)):
        return False
    rank, rope = kv_c.shape[1], k_pe.shape[1]
    if rope == 0 or rank == 0 or rank % rope != 0:
        return False
    if kv_cache.shape[1] == 0 or kv_cache.shape[2] != rank + rope:
        return False
    if not tile_fits_ub((rank + rope) * kv_c.element_size(), 1):
        return False
    return (
        kv_c.stride(1) == 1
        and k_pe.stride(1) == 1
        and slot_mapping.stride(0) == 1
        and kv_cache.stride(2) == 1
        and kv_cache.stride(1) == rank + rope
        and kv_cache.stride(0) == kv_cache.shape[1] * (rank + rope)
    )


def concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale):
    """Write MLA cache; retain generic FP8 and non-packed paths."""
    if _tv_eligible(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale):
        return tensorview_call(
            "CONCAT_AND_CACHE_MLA",
            _tensorview,
            _generic,
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            scale,
        )
    return _generic(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)

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

import logging
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.config import use_c_extension
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def reshape_and_cache_flash_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    slot_word_stride,
    block_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    k_scale,
    v_scale,
    n: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping + token_idx * slot_word_stride)
    if slot_idx < 0:
        return

    # slot words are always loaded as int32 (values are block indices bounded
    # by the KV cache size, < 2^31), so div/mod stay in the int32 domain;
    # widen only the block base to int64 so huge caches (> 2^31 elements)
    # keep 64-bit addressing.
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    i = tl.arange(0, triton.next_power_of_2(n))
    mask = i < n

    src_key_idx = token_idx * key_stride + i
    src_value_idx = token_idx * value_stride + i
    # The cache row [block_idx, block_offset] holds num_heads * head_size =
    # n contiguous elements, so i alone indexes it (no head decomposition).
    tgt_key_value_idx = block_idx.to(tl.int64) * block_stride + block_offset * n + i

    tgt_key = tl.load(key + src_key_idx, mask=mask)
    tgt_value = tl.load(value + src_value_idx, mask=mask)

    # TODO: support fp8 dtype
    tl.store(key_cache + tgt_key_value_idx, tgt_key, mask=mask)
    tl.store(value_cache + tgt_key_value_idx, tgt_value, mask=mask)


def _slot_word_stride() -> int:
    # torch_gcu without TORCH_GCU_ENABLE_INT64_AND_UINT64 keeps ATen int64
    # values as compact int32 words (element j lives at word j); with the gate
    # on, values sit in the low word of each 8-byte slot (word 2*j). Either
    # way, a Triton int64 load at the 8-byte dtype stride reads
    # neighbor-joined garbage, so slots are always read through a zero-copy
    # int32 view with the matching word stride.
    return 2 if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1" else 1


def _slot_words(slot_mapping):
    # Zero-copy int32 view of the slot indices (no extra cast kernel launch);
    # same pattern as the shipped indexer_k_quant_and_cache. Rare non-int32/
    # int64 dtypes keep the explicit cast.
    if slot_mapping.dtype == torch.int32:
        return slot_mapping, 1
    if slot_mapping.dtype != torch.int64:
        return slot_mapping.to(torch.int32), 1
    return slot_mapping.contiguous().view(torch.int32), _slot_word_stride()


def reshape_and_cache_flash(
    key,  # [num_tokens, num_heads, head_size]
    value,  # [num_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, block_size, num_heads, head_size]
    value_cache,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping,  # [num_tokens]
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    if use_c_extension:
        logger.debug("GEMS RESHAPE_AND_CACHE_FLASH(C EXTENSION)")
        torch.ops.flag_gems.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    else:
        logger.debug("GEMS RESHAPE_AND_CACHE_FLASH")
        num_tokens = slot_mapping.size(0)
        num_heads = key.size(1)
        head_size = key.size(2)
        block_size = key_cache.size(1)

        slot_mapping, slot_word_stride = _slot_words(slot_mapping)

        key_stride = key.stride(0)
        value_stride = value.stride(0)
        block_stride = key_cache.stride(0)
        assert key_cache.stride(0) == value_cache.stride(0)

        grid = (num_tokens,)
        with torch_device_fn.device(key.device):
            reshape_and_cache_flash_kernel[grid](
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                slot_word_stride,
                block_stride,
                key_stride,
                value_stride,
                num_heads,
                head_size,
                block_size,
                k_scale,
                v_scale,
                num_heads * head_size,
            )

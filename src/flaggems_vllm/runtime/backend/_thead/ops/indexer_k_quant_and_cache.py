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

# Adapted from vLLM v0.20.2:
# csrc/cache_kernels.cu::indexer_k_quant_and_cache_kernel

import torch
import triton
import triton.language as tl


# Token from FlagGems-vllm/src/flaggems_vllm/runtime/backend/_ascend/ops/per_token_group_quant_fp8.py
@triton.jit
def _f32_to_fp8_e4m3fn(y):
    b = y.to(tl.int32, bitcast=True)
    a = b & 0x7FFFFFFF
    t = a - 0x3C000000
    t += 0x0007FFFF + ((t >> 20) & 1)
    r_norm = t >> 20
    r_sub = (a.to(tl.float32, bitcast=True) * 512.0 + 8388608.0).to(
        tl.int32, bitcast=True
    ) - 0x4B000000
    r = tl.where(a >= 0x3C800000, r_norm, r_sub)
    return (r | ((b >> 24) & 0x80)).to(tl.uint8)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["QUANT_BLOCK_SIZE", "NUM_TOKENS_PAD"],
)
@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    slot_mapping_ptr,
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_quant_blocks,
    head_dim: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    NUM_TOKENS_PAD: tl.constexpr,
):
    tid = tl.program_id(0)
    quant_block_id = tl.program_id(1) * 4
    quant_block_offsets = tl.arange(0, 4)
    head_offsets = tl.arange(0, QUANT_BLOCK_SIZE)
    offsets = (
        quant_block_id + quant_block_offsets[:, None]
    ) * QUANT_BLOCK_SIZE + head_offsets[None, :]
    mask = offsets < head_dim

    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return

    block_id = slot_id // block_size
    block_offset = slot_id % block_size

    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    amax = tl.max(tl.abs(val).to(tl.float32), axis=1)
    scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = _f32_to_fp8_e4m3fn(val.to(tl.float32) / scale[:, None])
    dst_ptr = kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
    tl.store(dst_ptr + offsets, fp8_val, mask=mask)

    dst_scale_ptr = (
        kv_cache_scale_ptr
        + block_id * kv_cache_scale_stride
        + block_offset * num_quant_blocks
        + quant_block_id
    )
    scale_mask = quant_block_id + quant_block_offsets < num_quant_blocks
    tl.store(dst_scale_ptr + quant_block_offsets, scale, mask=scale_mask)


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size

    kv_cache_flat = kv_cache.view(num_blocks, -1)
    # replace torch.float8_e4m3fn with torch.uint8 to avoid the following CompilationError:
    #     "type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')"
    kv_cache_value = kv_cache_flat[:, : block_size * head_dim].view(torch.uint8)
    kv_cache_scale = kv_cache_flat[:, block_size * head_dim :].view(torch.float32)
    num_tokens_pad = triton.next_power_of_2(num_tokens)
    _indexer_k_quant_and_cache_kernel[(num_tokens, triton.cdiv(num_quant_blocks, 4))](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_quant_blocks,
        head_dim,
        quant_block_size,
        USE_UE8M0=scale_fmt == "ue8m0",
        NUM_TOKENS_PAD=num_tokens_pad,
    )

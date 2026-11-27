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

"""Ascend launch bounds, fused statistics, and exact uint32 sampling helpers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops import rejection_sampler as common
from flaggems_vllm.ops.rejection_sampler import _argmax_block, _combine_lse

# Ascend 910B rejects launches whose total program count exceeds coreDim.
_MAX_LAUNCH_PROGRAMS = 65535


@triton.jit
def _direct_lse(
    local_max_ptr,
    local_max_stride,
    local_sum_ptr,
    local_sum_stride,
    row,
    nblocks,
    PADDED: tl.constexpr,
    direct_logits_ptr,
    direct_valid,
    vocab_size,
    temperature,
    DIRECT_BLOCK: tl.constexpr,
):
    cols = tl.arange(0, DIRECT_BLOCK)
    if PADDED == 1:
        logits = (
            tl.load(
                direct_logits_ptr + cols,
                mask=(cols < vocab_size) & direct_valid,
                other=float("-inf"),
            ).to(tl.float32)
            / temperature
        )
        local_max = tl.max(logits, axis=0)
        local_sum = tl.where(
            local_max > float("-inf"), tl.sum(tl.exp(logits - local_max)), 0.0
        )
        return local_max + tl.log(local_sum * tl.exp(local_max - local_max))
    else:
        blocks = tl.arange(0, PADDED)
        cols = blocks[:, None] * DIRECT_BLOCK + cols[None, :]
        logits = (
            tl.load(
                direct_logits_ptr + cols,
                mask=(cols < vocab_size) & direct_valid,
                other=float("-inf"),
            ).to(tl.float32)
            / temperature
        )
        local_max = tl.max(logits, axis=1)
        local_sum = tl.where(
            local_max > float("-inf"),
            tl.sum(tl.exp(logits - local_max[:, None]), axis=1),
            0.0,
        )
        return _combine_lse(local_max, local_sum)


@triton.jit
def _direct_argmax(
    local_max_ptr,
    local_max_stride,
    local_argmax_ptr,
    local_argmax_stride,
    row,
    nblocks,
    PADDED: tl.constexpr,
    direct_logits_ptr,
    vocab_size,
    DIRECT_BLOCK: tl.constexpr,
):
    cols = tl.arange(0, DIRECT_BLOCK)
    if PADDED == 1:
        logits = tl.load(
            direct_logits_ptr + cols, mask=cols < vocab_size, other=float("-inf")
        ).to(tl.float32)
        (_, index) = tl.max(logits, axis=0, return_indices=True)
        return index.to(tl.int64)
    else:
        blocks = tl.arange(0, PADDED)
        cols = blocks[:, None] * DIRECT_BLOCK + cols[None, :]
        logits = tl.load(
            direct_logits_ptr + cols, mask=cols < vocab_size, other=float("-inf")
        ).to(tl.float32)
        (values, indices) = tl.max(logits, axis=1, return_indices=True)
        block = _argmax_block(values)
        index = tl.sum(tl.where(blocks == block, indices, 0), axis=0)
        return block.to(tl.int64) * DIRECT_BLOCK + index.to(tl.int64)


@triton.jit
def _wrap32(value):
    return value.to(tl.uint32)


@triton.jit
def _mul32(value, constant: tl.constexpr):
    factor = tl.full((), constant, tl.uint32)
    return (value.to(tl.uint32) * factor).to(tl.uint32)


@triton.jit
def _rshift32(value, shift: tl.constexpr):
    # Vector uint32 right shifts lower arithmetically on this compiler.
    # Signed bitcast plus a positive low-bit mask preserves the logical shift.
    signed = value.to(tl.int32, bitcast=True)
    return ((signed >> shift) & ((1 << (32 - shift)) - 1)).to(tl.uint32)


@triton.jit
def _rotl32(value, shift: tl.constexpr):
    value = value.to(tl.uint32)
    left = (value << shift).to(tl.uint32)
    right = _rshift32(value, 32 - shift)
    return (left | right).to(tl.uint32)


@triton.jit
def _mix32(value, key):
    key = _mul32(key, 0xCC9E2D51)
    key = _rotl32(key, 15)
    key = _mul32(key, 0x1B873593)
    value = _wrap32(value ^ key)
    value = _rotl32(value, 13)
    addend = tl.full((), 0xE6546B64, tl.uint32)
    return (_mul32(value, 5) + addend).to(tl.uint32)


@triton.jit
def _fmix32(value):
    value = _wrap32(value ^ _rshift32(value, 16))
    value = _mul32(value, 0x85EBCA6B)
    value = _wrap32(value ^ _rshift32(value, 13))
    value = _mul32(value, 0xC2B2AE35)
    return _wrap32(value ^ _rshift32(value, 16))


@triton.jit
def _hash32(seed, pos, offset):
    seed = seed.to(tl.int64)
    pos = pos.to(tl.int64)
    offset = offset.to(tl.uint32)
    value = tl.zeros_like(offset)
    value = _mix32(value, seed & 0xFFFFFFFF)
    value = _mix32(value, seed >> 32 & 0xFFFFFFFF)
    value = _mix32(value, pos & 0xFFFFFFFF)
    return _fmix32(_wrap32(_mix32(value, offset) ^ 16))


@triton.jit
def _uniform(seed, pos, offset):
    """Murmur3 midpoint uniform from the frozen vLLM Gumbel source."""
    value = _hash32(seed, pos, offset)
    high = _rshift32(value, 16) & 65535
    low = value & 65535
    high = high.to(tl.int32)
    low = low.to(tl.int32)
    return (
        high.to(tl.float32) * 1.52587890625e-05
        + (low.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )


def _launch_kernel(kernel, grid, *args, **kwargs):
    return common._launch_kernel(
        kernel, grid, *args, max_programs=_MAX_LAUNCH_PROGRAMS, **kwargs
    )


def _select_kernels(vocab_size):
    direct = triton.cdiv(vocab_size, common._VOCAB_BLOCK) <= 2
    # Keep 1024-column reduction groups for two blocks, including masked/NaN
    # behavior. Only the single-block vector shrinks with the vocabulary.
    return (
        direct,
        (
            triton.next_power_of_2(min(vocab_size, common._VOCAB_BLOCK))
            if direct
            else common._VOCAB_BLOCK
        ),
        _direct_lse if direct else common._global_lse,
        _direct_argmax if direct else common._global_argmax,
        _uniform,
    )


def rejection_sample(
    target_logits: torch.Tensor,
    draft_logits_cache,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    synthetic_conditional_rates=None,
    use_fp64: bool = False,
    use_block_verification: bool = False,
    contexts=None,
    watermarking=None,
    watermarking_skip_mask=None,
    watermark_key=None,
    fly_window_size: int = 0,
    fly_entropy_threshold: float = 0.3,
    fly_entropy_top_k: int = 3,
    skip_masked_vocab_blocks: bool = False,
):
    """Verify and sample an ordinary chain with Ascend-specific helpers."""
    return common._rejection_sample(
        target_logits,
        draft_logits_cache,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps,
        synthetic_conditional_rates,
        use_fp64,
        use_block_verification,
        contexts,
        watermarking,
        watermarking_skip_mask,
        watermark_key,
        fly_window_size,
        fly_entropy_threshold,
        fly_entropy_top_k,
        skip_masked_vocab_blocks,
        launch_kernel=_launch_kernel,
        select_kernels=_select_kernels,
    )


__all__ = ["rejection_sample"]

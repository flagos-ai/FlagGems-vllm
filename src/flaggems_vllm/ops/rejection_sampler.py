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

"""Triton implementation of vLLM's ordinary chain rejection sampler.

The implementation deliberately keeps the public surface identical to vLLM's
``rejection_sample`` callable.  The verification and residual draw are done in
Triton; the host wrapper only validates metadata, allocates output/scratch and
rejects optional vLLM modes which are outside the ordinary chain contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_VOCAB_BLOCK = 1024
_MIN_UNIFORM = tl.constexpr(4.6566127342e-10)


def _configs():
    # NVIDIA has a checked-in tune table.  The small fallback keeps import and
    # compilation usable on vendors which do not ship a table for this new op.
    configs = runtime.get_tuned_config("rejection_sample")
    if configs:
        return configs
    return [triton.Config({"BLOCK_SIZE": _VOCAB_BLOCK}, num_warps=4, num_stages=2)]


def _launch_kernel(kernel, grid, *args, max_programs=None, **kwargs):
    """Optionally bound a launch while preserving global row/request identities."""
    num_programs = grid[0]
    if len(grid) == 2:
        num_programs *= grid[1]
    split = max_programs is not None and num_programs > max_programs
    if len(grid) == 2:
        kwargs["FLAT_GRID"] = split
    if not split:
        kernel[grid](*args, program_offset=0, **kwargs)
        return
    # Flatten before chunking: bounding only rows can still exceed coreDim
    # when there are multiple vocabulary blocks (including >65535 blocks).
    for offset in range(0, num_programs, max_programs):
        count = min(max_programs, num_programs - offset)
        kernel[(count,)](*args, program_offset=offset, **kwargs)


@triton.jit
def _combine_lse(maxes, sums):
    max_value = tl.max(maxes, axis=0)
    # Keep fully masked rows' NaN propagation in the original block merge.
    return max_value + tl.log(tl.sum(sums * tl.exp(maxes - max_value)))


@triton.jit
def _argmax_block(values):
    values = tl.where(values != values, float("-inf"), values)
    return tl.argmax(values, axis=0)


@triton.jit
def _global_lse(
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
    blocks = tl.arange(0, PADDED)
    mask = blocks < nblocks
    maxes = tl.load(
        local_max_ptr + row * local_max_stride + blocks, mask=mask, other=float("-inf")
    )
    sums = tl.load(
        local_sum_ptr + row * local_sum_stride + blocks, mask=mask, other=0.0
    )
    return _combine_lse(maxes, sums)


@triton.jit
def _global_argmax(
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
    blocks = tl.arange(0, PADDED)
    mask = blocks < nblocks
    values = tl.load(
        local_max_ptr + row * local_max_stride + blocks, mask=mask, other=float("-inf")
    )
    block = _argmax_block(values)
    return tl.load(local_argmax_ptr + row * local_argmax_stride + block).to(tl.int64)


@triton.jit
def _wrap32(value):
    return value & 0xFFFFFFFF


@triton.jit
def _mul32(value, constant: tl.constexpr):
    return value * constant & 0xFFFFFFFF


@triton.jit
def _rshift32(value, shift: tl.constexpr):
    return value >> shift


@triton.jit
def _rotl32(value, shift: tl.constexpr):
    return (value << shift | value >> 32 - shift) & 0xFFFFFFFF


@triton.jit
def _mix32(value, key):
    key = _mul32(key, 0xCC9E2D51)
    key = _rotl32(key, 15)
    key = _mul32(key, 0x1B873593)
    value = _wrap32(value ^ key)
    value = _rotl32(value, 13)
    return value * 5 + 0xE6546B64 & 0xFFFFFFFF


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
    offset = offset.to(tl.int64)
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
    return (
        high.to(tl.float32) * 1.52587890625e-05
        + (low.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )


@triton.jit
def _tl_rand32(seed, pos):
    # vLLM's ordinary chain uses tl_rand32(seed, absolute_pos) for the ratio
    # test; the residual Gumbel stream is the separate frozen Murmur3 stream.
    return tl.maximum(tl.rand(seed, pos), _MIN_UNIFORM)


@triton.jit
def _log1p_neg_stable(value):
    # Match vLLM's fp32 Gumbel helper without depending on a vendor-specific
    # ``log1p`` extension.
    series = 1.0 / 8.0
    series = 1.0 / 7.0 + value * series
    series = 1.0 / 6.0 + value * series
    series = 1.0 / 5.0 + value * series
    series = 1.0 / 4.0 + value * series
    series = 1.0 / 3.0 + value * series
    series = 1.0 / 2.0 + value * series
    series = 1.0 + value * series
    return tl.where(
        value < 0.25,
        -value * series,
        tl.log(tl.maximum(1.0 - value, 5.960464477539063e-8)),
    )


@triton.jit
def _gumbel(logits, token_ids, seed, pos, mask, UNIFORM_FN: tl.constexpr):
    # The selected helper preserves the frozen Murmur3 midpoint stream.
    u = UNIFORM_FN(seed, pos, token_ids)
    gumbel = -tl.log(-_log1p_neg_stable(u))
    return tl.where(mask, logits + gumbel, float("-inf"))


@triton.jit
def _residual_log1p_neg(value):
    # Residual construction follows vLLM's log1p(-ratio) expression.  Keep it
    # separate from the Gumbel-tail polynomial above; clamping here changes the
    # residual law when p and q are nearly equal.
    return tl.log(1.0 - value)


@libentry()
@libtuner(
    configs=_configs(), key=["vocab_size"], strategy=["log"], use_cuda_graph=False
)
@triton.jit(do_not_specialize=["num_logits", "program_offset"])
def _compute_local_stats_kernel(
    target_local_argmax_ptr,
    target_local_max_ptr,
    target_local_sum_ptr,
    draft_local_max_ptr,
    draft_local_sum_ptr,
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    draft_row_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    temperature_ptr,
    vocab_size,
    num_logits,
    nblocks,
    program_offset,
    FLAT_GRID: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    HAS_DRAFT_ROW_MAPPING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if FLAT_GRID:
        program = tl.program_id(0).to(tl.int64) + program_offset
        row = program // nblocks
        block_id = program % nblocks
    else:
        row = tl.program_id(0).to(tl.int64)
        block_id = tl.program_id(1).to(tl.int64)
    cols = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < vocab_size
    logits = tl.load(
        target_logits_ptr + row * target_stride + cols, mask=mask, other=float("-inf")
    ).to(tl.float32)
    temp = tl.load(temperature_ptr + tl.load(expanded_idx_mapping_ptr + row)).to(
        tl.float32
    )
    next_local_pos = tl.load(
        expanded_local_pos_ptr + row + 1, mask=row + 1 < num_logits, other=0
    )
    is_bonus = next_local_pos == 0
    if temp == 0.0:
        value, index = tl.max(logits, axis=0, return_indices=True)
        tl.store(
            target_local_argmax_ptr + row * nblocks + block_id,
            block_id * BLOCK_SIZE + index,
        )
        tl.store(target_local_max_ptr + row * nblocks + block_id, value)
        return

    local_max = tl.max(logits, axis=0)
    local_sum = tl.where(
        local_max > float("-inf"), tl.sum(tl.exp(logits - local_max)), 0.0
    )
    tl.store(target_local_max_ptr + row * nblocks + block_id, local_max)
    tl.store(target_local_sum_ptr + row * nblocks + block_id, local_sum)
    if HAS_DRAFT_LOGITS:
        if not is_bonus:
            req = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
            step = tl.load(expanded_local_pos_ptr + row).to(tl.int64)
            cache_row = req
            if HAS_DRAFT_ROW_MAPPING:
                cache_row = tl.load(
                    draft_row_mapping_ptr + req, mask=req >= 0, other=-1
                ).to(tl.int64)
            draft = (
                tl.load(
                    draft_logits_ptr
                    + cache_row * draft_stride_0
                    + step * draft_stride_1
                    + cols,
                    mask=mask & (cache_row >= 0),
                    other=float("-inf"),
                ).to(tl.float32)
                / temp
            )
            draft_max = tl.max(draft, axis=0)
            draft_sum = tl.where(
                draft_max > float("-inf"), tl.sum(tl.exp(draft - draft_max)), 0.0
            )
            tl.store(draft_local_max_ptr + row * nblocks + block_id, draft_max)
            tl.store(draft_local_sum_ptr + row * nblocks + block_id, draft_sum)


@libentry()
@libtuner(
    configs=_configs(), key=["vocab_size"], strategy=["log"], use_cuda_graph=False
)
@triton.jit(do_not_specialize=["num_logits", "program_offset"])
def _rejection_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    target_logits_ptr,
    target_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sum_ptr,
    target_local_sum_stride,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sum_ptr,
    draft_local_sum_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    draft_row_mapping_ptr,
    draft_sampled_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    temperature_ptr,
    seed_ptr,
    pos_ptr,
    vocab_size,
    nblocks,
    num_logits,
    num_speculative_steps,
    program_offset,
    HAS_DRAFT_LOGITS: tl.constexpr,
    HAS_DRAFT_ROW_MAPPING: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
    DRAW_BLOCK: tl.constexpr,
    LSE_FN: tl.constexpr,
    ARGMAX_FN: tl.constexpr,
    UNIFORM_FN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64) + program_offset
    req_state = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    draft_count = end - start - 1
    temp = tl.load(temperature_ptr + req_state).to(tl.float32)
    seed = tl.load(seed_ptr + req_state)

    accepted_count = tl.zeros((), tl.int32)
    rejected = tl.full((), 0, tl.int1)

    for step in range(num_speculative_steps):
        active = step < draft_count
        row = start + step
        token = tl.load(draft_sampled_ptr + row + 1, mask=active, other=-1).to(tl.int64)
        valid = active & (token >= 0) & (token < vocab_size)
        safe_token = tl.maximum(token, 0)
        if not rejected and active:
            target_argmax = tl.zeros((), tl.int64)
            if temp == 0.0:
                target_argmax = ARGMAX_FN(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    row,
                    nblocks,
                    PADDED_BLOCKS,
                    target_logits_ptr + row * target_stride,
                    vocab_size,
                    DRAW_BLOCK,
                )
                accepted = valid & (safe_token == target_argmax)
                out_token = tl.where(accepted, safe_token, target_argmax)
            else:
                target_lse = LSE_FN(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sum_ptr,
                    target_local_sum_stride,
                    row,
                    nblocks,
                    PADDED_BLOCKS,
                    target_logits_ptr + row * target_stride,
                    True,
                    vocab_size,
                    1.0,
                    DRAW_BLOCK,
                )
                target_logit = tl.load(
                    target_logits_ptr + row * target_stride + safe_token,
                    mask=valid,
                    other=float("-inf"),
                ).to(tl.float32)
                target_logprob = target_logit - target_lse
                if HAS_DRAFT_LOGITS:
                    cache_row = req_state
                    if HAS_DRAFT_ROW_MAPPING:
                        cache_row = tl.load(
                            draft_row_mapping_ptr + req_state,
                            mask=req_state >= 0,
                            other=-1,
                        ).to(tl.int64)
                    draft_logit = (
                        tl.load(
                            draft_logits_ptr
                            + cache_row * draft_stride_0
                            + step * draft_stride_1
                            + safe_token,
                            mask=valid & (cache_row >= 0),
                            other=float("-inf"),
                        ).to(tl.float32)
                        / temp
                    )
                    draft_lse = LSE_FN(
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sum_ptr,
                        draft_local_sum_stride,
                        row,
                        nblocks,
                        PADDED_BLOCKS,
                        draft_logits_ptr
                        + cache_row * draft_stride_0
                        + step * draft_stride_1,
                        cache_row >= 0,
                        vocab_size,
                        temp,
                        DRAW_BLOCK,
                    )
                    draft_logprob = draft_logit - draft_lse
                else:
                    draft_logprob = tl.zeros((), tl.float32)
                u = _tl_rand32(seed, tl.load(pos_ptr + row))
                accepted = valid & (target_logprob > tl.log(u) + draft_logprob)
                out_token = safe_token

            tl.store(sampled_ptr + req_idx * sampled_stride + step, out_token)
            if accepted:
                accepted_count += 1
            else:
                rejected = True

    # A rejection samples from the residual at the first rejected position;
    # otherwise the final target row supplies the bonus token.
    resample_step = accepted_count
    resample_row = start + resample_step
    is_bonus = resample_step >= draft_count
    draw_pos = tl.load(pos_ptr + resample_row)
    best_value = tl.full((), float("-inf"), tl.float32)
    best_token = tl.zeros((), tl.int64)
    rejected_token = tl.load(
        draft_sampled_ptr + resample_row + 1, mask=not is_bonus, other=-1
    ).to(tl.int64)
    cache_row = req_state
    if HAS_DRAFT_ROW_MAPPING:
        cache_row = tl.load(
            draft_row_mapping_ptr + req_state, mask=req_state >= 0, other=-1
        ).to(tl.int64)
    target_lse = tl.zeros((), tl.float32)
    if temp != 0.0:
        target_lse = LSE_FN(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sum_ptr,
            target_local_sum_stride,
            resample_row,
            nblocks,
            PADDED_BLOCKS,
            target_logits_ptr + resample_row * target_stride,
            True,
            vocab_size,
            1.0,
            DRAW_BLOCK,
        )
    draft_lse = tl.zeros((), tl.float32)
    if HAS_DRAFT_LOGITS:
        if not is_bonus and temp != 0.0:
            draft_lse = LSE_FN(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sum_ptr,
                draft_local_sum_stride,
                resample_row,
                nblocks,
                PADDED_BLOCKS,
                draft_logits_ptr
                + cache_row * draft_stride_0
                + resample_step * draft_stride_1,
                cache_row >= 0,
                vocab_size,
                temp,
                DRAW_BLOCK,
            )
    if temp == 0.0 and is_bonus:
        best_token = ARGMAX_FN(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_argmax_ptr,
            target_local_argmax_stride,
            resample_row,
            nblocks,
            PADDED_BLOCKS,
            target_logits_ptr + resample_row * target_stride,
            vocab_size,
            DRAW_BLOCK,
        )

    # Single-block direct statistics use only the lanes needed by the vocab.
    # The block stride and general path remain the configured BLOCK_SIZE.
    draw_block: tl.constexpr = DRAW_BLOCK
    for block_id in range(0, nblocks):
        cols = block_id * BLOCK_SIZE + tl.arange(0, draw_block)
        mask = cols < vocab_size
        target = tl.load(
            target_logits_ptr + resample_row * target_stride + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        if temp == 0.0 and not is_bonus:
            # The greedy rejection value was written in the verification loop.
            residual = target
        elif is_bonus or not (rejected_token >= 0):
            residual = target
        elif HAS_DRAFT_LOGITS:
            draft = (
                tl.load(
                    draft_logits_ptr
                    + cache_row * draft_stride_0
                    + resample_step * draft_stride_1
                    + cols,
                    mask=mask & (cache_row >= 0),
                    other=float("-inf"),
                ).to(tl.float32)
                / temp
            )
            target_lp = target - target_lse
            draft_lp = draft - draft_lse
            ratio = tl.exp(draft_lp - target_lp)
            residual = tl.where(
                ratio < 1.0,
                target_lp + _residual_log1p_neg(ratio),
                float("-inf"),
            )
        else:
            residual = tl.where(cols != rejected_token, target, float("-inf"))
        noisy = _gumbel(residual, cols.to(tl.int64), seed, draw_pos, mask, UNIFORM_FN)
        local_value, local_index = tl.max(noisy, axis=0, return_indices=True)
        local_token = (block_id * BLOCK_SIZE + local_index).to(tl.int64)
        if temp != 0.0 and local_value > best_value:
            best_value = local_value
            best_token = local_token

    if temp == 0.0 and not is_bonus:
        # The rejection loop stored target_argmax at resample_step.
        pass
    else:
        tl.store(sampled_ptr + req_idx * sampled_stride + accepted_count, best_token)
    tl.store(num_sampled_ptr + req_idx, accepted_count + 1)


def _rejection_sample(
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
    *,
    launch_kernel,
    select_kernels,
):
    """Verify an ordinary speculative chain and sample its first recovery.

    Optional vLLM block/FLy/synthetic/watermark paths are intentionally rejected
    until they have an independent contract and implementation.

    Target logits already include vLLM sampling-parameter processing, including
    temperature. Draft cache values are pre-temperature. Integer metadata is
    contiguous and its device-resident values must follow the packed caller
    contract: bounded, monotone cu offsets and in-range request/cache indices.
    Only the prefix up to each num_sampled result is initialized in sampled.
    """
    unsupported = (
        synthetic_conditional_rates is not None,
        use_block_verification,
        contexts is not None,
        watermarking is not None,
        watermarking_skip_mask is not None,
        watermark_key is not None,
        fly_window_size != 0,
        skip_masked_vocab_blocks,
        use_fp64,
    )
    if any(unsupported):
        raise NotImplementedError(
            "rejection_sample supports only ordinary chain verification; "
            "block, FLy, synthetic and watermark modes are unsupported"
        )
    if target_logits.ndim != 2 or target_logits.stride(-1) != 1:
        raise ValueError("target_logits must be rank-2 with unit-stride vocabulary")
    if draft_sampled.ndim != 1 or cu_num_logits.ndim != 1:
        raise ValueError("draft_sampled and cu_num_logits must be one-dimensional")
    if num_speculative_steps < 0:
        raise ValueError("num_speculative_steps must be non-negative")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if target_logits.dtype not in supported_dtypes:
        raise TypeError("target_logits must have float16, bfloat16 or float32 dtype")
    if target_logits.device.type == "cpu":
        raise NotImplementedError("rejection_sample requires an accelerator")
    if cu_num_logits.numel() == 0:
        raise ValueError("cu_num_logits must contain the request sentinel")
    if any(
        tensor.ndim != 1
        for tensor in (
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
        )
    ):
        raise ValueError("rejection_sample metadata tensors must be one-dimensional")
    integer_metadata = (
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        seed,
    )
    if any(
        tensor.dtype not in (torch.int32, torch.int64) for tensor in integer_metadata
    ):
        raise TypeError(
            "rejection_sample index and seed metadata must be int32 or int64"
        )
    if temperature.dtype not in supported_dtypes:
        raise TypeError("temperature must have float16, bfloat16 or float32 dtype")
    if any(tensor.stride(0) != 1 for tensor in integer_metadata + (temperature,)):
        raise ValueError("rejection_sample metadata tensors must be contiguous")
    if not (
        target_logits.device
        == draft_sampled.device
        == cu_num_logits.device
        == pos.device
        == idx_mapping.device
        == expanded_idx_mapping.device
        == expanded_local_pos.device
        == temperature.device
        == seed.device
    ):
        raise ValueError("all rejection_sample tensors must share a device")

    draft_logits = None
    row_mapping = None
    if draft_logits_cache is not None:
        draft_logits = draft_logits_cache.values
        row_mapping = getattr(draft_logits_cache, "row_mapping", None)
        if draft_logits.ndim != 3 or draft_logits.stride(-1) != 1:
            raise ValueError(
                "draft logits cache must be rank-3 with unit-stride vocabulary"
            )
        if draft_logits.dtype not in supported_dtypes:
            raise TypeError("draft cache must have float16, bfloat16 or float32 dtype")
        if draft_logits.device != target_logits.device:
            raise ValueError("draft logits cache must share target_logits' device")
        if row_mapping is not None and row_mapping.device != target_logits.device:
            raise ValueError("draft row mapping must share target_logits' device")
        if draft_logits.shape[1] < num_speculative_steps:
            raise ValueError("draft logits cache has fewer steps than requested")
        if row_mapping is not None and (
            row_mapping.ndim != 1 or row_mapping.numel() < temperature.numel()
        ):
            raise ValueError("draft row mapping must cover request-state slots")
        if row_mapping is not None and row_mapping.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("draft row mapping must be int32 or int64")
        if row_mapping is not None and row_mapping.stride(0) != 1:
            raise ValueError("draft row mapping must be contiguous")
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    if num_logits == 0:
        raise ValueError("empty packed logits are unsupported")
    if draft_logits is not None:
        vocab_size = min(vocab_size, draft_logits.shape[-1])
    if vocab_size <= 0:
        raise ValueError("vocabulary must be non-empty")
    if num_reqs < 0:
        raise ValueError("cu_num_logits must have at least one entry")
    if num_reqs == 0:
        raise ValueError("empty request batches are unsupported")
    # Packed offset values are consumed directly by Triton.  Reading their
    # device values on the host would synchronize the production path; callers
    # must provide the frozen cu-offset contract (last offset <= num_logits and
    # draft_sampled length covering it).
    if draft_sampled.numel() < num_logits or pos.numel() < num_logits:
        raise ValueError("draft_sampled and pos must cover packed logits rows")
    if (
        expanded_idx_mapping.numel() < num_logits
        or expanded_local_pos.numel() < num_logits
    ):
        raise ValueError("expanded metadata is shorter than target_logits")
    if (
        idx_mapping.numel() < num_reqs
        or temperature.numel() < num_reqs
        or seed.numel() < temperature.numel()
    ):
        raise ValueError(
            "request metadata must cover the batch and request-state slots"
        )
    if (
        row_mapping is None
        and draft_logits is not None
        and draft_logits.shape[0] < temperature.numel()
    ):
        raise ValueError("replicated draft cache must cover request-state slots")

    num_blocks = triton.cdiv(vocab_size, _VOCAB_BLOCK)
    padded_blocks = triton.next_power_of_2(num_blocks)
    direct_stats, draw_block, lse_fn, argmax_fn, uniform_fn = select_kernels(vocab_size)
    target_local_argmax = None
    target_local_max = None
    target_local_sum = None
    draft_local_max = None
    draft_local_sum = None
    if not direct_stats:
        target_local_argmax = torch.empty(
            (num_logits, num_blocks), dtype=torch.int64, device=target_logits.device
        )
        target_local_max = torch.empty(
            (num_logits, num_blocks), dtype=torch.float32, device=target_logits.device
        )
        target_local_sum = torch.empty_like(target_local_max)
        draft_local_max = torch.empty_like(target_local_max)
        draft_local_sum = torch.empty_like(target_local_max)
        launch_kernel(
            _compute_local_stats_kernel,
            (num_logits, num_blocks),
            target_local_argmax,
            target_local_max,
            target_local_sum,
            draft_local_max,
            draft_local_sum,
            target_logits,
            target_logits.stride(0),
            draft_logits,
            draft_logits.stride(0) if draft_logits is not None else 0,
            draft_logits.stride(1) if draft_logits is not None else 0,
            row_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            vocab_size,
            num_logits,
            num_blocks,
            HAS_DRAFT_LOGITS=draft_logits is not None,
            HAS_DRAFT_ROW_MAPPING=row_mapping is not None,
        )
    sampled = torch.empty(
        (num_reqs, num_speculative_steps + 1),
        dtype=torch.int64,
        device=target_logits.device,
    )
    num_sampled = torch.empty(
        (num_reqs,), dtype=torch.int32, device=target_logits.device
    )
    launch_kernel(
        _rejection_kernel,
        (num_reqs,),
        sampled,
        sampled.stride(0),
        num_sampled,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        num_blocks,
        target_local_max,
        num_blocks,
        target_local_sum,
        num_blocks,
        draft_local_max,
        num_blocks,
        draft_local_sum,
        num_blocks,
        draft_logits,
        draft_logits.stride(0) if draft_logits is not None else 0,
        draft_logits.stride(1) if draft_logits is not None else 0,
        row_mapping,
        draft_sampled,
        cu_num_logits,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        pos,
        vocab_size,
        num_blocks,
        num_logits,
        num_speculative_steps,
        HAS_DRAFT_LOGITS=draft_logits is not None,
        HAS_DRAFT_ROW_MAPPING=row_mapping is not None,
        PADDED_BLOCKS=padded_blocks,
        DRAW_BLOCK=draw_block,
        LSE_FN=lse_fn,
        ARGMAX_FN=argmax_fn,
        UNIFORM_FN=uniform_fn,
    )
    return sampled, num_sampled


def _select_kernels(vocab_size):
    return False, _VOCAB_BLOCK, _global_lse, _global_argmax, _uniform


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
    """Verify and sample an ordinary chain using the shared two-stage path."""
    return _rejection_sample(
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

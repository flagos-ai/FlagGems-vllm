# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""Stride-aware scalar and vector trace replay for Ascend."""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _trace_replay_ascend_kernel(
    sampled_ptr,
    sampled_stride,
    idx_mapping_ptr,
    idx_mapping_stride,
    trace_token_ids_ptr,
    trace_token_ids_stride,
    trace_token_ids_col_stride,
    trace_len_ptr,
    trace_len_stride,
    total_len_ptr,
    total_len_stride,
    prompt_len_ptr,
    prompt_len_stride,
    num_states,
    trace_width,
    num_reqs,
    BLOCK: tl.constexpr,
    OFFSET_DTYPE: tl.constexpr,
):
    if BLOCK == 1:
        # FLIR rejects dynamic AddPtrOp offsets on singleton tensor pointers.
        batch_idx = tl.program_id(0).to(OFFSET_DTYPE)
        req_idx = tl.load(idx_mapping_ptr + batch_idx * idx_mapping_stride)
        if (req_idx >= 0) & (req_idx < num_states):
            state_idx = req_idx.to(OFFSET_DTYPE)
            req_trace_len = tl.load(trace_len_ptr + state_idx * trace_len_stride)
            total = tl.load(total_len_ptr + state_idx * total_len_stride)
            prompt = tl.load(prompt_len_ptr + state_idx * prompt_len_stride)
            step = total - prompt
            if (step >= 0) & (step < req_trace_len) & (step < trace_width):
                token_id = tl.load(
                    trace_token_ids_ptr
                    + state_idx * trace_token_ids_stride
                    + step.to(OFFSET_DTYPE) * trace_token_ids_col_stride
                )
                tl.store(
                    sampled_ptr + batch_idx * sampled_stride, token_id.to(tl.int64)
                )
    else:
        # Mask the tail rather than pulling it back: overlapping tile ranges
        # intermittently dropped writes on 910B4-1, even with an ownership mask.
        batch_idx = tl.program_id(0).to(OFFSET_DTYPE) * BLOCK + tl.arange(0, BLOCK).to(
            OFFSET_DTYPE
        )
        active = batch_idx < num_reqs
        req_idx = tl.load(idx_mapping_ptr + batch_idx * idx_mapping_stride, active, -1)
        valid = active & (req_idx >= 0) & (req_idx < num_states)

        # FlagTree may lower indirect masked loads to a gather followed by
        # select. Keep every gather address in bounds even when the lane is
        # masked out.
        req_idx = tl.where(valid, req_idx, 0).to(OFFSET_DTYPE)
        req_trace_len = tl.load(trace_len_ptr + req_idx * trace_len_stride, valid, 0)
        total = tl.load(total_len_ptr + req_idx * total_len_stride, valid, 0)
        prompt = tl.load(prompt_len_ptr + req_idx * prompt_len_stride, valid, 0)
        step = total - prompt
        valid = valid & (step >= 0) & (step < req_trace_len) & (step < trace_width)
        step = tl.where(valid, step, 0).to(OFFSET_DTYPE)
        token_id = tl.load(
            trace_token_ids_ptr
            + req_idx * trace_token_ids_stride
            + step * trace_token_ids_col_stride,
            valid,
            0,
        )
        tl.store(sampled_ptr + batch_idx * sampled_stride, token_id.to(tl.int64), valid)


def apply_trace_tokens(
    sampled: torch.Tensor,
    idx_mapping: torch.Tensor,
    trace_token_ids: torch.Tensor,
    trace_len: torch.Tensor,
    total_len: torch.Tensor,
    prompt_len: torch.Tensor,
) -> None:
    """Overwrite ``sampled`` in place with trace tokens for the current step."""
    logger.debug("GEMS_ASCEND APPLY_TRACE_TOKENS")
    num_reqs = idx_mapping.shape[0]
    num_states, trace_width = trace_token_ids.shape
    num_states = min(
        num_states, trace_len.shape[0], total_len.shape[0], prompt_len.shape[0]
    )
    if not (num_reqs and num_states and trace_width):
        return

    if num_reqs < 64:
        block = 1
    elif num_reqs < 2048:
        block = 8
    elif num_reqs < 3072:
        block = 64
    else:
        block = 128
    sampled_stride = sampled.stride(0)
    mapping_stride = idx_mapping.stride(0)
    trace_row_stride, trace_col_stride = trace_token_ids.stride()
    length_stride = trace_len.stride(0)
    total_stride = total_len.stride(0)
    prompt_stride = prompt_len.stride(0)

    # Include masked tail lanes in the offset bound and use int64 when a
    # possible element offset would overflow int32.
    num_programs = triton.cdiv(num_reqs, block)
    last_direct_lane = num_programs * block - 1
    last_state = num_states - 1
    max_offset = max(
        last_direct_lane * max(abs(sampled_stride), abs(mapping_stride)),
        last_state * max(abs(length_stride), abs(total_stride), abs(prompt_stride)),
        abs(last_state * trace_row_stride) + abs((trace_width - 1) * trace_col_stride),
    )
    offset_dtype = tl.int64 if max_offset >= 2**31 else tl.int32
    _trace_replay_ascend_kernel[(num_programs,)](
        sampled,
        sampled_stride,
        idx_mapping,
        mapping_stride,
        trace_token_ids,
        trace_row_stride,
        trace_col_stride,
        trace_len,
        length_stride,
        total_len,
        total_stride,
        prompt_len,
        prompt_stride,
        num_states,
        trace_width,
        num_reqs,
        block,
        offset_dtype,
    )

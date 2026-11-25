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

import torch
import triton
import triton.language as tl


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
):
    if BLOCK == 1:
        # Scalar pointers avoid the singleton tensor-pointer lowering path.
        batch_idx = tl.program_id(0)
        req_idx = tl.load(idx_mapping_ptr + batch_idx * idx_mapping_stride)
        if (req_idx >= 0) & (req_idx < num_states):
            req_trace_len = tl.load(trace_len_ptr + req_idx * trace_len_stride)
            total = tl.load(total_len_ptr + req_idx * total_len_stride)
            prompt = tl.load(prompt_len_ptr + req_idx * prompt_len_stride)
            step = total - prompt
            if (step >= 0) & (step < req_trace_len) & (step < trace_width):
                token_id = tl.load(
                    trace_token_ids_ptr
                    + req_idx * trace_token_ids_stride
                    + step * trace_token_ids_col_stride
                )
                tl.store(
                    sampled_ptr + batch_idx * sampled_stride, token_id.to(tl.int64)
                )
    else:
        batch_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = batch_idx < num_reqs
        req_idx = tl.load(
            idx_mapping_ptr + batch_idx * idx_mapping_stride, active, other=-1
        )
        valid = active & (req_idx >= 0) & (req_idx < num_states)

        # FlagTree may lower indirect masked loads to a gather followed by select.
        # Keep every gather address in bounds even when the lane is masked out.
        req_idx = tl.where(valid, req_idx, 0)
        req_trace_len = tl.load(trace_len_ptr + req_idx * trace_len_stride, valid, 0)
        total = tl.load(total_len_ptr + req_idx * total_len_stride, valid, 0)
        prompt = tl.load(prompt_len_ptr + req_idx * prompt_len_stride, valid, 0)
        step = total - prompt
        valid = valid & (step >= 0) & (step < req_trace_len) & (step < trace_width)
        step = tl.where(valid, step, 0)
        token_id = tl.load(
            trace_token_ids_ptr
            + req_idx * trace_token_ids_stride
            + step * trace_token_ids_col_stride,
            valid,
            0,
        )
        tl.store(sampled_ptr + batch_idx * sampled_stride, token_id.to(tl.int64), valid)


def apply_trace_tokens_ascend(
    sampled: torch.Tensor,
    idx_mapping: torch.Tensor,
    trace_token_ids: torch.Tensor,
    trace_len: torch.Tensor,
    total_len: torch.Tensor,
    prompt_len: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    num_states, trace_width = trace_token_ids.shape
    num_states = min(
        num_states, trace_len.shape[0], total_len.shape[0], prompt_len.shape[0]
    )
    if not (num_reqs and num_states and trace_width):
        return

    # Small batches benefit from scalar programs. Larger batches amortize
    # gather overhead while keeping enough programs to use the vector cores.
    if num_reqs < 32:
        block = 1
    elif num_reqs < 64:
        block = 4
    elif num_reqs < 1024:
        block = 8
    elif num_reqs < 3072:
        block = 64
    else:
        # Wider tiles reduce the number of programs for large batches.
        block = 128
    _trace_replay_ascend_kernel[(triton.cdiv(num_reqs, block),)](
        sampled,
        sampled.stride(0),
        idx_mapping,
        idx_mapping.stride(0),
        trace_token_ids,
        trace_token_ids.stride(0),
        trace_token_ids.stride(1),
        trace_len,
        trace_len.stride(0),
        total_len,
        total_len.stride(0),
        prompt_len,
        prompt_len.stride(0),
        num_states,
        trace_width,
        num_reqs,
        block,
    )

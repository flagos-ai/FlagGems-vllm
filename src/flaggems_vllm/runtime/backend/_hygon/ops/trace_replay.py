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

"""Replay trace tokens on the Hygon backend."""

import torch
import triton
import triton.language as tl


# No autotune: replay mutates sampled in place, so keep a single launch per call.
@triton.jit
def _trace_replay_kernel(
    sampled_ptr,
    idx_mapping_ptr,
    trace_token_ids_ptr,
    trace_token_ids_stride,
    trace_len_ptr,
    total_len_ptr,
    prompt_len_ptr,
    num_reqs: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=active, other=-1)
    valid = active & (req_state_idx >= 0)
    safe_req_idx = tl.maximum(req_state_idx, 0)

    trace_len = tl.load(trace_len_ptr + safe_req_idx, mask=valid, other=0)
    total_len = tl.load(total_len_ptr + safe_req_idx, mask=valid, other=0)
    prompt_len = tl.load(prompt_len_ptr + safe_req_idx, mask=valid, other=0)
    step = total_len - prompt_len
    valid = valid & (trace_len > 0) & (step >= 0) & (step < trace_len)

    token_id = tl.load(
        trace_token_ids_ptr + safe_req_idx * trace_token_ids_stride + step,
        mask=valid,
        other=0,
    )
    tl.store(sampled_ptr + batch_idx, token_id.to(tl.int64), mask=valid)


def apply_trace_tokens(
    sampled: torch.Tensor,
    idx_mapping: torch.Tensor,
    trace_token_ids: torch.Tensor,
    trace_len: torch.Tensor,
    total_len: torch.Tensor,
    prompt_len: torch.Tensor,
) -> None:
    """Overwrite ``sampled`` in place with trace tokens for the current step."""
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    block = min(triton.next_power_of_2(num_reqs), 128)
    _trace_replay_kernel[(triton.cdiv(num_reqs, block),)](
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_token_ids.stride(0),
        trace_len,
        total_len,
        prompt_len,
        num_reqs,
        block,
    )

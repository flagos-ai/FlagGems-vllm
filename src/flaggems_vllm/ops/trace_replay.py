# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the current sampled token from a request's decode trace."""

import torch
import triton
import triton.language as tl


@triton.jit
def _trace_replay_kernel(
    sampled_ptr,  # [num_reqs], int64, mutated in place
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx
    trace_token_ids_ptr,  # [max_num_reqs, max_model_len], int32
    trace_token_ids_stride,
    trace_len_ptr,  # [max_num_reqs], int32
    total_len_ptr,  # [max_num_reqs], int32
    prompt_len_ptr,  # [max_num_reqs], int32
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return

    trace_len = tl.load(trace_len_ptr + req_state_idx)
    if trace_len <= 0:
        return

    # Subtract prompt tokens to find the current decode step.
    step = tl.load(total_len_ptr + req_state_idx) - tl.load(
        prompt_len_ptr + req_state_idx
    )
    if step < 0 or step >= trace_len:
        return

    token_id = tl.load(
        trace_token_ids_ptr + req_state_idx * trace_token_ids_stride + step
    )
    tl.store(sampled_ptr + batch_idx, token_id.to(tl.int64))


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
    _trace_replay_kernel[(num_reqs,)](
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_token_ids.stride(0),
        trace_len,
        total_len,
        prompt_len,
    )

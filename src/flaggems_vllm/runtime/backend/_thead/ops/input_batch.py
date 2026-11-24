# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 128
_NUM_WARPS = 4


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, valid, other=0)
    seq_len = tl.load(seq_lens_ptr + batch_idx, valid, other=0)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, valid, other=0)
    num_sampled = tl.load(num_sampled_ptr + batch_idx, valid, other=0)
    logits_start = tl.load(cu_num_logits_ptr + batch_idx, valid, other=0)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1, valid, other=0)
    is_chunked_prefilling = seq_len < prefill_len
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    num_rejected = tl.where(
        is_chunked_prefilling, 0, logits_end - logits_start - num_sampled
    )
    tl.store(num_sampled_ptr + batch_idx, num_sampled, valid)
    tl.store(num_rejected_ptr + batch_idx, num_rejected, valid)


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_THEAD GET_NUM_SAMPLED_AND_REJECTED")
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    _get_num_sampled_and_rejected_kernel[(triton.cdiv(num_reqs, _BLOCK_SIZE),)](
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
        num_reqs,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )
    return num_sampled, num_rejected

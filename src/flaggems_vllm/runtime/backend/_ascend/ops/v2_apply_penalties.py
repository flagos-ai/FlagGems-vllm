# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM v0.32.0:
# https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu/sample/penalties.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend V2 penalties with a read-only source and an independent output view."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops._v2_sampling import (
    validate_logits,
    validate_vector,
)


@triton.jit
def _penalties_kernel(
    input_logits_ptr,
    input_logits_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        input_logits_ptr + token_idx * input_logits_stride + block, mask=mask
    )
    logits = logits.to(tl.float32)

    base_output_counts = tl.load(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )

    # Accumulate draft token counts from previous positions directly into
    # output_bin_counts (preserves its native tensor layout, avoiding an
    # expensive shared-memory layout conversion after the loop).
    pos = tl.load(expanded_local_pos_ptr + token_idx)
    start_idx = token_idx - pos
    output_bin_counts = base_output_counts
    for prev_pos in tl.range(pos):
        prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
        token_match = block == prev_token
        output_bin_counts = output_bin_counts + token_match.to(tl.int32)
    output_bin_mask = output_bin_counts > 0

    # Apply repetition penalties.
    if use_rep_penalty:
        packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed_mask = tl.load(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
            mask=packed_block < tl.cdiv(vocab_size, 32),
            other=0,
        )
        prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
        prompt_bin_mask = prompt_bin_mask.to(tl.int1)
        prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

        # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
        scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
        # If logits are positive, divide by penalty, otherwise multiply by penalty.
        logits *= tl.where(logits > 0, 1.0 / scale, scale)

    # Apply frequency penalties.
    logits -= freq_penalty * output_bin_counts
    # Apply presence penalties.
    logits -= pres_penalty * output_bin_mask
    # Store back to logits.
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def v2_apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """Apply vLLM V2 repetition, frequency and presence penalties in place.

    Logits are FP16, BF16, or FP32 with shape [tokens, vocabulary], contiguous
    columns and optional row padding. Request vectors contain FP32 penalties.
    Per-request prompt bits (32 tokens per word) and output counts are int32
    matrices with contiguous columns. Integer token metadata uses the same
    flattened request/position convention as vLLM's V2 batch. For draft rows,
    prior draft tokens contribute to output counts only for that row. Counts,
    masks and metadata are read-only. All operands are on the same Ascend
    device. Empty logits need no launch.
    """
    num_tokens, vocab_size = validate_logits(logits)
    if num_tokens == 0 or vocab_size == 0:
        return

    for name, vector in (
        ("expanded_idx_mapping", expanded_idx_mapping),
        ("expanded_local_pos", expanded_local_pos),
    ):
        validate_vector(
            vector, logits.device, (torch.int32, torch.int64), name, size=num_tokens
        )
    # vLLM may pass a padded token vector. Only the first num_tokens entries
    # participate in these logits rows.
    validate_vector(
        token_ids,
        logits.device,
        (torch.int32, torch.int64),
        "token_ids",
    )
    if token_ids.numel() < num_tokens:
        raise ValueError("token_ids must cover all logits rows")
    num_requests = repetition_penalty.numel()
    if num_requests == 0:
        raise ValueError("at least one request slot is required for nonempty logits")
    for name, vector in (
        ("repetition_penalty", repetition_penalty),
        ("frequency_penalty", frequency_penalty),
        ("presence_penalty", presence_penalty),
    ):
        validate_vector(
            vector, logits.device, (torch.float32,), name, size=num_requests
        )
    for name, matrix, width in (
        ("prompt_bin_mask", prompt_bin_mask, triton.cdiv(vocab_size, 32)),
        ("output_bin_counts", output_bin_counts, vocab_size),
    ):
        if matrix.device != logits.device or matrix.dtype != torch.int32:
            raise TypeError(f"{name} must contain int32 data on the logits device")
        if matrix.dim() != 2 or matrix.stride(1) != 1:
            raise ValueError(f"{name} must have contiguous columns")
        if matrix.size(0) < num_requests or matrix.size(1) < width:
            raise ValueError(f"{name} does not cover the request slots and vocabulary")

    with torch.npu.device(logits.device):
        source = logits.clone()
        # The upstream CUDA tile of 8192 exceeds the Ascend unified buffer.
        block_size = 1024
        num_blocks = triton.cdiv(vocab_size, block_size)
        _penalties_kernel[(num_tokens, num_blocks)](
            source,
            source.stride(0),
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            token_ids,
            expanded_local_pos,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            output_bin_counts,
            output_bin_counts.stride(0),
            vocab_size,
            BLOCK_SIZE=block_size,
        )

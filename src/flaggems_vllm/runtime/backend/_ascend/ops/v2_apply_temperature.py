# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM v0.32.0:
# https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu/sample/gumbel.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""V2 temperature scaling with an immutable source and original output view."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops._v2_sampling import (
    validate_logits,
    validate_vector,
)


@triton.jit
def _temperature_kernel(
    input_logits_ptr,
    input_logits_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(
        input_logits_ptr + token_idx * input_logits_stride + block, mask=mask
    )
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def v2_apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """Scale logits in place by the temperature of each row's request slot.

    Logits have shape [tokens, vocabulary] with contiguous columns and optional
    row padding. The integer mapping selects entries from a contiguous FP32
    temperature vector on the same Ascend device. Temperatures 0 and 1 are
    identity cases. Empty logits need no launch. Computation uses FP32 and
    writes back to the input dtype (FP16, BF16, or FP32).
    """
    num_tokens, vocab_size = validate_logits(logits)
    if num_tokens == 0 or vocab_size == 0:
        return
    validate_vector(
        expanded_idx_mapping,
        logits.device,
        (torch.int32, torch.int64),
        "expanded_idx_mapping",
        size=num_tokens,
    )
    validate_vector(
        temperature,
        logits.device,
        (torch.float32,),
        "temperature",
    )

    if temperature.numel() == 0:
        raise ValueError("at least one request slot is required for nonempty logits")

    # A separate source prevents duplicate tile execution by the Ascend
    # compiler from dividing the first output tile more than once.
    with torch.npu.device(logits.device):
        source = logits.clone()
        block_size = 1024
        num_blocks = triton.cdiv(vocab_size, block_size)
        _temperature_kernel[(num_tokens, num_blocks)](
            source,
            source.stride(0),
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            temperature,
            vocab_size,
            BLOCK_SIZE=block_size,
        )

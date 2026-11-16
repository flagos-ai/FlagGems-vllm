# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""MThreads context deduplication for speculative watermark sampling."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry


@triton.jit
def _scan_context_history(
    history_ptr,
    context_ptr,
    first,
    length,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)[None, :]
    repeated = tl.full((1,), False, tl.int1)
    for tile in range(first // BLOCK * BLOCK, length, BLOCK):
        positions = tile + offsets
        valid = (positions >= first) & (positions < length)
        # A complete context match must first match its last token.
        anchor = tl.load(context_ptr + CONTEXT_WIDTH - 1)
        last = tl.load(
            history_ptr + tl.maximum(positions - 1, 0),
            mask=valid & (positions >= 1),
            other=-1,
        )
        matches = valid & (last == anchor)
        if tl.sum(matches.to(tl.int32)) != 0:
            for column in tl.static_range(CONTEXT_WIDTH - 1):
                token = tl.load(context_ptr + column)
                position = positions + column - CONTEXT_WIDTH
                previous = tl.load(
                    history_ptr + tl.maximum(position, 0),
                    mask=matches & (position >= 0),
                    other=-1,
                )
                matches &= previous == token
        repeated |= tl.max(matches.to(tl.int32), 1) != 0
    return repeated


@libentry()
@triton.jit
def _repeated_context_mask_kernel(
    output_ptr,
    tokens_ptr,
    tokens_stride,
    req_ptr,
    prompt_ptr,
    total_ptr,
    prior_contexts_ptr,
    prior_row_stride,
    prior_step_stride,
    contexts_ptr,
    context_stride,
    steps_ptr,
    step_stride,
    enabled_ptr,
    CONTEXT_WIDTH: tl.constexpr,
    DRAFT_STEPS: tl.constexpr,
    MAX_HISTORY: tl.constexpr,
    INCLUDE_PROMPT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    request = tl.load(req_ptr + row).to(tl.int64)
    if request < 0:
        tl.store(output_ptr + row, False)
        return
    step = tl.load(steps_ptr + row * step_stride).to(tl.int64)
    start = 0 if INCLUDE_PROMPT else tl.load(prompt_ptr + request)
    length = tl.load(total_ptr + request) - start
    first = 0
    if MAX_HISTORY > 0:
        first = tl.maximum(length - tl.maximum(MAX_HISTORY - step, 0), 0)
    history_ptr = tokens_ptr + request * tokens_stride + start
    context_ptr = contexts_ptr + row * context_stride
    repeated = _scan_context_history(
        history_ptr, context_ptr, first, length, CONTEXT_WIDTH, BLOCK
    )
    previous_steps = tl.arange(0, triton.next_power_of_2(DRAFT_STEPS))[None, :]
    valid_steps = (previous_steps < step) & (previous_steps < DRAFT_STEPS)
    if MAX_HISTORY > 0:
        valid_steps &= previous_steps >= step - MAX_HISTORY
    matches = valid_steps
    partial = tl.full((), False, tl.int1)
    for column in tl.static_range(CONTEXT_WIDTH):
        token = tl.load(context_ptr + column)
        previous = tl.load(
            prior_contexts_ptr
            + row * prior_row_stride
            + previous_steps * prior_step_stride
            + column,
            mask=valid_steps,
            other=-2,
        )
        matches &= previous == token
        partial |= token < 0
        tl.store(
            prior_contexts_ptr
            + row * prior_row_stride
            + step * prior_step_stride
            + column,
            token,
        )
    repeated |= tl.max(matches.to(tl.int32), 1) != 0
    if INCLUDE_PROMPT:
        repeated |= partial
    enabled = tl.load(enabled_ptr + row)
    output_offsets = tl.arange(0, 1)
    tl.store(output_ptr + row + output_offsets, enabled & ~repeated)


def draft_watermarking_mask(
    all_token_ids: torch.Tensor,
    req_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    prior_contexts: torch.Tensor,
    contexts: torch.Tensor,
    steps: torch.Tensor,
    enabled: torch.Tensor,
    max_history: int | None,
    include_prompt: bool,
) -> torch.Tensor:
    """Return active watermark rows and record each valid row's current context.

    History tokens and request lengths are contiguous within rows. Context
    buffers have unit column stride. Steps may be a stride-zero broadcast.
    Each step is in ``[0, prior_contexts.shape[1])``.
    """
    if contexts.device.type == "cpu":
        raise NotImplementedError("draft_watermarking_mask requires an accelerator")
    if max_history is not None and max_history < 1:
        raise ValueError("max_history must be positive or None")
    if prior_contexts.stride(-1) != 1:
        raise ValueError("prior_contexts must have unit column stride")
    contexts = contexts.contiguous() if contexts.stride(-1) != 1 else contexts
    req_indices = req_indices.contiguous()
    enabled = enabled.contiguous()
    output = torch.empty(contexts.shape[0], device=contexts.device, dtype=torch.bool)
    if contexts.shape[0] == 0:
        return output
    _repeated_context_mask_kernel[(contexts.shape[0],)](
        output,
        all_token_ids,
        all_token_ids.stride(0),
        req_indices,
        prompt_lens,
        total_lens,
        prior_contexts,
        prior_contexts.stride(0),
        prior_contexts.stride(1),
        contexts,
        contexts.stride(0),
        steps,
        steps.stride(0),
        enabled,
        CONTEXT_WIDTH=contexts.shape[1],
        DRAFT_STEPS=prior_contexts.shape[1],
        MAX_HISTORY=0 if max_history is None else max_history,
        INCLUDE_PROMPT=include_prompt,
        BLOCK=512,
        num_warps=4,
        num_stages=1,
    )
    return output

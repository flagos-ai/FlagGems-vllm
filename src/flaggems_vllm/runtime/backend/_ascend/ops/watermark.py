# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Ascend context deduplication for speculative watermark sampling."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _load_history_column_ascend(
    history_ptr, start, length, offsets, BLOCK: tl.constexpr
):
    block = tl.make_block_ptr(
        history_ptr, (length,), (1,), (start.to(tl.int32),), (BLOCK,), (0,)
    )
    tokens = tl.load(block, boundary_check=(0,), padding_option="zero")
    return tl.where(start + offsets >= 0, tokens, -1)


@triton.jit
def _scan_context_history_ascend(
    history_ptr,
    context_ptr,
    first,
    length,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    anchor = tl.load(context_ptr + CONTEXT_WIDTH - 1)
    repeated = tl.full((), False, tl.int1)
    tile = first
    while (tile < length) & ~repeated:
        last = _load_history_column_ascend(
            history_ptr, tile - 1, length, offsets, BLOCK
        )
        matches = (tile + offsets < length) & (last == anchor)
        if tl.sum(matches.to(tl.int32), 0) != 0:
            for column in tl.static_range(CONTEXT_WIDTH - 1):
                token = tl.load(context_ptr + column)
                previous = _load_history_column_ascend(
                    history_ptr, tile + column - CONTEXT_WIDTH, length, offsets, BLOCK
                )
                matches &= previous == token
        repeated = tl.sum(matches.to(tl.int32), 0) != 0
        tile += BLOCK
    return repeated


@triton.jit
def _update_prior_contexts_ascend(
    prior_row,
    prior_step_stride,
    context_ptr,
    step,
    CONTEXT_WIDTH: tl.constexpr,
    MAX_HISTORY: tl.constexpr,
    INCLUDE_PROMPT: tl.constexpr,
):
    first_step = 0 if MAX_HISTORY == 0 else tl.maximum(0, step - MAX_HISTORY)
    repeated = tl.full((), False, tl.int1)
    # Masked strided loads across draft slots can trigger CANN 507035.
    for previous_step in range(first_step, step):
        previous_ptr = prior_row + previous_step * prior_step_stride
        matches = tl.full((), True, tl.int1)
        for column in tl.static_range(CONTEXT_WIDTH):
            token = tl.load(context_ptr + column)
            previous = tl.load(previous_ptr + column)
            matches &= previous == token
        repeated |= matches
    current_ptr = prior_row + step * prior_step_stride
    for column in tl.static_range(CONTEXT_WIDTH):
        token = tl.load(context_ptr + column)
        if INCLUDE_PROMPT:
            repeated |= token < 0
        tl.store(current_ptr + column, token)
    return repeated


@libentry()
@triton.jit
def _repeated_context_mask_ascend_kernel(
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
    if length - first <= 128:
        repeated = _scan_context_history_ascend(
            history_ptr, context_ptr, first, length, CONTEXT_WIDTH, 128
        )
    else:
        repeated = _scan_context_history_ascend(
            history_ptr, context_ptr, first, length, CONTEXT_WIDTH, BLOCK
        )
    repeated |= _update_prior_contexts_ascend(
        prior_contexts_ptr + row * prior_row_stride,
        prior_step_stride,
        context_ptr,
        step,
        CONTEXT_WIDTH,
        MAX_HISTORY,
        INCLUDE_PROMPT,
    )
    enabled = tl.load(enabled_ptr + row)
    tl.store(output_ptr + row, enabled & ~repeated)


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
    logger.debug("GEMS_ASCEND DRAFT_WATERMARKING_MASK")
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
    _repeated_context_mask_ascend_kernel[(contexts.shape[0],)](
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
        MAX_HISTORY=0 if max_history is None else max_history,
        INCLUDE_PROMPT=include_prompt,
        BLOCK=512,
        num_warps=4,
        num_stages=1,
    )
    return output

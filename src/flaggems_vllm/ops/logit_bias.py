# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl

LOGITS_BLOCK_SIZE = 8192


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    restore_when_all_masked_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
    CHECK_ALL_MASKED_ROWS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    request = tl.load(expanded_idx_mapping_ptr + row)
    row_ptr = logits_ptr + row * logits_stride
    offsets = tl.arange(0, BLOCK_SIZE)

    allowed_count = tl.load(num_allowed_token_ids_ptr + request)
    if allowed_count > 0:
        mask = offsets < allowed_count
        allowed_ids = tl.load(
            allowed_token_ids_ptr + request * allowed_token_ids_stride + offsets,
            mask=mask,
            other=0,
        )
        saved = tl.load(row_ptr + allowed_ids, mask=mask, other=0)
        tl.debug_barrier()
        for start in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            columns = start + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(row_ptr + columns, -float("inf"), columns < vocab_size)
        tl.debug_barrier()
        tl.store(row_ptr + allowed_ids, saved, mask)

    bias_count = tl.load(num_logit_bias_ptr + request)
    if bias_count > 0:
        mask = offsets < bias_count
        ids = tl.load(
            bias_token_ids_ptr + request * bias_token_ids_stride + offsets,
            mask=mask,
            other=0,
        )
        bias = tl.load(
            bias_ptr + request * bias_stride + offsets,
            mask=mask,
            other=0,
        )
        values = tl.load(row_ptr + ids, mask=mask, other=0)
        tl.store(row_ptr + ids, values + bias, mask)

    stop_count = tl.load(num_stop_token_ids_ptr + request)
    position = tl.load(pos_ptr + row)
    min_len = tl.load(min_lens_ptr + request)
    if stop_count > 0 and position + 1 < min_len:
        mask = offsets < stop_count
        stop_ids = tl.load(
            stop_token_ids_ptr + request * stop_token_ids_stride + offsets,
            mask=mask,
            other=0,
        )
        if CHECK_ALL_MASKED_ROWS:
            should_restore = tl.load(restore_when_all_masked_ptr + request)
            if should_restore:
                stop_values = tl.load(
                    row_ptr + stop_ids, mask=mask, other=-float("inf")
                )
                tl.debug_barrier()
                tl.store(row_ptr + stop_ids, -float("inf"), mask)
                tl.debug_barrier()
                row_max = tl.full((), -float("inf"), tl.float32)
                for start in range(0, vocab_size, LOGITS_BLOCK_SIZE):
                    columns = start + tl.arange(0, LOGITS_BLOCK_SIZE)
                    values = tl.load(
                        row_ptr + columns,
                        columns < vocab_size,
                        other=-float("inf"),
                    )
                    row_max = tl.maximum(row_max, tl.max(values, 0))
                if row_max == -float("inf"):
                    tl.store(
                        row_ptr + stop_ids,
                        stop_values,
                        mask
                        & (stop_values > -float("inf"))
                        & (stop_values < float("inf")),
                    )
            else:
                tl.store(row_ptr + stop_ids, -float("inf"), mask)
        else:
            tl.store(row_ptr + stop_ids, -float("inf"), mask)


def apply_logit_bias(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    restore_when_all_masked: torch.Tensor,
    stop_token_ids: torch.Tensor,
    check_all_masked_rows: bool = False,
) -> None:
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0 or vocab_size == 0:
        return
    max_width = max(
        allowed_token_ids.shape[-1],
        logit_bias_token_ids.shape[-1],
        stop_token_ids.shape[-1],
    )
    block_size = max(1, triton.next_power_of_2(max_width))
    _bias_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        allowed_token_ids.stride(0),
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias_token_ids.stride(0),
        logit_bias,
        logit_bias.stride(0),
        pos,
        min_lens,
        num_stop_token_ids,
        restore_when_all_masked,
        stop_token_ids,
        stop_token_ids.stride(0),
        BLOCK_SIZE=block_size,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
        CHECK_ALL_MASKED_ROWS=check_all_masked_rows,
    )

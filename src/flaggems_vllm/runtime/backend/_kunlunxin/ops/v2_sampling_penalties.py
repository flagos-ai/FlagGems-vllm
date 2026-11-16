# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def _clear_rows(out, mapping, width, stride, BLOCK: tl.constexpr):
    req = tl.load(mapping + tl.program_id(0)).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + req * stride + col, 0, mask=col < width)


@triton.jit
def _count_tokens(
    token_ids,
    mapping,
    prompt_len,
    prefill_len,
    prompt_count,
    output_count,
    prompt_stride,
    output_stride,
    token_limit,
    TOKEN_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_row = tl.program_id(0)
    req = tl.load(mapping + batch_row).to(tl.int64)
    # P800 atomics may drop writes between programs, so a request slot is
    # owned by one program even when the sequence spans several tiles.
    lane = tl.arange(0, BLOCK)
    for start in range(0, token_limit, BLOCK):
        pos = start + lane
        tok = tl.load(token_ids + req * TOKEN_STRIDE + pos, pos < token_limit, 0).to(
            tl.int64
        )
        n_prompt = tl.load(prompt_len + req).to(tl.int64)
        n_prefill = tl.load(prefill_len + req).to(tl.int64)
        target = tl.where(
            pos < n_prompt,
            prompt_count + batch_row.to(tl.int64) * prompt_stride + tok,
            output_count + req * output_stride + tok,
        )
        tl.atomic_add(target, 1, sem="relaxed", mask=pos < n_prefill)


@triton.jit
def _pack_prompt(
    packed,
    dense,
    mapping,
    words,
    packed_stride,
    dense_stride,
    BLOCK: tl.constexpr,
):
    batch_row = tl.program_id(0)
    req = tl.load(mapping + batch_row).to(tl.int64)
    group = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    bit = tl.arange(0, 32)
    token = group[:, None] * 32 + bit[None, :]
    count = tl.load(dense + batch_row * dense_stride + token, group[:, None] < words, 0)
    code = tl.sum(tl.where(count != 0, 1 << bit[None, :], 0), axis=1)
    tl.store(packed + req * packed_stride + group, code, mask=group < words)


def v2_penalty_bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None:
    """Rebuild packed prompt bits and output counts for updated request slots."""
    num_reqs = expanded_idx_mapping.numel()
    vocab = output_bin_counts.shape[-1]
    if not num_reqs or not vocab:
        return
    words = triton.cdiv(vocab, 32)
    if prompt_bin_mask.shape[-1] != words:
        raise ValueError("prompt_bin_mask width must cover one bit per token")
    if (
        not expanded_idx_mapping.is_contiguous()
        or not prompt_len.is_contiguous()
        or not prefill_len.is_contiguous()
        or all_token_ids.stride(-1) != 1
        or prompt_bin_mask.stride(-1) != 1
        or output_bin_counts.stride(-1) != 1
    ):
        raise NotImplementedError(
            "V2 penalty counting requires contiguous inner dimensions"
        )
    if not 0 <= max_prefill_len <= all_token_ids.shape[-1]:
        raise ValueError("max_prefill_len is outside the token buffer")

    clear_block = 4096
    block = 256
    with torch_device_fn.device(output_bin_counts.device):
        _clear_rows[(num_reqs, triton.cdiv(vocab, clear_block))](
            output_bin_counts,
            expanded_idx_mapping,
            vocab,
            output_bin_counts.stride(0),
            clear_block,
        )
        if max_prefill_len == 0:
            _clear_rows[(num_reqs, triton.cdiv(words, block))](
                prompt_bin_mask,
                expanded_idx_mapping,
                words,
                prompt_bin_mask.stride(0),
                block,
            )
            return

        dense_stride = words * 32
        prompt_count = output_bin_counts.new_zeros((num_reqs, dense_stride))
        _count_tokens[(num_reqs,)](
            all_token_ids,
            expanded_idx_mapping,
            prompt_len,
            prefill_len,
            prompt_count,
            output_bin_counts,
            dense_stride,
            output_bin_counts.stride(0),
            max_prefill_len,
            TOKEN_STRIDE=all_token_ids.stride(0),
            BLOCK=block,
        )
        pack_block = 64
        _pack_prompt[(num_reqs, triton.cdiv(words, pack_block))](
            prompt_bin_mask,
            prompt_count,
            expanded_idx_mapping,
            words,
            prompt_bin_mask.stride(0),
            dense_stride,
            pack_block,
        )


@triton.jit
def _add_draft_counts(
    output,
    mapping,
    token_ids,
    positions,
    counts,
    vocab,
    counts_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    position = tl.load(positions + row).to(tl.int64)
    if position == 0:
        return
    req = tl.load(mapping + row).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    for start in range(0, vocab, BLOCK):
        offset = start + lane
        valid = offset < vocab
        output_count = tl.load(counts + req * counts_stride + offset, valid, 0)
        for prev in range(position):
            tok = tl.load(token_ids + row - position + prev + 1)
            output_count += tl.where(tok == offset, 1, 0)
        tl.store(output + row * vocab + offset, output_count, valid)


@triton.jit
def _apply_penalties(
    logits,
    mapping,
    positions,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    prompt,
    counts,
    draft_counts,
    logits_stride,
    prompt_stride,
    counts_stride,
    vocab,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(mapping + row).to(tl.int64)
    rep = tl.load(repetition_penalty + req)
    freq = tl.load(frequency_penalty + req)
    pres = tl.load(presence_penalty + req)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < vocab
    dst = logits + row * logits_stride + offset
    original = tl.load(dst, valid, 0).to(tl.float32)
    dst_counts = tl.where(
        tl.load(positions + row) > 0,
        draft_counts + row * vocab + offset,
        counts + req * counts_stride + offset,
    )
    output_count = tl.load(dst_counts, valid, 0)
    packed = tl.load(prompt + req * prompt_stride + (offset >> 5), valid, 0)
    repeated = ((packed >> (offset & 31)) & 1) != 0
    appeared = output_count > 0
    scale = tl.where(repeated | appeared, rep, 1.0)
    value = original * tl.where(original > 0.0, 1.0 / scale, scale)
    value -= freq * output_count
    value -= pres * appeared
    active = (rep != 1.0) | (freq != 0.0) | (pres != 0.0)
    # Selecting the original value retains a signed zero on inactive rows.
    tl.store(dst, tl.where(active, value, original), valid)


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
    """Apply V2 repetition, frequency and presence penalties in place."""
    rows, vocab = logits.shape
    if not rows or not vocab:
        return
    words = triton.cdiv(vocab, 32)
    if prompt_bin_mask.shape[-1] != words or output_bin_counts.shape[-1] != vocab:
        raise ValueError("V2 penalty statistics must match the vocabulary size")
    if (
        logits.stride(-1) != 1
        or prompt_bin_mask.stride(-1) != 1
        or output_bin_counts.stride(-1) != 1
        or not expanded_idx_mapping.is_contiguous()
        or not token_ids.is_contiguous()
        or not expanded_local_pos.is_contiguous()
        or not repetition_penalty.is_contiguous()
        or not frequency_penalty.is_contiguous()
        or not presence_penalty.is_contiguous()
    ):
        raise NotImplementedError("V2 penalties require contiguous inner dimensions")

    draft_counts = output_bin_counts.new_empty((rows, vocab))
    draft_block = 64
    block = 1024
    with torch_device_fn.device(logits.device):
        _add_draft_counts[(rows,)](
            draft_counts,
            expanded_idx_mapping,
            token_ids,
            expanded_local_pos,
            output_bin_counts,
            vocab,
            output_bin_counts.stride(0),
            draft_block,
        )
        _apply_penalties[(rows, triton.cdiv(vocab, block))](
            logits,
            expanded_idx_mapping,
            expanded_local_pos,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            prompt_bin_mask,
            output_bin_counts,
            draft_counts,
            logits.stride(0),
            prompt_bin_mask.stride(0),
            output_bin_counts.stride(0),
            vocab,
            block,
        )

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def _apply_grammar_bitmask(
    logits,
    bitmask,
    mapping,
    offsets,
    scratch,
    vocab,
    logits_stride,
    bitmask_stride,
    MASK_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    mask_row = tl.program_id(0)
    map_index = tl.load(mapping + mask_row).to(tl.int64)
    req_index = map_index // MASK_STRIDE
    position = map_index % MASK_STRIDE
    start = tl.load(offsets + req_index).to(tl.int64)
    end = tl.load(offsets + req_index + 1).to(tl.int64)
    group = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    word = tl.load(
        bitmask + mask_row * bitmask_stride + group,
        group < tl.cdiv(vocab, 32),
        other=0,
    )
    for bit in tl.static_range(32):
        token = group * 32 + bit
        blocked = (word & (-2147483648 if bit == 31 else 1 << bit)) == 0
        # FlagTree XPU drops the row predicate in the packed vector store.
        target = tl.where(
            tl.where(position < end - start, blocked, False),
            logits + (start + position) * logits_stride + token,
            scratch + mask_row * vocab + token,
        )
        tl.store(target, -float("inf"), token < vocab)


def v2_apply_grammar_bitmask(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    logits_indices: torch.Tensor,
    cu_num_logits: torch.Tensor,
    mask_stride: int,
) -> None:
    """Apply packed grammar masks for the logit rows selected by V2."""
    vocab = logits.shape[-1]
    rows, words = bitmask.shape
    if not rows or not vocab:
        return
    if mask_stride < 1 or words != triton.cdiv(vocab, 32):
        raise ValueError("invalid V2 grammar mask stride or width")
    if (
        logits.stride(-1) != 1
        or bitmask.stride(-1) != 1
        or not logits_indices.is_contiguous()
        or not cu_num_logits.is_contiguous()
    ):
        raise NotImplementedError(
            "V2 grammar mask requires contiguous inner dimensions"
        )

    # Each mask has distinct scratch so unused writes cannot race each other.
    scratch = logits.new_empty((rows, vocab))
    block = 64
    with torch_device_fn.device(logits.device):
        _apply_grammar_bitmask[(rows, triton.cdiv(words, block))](
            logits,
            bitmask,
            logits_indices,
            cu_num_logits,
            scratch,
            vocab,
            logits.stride(0),
            bitmask.stride(0),
            MASK_STRIDE=mask_stride,
            BLOCK=block,
        )

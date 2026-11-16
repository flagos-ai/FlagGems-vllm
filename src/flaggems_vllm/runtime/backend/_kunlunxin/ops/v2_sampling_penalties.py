# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

# Bound temporary vocabulary-sized matrices to this many request rows.
_ROW_CHUNK = 16


def v2_penalty_bincount(
    expanded_idx_mapping,
    all_token_ids,
    prompt_len,
    prefill_len,
    prompt_bin_mask,
    output_bin_counts,
    max_prefill_len,
) -> None:
    """Rebuild packed prompt bits and output counts for selected V2 request slots."""

    idx = expanded_idx_mapping.to(torch.int64)
    # Use index_fill_ instead of `tensor[idx] = 0` to avoid a sync, as upstream
    # does.
    prompt_bin_mask.index_fill_(0, idx, 0)
    output_bin_counts.index_fill_(0, idx, 0)

    num_reqs = idx.numel()
    if num_reqs == 0 or max_prefill_len <= 0:
        return

    device = prompt_bin_mask.device
    num_words = prompt_bin_mask.shape[1]
    vocab_size = output_bin_counts.shape[1]

    tokens = all_token_ids.index_select(0, idx)[:, :max_prefill_len].to(torch.int64)
    p_len = prompt_len.index_select(0, idx).to(torch.int64).unsqueeze(1)
    pf_len = prefill_len.index_select(0, idx).to(torch.int64).unsqueeze(1)
    col = torch.arange(tokens.shape[1], device=device).unsqueeze(0)

    prompt_valid = col < p_len
    output_valid = (col >= p_len) & (col < pf_len)

    shifts = torch.arange(32, device=device, dtype=torch.int64)
    weights = torch.ones(32, device=device, dtype=torch.int64) << shifts

    for lo in range(0, num_reqs, _ROW_CHUNK):
        hi = min(lo + _ROW_CHUNK, num_reqs)
        rows = idx[lo:hi]
        tok = tokens[lo:hi]
        p_ok = prompt_valid[lo:hi]
        o_ok = output_valid[lo:hi]
        n = hi - lo

        # Prompt bitmask.  Invalid slots are routed to a scratch column so they
        # cannot clobber a real token's bit.
        dense = torch.zeros(n, num_words * 32 + 1, dtype=torch.bool, device=device)
        dense.scatter_(1, torch.where(p_ok, tok, dense.shape[1] - 1), True)
        bits = dense[:, : num_words * 32].view(n, num_words, 32).to(torch.int64)
        words = (bits * weights).sum(-1)
        prompt_bin_mask.index_copy_(0, rows, words.to(prompt_bin_mask.dtype))

        # Output token counts, same scratch-column trick.
        counts = torch.zeros(n, vocab_size + 1, dtype=torch.int32, device=device)
        counts.scatter_add_(
            1,
            torch.where(o_ok, tok, vocab_size),
            torch.ones(n, tok.shape[1], dtype=torch.int32, device=device),
        )
        output_bin_counts.index_copy_(
            0, rows, counts[:, :vocab_size].to(output_bin_counts.dtype)
        )


def v2_apply_penalties(
    logits,
    expanded_idx_mapping,
    token_ids,
    expanded_local_pos,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    prompt_bin_mask,
    output_bin_counts,
) -> None:
    """Apply V2 repetition, frequency and presence penalties to logits in place."""

    num_tokens, vocab_size = logits.shape
    if num_tokens == 0:
        return

    device = logits.device
    req = expanded_idx_mapping.to(torch.int64)
    rep = repetition_penalty.index_select(0, req).to(torch.float32)
    freq = frequency_penalty.index_select(0, req).to(torch.float32)
    pres = presence_penalty.index_select(0, req).to(torch.float32)

    # Same early-out as the kernel: rows with no active penalty are untouched.
    active = (rep != 1.0) | (freq != 0.0) | (pres != 0.0)
    sel = active.nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        return

    local_pos = expanded_local_pos.index_select(0, sel).to(torch.int64)
    max_pos = int(local_pos.max().item())
    num_words = prompt_bin_mask.shape[1]
    shifts = torch.arange(32, device=device, dtype=torch.int32)

    for lo in range(0, sel.numel(), _ROW_CHUNK):
        hi = min(lo + _ROW_CHUNK, sel.numel())
        rows = sel[lo:hi]
        req_rows = req.index_select(0, rows)
        n = hi - lo

        lg = logits.index_select(0, rows).to(torch.float32)

        counts = output_bin_counts.index_select(0, req_rows).to(torch.int32)
        if max_pos > 0:
            # Fold in the draft tokens at earlier positions of the same request,
            # which the kernel accumulates into output_bin_counts on the fly.
            pos = local_pos[lo:hi]
            start = rows - pos
            padded = torch.zeros(n, vocab_size + 1, dtype=torch.int32, device=device)
            for step in range(max_pos):
                taken = pos > step
                src = (start + step + 1).clamp_(0, token_ids.shape[0] - 1)
                tok = token_ids.index_select(0, src).to(torch.int64)
                padded.scatter_add_(
                    1,
                    torch.where(taken, tok, vocab_size).unsqueeze(1),
                    torch.ones(n, 1, dtype=torch.int32, device=device),
                )
            counts = counts + padded[:, :vocab_size]
        output_bin_mask = counts > 0

        # Repetition penalty.  rep == 1.0 makes the scale a no-op, which is why
        # the kernel's per-row `use_rep_penalty` branch needs no counterpart.
        words = prompt_bin_mask.index_select(0, req_rows).to(torch.int32)
        prompt_mask = (
            ((words.unsqueeze(-1) >> shifts) & 1)
            .to(torch.bool)
            .view(n, num_words * 32)[:, :vocab_size]
        )
        scale = torch.where(
            prompt_mask | output_bin_mask, rep.index_select(0, rows).unsqueeze(1), 1.0
        )
        # Positive logits are divided by the penalty, negative ones multiplied.
        lg *= torch.where(lg > 0, 1.0 / scale, scale)

        lg -= freq.index_select(0, rows).unsqueeze(1) * counts
        lg -= pres.index_select(0, rows).unsqueeze(1) * output_bin_mask

        logits.index_copy_(0, rows, lg.to(logits.dtype))

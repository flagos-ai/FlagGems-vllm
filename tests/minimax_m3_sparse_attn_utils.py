# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Test/benchmark-only fixtures; production code never imports this module."""

import itertools

import torch


def make_inputs(
    qlens=(17, 13),
    d=128,
    kv_heads=1,
    dtype=torch.bfloat16,
    mode="compact",
    topk=16,
    strided=False,
    context=4096,
):
    torch.manual_seed(20261022)
    device = "cuda"
    total = sum(qlens)
    heads = kv_heads * 8
    pages_per_request = (context + 127) // 128
    pages = pages_per_request * len(qlens)
    q = torch.randn((total, heads, d), device=device, dtype=torch.bfloat16)
    kv = (torch.randn((pages, kv_heads, 128, 2 * d), device=device) * 0.3).to(dtype)
    bt = torch.randperm(pages, device=device).to(torch.int32).view(len(qlens), -1)
    cu = torch.tensor(
        [0] + list(itertools.accumulate(qlens)), device=device, dtype=torch.int32
    )
    seq = torch.full((len(qlens),), context, device=device, dtype=torch.int32)
    prefix = seq - torch.tensor(qlens, device=device, dtype=torch.int32)
    ids = []
    for _ in range(total * kv_heads):
        if mode == "compact":
            row = torch.arange(topk, device=device) + max(0, pages_per_request - topk)
        else:
            row = torch.randperm(pages_per_request, device=device)[:topk]
        ids.append(row)
    top = torch.stack(ids).view(kv_heads, total, topk).to(torch.int32)
    if mode == "empty":
        top.fill_(-1)
    elif mode == "holes":
        top[:, :, 1::3] = -1
    elif mode == "duplicates":
        top[:, :, 1] = top[:, :, 0]
    if strided:
        base = torch.empty((pages, 128, kv_heads, 4 * d), device=device, dtype=dtype)
        view = base[:, :, :, ::2].permute(0, 2, 1, 3)
        view.copy_(kv)
        kv = view
        base = torch.empty(
            (total, kv_heads, 2 * topk), device=device, dtype=torch.int32
        )
        view = base[:, :, ::2].transpose(0, 1)
        view.copy_(top)
        top = view
        base = torch.empty(
            (len(qlens), 2 * pages_per_request), device=device, dtype=torch.int32
        )
        view = base[:, ::2]
        view.copy_(bt)
        bt = view
    return (q, kv, top, bt, cu, seq, prefix, max(qlens), kv_heads, d**-0.5)


def dense_reference(args, token_ids=None, k_scale=None, v_scale=None):
    """Independent FP32 selected-token oracle; duplicates retain multiplicity."""
    q, kv, top, bt, cu, seq, prefix, _, kv_heads, scale = args
    starts = cu.cpu().tolist()
    indices = list(range(q.shape[0])) if token_ids is None else list(token_ids)
    result = torch.empty(
        (len(indices),) + q.shape[1:], device=q.device, dtype=torch.float32
    )
    pos_offsets = torch.arange(128, device=q.device)
    for out_i, token in enumerate(indices):
        request = next(
            i for i in range(len(starts) - 1) if starts[i] <= token < starts[i + 1]
        )
        q_pos = prefix[request] + token - starts[request]
        for kh in range(kv_heads):
            blocks = top[kh, token].long()
            pages = bt[request, blocks.clamp_min(0)].long()
            keys = kv[pages, kh, :, : q.shape[-1]].float().flatten(0, 1)
            values = kv[pages, kh, :, q.shape[-1] :].float().flatten(0, 1)
            if k_scale is not None:
                keys = keys * k_scale
                values = values * v_scale
            positions = (blocks[:, None] * 128 + pos_offsets).flatten()
            valid = (blocks >= 0)[:, None].expand(-1, 128).flatten()
            valid &= (positions <= q_pos) & (positions < seq[request])
            logits = q[token, kh * 8 : (kh + 1) * 8].float() @ keys.T * scale
            weights = (
                logits.masked_fill(~valid[None, :], float("-inf"))
                .softmax(-1)
                .nan_to_num()
            )
            result[out_i, kh * 8 : (kh + 1) * 8] = weights @ values
    return result

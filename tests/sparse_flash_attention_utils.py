# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared inputs and reference calls; never imported by production code."""

from functools import lru_cache

import torch

CASES = [
    (1, 2560, 2560),
    (4, 6400, 2560),
    (8, 48000, 2560),
    (16, 48000, 2560),
    (16, 48000, 48000),
]
MODES = ("contiguous", "random", "sorted_random")


def make_inputs(case, mode, device):
    load_vllm_sfa()
    b, capacity, valid = case
    torch.manual_seed(20260928 + b + valid)
    gen = torch.Generator().manual_seed(20260928 + b + valid)
    q = torch.randn((b, 128, 512), dtype=torch.float16, device=device) * 0.5
    qr = torch.randn((b, 128, 64), dtype=torch.float16, device=device) * 0.5
    k = (
        torch.randn(
            (b * capacity // 128, 128, 1, 512), dtype=torch.float16, device=device
        )
        * 0.5
    )
    kr = torch.randn((*k.shape[:3], 64), dtype=torch.float16, device=device) * 0.5
    pages = (
        torch.randperm(k.shape[0], generator=gen)
        if valid == 48000
        else torch.arange(k.shape[0])
    )
    bt = pages.reshape(b, -1).int().to(device)
    selected = torch.stack(
        [torch.randperm(valid, generator=gen)[:2048] for _ in range(b)]
    ).int()
    if mode == "contiguous":
        selected = torch.arange(2048, dtype=torch.int32).repeat(b, 1)
    elif mode == "sorted_random":
        selected = selected.sort(-1).values
    elif mode == "duplicates":
        selected = selected % 128
        selected[:, 0] = 0
        selected[:, -1] = valid - 1
    return dict(
        query=q,
        key=k,
        value=k,
        query_rope=qr,
        key_rope=kr,
        sparse_indices=selected[:, None, :].to(device),
        block_table=bt,
        scale_value=0.5,
        actual_seq_lengths_query=torch.arange(
            1, b + 1, dtype=torch.int32, device=device
        ),
        actual_seq_lengths_kv=torch.full((b,), valid, dtype=torch.int32, device=device),
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
    )


def vllm_inputs(inputs):
    """Return matched full-interface kwargs without adding baseline compute.

    The lengths are prepared outside performance timing. No sparse gather or
    other compute is added to the baseline; value aliases the MLA key cache.
    """
    return dict(inputs)


@lru_cache(maxsize=1)
def load_vllm_sfa():
    """Load the installed vLLM vendor ops without falling back to TorchNPU."""
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("The installed vLLM-Ascend custom ops could not be loaded")
    return torch.ops._C_ascend.npu_sparse_flash_attention.default


def vllm_reference(inputs):
    return load_vllm_sfa()(**vllm_inputs(inputs))[0]


def cpu_reference(inputs, heads=2):
    q = inputs["query"][:, :heads].float().cpu()
    qr = inputs["query_rope"][:, :heads].float().cpu()
    ix = inputs["sparse_indices"][:, 0].long()
    rows = inputs["block_table"].long().gather(1, ix // 128) * 128 + ix % 128
    gathered_kv = inputs["key"].view(-1, 512)[rows].float().cpu()
    gathered_key_rope = inputs["key_rope"].view(-1, 64)[rows].float().cpu()
    scores = (
        q @ gathered_kv.transpose(1, 2) + qr @ gathered_key_rope.transpose(1, 2)
    ) * inputs["scale_value"]
    return scores.softmax(-1) @ gathered_kv


def make_contract_inputs(
    device,
    *,
    dtype=torch.float16,
    heads=16,
    num_sparse_blocks=37,
    page_size=128,
    sparse_block=1,
    query_layout="TND",
    kv_layout="PA_BSND",
    qlens=(2, 3),
    klens=(211, 173),
    mode=3,
    return_softmax_stats=False,
    omitted_lengths=False,
    all_padding=False,
):
    """Small ragged/prefill cases, including shuffled physical pages and padding."""
    load_vllm_sfa()
    gen = torch.Generator().manual_seed(20261012)
    b, qs = len(qlens), max(qlens)
    capacity = max(klens)
    if kv_layout == "PA_BSND":
        capacity = ((capacity + page_size - 1) // page_size) * page_size
    q = torch.randn((b, qs, heads, 512), generator=gen).to(dtype) * 0.25
    qr = torch.randn((b, qs, heads, 64), generator=gen).to(dtype) * 0.25
    k = torch.randn((b, capacity, 1, 512), generator=gen).to(dtype) * 0.25
    kr = torch.randn((b, capacity, 1, 64), generator=gen).to(dtype) * 0.25
    selected = torch.full((b, qs, 1, num_sparse_blocks), -1, dtype=torch.int32)
    for batch, length in enumerate(klens):
        nblocks = (length + sparse_block - 1) // sparse_block
        for pos in range(qs):
            count = min(num_sparse_blocks, nblocks)
            if count:
                indices = torch.randperm(nblocks, generator=gen)[:count]
                # Ensure valid causal rows have at least one selected past token.
                zero = (indices == 0).nonzero()
                if zero.numel():
                    j = int(zero[0, 0])
                    indices[0], indices[j] = indices[j].clone(), indices[0].clone()
                else:
                    indices[0] = 0
                selected[batch, pos, 0, :count] = indices.int()
    if all_padding:
        selected.fill_(-1)
    if query_layout == "TND":
        q = torch.cat([q[j, :n] for j, n in enumerate(qlens)])
        qr = torch.cat([qr[j, :n] for j, n in enumerate(qlens)])
        selected = torch.cat([selected[j, :n] for j, n in enumerate(qlens)])
    table = None
    if kv_layout == "PA_BSND":
        k = k.reshape(-1, page_size, 1, 512)
        kr = kr.reshape(-1, page_size, 1, 64)
        permutation = torch.randperm(k.shape[0], generator=gen)
        table = permutation.reshape(b, -1).int()
        original_k, original_kr = k.clone(), kr.clone()
        k[permutation] = original_k
        kr[permutation] = original_kr
    elif kv_layout == "TND":
        k = torch.cat([k[j, :n] for j, n in enumerate(klens)])
        kr = torch.cat([kr[j, :n] for j, n in enumerate(klens)])
    query_lengths = (
        torch.tensor(qlens).cumsum(0) if query_layout == "TND" else torch.tensor(qlens)
    ).int()
    kv_lengths = (
        torch.tensor(klens).cumsum(0) if kv_layout == "TND" else torch.tensor(klens)
    ).int()
    q, qr, k, kr = (x.to(device) for x in (q, qr, k, kr))
    return dict(
        query=q,
        key=k,
        value=k,
        sparse_indices=selected.to(device),
        query_rope=qr,
        key_rope=kr,
        scale_value=0.5,
        block_table=table.to(device) if table is not None else None,
        actual_seq_lengths_query=None if omitted_lengths else query_lengths.to(device),
        actual_seq_lengths_kv=None if omitted_lengths else kv_lengths.to(device),
        layout_query=query_layout,
        layout_kv=kv_layout,
        sparse_block_size=sparse_block,
        sparse_mode=mode,
        return_softmax_lse=return_softmax_stats,
    )


def contract_cpu_reference(inputs):
    """Independent FP32 formula with layouts, block expansion and causal clipping."""
    x = {
        name: value.cpu() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    q, qr, k, kr = (
        x[name].float() for name in ("query", "query_rope", "key", "key_rope")
    )
    query_is_tnd, kv_is_tnd = x["layout_query"] == "TND", x["layout_kv"] == "TND"
    paged = x["layout_kv"] == "PA_BSND"
    query_lengths, kv_lengths = (
        x["actual_seq_lengths_query"],
        x["actual_seq_lengths_kv"],
    )
    num_requests = query_lengths.numel() if query_is_tnd else q.shape[0]
    ends_q = query_lengths.tolist() if query_is_tnd else None
    ends_k = kv_lengths.tolist() if kv_is_tnd else None
    qlens = (
        [end - (ends_q[j - 1] if j else 0) for j, end in enumerate(ends_q)]
        if query_is_tnd
        else (
            query_lengths.tolist()
            if query_lengths is not None
            else [q.shape[1]] * num_requests
        )
    )
    klens = (
        [end - (ends_k[j - 1] if j else 0) for j, end in enumerate(ends_k)]
        if kv_is_tnd
        else (
            kv_lengths.tolist()
            if kv_lengths is not None
            else [k.shape[1]] * num_requests
        )
    )
    out = torch.zeros_like(q)
    softmax_max = torch.zeros(q.shape[:-1], dtype=torch.float32)
    softmax_sum = torch.zeros_like(softmax_max)
    sparse_block = x["sparse_block_size"]
    for request_idx in range(num_requests):
        qstart = ends_q[request_idx - 1] if query_is_tnd and request_idx else 0
        kstart = ends_k[request_idx - 1] if kv_is_tnd and request_idx else 0
        for pos in range(qlens[request_idx]):
            token = qstart + pos if query_is_tnd else (request_idx, pos)
            kv_end_exclusive = klens[request_idx]
            if x["sparse_mode"] == 3:
                kv_end_exclusive = max(
                    0,
                    min(
                        kv_end_exclusive,
                        kv_end_exclusive - qlens[request_idx] + pos + 1,
                    ),
                )
            count = min(
                x["sparse_indices"].shape[-1],
                (kv_end_exclusive + sparse_block - 1) // sparse_block,
            )
            indices = x["sparse_indices"][token][0, :count].long()
            indices = indices[indices >= 0]
            rows = (
                indices[:, None] * sparse_block + torch.arange(sparse_block)
            ).flatten()
            rows = rows[(rows >= 0) & (rows < kv_end_exclusive)]
            if rows.numel() == 0:
                # vLLM's <=4-token template uses finite masked scores, so its
                # auxiliary values differ from the mathematical empty sum.
                if kv_end_exclusive > 0 and sparse_block <= 4:
                    softmax_max[token] = -2.0e38
                    softmax_sum[token] = min(
                        x["sparse_indices"].shape[-1] * sparse_block, kv_end_exclusive
                    )
                continue
            if paged:
                page_size = k.shape[1]
                physical = (
                    x["block_table"][request_idx, rows // page_size].long() * page_size
                    + rows % page_size
                )
                gathered_kv = k.view(-1, 512)[physical]
                gathered_key_rope = kr.view(-1, 64)[physical]
            elif kv_is_tnd:
                gathered_kv, gathered_key_rope = (
                    k[kstart + rows, 0],
                    kr[kstart + rows, 0],
                )
            else:
                gathered_kv, gathered_key_rope = (
                    k[request_idx, rows, 0],
                    kr[request_idx, rows, 0],
                )
            score = (q[token] @ gathered_kv.T + qr[token] @ gathered_key_rope.T) * x[
                "scale_value"
            ]
            m = score.max(-1).values
            p = (score - m[:, None]).exp()
            out[token] = (p / p.sum(-1, keepdim=True)) @ gathered_kv
            softmax_max[token], softmax_sum[token] = m, p.sum(-1)
    softmax_stats_shape = (
        (1, q.shape[0], q.shape[1])
        if query_is_tnd
        else (q.shape[0], 1, q.shape[1], q.shape[2])
    )
    return (
        out,
        softmax_max.reshape(softmax_stats_shape),
        softmax_sum.reshape(softmax_stats_shape),
    )

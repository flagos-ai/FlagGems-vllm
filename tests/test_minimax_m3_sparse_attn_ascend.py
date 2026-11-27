# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check separate paged K/V views through the public MiniMax M3 APIs."""

import pytest
import torch

import flaggems_vllm


@pytest.mark.parametrize(
    "decode",
    [
        pytest.param(False, marks=pytest.mark.minimax_m3_sparse_attn),
        pytest.param(True, marks=pytest.mark.minimax_m3_sparse_attn_decode),
    ],
)
@pytest.mark.parametrize("groups,dim,stride", [(1, 128, 1), (4, 128, 1), (4, 96, 2)])
def test_selected_pages(decode, groups, dim, stride):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires an Ascend NPU")
    generator = torch.Generator().manual_seed(291)
    kv_heads = 2
    lengths = [1, 1] if decode else [3, 2]
    seq = [131, 7]
    prefix = [130, 6] if decode else [126, 2]
    offsets = [0, lengths[0], sum(lengths)]
    q = torch.randn(
        offsets[-1], groups * kv_heads, dim, dtype=torch.bfloat16, generator=generator
    )
    cache = torch.randn(4, 2, 128, kv_heads, dim, dtype=q.dtype, generator=generator)
    table = [[2, 0, 9], [1, -1, -1]]
    cache[3].fill_(float("nan"))
    cache[0, :, 3:].fill_(float("nan"))
    selected = torch.tensor([1, -1, 0, 2, 7], dtype=torch.int32).repeat(
        kv_heads, offsets[-1], 1
    )
    selected[1, 0].fill_(-1)
    expected = torch.zeros_like(q)
    scale = dim**-0.5
    for request, length in enumerate(lengths):
        for local in range(length):
            row = offsets[request] + local
            for head in range(q.shape[1]):
                kh = head // groups
                positions = [
                    (table[request][block], token)
                    for block in selected[kh, row].tolist()
                    if 0 <= block < len(table[request])
                    and 0 <= table[request][block] < cache.shape[0]
                    for token in range(128)
                    if block * 128 + token
                    <= min(prefix[request] + local, seq[request] - 1)
                ]
                if positions:
                    pages, tokens = zip(*positions)
                    kv = cache[list(pages), :, list(tokens), kh].double()
                    scores = kv[:, 0] @ q[row, head].double()
                    expected[row, head] = (scores * scale).softmax(0) @ kv[:, 1]
    device_cache = cache.to("npu")
    storage = torch.full(
        (*q.shape[:2], dim * stride), float("nan"), dtype=q.dtype, device="npu"
    )
    out = storage[..., ::stride]
    device_q = torch.empty_like(storage)[..., ::stride]
    device_q.copy_(q)
    args = (
        device_q,
        (device_cache[:, 0], device_cache[:, 1]),
        selected.to("npu"),
        torch.tensor(table, dtype=torch.int32, device="npu"),
    )
    seq_tensor = torch.tensor(seq, dtype=torch.int32, device="npu")
    if decode:
        result = flaggems_vllm.minimax_m3_sparse_attn_decode(
            *args, seq_tensor, kv_heads, scale, out, 1
        )
    else:
        result = flaggems_vllm.minimax_m3_sparse_attn(
            *args,
            torch.tensor(offsets, dtype=torch.int32, device="npu"),
            seq_tensor,
            torch.tensor(prefix, dtype=torch.int32, device="npu"),
            max(lengths),
            kv_heads,
            scale,
            out,
        )
    assert result is None
    if stride > 1:
        assert storage[..., 1::stride].isnan().all().item()
    torch.testing.assert_close(out.cpu(), expected, atol=0.01, rtol=0.02)


@pytest.mark.parametrize(
    "decode",
    [
        pytest.param(False, marks=pytest.mark.minimax_m3_sparse_attn),
        pytest.param(True, marks=pytest.mark.minimax_m3_sparse_attn_decode),
    ],
)
def test_replay_reads_live_metadata(decode):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires an Ascend NPU")
    q = torch.ones(1, 2, 128, device="npu", dtype=torch.bfloat16)
    cache = torch.zeros(2, 2, 128, 1, 128, device="npu", dtype=q.dtype)
    k, v = cache.unbind(1)
    idx = torch.zeros(1, 1, 1, device="npu", dtype=torch.int32)
    table = idx.view(1, 1).clone()
    seq = torch.ones(1, device="npu", dtype=torch.int32)
    prefix = torch.zeros_like(seq)
    cu = torch.arange(2, device="npu", dtype=torch.int32)
    out = torch.empty_like(q)

    def run():
        if decode:
            return flaggems_vllm.minimax_m3_sparse_attn_decode(
                q, (k, v), idx, table, seq, 1, 1.0, out, 1
            )
        return flaggems_vllm.minimax_m3_sparse_attn(
            q, (k, v), idx, table, cu, seq, prefix, 1, 1, 1.0, out
        )

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    q.fill_(2)
    table.fill_(1)
    seq.fill_(2)
    prefix.fill_(1)
    v[1, 0].fill_(2)
    v[1, 1].fill_(4)
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out.cpu(), torch.full_like(out.cpu(), 3))
    idx.fill_(-1)
    out.fill_(float("nan"))
    graph.replay()
    assert torch.count_nonzero(out.cpu()).item() == 0
    idx.zero_()
    seq.zero_()
    out.fill_(float("nan"))
    graph.replay()
    assert torch.count_nonzero(out.cpu()).item() == 0


@pytest.mark.minimax_m3_sparse_attn
def test_empty_batch_clears_padded_rows():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires an Ascend NPU")
    q = torch.ones(2, 1, 128, device="npu", dtype=torch.bfloat16)
    kv = torch.empty(0, 128, 1, 128, device=q.device, dtype=q.dtype)
    ids = torch.empty(1, 2, 0, device=q.device, dtype=torch.int32)
    lens = torch.empty(0, device=q.device, dtype=torch.int32)
    offsets = torch.zeros(1, device=q.device, dtype=torch.int32)
    out = torch.full_like(q, float("nan"))
    flaggems_vllm.minimax_m3_sparse_attn(
        q, (kv, kv), ids, lens.view(0, 1), offsets, lens, lens, 0, 1, 1.0, out
    )
    assert torch.count_nonzero(out.cpu()).item() == 0

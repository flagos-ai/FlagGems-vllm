# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm import fp8_fp4_paged_mqa_logits_graph
from flaggems_vllm.ops import fp8_fp4_paged_mqa_logits


def make_inputs(batch, next_n, page_size, fp4, heads=64, max_len=4096):
    device = "cuda"
    pages = 2 * max_len // page_size
    if fp4:
        q = torch.randint(
            -128, 128, (batch, next_n, heads, 64), device=device, dtype=torch.int8
        )
        scales = torch.full(
            (batch, next_n, heads), 0x7F7F7F7F, device=device, dtype=torch.int32
        )
    else:
        q = torch.randn(batch, next_n, heads, 128, device=device).to(
            torch.float8_e4m3fn
        )
        scales = None
    k = torch.randn(pages, page_size, 128, device=device).to(torch.float8_e4m3fn)
    k_scales = torch.rand(pages, page_size, device=device) * 0.01 + 0.001
    # Pad physical-page stride; each page retains the specified packed layout.
    storage = torch.empty(pages, page_size * 132 + 16, device=device, dtype=torch.uint8)
    packed = storage[:, : page_size * 132]
    packed[:, : page_size * 128] = k.view(torch.uint8).reshape(pages, -1)
    packed[:, page_size * 128 :] = k_scales.view(torch.uint8).reshape(pages, -1)
    cache = packed.view(pages, page_size, 1, 132)
    weights = torch.randn(batch * next_n, heads, device=device)
    # Both metadata tensors deliberately have nontrivial row strides.
    lengths = torch.zeros(batch, next_n + 1, device=device, dtype=torch.int32)[
        :, :next_n
    ]
    tables = torch.randint(
        0, pages, (batch, max_len // page_size + 3), device=device, dtype=torch.int32
    )
    tables = tables[:, : max_len // page_size]
    return (q, scales), cache, weights, lengths, tables


def reference(inputs, max_len, clean):
    (q, scales), cache, weights, lengths, tables = inputs
    b, n, h, d = q.shape
    if scales is None:
        q_ref = q.float()
    else:
        bits = q.view(torch.uint8)
        nibbles = torch.stack((bits & 15, bits >> 4), dim=-1).reshape(b, n, h, 128)
        values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=q.device,
        )
        block = torch.arange(128, device=q.device) // 32
        scale = torch.exp2(((scales[..., None] >> (8 * block)) & 255).float() - 127)
        q_ref = (values[nibbles.long()] * scale).half().float()
    page_size = cache.shape[1]
    raw = cache.view(cache.shape[0], -1)
    keys = raw[:, : page_size * 128].reshape(-1, 128).view(torch.float8_e4m3fn).float()
    kv_scales = raw[:, page_size * 128 :].contiguous().view(torch.float32).reshape(-1)
    out = torch.full((b * n, max_len), float("-inf") if clean else 0.0, device=q.device)
    for row in range(b * n):
        batch, pos = divmod(row, n)
        length = int(lengths[batch, pos])
        ids = torch.arange(length, device=q.device)
        physical = tables[batch, ids // page_size].long() * page_size + ids % page_size
        dots = q_ref[batch, pos] @ keys[physical].T
        out[row, :length] = (
            (dots * kv_scales[physical]).clamp_min(0) * weights[row, :, None]
        ).sum(0)
    return out


@pytest.mark.fp8_fp4_paged_mqa_logits_graph
@pytest.mark.parametrize("batch,next_n", [(1, 1), (1, 2), (4, 1), (4, 2)])
@pytest.mark.parametrize("page_size", [64, 256])
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("clean", [False, True])
def test_graph_replay_refreshes_paged_inputs(batch, next_n, page_size, fp4, clean):
    torch.manual_seed(453)
    args = make_inputs(batch, next_n, page_size, fp4)
    args[3].fill_(65)
    fn = lambda: fp8_fp4_paged_mqa_logits_graph(*args, None, 4096, clean)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    addresses = [x.data_ptr() for x in [args[0][0], args[1], args[2], args[3], args[4]]]
    for step, lengths in enumerate(
        [[0, 1, 63, 64], [65, 2047, 2048, 2049], [3016, 4096, 7, 0]]
    ):
        fresh = make_inputs(batch, next_n, page_size, fp4)
        args[0][0].copy_(fresh[0][0])
        if fp4:
            args[0][1].fill_(0x7E7E7E7E if step % 2 else 0x7F7F7F7F)
        for target, source in zip(args[1:], fresh[1:]):
            target.copy_(source)
        for row in range(batch * next_n):
            args[3][row // next_n, row % next_n] = lengths[row % len(lengths)]
        captured.fill_(float("nan"))
        graph.replay()
        expected = reference(args, 4096, clean)
        torch.testing.assert_close(captured, expected, atol=0.002, rtol=0.001)
        legacy = fp8_fp4_paged_mqa_logits(*args, None, 4096, clean)
        torch.testing.assert_close(captured, legacy, atol=0, rtol=0)
        assert addresses == [
            x.data_ptr() for x in [args[0][0], args[1], args[2], args[3], args[4]]
        ]


@pytest.mark.parametrize("heads", [16, 32])
def test_graph_logits_sequence_lengths_and_3d_query(heads):
    args = list(make_inputs(4, 1, 64, False, heads=heads))
    args[0] = (args[0][0].squeeze(1), None)
    args[3] = args[3][:, 0]
    args[3].copy_(torch.tensor([0, 1, 2049, 4096], device="cuda", dtype=torch.int32))
    out = fp8_fp4_paged_mqa_logits_graph(*args, None, 4096)
    legacy = fp8_fp4_paged_mqa_logits(*args, None, 4096)
    torch.testing.assert_close(out, legacy, atol=0, rtol=0)


def test_graph_logits_empty_and_unsupported_dtype():
    args = list(make_inputs(0, 1, 64, False))
    out = fp8_fp4_paged_mqa_logits_graph(*args, None, 4096)
    assert out.shape == (0, 4096)
    args = list(make_inputs(1, 1, 64, False))
    args[0] = (args[0][0].float(), None)
    with pytest.raises(NotImplementedError, match="FP8 e4m3fn"):
        fp8_fp4_paged_mqa_logits_graph(*args, None, 4096)

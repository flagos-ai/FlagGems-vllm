# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.mrope_strided_cache import mrope_strided_cache


def reference(q, k, cache, positions):
    index = torch.arange(32)
    h_mask = (index % 3 == 1) & (index <= 30)
    w_mask = (index % 3 == 2) & (index <= 30)
    t_mask = ~(h_mask | w_mask)
    selected = (
        cache[positions[0]].float() * t_mask.repeat(2)
        + cache[positions[1]].float() * h_mask.repeat(2)
        + cache[positions[2]].float() * w_mask.repeat(2)
    )
    cos, sin = selected[:, None, :32], selected[:, None, 32:]

    def rotate(x):
        a, b = x.float()[..., ::2], x.float()[..., 1::2]
        return (
            torch.stack((a * cos - b * sin, b * cos + a * sin), -1)
            .flatten(-2)
            .bfloat16()
        )

    return rotate(q), rotate(k)


@pytest.mark.mrope_strided_cache
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("rows", [4, 64, 1024])
@torch.inference_mode()
def test_mrope_strided_cache(rows):
    generator = torch.Generator(device="cpu").manual_seed(2026)
    q = torch.randn(rows, 4, 96, generator=generator).bfloat16().cuda()[..., :64]
    k = torch.randn(rows, 1, 96, generator=generator).bfloat16().cuda()[..., :64]
    cache = (torch.randn(2048, 64, generator=generator) / 4).bfloat16().cuda()
    positions = torch.stack(tuple(torch.arange(rows) + a for a in range(3))).cuda()
    expected = reference(q.cpu(), k.cpu(), cache.cpu(), positions.cpu())
    actual = mrope_strided_cache(q, k, cache, positions)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, rtol=0.02, atol=0.015625)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        mrope_strided_cache(q, k, cache, positions)
        with torch.cuda.graph(graph, stream=stream):
            captured = mrope_strided_cache(q, k, cache, positions)
    torch.cuda.current_stream().wait_stream(stream)
    q.neg_()
    k.neg_()
    positions.add_(1)
    expected = reference(q.cpu(), k.cpu(), cache.cpu(), positions.cpu())
    for output in captured:
        output.fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    for a, e in zip(captured, expected):
        torch.testing.assert_close(a.cpu(), e, rtol=0.02, atol=0.015625)

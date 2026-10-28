# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Finite inference math and ordinary CUDA graph checks."""

import pytest
import torch

from flaggems_vllm.ops.ihc_post_pre import ihc_post_pre

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps):
    materialized = (
        post[:, :, None] * x.float()[:, None, :] + residual.float()
    ).bfloat16()
    flat = materialized.float().flatten(1)
    rms = torch.rsqrt((flat * flat).mean(1, keepdim=True) + norm_eps)
    mixes = (flat @ weight.t()) * rms
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + hc_eps
    gates = magnitude * torch.sigmoid(mixes[:, 4:] * scale[1] + base[4:]) + hc_eps
    reduced = (pre[:, :, None] * materialized.float()).sum(1).bfloat16()
    return reduced, gates, materialized


def inputs(rows, device):
    generator = torch.Generator(device="cpu").manual_seed(2026 + rows)
    x = torch.randn(rows, 6144, generator=generator).bfloat16().to(device)
    residual = torch.randn(rows, 4, 6144, generator=generator).bfloat16().to(device)
    post = torch.rand(rows, 4, generator=generator).to(device)
    weight = (torch.randn(8, 24576, generator=generator) / 256).to(device)
    scale = torch.tensor([0.5, 0.5], device=device)
    base = torch.zeros(8, device=device)
    return x, residual, post, weight, scale, base, 2.0, 1e-6, 1e-5


@pytest.mark.ihc_post_pre
@pytest.mark.parametrize("rows", [0, 1, 4, 64, 1024])
@torch.inference_mode()
def test_ihc_post_pre(rows):
    args = inputs(rows, "cuda")
    actual = ihc_post_pre(*args)
    if not rows:
        assert [tuple(x.shape) for x in actual] == [(0, 6144), (0, 4), (0, 4, 6144)]
        return
    cpu_args = tuple(x.cpu() if isinstance(x, torch.Tensor) else x for x in args)
    expected = reference(*cpu_args)
    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=0.02, atol=0.0625)
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=2e-5, atol=2e-5)
    assert torch.equal(actual[2].cpu(), expected[2])
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ihc_post_pre(*args)
        with torch.cuda.graph(graph, stream=stream):
            captured = ihc_post_pre(*args)
    torch.cuda.current_stream().wait_stream(stream)
    args[0].neg_()
    cpu_args = tuple(x.cpu() if isinstance(x, torch.Tensor) else x for x in args)
    expected = reference(*cpu_args)
    for output in captured:
        output.fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured[0].cpu(), expected[0], rtol=0.02, atol=0.0625)
    torch.testing.assert_close(captured[1].cpu(), expected[1], rtol=2e-5, atol=2e-5)
    assert torch.equal(captured[2].cpu(), expected[2])

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent paired-iHC math with the existing post/pre tolerances."""

import pytest
import torch

import flaggems_vllm


def reference(x, residual, post, weight, scale, base):
    materialized = (
        post.float()[:, :, None] * x.float()[:, None, :] + residual.float()
    ).bfloat16()
    flat = materialized.double().flatten(1)
    mixes = (flat @ weight.double().t()) * torch.rsqrt(
        (flat * flat).mean(1, keepdim=True) + 1e-5
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1e-6
    gates = 2.0 * torch.sigmoid(mixes[:, 4:] * scale[1] + base[4:]) + 1e-6
    reduced = (pre[:, :, None] * materialized.double()).sum(1).bfloat16()
    return reduced, gates.float(), materialized


@pytest.mark.ihc_post_pre_pair
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="SM90 CUDA required",
)
@pytest.mark.parametrize("rows", [208, 256, 1024])
@torch.inference_mode()
def test_pair_math_and_mutable_graph(rows):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("selected pair schedule is scoped to SM90")
    generator = torch.Generator().manual_seed(2026 + rows)
    cpu = (
        torch.randn(rows, 6144, generator=generator).bfloat16(),
        torch.randn(rows, 4, 6144, generator=generator).bfloat16(),
        torch.rand(rows, 4, generator=generator),
        torch.randn(8, 24576, generator=generator) / 256,
        torch.tensor([0.5, 0.5]),
        torch.zeros(8),
    )
    args = tuple(value.cuda() for value in cpu)

    def run():
        return flaggems_vllm.ihc_post_pre_pair(*args, 2.0, 1e-6, 1e-5)

    def check(actual, source):
        expected = reference(*source)
        torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=0.02, atol=0.0625)
        torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=2e-5, atol=2e-5)
        assert torch.equal(actual[2].cpu(), expected[2])

    assert flaggems_vllm.can_ihc_pair(
        args[0],
        args[3],
        args[4],
        args[5],
        2.0,
        1e-6,
        1e-5,
        residual=args[1],
        post=args[2],
    )
    check(run(), cpu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        run()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
    torch.cuda.current_stream().wait_stream(stream)
    changed = (-cpu[0], *cpu[1:])
    args[0].copy_(changed[0])
    for output in captured:
        output.fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    check(captured, changed)

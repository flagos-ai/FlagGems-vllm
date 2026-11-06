# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Finite raw-bit transport and ordinary CUDA graph replay."""

import pytest
import torch

from flaggems_vllm.ops.concat_pad_mla_q_and_sink import concat_pad_mla_q_and_sink

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.concat_pad_mla_q_and_sink
@pytest.mark.parametrize("rows,rope_heads", [(1, 4), (4, 1), (64, 4), (1024, 1)])
@torch.inference_mode()
def test_concat_pad_mla_q_and_sink(rows, rope_heads):
    generator = torch.Generator(device="cpu").manual_seed(77)
    nope = torch.randn(rows, 4, 512, generator=generator).bfloat16().cuda()
    rope = torch.randn(rows, rope_heads, 64, generator=generator).bfloat16().cuda()
    sink = torch.tensor([-0.0, 0.5, -0.5, 2.0], device="cuda")
    out_q = torch.empty(rows, 64, 576, dtype=torch.bfloat16, device="cuda")
    out_sink = torch.empty(64, device="cuda")
    concat_pad_mla_q_and_sink(nope, rope, sink, out_q, out_sink)
    expected = torch.zeros_like(out_q)
    expected[:, :4] = torch.cat((nope, rope.expand(rows, 4, 64)), -1)
    expected_sink = torch.full_like(out_sink, float("-inf"))
    expected_sink[:4] = sink
    assert torch.equal(out_q.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(out_sink.view(torch.int32), expected_sink.view(torch.int32))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        concat_pad_mla_q_and_sink(nope, rope, sink, out_q, out_sink)
        with torch.cuda.graph(graph, stream=stream):
            concat_pad_mla_q_and_sink(nope, rope, sink, out_q, out_sink)
    torch.cuda.current_stream().wait_stream(stream)
    nope.neg_()
    rope.neg_()
    expected[:, :4] = torch.cat((nope, rope.expand(rows, 4, 64)), -1)
    out_q.fill_(17)
    out_sink.fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_q.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(out_sink.view(torch.int32), expected_sink.view(torch.int32))

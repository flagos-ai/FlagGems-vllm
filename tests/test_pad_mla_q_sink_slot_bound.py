# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Raw-bit transport and last-slot endpoint, including internal holes."""

import pytest
import torch

import flaggems_vllm


@pytest.mark.pad_mla_q_sink_slot_bound
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("tokens", [1, 3, 17])
@torch.inference_mode()
def test_padding_endpoint_and_fresh_input_graph(tokens):
    bits = torch.arange(tokens * 4 * 576, dtype=torch.int32).to(torch.int16)
    q = bits.view(torch.bfloat16).reshape(tokens, 4, 576).cuda()
    sink = torch.tensor([0.0, -0.0, 1.0, -2.0], device="cuda")
    indices = torch.full((tokens, 2048), -1, device="cuda", dtype=torch.int32)
    indices[:, 0] = 0
    indices[:, 2047] = 9
    out_q = torch.empty((tokens, 64, 576), dtype=q.dtype, device="cuda")
    out_sink = torch.empty((64,), device="cuda")
    out_bound = torch.empty((tokens,), dtype=torch.int32, device="cuda")

    def check(endpoint):
        assert torch.equal(out_q[:, :4].view(torch.int16), q.view(torch.int16))
        assert torch.count_nonzero(out_q[:, 4:].view(torch.int16)) == 0
        assert torch.equal(out_sink[:4].view(torch.int32), sink.view(torch.int32))
        assert torch.isneginf(out_sink[4:]).all()
        assert torch.equal(out_bound, torch.full_like(out_bound, endpoint))

    result = flaggems_vllm.pad_mla_q_sink_slot_bound(
        q, sink, indices, out_q, out_sink, out_bound
    )
    assert all(a is b for a, b in zip(result, (out_q, out_sink, out_bound)))
    check(2048)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        flaggems_vllm.pad_mla_q_sink_slot_bound(
            q, sink, indices, out_q, out_sink, out_bound
        )
        with torch.cuda.graph(graph, stream=stream):
            flaggems_vllm.pad_mla_q_sink_slot_bound(
                q, sink, indices, out_q, out_sink, out_bound
            )
    torch.cuda.current_stream().wait_stream(stream)
    q.view(torch.int16).bitwise_xor_(3)
    indices.fill_(-1)
    indices[:, 61] = 4
    out_q.fill_(17)
    out_sink.fill_(17)
    out_bound.fill_(-17)
    graph.replay()
    torch.cuda.synchronize()
    check(62)

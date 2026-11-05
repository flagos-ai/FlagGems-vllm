# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-token quantization at the two selected MoE widths."""

import pytest
import torch

import flaggems_vllm


def reference(x):
    epsilon = torch.tensor(1e-10).to(x.dtype).float()
    amax = torch.maximum(x.float().abs().amax(-1, keepdim=True), epsilon)
    scale = amax * torch.tensor(1.0 / 127.0, dtype=torch.float32)
    values = (x.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return values, scale


@pytest.mark.int8_quantize_per_token
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("width", [2048, 6144])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_quantize_per_token_independent_reference_and_graph(width, dtype):
    generator = torch.Generator().manual_seed(7200 + width)
    cpu = torch.randn((4, width), generator=generator).to(dtype)
    cpu[0].zero_()
    cpu[1, 0] = torch.finfo(dtype).max
    x = cpu.cuda()
    assert flaggems_vllm.supports_int8_quantize_per_token(x)

    def check(actual, source):
        expected = reference(source)
        assert torch.equal(actual[0].cpu(), expected[0])
        assert torch.equal(actual[1].cpu(), expected[1])

    check(flaggems_vllm.int8_quantize_per_token(x), cpu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        flaggems_vllm.int8_quantize_per_token(x)
        with torch.cuda.graph(graph, stream=stream):
            captured = flaggems_vllm.int8_quantize_per_token(x)
    torch.cuda.current_stream().wait_stream(stream)
    x.copy_(-cpu)
    captured[0].fill_(17)
    captured[1].fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    check(captured, -cpu)

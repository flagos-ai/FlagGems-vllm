# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Clamped activation and FP32 reciprocal-multiply quantization reference."""

import pytest
import torch

import flaggems_vllm


def reference(packed, limit):
    gate, up = packed.float().chunk(2, -1)
    gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    activation = (gate / (1 + torch.exp(-gate)) * up).to(packed.dtype)
    z = activation.float()
    amax = z.abs().amax(-1, keepdim=True).to(packed.dtype).float()
    epsilon = torch.tensor(1e-10).to(packed.dtype).float()
    amax = torch.maximum(amax, epsilon)
    scale = amax * torch.tensor(1.0 / 127.0, dtype=torch.float32)
    quant = (z / scale).round().clamp(-128, 127).to(torch.int8)
    return activation, quant, scale


@pytest.mark.silu_clamp_int8_quant
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("rows", [1, 208, 256])
@torch.inference_mode()
def test_clamp_quant_reference_and_graph(rows):
    generator = torch.Generator().manual_seed(5200 + rows)
    cpu = torch.randn(rows, 4096, generator=generator).bfloat16()
    packed = cpu.cuda()
    activation = torch.empty((rows, 2048), dtype=packed.dtype, device="cuda")
    assert flaggems_vllm.supports_silu_clamp_int8_quant(packed, activation, 10.0)

    def check(q, scale, source):
        expected, _, _ = reference(source, 10.0)
        torch.testing.assert_close(activation.cpu(), expected, rtol=0.01, atol=0.03125)
        # Quantization consumes the materialized activation, whose BF16
        # boundary is observable even when SiLU exp differs by one output ULP.
        rounded = activation.cpu().float()
        amax = rounded.abs().amax(-1, keepdim=True)
        gold_scale = amax * torch.tensor(1 / 127, dtype=torch.float32)
        gold_q = (rounded / gold_scale).round().clamp(-128, 127).to(torch.int8)
        assert torch.equal(q.cpu(), gold_q)
        assert torch.equal(scale.cpu(), gold_scale)

    q, scale = flaggems_vllm.silu_clamp_int8_quant(packed, activation, 10.0)
    check(q, scale, cpu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        flaggems_vllm.silu_clamp_int8_quant(packed, activation, 10.0)
        with torch.cuda.graph(graph, stream=stream):
            q, scale = flaggems_vllm.silu_clamp_int8_quant(packed, activation, 10.0)
    torch.cuda.current_stream().wait_stream(stream)
    packed.copy_(-cpu)
    activation.fill_(17)
    q.fill_(17)
    scale.fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    check(q, scale, -cpu)

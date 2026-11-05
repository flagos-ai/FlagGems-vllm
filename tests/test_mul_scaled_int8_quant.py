# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native finite quantization contract and ordinary graph replay."""

import pytest
import torch

from flaggems_vllm.ops.mul_scaled_int8_quant import mul_scaled_int8_quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(a, b):
    product = (a * b).bfloat16()
    z = product.float()
    absmax = z.abs().amax(-1, keepdim=True)
    scale = absmax / 127.0
    inverse = torch.where(absmax == 0, 0.0, 127.0 / absmax)
    q = (z * inverse).round().clamp(-128, 127).to(torch.int8)
    return q, scale, product


@pytest.mark.mul_scaled_int8_quant
@pytest.mark.parametrize(
    "rows,width", [(0, 1024), (1, 1024), (4, 1024), (64, 1024), (1024, 1024), (4, 1025)]
)
@torch.inference_mode()
def test_mul_scaled_int8_quant(rows, width):
    generator = torch.Generator(device="cpu").manual_seed(42)
    a_cpu = torch.randn(rows, width, generator=generator).bfloat16()
    b_cpu = torch.randn(rows, width, generator=generator).bfloat16()
    if rows:
        a_cpu[0].zero_()
    # Wider row strides are part of the API.
    a = torch.empty(rows, width + 7, dtype=torch.bfloat16, device="cuda")[:, :width]
    b = torch.empty(rows, width + 11, dtype=torch.bfloat16, device="cuda")[:, :width]
    a.copy_(a_cpu)
    b.copy_(b_cpu)
    product = torch.empty_like(a)
    q, scale = mul_scaled_int8_quant(a, b, product_out=product)
    expected = reference(a_cpu, b_cpu)
    assert torch.equal(q.cpu(), expected[0])
    assert torch.equal(scale.cpu(), expected[1])
    assert torch.equal(product.cpu(), expected[2])
    if not rows:
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        mul_scaled_int8_quant(a, b)
        with torch.cuda.graph(graph, stream=stream):
            graph_q, graph_scale = mul_scaled_int8_quant(a, b)
    torch.cuda.current_stream().wait_stream(stream)
    a_cpu.neg_()
    a.copy_(a_cpu)
    graph_q.fill_(17)
    graph_scale.fill_(17)
    expected = reference(a_cpu, b_cpu)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_q.cpu(), expected[0])
    assert torch.equal(graph_scale.cpu(), expected[1])

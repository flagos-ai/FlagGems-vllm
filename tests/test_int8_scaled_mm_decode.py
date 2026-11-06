# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Qualified SM90 output-projection geometry, finite math and ordinary graph."""

import pytest
import torch

from flaggems_vllm.ops.int8_scaled_mm_decode import int8_scaled_mm_decode
from flaggems_vllm.ops.triton_scaled_mm import triton_scaled_mm


@pytest.mark.int8_scaled_mm_decode
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 4, 64])
@torch.inference_mode()
def test_int8_scaled_mm_decode(rows):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("qualified SM90 kernel contract")
    generator = torch.Generator(device="cpu").manual_seed(2026)
    a_cpu = torch.randint(-16, 17, (rows, 1024), generator=generator, dtype=torch.int8)
    b_storage = torch.randint(
        -16, 17, (6144, 1024), generator=generator, dtype=torch.int8
    )
    scale_a_cpu = torch.rand(rows, 1, generator=generator) / 32
    scale_b_cpu = torch.rand(6144, 1, generator=generator) / 32
    a = a_cpu.cuda()
    b = b_storage.cuda().t()
    scale_a, scale_b = scale_a_cpu.cuda(), scale_b_cpu.cuda()
    actual = int8_scaled_mm_decode(a, b, scale_a, scale_b)
    # FP64 integer products/sums are exact for these bounded finite inputs.
    accumulator = (a_cpu.double() @ b_storage.t().double()).float()
    expected = (scale_a_cpu * (scale_b_cpu.t() * accumulator)).bfloat16()
    assert torch.equal(actual.cpu(), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        int8_scaled_mm_decode(a, b, scale_a, scale_b)
        with torch.cuda.graph(graph, stream=stream):
            captured = int8_scaled_mm_decode(a, b, scale_a, scale_b)
    torch.cuda.current_stream().wait_stream(stream)
    a_cpu.neg_()
    a.copy_(a_cpu)
    captured.fill_(17)
    accumulator = (a_cpu.double() @ b_storage.t().double()).float()
    expected = (scale_a_cpu * (scale_b_cpu.t() * accumulator)).bfloat16()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.cpu(), expected)


@pytest.mark.int8_scaled_mm_decode
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_shared_kernel_preserves_both_scale_orders():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("qualified SM90 kernel contract")
    # INT32 dot = 127*127 + 1 = 16130. These FP32 scale bits straddle a
    # BF16 rounding boundary depending on which scale is multiplied first.
    a_cpu = torch.zeros(1, 1024, dtype=torch.int8)
    a_cpu[0, :2] = torch.tensor([127, 1], dtype=torch.int8)
    b_cpu = torch.zeros(6144, 1024, dtype=torch.int8)
    b_cpu[:, 0] = 127
    b_cpu[:, 1] = 1
    scale_a_cpu = torch.tensor([1012647574], dtype=torch.int32).view(torch.float32)
    scale_b_cpu = torch.tensor([999179928], dtype=torch.int32).view(torch.float32)
    accumulator = torch.tensor(16130.0)
    default_expected = ((scale_a_cpu * accumulator) * scale_b_cpu).bfloat16()
    decode_expected = ((scale_b_cpu * accumulator) * scale_a_cpu).bfloat16()
    assert default_expected.item() == 0.9375
    assert decode_expected.item() == 0.94140625
    a, b = a_cpu.cuda(), b_cpu.cuda().t()
    scale_a = scale_a_cpu.reshape(1, 1).cuda()
    scale_b = scale_b_cpu.expand(6144).contiguous().reshape(6144, 1).cuda()
    default = triton_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16)
    decode = int8_scaled_mm_decode(a, b, scale_a, scale_b)
    assert torch.equal(default.cpu(), default_expected.expand_as(default))
    assert torch.equal(decode.cpu(), decode_expected.expand_as(decode))

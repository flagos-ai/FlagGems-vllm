# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent affine reference and refreshed-input graph replay."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import fused_mm_input_norm


def reference(inputs, outputs, weight, bias):
    return (inputs.float() * weight[None, :, None] + bias[None, :, None]).to(
        outputs.dtype
    )


@pytest.mark.fused_mm_input_norm
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("rows", [0, 1, 17])
@pytest.mark.parametrize(
    "source_dtype", [torch.uint8, torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("target_dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_fused_mm_input_norm(rows, source_dtype, target_dtype):
    x = torch.randint(0, 256, (rows, 3, 256), device="cuda").to(source_dtype)
    weight = torch.tensor([0.25, 0.125, 0.5], device="cuda")
    bias = torch.tensor([-2.0, 1.0, -3.0], device="cuda")
    output = torch.full(x.shape, float("nan"), dtype=target_dtype, device="cuda")
    actual = fused_mm_input_norm(x, output, weight, bias)
    assert actual is output
    torch.testing.assert_close(
        actual, reference(x, output, weight, bias), rtol=0, atol=0
    )
    if not rows:
        return
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_mm_input_norm(x, output, weight, bias)
        with torch.cuda.graph(graph, stream=stream):
            fused_mm_input_norm(x, output, weight, bias)
    torch.cuda.current_stream().wait_stream(stream)
    x.zero_()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output, reference(x, output, weight, bias), rtol=0, atol=0
    )


@pytest.mark.fused_mm_input_norm
def test_fused_mm_input_norm_rejects_cpu_before_launch():
    x = torch.zeros(1, 3, 4, dtype=torch.uint8)
    output = torch.full(x.shape, 17.0)
    with pytest.raises(NotImplementedError, match="NVIDIA CUDA"):
        fused_mm_input_norm(x, output, torch.ones(3), torch.zeros(3))
    assert torch.equal(output, torch.full_like(output, 17.0))


@pytest.mark.fused_mm_input_norm
def test_fused_mm_input_norm_rejects_shape_before_launch():
    x = torch.zeros(1, 3, 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="share shape"):
        fused_mm_input_norm(x, torch.empty(1, 3, 5), torch.ones(3), torch.zeros(3))

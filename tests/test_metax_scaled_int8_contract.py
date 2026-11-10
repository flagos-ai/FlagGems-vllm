# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent public-API contract checks for the existing MetaX quantizer."""
import inspect

import pytest
import torch

import flaggems_vllm as ops


def reference(x):
    value = x.cpu().float().reshape(-1, x.shape[-1])
    absolute = torch.where(torch.isnan(value), 0.0, value.abs())
    maximum = absolute.amax(-1, keepdim=True)
    scale = maximum / 127.0
    inverse = torch.where(maximum == 0, 0.0, 127.0 / maximum)
    rounded = (value * inverse).round()
    rounded = torch.where(torch.isnan(rounded), 0.0, rounded)
    return rounded.clamp(-128, 127).to(torch.int8).reshape(x.shape), scale


def check(output, scale, x):
    expected, expected_scale = reference(x)
    torch.testing.assert_close(output.cpu(), expected, atol=1, rtol=0)
    torch.testing.assert_close(
        scale.cpu(), expected_scale, atol=0, rtol=1e-6, equal_nan=True
    )


@pytest.mark.scaled_int8_quant
@pytest.mark.parametrize("width", [17, 4096, 5137, 13824, 16401])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_metax_scaled_int8_layout_empty_and_numerical_boundaries(width, dtype):
    assert ops.scaled_int8_quant.__module__ == "flaggems_vllm.ops.scaled_int8_quant"
    assert inspect.signature(ops.scaled_int8_quant).parameters["scale"].default is None
    backing = torch.randn(10, width * 2 + 2, dtype=dtype, device="cuda")
    x = backing[1:10:2, 1 : width * 2 + 1 : 2]
    assert not x.is_contiguous() and x.storage_offset() > 0
    x[0].zero_()
    q, scale, azp = ops.scaled_int8_quant(x)
    assert azp is None
    check(q, scale, x)
    q, scale, azp = ops.scaled_int8_quant(x[:0])
    assert q.shape == (0, width) and scale.shape == (0, 1) and azp is None
    special = torch.zeros(9, width, device="cuda", dtype=dtype)
    tiny = torch.finfo(dtype).tiny
    special[1].fill_(tiny)
    special[2].fill_(torch.finfo(dtype).max)
    special[3, ::2] = 0.125
    special[3, 1::2] = -0.125
    special[4, 0] = float("nan")
    special[5, 0] = float("inf")
    special[6, 0] = -float("inf")
    special[7, 0] = tiny
    special[8, 0] = tiny / 2
    q, scale, _ = ops.scaled_int8_quant(special)
    check(q, scale, special)


@pytest.mark.scaled_int8_quant
@pytest.mark.parametrize(
    "rows,width",
    [
        (1, 4096),
        (64, 4096),
        (65, 1024),
        (3, 13824),
        (3, 16401),
        (129, 512),
        (129, 1024),
        (129, 4096),
        (4096, 4096),
    ],
)
def test_metax_scaled_int8_graph_changed_input(rows, width):
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    result = []

    def call():
        result[:] = ops.scaled_int8_quant(x)

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for factor in (0.125, -0.25):
        x.copy_(torch.randn_like(x) * factor)
        if rows > 1:
            x[0].zero_()
        result[0].fill_(-33)
        result[1].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(result[0], result[1], x)


@pytest.mark.scaled_int8_quant
@pytest.mark.parametrize("width", [17, 4096])
def test_metax_scaled_int8_all_bf16_constant_patterns(width):
    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    # Include a zero lane so the documented fmax reduction has a non-NaN value.
    x = raw[:, None].expand(-1, width).contiguous()
    x[:, -1] = 0
    x = x.to("cuda")
    q, scale, _ = ops.scaled_int8_quant(x)
    check(q, scale, x)


@pytest.mark.scaled_int8_quant
def test_metax_scaled_int8_leading_dims_and_invalid_inputs():
    x = torch.randn(2, 3, 17, device="cuda", dtype=torch.bfloat16)
    q, s, z = ops.scaled_int8_quant(x)
    check(q, s, x)
    assert q.shape == x.shape and s.shape == (6, 1) and z is None
    with pytest.raises(ValueError, match="row width"):
        ops.scaled_int8_quant(x[..., :0])
    with pytest.raises(TypeError, match="floating tensor"):
        ops.scaled_int8_quant(x.to(torch.int8))
    with pytest.raises(ValueError, match="zero point"):
        ops.scaled_int8_quant(x, scale=torch.ones(1, device=x.device), symmetric=False)

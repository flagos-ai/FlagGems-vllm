# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8 as vllm_per_token_quant_int8,
)

import flaggems_vllm

from .conftest import QUICK_MODE

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = (
    [(7, 1025)]
    if QUICK_MODE
    else [
        (1, 1),
        (7, 17),
        (7, 128),
        (32, 1024),
        (65, 1024),
        (7, 1025),
        (32, 4096),
        (7, 5137),
        (64, 8192),
        (7, 13824),
        (0, 128),
    ]
)


def _check_quantization(x):
    ref_q, ref_scale = vllm_per_token_quant_int8(x)
    q, scale = flaggems_vllm.per_token_quant_int8(x)
    assert q.shape == x.shape and q.dtype == torch.int8
    assert scale.shape == x.shape[:-1] + (1,) and scale.dtype == torch.float32
    assert q.device == x.device and scale.device == x.device
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0, equal_nan=True)
    return q, scale


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
def test_per_token_quant_int8(shape, dtype, layout):
    torch.manual_seed(0)
    m, n = shape
    if layout == "strided":
        x = torch.randn((m * 2, n * 2), dtype=dtype, device=flaggems_vllm.device)
        x = x[::2, ::2]
    elif layout == "transposed":
        x = torch.randn((n, m), dtype=dtype, device=flaggems_vllm.device).t()
    else:
        x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    _check_quantization(x)


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(2, 3, 257), (2, 1, 3, 128), (2, 0, 128)])
def test_per_token_quant_int8_leading_dims(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    _check_quantization(x)


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("dtype", DTYPES)
def test_per_token_quant_int8_special_values(dtype):
    x = torch.tensor(
        [
            [-127, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127],
            [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0],
            [1e-12, -1e-12, 2e-12, -2e-12, 0.0, 0.0, 0.0, 0.0],
            [float("nan"), 1.0, -1.0, 0.0, 2.0, -2.0, 0.0, 0.0],
            [float("inf"), -float("inf"), 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=flaggems_vllm.device,
    )
    q, scale = _check_quantization(x)
    expected = torch.tensor(
        [-127, -3, -2, -1, 1, 2, 3, 127], dtype=torch.int8, device=x.device
    )
    torch.testing.assert_close(q[0], expected, rtol=0, atol=0)
    assert scale[1].item() > 0


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [128, 256])
def test_per_token_quant_int8_16bit_patterns(dtype, width):
    torch.manual_seed(0)
    bits = torch.arange(65536, device=flaggems_vllm.device, dtype=torch.int32)
    bits = bits.to(torch.int16)
    order = torch.randperm(bits.numel(), device=bits.device)
    x = bits[order].view(dtype).reshape(-1, width)
    _check_quantization(x)


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("dtype", DTYPES)
def test_per_token_quant_int8_cuda_graph(dtype):
    x = torch.randn((7, 1025), dtype=dtype, device=flaggems_vllm.device)
    flaggems_vllm.per_token_quant_int8(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, scale = flaggems_vllm.per_token_quant_int8(x)
    x.normal_()
    graph.replay()
    ref_q, ref_scale = vllm_per_token_quant_int8(x)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("case", ["rank_one", "empty_last_dim", "nonviewable"])
def test_per_token_quant_int8_invalid_input(case):
    if case == "rank_one":
        x = torch.empty((4,), device=flaggems_vllm.device)
        error = IndexError
    elif case == "empty_last_dim":
        x = torch.empty((2, 0), device=flaggems_vllm.device)
        error = ZeroDivisionError
    else:
        x = torch.empty((2, 3, 4), device=flaggems_vllm.device).transpose(0, 1)
        error = RuntimeError
    for op in (vllm_per_token_quant_int8, flaggems_vllm.per_token_quant_int8):
        with pytest.raises(error):
            op(x)

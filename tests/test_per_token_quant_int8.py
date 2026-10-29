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

from . import accuracy_utils as utils
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
        pytest.param((4097, 4097), id="row_tile_tail"),
        (0, 128),
    ]
)
COMMON_SHAPES = (
    [(1, 2048), (128, 4097)]
    if QUICK_MODE
    else [
        (1, 128),
        (1, 2048),
        (1, 8192),
        (16, 2048),
        (32, 4096),
        (64, 8192),
        (128, 1024),
        (256, 2048),
        (512, 8192),
        (1024, 4096),
        (4096, 2048),
        (4096, 4096),
        (4096, 8192),
        (128, 4097),
        (7, 5137),
    ]
)


def _check_quantization(x, quant_op=flaggems_vllm.per_token_quant_int8):
    ref_q, ref_scale = vllm_per_token_quant_int8(x)
    q, scale = quant_op(x)
    assert q.shape == x.shape and q.dtype == torch.int8
    assert scale.shape == x.shape[:-1] + (1,) and scale.dtype == torch.float32
    assert q.device == x.device and scale.device == x.device
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0, equal_nan=True)
    return q, scale


def _make_input(shape, dtype, layout):
    torch.manual_seed(0)
    m, n = shape
    if layout == "strided":
        x = torch.randn((m * 2, n * 2), dtype=dtype, device=flaggems_vllm.device)
        x = x[::2, ::2]
    elif layout == "transposed":
        x = torch.randn((n, m), dtype=dtype, device=flaggems_vllm.device).t()
    else:
        x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    return x


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("seed", utils.FP8_QUANT_SHAPES["SEEDS"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", utils.FP8_QUANT_SHAPES["D"])
@pytest.mark.parametrize("num_tokens", utils.FP8_QUANT_SHAPES["NUM_TOKENS"])
def test_per_token_quant_int8(num_tokens, d, dtype, seed):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype, device=flaggems_vllm.device)
    _check_quantization(x)


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", COMMON_SHAPES)
def test_per_token_quant_int8_common_shapes(shape, dtype):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    _check_quantization(x)


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
def test_per_token_quant_int8_layouts(shape, dtype, layout):
    _check_quantization(_make_input(shape, dtype, layout))


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
    # Signaling NaNs can change the reference reduction with its warp layout.
    # Keep all finite bit patterns and compare quiet NaNs on every backend.
    exponent, mantissa, quiet = (
        (0x7C00, 0x03FF, 0x0200) if dtype == torch.float16 else (0x7F80, 0x007F, 0x0040)
    )
    is_nan = ((bits & exponent) == exponent) & ((bits & mantissa) != 0)
    bits = torch.where(is_nan, bits | quiet, bits)
    bits = bits.to(torch.int16)
    order = torch.randperm(bits.numel(), device=bits.device)
    x = bits[order].view(dtype).reshape(-1, width)
    _check_quantization(x)


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "layout",
    [
        "storage_offset",
        "broadcast_rows",
        "broadcast_columns",
        "leading_stride",
        "repeat",
    ],
)
def test_per_token_quant_int8_views(dtype, layout):
    torch.manual_seed(42)
    device = flaggems_vllm.device
    if layout == "storage_offset":
        x = torch.randn((8, 1026), dtype=dtype, device=device)[1:, 1:]
    elif layout == "broadcast_rows":
        x = torch.randn((1, 1025), dtype=dtype, device=device).expand(7, -1)
    elif layout == "broadcast_columns":
        x = torch.randn((7, 1), dtype=dtype, device=device).expand(-1, 1025)
    elif layout == "leading_stride":
        x = torch.randn((2, 3, 514), dtype=dtype, device=device)[..., ::2]
    else:
        x = torch.randn((7, 1025), dtype=dtype, device=device)
        _check_quantization(x)
        x.normal_()
    _check_quantization(x)


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


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
@pytest.mark.parametrize(
    "reduce_group,row_tiles,contiguous_rows,quant_parts",
    [
        (4, 1, False, 1),
        (8, 1, False, 1),
        (16, 1, False, 1),
        (0, 2, False, 1),
        (0, 2, True, 1),
        (8, 1, False, 2),
        (8, 1, False, 4),
        (8, 1, False, 8),
        (8, 1, False, 16),
    ],
)
def test_per_token_quant_int8_reduction_and_row_tails(
    dtype, layout, reduce_group, row_tiles, contiguous_rows, quant_parts
):
    from flaggems_vllm.ops.per_token_quant_int8 import _per_token_quant_int8_kernel

    torch.manual_seed(0)
    m, n = 9, 257
    if layout == "strided":
        x = torch.randn((m * 2, n * 2), device=flaggems_vllm.device, dtype=dtype)
        x = x[::2, ::2]
    elif layout == "transposed":
        x = torch.randn((n, m), device=flaggems_vllm.device, dtype=dtype).t()
    else:
        x = torch.randn((m, n), device=flaggems_vllm.device, dtype=dtype)
    x[0] = float("nan")
    x[1] = float("nan")
    x[1, -1] = 1.0
    x[2] = 0.0
    x[3] = 1e-12
    x[4, 0] = float("inf")
    x[4, 1] = -float("inf")
    x[5] = 0.0
    x[5, :8] = torch.tensor(
        [-127, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127],
        device=x.device,
        dtype=dtype,
    )
    x[6, ::16] = float("nan")

    ref_q, ref_scale = vllm_per_token_quant_int8(x)
    q = torch.empty((m, n), device=x.device, dtype=torch.int8)
    scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    # Exercise each new schedule even when autotune prefers the original path.
    kernel = _per_token_quant_int8_kernel.fn.fn
    kernel[((m + row_tiles - 1) // row_tiles, quant_parts)](
        x,
        q,
        scale,
        m,
        n,
        x.stride(0),
        x.stride(1),
        512,
        1,
        16,
        row_tiles,
        contiguous_rows,
        reduce_group,
        quant_parts,
        num_warps=4,
        num_stages=1,
    )
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0, equal_nan=True)


@pytest.mark.per_token_quant_int8
@pytest.mark.parametrize("quant_parts", [2, 4, 8, 16])
@pytest.mark.parametrize("width", [256, 2048])
@pytest.mark.parametrize("shuffle", [False, True])
def test_per_token_quant_int8_partition_bit_patterns(quant_parts, width, shuffle):
    from flaggems_vllm.ops.per_token_quant_int8 import _per_token_quant_int8_kernel

    bits = torch.arange(65536, device=flaggems_vllm.device, dtype=torch.int32)
    is_nan = ((bits & 0x7F80) == 0x7F80) & ((bits & 0x007F) != 0)
    bits = torch.where(is_nan, bits | 0x0040, bits)
    if shuffle:
        generator = torch.Generator(device=flaggems_vllm.device).manual_seed(0)
        bits = bits[torch.randperm(65536, device=bits.device, generator=generator)]
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(-1, width)
    m, n = x.shape
    ref_q, ref_scale = vllm_per_token_quant_int8(x)
    q = torch.empty_like(ref_q)
    scale = torch.empty_like(ref_scale)
    _per_token_quant_int8_kernel.fn.fn[(m, quant_parts)](
        x,
        q,
        scale,
        m,
        n,
        n,
        1,
        n,
        1,
        8,
        1,
        False,
        8,
        quant_parts,
        num_warps=8,
        num_stages=1,
    )
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0, equal_nan=True)

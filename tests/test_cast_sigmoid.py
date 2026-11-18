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

import flaggems_vllm

from . import accuracy_utils as utils


def _reference(x, out_dtype):
    return torch.sigmoid(utils.to_reference(x, True)).to(out_dtype)


# Small / medium / large tiers. The large tier is dropped in --quick runs so
# the gate stays fast while the full run still covers bandwidth-bound shapes.
SMALL_SHAPES = ((), (1,), (7,), (33, 5), (1, 256))
MEDIUM_SHAPES = ((17, 129), (128, 256), (1023,), (4096, 256))
LARGE_SHAPES = (
    ()
    if utils.QUICK_MODE
    else ((1048576,), (4096, 4096), (257, 1024), (2, 4, 8, 64, 32))
)
SHAPES = SMALL_SHAPES + MEDIUM_SHAPES + LARGE_SHAPES

VLLM_FORM_SHAPES = ((1, 256), (128, 256), (4096, 256))
if not utils.QUICK_MODE:
    VLLM_FORM_SHAPES = VLLM_FORM_SHAPES + ((1048576,),)


@pytest.mark.cast_sigmoid
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cast_sigmoid(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    snapshot = x.clone()

    res = flaggems_vllm.cast_sigmoid(x)

    assert res.shape == x.shape
    assert res.dtype == dtype
    assert res.device == x.device
    utils.gems_assert_close(res, _reference(x, dtype), dtype)
    utils.gems_assert_equal(x, snapshot)


@pytest.mark.cast_sigmoid
@pytest.mark.parametrize("shape", VLLM_FORM_SHAPES)
@pytest.mark.parametrize("in_dtype", (torch.float16, torch.bfloat16))
def test_cast_sigmoid_vllm_fp32_form(shape, in_dtype):
    """vLLM's _cast_sigmoid: FP16/BF16 input, FP32 output."""
    x = torch.randn(shape, dtype=in_dtype, device=flaggems_vllm.device)

    res = flaggems_vllm.cast_sigmoid(x, out_dtype=torch.float32)

    assert res.dtype == torch.float32
    utils.gems_assert_close(res, x.float().sigmoid(), torch.float32)


@pytest.mark.cast_sigmoid
@pytest.mark.parametrize("in_dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("out_dtype", utils.FLOAT_DTYPES)
def test_cast_sigmoid_out_dtype(in_dtype, out_dtype):
    x = torch.randn(37, 129, dtype=in_dtype, device=flaggems_vllm.device)

    res = flaggems_vllm.cast_sigmoid(x, out_dtype=out_dtype)

    assert res.dtype == out_dtype
    utils.gems_assert_close(res, _reference(x, out_dtype), out_dtype)


@pytest.mark.cast_sigmoid
@pytest.mark.parametrize("compute_dtype", (torch.float32, None))
def test_cast_sigmoid_compute_dtype(compute_dtype):
    x = torch.randn(16, 1023, dtype=torch.float16, device=flaggems_vllm.device)

    res = flaggems_vllm.cast_sigmoid(x, compute_dtype=compute_dtype)

    assert res.dtype == x.dtype
    utils.gems_assert_close(res, _reference(x, x.dtype), x.dtype)


@pytest.mark.cast_sigmoid
@pytest.mark.parametrize("compute_dtype", (torch.float16, torch.bfloat16))
def test_cast_sigmoid_rejects_unsupported_compute_dtype(compute_dtype):
    x = torch.randn(16, dtype=torch.float32, device=flaggems_vllm.device)
    with pytest.raises(NotImplementedError, match="computes in FP32 only"):
        flaggems_vllm.cast_sigmoid(x, compute_dtype=compute_dtype)


@pytest.mark.cast_sigmoid
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cast_sigmoid_saturates(dtype):
    values = [-1e4, -88.0, -20.0, 0.0, 20.0, 88.0, 1e4, float("inf"), float("-inf")]
    x = torch.tensor(values, dtype=dtype, device=flaggems_vllm.device)

    res = flaggems_vllm.cast_sigmoid(x)

    utils.gems_assert_close(res, _reference(x, dtype), dtype)


@pytest.mark.cast_sigmoid
def test_cast_sigmoid_empty():
    x = torch.empty(0, 7, dtype=torch.float16, device=flaggems_vllm.device)
    res = flaggems_vllm.cast_sigmoid(x)
    assert res.shape == x.shape
    assert res.numel() == 0


@pytest.mark.cast_sigmoid
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cast_sigmoid_noncontiguous(dtype):
    storage = torch.randn(4, 129, dtype=dtype, device=flaggems_vllm.device)
    x = storage.t()

    res = flaggems_vllm.cast_sigmoid(x)

    assert res.shape == x.shape
    utils.gems_assert_close(res, _reference(x, dtype), dtype)


@pytest.mark.cast_sigmoid
def test_cast_sigmoid_rejects_unsupported_dtype():
    x = torch.randint(0, 8, (16,), device=flaggems_vllm.device, dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="FP16, BF16 and FP32"):
        flaggems_vllm.cast_sigmoid(x)


@pytest.mark.cast_sigmoid
def test_cast_sigmoid_rejects_unsupported_out_dtype():
    x = torch.randn(16, device=flaggems_vllm.device, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="output must be"):
        flaggems_vllm.cast_sigmoid(x, out_dtype=torch.int32)


@pytest.mark.cast_sigmoid
def test_cast_sigmoid_rejects_cpu_tensor():
    x = torch.randn(16, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="accelerator device"):
        flaggems_vllm.cast_sigmoid(x)

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

import math

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from . import conftest as cfg

DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES
CASES = [
    ((4, 64), (64,)),
    ((3, 513), (513,)),
    ((2, 5125), (5125,)),
    ((2, 8192), (8192,)),
]
EPS = 1e-5


def _torch_rms_norm(x, normalized_shape, weight, eps):
    reduction_dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    upcast_x = x.to(torch.float32)
    variance = upcast_x.square().mean(dim=reduction_dims, keepdim=True)
    normalized = (upcast_x * torch.rsqrt(variance + eps)).to(x.dtype)
    return (normalized * weight).to(x.dtype)


def _reference_tensor(tensor):
    return utils.to_reference(tensor.detach().clone()).requires_grad_()


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape,normalized_shape", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_forward_backward(shape, normalized_shape, dtype):
    utils.init_seed(0)
    x = (
        torch.empty(shape, dtype=dtype, device=flaggems_vllm.device)
        .uniform_(-0.2, 0.2)
        .requires_grad_()
    )
    weight = (
        torch.empty(normalized_shape, dtype=dtype, device=flaggems_vllm.device)
        .uniform_(-0.2, 0.2)
        .requires_grad_()
    )
    ref_x = _reference_tensor(x)
    ref_weight = _reference_tensor(weight)

    out = flaggems_vllm.rms_norm(x, list(normalized_shape), weight, EPS)
    ref_out = _torch_rms_norm(ref_x, normalized_shape, ref_weight, EPS)

    out_grad = torch.empty_like(out).uniform_(-0.02, 0.02)
    ref_out_grad = utils.to_reference(out_grad.detach().clone())
    x_grad, weight_grad = torch.autograd.grad(out, (x, weight), out_grad)
    ref_x_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_x, ref_weight), ref_out_grad
    )

    row_size = math.prod(normalized_shape)
    row_count = x.numel() // row_size
    utils.gems_assert_close(out, ref_out, dtype)
    utils.gems_assert_close(x_grad, ref_x_grad, dtype)
    utils.gems_assert_close(weight_grad, ref_weight_grad, dtype, reduce_dim=row_count)


@pytest.mark.rms_norm
@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_multidimensional_normalized_shape(dtype):
    utils.init_seed(1)
    shape = (2, 3, 5)
    normalized_shape = (3, 5)
    x = (
        torch.empty(shape, dtype=dtype, device=flaggems_vllm.device)
        .uniform_(-0.2, 0.2)
        .requires_grad_()
    )
    weight = (
        torch.empty(normalized_shape, dtype=dtype, device=flaggems_vllm.device)
        .uniform_(-0.2, 0.2)
        .requires_grad_()
    )
    ref_x = _reference_tensor(x)
    ref_weight = _reference_tensor(weight)

    out = flaggems_vllm.rms_norm(x, normalized_shape, weight, EPS)
    ref_out = _torch_rms_norm(ref_x, normalized_shape, ref_weight, EPS)
    out_grad = torch.empty_like(out).uniform_(-0.02, 0.02)
    ref_out_grad = utils.to_reference(out_grad.detach().clone())
    x_grad, weight_grad = torch.autograd.grad(out, (x, weight), out_grad)
    ref_x_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_x, ref_weight), ref_out_grad
    )

    utils.gems_assert_close(out, ref_out, dtype)
    utils.gems_assert_close(x_grad, ref_x_grad, dtype)
    utils.gems_assert_close(weight_grad, ref_weight_grad, dtype, reduce_dim=2)


@pytest.mark.rms_norm
@pytest.mark.parametrize("row_size", [513, 5125])
def test_rms_norm_sum_backward(row_size):
    x = torch.randn(
        (3, row_size),
        dtype=torch.float32,
        device=flaggems_vllm.device,
        requires_grad=True,
    )
    weight = torch.randn(
        (row_size,),
        dtype=torch.float32,
        device=flaggems_vllm.device,
        requires_grad=True,
    )
    ref_x = _reference_tensor(x)
    ref_weight = _reference_tensor(weight)

    flaggems_vllm.rms_norm(x, (row_size,), weight, EPS).sum().backward()
    _torch_rms_norm(ref_x, (row_size,), ref_weight, EPS).sum().backward()

    utils.gems_assert_close(x.grad, ref_x.grad, torch.float32)
    utils.gems_assert_close(weight.grad, ref_weight.grad, torch.float32, reduce_dim=3)


@pytest.mark.rms_norm
def test_rms_norm_aten_registration():
    x = torch.randn((2, 64), dtype=torch.float32, device=flaggems_vllm.device)
    weight = torch.randn((64,), dtype=torch.float32, device=flaggems_vllm.device)
    reference = _torch_rms_norm(x, (64,), weight, EPS)

    with flaggems_vllm.use_gems(include=["rms_norm"]):
        result = torch.nn.functional.rms_norm(x, (64,), weight, EPS)

    utils.gems_assert_close(result, reference, torch.float32)


@pytest.mark.rms_norm
def test_rms_norm_validates_inputs():
    device = flaggems_vllm.device
    x = torch.randn((2, 4), dtype=torch.float32, device=device)
    weight = torch.randn((4,), dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="must end with normalized_shape"):
        flaggems_vllm.rms_norm(x, (5,), torch.randn(5, device=device), EPS)

    with pytest.raises(ValueError, match="weight shape"):
        flaggems_vllm.rms_norm(x, (4,), torch.randn(5, device=device), EPS)

    with pytest.raises(NotImplementedError, match="requires a weight"):
        flaggems_vllm.rms_norm(x, (4,), None, EPS)

    with flaggems_vllm.use_gems(include=["rms_norm"]):
        with pytest.raises(NotImplementedError, match="requires a weight"):
            torch.nn.functional.rms_norm(x, (4,), None, EPS)

    with pytest.raises(TypeError, match="same dtype"):
        flaggems_vllm.rms_norm(x, (4,), weight.to(torch.float16), EPS)

    with pytest.raises(TypeError, match="only supports"):
        flaggems_vllm.rms_norm(x.to(torch.float64), (4,), weight.to(torch.float64), EPS)

    noncontiguous_x = torch.randn((4, 2), device=device).T
    assert not noncontiguous_x.is_contiguous()
    with pytest.raises(NotImplementedError, match="contiguous"):
        flaggems_vllm.rms_norm(noncontiguous_x, (4,), weight, EPS)

    noncontiguous_weight = torch.randn((8,), device=device)[::2]
    assert not noncontiguous_weight.is_contiguous()
    with pytest.raises(NotImplementedError, match="contiguous"):
        flaggems_vllm.rms_norm(x, (4,), noncontiguous_weight, EPS)

    with pytest.raises(NotImplementedError, match="empty"):
        flaggems_vllm.rms_norm(torch.empty((0, 4), device=device), (4,), weight, EPS)

    with pytest.raises(NotImplementedError, match="explicit epsilon"):
        flaggems_vllm.rms_norm(x, (4,), weight, None)

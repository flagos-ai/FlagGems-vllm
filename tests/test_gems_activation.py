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
import flaggems_vllm.modules as gems_modules
from flaggems_vllm.modules.activation import GemsSiluAndMul, gems_silu_and_mul

from . import accuracy_utils as utils
from . import conftest as cfg

SHAPES = [(4, 64)] if cfg.QUICK_MODE else [(4, 64), (7, 513), (2, 8, 4096)]
DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES


@pytest.mark.gems_activation
def test_modules_package_exports():
    expected = [
        "GemsDeepseekYarnRoPE",
        "GemsRMSNorm",
        "GemsRope",
        "GemsSiluAndMul",
    ]
    assert gems_modules.__all__ == expected
    for name in expected:
        assert getattr(flaggems_vllm, name) is getattr(gems_modules, name)


@pytest.mark.gems_activation
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gems_silu_and_mul_module_forward_backward(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device, requires_grad=True)
    y = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device, requires_grad=True)
    ref_x = utils.to_reference(x.detach(), True).requires_grad_()
    ref_y = utils.to_reference(y.detach(), True).requires_grad_()

    ref_out = torch.nn.functional.silu(ref_x) * ref_y
    out = GemsSiluAndMul()(x, y)

    out_grad = torch.randn_like(out)
    ref_out_grad = utils.to_reference(out_grad, True)
    ref_x_grad, ref_y_grad = torch.autograd.grad(ref_out, (ref_x, ref_y), ref_out_grad)
    x_grad, y_grad = torch.autograd.grad(out, (x, y), out_grad)

    utils.gems_assert_close(out, ref_out, dtype)
    utils.gems_assert_close(x_grad, ref_x_grad, dtype)
    utils.gems_assert_close(y_grad, ref_y_grad, dtype)


@pytest.mark.gems_activation
@pytest.mark.parametrize("dtype", DTYPES)
def test_gems_silu_and_mul_function_forward_backward(dtype):
    # Give the functional entry point independent non-power-of-two coverage.
    shape = (3, 17)
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device, requires_grad=True)
    y = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device, requires_grad=True)
    ref_x = utils.to_reference(x.detach(), True).requires_grad_()
    ref_y = utils.to_reference(y.detach(), True).requires_grad_()

    out = gems_silu_and_mul(x, y)
    ref_out = torch.nn.functional.silu(ref_x) * ref_y
    out_grad = torch.randn_like(out)
    ref_out_grad = utils.to_reference(out_grad, True)
    ref_x_grad, ref_y_grad = torch.autograd.grad(ref_out, (ref_x, ref_y), ref_out_grad)
    x_grad, y_grad = torch.autograd.grad(out, (x, y), out_grad)

    utils.gems_assert_close(out, ref_out, dtype)
    utils.gems_assert_close(x_grad, ref_x_grad, dtype)
    utils.gems_assert_close(y_grad, ref_y_grad, dtype)

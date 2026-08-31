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
from flaggems_vllm.modules.normalization import GemsRMSNorm, gems_rms_forward

from . import accuracy_utils as utils
from . import conftest as cfg

RMS_SHAPES = [(2, 64)] if cfg.QUICK_MODE else [(4, 64), (7, 1024), (2, 8192)]
RESIDUAL_SHAPES = [(2, 64)] if cfg.QUICK_MODE else [(4, 64), (3, 4096), (2, 8192)]
DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES
EPS = 1e-6


def _torch_rms_norm(x, weight, eps):
    upcast_x = x.to(torch.float32)
    variance = upcast_x.square().mean(dim=-1, keepdim=True)
    normalized = (upcast_x * torch.rsqrt(variance + eps)).to(x.dtype)
    return (normalized * weight).to(x.dtype)


def _torch_fused_add_rms_norm(x, residual, weight, eps):
    added = x.to(torch.float32) + residual.to(torch.float32)
    variance = added.square().mean(dim=-1, keepdim=True)
    normalized = (added * torch.rsqrt(variance + eps)).to(x.dtype)
    return (normalized * weight).to(x.dtype), added.to(x.dtype)


@pytest.mark.gems_normalization
@pytest.mark.parametrize("entrypoint", ["function", "module"])
@pytest.mark.parametrize("shape", RMS_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gems_rms_norm_without_residual_forward_backward(entrypoint, shape, dtype):
    x = (
        torch.empty(shape, dtype=dtype, device=flaggems_vllm.device)
        .uniform_(-0.2, 0.2)
        .requires_grad_()
    )
    weight_data = torch.empty(
        shape[-1], dtype=dtype, device=flaggems_vllm.device
    ).uniform_(-0.2, 0.2)
    ref_x = utils.to_reference(x.detach(), True).requires_grad_()
    ref_weight = utils.to_reference(weight_data, True).requires_grad_()
    ref_out = _torch_rms_norm(ref_x, ref_weight, EPS)
    x_before = x.detach().clone()

    if entrypoint == "function":
        weight = weight_data.detach().clone().requires_grad_()
        out = gems_rms_forward(x, None, weight, EPS)
    else:
        module = GemsRMSNorm(shape[-1], eps=EPS, device=x.device, dtype=dtype)
        with torch.no_grad():
            module.weight.copy_(weight_data)
        weight = module.weight
        out = module(x)

    out_grad = torch.empty_like(out).uniform_(-0.02, 0.02)
    ref_out_grad = utils.to_reference(out_grad, True)
    ref_x_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_x, ref_weight), ref_out_grad
    )
    x_grad, weight_grad = torch.autograd.grad(out, (x, weight), out_grad)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=shape[-1])
    utils.gems_assert_close(x_grad, ref_x_grad, dtype, reduce_dim=shape[-1])
    utils.gems_assert_close(weight_grad, ref_weight_grad, dtype, reduce_dim=shape[0])
    utils.gems_assert_equal(x, x_before)


@pytest.mark.gems_normalization
@pytest.mark.parametrize("entrypoint", ["function", "module"])
@pytest.mark.parametrize("shape", RESIDUAL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gems_rms_norm_with_residual_is_inplace(entrypoint, shape, dtype):
    x = torch.empty(shape, dtype=dtype, device=flaggems_vllm.device).uniform_(-0.2, 0.2)
    residual = torch.empty_like(x).uniform_(-0.2, 0.2)
    weight = torch.empty(shape[-1], dtype=dtype, device=flaggems_vllm.device).uniform_(
        -0.2, 0.2
    )
    ref_x = utils.to_reference(x, True)
    ref_residual = utils.to_reference(residual, True)
    ref_weight = utils.to_reference(weight, True)
    ref_out, ref_new_residual = _torch_fused_add_rms_norm(
        ref_x, ref_residual, ref_weight, EPS
    )

    if entrypoint == "function":
        out, new_residual = gems_rms_forward(x, residual, weight, EPS)
    else:
        module = GemsRMSNorm(shape[-1], eps=EPS, device=x.device, dtype=dtype)
        with torch.no_grad():
            module.weight.copy_(weight)
            out, new_residual = module(x, residual)

    assert out is x
    assert new_residual is residual
    assert out.data_ptr() == x.data_ptr()
    assert new_residual.data_ptr() == residual.data_ptr()
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=shape[-1])
    utils.gems_assert_close(new_residual, ref_new_residual, dtype)


@pytest.mark.gems_normalization
def test_gems_rms_norm_rejects_disabled_affine():
    module = GemsRMSNorm(
        64,
        eps=EPS,
        elementwise_affine=False,
        device=flaggems_vllm.device,
        dtype=torch.float32,
    )
    x = torch.randn((2, 64), device=flaggems_vllm.device)

    assert module.weight is None
    with pytest.raises(NotImplementedError, match="elementwise_affine"):
        module(x)

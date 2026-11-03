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

SILU_AND_MUL_WITH_CLAMP_LIMITS = [3.0, 7.0]


@pytest.mark.silu_and_mul_with_clamp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("limit", SILU_AND_MUL_WITH_CLAMP_LIMITS)
def test_silu_and_mul_with_clamp(shape, dtype, limit):
    inp1 = torch.randn(
        shape, dtype=dtype, device=flaggems_vllm.device, requires_grad=True
    )
    inp2 = torch.randn(
        shape, dtype=dtype, device=flaggems_vllm.device, requires_grad=True
    )
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_gate = torch.clamp(ref_inp1, max=limit)
    ref_up = torch.clamp(ref_inp2, min=-limit, max=limit)
    ref_out = torch.mul(torch.nn.functional.silu(ref_gate), ref_up)
    with flaggems_vllm.use_gems():
        res_out = flaggems_vllm.silu_and_mul_with_clamp(inp1, inp2, limit)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)

    ref_inp1_grad, ref_inp2_grad = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )

    res_inp1_grad, res_inp2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_inp1_grad, ref_inp1_grad, dtype)
    utils.gems_assert_close(res_inp2_grad, ref_inp2_grad, dtype)


@pytest.mark.silu_and_mul_with_clamp_out
@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "mthreads",
    reason="Issue #636: silu_and_mul_with_clamp_out accuracy failure on mthreads",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("limit", SILU_AND_MUL_WITH_CLAMP_LIMITS)
def test_silu_and_mul_with_clamp_out(shape, dtype, limit):
    inp1 = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_gate = torch.clamp(ref_inp1, max=limit)
    ref_up = torch.clamp(ref_inp2, min=-limit, max=limit)
    ref_out = torch.mul(torch.nn.functional.silu(ref_gate), ref_up)

    out = torch.empty_like(inp1)
    with flaggems_vllm.use_gems():
        ret = flaggems_vllm.silu_and_mul_with_clamp_out(inp1, inp2, out, limit)

    assert ret is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.silu_and_mul_with_clamp
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "layout", ["transpose", "broadcast2d", "broadcast_nd", "empty"]
)
def test_clamp_layout_autograd(dtype, layout):
    device = flaggems_vllm.device
    if layout == "transpose":
        x = torch.randn(7, 5, device=device, dtype=dtype).t()
        y = torch.randn_like(x)
    elif layout == "broadcast2d":
        x = torch.randn(7, 5, device=device, dtype=dtype).t()
        y = torch.randn(1, 7, device=device, dtype=dtype)
    elif layout == "broadcast_nd":
        x = torch.randn(2, 3, 5, device=device, dtype=dtype).transpose(1, 2)
        y = torch.randn(1, 5, 1, device=device, dtype=dtype)
    else:
        x = torch.empty(0, 3, device=device, dtype=dtype)
        y = torch.randn(1, 3, device=device, dtype=dtype)
    x.requires_grad_()
    y.requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    reference_y = y.detach().clone().requires_grad_()
    actual = flaggems_vllm.silu_and_mul_with_clamp(x, y, 2.0)
    expected = torch.nn.functional.silu(reference_x.float().clamp(max=2.0))
    expected = expected * reference_y.float().clamp(-2.0, 2.0)
    tolerance = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[dtype]
    torch.testing.assert_close(actual.float(), expected, rtol=tolerance, atol=tolerance)
    # Noncontiguous incoming gradients exercise the backward stride metadata.
    grad = torch.randn(tuple(reversed(actual.shape)), device=device, dtype=dtype)
    grad = grad.permute(tuple(reversed(range(actual.ndim))))
    actual.backward(grad)
    expected.backward(grad.float())
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(y.grad, reference_y.grad, rtol=tolerance, atol=tolerance)

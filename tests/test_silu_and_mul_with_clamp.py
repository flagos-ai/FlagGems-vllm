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
@pytest.mark.silu_and_mul_with_clamp_out
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["linear", "strided-nd", "packed-2d"])
def test_clamped_silu_signed_zero_and_special_values(dtype, layout):
    edges = torch.tensor(
        [
            -float("inf"),
            -100.0,
            -10.0,
            -0.0,
            0.0,
            10.0,
            100.0,
            float("inf"),
            float("nan"),
        ],
        device=flaggems_vllm.device,
        dtype=dtype,
    )
    gate, up = torch.meshgrid(edges, edges, indexing="ij")
    if layout == "packed-2d":
        packed = torch.cat((gate, up), dim=-1)
        gate, up = packed[:, :9], packed[:, 9:]
    else:
        gate, up = gate.reshape(-1), up.reshape(-1)
        if layout == "strided-nd":
            packed = torch.stack((gate, up), dim=-1)
            gate, up = packed[:, :1], packed[:, 1:]
    ref_gate, ref_up = gate.cpu(), up.cpu()
    expected = torch.nn.functional.silu(ref_gate.clamp(max=10.0)) * ref_up.clamp(
        -10.0, 10.0
    )
    allocated = flaggems_vllm.silu_and_mul_with_clamp(gate, up, 10.0)
    out = torch.full_like(allocated, 73.0)
    ret = flaggems_vllm.silu_and_mul_with_clamp_out(gate, up, out, 10.0)
    assert ret is out
    zero = torch.isfinite(expected) & (expected == 0)
    for actual in (allocated, out):
        actual = actual.cpu()
        utils.gems_assert_close(actual, expected, dtype, equal_nan=True)
        assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))


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

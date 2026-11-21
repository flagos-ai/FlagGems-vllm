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
    ret = flaggems_vllm.silu_and_mul_with_clamp_out(inp1, inp2, out, limit)

    assert ret is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.silu_and_mul_with_clamp_out
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX tuning buckets")
def test_strided_forward_reuses_size_bucket_with_distinct_masked_tails():
    from flaggems_vllm.ops.silu_and_mul_with_clamp import _metax_2d_size_key

    width = 1025
    rows = (1025, 1027)
    assert _metax_2d_size_key(31) == 31
    assert _metax_2d_size_key(rows[0] * width) == _metax_2d_size_key(rows[1] * width)
    for count in rows:
        packed = torch.randn(
            count, 2 * width, device=flaggems_vllm.device, dtype=torch.bfloat16
        )
        gate, up = packed.split(width, dim=-1)
        storage = torch.full(
            (count * width + 32,),
            float("nan"),
            dtype=packed.dtype,
            device=packed.device,
        )
        result = storage[:-32].view(count, width)
        flaggems_vllm.silu_and_mul_with_clamp_out(gate, up, result, 7.0)
        expected = torch.nn.functional.silu(
            gate.float().clamp(max=7)
        ) * up.float().clamp(-7, 7)
        torch.testing.assert_close(
            result, expected.bfloat16(), atol=0.015625, rtol=0.01
        )
        assert storage[-32:].isnan().all()

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

from .conftest import QUICK_MODE

device = flaggems_vllm.device
INT8_ABSMAX = 127.0
INT8_RANGE = 255.0
I32_MIN = -2147483648
I32_MAX = 2147483647

if QUICK_MODE:
    NUM_TOKENS = [7]
    HIDDEN_SIZES = [1024]
    DTYPES = [torch.bfloat16]
    SCALE = [0.1]
    AZP = [54]
else:
    NUM_TOKENS = [1, 7, 4096]
    HIDDEN_SIZES = [17, 1024, 1025, 1026, 5137, 8193]
    DTYPES = [torch.bfloat16, torch.float]
    SCALE = [0.1, 2.1]
    AZP = [-255, 54]


def _ref_dynamic_symmetric(x):
    x_f32 = x.float()
    absmax = x_f32.abs().amax(dim=-1, keepdim=True)
    scale = absmax / INT8_ABSMAX
    inv_s = torch.where(
        absmax == 0, torch.tensor(0.0, device=x.device), INT8_ABSMAX / absmax
    )
    q = (x_f32 * inv_s).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def _ref_dynamic_asymmetric(x):
    x_f32 = x.float()
    row_max = x_f32.amax(dim=-1, keepdim=True)
    row_min = x_f32.amin(dim=-1, keepdim=True)
    scale = (row_max - row_min) / INT8_RANGE
    azp = (-128.0 - row_min / scale).round().clamp(I32_MIN, I32_MAX).to(torch.int32)
    q = ((x_f32 / scale).round() + azp).clamp(-128, 127).to(torch.int8)
    return q, scale, azp


@pytest.mark.scaled_int8_quant
@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "mthreads",
    reason="Issue #636: scaled_int8_quant API is incompatible on mthreads",
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_dynamic_scaled_int8_quant(num_tokens, hidden_size, dtype):
    torch.manual_seed(0)
    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device) * 1000

    ref_out, ref_scale = _ref_dynamic_symmetric(x)
    ops_out, ops_scale, ops_azp = flaggems_vllm.scaled_int8_quant(x, symmetric=True)

    assert ops_azp is None
    torch.testing.assert_close(ops_scale, ref_scale)
    torch.testing.assert_close(ops_out, ref_out, atol=1, rtol=0.0)


@pytest.mark.scaled_int8_quant
@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "mthreads",
    reason="Issue #636: scaled_int8_quant API is incompatible on mthreads",
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_dynamic_scaled_int8_azp_quant(num_tokens, hidden_size, dtype):
    torch.manual_seed(0)
    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device) * 1000 - 300

    ref_out, ref_scale, ref_azp = _ref_dynamic_asymmetric(x)
    ops_out, ops_scale, ops_azp = flaggems_vllm.scaled_int8_quant(x, symmetric=False)

    torch.testing.assert_close(ops_scale, ref_scale)
    torch.testing.assert_close(ops_azp, ref_azp, atol=1, rtol=0.0)
    torch.testing.assert_close(ops_out, ref_out, atol=2, rtol=0.0)


@pytest.mark.scaled_int8_quant
@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "mthreads",
    reason="Issue #636: scaled_int8_quant API is incompatible on mthreads",
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_static_scaled_int8_quant(num_tokens, hidden_size, dtype, scale):
    torch.manual_seed(0)
    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device) * 1000
    scale_arg = torch.tensor([scale], dtype=torch.float32, device=device)

    ref_out = (x / scale_arg).round().clamp(-128, 127).to(torch.int8)
    ops_out, ops_scale, _ = flaggems_vllm.scaled_int8_quant(x, scale_arg)
    assert ops_scale is scale_arg

    torch.testing.assert_close(ref_out, ops_out, atol=1, rtol=0.0)


@pytest.mark.scaled_int8_quant
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", SCALE)
@pytest.mark.parametrize("azp", AZP)
@torch.inference_mode()
def test_static_scaled_int8_azp_quant(num_tokens, hidden_size, dtype, scale, azp):
    torch.manual_seed(0)
    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device) * 1000 - 300
    scale_arg = torch.tensor([scale], dtype=torch.float32, device=device)
    azp_arg = torch.tensor([azp], dtype=torch.int32, device=device)

    ref_out = ((x / scale).round() + azp).clamp(-128, 127).to(torch.int8)
    ops_out, ops_scale, ops_azp = flaggems_vllm.scaled_int8_quant(
        x, scale_arg, azp_arg, symmetric=False
    )
    assert ops_scale is scale_arg
    assert ops_azp is azp_arg

    torch.testing.assert_close(ref_out, ops_out, atol=1, rtol=0.0)


@pytest.mark.scaled_int8_quant
@pytest.mark.parametrize("is_max", [True, False])
@torch.inference_mode()
def test_static_scaled_int8_azp_quant_saturating_cast(is_max):
    from numpy import inf, nextafter

    i32_max = 2147483647
    i32_min = -2147483648
    val = float(i32_max if is_max else i32_min)

    x_vals = [[nextafter(val, inf), val + 1, val, val - 1, nextafter(val, -inf)]]
    x = torch.tensor(x_vals, dtype=torch.float32, device=device)

    scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    azp = torch.tensor([0], dtype=torch.int32, device=device)

    expected_val = 127 if is_max else -128
    expected = torch.full((1, 5), expected_val, dtype=torch.int8, device=device)

    out, _, _ = flaggems_vllm.scaled_int8_quant(x, scale, azp, symmetric=False)
    torch.testing.assert_close(expected, out, atol=0, rtol=0)


@pytest.mark.scaled_int8_quant
@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "mthreads",
    reason="Issue #636: scaled_int8_quant API is incompatible on mthreads",
)
@pytest.mark.parametrize("value", [0.0, 3.0, -7.5])
@torch.inference_mode()
def test_dynamic_scaled_int8_azp_quant_constant_row(value):
    # Constant rows make span == 0; scale becomes 0 and azp saturates. The
    # implementation must not produce NaN/Inf and must stay finite.
    x = torch.full((4, 1024), value, dtype=torch.bfloat16, device=device)
    ops_out, ops_scale, ops_azp = flaggems_vllm.scaled_int8_quant(x, symmetric=False)
    assert torch.isfinite(ops_scale).all()
    assert torch.isfinite(ops_azp.float()).all()
    assert torch.isfinite(ops_out.float()).all()

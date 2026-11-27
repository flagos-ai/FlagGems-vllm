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

import flaggems_vllm as gems

pytestmark = [
    pytest.mark.layer_norm_fwd_npu,
    pytest.mark.skipif(
        gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _inputs(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    return (x, torch.randn(shape[-1], dtype=dtype, device=device), None, 1e-6), dict(
        z=torch.randn_like(x), group_size=None, norm_before_gate=True, is_rms_norm=True
    )


def _reference():
    from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu

    return layer_norm_fwd_npu


@pytest.mark.parametrize("shape", [(16, 128), (16384, 128), (33, 128), (1, 64)])
def test_layer_norm_fwd_npu(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, gems.device)
    expected = _reference()(*args, **kwargs)
    actual = gems.layer_norm_fwd_npu(*args, **kwargs)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("before", [False, True])
def test_gated_layer_norm_preallocated(before):
    x = torch.randn((5, 64), device=gems.device, dtype=torch.bfloat16)
    weight = torch.randn(64, device=x.device, dtype=x.dtype)
    bias = torch.randn_like(weight)
    gate = torch.randn_like(x)
    out = torch.empty_like(x)
    native = _reference()
    expected = native(
        x,
        weight,
        bias,
        1e-6,
        z=gate,
        group_size=32,
        norm_before_gate=before,
        is_rms_norm=False,
    )
    result = gems.layer_norm_fwd_npu(
        x,
        weight,
        bias,
        1e-6,
        z=gate,
        out=out,
        group_size=32,
        norm_before_gate=before,
        is_rms_norm=False,
    )
    assert result[0] is out
    torch.testing.assert_close(result, expected)
    with pytest.raises(ValueError, match="group size"):
        gems.layer_norm_fwd_npu(x, weight, None, 1e-6, group_size=3)

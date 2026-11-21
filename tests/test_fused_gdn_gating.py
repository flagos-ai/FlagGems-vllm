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
    pytest.mark.fused_gdn_gating,
    pytest.mark.skipif(
        gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _inputs(shape, dtype, device):
    a = torch.randn(shape, device=device, dtype=dtype)
    return (
        torch.randn(shape[-1], device=device),
        a,
        torch.randn_like(a),
        torch.randn(shape[-1], device=device, dtype=dtype),
    ), {}


def _reference():
    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    return fused_gdn_gating_patch


@pytest.mark.parametrize("shape", [(1, 16), (1024, 16), (31, 16)])
def test_fused_gdn_gating(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, gems.device)
    expected = _reference()(*args, **kwargs)
    actual = gems.fused_gdn_gating(*args, **kwargs)
    torch.testing.assert_close(actual, expected)


def test_gating_rejects_mismatched_heads():
    args, kwargs = _inputs((1, 16), torch.bfloat16, gems.device)
    with pytest.raises(ValueError, match="per head"):
        gems.fused_gdn_gating(args[0][:8], *args[1:])


def test_gating_empty_batch():
    args, kwargs = _inputs((0, 16), torch.bfloat16, gems.device)
    decay, gate = gems.fused_gdn_gating(*args, **kwargs)
    assert decay.shape == gate.shape == (1, 0, 16)
    assert decay.dtype == torch.float32 and gate.dtype == torch.bfloat16

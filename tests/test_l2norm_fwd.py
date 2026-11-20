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
    pytest.mark.l2norm_fwd,
    pytest.mark.skipif(
        gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _inputs(shape, dtype, device):
    return (torch.randn(shape, device=device, dtype=dtype),), {}


def _reference():
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    return l2norm_fwd


@pytest.mark.parametrize("shape", [(8, 128), (1, 128), (31, 127), (1024, 128)])
def test_l2norm_fwd(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, gems.device)
    expected = _reference()(*args, **kwargs)
    actual = gems.l2norm_fwd(*args, **kwargs)
    torch.testing.assert_close(actual, expected)


def test_l2norm_empty_and_output_dtype():
    x = torch.empty((0, 128), device=gems.device, dtype=torch.bfloat16)
    y = gems.l2norm_fwd(x, output_dtype=torch.float32)
    assert y.shape == x.shape and y.dtype == torch.float32
    with pytest.raises(ValueError, match="output dtype"):
        gems.l2norm_fwd(x, output_dtype=torch.int64)

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
    pytest.mark.fla_causal_conv1d_fn,
    pytest.mark.skipif(
        gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _inputs(shape, dtype, device):
    t, d = shape
    x = torch.randn(t, d + 16, device=device, dtype=dtype)[:, :d]
    weight = torch.randn(4, d, device=device, dtype=dtype) * 0.1
    state = torch.randn(8, 4, d, device=device, dtype=dtype)[:, :3, :]
    indices = torch.tensor([2], dtype=torch.int32, device=device)
    cu = torch.tensor([0, t], dtype=torch.int32, device=device)

    has_initial = torch.tensor([True], device=device, dtype=torch.bool)
    return (x, weight, None, state, cu, indices, has_initial), dict(
        activation="silu", pad_slot_id=-1, null_block_id=0, head_num=32
    )


def _reference():
    from fla_npu.ops.ascendc import causal_conv1d_fn

    return causal_conv1d_fn


@pytest.mark.parametrize("shape", [(1, 4096), (37, 4096), (1024, 4096)])
def test_fla_causal_conv1d_fn(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, gems.device)
    ref_args = list(args)
    for i in [3]:
        ref_args[i] = torch.empty_strided(
            args[i].shape, args[i].stride(), dtype=args[i].dtype, device=args[i].device
        ).copy_(args[i])
    expected = _reference()(*ref_args, **kwargs)
    actual = gems.fla_causal_conv1d_fn(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
    for i in [3]:
        torch.testing.assert_close(args[i], ref_args[i])


def test_prefill_requests_head_major_output():
    args, kwargs = _inputs((1, 4096), torch.bfloat16, gems.device)
    with pytest.raises(ValueError, match="32 output heads"):
        gems.fla_causal_conv1d_fn(*args, **dict(kwargs, head_num=8))

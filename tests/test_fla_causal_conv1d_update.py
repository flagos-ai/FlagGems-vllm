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
    pytest.mark.fla_causal_conv1d_update,
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
    state = torch.randn(t + 2, 4, d, device=device, dtype=dtype)[:, :3, :]
    indices = torch.arange(1, t + 1, dtype=torch.int32, device=device)
    cu = torch.arange(t + 1, dtype=torch.int32, device=device)

    out = torch.empty_like(x)
    return (x, state, weight), dict(
        activation="silu",
        conv_state_indices=indices,
        query_start_loc=cu,
        null_block_id=0,
        out=out,
    )


def _reference():
    from fla_npu.ops.ascendc import causal_conv1d_update

    return causal_conv1d_update


@pytest.mark.parametrize("shape", [(1, 4096), (2, 4096), (8, 4096)])
def test_fla_causal_conv1d_update(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, gems.device)
    ref_args = list(args)
    for i in [1]:
        ref_args[i] = torch.empty_strided(
            args[i].shape, args[i].stride(), dtype=args[i].dtype, device=args[i].device
        ).copy_(args[i])
    expected = _reference()(*ref_args, **dict(kwargs, out=kwargs["out"].clone()))
    actual = gems.fla_causal_conv1d_update(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
    for i in [1]:
        torch.testing.assert_close(args[i], ref_args[i])


def test_update_returns_out():
    args, kwargs = _inputs((1, 4096), torch.bfloat16, gems.device)
    out = gems.fla_causal_conv1d_update(*args, **kwargs)
    assert out is kwargs["out"]
    with pytest.raises(ValueError, match="one token"):
        gems.fla_causal_conv1d_update(
            *args, **dict(kwargs, query_start_loc=kwargs["query_start_loc"][:1])
        )

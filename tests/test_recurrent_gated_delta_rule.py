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
    pytest.mark.recurrent_gated_delta_rule,
    pytest.mark.skipif(
        gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _inputs(shape, dtype, device):
    pool, hv = shape
    q = torch.randn((1, 8, 128), device=device, dtype=dtype)
    k = torch.nn.functional.normalize(
        torch.randn((1, 8, 128), device=device), dim=-1
    ).to(dtype)
    v = torch.randn((1, hv, 128), device=device, dtype=dtype)
    state = torch.randn((pool, hv + 2, 128, 128), device=device, dtype=torch.float32)[
        :, :hv
    ]
    beta = torch.rand((1, hv), device=device, dtype=dtype)
    g = -torch.rand((1, hv), device=device, dtype=torch.float32)
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    idx = torch.tensor([1], device=device, dtype=torch.int32)
    return (q, k, v, state), dict(
        beta=beta, g=g, actual_seq_lengths=cu, ssm_state_indices=idx, scale=128**-0.5
    )


def _reference():
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    return recurrent_gated_delta_rule


@pytest.mark.parametrize("shape", [(16, 16), (32, 16), (8, 8)])
def test_recurrent_gated_delta_rule(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, gems.device)
    ref_args = list(args)
    for i in [3]:
        ref_args[i] = torch.empty_strided(
            args[i].shape, args[i].stride(), dtype=args[i].dtype, device=args[i].device
        ).copy_(args[i])
    expected = _reference()(*ref_args, **kwargs)
    actual = gems.recurrent_gated_delta_rule(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
    for i in [3]:
        torch.testing.assert_close(args[i], ref_args[i])


def test_recurrent_rejects_incompatible_gate():
    args, kwargs = _inputs((8, 16), torch.bfloat16, gems.device)
    with pytest.raises(ValueError, match="gate per value head"):
        gems.recurrent_gated_delta_rule(*args, **dict(kwargs, g=kwargs["g"][:, :8]))


def test_recurrent_padding_does_not_write_state():
    args, kwargs = _inputs((8, 16), torch.bfloat16, gems.device)
    initial = args[3].clone()
    out = gems.recurrent_gated_delta_rule(
        *args,
        **dict(
            kwargs, actual_seq_lengths=torch.zeros_like(kwargs["actual_seq_lengths"])
        ),
    )
    torch.testing.assert_close(args[3], initial, rtol=0, atol=0)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)

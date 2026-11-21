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

from . import base

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


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def test_fused_gdn_gating():
    b = A5Benchmark(
        op_name="fused_gdn_gating",
        torch_op=_reference(),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
    )
    b.set_gems(gems.fused_gdn_gating)
    b.run()

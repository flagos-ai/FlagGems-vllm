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


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def test_layer_norm_fwd_npu():
    b = A5Benchmark(
        op_name="layer_norm_fwd_npu",
        torch_op=_reference(),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
    )
    b.set_gems(gems.layer_norm_fwd_npu)
    b.run()

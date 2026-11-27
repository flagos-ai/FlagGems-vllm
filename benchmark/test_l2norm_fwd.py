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


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def test_l2norm_fwd():
    b = A5Benchmark(
        op_name="l2norm_fwd",
        torch_op=_reference(),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
    )
    b.set_gems(gems.l2norm_fwd)
    b.run()

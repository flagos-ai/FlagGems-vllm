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

import importlib

import pytest
import torch

import flaggems_vllm as gems
from baseline import fused_q as workload
from tests.test_fused_q import assert_benchmark_case

pytestmark = pytest.mark.fused_q
_ASCEND_MODULE = "flaggems_vllm.runtime.backend._ascend.ops.fused_q"


def build_ascend_inputs(binding, dtype):
    _, reference = workload.build_inputs(binding, dtype, "cpu")
    values = {
        name: value.to(gems.device) if isinstance(value, torch.Tensor) else value
        for name, value in reference.items()
    }
    return reference, values


def assert_ascend_outputs(reference, values, outputs):
    assert isinstance(outputs, tuple) and len(outputs) == 3
    device = values["q_pe"].device
    assert all(tensor.device == device and tensor.is_contiguous() for tensor in outputs)
    fp8 = torch.float8_e4m3fn
    assert tuple(tensor.dtype for tensor in outputs) == (fp8, torch.float32, fp8)
    cpu = (
        outputs[0].view(torch.uint8).cpu().view(fp8),
        outputs[1].cpu(),
        outputs[2].view(torch.uint8).cpu().view(fp8),
    )
    assert all(torch.isfinite(tensor.float()).all() for tensor in cpu)
    assert_benchmark_case(reference, cpu)


@pytest.fixture(scope="module")
def ascend_operator():
    if gems.vendor_name != "ascend":
        pytest.skip("requires the Ascend backend")
    if not torch.npu.is_available():
        pytest.fail("Ascend backend selected, but no NPU is available")
    if not torch.npu.get_device_name().startswith("Ascend910B4"):
        pytest.skip("requires Ascend 910B4")
    module = importlib.import_module(_ASCEND_MODULE)
    assert gems.fused_q is module.fused_q
    return gems.fused_q


@pytest.mark.parametrize(
    "binding", workload.grid(), ids=lambda b: f"n{b['num_tokens']}_h{b['num_heads']}"
)
def test_fused_q_ascend(binding, ascend_operator, request):
    n = binding["num_tokens"]
    if request.config.getoption("quick", default=False) and n > 96:
        pytest.skip("--quick excludes large workloads")
    reference, values = build_ascend_inputs(binding, torch.bfloat16)
    outputs = ascend_operator(**values)
    assert_ascend_outputs(reference, values, outputs)
    if n == 1:
        # Rebind both an input pointer and a scalar while the first outputs stay live.
        changed = dict(
            reference,
            q_scale=torch.tensor([0.75], dtype=torch.float32),
            index_weights_softmax_scale=0.0625,
        )
        rebound = dict(
            values,
            q_scale=changed["q_scale"].to(values["q_pe"].device),
            index_weights_softmax_scale=changed["index_weights_softmax_scale"],
        )
        again = ascend_operator(**rebound)
        for first, second in zip(outputs, again):
            assert first.data_ptr() != second.data_ptr()
        assert_ascend_outputs(changed, rebound, again)
        assert_ascend_outputs(reference, values, outputs)


@pytest.mark.parametrize(
    "override",
    [{"has_indexer": False}, {"index_rope_interleave": True}, {"quantize_mqa": False}],
)
def test_fused_q_ascend_rejects_unsupported_mode(ascend_operator, override):
    _, values = build_ascend_inputs(workload.grid()[0], torch.bfloat16)
    with pytest.raises(NotImplementedError, match="default quantized indexer mode"):
        ascend_operator(**dict(values, **override))

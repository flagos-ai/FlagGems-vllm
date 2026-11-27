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

pytestmark = [
    pytest.mark.fused_q,
    pytest.mark.skipif(gems.vendor_name != "thead", reason="requires T-Head PPU"),
]


def require_ppu_operator():
    if not torch.cuda.is_available():
        pytest.fail("T-Head backend selected, but no PPU is available")
    module = importlib.import_module("flaggems_vllm.runtime.backend._thead.ops.fused_q")
    assert gems.fused_q is module.fused_q, "Public fused_q must use the T-Head backend"
    return gems.fused_q


def build_ppu_inputs(binding, dtype):
    _, reference = workload.build_inputs(binding, dtype, "cpu")
    values = {
        name: value.to(gems.device) if isinstance(value, torch.Tensor) else value
        for name, value in reference.items()
    }
    return reference, values


def assert_ppu_outputs(reference, values, outputs):
    fp8 = torch.float8_e4m3fn
    assert isinstance(outputs, tuple) and len(outputs) == 3
    assert tuple(out.dtype for out in outputs) == (fp8, torch.float32, fp8)
    device = values["q_pe"].device
    assert all(out.device == device and out.is_contiguous() for out in outputs)
    cpu_outputs = (
        outputs[0].view(torch.uint8).cpu().view(fp8),
        outputs[1].cpu(),
        outputs[2].view(torch.uint8).cpu().view(fp8),
    )
    assert all(torch.isfinite(out.float()).all() for out in cpu_outputs)
    assert_benchmark_case(reference, cpu_outputs)
    for name, value in reference.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(values[name].cpu(), value), f"Input {name} was modified"


@pytest.fixture(scope="module")
def ppu_operator():
    return require_ppu_operator()


@pytest.mark.parametrize(
    "binding", workload.grid(), ids=lambda b: f"n{b['num_tokens']}_h{b['num_heads']}"
)
def test_fused_q_ppu(binding, ppu_operator, request):
    if request.config.getoption("quick", default=False) and binding["num_tokens"] > 96:
        pytest.skip("--quick excludes large workloads")
    reference, values = build_ppu_inputs(binding, torch.bfloat16)
    assert_ppu_outputs(reference, values, ppu_operator(**values))


def test_fused_q_ppu_rebinds_scale(ppu_operator):
    binding = workload.binding_from_shape((96, 64, 512, 64, 32, 128))
    reference, values = build_ppu_inputs(binding, torch.bfloat16)
    first = ppu_operator(**values)

    # Exercise the division fallback in a shape selected for reciprocal scaling.
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
    second = ppu_operator(**rebound)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(first, second))
    assert_ppu_outputs(changed, rebound, second)
    assert_ppu_outputs(reference, values, first)

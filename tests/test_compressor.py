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

import flaggems_vllm

from .compressor_cases import (
    ALL_CASES,
    clone_inputs,
    compressor_torch,
    make_inputs,
    split_operator_inputs,
    valid_output_rows,
)
from .conftest import QUICK_MODE

try:
    import vllm_ascend.utils as vllm_ascend_utils

    HAS_VLLM_ASCEND = True
except ImportError:
    vllm_ascend_utils = None
    HAS_VLLM_ASCEND = False

# The case registry and comparison semantics are migrated from the source
# operator repository. Keep a small subset under --quick.
CASES = ALL_CASES[:4] if QUICK_MODE else ALL_CASES

# Triton kernel vs reference implementations: fp32 accumulation order differs,
# so compare with the same tolerance the source repository used against the
# official AscendC operator.
RTOL = 0.05
ATOL = 0.05


def _synchronize():
    if flaggems_vllm.vendor_name == "ascend":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


@pytest.fixture(scope="module", autouse=True)
def _register_ascendc_baseline():
    if HAS_VLLM_ASCEND:
        vllm_ascend_utils.enable_custom_op()


def _assert_outputs_close(triton_output, ref_output, inputs):
    valid_rows = valid_output_rows(inputs)
    torch.testing.assert_close(
        triton_output.reshape(-1, triton_output.shape[-1])[:valid_rows].float(),
        ref_output.reshape(-1, ref_output.shape[-1])[:valid_rows].float(),
        rtol=RTOL,
        atol=ATOL,
    )
    padding = triton_output.reshape(-1, triton_output.shape[-1])[valid_rows:]
    if padding.numel():
        torch.testing.assert_close(padding, torch.zeros_like(padding), rtol=0, atol=0)


@pytest.mark.compressor
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="compressor is only implemented for the ascend vendor",
)
@pytest.mark.parametrize("case_name", CASES)
def test_compressor_matches_torch_golden(case_name):
    inputs = make_inputs(case_name, device=flaggems_vllm.device, seed=42)
    tensor_inputs, config = split_operator_inputs(clone_inputs(inputs))
    golden_inputs = {
        key: value.cpu() if torch.is_tensor(value) else value
        for key, value in clone_inputs(inputs).items()
    }

    with torch.no_grad():
        triton_output = flaggems_vllm.compressor(**tensor_inputs, **config)
        golden_output = compressor_torch(**golden_inputs)
    _synchronize()

    _assert_outputs_close(triton_output.cpu(), golden_output, inputs)
    # The reference also updates state_cache in place with the same projections.
    torch.testing.assert_close(
        tensor_inputs["state_cache"].cpu(),
        golden_inputs["state_cache"],
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.compressor
@pytest.mark.skipif(
    not HAS_VLLM_ASCEND or flaggems_vllm.vendor_name != "ascend",
    reason="vllm-ascend custom operators are unavailable",
)
@pytest.mark.parametrize("case_name", CASES)
def test_compressor_matches_ascendc(case_name):
    if not vllm_ascend_utils.enable_custom_op():
        pytest.skip("vllm-ascend custom operators are unavailable")

    inputs = make_inputs(case_name, device=flaggems_vllm.device, seed=42)
    official_inputs = clone_inputs(inputs)
    tensor_inputs, config = split_operator_inputs(clone_inputs(inputs))

    with torch.no_grad():
        official_output = torch.ops._C_ascend.compressor(**official_inputs)
        triton_output = flaggems_vllm.compressor(**tensor_inputs, **config)
    _synchronize()

    _assert_outputs_close(triton_output.cpu(), official_output.cpu(), inputs)
    torch.testing.assert_close(
        tensor_inputs["state_cache"].float(),
        official_inputs["state_cache"].float(),
        rtol=RTOL,
        atol=ATOL,
    )

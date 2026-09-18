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
from tests.compressor_cases import PRODUCTION_CASES, make_inputs

from . import base

try:
    import vllm_ascend.utils as vllm_ascend_utils

    HAS_VLLM_ASCEND = True
except ImportError:
    vllm_ascend_utils = None
    HAS_VLLM_ASCEND = False


def _input_fn(case_name, dtype, device):
    inputs = make_inputs(case_name, device=device, seed=42)
    yield (inputs,)


class CompressorBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Each "shape" is a production case name; the concrete layout comes
        # from the shared case registry (see tests/compressor_cases.py).
        self.shapes = list(PRODUCTION_CASES)
        self.shape_desc = "case(B,S,T,H,D,R,C)"

    def set_more_shapes(self):
        return []


@pytest.mark.compressor
@pytest.mark.skipif(
    not HAS_VLLM_ASCEND or flaggems_vllm.vendor_name != "ascend",
    reason="compressor benchmark needs the ascend vendor and the vllm-ascend AscendC baseline",
)
def test_compressor():
    if not vllm_ascend_utils.enable_custom_op():
        pytest.skip("vllm-ascend custom operators are unavailable")
    baseline = torch.ops._C_ascend.compressor

    bench = CompressorBenchmark(
        op_name="compressor",
        input_fn=_input_fn,
        torch_op=baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.compressor)
    bench.run()

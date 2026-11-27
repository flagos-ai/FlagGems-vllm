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
from baseline.hc_collapse import SHAPES, build_inputs, vllm_ascend_replay

from . import base
from .conftest import Config
from .consts import BenchMode


class HCCollapseBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Measure the non-empty cases from the public vLLM shape matrix.
        self.shapes = [tuple(s) for s in SHAPES if s[0] > 0]
        self.shape_desc = "T, HC, H"


@pytest.mark.hc_collapse
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="Ascend implementation"
)
def test_hc_collapse():
    # Match the CUDA baseline shape key. The standalone runner measures both layouts.
    def inputs(shape, dtype, device):
        args, _ = build_inputs({"shape": shape, "strided": False}, dtype, device)
        yield args

    if Config.mode == BenchMode.CUDAGRAPH:
        raise ValueError("HC acceptance does not permit CUDA graph timing")
    bench = HCCollapseBenchmark(
        input_fn=inputs,
        op_name="hc_collapse",
        torch_op=vllm_ascend_replay,
        gems_op=flaggems_vllm.hc_collapse,
        dtypes=[torch.bfloat16],
        is_inplace=False,
    )
    bench.run()

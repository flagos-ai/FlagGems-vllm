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
import yaml

import flaggems_vllm
from flaggems_vllm import fused_kda_gate as gems_fused_kda_gate
from tests.test_fused_kda_gate import (
    captured_gate_cases,
    captured_gate_inputs,
    ref_fused_kda_gate,
)

from . import base, consts
from .conftest import Config


class FusedKdaGateBenchmark(base.Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]

    def set_shapes(self, shape_file_path=None):
        self.captured_cases = None
        # Keep the original branch's decode matrix, with the aligned safe_gate flag.
        self.shapes = [
            (T, H, D, safe_gate)
            for T in [1, 8, 64, 300]
            for H in [8, 96]
            for D in [64, 128]
            for safe_gate in [False, True]
        ]
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [
                (T, H, 128, safe_gate)
                for T in [1024, 4097, 16384]
                for H in [8, 96]
                for safe_gate in [False, True]
            ]
        if shape_file_path:
            with open(shape_file_path) as stream:
                payload = yaml.safe_load(stream)
            if payload.get("purpose") == "operator_shape_capture_only":
                self.captured_cases = captured_gate_cases(payload)
                self.shapes = self.captured_cases
                return
            entry = payload.get(self.op_name)
            if entry:
                self.shapes = [tuple(shape) for shape in entry["shapes"]]

    def get_input_iter(self, dtype):
        if self.captured_cases is not None:
            for case in self.captured_cases:
                if case["operator_g"]["dtype"] != str(dtype):
                    continue
                torch.manual_seed(42)
                args = captured_gate_inputs(case, self.device)
                ref_fused_kda_gate(*args)
                gems_fused_kda_gate(*args)
                flaggems_vllm.runtime.torch_device_fn.synchronize()
                yield args
            return
        for T, H, D, safe_gate in self.shapes:
            torch.manual_seed(42)
            g = torch.randn(T, H * D, device=self.device, dtype=dtype)
            A = torch.randn(H, device=self.device) * 0.5
            bias = torch.randn(H, D, device=self.device) * 0.1
            args = (g, A, D, bias, 1.0, 20.0, safe_gate, -5.0)
            # Compile/autotune both before the framework estimates iteration counts.
            # Every timed call still dispatches the complete op and allocates its output.
            ref_fused_kda_gate(*args)
            gems_fused_kda_gate(*args)
            flaggems_vllm.runtime.torch_device_fn.synchronize()
            yield args


@pytest.mark.fused_kda_gate
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available(),
    reason="requires NVIDIA CUDA",
)
def test_fused_kda_gate():
    bench = FusedKdaGateBenchmark(
        op_name="fused_kda_gate",
        torch_op=ref_fused_kda_gate,
    )
    bench.set_gems(gems_fused_kda_gate)
    bench.run()

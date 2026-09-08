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

import os

# ruff: noqa: I001
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ["FLAGTREE_AABS"] = "0"

from itertools import product  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import flaggems_vllm  # noqa: E402
from benchmark.base import Benchmark  # noqa: E402

try:
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    from flashinfer.norm import gemma_rmsnorm as baseline_op

    HAS_BASELINE_OP = True
except Exception as e:
    print(e)
    HAS_BASELINE_OP = False


class GemmaRmsNormBenchmark(Benchmark):
    _gemma_rmsnorm_ns = [1152, 5376, 16384]
    _gemma_rmsnorm_ms = [1, 32, 128, 256]
    _gemma_rmsnorm_shapes = list(product(_gemma_rmsnorm_ms, _gemma_rmsnorm_ns))

    def set_shapes(self, shape_file_path=None):
        self.shapes = GemmaRmsNormBenchmark._gemma_rmsnorm_shapes

    def get_input_iter(self, dtype):
        device = flaggems_vllm.runtime.device.name
        for shape in self.shapes:
            N = shape[-1]
            x = torch.randn(shape, dtype=dtype, device=device)
            w = torch.randn((N,), dtype=dtype, device=device)
            eps = 1e-5
            yield x, w, eps


@pytest.mark.skipif(
    not HAS_BASELINE_OP, reason="Missing baseline ops on current platform"
)
@pytest.mark.gemma_rmsnorm
def test_gemma_rmsnorm():
    dtypes = [torch.float16, torch.float32]
    bench = GemmaRmsNormBenchmark(
        op_name="gemma_rmsnorm",
        torch_op=baseline_op,
        dtypes=dtypes,
    )
    bench.set_gems(flaggems_vllm.gemma_rmsnorm)
    bench.run()

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
os.environ["FLAGTREE_AABS"] = "0"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

from itertools import product  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import flaggems_vllm  # noqa: E402
from benchmark.base import Benchmark  # noqa: E402

vendor = flaggems_vllm.vendor_name
try:
    if vendor == "nvidia" or vendor == "thead":

        os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        from flashinfer.norm import gemma_rmsnorm as baseline_op

        HAS_BASELINE_OP = True
    elif vendor == "hygon":

        from lightop import op

        def baseline_op(x, w, eps=1e-5):
            out = torch.empty_like(x)
            op.gemma_rmsnorm(out, x, w, eps)
            return out

        HAS_BASELINE_OP = True
    elif vendor == "ascend":

        from torch_npu import npu_gemma_rms_norm as baseline_op

        HAS_BASELINE_OP = True
except Exception as e:
    print(e)
    HAS_BASELINE_OP = False


class GemmaRmsNormBenchmark(Benchmark):
    # Shapes aligned to powers of two and the hidden dimensions of the Gemma‑series models
    _gemma_rms_norm_ns = [128, 256, 1024, 16384] + [
        1152,
        2048,
        2560,
        3072,
        3584,
        3840,
        4608,
        5376,
    ]
    # Batch size
    _gemma_rms_norm_ms = [1, 256, 1024]
    _gemma_rms_norm_shapes = list(product(_gemma_rms_norm_ms, _gemma_rms_norm_ns))

    def set_shapes(self, shape_file_path=None):
        self.shapes = GemmaRmsNormBenchmark._gemma_rms_norm_shapes

    def get_input_iter(self, dtype):
        device = flaggems_vllm.runtime.device.name
        for shape in self.shapes:
            N = shape[-1]
            x = torch.randn(shape, dtype=dtype, device=device)
            w = torch.randn((N,), dtype=dtype, device=device)
            eps = 1e-5
            yield x, w, eps


@pytest.mark.skipif(
    not HAS_BASELINE_OP, reason=f"Missing baseline ops on current platform: {vendor}"
)
@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm():
    dtypes = [torch.float16]
    bench = GemmaRmsNormBenchmark(
        op_name="gemma_rms_norm",
        torch_op=baseline_op,
        dtypes=dtypes,
    )
    bench.set_gems(flaggems_vllm.gemma_rms_norm)
    bench.run()

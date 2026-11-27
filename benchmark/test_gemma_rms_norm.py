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
import warnings

# ruff: noqa: I001
os.environ["FLAGTREE_AABS"] = "0"

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

    elif vendor == "hygon":

        from lightop import op

        def baseline_op(x, w, eps=1e-5):
            out = torch.empty_like(x)
            op.gemma_rmsnorm(out, x, w, eps)
            return out

    elif vendor == "ascend":

        from torch_npu import npu_gemma_rms_norm as baseline_op

    elif vendor == "mthreads":

        from vllm_musa.jit_kernel.csrc import gemma_rmsnorm

        def baseline_op(x, w, eps=1e-5):
            out = torch.empty_like(x)
            gemma_rmsnorm(x, w, eps, out, True)
            return out

    elif vendor == "iluvatar":

        import vllm_iluvatar

        vllm_iluvatar.register_platform()
        vllm_iluvatar.register_ops()

        from vllm_iluvatar.custom_kernels.gemma_rms_norm import gemma_rms_norm

        def baseline_op(x, w, eps=1e-5):
            out = torch.empty_like(x)
            gemma_rms_norm(out, x, w, eps)
            return out

except Exception as e:
    # Fall back to vllm.model_executor.layers.layernorm.GemmaRMSNorm when
    # the vendor-specific library (flashinfer, lightop, etc.) is absent.
    try:
        from vllm.model_executor.layers.layernorm import (
            GemmaRMSNorm as _VllmGemmaRMSNorm,
        )

        def baseline_op(x, w, eps=1e-5):
            # vLLM's CustomOp must be instantiated inside a vLLM config
            # context; newer vLLM raises "Current vLLM config is not set"
            # otherwise. CustomOp dispatch only reads .compilation_config,
            # and constructing a full VllmConfig() lazily imports
            # vllm.v1.executor -> ... -> xgrammar, which breaks on
            # environments with a mismatched tvm-ffi/xgrammar pair, so use
            # a minimal stand-in carrying only a CompilationConfig.
            from types import SimpleNamespace

            from vllm.config import CompilationConfig, set_current_vllm_config

            N = x.shape[-1]
            minimal_config = SimpleNamespace(compilation_config=CompilationConfig())
            with set_current_vllm_config(minimal_config):
                norm = _VllmGemmaRMSNorm(hidden_size=N, eps=eps)
            norm.weight = torch.nn.Parameter(w.float())
            norm = norm.to(x.device)
            with torch.inference_mode():
                return norm(x)

    except ImportError:
        print(e)
        baseline_op = None

if baseline_op is None:
    warnings.warn(
        f"Baseline op of current platform: {vendor} can't be loaded. "
        "There will be `N/A` outputs in some columns."
    )


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

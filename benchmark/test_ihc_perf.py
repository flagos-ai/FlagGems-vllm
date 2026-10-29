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

"""Performance benchmarks for iHC (independent Hyper-Connections) operators.

SpeedUp = latency_base / latency, where latency_base is the baseline op and
latency is the FlagGems-vllm Triton implementation. The baseline is vLLM's own
iHC Triton kernel (vllm.models.hy_v4.nvidia.triton_ihc) when vLLM is importable;
otherwise it falls back to the PyTorch reference (ihc_*_ref), matching the
pattern used by the mHC / MLA benchmarks in this repo.

Shapes target the HY V4 MoE model: hidden_size = 6144, hc_mult (residual
streams) = 4. iHC's hc_fn is a ReplicatedLinear, so hidden is NOT sharded under
tensor parallelism -- TP8 / TP4 only change the per-rank token count. We
therefore sweep a few representative num_tokens covering decode (small batch)
through prefill (large batch) under TP8 / TP4.
"""

import pytest
import torch

from flaggems_vllm.ops.ihc import ihc_post, ihc_post_ref, ihc_pre, ihc_pre_ref

from . import base

# Prefer vLLM's iHC Triton kernels as the baseline; fall back to the torch ref.
try:
    from vllm.models.hy_v4.nvidia.triton_ihc import triton_ihc_post as _vllm_ihc_post
    from vllm.models.hy_v4.nvidia.triton_ihc import triton_ihc_pre as _vllm_ihc_pre

    HAS_VLLM = True
    _pre_base = _vllm_ihc_pre
    _post_base = _vllm_ihc_post
except ImportError:
    HAS_VLLM = False
    _pre_base = ihc_pre_ref
    _post_base = ihc_post_ref

_HC_MULT = 4
_HIDDEN = 6144  # HY V4 model hidden_size (ReplicatedLinear -> not TP-sharded)

# Representative num_tokens: decode (1 tok/seq small batch) -> prefill (large).
# These bracket the per-rank token counts seen under TP8 / TP4 serving.
_TOKENS = [1, 64, 1024, 8192]


class IHCPreBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, hidden_size"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(n, _HIDDEN) for n in _TOKENS]

    def get_input_iter(self, dtype):
        for n, h in self.shapes:
            hc = _HC_MULT
            device = self.device
            torch.manual_seed(42)
            x = torch.randn((n, hc, h), dtype=dtype, device=device)
            weight = torch.randn((2 * hc, hc * h), dtype=torch.float32, device=device)
            scale = torch.randn((2,), dtype=torch.float32, device=device) * 0.01
            base_t = torch.randn((2 * hc,), dtype=torch.float32, device=device)
            yield x, weight, scale, base_t, 2.0, 1e-6, 1e-5


@pytest.mark.ihc_pre
def test_ihc_pre_perf():
    bench = IHCPreBenchmark(
        op_name="ihc_pre",
        torch_op=_pre_base,
        gems_op=ihc_pre,
        dtypes=[torch.bfloat16],
    )
    bench.run()


class IHCPostBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, hidden_size"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(n, _HIDDEN) for n in _TOKENS]

    def get_input_iter(self, dtype):
        for n, h in self.shapes:
            hc = _HC_MULT
            device = self.device
            torch.manual_seed(42)
            x = torch.randn((n, h), dtype=dtype, device=device)
            residual = torch.randn((n, hc, h), dtype=dtype, device=device)
            post = torch.randn((n, hc), dtype=torch.float32, device=device)
            yield x, residual, post


@pytest.mark.ihc_post
def test_ihc_post_perf():
    bench = IHCPostBenchmark(
        op_name="ihc_post",
        torch_op=_post_base,
        gems_op=ihc_post,
        dtypes=[torch.bfloat16],
    )
    bench.run()

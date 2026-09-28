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

import math
from typing import Generator

import pytest
import torch

import flaggems_vllm
from tests.test_kda_gate_cumsum import HAS_ASCENDC, _ascendc

from . import base

torch_npu = pytest.importorskip("torch_npu")

CHUNK = 64
H, D = 4, 128  # production per-rank shape [1,T,4,128] (TP16-sharded heads)
LOWER_BOUND = -5.0

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# Rows 1-7 (production): the recorded per-step token counts T in
# {8192,8188,8184,8180} plus the small chunked-prefill tails {40,36,24}.
# Rows 8-10 (stress): multi-request splits of a full block (request-boundary
# variants the profile does not itemise), a ragged split, and T=1.
GATE_CUMSUM_SHAPES = (
    # lens: per-sequence token counts
    # ---- production shapes (kernel_details) ----
    [8192],  # full block
    [8188],  # spec boundary
    [8184],  # spec boundary
    [8180],  # spec boundary (-12)
    [40],  # chunk tail
    [36],  # chunk tail
    [24],  # chunk tail
    # ---- stress shapes ----
    [4096, 4096],  # stress: 2-request split
    [4096, 2044, 2044],  # stress: ragged 3-way
    [2048, 2048, 2048, 2048],  # stress: even 4-way split (constructed)
)
# Shape provenance (production): [8192]/[4096,4096]/[4096,2044,2044]/[8180] are
# the real per-step token counts from the production profiler (GLM-5.3-Flash-W8A8
# 16k/1k/c4 prefill steps, kernel_details); [40,16] is a chunked-prefill tail step.


def _chunk_indices_host(cu_host, chunk):
    import triton

    idx = []
    for s in range(len(cu_host) - 1):
        n = triton.cdiv(cu_host[s + 1] - cu_host[s], chunk)
        idx.extend((s, c) for c in range(n))
    return (
        torch.tensor(idx, dtype=torch.int64)
        if idx
        else torch.zeros((0, 2), dtype=torch.int64)
    )


def _official(g, A_log, dt_bias, cu_host, lower_bound, chunk):
    _, T, Hh, Dd = g.shape
    gate = lower_bound * torch.sigmoid(
        (g.float() + dt_bias.view(1, 1, Hh, Dd)) * torch.exp(A_log).view(1, 1, Hh, 1)
    )
    out = torch.empty_like(gate)
    for s in range(len(cu_host) - 1):
        bos, eos = cu_host[s], cu_host[s + 1]
        for cs in range(bos, eos, chunk):
            ce = min(cs + chunk, eos)
            out[0, cs:ce] = gate[0, cs:ce].cumsum(dim=0)
    return out * (1.0 / math.log(2.0))


class KdaGateCumsumBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_desc = "lens"

    def set_shapes(self, shape_file_path=None):
        self.shapes = GATE_CUMSUM_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for i, lens in enumerate(self.shapes):
            torch.manual_seed(1000 + i)
            T = sum(lens)
            cu_host = [0]
            for n in lens:
                cu_host.append(cu_host[-1] + n)
            cu_dev = torch.tensor(cu_host, dtype=torch.int64, device=self.device)
            chunk_indices = _chunk_indices_host(cu_host, CHUNK).to(self.device)
            g = (
                torch.rand(1, T, H, D, dtype=torch.float32, device=self.device) * 6 - 3
            ).to(dtype)
            A_log = torch.rand(H, dtype=torch.float32, device=self.device) - 2.0
            dt_bias = (
                torch.rand(H * D, dtype=torch.float32, device=self.device) * 0.4 - 0.2
            )
            yield g, A_log, dt_bias, cu_dev, chunk_indices, LOWER_BOUND, CHUNK


@pytest.mark.kda_gate_cumsum
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized KDA gate cumsum targets Ascend",
)
def test_kda_gate_cumsum_perf():
    if HAS_ASCENDC:
        # the exact replaced operator: 9909 vs 319us at the 8192-token
        # production shape (31.1x); the E2E aggregate vs AscendC is ~96x
        # (84.6s -> 0.87s over the 16-rank serving run).
        torch_op = _ascendc
        op_name = "kda_gate_cumsum_vs_ascendc"
    else:
        # extension-free environments: torch-chain reference instead
        def torch_op(g, A_log, dt_bias, cu_dev, chunk_indices, lower_bound, chunk):
            return _official(g, A_log, dt_bias, cu_dev.tolist(), lower_bound, chunk)

        op_name = "kda_gate_cumsum"

    def gems_op(g, A_log, dt_bias, cu_dev, chunk_indices, lower_bound, chunk):
        return flaggems_vllm.kda_gate_cumsum_triton(
            g, A_log, dt_bias, cu_dev, chunk_indices, lower_bound, chunk
        )

    benchmark = KdaGateCumsumBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        gems_op=gems_op,
        dtypes=[torch.bfloat16],
    )
    benchmark.run()

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

from typing import Generator

import pytest
import torch

import flaggems_vllm

from . import base

torch_npu = pytest.importorskip("torch_npu")

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# Rows 1-4 (production): S=8256 recorded slot count, H=4 = 64 heads / TP16,
# V=K=128 = head_dim, N in the recorded per-step set {4,8,12,16}.
# Rows 5-10 (stress): N=32/64 beyond the recorded concurrency (max_num_seqs
# boundary / extreme), S=16384 doubled pool, H=8 deeper grid axis, K=32
# drives BK<64 -> BV=next_pow2(V) tiling branch, N=1 single-seq boundary.
SHAPES = (
    # S slots, N seqs, H heads, V, K
    # ---- production shapes (kernel_details) ----
    (8256, 4, 4, 128, 128),  # 4 seqs (production)
    (8256, 8, 4, 128, 128),  # 8 seqs
    (8256, 12, 4, 128, 128),  # 12 seqs
    (8256, 16, 4, 128, 128),  # 16 seqs
    # ---- stress shapes ----
    (8256, 32, 4, 128, 128),  # stress: max_num_seqs boundary
    (8256, 64, 4, 128, 128),  # stress: 64 seqs
    (16384, 16, 4, 128, 128),  # stress: doubled pool
    (8256, 16, 8, 128, 128),  # stress: H=8
    (8256, 16, 4, 64, 32),  # stress: K=32 tiling branch
    (8256, 1, 4, 128, 128),  # stress: 1 seq
)


class KdaStateScatterBenchmark(base.Benchmark):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shape_desc = "S,N,H,V,K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for i, (S, N, H, V, K) in enumerate(self.shapes):
            torch.manual_seed(1000 + i)
            cache = torch.randn(S, H, V, K, dtype=dtype, device=self.device)
            idx = torch.randperm(S, device="cpu")[:N].to(self.device)
            vals = torch.randn(N, H, K, V, dtype=dtype, device=self.device)
            yield cache, idx, vals


def _torch_scatter(cache, idx, vals):
    out = cache.clone()
    out[idx] = vals.transpose(-1, -2).contiguous().to(out.dtype)
    return out


def _gems_scatter(cache, idx, vals):
    out = cache.clone()
    flaggems_vllm.scatter_kda_state(out, idx, vals)
    return out


@pytest.mark.kda_state_scatter
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="targets Ascend")
def test_kda_state_scatter_perf():
    KdaStateScatterBenchmark(
        op_name="kda_state_scatter",
        torch_op=_torch_scatter,
        gems_op=_gems_scatter,
        dtypes=[torch.bfloat16],
    ).run()

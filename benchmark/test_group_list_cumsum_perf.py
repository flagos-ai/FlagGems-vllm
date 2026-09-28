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
# Production: n=18 = 288 routed experts / EP16 (the exact per-rank count);
# 288 = the unsharded (EP1) layout of the same expert count.
# Stress: 36/96 = TP8/TP3-style shard sizes, powers of two, and tails.
SHAPES = (
    # n (num_experts)
    # ---- production shapes (kernel_details) ----
    18,  # 288/EP16 (production)
    288,  # unsharded EP1
    # ---- stress shapes ----
    1,  # single
    2,  # minimal
    17,  # 18-1
    19,  # 18+1
    32,  # pow2
    36,  # 288/8
    96,  # 288/3
    576,  # 2x unsharded
)


class GroupListCumsumBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_desc = "num_experts"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for i, n in enumerate(self.shapes):
            torch.manual_seed(1000 + i)
            x = torch.randint(0, 4096, (n,), dtype=torch.int64, device=self.device)
            yield x


def _official(x):
    # torch.cumsum on integer dtypes dispatches to AI_CPU on Ascend (~82us
    # at the EP int64[18] shape); this is the exact replaced baseline.
    return x.cumsum(dim=0)


@pytest.mark.group_list_cumsum
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized group-list cumsum targets Ascend",
)
def test_group_list_cumsum_perf():
    benchmark = GroupListCumsumBenchmark(
        op_name="group_list_cumsum",
        torch_op=_official,
        gems_op=flaggems_vllm.group_list_cumsum,
        dtypes=[torch.int64],
    )
    benchmark.run()

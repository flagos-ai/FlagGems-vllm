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
from tests.test_indexer_epilogue import KPOOL, POOL_TOPK, _make_topk
from tests.test_indexer_epilogue import _official_epilogue as _ref_epilogue

from . import base

torch_npu = pytest.importorskip("torch_npu")

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# Production: rows = per-step token counts (8192/8180 full-block,
# 56 chunk tail, 16 decode) over the 6220-pool cache, POOL_TOPK=512.
# Stress: doubled rows/pools, non-pow2 pool tail tile, small pool,
# 1-row boundary.
SHAPES = (
    # rows, pools
    # ---- production shapes (kernel_details) ----
    (8192, 6220),  # production prefill
    (8180, 6220),  # spec boundary
    (56, 6220),  # chunk tail
    (16, 6220),  # decode rows
    # ---- stress shapes ----
    (16384, 6220),  # stress: 2x rows
    (8192, 12440),  # stress: 2x pools
    (8192, 6221),  # stress: non-pow2 pools
    (128, 1024),  # stress: small pool
    (1, 6220),  # stress: 1 row
    (64, 777),  # stress: ragged
)


class IndexerEpilogueBenchmark(base.Benchmark):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shape_desc = "rows,pools"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for rows, pools in self.shapes:
            tv, pid = _make_topk(rows, pools)
            positions = torch.arange(rows, dtype=torch.int64, device=self.device)
            yield tv, pid, positions, 0, POOL_TOPK, KPOOL


def _torch(tv, pid, positions, ts, pt, kp):
    return _ref_epilogue(tv, pid, positions)


@pytest.mark.indexer_epilogue
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="targets Ascend")
def test_indexer_epilogue_perf():
    IndexerEpilogueBenchmark(
        op_name="indexer_epilogue",
        torch_op=_torch,
        gems_op=flaggems_vllm.indexer_epilogue,
        dtypes=[torch.float32],
    ).run()

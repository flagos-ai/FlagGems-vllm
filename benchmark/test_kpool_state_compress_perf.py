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
from tests.test_kpool_state_compress import _make_case, _reference

from . import base

torch_npu = pytest.importorskip("torch_npu")

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# Production: lens = recorded per-step token mixes over the [6220,4,256]
# compressor cache ([8192] full block, [8184] spec boundary,
# [4096,4090] two-request block, [56] chunk tail).
# Stress: doubled block, four-way split, ragged pairs, small blocks,
# T=1 boundary.
SHAPES = (
    # lens: per-sequence token counts
    # ---- production shapes (kernel_details) ----
    [8192],  # full block
    [8184],  # spec boundary
    [4096, 4090],  # 2-request ragged
    [56],  # chunk tail
    # ---- stress shapes ----
    [16384],  # stress: 2x block
    [2048, 2048, 2048, 2048],  # stress: 4-way split
    [100, 37],  # stress: ragged pair
    [64],  # stress: small
    [1],  # stress: T=1
    [8188],  # spec boundary
)


class KpoolStateCompressBenchmark(base.Benchmark):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shape_desc = "lens"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for lens in self.shapes:
            yield _make_case(lens)


# torch_op is the vectorised torch chain (bucketize + two-source gather +
# softmax + weighted-sum small-op flood) -- the class of implementation the
# triton kernel replaces.  Measured 3044 vs 935us at the 8192-token
# production shape (3.26x); the in-service kernel-to-kernel delta (vs the
# Round-1 triton version) is 952 -> ~900us (3-5%, the batching change).
def _run_ref(*args):
    sc, ic = args[0].clone(), args[1].clone()
    _reference(sc, ic, *args[2:])


def _run_gems(*args):
    sc, ic = args[0].clone(), args[1].clone()
    flaggems_vllm.kpool_state_compress(sc, ic, *args[2:])


@pytest.mark.kpool_state_compress
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="targets Ascend")
def test_kpool_state_compress_perf():
    KpoolStateCompressBenchmark(
        op_name="kpool_state_compress",
        torch_op=_run_ref,
        gems_op=_run_gems,
        dtypes=[torch.float32],
    ).run()

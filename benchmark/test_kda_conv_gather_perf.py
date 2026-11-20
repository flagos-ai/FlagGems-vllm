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
from tests.test_kda_conv_gather import UPSTREAM_COPY_CONV_STATE

from . import base

torch_npu = pytest.importorskip("torch_npu")

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# Production: (state_len=6, conv_dim=1536) exact layout (conv kernel 4 - 1
# + MTP spec 3 = 6 rows, 3*64*128/TP16 = 1536 cols); N in the recorded
# per-step set {4,8,12,16} over the S=8256 slot pool.
# Stress: N=32 max_num_seqs boundary, S=16384 doubled pool, conv_dim=3072
# (TP8 sharding assumption), N=64 extreme depth, N=1 single-seq,
# small-S decode shape.
SHAPES = (
    # S slots, N seqs, W state_len, C conv_dim
    # ---- production shapes (kernel_details) ----
    (8256, 4, 6, 1536),  # 4 seqs (production)
    (8256, 8, 6, 1536),  # 8 seqs
    (8256, 12, 6, 1536),  # 12 seqs
    (8256, 16, 6, 1536),  # 16 seqs
    # ---- stress shapes ----
    (8256, 32, 6, 1536),  # stress: max_num_seqs boundary
    (16384, 16, 6, 1536),  # stress: doubled pool
    (8256, 16, 6, 3072),  # stress: TP8 conv_dim
    (8256, 64, 6, 1536),  # stress: 64 seqs
    (8256, 1, 6, 1536),  # stress: 1 seq
    (8256, 16, 12, 1536),  # stress: doubled state length (W=12)
)


class KdaConvGatherBenchmark(base.Benchmark):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shape_desc = "S,N,W,C"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for i, (S, N, W, C) in enumerate(self.shapes):
            torch.manual_seed(1000 + i)
            cstate = torch.randn(S, W, C, dtype=dtype, device=self.device)
            idx = torch.arange(N, dtype=torch.int32, device=self.device)
            idx[torch.randperm(N, device="cpu")[: N // 4].to(self.device)] = -1
            # Pre-built upstream metadata: query_start_loc already exists when
            # the packer runs in production, so keep its construction out of
            # the timed region for a fair head-to-head.
            starts = torch.arange(N + 1, dtype=torch.int32, device=self.device)
            yield cstate, idx, starts


def _torch_gather(cstate, idx, starts=None):
    flat = idx.flatten()
    valid = flat != -1
    safe = flat.masked_fill(~valid, 0).to(torch.long)
    staged = cstate.index_select(0, safe).contiguous()
    local = torch.arange(
        flat.numel(), dtype=torch.int32, device=flat.device
    ).masked_fill(~valid, -1)
    return staged, local


def _upstream_gather(cstate, idx, starts):
    """Timing-only upstream baseline: allocation + kernel, no marker
    mapping.  Production consumes packed_indices directly (null_block_id
    semantics, causal_conv1d.py), so the -1 mapping used by the accuracy
    test would only inflate the baseline (measured ~45% at N=16)."""
    flat = idx.reshape(-1)
    n = flat.shape[0]
    packed = torch.empty(
        (n + 1, *cstate.shape[1:]), dtype=cstate.dtype, device=cstate.device
    )
    packed_indices = torch.empty(n, dtype=torch.int32, device=flat.device)
    UPSTREAM_COPY_CONV_STATE(
        cstate, packed, flat, starts, packed_indices, write_back=False
    )
    return packed, packed_indices


def _gather_conv_state(cstate, idx, starts=None):
    return flaggems_vllm.gather_conv_state(cstate, idx)


@pytest.mark.kda_conv_gather
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="targets Ascend")
def test_kda_conv_gather_perf():
    baseline = (
        _upstream_gather
        if UPSTREAM_COPY_CONV_STATE is not None
        else _torch_gather
    )
    KdaConvGatherBenchmark(
        op_name="kda_conv_gather",
        torch_op=baseline,
        gems_op=_gather_conv_state,
        dtypes=[torch.bfloat16],
    ).run()

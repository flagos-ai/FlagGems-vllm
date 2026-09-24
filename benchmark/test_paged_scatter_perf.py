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
# Production: (6220,4,256)x8192/8180 = compressor-state prefill write /
# spec-merge tail (kernel_details); x16 decode step; x56 chunked-prefill tail.
# Stress: doubled pool, full-size MLA KV layout (18660,128,1,512) bf16
# (2.4GB; shrunk variant in the accuracy UT), small decode cache, n=16384
# beyond one block-table generation, MLA layout edges.
SHAPES = (
    # cache_shape, n rows, block_size
    # ---- production shapes (kernel_details) ----
    ((6220, 4, 256), 8192, 4),  # compressor prefill (production)
    ((6220, 4, 256), 8180, 4),  # spec-merge tail
    ((6220, 4, 256), 16, 4),  # decode step
    ((6220, 4, 256), 56, 4),  # chunk tail
    # ---- stress shapes ----
    ((12440, 4, 256), 8192, 4),  # stress: doubled pool
    ((18660, 128, 1, 512), 8192, 128),  # stress: full MLA KV
    ((64, 4, 256), 64, 4),  # stress: small cache
    ((6220, 4, 256), 16384, 4),  # stress: >1 table gen
    ((1166, 128, 1, 512), 512, 128),  # stress: MLA shrunk
    ((32, 128, 1, 512), 56, 128),  # stress: MLA tiny
)


class PagedScatterBenchmark(base.Benchmark):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shape_desc = "cache,n,bs"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for i, (shape, n, bs) in enumerate(self.shapes):
            torch.manual_seed(1000 + i)
            cache = torch.zeros(shape, dtype=dtype, device=self.device)
            values = torch.randn(n, shape[-1], dtype=dtype, device=self.device)
            total = shape[0] * bs
            slots = (
                torch.randperm(total, device="cpu")[: min(n, total)]
                .to(torch.int64)
                .to(self.device)
            )
            if slots.numel() < n:
                pad = torch.randint(0, total, (n - slots.numel(),), device="cpu")
                slots = torch.cat([slots, pad.to(torch.int64)]).to(self.device)
            slots[torch.randperm(n, device="cpu")[: n // 16].to(self.device)] = -1
            yield cache, slots, values, bs


def _torch_scatter(cache, slots, values, bs):
    valid = (slots >= 0) & (slots < cache.shape[0] * bs)
    safe = torch.where(valid, slots, torch.zeros_like(slots))
    bids = torch.div(safe, bs, rounding_mode="floor")
    offs = torch.remainder(safe, bs)
    sv = torch.where(valid.view(-1, 1), values, cache[0, 0].unsqueeze(0))
    out = cache.clone()
    out[bids, offs] = sv.reshape(-1, *cache.shape[2:])
    return out


def _gems_scatter(cache, slots, values, bs):
    out = cache.clone()
    assert flaggems_vllm.paged_scatter_triton(out, slots, values, bs)
    return out


@pytest.mark.paged_scatter
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="targets Ascend")
def test_paged_scatter_perf():
    PagedScatterBenchmark(
        op_name="paged_scatter",
        torch_op=_torch_scatter,
        gems_op=_gems_scatter,
        dtypes=[torch.bfloat16],
    ).run()

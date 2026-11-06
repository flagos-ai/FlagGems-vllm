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
from tests.test_slot_mapping import HAS_UPSTREAM, _upstream_slot_kernel

from . import base

torch_npu = pytest.importorskip("torch_npu")

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# Production: bs=128 (MLA KV table) with nreq in {4,8,12,16} and bs=16
# (compressor table) at the recorded max_num_batched_tokens=8192.
# Stress: nreq=32 max_num_seqs boundary / 128 extreme depth, compressor
# table at higher nreq, doubled maxtok, single request.
SHAPES = (
    # reqs, maxtokens, blocksize
    # ---- production shapes (kernel_details) ----
    (4, 8192, 128),  # 4 reqs, MLA table
    (8, 8192, 128),  # 8 reqs
    (12, 8192, 128),  # 12 reqs
    (16, 8192, 128),  # 16 reqs
    (4, 8192, 16),  # 4 reqs, compressor table
    # ---- stress shapes ----
    (32, 8192, 128),  # stress: 32 reqs
    (128, 8192, 128),  # stress: 128 reqs
    (8, 8192, 16),  # stress: compressor, 8 reqs
    (8, 16384, 128),  # stress: doubled maxtok
    (1, 8192, 128),  # stress: 1 req
)


class SlotMappingBenchmark(base.Benchmark):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shape_desc = "nreq,maxtok,bs"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for nreq, maxtok, bs in self.shapes:
            tlen = 200
            qsl = torch.tensor(
                [0] + [tlen * (i + 1) for i in range(nreq)],
                dtype=torch.int32,
                device=self.device,
            )
            pos = torch.cat(
                [torch.arange(tlen, dtype=torch.int32) for _ in range(nreq)]
            ).to(self.device)
            maxb = (16384 + bs - 1) // bs + 1
            btab = (
                torch.arange(nreq * maxb, dtype=torch.int32).reshape(nreq, maxb) % 900
            ).to(self.device)
            yield tlen * nreq, maxtok, qsl, nreq, pos, btab, bs


def _gems(num, maxtok, qsl, nreq, pos, btab, bs):
    out = torch.full((maxtok,), -1, dtype=torch.int64, device=qsl.device)
    flaggems_vllm.compute_slot_mapping_parallel(
        num, maxtok, qsl, nreq, pos, btab, out, block_size=bs
    )
    return out


def _torch_ref(num, maxtok, qsl, nreq, pos, btab, bs):
    out = torch.empty((maxtok,), dtype=torch.int64, device=qsl.device)
    rows = torch.repeat_interleave(
        torch.arange(nreq, device=qsl.device), qsl[1:] - qsl[:-1]
    )
    p = pos.to(torch.int64)
    out[:num] = btab[rows, p // bs].to(torch.int64) * bs + (p % bs)
    return out


def _upstream(num, maxtok, qsl, nreq, pos, btab, bs):
    """The exact replaced per-request kernel (E2E 12.9x at 1050.9 -> 81.2
    ms/rank over the serving run; the host-pipeline number here is smaller
    because the micro shape underfills its serial walk)."""
    out = torch.full((maxtok,), -1, dtype=torch.int64, device=qsl.device)
    _upstream_slot_kernel[(nreq + 1,)](
        num,
        maxtok,
        qsl,
        pos,
        btab,
        btab.stride(0),
        bs,
        out,
        BLOCK_SIZE=1024,
        TOTAL_CP_WORLD_SIZE=1,
        TOTAL_CP_RANK=0,
        CP_KV_CACHE_INTERLEAVE_SIZE=1,
        PAD_ID=-1,
    )
    return out


@pytest.mark.slot_mapping
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="targets Ascend")
def test_slot_mapping_perf():
    torch_op = _upstream if HAS_UPSTREAM else _torch_ref
    op_name = "slot_mapping_vs_upstream" if HAS_UPSTREAM else "slot_mapping"
    SlotMappingBenchmark(
        op_name=op_name, torch_op=torch_op, gems_op=_gems, dtypes=[torch.int32]
    ).run()

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

import importlib
from typing import Generator

import pytest
import torch

import flaggems_vllm
from tests.test_indexer_gemm_score import build_inputs

from . import base

_mod = importlib.import_module(
    "flaggems_vllm.runtime.backend._ascend.ops.indexer_gemm_score"
)

try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except (ImportError, AttributeError):
    _NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _NPU_AVAILABLE, reason="indexer_gemm_score requires an Ascend NPU"
)

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# Production: (2048,4,6220,8192) = the recorded prefill scorer shape
# (4 reqs x 2048 tokens, 6220 pools, high positions); (2048,1,...) the
# recorded single-request variant.
# Stress: higher nreq / doubled pools / non-pow2 pool tail tile / ragged
# short-seq from-zero variants / high positions / the 16-row matvec fallback;
# token counts stay within the scheduler's 8192-per-step cap (16384-token
# forms fault the vector cores and never occur in production).
SHAPES = (
    # tokens_per_req, n_req, max_pool, pos_base
    # ---- production shapes (kernel_details) ----
    (2048, 4, 6220, 8192),  # production prefill
    (2048, 1, 6220, 8192),  # single request
    # ---- stress shapes ----
    (1024, 8, 6220, 8192),  # stress: 8 reqs
    (4096, 2, 6220, 16384),  # stress: high pos
    (2048, 4, 12440, 8192),  # stress: 2x pools
    (2048, 4, 2049, 8192),  # stress: non-pow2
    (300, 3, 777, 1024),  # stress: ragged
    (128, 2, 2048, 4096),  # stress: short/high
    (64, 4, 1024, 0),  # stress: short/zero
    (4, 4, 6220, 8192),  # fallback path: 16 rows total (4 reqs x 4 spec tokens)
)


class IndexerGemmScoreBenchmark(base.Benchmark):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shape_desc = "tokens_per_req,n_req,max_pool,pos_base"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for spec in self.shapes:
            yield build_inputs(*spec)


def _matvec(qbar, cache, cum_q, seq_lens, bt, positions, token_start=0):
    """The exact replaced per-token scorer (same file, small-row fallback):
    13341 vs 955us at the production shape (14.0x); E2E 5343.6 -> 1130.2
    ms/rank (-67.4s/16rank)."""
    num_tokens = qbar.shape[0]
    max_pool = int(seq_lens.max().item())
    num_reqs = cum_q.shape[0]
    out = torch.full(
        (num_tokens, max_pool), float("-inf"), dtype=torch.float32, device=qbar.device
    )
    num_chunks = (
        max_pool + _mod.TRITON_POOL_CHUNK_SIZE - 1
    ) // _mod.TRITON_POOL_CHUNK_SIZE
    _mod._glm5_next_lightning_indexer_score_kernel[(num_tokens, num_chunks)](
        qbar,
        cache,
        cum_q,
        seq_lens,
        bt,
        positions,
        out,
        token_start,
        max_pool,
        num_reqs,
        cache.shape[0],
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        bt.stride(0),
        bt.stride(1),
        cache.shape[1],
        _mod._next_power_of_2(max(1, num_reqs)),
        128,
        4,
        _mod.TRITON_POOL_CHUNK_SIZE,
        _mod.TRITON_POOL_SUBTILE_SIZE,
    )
    return out


def _torch_reference(qbar, cache, cum_q, seq_lens, bt, positions):
    """Torch equivalent of the replaced per-token scorer (matvec order).

    Vectorised torch form: per row, gather visible pool rows and dot with qbar.
    Kept O(T*visible) on GPU via einsum on a padded gather for the benchmark.
    """
    num_tokens, head_dim = qbar.shape
    max_pool = int(seq_lens.max().item())
    ends = cum_q
    req = (torch.arange(num_tokens, device=qbar.device)[:, None] >= ends[None, :]).sum(
        1
    )
    req = req.clamp_max(cum_q.shape[0] - 1)
    pos = positions
    visible = torch.minimum((pos + 1) // 4, seq_lens[req]).clamp_max(max_pool)
    pool_block = cache.shape[1]
    p = torch.arange(max_pool, device=qbar.device)
    off = p % pool_block
    phys = bt[req][:, p // pool_block].long()
    rows = phys * pool_block + off
    cache_f = cache.view(-1, head_dim).float()
    k = cache_f[rows]  # [T, max_pool, D]
    vis_mask = p[None, :] < visible[:, None]
    scores = torch.einsum("td,tpd->tp", qbar, k)
    return torch.where(vis_mask, scores, torch.full_like(scores, float("-inf")))


@pytest.mark.indexer_gemm_score
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized indexer GEMM scorer targets Ascend",
)
def test_indexer_gemm_score_perf():
    IndexerGemmScoreBenchmark(
        op_name="indexer_gemm_score_vs_matvec",
        torch_op=_matvec,
        gems_op=flaggems_vllm.indexer_gemm_score,
        dtypes=[torch.float32],
    ).run()

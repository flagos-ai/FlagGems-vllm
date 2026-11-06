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

import pytest
import torch

import flaggems_vllm

from .conftest import QUICK_MODE

torch_npu = pytest.importorskip("torch_npu")

DEVICE = "npu"
KPOOL = 4
POOL_TOPK = 512

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# NOTE: accuracy parametrisation adds pos_base (the first token's global
# position) on top of the perf (rows, pools) pairs: production prefill rows
# sit at high positions (8192), decode/short rows at low positions.
FULL_CASES = (
    # rows, pools, pos_base
    # ---- production shapes (kernel_details) ----
    (8192, 6220, 8192),  # (rows,pools,pos_base) production prefill at high positions
    (8180, 6220, 8192),  # spec-boundary row count, high positions
    (56, 6220, 8192),  # chunked-prefill tail rows, high positions
    (16, 6220, 0),  # decode step rows, from-zero positions
    # ---- stress shapes ----
    (16384, 6220, 8192),  # stress: doubled rows (2 blocks)
    (8192, 12440, 8192),  # stress: doubled pool count
    (8192, 6221, 8192),  # stress: non-pow2 pools (tail tile partial)
    (128, 1024, 0),  # stress: small pool, decode rows
    (1, 6220, 8192),  # stress: single row
    (64, 777, 1024),  # stress: ragged small pool at mid positions
)
CASES = (FULL_CASES[0], FULL_CASES[2]) if QUICK_MODE else FULL_CASES

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized indexer epilogue targets Ascend",
)


def _make_topk(rows, pools, neg_frac=0.2):
    scores = torch.randn(rows, pools, dtype=torch.float32, device=DEVICE)
    neg = torch.rand(rows, pools, device=DEVICE) < neg_frac
    scores = torch.where(neg, torch.full_like(scores, float("-inf")), scores)
    topk = min(POOL_TOPK, pools)
    return torch.topk(scores, topk, dim=1)


def _official_epilogue(
    topk_vals, pool_ids, positions, pool_topk=POOL_TOPK, index_kpool=KPOOL
):
    """The exact replaced torch chain, lifted verbatim from the vllm-ascend
    wrapper (ops/triton/glm5_next_lightning_indexer.py, the post-topk section
    of the non-triton path): topk -> -inf masking -> pool*4+offset expansion
    -> pad to pool_topk -> causal tail -> cat/cast.
    """
    dev = topk_vals.device
    rows = topk_vals.shape[0]
    topk = topk_vals.shape[1]
    token_offsets = torch.arange(index_kpool, device=dev)
    tail_offsets = torch.arange(index_kpool - 1, device=dev)
    pool_ids = torch.where(
        topk_vals == float("-inf"),
        torch.full_like(pool_ids, -1),
        pool_ids,
    )
    history = pool_ids.unsqueeze(-1) * index_kpool + token_offsets
    history = torch.where(
        pool_ids.unsqueeze(-1) >= 0,
        history,
        torch.full_like(history, -1),
    ).reshape(rows, topk * index_kpool)
    if topk < pool_topk:
        history = torch.nn.functional.pad(
            history, (0, (pool_topk - topk) * index_kpool), value=-1
        )
    pos = positions.to(dev)
    tail_start = (pos + 1) // index_kpool * index_kpool
    tail_count = pos + 1 - tail_start
    tail = torch.where(
        tail_offsets[None, :] < tail_count[:, None],
        tail_start[:, None] + tail_offsets[None, :],
        -1,
    )
    return torch.cat([history, tail], dim=1).to(torch.int32).view(rows, 1, -1)


# Shape provenance: (8192, 6220) is the production post-topk epilogue shape
# (8192-token prefill over a 6220-pool cache, POOL_TOPK=512 = 2048//4);
# pos_base>0 exercises high positions as in production chunked prefill.


@pytest.mark.indexer_epilogue
@pytest.mark.parametrize("rows,pools,pos_base", CASES)
def test_indexer_epilogue_accuracy(rows, pools, pos_base):
    tv, pid = _make_topk(rows, pools)
    positions = (torch.arange(rows) + pos_base).to(torch.int64).to(DEVICE)
    expected = _official_epilogue(tv, pid, positions)
    actual = flaggems_vllm.indexer_epilogue(tv, pid, positions, 0, POOL_TOPK, KPOOL)
    torch.npu.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.indexer_epilogue
def test_indexer_epilogue_pool_out():
    tv, pid = _make_topk(64, 128)
    positions = torch.arange(64, dtype=torch.int64, device=DEVICE)
    out = flaggems_vllm.indexer_epilogue(
        tv, pid, positions, 0, POOL_TOPK, KPOOL, pool_out=True
    )
    assert out.shape == (64, 1, POOL_TOPK + 1)
    torch.npu.synchronize()
    # pair-leader contract applies to the topk columns only: within the first
    # POOL_TOPK columns a valid id must never follow a -1 (the consumer stops
    # scanning at the first -1).  The trailing tail-pool column is always
    # written when non-empty and may legitimately sit after -1s.
    topk_part = out.view(64, -1)[:, :POOL_TOPK]
    valid_after = ((topk_part == -1).float().cumsum(dim=1) > 0) & (topk_part != -1)
    assert not valid_after.any()

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

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vllm-plugin-FL quant_lightning_indexer.py.
"""Independent device-reference comparison; replay excludes compile and warmup.

Active set covers small decode, model decode, odd/general, and bounded prefill.
Metric: median milliseconds/replay; SpeedUp = device reference / FlagGems-vllm (higher
is better). Report each cell and geometric mean. Migration latency acceptance
uses the separate same-shape old-FL comparison.
"""

import json
import math
import statistics
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from tests.test_quant_lightning_indexer import check, data, invoke

pytestmark = [
    pytest.mark.quant_lightning_indexer,
    pytest.mark.skipif(flaggems_vllm.device != "npu", reason="Requires Ascend"),
]

# (tokens, heads, dimension, physical pages, page size, request table columns, topk)
ACTIVE_SET = [
    (1, 64, 128, 8, 32, 4, 32),
    (4, 64, 128, 40, 32, 32, 512),
    (4, 3, 33, 6, 8, 3, 7),
    (32, 64, 128, 40, 32, 8, 128),
]


def make_case(case):
    t, h, d, pages, block, cols, top = case
    # data()'s four-row fixture is extended here without changing the seed or
    # input distributions used by the correctness suite.
    args = data(t=1, h=h, d=d, pages=pages, block=block, top=top)
    generator = torch.Generator().manual_seed(2451 + t)
    args[0] = torch.randint(
        -7, 8, (t, h, d), generator=generator, dtype=torch.int8
    ).npu()
    args[2] = torch.rand((t, h), generator=generator).npu()
    args[3] = (torch.arange(2 * cols).reshape(2, cols) % pages).long().npu()
    args[4] = (torch.arange(t) % 2).long().npu()
    args[5] = torch.full((t,), cols * block * 4 - 1, dtype=torch.int64).npu()
    args[6] = torch.full((2,), cols * block, dtype=torch.int64).npu()
    args[8] = torch.rand((t, h, 1), generator=generator).npu()
    return args


def graph_latency(fn, *, samples=7, repeats=20):
    # Allocations and kernel compilation are outside the measured interval.
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fn()
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    values = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repeats)
    assert output.dtype == torch.int32
    return statistics.median(values)


def torch_reference(
    query,
    key_cache,
    weights,
    block_table,
    request_ids,
    positions,
    key_lengths,
    topk,
    compress_ratio=4,
    query_scale=None,
    key_scale_cache=None,
):
    """Independent FP32 device baseline for the all-valid benchmark workloads.

    The benchmark deliberately uses valid page/request IDs and C4 only; the
    functional suite's independent CPU oracle checks invalid metadata semantics.
    """
    assert compress_ratio == 4
    pages = block_table[request_ids.long()].long()
    keys = key_cache[pages].flatten(1, 2).float()
    q = query.float()
    if query_scale is not None:
        q = q * query_scale.reshape(*query.shape[:2], 1)
    if key_scale_cache is not None:
        keys = keys * key_scale_cache[pages].flatten(1, 2)
    dots = torch.einsum("thd,tcd->thc", q, keys)
    scores = (dots.relu() * weights[..., None]).sum(1)
    cap = keys.shape[1]
    limit = torch.minimum((positions + 1) // 4, key_lengths[request_ids.long()])
    valid = torch.arange(cap, device=query.device)[None, :] < limit[:, None]
    scores = scores.masked_fill(~valid, -torch.inf)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ids = order[:, : min(topk, cap)]
    picked = scores.gather(1, ids)
    ids = ids.masked_fill(~torch.isfinite(picked), -1).to(torch.int32)
    if topk > cap:
        ids = torch.cat(
            (
                ids,
                torch.full(
                    (query.shape[0], topk - cap),
                    -1,
                    dtype=torch.int32,
                    device=query.device,
                ),
            ),
            1,
        )
    return ids


def test_quant_lightning_indexer_benchmark():
    baseline = SimpleNamespace(quant_lightning_indexer=torch_reference)
    speedups = []
    for case in ACTIVE_SET:
        args = make_case(case)
        # Both paths must pass the independent FP64 CPU oracle before timing.
        check(args)
        check(args, mod=baseline)
        base_ms = graph_latency(lambda: invoke(baseline, args))
        gems_ms = graph_latency(lambda: invoke(flaggems_vllm, args))
        speedup = base_ms / gems_ms
        speedups.append(speedup)
        print(
            json.dumps(
                {
                    "op": "quant_lightning_indexer",
                    "shape": case,
                    "reference_ms": base_ms,
                    "flaggems_ms": gems_ms,
                    "SpeedUp": speedup,
                }
            ),
            flush=True,
        )
    geomean = math.exp(statistics.mean(math.log(x) for x in speedups))
    print(
        json.dumps({"op": "quant_lightning_indexer", "geomean_SpeedUp": geomean}),
        flush=True,
    )

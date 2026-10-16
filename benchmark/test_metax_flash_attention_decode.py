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

"""Compare full-API graph replay for legacy and Split-KV GQA6 decode.

Run on an idle MetaX device:
    python -m pytest -q -s benchmark/test_metax_flash_attention_decode.py

The baseline is the existing legacy route selected by num_splits=1. CUDA
events include layout conversion, partial attention, and merge. This is an
operator microbenchmark, not an end-to-end model throughput measurement.
"""

import json
import statistics

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX route"),
]


@pytest.mark.parametrize("batch", [1, 16, 64])
@pytest.mark.parametrize("length", [1024, 4096, 16384])
@pytest.mark.parametrize("graph_upper_bound", [False, True])
@torch.inference_mode()
def test_metax_gqa6_decode_graph(batch, length, graph_upper_bound):
    torch.manual_seed(41)
    page = 784
    pages = (length + page - 1) // page
    q = torch.randn(batch, 12, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * pages, page, 2, 256, device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.randperm(batch * pages, device=q.device).reshape(batch, pages)
    table = table.to(torch.int32)
    used = torch.full((batch,), length, device=q.device, dtype=torch.int32)
    cu = torch.arange(batch + 1, device=q.device, dtype=torch.int32)
    out = torch.empty_like(q)
    max_k = 262144 if graph_upper_bound else length

    def run(num_splits):
        return flaggems_vllm.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=used,
            max_seqlen_q=1,
            max_seqlen_k=max_k,
            block_table=table,
            causal=True,
            num_splits=num_splits,
        )

    graphs, outputs = {}, {}
    for role, splits in (("baseline", 1), ("candidate", 0), ("revert", 1)):
        for _ in range(4):
            run(splits)
        torch.cuda.synchronize()
        outputs[role] = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(splits)
        graphs[role] = graph
    torch.testing.assert_close(
        outputs["candidate"], outputs["baseline"], atol=0.002, rtol=0.02
    )

    rounds = {role: [] for role in graphs}
    for cycle in range(5):
        order = ("baseline", "candidate", "revert")
        if cycle % 2:
            order = tuple(reversed(order))
        for role in order:
            graph = graphs[role]
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(40):
                graph.replay()
            end.record()
            end.synchronize()
            rounds[role].append(start.elapsed_time(end) / 40)
    medians = {role: statistics.median(values) for role, values in rounds.items()}
    print(
        json.dumps(
            {
                "batch": batch,
                "kv_tokens": length,
                "max_seqlen_k_argument": max_k,
                "rounds_ms": rounds,
                "median_ms": medians,
                "speedup": medians["baseline"] / medians["candidate"],
            }
        )
    )

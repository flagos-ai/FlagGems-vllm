# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Matched native/new BF16-QK timing, distinct eager and CUDA Graph results.

Fixed 30 warmup calls and 100 interleaved event samples match the historical
attention campaign. Normal benchmark --warmup/--iter are millisecond budgets;
this test intentionally records fixed call counts instead of reinterpreting
those options. --level core uses q512/q2048 and full level adds q4096/q8192.
"""

import json
import statistics

import pytest
import torch

import flaggems_vllm
from tests.minimax_m3_sparse_attn_utils import dense_reference, make_inputs

from . import conftest as bench_cfg

pytestmark = [
    pytest.mark.minimax_m3_sparse_attn,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "nvidia", reason="NVIDIA only"),
]


def _pair(fns, graph, warmup=30, samples=100):
    for _ in range(warmup):
        for fn in fns:
            fn()
    torch.cuda.synchronize()
    graphs = []
    if graph:
        for fn in fns:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            graphs.append(g)
        fns = [g.replay for g in graphs]
        for _ in range(warmup):
            for fn in fns:
                fn()
        torch.cuda.synchronize()
    events = [
        [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(samples)
        ]
        for _ in fns
    ]
    for i in range(samples):
        order = range(len(fns)) if i % 2 == 0 else reversed(range(len(fns)))
        for j in order:
            start, end = events[j][i]
            start.record()
            fns[j]()
            end.record()
    torch.cuda.synchronize()
    result = []
    for pairs in events:
        values = [start.elapsed_time(end) * 1000 for start, end in pairs]
        result.append(
            {
                "median_us": statistics.median(values),
                "p10_us": sorted(values)[9],
                "p90_us": sorted(values)[89],
                "samples_us": values,
            }
        )
    return result


@pytest.mark.parametrize("qlen", [512, 2048, 4096, 8192])
@pytest.mark.parametrize("distribution", ["compact", "scattered"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_minimax_m3_sparse_attn(qlen, distribution, graph):
    if bench_cfg.Config.bench_level.value == "core" and qlen not in (512, 2048):
        pytest.skip("non-core shape")
    try:
        from vllm.models.minimax_m3.common.ops.sparse_attn import (
            minimax_m3_sparse_attn as native,
        )
    except ImportError:
        pytest.skip("native comparison requires the vLLM 0.28 MiniMax M3 operator")
    capability = torch.cuda.get_device_capability()
    dtype = torch.float8_e4m3fn if capability >= (8, 9) else torch.bfloat16
    args = make_inputs(qlens=(qlen,), dtype=dtype, mode=distribution, context=32768)
    out_native, out_candidate = torch.empty_like(args[0]), torch.empty_like(args[0])
    fns = [
        lambda: native(*args, output=out_native),
        lambda: flaggems_vllm.minimax_m3_sparse_attn(
            *args, output=out_candidate, qk_dtype="bf16"
        ),
    ]
    for fn in fns:
        fn()
    sample = [0, qlen // 2, qlen - 1]
    torch.testing.assert_close(
        out_candidate.float(), out_native.float(), atol=0.01, rtol=0
    )
    torch.testing.assert_close(
        out_candidate[sample].float(), dense_reference(args, sample), atol=0.01, rtol=0
    )
    times = _pair(fns, graph)
    record = {
        "q": qlen,
        "heads": 8,
        "D": 128,
        "context": 32768,
        "TOPK": 16,
        "page": 128,
        "distribution": distribution,
        "mode": "graph" if graph else "eager",
        "warmup_calls": 30,
        "samples": 100,
        "qk_dtype": "bf16",
        "kv_dtype": str(dtype),
        "device": torch.cuda.get_device_name(),
        "native": times[0],
        "candidate": times[1],
        "speedup": times[0]["median_us"] / times[1]["median_us"],
    }
    print(json.dumps(record), flush=True)
    if bench_cfg.Config.record_json:
        bench_cfg.update_result("minimax_m3_sparse_attn", record)

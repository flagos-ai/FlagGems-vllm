# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Full allocating non-TLE API benchmark, unchanged across baseline/candidate.

Run this same module twice with PYTHONPATH pointing at the two library commits:
  PYTHONPATH=/path/to/native/src python -m benchmark.test_moe_align_small_routes_perf
  PYTHONPATH=src python -m benchmark.test_moe_align_small_routes_perf
No global function or autotuner is replaced to obtain a baseline.
"""
import json
import statistics
import time

import pytest
import torch

from flaggems_vllm import runtime
from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle


def main():
    torch.set_num_threads(2)
    rows = []
    for m in [2, 4, 8, 12, 16, 24, 32, 64, 128, 4096]:
        torch.manual_seed(1291 + m)
        ids = torch.randint(0, 1024, (m, 16), device="cuda", dtype=torch.int32)

        def call():
            return moe_align_block_size_no_tle(ids, 16, 1024)

        for _ in range(20):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        raw = {"wall_ms": [], "graph_ms": []}
        for _ in range(7):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(30):
                call()
            torch.cuda.synchronize()
            raw["wall_ms"].append((time.perf_counter() - start) * 1000 / 30)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            for _ in range(30):
                graph.replay()
            b.record()
            b.synchronize()
            raw["graph_ms"].append(a.elapsed_time(b) / 30)
        row = {
            "M": m,
            "raw": raw,
            "median_ms": {k: statistics.median(v) for k, v in raw.items()},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"complete": True, "rows": rows}), flush=True)


@pytest.mark.skipif(
    runtime.device.vendor_name != "metax", reason="MetaX performance path"
)
def test_complete_operator_benchmark():
    main()


if __name__ == "__main__":
    main()

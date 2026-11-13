# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-call paired benchmark.

Run: PYTHONPATH=src python -m benchmark.test_grouped_topk_materialization_perf
The reference materializes the same sigmoid contract, then calls native FGV
selection. Neither baseline nor candidate omits output allocation.
"""
import json
import statistics
import time

import pytest
import torch

from flaggems_vllm import runtime
from tests.grouped_topk_materialization_utils import assert_exact, candidate, reference


def main():
    if runtime.device.vendor_name != "metax":
        raise SystemExit("This benchmark qualifies the MetaX mode only")
    torch.set_num_threads(2)
    results = []
    for dtype in [torch.float32, torch.bfloat16]:
        for m in [1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 512, 1024, 2048, 4096]:
            torch.manual_seed(9431 + m)
            x = torch.randn(m, 896, device="cuda", dtype=dtype)
            bias = torch.randn(896, device="cuda", dtype=dtype) * 0.1
            funcs = {
                "native_chain": lambda: reference(x, bias),
                "fusion": lambda: candidate(x, bias),
            }
            assert_exact(funcs["fusion"](), funcs["native_chain"]())
            graphs = {}
            for name, fn in funcs.items():
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                graphs[name] = graph
            values = {name: {"wall_ms": [], "graph_ms": []} for name in funcs}
            for trial in range(7):
                for name in (
                    ["native_chain", "fusion"]
                    if trial % 2 == 0
                    else ["fusion", "native_chain"]
                ):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(30):
                        funcs[name]()
                    torch.cuda.synchronize()
                    values[name]["wall_ms"].append(
                        (time.perf_counter() - start) * 1000 / 30
                    )
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    a.record()
                    for _ in range(30):
                        graphs[name].replay()
                    b.record()
                    b.synchronize()
                    values[name]["graph_ms"].append(a.elapsed_time(b) / 30)
            med = {
                n: {k: statistics.median(v) for k, v in d.items()}
                for n, d in values.items()
            }
            row = {"M": m, "dtype": str(dtype), "median_ms": med, "raw": values}
            results.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps({"complete": True, "rows": results}), flush=True)


@pytest.mark.skipif(
    runtime.device.vendor_name != "metax", reason="MetaX performance path"
)
def test_complete_operator_benchmark():
    main()


if __name__ == "__main__":
    main()

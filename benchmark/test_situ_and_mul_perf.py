# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete native SiTU primitive chain versus the fused out API.

Run: PYTHONPATH=src python -m benchmark.test_situ_and_mul_perf
Allocation and FP32/BF16 conversions are included in each eager call.
Graph replay measures all captured kernels; allocation occurs at capture time.
"""
import json
import statistics
import time

import flag_gems
import pytest
import torch

from flaggems_vllm import runtime, situ_and_mul


def native_chain(x):
    gate, up = x.float().chunk(2, -1)
    gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    up = 25.0 * torch.tanh(up / 25.0)
    return (gate * up).to(x.dtype)


def fused(x):
    out = torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)
    situ_and_mul(out, x, 4.0, 25.0)
    return out


def main():
    if runtime.device.vendor_name != "metax":
        raise SystemExit("This implementation is qualified on MetaX")
    torch.set_num_threads(2)
    rows = []
    for m in [1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 512, 1024, 4096]:
        for width in [1536, 8448]:
            torch.manual_seed(891 + m + width)
            x = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
            funcs = {
                "native_chain": lambda: native_chain(x),
                "fusion": lambda: fused(x),
            }
            with flag_gems.use_gems():
                graphs = {}
                for name, fn in funcs.items():
                    for _ in range(20):
                        fn()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fn()
                    graphs[name] = graph
                raw = {name: {"wall_ms": [], "graph_ms": []} for name in funcs}
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
                        raw[name]["wall_ms"].append(
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
                        raw[name]["graph_ms"].append(a.elapsed_time(b) / 30)
            row = {
                "M": m,
                "width": width,
                "raw": raw,
                "median_ms": {
                    n: {k: statistics.median(v) for k, v in d.items()}
                    for n, d in raw.items()
                },
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

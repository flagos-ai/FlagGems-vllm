# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Large router extension: compare complete cast+projection chains.

The prior fp32_router_gemm API rejects M8192. This is a shape extension, not
an infinite speedup over that unsupported call. Compare valid compositions.
"""
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import flaggems_vllm
import torch
import triton


def paired_calls(native, candidate, rounds=7):
    def capture(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [fn() for _ in range(8)]
        graph.replay()
        torch.cuda.synchronize()
        return graph, outputs

    ng, no = capture(native)
    cg, co = capture(candidate)

    def gpu(graph):
        samples = []
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 8)
        return statistics.median(samples)

    def eager(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            output = fn()
        torch.cuda.synchronize()
        _ = output  # Keep the last result alive through synchronization.
        return (time.perf_counter() - start) * 1e6 / 20

    samples = []
    for i in range(rounds):
        row = {}
        order = [("native", native, ng), ("candidate", candidate, cg)]
        if i % 2:
            order.reverse()
        for name, fn, graph in order:
            row[name + "_graph_us"] = gpu(graph)
            row[name + "_eager_us"] = eager(fn)
        samples.append(row)
    med = lambda key: statistics.median(row[key] for row in samples)
    result = {key: med(key) for key in samples[0]}
    result["graph_speedup"] = result["native_graph_us"] / result["candidate_graph_us"]
    result["eager_speedup"] = result["native_eager_us"] / result["candidate_eager_us"]
    result["graph_wins"] = sum(
        row["native_graph_us"] > row["candidate_graph_us"] for row in samples
    )
    result["rounds"] = samples
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-fg-root", type=Path, required=True)
    parser.add_argument("--native-fgv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(560)
    x = (torch.randn(8192, 6144, device="cuda") * 0.2).bfloat16()
    w = torch.randn(128, 6144, device="cuda") * 0.2
    candidate = lambda: flaggems_vllm.fp32_router_gemm(x, w)

    def reference(name, path, digest):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Native reference source differs: " + str(path))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    meta = reference(
        "native_meta_linear",
        args.native_fg_root / "runtime/backend/_metax/ops/linear.py",
        "9838150a35e7f0ef23c7d255bf22e459aac7c1b8eceeecfb00c5b0e630a33efe",
    )
    generic = reference(
        "native_generic_linear",
        args.native_fg_root / "ops/linear.py",
        "564f52ffba1f50da4c47fe07c01b90d63164a326779078f8234571b4e7fbb0b7",
    )
    route = reference(
        "native_router_gemm",
        args.native_fgv_root / "ops/router_gemm.py",
        "dada077f8726339ff85e43053af68bb000e6d600cf5ed30d34c3236f411aa24f",
    )

    refs = [
        ("FlagGems cast+linear", lambda: meta.linear(x.float(), w)),
        ("generic cast+linear", lambda: generic.linear(x.float(), w)),
        ("FGV cast+router_gemm", lambda: route.router_gemm(x.float(), w)),
        ("Torch cast+linear", lambda: torch.nn.functional.linear(x.float(), w)),
    ]
    expected = torch.nn.functional.linear(x.cpu().float(), w.cpu())
    rows = []
    for name, fn in refs:
        for actual in (candidate().cpu(), fn().cpu()):
            delta = actual - expected
            assert delta.norm() / expected.norm().clamp_min(1e-20) <= 2e-6
            assert delta.abs().max() / expected.abs().max().clamp_min(1e-20) <= 5e-6
        row = dict(comparator=name, **paired_calls(fn, candidate))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "rounds"}), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                rows=rows,
                shape=[8192, 128, 6144],
                input_dtype="bfloat16",
                weight_output_dtype="float32",
                torch=torch.__version__,
                triton=triton.__file__,
                scope="Full cast/layout/allocation boundaries; graph and synchronized eager reported separately.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

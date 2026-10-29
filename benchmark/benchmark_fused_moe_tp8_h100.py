# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Correctness-gated, paired complete-chain MiniMax TP8 MoE benchmark.

This is a standalone opt-in benchmark, separate from the generic pytest shapes.
Importing it does not import Torch/Triton or initialize CUDA. See docs/benchmarks/
minimax_tp8_h100_moe.md for scope, recorded precursor results and commands.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src/flaggems_vllm/ops/fused_moe.py"
BLACKLIST = [
    "topk",
    "_scaled_dot_product_attention_math",
    "maximum",
    "true_divide",
    "scatter_",
]
DEFAULT_MS = "1,8,16,24,32,40,48,56,64,128,256,512,1024,2048,4096,8192"


def load_variants(comparison):
    variants = []
    for index in range(2):
        name = f"_minimax_moe_bench_{index}"
        spec = importlib.util.spec_from_file_location(name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module._W8A8_INT32_DOWN = False
        module._W8A8_MOE_TUNING = index == 1 or comparison != "family"
        module._W8A8_QUANT_FUSION_MODE = (
            "quant_oai" if index == 1 or comparison == "small-decode" else "off"
        )
        if comparison == "small-decode" and index == 0:
            get_config = module._get_w8a8_tp8_h100_config

            def without_swap(tokens, stage):
                config = get_config(tokens, stage)
                if config is not None:
                    config.pop("SWAP_AB", None)
                return config

            module._get_w8a8_tp8_h100_config = without_swap
        variants.append(module)
    return variants


def routing(torch, m, route, shift=0):
    positions = torch.arange(m * 4, device="cuda").view(m, 4)
    ids = (positions + shift) % 128
    if route == "skew80_hot16":
        hot = ((torch.arange(m, device="cuda") + shift) % 10) < 8
        ids = torch.where(hot[:, None], (positions + shift) % 16, ids)
    return ids.int()


def same_bytes(torch, actual, expected):
    return torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def stats(samples):
    ordered = sorted(samples)
    return dict(
        median_us=statistics.median(ordered),
        min_us=ordered[0],
        max_us=ordered[-1],
        p90_us=ordered[int(0.9 * (len(ordered) - 1))],
        samples=len(ordered),
    )


def paired(torch, calls, warmup, repetitions):
    for _ in range(warmup):
        for call in calls:
            call()
    torch.cuda.synchronize()
    events = [
        [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(repetitions)
        ]
        for _ in calls
    ]
    host = [[] for _ in calls]
    for iteration in range(repetitions):
        if iteration % 8 == 0:
            torch.cuda.synchronize()
        order = range(len(calls)) if iteration % 2 == 0 else reversed(range(len(calls)))
        for index in order:
            start, end = events[index][iteration]
            start.record()
            begin = time.perf_counter_ns()
            calls[index]()
            host[index].append((time.perf_counter_ns() - begin) / 1000)
            end.record()
    torch.cuda.synchronize()
    device = [[start.elapsed_time(end) * 1000 for start, end in row] for row in events]
    return device, host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("verify", "bench"), default="verify")
    parser.add_argument("--verified-by", type=Path)
    parser.add_argument(
        "--comparison", choices=("family", "fusion", "small-decode"), default="family"
    )
    parser.add_argument("--ms", default=DEFAULT_MS)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing evidence file")
    if args.warmup < 1 or args.rep < 1:
        parser.error("warmup and rep must be positive")
    ms = list(map(int, args.ms.split(",")))
    if not ms or any(m < 1 or m > 8192 for m in ms):
        parser.error("Supported benchmark range is 1..8192 tokens")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if args.stage == "bench":
        if args.verified_by is None:
            parser.error("Run --stage verify before timing")
        rows = [json.loads(line) for line in args.verified_by.read_text().splitlines()]
        assert rows[-1]["status"] == "pass" and rows[-1]["kind"] == "summary"
        assert rows[0]["source_sha256"] == source_hash
        assert rows[0]["comparison"] == args.comparison
        verified = {(row["M"], row["route"]) for row in rows if row["kind"] == "case"}
        assert {
            (m, route) for m in ms for route in ("uniform", "skew80_hot16")
        } <= verified

    # Deliberately deferred until after --help/arguments/evidence checks.
    import flag_gems
    import torch
    import triton

    assert torch.cuda.get_device_name() == "NVIDIA H100 80GB HBM3"
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(20261021)
    torch.backends.cuda.matmul.allow_tf32 = False
    modules = load_variants(args.comparison)
    w1 = torch.randint(-32, 32, (128, 768, 6144), device="cuda", dtype=torch.int8)
    w2 = torch.randint(-32, 32, (128, 6144, 384), device="cuda", dtype=torch.int8)
    s1 = torch.rand((128, 768), device="cuda") * 0.001 + 0.001
    s2 = torch.rand((128, 6144), device="cuda") * 0.001 + 0.001
    kwargs = dict(
        activation="swigluoai_uninterleave",
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=s1,
        w2_scale=s2,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    with args.output.open("x") as handle:

        def record(row):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(
                json.dumps({k: v for k, v in row.items() if not k.startswith("raw_")}),
                flush=True,
            )

        record(
            dict(
                kind="metadata",
                source_sha256=source_hash,
                comparison=args.comparison,
                torch=str(torch.__version__),
                triton=triton.__version__,
                device=torch.cuda.get_device_name(),
                shape=dict(E=128, H=6144, I=384, topk=4),
                M=ms,
                warmup=args.warmup,
                rep=args.rep,
                blacklist=BLACKLIST,
                scope=(
                    "TP8 rank-local quant1/align/GEMM1/OAI/quant2/GEMM2/sum; "
                    "excludes router/shared expert/collectives"
                ),
                graph_measurement="device replay",
                eager_measurement="CUDA event elapsed and independent enqueue wall time; no subtraction",
                oracle="existing library family-off kernel, bitwise; not independent plain PyTorch",
            )
        )
        with flag_gems.use_gems(exclude=BLACKLIST):
            for module in modules:
                assert (
                    module._int8_quant_dispatch_policy()
                    == module._Int8QuantRecipe.FG_NATIVE_DIV
                )
            for m in ms:
                for route in ("uniform", "skew80_hot16"):
                    x = torch.randn((m, 6144), device="cuda", dtype=torch.bfloat16)
                    ids = routing(torch, m, route)
                    weights = torch.rand((m, 4), device="cuda")
                    weights /= weights.sum(-1, keepdim=True)
                    calls = [
                        (
                            lambda module=module: module.fused_experts_impl(
                                x, w1, w2, weights, ids, **kwargs
                            )
                        )
                        for module in modules
                    ]
                    baseline, candidate = [call() for call in calls]
                    assert same_bytes(torch, candidate, baseline), (
                        m,
                        route,
                        "eager mismatch",
                    )
                    for _ in range(args.warmup):
                        for call in calls:
                            call()
                    torch.cuda.synchronize()
                    graphs, outputs = [], []
                    for call in calls:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = call()
                        graph.replay()
                        graphs.append(graph)
                        outputs.append(output)
                    for mutation in range(4):
                        x.normal_()
                        if mutation % 2:
                            x[0].zero_()
                        ids.copy_(routing(torch, m, route, shift=mutation + 3))
                        weights.uniform_(0.01, 1)
                        weights.div_(weights.sum(-1, keepdim=True))
                        s1[0, 0].add_(1e-7)
                        s2[0, 0].add_(1e-7)
                        expected = calls[0]()
                        for graph, output in zip(graphs, outputs):
                            graph.replay()
                            assert same_bytes(torch, output, expected), (
                                m,
                                route,
                                "graph mismatch",
                                mutation,
                            )
                        preserved = [output.clone() for output in outputs]
                        calls[-1]()
                        assert all(
                            same_bytes(torch, a, b) for a, b in zip(outputs, preserved)
                        )
                    row = dict(
                        kind="case",
                        M=m,
                        route=route,
                        status="pass",
                        graph_mutations=4,
                        workspace_independence=True,
                    )
                    if args.stage == "bench":
                        for label, functions in (
                            ("eager", calls),
                            ("graph", [graph.replay for graph in graphs]),
                        ):
                            device, host = paired(
                                torch, functions, args.warmup, args.rep
                            )
                            row[label] = dict(
                                baseline=stats(device[0]), candidate=stats(device[1])
                            )
                            row[label + "_host_enqueue"] = dict(
                                baseline=stats(host[0]), candidate=stats(host[1])
                            )
                            record(
                                dict(
                                    kind="samples",
                                    M=m,
                                    route=route,
                                    timing=label,
                                    raw_device_us=device,
                                    raw_enqueue_us=host,
                                )
                            )
                    record(row)
                    del graphs, outputs, calls, baseline, candidate, preserved, expected
        record(dict(kind="summary", status="pass", source_sha256=source_hash))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Microbenchmark the opt-in EP decode path with caller-owned workspaces."""

from __future__ import annotations

import argparse
import json
import statistics
from types import SimpleNamespace

import pytest
import torch
import triton

from flaggems_vllm.ops.fused_moe import fused_experts_impl


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 64, 96, 128])
    parser.add_argument("--intermediate", type=int, nargs="+", default=[1280, 2048])
    parser.add_argument("--ep-rank", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            output = fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def _time_one_graph(graph, replays):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / replays)


def _time_graph_pair(baseline_graph, optimized_graph, rounds, replays=500):
    for _ in range(20):
        baseline_graph.replay()
        optimized_graph.replay()
    torch.cuda.synchronize()
    samples = {"baseline": [], "optimized": []}
    graphs = {"baseline": baseline_graph, "optimized": optimized_graph}
    for round_index in range(rounds):
        order = ("baseline", "optimized")
        if round_index % 2:
            order = tuple(reversed(order))
        for name in order:
            samples[name].append(_time_one_graph(graphs[name], replays))
    return samples


def _time_eager_pair(baseline, optimized, args):
    samples = {"baseline": [], "optimized": []}
    functions = {"baseline": baseline, "optimized": optimized}
    for round_index in range(args.rounds):
        order = ("baseline", "optimized")
        if round_index % 2:
            order = tuple(reversed(order))
        for name in order:
            samples[name].append(
                float(
                    triton.testing.do_bench(
                        functions[name],
                        warmup=args.warmup,
                        rep=args.rep,
                        return_mode="median",
                    )
                )
            )
    return samples


def _route_summary(topk_ids, expert_map):
    local_experts = expert_map[topk_ids.to(torch.int64)]
    local_counts = ((local_experts >= 0) & (local_experts < 18)).sum(dim=1)
    histogram = torch.bincount(local_counts, minlength=9).cpu().tolist()
    return int(local_counts.sum().item()), [int(value) for value in histogram]


def _verify_clamp_is_inactive(hidden, w1, topk_ids, expert_map):
    """Prove that the unclamped baseline has the same semantics for this fixture."""
    local_experts = expert_map[topk_ids.to(torch.int64)]
    gate_max = float("-inf")
    up_abs_max = 0.0
    for local_expert in range(18):
        positions = torch.nonzero(local_experts == local_expert, as_tuple=False)
        if positions.numel() == 0:
            continue
        token_indices = positions[:, 0].unique()
        projection = torch.matmul(
            hidden[token_indices].float(),
            w1[local_expert].transpose(0, 1).float(),
        ).to(torch.bfloat16)
        gate, up = projection.float().chunk(2, dim=-1)
        gate_max = max(gate_max, float(gate.max().item()))
        up_abs_max = max(up_abs_max, float(up.abs().max().item()))

    # Keep a full unit of margin for the different legal FP32 reduction trees.
    if gate_max >= 9.0 or up_abs_max >= 9.0:
        raise RuntimeError(
            "benchmark fixture is too close to clamp=10 for a same-semantics A/B; "
            f"gate_max={gate_max}, up_abs_max={up_abs_max}"
        )
    return gate_max if gate_max != float("-inf") else None, up_abs_max


def _run_shape(num_tokens, intermediate_size, args):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden = torch.randn((num_tokens, 4096), device=device, dtype=dtype)
    w1 = torch.empty(
        (18, 2 * intermediate_size, 4096), device=device, dtype=dtype
    ).normal_(std=4096**-0.5)
    w2 = torch.empty((18, 4096, intermediate_size), device=device, dtype=dtype).normal_(
        std=intermediate_size**-0.5
    )

    logits = torch.randn((num_tokens, 288), device=device)
    topk_weights, topk_ids = torch.topk(torch.sigmoid(logits), 8, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids.to(torch.int32)
    expert_map = torch.full((288,), -1, device=device, dtype=torch.int32)
    local_begin = args.ep_rank * 18
    expert_map[local_begin : local_begin + 18] = torch.arange(
        18, device=device, dtype=torch.int32
    )
    local_route_count, local_route_histogram = _route_summary(topk_ids, expert_map)
    reference_gate_max, reference_up_abs_max = _verify_clamp_is_inactive(
        hidden, w1, topk_ids, expert_map
    )
    optimized_cache13 = torch.empty(
        num_tokens * 8 * 4096,
        device=device,
        dtype=dtype,
    )
    optimized_cache2 = torch.empty(
        max(num_tokens * 8 * intermediate_size, num_tokens * 4096),
        device=device,
        dtype=dtype,
    )
    optimized_output = optimized_cache2[: num_tokens * 4096].view_as(hidden)

    def baseline():
        return fused_experts_impl(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=288,
            expert_map=expert_map,
        )

    def optimized():
        return fused_experts_impl(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=288,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            enable_ep_decode_optimization=True,
            output=optimized_output,
            intermediate_cache13=optimized_cache13,
            intermediate_cache2=optimized_cache2,
        )

    baseline_output = baseline().clone()
    optimized_output = optimized().clone()
    torch.cuda.synchronize()
    try:
        torch.testing.assert_close(
            optimized_output,
            baseline_output,
            rtol=3e-2,
            atol=2e-2,
        )
    except AssertionError as error:
        max_abs = float(
            (optimized_output.float() - baseline_output.float()).abs().max().item()
        )
        raise RuntimeError(
            "optimized and current-default outputs are not comparable; "
            f"max_abs={max_abs}"
        ) from error

    eager_samples = _time_eager_pair(baseline, optimized, args)
    baseline_graph, baseline_graph_output = _capture(baseline)
    optimized_graph, optimized_graph_output = _capture(optimized)
    baseline_graph.replay()
    optimized_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        baseline_graph_output,
        baseline_output,
        rtol=3e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        optimized_graph_output,
        optimized_output,
        rtol=3e-2,
        atol=2e-2,
    )
    graph_samples = _time_graph_pair(
        baseline_graph,
        optimized_graph,
        args.rounds,
    )
    baseline_samples = eager_samples["baseline"]
    optimized_samples = eager_samples["optimized"]
    baseline_graph_samples = graph_samples["baseline"]
    optimized_graph_samples = graph_samples["optimized"]
    baseline_ms = statistics.median(baseline_samples)
    optimized_ms = statistics.median(optimized_samples)
    baseline_graph_ms = statistics.median(baseline_graph_samples)
    optimized_graph_ms = statistics.median(optimized_graph_samples)
    return {
        "M": num_tokens,
        "global_E": 288,
        "local_E": 18,
        "H": 4096,
        "I": intermediate_size,
        "topk": 8,
        "dtype": "bfloat16",
        "optimized_workspace_mode": "caller_owned_output_alias_cache2",
        "local_route_count": local_route_count,
        "local_route_count_histogram": local_route_histogram,
        "reference_gate_max": reference_gate_max,
        "reference_up_abs_max": reference_up_abs_max,
        "baseline_eager_ms": baseline_ms,
        "optimized_eager_ms": optimized_ms,
        "eager_speedup_x": baseline_ms / optimized_ms,
        "eager_latency_reduction_percent": (baseline_ms - optimized_ms)
        / baseline_ms
        * 100.0,
        "baseline_eager_samples_ms": baseline_samples,
        "optimized_eager_samples_ms": optimized_samples,
        "baseline_cuda_graph_ms": baseline_graph_ms,
        "optimized_cuda_graph_ms": optimized_graph_ms,
        "cuda_graph_speedup_x": baseline_graph_ms / optimized_graph_ms,
        "cuda_graph_latency_reduction_percent": (
            (baseline_graph_ms - optimized_graph_ms) / baseline_graph_ms * 100.0
        ),
        "baseline_cuda_graph_samples_ms": baseline_graph_samples,
        "optimized_cuda_graph_samples_ms": optimized_graph_samples,
        "max_abs_error_vs_default": float(
            (optimized_output.float() - baseline_output.float()).abs().max().item()
        ),
        "optimized_eager_graph_bitwise_equal": bool(
            torch.equal(optimized_output, optimized_graph_output)
        ),
    }


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_name = torch.cuda.get_device_name()
    if "H20" not in device_name.replace(" ", "_").split("_"):
        raise RuntimeError("this fixed-schedule benchmark requires NVIDIA H20")
    if not 0 <= args.ep_rank < 16:
        raise ValueError("ep-rank must be in [0, 16)")
    if any(tokens < 1 or tokens > 128 for tokens in args.tokens):
        raise ValueError("tokens must be in [1, 128]")
    if any(size not in (1280, 2048) for size in args.intermediate):
        raise ValueError("intermediate sizes must be 1280 or 2048")

    torch.manual_seed(args.seed)
    results = [
        _run_shape(tokens, intermediate, args)
        for intermediate in args.intermediate
        for tokens in args.tokens
    ]
    print(json.dumps({"device": device_name, "results": results}, indent=2))


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "H20" not in torch.cuda.get_device_name().replace(" ", "_").split("_"),
    reason="exact optimized path requires NVIDIA H20",
)
def test_fused_moe_ep_decode_benchmark(request):
    """Compare the opt-in path with the repository's current default path."""
    level = request.config.getoption("--level")
    tokens = [1, 96, 128] if level == "core" else [1, 32, 64, 96, 128]
    args = SimpleNamespace(
        ep_rank=7,
        warmup=int(request.config.getoption("--warmup")),
        rep=int(request.config.getoption("--iter")),
        rounds=3,
    )
    torch.manual_seed(20260923)
    results = [
        _run_shape(num_tokens, intermediate_size, args)
        for intermediate_size in (1280, 2048)
        for num_tokens in tokens
    ]
    print(
        json.dumps(
            {"device": torch.cuda.get_device_name(), "results": results}, indent=2
        )
    )


if __name__ == "__main__":
    main()

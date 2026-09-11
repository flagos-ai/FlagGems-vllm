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

"""CUDA Graph A/B benchmark for the complete mHC post-to-pre path.

The candidate path is implemented entirely with Triton.  The vLLM public
TileLang/DeepGEMM operator is imported only as the frozen external baseline.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from flaggems_vllm.ops.mhc.mhc_fused_post_pre import mhc_fused_post_pre

HIDDEN_SIZE = 4096
HC_MULT = 4
MIX_COUNT = 24
REQUIRED_SPEEDUP = 1.2


def _register_vllm_baseline() -> None:
    # Deliberately lazy: importing the external baseline may compile or load
    # its own backend, and must never become a candidate-path dependency.
    import vllm.model_executor.kernels.mhc  # noqa: F401


@dataclass
class Inputs:
    x: torch.Tensor
    residual: torch.Tensor
    post_mix: torch.Tensor
    comb_mix: torch.Tensor
    fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    norm_weight: torch.Tensor


def _parse_tokens(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _make_inputs(num_tokens: int, seed: int) -> Inputs:
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn((num_tokens, HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
    residual = torch.randn(
        (num_tokens, HC_MULT, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    )
    post_mix = 2.0 * torch.sigmoid(
        torch.randn((num_tokens, HC_MULT), device=device, dtype=torch.float32) * 0.1
    )
    comb_mix = torch.softmax(
        torch.randn((num_tokens, HC_MULT, HC_MULT), device=device, dtype=torch.float32),
        dim=-1,
    )
    for _ in range(3):
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + 1e-6)
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + 1e-6)
    fn = (
        torch.randn(
            (MIX_COUNT, HC_MULT * HIDDEN_SIZE),
            device=device,
            dtype=torch.float32,
        )
        * 1e-4
    )
    hc_scale = torch.randn((3,), device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn((MIX_COUNT,), device=device, dtype=torch.float32) * 0.1
    norm_weight = (
        1.0 + torch.randn((HIDDEN_SIZE,), device=device, dtype=torch.float32) * 0.01
    ).to(torch.bfloat16)
    return Inputs(
        x=x,
        residual=residual,
        post_mix=post_mix.unsqueeze(-1),
        comb_mix=comb_mix,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        norm_weight=norm_weight,
    )


def _baseline(inputs: Inputs) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.ops.vllm.mhc_fused_post_pre_tilelang(
            inputs.x,
            inputs.residual,
            inputs.post_mix,
            inputs.comb_mix,
            inputs.fn,
            inputs.hc_scale,
            inputs.hc_base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            n_splits=1,
            tile_n=1,
            norm_weight=inputs.norm_weight,
            norm_eps=1e-6,
        )
    )


def _candidate(inputs: Inputs) -> tuple[torch.Tensor, ...]:
    return mhc_fused_post_pre(
        inputs.x,
        inputs.residual,
        inputs.post_mix,
        inputs.comb_mix,
        inputs.fn,
        inputs.hc_scale,
        inputs.hc_base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        n_splits=1,
        tile_n=1,
        norm_weight=inputs.norm_weight,
        norm_eps=1e-6,
    )


def _correctness(
    candidate: tuple[torch.Tensor, ...], baseline: tuple[torch.Tensor, ...]
) -> dict[str, Any]:
    names = ("residual", "post_mix", "comb_mix", "layer_input")
    tolerances = ((1e-2, 1e-2), (1e-3, 1e-3), (1e-3, 1e-3), (2e-2, 2e-2))
    outputs: dict[str, Any] = {}
    passed = True
    for name, actual, expected, (rtol, atol) in zip(
        names, candidate, baseline, tolerances
    ):
        max_abs = (actual.float() - expected.float()).abs().max().item()
        try:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            output_passed = True
        except AssertionError as error:
            output_passed = False
            outputs[name] = {
                "passed": False,
                "max_abs": max_abs,
                "error": str(error).replace("\n", " ")[:1000],
            }
        else:
            outputs[name] = {"passed": True, "max_abs": max_abs}
        passed &= output_passed
    return {"passed": passed, "outputs": outputs}


def _capture(
    fn: Callable[[], tuple[torch.Tensor, ...]], copies: int, warmup: int
) -> torch.cuda.CUDAGraph:
    holder: list[tuple[torch.Tensor, ...]] = []
    for _ in range(warmup):
        holder[:] = [fn()]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(copies):
            holder[:] = [fn()]
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, copies: int, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (copies * replays)


def _abba(
    baseline: torch.cuda.CUDAGraph,
    candidate: torch.cuda.CUDAGraph,
    copies: int,
    replays: int,
    rounds: int,
) -> dict[str, Any]:
    samples: dict[str, list[float]] = {"baseline_us": [], "candidate_us": []}
    for round_index in range(rounds):
        if round_index % 2 == 0:
            order = (
                ("baseline_us", baseline),
                ("candidate_us", candidate),
                ("candidate_us", candidate),
                ("baseline_us", baseline),
            )
        else:
            order = (
                ("candidate_us", candidate),
                ("baseline_us", baseline),
                ("baseline_us", baseline),
                ("candidate_us", candidate),
            )
        for name, graph in order:
            samples[name].append(_time_graph(graph, copies, replays))
    baseline_median = statistics.median(samples["baseline_us"])
    candidate_median = statistics.median(samples["candidate_us"])
    speedup = baseline_median / candidate_median
    return {
        **samples,
        "baseline_median_us": baseline_median,
        "candidate_median_us": candidate_median,
        "speedup": speedup,
        "latency_reduction_pct": 100.0 * (1.0 - candidate_median / baseline_median),
    }


def _run_case(
    num_tokens: int,
    seed: int,
    copies: int,
    replays: int,
    rounds: int,
    warmup: int,
) -> dict[str, Any]:
    inputs = _make_inputs(num_tokens, seed)
    # Warm the candidate first so one-time Triton weight packing is excluded
    # from the steady-state decode measurement, just like persistent weights in
    # a serving process.
    candidate_output = _candidate(inputs)
    baseline_output = _baseline(inputs)
    torch.cuda.synchronize()
    correctness = _correctness(candidate_output, baseline_output)
    if not correctness["passed"]:
        return {"correctness": correctness, "timing_skipped": True}

    baseline_graph = _capture(lambda: _baseline(inputs), copies, warmup)
    candidate_graph = _capture(lambda: _candidate(inputs), copies, warmup)
    return {
        "correctness": correctness,
        "timing": _abba(
            baseline_graph,
            candidate_graph,
            copies,
            replays,
            rounds,
        ),
    }


@pytest.mark.mhc_fused_post_pre
def test_mhc_fused_post_pre_performance() -> None:
    """Emit the active-set A/B result through the repository pytest lane."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    _register_vllm_baseline()
    cases = {
        str(num_tokens): _run_case(
            num_tokens,
            seed=42,
            copies=8,
            replays=100,
            rounds=5,
            warmup=3,
        )
        for num_tokens in (64, 96, 128)
    }
    print(json.dumps(cases, indent=2))
    assert all(case["correctness"]["passed"] for case in cases.values())
    assert all(case["timing"]["speedup"] >= REQUIRED_SPEEDUP for case in cases.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=_parse_tokens, default=[64, 96, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copies", type=int, default=16)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--required-speedup", type=float, default=REQUIRED_SPEEDUP)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    _register_vllm_baseline()
    results: dict[str, Any] = {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "candidate_gpu_implementation": "Triton only",
        "baseline": "vLLM public mhc_fused_post_pre_tilelang",
        "shape_contract": "H=4096, HC=4, BF16 activations, FP32 mixes",
        "sinkhorn_repeat": 20,
        "cases": {},
    }
    failed = False
    for num_tokens in args.tokens:
        case = _run_case(
            num_tokens,
            args.seed,
            args.copies,
            args.replays,
            args.rounds,
            args.warmup,
        )
        results["cases"][str(num_tokens)] = case
        failed |= not case["correctness"]["passed"]
        if args.required_speedup and "timing" in case:
            failed |= case["timing"]["speedup"] < args.required_speedup

    print(json.dumps(results, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

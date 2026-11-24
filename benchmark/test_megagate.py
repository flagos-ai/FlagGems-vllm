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

"""MegaGate performance tests."""

import statistics
from contextlib import contextmanager
from dataclasses import asdict

import pytest
import torch

from tests.test_megagate import (
    HYGON_CASES,
    HYGON_MEGAGATE_AVAILABLE,
    capture_hygon_graph,
    make_hygon_inputs,
    prepare_hygon_megagate,
)

from .conftest import emit_record_logger, update_result
from .consts import BenchmarkMetrics, BenchmarkResult


@contextmanager
def _hygon_vllm_baseline(monkeypatch, rendezvous):
    monkeypatch.setenv("VLLM_PLUGINS", "")
    vllm = pytest.importorskip("vllm")
    import vllm.platforms as platforms
    from vllm.platforms.rocm import RocmPlatform

    # DTK's vLLM can lack amdsmi-based ROCm detection.
    if not platforms.current_platform.is_rocm():
        monkeypatch.setattr(platforms, "current_platform", RocmPlatform())

    router_factory = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.router.router_factory"
    )
    create_fused_moe_router = router_factory.create_fused_moe_router

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.linear import ReplicatedLinear

    reported = set()

    def build(profile, x, weight, bias):
        layer_kwargs = {}
        if profile.routing == "softmax":
            layer_cls = ReplicatedLinear
        else:
            layer_cls = GateLinear
            layer_kwargs["out_dtype"] = (
                torch.float32 if profile.routing == "sqrtsoftplus" else None
            )
        layer = layer_cls(
            profile.hidden,
            profile.experts,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=f"benchmark.megagate.{profile.name}",
            **layer_kwargs,
        ).to(x.device)
        layer.eval()
        layer.weight.copy_(weight)

        grouped = profile.routing == "grouped_sigmoid"
        indices_dtype = (
            torch.int64 if profile.routing == "sqrtsoftplus" else torch.int32
        )
        router = create_fused_moe_router(
            top_k=profile.topk,
            global_num_experts=profile.experts,
            use_grouped_topk=grouped,
            num_expert_group=8 if grouped else None,
            topk_group=4 if grouped else None,
            scoring_func="sigmoid" if grouped else profile.routing,
            routed_scaling_factor=profile.scale,
            e_score_correction_bias=bias,
        )
        if profile.name not in reported:
            print(
                f"vllm={vllm.__version__} profile={profile.name} "
                f"gate={type(layer).__name__} router={type(router).__name__} "
                f"aiter_fmoe={bool(rocm_aiter_ops.is_fused_moe_enabled())}",
                flush=True,
            )
            reported.add(profile.name)

        def run():
            logits, _ = layer(x)
            return router.select_experts(x, logits, topk_indices_dtype=indices_dtype)

        return run

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=rendezvous.as_uri(),
            local_rank=0,
            backend="gloo",
        )
        try:
            initialize_model_parallel(1, 1)
            yield build
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _timed_graph(graph, calls):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls


def _compare_graphs(baseline, candidate, calls):
    samples = [[], []]
    for round_id in range(5):
        order = (0, 1, 1, 0) if round_id % 2 == 0 else (1, 0, 0, 1)
        for index in order:
            graph = (baseline, candidate)[index]
            samples[index].append(_timed_graph(graph, calls))
    return tuple(statistics.median(values) for values in samples)


@pytest.mark.megagate
@pytest.mark.skipif(not HYGON_MEGAGATE_AVAILABLE, reason="Requires Hygon gfx936")
@torch.inference_mode()
def test_megagate_hygon(monkeypatch, tmp_path):
    calls = 40
    metrics = []
    with _hygon_vllm_baseline(monkeypatch, tmp_path / "rendezvous") as build_baseline:
        for profile, tokens in HYGON_CASES:
            x, weight, bias = make_hygon_inputs(profile, tokens)
            run_vllm = build_baseline(profile, x, weight, bias)
            run_megagate = prepare_hygon_megagate(profile, x, weight, bias)

            baseline_graph, _ = capture_hygon_graph(run_vllm, calls)
            candidate_graph, _ = capture_hygon_graph(run_megagate, calls)
            base_ms, megagate_ms = _compare_graphs(
                baseline_graph, candidate_graph, calls
            )
            metrics.append(
                BenchmarkMetrics(
                    shape_detail=(
                        tokens,
                        profile.hidden,
                        profile.experts,
                        profile.topk,
                    ),
                    latency_base=base_ms,
                    latency=megagate_ms,
                    speedup=base_ms / megagate_ms,
                )
            )

    result = BenchmarkResult(
        level="comprehensive",
        op_name="megagate",
        dtype=str(torch.bfloat16),
        mode="cudagraph",
        result=metrics,
    )
    print(result)
    print(
        f"Mean speedup over {len(metrics)} cases: {statistics.mean(m.speedup for m in metrics):.4f}x"
    )
    update_result("megagate", asdict(result))
    emit_record_logger(result.to_json())

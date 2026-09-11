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

from __future__ import annotations

import json
import os
import statistics

import pytest
import torch
import torch.distributed as dist

import flaggems_vllm
from flaggems_vllm.ops.fused_allreduce_rms_norm import (
    create_fused_allreduce_rms_norm_workspace,
)

from .conftest import Config

try:
    import flashinfer.comm as flashinfer_comm
    from flashinfer.comm.mnnvl import TorchDistBackend

    HAS_FLASHINFER_ALLREDUCE = all(
        hasattr(flashinfer_comm, name)
        for name in ("allreduce_fusion", "create_allreduce_fusion_workspace")
    )
except (ImportError, OSError):
    flashinfer_comm = None
    TorchDistBackend = None
    HAS_FLASHINFER_ALLREDUCE = False


HIDDEN_SIZE = 6144
RMS_EPS = 1.0e-6
SHAPES = (
    ("glm52_c1_target_verify", 6),
    ("glm52_c64_target_verify_max_observed", 138),
)
WORKLOADS = {
    "glm52_c1_target_verify": {"isl": 8000, "osl": 1024, "concurrency": 1},
    "glm52_c64_target_verify_max_observed": {
        "isl": 1024,
        "osl": 128,
        "concurrency": 64,
    },
}
_HOPPER_VLLM_ONESHOT_LIMIT_MIB = {2: 32.0, 4: 2.0, 8: 0.5}
_MNNVL_ONESHOT_LIMIT_BYTES = 1024 * 1024
_VLLM_PDL_ADVANCE_LAUNCH_TOKENS = 16
# PyTorch symmetric peer memory must be initialized before CUDA multicast VMM
# in a shared process.  Keep MNNVL last so both backends can be benchmarked by
# one torchrun invocation without crossing allocator lifetimes in that order.
BACKEND_PAIRS = (("peer", "trtllm"), ("mnnvl", "mnnvl"))
_SUMMARY_ROWS = []


def _format_summary(rows) -> str:
    header = (
        f"{'vLLM / FlagGems':<19}"
        f"{'Strategy':<10}"
        f"{'M':>6}"
        f"{'ISL/OSL/C':>18}"
        f"{'vLLM CUDA (us)':>17}"
        f"{'FlagGems (us)':>17}"
        f"{'Speedup':>10}"
    )
    lines = [
        "\nFused AllReduce + RMSNorm Performance "
        f"(TP={rows[0]['tp']}, H={HIDDEN_SIZE}, BF16, CUDA Graph)",
        header,
        "-" * len(header),
    ]
    for row in rows:
        workload = row["workload"]
        workload_label = (
            f"{workload['isl']}/{workload['osl']}/{workload['concurrency']}"
        )
        backend_label = f"{row['vllm_backend']} / {row['flaggems_backend']}"
        lines.append(
            f"{backend_label:<19}"
            f"{row['strategy']:<10}"
            f"{row['shape'][0]:>6}"
            f"{workload_label:>18}"
            f"{row['vllm_median_us']:>17.3f}"
            f"{row['flaggems_median_us']:>17.3f}"
            f"{row['speedup']:>9.3f}x"
        )
    return "\n".join(lines)


@pytest.fixture(scope="module", autouse=True)
def benchmark_summary():
    yield
    if int(os.environ.get("RANK", "0")) == 0 and _SUMMARY_ROWS:
        print(_format_summary(_SUMMARY_ROWS), flush=True)


def _has_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) in (2, 4, 8)


@pytest.fixture(scope="module")
def distributed_context():
    if not _has_distributed_launch():
        pytest.skip("launch with torchrun and 2, 4, or 8 ranks")
    if not torch.cuda.is_available() or not HAS_FLASHINFER_ALLREDUCE:
        pytest.fail("distributed launch requires CUDA and FlashInfer all-reduce fusion")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.device_count() < world_size:
        pytest.fail(f"distributed launch requires {world_size} visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(local_rank) != (9, 0):
        pytest.fail("distributed launch requires NVIDIA Hopper (SM90)")

    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", init_method="env://", device_id=device)
    rank = dist.get_rank()
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    yield rank, world_size, device, cpu_group
    dist.barrier()
    dist.destroy_process_group(cpu_group)
    if owns_group:
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=BACKEND_PAIRS)
def workspaces(request, distributed_context):
    rank, world_size, _device, cpu_group = distributed_context
    triton_backend, baseline_backend = request.param
    comm_backend = TorchDistBackend(group=cpu_group)
    baseline_workspace = flashinfer_comm.create_allreduce_fusion_workspace(
        backend=baseline_backend,
        world_size=world_size,
        rank=rank,
        max_token_num=max(m for _name, m in SHAPES),
        hidden_dim=HIDDEN_SIZE,
        dtype=torch.bfloat16,
        force_oneshot_support=True,
        comm_backend=comm_backend,
        group=cpu_group,
    )
    triton_workspace = create_fused_allreduce_rms_norm_workspace(
        world_size=world_size,
        rank=rank,
        max_token_num=max(m for _name, m in SHAPES),
        hidden_dim=HIDDEN_SIZE,
        dtype=torch.bfloat16,
        group=dist.group.WORLD,
        backend=triton_backend,
    )
    yield triton_workspace, baseline_workspace, baseline_backend
    triton_workspace.destroy()
    baseline_workspace.destroy()


def _vllm_uses_oneshot(m: int, world_size: int, backend: str) -> bool:
    tensor_size_bytes = m * HIDDEN_SIZE * 2
    if backend == "mnnvl":
        return world_size * tensor_size_bytes <= _MNNVL_ONESHOT_LIMIT_BYTES
    tensor_size_mib = tensor_size_bytes / (1024 * 1024)
    return tensor_size_mib <= _HOPPER_VLLM_ONESHOT_LIMIT_MIB[world_size]


def _make_inputs(m: int, rank: int, device: torch.device):
    rank_generator = torch.Generator(device=device).manual_seed(2026 + rank)
    common_generator = torch.Generator(device=device).manual_seed(2026)
    shape = (m, HIDDEN_SIZE)
    allreduce_input = (
        torch.randn(shape, generator=rank_generator, device=device) * 0.125
    ).to(torch.bfloat16)
    residual_input = (
        torch.randn(shape, generator=common_generator, device=device) * 0.125
    ).to(torch.bfloat16)
    gamma = (
        1.0
        + torch.randn((HIDDEN_SIZE,), generator=common_generator, device=device) * 0.05
    ).to(torch.bfloat16)
    return allreduce_input, residual_input, gamma


def _reference(allreduce_input, residual_input, gamma):
    rank_sum = allreduce_input.float()
    dist.all_reduce(rank_sum, op=dist.ReduceOp.SUM)
    reduced_bf16 = rank_sum.to(torch.bfloat16)
    residual = (reduced_bf16.float() + residual_input.float()).to(torch.bfloat16)
    residual_f32 = residual.float()
    reciprocal_rms = torch.rsqrt(
        residual_f32.square().mean(dim=-1, keepdim=True) + RMS_EPS
    )
    norm = (residual_f32 * reciprocal_rms * gamma.float()).to(torch.bfloat16)
    return residual, norm


def _capture(call, warmup: int, device: torch.device):
    call()
    torch.cuda.synchronize(device)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize(device)
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()
    return graph


def _rank_max_latency_us(graph, iterations: int, device: torch.device) -> float:
    dist.barrier()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    latency = torch.tensor(
        begin.elapsed_time(end) * 1000.0 / iterations,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return float(latency.item())


@pytest.mark.fused_allreduce_rms_norm
@pytest.mark.parametrize(("case_name", "m"), SHAPES)
def test_fused_allreduce_rms_norm_benchmark(
    case_name,
    m,
    distributed_context,
    workspaces,
):
    rank, world_size, device, _cpu_group = distributed_context
    triton_workspace, baseline_workspace, baseline_backend = workspaces
    allreduce_input, residual_input, gamma = _make_inputs(m, rank, device)
    expected_residual, expected_norm = _reference(
        allreduce_input, residual_input, gamma
    )

    triton_input = allreduce_input.clone()
    triton_residual = residual_input.clone()
    baseline_input = allreduce_input.clone()
    baseline_residual = residual_input.clone()
    use_oneshot = _vllm_uses_oneshot(m, world_size, baseline_backend)
    baseline_trigger_completion_at_end = (
        use_oneshot or m > _VLLM_PDL_ADVANCE_LAUNCH_TOKENS
    )

    def call_triton():
        return flaggems_vllm.fused_allreduce_rms_norm(
            triton_input,
            triton_residual,
            gamma,
            RMS_EPS,
            world_size,
            True,
            True,
            triton_workspace.max_token_num,
            flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        )

    def call_baseline():
        return flashinfer_comm.allreduce_fusion(
            input=baseline_input,
            workspace=baseline_workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True,
            trigger_completion_at_end=baseline_trigger_completion_at_end,
            output=None,
            residual_out=baseline_residual,
            norm_out=baseline_input,
            residual_in=baseline_residual,
            rms_gamma=gamma,
            rms_eps=RMS_EPS,
            use_oneshot=use_oneshot,
            fp32_acc=True,
            weight_bias=0.0,
        )

    warmup = max(1, Config.warm_up)
    iterations = max(1, Config.repetition)
    baseline_graph = _capture(call_baseline, warmup, device)
    triton_graph = _capture(call_triton, warmup, device)

    baseline_input.copy_(allreduce_input)
    baseline_residual.copy_(residual_input)
    triton_input.copy_(allreduce_input)
    triton_residual.copy_(residual_input)
    baseline_graph.replay()
    triton_graph.replay()
    torch.cuda.synchronize(device)

    for actual in (baseline_residual, triton_residual):
        torch.testing.assert_close(actual, expected_residual, atol=0.04, rtol=0.04)
    for actual in (baseline_input, triton_input):
        torch.testing.assert_close(actual, expected_norm, atol=0.04, rtol=0.04)

    samples = {"vllm_baseline": [], "flaggems_vllm": []}
    providers = (
        ("vllm_baseline", baseline_graph),
        ("flaggems_vllm", triton_graph),
    )
    for repeat in range(7):
        order = providers if repeat % 2 == 0 else tuple(reversed(providers))
        for provider, graph in order:
            samples[provider].append(_rank_max_latency_us(graph, iterations, device))

    baseline_median = statistics.median(samples["vllm_baseline"])
    triton_median = statistics.median(samples["flaggems_vllm"])
    if rank == 0:
        result = {
            "operator": "fused_allreduce_rms_norm",
            "case": case_name,
            "workload": WORKLOADS[case_name],
            "shape": [m, HIDDEN_SIZE],
            "dtype": "bfloat16",
            "tp": world_size,
            "cuda_graph": True,
            "vllm_implementation": "FlashInfer CUDA",
            "vllm_backend": baseline_backend,
            "flaggems_backend": triton_workspace.backend,
            "strategy": "oneshot" if use_oneshot else "twoshot",
            "vllm_trigger_completion_at_end": (baseline_trigger_completion_at_end),
            "vllm_median_us": baseline_median,
            "flaggems_median_us": triton_median,
            "speedup": baseline_median / triton_median,
            "rank_max_samples_us": samples,
        }
        _SUMMARY_ROWS.append(result)
        print(
            "RESULT_JSON " + json.dumps(result, sort_keys=True),
            flush=True,
        )

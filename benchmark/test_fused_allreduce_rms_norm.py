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

import os
import statistics
from dataclasses import asdict

import pytest
import torch
import torch.distributed as dist
import triton

import flaggems_vllm
from flaggems_vllm.ops.fused_allreduce_rms_norm import (
    create_fused_allreduce_rms_norm_workspace,
)

from . import base, consts
from .conftest import Config, emit_record_logger, update_result

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


def _rank_max_latency_ms(call, device: torch.device) -> float:
    dist.barrier()
    # A collective graph must contain the same number of calls on every rank.
    if Config.mode == consts.BenchMode.CUDAGRAPH:
        latency_ms = base.do_bench_cudagraph(
            call,
            return_mode="mean",
            replay_count=max(1, Config.repetition),
            warmup_replay_count=max(1, Config.warm_up),
            rank_barrier=dist.barrier,
        )
    elif Config.mode == consts.BenchMode.KERNEL:
        latency_ms = triton.testing.do_bench(
            call, warmup=0, rep=0, return_mode="median"
        )
    else:
        raise ValueError(
            "fused_allreduce_rms_norm benchmark supports "
            "--mode kernel or --mode cudagraph"
        )
    latency = torch.tensor(
        latency_ms,
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

    baseline_input.copy_(allreduce_input)
    baseline_residual.copy_(residual_input)
    triton_input.copy_(allreduce_input)
    triton_residual.copy_(residual_input)
    call_baseline()
    call_triton()
    torch.cuda.synchronize(device)

    for actual in (baseline_residual, triton_residual):
        torch.testing.assert_close(actual, expected_residual, atol=0.04, rtol=0.04)
    for actual in (baseline_input, triton_input):
        torch.testing.assert_close(actual, expected_norm, atol=0.04, rtol=0.04)

    samples = {"vllm_baseline": [], "flaggems_vllm": []}
    providers = (
        ("vllm_baseline", call_baseline),
        ("flaggems_vllm", call_triton),
    )
    for repeat in range(7):
        order = providers if repeat % 2 == 0 else tuple(reversed(providers))
        for provider, call in order:
            samples[provider].append(_rank_max_latency_ms(call, device))

    baseline_median = statistics.median(samples["vllm_baseline"])
    triton_median = statistics.median(samples["flaggems_vllm"])
    if rank == 0:
        workload = WORKLOADS[case_name]
        metric = consts.BenchmarkMetrics(
            shape_detail=(
                f"shape=({m}, {HIDDEN_SIZE}), TP={world_size}, "
                f"backend={baseline_backend}/{triton_workspace.backend}, "
                f"strategy={'oneshot' if use_oneshot else 'twoshot'}, "
                f"ISL/OSL/C={workload['isl']}/{workload['osl']}/"
                f"{workload['concurrency']}"
            ),
            latency_base=baseline_median,
            latency=triton_median,
            speedup=baseline_median / triton_median,
        )
        result = consts.BenchmarkResult(
            level=Config.bench_level.value,
            op_name="fused_allreduce_rms_norm",
            dtype=str(torch.bfloat16),
            mode=Config.mode.value,
            result=[metric],
        )
        print(result)
        update_result(result.op_name, asdict(result))
        emit_record_logger(result.to_json())

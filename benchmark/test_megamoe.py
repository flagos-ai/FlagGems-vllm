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

"""Benchmark the multi-rank Hopper MegaMoE operator.

This standalone harness owns distributed/NVSHMEM setup and input prepacking;
the operator itself accepts caller-owned tensors and communication state.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# FlagTree's raw CUDA cache must be private to each worker process. Establish
# this before importing Triton through the benchmark harness or MegaMoE module.
_IS_WORKER = "--worker" in sys.argv
if _IS_WORKER:
    _local_rank = os.environ.get("LOCAL_RANK", "0")
    _cache_root = Path(
        os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache")
    )
    os.environ["TRITON_CACHE_DIR"] = str(
        _cache_root.parent / f"{_cache_root.name}-megamoe-rank-{_local_rank}"
    )

import torch  # noqa: E402

if _IS_WORKER:
    torch.cuda.set_device(int(_local_rank))

import pytest  # noqa: E402

from flaggems_vllm.testing.megamoe import StandaloneNvshmem  # noqa: E402
from flaggems_vllm.testing.megamoe import (  # noqa: E402
    configure_descriptor_allocator,
    interleave_l1_gate_up_rows,
    make_context,
    pool_tokens,
)

from . import base  # noqa: E402
from .conftest import Config  # noqa: E402
from .consts import DEFAULT_WARMUP_TIME  # noqa: E402
from .consts import DEFAULT_ITER_TIME, BenchLevel, BenchMode  # noqa: E402

OP_NAME = "megamoe"
NUM_RANKS = 8
STAGES = 4
CORE_TOKENS = (1, 64, 128, 256, 512, 2048, 16384)
COMPREHENSIVE_TOKENS = (
    1,
    2,
    4,
    8,
    16,
    *range(24, 257, 8),
    *range(272, 513, 16),
    2048,
    16384,
)
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 30
DEFAULT_TIMEOUT_S = 1800
DEFAULT_MAX_RECV = 4096
_RESULT = re.compile(
    r"MEGAMOE_RESULT scope=kernel rank=(\d+) world=(\d+) latency_us=([0-9.]+)"
)


@dataclass(frozen=True)
class _BenchmarkConfig:
    num_ranks: int
    tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    topk: int
    stages: int
    warmup: int
    iterations: int
    timeout: int = DEFAULT_TIMEOUT_S
    max_recv: int = DEFAULT_MAX_RECV


def _make_inputs(config: _BenchmarkConfig, runtime: StandaloneNvshmem):
    rank = runtime.rank
    world_size = runtime.world_size
    device = runtime.device
    experts_per_rank = config.num_experts // world_size
    n_l1_tiles = config.intermediate // 64
    n_l2_tiles = config.hidden // 128
    n_k1_tiles = config.hidden // 128
    n_k2_tiles = config.intermediate // 128
    scale_columns = config.hidden // 128
    fp8 = torch.float8_e4m3fn

    cpu_generator = torch.Generator().manual_seed(1234)
    scores = torch.randn(
        world_size * config.tokens,
        config.num_experts,
        generator=cpu_generator,
    )
    topk_all = torch.topk(
        scores, config.topk, dim=-1, largest=True, sorted=False
    ).indices.reshape(world_size, config.tokens, config.topk)
    x = torch.randn(config.tokens, config.hidden, generator=cpu_generator) * 0.5
    x = x.to(device=device, dtype=fp8)
    x_scales = (
        torch.rand(config.tokens, scale_columns, generator=cpu_generator).mul_(0.5)
        + 0.5
    ).to(device)
    topk_weights = (
        torch.rand(config.tokens, config.topk, generator=cpu_generator).mul_(0.8) + 0.2
    ).to(device)

    gpu_generator = torch.Generator(device=device).manual_seed(9000 + rank)
    w1 = torch.randn(
        experts_per_rank,
        n_l1_tiles * 128,
        config.hidden,
        generator=gpu_generator,
        device=device,
    ).mul_(0.4)
    w1 = interleave_l1_gate_up_rows(w1.to(fp8))
    w1_scales = (
        torch.rand(
            experts_per_rank,
            n_l1_tiles,
            n_k1_tiles,
            generator=gpu_generator,
            device=device,
        ).mul_(0.4)
        + 0.3
    )
    w2 = torch.randn(
        experts_per_rank,
        n_l2_tiles * 128,
        config.intermediate,
        generator=gpu_generator,
        device=device,
    ).mul_(0.4)
    w2 = w2.to(fp8)
    w2_scales = (
        torch.rand(
            experts_per_rank,
            n_l2_tiles,
            n_k2_tiles,
            generator=gpu_generator,
            device=device,
        ).mul_(0.4)
        + 0.3
    )

    return (
        x,
        x_scales,
        topk_all[rank].to(device),
        topk_weights,
        w1,
        w1_scales,
        w2,
        w2_scales,
        topk_all,
    )


def _run_worker(config: _BenchmarkConfig) -> None:
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megamoe import megamoe_fp8

    assert config.num_experts % config.num_ranks == 0
    with StandaloneNvshmem() as runtime:
        assert runtime.world_size == config.num_ranks
        configure_descriptor_allocator(runtime.device)
        inputs = _make_inputs(config, runtime)
        pool_size = pool_tokens(inputs[-1], config)
        context = make_context(config, runtime, pool_size)
        output = torch.empty(
            (config.tokens, config.hidden),
            dtype=torch.bfloat16,
            device=runtime.device,
        )
        operands = (*inputs[:-1], context, output)

        for _ in range(config.warmup):
            megamoe_fp8(*operands, stages=config.stages)
        torch.cuda.synchronize(runtime.device)

        samples = []
        for _ in range(config.iterations):
            runtime.arm_kernel_timing()
            megamoe_fp8(*operands, stages=config.stages)
            samples.append(runtime.finish_kernel_timing())

        latency = sum(samples) / len(samples)
        print(
            f"MEGAMOE_RESULT scope=kernel rank={runtime.rank} "
            f"world={runtime.world_size} "
            f"latency_us={latency:.3f}",
            flush=True,
        )


def _worker_arguments(config: _BenchmarkConfig) -> list[str]:
    arguments = [
        "--worker",
        "--num-ranks",
        str(config.num_ranks),
        "--tokens",
        str(config.tokens),
        "--hidden",
        str(config.hidden),
        "--intermediate",
        str(config.intermediate),
        "--num-experts",
        str(config.num_experts),
        "--topk",
        str(config.topk),
        "--stages",
        str(config.stages),
        "--warmup",
        str(config.warmup),
        "--iterations",
        str(config.iterations),
        "--max-recv",
        str(config.max_recv),
    ]
    return arguments


def _launch_megamoe(config: _BenchmarkConfig) -> float:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(config.num_ranks),
        "--module",
        "benchmark.test_megamoe",
        *_worker_arguments(config),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=config.timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout)
    latencies = {
        int(rank): float(latency) / 1e3
        for rank, world, latency in _RESULT.findall(stdout)
        if int(world) == config.num_ranks
    }
    if set(latencies) != set(range(config.num_ranks)):
        raise RuntimeError(f"missing rank results:\n{stdout}")
    return max(latencies.values())


def _iteration_count(option, configured, default, fallback):
    explicitly_set = any(
        arg == option or arg.startswith(f"{option}=") for arg in sys.argv
    )
    return configured if explicitly_set or configured != default else fallback


class MegaMoEBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency"]
    DEFAULT_DTYPES = [torch.float8_e4m3fn]
    DEFAULT_SHAPE_DESC = "tokens, hidden, intermediate, num_experts, topk, num_ranks"

    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            torch_op=None,
            gems_op=_launch_megamoe,
            dtypes=self.DEFAULT_DTYPES,
        )

    def set_shapes(self, shape_file_path=None):
        tokens = (
            COMPREHENSIVE_TOKENS
            if Config.bench_level is BenchLevel.COMPREHENSIVE
            else CORE_TOKENS
        )
        self.shapes = [(value, 2048, 512, 256, 8, NUM_RANKS) for value in tokens]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        yield from self.shapes

    def get_latency(
        self,
        op,
        tokens,
        hidden,
        intermediate,
        num_experts,
        topk,
        num_ranks,
    ):
        assert Config.mode is BenchMode.KERNEL
        return op(
            _BenchmarkConfig(
                num_ranks=num_ranks,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                num_experts=num_experts,
                topk=topk,
                stages=STAGES,
                warmup=_iteration_count(
                    "--warmup", Config.warm_up, DEFAULT_WARMUP_TIME, DEFAULT_WARMUP
                ),
                iterations=_iteration_count(
                    "--iter", Config.repetition, DEFAULT_ITER_TIME, DEFAULT_ITERS
                ),
            )
        )


@pytest.mark.megamoe
def test_megamoe():
    if Config.query:
        MegaMoEBenchmark().run()
        return
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip(f"requires SM90, got SM{major}0")
    if torch.cuda.device_count() < NUM_RANKS:
        pytest.skip(f"requires {NUM_RANKS} visible GPUs")
    MegaMoEBenchmark().run()


def _parse_worker_config() -> _BenchmarkConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--max-recv", type=int, required=True)
    arguments = vars(parser.parse_args())
    arguments.pop("worker")
    return _BenchmarkConfig(**arguments)


if __name__ == "__main__" and _IS_WORKER:
    _run_worker(_parse_worker_config())

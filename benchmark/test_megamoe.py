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

"""Benchmark the multi-rank MegaMoE operators.

This standalone harness owns distributed/NVSHMEM setup and input prepacking;
the Hopper operator itself accepts caller-owned tensors and communication state.
Ascend measures the complete BF16 call or Graph replay with a synchronized
host clock.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
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

from flaggems_vllm import vendor_name  # noqa: E402
from flaggems_vllm.testing.megamoe import SHAPE_DESC  # noqa: E402
from flaggems_vllm.testing.megamoe import StandaloneNvshmem  # noqa: E402
from flaggems_vllm.testing.megamoe import (  # noqa: E402
    configure_descriptor_allocator,
    interleave_l1_gate_up_rows,
    load_ascend_shapes,
    make_ascend_inputs,
    make_context,
    pool_tokens,
    require_ascend,
    run_ascend_workers,
    run_workers,
    shmem_rendezvous,
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
    stdout = run_workers(
        config.num_ranks,
        "--module",
        "benchmark.test_megamoe",
        *_worker_arguments(config),
        timeout=config.timeout,
    )
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


def _default_shapes(ep):
    tokens = (
        COMPREHENSIVE_TOKENS
        if Config.bench_level is BenchLevel.COMPREHENSIVE
        else CORE_TOKENS
    )
    return [(tokens_per_rank, 2048, 512, 256, 8, ep) for tokens_per_rank in tokens]


class MegaMoEBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency"]
    DEFAULT_DTYPES = [torch.float8_e4m3fn]
    DEFAULT_SHAPE_DESC = SHAPE_DESC

    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            torch_op=None,
            gems_op=_launch_megamoe,
            dtypes=self.DEFAULT_DTYPES,
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = _default_shapes(NUM_RANKS)
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
    if vendor_name == "ascend":
        pytest.skip("requires the NVIDIA Hopper backend")
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


_DEFAULT_SHAPE_FILE = Path(__file__).resolve().with_name("core_shapes.yaml")


def _measure_ascend(ep, shapes, directory):
    config = dict(
        shapes=shapes,
        warmup=Config.warm_up,
        iterations=Config.repetition,
        graph=Config.mode.value == "cudagraph",
    )
    (directory / "config.json").write_text(json.dumps(config))
    run_ascend_workers(ep, __name__, directory)
    ranks = [
        json.loads((directory / f"rank{rank}.json").read_text()) for rank in range(ep)
    ]
    if any(
        len(latencies) != len(shapes)
        or any(not math.isfinite(value) or value <= 0 for value in latencies)
        for latencies in ranks
    ):
        raise RuntimeError("missing or invalid MegaMoE timing results")
    return dict(zip(shapes, map(max, zip(*ranks))))


@torch.inference_mode()
def _run_ascend_worker(directory):
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from flaggems_vllm import megamoe, prepare_megamoe

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    group = dist.group.WORLD
    latencies = []
    try:
        for shape in config["shapes"]:
            tokens, _, _, _, topk, ep = shape
            if ep != dist.get_world_size(group):
                raise ValueError(f"shape {shape} does not match the worker group")
            x, ids, probs, w1, w2 = make_ascend_inputs(shape, seed=0, rank=rank)
            workspace = prepare_megamoe(
                w1,
                w2,
                group=group,
                capacity_tokens=max(1, tokens),
                topk=topk,
                rendezvous=shmem_rendezvous(),
            )

            def launch():
                return megamoe(x, ids, probs, workspace)

            graph = captured = None
            try:
                workspace.compile(x, ids, probs)
                call = launch
                if config["graph"]:
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        captured = launch()
                    call = graph.replay
                for _ in range(config["warmup"]):
                    call()
                torch.npu.synchronize()
                samples = []
                for _ in range(config["iterations"]):
                    dist.barrier(group=group)
                    torch.npu.synchronize()
                    start = time.perf_counter_ns()
                    result = call()
                    torch.npu.synchronize()
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                    del result
                latencies.append(statistics.mean(samples))
            finally:
                if graph is not None:
                    graph.reset()
                del captured
                workspace.close()
    finally:
        dist.destroy_process_group()
    (directory / f"rank{rank}.json").write_text(json.dumps(latencies))


class MegaMoEAscendBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency"]
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = SHAPE_DESC

    def __init__(self, ep, directory):
        super().__init__(OP_NAME, torch_op=None, gems_op=_measure_ascend)
        self.ep = ep
        self.directory = directory
        self.latencies = None

    def init_default_config(self):
        self.set_shapes(Config.shape_file)

    def set_shapes(self, shape_file_path=None):
        self.shape_desc = SHAPE_DESC
        if (
            shape_file_path is None
            or Path(shape_file_path).resolve() == _DEFAULT_SHAPE_FILE
        ):
            self.shapes = _default_shapes(self.ep)
        else:
            self.shapes = [
                shape
                for shape in load_ascend_shapes(shape_file_path)
                if shape[5] == self.ep
            ]
            if not self.shapes:
                pytest.skip(f"no EP{self.ep} shapes in {shape_file_path}")

    def get_input_iter(self, dtype):
        if Config.mode.value not in ("operator", "cudagraph"):
            raise ValueError("MegaMoE requires --mode operator or cudagraph")
        if Config.parallel:
            raise ValueError("MegaMoE runs all ranks together")
        if Config.warm_up < 0 or Config.repetition < 1:
            raise ValueError("MegaMoE requires --warmup >= 0 and --iter > 0")
        yield from self.shapes

    def get_latency(self, op, *shape):
        if self.latencies is None:
            self.latencies = op(self.ep, self.shapes, self.directory)
        return self.latencies[shape]


@pytest.mark.megamoe
@pytest.mark.parametrize("ep", [2, 4, 8], ids=["ep2", "ep4", "ep8"])
def test_megamoe_ascend(tmp_path, ep):
    if vendor_name != "ascend":
        pytest.skip("requires the Ascend backend")
    if not Config.query:
        require_ascend(ep)
    MegaMoEAscendBenchmark(ep, tmp_path).run()


if __name__ == "__main__":
    if _IS_WORKER:
        _run_worker(_parse_worker_config())
    elif "--ascend-worker" in sys.argv:
        _run_ascend_worker(sys.argv[-1])

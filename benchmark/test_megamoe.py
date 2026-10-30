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
import ctypes
import json
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

from . import base  # noqa: E402
from .conftest import Config  # noqa: E402
from .consts import DEFAULT_ITER_TIME, DEFAULT_WARMUP_TIME, BenchMode  # noqa: E402

OP_NAME = "megamoe"
NUM_RANKS = 8
STAGES = 4
DEFAULT_SHAPES = [
    (512, 2048, 512, 256, 8, NUM_RANKS),
    (1024, 2048, 512, 256, 8, NUM_RANKS),
    (2048, 2048, 512, 256, 8, NUM_RANKS),
]
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 30
DEFAULT_TIMEOUT_S = 1800
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
    max_recv: int = 512
    data_dir: Path | None = None


class _StandaloneNvshmem:
    """Own the standalone benchmark's process-level NVSHMEM resources."""

    def __init__(self):
        from triton.experimental.tle.raw.nvshmem.utils import (
            init_nvshmem_by_torch_pg,
            init_torch_distributed,
            load_common_host,
        )

        self.group = init_torch_distributed()
        self.rank = self.group.rank()
        self.world_size = self.group.size()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.common = load_common_host()
        init_nvshmem_by_torch_pg(self.common, self.group)
        # Resolve the standard host APIs through the same loaded object that
        # initialized NVSHMEM. Loading libnvshmem_host a second time can create
        # a separate process-local runtime state on some installations.
        self.library = self.common.library
        self.library.nvshmem_malloc.argtypes = [ctypes.c_size_t]
        self.library.nvshmem_malloc.restype = ctypes.c_void_p
        self.library.nvshmem_free.argtypes = [ctypes.c_void_p]
        self.library.nvshmem_free.restype = None
        self.library.nvshmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.nvshmem_ptr.restype = ctypes.c_void_p
        self.library.nvshmemx_barrier_all_on_stream.argtypes = [ctypes.c_void_p]
        self.library.nvshmemx_barrier_all_on_stream.restype = None

        self._allocations: list[int] = []
        self._closed = False
        self._timing_events = None
        self._timing_barriers = 0

    def malloc(self, nbytes: int) -> int:
        pointer = self.library.nvshmem_malloc(nbytes)
        if not pointer:
            raise MemoryError(f"NVSHMEM allocation failed for {nbytes} bytes")
        address = int(pointer)
        self._allocations.append(address)
        return address

    def peer_addresses(self, pointer: int) -> tuple[int, ...]:
        addresses = []
        for peer in range(self.world_size):
            address = (
                pointer
                if peer == self.rank
                else self.library.nvshmem_ptr(ctypes.c_void_p(pointer), peer)
            )
            if not address:
                raise RuntimeError(f"rank {peer} is not directly addressable")
            addresses.append(int(address))
        return tuple(addresses)

    def peer_table(self, pointer: int) -> torch.Tensor:
        return torch.tensor(
            self.peer_addresses(pointer), dtype=torch.int64, device=self.device
        )

    def barrier_all_on_stream(self, stream: torch.cuda.Stream) -> None:
        # The operator calls this once immediately before and once immediately
        # after the megakernel. Place events inside those barriers so setup and
        # teardown remain outside the kernel-only timing interval.
        if self._timing_events is not None and self._timing_barriers == 1:
            self._timing_events[1].record(stream)
        self.library.nvshmemx_barrier_all_on_stream(ctypes.c_void_p(stream.cuda_stream))
        if self._timing_events is not None and self._timing_barriers == 0:
            self._timing_events[0].record(stream)
        if self._timing_events is not None:
            self._timing_barriers += 1

    def arm_kernel_timing(self) -> None:
        assert self._timing_events is None
        self._timing_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        self._timing_barriers = 0

    def finish_kernel_timing(self) -> float:
        assert self._timing_events is not None and self._timing_barriers == 2
        start, end = self._timing_events
        end.synchronize()
        elapsed_us = start.elapsed_time(end) * 1e3
        self._timing_events = None
        self._timing_barriers = 0
        return elapsed_us

    def close(self) -> None:
        if self._closed:
            return
        stream = torch.cuda.current_stream(self.device)
        self.barrier_all_on_stream(stream)
        stream.synchronize()
        for pointer in reversed(self._allocations):
            self.library.nvshmem_free(ctypes.c_void_p(pointer))
        self._allocations.clear()
        torch.distributed.barrier(group=self.group)
        result = self.common.nvshmem_finalize_from_torch_distributed()
        assert result == 0
        torch.distributed.destroy_process_group()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _interleave_l1_gate_up_rows(weight: torch.Tensor) -> torch.Tensor:
    experts, rows, hidden = weight.shape
    half = rows // 2
    gate = weight[:, :half].reshape(experts, half // 8, 8, hidden)
    up = weight[:, half:].reshape(experts, half // 8, 8, hidden)
    return torch.stack((gate, up), dim=2).reshape(experts, rows, hidden).contiguous()


def _load_qwen_inputs(config: _BenchmarkConfig, rank: int):
    manifest = json.loads((config.data_dir / "manifest.json").read_text())
    files = manifest["files"]
    shared = torch.load(
        config.data_dir / files["shared"]["name"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    local = torch.load(
        config.data_dir / files["ranks"][str(rank)]["name"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    shared = {
        name: shared[name][:, : config.tokens]
        for name in ("input_fp8_bits", "input_scales", "topk_idx", "topk_weights")
    }
    return shared, local


def _make_inputs(config: _BenchmarkConfig, runtime: _StandaloneNvshmem):
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

    if config.data_dir is not None:
        shared, local = _load_qwen_inputs(config, rank)
        topk_all = (
            shared["topk_idx"]
            .reshape(world_size, config.tokens, config.topk)
            .contiguous()
        )
        x = shared["input_fp8_bits"][rank].contiguous().view(fp8).to(device)
        x_scales = shared["input_scales"][rank].contiguous().to(device)
        topk_weights = (
            shared["topk_weights"][rank].reshape(config.tokens, config.topk).to(device)
        )
        w1 = _interleave_l1_gate_up_rows(
            local["l1_weight_fp8_bits"].contiguous().view(fp8)
        ).to(device)
        w1_scales = local["l1_weight_scale_inv"].to(device).contiguous()
        w2 = local["l2_weight_fp8_bits"].contiguous().view(fp8).to(device)
        w2_scales = local["l2_weight_scale_inv"].to(device).contiguous()
    else:
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
            torch.rand(config.tokens, config.topk, generator=cpu_generator).mul_(0.8)
            + 0.2
        ).to(device)

        gpu_generator = torch.Generator(device=device).manual_seed(9000 + rank)
        w1 = torch.randn(
            experts_per_rank,
            n_l1_tiles * 128,
            config.hidden,
            generator=gpu_generator,
            device=device,
        ).mul_(0.4)
        w1 = _interleave_l1_gate_up_rows(w1.to(fp8))
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


def _pool_tokens(topk_all, config: _BenchmarkConfig) -> int:
    experts_per_rank = config.num_experts // config.num_ranks
    counts = [
        [[0] * config.num_ranks for _ in range(experts_per_rank)]
        for _ in range(config.num_ranks)
    ]
    for source_rank, experts in enumerate(
        topk_all.reshape(config.num_ranks, -1).tolist()
    ):
        for expert in experts:
            if expert < 0:
                continue
            counts[expert // experts_per_rank][expert % experts_per_rank][
                source_rank
            ] += 1
    peak_recv = max(
        count
        for rank_counts in counts
        for expert_counts in rank_counts
        for count in expert_counts
    )
    if peak_recv > config.max_recv:
        raise ValueError(
            f"max_recv={config.max_recv} is smaller than the observed "
            f"per-source expert load {peak_recv}"
        )
    blocks_per_rank = [
        sum((sum(expert) + 127) // 128 for expert in rank_counts)
        for rank_counts in counts
    ]
    return max(max(blocks_per_rank) * 128, 128)


def _make_context(config: _BenchmarkConfig, runtime, pool_tokens):
    from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe.api import (
        MegaMoEContext,
    )

    device = runtime.device
    world_size = runtime.world_size
    experts_per_rank = config.num_experts // world_size
    scale_columns = config.hidden // 128
    n_l1_tiles = config.intermediate // 64
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    pointers = (
        runtime.malloc(config.tokens * config.hidden),
        runtime.malloc(config.tokens * scale_columns * 4),
        runtime.malloc(config.tokens * config.topk * 4),
        runtime.malloc(experts_per_rank * world_size * config.max_recv * 4),
        runtime.malloc(world_size * experts_per_rank * 8),
        runtime.malloc(experts_per_rank * 8),
        runtime.malloc(config.topk * config.tokens * config.hidden * 2),
        runtime.malloc(16 * 4),
        runtime.malloc(pool_tokens * 3 * 4),
    )
    tables = tuple(runtime.peer_table(pointer) for pointer in pointers)
    local = lambda *shape, dtype: torch.empty(shape, dtype=dtype, device=device)

    return MegaMoEContext(
        rank=runtime.rank,
        world_size=world_size,
        num_sms=num_sms,
        num_experts=config.num_experts,
        hidden_size=config.hidden,
        intermediate_size=config.intermediate,
        max_tokens=config.tokens,
        max_topk=config.topk,
        max_recv=config.max_recv,
        pool_tokens=pool_tokens,
        token_ptrs=tables[0],
        scale_ptrs=tables[1],
        route_weight_ptrs=tables[2],
        queue_ptrs=tables[3],
        recv_count_ptrs=tables[4],
        recv_sum_ptrs=tables[5],
        combine_ptrs=tables[6],
        signal_ptrs=tables[7],
        metadata_ptrs=tables[8],
        combine_local=pointers[6],
        token_peer_ptrs=runtime.peer_addresses(pointers[0]),
        expert_send_count=local(config.num_experts, dtype=torch.int64),
        l1_arrival_count=local(pool_tokens // 128, dtype=torch.int32),
        l2_arrival_mask=local(pool_tokens // 128, dtype=torch.int64),
        global_counters=local(8, dtype=torch.int32),
        metadata_ready=local(num_sms, dtype=torch.int32),
        l1_tokens=local(pool_tokens, config.hidden, dtype=torch.float8_e4m3fn),
        l1_scales=local(scale_columns * pool_tokens, dtype=torch.float32),
        l1_weights=local(pool_tokens, dtype=torch.float32),
        l2_activations=local(
            pool_tokens, config.intermediate, dtype=torch.float8_e4m3fn
        ),
        l2_scales=local(n_l1_tiles * pool_tokens, dtype=torch.float32),
        barrier_all_on_stream=runtime.barrier_all_on_stream,
    )


def _configure_descriptor_allocator(device: torch.device) -> None:
    import triton

    def allocate(size: int, alignment: int, stream):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(allocate)


def _run_worker(config: _BenchmarkConfig) -> None:
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megamoe import megamoe_fp8

    assert config.num_experts % config.num_ranks == 0
    with _StandaloneNvshmem() as runtime:
        assert runtime.world_size == config.num_ranks
        _configure_descriptor_allocator(runtime.device)
        inputs = _make_inputs(config, runtime)
        pool_tokens = _pool_tokens(inputs[-1], config)
        context = _make_context(config, runtime, pool_tokens)
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
    if config.data_dir is not None:
        arguments.extend(("--data-dir", str(config.data_dir)))
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
            torch_op=_launch_megamoe,
            gems_op=_launch_megamoe,
            dtypes=self.DEFAULT_DTYPES,
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = DEFAULT_SHAPES
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
    parser.add_argument("--data-dir", type=Path)
    arguments = vars(parser.parse_args())
    arguments.pop("worker")
    return _BenchmarkConfig(**arguments)


if __name__ == "__main__" and _IS_WORKER:
    _run_worker(_parse_worker_config())

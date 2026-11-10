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

"""Standalone multi-rank test support for MegaMoE."""

from __future__ import annotations

import ctypes
import os
import signal
import socket
import subprocess
import sys
from contextlib import nullcontext, suppress
from pathlib import Path

import torch

SHAPE_DESC = "tokens_per_rank, hidden, intermediate, num_experts, topk, ep_size"


def run_workers(
    ep, *worker_args, timeout=1800, rendezvous=("--standalone",), log_path=None
):
    """Run a single-node torchrun job and reap its process group on timeout."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *rendezvous,
        f"--nproc-per-node={ep}",
        *worker_args,
    ]
    if log_path is None:
        output = nullcontext(subprocess.PIPE)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output = log_path.open("w")
    with output as stdout:
        with subprocess.Popen(
            command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                log, _ = process.communicate(timeout=timeout)
            except BaseException:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
    if process.returncode:
        detail = f"see {log_path}" if log_path is not None else log
        raise RuntimeError(f"MegaMoE workers failed ({process.returncode}): {detail}")
    return log


def run_ascend_workers(ep, module, directory):
    directory = Path(directory)
    run_workers(
        ep,
        "-m",
        module,
        "--ascend-worker",
        str(directory),
        rendezvous=("--standalone", "--local-addr=127.0.0.1"),
        log_path=directory / "workers.log",
    )


class StandaloneNvshmem:
    """Own process-level NVSHMEM resources for tests and benchmarks."""

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


def interleave_l1_gate_up_rows(weight: torch.Tensor) -> torch.Tensor:
    """Pack logical ``[gate, up]`` rows into the kernel's 8-row interleave."""

    experts, rows, hidden = weight.shape
    half = rows // 2
    gate = weight[:, :half].reshape(experts, half // 8, 8, hidden)
    up = weight[:, half:].reshape(experts, half // 8, 8, hidden)
    return torch.stack((gate, up), dim=2).reshape(experts, rows, hidden).contiguous()


def pool_tokens(topk_all, config) -> int:
    """Compute the padded receive-pool capacity required by a route plan."""

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


def make_context(config, runtime: StandaloneNvshmem, pool_size: int):
    """Allocate the caller-owned context used by a standalone invocation."""

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
        runtime.malloc(pool_size * 3 * 4),
    )
    tables = tuple(runtime.peer_table(pointer) for pointer in pointers)

    def local(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

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
        pool_tokens=pool_size,
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
        l1_arrival_count=local(pool_size // 128, dtype=torch.int32),
        l2_arrival_mask=local(pool_size // 128, dtype=torch.int64),
        global_counters=local(8, dtype=torch.int32),
        metadata_ready=local(num_sms, dtype=torch.int32),
        l1_tokens=local(pool_size, config.hidden, dtype=torch.float8_e4m3fn),
        l1_scales=local(scale_columns * pool_size, dtype=torch.float32),
        l1_weights=local(pool_size, dtype=torch.float32),
        l2_activations=local(pool_size, config.intermediate, dtype=torch.float8_e4m3fn),
        l2_scales=local(n_l1_tiles * pool_size, dtype=torch.float32),
        barrier_all_on_stream=runtime.barrier_all_on_stream,
    )


def configure_descriptor_allocator(device: torch.device) -> None:
    """Install the descriptor workspace allocator used by standalone workers."""

    import triton

    def allocate(size: int, alignment: int, stream):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(allocate)


def require_ascend(ep):
    import pytest

    try:
        import shmem  # noqa: F401
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as error:
        pytest.skip(f"optional MegaMoE dependency missing: {error.name}")
    if not torch.npu.is_available() or torch.npu.device_count() < ep:
        pytest.skip(f"MegaMoE requires {ep} visible NPUs")


def load_ascend_shapes(path):
    import yaml

    with open(path) as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or "megamoe" not in document:
        raise ValueError("shape file must contain a megamoe section")
    section = document["megamoe"]
    if not isinstance(section, dict) or section.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("megamoe supports BF16 shape files only")
    shapes = section.get("shapes")
    if not isinstance(shapes, (list, tuple)) or not shapes:
        raise ValueError("megamoe requires a non-empty shape list")
    result = []
    for shape in shapes:
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 6
            or any(type(value) is not int for value in shape)
            or shape[0] < 0
            or any(value <= 0 for value in shape[1:])
        ):
            raise ValueError(
                f"expected nonnegative tokens and five positive integers ({SHAPE_DESC}): {shape}"
            )
        _, _, _, experts, topk, ep = shape
        if ep not in (2, 4, 8) or experts % ep or topk > experts:
            raise ValueError(f"invalid expert partition/topk/EP: {shape}")
        if tuple(shape) in result:
            raise ValueError(f"duplicate megamoe shape: {shape}")
        result.append(tuple(shape))
    return result


def shmem_rendezvous():
    """Choose a free single-node SHMEM port and share it with all workers."""
    import torch.distributed as dist

    endpoint = [None]
    if dist.get_rank() == 0:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            endpoint[0] = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
    dist.broadcast_object_list(endpoint, src=0)
    return endpoint[0]


def make_ascend_inputs(shape, seed, rank, routing="random"):

    tokens, hidden, intermediate, experts, topk, ep = shape
    generator = torch.Generator(device="cpu").manual_seed(seed + 1200 + rank)
    w1 = [
        torch.randn(hidden, intermediate * 2, generator=generator, dtype=torch.bfloat16)
        * 0.02
        for _ in range(experts // ep)
    ]
    w2 = [
        torch.randn(intermediate, hidden, generator=generator, dtype=torch.bfloat16)
        * 0.02
        for _ in range(experts // ep)
    ]
    x = torch.randn(tokens, hidden, generator=generator, dtype=torch.bfloat16) * 0.1
    ids = (
        torch.rand(tokens, experts, generator=generator).topk(topk, dim=1).indices.int()
    )
    if routing == "balanced":
        ids = (
            (torch.arange(tokens)[:, None] * topk + torch.arange(topk)) % experts
        ).int()
    elif routing in ("remote", "hotspot"):
        if topk > experts // ep:
            raise ValueError("single-rank routing requires topk <= local experts")
        target = (rank + 1) % ep if routing == "remote" else 0
        ids = torch.arange(topk) + target * (experts // ep)
        ids = ids.expand(tokens, topk).contiguous().int()
    probs = torch.softmax(torch.randn(tokens, topk, generator=generator), dim=1)
    # CPU fixtures keep device RNG out of accuracy/benchmark setup.
    return (
        x.to("npu"),
        ids.to("npu"),
        probs.to("npu"),
        [w.to("npu") for w in w1],
        [w.to("npu") for w in w2],
    )


def ascend_reference(inputs, shape, rank):
    """Compute every output from original tensors in FP32, independent of CANN."""
    import torch.distributed as dist

    x, ids, probs, w1, w2 = inputs
    _, hidden, _, experts, _, ep = shape
    gathered = [None] * ep
    dist.all_gather_object(gathered, tuple(t.cpu() for t in (x, ids, probs)))
    lengths = [values[0].shape[0] for values in gathered]
    gathered = [
        torch.cat([values[j] for values in gathered]).to(x.device) for j in range(3)
    ]
    xs, routes, weights = gathered
    output = torch.zeros(sum(lengths), hidden, dtype=torch.float32, device=x.device)
    for local in range(experts // ep):
        row, slot = torch.where(routes == rank * (experts // ep) + local)
        if row.numel():
            gate, up = (xs[row].float() @ w1[local].float()).chunk(2, dim=1)
            partial = (torch.nn.functional.silu(gate) * up) @ w2[local].float()
            output.index_add_(0, row, partial * weights[row, slot, None])
    if sum(lengths):
        dist.all_reduce(output)
    counts = torch.bincount(routes.flatten().long(), minlength=experts).int()
    return (
        output[sum(lengths[:rank]) : sum(lengths[: rank + 1])],
        counts[rank * (experts // ep) : (rank + 1) * (experts // ep)],
    )


def check_ascend_output(result, expected, inputs):

    output, counts = result
    reference_output, reference_counts = expected
    assert output.shape == inputs[0].shape and output.dtype == torch.bfloat16
    assert torch.isfinite(output).all().item(), "non-finite output"
    assert counts.dtype == torch.int32 and counts.shape == reference_counts.shape
    assert counts.device == output.device
    assert torch.equal(counts, reference_counts), "expert counts differ"
    delta = output.float() - reference_output
    error = (delta.norm() / reference_output.norm().clamp_min(1e-12)).item()
    assert error < 0.03, f"relative L2 {error} >= 0.03"
    row_errors = delta.norm(dim=1) / reference_output.norm(dim=1).clamp_min(1e-12)
    maxrow = float(row_errors.max()) if output.shape[0] else 0.0
    assert maxrow < 0.03, f"maximum row relative L2 {maxrow} >= 0.03"
    return {
        "relative_l2": error,
        "max_row_relative_l2": maxrow,
        "expert_counts_match": True,
        "all_outputs_finite": True,
    }


__all__ = [
    "SHAPE_DESC",
    "StandaloneNvshmem",
    "ascend_reference",
    "check_ascend_output",
    "configure_descriptor_allocator",
    "interleave_l1_gate_up_rows",
    "load_ascend_shapes",
    "make_ascend_inputs",
    "make_context",
    "pool_tokens",
    "require_ascend",
    "run_ascend_workers",
    "run_workers",
    "shmem_rendezvous",
]

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

"""Standalone multi-rank test support for the Hopper MegaMoE operator."""

from __future__ import annotations

import ctypes
import os

import torch


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


__all__ = [
    "StandaloneNvshmem",
    "configure_descriptor_allocator",
    "interleave_l1_gate_up_rows",
    "make_context",
    "pool_tokens",
]

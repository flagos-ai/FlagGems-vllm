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

"""Standalone PPU MegaMoE support for tests and benchmarks."""

from __future__ import annotations

import ctypes
import ctypes.util
import os

import torch
import triton
import triton.language as tl


class _SailshmemUniqueId(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int),
        ("id", ctypes.c_void_p),
        ("rank", ctypes.c_int),
        ("world_size", ctypes.c_int),
    ]


class _SailshmemInitArgs(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int),
        ("unique_id", _SailshmemUniqueId),
        ("content", ctypes.c_char * 96),
    ]


class _SailshmemInitAttr(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int),
        ("mpi_comm", ctypes.c_void_p),
        ("args", _SailshmemInitArgs),
    ]


class StandaloneSailshmem:
    """Test-side owner of the installed sailSHMEM host ABI, without a C++ shim."""

    def __init__(self, config):
        self.rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
        self.world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(self.device)
        if self.world_size != config.num_ranks:
            raise ValueError("MPI world size does not match --num-ranks")
        self.library = None
        self.allocations = []
        self.start_event = None
        if self.world_size == 1:
            return
        name = ctypes.util.find_library("nvshmem_host") or "libnvshmem_host.so"
        self.library = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        signatures = {
            "nvshmemx_hostlib_init_attr": (
                [ctypes.c_uint, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "nvshmem_my_pe": ([], ctypes.c_int),
            "nvshmem_n_pes": ([], ctypes.c_int),
            "nvshmem_malloc": ([ctypes.c_size_t], ctypes.c_void_p),
            "nvshmem_ptr": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p),
            "nvshmem_free": ([ctypes.c_void_p], None),
            "nvshmem_barrier_all": ([], None),
            "nvshmemx_barrier_all_on_stream": ([ctypes.c_void_p], None),
            "nvshmemx_hostlib_finalize": ([], None),
        }
        for symbol, (arguments, result) in signatures.items():
            function = getattr(self.library, symbol)
            function.argtypes, function.restype = arguments, result
        attr = _SailshmemInitAttr()
        # Version 1 of the installed sailSHMEM host ABI, also used by the SDK bridge.
        attr.version = (1 << 16) + ctypes.sizeof(attr)
        attr.args.version = (1 << 16) + ctypes.sizeof(attr.args)
        attr.args.unique_id.version = (1 << 16) + ctypes.sizeof(attr.args.unique_id)
        attr.args.unique_id.rank = attr.args.unique_id.world_size = -1
        status = self.library.nvshmemx_hostlib_init_attr(0, ctypes.byref(attr))
        if status:
            raise RuntimeError(f"sailSHMEM initialization failed: {status}")
        if (self.library.nvshmem_my_pe(), self.library.nvshmem_n_pes()) != (
            self.rank,
            self.world_size,
        ):
            raise RuntimeError("sailSHMEM rank mapping does not match MPI")

    def allocate(self, sizes):
        if self.library is None:
            return None
        tables = {}
        for name, size in sizes.items():
            self.library.nvshmem_barrier_all()
            pointer = self.library.nvshmem_malloc(size)
            if not pointer:
                raise MemoryError(f"sailSHMEM allocation failed: {name} ({size} bytes)")
            self.allocations.append(pointer)
            self.library.nvshmem_barrier_all()
            peers = [
                (
                    pointer
                    if peer == self.rank
                    else self.library.nvshmem_ptr(pointer, peer)
                )
                for peer in range(self.world_size)
            ]
            if not all(peers):
                raise RuntimeError("all EP peers must be directly addressable")
            tables[name] = torch.tensor(peers, device=self.device, dtype=torch.int64)
        return tables

    def barrier(self, stream):
        if self.library is not None:
            self.library.nvshmemx_barrier_all_on_stream(ctypes.c_void_p(stream))
        if self.start_event is not None:
            self.start_event.record(torch.cuda.current_stream(self.device))

    def synchronize(self):
        torch.cuda.synchronize(self.device)
        if self.library is not None:
            self.library.nvshmem_barrier_all()

    def close(self):
        self.synchronize()
        if self.library is not None:
            for pointer in reversed(self.allocations):
                self.library.nvshmem_free(pointer)
            self.library.nvshmemx_hostlib_finalize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()


def reference_int8_moe(x, scales, w1, s1, w2, s2, router, ids):
    """Independent CPU reference, including the two BF16 rounding boundaries."""
    t, h = x.shape
    partial = torch.zeros((ids.shape[1], t, h), dtype=torch.bfloat16)
    for expert in range(w1.shape[0]):
        tokens, slots = torch.where(ids == expert)
        if not len(tokens):
            continue
        first = x[tokens].float() @ w1[expert].float().T
        first = (first * scales[tokens] * s1[expert].T).bfloat16().float()
        gate, up = first.chunk(2, dim=-1)
        silu = (gate / (1 + torch.exp(-gate))).bfloat16().float()
        activated = (silu * up).bfloat16().float()
        amax = activated.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        scale = amax / 127
        scaled = activated * (127 / amax)
        quantized = (scaled.sign() * (scaled.abs() + 0.5).floor()).to(torch.int8)
        second = quantized.float() @ w2[expert].float().T
        values = second * scale * s2[expert].T * router[tokens, slots, None]
        partial[slots, tokens] = values.bfloat16()
    return partial.float().sum(0).bfloat16()


def make_ppu_cpu_inputs(config):
    """Use the same seeded global inputs on every rank, without external datasets."""
    generator = torch.Generator().manual_seed(1234)
    world, t, h, i, e, k = (
        config.num_ranks,
        config.tokens,
        config.hidden,
        config.intermediate,
        config.num_experts,
        config.topk,
    )
    return dict(
        x=torch.randint(-8, 9, (world, t, h), generator=generator, dtype=torch.int8),
        scales=torch.rand(world, t, 1, generator=generator) * 0.02 + 0.01,
        ids=torch.topk(
            torch.randn(world, t, e, generator=generator), k, dim=-1
        ).indices,
        router=torch.rand(world, t, k, generator=generator),
        w1=torch.randint(-8, 9, (e, 2 * i, h), generator=generator, dtype=torch.int8),
        s1=torch.rand(e, 2 * i, 1, generator=generator) * 0.02 + 0.01,
        w2=torch.randint(-8, 9, (e, h, i), generator=generator, dtype=torch.int8),
        s2=torch.rand(e, h, 1, generator=generator) * 0.02 + 0.01,
    )


@triton.jit
def _write_tables(
    table,
    buffers,
    sizes: tl.constexpr,
    peer_tables=(),
    peer_sizes: tl.constexpr = (),
    RANK: tl.constexpr = 0,
):
    for index in tl.static_range(len(buffers)):
        tl.store(table + index * 2, buffers[index].to(tl.int64))
        tl.store(table + index * 2 + 1, sizes[index])
    for index in tl.static_range(len(peer_sizes)):
        row = len(sizes) + index
        tl.store(table + row * 2, tl.load(peer_tables[index] + RANK))
        tl.store(table + row * 2 + 1, peer_sizes[index])


def make_ppu_context(
    config,
    device,
    peer_tables=None,
    barrier=None,
    *,
    all_ranks_true=None,
    l1_tok_offset=0,
):
    """Allocate the PPU context in the caller, outside timed invocations."""
    from flaggems_vllm.runtime.backend._thead.mega.megamoe.api import (
        MegaMoEContext,
        _buffer_sizes,
        _is_tuned_shape,
        _tiles,
    )

    sizes = _buffer_sizes(config)
    if barrier is not None and not callable(barrier):
        raise TypeError("barrier must be a stream barrier callable")
    if config.num_ranks > 1 and (peer_tables is None or barrier is None):
        raise ValueError(
            "multiple ranks require explicit peer tables and a stream barrier"
        )
    with torch.cuda.device(device):
        references = []
        tables = dict(peer_tables or {})
        if config.num_ranks == 1 and not tables:
            for name, size in sizes.items():
                memory = torch.empty(size, device=device, dtype=torch.int8)
                table = torch.empty(2, device=device, dtype=torch.int64)
                _write_tables[(1,)](table, (memory,), (size,))
                tables[name] = table[:1]
                references.extend((memory, table))
        for name in sizes:
            if name not in tables:
                raise ValueError(f"missing peer pointer table: {name}")
            table = tables[name]
            if (
                table.dtype != torch.int64
                or table.device != device
                or table.shape != (config.num_ranks,)
                or not table.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be a contiguous INT64 pointer table of shape (num_ranks,) on {device}"
                )

        block_m, block_n, l2_block_n, l2_group_m = _tiles(config)
        epr = config.num_experts // config.num_ranks
        max_blocks = sizes["meta_tab"] // (block_m * 12)
        pool_tokens = sizes["meta_tab"] // 12
        route_chunks = triton.cdiv(config.tokens * config.topk, 32)
        nl1n = config.intermediate_size // (block_n // 2)
        nl2n = config.hidden_size // l2_block_n
        l1_tasks = max_blocks * nl1n
        l2_tasks = max_blocks * nl2n
        buffers = {}

        def allocate(name, shape, dtype=torch.int32):
            result = torch.empty(shape, device=device, dtype=dtype)
            buffers[name] = result
            return result

        allocate("block_base_tab", epr)
        allocate("active_routes", config.num_ranks * config.tokens * config.topk)
        allocate("active_count", config.num_ranks)
        allocate("block_rows", max_blocks)
        allocate("dispatch_counts", epr * config.num_ranks)
        allocate("dispatch_chunk_counts", config.num_ranks * route_chunks * epr)
        allocate("dispatch_chunk_offsets", config.num_ranks * route_chunks * epr)
        allocate("dispatch_block_arrived", max_blocks)
        allocate("dispatch_phase", 4)
        allocate("dispatch_cursor", 3)
        if l1_tok_offset:
            storage = torch.empty(
                pool_tokens * config.hidden_size + l1_tok_offset,
                device=device,
                dtype=torch.int8,
            )
            buffers["l1_tok"] = storage[l1_tok_offset:].view(
                pool_tokens, config.hidden_size
            )
        else:
            allocate("l1_tok", (pool_tokens, config.hidden_size), torch.int8)
        allocate("l1_sf", pool_tokens, torch.float32)
        allocate("l1_w", pool_tokens, torch.float32)
        allocate("l2_acts", (pool_tokens, config.intermediate_size), torch.int8)
        allocate("l2_bf16", (pool_tokens, config.intermediate_size), torch.bfloat16)
        allocate("l2_sf", pool_tokens, torch.float32)
        allocate("l2_mask", max_blocks, torch.int64)
        allocate("l1_meta", (l1_tasks, 4))
        allocate("l2_meta", (l2_tasks, 4))
        for name, size in (
            ("l2_done", 1),
            ("l1_done", 1),
            ("l1_ready", l1_tasks),
            ("task_cursor", 3),
            ("combine_ready", 1),
        ):
            allocate(name, size)
        output = allocate(
            "final_y", (config.tokens, config.hidden_size), torch.bfloat16
        )
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        canonical = _is_tuned_shape(config)
        large_np2 = canonical and config.num_ranks == 2 and config.tokens >= 2048
        packed_routes = (
            large_np2 and epr <= 128 and config.tokens * config.topk <= 1 << 19
        )
        options = dict(
            MY_PE=config.rank,
            NPES=config.num_ranks,
            NTOK=config.tokens,
            TOPK=config.topk,
            K=config.hidden_size,
            INTER=config.intermediate_size,
            L1_N=2 * config.intermediate_size,
            L2_N=config.hidden_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=128,
            L1_OUT_N=block_n // 2,
            NPAIR=block_n // 16,
            NK1=config.hidden_size // 128,
            NK2=config.intermediate_size // 128,
            POOL_TOKENS=pool_tokens,
            K_POW2=config.hidden_size,
            NL1N=nl1n,
            NL2N=nl2n,
            L2_GROUP_M=32 if packed_routes else l2_group_m,
            LARGE_NP2_TUNE=large_np2,
            PACK_ROUTE_META=packed_routes,
            AGGREGATE_COMPLETION=canonical
            and config.tokens > 512
            and config.num_ranks in (2, 4, 8),
            NSF=1,
            EPR=epr,
            EPR_POW2=triton.next_power_of_2(epr),
            ROLE_BUDGET=sms,
            ROUTE_CHUNK=32,
            ROUTE_CHUNKS=route_chunks,
            num_warps=4,
            launch_cooperative_grid=True,
        )
        reset_names = (
            "l2_mask",
            "dispatch_counts",
            "dispatch_chunk_counts",
            "dispatch_chunk_offsets",
            "active_count",
            "dispatch_block_arrived",
            "block_rows",
            "dispatch_phase",
            "dispatch_cursor",
            "l2_done",
            "l1_done",
            "l1_ready",
            "task_cursor",
            "combine_ready",
        )
        reset_items = [
            (
                buffers[name],
                buffers[name].numel() * buffers[name].element_size(),
            )
            for name in reset_names
        ]
        # Peer-buffer reset pointers are resolved on-device; no raw pointer table is
        # copied to the CPU by this module.
        peer_reset_names = ("meta_tab", "sig_tab", "cb_tab")
        reset_table = torch.empty(
            (len(reset_items) + len(peer_reset_names), 2),
            device=device,
            dtype=torch.int64,
        )
        _write_tables[(1,)](
            reset_table,
            tuple(x[0] for x in reset_items),
            tuple(x[1] for x in reset_items),
            tuple(tables[name] for name in peer_reset_names),
            tuple(sizes[name] for name in peer_reset_names),
            config.rank,
        )
        return MegaMoEContext(
            config,
            device,
            buffers,
            tables,
            options,
            reset_table,
            min(
                sms * 4,
                sum(
                    triton.cdiv(size, 4096)
                    for size in [x[1] for x in reset_items]
                    + [sizes[name] for name in peer_reset_names]
                ),
            ),
            barrier,
            tuple(references),
            output,
            all_ranks_true=all_ranks_true,
        )


def assert_relative_l2(actual, reference):
    difference = (actual.double() - reference.double()).square().sum()
    denominator = reference.double().square().sum().clamp_min(1e-30)
    error = (difference / denominator).sqrt().item()
    assert error < 0.005, f"relative L2 error {error:.6f} exceeds 0.005"

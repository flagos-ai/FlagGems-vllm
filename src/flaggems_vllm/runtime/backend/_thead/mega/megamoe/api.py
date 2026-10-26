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

"""INT8 MegaMoE host dispatch with dynamic device-side routing.

The activation is INT8 with one FP32 scale per token. Weights are INT8 with
FP32 per-output-channel scales. L1 gate/up rows and scales use alternating
8-row chunks; ``pack_gate_up_rows`` prepares that static weight layout.

Temporary tensors are allocated internally from shape-only upper bounds.
The device dispatch rebuilds expert offsets from the current routing IDs on
every call. Multi-rank callers supply their symmetric peer buffers and stream
barrier through their existing communication runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch
import triton
import triton.language as tl

from .kernel import cta_role_megakernel


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


@triton.jit
def _reset(table, BLOCK: tl.constexpr, BUFFERS: tl.constexpr):
    """Clear aligned words with a bounded persistent grid."""
    first = tl.program_id(0) * BLOCK
    stride = tl.num_programs(0) * BLOCK
    lanes = tl.arange(0, BLOCK)
    for index in tl.range(0, BUFFERS, loop_unroll_factor=1):
        pointer = tl.load(table + index * 2).to(tl.pointer_type(tl.int32))
        size = tl.load(table + index * 2 + 1) // 4
        for start in range(first, size, stride):
            offsets = start + lanes
            tl.store(pointer + offsets, 0, mask=offsets < size)


@triton.jit
def _copy_peer(
    source,
    table,
    RANK: tl.constexpr,
    SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    target = tl.load(table + RANK).to(tl.pointer_type(DTYPE))
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + offsets, mask=offsets < SIZE, other=0)
    tl.store(target + offsets, values.to(DTYPE), mask=offsets < SIZE)


@triton.jit
def _pack_rows(
    source,
    target,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    SIZE: tl.constexpr,
    GRANULARITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // WIDTH % ROWS
    half = ROWS // 2
    chunk = row // GRANULARITY
    original_row = chunk // 2 * GRANULARITY + row % GRANULARITY + chunk % 2 * half
    source_offset = (
        offsets // (ROWS * WIDTH) * ROWS * WIDTH
        + original_row * WIDTH
        + offsets % WIDTH
    )
    tl.store(
        target + offsets,
        tl.load(source + source_offset, mask=offsets < SIZE, other=0),
        mask=offsets < SIZE,
    )


def pack_gate_up_rows(weight: torch.Tensor) -> torch.Tensor:
    """Pack static N-major gate/up weights or channel scales with Triton."""
    if (
        weight.ndim != 3
        or any(size == 0 for size in weight.shape)
        or weight.shape[1] % 2
    ):
        raise ValueError("gate/up input must be rank 3 with an even row count")
    granularity = 8
    if weight.shape[1] // 2 % granularity:
        raise ValueError("gate/up half must be divisible by 8")
    if weight.requires_grad:
        raise NotImplementedError(
            "weight packing only supports inference without autograd"
        )
    if weight.device.type != "cuda" or not weight.is_contiguous():
        raise ValueError("gate/up input must be a contiguous CUDA/PPU tensor")
    if weight.dtype not in (torch.int8, torch.float32):
        raise TypeError("gate/up input must use INT8 weights or FP32 channel scales")
    with torch.cuda.device(weight.device):
        output = torch.empty_like(weight)
        _pack_rows[(triton.cdiv(weight.numel(), 256),)](
            weight,
            output,
            weight.shape[1],
            weight.shape[2],
            weight.numel(),
            granularity,
            256,
        )
        return output


@dataclass(frozen=True)
class _Config:
    tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int
    num_ranks: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        values = (
            self.tokens,
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.topk,
            self.num_ranks,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in values
        ):
            raise ValueError(
                "MegaMoE dimensions and rank count must be positive integers"
            )
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or not 0 <= self.rank < self.num_ranks
        ):
            raise ValueError("rank must be in [0, num_ranks)")
        if self.num_experts % self.num_ranks:
            raise ValueError("num_experts must be divisible by num_ranks")
        if self.topk > self.num_experts:
            raise ValueError("topk must not exceed num_experts")
        hidden = self.hidden_size
        if hidden % 128 or hidden > 65536 or hidden & (hidden - 1):
            raise NotImplementedError(
                "hidden_size must be a power of two in [128, 65536]"
            )
        if self.intermediate_size % 128 or self.intermediate_size // 64 >= 64:
            raise NotImplementedError(
                "intermediate_size must be divisible by 128 and below 4096"
            )


def _is_tuned_shape(config: _Config) -> bool:
    return (
        config.hidden_size,
        config.intermediate_size,
        config.num_experts,
        config.topk,
    ) == (2048, 512, 256, 8)


def _tiles(config: _Config) -> tuple[int, int, int, int]:
    canonical = _is_tuned_shape(config)
    if canonical and (
        (config.num_ranks in (2, 4) and config.tokens <= 512)
        or (config.num_ranks == 8 and config.tokens <= 128)
    ):
        return 16, 256, 256, 8
    if canonical and config.num_ranks == 2 and config.tokens < 2048:
        return 32, 256, 128, 64
    return 64, 128, 128, 64


@dataclass
class _Scratch:
    """Internal temporary tensors and their allocation owners."""

    buffers: dict[str, torch.Tensor]
    peer_tables: dict[str, torch.Tensor]
    options: dict
    reset_table: torch.Tensor
    reset_programs: int
    barrier: Callable[[int], None] | None
    owners: tuple
    output: torch.Tensor


def _buffer_sizes(config: _Config) -> dict[str, int]:
    block_m, _, _, _ = _tiles(config)
    epr = config.num_experts // config.num_ranks
    # Sum_e max(1, ceil(routes_e / M)) <= EPR + ceil(all_routes / M).
    # Shape-only capacity covers arbitrary routing skew without CPU readback.
    max_blocks = epr + triton.cdiv(
        config.num_ranks * config.tokens * config.topk, block_m
    )
    routes = config.tokens * config.topk
    return dict(
        tok_tab=config.tokens * config.hidden_size,
        sf_tab=config.tokens * 4,
        w_tab=routes * 4,
        route_tab=routes * 4,
        cb_tab=routes * config.hidden_size * 2,
        sig_tab=16 * 4,
        meta_tab=max_blocks * block_m * 3 * 4,
    )


def _allocate_scratch(
    config: _Config,
    device: torch.device,
    peer_tables: Mapping[str, torch.Tensor] | None,
    barrier: Callable[[int], None] | None,
) -> _Scratch:
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
        return _Scratch(
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
        )


_KERNEL_ARGUMENTS = (
    "route_tab",
    "active_routes",
    "active_count",
    "block_rows",
    "dispatch_counts",
    "dispatch_chunk_counts",
    "dispatch_chunk_offsets",
    "block_base_tab",
    "l1_meta",
    "l2_meta",
    "dispatch_block_arrived",
    "dispatch_phase",
    "dispatch_cursor",
    "topk_local",
    "l1_tok",
    "l1_sf",
    "l1_w",
    "tok_tab",
    "sf_tab",
    "w_tab",
    "w1_kt",
    "w1_sf",
    "l2_acts",
    "l2_sf",
    "l2_mask",
    "l2_done",
    "l1_done",
    "l1_ready",
    "task_cursor",
    "combine_ready",
    "meta_tab",
    "cb_tab",
    "w2_kt",
    "w2_sf",
    "final_y",
    "sig_tab",
    "l2_bf16",
)


def megamoe(
    hidden_states: torch.Tensor,
    input_scales: torch.Tensor,
    w1: torch.Tensor,
    w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    rank: int = 0,
    num_ranks: int = 1,
    peer_tables: Mapping[str, torch.Tensor] | None = None,
    barrier: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """Return a newly allocated rank-local BF16 result from INT8 activations.

    Inputs are contiguous tensors on one PPU device. W1 and its channel scales
    use the granularity-8 gate/up layout from pack_gate_up_rows. Temporary
    storage is internal; routing IDs and expert load may change on every call.

    Multi-rank use requires INT64 pointer tables sized by peer_buffer_sizes,
    and a stream barrier supplied by the application's communication runtime.
    All ranks must use matching dimensions and call this operator collectively;
    their symmetric peer buffers must remain alive until execution completes.

    Scales and routing weights must be finite; scales are nonnegative. Expert
    IDs are global indices, with -1 marking an unused slot. Value constraints
    are caller obligations, not synchronous host checks. Inference only.
    """
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or topk_ids.ndim != 2:
        raise ValueError("expected rank-2 activation/routing and rank-3 weights")
    config = _Config(
        hidden_states.shape[0],
        hidden_states.shape[1],
        w2.shape[2],
        w1.shape[0] * num_ranks,
        topk_ids.shape[1],
        num_ranks,
        rank,
    )
    device = hidden_states.device
    if device.type != "cuda":
        raise ValueError("MegaMoE requires a CUDA/PPU device")
    epr = config.num_experts // config.num_ranks
    contracts = (
        (hidden_states, (config.tokens, config.hidden_size), torch.int8),
        (input_scales, (config.tokens, 1), torch.float32),
        (w1, (epr, 2 * config.intermediate_size, config.hidden_size), torch.int8),
        (w1_scales, (epr, 2 * config.intermediate_size, 1), torch.float32),
        (w2, (epr, config.hidden_size, config.intermediate_size), torch.int8),
        (w2_scales, (epr, config.hidden_size, 1), torch.float32),
        (topk_weights, (config.tokens, config.topk), torch.float32),
    )
    for tensor, shape, dtype in contracts:
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"expected contiguous {dtype} tensor of shape {shape} on {device}"
            )
    if (
        topk_ids.shape != (config.tokens, config.topk)
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_ids.device != device
        or not topk_ids.is_contiguous()
    ):
        raise ValueError(
            "topk_ids must be a contiguous INT32/INT64 (tokens, topk) tensor on the input device"
        )
    inputs = [tensor for tensor, _, _ in contracts] + [topk_ids]
    if any(tensor.requires_grad for tensor in inputs):
        raise NotImplementedError("MegaMoE only supports inference without autograd")
    scratch = _allocate_scratch(config, device, peer_tables, barrier)
    with torch.cuda.device(device):
        _reset[(scratch.reset_programs,)](
            scratch.reset_table, 1024, scratch.reset_table.shape[0], num_warps=4
        )
        copies = (
            (hidden_states, "tok_tab", tl.int8),
            (input_scales, "sf_tab", tl.float32),
            (topk_weights, "w_tab", tl.float32),
            (topk_ids, "route_tab", tl.int32),
        )
        for source, name, dtype in copies:
            _copy_peer[(triton.cdiv(source.numel(), 256),)](
                source,
                scratch.peer_tables[name],
                config.rank,
                source.numel(),
                dtype,
                256,
            )
        if scratch.barrier is not None:
            scratch.barrier(torch.cuda.current_stream(device).cuda_stream)
        arguments = dict(scratch.buffers, **scratch.peer_tables)
        arguments.update(
            topk_local=topk_ids, w1_kt=w1, w1_sf=w1_scales, w2_kt=w2, w2_sf=w2_scales
        )
        cta_role_megakernel[(scratch.options["ROLE_BUDGET"],)](
            *(arguments[name] for name in _KERNEL_ARGUMENTS), **scratch.options
        )
        return scratch.output


def peer_buffer_sizes(
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    num_ranks: int,
) -> dict[str, int]:
    """Return symmetric allocation sizes in bytes for a collective invocation.

    Capacities depend only on shape, cover arbitrary expert load including all
    routes targeting one expert, and stay valid when routing IDs change.
    The caller creates INT64 pointer tables with one peer address per rank;
    those buffers are owned by its communication runtime, not this operator.
    """
    return _buffer_sizes(
        _Config(tokens, hidden_size, intermediate_size, num_experts, topk, num_ranks)
    )

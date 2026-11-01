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

The caller supplies a MegaMoEContext with shape-bounded workspace, peer
buffers and a stream barrier. Device dispatch rebuilds expert offsets from
the current routing IDs on every call; the same workspace can be reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl

from .kernel import cta_role_megakernel


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
class MegaMoEConfig:
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


def _is_tuned_shape(config: MegaMoEConfig) -> bool:
    return (
        config.hidden_size,
        config.intermediate_size,
        config.num_experts,
        config.topk,
    ) == (2048, 512, 256, 8)


def _tiles(config: MegaMoEConfig) -> tuple[int, int, int, int]:
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
class MegaMoEContext:
    """Caller-owned workspace for serial invocations on one stream.

    All tensors and communication allocations must remain alive until execution
    completes. Reusing this context overwrites its output; clone the result in
    the caller if it must survive the next invocation. Concurrent invocations
    require separate contexts and separate symmetric peer buffers.
    """

    config: MegaMoEConfig
    device: torch.device

    buffers: dict[str, torch.Tensor]
    peer_tables: dict[str, torch.Tensor]
    options: dict
    reset_table: torch.Tensor
    reset_programs: int
    barrier: Callable[[int], None] | None
    owners: tuple
    output: torch.Tensor


def _buffer_sizes(config: MegaMoEConfig) -> dict[str, int]:
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
    context: MegaMoEContext,
) -> torch.Tensor:
    """Write the rank-local BF16 result into caller-owned context.output.

    Inputs are contiguous tensors on one PPU device. W1 and its channel scales
    use the granularity-8 gate/up layout from pack_gate_up_rows. The context
    must match the input dimensions, device, rank and collective configuration.
    This call allocates no workspace; routing IDs and expert load may change.

    All ranks must call this operator collectively with matching dimensions.
    The caller must keep the context and symmetric peer buffers alive until
    execution completes. Before reusing them, ensure the previous invocation
    completed on every rank. The stream barrier orders this invocation's input
    copies. Overlapping invocations must use distinct contexts and peer buffers.
    The returned tensor aliases context.output and is overwritten on reuse.

    Scales and routing weights must be finite; scales are nonnegative. Expert
    IDs are global indices, with -1 marking an unused slot. Value constraints
    are caller obligations, not synchronous host checks. Inference only.
    """
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or topk_ids.ndim != 2:
        raise ValueError("expected rank-2 activation/routing and rank-3 weights")
    if not isinstance(context, MegaMoEContext):
        raise TypeError("context must be a MegaMoEContext")
    config = MegaMoEConfig(
        hidden_states.shape[0],
        hidden_states.shape[1],
        w2.shape[2],
        w1.shape[0] * context.config.num_ranks,
        topk_ids.shape[1],
        context.config.num_ranks,
        context.config.rank,
    )
    device = hidden_states.device
    if device.type != "cuda":
        raise ValueError("MegaMoE requires a CUDA/PPU device")
    if context.config != config or context.device != device:
        raise ValueError("context dimensions and device must match the inputs")
    if config.num_ranks > 1 and not callable(context.barrier):
        raise ValueError("multiple ranks require a stream barrier")
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
    with torch.cuda.device(device):
        _reset[(context.reset_programs,)](
            context.reset_table, 1024, context.reset_table.shape[0], num_warps=4
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
                context.peer_tables[name],
                config.rank,
                source.numel(),
                dtype,
                256,
            )
        if context.barrier is not None:
            context.barrier(torch.cuda.current_stream(device).cuda_stream)
        arguments = dict(context.buffers, **context.peer_tables)
        arguments.update(
            topk_local=topk_ids, w1_kt=w1, w1_sf=w1_scales, w2_kt=w2, w2_sf=w2_scales
        )
        cta_role_megakernel[(context.options["ROLE_BUDGET"],)](
            *(arguments[name] for name in _KERNEL_ARGUMENTS), **context.options
        )
        return context.output


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
        MegaMoEConfig(
            tokens, hidden_size, intermediate_size, num_experts, topk, num_ranks
        )
    )

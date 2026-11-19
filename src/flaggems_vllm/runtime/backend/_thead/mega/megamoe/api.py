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

from dataclasses import InitVar, dataclass, field
from typing import Callable

import torch
import triton
import triton.language as tl

from .kernel import cta_role_megakernel


@triton.jit
def _reset(
    table,
    BLOCK: tl.constexpr,
    BUFFERS: tl.constexpr,
    cb_tab=None,
    MY_PE: tl.constexpr = 0,
    SKIP_CB: tl.constexpr = False,
):
    """Clear aligned words with a bounded persistent grid."""
    first = tl.program_id(0).to(tl.int64) * BLOCK
    stride = tl.num_programs(0).to(tl.int64) * BLOCK
    lanes = tl.arange(0, BLOCK)
    if SKIP_CB:
        skip_pointer = tl.load(cb_tab + MY_PE)
    else:
        skip_pointer = tl.full((), -1, tl.int64)
    for index in tl.range(0, BUFFERS, loop_unroll_factor=1):
        pointer_value = tl.load(table + index.to(tl.int64) * 2)
        if (not SKIP_CB) or (pointer_value != skip_pointer):
            pointer = tl.load(table + index.to(tl.int64) * 2).to(
                tl.pointer_type(tl.int32)
            )
            size = tl.load(table + index.to(tl.int64) * 2 + 1) // 4
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
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + offsets, mask=offsets < SIZE, other=0)
    tl.store(target + offsets, values.to(DTYPE), mask=offsets < SIZE)


@triton.jit
def _publish_inputs(
    hidden,
    scales,
    weights,
    routes,
    tok_tab,
    sf_tab,
    w_tab,
    route_tab,
    RANK: tl.constexpr,
    HIDDEN_ELEMENTS: tl.constexpr,
    TOKENS: tl.constexpr,
    ROUTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Publish local inputs into the symmetric buffers before the rank barrier."""
    pid = tl.program_id(0)
    hidden_blocks: tl.constexpr = triton.cdiv(HIDDEN_ELEMENTS, BLOCK)
    route_blocks: tl.constexpr = triton.cdiv(ROUTES, BLOCK)
    lanes = tl.arange(0, BLOCK)
    if pid < hidden_blocks:
        offset = pid.to(tl.int64) * BLOCK + lanes
        dest = tl.load(tok_tab + RANK).to(tl.pointer_type(tl.int8))
        x = tl.load(hidden + offset, mask=offset < HIDDEN_ELEMENTS, other=0)
        tl.store(dest + offset, x, mask=offset < HIDDEN_ELEMENTS)
    elif pid < hidden_blocks + route_blocks:
        offset = (pid - hidden_blocks).to(tl.int64) * BLOCK + lanes
        w_dest = tl.load(w_tab + RANK).to(tl.pointer_type(tl.float32))
        r_dest = tl.load(route_tab + RANK).to(tl.pointer_type(tl.int32))
        w = tl.load(weights + offset, mask=offset < ROUTES, other=0)
        r = tl.load(routes + offset, mask=offset < ROUTES, other=-1)
        tl.store(w_dest + offset, w, mask=offset < ROUTES)
        tl.store(r_dest + offset, r.to(tl.int32), mask=offset < ROUTES)
    else:
        offset = (pid - hidden_blocks - route_blocks).to(tl.int64) * BLOCK + lanes
        dest = tl.load(sf_tab + RANK).to(tl.pointer_type(tl.float32))
        sf = tl.load(scales + offset, mask=offset < TOKENS, other=0)
        tl.store(dest + offset, sf, mask=offset < TOKENS)


@triton.jit
def _prepare(
    reset_table,
    RESET_PROGRAMS: tl.constexpr,
    RESET_BUFFERS: tl.constexpr,
    SKIP_CB: tl.constexpr,
    hidden,
    scales,
    weights,
    routes,
    tok_tab,
    sf_tab,
    w_tab,
    route_tab,
    cb_tab,
    RANK: tl.constexpr,
    HIDDEN_ELEMENTS: tl.constexpr,
    TOKENS: tl.constexpr,
    ROUTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Publish local inputs into the symmetric buffers before the rank barrier."""
    if tl.program_id(0) < RESET_PROGRAMS:
        first = tl.program_id(0).to(tl.int64) * BLOCK
        stride = RESET_PROGRAMS * BLOCK
        lanes = tl.arange(0, BLOCK)
        if SKIP_CB:
            skip_pointer = tl.load(cb_tab + RANK)
        else:
            skip_pointer = tl.full((), -1, tl.int64)
        for index in tl.range(0, RESET_BUFFERS, loop_unroll_factor=1):
            pointer_value = tl.load(reset_table + index.to(tl.int64) * 2)
            if (not SKIP_CB) or (pointer_value != skip_pointer):
                pointer = tl.load(reset_table + index.to(tl.int64) * 2).to(
                    tl.pointer_type(tl.int32)
                )
                size = tl.load(reset_table + index.to(tl.int64) * 2 + 1) // 4
                for start in range(first, size, stride):
                    offsets = start + lanes
                    tl.store(pointer + offsets, 0, mask=offsets < size)

    else:
        pid = tl.program_id(0) - RESET_PROGRAMS
        hidden_blocks: tl.constexpr = triton.cdiv(HIDDEN_ELEMENTS, BLOCK)
        route_blocks: tl.constexpr = triton.cdiv(ROUTES, BLOCK)
        lanes = tl.arange(0, BLOCK)
        if pid < hidden_blocks:
            offset = pid.to(tl.int64) * BLOCK + lanes
            dest = tl.load(tok_tab + RANK).to(tl.pointer_type(tl.int8))
            x = tl.load(hidden + offset, mask=offset < HIDDEN_ELEMENTS, other=0)
            tl.store(dest + offset, x, mask=offset < HIDDEN_ELEMENTS)
        elif pid < hidden_blocks + route_blocks:
            offset = (pid - hidden_blocks).to(tl.int64) * BLOCK + lanes
            w_dest = tl.load(w_tab + RANK).to(tl.pointer_type(tl.float32))
            r_dest = tl.load(route_tab + RANK).to(tl.pointer_type(tl.int32))
            w = tl.load(weights + offset, mask=offset < ROUTES, other=0)
            r = tl.load(routes + offset, mask=offset < ROUTES, other=-1)
            tl.store(w_dest + offset, w, mask=offset < ROUTES)
            tl.store(r_dest + offset, r.to(tl.int32), mask=offset < ROUTES)
        else:
            offset = (pid - hidden_blocks - route_blocks).to(tl.int64) * BLOCK + lanes
            dest = tl.load(sf_tab + RANK).to(tl.pointer_type(tl.float32))
            sf = tl.load(scales + offset, mask=offset < TOKENS, other=0)
            tl.store(dest + offset, sf, mask=offset < TOKENS)


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
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
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


def _use_l1_pipeline(config: MegaMoEConfig) -> bool:
    """Select the PPU GEMM1 tile and staging policy for validated workloads."""
    return _is_tuned_shape(config) and (
        (config.num_ranks in (2, 4, 8) and config.tokens in (2048, 16384))
        or (config.num_ranks == 8 and 128 < config.tokens <= 512)
    )


def _tiles(config: MegaMoEConfig) -> tuple[int, int, int, int]:
    canonical = _is_tuned_shape(config)
    if canonical and config.num_ranks == 2 and config.tokens == 16384:
        return 128, 128, 128, 32
    if canonical and config.num_ranks == 2 and 152 <= config.tokens < 256:
        return 32, 256, 256, 8
    if _use_l1_pipeline(config):
        return (32 if config.tokens <= 512 else 64), 256, 128, 64
    if canonical:
        ep, tokens = config.num_ranks, config.tokens
        if ep in (2, 4) and 64 <= tokens <= 512:
            threshold = 256 if ep == 2 else 384
            return (32 if tokens >= threshold else 16), 256, 256, 8
        if ep == 8 and 128 < tokens <= 512:
            return 32, 128, 128, 64
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

    To enable the tuned EP2/EP4 return paths, pass `all_ranks_true`, a
    synchronous host callback that returns the logical AND of its argument
    across this context's EP group. Construct matching contexts in the same
    order on every rank, with the callback supplied on every rank or none.
    Without the callback, use per-route returns. Keep the workspace and its
    layout options fixed after construction.
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
    all_ranks_true: InitVar[Callable[[bool], bool] | None] = None
    _ep2_partial_reduce: bool = field(init=False, default=False, repr=False)
    _gather_input: bool = field(init=False, default=False, repr=False)

    def __post_init__(self, all_ranks_true: Callable[[bool], bool] | None) -> None:
        config = self.config
        ep = config.num_ranks
        if (
            all_ranks_true is None
            or not _is_tuned_shape(config)
            or ep not in (2, 4)
            or config.tokens not in (2048, 16384)
        ):
            return

        tuned_device = (
            self.device.type == "cuda"
            and torch.cuda.get_device_properties(self.device).name == "PPU-ZW810E"
        )
        pool_tokens = self.options.get("POOL_TOKENS", 0)
        if ep == 2:
            block_m = self.options.get("BLOCK_M", 0)
            l2_meta = self.buffers.get("l2_meta")
            self._ep2_partial_reduce = all_ranks_true(
                tuned_device
                and self.options.get("AGGREGATE_COMPLETION", False)
                and block_m > 0
                and isinstance(l2_meta, torch.Tensor)
                and l2_meta.numel()
                >= (pool_tokens // block_m) * (config.hidden_size // 256) * 4
            )
            if not self._ep2_partial_reduce:
                return

        l1_tok = self.buffers.get("l1_tok")
        source_ranks = ep if ep == 2 else ep - 1
        self._gather_input = all_ranks_true(
            tuned_device
            and isinstance(l1_tok, torch.Tensor)
            and l1_tok.data_ptr() % 16 == 0
            and pool_tokens * (config.hidden_size - 4)
            >= source_ranks * config.tokens * config.hidden_size
        )


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


def _requires_wide_index(config, arguments, inputs) -> bool:
    # Valid contexts expose flat contiguous workspaces and shape-bounded peer
    # buffers. Their full capacities cover every nonnegative index intermediate,
    # including padded rows and worklists, for every valid routing pattern.
    # Metadata queries only: no tensor values or device synchronization are read.
    largest_elements = max(tensor.numel() for tensor in (*arguments.values(), *inputs))
    largest_peer_bytes = max(_buffer_sizes(config).values())
    return max(largest_elements, largest_peer_bytes) >= 1 << 31


def _load_policy(config) -> tuple[bool, bool, bool]:
    """Return full-tile A, asynchronous A and asynchronous B selections."""
    if not _is_tuned_shape(config):
        return False, False, False
    ep, tokens = config.num_ranks, config.tokens
    if ep == 4 and tokens <= 16384:
        return False, False, True
    if ep == 2:
        if tokens <= 248:
            return False, False, True
        if 256 <= tokens <= 512:
            return True, True, True
        if 2048 <= tokens <= 16384:
            return True, False, True
    if ep == 8 and (tokens <= 128 or tokens == 2048):
        return False, False, True
    return False, False, False


def _check_full_tile_a_context(config, context, device):
    """Require complete padded INT8 tiles before dropping the A row mask."""
    epr = config.num_experts // config.num_ranks
    padding_m = _tiles(config)[0]
    padding_pool = int(context.options.get("POOL_TOKENS", 0))
    padding_required = _buffer_sizes(config)["meta_tab"] // 12
    padding_contract = dict(
        BLOCK_M=padding_m,
        NPES=config.num_ranks,
        NTOK=config.tokens,
        TOPK=config.topk,
        EPR=epr,
        K=config.hidden_size,
        INTER=config.intermediate_size,
    )
    if (
        padding_pool < padding_required
        or padding_pool % padding_m != 0
        or any(
            context.options.get(key) != value for key, value in padding_contract.items()
        )
    ):
        raise ValueError("context options must cover the padded route pool")
    for padding_name, padding_width in (
        ("l1_tok", config.hidden_size),
        ("l2_acts", config.intermediate_size),
    ):
        padding_tensor = context.buffers.get(padding_name)
        if (
            not isinstance(padding_tensor, torch.Tensor)
            or padding_tensor.dtype != torch.int8
            or padding_tensor.device != device
            or not padding_tensor.is_contiguous()
            or padding_tensor.numel() < padding_pool * padding_width
        ):
            raise ValueError(f"{padding_name} must contain the full padded INT8 pool")


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
    "local_contrib",
    "token_cache",
    "profile_times",
    "copy_row_map",
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
    vendor_numerics: bool = False,
    dispatch_workers: int = 0,
    dispatch_m_block: bool | None = None,
    return_workers: int = 0,
) -> torch.Tensor:
    """Write the rank-local BF16 result into caller-owned context.output.

    Inputs are contiguous tensors on one PPU device. W1 and its channel scales
    use the granularity-8 gate/up layout from pack_gate_up_rows. The context
    must match the input dimensions, device, rank and collective configuration.
    Partial-reduction paths cache a BF16 workspace on first use. Routing IDs
    and expert load may change between invocations.

    All ranks must call this operator collectively with matching dimensions
    and numerical mode.
    The caller must keep the context and symmetric peer buffers alive until
    execution completes. Before reusing them, ensure the previous invocation
    completed on every rank. The stream barrier orders this invocation's input
    copies. Overlapping invocations must use distinct contexts and peer buffers.
    The returned tensor aliases context.output and is overwritten on reuse.

    vendor_numerics expects input scales and INT8 values prepared with the
    native per-token quantization. It matches native SwiGLU and intermediate
    requantization. The EP2 and EP4 rank-return paths weight each BF16
    contribution in FP32 before forming its BF16 remote partial sum.

    With native numerics, canonical EP8 shapes use whole-block dispatch at
    136–512, 2048 and 16384 tokens. A copy CTA publishes each expert M block
    after finishing its payload and metadata. dispatch_m_block and
    dispatch_workers override that policy for performance comparisons.

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
            "topk_ids must be a contiguous INT32/INT64 (tokens, topk) "
            "tensor on the input device"
        )
    inputs = [tensor for tensor, _, _ in contracts] + [topk_ids]
    if any(tensor.requires_grad for tensor in inputs):
        raise NotImplementedError("MegaMoE only supports inference without autograd")
    if dispatch_m_block is None:
        dispatch_m_block = (
            vendor_numerics
            and _is_tuned_shape(config)
            and config.num_ranks == 8
            and (136 <= config.tokens <= 512 or config.tokens in (2048, 16384))
            and torch.cuda.get_device_properties(device).name == "PPU-ZW810E"
        )
        if dispatch_m_block and not dispatch_workers:
            dispatch_workers = 16 if config.tokens <= 512 else (24 if config.tokens == 16384 else 12)
    if dispatch_m_block and not dispatch_workers:
        raise ValueError("dispatch_m_block requires copy CTAs")
    if dispatch_workers and (
        not _is_tuned_shape(config) or config.num_ranks != 8
        or not 24 <= config.tokens <= (1 << 19) // config.topk
        or not 0 < dispatch_workers < 80
        or torch.cuda.get_device_properties(device).name != "PPU-ZW810E"
    ):
        raise ValueError("dedicated copy CTAs require canonical PPU EP8")
    a_block_ptr, async_a, async_b = _load_policy(config)
    if a_block_ptr or (_use_l1_pipeline(config) and config.tokens >= 2048):
        _check_full_tile_a_context(config, context, device)
    partial_reduce = context._ep2_partial_reduce and (
        not vendor_numerics or config.tokens == 16384
    )
    gather_input = partial_reduce and context._gather_input
    # Large EP4 shapes can cache each remote token once and start GEMM1
    # for locally routed blocks while the remote cache is being filled.
    ep4_cache_workers = 0
    if vendor_numerics and config.num_ranks == 4 and context._gather_input:
        ep4_cache_workers = 16 if config.tokens == 16384 else 12
    ep4_rank_return = ep4_cache_workers != 0 and config.tokens == 16384
    partial_reduce = partial_reduce or ep4_rank_return
    gather_input = gather_input or ep4_cache_workers != 0
    cache_tokens = gather_input or (
        partial_reduce
        and config.tokens == 2048
        and context.options.get("PACK_ROUTE_META", False)
    )
    if return_workers and (
        not vendor_numerics or not gather_input or config.num_ranks != 2
        or config.tokens != 16384
        or not 0 < return_workers < 80
    ):
        raise ValueError("return_workers requires the canonical EP2 rank-return path")
    if vendor_numerics and not gather_input:
        # Only the rank-return specialization applies router weights inside
        # its local reduction. Other vendor paths combine per-expert results.
        partial_reduce = False
        cache_tokens = False
    with torch.cuda.device(device):
        if cache_tokens and not gather_input and "token_cache" not in context.buffers:
            context.buffers["token_cache"] = torch.empty(
                config.num_ranks * config.tokens * config.hidden_size,
                device=device,
                dtype=torch.int8,
            )
        if cache_tokens and not gather_input:
            cache = context.buffers["token_cache"]
            if (
                not isinstance(cache, torch.Tensor)
                or cache.dtype != torch.int8
                or cache.device != device
                or not cache.is_contiguous()
                or cache.numel() < config.num_ranks * config.tokens * config.hidden_size
            ):
                raise ValueError(
                    "token_cache must cover all contiguous INT8 source tokens"
                )
        if partial_reduce and "local_contrib" not in context.buffers:
            context.buffers["local_contrib"] = torch.empty(
                context.options["POOL_TOKENS"] * config.hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
        if partial_reduce:
            workspace = context.buffers["local_contrib"]
            if (
                not isinstance(workspace, torch.Tensor)
                or workspace.dtype != torch.bfloat16
                or workspace.device != device
                or not workspace.is_contiguous()
                or workspace.numel()
                < context.options["POOL_TOKENS"] * config.hidden_size
            ):
                raise ValueError(
                    "local_contrib must cover the contiguous BF16 output pool"
                )
        if vendor_numerics and gather_input and config.tokens == 16384:
            block = 1024
            tables = context.peer_tables
            grid = context.reset_programs + sum(
                triton.cdiv(size, block)
                for size in (hidden_states.numel(), topk_ids.numel(), input_scales.numel())
            )
            _prepare[(grid,)](
                context.reset_table, context.reset_programs, context.reset_table.shape[0],
                True,
                hidden_states, input_scales, topk_weights, topk_ids,
                tables["tok_tab"], tables["sf_tab"], tables["w_tab"], tables["route_tab"],
                tables["cb_tab"], config.rank, hidden_states.numel(), input_scales.numel(),
                topk_ids.numel(), block, num_warps=4,
            )
        else:
            _reset[(context.reset_programs,)](
                context.reset_table,
                1024,
                context.reset_table.shape[0],
                context.peer_tables["cb_tab"],
                config.rank,
                SKIP_CB=partial_reduce,
                num_warps=4,
            )
            if (
                _is_tuned_shape(config)
                and config.num_ranks in (2, 4, 8)
                and 8 <= config.tokens <= 512
                and torch.cuda.get_device_properties(device).name == "PPU-ZW810E"
            ):
                publish_block = 256
                input_tables = context.peer_tables
                _publish_inputs[
                    (
                        triton.cdiv(hidden_states.numel(), publish_block)
                        + triton.cdiv(topk_ids.numel(), publish_block)
                        + triton.cdiv(input_scales.numel(), publish_block),
                    )
                ](
                    hidden_states,
                    input_scales,
                    topk_weights,
                    topk_ids,
                    input_tables["tok_tab"],
                    input_tables["sf_tab"],
                    input_tables["w_tab"],
                    input_tables["route_tab"],
                    config.rank,
                    hidden_states.numel(),
                    input_scales.numel(),
                    topk_ids.numel(),
                    publish_block,
                )
            else:
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
        if dispatch_m_block and "copy_row_map" not in context.buffers:
            context.buffers["copy_row_map"] = torch.empty(
                context.options["POOL_TOKENS"], device=device, dtype=torch.int32
            )
        arguments = dict(context.buffers, **context.peer_tables)
        arguments["profile_times"] = context.buffers.get("profile_times", context.buffers["l2_done"])
        arguments["copy_row_map"] = context.buffers.get("copy_row_map", context.buffers["l2_meta"])
        # The unused pointer keeps the ordinary path's workspace contract.
        arguments["local_contrib"] = context.buffers.get(
            "local_contrib", context.buffers["l2_bf16"]
        )
        arguments["token_cache"] = context.buffers.get(
            "token_cache", context.buffers["l1_tok"]
        )
        arguments.update(
            topk_local=topk_ids, w1_kt=w1, w1_sf=w1_scales, w2_kt=w2, w2_sf=w2_scales
        )
        # All MoE stages execute inside one cooperative persistent kernel.
        # These options only specialize metadata and arithmetic within that launch.
        options = dict(context.options)
        canonical = _is_tuned_shape(config) and config.num_ranks in (2, 4, 8)
        ep, tokens = config.num_ranks, config.tokens
        if canonical:
            options["METADATA_BLOCK"] = 16 if ep in (2, 8) and tokens <= 512 else 128
            if tokens >= 16:
                if (ep in (2, 4) and tokens <= 512) or (ep == 8 and tokens <= 128):
                    options["PACK_ROUTE_META"] = True
                options["USE_DST_PTR"] = (ep == 4 and 64 <= tokens <= 512) or (
                    ep == 8 and 129 <= tokens <= 512
                )
                if ep in (2, 4) and 64 <= tokens <= 512:
                    options["L1_QUANT_ROWS"] = 4
                    options["L2_BATCH"] = (
                        2 if tokens >= (128 if ep == 4 else 256) else 0
                    )
        if (
            canonical
            and ep in (2, 4, 8)
            and 16 <= tokens <= 512
            and options["ROLE_BUDGET"] == 64
            and options["num_warps"] == 4
            and torch.cuda.get_device_properties(device).name == "PPU-ZW810E"
        ):
            # This model's MMA kernels have 20 effective units and admit four
            # resident CTAs per unit for the selected tiles, rather than the
            # 64 general-purpose units reported by the CUDA-compatible API.
            options["ROLE_BUDGET"] = 80
        # Keep measured specializations within one cooperative main launch.
        options["L2_STAGES"] = 2
        if canonical:
            if ep == 2 and 64 <= tokens <= 512:
                options["USE_DST_PTR"] = True
                if tokens >= 256:
                    options["L2_STAGES"] = 1
                    options["L1_QUANT_ROWS"] = options["BLOCK_M"]
            elif ep == 4 and tokens <= 16384:
                options["METADATA_BLOCK"] = 16
                options["PACK_ROUTE_META"] = True
            elif ep == 8 and 129 <= tokens <= 512:
                options["PACK_ROUTE_META"] = True
        options["WIDE_INDEX"] = _requires_wide_index(config, arguments, inputs)
        options["A_BLOCK_PTR"] = a_block_ptr
        options["ASYNC_A"] = async_a
        options["ASYNC_B"] = async_b
        if (
            canonical
            and ep in (2, 4)
            and 2048 <= tokens <= 16384
            and torch.cuda.get_device_properties(device).name == "PPU-ZW810E"
        ):
            options["ROLE_BUDGET"] = 80
        options["L1_PIPELINE"] = _use_l1_pipeline(config)
        options["L1_ASYNC"] = options["L1_PIPELINE"] and config.tokens >= 2048
        if options["L1_PIPELINE"] and config.num_ranks == 8:
            options["ROLE_BUDGET"] = 80
        options["CACHE_TOKENS"] = cache_tokens
        options["PARTIAL_REDUCE"] = partial_reduce
        options["STATIC_GEMM2"] = (
            vendor_numerics and gather_input and ep == 2 and tokens == 16384
        )
        if options["STATIC_GEMM2"]:
            options["ROLE_RETURN_BATCH"] = 2
            options["RANK_RETURN"] = True
            options["RANK_RETURN_WORKERS"] = return_workers or 28
            options["BULK_DISPATCH_READY"] = True
            options["STATIC_DISPATCH_META"] = True
            options["ROLE_BUDGET"] = 80
            options["L2_BATCH"] = 2
            options["L2_GROUP_M"] = 8
            options["LOCAL_INVERSE"] = True
            options["CACHE_WORKERS"] = 3
            options["CACHE_DECOUPLED"] = True
            options["LOCAL_G1_FIRST"] = True
        optimize_dispatch = (
            canonical and torch.cuda.get_device_properties(device).name == "PPU-ZW810E"
        )
        if optimize_dispatch and 24 <= tokens <= 512:
            options["ATOMIC_DISPATCH"] = True
            options["COMPACT_META"] = True
            options["COPY_K"] = 256
        if gather_input:
            options["ATOMIC_DISPATCH"] = True
        if optimize_dispatch and ep in (4, 8) and tokens in (2048, 16384):
            options["ATOMIC_DISPATCH"] = True
            options["COUNT_CHUNK"] = 256
            options["STATIC_PAYLOAD"] = True
            options["COPY_K"] = 256
        options["DEFER_ROUTER"] = vendor_numerics or (canonical and ep == 8 and tokens <= 128)
        options["VENDOR_NUMERICS"] = vendor_numerics
        if dispatch_workers:
            if dispatch_workers >= options["ROLE_BUDGET"]:
                raise ValueError("dispatch_workers must leave at least one GEMM CTA")
            options.update(
                COMPACT_META=True,
                ATOMIC_DISPATCH=True,
                COPY_K=256,
                COPY_WORKERS=dispatch_workers,
                COPY_M_BLOCK=dispatch_m_block,
            )

        if vendor_numerics and optimize_dispatch and ep == 8:
            if 136 <= tokens <= 512 and dispatch_m_block:
                options.update(
                    COUNT_CHUNK=128,
                    STATIC_PAYLOAD=True,
                    STATIC_DISPATCH_META=True,
                    DST_PTR64=True,
                )
            if tokens == 16384:
                options["G2_COMBINE_ROWS"] = 2

        options["REDUCE_ROWS"] = 2 if partial_reduce else 1
        if partial_reduce:
            options["USE_DST_PTR"] = False
            options["GATHER_INPUT"] = gather_input
            options["NL2N"] = config.hidden_size // (
                128 if config.tokens == 16384 else 256
            )
            options["L2_STAGES"] = 3
            if options["STATIC_GEMM2"]:
                options["COMPACT_META"] = True
                options["L1_QUANT_ROWS"] = 32
        if canonical and ep == 2 and tokens <= 8:
            options["PACK_ROUTE_META"] = True
        options["WARP_HIST"] = (
            canonical
            and ep == 2
            and tokens <= 16384
            and options["PACK_ROUTE_META"]
            and options["ROUTE_CHUNK"] == 32
            and options["num_warps"] == 4
            and torch.cuda.get_device_properties(device).name == "PPU-ZW810E"
        )
        if ep4_cache_workers:
            options.update(
                CACHE_DECOUPLED=True,
                LOCAL_G1_FIRST=True,
                GATHER_INPUT=True,
                BULK_DISPATCH_READY=True,
                STATIC_DISPATCH_META=True,
                COMPACT_META=True,
                CACHE_WORKERS=ep4_cache_workers,
                ATOMIC_DISPATCH=True,
                STATIC_GEMM2=True,
                L2_BATCH=2,
                L2_GROUP_M=8,
                AGGREGATE_COMPLETION=True,
            )
        if ep4_rank_return:
            options.update(
                RANK_RETURN=True,
                LOCAL_INVERSE=True,
                RANK_RETURN_WORKERS=48,
                A_BLOCK_PTR=True,
                ASYNC_A=True,
                ASYNC_B=True,
                L2_STAGES=2,
            )
        cta_role_megakernel[(options["ROLE_BUDGET"],)](
            *(arguments[name] for name in _KERNEL_ARGUMENTS), **options
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

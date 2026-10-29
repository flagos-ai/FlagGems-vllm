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

"""Caller-owned API for the Hygon W8A8 expert-parallel MegaMoE kernel.

This module performs metadata validation and launches one persistent Triton
kernel. It deliberately does not initialize DUSHMEM, allocate or clear device
memory, prepare weights, copy tensors, or invoke a PyTorch compute operator.
The application must create every tensor in :class:`MegaMoEContext`, build the
peer-pointer tables, and initialize the barrier state collectively.

The context is reusable only for serial calls on one stream. HIP/CUDA graph
capture and replay are not supported: the host-owned epoch must advance once
per collective launch, and every rank must observe launches in the same order.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from . import kernel as _kernel

_NUM_LOCAL_PHASES = 9
_NUM_RANK_PHASES = 2
_MAX_SAFE_EPOCH = (1 << 31) - 2

_BLOCK_M_CHOICES = (16, 32, 64, 128)
_BLOCK_N_CHOICES = (32, 64, 128)
_BLOCK_K_CHOICES = (32, 64, 128)
_BLOCK_Q_CHOICES = (16, 32, 64, 128, 256)
_BLOCK_R_CHOICES = (4, 8, 16, 32, 64, 128)
_NUM_WARPS_CHOICES = (2, 4, 8)


@dataclass(frozen=True)
class MegaMoETileConfig:
    """Compile-time tile configuration for one gfx936 specialization."""

    block_m: int
    block_n: int
    block_k: int
    block_q: int
    block_r: int
    num_warps: int


def default_tile_config(tokens_per_rank: int) -> MegaMoETileConfig:
    """Return the measured four-bucket default for one rank's token count.

    This helper is pure Python. It neither queries a device nor allocates a
    tensor, so callers may use it while constructing a reusable context.
    """

    _require_positive_int(tokens_per_rank, "tokens_per_rank")
    if tokens_per_rank <= 4:
        return MegaMoETileConfig(32, 64, 128, 64, 32, 4)
    if tokens_per_rank <= 64:
        return MegaMoETileConfig(32, 128, 128, 64, 32, 4)
    if tokens_per_rank <= 512:
        return MegaMoETileConfig(32, 128, 128, 64, 32, 8)
    if tokens_per_rank <= 16384:
        return MegaMoETileConfig(128, 128, 128, 64, 64, 8)
    raise ValueError("no measured default tile for tokens_per_rank > 16384")


@dataclass(frozen=True)
class MegaMoEWeights:
    """Caller-prepacked rank-local W8A8 expert weights.

    Experts are the contiguous global range owned by this rank. The INT8
    layouts are ``w1_*_t=[E_local,H,I]`` and ``w2_t=[E_local,I,H]``. The FP32
    per-output-channel scales are ``[E_local,I]``, ``[E_local,I]``, and
    ``[E_local,H]`` respectively. All tensors must be contiguous and already
    reside on the input device.
    """

    w1_gate_t: torch.Tensor
    w1_up_t: torch.Tensor
    w2_t: torch.Tensor
    w1_gate_scale: torch.Tensor
    w1_up_scale: torch.Tensor
    w2_scale: torch.Tensor


@dataclass
class MegaMoEContext:
    """Reusable caller-owned communication and scratch workspace.

    DUSHMEM setup, symmetric allocation, peer mapping, tensor construction,
    initial zeroing, and finalization are application responsibilities. The
    five payload tensors plus ``flags_local`` are views into live symmetric
    allocations. Each ``*_table`` is a contiguous device INT64 array of peer
    addresses. Local scratch and ``output`` must remain alive through the
    asynchronous launch.

    ``architecture`` must identify gfx936 (an optional feature suffix after
    ``:`` is accepted), ``world_size`` must be 2, 4, or 8, and experts are
    sharded as contiguous equal ranges. ``ncta`` is capped at 80 because the
    software grid barriers require every CTA to remain resident on the
    validated 80-CU target.

    The same context must not be used concurrently or from multiple streams.
    It cannot be captured into or replayed from a HIP/CUDA graph.

    ``sync_status`` is a sticky device-side failure flag. Because the forward
    is asynchronous, callers that need to detect a collective timeout must
    call :meth:`raise_if_sync_failed` after the stream is synchronized and
    before consuming the result. A failed context is poisoned and cannot be
    reused.
    """

    rank: int
    world_size: int
    tokens_per_rank: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    architecture: str
    ncta: int
    sync_max_spins: int
    tile: MegaMoETileConfig

    # Symmetric allocation views owned by the caller.
    q_local: torch.Tensor
    qscale_local: torch.Tensor
    ids_local: torch.Tensor
    route_weights_local: torch.Tensor
    partials_local: torch.Tensor
    flags_local: torch.Tensor

    # Device INT64 peer-pointer tables owned by the caller.
    q_table: torch.Tensor
    qscale_table: torch.Tensor
    ids_table: torch.Tensor
    route_weights_table: torch.Tensor
    partials_table: torch.Tensor
    flag_table: torch.Tensor

    # Rank-local scratch owned by the caller.
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    expert_write: torch.Tensor
    expert_block_offsets: torch.Tensor
    route_queue: torch.Tensor
    gathered_q: torch.Tensor
    block_to_expert: torch.Tensor
    gathered_qscale: torch.Tensor
    queue_route_weight: torch.Tensor
    activation_tmp: torch.Tensor
    activation_amax: torch.Tensor
    q2: torch.Tensor
    q2_scale: torch.Tensor
    owner_accum: torch.Tensor
    phase_done: torch.Tensor
    phase_released: torch.Tensor
    rank_released: torch.Tensor
    sync_status: torch.Tensor
    output: torch.Tensor

    epoch: int = 0
    lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    _poisoned: bool = field(default=False, init=False, repr=False, compare=False)

    def raise_if_sync_failed(self) -> None:
        """Synchronize and raise if the persistent collective timed out.

        This explicit diagnostic is intentionally outside the fast launch
        path. It performs a device synchronization and a scalar device-to-host
        read, so applications may call it at their existing synchronization
        boundary rather than paying that cost inside every forward.
        """

        with self.lock:
            if self._poisoned:
                raise RuntimeError(
                    "MegaMoE context is poisoned by an earlier launch failure"
                )
            try:
                torch.cuda.synchronize(self.sync_status.device)
                status = int(self.sync_status.item())
            except Exception:
                self._poisoned = True
                raise
            if status != 0:
                self._poisoned = True
                raise RuntimeError(
                    "MegaMoE reported an internal synchronization timeout; "
                    "discard the output and context"
                )


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _require_tensor(
    tensor: object,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_static_contract(context: MegaMoEContext) -> None:
    for name in (
        "world_size",
        "tokens_per_rank",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "ncta",
        "sync_max_spins",
    ):
        _require_positive_int(getattr(context, name), name)
    if isinstance(context.rank, bool) or not isinstance(context.rank, int):
        raise TypeError("rank must be an integer")
    if context.world_size not in (2, 4, 8):
        raise ValueError("world_size must be one of 2, 4, or 8")
    if not 0 <= context.rank < context.world_size:
        raise ValueError("rank must be in [0, world_size)")
    if context.num_experts % context.world_size:
        raise ValueError("contiguous expert shards must have equal size")
    if context.top_k > context.num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    if not isinstance(context.architecture, str):
        raise TypeError("architecture must be a string")
    if context.architecture.split(":", 1)[0] != "gfx936":
        raise NotImplementedError("Hygon MegaMoE is validated only on gfx936")
    if context.ncta > 80:
        raise ValueError("ncta must not exceed the validated 80-CU resident grid")
    if not isinstance(context.tile, MegaMoETileConfig):
        raise TypeError("tile must be MegaMoETileConfig")

    tile_choices = {
        "block_m": _BLOCK_M_CHOICES,
        "block_n": _BLOCK_N_CHOICES,
        "block_k": _BLOCK_K_CHOICES,
        "block_q": _BLOCK_Q_CHOICES,
        "block_r": _BLOCK_R_CHOICES,
        "num_warps": _NUM_WARPS_CHOICES,
    }
    for name, choices in tile_choices.items():
        value = getattr(context.tile, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in choices
        ):
            raise ValueError(f"tile.{name} must be one of {choices}")

    hidden = context.hidden_size
    intermediate = context.intermediate_size
    if hidden % 32 or intermediate % 32:
        raise ValueError("hidden_size and intermediate_size must be multiples of 32")
    if hidden % context.tile.block_q:
        raise ValueError("tile.block_q must divide hidden_size")
    if hidden % context.tile.block_k or intermediate % context.tile.block_k:
        raise ValueError("tile.block_k must divide both model dimensions")
    if intermediate % context.tile.block_n or hidden % context.tile.block_n:
        raise ValueError("tile.block_n must divide both model dimensions")
    if isinstance(context.epoch, bool) or not isinstance(context.epoch, int):
        raise TypeError("epoch must be an integer")
    if context.epoch < 0 or context.epoch >= _MAX_SAFE_EPOCH:
        raise RuntimeError("context epoch space is exhausted")
    if not hasattr(context.lock, "__enter__") or not hasattr(context.lock, "__exit__"):
        raise TypeError("context.lock must be a context-manager lock")


def _validate_tensors(
    input_int8: torch.Tensor,
    input_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoEWeights,
    context: MegaMoEContext,
    output: torch.Tensor,
) -> None:
    if not isinstance(input_int8, torch.Tensor):
        raise TypeError("input_int8 must be a torch.Tensor")
    device = input_int8.device
    if device.type != "cuda":
        raise ValueError("all tensors must reside on a CUDA-alike Hygon device")

    tokens = context.tokens_per_rank
    hidden = context.hidden_size
    intermediate = context.intermediate_size
    top_k = context.top_k
    experts_per_rank = context.num_experts // context.world_size
    total_tokens = context.world_size * tokens
    total_routes = total_tokens * top_k
    n_intermediate_tiles = intermediate // context.tile.block_n

    _require_tensor(input_int8, "input_int8", (tokens, hidden), torch.int8, device)
    _require_tensor(input_scales, "input_scales", (tokens, 1), torch.float32, device)
    _require_tensor(topk_ids, "topk_ids", (tokens, top_k), torch.int64, device)
    _require_tensor(
        topk_weights, "topk_weights", (tokens, top_k), torch.float32, device
    )
    _require_tensor(output, "output", (tokens, hidden), torch.bfloat16, device)

    if not isinstance(weights, MegaMoEWeights):
        raise TypeError("weights must be MegaMoEWeights")
    weight_specs = (
        (
            weights.w1_gate_t,
            "weights.w1_gate_t",
            (experts_per_rank, hidden, intermediate),
            torch.int8,
        ),
        (
            weights.w1_up_t,
            "weights.w1_up_t",
            (experts_per_rank, hidden, intermediate),
            torch.int8,
        ),
        (
            weights.w2_t,
            "weights.w2_t",
            (experts_per_rank, intermediate, hidden),
            torch.int8,
        ),
        (
            weights.w1_gate_scale,
            "weights.w1_gate_scale",
            (experts_per_rank, intermediate),
            torch.float32,
        ),
        (
            weights.w1_up_scale,
            "weights.w1_up_scale",
            (experts_per_rank, intermediate),
            torch.float32,
        ),
        (
            weights.w2_scale,
            "weights.w2_scale",
            (experts_per_rank, hidden),
            torch.float32,
        ),
    )
    for tensor, name, shape, dtype in weight_specs:
        _require_tensor(tensor, name, shape, dtype, device)

    symmetric_specs = (
        (context.q_local, "context.q_local", (tokens, hidden), torch.int8),
        (context.qscale_local, "context.qscale_local", (tokens,), torch.float32),
        (context.ids_local, "context.ids_local", (tokens, top_k), torch.int32),
        (
            context.route_weights_local,
            "context.route_weights_local",
            (tokens, top_k),
            torch.float32,
        ),
        (
            context.partials_local,
            "context.partials_local",
            (tokens, top_k, hidden),
            torch.bfloat16,
        ),
        (
            context.flags_local,
            "context.flags_local",
            (_NUM_RANK_PHASES, context.world_size),
            torch.int32,
        ),
    )
    for tensor, name, shape, dtype in symmetric_specs:
        _require_tensor(tensor, name, shape, dtype, device)

    for name in (
        "q_table",
        "qscale_table",
        "ids_table",
        "route_weights_table",
        "partials_table",
        "flag_table",
    ):
        _require_tensor(
            getattr(context, name),
            f"context.{name}",
            (context.world_size,),
            torch.int64,
            device,
        )

    scratch_specs = (
        (
            context.expert_counts,
            "context.expert_counts",
            (experts_per_rank,),
            torch.int32,
        ),
        (
            context.expert_offsets,
            "context.expert_offsets",
            (experts_per_rank + 1,),
            torch.int32,
        ),
        (
            context.expert_write,
            "context.expert_write",
            (experts_per_rank,),
            torch.int32,
        ),
        (
            context.expert_block_offsets,
            "context.expert_block_offsets",
            (experts_per_rank + 1,),
            torch.int32,
        ),
        (context.route_queue, "context.route_queue", (total_routes,), torch.int32),
        (context.gathered_q, "context.gathered_q", (total_tokens, hidden), torch.int8),
        (
            context.block_to_expert,
            "context.block_to_expert",
            (total_routes,),
            torch.int32,
        ),
        (
            context.gathered_qscale,
            "context.gathered_qscale",
            (total_tokens,),
            torch.float32,
        ),
        (
            context.queue_route_weight,
            "context.queue_route_weight",
            (total_routes,),
            torch.float32,
        ),
        (
            context.activation_tmp,
            "context.activation_tmp",
            (total_routes, intermediate),
            torch.bfloat16,
        ),
        (
            context.activation_amax,
            "context.activation_amax",
            (total_routes, n_intermediate_tiles),
            torch.float32,
        ),
        (context.q2, "context.q2", (total_routes, intermediate), torch.int8),
        (context.q2_scale, "context.q2_scale", (total_routes,), torch.float32),
        (context.owner_accum, "context.owner_accum", (1,), torch.float32),
        (
            context.phase_done,
            "context.phase_done",
            (_NUM_LOCAL_PHASES, context.ncta),
            torch.int32,
        ),
        (
            context.phase_released,
            "context.phase_released",
            (_NUM_LOCAL_PHASES,),
            torch.int32,
        ),
        (
            context.rank_released,
            "context.rank_released",
            (_NUM_RANK_PHASES,),
            torch.int32,
        ),
        (context.sync_status, "context.sync_status", (1,), torch.int32),
        (context.output, "context.output", (tokens, hidden), torch.bfloat16),
    )
    for tensor, name, shape, dtype in scratch_specs:
        _require_tensor(tensor, name, shape, dtype, device)


def megamoe_int8(
    input_int8: torch.Tensor,
    input_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoEWeights,
    context: MegaMoEContext,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch one asynchronous distributed W8A8 SwiGLU MegaMoE forward.

    Inputs are prequantized INT8 ``[T,H]`` with contiguous FP32 scales
    ``[T,1]``. Routes are INT64 and weights are FP32 ``[T,top_k]``. Expert IDs
    use contiguous global sharding; callers must supply values in
    ``[-1,num_experts)`` where ``-1`` drops a route. The returned/output tensor
    is BF16 ``[T,H]``. ``output=None`` selects the caller-owned
    ``context.output`` and never allocates memory.

    Every rank must invoke this function in the same order. The call enqueues
    exactly one kernel launch and returns asynchronously. Graph capture and
    replay are unsupported; reuse is serial on one stream only. The launch is
    guarded to the input tensor's Hygon device and restores the caller's
    current device afterward. Call ``context.raise_if_sync_failed()`` at a
    quiescent health-check boundary before consuming results when collective
    timeout detection is required; all ranks should check at the same point.
    """

    if not isinstance(context, MegaMoEContext):
        raise TypeError("context must be MegaMoEContext")
    result = context.output if output is None else output
    _validate_static_contract(context)
    _validate_tensors(
        input_int8,
        input_scales,
        topk_ids,
        topk_weights,
        weights,
        context,
        result,
    )
    with context.lock:
        if context._poisoned:
            raise RuntimeError("MegaMoE context is poisoned and cannot be reused")
        if context.epoch >= _MAX_SAFE_EPOCH:
            raise RuntimeError("context epoch space is exhausted")
        epoch = context.epoch + 1
        context.epoch = epoch

        tile = context.tile
        n_intermediate_tiles = context.intermediate_size // tile.block_n
        n_hidden_tiles = context.hidden_size // tile.block_n
        try:
            with torch.cuda.device(input_int8.device):
                _kernel.hcu_megamoe[(context.ncta,)](
                    input_int8,
                    input_scales,
                    topk_weights,
                    topk_ids,
                    weights.w1_gate_t,
                    weights.w1_up_t,
                    weights.w1_gate_scale,
                    weights.w1_up_scale,
                    weights.w2_t,
                    weights.w2_scale,
                    result,
                    context.q_local,
                    context.qscale_local,
                    context.ids_local,
                    context.route_weights_local,
                    context.partials_local,
                    context.q_table,
                    context.qscale_table,
                    context.ids_table,
                    context.route_weights_table,
                    context.partials_table,
                    context.flag_table,
                    context.expert_counts,
                    context.expert_offsets,
                    context.expert_write,
                    context.expert_block_offsets,
                    context.route_queue,
                    context.gathered_q,
                    context.block_to_expert,
                    context.gathered_qscale,
                    context.queue_route_weight,
                    context.activation_tmp,
                    context.activation_amax,
                    context.q2,
                    context.q2_scale,
                    context.owner_accum,
                    context.phase_done,
                    context.phase_released,
                    context.rank_released,
                    context.sync_status,
                    epoch,
                    MY_PE=context.rank,
                    NPES=context.world_size,
                    NPES_POW2=_next_power_of_two(context.world_size),
                    NCTA=context.ncta,
                    NCTA_POW2=_next_power_of_two(context.ncta),
                    NTOK=context.tokens_per_rank,
                    TOPK=context.top_k,
                    TOPK_POW2=_next_power_of_two(context.top_k),
                    NEXP=context.num_experts,
                    EPR=context.num_experts // context.world_size,
                    HIDDEN=context.hidden_size,
                    INTER=context.intermediate_size,
                    BLOCK_Q=tile.block_q,
                    BLOCK_R=tile.block_r,
                    BLOCK_M=tile.block_m,
                    BLOCK_N=tile.block_n,
                    BLOCK_K=tile.block_k,
                    NI=n_intermediate_tiles,
                    NH=n_hidden_tiles,
                    NI_POW2=_next_power_of_two(n_intermediate_tiles),
                    AGGREGATE_RETURN=False,
                    SYNC_MAX_SPINS=context.sync_max_spins,
                    num_warps=tile.num_warps,
                )
        except Exception:
            context._poisoned = True
            raise
    return result


__all__ = [
    "MegaMoEContext",
    "MegaMoETileConfig",
    "MegaMoEWeights",
    "default_tile_config",
    "megamoe_int8",
]

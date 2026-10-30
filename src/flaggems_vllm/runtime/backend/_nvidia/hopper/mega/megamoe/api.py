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

"""Public data contract for the Hopper FP8 MegaMoE kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


@dataclass(frozen=True)
class MegaMoEContext:
    """Caller-owned communication state and reusable MegaMoE workspace.

    The caller initializes NVSHMEM, allocates every symmetric buffer
    collectively, builds the peer-pointer tables, and keeps those allocations
    alive while this context is in use. ``barrier_all_on_stream`` must enqueue
    an NVSHMEM all-rank barrier on the supplied CUDA stream; it must not perform
    a host synchronization. Each ``*_ptrs`` value is a contiguous CUDA int64
    tensor containing one directly addressable pointer per rank.

    Symmetric buffers represented by the nine ``*_ptrs`` tables have these
    minimum byte capacities on every rank, where ``R=world_size``,
    ``E=num_experts / R``, ``T=max_tokens``, ``P=max_topk``, ``K=hidden_size``,
    ``S=K / 128``, ``M=max_recv``, and ``L=pool_tokens``::

        token          T * K
        scale          T * S * 4
        route_weight   T * P * 4
        queue          E * R * M * 4
        recv_count     R * E * 8
        recv_sum       E * 8
        combine        P * T * K * 2
        signal         16 * 4
        metadata       L * 3 * 4

    ``token_peer_ptrs`` contains the same token-buffer addresses as host
    integers. The raw D8 path passes those addresses as scalar kernel
    arguments and therefore supports at most eight ranks.

    The embedding runtime configures Triton's descriptor workspace allocator
    once before the first launch. MegaMoE does not replace the process-global
    allocator while executing an operator call.

    Local scratch tensors are contiguous CUDA tensors with these dtypes and
    minimum element counts, where ``N=num_sms``, ``X=num_experts``, and
    ``I=intermediate_size``::

        expert_send_count  int64   X
        l1_arrival_count   int32   L / 128
        l2_arrival_mask    int64   L / 128
        global_counters    int32   8
        metadata_ready     int32   N
        l1_tokens          fp8     L * K
        l1_scales          fp32    S * L
        l1_weights         fp32    L
        l2_activations     fp8     L * I
        l2_scales          fp32    (I / 64) * L

    Before launch, the caller must also guarantee that every
    ``(source rank, destination expert)`` route count is at most ``max_recv``
    and that ``sum(ceil(expert_routes / 128) * 128)`` on every destination
    rank is at most ``pool_tokens``. A model runtime can satisfy this with its
    routing planner; a standalone benchmark may compute it from gathered
    ``topk_ids`` before constructing the context.
    """

    rank: int
    world_size: int
    num_sms: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    max_tokens: int
    max_topk: int
    max_recv: int
    pool_tokens: int

    token_ptrs: torch.Tensor
    scale_ptrs: torch.Tensor
    route_weight_ptrs: torch.Tensor
    queue_ptrs: torch.Tensor
    recv_count_ptrs: torch.Tensor
    recv_sum_ptrs: torch.Tensor
    combine_ptrs: torch.Tensor
    signal_ptrs: torch.Tensor
    metadata_ptrs: torch.Tensor
    combine_local: int
    token_peer_ptrs: Tuple[int, ...]

    expert_send_count: torch.Tensor
    l1_arrival_count: torch.Tensor
    l2_arrival_mask: torch.Tensor
    global_counters: torch.Tensor
    metadata_ready: torch.Tensor
    l1_tokens: torch.Tensor
    l1_scales: torch.Tensor
    l1_weights: torch.Tensor
    l2_activations: torch.Tensor
    l2_scales: torch.Tensor

    barrier_all_on_stream: Callable[[torch.cuda.Stream], None]


def megamoe_fp8(
    input_fp8: torch.Tensor,
    input_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    context: MegaMoEContext,
    output: Optional[torch.Tensor] = None,
    *,
    stages: int = 4,
) -> torch.Tensor:
    """Launch one asynchronous expert-parallel MegaMoE step.

    All ranks in ``context`` must call this function in the same order. Input
    tensors and prepacked expert weights must already reside on the current
    CUDA device. NVSHMEM initialization and context destruction belong to the
    caller and are never performed here.

    ``input_fp8`` is ``[T, K]`` FP8, ``input_scales`` is ``[T, K/128]`` FP32,
    and routing tensors are ``[T, topk]``. Each rank supplies its expert shard
    as ``w1=[E, 2I, K]``, ``w1_scales=[E, I/64, K/128]``, ``w2=[E, K, I]``,
    and ``w2_scales=[E, K/128, I/128]``. W1 rows use the 8-row gate/up
    interleave expected by the kernel.
    """

    from .kernel import launch_megamoe_fp8 as _launch_megamoe_fp8

    return _launch_megamoe_fp8(
        input_fp8,
        input_scales,
        topk_ids,
        topk_weights,
        w1,
        w1_scales,
        w2,
        w2_scales,
        context,
        output,
        stages=stages,
    )


__all__ = ["MegaMoEContext", "megamoe_fp8"]

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

"""Define the caller-owned MUSA MegaMoE contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple

import torch


@dataclass
class MegaMoEContext:
    """Hold epoch and prepare caches in this Context, so it is not frozen.

    One persistent context serves a monotonic epoch stream, with every rank
    calling in the same order.

    Let T be the local token count, H the hidden dimension, I the intermediate
    dimension, E the local expert count, R the rank count, TOP the route count,
    CAP the queue capacity per destination expert, and P = 128 * POOL. All ranks
    use identical shapes, configuration and epoch order. Only H=2048, I=512,
    E*R=256, TOP=8, R in {1,2,4,8}, CTAS=60 and EPW=16 retain the source's
    validated configuration; no claim is made for additional shapes.

    Each *_ptrs tensor is contiguous device int64 with R entries (8*R bytes).
    It holds one accessible symmetric address per rank in rank order. The
    referenced allocations have these minimum byte capacities::

        input_ptrs          T * H                     fp8
        input_scale_ptrs    4 * T * (H // 128)         fp32
        route_weight_ptrs   4 * T * TOP               fp32
        queue_ptrs          4 * E * CAP               int32
        count_ptrs          4 * E                     int32
        partial_ptrs        2 * T * TOP * H           bf16
        signal_ptrs         12                        int32

    Local contiguous scratch has these minimum byte capacities, where
    P = 128 * POOL and S = max(2*P, CAP + R*T)::

        grid                    40                            int32
        input_workspace         S * H                         fp8
        input_scale_workspace   4 * S * (H // 128)             fp32
        route_weight_workspace  4 * P                         fp32
        quant_workspace         2 * P * I                     fp8
        quant_scale_workspace   4 * max(2*P*(I//64),
                                P*(I//64)+32*POOL+R*T*TOP)     fp32
        ready                   4 * POOL                      int32
        quant_ready             4 * POOL                      int32

    QuantScale also stores float32 task metadata and counters, so integer
    metadata must remain exactly representable.

    For every (source rank, destination expert), routes <= max_recv, and each
    destination rank must satisfy sum(ceil(routes_to_expert/128)*128) <=
    pool_tokens, where pool_tokens=128*POOL and routes_to_expert sums over source
    ranks. CAP >= R*max_recv, P >= CAP, and the total padded 32-row tasks must
    fit 8*POOL metadata slots. The caller must prove these bounds before launch;
    the operator does not inspect device routing data on the host.

    Before the first call, zero grid and all three local symmetric signals.
    Reuse the same persistent buffers, CTAS and rank group for the entire
    process epoch stream; do not reset counters between calls. All CTAs must
    reside concurrently. Calls are serialized on the caller's current stream;
    barrier is a caller-provided host rendezvous callback shared by all ranks.
    A failed launch invalidates this context and the distributed epoch stream.

    Public tensors must match the local entries of their symmetric pointer
    tables. Weights are caller-prepacked gu256: w1 is
    [E,2,I//128,128,H].permute(0,2,1,3,4).contiguous().reshape(E*2*I,H);
    w2 is contiguous [E*H,I]. Their FP32 block-128 scales retain the original
    expert/gate/up order. The caller owns all buffers and tensor lifetimes.
    The six descriptor fields a_descriptor, as_descriptor, b1_descriptor,
    b2_descriptor, q_descriptor and qs_descriptor plus the eight local scratch
    fields above replace the fourteen original workspace operands. Descriptors
    carry no independent byte allocation; they borrow the corresponding scratch
    or packed-weight storage. Shape/descriptor allocations must have complete
    rows (H, I, or 64 elements as applicable). Descriptors and compiled warmups
    belong to this context, not a global cache.
    """

    input_ptrs: torch.Tensor
    input_scale_ptrs: torch.Tensor
    route_weight_ptrs: torch.Tensor
    queue_ptrs: torch.Tensor
    count_ptrs: torch.Tensor
    partial_ptrs: torch.Tensor
    signal_ptrs: torch.Tensor
    grid: torch.Tensor
    input_workspace: torch.Tensor
    input_scale_workspace: torch.Tensor
    route_weight_workspace: torch.Tensor
    quant_workspace: torch.Tensor
    quant_scale_workspace: torch.Tensor
    ready: torch.Tensor
    quant_ready: torch.Tensor
    barrier: Callable[[], None]
    a_descriptor: Any = field(default=None, init=False, repr=False)
    as_descriptor: Any = field(default=None, init=False, repr=False)
    b1_descriptor: Any = field(default=None, init=False, repr=False)
    b2_descriptor: Any = field(default=None, init=False, repr=False)
    q_descriptor: Any = field(default=None, init=False, repr=False)
    qs_descriptor: Any = field(default=None, init=False, repr=False)
    H: int = 2048
    I: int = 512
    E: int = 256
    TOP: int = 8
    CTAS: int = 60
    CAP: int = 0
    MY: int = 0
    R: int = 1
    POOL: int = 0
    EPW: int = 16
    max_recv: int = 0
    pack: str = "gu256"
    a_tile: Tuple[int, int] = (32, 128)
    b1_tile: Tuple[int, int] = (256, 128)
    b2_tile: Tuple[int, int] = (256, 128)
    q_tile: Tuple[int, int] = (64, 64)
    _prepared: Dict[Tuple[int, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_epoch: int = field(default=0, init=False, repr=False)
    _epoch: ClassVar[int] = 0
    _owner: ClassVar[Any] = None

    @property
    def pool_tokens(self) -> int:
        """Return the padded route capacity of this context."""
        return 128 * self.POOL

    def next_epoch(self) -> int:
        """Advance the single persistent process epoch stream without resetting it."""
        cls = MegaMoEContext
        if cls._owner is not None and cls._owner is not self:
            raise RuntimeError(
                "A process epoch stream requires one persistent MegaMoEContext"
            )
        if cls._epoch >= (2**31 - 1) // max(self.CTAS, self.R, 1):
            raise RuntimeError("MegaMoE int32 rendezvous counters would overflow")
        cls._owner = self
        cls._epoch += 1
        self._last_epoch = cls._epoch
        return cls._epoch


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
    """Launch one fused FP8 MoE call with caller-owned symmetric workspaces.

    Input is contiguous [T,H] FP8; input scales are FP32 [T,H//128],
    topk_ids int32 [T,TOP] (negative IDs are inactive), and topk_weights
    FP32 [T,TOP]. Packed weights and all capacity/rendezvous preconditions are
    defined by MegaMoEContext. Return BF16 [T,H], allocating output when absent.
    stages=4 is the public API compatibility value; the retained MUSA codegen
    uses num_stages=1 and does not expose a tunable pipeline depth.
    """
    if stages != 4:
        raise ValueError("The retained MUSA implementation requires stages=4")

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
    )


__all__ = ["megamoe_fp8", "MegaMoEContext"]

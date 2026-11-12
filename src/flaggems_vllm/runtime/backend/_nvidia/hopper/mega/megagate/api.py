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

"""Public Hopper MegaGate API.

The context owns compiled exact-shape kernels and reusable workspaces.  Model
initialization calls :meth:`MegaGateContext.prepare` once per ``(T, stream)``
outside the timed path;
each :func:`megagate` call then performs constant-time dispatch and launches
exactly one fused GEMM-and-routing kernel.  The return order is
``(topk_weights, topk_indices)``.  Passing ``out=None`` includes output
allocation in the call; pass preallocated output tensors for an
allocation-free hot path.
"""

from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.dsv4 import (
    EXPERTS,
    HIDDEN,
    PROFILE,
    ROUTED_SCALING_FACTOR,
    SUPPORTED_TOKENS,
    TOPK,
    MegaGateContext,
    MegaGateContextError,
    UnpreparedShapeError,
    megagate,
)


def create_megagate_context(*, profile: str = PROFILE) -> MegaGateContext:
    """Create a caller-owned MegaGate context for a supported model profile."""

    return MegaGateContext(profile=profile)


__all__ = [
    "EXPERTS",
    "HIDDEN",
    "MegaGateContext",
    "MegaGateContextError",
    "PROFILE",
    "ROUTED_SCALING_FACTOR",
    "SUPPORTED_TOKENS",
    "TOPK",
    "UnpreparedShapeError",
    "create_megagate_context",
    "megagate",
]

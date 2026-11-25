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

"""Public exact-shape Hopper MegaGate API.

The legacy profile argument supplies a routing contract and default scaling.
Preparation selects the implementation module and source leaf with the complete
model-independent :class:`ShapeKey`. Launch looks up a prepared binding on the
tensor's device and exact token count, then the host validates the runtime ABI.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Any, Final

import torch

from .errors import MegaGateContextError, UnpreparedShapeError
from .registry import (
    DSV4,
    PROFILES,
    KernelPlan,
    MegaGateProfile,
    ShapeKey,
    get_kernel_plan,
    get_profile,
)

# Backward-compatible constants describe the default profile. New callers
# should query ``context.profile_spec`` rather than relying on these globals.
PROFILE: Final = DSV4.name
SUPPORTED_TOKENS: Final = DSV4.supported_tokens
HIDDEN: Final = DSV4.hidden
EXPERTS: Final = DSV4.experts
TOPK: Final = DSV4.topk
ROUTED_SCALING_FACTOR: Final = DSV4.routed_scaling_factor


def _validate_plan_module(module: ModuleType, plan: KernelPlan) -> None:
    """Fail closed if an implementation no longer matches its registry plan."""

    shape = plan.shape
    actual = (
        module.HIDDEN,
        module.EXPERTS,
        module.TOPK,
        module.ROUTED_SCALING_FACTOR,
    )
    expected = (
        shape.hidden,
        shape.experts,
        shape.topk,
        shape.routed_scaling_factor,
    )
    if actual != expected:
        raise RuntimeError(
            f"MegaGate implementation metadata drift for {shape!r}: "
            f"expected={expected}, actual={actual}"
        )
    if (
        module.MegaGateContextError is not MegaGateContextError
        or module.UnpreparedShapeError is not UnpreparedShapeError
    ):
        raise RuntimeError(f"MegaGate error ABI drift for {shape!r}")


def _load_plan_module(plan: KernelPlan) -> ModuleType:
    """Import the implementation selected by a model-independent plan."""

    module = importlib.import_module(plan.implementation_module)
    _validate_plan_module(module, plan)
    return module


class MegaGateContext:
    """Caller-owned state using one compatibility routing contract."""

    def __init__(self, *, profile: str = PROFILE) -> None:
        self._profile_spec = get_profile(profile)
        self._lock = threading.RLock()
        self._module_bindings: dict[str, Any] = {}
        self._shape_bindings: dict[tuple[torch.device, int], Any] = {}
        self._device_capabilities: dict[torch.device, tuple[int, int]] = {}

    def _capability_for_prepare(self, x: torch.Tensor) -> tuple[int, int]:
        """Query one CUDA device only while preparing, never while launching."""

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.device.type != "cuda":
            # Preserve cheap meta/CPU ABI validation without touching CUDA.
            return (9, 0)
        cached = self._device_capabilities.get(x.device)
        if cached is not None:
            return cached
        capability = torch.cuda.get_device_capability(x.device)
        if tuple(capability) != (9, 0):
            raise RuntimeError("MegaGate requires SM90")
        return (9, 0)

    def _shape_key(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        routed_scaling_factor: float,
        compute_capability: tuple[int, int],
    ) -> ShapeKey:
        """Combine input dimensions with the compatibility routing contract."""

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if not isinstance(gate_weight, torch.Tensor):
            raise TypeError("gate_weight must be a torch.Tensor")
        if x.ndim != 2 or gate_weight.ndim != 2:
            raise ValueError("x and gate_weight must both be rank-2 tensors")
        tokens = int(x.shape[0])
        self._require_supported_tokens(tokens)
        spec = self._profile_spec
        return ShapeKey(
            tokens=tokens,
            hidden=int(x.shape[1]),
            experts=int(gate_weight.shape[0]),
            topk=spec.topk,
            routing_mode=spec.routing_mode,
            uses_correction_bias=spec.uses_correction_bias,
            expert_group_count=spec.expert_group_count,
            topk_group=spec.topk_group,
            renormalize=spec.renormalize,
            routed_scaling_factor=float(routed_scaling_factor),
            input_dtype=str(x.dtype).split(".")[-1],
            compute_capability=compute_capability,
        )

    def _require_supported_tokens(self, tokens: int) -> None:
        if tokens not in self.supported_tokens:
            raise ValueError(
                f"MegaGate profile {self.profile!r} rejects T={tokens}; "
                f"supported={self.supported_tokens}"
            )

    def _context_for_prepare(self, plan: KernelPlan) -> Any:
        inner = self._module_bindings.get(plan.implementation_module)
        if inner is None:
            module = _load_plan_module(plan)
            inner = module.MegaGateContext()
            self._module_bindings[plan.implementation_module] = inner
        return inner

    @property
    def profile(self) -> str:
        return self._profile_spec.name

    @property
    def profile_spec(self) -> MegaGateProfile:
        return self._profile_spec

    @property
    def supported_tokens(self) -> tuple[int, ...]:
        return self._profile_spec.supported_tokens

    @property
    def source_leaf_by_tokens(self) -> dict[int, str]:
        return dict(self._profile_spec.source_leaf_by_tokens)

    @property
    def prepared_tokens(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted({tokens for _, tokens in self._shape_bindings}))

    @property
    def kernel_count_per_call(self) -> int:
        return 1

    def prepare(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float | None = None,
    ) -> Any:
        """Compile and bind one exact shape without launching the hot path."""

        scaling = (
            self._profile_spec.routed_scaling_factor
            if routed_scaling_factor is None
            else routed_scaling_factor
        )
        with self._lock:
            capability = self._capability_for_prepare(x)
            plan = get_kernel_plan(self._shape_key(x, gate_weight, scaling, capability))
            if x.device.type != "cuda":
                raise ValueError("MegaGate requires CUDA inputs")
            inner = self._context_for_prepare(plan)
            prepared = inner.prepare(
                x,
                gate_weight,
                correction_bias,
                routed_scaling_factor=scaling,
                plan=plan,
            )
            self._shape_bindings[x.device, plan.shape.tokens] = inner
            self._device_capabilities[x.device] = capability
            return prepared

    def clear_cached_descriptors(self) -> None:
        """Drop address-bound tensor descriptors retained by this context."""

        with self._lock:
            for inner in self._module_bindings.values():
                inner.clear_cached_descriptors()

    def close(self) -> None:
        """Release host-side compiled-leaf and workspace references."""

        with self._lock:
            for inner in self._module_bindings.values():
                inner.close()
            self._shape_bindings.clear()
            self._module_bindings.clear()
            self._device_capabilities.clear()

    def __getattr__(self, name: str) -> Any:
        # Preserve diagnostic access used by the operator tests without
        # exposing each profile implementation as part of the public API.
        bindings = tuple(self.__dict__.get("_module_bindings", {}).values())
        if len(bindings) != 1:
            raise AttributeError(name)
        return getattr(bindings[0], name)


def create_megagate_context(*, profile: str = PROFILE) -> MegaGateContext:
    """Create a caller-owned context for a registered model profile."""

    return MegaGateContext(profile=profile)


def megagate(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None = None,
    *,
    context: MegaGateContext,
    routed_scaling_factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch and launch one prepared exact-shape fused MegaGate kernel.

    Returns newly allocated ``(topk_weights, topk_indices)`` tensors.
    """

    if not isinstance(context, MegaGateContext):
        raise TypeError("context must be MegaGateContext")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError("x must be rank two")
    tokens = int(x.shape[0])
    device = x.device
    scaling = (
        context.profile_spec.routed_scaling_factor
        if routed_scaling_factor is None
        else routed_scaling_factor
    )
    with context._lock:
        inner = context._shape_bindings.get((device, tokens))
        if inner is None:
            context._require_supported_tokens(tokens)
            if device.type != "cuda":
                raise ValueError("MegaGate requires CUDA inputs")
            raise UnpreparedShapeError(
                f"T={tokens} was not prepared on {device}; call context.prepare first"
            )
        return inner.run(x, gate_weight, correction_bias, routed_scaling_factor=scaling)


__all__ = [
    "EXPERTS",
    "HIDDEN",
    "MegaGateContext",
    "MegaGateContextError",
    "MegaGateProfile",
    "PROFILE",
    "PROFILES",
    "ROUTED_SCALING_FACTOR",
    "SUPPORTED_TOKENS",
    "TOPK",
    "UnpreparedShapeError",
    "create_megagate_context",
    "get_profile",
    "megagate",
]

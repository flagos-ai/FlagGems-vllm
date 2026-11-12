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

"""Public multi-profile Hopper MegaGate API.

Model initialization binds a :class:`MegaGateContext` to one profile. The
profile implementation owns compiled exact-shape kernels and reusable
workspaces; a hot-path call therefore performs one profile-local exact-token
lookup and launches exactly one fused GEMM-and-routing kernel. Preparation
and compilation remain outside the timed path.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Final

import torch

from .errors import MegaGateContextError, UnpreparedShapeError
from .registry import DSV4, PROFILES, MegaGateProfile, get_profile

# Backward-compatible constants describe the default profile. New callers
# should query ``context.profile_spec`` rather than relying on these globals.
PROFILE: Final = DSV4.name
SUPPORTED_TOKENS: Final = DSV4.supported_tokens
HIDDEN: Final = DSV4.hidden
EXPERTS: Final = DSV4.experts
TOPK: Final = DSV4.topk
ROUTED_SCALING_FACTOR: Final = DSV4.routed_scaling_factor


def _load_profile_module(spec: MegaGateProfile) -> ModuleType:
    module = importlib.import_module(spec.module)
    actual = (
        module.PROFILE,
        module.HIDDEN,
        module.EXPERTS,
        module.TOPK,
        module.ROUTED_SCALING_FACTOR,
        tuple(module.SUPPORTED_TOKENS),
    )
    expected = (
        spec.name,
        spec.hidden,
        spec.experts,
        spec.topk,
        spec.routed_scaling_factor,
        spec.supported_tokens,
    )
    if actual != expected:
        raise RuntimeError(
            f"MegaGate profile metadata drift for {spec.name}: "
            f"expected={expected}, actual={actual}"
        )
    actual_dispatch = dict(module.TOKEN_TO_LEAF)
    expected_dispatch = dict(spec.source_leaf_by_tokens)
    if actual_dispatch != expected_dispatch:
        raise RuntimeError(
            f"MegaGate dispatch drift for {spec.name}: "
            f"expected={expected_dispatch}, actual={actual_dispatch}"
        )
    if (
        module.MegaGateContextError is not MegaGateContextError
        or module.UnpreparedShapeError is not UnpreparedShapeError
    ):
        raise RuntimeError(f"MegaGate error ABI drift for {spec.name}")
    return module


class MegaGateContext:
    """Caller-owned context bound to exactly one model profile and device."""

    def __init__(self, *, profile: str = PROFILE) -> None:
        self._profile_spec = get_profile(profile)
        self._module = _load_profile_module(self._profile_spec)
        self._inner = self._module.MegaGateContext()

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
        return tuple(self._inner.prepared_tokens)

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
        return self._inner.prepare(
            x,
            gate_weight,
            correction_bias,
            routed_scaling_factor=scaling,
        )

    def clear_cached_descriptors(self) -> None:
        """Drop address-bound tensor descriptors retained by this context."""

        clear = getattr(self._inner, "clear_cached_descriptors", None)
        if clear is None:
            clear = getattr(self._inner, "clear_cached_input_descriptors", None)
        if clear is not None:
            clear()

    def close(self) -> None:
        """Release host-side compiled-leaf and workspace references."""

        self._inner.close()

    def __getattr__(self, name: str) -> Any:
        # Preserve diagnostic access used by the operator tests without
        # exposing each profile implementation as part of the public API.
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


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
    """Launch one prepared profile-specific fused MegaGate kernel.

    Returns newly allocated ``(topk_weights, topk_indices)`` tensors.
    """

    if not isinstance(context, MegaGateContext):
        raise TypeError("context must be MegaGateContext")
    scaling = (
        context.profile_spec.routed_scaling_factor
        if routed_scaling_factor is None
        else routed_scaling_factor
    )
    return context._module.megagate(
        x,
        gate_weight,
        correction_bias,
        context=context._inner,
        routed_scaling_factor=scaling,
    )


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

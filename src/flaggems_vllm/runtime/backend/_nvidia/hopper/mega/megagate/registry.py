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

"""Auditable exact-shape registry for Hopper MegaGate.

The ``vNNN`` values are source provenance, not runtime module names.  Runtime
dispatch performs one lookup in :data:`DISPATCH_TABLE` using the complete
shape and routing contract.  Model profiles remain as compatibility aliases
for callers, but their names are deliberately absent from the dispatch key.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from .types import KernelPlan, RoutingMode, ShapeKey


@dataclass(frozen=True)
class MegaGateProfile:
    """Compatibility view over one static shape-and-routing contract."""

    name: str
    hidden: int
    experts: int
    topk: int
    routed_scaling_factor: float
    routing: str
    uses_correction_bias: bool
    expert_group_count: int | None
    topk_group: int | None
    renormalize: bool
    supported_tokens: tuple[int, ...]

    @property
    def routing_mode(self) -> RoutingMode:
        return RoutingMode(self.routing)

    def shape_key(self, tokens: int) -> ShapeKey:
        """Build the model-independent dispatch key for one exact shape."""

        return ShapeKey(
            tokens=tokens,
            hidden=self.hidden,
            experts=self.experts,
            topk=self.topk,
            routing_mode=self.routing_mode,
            uses_correction_bias=self.uses_correction_bias,
            expert_group_count=self.expert_group_count,
            topk_group=self.topk_group,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    @property
    def source_leaf_by_tokens(self) -> Mapping[int, str]:
        """Return the legacy provenance view derived from exact-shape plans."""

        return MappingProxyType(
            {
                tokens: get_kernel_plan(self.shape_key(tokens)).source_leaf
                for tokens in self.supported_tokens
            }
        )


@dataclass(frozen=True)
class _KernelFamilyKey:
    """Implementation family identity with no model or profile metadata."""

    routing_mode: RoutingMode
    hidden: int
    experts: int
    topk: int
    uses_correction_bias: bool
    expert_group_count: int | None
    topk_group: int | None
    renormalize: bool
    routed_scaling_factor: float
    input_dtype: str = "bfloat16"
    compute_capability: tuple[int, int] = (9, 0)


def _family_key(shape: ShapeKey) -> _KernelFamilyKey:
    return _KernelFamilyKey(
        routing_mode=shape.routing_mode,
        hidden=shape.hidden,
        experts=shape.experts,
        topk=shape.topk,
        uses_correction_bias=shape.uses_correction_bias,
        expert_group_count=shape.expert_group_count,
        topk_group=shape.topk_group,
        renormalize=shape.renormalize,
        routed_scaling_factor=shape.routed_scaling_factor,
        input_dtype=shape.input_dtype,
        compute_capability=shape.compute_capability,
    )


_PLAN_SEEDS: list[tuple[ShapeKey, str]] = []


def _profile(
    *,
    name: str,
    hidden: int,
    experts: int,
    topk: int,
    routed_scaling_factor: float,
    routing: str,
    uses_correction_bias: bool,
    expert_group_count: int | None = None,
    topk_group: int | None = None,
    renormalize: bool = True,
    leaves: Mapping[int, str],
) -> MegaGateProfile:
    profile = MegaGateProfile(
        name=name,
        hidden=hidden,
        experts=experts,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        routing=routing,
        uses_correction_bias=uses_correction_bias,
        expert_group_count=expert_group_count,
        topk_group=topk_group,
        renormalize=renormalize,
        supported_tokens=tuple(leaves),
    )
    _PLAN_SEEDS.extend(
        (profile.shape_key(tokens), source_leaf)
        for tokens, source_leaf in leaves.items()
    )
    return profile


DSV4: Final = _profile(
    name="dsv4_flash",
    hidden=4096,
    experts=256,
    topk=6,
    routed_scaling_factor=1.5,
    routing="sqrt_softplus_bias",
    uses_correction_bias=True,
    leaves={
        1028: "v363",
        2048: "v363",
        3076: "v363",
        3805: "v363",
        4100: "v363",
        4835: "v363",
        7884: "v363",
        8221: "v363",
        13021: "v376",
        16384: "v376",
    },
)

DSV32: Final = _profile(
    name="dsv32",
    hidden=7168,
    experts=256,
    topk=8,
    routed_scaling_factor=2.5,
    routing="grouped_sigmoid_bias",
    uses_correction_bias=True,
    expert_group_count=8,
    topk_group=4,
    leaves={
        1028: "v371",
        1756: "v371",
        2048: "v371",
        3076: "v371",
        3805: "v371",
        8276: "v371",
        12292: "v372",
        12372: "v372",
        14305: "v435",
        16384: "v435",
    },
)

QWEN_H2048: Final = _profile(
    name="qwen_h2048_e256_k8",
    hidden=2048,
    experts=256,
    topk=8,
    routed_scaling_factor=1.0,
    routing="softmax",
    uses_correction_bias=False,
    leaves={
        1034: "v366",
        1035: "v366",
        1036: "v366",
        1352: "v366",
        2048: "v366",
        3104: "v366",
        4107: "v365",
        4108: "v365",
        4138: "v365",
        4434: "v364",
        4435: "v364",
        4436: "v364",
        9309: "v363",
        9330: "v363",
        11469: "v363",
        13421: "v363",
        13422: "v363",
        16384: "v367",
    },
)

QWEN_H4096: Final = _profile(
    name="qwen_h4096_e512_k10",
    hidden=4096,
    experts=512,
    topk=10,
    routed_scaling_factor=1.0,
    routing="softmax",
    uses_correction_bias=False,
    leaves={
        1035: "v371",
        1036: "v371",
        2048: "v400",
        4107: "v412",
        4108: "v412",
        4434: "v414",
        4435: "v414",
        5240: "v431",
        8214: "v394",
        13421: "v430",
        13422: "v430",
        16384: "v415",
    },
)

PROFILES: Final = MappingProxyType(
    {profile.name: profile for profile in (DSV4, DSV32, QWEN_H2048, QWEN_H4096)}
)

_IMPLEMENTATION_MODULE_BY_FAMILY: Final = MappingProxyType(
    {
        _KernelFamilyKey(
            routing_mode=RoutingMode.SQRT_SOFTPLUS_BIAS,
            hidden=4096,
            experts=256,
            topk=6,
            uses_correction_bias=True,
            expert_group_count=None,
            topk_group=None,
            renormalize=True,
            routed_scaling_factor=1.5,
        ): f"{__package__}.sqrt_softplus_h4096_e256_k6",
        _KernelFamilyKey(
            routing_mode=RoutingMode.GROUPED_SIGMOID_BIAS,
            hidden=7168,
            experts=256,
            topk=8,
            uses_correction_bias=True,
            expert_group_count=8,
            topk_group=4,
            renormalize=True,
            routed_scaling_factor=2.5,
        ): f"{__package__}.grouped_sigmoid_h7168_e256_k8",
        _KernelFamilyKey(
            routing_mode=RoutingMode.SOFTMAX,
            hidden=2048,
            experts=256,
            topk=8,
            uses_correction_bias=False,
            expert_group_count=None,
            topk_group=None,
            renormalize=True,
            routed_scaling_factor=1.0,
        ): f"{__package__}.softmax_h2048_e256_k8",
        _KernelFamilyKey(
            routing_mode=RoutingMode.SOFTMAX,
            hidden=4096,
            experts=512,
            topk=10,
            uses_correction_bias=False,
            expert_group_count=None,
            topk_group=None,
            renormalize=True,
            routed_scaling_factor=1.0,
        ): f"{__package__}.softmax_h4096_e512_k10",
    }
)


def _build_dispatch_table() -> Mapping[ShapeKey, KernelPlan]:
    table: dict[ShapeKey, KernelPlan] = {}
    used_families: set[_KernelFamilyKey] = set()
    for shape, source_leaf in _PLAN_SEEDS:
        family = _family_key(shape)
        try:
            implementation_module = _IMPLEMENTATION_MODULE_BY_FAMILY[family]
        except KeyError as error:
            raise RuntimeError(
                f"no MegaGate implementation module for family {family!r}"
            ) from error
        used_families.add(family)
        plan = KernelPlan(
            shape=shape,
            implementation_module=implementation_module,
            source_leaf=source_leaf,
        )
        previous = table.setdefault(shape, plan)
        if previous != plan:
            raise RuntimeError(
                f"conflicting MegaGate plans for model-independent key {shape!r}: "
                f"{previous!r} versus {plan!r}"
            )
    unused_families = set(_IMPLEMENTATION_MODULE_BY_FAMILY) - used_families
    if unused_families:
        raise RuntimeError(
            f"MegaGate implementation families have no exact shapes: {unused_families!r}"
        )
    return MappingProxyType(table)


# This is the sole production selection table.  Profile names and aliases are
# intentionally not inputs to its exact O(1) lookup.
DISPATCH_TABLE: Final = _build_dispatch_table()
_PLAN_SEEDS.clear()

_ALIASES: Final = MappingProxyType(
    {
        "dsv4": DSV4.name,
        "deepseek_v4": DSV4.name,
        "deepseek_v3_2": DSV32.name,
        "qwen_h2048": QWEN_H2048.name,
        "qwen_h4096": QWEN_H4096.name,
    }
)


def get_profile(name: str) -> MegaGateProfile:
    """Return one supported profile, accepting a small stable alias set."""

    canonical = _ALIASES.get(name, name)
    try:
        return PROFILES[canonical]
    except KeyError as error:
        supported = ", ".join(PROFILES)
        raise ValueError(
            f"unsupported MegaGate profile {name!r}; supported: {supported}"
        ) from error


def get_kernel_plan(shape: ShapeKey) -> KernelPlan:
    """Return the exact plan for ``shape`` with one model-agnostic lookup."""

    try:
        return DISPATCH_TABLE[shape]
    except KeyError as error:
        raise ValueError(f"unsupported MegaGate shape contract: {shape!r}") from error


def get_module_dispatch(implementation_module: str) -> Mapping[int, str]:
    """Derive a host's supported tokens and provenance from the exact plans."""

    return MappingProxyType(
        {
            plan.shape.tokens: plan.source_leaf
            for plan in DISPATCH_TABLE.values()
            if plan.implementation_module == implementation_module
        }
    )


__all__ = [
    "DSV4",
    "DSV32",
    "DISPATCH_TABLE",
    "KernelPlan",
    "MegaGateProfile",
    "PROFILES",
    "QWEN_H2048",
    "QWEN_H4096",
    "RoutingMode",
    "ShapeKey",
    "get_kernel_plan",
    "get_profile",
]

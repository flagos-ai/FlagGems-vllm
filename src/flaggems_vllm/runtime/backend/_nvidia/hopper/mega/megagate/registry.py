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

"""Auditable model-profile and exact-shape registry for Hopper MegaGate.

The ``vNNN`` values are source provenance, not runtime module names.  Runtime
dispatch first binds one model profile and then performs one exact-token
dictionary lookup.  Keeping the provenance here makes the production mapping
reviewable without retaining the research dispatch wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


@dataclass(frozen=True)
class MegaGateProfile:
    """Static ABI and dispatch metadata for one supported model profile."""

    name: str
    module: str
    hidden: int
    experts: int
    topk: int
    routed_scaling_factor: float
    routing: str
    source_leaf_by_tokens: Mapping[int, str]

    @property
    def supported_tokens(self) -> tuple[int, ...]:
        return tuple(self.source_leaf_by_tokens)


def _profile(
    *,
    name: str,
    module: str,
    hidden: int,
    experts: int,
    topk: int,
    routed_scaling_factor: float,
    routing: str,
    leaves: Mapping[int, str],
) -> MegaGateProfile:
    return MegaGateProfile(
        name=name,
        module=module,
        hidden=hidden,
        experts=experts,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        routing=routing,
        source_leaf_by_tokens=MappingProxyType(dict(leaves)),
    )


DSV4: Final = _profile(
    name="dsv4_flash",
    module="flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.dsv4",
    hidden=4096,
    experts=256,
    topk=6,
    routed_scaling_factor=1.5,
    routing="sqrt_softplus_bias",
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
    module="flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.dsv32",
    hidden=7168,
    experts=256,
    topk=8,
    routed_scaling_factor=2.5,
    routing="grouped_sigmoid_bias",
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
    module="flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.qwen_h2048",
    hidden=2048,
    experts=256,
    topk=8,
    routed_scaling_factor=1.0,
    routing="softmax",
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
    module="flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.qwen_h4096",
    hidden=4096,
    experts=512,
    topk=10,
    routed_scaling_factor=1.0,
    routing="softmax",
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


__all__ = [
    "DSV4",
    "DSV32",
    "MegaGateProfile",
    "PROFILES",
    "QWEN_H2048",
    "QWEN_H4096",
    "get_profile",
]

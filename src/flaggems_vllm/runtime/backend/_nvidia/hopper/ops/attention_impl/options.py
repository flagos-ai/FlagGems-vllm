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

"""Environment options and shared host scheduling enums."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import lru_cache
from typing import TypeVar

_EnumArg = TypeVar("_EnumArg", bound=Enum)


class KernelFamily(str, Enum):
    AUTO = "auto"
    DIRECT = "direct"
    LONG = "long"


class PagedGatherMode(IntEnum):
    LEGACY = 0
    BLOCKWISE = 1
    AUTO = 2


class Toggle(str, Enum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


class HeadsInL2Mode(str, Enum):
    AUTO = "auto"
    L2_AUTO = "l2_auto"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class HeadsInL2Policy:
    """Host policy for the unified reverse-M and L2-head schedule."""

    mode: HeadsInL2Mode
    value: int = 0


@dataclass(frozen=True)
class RouteConfig:
    """Validated by load_config; internal overrides use the same canonical values."""

    decode_strategy: KernelFamily
    pack_gqa: Toggle
    paged_prefill_route: KernelFamily
    paged_prefill_min_q: int | None
    paged_prefill_min_avg_q: int
    paged_gather: PagedGatherMode
    wide_pack_gqa: bool
    force_paged_kv_tma: bool
    ragged_scheduler: str
    heads_in_l2: HeadsInL2Policy
    dynamic_scheduler: str
    dynamic_split: bool
    log_plan: bool


def _parse_enum(name: str, enum_class: type[_EnumArg]) -> _EnumArg:
    value = _parse_choice(name, "auto", {member.name.lower() for member in enum_class})
    return enum_class[value.upper()]


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid {name}={value!r}; expected an integer") from exc


def _env_int(name: str, default: int) -> int:
    value = _optional_env_int(name)
    return default if value is None else value


def _parse_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise RuntimeError(f"invalid {name}={value!r}; expected one of {choices}")
    return value


def _parse_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value not in {"0", "1", "false", "true", "off", "on"}:
        raise RuntimeError(f"invalid {name}={value!r}; expected a boolean")
    return value in {"1", "true", "on"}


def _parse_heads_in_l2() -> HeadsInL2Policy:
    name = "FLAG_GEMS_FA3_TLE_HEADS_IN_L2"
    choices = "auto, l2_auto, 0, 1, 2, 4, 8, or 16"
    value = os.getenv(name, HeadsInL2Mode.AUTO.value).strip().lower()
    if value == HeadsInL2Mode.AUTO.value:
        return HeadsInL2Policy(HeadsInL2Mode.AUTO)
    if value == HeadsInL2Mode.L2_AUTO.value:
        return HeadsInL2Policy(HeadsInL2Mode.L2_AUTO)
    try:
        explicit = int(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid {name}={value!r}; expected {choices}") from exc
    if explicit not in (0, 1, 2, 4, 8, 16):
        raise RuntimeError(f"invalid {name}={value!r}; expected {choices}")
    return HeadsInL2Policy(HeadsInL2Mode.EXPLICIT, explicit)


@lru_cache(maxsize=1)
def load_config() -> RouteConfig:
    """Read and cache one coherent scheduler configuration snapshot."""

    paged_gather = _parse_enum(
        "FLAG_GEMS_FA3_TLE_PAGED_GATHER",
        PagedGatherMode,
    )
    paged_prefill_route = _parse_enum(
        "FLAG_GEMS_FA3_TLE_PAGED_PREFILL_ROUTE",
        KernelFamily,
    )
    decode_strategy = _parse_enum(
        "FLAG_GEMS_FA3_TLE_DECODE_STRATEGY",
        KernelFamily,
    )
    paged_prefill_min_q = _optional_env_int("FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_Q")
    paged_prefill_min_avg_q = _env_int("FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_AVG_Q", 128)
    pack_gqa = _parse_enum(
        "FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK",
        Toggle,
    )
    ragged_scheduler = _parse_choice(
        "FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT",
        "auto",
        {"off", "auto", "ragged"},
    )
    heads_in_l2 = _parse_heads_in_l2()
    dynamic_scheduler = _parse_choice(
        "FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER",
        "auto",
        {"off", "auto", "on"},
    )
    return RouteConfig(
        decode_strategy=decode_strategy,
        pack_gqa=pack_gqa,
        paged_prefill_route=paged_prefill_route,
        paged_prefill_min_q=paged_prefill_min_q,
        paged_prefill_min_avg_q=paged_prefill_min_avg_q,
        paged_gather=paged_gather,
        wide_pack_gqa=_parse_bool("FLAG_GEMS_FA3_TLE_EXPERIMENT_WIDE_PACK_GQA"),
        force_paged_kv_tma=_parse_bool("FLAG_GEMS_FA3_TLE_PAGED_KV_TMA_EXPERIMENT"),
        ragged_scheduler=ragged_scheduler,
        heads_in_l2=heads_in_l2,
        dynamic_scheduler=dynamic_scheduler,
        dynamic_split=_parse_bool(
            "FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", default=True
        ),
        log_plan=_parse_bool("FLAG_GEMS_FA3_TLE_LOG_PLAN"),
    )

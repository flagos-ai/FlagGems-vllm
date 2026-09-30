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

"""Pure host routing for Hopper TLE FA3.

build extracts scalar facts, the bounded cache runs one route policy, and the
final plan derives launch controls only after family and split count are fixed.
Kernel-specific candidates and launch geometry live in kernel_config.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Any, NamedTuple

import torch

from .kernel_config import _ceil_div, _next_power_of_2
from .options import (
    HeadsInL2Mode,
    HeadsInL2Policy,
    KernelFamily,
    PagedGatherMode,
    RouteConfig,
    Toggle,
    load_config,
)
from .validation import PreparedFA3Inputs


class FA3RouteInputs(NamedTuple):
    """Tensor-free cache key; all values come from validated host metadata.

    No device lengths, tensors, workspace addresses, or launch grids are cached.
    Alignment booleans carry the only pointer/layout facts needed by routing.
    """

    dtype: Any
    element_size: int
    batch_size: int
    max_seqlen_q: int
    max_seqlen_k: int
    total_q: int
    num_heads: int
    num_heads_k: int
    head_dim: int
    has_cache_kv: bool
    is_paged: bool
    block_size: int
    qo_tma_aligned: bool
    kv_tma_aligned: bool
    arch: int
    num_sms: int
    is_alibi: bool
    is_causal: bool
    is_local: bool
    is_softcap: bool
    has_s_aux: bool
    has_seqused_k: bool
    max_num_splits: int
    forced_block_m: int


class MetadataMode(str, Enum):
    PREFILL = "prefill"
    DIRECT_DECODE = "direct_decode"
    MULTI_TOKEN_DECODE = "multi_token_decode"


class PagedKVLoadMode(str, Enum):
    NONE = "none"
    TMA = "tma"
    NON_TMA = "non_tma"


class TileWave(IntEnum):
    """Candidate tile count relative to one device wave."""

    UNDER = -1
    EXACT = 0
    OVER = 1


class FA3Workload(str, Enum):
    """Coarse serving workload selected before kernel-specific routing."""

    TOKEN_QUERY = "token_query"
    SPECULATIVE_QUERY = "speculative_query"
    SHORT_QUERY = "short_query"
    SERVING_PREFILL = "serving_prefill"
    PREFILL = "prefill"


class QueryPopulation(str, Enum):
    """Mutually exclusive query-population layout classes."""

    SINGLE = "single"
    UNIFORM = "uniform"
    RAGGED = "ragged"
    MIXED = "mixed"
    OVERSIZED = "oversized"

    @property
    def ragged_supported(self) -> bool:
        return self not in {QueryPopulation.SINGLE, QueryPopulation.OVERSIZED}

    @property
    def split_supported(self) -> bool:
        return self is not QueryPopulation.OVERSIZED


class FA3InputProfile(str, Enum):
    """Measured input cohort that refines one coarse workload."""

    GENERAL = "general"
    THROUGHPUT_DECODE = "throughput_decode"
    HEAVY = "heavy"
    LARGE_SINGLE_PREFILL = "large_single_prefill"


class TMAAlignment(str, Enum):
    """TMA legality exposed by volatile Q/O and K/V layouts."""

    NONE = "none"
    QO_ONLY = "qo_only"
    FULL = "full"


_DIRECT_NONE = 1 << 0
_LONG_NONE = 1 << 1
_DIRECT_NON_TMA = 1 << 2
_LONG_NON_TMA = 1 << 3
_DIRECT_TMA = 1 << 4
_LONG_TMA = 1 << 5

# Page-size literals below describe measured profiles or the smallest legal
# TMA tile.  They are deliberately centralized and are not address-generation
# constraints; Blockwise pointer generation supports every power-of-two page.
_DIRECT_TMA_TILE_ROWS = 16
_PERSISTENT_TMA_TILE_ROWS = 64
_COMPACT_PAGED_PROFILE_SIZES = frozenset({16})
_TMA_PAGED_PROFILE_SIZES = frozenset({128})
_DIRECT_WIDE_DECODE_LEGACY_GATHER_PAGE_SIZES = frozenset({32, 256})
_PERSISTENT_LEGACY_GATHER_PAGE_SIZES = frozenset({16, 32, 256})


class PersistentRouteProfile(str, Enum):
    """Persistent tile class selected by the cross-family route model."""

    GENERAL = "general"
    WIDE_DECODE = "wide_decode"
    SHORT_SPEC = "short_spec"


class FA3KernelProfile(str, Enum):
    """Static kernel/hardware profile combined with an input bucket."""

    GENERAL = "general"
    DENSE_MHA = "h100_dense_mha_work"
    PAGED_TMA_D128 = "h100_paged_tma_gqa4_q"
    PAGED_TMA_WIDE = "h100_paged_tma_wide_waves"
    PAGED_COMPACT_MHA = "h100_paged_compact_mha"
    PAGED_COMPACT_D128 = "h100_paged_compact_d128"
    PAGED_COMPACT_WIDE = "h100_paged_compact_wide"
    PAGED_D256_PREFILL = "h100_paged_d256_prefill"


@dataclass(frozen=True, slots=True)
class FA3ExecutionPlan:
    """Algorithm plan with no tensor ownership or launch-grid state."""

    kernel: KernelFamily
    kernel_name: str
    metadata_mode: MetadataMode
    workload: str
    paged_kv_load: PagedKVLoadMode
    paged_prefill_candidate: bool
    paged_d256_prefill_profile: bool
    reason: str
    pack_factor: int
    paged_gather_mode: int
    ragged_scheduler: bool
    heads_in_l2: HeadsInL2Policy
    dynamic_scheduler: bool
    persistent_num_splits: int
    explicit_split_k_chunk: int
    requires_tma_alignment: bool
    log_plan: bool

    @property
    def pack_gqa(self) -> bool:
        return self.pack_factor > 1

    @property
    def persistent_split_kv(self) -> bool:
        return self.persistent_num_splits > 1

    @property
    def paged_kv_non_tma(self) -> bool:
        return self.paged_kv_load is PagedKVLoadMode.NON_TMA


@dataclass(frozen=True)
class FA3RouteDecision:
    """Final legalized family/transport preference."""

    family: KernelFamily
    paged_kv_load: PagedKVLoadMode
    persistent_profile: PersistentRouteProfile
    kernel_profile: FA3KernelProfile


@dataclass(frozen=True, slots=True)
class FA3InputBucket:
    """Coarse, hash-stable summary extracted from serving-volatile inputs."""

    workload: FA3Workload
    population: QueryPopulation
    profile: FA3InputProfile
    alignment: TMAAlignment
    padded_unpacked_waves: tuple[TileWave, TileWave]
    padded_packed_waves: tuple[TileWave, TileWave]
    ragged_unpacked_waves: tuple[TileWave, TileWave]
    ragged_packed_waves: tuple[TileWave, TileWave]

    def select_waves(self, *, packed: bool, ragged: bool) -> tuple[TileWave, TileWave]:
        if ragged:
            return self.ragged_packed_waves if packed else self.ragged_unpacked_waves
        return self.padded_packed_waves if packed else self.padded_unpacked_waves


@dataclass(frozen=True, slots=True)
class FA3RouteFeatures:
    """Family/transport candidates derived from the input bucket and packing."""

    persistent_profile: PersistentRouteProfile
    kernel_profile: FA3KernelProfile
    default_legacy_tma: bool
    auto_family_without_tma: KernelFamily
    auto_family_with_tma: KernelFamily
    measured_family: KernelFamily
    measured_load: PagedKVLoadMode | None
    is_paged: bool
    persistent_tma_tile: bool
    capabilities: int


class FA3RoutePolicy:
    """Measured route rules plus a compatibility fallback for unseen shapes."""

    VERSION = "sm90_coarse_buckets_v9"
    DENSE_WORK_CROSSOVER = 1 << 26

    @staticmethod
    def _candidate_bit(family: KernelFamily, load: PagedKVLoadMode) -> int:
        if load is PagedKVLoadMode.NONE:
            return _DIRECT_NONE if family is KernelFamily.DIRECT else _LONG_NONE
        if load is PagedKVLoadMode.NON_TMA:
            return _DIRECT_NON_TMA if family is KernelFamily.DIRECT else _LONG_NON_TMA
        return _DIRECT_TMA if family is KernelFamily.DIRECT else _LONG_TMA

    @staticmethod
    def _capabilities(
        is_paged: bool,
        alignment: TMAAlignment,
        arch: int,
        block_size: int,
    ) -> int:
        qo_tma_aligned = alignment is not TMAAlignment.NONE
        kv_tma_aligned = alignment is TMAAlignment.FULL
        if not is_paged:
            return _DIRECT_NONE | (_LONG_NONE if alignment is TMAAlignment.FULL else 0)
        capabilities = _DIRECT_NON_TMA | (_LONG_NON_TMA if qo_tma_aligned else 0)
        if qo_tma_aligned and kv_tma_aligned and arch >= 90:
            if block_size % _DIRECT_TMA_TILE_ROWS == 0:
                capabilities |= _DIRECT_TMA
            if block_size % _PERSISTENT_TMA_TILE_ROWS == 0:
                capabilities |= _LONG_TMA
        return capabilities

    @classmethod
    def analyze(
        cls,
        input_bucket: FA3InputBucket,
        *,
        arch: int,
        element_size: int,
        head_size: int,
        num_heads: int,
        num_heads_k: int,
        pack_factor: int,
        has_cache_kv: bool,
        is_paged: bool,
        is_alibi: bool,
        is_local: bool,
        is_causal: bool,
        is_softcap: bool,
        block_size: int,
        paged_prefill_candidate: bool,
        measured_d256_prefill: bool,
        tile_wave: TileWave,
    ) -> FA3RouteFeatures:
        """Combine coarse input buckets with a static kernel profile."""

        gqa_ratio = num_heads // num_heads_k
        uses_kv_head_grid = num_heads > num_heads_k and pack_factor > 1
        dense_mha_profile = (
            not is_paged
            and num_heads == num_heads_k
            and not (is_causal or is_local or is_alibi or is_softcap)
            and head_size in (64, 128)
        )
        tma_profile_gqa4 = (
            arch == 90
            and element_size == 2
            and is_paged
            and block_size in _TMA_PAGED_PROFILE_SIZES
            and uses_kv_head_grid
            and gqa_ratio == 4
            and not (is_local or is_alibi or is_softcap)
        )
        compact_paged = is_paged and block_size in _COMPACT_PAGED_PROFILE_SIZES
        if dense_mha_profile:
            kernel_profile = FA3KernelProfile.DENSE_MHA
        elif tma_profile_gqa4 and head_size == 128 and is_causal:
            kernel_profile = FA3KernelProfile.PAGED_TMA_D128
        elif tma_profile_gqa4 and head_size == 192:
            kernel_profile = FA3KernelProfile.PAGED_TMA_WIDE
        elif measured_d256_prefill:
            kernel_profile = FA3KernelProfile.PAGED_D256_PREFILL
        elif (
            compact_paged
            and uses_kv_head_grid
            and gqa_ratio == 4
            and head_size in (192, 256)
        ):
            kernel_profile = FA3KernelProfile.PAGED_COMPACT_WIDE
        elif compact_paged and uses_kv_head_grid and head_size == 128:
            kernel_profile = FA3KernelProfile.PAGED_COMPACT_D128
        elif (
            compact_paged
            and num_heads == num_heads_k
            and head_size == 128
            and not (is_alibi or is_local)
        ):
            kernel_profile = FA3KernelProfile.PAGED_COMPACT_MHA
        else:
            kernel_profile = FA3KernelProfile.GENERAL

        if (
            kernel_profile is FA3KernelProfile.PAGED_COMPACT_WIDE
            and input_bucket.workload is FA3Workload.TOKEN_QUERY
        ):
            persistent_profile = PersistentRouteProfile.WIDE_DECODE
        elif (
            kernel_profile is FA3KernelProfile.PAGED_COMPACT_D128
            and gqa_ratio == 4
            and input_bucket.workload is FA3Workload.SPECULATIVE_QUERY
        ):
            persistent_profile = PersistentRouteProfile.SHORT_SPEC
        else:
            persistent_profile = PersistentRouteProfile.GENERAL

        default_legacy_tma = (
            input_bucket.profile is FA3InputProfile.LARGE_SINGLE_PREFILL
            and arch >= 90
            and element_size == 2
            and is_paged
            and block_size in _TMA_PAGED_PROFILE_SIZES
            and head_size == 128
            and gqa_ratio == 4
            and pack_factor > 1
            and is_causal
            and not (is_local or is_alibi or is_softcap)
        )
        compact_paged_ws = kernel_profile in {
            FA3KernelProfile.PAGED_COMPACT_MHA,
            FA3KernelProfile.PAGED_COMPACT_D128,
        }
        compact_serving_prefill = (
            compact_paged_ws
            and uses_kv_head_grid
            and input_bucket.workload is FA3Workload.SERVING_PREFILL
        )
        if has_cache_kv:
            common_paged_long = is_paged and (
                persistent_profile is not PersistentRouteProfile.GENERAL
                or compact_serving_prefill
                or (paged_prefill_candidate and compact_paged_ws)
            )
            auto_family_without_tma = (
                KernelFamily.LONG if common_paged_long else KernelFamily.DIRECT
            )
            auto_family_with_tma = (
                KernelFamily.LONG
                if common_paged_long or (is_paged and paged_prefill_candidate)
                else KernelFamily.DIRECT
            )
        else:
            auto_family_without_tma = (
                KernelFamily.DIRECT
                if input_bucket.workload
                in {
                    FA3Workload.TOKEN_QUERY,
                    FA3Workload.SPECULATIVE_QUERY,
                    FA3Workload.SHORT_QUERY,
                }
                else KernelFamily.LONG
            )
            auto_family_with_tma = auto_family_without_tma

        measured_family = KernelFamily.AUTO
        if (
            kernel_profile is FA3KernelProfile.DENSE_MHA
            and input_bucket.population
            in {QueryPopulation.SINGLE, QueryPopulation.UNIFORM}
        ):
            measured_family = (
                KernelFamily.LONG
                if input_bucket.profile
                in {
                    FA3InputProfile.HEAVY,
                    FA3InputProfile.LARGE_SINGLE_PREFILL,
                }
                else KernelFamily.DIRECT
            )
        elif kernel_profile is FA3KernelProfile.PAGED_TMA_D128:
            if input_bucket.workload is FA3Workload.TOKEN_QUERY:
                measured_family = KernelFamily.DIRECT
            elif input_bucket.workload is FA3Workload.SPECULATIVE_QUERY:
                measured_family = (
                    KernelFamily.LONG
                    if tile_wave is TileWave.UNDER
                    else KernelFamily.DIRECT
                )
            else:
                measured_family = KernelFamily.LONG
        elif kernel_profile is FA3KernelProfile.PAGED_D256_PREFILL:
            measured_family = KernelFamily.LONG
        elif (
            kernel_profile is FA3KernelProfile.PAGED_TMA_WIDE
            and input_bucket.workload is FA3Workload.TOKEN_QUERY
        ) or persistent_profile is not PersistentRouteProfile.GENERAL:
            measured_family = (
                KernelFamily.LONG
                if tile_wave is TileWave.UNDER
                else KernelFamily.DIRECT
            )

        measured_load = PagedKVLoadMode.NONE if not is_paged else None
        measured_tma_profile = kernel_profile is FA3KernelProfile.PAGED_TMA_D128 or (
            kernel_profile is FA3KernelProfile.PAGED_TMA_WIDE
            and input_bucket.workload is FA3Workload.TOKEN_QUERY
        )
        if (
            measured_tma_profile
            and input_bucket.profile is not FA3InputProfile.THROUGHPUT_DECODE
        ):
            measured_load = PagedKVLoadMode.TMA
        return FA3RouteFeatures(
            persistent_profile=persistent_profile,
            kernel_profile=kernel_profile,
            default_legacy_tma=default_legacy_tma,
            auto_family_without_tma=auto_family_without_tma,
            auto_family_with_tma=auto_family_with_tma,
            measured_family=measured_family,
            measured_load=measured_load,
            is_paged=is_paged,
            persistent_tma_tile=(block_size % _PERSISTENT_TMA_TILE_ROWS == 0),
            capabilities=cls._capabilities(
                is_paged,
                input_bucket.alignment,
                arch,
                block_size,
            ),
        )

    @classmethod
    def choose(
        cls,
        features: FA3RouteFeatures,
        family_override: KernelFamily,
        transport_override: bool | None,
    ) -> FA3RouteDecision:
        legacy_tma = (
            features.default_legacy_tma
            if transport_override is None
            else transport_override
        )
        if family_override is KernelFamily.AUTO:
            incumbent_family = (
                features.auto_family_with_tma
                if legacy_tma
                else features.auto_family_without_tma
            )
        else:
            incumbent_family = family_override
        if not features.is_paged:
            incumbent_load = PagedKVLoadMode.NONE
        elif legacy_tma and (
            incumbent_family is KernelFamily.DIRECT or features.persistent_tma_tile
        ):
            incumbent_load = PagedKVLoadMode.TMA
        else:
            incumbent_load = PagedKVLoadMode.NON_TMA

        preferred_family = (
            incumbent_family
            if features.measured_family is KernelFamily.AUTO
            else features.measured_family
        )
        preferred_load = (
            incumbent_load if features.measured_load is None else features.measured_load
        )
        other_family = (
            KernelFamily.LONG
            if preferred_family is KernelFamily.DIRECT
            else KernelFamily.DIRECT
        )
        family_order = (
            (family_override,)
            if family_override is not KernelFamily.AUTO
            else (preferred_family, other_family)
        )
        load_order = (
            (
                preferred_load,
                (
                    PagedKVLoadMode.TMA
                    if preferred_load is PagedKVLoadMode.NON_TMA
                    else PagedKVLoadMode.NON_TMA
                ),
            )
            if features.is_paged
            else (PagedKVLoadMode.NONE,)
        )

        if transport_override is not None and features.is_paged:
            forced_load = (
                PagedKVLoadMode.TMA if transport_override else PagedKVLoadMode.NON_TMA
            )
            for family in family_order:
                if features.capabilities & cls._candidate_bit(family, forced_load):
                    return FA3RouteDecision(
                        family,
                        forced_load,
                        features.persistent_profile,
                        features.kernel_profile,
                    )
        for family in family_order:
            for load in load_order:
                if features.capabilities & cls._candidate_bit(family, load):
                    return FA3RouteDecision(
                        family,
                        load,
                        features.persistent_profile,
                        features.kernel_profile,
                    )
        raise RuntimeError(
            "FA3 route model found no feasible family/transport candidate"
        )


class FA3Scheduler:
    """Build one concrete FA3 route and every heuristic derived from it."""

    RAGGED_SEQUENCE_GROUP_SIZE = 31
    RAGGED_MAX_GROUPS = 32
    RAGGED_MAX_BATCH_SIZE = RAGGED_SEQUENCE_GROUP_SIZE * RAGGED_MAX_GROUPS
    RAGGED_MAX_Q_FILL_RATIO = 0.75
    DEFAULT_PAGED_PREFILL_MIN_Q = 1024
    MAX_DYNAMIC_SPLITS = 3
    MIN_EXPLICIT_SPLIT_K = 12 * 128
    WIDE_SPLIT_BLOCK_M = 64
    TILE_WAVE_BLOCK_M = 128
    FINE_GRAINED_BLOCK_M = 64
    MIN_SPLIT_WAVE_FILL_NUMERATOR = 9
    MIN_SPLIT_WAVE_FILL_DENOMINATOR = 10
    D256_DIRECT_WAVE_MULTIPLIER = 4

    load_config = staticmethod(load_config)

    @staticmethod
    def clear_config_cache() -> None:
        load_config.cache_clear()
        FA3Scheduler._build_plan.cache_clear()
        FA3Scheduler._uses_default_d256_route_config.cache_clear()

    @classmethod
    def build(
        cls,
        inputs: PreparedFA3Inputs,
        config: RouteConfig | None = None,
    ) -> FA3ExecutionPlan:
        """Extract one host key and cache the same policy for every input."""

        if config is None:
            config = cls.load_config()
        facts = FA3RouteInputs(
            inputs.q.dtype,
            inputs.q.element_size(),
            inputs.batch_size,
            inputs.max_seqlen_q,
            inputs.max_seqlen_k,
            inputs.total_q,
            inputs.num_heads,
            inputs.num_heads_k,
            inputs.head_dim,
            inputs.has_cache_kv,
            inputs.is_paged,
            inputs.block_size,
            inputs.qo_tma_aligned,
            inputs.kv_tma_aligned,
            inputs.arch,
            inputs.num_sms,
            inputs.alibi_slopes is not None,
            inputs.window.causal,
            inputs.window.local,
            inputs.is_softcap,
            getattr(inputs, "s_aux", None) is not None,
            inputs.seqused_k is not None,
            inputs.max_num_splits,
            int(os.getenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_M", "0")),
        )
        # Diagnostics never partition the algorithm cache or select a policy.
        route_config = replace(config, log_plan=False) if config.log_plan else config
        plan = cls._build_plan(facts, route_config)
        return replace(plan, log_plan=True) if config.log_plan else plan

    @classmethod
    def _has_minimum_wave_fill(cls, work_items: int, num_sms: int) -> bool:
        return (
            work_items * cls.MIN_SPLIT_WAVE_FILL_DENOMINATOR
            >= num_sms * cls.MIN_SPLIT_WAVE_FILL_NUMERATOR
        )

    @classmethod
    def _splits_for_target_wave(cls, base_work: int, num_sms: int) -> int:
        """Choose a power-of-two split count without overfilling a near-full wave."""

        required_splits = _ceil_div(num_sms, base_work)
        upper_splits = _next_power_of_2(required_splits)
        lower_splits = upper_splits // 2
        if lower_splits >= 1 and cls._has_minimum_wave_fill(
            base_work * lower_splits, num_sms
        ):
            return lower_splits
        return upper_splits

    @classmethod
    def _splits_for_exact_target_wave(cls, base_work: int, num_sms: int) -> int:
        """Use an integer split only when it leaves a nearly full first wave."""

        lower_splits = max(1, num_sms // base_work)
        if cls._has_minimum_wave_fill(base_work * lower_splits, num_sms):
            return lower_splits
        return cls._splits_for_target_wave(base_work, num_sms)

    @staticmethod
    def paged_gather_name(mode: PagedGatherMode | int) -> str:
        try:
            return PagedGatherMode(mode).name.lower()
        except ValueError:
            return f"unknown({mode})"

    @staticmethod
    def classify_population(
        *,
        total_q: int,
        batch_size: int,
        padded_q: int,
    ) -> QueryPopulation:
        if batch_size == 1:
            return QueryPopulation.SINGLE
        if batch_size > FA3Scheduler.RAGGED_MAX_BATCH_SIZE:
            return QueryPopulation.OVERSIZED
        if total_q == padded_q:
            return QueryPopulation.UNIFORM
        if 4 * total_q <= 3 * padded_q:
            return QueryPopulation.RAGGED
        return QueryPopulation.MIXED

    @staticmethod
    def classify_workload(
        *,
        total_q: int,
        batch_size: int,
        max_seqlen_q: int,
        gqa_ratio: int,
        has_cache_kv: bool,
        prefill_ready: bool,
        population: QueryPopulation,
    ) -> FA3Workload:
        """Map volatile lengths to one serving-level routing bucket."""

        if max_seqlen_q == 1:
            return FA3Workload.TOKEN_QUERY
        if has_cache_kv and max_seqlen_q <= 8:
            return FA3Workload.SPECULATIVE_QUERY
        if prefill_ready:
            return FA3Workload.PREFILL
        if has_cache_kv:
            serving_prefill = (
                population is QueryPopulation.RAGGED and max_seqlen_q >= 256
            ) or (
                population is QueryPopulation.SINGLE
                and total_q == max_seqlen_q
                and max_seqlen_q >= 512
            )
            return (
                FA3Workload.SERVING_PREFILL
                if serving_prefill
                else FA3Workload.SHORT_QUERY
            )
        if max_seqlen_q * gqa_ratio <= 128 or total_q <= 64 * batch_size:
            return FA3Workload.SHORT_QUERY
        return FA3Workload.PREFILL

    @staticmethod
    def classify_input_profile(
        *,
        workload: FA3Workload,
        population: QueryPopulation,
        total_q: int,
        batch_size: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_heads: int,
        head_dim: int,
    ) -> FA3InputProfile:
        if (
            workload is FA3Workload.TOKEN_QUERY
            and batch_size >= 32
            and max_seqlen_k >= 4096
        ):
            return FA3InputProfile.THROUGHPUT_DECODE
        if (
            population is QueryPopulation.SINGLE
            and total_q == max_seqlen_q
            and max_seqlen_q >= 8192
            and max_seqlen_k >= 8192
        ):
            return FA3InputProfile.LARGE_SINGLE_PREFILL
        if (
            total_q * max_seqlen_k * num_heads * head_dim
            > FA3RoutePolicy.DENSE_WORK_CROSSOVER
        ):
            return FA3InputProfile.HEAVY
        return FA3InputProfile.GENERAL

    @staticmethod
    def metadata_mode(*, has_cache_kv: bool, max_query_len: int) -> MetadataMode:
        if not has_cache_kv:
            return MetadataMode.PREFILL
        if max_query_len <= 1:
            return MetadataMode.DIRECT_DECODE
        return MetadataMode.MULTI_TOKEN_DECODE

    @staticmethod
    def _tile_wave(tile_count: int, num_sms: int) -> TileWave:
        if tile_count < num_sms:
            return TileWave.UNDER
        if tile_count == num_sms:
            return TileWave.EXACT
        return TileWave.OVER

    @classmethod
    def analyze_inputs(
        cls,
        inputs: FA3RouteInputs,
        config: RouteConfig,
    ) -> FA3InputBucket:
        """Extract host-only, discrete facts from serving-volatile inputs."""

        batch_size = inputs.batch_size
        max_q = inputs.max_seqlen_q
        max_k = inputs.max_seqlen_k
        total_q = inputs.total_q
        num_heads = inputs.num_heads
        num_heads_k = inputs.num_heads_k
        head_dim = inputs.head_dim
        gqa_ratio = num_heads // num_heads_k if num_heads > num_heads_k else 1
        packed_heads = num_heads_k if gqa_ratio > 1 else num_heads
        padded_q = batch_size * max_q
        population = cls.classify_population(
            total_q=total_q,
            batch_size=batch_size,
            padded_q=padded_q,
        )

        min_prefill_q = config.paged_prefill_min_q
        if min_prefill_q is None:
            min_prefill_q = cls.DEFAULT_PAGED_PREFILL_MIN_Q
        prefill_ready = (
            max_q >= min_prefill_q
            and total_q >= config.paged_prefill_min_avg_q * batch_size
        )
        workload = cls.classify_workload(
            total_q=total_q,
            batch_size=batch_size,
            max_seqlen_q=max_q,
            gqa_ratio=gqa_ratio,
            has_cache_kv=inputs.has_cache_kv,
            prefill_ready=prefill_ready,
            population=population,
        )
        profile = cls.classify_input_profile(
            workload=workload,
            population=population,
            total_q=total_q,
            batch_size=batch_size,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        tile_wave = cls._tile_wave
        num_sms = inputs.num_sms

        def tile_waves(
            query_rows: int, head_groups: int, ragged_extra: int = 0
        ) -> tuple[TileWave, TileWave]:
            return tuple(
                tile_wave(
                    (_ceil_div(query_rows, block_m) + ragged_extra) * head_groups,
                    num_sms,
                )
                for block_m in (cls.TILE_WAVE_BLOCK_M, cls.FINE_GRAINED_BLOCK_M)
            )

        return FA3InputBucket(
            workload=workload,
            population=population,
            profile=profile,
            alignment=(
                TMAAlignment.FULL
                if inputs.qo_tma_aligned and inputs.kv_tma_aligned
                else (
                    TMAAlignment.QO_ONLY if inputs.qo_tma_aligned else TMAAlignment.NONE
                )
            ),
            padded_unpacked_waves=tile_waves(max_q, batch_size * num_heads),
            padded_packed_waves=tile_waves(
                max_q * gqa_ratio, batch_size * packed_heads
            ),
            ragged_unpacked_waves=tile_waves(total_q, num_heads, batch_size - 1),
            ragged_packed_waves=tile_waves(
                total_q * gqa_ratio, packed_heads, batch_size - 1
            ),
        )

    @staticmethod
    def select_ragged_scheduler(
        mode: str,
        *,
        supported: bool,
        auto_candidate: bool,
    ) -> bool:
        return supported and (mode == "ragged" or mode == "auto" and auto_candidate)

    @staticmethod
    def select_heads_in_l2(
        policy: HeadsInL2Policy, *, causal: bool, local: bool
    ) -> HeadsInL2Policy:
        """Resolve mask-dependent modes without using volatile shape data."""

        enabled = causal or local
        if policy.mode is HeadsInL2Mode.AUTO:
            return HeadsInL2Policy(HeadsInL2Mode.EXPLICIT, int(enabled))
        if policy.mode is HeadsInL2Mode.L2_AUTO and not enabled:
            return HeadsInL2Policy(HeadsInL2Mode.EXPLICIT, 0)
        return policy

    @staticmethod
    def select_dynamic_scheduler(
        mode: str,
        *,
        causal: bool,
        local: bool,
        tile_wave: TileWave,
        fine_grained_over: bool = False,
    ) -> bool:
        if mode == "off":
            return False
        scheduler_has_multiple_waves = tile_wave is TileWave.OVER or fine_grained_over
        if mode == "on":
            return scheduler_has_multiple_waves
        return (causal or local) and scheduler_has_multiple_waves

    @staticmethod
    def select_persistent_paged_gather(
        mode: int,
        *,
        is_paged: bool,
        paged_kv_non_tma: bool,
        pack_gqa: bool,
        block_size: int,
    ) -> int:
        if (
            mode == int(PagedGatherMode.AUTO)
            and is_paged
            and paged_kv_non_tma
            and pack_gqa
            and block_size in _PERSISTENT_LEGACY_GATHER_PAGE_SIZES
        ):
            return int(PagedGatherMode.LEGACY)
        return mode

    @staticmethod
    @lru_cache(maxsize=16)
    def _uses_default_d256_route_config(config: RouteConfig) -> bool:
        """Whether default options enable the measured D256 decode preference."""

        return (
            config.decode_strategy is KernelFamily.AUTO
            and config.pack_gqa is Toggle.AUTO
            and config.paged_prefill_route is KernelFamily.AUTO
            and config.paged_prefill_min_q is None
            and config.paged_prefill_min_avg_q == 128
            and config.paged_gather is PagedGatherMode.AUTO
            and not config.wide_pack_gqa
            and not config.force_paged_kv_tma
            and config.ragged_scheduler == "auto"
            and config.heads_in_l2 == HeadsInL2Policy(HeadsInL2Mode.AUTO)
            and config.dynamic_scheduler == "auto"
            and config.dynamic_split
        )

    @classmethod
    @lru_cache(maxsize=1024)
    def _build_plan(
        cls, inputs: FA3RouteInputs, config: RouteConfig
    ) -> FA3ExecutionPlan:
        """Resolve candidates and split count, then derive one complete plan."""

        gqa_ratio = inputs.num_heads // inputs.num_heads_k
        default_d256 = (
            cls._uses_default_d256_route_config(config)
            and inputs.forced_block_m == 0
            and inputs.arch == 90
            and inputs.dtype in (torch.float16, torch.bfloat16)
            and inputs.has_cache_kv
            and inputs.is_paged
            and inputs.head_dim == 256
            and (inputs.block_size, gqa_ratio) in {(16, 4), (32, 8)}
            # D256 single-token causal attention is normalized to a mask-free
            # window by input preparation; it still belongs to this route.
            and (inputs.is_causal or inputs.max_seqlen_q == 1)
            and not inputs.is_local
            and not inputs.is_alibi
            and not inputs.is_softcap
            and not inputs.has_s_aux
            and inputs.qo_tma_aligned
        )
        uniform_token_query = (
            inputs.batch_size > 1
            and inputs.max_seqlen_q == 1
            and inputs.total_q == inputs.batch_size
        )
        split_base_work = inputs.batch_size * inputs.num_heads_k
        direct_token_query = uniform_token_query and (
            (inputs.block_size, gqa_ratio) == (32, 8)
            or (
                (inputs.block_size, gqa_ratio) == (16, 4)
                # A measured direct launch is competitive once four waves of
                # its packed CTA work can cover the device.  Below this point,
                # preserve adaptive Split-KV parallelism.
                and split_base_work * cls.D256_DIRECT_WAVE_MULTIPLIER >= inputs.num_sms
            )
        )
        if default_d256 and direct_token_query:
            config = replace(config, decode_strategy=KernelFamily.DIRECT)
        bucket = cls.analyze_inputs(inputs, config)
        forced_split_block_m = inputs.forced_block_m
        if forced_split_block_m not in (0, 16, 64, 128):
            raise ValueError(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_M must be 0, 16, 64, or 128"
            )
        split_block_m = forced_split_block_m or cls.WIDE_SPLIT_BLOCK_M
        single_long_ragged_query = (
            bucket.population in {QueryPopulation.RAGGED, QueryPopulation.MIXED}
            and inputs.batch_size > 1
            and inputs.max_seqlen_q > 1
            and inputs.total_q == inputs.max_seqlen_q + inputs.batch_size - 1
        )
        dominant_long_ragged_query = (
            bucket.population in {QueryPopulation.RAGGED, QueryPopulation.MIXED}
            and inputs.batch_size > 1
            and inputs.max_seqlen_q > 1
            and inputs.total_q - inputs.max_seqlen_q <= 16 * (inputs.batch_size - 1)
        )
        prefer_padded_dominant_long_prefill = (
            config.ragged_scheduler == "auto"
            and dominant_long_ragged_query
            and bucket.workload is FA3Workload.PREFILL
            and inputs.batch_size <= 3
            and inputs.arch == 90
            and inputs.num_sms >= 128
            and inputs.dtype in (torch.float16, torch.bfloat16)
            and inputs.is_paged
            and inputs.head_dim == 256
            and (inputs.block_size, gqa_ratio) in {(16, 4), (32, 8)}
            and inputs.is_causal
            and not inputs.is_alibi
            and not (inputs.is_local or inputs.is_softcap)
        )
        # On 128+ SM Hopper parts, dominant-long prefill batches pay more for
        # the ragged mapper than they save in rectangular launch slots.
        # Preserve explicit user overrides and keep the H100 (114 SM) route.
        config = (
            replace(config, ragged_scheduler="off")
            if prefer_padded_dominant_long_prefill
            else config
        )
        if bucket.population is QueryPopulation.UNIFORM:
            # Uniform lengths are known exactly from host metadata.  The
            # compact ``ceil(total)+B-1`` upper bound can be almost 2x too
            # large for Q=9..16, causing adaptive Split-K to underfill H800.
            wide_split_base_work = (
                _ceil_div(inputs.max_seqlen_q * gqa_ratio, split_block_m)
                * inputs.batch_size
                * inputs.num_heads_k
            )
        elif single_long_ragged_query:
            # The host metadata proves that one request has max_seqlen_q rows
            # and every other request has exactly one.  Unlike a generic
            # ragged population, its packed compact work is therefore exact.
            wide_split_base_work = (
                _ceil_div(inputs.max_seqlen_q * gqa_ratio, split_block_m)
                + (inputs.batch_size - 1) * _ceil_div(gqa_ratio, split_block_m)
            ) * inputs.num_heads_k
        else:
            # Ragged per-sequence lengths live on device.  This compact upper
            # bound matches the launch geometry without synchronizing them to
            # the host; invalid compact slots are discarded by the mapper.
            wide_split_base_work = (
                _ceil_div(
                    inputs.total_q * gqa_ratio,
                    split_block_m,
                )
                + inputs.batch_size
                - 1
            ) * inputs.num_heads_k
        packed_padded_work = (
            _ceil_div(inputs.max_seqlen_q * gqa_ratio, 128)
            * inputs.batch_size
            * inputs.num_heads_k
        )
        near_full_uniform_decode = (
            inputs.arch == 90
            and inputs.dtype == torch.float16
            and inputs.is_paged
            and bucket.workload is FA3Workload.TOKEN_QUERY
            and bucket.population is QueryPopulation.UNIFORM
            and bucket.profile is FA3InputProfile.THROUGHPUT_DECODE
            and inputs.head_dim == 256
            and (inputs.block_size, gqa_ratio) == (16, 4)
            and not inputs.is_alibi
            and not (inputs.is_local or inputs.is_softcap)
            and packed_padded_work < inputs.num_sms
            and cls._has_minimum_wave_fill(packed_padded_work, inputs.num_sms)
        )
        if config.force_paged_kv_tma and not inputs.is_paged:
            raise RuntimeError(
                "FLAG_GEMS_FA3_TLE_PAGED_KV_TMA_EXPERIMENT requires paged KV"
            )

        measured_d256_prefill = (
            inputs.arch == 90
            and inputs.dtype in (torch.float16, torch.bfloat16)
            and inputs.has_cache_kv
            and inputs.is_paged
            and bucket.workload in {FA3Workload.PREFILL, FA3Workload.SERVING_PREFILL}
            and bucket.population is not QueryPopulation.OVERSIZED
            and bucket.profile
            in {FA3InputProfile.HEAVY, FA3InputProfile.LARGE_SINGLE_PREFILL}
            and inputs.head_dim == 256
            and (inputs.block_size, gqa_ratio) in {(16, 4), (32, 8)}
            and inputs.is_causal
            and not (inputs.is_local or inputs.is_alibi or inputs.is_softcap)
        )

        pack_without_split = (
            inputs.num_heads > inputs.num_heads_k
            and gqa_ratio <= 16
            and (gqa_ratio & (gqa_ratio - 1)) == 0
            and (
                inputs.head_dim <= 128
                or (
                    bucket.workload is FA3Workload.TOKEN_QUERY
                    and inputs.head_dim in (192, 256)
                )
                or measured_d256_prefill
                or config.wide_pack_gqa
            )
            and config.pack_gqa is not Toggle.OFF
        )
        measured_wide_prefill_pack = measured_d256_prefill and pack_without_split

        ragged_scheduler = cls.select_ragged_scheduler(
            config.ragged_scheduler,
            supported=bucket.population.ragged_supported,
            auto_candidate=bucket.population is QueryPopulation.RAGGED,
        )

        paged_prefill_candidate = (
            inputs.has_cache_kv
            and inputs.is_paged
            and bucket.workload is FA3Workload.PREFILL
        )

        explicit_wide_split_under_wave = (
            bucket.population is QueryPopulation.SINGLE
            or wide_split_base_work < inputs.num_sms
        )

        # Preserve precedence: decode override, measured preference, prefill override.
        requested_family = config.decode_strategy
        if requested_family is KernelFamily.AUTO and near_full_uniform_decode:
            requested_family = KernelFamily.DIRECT
        if (
            requested_family is KernelFamily.AUTO
            and inputs.has_cache_kv
            and inputs.is_paged
            and config.paged_prefill_route is not KernelFamily.AUTO
        ):
            requested_family = config.paged_prefill_route

        transport_override = True if config.force_paged_kv_tma else None

        # These measured single-request D256 cohorts otherwise launch only one
        # or two direct CTAs and serialize long K despite an explicit split cap.
        # Keep the exception separate from the default page/workload profiles.
        explicit_wide_split_candidate = (
            inputs.has_cache_kv
            and inputs.has_seqused_k
            and inputs.is_paged
            and config.dynamic_split
            and config.pack_gqa is not Toggle.OFF
            and (
                bucket.population is QueryPopulation.SINGLE
                or bucket.workload is FA3Workload.SHORT_QUERY
            )
            and explicit_wide_split_under_wave
            and bucket.profile is FA3InputProfile.HEAVY
            and bucket.workload in {FA3Workload.TOKEN_QUERY, FA3Workload.SHORT_QUERY}
            and inputs.head_dim == 256
            and (inputs.block_size, gqa_ratio) in {(16, 4), (32, 8)}
            and (bucket.workload is FA3Workload.TOKEN_QUERY or inputs.is_causal)
            and not (inputs.is_local or inputs.is_alibi or inputs.is_softcap)
        )

        for max_num_splits in (inputs.max_num_splits, 1):
            explicit_split_has_work = (
                max_num_splits > 1 and inputs.max_seqlen_k > cls.MIN_EXPLICIT_SPLIT_K
            )
            explicit_wide_split = (
                explicit_split_has_work and explicit_wide_split_candidate
            )
            # Explicit wide splitting already requires legal GQA4/8 packing.
            pack_gqa = pack_without_split or explicit_wide_split
            pack_factor = gqa_ratio if pack_gqa else 1
            tile_wave, fine_tile_wave = bucket.select_waves(
                packed=pack_gqa, ragged=ragged_scheduler
            )
            route_features = FA3RoutePolicy.analyze(
                bucket,
                arch=inputs.arch,
                element_size=inputs.element_size,
                head_size=inputs.head_dim,
                num_heads=inputs.num_heads,
                num_heads_k=inputs.num_heads_k,
                pack_factor=pack_factor,
                has_cache_kv=inputs.has_cache_kv,
                is_paged=inputs.is_paged,
                is_alibi=inputs.is_alibi,
                is_local=inputs.is_local,
                is_causal=inputs.is_causal,
                is_softcap=inputs.is_softcap,
                block_size=inputs.block_size,
                paged_prefill_candidate=paged_prefill_candidate,
                measured_d256_prefill=measured_d256_prefill,
                tile_wave=tile_wave,
            )
            family_override = requested_family
            if family_override is KernelFamily.AUTO and explicit_wide_split:
                family_override = KernelFamily.LONG
            decision = FA3RoutePolicy.choose(
                route_features,
                family_override,
                transport_override,
            )

            paged_kv_non_tma = decision.paged_kv_load is PagedKVLoadMode.NON_TMA
            split_profile_supported = (
                max_num_splits == 0
                and decision.persistent_profile is PersistentRouteProfile.SHORT_SPEC
            ) or (
                explicit_split_has_work
                and (
                    explicit_wide_split
                    or decision.persistent_profile
                    in {
                        PersistentRouteProfile.SHORT_SPEC,
                        PersistentRouteProfile.WIDE_DECODE,
                    }
                )
            )
            persistent_split_kv = (
                config.dynamic_split
                and inputs.is_paged
                and paged_kv_non_tma
                and inputs.has_seqused_k
                and bucket.population.split_supported
                and decision.family is KernelFamily.LONG
                and pack_gqa
                and split_profile_supported
            )
            persistent_num_splits = (
                max_num_splits
                if persistent_split_kv and max_num_splits > 1
                else cls.MAX_DYNAMIC_SPLITS if persistent_split_kv else 0
            )

            if (
                persistent_split_kv
                and max_num_splits > 1
                and bucket.population is not QueryPopulation.SINGLE
                and bucket.workload is FA3Workload.SHORT_QUERY
                and inputs.head_dim == 256
            ):
                splits_for_one_wave = (
                    cls._splits_for_exact_target_wave(
                        wide_split_base_work,
                        inputs.num_sms,
                    )
                    if single_long_ragged_query
                    else cls._splits_for_target_wave(
                        wide_split_base_work,
                        inputs.num_sms,
                    )
                )
                useful_k_splits = _ceil_div(
                    inputs.max_seqlen_k,
                    cls.MIN_EXPLICIT_SPLIT_K,
                )
                adaptive_splits = min(
                    persistent_num_splits,
                    useful_k_splits,
                    splits_for_one_wave,
                )
                if adaptive_splits == 1:
                    # Splitting enabled wide packing/family selection above. Resolve
                    # those candidates again with splitting disabled before deriving
                    # gather, dynamic scheduling, alignment, or the final plan.
                    continue
                persistent_num_splits = adaptive_splits
            break

        heads_in_l2 = cls.select_heads_in_l2(
            config.heads_in_l2,
            causal=inputs.is_causal,
            local=inputs.is_local,
        )
        if measured_wide_prefill_pack and config.heads_in_l2.mode is HeadsInL2Mode.AUTO:
            heads_in_l2 = HeadsInL2Policy(HeadsInL2Mode.L2_AUTO)
        dynamic_scheduler = cls.select_dynamic_scheduler(
            config.dynamic_scheduler,
            causal=inputs.is_causal,
            local=inputs.is_local,
            tile_wave=tile_wave,
            # Packed D256 prefill admits BM64 as well as the default BM128.
            # Use its real wave count so devices with more SMs do not suppress
            # work stealing solely because the BM128 candidate is under-wave.
            fine_grained_over=(
                measured_wide_prefill_pack and fine_tile_wave is TileWave.OVER
            ),
        )
        if persistent_split_kv:
            # A single compact query population has uniform split work and the
            # static program-id stride covers every item without scheduler
            # atomics or producer/consumer handshakes.  Keep dynamic claiming
            # for auto-s3, ragged/multi-request work, or an explicit force-on.
            static_explicit_split = (
                max_num_splits > 1
                and bucket.population is QueryPopulation.SINGLE
                and config.dynamic_scheduler != "on"
            )
            dynamic_scheduler = not static_explicit_split

        paged_gather_mode = int(config.paged_gather)
        if (
            paged_gather_mode == int(PagedGatherMode.AUTO)
            and decision.family is KernelFamily.DIRECT
            and inputs.is_paged
            and paged_kv_non_tma
            and pack_gqa
            and bucket.workload is FA3Workload.TOKEN_QUERY
            and inputs.head_dim == 192
            and inputs.block_size in _DIRECT_WIDE_DECODE_LEGACY_GATHER_PAGE_SIZES
        ):
            # H100 D192 decode: three isolated CUDA-Graph rounds and NCU both
            # favor Legacy here; the compact-page profile remains Blockwise.
            paged_gather_mode = int(PagedGatherMode.LEGACY)
        elif (
            paged_gather_mode == int(PagedGatherMode.AUTO)
            and persistent_split_kv
            and max_num_splits > 1
            and (
                explicit_wide_split
                or decision.persistent_profile is PersistentRouteProfile.WIDE_DECODE
            )
            and inputs.block_size in _COMPACT_PAGED_PROFILE_SIZES
        ):
            # Once wide decode is split into short K ranges, measured H100
            # CUDA-Graph latency favors blockwise address generation.  Keep the
            # existing one-pass and auto-s3 gather choices unchanged.
            paged_gather_mode = int(PagedGatherMode.BLOCKWISE)
        elif (
            paged_gather_mode == int(PagedGatherMode.AUTO)
            and measured_wide_prefill_pack
            and decision.family is KernelFamily.LONG
        ):
            paged_gather_mode = int(PagedGatherMode.BLOCKWISE)
        elif persistent_split_kv or decision.family is KernelFamily.LONG:
            paged_gather_mode = cls.select_persistent_paged_gather(
                paged_gather_mode,
                is_paged=inputs.is_paged,
                paged_kv_non_tma=paged_kv_non_tma,
                pack_gqa=pack_gqa,
                block_size=inputs.block_size,
            )

        explicit_split_k_chunk = cls.MIN_EXPLICIT_SPLIT_K
        if (
            config.paged_gather is PagedGatherMode.AUTO
            and persistent_split_kv
            and single_long_ragged_query
            and bucket.workload is FA3Workload.SHORT_QUERY
            and inputs.arch == 90
            and inputs.dtype in (torch.float16, torch.bfloat16)
            and inputs.head_dim == 256
            and (inputs.block_size, gqa_ratio) == (32, 8)
            and inputs.is_causal
            and not inputs.is_alibi
            and not (inputs.is_local or inputs.is_softcap)
        ):
            # Clean H100 replay for this exact one-long-query cohort favors
            # page-aligned loads: page32/BN64 needs two page-table entries per
            # tile instead of repeating per-token div/mod and table loads.
            paged_gather_mode = int(PagedGatherMode.BLOCKWISE)
        if (
            persistent_split_kv
            and persistent_num_splits == 32
            and bucket.population is QueryPopulation.SINGLE
            and bucket.workload is FA3Workload.SHORT_QUERY
            and inputs.dtype in (torch.float16, torch.bfloat16)
            and inputs.head_dim == 256
            and inputs.block_size == 16
            and gqa_ratio == 4
            and inputs.total_q == inputs.max_seqlen_q
            and 8 < inputs.max_seqlen_q <= 16
        ):
            # Qwen3.6 Q11/Q12 needs all 32 explicit splits to fill H100.  Keep
            # this measured chunk size out of unrelated high-cap profiles.
            explicit_split_k_chunk = 8 * 128

        if (
            inputs.has_cache_kv
            and inputs.is_paged
            and decision.family is KernelFamily.LONG
        ):
            kernel_name = "long_paged_prefill"
        else:
            kernel_name = decision.family.value
        if decision.family is KernelFamily.DIRECT and pack_gqa:
            kernel_name = "direct_packed_gqa"
        if ragged_scheduler:
            kernel_name = f"{kernel_name}_ragged"
        if persistent_split_kv:
            kernel_name = f"persistent_splitkv_s{persistent_num_splits}"

        return FA3ExecutionPlan(
            kernel=decision.family,
            kernel_name=kernel_name,
            metadata_mode=cls.metadata_mode(
                has_cache_kv=inputs.has_cache_kv,
                max_query_len=(1 if bucket.workload is FA3Workload.TOKEN_QUERY else 2),
            ),
            workload=bucket.workload.value,
            paged_kv_load=decision.paged_kv_load,
            paged_prefill_candidate=paged_prefill_candidate,
            paged_d256_prefill_profile=measured_wide_prefill_pack,
            reason=(
                f"model={FA3RoutePolicy.VERSION} "
                f"kernel_profile={decision.kernel_profile.value} "
                f"input_profile={bucket.profile.value} "
                f"profile={decision.persistent_profile.value} "
                f"family={decision.family.value} "
                f"load={decision.paged_kv_load.value} "
                f"wave={tile_wave.name.lower()}"
            ),
            pack_factor=pack_factor,
            paged_gather_mode=paged_gather_mode,
            ragged_scheduler=ragged_scheduler,
            heads_in_l2=heads_in_l2,
            dynamic_scheduler=dynamic_scheduler,
            persistent_num_splits=persistent_num_splits,
            explicit_split_k_chunk=explicit_split_k_chunk,
            requires_tma_alignment=(
                (inputs.is_paged and not paged_kv_non_tma)
                or (decision.family is KernelFamily.LONG and not persistent_split_kv)
            ),
            log_plan=config.log_plan,
        )

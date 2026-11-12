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

"""Cached paged-KV routing and adaptive Split-KV calculations for Hopper FA3."""

from functools import lru_cache

from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.config import (
    PAGED_GATHER_AUTO,
    PAGED_GATHER_BLOCKWISE,
    PAGED_GATHER_LEGACY,
    _ceil_div,
    _next_power_of_2,
)

MIN_EXPLICIT_SPLIT_K = 12 * 128
MIN_AUTO_SPLIT_K = 4 * 128


def _splits_for_target_wave(base_work: int, num_sms: int) -> int:
    """Choose a power-of-two split count without overfilling a near-full wave."""
    upper_splits = _next_power_of_2(_ceil_div(num_sms, base_work))
    lower_splits = upper_splits // 2
    if lower_splits >= 1 and base_work * lower_splits * 10 >= num_sms * 9:
        return lower_splits
    return upper_splits


def _splits_for_exact_target_wave(base_work: int, num_sms: int) -> int:
    """Use an integer split only when it leaves a nearly full first wave."""
    lower_splits = max(1, num_sms // base_work)
    if base_work * lower_splits * 10 >= num_sms * 9:
        return lower_splits
    return _splits_for_target_wave(base_work, num_sms)


def select_decode_schedule(
    batch_size, num_heads_k, max_seqlen_k, num_sms, num_splits, allow_split
):
    """Balance packed heads and KV parallelism for single-token GQA decode."""
    packed_work = batch_size * num_heads_k
    split_count = 1
    if (
        allow_split
        and num_splits != 1
        and max_seqlen_k >= 1024
        and packed_work * 2 < num_sms
    ):
        split_count = min(
            num_splits or 128,
            _ceil_div(max_seqlen_k, 128),
            _splits_for_target_wave(packed_work, num_sms),
        )
    return packed_work * 2 >= num_sms or split_count > 1, split_count


@lru_cache(maxsize=1024)
def select_paged_schedule(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    total_q,
    num_heads_k,
    head_dim,
    gqa_ratio,
    block_size,
    num_sms,
    is_fp16,
    is_causal,
    is_local_or_alibi,
    plain_scores,
    is_s_aux,
    qo_tma_aligned,
    kv_tma_aligned,
    population,
    workload,
    input_profile,
    pack_factor,
    paged_d256_prefill,
    num_splits,
):
    """Resolve coupled paged load, packing and split choices from scalar values.

    Only immutable launch controls are cached. Tensors, device sequence lengths,
    launch grids and mutable workspace belong to the current call.
    """
    wide_gqa = head_dim == 256 and (block_size, gqa_ratio) in {(16, 4), (32, 8)}
    default_d256 = (
        wide_gqa
        and (is_causal or max_seqlen_q == 1)
        and plain_scores
        and not is_s_aux
        and qo_tma_aligned
    )
    uniform_token_query = batch_size > 1 and max_seqlen_q == 1 and total_q == batch_size
    direct_token_query = uniform_token_query and (
        (block_size, gqa_ratio) == (32, 8)
        or (
            (block_size, gqa_ratio) == (16, 4)
            and batch_size * num_heads_k * 4 >= num_sms
        )
    )
    single_long_query = (
        population in {"ragged", "mixed"}
        and batch_size > 1
        and max_seqlen_q > 1
        and total_q == max_seqlen_q + batch_size - 1
    )
    dominant_long_query = (
        population in {"ragged", "mixed"}
        and batch_size > 1
        and max_seqlen_q > 1
        and total_q - max_seqlen_q <= 16 * (batch_size - 1)
    )
    # On 128+ SM parts, these small batches save more mapper work than padding.
    padded_prefill = (
        dominant_long_query
        and workload == "prefill"
        and batch_size <= 3
        and wide_gqa
        and num_sms >= 128
        and is_causal
        and plain_scores
    )
    ragged_scheduler = population == "ragged" and not padded_prefill

    if population == "uniform":
        wide_split_work = (
            _ceil_div(max_seqlen_q * gqa_ratio, 64) * batch_size * num_heads_k
        )
    elif single_long_query:
        # Host metadata proves that every query except the longest has one row.
        wide_split_work = (
            _ceil_div(max_seqlen_q * gqa_ratio, 64)
            + (batch_size - 1) * _ceil_div(gqa_ratio, 64)
        ) * num_heads_k
    else:
        # Bound the device-resident ragged lengths without synchronizing them.
        wide_split_work = (
            _ceil_div(total_q * gqa_ratio, 64) + batch_size - 1
        ) * num_heads_k
    padded_work = _ceil_div(max_seqlen_q * gqa_ratio, 128) * batch_size * num_heads_k
    near_full_decode = (
        is_fp16
        and workload == "token_query"
        and population == "uniform"
        and input_profile == "throughput_decode"
        and head_dim == 256
        and (block_size, gqa_ratio) == (16, 4)
        and plain_scores
        and padded_work < num_sms
        and padded_work * 10 >= num_sms * 9
    )
    prefer_direct = (default_d256 and direct_token_query) or near_full_decode
    explicit_wide_candidate = (
        wide_gqa
        and (population == "single" or workload == "short_query")
        and (population == "single" or wide_split_work < num_sms)
        and input_profile == "heavy"
        and workload in {"token_query", "short_query"}
        and (workload == "token_query" or is_causal)
        and plain_scores
    )
    auto_wide_split = (
        num_splits == 0
        and explicit_wide_candidate
        and workload == "short_query"
        and max_seqlen_k >= 8192
        and qo_tma_aligned
        and not is_s_aux
    )
    split_limit = num_splits
    if auto_wide_split:
        # Short queries need KV parallelism. Bound the launch by one wave,
        # useful KV work per slice, and the cost of combining partial outputs.
        split_limit = min(
            64,
            _ceil_div(max_seqlen_k, MIN_AUTO_SPLIT_K),
            _splits_for_target_wave(wide_split_work, num_sms),
        )
    base_pack_factor = pack_factor
    paged_prefill = workload == "prefill"
    for split_cap in (split_limit, 1):
        explicit_split_has_work = split_cap > 1 and max_seqlen_k > MIN_EXPLICIT_SPLIT_K
        explicit_wide_split = explicit_split_has_work and explicit_wide_candidate
        pack_factor = gqa_ratio if explicit_wide_split else base_pack_factor
        packed = pack_factor > 1
        tile_rows = (total_q if ragged_scheduler else max_seqlen_q) * pack_factor
        tile_heads = num_heads_k * (1 if packed else gqa_ratio)
        tile_groups = tile_heads if ragged_scheduler else batch_size * tile_heads
        tile_work = (
            _ceil_div(tile_rows, 128) + (batch_size - 1 if ragged_scheduler else 0)
        ) * tile_groups

        compact_d128 = block_size == 16 and packed and head_dim == 128
        compact_mha = (
            block_size == 16
            and gqa_ratio == 1
            and head_dim == 128
            and not is_local_or_alibi
        )
        if (
            block_size == 16
            and packed
            and gqa_ratio == 4
            and head_dim in (192, 256)
            and workload == "token_query"
        ):
            split_profile = "wide_decode"
        elif compact_d128 and gqa_ratio == 4 and workload == "speculative_query":
            split_profile = "short_spec"
        else:
            split_profile = "general"

        tma_gqa4 = block_size == 128 and packed and gqa_ratio == 4 and plain_scores
        tma_d128 = tma_gqa4 and head_dim == 128 and is_causal
        tma_wide_decode = tma_gqa4 and head_dim == 192 and workload == "token_query"
        prefer_tma = tma_d128 and input_profile == "large_single_prefill"
        prefer_long = (
            split_profile != "general"
            or (
                (compact_d128 or compact_mha)
                and (paged_prefill or (packed and workload == "serving_prefill"))
            )
            or (prefer_tma and paged_prefill)
        )
        kernel = "long" if prefer_long else "direct"
        # Direct decode overrides take precedence over explicit wide splitting.
        fixed_family = prefer_direct or explicit_wide_split
        if prefer_direct:
            kernel = "direct"
        elif explicit_wide_split:
            kernel = "long"
        elif tma_d128:
            kernel = (
                "long"
                if workload != "token_query"
                and (workload != "speculative_query" or tile_work < num_sms)
                else "direct"
            )
        elif paged_d256_prefill:
            kernel = "long"
        elif tma_wide_decode or split_profile != "general":
            kernel = "long" if tile_work < num_sms else "direct"
        use_tma = prefer_tma or (
            (tma_d128 or tma_wide_decode) and input_profile != "throughput_decode"
        )
        # A persistent kernel requires aligned Q/O. Ordinary paged loads do not
        # require aligned K/V; automatic selection may fall back to direct.
        if kernel == "long" and not qo_tma_aligned:
            if fixed_family:
                raise RuntimeError(
                    "FA3 route model found no feasible family/transport candidate"
                )
            kernel = "direct"
        paged_kv_non_tma = not (
            use_tma
            and qo_tma_aligned
            and kv_tma_aligned
            and block_size % (16 if kernel == "direct" else 64) == 0
        )
        split_supported = (split_cap == 0 and split_profile == "short_spec") or (
            explicit_split_has_work
            and (explicit_wide_split or split_profile in {"short_spec", "wide_decode"})
        )
        split_kv = (
            paged_kv_non_tma
            and population != "oversized"
            and kernel == "long"
            and packed
            and split_supported
        )
        persistent_num_splits = (
            split_cap if split_kv and split_cap > 1 else 3 if split_kv else 0
        )
        if (
            split_kv
            and not auto_wide_split
            and split_cap > 1
            and population != "single"
            and workload == "short_query"
            and head_dim == 256
        ):
            wave_splits = (
                _splits_for_exact_target_wave(wide_split_work, num_sms)
                if single_long_query
                else _splits_for_target_wave(wide_split_work, num_sms)
            )
            persistent_num_splits = min(
                persistent_num_splits,
                _ceil_div(max_seqlen_k, MIN_EXPLICIT_SPLIT_K),
                wave_splits,
            )
            if persistent_num_splits == 1:
                # Splitting enabled wide packing and family selection. Resolve
                # both again without splitting before deriving launch controls.
                continue
        break

    paged_gather_mode = PAGED_GATHER_AUTO
    if (
        kernel == "direct"
        and paged_kv_non_tma
        and packed
        and workload == "token_query"
        and head_dim == 192
        and block_size in {32, 256}
    ):
        paged_gather_mode = PAGED_GATHER_LEGACY
    elif (
        split_kv
        and split_cap > 1
        and (explicit_wide_split or split_profile == "wide_decode")
        and block_size == 16
    ):
        paged_gather_mode = PAGED_GATHER_BLOCKWISE
    elif paged_d256_prefill and kernel == "long":
        paged_gather_mode = PAGED_GATHER_BLOCKWISE
    elif (
        kernel == "long" and paged_kv_non_tma and packed and block_size in {16, 32, 256}
    ):
        paged_gather_mode = PAGED_GATHER_LEGACY

    if (
        split_kv
        and single_long_query
        and workload == "short_query"
        and head_dim == 256
        and (block_size, gqa_ratio) == (32, 8)
        and is_causal
        and plain_scores
    ):
        # This one-long-query cohort benefits from page-aligned KV loads.
        paged_gather_mode = PAGED_GATHER_BLOCKWISE
    explicit_split_k_chunk = (
        MIN_AUTO_SPLIT_K if auto_wide_split else MIN_EXPLICIT_SPLIT_K
    )
    if (
        split_kv
        and num_splits > 3
        and workload == "token_query"
        and block_size == 16
        and max_seqlen_k >= 1024
        and batch_size * num_heads_k * 4 < num_sms
    ):
        # The requested cap limits parallelism; choose the effective count
        # from device KV lengths even when the graph has a larger capacity.
        explicit_split_k_chunk = 128
    if (
        split_kv
        and persistent_num_splits == 32
        and population == "single"
        and workload == "short_query"
        and head_dim == 256
        and block_size == 16
        and gqa_ratio == 4
        and total_q == max_seqlen_q
        and 8 < max_seqlen_q <= 16
    ):
        # Preserve the measured Q11/Q12 chunk; unrelated splits retain 12*128.
        explicit_split_k_chunk = 8 * 128

    return (
        kernel,
        paged_kv_non_tma,
        pack_factor,
        ragged_scheduler,
        paged_gather_mode,
        persistent_num_splits,
        explicit_split_k_chunk,
    )

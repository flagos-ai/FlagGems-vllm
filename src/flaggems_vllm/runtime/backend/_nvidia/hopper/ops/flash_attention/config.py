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

"""FA3 configuration, autotune candidates, and launch geometry."""

from __future__ import annotations

import triton

# Values are passed unchanged to the kernel PAGED_GATHER_MODE constexpr.
PAGED_GATHER_LEGACY = 0
PAGED_GATHER_BLOCKWISE = 1
PAGED_GATHER_AUTO = 2


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def padded_head_dim(head_dim: int) -> int:
    return _next_power_of_2(head_dim)


def binary_heads_in_l2(policy: str | int) -> int:
    """Collapse an enabled policy for families without head swizzling."""
    return int(policy != 0)


def block_k(args):
    """Triton heuristic shared by direct and persistent kernels."""

    return padded_head_dim(args["d"])


def get_compact_m_upper(
    *, total_q: int, pack_factor: int, batch_size: int, block_m: int
) -> int:
    return _ceil_div(total_q * pack_factor, block_m) + batch_size - 1


DIRECT_GENERAL_BUCKET = 0
DIRECT_DENSE_DECODE_BUCKET = 11
DIRECT_PAGED_DECODE_BUCKET = 12
DIRECT_PAGED_PREFILL_BUCKET = 25
DIRECT_PAGED_PACKED_MEDIUM_BUCKET = 26
# Invalidate persisted decode tuning when its candidate policy changes.
DIRECT_DECODE_POLICY_VERSION = 2


def direct_configs():
    configs = []
    for block_m in (16, 32, 64, 128):
        for block_n in (16, 32, 64, 128, 256):
            stage_choices = (1, 2, 3) if block_m == 16 and block_n <= 128 else (2, 3)
            if block_n > 128:
                stage_choices = (3,)
            warp_choices = (4, 8) if block_n >= 64 else (4,)
            for num_stages in stage_choices:
                for num_warps in warp_choices:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )

    return configs


def prune_direct_configs(configs, nargs, **kwargs):
    q_ptr = kwargs.get("q_ptr", nargs.get("q_ptr"))
    batch_size = kwargs.get("b", nargs.get("b", 0))
    head_dim = kwargs.get("d", nargs.get("d"))
    is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
    max_seqlen_q = kwargs.get("seqlen_q", nargs.get("seqlen_q", 0))
    seqlen_k = kwargs.get("seqlen_k", nargs.get("seqlen_k", 0))
    total_q = kwargs.get("total_q", nargs.get("total_q", 0))
    h_hk_ratio = kwargs.get("h_hk_ratio", nargs.get("h_hk_ratio", 1))
    shape_bucket = kwargs.get(
        "DIRECT_SHAPE_BUCKET",
        nargs.get("DIRECT_SHAPE_BUCKET", DIRECT_DENSE_DECODE_BUCKET),
    )
    ragged_scheduler = kwargs.get(
        "RAGGED_SCHEDULER", nargs.get("RAGGED_SCHEDULER", False)
    )
    paged_kv_non_tma = kwargs.get(
        "PAGED_KV_NON_TMA", nargs.get("PAGED_KV_NON_TMA", True)
    )
    pack_gqa = kwargs.get("PACK_GQA", nargs.get("PACK_GQA", False))
    block_size = kwargs.get("block_size", nargs.get("block_size", 1))
    max_splits = kwargs.get("MAX_SPLITS", nargs.get("MAX_SPLITS", 1))

    # Cold-cache autotuning favors BM64/BN64 for this serving regime, but
    # repeated CUDA Graph measurements show that BF16 is consistently
    # faster with more Q parallelism and half as many KV loop iterations.
    # Keep the rule structural so nearby ragged serving batches benefit
    # without coupling scheduling to one benchmark name.
    short_ragged_bf16 = (
        str(getattr(q_ptr, "dtype", "")) == "torch.bfloat16"
        and shape_bucket == DIRECT_PAGED_PACKED_MEDIUM_BUCKET
        and is_paged
        and paged_kv_non_tma
        and ragged_scheduler
        and batch_size >= 32
        and max_seqlen_q <= 64
        and total_q <= batch_size * 8
        and 256 <= seqlen_k <= 1024
        and head_dim == 128
        and h_hk_ratio == 2
    )
    if short_ragged_bf16:
        measured = [
            config
            for config in configs
            if config.kwargs == {"BLOCK_M": 32, "BLOCK_N": 128}
            and config.num_warps == 4
            and config.num_stages == 2
        ]
        if measured:
            return measured

    # Cross-device exhaustive measurements agree on a complementary mapping
    # for compact-page D256 packed decode: page32 favors 4 warps while page16
    # favors 8 warps.  Every BN128 and every extra pipeline stage was strictly
    # dominated on both H100 and H800.  Selecting the measured winner by page
    # size also avoids a noisy two-way cold tune that can choose the slower
    # steady-state configuration.
    paged_d256_packed_decode = (
        shape_bucket == DIRECT_PAGED_DECODE_BUCKET
        and is_paged
        and paged_kv_non_tma
        and pack_gqa
        and head_dim == 256
        and max_seqlen_q == 1
        and block_size in (16, 32)
        and max_splits == 1
    )
    if paged_d256_packed_decode:
        measured_num_warps = 8 if block_size == 16 else 4
        measured = [
            config
            for config in configs
            if config.kwargs == {"BLOCK_M": 16, "BLOCK_N": 64}
            and config.num_warps == measured_num_warps
            and config.num_stages == 1
        ]
        if measured:
            return measured

    if (
        not is_paged
        and not pack_gqa
        and max_seqlen_q == 1
        and seqlen_k <= 64
        and head_dim == 256
        and 1 < h_hk_ratio <= 16
        and h_hk_ratio & (h_hk_ratio - 1) != 0
    ):
        # One effective KV tile cannot benefit from three pipeline stages.
        # Keep both measured winners: cold tuning can distinguish 1 vs 2.
        measured = [
            config
            for config in configs
            if config.kwargs == {"BLOCK_M": 16, "BLOCK_N": 64}
            and config.num_warps == 4
            and config.num_stages in (1, 2)
        ]
        if measured:
            return measured

    kept = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        if block_n == 16 and (
            not is_paged or paged_kv_non_tma or block_size % block_n != 0
        ):
            continue
        if is_paged and not paged_kv_non_tma and block_size % block_n != 0:
            continue
        if shape_bucket in (DIRECT_DENSE_DECODE_BUCKET, DIRECT_PAGED_DECODE_BUCKET):
            page_tma_tile = is_paged and not paged_kv_non_tma and block_n == block_size
            if block_m != 16 or (block_n < 64 and not page_tma_tile):
                continue
            if head_dim >= 192 and block_n > 128:
                continue
            if is_paged and block_n > 128:
                continue
        elif shape_bucket == DIRECT_PAGED_PACKED_MEDIUM_BUCKET:
            if not is_paged:
                continue
            if block_m < 32 or block_n > 128:
                continue
            if seqlen_k <= 128 and block_n != 128:
                continue
            if head_dim > 128 and block_n > 64:
                continue
        elif shape_bucket == DIRECT_PAGED_PREFILL_BUCKET:
            if not is_paged:
                continue
            if block_m < 64 or block_n > 128:
                continue
            if head_dim > 128 and block_n > 64:
                continue
        else:
            if block_n > 128:
                continue
            if block_m < 64:
                continue
            if seqlen_k <= 128 and block_n < 64:
                continue
            if head_dim > 128 and block_n > 64:
                continue
        if head_dim > 192 and block_n > 128:
            continue
        kept.append(config)
    return kept or [configs[0]]


def direct_launch_plan(
    *,
    max_seqlen_k: int,
    batch_size: int,
    effective_max_q: int,
    effective_num_heads: int,
    num_sms: int,
    is_paged: bool,
    paged_prefill: bool,
    pack_gqa: bool,
) -> tuple[bool, bool, int]:
    batch_first_grid = (
        is_paged and pack_gqa and 1 < batch_size <= 8 and effective_max_q <= 256
    )
    medium_packed_q = is_paged and pack_gqa and effective_max_q > 64
    single_kv_tile = medium_packed_q and max_seqlen_k <= 128
    # The decode buckets only admit BM16. Multi-row queries without a packed
    # bucket of their own run faster with BM64 once those tiles fill a wave.
    long_unpacked_q = (
        effective_max_q >= 64
        and (not is_paged or not pack_gqa)
        and _ceil_div(effective_max_q, 64) * batch_size * effective_num_heads >= num_sms
    )
    if paged_prefill:
        shape_bucket = DIRECT_PAGED_PREFILL_BUCKET
    elif medium_packed_q:
        shape_bucket = DIRECT_PAGED_PACKED_MEDIUM_BUCKET
    elif long_unpacked_q:
        shape_bucket = DIRECT_GENERAL_BUCKET
    elif is_paged:
        shape_bucket = DIRECT_PAGED_DECODE_BUCKET
    else:
        shape_bucket = DIRECT_DENSE_DECODE_BUCKET
    return batch_first_grid, single_kv_tile, shape_bucket


# Bump when candidate policy or compiler lowering invalidates cached timings.
# This version separates candidates by SM count and includes native N80
# for wide prefill/Split-KV while retaining conservative lower-SM choices.
PERSISTENT_AUTOTUNE_POLICY_VERSION = 22
PERSISTENT_DEFAULT_BLOCK_M = 128
PERSISTENT_LPT_HEURISTIC_BLOCK_N = 128
PERSISTENT_LPT_L2_BYTES = 32 * 1024 * 1024
PERSISTENT_COMPACT_1P2C_SMEM_BUDGET_BYTES = 225 * 1024
PERSISTENT_MAX_ACTIVE_WGMMA_N = 128
PERSISTENT_LEGACY_ACTIVE_WGMMA_N = 64
PERSISTENT_STANDARD_SM_COUNT_PROFILE = 0
PERSISTENT_LARGE_SM_COUNT_PROFILE = 1
PERSISTENT_LARGE_SM_COUNT_THRESHOLD = 128
PERSISTENT_COMPACT_N_P50_TIE_FRACTION = 0.03
PERSISTENT_COMPACT_N_ROBUST_TIE_FRACTION = 0.05
PERSISTENT_AUTOTUNE_SPREAD_WEIGHT = 0.25


def select_sm_count_profile(num_sms: int) -> int:
    """Bucket Hopper devices by wave capacity for autotune policy."""

    return (
        PERSISTENT_LARGE_SM_COUNT_PROFILE
        if num_sms >= PERSISTENT_LARGE_SM_COUNT_THRESHOLD
        else PERSISTENT_STANDARD_SM_COUNT_PROFILE
    )


def select_stable_config(timings):
    """Select a quantile-stable winner with a generic compact-N tie break."""

    def scores(config):
        samples = timings[config]
        p50 = float(samples[0])
        p20 = float(samples[1])
        p80 = float(samples[2])
        robust = p50 + PERSISTENT_AUTOTUNE_SPREAD_WEIGHT * max(p80 - p20, 0.0)
        return p50, robust

    configs = tuple(timings)
    fastest_p50 = min(scores(config)[0] for config in configs)
    robust_best = min(configs, key=lambda config: scores(config)[1])
    robust_floor = scores(robust_best)[1]
    compact_ties = [
        config
        for config in configs
        if config.kwargs["ACTIVE_WGMMA_N"] > PERSISTENT_LEGACY_ACTIVE_WGMMA_N
        and scores(config)[0]
        <= fastest_p50 * (1.0 + PERSISTENT_COMPACT_N_P50_TIE_FRACTION)
        and scores(config)[1]
        <= robust_floor * (1.0 + PERSISTENT_COMPACT_N_ROBUST_TIE_FRACTION)
    ]
    if compact_ties:
        return min(
            compact_ties,
            key=lambda config: (
                scores(config)[1],
                -config.kwargs["ACTIVE_WGMMA_N"],
            ),
        )
    return robust_best


def combine_launch_plan(
    *,
    max_seqlen_q: int,
    head_dim: int,
    total_q: int,
    batch_size: int,
    num_heads: int | None = None,
    num_heads_k: int | None = None,
    block_size: int | None = None,
    explicit_split_k_chunk: int = 12 * 128,
    max_splits: int = 1,
) -> tuple[int, int, int, bool]:
    """Choose the Split-KV reduction tile and compact work mapping."""

    short_single_d256 = (
        head_dim == 256
        and batch_size == 1
        and total_q == max_seqlen_q
        and 8 < max_seqlen_q <= 16
        and explicit_split_k_chunk == 8 * 128
    )
    tiny_gqa8_single_long = (
        head_dim == 256
        and batch_size > 1
        and num_heads is not None
        and num_heads_k is not None
        and (block_size, num_heads // num_heads_k) == (32, 8)
        and 1 < max_seqlen_q <= 64
        and total_q == max_seqlen_q + batch_size - 1
    )
    max_block_m = (
        1
        if tiny_gqa8_single_long
        else (8 if head_dim == 256 and (batch_size > 1 or short_single_d256) else 64)
    )
    if max_splits > 32:
        # More partial outputs need smaller row tiles to keep the reduction
        # parallel and avoid long per-CTA load chains.
        max_block_m = min(max_block_m, 2)
    block_m = min(max_block_m, _next_power_of_2(max_seqlen_q))
    block_k = padded_head_dim(head_dim)
    rectangular_mblocks = _ceil_div(max_seqlen_q, block_m) * batch_size
    compact_mblocks = _ceil_div(total_q, block_m) + batch_size - 1
    return block_m, block_k, compact_mblocks, compact_mblocks < rectangular_mblocks


def max_active_wgmma_n_for_1p2c(head_dim: int) -> int:
    """Return the largest 16-aligned logical N within the 1P2C budget."""

    head_dim_padded = padded_head_dim(head_dim)
    # Q1 with two M64 consumer groups contributes Dpad*256 bytes.  K/V
    # double buffering contributes Dpad*8*N bytes.
    max_n = (PERSISTENT_COMPACT_1P2C_SMEM_BUDGET_BYTES // head_dim_padded - 256) // 8
    max_n = min(PERSISTENT_MAX_ACTIVE_WGMMA_N, max_n)
    return max(16, max_n // 16 * 16)


def get_active_wgmma_n_candidates(
    head_dim: int,
    *,
    tiled_extent_eligible: bool,
) -> tuple[int, ...]:
    """Return the head-dimension-aware 1P2C active-N search space.

    The 1P2C shared-memory payload is
    ``Dpad * (256 + 8 * N)`` bytes for FP16 with Q1/KV2.  The compact
    candidate is derived from the per-CTA budget rather than tied to a
    particular N.  N64 remains as the measured no-regression candidate.
    """

    compact_n = max_active_wgmma_n_for_1p2c(head_dim)
    if compact_n == PERSISTENT_MAX_ACTIVE_WGMMA_N:
        return (PERSISTENT_MAX_ACTIVE_WGMMA_N, PERSISTENT_LEGACY_ACTIVE_WGMMA_N)
    if tiled_extent_eligible and compact_n > PERSISTENT_LEGACY_ACTIVE_WGMMA_N:
        return (compact_n, PERSISTENT_LEGACY_ACTIVE_WGMMA_N)
    return (PERSISTENT_LEGACY_ACTIVE_WGMMA_N,)


def make_persistent_config(
    *,
    block_n: int,
    num_buffers_kv: int,
    block_m: int = PERSISTENT_DEFAULT_BLOCK_M,
    num_mma_groups: int = 2,
    use_tma_qo: bool = True,
    stagger_kv: bool = False,
    active_wgmma_n: int | None = None,
    rescale_o_before_pv: bool = False,
    early_cast_p: bool = False,
):
    num_buffers_q = 1
    num_mma_warps = 4 * num_mma_groups
    q_stage_capacity = num_mma_groups * num_buffers_q
    if active_wgmma_n is None:
        active_wgmma_n = block_n
    if not 16 <= active_wgmma_n <= block_n or active_wgmma_n % 16:
        raise ValueError("ACTIVE_WGMMA_N must be a multiple of 16 in [16, BLOCK_N]")
    config_kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "NUM_BUFFERS_Q": num_buffers_q,
        "NUM_BUFFERS_KV": num_buffers_kv,
        "NUM_MMA_WARPS": num_mma_warps,
        "NUM_MMA_GROUPS": num_mma_groups,
        "Q_STAGE_CAPACITY": q_stage_capacity,
        "USE_TMA_QO": use_tma_qo,
        "Q_PIPE_ASYNC": True,
        "REUSE_Q_SMEM_O": True,
        "USE_TMA_KV": True,
        "STAGGER_KV": stagger_kv,
        "ACTIVE_WGMMA_N": active_wgmma_n,
        "RESCALE_O_BEFORE_PV": rescale_o_before_pv,
        "EARLY_CAST_P": early_cast_p,
    }
    return triton.Config(
        config_kwargs,
        num_warps=4,
    )


def persistent_configs():
    configs = []

    def add_config(**config_kwargs):
        configs.append(make_persistent_config(**config_kwargs))

    add_config(block_n=128, num_buffers_kv=2)
    add_config(block_n=64, num_buffers_kv=2)
    add_config(
        block_n=64,
        num_buffers_kv=2,
        stagger_kv=True,
    )
    # The compact D256 candidate has one legal transport/topology after
    # the static contract and production pruner are applied: staggered
    # K/V, early FP16 P carry, and descriptor/TMA K/V transport.
    compact_active_n = max_active_wgmma_n_for_1p2c(256)
    if compact_active_n < PERSISTENT_MAX_ACTIVE_WGMMA_N:
        configs.append(
            make_persistent_config(
                block_n=PERSISTENT_MAX_ACTIVE_WGMMA_N,
                num_buffers_kv=2,
                stagger_kv=True,
                active_wgmma_n=compact_active_n,
                rescale_o_before_pv=False,
                early_cast_p=True,
            )
        )
    for block_n in (64, 128):
        add_config(
            block_m=64,
            block_n=block_n,
            num_buffers_kv=2,
            num_mma_groups=1,
            use_tma_qo=False,
        )
    return configs


def config_smem_bytes(config, head_dim: int, *, pack_gqa: bool = False) -> int:
    block_k = _next_power_of_2(head_dim)
    block_m = config.kwargs["BLOCK_M"]
    block_n = config.kwargs["ACTIVE_WGMMA_N"]
    num_groups = config.kwargs["NUM_MMA_GROUPS"]
    bm_split = block_m // num_groups
    q_elems = config.kwargs["Q_STAGE_CAPACITY"] * bm_split * block_k
    kv_elems = 2 * config.kwargs["NUM_BUFFERS_KV"] * block_n * block_k
    # TensorDescriptor.store(register_tensor) uses an implicit shared
    # staging tile for each consumer.  Packed GQA explicitly reuses Q
    # SMEM, while wide unpacked D256 takes the direct global-store path.
    implicit_o_elems = 0
    if (
        config.kwargs["USE_TMA_QO"]
        and block_k < 256
        and not (pack_gqa and config.kwargs.get("REUSE_Q_SMEM_O", False))
    ):
        implicit_o_elems = num_groups * bm_split * block_k
    return (q_elems + kv_elems + implicit_o_elems) * 2


def prune_persistent_configs(configs, nargs, **kwargs):
    head_dim = kwargs.get("d", nargs.get("d"))
    is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
    paged_kv_non_tma = kwargs.get(
        "PAGED_KV_NON_TMA", nargs.get("PAGED_KV_NON_TMA", True)
    )
    pack_gqa = kwargs.get("PACK_GQA", nargs.get("PACK_GQA", False))
    split_kv = kwargs.get("SPLIT_KV", nargs.get("SPLIT_KV", False))
    paged_prefill_profile = kwargs.get(
        "PAGED_PREFILL_PROFILE",
        nargs.get("PAGED_PREFILL_PROFILE", False),
    )
    block_size = kwargs.get("block_size", nargs.get("block_size", 1))
    seqlen_q = kwargs.get("seqlen_q", nargs.get("seqlen_q"))
    num_heads = kwargs.get("h", nargs.get("h"))
    num_heads_k = kwargs.get("hk", nargs.get("hk"))
    is_seqused_k = kwargs.get("is_seqused_k", nargs.get("is_seqused_k", False))
    is_causal = kwargs.get("is_causal", nargs.get("is_causal", False))
    is_local = kwargs.get("is_local", nargs.get("is_local", False))
    is_alibi = kwargs.get("is_alibi", nargs.get("is_alibi", False))
    is_softcap = kwargs.get("is_softcap", nargs.get("is_softcap", False))
    is_s_aux = kwargs.get("is_s_aux", nargs.get("is_s_aux", False))
    sm_count_profile = kwargs.get(
        "SM_COUNT_PROFILE",
        nargs.get("SM_COUNT_PROFILE", PERSISTENT_STANDARD_SM_COUNT_PROFILE),
    )
    decode_packgqa_ws = (
        is_paged
        and paged_kv_non_tma
        and pack_gqa
        and block_size == 16
        and seqlen_q == 1
        and head_dim in (192, 256)
    )
    spec_packgqa_ws = (
        is_paged
        and paged_kv_non_tma
        and pack_gqa
        and block_size == 16
        and seqlen_q is not None
        and 1 < seqlen_q <= 8
        and head_dim == 128
    )
    explicit_wide_split_ws = (
        split_kv
        and is_paged
        and paged_kv_non_tma
        and pack_gqa
        and block_size in (16, 32)
        and head_dim == 256
    )
    wide_paged_prefill_packgqa_ws = (
        paged_prefill_profile
        and not split_kv
        and is_paged
        and paged_kv_non_tma
        and pack_gqa
        and block_size in (16, 32)
        and head_dim == 256
        and is_causal
        and not (is_local or is_alibi or is_softcap)
    )
    tiled_extent_profile = wide_paged_prefill_packgqa_ws or explicit_wide_split_ws
    tiled_extent_policy = (
        tiled_extent_profile
        and num_heads is not None
        and num_heads_k not in (None, 0)
        and (block_size, num_heads // num_heads_k) in {(16, 4), (32, 8)}
        and is_seqused_k
        and not is_s_aux
    )
    large_sm_count_profile = sm_count_profile == PERSISTENT_LARGE_SM_COUNT_PROFILE
    active_wgmma_n_candidates = set(
        get_active_wgmma_n_candidates(
            head_dim,
            tiled_extent_eligible=tiled_extent_policy,
        )
    )
    wide_paged_nontma = (
        is_paged and paged_kv_non_tma and not pack_gqa and head_dim > 128
    )

    def page_extent_compatible(config):
        return (
            not is_paged
            or not paged_kv_non_tma
            or config.kwargs["ACTIVE_WGMMA_N"] % block_size == 0
            or (
                tiled_extent_policy
                and config.kwargs["ACTIVE_WGMMA_N"] != config.kwargs["BLOCK_N"]
            )
        )

    def smem_compatible(config):
        budget = (
            225 * 1024
            if config.kwargs["ACTIVE_WGMMA_N"] != config.kwargs["BLOCK_N"]
            else 220 * 1024
        )
        return config_smem_bytes(config, head_dim, pack_gqa=pack_gqa) <= budget

    def sm_profile_compatible(config):
        return not (
            large_sm_count_profile
            and tiled_extent_policy
            and config.kwargs["NUM_MMA_GROUPS"] == 2
            and config.kwargs["ACTIVE_WGMMA_N"] == PERSISTENT_LEGACY_ACTIVE_WGMMA_N
        )

    def selected_shape_config(config):
        stagger_kv = config.kwargs.get("STAGGER_KV", False)
        tiled_extent = config.kwargs["ACTIVE_WGMMA_N"] != config.kwargs["BLOCK_N"]
        rescale_o_before_pv = config.kwargs.get("RESCALE_O_BEFORE_PV", False)
        early_cast_p = config.kwargs.get("EARLY_CAST_P", False)
        if rescale_o_before_pv and not tiled_extent:
            return False
        if early_cast_p and not tiled_extent:
            return False
        if (stagger_kv or tiled_extent or rescale_o_before_pv) and not (
            wide_paged_prefill_packgqa_ws or explicit_wide_split_ws
        ):
            return False
        if tiled_extent and (
            not tiled_extent_policy
            or not config.kwargs["REUSE_Q_SMEM_O"]
            or not stagger_kv
            or not early_cast_p
        ):
            return False
        if decode_packgqa_ws or wide_paged_nontma:
            return (
                config.kwargs["BLOCK_M"] == 64
                and config.kwargs["BLOCK_N"] == 64
                and config.kwargs["NUM_BUFFERS_KV"] == 2
                and config.kwargs["NUM_MMA_GROUPS"] == 1
                and not config.kwargs["USE_TMA_QO"]
            )
        if explicit_wide_split_ws or wide_paged_prefill_packgqa_ws:
            group2 = (
                config.kwargs["BLOCK_M"] == 128
                and (
                    (config.kwargs["BLOCK_N"] == 64 and not tiled_extent)
                    or (config.kwargs["BLOCK_N"] == 128 and tiled_extent)
                )
                and config.kwargs["NUM_BUFFERS_KV"] == 2
                and config.kwargs["NUM_MMA_GROUPS"] == 2
                and config.kwargs["USE_TMA_QO"]
            )
            group1 = (
                config.kwargs["BLOCK_M"] == 64
                and config.kwargs["BLOCK_N"] == 64
                and config.kwargs["NUM_BUFFERS_KV"] == 2
                and config.kwargs["NUM_MMA_GROUPS"] == 1
                and not config.kwargs["USE_TMA_QO"]
                and not stagger_kv
                and not tiled_extent
            )
            return group2 or group1
        if spec_packgqa_ws:
            return (
                config.kwargs["BLOCK_M"] == 64
                and config.kwargs["BLOCK_N"] == 128
                and config.kwargs["NUM_BUFFERS_KV"] == 2
                and config.kwargs["NUM_MMA_GROUPS"] == 1
                and not config.kwargs["USE_TMA_QO"]
            )
        return config.kwargs["BLOCK_M"] == PERSISTENT_DEFAULT_BLOCK_M

    kept = [
        config
        for config in configs
        if selected_shape_config(config)
        and page_extent_compatible(config)
        and config.kwargs["USE_TMA_KV"]
        and sm_profile_compatible(config)
        and not (
            config.kwargs["NUM_MMA_GROUPS"] == 2
            and config.kwargs["ACTIVE_WGMMA_N"] not in active_wgmma_n_candidates
        )
        and not (
            is_paged
            and paged_kv_non_tma
            and pack_gqa
            and config.kwargs["BLOCK_N"] < 128
            and not (
                decode_packgqa_ws
                or explicit_wide_split_ws
                or wide_paged_prefill_packgqa_ws
            )
        )
        and (
            not is_paged
            or paged_kv_non_tma
            or block_size % config.kwargs["BLOCK_N"] == 0
        )
        and smem_compatible(config)
    ]
    if kept:
        return kept
    fallback = [
        config
        for config in reversed(configs)
        if config.kwargs["USE_TMA_KV"]
        and sm_profile_compatible(config)
        and page_extent_compatible(config)
        and smem_compatible(config)
        and config.kwargs["ACTIVE_WGMMA_N"] == config.kwargs["BLOCK_N"]
        and (
            wide_paged_prefill_packgqa_ws or not config.kwargs.get("STAGGER_KV", False)
        )
    ]
    if fallback:
        return fallback[:1]
    if is_paged and paged_kv_non_tma:
        raise ValueError(
            "paged non-TMA has no legal ACTIVE_WGMMA_N: it must be "
            "divisible by block_size and fit the shared-memory budget"
        )
    return [configs[-1]]


def persistent_launch_plan(
    *,
    heads_in_l2: str | int,
    allow_head_swizzle: bool,
    pack_gqa: bool,
    gqa_ratio: int,
    effective_num_heads: int,
    max_seqlen_k: int,
    head_size: int,
    element_size: int,
) -> tuple[int, int, int]:
    num_mma_groups = 2
    block_m = PERSISTENT_DEFAULT_BLOCK_M
    if not allow_head_swizzle:
        resolved_heads_in_l2 = binary_heads_in_l2(heads_in_l2)
    elif heads_in_l2 == "l2_auto":
        resolved_heads_in_l2 = 1
        l2_divisor = min(16, _next_power_of_2(gqa_ratio))
        l2_budget = PERSISTENT_LPT_L2_BYTES // l2_divisor
        kv_block_bytes = (
            PERSISTENT_LPT_HEURISTIC_BLOCK_N * (head_size + head_size) * element_size
        )
        max_kv_blocks_in_l2 = l2_budget // kv_block_bytes
        num_kv_blocks = _ceil_div(max_seqlen_k, PERSISTENT_LPT_HEURISTIC_BLOCK_N)
        for candidate in (16, 8, 4, 2):
            if num_kv_blocks * candidate <= max_kv_blocks_in_l2:
                resolved_heads_in_l2 = candidate
                break
        if not pack_gqa:
            resolved_heads_in_l2 *= gqa_ratio
    else:
        resolved_heads_in_l2 = heads_in_l2
    resolved_heads_in_l2 = min(resolved_heads_in_l2, effective_num_heads)
    return num_mma_groups, block_m, resolved_heads_in_l2

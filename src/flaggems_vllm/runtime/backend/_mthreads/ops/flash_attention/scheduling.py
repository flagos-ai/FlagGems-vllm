# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

_TARGET_SPLIT_CTAS = 4 * 60  # Four waves across the S5000's 60 MPs.
_MAX_PARTIAL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class AttentionPlan:
    family: str
    block_m: int
    block_n: int
    num_warps: int = 4
    num_stages: int = 1
    num_splits: int = 1
    head_pack: int = 1


def select_plan(
    max_seqlen_q,
    head_groups,
    *,
    max_seqlen_k=0,
    total_q=0,
    num_heads=0,
    batch_size=1,
    head_size=0,
):
    """A conservative heuristic seed for every exact-shape search space."""
    if max_seqlen_k >= 8192 and 0 < total_q <= 256 and max_seqlen_q <= 256:
        packed = max_seqlen_q == 1 and head_groups <= 16
        ctas = (
            batch_size * (num_heads // head_groups)
            if packed
            else max(1, (total_q + 31) // 32) * num_heads
        )
        target = max(2, (_TARGET_SPLIT_CTAS + ctas - 1) // ctas)
        splits = min(32, 1 << (target - 1).bit_length())
        bytes_per_split = total_q * num_heads * (head_size + 1) * 4
        splits = min(splits, _MAX_PARTIAL_BYTES // max(1, bytes_per_split))
        # Share one KV tile across grouped heads for short queries.
        # Long prefill retains its separate scheduling policy.
        head_pack = 8 if head_groups == 8 and not packed else 1
        if head_groups == 4 and head_size == 256 and not packed:
            head_pack = 4
        if splits >= 2:
            return AttentionPlan(
                "split_decode" if packed else "split_kv",
                16 if packed else 32,
                32,
                num_splits=splits,
                head_pack=head_pack,
            )
    if max_seqlen_q == 1 and head_groups <= 16:
        return AttentionPlan("packed_decode", 16, 32)
    rectangular_tiles = batch_size * ((max_seqlen_q + 31) // 32)
    compact_upper = (total_q + 31) // 32 + batch_size - 1
    if 16 <= batch_size <= 128 and rectangular_tiles > 1.5 * compact_upper:
        return AttentionPlan("compact", 32, 32)
    return AttentionPlan("direct", 32, 32)

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionPlan:
    family: str
    block_m: int
    block_n: int
    num_warps: int = 4
    num_stages: int = 1


def _select_legacy_plan(
    max_seqlen_q,
    head_groups,
    *,
    fast_paged=False,
    max_seqlen_k=0,
    batches=0,
    heads=0,
    page_size=0,
    page_columns=0,
):
    if fast_paged and max_seqlen_q == 1 and (max_seqlen_k == 1):
        return AttentionPlan("single_key", 8, 256)
    if max_seqlen_q == 1 and head_groups <= 16:
        return AttentionPlan("packed_decode", 16, 32)
    if fast_paged:
        if (
            batches == 1
            and page_size == 32
            and (8192 <= max_seqlen_q <= 16384)
            and (32768 <= max_seqlen_k <= 65536)
            and (heads == 16)
            and (head_groups == 8)
        ):
            return AttentionPlan("bounded_prefill_long", 64, 32)
        if (
            batches == 1
            and page_size == 32
            and (512 <= max_seqlen_q <= 2048)
            and (max_seqlen_k <= 2048)
            and (heads == 16)
            and (head_groups == 8)
        ):
            return AttentionPlan("single_prefill", 64, 64)
        if max_seqlen_q <= 64 and max_seqlen_k >= 8192 and (head_groups <= 16):
            return AttentionPlan("split", 32, 64)
        if (
            batches > 1
            and page_size == 16
            and (512 <= max_seqlen_q <= 2048)
            and (max_seqlen_k <= 2048)
            and (heads == 4)
            and (head_groups == 4)
        ):
            if batches <= 64 and 1 <= page_columns <= 128:
                return AttentionPlan("bounded_prefill", 64, 32)
            family = "compact_folded" if batches <= 64 else "folded"
            return AttentionPlan(family, 32, 32, 2)
        if (
            2 <= batches <= 64
            and page_size == 32
            and (2048 <= max_seqlen_q <= 8192)
            and (max_seqlen_k <= 8192)
            and (heads == 16)
            and (head_groups == 8)
        ):
            return AttentionPlan("compact_paged", 64, 64)
        if batches == 1 and max_seqlen_q <= 2048 and (heads <= 4):
            return AttentionPlan("paged", 64, 128, 8)
        if (
            batches > 1
            and page_size == 16
            and (max_seqlen_q >= 2048)
            and (max_seqlen_k >= 8192)
            and (heads == 4)
            and (head_groups == 4)
        ):
            return AttentionPlan("hybrid_equal", 64, 64)
        if batches > 1 and max_seqlen_k >= 8192 and (head_groups <= 16):
            return AttentionPlan("hybrid", 64, 64)
        return AttentionPlan("paged", 64, 64)
    return AttentionPlan("direct", 64, 32)


def select_plan(
    max_seqlen_q,
    head_groups,
    *,
    fast_paged=False,
    max_seqlen_k=0,
    batches=0,
    heads=0,
    page_size=0,
    page_columns=0,
):
    if (
        fast_paged
        and max_seqlen_q == 1
        and (page_size == 32)
        and (heads == 8)
        and (head_groups == 8)
        and (1 <= batches <= 8)
        and (8192 <= max_seqlen_k <= 65536)
    ):
        return AttentionPlan("tp2_decode", 16, 64, 2)
    if (
        fast_paged
        and page_size == 32
        and (heads == 8)
        and (head_groups == 8)
        and (2 <= batches <= 8)
        and (8192 <= max_seqlen_q <= 16384)
        and (8192 <= max_seqlen_k <= 65536)
    ):
        return AttentionPlan("tp2_mixed", 64, 32)
    if fast_paged:
        if (
            page_size == 16
            and heads == 4
            and (head_groups == 4)
            and (32 < batches <= 64)
            and (512 <= max_seqlen_q <= 2048)
            and (max_seqlen_k <= 2048)
        ):
            return AttentionPlan("ordered_prefill", 64, 32)
        if (
            2 <= batches <= 128
            and 512 <= max_seqlen_q <= 8192
            and (max_seqlen_k <= 8192)
            and (
                page_size == 32
                and heads == 16
                and (head_groups == 8)
                or (page_size == 16 and heads == 4 and (head_groups == 4))
            )
        ):
            unified = (
                page_size == 16
                and heads == 4
                and (head_groups == 4)
                and (batches <= 32)
                and (max_seqlen_q <= 2048)
                and (max_seqlen_k <= 2048)
            )
            return AttentionPlan(
                "mixed_unified" if unified else "mixed_prefill", 64, 32
            )
        if (
            2 <= batches <= 8
            and 8192 <= max_seqlen_q <= 16384
            and (32768 <= max_seqlen_k <= 65600)
            and (
                page_size == 32
                and heads == 16
                and (head_groups == 8)
                or (page_size == 16 and heads == 4 and (head_groups == 4))
            )
        ):
            return AttentionPlan("linear_prefill", 64, 32)
    return _select_legacy_plan(
        max_seqlen_q,
        head_groups,
        fast_paged=fast_paged,
        max_seqlen_k=max_seqlen_k,
        batches=batches,
        heads=heads,
        page_size=page_size,
        page_columns=page_columns,
    )

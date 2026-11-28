# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionPlan:
    family: str
    splits: int = 1


def select_plan(
    max_seqlen_q,
    head_groups,
    *,
    page=0,
    dim=0,
    total_q=0,
    max_k=0,
    num_seqs=1,
    kv_heads=1,
    plain=False,
    bf16=False,
):
    if (
        bf16
        and page == 64
        and dim == 256
        and plain
        and head_groups == 8
        and kv_heads == 1
    ):
        if max_seqlen_q == 1 and max_k >= 8192:
            return AttentionPlan("split", splits=16)
        if max_seqlen_q >= 4096 and num_seqs == 1:
            return AttentionPlan("tp2_prefill")
    if max_seqlen_q == 1 and head_groups <= 16:
        return AttentionPlan("packed_decode")
    if page == 64 and dim == 256 and plain:
        if max_k >= 8192 and head_groups in (1, 2, 4, 8, 16):
            splits = 32 if head_groups <= 4 else 8
            short_workspace = splits * total_q * head_groups * kv_heads * 257 * 4
            if (
                max_seqlen_q <= 128
                and total_q <= 256
                and short_workspace <= 64 * 1024 * 1024
            ):
                return AttentionPlan("split", splits=splits)

            workspace = splits * num_seqs * 128 * head_groups * kv_heads * 257 * 4
            if num_seqs > 1 and workspace <= 64 * 1024 * 1024:
                return AttentionPlan("hybrid", splits=splits)
        return AttentionPlan("paged")
    return AttentionPlan("direct")

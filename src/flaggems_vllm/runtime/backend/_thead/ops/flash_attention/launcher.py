# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import triton

from flaggems_vllm.runtime.backend._thead.ops.flash_attention import (
    common,
    decode,
    direct,
    mixed,
    prefill,
    prefill_tle,
    scheduling,
)


def long_prefill_prepare_flags(
    q, k, cuq, table, used, maxq, scale, qm, prefill_only=False
):
    batch, hk = (used.numel(), k.shape[-2])
    kw = max(1, table.shape[1])
    qw = max(1, triton.cdiv(maxq * (q.shape[1] // hk), 32))
    kp = torch.empty((batch, hk, kw), device=k.device, dtype=torch.float32)
    qp = torch.empty((batch, hk, qw), device=q.device, dtype=torch.float32)
    flags = torch.empty((batch, hk), device=q.device, dtype=torch.int8)
    cap = triton.cdiv(q.shape[0], qm) + batch - 1
    meta = torch.empty((5 * cap + 1,), device=q.device, dtype=torch.int32)
    norm_kernel = (
        prefill.page_norm_prefill if prefill_only else prefill.linear_metadata_page_norm
    )
    combine_kernel = (
        prefill.prefill_worklist_combine_prefill
        if prefill_only
        else prefill.linear_metadata_combine_norms
    )
    norm_kernel[max(qw, kw), batch, hk](
        q,
        k,
        cuq,
        table,
        used,
        qp,
        kp,
        q.shape[1],
        hk,
        qw,
        kw,
        *q.stride()[:2],
        *k.stride()[:3],
        table.stride(0),
        k.shape[1],
        32,
        num_warps=4,
        num_stages=1,
    )
    combine_kernel[batch * hk,](
        qp,
        kp,
        flags,
        cuq,
        used,
        meta,
        qw,
        kw,
        triton.next_power_of_2(qw),
        triton.next_power_of_2(kw),
        scale,
        hk,
        batch,
        triton.next_power_of_2(batch),
        qm,
        triton.next_power_of_2(max(1, triton.cdiv(maxq, qm))),
        cap,
        num_warps=4,
        num_stages=1,
    )
    return (flags, meta)


def long_prefill_launch_linear(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
    prefill_only=False,
):
    norm_k, norm_table = (k, table)
    k, v, table = prefill.linear_stage_prepare(
        k, v, table, usedk, cuq, "interleaved", 64
    )
    paged = table is not None
    packed = False
    hkv = k.shape[-2]
    batches = cuq.numel() - 1
    qm = plan.block_m // 4
    flags, meta = long_prefill_prepare_flags(
        q, norm_k, cuq, norm_table, usedk, maxq, scale, qm, prefill_only
    )
    grid = (triton.cdiv(q.shape[0], qm) + batches - 1, 1, q.shape[1] // 4)
    prefill.linear_prefill_attention_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *out.stride()[:2],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        max(64, triton.next_power_of_2(q.shape[2])),
        BATCHES=batches,
        FOLD=4,
        ORDER=0,
        KNORM=flags,
        WORKLIST=meta,
        CAPACITY=(meta.numel() - 1) // 5,
        WRITE_LSE=write_lse,
        SHORT_LIMIT=64,
        REVERSE=True,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )
    return flags


def long_prefill_launch_decode(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    splits=16,
    packed=True,
    short_limit=0,
):
    paged = table is not None
    pt = short_limit * (cuq.numel() - 1) if short_limit else q.shape[0]
    partial = torch.empty(
        (splits, pt, q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (splits, 2, q.shape[1], pt), device=q.device, dtype=torch.float32
    )
    hkv = k.shape[-2]
    grid = (
        splits
        * triton.cdiv(
            short_limit or maxq,
            plan.block_m // (q.shape[1] // hkv) if packed else plan.block_m,
        ),
        cuq.numel() - 1,
        hkv if packed else q.shape[1],
    )
    decode.splitkv_split_attention_kernel[grid](
        q,
        k,
        v,
        partial,
        partial_lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *partial.stride()[1:3],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        max(64, triton.next_power_of_2(q.shape[2])),
        splits,
        pt,
        short_limit,
    )
    decode.splitkv_merge_kernel[
        short_limit or q.shape[0], q.shape[1], cuq.numel() - 1 if short_limit else 1
    ](
        partial,
        partial_lse,
        out,
        lse,
        cuq,
        usedk,
        q.shape[0],
        pt,
        q.shape[1],
        q.shape[2],
        *out.stride()[:2],
        splits,
        triton.next_power_of_2(splits),
        triton.next_power_of_2(q.shape[2]),
        num_warps=4,
        PART_Q=short_limit,
    )


def long_prefill_launch_long_prefill(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
):
    args = (
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        maxq,
        scale,
        causal,
        window,
        softcap,
    )
    if mixed.supports(q, k, v):
        eligibility = long_prefill_launch_linear(
            *args, plan, write_lse=write_lse, prefill_only=True
        )
        mixed.tp1_mixed_short_launch(
            q, k, v, out, lse, cuq, usedk, table, eligibility, scale, write_lse
        )
        return
    long_prefill_launch_linear(*args, plan, write_lse=write_lse)
    long_prefill_launch_decode(
        *args,
        scheduling.AttentionPlan("packed_decode", 32, 64),
        splits=32,
        packed=True,
        short_limit=64,
    )


def async_long_launch_linear(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
    prefill_only=False,
    copy_slots=1,
):
    norm_k, norm_table = (k, table)
    k, v, table = prefill.linear_stage_prepare(
        k, v, table, usedk, cuq, "interleaved", 64
    )
    paged = table is not None
    packed = False
    hkv = k.shape[-2]
    batches = cuq.numel() - 1
    qm = plan.block_m // 4
    flags, meta = long_prefill_prepare_flags(
        q, norm_k, cuq, norm_table, usedk, maxq, scale, qm, prefill_only
    )
    grid = (triton.cdiv(q.shape[0], qm) + batches - 1, 1, q.shape[1] // 4)
    prefill_tle.async_prefill_attention_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *out.stride()[:2],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        max(64, triton.next_power_of_2(q.shape[2])),
        BATCHES=batches,
        FOLD=4,
        ORDER=0,
        KNORM=flags,
        WORKLIST=meta,
        CAPACITY=(meta.numel() - 1) // 5,
        WRITE_LSE=write_lse,
        SHORT_LIMIT=64,
        REVERSE=True,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )
    mixed.attention_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *out.stride()[:2],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        max(64, triton.next_power_of_2(q.shape[2])),
        BATCHES=batches,
        FOLD=4,
        ORDER=0,
        KNORM=flags,
        WORKLIST=meta,
        CAPACITY=(meta.numel() - 1) // 5,
        WRITE_LSE=write_lse,
        SHORT_LIMIT=64,
        REVERSE=True,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
        FIXED_MODE=0,
        VORDER=0,
    )
    return flags


def async_long_launch_long_prefill(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
    copy_slots=1,
):
    args = (
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        maxq,
        scale,
        causal,
        window,
        softcap,
    )
    if mixed.supports(q, k, v):
        eligibility = async_long_launch_linear(
            *args, plan, write_lse=write_lse, prefill_only=True, copy_slots=copy_slots
        )
        mixed.tp1_mixed_short_launch(
            q, k, v, out, lse, cuq, usedk, table, eligibility, scale, write_lse
        )
        return eligibility
    eligibility = async_long_launch_linear(
        *args, plan, write_lse=write_lse, copy_slots=copy_slots
    )
    long_prefill_launch_decode(
        *args,
        scheduling.AttentionPlan("packed_decode", 32, 64, 4),
        splits=32,
        packed=True,
        short_limit=64,
    )
    return eligibility


def mixed_prefill_prepare_flags(q, k, cuq, table, used, maxq, scale, qm, ordered=False):
    batch, hk = (used.numel(), k.shape[-2])
    kw = max(1, table.shape[1])
    qw = max(1, triton.cdiv(maxq * (q.shape[1] // hk), 32))
    kp = torch.empty((batch, hk, kw), device=k.device, dtype=torch.float32)
    qp = torch.empty((batch, hk, qw), device=q.device, dtype=torch.float32)
    flags = torch.empty((batch, hk), device=q.device, dtype=torch.int8)
    cap = triton.cdiv(q.shape[0], qm) + batch - 1
    meta = torch.empty((5 * cap + 1,), device=q.device, dtype=torch.int32)
    prefill.page_norm_prefill[max(qw, kw), batch, hk](
        q,
        k,
        cuq,
        table,
        used,
        qp,
        kp,
        q.shape[1],
        hk,
        qw,
        kw,
        *q.stride()[:2],
        *k.stride()[:3],
        table.stride(0),
        k.shape[1],
        32,
        num_warps=4,
        num_stages=1,
    )
    combine = (
        prefill.ordered_worklist_combine_prefill
        if ordered
        else prefill.prefill_worklist_combine_prefill
    )
    combine[batch * hk,](
        qp,
        kp,
        flags,
        cuq,
        used,
        meta,
        qw,
        kw,
        triton.next_power_of_2(qw),
        triton.next_power_of_2(kw),
        scale,
        hk,
        batch,
        triton.next_power_of_2(batch),
        qm,
        triton.next_power_of_2(max(1, triton.cdiv(maxq, qm))),
        cap,
        num_warps=4,
        num_stages=1,
    )
    return (flags, meta)


def launch_prefill(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
    unified=False,
    norm_block=32,
    global_order=False,
):
    paged = table is not None
    packed = False
    hkv = k.shape[-2]
    batches = cuq.numel() - 1
    qm = plan.block_m // 4
    ordered = plan.family == "ordered_prefill"
    if norm_block != 32 or global_order:
        flags, meta = prefill.short_metadata_prepare(
            q,
            k,
            cuq,
            table,
            usedk,
            maxq,
            scale,
            qm,
            prefill_only=not unified,
            ordered=global_order or plan.family == "ordered_prefill",
            norm_block=norm_block,
            norm_warps=4,
        )
    else:
        prepare = prefill.prepare_flags if unified else mixed_prefill_prepare_flags
        if ordered:
            flags, meta = mixed_prefill_prepare_flags(
                q, k, cuq, table, usedk, maxq, scale, qm, ordered=True
            )
        else:
            flags, meta = prepare(q, k, cuq, table, usedk, maxq, scale, qm)
    grid = (triton.cdiv(q.shape[0], qm) + batches - 1, 1, q.shape[1] // 4)
    kernel = (
        prefill.ordered_prefill_attention_kernel
        if ordered
        else prefill.bounded_prefill_attention_kernel
    )
    return kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *out.stride()[:2],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        max(64, triton.next_power_of_2(q.shape[2])),
        BATCHES=batches,
        FOLD=4,
        ORDER=0 if plan.family == "bounded_prefill_long" else 1,
        KNORM=flags,
        WORKLIST=meta,
        CAPACITY=(meta.numel() - 1) // 5,
        REVERSE=True,
        WRITE_LSE=write_lse,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )


def launch_singleq(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    splits=8,
    packed=True,
    short_limit=1,
    write_lse=True,
):
    paged = table is not None
    pt = short_limit * (cuq.numel() - 1) if short_limit else q.shape[0]
    partial = torch.empty(
        (splits, pt, q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (splits, 2, q.shape[1], pt), device=q.device, dtype=torch.float32
    )
    hkv = k.shape[-2]
    grid = (
        splits
        * triton.cdiv(
            short_limit or maxq,
            plan.block_m // (q.shape[1] // hkv) if packed else plan.block_m,
        ),
        cuq.numel() - 1,
        hkv if packed else q.shape[1],
    )
    decode.q1_decode_split_attention_kernel[grid](
        q,
        k,
        v,
        partial,
        partial_lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *partial.stride()[1:3],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        max(64, triton.next_power_of_2(q.shape[2])),
        splits,
        pt,
        short_limit,
    )
    decode.q1_decode_merge_kernel[
        short_limit or q.shape[0], q.shape[1], cuq.numel() - 1 if short_limit else 1
    ](
        partial,
        partial_lse,
        out,
        lse,
        cuq,
        usedk,
        q.shape[0],
        pt,
        q.shape[1],
        q.shape[2],
        *out.stride()[:2],
        splits,
        triton.next_power_of_2(splits),
        triton.next_power_of_2(q.shape[2]),
        num_warps=4,
        PART_Q=short_limit,
        WRITE_LSE=write_lse,
    )


def launch_mixed_prefill(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
):
    args = (
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        maxq,
        scale,
        causal,
        window,
        softcap,
    )
    unified = plan.family == "mixed_unified"
    launch_prefill(*args, plan, write_lse=write_lse, unified=unified, norm_block=32)
    if not unified:
        launch_singleq(
            *args,
            scheduling.AttentionPlan("packed_decode", 16, 32, 4),
            splits=8,
            write_lse=write_lse,
        )


async_small_launch_decode = long_prefill_launch_decode


def async_small_launch_linear(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
    prefill_only=False,
    copy_slots=1,
    norm_block=32,
):
    norm_k, norm_table = (k, table)
    k, v, table = prefill.async_stage_prepare(
        k, v, table, usedk, cuq, "interleaved", 64, prefill_only=prefill_only
    )
    paged = table is not None
    packed = False
    hkv = k.shape[-2]
    batches = cuq.numel() - 1
    qm = plan.block_m // 4
    if norm_block == 32:
        flags, meta = long_prefill_prepare_flags(
            q, norm_k, cuq, norm_table, usedk, maxq, scale, qm, prefill_only
        )
    else:
        flags, meta = prefill.short_metadata_prepare(
            q,
            norm_k,
            cuq,
            norm_table,
            usedk,
            maxq,
            scale,
            qm,
            prefill_only=prefill_only,
            ordered=False,
            norm_block=norm_block,
            norm_warps=4,
        )
    grid = (triton.cdiv(q.shape[0], qm) + batches - 1, 1, q.shape[1] // 4)
    prefill_tle.async_small_prefill_attention_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *out.stride()[:2],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        max(64, triton.next_power_of_2(q.shape[2])),
        BATCHES=batches,
        FOLD=4,
        ORDER=0,
        KNORM=flags,
        WORKLIST=meta,
        CAPACITY=(meta.numel() - 1) // 5,
        WRITE_LSE=write_lse,
        SHORT_LIMIT=64,
        REVERSE=True,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )
    mixed.attention_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        hkv,
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        *out.stride()[:2],
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
        plan.block_m,
        plan.block_n,
        max(64, triton.next_power_of_2(q.shape[2])),
        BATCHES=batches,
        FOLD=4,
        ORDER=0,
        KNORM=flags,
        WORKLIST=meta,
        CAPACITY=(meta.numel() - 1) // 5,
        WRITE_LSE=write_lse,
        SHORT_LIMIT=64,
        REVERSE=True,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
        FIXED_MODE=0,
        VORDER=0,
    )
    return flags


def async_small_launch_long_prefill(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
    copy_slots=1,
):
    args = (
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        maxq,
        scale,
        causal,
        window,
        softcap,
    )
    if table.shape[1] * k.shape[1] >= 8192:
        return async_long_launch_long_prefill(
            *args, plan, write_lse=write_lse, copy_slots=copy_slots
        )
    eligibility = async_small_launch_linear(
        *args,
        plan,
        write_lse=write_lse,
        prefill_only=True,
        copy_slots=copy_slots,
        norm_block=32,
    )
    if not (cuq.numel() == 2 and q.shape[0] > 1):
        launch_singleq(
            *args,
            scheduling.AttentionPlan("packed_decode", 16, 32, 4),
            splits=8,
            write_lse=write_lse,
        )
    return eligibility


def launch_short_prefill(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
):
    args = (
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        maxq,
        scale,
        causal,
        window,
        softcap,
    )
    batch = cuq.numel() - 1
    norm_block = 32
    q1_plan = scheduling.AttentionPlan("packed_decode", 16, 32, 2)
    if q.shape[1] == 16:
        async_small_launch_linear(
            *args,
            scheduling.AttentionPlan("linear_prefill", 64, 32, 4, 1),
            write_lse=write_lse,
            prefill_only=True,
            norm_block=64,
        )
        launch_singleq(
            *args,
            scheduling.AttentionPlan("packed_decode", 16, 64, 4),
            splits=2,
            write_lse=write_lse,
        )
    elif batch <= 32:
        launch_prefill(
            *args,
            scheduling.AttentionPlan("mixed_unified", 64, 32, 4, 1),
            write_lse=write_lse,
            unified=True,
            global_order=True,
            norm_block=norm_block,
        )
    elif batch <= 64:
        launch_mixed_prefill(
            *args,
            scheduling.AttentionPlan("ordered_prefill", 64, 32, 4, 1),
            write_lse=write_lse,
        )
    elif q.shape[0] >= 8192:
        async_small_launch_linear(
            *args,
            scheduling.AttentionPlan("linear_prefill", 64, 32, 4, 1),
            write_lse=write_lse,
            prefill_only=True,
            norm_block=norm_block,
        )
        launch_singleq(*args, q1_plan, splits=8, write_lse=write_lse)
    else:
        launch_prefill(
            *args,
            scheduling.AttentionPlan("ordered_prefill", 64, 32, 4, 1),
            write_lse=write_lse,
            unified=False,
            norm_block=norm_block,
        )
        launch_singleq(*args, q1_plan, splits=8, write_lse=write_lse)


def tp2_mixed_launch_linear(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    write_lse=True,
    fold=4,
    order=0,
    mode=-1,
    short=64,
    early=True,
    norm_short=64,
    vorder=0,
):
    paged = table is not None
    packed = False
    hkv = k.shape[-2]
    batches = cuq.numel() - 1
    qm = plan.block_m // fold
    k, v, table, flags, meta = mixed.prepare(
        q,
        k,
        v,
        cuq,
        table,
        usedk,
        maxq,
        scale,
        qm,
        short=short,
        norm_short=norm_short,
        early=early,
    )
    grid = (triton.cdiv(q.shape[0], qm) + batches - 1, 1, q.shape[1] // fold)
    for kernel_mode in (1, 0) if mode == 2 else (mode,):
        kernel = (
            prefill_tle.async_prefill_attention_kernel
            if kernel_mode == 1
            else mixed.attention_kernel
        )
        extra = {} if kernel_mode == 1 else dict(VORDER=vorder)
        kernel[grid](
            q,
            k,
            v,
            out,
            lse,
            cuq,
            cuk,
            usedk,
            table,
            alibi,
            q.shape[0],
            q.shape[1],
            hkv,
            q.shape[2],
            *q.stride()[:2],
            k.stride(0) if paged else 0,
            k.stride(-3),
            k.stride(-2),
            v.stride(0) if paged else 0,
            v.stride(-3),
            v.stride(-2),
            *out.stride()[:2],
            table.stride(0) if paged else 0,
            alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
            k.shape[1] if paged else 1,
            scale,
            causal,
            *window,
            softcap,
            paged,
            alibi is not None,
            packed,
            plan.block_m,
            plan.block_n,
            max(64, triton.next_power_of_2(q.shape[2])),
            BATCHES=batches,
            FOLD=fold,
            ORDER=order,
            KNORM=flags,
            WORKLIST=meta,
            CAPACITY=(meta.numel() - 1) // 5,
            WRITE_LSE=write_lse,
            SHORT_LIMIT=short,
            REVERSE=True,
            num_warps=plan.num_warps,
            num_stages=plan.num_stages,
            FIXED_MODE=kernel_mode,
            **extra,
        )
    return flags


def launch_tp2_mixed(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    maxq,
    scale,
    causal,
    window,
    softcap,
    plan,
    *,
    max_seqlen_k,
    write_lse=True,
):
    batches = cuq.numel() - 1
    use_medium = batches <= 5
    short = 128 if use_medium else 64
    flags = tp2_mixed_launch_linear(
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        maxq,
        scale,
        causal,
        window,
        softcap,
        plan,
        write_lse=write_lse,
        fold=4,
        order=1,
        mode=2,
        short=short,
        early=True,
        norm_short=1 if use_medium else 64,
        vorder=2,
    )
    if use_medium:
        mixed.tp2_mixed_medium_launch(
            q,
            k,
            v,
            out,
            lse,
            cuq,
            usedk,
            table,
            max_seqlen_k=max_seqlen_k,
            flags=flags,
            scale=scale,
            limit=128,
            splits=32,
            workers=512,
            order=False,
            bm=32,
            bn=64,
            head_major=True,
            early_k=True,
            early_v=True,
            write_lse=write_lse,
        )
    else:
        mixed.tp2_mixed_short_launch(
            q,
            k,
            v,
            out,
            lse,
            cuq,
            usedk,
            table,
            max_seqlen_k=max_seqlen_k,
            scale=scale,
            limit=64,
            bm=16,
            bn=64,
            splits=64,
            workers=512,
            write_lse=write_lse,
        )
    return flags


def launch_attention(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits=0,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    fa_version=2,
):
    if dropout_p != 0 or return_attn_probs:
        raise NotImplementedError("Dropout and attention probabilities are unsupported")
    if any(
        (
            x is not None
            for x in (
                q_v,
                scheduler_metadata,
                q_descale,
                k_descale,
                v_descale,
                s_aux,
                cp_tot_seqused_k,
            )
        )
    ):
        raise NotImplementedError("FA3 auxiliary inputs are unsupported")
    if num_splits != 0 or cp_world_size != 1 or cp_rank != 0 or (fa_version != 2):
        raise NotImplementedError(
            "Explicit splits, context parallel and FA3 are unsupported"
        )
    common.validate_inputs(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        block_table,
        out,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
    )
    window = (-1, -1) if window_size is None else tuple(window_size)
    if len(window) != 2 or any((not isinstance(x, int) or x < -1 for x in window)):
        raise ValueError("window_size must contain two integers >= -1")
    scale = q.shape[-1] ** (-0.5) if softmax_scale is None else float(softmax_scale)
    softcap = float(softcap)
    if not math.isfinite(scale) or not math.isfinite(softcap) or softcap < 0:
        raise ValueError("Scale must be finite and softcap finite/nonnegative")
    fast_paged = (
        q.dtype == torch.bfloat16
        and q.shape[-1] == 256
        and (block_table is not None)
        and (k.shape[1] in (16, 32, 64))
        and causal
        and (window == (-1, -1))
        and (softcap == 0)
        and (alibi_slopes is None)
    )
    plan = scheduling.select_plan(
        max_seqlen_q,
        q.shape[1] // k.shape[-2],
        fast_paged=fast_paged,
        max_seqlen_k=max_seqlen_k,
        batches=cu_seqlens_q.numel() - 1,
        heads=q.shape[1],
        page_size=k.shape[1] if block_table is not None else 0,
        page_columns=block_table.shape[1] if block_table is not None else 0,
    )
    if plan.family in ("tp2_decode", "tp2_mixed") and any(
        (x.stride(i) % 256 for x in (k, v) for i in (0, 1))
    ):
        plan = (
            scheduling.AttentionPlan("packed_decode", 16, 32)
            if max_seqlen_q == 1
            else scheduling.AttentionPlan("hybrid", 64, 64)
        )
    async_route = None
    batches = cu_seqlens_q.numel() - 1
    if fast_paged and 512 <= max_seqlen_q <= 16384 and (1 <= batches <= 128):
        if (
            q.shape[1] == 8
            and k.shape[-2] == 1
            and (k.shape[1] == 32)
            and (max_seqlen_q >= 1024)
            and (8192 <= max_seqlen_k <= 65536)
            and (batches <= 8)
            and all((x.stride(i) % 256 == 0 for x in (k, v) for i in (0, 1)))
        ):
            plan = scheduling.AttentionPlan("tp2_mixed", 64, 32)
        elif (
            q.shape[1] == 16
            and k.shape[-2] == 2
            and (k.shape[1] == 32)
            or (q.shape[1] == 4 and k.shape[-2] == 1 and (k.shape[1] == 16))
        ):
            if block_table.shape[1] * k.shape[1] >= 8192:
                async_route = async_long_launch_long_prefill
            elif (
                not (max_seqlen_k <= 2048 and (batches == 1 or q.shape[1] == 4))
                and q.stride(1) >= q.shape[-1]
                and (q.stride(0) >= (q.shape[1] - 1) * q.stride(1) + q.shape[-1])
                and (q.data_ptr() % 16 == 0)
                and (q.stride(0) % 8 == q.stride(1) % 8 == 0)
            ):
                async_route = async_small_launch_long_prefill
            if async_route is not None:
                plan = scheduling.AttentionPlan("linear_prefill", 64, 32)
    if (
        fast_paged
        and 512 <= max_seqlen_q <= 2048
        and (max_seqlen_k <= 2048)
        and (2 <= batches <= 128)
        and (block_table.shape[1] * k.shape[1] < 8192)
        and (q.stride(1) >= q.shape[-1])
        and (q.stride(0) >= (q.shape[1] - 1) * q.stride(1) + q.shape[-1])
        and (q.data_ptr() % 16 == 0)
        and (q.stride(0) % 8 == q.stride(1) % 8 == 0)
        and (
            q.shape[1] == 4
            and k.shape[-2] == 1
            and (k.shape[1] == 16)
            or (
                q.shape[1] == 16
                and k.shape[-2] == 2
                and (k.shape[1] == 32)
                and (batches > 64)
                and (q.shape[0] < 8192)
            )
        )
    ):
        plan = scheduling.AttentionPlan("short_prefill", 64, 32, 4, 1)
    with torch.cuda.device(q.device):
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse = torch.empty(
            (q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device
        )
        if q.shape[0]:
            launch = direct.launch_direct
            if plan.family == "short_prefill":
                launch = launch_short_prefill
            elif plan.family == "tp2_decode":
                launch = decode.launch_tp2_decode
            elif plan.family == "tp2_mixed":
                launch = launch_tp2_mixed
            elif plan.family in ("mixed_prefill", "mixed_unified", "ordered_prefill"):
                launch = launch_mixed_prefill
            elif plan.family == "linear_prefill":
                launch = async_route or long_prefill_launch_long_prefill
            elif plan.family == "single_key":
                launch = direct.launch_single_key
            elif plan.family == "single_prefill":
                launch = direct.launch_single_prefill
            elif plan.family in ("bounded_prefill", "bounded_prefill_long"):
                launch = prefill.launch_bounded_prefill
            elif plan.family in ("compact_paged", "compact_folded"):
                launch = direct.launch_compact
            elif plan.family == "folded":
                launch = direct.launch_folded
            elif plan.family == "split":
                launch = decode.launch_split
            elif plan.family == "hybrid_equal":
                launch = direct.launch_equal_length
            elif plan.family in ("paged", "hybrid"):
                launch = direct.paged_launch_paged
            extra = {"splits": 32, "packed": True} if plan.family == "split" else {}
            if plan.family in (
                "tp2_decode",
                "tp2_mixed",
                "single_key",
                "single_prefill",
                "bounded_prefill",
                "bounded_prefill_long",
                "mixed_prefill",
                "mixed_unified",
                "ordered_prefill",
                "linear_prefill",
                "short_prefill",
            ):
                extra = {"write_lse": return_softmax_lse}
            if plan.family in ("tp2_decode", "tp2_mixed"):
                extra["max_seqlen_k"] = max_seqlen_k
            if plan.family in ("hybrid", "hybrid_equal"):
                extra = {"short_limit": 64}
            launch(
                q,
                k,
                v,
                out,
                lse,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                scale,
                causal,
                window,
                softcap,
                plan,
                **extra,
            )
            if plan.family in ("hybrid", "hybrid_equal"):
                decode.launch_split(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    seqused_k,
                    block_table,
                    alibi_slopes,
                    max_seqlen_q,
                    scale,
                    causal,
                    window,
                    softcap,
                    scheduling.AttentionPlan("split", 32, 64),
                    splits=32,
                    packed=True,
                    short_limit=64,
                )
    return (out, lse) if return_softmax_lse else out

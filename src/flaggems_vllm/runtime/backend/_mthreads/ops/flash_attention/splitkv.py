# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable MetaX Split-KV idea with S5000 launch sizes and v1 LSE semantics."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.bounds import (
    prepare_key_norms,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.direct import (
    launch_direct,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import launch


@triton.jit
def merge_splits_kernel(
    OUT,
    LSE,
    PART_O,
    PART_LSE,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    splits = tl.arange(0, BS)
    ls = tl.load(
        PART_LSE + splits * HQ * TQ + head * TQ + row, splits < SPLITS, -float("inf")
    )
    valid = (splits < SPLITS) & (ls > -float("inf")) & (ls < float("inf"))
    maximum = tl.max(tl.where(valid, ls, -float("inf")), 0)
    safe_maximum = tl.where(maximum > -float("inf"), maximum, 0.0)
    weights = tl.exp(tl.where(valid, ls - safe_maximum, -float("inf")))
    denominator = tl.sum(weights, 0)
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    dims = tl.arange(0, BD)
    offsets = (splits[:, None].to(tl.int64) * TQ * HQ + row * HQ + head) * D + dims[
        None, :
    ]
    partial = tl.load(PART_O + offsets, valid[:, None] & (dims[None, :] < D), 0.0)
    result = tl.sum(partial * (weights / safe_denominator)[:, None], 0)
    tl.store(OUT + row.to(tl.int64) * OS + head * OH + dims, result, dims < D)
    final_lse = tl.where(
        denominator > 0, safe_maximum + tl.log(safe_denominator), float("inf")
    )
    tl.store(LSE + head * TQ + row, final_lse)


def launch_splitkv(
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
    maxk=None,
):
    """The graph owns its partial buffers; every valid row is written per replay."""
    splits = plan.num_splits
    if not 2 <= splits <= 32:
        raise ValueError("MThreads Split-KV requires 2 to 32 splits")
    total, heads, dim = q.shape
    partial_out = torch.empty(
        (splits, total, heads, dim), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty(
        (splits, heads, total), dtype=torch.float32, device=q.device
    )
    norms, norm_tiles = None, 1
    if (
        tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 6)
        and q.dtype == torch.bfloat16
        and dim == 256
        and plan.head_pack in (4, 8)
        and q.shape[1] // k.shape[-2] == plan.head_pack
        and table is not None
        and table.shape[1] * k.shape[1] >= 8192
        and window == (-1, -1)
        and softcap == 0
        and alibi is None
        and scale > 0
    ):
        norms, norm_tiles = prepare_key_norms(k, table, usedk, maxk)
    compiled = launch_direct(
        q,
        k,
        v,
        partial_out,
        partial_lse,
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
        norm=norms,
        norm_tiles=norm_tiles,
    )
    launch(
        merge_splits_kernel,
        ((total, heads)),
        out,
        lse,
        partial_out,
        partial_lse,
        total,
        heads,
        dim,
        *out.stride()[:2],
        splits,
        triton.next_power_of_2(splits),
        triton.next_power_of_2(dim),
        num_warps=4,
        num_stages=1,
    )
    return compiled

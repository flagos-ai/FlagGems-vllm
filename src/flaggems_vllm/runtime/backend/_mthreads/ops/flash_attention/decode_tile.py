# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-tile paged decode; GQA heads share one QK/PV operation.

For a request with one query, right-aligned causal and noncausal attention
both see the entire KV cache. Empty KV produces zero OUT and positive-inf LSE.
Length and page-table contents are read on every invocation and Graph replay.
"""
import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.direct import (
    _BACKEND_OPT,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import launch


def supports_decode_tile(q, k, table, maxq, window, softcap, alibi, plan):
    return (
        table is not None
        and q.dtype == torch.bfloat16
        and q.shape[2] == 256
        and maxq == 1
        and q.shape[1] // k.shape[-2] in (4, 8)
        and 0 < table.shape[1] * k.shape[1] <= 32
        and window == (-1, -1)
        and not softcap
        and alibi is None
        and plan.num_splits == 1
        and q.is_contiguous()
    )


@triton.jit
def decode_tile(
    Q,
    K,
    V,
    OUT,
    LSE,
    CUQ,
    USEDK,
    TABLE,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    TQ: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    WRITE_FLAGS=None,
):
    batch, kh = tl.program_id(0), tl.program_id(1)
    qs = tl.load(CUQ + batch)
    qe = tl.load(CUQ + batch + 1)
    if qs == qe:
        return
    length = tl.load(USEDK + batch)
    rows = tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, BD)
    groups: tl.constexpr = HQ // HK
    heads = kh * groups + rows
    q = tl.load(
        Q + (qs.to(tl.int64) * HQ + heads[:, None]) * D + d[None, :],
        (rows[:, None] < groups) & (d[None, :] < D),
        0,
    )
    page = tl.load(TABLE + batch * TS + n // PAGE, n < length, 0)
    ko = page.to(tl.int64) * KP + n % PAGE * KS + kh * KH
    vo = page.to(tl.int64) * VP + n % PAGE * VS + kh * VH
    k = tl.load(
        K + ko[None, :] + d[:, None], (n[None, :] < length) & (d[:, None] < D), 0
    )
    scores = tl.dot(q, k) * (SCALE * 1.4426950408889634)
    scores = tl.where(
        (rows[:, None] < groups) & (n[None, :] < length), scores, -float("inf")
    )
    maximum = tl.max(scores, 1)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    p = tl.exp2(scores - safe[:, None])
    den = tl.sum(p, 1)
    v = tl.load(
        V + vo[:, None] + d[None, :], (n[:, None] < length) & (d[None, :] < D), 0
    )
    acc = tl.dot(p.to(v.dtype), v)
    out = acc / tl.where(den > 0, den, 1.0)[:, None]
    ls = tl.where(den > 0, (safe + tl.log2(den)) * 0.6931471805599453, float("inf"))
    write_rows = rows < groups
    if WRITE_FLAGS is not None:
        write_rows &= tl.load(WRITE_FLAGS + batch * HQ + heads, rows < groups, 0) != 0
    tl.store(
        OUT + (qs.to(tl.int64) * HQ + heads[:, None]) * D + d[None, :],
        out,
        write_rows[:, None] & (d[None, :] < D),
    )
    tl.store(LSE + heads * TQ + qs, ls, write_rows)


@triton.jit
def exceptional_decode_kernel(
    Q,
    K,
    V,
    OUT,
    LSE,
    FLAGS,
    CUQ,
    USEDK,
    TABLE,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    TQ: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
):
    batch, kh = tl.program_id(0), tl.program_id(1)
    heads = kh * (HQ // HK) + tl.arange(0, HQ // HK)
    if tl.sum(tl.load(FLAGS + batch * HQ + heads), 0) > 0:
        decode_tile(
            Q,
            K,
            V,
            OUT,
            LSE,
            CUQ,
            USEDK,
            TABLE,
            HQ,
            HK,
            256,
            TQ,
            KP,
            KS,
            KH,
            VP,
            VS,
            VH,
            TS,
            PAGE,
            SCALE,
            16,
            32,
            256,
            FLAGS,
        )


def launch_decode_exceptions(q, k, v, out, lse, flags, cuq, usedk, table, scale):
    """Recompute flagged heads with the established MMA nonfinite semantics."""
    return launch(
        exceptional_decode_kernel,
        (table.shape[0], k.shape[2]),
        q,
        k,
        v,
        out,
        lse,
        flags,
        cuq,
        usedk,
        table,
        q.shape[1],
        k.shape[2],
        q.shape[0],
        *k.stride()[:3],
        *v.stride()[:3],
        table.stride(0),
        k.shape[1],
        scale,
        num_warps=4,
        num_stages=1,
        **{_BACKEND_OPT: False},
    )


def launch_decode_tile(q, k, v, out, lse, cuq, usedk, table, scale):
    """Validated BM16/BN32, four-warps configuration (offline tuned)."""
    return launch(
        decode_tile,
        (table.shape[0], k.shape[2]),
        q,
        k,
        v,
        out,
        lse,
        cuq,
        usedk,
        table,
        q.shape[1],
        k.shape[2],
        q.shape[2],
        q.shape[0],
        *k.stride()[:3],
        *v.stride()[:3],
        table.stride(0),
        k.shape[1],
        scale,
        16,
        32,
        256,
        num_warps=4,
        num_stages=1,
        **{_BACKEND_OPT: False},
    )

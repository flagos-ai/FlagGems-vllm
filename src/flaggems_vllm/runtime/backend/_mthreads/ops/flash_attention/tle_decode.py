# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""A one-tile paged decode expressed with public TLE shared-memory MMA."""
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.decode_tile import (
    launch_decode_exceptions,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import launch


@triton.jit
def tle_decode_tile(
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
    BM: tl.constexpr,
):
    qs = tle.gpu.alloc((BM, 256), OUT.dtype.element_ty)
    ks = tle.gpu.alloc((256, 32), OUT.dtype.element_ty)
    vs = tle.gpu.alloc((32, 256), OUT.dtype.element_ty)
    ps = tle.gpu.alloc((BM, 32), OUT.dtype.element_ty)
    batch, kh = tl.program_id(0), tl.program_id(1)
    groups: tl.constexpr = HQ // HK
    flag_heads = kh * groups + tl.arange(0, groups)
    tl.store(FLAGS + batch * HQ + flag_heads, 0)
    begin = tl.load(CUQ + batch)
    if begin == tl.load(CUQ + batch + 1):
        return
    length = tl.load(USEDK + batch)
    m = tl.arange(0, BM)
    n = tl.arange(0, 32)
    d = tl.arange(0, 256)
    heads = kh * groups + m
    query = tl.load(
        Q + ((begin.to(tl.int64) * HQ + heads)[:, None] * 256) + d[None, :],
        m[:, None] < groups,
        0,
    )
    pages = tl.load(TABLE + batch * TS + n // PAGE, n < length, 0).to(tl.int64)
    key = tl.load(
        K + (pages * KP + n % PAGE * KS + kh * KH)[None, :] + d[:, None],
        n[None, :] < length,
        0,
    )
    value = tl.load(
        V + (pages * VP + n % PAGE * VS + kh * VH)[:, None] + d[None, :],
        n[:, None] < length,
        0,
    )
    tl.store(tle.gpu.local_ptr(qs), query)
    tl.store(tle.gpu.local_ptr(ks), key)
    tl.store(tle.gpu.local_ptr(vs), value)
    tl.debug_barrier()
    scores = tle.gpu.wgmma(qs, ks)
    scores = tle.gpu.wgmma_wait(0, scores)
    scores *= SCALE * 1.4426950408889634
    scores = tl.where(
        (m[:, None] < groups) & (n[None, :] < length), scores, -float("inf")
    )
    maximum = tl.max(scores, 1)
    bad_scores = (length > 0) & ~(tl.abs(maximum) <= 3.4028234663852886e38)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    p = tl.exp2(scores - safe[:, None])
    den = tl.sum(p, 1)
    tl.store(tle.gpu.local_ptr(ps), p.to(OUT.dtype.element_ty))
    tl.debug_barrier()
    acc = tle.gpu.wgmma(ps, vs)
    acc = tle.gpu.wgmma_wait(0, acc)
    bad_values = ~(tl.max(tl.abs(acc), 1) <= 3.4028234663852886e38)
    tl.store(FLAGS + batch * HQ + heads, bad_scores | bad_values, m < groups)
    output = acc / tl.where(den > 0, den, 1.0)[:, None]
    lse = tl.where(den > 0, (safe + tl.log2(den)) * 0.6931471805599453, float("inf"))
    tl.store(
        OUT + ((begin.to(tl.int64) * HQ + heads)[:, None] * 256) + d[None, :],
        output,
        m[:, None] < groups,
    )
    tl.store(LSE + heads * TQ + begin, lse, m < groups)


def launch_tle_decode_tile(q, k, v, out, lse, cuq, usedk, table, scale, bm=32, warps=4):
    flags = torch.empty(
        (table.shape[0] * q.shape[1],), device=q.device, dtype=torch.int32
    )
    launch(
        tle_decode_tile,
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
        bm,
        num_warps=warps,
        num_stages=1,
        enable_backend_opt=True,
    )
    return launch_decode_exceptions(q, k, v, out, lse, flags, cuq, usedk, table, scale)

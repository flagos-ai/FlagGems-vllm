# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Vector attention for a paged decode request with at most one KV token."""
import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.decode_tile import (
    launch_decode_exceptions,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.direct import (
    _BACKEND_OPT,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
    current_config,
    launch,
)


def supports_single_kv(q, k, table, maxq, maxk, window, softcap, alibi):
    return (
        table is not None
        and table.shape[1] > 0
        and q.dtype == torch.bfloat16
        and q.shape[2] == 256
        and q.shape[1] // k.shape[-2] in (4, 8)
        and maxq == 1
        and maxk <= 1
        and window == (-1, -1)
        and not softcap
        and alibi is None
        and q.is_contiguous()
    )


@triton.jit
def single_kv_kernel(
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
    KH: tl.constexpr,
    VP: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    BD: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    qs = tl.load(CUQ + batch)
    qe = tl.load(CUQ + batch + 1)
    if qs == qe:
        return
    length = tl.load(USEDK + batch)
    page = tl.load(TABLE + batch * TS, length > 0, 0).to(tl.int64)
    kh = head // (HQ // HK)
    d = tl.arange(0, BD)
    qo = (qs.to(tl.int64) * HQ + head) * D + d
    q = tl.load(Q + qo, d < D, 0).to(tl.float32)
    k = tl.load(K + page * KP + kh * KH + d, (length > 0) & (d < D), 0).to(tl.float32)
    v = tl.load(V + page * VP + kh * VH + d, (length > 0) & (d < D), 0).to(tl.float32)
    dot = tl.sum(q * k, 0)
    if (length == 0) | (tl.abs(dot) <= 3.4028234663852886e38):
        score = dot * (SCALE * 1.4426950408889634)
        score = tl.where(length > 0, score, -float("inf"))
        safe = tl.where(score == -float("inf"), 0.0, score)
        p = tl.exp2(score - safe)
        out = p * v / tl.where(p > 0, p, 1.0)
        ls = tl.where(p > 0, (safe + tl.log2(p)) * 0.6931471805599453, float("inf"))
        tl.store(OUT + qo, out, d < D)
        tl.store(LSE + head * TQ + qs, ls)
    else:
        # MUSA matrix instructions saturate infinite dot products. Preserve
        # the existing decode arithmetic for nonfinite or overflowing inputs.
        rows = tl.arange(0, 16)
        cols = tl.arange(0, 32)
        qm = tl.load(Q + qo[None, :], (rows[:, None] == 0) & (d[None, :] < D), 0)
        km = tl.load(
            K + page * KP + kh * KH + d[:, None],
            (d[:, None] < D) & (cols[None, :] == 0),
            0,
        )
        scores = tl.dot(qm, km) * (SCALE * 1.4426950408889634)
        scores = tl.where(
            (rows[:, None] == 0) & (cols[None, :] == 0), scores, -float("inf")
        )
        maximum = tl.max(scores, 1)
        safe_m = tl.where(maximum == -float("inf"), 0.0, maximum)
        p_m = tl.exp2(scores - safe_m[:, None])
        den = tl.sum(p_m, 1)
        vm = tl.load(
            V + page * VP + kh * VH + d[None, :],
            (cols[:, None] == 0) & (d[None, :] < D),
            0,
        )
        acc = tl.dot(p_m.to(vm.dtype), vm)
        out_m = acc / tl.where(den > 0, den, 1.0)[:, None]
        ls_m = tl.where(
            den > 0, (safe_m + tl.log2(den)) * 0.6931471805599453, float("inf")
        )
        tl.store(
            OUT + qo[None, :] + rows[:, None] * 0,
            out_m,
            (rows[:, None] == 0) & (d[None, :] < D),
        )
        tl.store(LSE + head * TQ + qs + rows, ls_m, rows == 0)


def launch_single_kv(q, k, v, out, lse, cuq, usedk, table, scale):
    config = current_config()
    if config is not None and config.plan.block_m:
        return launch_vector_single_kv(
            q,
            k,
            v,
            out,
            lse,
            cuq,
            usedk,
            table,
            scale,
            config.plan.block_m,
            config.plan.num_warps,
        )
    return launch(
        single_kv_kernel,
        (table.shape[0], q.shape[1]),
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
        k.stride(0),
        k.stride(2),
        v.stride(0),
        v.stride(2),
        table.stride(0),
        scale,
        256,
        num_warps=4,
        num_stages=1,
        **{_BACKEND_OPT: False},
    )


@triton.jit
def vector_single(
    Q,
    K,
    V,
    OUT,
    LSE,
    FLAGS,
    CUQ,
    USEDK,
    TABLE,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    TQ: tl.constexpr,
    KP: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    ROWS: tl.constexpr,
):
    index = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    batch, head = index // HQ, index % HQ
    valid = index < BATCH * HQ
    begin = tl.load(CUQ + batch, valid, 0)
    end = tl.load(CUQ + batch + 1, valid, 0)
    active = valid & (begin < end)
    length = tl.load(USEDK + batch, active, 0)
    page = tl.load(TABLE + batch * TS, active & (length > 0), 0).to(tl.int64)
    kh = head // (HQ // HK)
    d = tl.arange(0, 256)
    qo = (begin.to(tl.int64) * HQ + head) * D
    q = tl.load(Q + qo[:, None] + d[None, :], active[:, None], 0).to(tl.float32)
    k = tl.load(
        K + (page * KP + kh * KH)[:, None] + d[None, :],
        (active & (length > 0))[:, None],
        0,
    ).to(tl.float32)
    v = tl.load(
        V + (page * VP + kh * VH)[:, None] + d[None, :],
        (active & (length > 0))[:, None],
        0,
    ).to(tl.float32)
    dot = tl.sum(q * k, 1)
    finite = (length == 0) | (tl.abs(dot) <= 3.4028234663852886e38)
    tl.store(FLAGS + index, active & ~finite, valid)
    score = dot * (SCALE * 1.4426950408889634)
    score = tl.where(length > 0, score, -float("inf"))
    safe = tl.where(score == -float("inf"), 0.0, score)
    p = tl.exp2(score - safe)
    output = p[:, None] * v / tl.where(p > 0, p, 1.0)[:, None]
    lse = tl.where(p > 0, (safe + tl.log2(p)) * 0.6931471805599453, float("inf"))
    tl.store(OUT + qo[:, None] + d[None, :], output, active[:, None])
    tl.store(LSE + head * TQ + begin, lse, active)


def launch_vector_single_kv(
    q, k, v, out, lse, cuq, usedk, table, scale, rows=8, warps=4
):
    flags = torch.empty(
        (table.shape[0] * q.shape[1],), dtype=torch.int32, device=q.device
    )
    params = (
        q.shape[1],
        k.shape[2],
        q.shape[2],
        q.shape[0],
        k.stride(0),
        k.stride(2),
        v.stride(0),
        v.stride(2),
        table.stride(0),
        scale,
    )
    launch(
        vector_single,
        (triton.cdiv(flags.numel(), rows),),
        q,
        k,
        v,
        out,
        lse,
        flags,
        cuq,
        usedk,
        table,
        table.shape[0],
        *params,
        rows,
        num_warps=warps,
        num_stages=1,
        enable_backend_opt=True,
    )
    return launch_decode_exceptions(q, k, v, out, lse, flags, cuq, usedk, table, scale)

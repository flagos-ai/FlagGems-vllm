# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""FA3-inspired PV/softmax scheduling; original TLE, no partial wait."""
from pathlib import Path

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language import core
from triton.tools.tensor_descriptor import TensorDescriptor

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
    current_config,
    launch,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.kv_cache import (
    cache_extent,
    gather_cache_kernel,
)

_MATH_LIB = str(Path(__file__).with_name("math_packed4.bc"))


@core.extern
def exp24_lo(lo, hi, _semantic=None):
    return core.extern_elementwise(
        "fa_math4",
        _MATH_LIB,
        [lo, hi],
        {(tl.uint64, tl.uint64): ("fa_exp2_packed4_lo", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def exp24_hi(lo, hi, _semantic=None):
    return core.extern_elementwise(
        "fa_math4",
        _MATH_LIB,
        [lo, hi],
        {(tl.uint64, tl.uint64): ("fa_exp2_packed4_hi", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def mul4_lo(lo, hi, alpha, _semantic=None):
    return core.extern_elementwise(
        "fa_math4",
        _MATH_LIB,
        [lo, hi, alpha],
        {(tl.uint64, tl.uint64, tl.float32): ("fa_mul_packed4_lo", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def mul4_hi(lo, hi, alpha, _semantic=None):
    return core.extern_elementwise(
        "fa_math4",
        _MATH_LIB,
        [lo, hi, alpha],
        {(tl.uint64, tl.uint64, tl.float32): ("fa_mul_packed4_hi", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def pack_quads(x):
    M: tl.constexpr = x.shape[0]
    N: tl.constexpr = x.shape[1]
    x = tl.reshape(
        tl.permute(tl.reshape(x, (M, N // 8, 8)), (0, 2, 1)), (M, N // 4, 2, 2)
    )
    even, odd = tl.split(x)
    x0, x2 = tl.split(even)
    x1, x3 = tl.split(odd)
    lo = x0.to(tl.uint32, bitcast=True).to(tl.uint64) | (
        x1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    )
    hi = x2.to(tl.uint32, bitcast=True).to(tl.uint64) | (
        x3.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    )
    return lo, hi


@triton.jit
def unpack_quads(lo, hi, M: tl.constexpr, N: tl.constexpr):
    x0 = lo.to(tl.uint32).to(tl.float32, bitcast=True)
    x1 = (lo >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    x2 = hi.to(tl.uint32).to(tl.float32, bitcast=True)
    x3 = (hi >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    packed = tl.reshape(tl.join(tl.join(x0, x2), tl.join(x1, x3)), (M, 8, N // 8))
    return tl.reshape(tl.permute(packed, (0, 2, 1)), (M, N))


@triton.jit
def exp2_quads(x):
    lo, hi = pack_quads(x)
    result_lo = exp24_lo(lo, hi)
    result_hi = exp24_hi(lo, hi)
    return unpack_quads(result_lo, result_hi, x.shape[0], x.shape[1])


@triton.jit
def rescale_o(acc, alpha, SIMD_RESCALE: tl.constexpr):
    if SIMD_RESCALE:
        lo, hi = pack_quads(acc)
        result_lo = mul4_lo(lo, hi, alpha[:, None])
        result_hi = mul4_hi(lo, hi, alpha[:, None])
        return unpack_quads(result_lo, result_hi, acc.shape[0], acc.shape[1])
    return acc * alpha[:, None]


@core.extern
def direct_exp2(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.float32,): ("llvm.exp2.f32", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def exp2_select(
    x, FAST_EXP2: tl.constexpr, SIMD_EXP2: tl.constexpr, SIMD_RESCALE: tl.constexpr
):
    if SIMD_EXP2 and len(x.shape) == 2:
        return exp2_quads(x)
    elif FAST_EXP2:
        return direct_exp2(x)
    else:
        return tl.exp2(x)


@triton.jit
def update_softmax(
    score,
    maximum,
    denominator,
    rows,
    row_base,
    qlen,
    klen,
    i,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    RAW_MAX: tl.constexpr,
    FAST_EXP2: tl.constexpr,
    SIMD_EXP2: tl.constexpr,
    SIMD_RESCALE: tl.constexpr,
):
    if not RAW_MAX:
        score *= SCALE * 1.4426950408889634
    full = (i + 1) * BN <= klen
    if CAUSAL:
        full &= (i + 1) * BN <= row_base + klen - qlen + 1
    if not full:
        cols = i * BN + tl.arange(0, BN)
        valid = (rows[:, None] < qlen) & (cols[None, :] < klen)
        if CAUSAL:
            valid &= cols[None, :] <= rows[:, None] + klen - qlen
        score = tl.where(valid, score, -float("inf"))
    next_maximum = tl.maximum(maximum, tl.max(score, 1))
    safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
    if RAW_MAX:
        alpha = exp2_select(
            (maximum - safe_maximum) * (SCALE * 1.4426950408889634),
            FAST_EXP2,
            SIMD_EXP2,
            SIMD_RESCALE,
        )
        p = exp2_select(
            score * (SCALE * 1.4426950408889634)
            - (safe_maximum * (SCALE * 1.4426950408889634))[:, None],
            FAST_EXP2,
            SIMD_EXP2,
            SIMD_RESCALE,
        )
    else:
        alpha = exp2_select(maximum - safe_maximum, FAST_EXP2, SIMD_EXP2, SIMD_RESCALE)
        p = exp2_select(
            score - safe_maximum[:, None], FAST_EXP2, SIMD_EXP2, SIMD_RESCALE
        )
    M: tl.constexpr = p.shape[0]
    partial = tl.sum(tl.reshape(p, (M, BN // 64, 8, 8)), 2)
    partial = tl.reshape(partial, (M, BN // 8))
    denominator = denominator * alpha[:, None] + partial
    return p, alpha, next_maximum, denominator


@triton.jit
def attention(
    Q,
    K,
    V,
    OUT,
    LSE,
    CUQ,
    USEDK,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    NK: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    RAW_MAX: tl.constexpr,
    PREFETCH_V: tl.constexpr,
    R: tl.constexpr,
    OVERLAP_PV: tl.constexpr,
    PERMUTE: tl.constexpr,
    SKIP_RESCALE: tl.constexpr,
    FAST_EXP2: tl.constexpr,
    SIMD_EXP2: tl.constexpr,
    SIMD_RESCALE: tl.constexpr,
    PREFETCH_K: tl.constexpr,
):
    qs = tle.gpu.alloc((BM, 256), OUT.dtype.element_ty)
    qb = tle.gpu.alloc_barrier(expect_bytes=BM * 256 * 2)
    kb = tle.gpu.alloc_barriers(R, expect_bytes=256 * BN * 2)
    vb = tle.gpu.alloc_barriers(R, expect_bytes=BN * 256 * 2)
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    klen = tl.load(USEDK + batch)
    row_base = tile * BM
    if row_base >= qlen:
        return
    rows = row_base + tl.arange(0, BM)
    dims = tl.arange(0, 256)
    kh = batch * HK + head // (HQ // HK)
    high = klen
    if CAUSAL:
        high = tl.minimum(high, tl.minimum(row_base + BM, qlen) + klen - qlen)
    high = tl.maximum(high, 0)
    tle.gpu.copy(Q, qs, (BM, 256), (qstart + row_base, head * 256), barrier=qb)
    tle.gpu.barrier_wait(qb, phaseIdx=0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM, BN // 8), tl.float32)
    acc = tl.zeros((BM, 256), tl.float32)
    ps = tle.gpu.alloc((BM, BN), OUT.dtype.element_ty)
    ks = tle.gpu.alloc((256, BN), OUT.dtype.element_ty)
    vs = tle.gpu.alloc((BN, 256), OUT.dtype.element_ty)
    iterations = tl.cdiv(high, BN)
    if iterations > 0:
        tle.gpu.copy(K, ks, (256, BN), (kh * 256, 0), barrier=kb[0])
        tle.gpu.barrier_wait(kb[0], phaseIdx=0)
        score = tle.gpu.wgmma(qs, ks)
        score = tle.gpu.wgmma_wait(0, score)
        if PREFETCH_K and iterations > 1:
            tle.gpu.copy(K, ks, (256, BN), (kh * 256, BN), barrier=kb[1 % R])
        if PREFETCH_V:
            tle.gpu.copy(V, vs, (BN, 256), (kh * NK, 0), barrier=vb[0])
        p, alpha, maximum, denominator = update_softmax(
            score,
            maximum,
            denominator,
            rows,
            row_base,
            qlen,
            klen,
            0,
            BN,
            SCALE,
            CAUSAL,
            RAW_MAX,
            FAST_EXP2,
            SIMD_EXP2,
            SIMD_RESCALE,
        )
        p_store = p.to(OUT.dtype.element_ty)
        if PERMUTE:
            p_store = tl.reshape(
                tl.permute(tl.reshape(p_store, (BM, BN // 8, 8)), (0, 2, 1)), (BM, BN)
            )
        tl.store(tle.gpu.local_ptr(ps), p_store)
        for i in range(1, iterations):
            if not PREFETCH_K:
                tle.gpu.copy(K, ks, (256, BN), (kh * 256, i * BN), barrier=kb[i % R])
            if not PREFETCH_V:
                tle.gpu.copy(
                    V,
                    vs,
                    (BN, 256),
                    (kh * NK + (i - 1) * BN, 0),
                    barrier=vb[(i - 1) % R],
                )
            tle.gpu.barrier_wait(kb[i % R], phaseIdx=(i // R) & 1)
            score = tle.gpu.wgmma(qs, ks)
            # Original TLE cannot partially wait: finish QK before issuing PV.
            score = tle.gpu.wgmma_wait(0, score)
            if PREFETCH_K and i + 1 < iterations:
                tle.gpu.copy(
                    K, ks, (256, BN), (kh * 256, (i + 1) * BN), barrier=kb[(i + 1) % R]
                )
            tle.gpu.barrier_wait(vb[(i - 1) % R], phaseIdx=((i - 1) // R) & 1)
            pending_acc = tle.gpu.wgmma(ps, vs, acc)
            if not OVERLAP_PV:
                acc = tle.gpu.wgmma_wait(0, pending_acc)
            # No acc or shared P access while the previous PV is outstanding.
            p, alpha, maximum, denominator = update_softmax(
                score,
                maximum,
                denominator,
                rows,
                row_base,
                qlen,
                klen,
                i,
                BN,
                SCALE,
                CAUSAL,
                RAW_MAX,
                FAST_EXP2,
                SIMD_EXP2,
                SIMD_RESCALE,
            )
            if OVERLAP_PV:
                acc = tle.gpu.wgmma_wait(0, pending_acc)
            if PREFETCH_V:
                tle.gpu.copy(V, vs, (BN, 256), (kh * NK + i * BN, 0), barrier=vb[i % R])
            if SKIP_RESCALE:
                if tl.sum((alpha != 1.0).to(tl.int32), 0) > 0:
                    acc = rescale_o(acc, alpha, SIMD_RESCALE)
            else:
                acc = rescale_o(acc, alpha, SIMD_RESCALE)
            p_store = p.to(OUT.dtype.element_ty)
            if PERMUTE:
                p_store = tl.reshape(
                    tl.permute(tl.reshape(p_store, (BM, BN // 8, 8)), (0, 2, 1)),
                    (BM, BN),
                )
            tl.store(tle.gpu.local_ptr(ps), p_store)
        last = iterations - 1
        if not PREFETCH_V:
            tle.gpu.copy(
                V, vs, (BN, 256), (kh * NK + last * BN, 0), barrier=vb[last % R]
            )
        tle.gpu.barrier_wait(vb[last % R], phaseIdx=(last // R) & 1)
        acc = tle.gpu.wgmma(ps, vs, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
    denominator = tl.sum(denominator, 1)
    if RAW_MAX:
        maximum *= SCALE * 1.4426950408889634
    lse = tl.where(
        denominator > 0,
        (maximum + tl.log2(denominator)) * 0.6931471805599453,
        float("inf"),
    )
    tl.store(LSE + head * TQ + qstart + rows, lse, rows < qlen)
    output = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        OUT + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256 + dims[None, :],
        output,
        rows[:, None] < qlen,
    )


def launch_prefill(
    q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal, maxk=None
):
    """Descriptor prefill; paged gather is part of every call."""
    config = current_config()
    if (
        q.dtype == torch.bfloat16
        and scale > 0
        and (config is None or config.route != "tle")
    ):
        from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.bounded_prefill import (
            launch_bounded_prefill,
        )

        return launch_bounded_prefill(
            q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal, maxk
        )
    bm, bn = 128, 64
    batch, hk, dim = table.shape[0], k.shape[2], q.shape[2]
    nk = cache_extent(k, table, maxk, bn)
    kc = torch.empty((batch * hk * dim, nk), device=q.device, dtype=q.dtype)
    vc = torch.empty((batch * hk * nk, dim), device=q.device, dtype=q.dtype)
    qd = TensorDescriptor(
        q, (q.shape[0], q.shape[1] * dim), (q.shape[1] * dim, 1), (bm, dim)
    )
    kd = TensorDescriptor(kc, (batch * hk * dim, nk), (nk, 1), (dim, bn))
    vd = TensorDescriptor(vc, (batch * hk * nk, dim), (dim, 1), (bn, dim))
    gn = config.gather_tile if config is not None and config.gather_tile else 32
    gw = config.gather_warps if config is not None and config.gather_warps else 4
    launch(
        gather_cache_kernel,
        ((triton.cdiv(nk, gn), batch * hk, 4)),
        k,
        v,
        kc,
        vc,
        table,
        usedk,
        hk,
        dim,
        k.shape[1],
        *k.stride()[:3],
        *v.stride()[:3],
        table.stride(0),
        nk,
        True,
        bn,
        BN=gn,
        num_warps=gw,
        num_stages=1,
    )
    return launch(
        attention,
        ((triton.cdiv(maxq, bm), batch, q.shape[1])),
        qd,
        kd,
        vd,
        out,
        lse,
        cuq,
        usedk,
        q.shape[0],
        q.shape[1],
        hk,
        nk,
        scale,
        causal,
        bm,
        bn,
        RAW_MAX=scale > 0,
        PREFETCH_V=False,
        R=4,
        OVERLAP_PV=True,
        PERMUTE=True,
        SKIP_RESCALE=False,
        FAST_EXP2=True,
        SIMD_EXP2=True,
        SIMD_RESCALE=False,
        PREFETCH_K=True,
        num_warps=16,
        num_stages=1,
        enable_backend_opt=True,
        extern_libs={"fa_math4": _MATH_LIB},
    )

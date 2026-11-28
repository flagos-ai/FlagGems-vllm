# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""BF16 prefill with GPU-validated fixed softmax reference and online fallback.

Each invocation gathers fresh paged KV and its maximum squared K norm. Cauchy
bounds every finite logit by ||Q|| * max ||K|| * abs(scale). A CTA with bound <=30
uses a fixed reference ceil(bound * log2(e)) + 2. Thus its BF16 probabilities stay
between roughly 2**-90 and 1 without per-tile rescaling. Other CTAs retain stable
online softmax. P uses two buffers: the previous PV cannot race the next P store.
The fast loop issues QK and previous PV before wait-all; no partial wait is used.
"""
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
    gather_norm_kernel,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.ragged import (
    compact_tile_coords,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.tle_prefill import (
    _MATH_LIB,
    exp2_select,
    pack_quads,
    rescale_o,
    unpack_quads,
)

_AFFINE_LIB = str(Path(__file__).with_name("math_affine4.bc"))


@core.extern
def affine_exp24_lo(lo, hi, scale, bias, _semantic=None):
    return core.extern_elementwise(
        "fa_affine4",
        _AFFINE_LIB,
        [lo, hi, scale, bias],
        {
            (tl.uint64, tl.uint64, tl.float32, tl.float32): (
                "fa_affine_exp2_packed4_lo",
                tl.uint64,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def affine_exp24_hi(lo, hi, scale, bias, _semantic=None):
    return core.extern_elementwise(
        "fa_affine4",
        _AFFINE_LIB,
        [lo, hi, scale, bias],
        {
            (tl.uint64, tl.uint64, tl.float32, tl.float32): (
                "fa_affine_exp2_packed4_hi",
                tl.uint64,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def affine_exp2_quads(score, scale, bias):
    (lo, hi) = pack_quads(score)
    result_lo = affine_exp24_lo(lo, hi, scale, bias)
    result_hi = affine_exp24_hi(lo, hi, scale, bias)
    return unpack_quads(result_lo, result_hi, score.shape[0], score.shape[1])


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
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    bounded,
):
    full = (i + 1) * 64 <= klen
    if CAUSAL:
        full &= (i + 1) * 64 <= row_base + klen - qlen + 1
    if not full:
        cols = i * 64 + tl.arange(0, 64)
        valid = (rows[:, None] < qlen) & (cols[None, :] < klen)
        if CAUSAL:
            valid &= cols[None, :] <= rows[:, None] + klen - qlen
        score = tl.where(valid, score, -float("inf"))
    if bounded:
        next_maximum = maximum
    else:
        next_maximum = tl.maximum(maximum, tl.max(score, 1))
    safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
    alpha = exp2_select(
        (maximum - safe_maximum) * (SCALE * 1.4426950408889634), True, True, False
    )
    if bounded:
        p = affine_exp2_quads(
            score,
            SCALE * 1.4426950408889634,
            -(safe_maximum * (SCALE * 1.4426950408889634))[:, None],
        )
    else:
        p = exp2_select(
            score * (SCALE * 1.4426950408889634)
            - (safe_maximum * (SCALE * 1.4426950408889634))[:, None],
            True,
            True,
            False,
        )
    M: tl.constexpr = p.shape[0]
    partial = tl.sum(tl.reshape(p, (M, 1, 8, 8)), 2)
    partial = tl.reshape(partial, (M, 8))
    if bounded:
        denominator += partial
    else:
        denominator = denominator * alpha[:, None] + partial
    return (p, alpha, next_maximum, denominator)


@triton.jit
def attention_body(
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
    PACK: tl.constexpr,
    ONLY_BOUNDED: tl.constexpr,
    SHIFT,
    tile,
    batch,
    head,
):
    qs = tle.gpu.alloc((128, 256), OUT.dtype.element_ty)
    if PACK == 1:
        qb = tle.gpu.alloc_barrier(expect_bytes=65536)
    kb = tle.gpu.alloc_barriers(4, expect_bytes=32768)
    vb = tle.gpu.alloc_barriers(4, expect_bytes=32768)
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    klen = tl.load(USEDK + batch)
    row_base = tile * 128 // PACK
    if row_base >= qlen:
        return
    packed_rows = tile * 128 + tl.arange(0, 128)
    rows = packed_rows // PACK
    qheads = head * PACK + packed_rows % PACK
    dims = tl.arange(0, 256)
    kh = batch * HK + head * PACK // (HQ // HK)
    high = klen
    if CAUSAL:
        high = tl.minimum(high, tl.minimum(row_base + 128 // PACK, qlen) + klen - qlen)
    high = tl.maximum(high, 0)
    if PACK > 1:
        # Packing ragged query rows with GQA heads trims partial query tiles.
        # Ordinary loads also handle multiple KV heads without descriptor reshape.
        qvalues = tl.load(
            Q
            + ((qstart + rows[:, None]).to(tl.int64) * HQ + qheads[:, None]) * 256
            + dims[None, :],
            rows[:, None] < qlen,
            0,
        )
        tl.store(tle.gpu.local_ptr(qs), qvalues)
    else:
        tle.gpu.copy(Q, qs, (128, 256), (qstart + row_base, head * 256), barrier=qb)
        tle.gpu.barrier_wait(qb, phaseIdx=0)
    shift = SHIFT
    bounded: tl.constexpr = ONLY_BOUNDED
    maximum = tl.full((128,), -float("inf"), tl.float32)
    maximum = tl.where(bounded, shift / (SCALE * 1.4426950408889634), maximum)
    denominator = tl.zeros((128, 8), tl.float32)
    acc = tl.zeros((128, 256), tl.float32)
    ps = tle.gpu.alloc((2, 128, 64), OUT.dtype.element_ty)
    ks = tle.gpu.alloc((256, 64), OUT.dtype.element_ty)
    vs = tle.gpu.alloc((64, 256), OUT.dtype.element_ty)
    iterations = tl.cdiv(high, 64)
    if iterations > 0:
        tle.gpu.copy(K, ks, (256, 64), (kh * 256, 0), barrier=kb[0])
        tle.gpu.barrier_wait(kb[0], phaseIdx=0)
        score = tle.gpu.wgmma(qs, ks)
        score = tle.gpu.wgmma_wait(0, score)
        if iterations > 1:
            tle.gpu.copy(K, ks, (256, 64), (kh * 256, 64), barrier=kb[1])
        (p, alpha, maximum, denominator) = update_softmax(
            score,
            maximum,
            denominator,
            rows,
            row_base,
            qlen,
            klen,
            0,
            SCALE,
            CAUSAL,
            bounded,
        )
        p_store = p.to(OUT.dtype.element_ty)
        p_store = tl.reshape(
            tl.permute(tl.reshape(p_store, (128, 8, 8)), (0, 2, 1)), (128, 64)
        )
        tl.store(tle.gpu.local_ptr(ps.slot(0)), p_store)
        for i in range(1, iterations):
            tle.gpu.copy(
                V, vs, (64, 256), (kh * NK + (i - 1) * 64, 0), barrier=vb[(i - 1) % 4]
            )
            tle.gpu.barrier_wait(kb[i % 4], phaseIdx=i // 4 & 1)
            score = tle.gpu.wgmma(qs, ks)
            if ONLY_BOUNDED:
                tle.gpu.barrier_wait(vb[(i - 1) % 4], phaseIdx=(i - 1) // 4 & 1)
                pending_acc = tle.gpu.wgmma(ps.slot((i - 1) % 2), vs, acc)
            score = tle.gpu.wgmma_wait(0, score)
            if i + 1 < iterations:
                tle.gpu.copy(
                    K, ks, (256, 64), (kh * 256, (i + 1) * 64), barrier=kb[(i + 1) % 4]
                )
            if not ONLY_BOUNDED:
                tle.gpu.barrier_wait(vb[(i - 1) % 4], phaseIdx=(i - 1) // 4 & 1)
                pending_acc = tle.gpu.wgmma(ps.slot((i - 1) % 2), vs, acc)
            (p, alpha, maximum, denominator) = update_softmax(
                score,
                maximum,
                denominator,
                rows,
                row_base,
                qlen,
                klen,
                i,
                SCALE,
                CAUSAL,
                bounded,
            )
            acc = tle.gpu.wgmma_wait(0, pending_acc)
            if bounded:
                pass
            else:
                acc = rescale_o(acc, alpha, False)
            p_store = p.to(OUT.dtype.element_ty)
            p_store = tl.reshape(
                tl.permute(tl.reshape(p_store, (128, 8, 8)), (0, 2, 1)), (128, 64)
            )
            tl.store(tle.gpu.local_ptr(ps.slot(i % 2)), p_store)
        last = iterations - 1
        tle.gpu.copy(V, vs, (64, 256), (kh * NK + last * 64, 0), barrier=vb[last % 4])
        tle.gpu.barrier_wait(vb[last % 4], phaseIdx=last // 4 & 1)
        acc = tle.gpu.wgmma(ps.slot(last % 2), vs, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
    denominator = tl.sum(denominator, 1)
    maximum *= SCALE * 1.4426950408889634
    lse = tl.where(
        denominator > 0,
        (maximum + tl.log2(denominator)) * 0.6931471805599453,
        float("inf"),
    )
    tl.store(LSE + qheads * TQ + qstart + rows, lse, rows < qlen)
    output = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        OUT
        + ((qstart + rows[:, None]).to(tl.int64) * HQ + qheads[:, None]) * 256
        + dims[None, :],
        output,
        rows[:, None] < qlen,
    )


@triton.jit
def attention(
    Q,
    QRAW,
    NORM,
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
    BATCH: tl.constexpr,
    COMPACT: tl.constexpr,
    PACK: tl.constexpr,
    GN: tl.constexpr,
):
    (tile, batch, head) = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    if COMPACT:
        (batch, tile, valid) = compact_tile_coords(
            tile, CUQ, BATCH, max(32, triton.next_power_of_2(BATCH)), 128, PACK
        )
        if not valid:
            return
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    row_base = tile * 128 // PACK
    if row_base >= qlen:
        return
    packed_rows = tile * 128 + tl.arange(0, 128)
    rows = packed_rows // PACK
    qheads = head * PACK + packed_rows % PACK
    dims = tl.arange(0, 256)
    kh = batch * HK + head * PACK // (HQ // HK)
    eligible = False
    shift = 0.0
    qvalues = tl.load(
        QRAW
        + ((qstart + rows[:, None]).to(tl.int64) * HQ + qheads[:, None]) * 256
        + dims[None, :],
        rows[:, None] < qlen,
        0,
    ).to(tl.float32)
    qnorm2 = tl.max(tl.sum(qvalues * qvalues, 1), 0)
    ni = tl.arange(0, triton.next_power_of_2(triton.cdiv(NK, GN)))
    knorm2 = tl.max(
        tl.load(NORM + kh * tl.cdiv(NK, GN) + ni, ni < tl.cdiv(NK, GN), 0), 0
    )
    bound = tl.sqrt(qnorm2 * knorm2) * tl.abs(SCALE)
    eligible = (bound >= 0.0) & (bound <= 30.0)
    shift = tl.ceil(bound * 1.4426950408889634) + 2.0
    if eligible:
        attention_body(
            Q,
            K,
            V,
            OUT,
            LSE,
            CUQ,
            USEDK,
            TQ,
            HQ,
            HK,
            NK,
            SCALE,
            CAUSAL,
            PACK,
            True,
            shift,
            tile,
            batch,
            head,
        )
    else:
        attention_body(
            Q,
            K,
            V,
            OUT,
            LSE,
            CUQ,
            USEDK,
            TQ,
            HQ,
            HK,
            NK,
            SCALE,
            CAUSAL,
            PACK,
            False,
            0.0,
            tile,
            batch,
            head,
        )


def launch_bounded_prefill(
    q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal, maxk=None
):
    """Positive-scale BF16 cached route; all preparation is replayed in Graph."""
    config = current_config()
    (bm, bn) = (128, 64)
    (batch, hk, dim) = (table.shape[0], k.shape[2], q.shape[2])
    nk = cache_extent(k, table, maxk, bn)
    # Small batches need more gather CTAs to fill the device.
    gn, gather_warps = (16, 8) if batch == 1 and nk <= 2048 else (32, 4)
    compact = batch > 1
    # Measured for mixed GQA8 prefill with a short paged cache. Long KV and
    # single-request prefill retain one head per CTA.
    pack = 8 if compact and q.shape[1] // hk == 8 and nk <= 8192 else 1
    if compact and q.shape[1] // hk == 4 and nk <= 2048:
        pack = 4
    if config is not None:
        gn = config.gather_tile or gn
        gather_warps = config.gather_warps or gather_warps
        pack = config.pack or pack
        if config.compact != -1:
            compact = bool(config.compact)
    kc = torch.empty((batch * hk * dim, nk), device=q.device, dtype=q.dtype)
    vc = torch.empty((batch * hk * nk, dim), device=q.device, dtype=q.dtype)
    norms = torch.empty(
        (batch * hk, triton.cdiv(nk, gn)), device=q.device, dtype=torch.float32
    )
    qd = q
    if pack == 1:
        qd = TensorDescriptor(
            q, (q.shape[0], q.shape[1] * dim), (q.shape[1] * dim, 1), (bm, dim)
        )
    kd = TensorDescriptor(kc, (batch * hk * dim, nk), (nk, 1), (dim, bn))
    vd = TensorDescriptor(vc, (batch * hk * nk, dim), (dim, 1), (bn, dim))
    launch(
        gather_norm_kernel,
        (triton.cdiv(nk, gn), batch * hk, 1),
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
        BD=256,
        NORM=norms,
        num_warps=gather_warps,
        num_stages=1,
    )
    tiles = (
        triton.cdiv(q.shape[0] * pack, bm) + batch - 1
        if compact
        else triton.cdiv(maxq * pack, bm)
    )
    return launch(
        attention,
        (tiles, 1 if compact else batch, q.shape[1] // pack),
        qd,
        q,
        norms,
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
        BATCH=batch,
        COMPACT=compact,
        PACK=pack,
        GN=gn,
        num_warps=16,
        num_stages=1,
        enable_backend_opt=True,
        extern_libs={"fa_math4": _MATH_LIB, "fa_affine4": _AFFINE_LIB},
    )

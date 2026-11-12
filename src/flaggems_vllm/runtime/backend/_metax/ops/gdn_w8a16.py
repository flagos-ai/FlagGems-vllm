# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX GDN: FP32 core, opt-in mixed-precision chunks, and fused decode.

Shared chunk indexing/inverse helpers serve all precision paths. Distinct
scan/output kernels preserve the validated arithmetic and launch configurations.
Temporary workspace is bounded independently of returned output/final state.
"""

import torch
import triton as tr
import triton.backends.metax.compiler as backend
import triton.language as tl

from flaggems_vllm import runtime

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:
    gluon = None


_WORKSPACE_LIMIT_BYTES = 256 * 1024**2
_SHORT_MAX_CHUNKS = 64
_FP32_MAX_CHUNKS = 32


def _install_gdn_compile_hook():
    translate = backend.metax.translate_llvmir_to_mcfatbin
    if getattr(translate, "_gdn_w8a16_hook", False):
        return

    def compile_gdn_chunk(src, mxcc, maca, options):
        # Capture the original callable so module reload cannot make it recursive.
        if backend.maca_get_kernel_name(src) in (
            "_gdn_chunk_scan",
            "_gdn_fp32_chunk_scan",
        ):
            options += " -mllvm -metaxgpu-disable-licm=true"
        return translate(src, mxcc, maca, options)

    compile_gdn_chunk._gdn_w8a16_hook = True
    backend.metax.translate_llvmir_to_mcfatbin = compile_gdn_chunk


_install_gdn_compile_hook()


@tr.jit
def _gdn_chunk_dotfp(a, b, PREC: tl.constexpr):
    return tl.dot(a, b, input_precision=PREC)


@tr.jit
def _round_locate(counts, ns, idx, N: tl.constexpr):
    # Compact round-robin ordering: all live requests contribute one chunk
    # per round. Prefix(round) = sum(min(request_chunks, round)). This keeps
    # windows balanced without a CPU read or padding to the longest request.
    low = tl.full((), 0, counts.dtype)
    high = tl.max(counts, 0)
    while low < high:
        mid = (low + high + 1) // 2
        prefix = tl.sum(tl.minimum(counts, mid), 0)
        low = tl.where(prefix <= idx, mid, low)
        high = tl.where(prefix <= idx, high, mid - 1)
    within = idx - tl.sum(tl.minimum(counts, low), 0)
    active = counts > low
    ranks = tl.cumsum(active.to(tl.int32))
    seq = tl.min(tl.where(active & (ranks > within), ns, N), 0)
    return low, seq


@tr.jit
def _gdn_chunk_locate(
    Cu, idx, N: tl.constexpr, NN: tl.constexpr, WIDE: tl.constexpr, CAP: tl.constexpr
):
    # Interleave requests when every request can occupy a slot; otherwise
    # contiguous packing avoids repeatedly loading many request states.
    if N > 1 and N <= CAP:
        ns = tl.arange(0, NN)
        starts = tl.load(Cu + ns, ns < N, 0).to(tl.int64 if WIDE else tl.int32)
        ends = tl.load(Cu + ns + 1, ns < N, 0).to(tl.int64 if WIDE else tl.int32)
        counts = tl.where(ns < N, tl.cdiv(ends - starts, 64), 0)
        ch, seq = _round_locate(counts, ns, idx, N)
        start = tl.load(Cu + seq).to(tl.int64 if WIDE else tl.int32)
        end = tl.load(Cu + seq + 1, seq < N, 0).to(tl.int64 if WIDE else tl.int32)
        return start + ch * 64, end, seq
    ns = tl.arange(0, NN)
    a = tl.load(Cu + ns, ns < N, 0).to(tl.int64 if WIDE else tl.int32)
    b = tl.load(Cu + ns + 1, ns < N, 0).to(tl.int64 if WIDE else tl.int32)
    counts = tl.where(ns < N, tl.cdiv(b - a, 64), 0)
    ends = tl.cumsum(counts)
    seq = tl.sum(((ends <= idx) & (ns < N)).to(tl.int32), 0)
    off = tl.sum(tl.where(ns < seq, counts, 0), 0)
    start = tl.load(Cu + seq).to(tl.int64 if WIDE else tl.int32)
    end = tl.load(Cu + seq + 1, seq < N, 0).to(tl.int64 if WIDE else tl.int32)
    return start + (idx - off) * 64, end, seq


@tr.jit
def _gdn_chunk_inverse16(d, PREC: tl.constexpr):
    x = tl.arange(0, 16)
    power = -tl.where(x[:, None] > x[None, :], d, 0.0)
    inv = tl.where(x[:, None] == x[None, :], 1.0, 0.0) + power
    for _ in tl.static_range(3):
        power = _gdn_chunk_dotfp(power, power, PREC)
        inv = inv + _gdn_chunk_dotfp(inv, power, PREC)
    return inv


@tr.jit
def _gdn_chunk_store16(A, c, h, i: tl.constexpr, j: tl.constexpr, b, HV: tl.constexpr):
    x = tl.arange(0, 16)
    tl.store(
        A + ((c * HV + h) * 64 + i * 16 + x[:, None]) * 64 + j * 16 + x[None, :], b
    )


@tr.jit
def _gdn_chunk_prep(
    BASE,
    CAP: tl.constexpr,
    Q,
    K,
    G,
    B,
    Cu,
    A,
    GC,
    QK,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    BK: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    start, end, seq = _gdn_chunk_locate(Cu, c + BASE, N, NN, WIDE, CAP)
    if seq >= N:
        return
    t = tl.arange(0, 64)
    x = tl.arange(0, 16)
    d = tl.arange(0, BK)
    token = start + t
    gc = tl.cumsum(tl.load(G + token * HV + h, token < end, 0))
    beta = tl.load(B + token * HV + h, token < end, 0)
    gx, gy = tl.split(tl.reshape(tl.trans(tl.reshape(gc, (4, 16))), (16, 2, 2)))
    g0, g2 = tl.split(gx)
    g1, g3 = tl.split(gy)
    bx, by = tl.split(tl.reshape(tl.trans(tl.reshape(beta, (4, 16))), (16, 2, 2)))
    b0, b2 = tl.split(bx)
    b1, b3 = tl.split(by)
    kh = h // (HV // H)
    k0 = tl.load(
        K + (start + x[:, None]) * KS + kh * DK + d[None, :],
        (start + x[:, None] < end) & (d[None, :] < DK),
        0,
    )
    k1 = tl.load(
        K + (start + 16 + x[:, None]) * KS + kh * DK + d[None, :],
        (start + 16 + x[:, None] < end) & (d[None, :] < DK),
        0,
    )
    k2 = tl.load(
        K + (start + 32 + x[:, None]) * KS + kh * DK + d[None, :],
        (start + 32 + x[:, None] < end) & (d[None, :] < DK),
        0,
    )
    k3 = tl.load(
        K + (start + 48 + x[:, None]) * KS + kh * DK + d[None, :],
        (start + 48 + x[:, None] < end) & (d[None, :] < DK),
        0,
    )
    a00 = tl.dot(k0, tl.trans(k0)) * tl.exp(g0[:, None] - g0[None, :]) * b0[:, None]
    a11 = tl.dot(k1, tl.trans(k1)) * tl.exp(g1[:, None] - g1[None, :]) * b1[:, None]
    a22 = tl.dot(k2, tl.trans(k2)) * tl.exp(g2[:, None] - g2[None, :]) * b2[:, None]
    a33 = tl.dot(k3, tl.trans(k3)) * tl.exp(g3[:, None] - g3[None, :]) * b3[:, None]
    a10 = tl.dot(k1, tl.trans(k0)) * tl.exp(g1[:, None] - g0[None, :]) * b1[:, None]
    a20 = tl.dot(k2, tl.trans(k0)) * tl.exp(g2[:, None] - g0[None, :]) * b2[:, None]
    a21 = tl.dot(k2, tl.trans(k1)) * tl.exp(g2[:, None] - g1[None, :]) * b2[:, None]
    a30 = tl.dot(k3, tl.trans(k0)) * tl.exp(g3[:, None] - g0[None, :]) * b3[:, None]
    a31 = tl.dot(k3, tl.trans(k1)) * tl.exp(g3[:, None] - g1[None, :]) * b3[:, None]
    a32 = tl.dot(k3, tl.trans(k2)) * tl.exp(g3[:, None] - g2[None, :]) * b3[:, None]
    i00 = _gdn_chunk_inverse16(a00, PREC)
    i11 = _gdn_chunk_inverse16(a11, PREC)
    i22 = _gdn_chunk_inverse16(a22, PREC)
    i33 = _gdn_chunk_inverse16(a33, PREC)
    i10 = -_gdn_chunk_dotfp(_gdn_chunk_dotfp(i11, a10, PREC), i00, PREC)
    i21 = -_gdn_chunk_dotfp(_gdn_chunk_dotfp(i22, a21, PREC), i11, PREC)
    i32 = -_gdn_chunk_dotfp(_gdn_chunk_dotfp(i33, a32, PREC), i22, PREC)
    i20 = -_gdn_chunk_dotfp(
        i22, _gdn_chunk_dotfp(a20, i00, PREC) + _gdn_chunk_dotfp(a21, i10, PREC), PREC
    )
    i31 = -_gdn_chunk_dotfp(
        i33, _gdn_chunk_dotfp(a31, i11, PREC) + _gdn_chunk_dotfp(a32, i21, PREC), PREC
    )
    i30 = -_gdn_chunk_dotfp(
        i33,
        _gdn_chunk_dotfp(a30, i00, PREC)
        + _gdn_chunk_dotfp(a31, i10, PREC)
        + _gdn_chunk_dotfp(a32, i20, PREC),
        PREC,
    )
    _gdn_chunk_store16(A, c, h, 0, 0, i00, HV)
    _gdn_chunk_store16(A, c, h, 1, 1, i11, HV)
    _gdn_chunk_store16(A, c, h, 2, 2, i22, HV)
    _gdn_chunk_store16(A, c, h, 3, 3, i33, HV)
    _gdn_chunk_store16(A, c, h, 1, 0, i10, HV)
    _gdn_chunk_store16(A, c, h, 2, 0, i20, HV)
    _gdn_chunk_store16(A, c, h, 2, 1, i21, HV)
    _gdn_chunk_store16(A, c, h, 3, 0, i30, HV)
    _gdn_chunk_store16(A, c, h, 3, 1, i31, HV)
    _gdn_chunk_store16(A, c, h, 3, 2, i32, HV)
    z = tl.full((16, 16), 0.0, tl.float32)
    _gdn_chunk_store16(A, c, h, 0, 1, z, HV)
    _gdn_chunk_store16(A, c, h, 0, 2, z, HV)
    _gdn_chunk_store16(A, c, h, 0, 3, z, HV)
    _gdn_chunk_store16(A, c, h, 1, 2, z, HV)
    _gdn_chunk_store16(A, c, h, 1, 3, z, HV)
    _gdn_chunk_store16(A, c, h, 2, 3, z, HV)
    tl.store(GC + (c * HV + h) * 64 + t, gc)


@tr.jit
def _gdn_chunk_wy(
    BASE,
    CAP: tl.constexpr,
    K,
    V,
    B,
    Cu,
    A,
    GC,
    W,
    U,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
    BW: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    start, end, seq = _gdn_chunk_locate(Cu, c + BASE, N, NN, WIDE, CAP)
    if seq >= N:
        return
    t = tl.arange(0, 64)
    d = tile * BW + tl.arange(0, BW)
    token = start + t
    inv = tl.load(A + ((c * HV + h) * 64 + t[:, None]) * 64 + t[None, :])
    beta = tl.load(B + token * HV + h, token < end, 0)
    g = tl.load(GC + (c * HV + h) * 64 + t)
    kh = h // (HV // H)
    k = tl.load(
        K + token[:, None] * KS + kh * DK + d[None, :],
        (token[:, None] < end) & (d[None, :] < DK),
        0,
    ).to(tl.float32)
    v = tl.load(
        V + token[:, None] * VS + h * DV + d[None, :],
        (token[:, None] < end) & (d[None, :] < DV),
        0,
    ).to(tl.float32)
    w = tl.dot(
        inv.to(K.dtype.element_ty),
        (k * (beta * tl.exp(g))[:, None]).to(K.dtype.element_ty),
    )
    u = tl.dot(inv.to(V.dtype.element_ty), (v * beta[:, None]).to(V.dtype.element_ty))
    tl.store(W + ((c * HV + h) * 64 + t[:, None]) * DK + d[None, :], w, d[None, :] < DK)
    tl.store(U + ((c * HV + h) * 64 + t[:, None]) * DV + d[None, :], u, d[None, :] < DV)


@tr.jit
def _gdn_chunk_scan(
    BASE,
    CAP: tl.constexpr,
    Q,
    K,
    Cu,
    GC,
    QK,
    W,
    U,
    S,
    O,
    F,
    HS,
    UP,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    ns = tl.arange(0, NN)
    starts = tl.load(Cu + ns, ns < N, 0).to(tl.int64 if WIDE else tl.int32)
    ends = tl.load(Cu + ns + 1, ns < N, 0).to(tl.int64 if WIDE else tl.int32)
    offset = tl.sum(tl.where(ns < seq, tl.cdiv(ends - starts, 64), 0), 0)
    start = tl.load(Cu + seq).to(tl.int64 if WIDE else tl.int32)
    end = tl.load(Cu + seq + 1).to(tl.int64 if WIDE else tl.int32)
    if N == 1 or N > CAP:
        first = tl.maximum(0, BASE - offset)
        stop = tl.minimum(tl.cdiv(end - start, 64), BASE + CAP - offset)
    else:
        counts = tl.where(ns < N, tl.cdiv(ends - starts, 64), 0)
        first_round, first_seq = _round_locate(counts, ns, BASE, N)
        stop_round, stop_seq = _round_locate(counts, ns, BASE + CAP, N)
        first = tl.minimum(tl.cdiv(end - start, 64), first_round + (seq < first_seq))
        stop = tl.minimum(tl.cdiv(end - start, 64), stop_round + (seq < stop_seq))
    if BASE > 0 and first >= stop:
        return
    kdim = tl.arange(0, BK)
    vdim = tile * BV + tl.arange(0, BV)
    t = tl.arange(0, 64)
    stateoff = ((seq * HV + h) * DV + vdim[None, :]) * DK + kdim[:, None]
    state = tl.load(S + stateoff, (vdim[None, :] < DV) & (kdim[:, None] < DK), 0)
    for ch in range(first, stop):
        if N == 1 or N > CAP:
            c = offset + ch - BASE
        else:
            c = (
                tl.sum(tl.minimum(counts, ch), 0)
                + tl.sum(((ns < seq) & (counts > ch)).to(tl.int32), 0)
                - BASE
            )
        token = start + ch * 64 + t
        gc = tl.load(GC + (c * HV + h) * 64 + t)
        last = tl.sum(tl.where(t == 63, gc, 0.0), 0)
        w = tl.load(
            W + ((c * HV + h) * 64 + t[:, None]) * DK + kdim[None, :],
            kdim[None, :] < DK,
            0,
        )
        u = tl.load(
            U + ((c * HV + h) * 64 + t[:, None]) * DV + vdim[None, :],
            vdim[None, :] < DV,
            0,
        )
        update = u.to(tl.float32) - tl.dot(w, state.to(w.dtype))
        kh = h // (HV // H)
        tl.store(
            HS + ((c * HV + h) * DV + vdim[None, :]) * DK + kdim[:, None],
            state,
            (vdim[None, :] < DV) & (kdim[:, None] < DK),
        )
        tl.store(
            UP + ((c * HV + h) * 64 + t[:, None]) * DV + vdim[None, :],
            update,
            vdim[None, :] < DV,
        )
        k = tl.load(
            K + token[:, None] * KS + kh * DK + kdim[None, :],
            (token[:, None] < end) & (kdim[None, :] < DK),
            0,
        ).to(tl.float32)
        state = tl.exp(last) * state + tl.dot(
            tl.trans((k * tl.exp(last - gc)[:, None]).to(K.dtype.element_ty)),
            update.to(K.dtype.element_ty),
        )
    tl.store(F + stateoff, state, (vdim[None, :] < DV) & (kdim[:, None] < DK))


@tr.jit
def _gdn_chunk_output(
    BASE,
    CAP: tl.constexpr,
    Q,
    Cu,
    GC,
    K,
    HS,
    UP,
    O,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    start, end, seq = _gdn_chunk_locate(Cu, c + BASE, N, NN, WIDE, CAP)
    if seq >= N:
        return
    t = tl.arange(0, 64)
    kd = tl.arange(0, BK)
    vd = tile * BV + tl.arange(0, BV)
    token = start + t
    gc = tl.load(GC + (c * HV + h) * 64 + t)
    q = tl.load(
        Q + token[:, None] * QS + (h // (HV // H)) * DK + kd[None, :],
        (token[:, None] < end) & (kd[None, :] < DK),
        0,
    ).to(tl.float32)
    state = tl.load(
        HS + ((c * HV + h) * DV + vd[:, None]) * DK + kd[None, :],
        (vd[:, None] < DV) & (kd[None, :] < DK),
        0,
    )
    keys = tl.load(
        K + token[:, None] * KS + (h // (HV // H)) * DK + kd[None, :],
        (token[:, None] < end) & (kd[None, :] < DK),
        0,
    )
    qk = tl.dot(q.to(keys.dtype), tl.trans(keys)) * tl.exp(gc[:, None] - gc[None, :])
    qk = tl.where(t[:, None] >= t[None, :], qk, 0).to(UP.dtype.element_ty)
    update = tl.load(
        UP + ((c * HV + h) * 64 + t[:, None]) * DV + vd[None, :], vd[None, :] < DV, 0
    )
    o = (
        tl.dot((q * tl.exp(gc)[:, None]).to(state.dtype), tl.trans(state))
        + tl.dot(qk, update)
    ) * SCALE
    tl.store(
        O + (token[:, None] * HV + h) * DV + vd[None, :],
        o,
        (token[:, None] < end) & (vd[None, :] < DV),
    )


@tr.jit
def _gdn_short_chunk_prep(
    BASE,
    CAP: tl.constexpr,
    Q,
    K,
    G,
    B,
    Cu,
    A,
    GC,
    QK,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    BK: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    start, end, seq = _gdn_chunk_locate(Cu, c + BASE, N, NN, WIDE, CAP)
    if seq >= N:
        return
    t = tl.arange(0, 64)
    x = tl.arange(0, 16)
    d = tl.arange(0, BK)
    token = start + t
    gc = tl.cumsum(tl.load(G + token * HV + h, token < end, 0))
    beta = tl.load(B + token * HV + h, token < end, 0)
    gx, gy = tl.split(tl.reshape(tl.trans(tl.reshape(gc, (4, 16))), (16, 2, 2)))
    g0, g2 = tl.split(gx)
    g1, g3 = tl.split(gy)
    bx, by = tl.split(tl.reshape(tl.trans(tl.reshape(beta, (4, 16))), (16, 2, 2)))
    b0, b2 = tl.split(bx)
    b1, b3 = tl.split(by)
    kh = h // (HV // H)
    k0 = tl.load(
        K + (start + x[:, None]) * KS + kh * DK + d[None, :],
        (start + x[:, None] < end) & (d[None, :] < DK),
        0,
    )
    k1 = tl.load(
        K + (start + 16 + x[:, None]) * KS + kh * DK + d[None, :],
        (start + 16 + x[:, None] < end) & (d[None, :] < DK),
        0,
    )
    k2 = tl.load(
        K + (start + 32 + x[:, None]) * KS + kh * DK + d[None, :],
        (start + 32 + x[:, None] < end) & (d[None, :] < DK),
        0,
    )
    k3 = tl.load(
        K + (start + 48 + x[:, None]) * KS + kh * DK + d[None, :],
        (start + 48 + x[:, None] < end) & (d[None, :] < DK),
        0,
    )
    a00 = tl.dot(k0, tl.trans(k0)) * tl.exp(g0[:, None] - g0[None, :]) * b0[:, None]
    a11 = tl.dot(k1, tl.trans(k1)) * tl.exp(g1[:, None] - g1[None, :]) * b1[:, None]
    a22 = tl.dot(k2, tl.trans(k2)) * tl.exp(g2[:, None] - g2[None, :]) * b2[:, None]
    a33 = tl.dot(k3, tl.trans(k3)) * tl.exp(g3[:, None] - g3[None, :]) * b3[:, None]
    a10 = tl.dot(k1, tl.trans(k0)) * tl.exp(g1[:, None] - g0[None, :]) * b1[:, None]
    a20 = tl.dot(k2, tl.trans(k0)) * tl.exp(g2[:, None] - g0[None, :]) * b2[:, None]
    a21 = tl.dot(k2, tl.trans(k1)) * tl.exp(g2[:, None] - g1[None, :]) * b2[:, None]
    a30 = tl.dot(k3, tl.trans(k0)) * tl.exp(g3[:, None] - g0[None, :]) * b3[:, None]
    a31 = tl.dot(k3, tl.trans(k1)) * tl.exp(g3[:, None] - g1[None, :]) * b3[:, None]
    a32 = tl.dot(k3, tl.trans(k2)) * tl.exp(g3[:, None] - g2[None, :]) * b3[:, None]
    i00 = _gdn_chunk_inverse16(a00, PREC)
    i11 = _gdn_chunk_inverse16(a11, PREC)
    i22 = _gdn_chunk_inverse16(a22, PREC)
    i33 = _gdn_chunk_inverse16(a33, PREC)
    i10 = -_gdn_chunk_dotfp(_gdn_chunk_dotfp(i11, a10, PREC), i00, PREC)
    i21 = -_gdn_chunk_dotfp(_gdn_chunk_dotfp(i22, a21, PREC), i11, PREC)
    i32 = -_gdn_chunk_dotfp(_gdn_chunk_dotfp(i33, a32, PREC), i22, PREC)
    i20 = -_gdn_chunk_dotfp(
        i22, _gdn_chunk_dotfp(a20, i00, PREC) + _gdn_chunk_dotfp(a21, i10, PREC), PREC
    )
    i31 = -_gdn_chunk_dotfp(
        i33, _gdn_chunk_dotfp(a31, i11, PREC) + _gdn_chunk_dotfp(a32, i21, PREC), PREC
    )
    i30 = -_gdn_chunk_dotfp(
        i33,
        _gdn_chunk_dotfp(a30, i00, PREC)
        + _gdn_chunk_dotfp(a31, i10, PREC)
        + _gdn_chunk_dotfp(a32, i20, PREC),
        PREC,
    )
    _gdn_chunk_store16(A, c, h, 0, 0, i00, HV)
    _gdn_chunk_store16(A, c, h, 1, 1, i11, HV)
    _gdn_chunk_store16(A, c, h, 2, 2, i22, HV)
    _gdn_chunk_store16(A, c, h, 3, 3, i33, HV)
    _gdn_chunk_store16(A, c, h, 1, 0, i10, HV)
    _gdn_chunk_store16(A, c, h, 2, 0, i20, HV)
    _gdn_chunk_store16(A, c, h, 2, 1, i21, HV)
    _gdn_chunk_store16(A, c, h, 3, 0, i30, HV)
    _gdn_chunk_store16(A, c, h, 3, 1, i31, HV)
    _gdn_chunk_store16(A, c, h, 3, 2, i32, HV)
    z = tl.full((16, 16), 0.0, tl.float32)
    _gdn_chunk_store16(A, c, h, 0, 1, z, HV)
    _gdn_chunk_store16(A, c, h, 0, 2, z, HV)
    _gdn_chunk_store16(A, c, h, 0, 3, z, HV)
    _gdn_chunk_store16(A, c, h, 1, 2, z, HV)
    _gdn_chunk_store16(A, c, h, 1, 3, z, HV)
    _gdn_chunk_store16(A, c, h, 2, 3, z, HV)
    k = tl.load(
        K + token[:, None] * KS + kh * DK + d[None, :],
        (token[:, None] < end) & (d[None, :] < DK),
        0,
    )
    q = tl.load(
        Q + token[:, None] * QS + kh * DK + d[None, :],
        (token[:, None] < end) & (d[None, :] < DK),
        0,
    )
    qk = tl.dot(q, tl.trans(k)) * tl.exp(gc[:, None] - gc[None, :])
    qk = tl.where(t[:, None] >= t[None, :], qk, 0.0)
    tl.store(QK + ((c * HV + h) * 64 + t[:, None]) * 64 + t[None, :], qk)
    tl.store(GC + (c * HV + h) * 64 + t, gc)


@tr.jit
def _gdn_short_chunk_wy(
    BASE,
    CAP: tl.constexpr,
    K,
    V,
    B,
    Cu,
    A,
    GC,
    W,
    U,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    start, end, seq = _gdn_chunk_locate(Cu, c + BASE, N, NN, WIDE, CAP)
    if seq >= N:
        return
    t = tl.arange(0, 64)
    d = tile * 32 + tl.arange(0, 32)
    token = start + t
    inv = tl.load(A + ((c * HV + h) * 64 + t[:, None]) * 64 + t[None, :])
    beta = tl.load(B + token * HV + h, token < end, 0)
    g = tl.load(GC + (c * HV + h) * 64 + t)
    kh = h // (HV // H)
    k = tl.load(
        K + token[:, None] * KS + kh * DK + d[None, :],
        (token[:, None] < end) & (d[None, :] < DK),
        0,
    ).to(tl.float32)
    v = tl.load(
        V + token[:, None] * VS + h * DV + d[None, :],
        (token[:, None] < end) & (d[None, :] < DV),
        0,
    ).to(tl.float32)
    w = tl.dot(
        inv.to(K.dtype.element_ty),
        (k * (beta * tl.exp(g))[:, None]).to(K.dtype.element_ty),
    )
    u = tl.dot(inv.to(V.dtype.element_ty), (v * beta[:, None]).to(V.dtype.element_ty))
    tl.store(W + ((c * HV + h) * 64 + t[:, None]) * DK + d[None, :], w, d[None, :] < DK)
    tl.store(U + ((c * HV + h) * 64 + t[:, None]) * DV + d[None, :], u, d[None, :] < DV)


@tr.jit
def _gdn_short_chunk_output(
    BASE,
    CAP: tl.constexpr,
    Q,
    Cu,
    GC,
    QK,
    HS,
    UP,
    O,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    QS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    start, end, seq = _gdn_chunk_locate(Cu, c + BASE, N, NN, WIDE, CAP)
    if seq >= N:
        return
    t = tl.arange(0, 64)
    kd = tl.arange(0, BK)
    vd = tile * BV + tl.arange(0, BV)
    token = start + t
    gc = tl.load(GC + (c * HV + h) * 64 + t)
    q = tl.load(
        Q + token[:, None] * QS + (h // (HV // H)) * DK + kd[None, :],
        (token[:, None] < end) & (kd[None, :] < DK),
        0,
    ).to(tl.float32)
    state = tl.load(
        HS + ((c * HV + h) * DV + vd[:, None]) * DK + kd[None, :],
        (vd[:, None] < DV) & (kd[None, :] < DK),
        0,
    )
    qk = tl.load(QK + ((c * HV + h) * 64 + t[:, None]) * 64 + t[None, :])
    update = tl.load(
        UP + ((c * HV + h) * 64 + t[:, None]) * DV + vd[None, :], vd[None, :] < DV, 0
    )
    o = (
        tl.dot((q * tl.exp(gc)[:, None]).to(state.dtype), tl.trans(state))
        + tl.dot(qk, update)
    ) * SCALE
    tl.store(
        O + (token[:, None] * HV + h) * DV + vd[None, :],
        o,
        (token[:, None] < end) & (vd[None, :] < DV),
    )


@tr.jit
def _gdn_fp32_chunk_wy(
    BASE,
    CAP: tl.constexpr,
    K,
    V,
    B,
    Cu,
    A,
    GC,
    W,
    U,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    start, end, seq = _gdn_chunk_locate(Cu, c + BASE, N, NN, WIDE, CAP)
    if seq >= N:
        return
    t = tl.arange(0, 64)
    d = tile * 32 + tl.arange(0, 32)
    token = start + t
    inv = tl.load(A + ((c * HV + h) * 64 + t[:, None]) * 64 + t[None, :])
    beta = tl.load(B + token * HV + h, token < end, 0)
    g = tl.load(GC + (c * HV + h) * 64 + t)
    kh = h // (HV // H)
    k = tl.load(
        K + token[:, None] * KS + kh * DK + d[None, :],
        (token[:, None] < end) & (d[None, :] < DK),
        0,
    ).to(tl.float32)
    v = tl.load(
        V + token[:, None] * VS + h * DV + d[None, :],
        (token[:, None] < end) & (d[None, :] < DV),
        0,
    ).to(tl.float32)
    w = _gdn_chunk_dotfp(inv, k * (beta * tl.exp(g))[:, None], PREC=PREC)
    u = _gdn_chunk_dotfp(inv, v * beta[:, None], PREC=PREC)
    tl.store(W + ((c * HV + h) * 64 + t[:, None]) * DK + d[None, :], w, d[None, :] < DK)
    tl.store(U + ((c * HV + h) * 64 + t[:, None]) * DV + d[None, :], u, d[None, :] < DV)


@tr.jit
def _gdn_fp32_chunk_scan(
    BASE,
    CAP: tl.constexpr,
    Q,
    K,
    Cu,
    GC,
    QK,
    W,
    U,
    S,
    O,
    F,
    HS,
    UP,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    ns = tl.arange(0, NN)
    starts = tl.load(Cu + ns, ns < N, 0).to(tl.int64 if WIDE else tl.int32)
    ends = tl.load(Cu + ns + 1, ns < N, 0).to(tl.int64 if WIDE else tl.int32)
    offset = tl.sum(tl.where(ns < seq, tl.cdiv(ends - starts, 64), 0), 0)
    start = tl.load(Cu + seq).to(tl.int64 if WIDE else tl.int32)
    end = tl.load(Cu + seq + 1).to(tl.int64 if WIDE else tl.int32)
    if N == 1 or N > CAP:
        first = tl.maximum(0, BASE - offset)
        stop = tl.minimum(tl.cdiv(end - start, 64), BASE + CAP - offset)
    else:
        counts = tl.where(ns < N, tl.cdiv(ends - starts, 64), 0)
        first_round, first_seq = _round_locate(counts, ns, BASE, N)
        stop_round, stop_seq = _round_locate(counts, ns, BASE + CAP, N)
        first = tl.minimum(tl.cdiv(end - start, 64), first_round + (seq < first_seq))
        stop = tl.minimum(tl.cdiv(end - start, 64), stop_round + (seq < stop_seq))
    if BASE > 0 and first >= stop:
        return
    kdim = tl.arange(0, BK)
    vdim = tile * BV + tl.arange(0, BV)
    t = tl.arange(0, 64)
    stateoff = ((seq * HV + h) * DV + vdim[:, None]) * DK + kdim[None, :]
    state = tl.load(S + stateoff, (vdim[:, None] < DV) & (kdim[None, :] < DK), 0)
    for ch in range(first, stop):
        if N == 1 or N > CAP:
            c = offset + ch - BASE
        else:
            c = (
                tl.sum(tl.minimum(counts, ch), 0)
                + tl.sum(((ns < seq) & (counts > ch)).to(tl.int32), 0)
                - BASE
            )
        token = start + ch * 64 + t
        gc = tl.load(GC + (c * HV + h) * 64 + t)
        last = tl.sum(tl.where(t == 63, gc, 0.0), 0)
        w = tl.load(
            W + ((c * HV + h) * 64 + t[:, None]) * DK + kdim[None, :],
            kdim[None, :] < DK,
            0,
        )
        u = tl.load(
            U + ((c * HV + h) * 64 + t[:, None]) * DV + vdim[None, :],
            vdim[None, :] < DV,
            0,
        )
        update = u - _gdn_chunk_dotfp(w, tl.trans(state), PREC=PREC)
        kh = h // (HV // H)
        tl.store(
            HS + ((c * HV + h) * DV + vdim[:, None]) * DK + kdim[None, :],
            state,
            (vdim[:, None] < DV) & (kdim[None, :] < DK),
        )
        tl.store(
            UP + ((c * HV + h) * 64 + t[:, None]) * DV + vdim[None, :],
            update,
            vdim[None, :] < DV,
        )
        k = tl.load(
            K + token[:, None] * KS + kh * DK + kdim[None, :],
            (token[:, None] < end) & (kdim[None, :] < DK),
            0,
        ).to(tl.float32)
        state = tl.exp(last) * state + _gdn_chunk_dotfp(
            tl.trans(update), k * tl.exp(last - gc)[:, None], PREC=PREC
        )
    tl.store(F + stateoff, state, (vdim[:, None] < DV) & (kdim[None, :] < DK))


@tr.jit
def _gdn_fp32_chunk_output(
    BASE,
    CAP: tl.constexpr,
    Q,
    Cu,
    GC,
    QK,
    HS,
    UP,
    O,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    QS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    PREC: tl.constexpr,
    WIDE: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    start, end, seq = _gdn_chunk_locate(Cu, c + BASE, N, NN, WIDE, CAP)
    if seq >= N:
        return
    t = tl.arange(0, 64)
    kd = tl.arange(0, BK)
    vd = tile * BV + tl.arange(0, BV)
    token = start + t
    gc = tl.load(GC + (c * HV + h) * 64 + t)
    q = tl.load(
        Q + token[:, None] * QS + (h // (HV // H)) * DK + kd[None, :],
        (token[:, None] < end) & (kd[None, :] < DK),
        0,
    ).to(tl.float32)
    state = tl.load(
        HS + ((c * HV + h) * DV + vd[:, None]) * DK + kd[None, :],
        (vd[:, None] < DV) & (kd[None, :] < DK),
        0,
    )
    qk = tl.load(QK + ((c * HV + h) * 64 + t[:, None]) * 64 + t[None, :])
    update = tl.load(
        UP + ((c * HV + h) * 64 + t[:, None]) * DV + vd[None, :], vd[None, :] < DV, 0
    )
    o = (
        _gdn_chunk_dotfp(q * tl.exp(gc)[:, None], tl.trans(state), PREC)
        + _gdn_chunk_dotfp(qk, update, PREC)
    ) * SCALE
    tl.store(
        O + (token[:, None] * HV + h) * DV + vd[None, :],
        o,
        (token[:, None] < end) & (vd[None, :] < DV),
    )


@tr.jit
def _decode(
    Q,
    K,
    V,
    A,
    B,
    AL,
    Bias,
    S,
    O,
    Ids,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FACTOR: tl.constexpr,
    FMA: tl.constexpr,
    ROUND_BETA: tl.constexpr,
):
    n, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Ids + n).to(tl.int64)
    d = tl.arange(0, BK)
    vd = tile * BV + tl.arange(0, BV)
    if slot <= 0:
        tl.store(O + (n * HV + h) * DV + vd, 0, vd < DV)
        return
    kh = h // (HV // H)
    q = tl.load(Q + n * QS + kh * DK + d, d < DK, 0).to(tl.float32)
    k = tl.load(K + n * KS + kh * DK + d, d < DK, 0).to(tl.float32)
    v = tl.load(V + n * VS + h * DV + vd, vd < DV, 0).to(tl.float32)
    if NORMALIZE:
        q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6)
        k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    gate = tl.load(A + n * AS + h).to(tl.float32) + tl.load(Bias + h).to(tl.float32)
    sp = tl.where(gate <= 20, tl.log(1 + tl.exp(gate)), gate)
    decay = tl.exp(-tl.exp(tl.load(AL + h).to(tl.float32)) * sp)
    beta = tl.sigmoid(tl.load(B + n * BS + h).to(tl.float32))
    if ROUND_BETA:
        beta = beta.to(B.dtype.element_ty).to(tl.float32)
    off = ((slot * HV + h) * DV + vd[:, None]) * DK + d[None, :]
    mask = (vd[:, None] < DV) & (d[None, :] < DK)
    state = tl.load(S + off, mask, 0)
    if FACTOR:
        residual = (v - tl.sum(state * k[None, :], 1) * decay) * beta
        if FMA:
            state = tl.fma(residual[:, None], k[None, :], state * decay)
        else:
            state = state * decay + residual[:, None] * k[None, :]
    else:
        state *= decay
        residual = (v - tl.sum(state * k[None, :], 1)) * beta
        if FMA:
            state = tl.fma(residual[:, None], k[None, :], state)
        else:
            state += residual[:, None] * k[None, :]
    out = tl.sum(state * (q * SCALE)[None, :], 1)
    tl.store(O + (n * HV + h) * DV + vd, out, vd < DV)
    tl.store(S + off, state, mask)


if gluon is not None:

    @gluon.jit
    def _packed_decode_128(
        Q,
        K,
        V,
        A,
        B,
        AL,
        Bias,
        S,
        O,
        Ids,
        QS: gl.constexpr,
        KS: gl.constexpr,
        VS: gl.constexpr,
        SCALE: gl.constexpr,
        WIDE: gl.constexpr,
        PROJECT: gl.constexpr,
    ):
        # Keep Q/K in the state tile's register layout: no shared-memory
        # layout conversion or CTA barrier between normalization and GEMV.
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [1, 1], [1, 0])
        BV: gl.constexpr = 16
        pid = gl.program_id(0)
        tile, h, n = (
            pid % (128 // BV),
            (pid // (128 // BV)) % 32,
            pid // (32 * (128 // BV)),
        )
        slot = gl.load(Ids + n).to(gl.int64)
        if not WIDE:
            # Preserve every nonpositive int64 padding value before narrowing.
            slot = gl.where(slot > 0, slot, 0).to(gl.int32)
        d = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        vd = tile * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
        off = ((slot * 32 + h) * 128 + vd[:, None]) * 128 + d[None, :]
        # Issue the large state load before computing norms and gates.
        state = gl.load(S + off, slot > 0, 0)
        q = gl.load(Q + n * QS + (h // 2) * 128 + d).to(gl.float32)
        k = gl.load(K + n * KS + (h // 2) * 128 + d).to(gl.float32)
        q2, k2 = gl.sum(q * q, 0), gl.sum(k * k, 0)
        q *= gl.rsqrt(q2 + 1e-6)
        k *= gl.rsqrt(k2 + 1e-6)
        gate = gl.load(A + n * 32 + h).to(gl.float32) + gl.load(Bias + h)
        sp = gl.where(gate <= 20, gl.log(1 + gl.exp(gate)), gate)
        decay = gl.exp(-gl.exp(gl.load(AL + h)) * sp)
        beta = 1.0 / (1.0 + gl.exp(-gl.load(B + n * 32 + h).to(gl.float32)))
        beta = beta.to(B.dtype.element_ty).to(gl.float32)
        v = gl.load(V + n * VS + h * 128 + vd).to(gl.float32)
        state *= decay
        if PROJECT:
            # Project the decayed state before the dependent rank-one update:
            # (state + residual * k) @ q = state @ q + residual * (k @ q).
            sk = gl.sum(state * k[None, :], 1)
            sq = gl.sum(state * (q * SCALE)[None, :], 1)
            kq = gl.sum(k * (q * SCALE), 0)
            residual = (v - sk) * beta
            state += residual[:, None] * k[None, :]
            out = sq + residual * kq
        else:
            residual = (v - gl.sum(state * k[None, :], 1)) * beta
            state += residual[:, None] * k[None, :]
            out = gl.sum(state * (q * SCALE)[None, :], 1)
        gl.store(O + (n * 32 + h) * 128 + vd, gl.where(slot > 0, out, 0.0))
        gl.store(S + off, state, slot > 0)


def _mixed_workspace_plan(tokens, sequences, value_heads, target=64):
    # Sum ceil(length / 64) <= floor((T + 63*N) / 64), including empty requests.
    total = (tokens + 63 * sequences) // 64
    # A + GC + W + U + HS; QK is fused into output; UP aliases U after its last read.
    bytes_per_chunk = value_heads * ((64**2 + 2 * 64 * 128 + 128**2) * 2 + 64 * 4)
    max_slots = _WORKSPACE_LIMIT_BYTES // bytes_per_chunk
    if not max_slots:
        return total, 0, 0
    if sequences > 1:
        target = ((target + sequences - 1) // sequences) * sequences
    capacity = min(total, target, max_slots)
    return total, capacity, capacity * bytes_per_chunk


def _run_mixed_chunk(args, scale):
    # Fixed MetaX configuration selected outside graph capture.
    bv, warps, prec = 32, 4, "tf32x3"
    q, k, v, g, b, s, cu = args
    t, h, dk = q.shape[1:]
    if t < 128:
        return _run_short_chunk(args, scale)
    hv, dv = v.shape[2:]
    n = cu.numel() - 1
    bv = 32 if n <= 2 else 64
    target = 96 if q.dtype == torch.bfloat16 and n == 16 and t >= 4096 else 64
    total, c, _ = _mixed_workspace_plan(t, n, hv, target=target)
    if not c:
        raise NotImplementedError("GDN chunk workspace cannot fit one chunk")
    wide = (
        max(t * q.stride(1), t * k.stride(1), t * v.stride(1), s.numel(), t * hv)
        >= 2**31
    )
    nn = tr.next_power_of_2(n)
    bw = 64
    obv = 128
    owarps = 4
    alloc = lambda shape: torch.empty(shape, device=q.device, dtype=q.dtype)
    a = alloc((c, hv, 64, 64))
    qk = a  # QK is computed in output; no separate buffer
    gc = torch.empty((c, hv, 64), device=q.device, dtype=torch.float32)
    w = alloc((c, hv, 64, dk))
    u = alloc((c, hv, 64, dv))
    hs = alloc((c, hv, dv, dk))
    # Each scan CTA owns disjoint V columns and loads U before replacing it.
    # The output kernel consumes these updates before the next window's WY.
    up = u
    o = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    f = torch.empty_like(s)
    # Reuse the same workspace across graph launches. After the first window,
    # F carries state; S is never modified. All mappings are derived from Cu
    # on device, so replay remains valid when Cu contents change.
    for base in range(0, total, c):
        _gdn_chunk_prep[(c, hv)](
            base,
            c,
            q,
            k,
            g,
            b,
            cu,
            a,
            gc,
            qk,
            t,
            h,
            hv,
            dk,
            n,
            nn,
            q.stride(1),
            k.stride(1),
            tr.next_power_of_2(dk),
            prec,
            wide,
            num_warps=4,
        )
        _gdn_chunk_wy[(c, hv, tr.cdiv(max(dk, dv), bw))](
            base,
            c,
            k,
            v,
            b,
            cu,
            a,
            gc,
            w,
            u,
            h,
            hv,
            dk,
            dv,
            n,
            nn,
            k.stride(1),
            v.stride(1),
            prec,
            wide,
            bw,
            num_warps=4,
        )
        _gdn_chunk_scan[(n, hv, tr.cdiv(dv, bv))](
            base,
            c,
            q,
            k,
            cu,
            gc,
            qk,
            w,
            u,
            s if base == 0 else f,
            o,
            f,
            hs,
            up,
            h,
            hv,
            dk,
            dv,
            n,
            nn,
            q.stride(1),
            k.stride(1),
            tr.next_power_of_2(dk),
            bv,
            scale,
            prec,
            wide,
            num_warps=warps,
            num_stages=1,
            pipeline="cpasync-mixed",
        )
        _gdn_chunk_output[(c, hv, tr.cdiv(dv, obv))](
            base,
            c,
            q,
            cu,
            gc,
            k,
            hs,
            up,
            o,
            h,
            hv,
            dk,
            dv,
            n,
            nn,
            q.stride(1),
            k.stride(1),
            tr.next_power_of_2(dk),
            obv,
            scale,
            prec,
            wide,
            num_warps=owarps,
        )
    return o, f


def _short_workspace_plan(tokens, sequences, value_heads):
    # Sum ceil(length / 64) <= floor((T + 63*N) / 64), including empty requests.
    total = (tokens + 63 * sequences) // 64
    # A + QK + GC + W + U + HS; UP aliases U after its last read.
    bytes_per_chunk = value_heads * ((2 * 64**2 + 2 * 64 * 128 + 128**2) * 2 + 64 * 4)
    max_slots = _WORKSPACE_LIMIT_BYTES // bytes_per_chunk
    if not max_slots:
        return total, 0, 0
    capacity = min(total, _SHORT_MAX_CHUNKS, 1 << (max_slots.bit_length() - 1))
    return total, capacity, capacity * bytes_per_chunk


def _run_short_chunk(args, scale):
    # Fixed MetaX configuration selected outside graph capture.
    bv, warps, prec = 32, 4, "tf32x3"
    q, k, v, g, b, s, cu = args
    t, h, dk = q.shape[1:]
    hv, dv = v.shape[2:]
    n = cu.numel() - 1
    bv = 32 if n <= 2 else 64
    total, c, _ = _short_workspace_plan(t, n, hv)
    if not c:
        raise NotImplementedError("GDN chunk workspace cannot fit one chunk")
    wide = (
        max(t * q.stride(1), t * k.stride(1), t * v.stride(1), s.numel(), t * hv)
        >= 2**31
    )
    nn = tr.next_power_of_2(n)
    alloc = lambda shape: torch.empty(shape, device=q.device, dtype=q.dtype)
    a = alloc((c, hv, 64, 64))
    qk = torch.empty_like(a)
    gc = torch.empty((c, hv, 64), device=q.device, dtype=torch.float32)
    w = alloc((c, hv, 64, dk))
    u = alloc((c, hv, 64, dv))
    hs = alloc((c, hv, dv, dk))
    # Each scan CTA owns disjoint V columns and loads U before replacing it.
    # The output kernel consumes these updates before the next window's WY.
    up = u
    o = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    f = torch.empty_like(s)
    # Reuse the same workspace across graph launches. After the first window,
    # F carries state; S is never modified. All mappings are derived from Cu
    # on device, so replay remains valid when Cu contents change.
    for base in range(0, total, c):
        _gdn_short_chunk_prep[(c, hv)](
            base,
            c,
            q,
            k,
            g,
            b,
            cu,
            a,
            gc,
            qk,
            t,
            h,
            hv,
            dk,
            n,
            nn,
            q.stride(1),
            k.stride(1),
            tr.next_power_of_2(dk),
            prec,
            wide,
            num_warps=4,
        )
        _gdn_short_chunk_wy[(c, hv, tr.cdiv(max(dk, dv), 32))](
            base,
            c,
            k,
            v,
            b,
            cu,
            a,
            gc,
            w,
            u,
            h,
            hv,
            dk,
            dv,
            n,
            nn,
            k.stride(1),
            v.stride(1),
            prec,
            wide,
            num_warps=4,
        )
        _gdn_chunk_scan[(n, hv, tr.cdiv(dv, bv))](
            base,
            c,
            q,
            k,
            cu,
            gc,
            qk,
            w,
            u,
            s if base == 0 else f,
            o,
            f,
            hs,
            up,
            h,
            hv,
            dk,
            dv,
            n,
            nn,
            q.stride(1),
            k.stride(1),
            tr.next_power_of_2(dk),
            bv,
            scale,
            prec,
            wide,
            num_warps=warps,
            num_stages=1,
            pipeline="cpasync-mixed",
        )
        _gdn_short_chunk_output[(c, hv, tr.cdiv(dv, bv))](
            base,
            c,
            q,
            cu,
            gc,
            qk,
            hs,
            up,
            o,
            h,
            hv,
            dk,
            dv,
            n,
            nn,
            q.stride(1),
            tr.next_power_of_2(dk),
            bv,
            scale,
            prec,
            wide,
            num_warps=4,
        )
    return o, f


def _fp32_workspace_plan(tokens, sequences, value_heads):
    # Sum ceil(length / 64) <= floor((T + 63*N) / 64), including empty requests.
    total = (tokens + 63 * sequences) // 64
    # A + QK + GC + W + U + HS; UP aliases U after its last read.
    bytes_per_chunk = value_heads * (2 * 64**2 + 64 + 2 * 64 * 128 + 128**2) * 4
    max_slots = _WORKSPACE_LIMIT_BYTES // bytes_per_chunk
    if not max_slots:
        return total, 0, 0
    capacity = min(total, _FP32_MAX_CHUNKS, 1 << (max_slots.bit_length() - 1))
    return total, capacity, capacity * bytes_per_chunk


def _run_fp32_chunk(args, scale):
    # Fixed MetaX configuration selected outside graph capture.
    bv, warps, prec = 64, 4, "tf32x3"
    q, k, v, g, b, s, cu = args
    t, h, dk = q.shape[1:]
    hv, dv = v.shape[2:]
    n = cu.numel() - 1
    total, c, _ = _fp32_workspace_plan(t, n, hv)
    if not c:
        raise NotImplementedError("GDN chunk workspace cannot fit one chunk")
    wide = (
        max(t * q.stride(1), t * k.stride(1), t * v.stride(1), s.numel(), t * hv)
        >= 2**31
    )
    nn = tr.next_power_of_2(n)
    alloc = lambda shape: torch.empty(shape, device=q.device, dtype=torch.float32)
    a = alloc((c, hv, 64, 64))
    qk = torch.empty_like(a)
    gc = alloc((c, hv, 64))
    w = alloc((c, hv, 64, dk))
    u = alloc((c, hv, 64, dv))
    hs = alloc((c, hv, dv, dk))
    # Each scan CTA owns disjoint V columns and loads U before replacing it.
    # The output kernel consumes these updates before the next window's WY.
    up = u
    o = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    f = torch.empty_like(s)
    # Reuse the same workspace across graph launches. After the first window,
    # F carries state; S is never modified. All mappings are derived from Cu
    # on device, so replay remains valid when Cu contents change.
    for base in range(0, total, c):
        _gdn_short_chunk_prep[(c, hv)](
            base,
            c,
            q,
            k,
            g,
            b,
            cu,
            a,
            gc,
            qk,
            t,
            h,
            hv,
            dk,
            n,
            nn,
            q.stride(1),
            k.stride(1),
            tr.next_power_of_2(dk),
            prec,
            wide,
            num_warps=4,
        )
        _gdn_fp32_chunk_wy[(c, hv, tr.cdiv(max(dk, dv), 32))](
            base,
            c,
            k,
            v,
            b,
            cu,
            a,
            gc,
            w,
            u,
            h,
            hv,
            dk,
            dv,
            n,
            nn,
            k.stride(1),
            v.stride(1),
            prec,
            wide,
            num_warps=4,
        )
        _gdn_fp32_chunk_scan[(n, hv, tr.cdiv(dv, bv))](
            base,
            c,
            q,
            k,
            cu,
            gc,
            qk,
            w,
            u,
            s if base == 0 else f,
            o,
            f,
            hs,
            up,
            h,
            hv,
            dk,
            dv,
            n,
            nn,
            q.stride(1),
            k.stride(1),
            tr.next_power_of_2(dk),
            bv,
            scale,
            prec,
            wide,
            num_warps=warps,
            num_stages=1,
            pipeline="cpasync-mixed",
        )
        _gdn_fp32_chunk_output[(c, hv, tr.cdiv(dv, bv))](
            base,
            c,
            q,
            cu,
            gc,
            qk,
            hs,
            up,
            o,
            h,
            hv,
            dk,
            dv,
            n,
            nn,
            q.stride(1),
            tr.next_power_of_2(dk),
            bv,
            scale,
            prec,
            wide,
            num_warps=4,
        )
    return o, f


def _fused_decode(
    qkv,
    a,
    b,
    alog,
    bias,
    state,
    ids,
    out=None,
    *,
    scale=None,
    normalize=True,
    bv=None,
    warps=None,
    factor=False,
    fma=False,
    round_beta=True,
):
    """One token per request, distinct positive state slots; <=0 denotes padding.

    Packed [T,2*H*K+HV*V] or separate [1,T,H,K]/[1,T,HV,V] no-copy views.
    Mutates FP32 state; caller owns out/cache. No data tensors are cached.
    """
    if state.ndim != 4 or state.dtype != torch.float32 or not state.is_contiguous():
        raise ValueError("state must be contiguous FP32 [slots,HV,V,K]")
    slots, hv, dv, dk = state.shape
    if isinstance(qkv, tuple):
        q, k, v = qkv
        if q.ndim != 4 or q.shape != k.shape or q.shape[0] != 1:
            raise ValueError("separate Q/K must have shape [1,T,H,K]")
        n, h = q.shape[1:3]
        if q.shape[3] != dk or v.shape != (1, n, hv, dv):
            raise ValueError("invalid QKV dimensions")
    else:
        if qkv.ndim != 2:
            raise ValueError("packed QKV must be rank two")
        n, packed = qkv.shape
        width = packed - hv * dv
        if width <= 0 or width % (2 * dk):
            raise ValueError("invalid packed QKV dimensions")
        h = width // (2 * dk)
        q = qkv[:, : h * dk].view(1, n, h, dk)
        k = qkv[:, h * dk : 2 * h * dk].view(1, n, h, dk)
        v = qkv[:, 2 * h * dk :].view(1, n, hv, dv)
    if h < 1 or hv % h or min(dk, dv) < 1 or max(dk, dv) > 256:
        raise NotImplementedError("unsupported head dimensions")
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("QKV must share BF16/FP16 dtype")
    if any(t.stride(-1) != 1 or t.stride(-2) != t.shape[-1] for t in (q, k, v)):
        raise ValueError("head dimensions must be contiguous")
    if (
        a.shape != (n, hv)
        or b.shape != a.shape
        or a.dtype != q.dtype
        or b.dtype != q.dtype
    ):
        raise ValueError("invalid gate activations")
    if (
        alog.shape != (hv,)
        or bias.shape != (hv,)
        or alog.dtype != torch.float32
        or bias.dtype != torch.float32
    ):
        raise ValueError("A_log and bias must be FP32 [HV]")
    if ids.shape != (n,) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("invalid state indices")
    if any(not t.is_contiguous() for t in (a, b, alog, bias, ids)):
        raise ValueError("gate and index buffers must be contiguous")
    if out is None:
        out = torch.empty((1, n, hv, dv), device=q.device, dtype=q.dtype)
    if (
        out.shape not in ((1, n, hv, dv), (n, 1, hv, dv))
        or out.dtype != q.dtype
        or not out.is_contiguous()
    ):
        raise ValueError("invalid output")
    tensors = (q, k, v, a, b, alog, bias, state, ids, out)
    if q.device.type != "cuda" or any(
        t.device != q.device or t.requires_grad for t in tensors
    ):
        raise ValueError("forward-only tensors on the same accelerator required")
    if n and any(out.data_ptr() == t.data_ptr() for t in tensors[:-1]):
        raise ValueError("output must not alias an input")
    if (
        n in (1, 3)
        and q.dtype == torch.bfloat16
        and gluon is not None
        and not isinstance(qkv, tuple)
        and (h, hv, dk, dv) == (16, 32, 128, 128)
        and normalize
        and bv is None
        and warps is None
        and not factor
        and not fma
        and round_beta
    ):
        with torch.cuda.device(q.device):
            _packed_decode_128[(n * hv * 8,)](
                q,
                k,
                v,
                a,
                b,
                alog,
                bias,
                state,
                out,
                ids,
                q.stride(1),
                k.stride(1),
                v.stride(1),
                dk**-0.5 if scale is None else scale,
                state.numel() >= 2**31,
                n == 3,
                num_warps=1,
                num_stages=1,
                enable_fp_fusion=False,
            )
        return out, state
    if bv is None:
        bv = 8 if n < 8 else 64
    if warps is None:
        warps = 1 if n < 8 else 4
    if n:
        with torch.cuda.device(q.device):
            _decode[(n, hv, tr.cdiv(dv, bv))](
                q,
                k,
                v,
                a,
                b,
                alog,
                bias,
                state,
                out,
                ids,
                h,
                hv,
                dk,
                dv,
                q.stride(1),
                k.stride(1),
                v.stride(1),
                a.stride(0),
                b.stride(0),
                dk**-0.5 if scale is None else scale,
                normalize,
                tr.next_power_of_2(dk),
                bv,
                factor,
                fma,
                round_beta,
                num_warps=warps,
                num_stages=1,
                enable_fp_fusion=fma,
            )
    return out, state


@tr.jit
def gdn_w8a16_kernel(
    Q,
    Kp,
    Vp,
    G,
    Beta,
    State,
    Cu,
    O,
    Final,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    SCALE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SINGLE: tl.constexpr,
    WIDE: tl.constexpr,
):
    seq, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, end = tl.load(Cu + seq), tl.load(Cu + seq + 1)
    if WIDE:
        seq = seq.to(tl.int64)
        start, end = start.to(tl.int64), end.to(tl.int64)
    key_idx = tl.arange(0, BK)
    value_idx = tile * BV + tl.arange(0, BV)
    mask = (value_idx[:, None] < DV) & (key_idx[None, :] < DK)
    offset = ((seq * HV + head) * DV + value_idx[:, None]) * DK + key_idx[None, :]
    state = tl.load(State + offset, mask, other=0).to(tl.float32)
    qhead = head // (HV // H)
    if SINGLE and end - start == 1:
        token = start
        query = tl.load(
            Q + token * QS + qhead * DK + key_idx, key_idx < DK, other=0
        ).to(tl.float32)
        key = tl.load(Kp + token * KS + qhead * DK + key_idx, key_idx < DK, other=0).to(
            tl.float32
        )
        value = tl.load(
            Vp + token * VS + head * DV + value_idx, value_idx < DV, other=0
        ).to(tl.float32)
        gate = tl.load(G + token * HV + head).to(tl.float32)
        beta = tl.load(Beta + token * HV + head).to(tl.float32)
        state *= tl.exp(gate)
        delta = (value - tl.sum(state * key[None, :], 1)) * beta
        state += delta[:, None] * key[None, :]
        out = tl.sum(state * query[None, :], 1) * SCALE
        tl.store(O + (token * HV + head) * DV + value_idx, out, value_idx < DV)
    else:
        for token in range(start, end):
            query = tl.load(
                Q + token * QS + qhead * DK + key_idx, key_idx < DK, other=0
            ).to(tl.float32)
            key = tl.load(
                Kp + token * KS + qhead * DK + key_idx, key_idx < DK, other=0
            ).to(tl.float32)
            value = tl.load(
                Vp + token * VS + head * DV + value_idx, value_idx < DV, other=0
            ).to(tl.float32)
            gate = tl.load(G + token * HV + head).to(tl.float32)
            beta = tl.load(Beta + token * HV + head).to(tl.float32)
            state *= tl.exp(gate)
            delta = (value - tl.sum(state * key[None, :], 1)) * beta
            state += delta[:, None] * key[None, :]
            out = tl.sum(state * query[None, :], 1) * SCALE
            tl.store(O + (token * HV + head) * DV + value_idx, out, value_idx < DV)
    tl.store(Final + offset, state, mask)


def gdn_w8a16(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    *,
    decode: bool = False,
    mixed_precision: bool = False,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GDN core forward with precomputed log-decay and sigmoid beta.

    q/k: [1,T,H,K], v: [1,T,HV,V], g/beta: [1,T,HV].
    State is FP32 [num_sequences,HV,V,K]. Packed QKV views are supported.
    W8 refers to the surrounding model weights; no projection is performed here.
    This forward-only implementation leaves input state unchanged. MetaX uses
    a 64-token chunk path for long sequences with 128-dimensional heads;
    other shapes retain the recurrence. Temporary workspace is capped at 256 MiB.
    Default core mode preserves FP32 intermediates. mixed_precision=True opts
    into activation-dtype chunk intermediates/dot operands with FP32 state and
    accumulation. Its error is input-distribution dependent; validate model
    quality before enabling it. Decode always uses FP32 recurrent state.

    With decode=True, q is packed [T,2*H*K+HV*V], or q/k/v are separate.
    a/b are raw BF16/FP16 gates, A_log/dt_bias are FP32 [HV], state_indices
    is an integer [T] vector. Positive indices must be distinct valid slots;
    nonpositive indices are padding. This mode updates initial_state in place
    and returns (out, initial_state), optionally using a preallocated out.
    QK normalization and gate generation are fused into the Triton kernel.
    """
    if decode:
        if any(x is None for x in (a, b, A_log, dt_bias, initial_state, state_indices)):
            raise ValueError("decode requires raw gates, FP32 state and state_indices")
        if (k is None) != (v is None):
            raise ValueError("provide both k/v or packed q")
        return _fused_decode(
            q if k is None else (q, k, v),
            a,
            b,
            A_log,
            dt_bias,
            initial_state,
            state_indices,
            out,
            scale=scale,
            normalize=normalize,
        )
    if any(x is None for x in (k, v, g, beta, initial_state, cu_seqlens)):
        raise ValueError("core mode requires q/k/v, g/beta, state and cu_seqlens")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Expected rank-four q/k/v and matching q/k shapes")
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if (
        batch != 1
        or v.shape[:2] != (1, tokens)
        or heads < 1
        or value_heads < 1
        or value_heads % heads
    ):
        raise ValueError("Expected packed batch=1 and HV divisible by H")
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise NotImplementedError("Only BF16/FP16 activations are supported")
    if (
        any(t.device != q.device for t in (k, v, g, beta, initial_state, cu_seqlens))
        or q.device.type != "cuda"
    ):
        raise ValueError("All inputs must be on the same GPU")
    if any(t.stride(-1) != 1 or t.stride(-2) != t.shape[-1] for t in (q, k, v)):
        raise NotImplementedError("Only contiguous head dimensions are supported")
    if any(
        t.shape != (1, tokens, value_heads) or not t.is_contiguous() for t in (g, beta)
    ):
        raise ValueError("g/beta must be contiguous [1,T,HV]")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise NotImplementedError("g/beta must be FP32")
    if (
        cu_seqlens.ndim != 1
        or not cu_seqlens.is_contiguous()
        or cu_seqlens.numel() < 2
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("cu_seqlens must be a contiguous integer vector")
    sequences = cu_seqlens.numel() - 1
    if (
        initial_state.shape != (sequences, value_heads, value_dim, key_dim)
        or initial_state.dtype != torch.float32
        or not initial_state.is_contiguous()
    ):
        raise ValueError("State must be contiguous FP32 [N,HV,V,K]")
    if key_dim > 256 or value_dim > 256 or key_dim < 1 or value_dim < 1:
        raise NotImplementedError("Head dimensions must be in [1,256]")
    if any(t.requires_grad for t in (q, k, v, g, beta, initial_state)):
        raise NotImplementedError("Only inference forward is supported")
    # cu_seqlens values follow the caller's packed-sequence contract: monotone,
    # starting at zero and ending at T; reading them here would synchronize.
    # MetaX measurements select chunking for long, 128-dimensional heads.
    # Keep other backends, dimensions, and short sequences on the recurrence.
    if (
        runtime.device.vendor_name == "metax"
        and key_dim == value_dim == 128
        and tokens >= (32 if mixed_precision else 128) * sequences
        and sequences <= 64
    ):
        plan = _mixed_workspace_plan if mixed_precision else _fp32_workspace_plan
        run_chunk = _run_mixed_chunk if mixed_precision else _run_fp32_chunk

        if plan(tokens, sequences, value_heads)[1]:
            return run_chunk(
                (q, k, v, g, beta, initial_state, cu_seqlens),
                key_dim**-0.5 if scale is None else scale,
            )
    single = tokens == sequences
    value_block = 32 if single and sequences >= 3 else 16
    output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    final_state = torch.empty(initial_state.shape, device=q.device, dtype=torch.float32)
    gdn_w8a16_kernel[(sequences, value_heads, tr.cdiv(value_dim, value_block))](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        output,
        final_state,
        tokens,
        heads,
        value_heads,
        key_dim,
        value_dim,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        key_dim**-0.5 if scale is None else scale,
        tr.next_power_of_2(key_dim),
        value_block,
        single,
        max(
            tokens * max(q.stride(1), k.stride(1), v.stride(1)),
            initial_state.numel(),
            v.numel(),
        )
        >= 2**31,
        num_warps=4,
        num_stages=1,
    )
    return output, final_state

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""64-token GDN chunks with FP32 workspace and accumulation.

The 16x16 blocked triangular inverse and V-major state layout follow the
MetaX GDN design in FlagGems-vllm PR #771. Unlike that path, this variant
keeps intermediate states/W/U in FP32 and constructs chunk mappings on GPU.
A bounded streaming window reuses temporary storage. Requests are interleaved
when they fit in one window, preserving state-update parallelism.
FP32 dot operands use three-term TF32 decomposition (tf32x3), not TF32.
"""

import torch
import triton as tr
import triton.backends.metax.compiler as backend
import triton.language as tl

_original_translate = backend.metax.translate_llvmir_to_mcfatbin


def _compile_gdn_chunk(src, mxcc, maca, options):
    # Restrict the compiler workaround to this recurrence, as in GDN PR #771.
    if backend.maca_get_kernel_name(src) == "_gdn_chunk_scan":
        options += " -mllvm -metaxgpu-disable-licm=true"
    return _original_translate(src, mxcc, maca, options)


backend.metax.translate_llvmir_to_mcfatbin = _compile_gdn_chunk


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
def _gdn_chunk_output(
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


# Temporary FP32 tensors only; output/final state are separate API results.
_WORKSPACE_LIMIT_BYTES = 256 * 1024**2
_MAX_CHUNKS_PER_WINDOW = 32


def _workspace_plan(tokens, sequences, value_heads):
    # Sum ceil(length / 64) <= floor((T + 63*N) / 64), including empty requests.
    total = (tokens + 63 * sequences) // 64
    # A + QK + GC + W + U + HS; UP aliases U after its last read.
    bytes_per_chunk = value_heads * (2 * 64**2 + 64 + 2 * 64 * 128 + 128**2) * 4
    max_slots = _WORKSPACE_LIMIT_BYTES // bytes_per_chunk
    if not max_slots:
        return total, 0, 0
    capacity = min(total, _MAX_CHUNKS_PER_WINDOW, 1 << (max_slots.bit_length() - 1))
    return total, capacity, capacity * bytes_per_chunk


def gdn_w8a16_chunk(args, scale):
    # Fixed MetaX configuration selected outside graph capture.
    bv, warps, prec = 64, 4, "tf32x3"
    q, k, v, g, b, s, cu = args
    t, h, dk = q.shape[1:]
    hv, dv = v.shape[2:]
    n = cu.numel() - 1
    total, c, _ = _workspace_plan(t, n, hv)
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
        _gdn_chunk_wy[(c, hv, tr.cdiv(max(dk, dv), 32))](
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
        _gdn_chunk_output[(c, hv, tr.cdiv(dv, bv))](
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

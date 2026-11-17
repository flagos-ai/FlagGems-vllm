# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _stage_rhs(b):
    # TF32 needs a K-major B descriptor. Store [N,K] and transpose its descriptor.
    return tle.gpu.alloc(
        (b.shape[1], b.shape[0]), tl.float32, scope=tle.gpu.smem, init_value=tl.trans(b)
    )


@triton.jit
def _issue_wgmma(a, rhs):
    return tle.gpu.wgmma(a, rhs, trans_b=True, input_precision="tf32")


@triton.jit
def _matmul(a, b, WARP_MMA: tl.constexpr = False):
    if WARP_MMA:
        result = tl.dot(a, b, input_precision="tf32")
    else:
        result = tle.gpu.wgmma_wait(0, _issue_wgmma(a, _stage_rhs(b)))
    return result


@triton.jit
def _chunk_summary(
    K,
    V,
    Slope,
    Partial,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    part, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    chunk, tile = part // ((E + BV - 1) // BV), part % ((E + BV - 1) // BV)
    row = tl.arange(0, C)
    t = chunk * C + row
    d = tl.arange(0, BK)
    e = tile * BV + tl.arange(0, BV)
    key = tl.load(
        K + batch * SK[0] + head * SK[1] + t[:, None] * SK[2] + d[None, :] * SK[3],
        (t[:, None] < T) & (d[None, :] < D),
        other=0,
    ).to(tl.float32)
    val = tl.load(
        V + batch * SV[0] + head * SV[1] + t[:, None] * SV[2] + e[None, :] * SV[3],
        (t[:, None] < T) & (e[None, :] < E),
        other=0,
    ).to(tl.float32)
    slope = tl.load(Slope + head * SS).to(tl.float32)
    length = tl.minimum(C, T - chunk * C)
    # Invalid rows must stay zero even for a steep decay on a partial chunk.
    weight = tl.exp(-slope * tl.maximum(length - 1 - row, 0))
    summary = _matmul(tl.trans(key), val * weight[:, None])
    off = ((batch * H + head) * ((T + C - 1) // C) + chunk) * D * E
    tl.store(
        Partial + off + d[:, None] * E + e[None, :],
        summary,
        (d[:, None] < D) & (e[None, :] < E),
    )


@triton.jit
def _scan_chunks(
    Partial,
    Initial,
    Slope,
    History,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    SI: tl.constexpr,
    SS: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tile // ((E + BV - 1) // BV) * BK + tl.arange(0, BK)
    e = tile % ((E + BV - 1) // BV) * BV + tl.arange(0, BV)
    mask = (d[:, None] < D) & (e[None, :] < E)
    elem = d[:, None] * E + e[None, :]
    bh = batch * H + head
    chunks: tl.constexpr = (T + C - 1) // C
    snapshots: tl.constexpr = ((T + 255) // 256) + 1
    prefix = Partial + bh * chunks * D * E + elem
    hist = History + bh * snapshots * D * E + elem
    if HAS_INITIAL:
        state = tl.load(
            Initial
            + batch * SI[0]
            + head * SI[1]
            + d[:, None] * SI[2]
            + e[None, :] * SI[3],
            mask,
            0,
        )
    else:
        state = tl.full((BK, BV), 0, tl.float32)
    slope = tl.load(Slope + head * SS)
    boundary_state = state
    decay = tl.exp(-slope)
    for i in range(chunks):
        summary = tl.load(prefix + i * D * E, mask, 0)
        pos = (i % (256 // C)) * C
        if pos == 0:
            boundary_state = state
            carry = state
            tl.store(hist + (i // (256 // C)) * D * E, state, mask)
        else:
            # The vLLM prefix read uses exponent r (first token r=0)
            # for the previous 256-token block only. History itself is canonical.
            carry = (
                decay * state + ((1.0 - decay) * tl.exp(-slope * pos)) * boundary_state
            )
        tl.store(prefix + i * D * E, carry, mask)
        state = state * tl.exp(-slope * tl.minimum(C, T - i * C)) + summary
    tl.store(hist + (snapshots - 1) * D * E, state, mask)


@triton.jit
def _compose_decay(left_a, left_b, right_a, right_b):
    return right_a * left_a, right_a * left_b + right_b


@triton.jit
def _scan_parallel(
    Partial,
    Initial,
    Slope,
    History,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    SI: tl.constexpr,
    SS: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    WIDTH: tl.constexpr,
):
    # Each row is a (decay, KV update) affine map. Scan time across lanes so
    # one CTA issues a matrix of loads instead of serial loads for each chunk.
    col = tl.program_id(0) * WIDTH + tl.arange(0, WIDTH)
    head, batch = tl.program_id(1), tl.program_id(2)
    bh = batch * H + head
    chunks: tl.constexpr = (T + C - 1) // C
    snapshots: tl.constexpr = (T + 255) // 256 + 1
    prefix = Partial + bh * chunks * D * E
    hist = History + bh * snapshots * D * E
    if HAS_INITIAL:
        base = tl.load(
            Initial + batch * SI[0] + head * SI[1] + col // E * SI[2] + col % E * SI[3],
            col < D * E,
            0,
        ).to(tl.float32)
    else:
        base = tl.full((WIDTH,), 0, tl.float32)
    slope = tl.load(Slope + head * SS)
    decay = tl.exp(-slope)
    row = tl.arange(0, L)
    prev_idx = tl.broadcast_to(tl.maximum(row - 1, 0)[:, None], (L, WIDTH))
    boundary_idx = tl.broadcast_to(
        ((row // (256 // C)) * (256 // C))[:, None], (L, WIDTH)
    )
    for start in range((chunks + L - 1) // L):
        i = start * L + row
        ptr = prefix + i[:, None] * D * E + col[None, :]
        mask = (i[:, None] < chunks) & (col[None, :] < D * E)
        summary = tl.load(ptr, mask, 0)
        length = tl.minimum(C, tl.maximum(0, T - i * C))
        a = tl.broadcast_to(tl.exp(-slope * length)[:, None], (L, WIDTH))
        total_a, total_b = tl.associative_scan((a, summary), 0, _compose_decay)
        after = total_b + total_a * base[None, :]
        before = tl.where(
            row[:, None] == 0, base[None, :], tl.gather(after, prev_idx, 0)
        )
        pos = (row % (256 // C)) * C
        boundary = tl.gather(before, boundary_idx, 0)
        carry = (
            decay * before + ((1.0 - decay) * tl.exp(-slope * pos))[:, None] * boundary
        )
        tl.store(ptr, tl.where(pos[:, None] == 0, before, carry), mask)
        dest = hist + (i // (256 // C))[:, None] * D * E + col[None, :]
        tl.store(dest, before, mask & (pos[:, None] == 0))
        base = tl.sum(tl.where(row[:, None] == L - 1, after, 0), 0)
    tl.store(hist + (snapshots - 1) * D * E + col, base, col < D * E)


@triton.jit
def _chunk_output(
    Q,
    K,
    V,
    Slope,
    Prefix,
    Output,
    History,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SI: tl.constexpr,
    SS: tl.constexpr,
    SINGLE: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    OVERLAP_Q: tl.constexpr,
):
    part, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    chunk, tile = part // ((E + BV - 1) // BV), part % ((E + BV - 1) // BV)
    row = tl.arange(0, C)
    t = chunk * C + row
    d = tl.arange(0, BK)
    e = tile * BV + tl.arange(0, BV)
    bh = batch * H + head
    mask = (d[:, None] < D) & (e[None, :] < E)
    elem = d[:, None] * E + e[None, :]
    q = tl.load(
        Q + batch * SQ[0] + head * SQ[1] + t[:, None] * SQ[2] + d[None, :] * SQ[3],
        (t[:, None] < T) & (d[None, :] < D),
        0,
    ).to(tl.float32)
    k = tl.load(
        K + batch * SK[0] + head * SK[1] + t[:, None] * SK[2] + d[None, :] * SK[3],
        (t[:, None] < T) & (d[None, :] < D),
        0,
    ).to(tl.float32)
    v = tl.load(
        V + batch * SV[0] + head * SV[1] + t[:, None] * SV[2] + e[None, :] * SV[3],
        (t[:, None] < T) & (e[None, :] < E),
        0,
    ).to(tl.float32)
    slope = tl.load(Slope + head * SS)
    if SINGLE:
        if HAS_INITIAL:
            state = tl.load(
                Prefix
                + batch * SI[0]
                + head * SI[1]
                + d[:, None] * SI[2]
                + e[None, :] * SI[3],
                mask,
                0,
            )
        else:
            state = tl.full((BK, BV), 0, tl.float32)
        hist = History + bh * 2 * D * E + elem
        tl.store(hist, state, mask)
        weight = tl.exp(-slope * tl.maximum(T - 1 - row, 0))
        final = (
            _matmul(tl.trans(k), v * weight[:, None], C < 64)
            + tl.exp(-slope * T) * state
        )
        tl.store(hist + D * E, final, mask)
    else:
        state = tl.load(
            Prefix + (bh * ((T + C - 1) // C) + chunk) * D * E + elem, mask, 0
        )
    if OVERLAP_Q:
        # Materialize both descriptors before issuing either product. Keep each
        # RHS intact until its group has completed.
        keys = _stage_rhs(tl.trans(k))
        history_rhs = _stage_rhs(state)
        score_acc = _issue_wgmma(q, keys)
        cross_acc = _issue_wgmma(q, history_rhs)
    age = row[:, None] - row[None, :]
    weight = tl.exp(-slope * tl.maximum(age, 0))
    if OVERLAP_Q:
        scores = tle.gpu.wgmma_wait(1, score_acc)
    else:
        scores = _matmul(q, tl.trans(k), C < 64)
    causal = tl.where(age >= 0, weight * scores, 0)
    if OVERLAP_Q:
        # Drain the remaining group before staging V into shared memory.
        carry = tle.gpu.wgmma_wait(0, cross_acc) * tl.exp(-slope * row)[:, None]
    y = _matmul(causal, v, C < 64)
    if OVERLAP_Q:
        y += carry
    elif not SINGLE or HAS_INITIAL:
        y += _matmul(q, state, C < 64) * tl.exp(-slope * row)[:, None]
    tl.store(
        Output + (bh * T + t[:, None]) * E + e[None, :],
        y,
        (t[:, None] < T) & (e[None, :] < E),
    )


@triton.jit
def _recurrent(
    Q,
    K,
    V,
    Slope,
    Slots,
    Initial,
    Output,
    History,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SI: tl.constexpr,
    SS: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    DECODE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    e = tile * BV + tl.arange(0, BV)
    d = tl.arange(0, BK)
    out = Output + (batch * H + head) * T * E + e
    if DECODE:
        slot = tl.load(Slots + batch * SLOT_STRIDE).to(tl.int64)
        if slot == -1:
            tl.store(out, 0, e < E)
            return
    else:
        slot = batch
    mask = (d[:, None] < D) & (e[None, :] < E)
    offsets = tle.gpu.set_layout(d[:, None] * SI[2] + e[None, :] * SI[3], LAYOUT)
    base = Initial + slot * SI[0] + head * SI[1] + offsets
    if HAS_INITIAL:
        state = tl.load(base, mask, 0).to(tl.float32)
    else:
        state = tl.full((BK, BV), 0, tl.float32)
    if not DECODE:
        count: tl.constexpr = ((T + 255) // 256) + 1
        hist = (
            History + (batch * H + head) * count * D * E + d[:, None] * E + e[None, :]
        )
    decay = tl.exp(-tl.load(Slope + head * SS))
    boundary_state = state
    correction = 1.0 - decay
    for i in range(T):
        if not DECODE:
            if i % 256 == 0:
                boundary_state = state
                correction = 1.0 - decay
                tl.store(hist + i // 256 * D * E, state, mask)
        q = tl.load(
            Q + batch * SQ[0] + head * SQ[1] + i * SQ[2] + d * SQ[3], d < D, 0
        ).to(tl.float32)
        k = tl.load(
            K + batch * SK[0] + head * SK[1] + i * SK[2] + d * SK[3], d < D, 0
        ).to(tl.float32)
        v = tl.load(
            V + batch * SV[0] + head * SV[1] + i * SV[2] + e * SV[3], e < E, 0
        ).to(tl.float32)
        state = decay * state + k[:, None] * v[None, :]
        if DECODE:
            read_state = state
        else:
            read_state = state + correction * boundary_state
            correction *= decay
        tl.store(out + i * E, tl.sum(q[:, None] * read_state, axis=0), e < E)
    if DECODE:
        tl.store(base, state, mask)
    else:
        tl.store(hist + (count - 1) * D * E, state, mask)


_CACHE_LAYOUT = tle.gpu.BlockEncoding([1, 2], [4, 8], [4, 1], [1, 0])
_SMALL_LAYOUT = tle.gpu.BlockEncoding([1, 2], [4, 8], [1, 1], [1, 0])
_NARROW_LAYOUT = tle.gpu.BlockEncoding([1, 2], [4, 8], [2, 1], [1, 0])
_TRANSPOSED_LAYOUT = tle.gpu.BlockEncoding([2, 1], [16, 2], [1, 1], [0, 1])
_WIDE_LAYOUT = tle.gpu.BlockEncoding([1, 4], [4, 8], [4, 1], [1, 0])


def _check_qkv(q, k, v, slope):
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3] or v.ndim != 4:
        raise ValueError("expected Q/K [B,H,T,D] and V [B,H,T,E]")
    if q.shape[-1] not in (64, 128) or v.shape[-1] < 1:
        raise ValueError("Hopper Lightning Attention requires D=64 or 128 and E>0")
    if (
        q.dtype not in (torch.float32, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("Q, K and V must share a float32 or bfloat16 dtype")
    if slope.numel() != q.shape[1] or slope.dtype != torch.float32:
        raise ValueError("slope must contain one FP32 value per head")


def lightning_attention(q, k, v, ed, block_size=256, kv_history=None):
    """Return [B,H,T,E] output and FP32 states before each 256-token block plus the tail.

    As in vLLM 0.24, the history interval is 256 independently of block_size.
    All inputs are on the same Hopper GPU. The optional initial state is read-only.
    """
    slope, initial = ed, kv_history
    _check_qkv(q, k, v, slope)
    b, h, t, d = q.shape
    e = v.shape[-1]
    if initial is not None and (
        initial.shape != (b, h, d, e) or initial.dtype != torch.float32
    ):
        raise ValueError("kv_history must have shape [B,H,D,E] and dtype float32")
    c = 64
    warps = 4
    if t <= 32:
        c = 16 if t <= 16 else 32
        warps = 2
    elif 64 < t <= 128 and b * h >= 16:
        c = 128
    n = triton.cdiv(t, c)
    if t <= 128:
        value_tile = 32 if c == 128 else 16
    elif t <= 256 and d == 64:
        value_tile = 16
    else:
        value_tile = 64 if e >= 128 and b * h * n > 64 else 32
    # Overlapping the two Q products helps short, single-tile prefixes with
    # relatively few heads. Larger launches run faster with immediate waits.
    overlap_q = (
        c == 64
        and n == 1
        and d == 128
        and e >= 32
        and b * h <= 16
        and initial is not None
        and k.stride(3) == 1
    )
    y = torch.empty((b, h, t, e), dtype=q.dtype, device=q.device)
    hist = torch.empty(
        (b, h, triton.cdiv(t, 256) + 1, d, e), dtype=torch.float32, device=q.device
    )
    si = initial.stride() if initial is not None else (h * d * e, d * e, e, 1)
    base = initial if initial is not None else q
    sv = (q.stride(), k.stride(), v.stride())
    ss = slope.view(-1).stride(0)
    bk = max(64, triton.next_power_of_2(d))
    tiles = triton.cdiv(e, value_tile)
    with torch.cuda.device(q.device):
        if t <= 1 or (t <= 4 and d == 64):
            _recurrent[(tiles, h, b)](
                q,
                k,
                v,
                slope,
                q,
                base,
                y,
                hist,
                h,
                t,
                d,
                e,
                *sv,
                si,
                ss,
                0,
                initial is not None,
                False,
                bk,
                value_tile,
                _CACHE_LAYOUT,
                num_warps=4,
            )
        else:
            prefix = base
            if n != 1:
                prefix = torch.empty(
                    (b, h, n, d, e), dtype=torch.float32, device=q.device
                )
                _chunk_summary[(n * tiles, h, b)](
                    k,
                    v,
                    slope,
                    prefix,
                    h,
                    t,
                    d,
                    e,
                    sv[1],
                    sv[2],
                    ss,
                    c,
                    bk,
                    value_tile,
                    num_warps=4,
                )
                if n >= 24:
                    width = 256 if n <= 128 and (d * e >= 16384 or b * h >= 16) else 64
                    _scan_parallel[(triton.cdiv(d * e, width), h, b)](
                        prefix,
                        base,
                        slope,
                        hist,
                        h,
                        t,
                        d,
                        e,
                        si,
                        ss,
                        initial is not None,
                        c,
                        16,
                        width,
                        num_warps=4,
                    )
                else:
                    _scan_chunks[(triton.cdiv(d, 16) * triton.cdiv(e, 32), h, b)](
                        prefix,
                        base,
                        slope,
                        hist,
                        h,
                        t,
                        d,
                        e,
                        si,
                        ss,
                        initial is not None,
                        c,
                        16,
                        32,
                    )
            _chunk_output[(n * tiles, h, b)](
                q,
                k,
                v,
                slope,
                prefix,
                y,
                hist,
                h,
                t,
                d,
                e,
                *sv,
                si,
                ss,
                n == 1,
                initial is not None,
                c,
                bk,
                value_tile,
                overlap_q,
                num_warps=warps,
            )
    return y, hist


def lightning_attention_decode(q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE=32):
    """Update active cache slots in-place and return contiguous [B,H*D]."""
    _check_qkv(q, k, v, slope_rate)
    b, h, t, d = q.shape
    if (
        t != 1
        or v.shape[-1] != d
        or kv_caches.ndim != 4
        or kv_caches.shape[1:] != (h, d, d)
    ):
        raise ValueError(
            "decode expects one token, square state and [slots,H,D,D] cache"
        )
    if (
        kv_caches.dtype != torch.float32
        or slot_idx.shape != (b,)
        or slot_idx.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "decode expects an FP32 cache and one integer slot per batch row"
        )
    if BLOCK_SIZE < 1 or BLOCK_SIZE & (BLOCK_SIZE - 1):
        raise ValueError("BLOCK_SIZE must be a positive power of two")
    if kv_caches.stride(2) == 1:
        tile = min(BLOCK_SIZE, 8 if d == 64 else 4)
        warps, layout = 1, _TRANSPOSED_LAYOUT
    elif kv_caches.stride(3) != 1:
        tile, warps, layout = min(BLOCK_SIZE, 16), 2, _NARROW_LAYOUT
    elif d == 64:
        tile, warps, layout = min(BLOCK_SIZE, 32), 1, _SMALL_LAYOUT
    elif b * h <= 128:
        tile, warps, layout = min(BLOCK_SIZE, 16), 2, _NARROW_LAYOUT
    else:
        tile, warps, layout = min(2 * BLOCK_SIZE, 64), 4, _WIDE_LAYOUT
    out = torch.empty((b, h * d), dtype=q.dtype, device=q.device)
    with torch.cuda.device(q.device):
        _recurrent[(triton.cdiv(d, tile), h, b)](
            q,
            k,
            v,
            slope_rate,
            slot_idx,
            kv_caches,
            out,
            kv_caches,
            h,
            1,
            d,
            d,
            q.stride(),
            k.stride(),
            v.stride(),
            kv_caches.stride(),
            slope_rate.view(-1).stride(0),
            slot_idx.stride(0),
            True,
            True,
            d,
            tile,
            layout,
            num_warps=warps,
        )
    return out

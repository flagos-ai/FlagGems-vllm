# SPDX-License-Identifier: Apache-2.0
"""Single-launch Triton kernels for gfx936 Normal BF16 MegaMHC."""

import triton
import triton.language as tl


@triton.jit
def _sinkhorn_reciprocal(x):
    # The denominators in Sinkhorn are positive and normal.
    estimate = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )
    return estimate * tl.fma(-x, estimate, 2.0)


@triton.jit
def _load_post_coefficients(Post, Comb, stream):
    return (
        tl.load(Post + stream),
        tl.load(Comb + stream),
        tl.load(Comb + stream + 4),
        tl.load(Comb + stream + 8),
        tl.load(Comb + stream + 12),
    )


@triton.jit
def _post_from_coeffs(x, r0, r1, r2, r3, coeff):
    p, c0, c1, c2, c3 = coeff
    a = x * p
    a = tl.fma(r0, c0, a)
    a = tl.fma(r1, c1, a)
    a = tl.fma(r2, c2, a)
    return tl.fma(r3, c3, a).to(tl.bfloat16)


@triton.jit
def _post_value(x, r0, r1, r2, r3, Post, Comb, stream):
    return _post_from_coeffs(
        x, r0, r1, r2, r3, _load_post_coefficients(Post, Comb, stream)
    )


@triton.jit
def _load_inputs(X, Residual, h, H: tl.constexpr, valid):
    x = tl.load(X + h, valid, other=0).to(tl.float32)
    r0 = tl.load(Residual + h, valid, other=0).to(tl.float32)
    r1 = tl.load(Residual + H + h, valid, other=0).to(tl.float32)
    r2 = tl.load(Residual + 2 * H + h, valid, other=0).to(tl.float32)
    r3 = tl.load(Residual + 3 * H + h, valid, other=0).to(tl.float32)
    return x, r0, r1, r2, r3


@triton.jit
def _sinkhorn(logits, BLOCK_M: tl.constexpr, EPS: tl.constexpr, ROUNDS: tl.constexpr):
    logits = tl.reshape(logits, (BLOCK_M, 4, 4))
    comb = tl.exp(logits - tl.max(logits, 2)[:, :, None])
    comb = comb * _sinkhorn_reciprocal(tl.sum(comb, 2))[:, :, None] + EPS
    comb = comb * _sinkhorn_reciprocal(tl.sum(comb, 1) + EPS)[:, None, :]
    for _ in range(ROUNDS - 1):
        comb = comb * _sinkhorn_reciprocal(tl.sum(comb, 2) + EPS)[:, :, None]
        comb = comb * _sinkhorn_reciprocal(tl.sum(comb, 1) + EPS)[:, None, :]
    return tl.reshape(comb, (BLOCK_M, 16))


@triton.jit
def _decode_sinkhorn(
    logits,
    EPS: tl.constexpr,
    ROUNDS: tl.constexpr,
    CHECK_EVERY: tl.constexpr,
    UNROLL: tl.constexpr,
):
    logits = tl.reshape(logits, (1, 4, 4))
    comb = tl.exp(logits - tl.max(logits, 2)[:, :, None])
    comb = comb * _sinkhorn_reciprocal(tl.sum(comb, 2))[:, :, None] + EPS
    comb = comb * _sinkhorn_reciprocal(tl.sum(comb, 1) + EPS)[:, None, :]
    fixed = False
    for i in tl.range(ROUNDS - 1, loop_unroll_factor=UNROLL):
        if not fixed:
            previous = comb
            comb = comb * _sinkhorn_reciprocal(tl.sum(comb, 2) + EPS)[:, :, None]
            comb = comb * _sinkhorn_reciprocal(tl.sum(comb, 1) + EPS)[:, None, :]
            # A full iteration reproduced these FP32 values. All further
            # deterministic iterations would return the same matrix.
            if i % CHECK_EVERY == CHECK_EVERY - 1:
                fixed = tl.sum((previous != comb).to(tl.int32)) == 0
    return tl.reshape(comb, (1, 16))


@triton.jit
def _mix(
    acc,
    square_sum,
    Scale,
    Base,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    RMS_EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    POST_SCALE: tl.constexpr,
    SINKHORN_ROUNDS: tl.constexpr,
):
    rms = tl.rsqrt(square_sum / (4 * H) + RMS_EPS)
    streams = tl.arange(0, 4)
    streams = tl.broadcast_to(streams[None, :], (BLOCK_M, 4))
    pre = tl.gather(acc, streams, axis=1) * rms[:, None]
    pre = tl.fma(pre, tl.load(Scale), tl.load(Base + streams))
    pre = tl.sigmoid(pre) + PRE_EPS
    post = tl.gather(acc, streams + 4, axis=1) * rms[:, None]
    post = tl.fma(post, tl.load(Scale + 1), tl.load(Base + streams + 4))
    post = tl.sigmoid(post) * POST_SCALE

    pairs = tl.arange(0, 16)
    pairs = tl.broadcast_to(pairs[None, :], (BLOCK_M, 16))
    logits = tl.gather(acc, pairs + 8, axis=1) * rms[:, None]
    logits = tl.fma(logits, tl.load(Scale + 2), tl.load(Base + pairs + 8))
    return pre, post, _sinkhorn(logits, BLOCK_M, SINKHORN_EPS, SINKHORN_ROUNDS)


@triton.jit
def _collapse(
    Updated, h, valid, pre, H: tl.constexpr, BLOCK_M: tl.constexpr, CHUNK: tl.constexpr
):
    collapsed = tl.full((BLOCK_M, CHUNK), 0, tl.float32)
    for stream in tl.static_range(4):
        idx = tl.full((BLOCK_M, 1), stream, tl.int32)
        mix = tl.gather(pre, idx, axis=1)
        residual = tl.load(Updated + stream * H + h, valid, other=0).to(tl.float32)
        collapsed = tl.fma(mix, residual, collapsed)
    return collapsed.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _normalize_rows(
    Updated,
    Y,
    Norm,
    pre,
    tile,
    Tokens,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NORM_ROWS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    h = tl.arange(0, H)[None, :]
    weight = tl.load(Norm + h).to(tl.float32)
    for group in range(BLOCK_M // NORM_ROWS):
        local = group * NORM_ROWS + tl.arange(0, NORM_ROWS)
        rows = tile * BLOCK_M + local
        indices = tl.broadcast_to(local[:, None], (NORM_ROWS, 4))
        mix = tl.gather(pre, indices, axis=0)
        active = rows[:, None] < Tokens
        residual = Updated + rows[:, None] * (4 * H)
        value = _collapse(residual, h, active, mix, H, NORM_ROWS, H)
        rms = tl.rsqrt(tl.sum(value * value, 1) / H + NORM_EPS)
        tl.store(Y + rows[:, None] * H + h, value * rms[:, None] * weight, active)


@triton.jit
def mega_mhc_tile_kernel(
    X,
    Residual,
    Post,
    Comb,
    Weight,
    Scale,
    Base,
    Norm,
    Updated,
    NextPost,
    NextComb,
    Y,
    Tokens,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    POST_CHUNK: tl.constexpr,
    PANEL: tl.constexpr,
    K_PARTITIONS: tl.constexpr,
    NORM_ROWS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    POST_SCALE: tl.constexpr,
    SINKHORN_ROUNDS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    U0, Y0 = Updated, Y
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = rows[:, None] < Tokens
    X = X + rows[:, None] * H
    Residual = Residual + rows[:, None] * 4 * H
    Updated = Updated + rows[:, None] * 4 * H
    safe_rows = tl.minimum(rows[:, None], Tokens - 1)
    post_ptr = Post + safe_rows * 4
    comb_ptr = Comb + safe_rows * 16
    coeff = (
        _load_post_coefficients(post_ptr, comb_ptr, 0),
        _load_post_coefficients(post_ptr, comb_ptr, 1),
        _load_post_coefficients(post_ptr, comb_ptr, 2),
        _load_post_coefficients(post_ptr, comb_ptr, 3),
    )
    heads = tl.arange(0, 32)
    k = tl.arange(0, BLOCK_K)
    h = tl.arange(0, POST_CHUNK)[None, :]
    squares = tl.full((BLOCK_M, POST_CHUNK), 0, tl.float32)

    if K_PARTITIONS == 1:
        acc = tl.full((BLOCK_M, 32), 0, tl.float32)
    else:
        partial = tl.full((K_PARTITIONS, BLOCK_M, 32), 0, tl.float32)
        partitions_per_stream: tl.constexpr = K_PARTITIONS // 4
        span: tl.constexpr = PANEL // partitions_per_stream
        part = tl.arange(0, K_PARTITIONS)
        initial = (
            part // partitions_per_stream * H + part % partitions_per_stream * span
        )[:, None] + k[None, :]

    for panel in range(H // PANEL):
        start = panel * PANEL
        # Consume each Post panel while its output is still recent in cache.
        for block in range(PANEL // POST_CHUNK):
            pos = start + block * POST_CHUNK + h
            x, r0, r1, r2, r3 = _load_inputs(X, Residual, pos, H, valid)
            for stream in tl.static_range(4):
                value = _post_from_coeffs(x, r0, r1, r2, r3, coeff[stream])
                tl.store(Updated + stream * H + pos, value, valid)
                f32 = value.to(tl.float32)
                squares = tl.fma(f32, f32, squares)
        tl.debug_barrier()

        # Interleave the two BF16 weight components along K and accumulate
        # both in FP32. The partitioned case assigns different K spans to waves.
        if K_PARTITIONS == 1:
            blocks_per_stream: tl.constexpr = PANEL // BLOCK_K
            for block in tl.range(4 * PANEL // BLOCK_K, num_stages=2):
                offset = (
                    start
                    + (block % blocks_per_stream) * BLOCK_K
                    + (block // blocks_per_stream) * H
                    + k
                )
                value = tl.load(Updated + offset[None, :], valid, other=0)
                ptr = Weight + heads[None, :] * (4 * H) + offset[:, None]
                weight = tl.load(ptr, heads[None, :] < 24, other=0)
                high = weight.to(tl.bfloat16)
                low = (weight - high.to(tl.float32)).to(tl.bfloat16)
                a = tl.reshape(tl.join(value, value), (BLOCK_M, 2 * BLOCK_K))
                b = tl.reshape(
                    tl.permute(tl.join(high, low), (0, 2, 1)), (2 * BLOCK_K, 32)
                )
                acc = tl.dot(a, b, acc)
        else:
            for block in tl.range(
                span // BLOCK_K, num_stages=1 if K_PARTITIONS == 8 else 2
            ):
                off = initial + start + block * BLOCK_K
                a = tl.load(
                    Updated[None, :, :] + off[:, None, :], valid[None, :, :], other=0
                )
                ptr = Weight + heads[None, None, :] * (4 * H) + off[:, :, None]
                w = tl.load(ptr, heads[None, None, :] < 24, other=0)
                wh = w.to(tl.bfloat16)
                wl = (w - wh.to(tl.float32)).to(tl.bfloat16)
                packed_a = tl.reshape(
                    tl.join(a, a), (K_PARTITIONS, BLOCK_M, 2 * BLOCK_K)
                )
                packed_w = tl.reshape(
                    tl.permute(tl.join(wh, wl), (0, 1, 3, 2)),
                    (K_PARTITIONS, 2 * BLOCK_K, 32),
                )
                partial = tl.dot(packed_a, packed_w, partial)
    if K_PARTITIONS != 1:
        acc = tl.sum(partial, 0)
    pre, post, comb = _mix(
        acc,
        tl.sum(squares, 1),
        Scale,
        Base,
        H,
        BLOCK_M,
        RMS_EPS,
        PRE_EPS,
        SINKHORN_EPS,
        POST_SCALE,
        SINKHORN_ROUNDS,
    )
    tl.store(NextPost + rows[:, None] * 4 + tl.arange(0, 4)[None, :], post, valid)
    tl.store(NextComb + rows[:, None] * 16 + tl.arange(0, 16)[None, :], comb, valid)

    _normalize_rows(
        U0, Y0, Norm, pre, tl.program_id(0), Tokens, H, BLOCK_M, NORM_ROWS, NORM_EPS
    )


@triton.jit
def _project_group(
    X,
    Residual,
    Post,
    Comb,
    Weight,
    Updated,
    row,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    UNROLL: tl.constexpr,
    FIRST_HEAD: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
):
    k = tl.arange(0, BLOCK_K)
    heads = FIRST_HEAD + tl.arange(0, HEAD_COUNT)
    sum_sq = tl.full((1, BLOCK_K), 0, tl.float32)
    dot = tl.full((HEAD_COUNT, BLOCK_K), 0, tl.float32)
    for block in tl.range(H // BLOCK_K, loop_unroll_factor=UNROLL):
        h = block * BLOCK_K + k
        x, r0, r1, r2, r3 = _load_inputs(
            X + row * H,
            Residual + row * (4 * H),
            h[None, :],
            H,
            True,
        )
        for stream in tl.static_range(4):
            value = _post_value(
                x, r0, r1, r2, r3, Post + row * 4, Comb + row * 16, stream
            )
            if FIRST_HEAD == 0:
                tl.store(Updated + (row * 4 + stream) * H + h[None, :], value)
            f32 = value.to(tl.float32)
            sum_sq = tl.fma(f32, f32, sum_sq)
            w = tl.load(Weight + heads[:, None] * (4 * H) + stream * H + h[None, :])
            dot = tl.fma(w, f32, dot)
    return tl.sum(dot, 1)[None, :], tl.sum(sum_sq, 1)


@triton.jit
def mega_mhc_row_kernel(
    X,
    Residual,
    Post,
    Comb,
    Weight,
    Scale,
    Base,
    Norm,
    Updated,
    NextPost,
    NextComb,
    Y,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    UNROLL: tl.constexpr,
    RMS_EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    POST_SCALE: tl.constexpr,
    SINKHORN_ROUNDS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    row = tl.program_id(0)
    # These workgroups own disjoint outputs. Each computes the BF16
    # post values it needs for its own projection heads.
    if tl.program_id(1) == 0:
        acc, square_sum = _project_group(
            X,
            Residual,
            Post,
            Comb,
            Weight,
            Updated,
            row,
            H,
            BLOCK_K,
            UNROLL,
            0,
            8,
        )
        rms = tl.rsqrt(square_sum / (4 * H) + RMS_EPS)
        streams = tl.arange(0, 4)[None, :]
        pre = tl.gather(acc, streams, axis=1) * rms[:, None]
        pre = tl.sigmoid(tl.fma(pre, tl.load(Scale), tl.load(Base + streams))) + PRE_EPS
        post = tl.gather(acc, streams + 4, axis=1) * rms[:, None]
        post = tl.fma(post, tl.load(Scale + 1), tl.load(Base + streams + 4))
        tl.store(NextPost + row * 4 + streams, tl.sigmoid(post) * POST_SCALE)
        tl.debug_barrier()
        h = tl.arange(0, H)[None, :]
        value = _collapse(Updated + row * 4 * H, h, True, pre, H, 1, H)
        norm = tl.rsqrt(tl.sum(value * value, 1) / H + NORM_EPS)
        weight = tl.load(Norm + h).to(tl.float32)
        tl.store(Y + row * H + h, value * norm[:, None] * weight)
    else:
        comb_acc, square_sum = _project_group(
            X,
            Residual,
            Post,
            Comb,
            Weight,
            Updated,
            row,
            H,
            BLOCK_K,
            UNROLL,
            8,
            16,
        )
        rms = tl.rsqrt(square_sum / (4 * H) + RMS_EPS)
        pairs = tl.arange(0, 16)[None, :]
        logits = tl.fma(
            comb_acc * rms[:, None], tl.load(Scale + 2), tl.load(Base + pairs + 8)
        )
        comb = _sinkhorn(logits, 1, SINKHORN_EPS, SINKHORN_ROUNDS)
        tl.store(NextComb + row * 16 + pairs, comb)


@triton.jit
def _accumulate_group(
    X,
    Residual,
    Post,
    Comb,
    Weight,
    Updated,
    rows,
    active,
    H: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    FIRST: tl.constexpr,
    HEADS: tl.constexpr,
):
    h = tl.arange(0, K)
    heads = FIRST + tl.arange(0, HEADS)
    sq = tl.full((B, K), 0, tl.float32)
    acc = tl.full((HEADS, B, K), 0, tl.float32)
    p = Post + rows[:, None] * 4
    c = Comb + rows[:, None] * 16
    coeffs = (
        _load_post_coefficients(p, c, 0),
        _load_post_coefficients(p, c, 1),
        _load_post_coefficients(p, c, 2),
        _load_post_coefficients(p, c, 3),
    )
    x_ptr = X + rows[:, None] * H
    r_ptr = Residual + rows[:, None] * (4 * H)
    u_ptr = Updated + rows[:, None] * (4 * H)
    for chunk in range(H // K):
        pos = chunk * K + h
        x, r0, r1, r2, r3 = _load_inputs(x_ptr, r_ptr, pos[None, :], H, active)
        for stream in tl.static_range(4):
            value = _post_from_coeffs(x, r0, r1, r2, r3, coeffs[stream])
            if FIRST == 0:
                tl.store(u_ptr + stream * H + pos[None, :], value, active)
            v = value.to(tl.float32)
            sq = tl.fma(v, v, sq)
            w = tl.load(Weight + heads[:, None] * (4 * H) + stream * H + pos[None, :])
            acc = tl.fma(w[:, None, :], v[None, :, :], acc)
    return tl.trans(tl.sum(acc, 2)), tl.sum(sq, 1)


@triton.jit
def mega_mhc_batch_kernel(
    X,
    Residual,
    Post,
    Comb,
    Weight,
    Scale,
    Base,
    Norm,
    Updated,
    NextPost,
    NextComb,
    Y,
    Tokens,
    H: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    NORM_ROWS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    POST_SCALE: tl.constexpr,
    SINKHORN_ROUNDS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * B + tl.arange(0, B)
    active = rows[:, None] < Tokens
    safe = tl.minimum(rows, Tokens - 1)
    # Reuse each weight tile across several tokens and keep the two output groups independent.
    if tl.program_id(1) == 0:
        gate, sq = _accumulate_group(
            X,
            Residual,
            Post,
            Comb,
            Weight,
            Updated,
            safe,
            active,
            H,
            B,
            K,
            0,
            8,
        )
        rms = tl.rsqrt(sq / (4 * H) + RMS_EPS)
        p = tl.broadcast_to(tl.arange(0, 4)[None, :], (B, 4))
        pre = tl.gather(gate, p, axis=1) * rms[:, None]
        pre = tl.sigmoid(tl.fma(pre, tl.load(Scale), tl.load(Base + p))) + PRE_EPS
        post = tl.gather(gate, p + 4, axis=1) * rms[:, None]
        post = tl.fma(post, tl.load(Scale + 1), tl.load(Base + p + 4))
        tl.store(
            NextPost + rows[:, None] * 4 + p, tl.sigmoid(post) * POST_SCALE, active
        )
        tl.debug_barrier()
        _normalize_rows(Updated, Y, Norm, pre, tile, Tokens, H, B, NORM_ROWS, NORM_EPS)
    else:
        mix, sq = _accumulate_group(
            X,
            Residual,
            Post,
            Comb,
            Weight,
            Updated,
            safe,
            active,
            H,
            B,
            K,
            8,
            16,
        )
        scale = tl.rsqrt(sq / (4 * H) + RMS_EPS)
        pairs = tl.arange(0, 16)[None, :]
        logits = tl.fma(
            mix * scale[:, None], tl.load(Scale + 2), tl.load(Base + pairs + 8)
        )
        comb = _sinkhorn(logits, B, SINKHORN_EPS, SINKHORN_ROUNDS)
        tl.store(NextComb + rows[:, None] * 16 + pairs, comb, active)


@triton.jit
def _pack_mhc_weights(
    Weight,
    Packed,
    Arrivals,
    K: tl.constexpr,
    ARRIVALS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = offsets // K
    weight = tl.load(Weight + offsets, head < 24, other=0)
    high = weight.to(tl.bfloat16)
    low = (weight - high.to(tl.float32)).to(tl.bfloat16)
    tl.store(Packed + 2 * offsets, high)
    tl.store(Packed + 2 * offsets + 1, low)
    if ARRIVALS:
        tl.store(Arrivals + offsets, 0, offsets < ARRIVALS)


@triton.jit
def mega_mhc_packed_kernel(
    X,
    Residual,
    Post,
    Comb,
    Packed,
    Scale,
    Base,
    Norm,
    Updated,
    NextPost,
    NextComb,
    Y,
    Tokens,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PARTITIONS: tl.constexpr,
    SPLIT_HEADS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    POST_SCALE: tl.constexpr,
    SINKHORN_ROUNDS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    tile = tl.program_id(0)
    if SPLIT_HEADS:
        # The gate workgroup owns the residual and normalized output. The
        # other workgroup only projects and applies Sinkhorn to heads 8:24.
        gate = tl.program_id(1) == 0
        heads = tl.program_id(1) * 8 + tl.arange(0, 16)
        NUM_HEADS: tl.constexpr = 16
    else:
        gate = True
        heads = tl.arange(0, 32)
        NUM_HEADS: tl.constexpr = 32

    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = rows[:, None] < Tokens
    x_ptr = X + rows[:, None] * H
    r_ptr = Residual + rows[:, None] * (4 * H)
    u_ptr = Updated + rows[:, None] * (4 * H)
    safe_rows = tl.minimum(rows[:, None], Tokens - 1)
    post_ptr = Post + safe_rows * 4
    comb_ptr = Comb + safe_rows * 16
    c0 = _load_post_coefficients(post_ptr, comb_ptr, 0)
    c1 = _load_post_coefficients(post_ptr, comb_ptr, 1)
    c2 = _load_post_coefficients(post_ptr, comb_ptr, 2)
    c3 = _load_post_coefficients(post_ptr, comb_ptr, 3)

    PARTS_PER_STREAM: tl.constexpr = K_PARTITIONS // 4
    PANEL: tl.constexpr = PARTS_PER_STREAM * BLOCK_K
    part = tl.arange(0, K_PARTITIONS)
    part_start = part // PARTS_PER_STREAM * H + part % PARTS_PER_STREAM * BLOCK_K
    h = tl.arange(0, PANEL)[None, :]
    squares = tl.full((BLOCK_M, PANEL), 0, tl.float32)
    acc = tl.full((K_PARTITIONS, BLOCK_M, NUM_HEADS), 0, tl.float32)
    for panel in tl.range(H // PANEL, num_stages=1):
        start = panel * PANEL
        pos = start + h
        x, r0, r1, r2, r3 = _load_inputs(x_ptr, r_ptr, pos, H, valid)
        v0 = _post_from_coeffs(x, r0, r1, r2, r3, c0)
        v1 = _post_from_coeffs(x, r0, r1, r2, r3, c1)
        v2 = _post_from_coeffs(x, r0, r1, r2, r3, c2)
        v3 = _post_from_coeffs(x, r0, r1, r2, r3, c3)
        if gate:
            tl.store(u_ptr + pos, v0, valid)
            tl.store(u_ptr + H + pos, v1, valid)
            tl.store(u_ptr + 2 * H + pos, v2, valid)
            tl.store(u_ptr + 3 * H + pos, v3, valid)
        f0, f1 = v0.to(tl.float32), v1.to(tl.float32)
        f2, f3 = v2.to(tl.float32), v3.to(tl.float32)
        squares = tl.fma(f0, f0, squares)
        squares = tl.fma(f1, f1, squares)
        squares = tl.fma(f2, f2, squares)
        squares = tl.fma(f3, f3, squares)

        # Feed the BF16 Post values directly to projection before discarding
        # this panel. Match the high/low BF16 pairs in the packed K dimension.
        values = tl.permute(tl.join(tl.join(v0, v1), tl.join(v2, v3)), (3, 2, 0, 1))
        values = tl.reshape(values, (4, BLOCK_M, PARTS_PER_STREAM, BLOCK_K))
        values = tl.reshape(
            tl.permute(values, (0, 2, 1, 3)), (K_PARTITIONS, BLOCK_M, BLOCK_K)
        )
        a = tl.reshape(tl.join(values, values), (K_PARTITIONS, BLOCK_M, 2 * BLOCK_K))
        k = 2 * (part_start[:, None] + start) + tl.arange(0, 2 * BLOCK_K)[None, :]
        w = tl.load(Packed + heads[None, None, :] * (8 * H) + k[:, :, None])
        acc = tl.dot(a, w, acc)

    acc = tl.sum(acc, 0)
    sum_square = tl.sum(squares, 1)
    if SPLIT_HEADS:
        rms = tl.rsqrt(sum_square / (4 * H) + RMS_EPS)[:, None]
        if gate:
            streams = tl.broadcast_to(tl.arange(0, 4)[None, :], (BLOCK_M, 4))
            pre = tl.gather(acc, streams, axis=1) * rms
            pre = (
                tl.sigmoid(tl.fma(pre, tl.load(Scale), tl.load(Base + streams)))
                + PRE_EPS
            )
            post = tl.gather(acc, streams + 4, axis=1) * rms
            post = (
                tl.sigmoid(
                    tl.fma(post, tl.load(Scale + 1), tl.load(Base + streams + 4))
                )
                * POST_SCALE
            )
            tl.store(NextPost + rows[:, None] * 4 + streams, post, valid)
            tl.debug_barrier()
            _normalize_rows(
                Updated, Y, Norm, pre, tile, Tokens, H, BLOCK_M, 2, NORM_EPS
            )
        else:
            pairs = tl.arange(0, 16)[None, :]
            logits = tl.fma(acc * rms, tl.load(Scale + 2), tl.load(Base + pairs + 8))
            comb = _sinkhorn(logits, BLOCK_M, SINKHORN_EPS, SINKHORN_ROUNDS)
            tl.store(NextComb + rows[:, None] * 16 + pairs, comb, valid)
    else:
        pre, post, comb = _mix(
            acc,
            sum_square,
            Scale,
            Base,
            H,
            BLOCK_M,
            RMS_EPS,
            PRE_EPS,
            SINKHORN_EPS,
            POST_SCALE,
            SINKHORN_ROUNDS,
        )
        tl.store(NextPost + rows[:, None] * 4 + tl.arange(0, 4)[None, :], post, valid)
        tl.store(NextComb + rows[:, None] * 16 + tl.arange(0, 16)[None, :], comb, valid)
        tl.debug_barrier()
        _normalize_rows(Updated, Y, Norm, pre, tile, Tokens, H, BLOCK_M, 2, NORM_EPS)


@triton.jit
def _decode_split_group(
    X,
    R,
    Post,
    Comb,
    W,
    Scale,
    Base,
    Norm,
    Updated,
    NextPost,
    NextComb,
    Y,
    Partial,
    Count,
    row,
    split,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    FIRST: tl.constexpr,
    HEADS: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
    INTERLEAVE_GROUPS: tl.constexpr,
):
    K: tl.constexpr = H // SPLITS
    columns = tl.arange(0, K)
    heads = FIRST + tl.arange(0, HEADS)
    sq = tl.full((K,), 0, tl.float32)
    acc = tl.full((HEADS, K), 0, tl.float32)
    coeffs = (
        _load_post_coefficients(Post + row * 4, Comb + row * 16, 0),
        _load_post_coefficients(Post + row * 4, Comb + row * 16, 1),
        _load_post_coefficients(Post + row * 4, Comb + row * 16, 2),
        _load_post_coefficients(Post + row * 4, Comb + row * 16, 3),
    )
    # Bound the split coordinate so FlagTree can infer the segment alignment.
    h = (split % SPLITS) * K + columns
    x, r0, r1, r2, r3 = _load_inputs(X + row * H, R + row * 4 * H, h, H, True)
    for stream in tl.static_range(4):
        v = _post_from_coeffs(x, r0, r1, r2, r3, coeffs[stream])
        if FIRST == 0:
            tl.store(Updated + (row * 4 + stream) * H + h, v, cache_modifier=".cg")
        vf = v.to(tl.float32)
        sq = tl.fma(vf, vf, sq)
        w = tl.load(W + (heads[:, None] * 4 + stream) * H + h[None, :])
        acc = tl.fma(w, vf[None, :], acc)
    group = tl.program_id(1 if INTERLEAVE_GROUPS else 2)
    stats = Partial + (row * 2 + group) * SPLITS * 32
    slot = stats + split * 32
    result = tl.sum(acc, 1)
    local_heads = tl.arange(0, HEADS)
    tl.store(slot + local_heads, result, cache_modifier=".cg")
    tl.store(slot + HEADS, tl.sum(sq, 0), cache_modifier=".cg")
    # Each workgroup publishes its segment before signaling. The last
    # arrival alone finishes the group, so earlier workgroups never wait.
    tl.debug_barrier()
    count = Count + row * 2 + group
    arrival = tl.atomic_add(count, 1, sem="acq_rel", scope="gpu")
    if arrival == SPLITS - 1:
        parts = tl.arange(0, SPLITS)
        local_sum = tl.load(stats + parts * 32 + HEADS, cache_modifier=".cg")
        local_proj = tl.load(
            stats + parts[:, None] * 32 + local_heads[None, :], cache_modifier=".cg"
        )
        rms = tl.rsqrt(tl.sum(local_sum, 0) / (4 * H) + 1e-6)
        projected = tl.sum(local_proj, 0) * rms
        if FIRST == 0:
            lanes = tl.arange(0, 4)
            pre = tl.gather(projected, lanes, axis=0)
            pre = tl.sigmoid(tl.fma(pre, tl.load(Scale), tl.load(Base + lanes))) + 1e-6
            post = tl.gather(projected, lanes + 4, axis=0)
            post = (
                tl.sigmoid(tl.fma(post, tl.load(Scale + 1), tl.load(Base + lanes + 4)))
                * 2.0
            )
            tl.store(NextPost + row * 4 + lanes, post)
            norm_col = tl.arange(0, H)
            value = tl.full((H,), 0, tl.float32)
            for stream in tl.static_range(4):
                weight = tl.sum(tl.where(lanes == stream, pre, 0.0), 0)
                value_bf16 = tl.load(
                    Updated + (row * 4 + stream) * H + norm_col, cache_modifier=".cg"
                ).to(tl.float32)
                value = tl.fma(value_bf16, weight, value)
            value = value.to(tl.bfloat16).to(tl.float32)
            norm = tl.rsqrt(tl.sum(value * value, 0) / H + 1e-6)
            gamma = tl.load(Norm + norm_col).to(tl.float32)
            tl.store(Y + row * H + norm_col, value * norm * gamma)
        else:
            logits = tl.fma(
                projected, tl.load(Scale + 2), tl.load(Base + local_heads + 8)
            )
            normalized_comb = _decode_sinkhorn(
                logits[None, :],
                1e-6,
                20,
                3 if SINGLE_TOKEN else 8 if INTERLEAVE_GROUPS else 6,
                1 if SINGLE_TOKEN else 4,
            )
            tl.store(NextComb + row * 16 + local_heads[None, :], normalized_comb)
        # The workspace is reused by the next ordered invocation or replay.
        tl.atomic_xchg(count, 0, sem="release", scope="gpu")


@triton.jit
def mega_mhc_decode_kernel(
    X,
    R,
    Post,
    Comb,
    W,
    Scale,
    Base,
    Norm,
    Updated,
    NextPost,
    NextComb,
    Y,
    Partial,
    Count,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
    INTERLEAVE_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(2 if INTERLEAVE_GROUPS else 1)
    if tl.program_id(1 if INTERLEAVE_GROUPS else 2) == 0:
        _decode_split_group(
            X,
            R,
            Post,
            Comb,
            W,
            Scale,
            Base,
            Norm,
            Updated,
            NextPost,
            NextComb,
            Y,
            Partial,
            Count,
            row,
            split,
            H,
            SPLITS,
            0,
            8,
            SINGLE_TOKEN,
            INTERLEAVE_GROUPS,
        )
    else:
        _decode_split_group(
            X,
            R,
            Post,
            Comb,
            W,
            Scale,
            Base,
            Norm,
            Updated,
            NextPost,
            NextComb,
            Y,
            Partial,
            Count,
            row,
            split,
            H,
            SPLITS,
            8,
            16,
            SINGLE_TOKEN,
            INTERLEAVE_GROUPS,
        )

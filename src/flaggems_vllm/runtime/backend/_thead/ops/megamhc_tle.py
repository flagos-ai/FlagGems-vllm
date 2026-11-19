# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""PPU TLE kernels and native bindings for AIU and SIMD fusion."""

import triton
import triton.language as tl
from triton.experimental.tle import language as tle
from triton.language import core

from flaggems_vllm.ops.mhc.mhc_pre import _mhc_recip

from .megamhc_kernel import _pack_weights


@core.extern
def _wait_native(address, expected, _semantic=None):
    return core.extern_elementwise(
        "megamhc",
        "",
        [address, expected],
        {(tl.uint64, tl.uint64): ("megamhc_wait", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _wait(address, expected, NATIVE: tl.constexpr):
    if NATIVE:
        _wait_native(address.to(tl.uint64), tl.full((), expected, tl.uint64))
    else:
        # Poll with volatile loads before acquiring readiness. The acquire
        # broadcasts through CTA shared memory, so only do it when ready.
        ready = tl.full((), 0, tl.int64)
        while ready < expected:
            current = tl.load(address, volatile=True)
            while current < expected:
                current = tl.load(address, volatile=True)

            ready = tl.atomic_add(address, 0, sem="acquire", scope="gpu")
            # Finish reading the broadcast slot before it can be reused.
            tl.debug_barrier()


@core.extern
def _post_async_call(
    x,
    residual,
    weights,
    comb,
    output,
    squares,
    row,
    comb_stride_token,
    comb_stride_src,
    comb_stride_dst,
    shared,
    hidden,
    rows,
    R: tl.constexpr,
    _semantic=None,
):
    name = "megamhc_post_2" if R == 2 else "megamhc_post_4"
    return core.extern_elementwise(
        "megamhc",
        "",
        [
            x,
            residual,
            weights,
            comb,
            output,
            squares,
            row,
            comb_stride_token,
            comb_stride_src,
            comb_stride_dst,
            shared,
            hidden,
            rows,
        ],
        {(tl.uint64,) * 10 + (tl.uint32,) * 3: (name, tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _post_async(
    x,
    residual,
    weights,
    comb,
    output,
    squares,
    row,
    comb_stride_token,
    comb_stride_src,
    comb_stride_dst,
    shared_buffer,
    H: tl.constexpr,
    R: tl.constexpr,
    ROWS: tl.constexpr,
):
    address = tle.gpu.local_ptr(shared_buffer, (0,)).to(tl.uint64).to(tl.uint32)
    _post_async_call(
        x.to(tl.uint64),
        residual.to(tl.uint64),
        weights.to(tl.uint64),
        comb.to(tl.uint64),
        output.to(tl.uint64),
        squares.to(tl.uint64),
        row.to(tl.uint64),
        tl.cast(comb_stride_token, tl.uint64),
        tl.cast(comb_stride_src, tl.uint64),
        tl.cast(comb_stride_dst, tl.uint64),
        address,
        tl.full((), H, tl.uint32),
        tl.full((), ROWS, tl.uint32),
        R,
    )


@triton.jit
def _post(
    X,
    Residual,
    Post,
    Comb,
    NewResidual,
    SumSq,
    row_start,
    comb_stride_token,
    comb_stride_src,
    comb_stride_dst,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ALIGNED_ROWS: tl.constexpr,
    I32_RESIDUAL_OFFSETS: tl.constexpr,
    WRITE_THROUGH: tl.constexpr,
):
    rows = row_start + tl.arange(0, ROWS)
    dense_rows = rows.to(tl.int32) if I32_RESIDUAL_OFFSETS else rows
    valid_rows = None if ALIGNED_ROWS else rows < T
    other_row = None if ALIGNED_ROWS else 0
    post_weights = ()
    comb_weights = ()
    for dst in tl.static_range(R):
        post_weights += (tl.load(Post + rows * R + dst, valid_rows, other_row),)
        for src in tl.static_range(R):
            source = (
                Comb
                + rows * comb_stride_token
                + src * comb_stride_src
                + dst * comb_stride_dst
            )
            comb_weights += (tl.load(source, valid_rows, other_row),)
    columns = tl.arange(0, BLOCK_H)
    square = tl.zeros((ROWS, BLOCK_H), tl.float32)
    # Keep less data live when the software pipeline has many hidden tiles.
    UNROLL: tl.constexpr = 1 if H >= 8 * BLOCK_H else 2
    for hstart in tl.range(0, H, BLOCK_H, num_stages=3, loop_unroll_factor=UNROLL):
        h = hstart + columns
        if H % BLOCK_H == 0:
            mask = None if ALIGNED_ROWS else valid_rows[:, None]
        elif ALIGNED_ROWS:
            mask = (h < H)[None, :]
        else:
            mask = valid_rows[:, None] & (h < H)[None, :]
        other = None if ALIGNED_ROWS and H % BLOCK_H == 0 else 0
        xv = tl.load(
            X + dense_rows[:, None] * H + h[None, :],
            mask,
            other,
            eviction_policy="evict_first",
        ).to(tl.float32)
        accum = ()
        for dst in tl.static_range(R):
            accum += (post_weights[dst][:, None] * xv,)
        for src in tl.static_range(R):
            offset = (dense_rows[:, None] * R + src) * H + h[None, :]
            rv = tl.load(
                Residual + offset, mask, other, eviction_policy="evict_first"
            ).to(tl.float32)
            updated = ()
            for dst in tl.static_range(R):
                updated += (
                    tl.fma(comb_weights[dst * R + src][:, None], rv, accum[dst]),
                )
            accum = updated
        for dst in tl.static_range(R):
            rounded = accum[dst].to(tl.bfloat16)
            f = rounded.to(tl.float32)
            square = tl.fma(f, f, square)
            # Projection reads complete M tiles, including zero-padded rows.
            offset = (dense_rows[:, None] * R + dst) * H + h[None, :]
            tl.store(
                NewResidual + offset,
                rounded,
                None if H % BLOCK_H == 0 else (h < H)[None, :],
                cache_modifier=".wt" if WRITE_THROUGH else "",
                eviction_policy="evict_last",
            )
    tl.store(SumSq + rows, tl.sum(square, 1), valid_rows)


@triton.jit
def _project(
    NewResidual,
    Packed,
    Partial,
    group,
    part,
    PAD_T: tl.constexpr,
    K: tl.constexpr,
    Q: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    GROUPED_PARTIALS: tl.constexpr,
    FULL_K: tl.constexpr,
    SPLIT_PACKED: tl.constexpr = False,
):
    STEPS: tl.constexpr = triton.cdiv(triton.cdiv(K, BK), SPLITS)
    k0 = part * STEPS * BK
    a = tl.make_block_ptr(
        NewResidual, (PAD_T, K), (K, 1), (group * M, k0), (M, BK), (1, 0)
    )
    if SPLIT_PACKED:
        B_K: tl.constexpr = K // SPLITS
        weights = Packed + part * Q * B_K
        b_start = 0
    else:
        B_K: tl.constexpr = K
        weights = Packed
        b_start = k0
    b0 = tl.make_block_ptr(weights, (B_K, Q), (1, B_K), (b_start, 0), (BK, Q), (0, 1))
    b1 = tl.make_block_ptr(
        weights + K * Q, (B_K, Q), (1, B_K), (b_start, 0), (BK, Q), (0, 1)
    )
    b2 = tl.make_block_ptr(
        weights + 2 * K * Q, (B_K, Q), (1, B_K), (b_start, 0), (BK, Q), (0, 1)
    )
    main = tl.zeros((M, Q), tl.float32)
    correction = tl.zeros((M, Q), tl.float32)
    for _ in tl.range(STEPS, num_stages=2):
        if FULL_K:
            av = tle.load(a, is_async=True)
            high = tle.load(b0, is_async=True)
            middle = tle.load(b1, is_async=True)
            low = tle.load(b2, is_async=True)
        else:
            av = tle.load(a, boundary_check=(1,), padding_option="zero", is_async=True)
            high = tle.load(
                b0, boundary_check=(0,), padding_option="zero", is_async=True
            )
            middle = tle.load(
                b1, boundary_check=(0,), padding_option="zero", is_async=True
            )
            low = tle.load(
                b2, boundary_check=(0,), padding_option="zero", is_async=True
            )
        main = tl.dot(av, high, main)
        correction = tl.dot(av, middle, correction)
        correction = tl.dot(av, low, correction)
        a = tl.advance(a, (0, BK))
        b0 = tl.advance(b0, (BK, 0))
        b1 = tl.advance(b1, (BK, 0))
        b2 = tl.advance(b2, (BK, 0))
    if GROUPED_PARTIALS:
        row = (group.to(tl.int64) * SPLITS + part) * M + tl.arange(0, M)
    else:
        row = part.to(tl.int64) * PAD_T + group.to(tl.int64) * M + tl.arange(0, M)
    tl.store(
        Partial + row[:, None] * Q + tl.arange(0, Q)[None, :],
        main + correction,
        cache_modifier=".wt" if GROUPED_PARTIALS else "",
    )


@triton.jit
def _epilogue(
    Partial,
    SumSq,
    HCScale,
    HCBase,
    NewResidual,
    NewPost,
    NewComb,
    Norm,
    Y,
    rows,
    rms_eps,
    pre_eps,
    sink_eps,
    multiplier,
    repeats,
    norm_eps,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    Q: tl.constexpr,
    PAD_T: tl.constexpr,
    M: tl.constexpr,
    SPLITS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_NORM: tl.constexpr,
    ALIGNED_ROWS: tl.constexpr,
    LONG_QUEUE: tl.constexpr,
):
    valid = None if ALIGNED_ROWS else rows < T
    projection = tl.zeros((ROWS, Q), tl.float32)
    q = tl.arange(0, Q)
    if LONG_QUEUE:
        partial_rows = rows // M * SPLITS * M + rows % M
        stride = M
    else:
        partial_rows = rows
        stride = PAD_T
    for part in range(SPLITS):
        offsets = (part * stride + partial_rows[:, None]) * Q + q[None, :]
        projection += tl.load(Partial + offsets)
    sq = tl.load(SumSq + rows, valid, None if ALIGNED_ROWS else 0)
    logits = projection * tl.rsqrt(sq / (R * H) + rms_eps)[:, None]

    d = tl.arange(0, R)
    pairs = tl.arange(0, R * R)
    pidx = tl.broadcast_to(d[None, :], (ROWS, R))
    cidx = tl.broadcast_to(pairs[None, :] + 2 * R, (ROWS, R * R))
    pre = (
        tl.sigmoid(
            tl.gather(logits, pidx, 1) * tl.load(HCScale) + tl.load(HCBase + d)[None, :]
        )
        + pre_eps
    )
    post = tl.sigmoid(
        tl.gather(logits, pidx + R, 1) * tl.load(HCScale + 1)
        + tl.load(HCBase + R + d)[None, :]
    )
    row_mask = None if ALIGNED_ROWS else valid[:, None]
    tl.store(NewPost + rows[:, None] * R + d[None, :], post * multiplier, row_mask)
    c = (
        tl.gather(logits, cidx, 1) * tl.load(HCScale + 2)
        + tl.load(HCBase + 2 * R + pairs)[None, :]
    ).reshape((ROWS, R, R))
    c = tl.exp(c - tl.max(c, 2)[:, :, None])
    c = c * _mhc_recip(tl.sum(c, 2))[:, :, None] + sink_eps
    c = c * _mhc_recip(tl.sum(c, 1) + sink_eps)[:, None, :]
    for _ in tl.range(1, repeats, loop_unroll_factor=4):
        c = c * _mhc_recip(tl.sum(c, 2) + sink_eps)[:, :, None]
        c = c * _mhc_recip(tl.sum(c, 1) + sink_eps)[:, None, :]
    tl.store(
        NewComb + rows[:, None] * R * R + pairs[None, :],
        c.reshape((ROWS, R * R)),
        row_mask,
    )

    cols = tl.arange(0, BLOCK_H)
    square = tl.zeros((ROWS, BLOCK_H), tl.float32)
    weights = ()
    for src in tl.static_range(R):
        weights += (
            tl.gather(pre, tl.full((ROWS, 1), src, tl.int32), 1).reshape((ROWS,)),
        )
    for start in tl.range(0, H, BLOCK_H, num_stages=2):
        h = start + cols
        if H % BLOCK_H == 0:
            mask = row_mask
        elif ALIGNED_ROWS:
            mask = (h < H)[None, :]
        else:
            mask = row_mask & (h < H)[None, :]
        other = None if ALIGNED_ROWS and H % BLOCK_H == 0 else 0
        mix = tl.zeros((ROWS, BLOCK_H), tl.float32)
        for src in tl.static_range(R):
            v = tl.load(
                NewResidual + (rows[:, None] * R + src) * H + h[None, :],
                mask,
                other,
                eviction_policy="evict_first" if LONG_QUEUE else "",
            ).to(tl.float32)
            mix = tl.fma(weights[src][:, None], v, mix)
        rounded = mix.to(tl.bfloat16)
        if HAS_NORM:
            f = rounded.to(tl.float32)
            square = tl.fma(f, f, square)
        tl.store(
            Y + rows[:, None] * H + h[None, :],
            rounded,
            mask,
            cache_modifier=".wb",
            eviction_policy="evict_last",
        )
    if HAS_NORM:
        tl.debug_barrier()
        inv = tl.rsqrt(tl.sum(square, 1) / H + norm_eps)
        for start in tl.range(0, H, BLOCK_H, num_stages=1):
            if LONG_QUEUE:
                # Read the most recently written BF16 tiles first.
                h = (triton.cdiv(H, BLOCK_H) - 1) * BLOCK_H - start + cols
            else:
                h = start + cols
            if H % BLOCK_H == 0:
                mask = row_mask
                col_mask = None
            elif ALIGNED_ROWS:
                col_mask = h < H
                mask = col_mask[None, :]
            else:
                col_mask = h < H
                mask = row_mask & col_mask[None, :]
            other = None if ALIGNED_ROWS and H % BLOCK_H == 0 else 0
            off = rows[:, None] * H + h[None, :]
            value = tl.load(
                Y + off, mask, other, cache_modifier=".ca", eviction_policy="evict_last"
            ).to(tl.float32)
            norm = (
                tl.load(Norm + h, col_mask, None if H % BLOCK_H == 0 else 0)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            tl.store(
                Y + off,
                value * inv[:, None] * norm[None, :],
                mask,
                cache_modifier=".wt" if LONG_QUEUE else "",
            )


@triton.jit
def _compact_wave_slot(
    work,
    WAVES: tl.constexpr,
    LAG: tl.constexpr,
    POST_JOBS: tl.constexpr,
    GEMM_JOBS: tl.constexpr,
    EPI_JOBS: tl.constexpr,
):
    # At startup, post runs alone, then overlaps projection. During the
    # drain, only projection and epilogue remain, followed by epilogue alone.
    WAVE_JOBS: tl.constexpr = POST_JOBS + GEMM_JOBS + EPI_JOBS
    HEAD0: tl.constexpr = LAG * POST_JOBS
    HEAD1: tl.constexpr = HEAD0 + LAG * (POST_JOBS + GEMM_JOBS)
    MIDDLE: tl.constexpr = HEAD1 + (WAVES - 2 * LAG) * WAVE_JOBS
    TAIL: tl.constexpr = MIDDLE + LAG * (GEMM_JOBS + EPI_JOBS)
    if work < HEAD0:
        stripe = work // POST_JOBS
        slot = work % POST_JOBS
    elif work < HEAD1:
        idx = work - HEAD0
        stripe = LAG + idx // (POST_JOBS + GEMM_JOBS)
        slot = idx % (POST_JOBS + GEMM_JOBS)
    elif work < MIDDLE:
        idx = work - HEAD1
        stripe = 2 * LAG + idx // WAVE_JOBS
        slot = idx % WAVE_JOBS
    elif work < TAIL:
        idx = work - MIDDLE
        stripe = WAVES + idx // (GEMM_JOBS + EPI_JOBS)
        slot = POST_JOBS + idx % (GEMM_JOBS + EPI_JOBS)
    else:
        idx = work - TAIL
        stripe = WAVES + LAG + idx // EPI_JOBS
        slot = POST_JOBS + GEMM_JOBS + idx % EPI_JOBS
    return stripe, slot


@triton.jit
def _compact_partial_wave_slot(
    work,
    GROUPS: tl.constexpr,
    WAVE: tl.constexpr,
    LAG: tl.constexpr,
    POSTS: tl.constexpr,
    SPLITS: tl.constexpr,
    EPILOGUES: tl.constexpr,
):
    FULL: tl.constexpr = GROUPS // WAVE
    REM: tl.constexpr = GROUPS % WAVE
    # Before the last partial wave, post leads GEMM by LAG and epilogue by 2*LAG.
    PREFIX: tl.constexpr = (
        FULL * (POSTS + SPLITS + EPILOGUES) - LAG * (SPLITS + 2 * EPILOGUES)
    ) * WAVE
    if work < PREFIX:
        stripe, slot = _compact_wave_slot(
            work, FULL, LAG, WAVE * POSTS, WAVE * SPLITS, WAVE * EPILOGUES
        )
    else:
        tail = work - PREFIX
        stripe = FULL
        slot = 0
        for i in tl.static_range(2 * LAG + 1):
            posts = REM * POSTS if i == 0 else 0
            gemms = (WAVE if i < LAG else REM if i == LAG else 0) * SPLITS
            epilogues = (WAVE if i < 2 * LAG else REM) * EPILOGUES
            if tail >= 0 and tail < posts + gemms + epilogues:
                stripe = FULL + i
                # Keep the same slot ranges used by the full-wave dispatcher.
                if tail < posts:
                    slot = tail
                elif tail < posts + gemms:
                    slot = WAVE * POSTS + tail - posts
                else:
                    slot = WAVE * (POSTS + SPLITS) + tail - posts - gemms
            tail -= posts + gemms + epilogues
    return stripe, slot


@triton.jit
def mega_mhc_pipeline(
    X,
    Residual,
    Post,
    Comb,
    Fn,
    HCScale,
    HCBase,
    Norm,
    NewResidual,
    NewPost,
    NewComb,
    Y,
    Counters,
    Packed,
    Partial,
    SumSq,
    comb_stride_token,
    comb_stride_src,
    comb_stride_dst,
    rms_eps,
    pre_eps,
    sink_eps,
    multiplier,
    repeats,
    norm_eps,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    M: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    PACK_BLOCK: tl.constexpr,
    WAVE: tl.constexpr,
    LAG: tl.constexpr,
    HAS_NORM: tl.constexpr,
    LONG_QUEUE: tl.constexpr,
    COMPACT_WAVES: tl.constexpr,
    PREPACKED: tl.constexpr = False,
    SPLIT_PACKED: tl.constexpr = False,
    ASYNC_POST: tl.constexpr = False,
    NATIVE_WAIT: tl.constexpr = False,
):
    GROUPS: tl.constexpr = triton.cdiv(T, M)
    K: tl.constexpr = R * H
    PACK_JOBS: tl.constexpr = 0 if PREPACKED else triton.cdiv(K * Q, PACK_BLOCK)
    POST_ROWS: tl.constexpr = 2
    ROWS_PER_JOB: tl.constexpr = 8
    POSTS: tl.constexpr = M // ROWS_PER_JOB
    EPILOGUES: tl.constexpr = M // ROWS_PER_JOB
    WAVE_JOBS: tl.constexpr = WAVE * (POSTS + SPLITS + EPILOGUES)
    ALIGN_T: tl.constexpr = T % M == 0
    # Include the padded rows written by post in the largest dense offset.
    I32_RESIDUAL_OFFSETS: tl.constexpr = LONG_QUEUE and GROUPS * M * K <= 2**31
    # Tickets impose a topological order independent of CTA launch order:
    # every poll waits only on producers that already received their tickets.
    ticket = tl.atomic_add(Counters, 1, sem="relaxed").to(tl.int32)
    packed_ready = Counters + 1
    ready = Counters + (1 if PREPACKED else 2)
    # Release publishes every producer's writes. Acquiring the RMW from
    # the preceding producer chains publication across all CTAs in the stage.
    if not PREPACKED and ticket < PACK_JOBS:
        _pack_weights(Fn, Packed, ticket, K, P, Q, PACK_BLOCK)
        tl.debug_barrier()
        tl.atomic_add(packed_ready, 1, sem="acq_rel", scope="gpu")
    else:
        work = ticket - PACK_JOBS
        if COMPACT_WAVES and GROUPS % WAVE:
            stripe, slot = _compact_partial_wave_slot(
                work, GROUPS, WAVE, LAG, POSTS, SPLITS, EPILOGUES
            )
        elif COMPACT_WAVES:
            stripe, slot = _compact_wave_slot(
                work, GROUPS // WAVE, LAG, WAVE * POSTS, WAVE * SPLITS, WAVE * EPILOGUES
            )
        else:
            stripe = work // WAVE_JOBS
            slot = work % WAVE_JOBS
        start = stripe * WAVE
        if slot < WAVE * POSTS:
            group = start + slot // POSTS
            if group < GROUPS:
                row_start = group.to(tl.int64) * M + (slot % POSTS) * ROWS_PER_JOB
                if ASYNC_POST and (ALIGN_T or row_start + ROWS_PER_JOB <= T):
                    # Two copies of X and the residual streams for this row tile.
                    shared_buffer = tle.gpu.alloc(
                        [triton.next_power_of_2(2 * POST_ROWS * 256 * (R + 1))],
                        dtype=tl.uint16,
                        scope=tle.gpu.smem,
                        layout=None,
                    )
                    _post_async(
                        X,
                        Residual,
                        Post,
                        Comb,
                        NewResidual,
                        SumSq,
                        row_start,
                        comb_stride_token,
                        comb_stride_src,
                        comb_stride_dst,
                        shared_buffer,
                        H,
                        R,
                        ROWS_PER_JOB,
                    )
                else:
                    for step in tl.range(
                        ROWS_PER_JOB // POST_ROWS, loop_unroll_factor=1
                    ):
                        _post(
                            X,
                            Residual,
                            Post,
                            Comb,
                            NewResidual,
                            SumSq,
                            row_start + step * POST_ROWS,
                            comb_stride_token,
                            comb_stride_src,
                            comb_stride_dst,
                            T,
                            H,
                            R,
                            POST_ROWS,
                            256,
                            ALIGN_T,
                            I32_RESIDUAL_OFFSETS,
                            LONG_QUEUE,
                        )
                tl.debug_barrier()
                tl.atomic_add(ready + group * 2, 1, sem="acq_rel", scope="gpu")
        elif slot < WAVE * (POSTS + SPLITS):
            phase_slot = slot - WAVE * POSTS
            group = start - LAG * WAVE + phase_slot // SPLITS
            if group >= 0 and group < GROUPS:
                _wait(ready + group * 2, POSTS, NATIVE_WAIT)
                if not PREPACKED:
                    _wait(packed_ready, PACK_JOBS, NATIVE_WAIT)
                part = phase_slot % SPLITS
                # A narrow projection can use unmasked loads for complete
                # K splits while keeping the bounds checks on its tail.
                SPLIT_K: tl.constexpr = triton.cdiv(K, BK * SPLITS) * BK
                project_args = (NewResidual, Packed, Partial, group, part)
                if K % (SPLITS * BK) == 0 or (Q <= 16 and (part + 1) * SPLIT_K <= K):
                    _project(
                        *project_args,
                        GROUPS * M,
                        K,
                        Q,
                        M,
                        BK,
                        SPLITS,
                        LONG_QUEUE,
                        True,
                        SPLIT_PACKED,
                    )
                else:
                    _project(
                        *project_args,
                        GROUPS * M,
                        K,
                        Q,
                        M,
                        BK,
                        SPLITS,
                        LONG_QUEUE,
                        False,
                        SPLIT_PACKED,
                    )
                tl.debug_barrier()
                tl.atomic_add(ready + group * 2 + 1, 1, sem="acq_rel", scope="gpu")
        else:
            phase_slot = slot - WAVE * (POSTS + SPLITS)
            group = start - 2 * LAG * WAVE + phase_slot // EPILOGUES
            if group >= 0 and group < GROUPS:
                _wait(ready + group * 2 + 1, SPLITS, NATIVE_WAIT)
                rows = (
                    group.to(tl.int64) * M
                    + (phase_slot % EPILOGUES) * ROWS_PER_JOB
                    + tl.arange(0, ROWS_PER_JOB)
                )
                epilogue_args = (
                    Partial,
                    SumSq,
                    HCScale,
                    HCBase,
                    NewResidual,
                    NewPost,
                    NewComb,
                    Norm,
                    Y,
                    rows,
                    rms_eps,
                    pre_eps,
                    sink_eps,
                    multiplier,
                    repeats,
                    norm_eps,
                )
                if ALIGN_T or (H % 256 == 0 and group < GROUPS - 1):
                    _epilogue(
                        *epilogue_args,
                        T,
                        H,
                        R,
                        Q,
                        GROUPS * M,
                        M,
                        SPLITS,
                        ROWS_PER_JOB,
                        256,
                        HAS_NORM,
                        True,
                        LONG_QUEUE,
                    )
                else:
                    _epilogue(
                        *epilogue_args,
                        T,
                        H,
                        R,
                        Q,
                        GROUPS * M,
                        M,
                        SPLITS,
                        ROWS_PER_JOB,
                        256,
                        HAS_NORM,
                        False,
                        LONG_QUEUE,
                    )


@core.extern
def _simt_tile(*args, W, R, N, _semantic=None):
    w, r, n = W.value, R.value, N.value
    return core.extern_elementwise(
        "megamhc",
        "",
        list(args),
        {
            (tl.uint64,) * 16
            + (tl.uint32,) * 3
            + (tl.float32,) * 5
            + (tl.uint32,): (f"megamhc_simt_{w}_{r}_{n}", tl.int32)
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _simt_rows(
    pointers,
    scalars,
    first,
    T: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
):
    if T % N == 0 or first + N <= T:
        _simt_tile(*pointers, first, *scalars, W=W, R=R, N=N)
    else:
        for row in range(N):
            token = first + row
            if token < T:
                _simt_tile(*pointers, token, *scalars, W=W, R=R, N=1)


@triton.jit
def mega_mhc_simt_kernel(
    X,
    Residual,
    Post,
    Comb,
    Fn,
    Scale,
    Base,
    Norm,
    NewResidual,
    NewPost,
    NewComb,
    Y,
    comb_stride_t,
    comb_stride_src,
    comb_stride_dst,
    rms_eps,
    pre_eps,
    sinkhorn_eps,
    post_multiplier,
    sinkhorn_repeat,
    norm_eps,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    HAS_NORM: tl.constexpr,
    ROWS: tl.constexpr,
    WARPS: tl.constexpr,
    LARGE_BLOCKS: tl.constexpr = 0,
):
    shared = tle.gpu.alloc(
        [triton.next_power_of_2(2 * ROWS * 32 + ROWS * R + WARPS + ROWS)],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        layout=None,
    )
    address = tle.gpu.local_ptr(shared, (0,)).to(tl.uint64).to(tl.uint32)
    pointers = (
        X.to(tl.uint64),
        Residual.to(tl.uint64),
        Post.to(tl.uint64),
        Comb.to(tl.uint64),
        Fn.to(tl.uint64),
        Scale.to(tl.uint64),
        Base.to(tl.uint64),
        Norm.to(tl.uint64),
        NewResidual.to(tl.uint64),
        NewPost.to(tl.uint64),
        NewComb.to(tl.uint64),
        Y.to(tl.uint64),
    )
    scalars = (
        tl.cast(comb_stride_t, tl.uint64),
        tl.cast(comb_stride_src, tl.uint64),
        tl.cast(comb_stride_dst, tl.uint64),
        tl.full((), H, tl.uint32),
        address,
        tl.cast(sinkhorn_repeat, tl.uint32),
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        norm_eps,
        tl.full((), int(HAS_NORM), tl.uint32),
    )
    pid = tl.program_id(0).to(tl.uint64)
    if LARGE_BLOCKS:
        if pid < LARGE_BLOCKS:
            _simt_rows(pointers, scalars, pid * ROWS, T, WARPS, R, ROWS)
        else:
            first = LARGE_BLOCKS * ROWS + (pid - LARGE_BLOCKS) * (ROWS // 2)
            _simt_rows(pointers, scalars, first, T, WARPS, R, ROWS // 2)
    else:
        _simt_rows(pointers, scalars, pid * ROWS, T, WARPS, R, ROWS)

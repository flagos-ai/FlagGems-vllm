# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hygon BF16 MegaPreAttention kernels, derived from NVIDIA PR #1007 (89b0a512).

A single persistent grid owns peer transfers, row-ready publication, projection
and the fused epilogue. Communication CTAs stage arriving rows into private cached
memory while compute CTAs consume them, then join the projection queue.
IPC payloads and reuse flags remain fine-grained.
"""

import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _row_offset(row, stride, column=0):
    # Promote before multiplication, including offsets beyond INT32/UINT32.
    return row.to(tl.int64) * stride + column


@triton.jit
def _rms_scale(x, EPS: tl.constexpr, ROWS: tl.constexpr):
    # Match PyTorch's four-value sums and 64-lane FP32 reduction order.
    columns = tl.arange(0, 256)
    base = tl.broadcast_to((columns & ~3)[None, :], (ROWS, 256))
    squared = x * x
    x0 = tl.gather(squared, base, 1)
    x1 = tl.gather(squared, base + 1, 1)
    x2 = tl.gather(squared, base + 2, 1)
    x3 = tl.gather(squared, base + 3, 1)
    values = ((x0 + x1) + x2) + x3
    for bit in tl.static_range(2, 8):
        partner = tl.broadcast_to((columns ^ (1 << bit))[None, :], (ROWS, 256))
        values = values + tl.gather(values, partner, 1)
    total = tl.gather(values, tl.full((ROWS, 1), 0, tl.int32), 1).reshape(ROWS)
    # HIP resolves PyTorch's ::rsqrt(float) through its double overload.
    return libdevice.rsqrt((total / 256.0 + EPS).to(tl.float64)).to(tl.float32)


@triton.jit
def _store_half_head(
    acc,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    positions,
    rope,
    cache,
    slots,
    row_start,
    pid_m,
    pid_n,
    LOCAL_T: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    POS_AXIS,
    POS_TOKEN,
    ROPE_STRIDE,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    EPS: tl.constexpr = 1.0e-6,
):
    row = _row_offset(pid_m, BLOCK_M, tl.arange(0, BLOCK_M))
    token = row_start + row
    valid = row < LOCAL_T
    cols = tl.arange(0, 256)
    raw = acc.to(tl.bfloat16)
    # Preserve the materialized BF16 projection and norm rounding boundaries.
    if pid_n < 2 * Q_HEADS and pid_n % 2 == 1:
        head = pid_n // 2
        tl.store(
            gate + token[:, None] * (Q_HEADS * 256) + head * 256 + cols[None, :],
            raw,
            mask=valid[:, None],
        )
    elif pid_n >= 2 * Q_HEADS + KV_HEADS:
        value_head = (
            (pid_n - 2 * Q_HEADS - KV_HEADS).to(tl.int64) if KV_HEADS > 1 else 0
        )
        tl.store(
            value
            + token[:, None] * (KV_HEADS * 256)
            + value_head * 256
            + cols[None, :],
            raw,
            mask=valid[:, None],
        )
        slot = tl.load(slots + token * SLOT_STRIDE, mask=valid, other=-1).to(tl.int64)
        target = (
            cache
            + (slot // CACHE_BLOCK_SIZE)[:, None] * CACHE_BLOCK
            + value_head * CACHE_HEAD
            + (slot % CACHE_BLOCK_SIZE)[:, None] * CACHE_TOKEN
        )
        tl.store(
            target + (256 + cols[None, :].to(tl.int64)) * CACHE_DIM,
            raw,
            mask=(valid & (slot >= 0))[:, None],
        )
    else:
        is_key = pid_n >= 2 * Q_HEADS if KV_HEADS > 1 else pid_n == 2 * Q_HEADS
        head = pid_n // 2
        weight = tl.load(tl.where(is_key, knorm, qnorm) + cols).to(tl.float32)
        x = raw.to(tl.float32)
        scale = _rms_scale(x, EPS, BLOCK_M)
        normalized = (x * scale[:, None] * (weight[None, :] + 1.0)).to(tl.bfloat16)
        frequency = tl.arange(0, 32)
        axis = tl.where(
            (frequency % 3 == 1) & (frequency < 3 * SECTION_H),
            1,
            tl.where((frequency % 3 == 2) & (frequency < 3 * SECTION_W), 2, 0),
        )
        pos = tl.load(
            positions
            + axis[None, :].to(tl.int64) * POS_AXIS
            + token[:, None] * POS_TOKEN,
            valid[:, None],
            other=0,
        )
        co = tl.load(
            rope + pos * ROPE_STRIDE + frequency[None, :], valid[:, None], other=1
        ).to(tl.float32)
        si = tl.load(
            rope + pos * ROPE_STRIDE + 32 + frequency[None, :], valid[:, None], other=0
        ).to(tl.float32)
        # Only the first 64 dimensions rotate; pass the other 192 through.
        # Static splits avoid a full-width gather and unused rotary arithmetic.
        low128, high128 = tl.split(normalized.reshape(BLOCK_M, 2, 128).permute(0, 2, 1))
        low64, high64 = tl.split(low128.reshape(BLOCK_M, 2, 64).permute(0, 2, 1))
        first, second = tl.split(low64.reshape(BLOCK_M, 2, 32).permute(0, 2, 1))
        first = first.to(tl.float32)
        second = second.to(tl.float32)
        first_out = first * co - second * si
        second_out = second * co + first * si
        rotated64 = (
            tl.join(first_out, second_out)
            .permute(0, 2, 1)
            .reshape(BLOCK_M, 64)
            .to(tl.bfloat16)
        )
        result128 = tl.join(rotated64, high64).permute(0, 2, 1).reshape(BLOCK_M, 128)
        result = tl.join(result128, high128).permute(0, 2, 1).reshape(BLOCK_M, 256)
        if is_key:
            kv_head = (pid_n - 2 * Q_HEADS).to(tl.int64) if KV_HEADS > 1 else 0
            tl.store(
                key + token[:, None] * (KV_HEADS * 256) + kv_head * 256 + cols[None, :],
                result,
                mask=valid[:, None],
            )
            slot = tl.load(slots + token * SLOT_STRIDE, mask=valid, other=-1).to(
                tl.int64
            )
            target = (
                cache
                + (slot // CACHE_BLOCK_SIZE)[:, None] * CACHE_BLOCK
                + kv_head * CACHE_HEAD
                + (slot % CACHE_BLOCK_SIZE)[:, None] * CACHE_TOKEN
            )
            tl.store(
                target + cols[None, :].to(tl.int64) * CACHE_DIM,
                result,
                mask=(valid & (slot >= 0))[:, None],
            )
        else:
            tl.store(
                query + token[:, None] * (Q_HEADS * 256) + head * 256 + cols[None, :],
                result,
                mask=valid[:, None],
            )


@triton.jit
def _store_head(
    acc,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    positions,
    rope,
    cache,
    slots,
    row_start,
    pid_m,
    pid_n,
    LOCAL_T: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    POS_AXIS,
    POS_TOKEN,
    ROPE_STRIDE,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    EPS: tl.constexpr = 1.0e-6,
):
    # Keep the GEMM tile intact; shorten live temporaries in norm/RoPE/cache.
    low, high = tl.split(acc.reshape(2, BLOCK_M // 2, 256).permute(1, 2, 0))
    if BLOCK_M == 256:
        low0, low1 = tl.split(low.reshape(2, 64, 256).permute(1, 2, 0))
        high0, high1 = tl.split(high.reshape(2, 64, 256).permute(1, 2, 0))
        chunks = (low0, low1, high0, high1)
    else:
        chunks = (low, high)
    for half in tl.static_range(BLOCK_M // 64):
        part = chunks[half]
        _store_half_head(
            part,
            query,
            key,
            value,
            gate,
            qnorm,
            knorm,
            positions,
            rope,
            cache,
            slots,
            row_start,
            (BLOCK_M // 64) * pid_m + half,
            pid_n,
            LOCAL_T,
            Q_HEADS,
            KV_HEADS,
            POS_AXIS,
            POS_TOKEN,
            ROPE_STRIDE,
            SLOT_STRIDE,
            CACHE_BLOCK,
            CACHE_HEAD,
            CACHE_TOKEN,
            CACHE_DIM,
            CACHE_BLOCK_SIZE,
            64,
            SECTION_H,
            SECTION_W,
            EPS,
        )


@triton.jit
def _init_metadata(
    flags,
    ready,
    staged,
    epochs,
    peers,
    ADDRESSES: tl.constexpr,
    WORLD: tl.constexpr,
    COMM: tl.constexpr,
    PROGRAMS: tl.constexpr,
    ROWS: tl.constexpr,
):
    i = tl.arange(0, triton.next_power_of_2(max(WORLD * COMM, PROGRAMS + 2, ROWS)))
    tl.store(flags + i, 0, i < WORLD * COMM)
    tl.store(ready + i, 0, i < ROWS)
    tl.store(staged + i, 0, i < ROWS)
    tl.store(epochs + i, 0, i < PROGRAMS + 2)
    for rank in tl.static_range(WORLD):
        tl.store(peers + rank, ADDRESSES[rank])


@triton.jit
def _block_fence():
    # Match HIP __threadfence_system in every wave before the CTA barrier.
    # Cover every wave of the largest (16-wave) projection CTA.
    lanes = tl.arange(0, 1024)
    tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0) lgkmcnt(0); buffer_wbinvl1_vol; "
        "s_waitcnt lgkmcnt(0); v_mov_b32 $0, $1;",
        constraints="=v,v,~{memory}",
        args=[lanes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()


@triton.jit
def _wait_ready(pointer, target):
    seen = tl.load(pointer, volatile=True, cache_modifier=".cv")
    while seen < target:
        seen = tl.load(pointer, volatile=True, cache_modifier=".cv")
    _block_fence()


@triton.jit
def _transport(
    hidden,
    buffer,
    staging,
    peers,
    flags,
    ready,
    staged,
    rank,
    epoch,
    DATA_WORDS: tl.constexpr,
    M: tl.constexpr,
    HIDDEN: tl.constexpr,
    WORLD: tl.constexpr,
    COMM: tl.constexpr,
    BM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    ranks = tl.arange(0, WORLD)
    destinations = tl.load(peers + ranks).to(tl.pointer_type(tl.int64))
    tl.atomic_xchg(
        destinations + DATA_WORDS + pid * WORLD + rank,
        epoch,
        sem="release",
        scope="sys",
    )
    waiting = tl.full((), 1, tl.int32)
    while waiting != 0:
        seen = tl.atomic_add(flags + pid * WORLD + ranks, 0, sem="acquire", scope="sys")
        waiting = tl.sum((seen < epoch).to(tl.int32), 0)
    _block_fence()
    local_rows: tl.constexpr = M // WORLD // BM
    # Copy bits as 64-bit words; no BF16 conversion or lane repacking.
    hidden = hidden.to(tl.pointer_type(tl.uint64))
    buffer = buffer.to(tl.pointer_type(tl.uint64))
    staging = staging.to(tl.pointer_type(tl.uint64))
    chunk_size: tl.constexpr = BM * HIDDEN // 4
    offsets = tl.arange(0, BLOCK)
    for chunk in range(pid, local_rows, COMM):
        output_row = rank * local_rows + chunk
        for block in tl.range(0, tl.cdiv(chunk_size, BLOCK), num_stages=3):
            index = block * BLOCK + offsets
            values = tl.load(
                hidden + _row_offset(chunk, chunk_size, index), index < chunk_size, 0
            )
            if WORLD == 4:
                # Publish once locally; peers pull into their cacheable staging.
                tl.store(
                    buffer + _row_offset(output_row, chunk_size, index),
                    values,
                    index < chunk_size,
                )
            else:
                for step in tl.static_range(1, WORLD):
                    peer = (rank + step) % WORLD
                    target = tl.load(peers + peer).to(tl.pointer_type(tl.uint64))
                    tl.store(
                        target + _row_offset(output_row, chunk_size, index),
                        values,
                        index < chunk_size,
                    )
        _block_fence()
        for step in tl.static_range(1, WORLD):
            peer = (rank + step) % WORLD
            peer_words = tl.load(peers + peer).to(tl.pointer_type(tl.int64))
            tl.atomic_xchg(
                peer_words + DATA_WORDS + COMM * WORLD + output_row,
                epoch,
                sem="release",
                scope="sys",
            )
        # Every rank sends this chunk before waiting for any incoming chunk.
        # One producer owns each row, including its cacheable local staging.
        for step in tl.static_range(1, WORLD):
            peer = (rank + step) % WORLD
            row = peer * local_rows + chunk
            _wait_ready(ready + row, epoch)
            if WORLD == 4:
                incoming = tl.load(peers + peer).to(tl.pointer_type(tl.uint64))
            else:
                incoming = buffer
            for block in tl.range(0, tl.cdiv(chunk_size, BLOCK), num_stages=3):
                index = block * BLOCK + offsets
                values = tl.load(
                    incoming + _row_offset(row, chunk_size, index),
                    index < chunk_size,
                    0,
                )
                tl.store(
                    staging + _row_offset(row, chunk_size, index),
                    values,
                    index < chunk_size,
                )
            _block_fence()
            tl.atomic_xchg(staged + row, epoch, sem="release", scope="gpu")


@triton.jit
def _persistent(
    hidden,
    staging,
    weight,
    buffer,
    peers,
    flags,
    ready,
    staged,
    epochs,
    rank,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    positions,
    rope,
    cache,
    slots,
    POS_AXIS,
    POS_TOKEN,
    ROPE_STRIDE,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    DATA_WORDS: tl.constexpr,
    M: tl.constexpr,
    HIDDEN: tl.constexpr,
    WORLD: tl.constexpr,
    COMM: tl.constexpr,
    PROGRAMS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    EPS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    epoch = tl.atomic_add(epochs + pid, 1, sem="relaxed") + 1
    # The previous invocation has finished on this stream before resetting work.
    if pid == 0:
        tl.store(epochs + PROGRAMS, 0)
        _block_fence()
        tl.atomic_xchg(epochs + PROGRAMS + 1, epoch, sem="release", scope="gpu")
    if pid < COMM:
        _transport(
            hidden,
            buffer,
            staging,
            peers,
            flags,
            ready,
            staged,
            rank,
            epoch,
            DATA_WORDS,
            M,
            HIDDEN,
            WORLD,
            COMM,
            128,
            1024,
        )
    _wait_ready(epochs + PROGRAMS + 1, epoch)
    num_m: tl.constexpr = M // BM
    local_m: tl.constexpr = num_m // WORLD
    heads: tl.constexpr = 2 * (Q_HEADS + KV_HEADS)
    rows = tl.arange(0, BM)
    cols = tl.arange(0, 256)
    kk = tl.arange(0, BK)
    # Consumers alone can cover every tile; producers join after transport.
    # The bounded surplus claims avoid a second grid-wide reset/barrier.
    for task in tl.range(0, tl.cdiv(num_m * heads, PROGRAMS - COMM), disable_licm=True):
        tile = tl.atomic_add(epochs + PROGRAMS, 1, sem="relaxed").to(tl.int32)
        if tile < num_m * heads:
            group = tile // (GROUP_M * heads)
            first_m = group * GROUP_M
            group_m = tl.minimum(num_m - first_m, GROUP_M)
            within = tile % (GROUP_M * heads)
            pm = (first_m + within % group_m + rank * local_m) % num_m
            pn = within // group_m
            local = (pm >= rank * local_m) & (pm < (rank + 1) * local_m)
            if local:
                source = hidden + _row_offset(pm - rank * local_m, BM * HIDDEN)
            else:
                for packet in tl.static_range(BM // 128):
                    _wait_ready(staged + pm * BM // 128 + packet, epoch)
                source = staging + _row_offset(pm, BM * HIDDEN)
            a_ptr = source + _row_offset(rows[:, None], HIDDEN, kk[None, :])
            b_ptr = weight + _row_offset(pn * 256 + cols[None, :], HIDDEN, kk[:, None])
            acc = tl.zeros((BM, 256), tl.float32)
            for k in range(HIDDEN // BK):
                a = tl.load(a_ptr)
                b = tl.load(b_ptr)
                acc = tl.dot(a, b, acc)
                a_ptr += BK
                b_ptr += BK
            _store_head(
                acc,
                query,
                key,
                value,
                gate,
                qnorm,
                knorm,
                positions,
                rope,
                cache,
                slots,
                0,
                pm,
                pn,
                M,
                Q_HEADS,
                KV_HEADS,
                POS_AXIS,
                POS_TOKEN,
                ROPE_STRIDE,
                SLOT_STRIDE,
                CACHE_BLOCK,
                CACHE_HEAD,
                CACHE_TOKEN,
                CACHE_DIM,
                CACHE_BLOCK_SIZE,
                BM,
                SECTION_H,
                SECTION_W,
                EPS,
            )


@triton.jit
def _fill_buffer(pointer, value, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(pointer + i, value, i < N)

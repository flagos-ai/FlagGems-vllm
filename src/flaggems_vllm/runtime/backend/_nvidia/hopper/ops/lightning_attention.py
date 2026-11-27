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

"""Hopper TLE kernels for MiniMax Lightning Attention (prefill and decode).

Prefill decomposes causal linear attention with exponential decay into:
1. Per-chunk state summaries via TF32 WGMMA / warp MMA (_chunk_summary).
2. Inter-chunk decay scan producing per-chunk carries and 256-token history
   snapshots (_scan_chunks or windowed _scan_parallel).
3. Intra-chunk causal attention fused with cross-chunk state projection
   (_chunk_output).
Single-chunk prefixes (num_chunks == 1) fuse summary and output into one launch;
T <= 1 uses a direct single-step kernel (_prefix_tiny_kernel).

Decode updates active [slots, H, D, D] cache rows in-place using TLE
BlockEncoding layouts matched to cache strides and SM wave occupancy.

Autotune exemption: prefill orchestrates a multi-kernel pipeline and decode
mutates persistent KV caches in-place (~2-5 us per call). Host dispatch derives
tile and wave geometry directly from Hopper hardware properties without online
autotuning.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

# vLLM Lightning Attention contract: emit one FP32 state snapshot before each
# 256-token block plus the final tail state, regardless of block_size.
_HISTORY_BLOCK = 256

# Hopper Tensor Core instruction row granularity:
# - Warpgroup WGMMA (wgmma.mma_async.sync.aligned.m64n*k8) requires M = 64 (4 warps).
# - Warp-level MMA (mma.sync.aligned.m16n8k8) requires M >= 16 (2 warps).
_WGMMA_M = 64
_WARP_MMA_M = 16

# Inter-chunk state scan tile geometry:
# - Sequential _scan_chunks tiles [D, E] into 16 x 32 blocks (512 elements/CTA).
# - Windowed _scan_parallel scans _SCAN_WINDOW = 16 chunks per outer step.
# - _EMPIRICAL_SCAN_CHUNKS = 48 (3 * _SCAN_WINDOW): empirical Hopper threshold
#   (measured on H20 and H800) for small states that remain L2-resident; above 48
#   chunks, O(n / 16) associative scan amortizes its shared-memory scan/gather cost
#   over _scan_chunks's O(n) serial dependency chain.
_SCAN_TILE_D = 16
_SCAN_TILE_V = 32
_SCAN_WINDOW = 16
_EMPIRICAL_SCAN_CHUNKS = 3 * _SCAN_WINDOW

# TLE register/thread layouts for [D, BV] state tiles:
# - _SMALL_LAYOUT:      1 warp,  [4, 8] threads x [1, 2] elems -> [4, 16] per step
# - _NARROW_LAYOUT:     2 warps, [4, 8] threads x [1, 2] elems -> [8, 16] per step
# - _WIDE_LAYOUT:       4 warps, [4, 8] threads x [1, 4] elems -> [16, 32] per step
# - _TRANSPOSED_LAYOUT: 1 warp, [16, 2] threads x [2, 1] elems -> [32, 2] (D-contiguous)
_SMALL_LAYOUT = tle.gpu.BlockEncoding([1, 2], [4, 8], [1, 1], [1, 0])
_NARROW_LAYOUT = tle.gpu.BlockEncoding([1, 2], [4, 8], [2, 1], [1, 0])
_WIDE_LAYOUT = tle.gpu.BlockEncoding([1, 4], [4, 8], [4, 1], [1, 0])
_TRANSPOSED_LAYOUT = tle.gpu.BlockEncoding([2, 1], [16, 2], [1, 1], [0, 1])


@lru_cache(maxsize=8)
def _device_arch_by_index(device_index: int) -> tuple[int, int]:
    props = torch.cuda.get_device_properties(device_index)
    return props.multi_processor_count, props.L2_cache_size


def _device_arch(device: torch.device) -> tuple[int, int]:
    index = torch.cuda.current_device() if device.index is None else device.index
    return _device_arch_by_index(index)


@triton.jit
def _stage_rhs(b):
    # TF32 WGMMA requires a K-major B descriptor; stage [N, K] in shared memory
    # with trans_b=True so the logical product remains A[M, K] @ B[K, N].
    return tle.gpu.alloc(
        (b.shape[1], b.shape[0]),
        tl.float32,
        scope=tle.gpu.smem,
        init_value=tl.trans(b),
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
    BV: tl.constexpr,
):
    part, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    num_v_tiles: tl.constexpr = (E + BV - 1) // BV
    chunk, tile = part // num_v_tiles, part % num_v_tiles
    row = tl.arange(0, C)
    t = chunk * C + row
    d = tl.arange(0, D)
    e = tile * BV + tl.arange(0, BV)
    t_mask = t[:, None] < T
    e_mask = e[None, :] < E

    key = tl.load(
        K + batch * SK[0] + head * SK[1] + t[:, None] * SK[2] + d[None, :] * SK[3],
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)
    val = tl.load(
        V + batch * SV[0] + head * SV[1] + t[:, None] * SV[2] + e[None, :] * SV[3],
        mask=t_mask & e_mask,
        other=0.0,
    ).to(tl.float32)
    slope = tl.load(Slope + head * SS).to(tl.float32)
    length = tl.minimum(C, T - chunk * C)
    # Clamp exponent on padded tail rows so 0 * exp(+large) never produces NaN.
    weight = tl.exp(-slope * tl.maximum(length - 1 - row, 0))
    summary = _matmul(tl.trans(key), val * weight[:, None])
    num_chunks: tl.constexpr = (T + C - 1) // C
    off = ((batch * H + head) * num_chunks + chunk) * D * E
    tl.store(Partial + off + d[:, None] * E + e[None, :], summary, mask=e_mask)


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
    BD: tl.constexpr,
    BV: tl.constexpr,
):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    num_v_tiles: tl.constexpr = (E + BV - 1) // BV
    d = (tile // num_v_tiles) * BD + tl.arange(0, BD)
    e = (tile % num_v_tiles) * BV + tl.arange(0, BV)
    mask = e[None, :] < E
    elem = d[:, None] * E + e[None, :]
    bh = batch * H + head
    chunks: tl.constexpr = (T + C - 1) // C
    snapshots: tl.constexpr = (T + 255) // 256 + 1
    chunks_per_block: tl.constexpr = 256 // C
    prefix = Partial + bh * chunks * D * E + elem
    hist = History + bh * snapshots * D * E + elem

    if HAS_INITIAL:
        state = tl.load(
            Initial
            + batch * SI[0]
            + head * SI[1]
            + d[:, None] * SI[2]
            + e[None, :] * SI[3],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    else:
        state = tl.zeros((BD, BV), dtype=tl.float32)

    slope = tl.load(Slope + head * SS).to(tl.float32)
    boundary_state = state
    decay = tl.exp(-slope)
    for i in range(chunks):
        summary = tl.load(prefix + i * D * E, mask=mask, other=0.0)
        pos = (i % chunks_per_block) * C
        if pos == 0:
            boundary_state = state
            carry = state
            tl.store(hist + (i // chunks_per_block) * D * E, state, mask=mask)
        else:
            # vLLM's 256-token block contract applies exp(-slope * r) (r=0 at block
            # start) to the previous block's state, while intra-block prior chunks
            # carry an extra decay step exp(-slope). Fold both into one carry matrix.
            carry = (
                decay * state + ((1.0 - decay) * tl.exp(-slope * pos)) * boundary_state
            )
        tl.store(prefix + i * D * E, carry, mask=mask)
        state = state * tl.exp(-slope * tl.minimum(C, T - i * C)) + summary
    tl.store(hist + (snapshots - 1) * D * E, state, mask=mask)


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
    # Represent each chunk update as an affine map s' = a * s + summary and scan L
    # chunks in parallel across rows so each CTA streams [L, WIDTH] blocks.
    col = tl.program_id(0) * WIDTH + tl.arange(0, WIDTH)
    col_mask = col < D * E
    head, batch = tl.program_id(1), tl.program_id(2)
    bh = batch * H + head
    chunks: tl.constexpr = (T + C - 1) // C
    snapshots: tl.constexpr = (T + 255) // 256 + 1
    chunks_per_block: tl.constexpr = 256 // C
    prefix = Partial + bh * chunks * D * E
    hist = History + bh * snapshots * D * E

    if HAS_INITIAL:
        base = tl.load(
            Initial
            + batch * SI[0]
            + head * SI[1]
            + (col // E) * SI[2]
            + (col % E) * SI[3],
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        base = tl.zeros((WIDTH,), dtype=tl.float32)

    slope = tl.load(Slope + head * SS).to(tl.float32)
    decay = tl.exp(-slope)
    row = tl.arange(0, L)
    prev_idx = tl.broadcast_to(tl.maximum(row - 1, 0)[:, None], (L, WIDTH))
    boundary_idx = tl.broadcast_to(
        ((row // chunks_per_block) * chunks_per_block)[:, None], (L, WIDTH)
    )
    for start in range((chunks + L - 1) // L):
        i = start * L + row
        ptr = prefix + i[:, None] * D * E + col[None, :]
        mask = (i[:, None] < chunks) & col_mask[None, :]
        summary = tl.load(ptr, mask=mask, other=0.0)
        length = tl.minimum(C, tl.maximum(0, T - i * C))
        a = tl.broadcast_to(tl.exp(-slope * length)[:, None], (L, WIDTH))
        total_a, total_b = tl.associative_scan((a, summary), 0, _compose_decay)
        after = total_b + total_a * base[None, :]
        before = tl.where(
            row[:, None] == 0, base[None, :], tl.gather(after, prev_idx, 0)
        )
        pos = (row % chunks_per_block) * C
        boundary = tl.gather(before, boundary_idx, 0)
        carry = (
            decay * before + ((1.0 - decay) * tl.exp(-slope * pos))[:, None] * boundary
        )
        tl.store(ptr, tl.where(pos[:, None] == 0, before, carry), mask=mask)
        dest = hist + (i // chunks_per_block)[:, None] * D * E + col[None, :]
        tl.store(dest, before, mask=mask & (pos[:, None] == 0))
        base = tl.sum(tl.where(row[:, None] == L - 1, after, 0.0), 0)
    tl.store(hist + (snapshots - 1) * D * E + col, base, mask=col_mask)


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
    BV: tl.constexpr,
    OVERLAP_Q: tl.constexpr,
):
    part, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    num_v_tiles: tl.constexpr = (E + BV - 1) // BV
    chunk, tile = part // num_v_tiles, part % num_v_tiles
    row = tl.arange(0, C)
    t = chunk * C + row
    d = tl.arange(0, D)
    e = tile * BV + tl.arange(0, BV)
    bh = batch * H + head
    t_mask = t[:, None] < T
    e_mask = e[None, :] < E
    elem = d[:, None] * E + e[None, :]

    q = tl.load(
        Q + batch * SQ[0] + head * SQ[1] + t[:, None] * SQ[2] + d[None, :] * SQ[3],
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        K + batch * SK[0] + head * SK[1] + t[:, None] * SK[2] + d[None, :] * SK[3],
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        V + batch * SV[0] + head * SV[1] + t[:, None] * SV[2] + e[None, :] * SV[3],
        mask=t_mask & e_mask,
        other=0.0,
    ).to(tl.float32)
    slope = tl.load(Slope + head * SS).to(tl.float32)

    if SINGLE:
        if HAS_INITIAL:
            state = tl.load(
                Prefix
                + batch * SI[0]
                + head * SI[1]
                + d[:, None] * SI[2]
                + e[None, :] * SI[3],
                mask=e_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            state = tl.zeros((D, BV), dtype=tl.float32)
        hist = History + bh * 2 * D * E + elem
        tl.store(hist, state, mask=e_mask)
        tail_weight = tl.exp(-slope * tl.maximum(T - 1 - row, 0))
        final = (
            _matmul(tl.trans(k), v * tail_weight[:, None], C < 64)
            + tl.exp(-slope * T) * state
        )
        tl.store(hist + D * E, final, mask=e_mask)
    else:
        num_chunks: tl.constexpr = (T + C - 1) // C
        state = tl.load(
            Prefix + (bh * num_chunks + chunk) * D * E + elem,
            mask=e_mask,
            other=0.0,
        )

    age = row[:, None] - row[None, :]
    causal_decay = tl.exp(-slope * tl.maximum(age, 0))
    row_decay = tl.exp(-slope * row)[:, None]
    if OVERLAP_Q:
        # Issue Q @ K^T and Q @ state back-to-back while keeping both SMEM RHS
        # descriptors live until their respective WGMMA commit groups retire.
        keys_rhs = _stage_rhs(tl.trans(k))
        state_rhs = _stage_rhs(state)
        score_acc = _issue_wgmma(q, keys_rhs)
        state_acc = _issue_wgmma(q, state_rhs)
        scores = tle.gpu.wgmma_wait(1, score_acc)
        causal = tl.where(age >= 0, causal_decay * scores, 0.0)
        cross = tle.gpu.wgmma_wait(0, state_acc) * row_decay
        y = _matmul(causal, v) + cross
    else:
        scores = _matmul(q, tl.trans(k), C < 64)
        causal = tl.where(age >= 0, causal_decay * scores, 0.0)
        y = _matmul(causal, v, C < 64)
        if not SINGLE or HAS_INITIAL:
            y += _matmul(q, state, C < 64) * row_decay

    tl.store(Output + (bh * T + t[:, None]) * E + e[None, :], y, mask=t_mask & e_mask)


@triton.jit
def _prefix_tiny_kernel(
    Q,
    K,
    V,
    Slope,
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
    HAS_INITIAL: tl.constexpr,
    BV: tl.constexpr,
):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, D)
    e = tile * BV + tl.arange(0, BV)
    e_mask = e < E
    bh = batch * H + head
    elem = d[:, None] * E + e[None, :]

    if HAS_INITIAL:
        state = tl.load(
            Initial
            + batch * SI[0]
            + head * SI[1]
            + d[:, None] * SI[2]
            + e[None, :] * SI[3],
            mask=e_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        state = tl.zeros((D, BV), dtype=tl.float32)

    if T == 0:
        tl.store(History + bh * D * E + elem, state, mask=e_mask[None, :])
        return

    hist_ptr = History + bh * 2 * D * E + elem
    tl.store(hist_ptr, state, mask=e_mask[None, :])
    q = tl.load(Q + batch * SQ[0] + head * SQ[1] + d * SQ[3]).to(tl.float32)
    k = tl.load(K + batch * SK[0] + head * SK[1] + d * SK[3]).to(tl.float32)
    v = tl.load(
        V + batch * SV[0] + head * SV[1] + e * SV[3], mask=e_mask, other=0.0
    ).to(tl.float32)
    kv = k[:, None] * v[None, :]
    # At block start (r = 0), vLLM's prefix read uses exp(-slope * 0) = 1 on state.
    tl.store(
        Output + bh * E + e, tl.sum(q[:, None] * (state + kv), axis=0), mask=e_mask
    )
    decay = tl.exp(-tl.load(Slope + head * SS).to(tl.float32))
    tl.store(hist_ptr + D * E, decay * state + kv, mask=e_mask[None, :])


@triton.jit
def _decode_kernel(
    Q,
    K,
    V,
    Slope,
    Slots,
    Cache,
    Output,
    H: tl.constexpr,
    D: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SC: tl.constexpr,
    SS: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    BV: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    e = tile * BV + tl.arange(0, BV)
    e_mask = e < D
    out_ptr = Output + (batch * H + head) * D + e
    slot = tl.load(Slots + batch * SLOT_STRIDE).to(tl.int64)
    if slot < 0:
        tl.store(out_ptr, 0.0, mask=e_mask)
        return

    d = tl.arange(0, D)
    offsets = tle.gpu.set_layout(d[:, None] * SC[2] + e[None, :] * SC[3], LAYOUT)
    cache_ptr = Cache + slot * SC[0] + head * SC[1] + offsets

    state = tl.load(cache_ptr, mask=e_mask[None, :], other=0.0).to(tl.float32)
    q = tl.load(Q + batch * SQ[0] + head * SQ[1] + d * SQ[3]).to(tl.float32)
    k = tl.load(K + batch * SK[0] + head * SK[1] + d * SK[3]).to(tl.float32)
    v = tl.load(
        V + batch * SV[0] + head * SV[1] + e * SV[3], mask=e_mask, other=0.0
    ).to(tl.float32)
    decay = tl.exp(-tl.load(Slope + head * SS).to(tl.float32))

    state = decay * state + k[:, None] * v[None, :]
    tl.store(cache_ptr, state, mask=e_mask[None, :])
    tl.store(out_ptr, tl.sum(q[:, None] * state, axis=0), mask=e_mask)


def _select_prefix_plan(
    b: int,
    h: int,
    t: int,
    d: int,
    e: int,
    has_initial: bool,
    device: torch.device,
) -> tuple[int, int, int, bool, int | None]:
    num_sms, l2_bytes = _device_arch(device)
    bh = b * h

    # 1. Sequence chunk length C (derived from Hopper MMA/WGMMA M-dimension):
    #    - t <= 32 (<= WGMMA_M / 2): 64-row WGMMA would waste >= 50% rows; switch to
    #      2-warp TF32 MMA (M >= 16) with tight power-of-two chunk size in {16, 32}.
    #    - 64 < t <= 128: C=64 splits into 2 chunks (3 kernel launches + global prefix
    #      buffer), whereas C=128 fuses into a single kernel launch. Fuse when 1 chunk
    #      at BV=32 fills >= 0.5 SM wave without starving wide-V parallelism.
    #    Note: the 1.5-wave gate for BV=64 is empirically tuned on Hopper to avoid
    #    tail-wave imbalance when halving CTA count from BV=32 to BV=64.
    wide_64_ctas = bh * triton.cdiv(t, _WGMMA_M) * triton.cdiv(e, 64)
    saturates_wide_v = e >= 128 and 2 * wide_64_ctas >= 3 * num_sms
    if t <= _WGMMA_M // 2:
        chunk_size = max(_WARP_MMA_M, triton.next_power_of_2(t))
        num_warps = 2
    elif (
        _WGMMA_M < t <= 2 * _WGMMA_M
        and not saturates_wide_v
        and 2 * bh * triton.cdiv(e, 32) >= num_sms
    ):
        chunk_size = 2 * _WGMMA_M
        num_warps = 4
    else:
        chunk_size = _WGMMA_M
        num_warps = 4

    num_chunks = triton.cdiv(t, chunk_size)

    # 2. Value tile BV in {64, 32, 16} (derived from CTA wave occupancy):
    #    Pick the widest V tile that keeps >= 1 SM wave (or >= 1.5 waves for BV=64)
    #    so Q and K loads are reused across V columns without leaving SMs idle.
    ctas_bv32 = bh * num_chunks * triton.cdiv(e, 32)
    if saturates_wide_v and chunk_size == _WGMMA_M:
        value_tile = 64
    elif chunk_size == 2 * _WGMMA_M or ctas_bv32 >= num_sms:
        value_tile = 32
    else:
        value_tile = 16

    # 3. Overlap Q @ K^T and Q @ state WGMMAs (derived from 1-CTA/SM residency):
    #    Staging two SMEM RHS descriptors doubles peak shared memory; when total CTAs
    #    <= num_sms (at most 1 CTA per SM), this incurs zero SM occupancy loss.
    output_ctas = bh * num_chunks * triton.cdiv(e, value_tile)
    overlap_q = (
        chunk_size == _WGMMA_M
        and (num_chunks > 1 or has_initial)
        and output_ctas <= num_sms
    )

    # 4. Inter-chunk state scan routing (when num_chunks > 1):
    #    - Calculated L2 working-set bound: prefix ([B, H, num_chunks, D, E] FP32)
    #      shares L2 with history snapshots and QKV tiles; once prefix exceeds 30%
    #      of L2 (10 * prefix_bytes >= 3 * l2_bytes), sequential per-chunk pointer
    #      chasing across CTAs spills to HBM, so switch to windowed _scan_parallel.
    #    - Empirical depth bound: when prefix fits in L2, _scan_parallel's O(n / 16)
    #      associative scan overtakes _scan_chunks above _EMPIRICAL_SCAN_CHUNKS (48).
    prefix_bytes = bh * num_chunks * d * e * 4
    if num_chunks > 1 and (
        10 * prefix_bytes >= 3 * l2_bytes or num_chunks > _EMPIRICAL_SCAN_CHUNKS
    ):
        scan_width = 256 if bh * triton.cdiv(d * e, 256) >= num_sms else 64
    else:
        scan_width = None

    return chunk_size, value_tile, num_warps, overlap_q, scan_width


def _select_decode_plan(
    b: int,
    h: int,
    d: int,
    kv_caches: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[int, int, tle.gpu.BlockEncoding]:
    num_sms, _ = _device_arch(device)
    bh = b * h

    # Align thread layout with the contiguous axis of the [slots, H, D, D] cache.
    if kv_caches.stride(2) == 1:
        return min(block_size, 8 if d == 64 else 4), 1, _TRANSPOSED_LAYOUT
    if kv_caches.stride(3) != 1:
        return min(block_size, 16), 2, _NARROW_LAYOUT
    if d == 64:
        return min(block_size, 32), 1, _SMALL_LAYOUT

    # D == 128 contiguous row-major state:
    # _WIDE_LAYOUT spans 8 threads x 4 contiguous cols = 32 cols per warp row (64B
    # BF16 / 128B FP32 coalesced sectors). When wide tiles launch < 0.5 SM wave (or
    # caller requests block_size < 32), split to 16-col 2-warp CTAs to fill SMs.
    wide_tile = min(block_size, 64)
    if wide_tile < 32 or 2 * bh * triton.cdiv(d, wide_tile) < num_sms:
        return min(block_size, 16), 2, _NARROW_LAYOUT
    return wide_tile, 4, _WIDE_LAYOUT


def lightning_attention(q, k, v, ed, block_size=256, kv_history=None):
    """Compute [B, H, T, E] Lightning Attention output and FP32 256-token states."""
    b, h, t, d = q.shape
    e = v.shape[-1]
    if d not in (64, 128):
        raise NotImplementedError(
            f"Hopper Lightning Attention supports head dim 64 or 128, got {d}"
        )

    has_initial = kv_history is not None
    y = torch.empty((b, h, t, e), dtype=q.dtype, device=q.device)
    hist = torch.empty(
        (b, h, triton.cdiv(t, _HISTORY_BLOCK) + 1, d, e),
        dtype=torch.float32,
        device=q.device,
    )
    si = kv_history.stride() if has_initial else (0, 0, 0, 0)
    ss = ed.view(-1).stride(0)

    with torch.cuda.device(q.device):
        # T <= 1 has zero causal token pairs (T * (T - 1) / 2 == 0); execute direct
        # single-step update (or state copy at T == 0) instead of padding to M = 16.
        if t <= 1:
            tiny_bv = 16
            _prefix_tiny_kernel[(triton.cdiv(e, tiny_bv), h, b)](
                q,
                k,
                v,
                ed,
                kv_history,
                y,
                hist,
                h,
                t,
                d,
                e,
                q.stride(),
                k.stride(),
                v.stride(),
                si,
                ss,
                has_initial,
                tiny_bv,
                num_warps=4,
            )
            return y, hist

        chunk_size, value_tile, num_warps, overlap_q, scan_width = _select_prefix_plan(
            b, h, t, d, e, has_initial, q.device
        )
        num_chunks = triton.cdiv(t, chunk_size)
        num_v_tiles = triton.cdiv(e, value_tile)
        prefix = kv_history
        if num_chunks > 1:
            prefix = torch.empty(
                (b, h, num_chunks, d, e), dtype=torch.float32, device=q.device
            )
            _chunk_summary[(num_chunks * num_v_tiles, h, b)](
                k,
                v,
                ed,
                prefix,
                h,
                t,
                d,
                e,
                k.stride(),
                v.stride(),
                ss,
                chunk_size,
                value_tile,
                num_warps=4,
            )
            if scan_width is not None:
                _scan_parallel[(triton.cdiv(d * e, scan_width), h, b)](
                    prefix,
                    kv_history,
                    ed,
                    hist,
                    h,
                    t,
                    d,
                    e,
                    si,
                    ss,
                    has_initial,
                    chunk_size,
                    _SCAN_WINDOW,
                    scan_width,
                    num_warps=4,
                )
            else:
                scan_tiles = triton.cdiv(d, _SCAN_TILE_D) * triton.cdiv(e, _SCAN_TILE_V)
                _scan_chunks[(scan_tiles, h, b)](
                    prefix,
                    kv_history,
                    ed,
                    hist,
                    h,
                    t,
                    d,
                    e,
                    si,
                    ss,
                    has_initial,
                    chunk_size,
                    _SCAN_TILE_D,
                    _SCAN_TILE_V,
                )

        _chunk_output[(num_chunks * num_v_tiles, h, b)](
            q,
            k,
            v,
            ed,
            prefix,
            y,
            hist,
            h,
            t,
            d,
            e,
            q.stride(),
            k.stride(),
            v.stride(),
            si,
            ss,
            num_chunks == 1,
            has_initial,
            chunk_size,
            value_tile,
            overlap_q,
            num_warps=num_warps,
        )
    return y, hist


def lightning_attention_decode(q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE=32):
    """Update active [slots, H, D, D] cache slots in-place and return [B, H * D]."""
    b, h, t, d = q.shape
    assert t == 1 and k.shape == q.shape and v.shape == q.shape
    if d not in (64, 128):
        raise NotImplementedError(
            f"Hopper Lightning Attention supports head dim 64 or 128, got {d}"
        )

    tile, num_warps, layout = _select_decode_plan(
        b, h, d, kv_caches, BLOCK_SIZE, q.device
    )
    out = torch.empty((b, h * d), dtype=q.dtype, device=q.device)
    with torch.cuda.device(q.device):
        _decode_kernel[(triton.cdiv(d, tile), h, b)](
            q,
            k,
            v,
            slope_rate,
            slot_idx,
            kv_caches,
            out,
            h,
            d,
            q.stride(),
            k.stride(),
            v.stride(),
            kv_caches.stride(),
            slope_rate.view(-1).stride(0),
            slot_idx.stride(0),
            tile,
            layout,
            num_warps=num_warps,
        )
    return out

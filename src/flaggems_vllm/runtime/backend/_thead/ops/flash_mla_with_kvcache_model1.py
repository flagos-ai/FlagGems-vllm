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

"""MODEL1 sparse decode with an FP8 KV cache for T-Head PPU (FlagTree TLE).

Three kernels: ``_schedule`` lays every request's 64-token tiles (main topk
tiles, then extra tiles) end to end and splits them over PARTS partitions;
``_attention`` runs one partition per program, keeping an online softmax over
the 64 heads; ``_combine`` merges requests that span several partitions and
applies the attention sink. The schedule is computed on the first call with a
given FlashMLASchedMeta and reused by later calls with the same object.

KV cache layout (MODEL1 FP8): per page, ``page_size`` tokens of 576 bytes
(448 e4m3 NoPE values, then 64 BF16 RoPE values), followed by 8 bytes per
token (7 ue8m0 scales for the seven 64-wide NoPE groups, 1 pad byte).
"""
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

HEADS = tl.constexpr(64)
HEAD_DIM = tl.constexpr(512)
NOPE_DIM = tl.constexpr(448)
TOKEN_BYTES = tl.constexpr(576)
SCALE_BYTES = tl.constexpr(8)
BLOCK_N = tl.constexpr(64)
LN2 = tl.constexpr(0.6931471805599453)
LOG2E = 1.4426950408889634
# Partitions (one program each). On the 810E, a sweep over 16..128 puts the
# best at 20 (or 40).
PARTS = 20
# Cost of starting a request, in tiles, when balancing partitions: a partition
# that also starts a request (Q load, first tile, final write) gets fewer tiles.
SCHEDULE_OVERHEAD = 5
# _combine: heads per program and warps; one head per program parallelizes
# best when a few long requests are spread over all partitions.
COMBINE_HEADS = 1
COMBINE_WARPS = 4


@triton.jit
def _request_tiles(lengths, extra_lengths, b, HAS_EXTRA: tl.constexpr):
    """64-token tiles of request b: its main tiles (at least one), then its extra tiles."""
    tiles = (tl.maximum(tl.load(lengths + b), 1) + BLOCK_N - 1) // BLOCK_N
    if HAS_EXTRA:
        tiles += (tl.maximum(tl.load(extra_lengths + b), 0) + BLOCK_N - 1) // BLOCK_N
    return tiles


@triton.jit
def _schedule(
    lengths,
    extra_lengths,
    metadata,
    split_prefix,
    batch,
    PARTS: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    OVERHEAD: tl.constexpr,
):
    """One program fills the partitions greedily: each request costs its tiles
    plus OVERHEAD, and each partition takes up to an even share of the total.
    Per partition it records the first request, its first tile, the last
    request, its end tile (exclusive), and which split of the first request
    the partition holds; split_prefix numbers every request's splits."""
    total = 0
    for b in range(batch):
        total += _request_tiles(lengths, extra_lengths, b, HAS_EXTRA) + OVERHEAD
    payload = (total + PARTS - 1) // PARTS + OVERHEAD
    tl.store(split_prefix, 0)
    now_idx = 0
    now_block = 0
    now_split = 0
    splits = 0
    for part in range(PARTS):
        base = metadata + part * 8
        tl.store(base, now_idx)
        tl.store(base + 1, now_block)
        tl.store(base + 4, now_split)
        remain = payload
        going = now_idx < batch
        while going:
            rest = (
                _request_tiles(lengths, extra_lengths, now_idx, HAS_EXTRA) - now_block
            )
            if remain >= rest + OVERHEAD:
                splits += now_split + 1
                tl.store(split_prefix + now_idx + 1, splits)
                remain -= rest + OVERHEAD
                now_idx += 1
                now_block = 0
                now_split = 0
                going = now_idx < batch
            else:
                if remain > OVERHEAD:
                    now_block += remain - OVERHEAD
                    now_split += 1
                going = False
        if now_block > 0:
            tl.store(base + 2, now_idx)
            tl.store(base + 3, now_block)
        else:
            tl.store(base + 2, now_idx - 1)
            tl.store(
                base + 3,
                _request_tiles(
                    lengths, extra_lengths, tl.maximum(now_idx - 1, 0), HAS_EXTRA
                ),
            )


@triton.jit
def _tile_tokens(
    t,
    in_range,
    main_tiles,
    main_indices,
    extra_indices,
    length,
    extra_length,
    TOPK: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
):
    """Token ids of tile t (main tiles first, then extra tiles) and which are valid."""
    is_main = t < main_tiles
    position = tl.where(is_main, t, t - main_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    token = tl.load(
        tl.where(is_main, main_indices, extra_indices) + position,
        (position < tl.where(is_main, TOPK, EXTRA_TOPK)) & in_range,
        other=-1,
    )
    valid = (
        (position < tl.where(is_main, length, extra_length)) & (token >= 0) & in_range
    )
    return token, valid


@triton.jit
def _tile_bytes(
    t,
    token,
    valid,
    main_tiles,
    cache,
    page_stride,
    extra_cache,
    extra_page_stride,
    PAGE_SHIFT: tl.constexpr,
    EXTRA_PAGE_SHIFT: tl.constexpr,
):
    """Load tile t's raw bytes: the first 512 bytes of each token (448 e4m3 used),
    its 64 BF16 RoPE values, and its 8 scale bytes as one int64."""
    d = tl.arange(0, HEAD_DIM)
    is_main = t < main_tiles
    # A select between constant page sizes keeps the divisibility of the
    # addresses visible to the compiler, so the loads stay vectorized.
    page_size = tl.where(is_main, 1 << PAGE_SHIFT, 1 << EXTRA_PAGE_SHIFT)
    page = (token >> tl.where(is_main, PAGE_SHIFT, EXTRA_PAGE_SHIFT)).to(tl.int64)
    slot = token & (page_size - 1)
    row = tl.where(is_main, cache, extra_cache) + page * tl.where(
        is_main, page_stride, extra_page_stride
    )
    data = row + slot * TOKEN_BYTES
    raw = tl.load(
        data[:, None] + d[None, :], valid[:, None] & (d[None, :] < NOPE_DIM), other=0
    )
    # BF16 pointer plus element offsets (not byte offsets) so the loads vectorize.
    rope = tl.load(
        (data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))[:, None]
        + tl.arange(0, 64)[None, :],
        valid[:, None],
        other=0.0,
    )
    scales = tl.load(
        (row + TOKEN_BYTES * page_size + slot * SCALE_BYTES).to(
            tl.pointer_type(tl.int64)
        ),
        valid,
        other=0,
    )
    return raw, rope, scales


@triton.jit
def _dequant(raw, rope, scales):
    """[64, 512] BF16 tile: e4m3 NoPE values times their group's ue8m0 scale, then RoPE."""
    g = tl.arange(0, 8)
    exponent = (scales[:, None] >> (8 * g[None, :]).to(tl.int64)) & 255
    scale = (exponent.to(tl.int32) << 23).to(tl.float32, bitcast=True).to(tl.bfloat16)
    scale = tl.reshape(
        tl.broadcast_to(scale[:, :, None], (BLOCK_N, 8, 64)), (BLOCK_N, HEAD_DIM)
    )
    # e4m3 x4 -> bf16 x4 times a power-of-two scale, with integer and bf16x2
    # operations: the 3 mantissa and 4 exponent bits are placed at their BF16
    # positions with the exponent bias raised by 120 (normal values come out
    # exact), subnormals are fixed with min(2x - 2^-6, x), then the scale is
    # applied, the sign is added, and the lanes are put back in order.
    # The backend's own e4m3 conversion is a long software sequence on this
    # target (capability 8.0), and the e4m3x2 cvt instruction is rejected on ppu1.0.
    value = tl.inline_asm_elementwise(
        """{
        .reg .b32 m0, m1, e0, e1, x, y, s0, s1, two, eps, t;
        ppu.mov.b32 two, 0x40004000;
        ppu.mov.b32 eps, 0xBC80BC80;
        ppu.and.b32 m0, $2, 0x00070007;
        ppu.and.b32 m1, $2, 0x07000700;
        ppu.shl.b32 x, m0, 4;
        ppu.shr.b32 y, m1, 4;
        ppu.and.b32 e0, $2, 0x00780078;
        ppu.and.b32 e1, $2, 0x78007800;
        ppu.shl.b32 e0, e0, 4;
        ppu.shr.b32 e1, e1, 4;
        ppu.add.u32 e0, e0, 0x3C003C00;
        ppu.add.u32 e1, e1, 0x3C003C00;
        ppu.or.b32 x, x, e0;
        ppu.or.b32 y, y, e1;
        ppu.fma.rn.bf16x2 t, x, two, eps;
        ppu.min.bf16x2 x, t, x;
        ppu.fma.rn.bf16x2 t, y, two, eps;
        ppu.min.bf16x2 y, t, y;
        ppu.mul.rn.bf16x2 x, x, $3;
        ppu.mul.rn.bf16x2 y, y, $3;
        ppu.and.b32 s0, $2, 0x00800080;
        ppu.and.b32 s1, $2, 0x80008000;
        ppu.shl.b32 s0, s0, 8;
        ppu.or.b32 x, x, s0;
        ppu.or.b32 y, y, s1;
        ppu.prmt.b32 $0, x, y, 0x5410;
        ppu.prmt.b32 $1, x, y, 0x7632;
        }""",
        "=r,=r,r,r,r",
        [raw, scale],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )
    rope = tl.reshape(
        tl.broadcast_to(rope[:, None, :], (BLOCK_N, 8, 64)), (BLOCK_N, HEAD_DIM)
    )
    return tl.where(tl.arange(0, HEAD_DIM)[None, :] < NOPE_DIM, value, rope)


@triton.jit
def _sink_factor(sink, h, lse):
    """exp(lse) / (exp(lse) + exp(sink)), including sink = +-inf and lse = +inf."""
    sink_value = tl.load(sink + h)
    return tl.where(
        sink_value == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink_value - lse))
    )


@triton.jit
def _attention(
    q,
    cache,
    page_stride,
    indices,
    lengths,
    extra_cache,
    extra_page_stride,
    extra_indices,
    extra_lengths,
    metadata,
    split_prefix,
    partial,
    partial_lse,
    out,
    lse,
    sink,
    scale_log2,
    batch,
    TOPK: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    EXTRA_PAGE_SHIFT: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """One partition: for each request in its range, Q goes to shared memory;
    per tile the dequantized KV tile goes to shared memory (row-major), then
    S = Q K^T, online softmax, P through shared memory, O += P V. Tile t+1's
    bytes and tile t+2's token ids are loaded before tile t is computed, so
    those loads are in flight meanwhile."""
    part = tl.program_id(0)
    first = tl.load(metadata + part * 8)
    if first >= batch:
        return
    first_tile = tl.load(metadata + part * 8 + 1)
    last = tl.load(metadata + part * 8 + 2)
    last_tile = tl.load(metadata + part * 8 + 3)
    first_split = tl.load(metadata + part * 8 + 4)
    h = tl.arange(0, HEADS)
    d = tl.arange(0, HEAD_DIM)
    rows = h[:, None] * HEAD_DIM + d[None, :]
    # Row-major swizzled Q and KV tiles (16-byte vectors). The XOR period matters
    # on PPU: the NVIDIA-style 128-byte pattern (8 phases) leaves heavy bank
    # conflicts; measured best are 16 phases for Q and 64 for the KV tile.
    q_shared = tle.gpu.local_ptr(
        tle.gpu.alloc(
            [HEADS, HEAD_DIM],
            dtype=tl.bfloat16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
            layout=tle.gpu.swizzled_shared_layout(
                8, 1, 16, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]
            ),
        )
    )
    kv_shared = tle.gpu.local_ptr(
        tle.gpu.alloc(
            [BLOCK_N, HEAD_DIM],
            dtype=tl.bfloat16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
            layout=tle.gpu.swizzled_shared_layout(
                8, 1, 64, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]
            ),
        )
    )
    p_shared = tle.gpu.local_ptr(
        tle.gpu.alloc([HEADS, BLOCK_N], dtype=tl.bfloat16, scope=tle.gpu.smem)
    )
    for b in range(first, last + 1):
        length = tl.load(lengths + b)
        main_tiles = (tl.maximum(length, 1) + BLOCK_N - 1) // BLOCK_N
        tiles = main_tiles
        extra_length = 0
        if HAS_EXTRA:
            extra_length = tl.load(extra_lengths + b)
            tiles += (tl.maximum(extra_length, 0) + BLOCK_N - 1) // BLOCK_N
        lo = tl.where(b == first, first_tile, 0)
        hi = tl.where(b == last, last_tile, tiles)
        main_ids = indices + b * TOPK
        extra_ids = extra_indices + b * EXTRA_TOPK
        token, valid = _tile_tokens(
            lo,
            True,
            main_tiles,
            main_ids,
            extra_ids,
            length,
            extra_length,
            TOPK,
            EXTRA_TOPK,
        )
        raw, rope, scales = _tile_bytes(
            lo,
            token,
            valid,
            main_tiles,
            cache,
            page_stride,
            extra_cache,
            extra_page_stride,
            PAGE_SHIFT,
            EXTRA_PAGE_SHIFT,
        )
        next_token, next_valid = _tile_tokens(
            lo + 1,
            lo + 1 < hi,
            main_tiles,
            main_ids,
            extra_ids,
            length,
            extra_length,
            TOPK,
            EXTRA_TOPK,
        )
        tl.store(q_shared, tl.load(q + b * HEADS * HEAD_DIM + rows))
        m_i = tl.full((HEADS,), float("-inf"), tl.float32)
        l_i = tl.zeros((HEADS,), tl.float32)
        acc = tl.zeros((HEADS, HEAD_DIM), tl.float32)
        for t in range(lo, hi):
            raw_next, rope_next, scales_next = _tile_bytes(
                t + 1,
                next_token,
                next_valid,
                main_tiles,
                cache,
                page_stride,
                extra_cache,
                extra_page_stride,
                PAGE_SHIFT,
                EXTRA_PAGE_SHIFT,
            )
            token_after, valid_after = _tile_tokens(
                t + 2,
                t + 2 < hi,
                main_tiles,
                main_ids,
                extra_ids,
                length,
                extra_length,
                TOPK,
                EXTRA_TOPK,
            )
            tl.store(kv_shared, _dequant(raw, rope, scales))
            s = tl.dot(tl.load(q_shared), tl.trans(tl.load(kv_shared))) * scale_log2
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(s - m_safe[:, None])
            alpha = tl.exp2(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            # P is staged in shared memory as the A operand of O += P V.
            tl.store(p_shared, p.to(tl.bfloat16))
            acc = acc * alpha[:, None] + tl.dot(tl.load(p_shared), tl.load(kv_shared))
            m_i = m_new
            raw, rope, scales, valid = raw_next, rope_next, scales_next, next_valid
            next_token, next_valid = token_after, valid_after
        nonempty = l_i > 0
        result = acc * tl.where(nonempty, 1.0 / l_i, 0.0)[:, None]
        log2_sum = tl.where(nonempty, m_i + tl.log2(l_i), float("-inf"))
        if (lo == 0) & (hi == tiles):
            # The whole request is here: write the final output.
            natural = tl.where(nonempty, log2_sum * LN2, float("inf"))
            if HAS_SINK:
                result *= _sink_factor(sink, h, natural)[:, None]
            tl.store(out + b * HEADS * HEAD_DIM + rows, result.to(tl.bfloat16))
            tl.store(lse + b * HEADS + h, natural)
        else:
            split = tl.load(split_prefix + b) + tl.where(b == first, first_split, 0)
            tl.store(partial + split * HEADS * HEAD_DIM + rows, result)
            tl.store(partial_lse + split * HEADS + h, log2_sum)


@triton.jit
def _combine(
    partial,
    partial_lse,
    split_prefix,
    sink,
    out,
    lse,
    HAS_SINK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Merge a request's splits (GROUP heads per program), then apply the sink."""
    b = tl.program_id(0)
    first = tl.load(split_prefix + b)
    end = tl.load(split_prefix + b + 1)
    if end - first > 1:
        h = tl.program_id(1) * GROUP + tl.arange(0, GROUP)
        rows = h[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        maximum = tl.full((GROUP,), float("-inf"), tl.float32)
        denominator = tl.zeros((GROUP,), tl.float32)
        result = tl.zeros((GROUP, HEAD_DIM), tl.float32)
        for split in range(first, end):
            value = tl.load(partial_lse + split * HEADS + h)
            new_maximum = tl.maximum(maximum, value)
            safe = tl.where(new_maximum == float("-inf"), 0.0, new_maximum)
            alpha = tl.exp2(maximum - safe)
            weight = tl.exp2(value - safe)
            result = (
                result * alpha[:, None]
                + tl.load(partial + split * HEADS * HEAD_DIM + rows) * weight[:, None]
            )
            denominator = denominator * alpha + weight
            maximum = new_maximum
        valid = denominator > 0
        natural = tl.where(valid, (tl.log2(denominator) + maximum) * LN2, float("inf"))
        factor = tl.where(valid, 1.0 / denominator, 0.0)
        if HAS_SINK:
            factor *= _sink_factor(sink, h, natural)
        tl.store(
            out + b * HEADS * HEAD_DIM + rows,
            (result * factor[:, None]).to(tl.bfloat16),
        )
        tl.store(lse + b * HEADS + h, natural)


def _is_thead_device():
    try:
        from flaggems_vllm import runtime

        return runtime.device.vendor_name == "thead"
    except Exception:
        return False


def _cache_supported(cache):
    page = cache.shape[1]
    return (
        cache.ndim == 4
        and cache.shape[2] == 1
        and cache.shape[3] == 584
        and page & (page - 1) == 0
        and cache.element_size() == 1
        and cache.stride(1) == 584
    )


def can_use_model1_thead(
    q,
    kv,
    indices,
    out,
    lse,
    extra_kv=None,
    extra_indices=None,
    topk_length=None,
    extra_topk_length=None,
):
    """Whether this path can serve the call; callers fall back if not."""
    if not _is_thead_device():
        return False
    if (
        q.dtype != torch.bfloat16
        or q.ndim != 4
        or tuple(q.shape[1:]) != (1, 64, 512)
        or not q.is_contiguous()
    ):
        return False
    if (
        indices is None
        or not indices.is_contiguous()
        or indices.shape[-1] % BLOCK_N.value != 0
    ):
        return False
    if topk_length is None or not _cache_supported(kv):
        return False
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous()
    ):
        return False
    if lse is not None and (lse.dtype != torch.float32 or not lse.is_contiguous()):
        return False
    if extra_kv is not None:
        if (
            extra_indices is None
            or not extra_indices.is_contiguous()
            or extra_topk_length is None
        ):
            return False
        if extra_indices.shape[-1] % BLOCK_N.value != 0 or not _cache_supported(
            extra_kv
        ):
            return False
    return True


def sparse_decode_model1_thead(
    q,
    kv,
    indices,
    *,
    attn_sink,
    topk_length,
    extra_kv,
    extra_indices,
    extra_topk_length,
    out,
    lse,
    sm_scale,
    sched_meta=None,
):
    """Writes out [B, 1, 64, 512] and lse [B, 64, 1]. The tile schedule is kept
    in sched_meta.tile_scheduler_metadata / num_splits and reused while set."""
    batch = q.shape[0]
    has_extra = extra_kv is not None
    kv = kv.view(torch.uint8)
    extra_cache = extra_kv.view(torch.uint8) if has_extra else kv
    extra_indices = extra_indices if has_extra else indices
    extra_lengths = extra_topk_length if has_extra else topk_length
    if sched_meta is not None and sched_meta.tile_scheduler_metadata is not None:
        metadata, split_prefix = (
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
        )
    else:
        metadata = torch.empty((PARTS, 8), dtype=torch.int32, device=q.device)
        split_prefix = torch.empty((batch + 1,), dtype=torch.int32, device=q.device)
        _schedule[(1,)](
            topk_length,
            extra_lengths,
            metadata,
            split_prefix,
            batch,
            PARTS,
            has_extra,
            SCHEDULE_OVERHEAD,
            num_warps=1,
        )
        if sched_meta is not None:
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits = (
                metadata,
                split_prefix,
            )
    # Every partition adds at most one split beyond one per request.
    partial = torch.empty(
        (batch + PARTS, 64, 512), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty((batch + PARTS, 64), dtype=torch.float32, device=q.device)
    sink = attn_sink if attn_sink is not None else lse
    _attention[(PARTS,)](
        q,
        kv,
        kv.stride(0),
        indices,
        topk_length,
        extra_cache,
        extra_cache.stride(0),
        extra_indices,
        extra_lengths,
        metadata,
        split_prefix,
        partial,
        partial_lse,
        out,
        lse,
        sink,
        sm_scale * LOG2E,
        batch,
        indices.shape[-1],
        kv.shape[1].bit_length() - 1,
        has_extra,
        extra_indices.shape[-1],
        extra_cache.shape[1].bit_length() - 1,
        attn_sink is not None,
        num_warps=16,
    )
    _combine[(batch, 64 // COMBINE_HEADS)](
        partial,
        partial_lse,
        split_prefix,
        sink,
        out,
        lse,
        attn_sink is not None,
        COMBINE_HEADS,
        num_warps=COMBINE_WARPS,
    )

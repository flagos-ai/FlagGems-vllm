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

"""MODEL1 sparse decode with an FP8 KV cache for Hygon HCU.

Three kernels: ``_schedule`` lays every request's tiles (main topk tiles, then
extra tiles) end to end and splits them over the partitions; ``_attention``
runs one partition per program, keeping an online softmax over the 64 heads;
``_combine`` merges requests that span several partitions and applies the
attention sink. The schedule is computed on the first call with a given
FlashMLASchedMeta and reused by later calls with the same object.

KV cache layout (MODEL1 FP8): per page, ``page_size`` tokens of 576 bytes
(448 e4m3 NoPE values, then 64 BF16 RoPE values), followed by 8 bytes per
token (7 ue8m0 scales for the seven 64-wide NoPE groups, 1 pad byte).

MFMA reads its operands from registers and a workgroup has 64 KB of LDS, which
a single [64, 512] BF16 Q tile would fill on its own. Q, the KV tile and the
accumulator therefore stay in registers, nothing is carried across the tile
loop, and register pressure is what sets the tile width and the partition
count (see _tile_width and _partition_count).
"""
import torch
import triton
import triton.language as tl

HEADS = tl.constexpr(64)
HEAD_DIM = tl.constexpr(512)
NOPE_DIM = tl.constexpr(448)
TOKEN_BYTES = tl.constexpr(576)
SCALE_BYTES = tl.constexpr(8)
LN2 = tl.constexpr(0.6931471805599453)
LOG2E = 1.4426950408889634

# Compute units on the card, and the two partition counts _partition_count
# chooses between (a sweep over 80..160 puts them here).
COMPUTE_UNITS = 80
SMALL_BATCH_PARTS = 120
LARGE_BATCH_PARTS = 104
# Cost of starting a request, in tiles, when balancing partitions: a partition
# that also starts a request pays for a Q load, a first tile and a final write.
SCHEDULE_OVERHEAD = 5
# Warps per _attention program. A sweep over 4..16 puts the best at 8: fewer
# warps spill registers, more get a worse dot layout from the compiler.
ATTENTION_WARPS = 8
# No software pipelining of the tile loop. This has to be passed explicitly:
# the default would stage a tile across the loop, and registers are the limit.
ATTENTION_STAGES = 1
# _combine: heads per program and warps. One head per program spreads a few
# long requests best but launches batch*64 programs, most of which return at
# once; a sweep over 1..64 puts the best at 4.
COMBINE_HEADS = 4
COMBINE_WARPS = 4


@triton.jit
def _request_tiles(
    lengths, extra_lengths, b, HAS_EXTRA: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Tiles of request b: its main tiles (at least one), then its extra tiles."""
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
    BLOCK_N: tl.constexpr,
):
    """One program fills the partitions greedily: each request costs its tiles
    plus OVERHEAD, and each partition takes up to an even share of the total.
    Per partition it records the first request, its first tile, the last
    request, its end tile (exclusive), and which split of the first request
    the partition holds; split_prefix numbers every request's splits."""
    total = 0
    for b in range(batch):
        total += (
            _request_tiles(lengths, extra_lengths, b, HAS_EXTRA, BLOCK_N) + OVERHEAD
        )
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
                _request_tiles(lengths, extra_lengths, now_idx, HAS_EXTRA, BLOCK_N)
                - now_block
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
                    lengths,
                    extra_lengths,
                    tl.maximum(now_idx - 1, 0),
                    HAS_EXTRA,
                    BLOCK_N,
                ),
            )


@triton.jit
def _tile_tokens(
    t,
    main_tiles,
    main_indices,
    extra_indices,
    length,
    extra_length,
    TOPK: tl.constexpr,
    EXTRA_TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Token ids of tile t (main tiles first, then extra tiles) and which are valid."""
    is_main = t < main_tiles
    position = tl.where(is_main, t, t - main_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    token = tl.load(
        tl.where(is_main, main_indices, extra_indices) + position,
        position < tl.where(is_main, TOPK, EXTRA_TOPK),
        other=-1,
    )
    valid = (position < tl.where(is_main, length, extra_length)) & (token >= 0)
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
    """Tile t's packed bytes: the NoPE values as e4m3, the RoPE values as BF16,
    and the eight scale bytes as one int64."""
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
        data.to(tl.pointer_type(tl.float8e4nv))[:, None] + d[None, :],
        valid[:, None] & (d[None, :] < NOPE_DIM),
        other=0.0,
    )
    # BF16 pointer plus element offsets (not byte offsets) so the loads vectorize.
    rope = tl.load(
        (data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))[:, None]
        + tl.arange(0, 64)[None, :],
        valid[:, None],
        other=0.0,
    )
    # An invalid lane loads raw == 0, which dequantizes to 0 under any scale;
    # 0x0808... just keeps that scale a normal number.
    scales = tl.load(
        (row + TOKEN_BYTES * page_size + slot * SCALE_BYTES).to(
            tl.pointer_type(tl.int64)
        ),
        valid,
        other=0x0808080808080808,
    )
    return raw, rope, scales


@triton.jit
def _dequant(raw, rope, scales, BLOCK_N: tl.constexpr):
    """[BLOCK_N, HEAD_DIM] BF16 tile from one tile's packed bytes."""
    d = tl.arange(0, HEAD_DIM)
    g = tl.arange(0, 8)
    exponent = ((scales[:, None] >> (8 * g[None, :]).to(tl.int64)) & 255).to(tl.int32)
    exponent = tl.reshape(
        tl.broadcast_to(exponent[:, :, None], (BLOCK_N, 8, 64)), (BLOCK_N, HEAD_DIM)
    )
    # A ue8m0 scale is a power of two, so its byte is already an FP32 exponent.
    scale = (exponent << 23).to(tl.float32, bitcast=True).to(tl.bfloat16)
    value = raw.to(tl.bfloat16) * scale
    # Broadcasting RoPE over the eight groups puts its 64 values exactly in the
    # last group, which is where the NoPE part ends.
    rope = tl.reshape(
        tl.broadcast_to(rope[:, None, :], (BLOCK_N, 8, 64)), (BLOCK_N, HEAD_DIM)
    )
    return tl.where(d[None, :] < NOPE_DIM, value, rope)


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
    BLOCK_N: tl.constexpr,
):
    """One partition: for each request in its range, an online softmax over the
    partition's tiles, with Q, the KV tile and the accumulator in registers."""
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
        m_i = tl.full((HEADS,), float("-inf"), tl.float32)
        l_i = tl.zeros((HEADS,), tl.float32)
        # O is accumulated transposed, see the P V dot below.
        acc_t = tl.zeros((HEAD_DIM, HEADS), tl.float32)
        for t in range(lo, hi):
            token, valid = _tile_tokens(
                t,
                main_tiles,
                main_ids,
                extra_ids,
                length,
                extra_length,
                TOPK,
                EXTRA_TOPK,
                BLOCK_N,
            )
            raw, rope, scales = _tile_bytes(
                t,
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
            kv = _dequant(raw, rope, scales, BLOCK_N)
            # The transposed tile is both QK's B operand and P V's A operand,
            # so it is the only thing the compiler has to stage for the dots.
            kv_t = tl.trans(kv)
            # Q is read here rather than held across the loop; the compiler
            # hoists the load and its dot-operand staging out of it.
            q_tile = tl.load(q + b * HEADS * HEAD_DIM + rows)
            # S itself is not transposed, which keeps the softmax reduction
            # along axis 1: MFMA's output layout holds a row's columns within
            # a lane, so reducing over the KV axis stays lane-local.
            s = tl.dot(q_tile, kv_t) * scale_log2
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(s - m_safe[:, None])
            alpha = tl.exp2(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            # Rescaling first lets the MFMA accumulate straight into acc_t,
            # instead of producing a second [HEAD_DIM, HEADS] FP32 tile to add.
            # O^T = V^T P^T also spreads the dot over the warps evenly, which
            # O = P V does not: its M is only HEADS.
            acc_t *= alpha[None, :]
            acc_t = tl.dot(kv_t, tl.trans(p).to(tl.bfloat16), acc_t)
            m_i = m_new
        nonempty = l_i > 0
        factor = tl.where(nonempty, 1.0 / l_i, 0.0)
        log2_sum = tl.where(nonempty, m_i + tl.log2(l_i), float("-inf"))
        natural = tl.where(nonempty, log2_sum * LN2, float("inf"))
        whole_request = (lo == 0) & (hi == tiles)
        if HAS_SINK:
            # Folding the sink into the per-head factor keeps it off the
            # [HEAD_DIM, HEADS] tile. It only applies when this partition owns
            # the whole request; otherwise _combine applies it after merging.
            factor = tl.where(
                whole_request, factor * _sink_factor(sink, h, natural), factor
            )
        # Transposed indices write acc_t out without transposing it back.
        result_t = acc_t * factor[None, :]
        columns = h[None, :] * HEAD_DIM + d[:, None]
        if whole_request:
            tl.store(out + b * HEADS * HEAD_DIM + columns, result_t.to(tl.bfloat16))
            tl.store(lse + b * HEADS + h, natural)
        else:
            split = tl.load(split_prefix + b) + tl.where(b == first, first_split, 0)
            tl.store(partial + split * HEADS * HEAD_DIM + columns, result_t)
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


def _tiles_per_request(topk, extra_topk, has_extra, block_n):
    """Tiles a request takes at this width, counted from topk rather than from
    the live lengths, which only the kernels read."""
    return -(-topk // block_n) + (-(-extra_topk // block_n) if has_extra else 0)


def _tile_width(topk, extra_topk, has_extra):
    """Tile width for this call.

    What decides it is how many tiles a *request* has: with few tiles per
    request the per-request costs dominate and the finer width spreads the
    work over the partitions better, while a request with hundreds of tiles
    wants the per-tile fixed cost halved instead.
    """
    return 16 if _tiles_per_request(topk, extra_topk, has_extra, 16) <= 64 else 32


def _partition_count(batch, tiles, has_extra):
    """How many partitions to cut the work into.

    Two counts bound the useful range. Above ``no_split`` a request no longer
    fits in one partition's payload, so it is split and pays for a combine
    pass; above ``queue`` more partitions hold work than there are compute
    units, so the surplus workgroups wait behind the first wave. Which bound
    binds depends on the batch, hence three cases: too small for a request to
    fit however it is scheduled (go wide, the split is unavoidable), fitting
    within both bounds (a little above the unit count), and large enough that
    keeping requests whole would queue (use the queueing bound).

    With an extra cache a request is hundreds of tiles and both bounds land
    just above the unit count, which leaves nothing to trade: use the count.
    """
    if has_extra:
        return COMPUTE_UNITS
    cost = tiles + SCHEDULE_OVERHEAD
    no_split = batch * cost // tiles
    queue = COMPUTE_UNITS * cost // tiles
    if no_split < COMPUTE_UNITS:
        return SMALL_BATCH_PARTS
    if no_split <= queue:
        return LARGE_BATCH_PARTS
    return queue


def _is_hygon_device():
    try:
        from flaggems_vllm import runtime

        return runtime.device.vendor_name == "hygon"
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


def can_use_model1_hygon(
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
    if not _is_hygon_device():
        return False
    if (
        q.dtype != torch.bfloat16
        or q.ndim != 4
        or tuple(q.shape[1:]) != (1, 64, 512)
        or not q.is_contiguous()
    ):
        return False
    if indices is None or not indices.is_contiguous():
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
            or not _cache_supported(extra_kv)
        ):
            return False
    return True


def sparse_decode_model1_hygon(
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
    topk, extra_topk = indices.shape[-1], extra_indices.shape[-1]
    block_n = _tile_width(topk, extra_topk, has_extra)
    if sched_meta is not None and sched_meta.tile_scheduler_metadata is not None:
        metadata, split_prefix = (
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
        )
        # The cached schedule was built for this many partitions.
        parts = metadata.shape[0]
    else:
        parts = _partition_count(
            batch, _tiles_per_request(topk, extra_topk, has_extra, block_n), has_extra
        )
        metadata = torch.empty((parts, 8), dtype=torch.int32, device=q.device)
        split_prefix = torch.empty((batch + 1,), dtype=torch.int32, device=q.device)
        # The schedule counts tiles, so it has to use the same width.
        _schedule[(1,)](
            topk_length,
            extra_lengths,
            metadata,
            split_prefix,
            batch,
            parts,
            has_extra,
            SCHEDULE_OVERHEAD,
            block_n,
            num_warps=1,
        )
        if sched_meta is not None:
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits = (
                metadata,
                split_prefix,
            )
    # Every partition adds at most one split beyond one per request.
    partial = torch.empty(
        (batch + parts, 64, 512), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty((batch + parts, 64), dtype=torch.float32, device=q.device)
    sink = attn_sink if attn_sink is not None else lse
    _attention[(parts,)](
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
        topk,
        kv.shape[1].bit_length() - 1,
        has_extra,
        extra_topk,
        extra_cache.shape[1].bit_length() - 1,
        attn_sink is not None,
        block_n,
        num_warps=ATTENTION_WARPS,
        num_stages=ATTENTION_STAGES,
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

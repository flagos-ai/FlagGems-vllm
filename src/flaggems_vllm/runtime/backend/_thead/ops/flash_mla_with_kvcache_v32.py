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

"""V32 sparse decode with an FP8 KV cache for T-Head PPU (FlagTree TLE).

Two kernels, over the shared ``combine``: ``_schedule`` lays every request's
64-token tiles end to end and splits them over PARTS partitions, and
``_attention`` runs one partition per program, keeping an online softmax over
the 64 heads. The schedule is computed on the first call with a given
FlashMLASchedMeta and reused by later calls with the same object.

KV cache layout (V32 FP8), per token 656 bytes:

    [  0, 512)  512 e4m3 NoPE values
    [512, 528)  4 float32 scales, one per 128 NoPE dims
    [528, 656)  64 BF16 RoPE values, not quantised

Q carries NoPE and RoPE concatenated (head_dim_k = 576), so
S = Q_nope K_nope^T + Q_rope K_rope^T is two dots, and O += P V reads the NoPE
tile again as V.
"""
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_vllm.runtime.backend._thead.ops.flash_mla_with_kvcache_common import (
    BLOCK_N,
    COMBINE_HEADS,
    COMBINE_WARPS,
    HEADS,
    LN2,
    LOG2E,
    PARTS,
    combine,
    sink_factor,
)

NOPE_DIM = tl.constexpr(512)  # d_v as well: V is the NoPE block
ROPE_DIM = tl.constexpr(64)
TOKEN_BYTES = tl.constexpr(656)
SCALE_OFFSET = tl.constexpr(512)
SCALE_TILES = tl.constexpr(4)  # one float32 scale per 128 NoPE dims
SCALE_TILE = tl.constexpr(128)
ROPE_OFFSET = tl.constexpr(528)
# Cost of starting a request, in tiles, when balancing partitions: a partition
# that also starts a request (Q load, first tile, final write) gets fewer
# tiles. One tile measures best at every batch; with 20 partitions each
# covering several whole requests there is little for a larger charge to
# correct. It has to stay at least 1 for the no-split argument in
# sparse_decode_v32_thead to hold.
SCHEDULE_OVERHEAD = 1


@triton.jit
def _schedule(
    metadata,
    split_prefix,
    batch,
    PARTS: tl.constexpr,
    TILES: tl.constexpr,
    OVERHEAD: tl.constexpr,
):
    """One program fills the partitions greedily: each request costs its tiles
    plus OVERHEAD, and each partition takes up to an even share of the total.
    Per partition it records the first request, its first tile, the last
    request, its end tile (exclusive), and which split of the first request
    the partition holds; split_prefix numbers every request's splits."""
    payload = (batch * (TILES + OVERHEAD) + PARTS - 1) // PARTS + OVERHEAD
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
            rest = TILES - now_block
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
            tl.store(base + 3, TILES)


@triton.jit
def _tile_tokens(t, in_range, indices, TOPK: tl.constexpr):
    """Token ids of tile t and which of them are valid."""
    position = t * BLOCK_N + tl.arange(0, BLOCK_N)
    token = tl.load(indices + position, (position < TOPK) & in_range, other=-1)
    # The range term is redundant -- out of range the load already yields -1 --
    # but keeping it in the mask that guards the tile's three loads is worth 7%:
    # without it the compiler no longer vectorizes them as widely.
    return token, (position < TOPK) & (token >= 0) & in_range


@triton.jit
def _tile_bytes(token, valid, cache, page_stride, PAGE_SHIFT: tl.constexpr):
    """Load a tile's raw bytes: 512 e4m3 NoPE bytes, 4 float32 scales, and 64
    BF16 RoPE values per token."""
    page = (token >> PAGE_SHIFT).to(tl.int64)
    slot = token & ((1 << PAGE_SHIFT) - 1)
    data = cache + page * page_stride + slot * TOKEN_BYTES
    raw = tl.load(
        data[:, None] + tl.arange(0, NOPE_DIM)[None, :], valid[:, None], other=0
    )
    # Typed pointers plus element offsets (not byte offsets) so the loads vectorize.
    scales = tl.load(
        (data + SCALE_OFFSET).to(tl.pointer_type(tl.float32))[:, None]
        + tl.arange(0, SCALE_TILES)[None, :],
        valid[:, None],
        other=0.0,
    )
    rope = tl.load(
        (data + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))[:, None]
        + tl.arange(0, ROPE_DIM)[None, :],
        valid[:, None],
        other=0.0,
    )
    return raw, scales, rope


@triton.jit
def _spread(value):
    """One value per 128-wide scale group to one per NoPE column, as BF16."""
    return tl.reshape(
        tl.broadcast_to(
            value.to(tl.bfloat16)[:, :, None], (BLOCK_N, SCALE_TILES, SCALE_TILE)
        ),
        (BLOCK_N, NOPE_DIM),
    )


@triton.jit
def _dequant(raw, scales, POW2_SCALES: tl.constexpr):
    """[64, 512] BF16 NoPE tile: e4m3 values times their 128-wide group's scale.

    Each scale is split into its power of two, which the exponent arithmetic
    below carries for nothing, and its significand in [1, 2), which needs a
    real multiply over the tile. A ue8m0 scale -- rounded to a power of two,
    what indexer_k_quant_and_cache writes with scale_fmt="ue8m0" -- leaves
    every significand at 1, and POW2_SCALES then drops the multiply, worth 2.7%
    of the call. That has to be a compile-time choice: deciding per tile from
    the tile's own scales either does not compile (make_ttgir rejects the loop)
    or, written so that it does, costs 4.9x by breaking the pipeline, and a
    device-side probe over the scales this call reads costs 7-11% of it.
    """
    bits = scales.to(tl.uint32, bitcast=True)
    pow2 = (bits & 0xFF800000).to(tl.float32, bitcast=True)
    significand = ((bits & 0x007FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    # An e4m3 byte is s.eeee.mmm, and BF16 wants the exponent at bits 14..7 and
    # the mantissa at 6..0, so *both* fields move left by exactly 4: masking
    # bits 6..0 and shifting once places them together. Adding the BF16 pattern
    # of the power of two then raises the exponent by (its exponent + 120),
    # which is the 120 bias e4m3 needs and the scaling by it at once -- hence
    # pow2 * 2**-7 below, 127 - 7 being the bias difference.
    biased = _spread(pow2 * 0.0078125)
    # A subnormal comes out of that as 2**-7 * (1 + m/8) where it should be
    # 2**-6 * m/8, and the second is exactly twice the first less 2**-6, so
    # min(2x - 2**-6, x) picks the right one either way: for a normal value
    # 2x - 2**-6 is above x, for a subnormal below it. The bound is scaled like
    # the values.
    epsilon = _spread(pow2 * -0.015625)
    # The backend's own e4m3 conversion is a long software sequence on this
    # target (capability 8.0), and the e4m3x2 cvt instruction is rejected on
    # ppu1.0. All four values of a pack share one scale, so $3 and $5 serve
    # both halves. x ends up holding the pack's bytes 0 and 2 and y bytes 1 and
    # 3, so the two permutes at the end restore the column order.
    tile = tl.inline_asm_elementwise(
        """{
        .reg .b32 t0, t1, x, y, s0, s1, two, t;
        ppu.mov.b32 two, 0x40004000;
        ppu.and.b32 t0, $2, 0x007F007F;
        ppu.and.b32 t1, $2, 0x7F007F00;
        ppu.shl.b32 x, t0, 4;
        ppu.shr.b32 y, t1, 4;
        ppu.add.u32 x, x, $3;
        ppu.add.u32 y, y, $3;
        ppu.fma.rn.bf16x2 t, x, two, $5;
        ppu.min.bf16x2 x, t, x;
        ppu.fma.rn.bf16x2 t, y, two, $5;
        ppu.min.bf16x2 y, t, y;
        ppu.and.b32 s0, $2, 0x00800080;
        ppu.and.b32 s1, $2, 0x80008000;
        ppu.shl.b32 s0, s0, 8;
        ppu.or.b32 x, x, s0;
        ppu.or.b32 y, y, s1;
        ppu.prmt.b32 $0, x, y, 0x5410;
        ppu.prmt.b32 $1, x, y, 0x7632;
        }""",
        "=r,=r,r,r,r,r,r",
        [raw, biased, epsilon],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )
    if POW2_SCALES:
        return tile
    return tile * _spread(significand)


@triton.jit
def _tile_step(
    raw,
    scales,
    krope,
    valid,
    q_a,
    qr_a,
    m_i,
    l_i,
    acc,
    kv_shared,
    kr_shared,
    p_shared,
    scale_log2,
    POW2_SCALES: tl.constexpr,
):
    """One tile: dequantize into the shared NoPE tile, S = Q K^T over both
    halves, online softmax, then O += P V with V the same tile."""
    tl.store(kv_shared, _dequant(raw, scales, POW2_SCALES))
    tl.store(kr_shared, krope)
    s = tl.dot(q_a, tl.trans(tl.load(kv_shared)))
    s = tl.dot(qr_a, tl.trans(tl.load(kr_shared)), acc=s)
    # S is the cheap place to apply the softmax scale: the dot folds it into
    # its epilogue, while folding it into Q once per request is a real
    # [64, 576] multiply plus BF16 rounding and measures 2% slower.
    s = s * scale_log2
    s = tl.where(valid[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.exp2(s - m_safe[:, None])
    alpha = tl.exp2(m_i - m_safe)
    l_i = l_i * alpha + tl.sum(p, 1)
    # P is staged in shared memory as the A operand of O += P V.
    tl.store(p_shared, p.to(tl.bfloat16))
    acc = acc * alpha[:, None] + tl.dot(tl.load(p_shared), tl.load(kv_shared))
    return m_new, l_i, acc


@triton.jit
def _attention(
    q,
    cache,
    page_stride,
    indices,
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
    TILES: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    HAS_SINK: tl.constexpr,
    WHOLE_REQUESTS: tl.constexpr,
    POW2_SCALES: tl.constexpr,
):
    """One partition: for each request in its range, Q goes to shared memory;
    per tile the dequantized NoPE tile and the raw RoPE tile go to shared
    memory (row-major), then S = Q_nope K_nope^T + Q_rope K_rope^T, online
    softmax, P through shared memory, O += P V with V the NoPE tile. Tile t+1's
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
    nope = tl.arange(0, NOPE_DIM)
    rope = tl.arange(0, ROPE_DIM)
    dqk = NOPE_DIM + ROPE_DIM
    # Row-major swizzled tiles (16-byte vectors). kv_shared is read transposed
    # by the QK dot and directly by the PV dot, so its layout is chosen for the
    # stores and kept for both. The XOR period matters on PPU: the NVIDIA-style
    # 128-byte pattern (8 phases) leaves heavy bank conflicts; measured best are
    # 32 phases for Q and the NoPE tile and 8 for the RoPE tile.
    q_shared = tle.gpu.local_ptr(
        tle.gpu.alloc(
            [HEADS, NOPE_DIM],
            dtype=tl.bfloat16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
            layout=tle.gpu.swizzled_shared_layout(
                8, 1, 32, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]
            ),
        )
    )
    qr_shared = tle.gpu.local_ptr(
        tle.gpu.alloc([HEADS, ROPE_DIM], dtype=tl.bfloat16, scope=tle.gpu.smem)
    )
    kv_shared = tle.gpu.local_ptr(
        tle.gpu.alloc(
            [BLOCK_N, NOPE_DIM],
            dtype=tl.bfloat16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
            layout=tle.gpu.swizzled_shared_layout(
                8, 1, 32, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]
            ),
        )
    )
    kr_shared = tle.gpu.local_ptr(
        tle.gpu.alloc(
            [BLOCK_N, ROPE_DIM],
            dtype=tl.bfloat16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
            layout=tle.gpu.swizzled_shared_layout(
                8, 1, 8, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]
            ),
        )
    )
    p_shared = tle.gpu.local_ptr(
        tle.gpu.alloc([HEADS, BLOCK_N], dtype=tl.bfloat16, scope=tle.gpu.smem)
    )
    for b in range(first, last + 1):
        lo = tl.where(b == first, first_tile, 0)
        hi = tl.where(b == last, last_tile, TILES)
        ids = indices + b * TOPK
        token, valid = _tile_tokens(lo, True, ids, TOPK)
        next_token, next_valid = _tile_tokens(lo + 1, lo + 1 < hi, ids, TOPK)
        raw, scales, krope = _tile_bytes(token, valid, cache, page_stride, PAGE_SHIFT)
        # Q is one contiguous [64, 576] block: NoPE and RoPE go to their tiles.
        tl.store(
            q_shared,
            tl.load(q + b * HEADS * dqk + h[:, None] * dqk + nope[None, :]),
        )
        tl.store(
            qr_shared,
            tl.load(q + b * HEADS * dqk + h[:, None] * dqk + NOPE_DIM + rope[None, :]),
        )
        m_i = tl.full((HEADS,), float("-inf"), tl.float32)
        l_i = tl.zeros((HEADS,), tl.float32)
        acc = tl.zeros((HEADS, NOPE_DIM), tl.float32)
        for t in range(lo, hi):
            # The next tile's bytes are put in flight before this tile is used,
            # and the tile after that's token ids before them: the loop is
            # latency bound on this scattered gather, not bandwidth bound. One
            # tile of lookahead is as deep as it pays to go, two spills enough
            # registers to cost 38%.
            raw_next, scales_next, krope_next = _tile_bytes(
                next_token, next_valid, cache, page_stride, PAGE_SHIFT
            )
            token_after, valid_after = _tile_tokens(t + 2, t + 2 < hi, ids, TOPK)
            m_i, l_i, acc = _tile_step(
                raw,
                scales,
                krope,
                valid,
                tl.load(q_shared),
                tl.load(qr_shared),
                m_i,
                l_i,
                acc,
                kv_shared,
                kr_shared,
                p_shared,
                scale_log2,
                POW2_SCALES,
            )
            raw, scales, krope, valid = raw_next, scales_next, krope_next, next_valid
            next_token, next_valid = token_after, valid_after
        nonempty = l_i > 0
        result = acc * tl.where(nonempty, 1.0 / l_i, 0.0)[:, None]
        log2_sum = tl.where(nonempty, m_i + tl.log2(l_i), float("-inf"))
        rows = h[:, None] * NOPE_DIM + nope[None, :]
        if (lo == 0) & (hi == TILES):
            # The whole request is here: write the final output. acc leaves the
            # dot in an MMA layout and the store wants a blocked one; letting
            # the compiler stage that conversion through its own scratch costs
            # another buffer the size of the KV tile, which fits, and measures
            # 4% less than staging it through the KV tile by hand.
            natural = tl.where(nonempty, log2_sum * LN2, float("inf"))
            if HAS_SINK:
                result *= sink_factor(sink, h, natural)[:, None]
            tl.store(out + b * HEADS * NOPE_DIM + rows, result.to(tl.bfloat16))
            tl.store(lse + b * HEADS + h, natural)
        elif not WHOLE_REQUESTS:
            split = tl.load(split_prefix + b) + tl.where(b == first, first_split, 0)
            tl.store(partial + split * HEADS * NOPE_DIM + rows, result)
            tl.store(partial_lse + split * HEADS + h, log2_sum)


def can_use_v32_thead(q, kv, indices, out, lse, topk_length=None):
    """Whether this path can serve the call, the device aside (the dispatcher
    in flash_mla_with_kvcache.py checks that); callers fall back if not."""
    if (
        q.dtype != torch.bfloat16
        or q.ndim != 4
        or tuple(q.shape[1:]) != (1, 64, 576)
        or not q.is_contiguous()
    ):
        return False
    if (
        indices is None
        or not indices.is_contiguous()
        or indices.shape[-1] % BLOCK_N.value != 0
    ):
        return False
    if topk_length is not None:
        return False
    page = kv.shape[1]
    if (
        kv.ndim != 4
        or kv.shape[2] != 1
        or kv.shape[3] != 656
        or page & (page - 1) != 0
        or kv.element_size() != 1
        or kv.stride(1) != 656
    ):
        return False
    # Both are written by the kernel, and the no-split path uses lse for the
    # partial buffers it does not allocate, so neither may be missing.
    if (
        out is None
        or out.shape != q.shape[:-1] + (NOPE_DIM.value,)
        or out.dtype != q.dtype
        or not out.is_contiguous()
    ):
        return False
    if lse is None or lse.dtype != torch.float32 or not lse.is_contiguous():
        return False
    return True


def sparse_decode_v32_thead(
    q,
    kv,
    indices,
    *,
    attn_sink,
    out,
    lse,
    sm_scale,
    sched_meta=None,
    pow2_kv_scales=False,
):
    """Writes out [B, 1, 64, 512] and lse [B, 64, 1].

    pow2_kv_scales says the cache's scales are all powers of two, as they are
    when the producer quantized with scale_fmt="ue8m0"; it drops a multiply
    worth 2.7% of the call. It defaults to off because the interface carries no
    such promise -- FlagGems-vllm's own generate_v32_fp8_kv_cache writes
    amax / 448 -- and a caller that passes it wrongly gets wrong results, not a
    slower kernel.

    The tile schedule is kept in sched_meta.tile_scheduler_metadata /
    num_splits and reused while set, which is sound here because without
    topk_length it is a function of the batch size and topk alone -- both of
    them pinned for the life of a sched_meta by its Config.
    """
    batch = q.shape[0]
    topk = indices.shape[-1]
    tiles = topk // BLOCK_N.value
    kv = kv.view(torch.uint8)
    if sched_meta is not None and sched_meta.tile_scheduler_metadata is not None:
        metadata, split_prefix = (
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
        )
    else:
        metadata = torch.empty((PARTS, 8), dtype=torch.int32, device=q.device)
        split_prefix = torch.empty((batch + 1,), dtype=torch.int32, device=q.device)
        _schedule[(1,)](
            metadata,
            split_prefix,
            batch,
            PARTS,
            tiles,
            SCHEDULE_OVERHEAD,
            num_warps=1,
        )
        if sched_meta is not None:
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits = (
                metadata,
                split_prefix,
            )
    # Every request is the same number of tiles here, so _schedule charges each
    # one the same tiles + OVERHEAD units and hands each partition
    # ceil(batch * (tiles + OVERHEAD) / PARTS) + OVERHEAD of them. If PARTS
    # divides batch that is exactly batch / PARTS whole requests with OVERHEAD
    # units to spare, and starting one more needs OVERHEAD + 1, so no request
    # is ever split. Worth telling the kernel: the partial write-back and its
    # split_prefix load compile away, and combine, whose programs would each
    # load a split count and return, need not be launched at all.
    whole_requests = batch % PARTS == 0
    if whole_requests:
        # Nothing is written to the partial buffers and nothing reads them, so
        # they are not allocated at all; the kernel still needs two float32
        # pointers for the arguments it compiles away, and lse serves as both.
        partial = partial_lse = lse
    else:
        # Every partition adds at most one split beyond one per request.
        partial = torch.empty(
            (batch + PARTS, 64, 512), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (batch + PARTS, 64), dtype=torch.float32, device=q.device
        )
    sink = attn_sink if attn_sink is not None else lse
    _attention[(PARTS,)](
        q,
        kv,
        kv.stride(0),
        indices,
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
        tiles,
        kv.shape[1].bit_length() - 1,
        attn_sink is not None,
        whole_requests,
        pow2_kv_scales,
        num_warps=16,
    )
    if not whole_requests:
        combine[(batch, 64 // COMBINE_HEADS)](
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

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

Three kernels: ``_schedule`` lays every request's 64-token tiles end to end and
splits them over PARTS partitions; ``_attention`` runs one partition per
program, keeping an online softmax over the 64 heads; ``_combine`` merges
requests that span several partitions and applies the attention sink. The
schedule is computed on the first call with a given FlashMLASchedMeta and
reused by later calls with the same object.

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

HEADS = tl.constexpr(64)
NOPE_DIM = tl.constexpr(512)  # d_v as well: V is the NoPE block
ROPE_DIM = tl.constexpr(64)
TOKEN_BYTES = tl.constexpr(656)
SCALE_OFFSET = tl.constexpr(512)
SCALE_TILES = tl.constexpr(4)  # one float32 scale per 128 NoPE dims
SCALE_TILE = tl.constexpr(128)
ROPE_OFFSET = tl.constexpr(528)
# KV tokens per tile. A 128-token tile halves the per-tile fixed costs but only
# fits with Q in registers instead of a shared tile, and measures slower both
# ways; 32 needs twice the tiles for the same work and measures twice as slow.
BLOCK_N = tl.constexpr(64)
LN2 = tl.constexpr(0.6931471805599453)
LOG2E = 1.4426950408889634
# Partitions (one program each). On the 810E this dominates every other tuning
# knob: a sweep of 8..128 is good at 20, 40 and 80 and markedly worse at 16,
# 24, 32 and 64, with 20 the best at every batch; below 20 the time scales as
# 1/PARTS. A high-water counter added to _attention for the measurement showed
# the device keeping exactly 20 of its programs resident, and the same 20
# whatever PARTS, the warp count and the shared-memory footprint were.
# Re-measure it if the schedule, the tile size or that footprint changes.
PARTS = 20
# Cost of starting a request, in tiles, when balancing partitions: a partition
# that also starts a request (Q load, first tile, final write) gets fewer
# tiles. One tile measures best at every batch; with 20 partitions each
# covering several whole requests there is little for a larger charge to
# correct. It has to stay at least 1 for the no-split argument in
# sparse_decode_v32_thead to hold.
SCHEDULE_OVERHEAD = 1
# _combine: heads per program and warps; one head per program parallelizes
# best when a few long requests are spread over all partitions.
COMBINE_HEADS = 1
COMBINE_WARPS = 4


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
def _dequant(raw, scales):
    """[64, 512] BF16 NoPE tile: e4m3 values times their 128-wide group's scale.

    The producer rounds every scale to a power of two (ue8m0), so BF16 holds it
    exactly and the exponent arithmetic below is lossless.
    """
    # An e4m3 byte is s.eeee.mmm, and BF16 wants the exponent at bits 14..7 and
    # the mantissa at 6..0, so *both* fields move left by exactly 4: masking
    # bits 6..0 and shifting once places them together. Adding the BF16 pattern
    # of the scale then raises the exponent by (scale's exponent + 120), which
    # is the 120 bias e4m3 needs and the multiplication by the scale at once --
    # hence scale * 2**-7 below, 127 - 7 being the bias difference.
    biased = tl.reshape(
        tl.broadcast_to(
            (scales * 0.0078125).to(tl.bfloat16)[:, :, None],
            (BLOCK_N, SCALE_TILES, SCALE_TILE),
        ),
        (BLOCK_N, NOPE_DIM),
    )
    # A subnormal comes out of that as 2**-7 * (1 + m/8) where it should be
    # 2**-6 * m/8, and the second is exactly twice the first less 2**-6, so
    # min(2x - 2**-6, x) picks the right one either way: for a normal value
    # 2x - 2**-6 is above x, for a subnormal below it. The bound is scaled the
    # same way as the values.
    epsilon = tl.reshape(
        tl.broadcast_to(
            (scales * -0.015625).to(tl.bfloat16)[:, :, None],
            (BLOCK_N, SCALE_TILES, SCALE_TILE),
        ),
        (BLOCK_N, NOPE_DIM),
    )
    # The backend's own e4m3 conversion is a long software sequence on this
    # target (capability 8.0), and the e4m3x2 cvt instruction is rejected on
    # ppu1.0. All four values of a pack share one scale, so $3 and $5 serve
    # both halves. x ends up holding the pack's bytes 0 and 2 and y bytes 1 and
    # 3, so the two permutes at the end restore the column order.
    return tl.inline_asm_elementwise(
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


@triton.jit
def _sink_factor(sink, h, lse):
    """exp(lse) / (exp(lse) + exp(sink)), including sink = +-inf and lse = +inf."""
    sink_value = tl.load(sink + h)
    return tl.where(
        sink_value == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink_value - lse))
    )


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
):
    """One tile: dequantize into the shared NoPE tile, S = Q K^T over both
    halves, online softmax, then O += P V with V the same tile."""
    tl.store(kv_shared, _dequant(raw, scales))
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
                result *= _sink_factor(sink, h, natural)[:, None]
            tl.store(out + b * HEADS * NOPE_DIM + rows, result.to(tl.bfloat16))
            tl.store(lse + b * HEADS + h, natural)
        elif not WHOLE_REQUESTS:
            split = tl.load(split_prefix + b) + tl.where(b == first, first_split, 0)
            tl.store(partial + split * HEADS * NOPE_DIM + rows, result)
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
        rows = h[:, None] * NOPE_DIM + tl.arange(0, NOPE_DIM)[None, :]
        maximum = tl.full((GROUP,), float("-inf"), tl.float32)
        denominator = tl.zeros((GROUP,), tl.float32)
        result = tl.zeros((GROUP, NOPE_DIM), tl.float32)
        for split in range(first, end):
            value = tl.load(partial_lse + split * HEADS + h)
            new_maximum = tl.maximum(maximum, value)
            safe = tl.where(new_maximum == float("-inf"), 0.0, new_maximum)
            alpha = tl.exp2(maximum - safe)
            weight = tl.exp2(value - safe)
            result = (
                result * alpha[:, None]
                + tl.load(partial + split * HEADS * NOPE_DIM + rows) * weight[:, None]
            )
            denominator = denominator * alpha + weight
            maximum = new_maximum
        valid = denominator > 0
        natural = tl.where(valid, (tl.log2(denominator) + maximum) * LN2, float("inf"))
        factor = tl.where(valid, 1.0 / denominator, 0.0)
        if HAS_SINK:
            factor *= _sink_factor(sink, h, natural)
        tl.store(
            out + b * HEADS * NOPE_DIM + rows,
            (result * factor[:, None]).to(tl.bfloat16),
        )
        tl.store(lse + b * HEADS + h, natural)


def _is_thead_device():
    try:
        from flaggems_vllm import runtime

        return runtime.device.vendor_name == "thead"
    except Exception:
        return False


def can_use_v32_thead(q, kv, indices, out, lse, topk_length=None):
    """Whether this path can serve the call; callers fall back if not."""
    if not _is_thead_device():
        return False
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
    if out is not None and (
        out.shape != q.shape[:-1] + (NOPE_DIM.value,)
        or out.dtype != q.dtype
        or not out.is_contiguous()
    ):
        return False
    if lse is not None and (lse.dtype != torch.float32 or not lse.is_contiguous()):
        return False
    return True


def sparse_decode_v32_thead(
    q, kv, indices, *, attn_sink, out, lse, sm_scale, sched_meta=None
):
    """Writes out [B, 1, 64, 512] and lse [B, 64, 1]. The tile schedule is kept
    in sched_meta.tile_scheduler_metadata / num_splits and reused while set."""
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
    # split_prefix load compile away, and _combine, whose programs would each
    # load a split count and return, need not be launched at all.
    whole_requests = batch % PARTS == 0
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
        num_warps=16,
    )
    if not whole_requests:
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

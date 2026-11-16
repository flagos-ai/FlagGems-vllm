# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial
from typing import Callable, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def sparse_int8_pv(
    base,
    valid,
    high,
    low,
    correction,
    scale,
    acc,
    offset: tl.constexpr,
    HeadMajor: tl.constexpr,
):
    value = tl.load(base + offset + tl.arange(0, 128)[None, :], valid[:, None], 0)
    if HeadMajor:
        high_pv = tl.dot(high, value, out_dtype=tl.int32)
        pv = tl.dot(low, value, high_pv * 128, out_dtype=tl.int32)
        return acc * correction[:, None] + pv.to(tl.float32) * (
            scale[:, None] * (1.0 / 128)
        )
    else:
        high_pv = tl.dot(tl.trans(value), high, out_dtype=tl.int32)
        pv = tl.dot(tl.trans(value), low, high_pv * 128, out_dtype=tl.int32)
        return acc * correction[None, :] + pv.to(tl.float32) * (
            scale[None, :] * (1.0 / 128)
        )


@triton.jit
def sparse_int8_attention(
    Q,
    QRope,
    KV,
    KRope,
    QScale,
    KScale,
    Indices,
    Length,
    Sink,
    Output,
    LSE,
    Partial,
    Stats,
    SQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    Tokens: tl.constexpr,
    Scale: tl.constexpr,
    Rope: tl.constexpr,
    HasLength: tl.constexpr,
    HasSink: tl.constexpr,
    Splits: tl.constexpr,
    SplitTiles: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HeadMajor: tl.constexpr,
):
    head_tile, row, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tl.static_assert(not HeadMajor or not Rope)
    heads = head_tile * BH + tl.arange(0, BH)
    lanes = tl.arange(0, BN)
    dims = tl.arange(0, BK)
    vd = tl.arange(0, 128)
    query_row = row * H + heads
    query_scale = tl.load(QScale + query_row) * (Scale * 1.4426950408889634)
    if Rope:
        rd = tl.arange(0, 64)
        qr = tl.load(QRope + query_row[:, None] * 64 + rd[None, :])
    length = tl.minimum(tl.maximum(tl.load(Length + row), 0), K) if HasLength else K
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.full((BH,), 0.0, tl.float32)
    # Bound the live QK register tile and reload V in channel panels.
    AccShape: tl.constexpr = (BH, 128) if HeadMajor else (128, BH)
    TokenAxis: tl.constexpr = 1 if HeadMajor else 0
    acc0 = tl.full(AccShape, 0.0, tl.float32)
    acc1 = tl.full(AccShape, 0.0, tl.float32)
    acc2 = tl.full(AccShape, 0.0, tl.float32)
    acc3 = tl.full(AccShape, 0.0, tl.float32)
    begin = split * SplitTiles * BN
    end = tl.minimum(begin + SplitTiles * BN, length)
    for start in range(begin, end, BN):
        offsets = start + lanes
        token = tl.load(Indices + row * K + offsets, offsets < length, -1)
        valid = (token >= 0) & (token < Tokens)
        token = tl.where(valid, token, 0)
        if Tokens <= 4194304:
            kv_base = KV + token[:, None] * 512
        else:
            kv_base = KV + token[:, None].to(tl.int64) * 512
        qs_ptr = Q + query_row[:, None] * 512
        raw = tl.full((BH, BN) if HeadMajor else (BN, BH), 0, tl.int32)
        for ck in tl.range(0, 512, BK, loop_unroll_factor=1):
            q = tl.load(qs_ptr + ck + dims[None, :])
            k = tl.load(kv_base + ck + dims[None, :], valid[:, None], 0)
            if HeadMajor:
                raw = tl.dot(q, tl.trans(k), raw, out_dtype=tl.int32)
            else:
                raw = tl.dot(k, tl.trans(q), raw, out_dtype=tl.int32)
        score = raw.to(tl.float32)
        if Rope:
            kr = tl.load(
                KRope + token[:, None].to(tl.int64) * 64 + rd[None, :],
                valid[:, None],
                0,
            )
            score += tl.dot(kr, tl.trans(qr), out_dtype=tl.float32)
        ks = tl.load(KScale + token, valid, 0.0)
        if HeadMajor:
            score *= ks[None, :] * query_scale[:, None]
            score = tl.where(valid[None, :], score, -float("inf"))
        else:
            score *= ks[:, None] * query_scale[None, :]
            score = tl.where(valid[:, None], score, -float("inf"))

        next_maximum = tl.maximum(maximum, tl.max(score, TokenAxis))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        if HeadMajor:
            probability = tl.exp2(score - safe_maximum[:, None])
            weight = probability * ks[None, :]
        else:
            probability = tl.exp2(score - safe_maximum[None, :])
            weight = probability * ks[:, None]
        denominator = denominator * correction + tl.sum(probability, TokenAxis)
        weight_scale = tl.maximum(tl.max(weight, TokenAxis) * (1.0 / 127), 1.0e-30)
        if HeadMajor:
            scaled = weight * (1.0 / weight_scale[:, None])
        else:
            scaled = weight * (1.0 / weight_scale[None, :])
        # The FP32 ULP at 65536 is 1/128: split the rounded fixed point into signed bytes.
        quant = (scaled + 65536.0).to(tl.int32, bitcast=True) - 0x47800000
        high_raw = (quant + 64) >> 7
        high = high_raw.to(tl.int8)
        low = (quant - (high_raw << 7)).to(tl.int8)
        acc0 = sparse_int8_pv(
            kv_base, valid, high, low, correction, weight_scale, acc0, 0, HeadMajor
        )
        acc1 = sparse_int8_pv(
            kv_base, valid, high, low, correction, weight_scale, acc1, 128, HeadMajor
        )
        acc2 = sparse_int8_pv(
            kv_base, valid, high, low, correction, weight_scale, acc2, 256, HeadMajor
        )
        acc3 = sparse_int8_pv(
            kv_base, valid, high, low, correction, weight_scale, acc3, 384, HeadMajor
        )
        maximum = next_maximum

    maximum *= 0.6931471805599453
    if Splits == 1:
        logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
        inv = 1.0 / tl.where(denominator > 0, denominator, 1.0)
        if HasSink:
            sink = tl.load(Sink + heads)
            inv *= tl.where(denominator > 0, 1.0 / (1.0 + tl.exp(sink - logsum)), 0.0)
        if HeadMajor:
            ptr = Output + query_row[:, None] * 512 + vd[None, :]
            inv = inv[:, None]
        else:
            ptr = Output + query_row[None, :] * 512 + vd[:, None]
            inv = inv[None, :]
        tl.store(ptr, acc0 * inv)
        tl.store(ptr + 128, acc1 * inv)
        tl.store(ptr + 256, acc2 * inv)
        tl.store(ptr + 384, acc3 * inv)
        tl.store(LSE + (row // SQ * H + heads) * SQ + row % SQ, logsum)
    else:
        partial_row = (row * Splits + split) * H + heads
        if HeadMajor:
            ptr = Partial + partial_row[:, None] * 512 + vd[None, :]
        else:
            ptr = Partial + partial_row[None, :] * 512 + vd[:, None]
        tl.store(ptr, acc0)
        tl.store(ptr + 128, acc1)
        tl.store(ptr + 256, acc2)
        tl.store(ptr + 384, acc3)
        tl.store(Stats + partial_row * 2, maximum)
        tl.store(Stats + partial_row * 2 + 1, denominator)


@triton.jit
def merge_sparse_int8_attention(
    Partial,
    Stats,
    Sink,
    Output,
    LSE,
    SQ: tl.constexpr,
    H: tl.constexpr,
    Splits: tl.constexpr,
    HasSink: tl.constexpr,
    SplitBlock: tl.constexpr,
    BV: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    splits = tl.arange(0, SplitBlock)
    channels = tile * BV + tl.arange(0, BV)
    partial_row = (row * Splits + splits) * H + head
    maxima = tl.load(Stats + partial_row * 2, splits < Splits, -float("inf"))
    sums = tl.load(Stats + partial_row * 2 + 1, splits < Splits, 0.0)
    maximum = tl.max(maxima, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(maxima - maximum)
    denominator = tl.sum(sums * weights, 0)
    logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
    inv = 1.0 / tl.where(denominator > 0, denominator, 1.0)
    if HasSink:
        sink = tl.load(Sink + head)
        inv *= tl.where(denominator > 0, 1.0 / (1.0 + tl.exp(sink - logsum)), 0.0)
    partial = tl.load(
        Partial + partial_row[:, None] * 512 + channels[None, :],
        splits[:, None] < Splits,
        0.0,
    )
    output = tl.sum(partial * weights[:, None], 0) * inv
    tl.store(Output + (row * H + head) * 512 + channels, output)
    if tile == 0:
        tl.store(LSE + (row // SQ * H + head) * SQ + row % SQ, logsum)


def prepare_flash_mla_sparse_fwd_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    if q_nope.ndim != 4 or k_cache_lora.ndim != 3 or indices.ndim != 3:
        raise ValueError("Expected rank-four Q, rank-three KV and rank-three indices")
    batch, sq, heads, content_dim = q_nope.shape
    pages = k_cache_lora.shape[0]
    topk = indices.shape[-1]
    if q_rope.ndim != 4 or q_rope.shape[:3] != q_nope.shape[:3]:
        raise ValueError("Q RoPE must match Q NoPE in batch, sequence and heads")
    rope = q_rope.shape[-1]
    if content_dim != 512 or rope not in (0, 64) or heads not in (16, 32, 64, 128):
        raise NotImplementedError("Requires NoPE512, RoPE0/64 and 16/32/64/128 heads")
    if sq == 0 or k_cache_lora.shape != (pages, 64, 512):
        raise ValueError("Requires positive SQ and a cache with page size 64")
    if k_cache_rope.shape != (pages, 64, rope):
        raise ValueError("KV RoPE must match the cache pages and Q RoPE dimension")
    if indices.shape != (batch, sq, topk) or indices.dtype != torch.int32:
        raise ValueError("Expected int32 indices [B,SQ,TOPK]")
    if q_nope.dtype != torch.int8 or k_cache_lora.dtype != torch.int8:
        raise TypeError("NoPE must be int8")
    if (
        q_rope.dtype not in (torch.bfloat16, torch.float16)
        or k_cache_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE tensors must have the same BF16 or FP16 dtype")
    if (
        q_scale.dtype != torch.float32
        or k_scale.dtype != torch.float32
        or q_scale.shape != (batch, sq, heads, 1)
        or k_scale.shape != (pages, 64, 1)
    ):
        raise ValueError("Expected FP32 scales [B,SQ,H,1] and [P,64,1]")
    rows = batch * sq
    if topk_length is not None and (
        topk_length.dtype != torch.int32 or topk_length.shape != (rows,)
    ):
        raise ValueError("Expected int32 topk_length [B*SQ]")
    if attn_sink is not None and (
        attn_sink.dtype != torch.float32 or attn_sink.shape != (heads,)
    ):
        raise ValueError("Expected FP32 attn_sink [H]")
    device = q_nope.device
    for tensor in (
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        attn_sink,
        topk_length,
    ):
        if tensor is not None and (
            tensor.device != device or not tensor.is_contiguous()
        ):
            raise ValueError("All inputs must be contiguous and on the same device")
    scale = (512 + rope) ** -0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    output = torch.empty_like(q_nope, dtype=q_rope.dtype)
    lse = torch.empty((batch, heads, sq), dtype=torch.float32, device=device)
    if rows == 0:
        return lambda: (output, lse)

    bh = min(heads, 32)
    small_decode = rows <= 32 and heads == 64 and not rope and topk > 128
    head_major = heads == 64 and not rope and (small_decode or rows > 256)
    if small_decode:
        bn, bk, warps = 64, 256 if topk <= 1024 else 128, 4
    elif head_major and topk > 1024:
        # Large batches have enough CTAs without splitting. Wider KV tiles
        # amortize online softmax, and wider QK panels halve the QK loop trips.
        bn, bk, warps = 128, 256, 4
    elif not rope and heads >= 32 and topk <= 1024 and rows <= 128:
        bn, bk, warps = 128, 64, 8
    else:
        bn, bk, warps = 64, 256 if head_major else 128, 4
    tiles = triton.cdiv(topk, bn)
    # Target four CTAs per CU on the 80-CU HCU BW.
    occupancy = triton.cdiv(320, rows * (heads // bh))
    num_splits = min(32, triton.next_power_of_2(occupancy))
    if small_decode and topk <= 1024:
        # Keep at least two KV tiles in most splits to amortize scratch and merge.
        num_splits = min(num_splits, triton.cdiv(tiles, 2))
    split_tiles = max(1, triton.cdiv(tiles, num_splits))
    splits = max(1, triton.cdiv(tiles, split_tiles))
    partial_output = output
    stats = lse
    if splits > 1:
        partial_output = torch.empty(
            (rows, splits, heads, 512), device=device, dtype=torch.float32
        )
        stats = torch.empty(
            (rows, splits, heads, 2), device=device, dtype=torch.float32
        )

    # Keep head tiles for the same sparse indices adjacent in the grid.
    run_attention = partial(
        sparse_int8_attention[(heads // bh, rows, splits)],
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        topk_length,
        attn_sink,
        output,
        lse,
        partial_output,
        stats,
        sq,
        heads,
        topk,
        pages * 64,
        scale,
        rope,
        topk_length is not None,
        attn_sink is not None,
        splits,
        split_tiles,
        bh,
        bn,
        bk,
        head_major,
        num_warps=warps,
        num_stages=1,
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        run_attention()
        if splits > 1:
            merge_sparse_int8_attention[(rows, heads, 2)](
                partial_output,
                stats,
                attn_sink,
                output,
                lse,
                sq,
                heads,
                splits,
                attn_sink is not None,
                triton.next_power_of_2(splits),
                256,
                num_warps=2 if small_decode else 4,
            )
        return output, lse

    return run


def flash_mla_sparse_fwd_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the attention output and natural-log LSE of sparse INT8 decode.

    Q and its per-head scale have leading axes [B, SQ, H]. KV and its
    per-token scale have leading axes [pages, 64]. NoPE is signed INT8 with
    512 channels; RoPE is BF16 or FP16 with 0 or 64 channels, predivided by
    the same positive FP32 scale as NoPE.

    Indices [B, SQ, K] are physical token IDs. Negative and out-of-range
    IDs are ignored; duplicates contribute repeatedly. Optional lengths
    flatten [B, SQ] into row order. The sink affects only the output.
    Empty attention returns zero output and positive-infinite LSE.
    """
    return prepare_flash_mla_sparse_fwd_w8a8_int8(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        softmax_scale,
        attn_sink,
        topk_length,
    )()

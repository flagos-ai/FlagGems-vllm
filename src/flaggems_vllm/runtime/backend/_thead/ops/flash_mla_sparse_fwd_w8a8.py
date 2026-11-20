# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language import core


@core.extern
def _reciprocal(probability_scale, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [probability_scale],
        {(core.float32,): ("__builtin_ppu_rcpf", core.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


P15_NORMALIZED_PACKED_ASM = tl.constexpr(
    """{
.reg .s32 quantized<4>;
.reg .f32 rounded<4>;
.reg .b32 pair0, pair1;
ppu.fma.rn.f32 rounded0, $2, 0f43800000, 0f3F000000;
ppu.cvt.rmi.s32.f32 quantized0, rounded0;
ppu.fma.rn.f32 rounded1, $3, 0f43800000, 0f3F000000;
ppu.cvt.rmi.s32.f32 quantized1, rounded1;
ppu.fma.rn.f32 rounded2, $4, 0f43800000, 0f3F000000;
ppu.cvt.rmi.s32.f32 quantized2, rounded2;
ppu.fma.rn.f32 rounded3, $5, 0f43800000, 0f3F000000;
ppu.cvt.rmi.s32.f32 quantized3, rounded3;
ppu.prmt.b32 pair0, quantized0, quantized1, 0x5410;
ppu.prmt.b32 pair1, quantized2, quantized3, 0x5410;
ppu.prmt.b32 $1, pair0, pair1, 0x6420;
ppu.add.u32 pair0, pair0, 0x00800080;
ppu.add.u32 pair1, pair1, 0x00800080;
ppu.prmt.b32 $0, pair0, pair1, 0x7531;
}"""
)


@triton.jit
def _sum_and_max(sum_left, max_left, sum_right, max_right):
    return sum_left + sum_right, tl.maximum(max_left, max_right)


@triton.jit
def _sparse_decode(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Lengths,
    Sink,
    Out,
    LSE,
    H: tl.constexpr,
    SQ: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    R: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(1)
    heads = tl.program_id(0) * BH + tl.arange(0, BH)
    qrow = row.to(tl.int64) * H + heads
    dims = tl.arange(0, 512)
    q = tl.load(Q + qrow[:, None] * 512 + dims[None, :], heads[:, None] < H, 0)
    qs = tl.load(QS + qrow, heads < H, 0)
    if R:
        rd = tl.arange(0, R)
        qr = tl.load(QR + qrow[:, None] * R + rd[None, :], heads[:, None] < H, 0)
    length = tl.minimum(K, tl.load(Lengths + row // SQ)) if HAS_LENGTH else K
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denom = tl.full((BH,), 0.0, tl.float32)
    acc = tl.full((BH, 512), 0.0, tl.float32)
    for start in range(0, length, BN):
        col = start + tl.arange(0, BN)
        ids = tl.load(Indices + row.to(tl.int64) * K + col, col < length, -1)
        valid = (ids >= 0) & (ids < TOKENS)
        ids = tl.where(valid, ids, 0)
        if TOKENS > (1 << 31) // 512:
            ids = ids.to(tl.int64)
        kv = tl.load(
            KV + ids[:, None] * 512 + dims[None, :],
            valid[:, None],
            0,
            cache_modifier=".cg",
        )

        logits = tl.dot(q, tl.trans(kv)).to(tl.float32)
        if R:
            kr = tl.load(KR + ids[:, None] * R + rd[None, :], valid[:, None], 0)
            logits += tl.dot(qr, tl.trans(kr))
        ks = tl.load(KS + ids, valid, 0)
        logits = (
            tl.trans(logits) * qs[None, :] * ks[:, None] * (SCALE * 1.4426950408889634)
        )
        logits = tl.where(valid[:, None], logits, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(logits, 0))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        correction = tl.exp2(maximum - safe_max)
        p = tl.exp2(logits - safe_max[None, :])
        weights = p * ks[:, None]
        psum, pmax = tl.reduce((p, weights), 0, _sum_and_max)
        denom = denom * correction + psum
        unit = tl.maximum(pmax * (1.0 / 127.0), 1.0e-30)
        norm = weights * _reciprocal(unit)[None, :]
        # Encode round(256 * norm) as 256 * hi + signed lo (#1241).
        hi, lo = tl.inline_asm_elementwise(
            P15_NORMALIZED_PACKED_ASM,
            "=r,=r,f,f,f,f",
            [norm],
            dtype=(tl.int8, tl.int8),
            is_pure=True,
            pack=4,
        )
        hi_sum = tl.dot(tl.trans(hi), kv)
        pv = tl.dot(tl.trans(lo), kv, hi_sum * 256).to(tl.float32)
        unit *= 1.0 / 256.0
        acc = acc * correction[:, None] + pv * unit[:, None]
        maximum = next_max
    logsum = tl.where(
        denom > 0, maximum * 0.6931471805599453 + tl.log(denom), float("inf")
    )
    inverse = 1.0 / tl.where(denom > 0, denom, 1.0)
    if HAS_SINK:
        sink = tl.load(Sink + heads, heads < H, 0.0)
        inverse *= tl.where(denom > 0, 1.0 / (1.0 + tl.exp(sink - logsum)), 0.0)
    tl.store(
        Out + qrow[:, None] * 512 + dims[None, :],
        acc * inverse[:, None],
        heads[:, None] < H,
    )
    tl.store(LSE + (row // SQ * H + heads) * SQ + row % SQ, logsum, heads < H)


def flash_mla_sparse_fwd_w8a8(
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
):
    """Sparse INT8 decode with scaled NoPE/RoPE and absolute page64 token IDs.

    Invalid IDs are ignored. The optional per-request length bounds each query's
    index list. Both content and RoPE are stored divided by their positive
    finite FP32 scale. Sink logits affect output only. Empty rows return zero
    and +inf. Output uses the RoPE dtype and LSE is FP32 [batch, heads, queries].
    """
    if q_nope.ndim != 4 or k_cache_lora.ndim not in (3, 4) or q_rope.ndim != 4:
        raise ValueError("Expected rank-4 queries and a paged single-head cache")
    batch, queries, heads, width = q_nope.shape
    if queries not in (1, 2) or heads < 1 or width != 512:
        raise NotImplementedError(
            "PPU sparse MLA requires 1 or 2 queries and NoPE width 512"
        )
    rope = q_rope.shape[-1]
    if rope not in (0, 64) or q_rope.shape[:-1] != q_nope.shape[:-1]:
        raise ValueError("Expected matching query rows and RoPE width 0 or 64")
    prefix = k_cache_lora.shape[:-1]
    pages = prefix[0]
    if prefix not in ((pages, 64), (pages, 64, 1)) or k_cache_lora.shape[-1] != 512:
        raise ValueError("Expected a page64, single-head, 512-wide cache")
    if k_cache_rope.shape != (*prefix, rope) or k_scale.shape != (*prefix, 1):
        raise ValueError("Cache RoPE and scales must match the content rows")
    if q_scale.shape != (*q_nope.shape[:-1], 1):
        raise ValueError("Expected one scale per query head")
    if q_nope.dtype != torch.int8 or k_cache_lora.dtype != torch.int8:
        raise TypeError("PPU sparse MLA requires INT8 NoPE")
    if (
        q_rope.dtype not in (torch.bfloat16, torch.float16)
        or k_cache_rope.dtype != q_rope.dtype
    ):
        raise TypeError("Expected matching BF16 or FP16 RoPE")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("Query and cache scales must be FP32")
    if (
        indices.dtype != torch.int32
        or indices.ndim != 3
        or indices.shape[:2] != (batch, queries)
    ):
        raise ValueError("Expected int32 indices [batch, queries, topk]")
    if topk_length is not None and (
        topk_length.dtype != torch.int32 or topk_length.shape != (batch,)
    ):
        raise ValueError("Expected int32 topk_length [batch]")
    if attn_sink is not None and (
        attn_sink.dtype != torch.float32 or attn_sink.shape != (heads,)
    ):
        raise ValueError("Expected FP32 attn_sink [heads]")
    tensors = (
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        topk_length,
        attn_sink,
    )
    if q_nope.device.type != "cuda" or any(
        t.device != q_nope.device or not t.is_contiguous()
        for t in tensors
        if t is not None
    ):
        raise ValueError("Expected contiguous tensors on one PPU")
    out = torch.empty_like(q_nope, dtype=q_rope.dtype)
    lse = torch.empty(
        (batch, heads, queries), device=q_nope.device, dtype=torch.float32
    )
    if batch:
        bh = 64 if heads >= 128 else min(32, max(16, triton.next_power_of_2(heads)))
        bn = 256 if bh == 32 else 128
        scale = (512 + rope) ** -0.5 if softmax_scale is None else float(softmax_scale)
        with torch.cuda.device(q_nope.device):
            _sparse_decode[(triton.cdiv(heads, bh), batch * queries)](
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                q_scale,
                k_scale,
                indices,
                topk_length,
                attn_sink,
                out,
                lse,
                heads,
                queries,
                indices.shape[-1],
                pages * 64,
                rope,
                scale,
                topk_length is not None,
                attn_sink is not None,
                bh,
                bn,
                num_warps=8 if bh == 16 else 16,
                num_stages=1,
            )
    return out, lse

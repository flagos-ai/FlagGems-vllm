# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 sparse GQA over separate block-major K and V views."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_gqa(
    Q,
    K,
    V,
    Ids,
    Table,
    Cu,
    Seq,
    Prefix,
    Out,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    IS: tl.constexpr,
    TS: tl.constexpr,
    OS: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    PS: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    N: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    LIMIT: tl.constexpr,
    SCALE: tl.constexpr,
    DECODE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    kh = head // G
    if DECODE:
        req = row // LIMIT
        length = tl.load(Seq + req * SS)
        last = tl.minimum(tl.maximum(length - LIMIT, 0) + row % LIMIT, length - 1)
        active = True
    else:
        req = tl.full((), 0, tl.int32)
        for b in range(B):
            req += (row >= tl.load(Cu + (b + 1) * CS)).to(tl.int32)
        req = tl.minimum(req, B - 1)
        local = row - tl.load(Cu + req * CS)
        active = (local >= 0) & (local < LIMIT) & (row < tl.load(Cu + (req + 1) * CS))
        last = tl.minimum(
            tl.load(Prefix + req * PS) + local, tl.load(Seq + req * SS) - 1
        )
    d = tl.arange(0, BD)
    q = tl.load(Q + row * QS[0] + head * QS[1] + d * QS[2], d < D, 0).to(tl.float32)
    acc = tl.full((BD,), 0, tl.float32)
    denom = 0.0
    peak = -float("inf")
    for i in range(N):
        block = tl.load(Ids + kh * IS[0] + row * IS[1] + i * IS[2]).to(tl.int64)
        valid = active & (block >= 0) & (block < W) & (block * 128 <= last)
        page = tl.load(Table + req * TS[0] + block * TS[1], valid, -1).to(tl.int64)
        if (page >= 0) & (page < P):
            for j in range(128 // TILE):
                tok = j * TILE + tl.arange(0, TILE)
                visible = block * 128 + tok <= last
                mask = visible[:, None] & (d[None, :] < D)
                key = tl.load(
                    K
                    + page * KS[0]
                    + tok[:, None] * KS[1]
                    + kh * KS[2]
                    + d[None, :] * KS[3],
                    mask,
                    0,
                ).to(tl.float32)
                val = tl.load(
                    V
                    + page * VS[0]
                    + tok[:, None] * VS[1]
                    + kh * VS[2]
                    + d[None, :] * VS[3],
                    mask,
                    0,
                ).to(tl.float32)
                s = tl.where(
                    visible, tl.sum(key * q[None, :], 1) * SCALE, -float("inf")
                )
                m = tl.maximum(peak, tl.max(s, 0))
                shift = tl.where(m == -float("inf"), 0.0, m)
                a = tl.exp(peak - shift)
                w = tl.exp(s - shift)
                acc = acc * a + tl.sum(w[:, None] * val, 0)
                denom = denom * a + tl.sum(w, 0)
                peak = m
    tl.store(
        Out + row * OS[0] + head * OS[1] + d * OS[2],
        acc / tl.maximum(denom, 1.0),
        d < D,
    )


def separate_kv_attention(
    q, cache, ids, table, cu, seq, prefix, limit, kv_heads, scale, out, decode
):
    if not isinstance(cache, (tuple, list)) or len(cache) != 2:
        raise ValueError("Expected separate K and V views")
    k, v = cache
    if q.ndim != 3 or seq.ndim != 1 or table.ndim != 2:
        raise ValueError("Invalid query or request metadata rank")
    rows, heads, dim = q.shape
    batch = seq.numel()
    if type(kv_heads) is not int or kv_heads <= 0 or heads % kv_heads or dim <= 0:
        raise ValueError("Invalid grouped query dimensions")
    if k.ndim != 4 or k.shape != v.shape or k.shape[1:] != (128, kv_heads, dim):
        raise ValueError("Expected [pages, 128, KV heads, head dim] K/V views")
    if (
        out.shape != q.shape
        or table.shape[0] != batch
        or ids.ndim != 3
        or ids.shape[:2] != (kv_heads, rows)
    ):
        raise ValueError("Incompatible output, block table or selected block shape")
    if type(limit) is not int or limit < int(decode):
        raise ValueError("Invalid query length")
    if type(scale) not in (float, int) or not math.isfinite(scale):
        raise ValueError("Softmax scale must be a finite host number")
    if decode and rows != batch * limit:
        raise ValueError("Decode rows must equal batch times query length")
    if not decode and (cu.shape != (batch + 1,) or prefix.shape != seq.shape):
        raise ValueError("Invalid prefill request boundaries")
    metadata = (ids, table, seq) if decode else (ids, table, seq, cu, prefix)
    if any(t.dtype != torch.int32 or t.device != q.device for t in metadata):
        raise ValueError("Expected int32 metadata on the query device")
    if any(
        t.dtype != torch.bfloat16 or t.device != q.device or t.requires_grad
        for t in (q, k, v, out)
    ):
        raise ValueError("Expected inference-only BF16 data on the query device")
    if rows and heads:
        if not batch or not limit:
            out.zero_()
            return
        with torch.npu.device(q.device):
            _sparse_gqa[(rows, heads)](
                q,
                k,
                v,
                ids,
                table,
                cu,
                seq,
                prefix,
                out,
                q.stride(),
                k.stride(),
                v.stride(),
                ids.stride(),
                table.stride(),
                out.stride(),
                cu.stride(0),
                seq.stride(0),
                prefix.stride(0),
                batch,
                heads // kv_heads,
                dim,
                triton.next_power_of_2(dim),
                ids.shape[2],
                table.shape[1],
                k.shape[0],
                limit,
                scale,
                decode,
                16,
            )

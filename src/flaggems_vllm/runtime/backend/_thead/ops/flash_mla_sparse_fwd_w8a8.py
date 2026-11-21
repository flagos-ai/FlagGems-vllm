# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language import core
from triton.runtime.cache import get_cache_manager

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None

from flaggems_vllm.runtime.backend._thead.ops.flash_attention.decode import (
    splitkv_merge_kernel,
)


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
def _store_partition(
    Out,
    Stats,
    row,
    heads,
    dims,
    acc,
    maximum,
    denom,
    ROWS: tl.constexpr,
    H: tl.constexpr,
):
    split = tl.program_id(2)
    dst = ((split * ROWS + row).to(tl.int64) * H + heads) * 512
    tl.store(Out + dst[:, None] + dims[None, :], acc, heads[:, None] < H)
    stat = (split * 2 * H + heads) * ROWS + row
    tl.store(Stats + stat, maximum, heads < H)
    tl.store(Stats + stat + H * ROWS, denom, heads < H)


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
    ROWS: tl.constexpr = 1,
    SPLITS: tl.constexpr = 1,
    REFINE: tl.constexpr = False,
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
    if SPLITS == 1:
        first, end = 0, length
    else:
        tiles = tl.cdiv(length, BN)
        split = tl.program_id(2)
        first = (tiles * split // SPLITS) * BN
        end = (tiles * (split + 1) // SPLITS) * BN
    for start in range(first, end, BN):
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
        if REFINE:
            # Large token scales can amplify a half-step P15 residual.
            residual = (
                norm - (hi.to(tl.float32) + lo.to(tl.float32) / 256.0)
            ) * 65536.0
            fine = tl.minimum(127.0, tl.maximum(-128.0, tl.floor(residual + 0.5))).to(
                tl.int8
            )
            pv += tl.dot(tl.trans(fine), kv).to(tl.float32) / 256.0
        unit *= 1.0 / 256.0
        acc = acc * correction[:, None] + pv * unit[:, None]
        maximum = next_max
    if SPLITS > 1:
        _store_partition(Out, LSE, row, heads, dims, acc, maximum, denom, ROWS, H)
        return
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


@triton.jit
def _fixed_decode(
    Q,
    KV,
    QS,
    KS,
    Ids,
    Out,
    Lse,
    H: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(1)
    heads = tl.program_id(0) * BH + tl.arange(0, BH)
    dims = tl.arange(0, 512)
    cols = tl.arange(0, BN)
    qrow = row.to(tl.int64) * H + heads
    q = tl.load(Q + qrow[:, None] * 512 + dims[None, :])
    qs = tl.load(QS + qrow)
    split = tl.program_id(2)
    tiles = tl.cdiv(K, BN)
    first, end = tiles * split // SPLITS, tiles * (split + 1) // SPLITS
    ids_base = Ids + row.to(tl.int64) * K
    initial = tl.load(ids_base + first * BN + cols, first * BN + cols < K, -1)
    mask = (initial >= 0) & (initial < TOKENS)
    initial_scale = tl.load(KS + tl.where(mask, initial, 0), mask, 0.0)
    anchor = tl.maximum(tl.max(initial_scale, 0), 1.0e-30)
    factor = 32.0 / anchor
    high_sum = tl.full((BH, 512), 0, tl.int32)
    low_sum = tl.full((BH, 512), 0, tl.int32)
    denominator = tl.full((BN, BH), 0.0, tl.float32)
    unsafe = tl.full((BN, BH), False, tl.int1)
    for tile in range(first, end):
        col = tile * BN + cols
        ids = tl.load(ids_base + col, col < K, -1)
        valid = (ids >= 0) & (ids < TOKENS)
        idx = tl.where(valid, ids, 0)
        if TOKENS > (1 << 31) // 512:
            idx = idx.to(tl.int64)
        values = tl.load(
            KV + idx[:, None] * 512 + dims[None, :],
            valid[:, None],
            0,
            cache_modifier=".cg",
        )
        dot = tl.dot(q, tl.trans(values)).to(tl.float32)
        ks = tl.load(KS + idx, valid, 0.0)
        logit = tl.trans(dot) * qs[None, :] * ks[:, None] * (SCALE * 1.4426950408889634)
        prob = tl.where(valid[:, None], tl.exp2(logit), 0.0)
        denominator += prob
        normalized = prob * (ks * factor)[:, None]
        unsafe |= valid[:, None] & ~(
            (normalized <= 127.0)
            & (normalized >= 1.0)
            & (prob < 1.0e16)
            & (prob > 1.0e-16)
        )
        hi, lo = tl.inline_asm_elementwise(
            P15_NORMALIZED_PACKED_ASM,
            "=r,=r,f,f,f,f",
            [normalized],
            dtype=(tl.int8, tl.int8),
            is_pure=True,
            pack=4,
        )
        high_sum = tl.dot(tl.trans(hi), values, high_sum)
        low_sum = tl.dot(tl.trans(lo), values, low_sum)
    total = tl.sum(denominator, 0)
    # Half an encoding step times the largest INT8 magnitude bounds each output component.
    bound = (anchor / 32.0) * (0.25 * BN) * (end - first)
    bad_bound = tl.sum(((total > 0) & (bound > 0.0005 * total)).to(tl.int32), 0) > 0
    if bad_bound or tl.sum(tl.sum(unsafe.to(tl.int32), 0), 0) > 0:
        _sparse_decode(
            Q,
            None,
            KV,
            None,
            QS,
            KS,
            Ids,
            None,
            None,
            Out,
            Lse,
            H,
            1,
            K,
            TOKENS,
            0,
            SCALE,
            False,
            False,
            BH,
            BN,
            ROWS,
            SPLITS,
            REFINE=True,
        )
    else:
        acc = high_sum.to(tl.float32) * 256.0 + low_sum.to(tl.float32)
        if SPLITS > 1:
            _store_partition(
                Out,
                Lse,
                row,
                heads,
                dims,
                acc * (anchor / (32.0 * 256.0)),
                tl.where(total > 0, 0.0, -float("inf")),
                total,
                ROWS,
                H,
            )
        else:
            inv = (anchor / (32.0 * 256.0)) / tl.where(total > 0, total, 1.0)
            tl.store(Out + qrow[:, None] * 512 + dims[None, :], acc * inv[:, None])
            tl.store(Lse + qrow, tl.where(total > 0, tl.log(total), float("inf")))


@lru_cache(maxsize=1)
def _sdk_clang():
    path = Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")) / "bin" / "clang++"
    if not path.is_file():
        return None
    return path, subprocess.check_output([str(path), "--version"])


@lru_cache(maxsize=None)
def _native_library(rope, splits, refine_last=False, short_local=False):
    compiler, version = _sdk_clang()
    source = Path(__file__).with_suffix(".cpp")
    flags = (
        "-target",
        "nvptx64-nvidia-cuda",
        "-x",
        "c++",
        "-std=c++17",
        "-O3",
        "-ffreestanding",
        "-nostdinc",
        "-S",
        "-emit-llvm",
        f"-DROPE={rope}",
        f"-DPARTS={splits}",
        f"-DREFINE_LAST={int(refine_last)}",
        f"-DSHORT_LOCAL={int(short_local)}",
    )
    digest = hashlib.sha256(
        source.read_bytes() + version + str(compiler).encode() + repr(flags).encode()
    ).hexdigest()
    cache = get_cache_manager(digest)
    name = "ppu_sparse_mla.ll"
    path = cache.get_file(name)
    if path is None:
        with tempfile.TemporaryDirectory(prefix="ppu-sparse-mla-") as temp:
            output = Path(temp) / name
            subprocess.run(
                [str(compiler), *flags, str(source), "-o", str(output)], check=True
            )
            path = cache.put(output.read_bytes(), name)
    return {"sparse_mla": path}


@core.extern
def _native_sparse_device(*args, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        list(args),
        {
            (tl.uint64,) * 11
            + (tl.uint32,)
            + (tl.int32,) * 4
            + (tl.float32,): ("ppu_sparse_mla", tl.uint32)
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _native_sparse_kernel(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Ids,
    Out,
    LSE,
    Error,
    Anchor,
    K: tl.constexpr,
    T: tl.constexpr,
    R: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Two KV/RoPE buffers, packed probabilities, scales and warp reduction storage.
    shared_bytes: tl.constexpr = (
        2 * 128 * (512 + 2 * R)
        + 64 * 128
        + 2 * 128 * 4
        + 2 * 4 * 64 * 4
        + (64 * 128 if K == 640 and R == 0 else 0)
    )
    shared = tle.gpu.alloc(
        (triton.next_power_of_2(shared_bytes // 4),),
        tl.int32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    ptr = (
        tle.gpu.local_ptr(shared, (tl.full((), 0, tl.int32),))
        .to(tl.uint64)
        .to(tl.uint32)
    )
    _native_sparse_device(
        Q.to(tl.uint64),
        QR.to(tl.uint64),
        KV.to(tl.uint64),
        KR.to(tl.uint64),
        QS.to(tl.uint64),
        KS.to(tl.uint64),
        Ids.to(tl.uint64),
        Out.to(tl.uint64),
        LSE.to(tl.uint64),
        Error.to(tl.uint64),
        Anchor.to(tl.uint64),
        ptr,
        tl.program_id(1),
        tl.program_id(0),
        tl.full((), K, tl.int32),
        tl.full((), T, tl.int32),
        tl.full((), SCALE * 1.4426950408889634, tl.float32),
    )


@triton.jit
def _selected_scale_max(
    KS, Ids, Anchor, K: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    n = tl.arange(0, BLOCK)
    ids = tl.load(Ids + row * K + n, n < K, -1)
    valid = (ids >= 0) & (ids < T)
    ks = tl.load(KS + tl.where(valid, ids, 0), valid, 0.0)
    peak = tl.maximum(tl.max(ks, 0), 1.0e-30)
    if K == 640:
        # Rare large scales and typical scales both constrain the U8 step.
        count = tl.maximum(tl.sum(valid.to(tl.int32), 0), 1)
        mean = tl.sum(ks, 0) / count
        peak = 224.0 * tl.maximum(peak / 152.0, mean / 124.0)
    tl.store(Anchor + row, peak)


@triton.jit
def _merge_native_partitions(
    Part, Den, PartError, Out, LSE, Error, B: tl.constexpr, S: tl.constexpr
):
    head, row = tl.program_id(0), tl.program_id(1)
    splits = tl.arange(0, triton.next_power_of_2(S))
    d = tl.arange(0, 512)
    pos = (splits * B + row) * 64 + head
    total = tl.sum(tl.load(Den + pos, splits < S, 0.0), 0)
    bound = tl.sum(tl.load(PartError + pos, splits < S, 0.0), 0)
    acc = tl.sum(
        tl.load(
            Part + pos[:, None].to(tl.int64) * 512 + d[None, :],
            splits[:, None] < S,
            0.0,
        ),
        0,
    )
    inv = 1.0 / tl.where(total > 0, total, 1.0)
    tl.store(Out + (row * 64 + head) * 512 + d, acc * inv)
    tl.store(LSE + row * 64 + head, tl.where(total > 0, tl.log(total), float("inf")))
    tl.store(Error + row * 64 + head, bound * inv)


@triton.jit
def _repair_probability_error(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Ids,
    Out,
    LSE,
    Error,
    K: tl.constexpr,
    T: tl.constexpr,
    R: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(1)
    heads = tl.program_id(0) * 16 + tl.arange(0, 16)
    bound = tl.load(Error + row * 64 + heads)
    # NaN bounds, including inf/inf for an overflowing denominator, require rebasing.
    if tl.sum((~(bound <= 0.0015)).to(tl.int32), 0) > 0:
        _sparse_decode(
            Q,
            QR,
            KV,
            KR,
            QS,
            KS,
            Ids,
            None,
            None,
            Out,
            LSE,
            64,
            1,
            K,
            T,
            R,
            SCALE,
            False,
            False,
            16,
            128,
            REFINE=True,
        )


def _launch_native(q, qr, kv, kr, qs, ks, ids, scale, out):
    batch = q.shape[0]
    topk, rope, tokens = ids.shape[-1], qr.shape[-1], kv.shape[0] * 64
    splits = 1
    if batch <= 16 and topk == 8320:
        splits = 2 if 9 <= batch <= 10 else 5
    lse = torch.empty((batch, 64, 1), device=q.device, dtype=torch.float32)
    error = torch.empty_like(lse)
    short_local = batch <= 16 and topk == 640
    local_scale = short_local or (splits > 1 and rope != 0)
    anchor = lse
    if not local_scale:
        anchor = torch.empty((batch,), device=q.device, dtype=torch.float32)
        _selected_scale_max[(batch,)](
            ks, ids, anchor, topk, tokens, triton.next_power_of_2(topk), num_warps=4
        )
    if splits == 1:
        partial, den, partial_error = out, lse, error
    else:
        partial = torch.empty(
            (splits, batch, 64, 512), device=q.device, dtype=torch.float32
        )
        den = torch.empty((splits, batch, 64), device=q.device, dtype=torch.float32)
        partial_error = torch.empty_like(den)
    _native_sparse_kernel[(1, batch * splits)](
        q,
        qr,
        kv,
        kr,
        qs,
        ks,
        ids,
        partial,
        den,
        partial_error,
        anchor,
        topk,
        tokens,
        rope,
        scale,
        num_warps=16,
        num_stages=1,
        extern_libs=_native_library(
            rope, splits, topk == 640 and rope == 0, short_local
        ),
    )
    if splits > 1:
        _merge_native_partitions[(64, batch)](
            partial, den, partial_error, out, lse, error, batch, splits, num_warps=4
        )
    _repair_probability_error[(4, batch)](
        q,
        qr,
        kv,
        kr,
        qs,
        ks,
        ids,
        out,
        lse,
        error,
        topk,
        tokens,
        rope,
        scale,
        num_warps=8,
        num_stages=1,
    )
    return out, lse


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
    scale = (512 + rope) ** -0.5 if softmax_scale is None else float(softmax_scale)
    topk = indices.shape[-1]
    rows = batch * queries
    bh = 64 if heads >= 128 else min(32, max(16, triton.next_power_of_2(heads)))
    # The no-RoPE 32x256 lowering can exceed the shared-memory limit.
    bn = 256 if bh == 32 and rope and topk > 128 else 128
    short_no_rope = bh == 32 and rope == 0 and topk <= 128 and rows >= 16
    warps = 8 if bh == 16 or short_no_rope else 16
    splits = 1
    full_heads = (
        heads == 64 and queries == 1 and attn_sink is None and topk_length is None
    )
    if (
        full_heads
        and 0 < pages * 64 < 2**31
        and q_rope.dtype == torch.bfloat16
        and (
            (0 < batch <= 16 and topk == 8320)
            or (13 <= batch <= 16 and topk == 640)
            or (
                64 <= batch <= 16384
                and (topk in (2048, 8320) or (rope == 0 and topk == 640))
            )
        )
        and all(
            t.data_ptr() % 16 == 0 for t in (q_nope, q_rope, k_cache_lora, k_cache_rope)
        )
        and all(t.data_ptr() % 4 == 0 for t in (q_scale, k_scale, indices))
        and tle is not None
        and torch.cuda.get_device_name(q_nope.device) == "PPU-ZW810E"
        and _sdk_clang() is not None
    ):
        with torch.cuda.device(q_nope.device):
            return _launch_native(
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                q_scale,
                k_scale,
                indices,
                scale,
                out,
            )
    fixed = full_heads and rope == 0 and 512 <= topk <= 32768
    if full_heads and topk >= 512:
        bh, bn, warps = 64, 128, 16
        if batch <= 16 and topk >= 4096:
            splits = 4 if batch <= 12 else 6
        elif rope and topk == 640 and batch <= 12:
            # Five one-tile partitions fill the PPU for small decode batches.
            splits = 2 if batch == 9 else 5
        elif fixed and batch <= 8:
            splits = min(triton.cdiv(topk, bn), 5)
    if splits > 1:
        lse = torch.empty_strided(
            (batch, heads, 1),
            (1, batch, batch * heads),
            device=q_nope.device,
            dtype=torch.float32,
        )
        partial = torch.empty(
            (splits, rows, heads, 512), device=q_nope.device, dtype=torch.float32
        )
        stats = torch.empty(
            (splits, 2, heads, rows), device=q_nope.device, dtype=torch.float32
        )
    else:
        lse = torch.empty(
            (batch, heads, queries), device=q_nope.device, dtype=torch.float32
        )
        partial, stats = out, lse
    if not batch:
        return out, lse
    grid = (triton.cdiv(heads, bh), rows, splits)
    with torch.cuda.device(q_nope.device):
        if fixed:
            # Both INT32 PV sums are bounded for at most 32768 signed INT8 products.
            _fixed_decode[grid](
                q_nope,
                k_cache_lora,
                q_scale,
                k_scale,
                indices,
                partial,
                stats,
                heads,
                topk,
                pages * 64,
                scale,
                bh,
                bn,
                rows,
                splits,
                num_warps=warps,
                num_stages=1,
            )
        else:
            _sparse_decode[grid](
                q_nope,
                q_rope,
                k_cache_lora,
                k_cache_rope,
                q_scale,
                k_scale,
                indices,
                topk_length,
                attn_sink,
                partial,
                stats,
                heads,
                queries,
                topk,
                pages * 64,
                rope,
                scale,
                topk_length is not None,
                attn_sink is not None,
                bh,
                bn,
                rows,
                splits,
                num_warps=warps,
                num_stages=1,
            )
        if splits > 1:
            splitkv_merge_kernel[(rows, heads, 1)](
                partial,
                stats,
                out,
                lse,
                None,
                None,
                rows,
                rows,
                heads,
                512,
                heads * 512,
                512,
                splits,
                triton.next_power_of_2(splits),
                512,
                0,
                num_warps=4,
            )
    return out, lse

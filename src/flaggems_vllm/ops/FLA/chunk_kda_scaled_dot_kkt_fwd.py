# SPDX-License-Identifier: Apache-2.0
"""Independent causal KDA Gram matrices with cumulative log2 gates."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


def _kda_kkt_bench(call, quantiles):
    """Measure tuning candidates with device events, without Graph capture."""
    device = runtime.torch_device_fn
    call()
    device.synchronize()
    samples = []
    for _ in range(5):
        start, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return [samples[round(q * (len(samples) - 1))] for q in quantiles]


@triton.jit
def _kda_kkt_zero(A, QK, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(A + i, 0.0, i < N)
    tl.store(QK + i, 0.0, i < N)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("kda_kkt_pairs"),
    key=["T", "H", "K", "BT", "NSEQ", "VARLEN", "INDEXED", "DIAGONAL"],
)
@triton.jit
def _kda_kkt_pairs(
    Q,
    Kptr,
    G,
    Beta,
    A,
    QK,
    CU,
    CI,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NSEQ: tl.constexpr,
    VARLEN: tl.constexpr,
    INDEXED: tl.constexpr,
    DIAGONAL: tl.constexpr,
    PAIRS: tl.constexpr,
):
    col = tl.program_id(0) * PAIRS + tl.arange(0, PAIRS)
    row = tl.program_id(1)
    group = tl.program_id(2)
    head = group % H
    valid_row = True
    if INDEXED:
        chunk = group // H
        seq = tl.load(CI + chunk * 2).to(tl.int32)
        sub = tl.load(CI + chunk * 2 + 1).to(tl.int32)
        valid_row = (seq >= 0) & (seq < NSEQ) & (sub >= 0)
        bos = tl.load(CU + seq, valid_row, other=0).to(tl.int64)
        eos = tl.load(CU + seq + 1, valid_row, other=0).to(tl.int64)
        pos = bos + sub.to(tl.int64) * BT + row
        local = sub * BT + row
    elif VARLEN:
        pos = row.to(tl.int64)
        lo = 0
        hi = NSEQ
        while lo < hi:
            mid = (lo + hi) // 2
            end = tl.load(CU + mid + 1).to(tl.int64)
            later = pos >= end
            lo = tl.where(later, mid + 1, lo)
            hi = tl.where(later, hi, mid)
        valid_row = lo < NSEQ
        bos = tl.load(CU + lo, valid_row, other=0).to(tl.int64)
        eos = tl.load(CU + lo + 1, valid_row, other=0).to(tl.int64)
        local = pos - bos
    else:
        bos = (group // H).to(tl.int64) * T
        eos = bos + T
        pos = bos + row
        local = row
    valid_row = valid_row & (pos >= bos) & (pos < eos)
    if DIAGONAL:
        col += (local % BT // 16) * 16
    peer = bos + (local // BT) * BT + col
    causal = valid_row & (col < BT) & (col <= local % BT) & (peer < eos)
    d = tl.arange(0, BK)
    offset = (pos * H + head) * K + d
    qi = tl.load(Q + offset, valid_row & (d < K), other=0).to(tl.float32)
    ki = tl.load(Kptr + offset, valid_row & (d < K), other=0).to(tl.float32)
    gi = tl.load(G + offset, valid_row & (d < K), other=0).to(tl.float32)
    other_offset = (peer[:, None] * H + head) * K + d[None, :]
    mask = causal[:, None] & (d[None, :] < K)
    kj = tl.load(Kptr + other_offset, mask, other=0).to(tl.float32)
    gj = tl.load(G + other_offset, mask, other=0).to(tl.float32)
    delta = tl.where(mask, gi[None, :] - gj, 0.0)
    weighted = kj * tl.exp2(delta)
    beta = tl.load(Beta + pos * H + head, valid_row, other=0).to(tl.float32)
    akk = tl.sum(ki[None, :] * weighted, 1) * beta
    aqk = tl.sum(qi[None, :] * weighted, 1) * scale
    out = (pos * H + head) * BT + col
    tl.store(
        A + out, tl.where(causal & (col < local % BT), akk, 0.0), valid_row & (col < BT)
    )
    tl.store(QK + out, tl.where(causal, aqk, 0.0), valid_row & (col < BT))


@triton.jit
def _kda_kkt_chunk(
    CU,
    CI,
    chunk,
    bh,
    H: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    INDEXED: tl.constexpr,
    TIGHT: tl.constexpr,
    NSEQ: tl.constexpr,
    NI: tl.constexpr,
):
    selected = True
    if TIGHT:
        ids = tl.arange(0, triton.next_power_of_2(NSEQ))
        begin = tl.load(CU + ids, ids < NSEQ, other=0).to(tl.int64)
        end = tl.load(CU + ids + 1, ids < NSEQ, other=0).to(tl.int64)
        counts = (end - begin + BT - 1) // BT
        prefix = tl.cumsum(counts, 0)
        seq = tl.sum((chunk >= prefix).to(tl.int32), 0)
        preceding = tl.sum(tl.where(ids < seq, counts, 0), 0)
        chunk = chunk - preceding
        valid = seq < NSEQ
        bos = tl.load(CU + seq, valid, other=0).to(tl.int64)
        eos = tl.load(CU + seq + 1, valid, other=0).to(tl.int64)
        ci = tl.arange(0, triton.next_power_of_2(NI))
        seqs = tl.load(CI + ci * 2, ci < NI, other=-1)
        chunks = tl.load(CI + ci * 2 + 1, ci < NI, other=-1)
        selected = valid & (
            tl.sum(((seqs == seq) & (chunks == chunk)).to(tl.int32), 0) > 0
        )
    elif INDEXED:
        seq = tl.load(CI + chunk * 2).to(tl.int32)
        chunk = tl.load(CI + chunk * 2 + 1).to(tl.int32)
        bos = tl.load(CU + seq).to(tl.int64)
        eos = tl.load(CU + seq + 1).to(tl.int64)
    else:
        bos = (bh // H).to(tl.int64) * T
        eos = bos + T
    return chunk, bos, eos, selected


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("kda_kkt_inter"),
    key=["T", "H", "K", "BT", "INDEXED", "ZERO_UPPER", "TIGHT", "NSEQ", "NI"],
    do_bench=_kda_kkt_bench,
    use_cuda_graph=False,
)
@triton.jit
def _kda_kkt_inter(
    Q,
    Key,
    G,
    Beta,
    A,
    QK,
    CU,
    CI,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    INDEXED: tl.constexpr,
    TIGHT: tl.constexpr,
    NSEQ: tl.constexpr,
    NI: tl.constexpr,
    ZERO_UPPER: tl.constexpr,
    BK: tl.constexpr,
    DP: tl.constexpr,
):
    chunk, tile, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    nc = BT // 16
    ib, jb = tile // nc, tile % nc
    if ib == jb:
        return
    if not ZERO_UPPER:
        if ib < jb:
            return
    head = bh % H
    chunk, bos, eos, selected = _kda_kkt_chunk(
        CU, CI, chunk, bh, H, T, BT, INDEXED, TIGHT, NSEQ, NI
    )
    first = bos + chunk * BT + ib * 16
    if first >= eos:
        return
    rows = first + tl.arange(0, 16)
    cols = bos + chunk * BT + jb * 16 + tl.arange(0, 16)
    if (ib < jb) | (selected == 0):
        out = (rows[:, None] * H + head) * BT + jb * 16 + tl.arange(0, 16)[None, :]
        tl.store(A + out, 0.0, rows[:, None] < eos)
        tl.store(QK + out, 0.0, rows[:, None] < eos)
        return
    acc_k = tl.zeros((16, 16), tl.float32)
    acc_q = tl.zeros((16, 16), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        d = block * BK + tl.arange(0, BK)
        anchor = tl.load(G + (first * H + head) * K + d, d < K, other=0).to(tl.float32)
        ro = (rows[:, None] * H + head) * K + d[None, :]
        co = (cols[None, :] * H + head) * K + d[:, None]
        mask_r = (rows[:, None] < eos) & (d[None, :] < K)
        mask_c = (cols[None, :] < eos) & (d[:, None] < K)
        q = tl.load(Q + ro, mask_r, other=0).to(tl.float32)
        k = tl.load(Key + ro, mask_r, other=0).to(tl.float32)
        gr = tl.load(G + ro, mask_r, other=0).to(tl.float32)
        kc = tl.load(Key + co, mask_c, other=0).to(tl.float32)
        gc = tl.load(G + co, mask_c, other=0).to(tl.float32)
        left = tl.exp2(tl.where(mask_r, gr - anchor[None, :], 0.0))
        right = kc * tl.exp2(tl.where(mask_c, anchor[:, None] - gc, 0.0))
        acc_k += tl.dot(k * left, right, input_precision=DP)
        acc_q += tl.dot(q * left, right, input_precision=DP)
    beta = tl.load(Beta + rows * H + head, rows < eos, other=0).to(tl.float32)
    out = (rows[:, None] * H + head) * BT + jb * 16 + tl.arange(0, 16)[None, :]
    tl.store(A + out, acc_k * beta[:, None], rows[:, None] < eos)
    tl.store(QK + out, acc_q * scale, rows[:, None] < eos)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("kda_kkt_diag"),
    key=["T", "H", "K", "BT", "INDEXED", "TIGHT", "NSEQ", "NI"],
    do_bench=_kda_kkt_bench,
    use_cuda_graph=False,
)
@triton.jit
def _kda_kkt_diag(
    Q,
    Key,
    G,
    Beta,
    A,
    QK,
    CU,
    CI,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    INDEXED: tl.constexpr,
    TIGHT: tl.constexpr,
    NSEQ: tl.constexpr,
    NI: tl.constexpr,
):
    chunk, block, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head = bh % H
    chunk, bos, eos, selected = _kda_kkt_chunk(
        CU, CI, chunk, bh, H, T, BT, INDEXED, TIGHT, NSEQ, NI
    )
    first = bos + chunk * BT + block * 16
    if first >= eos:
        return
    rr = tl.arange(0, 16)
    rows = first + rr
    if selected == 0:
        col = block * 16 + tl.arange(0, 16)
        out = (rows[:, None] * H + head) * BT + col[None, :]
        tl.store(A + out, 0.0, rows[:, None] < eos)
        tl.store(QK + out, 0.0, rows[:, None] < eos)
        return
    d = tl.arange(0, BK)
    offset = (rows[:, None] * H + head) * K + d[None, :]
    mask = (rows[:, None] < eos) & (d[None, :] < K)
    q = tl.load(Q + offset, mask, other=0).to(tl.float32)
    k = tl.load(Key + offset, mask, other=0).to(tl.float32)
    g = tl.load(G + offset, mask, other=0).to(tl.float32)
    beta = tl.load(Beta + rows * H + head, rows < eos, other=0).to(tl.float32)
    for j in range(16):
        peer = first + j
        ko = tl.load(
            Key + (peer * H + head) * K + d, (peer < eos) & (d < K), other=0
        ).to(tl.float32)
        go = tl.load(G + (peer * H + head) * K + d, (peer < eos) & (d < K), other=0).to(
            tl.float32
        )
        causal = (rr >= j) & (peer < eos)
        delta = g - go[None, :]
        weight = ko[None, :] * tl.exp2(delta)
        a = tl.sum(k * weight, 1) * beta
        aqk = tl.sum(q * weight, 1) * scale
        out = (rows * H + head) * BT + block * 16 + j
        tl.store(A + out, tl.where(causal & (rr > j), a, 0.0), rows < eos)
        tl.store(QK + out, tl.where(causal, aqk, 0.0), rows < eos)


def chunk_kda_scaled_dot_kkt_fwd(
    q,
    k,
    gk=None,
    beta=None,
    scale=None,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=64,
    output_dtype=torch.float32,
):
    """Return strict-lower beta*KKT and lower-inclusive scaled QKT.

    q/k/gk have contiguous [B,T,H,K] layout; beta is [B,T,H]. gk
    contains chunk-local cumulative log2 gates, without activation here.
    Packed inputs use B=1 and monotonic cu_seqlens spanning [0,T]. Supplied
    chunk_indices contains unique (sequence, chunk) pairs; omitted chunks
    produce zeros. Without indices, all chunks are computed on the device.
    Inputs are not modified. Backward and non-contiguous inputs are unsupported.
    """
    floats = (torch.float16, torch.bfloat16, torch.float32)
    if gk is None or beta is None:
        raise NotImplementedError("explicit cumulative gk and beta are required")
    if q.ndim != 4 or k.shape != q.shape or gk.shape != q.shape:
        raise ValueError("q, k and gk must share [B,T,H,K] shape")
    b, t, h, dk = q.shape
    if beta.shape != (b, t, h):
        raise ValueError("beta must have [B,T,H] shape")
    if q.dtype not in floats or k.dtype != q.dtype:
        raise NotImplementedError("q/k must share fp16, bf16 or fp32 dtype")
    if gk.dtype not in floats or beta.dtype not in floats or output_dtype not in floats:
        raise NotImplementedError("gk, beta and outputs must be floating point")
    if dk < 1 or dk > 256 or h < 1 or b < 1:
        raise ValueError("requires B,H>0 and 1<=K<=256")
    if not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be a Python integer")
    if scale is not None and not isinstance(scale, (int, float)):
        raise TypeError("scale must be a Python number")
    if chunk_size not in (16, 32, 64, 128):
        raise NotImplementedError("chunk_size must be 16, 32, 64 or 128")
    tensors = (q, k, gk, beta)
    if any(x.device != q.device for x in tensors):
        raise ValueError("all inputs must share a device")
    if any(not x.is_contiguous() for x in tensors):
        raise NotImplementedError("contiguous seq-first inputs are required")
    if q.device.type == "cpu":
        raise NotImplementedError("a Triton accelerator is required")
    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise NotImplementedError("only forward inference is supported")
    nseq = b
    if cu_seqlens is not None:
        if b != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("packed CU requires B=1 and at least two offsets")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be int32 or int64")
        if cu_seqlens.device != q.device or not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous on the input device")
        nseq = cu_seqlens.numel() - 1
    if chunk_indices is not None:
        if cu_seqlens is None:
            raise ValueError("chunk_indices requires cu_seqlens")
        if (
            chunk_indices.ndim != 2
            or chunk_indices.shape[1] != 2
            or chunk_indices.dtype not in (torch.int32, torch.int64)
            or chunk_indices.device != q.device
            or not chunk_indices.is_contiguous()
        ):
            raise ValueError(
                "chunk_indices must be contiguous device int32/int64 [N,2]"
            )
    scale = dk**-0.5 if scale is None else float(scale)
    a = torch.empty((b, t, h, chunk_size), device=q.device, dtype=output_dtype)
    aqk = torch.empty_like(a)
    if not t:
        return a, aqk
    indexed = chunk_indices is not None
    matrix = (
        runtime.device.vendor_name in ("nvidia", "hygon", "metax", "mthreads", "thead")
        and t >= 512
        and (runtime.device.vendor_name == "metax" or b * t * h >= 8192)
        and chunk_size == 64
        and output_dtype == torch.float32
        and gk.dtype == torch.float32
        and beta.dtype == torch.float32
        and (cu_seqlens is None or indexed)
    )
    ni = chunk_indices.shape[0] if indexed else 0
    tight = (
        matrix
        and indexed
        and runtime.device.vendor_name in ("nvidia", "hygon")
        and nseq <= 8
        and 0 < ni <= 64
    )
    zero_upper = tight or (
        matrix
        and not indexed
        and (
            runtime.device.vendor_name in ("nvidia", "mthreads")
            or (runtime.device.vendor_name == "metax" and t <= 1024)
        )
    )
    if (indexed and not tight) or (matrix and not zero_upper):
        # Fixed one-config fill: memory initialization has no tuning search.
        _kda_kkt_zero[(triton.cdiv(a.numel(), 256),)](a, aqk, a.numel(), 256)
        if indexed and chunk_indices.shape[0] == 0:
            return a, aqk
    rows = chunk_size if indexed else t
    groups = chunk_indices.shape[0] * h if indexed else b * h
    if matrix:
        nt = (
            triton.cdiv(t, chunk_size) + nseq - 1
            if tight
            else chunk_indices.shape[0] if indexed else triton.cdiv(t, chunk_size)
        )
        _kda_kkt_inter[(nt, (chunk_size // 16) ** 2, b * h)](
            q,
            k,
            gk,
            beta,
            a,
            aqk,
            cu_seqlens,
            chunk_indices,
            scale,
            t,
            h,
            dk,
            chunk_size,
            indexed,
            tight,
            nseq,
            ni,
            zero_upper,
            DP="ieee",
        )
        # Retain 16 rows across the diagonal-column loop; per-backend singleton
        # configurations come from the measured no-Graph launch sweep.
        _kda_kkt_diag[(nt, chunk_size // 16, b * h)](
            q,
            k,
            gk,
            beta,
            a,
            aqk,
            cu_seqlens,
            chunk_indices,
            scale,
            t,
            h,
            dk,
            chunk_size,
            triton.next_power_of_2(dk),
            indexed,
            tight,
            nseq,
            ni,
        )
        return a, aqk
    width = 16 if matrix else chunk_size
    _kda_kkt_pairs[lambda m: (triton.cdiv(width, m["PAIRS"]), rows, groups)](
        q,
        k,
        gk,
        beta,
        a,
        aqk,
        cu_seqlens,
        chunk_indices,
        scale,
        t,
        h,
        dk,
        chunk_size,
        triton.next_power_of_2(dk),
        nseq,
        cu_seqlens is not None,
        indexed,
        matrix,
    )
    return a, aqk

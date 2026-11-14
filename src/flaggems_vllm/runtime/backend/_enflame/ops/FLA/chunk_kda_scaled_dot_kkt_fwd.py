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
def _kda_kkt_zero(A, QK, N, GRID_X, GRID_XY, BLOCK: tl.constexpr):
    pid = tl.program_id(0) + tl.program_id(1) * GRID_X + tl.program_id(2) * GRID_XY
    i = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(A + i, 0.0, i < N)
    tl.store(QK + i, 0.0, i < N)


# GCU launch limits: grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255


def _launch_grid(total):
    """Spread ``total`` flat program ids across the GCU grid-axis limits."""
    if total <= _GRID_X_MAX:
        return (total, 1, 1)
    grid_y = triton.cdiv(total, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (triton.cdiv(total, grid_y), grid_y, 1)
    grid_z = triton.cdiv(total, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = triton.cdiv(total, _GRID_X_MAX * grid_z)
    return (triton.cdiv(total, grid_y * grid_z), grid_y, grid_z)


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
    GT,
    ROWCAP,
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
    FLAT: tl.constexpr,
):
    if FLAT:
        # Rows/groups exceed the 255-deep grid.y/z limits, so the host moves
        # column blocks to grid.z and packs (group, row tile) into grid.x:
        # GT = ceil(rows / ROWCAP) tiles of ROWCAP rows per group.
        ROWS: tl.constexpr = BT if INDEXED else T
        group = tl.program_id(0) // GT
        row = (tl.program_id(0) % GT) * ROWCAP + tl.program_id(1)
        col = tl.program_id(2) * PAIRS + tl.arange(0, PAIRS)
        valid_row = row < ROWS
    else:
        valid_row = True
        col = tl.program_id(0) * PAIRS + tl.arange(0, PAIRS)
        row = tl.program_id(1)
        group = tl.program_id(2)
    head = group % H
    if INDEXED:
        chunk = group // H
        seq = tl.load(CI + chunk * 2).to(tl.int32)
        sub = tl.load(CI + chunk * 2 + 1).to(tl.int32)
        valid_row = valid_row & (seq >= 0) & (seq < NSEQ) & (sub >= 0)
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
        valid_row = valid_row & (lo < NSEQ)
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
    out = (pos * H + head) * BT + col
    if K > BK:
        # GCU: two concurrent [PAIRS, BK] fp32 tile loads of q/k corrupt
        # above 128 active lanes, so wide fp32 K is processed in <=128-lane
        # blocks accumulated in fp32.
        beta = tl.load(Beta + pos * H + head, valid_row, other=0).to(tl.float32)
        akk = tl.zeros((PAIRS,), tl.float32)
        aqk = tl.zeros((PAIRS,), tl.float32)
        for kb in range(0, K, BK):
            d = kb + tl.arange(0, BK)
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
            akk += tl.sum(ki[None, :] * weighted, 1)
            aqk += tl.sum(qi[None, :] * weighted, 1)
        tl.store(
            A + out,
            tl.where(causal & (col < local % BT), akk * beta, 0.0),
            valid_row & (col < BT),
        )
        tl.store(QK + out, tl.where(causal, aqk * scale, 0.0), valid_row & (col < BT))
    else:
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
        tl.store(
            A + out,
            tl.where(causal & (col < local % BT), akk, 0.0),
            valid_row & (col < BT),
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


# GCU matrix path (single launch): one program per 16x16 output subtile.
# Strictly-lower blocks use the anchor-factorized form
# exp2(g_r - g_a) * exp2(g_a - g_c) with a = row-block first row, so both
# exponents stay <= 0 for causal gates and never overflow (gate extremes);
# diagonal blocks keep the direct per-pair difference; upper blocks store
# zeros so a dense buffer needs no separate fill kernel.
# GCU hazard guards: fp32 tl.dot is single-shot per program (no K-loop
# accumulation, P06) and requires num_warps >= 4 (P07), so the host gates
# this path to dk <= BK; CU/CI are already int32-normalized on the host.
_MAT_CONFIGS = [triton.Config({}, num_warps=4, num_stages=1)]


@libentry()
@libtuner(
    configs=_MAT_CONFIGS,
    key=["T", "H", "K", "BT", "INDEXED", "NSEQ", "NI"],
)
@triton.jit
def _kda_kkt_mat(
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
    NSEQ: tl.constexpr,
    NI: tl.constexpr,
    BK: tl.constexpr,
    DP: tl.constexpr,
):
    chunk, tile, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    nc = BT // 16
    ib = tile // nc
    jb = tile % nc
    head = bh % H
    if INDEXED:
        seq = tl.load(CI + chunk * 2).to(tl.int32)
        chunk = tl.load(CI + chunk * 2 + 1).to(tl.int32)
        if (seq < 0) | (seq >= NSEQ) | (chunk < 0):
            return
        bos = tl.load(CU + seq).to(tl.int32)
        eos = tl.load(CU + seq + 1).to(tl.int32)
    else:
        bos = (bh // H) * T
        eos = bos + T
    first = bos + chunk * BT + ib * 16
    if first >= eos:
        return
    rows = first + tl.arange(0, 16)
    d = tl.arange(0, BK)
    if ib == jb:
        offset = (rows[:, None] * H + head) * K + d[None, :]
        mask = (rows[:, None] < eos) & (d[None, :] < K)
        q = tl.load(Q + offset, mask, other=0).to(tl.float32)
        k = tl.load(Key + offset, mask, other=0).to(tl.float32)
        g = tl.load(G + offset, mask, other=0).to(tl.float32)
        beta = tl.load(Beta + rows * H + head, rows < eos, other=0).to(tl.float32)
        rr = tl.arange(0, 16)
        # Centered-midpoint dot for full 16-row diagonal blocks with bounded
        # within-block gate range: both split exponents stay within +/-W/2
        # (W <= 200 guard), so no intermediate overflow; partial blocks and
        # extreme gates keep the exact direct-difference j-loop.
        dmask = d < K
        gfirst = tl.load(G + (first * H + head) * K + d, dmask, other=0).to(tl.float32)
        glast = tl.load(
            G + ((first + 15) * H + head) * K + d, (first + 15 < eos) & dmask, other=0
        ).to(tl.float32)
        if (first + 15 < eos) & (tl.max(gfirst - glast, 0) <= 200.0):
            mid = (gfirst + glast) * 0.5
            gate = tl.exp2(g - mid[None, :])
            right = tl.trans(k * tl.exp2(mid[None, :] - g))
            acc_k = tl.dot(k * gate, right, input_precision=DP)
            acc_q = tl.dot(q * gate, right, input_precision=DP)
            cc = tl.arange(0, 16)
            out = (rows[:, None] * H + head) * BT + (ib * 16 + cc)[None, :]
            tl.store(
                A + out,
                tl.where(rr[:, None] > cc[None, :], acc_k * beta[:, None], 0.0),
                rows[:, None] < eos,
            )
            tl.store(
                QK + out,
                tl.where(rr[:, None] >= cc[None, :], acc_q * scale, 0.0),
                rows[:, None] < eos,
            )
            return
        for j in range(16):
            peer = first + j
            ko = tl.load(
                Key + (peer * H + head) * K + d, (peer < eos) & (d < K), other=0
            ).to(tl.float32)
            go = tl.load(
                G + (peer * H + head) * K + d, (peer < eos) & (d < K), other=0
            ).to(tl.float32)
            causal = (rr >= j) & (peer < eos)
            weight = ko[None, :] * tl.exp2(g - go[None, :])
            out = (rows * H + head) * BT + ib * 16 + j
            tl.store(
                A + out, tl.where(rr > j, tl.sum(k * weight, 1) * beta, 0.0), rows < eos
            )
            tl.store(
                QK + out,
                tl.where(causal, tl.sum(q * weight, 1) * scale, 0.0),
                rows < eos,
            )
        return
    cols16 = jb * 16 + tl.arange(0, 16)
    out = (rows[:, None] * H + head) * BT + cols16[None, :]
    if ib < jb:
        tl.store(A + out, 0.0, rows[:, None] < eos)
        tl.store(QK + out, 0.0, rows[:, None] < eos)
        return
    cols = bos + chunk * BT + cols16
    anchor = tl.load(G + (first * H + head) * K + d, d < K, other=0).to(tl.float32)
    ro = (rows[:, None] * H + head) * K + d[None, :]
    co = (cols[:, None] * H + head) * K + d[None, :]
    mask_r = (rows[:, None] < eos) & (d[None, :] < K)
    mask_c = (cols[:, None] < eos) & (d[None, :] < K)
    q = tl.load(Q + ro, mask_r, other=0).to(tl.float32)
    k = tl.load(Key + ro, mask_r, other=0).to(tl.float32)
    gr = tl.load(G + ro, mask_r, other=0).to(tl.float32)
    kc = tl.load(Key + co, mask_c, other=0).to(tl.float32)
    gc = tl.load(G + co, mask_c, other=0).to(tl.float32)
    gate = tl.exp2(gr - anchor[None, :])
    left_k = k * gate
    left_q = q * gate
    # GCU trap: a dot B operand whose contraction-axis rows are
    # memory-contiguous (K-contiguous transposed-B form) miscompiles; load the
    # peer tile token-major and transpose in registers instead.
    right = tl.trans(kc * tl.exp2(anchor[None, :] - gc))
    acc_k = tl.dot(left_k, right, input_precision=DP)
    acc_q = tl.dot(left_q, right, input_precision=DP)
    beta = tl.load(Beta + rows * H + head, rows < eos, other=0).to(tl.float32)
    tl.store(A + out, acc_k * beta[:, None], rows[:, None] < eos)
    tl.store(QK + out, acc_q * scale, rows[:, None] < eos)


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
    # GCU: Triton reads int64 buffers with garbage values; normalize to int32
    # (offsets and indices are far below 2**31).
    if cu_seqlens is not None and cu_seqlens.dtype == torch.int64:
        cu_seqlens = cu_seqlens.to(torch.int32)
    if chunk_indices is not None and chunk_indices.dtype == torch.int64:
        chunk_indices = chunk_indices.to(torch.int32)
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
    # GCU matrix arm: fused single-launch subtile kernel (dot for strictly
    # lower 16x16 blocks, direct-delta loop on the diagonal, zero stores for
    # upper blocks). fp16/bf16 q/k dodge the concurrent fp32 wide-tile load
    # corruption; dk <= 128 keeps the fp32 dot single-shot (no K-loop
    # accumulation); b*h <= 255 respects the grid.z limit.
    matrix_gcu = (
        runtime.device.vendor_name == "enflame"
        and t >= 512
        and chunk_size == 64
        and output_dtype == torch.float32
        and gk.dtype == torch.float32
        and beta.dtype == torch.float32
        and q.dtype in (torch.float16, torch.bfloat16)
        and dk <= 128
        and b * h <= 255
        and q.numel() < 2**31
        and (cu_seqlens is None or indexed)
    )
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
        zero_grid = _launch_grid(triton.cdiv(a.numel(), 256))
        _kda_kkt_zero[zero_grid](
            a, aqk, a.numel(), zero_grid[0], zero_grid[0] * zero_grid[1], 256
        )
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
    if matrix_gcu:
        nt = chunk_indices.shape[0] if indexed else triton.cdiv(t, chunk_size)
        _kda_kkt_mat[(nt, (chunk_size // 16) ** 2, b * h)](
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
            nseq,
            ni,
            128,
            "ieee",
        )
        return a, aqk
    width = 16 if matrix else chunk_size
    # GCU: fresh compiles of [PAIRS, BK] tiles with BK < 128 trip the
    # memref_ext.slice_start toolchain bug, and concurrent fp32 q/k tile
    # loads corrupt above 128 active lanes. Clamp every launch to BK in
    # {128, 256}: wide fp32 K runs the blocked path with BK=128.
    bk = 256 if dk > 128 and q.dtype != torch.float32 else 128
    # GCU grid limits (x<=65535, y/z<=255): once rows or groups overflow
    # grid.y/z, pack (group, row tile) into grid.x and column blocks into
    # grid.z; GT and ROWCAP stay independent of the tuned PAIRS value.
    flat = not matrix and (rows > 255 or groups > 255)
    rowcap = min(rows, 255)
    gt = triton.cdiv(rows, rowcap)

    def pairs_grid(config):
        blocks = triton.cdiv(width, config["PAIRS"])
        if flat:
            return (groups * gt, rowcap, blocks)
        return (blocks, rows, groups)

    _kda_kkt_pairs[pairs_grid](
        q,
        k,
        gk,
        beta,
        a,
        aqk,
        cu_seqlens,
        chunk_indices,
        scale,
        gt,
        rowcap,
        t,
        h,
        dk,
        chunk_size,
        bk,
        nseq,
        cu_seqlens is not None,
        indexed,
        matrix,
        FLAT=flat,
    )
    return a, aqk

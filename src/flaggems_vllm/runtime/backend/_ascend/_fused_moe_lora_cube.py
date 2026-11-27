# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Dense sorted-tile Cube projections for Ascend 910B."""

import torch
import triton
import triton.language as tl


@triton.jit
def _metadata(
    Perm,
    Experts,
    Post,
    Ids,
    Enabled,
    Count,
    Codes,
    Pairs,
    P: tl.constexpr,
    RAW: tl.constexpr,
    BLOCKS: tl.constexpr,
    L: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    DEVICE_COUNT: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)
    block = group % tl.cdiv(RAW, BM)
    slot = group // tl.cdiv(RAW, BM)
    adapter = tl.load(Ids + slot).to(tl.int64)
    valid = (adapter >= 0) & (adapter < L)
    if DEVICE_COUNT:
        valid &= slot < tl.load(Count)
    a = tl.where(valid, adapter, 0)
    valid &= tl.load(Enabled + a) != 0
    count = tl.load(Post + a)
    start = block * BM
    expert = tl.load(Experts + a * BLOCKS + block).to(tl.int64)
    valid &= (start < count) & (expert >= 0) & (expert < E)
    # Each independently written scalar owns a 32-byte DMA block.
    tl.store(Codes + group * 4, tl.where(valid, a * E + expert, -1))
    lane = start + tl.arange(0, BM)
    p = tl.load(Perm + a * RAW + lane, lane < RAW, other=-1).to(tl.int64)
    keep = valid & (lane < count) & (p >= 0) & (p < P)
    tl.store(Pairs + group * BM + tl.arange(0, BM), tl.where(keep, p, -1))


@triton.jit
def _pack_rows(
    X, Packed, Pairs, Codes, FACTOR: tl.constexpr, K: tl.constexpr, BM: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    if tl.load(Codes + row // BM * 4) < 0:
        return
    p = tl.load(Pairs + row)
    k = tl.arange(0, K)
    x = tl.load(X + (tl.maximum(p, 0) // FACTOR) * K + k)
    tl.store(Packed + row * K + k, tl.where(p >= 0, x, 0))


@triton.jit
def _cube(
    X,
    W,
    Y,
    Codes,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)
    code = tl.load(Codes + group * 4)
    if code < 0:
        return
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    m = group * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(0, K, BK):
        a = tl.load(X + m[:, None] * K + i + k[None, :])
        b = tl.load(W + code * N * K + n[:, None].to(tl.int64) * K + i + k[None, :])
        acc = tl.dot(a, tl.trans(b), acc=acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc)


@triton.jit
def _scatter(
    Delta,
    Y,
    Pairs,
    Weights,
    N: tl.constexpr,
    YS: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    ADD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    p = tl.load(Pairs + row)
    if p < 0:
        return
    n = tl.arange(0, BN)
    v = tl.load(Delta + row * N + n, n < N, other=0).to(tl.float32)
    if SCALE:
        v *= tl.load(Weights + p).to(tl.float32)
    v = v.to(Y.dtype.element_ty).to(tl.float32)
    y = Y + p * YS + n
    if ADD:
        v += tl.load(y, n < N, other=0).to(tl.float32)
    tl.store(y, v, n < N)


def _dense(tensor):
    if tensor.is_contiguous() and tensor.storage_offset() * tensor.element_size() % 32:
        return tensor.clone(memory_format=torch.contiguous_format)
    return tensor.contiguous()


def selected(output, aa, bb, weights, perm, bm, active):
    pairs = weights.numel()
    l, e, rank, k = aa[0].shape
    if (
        output.dtype not in (torch.float16, torch.bfloat16)
        or weights.dtype != torch.float32
        or not 256 <= pairs <= 4096
        or not 1 <= len(aa) <= 2
        or l <= 0
        or e <= 0
        or (bm, rank, k) != (64, 16, 2048)
        or any(not 128 <= b.shape[2] <= 1408 or b.shape[2] % 128 for b in bb)
    ):
        return False
    # Bound the packed activations to 72 MiB and weighted FP32 deltas to 99 MiB.
    # The largest grids launch 18432 row programs or 288 * 11 Cube tiles.
    rows = active * triton.cdiv(perm.shape[1], bm) * bm
    return 0 < rows <= min(18432, 8 * pairs)


def launch(
    output,
    x,
    aa,
    bb,
    weights,
    perm,
    experts,
    post,
    ids,
    count,
    enabled,
    bm,
    active,
    weighted,
    offset,
    add,
):
    t, topk = weights.shape
    p = t * topk
    l, e, rank, k = aa[0].shape
    groups = active * triton.cdiv(perm.shape[1], bm)
    rows = groups * bm
    dev = output.device
    opts = dict(num_warps=4, num_stages=1)
    codes = torch.empty((groups, 4), dtype=torch.int64, device=dev)
    pairs = torch.empty(rows, dtype=torch.int64, device=dev)
    device_count = count.device.type != "cpu"
    _metadata[(groups,)](
        perm.contiguous(),
        experts.contiguous(),
        post.contiguous(),
        ids.contiguous(),
        enabled.contiguous(),
        count if device_count else None,
        codes,
        pairs,
        p,
        perm.shape[1],
        experts.shape[1],
        l,
        e,
        bm,
        device_count,
        **opts,
    )
    xpack = torch.empty((rows, k), dtype=x.dtype, device=dev)
    _pack_rows[(rows,)](
        _dense(x), xpack, pairs, codes, 1 if weighted else topk, k, bm, **opts
    )
    z = torch.empty((rows, rank), dtype=x.dtype, device=dev)
    weights = weights.contiguous()
    # Weighted expansion rounds after multiplying by the routed weight.
    delta = torch.empty(
        rows * max(b.shape[2] for b in bb),
        dtype=torch.float32 if weighted else x.dtype,
        device=dev,
    )
    for a, b in zip(aa, bb):
        n = b.shape[2]
        _cube[(groups, 1)](xpack, _dense(a), z, codes, k, rank, bm, rank, 128, **opts)
        _cube[(groups, n // 128)](
            z, _dense(b), delta, codes, rank, n, bm, 128, rank, **opts
        )
        y = output[:, :, offset : offset + n]
        direct = (
            y.stride(2) == 1
            and y.stride(0) == topk * y.stride(1)
            and y.stride(1) % 16 == 0
            and y.storage_offset() % 16 == 0
        )
        # Misaligned output rows can share a DMA block with neighboring rows.
        target = (
            y.view(p, n)
            if direct
            else y.clone(memory_format=torch.contiguous_format).view(p, n)
        )
        _scatter[(rows,)](
            delta,
            target,
            pairs,
            weights,
            n,
            target.stride(0),
            triton.next_power_of_2(n),
            weighted,
            add,
            **opts,
        )
        if not direct:
            y.copy_(target.view(t, topk, n))
        offset += n

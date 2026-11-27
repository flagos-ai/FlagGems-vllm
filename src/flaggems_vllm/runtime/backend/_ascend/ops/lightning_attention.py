# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-style Lightning Attention kernels for Ascend."""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

_HEAD_DIM = 128
_HISTORY_BLOCK = 256
_MAX_PREFILL_TOKENS = 16384
_STATE_DTYPES = (torch.float32, torch.bfloat16)


@triton.jit
def _short_summary(
    K,
    V,
    S,
    Initial,
    History,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BE: tl.constexpr,
):
    bh, block, ve = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, D)
    e = ve * BE + tl.arange(0, BE)
    origin = block * BLOCK
    count = tl.minimum(T - origin, BLOCK)
    s = tl.load(S + bh % H)
    acc = tl.zeros((D, BE), tl.float32)
    for start in range(0, tl.cdiv(count, BK)):
        token = start * BK + tl.arange(0, BK)
        valid = token < count
        idx = bh * T + origin + token
        k = tl.load(K + idx[:, None] * D + d[None, :], valid[:, None], 0).to(tl.float32)
        v = tl.load(V + idx[:, None] * E + e[None, :], valid[:, None], 0).to(tl.float32)
        decay = tl.exp(-s * tl.maximum(count - 1 - token, 0))
        wk = tl.where(valid[:, None], k * decay[:, None], 0.0)
        v = tl.where(valid[:, None], v, 0.0)
        acc += tl.dot(tl.trans(wk), v)
    offset = ((bh * (NB + 1) + block) * D + d[:, None]) * E + e[None, :]
    prev = tl.load(Initial + (bh * D + d[:, None]) * E + e[None, :]).to(tl.float32)
    tl.store(History + offset, prev)
    last = prev * tl.exp(-s * count) + acc
    tl.store(History + offset + D * E, last.to(Initial.dtype.element_ty).to(tl.float32))


@triton.jit
def _short_output(
    Q,
    K,
    V,
    S,
    History,
    Out,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    BT: tl.constexpr,
    BE: tl.constexpr,
):
    bh, tile, ve = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    block, sub = tile // (BLOCK // BT), tile % (BLOCK // BT)
    r = tl.arange(0, BT)
    d = tl.arange(0, D)
    e = ve * BE + tl.arange(0, BE)
    q_row = sub * BT + r
    q_pos = block * BLOCK + q_row
    q_base = bh * T + q_pos
    q = tl.load(Q + q_base[:, None] * D + d[None, :], q_pos[:, None] < T, 0).to(
        tl.float32
    )
    q = tl.where(q_pos[:, None] < T, q, 0.0)
    s = tl.load(S + bh % H)
    local = tl.zeros((BT, BE), tl.float32)
    for j in range(sub + 1):
        k_row = j * BT + r
        k_pos = block * BLOCK + k_row
        kv_base = bh * T + k_pos
        k = tl.load(K + kv_base[:, None] * D + d[None, :], k_pos[:, None] < T, 0).to(
            tl.float32
        )
        v = tl.load(V + kv_base[:, None] * E + e[None, :], k_pos[:, None] < T, 0).to(
            tl.float32
        )
        k = tl.where(k_pos[:, None] < T, k, 0.0)
        v = tl.where(k_pos[:, None] < T, v, 0.0)
        distance = q_row[:, None] - k_row[None, :]
        decay = tl.exp(tl.where(distance >= 0, -s * distance, -float("inf")))
        scores = tl.dot(q, tl.trans(k)) * decay
        local += tl.dot(scores, v)
    history = tl.load(History + (bh * D + d[:, None]) * E + e[None, :]).to(tl.float32)
    cross = tl.dot(q, history) * tl.exp(-s * q_row[:, None])
    out = local.to(Out.dtype.element_ty).to(tl.float32) + cross
    tl.store(Out + q_base[:, None] * E + e[None, :], out, q_pos[:, None] < T)


@triton.jit
def _compact_summary(
    K,
    V,
    S,
    Work,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    BE: tl.constexpr,
):
    bh, tile, ve = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = tl.arange(0, BT)
    d = tl.arange(0, D)
    e = ve * BE + tl.arange(0, BE)
    token = tile * BT + r
    base = (bh * T + token)[:, None]
    valid = token[:, None] < T
    k = tl.load(K + base * D + d[None, :], valid, 0).to(tl.float32)
    v = tl.load(V + base * E + e[None, :], valid, 0).to(tl.float32)
    slope = tl.load(S + bh % H)
    count = tl.minimum(T - tile * BT, BT)
    decay = tl.exp(-slope * tl.maximum(count - 1 - r, 0))
    wk = tl.where(valid, k * decay[:, None], 0.0)
    v = tl.where(valid, v, 0.0)
    summary = tl.dot(tl.trans(wk), v)
    dst = ((bh * (NT + 1) + tile) * D + d[:, None]) * E + e[None, :]
    tl.store(Work + dst, summary)


@triton.jit
def _compact_prefix(
    S,
    Initial,
    Work,
    History,
    H: tl.constexpr,
    T: tl.constexpr,
    NT: tl.constexpr,
    NB: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    WIDTH: tl.constexpr,
):
    bh = tl.program_id(0)
    offsets = tl.program_id(1) * WIDTH + tl.arange(0, WIDTH)
    STRIDE: tl.constexpr = D * E
    src = Work + bh * (NT + 1) * STRIDE + offsets
    dst = History + bh * (NB + 1) * STRIDE + offsets
    parent = tl.load(Initial + bh * STRIDE + offsets).to(tl.float32)
    s = tl.load(S + bh % H)
    step = tl.exp(-s * BT)
    corr = tl.exp(-s)
    block_decay = tl.exp(-s * 256)
    tl.store(dst, parent)

    # Hoist the block-boundary branch out of the 64/128-token inner scan.
    for block in range(NB - 1):
        local = tl.full((WIDTH,), 0, tl.float32)
        from_parent = parent
        for j in tl.static_range(256 // BT):
            tile = block * (256 // BT) + j
            ptr = src + tile * STRIDE
            summary = tl.load(ptr)
            if j == 0:
                tl.store(ptr, from_parent)
            else:
                tl.store(ptr, from_parent + local * corr)
            local = local * step + summary
            from_parent = from_parent * step
        parent = parent * block_decay + local
        tl.store(dst + (block + 1) * STRIDE, parent)

    # Tail is specialized by length. Round only the externally visible state.
    local = tl.full((WIDTH,), 0, tl.float32)
    from_parent = parent
    for j in tl.static_range(NT - (NB - 1) * (256 // BT)):
        tile = (NB - 1) * (256 // BT) + j
        ptr = src + tile * STRIDE
        summary = tl.load(ptr)
        if j == 0:
            tl.store(ptr, from_parent)
        else:
            tl.store(ptr, from_parent + local * corr)
        n = tl.minimum(T - tile * BT, BT)
        local = local * tl.exp(-s * n) + summary
        from_parent = from_parent * step
    parent = parent * tl.exp(-s * (T - (NB - 1) * 256)) + local
    tl.store(dst + NB * STRIDE, parent.to(Initial.dtype.element_ty).to(tl.float32))


@triton.jit
def _tile_output(
    Q,
    K,
    V,
    S,
    Work,
    Out,
    H: tl.constexpr,
    T,
    NT,
    D: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BE: tl.constexpr,
):
    bh, tile, ve = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = tl.arange(0, BT)
    d = tl.arange(0, D)
    e = ve * BE + tl.arange(0, BE)
    token = tile * BT + r
    base = (bh * T + token)[:, None]
    valid = token[:, None] < T
    q = tl.load(Q + base * D + d[None, :], valid, 0).to(tl.float32)
    k = tl.load(K + base * D + d[None, :], valid, 0).to(tl.float32)
    v = tl.load(V + base * E + e[None, :], valid, 0).to(tl.float32)
    q = tl.where(valid, q, 0.0)
    k = tl.where(valid, k, 0.0)
    v = tl.where(valid, v, 0.0)
    s = tl.load(S + bh % H)
    delta = r[:, None] - r[None, :]
    weights = tl.exp(tl.where(delta >= 0, -s * delta, -float("inf")))
    local = tl.dot(tl.dot(q, tl.trans(k)) * weights, v)
    state = tl.load(Work + ((bh * (NT + 1) + tile) * D + d[:, None]) * E + e[None, :])
    cross = tl.dot(q, state) * tl.exp(-s * r[:, None])
    tl.store(Out + base * E + e[None, :], local + cross, valid)


@triton.jit
def _split_q_output(
    Q,
    K,
    V,
    S,
    Work,
    Out,
    H: tl.constexpr,
    T,
    NT,
    D: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    QW: tl.constexpr,
    BT: tl.constexpr,
):
    bh, task, ve = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tile, sub = task // (BT // QW), task % (BT // QW)
    r = tl.arange(0, QW)
    d = tl.arange(0, D)
    e = ve * BE + tl.arange(0, BE)
    qrow = sub * QW + r
    qt = tile * BT + qrow
    base = (bh * T + qt)[:, None]
    valid = qt[:, None] < T
    q = tl.load(Q + base * D + d[None, :], valid, 0).to(tl.float32)
    q = tl.where(valid, q, 0.0)
    s = tl.load(S + bh % H)
    local = tl.zeros((QW, BE), tl.float32)
    for j in range(sub + 1):
        krow = j * QW + r
        kt = tile * BT + krow
        kbase = (bh * T + kt)[:, None]
        kv_valid = kt[:, None] < T
        k = tl.load(K + kbase * D + d[None, :], kv_valid, 0).to(tl.float32)
        v = tl.load(V + kbase * E + e[None, :], kv_valid, 0).to(tl.float32)
        k = tl.where(kv_valid, k, 0.0)
        v = tl.where(kv_valid, v, 0.0)
        delta = qrow[:, None] - krow[None, :]
        weights = tl.exp(tl.where(delta >= 0, -s * delta, -float("inf")))
        local += tl.dot(tl.dot(q, tl.trans(k)) * weights, v)
    state = tl.load(Work + ((bh * (NT + 1) + tile) * D + d[:, None]) * E + e[None, :])
    cross = tl.dot(q, state) * tl.exp(-s * qrow[:, None])
    tl.store(Out + base * E + e[None, :], local + cross, valid)


@triton.jit
def _decode_state_columns(
    Q,
    K,
    V,
    Cache,
    Slope,
    Slots,
    Out,
    CACHE_ROWS,
    H: tl.constexpr,
    D: tl.constexpr,
    TILES: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    C_STRIDES: tl.constexpr,
    S_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // (H * TILES)
    head = (pid // TILES) % H
    col = (pid % TILES) * BV + tl.arange(0, BV)
    row = tl.arange(0, BD)
    slot = tl.load(Slots + batch * SLOT_STRIDE).to(tl.int64)
    out = Out + (batch * H + head) * D + col
    if (slot >= 0) & (slot < CACHE_ROWS):
        q = tl.load(
            Q + batch * Q_STRIDES[0] + head * Q_STRIDES[1] + row * Q_STRIDES[3],
            row < D,
            0,
        ).to(tl.float32)
        k = tl.load(
            K + batch * K_STRIDES[0] + head * K_STRIDES[1] + row * K_STRIDES[3],
            row < D,
            0,
        ).to(tl.float32)
        v = tl.load(
            V + batch * V_STRIDES[0] + head * V_STRIDES[1] + col * V_STRIDES[3],
            col < D,
            0,
        ).to(tl.float32)
        decay = tl.exp(-tl.load(Slope + head * S_STRIDE).to(tl.float32))
        ptr = (
            Cache
            + slot * C_STRIDES[0]
            + head * C_STRIDES[1]
            + row[:, None] * C_STRIDES[2]
            + col[None, :] * C_STRIDES[3]
        )
        mask = (row[:, None] < D) & (col[None, :] < D)
        old = tl.load(ptr, mask, 0).to(tl.float32)
        updated = decay * old + k[:, None] * v[None, :]
        output = tl.sum(q[:, None] * updated, axis=0)
        tl.store(ptr, updated, mask)
        tl.store(out, output, col < D)
    else:
        tl.store(out, 0, col < D)


def _device_guard(device):
    # Avoid the device exchange on each dispatch when the caller's NPU is current.
    if torch.npu.current_device() == device.index:
        return nullcontext()
    return torch.npu.device(device.index)


def _check_qkv(q, k, v, slope):
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("expected Q, K and V of the same shape [B,H,T,D]")
    if q.device.type != "npu" or any(x.device != q.device for x in (k, v, slope)):
        raise ValueError("Q, K, V and slope must be on the same NPU")
    _, heads, _, dim = q.shape
    if not heads or slope.numel() != heads or slope.dtype != torch.float32:
        raise ValueError("expected one FP32 slope per head")
    if dim != _HEAD_DIM or any(x.dtype != torch.float32 for x in (q, k, v)):
        raise NotImplementedError(
            "Ascend Lightning Attention requires FP32 Q/K/V and D=128"
        )


def lightning_attention(q, k, v, ed, block_size=256, kv_history=None):
    """Return [B,H,T,D] output and FP32 snapshots at 256-token boundaries.

    The optional BF16 or FP32 incoming history is read-only. The last snapshot
    follows vLLM's rounding to the incoming state dtype.
    """
    _check_qkv(q, k, v, ed)
    b, h, t, d = q.shape
    if block_size != _HISTORY_BLOCK or not 0 < t <= _MAX_PREFILL_TOKENS:
        raise NotImplementedError(
            "Ascend Prefill requires block_size=256 and 1–16384 tokens"
        )
    if kv_history is not None and (
        kv_history.shape != (b, h, d, d)
        or kv_history.dtype not in _STATE_DTYPES
        or kv_history.device != q.device
    ):
        raise ValueError(
            "expected BF16 or FP32 kv_history of shape [B,H,D,D] on the same NPU"
        )
    blocks = triton.cdiv(t, block_size)
    coarse = triton.cdiv(t, 128) >= 64
    tile = 128 if coarse else 64
    tiles = triton.cdiv(t, tile)
    workspace = 0 if blocks == 1 else tiles + 1
    # Kernels use 32-bit element offsets for the contiguous Prefill tensors.
    if b * h * max(t * d, max(blocks + 1, workspace) * d * d) >= 2**31:
        raise NotImplementedError(
            "Ascend Prefill inputs or state exceed 32-bit element indexing"
        )
    history = torch.empty(
        (b, h, blocks + 1, d, d), dtype=torch.float32, device=q.device
    )
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if not b:
        return out, history
    with _device_guard(q.device):
        q, k, v, slope = (x.contiguous() for x in (q, k, v, ed))
        initial = (
            torch.zeros((b, h, d, d), dtype=torch.float32, device=q.device)
            if kv_history is None
            else kv_history.contiguous()
        )
        if blocks == 1:
            _short_summary[(b * h, 1, 2)](
                k,
                v,
                slope,
                initial,
                history,
                h,
                t,
                d,
                d,
                1,
                block_size,
                64,
                64,
            )
            _short_output[(b * h, triton.cdiv(t, 32), 1)](
                q,
                k,
                v,
                slope,
                initial,
                out,
                h,
                t,
                d,
                d,
                1,
                block_size,
                32,
                d,
                multibuffer=True,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                set_workspace_multibuffer=4,
                tile_mix_vector_loop=2,
                tile_mix_cube_loop=2,
            )
        else:
            work = torch.empty(
                (b, h, workspace, d, d), dtype=torch.float32, device=q.device
            )
            grid = (b * h, tiles, 1)
            _compact_summary[grid](k, v, slope, work, h, t, d, d, tiles, tile, d)
            _compact_prefix[(b * h, d * d // 1024)](
                slope,
                initial,
                work,
                history,
                h,
                t,
                tiles,
                blocks,
                d,
                d,
                tile,
                1024,
            )
            if coarse:
                _split_q_output[(b * h, triton.cdiv(t, 64), 1)](
                    q,
                    k,
                    v,
                    slope,
                    work,
                    out,
                    h,
                    t,
                    tiles,
                    d,
                    d,
                    d,
                    64,
                    tile,
                )
            else:
                _tile_output[grid](
                    q, k, v, slope, work, out, h, t, tiles, d, d, tile, d
                )
    return out, history


def lightning_attention_decode(q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE=32):
    """Update distinct active cache slots in place and return [B,H*D].

    Negative slots write zero output without touching the cache. As in vLLM,
    callers must give each live batch row a distinct slot.
    """
    _check_qkv(q, k, v, slope_rate)
    b, h, t, d = q.shape
    if t != 1 or kv_caches.ndim != 4 or kv_caches.shape[1:] != (h, d, d):
        raise ValueError("expected one token and [slots,H,D,D] cache")
    if (
        kv_caches.device != q.device
        or slot_idx.device != q.device
        or kv_caches.dtype not in _STATE_DTYPES
        or slot_idx.shape != (b,)
        or slot_idx.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "expected BF16/FP32 cache and one integer slot per row on the same NPU"
        )
    if BLOCK_SIZE != 32:
        raise NotImplementedError("Ascend Lightning Decode requires BLOCK_SIZE=32")
    out = torch.empty((b, h * d), dtype=q.dtype, device=q.device)
    if b:
        # This is a schedule choice; BLOCK_SIZE is the vLLM API argument.
        width = 32 if b * h <= 256 else 64
        tiles = triton.cdiv(d, width)
        # All supported slope layouts have a single head axis. Pass its stride
        # directly to avoid a Torch view operation on the Decode hot path.
        if slope_rate.shape in ((h,), (h, 1, 1)):
            slope_stride = slope_rate.stride(0)
        elif slope_rate.shape == (1, h, 1, 1):
            slope_stride = slope_rate.stride(1)
        else:
            raise ValueError("expected slope shape [H], [H,1,1], or [1,H,1,1]")
        with _device_guard(q.device):
            _decode_state_columns[(b * h * tiles,)](
                q,
                k,
                v,
                kv_caches,
                slope_rate,
                slot_idx,
                out,
                kv_caches.shape[0],
                h,
                d,
                tiles,
                q.stride(),
                k.stride(),
                v.stride(),
                kv_caches.stride(),
                slope_stride,
                slot_idx.stride(0),
                d,
                width,
            )
    return out

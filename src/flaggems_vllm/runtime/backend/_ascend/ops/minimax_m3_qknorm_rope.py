# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Gemma normalization and partial NeoX rotation for packed MiniMax main heads."""

import triton
import triton.language as tl


@triton.jit
def _rot64_values(
    Packed,
    Weight,
    Phase,
    Positions,
    rows,
    head,
    valid,
    STRIDE: tl.constexpr,
    EPS: tl.constexpr,
):
    d = tl.arange(0, 32)
    addr = rows.to(tl.int64)[:, None] * STRIDE + head * 128 + d[None, :]
    a = tl.load(Packed + addr, valid[:, None], 0).to(tl.float32)
    b = tl.load(Packed + addr + 32, valid[:, None], 0).to(tl.float32)
    c = tl.load(Packed + addr + 64, valid[:, None], 0).to(tl.float32)
    e = tl.load(Packed + addr + 96, valid[:, None], 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(a * a + b * b + c * c + e * e, 1) / 128.0 + EPS)
    a = a * inv[:, None] * (1 + tl.load(Weight + d).to(tl.float32)[None, :])
    b = b * inv[:, None] * (1 + tl.load(Weight + d + 32).to(tl.float32)[None, :])
    c = c * inv[:, None] * (1 + tl.load(Weight + d + 64).to(tl.float32)[None, :])
    e = e * inv[:, None] * (1 + tl.load(Weight + d + 96).to(tl.float32)[None, :])
    pos = tl.load(Positions + rows, valid, 0)
    phase = pos[:, None] * 64 + d[None, :]
    cos = tl.load(Phase + phase).to(tl.float32)
    sin = tl.load(Phase + phase + 32).to(tl.float32)
    return a * cos - b * sin, b * cos + a * sin, c, e


@triton.jit
def _store_rot64(Dest, base, valid, a, b, c, e):
    addr = base[:, None] + tl.arange(0, 32)[None, :]
    tl.store(Dest + addr, a, valid[:, None])
    tl.store(Dest + addr + 32, b, valid[:, None])
    tl.store(Dest + addr + 64, c, valid[:, None])
    tl.store(Dest + addr + 96, e, valid[:, None])


@triton.jit
def _main_qk_kernel(
    Packed,
    QW,
    KW,
    Phase,
    Positions,
    QOut,
    N,
    STRIDE: tl.constexpr,
    NQ: tl.constexpr,
    ROT: tl.constexpr,
    EPS: tl.constexpr,
    GATHER_Q: tl.constexpr,
    ROWS: tl.constexpr,
):
    head = tl.program_id(1)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    if ROT == 64:
        valid = rows < N
        base = rows.to(tl.int64) * STRIDE + head * 128
        if head < NQ:
            a, b, c, e = _rot64_values(
                Packed, QW, Phase, Positions, rows, head, valid, STRIDE, EPS
            )
            if GATHER_Q:
                _store_rot64(
                    QOut, (rows.to(tl.int64) * NQ + head) * 128, valid, a, b, c, e
                )
            else:
                _store_rot64(Packed, base, valid, a, b, c, e)
        else:
            a, b, c, e = _rot64_values(
                Packed, KW, Phase, Positions, rows, head, valid, STRIDE, EPS
            )
            _store_rot64(Packed, base, valid, a, b, c, e)
        return
    lane = tl.arange(0, 64)
    half: tl.constexpr = ROT // 2
    # Each lane owns a pair, so both halves are available before either store.
    d = (lane // half) * ROT + lane % half
    valid = rows < N
    addr = rows.to(tl.int64)[:, None] * STRIDE + head * 128 + d[None, :]
    a = tl.load(Packed + addr, valid[:, None], 0).to(tl.float32)
    b = tl.load(Packed + addr + half, valid[:, None], 0).to(tl.float32)
    if head < NQ:
        wa = tl.load(QW + d).to(tl.float32)
        wb = tl.load(QW + d + half).to(tl.float32)
    else:
        wa = tl.load(KW + d).to(tl.float32)
        wb = tl.load(KW + d + half).to(tl.float32)
    inv = tl.rsqrt(tl.sum(a * a + b * b, 1) / 128.0 + EPS)
    a = a * inv[:, None] * (1.0 + wa[None, :])
    b = b * inv[:, None] * (1.0 + wb[None, :])
    pos = tl.load(Positions + rows, valid, 0)
    phase = pos[:, None] * ROT + lane[None, :]
    rotated = lane[None, :] < half
    c = tl.load(Phase + phase, rotated, 0).to(tl.float32)
    s = tl.load(Phase + phase + half, rotated, 0).to(tl.float32)
    first = tl.where(rotated, a * c - b * s, a)
    second = tl.where(rotated, b * c + a * s, b)
    if GATHER_Q and head < NQ:
        dest = (rows.to(tl.int64)[:, None] * NQ + head) * 128 + d[None, :]
        tl.store(QOut + dest, first, valid[:, None])
        tl.store(QOut + dest + half, second, valid[:, None])
    else:
        tl.store(Packed + addr, first, valid[:, None])
        tl.store(Packed + addr + half, second, valid[:, None])


def main_qk_norm_rope(qkv, qw, kw, phase, positions, nq, nkv, rot, eps, q_out):
    rows = min(16, triton.next_power_of_2(qkv.shape[0]))
    _main_qk_kernel[(triton.cdiv(qkv.shape[0], rows), nq + nkv)](
        qkv,
        qw,
        kw,
        phase,
        positions,
        q_out,
        qkv.shape[0],
        qkv.stride(0),
        nq,
        rot,
        eps,
        q_out is not None,
        rows,
    )

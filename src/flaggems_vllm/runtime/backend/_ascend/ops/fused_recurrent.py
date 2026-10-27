# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Retains the recurrent formulation from flash-linear-attention (MIT):
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""Ascend recurrent GDN: unique logical tiles and strided state gathers."""
from functools import lru_cache

import triton
import triton.language as tl
from triton.backends.ascend.driver import NPUUtils


@lru_cache(maxsize=None)
def _aivector_core_count(device_index):
    # Cache hardware metadata per device, never tensor data or state indices.
    # NPUUtils.__init__ reloads an extension even on cache-warm calls.
    return NPUUtils().get_aivector_core_num()


@triton.jit
def _recurrent_gdn_kernel(
    Q,
    Kp,
    Vp,
    G,
    Beta,
    State,
    Out,
    Final,
    Cu,
    Indices,
    Accepted,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    POOL: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    GS: tl.constexpr,
    BS: tl.constexpr,
    SS: tl.constexpr,
    IX: tl.constexpr,
    WIDTH: tl.constexpr,
    VARLEN: tl.constexpr,
    INDEXED: tl.constexpr,
    TABLE: tl.constexpr,
    SPEC: tl.constexpr,
    INITIAL: tl.constexpr,
    INPLACE: tl.constexpr,
    VECTOR_BETA: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_G: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TOTAL: tl.constexpr,
):
    # Bound the physical grid in the host; oversized auto-parallel launches
    # can replay task zero and update the same state tile more than once.
    for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        nv = tl.cdiv(V, BV)
        iv = pid % nv
        nh = pid // nv
        n, hv = nh // HV, nh % HV
        h = hv // (HV // H)
        if VARLEN:
            start = tl.load(Cu + n).to(tl.int64)
            end = tl.load(Cu + n + 1).to(tl.int64)
            batch = 0
        else:
            start, end = 0, T
            batch = n
        rk = tl.arange(0, BK)
        rv = iv * BV + tl.arange(0, BV)
        mask = (rv[:, None] < V) & (rk[None, :] < K)
        slot = n.to(tl.int64)
        valid = True
        if INDEXED:
            col = 0
            if SPEC:
                col = tl.load(Accepted + n).to(tl.int64) - 1
            valid = (col >= 0) & (col < WIDTH)
            if valid:
                slot = tl.load(Indices + n * IX[0] + col * IX[1]).to(tl.int64)
            else:
                slot = tl.full((), 0, tl.int64)
            valid = valid & (slot > 0) & (slot < POOL)
        if SS[3] == 1 and SS[2] == K:
            # Dense rows retain vectorized 2D transfers.
            state_offsets = hv * SS[1] + rv[:, None] * SS[2] + rk[None, :]
            state_mask = mask
        else:
            # The tested Ascend compiler miscopies 2D strided state tiles.
            # Gather/scatter flat addresses and reshape only in registers.
            rr = tl.arange(0, BV * BK)
            flat_v = iv * BV + rr // BK
            flat_k = rr % BK
            state_offsets = hv * SS[1] + flat_v * SS[2] + flat_k * SS[3]
            state_mask = (flat_v < V) & (flat_k < K)
        acc = tl.full((BV, BK), 0, tl.float32)
        if INITIAL:
            loaded = tl.load(
                State + slot * SS[0] + state_offsets, mask=state_mask & valid, other=0
            ).to(tl.float32)
            if SS[3] == 1 and SS[2] == K:
                acc = loaded
            else:
                acc = loaded.reshape((BV, BK))
        for t in range(start, end):
            q = tl.load(
                Q + batch * QS[0] + t * QS[1] + h * QS[2] + rk * QS[3], rk < K, 0
            ).to(tl.float32)
            k = tl.load(
                Kp + batch * KS[0] + t * KS[1] + h * KS[2] + rk * KS[3], rk < K, 0
            ).to(tl.float32)
            v = tl.load(
                Vp + batch * VS[0] + t * VS[1] + hv * VS[2] + rv * VS[3], rv < V, 0
            ).to(tl.float32)
            if NORMALIZE:
                q = q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)
                k = k / tl.sqrt(tl.sum(k * k, 0) + 1.0e-6)
            q *= scale
            if HAS_G:
                g = tl.load(G + batch * GS[0] + t * GS[1] + hv * GS[2]).to(tl.float32)
                acc *= tl.exp(g)
            delta = v - tl.sum(acc * k[None, :], 1)
            if HAS_BETA:
                bp = Beta + batch * BS[0] + t * BS[1] + hv * BS[2]
                if VECTOR_BETA:
                    beta = tl.load(bp + rv * BS[3], rv < V, 0).to(tl.float32)
                else:
                    beta = tl.load(bp).to(tl.float32)
                delta *= beta
            acc += delta[:, None] * k[None, :]
            out = tl.sum(acc * q[None, :], 1)
            token = batch * T + t
            tl.store(Out + (token * HV + hv) * V + rv, tl.where(valid, out, 0), rv < V)
            if INPLACE:
                dest = slot
                store = valid
                if TABLE:
                    final_col = t - start
                    if final_col < WIDTH:
                        dest = tl.load(Indices + n * IX[0] + final_col * IX[1]).to(
                            tl.int64
                        )
                    else:
                        dest = tl.full((), 0, tl.int64)
                    store = store & (dest > 0) & (dest < POOL) & (final_col < WIDTH)
                else:
                    store = store & (t == end - 1)
                if SS[3] == 1 and SS[2] == K:
                    tl.store(
                        Final + dest * SS[0] + state_offsets, acc, state_mask & store
                    )
                else:
                    tl.store(
                        Final + dest * SS[0] + state_offsets,
                        acc.reshape((BV * BK,)),
                        state_mask & store,
                    )
            else:
                offsets = ((token * HV + hv) * V + rv[:, None]) * K + rk[None, :]
                tl.store(Final + offsets, tl.where(valid, acc, 0), mask)

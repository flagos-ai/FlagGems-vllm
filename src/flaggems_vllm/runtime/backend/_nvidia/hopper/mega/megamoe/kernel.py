# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM90 FP8 expert-parallel MegaMoE kernel.

Each rank owns a contiguous expert shard. A persistent TLE kernel combines
routing, NVLink dispatch and pull, FP8 L1/SwiGLU/L2 compute, remote scatter, and
final reduction in one launch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import triton
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
from triton.experimental.tle.raw import dialect
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor

if TYPE_CHECKING:
    from .api import MegaMoEContext

RAW_DIR = Path(__file__).resolve().parent

# Loader role width used by the current kernel configuration.
LOADER_WARPS = 4


@dialect(
    name="cuda",
    file=RAW_DIR / "raw_l2_wide_scatter.cu",
    extern_func_name="TleL2WideScatter",
)
def l2_wide_scatter_edsl(*args, **kwargs): ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=RAW_DIR / "raw_d8_unified_dispatch.cu",
    extern=RAW_DIR / "raw_d8_unified_dispatch_extern.py",
    extern_func_name="TleD8UnifiedDispatchPull",
)
def d8_unified_dispatch_edsl(*args, **kwargs): ...


# ---------------------------------------------------------------------------
# peer-pointer primitives  ( == sym_buffer.map )
# ---------------------------------------------------------------------------
@triton.jit
def peer_i8(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.int8))


@triton.jit
def peer_i32(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.int32))


@triton.jit
def peer_i64(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.int64))


@triton.jit
def peer_bf16(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.bfloat16))


@triton.jit
def fence_sys():
    _ = tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0, 0;",
        "=r",
        [],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def smem_generic_addr(ptr):
    """Convert TLE's shared offset to the generic address expected by raw CUDA."""
    offset = tl.cast(ptr, tl.uint64)
    return tl.inline_asm_elementwise(
        "cvta.shared.u64 $0, $1;",
        "=l,l",
        [offset],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rank_barrier(counter, NUM_SMS: tl.constexpr):
    """comm::grid_sync across this rank's own CTAs. Partition-safe."""
    tl.debug_barrier()
    tl.atomic_add(counter, 1, sem="release")
    v = tl.atomic_add(counter, 0, sem="acquire")
    while v < NUM_SMS:
        v = tl.atomic_add(counter, 0, sem="acquire")
    tl.debug_barrier()


@triton.jit
def dispatch_sync(USE_D8_TMA1D: tl.constexpr):
    """Dispatch-only wrapper keeps the role's barrier call graph unambiguous."""
    tl.debug_barrier()


@triton.jit
def dispatch_rank_barrier(counter, NUM_SMS: tl.constexpr, USE_D8_TMA1D: tl.constexpr):
    dispatch_sync(USE_D8_TMA1D)
    tl.atomic_add(counter, 1, sem="release")
    v = tl.atomic_add(counter, 0, sem="acquire")
    while v < NUM_SMS:
        v = tl.atomic_add(counter, 0, sem="acquire")
    dispatch_sync(USE_D8_TMA1D)


@triton.jit
def dispatch_nvlink_barrier(
    sig_tab,
    slot,
    ctr,
    sm_idx,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
):
    # D3 already performed a rank-wide grid sync, so the first NPES threads can
    # signal peers concurrently without another prologue barrier.
    dispatch_sync(USE_D8_TMA1D)
    fence_sys()
    if sm_idx == 0:
        pe = tl.arange(0, 8)
        tl.atomic_add(
            peer_i32(sig_tab, pe) + slot,
            1,
            mask=pe < NPES,
            sem="release",
            scope="sys",
        )
        sl = peer_i32(sig_tab, MY_PE)
        v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
        while v < NPES:
            v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
    dispatch_rank_barrier(ctr + 1, NUM_SMS, USE_D8_TMA1D)


@triton.jit
def nvlink_barrier(
    sig_tab,
    slot,
    ctr,
    sm_idx,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """comm::nvlink_barrier, barrier.cuh:38-80."""
    fence_sys()
    rank_barrier(ctr, NUM_SMS)
    if sm_idx == 0:
        # Assign one signaling thread to each peer.
        pe = tl.arange(0, 8)
        tl.atomic_add(
            peer_i32(sig_tab, pe) + slot, 1, mask=pe < NPES, sem="release", scope="sys"
        )
        sl = peer_i32(sig_tab, MY_PE)
        v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
        while v < NPES:
            v = tl.atomic_add(sl + slot, 0, sem="acquire", scope="sys")
    rank_barrier(ctr + 1, NUM_SMS)


@triton.jit
def wait_flag(ptr):
    v = tl.atomic_add(ptr, 0, sem="acquire")
    while v < 1:
        v = tl.atomic_add(ptr, 0, sem="acquire")


@triton.jit
def pool_layout(rsum, EPR: tl.constexpr, EPR_POW2: tl.constexpr, BLOCK_M: tl.constexpr):
    offs = tl.arange(0, EPR_POW2)
    ok = offs < EPR
    n_vec = (tl.load(rsum + offs, mask=ok, other=0) & 0xFFFFFFFF).to(tl.int32)
    blk_vec = tl.where(ok, (n_vec + BLOCK_M - 1) // BLOCK_M, 0)
    off_vec = tl.cumsum(blk_vec, axis=0) - blk_vec
    return n_vec, blk_vec, off_vec


@triton.jit
def pick(vec, idx):
    """Extract one dynamic scalar without one-hot reduction lowering."""
    index = tl.full((1,), idx, tl.int32)
    return tl.reshape(tl.gather(vec, index, axis=0), ())


# ---------------------------------------------------------------------------
# symmetric-heap initialization helpers
# ---------------------------------------------------------------------------
@triton.jit
def sym_zero(tab, PE, NB, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(peer_i8(tab, PE) + offs, tl.zeros((BLOCK,), dtype=tl.int8), mask=offs < NB)


@triton.jit
def sym_h2d(src, tab, PE, NB, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < NB
    src_bytes = src.to(tl.pointer_type(tl.int8))
    tl.store(peer_i8(tab, PE) + offs, tl.load(src_bytes + offs, mask=m), mask=m)


@triton.jit
def fill_bytes(dst, NB, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NB
    dst_bytes = dst.to(tl.pointer_type(tl.int8))
    tl.store(dst_bytes + offs, tl.full((BLOCK,), VALUE, tl.int8), mask=mask)


# ---------------------------------------------------------------------------
# ROLE: independent A+SFA / B producers
# ---------------------------------------------------------------------------
@triton.jit
def loader_role(
    writer,
    rsum_tab,
    l1_arrival_count,
    l2_arrival_mask,
    a1_desc,
    b1_desc,
    a2_desc,
    b2_desc,
    sfa1_desc,
    sfa2_desc,
    LOAD_A: tl.constexpr,
    MY_PE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPR: tl.constexpr,
    EPW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    L1_N: tl.constexpr,
    L2_N: tl.constexpr,
    NK1: tl.constexpr,
    NK2: tl.constexpr,
    NL1N: tl.constexpr,
    NL2N: tl.constexpr,
    RSUM_NPES: tl.constexpr,
):
    sm_idx = tl.program_id(0)
    rsum_l = peer_i64(rsum_tab, MY_PE)

    # D4's high word is the scheduler-readiness contract.  B can schedule and
    # issue before dispatch has finished the later pre-pull barrier.
    expected = NUM_SMS * RSUM_NPES
    ready_e = 0
    while ready_e < EPR:
        status = tl.atomic_add(rsum_l + ready_e, 0, sem="acquire", scope="sys")
        while (status >> 32) != expected:
            status = tl.atomic_add(rsum_l + ready_e, 0, sem="acquire", scope="sys")
        ready_e += 1

    full_mask = ((1 << NL1N) - 1) | (((1 << NL1N) - 1) << 32)

    block_idx = sm_idx
    cur_e = 0
    pool_prefix = 0
    wave_pool_prefix = 0
    phase = 1
    ck = 0
    stop = 0
    while stop == 0:
        got = 0
        while (got == 0) & (stop == 0):
            if cur_e >= EPR:
                stop = 1
            else:
                wave_end = ((cur_e // EPW) + 1) * EPW
                if phase == 1:
                    found = 0
                    while (found == 0) & (cur_e < wave_end):
                        expert_status = tl.atomic_add(
                            rsum_l + cur_e, 0, sem="acquire", scope="sys"
                        )
                        expert_tokens = (expert_status & 0xFFFFFFFF).to(tl.int32)
                        num_m = (expert_tokens + BLOCK_M - 1) // BLOCK_M
                        if (block_idx // NL1N) < num_m:
                            found = 1
                        else:
                            block_idx -= num_m * NL1N
                            pool_prefix += num_m
                            cur_e += 1
                    if found == 1:
                        got = 1
                    else:
                        phase = 2
                        cur_e = ((cur_e - 1) // EPW) * EPW
                        pool_prefix = wave_pool_prefix
                else:
                    found = 0
                    while (found == 0) & (cur_e < wave_end):
                        expert_status = tl.atomic_add(
                            rsum_l + cur_e, 0, sem="acquire", scope="sys"
                        )
                        expert_tokens = (expert_status & 0xFFFFFFFF).to(tl.int32)
                        num_m = (expert_tokens + BLOCK_M - 1) // BLOCK_M
                        if block_idx < num_m * NL2N:
                            found = 1
                        else:
                            block_idx -= num_m * NL2N
                            pool_prefix += num_m
                            cur_e += 1
                    if found == 1:
                        got = 1
                    else:
                        phase = 1
                        wave_pool_prefix = pool_prefix

        if stop == 0:
            nbn = NL1N if phase == 1 else NL2N
            m_block = block_idx // nbn
            n_block = block_idx - m_block * nbn
            block_idx += NUM_SMS
            expert_status = tl.atomic_add(rsum_l + cur_e, 0, sem="acquire", scope="sys")
            n_tok = (expert_status & 0xFFFFFFFF).to(tl.int32)
            pool_block = pool_prefix + m_block
            valid_m = tl.minimum(n_tok - m_block * BLOCK_M, BLOCK_M)

            if phase == 1:
                if LOAD_A:
                    l1_arrival = tl.atomic_add(
                        l1_arrival_count + pool_block, 0, sem="acquire"
                    )
                    while l1_arrival != valid_m:
                        l1_arrival = tl.atomic_add(
                            l1_arrival_count + pool_block, 0, sem="acquire"
                        )
                for kb in tl.range(0, NK1):
                    slot = writer.acquire(ck)
                    if LOAD_A:
                        tle.gpu.copy(
                            a1_desc,
                            slot.a,
                            [BLOCK_M, BLOCK_K],
                            [pool_block * BLOCK_M, kb * BLOCK_K],
                        )
                        # L1 consumes one activation-scale row per K=128 tile.
                        tle.gpu.copy(
                            sfa1_desc,
                            slot.sfa.subslice(0, 1, 0),
                            [1, BLOCK_M],
                            [kb, pool_block * BLOCK_M],
                        )
                    else:
                        tle.gpu.copy(
                            b1_desc,
                            slot.b,
                            [BLOCK_N, BLOCK_K],
                            [cur_e * L1_N + n_block * BLOCK_N, kb * BLOCK_K],
                        )
                    writer.commit(ck)
                    ck += 1
            else:
                if LOAD_A:
                    l2_arrival = tl.atomic_add(
                        l2_arrival_mask + pool_block, 0, sem="acquire"
                    )
                    while l2_arrival != full_mask:
                        l2_arrival = tl.atomic_add(
                            l2_arrival_mask + pool_block, 0, sem="acquire"
                        )
                for kb in tl.range(0, NK2):
                    slot = writer.acquire(ck)
                    if LOAD_A:
                        tle.gpu.copy(
                            a2_desc,
                            slot.a,
                            [BLOCK_M, BLOCK_K],
                            [pool_block * BLOCK_M, kb * BLOCK_K],
                        )
                        tle.gpu.copy(
                            sfa2_desc,
                            slot.sfa,
                            [2, BLOCK_M],
                            [2 * kb, pool_block * BLOCK_M],
                        )
                    else:
                        tle.gpu.copy(
                            b2_desc,
                            slot.b,
                            [BLOCK_N, BLOCK_K],
                            [cur_e * L2_N + n_block * BLOCK_N, kb * BLOCK_K],
                        )
                    writer.commit(ck)
                    ck += 1


# ---------------------------------------------------------------------------
# ROLE: math (worker partition)
# ---------------------------------------------------------------------------
@triton.jit
def math_role(
    reader,
    cd_s,
    dst_rows_s,
    meta_ready,
    rsum_tab,
    cb_tab,
    sig_tab,
    meta_tab,
    cb_local,
    l1_w,
    l2_acts,
    l2_sf,
    l2_arrival_mask,
    topk_local,
    final_y,
    gctr,
    w1_sf,
    w2_sf,
    WG: tl.constexpr,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPR: tl.constexpr,
    EPR_POW2: tl.constexpr,
    EPW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INTER: tl.constexpr,
    NK1: tl.constexpr,
    NK2: tl.constexpr,
    NL1N: tl.constexpr,
    NL2N: tl.constexpr,
    L1_OUT_N: tl.constexpr,
    NPAIR: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
):
    sm_idx = tl.program_id(0)
    rsum_l = peer_i64(rsum_tab, MY_PE)
    wait_flag(meta_ready + sm_idx)
    n_vec, blk_vec, off_vec = pool_layout(rsum_l, EPR, EPR_POW2, BLOCK_M)

    WG_M: tl.constexpr = BLOCK_M // 2
    row_base: tl.constexpr = WG * WG_M
    offs_m = tl.arange(0, WG_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Gate and up projections are interleaved every eight columns. The column
    # group parity selects the corresponding block-granular weight scale.
    _is_up = ((offs_n // 8) % 2) == 1
    meta_l = peer_i32(meta_tab, MY_PE)

    block_idx = sm_idx
    cur_e = 0
    phase = 1
    ck = 0
    stop = 0
    while stop == 0:
        got = 0
        while (got == 0) & (stop == 0):
            if cur_e >= EPR:
                stop = 1
            else:
                wave_end = ((cur_e // EPW) + 1) * EPW
                if phase == 1:
                    f = 0
                    while (f == 0) & (cur_e < wave_end):
                        num_m = pick(blk_vec, cur_e)
                        if (block_idx // NL1N) < num_m:
                            f = 1
                        else:
                            block_idx -= num_m * NL1N
                            cur_e += 1
                    if f == 1:
                        got = 1
                    else:
                        phase = 2
                        cur_e = ((cur_e - 1) // EPW) * EPW
                else:
                    f = 0
                    while (f == 0) & (cur_e < wave_end):
                        num_m = pick(blk_vec, cur_e)
                        if block_idx < num_m * NL2N:
                            f = 1
                        else:
                            block_idx -= num_m * NL2N
                            cur_e += 1
                    if f == 1:
                        got = 1
                    else:
                        phase = 1

        if stop == 0:
            nbn = NL1N if phase == 1 else NL2N
            m_block = block_idx // nbn
            n_block = block_idx - m_block * nbn
            block_idx += NUM_SMS
            n_tok = pick(n_vec, cur_e)
            pool_block = pick(off_vec, cur_e) + m_block
            valid_m = tl.minimum(n_tok - m_block * BLOCK_M, BLOCK_M)
            m_ok = (row_base + offs_m) < valid_m
            nk = NK1 if phase == 1 else NK2
            rows = pool_block * BLOCK_M + row_base + offs_m

            acc = tl.zeros((WG_M, BLOCK_N), dtype=tl.float32)
            for kb in tl.range(0, nk):
                sl = reader.wait(ck).slot
                # Weight SF stays in global memory and is loaded once per
                # (BLOCK_N, BLOCK_K) tile. L1 needs separate gate and up scales;
                # L2 uses a single scale.
                if phase == 1:
                    _gn = n_block // 2
                    sfb_g = tl.load(w1_sf + (cur_e * NL1N + _gn) * NK1 + kb)
                    sfb_u = tl.load(w1_sf + (cur_e * NL1N + NL1N // 2 + _gn) * NK1 + kb)
                else:
                    sfb_g = tl.load(w2_sf + (cur_e * NL2N + n_block) * NK2 + kb)
                    sfb_u = sfb_g

                # FP8 WGMMA accumulates into FP32 before applying block scales.
                # L1 has one activation scale per K=128 tile, while L2 changes
                # scale at K=64 and therefore uses two K=64 WGMMA operations.
                if phase == 1:
                    # The two math warp groups split the rows and share B.
                    acc_lo = tle.gpu.wgmma(
                        sl.a.subslice(row_base, WG_M, 0),
                        sl.b,
                        out_dtype=tl.float32,
                        trans_b=True,
                    )
                    # The activation scale can load while WGMMA is in flight.
                    sf_lo = tl.reshape(
                        tl.load(
                            tle.gpu.local_ptr(
                                sl.sfa.slot(0).subslice(row_base, WG_M, 0)
                            )
                        ),
                        (WG_M,),
                    )
                    sf_hi = sf_lo
                    acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
                    acc_hi = acc_lo
                else:
                    # Consume the low K=64 fragment before issuing the high half.
                    HK: tl.constexpr = BLOCK_K // 2
                    a_my = sl.a.subslice(row_base, WG_M, 0)
                    acc_lo = tle.gpu.wgmma(
                        a_my.subslice(0, HK, -1),
                        sl.b.subslice(0, HK, -1),
                        out_dtype=tl.float32,
                        trans_b=True,
                    )
                    sf_lo = tl.reshape(
                        tl.load(
                            tle.gpu.local_ptr(
                                sl.sfa.slot(0).subslice(row_base, WG_M, 0)
                            )
                        ),
                        (WG_M,),
                    )
                    acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
                    w_lo_serial = tl.where(
                        _is_up[None, :],
                        (sf_lo * sfb_u)[:, None],
                        (sf_lo * sfb_g)[:, None],
                    )
                    acc += acc_lo * w_lo_serial

                    acc_hi = tle.gpu.wgmma(
                        a_my.subslice(HK, HK, -1),
                        sl.b.subslice(HK, HK, -1),
                        out_dtype=tl.float32,
                        trans_b=True,
                    )
                    sf_hi = tl.reshape(
                        tl.load(
                            tle.gpu.local_ptr(
                                sl.sfa.slot(1).subslice(row_base, WG_M, 0)
                            )
                        ),
                        (WG_M,),
                    )
                    acc_hi = tle.gpu.wgmma_wait(0, acc_hi)
                # Fold activation and weight scales before multiplying the
                # accumulator so the compiler can emit fused accumulation.
                if phase == 1:
                    w_lo = tl.where(
                        _is_up[None, :],
                        (sf_lo * sfb_u)[:, None],
                        (sf_lo * sfb_g)[:, None],
                    )
                    acc += acc_lo * w_lo  # single scale for all 128 K
                else:
                    # The low fragment was already consumed before high issue.
                    w_hi = tl.where(
                        _is_up[None, :],
                        (sf_hi * sfb_u)[:, None],
                        (sf_hi * sfb_g)[:, None],
                    )
                    acc += acc_hi * w_hi
                reader.release(ck)
                ck += 1

            # WG1 still consumes every GEMM stage and publishes L1 readiness,
            # but skips the epilogue when it owns no valid rows.
            do_epilogue = True
            if WG == 1:
                do_epilogue = valid_m > row_base

            if phase == 1:
                if do_epilogue:
                    # L1 EPILOGUE: SwiGLU on the granularity-8 interleave + UE8M0 + FP8
                    t4 = tl.reshape(acc, (WG_M, NPAIR, 2, 8))
                    t4 = tl.permute(t4, (0, 1, 3, 2))
                    gate, up = tl.split(t4)
                    gate = tl.reshape(gate, (WG_M, L1_OUT_N))
                    up = tl.reshape(up, (WG_M, L1_OUT_N))
                    # Match UserHopper's approximate FTZ reciprocal path.
                    sw = libdevice.fast_dividef(gate, 1.0 + tl.exp(-gate)) * up
                    weight = tl.load(l1_w + rows, mask=m_ok, other=0.0)
                    # Fold the top-k weight into the row scale.
                    amax = tl.max(tl.abs(sw), axis=1) * tl.abs(weight)
                    scaled = amax * (1.0 / 448.0)
                    pos = scaled > 0.0
                    e = tl.ceil(tl.log2(tl.where(pos, scaled, 1.0)))
                    sf = tl.where(pos, tl.exp2(e), 1.0)
                    sf_inv = tl.where(pos, tl.exp2(-e), 1.0)
                    row_scale = weight * sf_inv
                    q = (sw * row_scale[:, None]).to(tl.float8e4nv)
                    cols_o = n_block * L1_OUT_N + tl.arange(0, L1_OUT_N)
                    tl.store(
                        l2_acts + rows[:, None] * INTER + cols_o[None, :],
                        q,
                        mask=m_ok[:, None],
                    )
                    tl.store(l2_sf + n_block * POOL_TOKENS + rows, sf, mask=m_ok)
                # Both WGs publish readiness even when WG1 skipped the heavy
                # epilogue; the L2 producer waits for the complete bit mask.
                tl.debug_barrier()
                tl.atomic_or(
                    l2_arrival_mask + pool_block,
                    (tl.full((), 1, tl.int64) << (n_block + WG * 32)),
                    sem="release",
                )
            else:
                if do_epilogue:
                    # L2 EPILOGUE: BF16 cast + NVLink scatter to the SOURCE rank
                    md = meta_l + rows * 3
                    dst_rank = tl.load(md + 0, mask=m_ok, other=0)
                    dst_tok = tl.load(md + 1, mask=m_ok, other=0)
                    dst_topk = tl.load(md + 2, mask=m_ok, other=0)
                    cbb = peer_bf16(cb_tab, dst_rank)  # VECTOR of peer bases
                    base = (dst_topk * NTOK + dst_tok) * K
                    # Layout hints cannot change physical WGMMA fragment ownership.
                    # Materialize a linear BF16 tile plus one fully resolved remote
                    # destination per row, then let the 4 physical math warps remap it
                    # as 16 lanes/row x 8 BF16/lane.
                    dst_row = tl.cast(cbb + base + n_block * BLOCK_N, tl.uint64)
                    tl.store(tle.gpu.local_ptr(dst_rows_s.slot(WG)), dst_row, mask=m_ok)
                    # Scatter consumes only valid rows, so padding entries can be
                    # materialized without a masked SMEM read/modify/write.
                    tl.store(tle.gpu.local_ptr(cd_s.slot(WG)), acc.to(tl.bfloat16))
                    cd_addr = smem_generic_addr(
                        tle.gpu.local_ptr(cd_s.slot(WG), (0, 0))
                    )
                    dst_addr = smem_generic_addr(
                        tle.gpu.local_ptr(dst_rows_s.slot(WG), (0,))
                    )
                    valid_m_wg = tl.maximum(tl.minimum(valid_m - row_base, WG_M), 0)
                    bar_wg = tl.full(
                        (), 9 + WG, tl.int32
                    )  # per-WG named barrier (9 / 10)
                    scatter_cols = tl.full((), BLOCK_N, tl.int32)
                    tle_raw.call(
                        l2_wide_scatter_edsl,
                        [cd_addr, dst_addr, valid_m_wg, scatter_cols, bar_wg],
                    )

    # Only WG0 runs the cross-rank barrier and the per-token combine step.
    if WG == 0:
        nvlink_barrier(sig_tab, 1, gctr + 3, sm_idx, MY_PE, NPES, NUM_SMS)
        cb_l = tl.multiple_of(cb_local.to(tl.pointer_type(tl.bfloat16)), 16)
        offs_hid = tl.arange(0, K_POW2)
        hid_ok = offs_hid < K
        t = sm_idx
        while t < NTOK:
            acc_c = tl.zeros((K_POW2,), dtype=tl.float32)
            for k in tl.static_range(0, TOPK):
                e = tl.load(topk_local + t * TOPK + k).to(tl.int32)
                v = tl.load(
                    cb_l + ((k * NTOK + t) * K) + offs_hid,
                    mask=(e >= 0) & hid_ok,
                    other=0.0,
                )
                acc_c += v.to(tl.float32)
            tl.store(final_y + t * K + offs_hid, acc_c.to(tl.bfloat16), mask=hid_ok)
            t += NUM_SMS


# ---------------------------------------------------------------------------
@triton.jit
def dispatch_frontend_dual_pull(
    topk_local,
    q_tab,
    recv_tab,
    rsum_tab,
    sig_tab,
    expert_send_count,
    gctr,
    meta_ready,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    NEXP: tl.constexpr,
    EPR: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ITERS: tl.constexpr,
    NEXP_POW2: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
):
    """Two-warp D1-D5 frontend; publish finalized receive metadata."""
    sm_idx = tl.program_id(0)
    routes = NTOK * TOPK
    offs_e = tl.arange(0, NEXP_POW2)
    e_ok = offs_e < NEXP
    ec_s = tle.gpu.alloc(
        [NEXP_POW2],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    ec_vec = tle.gpu.local_ptr(ec_s, (offs_e,))
    tl.store(ec_vec, tl.zeros((NEXP_POW2,), dtype=tl.int32))
    dispatch_sync(USE_D8_TMA1D)

    for it in tl.static_range(0, ITERS):
        r = (it * NUM_SMS + sm_idx) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = r < routes
        e = tl.load(topk_local + r, mask=m, other=-1).to(tl.int32)
        valid = m & (e >= 0)
        tl.atomic_add(
            tle.gpu.local_ptr(ec_s, (tl.where(valid, e, 0),)),
            1,
            mask=valid,
            sem="relaxed",
            scope="cta",
        )
    dispatch_sync(USE_D8_TMA1D)

    cnt = tl.load(ec_vec)
    # Pack one completed CTA in the high word and its route count in the low word.
    send_value = (tl.full((NEXP_POW2,), 1, tl.int64) << 32) | cnt.to(tl.int64)
    # The previous count reserves this CTA's range in each expert queue.
    old = tl.atomic_add(
        expert_send_count + offs_e, send_value, mask=e_ok, sem="relaxed"
    )
    tl.store(ec_vec, (old & 0xFFFFFFFF).to(tl.int32))
    dispatch_sync(USE_D8_TMA1D)

    for it in tl.static_range(0, ITERS):
        r = (it * NUM_SMS + sm_idx) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = r < routes
        e = tl.load(topk_local + r, mask=m, other=-1).to(tl.int32)
        valid = m & (e >= 0)
        owner = tl.where(valid, e // EPR, 0)
        local_e = tl.where(valid, e % EPR, 0)
        # Allocate the route's slot within the reserved expert range.
        d3_slot = tl.atomic_add(
            tle.gpu.local_ptr(ec_s, (tl.where(valid, e, 0),)),
            1,
            mask=valid,
            sem="relaxed",
            scope="cta",
        )
        qb = peer_i32(q_tab, owner)
        tl.store(qb + (local_e * NPES + MY_PE) * MAX_RECV + d3_slot, r, mask=valid)
    dispatch_sync(USE_D8_TMA1D)

    dispatch_rank_barrier(gctr + 0, NUM_SMS, USE_D8_TMA1D)

    if sm_idx == 0:
        d4_owner = tl.where(e_ok, offs_e // EPR, 0)
        d4_le = tl.where(e_ok, offs_e % EPR, 0)
        status = tl.load(expert_send_count + offs_e, mask=e_ok, other=0)
        # Publish per-source counts and aggregate them at each destination rank.
        tl.store(
            peer_i64(recv_tab, d4_owner) + (MY_PE * EPR + d4_le),
            status & 0xFFFFFFFF,
            mask=e_ok,
        )
        tl.atomic_add(
            peer_i64(rsum_tab, d4_owner) + d4_le,
            status,
            mask=e_ok,
            sem="relaxed",
            scope="sys",
        )

    dispatch_nvlink_barrier(
        sig_tab,
        0,
        gctr + 1,
        sm_idx,
        MY_PE,
        NPES,
        NUM_SMS,
        USE_D8_TMA1D,
    )
    tl.atomic_add(meta_ready + sm_idx, 1, sem="release")


@triton.jit
def dispatch_raw_tail(
    rsum_tab,
    recv_tab,
    q_tab,
    meta_tab,
    sf_tab,
    w_tab,
    l1_arrival_count,
    l1_token,
    l1_sf,
    l1_w,
    d8_token0_s,
    d8_state0_s,
    d8_token1_s,
    d8_state1_s,
    tp0,
    tp1,
    tp2,
    tp3,
    tp4,
    tp5,
    tp6,
    tp7,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPR: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
    NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
):
    # Keep the raw ABI tail separate so the dispatch frontend can inline into
    # its warp-specialized partition and receive the correct named barrier ID.
    rsum_l = peer_i64(rsum_tab, MY_PE)
    recv_l = peer_i64(recv_tab, MY_PE)
    queue_l = peer_i32(q_tab, MY_PE)
    metadata_l = peer_i32(meta_tab, MY_PE)
    stage0 = smem_generic_addr(tle.gpu.local_ptr(d8_token0_s, (0,)))
    state0 = smem_generic_addr(tle.gpu.local_ptr(d8_state0_s, (0,)))
    stage1 = smem_generic_addr(tle.gpu.local_ptr(d8_token1_s, (0,)))
    state1 = smem_generic_addr(tle.gpu.local_ptr(d8_state1_s, (0,)))
    npes_v = tl.full((), NPES, tl.int32)
    num_sms_v = tl.full((), NUM_SMS, tl.int32)
    epr_v = tl.full((), EPR, tl.int32)
    max_recv_v = tl.full((), MAX_RECV, tl.int32)
    block_m_v = tl.full((), BLOCK_M, tl.int32)
    k_v = tl.full((), K, tl.int32)
    nsf_v = tl.full((), NSF, tl.int32)
    pool_tokens_v = tl.full((), POOL_TOKENS, tl.int32)
    topk_v = tl.full((), TOPK, tl.int32)
    tle_raw.call(
        d8_unified_dispatch_edsl,
        [
            tl.cast(rsum_l, tl.int64),
            tl.cast(recv_l, tl.int64),
            tl.cast(queue_l, tl.int64),
            tl.cast(metadata_l, tl.int64),
            tl.cast(sf_tab, tl.int64),
            tl.cast(w_tab, tl.int64),
            tl.cast(l1_arrival_count, tl.int64),
            tl.cast(l1_token, tl.int64),
            tl.cast(l1_sf, tl.int64),
            tl.cast(l1_w, tl.int64),
            stage0,
            state0,
            stage1,
            state1,
            tp0,
            tp1,
            tp2,
            tp3,
            tp4,
            tp5,
            tp6,
            tp7,
            npes_v,
            num_sms_v,
            epr_v,
            max_recv_v,
            block_m_v,
            k_v,
            nsf_v,
            pool_tokens_v,
            topk_v,
        ],
    )


@triton.jit
def dispatch_role_dual_pull(
    topk_local,
    sf_tab,
    w_tab,
    q_tab,
    recv_tab,
    rsum_tab,
    sig_tab,
    meta_tab,
    expert_send_count,
    l1_arrival_count,
    gctr,
    meta_ready,
    l1_token,
    l1_sf,
    l1_w,
    d8_token0_s,
    d8_state0_s,
    d8_token1_s,
    d8_state1_s,
    tp0,
    tp1,
    tp2,
    tp3,
    tp4,
    tp5,
    tp6,
    tp7,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    NEXP: tl.constexpr,
    EPR: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
    NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ITERS: tl.constexpr,
    NEXP_POW2: tl.constexpr,
    USE_D8_TMA1D: tl.constexpr,
):
    # D1-D5: both warps cooperate under the same role-scoped barriers.
    dispatch_frontend_dual_pull(
        topk_local,
        q_tab,
        recv_tab,
        rsum_tab,
        sig_tab,
        expert_send_count,
        gctr,
        meta_ready,
        MY_PE,
        NPES,
        NUM_SMS,
        NTOK,
        TOPK,
        NEXP,
        EPR,
        MAX_RECV,
        BLOCK_R,
        ITERS,
        NEXP_POW2,
        USE_D8_TMA1D,
    )

    # D6-D9: keep the raw ABI tail in a separate helper.  This wrapper then
    # contains the frontend barriers in the warp-specialized partition.
    dispatch_raw_tail(
        rsum_tab,
        recv_tab,
        q_tab,
        meta_tab,
        sf_tab,
        w_tab,
        l1_arrival_count,
        l1_token,
        l1_sf,
        l1_w,
        d8_token0_s,
        d8_state0_s,
        d8_token1_s,
        d8_state1_s,
        tp0,
        tp1,
        tp2,
        tp3,
        tp4,
        tp5,
        tp6,
        tp7,
        MY_PE,
        NPES,
        NUM_SMS,
        EPR,
        MAX_RECV,
        BLOCK_M,
        K,
        NSF,
        POOL_TOKENS,
        TOPK,
    )


@triton.jit
def ws_megakernel(
    topk_local,
    sf_tab,
    w_tab,
    q_tab,
    recv_tab,
    rsum_tab,
    sig_tab,
    meta_tab,
    cb_tab,
    cb_local,
    expert_send_count,
    l1_arrival_count,
    l2_arrival_mask,
    gctr,
    meta_ready,
    l1_token,
    l1_sf,
    l1_w,
    l2_acts,
    l2_sf,
    w1_sf,
    w2_sf,
    final_y,
    a1_desc,
    b1_desc,
    a2_desc,
    b2_desc,
    sfa1_desc,
    sfa2_desc,
    tp0,
    tp1,
    tp2,
    tp3,
    tp4,
    tp5,
    tp6,
    tp7,
    USE_D8_TMA1D: tl.constexpr,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    NEXP: tl.constexpr,
    EPR: tl.constexpr,
    EPR_POW2: tl.constexpr,
    EPW: tl.constexpr,
    MAX_RECV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    INTER: tl.constexpr,
    L1_N: tl.constexpr,
    L2_N: tl.constexpr,
    NK1: tl.constexpr,
    NK2: tl.constexpr,
    NL1N: tl.constexpr,
    NL2N: tl.constexpr,
    L1_OUT_N: tl.constexpr,
    NPAIR: tl.constexpr,
    NSF: tl.constexpr,
    POOL_TOKENS: tl.constexpr,
    NEXP_POW2: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ITERS: tl.constexpr,
    STAGES: tl.constexpr,
):
    tl.static_assert(USE_D8_TMA1D == 2, "raw D8 dispatch requires TMA1D mode")
    tl.static_assert(BLOCK_R == 256, "D8 dispatch requires BLOCK_R=256")

    a_s = tle.gpu.alloc(
        [STAGES, BLOCK_M, BLOCK_K], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
    )
    b_s = tle.gpu.alloc(
        [STAGES, BLOCK_N, BLOCK_K], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
    )
    sfa_s = tle.gpu.alloc(
        [STAGES, 2, BLOCK_M], dtype=tl.float32, layout=None, scope=tle.gpu.smem
    )
    cd_s = tle.gpu.alloc(
        [2, BLOCK_M // 2, BLOCK_N],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dst_rows_s = tle.gpu.alloc(
        [2, BLOCK_M // 2],
        dtype=tl.uint64,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    d8_token0_s = tle.gpu.alloc(
        [K],
        dtype=tl.float8e4nv,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    d8_token1_s = tle.gpu.alloc(
        [K],
        dtype=tl.float8e4nv,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    d8_state0_s = tle.gpu.alloc(
        [1],
        dtype=tl.uint64,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    d8_state1_s = tle.gpu.alloc(
        [1],
        dtype=tl.uint64,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    p = tle.pipe(
        capacity=STAGES,
        scope="cta",
        name="megamoe_gemm",
        readers=["m0", "m1"],
        a=a_s,
        b=b_s,
        sfa=sfa_s,
    )

    tle.gpu.warp_specialize(
        [
            (
                math_role,
                (
                    p.reader(name="m0"),
                    cd_s,
                    dst_rows_s,
                    meta_ready,
                    rsum_tab,
                    cb_tab,
                    sig_tab,
                    meta_tab,
                    cb_local,
                    l1_w,
                    l2_acts,
                    l2_sf,
                    l2_arrival_mask,
                    topk_local,
                    final_y,
                    gctr,
                    w1_sf,
                    w2_sf,
                    0,
                    MY_PE,
                    NPES,
                    NUM_SMS,
                    EPR,
                    EPR_POW2,
                    EPW,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    INTER,
                    NK1,
                    NK2,
                    NL1N,
                    NL2N,
                    L1_OUT_N,
                    NPAIR,
                    POOL_TOKENS,
                    NTOK,
                    TOPK,
                    K,
                    K_POW2,
                ),
            ),
            (
                math_role,
                (
                    p.reader(name="m1"),
                    cd_s,
                    dst_rows_s,
                    meta_ready,
                    rsum_tab,
                    cb_tab,
                    sig_tab,
                    meta_tab,
                    cb_local,
                    l1_w,
                    l2_acts,
                    l2_sf,
                    l2_arrival_mask,
                    topk_local,
                    final_y,
                    gctr,
                    w1_sf,
                    w2_sf,
                    1,
                    MY_PE,
                    NPES,
                    NUM_SMS,
                    EPR,
                    EPR_POW2,
                    EPW,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    INTER,
                    NK1,
                    NK2,
                    NL1N,
                    NL2N,
                    L1_OUT_N,
                    NPAIR,
                    POOL_TOKENS,
                    NTOK,
                    TOPK,
                    K,
                    K_POW2,
                ),
            ),
            (
                dispatch_role_dual_pull,
                (
                    topk_local,
                    sf_tab,
                    w_tab,
                    q_tab,
                    recv_tab,
                    rsum_tab,
                    sig_tab,
                    meta_tab,
                    expert_send_count,
                    l1_arrival_count,
                    gctr,
                    meta_ready,
                    l1_token,
                    l1_sf,
                    l1_w,
                    d8_token0_s,
                    d8_state0_s,
                    d8_token1_s,
                    d8_state1_s,
                    tp0,
                    tp1,
                    tp2,
                    tp3,
                    tp4,
                    tp5,
                    tp6,
                    tp7,
                    MY_PE,
                    NPES,
                    NUM_SMS,
                    NTOK,
                    TOPK,
                    NEXP,
                    EPR,
                    MAX_RECV,
                    BLOCK_M,
                    K,
                    NSF,
                    POOL_TOKENS,
                    BLOCK_R,
                    ITERS,
                    NEXP_POW2,
                    USE_D8_TMA1D,
                ),
            ),
            (
                loader_role,
                (
                    p.writer(),
                    rsum_tab,
                    l1_arrival_count,
                    l2_arrival_mask,
                    a1_desc,
                    b1_desc,
                    a2_desc,
                    b2_desc,
                    sfa1_desc,
                    sfa2_desc,
                    True,
                    MY_PE,
                    NUM_SMS,
                    EPR,
                    EPW,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    L1_N,
                    L2_N,
                    NK1,
                    NK2,
                    NL1N,
                    NL2N,
                    NPES,
                ),
            ),
            (
                loader_role,
                (
                    p.writer(),
                    rsum_tab,
                    l1_arrival_count,
                    l2_arrival_mask,
                    a1_desc,
                    b1_desc,
                    a2_desc,
                    b2_desc,
                    sfa1_desc,
                    sfa2_desc,
                    False,
                    MY_PE,
                    NUM_SMS,
                    EPR,
                    EPW,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    L1_N,
                    L2_N,
                    NK1,
                    NK2,
                    NL1N,
                    NL2N,
                    NPES,
                ),
            ),
        ],
        [4, 2, 1, 1],
        [224, 48, 48, 48],
    )


def _fill_tensor_bytes(tensor, value: int) -> None:
    block = 1024
    nbytes = tensor.numel() * tensor.element_size()
    fill_bytes[(triton.cdiv(nbytes, block),)](
        tensor,
        nbytes,
        VALUE=value,
        BLOCK=block,
    )


def _zero_symmetric(table, rank: int, nbytes: int) -> None:
    block = 1024
    sym_zero[(triton.cdiv(nbytes, block),)](
        table,
        rank,
        nbytes,
        BLOCK=block,
    )


def _stage_symmetric(source, table, rank: int) -> None:
    block = 1024
    nbytes = source.numel() * source.element_size()
    sym_h2d[(triton.cdiv(nbytes, block),)](
        source,
        table,
        rank,
        nbytes,
        BLOCK=block,
    )


def launch_megamoe_fp8(
    input_fp8,
    input_scales,
    topk_ids,
    topk_weights,
    w1,
    w1_scales,
    w2,
    w2_scales,
    context: MegaMoEContext,
    output=None,
    *,
    stages: int = 4,
):
    """Launch the SM90 FP8 expert-parallel MegaMoE kernel asynchronously."""

    block_m = 128
    block_n = 128
    block_k = 128
    block_r = 256

    assert input_fp8.ndim == 2
    ntok, hidden = input_fp8.shape
    assert ntok > 0
    assert hidden == context.hidden_size and 0 < hidden <= 4096
    assert hidden % block_k == 0
    assert context.intermediate_size > 0
    assert context.intermediate_size % block_k == 0
    assert context.world_size in (2, 4, 8)
    assert 0 <= context.rank < context.world_size
    assert context.num_experts % context.world_size == 0
    assert context.num_sms > 0
    assert context.pool_tokens >= block_m
    assert context.pool_tokens % block_m == 0
    assert context.max_recv > 0
    assert 0 < stages

    device = input_fp8.device
    assert device.type == "cuda"
    num_sf = hidden // 128
    assert topk_ids.ndim == 2
    topk = topk_ids.shape[1]
    assert 0 < topk <= context.num_experts
    assert ntok <= context.max_tokens
    assert topk <= context.max_topk
    assert tuple(input_scales.shape) == (ntok, num_sf)
    assert tuple(topk_ids.shape) == (ntok, topk)
    assert tuple(topk_weights.shape) == (ntok, topk)

    ranks = context.world_size
    experts_per_rank = context.num_experts // ranks
    intermediate = context.intermediate_size
    nl1n = intermediate // 64
    l1_n = 2 * intermediate
    l2_n = hidden
    nl2n = hidden // block_n
    nk1 = hidden // block_k
    nk2 = intermediate // block_k
    pool_tokens = context.pool_tokens
    # These are kernel layout requirements. Communication buffers and scratch
    # capacities are part of MegaMoEContext's caller-owned contract.
    assert input_fp8.dtype == torch.float8_e4m3fn and input_fp8.is_contiguous()
    assert input_scales.dtype == torch.float32 and input_scales.is_contiguous()
    assert topk_ids.dtype in (torch.int32, torch.int64) and topk_ids.is_contiguous()
    assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
    assert w1.dtype == torch.float8_e4m3fn and w1.is_contiguous()
    assert w1_scales.dtype == torch.float32 and w1_scales.is_contiguous()
    assert w2.dtype == torch.float8_e4m3fn and w2.is_contiguous()
    assert w2_scales.dtype == torch.float32 and w2_scales.is_contiguous()
    assert all(
        tensor.device == device
        for tensor in (
            input_scales,
            topk_ids,
            topk_weights,
            w1,
            w1_scales,
            w2,
            w2_scales,
        )
    )
    assert tuple(w1.shape) == (experts_per_rank, l1_n, hidden)
    assert tuple(w1_scales.shape) == (experts_per_rank, nl1n, nk1)
    assert tuple(w2.shape) == (experts_per_rank, l2_n, intermediate)
    assert tuple(w2_scales.shape) == (experts_per_rank, nl2n, nk2)

    if output is None:
        output = torch.empty(
            (ntok, hidden),
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        assert output.device == device
        assert output.dtype == torch.bfloat16 and output.is_contiguous()
        assert tuple(output.shape) == (ntok, hidden)

    peer_tokens = tuple(int(pointer) for pointer in context.token_peer_ptrs)
    assert len(peer_tokens) == ranks
    assert all(pointer != 0 for pointer in peer_tokens)
    peer_tokens = (peer_tokens + (peer_tokens[0],) * 8)[:8]

    a1_desc = TensorDescriptor(
        context.l1_tokens,
        shape=[pool_tokens, hidden],
        strides=[hidden, 1],
        block_shape=[block_m, block_k],
    )
    b1_desc = TensorDescriptor(
        w1.view(experts_per_rank * l1_n, hidden),
        shape=[experts_per_rank * l1_n, hidden],
        strides=[hidden, 1],
        block_shape=[block_n, block_k],
    )
    a2_desc = TensorDescriptor(
        context.l2_activations,
        shape=[pool_tokens, intermediate],
        strides=[intermediate, 1],
        block_shape=[block_m, block_k],
    )
    b2_desc = TensorDescriptor(
        w2.view(experts_per_rank * l2_n, intermediate),
        shape=[experts_per_rank * l2_n, intermediate],
        strides=[intermediate, 1],
        block_shape=[block_n, block_k],
    )
    sfa1_desc = TensorDescriptor(
        context.l1_scales.view(-1)[: num_sf * pool_tokens].view(num_sf, pool_tokens),
        shape=[num_sf, pool_tokens],
        strides=[pool_tokens, 1],
        block_shape=[1, block_m],
    )
    sfa2_desc = TensorDescriptor(
        context.l2_scales.view(-1)[: nl1n * pool_tokens].view(nl1n, pool_tokens),
        shape=[nl1n, pool_tokens],
        strides=[pool_tokens, 1],
        block_shape=[2, block_m],
    )

    _fill_tensor_bytes(context.expert_send_count, 0)
    _fill_tensor_bytes(context.l1_arrival_count, 0)
    _fill_tensor_bytes(context.l2_arrival_mask, 0)
    _fill_tensor_bytes(context.global_counters, 0)
    _fill_tensor_bytes(context.metadata_ready, 0)
    _zero_symmetric(
        context.recv_count_ptrs,
        context.rank,
        ranks * experts_per_rank * 8,
    )
    _zero_symmetric(context.recv_sum_ptrs, context.rank, experts_per_rank * 8)
    _zero_symmetric(context.signal_ptrs, context.rank, 16 * 4)
    _stage_symmetric(input_fp8, context.token_ptrs, context.rank)
    _stage_symmetric(input_scales, context.scale_ptrs, context.rank)
    _stage_symmetric(topk_weights, context.route_weight_ptrs, context.rank)

    stream = torch.cuda.current_stream(device)
    context.barrier_all_on_stream(stream)

    routes = ntok * topk
    iters = triton.cdiv(routes, context.num_sms * block_r)
    epr_pow2 = 1 << (experts_per_rank - 1).bit_length()
    nexp_pow2 = 1 << (context.num_experts - 1).bit_length()
    hidden_pow2 = 1 << (hidden - 1).bit_length()

    ws_megakernel[(context.num_sms,)](
        topk_ids,
        context.scale_ptrs,
        context.route_weight_ptrs,
        context.queue_ptrs,
        context.recv_count_ptrs,
        context.recv_sum_ptrs,
        context.signal_ptrs,
        context.metadata_ptrs,
        context.combine_ptrs,
        context.combine_local,
        context.expert_send_count,
        context.l1_arrival_count,
        context.l2_arrival_mask,
        context.global_counters,
        context.metadata_ready,
        context.l1_tokens,
        context.l1_scales,
        context.l1_weights,
        context.l2_activations,
        context.l2_scales,
        w1_scales,
        w2_scales,
        output,
        a1_desc,
        b1_desc,
        a2_desc,
        b2_desc,
        sfa1_desc,
        sfa2_desc,
        *peer_tokens,
        USE_D8_TMA1D=2,
        MY_PE=context.rank,
        NPES=ranks,
        NUM_SMS=context.num_sms,
        NTOK=ntok,
        TOPK=topk,
        NEXP=context.num_experts,
        EPR=experts_per_rank,
        EPR_POW2=epr_pow2,
        EPW=min(8, experts_per_rank),
        MAX_RECV=context.max_recv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        K=hidden,
        INTER=intermediate,
        L1_N=l1_n,
        L2_N=l2_n,
        NK1=nk1,
        NK2=nk2,
        NL1N=nl1n,
        NL2N=nl2n,
        L1_OUT_N=block_n // 2,
        NPAIR=block_n // 16,
        NSF=num_sf,
        POOL_TOKENS=pool_tokens,
        NEXP_POW2=nexp_pow2,
        K_POW2=hidden_pow2,
        BLOCK_R=block_r,
        ITERS=iters,
        STAGES=stages,
        num_warps=LOADER_WARPS,
    )

    context.barrier_all_on_stream(stream)
    return output

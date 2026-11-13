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

"""Implement the fused MUSA MegaMoE kernel and standard launch.

Terminology
-----------
A rank owns E of the E*R experts; MY identifies the local rank, T the local
token count, TOP the routes per token, H the hidden width and I the
intermediate width. CAP bounds each destination-expert queue, while POOL
counts padded 128-token work tiles. Epoch orders repeated collective launches.
A descriptor names a caller-owned tensor and its fixed TMA transfer tile.

Memory layout
-------------
Peer tables hold one int64 address per rank for symmetric input [T,H],
input scales [T,H/128], route weights [T,TOP], expert queues [E,CAP],
route counts [E], partial outputs [T,TOP,H] and three signals. Scratch input
and quantized activation pools use token rows, with scale groups on their
last dimension. Packed gu256 weights interleave 128 gate rows and 128 up
rows within each expert; down weights use contiguous [E*H,I] rows.
Grid is the local int32 rendezvous counter array. CTA barriers coordinate
partitions; Grid/Signals atomics provide the stated GPU/system publication.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

if TYPE_CHECKING:
    from .api import MegaMoEContext


# Retain the per-kernel codegen workaround; this is not a compiler patch.
LAUNCH_OPTIONS = {
    "num_warps": 4,
    "num_stages": 1,
    "llc_options": "-enable-misched=false",
}


@contextmanager
def _musa_llc_opt_disabled():
    """Scope the MUSA LLC optimisation pass off while this kernel compiles.

    ``flaggems_vllm.ops.mhc.mhc_pre`` enables ``TRITON_MUSA_ENABLE_LLC_OPT`` at
    import time.  With that pass enabled, the MUSA linear-scan register
    allocator crashes for every multi-rank specialisation of this kernel, so the
    knob is disabled for the duration of the compile and restored afterwards.
    """
    key = "TRITON_MUSA_ENABLE_LLC_OPT"
    previous = os.environ.get(key)
    os.environ[key] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def check_barrier_map(ir: str) -> int:
    """Validate the fused kernel's eighteen barriers and return their count.

    The two SMALL branches in megamoe_kernel each explicitly declare eight
    alloc_barrier(expect_bytes=...) TMA barriers, sync0/sync1/dispatch_sync
    with arrive_count=4, and join with arrive_count=12. Existing lowered IR
    contains six additional arrival=4 barriers: their ownership is unassigned,
    and they are included only by arrival count. No source-role attribution
    or stability guarantee is inferred for those six barriers.

    Accept eighteen unique consecutive indices at any base with arrival
    multiset [1]*8 + [4]*9 + [12]. Reject changes to that structure before
    launch; the index base is a lowering detail and is not fixed here.
    """
    pairs = re.findall(
        r"call void @llvm\.musa\.async\.init\.arrival"
        r"\(i32 (\d+), i32 (\d+), i32 0\)",
        ir,
    )
    counts = {int(index): int(arrivals) for index, arrivals in pairs}
    indices = sorted(counts)
    expected = [1] * 8 + [4] * 9 + [12]
    consecutive = len(indices) == 18 and indices == list(
        range(indices[0], indices[0] + 18)
    )
    if len(pairs) != 18 or not consecutive or sorted(counts.values()) != expected:
        raise RuntimeError(
            f"Unexpected fused MegaMoE barriers: counts={counts}, "
            f"expected_multiset={expected}, expected_count=18, "
            f"expected_consecutive=True, observed_declarations={len(pairs)}"
        )
    return len(counts)


@triton.jit
def peers(tab, pe, DT: tl.constexpr):
    # ---- Resolve a rank-owned symmetric pointer
    return tl.load(tab + pe).to(tl.pointer_type(DT))


@triton.jit
def scatter_rendezvous(
    Grid,
    Signals,
    MY: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    OFF: tl.constexpr,
    SLOT,
    EPOCH,
):
    # ---- Publish CTA progress and rendezvous across ranks
    if R == 1:
        tl.atomic_add(Grid + OFF, 1, sem="acq_rel", scope="gpu")
        seen = tl.atomic_add(Grid + OFF, 0, sem="acquire", scope="gpu")
        while seen < C * EPOCH:
            seen = tl.atomic_add(Grid + OFF, 0, sem="acquire", scope="gpu")
    else:
        tl.atomic_add(Grid + OFF, 1, sem="release", scope="sys")
        z = tl.atomic_add(Grid + OFF, 0, sem="acquire", scope="sys")
        while z < C * EPOCH:
            z = tl.atomic_add(Grid + OFF, 0, sem="acquire", scope="sys")
        if tl.program_id(0) == 0:
            for pe in tl.static_range(R):
                tl.atomic_add(
                    peers(Signals, pe, tl.int32) + SLOT, 1, sem="release", scope="sys"
                )
            local = peers(Signals, MY, tl.int32)
            z = tl.atomic_add(local + SLOT, 0, sem="acquire", scope="sys")
            while z < R * EPOCH:
                z = tl.atomic_add(local + SLOT, 0, sem="acquire", scope="sys")
        tl.atomic_add(Grid + OFF + 1, 1, sem="release", scope="sys")
        z = tl.atomic_add(Grid + OFF + 1, 0, sem="acquire", scope="sys")
        while z < C * EPOCH:
            z = tl.atomic_add(Grid + OFF + 1, 0, sem="acquire", scope="sys")


@triton.jit
def quantize_64(x, route_weight):
    # ---- Derive per-block scales and quantize routed activation
    FP8_MAX: tl.constexpr = 448.0
    magnitude = tl.max(tl.abs(x), 1) * tl.abs(route_weight) / FP8_MAX
    scale = tl.where(
        magnitude > 0,
        tl.exp2(tl.ceil(tl.log2(tl.where(magnitude > 0, magnitude, 1.0)))),
        1.0,
    )
    return (x * (route_weight / scale)[:, None]).to(tl.float8e4nv), scale


@triton.jit
def local_gather_rendezvous(Grid, C: tl.constexpr, Epoch):
    # ---- Rendezvous local gather workers at GPU scope
    tl.atomic_add(Grid + 6, 1, sem="release", scope="gpu")
    seen = tl.atomic_add(Grid + 6, 0, sem="acquire", scope="gpu")
    while seen < C * Epoch:
        seen = tl.atomic_add(Grid + 6, 0, sem="acquire", scope="gpu")


@triton.jit
def worker(
    X,
    S,
    Ids,
    RW,
    S1,
    S2,
    QueueTable,
    CountTable,
    Grid,
    Part,
    Y,
    Signals,
    A,
    AS,
    B1,
    B2,
    Q,
    QS,
    Input,
    InputScale,
    Weight,
    Quant,
    QuantScale,
    Ready,
    QuantReady,
    Epoch,
    ba0,
    ba1,
    bw0,
    bw1,
    qb,
    phase_bar,
    join,
    ROLE: tl.constexpr,
    T,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOP: tl.constexpr,
    CTAS: tl.constexpr,
    CAP: tl.constexpr,
    MY: tl.constexpr,
    R: tl.constexpr,
    POOL: tl.constexpr,
    EPW: tl.constexpr,
):
    # ---- Gather inputs, execute gate/up and down tasks, publish partials
    SCALE_GROUP: tl.constexpr = 128
    QUANT_BLOCK: tl.constexpr = 64
    PAIR_BLOCK: tl.constexpr = 256
    cta = tl.program_id(0)
    queue = peers(QueueTable, MY, tl.int32)
    counts = peers(CountTable, MY, tl.int32)
    BM: tl.constexpr = 32
    BN1: tl.constexpr = 128
    BN2: tl.constexpr = 256
    row = tl.arange(0, BM)
    col = tl.arange(0, BN2)
    lanes = tle.gpu.set_layout(
        tl.arange(0, 256), tle.gpu.BlockEncoding([1], [32], [4], [0])
    )
    # QuantScale capacity is POOL*256*(I//64); actual q uses the low half.
    META: tl.constexpr = POOL * 128 * (I // QUANT_BLOCK)
    COUNTERS: tl.constexpr = META + POOL * 16
    TASKS: tl.constexpr = COUNTERS + POOL * 8
    qbs = tle.gpu.alloc([I // SCALE_GROUP, BM, 128], tl.float8e4nv)

    ready = Grid + (2 if R == 1 else 3)
    seen = tl.atomic_add(ready, 0, sem="acquire", scope="sys")
    while seen < CTAS * Epoch:
        seen = tl.atomic_add(ready, 0, sem="acquire", scope="sys")
    if ROLE == 0:
        if cta == 0:
            tl.store(QuantScale + TASKS, 0.0)
            tl.store(QuantScale + TASKS + 1, 0.0)
        for base in range(cta * 256, POOL * 8, CTAS * 256):
            slot = base + lanes
            tl.store(QuantScale + COUNTERS + slot, 0.0, slot < POOL * 8)
    if R == 2 or R == 4:
        tc_ks = tle.gpu.set_layout(
            tl.arange(0, H), tle.gpu.BlockEncoding([16], [32], [4], [0])
        )
        tc_wr = tle.gpu.set_layout(
            tl.arange(0, TOP), tle.gpu.BlockEncoding([1], [32], [4], [0])
        )
        tc_ss = tle.gpu.set_layout(
            tl.arange(0, H // SCALE_GROUP), tle.gpu.BlockEncoding([1], [32], [4], [0])
        )
        for tc_pe in tl.static_range(R):
            if tc_pe != MY:
                tc_xp = peers(X, tc_pe, tl.float8e4nv)
                tc_sp = peers(S, tc_pe, tl.float32)
                tc_wp = peers(RW, tc_pe, tl.float32)
                for tc_tok in range(tl.program_id(0) * 2 + ROLE, T, CTAS * 2):
                    tc_xv = tl.load(tc_xp + tc_tok * H + tc_ks)
                    tc_sv = tl.load(tc_sp + tc_tok * (H // SCALE_GROUP) + tc_ss)
                    tc_wv = tl.load(tc_wp + tc_tok * TOP + tc_wr)
                    tl.store(Input + CAP * H + (tc_pe * T + tc_tok) * H + tc_ks, tc_xv)
                    tl.store(
                        InputScale
                        + CAP * (H // SCALE_GROUP)
                        + (tc_pe * T + tc_tok) * (H // SCALE_GROUP)
                        + tc_ss,
                        tc_sv,
                    )
                    tl.store(
                        QuantScale
                        + POOL * 128 * (I // QUANT_BLOCK)
                        + POOL * 32
                        + (tc_pe * T + tc_tok) * TOP
                        + tc_wr,
                        tc_wv,
                    )
        fence_system(Grid)
        tle.gpu.barrier_arrive(phase_bar, phaseIdx=(0))
        tle.gpu.barrier_wait(phase_bar, phaseIdx=(0))
        tle.gpu.barrier_arrive(phase_bar, phaseIdx=1 - (0))
        tle.gpu.barrier_wait(phase_bar, phaseIdx=1 - (0))
        tl.atomic_add(Grid + 7, 1, sem="release", scope="gpu")
        tc_seen = tl.atomic_add(Grid + 7, 0, sem="acquire", scope="gpu")
        # Two math partitions per CTA arrive at this GPU-scope barrier.
        while tc_seen < CTAS * 2 * Epoch:
            tc_seen = tl.atomic_add(Grid + 7, 0, sem="acquire", scope="gpu")
    worker_id = cta * 2 + ROLE
    # Two math partitions per CTA arrive at this GPU-scope gather rendezvous;
    # the system-scope cross-rank rendezvous below counts CTAs instead.
    MATH_PARTITIONS: tl.constexpr = 2
    workers: tl.constexpr = CTAS * MATH_PARTITIONS
    experts = tl.arange(0, triton.next_power_of_2(E))[None, :]
    counts_by_expert = tl.load(counts + experts, experts < E, 0)
    expert_tiles = tl.cdiv(counts_by_expert, BM)
    tile_ends = tl.cumsum(expert_tiles, axis=1)
    tile_count = tl.reshape(tl.sum(expert_tiles, axis=1), ())

    # ---- stage every tile's routed input once
    source = worker_id
    while source < tile_count:
        earlier = tile_ends <= source
        expert = tl.reshape(tl.sum(earlier.to(tl.int32), axis=1), ())
        tile = source - tl.reshape(tl.max(tl.where(earlier, tile_ends, 0), axis=1), ())
        # Build once during gather; tasks below perform two scalar loads.
        tl.store(QuantScale + META + source * 2, expert.to(tl.float32))
        tl.store(QuantScale + META + source * 2 + 1, tile.to(tl.float32))
        count = tl.load(counts + expert)
        gather: tl.constexpr = tle.gpu.BlockEncoding([1, 16], [1, 32], [4, 1], [1, 0])
        rr = tle.gpu.set_layout(tl.arange(0, 4), tle.gpu.SlicedEncoding(1, gather))
        kk = tle.gpu.set_layout(tl.arange(0, H), tle.gpu.SlicedEncoding(0, gather))
        sk = tle.gpu.set_layout(
            tl.arange(0, H // SCALE_GROUP), tle.gpu.SlicedEncoding(0, gather)
        )
        if R == 1:
            xp = peers(X, MY, tl.float8e4nv)
            sfp = peers(S, MY, tl.float32)
            wr = peers(RW, MY, tl.float32)
            for offset in range(0, tl.minimum(BM, count - tile * BM), 4):
                rr0 = offset + rr
                idx = tile * BM + rr0
                rvalid = idx < count
                route_id = tl.load(queue + expert * CAP + idx, rvalid, 0)
                tok = route_id // TOP
                x = tl.load(xp + tok[:, None] * H + kk[None, :], rvalid[:, None], 0.0)
                tl.store(
                    Input + (source * BM + rr0[:, None]) * H + kk[None, :],
                    x,
                    rvalid[:, None],
                )
                sf = tl.load(
                    sfp + tok[:, None] * (H // SCALE_GROUP) + sk[None, :],
                    rvalid[:, None],
                    0.0,
                )
                tl.store(
                    InputScale
                    + (source * (H // SCALE_GROUP) + sk[None, :]) * BM
                    + rr0[:, None],
                    sf,
                    rvalid[:, None],
                )
                rwv = tl.load(wr + route_id, rvalid, 0.0)
                tl.store(Weight + source * BM + rr0, rwv, rvalid)
        else:
            for offset in range(0, tl.minimum(BM, count - tile * BM), 4):
                rr0 = offset + rr
                idx = tile * BM + rr0
                rvalid = idx < count
                route_id = tl.load(queue + expert * CAP + idx, rvalid, 0)
                src = route_id // (T * TOP)
                local_route = route_id % (T * TOP)
                tok = local_route // TOP
                if R == 2 or R == 4:
                    xp = tl.where(
                        src == MY,
                        peers(X, MY, tl.float8e4nv),
                        Input + CAP * H + src * T * H,
                    )[:, None]
                    sfp = tl.where(
                        src == MY,
                        peers(S, MY, tl.float32),
                        InputScale
                        + CAP * (H // SCALE_GROUP)
                        + src * T * (H // SCALE_GROUP),
                    )[:, None]
                else:
                    xp = peers(X, src, tl.float8e4nv)[:, None]
                    sfp = peers(S, src, tl.float32)[:, None]
                x = tl.load(xp + tok[:, None] * H + kk[None, :], rvalid[:, None], 0.0)
                tl.store(
                    Input + (source * BM + rr0[:, None]) * H + kk[None, :],
                    x,
                    rvalid[:, None],
                )
                sf = tl.load(
                    sfp + tok[:, None] * (H // SCALE_GROUP) + sk[None, :],
                    rvalid[:, None],
                    0.0,
                )
                tl.store(
                    InputScale
                    + (source * (H // SCALE_GROUP) + sk[None, :]) * BM
                    + rr0[:, None],
                    sf,
                    rvalid[:, None],
                )
                if R == 2 or R == 4:
                    wptr = tl.where(
                        src == MY,
                        peers(RW, MY, tl.float32),
                        QuantScale
                        + POOL * 128 * (I // QUANT_BLOCK)
                        + POOL * 32
                        + src * T * TOP,
                    )
                    rwv = tl.load(wptr + local_route, rvalid, 0.0)
                else:
                    rwv = tl.load(peers(RW, src, tl.float32) + local_route, rvalid, 0.0)
                tl.store(Weight + source * BM + rr0, rwv, rvalid)
        fence_system(Grid)
        source += workers
    tle.gpu.barrier_arrive(qb, phaseIdx=0)
    tle.gpu.barrier_wait(qb, phaseIdx=0)
    local_gather_rendezvous(Grid, workers, Epoch)

    # ---- up tasks: one (tile, 128-column I block) per task
    in_sf = tle.gpu.alloc(
        [H // SCALE_GROUP, BM], tl.float32, nv_mma_shared_layout=False
    )
    q_sf = tle.gpu.alloc(  # noqa: F841
        [I // SCALE_GROUP, BM], tl.float32, nv_mma_shared_layout=False
    )
    sfkb = tl.arange(0, H // SCALE_GROUP)
    qskb = tl.arange(0, I // SCALE_GROUP)  # noqa: F841
    group = tl.where(tile_count >= workers // 2, 2, 1)
    tasks_per_tile = 4 // group
    qp = 1
    u = tl.atomic_add(QuantScale + TASKS, 1.0, sem="relaxed", scope="gpu").to(tl.int32)
    while u < tasks_per_tile * tile_count:
        source = u // tasks_per_tile
        ib_group = u % tasks_per_tile
        expert = tl.load(QuantScale + META + source * 2).to(tl.int32)
        tile = tl.load(QuantScale + META + source * 2 + 1).to(tl.int32)
        route_weight = tl.load(Weight + source * BM + row)
        a0 = tle.gpu.alloc([BM, 128], tl.float8e4nv)
        w0 = tle.gpu.alloc([2 * BN1, 128], tl.float8e4nv)
        a1 = tle.gpu.alloc([BM, 128], tl.float8e4nv)
        w1 = tle.gpu.alloc([2 * BN1, 128], tl.float8e4nv)
        ab, wb = (a0, a1), (w0, w1)
        ba, bw = (ba0, ba1), (bw0, bw1)
        sf_all = tl.load(
            InputScale
            + (source * (H // SCALE_GROUP) + sfkb[:, None]) * BM
            + row[None, :]
        )
        tl.store(tle.gpu.local_ptr(in_sf), sf_all)
        tle.gpu.barrier_arrive(qb, phaseIdx=qp)
        tle.gpu.barrier_wait(qb, phaseIdx=qp)
        qp = 1 - qp
        for ig in range(group):
            ib = ib_group * group + ig
            first_weight = expert * (2 * I) + ib * (2 * BN1)
            tle.gpu.copy(A, a0, [BM, 128], [source * BM, 0], barrier=ba0)
            tle.gpu.copy(B1, w0, [2 * BN1, 128], [first_weight, 0], barrier=bw0)
            acc = tl.zeros((BM, 2 * BN1), tl.float32)
            for pair in range(H // PAIR_BLOCK):
                for side in tl.static_range(2):
                    kb = pair * 2 + side
                    if kb + 1 < H // SCALE_GROUP:
                        k_next = ((kb + 1) % (H // SCALE_GROUP)) * 128
                        tle.gpu.copy(
                            A,
                            ab[1 - side],
                            [BM, 128],
                            [source * BM, k_next],
                            barrier=ba[1 - side],
                        )
                        tle.gpu.copy(
                            B1,
                            wb[1 - side],
                            [2 * BN1, 128],
                            [first_weight, k_next],
                            barrier=bw[1 - side],
                        )
                    sf = tl.load(
                        tle.gpu.local_ptr(in_sf, (tl.zeros((BM,), tl.int32) + kb, row))
                    )
                    base = (expert * (2 * I // 128) + ib) * (H // SCALE_GROUP) + kb
                    gs = tl.load(S1 + base)
                    us = tl.load(S1 + base + (I // SCALE_GROUP) * (H // SCALE_GROUP))
                    tle.gpu.barrier_wait(ba[side], phaseIdx=pair % 2)
                    tle.gpu.barrier_wait(bw[side], phaseIdx=pair % 2)
                    product = tle.gpu.wgmma(ab[side], wb[side], trans_b=True)
                    product = tle.gpu.wgmma_wait(0, product)
                    scale_b = tl.reshape(
                        tl.trans(
                            tl.join(
                                tl.broadcast_to(gs, (BN1,)), tl.broadcast_to(us, (BN1,))
                            )
                        ),
                        (2 * BN1,),
                    )
                    acc += product * (sf[:, None] * scale_b[None, :])
            pair_acc = tl.permute(tl.reshape(acc, (BM, 2, BN1)), (0, 2, 1))
            gate, up = tl.split(pair_acc)
            activated = gate / (1.0 + tl.exp(-gate)) * up
            q128, common = quantize_64(activated, route_weight)
            tl.store(
                Quant
                + (source * BM + row[:, None]) * I
                + ib * 128
                + tl.arange(0, 128)[None, :],
                q128,
            )
            tl.store(QuantScale + (source * (I // SCALE_GROUP) + ib) * BM + row, common)
        fence_system(Grid)
        tle.gpu.barrier_arrive(qb, phaseIdx=qp)
        tle.gpu.barrier_wait(qb, phaseIdx=qp)
        qp = 1 - qp
        tl.atomic_add(QuantScale + COUNTERS + source, 1.0, sem="release", scope="gpu")
        u = tl.atomic_add(QuantScale + TASKS, 1.0, sem="relaxed", scope="gpu").to(
            tl.int32
        )
    fence_system(Grid)

    # ---- down tasks: one (tile, 256-column H block) per task
    d = tl.atomic_add(QuantScale + TASKS + 1, 1.0, sem="relaxed", scope="gpu").to(
        tl.int32
    )
    while d < 4 * tile_count:
        source = d // 4
        hp = d % 4
        expert = tl.load(QuantScale + META + source * 2).to(tl.int32)
        tile = tl.load(QuantScale + META + source * 2 + 1).to(tl.int32)
        complete = tl.atomic_add(
            QuantScale + COUNTERS + source, 0.0, sem="acquire", scope="gpu"
        )
        while complete < tasks_per_tile.to(tl.float32):
            complete = tl.atomic_add(
                QuantScale + COUNTERS + source, 0.0, sem="acquire", scope="gpu"
            )
        count = tl.load(counts + expert)
        for qi in tl.static_range(4):
            qv = tl.load(
                Quant
                + (source * BM + row[:, None]) * I
                + qi * 128
                + tl.arange(0, 128)[None, :]
            )
            tl.store(tle.gpu.local_ptr(qbs.slot(qi)), qv)
        tle.gpu.barrier_arrive(qb, phaseIdx=qp)
        tle.gpu.barrier_wait(qb, phaseIdx=qp)
        qp = 1 - qp
        index = tile * BM + row
        valid = index < count
        route = tl.load(queue + expert * CAP + index, valid, 0)
        if R == 1:
            dest = peers(Part, MY, tl.bfloat16)
            local = route
        else:
            dest = peers(Part, route // (T * TOP), tl.bfloat16)
            local = route % (T * TOP)
        w0 = tle.gpu.alloc([BN2, 128], tl.float8e4nv)
        w1 = tle.gpu.alloc([BN2, 128], tl.float8e4nv)
        wb = (w0, w1)
        bw = (bw0, bw1)
        for sub in tl.static_range(2):
            hb = hp * 2 + sub
            down_row = expert * H + hb * BN2
            tle.gpu.copy(B2, w0, [BN2, 128], [down_row, 0], barrier=bw0)
            acc = tl.zeros((BM, BN2), tl.float32)
            for pair in range(I // PAIR_BLOCK):
                for side in tl.static_range(2):
                    kq = pair * 2 + side
                    if kq + 1 < I // SCALE_GROUP:
                        k_next = ((kq + 1) % (I // SCALE_GROUP)) * 128
                        tle.gpu.copy(
                            B2,
                            wb[1 - side],
                            [BN2, 128],
                            [down_row, k_next],
                            barrier=bw[1 - side],
                        )
                    base = (expert * (H // SCALE_GROUP) + hb * 2) * (
                        I // SCALE_GROUP
                    ) + kq
                    s0 = tl.load(S2 + base)
                    s1 = tl.load(S2 + base + I // SCALE_GROUP)
                    qs = tl.load(
                        QuantScale + (source * (I // SCALE_GROUP) + kq) * BM + row
                    )
                    tle.gpu.barrier_wait(bw[side], phaseIdx=pair % 2)
                    product = tle.gpu.wgmma(qbs.slot(kq), wb[side], trans_b=True)
                    product = tle.gpu.wgmma_wait(0, product)
                    scales = tl.join(
                        tl.broadcast_to(s0, (128,)), tl.broadcast_to(s1, (128,))
                    )
                    sb = tl.reshape(tl.trans(scales), (BN2,))
                    acc += product * (qs[:, None] * sb[None, :])
            tl.store(
                dest[:, None] + local[:, None] * H + hb * BN2 + col[None, :],
                acc.to(tl.bfloat16),
                valid[:, None],
            )
        d = tl.atomic_add(QuantScale + TASKS + 1, 1.0, sem="relaxed", scope="gpu").to(
            tl.int32
        )

    fence_system(Grid)
    tle.gpu.barrier_arrive(qb, phaseIdx=qp)
    tle.gpu.barrier_wait(qb, phaseIdx=qp)
    tle.gpu.barrier_arrive(join, phaseIdx=0)
    tle.gpu.barrier_wait(join, phaseIdx=0)


@triton.jit
def dispatch_combine(
    Ids,
    QueueTable,
    CountTable,
    Grid,
    Part,
    Y,
    Signals,
    Epoch,
    phase_bar,
    join,
    Ready,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    TOP: tl.constexpr,
    CTAS: tl.constexpr,
    CAP: tl.constexpr,
    MY: tl.constexpr,
    R: tl.constexpr,
):
    # ---- Initialize queues, dispatch routes, combine ordered partials
    cta = tl.program_id(0)
    queue = peers(QueueTable, MY, tl.int32)
    counts = peers(CountTable, MY, tl.int32)
    lanes = tle.gpu.set_layout(
        tl.arange(0, 256), tle.gpu.BlockEncoding([1], [32], [4], [0])
    )
    # ---- 1. clear counters + ready slots
    if cta == 0:
        tl.store(Ready, CTAS * 2)
    for base in range(cta * 256, E, CTAS * 256):
        expert = base + lanes
        tl.store(counts + expert, 0, expert < E)
    tle.gpu.barrier_arrive(phase_bar, phaseIdx=0)
    tle.gpu.barrier_wait(phase_bar, phaseIdx=0)
    # One dispatcher partition per CTA arrives at the system-scope barrier.
    scatter_rendezvous(Grid, Signals, MY, R, CTAS, 0, 0, Epoch)
    # ---- 2. routing
    for base in range(cta * 256, T * TOP, CTAS * 256):
        route = base + lanes
        expert = tl.load(Ids + route, route < T * TOP, -1)
        valid = (expert >= 0) & (expert < E * R) & (route < T * TOP)
        owner = tl.where(valid, expert // E, 0)
        local = tl.where(valid, expert % E, 0)
        if R == 1:
            index = tl.atomic_add(
                counts + local, 1, mask=valid, sem="relaxed", scope="gpu"
            )
            tl.store(queue + local * CAP + index, route, valid)
        else:
            index = tl.atomic_add(
                peers(CountTable, owner, tl.int32) + local,
                1,
                mask=valid,
                sem="relaxed",
                scope="sys",
            )
            tl.store(
                peers(QueueTable, owner, tl.int32) + local * CAP + index,
                MY * T * TOP + route,
                valid,
            )
    tle.gpu.barrier_arrive(phase_bar, phaseIdx=1)
    tle.gpu.barrier_wait(phase_bar, phaseIdx=1)
    scatter_rendezvous(Grid, Signals, MY, R, CTAS, 2, 1, Epoch)
    tle.gpu.barrier_arrive(join, phaseIdx=0)
    tle.gpu.barrier_wait(join, phaseIdx=0)
    scatter_rendezvous(Grid, Signals, MY, R, CTAS, 4, 2, Epoch)
    partial = peers(Part, MY, tl.bfloat16)
    hidden = tle.gpu.set_layout(
        tl.arange(0, H), tle.gpu.BlockEncoding([16], [32], [4], [0])
    )
    for token in range(cta, T, CTAS):
        accum = tl.zeros((H,), tl.float32)
        for top in tl.static_range(TOP):
            expert = tl.load(Ids + token * TOP + top)
            value = tl.load(
                partial + (token * TOP + top) * H + hidden,
                (expert >= 0) & (expert < E * R),
                0,
            )
            accum += value.to(tl.float32)
        tl.store(Y + token * H + hidden, accum.to(tl.bfloat16))


@triton.jit
def tile_worker(
    X,
    S,
    Ids,
    RW,
    S1,
    S2,
    QueueTable,
    CountTable,
    Grid,
    Part,
    Y,
    Signals,
    A,
    AS,
    B1,
    B2,
    Q,
    QS,
    Input,
    InputScale,
    Weight,
    Quant,
    QuantScale,
    Ready,
    QuantReady,
    Epoch,
    ba0,
    ba1,
    bw0,
    bw1,
    phase_bar,
    join,
    ROLE: tl.constexpr,
    T,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOP: tl.constexpr,
    CTAS: tl.constexpr,
    CAP: tl.constexpr,
    MY: tl.constexpr,
    R: tl.constexpr,
    POOL: tl.constexpr,
    EPW: tl.constexpr,
):
    # ---- Stage token tiles, execute paired projections, publish partials
    SCALE_GROUP: tl.constexpr = 128
    QUANT_BLOCK: tl.constexpr = 64
    PAIR_BLOCK: tl.constexpr = 256
    cta = tl.program_id(0)
    queue = peers(QueueTable, MY, tl.int32)
    counts = peers(CountTable, MY, tl.int32)
    BM: tl.constexpr = 32
    BN1: tl.constexpr = 128
    BN2: tl.constexpr = 256
    row = tl.arange(0, BM)
    col = tl.arange(0, BN2)
    lanes = tle.gpu.set_layout(  # noqa: F841
        tl.arange(0, 256), tle.gpu.BlockEncoding([1], [32], [4], [0])
    )
    kfull = tle.gpu.set_layout(  # noqa: F841
        tl.arange(0, H), tle.gpu.BlockEncoding([16], [32], [4], [0])
    )
    kbhd = tle.gpu.set_layout(  # noqa: F841
        tl.arange(0, H // SCALE_GROUP), tle.gpu.BlockEncoding([1], [32], [4], [0])
    )
    qbs = tle.gpu.alloc([I // SCALE_GROUP, BM, 128], tl.float8e4nv)
    input_scales = tle.gpu.alloc(
        [H // SCALE_GROUP, BM], tl.float32, nv_mma_shared_layout=False
    )

    ready = Grid + (2 if R == 1 else 3)
    arrivals = tl.atomic_add(ready, 0, sem="acquire", scope="sys")
    while arrivals < CTAS * Epoch:
        arrivals = tl.atomic_add(ready, 0, sem="acquire", scope="sys")
    if R == 2 or R == 4:
        tc_ks = tle.gpu.set_layout(
            tl.arange(0, H), tle.gpu.BlockEncoding([16], [32], [4], [0])
        )
        tc_wr = tle.gpu.set_layout(
            tl.arange(0, TOP), tle.gpu.BlockEncoding([1], [32], [4], [0])
        )
        tc_ss = tle.gpu.set_layout(
            tl.arange(0, H // SCALE_GROUP), tle.gpu.BlockEncoding([1], [32], [4], [0])
        )
        for tc_pe in tl.static_range(R):
            if tc_pe != MY:
                tc_xp = peers(X, tc_pe, tl.float8e4nv)
                tc_sp = peers(S, tc_pe, tl.float32)
                tc_wp = peers(RW, tc_pe, tl.float32)
                for tc_tok in range(tl.program_id(0) * 2 + ROLE, T, CTAS * 2):
                    tc_xv = tl.load(tc_xp + tc_tok * H + tc_ks)
                    tc_sv = tl.load(tc_sp + tc_tok * (H // SCALE_GROUP) + tc_ss)
                    tc_wv = tl.load(tc_wp + tc_tok * TOP + tc_wr)
                    tl.store(Input + CAP * H + (tc_pe * T + tc_tok) * H + tc_ks, tc_xv)
                    tl.store(
                        InputScale
                        + CAP * (H // SCALE_GROUP)
                        + (tc_pe * T + tc_tok) * (H // SCALE_GROUP)
                        + tc_ss,
                        tc_sv,
                    )
                    tl.store(
                        QuantScale
                        + POOL * 128 * (I // QUANT_BLOCK)
                        + POOL * 32
                        + (tc_pe * T + tc_tok) * TOP
                        + tc_wr,
                        tc_wv,
                    )
        fence_system(Grid)
        tle.gpu.barrier_arrive(phase_bar, phaseIdx=(0))
        tle.gpu.barrier_wait(phase_bar, phaseIdx=(0))
        tle.gpu.barrier_arrive(phase_bar, phaseIdx=1 - (0))
        tle.gpu.barrier_wait(phase_bar, phaseIdx=1 - (0))
        tl.atomic_add(Grid + 7, 1, sem="release", scope="gpu")
        tc_seen = tl.atomic_add(Grid + 7, 0, sem="acquire", scope="gpu")
        # Two math partitions per CTA arrive at this GPU-scope barrier.
        while tc_seen < CTAS * 2 * Epoch:
            tc_seen = tl.atomic_add(Grid + 7, 0, sem="acquire", scope="gpu")
    worker_id = cta * 2 + ROLE
    # Two math partitions per CTA arrive at this GPU-scope gather rendezvous;
    # the system-scope cross-rank rendezvous below counts CTAs instead.
    MATH_PARTITIONS: tl.constexpr = 2
    workers: tl.constexpr = CTAS * MATH_PARTITIONS
    # ---- 3. per tile: gather -> phase 0 -> phase 1
    experts = tl.arange(0, triton.next_power_of_2(E))[None, :]
    counts_by_expert = tl.load(counts + experts, experts < E, 0)
    expert_tiles = tl.cdiv(counts_by_expert, BM)
    tile_ends = tl.cumsum(expert_tiles, axis=1)
    tile_count = tl.reshape(tl.sum(expert_tiles, axis=1), ())
    source = worker_id
    while source < tile_count:
        earlier = tile_ends <= source
        expert = tl.reshape(tl.sum(earlier.to(tl.int32), axis=1), ())
        tile = source - tl.reshape(tl.max(tl.where(earlier, tile_ends, 0), axis=1), ())
        count = tl.load(counts + expert)
        if R == 1:
            gather: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 16], [1, 32], [4, 1], [1, 0]
            )
            rr = tle.gpu.set_layout(tl.arange(0, 4), tle.gpu.SlicedEncoding(1, gather))
            kk = tle.gpu.set_layout(tl.arange(0, H), tle.gpu.SlicedEncoding(0, gather))
            sk = tle.gpu.set_layout(
                tl.arange(0, H // SCALE_GROUP), tle.gpu.SlicedEncoding(0, gather)
            )
            xp = peers(X, MY, tl.float8e4nv)
            sfp = peers(S, MY, tl.float32)
            wr = peers(RW, MY, tl.float32)
            for offset in range(0, tl.minimum(BM, count - tile * BM), 4):
                rr0 = offset + rr
                idx = tile * BM + rr0
                rvalid = idx < count
                route_id = tl.load(queue + expert * CAP + idx, rvalid, 0)
                tok = route_id // TOP
                x = tl.load(xp + tok[:, None] * H + kk[None, :], rvalid[:, None], 0.0)
                tl.store(
                    Input + (source * BM + rr0[:, None]) * H + kk[None, :],
                    x,
                    rvalid[:, None],
                )
                sf = tl.load(
                    sfp + tok[:, None] * (H // SCALE_GROUP) + sk[None, :],
                    rvalid[:, None],
                    0.0,
                )
                tl.store(
                    tle.gpu.local_ptr(
                        input_scales,
                        (
                            tl.broadcast_to(sk[None, :], (4, H // SCALE_GROUP)),
                            tl.broadcast_to(rr0[:, None], (4, H // SCALE_GROUP)),
                        ),
                    ),
                    sf,
                    rvalid[:, None],
                )
                rwv = tl.load(wr + route_id, rvalid, 0.0)
                tl.store(Weight + source * BM + rr0, rwv, rvalid)
        else:
            # multi-row (4 rows per iteration) remote gather, same as fused_n08_multigather
            gather: tl.constexpr = tle.gpu.BlockEncoding(
                [1, 16], [1, 32], [4, 1], [1, 0]
            )
            rr = tle.gpu.set_layout(tl.arange(0, 4), tle.gpu.SlicedEncoding(1, gather))
            kk = tle.gpu.set_layout(tl.arange(0, H), tle.gpu.SlicedEncoding(0, gather))
            sk = tle.gpu.set_layout(
                tl.arange(0, H // SCALE_GROUP), tle.gpu.SlicedEncoding(0, gather)
            )
            for offset in range(0, tl.minimum(BM, count - tile * BM), 4):
                rr0 = offset + rr
                idx = tile * BM + rr0
                rvalid = idx < count
                route_id = tl.load(queue + expert * CAP + idx, rvalid, 0)
                src = route_id // (T * TOP)
                local_route = route_id % (T * TOP)
                tok = local_route // TOP
                if R == 2 or R == 4:
                    xp = tl.where(
                        src == MY,
                        peers(X, MY, tl.float8e4nv),
                        Input + CAP * H + src * T * H,
                    )[:, None]
                    sfp = tl.where(
                        src == MY,
                        peers(S, MY, tl.float32),
                        InputScale
                        + CAP * (H // SCALE_GROUP)
                        + src * T * (H // SCALE_GROUP),
                    )[:, None]
                else:
                    xp = peers(X, src, tl.float8e4nv)[:, None]
                    sfp = peers(S, src, tl.float32)[:, None]
                x = tl.load(xp + tok[:, None] * H + kk[None, :], rvalid[:, None], 0.0)
                tl.store(
                    Input + (source * BM + rr0[:, None]) * H + kk[None, :],
                    x,
                    rvalid[:, None],
                )
                sf = tl.load(
                    sfp + tok[:, None] * (H // SCALE_GROUP) + sk[None, :],
                    rvalid[:, None],
                    0.0,
                )
                tl.store(
                    tle.gpu.local_ptr(
                        input_scales,
                        (
                            tl.broadcast_to(sk[None, :], (4, H // SCALE_GROUP)),
                            tl.broadcast_to(rr0[:, None], (4, H // SCALE_GROUP)),
                        ),
                    ),
                    sf,
                    rvalid[:, None],
                )
                if R == 2 or R == 4:
                    wptr = tl.where(
                        src == MY,
                        peers(RW, MY, tl.float32),
                        QuantScale
                        + POOL * 128 * (I // QUANT_BLOCK)
                        + POOL * 32
                        + src * T * TOP,
                    )
                    rwv = tl.load(wptr + local_route, rvalid, 0.0)
                else:
                    rwv = tl.load(peers(RW, src, tl.float32) + local_route, rvalid, 0.0)
                tl.store(Weight + source * BM + rr0, rwv, rvalid)
        fence_system(Grid)
        tle.gpu.barrier_arrive(phase_bar, phaseIdx=0)
        tle.gpu.barrier_wait(phase_bar, phaseIdx=0)
        a0 = tle.gpu.alloc([BM, 128], tl.float8e4nv)
        w0 = tle.gpu.alloc([2 * BN1, 128], tl.float8e4nv)
        a1 = tle.gpu.alloc([BM, 128], tl.float8e4nv)
        w1 = tle.gpu.alloc([2 * BN1, 128], tl.float8e4nv)
        ab, wb = (a0, a1), (w0, w1)
        ba, bw = (ba0, ba1), (bw0, bw1)
        # Gate and up share a TMA box. Scale each projection separately.
        route_weight = tl.load(Weight + source * BM + row)
        first_weight = expert * (2 * I)
        tle.gpu.copy(A, a0, [BM, 128], [source * BM, 0], barrier=ba0)
        tle.gpu.copy(B1, w0, [2 * BN1, 128], [first_weight, 0], barrier=bw0)
        for nb in range(I // BN1):
            acc = tl.zeros((BM, 2 * BN1), tl.float32)
            for pair in range(H // PAIR_BLOCK):
                for side in tl.static_range(2):
                    kb = pair * 2 + side
                    if kb + 1 < H // SCALE_GROUP or nb + 1 < I // BN1:
                        k_next = ((kb + 1) % (H // SCALE_GROUP)) * 128
                        n_next = nb + (kb + 1 == H // SCALE_GROUP).to(tl.int32)
                        tle.gpu.copy(
                            A,
                            ab[1 - side],
                            [BM, 128],
                            [source * BM, k_next],
                            barrier=ba[1 - side],
                        )
                        tle.gpu.copy(
                            B1,
                            wb[1 - side],
                            [2 * BN1, 128],
                            [first_weight + n_next * (2 * BN1), k_next],
                            barrier=bw[1 - side],
                        )
                    else:
                        tle.gpu.copy(B2, w0, [BN2, 128], [expert * H, 0], barrier=bw0)
                    sf = tl.load(
                        tle.gpu.local_ptr(
                            input_scales, (tl.zeros((BM,), tl.int32) + kb, row)
                        )
                    )
                    base = (expert * (2 * I // 128) + nb) * (H // SCALE_GROUP) + kb
                    gs = tl.load(S1 + base)
                    us = tl.load(S1 + base + (I // SCALE_GROUP) * (H // SCALE_GROUP))
                    tle.gpu.barrier_wait(ba[side], phaseIdx=pair % 2)
                    tle.gpu.barrier_wait(bw[side], phaseIdx=pair % 2)
                    product = tle.gpu.wgmma(ab[side], wb[side], trans_b=True)
                    product = tle.gpu.wgmma_wait(0, product)
                    scale_b = tl.reshape(
                        tl.trans(
                            tl.join(
                                tl.broadcast_to(gs, (BN1,)), tl.broadcast_to(us, (BN1,))
                            )
                        ),
                        (2 * BN1,),
                    )
                    acc += product * (sf[:, None] * scale_b[None, :])
            pair_acc = tl.permute(tl.reshape(acc, (BM, 2, BN1)), (0, 2, 1))
            gate, up = tl.split(pair_acc)
            activated = gate / (1.0 + tl.exp(-gate)) * up
            q128, common = quantize_64(activated, route_weight)
            tl.store(tle.gpu.local_ptr(qbs.slot(nb)), q128)
            tl.store(QuantScale + (source * (I // SCALE_GROUP) + nb) * BM + row, common)
        tle.gpu.barrier_arrive(phase_bar, phaseIdx=1)
        tle.gpu.barrier_wait(phase_bar, phaseIdx=1)
        # Down projection consumes the same per-64 FP8 blocks as the baseline.
        index = tile * BM + row
        valid = index < count
        route = tl.load(queue + expert * CAP + index, valid, 0)
        if R == 1:
            dest = peers(Part, MY, tl.bfloat16)
            local = route
        else:
            dest = peers(Part, route // (T * TOP), tl.bfloat16)
            local = route % (T * TOP)
        for nbH in range(H // BN2):
            acc = tl.zeros((BM, BN2), tl.float32)
            down_row = expert * H + nbH * BN2
            # Both projections use the same physical B tile shape and
            # swizzle. Gate/up has drained before reusing its buffers.
            for pair in range(I // PAIR_BLOCK):
                for side in tl.static_range(2):
                    kq = pair * 2 + side
                    if kq + 1 < I // SCALE_GROUP or nbH + 1 < H // BN2:
                        k_next = ((kq + 1) % (I // SCALE_GROUP)) * 128
                        row_next = (
                            down_row + (kq + 1 == I // SCALE_GROUP).to(tl.int32) * BN2
                        )
                        tle.gpu.copy(
                            B2,
                            wb[1 - side],
                            [BN2, 128],
                            [row_next, k_next],
                            barrier=bw[1 - side],
                        )
                    base = (expert * (H // SCALE_GROUP) + nbH * 2) * (
                        I // SCALE_GROUP
                    ) + kq
                    s0 = tl.load(S2 + base)
                    s1 = tl.load(S2 + base + I // SCALE_GROUP)
                    qs = tl.load(
                        QuantScale + (source * (I // SCALE_GROUP) + kq) * BM + row
                    )
                    tle.gpu.barrier_wait(bw[side], phaseIdx=pair % 2)
                    product = tle.gpu.wgmma(qbs.slot(kq), wb[side], trans_b=True)
                    product = tle.gpu.wgmma_wait(0, product)
                    scales = tl.join(
                        tl.broadcast_to(s0, (128,)), tl.broadcast_to(s1, (128,))
                    )
                    sb = tl.reshape(tl.trans(scales), (BN2,))
                    acc += product * (qs[:, None] * sb[None, :])
            tl.store(
                dest[:, None] + local[:, None] * H + nbH * BN2 + col[None, :],
                acc.to(tl.bfloat16),
                valid[:, None],
            )

        if tile_count > workers:
            source = tl.atomic_add(Ready, 1, sem="relaxed", scope="gpu")
        else:
            source = tile_count
    # ---- 4. combine
    fence_system(Grid)
    tle.gpu.barrier_arrive(join, phaseIdx=0)
    tle.gpu.barrier_wait(join, phaseIdx=0)


@triton.jit(do_not_specialize=["T"])
def megamoe_kernel(
    X,
    S,
    Ids,
    RW,
    S1,
    S2,
    QueueTable,
    CountTable,
    Grid,
    Part,
    Y,
    Signals,
    A,
    AS,
    B1,
    B2,
    Q,
    QS,
    Input,
    InputScale,
    Weight,
    Quant,
    QuantScale,
    Ready,
    QuantReady,
    Epoch,
    T,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOP: tl.constexpr,
    CTAS: tl.constexpr,
    CAP: tl.constexpr,
    MY: tl.constexpr,
    R: tl.constexpr,
    POOL: tl.constexpr,
    EPW: tl.constexpr,
    SMALL: tl.constexpr,
):
    # ---- Partition dispatcher and compute workers by token regime
    A_BARRIER_BYTES: tl.constexpr = 32 * 128
    B_BARRIER_BYTES: tl.constexpr = 256 * 128
    PARTITION_ARRIVALS: tl.constexpr = 4
    JOIN_ARRIVALS: tl.constexpr = 12
    if not SMALL:
        a00 = tle.gpu.alloc_barrier(
            expect_bytes=A_BARRIER_BYTES
        )  # 32 rows x 128 cols x 1B (FP8).
        a01 = tle.gpu.alloc_barrier(
            expect_bytes=A_BARRIER_BYTES
        )  # 32 rows x 128 cols x 1B (FP8).
        b00 = tle.gpu.alloc_barrier(
            expect_bytes=B_BARRIER_BYTES
        )  # 256 rows x 128 cols x 1B (FP8).
        b01 = tle.gpu.alloc_barrier(
            expect_bytes=B_BARRIER_BYTES
        )  # 256 rows x 128 cols x 1B (FP8).
        a10 = tle.gpu.alloc_barrier(
            expect_bytes=A_BARRIER_BYTES
        )  # 32 rows x 128 cols x 1B (FP8).
        a11 = tle.gpu.alloc_barrier(
            expect_bytes=A_BARRIER_BYTES
        )  # 32 rows x 128 cols x 1B (FP8).
        b10 = tle.gpu.alloc_barrier(
            expect_bytes=B_BARRIER_BYTES
        )  # 256 rows x 128 cols x 1B (FP8).
        b11 = tle.gpu.alloc_barrier(
            expect_bytes=B_BARRIER_BYTES
        )  # 256 rows x 128 cols x 1B (FP8).
        sync0 = tle.gpu.alloc_barrier(arrive_count=PARTITION_ARRIVALS)
        sync1 = tle.gpu.alloc_barrier(arrive_count=PARTITION_ARRIVALS)
        dispatch_sync = tle.gpu.alloc_barrier(arrive_count=PARTITION_ARRIVALS)
        join = tle.gpu.alloc_barrier(arrive_count=JOIN_ARRIVALS)
        tle.gpu.warp_specialize(
            [
                (
                    tile_worker,
                    (
                        X,
                        S,
                        Ids,
                        RW,
                        S1,
                        S2,
                        QueueTable,
                        CountTable,
                        Grid,
                        Part,
                        Y,
                        Signals,
                        A,
                        AS,
                        B1,
                        B2,
                        Q,
                        QS,
                        Input,
                        InputScale,
                        Weight,
                        Quant,
                        QuantScale,
                        Ready,
                        QuantReady,
                        Epoch,
                        a00,
                        a01,
                        b00,
                        b01,
                        sync0,
                        join,
                        0,
                        T,
                        H,
                        I,
                        E,
                        TOP,
                        CTAS,
                        CAP,
                        MY,
                        R,
                        POOL,
                        EPW,
                    ),
                ),
                (
                    tile_worker,
                    (
                        X,
                        S,
                        Ids,
                        RW,
                        S1,
                        S2,
                        QueueTable,
                        CountTable,
                        Grid,
                        Part,
                        Y,
                        Signals,
                        A,
                        AS,
                        B1,
                        B2,
                        Q,
                        QS,
                        Input,
                        InputScale,
                        Weight,
                        Quant,
                        QuantScale,
                        Ready,
                        QuantReady,
                        Epoch,
                        a10,
                        a11,
                        b10,
                        b11,
                        sync1,
                        join,
                        1,
                        T,
                        H,
                        I,
                        E,
                        TOP,
                        CTAS,
                        CAP,
                        MY,
                        R,
                        POOL,
                        EPW,
                    ),
                ),
                (
                    dispatch_combine,
                    (
                        Ids,
                        QueueTable,
                        CountTable,
                        Grid,
                        Part,
                        Y,
                        Signals,
                        Epoch,
                        dispatch_sync,
                        join,
                        Ready,
                        T,
                        H,
                        E,
                        TOP,
                        CTAS,
                        CAP,
                        MY,
                        R,
                    ),
                ),
            ],
            worker_num_warps=[4, 4],
            worker_num_regs=[128, 32],
        )
    else:
        a00 = tle.gpu.alloc_barrier(
            expect_bytes=A_BARRIER_BYTES
        )  # 32 rows x 128 cols x 1B (FP8).
        a01 = tle.gpu.alloc_barrier(
            expect_bytes=A_BARRIER_BYTES
        )  # 32 rows x 128 cols x 1B (FP8).
        b00 = tle.gpu.alloc_barrier(
            expect_bytes=B_BARRIER_BYTES
        )  # 256 rows x 128 cols x 1B (FP8).
        b01 = tle.gpu.alloc_barrier(
            expect_bytes=B_BARRIER_BYTES
        )  # 256 rows x 128 cols x 1B (FP8).
        a10 = tle.gpu.alloc_barrier(
            expect_bytes=A_BARRIER_BYTES
        )  # 32 rows x 128 cols x 1B (FP8).
        a11 = tle.gpu.alloc_barrier(
            expect_bytes=A_BARRIER_BYTES
        )  # 32 rows x 128 cols x 1B (FP8).
        b10 = tle.gpu.alloc_barrier(
            expect_bytes=B_BARRIER_BYTES
        )  # 256 rows x 128 cols x 1B (FP8).
        b11 = tle.gpu.alloc_barrier(
            expect_bytes=B_BARRIER_BYTES
        )  # 256 rows x 128 cols x 1B (FP8).
        sync0 = tle.gpu.alloc_barrier(arrive_count=PARTITION_ARRIVALS)
        sync1 = tle.gpu.alloc_barrier(arrive_count=PARTITION_ARRIVALS)
        dispatch_sync = tle.gpu.alloc_barrier(arrive_count=PARTITION_ARRIVALS)
        join = tle.gpu.alloc_barrier(arrive_count=JOIN_ARRIVALS)
        tle.gpu.warp_specialize(
            [
                (
                    worker,
                    (
                        X,
                        S,
                        Ids,
                        RW,
                        S1,
                        S2,
                        QueueTable,
                        CountTable,
                        Grid,
                        Part,
                        Y,
                        Signals,
                        A,
                        AS,
                        B1,
                        B2,
                        Q,
                        QS,
                        Input,
                        InputScale,
                        Weight,
                        Quant,
                        QuantScale,
                        Ready,
                        QuantReady,
                        Epoch,
                        a00,
                        a01,
                        b00,
                        b01,
                        sync0,
                        sync0,
                        join,
                        0,
                        T,
                        H,
                        I,
                        E,
                        TOP,
                        CTAS,
                        CAP,
                        MY,
                        R,
                        POOL,
                        EPW,
                    ),
                ),
                (
                    worker,
                    (
                        X,
                        S,
                        Ids,
                        RW,
                        S1,
                        S2,
                        QueueTable,
                        CountTable,
                        Grid,
                        Part,
                        Y,
                        Signals,
                        A,
                        AS,
                        B1,
                        B2,
                        Q,
                        QS,
                        Input,
                        InputScale,
                        Weight,
                        Quant,
                        QuantScale,
                        Ready,
                        QuantReady,
                        Epoch,
                        a10,
                        a11,
                        b10,
                        b11,
                        sync1,
                        sync1,
                        join,
                        1,
                        T,
                        H,
                        I,
                        E,
                        TOP,
                        CTAS,
                        CAP,
                        MY,
                        R,
                        POOL,
                        EPW,
                    ),
                ),
                (
                    dispatch_combine,
                    (
                        Ids,
                        QueueTable,
                        CountTable,
                        Grid,
                        Part,
                        Y,
                        Signals,
                        Epoch,
                        dispatch_sync,
                        join,
                        Ready,
                        T,
                        H,
                        E,
                        TOP,
                        CTAS,
                        CAP,
                        MY,
                        R,
                    ),
                ),
            ],
            worker_num_warps=[4, 4],
            worker_num_regs=[128, 32],
        )


def prepare_megamoe_fp8(
    input_fp8: torch.Tensor,
    input_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    context: MegaMoEContext,
    output: torch.Tensor,
):
    """Warm up three epoch signatures without advancing the distributed epoch.

    Cache only compilation readiness and descriptors in this context. Real
    launches always bind current arguments through the standard JIT launcher.
    No device work is enqueued by warmup; loading and IR validation may fail.
    """
    _validate_call(
        input_fp8,
        input_scales,
        topk_ids,
        topk_weights,
        w1,
        w1_scales,
        w2,
        w2_scales,
        context,
        output,
    )
    t = input_fp8.shape[0]
    key = (
        t,
        context.CTAS,
        context.E,
        context.TOP,
        context.CAP,
        context.R,
        context.POOL,
        context.EPW,
    )
    identity = tuple(
        (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        for tensor in (
            w1,
            w2,
            context.input_workspace,
            context.input_scale_workspace,
            context.quant_workspace,
            context.quant_scale_workspace,
        )
    )
    cached = context._prepared.get(key)
    if cached is not None:
        if cached[0] != identity:
            raise RuntimeError(
                "Prepared context descriptors require persistent buffers"
            )
        return cached[1]
    descriptors = (
        TensorDescriptor.from_tensor(
            context.input_workspace.reshape(-1, context.H), list(context.a_tile)
        ),
        TensorDescriptor.from_tensor(
            context.input_scale_workspace.reshape(-1, 64), [1, 64]
        ),
        TensorDescriptor.from_tensor(w1, list(context.b1_tile)),
        TensorDescriptor.from_tensor(w2, list(context.b2_tile)),
        TensorDescriptor.from_tensor(
            context.quant_workspace.reshape(-1, context.I), list(context.q_tile)
        ),
        TensorDescriptor.from_tensor(
            context.quant_scale_workspace.reshape(-1, 64), [1, 64]
        ),
    )
    (
        context.a_descriptor,
        context.as_descriptor,
        context.b1_descriptor,
        context.b2_descriptor,
        context.q_descriptor,
        context.qs_descriptor,
    ) = descriptors
    compiled_signatures = []
    for epoch in (1, 2, 16):
        args = _call_args(
            topk_ids, w1_scales, w2_scales, context, output, descriptors, epoch
        )
        with _musa_llc_opt_disabled():
            compiled = megamoe_kernel.warmup(
                *args,
                grid=(context.CTAS,),
                extern_libs=EXTERN_LIBS,
                **LAUNCH_OPTIONS,
                **_launch_config(t, context),
            )
        check_barrier_map(compiled.asm["llir"])
        _ = compiled.run
        compiled_signatures.append(compiled)
    context._prepared[key] = (identity, descriptors, tuple(compiled_signatures))
    return descriptors


def launch_megamoe_fp8(
    input_fp8: torch.Tensor,
    input_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    context: MegaMoEContext,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prepare on demand and enqueue one standard JIT launch per invocation."""
    if output is None:
        output = torch.empty(
            (input_fp8.shape[0], context.H),
            dtype=torch.bfloat16,
            device=input_fp8.device,
        )
    descriptors = prepare_megamoe_fp8(
        input_fp8,
        input_scales,
        topk_ids,
        topk_weights,
        w1,
        w1_scales,
        w2,
        w2_scales,
        context,
        output,
    )
    context.barrier()
    epoch = context.next_epoch()
    args = _call_args(
        topk_ids, w1_scales, w2_scales, context, output, descriptors, epoch
    )
    with _musa_llc_opt_disabled():
        megamoe_kernel[(context.CTAS,)](
            *args,
            extern_libs=EXTERN_LIBS,
            **LAUNCH_OPTIONS,
            **_launch_config(input_fp8.shape[0], context),
        )
    return output


def _validate_call(
    input_fp8,
    input_scales,
    topk_ids,
    topk_weights,
    w1,
    w1_scales,
    w2,
    w2_scales,
    context,
    output,
):
    t = input_fp8.shape[0]
    if (
        context.H,
        context.I,
        context.E * context.R,
        context.TOP,
        context.CTAS,
        context.EPW,
    ) != (2048, 512, 256, 8, 60, 16):
        raise ValueError("Only the retained source configuration is supported")
    if context.R not in (1, 2, 4, 8) or not 0 <= context.MY < context.R:
        raise ValueError("Invalid MUSA rank group")
    if t < 1 or context.max_recv < 1 or context.CAP < context.R * context.max_recv:
        raise ValueError("CAP must cover R * max_recv and T must be positive")
    if context.pool_tokens < context.CAP:
        raise ValueError("pool_tokens must cover CAP and all padded expert routes")
    if (
        context.pack,
        context.a_tile,
        context.b1_tile,
        context.b2_tile,
        context.q_tile,
    ) != ("gu256", (32, 128), (256, 128), (256, 128), (64, 64)):
        raise ValueError("Weight packing and descriptor tiles must match the kernel")
    device = input_fp8.device
    if device.type != "musa":
        raise ValueError("MegaMoE requires MUSA tensors")
    specs = [
        (input_fp8, (t, context.H), torch.float8_e4m3fn),
        (input_scales, (t, context.H // 128), torch.float32),
        (topk_ids, (t, context.TOP), torch.int32),
        (topk_weights, (t, context.TOP), torch.float32),
        (w1, (context.E * 2 * context.I, context.H), torch.float8_e4m3fn),
        (w2, (context.E * context.H, context.I), torch.float8_e4m3fn),
    ]
    for tensor, shape, dtype in specs:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Expected contiguous {shape} {dtype} on {device}")
    scale_specs = [
        (w1_scales, context.E * (2 * context.I // 128) * (context.H // 128)),
        (w2_scales, context.E * (context.H // 128) * (context.I // 128)),
    ]
    for tensor, size in scale_specs:
        if (
            tensor.numel() != size
            or tensor.dtype != torch.float32
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError("Invalid expert block scales")
    for name in (
        "input_ptrs",
        "input_scale_ptrs",
        "route_weight_ptrs",
        "queue_ptrs",
        "count_ptrs",
        "partial_ptrs",
        "signal_ptrs",
    ):
        tensor = getattr(context, name)
        if (
            tensor.numel() != context.R
            or tensor.dtype != torch.int64
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must contain R contiguous device int64 pointers")
    p = context.pool_tokens
    specs = [
        ("grid", 10, torch.int32),
        (
            "input_workspace",
            max(2 * p, context.CAP + context.R * t) * context.H,
            torch.float8_e4m3fn,
        ),
        (
            "input_scale_workspace",
            max(2 * p, context.CAP + context.R * t) * (context.H // 128),
            torch.float32,
        ),
        ("route_weight_workspace", p, torch.float32),
        ("quant_workspace", 2 * p * context.I, torch.float8_e4m3fn),
        (
            "quant_scale_workspace",
            max(
                2 * p * (context.I // 64),
                p * (context.I // 64) + 32 * context.POOL + context.R * t * context.TOP,
            ),
            torch.float32,
        ),
        ("ready", context.POOL, torch.int32),
        ("quant_ready", context.POOL, torch.int32),
    ]
    for name, size, dtype in specs:
        tensor = getattr(context, name)
        if (
            tensor.numel() < size
            or tensor.dtype != dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} requires at least {size} contiguous {dtype} elements"
            )
    for name, width in (
        ("input_workspace", context.H),
        ("quant_workspace", context.I),
        ("input_scale_workspace", 64),
        ("quant_scale_workspace", 64),
    ):
        if getattr(context, name).numel() % width:
            raise ValueError(
                f"{name} size must be divisible by descriptor width {width}"
            )
    if output is not None:
        if (
            tuple(output.shape) != (t, context.H)
            or output.dtype != torch.bfloat16
            or output.device != device
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous MUSA BF16 [T,H]")


def _call_args(topk_ids, w1_scales, w2_scales, context, output, descriptors, epoch):
    return (
        context.input_ptrs,
        context.input_scale_ptrs,
        topk_ids,
        context.route_weight_ptrs,
        w1_scales,
        w2_scales,
        context.queue_ptrs,
        context.count_ptrs,
        context.grid,
        context.partial_ptrs,
        output,
        context.signal_ptrs,
        *descriptors,
        context.input_workspace,
        context.input_scale_workspace,
        context.route_weight_workspace,
        context.quant_workspace,
        context.quant_scale_workspace,
        context.ready,
        context.quant_ready,
        epoch,
    )


def _launch_config(t, context):
    return dict(
        T=t,
        H=context.H,
        I=context.I,
        E=context.E,
        TOP=context.TOP,
        CTAS=context.CTAS,
        CAP=context.CAP,
        MY=context.MY,
        R=context.R,
        POOL=context.POOL,
        EPW=context.EPW,
        SMALL=t < 1024,
    )


# ---- device-side system-scope fence --------------------------------------
# Keep the launch contract while requiring no external library.
EXTERN_LIBS = {}


@triton.jit
def fence_system(ptr):
    """Release prior writes at system scope without changing Grid."""
    tl.atomic_add(ptr, 0, sem="release", scope="sys")
    return tl.full((), 0, tl.int32)

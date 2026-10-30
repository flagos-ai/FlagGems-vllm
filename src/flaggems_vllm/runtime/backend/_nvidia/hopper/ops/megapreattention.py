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

"""TLE peer-to-peer tile-ready prefill, implemented entirely in this file.

Owns peer registration, signaling, projection, postprocessing and runtime lifetime.
The persistent grid uses one CTA per available SM and complete M128 input tiles.
Only the fused D256/R64, BF16 window is supported.

This backend-local candidate exposes its callable API directly from this module.
Top-level exports and aten registration are intentionally omitted until generic
vLLM integration; there is no corresponding aten schema.
"""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.tools.tensor_descriptor import TensorDescriptor

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    import triton.experimental.tle.language as tle
else:
    tle = None

__all__ = ["megapreattention", "clear_megapreattention_cache"]


@dataclass(frozen=True)
class _Config:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    mrope_section: tuple[int, int, int]
    eps: float
    hidden_size: int

    @property
    def q_size(self):
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self):
        return self.num_kv_heads * self.head_dim

    @property
    def packed_size(self):
        return 2 * (self.q_size + self.kv_size)


@dataclass
class _Inputs:
    hidden: torch.Tensor
    weight: torch.Tensor
    positions: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    rope: torch.Tensor
    cache: torch.Tensor
    slots: torch.Tensor
    group: object
    config: _Config


@triton.jit
def _prepare_rope(
    positions,
    rope,
    cos,
    sin,
    queue,
    ready,
    POS_AXIS,
    POS_TOKEN,
    ROPE_STRIDE,
    M: tl.constexpr,
    INITIAL: tl.constexpr,
    BM: tl.constexpr,
    SECTION_H: tl.constexpr = 11,
    SECTION_W: tl.constexpr = 10,
):
    pid = tl.program_id(0)
    if pid == 0:
        # Compute CTAs own the initial IDs; all later work is claimed atomically.
        tl.store(queue, INITIAL)
        # Reset this call's arrivals before the producer rendezvous.
        rows = tl.arange(0, triton.next_power_of_2(triton.cdiv(M, 128)))
        tl.store(ready + rows, 0, rows < triton.cdiv(M, 128))
    row = pid * BM + tl.arange(0, BM)
    f = tl.arange(0, 32)
    if SECTION_H == 11 and SECTION_W == 10:
        axis = tl.where(f % 3 == 1, 1, tl.where((f % 3 == 2) & (f <= 30), 2, 0))
    else:
        axis = tl.where(
            (f % 3 == 1) & (f < 3 * SECTION_H),
            1,
            tl.where((f % 3 == 2) & (f < 3 * SECTION_W), 2, 0),
        )
    pos = tl.load(
        positions + axis[None, :] * POS_AXIS + row[:, None] * POS_TOKEN,
        row[:, None] < M,
        other=0,
    ).to(tl.int64)
    co = tl.load(rope + pos * ROPE_STRIDE + f[None, :], row[:, None] < M, other=1)
    si = tl.load(rope + pos * ROPE_STRIDE + 32 + f[None, :], row[:, None] < M, other=0)
    tl.store(cos + row[:, None] * 32 + f[None, :], co, row[:, None] < M)
    tl.store(sin + row[:, None] * 32 + f[None, :], si, row[:, None] < M)


@triton.jit
def _initialize_metadata(flags, counters, WORDS: tl.constexpr, COUNTERS: tl.constexpr):
    i = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(flags + i, 0, i < WORDS)
    tl.store(counters + i, 0, i < COUNTERS)


@triton.jit
def _initialize_peer_table(output, ADDRESSES: tl.constexpr):
    for i in tl.static_range(len(ADDRESSES)):
        tl.store(output + i, ADDRESSES[i])


@triton.jit
def _rendezvous(
    local, peers, pid, rank, target, FLAGS_WORDS: tl.constexpr, WORLD: tl.constexpr
):
    ranks = tl.arange(0, WORLD)
    remote = tl.load(peers + ranks).to(tl.pointer_type(tl.uint64))
    tl.atomic_xchg(
        remote + FLAGS_WORDS + pid * WORLD + rank, target, sem="release", scope="sys"
    )
    pending = tl.full((), 1, tl.int32)
    while pending != 0:
        # Volatile PTX loads are relaxed system-scope reads. Pair the final
        # observation with an acquire fence instead of contended atomic RMWs.
        seen = tl.load(
            local.to(tl.pointer_type(tl.uint64)) + FLAGS_WORDS + pid * WORLD + ranks,
            volatile=True,
        )
        pending = tl.sum((seen < target).to(tl.int32), 0)
    tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0,0;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    # Propagate the acquire performed by signal-reading lanes to the CTA.
    tl.debug_barrier()


class _MappedCommunicator:
    def __init__(self, group, tokens, hidden_size, programs):
        from torch._C._distributed_c10d import _SymmetricMemory

        self.device = torch.device("cuda", torch.cuda.current_device())
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.capacity = tokens
        self.count = tokens * hidden_size
        self.cos = torch.empty((tokens, 32), device="cuda", dtype=torch.bfloat16)
        self.sin = torch.empty_like(self.cos)
        self.work_queue = torch.empty((), device="cuda", dtype=torch.int32)
        self.max_programs = programs
        # Handshake slots belong to producer CTAs; ready slots belong to row
        # tiles. A persistent consumer processes multiple tiles, so their count
        # is independent of the number of concurrently resident CTAs.
        self.handshake_words = programs * self.world
        # Reserve per-row capacity; grouped publication uses only a prefix.
        self.ready_words = triton.cdiv(tokens, 128)
        self.flag_words = self.handshake_words + self.ready_words
        data_bytes = triton.cdiv(self.count * 2, 8) * 8
        self.data_words = data_bytes // 8
        size = data_bytes + self.flag_words * 8
        self.allocation = _SymmetricMemory.empty_strided_p2p(
            (size,),
            (1,),
            torch.uint8,
            torch.device("cuda", torch.cuda.current_device()),
            group.group_name,
        )
        self.mapping = _SymmetricMemory.rendezvous(self.allocation, group.group_name)
        self.buffer = self.mapping.get_buffer(
            self.rank, (tokens, hidden_size), torch.bfloat16
        )
        self.flags = self.mapping.get_buffer(
            self.rank,
            (self.flag_words,),
            torch.uint64,
            storage_offset=data_bytes // 8,
        )
        self.counters = torch.empty(self.max_programs, device="cuda", dtype=torch.int64)
        _initialize_metadata[(triton.cdiv(self.flag_words, 1024),)](
            self.flags,
            self.counters,
            self.flag_words,
            COUNTERS=self.max_programs,
            num_warps=4,
        )
        self.peers = torch.empty(self.world, device="cuda", dtype=torch.uint64)
        _initialize_peer_table[(1,)](
            self.peers, tuple(self.mapping.buffer_ptrs), num_warps=1
        )
        assert all(p % 16 == 0 for p in self.mapping.buffer_ptrs)
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def close(self):
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            self.buffer = None
            self.flags = None
            self.mapping = None
            self.allocation = None


@triton.jit
def _wgmma_slot_phase(k_iter, num_slots: tl.constexpr):
    return k_iter % num_slots, k_iter // num_slots


@triton.jit
def _store_head(
    acc,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    cos,
    sin,
    cache,
    slots,
    row_start,
    pid_m,
    pid_n,
    LOCAL_T: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EPS: tl.constexpr = 1.0e-6,
):
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token = row_start + row
    valid = row < LOCAL_T
    cols = tl.arange(0, 256)
    raw = acc.to(tl.bfloat16)
    # Preserve the materialized BF16 projection and norm rounding boundaries.
    if pid_n < 2 * Q_HEADS and pid_n % 2 == 1:
        head = pid_n // 2
        tl.store(
            gate + token[:, None] * (Q_HEADS * 256) + head * 256 + cols[None, :],
            raw,
            mask=valid[:, None],
        )
    elif pid_n >= 2 * Q_HEADS + KV_HEADS:
        head = pid_n - 2 * Q_HEADS - KV_HEADS if KV_HEADS > 1 else 0
        tl.store(
            value + token[:, None] * (KV_HEADS * 256) + head * 256 + cols[None, :],
            raw,
            mask=valid[:, None],
        )
        slot = tl.load(slots + token * SLOT_STRIDE, mask=valid, other=-1).to(tl.int64)
        target = (
            cache
            + (slot // CACHE_BLOCK_SIZE)[:, None] * CACHE_BLOCK
            + head * CACHE_HEAD
            + (slot % CACHE_BLOCK_SIZE)[:, None] * CACHE_TOKEN
        )
        tl.store(
            target + (256 + cols[None, :]) * CACHE_DIM,
            raw,
            mask=(valid & (slot >= 0))[:, None],
        )
    else:
        is_key = pid_n >= 2 * Q_HEADS if KV_HEADS > 1 else pid_n == 2 * Q_HEADS
        head = pid_n // 2
        weight = tl.load(tl.where(is_key, knorm, qnorm) + cols).to(tl.float32)
        x = raw.to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x, 1) / 256.0 + EPS)
        normalized = (
            (x * scale[:, None] * (weight[None, :] + 1.0))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        frequency = tl.arange(0, 32)
        co = tl.load(
            cos + token[:, None] * 32 + frequency[None, :],
            mask=valid[:, None],
            other=1,
        ).to(tl.float32)
        si = tl.load(
            sin + token[:, None] * 32 + frequency[None, :],
            mask=valid[:, None],
            other=0,
        ).to(tl.float32)
        # Only the first 64 dimensions rotate; pass the other 192 through.
        # Static splits avoid a full-width gather and unused rotary arithmetic.
        low128, high128 = tl.split(normalized.reshape(BLOCK_M, 2, 128).permute(0, 2, 1))
        low64, high64 = tl.split(low128.reshape(BLOCK_M, 2, 64).permute(0, 2, 1))
        first, second = tl.split(low64.reshape(BLOCK_M, 2, 32).permute(0, 2, 1))
        first_out = first * co - second * si
        second_out = second * co + first * si
        rotated64 = tl.join(first_out, second_out).permute(0, 2, 1).reshape(BLOCK_M, 64)
        result128 = tl.join(rotated64, high64).permute(0, 2, 1).reshape(BLOCK_M, 128)
        result = tl.join(result128, high128).permute(0, 2, 1).reshape(BLOCK_M, 256)
        if is_key:
            kv_head = pid_n - 2 * Q_HEADS if KV_HEADS > 1 else 0
            tl.store(
                key + token[:, None] * (KV_HEADS * 256) + kv_head * 256 + cols[None, :],
                result,
                mask=valid[:, None],
            )
            slot = tl.load(slots + token * SLOT_STRIDE, mask=valid, other=-1).to(
                tl.int64
            )
            target = (
                cache
                + (slot // CACHE_BLOCK_SIZE)[:, None] * CACHE_BLOCK
                + kv_head * CACHE_HEAD
                + (slot % CACHE_BLOCK_SIZE)[:, None] * CACHE_TOKEN
            )
            tl.store(
                target + cols[None, :] * CACHE_DIM,
                result,
                mask=(valid & (slot >= 0))[:, None],
            )
        else:
            tl.store(
                query + token[:, None] * (Q_HEADS * 256) + head * 256 + cols[None, :],
                result,
                mask=valid[:, None],
            )


@triton.jit
def _fused_projection(
    BULK_END: tl.constexpr,
    STRIDE: tl.constexpr,
    work_queue,
    first_tile,
    a_desc,
    source_desc,
    rank,
    b_desc,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    cos,
    sin,
    cache,
    slots,
    row_start,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    M: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HIDDEN: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    ready,
    target,
    WORLD: tl.constexpr = 1,
    READY_GROUP: tl.constexpr = 1,
    EPS: tl.constexpr = 1.0e-6,
):
    ROW_GROUP: tl.constexpr = min(4, (M // BM) & -(M // BM))
    num_n: tl.constexpr = 2 * (Q_HEADS + KV_HEADS)
    k_tiles: tl.constexpr = HIDDEN // BK
    USE_LOCAL_INPUT: tl.constexpr = NUM_SLOTS == 3
    PREFETCH_COUNT: tl.constexpr = 2 if WORLD == 2 and USE_LOCAL_INPUT else 0
    a = tle.gpu.alloc(
        [NUM_SLOTS, BM, BK], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    b = tle.gpu.alloc(
        [NUM_SLOTS, 256, BK], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    af = tle.gpu.alloc_barriers(
        triton.next_power_of_2(NUM_SLOTS), expect_bytes=BM * BK * 2
    )
    bf = tle.gpu.alloc_barriers(
        triton.next_power_of_2(NUM_SLOTS), expect_bytes=256 * BK * 2
    )
    seq = 0
    tile_id = first_tile
    while tile_id < tl.cdiv(M, BM) * num_n:
        # Reuse head pairs across a row group; the divisor also handles short tails.
        group_id = tile_id // (ROW_GROUP * num_n)
        within = tile_id % (ROW_GROUP * num_n)
        pair = within // (2 * ROW_GROUP)
        pm = group_id * ROW_GROUP + (within // 2) % ROW_GROUP
        kind = tile_id % 2
        pn = tl.where(
            pair < Q_HEADS,
            2 * pair + kind,
            2 * Q_HEADS + pair - Q_HEADS + kind * KV_HEADS,
        )
        if PREFETCH_COUNT > 0:
            pm = (pm + rank * (M // WORLD // BM)) % (M // BM)
        else:
            pm = (pm % WORLD) * (M // WORLD // BM) + pm // WORLD
            if USE_LOCAL_INPUT:
                pm = (pm + rank * (M // WORLD // BM)) % (M // BM)
        local_row = (
            USE_LOCAL_INPUT
            and pm >= rank * (M // WORLD // BM)
            and pm < (rank + 1) * (M // WORLD // BM)
        )
        if (PREFETCH_COUNT > 0 and seq == 0) or (PREFETCH_COUNT == 0 and not local_row):
            seen = tl.where(local_row, target, 0).to(tl.uint64)
            # Acquire the complete producer arrival chain before TMA reads the payload.
            while seen < target:
                seen = tl.inline_asm_elementwise(
                    "ld.acquire.sys.global.u64 $0, [$1];",
                    constraints="=l,l",
                    args=[ready + pm // READY_GROUP],
                    dtype=tl.uint64,
                    is_pure=False,
                    pack=1,
                )
            tl.debug_barrier()
            tl.inline_asm_elementwise(
                "fence.proxy.async.global; mov.u32 $0,0;",
                constraints="=r",
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        for initial_slot in tl.static_range(NUM_SLOTS):
            if seq == 0 or initial_slot >= PREFETCH_COUNT:
                if local_row:
                    tle.gpu.copy(
                        source_desc,
                        a.slot(initial_slot),
                        [BM, BK],
                        [(pm - rank * (M // WORLD // BM)) * BM, initial_slot * BK],
                        barrier=af[initial_slot],
                    )
                else:
                    tle.gpu.copy(
                        a_desc,
                        a.slot(initial_slot),
                        [BM, BK],
                        [pm * BM, initial_slot * BK],
                        barrier=af[initial_slot],
                    )
                tle.gpu.copy(
                    b_desc,
                    b.slot(initial_slot),
                    [256, BK],
                    [pn * 256, initial_slot * BK],
                    barrier=bf[initial_slot],
                )
        acc = tl.zeros((BM, 256), tl.float32)
        first_phase = seq * tl.cdiv(k_tiles, NUM_SLOTS)
        tle.gpu.barrier_wait(af[0], phaseIdx=first_phase)
        tle.gpu.barrier_wait(bf[0], phaseIdx=first_phase)
        acc = tle.gpu.wgmma(a.slot(0), b.slot(0), acc, trans_b=True)
        for k in tl.range(1, k_tiles):
            # Non-divisible K loops advance each stage by its own reuse count.
            if k_tiles % NUM_SLOTS == 0:
                slot, phase = _wgmma_slot_phase(seq * k_tiles + k, NUM_SLOTS)
            else:
                slot = k % NUM_SLOTS
                phase = seq * tl.cdiv(k_tiles - slot, NUM_SLOTS) + k // NUM_SLOTS
            tle.gpu.barrier_wait(af[slot], phaseIdx=phase)
            tle.gpu.barrier_wait(bf[slot], phaseIdx=phase)
            acc = tle.gpu.wgmma(a.slot(slot), b.slot(slot), acc, trans_b=True)
            acc = tle.gpu.wgmma_wait(1, acc)
            if k - 1 + NUM_SLOTS < k_tiles:
                previous = (k - 1) % NUM_SLOTS
                next_k = (k - 1 + NUM_SLOTS) * BK
                if local_row:
                    tle.gpu.copy(
                        source_desc,
                        a.slot(previous),
                        [BM, BK],
                        [(pm - rank * (M // WORLD // BM)) * BM, next_k],
                        barrier=af[previous],
                    )
                else:
                    tle.gpu.copy(
                        a_desc,
                        a.slot(previous),
                        [BM, BK],
                        [pm * BM, next_k],
                        barrier=af[previous],
                    )
                tle.gpu.copy(
                    b_desc,
                    b.slot(previous),
                    [256, BK],
                    [pn * 256, next_k],
                    barrier=bf[previous],
                )
        acc = tle.gpu.wgmma_wait(0, acc)
        if PREFETCH_COUNT > 0:
            # Claim and stage the following tile while this tile's epilogue runs.
            if tile_id < BULK_END and tile_id + STRIDE < BULK_END:
                next_tile = tile_id + STRIDE
            else:
                next_tile = tl.atomic_add(work_queue, 1, sem="relaxed")
            if next_tile < tl.cdiv(M, BM) * num_n:
                next_group = next_tile // (ROW_GROUP * num_n)
                next_within = next_tile % (ROW_GROUP * num_n)
                next_pair = next_within // (2 * ROW_GROUP)
                next_pm = next_group * ROW_GROUP + (next_within // 2) % ROW_GROUP
                next_kind = next_tile % 2
                next_pn = tl.where(
                    next_pair < Q_HEADS,
                    2 * next_pair + next_kind,
                    2 * Q_HEADS + next_pair - Q_HEADS + next_kind * KV_HEADS,
                )
                next_pm = (next_pm + rank * (M // WORLD // BM)) % (M // BM)
                next_local_row = next_pm >= rank * (M // WORLD // BM) and next_pm < (
                    rank + 1
                ) * (M // WORLD // BM)
                next_seen = tl.where(next_local_row, target, 0).to(tl.uint64)
                while next_seen < target:
                    next_seen = tl.inline_asm_elementwise(
                        "ld.acquire.sys.global.u64 $0, [$1];",
                        "=l,l",
                        [ready + next_pm // READY_GROUP],
                        dtype=tl.uint64,
                        is_pure=False,
                        pack=1,
                    )
                tl.debug_barrier()
                tl.inline_asm_elementwise(
                    "fence.proxy.async.global; mov.u32 $0,0;",
                    "=r",
                    [],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                for ns in tl.static_range(PREFETCH_COUNT):
                    if next_local_row:
                        tle.gpu.copy(
                            source_desc,
                            a.slot(ns),
                            [BM, BK],
                            [(next_pm - rank * (M // WORLD // BM)) * BM, ns * BK],
                            barrier=af[ns],
                        )
                    else:
                        tle.gpu.copy(
                            a_desc,
                            a.slot(ns),
                            [BM, BK],
                            [next_pm * BM, ns * BK],
                            barrier=af[ns],
                        )
                    tle.gpu.copy(
                        b_desc,
                        b.slot(ns),
                        [256, BK],
                        [next_pn * 256, ns * BK],
                        barrier=bf[ns],
                    )
        _store_head(
            acc,
            query,
            key,
            value,
            gate,
            qnorm,
            knorm,
            cos,
            sin,
            cache,
            slots,
            row_start,
            pm,
            pn,
            M,
            Q_HEADS,
            KV_HEADS,
            SLOT_STRIDE,
            CACHE_BLOCK,
            CACHE_HEAD,
            CACHE_TOKEN,
            CACHE_DIM,
            CACHE_BLOCK_SIZE,
            BM,
            EPS=EPS,
        )
        if PREFETCH_COUNT == 0:
            if tile_id < BULK_END and tile_id + STRIDE < BULK_END:
                next_tile = tile_id + STRIDE
            else:
                next_tile = tl.atomic_add(work_queue, 1, sem="relaxed")
        seq += 1
        tile_id = next_tile


@triton.jit
def _persistent(
    BULK_END: tl.constexpr,
    work_queue,
    source,
    local,
    peers,
    counters,
    rank,
    a_desc,
    source_desc,
    b_desc,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    cos,
    sin,
    cache,
    slots,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    M: tl.constexpr,
    WORLD: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HIDDEN: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    COMM: tl.constexpr,
    BLOCK: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    PROGRAMS: tl.constexpr,
    DEDUP_PUBLISH: tl.constexpr,
    FLAGS_WORDS: tl.constexpr,
    HANDSHAKE_WORDS: tl.constexpr,
    EPS: tl.constexpr = 1.0e-6,
):
    first_peer: tl.constexpr = 1 if PIPE_STAGES == 3 else 0
    ROW_PARTS: tl.constexpr = (1 if WORLD == 2 else 2) if DEDUP_PUBLISH else COMM
    pid = tl.program_id(0)
    epoch = tl.atomic_add(counters + pid, 1, sem="relaxed").to(tl.uint64)
    flags: tl.constexpr = FLAGS_WORDS
    ready = local.to(tl.pointer_type(tl.uint64)) + flags + HANDSHAKE_WORDS
    if pid < COMM:
        # Previous invocation's local compute has completed before this launch;
        # all producers rendezvous before overwriting any registered input.
        _rendezvous(local, peers, pid, rank, 2 * epoch + 1, flags, WORLD)
        ranks = tl.arange(0, WORLD)
        destinations = tl.load(peers + ranks).to(tl.pointer_type(tl.uint64))
        # Each row group is owned by ROW_PARTS producers. The legacy path
        # retains all producers per group and its original ordering.
        first_group = 0 if ROW_PARTS == COMM else pid // ROW_PARTS
        group_stride: tl.constexpr = 1 if ROW_PARTS == COMM else COMM // ROW_PARTS
        first_block = pid if ROW_PARTS == COMM else pid % ROW_PARTS
        for row_chunk in range(first_group, M // WORLD // BM // GROUP, group_stride):
            row_tile = row_chunk * GROUP
            pm = rank * (M // WORLD // BM) + row_tile
            src = source.to(tl.pointer_type(tl.uint32))
            # Pipeline producer loads while publishing complete row groups.
            for block in tl.range(
                first_block,
                tl.cdiv(BM * GROUP * HIDDEN // 8, BLOCK),
                ROW_PARTS,
                num_stages=((6 if WORLD == 2 else 3) if DEDUP_PUBLISH else 3),
            ):
                group = block * BLOCK + tl.arange(0, BLOCK)
                valid = group < BM * GROUP * HIDDEN // 8
                offset = group * 4
                values = tl.load(
                    src
                    + row_tile * (BM * HIDDEN // 2)
                    + offset[:, None]
                    + tl.arange(0, 4)[None, :],
                    valid[:, None],
                    0,
                )
                even, odd = tl.split(values.reshape(BLOCK, 2, 2))
                x0, x2 = tl.split(even)
                x1, x3 = tl.split(odd)
                for peer_step in tl.static_range(first_peer, WORLD):
                    destination = (
                        (rank + peer_step) % WORLD if PIPE_STAGES == 3 else peer_step
                    )
                    address = (
                        tl.load(peers + destination)
                        + (pm * (BM * HIDDEN // 2) + offset).to(tl.uint64) * 4
                    )
                    tl.inline_asm_elementwise(
                        "{.reg .pred p; setp.ne.u32 p,$6,0; "
                        "@p st.relaxed.sys.global.v4.u32 [$1],{$2,$3,$4,$5}; "
                        "mov.u32 $0,0;}",
                        constraints="=r,l,r,r,r,r,r",
                        args=[address, x0, x1, x2, x3, valid.to(tl.int32)],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
            # Publish only after every writing thread has fenced the payload.
            # One owner releases the whole group directly; multiple owners
            # retain the system acq_rel arrival chain before consumers acquire.
            tl.inline_asm_elementwise(
                "fence.proxy.alias; mov.u32 $0,0;",
                constraints="=r",
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            tl.debug_barrier()
            if DEDUP_PUBLISH and ROW_PARTS == 1:
                tl.atomic_xchg(
                    destinations + flags + HANDSHAKE_WORDS + pm // GROUP,
                    1,
                    mask=ranks != rank,
                    sem="release",
                    scope="sys",
                )
            else:
                tl.atomic_add(
                    destinations + flags + HANDSHAKE_WORDS + pm // GROUP,
                    1,
                    mask=(ranks != rank) | (not DEDUP_PUBLISH),
                    sem="acq_rel",
                    scope="sys",
                )
        first_tile = tl.atomic_add(work_queue, 1, sem="relaxed")
    else:
        first_tile = pid - COMM
        if first_tile >= BULK_END:
            first_tile = tl.atomic_add(work_queue, 1, sem="relaxed")
    _fused_projection(
        BULK_END,
        PROGRAMS - COMM,
        work_queue,
        first_tile,
        a_desc,
        source_desc,
        rank,
        b_desc,
        query,
        key,
        value,
        gate,
        qnorm,
        knorm,
        cos,
        sin,
        cache,
        slots,
        0,
        SLOT_STRIDE,
        CACHE_BLOCK,
        CACHE_HEAD,
        CACHE_TOKEN,
        CACHE_DIM,
        M,
        Q_HEADS,
        KV_HEADS,
        HIDDEN,
        CACHE_BLOCK_SIZE,
        BM,
        64,
        PIPE_STAGES,
        ready,
        ROW_PARTS,
        WORLD,
        GROUP,
        EPS=EPS,
    )


@dataclass(frozen=True)
class _DevicePolicy:
    comm_programs: int
    block: int
    row_group: int
    pipeline_stages: int
    programs: int
    ownership_enabled: bool

    @property
    def compute_programs(self):
        return self.programs - self.comm_programs


def _select_device_policy(tp, tokens, sm_count, hidden_size):
    """Choose the shared offline policy before collective allocation.

    All ranks must choose the same policy. Keep the measured short-K settings;
    other workloads use the established four-stage transport and schedule.
    """
    exact_stages = hasattr(ir.builder, "mark_logical_alloc_candidate")
    short_k = exact_stages and hidden_size // 64 <= 32 and tp in (2, 4)
    pipeline_stages = 3 if short_k else 4
    comm_programs = min(7, sm_count - 1)
    block = 1024  # Each block contains 1024 packed 16-byte input vectors.
    row_group = 2 if tokens % (tp * 256) == 0 else 1
    ownership_enabled = False

    if short_k:
        # (producer CTAs, producers per row group, row tiles per group).
        producers, row_parts, owner_group = (4, 1, 2) if tp == 2 else (6, 2, 1)
        owner_count = producers // row_parts
        groups_per_rank = tokens // (tp * 128 * owner_group)
        min_groups_per_owner = 16
        if (
            producers < sm_count
            and groups_per_rank >= min_groups_per_owner * owner_count
        ):
            ownership_enabled = True
            comm_programs = producers
            if tokens % (tp * 128 * owner_group) == 0:
                row_group = owner_group

    return _DevicePolicy(
        comm_programs=comm_programs,
        block=block,
        row_group=row_group,
        pipeline_stages=pipeline_stages,
        programs=sm_count,
        ownership_enabled=ownership_enabled,
    )


@triton.jit
def _copy_tma_input(source, destination, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offset, offset < N, other=0)
    tl.store(destination + offset, value, offset < N)


def _align_tma_input(value):
    """Stage exceptional layouts with fixed copy tiles; no online autotuning."""
    if value.data_ptr() % 16 == 0:
        return value
    aligned = torch.empty_like(value)
    _copy_tma_input[(triton.cdiv(value.numel(), 1024),)](
        value, aligned, value.numel(), BLOCK=1024, num_warps=4
    )
    return aligned


class _DevicePrefill:
    """Per-call persistent launch using a capacity-managed communicator.

    Calls for one instance must be serialized on one stream. All ranks must use
    the same transport and call order. All selected CTAs must be concurrently resident.
    """

    def __init__(self, group, tokens, config, communication, sm_count):
        self.tokens = tokens
        self.communication = communication
        self.policy = _select_device_policy(
            group.size(), tokens, sm_count, config.hidden_size
        )

    def __call__(self, x):
        c, t, rt = x.config, self.tokens, self.communication
        policy = self.policy
        # Preserve the same collective protocol even when only one rank has
        # offset views. Copy on every call so Graph replay also sees new data.
        hidden = _align_tma_input(x.hidden)
        weight = _align_tma_input(x.weight)
        # Keep a dynamic tail for producers after communication and for load balance.
        # Align the static prefix so every compute CTA owns the same tile count.
        num_tiles = (t // 128) * 2 * (c.num_q_heads + c.num_kv_heads)
        bulk = (num_tiles * 7 // 8 // policy.compute_programs) * policy.compute_programs
        q = x.hidden.new_empty((t, c.q_size))
        k = x.hidden.new_empty((t, c.kv_size))
        v, g = torch.empty_like(k), torch.empty_like(q)
        _prepare_rope[(triton.cdiv(t, 128),)](
            x.positions,
            x.rope,
            rt.cos,
            rt.sin,
            rt.work_queue,
            rt.flags[rt.handshake_words :],
            *x.positions.stride(),
            x.rope.stride(0),
            M=t,
            INITIAL=bulk,
            BM=128,
            SECTION_H=c.mrope_section[1],
            SECTION_W=c.mrope_section[2],
            num_warps=4,
        )
        kernel = _persistent[(policy.programs,)](
            bulk,
            rt.work_queue,
            hidden,
            rt.buffer,
            rt.peers,
            rt.counters,
            rt.rank,
            TensorDescriptor.from_tensor(rt.buffer, [128, 64]),
            TensorDescriptor.from_tensor(hidden, [128, 64]),
            TensorDescriptor.from_tensor(weight, [256, 64]),
            q,
            k,
            v,
            g,
            x.q_norm,
            x.k_norm,
            rt.cos,
            rt.sin,
            x.cache,
            x.slots,
            x.slots.stride(0),
            *x.cache.stride(),
            M=t,
            WORLD=rt.world,
            Q_HEADS=c.num_q_heads,
            KV_HEADS=c.num_kv_heads,
            HIDDEN=c.hidden_size,
            CACHE_BLOCK_SIZE=x.cache.shape[2],
            COMM=policy.comm_programs,
            BLOCK=policy.block,
            BM=128,
            GROUP=policy.row_group,
            PIPE_STAGES=policy.pipeline_stages,
            PROGRAMS=policy.programs,
            DEDUP_PUBLISH=policy.ownership_enabled,
            FLAGS_WORDS=rt.data_words,
            HANDSHAKE_WORDS=rt.handshake_words,
            EPS=c.eps,
            num_warps=8,
        )
        self.resources = {
            "registers": kernel.n_regs,
            "spills": kernel.n_spills,
            "shared_bytes": kernel.metadata.shared,
        }
        return q, k, v, g.view(t, c.num_q_heads, c.head_dim)


class _Workspace:
    """Reuse capacity; retain captured generations until explicit collective cleanup."""

    def __init__(self, group, config, sm_count):
        self.group = group
        self.config = config
        self.sm_count = sm_count
        self.communication = None
        self.captured = False
        self.retired = []

    def __call__(self, inputs):
        tokens = inputs.positions.shape[1]
        capturing = torch.cuda.is_current_stream_capturing()
        current = self.communication
        if current is None or tokens > current.capacity:
            if capturing:
                raise RuntimeError(
                    "warm up sufficient workspace capacity before capture"
                )
            capacity = tokens if current is None else max(tokens, 2 * current.capacity)
            replacement = _MappedCommunicator(
                self.group, capacity, self.config.hidden_size, self.sm_count
            )
            if current is not None:
                if self.captured:
                    self.retired.append(current)
                else:
                    current.close()
            self.communication = replacement
            self.captured = False
        self.captured |= capturing
        launch = _DevicePrefill(
            self.group, tokens, self.config, self.communication, self.sm_count
        )
        result = launch(inputs)
        self.resources = launch.resources
        return result

    def close(self):
        for communication in self.retired:
            communication.close()
        if self.communication is not None:
            self.communication.close()
        self.retired.clear()
        self.communication = None


_windows = {}


def megapreattention(
    hidden,
    weight,
    positions,
    q_norm,
    k_norm,
    cos_sin_cache,
    kv_cache,
    slot_mapping,
    group,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    mrope_section,
    eps=1e-6,
):
    """AllGather, gated QKV projection, Gemma RMSNorm/MRoPE and KV-cache update.

    Head counts are global (before TP sharding). Q heads divide evenly across
    TP2/4/8; KV heads shard or replicate evenly. BF16 inference on SM90 only.
    Hidden width is at least 256 and divisible by 64; head_dim=256 and
    rotary_dim=64. Three nonnegative interleaved MRoPE sections sum to 32.
    eps is finite and nonnegative; positions has three axes.

    Each rank supplies a positive multiple of 128 tokens. The persistent grid
    uses one CTA per device SM; all ranks must have the same SM count and policy.
    Unsupported shapes are rejected rather than dispatched to another algorithm.
    Contiguous inputs with unaligned base addresses use a Triton staging copy
    on the persistent path; all ranks retain the same communication protocol.

    hidden is contiguous [T/TP, H]. weight is contiguous
    [2 * (local_q_heads + local_kv_heads) * D, H], packed as interleaved Q/gate
    per query head, then all K heads, then all V heads. positions is int64
    [3,T]; cos_sin_cache is BF16 [max_position, rotary_dim]. Norm weights are
    [D] (Gemma uses 1+w).
    kv_cache is [blocks, local_kv_heads, block_size, 2*D], with K then V;
    valid slots must be unique and in bounds, and negative slots are ignored.
    Positions must index cos_sin_cache. No tensor-value scans are performed.

    Returns (Q, K, V, gate), BF16 shapes [T,local_q_heads*D],
    [T,local_kv_heads*D], [T,local_kv_heads*D],
    [T,local_q_heads,D]. gate is before activation. kv_cache is updated in-place.
    Inference only; no autograd. Inputs/outputs must not alias kv_cache.

    Workspace initializes lazily and reuses capacity across token counts. Growth
    reserves at least twice the previous capacity. Warm up sufficient capacity
    and kernel specializations before CUDA Graph capture; capture cannot grow it.
    Calls must be serialized on one stream with identical rank order. Captured
    allocations survive growth so existing graphs remain valid. Cleanup is
    optional for normal use; to reclaim cached allocations early, reset all
    dependent graphs and call clear_megapreattention_cache on every rank.

    No online autotune: producer CTA counts and ready-signal layouts must match
    on all ranks. Independent per-rank tuning can deadlock this protocol.
    The offline policy is kept in _select_device_policy.
    """
    # All public parameter/layout checks precede collective initialization.
    assert tle is not None, "Triton TLE is required"
    assert hidden.ndim == 2 and hidden.device.type == "cuda"
    local_tokens, hidden_size = hidden.shape
    tp = group.size()
    assert tp in (2, 4, 8)
    assert local_tokens > 0 and local_tokens % 128 == 0
    assert hidden_size >= 256 and hidden_size % 64 == 0
    assert head_dim == 256 and rotary_dim == 64
    assert num_heads > 0 and num_heads % tp == 0
    assert num_kv_heads > 0 and (
        num_kv_heads % tp == 0 if num_kv_heads >= tp else tp % num_kv_heads == 0
    )
    sections = tuple(mrope_section)
    assert len(sections) == 3 and sum(sections) == 32
    assert all(isinstance(part, int) and part >= 0 for part in sections)
    assert math.isfinite(eps) and eps >= 0
    device = hidden.device
    assert torch.cuda.get_device_capability(device)[0] == 9
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    assert sm_count > 1
    tokens = local_tokens * tp
    config = _Config(
        num_heads // tp,
        max(1, num_kv_heads // tp),
        head_dim,
        rotary_dim,
        sections,
        eps,
        hidden_size,
    )
    assert hidden.is_contiguous()
    assert weight.shape == (config.packed_size, hidden_size) and weight.is_contiguous()
    assert positions.shape == (3, tokens) and slot_mapping.shape == (tokens,)
    assert positions.dtype == slot_mapping.dtype == torch.int64
    assert positions.device == slot_mapping.device == device
    assert all(w.shape == (head_dim,) and w.is_contiguous() for w in (q_norm, k_norm))
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] == rotary_dim
    assert cos_sin_cache.shape[0] > 0 and cos_sin_cache.stride(1) == 1
    assert kv_cache.ndim == 4 and kv_cache.shape[0] > 0 and kv_cache.shape[2] > 0
    assert (
        kv_cache.shape[1] == config.num_kv_heads and kv_cache.shape[3] == 2 * head_dim
    )
    assert all(
        value.dtype == torch.bfloat16
        and value.device == device
        and not value.requires_grad
        for value in (hidden, weight, q_norm, k_norm, cos_sin_cache, kv_cache)
    )
    inputs = _Inputs(
        hidden,
        weight,
        positions,
        q_norm,
        k_norm,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        group,
        config,
    )
    key = (id(group), hidden.device, config)
    with torch.cuda.device(hidden.device):
        if key not in _windows:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up this shape before capture")
            _windows[key] = _Workspace(group, config, sm_count)
        return _windows[key](inputs)


def clear_megapreattention_cache():
    """Optional early release; reset all dependent graphs and call on every rank."""
    for window in _windows.values():
        window.close()
    _windows.clear()

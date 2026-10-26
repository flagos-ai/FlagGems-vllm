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

"""Persistent INT8 MoE with dynamic CTA transitions between stage-local queues.

The acquire CAS must remain an atomic transaction: role CTAs poll device-side
work queues while other resident CTAs publish their payloads.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def peer_i8(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.int8))


@triton.jit
def peer_i32(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.int32))


@triton.jit
def peer_f32(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.float32))


@triton.jit
def peer_bf16(tab, pe):
    return tl.load(tab + pe).to(tl.pointer_type(tl.bfloat16))


@triton.jit
def fence_sys():
    _ = tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0, 0;",
        "=r,~{memory}",
        [],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def load_acquire(ptr):
    sentinel = tl.full((), -1, ptr.dtype.element_ty)
    return tl.atomic_cas(ptr, sentinel, sentinel, sem="acquire", scope="gpu")


@triton.jit
def peer_barrier(sig_tab, slot, MY_PE: tl.constexpr, NPES: tl.constexpr):
    """One combine CTA per rank participates after all L2 peer stores finish."""
    fence_sys()
    tl.debug_barrier()
    for pe in tl.static_range(0, NPES):
        tl.atomic_add(peer_i32(sig_tab, pe) + slot, 1, sem="release", scope="sys")
    sl = peer_i32(sig_tab, MY_PE)
    v = tl.atomic_cas(sl + slot, -1, -1, sem="acquire", scope="sys")
    while v < NPES:
        v = tl.atomic_cas(sl + slot, -1, -1, sem="acquire", scope="sys")
    tl.debug_barrier()
    fence_sys()


@triton.jit
def wait_count(ptr, expected):
    """Wait for a role CTA group without requiring a whole-grid barrier."""
    v = load_acquire(ptr)
    while v < expected:
        v = load_acquire(ptr)


@triton.jit
def swiglu_bf16(gate, up):
    silu = tl.div_rn(gate, 1.0 + tl.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    return (silu * up).to(tl.bfloat16)


@triton.jit
def quantize_token_int8(x):
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-10)
    scale = amax / 127.0
    q = libdevice.round(x * (127.0 / amax[:, None])).to(tl.int8)
    return (q, scale)


@triton.jit
def cta_l1_gemm(
    block_meta,
    l1_token,
    l1_sf,
    w1_kt,
    w1_sf,
    l2_acts,
    l2_sf,
    l2_bf16,
    l2_arrival_mask,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    INTER: tl.constexpr,
    L1_N: tl.constexpr,
    L1_OUT_N: tl.constexpr,
    NPAIR: tl.constexpr,
    NK1: tl.constexpr,
    NL1N: tl.constexpr,
    task_id,
):
    bid = task_id
    pool_block = tl.load(block_meta + bid * 4 + 0, cache_modifier=".cg")
    cur_e = tl.load(block_meta + bid * 4 + 1, cache_modifier=".cg")
    n_block = tl.load(block_meta + bid * 4 + 2, cache_modifier=".cg")
    valid_m = tl.load(block_meta + bid * 4 + 3, cache_modifier=".cg")
    if valid_m > 0:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_ok = offs_m < valid_m
        n_cols = n_block * BLOCK_N + offs_n
        n_ok = n_cols < L1_N
        rows = pool_block * BLOCK_M + offs_m
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for kb in tl.range(0, NK1, num_stages=2, loop_unroll_factor=1):
            a = tl.load(
                l1_token + rows[:, None] * K + (kb * BLOCK_K + offs_k)[None, :],
                mask=m_ok[:, None],
                other=0,
            )
            bp = tl.make_block_ptr(
                base=w1_kt + cur_e * K * L1_N,
                shape=(K, L1_N),
                strides=(1, K),
                offsets=(kb * BLOCK_K, n_block * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            b = tl.load(bp)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        sf = tl.load(l1_sf + rows, mask=m_ok, other=0.0)
        sfb = tl.load(w1_sf + cur_e * L1_N + n_cols, mask=n_ok, other=0.0)
        acc = (
            (acc.to(tl.float32) * sf[:, None] * sfb[None, :])
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        t4 = tl.reshape(acc, (BLOCK_M, NPAIR, 2, 8))
        t4 = tl.permute(t4, (0, 1, 3, 2))
        gate, up = tl.split(t4)
        gate = tl.reshape(gate, (BLOCK_M, L1_OUT_N))
        up = tl.reshape(up, (BLOCK_M, L1_OUT_N))
        sw = swiglu_bf16(gate, up)
        out_cols = n_block * L1_OUT_N + tl.arange(0, L1_OUT_N)
        tl.store(
            l2_bf16 + rows[:, None] * INTER + out_cols[None, :],
            sw.to(tl.bfloat16),
            mask=m_ok[:, None] & (out_cols[None, :] < INTER),
        )
        tl.debug_barrier()
    bit = tl.full((), 1, tl.int64) << n_block
    old_mask = tl.atomic_or(l2_arrival_mask + pool_block, bit, sem="acq_rel")
    if old_mask | bit == (1 << NL1N) - 1:
        qc = tl.arange(0, triton.next_power_of_2(INTER))
        for qr in tl.range(0, valid_m, 8):
            rr = qr + tl.arange(0, 8)
            pr = pool_block * BLOCK_M + rr
            x = tl.load(
                l2_bf16 + pr[:, None] * INTER + qc[None, :],
                mask=(rr[:, None] < valid_m) & (qc[None, :] < INTER),
                other=0,
            ).to(tl.float32)
            q, scale = quantize_token_int8(x)
            tl.store(
                l2_acts + pr[:, None] * INTER + qc[None, :],
                q,
                mask=(rr[:, None] < valid_m) & (qc[None, :] < INTER),
            )
            tl.store(l2_sf + pr, scale, mask=rr < valid_m)
        tl.debug_barrier()


@triton.jit
def cta_l2_gemm(
    block_meta,
    l1_w,
    l2_acts,
    l2_sf,
    w2_kt,
    w2_sf,
    meta_tab,
    cb_tab,
    MY_PE: tl.constexpr,
    NTOK,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    INTER: tl.constexpr,
    L2_N: tl.constexpr,
    NK2: tl.constexpr,
    task_id,
):
    bid = task_id
    pool_block = tl.load(block_meta + bid * 4 + 0, cache_modifier=".cg")
    cur_e = tl.load(block_meta + bid * 4 + 1, cache_modifier=".cg")
    n_block = tl.load(block_meta + bid * 4 + 2, cache_modifier=".cg")
    valid_m = tl.load(block_meta + bid * 4 + 3, cache_modifier=".cg")
    if valid_m > 0:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        m_ok = offs_m < valid_m
        rows = pool_block * BLOCK_M + offs_m
        cols = n_block * BLOCK_N + offs_n
        n_ok = cols < L2_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        kk = tl.arange(0, BLOCK_K)
        for kb in tl.range(0, NK2, num_stages=2):
            a = tl.load(
                l2_acts + rows[:, None] * INTER + (kb * BLOCK_K + kk)[None, :],
                mask=m_ok[:, None],
                other=0,
            )
            bp = tl.make_block_ptr(
                base=w2_kt + cur_e * INTER * L2_N,
                shape=(INTER, L2_N),
                strides=(1, INTER),
                offsets=(kb * BLOCK_K, n_block * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            b = tl.load(bp)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        sf = tl.load(l2_sf + rows, mask=m_ok, other=0.0)
        sfb = tl.load(w2_sf + cur_e * L2_N + cols, mask=n_ok, other=0.0)
        router = tl.load(l1_w + rows, mask=m_ok, other=0.0)
        acc = acc.to(tl.float32) * sf[:, None] * sfb[None, :] * router[:, None]
        meta_l = peer_i32(meta_tab, MY_PE)
        md = meta_l + rows * 3
        dst_rank = tl.load(md + 0, mask=m_ok, other=0)
        dst_tok = tl.load(md + 1, mask=m_ok, other=0)
        dst_topk = tl.load(md + 2, mask=m_ok, other=0)
        cbb = peer_bf16(cb_tab, dst_rank)
        base = (dst_topk * NTOK + dst_tok) * K
        tl.store(
            tl.max_contiguous(
                tl.multiple_of(cbb[:, None] + base[:, None] + cols[None, :], (1, 8)),
                (1, 8),
            ),
            acc.to(tl.bfloat16),
            mask=m_ok[:, None] & n_ok[None, :],
        )


@triton.jit
def cta_l2_batch_finish(
    l2_done,
    combine_ready,
    sig_tab,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    total_tasks,
    count,
):
    # Exactly one worker reaches the total through this acquire-release chain.
    old_done = tl.atomic_add(l2_done, count, sem="acq_rel")
    if old_done + count == total_tasks:
        if NPES > 1:
            peer_barrier(sig_tab, 1, MY_PE, NPES)
        tl.debug_barrier()
        fence_sys()
        tl.atomic_xchg(combine_ready, 1, sem="release")


@triton.jit
def cta_combine_one(
    topk_local,
    cb_tab,
    final_y,
    MY_PE: tl.constexpr,
    NTOK,
    TOPK: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    task_id,
):
    """One dynamic combine task for one source token."""
    cb_l = peer_bf16(cb_tab, MY_PE)
    offs_hid = tl.arange(0, K_POW2)
    hid_ok = offs_hid < K
    t_ok = task_id < NTOK
    acc = tl.zeros((K_POW2,), dtype=tl.float32)
    for k in tl.static_range(0, TOPK):
        e = tl.load(topk_local + task_id * TOPK + k, mask=t_ok, other=-1).to(tl.int32)
        v = tl.load(
            tl.max_contiguous(
                tl.multiple_of(cb_l + (k * NTOK + task_id) * K + offs_hid, 8), 8
            ),
            mask=(e >= 0) & t_ok & hid_ok,
            other=0,
        )
        acc += v.to(tl.float32)
    tl.store(final_y + task_id * K + offs_hid, acc.to(tl.bfloat16), mask=t_ok & hid_ok)


@triton.jit
def dispatch_payload(buffers, config: tl.constexpr):
    """Build the complete route queue, payload pool, and worklists on device."""
    (
        route_tab,
        active_routes,
        active_count,
        tok_tab,
        sf_tab,
        w_tab,
        meta_tab,
        block_rows,
        dispatch_counts,
        dispatch_chunk_counts,
        dispatch_chunk_offsets,
        block_base_tab,
        l1_token,
        l1_sf,
        l1_w,
        l1_meta,
        l2_meta,
        dispatch_block_arrived,
        dispatch_phase,
        dispatch_cursor,
        l1_ready,
        NTOK,
        POOL_TOKENS,
        worker_id,
    ) = buffers
    MY_PE: tl.constexpr = config[0]
    NPES: tl.constexpr = config[1]
    TOPK: tl.constexpr = config[2]
    EPR: tl.constexpr = config[3]
    K: tl.constexpr = config[4]
    NSF: tl.constexpr = config[5]
    BLOCK_M: tl.constexpr = config[6]
    BLOCK_K: tl.constexpr = config[7]
    NK1: tl.constexpr = config[8]
    EPR_POW2: tl.constexpr = config[9]
    NL1N: tl.constexpr = config[10]
    NL2N: tl.constexpr = config[11]
    ROLE_BUDGET: tl.constexpr = config[12]
    ROUTE_CHUNK: tl.constexpr = config[13]
    ROUTE_CHUNKS: tl.constexpr = config[14]
    PACK_ROUTE_META: tl.constexpr = config[15]
    meta_l = peer_i32(meta_tab, MY_PE)
    work_id = tl.atomic_add(dispatch_cursor + 0, 1, sem="relaxed")
    while work_id < NPES * ROUTE_CHUNKS:
        src_id = work_id // ROUTE_CHUNKS
        base = work_id % ROUTE_CHUNKS * ROUTE_CHUNK
        routes_src = peer_i32(route_tab, src_id)
        lane = tl.arange(0, ROUTE_CHUNK)
        rid = base + lane
        valid = rid < NTOK * TOPK
        expert = tl.load(routes_src + rid, mask=valid, other=-1)
        owner = expert // EPR
        local_expert = expert - MY_PE * EPR
        take = valid & (expert >= 0) & (owner == MY_PE)
        safe_local_expert = tl.where(take, local_expert, 0)
        chunk_id = work_id % ROUTE_CHUNKS
        experts = tl.arange(0, EPR_POW2)
        if PACK_ROUTE_META:
            chunk_hist = tl.histogram(safe_local_expert, EPR_POW2, mask=take)
        else:
            chunk_hist = tl.sum(
                ((safe_local_expert[:, None] == experts[None, :]) & take[:, None]).to(
                    tl.int32
                ),
                axis=0,
            )
        tl.store(
            dispatch_chunk_counts + (src_id * ROUTE_CHUNKS + chunk_id) * EPR + experts,
            chunk_hist,
            mask=experts < EPR,
        )
        tl.atomic_add(
            dispatch_counts + experts * NPES + src_id,
            chunk_hist,
            sem="release",
            mask=(experts < EPR) & (chunk_hist > 0),
        )
        nactive = tl.sum(take.to(tl.int32))
        if nactive > 0:
            active_base = tl.atomic_add(active_count + src_id, nactive, sem="relaxed")
            lane_ord = tl.cumsum(take.to(tl.int32)) - 1
            if PACK_ROUTE_META:
                intra = tl.sum(
                    (
                        (lane[:, None] > lane[None, :])
                        & (expert[:, None] == expert[None, :])
                        & take[None, :]
                    ).to(tl.int32),
                    axis=1,
                )
                packed = rid | intra << 19 | safe_local_expert << 24
                tl.store(
                    active_routes + src_id * (NTOK * TOPK) + active_base + lane_ord,
                    packed,
                    mask=take,
                )
            else:
                tl.store(
                    active_routes + src_id * (NTOK * TOPK) + active_base + lane_ord,
                    rid,
                    mask=take,
                )
        work_id = tl.atomic_add(dispatch_cursor + 0, 1, sem="relaxed")
    tl.debug_barrier()
    tl.atomic_add(dispatch_phase + 0, 1, sem="release")
    wait_count(dispatch_phase + 0, ROLE_BUDGET)
    if worker_id == 0:
        experts = tl.arange(0, EPR_POW2)
        expert_ok = experts < EPR
        total = tl.zeros((EPR_POW2,), dtype=tl.int32)
        for src in tl.static_range(0, NPES):
            c = tl.load(dispatch_counts + experts * NPES + src, mask=expert_ok, other=0)
            total += c
        # Rebuild routing-dependent offsets on device for every invocation.
        block_counts = tl.where(expert_ok, tl.maximum(tl.cdiv(total, BLOCK_M), 1), 0)
        block_bases = tl.cumsum(block_counts) - block_counts
        tl.store(block_base_tab + experts, block_bases, mask=expert_ok)
        tl.store(dispatch_phase + 2, tl.sum(block_counts))
        tl.debug_barrier()
        tl.atomic_xchg(dispatch_phase + 1, 1, sem="release")
    if ROUTE_CHUNKS <= 1024:
        chunks = tl.arange(0, triton.next_power_of_2(ROUTE_CHUNKS))
        subgroup = tl.arange(0, 4)
        expert_groups = triton.cdiv(EPR, 4)
        prefix_group = worker_id
        while prefix_group < NPES * expert_groups:
            src = prefix_group // expert_groups
            le = prefix_group % expert_groups * 4 + subgroup
            counts = tl.load(
                dispatch_chunk_counts
                + (src * ROUTE_CHUNKS + chunks[:, None]) * EPR
                + le[None, :],
                mask=(chunks[:, None] < ROUTE_CHUNKS) & (le[None, :] < EPR),
                other=0,
                cache_modifier=".cg",
            )
            offsets = tl.cumsum(counts, axis=0) - counts
            tl.store(
                dispatch_chunk_offsets
                + (src * ROUTE_CHUNKS + chunks[:, None]) * EPR
                + le[None, :],
                offsets,
                mask=(chunks[:, None] < ROUTE_CHUNKS) & (le[None, :] < EPR),
            )
            prefix_group += ROLE_BUDGET
        tl.debug_barrier()
    if ROUTE_CHUNKS > 1024:
        chunks = tl.arange(0, 256)
        subgroup = tl.arange(0, 4)
        expert_groups = triton.cdiv(EPR, 4)
        prefix_group = worker_id
        while prefix_group < NPES * expert_groups:
            src = prefix_group // expert_groups
            le = prefix_group % expert_groups * 4 + subgroup
            running = tl.zeros((4,), dtype=tl.int32)
            for begin in tl.range(0, ROUTE_CHUNKS, 256, num_stages=1):
                chunk_ids = begin + chunks
                counts = tl.load(
                    dispatch_chunk_counts
                    + (src * ROUTE_CHUNKS + chunk_ids[:, None]) * EPR
                    + le[None, :],
                    mask=(chunk_ids[:, None] < ROUTE_CHUNKS) & (le[None, :] < EPR),
                    other=0,
                    cache_modifier=".cg",
                )
                offsets = tl.cumsum(counts, axis=0) - counts + running[None, :]
                tl.store(
                    dispatch_chunk_offsets
                    + (src * ROUTE_CHUNKS + chunk_ids[:, None]) * EPR
                    + le[None, :],
                    offsets,
                    mask=(chunk_ids[:, None] < ROUTE_CHUNKS) & (le[None, :] < EPR),
                )
                running += tl.sum(counts, axis=0)
            prefix_group += ROLE_BUDGET
        tl.debug_barrier()
    wait_count(dispatch_phase + 1, 1)
    active_blocks = tl.load(dispatch_phase + 2)
    expert_ids = tl.arange(0, EPR_POW2)
    expert_bases = tl.load(block_base_tab + expert_ids, mask=expert_ids < EPR, other=0)
    meta_id = tl.atomic_add(dispatch_cursor + 2, 128, sem="relaxed")
    while meta_id < active_blocks:
        blocks = meta_id + tl.arange(0, 128)
        block_ok = blocks < active_blocks
        safe_blocks = tl.where(block_ok, blocks, 0)
        block_expert = (
            tl.sum(
                (
                    (safe_blocks[:, None] >= expert_bases[None, :])
                    & (expert_ids[None, :] < EPR)
                ).to(tl.int32),
                axis=1,
            )
            - 1
        )
        block_base = tl.load(block_base_tab + block_expert, mask=block_ok, other=0)
        block_in_expert = safe_blocks - block_base
        block_total = tl.zeros((128,), dtype=tl.int32)
        for src in tl.static_range(0, NPES):
            block_total += tl.load(
                dispatch_counts + block_expert * NPES + src, mask=block_ok, other=0
            )
        valid_m = tl.minimum(
            tl.maximum(block_total - block_in_expert * BLOCK_M, 0), BLOCK_M
        )
        tl.store(block_rows + safe_blocks, valid_m, mask=block_ok)
        for n_block in tl.static_range(0, NL1N):
            dst = l1_meta + (safe_blocks * NL1N + n_block) * 4
            tl.store(dst + 0, safe_blocks, mask=block_ok)
            tl.store(dst + 1, block_expert, mask=block_ok)
            tl.store(dst + 2, n_block, mask=block_ok)
            tl.store(dst + 3, valid_m, mask=block_ok)
        for n_block in tl.static_range(0, NL2N):
            dst = l2_meta + (safe_blocks * NL2N + n_block) * 4
            tl.store(dst + 0, safe_blocks, mask=block_ok)
            tl.store(dst + 1, block_expert, mask=block_ok)
            tl.store(dst + 2, n_block, mask=block_ok)
            tl.store(dst + 3, valid_m, mask=block_ok)
        empty = block_ok & (valid_m == 0)
        for n_block in tl.static_range(0, NL1N):
            tl.atomic_xchg(
                l1_ready + safe_blocks * NL1N + n_block,
                1,
                sem="release",
                mask=empty,
            )
        fence_sys()
        meta_id = tl.atomic_add(dispatch_cursor + 2, 128, sem="relaxed")
    tl.debug_barrier()
    tl.atomic_add(dispatch_phase + 3, 1, sem="release")
    wait_count(dispatch_phase + 3, ROLE_BUDGET)
    fence_sys()
    tl.debug_barrier()
    wait_count(dispatch_phase + 1, 1)
    source_ids = tl.arange(0, triton.next_power_of_2(NPES))
    source_routes = tl.load(active_count + source_ids, mask=source_ids < NPES, other=0)
    source_chunks = (source_routes + ROUTE_CHUNK - 1) // ROUTE_CHUNK
    chunk_ends = tl.cumsum(source_chunks)
    pass2_chunks = tl.sum(source_chunks)
    work_id = tl.atomic_add(dispatch_cursor + 1, 1, sem="relaxed")
    while work_id < pass2_chunks:
        src_id = tl.sum(((work_id >= chunk_ends) & (source_ids < NPES)).to(tl.int32))
        source_begin = tl.sum(tl.where(source_ids < src_id, source_chunks, 0))
        base = (work_id - source_begin) * ROUTE_CHUNK
        active_total = tl.load(active_count + src_id)
        lane = tl.arange(0, ROUTE_CHUNK)
        active_id = base + lane
        active = active_id < active_total
        if PACK_ROUTE_META:
            packed = tl.load(
                active_routes + src_id * (NTOK * TOPK) + active_id, mask=active, other=0
            )
            rid = packed & (1 << 19) - 1
            intra = packed >> 19 & 31
            safe_expert = packed >> 24
            take = active
            src_tok_ptr = peer_i8(tok_tab, src_id)
            src_sf_ptr = peer_f32(sf_tab, src_id)
            src_w_ptr = peer_f32(w_tab, src_id)
            chunk_id = rid // ROUTE_CHUNK
        else:
            rid = tl.load(
                active_routes + src_id * (NTOK * TOPK) + active_id, mask=active, other=0
            )
            routes_src = peer_i32(route_tab, src_id)
            src_tok_ptr = peer_i8(tok_tab, src_id)
            src_sf_ptr = peer_f32(sf_tab, src_id)
            src_w_ptr = peer_f32(w_tab, src_id)
            expert = tl.load(routes_src + rid, mask=active, other=-1)
            owner = expert // EPR
            local_expert = expert - MY_PE * EPR
            take = active & (expert >= 0) & (owner == MY_PE)
            safe_expert = tl.where(take, local_expert, 0)
            chunk_id = rid // ROUTE_CHUNK
            chunk_base = chunk_id * ROUTE_CHUNK
            chunk_lane = tl.arange(0, ROUTE_CHUNK)
            chunk_rid = chunk_base[:, None] + chunk_lane[None, :]
            chunk_valid = chunk_rid < NTOK * TOPK
            chunk_expert = tl.load(routes_src + chunk_rid, mask=chunk_valid, other=-1)
            intra = tl.sum(
                (
                    (chunk_lane[None, :] < (rid % ROUTE_CHUNK)[:, None])
                    & (chunk_expert == expert[:, None])
                    & (chunk_expert >= 0)
                ).to(tl.int32),
                axis=1,
            )
        chunk_off = tl.load(
            dispatch_chunk_offsets
            + (src_id * ROUTE_CHUNKS + chunk_id) * EPR
            + safe_expert,
            mask=take,
            other=0,
        )
        local_ord = chunk_off + intra
        global_base = tl.zeros_like(local_ord)
        for prev_src in tl.static_range(0, NPES):
            prev_count = tl.load(
                dispatch_counts + safe_expert * NPES + prev_src, mask=take, other=0
            )
            global_base += prev_count * (prev_src < src_id).to(tl.int32)
        global_ord = global_base + local_ord
        pool_row = (
            tl.load(block_base_tab + safe_expert, mask=take, other=0) * BLOCK_M
            + global_ord
        )
        src_tok = rid // TOPK
        src_topk = rid % TOPK
        for kb in tl.range(0, NK1, num_stages=2, loop_unroll_factor=1):
            offs_k = tl.arange(0, BLOCK_K)
            x = tl.load(
                tl.max_contiguous(
                    tl.multiple_of(
                        src_tok_ptr
                        + src_tok[:, None] * K
                        + kb * BLOCK_K
                        + offs_k[None, :],
                        (1, 16),
                    ),
                    (1, 16),
                ),
                mask=take[:, None],
                other=0,
            )
            tl.store(
                l1_token + pool_row[:, None] * K + kb * BLOCK_K + offs_k[None, :],
                x,
                mask=take[:, None],
            )
        for sf_block in tl.static_range(0, NSF):
            sf = tl.load(src_sf_ptr + src_tok * NSF + sf_block, mask=take, other=0.0)
            tl.store(l1_sf + sf_block * POOL_TOKENS + pool_row, sf, mask=take)
        gate_w = tl.load(src_w_ptr + rid, mask=take, other=0.0)
        tl.store(l1_w + pool_row, gate_w, mask=take)
        tl.store(meta_l + pool_row * 3 + 0, src_id, mask=take)
        tl.store(meta_l + pool_row * 3 + 1, src_tok, mask=take)
        tl.store(meta_l + pool_row * 3 + 2, src_topk, mask=take)
        tl.debug_barrier()
        fence_sys()
        block_id = pool_row // BLOCK_M
        valid_block_m = tl.load(
            block_rows + block_id, mask=take, other=0, cache_modifier=".cg"
        )
        # The last arrival acquires all earlier payload publications before
        # it releases the separate L1-ready flag.
        old_arrived = tl.atomic_add(
            dispatch_block_arrived + block_id, 1, sem="acq_rel", mask=take
        )
        ready = take & (old_arrived + 1 == valid_block_m)
        for n_block in tl.static_range(0, NL1N):
            tl.atomic_xchg(
                l1_ready + block_id * NL1N + n_block, 1, sem="release", mask=ready
            )
        work_id = tl.atomic_add(dispatch_cursor + 1, 1, sem="relaxed")


# Bound dispatch register live ranges for larger tiles. Small tiles keep
# dispatch inline to avoid device-call overhead.
@triton.jit(noinline=True)
def dispatch_payload_noinline(buffers, config: tl.constexpr):
    dispatch_payload(buffers, config)


@triton.jit(
    do_not_specialize=[
        "NTOK",
        "POOL_TOKENS",
    ]
)
def cta_role_megakernel(
    route_tab,
    active_routes,
    active_count,
    block_rows,
    dispatch_counts,
    dispatch_chunk_counts,
    dispatch_chunk_offsets,
    block_base_tab,
    l1_meta,
    l2_meta,
    dispatch_block_arrived,
    dispatch_phase,
    dispatch_cursor,
    topk_local,
    l1_token,
    l1_sf,
    l1_w,
    tok_tab,
    sf_tab,
    w_tab,
    w1_kt,
    w1_sf,
    l2_acts,
    l2_sf,
    l2_arrival_mask,
    l2_done,
    l1_done,
    l1_ready,
    task_cursor,
    combine_ready,
    meta_tab,
    cb_tab,
    w2_kt,
    w2_sf,
    final_y,
    sig_tab,
    l2_bf16,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NTOK,
    TOPK: tl.constexpr,
    K: tl.constexpr,
    INTER: tl.constexpr,
    L1_N: tl.constexpr,
    L2_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    L1_OUT_N: tl.constexpr,
    NPAIR: tl.constexpr,
    NK1: tl.constexpr,
    NK2: tl.constexpr,
    POOL_TOKENS,
    K_POW2: tl.constexpr,
    NL1N: tl.constexpr,
    NL2N: tl.constexpr,
    NSF: tl.constexpr,
    EPR: tl.constexpr,
    EPR_POW2: tl.constexpr,
    ROLE_BUDGET: tl.constexpr,
    ROUTE_CHUNK: tl.constexpr,
    ROUTE_CHUNKS: tl.constexpr,
    L2_GROUP_M: tl.constexpr = 64,
    AGGREGATE_COMPLETION: tl.constexpr = False,
    LARGE_NP2_TUNE: tl.constexpr = False,
    PACK_ROUTE_META: tl.constexpr = False,
):
    """Fuse dispatch, both GEMMs and combine using persistent CTA work queues.

    All resident CTAs begin with dispatch. A CTA that drains its payload work
    can acquire a ready L1 tile while other CTAs still move payloads. Roles
    change dynamically; there is no fixed communication/math CTA partition.
    L1 completion gates L2, and all peer L2 stores finish before combine.
    """
    tl.debug_barrier()
    worker_id = tl.program_id(0)
    dispatch_buffers = (
        route_tab,
        active_routes,
        active_count,
        tok_tab,
        sf_tab,
        w_tab,
        meta_tab,
        block_rows,
        dispatch_counts,
        dispatch_chunk_counts,
        dispatch_chunk_offsets,
        block_base_tab,
        l1_token,
        l1_sf,
        l1_w,
        l1_meta,
        l2_meta,
        dispatch_block_arrived,
        dispatch_phase,
        dispatch_cursor,
        l1_ready,
        NTOK,
        POOL_TOKENS,
        worker_id,
    )
    dispatch_config: tl.constexpr = (
        MY_PE,
        NPES,
        TOPK,
        EPR,
        K,
        NSF,
        BLOCK_M,
        BLOCK_K,
        NK1,
        EPR_POW2,
        NL1N,
        NL2N,
        ROLE_BUDGET,
        ROUTE_CHUNK,
        ROUTE_CHUNKS,
        PACK_ROUTE_META,
    )
    if BLOCK_M >= 32:
        dispatch_payload_noinline(dispatch_buffers, dispatch_config)
    else:
        dispatch_payload(dispatch_buffers, dispatch_config)
    fence_sys()
    tl.debug_barrier()
    # Schedule only the tiles produced by this invocation's routing.
    active_blocks = tl.load(dispatch_phase + 2)
    active_l1_tasks = active_blocks * NL1N
    active_l2_tasks = active_blocks * NL2N
    l1_count = tl.full((), 0, tl.int32)
    task = tl.atomic_add(task_cursor + 0, 1, sem="relaxed")
    while task < active_l1_tasks:
        claimed = tl.atomic_cas(l1_ready + task, 1, 2, sem="acq_rel")
        if claimed != 1:
            while claimed != 1:
                claimed = tl.atomic_cas(l1_ready + task, 1, 2, sem="acq_rel")
        tl.debug_barrier()
        cta_l1_gemm(
            l1_meta,
            l1_token,
            l1_sf,
            w1_kt,
            w1_sf,
            l2_acts,
            l2_sf,
            l2_bf16,
            l2_arrival_mask,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=256 if LARGE_NP2_TUNE else BLOCK_K,
            K=K,
            INTER=INTER,
            L1_N=L1_N,
            L1_OUT_N=L1_OUT_N,
            NPAIR=NPAIR,
            NK1=K // 256 if LARGE_NP2_TUNE else NK1,
            NL1N=NL1N,
            task_id=task,
        )
        tl.debug_barrier()
        if AGGREGATE_COMPLETION:
            l1_count += 1
        else:
            tl.atomic_add(l1_done, 1, sem="release")
        task = tl.atomic_add(task_cursor + 0, 1, sem="relaxed")
    if AGGREGATE_COMPLETION:
        tl.debug_barrier()
        if l1_count > 0:
            tl.atomic_add(l1_done, l1_count, sem="acq_rel")
    wait_count(l1_done, active_l1_tasks)
    fence_sys()
    tl.debug_barrier()
    count = tl.full((), 0, tl.int32)
    batch_size: tl.constexpr = 4 if AGGREGATE_COMPLETION else 1
    batch = tl.atomic_add(task_cursor + 1, batch_size, sem="relaxed")
    while batch < active_l2_tasks:
        for offset in tl.range(0, batch_size, loop_unroll_factor=1):
            logical = batch + offset
            if logical < active_l2_tasks:
                first = logical // (L2_GROUP_M * NL2N) * L2_GROUP_M
                rem = logical % (L2_GROUP_M * NL2N)
                if first + L2_GROUP_M <= active_blocks:
                    task = (first + rem % L2_GROUP_M) * NL2N + rem // L2_GROUP_M
                else:
                    group_m = active_blocks - first
                    task = (first + rem % group_m) * NL2N + rem // group_m
                cta_l2_gemm(
                    l2_meta,
                    l1_w,
                    l2_acts,
                    l2_sf,
                    w2_kt,
                    w2_sf,
                    meta_tab,
                    cb_tab,
                    MY_PE=MY_PE,
                    NTOK=NTOK,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=K // NL2N,
                    BLOCK_K=256 if LARGE_NP2_TUNE else BLOCK_K,
                    K=K,
                    INTER=INTER,
                    L2_N=L2_N,
                    NK2=INTER // 256 if LARGE_NP2_TUNE else NK2,
                    task_id=task,
                )
                count += 1
        batch = tl.atomic_add(task_cursor + 1, batch_size, sem="relaxed")
    fence_sys()
    tl.debug_barrier()
    if count > 0:
        cta_l2_batch_finish(
            l2_done,
            combine_ready,
            sig_tab,
            MY_PE=MY_PE,
            NPES=NPES,
            total_tasks=active_l2_tasks,
            count=count,
        )
    wait_count(combine_ready, 1)
    fence_sys()
    tl.debug_barrier()
    if AGGREGATE_COMPLETION:
        task = worker_id
        while task < NTOK:
            cta_combine_one(
                topk_local,
                cb_tab,
                final_y,
                MY_PE=MY_PE,
                NTOK=NTOK,
                TOPK=TOPK,
                K=K,
                K_POW2=K_POW2,
                task_id=task,
            )
            task += ROLE_BUDGET
    else:
        task = tl.atomic_add(task_cursor + 2, 1, sem="relaxed")
        while task < NTOK:
            cta_combine_one(
                topk_local,
                cb_tab,
                final_y,
                MY_PE=MY_PE,
                NTOK=NTOK,
                TOPK=TOPK,
                K=K,
                K_POW2=K_POW2,
                task_id=task,
            )
            task = tl.atomic_add(task_cursor + 2, 1, sem="relaxed")

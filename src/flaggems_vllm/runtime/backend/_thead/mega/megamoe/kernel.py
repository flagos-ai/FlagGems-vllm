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
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language.extra.ppu import globaltimer
from triton.language.extra.cuda import libdevice


@triton.jit
def warp_stats(key):
    # Return the physical lane with the statistics, in the same tensor layout.
    return tl.inline_asm_elementwise(
        "ppu.activemask.b32 $3; "
        "ppu.match.any.sync.b32 $1, $4, $3; "
        "ppu.mov.u32 $2, %laneid; "
        "ppu.mov.u32 $0, %lanemask_lt; "
        "ppu.and.b32 $0, $0, $1; "
        "ppu.popc.b32 $0, $0; "
        "ppu.popc.b32 $1, $1;",
        constraints="=r,=r,=r,=&r,r",
        args=[key],
        dtype=(tl.int32, tl.int32, tl.int32, tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def address_index(value, WIDE_INDEX: tl.constexpr = True):
    if WIDE_INDEX:
        result = value.to(tl.int64)
    else:
        result = value
    return result


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
def load_acquire(ptr):
    sentinel = tl.full((), -1, ptr.dtype.element_ty)
    return tl.atomic_cas(ptr, sentinel, sentinel, sem="acquire", scope="gpu")


@triton.jit
def peer_barrier(sig_tab, slot, MY_PE: tl.constexpr, NPES: tl.constexpr):
    """One combine CTA per rank participates after all L2 peer stores finish."""
    tl.debug_barrier()
    for pe in tl.static_range(0, NPES):
        tl.atomic_add(peer_i32(sig_tab, pe) + slot, 1, sem="release", scope="sys")
    sl = peer_i32(sig_tab, MY_PE)
    v = tl.atomic_cas(sl + slot, -1, -1, sem="acquire", scope="sys")
    while v < NPES:
        v = tl.atomic_cas(sl + slot, -1, -1, sem="acquire", scope="sys")
    tl.debug_barrier()


@triton.jit
def wait_count(ptr, expected):
    """Wait for a role CTA group without requiring a whole-grid barrier."""
    v = load_acquire(ptr)
    while v < expected:
        v = load_acquire(ptr)


@triton.jit
def swiglu_bf16(gate, up, USE_FAST: tl.constexpr = False):
    if USE_FAST:
        # Gate is already BF16. Match div_rn at the one BF16 midpoint where
        # reciprocal division rounds differently on PPU80. The full BF16
        # input domain is checked in the numerical regression test.
        silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        silu = tl.where(gate == 5.9375, 5.90625, silu)
    else:
        silu = tl.div_rn(gate, 1.0 + tl.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    return (silu * up).to(tl.bfloat16)


@triton.jit
def quantize_token_int8(x, VENDOR_NUMERICS: tl.constexpr = False):
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-10)
    scale = amax / 127.0
    if VENDOR_NUMERICS:
        inverse_scale = tl.div_rn(1.0, scale)
        magnitude = tl.abs(x)
        normalized = magnitude * inverse_scale[:, None]
        # The bias covers reciprocal and multiply rounding over [0, 127],
        # so this candidate is at most one above the truncated result.
        upper = tl.floor(tl.minimum(normalized + 0.000030517578125, 127.0))
        bits = upper.to(tl.int32, bitcast=True)
        predecessor = tl.maximum(bits - 1, 0).to(tl.float32, bitcast=True)
        midpoint_delta = (upper - predecessor) * 0.5
        # At the midpoint, div_rn rounds toward the integer since its FP32
        # significand is even. FMA tests the boundary in the input domain.
        residual = tl.fma(-upper, scale[:, None], magnitude)
        below_midpoint = residual < -(midpoint_delta * scale[:, None])
        quantized = upper - below_midpoint.to(tl.float32)
        q = tl.where(x < 0.0, -quantized, quantized).to(tl.int8)
    else:
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
    L1_STAGES: tl.constexpr = 2,
    USE_FAST: tl.constexpr = False,
    QUANT_ROWS: tl.constexpr = 8,
    WIDE_INDEX: tl.constexpr = True,
    A_BLOCK_PTR: tl.constexpr = False,
    ASYNC_A: tl.constexpr = False,
    ASYNC_B: tl.constexpr = False,
    GATHER_INPUT: tl.constexpr = False,
    CACHE_TOKENS: tl.constexpr = False,
    NTOK=0,
    token_cache=None,
    tok_tab=None,
    MY_PE: tl.constexpr = 0,
    COMPACT_META: tl.constexpr = False,
    block_rows=None,
    ROW_COUNTER: tl.constexpr = False,
    VENDOR_NUMERICS: tl.constexpr = False,
):
    if COMPACT_META:
        pool_block = task_id // NL1N
        n_block = task_id % NL1N
        cur_e = tl.load(block_meta + pool_block, cache_modifier=".cg")
        valid_m = tl.load(block_rows + pool_block, cache_modifier=".cg")
    else:
        bid = task_id
        pool_block = tl.load(
            block_meta + address_index(bid, WIDE_INDEX) * 4 + 0, cache_modifier=".cg"
        )
        cur_e = tl.load(
            block_meta + address_index(bid, WIDE_INDEX) * 4 + 1, cache_modifier=".cg"
        )
        n_block = tl.load(
            block_meta + address_index(bid, WIDE_INDEX) * 4 + 2, cache_modifier=".cg"
        )
        valid_m = tl.load(
            block_meta + address_index(bid, WIDE_INDEX) * 4 + 3, cache_modifier=".cg"
        )
    if valid_m > 0:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_ok = offs_m < valid_m
        n_cols = n_block * BLOCK_N + offs_n
        n_ok = n_cols < L1_N
        rows = address_index(pool_block, WIDE_INDEX) * BLOCK_M + offs_m
        if GATHER_INPUT:
            src_row = tl.load(
                l1_token.to(tl.pointer_type(tl.int32)) + rows,
                mask=m_ok,
                other=0,
            )
            if CACHE_TOKENS:
                # EP2 caches the remote input locally. Dispatch supplies a
                # byte offset plus a bit selecting the remote input cache.
                local_ptr = peer_i8(tok_tab, MY_PE)
                src_ptr = tl.where((src_row & (1 << 30)) == 0, local_ptr, token_cache)
                src_base = src_ptr + address_index(src_row & ((1 << 30) - 1), WIDE_INDEX)
            else:
                src_pe = src_row // NTOK
                src_tok = src_row % NTOK
                src_ptr = tl.load(tok_tab + src_pe).to(tl.pointer_type(tl.int8))
                src_base = src_ptr + address_index(src_tok, WIDE_INDEX) * K
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for kb in tl.range(0, NK1, num_stages=L1_STAGES, loop_unroll_factor=1):
            if GATHER_INPUT:
                a_ptr = tl.max_contiguous(
                    tl.multiple_of(
                        src_base[:, None] + kb * BLOCK_K + offs_k[None, :], (1, 16)
                    ),
                    (1, 16),
                )
                if CACHE_TOKENS:
                    # Invalid lanes load local token 0. Rows are independent
                    # in the dot, and all invalid outputs are masked.
                    a = tl.load(a_ptr)
                else:
                    a = tl.load(a_ptr, mask=m_ok[:, None], other=0)
            elif A_BLOCK_PTR:
                ap = tl.make_block_ptr(
                    base=l1_token + address_index(pool_block, WIDE_INDEX) * BLOCK_M * K,
                    shape=(BLOCK_M, K),
                    strides=(K, 1),
                    offsets=(0, kb * BLOCK_K),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )
                if ASYNC_A:
                    a = tle.load(ap, is_async=True)
                else:
                    a = tl.load(ap)
            else:
                a = tl.load(
                    l1_token + rows[:, None] * K + (kb * BLOCK_K + offs_k)[None, :],
                    mask=m_ok[:, None],
                    other=0,
                )
            bp = tl.make_block_ptr(
                base=w1_kt + address_index(cur_e, WIDE_INDEX) * K * L1_N,
                shape=(K, L1_N),
                strides=(1, K),
                offsets=(kb * BLOCK_K, n_block * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            if ASYNC_B:
                b = tle.load(bp, is_async=True)
            else:
                b = tl.load(bp)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        sf = tl.load(l1_sf + rows, mask=m_ok, other=0.0)
        sfb = tl.load(
            w1_sf + address_index(cur_e, WIDE_INDEX) * L1_N + n_cols,
            mask=n_ok,
            other=0.0,
        )
        acc = acc.to(tl.float32) * sf[:, None] * sfb[None, :]
        if not VENDOR_NUMERICS:
            acc = acc.to(tl.bfloat16).to(tl.float32)
        t4 = tl.reshape(acc, (BLOCK_M, NPAIR, 2, 8))
        t4 = tl.permute(t4, (0, 1, 3, 2))
        gate, up = tl.split(t4)
        gate = tl.reshape(gate, (BLOCK_M, L1_OUT_N))
        up = tl.reshape(up, (BLOCK_M, L1_OUT_N))
        if VENDOR_NUMERICS:
            sw = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16)
        else:
            sw = swiglu_bf16(gate, up, USE_FAST)
        out_cols = n_block * L1_OUT_N + tl.arange(0, L1_OUT_N)
        tl.store(
            l2_bf16 + rows[:, None] * INTER + out_cols[None, :],
            sw.to(tl.bfloat16),
            mask=m_ok[:, None] & (out_cols[None, :] < INTER),
        )
        tl.debug_barrier()
    if ROW_COUNTER:
        # The existing l2_mask storage is cleared on every replay.
        counter = l2_arrival_mask.to(tl.pointer_type(tl.int32)) + pool_block
        previous = tl.atomic_add(counter, 1, sem="acq_rel")
        quantize = previous == NL1N - 1
    else:
        bit = tl.full((), 1, tl.int64) << n_block
        previous = tl.atomic_or(l2_arrival_mask + pool_block, bit, sem="acq_rel")
        quantize = (previous | bit) == (1 << NL1N) - 1
    if quantize:
        qc = tl.arange(0, triton.next_power_of_2(INTER))
        for qr in tl.range(0, valid_m, QUANT_ROWS):
            rr = qr + tl.arange(0, QUANT_ROWS)
            pr = address_index(pool_block, WIDE_INDEX) * BLOCK_M + rr
            x = tl.load(
                l2_bf16 + pr[:, None] * INTER + qc[None, :],
                mask=(rr[:, None] < valid_m) & (qc[None, :] < INTER),
                other=0,
            ).to(tl.float32)
            q, scale = quantize_token_int8(x, VENDOR_NUMERICS)
            tl.store(
                l2_acts + pr[:, None] * INTER + qc[None, :],
                q,
                mask=(rr[:, None] < valid_m) & (qc[None, :] < INTER),
            )
            tl.store(l2_sf + pr, scale, mask=rr < valid_m)
        tl.debug_barrier()
        if ROW_COUNTER:
            # N arrivals mean that quantization may start. Publish N+1
            # only after the INT8 activation and FP32 scale are visible.
            tl.atomic_xchg(counter, NL1N + 1, sem="release")


@triton.jit
def cta_l2_gemm(
    block_meta,
    local_contrib,
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
    USE_DST_PTR: tl.constexpr = False,
    L2_STAGES: tl.constexpr = 2,
    WIDE_INDEX: tl.constexpr = True,
    A_BLOCK_PTR: tl.constexpr = False,
    ASYNC_A: tl.constexpr = False,
    ASYNC_B: tl.constexpr = False,
    PARTIAL_REDUCE: tl.constexpr = False,
    DEFER_ROUTER: tl.constexpr = False,
    COMPACT_META: tl.constexpr = False,
    block_rows=None,
    COMPACT_N: tl.constexpr = 1,
    CONTRIB_STRIPES: tl.constexpr = False,
    DST_PTR64: tl.constexpr = False,
):
    if COMPACT_META:
        pool_block = task_id // COMPACT_N
        n_block = task_id % COMPACT_N
        cur_e = tl.load(block_meta + pool_block, cache_modifier=".cg")
        valid_m = tl.load(block_rows + pool_block, cache_modifier=".cg")
    else:
        bid = task_id
        pool_block = tl.load(
            block_meta + address_index(bid, WIDE_INDEX) * 4 + 0, cache_modifier=".cg"
        )
        cur_e = tl.load(
            block_meta + address_index(bid, WIDE_INDEX) * 4 + 1, cache_modifier=".cg"
        )
        n_block = tl.load(
            block_meta + address_index(bid, WIDE_INDEX) * 4 + 2, cache_modifier=".cg"
        )
        valid_m = tl.load(
            block_meta + address_index(bid, WIDE_INDEX) * 4 + 3, cache_modifier=".cg"
        )
    if valid_m > 0:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        m_ok = offs_m < valid_m
        rows = address_index(pool_block, WIDE_INDEX) * BLOCK_M + offs_m
        cols = n_block * BLOCK_N + offs_n
        n_ok = cols < L2_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        kk = tl.arange(0, BLOCK_K)
        for kb in tl.range(0, NK2, num_stages=L2_STAGES):
            if A_BLOCK_PTR:
                ap = tl.make_block_ptr(
                    base=l2_acts
                    + address_index(pool_block, WIDE_INDEX) * BLOCK_M * INTER,
                    shape=(BLOCK_M, INTER),
                    strides=(INTER, 1),
                    offsets=(0, kb * BLOCK_K),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )
                if ASYNC_A:
                    a = tle.load(ap, is_async=True)
                else:
                    a = tl.load(ap)
            else:
                a = tl.load(
                    l2_acts + rows[:, None] * INTER + (kb * BLOCK_K + kk)[None, :],
                    mask=m_ok[:, None],
                    other=0,
                )
            bp = tl.make_block_ptr(
                base=w2_kt + address_index(cur_e, WIDE_INDEX) * INTER * L2_N,
                shape=(INTER, L2_N),
                strides=(1, INTER),
                offsets=(kb * BLOCK_K, n_block * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            if ASYNC_B:
                b = tle.load(bp, is_async=True)
            else:
                b = tl.load(bp)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        sf = tl.load(l2_sf + rows, mask=m_ok, other=0.0)
        sfb = tl.load(
            w2_sf + address_index(cur_e, WIDE_INDEX) * L2_N + cols, mask=n_ok, other=0.0
        )
        if DEFER_ROUTER:
            # DeepGemm writes unweighted BF16 GEMM2, then ep_gather applies
            # FP32 router weights before reducing local experts.
            acc = acc.to(tl.float32) * sf[:, None] * sfb[None, :]
        else:
            router = tl.load(l1_w + rows, mask=m_ok, other=0.0)
            acc = acc.to(tl.float32) * sf[:, None] * sfb[None, :] * router[:, None]
        if PARTIAL_REDUCE:
            if CONTRIB_STRIPES:
                # Place each group of 128 columns in its own contiguous slab
                # within an expert row block, so the output tile streams.
                tl.static_assert(BLOCK_N == 128)
                out = (address_index(pool_block, WIDE_INDEX) * K + n_block * 128) * BLOCK_M
                out = out + offs_m[:, None] * 128 + offs_n[None, :]
            else:
                out = rows[:, None] * K + cols[None, :]
            tl.store(
                tl.max_contiguous(
                    tl.multiple_of(
                        local_contrib + out, (1, 8)
                    ),
                    (1, 8),
                ),
                acc.to(tl.bfloat16),
                mask=m_ok[:, None] & n_ok[None, :],
            )
        else:
            meta_l = peer_i32(meta_tab, MY_PE)
            md = meta_l + rows * 3
            if USE_DST_PTR:
                if DST_PTR64 and (meta_l.to(tl.uint64) & 7) == 0:
                    addr = tl.load(meta_l.to(tl.pointer_type(tl.uint64)) + rows, mask=m_ok, other=0)
                else:
                    ptr_lo = tl.load(md + 0, mask=m_ok, other=0).to(tl.uint32).to(tl.uint64)
                    ptr_hi = tl.load(md + 1, mask=m_ok, other=0).to(tl.uint32).to(tl.uint64)
                    addr = ptr_lo | (ptr_hi << 32)
                cbb = addr.to(tl.pointer_type(tl.bfloat16))
                base = tl.full((BLOCK_M,), 0, tl.int32)
            else:
                dst_rank = tl.load(md + 0, mask=m_ok, other=0)
                dst_tok = tl.load(md + 1, mask=m_ok, other=0)
                dst_topk = tl.load(md + 2, mask=m_ok, other=0)
                cbb = peer_bf16(cb_tab, dst_rank)
                base = (address_index(dst_topk, WIDE_INDEX) * NTOK + dst_tok) * K
            tl.store(
                tl.max_contiguous(
                    tl.multiple_of(
                        cbb[:, None] + base[:, None] + cols[None, :], (1, 8)
                    ),
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
        tl.atomic_xchg(combine_ready, 1, sem="release")


@triton.jit
def fold_rank_partials(
    cb_tab, final_y, task, NTOK, K: tl.constexpr, ROWS: tl.constexpr,
    MY_PE: tl.constexpr, WIDE_INDEX: tl.constexpr, RETURN_PULL: tl.constexpr,
):
    tok = task + tl.arange(0, ROWS)
    h = tl.arange(0, K)
    values = tl.zeros((ROWS, K), tl.float32)
    for rank in tl.static_range(0, 2):
        if RETURN_PULL:
            source = peer_bf16(cb_tab, rank)
            start = MY_PE * address_index(NTOK, WIDE_INDEX)
        else:
            source = peer_bf16(cb_tab, MY_PE)
            start = rank * address_index(NTOK, WIDE_INDEX)
        ptrs = source + (start + tok[:, None]) * K + h[None, :]
        if (source.to(tl.uint64) & 15) == 0:
            partial = tl.load(tl.max_contiguous(tl.multiple_of(ptrs, (1, 8)), (1, 8)),
                              mask=tok[:, None] < NTOK, other=0)
        else:
            partial = tl.load(ptrs, mask=tok[:, None] < NTOK, other=0)
        values += partial.to(tl.float32)
    tl.store(final_y + address_index(tok, WIDE_INDEX)[:, None] * K + h[None, :],
             values.to(tl.bfloat16), mask=tok[:, None] < NTOK)


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
    ROWS: tl.constexpr = 1,
    WIDE_INDEX: tl.constexpr = True,
    PARTIAL_REDUCE: tl.constexpr = False,
    router_tab=None,
    DEFER_ROUTER: tl.constexpr = False,
):
    """One dynamic combine task for one source token."""
    if ROWS > 1:
        tl.static_assert(DEFER_ROUTER and not PARTIAL_REDUCE)
        cb = peer_bf16(cb_tab, MY_PE)
        weights = peer_f32(router_tab, MY_PE)
        tokens = task_id + tl.arange(0, ROWS)
        valid = tokens < NTOK
        h = tl.arange(0, K_POW2)
        row = address_index(tokens, WIDE_INDEX)
        acc = tl.zeros((ROWS, K_POW2), tl.float32)
        for slot in tl.static_range(0, TOPK):
            route = tl.load(topk_local + row * TOPK + slot, mask=valid, other=-1)
            weight = tl.load(weights + row * TOPK + slot, mask=valid, other=0.0)
            p = cb + (slot * address_index(NTOK, WIDE_INDEX) + row[:, None]) * K + h[None, :]
            p = tl.max_contiguous(tl.multiple_of(p, (1, 8)), (1, 8))
            value = tl.load(p, mask=valid[:, None] & (route[:, None] >= 0), other=0)
            acc += value.to(tl.float32) * weight[:, None]
        tl.store(final_y + row[:, None] * K + h[None, :], acc.to(tl.bfloat16), mask=valid[:, None])
    elif PARTIAL_REDUCE:
        cb_l = peer_bf16(cb_tab, MY_PE)
        h = tl.arange(0, K_POW2)
        acc = tl.zeros((K_POW2,), tl.float32)
        for pe in tl.static_range(0, 2):
            ptr = cb_l + (pe * address_index(NTOK, WIDE_INDEX) + task_id) * K
            if (ptr.to(tl.uint64) & 15) == 0:
                value = tl.load(
                    tl.max_contiguous(tl.multiple_of(ptr + h, 8), 8),
                    mask=(task_id < NTOK) & (h < K),
                    other=0,
                )
            else:
                value = tl.load(ptr + h, mask=(task_id < NTOK) & (h < K), other=0)
            acc += value.to(tl.float32)
        tl.store(
            final_y + address_index(task_id, WIDE_INDEX) * K + h,
            acc.to(tl.bfloat16),
            mask=(task_id < NTOK) & (h < K),
        )
    else:
        cb_l = peer_bf16(cb_tab, MY_PE)
        offs_hid = tl.arange(0, K_POW2)
        hid_ok = offs_hid < K
        t_ok = task_id < NTOK
        acc = tl.zeros((K_POW2,), dtype=tl.float32)
        for k in tl.static_range(0, TOPK):
            e = tl.load(
                topk_local + address_index(task_id, WIDE_INDEX) * TOPK + k,
                mask=t_ok,
                other=-1,
            ).to(tl.int32)
            v = tl.load(
                tl.max_contiguous(
                    tl.multiple_of(
                        cb_l
                        + (k * address_index(NTOK, WIDE_INDEX) + task_id) * K
                        + offs_hid,
                        8,
                    ),
                    8,
                ),
                mask=(e >= 0) & t_ok & hid_ok,
                other=0,
            )
            if DEFER_ROUTER:
                router = tl.load(
                    peer_f32(router_tab, MY_PE)
                    + address_index(task_id, WIDE_INDEX) * TOPK
                    + k
                )
                acc += v.to(tl.float32) * router
            else:
                acc += v.to(tl.float32)
        tl.store(
            final_y + address_index(task_id, WIDE_INDEX) * K + offs_hid,
            acc.to(tl.bfloat16),
            mask=t_ok & hid_ok,
        )


@triton.jit
def dispatch_payload(buffers, config: tl.constexpr, WIDE_INDEX: tl.constexpr = True):
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
        cb_tab,
        token_cache,
        profile_times,
        copy_row_map,
    ) = buffers
    MY_PE: tl.constexpr = config[0]
    NPES: tl.constexpr = config[1]
    TOPK: tl.constexpr = config[2]
    EPR: tl.constexpr = config[3]
    K: tl.constexpr = config[4]
    NSF: tl.constexpr = config[5]
    BLOCK_M: tl.constexpr = config[6]
    BLOCK_K: tl.constexpr = config[7]
    EPR_POW2: tl.constexpr = config[9]
    NL1N: tl.constexpr = config[10]
    NL2N: tl.constexpr = config[11]
    ROLE_BUDGET: tl.constexpr = config[12]
    ROUTE_CHUNK: tl.constexpr = config[13]
    ROUTE_CHUNKS: tl.constexpr = config[14]
    PACK_ROUTE_META: tl.constexpr = config[15]
    SKIP_EMPTY_BLOCKS: tl.constexpr = config[16]
    META_BLOCK: tl.constexpr = config[17]
    USE_DST_PTR: tl.constexpr = config[18]
    PARTIAL_REDUCE: tl.constexpr = config[19]
    CACHE_TOKENS: tl.constexpr = config[20]
    WARP_HIST: tl.constexpr = config[21]
    ATOMIC_DISPATCH: tl.constexpr = config[22]
    COPY_K: tl.constexpr = config[23] or BLOCK_K
    GATHER_INPUT: tl.constexpr = config[24]
    STATIC_COUNT: tl.constexpr = config[25] != 0
    COUNT_CHUNK: tl.constexpr = config[25] or ROUTE_CHUNK
    STATIC_PAYLOAD: tl.constexpr = config[26]
    COMPACT_META: tl.constexpr = config[27]
    LOCAL_INVERSE: tl.constexpr = config[28]
    BULK_DISPATCH_READY: tl.constexpr = config[29]
    STATIC_DISPATCH_META: tl.constexpr = config[30]
    RANK_RETURN: tl.constexpr = config[31]
    PAYLOAD_WORKERS: tl.constexpr = config[32] or ROLE_BUDGET
    DISPATCH_OVERLAP: tl.constexpr = config[33]
    DISPATCH_TRACE: tl.constexpr = config[34]
    CACHE_DECOUPLED: tl.constexpr = config[35]
    CACHE_WORKERS: tl.constexpr = config[36]
    COPY_WORKERS: tl.constexpr = config[37]
    COPY_M_BLOCK: tl.constexpr = config[38]
    DEFER_ROUTER: tl.constexpr = config[39]
    DST_PTR64: tl.constexpr = config[40]
    if CACHE_DECOUPLED:
        tl.static_assert(BULK_DISPATCH_READY and RANK_RETURN and ATOMIC_DISPATCH)
        plan_workers: tl.constexpr = ROLE_BUDGET - CACHE_WORKERS
        plan_id = worker_id - CACHE_WORKERS
    else:
        plan_workers: tl.constexpr = ROLE_BUDGET
        plan_id = worker_id
    # The compact GEMM2 path leaves l2_meta free. The queue is written
    # exactly once per row block on each replay. A worker may claim only
    # published tickets, so stale entries cannot be observed.
    if DISPATCH_OVERLAP:
        ready_count = l1_ready + 2
        ready_claim = l1_ready + 3
        ready_slots = l2_meta
        init = worker_id * 128
        while init < POOL_TOKENS // BLOCK_M:
            idx = init + tl.arange(0, 128)
            tl.store(ready_slots + idx, 0, mask=idx < POOL_TOKENS // BLOCK_M)
            init += ROLE_BUDGET * 128
        tl.debug_barrier()
    if DISPATCH_TRACE:
        trace = profile_times + worker_id * 16
        tl.store(trace + 8, globaltimer())
    BLOCK_READY: tl.constexpr = COMPACT_META
    meta_l = peer_i32(meta_tab, MY_PE)
    if STATIC_COUNT:
        count_chunks = (NTOK * TOPK + COUNT_CHUNK - 1) // COUNT_CHUNK
        work_id = plan_id
    else:
        count_chunks = ROUTE_CHUNKS
        work_id = tl.atomic_add(dispatch_cursor + 0, 1, sem="relaxed")
    while work_id < NPES * count_chunks:
        src_id = work_id // count_chunks
        base = work_id % count_chunks * COUNT_CHUNK
        routes_src = peer_i32(route_tab, src_id)
        lane = tl.arange(0, COUNT_CHUNK)
        rid = base + lane
        valid = rid < NTOK * TOPK
        if LOCAL_INVERSE:
            tl.store(meta_l + address_index(src_id, WIDE_INDEX) * NTOK * TOPK + rid,
                     -1, mask=valid)
        expert = tl.load(routes_src + rid, mask=valid, other=-1)
        owner = expert // EPR
        local_expert = expert - MY_PE * EPR
        take = valid & (expert >= 0) & (owner == MY_PE)
        safe_local_expert = tl.where(take, local_expert, 0)
        chunk_id = work_id % ROUTE_CHUNKS
        if ATOMIC_DISPATCH:
            experts = tl.arange(0, EPR_POW2)
            chunk_hist = tl.histogram(safe_local_expert, EPR_POW2, mask=take)
            tl.atomic_add(
                dispatch_counts + address_index(experts, WIDE_INDEX) * NPES + src_id,
                chunk_hist,
                sem="relaxed",
                mask=(experts < EPR) & (chunk_hist > 0),
            )
        else:
            if WARP_HIST and PACK_ROUTE_META:
                intra, group_count, physical_lane, active_mask = warp_stats(expert)
                layout_ok = (
                    tl.sum(
                        (
                            (physical_lane != tl.arange(0, ROUTE_CHUNK))
                            | (active_mask != 0xFFFFFFFF)
                        ).to(tl.int32),
                        0,
                    )
                    == 0
                )
                if layout_ok:
                    tl.static_assert(ROUTE_CHUNK == 32)
                    bins = tl.arange(0, EPR_POW2)
                    tl.store(
                        dispatch_chunk_counts
                        + (address_index(src_id, WIDE_INDEX) * ROUTE_CHUNKS + chunk_id)
                        * EPR
                        + bins,
                        tl.full((EPR_POW2,), 0, tl.int32),
                        mask=bins < EPR,
                    )
                    tl.debug_barrier()
                    leader = take & (intra == 0)
                    # Initialize every bin in main, preserving the original reset contract.
                    tl.store(
                        dispatch_chunk_counts
                        + (address_index(src_id, WIDE_INDEX) * ROUTE_CHUNKS + chunk_id)
                        * EPR
                        + safe_local_expert,
                        group_count,
                        mask=leader,
                    )
                    tl.atomic_add(
                        dispatch_counts
                        + address_index(safe_local_expert, WIDE_INDEX) * NPES
                        + src_id,
                        group_count,
                        sem="release",
                        mask=leader,
                    )
                else:
                    experts = tl.arange(0, EPR_POW2)
                    if PACK_ROUTE_META:
                        chunk_hist = tl.histogram(
                            safe_local_expert, EPR_POW2, mask=take
                        )
                    else:
                        chunk_hist = tl.sum(
                            (
                                (safe_local_expert[:, None] == experts[None, :])
                                & take[:, None]
                            ).to(tl.int32),
                            axis=0,
                        )
                    tl.store(
                        dispatch_chunk_counts
                        + (address_index(src_id, WIDE_INDEX) * ROUTE_CHUNKS + chunk_id)
                        * EPR
                        + experts,
                        chunk_hist,
                        mask=experts < EPR,
                    )
                    tl.atomic_add(
                        dispatch_counts
                        + address_index(experts, WIDE_INDEX) * NPES
                        + src_id,
                        chunk_hist,
                        sem="release",
                        mask=(experts < EPR) & (chunk_hist > 0),
                    )
                    intra = tl.sum(
                        (
                            (lane[:, None] > lane[None, :])
                            & (expert[:, None] == expert[None, :])
                            & take[None, :]
                        ).to(tl.int32),
                        axis=1,
                    )
            else:
                experts = tl.arange(0, EPR_POW2)
                if PACK_ROUTE_META:
                    chunk_hist = tl.histogram(safe_local_expert, EPR_POW2, mask=take)
                else:
                    chunk_hist = tl.sum(
                        (
                            (safe_local_expert[:, None] == experts[None, :])
                            & take[:, None]
                        ).to(tl.int32),
                        axis=0,
                    )
                tl.store(
                    dispatch_chunk_counts
                    + (address_index(src_id, WIDE_INDEX) * ROUTE_CHUNKS + chunk_id)
                    * EPR
                    + experts,
                    chunk_hist,
                    mask=experts < EPR,
                )
                tl.atomic_add(
                    dispatch_counts
                    + address_index(experts, WIDE_INDEX) * NPES
                    + src_id,
                    chunk_hist,
                    sem="release",
                    mask=(experts < EPR) & (chunk_hist > 0),
                )
        nactive = tl.sum(take.to(tl.int32))
        if nactive > 0:
            active_base = tl.atomic_add(active_count + src_id, nactive, sem="relaxed")
            lane_ord = tl.cumsum(take.to(tl.int32)) - 1
            if ATOMIC_DISPATCH:
                tl.store(
                    active_routes
                    + address_index(src_id, WIDE_INDEX) * NTOK * TOPK
                    + active_base
                    + lane_ord,
                    rid | safe_local_expert << 19,
                    mask=take,
                )
            elif PACK_ROUTE_META:
                if not WARP_HIST:
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
                    active_routes
                    + address_index(src_id, WIDE_INDEX)
                    * (address_index(NTOK, WIDE_INDEX) * TOPK)
                    + active_base
                    + lane_ord,
                    packed,
                    mask=take,
                )
            else:
                tl.store(
                    active_routes
                    + address_index(src_id, WIDE_INDEX)
                    * (address_index(NTOK, WIDE_INDEX) * TOPK)
                    + active_base
                    + lane_ord,
                    rid,
                    mask=take,
                )
        if STATIC_COUNT:
            work_id += plan_workers
        else:
            work_id = tl.atomic_add(dispatch_cursor + 0, 1, sem="relaxed")
    tl.debug_barrier()
    tl.atomic_add(dispatch_phase + 0, 1, sem="acq_rel" if CACHE_TOKENS else "release")
    wait_count(dispatch_phase + 0, plan_workers)
    if plan_id == 0:
        if RANK_RETURN:
            tl.static_assert(ATOMIC_DISPATCH and COMPACT_META and BULK_DISPATCH_READY)
            experts = tl.arange(0, EPR_POW2)
            l_count = tl.load(dispatch_counts + NPES * experts + MY_PE, mask=experts < EPR, other=0)
            r_count = tl.full((EPR_POW2,), 0, tl.int32)
            for dst in tl.static_range(0, NPES):
                if dst != MY_PE:
                    r_count += tl.load(dispatch_counts + NPES * experts + dst, mask=experts < EPR, other=0)
            tl.store(dispatch_chunk_offsets + 3 * EPR + experts, r_count, mask=experts < EPR)
            r_blocks = tl.cdiv(r_count, BLOCK_M)
            l_blocks = tl.cdiv(r_count + l_count, BLOCK_M) - r_blocks
            r_end = tl.sum(r_blocks)
            tl.store(block_base_tab + experts, tl.cumsum(r_blocks) - r_blocks, mask=experts < EPR)
            # Reserve the offsets workspace after its two per-expert cursors.
            tl.store(dispatch_chunk_offsets + 2 * EPR + experts,
                     r_end + tl.cumsum(l_blocks) - l_blocks, mask=experts < EPR)
            tl.store(dispatch_chunk_offsets + experts, 0, mask=experts < EPR)
            tl.store(dispatch_chunk_offsets + EPR + experts, 0, mask=experts < EPR)
            tl.store(l1_ready, r_end)
            tl.store(dispatch_phase + 2, r_end + tl.sum(l_blocks))
        else:
            experts = tl.arange(0, EPR_POW2)
            expert_ok = experts < EPR
            total = tl.zeros((EPR_POW2,), dtype=tl.int32)
            for src in tl.static_range(0, NPES):
                c = tl.load(
                    dispatch_counts + address_index(experts, WIDE_INDEX) * NPES + src,
                    mask=expert_ok,
                    other=0,
                )
                total += c
            # Rebuild routing-dependent offsets on device for every invocation.
            if SKIP_EMPTY_BLOCKS:
                block_counts = tl.where(expert_ok, tl.cdiv(total, BLOCK_M), 0)
            else:
                block_counts = tl.where(
                    expert_ok, tl.maximum(tl.cdiv(total, BLOCK_M), 1), 0
                )
            block_bases = tl.cumsum(block_counts) - block_counts
            tl.store(block_base_tab + experts, block_bases, mask=expert_ok)
            # The phase-1 release publishes these cursors before any CTA
            # reserves a row.
            if ATOMIC_DISPATCH:
                tl.store(dispatch_chunk_offsets + experts, 0, mask=expert_ok)
            tl.store(dispatch_phase + 2, tl.sum(block_counts))
        tl.debug_barrier()
        tl.atomic_xchg(dispatch_phase + 1, 1, sem="release")
    if not ATOMIC_DISPATCH and ROUTE_CHUNKS <= 1024:
        chunks = tl.arange(0, triton.next_power_of_2(ROUTE_CHUNKS))
        subgroup = tl.arange(0, 4)
        expert_groups = triton.cdiv(EPR, 4)
        prefix_group = worker_id
        while prefix_group < NPES * expert_groups:
            src = prefix_group // expert_groups
            le = prefix_group % expert_groups * 4 + subgroup
            counts = tl.load(
                dispatch_chunk_counts
                + (address_index(src, WIDE_INDEX) * ROUTE_CHUNKS + chunks[:, None])
                * EPR
                + le[None, :],
                mask=(chunks[:, None] < ROUTE_CHUNKS) & (le[None, :] < EPR),
                other=0,
                cache_modifier=".cg",
            )
            offsets = tl.cumsum(counts, axis=0) - counts
            tl.store(
                dispatch_chunk_offsets
                + (address_index(src, WIDE_INDEX) * ROUTE_CHUNKS + chunks[:, None])
                * EPR
                + le[None, :],
                offsets,
                mask=(chunks[:, None] < ROUTE_CHUNKS) & (le[None, :] < EPR),
            )
            prefix_group += plan_workers
        tl.debug_barrier()
    if not ATOMIC_DISPATCH and ROUTE_CHUNKS > 1024:
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
                    + (
                        address_index(src, WIDE_INDEX) * ROUTE_CHUNKS
                        + chunk_ids[:, None]
                    )
                    * EPR
                    + le[None, :],
                    mask=(chunk_ids[:, None] < ROUTE_CHUNKS) & (le[None, :] < EPR),
                    other=0,
                    cache_modifier=".cg",
                )
                offsets = tl.cumsum(counts, axis=0) - counts + running[None, :]
                tl.store(
                    dispatch_chunk_offsets
                    + (
                        address_index(src, WIDE_INDEX) * ROUTE_CHUNKS
                        + chunk_ids[:, None]
                    )
                    * EPR
                    + le[None, :],
                    offsets,
                    mask=(chunk_ids[:, None] < ROUTE_CHUNKS) & (le[None, :] < EPR),
                )
                running += tl.sum(counts, axis=0)
            prefix_group += plan_workers
        tl.debug_barrier()
    wait_count(dispatch_phase + 1, 1)
    active_blocks = tl.load(dispatch_phase + 2)
    expert_ids = tl.arange(0, EPR_POW2)
    expert_bases = tl.load(block_base_tab + expert_ids, mask=expert_ids < EPR, other=0)
    if RANK_RETURN:
        local_bases = tl.load(dispatch_chunk_offsets + 2 * EPR + expert_ids, mask=expert_ids < EPR, other=0)
        remote_blocks = tl.load(l1_ready)
    if STATIC_DISPATCH_META:
        meta_id = plan_id * META_BLOCK
    else:
        meta_id = tl.atomic_add(dispatch_cursor + 2, META_BLOCK, sem="relaxed")
    while meta_id < active_blocks:
        blocks = meta_id + tl.arange(0, META_BLOCK)
        block_ok = blocks < active_blocks
        safe_blocks = tl.where(block_ok, blocks, 0)
        if RANK_RETURN:
            local = safe_blocks >= remote_blocks
            bases = tl.where(local[:, None], local_bases[None, :], expert_bases[None, :])
            block_expert = tl.sum(((safe_blocks[:, None] >= bases) &
                                   (expert_ids[None, :] < EPR)).to(tl.int32), 1) - 1
            safe_expert = tl.where(block_ok, block_expert, 0)
            expert_base = tl.load(block_base_tab + safe_expert, mask=block_ok & ~local, other=0)
            local_base = tl.load(dispatch_chunk_offsets + 2 * EPR + safe_expert, mask=block_ok & local, other=0)
            block_in_expert = safe_blocks - tl.where(local, local_base, expert_base)
            remote_rows = tl.load(dispatch_chunk_offsets + 3 * EPR + safe_expert, mask=block_ok, other=0)
            local_rows = tl.load(dispatch_counts + NPES * safe_expert + MY_PE, mask=block_ok, other=0)
            block_in_expert += tl.where(local, tl.cdiv(remote_rows, BLOCK_M), 0)
            block_total = remote_rows + local_rows
        else:
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
            block_total = tl.zeros((META_BLOCK,), dtype=tl.int32)
            for src in tl.static_range(0, NPES):
                block_total += tl.load(
                    dispatch_counts + address_index(block_expert, WIDE_INDEX) * NPES + src,
                    mask=block_ok,
                    other=0,
                )
        valid_m = tl.minimum(
            tl.maximum(block_total - block_in_expert * BLOCK_M, 0), BLOCK_M
        )
        tl.store(block_rows + safe_blocks, valid_m, mask=block_ok)
        if COMPACT_META:
            tl.store(l1_meta + safe_blocks, block_expert, mask=block_ok)
        else:
            for n_block in tl.static_range(0, NL1N):
                dst = (
                    l1_meta
                    + (address_index(safe_blocks, WIDE_INDEX) * NL1N + n_block) * 4
                )
                tl.store(dst + 0, safe_blocks, mask=block_ok)
                tl.store(dst + 1, block_expert, mask=block_ok)
                tl.store(dst + 2, n_block, mask=block_ok)
                tl.store(dst + 3, valid_m, mask=block_ok)
            for n_block in tl.static_range(0, NL2N):
                dst = (
                    l2_meta
                    + (address_index(safe_blocks, WIDE_INDEX) * NL2N + n_block) * 4
                )
                tl.store(dst + 0, safe_blocks, mask=block_ok)
                tl.store(dst + 1, block_expert, mask=block_ok)
                tl.store(dst + 2, n_block, mask=block_ok)
                tl.store(dst + 3, valid_m, mask=block_ok)
        if not BULK_DISPATCH_READY and not COPY_M_BLOCK:
            empty = block_ok & (valid_m == 0)
            if BLOCK_READY:
                tl.atomic_xchg(l1_ready + safe_blocks, 1, sem="release", mask=empty)
            else:
                for n_block in tl.static_range(0, NL1N):
                    tl.atomic_xchg(
                        l1_ready + address_index(safe_blocks, WIDE_INDEX) * NL1N + n_block,
                        1,
                        sem="release",
                        mask=empty,
                    )
        if STATIC_DISPATCH_META:
            meta_id += plan_workers * META_BLOCK
        else:
            meta_id = tl.atomic_add(dispatch_cursor + 2, META_BLOCK, sem="relaxed")
    tl.debug_barrier()
    tl.atomic_add(dispatch_phase + 3, 1, sem="acq_rel" if (CACHE_DECOUPLED or COPY_M_BLOCK) else "release")
    wait_count(dispatch_phase + 3, plan_workers)
    tl.debug_barrier()
    wait_count(dispatch_phase + 1, 1)
    source_ids = tl.arange(0, triton.next_power_of_2(NPES))
    source_routes = tl.load(active_count + source_ids, mask=source_ids < NPES, other=0)
    source_chunks = (source_routes + ROUTE_CHUNK - 1) // ROUTE_CHUNK
    chunk_ends = tl.cumsum(source_chunks)
    pass2_chunks = tl.sum(source_chunks)
    if DISPATCH_TRACE:
        tl.store(trace + 9, globaltimer())
    if COPY_M_BLOCK:
        work_id = worker_id
    elif COPY_WORKERS:
        work_id = tl.where(worker_id < COPY_WORKERS, worker_id, pass2_chunks)
    elif worker_id < PAYLOAD_WORKERS:
        if STATIC_PAYLOAD:
            work_id = plan_id
        else:
            work_id = tl.atomic_add(dispatch_cursor + 1, 1, sem="relaxed")
    else:
        work_id = pass2_chunks
    while work_id < pass2_chunks:
        src_id = tl.sum(((work_id >= chunk_ends) & (source_ids < NPES)).to(tl.int32))
        source_begin = tl.sum(tl.where(source_ids < src_id, source_chunks, 0))
        base = (work_id - source_begin) * ROUTE_CHUNK
        active_total = tl.load(active_count + src_id)
        lane = tl.arange(0, ROUTE_CHUNK)
        active_id = base + lane
        active = active_id < active_total
        if ATOMIC_DISPATCH:
            packed = tl.load(
                active_routes
                + address_index(src_id, WIDE_INDEX) * NTOK * TOPK
                + active_id,
                mask=active,
                other=0,
            )
            rid = packed & ((1 << 19) - 1)
            safe_expert = packed >> 19
            take = active
            if CACHE_TOKENS:
                src_tok_ptr = token_cache + address_index(src_id, WIDE_INDEX) * NTOK * K
            else:
                src_tok_ptr = peer_i8(tok_tab, src_id)
            src_sf_ptr = peer_f32(sf_tab, src_id)
            src_w_ptr = peer_f32(w_tab, src_id)
        elif PACK_ROUTE_META:
            packed = tl.load(
                active_routes
                + address_index(src_id, WIDE_INDEX)
                * (address_index(NTOK, WIDE_INDEX) * TOPK)
                + active_id,
                mask=active,
                other=0,
            )
            rid = packed & (1 << 19) - 1
            intra = packed >> 19 & 31
            safe_expert = packed >> 24
            take = active
            if CACHE_TOKENS:
                src_tok_ptr = token_cache + address_index(src_id, WIDE_INDEX) * NTOK * K
            else:
                src_tok_ptr = peer_i8(tok_tab, src_id)
            src_sf_ptr = peer_f32(sf_tab, src_id)
            src_w_ptr = peer_f32(w_tab, src_id)
            chunk_id = rid // ROUTE_CHUNK
        else:
            rid = tl.load(
                active_routes
                + address_index(src_id, WIDE_INDEX)
                * (address_index(NTOK, WIDE_INDEX) * TOPK)
                + active_id,
                mask=active,
                other=0,
            )
            routes_src = peer_i32(route_tab, src_id)
            if CACHE_TOKENS:
                src_tok_ptr = token_cache + address_index(src_id, WIDE_INDEX) * NTOK * K
            else:
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
        if RANK_RETURN:
            local = (src_id == MY_PE).to(tl.int32)
            global_ord = tl.atomic_add(dispatch_chunk_offsets + local * EPR + safe_expert, 1,
                                       sem="relaxed", mask=take)
            remote_rows = tl.load(dispatch_chunk_offsets + 3 * EPR + safe_expert, mask=take, other=0)
            global_ord += tl.where(local != 0, remote_rows, 0)
            remote_padded = tl.cdiv(remote_rows, BLOCK_M) * BLOCK_M
            local_only = global_ord >= remote_padded
            global_ord -= tl.where(local_only, remote_padded, 0)
            base_ptr = tl.where(local_only, dispatch_chunk_offsets + 2 * EPR, block_base_tab)
        elif ATOMIC_DISPATCH:
            global_ord = tl.atomic_add(
                dispatch_chunk_offsets + safe_expert,
                1,
                sem="relaxed",
                mask=take,
            )
        else:
            chunk_off = tl.load(
                dispatch_chunk_offsets
                + (address_index(src_id, WIDE_INDEX) * ROUTE_CHUNKS + chunk_id) * EPR
                + safe_expert,
                mask=take,
                other=0,
            )
            local_ord = chunk_off + intra
            global_base = tl.zeros_like(local_ord)
            for prev_src in tl.static_range(0, NPES):
                prev_count = tl.load(
                    dispatch_counts
                    + address_index(safe_expert, WIDE_INDEX) * NPES
                    + prev_src,
                    mask=take,
                    other=0,
                )
                global_base += prev_count * (prev_src < src_id).to(tl.int32)
            global_ord = global_base + local_ord
        if not RANK_RETURN:
            base_ptr = block_base_tab
        pool_row = (
            address_index(
                tl.load(base_ptr + safe_expert, mask=take, other=0), WIDE_INDEX
            )
            * BLOCK_M
            + global_ord
        )
        if COPY_M_BLOCK:
            tl.store(copy_row_map + pool_row, (src_id << 19) | rid, mask=take)
        else:
            src_tok = rid // TOPK
            src_topk = rid % TOPK
            if GATHER_INPUT:
                if CACHE_TOKENS:
                    cache_rank = src_id - (src_id > MY_PE).to(tl.int32)
                    offset = src_tok + tl.where(src_id != MY_PE, cache_rank * NTOK, 0)
                    src_index = offset * K | ((src_id != MY_PE).to(tl.int32) << 30)
                else:
                    src_index = src_id * NTOK + src_tok
                tl.store(
                    l1_token.to(tl.pointer_type(tl.int32)) + pool_row,
                    src_index,
                    mask=take,
                )
            else:
                for kb in tl.range(0, K // COPY_K, num_stages=2, loop_unroll_factor=1):
                    offs_k = tl.arange(0, COPY_K)
                    x = tl.load(
                        tl.max_contiguous(
                            tl.multiple_of(
                                src_tok_ptr
                                + address_index(src_tok, WIDE_INDEX)[:, None] * K
                                + kb * COPY_K
                                + offs_k[None, :],
                                (1, 16),
                            ),
                            (1, 16),
                        ),
                        mask=take[:, None],
                        other=0,
                    )
                    tl.store(
                        l1_token + pool_row[:, None] * K + kb * COPY_K + offs_k[None, :],
                        x,
                        mask=take[:, None],
                    )
            for sf_block in tl.static_range(0, NSF):
                sf = tl.load(
                    src_sf_ptr + address_index(src_tok, WIDE_INDEX) * NSF + sf_block,
                    mask=take,
                    other=0.0,
                )
                tl.store(
                    l1_sf + sf_block * address_index(POOL_TOKENS, WIDE_INDEX) + pool_row,
                    sf,
                    mask=take,
                )
            gate_w = tl.load(src_w_ptr + rid, mask=take, other=0.0)
            tl.store(l1_w + pool_row, gate_w, mask=take)
            if PARTIAL_REDUCE:
                # Each valid route has one producer; stale unused entries are masked.
                tl.store(
                    meta_l + address_index(src_id, WIDE_INDEX) * NTOK * TOPK + rid,
                    pool_row.to(tl.int32),
                    mask=take,
                )
            else:
                if USE_DST_PTR:
                    # Preserve the three-i32 allocation; rebuild addresses per replay.
                    src_cb = peer_bf16(cb_tab, src_id)
                    dst_addr = (
                        src_cb + (address_index(src_topk, WIDE_INDEX) * NTOK + src_tok) * K
                    ).to(tl.uint64)
                    tl.store(meta_l + pool_row * 3 + 0, dst_addr.to(tl.uint32), mask=take)
                    tl.store(
                        meta_l + pool_row * 3 + 1, (dst_addr >> 32).to(tl.uint32), mask=take
                    )
                    tl.store(meta_l + pool_row * 3 + 2, 0, mask=take)
                else:
                    tl.store(meta_l + pool_row * 3 + 0, src_id, mask=take)
                    tl.store(meta_l + pool_row * 3 + 1, src_tok, mask=take)
                    tl.store(meta_l + pool_row * 3 + 2, src_topk, mask=take)
            if DISPATCH_OVERLAP or not BULK_DISPATCH_READY:
                tl.debug_barrier()
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
                if DISPATCH_OVERLAP:
                    # Reserve queue slots in one atomic per nonempty lane group.
                    # Consumers acquire each slot, including when a publisher
                    # reserving an earlier slot is still finishing.
                    nready = tl.sum(ready.to(tl.int32), 0)
                    if nready > 0:
                        slot = tl.atomic_add(ready_count, nready, sem="relaxed")
                        offset = tl.cumsum(ready.to(tl.int32)) - 1
                        tl.atomic_xchg(ready_slots + slot + offset, block_id + 1,
                                       sem="release", mask=ready)
                elif BLOCK_READY:
                    tl.atomic_xchg(l1_ready + block_id, 1, sem="release", mask=ready)
                else:
                    for n_block in tl.static_range(0, NL1N):
                        tl.atomic_xchg(
                            l1_ready + block_id * NL1N + n_block, 1, sem="release", mask=ready
                        )
        if COPY_M_BLOCK:
            work_id += ROLE_BUDGET
        elif COPY_WORKERS:
            work_id += COPY_WORKERS
        elif STATIC_PAYLOAD:
            work_id += plan_workers if CACHE_DECOUPLED else PAYLOAD_WORKERS
        else:
            work_id = tl.atomic_add(dispatch_cursor + 1, 1, sem="relaxed")

    if DISPATCH_TRACE:
        tl.debug_barrier()
        tl.store(trace + 10, globaltimer())
    if COPY_M_BLOCK:
        # This cursor is unused by the static row-map traversal. The RMW
        # chain publishes each CTA's row assignments to the copy producers.
        tl.debug_barrier()
        tl.atomic_add(dispatch_cursor + 1, 1, sem="acq_rel")
        wait_count(dispatch_cursor + 1, ROLE_BUDGET)
        tl.debug_barrier()
        block = worker_id
        if worker_id < COPY_WORKERS:
            while block < active_blocks:
                copy_expert_block(
                    copy_row_map, block_rows, tok_tab, sf_tab, w_tab, cb_tab,
                    meta_l, l1_token, l1_sf, l1_w, l1_ready, block, NTOK,
                    BLOCK_M, K, COPY_K, TOPK, MY_PE, USE_DST_PTR,
                    DEFER_ROUTER, WIDE_INDEX, DST_PTR64,
                )
                block += COPY_WORKERS

    if BULK_DISPATCH_READY and not DISPATCH_OVERLAP:
        tl.debug_barrier()
        tl.atomic_add(dispatch_phase + 3, 1, sem="acq_rel")
        wait_count(dispatch_phase + 3, 2 * plan_workers)


@triton.jit
def copy_expert_block(
    row_map, block_rows, tok_tab, sf_tab, w_tab, cb_tab, meta,
    l1_token, l1_sf, l1_w, l1_ready, block, NTOK,
    BLOCK_M: tl.constexpr, K: tl.constexpr, COPY_K: tl.constexpr,
    TOPK: tl.constexpr, MY_PE: tl.constexpr, USE_DST_PTR: tl.constexpr,
    DEFER_ROUTER: tl.constexpr,
    WIDE_INDEX: tl.constexpr, DST_PTR64: tl.constexpr,
):
    lane = tl.arange(0, BLOCK_M)
    valid_m = tl.load(block_rows + block)
    valid = lane < valid_m
    row = address_index(block, WIDE_INDEX) * BLOCK_M + lane
    route = tl.load(row_map + row, mask=valid, other=0)
    rank = route >> 19
    rid = route & ((1 << 19) - 1)
    token = rid // TOPK
    slot = rid % TOPK
    source = peer_i8(tok_tab, rank) + address_index(token, WIDE_INDEX) * K
    offs = tl.arange(0, COPY_K)
    for start in tl.range(0, K, COPY_K, num_stages=2, loop_unroll_factor=1):
        ptr = source[:, None] + start + offs[None, :]
        ptr = tl.max_contiguous(tl.multiple_of(ptr, (1, 16)), (1, 16))
        value = tl.load(ptr, mask=valid[:, None], other=0)
        tl.store(
            l1_token + row[:, None] * K + start + offs[None, :],
            value, mask=valid[:, None],
        )
    scale = tl.load(peer_f32(sf_tab, rank) + token, mask=valid, other=0.0)
    tl.store(l1_sf + row, scale, mask=valid)
    if not DEFER_ROUTER:
        weight = tl.load(peer_f32(w_tab, rank) + rid, mask=valid, other=0.0)
        tl.store(l1_w + row, weight, mask=valid)
    if USE_DST_PTR:
        dest = peer_bf16(cb_tab, rank)
        ptr = (dest + (address_index(slot, WIDE_INDEX) * NTOK + token) * K).to(tl.uint64)
        if DST_PTR64 and (meta.to(tl.uint64) & 7) == 0:
            tl.store(meta.to(tl.pointer_type(tl.uint64)) + row, ptr, mask=valid)
        else:
            tl.store(meta + row * 3, ptr.to(tl.uint32), mask=valid)
            tl.store(meta + row * 3 + 1, (ptr >> 32).to(tl.uint32), mask=valid)
    else:
        tl.store(meta + row * 3, rank, mask=valid)
        tl.store(meta + row * 3 + 1, token, mask=valid)
        tl.store(meta + row * 3 + 2, slot, mask=valid)
    tl.debug_barrier()
    tl.atomic_xchg(l1_ready + block, 1, sem="release")


# Bound dispatch register live ranges for larger tiles. Small tiles keep
# dispatch inline to avoid device-call overhead.
@triton.jit(noinline=True)
def dispatch_payload_noinline(
    buffers, config: tl.constexpr, WIDE_INDEX: tl.constexpr = True
):
    dispatch_payload(buffers, config, WIDE_INDEX=WIDE_INDEX)


@triton.jit
def reduce_local_experts(
    local_contrib,
    route_tab,
    meta_tab,
    cb_tab,
    task,
    MY_PE: tl.constexpr,
    NTOK,
    EPR: tl.constexpr,
    TOPK: tl.constexpr,
    K: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
    REDUCE_ROWS: tl.constexpr = 1,
    RETURN_PULL: tl.constexpr = False,
    CONTRIB_STRIPES: tl.constexpr = False,
    BLOCK_M: tl.constexpr = 128,
    LOCAL_INVERSE: tl.constexpr = False,
    FOLD_LOCAL: tl.constexpr = False,
    NPES: tl.constexpr = 2,
    final_y=None,
    expert_weights=None,
    VENDOR_NUMERICS: tl.constexpr = False,
):
    if REDUCE_ROWS == 1:
        src = task // NTOK
        tok = task % NTOK
        routes = peer_i32(route_tab, src)
        h = tl.arange(0, triton.next_power_of_2(K))
        acc = tl.zeros((triton.next_power_of_2(K),), tl.float32)
        for slot in tl.static_range(0, TOPK):
            e = tl.load(routes + address_index(tok, WIDE_INDEX) * TOPK + slot)
            take = (e >= 0) & (e // EPR == MY_PE)
            row = tl.load(
                peer_i32(meta_tab, MY_PE)
                + (src * address_index(NTOK, WIDE_INDEX) + tok) * TOPK
                + slot,
                mask=take,
                other=0,
            )
            base = address_index(row, WIDE_INDEX) * K
            value = tl.load(local_contrib + base + h, mask=take & (h < K), other=0)
            acc += value.to(tl.float32)
        # Reuse two of the existing BF16 combine slots for rank partials.
        target = (
            peer_bf16(cb_tab, src) + (MY_PE * address_index(NTOK, WIDE_INDEX) + tok) * K
        )
        tl.store(target + h, acc.to(tl.bfloat16), mask=h < K)
    else:
        base_task = task * REDUCE_ROWS
        src = base_task // NTOK
        tok = base_task % NTOK + tl.arange(0, REDUCE_ROWS)
        if not LOCAL_INVERSE:
            routes = peer_i32(route_tab, src)
        inverse = peer_i32(meta_tab, MY_PE) + (src * address_index(NTOK, WIDE_INDEX) + tok) * TOPK
        h = tl.arange(0, triton.next_power_of_2(K))
        acc = tl.zeros((REDUCE_ROWS, triton.next_power_of_2(K)), tl.float32)
        for slot in tl.static_range(0, TOPK):
            if LOCAL_INVERSE:
                row = tl.load(inverse + slot, mask=tok < NTOK, other=-1)
                take = row >= 0
            else:
                e = tl.load(
                    routes + address_index(tok, WIDE_INDEX) * TOPK + slot,
                    mask=tok < NTOK,
                    other=-1,
                )
                take = (tok < NTOK) & (e >= 0) & (e // EPR == MY_PE)
                row = tl.load(
                    peer_i32(meta_tab, MY_PE)
                    + (src * address_index(NTOK, WIDE_INDEX) + tok) * TOPK
                    + slot,
                    mask=take,
                    other=0,
                )
            index = address_index(row, WIDE_INDEX)
            if CONTRIB_STRIPES:
                slab = index // BLOCK_M * K * BLOCK_M
                slab_offset = h // 128 * (128 * BLOCK_M) + h % 128
                offset = slab[:, None] + (index % BLOCK_M)[:, None] * 128 + slab_offset[None, :]
            else:
                offset = index[:, None] * K + h[None, :]
            value = tl.load(
                local_contrib + offset,
                mask=take[:, None] & (h[None, :] < K),
                other=0,
            )
            contribution = value.to(tl.float32)
            if VENDOR_NUMERICS:
                # The saved expert contribution is BF16 before router weighting.
                weight = tl.load(expert_weights + row, mask=take, other=0.0)
                contribution *= weight[:, None]
            acc += contribution
        if FOLD_LOCAL:
            peer = peer_bf16(cb_tab, MY_PE)
            if NPES == 2:
                start = ((1 - MY_PE) * address_index(NTOK, WIDE_INDEX) + tok) * K
                ptr = peer + start[:, None] + h[None, :]
                mask = (tok[:, None] < NTOK) & (h[None, :] < K)
                if (peer.to(tl.uint64) & 15) == 0:
                    value = tl.load(tl.max_contiguous(tl.multiple_of(ptr, (1, 8)), (1, 8)),
                                    mask=mask, other=0)
                else:
                    value = tl.load(ptr, mask=mask, other=0)
                local = acc if VENDOR_NUMERICS else acc.to(tl.bfloat16).to(tl.float32)
                result = local + value.to(tl.float32)
            else:
                mask = (tok[:, None] < NTOK) & (h[None, :] < K)
                result = tl.zeros((REDUCE_ROWS, triton.next_power_of_2(K)), tl.float32)
                for src in tl.static_range(0, NPES):
                    if src == MY_PE:
                        value = acc.to(tl.bfloat16)
                    else:
                        start = (src * address_index(NTOK, WIDE_INDEX) + tok) * K
                        ptr = peer + start[:, None] + h[None, :]
                        value = tl.load(ptr, mask=mask, other=0)
                    result += value.to(tl.float32)
            out = final_y + address_index(tok, WIDE_INDEX)[:, None] * K + h[None, :]
            tl.store(out, result.to(tl.bfloat16), mask=mask)
        else:
            if RETURN_PULL:
                peer = peer_bf16(cb_tab, MY_PE)
                target = peer + (src * address_index(NTOK, WIDE_INDEX) + tok) * K
            else:
                peer = peer_bf16(cb_tab, src)
                target = peer + (MY_PE * address_index(NTOK, WIDE_INDEX) + tok) * K
            if (peer.to(tl.uint64) & 15) == 0:
                tl.store(
                    tl.max_contiguous(
                        tl.multiple_of(target[:, None] + h[None, :], (1, 8)), (1, 8)
                    ),
                    acc.to(tl.bfloat16),
                    mask=(tok[:, None] < NTOK) & (h[None, :] < K),
                )
            else:
                tl.store(
                    target[:, None] + h[None, :],
                    acc.to(tl.bfloat16),
                    mask=(tok[:, None] < NTOK) & (h[None, :] < K),
                )



@triton.jit(noinline=True)
def return_rank_rows(local_contrib, route_tab, meta_tab, cb_tab, expert_weights, token_group, dst, NTOK,
                     MY_PE: tl.constexpr, EPR: tl.constexpr, TOPK: tl.constexpr,
                     K: tl.constexpr, ROWS: tl.constexpr, WIDE_INDEX: tl.constexpr,
                     VENDOR_NUMERICS: tl.constexpr):
    reduce_local_experts(
        local_contrib, route_tab, meta_tab, cb_tab, token_group + dst * (NTOK // ROWS),
        MY_PE=MY_PE, NTOK=NTOK, EPR=EPR, TOPK=TOPK, K=K, WIDE_INDEX=WIDE_INDEX,
        REDUCE_ROWS=ROWS, LOCAL_INVERSE=True, expert_weights=expert_weights,
        VENDOR_NUMERICS=VENDOR_NUMERICS,
    )


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
    local_contrib,
    token_cache,
    profile_times,
    copy_row_map,
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
    L1_PIPELINE: tl.constexpr = False,
    L1_ASYNC: tl.constexpr = False,
    LARGE_NP2_TUNE: tl.constexpr = False,
    PACK_ROUTE_META: tl.constexpr = False,
    METADATA_BLOCK: tl.constexpr = 128,
    L2_BATCH: tl.constexpr = 0,
    L1_QUANT_ROWS: tl.constexpr = 8,
    USE_DST_PTR: tl.constexpr = False,
    L2_STAGES: tl.constexpr = 2,
    WIDE_INDEX: tl.constexpr = True,
    A_BLOCK_PTR: tl.constexpr = False,
    ASYNC_A: tl.constexpr = False,
    ASYNC_B: tl.constexpr = False,
    PARTIAL_REDUCE: tl.constexpr = False,
    CACHE_TOKENS: tl.constexpr = False,
    WARP_HIST: tl.constexpr = False,
    REDUCE_ROWS: tl.constexpr = 1,
    DEFER_ROUTER: tl.constexpr = False,
    VENDOR_NUMERICS: tl.constexpr = False,
    ATOMIC_DISPATCH: tl.constexpr = False,
    COPY_K: tl.constexpr = 0,
    GATHER_INPUT: tl.constexpr = False,
    COUNT_CHUNK: tl.constexpr = 0,
    STATIC_PAYLOAD: tl.constexpr = False,
    COMPACT_META: tl.constexpr = False,
    ROW_COUNTER: tl.constexpr = False,
    SINGLE_CURSOR: tl.constexpr = False,
    STATIC_GEMM2: tl.constexpr = False,
    RANK_RETURN: tl.constexpr = False,
    MULTI_RETURN_WORKERS: tl.constexpr = 0,
    SERIAL_CACHE: tl.constexpr = False,
    PAYLOAD_WORKERS: tl.constexpr = 0,
    DISPATCH_OVERLAP: tl.constexpr = False,
    CACHE_DECOUPLED: tl.constexpr = False,
    LOCAL_G1_FIRST: tl.constexpr = False,
    DISPATCH_TRACE: tl.constexpr = False,
    ROLE_RETURN_BATCH: tl.constexpr = 2,
    RANK_RETURN_WORKERS: tl.constexpr = 0,
    RANK_RETURN_STRIDE: tl.constexpr = 2,
    RANK_RETURN_TRACE: tl.constexpr = False,
    EARLY_REMOTE_ONLY: tl.constexpr = True,
    BULK_DISPATCH_READY: tl.constexpr = False,
    STATIC_DISPATCH_META: tl.constexpr = False,
    RETURN_PULL: tl.constexpr = False,
    RETURN_ORDER: tl.constexpr = "original",
    RETURN_BLOCK: tl.constexpr = 1,
    RETURN_SWAP: tl.constexpr = False,
    FOLD_ROWS: tl.constexpr = 1,
    CONTRIB_STRIPES: tl.constexpr = False,
    LOCAL_INVERSE: tl.constexpr = False,
    CACHE_WORKERS: tl.constexpr = 16,
    COUNTER_TRACE: tl.constexpr = False,
    TAIL_FIRST: tl.constexpr = False,
    CACHE_READY_ROW: tl.constexpr = False,
    ROW_ORDER: tl.constexpr = False,
    COPY_WORKERS: tl.constexpr = 0,
    COPY_M_BLOCK: tl.constexpr = False,
    DST_PTR64: tl.constexpr = False,
    G2_COMBINE_ROWS: tl.constexpr = 1,
):
    """Fuse dispatch, both GEMMs and combine using persistent CTA work queues.

    All resident CTAs begin with dispatch. A CTA that drains its payload work
    can acquire a ready L1 tile while other CTAs still move payloads. Roles
    change dynamically; there is no fixed communication/math CTA partition.
    L1 completion gates L2, and all peer L2 stores finish before combine.
    """
    USE_EP_OPT: tl.constexpr = (
        (NPES == 2 or NPES == 4 or NPES == 8)
        and K == 2048
        and INTER == 512
        and EPR * NPES == 256
        and TOPK == 8
    )
    if PARTIAL_REDUCE:
        tl.static_assert(
            NPES <= TOPK and TOPK == 8 and AGGREGATE_COMPLETION and not USE_DST_PTR
        )
    tl.debug_barrier()
    worker_id = tl.program_id(0)
    if DST_PTR64:
        tl.static_assert(COPY_M_BLOCK and USE_DST_PTR and not PARTIAL_REDUCE)
    if COPY_M_BLOCK:
        tl.static_assert(COMPACT_META and ATOMIC_DISPATCH and COPY_WORKERS > 0)
        tl.static_assert(NSF == 1 and not GATHER_INPUT and not PARTIAL_REDUCE)
    if DISPATCH_TRACE:
        trace_dispatch = profile_times + worker_id * 16
        tl.store(trace_dispatch, globaltimer())
        tl.store(trace_dispatch + 11, 0)
    if COUNTER_TRACE:
        record = profile_times + worker_id * 16
        tl.store(record + tl.arange(0, 16), tl.full((16,), 0, tl.int64))
        tl.debug_barrier()
        tl.store(record, globaltimer())
        last_g1 = tl.full((), 0, tl.int64)
        first_g2 = tl.full((), 0, tl.int64)
        waits = tl.full((), 0, tl.int32)
        wait_ticks = tl.full((), 0, tl.int64)
        gated_tiles = tl.full((), 0, tl.int32)
    if GATHER_INPUT:
        # The first four bytes per expert row store the token index. The
        # activation cache fits in the remaining INT8 dispatch workspace.
        token_cache = l1_token + address_index(POOL_TOKENS, WIDE_INDEX) * 4
    if CACHE_TOKENS:
        tl.static_assert((GATHER_INPUT or PARTIAL_REDUCE) and ROLE_BUDGET >= CACHE_WORKERS)
        if worker_id < CACHE_WORKERS:
            copy_lane = tl.arange(0, 8192)
            if NPES > 2 and CACHE_DECOUPLED:
                tiles_per_peer = tl.cdiv(NTOK * K, 8192)
                for work in tl.range(worker_id, (NPES - 1) * tiles_per_peer, CACHE_WORKERS, num_stages=2):
                    slot = work % (NPES - 1)
                    src = slot + (slot >= MY_PE).to(tl.int32)
                    src_ptr = peer_i8(tok_tab, src)
                    idx = work // (NPES - 1) * 8192 + copy_lane
                    valid = idx < NTOK * K
                    value = tl.load(tl.max_contiguous(tl.multiple_of(src_ptr + idx, 16), 8192),
                                    mask=valid, other=0)
                    dst = token_cache + address_index(slot, WIDE_INDEX) * NTOK * K + idx
                    tl.store(tl.max_contiguous(tl.multiple_of(dst, 16), 8192), value, mask=valid)
            else:
                for cache_rank in tl.static_range(0, 1 if GATHER_INPUT else NPES):
                    source_rank = (1 - MY_PE) if GATHER_INPUT else cache_rank
                    source = peer_i8(tok_tab, source_rank)
                    for copy_base in tl.range(
                        worker_id * 8192, NTOK * K, CACHE_WORKERS * 8192, num_stages=2
                    ):
                        ix = copy_base + copy_lane
                        value = tl.load(
                            tl.max_contiguous(tl.multiple_of(source + ix, 16), 8192),
                            mask=ix < NTOK * K,
                            other=0,
                        )
                        tl.store(
                            tl.max_contiguous(
                                tl.multiple_of(
                                    token_cache
                                    + address_index(cache_rank, WIDE_INDEX) * NTOK * K
                                    + ix,
                                    16,
                                ),
                                8192,
                            ),
                            value,
                            mask=ix < NTOK * K,
                        )
            tl.debug_barrier()
            if CACHE_DECOUPLED:
                tl.atomic_add(l1_ready + 4, 1, sem="acq_rel")
                if DISPATCH_TRACE:
                    tl.store(trace_dispatch + 8, globaltimer())
                    tl.store(trace_dispatch + 9, globaltimer())
                    tl.store(trace_dispatch + 10, globaltimer())
        # Other CTAs count routes while this cache is populated. The phase-0
        # acquire-release rendezvous publishes every cache store before payload
        # packing. GEMM1 retains its contiguous expert pool and async loads.
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
        cb_tab,
        token_cache,
        profile_times,
        copy_row_map,
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
        USE_EP_OPT,
        METADATA_BLOCK,
        USE_DST_PTR,
        PARTIAL_REDUCE,
        CACHE_TOKENS,
        WARP_HIST,
        ATOMIC_DISPATCH,
        COPY_K,
        GATHER_INPUT,
        COUNT_CHUNK,
        STATIC_PAYLOAD,
        COMPACT_META,
        LOCAL_INVERSE,
        BULK_DISPATCH_READY,
        STATIC_DISPATCH_META,
        RANK_RETURN or CACHE_DECOUPLED,
        PAYLOAD_WORKERS,
        DISPATCH_OVERLAP,
        DISPATCH_TRACE,
        CACHE_DECOUPLED,
        CACHE_WORKERS,
        COPY_WORKERS,
        COPY_M_BLOCK,
        DEFER_ROUTER,
        DST_PTR64,
    )
    if CACHE_DECOUPLED:
        if worker_id >= CACHE_WORKERS:
            dispatch_payload_noinline(dispatch_buffers, dispatch_config, WIDE_INDEX=WIDE_INDEX)
        else:
            wait_count(dispatch_phase + 3, 2 * (ROLE_BUDGET - CACHE_WORKERS))
    elif BLOCK_M >= 32:
        dispatch_payload_noinline(
            dispatch_buffers, dispatch_config, WIDE_INDEX=WIDE_INDEX
        )
    else:
        dispatch_payload(dispatch_buffers, dispatch_config, WIDE_INDEX=WIDE_INDEX)
    tl.debug_barrier()
    # Schedule only the tiles produced by this invocation's routing.
    active_blocks = tl.load(dispatch_phase + 2)
    active_l1_tasks = active_blocks * NL1N
    active_l2_tasks = active_blocks * NL2N
    l1_count = tl.full((), 0, tl.int32)
    if DISPATCH_OVERLAP:
        tl.static_assert(BULK_DISPATCH_READY and COMPACT_META and RANK_RETURN and not SINGLE_CURSOR)
        # A ticket addresses an output-column tile in dispatch-ready order.
        # A waiting GEMM1 CTA owns no dispatch work needed by its producer.
        task = tl.atomic_add(task_cursor, 1, sem="relaxed")
        ready_slot = tl.full((), -1, tl.int32)
        row = tl.full((), 0, tl.int32)
        while task < active_l1_tasks:
            slot = task // NL1N
            col = task % NL1N
            if slot != ready_slot:
                ptr = l2_meta + slot
                row = load_acquire(ptr)
                while row == 0:
                    row = load_acquire(ptr)
                ready_slot = slot
                tl.debug_barrier()
            if DISPATCH_TRACE and l1_count == 0:
                tl.store(profile_times + worker_id * 16 + 11, globaltimer())
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
                BLOCK_K=64 if L1_PIPELINE else (256 if LARGE_NP2_TUNE else BLOCK_K),
                K=K,
                INTER=INTER,
                L1_N=L1_N,
                L1_OUT_N=L1_OUT_N,
                NPAIR=NPAIR,
                NK1=K // 64 if L1_PIPELINE else (K // 256 if LARGE_NP2_TUNE else NK1),
                NL1N=NL1N,
                task_id=(row - 1) * NL1N + col,
                L1_STAGES=3 if L1_PIPELINE else 2,
                USE_FAST=USE_EP_OPT,
                QUANT_ROWS=L1_QUANT_ROWS,
                WIDE_INDEX=WIDE_INDEX,
                A_BLOCK_PTR=L1_ASYNC or A_BLOCK_PTR,
                ASYNC_A=L1_ASYNC or ASYNC_A,
                ASYNC_B=L1_ASYNC or ASYNC_B,
                GATHER_INPUT=GATHER_INPUT,
                CACHE_TOKENS=CACHE_TOKENS,
                NTOK=NTOK,
                token_cache=token_cache,
                tok_tab=tok_tab,
                MY_PE=MY_PE,
                COMPACT_META=COMPACT_META,
                block_rows=block_rows,
                ROW_COUNTER=ROW_COUNTER,
                VENDOR_NUMERICS=VENDOR_NUMERICS,
            )
            l1_count += 1
            task = tl.atomic_add(task_cursor, 1, sem="relaxed")
    else:
        if CACHE_DECOUPLED:
            remote_rows = tl.load(l1_ready)
            local_rows = active_blocks - remote_rows
            cache_ready = tl.full((), False, tl.int1)
        task = tl.atomic_add(task_cursor + 0, 1, sem="relaxed")
        while task < active_l1_tasks:
            if CACHE_DECOUPLED:
                row = task // NL1N
                if LOCAL_G1_FIRST:
                    row = tl.where(row < local_rows, row + remote_rows, row - local_rows)
                actual_task = row * NL1N + task % NL1N
                if row < remote_rows and not cache_ready:
                    wait_count(l1_ready + 4, CACHE_WORKERS)
                    tl.debug_barrier()
                    cache_ready = True
            else:
                actual_task = task
            if BULK_DISPATCH_READY:
                tl.static_assert(COMPACT_META and not SINGLE_CURSOR)
            elif COMPACT_META:
                wait_count(l1_ready + task // NL1N, 1)
            else:
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
                BLOCK_K=64 if L1_PIPELINE else (256 if LARGE_NP2_TUNE else BLOCK_K),
                K=K,
                INTER=INTER,
                L1_N=L1_N,
                L1_OUT_N=L1_OUT_N,
                NPAIR=NPAIR,
                NK1=K // 64 if L1_PIPELINE else (K // 256 if LARGE_NP2_TUNE else NK1),
                NL1N=NL1N,
                task_id=actual_task,
                L1_STAGES=3 if L1_PIPELINE else 2,
                USE_FAST=USE_EP_OPT,
                QUANT_ROWS=L1_QUANT_ROWS,
                WIDE_INDEX=WIDE_INDEX,
                A_BLOCK_PTR=L1_ASYNC or A_BLOCK_PTR,
                ASYNC_A=L1_ASYNC or ASYNC_A,
                ASYNC_B=L1_ASYNC or ASYNC_B,
                GATHER_INPUT=GATHER_INPUT,
                CACHE_TOKENS=CACHE_TOKENS,
                NTOK=NTOK,
                token_cache=token_cache,
                tok_tab=tok_tab,
                MY_PE=MY_PE,
                COMPACT_META=COMPACT_META,
                block_rows=block_rows,
                ROW_COUNTER=ROW_COUNTER,
                VENDOR_NUMERICS=VENDOR_NUMERICS,
            )
            tl.debug_barrier()
            if AGGREGATE_COMPLETION:
                l1_count += 1
            else:
                tl.atomic_add(l1_done, 1, sem="release")
            if COUNTER_TRACE:
                last_g1 = globaltimer()
            task = tl.atomic_add(task_cursor + 0, 1, sem="relaxed")
    if AGGREGATE_COMPLETION:
        tl.debug_barrier()
        if l1_count > 0:
            tl.atomic_add(l1_done, l1_count, sem="acq_rel")
    if ROW_COUNTER:
        tl.static_assert(AGGREGATE_COMPLETION and PARTIAL_REDUCE)
    else:
        wait_count(l1_done, active_l1_tasks)
    tl.debug_barrier()
    count = tl.full((), 0, tl.int32)
    batch_size: tl.constexpr = (
        L2_BATCH if L2_BATCH else (4 if AGGREGATE_COMPLETION else 1)
    )
    if RANK_RETURN:
        tl.static_assert(STATIC_GEMM2 and PARTIAL_REDUCE and REDUCE_ROWS > 1)
        remote_blocks = tl.load(l1_ready)
        n_returns = NTOK // REDUCE_ROWS
        remote_worker = worker_id % NPES != MY_PE
        return_step = ROLE_BUDGET // NPES if EARLY_REMOTE_ONLY else ROLE_BUDGET
        RETURN_WORKERS: tl.constexpr = RANK_RETURN_WORKERS or MULTI_RETURN_WORKERS or 20
        return_id = worker_id
        is_return_worker = worker_id < RETURN_WORKERS
        early_count = tl.full((), 0, tl.int32)
        first_return = tl.full((), 0, tl.int64)
        if RANK_RETURN_TRACE:
            trace = profile_times + worker_id * 16
            tl.store(trace, globaltimer())
        for group in tl.range(0, 2, loop_unroll_factor=2):
            start_block = tl.where(group == 0, 0, remote_blocks)
            n_blocks = tl.where(group == 0, remote_blocks, active_blocks - remote_blocks)
            n_tasks = n_blocks * NL2N
            compute_workers = tl.where(group == 0, ROLE_BUDGET, ROLE_BUDGET - RETURN_WORKERS)
            compute_id = tl.where(group == 0, worker_id, worker_id - RETURN_WORKERS)
            batch = tl.where(compute_id >= 0, compute_id * ROLE_RETURN_BATCH, n_tasks)
            if group == 1 and is_return_worker:
                wait_count(l1_ready + 1, remote_blocks * NL2N)
                tl.debug_barrier()
                while return_id < (NPES - 1) * n_returns:
                    if RANK_RETURN_TRACE and early_count == 0:
                        first_return = globaltimer()
                    dst = (MY_PE + 1 + return_id % (NPES - 1)) % NPES
                    return_rank_rows(
                        local_contrib, route_tab, meta_tab, cb_tab, l1_w, return_id // (NPES - 1), dst, NTOK,
                        MY_PE, EPR, TOPK, K, REDUCE_ROWS, WIDE_INDEX, VENDOR_NUMERICS,
                    )
                    early_count += 1
                    return_id += RETURN_WORKERS
            published = tl.full((), 0, tl.int32)
            rounds = tl.full((), 0, tl.int32)
            while batch < n_tasks:
                for offset in tl.range(0, ROLE_RETURN_BATCH, loop_unroll_factor=1):
                    logical = batch + offset
                    first = logical // (L2_GROUP_M * NL2N) * L2_GROUP_M
                    rem = logical % (L2_GROUP_M * NL2N)
                    if first + L2_GROUP_M <= n_blocks:
                        task = (start_block + first + rem % L2_GROUP_M) * NL2N + rem // L2_GROUP_M
                    else:
                        group_m = n_blocks - first
                        task = (start_block + first + rem % group_m) * NL2N + rem // group_m
                    cta_l2_gemm(
                        l1_meta if COMPACT_META else l2_meta,
                        local_contrib,
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
                        BLOCK_K=(
                            64 if PARTIAL_REDUCE else (256 if LARGE_NP2_TUNE else BLOCK_K)
                        ),
                        K=K,
                        INTER=INTER,
                        L2_N=L2_N,
                        NK2=(
                            INTER // 64
                            if PARTIAL_REDUCE
                            else (INTER // 256 if LARGE_NP2_TUNE else NK2)
                        ),
                        task_id=task,
                        USE_DST_PTR=USE_DST_PTR,
                        DST_PTR64=DST_PTR64,
                        L2_STAGES=L2_STAGES,
                        WIDE_INDEX=WIDE_INDEX,
                        A_BLOCK_PTR=A_BLOCK_PTR,
                        ASYNC_A=PARTIAL_REDUCE or ASYNC_A,
                        ASYNC_B=ASYNC_B,
                        PARTIAL_REDUCE=PARTIAL_REDUCE,
                        DEFER_ROUTER=DEFER_ROUTER,
                        COMPACT_META=COMPACT_META,
                        COMPACT_N=NL2N,
                        CONTRIB_STRIPES=CONTRIB_STRIPES,
                        block_rows=block_rows,
                    )
                published += ROLE_RETURN_BATCH
                batch += compute_workers * ROLE_RETURN_BATCH
            tl.debug_barrier()
            if group == 0 and published > 0:
                tl.atomic_add(l1_ready + 1, published, sem="acq_rel")
        tl.debug_barrier()
        if RANK_RETURN_TRACE:
            tl.store(trace + 1, globaltimer())
            tl.store(trace + 8, early_count)
            tl.store(trace + 9, first_return)
        tl.atomic_add(l2_done, 1, sem="acq_rel")
        wait_count(l2_done, ROLE_BUDGET)
        tl.debug_barrier()
        if RANK_RETURN_TRACE:
            tl.store(trace + 2, globaltimer())
        tl.debug_barrier()
        if RANK_RETURN_TRACE:
            tl.store(trace + 3, globaltimer())
        cta_l2_batch_finish(
            l1_done, combine_ready, sig_tab, MY_PE, NPES, active_l1_tasks + ROLE_BUDGET, 1,
        )
    else:
        if SINGLE_CURSOR:
            # Exhausting the GEMM1 interval already gives us a GEMM2 ticket.
            # No producer can claim new GEMM1 work after this transition.
            batch = (task - active_l1_tasks) * batch_size
        elif STATIC_GEMM2:
            batch = worker_id * batch_size
        else:
            batch = tl.atomic_add(task_cursor + 1, batch_size, sem="relaxed")
        all_ready = tl.full((), False, tl.int1)
        ready_row = tl.full((), -1, tl.int32)
        while batch < active_l2_tasks:
            if ROW_COUNTER and not all_ready:
                all_ready = load_acquire(l1_done) >= active_l1_tasks
                if all_ready:
                    tl.debug_barrier()
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
                    if ROW_ORDER:
                        task = logical
                    if TAIL_FIRST:
                        task = (active_blocks - 1 - task // NL2N) * NL2N + task % NL2N
                    if ROW_COUNTER and not all_ready:
                        row = task // NL2N
                        if not CACHE_READY_ROW or row != ready_row:
                            ptr = l2_arrival_mask.to(tl.pointer_type(tl.int32)) + row
                            value = load_acquire(ptr)
                            if value <= NL1N:
                                if COUNTER_TRACE:
                                    wait_start = globaltimer()
                                    waits += 1
                                while value <= NL1N:
                                    value = load_acquire(ptr)
                                if COUNTER_TRACE:
                                    wait_ticks += globaltimer() - wait_start
                            tl.debug_barrier()
                            if CACHE_READY_ROW:
                                ready_row = row
                        if COUNTER_TRACE:
                            gated_tiles += 1
                    if COUNTER_TRACE and first_g2 == 0:
                        first_g2 = globaltimer()
                    cta_l2_gemm(
                        l1_meta if COMPACT_META else l2_meta,
                        local_contrib,
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
                        BLOCK_K=(
                            64 if PARTIAL_REDUCE else (256 if LARGE_NP2_TUNE else BLOCK_K)
                        ),
                        K=K,
                        INTER=INTER,
                        L2_N=L2_N,
                        NK2=(
                            INTER // 64
                            if PARTIAL_REDUCE
                            else (INTER // 256 if LARGE_NP2_TUNE else NK2)
                        ),
                        task_id=task,
                        USE_DST_PTR=USE_DST_PTR,
                        DST_PTR64=DST_PTR64,
                        L2_STAGES=L2_STAGES,
                        WIDE_INDEX=WIDE_INDEX,
                        A_BLOCK_PTR=A_BLOCK_PTR,
                        ASYNC_A=PARTIAL_REDUCE or ASYNC_A,
                        ASYNC_B=ASYNC_B,
                        PARTIAL_REDUCE=PARTIAL_REDUCE,
                        DEFER_ROUTER=DEFER_ROUTER,
                        COMPACT_META=COMPACT_META,
                        COMPACT_N=NL2N,
                        CONTRIB_STRIPES=CONTRIB_STRIPES,
                        block_rows=block_rows,
                    )
                    count += 1
            if SINGLE_CURSOR:
                ticket = tl.atomic_add(task_cursor, 1, sem="relaxed")
                batch = (ticket - active_l1_tasks) * batch_size
            elif STATIC_GEMM2:
                batch += ROLE_BUDGET * batch_size
            else:
                batch = tl.atomic_add(task_cursor + 1, batch_size, sem="relaxed")
        tl.debug_barrier()
        if COUNTER_TRACE:
            tl.store(record + 1, last_g1)
            tl.store(record + 2, first_g2)
            tl.store(record + 3, globaltimer())
            tl.store(record + 4, waits)
            tl.store(record + 5, wait_ticks)
            tl.store(record + 6, gated_tiles)
            tl.store(record + 7, count)
        if PARTIAL_REDUCE:
            if count > 0:
                tl.atomic_add(l2_done, count, sem="acq_rel")
            wait_count(l2_done, active_l2_tasks)
            tl.debug_barrier()
            reduce_task = worker_id
            while reduce_task < ((NPES * NTOK + REDUCE_ROWS - 1) // REDUCE_ROWS):
                if RETURN_ORDER == "interleave":
                    stripe = reduce_task // RETURN_BLOCK
                    dst = stripe % NPES
                    if RETURN_SWAP:
                        dst = dst ^ MY_PE
                    local_group = stripe // NPES * RETURN_BLOCK + reduce_task % RETURN_BLOCK
                    reduce_logical = local_group + dst * (NTOK // REDUCE_ROWS)
                elif RETURN_ORDER == "remote_first":
                    reduce_logical = (reduce_task + (1 - MY_PE) * (NTOK // REDUCE_ROWS)) % (NPES * NTOK // REDUCE_ROWS)
                else:
                    reduce_logical = reduce_task
                reduce_local_experts(
                    local_contrib,
                    route_tab,
                    meta_tab,
                    cb_tab,
                    reduce_logical,
                    MY_PE=MY_PE,
                    NTOK=NTOK,
                    EPR=EPR,
                    TOPK=TOPK,
                    K=K,
                    WIDE_INDEX=WIDE_INDEX,
                    REDUCE_ROWS=REDUCE_ROWS,
                    RETURN_PULL=RETURN_PULL,
                    CONTRIB_STRIPES=CONTRIB_STRIPES,
                    BLOCK_M=BLOCK_M,
                    LOCAL_INVERSE=LOCAL_INVERSE,
                )
                reduce_task += ROLE_BUDGET
            tl.debug_barrier()
            # The acq_rel chain orders every partial store before the peer barrier.
            cta_l2_batch_finish(
                l1_done,
                combine_ready,
                sig_tab,
                MY_PE=MY_PE,
                NPES=NPES,
                total_tasks=active_l1_tasks + ROLE_BUDGET,
                count=1,
            )
        else:
            # Empty local work must still join peer completion exactly once.
            if (count > 0) | (USE_EP_OPT & (active_l2_tasks == 0) & (worker_id == 0)):
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
    tl.debug_barrier()
    if RANK_RETURN:
        LOCAL_ROWS: tl.constexpr = 2
        local_id = worker_id
        while local_id < NTOK // LOCAL_ROWS:
            reduce_local_experts(
                local_contrib, route_tab, meta_tab, cb_tab,
                local_id + MY_PE * (NTOK // LOCAL_ROWS), MY_PE, NTOK, EPR, TOPK, K, WIDE_INDEX,
                REDUCE_ROWS=LOCAL_ROWS, LOCAL_INVERSE=True, FOLD_LOCAL=True, NPES=NPES, final_y=final_y,
                expert_weights=l1_w, VENDOR_NUMERICS=VENDOR_NUMERICS,
            )
            local_id += ROLE_BUDGET
    elif RETURN_PULL or FOLD_ROWS > 1:
        tl.static_assert(PARTIAL_REDUCE and NPES == 2)
        task = worker_id * FOLD_ROWS
        while task < NTOK:
            fold_rank_partials(cb_tab, final_y, task, NTOK, K, FOLD_ROWS, MY_PE, WIDE_INDEX, RETURN_PULL)
            task += ROLE_BUDGET * FOLD_ROWS
    elif AGGREGATE_COMPLETION:
        task = worker_id * G2_COMBINE_ROWS
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
                WIDE_INDEX=WIDE_INDEX,
                PARTIAL_REDUCE=PARTIAL_REDUCE,
                router_tab=w_tab,
                DEFER_ROUTER=DEFER_ROUTER,
                task_id=task, ROWS=G2_COMBINE_ROWS,
            )
            task += ROLE_BUDGET * G2_COMBINE_ROWS
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
                WIDE_INDEX=WIDE_INDEX,
                PARTIAL_REDUCE=PARTIAL_REDUCE,
                router_tab=w_tab,
                DEFER_ROUTER=DEFER_ROUTER,
                task_id=task,
            )
            task = tl.atomic_add(task_cursor + 2, 1, sem="relaxed")
    if COUNTER_TRACE:
        tl.store(record + 8, globaltimer())

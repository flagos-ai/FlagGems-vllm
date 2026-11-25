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

"""Hopper sparse MLA over native V4 cache pages with seven E8M0 scale groups."""


import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    NVMMADistributedLayout,
    NVMMASharedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.hopper import async_copy

from flaggems_vllm.ops.flash_mla_with_kvcache_model1 import _cache_views as cache_views
from flaggems_vllm.ops.flash_mla_with_kvcache_model1 import tle_subslice
from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import (
    publish_p_fp8_sw64_coupled_stmatrix,
)
from flaggems_vllm.runtime.backend._nvidia.sparse_mla_model1_gather import (
    normalize_cache,
    normalize_query,
    producer,
    split_block_range,
)
from flaggems_vllm.runtime.backend._nvidia.sparse_mla_tiles import (
    publish_sparse_p128_fp8,
    sparse_fp8_publish_output,
    sparse_fp8_transpose_values,
    sparse_named_barriers,
)


@gluon.jit
def publish(
    Output,
    row0,
    acc,
    scale,
    OFFSET: gl.constexpr,
    WIDTH: gl.constexpr,
    STREAM: gl.constexpr,
):
    layout: gl.constexpr = acc.type.layout
    row = row0 + gl.arange(0, 64, layout=gl.constexpr(SliceLayout(1, layout)))
    col = OFFSET + gl.arange(0, WIDTH, layout=gl.constexpr(SliceLayout(0, layout)))
    factor = gl.convert_layout(scale, gl.constexpr(SliceLayout(1, layout)))
    gl.store(
        Output + row[:, None] * 512 + col[None, :],
        (acc * factor[:, None]).to(Output.dtype.element_ty),
        cache_modifier=".cs" if STREAM else "",
    )


@gluon.jit
def bf_band(
    Q,
    QS,
    K,
    q,
    kv,
    h,
    offset,
    valid,
    scales,
    layout: gl.constexpr,
    policy: gl.constexpr,
    start: gl.constexpr,
    width: gl.constexpr,
):
    col = start + gl.arange(0, width, layout=gl.constexpr(SliceLayout(0, layout)))
    qv = gl.load(Q + (h * 448)[:, None] + col[None, :], eviction_policy="evict_first")
    qs = gl.load(
        QS + (h * 7)[:, None] + (col // 64)[None, :], eviction_policy="evict_first"
    )
    q.slice(start, width, 1).store((qv.to(gl.float32) * qs).to(gl.bfloat16))
    quantized_values = gl.load(
        K + offset[:, None] + col[None, :], valid[:, None], 0.0, eviction_policy=policy
    )
    e = ((scales[:, None] >> ((col // 64) * 8)[None, :]) & 255).to(gl.uint16)
    power = (e << 7).to(gl.bfloat16, bitcast=True)
    kv.slice(start, width, 1).store(
        gl.inline_asm_elementwise(
            "mul.rn.bf16x2 $0,$1,$2;",
            "=r,r,r",
            [quantized_values.to(gl.bfloat16), power],
            gl.bfloat16,
            is_pure=True,
            pack=2,
        )
    )


@gluon.jit
def one_block(
    Q,
    QR,
    QS,
    K,
    KS,
    KR,
    Indices,
    Output,
    L,
    scale,
    N: gl.constexpr,
    TOPK: gl.constexpr,
    STRIDE: gl.constexpr,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    BK: gl.constexpr,
):
    # All eight warps cooperate when a split is exactly one block. There
    # is no running softmax state, inter-partition handoff or KV recycling.
    layout: gl.constexpr = BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0])
    qkl: gl.constexpr = NVMMADistributedLayout([3, 0], [4, 2], [16, BK // 2, 16])
    pvl: gl.constexpr = NVMMADistributedLayout([3, 0], [4, 2], [16, 128, 16])
    shared: gl.constexpr = NVMMASharedLayout(128, 16)
    q = gl.allocate_shared_memory(gl.bfloat16, (64, 512), shared)
    kv = gl.allocate_shared_memory(gl.bfloat16, (BK, 512), shared)
    p = gl.allocate_shared_memory(gl.bfloat16, (64, BK), shared)
    batch = gl.program_id(0).to(gl.int64)
    head = gl.program_id(1) * 64
    block = gl.program_id(2)
    row0 = batch * H + head
    row_out = (batch * SPLITS + block) * H + head
    r = gl.arange(0, BK, layout=gl.constexpr(SliceLayout(1, layout)))
    h = row0 + gl.arange(0, 64, layout=gl.constexpr(SliceLayout(1, layout)))
    d = gl.arange(0, 64, layout=gl.constexpr(SliceLayout(0, layout)))
    pos = block * BK + r
    idx = gl.load(
        Indices + batch * TOPK + pos, pos < TOPK, -1, eviction_policy="evict_first"
    )
    valid = (idx >= 0) & (idx < N)
    idx = gl.where(valid, idx, 0)
    page = (idx // 64).to(gl.int64) * STRIDE
    slot = idx % 64
    offset = page + slot * 576
    policy: gl.constexpr = "evict_first" if N > 131072 else ""
    scales = gl.load(KS + page // 8 + 4608 + slot, valid, 0, eviction_policy=policy)
    async_copy.async_copy_global_to_shared(
        kv.slice(448, 64, 1),
        KR + (offset // 2 + 224)[:, None] + d[None, :],
        valid[:, None],
        cache_modifier=".cg",
    )
    async_copy.async_copy_global_to_shared(
        q.slice(448, 64, 1), QR + h[:, None] * 64 + d[None, :], cache_modifier=".cg"
    )
    async_copy.commit_group()
    bf_band(Q, QS, K, q, kv, h, offset, valid, scales, layout, policy, 0, 256)
    bf_band(Q, QS, K, q, kv, h, offset, valid, scales, layout, policy, 256, 128)
    bf_band(Q, QS, K, q, kv, h, offset, valid, scales, layout, policy, 384, 64)
    async_copy.wait_group(0)
    hopper.fence_async_shared()
    gl.thread_barrier()
    zero_scores = gl.full((64, BK), 0, gl.float32, layout=gl.constexpr(qkl))
    score = hopper.warpgroup_mma(q, kv.permute((1, 0)), zero_scores, use_acc=False) * (
        scale * 1.4426950408889634
    )
    score = gl.where(
        gl.convert_layout(valid, gl.constexpr(SliceLayout(0, qkl)))[None, :],
        score,
        -float("inf"),
    )
    m = gl.max(score, 1)
    prob = gl.exp2(score - gl.where(m == -float("inf"), 0.0, m)[:, None])
    denom = gl.sum(prob, 1)
    safe = gl.where(denom == 0.0, 1.0, denom)
    p.store(prob.to(gl.bfloat16))
    hopper.fence_async_shared()
    gl.thread_barrier()
    zo = gl.full((64, 512), 0, gl.float32, layout=gl.constexpr(pvl))
    output = hopper.warpgroup_mma(p, kv, zo, use_acc=False)
    publish(Output, row_out, output, 1.0 / safe, 0, 512, SPLITS == 1)
    rows = row_out + gl.arange(0, 64, layout=gl.constexpr(SliceLayout(1, qkl)))
    gl.store(
        L + rows,
        gl.where(
            denom > 0.0,
            m * 0.6931471805599453 + gl.log(safe),
            float("inf") if SPLITS == 1 else -float("inf"),
        ),
    )


@triton.jit
def merge_two(V, L, Output, Lout, H: tl.constexpr):
    batch = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, 512)
    for off in tl.static_range(2):
        h = tl.program_id(1) * 2 + off
        row = batch * H + h
        base = batch * (2 * H) + h
        a = tl.load(L + base)
        b = tl.load(L + base + H)
        va = (a != -float("inf")) & (a != float("inf"))
        vb = (b != -float("inf")) & (b != float("inf"))
        if va & vb:
            m = tl.maximum(a, b)
            wa = tl.exp(a - m)
            wb = tl.exp(b - m)
            den = wa + wb
            value = (
                wa * tl.load(V + base * 512 + dims).to(tl.float32)
                + wb * tl.load(V + (base + H) * 512 + dims).to(tl.float32)
            ) / den
            stat = m + tl.log(den)
        else:
            ix = tl.where(va, base, base + H)
            value = tl.load(V + ix * 512 + dims, va | vb, 0.0).to(tl.float32)
            stat = tl.where(va, a, tl.where(vb, b, float("inf")))
        tl.store(Output + row * 512 + dims, value)
        tl.store(Lout + row, stat)


@triton.jit
def leader(
    Output,
    L,
    sq,
    sr,
    qs,
    sk,
    skr,
    ks,
    sv64,
    sp,
    alpha,
    factor,
    qfull,
    kfull,
    kfree,
    pfull,
    vfull,
    vfree,
    ofull,
    SCALE: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BK: tl.constexpr,
    BUFS: tl.constexpr,
    RING: tl.constexpr,
):
    b, hb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = (b * S + split) * H + hb * 64 + tl.arange(0, 64)
    _, count = split_block_range(split, tl.cdiv(T, BK), S)
    maximum = tl.full((64,), -float("inf"), tl.float32)
    denom = tl.full((64,), 0.0, tl.float32)
    prev = tl.full((64,), 1.0, tl.float32)
    acc = tl.full((64, 64), 0.0, tl.float32)
    tail = tl.full((64, 64), 0.0, tl.float32)
    tle.gpu.barrier_wait(qfull[0])
    qscale = tl.load(tle.gpu.local_ptr(qs))
    for step in range(count):
        buf, cycle = step % BUFS, step // BUFS
        if BUFS == 1:
            tle.gpu.barrier_wait(kfull[0])
        else:
            tle.gpu.barrier_wait(kfull[buf], phaseIdx=cycle)
        logits = tle.gpu.wgmma(
            tle_subslice(sq, [0, 0], [64, 256]),
            tle_subslice(sk.slot(buf), [0, 0], [BK, 256]),
            out_dtype=tl.float32,
            trans_b=True,
        )
        logits = tle.gpu.wgmma(
            tle_subslice(sq, [0, 256], [64, 128]),
            tle_subslice(sk.slot(buf), [0, 256], [BK, 128]),
            logits,
            trans_b=True,
        )
        logits = tle.gpu.wgmma(
            tle_subslice(sq, [0, 384], [64, 64]),
            tle_subslice(sk.slot(buf), [0, 384], [BK, 64]),
            logits,
            trans_b=True,
        )
        rope = tle.gpu.wgmma(
            sr, skr.slot(step % RING), out_dtype=tl.float32, trans_b=True
        )
        logits = tle.gpu.wgmma_wait(0, logits)
        rope = tle.gpu.wgmma_wait(0, rope)
        sc = tl.load(tle.gpu.local_ptr(ks.slot(buf)))
        if RING > BUFS:
            tle.gpu.barrier_arrive(kfree[buf], phaseIdx=cycle)
        else:
            pass
        scores = tl.where(
            sc[None, :] >= 0.0,
            (logits * qscale[:, None] * sc[None, :] + rope)
            * (SCALE * 1.4426950408889634),
            -float("inf"),
        )
        maximum_new = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(maximum_new == -float("inf"), 0.0, maximum_new)
        corr = tl.exp2(maximum - safe)
        p = tl.exp2(scores - safe[:, None])
        denom = denom * corr + tl.sum(p, 1)
        maximum = maximum_new
        weighted = p * tl.maximum(sc, 0.0)[None, :]
        w = tl.max(weighted, 1) / 448.0
        w = tl.where(w > 0.0, w, 1.0)
        normalized = weighted / w[:, None]
        beta = prev * corr / w
        prev = w
        if BK == 64:
            publish_p_fp8_sw64_coupled_stmatrix(sp.slot(buf), normalized)
        else:
            publish_sparse_p128_fp8(sp.slot(0), normalized, sp.slot(1))
        tl.store(tle.gpu.local_ptr(alpha.slot(buf)), beta)
        tle.gpu.barrier_arrive(pfull[0])
        tail = tle.gpu.wgmma(
            p.to(tl.bfloat16), skr.slot(step % RING), tail * corr[:, None]
        )
        if RING == BUFS:
            tail = tle.gpu.wgmma_wait(0, tail)
            if BUFS == 1:
                tle.gpu.barrier_arrive(kfree[0])
            else:
                tle.gpu.barrier_arrive(kfree[buf], phaseIdx=cycle)
        else:
            pass
        tle.gpu.barrier_wait(vfull[0])
        acc *= beta[:, None]
        for half in tl.static_range(BK // 64):
            vbuf = buf * (BK // 64) + half
            acc = tle.gpu.wgmma(
                sp.slot(vbuf),
                sv64.slot(vbuf),
                acc,
                trans_b=True,
            )
        acc = tle.gpu.wgmma_wait(0, acc)
        if RING > BUFS:
            tail = tle.gpu.wgmma_wait(0, tail)
        else:
            pass
        if BUFS == 1 and tl.cdiv(T, BK) > S:
            tle.gpu.barrier_wait(vfree[0])
        else:
            pass
    inv = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    f = inv * prev
    tl.store(tle.gpu.local_ptr(factor), f)
    tle.gpu.barrier_arrive(ofull[0])
    d = tl.arange(0, 64)
    cols = (d // 16) * 16 + (d % 8) * 2 + (d % 16) // 8
    tl.store(
        Output + rows[:, None] * 512 + 384 + cols[None, :],
        acc * f[:, None],
        cache_modifier=".cs" if S == 1 else "",
    )
    tl.store(
        Output + rows[:, None] * 512 + 448 + tl.arange(0, 64)[None, :],
        tail * inv[:, None],
        cache_modifier=".cs" if S == 1 else "",
    )
    tl.store(
        L + rows,
        tl.where(
            denom > 0.0,
            maximum * 0.6931471805599453 + tl.log(denom),
            float("inf") if S == 1 else -float("inf"),
        ),
    )


@triton.jit
def follower(
    Q,
    QR,
    QS,
    Flags,
    RepairState,
    sq,
    sr,
    qs,
    qfull,
    Output,
    sk,
    sv,
    sv128,
    sv64,
    sp,
    alpha,
    factor,
    kfull,
    kfree,
    pfull,
    vfull,
    vfree,
    ofull,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BK: tl.constexpr,
    BUFS: tl.constexpr,
):
    b, hb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = (b * H + tl.program_id(1) * 64) + tl.arange(0, 64)
    d = tl.arange(0, 512)
    if S > 1 and T > 640:
        signed_scale = tl.load(QS + h)
        mx = tl.abs(signed_scale)
        qrepair = tl.max((signed_scale < 0.0).to(tl.int32), 0)
        tl.store(
            tle.gpu.local_ptr(sq),
            tl.load(
                Q + h[:, None] * 448 + d[None, :],
                d[None, :] < 448,
                0.0,
                eviction_policy="evict_first",
            ),
        )
    else:
        g = tl.arange(0, 8)
        scales = tl.load(QS + h[:, None] * 7 + g[None, :], g[None, :] < 7, 0.0)
        mx = tl.max(scales, 1)
        mn = tl.min(tl.where(scales > 0.0, scales, float("inf")), 1)
        qrepair = tl.max((mx > 4.0 * mn).to(tl.int32), 0)
        cols128 = tl.arange(0, 128)
        inv = 1.0 / tl.where(mx > 0.0, mx, 1.0)
        for pair in tl.static_range(4):
            dims = pair * 128 + cols128
            qpart = tl.load(
                Q + h[:, None] * 448 + dims[None, :],
                dims[None, :] < 448,
                0.0,
                eviction_policy="evict_first",
            )
            qleft = tl.load(QS + h * 7 + 2 * pair) * inv
            if pair < 3:
                qright = tl.load(QS + h * 7 + 2 * pair + 1) * inv
            else:
                qright = 0.0
            ratio = tl.where(cols128[None, :] < 64, qleft[:, None], qright[:, None])
            qnorm = (qpart.to(tl.float32) * ratio).to(tl.float8e4nv)
            tl.store(
                tle.gpu.local_ptr(tle_subslice(sq, [0, pair * 128], [64, 128])),
                qnorm,
            )
    tl.store(tle.gpu.local_ptr(qs), mx)
    tl.store(
        tle.gpu.local_ptr(sr),
        tl.load(
            QR + h[:, None] * 64 + tl.arange(0, 64)[None, :],
            eviction_policy="evict_first",
        ),
    )
    tle.gpu.barrier_arrive(qfull[0])
    tl.atomic_or(Flags + b, 1, mask=qrepair != 0, sem="relaxed")
    tl.atomic_or(RepairState, 1, mask=qrepair != 0, sem="relaxed")
    _, count = split_block_range(split, tl.cdiv(T, BK), S)
    a1 = tl.full((64, 128), 0.0, tl.float32)
    a = tl.full((64, 128), 0.0, tl.float32)
    c = tl.full((64, 128), 0.0, tl.float32)
    for step in range(count):
        buf, cycle = step % BUFS, step // BUFS
        if BUFS == 1:
            tle.gpu.barrier_wait(kfull[0])
        else:
            tle.gpu.barrier_wait(kfull[buf], phaseIdx=cycle)
        for half in tl.static_range(BK // 64):
            for group in tl.static_range(4):
                src = tle_subslice(sk.slot(buf), [half * 64, group * 128], [64, 128])
                for warp_half in tl.static_range(1 if group == 3 else 2):
                    sparse_fp8_transpose_values(
                        src,
                        (sv if group < 2 else sv128 if group == 2 else sv64).slot(
                            buf * (BK // 64) + half
                        ),
                        group * 128 if group < 2 else 0,
                        warp_half * 4,
                    )
        tl.inline_asm_elementwise(
            "{ fence.proxy.async.shared::cta; mov.b32 $0, $1; }",
            "=r,r",
            [tl.arange(0, 128)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tle.gpu.barrier_arrive(vfull[0])
        if BUFS == 1:
            tle.gpu.barrier_arrive(kfree[0])
        else:
            tle.gpu.barrier_arrive(kfree[buf], phaseIdx=cycle)
        tle.gpu.barrier_wait(pfull[0])
        beta = tl.load(tle.gpu.local_ptr(alpha.slot(buf)))
        a *= beta[:, None]
        a1 *= beta[:, None]
        c *= beta[:, None]
        for half in tl.static_range(BK // 64):
            vbuf = buf * (BK // 64) + half
            a = tle.gpu.wgmma(
                sp.slot(vbuf),
                tle_subslice(sv.slot(vbuf), [0, 0], [128, 64]),
                a,
                trans_b=True,
            )
            a1 = tle.gpu.wgmma(
                sp.slot(vbuf),
                tle_subslice(sv.slot(vbuf), [128, 0], [128, 64]),
                a1,
                trans_b=True,
            )
            c = tle.gpu.wgmma(
                sp.slot(vbuf),
                sv128.slot(vbuf),
                c,
                trans_b=True,
            )
        a1 = tle.gpu.wgmma_wait(0, a1)
        a = tle.gpu.wgmma_wait(0, a)
        c = tle.gpu.wgmma_wait(0, c)
        if BUFS == 1 and tl.cdiv(T, BK) > S:
            tle.gpu.barrier_wait(vfree[0])
        else:
            pass
    tle.gpu.barrier_wait(ofull[0])
    f = tl.load(tle.gpu.local_ptr(factor))
    sparse_fp8_publish_output(a, f, Output, b * S + split, hb, H, 0, S == 1)
    sparse_fp8_publish_output(a1, f, Output, b * S + split, hb, H, 128, S == 1)
    sparse_fp8_publish_output(c, f, Output, b * S + split, hb, H, 256, S == 1)


@triton.jit
def model1_fp8(
    PackedRope,
    Q,
    QR,
    QS,
    E,
    Idx,
    Output,
    L,
    Flags,
    RepairState,
    Normalized,
    Scale,
    N,
    T: tl.constexpr,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    S: tl.constexpr,
    BK: tl.constexpr,
    BUFS: tl.constexpr,
    PREPACK: tl.constexpr,
):
    K = E.to(tl.pointer_type(tl.float8e4nv))
    KR = PackedRope if PREPACK else E.to(tl.pointer_type(tl.bfloat16))
    sq = tle.gpu.alloc([64, 512], tl.float8e4nv, scope=tle.gpu.smem)
    sr = tle.gpu.alloc([64, 64], tl.bfloat16, scope=tle.gpu.smem)
    qs = tle.gpu.alloc([64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    sk = tle.gpu.alloc([BUFS, BK, 512], tl.float8e4nv, scope=tle.gpu.smem)
    RING: tl.constexpr = 4 if BUFS == 2 and not PREPACK else BUFS
    skr = tle.gpu.alloc([RING, BK, 64], tl.bfloat16, scope=tle.gpu.smem)
    ks = tle.gpu.alloc(
        [BUFS, BK], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    sv = tle.gpu.alloc([BUFS * (BK // 64), 256, 64], tl.float8e4nv, scope=tle.gpu.smem)
    sv128 = tle.gpu.alloc(
        [BUFS * (BK // 64), 128, 64], tl.float8e4nv, scope=tle.gpu.smem
    )
    sv64 = tle.gpu.alloc([BUFS * (BK // 64), 64, 64], tl.float8e4nv, scope=tle.gpu.smem)
    sp = tle.gpu.alloc([BUFS * (BK // 64), 64, 64], tl.float8e4nv, scope=tle.gpu.smem)
    alpha = tle.gpu.alloc(
        [BUFS, 64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    factor = tle.gpu.alloc(
        [64], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    qfull = sparse_named_barriers(1, 256, 16)
    if BUFS == 1:
        kfull = sparse_named_barriers(1, 512, 18)
        kfree = sparse_named_barriers(1, 512, 19)
    else:
        kfull = tle.gpu.alloc_barriers(BUFS, arrive_count=1)
        kfree = tle.gpu.alloc_barriers(BUFS, arrive_count=2, init=tle.gpu.READY)
    vfull = sparse_named_barriers(1, 256, 20)
    pfull = sparse_named_barriers(1, 256, 21)
    vfree = sparse_named_barriers(1, 256, 23)
    ofull = sparse_named_barriers(1, 256, 17)
    tle.gpu.warp_specialize(
        [
            (
                leader,
                (
                    Output,
                    L,
                    sq,
                    sr,
                    qs,
                    sk,
                    skr,
                    ks,
                    sv64,
                    sp,
                    alpha,
                    factor,
                    qfull,
                    kfull,
                    kfree,
                    pfull,
                    vfull,
                    vfree,
                    ofull,
                    SCALE,
                    T,
                    H,
                    S,
                    BK,
                    BUFS,
                    RING,
                ),
            ),
            (
                producer,
                (
                    K,
                    KR,
                    Idx,
                    Flags,
                    RepairState,
                    Normalized,
                    Scale,
                    sk,
                    skr,
                    ks,
                    kfull,
                    kfree,
                    N,
                    T,
                    STRIDE,
                    S,
                    BK,
                    BUFS,
                    PREPACK,
                ),
            ),
            (
                follower,
                (
                    Q,
                    QR,
                    QS,
                    Flags,
                    RepairState,
                    sq,
                    sr,
                    qs,
                    qfull,
                    Output,
                    sk,
                    sv,
                    sv128,
                    sv64,
                    sp,
                    alpha,
                    factor,
                    kfull,
                    kfree,
                    pfull,
                    vfull,
                    vfree,
                    ofull,
                    T,
                    H,
                    S,
                    BK,
                    BUFS,
                ),
            ),
        ],
        [8, 4],
        [40 if S > 1 and BK == 64 else 48, 248],
    )


@triton.jit
def repair_heads(
    Q,
    QR,
    QS,
    K,
    E,
    KR,
    Idx,
    Output,
    L,
    b,
    group,
    N,
    T: tl.constexpr,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
):
    rows = b * H + group * 16 + tl.arange(0, 16)
    d = tl.arange(0, 512)
    q = tl.load(Q + rows[:, None] * 448 + d[None, :], d[None, :] < 448, 0.0)
    qs = tl.load(QS + rows[:, None] * 7 + d[None, :] // 64, d[None, :] < 448, 0.0)
    qr = tl.load(QR + rows[:, None] * 64 + (d % 64)[None, :], d[None, :] >= 448, 0.0)
    qbf = tl.where(d[None, :] < 448, q.to(tl.float32) * qs, qr).to(tl.bfloat16)
    m = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.full((16,), 0.0, tl.float32)
    acc = tl.full((16, 512), 0.0, tl.float32)
    for it in range(tl.cdiv(T, 32)):
        pos = it * 32 + tl.arange(0, 32)
        ids = tl.load(
            Idx + b.to(tl.int64) * T + pos, pos < T, -1, eviction_policy="evict_first"
        )
        valid = (ids >= 0) & (ids < N)
        ids = tl.where(valid, ids, 0)
        page = (ids >> 6).to(tl.int64) * STRIDE
        slot = ids & 63
        base = page + slot * 576
        e = tl.load(E + (page + 36864 + slot * 8)[:, None] + (d // 64)[None, :]).to(
            tl.int32
        )
        ks = tl.exp2((e - 127).to(tl.float32))
        raw = tl.load(
            K + base[:, None] + d[None, :], valid[:, None] & (d[None, :] < 448), 0.0
        )
        r = tl.load(
            KR + (base // 2 + 224)[:, None] + (d % 64)[None, :],
            valid[:, None] & (d[None, :] >= 448),
            0.0,
        )
        kv = tl.where(d[None, :] < 448, raw.to(tl.float32) * ks, r).to(tl.bfloat16)
        scores = tl.where(
            valid[None, :],
            tl.dot(qbf, tl.trans(kv)) * (SCALE * 1.4426950408889634),
            -float("inf"),
        )
        nm = tl.maximum(m, tl.max(scores, 1))
        safe = tl.where(nm == -float("inf"), 0.0, nm)
        p = tl.exp2(scores - safe[:, None])
        corr = tl.exp2(m - safe)
        acc = tl.dot(p.to(tl.bfloat16), kv, acc * corr[:, None])
        denominator = denominator * corr + tl.sum(p, 1)
        m = nm
    tl.store(
        Output + rows[:, None] * 512 + d[None, :],
        acc * tl.where(denominator > 0.0, 1.0 / denominator, 0.0)[:, None],
    )
    tl.store(
        L + rows,
        tl.where(
            denominator > 0.0,
            m * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        ),
    )


@triton.jit
def repair(
    Q,
    QR,
    QS,
    K,
    E,
    KR,
    Idx,
    Output,
    L,
    Flags,
    RepairState,
    B,
    N,
    T: tl.constexpr,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
):
    # The usual scale range needs no second attention pass.
    if tl.load(RepairState) == 0:
        return
    grid = tl.num_programs(0)
    for b in range(tl.program_id(0), B, grid):
        if tl.load(Flags + b):
            for group in range(H // 16):
                repair_heads(
                    Q,
                    QR,
                    QS,
                    K,
                    E,
                    KR,
                    Idx,
                    Output,
                    L,
                    b,
                    group,
                    N,
                    T,
                    H,
                    STRIDE,
                    SCALE,
                )
        else:
            pass


@gluon.jit
def merge(
    o_part, lse_part, out, lse, H: gl.constexpr, SPLITS: gl.constexpr, D: gl.constexpr
):
    # A warp covers four splits and eight column vectors. Reductions use
    # warp shuffles, including for padded and completely empty splits.
    layout: gl.constexpr = BlockedLayout([1, 8], [4, 8], [1, 2], [1, 0])
    ss = gl.arange(
        0, triton.next_power_of_2(SPLITS), layout=gl.constexpr(SliceLayout(1, layout))
    )
    d = gl.program_id(2) * D + gl.arange(
        0, D, layout=gl.constexpr(SliceLayout(0, layout))
    )
    batch = gl.program_id(0).to(gl.int64)
    h = gl.program_id(1)
    rows = (batch * SPLITS + ss) * H + h
    stats = gl.load(lse_part + rows, ss < SPLITS, -float("inf"))
    stats = gl.where(stats == float("inf"), -float("inf"), stats)
    m = gl.max(stats, 0)
    empty = m == -float("inf")
    w = gl.exp(stats - gl.where(empty, 0.0, m))
    denom = gl.sum(w, 0)
    v = gl.load(
        o_part + rows[:, None] * 512 + d[None, :],
        ss[:, None] < SPLITS,
        0.0,
        eviction_policy="evict_first",
    )
    merged_value = gl.sum(v.to(gl.float32) * w[:, None], 0) / gl.where(
        empty, 1.0, denom
    )
    gl.store(out + (batch * H + h) * 512 + d, merged_value.to(gl.bfloat16))
    gl.store(
        lse + batch * H + h,
        gl.where(empty, float("inf"), m + gl.log(denom)),
        mask=gl.program_id(2) == 0,
    )


def model1_attention(
    q: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    packed_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    tokens: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, width = q.shape
    pages = packed_cache.shape[0]
    topk = indices.shape[-1]
    if batch == 0 or heads not in (64, 128) or width != 448 or topk == 0:
        raise ValueError(
            "Model1 requires nonempty queries, 64/128 heads and 448 FP8 dims"
        )
    if q.dtype != torch.float8_e4m3fn or not q.is_contiguous():
        raise ValueError("Query content must be contiguous E4M3")
    if (
        q_rope.shape != (batch, heads, 64)
        or q_rope.dtype != torch.bfloat16
        or not q_rope.is_contiguous()
    ):
        raise ValueError("Query RoPE must be contiguous BF16 [batch, heads, 64]")
    if (
        q_scale.shape != (batch, heads, 7)
        or q_scale.dtype != torch.float32
        or not q_scale.is_contiguous()
    ):
        raise ValueError("Query scales must be contiguous FP32 [batch, heads, 7]")
    if packed_cache.shape != (pages, 64, 1, 584) or packed_cache.dtype != torch.uint8:
        raise ValueError("Expected native V4 cache pages [pages, 64, 1, 584]")
    if (
        packed_cache.stride()[1:] != (584, 584, 1)
        or packed_cache.stride(0) % 16 != 0
        or packed_cache.data_ptr() % 16 != 0
        or packed_cache.stride(0) < 64 * 584
    ):
        raise ValueError(
            "V4 cache page data must be 16B-aligned and contiguous within a page"
        )
    if (
        indices.shape != (batch, 1, topk)
        or indices.dtype != torch.int32
        or not indices.is_contiguous()
    ):
        raise ValueError("Expected contiguous int32 indices [batch, 1, topk]")
    if not 0 < tokens <= pages * 64:
        raise ValueError("Token count is outside the allocated cache")
    if q.device.type != "cuda" or any(
        quantized_values.device != q.device
        for quantized_values in (q_rope, q_scale, packed_cache, indices)
    ):
        raise ValueError("Model1 tensors must share a CUDA device")

    fp8_keys, exponent_words, bf16_keys, page_stride, _, _ = cache_views(packed_cache)
    sm = torch.cuda.get_device_properties(q.device).multi_processor_count
    work = batch * (heads // 64)
    sm_limit = sm // work
    block = 128 if 128 < topk <= 640 and 1 < triton.cdiv(topk, 128) <= sm_limit else 64
    num_blocks = triton.cdiv(topk, block)
    target = max(1, sm_limit)
    exact = topk <= 128 and work <= 64
    if exact:
        block = 64 if topk <= 64 else 128
        splits = 1
    elif num_blocks >= 32 and sm < work < 4 * sm and 0 < work % sm <= sm // 4:
        splits = 2
    else:
        splits = (
            1
            if topk <= 128
            else triton.cdiv(num_blocks, triton.cdiv(num_blocks, target))
        )
    # Bound FP8 MMA accumulation before the FP32 split reduction.
    splits = max(splits, triton.cdiv(topk, 4096))
    out = torch.empty((batch, heads, 512), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((batch, heads), dtype=torch.float32, device=q.device)
    part = out if splits == 1 else out.new_empty((batch, splits, heads, 512))
    stats = lse if splits == 1 else lse.new_empty((batch, splits, heads))
    if exact:
        one_block[(batch, heads // 64, splits)](
            q,
            q_rope,
            q_scale,
            fp8_keys,
            exponent_words,
            bf16_keys,
            indices,
            part,
            stats,
            scale,
            tokens,
            topk,
            page_stride,
            heads,
            splits,
            block,
            num_warps=8,
            num_stages=1,
        )
        return out, lse
    repair_flags = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    flags, repair_state = repair_flags[1:], repair_flags[:1]
    exponent_bytes = exponent_words.view(torch.uint8)
    prepack = topk > 128 and batch * topk >= 10 * tokens
    if splits > 1 and topk > 640:
        query = torch.empty_like(q)
        query_magnitude = q_scale.new_empty((batch, heads))
        normalize_query[(batch, heads // 4)](
            q, q_scale, query, query_magnitude, heads, num_warps=4
        )
    else:
        query, query_magnitude = q, q_scale
    if prepack:
        normalized = q.new_empty((tokens, 448))
        token_scale = q_scale.new_empty((tokens,))
        rope_cache = q_rope.new_empty((tokens, 64))
        normalize_cache[(triton.cdiv(tokens, 32),)](
            fp8_keys,
            exponent_bytes,
            normalized,
            token_scale,
            rope_cache,
            tokens,
            page_stride,
            num_warps=8,
        )
    else:
        normalized, token_scale, rope_cache = fp8_keys, q_scale, bf16_keys
    model1_fp8[(batch, heads // 64, splits)](
        rope_cache,
        query,
        q_rope,
        query_magnitude,
        exponent_bytes,
        indices,
        part,
        stats,
        flags,
        repair_state,
        normalized,
        token_scale,
        tokens,
        topk,
        heads,
        page_stride,
        scale,
        splits,
        block,
        128 // block,
        prepack,
        ptx_options="--register-usage-level=2",
        num_warps=4,
        num_stages=1,
    )
    if splits == 1:
        pass  # The attention kernel wrote the final output directly.
    elif splits == 2:
        merge_two[(batch, heads // 2)](part, stats, out, lse, heads, num_warps=2)
    else:
        d = 128 if batch <= 4 else 512
        merge[(batch, heads, 512 // d)](
            part,
            stats,
            out,
            lse,
            heads,
            splits,
            d,
            num_warps=2,
        )
    repair[(min(batch, 64),)](
        q,
        q_rope,
        q_scale,
        fp8_keys,
        exponent_bytes,
        bf16_keys,
        indices,
        out,
        lse,
        flags,
        repair_state,
        batch,
        tokens,
        topk,
        heads,
        page_stride,
        scale,
        num_warps=8,
        num_stages=1,
    )
    return out, lse

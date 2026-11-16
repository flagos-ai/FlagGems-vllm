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

"""Dense (non-causal) FlashAttention forward, split out from fa_naive.

Separated from causal so the two can be tuned independently: the dense kernel
has no mask, no diagonal, and every Q tile walks the full S/BLOCK_N KV blocks,
so its tile-shape tradeoff is different. Round 4's sweep found dense alone
benefits from a wide-N tile (64/256) at large S, which causal cannot use.

Shares `build_tile_order`, `_max_nan` and `_COMPILE_OPTS` with fa_naive.
"""

import torch
import triton
import triton.language as tl

from .fa_naive import (
    _COMPILE_OPTS,
    _max_nan,
    build_tile_order,
    check_bnsd,
    validate_shape,
)


@triton.jit
def _attn_fwd_dense_grp2(
    Q,
    K,
    V,
    Out,
    Tiles,
    ntiles,
    sm_scale,
    sQb,
    sQh,
    sQs,
    sKb,
    sKh,
    sKs,
    sOb,
    sOh,
    sOs,
    B,
    Hq,
    S,
    GQA: tl.constexpr,
    NUM_CORES: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRESCALED: tl.constexpr,
):
    """GROUP=2 grouped-accumulation kernel. Valid when BLOCK_M==64 and S%(2*BLOCK_N)==0.

    Processes two adjacent KV blocks per loop iteration. Within each group:
      pass 1  — compute qk0 and qk1, derive the group-joint max m_new; acc untouched.
      rescale — acc = acc * exp(m_i - m_new); one touch per group instead of two.
      pass 2  — dot(p0,vb0,acc) then dot(p1,vb1,acc) back-to-back; acc not touched
                between the two dots, so the compiler can run both PV matmuls
                without a Vector-side sync_block barrier in between.

    The two PV matmuls running consecutively without a Vector rescale between them
    are what the bishengir compiler needs to fold the two fixpipe transfers into one
    wider one (64x128 instead of 32x128+32x128) and to let Cube keep running while
    Vector drains the first PV result. This halves the Cube/Vector sync count per
    group and is the source of the measured 1.11-1.29x speedup over the per-block
    baseline (see grouped_accumulation_optimization.md).

    Implementation note: cannot use `for j in range(GROUP): ...` with shared
    loop-carried scalars — bishengir emits 'hivm.hir.vadd operands must have same
    rank' on that pattern. The two iterations are written out as explicit flat code.
    """
    nblk_m = S // BLOCK_M
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_m0 = tl.arange(0, BLOCK_M)
    GN = 2 * BLOCK_N

    for t in range(pid, ntiles, NUM_CORES):
        blk = tl.load(Tiles + t)
        hz = blk // nblk_m
        mi = blk % nblk_m
        b = hz // Hq
        h = hz % Hq
        hkv = h // GQA

        q_base = Q + b * sQb + h * sQh
        k_base = K + b * sKb + hkv * sKh
        v_base = V + b * sKb + hkv * sKh
        o_base = Out + b * sOb + h * sOh

        offs_m = mi * BLOCK_M + offs_m0
        q = tl.load(q_base + offs_m[:, None] * sQs + offs_d[None, :])

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        for g in range(0, S, GN):
            # pass 1: compute both QK scores and derive group-joint max
            kn0 = g + offs_n
            kb0 = tl.load(k_base + kn0[:, None] * sKs + offs_d[None, :])
            qk0 = tl.dot(q, tl.trans(kb0))
            if not PRESCALED:
                qk0 = qk0 * sm_scale
            m_g = tl.reduce(qk0, 1, _max_nan)

            kn1 = g + BLOCK_N + offs_n
            kb1 = tl.load(k_base + kn1[:, None] * sKs + offs_d[None, :])
            qk1 = tl.dot(q, tl.trans(kb1))
            if not PRESCALED:
                qk1 = qk1 * sm_scale
            m_g = _max_nan(m_g, tl.reduce(qk1, 1, _max_nan))

            m_new = _max_nan(m_i, m_g)

            # one rescale per group: acc untouched between the two PV dots below
            alpha = tl.exp(m_i - m_new)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha

            # pass 2: two consecutive PV matmuls with no Vector touch between them
            p0 = tl.exp(qk0 - m_new[:, None])
            l_i = l_i + tl.sum(p0, 1)
            vb0 = tl.load(v_base + kn0[:, None] * sKs + offs_d[None, :])
            acc = tl.dot(p0.to(vb0.dtype), vb0, acc)

            p1 = tl.exp(qk1 - m_new[:, None])
            l_i = l_i + tl.sum(p1, 1)
            vb1 = tl.load(v_base + kn1[:, None] * sKs + offs_d[None, :])
            acc = tl.dot(p1.to(vb1.dtype), vb1, acc)

            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(
            o_base + offs_m[:, None] * sOs + offs_d[None, :],
            acc.to(Out.dtype.element_ty),
        )


@triton.jit
def _attn_fwd_dense(
    Q,
    K,
    V,
    Out,
    Tiles,
    ntiles,
    sm_scale,
    sQb,
    sQh,
    sQs,
    sKb,
    sKh,
    sKs,
    sOb,
    sOh,
    sOs,
    B,
    Hq,
    S,
    GQA: tl.constexpr,
    NUM_CORES: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRESCALED: tl.constexpr,
):
    """No mask, no peel: every tile walks all S/BLOCK_N KV blocks."""
    nblk_m = S // BLOCK_M
    pid = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_m0 = tl.arange(0, BLOCK_M)

    for t in range(pid, ntiles, NUM_CORES):
        blk = tl.load(Tiles + t)
        hz = blk // nblk_m
        mi = blk % nblk_m
        b = hz // Hq
        h = hz % Hq
        hkv = h // GQA

        q_base = Q + b * sQb + h * sQh
        k_base = K + b * sKb + hkv * sKh
        v_base = V + b * sKb + hkv * sKh
        o_base = Out + b * sOb + h * sOh

        offs_m = mi * BLOCK_M + offs_m0
        q = tl.load(q_base + offs_m[:, None] * sQs + offs_d[None, :])

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        for start_n in range(0, S, BLOCK_N):
            kn = start_n + offs_n
            kb = tl.load(k_base + kn[:, None] * sKs + offs_d[None, :])
            qk = tl.dot(q, tl.trans(kb))
            if not PRESCALED:
                qk = qk * sm_scale
            m_ij = _max_nan(m_i, tl.reduce(qk, 1, _max_nan))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            # V after the Q@K^T dot, not alongside K: its MTE2 then overlaps
            # the cube work instead of queueing behind K's load.
            vb = tl.load(v_base + kn[:, None] * sKs + offs_d[None, :])
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(vb.dtype), vb, acc)
            m_i = m_ij

        acc = acc / l_i[:, None]
        tl.store(
            o_base + offs_m[:, None] * sOs + offs_d[None, :],
            acc.to(Out.dtype.element_ty),
        )


# Base config list. (64, 256) wins only on large dense at D=64.
# Autotune gates on S >= 4096.
#
# (64, 128) is absent on purpose: it is the only tile that dispatches to the
# GROUP=2 kernel, and GROUP=2 does not pay off at either supported dim.
#
#   D=64  : loses 8-14% across all shapes (2026-11-10, card 7, 120 shapes,
#           mean -10.0%).
#   D=128 : **the "beneficial at D>=128" note that used to sit here is wrong.**
#           Measured 2026-11-13 on card 7, median of 20, interleaved:
#             - grouping does work at a fixed tile: (64,128) grp2 beats
#               (64,128) plain by 1.07-1.23x, matching the kernel docstring;
#             - but BLOCK_M=64 itself loses more than grouping wins, so
#               (64,128) grp2 is 0.79-0.85x of (128,128) plain on every shape
#               measured (B1/Hq8/S4096, B1/Hq32/S4096, B4/Hq16/S2048,
#               B1/Hq16/S8192; only B1/Hq8/S1024 is near parity at 0.99x).
#           Lifting grouping onto the good tile does not fit: grp2 at (128,128)
#           keeps two score tiles live and overflows UB by 8 KiB
#           ("requires 2097152 bits while 2031616 bits available").
#
# So the grouped kernel stays reachable only by passing block_m=64,
# block_n=128 explicitly. Keep it for the UB-budget question above -- if a
# future tile or dtype frees ~8 KiB, grouping at BLOCK_M=128 is the thing to
# retest. See grouped_accumulation_optimization.md.
_CONFIGS_BASE = ((128, 128), (64, 256))
_cache: dict = {}


def flash_attention_fwd_dense(
    q, k, v, block_m=None, block_n=None, num_cores=32, autotune=True, **opts
):
    check_bnsd(q, k, v)
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]

    explicit = block_m is not None or block_n is not None
    if block_m is None:
        block_m = 128
    if block_n is None:
        block_n = 128

    validate_shape(q, k, v, block_m, block_n, causal=False)

    if autotune and not explicit and S >= 4096 and len(_CONFIGS_BASE) > 1:
        key = (B, Hq, Hkv, S, D, str(q.device))
        if key not in _cache:

            def _probe(bm, bn):
                return flash_attention_fwd_dense(
                    q,
                    k,
                    v,
                    block_m=bm,
                    block_n=bn,
                    num_cores=num_cores,
                    autotune=False,
                    **opts,
                )

            _cache[key] = _tune(q, k, v, S, _probe, _CONFIGS_BASE)
        block_m, block_n = _cache[key]

    out = torch.empty_like(q)
    tiles = build_tile_order(B, Hq, S, block_m, False, q.device)

    sm_scale = (1.0 / D) ** 0.5
    prescale = S >= 4096
    qs = q * sm_scale if prescale else q
    if not qs.is_contiguous():
        qs = qs.contiguous()

    launch_opts = dict(_COMPILE_OPTS)
    launch_opts.update(opts)

    # GROUP=2 path: BM=64 + S divisible by 2*BN=256. The grp2 kernel runs two
    # adjacent KV blocks per iteration so acc is only rescaled once per group
    # instead of once per block, halving the Cube/Vector sync count and letting
    # the two back-to-back PV matmuls run without a Vector barrier between them.
    use_grp2 = block_m == 64 and block_n == 128 and S % 256 == 0
    kernel = _attn_fwd_dense_grp2 if use_grp2 else _attn_fwd_dense

    # `qs`, not `q`: see validate_shape.
    kernel[(num_cores,)](
        qs,
        k,
        v,
        out,
        tiles,
        tiles.numel(),
        sm_scale,
        qs.stride(0),
        qs.stride(1),
        qs.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B,
        Hq,
        S,
        GQA=Hq // Hkv,
        NUM_CORES=num_cores,
        D=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        PRESCALED=prescale,
        **launch_opts,
    )
    return out


def _tune(q, k, v, S, launch, configs, warmup=3, rep=7):
    """Pick (block_m, block_n) for one shape by measuring both.

    Host-side rather than `@triton.autotune` because `Tiles` is built on the
    host and keyed to `block_m`: a decorator sweeping BLOCK_M as a
    `tl.constexpr` would hand the kernel a tile list built for a *different*
    block size, which is silently wrong rather than slow. `pre_hook` cannot
    rescue it either -- `ntiles` changes with `block_m`, so the fix needs a
    differently-sized positional tensor, not an in-place mutation.

    Gains are 1.03-1.07x, so the tuner must stay cheaper than what it wins:
    result cached per shape key, median of 7.
    """
    import time as _time

    best, best_ms = None, float("inf")
    for bm, bn in configs:
        if S % bm or S % bn:
            continue
        try:
            for _ in range(warmup):
                launch(bm, bn)
            torch.npu.synchronize()
            lat = []
            for _ in range(rep):
                torch.npu.synchronize()
                t0 = _time.perf_counter()
                launch(bm, bn)
                torch.npu.synchronize()
                lat.append(_time.perf_counter() - t0)
            lat.sort()
            ms = lat[len(lat) // 2]
        except Exception:
            continue
        if ms < best_ms:
            best, best_ms = (bm, bn), ms
    return best or (128, 128)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Retains the gated-delta-rule formulation from flash-linear-attention (MIT):
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""Ascend 950DT (A5 / arch35) triton-ascend GDN: chunk prefill and recurrent decode.

Two entry points, both self-contained:

- :func:`chunk_gdn_950dt` -- chunked prefill. One kernel per chunk stage, all
  state carried in GM between stages so no stage depends on UB residency.
- :func:`fused_recurrent_gdn_950dt` -- single-token recurrent decode, one
  sequential pass over T with the recurrent state held in registers.

950DT notes that shaped the design:

- 64 AI vector cores, 32 cube cores. The grid is capped at the vector-core
  count so a tile maps to exactly one core and the state update stays
  single-writer.
- ``triton.experimental.tle`` is absent on this backend, so nothing here
  gates on it; there is no TLE path to fall back from.
- arch35 lacks the arch22 repeat-stride vector blocks, so reductions are
  written as plain ``tl.sum`` over a 2D tile and left to the compiler rather
  than hand-mapped onto vector primitives.
- Backward is deliberately not implemented: Triton GDN backward is reported
  to produce NaNs on Ascend950 (ByteDance-Seed/VeOmni#1369). Forward only,
  and the unsupported path raises instead of degrading.

Numerical note for callers -- normalize q/k. The gated delta rule
``S <- S*exp(g) + beta*(v - S k) k^T`` is only stable for unit-norm k. With
DK=128 and k whose entries are O(1), the feedback ``S k`` sums 128 products,
giving a gain of order 100, and the state diverges: measured on this device,
the state grows from ~10 to ~4e6 over eight tokens, while unit-norm k keeps
it contractive (~0.76 over the same tokens). The same normalization also
conditions the chunk path's WY solve, which inverts
``I + tril(diag(beta) K K^T, -1)`` -- ``cond`` ~2e4 for row norms ~7 versus
~1.2 for unit rows.

So ``use_qk_l2norm_in_kernel=True`` is the normal mode, not an option: on
unnormalized inputs every GDN implementation produces garbage (the aclnn
AscendC op returns NaN on such data). This kernel does not force it, because
callers that already normalize upstream should not pay for it twice.

No torch compute operators appear in the production path, per
``NO_TORCH_COMPUTE_FALLBACK``. Only allocation (``torch.empty``/``zeros``),
copies (``copy_``/``contiguous``) and metadata reads are used.
"""

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl

try:  # pragma: no cover - backend probe, exercised on NPU only
    from triton.backends.ascend.driver import NPUUtils
except ImportError:  # pragma: no cover - non-Ascend import of this module
    NPUUtils = None


# 950DT: 64 AI vector cores. Cache per device; NPUUtils() reloads an
# extension on every construction, even when the value is already known.
@lru_cache(maxsize=None)
def _vector_core_count(device_index):
    if NPUUtils is None:
        return 64
    try:
        return int(NPUUtils().get_aivector_core_num())
    except Exception:
        return 64


CHUNK_SIZE = 64
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# bishengir reports a UB overflow as a compilation error, not a typed one, so
# the retry has to match on the message. Keep the tuple broad and the message
# test narrow, so an unrelated compile failure is re-raised rather than
# silently retried at a smaller tile.
try:  # pragma: no cover - import shape differs across triton builds
    from triton.compiler.errors import CompilationError, MLIRCompilationError

    _UB_OVERFLOW_ERRORS = (MLIRCompilationError, CompilationError, RuntimeError)
except ImportError:  # pragma: no cover
    _UB_OVERFLOW_ERRORS = (RuntimeError,)


def _is_ub_overflow(exc):
    text = str(exc)
    return "ub overflow" in text or "buildFinalHIVMPipelines" in text


def _bv_candidates(dv, tiles_per_bv=None, cores=None):
    """Value-tile sizes to try, largest first.

    Largest-first is the measured choice, not the obvious one. The chunk
    output kernel launches ``n * HV * ceil(DV/BV)`` programs, so at n=1,
    HV=8, DV=128 a BV of 32 fills only 32 of the 64 vector cores while BV=16
    fills all 64. Ranking by occupancy was tried and is *worse*: at s1024 it
    picks BV=16 and costs 367 us against 286 us for BV=32, and at s4096
    1100 us against 745 us. Halving BV halves the K extent of the state
    tile, and the resulting tl.dot inefficiency outweighs the extra
    parallelism. So prefer the widest tile that compiles and accept the idle
    cores.

    ``tiles_per_bv``/``cores`` are kept in the signature because the
    occupancy numbers are worth being able to re-derive, but they no longer
    change the order.
    """
    del tiles_per_bv, cores
    sizes = [bv for bv in (64, 32, 16, 8) if bv <= dv]
    return sizes or [dv]


# Remember which tile actually compiled, keyed by every property that moves
# UB occupancy: head dims, head grouping, the l2-norm flag, the element dtype
# and whether the varlen path is taken. Leaving dtype out of this key made a
# tile probed in bfloat16 get reused for float32, where it overflows UB and
# the launch fails -- a bug that only appeared when both dtypes ran in one
# process.
#
# This cache is not an optimization, it is required for correctness of the
# probe: triton caches *successful* compilations, not failed ones, so a
# rejected tile is recompiled (and re-rejected) on every call. Without this,
# probing BV=64 before settling on 32 costs ~1.7 s per invocation -- measured
# as 1707 ms/call at s1024 versus 768 us once the tile is remembered.
_BV_CACHE = {}


def _resolve_bv(kernel_tag, dv, signature, launch, tiles_per_bv=None, cores=None):
    """Return a working BV for this shape, probing at most once.

    ``launch(bv)`` must run the kernel(s) for that tile and raise on a UB
    overflow. Any other failure propagates: a tile search must not turn an
    unrelated compile error into a silent fallback.
    """
    key = (kernel_tag, signature)
    cached = _BV_CACHE.get(key)
    if cached is not None:
        launch(cached)
        return cached
    candidates = _bv_candidates(dv, tiles_per_bv, cores)
    for index, bv in enumerate(candidates):
        try:
            launch(bv)
        except _UB_OVERFLOW_ERRORS as exc:
            if not _is_ub_overflow(exc) or index == len(candidates) - 1:
                raise
            continue
        _BV_CACHE[key] = bv
        return bv
    raise NotImplementedError(
        f"{kernel_tag} on 950DT: no BV tile in {candidates} fits the UB budget "
        f"for {signature}"
    )


# --------------------------------------------------------------------------
# chunk prefill
# --------------------------------------------------------------------------


@triton.jit
def _chunk_cumsum_gate_kernel(
    G,
    GC,
    GS,
    Cu,
    T: tl.constexpr,
    HV: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    N: tl.constexpr,
    VARLEN: tl.constexpr,
    TOTAL: tl.constexpr,
):
    """Per-chunk inclusive cumsum of the log-decay, plus the chunk total.

    ``GC`` keeps the running cumsum inside the chunk, ``GS`` the chunk sum,
    so later stages never recompute a prefix.

    One program handles ``BH`` value heads of one chunk rather than a single
    head: the per-head work is only BT=64 elements, so a head-at-a-time grid
    spends more on launch and index arithmetic than on the scan.
    """
    nh = tl.cdiv(HV, BH)
    for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        i_h_blk = pid % nh
        i_rest = pid // nh
        nt = tl.cdiv(T, BT)
        i_t = i_rest % nt
        i_n = i_rest // nt
        if VARLEN:
            start = tl.load(Cu + i_n).to(tl.int64)
            end = tl.load(Cu + i_n + 1).to(tl.int64)
        else:
            start = i_n * T
            end = start + T
        base = start + i_t * BT
        if base < end:
            rows = tl.arange(0, BT)
            hs = i_h_blk * BH + tl.arange(0, BH)
            offs = base + rows
            rmask = offs < end
            hmask = hs < HV
            g = tl.load(
                G + offs[:, None] * HV + hs[None, :],
                mask=rmask[:, None] & hmask[None, :],
                other=0.0,
            ).to(tl.float32)
            # Inclusive scan as a matmul with a lower-triangular ones matrix:
            # cum[i, h] = sum_{j<=i} g[j, h]. tl.dot runs on the cube unit,
            # whereas the previous tl.where(tri, ...) + tl.sum formed a
            # [BT, BT] mask and reduced it on the vector unit for every head.
            tri = (rows[:, None] >= rows[None, :]).to(tl.float32)
            cum = tl.dot(tri, g)
            tl.store(
                GC + offs[:, None] * HV + hs[None, :],
                cum,
                mask=rmask[:, None] & hmask[None, :],
            )
            # The chunk total is the last valid row of the scan, so it needs
            # no second reduction over g.
            total = tl.sum(g, axis=0)
            tl.store(
                GS + (i_n * nt + i_t) * HV + hs,
                total,
                mask=hmask,
            )


@triton.jit
def _chunk_wy_kernel(
    K,
    Beta,
    GC,
    W,
    U,
    V,
    Cu,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    NORMALIZE: tl.constexpr,
    VARLEN: tl.constexpr,
    TOTAL: tl.constexpr,
):
    """UT transform: build the WY representation of one chunk.

    Solves ``(I + tril(diag(beta) K K^T, -1))^-1`` by forward substitution
    over BT rows, then applies it to ``beta*K`` and ``beta*V``.
    """
    for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        i_hv = pid % HV
        i_rest = pid // HV
        nt = tl.cdiv(T, BT)
        i_t = i_rest % nt
        i_n = i_rest // nt
        i_h = i_hv // (HV // H)
        if VARLEN:
            start = tl.load(Cu + i_n).to(tl.int64)
            end = tl.load(Cu + i_n + 1).to(tl.int64)
        else:
            start = i_n * T
            end = start + T
        base = start + i_t * BT
        if base < end:
            rows = tl.arange(0, BT)
            rk = tl.arange(0, DK)
            rv = tl.arange(0, DV)
            offs = base + rows
            mask = offs < end

            k = tl.load(
                K + offs[:, None] * (H * DK) + i_h * DK + rk[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if NORMALIZE:
                norm = tl.sqrt(tl.sum(k * k, axis=1) + 1e-6)
                k = k / norm[:, None]
            beta = tl.load(Beta + offs * HV + i_hv, mask=mask, other=0.0).to(tl.float32)
            gc = tl.load(GC + offs * HV + i_hv, mask=mask, other=0.0).to(tl.float32)

            # A = tril(beta_i * <k_i, k_j> * exp(gc_i - gc_j), -1)
            kk = tl.dot(k, tl.trans(k))
            decay = tl.exp(gc[:, None] - gc[None, :])
            tri = (rows[:, None] > rows[None, :]) & mask[:, None] & mask[None, :]
            a = tl.where(tri, beta[:, None] * kk * decay, 0.0)

            v = tl.load(
                V + offs[:, None] * (HV * DV) + i_hv * DV + rv[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            # Solve (I + A) w = beta * exp(gc) * k and (I + A) u = beta * v.
            # The exp(gc) factor belongs on the right-hand side, before the
            # triangular solve: delta = u - w @ S0^T only holds when w already
            # carries the chunk-start-to-token decay.
            rhs_w = beta[:, None] * k * tl.exp(gc)[:, None]
            rhs_u = beta[:, None] * v

            # Invert (I + A) by doubling instead of forward substitution.
            #
            # A is strictly lower triangular, hence nilpotent with A^BT = 0.
            # From (I - A)(I + A) = I - A^2 comes
            #     (I + A)^-1 = (I - A) (I - A^2)^-1
            # and applying the same step to the residual telescopes into
            #     (I + A)^-1 = (I - A)(I + A^2)(I + A^4)...(I + A^(BT/2))
            # which terminates once A^(2^k) vanishes. At BT=64 that is 11
            # dense [BT, BT] matmuls on the cube unit, replacing 63 dependent
            # iterations that ran 189 tl.where and 126 tl.sum on the vector
            # unit. Verified against torch.linalg.inv at 7e-15 in float64.
            #
            # Note the sign: (I - A^2)^-1 expands to (I + A^2)(I + A^4)...,
            # so every factor after the first is a *plus*. Using (I + A^2)^-1
            # here instead is a different (wrong) matrix and shows up as a
            # ~1.0 absolute error.
            ia = -a
            a_pow = tl.dot(a, a)
            for _ in range(5):
                ia = ia + tl.dot(ia, a_pow) + a_pow
                a_pow = tl.dot(a_pow, a_pow)
            # ia now holds ((I + A)^-1 - I), so applying it keeps the rhs.
            w = rhs_w + tl.dot(ia, rhs_w)
            u = rhs_u + tl.dot(ia, rhs_u)
            tl.store(
                W + offs[:, None] * (HV * DK) + i_hv * DK + rk[None, :],
                w,
                mask=mask[:, None],
            )
            tl.store(
                U + offs[:, None] * (HV * DV) + i_hv * DV + rv[None, :],
                u,
                mask=mask[:, None],
            )


@triton.jit
def _chunk_state_kernel(
    K,
    W,
    U,
    GC,
    GS,
    State,
    Final,
    Cu,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NORMALIZE: tl.constexpr,
    INITIAL: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    VARLEN: tl.constexpr,
    TOTAL: tl.constexpr,
):
    """Sequential chunk recurrence for the state, tiled by ``BV``.

    One program owns one (sequence, v-head, v-tile) triple and walks its
    chunks in order, so each state tile is written by a single core.
    """
    nv = tl.cdiv(DV, BV)
    for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        i_v = pid % nv
        i_rest = pid // nv
        i_hv = i_rest % HV
        i_n = i_rest // HV
        i_h = i_hv // (HV // H)
        if VARLEN:
            start = tl.load(Cu + i_n).to(tl.int64)
            end = tl.load(Cu + i_n + 1).to(tl.int64)
        else:
            start = i_n * T
            end = start + T
        rk = tl.arange(0, DK)
        rv = i_v * BV + tl.arange(0, BV)
        vmask = rv < DV
        rows = tl.arange(0, BT)

        state = tl.zeros([BV, DK], dtype=tl.float32)
        if INITIAL:
            state += tl.load(
                State + i_n * (HV * DV * DK) + i_hv * (DV * DK) + rv[:, None] * DK + rk[None, :],
                mask=vmask[:, None],
                other=0.0,
            ).to(tl.float32)

        nt = tl.cdiv(end - start, BT)
        for i_t in range(nt):
            base = start + i_t * BT
            offs = base + rows
            mask = offs < end
            k = tl.load(
                K + offs[:, None] * (H * DK) + i_h * DK + rk[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if NORMALIZE:
                norm = tl.sqrt(tl.sum(k * k, axis=1) + 1e-6)
                k = k / norm[:, None]
            w = tl.load(
                W + offs[:, None] * (HV * DK) + i_hv * DK + rk[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            u = tl.load(
                U + offs[:, None] * (HV * DV) + i_hv * DV + rv[None, :],
                mask=mask[:, None] & vmask[None, :],
                other=0.0,
            ).to(tl.float32)
            gc = tl.load(GC + offs * HV + i_hv, mask=mask, other=0.0).to(tl.float32)
            gs = tl.load(GS + (i_n * tl.cdiv(T, BT) + i_t) * HV + i_hv).to(tl.float32)

            # v_new = u - w @ state^T, decayed to the chunk end.
            # w already carries exp(gc) from the WY stage, so it is used as
            # is; k is decayed from the token to the chunk end.
            decay_k = tl.exp(gs - gc)
            wd = tl.where(mask[:, None], w, 0.0)
            delta = u - tl.dot(wd, tl.trans(state))
            delta = tl.where(mask[:, None] & vmask[None, :], delta, 0.0)
            kd = tl.where(mask[:, None], k * decay_k[:, None], 0.0)
            state = state * tl.exp(gs) + tl.dot(tl.trans(delta), kd)

        if STORE_FINAL:
            tl.store(
                Final + i_n * (HV * DV * DK) + i_hv * (DV * DK) + rv[None, :] * DK + rk[:, None],
                state_t,
                mask=vmask[None, :],
            )


@triton.jit
def _chunk_output_kernel(
    Q,
    K,
    V,
    W,
    U,
    GC,
    GS,
    State,
    Out,
    Final,
    Cu,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NORMALIZE: tl.constexpr,
    INITIAL: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    VARLEN: tl.constexpr,
    TOTAL: tl.constexpr,
):
    """Intra-chunk attention, inter-chunk state, and the final state.

    Recomputes the running state per sequence rather than reading a
    per-chunk snapshot: that trades flops for GM traffic, which is the
    better side of the trade on 950DT, and keeps peak memory at one state.

    The value dimension is tiled by ``BV`` so the resident state is
    ``[BV, DK]`` rather than ``[DV, DK]``. At DK=DV=128 the untiled form
    overflows the 950DT UB budget (248 KB) during codegen, so ``BV`` is the
    knob that keeps this kernel compilable at Qwen3-Next head dims.
    """
    nv = tl.cdiv(DV, BV)
    for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        i_v = pid % nv
        i_rest = pid // nv
        i_hv = i_rest % HV
        i_n = i_rest // HV
        i_h = i_hv // (HV // H)
        if VARLEN:
            start = tl.load(Cu + i_n).to(tl.int64)
            end = tl.load(Cu + i_n + 1).to(tl.int64)
        else:
            start = i_n * T
            end = start + T
        rk = tl.arange(0, DK)
        rv = i_v * BV + tl.arange(0, BV)
        vmask = rv < DV
        rows = tl.arange(0, BT)

        # State is held transposed, as [DK, BV] rather than [BV, DK]. Both
        # consumers in the loop want state^T, so storing it this way removes
        # two tl.trans of a [BV, DK] tile per chunk; the only cost is that
        # the GM load and store below are strided the other way, which they
        # pay once per sequence instead of once per chunk.
        state_t = tl.zeros([DK, BV], dtype=tl.float32)
        if INITIAL:
            state_t += tl.load(
                State + i_n * (HV * DV * DK) + i_hv * (DV * DK) + rv[None, :] * DK + rk[:, None],
                mask=vmask[None, :],
                other=0.0,
            ).to(tl.float32)

        nt = tl.cdiv(end - start, BT)
        for i_t in range(nt):
            base = start + i_t * BT
            offs = base + rows
            mask = offs < end
            q = tl.load(
                Q + offs[:, None] * (H * DK) + i_h * DK + rk[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                K + offs[:, None] * (H * DK) + i_h * DK + rk[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if NORMALIZE:
                qn = tl.sqrt(tl.sum(q * q, axis=1) + 1e-6)
                q = q / qn[:, None]
                kn = tl.sqrt(tl.sum(k * k, axis=1) + 1e-6)
                k = k / kn[:, None]
            q = q * scale
            w = tl.load(
                W + offs[:, None] * (HV * DK) + i_hv * DK + rk[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            u = tl.load(
                U + offs[:, None] * (HV * DV) + i_hv * DV + rv[None, :],
                mask=mask[:, None] & vmask[None, :],
                other=0.0,
            ).to(tl.float32)
            gc = tl.load(GC + offs * HV + i_hv, mask=mask, other=0.0).to(tl.float32)
            gs = tl.load(GS + (i_n * tl.cdiv(T, BT) + i_t) * HV + i_hv).to(tl.float32)

            # w already carries exp(gc) from the WY stage; k is decayed from
            # the token to the chunk end for the state update.
            #
            # No re-masking of w/u here: the WY kernel stored them under the
            # same row mask, so padding rows are already zero in GM. The
            # previous tl.where on each was a redundant second clear, and
            # tl.where is comparatively slow on this backend.
            decay_k = tl.exp(gs - gc)
            v_new = u - tl.dot(w, state_t)

            # Inter-chunk: q, decayed from the chunk start, against the state.
            o = tl.dot(q * tl.exp(gc)[:, None], state_t)
            # Intra-chunk: strictly causal. The causal mask is folded into the
            # decay matrix as a multiply instead of a tl.where -- exp() of the
            # gate difference is needed anyway, so masking costs one extra
            # elementwise product rather than a select.
            qk = tl.dot(q, tl.trans(k))
            causal = (rows[:, None] >= rows[None, :]).to(tl.float32)
            attn = qk * tl.exp(gc[:, None] - gc[None, :]) * causal
            o += tl.dot(attn, v_new)
            tl.store(
                Out + offs[:, None] * (HV * DV) + i_hv * DV + rv[None, :],
                o,
                mask=mask[:, None] & vmask[None, :],
            )

            # k was loaded with other=0.0, so its padding rows are already
            # zero; the state update contracts over rows, and a zero row of
            # kd kills that row's contribution regardless of v_new there.
            # That is what makes dropping the v_new mask above safe.
            kd = k * decay_k[:, None]
            state_t = state_t * tl.exp(gs) + tl.dot(tl.trans(kd), v_new)

        # The running state after the last chunk *is* the final state, so
        # write it here. A separate state kernel used to recompute the whole
        # recurrence for this, costing an extra 53 us of the 337 us total at
        # s1024 (16%) for a value this loop already holds.
        if STORE_FINAL:
            tl.store(
                Final + i_n * (HV * DV * DK) + i_hv * (DV * DK) + rv[None, :] * DK + rk[:, None],
                state_t,
                mask=vmask[None, :],
            )


def _grid_for(total, device_index):
    return (min(total, _vector_core_count(device_index)),)


def chunk_gdn_950dt(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
):
    """Chunked GDN forward for Ascend 950DT.

    Layout is sequence-major and packed, matching the aclnn TND contract so
    the two can be compared without a repack:

    - ``q``/``k``: ``[T, H, DK]``, ``v``: ``[T, HV, DV]``
    - ``g``/``beta``: ``[T, HV]``, ``g`` a raw float32 log-decay (<= 0)
    - ``initial_state``: ``[N, HV, DV, DK]``
    - ``cu_seqlens``: ``[N+1]`` int32/int64 prefix sums, or ``None`` for one
      sequence spanning all T rows

    Returns ``(o, final_state)`` with ``o`` as ``[T, HV, DV]`` and
    ``final_state`` as ``[N, HV, DV, DK]`` (``None`` when not requested).
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("chunk GDN expects packed q/k [T,H,DK] and v [T,HV,DV]")
    if k.shape != q.shape:
        raise ValueError("q and k must have identical shapes")
    T, H, DK = q.shape
    HV, DV = v.shape[1], v.shape[2]
    if v.shape[0] != T:
        raise ValueError("v must have the same token count as q/k")
    if HV % H:
        raise ValueError("HV must be a multiple of H for grouped value attention")
    if q.dtype not in _FLOAT_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must share one of float16/bfloat16/float32")
    if DK > 128 or DV > 128:
        raise NotImplementedError(
            "chunk GDN on 950DT supports DK,DV <= 128; larger head dims need "
            "a blocked state that this kernel does not implement"
        )
    if g is None or beta is None:
        raise NotImplementedError(
            "chunk GDN on 950DT requires both g and beta; the ungated variant "
            "is not implemented"
        )
    if g.shape != (T, HV) or beta.shape != (T, HV):
        raise ValueError("g and beta must both have shape [T, HV]")
    for tensor in (k, v, g, beta, initial_state, cu_seqlens):
        if tensor is not None and tensor.device != q.device:
            raise ValueError("All inputs must be on the same device")
    if scale is None:
        scale = DK**-0.5
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    varlen = cu_seqlens is not None
    if varlen:
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a 1D [N+1] prefix-sum tensor")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be int32/int64")
        n = cu_seqlens.numel() - 1
        t_per_seq = T
    else:
        n = 1
        t_per_seq = T
    if initial_state is not None:
        if initial_state.shape != (n, HV, DV, DK):
            raise ValueError("initial_state must have shape [N, HV, DV, DK]")
        if initial_state.dtype not in _FLOAT_DTYPES:
            raise ValueError("initial_state must be floating point")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()

    o = torch.empty((T, HV, DV), dtype=v.dtype, device=v.device)
    final = (
        torch.empty((n, HV, DV, DK), dtype=torch.float32, device=q.device)
        if output_final_state
        else None
    )
    if T == 0:
        return o, final

    nt = triton.cdiv(t_per_seq, CHUNK_SIZE)
    gc = torch.empty((T, HV), dtype=torch.float32, device=q.device)
    gs = torch.empty((n * nt, HV), dtype=torch.float32, device=q.device)
    w = torch.empty((T, HV, DK), dtype=torch.float32, device=q.device)
    u = torch.empty((T, HV, DV), dtype=torch.float32, device=q.device)

    dev = q.device.index
    # One program per (chunk, head-block). 16 heads per block keeps the
    # scan's [BT, BH] tile dense while still leaving enough programs to fill
    # the 64 vector cores at the shapes this operator sees.
    bh = min(16, HV)
    total_cumsum = n * nt * triton.cdiv(HV, bh)
    _chunk_cumsum_gate_kernel[_grid_for(total_cumsum, dev)](
        g,
        gc,
        gs,
        cu_seqlens,
        T=t_per_seq,
        HV=HV,
        BT=CHUNK_SIZE,
        BH=bh,
        N=n,
        VARLEN=varlen,
        TOTAL=total_cumsum,
    )
    total_chunk = n * nt * HV
    _chunk_wy_kernel[_grid_for(total_chunk, dev)](
        k,
        beta,
        gc,
        w,
        u,
        v,
        cu_seqlens,
        scale,
        T=t_per_seq,
        H=H,
        HV=HV,
        DK=DK,
        DV=DV,
        BT=CHUNK_SIZE,
        NORMALIZE=use_qk_l2norm_in_kernel,
        VARLEN=varlen,
        TOTAL=total_chunk,
    )
    total_seq = n * HV

    def _launch_output(bv):
        nv = triton.cdiv(DV, bv)
        total_tiles = total_seq * nv
        _chunk_output_kernel[_grid_for(total_tiles, dev)](
            q,
            k,
            v,
            w,
            u,
            gc,
            gs,
            initial_state,
            o,
            final,
            cu_seqlens,
            scale,
            T=t_per_seq,
            H=H,
            HV=HV,
            DK=DK,
            DV=DV,
            BT=CHUNK_SIZE,
            BV=bv,
            NORMALIZE=use_qk_l2norm_in_kernel,
            INITIAL=initial_state is not None,
            STORE_FINAL=output_final_state,
            VARLEN=varlen,
            TOTAL=total_tiles,
        )

    # BV caps the resident [BV, DK] state so this kernel fits the 950DT UB
    # budget (~2.0 Mbit); the working tile is probed once per shape and then
    # remembered (see _resolve_bv).
    # No autotune: replaying these kernels mutates the state pool, so a timed
    # tile search cannot be done safely (documented exemption, mirroring the
    # existing recurrent GDN kernel in this backend). The probe picks the
    # largest *compilable* tile, not the fastest one.
    cores = _vector_core_count(dev)
    bv = _resolve_bv(
        "chunk_output",
        DV,
        (DK, DV, H, HV, use_qk_l2norm_in_kernel, q.dtype, varlen, n),
        _launch_output,
        tiles_per_bv=lambda bv: total_seq * triton.cdiv(DV, bv),
        cores=cores,
    )
    # No separate state pass: _chunk_output_kernel already carries the
    # recurrence to the end of the sequence and stores the final state.
    return o, final


# --------------------------------------------------------------------------
# recurrent decode
# --------------------------------------------------------------------------


@triton.jit
def _recurrent_gdn_950dt_kernel(
    Q,
    K,
    V,
    G,
    Beta,
    State,
    Out,
    Cu,
    Indices,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    POOL: tl.constexpr,
    BV: tl.constexpr,
    NORMALIZE: tl.constexpr,
    VARLEN: tl.constexpr,
    INDEXED: tl.constexpr,
    TOTAL: tl.constexpr,
):
    """One program per (sequence, v-head, v-tile): sequential walk over T.

    The state tile is ``[BV, DK]`` and stays in registers across tokens, so
    the per-token cost is two rank-structured products and no GM round trip.
    Tiling the value dimension both bounds UB (the l2-norm path overflows at
    BV=DV=128) and widens the grid, which matters at decode where the
    sequence count alone leaves most of the 64 vector cores idle.
    """
    nv = tl.cdiv(DV, BV)
    for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        i_v = pid % nv
        i_rest = pid // nv
        i_hv = i_rest % HV
        i_n = i_rest // HV
        i_h = i_hv // (HV // H)
        if VARLEN:
            start = tl.load(Cu + i_n).to(tl.int64)
            end = tl.load(Cu + i_n + 1).to(tl.int64)
        else:
            start = i_n * T
            end = start + T

        slot = i_n
        if INDEXED:
            slot = tl.load(Indices + i_n).to(tl.int64)
        valid = (slot >= 0) & (slot < POOL)

        rk = tl.arange(0, DK)
        rv = i_v * BV + tl.arange(0, BV)
        vmask = rv < DV
        state = tl.zeros([BV, DK], dtype=tl.float32)
        if valid:
            state += tl.load(
                State + slot * (HV * DV * DK) + i_hv * (DV * DK) + rv[:, None] * DK + rk[None, :],
                mask=vmask[:, None],
                other=0.0,
            ).to(tl.float32)

        for t in range(start, end):
            q = tl.load(Q + t * (H * DK) + i_h * DK + rk).to(tl.float32)
            k = tl.load(K + t * (H * DK) + i_h * DK + rk).to(tl.float32)
            if NORMALIZE:
                q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
                k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
            q = q * scale
            v = tl.load(
                V + t * (HV * DV) + i_hv * DV + rv, mask=vmask, other=0.0
            ).to(tl.float32)
            g = tl.load(G + t * HV + i_hv).to(tl.float32)
            beta = tl.load(Beta + t * HV + i_hv).to(tl.float32)

            state = state * tl.exp(g)
            # delta rule: correct the state by the prediction error on v.
            pred = tl.sum(state * k[None, :], axis=1)
            delta = (v - pred) * beta
            state += delta[:, None] * k[None, :]
            o = tl.sum(state * q[None, :], axis=1)
            tl.store(Out + t * (HV * DV) + i_hv * DV + rv, o, mask=vmask)

        if valid:
            tl.store(
                State + slot * (HV * DV * DK) + i_hv * (DV * DK) + rv[:, None] * DK + rk[None, :],
                state,
                mask=vmask[:, None],
            )


def fused_recurrent_gdn_950dt(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    cu_seqlens=None,
    ssm_state_indices=None,
    use_qk_l2norm_in_kernel=False,
):
    """Recurrent GDN forward for Ascend 950DT, updating the state in place.

    Same packed layout as :func:`chunk_gdn_950dt`. ``initial_state`` is the
    state pool ``[POOL, HV, DV, DK]``; ``ssm_state_indices`` is a ``[N]`` map
    from sequence to pool slot (identity when omitted). Slots outside
    ``[0, POOL)`` are read as zero and not written, matching the repo's
    existing recurrent contract.

    Returns ``(o, final_state)`` where ``final_state`` aliases the updated
    pool. Unlike the aclnn op this accepts a single-sequence batch (the aclnn
    kernel rejects N=1) and tolerates repeated invocation.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("recurrent GDN expects packed q/k [T,H,DK] and v [T,HV,DV]")
    if k.shape != q.shape:
        raise ValueError("q and k must have identical shapes")
    T, H, DK = q.shape
    HV, DV = v.shape[1], v.shape[2]
    if v.shape[0] != T:
        raise ValueError("v must have the same token count as q/k")
    if HV % H:
        raise ValueError("HV must be a multiple of H for grouped value attention")
    if q.dtype not in _FLOAT_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must share one of float16/bfloat16/float32")
    if DK > 128 or DV > 128:
        raise NotImplementedError("recurrent GDN on 950DT supports DK,DV <= 128")
    if g is None or beta is None:
        raise NotImplementedError("recurrent GDN on 950DT requires both g and beta")
    if g.shape != (T, HV) or beta.shape != (T, HV):
        raise ValueError("g and beta must both have shape [T, HV]")
    if initial_state is None:
        raise NotImplementedError(
            "recurrent GDN on 950DT requires a state pool; the stateless "
            "variant is not implemented"
        )
    for tensor in (k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices):
        if tensor is not None and tensor.device != q.device:
            raise ValueError("All inputs must be on the same device")
    if scale is None:
        scale = DK**-0.5
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    varlen = cu_seqlens is not None
    if varlen:
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a 1D [N+1] prefix-sum tensor")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be int32/int64")
        n = cu_seqlens.numel() - 1
        t_per_seq = T
    else:
        n = 1
        t_per_seq = T
    if initial_state.ndim != 4 or initial_state.shape[1:] != (HV, DV, DK):
        raise ValueError("initial_state must have shape [POOL, HV, DV, DK]")
    pool = initial_state.shape[0]
    if ssm_state_indices is not None:
        if ssm_state_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("ssm_state_indices must be int32/int64")
        if ssm_state_indices.ndim != 1 or ssm_state_indices.numel() != n:
            raise ValueError("ssm_state_indices must have shape [N]")
    elif pool < n:
        raise ValueError("State pool is smaller than the sequence count")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    state = initial_state.contiguous()
    if ssm_state_indices is not None:
        ssm_state_indices = ssm_state_indices.contiguous()

    o = torch.empty((T, HV, DV), dtype=v.dtype, device=v.device)
    if T == 0:
        return o, state

    def _launch_recurrent(bv):
        nv = triton.cdiv(DV, bv)
        total = n * HV * nv
        _recurrent_gdn_950dt_kernel[_grid_for(total, q.device.index)](
            q,
            k,
            v,
            g,
            beta,
            state,
            o,
            cu_seqlens,
            ssm_state_indices,
            scale,
            T=t_per_seq,
            H=H,
            HV=HV,
            DK=DK,
            DV=DV,
            POOL=pool,
            BV=bv,
            NORMALIZE=use_qk_l2norm_in_kernel,
            VARLEN=varlen,
            INDEXED=ssm_state_indices is not None,
            TOTAL=total,
        )

    # BV bounds the resident [BV, DK] state, and tiling also widens the grid,
    # which is what limits decode throughput when N*HV alone is far below the
    # 64 vector cores. The working tile is probed once per shape and cached
    # (see _resolve_bv); at DV=128 with the in-kernel l2-norm the untiled
    # form overflows UB by ~5%.
    # No autotune: this kernel mutates the state pool in place, so a timed
    # tile search would compound state across trials (documented exemption,
    # matching the existing recurrent GDN kernel in this backend).
    _resolve_bv(
        "recurrent",
        DV,
        (DK, DV, H, HV, use_qk_l2norm_in_kernel, q.dtype, varlen, n),
        _launch_recurrent,
        tiles_per_bv=lambda bv: n * HV * triton.cdiv(DV, bv),
        cores=_vector_core_count(q.device.index),
    )
    return o, state




__all__ = ["chunk_gdn_950dt", "fused_recurrent_gdn_950dt"]

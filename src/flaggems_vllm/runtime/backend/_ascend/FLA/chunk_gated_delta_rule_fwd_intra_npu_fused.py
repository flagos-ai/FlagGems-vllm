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


def _grid_for(total, device_index):
    """Cap the grid at the vector-core count.

    An oversized auto-parallel launch can replay task zero, which matters
    for kernels that write state; here it also keeps one tile per core.
    """
    return (min(total, _vector_core_count(device_index)),)

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


# --------------------------------------------------------------------------
# chunk_gated_delta_rule_fwd_intra: FLA-compatible entry
# --------------------------------------------------------------------------


@triton.jit
def _fwd_intra_fused_kernel(
    K,
    V,
    G,
    Beta,
    W,
    U,
    A,
    Cu,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    STEPS: tl.constexpr,
    USE_G: tl.constexpr,
    VARLEN: tl.constexpr,
    TOTAL,
):
    """kkt, triangular inverse and w/u for one chunk, in a single pass.

    The reference path runs three kernels (chunk_scaled_dot_kkt_fwd,
    solve_tril, recompute_w_u_fwd) and moves the [BT, BT] matrix through GM
    twice between them. Here the inverse never leaves UB: it is built from
    the kkt product and consumed by the w/u matmuls in the same program.

    Gate convention follows FLA: ``g`` is already a per-chunk cumulative sum
    and the decay uses ``exp2``, not ``exp``.
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
            beta = tl.load(Beta + offs * HV + i_hv, mask=mask, other=0.0).to(tl.float32)

            kk = tl.dot(k, tl.trans(k))
            if USE_G:
                g = tl.load(G + offs * HV + i_hv, mask=mask, other=0.0).to(tl.float32)
                kk = kk * tl.exp2(g[:, None] - g[None, :])
            strict = (rows[:, None] > rows[None, :]) & mask[:, None] & mask[None, :]
            a = tl.where(strict, kk * beta[:, None], 0.0)

            # (I + a)^-1 - I by doubling; see _chunk_wy_kernel for the
            # identity and the step-count choice.
            ia = -a
            a_pow = tl.dot(a, a)
            for _ in range(STEPS):
                ia = ia + tl.dot(ia, a_pow) + a_pow
                a_pow = tl.dot(a_pow, a_pow)

            # A is returned as the full inverse, matching solve_tril's output.
            tl.store(
                A + (offs[:, None] * HV + i_hv) * BT + rows[None, :],
                ia + (rows[:, None] == rows[None, :]).to(tl.float32),
                mask=mask[:, None],
            )

            # v and k are dead once their rhs is formed, so the products go
            # back into the same names. Holding k/v and separate rhs tiles
            # live together put per-program UB at 192 KB of the ~248 KB
            # budget, leaving no room for double buffering.
            v = tl.load(
                V + offs[:, None] * (HV * DV) + i_hv * DV + rv[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            v = v * beta[:, None]
            tl.store(
                U + offs[:, None] * (HV * DV) + i_hv * DV + rv[None, :],
                v + tl.dot(ia, v),
                mask=mask[:, None],
            )

            k = k * beta[:, None]
            if USE_G:
                k = k * tl.exp2(g)[:, None]
            tl.store(
                W + offs[:, None] * (HV * DK) + i_hv * DK + rk[None, :],
                k + tl.dot(ia, k),
                mask=mask[:, None],
            )


def chunk_gated_delta_rule_fwd_intra_npu_fused(
    k,
    v,
    g=None,
    beta=None,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    """Single-kernel ``chunk_gated_delta_rule_fwd_intra``; kept, not dispatched.

    Same signature, layouts, gate convention (``g`` pre-summed per chunk,
    ``exp2`` decay) and return triple ``(w, u, A)`` with ``w: [B, T, HV, K]``,
    ``u: [B, T, HV, V]``, ``A: [B, T, HV, BT]``.

    Differs only in execution: the reference composes three kernels and
    round-trips the [BT, BT] matrix through GM twice, while this fuses them
    so the triangular inverse stays in UB.

    .. note::
       ``chunk_gated_delta_rule_fwd_intra_950dt`` no longer resolves here.
       It now routes to the three-stage implementation in
       ``GDN_950DT_3stage.py``, which keeps the reference's kernel
       decomposition and optimizes each stage separately. This fused version
       stays for comparison -- it is the faster of the two, but it cannot
       answer how fast an individual stage can be made, which is what the
       per-stage profiles measure.

    ``chunk_indices`` is accepted for signature compatibility and ignored --
    chunk bounds come from ``cu_seqlens`` directly, so there is no
    precomputed index table to consume.
    """
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"`chunk_size` must be 16, 32, or 64, got {chunk_size}.")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must be [B, T, H, K] / [B, T, HV, V]")
    if beta is None:
        raise NotImplementedError(
            "fwd_intra on 950DT requires beta; the unweighted variant is not "
            "implemented"
        )
    b, t, h, dk = k.shape
    hv, dv = v.shape[2], v.shape[3]
    if v.shape[:2] != (b, t):
        raise ValueError("v must share batch and token extents with k")
    if hv % h:
        raise ValueError("HV must be a multiple of H")
    if dk > 128 or dv > 128:
        raise NotImplementedError("fwd_intra on 950DT supports K, V <= 128")
    if beta.shape != (b, t, hv):
        raise ValueError("beta must be [B, T, HV]")
    if g is not None and g.shape != (b, t, hv):
        raise ValueError("g must be [B, T, HV]")
    del chunk_indices

    bt = chunk_size
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    if g is not None:
        g = g.contiguous()

    w = torch.empty((b, t, hv, dk), dtype=k.dtype, device=k.device)
    u = torch.empty((b, t, hv, dv), dtype=v.dtype, device=v.device)
    a_out = torch.empty((b, t, hv, bt), dtype=k.dtype, device=k.device)
    if t == 0:
        return w, u, a_out

    varlen = cu_seqlens is not None
    if varlen:
        if b != 1:
            raise ValueError("packed varlen input requires B == 1")
        n = cu_seqlens.numel() - 1
    else:
        n = b
    nt = triton.cdiv(t, bt)
    # Doubling step k makes the series exact through A^(2^(k+1) - 1). A is
    # BT x BT strictly lower triangular, so A^BT = 0 and exactness through
    # A^(BT-1) is required: STEPS >= log2(BT) - 1, i.e. 5 for BT=64, 4 for
    # 32, 3 for 16.
    #
    # Do not shave a step off this. With small A the truncated series still
    # converges numerically -- at the L2-normalized-k magnitudes these ops
    # see (absmax(A) ~ 0.4) STEPS=4 at BT=64 matches torch.linalg.inv to
    # 1e-16, so a test sweep passes. But the error tracks A's magnitude, not
    # the shape: in float64 against the exact inverse, STEPS=4 at BT=64 gives
    # 2.5e-7 relative at absmax(A)=1.4, 6.7e-2 at 3.2 and 3.4 at 7.8.
    steps = max(3, int(math.log2(bt)) - 1)
    total = n * nt * hv
    _fwd_intra_fused_kernel[_grid_for(total, k.device.index)](
        k.view(-1, h, dk),
        v.view(-1, hv, dv),
        g.view(-1, hv) if g is not None else None,
        beta.view(-1, hv),
        w.view(-1, hv, dk),
        u.view(-1, hv, dv),
        a_out.view(-1, hv, bt),
        cu_seqlens,
        t,
        H=h,
        HV=hv,
        DK=dk,
        DV=dv,
        BT=bt,
        STEPS=steps,
        USE_G=g is not None,
        VARLEN=varlen,
        TOTAL=total,
    )
    return w, u, a_out


__all__ = ["chunk_gated_delta_rule_fwd_intra_npu_fused"]

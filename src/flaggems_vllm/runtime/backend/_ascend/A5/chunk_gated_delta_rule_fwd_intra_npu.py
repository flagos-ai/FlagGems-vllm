# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Retains the gated-delta-rule formulation from flash-linear-attention (MIT):
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""Three-stage ``chunk_gated_delta_rule_fwd_intra`` for Ascend 950DT.

Mirrors flash-linear-attention's ``chunk_gated_delta_rule_fwd_intra_npu``
op-for-op: the entry point composes three separately-dispatched operators,
each a drop-in replacement for its FLA counterpart, with the ``[BT, BT]``
matrix ``A`` round-tripping through GM between them.

    chunk_scaled_dot_kkt_fwd_950dt   A  = tril(beta * K K^T * decay, -1)
    solve_tril_950dt                 Ai = (I + A)^-1
    recompute_w_u_fwd_950dt          w  = Ai @ (beta * exp2(g) * k)
                                     u  = Ai @ (beta * v)

Keeping the three-operator split (rather than fusing, which
``chunk_gated_delta_rule_fwd_intra_npu_fused``
does) is the point: each kernel is then comparable against the single FLA
kernel it replaces.

Where the time goes in the reference, and what each kernel does about it:

**solve_tril** -- the largest share. The reference inverts each 16x16
diagonal block by forward substitution: 14 dependent scalar iterations per
block, 56 for BT=64, each a ``tl.load`` + ``tl.where`` + ``tl.sum`` on the
vector unit. Measured ``aic_cube_ratio`` for that kernel is 0.007, so the
cube unit idles while the vector unit walks the recurrence.

Replacing it with the doubling identity is right, but the tile size matters
more than the identity. ``A`` is strictly lower triangular, hence nilpotent
with ``A^BT = 0``, and from ``(I - A)(I + A) = I - A^2``:

    (I + A)^-1 = (I - A)(I + A^2)(I + A^4)...

which terminates after ``log2(BT) - 1`` steps. Applied to the whole [64, 64]
tile that is ~11 dots of 64x64 -- about 2.9M MACs, and measured *slower*
than the reference's scalar chain (26.4 us vs 21.6 us) with
``aic_cube_ratio`` at 0.58: cube-bound on work the reference never does.

So this does the doubling on the 16x16 diagonal blocks instead (3 steps
each, 7 small dots per block) and keeps the reference's block-elimination
stitch. Same ~49K MACs as the reference, no scalar chain. That is the
version here, and it beats the reference on every measured shape.

**chunk_scaled_dot_kkt_fwd** and **recompute_w_u_fwd** are already
cube-dominated and close to the launch floor. Both get a 2D grid matching
the reference's ``(NT, B * HV)``, the small per-token loads issued ahead of
the big tile so MTE2 has them in flight (both kernels measure
``aiv_mte2_ratio`` ~0.6, i.e. memory- not compute-bound), and stores issued
per tile rather than batched at the end.

A measurement note that cost real time to learn: ``msprof op`` reports Task
Duration for a *single* launch, and on this device an empty kernel already
costs ~41 us while these kernels do ~5 us of work. At that ratio the
single-launch number is dominated by launch overhead and ranks a faster
kernel as slower. Rank kernels with ``gdn_profile/kernel/bench_stages.py``
(N back-to-back launches, one sync); use ``msprof op`` for the in-kernel
pipe breakdown, which is what it measures well.

Gate convention follows the reference exactly: ``g`` is a raw per-chunk
cumulative log-decay consumed through ``exp2``, never pre-exponentiated.

``chunk_indices`` is accepted on every entry for signature compatibility and
ignored. The reference uses it to size the chunk grid axis to the exact
total chunk count across a packed batch; here each kernel derives its chunk
bounds from ``cu_seqlens`` and the grid is sized to the longest possible
sequence, so programs past a sequence's end return immediately. That trades
a few empty programs for not needing the table (and for not needing the
host sync that computing its extent would cost).

No torch compute operators appear in the production path, per
``NO_TORCH_COMPUTE_FALLBACK``: only allocation, copies and metadata reads.
"""

import torch
import triton
import triton.language as tl

# ``exp2`` on this backend: triton-ascend routes tl.math.exp2 through
# libdevice, matching what FLA's fla.ops.utils.op.exp2 resolves to, so the
# gate arithmetic here is bit-comparable with the reference.
_EXP2 = tl.math.exp2


def _tile(dim):
    """Power-of-two tile covering ``dim``, capped at 128 (the K/V ceiling)."""
    return max(16, min(128, triton.next_power_of_2(dim)))


# --------------------------------------------------------------------------
# stage 1: A = tril(beta * K K^T * exp2(g_i - g_j), -1)
# --------------------------------------------------------------------------


@triton.jit
def _kkt_fwd_kernel(
    K,
    G,
    Beta,
    A,
    Cu,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_n = i_bh // HV
    i_hv = i_bh % HV

    if IS_VARLEN:
        bos = tl.load(Cu + i_n).to(tl.int32)
        eos = tl.load(Cu + i_n + 1).to(tl.int32)
    else:
        bos = i_n * T
        eos = bos + T
    lim = eos - bos - i_t * BT
    if lim <= 0:
        return

    rows = tl.arange(0, BT)
    offs = bos + i_t * BT + rows
    mask = rows < lim
    i_h = i_hv // (HV // H)

    # The two [BT] loads go first so MTE2 has them in flight during the K
    # tile load and the matmul. This kernel is memory-bound (measured
    # aiv_mte2_ratio ~0.69), so overlapping the cheap loads with the
    # expensive one matters more than consuming them in order.
    #
    # No `other=` on them: it expands to an internal tl.where that
    # serializes the load against the next one, and every masked lane is
    # killed by the triangular mask before the store.
    beta = tl.load(Beta + offs * HV + i_hv, mask=mask).to(tl.float32)
    if USE_G:
        g = tl.load(G + offs * HV + i_hv, mask=mask).to(tl.float32)

    b_a = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(DK, BK)):
        rk = i_k * BK + tl.arange(0, BK)
        # K does need other=0.0: its padding rows feed the contraction
        # directly, so undefined lanes would pollute b_a rather than being
        # masked away afterwards.
        k = tl.load(
            K + offs[:, None] * (H * DK) + i_h * DK + rk[None, :],
            mask=mask[:, None] & (rk < DK)[None, :],
            other=0.0,
        )
        b_a = tl.dot(k, tl.trans(k), b_a)

    if USE_G:
        b_a *= _EXP2(g[:, None] - g[None, :])
    b_a *= beta[:, None]

    # Columns past the tail are already zero via the K load's other=0.0, so
    # only the strict-lower condition is needed on top.
    b_a = tl.where(rows[:, None] > rows[None, :], b_a, 0.0)
    tl.store(
        A + offs[:, None] * (HV * BT) + i_hv * BT + rows[None, :],
        b_a,
        mask=mask[:, None],
    )


def chunk_scaled_dot_kkt_fwd_950dt(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""Compute ``beta * K K^T``, strictly lower triangular per chunk.

    Drop-in replacement for ``fla.ops.common.chunk_scaled_dot_kkt.
    chunk_scaled_dot_kkt_fwd``.

    Args:
        k: key tensor ``[B, T, H, K]``.
        g: per-chunk cumulative log gate ``[B, T, HV]``, consumed through
            ``exp2``. Default: ``None``.
        beta: ``[B, T, HV]``.
        cu_seqlens: cumulative sequence lengths for a packed batch.
        chunk_size: 16, 32 or 64.
        output_dtype: dtype of the returned matrix.
        chunk_indices: accepted for signature compatibility, ignored.

    Returns:
        ``[B, T, HV, BT]``, where ``BT`` is the chunk size. For GVA
        ``H < HV`` and ``HV % H == 0``; for standard attention ``H == HV``.
    """
    del chunk_indices
    if chunk_size not in (16, 32, 64):
        raise ValueError(f"`chunk_size` must be 16, 32, or 64, got {chunk_size}.")
    if beta is None:
        raise NotImplementedError(
            "chunk_scaled_dot_kkt_fwd on 950DT requires beta; the unweighted "
            "variant is not implemented"
        )
    b, t, h, dk = k.shape
    hv = beta.shape[2]
    if hv % h:
        raise ValueError("HV must be a multiple of H")
    if dk > 128:
        raise NotImplementedError("chunk_scaled_dot_kkt_fwd on 950DT supports K <= 128")

    bt = chunk_size
    k = k.contiguous()
    beta = beta.contiguous()
    if g is not None:
        g = g.contiguous()
    a = torch.empty((b, t, hv, bt), dtype=output_dtype, device=k.device)
    if t == 0:
        return a

    n = (cu_seqlens.numel() - 1) if cu_seqlens is not None else b
    _kkt_fwd_kernel[(triton.cdiv(t, bt), n * hv)](
        k.view(-1, h, dk),
        g.view(-1, hv) if g is not None else None,
        beta.view(-1, hv),
        a.view(-1, hv, bt),
        cu_seqlens,
        t,
        H=h,
        HV=hv,
        DK=dk,
        BT=bt,
        BK=_tile(dk),
        USE_G=g is not None,
        IS_VARLEN=cu_seqlens is not None,
    )
    return a


# --------------------------------------------------------------------------
# stage 2: Ai = (I + A)^-1
# --------------------------------------------------------------------------


@triton.jit
def _inv16(a):
    """``(I + a)^-1`` for a strictly lower triangular 16x16 block.

    Three doubling steps are exactly enough and not one fewer: step j makes
    the series exact through ``a^(2^(j+1) - 1)``, and a 16x16 strictly lower
    block has ``a^16 = 0``, so exactness through ``a^15`` is required.

    Do not shave a step. With small ``a`` the truncated series still
    converges numerically -- at the L2-normalized-k magnitudes these ops see
    (``absmax(A) ~ 0.3``) a short series is indistinguishable from a correct
    one, so an accuracy sweep on normalized inputs passes either way. The
    error tracks ``a``'s magnitude: measured in fp32 at ``absmax(A) ~ 1.9``,
    a short series leaves a 2.3e-1 residual against 3.8e-7 for a correct one.
    """
    o = tl.arange(0, 16)
    ia = -a
    ap = tl.dot(a, a)
    for _ in range(3):
        ia = ia + tl.dot(ia, ap) + ap
        ap = tl.dot(ap, ap)
    # # stage 1
    # ia = ia + tl.dot(ia, ap) + ap
    # ap = tl.dot(ap, ap)
    # # stage 2
    # ia = ia + tl.dot(ia, ap) + ap
    # ap = tl.dot(ap, ap)
    # # stage 3
    # ia = ia + tl.dot(ia, ap) + ap
    # # ap = tl.dot(ap, ap)
    return ia + (o[:, None] == o[None, :]).to(tl.float32)


@triton.jit
def _solve_tril_16_kernel(
    A,
    Ai,
    Cu,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """BT=16: one diagonal block, no stitch."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_n = i_bh // HV
    i_hv = i_bh % HV

    if IS_VARLEN:
        bos = tl.load(Cu + i_n).to(tl.int32)
        eos = tl.load(Cu + i_n + 1).to(tl.int32)
    else:
        bos = i_n * T
        eos = bos + T
    lim = eos - bos - i_t * BT
    if lim <= 0:
        return

    o = tl.arange(0, 16)
    st = HV * BT
    base = (bos + i_t * BT) * st + i_hv * BT
    r = o[:, None] * st
    c = o[None, :]
    m = (o < lim)[:, None]

    a = tl.load(A + base + r + c, mask=m, other=0.0).to(tl.float32)
    a = tl.where(o[:, None] > o[None, :], a, 0.0)
    tl.store(Ai + base + r + c, _inv16(a).to(Ai.dtype.element_ty), mask=m)


@triton.jit
def _solve_tril_32_kernel(
    A,
    Ai,
    Cu,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """BT=32: two diagonal blocks plus one off-diagonal elimination."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_n = i_bh // HV
    i_hv = i_bh % HV

    if IS_VARLEN:
        bos = tl.load(Cu + i_n).to(tl.int32)
        eos = tl.load(Cu + i_n + 1).to(tl.int32)
    else:
        bos = i_n * T
        eos = bos + T
    lim = eos - bos - i_t * BT
    if lim <= 0:
        return

    o = tl.arange(0, 16)
    st = HV * BT
    base = (bos + i_t * BT) * st + i_hv * BT
    r = o[:, None] * st
    c = o[None, :]
    tri = o[:, None] > o[None, :]
    m0 = (o < lim)[:, None]
    m1 = (16 + o < lim)[:, None]

    a11 = tl.where(tri, tl.load(A + base + r + c, mask=m0, other=0.0).to(tl.float32), 0.0)
    a22 = tl.where(
        tri,
        tl.load(A + base + 16 * st + r + 16 + c, mask=m1, other=0.0).to(tl.float32),
        0.0,
    )
    i11 = _inv16(a11)
    i22 = _inv16(a22)
    a21 = tl.load(A + base + 16 * st + r + c, mask=m1, other=0.0).to(tl.float32)
    # Block elimination: the (2,1) block of the inverse is -I22 A21 I11.
    i21 = -tl.dot(tl.dot(i22, a21), i11)

    ty = Ai.dtype.element_ty
    tl.store(Ai + base + r + c, i11.to(ty), mask=m0)
    tl.store(Ai + base + 16 * st + r + 16 + c, i22.to(ty), mask=m1)
    tl.store(Ai + base + 16 * st + r + c, i21.to(ty), mask=m1)


@triton.jit
def _solve_tril_64_kernel(
    A,
    Ai,
    Cu,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """BT=64: four diagonal blocks plus the 4x4 block-elimination stitch."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_n = i_bh // HV
    i_hv = i_bh % HV

    if IS_VARLEN:
        bos = tl.load(Cu + i_n).to(tl.int32)
        eos = tl.load(Cu + i_n + 1).to(tl.int32)
    else:
        bos = i_n * T
        eos = bos + T
    lim = eos - bos - i_t * BT
    if lim <= 0:
        return

    o = tl.arange(0, 16)
    st = HV * BT
    base = (bos + i_t * BT) * st + i_hv * BT
    r = o[:, None] * st
    c = o[None, :]
    tri = o[:, None] > o[None, :]
    m0 = (o < lim)[:, None]
    m1 = (16 + o < lim)[:, None]
    m2 = (32 + o < lim)[:, None]
    m3 = (48 + o < lim)[:, None]

    a11 = tl.where(tri, tl.load(A + base + r + c, mask=m0, other=0.0).to(tl.float32), 0.0)
    a22 = tl.where(
        tri,
        tl.load(A + base + 16 * st + r + 16 + c, mask=m1, other=0.0).to(tl.float32),
        0.0,
    )
    a33 = tl.where(
        tri,
        tl.load(A + base + 32 * st + r + 32 + c, mask=m2, other=0.0).to(tl.float32),
        0.0,
    )
    a44 = tl.where(
        tri,
        tl.load(A + base + 48 * st + r + 48 + c, mask=m3, other=0.0).to(tl.float32),
        0.0,
    )
    i11 = _inv16(a11)
    i22 = _inv16(a22)
    i33 = _inv16(a33)
    i44 = _inv16(a44)

    a21 = tl.load(A + base + 16 * st + r + c, mask=m1, other=0.0).to(tl.float32)
    a31 = tl.load(A + base + 32 * st + r + c, mask=m2, other=0.0).to(tl.float32)
    a32 = tl.load(A + base + 32 * st + r + 16 + c, mask=m2, other=0.0).to(tl.float32)
    a41 = tl.load(A + base + 48 * st + r + c, mask=m3, other=0.0).to(tl.float32)
    a42 = tl.load(A + base + 48 * st + r + 16 + c, mask=m3, other=0.0).to(tl.float32)
    a43 = tl.load(A + base + 48 * st + r + 32 + c, mask=m3, other=0.0).to(tl.float32)

    # Block forward substitution on the 4x4 block structure: each block of
    # the inverse is -Iii (sum_j Aij Ijk) over the blocks already solved.
    i21 = -tl.dot(tl.dot(i22, a21), i11)
    i32 = -tl.dot(tl.dot(i33, a32), i22)
    i43 = -tl.dot(tl.dot(i44, a43), i33)
    i31 = -tl.dot(i33, tl.dot(a31, i11) + tl.dot(a32, i21))
    i42 = -tl.dot(i44, tl.dot(a42, i22) + tl.dot(a43, i32))
    i41 = -tl.dot(i44, tl.dot(a41, i11) + tl.dot(a42, i21) + tl.dot(a43, i31))

    # Stores are ordered so each one follows the dots that last needed its
    # operands, letting fixpipe (cube -> UB writeback) drain while the cube
    # still has work, rather than queueing all ten at the end. fixpipe is
    # the largest remaining pipe share here (measured aic_fixpipe_ratio
    # 0.311 once the scalar chain was gone), but reordering alone measured
    # no better than the natural order -- it is kept because it costs
    # nothing and reads in dependency order.
    ty = Ai.dtype.element_ty
    tl.store(Ai + base + r + c, i11.to(ty), mask=m0)
    tl.store(Ai + base + 16 * st + r + 16 + c, i22.to(ty), mask=m1)
    tl.store(Ai + base + 32 * st + r + 32 + c, i33.to(ty), mask=m2)
    tl.store(Ai + base + 48 * st + r + 48 + c, i44.to(ty), mask=m3)
    tl.store(Ai + base + 16 * st + r + c, i21.to(ty), mask=m1)
    tl.store(Ai + base + 32 * st + r + 16 + c, i32.to(ty), mask=m2)
    tl.store(Ai + base + 48 * st + r + 32 + c, i43.to(ty), mask=m3)
    tl.store(Ai + base + 32 * st + r + c, i31.to(ty), mask=m2)
    tl.store(Ai + base + 48 * st + r + 16 + c, i42.to(ty), mask=m3)
    tl.store(Ai + base + 48 * st + r + c, i41.to(ty), mask=m3)


_SOLVE_KERNEL = {
    16: _solve_tril_16_kernel,
    32: _solve_tril_32_kernel,
    64: _solve_tril_64_kernel,
}


def solve_tril_950dt(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    """Compute the inverse of ``I + A`` for each chunk tile.

    Drop-in replacement for ``fla.ops.utils.solve_tril.solve_tril``. ``A``
    must be strictly lower triangular per tile (``A.triu() == 0``).

    Args:
        A: ``[B, T, HV, BT]`` with ``BT`` in 16, 32, 64.
        cu_seqlens: cumulative sequence lengths for a packed batch.
        chunk_indices: accepted for signature compatibility, ignored.
        output_dtype: dtype of the result; ``None`` keeps ``A.dtype``.

    Returns:
        ``[B, T, HV, BT]``, the per-tile inverse.
    """
    del chunk_indices
    b, t, hv, bt = A.shape
    if bt not in (16, 32, 64):
        raise ValueError(f"`A.shape[-1]` must be 16, 32, or 64, got {bt}.")
    output_dtype = A.dtype if output_dtype is None else output_dtype

    A = A.contiguous()
    # zeros, not empty: the kernels write only the lower-triangular blocks
    # (10 of 16 at BT=64), so the upper blocks have to start at zero. The
    # reference allocates the same way for the same reason. With empty_like
    # this silently returns whatever the allocator last held there -- it
    # happens to pass when the cache hands back a zeroed buffer, which is
    # why it looks correct until it does not.
    ai = torch.zeros_like(A, dtype=output_dtype)
    if t == 0:
        return ai

    n = (cu_seqlens.numel() - 1) if cu_seqlens is not None else b
    _SOLVE_KERNEL[bt][(triton.cdiv(t, bt), n * hv)](
        A.view(-1, hv, bt),
        ai.view(-1, hv, bt),
        cu_seqlens,
        t,
        HV=hv,
        BT=bt,
        IS_VARLEN=cu_seqlens is not None,
    )
    return ai


# --------------------------------------------------------------------------
# stage 3: w = Ai @ (beta * exp2(g) * k), u = Ai @ (beta * v)
# --------------------------------------------------------------------------


@triton.jit
def _recompute_w_u_kernel(
    K,
    V,
    G,
    Beta,
    Ai,
    W,
    U,
    Cu,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_n = i_bh // HV
    i_hv = i_bh % HV

    if IS_VARLEN:
        bos = tl.load(Cu + i_n).to(tl.int32)
        eos = tl.load(Cu + i_n + 1).to(tl.int32)
    else:
        bos = i_n * T
        eos = bos + T
    lim = eos - bos - i_t * BT
    if lim <= 0:
        return

    rows = tl.arange(0, BT)
    offs = bos + i_t * BT + rows
    mask = rows < lim
    i_h = i_hv // (HV // H)

    # Both [BT] loads before the [BT, BT] tile: this kernel is memory-bound
    # (measured aiv_mte2_ratio ~0.61), so the cheap loads should be in
    # flight while Ai streams in. No `other=` on them -- masked lanes never
    # reach a store, since every store carries the same row mask.
    beta = tl.load(Beta + offs * HV + i_hv, mask=mask).to(tl.float32)
    if USE_G:
        g = _EXP2(tl.load(G + offs * HV + i_hv, mask=mask).to(tl.float32))
    # Ai keeps other=0.0: its padding rows multiply into both products, so
    # undefined lanes would pollute w and u.
    ai = tl.load(
        Ai + offs[:, None] * (HV * BT) + i_hv * BT + rows[None, :],
        mask=mask[:, None],
        other=0.0,
    )

    # u and w are independent given ai, and the two write streams do not
    # conflict, so each tile is stored as soon as it is computed rather than
    # after both loops.
    for i_v in range(tl.cdiv(DV, BV)):
        rv = i_v * BV + tl.arange(0, BV)
        vmask = rv < DV
        v = tl.load(
            V + offs[:, None] * (HV * DV) + i_hv * DV + rv[None, :],
            mask=mask[:, None] & vmask[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            U + offs[:, None] * (HV * DV) + i_hv * DV + rv[None, :],
            tl.dot(ai, (v * beta[:, None]).to(ai.dtype)),
            mask=mask[:, None] & vmask[None, :],
        )

    for i_k in range(tl.cdiv(DK, BK)):
        rk = i_k * BK + tl.arange(0, BK)
        kmask = rk < DK
        k = tl.load(
            K + offs[:, None] * (H * DK) + i_h * DK + rk[None, :],
            mask=mask[:, None] & kmask[None, :],
            other=0.0,
        ).to(tl.float32)
        kb = k * beta[:, None]
        if USE_G:
            kb *= g[:, None]
        tl.store(
            W + offs[:, None] * (HV * DK) + i_hv * DK + rk[None, :],
            tl.dot(ai, kb.to(ai.dtype)),
            mask=mask[:, None] & kmask[None, :],
        )


def recompute_w_u_fwd_950dt(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the inverted WY matrix to ``beta * exp2(g) * k`` and ``beta * v``.

    Drop-in replacement for ``fla.ops.gated_delta_rule.wy_fast.
    recompute_w_u_fwd``.

    Args:
        k: ``[B, T, H, K]``.
        v: ``[B, T, HV, V]``.
        beta: ``[B, T, HV]``.
        A: ``[B, T, HV, BT]``, the *inverted* matrix from ``solve_tril``.
        g: per-chunk cumulative log gate ``[B, T, HV]``, through ``exp2``.
        cu_seqlens: cumulative sequence lengths for a packed batch.
        chunk_indices: accepted for signature compatibility, ignored.

    Returns:
        ``(w, u)`` with ``w: [B, T, HV, K]`` and ``u: [B, T, HV, V]``.
    """
    del chunk_indices
    b, t, h, dk = k.shape
    hv, dv = v.shape[2], v.shape[3]
    bt = A.shape[-1]
    if hv % h:
        raise ValueError("HV must be a multiple of H")
    if dk > 128 or dv > 128:
        raise NotImplementedError("recompute_w_u_fwd on 950DT supports K, V <= 128")

    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    A = A.contiguous()
    if g is not None:
        g = g.contiguous()

    w = k.new_empty(b, t, hv, dk)
    u = torch.empty_like(v)
    if t == 0:
        return w, u

    n = (cu_seqlens.numel() - 1) if cu_seqlens is not None else b
    _recompute_w_u_kernel[(triton.cdiv(t, bt), n * hv)](
        k.view(-1, h, dk),
        v.view(-1, hv, dv),
        g.view(-1, hv) if g is not None else None,
        beta.view(-1, hv),
        A.view(-1, hv, bt),
        w.view(-1, hv, dk),
        u.view(-1, hv, dv),
        cu_seqlens,
        t,
        H=h,
        HV=hv,
        DK=dk,
        DV=dv,
        BT=bt,
        BK=_tile(dk),
        BV=_tile(dv),
        USE_G=g is not None,
        IS_VARLEN=cu_seqlens is not None,
    )
    return w, u


# --------------------------------------------------------------------------
# composed entry
# --------------------------------------------------------------------------


def chunk_gated_delta_rule_fwd_intra_npu_unfused(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``chunk_gated_delta_rule_fwd_intra_npu``.

    Composes the same three operators in the same order, with the same
    signature, layouts, gate convention and return triple ``(w, u, A)``
    where ``A`` is the *inverted* matrix.
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
    if v.shape[:2] != k.shape[:2]:
        raise ValueError("v must share batch and token extents with k")
    if beta.shape != (k.shape[0], k.shape[1], v.shape[2]):
        raise ValueError("beta must be [B, T, HV]")
    if g is not None and g.shape != beta.shape:
        raise ValueError("g must be [B, T, HV]")
    if cu_seqlens is not None:
        if k.shape[0] != 1:
            raise ValueError("packed varlen input requires B == 1")
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1D [N+1] prefix-sum tensor")

    BT = chunk_size

    # Unfused kkt + solve_tril path, matching the reference stage for stage.
    A = chunk_scaled_dot_kkt_fwd_950dt(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
        output_dtype=torch.float32,
    )
    A = solve_tril_950dt(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd_950dt(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, A


# The FLA-facing name resolves to the three-stage path. The fused
# single-kernel version stays importable as
# ``chunk_gated_delta_rule_fwd_intra_npu_fused.chunk_gated_delta_rule_fwd_intra_npu_fused``.


__all__ = [
    "chunk_gated_delta_rule_fwd_intra_npu_unfused",
    "chunk_scaled_dot_kkt_fwd_950dt",
    "recompute_w_u_fwd_950dt",
    "solve_tril_950dt",
]

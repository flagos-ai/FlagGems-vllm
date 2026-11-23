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

"""Standalone mHC pre operator for Ascend NPU: ``npu_mhc_pre_clamp_sinkhorn``.

Port of the generic Triton ``mhc_pre`` variants (copied from
``flaggems_vllm.ops.mhc.mhc_pre`` so that the generic fallback stays
untouched), with two deliberate deltas:

* **Model contract ``clamp(-30, 30)`` before Sinkhorn** - the kernels in this
  module take ``clamp_logits`` as a compile-time flag and the operator always
  enables it: both the ``hc_mult == 4`` TLE fast path and the generic clamped
  kernel that every other case (other ``hc_mult`` values, or a build without
  TLE) falls back to satisfy the contract. Provenance:
  XingChen4-29B-A4B checkpoint contract (formal_issues/OP-ACC-003,
  REVIEW.md: "Model contract requires clamp(-30, 30) before Sinkhorn").
* **Dynamic Sinkhorn loop** - ``sinkhorn_repeat`` is a runtime argument
  (``range`` instead of ``tl.static_range``), so ONE compiled artifact
  serves every repeat count; first-call compilation drops from ~215-350s
  (fresh constexpr set x ~10 autotune configs, measured baseline) to a
  single small compile. P0-verified on npu: dynamic bound n=3 -> n=20
  reuses the same artifact (0.000s), numerics bitwise-identical.

Shape/dtype contract (ValueError) and the NPU-only device gate are kept
from the original standalone implementation; outputs follow the generic
contract: post ``(..., hc, 1)``, comb ``(..., hc, hc)``, hin ``(..., D)``.
For ``hc_mult == 4`` the operator now runs a TLE fast path on a transposed
(mix-major) layout; see the TLE section below for the layout reasoning. The
tiled Triton twins above remain the fallback when TLE is unavailable.
"""

from __future__ import annotations

import weakref

import torch
import triton
import triton.language as tl

__all__ = [
    "npu_mhc_pre_clamp_sinkhorn",
    "mhc_pre_ascend_proj_kernel",
    "mhc_pre_ascend_mix_kernel",
    "mhc_pre_ascend_reduce_kernel",
    "mhc_pre_ascend_tle_proj_kernel",
    "mhc_pre_ascend_tle_mix_kernel",
    "mhc_pre_ascend_tle_reduce_kernel",
]

# XingChen4 checkpoint contract: comb logits are clamped into this range
# before the Sinkhorn rounds (formal_issues/OP-ACC-003).
CLAMP_MIN = -30.0
CLAMP_MAX = 30.0


_FN_BF16_CACHE: weakref.WeakKeyDictionary[torch.Tensor, tuple[int, torch.Tensor]] = (
    weakref.WeakKeyDictionary()
)


def _get_fn_bf16_cached(fn: torch.Tensor) -> torch.Tensor:
    if fn.requires_grad or torch.is_grad_enabled():
        return fn.to(dtype=torch.bfloat16)
    version = fn._version
    cached = _FN_BF16_CACHE.get(fn)
    if cached is not None:
        cached_version, cached_bf16 = cached
        if cached_version == version:
            return cached_bf16
    fn_bf16 = fn.to(dtype=torch.bfloat16)
    _FN_BF16_CACHE[fn] = (version, fn_bf16)
    return fn_bf16


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
    ],
    key=["hidden_size", "num_tokens_bucket", "HC"],
)
@triton.jit
def mhc_pre_generic_clamped_kernel(
    gemm_out_ptr,  # (num_tokens, hc_mult3), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, HC, hidden_size), bfloat16
    post_mix_ptr,  # (num_tokens, HC), float32
    comb_mix_ptr,  # (num_tokens, HC*HC), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat,
    clamp_logits: tl.constexpr,
    HC,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= num_tokens:
        return

    res_base = pid_n * res_stride_n
    go_base = pid_n * (HC * 2 + HC * HC)
    comb_base = pid_n * (HC * HC)

    sq = 0.0
    for k in range(HC):
        head_base = res_base + k * res_stride_i
        for h_start in range(0, hidden_size, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < hidden_size
            v = tl.load(
                residual_ptr + head_base + h_offsets * res_stride_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            sq += tl.sum(v * v)

    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    for i in range(HC):
        post_i = (
            tl.sigmoid(
                tl.load(gemm_out_ptr + go_base + HC + i) * rms_inv * scale_1
                + tl.load(hc_base_ptr + HC + i)
            )
            * hc_post_mult_value
        )
        tl.store(post_mix_ptr + pid_n * HC + i, post_i)

    cb = 2 * HC
    for i in range(HC):
        for j in range(HC):
            idx = i * HC + j
            v = tl.load(
                gemm_out_ptr + go_base + cb + idx
            ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + idx)
            if clamp_logits:
                v = tl.clamp(v, -30.0, 30.0)
            tl.store(comb_mix_ptr + comb_base + idx, v)

    for i in range(HC):
        row_max = tl.load(comb_mix_ptr + comb_base + i * HC + 0)
        for j in range(1, HC):
            row_max = tl.maximum(
                row_max, tl.load(comb_mix_ptr + comb_base + i * HC + j)
            )

        row_sum = 0.0
        for j in range(HC):
            e = tl.exp(tl.load(comb_mix_ptr + comb_base + i * HC + j) - row_max)
            tl.store(comb_mix_ptr + comb_base + i * HC + j, e)
            row_sum += e

        inv_row_sum = 1.0 / row_sum
        for j in range(HC):
            v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
            tl.store(
                comb_mix_ptr + comb_base + i * HC + j, v * inv_row_sum + hc_sinkhorn_eps
            )

    for j in range(HC):
        col_sum = 0.0
        for i in range(HC):
            col_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
        inv_col_sum = 1.0 / (col_sum + hc_sinkhorn_eps)
        for i in range(HC):
            v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
            tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_col_sum)

    for _ in range(sinkhorn_repeat - 1):
        for i in range(HC):
            row_sum = 0.0
            for j in range(HC):
                row_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
            inv_row_sum = 1.0 / (row_sum + hc_sinkhorn_eps)
            for j in range(HC):
                v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
                tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_row_sum)

        for j in range(HC):
            col_sum = 0.0
            for i in range(HC):
                col_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
            inv_col_sum = 1.0 / (col_sum + hc_sinkhorn_eps)
            for i in range(HC):
                v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
                tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_col_sum)

    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k in range(HC):
            pre_k = (
                tl.sigmoid(
                    tl.load(gemm_out_ptr + go_base + k) * rms_inv * scale_0
                    + tl.load(hc_base_ptr + k)
                )
                + hc_pre_eps
            )
            rk = tl.load(
                residual_ptr + res_base + k * res_stride_i + h_offsets * res_stride_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            acc += pre_k * rk

        tl.store(
            layer_input_ptr + pid_n * li_stride_n + h_offsets * li_stride_h,
            acc.to(tl.bfloat16),
            mask=h_mask,
        )


def _validate_inputs(residual, fn, hc_scale, hc_base):
    """Shape/dtype validation. Raises ValueError on any mismatch."""
    if not isinstance(residual, torch.Tensor) or residual.dim() < 3:
        raise ValueError(
            "residual must be a tensor of shape (..., hc_mult, hidden), got "
            "{!r}".format(getattr(residual, "shape", residual))
        )
    if residual.dtype != torch.bfloat16:
        raise ValueError("residual must be bfloat16, got {}".format(residual.dtype))
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    if (
        not isinstance(fn, torch.Tensor)
        or fn.dim() != 2
        or tuple(fn.shape)
        != (
            hc_mult3,
            hc_mult * hidden_size,
        )
    ):
        raise ValueError(
            "fn must have shape ({}, {}), got {!r}".format(
                hc_mult3, hc_mult * hidden_size, tuple(getattr(fn, "shape", ()))
            )
        )
    if fn.dtype != torch.float32:
        raise ValueError("fn must be float32, got {}".format(fn.dtype))
    if tuple(hc_scale.shape) != (3,):
        raise ValueError(
            "hc_scale must have shape (3,), got {}".format(tuple(hc_scale.shape))
        )
    if hc_scale.dtype != torch.float32:
        raise ValueError("hc_scale must be float32, got {}".format(hc_scale.dtype))
    if tuple(hc_base.shape) != (hc_mult3,):
        raise ValueError(
            "hc_base must have shape ({},), got {}".format(
                hc_mult3, tuple(hc_base.shape)
            )
        )
    if hc_base.dtype != torch.float32:
        raise ValueError("hc_base must be float32, got {}".format(hc_base.dtype))
    return hc_mult, hidden_size, hc_mult3


# ===========================================================================
# TLE fast path for hc_mult == 4
# ===========================================================================
#
# Why this path exists
# --------------------
# mHC is dominated by "many 4-wide operations": the projection has only
# ``hc_mult3 == 24`` output columns and the Sinkhorn normalisation is a
# per-token 4x4 matrix.  Expressed token-major, every one of those operations
# becomes a cross-lane reduction over 4 adjacent elements (or a strided
# access), which is exactly what the Ascend vector unit is worst at: the
# generic tiled kernel spends ~376us at 4096x1280 on the mix stage alone,
# i.e. 65% of the operator, for a few dozen flops per token.
#
# The fix is a layout transform, done for free by transposing the matmul:
# instead of ``x @ fn^T -> (T, 24)`` we ask the cube for
# ``fn @ x^T -> (NC, T)`` by swapping the A/B operands (``tl.dot`` of the two
# transposed tiles).  The measurement shows the swap costs nothing
# (46.5us vs 46.5us at 4096x1280) and hands the epilogue a layout where
#   * every mix column is one contiguous row of ``T`` values,
#   * the 4x4 Sinkhorn becomes 8 element-wise vector adds + 32 multiplies per
#     round, with no reduction primitive and no layout conversion at all,
#   * the 20 Sinkhorn rounds live in UB as plain vectors.
# TLE (``triton.experimental.tle.dsa``) is used for the mix stage: one 2D
# DMA per program brings the whole ``(NC, BLOCK_T)`` tile from GM to UB
# (``dsa.alloc`` / ``dsa.copy`` / ``dsa.to_tensor``), so the fine-grained
# per-token work runs on wide, contiguous UB vectors, and the result is
# written back as contiguous rows.
#
# Stage layout (all stage boundaries are contiguous rows over tokens):
#   proj   : x(T,4D) @ fn^T -> mix_t(NC,T) f32 + sq(T)     [AB swapped dot]
#   mix    : Sinkhorn/gates -> pre(4,T) mix-major, plus post(T,4) / comb(T,16)
#            rebuilt token-major inside the kernel with interleave trees
#   reduce : hin(T,D) = sum_k pre[k,:] * residual[:,k,:]   [tiled 2D tiles]
# Rebuilding post/comb in-kernel is deliberate: an ATen ``.t().contiguous()``
# after the fact costs ~175us at 256 tokens (enough to dominate short
# sequences), and 20 scattered 4-byte stores measure ~20x slower than the one
# contiguous store per output that the interleave tree enables.
try:
    import triton.experimental.tle as tle

    _HAS_TLE = hasattr(tle, "dsa") and hasattr(tle.dsa, "alloc")
except (ImportError, AttributeError):  # stock Triton: no TLE surface
    tle = None
    _HAS_TLE = False

# Tile sizes below are fixed heuristics rather than @triton.autotune configs:
# autotuning this stage means compiling ~18 artifacts per shape key and was
# measured at ~150s per new shape (the tiled kernels above pay that), so the
# tiles here were chosen by hand on 910B4 and are scaled by token count below.
# Revisit if a shape shows up that these tiles fit badly.
#
# hc_mult3 == 24 mix rows, padded to 32 so the proj column mask disappears
# (an unmasked tile is worth ~3.5x at large K) and 32 is a power of two.
_TLE_NC = 32
# proj: 128 is the best tile at 4096 tokens (46.5us at 4096x1280, the same as
# the token-major tile); smaller token counts shrink it so the K loop still
# spreads over every AI core.
_TLE_PROJ_BLOCK_M = 128
_TLE_PROJ_BLOCK_K = 512
_TLE_PROJ_BLOCK_K_SQ = 256
# mix: up to 256 tokens per program; the heuristic keeps enough programs for
# short sequences (64 at 1024 tokens) where the per-program overhead dominates.
_TLE_MIX_BLOCK_T = 256
# reduce: the stage walks the hidden axis inside each program, so a small token
# tile buys occupancy (16x256 = five 16-bit tiles + the f32 accumulator fits
# comfortably in UB, and 256 programs at 4096 tokens fill the vector cores).
# Short sequences already have few token blocks, so they keep the wider tile.
_TLE_REDUCE_BLOCK_T = 16
_TLE_REDUCE_BLOCK_T_SHORT = 32
_TLE_REDUCE_BLOCK_D = 256

# id -> (weakref to fn, fn._version, fn.data_ptr(), padded bt). Keyed by id():
# a WeakKeyDictionary compares the two referents with Tensor.__eq__, which is
# element-wise and raises for tensors with more than one element.
_TLE_BT_CACHE: dict[int, tuple[weakref.ref, int, int, torch.Tensor]] = {}


def _tle_proj_block_m(num_tokens: int) -> int:
    """BLOCK_M for the projection: keep enough programs for the K loop."""
    want = triton.next_power_of_2(max(1, (num_tokens + 15) // 16))
    return max(16, min(_TLE_PROJ_BLOCK_M, want))


def _tle_mix_block_t(num_tokens: int) -> int:
    """Tokens per mix program: wide tiles for long sequences, more for short."""
    want = triton.next_power_of_2(max(1, (num_tokens + 15) // 16))
    return max(32, min(_TLE_MIX_BLOCK_T, want))


def _tle_reduce_block_t(num_tokens: int) -> int:
    """Token block for the reduce stage, which now walks the hidden axis.

    Large token counts need the occupancy smaller UB tiles buy; short
    sequences already have few token blocks, so keep those larger.
    """
    return _TLE_REDUCE_BLOCK_T_SHORT if num_tokens <= 1024 else _TLE_REDUCE_BLOCK_T


def _build_bt_padded(fn: torch.Tensor) -> torch.Tensor:
    bt = torch.zeros(fn.shape[1], _TLE_NC, dtype=torch.bfloat16, device=fn.device)
    bt[:, : fn.shape[0]] = fn.to(dtype=torch.bfloat16).t()
    return bt


def _get_bt_padded_cached(fn: torch.Tensor) -> torch.Tensor:
    """``fn^T`` as ``(hc_hidden, 32)`` bfloat16, zero padded, cached.

    Invalidated on identity, ``_version`` and ``data_ptr`` (so an in-place edit
    rebuilds it), the same contract the generic op uses for its packed weights.
    Caching matters: the strided bf16 copy costs ~125us at 4096x4096, far more
    than the kernels themselves.
    """
    key = id(fn)
    version = fn._version
    ptr = fn.data_ptr()
    entry = _TLE_BT_CACHE.get(key)
    if (
        entry is not None
        and entry[0]() is fn
        and entry[1] == version
        and entry[2] == ptr
    ):
        return entry[3]
    if len(_TLE_BT_CACHE) > 64:
        _TLE_BT_CACHE.clear()
    bt = _build_bt_padded(fn)
    _TLE_BT_CACHE[key] = (weakref.ref(fn), version, ptr, bt)
    return bt


@triton.jit
def _tle_row(t, r, BLOCK_T: tl.constexpr):
    """Row ``r`` of a ``(NC, BLOCK_T)`` UB tensor as a ``(BLOCK_T,)`` vector."""
    return tl.reshape(tle.dsa.extract_slice(t, (r, 0), (1, BLOCK_T), (1, 1)), [BLOCK_T])


@triton.jit
def _tle_comb_logit(
    t, r, rms_inv, scale, base, CLAMP_LOGITS: tl.constexpr, BLOCK_T: tl.constexpr
):
    """One comb logit over the token axis, clamped when the model requires it."""
    v = _tle_row(t, r, BLOCK_T) * rms_inv * scale + base
    if CLAMP_LOGITS:
        v = tl.clamp(v, -30.0, 30.0)
    return v


@triton.jit
def mhc_pre_ascend_tle_proj_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, NC) bfloat16, fn^T zero padded to NC columns
    out_t_ptr,  # (NC, out_stride) float32, transposed mix layout
    sq_ptr,  # (num_tokens,) float32 -- row sum of squares of a
    num_tokens,
    hc_hidden_size,
    a_stride_n,
    bt_stride_k,
    out_stride,
    NC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K_SQ: tl.constexpr,
):
    """``fn @ x^T`` (AB swapped), i.e. the transposed mix, plus the RMS sumsq.

    Swapping A/B is what makes the transpose free: the cube computes
    ``(NC, BLOCK_M)`` instead of ``(BLOCK_M, NC)``, so the epilogue reads one
    contiguous row per mix column and never transposes a tile.
    """
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NC)
    m_mask = offs_m < num_tokens
    acc_t = tl.zeros((NC, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hc_hidden_size
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # bt is zero padded to NC columns, so no column mask is needed.
        b = tl.load(
            bt_ptr + offs_k[:, None] * bt_stride_k + offs_n[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        acc_t += tl.dot(tl.trans(b), tl.trans(a))
    tl.store(
        out_t_ptr + offs_n[:, None] * out_stride + offs_m[None, :],
        acc_t,
        mask=m_mask[None, :],
    )
    # The sum of squares needs its own loop.  A vector reduction inside the dot
    # loop makes the compiler mis-schedule the MIX (AIC/AIV) loop: with a large
    # BLOCK_K it fails to tile (UB overflow + "'scf.for' op Failed to collect
    # vector loop tiling info"), and with a small one it compiles but the
    # reduction reads stale UB - sq comes back NaN/inf and changes between
    # identical runs, while the dot output stays bit-identical.  Keep the two
    # loops separate.
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K_SQ):
        offs_k = k0 + tl.arange(0, BLOCK_K_SQ)
        k_mask = offs_k < hc_hidden_size
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        a_f32 = a.to(tl.float32)
        sq += tl.sum(a_f32 * a_f32, 1)
    tl.store(sq_ptr + offs_m, sq, mask=m_mask)


@triton.jit
def mhc_pre_ascend_tle_mix_kernel(
    mix_t_ptr,  # (NC, mix_stride) float32, transposed mix
    sq_ptr,  # (num_tokens,) float32
    hc_scale_ptr,  # (3,) float32
    hc_base_ptr,  # (hc_mult3,) float32
    pre_ptr,  # (4, mix_stride) float32  -- gate consumed by the reduce stage
    post_t_ptr,  # (num_tokens, 4) float32, token-major public layout
    comb_t_ptr,  # (num_tokens, 16) float32, token-major public layout
    num_tokens,
    hc_hidden_size,
    mix_stride,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat,
    NC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    CLAMP_LOGITS: tl.constexpr,
):
    """Gates + Sinkhorn for a tile of ``BLOCK_T`` tokens, one row per mix column.

    Every quantity here is a vector over the token axis, so the 20 Sinkhorn
    rounds are pure element-wise arithmetic: 4 row sums and 4 column sums are
    the only "reductions" and they are written as explicit vector adds (never
    ``tl.sum``), which is what keeps the cost down to ~35us at 4096x1280.
    """
    pid = tl.program_id(0)
    t0 = pid * BLOCK_T
    t_off = t0 + tl.arange(0, BLOCK_T)
    rows = tl.arange(0, NC)
    mask = t_off < num_tokens
    # The caller pads the sequence to BLOCK_T, so every program copies a full
    # tile: a partial inner extent changes the UB row stride and would scramble
    # the tile. Padding lanes are masked on store and never read back.
    mix_ub = tle.dsa.alloc(
        [NC, BLOCK_T], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(
            mix_t_ptr + rows[:, None] * mix_stride + t_off[None, :],
            mix_ub,
            [NC, BLOCK_T],
        )
    mt = tle.dsa.to_tensor(mix_ub)

    sq_ub = tle.dsa.alloc([BLOCK_T], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(sq_ptr + t_off, sq_ub, [BLOCK_T])
    sq = tle.dsa.to_tensor(sq_ub)
    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    s0 = tl.load(hc_scale_ptr + 0)
    s1 = tl.load(hc_scale_ptr + 1)
    s2 = tl.load(hc_scale_ptr + 2)

    b0 = tl.load(hc_base_ptr + 0)
    b1 = tl.load(hc_base_ptr + 1)
    b2 = tl.load(hc_base_ptr + 2)
    b3 = tl.load(hc_base_ptr + 3)
    b4 = tl.load(hc_base_ptr + 4)
    b5 = tl.load(hc_base_ptr + 5)
    b6 = tl.load(hc_base_ptr + 6)
    b7 = tl.load(hc_base_ptr + 7)
    b8 = tl.load(hc_base_ptr + 8)
    b9 = tl.load(hc_base_ptr + 9)
    b10 = tl.load(hc_base_ptr + 10)
    b11 = tl.load(hc_base_ptr + 11)
    b12 = tl.load(hc_base_ptr + 12)
    b13 = tl.load(hc_base_ptr + 13)
    b14 = tl.load(hc_base_ptr + 14)
    b15 = tl.load(hc_base_ptr + 15)
    b16 = tl.load(hc_base_ptr + 16)
    b17 = tl.load(hc_base_ptr + 17)
    b18 = tl.load(hc_base_ptr + 18)
    b19 = tl.load(hc_base_ptr + 19)
    b20 = tl.load(hc_base_ptr + 20)
    b21 = tl.load(hc_base_ptr + 21)
    b22 = tl.load(hc_base_ptr + 22)
    b23 = tl.load(hc_base_ptr + 23)

    p0 = tl.sigmoid(_tle_row(mt, 0, BLOCK_T) * rms_inv * s0 + b0) + hc_pre_eps
    p1 = tl.sigmoid(_tle_row(mt, 1, BLOCK_T) * rms_inv * s0 + b1) + hc_pre_eps
    p2 = tl.sigmoid(_tle_row(mt, 2, BLOCK_T) * rms_inv * s0 + b2) + hc_pre_eps
    p3 = tl.sigmoid(_tle_row(mt, 3, BLOCK_T) * rms_inv * s0 + b3) + hc_pre_eps

    q0 = tl.sigmoid(_tle_row(mt, 4, BLOCK_T) * rms_inv * s1 + b4) * hc_post_mult_value
    q1 = tl.sigmoid(_tle_row(mt, 5, BLOCK_T) * rms_inv * s1 + b5) * hc_post_mult_value
    q2 = tl.sigmoid(_tle_row(mt, 6, BLOCK_T) * rms_inv * s1 + b6) * hc_post_mult_value
    q3 = tl.sigmoid(_tle_row(mt, 7, BLOCK_T) * rms_inv * s1 + b7) * hc_post_mult_value

    # Model contract: clamp the comb logits before Sinkhorn (CLAMP_LOGITS).
    c00 = _tle_comb_logit(mt, 8, rms_inv, s2, b8, CLAMP_LOGITS, BLOCK_T)
    c01 = _tle_comb_logit(mt, 9, rms_inv, s2, b9, CLAMP_LOGITS, BLOCK_T)
    c02 = _tle_comb_logit(mt, 10, rms_inv, s2, b10, CLAMP_LOGITS, BLOCK_T)
    c03 = _tle_comb_logit(mt, 11, rms_inv, s2, b11, CLAMP_LOGITS, BLOCK_T)
    c10 = _tle_comb_logit(mt, 12, rms_inv, s2, b12, CLAMP_LOGITS, BLOCK_T)
    c11 = _tle_comb_logit(mt, 13, rms_inv, s2, b13, CLAMP_LOGITS, BLOCK_T)
    c12 = _tle_comb_logit(mt, 14, rms_inv, s2, b14, CLAMP_LOGITS, BLOCK_T)
    c13 = _tle_comb_logit(mt, 15, rms_inv, s2, b15, CLAMP_LOGITS, BLOCK_T)
    c20 = _tle_comb_logit(mt, 16, rms_inv, s2, b16, CLAMP_LOGITS, BLOCK_T)
    c21 = _tle_comb_logit(mt, 17, rms_inv, s2, b17, CLAMP_LOGITS, BLOCK_T)
    c22 = _tle_comb_logit(mt, 18, rms_inv, s2, b18, CLAMP_LOGITS, BLOCK_T)
    c23 = _tle_comb_logit(mt, 19, rms_inv, s2, b19, CLAMP_LOGITS, BLOCK_T)
    c30 = _tle_comb_logit(mt, 20, rms_inv, s2, b20, CLAMP_LOGITS, BLOCK_T)
    c31 = _tle_comb_logit(mt, 21, rms_inv, s2, b21, CLAMP_LOGITS, BLOCK_T)
    c32 = _tle_comb_logit(mt, 22, rms_inv, s2, b22, CLAMP_LOGITS, BLOCK_T)
    c33 = _tle_comb_logit(mt, 23, rms_inv, s2, b23, CLAMP_LOGITS, BLOCK_T)

    # row softmax over j, then the first column normalisation
    m0 = tl.maximum(tl.maximum(c00, c01), tl.maximum(c02, c03))
    e00 = tl.exp(c00 - m0)
    e01 = tl.exp(c01 - m0)
    e02 = tl.exp(c02 - m0)
    e03 = tl.exp(c03 - m0)
    r0 = e00 + e01 + e02 + e03
    c00 = e00 / r0 + hc_sinkhorn_eps
    c01 = e01 / r0 + hc_sinkhorn_eps
    c02 = e02 / r0 + hc_sinkhorn_eps
    c03 = e03 / r0 + hc_sinkhorn_eps

    m1 = tl.maximum(tl.maximum(c10, c11), tl.maximum(c12, c13))
    e10 = tl.exp(c10 - m1)
    e11 = tl.exp(c11 - m1)
    e12 = tl.exp(c12 - m1)
    e13 = tl.exp(c13 - m1)
    r1 = e10 + e11 + e12 + e13
    c10 = e10 / r1 + hc_sinkhorn_eps
    c11 = e11 / r1 + hc_sinkhorn_eps
    c12 = e12 / r1 + hc_sinkhorn_eps
    c13 = e13 / r1 + hc_sinkhorn_eps

    m2 = tl.maximum(tl.maximum(c20, c21), tl.maximum(c22, c23))
    e20 = tl.exp(c20 - m2)
    e21 = tl.exp(c21 - m2)
    e22 = tl.exp(c22 - m2)
    e23 = tl.exp(c23 - m2)
    r2 = e20 + e21 + e22 + e23
    c20 = e20 / r2 + hc_sinkhorn_eps
    c21 = e21 / r2 + hc_sinkhorn_eps
    c22 = e22 / r2 + hc_sinkhorn_eps
    c23 = e23 / r2 + hc_sinkhorn_eps

    m3 = tl.maximum(tl.maximum(c30, c31), tl.maximum(c32, c33))
    e30 = tl.exp(c30 - m3)
    e31 = tl.exp(c31 - m3)
    e32 = tl.exp(c32 - m3)
    e33 = tl.exp(c33 - m3)
    r3 = e30 + e31 + e32 + e33
    c30 = e30 / r3 + hc_sinkhorn_eps
    c31 = e31 / r3 + hc_sinkhorn_eps
    c32 = e32 / r3 + hc_sinkhorn_eps
    c33 = e33 / r3 + hc_sinkhorn_eps

    cs0 = c00 + c10 + c20 + c30
    cs1 = c01 + c11 + c21 + c31
    cs2 = c02 + c12 + c22 + c32
    cs3 = c03 + c13 + c23 + c33
    i0 = 1.0 / (cs0 + hc_sinkhorn_eps)
    i1 = 1.0 / (cs1 + hc_sinkhorn_eps)
    i2 = 1.0 / (cs2 + hc_sinkhorn_eps)
    i3 = 1.0 / (cs3 + hc_sinkhorn_eps)
    c00 *= i0
    c10 *= i0
    c20 *= i0
    c30 *= i0
    c01 *= i1
    c11 *= i1
    c21 *= i1
    c31 *= i1
    c02 *= i2
    c12 *= i2
    c22 *= i2
    c32 *= i2
    c03 *= i3
    c13 *= i3
    c23 *= i3
    c33 *= i3

    for _ in range(sinkhorn_repeat - 1):
        r0 = c00 + c01 + c02 + c03
        i0 = 1.0 / (r0 + hc_sinkhorn_eps)
        c00 *= i0
        c01 *= i0
        c02 *= i0
        c03 *= i0
        r1 = c10 + c11 + c12 + c13
        i1 = 1.0 / (r1 + hc_sinkhorn_eps)
        c10 *= i1
        c11 *= i1
        c12 *= i1
        c13 *= i1
        r2 = c20 + c21 + c22 + c23
        i2 = 1.0 / (r2 + hc_sinkhorn_eps)
        c20 *= i2
        c21 *= i2
        c22 *= i2
        c23 *= i2
        r3 = c30 + c31 + c32 + c33
        i3 = 1.0 / (r3 + hc_sinkhorn_eps)
        c30 *= i3
        c31 *= i3
        c32 *= i3
        c33 *= i3

        cs0 = c00 + c10 + c20 + c30
        cs1 = c01 + c11 + c21 + c31
        cs2 = c02 + c12 + c22 + c32
        cs3 = c03 + c13 + c23 + c33
        i0 = 1.0 / (cs0 + hc_sinkhorn_eps)
        i1 = 1.0 / (cs1 + hc_sinkhorn_eps)
        i2 = 1.0 / (cs2 + hc_sinkhorn_eps)
        i3 = 1.0 / (cs3 + hc_sinkhorn_eps)
        c00 *= i0
        c10 *= i0
        c20 *= i0
        c30 *= i0
        c01 *= i1
        c11 *= i1
        c21 *= i1
        c31 *= i1
        c02 *= i2
        c12 *= i2
        c22 *= i2
        c32 *= i2
        c03 *= i3
        c13 *= i3
        c23 *= i3
        c33 *= i3

    # ``pre`` stays mix-major: it is the only gate the reduce stage reads, and
    # one contiguous row per head is exactly what that stage wants.
    tl.store(pre_ptr + 0 * mix_stride + t_off, p0, mask=mask)
    tl.store(pre_ptr + 1 * mix_stride + t_off, p1, mask=mask)
    tl.store(pre_ptr + 2 * mix_stride + t_off, p2, mask=mask)
    tl.store(pre_ptr + 3 * mix_stride + t_off, p3, mask=mask)

    # ``post`` / ``comb`` go back to the token-major public contract here, with
    # interleave trees whose leaves are the bit-reversed pairs (v_j, v_j + NC/2):
    # a single contiguous store each, instead of 20 scattered 4-byte stores
    # (measured ~20x slower) or an ATen transpose afterwards (~175us at 256
    # tokens, which would dominate short sequences).
    post_il = tl.reshape(
        tl.interleave(tl.interleave(q0, q2), tl.interleave(q1, q3)), [BLOCK_T, 4]
    )
    e00i = tl.interleave(c00, c20)
    e01i = tl.interleave(c10, c30)
    e10i = tl.interleave(c02, c22)
    e11i = tl.interleave(c12, c32)
    o00i = tl.interleave(c01, c21)
    o01i = tl.interleave(c11, c31)
    o10i = tl.interleave(c03, c23)
    o11i = tl.interleave(c13, c33)
    eei = tl.interleave(tl.interleave(e00i, e01i), tl.interleave(e10i, e11i))
    ooi = tl.interleave(tl.interleave(o00i, o01i), tl.interleave(o10i, o11i))
    comb_il = tl.reshape(tl.interleave(eei, ooi), [BLOCK_T, 16])

    tl.store(
        post_t_ptr + t_off[:, None] * 4 + tl.arange(0, 4)[None, :],
        post_il,
        mask=mask[:, None],
    )
    tl.store(
        comb_t_ptr + t_off[:, None] * 16 + tl.arange(0, 16)[None, :],
        comb_il,
        mask=mask[:, None],
    )


@triton.jit
def mhc_pre_ascend_tle_reduce_kernel(
    pre_ptr,  # (4, pre_stride) float32
    residual_ptr,  # (num_tokens, 4, hidden_size) bfloat16
    out_ptr,  # (num_tokens, hidden_size) bfloat16
    num_tokens,
    hidden_size,
    pre_stride,
    res_stride_n,
    res_stride_i,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """``hin[t, d] = sum_k pre[k, t] * residual[t, k, d]`` on 2D contiguous tiles.

    One program owns a token block and walks the hidden axis, so the four gate
    loads happen once per block instead of once per (block, d-tile) and the
    program count drops by the number of d-tiles.
    """
    pid_t = tl.program_id(0)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < num_tokens

    p0 = tl.load(pre_ptr + 0 * pre_stride + t, mask=tmask, other=0.0)
    p1 = tl.load(pre_ptr + 1 * pre_stride + t, mask=tmask, other=0.0)
    p2 = tl.load(pre_ptr + 2 * pre_stride + t, mask=tmask, other=0.0)
    p3 = tl.load(pre_ptr + 3 * pre_stride + t, mask=tmask, other=0.0)

    for d0 in range(0, hidden_size, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m = tmask[:, None] & (d < hidden_size)[None, :]
        base = residual_ptr + t[:, None] * res_stride_n + d[None, :]
        x0 = tl.load(base + 0 * res_stride_i, mask=m, other=0.0).to(tl.float32)
        x1 = tl.load(base + 1 * res_stride_i, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(base + 2 * res_stride_i, mask=m, other=0.0).to(tl.float32)
        x3 = tl.load(base + 3 * res_stride_i, mask=m, other=0.0).to(tl.float32)
        acc = p0[:, None] * x0 + p1[:, None] * x1 + p2[:, None] * x2 + p3[:, None] * x3
        tl.store(
            out_ptr + t[:, None] * hidden_size + d[None, :],
            acc.to(tl.bfloat16),
            mask=m,
        )


def _mhc_pre_tle_4(
    residual_flat,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    clamp_logits=True,
    post_out=None,
    comb_out=None,
    hin_out=None,
):
    """Run proj -> mix -> reduce on the transposed (mix-major) layout.

    ``clamp_logits`` is the model-contract switch: ``npu_mhc_pre_clamp_sinkhorn``
    passes ``True`` (clamp(-30, 30) before Sinkhorn), the generic ``mhc_pre``
    passes ``False`` so it keeps the reference math.  It is a compile-time flag,
    so the two callers compile one artifact each and share nothing else.

    ``post_out`` / ``comb_out`` / ``hin_out`` let a caller hand in the
    already-allocated public outputs; the kernel layouts are token-major and
    match them exactly, so no copy is needed.
    """
    num_tokens = residual_flat.shape[0]
    hidden_size = residual_flat.shape[-1]
    hc_hidden_size = 4 * hidden_size
    device = residual_flat.device

    block_t = _tle_mix_block_t(num_tokens)
    block_m = _tle_proj_block_m(num_tokens)
    t_pad = triton.cdiv(num_tokens, block_t) * block_t
    mix_stride = t_pad

    bt = _get_bt_padded_cached(fn)
    # Padded to the mix tile: the mix stage reads whole BLOCK_T tiles.
    sq = torch.zeros(t_pad, dtype=torch.float32, device=device)
    # Only the first num_tokens columns are written by the projection; the
    # padding lanes are masked on store and never read back.
    mix_t = torch.empty(_TLE_NC, mix_stride, dtype=torch.float32, device=device)

    mhc_pre_ascend_tle_proj_kernel[(triton.cdiv(num_tokens, block_m),)](
        residual_flat,
        bt,
        mix_t,
        sq,
        num_tokens,
        hc_hidden_size,
        residual_flat.stride(0),
        bt.stride(0),
        mix_stride,
        NC=_TLE_NC,
        BLOCK_M=block_m,
        BLOCK_K=_TLE_PROJ_BLOCK_K,
        BLOCK_K_SQ=_TLE_PROJ_BLOCK_K_SQ,
    )

    pre_t = torch.empty(4, mix_stride, dtype=torch.float32, device=device)
    if post_out is None:
        post_out = torch.empty(num_tokens, 4, dtype=torch.float32, device=device)
    if comb_out is None:
        comb_out = torch.empty(num_tokens, 16, dtype=torch.float32, device=device)
    mhc_pre_ascend_tle_mix_kernel[(t_pad // block_t,)](
        mix_t,
        sq,
        hc_scale,
        hc_base,
        pre_t,
        post_out,
        comb_out,
        num_tokens,
        hc_hidden_size,
        mix_stride,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        NC=_TLE_NC,
        BLOCK_T=block_t,
        CLAMP_LOGITS=clamp_logits,
    )

    if hin_out is None:
        hin_out = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
    reduce_block_t = _tle_reduce_block_t(num_tokens)
    mhc_pre_ascend_tle_reduce_kernel[(triton.cdiv(num_tokens, reduce_block_t),)](
        pre_t,
        residual_flat,
        hin_out,
        num_tokens,
        hidden_size,
        mix_stride,
        residual_flat.stride(0),
        residual_flat.stride(1),
        BLOCK_T=reduce_block_t,
        BLOCK_D=_TLE_REDUCE_BLOCK_D,
    )

    return post_out, comb_out, hin_out


def npu_mhc_pre_clamp_sinkhorn(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
):
    """Standalone mHC pre block for Ascend NPU (ported Triton variants).

    Args:
        residual: (..., hc_mult, hidden), bfloat16.
        fn: (hc_mult3, hc_mult * hidden), float32 projection weights.
        hc_scale: (3,), float32 per-block scales.
        hc_base: (hc_mult3,), float32 per-block biases.
        rms_eps: RMS normalisation epsilon.
        hc_pre_eps: epsilon added to sigmoid(pre_logits).
        hc_sinkhorn_eps: Sinkhorn renormalisation epsilon.
        hc_post_mult_value: multiplier applied to sigmoid(post_logits).
        sinkhorn_repeat: Sinkhorn rounds (model uses 20); runtime argument,
            changing it does not trigger recompilation.
        n_splits: accepted for signature parity with the generic op, unused.

    Returns:
        (post, comb, hin) matching the generic ``mhc_pre`` contract, computed
        with the model-required ``clamp(-30, 30)`` before Sinkhorn.

    Execution:
        ``hc_mult == 4`` runs the TLE path below (``proj -> mix -> reduce`` on a
        transposed, mix-major layout) when TLE is available; every other case
        (other ``hc_mult`` values, or a Triton build without TLE) runs the
        clamped generic kernel, which also applies ``clamp(-30, 30)``.
    """
    if residual.device.type != "npu":
        raise NotImplementedError("npu_mhc_pre_clamp_sinkhorn requires NPU tensors")
    hc_mult, hidden_size, hc_mult3 = _validate_inputs(residual, fn, hc_scale, hc_base)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()

    # hc==4: TLE fast path on the transposed (mix-major) layout. Without TLE
    # (stock Triton) fall through to the clamped generic kernel below -- do NOT
    # delegate to the generic ``mhc_pre``, whose Ascend paths do not clamp.
    if hc_mult == 4 and _HAS_TLE:
        post_mix, comb_mix, layer_input = _mhc_pre_tle_4(
            residual_flat,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            clamp_logits=True,
        )
        post_mix = post_mix.view(*outer_shape, hc_mult, 1)
        comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
        layer_input = layer_input.view(*outer_shape, hidden_size)
        return post_mix, comb_mix, layer_input

    num_tokens = residual_flat.shape[0]
    hc_hidden_size = hc_mult * hidden_size
    if num_tokens <= 512:
        num_tokens_bucket = 1
    elif num_tokens <= 1024:
        num_tokens_bucket = 2
    elif num_tokens <= 2048:
        num_tokens_bucket = 3
    elif num_tokens <= 4096:
        num_tokens_bucket = 4
    else:
        num_tokens_bucket = 5

    # projection: bf16 GEMM (ported generic path)
    x_flat = residual_flat.reshape(num_tokens, hc_hidden_size)
    fn_bf16 = _get_fn_bf16_cached(fn)
    from flag_gems.runtime.backend._ascend.ops.mm import mm as _gems_mm

    gemm_out = _gems_mm(x_flat, fn_bf16.t()).float()

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    common = (
        gemm_out,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        num_tokens,
        num_tokens_bucket,
        residual_flat.stride(0),
        residual_flat.stride(1),
        residual_flat.stride(2),
        layer_input.stride(0),
        layer_input.stride(1),
        hidden_size,
        hc_hidden_size,
    )
    kwargs = dict(
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        clamp_logits=True,  # model contract, always on for this operator
    )
    mhc_pre_generic_clamped_kernel[(num_tokens,)](*common, HC=hc_mult, **kwargs)

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


# Tiled epilogue kernels (proj / mix / reduce): what the generic ``mhc_pre`` runs
# on Ascend.


def _autotune_bench(fn, **kwargs):
    """Timer for ``triton.autotune`` on Ascend: default timing misses device time here."""
    try:
        from triton.backends.ascend.testing import do_bench_npu
    except Exception:  # noqa: BLE001 - keep the default timing elsewhere
        return triton.testing.do_bench(fn, **kwargs)
    return do_bench_npu(fn)


@triton.autotune(
    configs=[
        # Two loops, two tile widths: the sum-of-squares loop cannot use BLOCK_K>=1024.
        # A tile that overflows UB silently miscomputes, so each entry below was checked.
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 512, "BLOCK_K_SQ": 256},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 512, "BLOCK_K_SQ": 256},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 512, "BLOCK_K_SQ": 256},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_K_SQ": 256},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 512, "BLOCK_K_SQ": 512},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 512, "BLOCK_K_SQ": 256},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 512, "BLOCK_K_SQ": 256},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["num_tokens", "hc_hidden_size", "HC_MULT3"],
    do_bench=_autotune_bench,
)
@triton.jit
def mhc_pre_ascend_proj_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, HC_MULT3) bfloat16, fn^T, zero padded to power of 2
    out_ptr,  # (num_tokens, HC_MULT3) float32
    sq_ptr,  # (num_tokens,), float32 -- row sums of squares of a
    num_tokens,
    hc_hidden_size,
    a_stride_n,
    bt_stride_k,
    bt_stride_j,
    HC_MULT3: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K_SQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BT_PADDED: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens
    n_mask = offs_n < HC_MULT3
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hc_hidden_size
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # BT_PADDED: the tile really has BLOCK_N columns, so skip the column mask
        # (unmasked is 3.5x faster at large K: the mask drops the block DMA).
        if BT_PADDED:
            b = tl.load(
                bt_ptr + offs_k[:, None] * bt_stride_k + offs_n[None, :] * bt_stride_j,
                mask=k_mask[:, None],
                other=0.0,
            )
        else:
            b = tl.load(
                bt_ptr + offs_k[:, None] * bt_stride_k + offs_n[None, :] * bt_stride_j,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
        # No vector reduction in this loop: it silently corrupts the product or NaNs.
        acc += tl.dot(a, b)
    tl.store(
        out_ptr + offs_m[:, None] * HC_MULT3 + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )
    # The sum of squares needs its own loop: one load cannot feed both units.
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K_SQ):
        offs_k = k0 + tl.arange(0, BLOCK_K_SQ)
        k_mask = offs_k < hc_hidden_size
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        a_f32 = a.to(tl.float32)
        sq += tl.sum(a_f32 * a_f32, 1)
    tl.store(sq_ptr + offs_m, sq, mask=m_mask)


@triton.jit
def _mhc_ascend_rms_inv(sq_ptr, idx, tmask, hc_hidden_size, rms_eps: tl.constexpr):
    """rsqrt(row_sum_of_squares / hidden + eps); idx/tmask are scalars or a token tile."""
    sq = tl.load(sq_ptr + idx, mask=tmask, other=1.0)
    return tl.rsqrt(sq / hc_hidden_size + rms_eps)


@triton.jit
def _mhc_ascend_gate(gemm_ptr, hc_base_ptr, base, off, tmask, rms_inv, scale, eps):
    """sigmoid(mix_logit * rms_inv * scale + base) + eps, shared by both gates."""
    logit = tl.load(gemm_ptr + base + off, mask=tmask, other=0.0)
    return tl.sigmoid(logit * rms_inv * scale + tl.load(hc_base_ptr + off)) + eps


@triton.autotune(
    configs=[
        # Tokens per program: 16 below 2048 tokens, 32 above; 64 exceeds UB.
        triton.Config({"BLOCK_N": 8}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_N": 16}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_N": 32}, num_warps=1, num_stages=1),
    ],
    key=["num_tokens"],
    do_bench=_autotune_bench,
)
@triton.jit
def mhc_pre_ascend_mix_kernel(
    gemm_out_ptr,  # (num_tokens, hc_mult3), float32
    sq_ptr,  # (num_tokens,), float32 -- row sums of squares, from the projection
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    post_mix_ptr,  # (num_tokens, HC), float32
    comb_mix_ptr,  # (num_tokens, HC * HC), float32
    num_tokens,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m = t < num_tokens
    rows = tl.arange(0, HC)
    cols = tl.arange(0, HC)
    gemm_base = t * (HC * 2 + HC * HC)

    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)
    rms_inv = _mhc_ascend_rms_inv(sq_ptr, t, m, hc_hidden_size, rms_eps)

    comb_off = gemm_base[:, None, None] + (
        2 * HC + rows[None, :, None] * HC + cols[None, None, :]
    )
    comb = tl.load(gemm_out_ptr + comb_off, mask=m[:, None, None], other=0.0)
    comb = (
        comb * rms_inv[:, None, None] * scale_2
        + tl.load(hc_base_ptr + 2 * HC + rows[:, None] * HC + cols[None, :])[None, :, :]
    )
    exponents = tl.exp(comb - tl.max(comb, 2)[:, :, None])
    comb = exponents / tl.sum(exponents, 2)[:, :, None] + hc_sinkhorn_eps
    comb = comb / (tl.sum(comb, 1)[:, None, :] + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (tl.sum(comb, 2)[:, :, None] + hc_sinkhorn_eps)
        comb = comb / (tl.sum(comb, 1)[:, None, :] + hc_sinkhorn_eps)
    tl.store(
        comb_mix_ptr
        + t[:, None, None] * (HC * HC)
        + rows[None, :, None] * HC
        + cols[None, None, :],
        comb,
        mask=m[:, None, None],
    )

    for k in tl.static_range(HC):
        # No epsilon here; the pre gate is the same expression with eps.
        post = (
            _mhc_ascend_gate(
                gemm_out_ptr, hc_base_ptr, gemm_base, HC + k, m, rms_inv, scale_1, 0.0
            )
            * hc_post_mult_value
        )
        tl.store(post_mix_ptr + t * HC + k, post, mask=m)


@triton.autotune(
    configs=[
        # BLOCK_T tokens per program, each with its own (HC, BLOCK_H) tile: a single
        # 2-D tile is lowered to scalar accesses (83 ms at 8192x4096).
        triton.Config({"BLOCK_T": 1, "BLOCK_H": 2048}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_T": 1, "BLOCK_H": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 2, "BLOCK_H": 4096}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_T": 2, "BLOCK_H": 2048}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_T": 4, "BLOCK_H": 4096}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_T": 4, "BLOCK_H": 2048}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_T": 2, "BLOCK_H": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 4, "BLOCK_H": 1024}, num_warps=4, num_stages=1),
    ],
    key=["hidden_size", "num_tokens_bucket"],
    do_bench=_autotune_bench,
)
@triton.jit
def mhc_pre_ascend_reduce_kernel(
    gemm_out_ptr,  # (num_tokens, hc_mult3), float32
    sq_ptr,  # (num_tokens,), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, HC, hidden_size), bfloat16
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, HC)
    scale_0 = tl.load(hc_scale_ptr + 0)

    for t in tl.static_range(BLOCK_T):
        pid_n = pid * BLOCK_T + t
        keep = pid_n < num_tokens
        res_base = pid_n * res_stride_n
        gemm_base = pid_n * (HC * 2 + HC * HC)

        rms_inv = _mhc_ascend_rms_inv(sq_ptr, pid_n, keep, hc_hidden_size, rms_eps)
        pre_mix = _mhc_ascend_gate(
            gemm_out_ptr,
            hc_base_ptr,
            gemm_base,
            rows,
            keep,
            rms_inv,
            scale_0,
            hc_pre_eps,
        )

        for h_start in range(0, hidden_size, BLOCK_H):
            h_offs = h_start + tl.arange(0, BLOCK_H)
            h_mask = (h_offs < hidden_size) & keep
            tile = tl.load(
                residual_ptr
                + res_base
                + rows[:, None] * res_stride_i
                + h_offs[None, :] * res_stride_h,
                mask=h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = tl.sum(pre_mix[:, None] * tile, 0)
            tl.store(
                layer_input_ptr + pid_n * li_stride_n + h_offs * li_stride_h,
                acc.to(tl.bfloat16),
                mask=h_mask,
            )

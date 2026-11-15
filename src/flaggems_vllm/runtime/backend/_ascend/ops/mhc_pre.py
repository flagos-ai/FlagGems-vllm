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

"""Standalone mHC pre operator for Ascend NPU: ``npu_mhc_pre``.

Port of the generic Triton ``mhc_pre`` variants (copied from
``flaggems_vllm.ops.mhc.mhc_pre`` so that the generic fallback stays
untouched), with two deliberate deltas:

* **Model contract ``clamp(-30, 30)`` before Sinkhorn** - the kernels in this
  module take ``clamp_logits`` as a compile-time flag and are always called
  with ``True``. ``hc_mult == 4`` returns early to the tiled ``ASCEND_TILED``
  path, which does not clamp. Provenance:
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
TLE twins are NOT ported (triton 3.2.0 < 3.6 gate is false on this box).
"""

from __future__ import annotations

import weakref

import torch
import triton
import triton.language as tl

__all__ = [
    "npu_mhc_pre",
    "mhc_pre_ascend_proj_kernel",
    "mhc_pre_ascend_mix_kernel",
    "mhc_pre_ascend_reduce_kernel",
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


def npu_mhc_pre(
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
    """
    if residual.device.type != "npu":
        raise NotImplementedError("npu_mhc_pre requires NPU tensors")
    hc_mult, hidden_size, hc_mult3 = _validate_inputs(residual, fn, hc_scale, hc_base)

    # hc==4 shares the tiled epilogue path (ASCEND_TILED), which does not clamp.
    if hc_mult == 4:
        from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre as _tiled_mhc_pre

        return _tiled_mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
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

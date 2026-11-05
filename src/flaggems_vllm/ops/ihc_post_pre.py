# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 post-scatter + split pre reduction, with native rounded residual."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _ihc_post_pre_stage1(
    x_ptr,
    residual_ptr,
    post_ptr,
    materialized_ptr,
    weight_ptr,
    partial_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    k_offsets = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K_TOTAL
    # HIDDEN_SIZE is a multiple of BLOCK_K: one split belongs to one channel.
    channel_idx = split_idx * BLOCK_K // HIDDEN_SIZE
    hidden_offsets = k_offsets % HIDDEN_SIZE
    post = tl.load(
        post_ptr + token_idx * HC_MULT + hc_idx,
        mask=hc_mask,
        other=0.0,
    )
    x = tl.load(
        x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + token_idx * K_TOTAL + k_offsets,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    channel_gate = tl.sum(tl.where(hc_idx == channel_idx, post, 0.0), axis=0)
    output = channel_gate * x + residual
    rounded = output.to(materialized_ptr.dtype.element_ty)
    tl.store(materialized_ptr + token_idx * K_TOTAL + k_offsets, rounded, mask=k_mask)
    # The original pre kernel loads the BF16 post output. Keep that rounding
    # boundary even though this instance can consume the value in registers.
    x = rounded.to(tl.float32)
    sum_squares = tl.sum(x * x, axis=0)

    weight_offsets = hc_idx[:, None] * K_TOTAL + k_offsets[None, :]
    weight_mask = hc_mask[:, None] & k_mask[None, :]
    pre_mix = tl.sum(
        tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0) * x[None, :],
        axis=1,
    )
    post_mix = tl.sum(
        tl.load(
            weight_ptr + HC_MULT * K_TOTAL + weight_offsets,
            mask=weight_mask,
            other=0.0,
        )
        * x[None, :],
        axis=1,
    )

    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, sum_squares)
    tl.store(partial + 1 + hc_idx, pre_mix, mask=hc_mask)
    tl.store(partial + 1 + HC_POW2 + hc_idx, post_mix, mask=hc_mask)


@triton.jit
def _ihc_pre_stage2(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    sum_squares = tl.zeros((), dtype=tl.float32)
    pre_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    post_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    for split_idx in tl.static_range(NUM_SPLITS):
        partial = partial_row + split_idx * PARTIAL_STRIDE
        sum_squares += tl.load(partial)
        pre_mix += tl.load(partial + 1 + hc_idx, mask=hc_mask, other=0.0)
        post_mix += tl.load(
            partial + 1 + HC_POW2 + hc_idx,
            mask=hc_mask,
            other=0.0,
        )

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS

    if hidden_block_idx == 0:
        post = (
            MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
            + HC_EPS
        )
        tl.store(
            post_ptr + token_idx * HC_MULT + hc_idx,
            post,
            mask=hc_mask,
        )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel

    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


def _unsupported_reason(
    x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps
):
    tensors = (x, residual, post, weight, scale, base)
    if x.ndim != 2 or not 0 <= x.shape[0] <= 1024 or x.shape[1] != 6144:
        return "BF16 rows0..1024 with hidden6144 required"
    rows = x.shape[0]
    shapes = ((rows, 6144), (rows, 4, 6144), (rows, 4), (8, 24576), (2,), (8,))
    strides = ((6144, 1), (24576, 6144, 1), (4, 1), (24576, 1), (1,), (1,))
    dtypes = (
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.float32,
    )
    for tensor, shape, stride, dtype in zip(tensors, shapes, strides, dtypes):
        if (
            tuple(tensor.shape) != shape
            or tuple(tensor.stride()) != stride
            or tensor.dtype != dtype
            or not tensor.is_cuda
            or tensor.device != x.device
            or tensor.layout != torch.strided
            or tensor.is_conj()
            or tensor.is_neg()
        ):
            return "same-device canonical BF16/FP32 tensors required"
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return "inference-only direct API"
    if any(
        not isinstance(v, (float, int)) or not math.isfinite(v)
        for v in (magnitude, hc_eps, norm_eps)
    ):
        return "finite scalar parameters required"
    return None


def can_ihc_post_pre(
    x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps
):
    """Metadata-only capability check; no value reads, copy or dispatch cache."""
    return (
        _unsupported_reason(
            x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps
        )
        is None
    )


def ihc_post_pre(x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps):
    """Return (reduced, next_post_gates, full_BF16_residual), all fresh owners."""
    why = _unsupported_reason(
        x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps
    )
    if why is not None:
        raise NotImplementedError(why)
    rows = x.shape[0]
    materialized = torch.empty((rows, 4, 6144), dtype=x.dtype, device=x.device)
    reduced = torch.empty((rows, 6144), dtype=x.dtype, device=x.device)
    next_post = torch.empty((rows, 4), dtype=torch.float32, device=x.device)
    if rows == 0:
        return reduced, next_post, materialized
    partial = torch.empty((rows, 24, 9), dtype=torch.float32, device=x.device)
    _ihc_post_pre_stage1[(rows, 24)](
        x,
        residual,
        post,
        materialized,
        weight,
        partial,
        HIDDEN_SIZE=6144,
        K_TOTAL=24576,
        HC_MULT=4,
        HC_POW2=4,
        NUM_SPLITS=24,
        BLOCK_K=1024,
        PARTIAL_STRIDE=9,
        num_warps=8,
        enable_fp_fusion=False,
    )
    _ihc_pre_stage2[(rows, 6)](
        materialized,
        partial,
        scale,
        base,
        reduced,
        next_post,
        HIDDEN_SIZE=6144,
        K_TOTAL=24576,
        HC_MULT=4,
        HC_POW2=4,
        NUM_SPLITS=24,
        PARTIAL_STRIDE=9,
        BLOCK_D=1024,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return reduced, next_post, materialized

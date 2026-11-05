# SPDX-License-Identifier: Apache-2.0
"""IHC stage1 token-pair weight reuse with explicit metadata guards.

Public direct op rejects unsupported metadata. No Torch numerical fallback.
Per-token BK1024 reduction and native partial layout stay unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.ihc_post_pre import _ihc_pre_stage2
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)
__all__ = ["ihc_stage1_pair"]


@triton.jit
def _pair_input(
    x_ptr,
    residual_ptr,
    post_ptr,
    materialized_ptr,
    token,
    offsets,
    hc_idx,
    channel_idx,
    ROWS: tl.constexpr,
    FUSE_POST: tl.constexpr,
):
    live = token < ROWS
    if FUSE_POST:
        post = tl.load(post_ptr + token * 4 + hc_idx, mask=live, other=0.0)
        x = tl.load(x_ptr + token * 6144 + offsets % 6144, mask=live, other=0.0).to(
            tl.float32
        )
        residual = tl.load(
            residual_ptr + token * 24576 + offsets, mask=live, other=0.0
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, post, 0.0), axis=0)
        output = channel_gate * x + residual
        rounded = output.to(materialized_ptr.dtype.element_ty)
        tl.store(materialized_ptr + token * 24576 + offsets, rounded, mask=live)
        x = rounded.to(tl.float32)
    else:
        x = tl.load(x_ptr + token * 24576 + offsets, mask=live, other=0.0).to(
            tl.float32
        )
    return x


@libentry()
@triton.jit
def _ihc_stage1_pair(
    x_ptr,
    residual_ptr,
    post_ptr,
    materialized_ptr,
    weight_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    FUSE_POST: tl.constexpr,
):
    token0 = (tl.program_id(0) * 2).to(tl.int64)
    token1 = token0 + 1
    split_idx = tl.program_id(1)
    hc_idx = tl.arange(0, 4)
    offsets = split_idx * 1024 + tl.arange(0, 1024)
    channel_idx = split_idx * 1024 // 6144
    x0 = _pair_input(
        x_ptr,
        residual_ptr,
        post_ptr,
        materialized_ptr,
        token0,
        offsets,
        hc_idx,
        channel_idx,
        ROWS,
        FUSE_POST,
    )
    x1 = _pair_input(
        x_ptr,
        residual_ptr,
        post_ptr,
        materialized_ptr,
        token1,
        offsets,
        hc_idx,
        channel_idx,
        ROWS,
        FUSE_POST,
    )
    # Same rank-2 weight vectors and rank-1 reductions as native, not a new
    # GEMM/TF32 reduction. Each W load is reused by both independent tokens.
    weight_offsets = hc_idx[:, None] * 24576 + offsets[None, :]
    pre_weight = tl.load(weight_ptr + weight_offsets)
    sum_squares0 = tl.sum(x0 * x0, axis=0)
    sum_squares1 = tl.sum(x1 * x1, axis=0)
    pre_mix0 = tl.sum(pre_weight * x0[None, :], axis=1)
    pre_mix1 = tl.sum(pre_weight * x1[None, :], axis=1)
    post_weight = tl.load(weight_ptr + 4 * 24576 + weight_offsets)
    post_mix0 = tl.sum(post_weight * x0[None, :], axis=1)
    post_mix1 = tl.sum(post_weight * x1[None, :], axis=1)
    partial0 = partial_ptr + (token0 * 24 + split_idx) * 9
    partial1 = partial_ptr + (token1 * 24 + split_idx) * 9
    tl.store(partial0, sum_squares0)
    tl.store(partial0 + 1 + hc_idx, pre_mix0)
    tl.store(partial0 + 1 + 4 + hc_idx, post_mix0)
    tl.store(partial1, sum_squares1, mask=token1 < ROWS)
    tl.store(partial1 + 1 + hc_idx, pre_mix1, mask=token1 < ROWS)
    tl.store(partial1 + 1 + 4 + hc_idx, post_mix1, mask=token1 < ROWS)


def unsupported_reason(x, weight, *, residual=None, post=None):
    fused = residual is not None or post is not None
    if x.ndim not in (2, 3) or not 0 <= x.shape[0] <= 1024:
        return "rows0..1024 canonical IHC tensors required"
    rows = x.shape[0]
    expected = (
        ((rows, 6144), (6144, 1)) if fused else ((rows, 4, 6144), (24576, 6144, 1))
    )
    tensors = [
        (x, *expected, torch.bfloat16),
        (weight, (8, 24576), (24576, 1), torch.float32),
    ]
    if fused:
        if residual is None or post is None:
            return "both residual and post required"
        tensors.extend(
            [
                (residual, (rows, 4, 6144), (24576, 6144, 1), torch.bfloat16),
                (post, (rows, 4), (4, 1), torch.float32),
            ]
        )
    for value, shape, stride, dtype in tensors:
        if (
            tuple(value.shape) != shape
            or tuple(value.stride()) != stride
            or value.dtype != dtype
            or not value.is_cuda
            or value.device != x.device
            or value.layout != torch.strided
            or value.is_conj()
            or value.is_neg()
        ):
            return "canonical same-device BF16/F32 metadata required"
    if torch.is_grad_enabled() and any(v.requires_grad for v, *_ in tensors):
        return "inference-only direct op"
    return None


def ihc_stage1_pair(x, weight, *, residual=None, post=None):
    """Return FP32[M,24,9] partial and optional fresh BF16[M,4,6144].

    This primitive has no implicit serving selection. The framework route
    must remain default OFF and scope selection to qualified prefill metadata.
    """
    why = unsupported_reason(x, weight, residual=residual, post=post)
    if why is not None:
        raise NotImplementedError(why)
    rows = x.shape[0]
    partial = torch.empty((rows, 24, 9), device=x.device, dtype=torch.float32)
    fused = residual is not None
    materialized = (
        torch.empty((rows, 4, 6144), device=x.device, dtype=torch.bfloat16)
        if fused
        else None
    )
    if rows:
        _ihc_stage1_pair[(triton.cdiv(rows, 2), 24)](
            x,
            residual,
            post,
            materialized,
            weight,
            partial,
            ROWS=rows,
            FUSE_POST=fused,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return partial, materialized


# Fixed paired-token schedule: BK1024, two tokens/program, eight warps.
# No autotuning: preserving the qualified reduction tree and launch is an
# explicit migration exemption; retuning requires separate qualification.

_PAIR_ROWS = frozenset((208, 256, 1024))
__all__ = ["can_ihc_pair", "ihc_pre_pair", "ihc_post_pre_pair"]


def _full_unsupported_reason(
    x, weight, scale, base, magnitude, hc_eps, norm_eps, *, residual=None, post=None
):
    why = unsupported_reason(x, weight, residual=residual, post=post)
    if why is not None:
        return why
    if x.shape[0] not in _PAIR_ROWS:
        return "pair serving candidate only for physical M208/256/1024"
    for value, size in ((scale, 2), (base, 8)):
        if (
            tuple(value.shape) != (size,)
            or tuple(value.stride()) != (1,)
            or value.dtype != torch.float32
            or not value.is_cuda
            or value.device != x.device
            or value.layout != torch.strided
            or value.is_conj()
            or value.is_neg()
        ):
            return "canonical same-device F32 HC scale/base required"
    if torch.is_grad_enabled() and (scale.requires_grad or base.requires_grad):
        return "inference-only direct op"
    if any(type(v) not in (float, int) for v in (magnitude, hc_eps, norm_eps)):
        return "real scalar parameters required"
    if (magnitude, hc_eps, norm_eps) != (2.0, 1e-6, 1e-5):
        return "only the measured scalar contract is admitted"
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        return "pair candidate qualified on SM90 only"
    return None


def can_ihc_pair(
    x, weight, scale, base, magnitude, hc_eps, norm_eps, *, residual=None, post=None
):
    """Metadata-only route: no value read, copy, normalization or fallback."""
    logger.debug("GEMS_NVIDIA CAN_IHC_PAIR")
    return (
        _full_unsupported_reason(
            x,
            weight,
            scale,
            base,
            magnitude,
            hc_eps,
            norm_eps,
            residual=residual,
            post=post,
        )
        is None
    )


def _finish_pair(value, partial, scale, base, magnitude, hc_eps, norm_eps):
    rows = value.shape[0]
    reduced = torch.empty((rows, 6144), device=value.device, dtype=torch.bfloat16)
    gates = torch.empty((rows, 4), device=value.device, dtype=torch.float32)
    _ihc_pre_stage2[(rows, 6)](
        value,
        partial,
        scale,
        base,
        reduced,
        gates,
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
    return reduced, gates


def ihc_pre_pair(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    """Return fresh BF16 reduced activation and F32 post gates."""
    logger.debug("GEMS_NVIDIA IHC_PRE_PAIR")
    why = _full_unsupported_reason(x, weight, scale, base, magnitude, hc_eps, norm_eps)
    if why is not None:
        raise NotImplementedError(why)
    partial, unused = ihc_stage1_pair(x, weight)
    return _finish_pair(x, partial, scale, base, magnitude, hc_eps, norm_eps)


def ihc_post_pre_pair(
    x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps
):
    """Retain BF16 post materialization and return all fresh output owners."""
    logger.debug("GEMS_NVIDIA IHC_POST_PRE_PAIR")
    why = _full_unsupported_reason(
        x,
        weight,
        scale,
        base,
        magnitude,
        hc_eps,
        norm_eps,
        residual=residual,
        post=post,
    )
    if why is not None:
        raise NotImplementedError(why)
    partial, materialized = ihc_stage1_pair(x, weight, residual=residual, post=post)
    reduced, gates = _finish_pair(
        materialized, partial, scale, base, magnitude, hc_eps, norm_eps
    )
    return reduced, gates, materialized

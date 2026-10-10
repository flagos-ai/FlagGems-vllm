# SPDX-License-Identifier: Apache-2.0
"""MHC fused-op ABI, preserving BF16 materialization before RMS normalization."""
import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.data_movement import contiguous_copy
from flaggems_vllm.ops.mhc import mhc_post, mhc_pre
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(configs=runtime.get_tuned_config("mhc_rms_norm"), key=["D"])
@triton.jit
def _norm_kernel(X, W, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    value = tl.load(X + row * D + d, d < D, other=0).to(tl.float32)
    weight = tl.load(W + d, d < D, other=0).to(tl.float32)
    result = value * tl.rsqrt(tl.sum(value * value, 0) / D + EPS) * weight
    tl.store(Y + row * D + d, result, d < D)


def mhc_rms_norm(layer_input, norm_weight, norm_eps):
    if norm_weight is None:
        return layer_input
    dim = layer_input.shape[-1]
    if norm_weight.shape != (dim,) or norm_weight.device != layer_input.device:
        raise ValueError("MHC norm weight must match hidden dimension and device")
    layer_input, norm_weight = map(contiguous_copy, (layer_input, norm_weight))
    out = torch.empty(
        layer_input.shape, dtype=layer_input.dtype, device=layer_input.device
    )
    if out.numel():
        # BLOCK is the semantic reduction extent, fixed by D. Only warps tune.
        _norm_kernel[(out.numel() // dim,)](
            layer_input,
            norm_weight,
            out,
            dim,
            norm_eps,
            BLOCK=triton.next_power_of_2(dim),
        )
    return out


def mhc_pre_with_norm(
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
    norm_weight=None,
    norm_eps=0.0,
):
    if norm_weight is not None and (
        norm_weight.shape != (residual.shape[-1],)
        or norm_weight.device != residual.device
    ):
        raise ValueError("MHC norm weight shape/device mismatch")
    post, comb, value = mhc_pre(
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
    return post, comb, mhc_rms_norm(value, norm_weight, norm_eps)


def mhc_fused_post_pre_with_norm(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=0.0,
):
    # Preflight the optional norm before a stateful post operation starts.
    if norm_weight is not None and (
        norm_weight.shape != (residual.shape[-1],)
        or norm_weight.device != residual.device
    ):
        raise ValueError("MHC norm weight shape/device mismatch")
    residual_cur = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    return (
        residual_cur,
        *mhc_pre_with_norm(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        ),
    )

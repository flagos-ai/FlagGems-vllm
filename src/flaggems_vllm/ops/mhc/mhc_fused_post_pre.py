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

"""Optimized mHC post-to-pre path for the HC=4/H=4096 decode workload."""

from __future__ import annotations

import math

import torch

from flaggems_vllm.ops.mhc.mhc_post import mhc_post
from flaggems_vllm.ops.mhc.mhc_pre_with_norm import mhc_pre_with_norm
from flaggems_vllm.ops.mhc.mhc_prenorm import mhc_prenorm_gemm

_HC_MULT = 4
_HIDDEN_SIZE = 4096
_MIX_COUNT = 24
_SUPPORTED_TOKEN_COUNTS = (64, 96, 128)


def _validate_inputs(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor | None,
) -> tuple[tuple[int, ...], int]:
    if residual.ndim < 3 or residual.shape[-2:] != (_HC_MULT, _HIDDEN_SIZE):
        raise NotImplementedError("optimized mHC requires residual[..., 4, 4096]")

    outer_shape = tuple(residual.shape[:-2])
    num_tokens = math.prod(outer_shape)
    if num_tokens == 0:
        raise NotImplementedError("optimized mHC does not support empty inputs")
    if num_tokens not in _SUPPORTED_TOKEN_COUNTS:
        raise NotImplementedError(
            f"optimized mHC supports token counts {_SUPPORTED_TOKEN_COUNTS}"
        )

    expected_shapes = {
        "x": (*outer_shape, _HIDDEN_SIZE),
        "comb_res_mix": (*outer_shape, _HC_MULT, _HC_MULT),
        "fn": (_MIX_COUNT, _HC_MULT * _HIDDEN_SIZE),
        "hc_scale": (3,),
        "hc_base": (_MIX_COUNT,),
    }
    tensors_by_name = {
        "x": x,
        "comb_res_mix": comb_res_mix,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
    }
    for name, expected_shape in expected_shapes.items():
        if tensors_by_name[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    valid_post_shapes = (
        (*outer_shape, _HC_MULT),
        (*outer_shape, _HC_MULT, 1),
    )
    if post_layer_mix.shape not in valid_post_shapes:
        raise ValueError(
            f"post_layer_mix must have shape {valid_post_shapes[0]} "
            f"or {valid_post_shapes[1]}"
        )
    if norm_weight is None:
        raise NotImplementedError("optimized mHC currently requires fused RMSNorm")
    if norm_weight.shape != (_HIDDEN_SIZE,):
        raise ValueError(f"norm_weight must have shape ({_HIDDEN_SIZE},)")

    bf16_tensors = {
        "x": x,
        "residual": residual,
        "norm_weight": norm_weight,
    }
    for name, tensor in bf16_tensors.items():
        if tensor.dtype != torch.bfloat16:
            raise NotImplementedError(f"{name} must use bfloat16")

    fp32_tensors = {
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
    }
    for name, tensor in fp32_tensors.items():
        if tensor.dtype != torch.float32:
            raise NotImplementedError(f"{name} must use float32")

    tensors = (
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
    )
    if not residual.is_cuda:
        raise NotImplementedError("optimized mHC requires CUDA tensors")
    if not all(tensor.device == residual.device for tensor in tensors):
        raise ValueError("all mHC tensors must be on the same device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise NotImplementedError("optimized mHC requires contiguous tensors")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("optimized mHC is an inference-only path")
    if residual.device.index != torch.cuda.current_device():
        raise NotImplementedError(
            "optimized mHC requires its tensor device to be the current CUDA device"
        )
    if torch.cuda.get_device_capability(residual.device)[0] < 9:
        raise NotImplementedError("optimized mHC requires SM90 or newer")
    return outer_shape, num_tokens


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Triton-only post mapping, prenorm, mixing, and RMSNorm chain.

    ``n_splits`` and ``tile_n`` are retained for call-site compatibility with
    the vLLM entry point. The optimized path selects a validated Triton split
    specialization internally. Call once with each immutable ``fn`` tensor
    before CUDA Graph capture so its Triton-packed weight cache is ready.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an integer")
    if n_splits not in (1, 2, 4, 8):
        raise ValueError("n_splits must be one of 1, 2, 4, or 8")
    if isinstance(tile_n, bool) or not isinstance(tile_n, int) or tile_n < 1:
        raise ValueError("tile_n must be a positive integer")
    if (
        isinstance(sinkhorn_repeat, bool)
        or not isinstance(sinkhorn_repeat, int)
        or sinkhorn_repeat < 1
    ):
        raise ValueError("sinkhorn_repeat must be at least one")
    outer_shape, num_tokens = _validate_inputs(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
    )
    assert norm_weight is not None

    residual_flat = residual.view(num_tokens, _HC_MULT, _HIDDEN_SIZE)
    x_flat = x.view(num_tokens, _HIDDEN_SIZE)
    comb_mix_flat = comb_res_mix.view(num_tokens, _HC_MULT, _HC_MULT)
    if post_layer_mix.shape[-1] == 1:
        post_mix_flat = post_layer_mix.view(num_tokens, _HC_MULT, 1)
    else:
        post_mix_flat = post_layer_mix.view(num_tokens, _HC_MULT)

    residual_cur = mhc_post(x_flat, residual_flat, post_mix_flat, comb_mix_flat)
    gemm_out_mul, gemm_out_sqrsum = mhc_prenorm_gemm(
        residual_cur.view(num_tokens, _HC_MULT * _HIDDEN_SIZE), fn
    )
    post_mix_cur = torch.empty(
        (num_tokens, _HC_MULT), dtype=torch.float32, device=residual.device
    )
    comb_mix_cur = torch.empty(
        (num_tokens, _HC_MULT * _HC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        (num_tokens, _HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=residual.device,
    )

    mhc_pre_with_norm(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
    )

    return (
        residual_cur.view(*outer_shape, _HC_MULT, _HIDDEN_SIZE),
        post_mix_cur.view(*outer_shape, _HC_MULT, 1),
        comb_mix_cur.view(*outer_shape, _HC_MULT, _HC_MULT),
        layer_input_cur.view(*outer_shape, _HIDDEN_SIZE),
    )


__all__ = ["mhc_fused_post_pre"]

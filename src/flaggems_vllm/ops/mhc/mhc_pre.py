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

"""Triton-only mHC pre operator for the active decode workload."""

from __future__ import annotations

import math

import torch

from flaggems_vllm.ops.mhc.mhc_pre_with_norm import mhc_pre_without_norm
from flaggems_vllm.ops.mhc.mhc_prenorm import mhc_prenorm_gemm

_HC_MULT = 4
_HIDDEN_SIZE = 4096
_MIX_COUNT = 24
_SUPPORTED_TOKEN_COUNTS = (64, 96, 128)


def _validate_inputs(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> tuple[tuple[int, ...], int]:
    if residual.ndim < 3 or residual.shape[-2:] != (_HC_MULT, _HIDDEN_SIZE):
        raise NotImplementedError("mHC pre requires residual[..., 4, 4096]")
    outer_shape = tuple(residual.shape[:-2])
    num_tokens = math.prod(outer_shape)
    if num_tokens not in _SUPPORTED_TOKEN_COUNTS:
        raise NotImplementedError(
            f"mHC pre supports token counts {_SUPPORTED_TOKEN_COUNTS}"
        )
    if fn.shape != (_MIX_COUNT, _HC_MULT * _HIDDEN_SIZE):
        raise ValueError(
            f"fn must have shape ({_MIX_COUNT}, {_HC_MULT * _HIDDEN_SIZE})"
        )
    if hc_scale.shape != (3,):
        raise ValueError("hc_scale must have shape (3,)")
    if hc_base.shape != (_MIX_COUNT,):
        raise ValueError(f"hc_base must have shape ({_MIX_COUNT},)")
    if residual.dtype != torch.bfloat16:
        raise NotImplementedError("mHC pre residual must use bfloat16")
    if any(tensor.dtype != torch.float32 for tensor in (fn, hc_scale, hc_base)):
        raise NotImplementedError("mHC pre parameters must use float32")
    tensors = (residual, fn, hc_scale, hc_base)
    if not residual.is_cuda:
        raise NotImplementedError("mHC pre requires CUDA tensors")
    if not all(tensor.device == residual.device for tensor in tensors):
        raise ValueError("all mHC pre tensors must be on the same device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise NotImplementedError("mHC pre requires contiguous tensors")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("mHC pre is an inference-only path")
    if residual.device.index != torch.cuda.current_device():
        raise NotImplementedError(
            "mHC pre requires its tensor device to be the current CUDA device"
        )
    return outer_shape, num_tokens


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute prenorm mixes, Sinkhorn, and the weighted layer input.

    All GPU computation is dispatched to Triton kernels. ``n_splits`` is
    retained for API compatibility; the active path selects a measured split
    specialization from the token count.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an integer")
    if n_splits not in (1, 2, 4, 8):
        raise ValueError("n_splits must be one of 1, 2, 4, or 8")
    if (
        isinstance(sinkhorn_repeat, bool)
        or not isinstance(sinkhorn_repeat, int)
        or sinkhorn_repeat < 1
    ):
        raise ValueError("sinkhorn_repeat must be at least one")
    outer_shape, num_tokens = _validate_inputs(residual, fn, hc_scale, hc_base)
    residual_flat = residual.view(num_tokens, _HC_MULT, _HIDDEN_SIZE)
    partial, partial_sqrsum = mhc_prenorm_gemm(
        residual_flat.view(num_tokens, _HC_MULT * _HIDDEN_SIZE), fn
    )
    post_mix = torch.empty(
        (num_tokens, _HC_MULT), dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        (num_tokens, _HC_MULT * _HC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        (num_tokens, _HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=residual.device,
    )
    mhc_pre_without_norm(
        partial,
        partial_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return (
        post_mix.view(*outer_shape, _HC_MULT, 1),
        comb_mix.view(*outer_shape, _HC_MULT, _HC_MULT),
        layer_input.view(*outer_shape, _HIDDEN_SIZE),
    )


__all__ = ["mhc_pre"]

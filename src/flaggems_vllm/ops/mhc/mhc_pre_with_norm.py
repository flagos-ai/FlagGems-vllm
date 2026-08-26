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

"""Triton-only mHC pre epilogues for the HC=4/H=4096 decode path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_HIDDEN_SIZE = 4096
_HC_MULT = 4
_PADDED_MIX_COUNT = 32
_SEMANTIC_MIX_COUNT = 24
_SPLIT_COUNTS = {64: 64, 96: 32, 128: 32}
_NUM_WARPS = {64: 4, 96: 2, 128: 4}
_MAX_NUM_REGS = {64: 80, 96: None, 128: 80}


@triton.jit
def _sinkhorn_4x4(values, eps: tl.constexpr, repeat: tl.constexpr):
    matrix = tl.reshape(values, (4, 4))
    row_max = tl.max(matrix, axis=1)
    matrix = tl.exp(matrix - tl.reshape(row_max, (4, 1)))
    row_sum = tl.sum(matrix, axis=1)
    matrix = (
        tl.extra.cuda.libdevice.fast_dividef(matrix, tl.reshape(row_sum, (4, 1))) + eps
    )
    column_sum = tl.sum(matrix, axis=0)
    matrix = tl.extra.cuda.libdevice.fast_dividef(
        matrix, tl.reshape(column_sum, (1, 4)) + eps
    )
    for _ in tl.range(0, repeat - 1, loop_unroll_factor=1):
        row_sum = tl.sum(matrix, axis=1)
        matrix = tl.extra.cuda.libdevice.fast_dividef(
            matrix, tl.reshape(row_sum, (4, 1)) + eps
        )
        column_sum = tl.sum(matrix, axis=0)
        matrix = tl.extra.cuda.libdevice.fast_dividef(
            matrix, tl.reshape(column_sum, (1, 4)) + eps
        )
    return tl.reshape(matrix, (16,))


@triton.jit
def _mhc_pre_epilogue_kernel(
    partial_ptr,
    partial_sqrsum_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    norm_weight_ptr,
    M: tl.constexpr,
    SPLITS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    HC_PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINKHORN_REPEAT: tl.constexpr,
    NORM_EPS: tl.constexpr,
    FUSE_NORM: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    token = tl.program_id(0)
    role = tl.program_id(1)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()

    mix_offsets = tl.arange(0, 32)
    mixes = tl.zeros((32,), dtype=tl.float32)
    residual_sqrsum = 0.0
    # One contiguous 32-wide transaction per split is faster than separately
    # reducing pre/post/comb, and preserves the split-order FP32 accumulation.
    for split in tl.static_range(SPLITS):
        mixes += tl.load(partial_ptr + (split * M + token) * 32 + mix_offsets)
        residual_sqrsum += tl.load(partial_sqrsum_ptr + split * M + token)

    residual_inv_rms = tl.rsqrt(residual_sqrsum * (1.0 / 16384.0) + RMS_EPS)
    # Two independent CTA roles share one launch. Both repeat the small split
    # reduction, then the scheduler can overlap the long-latency 4x4 Sinkhorn
    # role with the bandwidth-heavy weighted-sum/RMSNorm role. The roles write
    # disjoint outputs, so no inter-CTA synchronization or scratch is needed.
    if role == 0:
        offsets4 = tl.arange(0, 4)
        offsets16 = tl.arange(0, 16)
        post_mix = tl.gather(mixes, 4 + offsets4, 0)
        comb_mix = tl.gather(mixes, 8 + offsets16, 0)
        post_mix = (
            tl.sigmoid(
                post_mix * residual_inv_rms * tl.load(hc_scale_ptr + 1)
                + tl.load(hc_base_ptr + 4 + offsets4)
            )
            * POST_MULT
        )
        comb_mix = comb_mix * residual_inv_rms * tl.load(hc_scale_ptr + 2) + tl.load(
            hc_base_ptr + 8 + offsets16
        )
        comb_mix = _sinkhorn_4x4(comb_mix, SINKHORN_EPS, SINKHORN_REPEAT)
        tl.store(post_mix_ptr + token * 4 + offsets4, post_mix)
        tl.store(comb_mix_ptr + token * 16 + offsets16, comb_mix)
    else:
        offsets4 = tl.arange(0, 4)
        pre_mix = tl.gather(mixes, offsets4, 0)
        pre_mix = (
            tl.sigmoid(
                pre_mix * residual_inv_rms * tl.load(hc_scale_ptr)
                + tl.load(hc_base_ptr + offsets4)
            )
            + HC_PRE_EPS
        )
        pre0 = tl.sum(tl.where(offsets4 == 0, pre_mix, 0.0), axis=0)
        pre1 = tl.sum(tl.where(offsets4 == 1, pre_mix, 0.0), axis=0)
        pre2 = tl.sum(tl.where(offsets4 == 2, pre_mix, 0.0), axis=0)
        pre3 = tl.sum(tl.where(offsets4 == 3, pre_mix, 0.0), axis=0)
        hidden_offsets = tl.arange(0, 4096)
        residual_base = token * 16384
        weighted = (
            tl.load(residual_ptr + residual_base + hidden_offsets).to(tl.float32) * pre0
        )
        weighted += (
            tl.load(residual_ptr + residual_base + 4096 + hidden_offsets).to(tl.float32)
            * pre1
        )
        weighted += (
            tl.load(residual_ptr + residual_base + 8192 + hidden_offsets).to(tl.float32)
            * pre2
        )
        weighted += (
            tl.load(residual_ptr + residual_base + 12288 + hidden_offsets).to(
                tl.float32
            )
            * pre3
        )
        # RMS statistics use the FP32 weighted sum, while RMSNorm consumes the
        # BF16-rounded weighted sum. This matches the public operator semantics.
        rounded_weighted = weighted.to(tl.bfloat16)
        if FUSE_NORM:
            layer_sqrsum = tl.sum(weighted * weighted, axis=0)
            layer_inv_rms = tl.rsqrt(layer_sqrsum * (1.0 / 4096.0) + NORM_EPS)
            norm_weight = tl.load(norm_weight_ptr + hidden_offsets)
            rounded_weighted = rounded_weighted * layer_inv_rms * norm_weight
        tl.store(
            layer_input_ptr + token * 4096 + hidden_offsets,
            rounded_weighted,
        )

    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _validate_common(
    partial: torch.Tensor,
    partial_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
) -> tuple[int, int]:
    if residual.ndim != 3 or residual.shape[1:] != (_HC_MULT, _HIDDEN_SIZE):
        raise NotImplementedError("mHC pre requires residual[M, 4, 4096]")
    num_tokens = residual.shape[0]
    expected_splits = _SPLIT_COUNTS.get(num_tokens)
    if expected_splits is None:
        raise NotImplementedError(
            f"mHC pre supports token counts {tuple(_SPLIT_COUNTS)}"
        )
    expected_shapes = {
        "partial": (expected_splits, num_tokens, _PADDED_MIX_COUNT),
        "partial_sqrsum": (expected_splits, num_tokens),
        "hc_scale": (3,),
        "hc_base": (_SEMANTIC_MIX_COUNT,),
        "post_mix": (num_tokens, _HC_MULT),
        "comb_mix": (num_tokens, _HC_MULT * _HC_MULT),
        "layer_input": (num_tokens, _HIDDEN_SIZE),
    }
    tensors = {
        "partial": partial,
        "partial_sqrsum": partial_sqrsum,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "post_mix": post_mix,
        "comb_mix": comb_mix,
        "layer_input": layer_input,
    }
    for name, expected_shape in expected_shapes.items():
        if tensors[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    fp32_tensors = (
        partial,
        partial_sqrsum,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
    )
    if not all(tensor.dtype == torch.float32 for tensor in fp32_tensors):
        raise NotImplementedError("mHC pre mix tensors must use float32")
    if residual.dtype != torch.bfloat16 or layer_input.dtype != torch.bfloat16:
        raise NotImplementedError("mHC residual and layer input must use bfloat16")
    all_tensors = (*fp32_tensors, residual, layer_input)
    if not residual.is_cuda:
        raise NotImplementedError("mHC pre requires CUDA tensors")
    if not all(tensor.device == residual.device for tensor in all_tensors):
        raise ValueError("all mHC pre tensors must be on the same device")
    if not all(tensor.is_contiguous() for tensor in all_tensors):
        raise NotImplementedError("mHC pre requires contiguous tensors")
    if any(tensor.requires_grad for tensor in all_tensors):
        raise NotImplementedError("mHC pre is an inference-only path")
    if residual.device.index != torch.cuda.current_device():
        raise NotImplementedError(
            "mHC pre requires its input device to be the current CUDA device"
        )
    if torch.cuda.get_device_capability(residual.device)[0] < 9:
        raise NotImplementedError("the optimized mHC pre path requires SM90+")
    return num_tokens, expected_splits


def _launch(
    partial: torch.Tensor,
    partial_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    fuse_norm: bool,
) -> None:
    num_tokens, split_count = _validate_common(
        partial,
        partial_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    )
    if (
        isinstance(sinkhorn_repeat, bool)
        or not isinstance(sinkhorn_repeat, int)
        or sinkhorn_repeat < 1
    ):
        raise ValueError("sinkhorn_repeat must be a positive integer")
    if fuse_norm:
        if norm_weight.shape != (_HIDDEN_SIZE,):
            raise ValueError(f"norm_weight must have shape ({_HIDDEN_SIZE},)")
        if norm_weight.dtype != torch.bfloat16:
            raise NotImplementedError("norm_weight must use bfloat16")
        if norm_weight.device != residual.device or not norm_weight.is_contiguous():
            raise ValueError("norm_weight must be contiguous and on the input device")
        if norm_weight.requires_grad:
            raise NotImplementedError("mHC pre is an inference-only path")
    launch_pdl = torch.version.hip is None
    launch_kwargs = {
        "num_warps": _NUM_WARPS[num_tokens],
        "num_stages": 1,
    }
    max_num_regs = _MAX_NUM_REGS[num_tokens]
    if max_num_regs is not None:
        launch_kwargs["maxnreg"] = max_num_regs
    _mhc_pre_epilogue_kernel[(num_tokens, 2)](
        partial,
        partial_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        M=num_tokens,
        SPLITS=split_count,
        RMS_EPS=rms_eps,
        HC_PRE_EPS=hc_pre_eps,
        SINKHORN_EPS=hc_sinkhorn_eps,
        POST_MULT=hc_post_mult_value,
        SINKHORN_REPEAT=sinkhorn_repeat,
        NORM_EPS=norm_eps,
        FUSE_NORM=fuse_norm,
        LAUNCH_PDL=launch_pdl,
        launch_pdl=launch_pdl,
        **launch_kwargs,
    )


def mhc_pre_with_norm(
    partial: torch.Tensor,
    partial_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> None:
    """Run split reduction, mixing, Sinkhorn, and fused RMSNorm in Triton."""
    _launch(
        partial,
        partial_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        True,
    )


def mhc_pre_without_norm(
    partial: torch.Tensor,
    partial_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> None:
    """Run split reduction, mixing, and Sinkhorn without RMSNorm in Triton."""
    _launch(
        partial,
        partial_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        0.0,
        False,
    )


__all__ = ["mhc_pre_with_norm", "mhc_pre_without_norm"]

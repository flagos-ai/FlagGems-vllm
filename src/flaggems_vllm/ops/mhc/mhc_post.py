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

"""
Triton implementation of mHC Post operator (optimized v3).

Computes:
    out[n, i, h] = post_layer_mix[n, i] * x[n, h]
                 + sum_j(comb_res_mix[n, j, i] * residual[n, j, h])

Key optimizations (v3):
- 2D grid = (N, cdiv(H, BLOCK_H)): high program count for latency hiding.
- M/H-aware library tuning over BLOCK_H and num_warps using CUDA Graph timing.
- Contiguous layout: stride math removed, enabling LDG.128.
- All 4 accumulators computed then stored (better ILP).
- BLOCK_H chosen to evenly divide H when possible (256 divides all targets).
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_post_hc4"),
    key=["N", "H"],
    use_cuda_graph=True,
)
@triton.jit
def mhc_post_kernel_hc_mult_4(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 — a[n, j, i]
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4),   float32
    d_ptr,  # x            : (N, H),    bfloat16
    out_ptr,  # output       : (N, 4, H), bfloat16
    N,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    """
    Grid: (N, cdiv(H, BLOCK_H)).
    Each program handles one token × one h-tile × all 4 hc streams.
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    # ── pointer bases (contiguous layout) ──
    a_base = pid_n * 16  # (N, 4, 4) → stride_n = 16
    c_base = pid_n * 4  # (N, 4)    → stride_n = 4
    b_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H
    d_base = pid_n * H  # (N, H)    → stride_n = H
    out_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H

    # ── load 20 scalars (L1 cached across h-tiles) ──
    c0 = tl.load(c_ptr + c_base + 0).to(tl.float32)
    c1 = tl.load(c_ptr + c_base + 1).to(tl.float32)
    c2 = tl.load(c_ptr + c_base + 2).to(tl.float32)
    c3 = tl.load(c_ptr + c_base + 3).to(tl.float32)

    a00 = tl.load(a_ptr + a_base + 0).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 1).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 2).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 3).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 4).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 5).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 6).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 7).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 8).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 9).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 10).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 11).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 12).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 13).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 14).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 15).to(tl.float32)

    # ── load vectors (bf16 → f32) ──
    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + b_base + 0 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + b_base + 1 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b2 = tl.load(b_ptr + b_base + 2 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b3 = tl.load(b_ptr + b_base + 3 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)

    # ── compute all 4 output streams ──
    acc0 = c0 * d_vals + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    acc1 = c1 * d_vals + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    acc2 = c2 * d_vals + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    acc3 = c3 * d_vals + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

    # ── store all 4 outputs ──
    tl.store(out_ptr + out_base + 0 * H + h_off, acc0.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_off, acc1.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_off, acc2.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_off, acc3.to(tl.bfloat16), mask=h_mask)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_post_generic"),
    key=["N", "H", "HC"],
    use_cuda_graph=True,
)
@triton.jit
def mhc_post_kernel_generic(
    a_ptr,  # comb_res_mix : (N, HC, HC), float32
    b_ptr,  # residual     : (N, HC, H), bfloat16
    c_ptr,  # post_layer_mix: (N, HC), float32
    d_ptr,  # x            : (N, H), bfloat16
    out_ptr,  # output      : (N, HC, H), bfloat16
    N,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    """Generic mHC post kernel for arbitrary HC.

    Grid: (N, HC, cdiv(H, BLOCK_H)).
    Each program handles one token × one output-stream(i) × one h-tile.
    """
    pid_n = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    a_base = pid_n * HC * HC
    b_base = pid_n * HC * H
    c_base = pid_n * HC
    d_base = pid_n * H
    out_base = pid_n * HC * H + pid_i * H

    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    c_i = tl.load(c_ptr + c_base + pid_i).to(tl.float32)

    acc = c_i * d_vals
    for j in tl.static_range(0, HC):
        a_ji = tl.load(a_ptr + a_base + j * HC + pid_i).to(tl.float32)
        b_j = tl.load(b_ptr + b_base + j * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        acc += a_ji * b_j

    tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16), mask=h_mask)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """
    mHC post-processing operator.

    Args:
        x: (N, H), bfloat16 — layer output
        residual: (N, hc_mult, H), bfloat16 — multi-head residual
        post_layer_mix: (N, hc_mult, 1), float32 — per-stream scale for x
        comb_res_mix: (N, hc_mult, hc_mult), float32 — combination matrix

    Returns:
        out: (N, hc_mult, H), bfloat16
    """
    logger.debug(
        "GEMS MHC_POST FORWARD, x=%s, residual=%s, post_layer_mix=%s, comb_res_mix=%s",
        x.shape,
        residual.shape,
        post_layer_mix.shape,
        comb_res_mix.shape,
    )

    if residual.ndim != 3:
        raise ValueError("residual must have shape [N, HC, H]")
    N, hc, H = residual.shape
    if N == 0 or hc == 0 or H == 0:
        raise NotImplementedError("mHC post does not support empty inputs")
    if x.shape != (N, H):
        raise ValueError(f"x must have shape ({N}, {H})")
    if post_layer_mix.shape not in ((N, hc, 1), (N, hc)):
        raise ValueError(
            f"post_layer_mix must have shape ({N}, {hc}, 1) or ({N}, {hc})"
        )
    if comb_res_mix.shape != (N, hc, hc):
        raise ValueError(f"comb_res_mix must have shape ({N}, {hc}, {hc})")
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise NotImplementedError("x and residual must use bfloat16")
    if post_layer_mix.dtype != torch.float32 or comb_res_mix.dtype != torch.float32:
        raise NotImplementedError("mHC post mix tensors must use float32")
    inputs = (x, residual, post_layer_mix, comb_res_mix)
    if not residual.is_cuda:
        raise NotImplementedError("mHC post requires CUDA tensors")
    if not all(tensor.device == residual.device for tensor in inputs):
        raise ValueError("all mHC post tensors must be on the same device")
    if not all(tensor.is_contiguous() for tensor in inputs):
        raise NotImplementedError("mHC post requires contiguous tensors")
    if any(tensor.requires_grad for tensor in inputs):
        raise NotImplementedError("mHC post is an inference-only path")
    if residual.device.index != torch.cuda.current_device():
        raise NotImplementedError(
            "mHC post requires its tensor device to be the current CUDA device"
        )

    out = torch.empty_like(residual)

    c = post_layer_mix.squeeze(-1)  # (N, hc), no-copy view
    a = comb_res_mix
    b = residual
    d = x
    launch_pdl = (
        torch.version.hip is None
        and torch.cuda.get_device_capability(residual.device)[0] >= 9
    )

    if hc == 4:

        def grid_specialized(META):
            return (N, triton.cdiv(H, META["BLOCK_H"]))

        mhc_post_kernel_hc_mult_4[grid_specialized](
            a,
            b,
            c,
            d,
            out,
            N,
            H=H,
            LAUNCH_PDL=launch_pdl,
            launch_pdl=launch_pdl,
        )
    else:

        def grid_generic(META):
            return (N, hc, triton.cdiv(H, META["BLOCK_H"]))

        mhc_post_kernel_generic[grid_generic](
            a,
            b,
            c,
            d,
            out,
            N,
            H=H,
            HC=hc,
            LAUNCH_PDL=launch_pdl,
            launch_pdl=launch_pdl,
        )
    return out


__all__ = ["mhc_post"]

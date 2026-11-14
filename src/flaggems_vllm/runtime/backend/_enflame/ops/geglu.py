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

import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import tl_extra_shim

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh

logger = logging.getLogger(__name__)

# Flat 1D row-chunk kernel for the GCU backend (L600/Libra).
#
# Why not the classic 2D [BLOCK_M, BLOCK_H] tile: on this toolchain, rank-2
# unmasked affine loads/stores fail to compile for any tile size, and rank-2
# masked tiles compiled from a single base pointer with pure-arith math hit the
# same `memref_ext.slice_(pad_)start` property-conversion bug (see
# compiler_issues/vllm_silu_flat_slice_start.md). The flat 1D form (scalar
# `tile // h_tiles` row decomposition + contiguous per-row-chunk access) is
# immune and measured 10-25x faster than the shipped 2D masked kernel once the
# math is also de-transcendentalized (libdevice `pow(x, 2)` lowers to an
# exp2*log2 chain: ~500-620us of a 930us kernel at (4096, 4096) fp32).
#
# Math: 0.5*x*(1+tanh(0.79788456*x*(1+0.044715*x^2))) == x*sigmoid(2z), with
# 2z = 1.59576912*x*(1+0.044715*x^2) -- identical in exact arithmetic, ~1 ulp
# fp32 difference, computed in fp32 like the reference kernel.
#
# Config (measured on L600, kernel-mode do_bench, core bench shapes):
#   - grid capped at 48 (SPU count; grid-stride covers any tile count, so no
#     hardware grid-dimension limit is ever reachable);
#   - BH <= 2048 keeps every 1D slice at <=8KB fp32 / 512 elems per warp at
#     num_warps=4, below the slice_(pad_)start per-slice bug thresholds;
#   - UNROLL batches 4 (8 for 2-byte dtypes with BH <= 512) consecutive tiles
#     per loop iteration: deeper in-flight load pipelining hides the exp/div
#     latency (u4@BH=2048 fp16 31.2us vs u1 58.1us; u8@BH=512 fp16 3.71ms vs
#     u4 3.89ms). 2-byte dtypes tolerate the deeper unroll because their tiles
#     carry half the bytes.
#   - EVEN_H (H % BH == 0) drops the tail mask entirely; odd H falls back to a
#     masked single-tile body (correctness path, still grid-stride flat).
GCU_GRID_CAP = 48
GCU_MAX_BLOCK_H = 2048


def _next_pow2(value):
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


@triton.jit
def geglu_kernel(
    input_ptr,
    output_ptr,
    M,
    H,
    h_tiles,
    total_tiles,
    BH: tl.constexpr,
    UNROLL: tl.constexpr,
    EVEN_H: tl.constexpr,
):
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    if EVEN_H:
        if UNROLL == 8:
            stride = nprogs * 8
            for tile in tl.range(pid * 8, total_tiles, stride):
                ar = tl.arange(0, BH)
                t0 = tile
                t1 = tl.minimum(tile + 1, total_tiles - 1)
                t2 = tl.minimum(tile + 2, total_tiles - 1)
                t3 = tl.minimum(tile + 3, total_tiles - 1)
                t4 = tl.minimum(tile + 4, total_tiles - 1)
                t5 = tl.minimum(tile + 5, total_tiles - 1)
                t6 = tl.minimum(tile + 6, total_tiles - 1)
                t7 = tl.minimum(tile + 7, total_tiles - 1)
                m0 = t0 // h_tiles
                m1 = t1 // h_tiles
                m2 = t2 // h_tiles
                m3 = t3 // h_tiles
                m4 = t4 // h_tiles
                m5 = t5 // h_tiles
                m6 = t6 // h_tiles
                m7 = t7 // h_tiles
                o0f = (t0 - m0 * h_tiles) * BH + ar
                o1f = (t1 - m1 * h_tiles) * BH + ar
                o2f = (t2 - m2 * h_tiles) * BH + ar
                o3f = (t3 - m3 * h_tiles) * BH + ar
                o4f = (t4 - m4 * h_tiles) * BH + ar
                o5f = (t5 - m5 * h_tiles) * BH + ar
                o6f = (t6 - m6 * h_tiles) * BH + ar
                o7f = (t7 - m7 * h_tiles) * BH + ar
                a0 = tl.load(input_ptr + m0 * 2 * H + o0f).to(tl.float32)
                b0 = tl.load(input_ptr + m0 * 2 * H + H + o0f).to(tl.float32)
                a1 = tl.load(input_ptr + m1 * 2 * H + o1f).to(tl.float32)
                b1 = tl.load(input_ptr + m1 * 2 * H + H + o1f).to(tl.float32)
                a2 = tl.load(input_ptr + m2 * 2 * H + o2f).to(tl.float32)
                b2 = tl.load(input_ptr + m2 * 2 * H + H + o2f).to(tl.float32)
                a3 = tl.load(input_ptr + m3 * 2 * H + o3f).to(tl.float32)
                b3 = tl.load(input_ptr + m3 * 2 * H + H + o3f).to(tl.float32)
                a4 = tl.load(input_ptr + m4 * 2 * H + o4f).to(tl.float32)
                b4 = tl.load(input_ptr + m4 * 2 * H + H + o4f).to(tl.float32)
                a5 = tl.load(input_ptr + m5 * 2 * H + o5f).to(tl.float32)
                b5 = tl.load(input_ptr + m5 * 2 * H + H + o5f).to(tl.float32)
                a6 = tl.load(input_ptr + m6 * 2 * H + o6f).to(tl.float32)
                b6 = tl.load(input_ptr + m6 * 2 * H + H + o6f).to(tl.float32)
                a7 = tl.load(input_ptr + m7 * 2 * H + o7f).to(tl.float32)
                b7 = tl.load(input_ptr + m7 * 2 * H + H + o7f).to(tl.float32)
                s0 = 1.59576912 * a0 * (1 + 0.044715 * (a0 * a0))
                s1 = 1.59576912 * a1 * (1 + 0.044715 * (a1 * a1))
                s2 = 1.59576912 * a2 * (1 + 0.044715 * (a2 * a2))
                s3 = 1.59576912 * a3 * (1 + 0.044715 * (a3 * a3))
                s4 = 1.59576912 * a4 * (1 + 0.044715 * (a4 * a4))
                s5 = 1.59576912 * a5 * (1 + 0.044715 * (a5 * a5))
                s6 = 1.59576912 * a6 * (1 + 0.044715 * (a6 * a6))
                s7 = 1.59576912 * a7 * (1 + 0.044715 * (a7 * a7))
                r0 = a0 * (1.0 / (1.0 + tl.exp(-s0))) * b0
                r1 = a1 * (1.0 / (1.0 + tl.exp(-s1))) * b1
                r2 = a2 * (1.0 / (1.0 + tl.exp(-s2))) * b2
                r3 = a3 * (1.0 / (1.0 + tl.exp(-s3))) * b3
                r4 = a4 * (1.0 / (1.0 + tl.exp(-s4))) * b4
                r5 = a5 * (1.0 / (1.0 + tl.exp(-s5))) * b5
                r6 = a6 * (1.0 / (1.0 + tl.exp(-s6))) * b6
                r7 = a7 * (1.0 / (1.0 + tl.exp(-s7))) * b7
                tl.store(output_ptr + m0 * H + o0f, r0.to(tl.float32))
                tl.store(output_ptr + m1 * H + o1f, r1.to(tl.float32))
                tl.store(output_ptr + m2 * H + o2f, r2.to(tl.float32))
                tl.store(output_ptr + m3 * H + o3f, r3.to(tl.float32))
                tl.store(output_ptr + m4 * H + o4f, r4.to(tl.float32))
                tl.store(output_ptr + m5 * H + o5f, r5.to(tl.float32))
                tl.store(output_ptr + m6 * H + o6f, r6.to(tl.float32))
                tl.store(output_ptr + m7 * H + o7f, r7.to(tl.float32))
        else:
            stride = nprogs * 4
            for tile in tl.range(pid * 4, total_tiles, stride):
                ar = tl.arange(0, BH)
                t0 = tile
                t1 = tl.minimum(tile + 1, total_tiles - 1)
                t2 = tl.minimum(tile + 2, total_tiles - 1)
                t3 = tl.minimum(tile + 3, total_tiles - 1)
                m0 = t0 // h_tiles
                m1 = t1 // h_tiles
                m2 = t2 // h_tiles
                m3 = t3 // h_tiles
                o0f = (t0 - m0 * h_tiles) * BH + ar
                o1f = (t1 - m1 * h_tiles) * BH + ar
                o2f = (t2 - m2 * h_tiles) * BH + ar
                o3f = (t3 - m3 * h_tiles) * BH + ar
                a0 = tl.load(input_ptr + m0 * 2 * H + o0f).to(tl.float32)
                b0 = tl.load(input_ptr + m0 * 2 * H + H + o0f).to(tl.float32)
                a1 = tl.load(input_ptr + m1 * 2 * H + o1f).to(tl.float32)
                b1 = tl.load(input_ptr + m1 * 2 * H + H + o1f).to(tl.float32)
                a2 = tl.load(input_ptr + m2 * 2 * H + o2f).to(tl.float32)
                b2 = tl.load(input_ptr + m2 * 2 * H + H + o2f).to(tl.float32)
                a3 = tl.load(input_ptr + m3 * 2 * H + o3f).to(tl.float32)
                b3 = tl.load(input_ptr + m3 * 2 * H + H + o3f).to(tl.float32)
                s0 = 1.59576912 * a0 * (1 + 0.044715 * (a0 * a0))
                s1 = 1.59576912 * a1 * (1 + 0.044715 * (a1 * a1))
                s2 = 1.59576912 * a2 * (1 + 0.044715 * (a2 * a2))
                s3 = 1.59576912 * a3 * (1 + 0.044715 * (a3 * a3))
                r0 = a0 * (1.0 / (1.0 + tl.exp(-s0))) * b0
                r1 = a1 * (1.0 / (1.0 + tl.exp(-s1))) * b1
                r2 = a2 * (1.0 / (1.0 + tl.exp(-s2))) * b2
                r3 = a3 * (1.0 / (1.0 + tl.exp(-s3))) * b3
                tl.store(output_ptr + m0 * H + o0f, r0.to(tl.float32))
                tl.store(output_ptr + m1 * H + o1f, r1.to(tl.float32))
                tl.store(output_ptr + m2 * H + o2f, r2.to(tl.float32))
                tl.store(output_ptr + m3 * H + o3f, r3.to(tl.float32))
    else:
        for tile in tl.range(pid, total_tiles, nprogs):
            m = tile // h_tiles
            offs = (tile - m * h_tiles) * BH + tl.arange(0, BH)
            hmask = offs < H
            x_a = tl.load(input_ptr + m * 2 * H + offs, mask=hmask, other=0.0).to(
                tl.float32
            )
            x_b = tl.load(input_ptr + m * 2 * H + H + offs, mask=hmask, other=0.0).to(
                tl.float32
            )
            s = 1.59576912 * x_a * (1 + 0.044715 * (x_a * x_a))
            out = x_a * (1.0 / (1.0 + tl.exp(-s))) * x_b
            tl.store(output_ptr + m * H + offs, out.to(tl.float32), mask=hmask)


@triton.jit
def dgeglu_kernel(
    grad_out_ptr,
    input_ptr,
    grad_in_ptr,
    M,
    H,
    stride_grad_out_m,
    stride_grad_out_h,
    stride_in_m,
    stride_in_h,
    stride_grad_in_m,
    stride_grad_in_h,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)

    grad_out_ptr = (
        grad_out_ptr
        + offs_m[:, None] * stride_grad_out_m
        + offs_h[None, :] * stride_grad_out_h
    )
    input_a_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    )
    input_b_ptr = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    )
    grad_a_ptr = (
        grad_in_ptr
        + offs_m[:, None] * stride_grad_in_m
        + offs_h[None, :] * stride_grad_in_h
    )
    grad_b_ptr = (
        grad_in_ptr
        + offs_m[:, None] * stride_grad_in_m
        + (offs_h[None, :] + H) * stride_grad_in_h
    )

    grad_out = tl.load(grad_out_ptr, mask=mask, other=0.0).to(tl.float32)
    x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
    x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)

    tanh_out = tanh(0.79788456 * x_a * (1 + 0.044715 * pow(x_a, 2)))
    gelu_out = 0.5 * x_a * (1 + tanh_out)

    # dgelu/dx
    sech2 = 1 - pow(tanh_out, 2)
    dgelu = 0.5 * (1 + tanh_out) + 0.5 * x_a * sech2 * 0.79788456 * (
        1 + 3 * 0.044715 * pow(x_a, 2)
    )

    grad_a = grad_out * x_b * dgelu
    grad_b = grad_out * gelu_out

    tl.store(grad_a_ptr, grad_a.to(x_a.dtype), mask=mask)
    tl.store(grad_b_ptr, grad_b.to(x_a.dtype), mask=mask)


def geglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    logger.debug("GEMS GEGLU")
    if input_tensor.numel() == 0:
        half_last = input_tensor.shape[-1] // 2
        return torch.empty(
            (*input_tensor.shape[:-1], half_last),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)

    input_2d = input_tensor.contiguous().view(M, 2 * H)
    output_2d = torch.empty(M, H, device=input_tensor.device, dtype=input_tensor.dtype)

    block_h = min(GCU_MAX_BLOCK_H, _next_pow2(H))
    h_tiles = triton.cdiv(H, block_h)
    total_tiles = M * h_tiles
    even_h = H % block_h == 0
    unroll = 8 if (input_tensor.element_size() == 2 and block_h <= 512) else 4

    geglu_kernel[(min(total_tiles, GCU_GRID_CAP),)](
        input_2d,
        output_2d,
        M,
        H,
        h_tiles,
        total_tiles,
        BH=block_h,
        UNROLL=unroll,
        EVEN_H=even_h,
        num_warps=4,
    )
    return output_2d.view(*shape[:-1], H)


def dgeglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS DGEGLU")
    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)

    grad_out_2d = grad_output.contiguous().view(M, H)
    input_2d = input_tensor.contiguous().view(M, 2 * H)
    grad_in_2d = torch.empty_like(input_2d)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(H, META["BLOCK_SIZE_H"]),
    )

    dgeglu_kernel[grid](
        grad_out_2d,
        input_2d,
        grad_in_2d,
        M,
        H,
        grad_out_2d.stride(0),
        grad_out_2d.stride(1),
        input_2d.stride(0),
        input_2d.stride(1),
        grad_in_2d.stride(0),
        grad_in_2d.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_H=64,
    )
    # print(dgeglu)
    return grad_in_2d.view_as(input_tensor)

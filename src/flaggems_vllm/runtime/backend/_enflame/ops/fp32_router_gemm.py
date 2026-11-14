# SPDX-License-Identifier: Apache-2.0
"""FP32 MiniMax-M3 router GEMM through FlagGems, without vLLM extensions.

Import before torch.compile / CUDA graph warmup. BF16 activations are converted
to FP32 on load; router weights and returned logits stay FP32. No bias is
applied here. Unsupported dimensions or larger batches are explicitly rejected.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import device as runtime_device
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


def supports_fp32_router_gemm(hidden_size, num_experts, input_dtype, weight_dtype):
    """Metadata predicate for the model dimensions validated by this adapter."""
    return (
        (hidden_size, num_experts) == (6144, 128)
        and input_dtype in (torch.bfloat16, torch.float32)
        and weight_dtype == torch.float32
    )


def _check(input, router_weight):
    if input.ndim != 2 or router_weight.ndim != 2:
        raise ValueError("FP32 router GEMM requires 2D input and router_weight")
    if input.device.type != runtime_device.name or router_weight.device != input.device:
        raise ValueError(
            f"FP32 router input and weight must share a {runtime_device.name} device"
        )
    if input.shape[1] != router_weight.shape[1]:
        raise ValueError("FP32 router input and weight hidden dimensions must match")
    if not supports_fp32_router_gemm(
        input.shape[1], router_weight.shape[0], input.dtype, router_weight.dtype
    ):
        raise NotImplementedError(
            "FP32 router GEMM supports MiniMax-M3 [M,6144] input in BF16/FP32 "
            "and [128,6144] FP32 weight"
        )
    if input.shape[0] > 32:
        raise NotImplementedError("FP32 router specialization supports M<=32")


@libentry()
@triton.jit
def _router_m1_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    E: tl.constexpr,
    H: tl.constexpr,
    input_stride_k: tl.constexpr,
    weight_stride_e: tl.constexpr,
    weight_stride_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M==1 form with H == 3*BLOCK_K: three fully independent load+mul chains
    # (no loop-carried accumulator dependency), one elementwise combine, and a
    # single final reduction. All six loads issue back to back, so the DRAM
    # latency of one K chunk overlaps the others.
    expert = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    base_w = weight_ptr + expert * weight_stride_e
    x0 = tl.load(input_ptr + cols * input_stride_k).to(tl.float32)
    w0 = tl.load(base_w + cols * weight_stride_k)
    x1 = tl.load(input_ptr + (BLOCK_K + cols) * input_stride_k).to(tl.float32)
    w1 = tl.load(base_w + (BLOCK_K + cols) * weight_stride_k)
    x2 = tl.load(input_ptr + (2 * BLOCK_K + cols) * input_stride_k).to(tl.float32)
    w2 = tl.load(base_w + (2 * BLOCK_K + cols) * weight_stride_k)
    logit = tl.sum(x0 * w0 + x1 * w1 + x2 * w2)
    tl.store(output_ptr + expert, logit)


@libentry()
@triton.jit
def _router_rows_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M,
    E: tl.constexpr,
    H: tl.constexpr,
    input_stride_m: tl.constexpr,
    input_stride_k: tl.constexpr,
    weight_stride_e: tl.constexpr,
    weight_stride_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Row-blocked 2D form: one program per (expert, row block); each program
    # keeps a [BLOCK_M, BLOCK_K] fp32 accumulator tile, applies a broadcast fma
    # against the [BLOCK_K] weight chunk per K step, and performs a single
    # axis=1 reduction at the end. That amortizes the per-row reduction trees
    # (BLOCK_M serialized tl.sum per K step in the 1D static_range form) into
    # one tree with cross-row overlap.
    #
    # The 11-07 1D static_range form was a workaround for the 2D-tile
    # operandSegmentSizes compile failure (compiler_issues/vllm_moe_sum.md);
    # with the toolchain re-synced (2026-11-10, see the sibling router_gemm
    # dot kernel) the 2D form compiles natively and wins 1.2-2.1x.
    #
    # num_warps stays 1: multi-warp reductions on BLOCK_K<=1024 tiles silently
    # miscompile (probe 2026-11-12: BLOCK_K=1024 x nw>=4 wrong values, nw=2 in
    # the same danger family; nw=1 correct across the sweep).
    expert = tl.program_id(0)
    row_base = tl.program_id(1) * BLOCK_M
    rows = row_base + tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    for start in range(0, H, BLOCK_K):
        cols = start + offsets_k
        x = tl.load(
            input_ptr + rows[:, None] * input_stride_m + cols[None, :] * input_stride_k,
            mask=(rows[:, None] < M) & (cols[None, :] < H),
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + expert * weight_stride_e + cols * weight_stride_k,
            mask=cols < H,
            other=0.0,
        )
        accumulator = tl.fma(x, weight[None, :], accumulator)
    logits = tl.sum(accumulator, axis=1)
    tl.store(output_ptr + rows * E + expert, logits, mask=rows < M)


def _launch_m1(input, router_weight, *, block_k, num_warps):
    output = torch.empty(
        (input.shape[0], router_weight.shape[0]),
        dtype=torch.float32,
        device=input.device,
    )
    with torch_device_fn.device(input.device):
        _router_m1_kernel[(router_weight.shape[0],)](
            input,
            router_weight,
            output,
            router_weight.shape[0],
            input.shape[1],
            input.stride(1),
            router_weight.stride(0),
            router_weight.stride(1),
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
    return output


def _launch(input, router_weight, *, block_m, block_k, num_warps):
    output = torch.empty(
        (input.shape[0], router_weight.shape[0]),
        dtype=torch.float32,
        device=input.device,
    )
    m = input.shape[0]
    with torch_device_fn.device(input.device):
        _router_rows_kernel[(router_weight.shape[0], triton.cdiv(m, block_m))](
            input,
            router_weight,
            output,
            m,
            router_weight.shape[0],
            input.shape[1],
            input.stride(0),
            input.stride(1),
            router_weight.stride(0),
            router_weight.stride(1),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
    return output


@torch.library.custom_op("flaggems_vllm::fp32_router_gemm", mutates_args=())
def _fp32_router_gemm_op(
    input: torch.Tensor, router_weight: torch.Tensor
) -> torch.Tensor:
    _check(input, router_weight)
    if input.shape[0] == 0:
        return torch.empty(
            (0, router_weight.shape[0]), dtype=torch.float32, device=input.device
        )
    # Fixed launch metadata; no runtime tuning or unordered atomic reductions.
    # Probe matrix (2026-11-12, GCU400, do_bench w1000/r100, kernel mode):
    #   M==1 -> 3-chain unrolled kernel (independent loads, one reduction);
    #   2<=M<=16 -> 2D row tile BLOCK_M=next_pow2(M), BLOCK_K=2048 (divides H,
    #               DSM 56-264KB, dual-program envelope);
    #   M>16 -> BLOCK_M=32 with BLOCK_K=1024: at BLOCK_K=2048 the
    #               32x2048 tile exceeds the 448KB dual-program envelope and
    #               halves occupancy (bf16 73.8us vs 46.5us).
    m = input.shape[0]
    if m == 1:
        return _launch_m1(input, router_weight, block_k=2048, num_warps=2)
    if m <= 4:
        block_m = 4
    elif m <= 8:
        block_m = 8
    elif m <= 16:
        block_m = 16
    else:
        block_m = 32
    block_k = 2048 if m <= 16 else 1024
    return _launch(input, router_weight, block_m=block_m, block_k=block_k, num_warps=1)


@_fp32_router_gemm_op.register_fake
def _fp32_router_gemm_fake(input, router_weight):
    _check(input, router_weight)
    return torch.empty(
        (input.shape[0], router_weight.shape[0]),
        dtype=torch.float32,
        device=input.device,
    )


def fp32_router_gemm(input, router_weight):
    """Fresh FP32 [M,128] logits for input [M,6144], weight [128,6144]."""
    return _fp32_router_gemm_op(input, router_weight)

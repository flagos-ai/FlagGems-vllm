# SPDX-License-Identifier: Apache-2.0
"""FP32 router GEMM for MiniMax-M3 and Hy4 Flash without vLLM extensions.

Import before torch.compile / CUDA graph warmup. BF16 activations are converted
to FP32 on load; router weights and returned logits stay FP32. No bias is
applied here. The MiniMax-M3 specialization accepts up to 32 rows.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


def supports_fp32_router_gemm(hidden_size, num_experts, input_dtype, weight_dtype):
    """Metadata predicate for the model dimensions validated by this adapter."""
    return (
        (hidden_size, num_experts) in ((6144, 128), (3072, 512))
        and input_dtype in (torch.bfloat16, torch.float32)
        and weight_dtype == torch.float32
    )


def _check(input, router_weight):
    if input.ndim != 2 or router_weight.ndim != 2:
        raise ValueError("FP32 router GEMM requires 2D input and router_weight")
    if input.device.type != "cuda" or router_weight.device != input.device:
        raise ValueError("FP32 router input and weight must share a CUDA device")
    if input.shape[1] != router_weight.shape[1]:
        raise ValueError("FP32 router input and weight hidden dimensions must match")
    if not supports_fp32_router_gemm(
        input.shape[1], router_weight.shape[0], input.dtype, router_weight.dtype
    ):
        raise NotImplementedError(
            "FP32 router GEMM requires BF16/FP32 activations and FP32 weights "
            "with (hidden size, experts) equal to (6144,128) or (3072,512)"
        )
    if input.shape[1] == 6144 and input.shape[0] > 32:
        raise NotImplementedError("MiniMax-M3 router specialization supports M<=32")


@libentry()
@triton.jit
def _router_kernel(
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
    expert = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.full((BLOCK_M, BLOCK_K), 0.0, tl.float32)
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


def _launch(input, router_weight, *, block_m, block_k, num_warps):
    output = torch.empty(
        (input.shape[0], router_weight.shape[0]),
        dtype=torch.float32,
        device=input.device,
    )
    with torch_device_fn.device(input.device):
        _router_kernel[(router_weight.shape[0], triton.cdiv(input.shape[0], block_m))](
            input,
            router_weight,
            output,
            input.shape[0],
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


@libentry()
@triton.jit
def _router_matrix(
    x,
    w,
    out,
    rows,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # K=3072 and N=512 divide the tile sizes. Preserve the FP32 gate weights
    # and use three tensor-core products for FP32 input precision.
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    expert = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for offset in range(0, 3072, BK):
        kk = k + offset
        a = tl.load(
            x + row[:, None] * stride_xm + kk[None, :] * stride_xk,
            row[:, None] < rows,
            0.0,
        ).to(tl.float32)
        b = tl.load(w + expert[None, :] * stride_wn + kk[:, None] * stride_wk)
        acc = tl.dot(a, b, acc, input_precision="tf32x3")
    tl.store(out + row[:, None] * 512 + expert[None, :], acc, row[:, None] < rows)


def _launch_matrix(x, weight):
    n = x.shape[0]
    out = torch.empty((n, 512), dtype=torch.float32, device=x.device)
    with torch_device_fn.device(x.device):
        _router_matrix[(triton.cdiv(n, 32), 16)](
            x,
            weight,
            out,
            n,
            *x.stride(),
            *weight.stride(),
            BM=32,
            BN=32,
            BK=64,
            num_warps=4,
            num_stages=3,
        )
    return out


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
    m = input.shape[0]
    if (
        input.shape[1] == 3072
        and m > 32
        and triton.runtime.driver.active.get_current_target().backend == "cuda"
        and torch.cuda.get_device_capability(input.device)[0] >= 8
    ):
        return _launch_matrix(input, router_weight)
    if m == 1:
        block_m, block_k = 1, 256
    elif input.dtype == torch.float32:
        block_m, block_k = min(triton.next_power_of_2(m), 4), 1024
    elif m <= 4:
        block_m, block_k = triton.next_power_of_2(m), 1024
    elif m <= 8:
        block_m, block_k = 4, 1024
    elif m <= 16:
        block_m, block_k = 8, 1024
    else:
        block_m, block_k = 16, 512
    return _launch(input, router_weight, block_m=block_m, block_k=block_k, num_warps=4)


@_fp32_router_gemm_op.register_fake
def _fp32_router_gemm_fake(input, router_weight):
    _check(input, router_weight)
    return torch.empty(
        (input.shape[0], router_weight.shape[0]),
        dtype=torch.float32,
        device=input.device,
    )


def fp32_router_gemm(input, router_weight):
    """Return fresh FP32 [tokens, experts] logits with FP32 gate weights."""
    return _fp32_router_gemm_op(input, router_weight)

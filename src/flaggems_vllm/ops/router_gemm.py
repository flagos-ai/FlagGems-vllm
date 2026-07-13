import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _router_gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_idxs = k_start + offs_k
        a = tl.load(
            A + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + k_idxs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _router_gemm_splitk_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(k_start, k_end):
        k_idxs = k_iter * BLOCK_K + offs_k
        a = tl.load(
            A + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + k_idxs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    tl.atomic_add(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _general_router_gemm(x, weight_t, out, M, N, K):
    logger.debug(
        "FlagGems-vllm router_gemm, general, shape: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(x.device):
        _router_gemm_kernel[grid](
            x,
            weight_t,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            weight_t.stride(0),
            weight_t.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=64,
            GROUP_M=8,
            num_warps=4,
            num_stages=4,
        )
    return out


def _splitk_router_gemm(x, weight_t, out, M, N, K):
    logger.debug(
        "FlagGems-vllm router_gemm, splitk, shape: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(x.device):
        _router_gemm_splitk_kernel[grid](
            x,
            weight_t,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            weight_t.stride(0),
            weight_t.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=256,
            SPLIT_K=8,
            num_warps=4,
            num_stages=4,
        )
    return out


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM for MoE router gate.

    Args:
        x: Router input with shape (M, K).
        weight: Router weight with shape (N, K).
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("router_gemm expects 2D x and weight tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("router_gemm expects x.shape[1] == weight.shape[1]")
    if x.device != weight.device:
        raise ValueError("router_gemm expects x and weight to be on the same device")

    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()

    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    weight_t = weight.t()

    if M < 2048 and N < 2048 and K >= 4096:
        out.zero_()
        return _splitk_router_gemm(x, weight_t, out, M, N, K)
    return _general_router_gemm(x, weight_t, out, M, N, K)

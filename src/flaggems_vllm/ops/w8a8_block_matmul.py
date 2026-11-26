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

"""Block-scaled INT8 matmul and shared MetaX/PPU Triton kernels."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

# The last workspace element records how many split slices are initialized.
_MAX_SPLITS = tl.constexpr(16)


@triton.jit
def _w8a8_block_int8_kernel_core(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SWAP: tl.constexpr,
    SPLITS: tl.constexpr,
    WIDE_OFFSETS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * num_pid_n
    group_id = pid // group_width
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = (pid % group_width) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if WIDE_OFFSETS:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = As + offs_m * stride_asm
    b_scale_ptrs = Bs + (offs_n // GROUP_N) * stride_bsn
    scalar_b_scale = BLOCK_N <= GROUP_N and GROUP_N % BLOCK_N == 0

    split = tl.program_id(1)
    a_ptrs += split * BLOCK_K * stride_ak
    b_ptrs += split * BLOCK_K * stride_bk
    if SWAP:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in range(split, tl.cdiv(K, BLOCK_K), SPLITS):
        k_remaining = K - k_tile * BLOCK_K
        if SWAP:
            # The swap path maps the wide N dimension to the first dot
            # operand.  Load directly in [BN, BK] x [BK, BM] order so the backend
            # can use its native INT8 MMA layout without a transpose.
            a = tl.load(
                A
                + (k_tile * BLOCK_K + offs_k[:, None]) * stride_ak
                + offs_m[None, :] * stride_am,
                mask=(offs_k[:, None] < k_remaining) & mask_m[None, :],
                other=0,
            )
            b = tl.load(
                B
                + offs_n[:, None] * stride_bn
                + (k_tile * BLOCK_K + offs_k[None, :]) * stride_bk,
                mask=mask_n[:, None] & (offs_k[None, :] < k_remaining),
                other=0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=mask_m[:, None] & (offs_k[None, :] < k_remaining),
                other=0,
            )
            b = tl.load(
                b_ptrs,
                mask=mask_n[None, :] & (offs_k[:, None] < k_remaining),
                other=0,
            )
        a_scale = tl.load(
            a_scale_ptrs + (k_tile * BLOCK_K // GROUP_K) * stride_ask,
            mask=mask_m,
            other=0.0,
        )
        if SWAP:
            partial = tl.dot(b, a, out_dtype=tl.int32)
            if scalar_b_scale:
                b_scale_scalar = tl.load(
                    Bs
                    + (pid_n * BLOCK_N // GROUP_N) * stride_bsn
                    + (k_tile * BLOCK_K // GROUP_K) * stride_bsk
                )
                acc += partial.to(tl.float32) * (b_scale_scalar * a_scale[None, :])
            else:
                b_scale = tl.load(
                    b_scale_ptrs + (k_tile * BLOCK_K // GROUP_K) * stride_bsk,
                    mask=mask_n,
                    other=0.0,
                )
                acc += partial.to(tl.float32) * (b_scale[:, None] * a_scale[None, :])
        else:
            partial = tl.dot(a, b, out_dtype=tl.int32)
            if scalar_b_scale:
                b_scale_scalar = tl.load(
                    Bs
                    + (pid_n * BLOCK_N // GROUP_N) * stride_bsn
                    + (k_tile * BLOCK_K // GROUP_K) * stride_bsk
                )
                acc += partial.to(tl.float32) * (a_scale[:, None] * b_scale_scalar)
            else:
                b_scale = tl.load(
                    b_scale_ptrs + (k_tile * BLOCK_K // GROUP_K) * stride_bsk,
                    mask=mask_n,
                    other=0.0,
                )
                acc += partial.to(tl.float32) * (a_scale[:, None] * b_scale[None, :])
        a_ptrs += SPLITS * BLOCK_K * stride_ak
        b_ptrs += SPLITS * BLOCK_K * stride_bk

    if WIDE_OFFSETS:
        split = split.to(tl.int64)
    C += split * M * N
    if SWAP:
        # ``acc`` is [BLOCK_N, BLOCK_M] on the swap path.  Store it with
        # transposed pointer arithmetic so the backend does not need a separate
        # layout conversion before the store.
        c_ptrs = C + offs_n[:, None] * stride_cn + offs_m[None, :] * stride_cm
        mask_c = mask_n[:, None] & mask_m[None, :]
    else:
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask_c = mask_m[:, None] & mask_n[None, :]
    if C.dtype.element_ty == tl.float16:
        result = acc.to(tl.float16)
    elif C.dtype.element_ty == tl.bfloat16:
        result = acc.to(tl.bfloat16)
    else:
        result = acc
    tl.store(c_ptrs, result, mask=mask_c)


@triton.jit
def _w8a8_block_int8_gemv_kernel(
    A,
    B,
    C,
    As,
    Bs,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ask,
    stride_bsn,
    stride_bsk,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WIDE_OFFSETS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if WIDE_OFFSETS:
        offs_n = offs_n.to(tl.int64)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        remaining = K - k_tile * BLOCK_K
        a = tl.load(
            A + k_tile * BLOCK_K * stride_ak + offs_k * stride_ak,
            mask=offs_k < remaining,
            other=0,
        )
        b = tl.load(
            B
            + (k_tile * BLOCK_K + offs_k[:, None]) * stride_bk
            + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < remaining) & mask_n[None, :],
            other=0,
        )
        partial = tl.dot(a[None, :], b, out_dtype=tl.int32)
        a_scale = tl.load(As + k_tile * BLOCK_K // GROUP_K * stride_ask)
        b_scale = tl.load(
            Bs
            + (offs_n // GROUP_N) * stride_bsn
            + k_tile * BLOCK_K // GROUP_K * stride_bsk,
            mask=mask_n,
            other=0.0,
        )
        acc += tl.reshape(partial, (BLOCK_N,)).to(tl.float32) * a_scale * b_scale
    if C.dtype.element_ty == tl.float16:
        result = acc.to(tl.float16)
    elif C.dtype.element_ty == tl.bfloat16:
        result = acc.to(tl.bfloat16)
    else:
        result = acc
    tl.store(C + offs_n, result, mask=mask_n)


@triton.jit
def _w8a8_block_int8_reduce_splitk(
    P, C, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, _MAX_SPLITS)
    active_splits = tl.load(P + tl.cast(M, tl.int64) * N * _MAX_SPLITS).to(tl.int32)
    partial = tl.load(
        P + split[:, None].to(tl.int64) * M * N + offsets[None, :],
        (split[:, None] < active_splits) & (offsets[None, :] < M * N),
        other=0.0,
    )
    tl.store(C + offsets, tl.sum(partial, axis=0), offsets < M * N)


def _splitk_post_hook(args, exception):
    # Include reduction in candidate timings rather than tuning partial GEMM
    # alone.  LibEntry launches the selected partial kernel on cache hits;
    # the public wrapper launches the same reduction after it.
    if exception is None:
        _w8a8_block_int8_reduce_splitk[(triton.cdiv(args["M"] * args["N"], 256),)](
            args["P"], args["C"], args["M"], args["N"], 256, num_warps=4
        )


def w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Compute ``A @ B.T`` for block-scaled INT8 inputs using a fixed Triton configuration."""
    if A.ndim < 2 or B.ndim != 2 or As.ndim != A.ndim or Bs.ndim != 2:
        raise ValueError("A must have rank >= 2, B/Bs rank 2, and As the rank of A")
    if any(t.device != A.device for t in (B, As, Bs)):
        raise ValueError("all inputs must be on the same device")
    if As.requires_grad or Bs.requires_grad:
        raise NotImplementedError("w8a8_block_int8_matmul is inference-only")
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise NotImplementedError("A and B must be int8")
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        raise NotImplementedError("scales must be float32")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("output_dtype must be float16, bfloat16 or float32")
    if len(block_size) != 2 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain two positive integers")
    if A.shape[-1] != B.shape[-1]:
        raise ValueError("A and B must have the same K dimension")
    if not all(t.is_contiguous() for t in (A, B, As, Bs)):
        raise NotImplementedError("inputs must be contiguous")
    group_n, group_k = block_size
    if group_k not in (32, 64, 128, 256):
        raise NotImplementedError("block_k must be 32, 64, 128 or 256")
    K = A.shape[-1]
    M = 1
    for dim in A.shape[:-1]:
        M *= dim
    N = B.shape[0]
    groups = triton.cdiv(K, group_k)
    if As.shape != A.shape[:-1] + (groups,):
        raise ValueError("As has an invalid block-scale shape")
    if Bs.shape != (triton.cdiv(N, group_n), groups):
        raise ValueError("Bs has an invalid block-scale shape")
    if M == 0 or N == 0:
        return A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    block_m, block_n, block_k = 16, 32, 32
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), 1)
    with torch_device_fn.device(A.device):
        _w8a8_block_int8_kernel_core[grid](
            A,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            K,
            1,
            1,
            K,
            N,
            1,
            groups,
            1,
            groups,
            1,
            GROUP_N=group_n,
            GROUP_K=group_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=1,
            SWAP=False,
            SPLITS=1,
            WIDE_OFFSETS=True,
            num_warps=4,
            num_stages=1,
        )
    return C

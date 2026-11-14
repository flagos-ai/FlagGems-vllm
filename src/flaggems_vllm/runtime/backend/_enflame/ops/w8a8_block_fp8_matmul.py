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

"""Block-scaled FP8 GEMM (w8a8_block_fp8_matmul) for Enflame GCU.

Semantics match vLLM ``w8a8_block_fp8_matmul``:
``C = (A * As) @ (B * Bs)^T`` with per-token-group activation scales
``As[M, K/group_k]`` and per-weight-block scales ``Bs[N/group_n, K/group_k]``.

GCU adaptation of the Hopper reference (``_nvidia/hopper/ops``):
one tiled ``tl.dot`` kernel launch per call, no atomics, 1-D grid kept under
the GCU grid.x hardware limit. Row/col tails use 2-D single-dim masked loads
(wrapping indices with ``% M``/``% N`` puts a vector divide into the load
addresses, which scalarizes the whole K loop on GCU, ~150x slower; see the
fp8 blockwise GEMM entry in the workspace perf playbook). Ragged K is routed
to a dedicated kernel whose tail block sits outside the main loop: a compound
(row & K-tail) mask inside a K loop miscompiles on GCU
(compiler_issues/vllm_cutlass_scaled_mm_fp8_dot_compound_mask.md). Operand
tiles keep their storage dtype when the GCU MMA consumes it directly
(e4m3fn/fp16/bf16); other storage dtypes (fp32 GCU fp8 test stand-in, e5m2)
are cast to fp16 in-kernel before ``tl.dot``, which is exact for every
representable fp8 payload.
"""

from typing import List

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import device as runtime_device
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

# GCU launch constraint (compiler_issues/grid_hard_limits.md): grid.x <= 65535.
_GCU_GRID_X_LIMIT = 65535

_NATIVE_DOT_DTYPES = (torch.float8_e4m3fn, torch.float16, torch.bfloat16)
_SUPPORTED_INPUT_DTYPES = _NATIVE_DOT_DTYPES + (torch.float8_e5m2, torch.float32)
_SUPPORTED_GROUP_K = (32, 64, 128, 256)
_SUPPORTED_OUTPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _get_tuning_configs():
    # num_warps stays at the GCU hard cap (4). Worst-case DSM footprint
    # (fp32 operands, BM=128/BN=128/BK=128, double-buffered loads + fp16 cast
    # tiles) stays under the 458752B dual-program budget on gcu400.
    return [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4,
            num_stages=2,
        ),
    ]


def _prune_matmul_configs(configs, named_args, **kwargs):
    m = named_args["M"]
    n = named_args["N"]
    fitting = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        if triton.cdiv(m, block_m) * triton.cdiv(n, block_n) > _GCU_GRID_X_LIMIT:
            continue
        fitting.append(config)
    if not fitting:
        raise NotImplementedError(
            "w8a8_block_fp8_matmul: GCU grid.x limit "
            f"{_GCU_GRID_X_LIMIT} exceeded for M={m}, N={n} with every "
            "supported tile"
        )
    tight = [
        config
        for config in fitting
        if config.kwargs["BLOCK_M"] <= max(16, triton.next_power_of_2(m))
    ]
    return tight or fitting


@libentry()
@libtuner(
    configs=_get_tuning_configs(),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    prune_configs_by={"early_config_prune": _prune_matmul_configs},
    warmup=5,
    rep=5,
    flagtune_op_name="w8a8_block_fp8_matmul",
)
@triton.jit
def w8a8_block_fp8_matmul_kernel(
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
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_NATIVE_DOT: tl.constexpr,
):
    tl.static_assert(GROUP_K % BLOCK_K == 0)

    SCALAR_BS: tl.constexpr = (BLOCK_N <= GROUP_N) and (GROUP_N % BLOCK_N == 0)

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Row/col tails use 2-D single-dim masks: wrapping indices with % M / % N
    # or clamping with tl.minimum puts a non-affine index into the load
    # addresses, which scalarizes the whole K loop on GCU (~150x slower).
    # The host only routes K % BLOCK_K == 0 here; a compound (row & K-tail)
    # mask inside the K loop miscompiles on GCU, so ragged K goes to
    # w8a8_block_fp8_matmul_kernel_ragged.
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    masks_am = offs_am < M
    masks_bn = offs_bn < N
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    as_base = As + offs_am * stride_As_m
    if SCALAR_BS:
        b_scale_group = (pid_n * BLOCK_N) // GROUP_N
        bs_base = Bs + b_scale_group * stride_Bs_n
    else:
        offs_bsn = offs_bn // GROUP_N
        bs_base = Bs + offs_bsn * stride_Bs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=masks_am[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=masks_bn[None, :], other=0.0)
        if not A_NATIVE_DOT:
            a = a.to(tl.float16)
            b = b.to(tl.float16)

        offs_ks = (k * BLOCK_K) // GROUP_K
        a_s = tl.load(as_base + offs_ks * stride_As_k, mask=masks_am, other=0.0)
        if SCALAR_BS:
            b_s = tl.load(bs_base + offs_ks * stride_Bs_k)
            acc += tl.dot(a, b) * (a_s * b_s)[:, None]
        else:
            b_s = tl.load(bs_base + offs_ks * stride_Bs_k, mask=masks_bn, other=0.0)
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = acc.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = acc.to(tl.float16)
    else:
        c = acc.to(tl.float32)

    c_ptrs = C + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=masks_am[:, None] & masks_bn[None, :])


@triton.jit
def w8a8_block_fp8_matmul_kernel_ragged(
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
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TAIL_K: tl.constexpr,
    A_NATIVE_DOT: tl.constexpr,
):
    tl.static_assert(GROUP_K % BLOCK_K == 0)

    SCALAR_BS: tl.constexpr = (BLOCK_N <= GROUP_N) and (GROUP_N % BLOCK_N == 0)

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    masks_am = offs_am < M
    masks_bn = offs_bn < N
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    as_base = As + offs_am * stride_As_m
    if SCALAR_BS:
        b_scale_group = (pid_n * BLOCK_N) // GROUP_N
        bs_base = Bs + b_scale_group * stride_Bs_n
    else:
        offs_bsn = offs_bn // GROUP_N
        bs_base = Bs + offs_bsn * stride_Bs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Full K tiles only: a compound (row & K-tail) mask inside this loop
    # miscompiles on GCU, so the ragged tail is split out below.
    k_main = (K // BLOCK_K) * BLOCK_K
    for k in range(0, k_main, BLOCK_K):
        a = tl.load(a_ptrs, mask=masks_am[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=masks_bn[None, :], other=0.0)
        if not A_NATIVE_DOT:
            a = a.to(tl.float16)
            b = b.to(tl.float16)

        offs_ks = k // GROUP_K
        a_s = tl.load(as_base + offs_ks * stride_As_k, mask=masks_am, other=0.0)
        if SCALAR_BS:
            b_s = tl.load(bs_base + offs_ks * stride_Bs_k)
            acc += tl.dot(a, b) * (a_s * b_s)[:, None]
        else:
            b_s = tl.load(bs_base + offs_ks * stride_Bs_k, mask=masks_bn, other=0.0)
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if K > k_main:
        # Tail block: recomputed affine pointers (consuming the loop-carried
        # a_ptrs/b_ptrs here inflates the kernel's DSM allocation ~4x and can
        # push fitting tiles past the hardware limit) and a TAIL_K-wide tile
        # (next_pow2 of the tail length) instead of a full BLOCK_K tile.
        k_tail = K - k_main
        offs_kt = tl.arange(0, TAIL_K)
        k_mask = offs_kt < k_tail
        a_ptrs_t = A + (
            offs_am[:, None] * stride_am + (k_main + offs_kt[None, :]) * stride_ak
        )
        b_ptrs_t = B + (
            (k_main + offs_kt[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
        )
        a = tl.load(a_ptrs_t, mask=masks_am[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs_t, mask=k_mask[:, None] & masks_bn[None, :], other=0.0)
        if not A_NATIVE_DOT:
            a = a.to(tl.float16)
            b = b.to(tl.float16)

        offs_ks = k_main // GROUP_K
        a_s = tl.load(as_base + offs_ks * stride_As_k, mask=masks_am, other=0.0)
        if SCALAR_BS:
            b_s = tl.load(bs_base + offs_ks * stride_Bs_k)
            acc += tl.dot(a, b) * (a_s * b_s)[:, None]
        else:
            b_s = tl.load(bs_base + offs_ks * stride_Bs_k, mask=masks_bn, other=0.0)
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

    if C.dtype.element_ty == tl.bfloat16:
        c = acc.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = acc.to(tl.float16)
    else:
        c = acc.to(tl.float32)

    c_ptrs = C + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=masks_am[:, None] & masks_bn[None, :])


def _tail_width(k_tail):
    # tl.dot needs K >= 16; shorter tails are zero-padded by the tail mask.
    return max(16, triton.next_power_of_2(k_tail))


def _launch_block_scaled_matmul(
    a, b, c, a_s, b_s, m, n, k, group_n, group_k, a_native_dot
):
    stride_args = (
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        a_s.stride(0),
        a_s.stride(1),
        b_s.stride(1),
        b_s.stride(0),
    )
    if group_k % 128 == 0:
        if k % 128 == 0:
            grid = lambda META: (
                triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),
            )
            w8a8_block_fp8_matmul_kernel[grid](
                a,
                b,
                c,
                a_s,
                b_s,
                m,
                n,
                k,
                *stride_args,
                GROUP_N=group_n,
                GROUP_K=group_k,
                A_NATIVE_DOT=a_native_dot,
            )
            return

        # Ragged K: fixed tile, tail block split out of the K loop
        # (compound masks inside the K loop miscompile on GCU). For M > 64,
        # BM=128 keeps num_pid_m at 1, halving B re-reads (measured -27% on
        # (83,7748,3884)); masked rows absorb the M tail.
        block_m = 16 if m <= 16 else (64 if m <= 64 else 128)
        while (
            triton.cdiv(m, block_m) * triton.cdiv(n, 128) > _GCU_GRID_X_LIMIT
            and block_m < 128
        ):
            block_m *= 2
        if triton.cdiv(m, block_m) * triton.cdiv(n, 128) > _GCU_GRID_X_LIMIT:
            raise NotImplementedError(
                "w8a8_block_fp8_matmul: GCU grid.x limit "
                f"{_GCU_GRID_X_LIMIT} exceeded for M={m}, N={n}"
            )
        w8a8_block_fp8_matmul_kernel_ragged[
            (triton.cdiv(m, block_m) * triton.cdiv(n, 128),)
        ](
            a,
            b,
            c,
            a_s,
            b_s,
            m,
            n,
            k,
            *stride_args,
            GROUP_N=group_n,
            GROUP_K=group_k,
            BLOCK_M=block_m,
            BLOCK_N=128,
            BLOCK_K=128,
            TAIL_K=_tail_width(k % 128),
            A_NATIVE_DOT=a_native_dot,
            num_warps=4,
            num_stages=2,
        )
        return

    # group_k in (32, 64): one scale group per K tile, launched raw with
    # BLOCK_K == group_k so the tuned config space can stay BLOCK_K=128.
    block_m = 16 if m <= 16 else 64
    while (
        triton.cdiv(m, block_m) * triton.cdiv(n, 128) > _GCU_GRID_X_LIMIT
        and block_m < 128
    ):
        block_m *= 2
    if triton.cdiv(m, block_m) * triton.cdiv(n, 128) > _GCU_GRID_X_LIMIT:
        raise NotImplementedError(
            "w8a8_block_fp8_matmul: GCU grid.x limit "
            f"{_GCU_GRID_X_LIMIT} exceeded for M={m}, N={n}"
        )
    grid_size = (triton.cdiv(m, block_m) * triton.cdiv(n, 128),)
    extra_args = {}
    if k % group_k == 0:
        kernel = w8a8_block_fp8_matmul_kernel.fn.fn
    else:
        kernel = w8a8_block_fp8_matmul_kernel_ragged
        extra_args["TAIL_K"] = _tail_width(k % group_k)
    kernel[grid_size](
        a,
        b,
        c,
        a_s,
        b_s,
        m,
        n,
        k,
        *stride_args,
        GROUP_N=group_n,
        GROUP_K=group_k,
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_K=group_k,
        A_NATIVE_DOT=a_native_dot,
        num_warps=4,
        num_stages=2,
        **extra_args,
    )


def _matrix_view(tensor, rows, cols):
    if tensor.ndim == 2:
        return tensor
    try:
        return tensor.view(rows, cols)
    except RuntimeError as exc:
        raise NotImplementedError("leading dimensions must be view-compatible") from exc


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Return A @ B.T with per-token-group and per-weight-block FP8 scales."""
    if len(block_size) != 2 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain two positive integers")
    group_n, group_k = block_size
    if group_k not in _SUPPORTED_GROUP_K:
        raise NotImplementedError(f"supported K group sizes are {_SUPPORTED_GROUP_K}")
    if A.ndim < 2 or B.ndim != 2 or As.ndim != A.ndim or Bs.ndim != 2:
        raise ValueError("A/As must have rank >= 2 and B/Bs must be rank 2")
    if A.device.type != runtime_device.name or any(
        tensor.device != A.device for tensor in (B, As, Bs)
    ):
        raise ValueError(f"all inputs must be on the same {runtime_device.name} device")
    if A.dtype not in _SUPPORTED_INPUT_DTYPES or B.dtype != A.dtype:
        raise NotImplementedError(
            "A/B must share an E4M3FN or E5M2 FP8 dtype, or the FP16/BF16/FP32 "
            "storage stand-in"
        )
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        raise NotImplementedError("scales must be FP32")
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise NotImplementedError("output dtype must be FP16, BF16 or FP32")
    if any(tensor.requires_grad for tensor in (A, B, As, Bs)):
        raise NotImplementedError("w8a8_block_fp8_matmul is inference-only")

    m = A.numel() // A.shape[-1]
    n, k = B.shape
    if A.shape[-1] != k:
        raise ValueError("incompatible A/B K dimensions")
    if As.shape != A.shape[:-1] + (triton.cdiv(k, group_k),):
        raise ValueError("invalid As shape")
    if Bs.shape != (triton.cdiv(n, group_n), triton.cdiv(k, group_k)):
        raise ValueError("invalid Bs shape")

    output_shape = A.shape[:-1] + (n,)
    c = torch.empty(output_shape, device=A.device, dtype=output_dtype)
    if m == 0 or n == 0:
        return c

    a_2d = _matrix_view(A, m, k)
    as_2d = _matrix_view(As, m, As.shape[-1])
    c_2d = _matrix_view(c, m, n)
    a_native_dot = A.dtype in _NATIVE_DOT_DTYPES

    with torch_device_fn.device(A.device):
        _launch_block_scaled_matmul(
            a_2d,
            B,
            c_2d,
            as_2d,
            Bs,
            m,
            n,
            k,
            group_n,
            group_k,
            a_native_dot,
        )
    return c

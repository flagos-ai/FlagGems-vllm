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

"""MetaX W8A8 block INT8 matrix multiplication.

The inputs are already quantized INT8 tensors.  The kernel performs the
integer dot product and accumulates in INT32, then applies the activation and
weight block scales in FP32 before converting the result to the requested
output dtype.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.flagtune.inference.cost_model import tensor_dtypes
from flaggems_vllm.ops.w8a8_block_matmul import (
    _MAX_SPLITS,
    _splitk_post_hook,
    _w8a8_block_int8_gemv_kernel,
    _w8a8_block_int8_kernel_core,
    _w8a8_block_int8_reduce_splitk,
)
from flaggems_vllm.ops.w8a8_block_matmul import (
    w8a8_block_matmul as _generic_w8a8_block_matmul,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner


def _prune_matmul_configs(configs, named_args, **kwargs):
    result = []
    for config in configs:
        bm, bn = config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"]
        if bm * bn > 16384:
            continue
        # Unaligned INT8 rows can exceed C550's private-memory limit with
        # larger layouts.  The compact tiles preserve the tail path.
        if named_args["K"] % 16 and (bm != 16 or bn > 64):
            continue
        result.append(config)
    return result


def _prune_gemv_configs(configs, named_args, **kwargs):
    if named_args["K"] % 16 == 0:
        return configs
    # Unaligned INT8 rows need compact per-warp tiles to avoid private-memory spills.
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_N"] <= 32 * config.num_warps
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_matmul_metax"),
    key=["M", "N", "K", "GROUP_N", "GROUP_K", "SPLITS"],
    strategy="default",
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    policy="flagtune",
    flagtune_op_id="flaggems_vllm/w8a8_block_matmul",
    flagtune_variant="metax_general",
    flagtune_dtype_resolver=tensor_dtypes("A", "B", "As", "Bs", "C"),
    flagtune_op_name="w8a8_block_matmul",
    flagtune_expand_op_name="w8a8_block_int8_matmul",
    prune_configs_by={"early_config_prune": _prune_matmul_configs},
)
@triton.jit
def _w8a8_block_int8_kernel_general(
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
    SPLITS: tl.constexpr,
):
    _w8a8_block_int8_kernel_core(
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
        GROUP_N=GROUP_N,
        GROUP_K=GROUP_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        SWAP=False,
        SPLITS=SPLITS,
        WIDE_OFFSETS=True,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_swap_metax"),
    key=["M", "N", "K", "GROUP_N", "GROUP_K", "SPLITS"],
    strategy="default",
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    policy="flagtune",
    flagtune_op_id="flaggems_vllm/w8a8_block_matmul",
    flagtune_variant="metax_swap",
    flagtune_dtype_resolver=tensor_dtypes("A", "B", "As", "Bs", "C"),
    flagtune_op_name="w8a8_block_matmul",
    flagtune_expand_op_name="w8a8_block_int8_swap",
    prune_configs_by={"early_config_prune": _prune_matmul_configs},
)
@triton.jit
def _w8a8_block_int8_kernel_swap(
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
    SPLITS: tl.constexpr,
):
    _w8a8_block_int8_kernel_core(
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
        GROUP_N=GROUP_N,
        GROUP_K=GROUP_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        SWAP=True,
        SPLITS=SPLITS,
        WIDE_OFFSETS=True,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_gemv_metax"),
    key=["N", "K"],
    strategy="default",
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    policy="flagtune",
    flagtune_op_id="flaggems_vllm/w8a8_block_matmul",
    flagtune_variant="metax_gemv",
    flagtune_dtype_resolver=tensor_dtypes("A", "B", "As", "Bs", "C"),
    flagtune_op_name="w8a8_block_matmul",
    flagtune_expand_op_name="w8a8_block_int8_gemv",
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
)
@triton.jit
def _w8a8_block_int8_gemv_kernel_tuned(
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
):
    _w8a8_block_int8_gemv_kernel(
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
        GROUP_N=GROUP_N,
        GROUP_K=GROUP_K,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        WIDE_OFFSETS=True,
    )


def _prune_splitk_configs(configs, named_args, **kwargs):
    # An unaligned INT8 row stride expands the larger dot layouts into
    # scalar loads and can exceed C550's 4 KiB/thread private-memory limit.
    # The compact swap layout remains valid for these K tails.
    if named_args["K"] % 16:
        return [
            config
            for config in configs
            if config.kwargs["BLOCK_M"] == 16
            and config.kwargs["BLOCK_N"] == 32
            and config.kwargs["SWAP"]
        ]
    return configs


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_splitk_metax"),
    key=["M", "N", "K", "GROUP_N", "GROUP_K"],
    strategy="default",
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    post_hook=_splitk_post_hook,
    policy="flagtune",
    flagtune_op_id="flaggems_vllm/w8a8_block_matmul",
    flagtune_variant="metax_splitk_complete",
    flagtune_dtype_resolver=tensor_dtypes("A", "B", "As", "Bs", "C"),
    prune_configs_by={"early_config_prune": _prune_splitk_configs},
    flagtune_op_name="w8a8_block_matmul",
    flagtune_expand_op_name="w8a8_block_int8_splitk",
)
@triton.jit
def _w8a8_block_int8_kernel_splitk(
    A,
    B,
    P,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLITS: tl.constexpr,
    SWAP: tl.constexpr,
):
    groups = tl.cdiv(K, GROUP_K)
    _w8a8_block_int8_kernel_core(
        A,
        B,
        P,
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
        GROUP_N=GROUP_N,
        GROUP_K=GROUP_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=GROUP_K,
        GROUP_M=GROUP_M,
        SWAP=SWAP,
        SPLITS=SPLITS,
        WIDE_OFFSETS=True,
    )
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.store(P + tl.cast(M, tl.int64) * N * _MAX_SPLITS, SPLITS)


def _splitk_matmul(A, B, As, Bs, M, N, K, group_n, group_k, output_dtype):
    C = torch.empty(A.shape[:-1] + (N,), device=A.device, dtype=output_dtype)
    P = torch.empty(
        (_MAX_SPLITS.value * M * N + 1,), device=A.device, dtype=torch.float32
    )
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        meta["SPLITS"],
    )
    with torch_device_fn.device(A.device):
        _w8a8_block_int8_kernel_splitk[grid](
            A, B, P, C, As, Bs, M, N, K, GROUP_N=group_n, GROUP_K=group_k
        )
        _w8a8_block_int8_reduce_splitk[(triton.cdiv(M * N, 256),)](
            P, C, M, N, 256, num_warps=4
        )
    return C


def w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Compute ``A @ B.T`` for block-scaled INT8 inputs on MetaX."""
    if A.ndim < 2 or B.ndim != 2 or As.ndim != A.ndim or Bs.ndim != 2:
        raise ValueError("A must have rank >= 2, B/Bs rank 2, and As the rank of A")
    if any(t.device != A.device for t in (B, As, Bs)) or A.device.type != "cuda":
        raise ValueError("all inputs must be on the same MetaX device")
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
    # Long reductions with few output tiles benefit from additional CTAs.
    # Keep the general path's FP32 workspace small; LibTuner selects the
    # split count and tile layout for this case.
    if (
        group_k == 128
        and 64 <= M <= 128
        and N >= 4096
        and 3072 <= K < 4096
        and K % 16 != 0
        and M * N * _MAX_SPLITS.value * 4 <= 64 * 1024 * 1024
    ):
        return _splitk_matmul(A, B, As, Bs, M, N, K, group_n, group_k, output_dtype)
    if group_k == 128 and K >= 4096 and M <= 128 and N <= 1536 and M * N <= 65536:
        return _splitk_matmul(A, B, As, Bs, M, N, K, group_n, group_k, output_dtype)
    expand = runtime.flagtune_enabled("w8a8_block_matmul")
    skinny_swap = 2 <= M <= 64 and N <= 1536 and 2304 < K <= 8192
    if not expand and M == 1 and K <= 2304 and group_k == 128:
        C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
        # A 128-wide tile keeps the MetaX INT8 MMA path in its native
        # 4-warp layout.  The 256-wide variant needs a larger shared-memory
        # tile and is slower for the small-M GEMV shapes that this dispatch
        # handles (and can exceed the 64 KiB shared-memory limit at lower
        # warp counts).
        block_n = 64 if N <= 128 else 128
        with torch_device_fn.device(A.device):
            _w8a8_block_int8_gemv_kernel[(triton.cdiv(N, block_n),)](
                A,
                B,
                C,
                As,
                Bs,
                N,
                K,
                A.stride(-1),
                B.stride(1),
                B.stride(0),
                As.stride(-1),
                Bs.stride(0),
                Bs.stride(1),
                GROUP_N=group_n,
                GROUP_K=group_k,
                BLOCK_N=block_n,
                BLOCK_K=128,
                WIDE_OFFSETS=True,
                num_warps=4,
                num_stages=2,
            )
        return C
    if not expand and group_k == 128 and ((M <= 16 and K <= 2304) or skinny_swap):
        C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
        block_n = 16 if N <= 16 else 32
        with torch_device_fn.device(A.device):
            _w8a8_block_int8_kernel_core[
                (triton.cdiv(M, 16) * triton.cdiv(N, block_n), 1)
            ](
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
                BLOCK_M=16,
                BLOCK_N=block_n,
                BLOCK_K=group_k,
                GROUP_M=8,
                SWAP=True,
                SPLITS=1,
                WIDE_OFFSETS=True,
                num_warps=4,
                num_stages=2,
            )
        return C
    if not expand and M >= 17 and K <= 2304 and group_k == 128:
        # Large INT8 tiles on unaligned rows can exceed the MACA launch limits.
        if K % 16:
            return _generic_w8a8_block_matmul(A, B, As, Bs, block_size, output_dtype)
        C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
        with torch_device_fn.device(A.device):
            _w8a8_block_int8_kernel_core[(triton.cdiv(M, 64) * triton.cdiv(N, 128), 1)](
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
                BLOCK_M=64,
                BLOCK_N=128,
                BLOCK_K=128,
                GROUP_M=8,
                SWAP=False,
                SPLITS=1,
                WIDE_OFFSETS=True,
                num_warps=4,
                num_stages=2,
            )
        return C
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    splits = 1
    swap = M <= 16 or (group_k == 128 and skinny_swap)
    if M == 1:
        if group_k != 128:
            raise NotImplementedError("MetaX GEMV currently requires block_k=128")
        with torch_device_fn.device(A.device):
            _w8a8_block_int8_gemv_kernel_tuned[
                lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
            ](
                A,
                B,
                C,
                As,
                Bs,
                N,
                K,
                A.stride(-1),
                B.stride(1),
                B.stride(0),
                As.stride(-1),
                Bs.stride(0),
                Bs.stride(1),
                GROUP_N=group_n,
                GROUP_K=group_k,
            )
        return C
    entry = _w8a8_block_int8_kernel_swap if swap else _w8a8_block_int8_kernel_general
    P = C

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
            splits,
        )

    with torch_device_fn.device(A.device):
        entry[grid](
            A,
            B,
            P,
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
            BLOCK_K=group_k,
            SPLITS=splits,
        )
    return C

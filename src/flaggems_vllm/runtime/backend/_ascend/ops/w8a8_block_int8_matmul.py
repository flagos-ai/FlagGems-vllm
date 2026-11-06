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

"""Ascend 910B W8A8 block INT8 matrix multiplication.

The inputs are already quantized INT8 tensors.  The kernel performs the
integer dot product and accumulates in INT32, then applies the activation and
weight block scales in FP32 before converting the result to the requested
output dtype.
"""

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend import (
    PIPE,
    sub_vec_id,
    sync_block_set,
    sync_block_wait,
)

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner


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
            # operand.  Load directly in [BN, BK] x [BK, BM] order so Ascend 910B
            # can test the transposed task layout without a separate output transpose.
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

    C += split * M * N
    if SWAP:
        # ``acc`` is [BLOCK_N, BLOCK_M] on the swap path.  Store it with
        # transposed pointer arithmetic so Ascend 910B does not need a separate
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
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
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
        rows = tl.arange(0, 16)
        a_matrix = tl.broadcast_to(a[None, :], (16, BLOCK_K))
        partial = tl.dot(a_matrix, b, out_dtype=tl.int32)
        partial = tl.sum(tl.where(rows[:, None] == 0, partial, 0), axis=0)
        a_scale = tl.load(As + k_tile * BLOCK_K // GROUP_K * stride_ask)
        b_scale = tl.load(
            Bs
            + (offs_n // GROUP_N) * stride_bsn
            + k_tile * BLOCK_K // GROUP_K * stride_bsk,
            mask=mask_n,
            other=0.0,
        )
        acc += partial.to(tl.float32) * a_scale * b_scale
    if C.dtype.element_ty == tl.float16:
        result = acc.to(tl.float16)
    elif C.dtype.element_ty == tl.bfloat16:
        result = acc.to(tl.bfloat16)
    else:
        result = acc
    tl.store(C + offs_n, result, mask=mask_n)


def _prune_matmul_configs(configs, named_args, **kwargs):
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_N"] <= 8192
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_matmul_ascend"),
    key=["M", "N", "K", "GROUP_N", "GROUP_K", "SPLITS"],
    strategy="default",
    warmup=1,
    rep=5,
    use_cuda_graph=False,
    flagtune_op_name="w8a8_block_int8_matmul",
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
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_swap_ascend"),
    key=["M", "N", "K", "GROUP_N", "GROUP_K", "SPLITS"],
    strategy="default",
    warmup=1,
    rep=5,
    use_cuda_graph=False,
    flagtune_op_name="w8a8_block_int8_matmul",
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
    )


def _prune_gemv_configs(configs, named_args, **kwargs):
    # Each integer dot must stay inside one independently scaled K group.
    group_k = {**named_args, **kwargs}["GROUP_K"]
    return [config for config in configs if config.kwargs["BLOCK_K"] == group_k]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_gemv_ascend"),
    key=["N", "K"],
    strategy="default",
    warmup=1,
    rep=5,
    use_cuda_graph=False,
    flagtune_op_name="w8a8_block_int8_matmul",
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
    )


# The last workspace element records the active split count.  The reduction
# consumes only initialized slices, even when LibTuner selects fewer splits.
_MAX_SPLITS = tl.constexpr(16)


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


def _prune_splitk_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    groups = triton.cdiv(args["K"], args["GROUP_K"])
    return [config for config in configs if config.kwargs["SPLITS"] <= min(groups, 16)]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_splitk_ascend"),
    key=["M", "N", "K", "GROUP_N", "GROUP_K"],
    strategy="default",
    warmup=1,
    rep=5,
    use_cuda_graph=False,
    post_hook=_splitk_post_hook,
    flagtune_op_name="w8a8_block_int8_matmul",
    flagtune_expand_op_name="w8a8_block_int8_splitk",
    prune_configs_by={"early_config_prune": _prune_splitk_configs},
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


@triton.jit
def _pad_int8_rows(
    X,
    Y,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    RP: tl.constexpr,
    KP: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWBLOCK: tl.constexpr = 8,
):
    r = tl.program_id(0) * ROWBLOCK + tl.arange(0, ROWBLOCK)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(
        X + r[:, None] * K + k[None, :], (r[:, None] < ROWS) & (k[None, :] < K), 0
    )
    tl.store(Y + r[:, None] * KP + k[None, :], v, (r[:, None] < RP) & (k[None, :] < KP))


@triton.jit
def _w8a8_block_int8_group_dots(
    A,
    B,
    P,
    MP: tl.constexpr,
    NP: tl.constexpr,
    KP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GK: tl.constexpr,
):
    with tle.scope(core_mode="cube"):
        task = tl.program_id(0)
        g = tl.program_id(1)
        pm = task // (NP // BN)
        pn = task % (NP // BN)
        ab = tle.dsa.alloc([BM, GK], tl.int8, tle.dsa.ascend.L1)
        bb = tle.dsa.alloc([BN, GK], tl.int8, tle.dsa.ascend.L1)
        ap = tl.make_block_ptr(
            A + pm * BM * KP + g * GK, (BM, GK), (KP, 1), (0, 0), (BM, GK), (1, 0)
        )
        bp = tl.make_block_ptr(
            B + pn * BN * KP + g * GK, (BN, GK), (KP, 1), (0, 0), (BN, GK), (1, 0)
        )
        tle.dsa.copy(ap, ab, [BM, GK])
        tle.dsa.copy(bp, bb, [BN, GK])
        a = tle.dsa.to_tensor(ab, writable=False)
        b = tle.dsa.to_tensor(bb, writable=False)
        dot = tl.dot(a, tl.trans(b), out_dtype=tl.int32)
        tl.store(
            P
            + (task * (KP // GK) + g) * BM * BN
            + tl.arange(0, BM)[:, None] * BN
            + tl.arange(0, BN)[None, :],
            dot,
        )


def _prune_group_reduce(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    groups = triton.next_power_of_2(args["G"])
    vector_rows = 2 if args["BN"] == 128 and args["G"] > 1 else 4
    return [
        cfg
        for cfg in configs
        if cfg.kwargs["VR"] == vector_rows
        and cfg.kwargs["VR"] <= args["BM"]
        and groups * cfg.kwargs["VR"] * args["BN"] <= 16384
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_group_reduce_ascend"),
    key=["M", "N", "G", "BM", "BN", "RG"],
    strategy="default",
    warmup=1,
    rep=5,
    use_cuda_graph=False,
    prune_configs_by={"early_config_prune": _prune_group_reduce},
)
@triton.jit
def _w8a8_block_int8_group_reduce_rows(
    P,
    As,
    Bs,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    G: tl.constexpr,
    GN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    RG: tl.constexpr,
    VR: tl.constexpr,
):
    task = tl.program_id(0)
    pm = task // (NP // BN)
    pn = task % (NP // BN)
    r = tl.program_id(1) * VR + tl.arange(0, VR)
    g = tl.arange(0, RG)
    c = tl.arange(0, BN)
    v = tl.load(
        P
        + (task * G + g[:, None, None]) * BM * BN
        + r[None, :, None] * BN
        + c[None, None, :],
        g[:, None, None] < G,
        0,
    ).to(tl.float32)
    sa = tl.load(
        As + (pm * BM + r[None, :]) * G + g[:, None],
        (g[:, None] < G) & ((pm * BM + r[None, :]) < M),
        0,
    )
    sb = tl.load(Bs + (pn * BN // GN) * G + g, (g < G) & (pn * BN < N), 0)
    res = tl.sum(v * sa[:, :, None] * sb[:, None, None], 0)
    tl.store(
        C + (pm * BM + r[:, None]) * N + pn * BN + c[None, :],
        res.to(C.dtype.element_ty),
        ((pm * BM + r[:, None]) < M) & ((pn * BN + c[None, :]) < N),
    )


def _stream_block_m(M, groups):
    if groups <= 2:
        # For large M, a wider tile amortizes per-task synchronization; a
        # single padded tail is cheaper than shrinking every task to 32 rows.
        if M >= 1024 and M % 64 != 0:
            return 128
        if M % 128 == 0:
            return 128
        if M % 64 == 0:
            return 64
        return 32
    return 128


def _prune_stream_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    bm = _stream_block_m(args["MR"], args["K"] // 128)
    return [cfg for cfg in configs if cfg.kwargs["BM"] == bm]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_stream_ascend"),
    prune_configs_by={"early_config_prune": _prune_stream_configs},
    # Padded M also identifies the host-selected tile geometry, so an old
    # cached BM32 choice cannot survive a switch to BM128.
    key=["MR", "NR", "K", "PRELOAD", "M"],
    strategy="default",
    warmup=1,
    rep=5,
    use_cuda_graph=False,
)
@triton.jit
def _w8a8_block_int8_stream(
    A,
    B,
    As,
    Bs,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    VR: tl.constexpr,
    WORKERS: tl.constexpr,
    RG: tl.constexpr,
    MR: tl.constexpr,
    NR: tl.constexpr,
    PRELOAD: tl.constexpr = True,
):
    worker = tl.program_id(0)
    ab = tle.dsa.alloc([BM, 128], tl.int8, tle.dsa.ascend.L1)
    bb = tle.dsa.alloc([BN, 128], tl.int8, tle.dsa.ascend.L1)
    accum = tle.dsa.alloc([BM // 2, BN], tl.float32, tle.dsa.ascend.UB)
    if PRELOAD:
        asbuf = tle.dsa.alloc([RG, BM // 2], tl.float32, tle.dsa.ascend.UB)
        bsbuf = tle.dsa.alloc([RG], tl.float32, tle.dsa.ascend.UB)
    scratch = P + worker * BM * BN
    for task in range(worker, (M // BM) * (N // BN), WORKERS):
        pm = task // (N // BN)
        pn = task % (N // BN)
        with tle.scope(core_mode="vector"):
            tle.dsa.to_buffer(tl.full((BM // 2, BN), 0, tl.float32), bind_buffer=accum)
            if PRELOAD:
                sub = sub_vec_id().to(tl.int32)
                rr = pm * BM + sub * (BM // 2) + tl.arange(0, BM // 2)
                gg = tl.arange(0, RG)
                if M == MR:
                    av = tl.load(
                        As + rr[:, None] * (K // 128) + gg[None, :],
                        gg[None, :] < K // 128,
                        0,
                    )
                else:
                    av = tl.load(
                        As + rr[:, None] * (K // 128) + gg[None, :],
                        (gg[None, :] < K // 128) & (rr[:, None] < MR),
                        0,
                    )
                bv = tl.load(
                    Bs + (pn * BN // 128) * (K // 128) + gg,
                    (gg < K // 128) & (pn * BN < NR),
                    0,
                )
                tle.dsa.to_buffer(tl.trans(av), bind_buffer=asbuf)
                tle.dsa.to_buffer(bv, bind_buffer=bsbuf)
        for g in range(K // 128):
            with tle.scope(core_mode="cube"):
                ap = tl.make_block_ptr(
                    A + pm * BM * K + g * 128,
                    (BM, 128),
                    (K, 1),
                    (0, 0),
                    (BM, 128),
                    (1, 0),
                )
                bp = tl.make_block_ptr(
                    B + pn * BN * K + g * 128,
                    (BN, 128),
                    (K, 1),
                    (0, 0),
                    (BN, 128),
                    (1, 0),
                )
                tle.dsa.copy(ap, ab, [BM, 128])
                tle.dsa.copy(bp, bb, [BN, 128])
                a = tle.dsa.to_tensor(ab, writable=False)
                b = tle.dsa.to_tensor(bb, writable=False)
                dot = tl.dot(a, tl.trans(b), out_dtype=tl.int32)
                tl.store(
                    scratch
                    + tl.arange(0, BM)[:, None] * BN
                    + tl.arange(0, BN)[None, :],
                    dot,
                )
            sync_block_set("cube", "vector", 0, PIPE.PIPE_FIX, PIPE.PIPE_S)
            sync_block_wait("cube", "vector", 0, PIPE.PIPE_FIX, PIPE.PIPE_S)
            with tle.scope(core_mode="vector"):
                sub = sub_vec_id().to(tl.int32)
                if PRELOAD:
                    bview = tle.dsa.subview(bsbuf, [g], [1], [1])
                    bs = tl.sum(tle.dsa.to_tensor(bview, writable=False), 0)
                else:
                    bs = tl.load(Bs + (pn * BN // 128) * (K // 128) + g)
                for r0 in range(0, BM // 2, VR):
                    r = sub * (BM // 2) + r0 + tl.arange(0, VR)
                    part = tl.load(
                        scratch + r[:, None] * BN + tl.arange(0, BN)[None, :],
                        volatile=True,
                    ).to(tl.float32)
                    if PRELOAD:
                        aview = tle.dsa.subview(asbuf, [g, r0], [1, VR], [1, 1])
                        sa = tl.reshape(tle.dsa.to_tensor(aview, writable=False), (VR,))
                    else:
                        if M == MR or (not PRELOAD and pm * BM + BM <= MR):
                            sa = tl.load(As + (pm * BM + r) * (K // 128) + g)
                        else:
                            sa = tl.load(
                                As + (pm * BM + r) * (K // 128) + g, pm * BM + r < MR, 0
                            )
                    accv = tle.dsa.subview(accum, [r0, 0], [VR, BN], [1, 1])
                    old = tle.dsa.to_tensor(accv, writable=False)
                    tle.dsa.to_buffer(old + part * sa[:, None] * bs, bind_buffer=accv)
            sync_block_set("vector", "cube", 1, PIPE.PIPE_V, PIPE.PIPE_S)
            sync_block_wait("vector", "cube", 1, PIPE.PIPE_V, PIPE.PIPE_S)
        with tle.scope(core_mode="vector"):
            sub = sub_vec_id().to(tl.int32)
            for r0 in range(0, BM // 2, VR):
                r = sub * (BM // 2) + r0 + tl.arange(0, VR)
                accv = tle.dsa.subview(accum, [r0, 0], [VR, BN], [1, 1])
                v = tle.dsa.to_tensor(accv, writable=False)
                if (M == MR and N == NR) or (
                    not PRELOAD and pm * BM + BM <= MR and pn * BN + BN <= NR
                ):
                    tl.store(
                        C
                        + (pm * BM + r[:, None]) * NR
                        + pn * BN
                        + tl.arange(0, BN)[None, :],
                        v.to(C.dtype.element_ty),
                    )
                else:
                    tl.store(
                        C
                        + (pm * BM + r[:, None]) * NR
                        + pn * BN
                        + tl.arange(0, BN)[None, :],
                        v.to(C.dtype.element_ty),
                        ((pm * BM + r[:, None]) < MR)
                        & ((pn * BN + tl.arange(0, BN)[None, :]) < NR),
                    )


def _group_tile(M, N, K):
    if M <= 16:
        return 16, 128
    if M == 64 and N >= 512 and K >= 1024:
        return 64, 128
    return 32, 64


def _pad_group_inputs(A, B, M, N, K, mp, np, kp):
    ap = A
    bp = B
    for source, rows, padded_rows, is_a in ((A, M, mp, True), (B, N, np, False)):
        if rows == padded_rows and K == kp:
            continue
        padded = torch.empty((padded_rows, kp), device=A.device, dtype=torch.int8)
        # Reinterpret copies four INT8 values as one word; no conversion or
        # dequantization. Arbitrary byte tails retain the byte-copy kernel.
        if K % 4 == 0:
            _pad_int8_rows[(triton.cdiv(padded_rows, 8), triton.cdiv(kp // 4, 1024))](
                triton.reinterpret(source, tl.int32),
                triton.reinterpret(padded, tl.int32),
                rows,
                K // 4,
                padded_rows,
                kp // 4,
                1024,
            )
        else:
            _pad_int8_rows[(triton.cdiv(padded_rows, 8), triton.cdiv(kp, 1024))](
                source,
                padded,
                rows,
                K,
                padded_rows,
                kp,
                1024,
            )
        if is_a:
            ap = padded
        else:
            bp = padded
    return ap, bp


def _grouped_matmul(A, B, As, Bs, M, N, K, output_dtype):
    # One independent INT32 Cube dot per K scale group. The packed workspace
    # makes all values/scales needed for a Vector reduction local to its tile.
    bm, bn = _group_tile(M, N, K)
    mp = triton.cdiv(M, bm) * bm
    np = triton.cdiv(N, bn) * bn
    kp = triton.cdiv(K, 128) * 128
    groups = kp // 128
    C = torch.empty(A.shape[:-1] + (N,), device=A.device, dtype=output_dtype)
    P = torch.empty((mp * np * groups,), device=A.device, dtype=torch.int32)
    with torch_device_fn.device(A.device):
        ap, bp = _pad_group_inputs(A, B, M, N, K, mp, np, kp)
        tasks = (mp // bm) * (np // bn)
        _w8a8_block_int8_group_dots[(tasks, groups)](
            ap,
            bp,
            P,
            mp,
            np,
            kp,
            bm,
            bn,
            128,
            num_warps=4,
            num_stages=1,
        )
        _w8a8_block_int8_group_reduce_rows[
            lambda meta: (tasks, triton.cdiv(min(M, bm), meta["VR"]))
        ](
            P,
            As,
            Bs,
            C,
            M,
            N,
            np,
            groups,
            128,
            bm,
            bn,
            triton.next_power_of_2(groups),
        )
    return C


def _stream_matmul(A, B, As, Bs, M, N, K, output_dtype):
    # Bounded tiles keep per-core UB state and the 20-worker scratch small.
    # Both matrix operands remain INT8; only partials/scales use Vector FP32.
    bm = _stream_block_m(M, triton.cdiv(K, 128))
    mp = triton.cdiv(M, bm) * bm
    np = triton.cdiv(N, 128) * 128
    kp = triton.cdiv(K, 128) * 128
    C = torch.empty(A.shape[:-1] + (N,), device=A.device, dtype=output_dtype)
    P = torch.empty((20 * bm * 128,), device=A.device, dtype=torch.int32)
    with torch_device_fn.device(A.device):
        ap, bp = _pad_group_inputs(A, B, M, N, K, mp, np, kp)
        _w8a8_block_int8_stream[(20,)](
            ap,
            bp,
            As,
            Bs,
            P,
            C,
            mp,
            np,
            kp,
            WORKERS=20,
            RG=triton.next_power_of_2(kp // 128),
            MR=M,
            NR=N,
            PRELOAD=kp // 128 >= 4,
        )
    return C


def w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Block-scaled INT8 A @ B.T; inputs are already quantized.

    Every K group is accumulated in INT32, scaled and accumulated in FP32.
    Split-K partials also remain FP32 until the final reduction.
    """
    if A.ndim < 2 or B.ndim != 2 or As.ndim != A.ndim or Bs.ndim != 2:
        raise ValueError("A/As must have rank >= 2, B/Bs rank 2")
    if A.device.type != "npu" or any(t.device != A.device for t in (B, As, Bs)):
        raise ValueError("all inputs must be on the same Ascend device")
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise NotImplementedError("A and B must be int8")
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        raise NotImplementedError("scales must be float32")
    if any(t.requires_grad for t in (A, B, As, Bs)):
        raise NotImplementedError("w8a8_block_int8_matmul is inference-only")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("output_dtype must be float16, bfloat16 or float32")
    if len(block_size) != 2 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain two positive integers")
    group_n, group_k = block_size
    if group_k not in (32, 64, 128, 256):
        raise NotImplementedError("block_k must be 32, 64, 128 or 256")
    if not all(t.is_contiguous() for t in (A, B, As, Bs)):
        raise NotImplementedError("inputs must be contiguous")
    K = A.shape[-1]
    M = 1
    for size in A.shape[:-1]:
        M *= size
    N = B.shape[0]
    groups = triton.cdiv(K, group_k)
    if B.shape[-1] != K:
        raise ValueError("A and B must have the same K dimension")
    if As.shape != A.shape[:-1] + (groups,):
        raise ValueError("As has an invalid block-scale shape")
    if Bs.shape != (triton.cdiv(N, group_n), groups):
        raise ValueError("Bs has an invalid block-scale shape")
    # General kernels mask tails; the group-parallel path zero-pads its input.
    # An empty K runs zero iterations and writes 0.
    if M == 0 or N == 0:
        return torch.empty(A.shape[:-1] + (N,), device=A.device, dtype=output_dtype)
    # Narrow N, long K otherwise misses both TLE paths. Small M benefits
    # from group parallelism (at most 24 MiB of partials here); larger M
    # amortizes the persistent path while keeping its scratch bounded.
    if group_n == 128 and group_k == 128 and N == 128 and M > 128 and 4096 <= K <= 8192:
        if M <= 768:
            return _grouped_matmul(A, B, As, Bs, M, N, K, output_dtype)
        return _stream_matmul(A, B, As, Bs, M, N, K, output_dtype)
    # A persistent mixed Cube/Vector path bounds scratch independently of M
    # and K groups. Short K avoids a separate partial-reduction launch; narrow
    # long-K workloads keep the more parallel group path.
    if (
        group_n == 128
        and group_k == 128
        and M >= 32
        and N >= 512
        and 1 < groups <= 64
        and (groups <= 2 or (M >= 65 and (M > 128 or N >= 1024)))
    ):
        return _stream_matmul(A, B, As, Bs, M, N, K, output_dtype)
    # TLE group-parallel path is bounded to avoid a large INT32 workspace.
    # Other scale groups / large workloads retain the existing tuned kernels.
    bm, bn = _group_tile(M, N, K)
    workspace_bytes = 4 * triton.cdiv(M, bm) * bm * triton.cdiv(N, bn) * bn * groups
    if (
        group_n == 128
        and group_k == 128
        and 1 <= M <= 128
        and (M > 1 or (N >= 256 and K >= 1024))
        and 0 < groups <= 64
        and workspace_bytes <= 128 * 1024 * 1024
    ):
        return _grouped_matmul(A, B, As, Bs, M, N, K, output_dtype)
    # Keep temporary storage bounded. Split selection is tuned together with
    # its reduction via the post-hook, rather than timing partial GEMM alone.
    if K >= 4096 and M <= 128 and N <= 1536 and M * N <= 65536:
        return _splitk_matmul(A, B, As, Bs, M, N, K, group_n, group_k, output_dtype)
    C = torch.empty(A.shape[:-1] + (N,), device=A.device, dtype=output_dtype)
    with torch_device_fn.device(A.device):
        # The shared GEMV tuning key is (N, K). Keep its K group fixed;
        # other group sizes use the fully keyed small-M path.
        if M == 1 and group_k == 128:
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
                1,
                1,
                K,
                1,
                groups,
                1,
                GROUP_N=group_n,
                GROUP_K=group_k,
            )
        else:
            entry = (
                _w8a8_block_int8_kernel_swap
                if M <= 16
                else _w8a8_block_int8_kernel_general
            )
            entry[
                lambda meta: (
                    triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
                    1,
                )
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
                BLOCK_K=group_k,
                SPLITS=1,
            )
    return C

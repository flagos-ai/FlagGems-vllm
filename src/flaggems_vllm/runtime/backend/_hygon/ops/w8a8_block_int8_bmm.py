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
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@triton.jit
def _scale_flags_or(left, right):
    return left | right


# BM, BN, Split-K, warps, combined-exponent validation.
_H1_TUNED_INTEGER_CONFIGS = {
    56: (16, 32, 5, 2, False),
    64: (16, 32, 4, 2, False),
    72: (16, 64, 4, 2, False),
    80: (16, 64, 4, 2, False),
    112: (64, 64, 4, 4, True),
    136: (32, 64, 4, 2, False),
    144: (32, 64, 4, 2, False),
    160: (32, 64, 4, 2, False),
    400: (64, 64, 2, 4, False),
    416: (64, 64, 2, 4, False),
    448: (64, 64, 2, 4, True),
}


@lru_cache(maxsize=16)
def _single_head_split_workspace(device, raw_stream):
    # Independent streams need separate partials and self-resetting counters.
    partial = torch.empty((2, 80, 64, 64), dtype=torch.int32, device=device)
    counter = torch.zeros((80,), dtype=torch.int32, device=device)
    return partial, counter


@lru_cache(maxsize=16)
def _single_head_uneven_split_workspace(device, raw_stream):
    # 128 counters also cover BN=32 tiles; partial storage stays at 5 MiB.
    partial = torch.empty((5, 64, 64, 64), dtype=torch.int32, device=device)
    counter = torch.zeros((128,), dtype=torch.int32, device=device)
    return partial, counter


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=[
        "log",
        "log",
        "log",
        "align32",
        "align32",
    ],
    flagtune_op_name="bmm",
    flagtune_expand_op_name="bmm",
    flagtune_pre_hook=None,
)
@triton.heuristics(runtime.get_heuristic_config("bmm"))
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    IS_FP64: tl.constexpr = False,
    USE_INT64: tl.constexpr = False,
):
    # batch offsets
    pid_b = tl.program_id(2)
    if USE_INT64:
        pid_b = pid_b.to(tl.int64)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = tl.num_programs(0)
        gridy = tl.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    if USE_INT64:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)
        offs_k = offs_k.to(tl.int64)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    num_iters = tl.cdiv(K, TILE_K)
    if IS_FP64:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float64)
    else:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask_a)
        b = tl.load(b_ptrs, mask_b)

        offs_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)


def _bmm_requires_int64(*tensors):
    return any(
        sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) >= 2**31
        for t in tensors
    )


def bmm(
    A, B, *, a_scale=None, b_scale=None, block_size=(128, 128, 128), out_dtype=None
):
    if A.dtype == torch.int8:
        return _int8_block_bmm(
            A,
            B,
            a_scale,
            b_scale,
            block_size=block_size,
            out_dtype=out_dtype or torch.bfloat16,
        )
    logger.debug("GEMS_HYGON BMM")
    assert A.shape[0] == B.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](
            A,
            B,
            out,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            IS_FP64=A.dtype == torch.float64,
            USE_INT64=_bmm_requires_int64(A, B, out),
        )
    return out


def bmm_out(A, B, out, *, a_scale=None, b_scale=None, block_size=(128, 128, 128)):
    if A.dtype == torch.int8:
        return _int8_block_bmm(
            A, B, a_scale, b_scale, block_size=block_size, out_dtype=out.dtype, out=out
        )
    logger.debug("GEMS_HYGON BMM_OUT")
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](
            A,
            B,
            out,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            IS_FP64=A.dtype == torch.float64,
            USE_INT64=_bmm_requires_int64(A, B, out),
        )
    return out


# Hygon DCU, torch (ROCm, HIP 6.1.25065) / Triton 3.6.0, 2026-09-18.
# Values: BLOCK_M, BLOCK_N, num_warps, num_stages, GROUP_M, ACC_FP32, SWAP_AB,
# optionally TILE_K (defaults to the quantization block_k) and UNROLL
# (K-loop unroll factor for K > 256, defaults to 1).
EXACT_CONFIGS = {
    (8, 1, 1024, 4096): (16, 128, 8, 1, 8, True, False, 128, 2),
    (8, 4, 1024, 4096): (16, 128, 4, 2, 8, True, False, 128, 2),
    (8, 8, 1024, 4096): (16, 128, 8, 1, 8, True, False, 128, 4),
    (8, 16, 1024, 4096): (32, 128, 4, 2, 8, True, False, 128, 2),
    (8, 32, 1024, 4096): (32, 64, 4, 2, 8, True, False),
    (8, 64, 1024, 4096): (64, 128, 4, 2, 8, True, False, 128, 2),
    (8, 128, 1024, 4096): (64, 128, 4, 2, 8, True, False, 128, 2),
    (8, 4096, 1024, 4096): (128, 128, 8, 1, 8, True, False),
    (8, 8192, 1024, 4096): (128, 128, 8, 1, 8, True, False),
    (8, 16384, 1024, 4096): (128, 128, 8, 1, 8, True, False),
    (8, 32768, 1024, 4096): (128, 128, 8, 1, 8, True, False),
    (16, 1, 1024, 7168): (16, 128, 4, 1, 1, False, False),
    (16, 4, 1024, 7168): (16, 128, 4, 1, 1, False, False),
    (16, 8, 1024, 7168): (16, 128, 4, 1, 1, False, False),
    (16, 16, 1024, 7168): (16, 128, 4, 3, 1, False, False),
    (16, 32, 1024, 7168): (32, 128, 4, 2, 1, False, False),
    (16, 64, 1024, 7168): (64, 128, 4, 2, 8, True, False),
    (16, 128, 1024, 7168): (128, 128, 16, 2, 8, True, False),
    (16, 4096, 1024, 7168): (128, 128, 8, 1, 8, True, False),
    (16, 8192, 1024, 7168): (128, 128, 8, 1, 8, True, False),
    (16, 16384, 1024, 7168): (128, 128, 8, 1, 8, True, False),
    (16, 32768, 1024, 7168): (128, 128, 8, 1, 8, True, False),
}


# Four-scale loads with 64x64 output tiles. These row group sizes were
# measured with contiguous K groups, and both scale layouts for B=4100.
_SINGLE_HEAD_PACK_SCALE_GROUPS = {
    512: 8,
    1028: 1,
    3076: 1,
    3805: 1,
    4096: 1,
    4100: 1,
    13021: 1,
}


def _get_int8_config(batch, m, n, k, scale_n=128, scale_m=1, scale_k_contiguous=False):
    key = (batch, m, n, k)
    if (batch, n, k, scale_m, scale_n) == (1, 1024, 4096, 1, 128):
        pack_group = (
            _SINGLE_HEAD_PACK_SCALE_GROUPS.get(m)
            if scale_k_contiguous or m == 4100
            else None
        )
        pack_scales = pack_group is not None
        scale_groups = 4
        num_stages = 1
        unroll = 4 if m <= 96 else (2 if m <= 512 else 1)
        kpack = 2
        if 16 < m <= 72 and scale_k_contiguous:
            bm, bn, warps, group = 16, 32, 2, (1 if 32 <= m <= 64 else 4)
            num_stages, pack_scales = 2, True
            if 32 < m <= 48 and scale_k_contiguous:
                bn, group, kpack = 16, 1, 1
        elif m in (496, 512) and scale_k_contiguous:
            bm, bn, warps, group = 64, 64, 4, 4
            num_stages, pack_scales, scale_groups = 2, True, 2
        elif pack_group is not None:
            bm, bn, warps, group = 64, 64, 4, pack_group
        elif m <= 96:
            bm, bn, warps, group = 32, 16, 2, 1
        elif m <= 128:
            bm, bn, warps, group = 16, 32, 2, 1
        elif m <= 224:
            bm, bn, warps, group = 16, 64, 2, 1
            num_stages, unroll = 2, 1
        elif m <= 480 and scale_k_contiguous:
            bm, bn, warps, group = 16, 64, 2, 1
            pack_scales = True
        elif m <= 256:
            bm, bn, warps, group = 16, 64, 2, 1
        elif m <= 384:
            bm, bn, warps, group = 32, 64, 4, 4
        elif m <= 512:
            bm, bn, warps, group = 64, 32, 2, 1
        elif m < 2048:
            bm, bn, warps, group = 64, 128, 4, 4
        else:
            bm, bn, warps, group = 128, 128, 4, 4
        return dict(
            BLOCK_M=bm,
            BLOCK_N=bn,
            num_warps=warps,
            num_stages=num_stages,
            GROUP_M=group,
            ACC_FP32=True,
            TILE_K=128,
            UNROLL=unroll,
            PACK_SCALES=pack_scales,
            SCALE_GROUPS=scale_groups,
            kpack=kpack,
        )
    acc_fp32, swap_ab, tile_k, unroll = False, False, None, 1
    if key in EXACT_CONFIGS:
        values = EXACT_CONFIGS[key]
        bm, bn, warps, stages, group, acc_fp32, swap_ab, tile_k, unroll = (
            *values,
            None,
            1,
        )[:9]
    elif n == 1024 and (batch, k) in ((8, 4096), (16, 7168)):
        if m <= 64:
            bm, bn, warps, stages, group = 32, 128, 4, 2, 8
            acc_fp32 = (batch, k) == (8, 4096)
            unroll = 2 if (batch, k) == (8, 4096) else 1
        else:
            bm, bn, warps, stages, group = 128, 128, 8, 1, 8
    elif m <= 64:
        bm, bn, warps, stages, group = 32, 64, 4, 2, 1
    elif k >= 4096:
        bm, bn, warps, stages, group = 128, 128, 8, 1, 8
    else:
        bm, bn, warps, stages, group = 64, 64, 4, 2, 4
    result = dict(
        BLOCK_M=bm,
        BLOCK_N=min(bn, scale_n),
        num_warps=warps,
        num_stages=stages,
        GROUP_M=group,
        ACC_FP32=acc_fp32,
        SWAP_AB=swap_ab,
        TILE_K=tile_k,
        UNROLL=unroll,
    )
    return result


@libentry()
@triton.jit
def small_bmm_kernel(
    A,
    W,
    AS,
    WS,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA: tl.constexpr,
    SW: tl.constexpr,
    SAS: tl.constexpr,
    SWS: tl.constexpr,
    SO: tl.constexpr,
    SM: tl.constexpr,
    SN: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    row = pid // tl.cdiv(N, BN)
    cols = (pid % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    a = tl.load(A + batch * SA[0] + row * SA[1] + kk * SA[2], kk < K, other=0).to(
        tl.int32
    )
    w = tl.load(
        W + batch * SW[0] + kk[:, None] * SW[1] + cols[None, :] * SW[2],
        (kk[:, None] < K) & (cols[None, :] < N),
        other=0,
    ).to(tl.int32)
    partial = tl.sum(a[:, None] * w, 0).to(tl.float32)
    asc = tl.load(AS + batch * SAS[0] + (row // SM) * SAS[1]).to(tl.float32)
    wsc = tl.load(WS + batch * SWS[0] + (cols // SN) * SWS[2], cols < N, other=0).to(
        tl.float32
    )
    result = partial * (asc * wsc)
    tl.store(O + batch * SO[0] + row * SO[1] + cols * SO[2], result, cols < N)


@triton.jit
def _int8_block_bmm_kernel(
    A,
    W,
    AS,
    WS,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA: tl.constexpr,
    SW: tl.constexpr,
    SAS: tl.constexpr,
    SWS: tl.constexpr,
    SO: tl.constexpr,
    SCALE_M: tl.constexpr,
    SCALE_N: tl.constexpr,
    SCALE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALIGNED: tl.constexpr,
    TILE_K: tl.constexpr,
    UNROLL: tl.constexpr = 1,
    ACC_FP32: tl.constexpr = False,
    SWAP_AB: tl.constexpr = False,
    PACK_SCALES: tl.constexpr = False,
    SCALE_GROUPS: tl.constexpr = 4,
):
    batch = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BLOCK_M)
    nn = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    size_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    pm = first_m + local % size_m
    pn = local // size_m
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, TILE_K)
    if (
        max(M * SA[1] + K * SA[2], K * SW[1] + N * SW[2], M * SO[1] + N * SO[2])
        >= 2**31
    ):
        rm = rm.to(tl.int64)
        rn = rn.to(tl.int64)
        rk = rk.to(tl.int64)
    acc_dtype: tl.constexpr = tl.float32 if ACC_FP32 else tl.bfloat16
    if SWAP_AB:
        acc = tl.zeros((BLOCK_N, BLOCK_M), acc_dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), acc_dtype)
    k_tiles: tl.constexpr = (K + TILE_K - 1) // TILE_K
    if PACK_SCALES:
        tl.static_assert(SCALE_GROUPS == 2 or SCALE_GROUPS == 4)
        tl.static_assert(
            SCALE_M == 1 and TILE_K == SCALE_K and K % (SCALE_GROUPS * TILE_K) == 0
        )
        large_scale_offsets: tl.constexpr = (
            max(
                M * SAS[1] + k_tiles * SAS[2],
                k_tiles * SWS[1] + (N // SCALE_N) * SWS[2],
            )
            >= 2**31
        )
        scale_row = rm
        scale_col = pn * BLOCK_N // SCALE_N
        if large_scale_offsets:
            scale_row = scale_row.to(tl.int64)
            scale_col = scale_col.to(tl.int64)
        for first_k in tl.range(0, k_tiles, SCALE_GROUPS):
            # Fetch adjacent scales together while keeping each dot independent.
            groups = first_k + tl.arange(0, SCALE_GROUPS)
            if large_scale_offsets:
                groups = groups.to(tl.int64)
            scale_rows = (
                AS
                + batch * SAS[0]
                + scale_row[:, None] * SAS[1]
                + groups[None, :] * SAS[2]
            )
            scale_cols = WS + batch * SWS[0] + scale_col * SWS[2] + groups * SWS[1]
            scale_rows = tl.load(scale_rows, rm[:, None] < M, 0).to(tl.float32)
            scale_cols = tl.load(scale_cols).to(tl.float32)
            if SCALE_GROUPS == 2:
                scale0, scale1 = tl.split(scale_rows * scale_cols[None, :])
            else:
                even, odd = tl.split(
                    tl.reshape(scale_rows * scale_cols[None, :], (BLOCK_M, 2, 2))
                )
                scale0, scale2 = tl.split(even)
                scale1, scale3 = tl.split(odd)
            for part in tl.static_range(SCALE_GROUPS):
                kb = first_k + part
                if part == 0:
                    row_scale = scale0
                elif part == 1:
                    row_scale = scale1
                elif part == 2:
                    row_scale = scale2
                else:
                    row_scale = scale3
                kk = kb * TILE_K + rk
                ap = A + batch * SA[0] + rm[:, None] * SA[1] + kk[None, :] * SA[2]
                wp = W + batch * SW[0] + kk[:, None] * SW[1] + rn[None, :] * SW[2]
                a = tl.load(ap, (rm[:, None] < M) & (kk[None, :] < K), other=0)
                w = tl.load(wp, (kk[:, None] < K) & (rn[None, :] < N), other=0)
                partial = tl.dot(a, w, out_dtype=tl.int32).to(tl.float32)
                acc = tl.fma(partial, row_scale[:, None], acc)
    else:
        for kb in tl.range(
            0,
            k_tiles,
            loop_unroll_factor=(
                (K + TILE_K - 1) // TILE_K if K > 0 and K <= 256 else UNROLL
            ),
        ):
            kk = kb * TILE_K + rk
            scale_k = kb * TILE_K // SCALE_K
            ap = A + batch * SA[0] + rm[:, None] * SA[1] + kk[None, :] * SA[2]
            wp = W + batch * SW[0] + kk[:, None] * SW[1] + rn[None, :] * SW[2]
            if K <= 256 and SCALE_M >= BLOCK_M and SCALE_M % BLOCK_M == 0:
                asp = (
                    AS
                    + batch * SAS[0]
                    + (pm * BLOCK_M // SCALE_M) * SAS[1]
                    + scale_k * SAS[2]
                )
            else:
                asp = AS + batch * SAS[0] + (rm // SCALE_M) * SAS[1] + scale_k * SAS[2]
            # BLOCK_N divides SCALE_N: all columns of the tile share one weight scale.
            wsp = (
                WS
                + batch * SWS[0]
                + scale_k * SWS[1]
                + (pn * BLOCK_N // SCALE_N) * SWS[2]
            )
            if ALIGNED:
                a = tl.load(ap)
                w = tl.load(wp)
                a_scale = tl.load(asp)
            else:
                a = tl.load(ap, (rm[:, None] < M) & (kk[None, :] < K), other=0)
                w = tl.load(wp, (kk[:, None] < K) & (rn[None, :] < N), other=0)
                if K <= 256 and SCALE_M >= BLOCK_M and SCALE_M % BLOCK_M == 0:
                    a_scale = tl.load(asp)
                else:
                    a_scale = tl.load(asp, rm < M, other=0)
            w_scale = tl.load(wsp)
            row_scale = (a_scale.to(tl.float32) * w_scale.to(tl.float32)).to(acc_dtype)
            # Swap small-M contractions so M occupies the narrower dot dimension.
            if SWAP_AB:
                partial = tl.dot(tl.trans(w), tl.trans(a), out_dtype=tl.int32).to(
                    acc_dtype
                )
            else:
                partial = tl.dot(a, w, out_dtype=tl.int32).to(acc_dtype)
            if K <= 256 and SCALE_M >= BLOCK_M and SCALE_M % BLOCK_M == 0:
                scale = row_scale
            elif SWAP_AB:
                scale = row_scale[None, :]
            else:
                scale = row_scale[:, None]
            if ACC_FP32:
                acc = tl.fma(partial, scale, acc)
            else:
                dequant = (partial * scale).to(tl.bfloat16)
                if K <= TILE_K:
                    acc = dequant
                else:
                    acc = (acc + dequant).to(tl.bfloat16)
    if SWAP_AB:
        acc = tl.trans(acc)
    op = O + batch * SO[0] + rm[:, None] * SO[1] + rn[None, :] * SO[2]
    if ALIGNED:
        tl.store(op, acc)
    else:
        tl.store(op, acc, (rm[:, None] < M) & (rn[None, :] < N))


# A summed scale-exponent span of at most three bounds the K=4096 INT8
# reduction by 4096 * 128**2 * 8 == 2**29, enabling exact integer accumulation
# until the final FP32 conversion. Four relative shifts fit in one packed word.
@triton.jit
def _pack_scale_rows(
    S,
    INFO,
    P,
    COUNT: tl.constexpr,
    RS: tl.constexpr,
    KS: tl.constexpr,
    BASE: tl.constexpr,
    START,
    BM: tl.constexpr,
):
    rows = START + tl.arange(0, BM)
    kg = tl.arange(0, 32)
    scale = tl.load(
        S + rows[:, None] * RS + kg[None, :] * KS, rows[:, None] < COUNT, other=1.0
    )
    bits = scale.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 255
    emin = tl.min(exp, 1)
    emax = tl.max(exp, 1)
    ok = tl.min(((bits & -2139095041) == 0) & (exp > 0) & (exp < 255), 1) == 1
    info = tl.where(ok, emin | ((emax - emin) << 8), -1)
    diff = tl.reshape(exp - emin[:, None], (BM, 8, 4)).to(tl.uint32)
    packed = tl.sum(diff << (tl.arange(0, 4)[None, None, :] * 8), 2)
    tl.store(INFO + BASE + rows, info, rows < COUNT)
    tl.store(
        P + (BASE + rows[:, None]) * 8 + tl.arange(0, 8)[None, :],
        packed,
        rows[:, None] < COUNT,
    )


@triton.jit
def prepare_scale_factors(
    XS,
    WS,
    INFO,
    P,
    M: tl.constexpr,
    XSM: tl.constexpr,
    XSK: tl.constexpr,
    WSN: tl.constexpr,
    WSK: tl.constexpr,
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < tl.cdiv(M, BM):
        _pack_scale_rows(XS, INFO, P, M, XSM, XSK, 0, pid * BM, BM)
    else:
        _pack_scale_rows(WS, INFO, P, 8, WSN, WSK, M, 0, 8)


@triton.jit
def _valid_scale_factors(xi, wi):
    low = (xi & 255) + (wi & 255) - 127
    span = ((xi >> 8) & 255) + ((wi >> 8) & 255)
    ok = (xi >= 0) & (wi >= 0) & (span <= 3) & (low >= 63) & (low + span <= 159)
    return ok, low


@triton.jit
def _integer_factor_body(
    A,
    W,
    P,
    O,
    rm,
    rn,
    emin,
    M: tl.constexpr,
    PN,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ALIGNED: tl.constexpr,
    INNER: tl.constexpr,
    IST: tl.constexpr,
    WORD0=None,
    WORD1=None,
    INLINE_EXPONENTS: tl.constexpr = False,
):
    rk = tl.arange(0, 128)
    # Keep activation rows on the dot's column axis to reduce register
    # pressure in the broadcast shift-add and improve the MMAC feed layout.
    acc = tl.zeros((BN, BM), tl.int32)
    for outer in range(8):
        if INLINE_EXPONENTS:
            words = tl.where(outer < 4, WORD0, WORD1) >> ((outer % 4) * 8)
        else:
            xp = tl.load(P + rm * 8 + outer, rm < M, other=0).to(tl.uint32)
            wp = tl.load(P + (M + PN) * 8 + outer).to(tl.uint32)
            words = xp + wp
        for sub in tl.range(4, loop_unroll_factor=INNER, num_stages=IST):
            kk = (outer * 4 + sub) * 128 + rk
            a = tl.load(W + rn[:, None] * 4096 + kk[None, :])
            if ALIGNED:
                w = tl.load(A + rm[None, :] * 4096 + kk[:, None])
            else:
                w = tl.load(
                    A + rm[None, :] * 4096 + kk[:, None], rm[None, :] < M, other=0
                )
            part = tl.dot(a, w, out_dtype=tl.int32)
            shifts = words & 3 if INLINE_EXPONENTS else words
            acc = tl.inline_asm_elementwise(
                "v_lshl_add_u32 $0, $1, $2, $3",
                "=v,v,v,v",
                [part, shifts[None, :], acc],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            words = words >> (2 if INLINE_EXPONENTS else 8)
    if INLINE_EXPONENTS:
        scale = (emin << 23).to(tl.float32, bitcast=True)
        value = acc.to(tl.float32) * scale[None, :]
    else:
        value = acc.to(tl.float32) * tl.exp2(emin.to(tl.float32) - 127.0)[None, :]
    if ALIGNED:
        tl.store(O + rm[None, :] * 1024 + rn[:, None], value)
    else:
        tl.store(O + rm[None, :] * 1024 + rn[:, None], value, rm[None, :] < M)


@triton.jit
def integer_factor_bmm_kernel(
    A,
    W,
    XS,
    WS,
    INFO,
    P,
    O,
    M: tl.constexpr,
    XSM: tl.constexpr,
    XSK: tl.constexpr,
    WSN: tl.constexpr,
    WSK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GM: tl.constexpr,
    INNER: tl.constexpr,
    IST: tl.constexpr,
):
    tl.static_assert(BM == 128 and BN == 128 and (GM == 4 or GM == 8))
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = 1024 // BN
    g = pid // (GM * nn)
    first = g * GM
    size = tl.minimum(nm - first, GM)
    loc = pid % (GM * nn)
    pm = first + loc % size
    pn = loc // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    xi = tl.load(INFO + rm, rm < M, other=127)
    wi = tl.load(INFO + M + pn * BN // 128)
    valid, emin = _valid_scale_factors(xi, wi)
    if tl.min(valid | (rm >= M), 0) == 1:
        if M % BM == 0:
            _integer_factor_body(
                A, W, P, O, rm, rn, emin, M, pn * BN // 128, BM, BN, True, INNER, IST
            )
        else:
            _integer_factor_body(
                A, W, P, O, rm, rn, emin, M, pn * BN // 128, BM, BN, False, INNER, IST
            )
    else:
        _int8_block_bmm_kernel(
            A,
            W,
            XS,
            WS,
            O,
            M,
            1024,
            4096,
            (0, 4096, 1),
            (0, 1, 4096),
            (0, XSM, XSK),
            (0, WSK, WSN),
            (0, 1024, 1),
            1,
            128,
            128,
            BLOCK_M=BM,
            BLOCK_N=BN,
            GROUP_M=GM,
            ALIGNED=M % BM == 0,
            TILE_K=128,
            UNROLL=1,
            ACC_FP32=True,
        )


@triton.jit
def _single_head_integer_body(
    A,
    W,
    Out,
    rm,
    rn,
    base,
    word0,
    word1,
    M: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
    Partial=None,
    Counter=None,
    SPLIT_K: tl.constexpr = 1,
    A0=None,
    W0=None,
    TUNED: tl.constexpr = False,
    CLAMP_A: tl.constexpr = False,
):
    safe_rm = rm if M % BM == 0 else tl.minimum(rm, M - 1)
    acc = tl.zeros((BM, BN), tl.int32)
    rk = tl.arange(0, 128)
    split = tl.program_id(1) if SPLIT_K > 1 else 0
    if SPLIT_K > 2 or TUNED:
        first = split * 32 // SPLIT_K
        end = (split + 1) * 32 // SPLIT_K
        words = (word0.to(tl.uint64) | (word1.to(tl.uint64) << 32)) >> (first * 2)
        if TUNED and SPLIT_K > 1:
            words = words.to(tl.uint32)
        part = tl.dot(A0, W0, out_dtype=tl.int32)
        acc = tl.inline_asm_elementwise(
            "v_lshl_add_u32 $0, $1, $2, $3",
            "=v,v,v,v",
            [part, (words & 3).to(tl.uint32)[:, None], acc],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        words = words >> 2
        for step in tl.range(1, end - first, num_stages=STAGES):
            kk = (first + step) * 128 + rk
            left = tl.load(A + safe_rm[:, None] * 4096 + kk[None, :])
            right = tl.load(W + rn[None, :] * 4096 + kk[:, None])
            part = tl.dot(left, right, out_dtype=tl.int32)
            acc = tl.inline_asm_elementwise(
                "v_lshl_add_u32 $0, $1, $2, $3",
                "=v,v,v,v",
                [part, (words & 3).to(tl.uint32)[:, None], acc],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            words = words >> 2
    else:
        for outer in range(2 // SPLIT_K):
            half = split * (2 // SPLIT_K) + outer
            words = tl.where(half == 0, word0, word1)
            for sub in tl.range(16, num_stages=STAGES):
                kk = (half * 16 + sub) * 128 + rk
                shift = words & 3
                if M <= 320 or CLAMP_A:
                    left = tl.load(A + safe_rm[:, None] * 4096 + kk[None, :])
                else:
                    left = tl.load(
                        A + rm[:, None] * 4096 + kk[None, :], rm[:, None] < M, 0
                    )
                right = tl.load(W + rn[None, :] * 4096 + kk[:, None])
                part = tl.dot(left, right, out_dtype=tl.int32)
                acc = tl.inline_asm_elementwise(
                    "v_lshl_add_u32 $0, $1, $2, $3",
                    "=v,v,v,v",
                    [part, shift[:, None], acc],
                    dtype=tl.int32,
                    is_pure=True,
                    pack=1,
                )
                words = words >> 2
    if SPLIT_K > 1:
        tile = tl.program_id(0)
        offset = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        tl.store(Partial + (tile * SPLIT_K + split) * (BM * BN) + offset, acc)
        tl.debug_barrier()
        ticket = tl.atomic_add(Counter + tile, 1, sem="acq_rel")
        if ticket == SPLIT_K - 1:
            total = tl.zeros((BM, BN), tl.int32)
            for other in tl.static_range(SPLIT_K):
                total += tl.load(
                    Partial + (tile * SPLIT_K + other) * (BM * BN) + offset
                )
            scale = ((base - 127) << 23).to(tl.float32, bitcast=True)
            value = total.to(tl.float32) * scale[:, None]
            tl.store(Out + rm[:, None] * 1024 + rn[None, :], value, rm[:, None] < M)
            tl.atomic_xchg(Counter + tile, 0, sem="release")
    else:
        scale = ((base - 127) << 23).to(tl.float32, bitcast=True)
        value = acc.to(tl.float32) * scale[:, None]
        tl.store(Out + rm[:, None] * 1024 + rn[None, :], value, rm[:, None] < M)


@triton.jit
def int8_single_head_integer_kernel(
    A,
    W,
    XS,
    WS,
    Out,
    M: tl.constexpr,
    XSM: tl.constexpr,
    XSK: tl.constexpr,
    WSN: tl.constexpr,
    WSK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GM: tl.constexpr,
    STAGES: tl.constexpr,
    Partial=None,
    Counter=None,
    SPLIT_K: tl.constexpr = 1,
    PATH: tl.constexpr = False,
):
    # Keep one specialization argument for legacy and tuned integer paths.
    # 0: legacy; 1: swapped; 2/3: prefetched; 4: fused sum; 5: clamped A.
    TUNED: tl.constexpr = PATH == 2 or PATH == 3
    CLAMP_SCALES: tl.constexpr = TUNED
    COMBINED_EXP: tl.constexpr = PATH == 3 or PATH == 4
    SWAP_AB: tl.constexpr = PATH == 1 or PATH == 4
    tl.static_assert(BN <= 128 and 128 % BN == 0)
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = 1024 // BN
    group_id = pid // (GM * nn)
    first_m = group_id * GM
    size_m = tl.minimum(nm - first_m, GM)
    local = pid % (GM * nn)
    pm = first_m + local % size_m
    pn = local // size_m
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    kg = tl.arange(0, 32)
    if CLAMP_SCALES:
        safe_rows = rm if M % BM == 0 else tl.minimum(rm, M - 1)
        x = tl.load(XS + safe_rows[:, None] * XSM + kg[None, :] * XSK)
    else:
        x = tl.load(
            XS + rm[:, None] * XSM + kg[None, :] * XSK, rm[:, None] < M, other=1.0
        )
    w = tl.load(WS + (pn * BN // 128) * WSN + kg * WSK)
    if SPLIT_K > 2 or TUNED:
        first_k = (tl.program_id(1) * 32 // SPLIT_K) * 128 + tl.arange(0, 128)
        safe = tl.minimum(rm, M - 1)
        a0 = tl.load(A + safe[:, None] * 4096 + first_k[None, :])
        w0 = tl.load(W + rn[None, :] * 4096 + first_k[:, None])
    else:
        a0 = None
        w0 = None
    xb = x.to(tl.int32, bitcast=True)
    wb = w.to(tl.int32, bitcast=True)
    xe = (xb >> 23) & 255
    we = (wb >> 23) & 255
    if COMBINED_EXP:
        combined_exp = xe + we[None, :]
        base = tl.min(combined_exp, 1)
        maximum = tl.max(combined_exp, 1)
        xbad = (xb.to(tl.uint32) & 0x807FFFFF) | tl.where(
            (xe == 0) | (xe == 255), 0x80000000, 0
        )
        wbad = (wb.to(tl.uint32) & 0x807FFFFF) | tl.where(
            (we == 0) | (we == 255), 0x80000000, 0
        )
        xflags = tl.reduce(xbad, 1, _scale_flags_or)
        wflags = tl.reduce(wbad, 0, _scale_flags_or)
        valid = (
            (xflags == 0)
            & (wflags == 0)
            & (maximum - base <= 3)
            & (base >= 190)
            & (maximum <= 286)
        )
    else:
        xmin = tl.min(xe, 1)
        xmax = tl.max(xe, 1)
        wmin = tl.min(we, 0)
        wmax = tl.max(we, 0)
        xflags = tl.reduce(xb.to(tl.uint32) & 0x807FFFFF, 1, _scale_flags_or)
        wflags = tl.reduce(wb.to(tl.uint32) & 0x807FFFFF, 0, _scale_flags_or)
        xok = (xflags == 0) & (xmin > 0) & (xmax < 255)
        wok = (wflags == 0) & (wmin > 0) & (wmax < 255)
        base = xmin + wmin
        # A shift span of at most three bounds the worst-case INT8 sum by 2**29.
        # Keep products normal and the final scaled sum within FP32 range.
        valid = (
            xok
            & wok
            & (xmax - xmin + wmax - wmin <= 3)
            & (base >= 190)
            & (xmax + wmax <= 286)
        )
    if tl.min(valid if CLAMP_SCALES else (valid | (rm >= M)), 0) == 1:
        # Two bits per K group keep all 32 shifts in two registers per row.
        shifts = (xe + we[None, :] - base[:, None]).to(tl.uint32)
        packed = tl.sum(
            tl.reshape(shifts, (BM, 2, 16)) << (tl.arange(0, 16)[None, None, :] * 2), 2
        )
        word0, word1 = tl.split(packed)
        if SWAP_AB:
            tl.static_assert(SPLIT_K == 1)
            _integer_factor_body(
                A,
                W,
                None,
                Out,
                rm,
                rn,
                base - 127,
                M,
                pn * BN // 128,
                BM,
                BN,
                M % BM == 0,
                1,
                STAGES,
                WORD0=word0,
                WORD1=word1,
                INLINE_EXPONENTS=True,
            )
        else:
            _single_head_integer_body(
                A,
                W,
                Out,
                rm,
                rn,
                base,
                word0,
                word1,
                M,
                BM,
                BN,
                STAGES,
                Partial,
                Counter,
                SPLIT_K,
                A0=a0,
                W0=w0,
                TUNED=TUNED,
                CLAMP_A=PATH == 5,
            )
    elif SPLIT_K == 1 or tl.program_id(1) == 0:
        _int8_block_bmm_kernel(
            A,
            W,
            XS,
            WS,
            Out,
            M,
            1024,
            4096,
            (0, 4096, 1),
            (0, 1, 4096),
            (0, XSM, XSK),
            (0, WSK, WSN),
            (0, 1024, 1),
            1,
            128,
            128,
            BLOCK_M=BM,
            BLOCK_N=BN,
            GROUP_M=GM,
            ALIGNED=M % BM == 0,
            TILE_K=128,
            UNROLL=1,
            ACC_FP32=True,
            PACK_SCALES=not SWAP_AB and SPLIT_K <= 2 and not TUNED,
            SCALE_GROUPS=2,
        )


@triton.jit
def int8_single_head_paired_k_kernel(
    A,
    W,
    XS,
    WS,
    Out,
    M: tl.constexpr,
    XS_M: tl.constexpr,
    XS_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNMASK_A: tl.constexpr = False,
    UNMASK_S: tl.constexpr = False,
):
    block_m: tl.constexpr = 16
    n: tl.constexpr = 1024
    k: tl.constexpr = 4096
    scale_block: tl.constexpr = 128
    k_groups: tl.constexpr = k // scale_block
    pid = tl.program_id(0)
    pm = pid // (n // BLOCK_N)
    pn = pid % (n // BLOCK_N)
    row = pm * block_m + tl.arange(0, block_m)
    col = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    inner_k = tl.arange(0, scale_block)
    valid = row < M
    load_row = tl.minimum(row, M - 1) if UNMASK_A and M % block_m != 0 else row
    ap = A + load_row[:, None] * k + inner_k[None, :]
    wp = W + col[None, :] * k + inner_k[:, None]
    asp = XS + row * XS_M
    wsp = WS + (pn * BLOCK_N // scale_block) * k_groups
    acc = tl.zeros((block_m, BLOCK_N), tl.float32)
    for pair in tl.range(k_groups // 2, num_stages=2 if UNMASK_S else None):
        group = pair * 2
        groups = group + tl.arange(0, 2)
        if UNMASK_S:
            row_scales = tl.load(asp[:, None] + groups[None, :] * XS_K).to(tl.float32)
        else:
            row_scales = tl.load(
                asp[:, None] + groups[None, :] * XS_K, valid[:, None], 0
            ).to(tl.float32)
        scales = row_scales * tl.load(wsp + groups)[None, :].to(tl.float32)
        scale0, scale1 = tl.split(scales)
        for sub in tl.static_range(2):
            offset = (group + sub) * scale_block
            if UNMASK_A:
                a = tl.load(ap + offset)
            else:
                a = tl.load(ap + offset, valid[:, None], 0)
            w = tl.load(wp + offset)
            scale = scale0 if sub == 0 else scale1
            dot = tl.dot(a, w, out_dtype=tl.int32)
            acc = tl.fma(dot.to(tl.float32), scale[:, None], acc)
    tl.store(Out + row[:, None] * n + col[None, :], acc, valid[:, None])


@triton.jit
def int8_single_head_bmm_kernel(
    A,
    W,
    AS,
    WS,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA: tl.constexpr,
    SW: tl.constexpr,
    SAS: tl.constexpr,
    SWS: tl.constexpr,
    SO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_start = tl.program_id(1) * BLOCK_N
    col = col_start + tl.arange(0, BLOCK_N)
    scale_block: tl.constexpr = 128
    k_groups: tl.constexpr = K // scale_block
    groups = tl.arange(0, BLOCK_GROUPS)
    inner_k = tl.arange(0, scale_block)
    use_int64: tl.constexpr = (
        max(
            M * SA[1] + K * SA[2],
            K * SW[1] + N * SW[2],
            M * SAS[1] + k_groups * SAS[2],
            k_groups * SWS[1] + (N // scale_block) * SWS[2],
            M * SO[1] + N * SO[2],
        )
        >= 2**31
    )
    if use_int64:
        row = row.to(tl.int64)
        col = col.to(tl.int64)
        col_start = col_start.to(tl.int64)
        groups = groups.to(tl.int64)
        inner_k = inner_k.to(tl.int64)
    k = groups[:, None] * scale_block + inner_k[None, :]
    a_ptr = A + row[None, :, None] * SA[1] + k[:, None, :] * SA[2]
    w_ptr = W + k[:, :, None] * SW[1] + col[None, None, :] * SW[2]
    as_ptr = AS + groups[:, None] * SAS[2] + row[None, :] * SAS[1]
    ws_ptr = WS + groups * SWS[1] + (col_start // scale_block) * SWS[2]
    valid_row = row < M
    acc = tl.zeros((BLOCK_GROUPS, BLOCK_M, BLOCK_N), tl.float32)
    # Each dot has its own scale. Reduce the FP32 results inside the program
    # to avoid a workspace and a second launch for small single-head batches.
    for group in range(k_groups // BLOCK_GROUPS):
        if use_int64:
            group_offset = group.to(tl.int64) * BLOCK_GROUPS
        else:
            group_offset = group * BLOCK_GROUPS
        k_offset = group_offset * scale_block
        a = tl.load(a_ptr + k_offset * SA[2], valid_row[None, :, None], other=0)
        w = tl.load(w_ptr + k_offset * SW[1])
        a_scale = tl.load(
            as_ptr + group_offset * SAS[2], valid_row[None, :], other=0
        ).to(tl.float32)
        w_scale = tl.load(ws_ptr + group_offset * SWS[1]).to(tl.float32)
        dot = tl.dot(a, w, out_dtype=tl.int32).to(tl.float32)
        acc = tl.fma(dot, a_scale[:, :, None] * w_scale[:, None, None], acc)
    result = tl.sum(acc, axis=0)
    tl.store(
        Out + row[:, None] * SO[1] + col[None, :] * SO[2],
        result,
        valid_row[:, None],
    )


@triton.jit
def int8_single_head_packed_gemv_kernel(
    A,
    W,
    XS,
    WS,
    Out,
    XS_M: tl.constexpr,
    XS_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * 4 + tl.arange(0, 4)
    groups = tl.arange(0, BLOCK_GROUPS)
    packed_k = tl.arange(0, 32)
    a_packed = A.to(tl.pointer_type(tl.int32))
    w_packed = W.to(tl.pointer_type(tl.int32))
    acc = tl.zeros((BLOCK_GROUPS, 4), tl.float32)
    for start in range(32 // BLOCK_GROUPS):
        scale_k = start * BLOCK_GROUPS + groups
        offsets = scale_k[:, None] * 32 + packed_k[None, :]
        a = tl.load(a_packed + row * 1024 + offsets)
        w = tl.load(w_packed + col[None, :, None] * 1024 + offsets[:, None, :])
        # Four signed INT8 products per lane avoid padding a one-row dot to
        # the matrix instruction's minimum row tile. Reduce each K=128 group
        # before applying its original activation and weight scales.
        products = tl.inline_asm_elementwise(
            "v_dot4_i32_i8 $0, $1, $2, $3",
            constraints="=v,v,v,v",
            args=(a[:, None, :], w, tl.full((), 0, tl.int32)),
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        partial = tl.sum(products, 2).to(tl.float32)
        x_scale = tl.load(XS + row * XS_M + scale_k * XS_K).to(tl.float32)
        w_scale = tl.load(WS + (col[None, :] // 128) * 32 + scale_k[:, None]).to(
            tl.float32
        )
        acc = tl.fma(partial, x_scale[:, None] * w_scale, acc)
    tl.store(Out + row * 1024 + col, tl.sum(acc, 0))


def _int8_block_bmm(
    A,
    B,
    A_scale,
    B_scale,
    block_size=(128, 128, 128),
    out_dtype=torch.bfloat16,
    out=None,
):
    """Compute A[B,M,K] @ B[B,K,N] with block scales and shape-specific accumulation.

    A_scale is [B,ceil(M/block_m),ceil(K/block_k)]; B_scale is
    [B,ceil(K/block_k),ceil(N/block_n)]. Set block_m=1 for per-row scales.
    Unlike the NVIDIA FP8 entry point, input tensors must be signed INT8.
    Arbitrary nonnegative input strides are supported; output columns must be contiguous.
    The single-group projection and other tuned shapes accumulate in FP32;
    fallback shapes accumulate in BF16.
    The output dtype does not override this internal accumulation choice.
    """
    if A_scale is None or B_scale is None:
        raise ValueError("INT8 block BMM requires both scale tensors")
    tensors = (A, B, A_scale, B_scale)
    if any(t.ndim != 3 for t in tensors):
        raise ValueError("inputs and scales must have three dimensions")
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise TypeError("Hygon DCU block BMM requires INT8 A and B")
    if any(t.device != A.device for t in tensors) or A.device.type != "cuda":
        raise ValueError("all inputs must be on the same DCU device")
    if any(
        t.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        for t in tensors[2:]
    ):
        raise TypeError("scales must be float32, bfloat16 or float16")
    if len(block_size) != 3 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain three positive integers")
    sm, sn, sk = block_size
    if sn < 16 or sn & (sn - 1) or sk != 128:
        raise NotImplementedError("requires power-of-two block_n >= 16 and block_k=128")
    batch, m, k = A.shape
    if B.shape[:2] != (batch, k):
        raise ValueError("batch or K dimension mismatch")
    n = B.shape[2]
    k_scales = (k + sk - 1) // sk
    if A_scale.shape != (batch, (m + sm - 1) // sm, k_scales):
        raise ValueError("incorrect A_scale shape")
    if B_scale.shape != (batch, k_scales, (n + sn - 1) // sn):
        raise ValueError("incorrect B_scale shape")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("unsupported output dtype")
    if out is None:
        out = torch.empty((batch, m, n), dtype=out_dtype, device=A.device)
    elif (
        out.shape != (batch, m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or out.stride(2) != 1
        or out.stride(0) <= 0
        or out.stride(1) <= 0
    ):
        raise ValueError(
            "out must have unit column stride and matching shape, dtype and device"
        )
    if batch == 0 or m == 0 or n == 0:
        return out
    with torch_device_fn.device(A.device):
        if m <= 16 and n <= 32 and 0 < k <= 64:
            small_bmm_kernel[(m * triton.cdiv(n, 2), batch)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                m,
                n,
                k,
                A.stride(),
                B.stride(),
                A_scale.stride(),
                B_scale.stride(),
                out.stride(),
                sm,
                sn,
                triton.next_power_of_2(k),
                2,
                num_warps=1,
                num_stages=1,
            )
            return out
        if (
            8 <= m <= 48
            and (batch, n, k, sm, sn) == (1, 1024, 4096, 1, 128)
            and A.stride()[1:] == (4096, 1)
            and B.stride()[1:] == (1, 4096)
            and A_scale.stride()[1:] == (32, 1)
            and B_scale.stride()[1:] == (1, 32)
            and out.stride()[1:] == (1024, 1)
            and A_scale.dtype == B_scale.dtype == torch.float32
            and out_dtype in (torch.bfloat16, torch.float32)
            and A.data_ptr() % 4 == 0
            and B.data_ptr() % 4 == 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            partial, counter = _single_head_small_split_workspace(
                A.device, torch.cuda.current_stream(A.device).cuda_stream
            )
            int8_single_head_split_float_kernel[(triton.cdiv(m, 16) * 32, 4)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                partial,
                counter,
                m,
                32,
                1,
                32,
                4,
                num_warps=2,
                num_stages=2,
                kpack=2,
            )
            return out
        if (
            m <= 64
            and (m <= 16 or A_scale.stride(2) != 1)
            and (batch, n, k, sm, sn) == (1, 1024, 4096, 1, 128)
        ):
            if (
                m <= 4
                and A.stride()[1:] == (4096, 1)
                and B.stride()[1:] == (1, 4096)
                and B_scale.stride()[1:] == (1, 32)
                and out.stride(1) == 1024
                and A.data_ptr() % 4 == 0
                and B.data_ptr() % 4 == 0
            ):
                int8_single_head_packed_gemv_kernel[(m, 256)](
                    A,
                    B,
                    A_scale,
                    B_scale,
                    out,
                    A_scale.stride(1),
                    A_scale.stride(2),
                    BLOCK_GROUPS=8 if m == 1 else 32,
                    num_warps=4 if m == 1 else 2,
                    num_stages=1,
                )
                return out
            if m <= 16:
                bm, bn, groups = 16, 16, 8
            elif m <= 32:
                bm, bn, groups = 16, 32, 4
            else:
                bm, bn, groups = 32, 32, 4
            int8_single_head_bmm_kernel[((m + bm - 1) // bm, n // bn)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                m,
                n,
                k,
                A.stride(),
                B.stride(),
                A_scale.stride(),
                B_scale.stride(),
                out.stride(),
                bm,
                bn,
                groups,
                num_warps=4,
                kpack=2,
            )
            return out
        a_scale_stride = A_scale.stride()
        integer_cfg = _H1_TUNED_INTEGER_CONFIGS.get(m)
        if (
            integer_cfg is not None
            and (batch, n, k, sm, sn, sk) == (1, 1024, 4096, 1, 128, 128)
            and A.stride()[1:] == (4096, 1)
            and B.stride()[1:] == (1, 4096)
            and a_scale_stride[1:] == (32, 1)
            and B_scale.stride()[1:] == (1, 32)
            and out.stride()[1:] == (1024, 1)
            and A_scale.dtype == B_scale.dtype == torch.float32
            and out_dtype in (torch.bfloat16, torch.float32)
            and not torch.cuda.is_current_stream_capturing()
        ):
            bm, bn, split, warps, combined = integer_cfg
            if split > 1:
                partial, counter = _single_head_uneven_split_workspace(
                    A.device, torch.cuda.current_stream(A.device).cuda_stream
                )
            else:
                partial, counter = None, None
            int8_single_head_integer_kernel[(triton.cdiv(m, bm) * (1024 // bn), split)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                m,
                32,
                1,
                32,
                1,
                BM=bm,
                BN=bn,
                GM=4,
                STAGES=2,
                Partial=partial,
                Counter=counter,
                SPLIT_K=split,
                PATH=3 if combined else 2,
                num_warps=warps,
                num_stages=1,
                kpack=2,
            )
            return out
        if (
            104 < m <= 192
            and (batch, n, k, sm, sn, sk) == (1, 1024, 4096, 1, 128, 128)
            and A.dtype == B.dtype == torch.int8
            and A.stride()[1:] == (4096, 1)
            and B.stride()[1:] == (1, 4096)
            and a_scale_stride[1:] == (32, 1)
            and B_scale.stride()[1:] == (1, 32)
            and out.stride()[1:] == (1024, 1)
            and A_scale.dtype == B_scale.dtype == torch.float32
            and out_dtype in (torch.bfloat16, torch.float32)
            and not torch.cuda.is_current_stream_capturing()
        ):
            partial, counter = _single_head_uneven_split_workspace(
                A.device, torch.cuda.current_stream(A.device).cuda_stream
            )
            split = 5 if m <= 128 else 3
            bm = 32 if 128 < m <= 160 else 64
            int8_single_head_integer_kernel[(triton.cdiv(m, bm) * 16, split)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                m,
                32,
                1,
                32,
                1,
                BM=bm,
                BN=64,
                GM=4,
                STAGES=2,
                Partial=partial,
                Counter=counter,
                SPLIT_K=split,
                num_warps=2 if bm == 32 else 4,
                num_stages=1,
                kpack=2,
            )
            return out
        if (
            160 < m <= 320
            and (batch, n, k, sm, sn) == (1, 1024, 4096, 1, 128)
            and A.stride()[1:] == (4096, 1)
            and B.stride()[1:] == (1, 4096)
            and a_scale_stride[1:] == (32, 1)
            and B_scale.stride()[1:] == (1, 32)
            and out.stride()[1:] == (1024, 1)
            and A_scale.dtype == B_scale.dtype == torch.float32
            and not torch.cuda.is_current_stream_capturing()
        ):
            partial, counter = _single_head_split_workspace(
                A.device, torch.cuda.current_stream(A.device).cuda_stream
            )
            int8_single_head_integer_kernel[(triton.cdiv(m, 64) * 16, 2)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                m,
                32,
                1,
                32,
                1,
                BM=64,
                BN=64,
                GM=4,
                STAGES=2,
                Partial=partial,
                Counter=counter,
                SPLIT_K=2,
                num_warps=4,
                num_stages=2,
                kpack=1,
            )
            return out
        if (
            224 < m <= 512
            and (batch, n, k, sm, sn) == (1, 1024, 4096, 1, 128)
            and A.stride()[1:] == (4096, 1)
            and B.stride()[1:] == (1, 4096)
            and a_scale_stride[1:] == (32, 1)
            and B_scale.stride()[1:] == (1, 32)
            and out.stride()[1:] == (1024, 1)
            and A_scale.dtype == B_scale.dtype == torch.float32
        ):
            int8_single_head_integer_kernel[(triton.cdiv(m, 64) * 16,)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                m,
                32,
                1,
                32,
                1,
                BM=64,
                BN=64,
                GM=4,
                STAGES=2,
                num_warps=4,
                num_stages=2,
                kpack=1,
            )
            return out
        if (
            64 < m <= 224
            and (m > 72 or a_scale_stride[2] != 1)
            and (batch, n, k, sm, sn) == (1, 1024, 4096, 1, 128)
            and A.stride()[1:] == (4096, 1)
            and B.stride()[1:] == (1, 4096)
            and B_scale.stride()[1:] == (1, 32)
            and a_scale_stride[1:] in ((32, 1), (1, m))
            and out.stride(1) == 1024
        ):
            small_n_tile = m <= 96 or (m <= 112 and a_scale_stride[2] == 1)
            bn = 32 if small_n_tile else 64
            int8_single_head_paired_k_kernel[(triton.cdiv(m, 16) * (n // bn),)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                m,
                a_scale_stride[1],
                a_scale_stride[2],
                bn,
                UNMASK_A=m <= 160,
                UNMASK_S=small_n_tile and m % 16 == 0,
                num_warps=2,
                num_stages=2,
                kpack=2,
            )
            return out
        if (
            (
                m == 1028
                or (
                    m in (2048, 3076, 3805, 4100)
                    and out_dtype in (torch.bfloat16, torch.float32)
                )
            )
            and (batch, n, k, sm, sn) == (1, 1024, 4096, 1, 128)
            and A.stride()[1:] == (4096, 1)
            and B.stride()[1:] == (1, 4096)
            and a_scale_stride[1:] == (32, 1)
            and B_scale.stride()[1:] == (1, 32)
            and out.stride()[1:] == (1024, 1)
            and A_scale.dtype == B_scale.dtype == torch.float32
        ):
            int8_single_head_integer_kernel[(triton.cdiv(m, 128) * 8,)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                m,
                32,
                1,
                32,
                1,
                BM=128,
                BN=128,
                GM=4,
                STAGES=2,
                PATH=(4 if m in (3076, 3805, 4100) else 5 if m == 1028 else True),
                num_warps=8,
                num_stages=1 if m in (2048, 3076, 3805, 4100) else 2,
                kpack=1,
            )
            return out
        if (
            m >= 3072
            and (
                m >= 4096
                or (a_scale_stride[1:] == (32, 1) and B_scale.stride()[1:] == (1, 32))
            )
            and (batch, n, k, sm, sn) == (1, 1024, 4096, 1, 128)
            and A.stride()[1:] == (4096, 1)
            and B.stride()[1:] == (1, 4096)
            and out.stride()[1:] == (1024, 1)
            and A_scale.dtype == B_scale.dtype == torch.float32
        ):
            scale_info = torch.empty((m + 8,), dtype=torch.int32, device=A.device)
            packed_scales = torch.empty((m + 8, 8), dtype=torch.int32, device=A.device)
            scale_strides = (
                a_scale_stride[1],
                a_scale_stride[2],
                B_scale.stride(2),
                B_scale.stride(1),
            )
            prepare_scale_factors[(triton.cdiv(m, 32) + 1,)](
                A_scale,
                B_scale,
                scale_info,
                packed_scales,
                m,
                *scale_strides,
                32,
                num_warps=4,
            )
            integer_factor_bmm_kernel[(triton.cdiv(m, 128) * 8,)](
                A,
                B,
                A_scale,
                B_scale,
                scale_info,
                packed_scales,
                out,
                m,
                *scale_strides,
                BM=128,
                BN=128,
                GM=4 if m % 128 == 0 else 8,
                INNER=1,
                IST=2,
                num_warps=8,
                num_stages=1,
                kpack=1,
            )
            return out
        cfg = _get_int8_config(batch, m, n, k, sn, sm, a_scale_stride[2] == 1)
        bm, bn = cfg["BLOCK_M"], cfg["BLOCK_N"]
        tile_k = cfg.pop("TILE_K")
        if tile_k is None:
            tile_k = min(sk, max(32, triton.next_power_of_2(k)))
        aligned = m % bm == 0 and n % bn == 0 and k % tile_k == 0
        # These kernels have fixed launch configurations. Use the direct JIT
        # launcher to avoid rebuilding libentry's specialization key on every call.
        tiles = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
        _int8_block_bmm_kernel[(tiles, batch, 1)](
            A,
            B,
            A_scale,
            B_scale,
            out,
            m,
            n,
            k,
            A.stride(),
            B.stride(),
            a_scale_stride,
            B_scale.stride(),
            out.stride(),
            sm,
            sn,
            sk,
            ALIGNED=aligned,
            TILE_K=tile_k,
            **cfg,
        )
    return out


@libentry()
@triton.jit
def zero_bmm_kernel(
    Output,
    NUMEL: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch = offsets // (M * N)
    rows = offsets // N % M
    columns = offsets % N
    pointers = Output + batch * STRIDES[0] + rows * STRIDES[1] + columns * STRIDES[2]
    tl.store(pointers, 0.0, offsets < NUMEL)


def w8a8_block_int8_bmm(
    x: torch.Tensor,
    y: torch.Tensor,
    xs: torch.Tensor | None,
    ys: torch.Tensor | None,
    block_size=(128, 128),
    z: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Hygon DCU implementation of the upstream W8A8 BMM interface.

    x/y use [B,M,K]/[B,N,K], xs/ys use [B,M,ceil(K/block_k)] /
    [B,ceil(N/block_n),ceil(K/block_k)]. Hygon DCU quantized inputs are INT8;
    floating inputs with xs=ys=None use the floating kernel in this file.
    A supplied z[B,M,N] is written in place, including interleaved batch views.
    """
    logger.debug("GEMS_HYGON W8A8_BLOCK_INT8_BMM")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("W8A8 BMM inputs must have three dimensions")
    batch, m, k = x.shape
    if y.shape[0] != batch or y.shape[2] != k or x.device != y.device:
        raise ValueError("W8A8 BMM input shape or device mismatch")
    if x.dtype != y.dtype:
        raise TypeError("W8A8 BMM inputs must have matching dtypes")
    if len(block_size) != 2 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain block_n and block_k")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("unsupported output dtype")
    n = y.shape[1]
    if x.dtype == torch.int8:
        if xs is None or ys is None:
            raise ValueError("INT8 W8A8 BMM requires both scale tensors")
        block_n, block_k = block_size
        k_scales = (k + block_k - 1) // block_k
        if xs.shape != (batch, m, k_scales) or ys.shape != (
            batch,
            (n + block_n - 1) // block_n,
            k_scales,
        ):
            raise ValueError("incorrect W8A8 BMM scale shape")
        return _int8_block_bmm(
            x,
            y.transpose(1, 2),
            xs,
            ys.transpose(1, 2),
            block_size=(1, block_n, block_k),
            out_dtype=output_dtype,
            out=z,
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("unsupported W8A8 BMM input dtype")
    if xs is not None or ys is not None:
        raise ValueError("floating BMM inputs must not have quantization scales")
    if z is None:
        z = torch.empty((batch, m, n), device=x.device, dtype=output_dtype)
    elif (
        z.shape != (batch, m, n)
        or z.device != x.device
        or z.dtype != output_dtype
        or z.stride(2) != 1
        or z.stride(0) <= 0
        or z.stride(1) <= 0
    ):
        raise ValueError(
            "z must have matching shape, device, dtype and unit column stride"
        )
    if z.numel() == 0:
        return z
    if k == 0:
        # Empty reductions have no dot product to tune; the log autotune key requires K > 0.
        with torch_device_fn.device(x.device):
            zero_bmm_kernel[(triton.cdiv(z.numel(), 256),)](
                z, z.numel(), m, n, z.stride(), BLOCK_SIZE=256
            )
        return z
    return bmm_out(x, y.transpose(1, 2), z)


@lru_cache(maxsize=16)
def _single_head_small_split_workspace(device, raw_stream):
    # Four K slices for at most three 16-row tiles and 32 column tiles.
    partial = torch.empty((96, 4, 16, 32), dtype=torch.float32, device=device)
    counter = torch.zeros((96,), dtype=torch.int32, device=device)
    return partial, counter


@triton.jit
def _single_head_split_float_fallback(
    A,
    W,
    XS,
    WS,
    O,
    PM,
    PN,
    M: tl.constexpr,
    XSM: tl.constexpr,
    XSK: tl.constexpr,
    BN: tl.constexpr,
):
    G: tl.constexpr = 8 if M <= 16 else 1
    groups = tl.arange(0, G)
    rk = tl.arange(0, 32)
    col = PN * BN + tl.arange(0, BN)
    ap = A.to(tl.pointer_type(tl.int32))
    wp = W.to(tl.pointer_type(tl.int32))
    for inner in range(16):
        row = PM * 16 + inner
        if row < M:
            accum = tl.zeros((G, BN), tl.float32)
            for block in range(32 // G):
                kg = block * G + groups
                kk = kg[:, None] * 32 + rk[None, :]
                av = tl.load(ap + row * 1024 + kk)
                wv = tl.load(wp + col[None, :, None] * 1024 + kk[:, None, :])
                prod = tl.inline_asm_elementwise(
                    "v_dot4_i32_i8 $0, $1, $2, $3",
                    "=v,v,v,v",
                    [av[:, None, :], wv, tl.full((), 0, tl.int32)],
                    dtype=tl.int32,
                    is_pure=True,
                    pack=1,
                )
                part = tl.sum(prod, 2).to(tl.float32)
                xs = tl.load(XS + row * XSM + kg * XSK)
                ws = tl.load(WS + (PN * BN // 128) * 32 + kg)
                accum = tl.fma(part, (xs * ws)[:, None], accum)
            tl.store(O + row * 1024 + col, tl.sum(accum, 0))


@triton.jit
def int8_single_head_split_float_kernel(
    A,
    W,
    XS,
    WS,
    O,
    P,
    C,
    M: tl.constexpr,
    XSM: tl.constexpr,
    XSK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    pm = pid // (1024 // BN)
    pn = pid % (1024 // BN)
    row = pm * 16 + tl.arange(0, 16)
    col = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, 128)
    ap = A + row[:, None] * 4096 + rk[None, :]
    wp = W + col[None, :] * 4096 + rk[:, None]
    asp = XS + row * XSM
    wsp = WS + (pn * BN // 128) * 32
    acc = tl.zeros((16, BN), tl.float32)
    # OR only predicates; combining float exponent bits can create false alarms.
    scale_limit: tl.constexpr = 18446744073709551616.0
    unsafe_flags = tl.zeros((16,), tl.int32)
    for pair in tl.range(16 // SPLIT, num_stages=2):
        group = split * (32 // SPLIT) + pair * 2
        groups = group + tl.arange(0, 2)
        if M % 16 == 0:
            xs = tl.load(asp[:, None] + groups[None, :] * XSK)
        else:
            xs = tl.load(asp[:, None] + groups[None, :] * XSK, row[:, None] < M, 0)
        scales = xs * tl.load(wsp + groups)[None, :]
        scale0, scale1 = tl.split(scales)
        unsafe_flags |= (~(tl.abs(scale0) < scale_limit)).to(tl.int32) | (
            ~(tl.abs(scale1) < scale_limit)
        ).to(tl.int32)
        for sub in tl.static_range(2):
            offset = (group + sub) * 128
            if M % 16 == 0:
                a = tl.load(ap + offset)
            else:
                a = tl.load(ap + offset, row[:, None] < M, 0)
            w = tl.load(wp + offset)
            scale = scale0 if sub == 0 else scale1
            part = tl.dot(a, w, out_dtype=tl.int32)
            acc = tl.fma(part.to(tl.float32), scale[:, None], acc)
    offset = tl.arange(0, 16)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(P + (pid * SPLIT + split) * (16 * BN) + offset, acc)
    tl.debug_barrier()
    unsafe = tl.reduce(unsafe_flags, 0, _scale_flags_or)
    ticket = tl.atomic_add(C + pid * 1, 1 + (unsafe.to(tl.int32) << 8), sem="acq_rel")
    if (ticket & 255) == SPLIT - 1:
        if ((ticket >> 8) + unsafe) > 0:
            _single_head_split_float_fallback(A, W, XS, WS, O, pm, pn, M, XSM, XSK, BN)
        else:
            total = tl.zeros((16, BN), tl.float32)
            for other in tl.static_range(SPLIT):
                total += tl.load(P + (pid * SPLIT + other) * (16 * BN) + offset)
            tl.store(O + row[:, None] * 1024 + col[None, :], total, row[:, None] < M)
        tl.atomic_xchg(C + pid * 1, 0, sem="release")

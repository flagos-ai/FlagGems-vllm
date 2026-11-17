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

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


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
        num_stages = 1
        unroll = 4 if m <= 96 else (2 if m <= 512 else 1)
        if pack_group is not None:
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
        elif m == 496 and scale_k_contiguous:
            bm, bn, warps, group = 64, 32, 2, 1
            pack_scales = True
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
            kpack=2,
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
        tl.static_assert(SCALE_M == 1 and TILE_K == SCALE_K and K % (4 * TILE_K) == 0)
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
        for first_k in tl.range(0, k_tiles, 4):
            # Fetch adjacent scales together while keeping all four dots independent.
            groups = first_k + tl.arange(0, 4)
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
            even, odd = tl.split(
                tl.reshape(scale_rows * scale_cols[None, :], (BLOCK_M, 2, 2))
            )
            scale0, scale2 = tl.split(even)
            scale1, scale3 = tl.split(odd)
            for part in tl.static_range(4):
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
    ap = A + row[:, None] * k + inner_k[None, :]
    wp = W + col[None, :] * k + inner_k[:, None]
    asp = XS + row * XS_M
    wsp = WS + (pn * BLOCK_N // scale_block) * k_groups
    even = tl.zeros((block_m, BLOCK_N), tl.float32)
    odd = tl.zeros((block_m, BLOCK_N), tl.float32)
    # Separate accumulators let the adjacent dot and scale operations overlap.
    for pair in range(k_groups // 2):
        group = pair * 2
        offset = group * scale_block
        a0 = tl.load(ap + offset, valid[:, None], 0)
        w0 = tl.load(wp + offset)
        a1 = tl.load(ap + offset + scale_block, valid[:, None], 0)
        w1 = tl.load(wp + offset + scale_block)
        s0 = tl.load(asp + group * XS_K, valid, 0).to(tl.float32) * tl.load(
            wsp + group
        ).to(tl.float32)
        s1 = tl.load(asp + (group + 1) * XS_K, valid, 0).to(tl.float32) * tl.load(
            wsp + group + 1
        ).to(tl.float32)
        d0 = tl.dot(a0, w0, out_dtype=tl.int32)
        d1 = tl.dot(a1, w1, out_dtype=tl.int32)
        even = tl.fma(d0.to(tl.float32), s0[:, None], even)
        odd = tl.fma(d1.to(tl.float32), s1[:, None], odd)
    tl.store(Out + row[:, None] * n + col[None, :], even + odd, valid[:, None])


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
        if m <= 64 and (batch, n, k, sm, sn) == (1, 1024, 4096, 1, 128):
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
        if (
            64 < m <= 176
            and (batch, n, k, sm, sn) == (1, 1024, 4096, 1, 128)
            and A.stride()[1:] == (4096, 1)
            and B.stride()[1:] == (1, 4096)
            and B_scale.stride()[1:] == (1, 32)
            and a_scale_stride[1:] in ((32, 1), (1, m))
            and out.stride(1) == 1024
        ):
            bn = 32 if m <= 96 else 64
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
                num_warps=2,
                num_stages=2,
                kpack=2,
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

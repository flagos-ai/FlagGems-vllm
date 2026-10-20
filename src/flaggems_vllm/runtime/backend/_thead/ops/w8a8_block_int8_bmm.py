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
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils import triton_lang_extension as ext

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
    USE_INT64: tl.constexpr = False,
):
    # batch offsets
    pid_b = ext.program_id(2)
    if USE_INT64:
        pid_b = pid_b.to(tl.int64)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
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


def bmm_out(A, B, out):
    batch, M, K = A.shape
    N = B.shape[2]

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
            USE_INT64=_bmm_requires_int64(A, B, out),
        )
    return out


# PPU-ZW810E, torch 2.10.0 / Triton 3.6.0, FlagTree d96f5339 / SDK 2.1, 2026-09-09.
# Values: BLOCK_M, BLOCK_N, num_warps, num_stages, GROUP_M, ACC_FP32, SWAP_AB.
EXACT_CONFIGS = {
    (8, 1, 1024, 4096): (16, 128, 4, 2, 1, True, True),
    (8, 4, 1024, 4096): (16, 128, 4, 2, 1, True, True),
    (8, 8, 1024, 4096): (16, 128, 4, 2, 1, True, True),
    (8, 16, 1024, 4096): (32, 128, 4, 2, 1, False, False),
    (8, 32, 1024, 4096): (32, 128, 4, 2, 1, False, False),
    (8, 64, 1024, 4096): (32, 128, 4, 2, 1, False, False),
    (8, 128, 1024, 4096): (64, 128, 4, 3, 8, False, False),
    (8, 4096, 1024, 4096): (64, 128, 4, 3, 8, False, False),
    (8, 8192, 1024, 4096): (64, 128, 4, 3, 8, False, False),
    (8, 16384, 1024, 4096): (64, 128, 4, 3, 32, True, False),
    (8, 32768, 1024, 4096): (64, 128, 4, 3, 32, True, False),
    (16, 1, 1024, 7168): (16, 128, 4, 2, 1, True, True),
    (16, 4, 1024, 7168): (16, 128, 4, 2, 1, True, True),
    (16, 8, 1024, 7168): (16, 128, 4, 2, 1, True, True),
    (16, 16, 1024, 7168): (32, 128, 4, 2, 1, False, True),
    (16, 32, 1024, 7168): (32, 128, 4, 2, 1, False, False),
    (16, 64, 1024, 7168): (64, 128, 4, 3, 8, True, False),
    (16, 128, 1024, 7168): (64, 128, 4, 3, 8, False, False),
    (16, 4096, 1024, 7168): (64, 128, 4, 3, 8, False, False),
    (16, 8192, 1024, 7168): (64, 128, 4, 3, 32, True, False),
    (16, 16384, 1024, 7168): (64, 128, 4, 3, 32, True, False),
    (16, 32768, 1024, 7168): (64, 128, 4, 3, 16, True, False),
}
# Single-head (h=1) FlagOSTune shapes with N=1024, K=4096, same date and stack.
# Rows: (max M, BLOCK_M, BLOCK_N, num_warps, num_stages); FP32 accumulation.
SINGLE_HEAD_CONFIGS = (
    (16, 16, 16, 2, 3),
    (32, 32, 16, 2, 3),
    (112, 32, 32, 2, 3),
    (128, 64, 32, 4, 3),
    (448, 64, 64, 8, 3),
    (480, 32, 64, 4, 2),
    (512, 128, 64, 4, 2),
    (float("inf"), 64, 128, 8, 4),
)
SINGLE_HEAD_SHAPE = (1, 1024, 4096)
# Rows up to this M use the GEMV kernel: an MMA tile wastes most of its rows and
# its N tiles cannot fill the SMs, while the BF16 baseline is a dedicated GEMV.
SINGLE_ROW_MAX_M = 2
SINGLE_ROW_BLOCK_N = 2
SINGLE_ROW_BLOCK_K = 2048


def _get_int8_config(batch, m, n, k, scale_n=128):
    key = (batch, m, n, k)
    acc_fp32, swap_ab, runtime_k_loop = False, False, False
    if key in EXACT_CONFIGS:
        bm, bn, warps, stages, group, acc_fp32, swap_ab = EXACT_CONFIGS[key]
    elif (batch, n, k) == SINGLE_HEAD_SHAPE:
        acc_fp32, runtime_k_loop = True, True
        bm, bn, warps, stages = next(
            cfg[1:] for cfg in SINGLE_HEAD_CONFIGS if m <= cfg[0]
        )
        group = 1
    elif n == 1024 and (batch, k) in ((8, 4096), (16, 7168)):
        if m <= 64:
            bm, bn, warps, stages, group = 32, 128, 4, 2, 1
        else:
            bm, bn, warps, stages, group = 64, 128, 4, 3, 8
    elif m <= 64:
        bm, bn, warps, stages, group = 32, 64, 4, 2, 1
    elif k >= 4096:
        bm, bn, warps, stages, group = 128, 64, 4, 3, 8
    else:
        bm, bn, warps, stages, group = 64, 64, 4, 2, 4
    return dict(
        BLOCK_M=bm,
        BLOCK_N=min(bn, scale_n),
        num_warps=warps,
        num_stages=stages,
        GROUP_M=group,
        ACC_FP32=acc_fp32,
        SWAP_AB=swap_ab,
        RUNTIME_K_LOOP=runtime_k_loop,
    )


AIU_OVERRIDES = {(4, 64, 64, 128): False, (2, 128, 128, 256): False}


@libentry()
@triton.jit
def single_row_bmm_kernel(
    A,
    W,
    AS,
    WS,
    O,
    N,
    K,
    SA: tl.constexpr,
    SW: tl.constexpr,
    SAS: tl.constexpr,
    SWS: tl.constexpr,
    SO: tl.constexpr,
    SCALE_M: tl.constexpr,
    SCALE_N: tl.constexpr,
    SCALE_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pn = tl.program_id(0)
    row = tl.program_id(1)
    batch = tl.program_id(2).to(tl.int64)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    groups: tl.constexpr = BLOCK_K // SCALE_K
    group_ids = tl.arange(0, groups)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k_start in range(0, K, BLOCK_K):
        kk = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(A + batch * SA[0] + row * SA[1] + kk * SA[2]).to(tl.float32)
        w = tl.load(W + batch * SW[0] + kk[None, :] * SW[1] + cols[:, None] * SW[2]).to(
            tl.float32
        )
        partial = tl.sum(tl.reshape(w * a[None, :], (BLOCK_N, groups, SCALE_K)), 2)
        scale_k = k_start // SCALE_K + group_ids
        a_scale = tl.load(
            AS + batch * SAS[0] + (row // SCALE_M) * SAS[1] + scale_k * SAS[2]
        ).to(tl.float32)
        w_scale = tl.load(
            WS
            + batch * SWS[0]
            + scale_k[None, :] * SWS[1]
            + (cols // SCALE_N)[:, None] * SWS[2]
        ).to(tl.float32)
        acc += tl.sum(partial * (a_scale[None, :] * w_scale), 1)
    tl.store(O + batch * SO[0] + row * SO[1] + cols * SO[2], acc)


@libentry()
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
    k_tiles,
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
    USE_AIU: tl.constexpr,
    USE_AIU_W: tl.constexpr,
    TILE_K: tl.constexpr,
    ACC_FP32: tl.constexpr = False,
    SWAP_AB: tl.constexpr = False,
    RUNTIME_K_LOOP: tl.constexpr = False,
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
    if USE_AIU:
        # Describe the full physical row and select an interleaved head with
        # a column offset, so AIU bounds include the allocation's true row end.
        interleaved: tl.constexpr = SA[1] != K
        a_bp = tl.make_block_ptr(
            A if interleaved else A + batch * SA[0],
            shape=(M, SA[1]),
            strides=(SA[1], SA[2]),
            offsets=(pm * BLOCK_M, (batch * SA[0]).to(tl.int32) if interleaved else 0),
            block_shape=(BLOCK_M, TILE_K),
            order=(1, 0),
        )
        w_bp = tl.make_block_ptr(
            W + batch * SW[0],
            shape=(SW[2] if SW[1] == 1 else K, N),
            strides=(SW[1], SW[2]),
            offsets=(0, pn * BLOCK_N),
            block_shape=(TILE_K, BLOCK_N),
            order=(0, 1) if SW[1] == 1 else (1, 0),
        )
    acc_dtype: tl.constexpr = tl.float32 if ACC_FP32 else tl.bfloat16
    if SWAP_AB:
        acc = tl.zeros((BLOCK_N, BLOCK_M), acc_dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), acc_dtype)
    # On the single-head AIU path a constant trip count makes the PPU backend
    # schedule the pipelined loop about 25% slower on large M.
    if USE_AIU and RUNTIME_K_LOOP:
        num_k_tiles = k_tiles
    else:
        num_k_tiles = tl.cdiv(K, TILE_K)
    for kb in tl.range(
        num_k_tiles,
        loop_unroll_factor=(
            (K + TILE_K - 1) // TILE_K if K > 0 and K <= 256 and not USE_AIU else 1
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
            WS + batch * SWS[0] + scale_k * SWS[1] + (pn * BLOCK_N // SCALE_N) * SWS[2]
        )
        if USE_AIU:
            if ALIGNED:
                a = tle.load(a_bp, is_async=True)
            else:
                # AIU zero-fills the M tail, so unaligned M keeps async copies.
                a = tle.load(
                    a_bp, is_async=True, boundary_check=(0,), padding_option="zero"
                )
            if USE_AIU_W:
                w = tle.load(w_bp, is_async=True)
            else:
                w = tl.load(wp)
            a_bp = tl.advance(a_bp, (0, TILE_K))
            w_bp = tl.advance(w_bp, (TILE_K, 0))
        elif ALIGNED:
            a = tl.load(ap)
            w = tl.load(wp)
        else:
            a = tl.load(ap, (rm[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(wp, (kk[:, None] < K) & (rn[None, :] < N), other=0)
        if ALIGNED or (K <= 256 and SCALE_M >= BLOCK_M and SCALE_M % BLOCK_M == 0):
            a_scale = tl.load(asp)
        else:
            a_scale = tl.load(asp, rm < M, other=0)
        w_scale = tl.load(wsp)
        row_scale = (a_scale.to(tl.float32) * w_scale.to(tl.float32)).to(acc_dtype)
        # Swap small-M contractions so M occupies the narrower dot dimension.
        if SWAP_AB:
            partial = tl.dot(tl.trans(w), tl.trans(a), out_dtype=tl.int32).to(acc_dtype)
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


def _prepare_out(out, shape, dtype, device):
    if out is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        out.shape != shape
        or out.dtype != dtype
        or out.device != device
        or out.stride(2) != 1
        or out.stride(0) <= 0
        or out.stride(1) <= 0
    ):
        raise ValueError(
            "z must have matching shape, dtype, device and unit column stride"
        )
    return out


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
    Tuned shapes may accumulate in FP32; fallback shapes accumulate in BF16.
    The output dtype does not override this internal accumulation choice.
    """
    tensors = (A, B, A_scale, B_scale)
    if any(t.device != A.device for t in tensors) or A.device.type != "cuda":
        raise ValueError("all inputs must be on the same PPU device")
    if any(
        t.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        for t in tensors[2:]
    ):
        raise TypeError("scales must be float32, bfloat16 or float16")
    sm, sn, sk = block_size
    if sn < 16 or sn & (sn - 1) or sk != 128:
        raise NotImplementedError("requires power-of-two block_n >= 16 and block_k=128")
    batch, m, k = A.shape
    n = B.shape[2]
    out = _prepare_out(out, (batch, m, n), out_dtype, A.device)
    if batch == 0 or m == 0 or n == 0:
        return out
    with torch_device_fn.device(A.device):
        if (batch, n, k) == SINGLE_HEAD_SHAPE and m <= SINGLE_ROW_MAX_M:
            single_row_bmm_kernel[(n // SINGLE_ROW_BLOCK_N, m, batch)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                n,
                k,
                A.stride(),
                B.stride(),
                A_scale.stride(),
                B_scale.stride(),
                out.stride(),
                sm,
                sn,
                sk,
                SINGLE_ROW_BLOCK_N,
                SINGLE_ROW_BLOCK_K,
                num_warps=4,
                num_stages=1,
            )
            return out
        cfg = _get_int8_config(batch, m, n, k, sn)
        bm, bn = cfg["BLOCK_M"], cfg["BLOCK_N"]
        tile_k = min(sk, max(32, triton.next_power_of_2(k)))
        aligned = m % bm == 0 and n % bn == 0 and k % tile_k == 0
        use_aiu = (
            n % bn == 0
            and k % tile_k == 0
            and A.stride(2) == 1
            and A.stride(1) < 2**31
            and (
                B.stride(2) == 1
                or (
                    B.stride(1) == 1
                    and B.data_ptr() % 128 == 0
                    and B.stride(0) == n * k
                    and B.stride(2) == k
                )
            )
            and A.data_ptr() % 128 == 0
            and (
                (A.stride(1) == k and A.stride(0) == m * k)
                or (A.stride(0) == k and A.stride(1) == batch * k)
            )
            and A.stride(0) % 128 == 0
            and A.stride(1) % 128 == 0
            and AIU_OVERRIDES.get((batch, m, n, k), True)
        )
        _int8_block_bmm_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), batch)](
            A,
            B,
            A_scale,
            B_scale,
            out,
            m,
            n,
            k,
            triton.cdiv(k, tile_k),
            A.stride(),
            B.stride(),
            A_scale.stride(),
            B_scale.stride(),
            out.stride(),
            sm,
            sn,
            sk,
            ALIGNED=aligned,
            USE_AIU=use_aiu,
            USE_AIU_W=(use_aiu and B.stride(1) == 1),
            TILE_K=tile_k,
            **cfg,
        )
    return out


def w8a8_block_int8_bmm(
    x: torch.Tensor,
    y: torch.Tensor,
    xs: torch.Tensor | None,
    ys: torch.Tensor | None,
    block_size=(128, 128),
    z: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """PPU implementation of the upstream W8A8 BMM interface.

    x/y use [B,M,K]/[B,N,K], xs/ys use [B,M,ceil(K/block_k)] /
    [B,ceil(N/block_n),ceil(K/block_k)]. PPU quantized inputs are INT8;
    floating inputs with xs=ys=None use the floating kernel in this file.
    A supplied z[B,M,N] is written in place, including interleaved batch views.
    """
    logger.debug("GEMS_THEAD W8A8_BLOCK_INT8_BMM")
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
        if xs.shape != (batch, m, triton.cdiv(k, block_size[1])) or ys.shape != (
            batch,
            triton.cdiv(n, block_size[0]),
            triton.cdiv(k, block_size[1]),
        ):
            raise ValueError("incorrect W8A8 BMM scale shape")
        return _int8_block_bmm(
            x,
            y.transpose(1, 2),
            xs,
            ys.transpose(1, 2),
            block_size=(1, block_size[0], block_size[1]),
            out_dtype=output_dtype,
            out=z,
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("unsupported W8A8 BMM input dtype")
    if xs is not None or ys is not None:
        raise ValueError("floating BMM inputs must not have quantization scales")
    z = _prepare_out(z, (batch, m, n), output_dtype, x.device)
    if z.numel() == 0:
        return z
    if k == 0:
        # Empty contraction dim: the floating BMM tuning key uses a log strategy
        # that cannot take K=0, and the mathematical result is the zero tensor.
        return z.zero_()
    return bmm_out(x, y.transpose(1, 2), z)

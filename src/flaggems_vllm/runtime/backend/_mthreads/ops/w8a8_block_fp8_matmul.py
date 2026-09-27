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

"""Block-scaled FP8 GEMM for MThreads.

SQMMA uses TME or masked copies and optional warp-specialized pipelining.
Eligible E5M2 and strict FP32 paths prepare exact FP16 operands in one kernel.
Very small GEMMs use direct FP32 vector reductions without prepared weights.
The matrix path has independent general, swap-AB, Split-K, and short-K tuning
entries while sharing one Triton body.
"""

import math
import os
from typing import List

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import has_triton_tle_attrs

HAS_TLE = has_triton_tle_attrs(
    ("pipe", "gpu.copy", "gpu.wgmma", "gpu.wgmma_wait", "gpu.warp_specialize"),
    3,
    6,
    0,
)
if HAS_TLE:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor
else:
    tle = None

EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "w8a8_block_fp8_matmul_mthreads_expand.yaml",
    )
)


def _tle_descriptor_shapes(args):
    if args.get("USE_TME", False):
        tile_k = args.get("TILE_K", 128)
        args["A"].block_shape = [max(16, args["BM"]), tile_k]
        args["B"].block_shape = [args["BN"], tile_k]


def _tle_configs():
    configs = runtime.get_tuned_config("w8a8_block_fp8_matmul_mthreads_tle")
    for config in configs:
        config.pre_hook = _tle_descriptor_shapes
    return configs


def _prune_tle_configs(configs, named_args, **kwargs):
    block_m = 16 if named_args["M"] <= 16 else None
    return [
        c
        for c in configs
        if (block_m is None or max(16, c.kwargs["BM"]) == block_m)
        # Warp-specialized pipelines can fault during repeated MUSA graph
        # execution. Keep the copy paths on unified-CTA pipelines, and tune
        # their row tile so ragged/small M does not pay for a fixed BM=64.
        and not c.kwargs["WS"]
        and (
            c.kwargs["WS"]
            or c.kwargs["BN"] == 64
            or (kwargs.get("HALF", False) and named_args["M"] >= 512)
        )
    ]


@triton.jit
def _tle_producer(
    writer,
    A,
    B,
    pm,
    pn,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    USE_TME: tl.constexpr,
    TILE_K: tl.constexpr = 128,
):
    for it in tl.range(0, tl.cdiv(K, TILE_K), num_stages=1):
        slot = writer.acquire(it)
        if USE_TME:
            tle.gpu.copy(A, slot.a, (BM, TILE_K), (pm * BM, it * TILE_K))
            tle.gpu.copy(B, slot.b, (BN, TILE_K), (pn * BN, it * TILE_K))
        else:
            # TME requires aligned row strides. Masked copies zero-fill the last
            # quantization block and still feed native SQMMA without padding A/B.
            rm = pm * BM + tl.arange(0, BM)
            rn = pn * BN + tl.arange(0, BN)
            rk = it * TILE_K + tl.arange(0, TILE_K)
            tle.gpu.copy(
                A + rm[:, None] * K + rk[None, :],
                slot.a,
                (BM, TILE_K),
                mask=(rm[:, None] < M) & (rk[None, :] < K),
            )
            tle.gpu.copy(
                B + rn[:, None] * K + rk[None, :],
                slot.b,
                (BN, TILE_K),
                mask=(rn[:, None] < N) & (rk[None, :] < K),
            )
        writer.commit(it)


@triton.jit
def _tle_consumer(
    reader,
    As,
    Bs,
    C,
    pm,
    pn,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TILE_K: tl.constexpr = 128,
):
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for it in tl.range(0, tl.cdiv(K, TILE_K), num_stages=1):
        wait = reader.wait(it)
        partial = tle.gpu.wgmma(wait.slot.a, wait.slot.b, trans_b=True)
        partial = tle.gpu.wgmma_wait(0, partial)
        reader.release(it)
        # Match one SQMMA K tile to one scale group when GROUP_K is 32/64/128
        # 128; GROUP_K=256 reuses the scale across two 128-wide tiles.
        scale_it = it * TILE_K // GROUP_K
        sa = tl.load(As + rm * SAM + scale_it * SAK, rm < M, other=0)
        if BN <= GROUP_N and GROUP_N % BN == 0:
            sb = tl.load(Bs + (pn * BN // GROUP_N) * SBN + scale_it * SBK)
            acc += partial * (sa * sb)[:, None]
        else:
            sb = tl.load(Bs + (rn // GROUP_N) * SBN + scale_it * SBK, rn < N, other=0)
            acc += partial * sa[:, None] * sb[None, :]
    tl.store(
        C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N)
    )


@libentry()
@libtuner(
    configs=_tle_configs(),
    key=[
        "M",
        "N",
        "K",
        "SAM",
        "SAK",
        "SBN",
        "SBK",
        "GROUP_K",
        "GROUP_N",
        "TILE_K",
        "HALF",
        "USE_TME",
    ],
    strategy="default",
    prune_configs_by={"early_config_prune": _prune_tle_configs},
    warmup=5,
    rep=20,
)
@triton.jit
def _block_fp8_matmul_tle(
    A,
    B,
    As,
    Bs,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
    WS: tl.constexpr = True,
    HALF: tl.constexpr = False,
    GROUP_K: tl.constexpr = 128,
    GROUP_N: tl.constexpr = 128,
    TILE_K: tl.constexpr = 128,
    USE_TME: tl.constexpr = False,
):
    tl.static_assert(not WS, "MUSA graph execution requires the unified-CTA path")
    # SQMMA requires M tiles of at least 16. Keep this hardware constraint
    # explicit even if automatic block-size adjustment shrinks the candidate.
    TILE_M: tl.constexpr = max(16, BM)
    pid = tl.program_id(0)
    nm = tl.cdiv(M, TILE_M)
    nn = tl.cdiv(N, BN)
    group = pid // (8 * nn)
    first = group * 8
    gm = tl.minimum(nm - first, 8)
    pm = first + pid % gm
    pn = (pid % (8 * nn)) // gm
    if HALF:
        a = tle.gpu.alloc(
            (STAGES, TILE_M, TILE_K),
            tl.float16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (STAGES, BN, TILE_K),
            tl.float16,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
    else:
        a = tle.gpu.alloc(
            (STAGES, TILE_M, TILE_K),
            tl.float8e4nv,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (STAGES, BN, TILE_K),
            tl.float8e4nv,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=True,
        )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="w8a8", a=a, b=b)
    if WS:
        tle.gpu.warp_specialize(
            [
                (
                    _tle_consumer,
                    (
                        pipe.reader(),
                        As,
                        Bs,
                        C,
                        pm,
                        pn,
                        M,
                        N,
                        K,
                        SAM,
                        SAK,
                        SBN,
                        SBK,
                        GROUP_K,
                        GROUP_N,
                        TILE_M,
                        BN,
                        TILE_K,
                    ),
                ),
                (
                    _tle_producer,
                    (pipe.writer(), A, B, pm, pn, M, N, K, TILE_M, BN, USE_TME, TILE_K),
                ),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )
    else:
        # Short kernels or high CTA counts can favor a smaller, unified CTA.
        # Keep copies in this scope: the current SQMMA layout pass does not
        # update shared-memory types across calls to a separate copy helper.
        tl.static_assert(BN <= 128)
        reader = pipe.reader()
        writer = pipe.writer()
        rm = pm * TILE_M + tl.arange(0, TILE_M)
        rn = pn * BN + tl.arange(0, BN)
        acc = tl.zeros((TILE_M, BN), tl.float32)
        for it in tl.range(0, tl.cdiv(K, TILE_K), num_stages=1):
            slot = writer.acquire(it)
            if USE_TME:
                tle.gpu.copy(A, slot.a, (TILE_M, TILE_K), (pm * TILE_M, it * TILE_K))
                tle.gpu.copy(B, slot.b, (BN, TILE_K), (pn * BN, it * TILE_K))
            else:
                # TME requires aligned row strides. Masked copies zero-fill the last
                # quantization block and still feed native SQMMA without padding A/B.
                rm = pm * TILE_M + tl.arange(0, TILE_M)
                rn = pn * BN + tl.arange(0, BN)
                rk = it * TILE_K + tl.arange(0, TILE_K)
                tle.gpu.copy(
                    A + rm[:, None] * K + rk[None, :],
                    slot.a,
                    (TILE_M, TILE_K),
                    mask=(rm[:, None] < M) & (rk[None, :] < K),
                )
                tle.gpu.copy(
                    B + rn[:, None] * K + rk[None, :],
                    slot.b,
                    (BN, TILE_K),
                    mask=(rn[:, None] < N) & (rk[None, :] < K),
                )
            writer.commit(it)
            wait = reader.wait(it)
            partial = tle.gpu.wgmma(wait.slot.a, wait.slot.b, trans_b=True)
            partial = tle.gpu.wgmma_wait(0, partial)
            reader.release(it)
            scale_it = it * TILE_K // GROUP_K
            sa = tl.load(As + rm * SAM + scale_it * SAK, rm < M, other=0)
            if BN <= GROUP_N and GROUP_N % BN == 0:
                sb = tl.load(Bs + (pn * BN // GROUP_N) * SBN + scale_it * SBK)
                acc += partial * (sa * sb)[:, None]
            else:
                sb = tl.load(
                    Bs + (rn // GROUP_N) * SBN + scale_it * SBK, rn < N, other=0
                )
                acc += partial * sa[:, None] * sb[None, :]
        tl.store(
            C + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _block_fp8_matmul_kernel(
    A,
    B,
    As,
    Bs,
    C,
    M,
    N,
    K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsk: tl.constexpr,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    SWAP_AB: tl.constexpr,
    MIN_Y: tl.constexpr,
    Y_SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    GROUP_X: tl.constexpr,
    BLOCK_K: tl.constexpr = 0,
):
    # Keep native FP8 dot / Split-K on a matrix tile when AABS shrinks M.
    # An explicit tile expression is opaque to AABS, so retain its small-shape
    # shrink as well as the matrix minimum selected by dispatch.
    TILE_Y: tl.constexpr = max(MIN_Y, min(BLOCK_Y, Y_SIZE))
    # This tile is a quantization boundary, not an independently tunable axis.
    TILE_K: tl.constexpr = min(GROUP_K, BLOCK_K if BLOCK_K > 0 else 128)
    tl.static_assert(GROUP_K % TILE_K == 0)
    if SWAP_AB:
        X, Y = N, M
    else:
        X, Y = M, N
    pid = tl.program_id(0)
    num_x, num_y = tl.cdiv(X, BLOCK_X), tl.cdiv(Y, TILE_Y)
    group = pid // (GROUP_X * num_y)
    first_x = group * GROUP_X
    group_x = tl.minimum(num_x - first_x, GROUP_X)
    px = first_x + pid % group_x
    py = (pid % (GROUP_X * num_y)) // group_x
    x = px * BLOCK_X + tl.arange(0, BLOCK_X)
    y = py * TILE_Y + tl.arange(0, TILE_Y)
    rk = tl.arange(0, TILE_K)
    split = tl.program_id(1)
    tiles = tl.cdiv(K, TILE_K)
    per_split = tl.cdiv(tiles, SPLIT_K)
    start, end = split * per_split, tl.minimum((split + 1) * per_split, tiles)

    acc = tl.zeros((BLOCK_X, TILE_Y), tl.float32)
    for tile in range(start, end):
        kk = tile * TILE_K + rk
        sk = tile * TILE_K // GROUP_K
        if SWAP_AB:
            lhs_ptr = B + x[:, None] * stride_bn + kk[None, :] * stride_bk
            rhs_ptr = A + kk[:, None] * stride_ak + y[None, :] * stride_am
        else:
            lhs_ptr = A + x[:, None] * stride_am + kk[None, :] * stride_ak
            rhs_ptr = B + kk[:, None] * stride_bk + y[None, :] * stride_bn
        if EVEN_K:
            lhs = tl.load(lhs_ptr, x[:, None] < X, other=0.0)
            rhs = tl.load(rhs_ptr, y[None, :] < Y, other=0.0)
        else:
            lhs = tl.load(lhs_ptr, (x[:, None] < X) & (kk[None, :] < K), other=0.0)
            rhs = tl.load(rhs_ptr, (kk[:, None] < K) & (y[None, :] < Y), other=0.0)
        # FP8 values are exactly representable in FP16. S5000's native FP8
        # dot can lose small products; retain them for FP32 output/partials.
        if C.dtype.element_ty == tl.float32:
            lhs = lhs.to(tl.float16)
            rhs = rhs.to(tl.float16)
        partial = tl.dot(lhs, rhs, out_dtype=tl.float32)
        if SWAP_AB:
            sa = tl.load(As + y * stride_asm + sk * stride_ask, y < M, other=0)
            if BLOCK_X <= GROUP_N and GROUP_N % BLOCK_X == 0:
                sb = tl.load(
                    Bs + (px * BLOCK_X // GROUP_N) * stride_bsn + sk * stride_bsk
                )
                acc += partial * (sa * sb)[None, :]
            else:
                sb = tl.load(
                    Bs + (x // GROUP_N) * stride_bsn + sk * stride_bsk, x < N, other=0
                )
                acc += partial * sb[:, None] * sa[None, :]
        else:
            sa = tl.load(As + x * stride_asm + sk * stride_ask, x < M, other=0)
            if TILE_Y <= GROUP_N and GROUP_N % TILE_Y == 0:
                sb = tl.load(
                    Bs + (py * TILE_Y // GROUP_N) * stride_bsn + sk * stride_bsk
                )
                acc += partial * (sa * sb)[:, None]
            else:
                sb = tl.load(
                    Bs + (y // GROUP_N) * stride_bsn + sk * stride_bsk, y < N, other=0
                )
                acc += partial * sa[:, None] * sb[None, :]
    if SWAP_AB:
        offsets = y[None, :] * N + x[:, None]
    else:
        offsets = x[:, None] * N + y[None, :]
    tl.store(C + split * M * N + offsets, acc, (x[:, None] < X) & (y[None, :] < Y))


_MATMUL_TUNING_KEY = [
    "M",
    "N",
    "K",
    "stride_am",
    "stride_ak",
    "stride_bn",
    "stride_bk",
    "stride_asm",
    "stride_ask",
    "stride_bsn",
    "stride_bsk",
    "GROUP_N",
    "GROUP_K",
    "SWAP_AB",
    "MIN_Y",
]


def _make_matmul_entry(config_name, expand_name):
    return libentry()(
        libtuner(
            configs=runtime.get_tuned_config(config_name),
            key=_MATMUL_TUNING_KEY,
            strategy="default",
            warmup=5,
            rep=20,
            flagtune_op_name="w8a8_block_fp8_matmul",
            flagtune_expand_op_name=expand_name,
            flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
        )(_block_fp8_matmul_kernel)
    )


_block_fp8_matmul_general = _make_matmul_entry(
    "w8a8_block_fp8_matmul_mthreads_general",
    "w8a8_block_fp8_mthreads_general",
)
_block_fp8_matmul_swap = _make_matmul_entry(
    "w8a8_block_fp8_matmul_mthreads_swap",
    "w8a8_block_fp8_mthreads_swap",
)
_block_fp8_matmul_shortk = _make_matmul_entry(
    "w8a8_block_fp8_matmul_mthreads_shortk",
    "w8a8_block_fp8_mthreads_shortk",
)
_block_fp8_matmul_splitk = _make_matmul_entry(
    "w8a8_block_fp8_matmul_mthreads_splitk",
    "w8a8_block_fp8_mthreads_splitk",
)
_block_fp8_matmul_narrow_splitk = _make_matmul_entry(
    "w8a8_block_fp8_matmul_mthreads_narrow_splitk",
    "w8a8_block_fp8_mthreads_narrow_splitk",
)
_block_fp8_matmul_skinny_swap_full = libentry()(
    libtuner(
        configs=runtime.get_tuned_config(
            "w8a8_block_fp8_matmul_mthreads_skinny_swap_full"
        ),
        key=_MATMUL_TUNING_KEY,
        strategy="default",
        warmup=5,
        rep=20,
    )(_block_fp8_matmul_kernel)
)


@libentry()
@triton.jit
def _finish_split_k(P, C, SIZE, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(P + split * SIZE + offsets, offsets < SIZE, other=0)
    tl.store(C + offsets, acc, offsets < SIZE)


@triton.jit
def _gemv_row(
    A,
    As,
    b,
    bs,
    kk,
    kg,
    row: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    a = tl.load(A + row * K + kk, kk < K, other=0.0).to(tl.float32)
    a = tl.reshape(a, (GROUPS, 128))
    sa = tl.load(As + row * SAM + kg * SAK, kg < tl.cdiv(K, 128), other=0.0)
    partial = tl.sum(b * a[None, :, :], 2)
    return tl.sum(partial * (bs * sa[None, :]), 1)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_fp8_matmul_mthreads_vector"),
    key=["M", "N", "K", "SAM", "SAK", "SBN", "SBK", "SPLITS"],
    strategy="default",
    warmup=5,
    rep=20,
)
@triton.jit
def _block_fp8_gemv(
    A,
    B,
    As,
    Bs,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBN: tl.constexpr,
    SBK: tl.constexpr,
    SPLITS: tl.constexpr,
    K_BOUND: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Direct reductions avoid a padded 16-row matrix tile and FP16 workspace.
    # Keep the scale-group boundary even when AABS shrinks a candidate.
    TILE_K: tl.constexpr = max(128, min(BK, K_BOUND))
    GROUPS: tl.constexpr = TILE_K // 128
    nr = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    chunks = tl.cdiv(K, TILE_K)
    count = tl.cdiv(chunks, SPLITS)
    start = split * count
    end = tl.minimum(start + count, chunks)
    acc0 = tl.zeros((BN,), tl.float32)
    if M >= 2:
        acc1 = tl.zeros((BN,), tl.float32)
    if M >= 3:
        acc2 = tl.zeros((BN,), tl.float32)
    if M >= 4:
        acc3 = tl.zeros((BN,), tl.float32)
    if M >= 5:
        acc4 = tl.zeros((BN,), tl.float32)
    if M >= 6:
        acc5 = tl.zeros((BN,), tl.float32)
    if M >= 7:
        acc6 = tl.zeros((BN,), tl.float32)
    if M >= 8:
        acc7 = tl.zeros((BN,), tl.float32)
    for tile in range(start, end):
        kk = tile * TILE_K + tl.arange(0, TILE_K)
        kg = tile * GROUPS + tl.arange(0, GROUPS)
        b = tl.load(
            B + nr[:, None] * K + kk[None, :],
            (nr[:, None] < N) & (kk[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.reshape(b, (BN, GROUPS, 128))
        bs = tl.load(
            Bs + (nr[:, None] // 128) * SBN + kg[None, :] * SBK,
            (nr[:, None] < N) & (kg[None, :] < tl.cdiv(K, 128)),
            other=0.0,
        )
        acc0 += _gemv_row(A, As, b, bs, kk, kg, 0, K, SAM, SAK, GROUPS)
        if M >= 2:
            acc1 += _gemv_row(A, As, b, bs, kk, kg, 1, K, SAM, SAK, GROUPS)
        if M >= 3:
            acc2 += _gemv_row(A, As, b, bs, kk, kg, 2, K, SAM, SAK, GROUPS)
        if M >= 4:
            acc3 += _gemv_row(A, As, b, bs, kk, kg, 3, K, SAM, SAK, GROUPS)
        if M >= 5:
            acc4 += _gemv_row(A, As, b, bs, kk, kg, 4, K, SAM, SAK, GROUPS)
        if M >= 6:
            acc5 += _gemv_row(A, As, b, bs, kk, kg, 5, K, SAM, SAK, GROUPS)
        if M >= 7:
            acc6 += _gemv_row(A, As, b, bs, kk, kg, 6, K, SAM, SAK, GROUPS)
        if M >= 8:
            acc7 += _gemv_row(A, As, b, bs, kk, kg, 7, K, SAM, SAK, GROUPS)
    out = C + split * M * N + nr
    tl.store(out, acc0, nr < N)
    if M >= 2:
        tl.store(out + N, acc1, nr < N)
    if M >= 3:
        tl.store(out + 2 * N, acc2, nr < N)
    if M >= 4:
        tl.store(out + 3 * N, acc3, nr < N)
    if M >= 5:
        tl.store(out + 4 * N, acc4, nr < N)
    if M >= 6:
        tl.store(out + 5 * N, acc5, nr < N)
    if M >= 7:
        tl.store(out + 6 * N, acc6, nr < N)
    if M >= 8:
        tl.store(out + 7 * N, acc7, nr < N)


@triton.jit
def _narrow_n_gemv_kernel(
    A,
    B,
    As,
    Bs,
    C,
    M,
    N,
    K,
    SAM,
    SAK,
    SBN,
    SBK,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program owns one output row and one K split for N<=16."""
    pid = tl.program_id(0)
    row = pid // SPLITS
    split = pid % SPLITS
    nidx = tl.arange(0, BLOCK_N)
    chunks = tl.cdiv(K, BLOCK_K)
    per_split = tl.cdiv(chunks, SPLITS)
    start = split * per_split
    end = tl.minimum((split + 1) * per_split, chunks)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for tile in range(start, end):
        kk = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        kg = tile * (BLOCK_K // 128) + tl.arange(0, BLOCK_K // 128)
        av = tl.load(A + row * K + kk, (row < M) & (kk < K), other=0.0).to(tl.float32)
        sa = tl.load(
            As + row * SAM + kg * SAK, (row < M) & (kg < tl.cdiv(K, 128)), other=0.0
        )
        bv = tl.load(
            B + nidx[:, None] * K + kk[None, :],
            (nidx[:, None] < N) & (kk[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        sb = tl.load(
            Bs + (nidx[:, None] // 128) * SBN + kg[None, :] * SBK,
            (nidx[:, None] < N) & (kg[None, :] < tl.cdiv(K, 128)),
            other=0.0,
        )
        acc += tl.sum(bv * (av * sa)[None, :] * sb, axis=1)
    tl.store(C + split * M * N + row * N + nidx, acc, (row < M) & (nidx < N))


@libentry()
@triton.jit
def _prepare_tle_inputs(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Four FP8 values per aligned word avoid scalar-byte traffic for ragged K.
    tl.static_assert(K % 4 == 0)
    W: tl.constexpr = BLOCK // 4
    KW: tl.constexpr = KP // 4
    pid = tl.program_id(0)
    ma = tl.cdiv(M * KP, BLOCK)
    if pid < ma:
        x = pid * W + tl.arange(0, W)
        src = A.to(tl.pointer_type(tl.uint32))
        rows = M
        base = 0
    else:
        x = (pid - ma) * W + tl.arange(0, W)
        src = B.to(tl.pointer_type(tl.uint32))
        rows = N
        base = M * KP
    row, col = x // KW, x % KW
    bits = tl.load(src + row * (K // 4) + col, (row < rows) & (col < K // 4), other=0)
    if P.dtype.element_ty == tl.float16:
        lane = tl.arange(0, 4)
        byte = (bits[:, None] >> (lane[None, :] * 8)).to(tl.uint8)
        if A.dtype.element_ty == tl.float8e5:
            # E5M2 and FP16 share sign/exponent fields; shifting the raw bits
            # preserves finite values, subnormals, signed zero, Inf and NaN.
            value = (byte.to(tl.uint16) << 8).to(tl.float16, bitcast=True)
        else:
            value = byte.to(tl.float8e4nv, bitcast=True).to(tl.float16)
        tl.store(P + base + x[:, None] * 4 + lane[None, :], value, row[:, None] < rows)
    else:
        tl.store(P.to(tl.pointer_type(tl.uint32)) + base // 4 + x, bits, row < rows)


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
    """Return A @ B.T with per-token K-group and per-weight-block scales.

    Inputs may be strided. Leading A/As dimensions must flatten without a copy.
    Only inference is supported; no autograd implementation is registered.
    """
    if len(block_size) != 2 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain two positive integers")
    group_n, group_k = block_size
    if group_k not in (32, 64, 128, 256):
        raise NotImplementedError("supported K group sizes are 32, 64, 128, 256")
    if A.ndim < 2 or B.ndim != 2 or As.ndim != A.ndim or Bs.ndim != 2:
        raise ValueError("A/As must have rank >= 2 and B/Bs must have rank 2")
    if A.device.type != "musa" or any(t.device != A.device for t in (B, As, Bs)):
        raise ValueError("all inputs must be on the same MUSA device")
    if A.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) or B.dtype != A.dtype:
        raise NotImplementedError("A/B must have the same E4M3FN or E5M2 FP8 dtype")
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        raise NotImplementedError("scales must be FP32")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("output dtype must be FP16, BF16 or FP32")
    if any(t.requires_grad for t in (A, B, As, Bs)):
        raise NotImplementedError("w8a8_block_fp8_matmul is inference-only")
    m = math.prod(A.shape[:-1])
    n, k = B.shape
    if A.shape[-1] != k:
        raise ValueError("incompatible A/B K dimensions")
    if As.shape != A.shape[:-1] + (triton.cdiv(k, group_k),):
        raise ValueError("invalid As shape")
    if Bs.shape != (triton.cdiv(n, group_n), triton.cdiv(k, group_k)):
        raise ValueError("invalid Bs shape")
    # Bound the N dimension for very wide GEMV-like launches.  The MThreads
    # TLE path can otherwise generate an out-of-bounds access for N>65536.
    if n > 65536 or (m < 512 and n > 32768):
        chunk_n = 32768 if m <= 128 else 4096
        chunks = []
        for start in range(0, n, chunk_n):
            end = min(start + chunk_n, n)
            chunks.append(
                w8a8_block_fp8_matmul(
                    A,
                    B[start:end],
                    As,
                    Bs[start // group_n : triton.cdiv(end, group_n)],
                    block_size,
                    output_dtype,
                )
            )
        return torch.cat(chunks, dim=-1)
    c = torch.empty(A.shape[:-1] + (n,), dtype=output_dtype, device=A.device)
    if m == 0 or n == 0:
        return c
    with torch_device_fn.device(A.device):
        if k == 0:
            _finish_split_k[(triton.cdiv(m * n, 256),)](
                c, c, m * n, SPLIT_K=0, BLOCK=256
            )
            return c
        a = _matrix_view(A, m, k)
        a_s = _matrix_view(As, m, As.shape[-1])
        if (
            (m <= 2 or (m <= 8 and n <= 32768) or (m <= 4 and k <= 512))
            and n >= 128
            and k >= 128
            and group_n == group_k == 128
            and (
                n < 1024
                or A.dtype == torch.float8_e5m2
                or output_dtype == torch.float32
                # Direct row reduction is faster for the small-N single-row
                # cases that otherwise fall into the generic split-K matrix
                # path (for example M=1, N=1024/1536, K=4096).
                or (m <= 8 and n <= 2048 and k >= 2048)
                or (k % 16 != 0 and (m == 1 or k < 2048 or k % 4 != 0))
            )
            and a.is_contiguous()
            and B.is_contiguous()
            and torch_device_fn.get_device_capability(A.device) == (3, 1)
        ):
            # Narrow N needs more K parallelism; cap empty work and scratch.
            splits = (
                min(
                    16,
                    max(4, triton.next_power_of_2(triton.cdiv(2048, n))),
                    triton.cdiv(k, 256),
                )
                if k >= 2048
                else 1
            )
            partials = (
                torch.empty((splits, m, n), dtype=torch.float32, device=A.device)
                if splits > 1
                else c
            )
            _block_fp8_gemv[lambda meta: (triton.cdiv(n, meta["BN"]), splits)](
                a,
                B,
                a_s,
                Bs,
                partials,
                m,
                n,
                k,
                a_s.stride(0),
                a_s.stride(1),
                Bs.stride(0),
                Bs.stride(1),
                SPLITS=splits,
                K_BOUND=triton.next_power_of_2(k),
            )
            if splits > 1:
                _finish_split_k[(triton.cdiv(m * n, 256),)](
                    partials,
                    c,
                    m * n,
                    SPLIT_K=splits,
                    BLOCK=256,
                )
            return c
        short_k_requested = (
            m < 512
            and n > 2112
            and k == 256
            and group_n == group_k == 128
            and A.dtype == torch.float8_e4m3fn
            and output_dtype != torch.float32
            and a.is_contiguous()
            and B.is_contiguous()
        )
        # Match the skinny-GEMM shape split.  The wider swap range keeps
        # N parallelism high for medium M; the Split-K range adds K parallelism
        # when N cannot provide enough independent tiles.
        tle_k_compatible = group_k >= 128 or k % 128 == 0
        wide_swap_requested = (
            m < 512
            and n > 2112
            and k >= 1024
            and not short_k_requested
            # TLE supports N scale groups that can be indexed by the 64/128-wide
            # configured tile and K groups that fit a 32/64/128-wide SQMMA tile.
            # Keep the generic skinny path for other layouts.
            and not (
                group_n in (4, 8, 16, 32, 64, 128, 256)
                and group_k in (32, 64, 128, 256)
                and tle_k_compatible
            )
        )
        wide_split_requested = (
            m < 512
            and n < 2112
            and k >= 4096
            and not (
                group_n in (4, 8, 16, 32, 64, 128, 256)
                and group_k in (32, 64, 128, 256)
                and tle_k_compatible
            )
        )
        native_fp8 = A.dtype == torch.float8_e4m3fn and output_dtype != torch.float32
        skinny_ragged = m < 64 and n >= 1024 and k >= 2048 and k % 16 != 0
        prepare_half = (
            not native_fp8
            and k % 4 == 0
            and (
                (
                    m >= 32
                    and (
                        k >= 512
                        or (k >= 256 and max(m, n) >= 128)
                        or (output_dtype == torch.float32 and n >= 1024)
                    )
                )
                or skinny_ragged
            )
        )
        prepare_fp8 = (
            native_fp8
            and k % 16 != 0
            and k % 4 == 0
            and ((m >= 32 and max(m, n) >= 256 and k >= 256) or skinny_ragged)
        )
        native_shape = (
            m >= 64
            or (m <= 16 and n >= 1024 and k % 16 == 0)
            or (16 < m < 64 and n >= 1024 and k >= 2048)
        )
        if (
            not short_k_requested
            and not wide_swap_requested
            and not wide_split_requested
            and HAS_TLE
            and ((native_fp8 and (native_shape or prepare_fp8)) or prepare_half)
            and group_n in (4, 8, 16, 32, 64, 128, 256)
            and group_k in (32, 64, 128, 256)
            and tle_k_compatible
            and n >= 64
            and k >= 128
            # Narrow-N long-K shapes use the dedicated split/swap path.
            and m != 16
            and not (m >= 512 and k <= 512)
            and not (m <= 16 and n < 2112 and k >= 4096)
            # TLE is most useful for small/medium rows.  Wide contiguous
            # ragged-M workloads also use its masked-copy path.
            and (m < 2048 or (k <= 128 and n >= 2048))
            # Complete 64-wide rows can use descriptor copies.  For wide,
            # contiguous workloads below M=2048, masked TLE handles ragged rows.
            and (
                m % 64 == 0
                or m <= 16
                or (
                    m < 2048
                    and n >= 2048
                    and a_s.is_contiguous()
                    and Bs.is_contiguous()
                )
            )
            # WS TLE candidates include BN=128, so N must cover the
            # largest unmasked copy tile (BN=128), not merely BN=64.
            and n % 128 == 0
            # 16-aligned K tails use pointer-based masked copies; non-16-aligned
            # tails remain on the generic masked kernel.
            and k % 16 == 0
            and a.is_contiguous()
            and B.is_contiguous()
            and a.data_ptr() % 16 == 0
            and B.data_ptr() % 16 == 0
            and torch_device_fn.get_device_capability(A.device) == (3, 1)
        ):
            gemm_a, gemm_b, gemm_k = a, B, k
            if prepare_half or prepare_fp8:
                gemm_k = triton.cdiv(k, 128) * 128
                packed = torch.empty(
                    (m + n, gemm_k),
                    dtype=torch.float16 if prepare_half else A.dtype,
                    device=A.device,
                )
                # One linear preparation kernel for both operands. BLOCK=1024
                # is a fixed copy tile, independent of the GEMM tuning space.
                _prepare_tle_inputs[
                    (triton.cdiv(m * gemm_k, 1024) + triton.cdiv(n * gemm_k, 1024),)
                ](a, B, packed, m, n, k, gemm_k, BLOCK=1024, num_warps=4)
                gemm_a, gemm_b = packed[:m], packed[m:]
            use_tme = gemm_k % 16 == 0 and m % 64 == 0 and n % 128 == 0
            if use_tme:
                # Descriptor copies are safe only for complete BM/BN tiles.
                ad = TensorDescriptor.from_tensor(gemm_a, [64, 128])
                bd = TensorDescriptor.from_tensor(gemm_b, [64, 128])
            else:
                # Use pointer arithmetic so the producer can apply M/N masks.
                ad, bd = gemm_a, gemm_b
            _block_fp8_matmul_tle[
                lambda meta: (
                    triton.cdiv(m, max(16, meta["BM"])) * triton.cdiv(n, meta["BN"]),
                )
            ](
                ad,
                bd,
                a_s,
                Bs,
                c,
                m,
                n,
                gemm_k,
                a_s.stride(0),
                a_s.stride(1),
                Bs.stride(0),
                Bs.stride(1),
                GROUP_K=group_k,
                GROUP_N=group_n,
                TILE_K=32 if group_k == 32 else (64 if group_k == 64 else 128),
                HALF=prepare_half,
                USE_TME=use_tme,
            )
            return c
        # Very narrow N with large M is a row-wise reduction problem. The
        # generic matrix split-K path launches oversized 2-D tiles and pays a
        # large partial-tile overhead; use one row and one K split per program.
        if (
            n <= 16
            and m >= 1024
            and group_n == 128
            and group_k == 128
            and a.is_contiguous()
            and B.is_contiguous()
        ):
            narrow_splits = min(16, max(4, triton.cdiv(k, 256)))
            narrow_bn = triton.next_power_of_2(n)
            narrow_partials = torch.empty(
                (narrow_splits, m, n), dtype=torch.float32, device=A.device
            )
            _narrow_n_gemv_kernel[(m * narrow_splits,)](
                a,
                B,
                a_s,
                Bs,
                narrow_partials,
                m,
                n,
                k,
                a_s.stride(0),
                a_s.stride(1),
                Bs.stride(0),
                Bs.stride(1),
                SPLITS=narrow_splits,
                BLOCK_N=narrow_bn,
                BLOCK_K=128,
            )
            _finish_split_k[(triton.cdiv(m * n, 256),)](
                narrow_partials, c, m * n, SPLIT_K=narrow_splits, BLOCK=256
            )
            return c

        split_path_requested = (
            wide_split_requested
            or (m <= 16 and n < 2112 and k >= 2048)
            or (n <= 16 and k >= 2048 and m >= 1024)
        )
        swap = not split_path_requested and (m <= 16 or wide_swap_requested)
        split_k = 4 if split_path_requested else 1
        split_path = split_k > 1
        split_capacity = 20 if split_path else 1
        partials = (
            torch.empty((split_capacity, m, n), dtype=torch.float32, device=A.device)
            if split_k > 1
            else c
        )
        block_k = min(group_k, 128)
        # Swap-AB is used for skinny M. For M<=8, forcing a 16-row tile
        # performs mostly masked work and wastes registers on every N tile.
        # Keep the wider tile for larger M, but use one effective row here.
        min_y = (
            16
            if swap and n >= 16 and (split_k > 1 or output_dtype != torch.float32)
            else 1
        )
        y_size = triton.next_power_of_2(min(m, n))

        def grid(meta):
            x, y = (n, m) if swap else (m, n)
            split_dim = meta["SPLIT_K"] if split_path else 1
            return (
                triton.cdiv(x, meta["BLOCK_X"])
                * triton.cdiv(y, max(min_y, min(meta["BLOCK_Y"], y_size))),
                split_dim,
            )

        if short_k_requested:
            entry = _block_fp8_matmul_shortk
        elif split_path:
            entry = (
                _block_fp8_matmul_narrow_splitk
                if n <= 16 and m >= 1024
                else _block_fp8_matmul_splitk
            )
        elif swap:
            # Keep Default on the compact curated swap space.  The large
            # expansion-only skinny space is reserved for explicit tuning.
            entry = _block_fp8_matmul_swap
        else:
            entry = _block_fp8_matmul_general

        launch = dict(
            GROUP_N=group_n,
            GROUP_K=group_k,
            SWAP_AB=swap,
            MIN_Y=min_y,
            Y_SIZE=y_size,
            EVEN_K=k % block_k == 0,
        )
        if not split_path:
            launch["SPLIT_K"] = 1

        entry[grid](
            a,
            B,
            a_s,
            Bs,
            partials,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            B.stride(0),
            B.stride(1),
            a_s.stride(0),
            a_s.stride(1),
            Bs.stride(0),
            Bs.stride(1),
            **launch,
        )

        if split_path:
            selected = getattr(entry.fn, "best_config", None)
            selected_split_k = (
                selected.kwargs.get("SPLIT_K", split_k)
                if selected is not None
                else split_k
            )
            _finish_split_k[(triton.cdiv(m * n, 256),)](
                partials,
                c,
                m * n,
                SPLIT_K=selected_split_k,
                BLOCK=256,
            )
    return c

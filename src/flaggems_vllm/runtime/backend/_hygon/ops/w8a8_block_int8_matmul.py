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

"""Hygon DCU W8A8 block INT8 matrix multiplication.

Hygon's native INT8 matrix path is separate from the block BMM entry point.
This implementation keeps the 2-D vLLM layout (A[M,K], B[N,K]) and chooses
small, DCU-friendly tiles without changing the existing BMM kernel.
"""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner


def _device_context(device):
    """Avoid an exchange_device call on the common already-current path."""
    device_index = device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    if torch_device_fn.current_device() == device_index:
        return nullcontext()
    return torch_device_fn.device(device)


@triton.jit
def _w8a8_block_int8_small_m_core(
    A,
    B,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CH: tl.constexpr,
):
    tl.static_assert(M <= 8)
    rn = (tl.program_id(0) % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    nl = rn % N
    if N * K >= 2**31:
        nl = nl.to(tl.int64)
    KW: tl.constexpr = 128
    rk = tl.arange(0, 128 * CH)
    rg = tl.arange(0, CH)
    groups = tl.cdiv(K, 128)
    accs = ()
    for row in tl.static_range(BM):
        accs += (tl.zeros((BN, CH), tl.float32),)
    for ki in range(tl.cdiv(K, 128 * CH)):
        kk = ki * 128 * CH + rk
        b = tl.load(B + nl[:, None] * K + kk[None, :], kk[None, :] < K, other=0).to(
            tl.int32
        )
        sk = ki * CH + rg
        if BN <= GN and GN % BN == 0:
            sb = tl.load(
                Bs + (tl.program_id(0) % tl.cdiv(N, BN)) * BN // GN * groups + sk,
                sk < groups,
                other=0,
            ).to(tl.float32)
        else:
            sb = tl.load(
                Bs + nl[:, None] // GN * groups + sk[None, :],
                sk[None, :] < groups,
                other=0,
            ).to(tl.float32)
        updated = ()
        for row in tl.static_range(BM):
            rm = (tl.program_id(0) // tl.cdiv(N, BN)) * BM + row
            ml = rm % M
            if M * K >= 2**31:
                ml = ml.to(tl.int64)
            a = tl.load(A + ml * K + kk, kk < K, other=0).to(tl.int32)
            product = b * a[None, :]
            dot = tl.sum(tl.reshape(product, (BN, CH, KW)), 2).to(tl.float32)
            sa = tl.load(As + ml * groups + sk, sk < groups, other=0).to(tl.float32)
            if BN <= GN and GN % BN == 0:
                scaled = dot * (sa * sb)[None, :]
            else:
                scaled = dot * (sa[None, :] * sb)
            updated += (accs[row] + scaled,)
        accs = updated
    for row in tl.static_range(BM):
        rm = (tl.program_id(0) // tl.cdiv(N, BN)) * BM + row
        tl.store(C + rm * N + rn, tl.sum(accs[row], 1), (rm < M) & (rn < N))


@triton.jit
def _w8a8_block_int8_address_core(
    A,
    B,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GM: tl.constexpr,
    STAGES: tl.constexpr,
    SPLITS: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    if SPLITS > 1:
        C += split.to(tl.int64) * M * N
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    first = pid // (GM * npn) * GM
    size = tl.minimum(npm - first, GM)
    pm = first + pid % (GM * npn) % size
    pn = pid % (GM * npn) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, 128)
    ml = (rm % M).to(tl.int64)
    nl = (rn % N).to(tl.int64)
    groups = tl.cdiv(K, 128)
    ap = A + ml[:, None] * K + rk[None, :] + split * 128
    bp = B + nl[None, :] * K + rk[:, None] + split * 128
    sap = As + ml * groups + split
    sbp = Bs + nl // GN * groups + split
    acc = tl.zeros((BM, BN), tl.float32)
    for kt in tl.range(0, tl.cdiv(groups, SPLITS), num_stages=STAGES):
        sk = split + kt * SPLITS
        kk = sk * 128 + rk
        a = tl.load(ap, kk[None, :] < K, other=0)
        b = tl.load(bp, kk[:, None] < K, other=0)
        sa = tl.load(sap, sk < groups, other=0).to(tl.float32)
        sb = tl.load(sbp, sk < groups, other=0).to(tl.float32)
        acc += tl.dot(a, b, out_dtype=tl.int32).to(tl.float32) * (
            sa[:, None] * sb[None, :]
        )
        ap += SPLITS * 128
        bp += SPLITS * 128
        sap += SPLITS
        sbp += SPLITS
    tl.store(
        C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N)
    )


def _config_key(config):
    return (
        tuple(sorted(config.kwargs.items())),
        config.num_warps,
        config.num_stages,
        config.num_ctas,
    )


_DEFAULT_MATMUL_KEYS = frozenset(
    _config_key(config)
    for config in runtime.get_tuned_config("w8a8_block_int8_matmul_hygon")
)
_DEFAULT_SPLITK_KEYS = frozenset(
    _config_key(config)
    for config in runtime.get_tuned_config("w8a8_block_int8_splitk_hygon")
)


def _keep_matmul_expansion(config, k):
    # Keep the default space as insurance for workloads not yet measured.
    if _config_key(config) in _DEFAULT_MATMUL_KEYS:
        return True
    meta = config.kwargs
    bm, bn, bk = meta["BLOCK_M"], meta["BLOCK_N"], meta["BLOCK_K"]
    schedule = meta.get("SCHEDULE", 0)
    warps, stages = config.num_warps, config.num_stages
    if schedule in (2, 3):
        return warps in (1, 2) and stages == 1
    if bm < 16:
        return warps in (1, 2) and stages in (1, 2)
    if bk == 64:
        return k < 4096 and bm == 16 and warps in (2, 4) and stages in (1, 2)
    if schedule == 1:
        return bm == 32 and bn == 64 and meta["GROUP_M"] == 1 and warps == 4
    if meta["UNROLL"] == 2:
        return bm == 64 and bn == 64 and warps == 4 and stages == 1
    if meta["GROUP_M"] == 32:
        return bm == 128 and bn == 128 and warps == 4 and stages == 1
    if stages == 3:
        return bm == 16 and bn == 64 and warps == 1
    if warps in (1, 2):
        return bm == 16 and bn == 64
    if warps == 8:
        return (
            meta["SWAP_AB"]
            and bm in (32, 64)
            and (stages == 1 or (bm, bn, meta["GROUP_M"]) == (64, 64, 1))
        )
    return warps == 4 and stages in (1, 2)


def _keep_splitk_expansion(config, k):
    if _config_key(config) in _DEFAULT_SPLITK_KEYS:
        return True
    meta = config.kwargs
    bm, bn = meta["BLOCK_M"], meta["BLOCK_N"]
    warps, stages = config.num_warps, config.num_stages
    if meta.get("VECTOR", False):
        return True
    if bm == 128 or warps == 8:
        return False
    if bm == 64 and not (
        bn in (32, 64) and meta["GROUP_M"] == 1 and not meta["SWAP"] and stages == 1
    ):
        return False
    if meta["GROUP_M"] == 8 and not (
        bm in (16, 32)
        and warps == 2
        and (
            (bn == 64 and not meta["SWAP"]) or (bm == 16 and bn == 32 and meta["SWAP"])
        )
    ):
        return False
    # Tiny K must retain at least one nonempty partition.
    if k >= 512 and meta["SPLITS"] < 4:
        return False
    if stages == 2:
        return bm == 32 and bn == 64 and meta["SWAP"] and warps == 4
    return warps in (2, 4)


def _prune_matmul_configs(configs, named_args, **kwargs):
    result = []
    for config in configs:
        meta = config.kwargs
        bm, bn, bk = meta["BLOCK_M"], meta["BLOCK_N"], meta["BLOCK_K"]
        schedule = meta.get("SCHEDULE", 0)
        if schedule in (2, 3):
            # Vector schedules ignore dot geometry; keep one canonical tile.
            if (
                named_args["M"] > 8
                or (bm, bn, bk, meta["GROUP_M"]) != (16, 64, 128, 1)
                or meta["SWAP_AB"]
                or meta["UNROLL"] != 1
            ):
                continue
        elif schedule == 1:
            # The contiguous address core has a fixed 128-element K block.
            if (
                bm < 32
                or bk != 128
                or meta["SWAP_AB"]
                or meta["UNROLL"] != 1
                or meta["GROUP_M"] not in (1, 8)
            ):
                continue
        elif bm < 16:
            if (bn, bk, meta["GROUP_M"], meta["SWAP_AB"], meta["UNROLL"]) != (
                128,
                64,
                8,
                True,
                1,
            ):
                continue
        elif meta["UNROLL"] == 2:
            if (
                bm not in (32, 64)
                or bk != 128
                or meta["SWAP_AB"]
                or meta["GROUP_M"] != 8
            ):
                continue
        elif bk == 64 and (bm != 16 or meta["GROUP_M"] not in (1, 8)):
            continue
        elif meta["GROUP_M"] == 32 and (bm < 64 or bk != 128 or meta["SWAP_AB"]):
            continue
        if schedule < 2 and bm >= 32 and config.num_warps < 4:
            continue
        if _keep_matmul_expansion(config, named_args["K"]):
            result.append(config)
    return result


def _prune_gemv_configs(configs, named_args, **kwargs):
    result = []
    for config in configs:
        meta = config.kwargs
        if meta.get("VECTOR", False):
            # Vector reduction consumes whole quantization blocks; BLOCK_K is
            # ignored by that core, so retain only its canonical value.
            if meta["BLOCK_K"] != 128 or meta["BLOCK_N"] > 64 or config.num_stages != 1:
                continue
        elif meta.get("CHUNKS", 1) != 1 or meta["BLOCK_N"] < 32:
            continue
        result.append(config)
    return result


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_matmul_hygon"),
    key=["M", "N", "K", "GROUP_N", "GROUP_K"],
    strategy="default",
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _prune_matmul_configs},
    flagtune_op_name="w8a8_block_int8_matmul",
    flagtune_expand_op_name="w8a8_block_int8_general",
)
@triton.jit
def _w8a8_block_int8_matmul_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
    ACC_FP32: tl.constexpr,
    UNROLL: tl.constexpr,
    SWAP_AB: tl.constexpr,
    SCHEDULE: tl.constexpr = 0,
):
    # Vector schedules keep their geometry independent of dot tile adjustment.
    if SCHEDULE >= 2:
        _w8a8_block_int8_small_m_core(
            A,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            GROUP_N,
            2,
            8 if SCHEDULE == 2 else 16,
            1 if SCHEDULE == 2 else 8,
        )
        return
    if SCHEDULE == 1:
        _w8a8_block_int8_address_core(
            A, B, C, As, Bs, M, N, K, GROUP_N, BLOCK_M, BLOCK_N, GROUP_M, 2
        )
        return
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_id = pid // (GROUP_M * num_pid_n)
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    local_pid = pid % (GROUP_M * num_pid_n)
    pid_m = first_pid_m + local_pid % group_size_m
    pid_n = local_pid // group_size_m

    # Compile-time shapes simplify tile/tail addressing and short K loops.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < M
    mask_n = rn < N
    # Match vLLM's tail strategy: invalid M/N lanes read a wrapped, valid
    # element and are masked only on the final store.  This removes repeated
    # M/N predicates from the A/B and scale loads while preserving C exactly.
    rm_load = rm % M
    rn_load = rn % N
    acc_dtype: tl.constexpr = tl.float32 if ACC_FP32 else tl.bfloat16
    if SWAP_AB:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=acc_dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k_tile in tl.range(
        0,
        tl.cdiv(K, BLOCK_K),
        loop_unroll_factor=UNROLL,
    ):
        kk = k_tile * BLOCK_K + rk
        mask_k = kk < K
        if SWAP_AB:
            a = tl.load(
                A + kk[:, None] * stride_ak + rm_load[None, :] * stride_am,
                mask=mask_k[:, None],
                other=0,
            )
            b = tl.load(
                B + rn_load[:, None] * stride_bn + kk[None, :] * stride_bk,
                mask=mask_k[None, :],
                other=0,
            )
        else:
            a = tl.load(
                A + rm_load[:, None] * stride_am + kk[None, :] * stride_ak,
                mask=mask_k[None, :],
                other=0,
            )
            b = tl.load(
                B + kk[:, None] * stride_bk + rn_load[None, :] * stride_bn,
                mask=mask_k[:, None],
                other=0,
            )
        scale_k = (k_tile * BLOCK_K) // GROUP_K
        a_scale = tl.load(
            As + rm_load * stride_asm + scale_k * stride_ask,
        ).to(tl.float32)
        if BLOCK_N <= GROUP_N and GROUP_N % BLOCK_N == 0:
            b_scale_scalar = tl.load(
                Bs + (pid_n * BLOCK_N // GROUP_N) * stride_bsn + scale_k * stride_bsk
            ).to(tl.float32)
            if SWAP_AB:
                partial = tl.dot(b, a, out_dtype=tl.int32).to(tl.float32)
                scaled = (partial * (b_scale_scalar * a_scale[None, :])).to(acc_dtype)
            else:
                partial = tl.dot(a, b, out_dtype=tl.int32).to(tl.float32)
                scaled = (partial * (a_scale[:, None] * b_scale_scalar)).to(acc_dtype)
        else:
            b_scale = tl.load(
                Bs + (rn_load // GROUP_N) * stride_bsn + scale_k * stride_bsk,
            ).to(tl.float32)
            if SWAP_AB:
                partial = tl.dot(b, a, out_dtype=tl.int32).to(tl.float32)
                scaled = (partial * (b_scale[:, None] * a_scale[None, :])).to(acc_dtype)
            else:
                partial = tl.dot(a, b, out_dtype=tl.int32).to(tl.float32)
                scaled = (partial * (a_scale[:, None] * b_scale[None, :])).to(acc_dtype)
        acc = (acc + scaled).to(acc_dtype)

    if SWAP_AB:
        tl.store(
            C + rn[:, None] * stride_cn + rm[None, :] * stride_cm,
            acc.to(C.dtype.element_ty),
            mask=mask_n[:, None] & mask_m[None, :],
        )
    else:
        tl.store(
            C + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            acc.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


@triton.jit
def _w8a8_block_int8_gemv_reduce_core(
    A,
    B,
    C,
    As,
    Bs,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    # Reduce each 128-element quantization block in INT32 before applying
    # its FP32 scale; CHUNKS groups share one coalesced K load.
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rnl = rn % N
    if N * K >= 2**31:
        rnl = rnl.to(tl.int64)
    rk = tl.arange(0, 128 * CHUNKS)
    rg = tl.arange(0, CHUNKS)
    groups = tl.cdiv(K, 128)
    acc = tl.zeros((BLOCK_N, CHUNKS), tl.float32)
    for ki in range(tl.cdiv(K, 128 * CHUNKS)):
        kk = ki * 128 * CHUNKS + rk
        a = tl.load(A + kk, kk < K, other=0).to(tl.int32)
        b = tl.load(B + rnl[:, None] * K + kk[None, :], kk[None, :] < K, other=0).to(
            tl.int32
        )
        dot = tl.sum(tl.reshape(b * a[None, :], (BLOCK_N, CHUNKS, 128)), 2).to(
            tl.float32
        )
        sk = ki * CHUNKS + rg
        sa = tl.load(As + sk, sk < groups, other=0).to(tl.float32)
        if BLOCK_N <= GROUP_N and GROUP_N % BLOCK_N == 0:
            sb = tl.load(
                Bs + (tl.program_id(0) * BLOCK_N // GROUP_N) * groups + sk,
                sk < groups,
                other=0,
            ).to(tl.float32)
            acc += dot * (sa * sb)[None, :]
        else:
            sb = tl.load(
                Bs + (rnl[:, None] // GROUP_N) * groups + sk[None, :],
                sk[None, :] < groups,
                other=0,
            ).to(tl.float32)
            acc += dot * (sa[None, :] * sb)
    tl.store(C + rn, tl.sum(acc, 1), rn < N)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_gemv_hygon"),
    key=["N", "K", "GROUP_N", "GROUP_K"],
    strategy="default",
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    flagtune_op_name="w8a8_block_int8_matmul",
    flagtune_expand_op_name="w8a8_block_int8_gemv_block",
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
)
@triton.jit
def _w8a8_block_int8_gemv_kernel(
    A,
    B,
    C,
    As,
    Bs,
    N: tl.constexpr,
    K: tl.constexpr,
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
    VECTOR: tl.constexpr = False,
    CHUNKS: tl.constexpr = 1,
):
    """Tune native INT8 dot and quantization-block vector reductions."""
    if VECTOR:
        _w8a8_block_int8_gemv_reduce_core(
            A, B, C, As, Bs, N, K, GROUP_N, BLOCK_N, CHUNKS
        )
    else:
        """One-row INT8 matmul with a vector dot layout."""
        pid_n = tl.program_id(0)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        mask_n = rn < N
        rn_load = rn % N
        if N * K >= 2**31:
            rn_load = rn_load.to(tl.int64)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_tile in tl.range(0, tl.cdiv(K, BLOCK_K)):
            kk = k_tile * BLOCK_K + rk
            mask_k = kk < K
            a = tl.load(A + kk * stride_ak, mask=mask_k, other=0)
            b = tl.load(
                B + rn_load[:, None] * stride_bn + kk[None, :] * stride_bk,
                mask=mask_k[None, :],
                other=0,
            )
            partial = tl.dot(a[None, :], tl.trans(b), out_dtype=tl.int32).to(tl.float32)
            scale_k = (k_tile * BLOCK_K) // GROUP_K
            a_scale = tl.load(As + scale_k * stride_ask).to(tl.float32)
            b_scale = tl.load(
                Bs + (rn_load // GROUP_N) * stride_bsn + scale_k * stride_bsk,
            ).to(tl.float32)
            acc += tl.reshape(partial, (BLOCK_N,)) * (a_scale * b_scale)
        if C.dtype.element_ty == tl.float16:
            result = acc.to(tl.float16)
        elif C.dtype.element_ty == tl.bfloat16:
            result = acc.to(tl.bfloat16)
        else:
            result = acc
        tl.store(C + rn, result, mask=mask_n)


@triton.jit
def _w8a8_block_int8_splitk_core(
    A,
    B,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
    ACC_FP32: tl.constexpr,
    UNROLL: tl.constexpr,
    SWAP_AB: tl.constexpr,
    SPLITS: tl.constexpr = 1,
):
    if SPLITS == 1:
        split = 0
    else:
        split = tl.program_id(1)
        C += split.to(tl.int64) * M * N
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_id = pid // (GROUP_M * num_pid_n)
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    local_pid = pid % (GROUP_M * num_pid_n)
    pid_m = first_pid_m + local_pid % group_size_m
    pid_n = local_pid // group_size_m

    # Compile-time shapes simplify tile/tail addressing and short K loops.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < M
    mask_n = rn < N
    # Match vLLM's tail strategy: invalid M/N lanes read a wrapped, valid
    # element and are masked only on the final store.  This removes repeated
    # M/N predicates from the A/B and scale loads while preserving C exactly.
    rm_load = rm % M
    rn_load = rn % N
    if M * K >= 2**31:
        rm_load = rm_load.to(tl.int64)
    if N * K >= 2**31:
        rn_load = rn_load.to(tl.int64)
    acc_dtype: tl.constexpr = tl.float32 if ACC_FP32 else tl.bfloat16
    if SWAP_AB:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=acc_dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k_tile in tl.range(
        split,
        tl.cdiv(K, BLOCK_K),
        SPLITS,
        loop_unroll_factor=UNROLL,
    ):
        kk = k_tile * BLOCK_K + rk
        mask_k = kk < K
        if SWAP_AB:
            a = tl.load(
                A + kk[:, None] * stride_ak + rm_load[None, :] * stride_am,
                mask=mask_k[:, None],
                other=0,
            )
            b = tl.load(
                B + rn_load[:, None] * stride_bn + kk[None, :] * stride_bk,
                mask=mask_k[None, :],
                other=0,
            )
        else:
            a = tl.load(
                A + rm_load[:, None] * stride_am + kk[None, :] * stride_ak,
                mask=mask_k[None, :],
                other=0,
            )
            b = tl.load(
                B + kk[:, None] * stride_bk + rn_load[None, :] * stride_bn,
                mask=mask_k[:, None],
                other=0,
            )
        scale_k = (k_tile * BLOCK_K) // GROUP_K
        a_scale = tl.load(
            As + rm_load * stride_asm + scale_k * stride_ask,
        ).to(tl.float32)
        if BLOCK_N <= GROUP_N and GROUP_N % BLOCK_N == 0:
            b_scale_scalar = tl.load(
                Bs + (pid_n * BLOCK_N // GROUP_N) * stride_bsn + scale_k * stride_bsk
            ).to(tl.float32)
            if SWAP_AB:
                partial = tl.dot(b, a, out_dtype=tl.int32).to(tl.float32)
                scaled = (partial * (b_scale_scalar * a_scale[None, :])).to(acc_dtype)
            else:
                partial = tl.dot(a, b, out_dtype=tl.int32).to(tl.float32)
                scaled = (partial * (a_scale[:, None] * b_scale_scalar)).to(acc_dtype)
        else:
            b_scale = tl.load(
                Bs + (rn_load // GROUP_N) * stride_bsn + scale_k * stride_bsk,
            ).to(tl.float32)
            if SWAP_AB:
                partial = tl.dot(b, a, out_dtype=tl.int32).to(tl.float32)
                scaled = (partial * (b_scale[:, None] * a_scale[None, :])).to(acc_dtype)
            else:
                partial = tl.dot(a, b, out_dtype=tl.int32).to(tl.float32)
                scaled = (partial * (a_scale[:, None] * b_scale[None, :])).to(acc_dtype)
        acc = (acc + scaled).to(acc_dtype)

    if SWAP_AB:
        tl.store(
            C + rn[:, None] * stride_cn + rm[None, :] * stride_cm,
            acc.to(C.dtype.element_ty),
            mask=mask_n[:, None] & mask_m[None, :],
        )
    else:
        tl.store(
            C + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            acc.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


# The last workspace element records the active split count.  The reduction
# consumes only initialized slices, even when LibTuner selects fewer splits.
_MAX_SPLITS = tl.constexpr(64)


@libentry()
@triton.jit
def _w8a8_block_int8_reduce_splitk(
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLITS: tl.constexpr = 0,
    CAPACITY: tl.constexpr = 64,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if SPLITS:
        split = tl.arange(0, triton.next_power_of_2(SPLITS))
        active_splits = SPLITS
    else:
        split = tl.arange(0, _MAX_SPLITS)
        active_splits = tl.load(P + tl.cast(M, tl.int64) * N * CAPACITY).to(tl.int32)
    # A single split writes C directly; its reduction launch is a no-op.
    if active_splits > 1:
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
    if exception is None and args["SPLITS"] > 1:
        _w8a8_block_int8_reduce_splitk[(triton.cdiv(args["M"] * args["N"], 64),)](
            args["P"],
            args["C"],
            args["M"],
            args["N"],
            64,
            SPLITS=args["SPLITS"],
            CAPACITY=args["CAPACITY"],
            num_warps=4,
        )


def _prune_splitk_configs(configs, named_args, **kwargs):
    result = []
    args = {**named_args, **kwargs}
    groups = triton.cdiv(args["K"], args["GROUP_K"])
    has_nonempty_split = any(
        not config.kwargs.get("VECTOR", False) and config.kwargs["SPLITS"] <= groups
        for config in configs
    )
    for config in configs:
        meta = config.kwargs
        bm, bn = meta["BLOCK_M"], meta["BLOCK_N"]
        if meta.get("VECTOR", False):
            if (
                named_args["M"] != 1
                or meta["SPLITS"] != 1
                or bm != 1
                or bn not in (8, 16, 32)
                or meta["SWAP"]
                or meta["GROUP_M"] != 1
                or config.num_stages != 1
            ):
                continue
        else:
            if bm < 16 or bn < 32 or meta.get("CHUNKS", 1) != 1:
                continue
            if bm == 128 and (
                bn != 64 or meta["SPLITS"] != 1 or meta["SWAP"] or meta["GROUP_M"] != 8
            ):
                continue
            if has_nonempty_split and meta["SPLITS"] > groups:
                continue
        if _keep_splitk_expansion(config, args["K"]):
            result.append(config)
    return result


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_splitk_hygon"),
    key=["M", "N", "K", "GROUP_N", "GROUP_K"],
    strategy="default",
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    post_hook=_splitk_post_hook,
    prune_configs_by={"early_config_prune": _prune_splitk_configs},
    flagtune_op_name="w8a8_block_int8_matmul",
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
    VECTOR: tl.constexpr = False,
    CHUNKS: tl.constexpr = 1,
    CAPACITY: tl.constexpr = 64,
):
    """Tune GEMM with its required epilogue using device graphs."""
    tl.static_assert(SPLITS == 1 or SPLITS <= CAPACITY)
    if VECTOR:
        tl.static_assert(M == 1 and SPLITS == 1)
        _w8a8_block_int8_gemv_reduce_core(
            A, B, C, As, Bs, N, K, GROUP_N, BLOCK_N, CHUNKS
        )
    else:
        groups = tl.cdiv(K, GROUP_K)
        destination = C if SPLITS == 1 else P
        _w8a8_block_int8_splitk_core(
            A,
            B,
            destination,
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
            SWAP_AB=SWAP,
            ACC_FP32=True,
            UNROLL=1,
            SPLITS=SPLITS,
        )
    if CAPACITY > 0 and tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.store(P + tl.cast(M, tl.int64) * N * CAPACITY, SPLITS)


# Plans are keyed by the tuner configuration-list identity.  Keeping the
# list itself alive prevents identity reuse when tests or callers replace it.
_splitk_plans = {}
_splitk_dummy_buffers = {}


def _splitk_matmul(A, B, As, Bs, C, M, N, K, group_n, group_k):
    # Switch spaces before planning the workspace; Expand can choose more
    # splits than a cached Default plan can hold.
    _w8a8_block_int8_kernel_splitk._apply_flagtune()
    configs = _w8a8_block_int8_kernel_splitk.fn.configs
    key = (
        A.device,
        M,
        N,
        K,
        group_n,
        group_k,
        As.dtype,
        Bs.dtype,
        C.dtype,
        id(configs),
    )
    plan = _splitk_plans.get(key)
    capacity = _MAX_SPLITS.value if plan is None else plan[1]
    if capacity == 1:
        # This tensor is never read or written by the single-split binary.
        capacity = 0
        P = _splitk_dummy_buffers.get(A.device)
        if P is None:
            P = torch.empty((1,), device=A.device, dtype=torch.float32)
            _splitk_dummy_buffers[A.device] = P
    else:
        P = torch.empty((capacity * M * N + 1,), device=A.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        meta["SPLITS"],
    )
    with _device_context(A.device):
        _, meta = _w8a8_block_int8_kernel_splitk[grid](
            A,
            B,
            P,
            C,
            As,
            Bs,
            M,
            N,
            K,
            GROUP_N=group_n,
            GROUP_K=group_k,
            CAPACITY=capacity,
        )
        splits = meta["SPLITS"]
        _splitk_plans[key] = (configs, splits)
        if splits > 1:
            _w8a8_block_int8_reduce_splitk[(triton.cdiv(M * N, 64),)](
                P, C, M, N, 64, SPLITS=splits, CAPACITY=capacity, num_warps=4
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
    """Compute ``A @ B.T`` for block-scaled INT8 inputs on Hygon DCU."""
    if A.ndim < 2 or B.ndim != 2 or As.ndim != A.ndim or Bs.ndim != 2:
        raise ValueError("A must be rank >= 2, B/Bs rank 2, and As match A rank")
    if any(t.device != A.device for t in (B, As, Bs)) or A.device.type != "cuda":
        raise ValueError("all inputs must be on the same Hygon device")
    if As.requires_grad or Bs.requires_grad:
        raise NotImplementedError("w8a8_block_int8_matmul is inference-only")
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise TypeError("A and B must be int8")
    supported_scale_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if As.dtype not in supported_scale_dtypes or Bs.dtype not in supported_scale_dtypes:
        raise TypeError("scales must be floating point")
    if len(block_size) != 2 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain two positive integers")
    group_n, group_k = block_size
    if group_n < 16 or group_n & (group_n - 1) or group_k != 128:
        raise NotImplementedError(
            "Hygon requires power-of-two block_n >= 16 and block_k=128"
        )
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("unsupported output dtype")
    if not all(t.is_contiguous() for t in (A, B, As, Bs)):
        raise NotImplementedError("inputs must be contiguous")

    K = A.shape[-1]
    M = 1
    for dim in A.shape[:-1]:
        M *= dim
    N = B.shape[0]
    groups_k = triton.cdiv(K, group_k)
    if B.shape[1] != K or As.shape != A.shape[:-1] + (groups_k,):
        raise ValueError("A/B or activation scale shape mismatch")
    if Bs.shape != (triton.cdiv(N, group_n), groups_k):
        raise ValueError("weight scale shape mismatch")
    C = torch.empty(A.shape[:-1] + (N,), dtype=output_dtype, device=A.device)
    if M == 0 or N == 0:
        return C

    # Split K only when spatial tiling provides limited parallelism, and
    # bound the FP32 workspace to 16 MiB.  LibTuner selects the split count
    # and dot layout using complete GEMM + reduction timings.
    spatial_tiles = triton.cdiv(M, 16) * triton.cdiv(N, 64)
    if K >= 4096 and M * N <= 65536 and spatial_tiles <= 128:
        return _splitk_matmul(A, B, As, Bs, C, M, N, K, group_n, group_k)

    # A single activation row is a GEMV.  Dispatching the vector layout avoids
    # carrying masked M lanes through every INT8 dot and mirrors the dedicated
    # small-M paths used by the other backends.
    if M == 1:
        gemv_grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
        with _device_context(A.device):
            _w8a8_block_int8_gemv_kernel[gemv_grid](
                A,
                B,
                C,
                As,
                Bs,
                N,
                K,
                A.stride(-1),
                B.stride(-1),
                B.stride(-2),
                As.stride(-1),
                Bs.stride(-2),
                Bs.stride(-1),
                GROUP_N=group_n,
                GROUP_K=group_k,
            )
        return C

    if N == 1:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_N"]),)
        with _device_context(A.device):
            _w8a8_block_int8_gemv_kernel[grid](
                B,
                A,
                C,
                Bs,
                As,
                M,
                K,
                B.stride(-1),
                A.stride(-1),
                A.stride(-2),
                Bs.stride(-1),
                As.stride(-2),
                As.stride(-1),
                GROUP_N=1,
                GROUP_K=group_k,
            )
        return C

    grid = lambda meta: (
        (
            triton.cdiv(M, 2) * triton.cdiv(N, 8 if meta["SCHEDULE"] == 2 else 16)
            if meta["SCHEDULE"] >= 2
            else triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])
        ),
    )
    with _device_context(A.device):
        _w8a8_block_int8_matmul_kernel[grid](
            A,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            A.stride(-2),
            A.stride(-1),
            B.stride(-1),
            B.stride(-2),
            C.stride(-2),
            C.stride(-1),
            As.stride(-2),
            As.stride(-1),
            Bs.stride(-2),
            Bs.stride(-1),
            GROUP_N=group_n,
            GROUP_K=group_k,
            # vLLM's reference accumulates every scaled K block in FP32 even
            # when the requested output is FP16/BF16.  Keeping this explicit
            # avoids per-tile BF16 rounding while preserving the output dtype.
            ACC_FP32=True,
        )
    return C

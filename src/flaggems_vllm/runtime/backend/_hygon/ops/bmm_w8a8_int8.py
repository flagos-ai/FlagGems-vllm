# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._hygon.ops.w8a8_block_int8_bmm import (
    _get_int8_config as get_int8_config,
)
from flaggems_vllm.runtime.backend._hygon.ops.w8a8_block_int8_bmm import zero_bmm_kernel
from flaggems_vllm.utils import libentry

MAX_INT32_REDUCTION = (2**31 - 1) // (128 * 128)


@libentry()
@triton.jit
def bmm_channel_int8_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALIGNED: tl.constexpr,
    TILE_K: tl.constexpr,
    UNROLL: tl.constexpr = 1,
    SWAP_AB: tl.constexpr = False,
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
    if SWAP_AB:
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.int32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    for kb in tl.range(
        tl.cdiv(K, TILE_K),
        loop_unroll_factor=((K + TILE_K - 1) // TILE_K if K <= 256 else UNROLL),
    ):
        kk = kb * TILE_K + rk
        ap = A + batch * SA[0] + rm[:, None] * SA[1] + kk[None, :] * SA[2]
        wp = W + batch * SW[0] + kk[:, None] * SW[1] + rn[None, :] * SW[2]
        if ALIGNED:
            a = tl.load(ap)
            w = tl.load(wp)
        else:
            a = tl.load(ap, (rm[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(wp, (kk[:, None] < K) & (rn[None, :] < N), other=0)
        if SWAP_AB:
            acc = tl.dot(tl.trans(w), tl.trans(a), acc, out_dtype=tl.int32)
        else:
            acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    if SWAP_AB:
        acc = tl.trans(acc)
    a_scale = tl.load(AS + batch * SAS[0] + rm * SAS[1], rm < M, other=0)
    w_scale = tl.load(WS + batch * SWS[0] + rn * SWS[2], rn < N, other=0)
    acc = acc.to(tl.float32) * a_scale[:, None]
    acc = acc * w_scale[None, :]
    op = O + batch * SO[0] + rm[:, None] * SO[1] + rn[None, :] * SO[2]
    if ALIGNED:
        tl.store(op, acc)
    else:
        tl.store(op, acc, (rm[:, None] < M) & (rn[None, :] < N))


def bmm_w8a8_int8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Channel-scaled INT8 BMM with INT32 accumulation on Hygon.

    A[B,M,K] is contiguous; W[B,K,N] is a transpose view of [B,N,K].
    FP32 scales have shapes [B,M] / [B,N], optionally [B,M,1] / [B,1,N].
    Inputs are prequantized; batch broadcasting and autograd are unsupported.
    """
    if activation.ndim != 3 or weight.ndim != 3:
        raise ValueError("expected activation[B,M,K] and weight[B,K,N]")
    batch, rows, reduction = activation.shape
    columns = weight.shape[2]
    if weight.shape[:2] != (batch, reduction):
        raise ValueError("batch and reduction dimensions must match")
    if activation.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("activation and weight must be signed INT8 tensors")
    if activation.device.type != "cuda" or any(
        tensor.device != activation.device
        for tensor in (weight, activation_scale, weight_scale)
    ):
        raise ValueError("inputs and scales must be on the same Hygon device")
    if not activation.is_contiguous() or not weight.transpose(1, 2).is_contiguous():
        raise ValueError("expected contiguous activation and column-major weight")
    if activation_scale.shape not in ((batch, rows), (batch, rows, 1)):
        raise ValueError("activation_scale must have shape [B,M] or [B,M,1]")
    if weight_scale.shape not in ((batch, columns), (batch, 1, columns)):
        raise ValueError("weight_scale must have shape [B,N] or [B,1,N]")
    if any(
        scale.dtype != torch.float32 or not scale.is_contiguous()
        for scale in (activation_scale, weight_scale)
    ):
        raise TypeError("scales must be contiguous FP32 tensors")
    if any(scale.requires_grad for scale in (activation_scale, weight_scale)):
        raise NotImplementedError("bmm_w8a8_int8 is forward-only")
    if reduction > MAX_INT32_REDUCTION:
        raise NotImplementedError("reduction may overflow signed INT32 accumulation")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("output dtype must be bfloat16, float16 or float32")
    if out is None:
        out = torch.empty(
            (batch, rows, columns), dtype=out_dtype, device=activation.device
        )
    elif (
        out.shape != (batch, rows, columns)
        or out.dtype != out_dtype
        or out.device != activation.device
        or not out.is_contiguous()
        or out.requires_grad
    ):
        raise ValueError("out must be contiguous with matching shape, dtype and device")
    elif out.numel() and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (activation, weight, activation_scale, weight_scale)
    ):
        raise ValueError("out must not share storage with an input")
    if out.numel() == 0:
        return out
    with torch_device_fn.device(activation.device):
        if reduction == 0:
            zero_bmm_kernel[(triton.cdiv(out.numel(), 256),)](
                out, out.numel(), rows, columns, out.stride(), BLOCK_SIZE=256
            )
            return out
        config = get_int8_config(batch, rows, columns, reduction)
        block_m, block_n = config["BLOCK_M"], config["BLOCK_N"]
        block_k = config["TILE_K"] or min(
            128, max(32, triton.next_power_of_2(reduction))
        )
        bmm_channel_int8_kernel[
            (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n), batch)
        ](
            activation,
            weight,
            activation_scale,
            weight_scale,
            out,
            rows,
            columns,
            reduction,
            activation.stride(),
            weight.stride(),
            (activation_scale.stride(0), activation_scale.stride(1), 0),
            (weight_scale.stride(0), 0, weight_scale.stride(-1)),
            out.stride(),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            ALIGNED=rows % block_m == 0
            and columns % block_n == 0
            and reduction % block_k == 0,
            TILE_K=block_k,
            GROUP_M=config["GROUP_M"],
            UNROLL=config["UNROLL"],
            SWAP_AB=config["SWAP_AB"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
    return out

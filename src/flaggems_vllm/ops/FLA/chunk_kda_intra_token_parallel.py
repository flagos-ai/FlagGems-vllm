# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# Token-parallel formulation adapted from flash-linear-attention:
# fla/ops/kda/chunk_intra_token_parallel.py at be38e04703b9.
# Upstream portions retain the following MIT license.
#
# MIT License
#
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Token-parallel diagonal sub-blocks for chunk Kimi Delta Attention."""

import logging
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

__all__ = ["chunk_kda_fwd_intra_token_parallel"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("chunk_kda_fwd_intra_token_parallel"),
    key=["H", "HV", "K", "BT", "BC", "IS_VARLEN"],
    use_cuda_graph=False,
)
@triton.jit(do_not_specialize=["T", "N"])
def _chunk_kda_intra_token_parallel_kernel(
    Q,
    K_ptr,
    G,
    Beta,
    Aqk,
    Akk,
    Cu,
    scale,
    T,
    N,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BH: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1).to(tl.int64) * BH + tl.arange(0, BH)
    if IS_VARLEN:
        # Upper-bound search also handles empty sequences between real tokens.
        if token >= tl.load(Cu + N).to(tl.int64):
            return
        left = 0
        right = N
        while left < right:
            mid = (left + right) // 2
            end = tl.load(Cu + mid + 1).to(tl.int64)
            if token < end:
                right = mid
            else:
                left = mid + 1
        begin = tl.load(Cu + left).to(tl.int64)
        local_t = token - begin
    else:
        local_t = token % T
        begin = token - local_t

    channels = tl.arange(0, BK)
    qk_heads = heads // (HV // H)
    valid = (heads[:, None] < HV) & (channels[None, :] < K)
    qk_offset = qk_heads[:, None] * K + channels[None, :]
    g_offset = heads[:, None] * K + channels[None, :]
    q_i = tl.load(Q + token * H * K + qk_offset, valid, 0).to(tl.float32)
    k_i = tl.load(K_ptr + token * H * K + qk_offset, valid, 0).to(tl.float32)
    g_i = tl.load(G + token * HV * K + g_offset, valid, 0).to(tl.float32)
    beta_i = tl.load(Beta + token * HV + heads, heads < HV, 0).to(tl.float32)
    k_beta = k_i * beta_i[:, None]

    chunk_begin = (local_t // BT) * BT
    sub_begin = chunk_begin + ((local_t % BT) // BC) * BC
    for j in range(sub_begin, local_t + 1):
        other = begin + j
        k_j = tl.load(K_ptr + other * H * K + qk_offset, valid, 0).to(tl.float32)
        g_j = tl.load(G + other * HV * K + g_offset, valid, 0).to(tl.float32)
        weighted_key = tl.where(valid, k_j * tl.exp2(g_i - g_j), 0.0)
        qk = tl.sum(q_i * weighted_key, axis=1) * scale
        kk = tl.sum(k_beta * weighted_key, axis=1) * tl.where(j < local_t, 1.0, 0.0)
        tl.store(Aqk + (token * HV + heads) * BT + j % BT, qk, heads < HV)
        tl.store(Akk + (token * HV + heads) * BC + j - sub_begin, kk, heads < HV)


def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write causal diagonal sub-blocks, returning the original output buffers.

    q/k are [B,T,H,K], gk is [B,T,HV,K], beta is [B,T,HV],
    Aqk is [B,T,HV,BT], and Akk is [B,T,HV,BC]. gk contains already
    accumulated base-2 gates; no gate activation or normalization is performed.
    HV must be a positive multiple of H. All tensors must be contiguous on
    one accelerator, with float16, bfloat16 or float32 elements (q/k must
    share a dtype). Accumulation is float32. K is in [1,256].

    For each sequence-local token i, only j in its current BC sub-block with
    j <= i is written. Aqk uses column j % BT; Akk uses the sub-block-local
    column. Its finite-input diagonal is zero. All other elements remain
    unchanged. NaN/Inf follow the underlying floating-point arithmetic,
    including NaN from an invalid product on the diagonal.

    Packed varlen requires B=1 and a contiguous device int32/int64
    cu_seqlens: nondecreasing, starting at zero, ending at T. Empty sequences
    are allowed. These data-dependent preconditions are the caller's
    responsibility; this wrapper does not synchronize to inspect them.

    Noncontiguous layouts, overlapping outputs/inputs, autograd and Graph
    mode are unsupported. No CPU or external-library fallback is used.
    """
    logger.debug("GEMS CHUNK_KDA_FWD_INTRA_TOKEN_PARALLEL")
    tensors = (q, k, gk, beta, Aqk, Akk)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("q, k, gk, beta, Aqk and Akk must be tensors")
    if use_graph:
        raise NotImplementedError("Graph mode is not supported")
    if q.device.type == "cpu":
        raise NotImplementedError(
            "chunk_kda_fwd_intra_token_parallel requires an accelerator"
        )
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(
        tensor.layout != torch.strided or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise NotImplementedError("only contiguous strided tensors are supported")
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if any(tensor.dtype not in dtypes for tensor in tensors):
        raise NotImplementedError("only float16, bfloat16 and float32 are supported")
    if q.dtype != k.dtype:
        raise ValueError("q and k must have the same dtype")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError(
            "only forward inference without autograd is supported"
        )
    if q.ndim != 4 or k.shape != q.shape or gk.ndim != 4:
        raise ValueError(
            "q/k must have equal [B,T,H,K] shapes and gk must be rank four"
        )
    B, T, H, K = q.shape
    HV = gk.shape[2]
    if H <= 0 or HV <= 0 or HV % H:
        raise ValueError("HV must be a positive multiple of positive H")
    if not 1 <= K <= 256:
        raise NotImplementedError("K must be in [1,256]")
    if gk.shape != (B, T, HV, K) or beta.shape != (B, T, HV):
        raise ValueError("gk or beta shape does not match q and HV")
    BT, BC = chunk_size, sub_chunk_size
    if type(BT) is not int or type(BC) is not int or BC <= 0 or BT <= 0 or BT % BC:
        raise ValueError(
            "chunk sizes must be positive integers with BT divisible by BC"
        )
    if Aqk.shape != (B, T, HV, BT) or Akk.shape != (B, T, HV, BC):
        raise ValueError("Aqk/Akk must have shapes [B,T,HV,BT] and [B,T,HV,BC]")
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite real scalar")

    inputs = (q, k, gk, beta)
    N = B
    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a tensor")
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                "packed varlen requires B=1 and at least two sequence offsets"
            )
        if cu_seqlens.device != q.device or cu_seqlens.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("cu_seqlens must be int32/int64 on the input device")
        if not cu_seqlens.is_contiguous():
            raise NotImplementedError("cu_seqlens must be contiguous")
        N = cu_seqlens.numel() - 1
        inputs += (cu_seqlens,)

    # Contiguous tensors occupy one byte interval, including storage offsets.
    for output, others in ((Aqk, inputs + (Akk,)), (Akk, inputs)):
        if not output.numel():
            continue
        start = output.data_ptr()
        end = start + output.numel() * output.element_size()
        for other in others:
            other_start = other.data_ptr()
            other_end = other_start + other.numel() * other.element_size()
            if other.numel() and start < other_end and other_start < end:
                raise NotImplementedError(
                    "outputs must not overlap inputs or each other"
                )
    if B == 0 or T == 0:
        return Aqk, Akk

    def grid(meta):
        return (B * T, triton.cdiv(HV, meta["BH"]))

    with torch_device_fn.device(q.device):
        _chunk_kda_intra_token_parallel_kernel[grid](
            q,
            k,
            gk,
            beta,
            Aqk,
            Akk,
            cu_seqlens,
            float(scale),
            T=T,
            N=N,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=triton.next_power_of_2(K),
            IS_VARLEN=cu_seqlens is not None,
        )
    return Aqk, Akk

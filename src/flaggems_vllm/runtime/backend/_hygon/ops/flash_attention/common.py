# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


def validate_inputs(q, k, v, cuq, cuk, usedk, table, out, alibi, maxq, maxk):
    if q.device.type != "cuda":
        raise ValueError("Hygon FlashAttention requires HIP tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("Hygon FlashAttention supports FP16 and BF16")
    if q.ndim != 3 or k.ndim != (4 if table is not None else 3):
        raise ValueError("Invalid query or KV rank")
    if k.shape != v.shape or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Q/K/V must have matching dtype and K/V shape")
    for tensor in (q, k, v):
        if tensor.requires_grad:
            raise NotImplementedError("Hygon FlashAttention v1 is inference-only")
        if tensor.device != q.device:
            raise ValueError("Q/K/V must use the same device")
        if tensor.stride(-1) != 1:
            raise NotImplementedError("Last dimension must have stride one")
    if q.shape[-1] != k.shape[-1] or not 0 < q.shape[-1] <= 256 or q.shape[-1] % 8:
        raise ValueError("Head dimensions must match and be multiples of 8 up to 256")
    if k.shape[-2] == 0 or q.shape[1] == 0 or q.shape[1] % k.shape[-2]:
        raise ValueError("KV heads must divide query heads")
    if not isinstance(maxq, int) or not isinstance(maxk, int) or min(maxq, maxk) < 0:
        raise ValueError("Maximum sequence lengths must be nonnegative Python integers")
    if q.shape[0] and maxq == 0:
        raise ValueError("Nonempty query requires a positive maximum length")
    if cuq is None or cuq.ndim != 1 or cuq.numel() < 1:
        raise ValueError("cu_seqlens_q must contain batch_size + 1 entries")
    batch = cuq.numel() - 1
    if q.shape[0] and batch == 0:
        raise ValueError("Nonempty query requires at least one sequence")
    if table is not None:
        if cuk is not None or usedk is None:
            raise ValueError("Paged KV requires seqused_k and no cu_seqlens_k")
        if k.shape[1] == 0 or table.ndim != 2 or table.shape[0] != batch:
            raise ValueError("Invalid page table or page size")
        if table.shape[1] * k.shape[1] < maxk or usedk.shape != (batch,):
            raise ValueError("Page table capacity or seqused_k shape is invalid")
    elif cuk is None or usedk is not None or cuk.shape != (batch + 1,):
        raise ValueError("Nonpaged KV requires cu_seqlens_k and no seqused_k")
    for tensor in (cuq, cuk, usedk, table):
        if tensor is not None and (
            tensor.device != q.device
            or tensor.dtype != torch.int32
            or (tensor.ndim == 1 and not tensor.is_contiguous())
            or tensor.stride(-1) != 1
        ):
            raise ValueError("Sequence metadata must be int32 on the query device")
    if out is not None:
        if (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
            or out.stride(-1) != 1
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous with the query shape/dtype/device")

        for tensor in (q, k, v):
            if (
                out.numel()
                and tensor.numel()
                and (
                    out.untyped_storage().data_ptr()
                    == tensor.untyped_storage().data_ptr()
                )
            ):
                raise ValueError("out must not alias Q/K/V storage")
    if alibi is not None and (
        alibi.shape not in ((q.shape[1],), (batch, q.shape[1]))
        or alibi.dtype != torch.float32
        or alibi.device != q.device
        or alibi.stride(-1) != 1
    ):
        raise ValueError("ALiBi must be FP32 with shape [Hq] or [batch,Hq]")


ATTENTION_TUNING_KEY = [
    "TQ",
    "HQ",
    "HK",
    "D",
    "QS",
    "QH",
    "KP",
    "KS",
    "KH",
    "VP",
    "VS",
    "VH",
    "OS",
    "OH",
    "TS",
    "AS",
    "PAGE",
    "SCALE",
    "CAUSAL",
    "WL",
    "WR",
    "SOFTCAP",
    "PAGED",
    "HAS_ALIBI",
    "PACKED",
    "BD",
    "MAX_Q",
    "MAX_K",
    "BATCHES",
]


def kernel_arguments(
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    usedk,
    table,
    alibi,
    scale,
    causal,
    window,
    softcap,
    packed,
):
    paged = table is not None
    return (
        q,
        k,
        v,
        out,
        lse,
        cuq,
        cuk,
        usedk,
        table,
        alibi,
        q.shape[0],
        q.shape[1],
        k.shape[-2],
        q.shape[2],
        *q.stride()[:2],
        k.stride(0) if paged else 0,
        k.stride(-3),
        k.stride(-2),
        v.stride(0) if paged else 0,
        v.stride(-3),
        v.stride(-2),
        out.stride(-3),
        out.stride(-2),
        table.stride(0) if paged else 0,
        alibi.stride(0) if alibi is not None and alibi.ndim == 2 else 0,
        k.shape[1] if paged else 1,
        scale,
        causal,
        *window,
        softcap,
        paged,
        alibi is not None,
        packed,
    )


@triton.jit
def scan_attention(
    q,
    K,
    V,
    TABLE,
    ALIBI,
    dims,
    qi,
    qh,
    valid_q,
    batch,
    kh,
    kstart,
    klen,
    qlen,
    low,
    high,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    D: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    AS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    CAUSAL: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    PAGED: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
):
    m = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    if HAS_ALIBI:
        slope = tl.load(ALIBI + batch * AS + qh, valid_q, 0)
    for start in range(low, high, BN):
        cols = start + tl.arange(0, BN)
        if PAGED:
            page = tl.load(TABLE + batch * TS + cols // PAGE, cols < klen, 0)

            ko = page.to(tl.int64) * KP + (cols % PAGE) * KS
            vo = page.to(tl.int64) * VP + (cols % PAGE) * VS
        else:
            ko = (kstart + cols).to(tl.int64) * KS
            vo = (kstart + cols).to(tl.int64) * VS
        k = tl.load(
            K + ko[None, :] + kh * KH + dims[:, None],
            (cols[None, :] < klen) & (dims[:, None] < D),
            0,
        )
        scores = tl.dot(q, k) * SCALE
        if SOFTCAP > 0:
            z = scores / SOFTCAP
            scores = SOFTCAP * (2.0 / (1.0 + tl.exp(2.0 * -z)) - 1.0)
        pos = qi[:, None] + klen - qlen
        if HAS_ALIBI:
            scores -= slope[:, None] * tl.abs(pos - cols[None, :])
        mask = valid_q[:, None] & (cols[None, :] < klen)
        if CAUSAL:
            mask &= cols[None, :] <= pos
        if WL >= 0:
            mask &= cols[None, :] >= pos - WL
        if WR >= 0:
            mask &= cols[None, :] <= pos + WR
        scores = tl.where(mask, scores * 1.4426950408889634, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(scores - safe_m[:, None])
        row_sum = row_sum * alpha + tl.sum(p, 1)
        acc *= alpha[:, None]
        v = tl.load(
            V + vo[:, None] + kh * VH + dims[None, :],
            (cols[:, None] < klen) & (dims[None, :] < D),
            0,
        )
        acc = tl.dot(p.to(v.dtype), v, acc)
        m = new_m
    result = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    lse = tl.where(
        row_sum > 0, (m + tl.log2(row_sum)) * 0.6931471805599453, float("inf")
    )
    return result, lse, row_sum

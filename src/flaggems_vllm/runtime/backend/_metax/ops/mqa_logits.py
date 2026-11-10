# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Weighted MQA logits with explicit shared or per-query key storage."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.fp8_fp4_mqa_logits import _fp8_fp4_mqa_logits_kernel
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


def mqa_logits(
    q,
    kv,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
    *,
    out=None,
    row_shard=False,
):
    """Return FP32 weighted ReLU logits, preserving caller-owned output.

    MetaX supports BF16 Q[M,64,128] in two explicit storage contracts:
    * BF16 KV[N,128], INT32 start/end[M]: write out[:M,:N]. Output may
      have extra padding rows or a pitched row stride, which are preserved.
    * (FP8 keys[M,N,128], FP32 scales[M,N]), starts=None, integer ends[M]:
      each query has separate gathered keys. N is a positive multiple of 128.
      Keys and scales are rounded/multiplied in BF16 before MMA, as in the
      original gathered path. This is not the FP8-query public API.

    Invalid positions are -inf. ``row_shard`` selects the existing local
    shared-key implementation; no communication is performed by this API.
    ``out`` is returned by identity. No cross-request workspace is retained.
    """
    logger.debug("GEMS_METAX MQA_LOGITS")
    if q.ndim != 3 or q.shape[1:] != (64, 128) or q.dtype != torch.bfloat16:
        raise ValueError("MQA requires BF16 queries [M,64,128]")
    rows = q.shape[0]
    if isinstance(kv, tuple):
        if len(kv) != 2:
            raise ValueError("Gathered keys require a (values, scales) pair")
        keys, scales = kv
        if cu_seqlen_ks is not None:
            raise ValueError("Gathered MQA starts must be None (zero starts)")
        if row_shard:
            raise ValueError("row_shard is only supported for shared BF16 keys")
        if (
            keys.ndim != 3
            or keys.shape[0] != rows
            or keys.shape[2] != 128
            or keys.dtype != torch.float8_e4m3fn
            or scales.shape != keys.shape[:2]
            or scales.dtype != torch.float32
            or not keys.is_contiguous()
            or not scales.is_contiguous()
        ):
            raise ValueError("Gathered MQA requires contiguous FP8/FP32 key storage")
        slots = keys.shape[1]
        if slots <= 0 or slots % 128:
            raise ValueError("Gathered key capacity must be a positive multiple of 128")
        if out is None:
            out = torch.empty((rows, slots), device=q.device, dtype=torch.float32)
        _batched_mqa_logits(
            q,
            keys.view(rows * slots, 128),
            scales.view(rows * slots),
            weights,
            cu_seqlen_ke,
            out,
            slots,
        )
        return out

    if kv.ndim != 2 or cu_seqlen_ks is None or cu_seqlen_ke is None:
        raise ValueError("Shared MQA requires rank-2 BF16 keys and start/end bounds")
    if out is None:
        out = torch.empty((rows, kv.shape[0]), device=q.device, dtype=torch.float32)
    _mqa_logits_into(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, out, row_shard=row_shard
    )
    return out


def _bind_shared_mqa(config_name):
    # Standard library wrappers own tuning/cache behavior. The JIT object is
    # shared; neither the generic FP8 wrapper nor its configurations are mutated.
    return libentry()(
        libtuner(
            configs=runtime.get_tuned_config(config_name), key=["M", "N", "H", "D"]
        )(_fp8_fp4_mqa_logits_kernel.jit_function)
    )


_row_tile_logits = _bind_shared_mqa("mqa_logits_into")
_row_shard_logits = _bind_shared_mqa("mqa_logits_into_row_shard")


def _mqa_logits_into(q, kv, weights, ks, ke, out, *, row_shard=False):
    """Write logits for the supplied query rows; communication is caller-owned.

    BF16 q[m,64,128], kv[n,128], FP32 weights[m,64]/out[>=m,n],
    INT32 ks/ke[m]. Mutates out[:m], preserving any padding rows. No cache.
    """
    logger.debug("GEMS_METAX MQA_LOGITS_INTO")
    m, h, d = q.shape
    n = kv.shape[0]
    if (h, d) != (64, 128) or kv.shape != (n, d) or weights.shape != (m, h):
        raise ValueError("Unsupported MQA input shape")
    if (
        q.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
        or weights.dtype != torch.float32
        or out.dtype != torch.float32
        or ks.dtype != torch.int32
        or ke.dtype != torch.int32
    ):
        raise TypeError("Unsupported MQA dtype")
    if out.shape[0] < m or out.shape[1] != n or ks.numel() != m or ke.numel() != m:
        raise ValueError("Invalid MQA output or bounds")
    if (
        any(x.device != q.device for x in (kv, weights, ks, ke, out))
        or not all(x.is_contiguous() for x in (q, kv, weights, ks, ke))
        or out.stride(1) != 1
    ):
        raise ValueError("Invalid MQA device/layout")
    if m == 0 or n == 0:
        return
    kernel = _row_shard_logits if row_shard else _row_tile_logits
    kernel[
        lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    ](
        q,
        kv,
        None,
        weights,
        out,
        m,
        n,
        h,
        d,
        *q.stride(),
        *kv.stride(),
        *out.stride(),
        *weights.stride(),
        HAS_K_SCALE=False,
        KS_ptr=ks,
        KE_ptr=ke,
        APPLY_ROW_MASK=True,
    )


@triton.jit
def _batched_mqa_logits_kernel(
    q_ptr,  # [rows, H, D] bf16
    k_fp8_ptr,  # [rows * NB * BC, D] fp8 (e4m3) gathered workspace
    k_scale_ptr,  # [rows * NB * BC] fp32
    w_ptr,  # [rows, H] fp32
    valid_ptr,  # [rows] int32
    out_ptr,  # [rows, NB * BC] fp32 logits
    stride_q_row,
    stride_out_row,
    NB: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(1)
    blk = tl.program_id(0)
    valid_len = tl.load(valid_ptr + row)

    h_offs = tl.arange(0, H)
    d_offs = tl.arange(0, D)
    # load q row: [H, D] bf16
    q = tl.load(q_ptr + row * stride_q_row + h_offs[:, None] * D + d_offs[None, :])

    k_offs = blk * BC + tl.arange(0, BC)
    k_mask = k_offs < valid_len
    # load fp8 K block: [BC, D] -> dequant to bf16
    kbase = (row * NB * BC + k_offs)[:, None] * D + d_offs[None, :]
    k_fp8 = tl.load(k_fp8_ptr + kbase, mask=k_mask[:, None], other=0.0)
    scale = tl.load(k_scale_ptr + row * NB * BC + k_offs, mask=k_mask, other=0.0)
    k_bf16 = k_fp8.to(tl.bfloat16) * scale[:, None].to(tl.bfloat16)

    # scores[H, BC] = q @ K^T
    scores = tl.dot(q, tl.trans(k_bf16))  # [H, BC] fp32
    w = tl.load(w_ptr + row * H + h_offs)  # [H] fp32
    scores = tl.maximum(scores, 0.0)
    row_scores = tl.sum(scores * w[:, None], axis=0)  # [BC] fp32

    out = tl.where(k_mask, row_scores, -float("inf"))
    tl.store(out_ptr + row * stride_out_row + k_offs, out)


def _batched_mqa_logits(q, k_fp8, k_scale, weights, valid_lens, out, num_slots):
    """q [rows,H,D] bf16; k_fp8 [rows*num_slots,D] fp8; k_scale [rows*num_slots] fp32;
    weights [rows,H] fp32; valid_lens [rows] int; out [rows,num_slots] fp32."""
    logger.debug("GEMS_METAX BATCHED_MQA_LOGITS")
    if q.ndim != 3 or q.shape[1:] != (64, 128) or q.dtype != torch.bfloat16:
        raise ValueError("MQA decode requires BF16 [rows,64,128] queries")
    rows = q.shape[0]
    if num_slots <= 0 or num_slots % 128:
        raise ValueError("MQA slot capacity must be a positive multiple of 128")
    if k_fp8.shape != (rows * num_slots, 128) or k_scale.numel() != rows * num_slots:
        raise ValueError("Invalid gathered key/scale capacity")
    if (
        weights.shape != (rows, 64)
        or valid_lens.shape != (rows,)
        or out.shape != (rows, num_slots)
    ):
        raise ValueError("Invalid MQA row/output shape")
    if (
        k_scale.dtype != torch.float32
        or weights.dtype != torch.float32
        or out.dtype != torch.float32
        or valid_lens.dtype not in (torch.int32, torch.int64)
    ):
        raise TypeError("Unsupported MQA scale/weight/bounds/output dtype")
    tensors = (k_fp8, k_scale, weights, valid_lens, out)
    if (
        any(x.device != q.device for x in tensors)
        or out.stride(1) != 1
        or q.stride()[1:] != (128, 1)
    ):
        raise ValueError("Invalid MQA device/layout")
    if not all(x.is_contiguous() for x in (k_fp8, k_scale, weights, valid_lens)):
        raise ValueError("Gathered MQA inputs must be contiguous")
    if rows == 0:
        return out

    rows, H, D = q.shape
    BC = 128
    NB = num_slots // BC
    if not (num_slots % BC == 0):
        raise ValueError("Invalid operator argument: num_slots % BC == 0")
    _batched_mqa_logits_kernel[(NB, rows)](
        q,
        k_fp8,
        k_scale,
        weights,
        valid_lens,
        out,
        q.stride(0),
        out.stride(0),
        NB=NB,
        H=H,
        D=D,
        BC=BC,
        num_warps=4,
    )
    return out

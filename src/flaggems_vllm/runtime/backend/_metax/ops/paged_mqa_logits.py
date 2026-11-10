# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paged MQA dispatch preserving the common FP8 and MetaX BF16 contracts."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.paged_mqa import paged_mqa_logits as _common_paged_mqa_logits

logger = logging.getLogger(__name__)


def paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
    *,
    out=None,
):
    """Keep the common positional API; expose a bounded MetaX BF16 extension.

    A tuple query retains the unchanged common FP8-query path and does not
    support ``out``. A BF16 Tensor query [B,64,128] or [B,1,64,128] selects
    the deployed FP8 paged-cache path, which rounds key dequantization to BF16
    before MMA. This extension requires clean_logits=True and supports a
    caller-owned FP32 out[B,max_model_len] returned by identity. The page
    layout is uint8[Bpages,page_size,1,132], values followed by FP32 scales
    within each physical page. Pitched physical pages are preserved.

    All capability branches run before kernel execution. There is no runtime
    exception retry, data conversion to FP8, or workspace ownership transfer.
    """
    logger.debug("GEMS_METAX PAGED_MQA_LOGITS")
    if isinstance(q, tuple):
        if out is not None:
            raise ValueError("Common FP8 paged MQA does not support caller-owned out")
        return _common_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits,
        )
    if q.dtype != torch.bfloat16 or q.ndim not in (3, 4):
        raise ValueError("MetaX paged MQA requires a BF16 Tensor query")
    if q.ndim == 4:
        if q.shape[1] != 1:
            raise ValueError(
                "MetaX BF16 paged MQA supports exactly one query per batch"
            )
        q = q[:, 0]
    rows = q.shape[0]
    if q.shape[1:] != (64, 128):
        raise ValueError("MetaX BF16 paged MQA requires 64 heads of dimension 128")
    if clean_logits is not True:
        raise ValueError("MetaX BF16 paged MQA requires clean_logits=True")
    if (
        kv_cache.ndim != 4
        or kv_cache.dtype != torch.uint8
        or kv_cache.shape[2:] != (1, 132)
        or kv_cache.stride(-1) != 1
        or kv_cache.stride(1) != 132
        or kv_cache.stride(0) < kv_cache.shape[1] * 132
        or kv_cache.stride(0) % 4
    ):
        raise ValueError("Invalid MetaX paged FP8 cache layout")
    if (
        context_lens.shape not in ((rows,), (rows, 1))
        or context_lens.dtype not in (torch.int32, torch.int64)
        or weights.shape != (rows, 64)
        or weights.dtype != torch.float32
        or block_tables.ndim != 2
        or block_tables.shape[0] != rows
        or block_tables.dtype not in (torch.int32, torch.int64)
        or max_model_len <= 0
        or max_model_len % 128
        or block_tables.shape[1] * kv_cache.shape[1] < max_model_len
    ):
        raise ValueError("Invalid MetaX paged MQA bounds, weights, or capacity")
    tensors = (kv_cache, weights, context_lens, block_tables)
    if any(t.device != q.device for t in tensors):
        raise ValueError("Paged MQA tensors must share the query device")
    if out is not None and (
        out.shape != (rows, max_model_len)
        or out.dtype != torch.float32
        or out.device != q.device
        or out.stride(1) != 1
    ):
        raise ValueError("Invalid caller-owned paged MQA output")
    return _paged_indexer_mqa_logits_match(
        q,
        kv_cache,
        weights,
        context_lens.reshape(rows),
        block_tables,
        out=out,
        num_slots=max_model_len,
    )


@triton.jit
def _paged_indexer_mqa_logits_match_kernel(
    Q,
    KV_VALUES,
    KV_SCALES,
    WEIGHTS,
    VALID_LENS,
    BLOCK_TABLE,
    OUT,
    stride_q_row: tl.constexpr,
    stride_out_row: tl.constexpr,
    stride_kv_block: tl.constexpr,
    stride_ks_block: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_bts: tl.constexpr,
    NB: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_QBLOCKS: tl.constexpr,
):
    row = tl.program_id(1)
    blk = tl.program_id(0)
    valid_len = tl.load(VALID_LENS + row)
    k_offs = blk * BC + tl.arange(0, BC)
    # Early-exit (ml-static-waste fix): blocks entirely beyond valid_len produce
    # all -inf in the original semantics (k_mask all false -> out = -inf).
    # Skip the q load, KV loads and tl.dot; only store -inf. This removes the
    # per-step compute/scratch that scaled statically with max_model_len instead
    # of the actual context length. Grid stays static -> graph-replay safe.
    if blk * BC >= valid_len:
        tl.store(
            OUT + row * stride_out_row + k_offs,
            tl.full((BC,), -float("inf"), dtype=tl.float32),
        )
        return
    h_offs = tl.arange(0, H)
    d_offs = tl.arange(0, D)
    q = tl.load(Q + row * stride_q_row + h_offs[:, None] * D + d_offs[None, :])

    k_mask = k_offs < valid_len
    phys_block_idx = k_offs // BLOCK_SIZE
    intra = k_offs - phys_block_idx * BLOCK_SIZE
    block_ids = tl.load(
        BLOCK_TABLE + row * stride_btb + phys_block_idx * stride_bts,
        mask=k_mask,
        other=0,
    )
    block_ids = tl.where(k_mask, block_ids, 0)
    intra = tl.where(k_mask, intra, 0)

    k_ptrs = (
        KV_VALUES
        + block_ids[:, None].to(tl.int64) * stride_kv_block
        + intra[:, None] * D
        + d_offs[None, :]
    )
    k_u8 = tl.load(k_ptrs, mask=k_mask[:, None], other=0)
    k_fp8 = k_u8.to(tl.float8e4nv, bitcast=True)
    scale = tl.load(
        KV_SCALES + block_ids.to(tl.int64) * stride_ks_block + intra * NUM_QBLOCKS,
        mask=k_mask,
        other=0.0,
    )
    k_bf16 = k_fp8.to(tl.bfloat16) * scale[:, None].to(tl.bfloat16)

    scores = tl.dot(q, tl.trans(k_bf16))
    w = tl.load(WEIGHTS + row * H + h_offs)
    scores = tl.maximum(scores, 0.0)
    row_scores = tl.sum(scores * w[:, None], axis=0)
    out = tl.where(k_mask, row_scores, -float("inf"))
    tl.store(OUT + row * stride_out_row + k_offs, out)


def _paged_indexer_mqa_logits_match(
    q, kv_cache, weights, valid_lens, block_table, out=None, num_slots=None
):
    logger.debug("GEMS_METAX PAGED_INDEXER_MQA_LOGITS_MATCH")
    if not (q.is_cuda and kv_cache.is_cuda):
        raise ValueError("Invalid operator argument: q.is_cuda and kv_cache.is_cuda")
    B, H, D = q.shape
    if not (D == 128 and H == 64):
        raise ValueError((H, D))
    block_size = kv_cache.shape[1]
    if num_slots is None:
        num_slots = block_table.shape[1] * block_size
    if out is None:
        out = torch.empty((B, num_slots), device=q.device, dtype=torch.float32)
    kv_flat = kv_cache.view(kv_cache.shape[0], -1)
    kv_values = kv_flat[:, : block_size * D]
    kv_scales = kv_flat[:, block_size * D :].view(torch.float32)
    BC = 128
    if not (num_slots % BC == 0):
        raise ValueError("Invalid operator argument: num_slots % BC == 0")
    NB = num_slots // BC
    qc = q.contiguous()
    wc = weights.contiguous()
    btc = block_table.contiguous()
    vlc = valid_lens.contiguous()
    _paged_indexer_mqa_logits_match_kernel[(NB, B)](
        qc,
        kv_values,
        kv_scales,
        wc,
        vlc,
        btc,
        out,
        qc.stride(0),
        out.stride(0),
        kv_values.stride(0),
        kv_scales.stride(0),
        btc.stride(0),
        btc.stride(1),
        NB=NB,
        H=H,
        D=D,
        BC=BC,
        BLOCK_SIZE=block_size,
        NUM_QBLOCKS=1,
        num_warps=4,
    )
    return out

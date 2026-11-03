# SPDX-License-Identifier: Apache-2.0
"""Static-grid paged MQA logits with metadata refreshed on graph replay."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fp8_fp4_paged_mqa_logits import (
    _mqa_logits_kernel,
    _select_block_kv,
)


@triton.jit
def _fill_logits(Output, count, CLEAN: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = float("-inf") if CLEAN else 0.0
    tl.store(Output + offsets, value, offsets < count)


def fp8_fp4_paged_mqa_logits_graph(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
):
    """Graph-safe counterpart to :func:`fp8_fp4_paged_mqa_logits`.

    The grid uses the static model/table capacity. Lengths, page mappings,
    queries, scales and weights are read on-device on every replay. Cache pages
    are consumed directly, without expanding tables or copying the KV cache.
    This entry point uses the existing baseline MMA; it does not select TLE.
    """
    q_values, q_scale = q
    if q_values.ndim == 3:
        q_values = q_values.unsqueeze(1)
    if q_values.ndim != 4 or not q_values.is_contiguous():
        raise NotImplementedError("Paged graph logits require contiguous 3D/4D Q")
    batch, next_n, heads, q_dim = q_values.shape
    rows = batch * next_n
    is_fp4 = q_scale is not None
    head_dim = kv_cache.shape[-1] - 4
    page_size = kv_cache.shape[1]
    if (
        head_dim != 128
        or heads not in (16, 32, 64)
        or page_size not in (64, 128, 256)
        or q_dim != (head_dim // 2 if is_fp4 else head_dim)
    ):
        raise NotImplementedError("Unsupported paged graph logits dimensions")
    if kv_cache.ndim != 4 or kv_cache.shape[2] != 1:
        raise NotImplementedError("Paged graph logits require single-head KV pages")
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.stride(-1) != 1
        or kv_cache.stride(1) != head_dim + 4
        or kv_cache.stride(0) % 4
    ):
        raise NotImplementedError("Unsupported paged graph KV storage layout")
    if weights.shape != (rows, heads) or weights.stride(1) != 1:
        raise NotImplementedError("Paged graph logits require [B*N,H] weights")
    if weights.dtype != torch.float32:
        raise NotImplementedError("Paged graph logits require FP32 weights")
    if is_fp4:
        if (
            q_values.dtype not in (torch.uint8, torch.int8)
            or q_scale.dtype != torch.int32
            or q_scale.numel() != rows * heads
            or not q_scale.is_contiguous()
        ):
            raise NotImplementedError("Unsupported MXFP4 Q/scales layout")
    elif q_values.dtype != torch.float8_e4m3fn:
        raise NotImplementedError("Paged graph logits require FP8 e4m3fn Q")
    if context_lens.dtype != torch.int32 or block_tables.dtype != torch.int32:
        raise NotImplementedError("Paged graph logits require int32 metadata")
    if context_lens.shape not in ((batch,), (batch, next_n)):
        raise NotImplementedError("Unsupported paged graph context-length shape")
    if block_tables.ndim not in (2, 3) or block_tables.shape[0] != batch:
        raise NotImplementedError("Unsupported paged graph block-table shape")
    if block_tables.ndim == 3 and block_tables.shape[1] != next_n:
        raise NotImplementedError("Unsupported paged graph block-table next_n")
    if block_tables.stride(-1) != 1:
        raise NotImplementedError("Paged graph block-table columns must be contiguous")
    if max_model_len < 0:
        raise ValueError("max_model_len must be nonnegative")
    tensors = [kv_cache, weights, context_lens, block_tables]
    if q_scale is not None:
        tensors.append(q_scale)
    if q_values.device.type != "cuda" or any(
        tensor.device != q_values.device for tensor in tensors
    ):
        raise NotImplementedError("Paged graph logits require one CUDA device")

    logits = torch.empty(
        (rows, max_model_len), dtype=torch.float32, device=q_values.device
    )
    if rows == 0 or max_model_len == 0:
        return logits
    _fill_logits[(triton.cdiv(rows * max_model_len, 1024),)](
        logits, rows * max_model_len, clean_logits, 1024, num_warps=4
    )
    capacity = min(max_model_len, block_tables.shape[-1] * page_size)
    if capacity == 0:
        return logits
    block_kv, num_blocks = _select_block_kv(capacity, page_size)
    q_u8 = q_values.view(torch.uint8)
    # A no-copy bitcast exposes each page's trailing per-token FP32 scales.
    kv_scales = kv_cache.view(torch.float32)
    _mqa_logits_kernel[(triton.cdiv(capacity, block_kv), rows)](
        q_u8,
        q_scale if is_fp4 else q_u8,
        kv_cache,
        kv_scales,
        weights,
        block_tables,
        logits,
        context_lens,
        total_rows=rows,
        max_ctx=capacity,
        num_heads=heads,
        head_dim=head_dim,
        max_model_len=max_model_len,
        block_size=page_size,
        max_blocks_per_seq=block_tables.shape[-1],
        num_phys_blocks=kv_cache.shape[0],
        stride_q_row=heads * q_dim,
        stride_qs_row=heads,
        stride_kv_flat=head_dim,
        stride_bt_row=block_tables.stride(0),
        stride_out_row=max_model_len,
        stride_w_row=weights.stride(0),
        BLOCK_KV=block_kv,
        BLOCK_D=128,
        NUM_BLOCKS=num_blocks,
        IS_MXFP4=is_fp4,
        PAGED_KV=True,
        NEXT_N=next_n,
        stride_kv_page=kv_cache.stride(0),
        stride_ctx_batch=context_lens.stride(0),
        stride_ctx_next=context_lens.stride(1) if context_lens.ndim == 2 else 0,
        stride_bt_next=block_tables.stride(1) if block_tables.ndim == 3 else 0,
    )
    return logits


fp8_fp4_paged_mqa_logits_graph.supports_cuda_graph = True

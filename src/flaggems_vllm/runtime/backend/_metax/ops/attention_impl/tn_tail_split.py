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

"""D256 async-TN Split-KV for peeled Q prefixes and aligned paged tails."""
import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

from .tn_addressing import tn_compile_scenario
from .tn_direct import flash_varlen_fwd_d256_tn_kernel

logger = logging.getLogger(__name__)


def launch_tail_split(
    params,
    *,
    max_seqlen_q,
    batch_size,
    num_heads,
    total_q,
    num_splits,
    peel_q=False,
    prefix_only=False,
    defer_merge=False,
    single_tile=False,
    aligned_single_mask=False,
):
    """Split bounded prefix/tail rows per request and merge FP32 partials."""
    if prefix_only and peel_q:
        raise ValueError("Prefix Split-KV must retain the original Q base and length")
    if params.q_ptr.dtype != torch.bfloat16:
        raise ValueError("Async-TN tail Split-KV requires BF16 Q/K/V")
    if not params.is_paged or params.block_size not in (16, 32) or params.d != 256:
        raise ValueError(
            "Async-TN tail Split-KV requires paged D256 with page size 16 or 32"
        )
    if not params.is_causal or params.is_local:
        raise ValueError("Async-TN tail Split-KV only supports causal global attention")
    if params.is_dropout or params.is_alibi or params.is_softcap:
        raise ValueError(
            "Async-TN tail Split-KV does not support optional attention features"
        )
    if params.cu_seqlens_q_ptr is None:
        raise ValueError("Async-TN tail Split-KV requires varlen Q prefix sums")
    if params.cu_seqlens_k_ptr is None and params.seqused_k_ptr is None:
        raise ValueError("Async-TN tail Split-KV requires KV lengths")
    if not 2 <= num_splits <= 32:
        raise ValueError("Async-TN tail Split-KV requires 2 to 32 splits")
    if num_heads % params.h_hk_ratio != 0:
        raise ValueError("Async-TN tail Split-KV received an invalid GQA ratio")
    if params.h_hk_ratio != 8 and (
        not (params.block_size == 16 and params.h_hk_ratio == 4)
    ):
        raise ValueError("TN tail Split-KV requires GQA8 or page16/GQA4")
    block_m = 32
    block_n = 128
    block_k = 128
    rows_per_request = 64 // params.h_hk_ratio if prefix_only else 128
    q_tiles = rows_per_request * params.h_hk_ratio // block_m
    tail_rows = batch_size * rows_per_request
    head_programs = num_heads // params.h_hk_ratio
    original_out = params.o_ptr
    original_lse = params.softmax_lse_ptr
    out_splits = torch.empty(
        (num_splits, tail_rows, num_heads, 256),
        dtype=torch.float32,
        device=original_out.device,
    )
    lse_splits = torch.empty(
        (num_splits, num_heads, tail_rows),
        dtype=torch.float32,
        device=original_out.device,
    )
    grid = (head_programs, batch_size, num_splits * q_tiles)
    cfg = {
        "PAGE_SIZE": params.block_size,
        "COMPACT_WORKLIST": False,
        "BULK_TAIL": not prefix_only,
        "PREFIX_ONLY": prefix_only,
        "PEEL_Q": peel_q,
        "TAIL_ROWS": tail_rows,
        "SPLIT_KV": True,
        "NUM_SPLITS": num_splits,
        "Q_TILES": q_tiles,
        "GRID_ORDER": 1,
        "REVERSE_Q_TILES": False,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "QK_BLOCK_K": 64,
        "FAST_CAUSAL_ALIGNED": False,
        "num_warps": 4,
        "num_stages": 4,
        "pipeline": "cpasync",
        "scenario": tn_compile_scenario(
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            out_splits,
            lse_splits,
            params.page_table_ptr,
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
        ),
        "pipeline_load_num": -1,
        "inner_stages": (0, 0),
    }
    logger.debug("Running D256 async-TN Split-KV with config: %s", cfg)
    kernel = flash_varlen_fwd_d256_tn_kernel
    if single_tile:
        assert prefix_only and 2 <= num_splits <= 16 and params.block_size == 16
        from .tn_prefix_single import flash_varlen_fwd_d256_tn_kernel as prefix_kernel

        kernel = prefix_kernel
    if aligned_single_mask:
        assert not prefix_only and peel_q and params.h_hk_ratio == 4
        from .tn_tail_aligned import flash_varlen_fwd_d256_tn_kernel as aligned_kernel

        kernel = aligned_kernel
    partial = kernel[grid](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        out_splits,
        lse_splits,
        params.q_row_stride,
        params.k_row_stride,
        params.v_row_stride,
        params.q_head_stride,
        params.k_head_stride,
        params.v_head_stride,
        out_splits.stride(1),
        out_splits.stride(2),
        params.scale_softmax,
        params.scale_softmax_log2,
        params.cu_seqlens_q_ptr,
        params.cu_seqlens_k_ptr,
        params.seqused_k_ptr,
        params.is_cu_seqlens_k,
        params.is_seqused_k,
        num_heads,
        params.h_hk_ratio,
        total_q,
        params.page_table_ptr,
        params.page_table_batch_stride,
        params.k_page_stride,
        params.v_ptr.stride(0),
        params.page_table_ptr,
        **cfg,
    )
    if defer_merge:
        return (partial, out_splits, lse_splits)
    merge = tail_split_merge_kernel[rows_per_request, num_heads, batch_size]
    merge(
        original_out,
        original_lse,
        out_splits,
        lse_splits,
        params.o_row_stride,
        params.o_head_stride,
        total_q,
        num_heads,
        256,
        num_splits,
        triton.next_power_of_2(num_splits),
        256,
        params.cu_seqlens_q_ptr,
        params.cu_seqlens_k_ptr,
        params.seqused_k_ptr,
        params.is_seqused_k,
        tail_rows,
        peel_q,
        prefix_only,
        64 // params.h_hk_ratio,
        num_warps=1,
        num_stages=1,
    )
    return partial


@libentry()
@triton.jit
def tail_split_merge_kernel(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    o_row_stride,
    o_head_stride,
    total_q: tl.constexpr,
    h: tl.constexpr,
    d: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MAX_N_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    cu_q_ptr,
    cu_k_ptr,
    used_k_ptr,
    IS_SEQUSED_K: tl.constexpr,
    TAIL_ROWS: tl.constexpr,
    PEEL_Q: tl.constexpr,
    PREFIX_ONLY: tl.constexpr,
    Q_ALIGNMENT: tl.constexpr = 8,
):
    tail_idx = tl.program_id(0)
    bid = tl.program_id(2)
    q_bos = tl.load(cu_q_ptr + bid)
    q_len = tl.load(cu_q_ptr + bid + 1) - q_bos
    if IS_SEQUSED_K:
        k_len = tl.load(used_k_ptr + bid)
    else:
        k_len = tl.load(cu_k_ptr + bid + 1) - tl.load(cu_k_ptr + bid)
    if PEEL_Q:
        prefix_q = min(q_len, q_len - k_len & Q_ALIGNMENT - 1)
        q_bos += prefix_q
        q_len -= prefix_q
    if PREFIX_ONLY:
        prefix_q = min(q_len, q_len - k_len & Q_ALIGNMENT - 1)
        if tail_idx >= prefix_q:
            return
        q_idx = q_bos + tail_idx
        compact_idx = bid * Q_ALIGNMENT + tail_idx
    else:
        shift = k_len - q_len
        if shift % Q_ALIGNMENT != 0:
            return
        full_q = max(0, min(q_len, k_len // 128 * 128 - shift))
        full_q = full_q // Q_ALIGNMENT * Q_ALIGNMENT
        if full_q + tail_idx >= q_len:
            return
        q_idx = q_bos + full_q + tail_idx
        compact_idx = bid * 128 + tail_idx
    head_idx = tl.program_id(1)
    split_idx = tl.arange(0, MAX_N_SPLITS)
    split_mask = split_idx < NUM_SPLITS
    lse_offsets = split_idx * h * TAIL_ROWS + head_idx * TAIL_ROWS + compact_idx
    split_lse = tl.load(
        lse_splits_ptr + lse_offsets, mask=split_mask, other=float("-inf")
    )
    valid_lse = split_mask & (split_lse > float("-inf")) & (split_lse < float("inf"))
    max_lse = tl.max(tl.where(valid_lse, split_lse, float("-inf")), axis=0)
    has_valid_lse = max_lse > float("-inf")
    safe_max_lse = tl.where(has_valid_lse, max_lse, 0.0)
    split_scale = tl.where(valid_lse, tl.exp(split_lse - safe_max_lse), 0.0)
    scale_sum = tl.sum(split_scale, axis=0)
    safe_scale_sum = tl.where(has_valid_lse, scale_sum, 1.0)
    weights = split_scale / safe_scale_sum
    col = tl.arange(0, BLOCK_K)
    split_out_offsets = (
        split_idx[:, None] * TAIL_ROWS * h * d
        + (compact_idx * h + head_idx) * d
        + col[None, :]
    )
    split_out = tl.load(
        out_splits_ptr + split_out_offsets,
        mask=split_mask[:, None] & (col[None, :] < d),
        other=0.0,
    )
    out = tl.sum(weights[:, None] * split_out, axis=0)
    out_offsets = q_idx * o_row_stride + head_idx * o_head_stride + col
    tl.store(out_ptr + out_offsets, out, mask=col < d)
    merged_lse = tl.where(
        has_valid_lse, tl.log(safe_scale_sum) + safe_max_lse, float("-inf")
    )
    tl.store(lse_ptr + head_idx * total_q + q_idx, merged_lse)


__all__ = ["launch_tail_split"]

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

"""Four boundary query rows per CTA with streamed FP32 partials."""

import triton
import triton.language as tl

from flaggems_vllm.utils import libentry


@triton.jit
def _merge_region(
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
    Q_ALIGNMENT: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    first_idx = tl.program_id(0) * 4 - ROW_OFFSET
    tail_idx = first_idx + tl.arange(0, 4)
    bid = tl.program_id(2)
    q_bos = tl.load(cu_q_ptr + bid)
    q_len = tl.load(cu_q_ptr + bid + 1) - q_bos
    if IS_SEQUSED_K:
        k_len = tl.load(used_k_ptr + bid)
    else:
        k_len = tl.load(cu_k_ptr + bid + 1) - tl.load(cu_k_ptr + bid)
    if PEEL_Q:
        prefix_q = min(q_len, (q_len - k_len) & (Q_ALIGNMENT - 1))
        q_bos += prefix_q
        q_len -= prefix_q
    if PREFIX_ONLY:
        prefix_q = min(q_len, (q_len - k_len) & (Q_ALIGNMENT - 1))
        if first_idx >= prefix_q:
            return
        row_mask = tail_idx < prefix_q
        q_idx = q_bos + tail_idx
        compact_idx = bid * Q_ALIGNMENT + tail_idx
    else:
        shift = k_len - q_len
        if shift % Q_ALIGNMENT != 0:
            return
        full_q = max(0, min(q_len, (k_len // 128) * 128 - shift))
        full_q = (full_q // Q_ALIGNMENT) * Q_ALIGNMENT
        if full_q + first_idx >= q_len:
            return
        row_mask = full_q + tail_idx < q_len
        q_idx = q_bos + full_q + tail_idx
        compact_idx = bid * 128 + tail_idx
    head_idx = tl.program_id(1)
    split_idx = tl.arange(0, MAX_N_SPLITS)
    split_mask = split_idx < NUM_SPLITS
    valid = split_mask[:, None] & row_mask[None, :]
    lse_offsets = (
        split_idx[:, None] * h * TAIL_ROWS + head_idx * TAIL_ROWS + compact_idx[None, :]
    )
    split_lse = tl.load(lse_splits_ptr + lse_offsets, mask=valid, other=float("-inf"))
    valid_lse = valid & (split_lse > float("-inf")) & (split_lse < float("inf"))
    max_lse = tl.max(tl.where(valid_lse, split_lse, float("-inf")), axis=0)
    has_valid_lse = max_lse > float("-inf")
    safe_max_lse = tl.where(has_valid_lse, max_lse, 0.0)
    split_scale = tl.where(valid_lse, tl.exp(split_lse - safe_max_lse[None, :]), 0.0)
    scale_sum = tl.sum(split_scale, axis=0)
    safe_scale_sum = tl.where(has_valid_lse, scale_sum, 1.0)
    col = tl.arange(0, BLOCK_K)
    out = tl.full((4, BLOCK_K), 0.0, tl.float32)
    for split in range(NUM_SPLITS):
        lse_i = tl.load(
            lse_splits_ptr + split * h * TAIL_ROWS + head_idx * TAIL_ROWS + compact_idx,
            mask=row_mask,
            other=float("-inf"),
        )
        valid_i = row_mask & (lse_i > float("-inf")) & (lse_i < float("inf"))
        weight = tl.where(valid_i, tl.exp(lse_i - safe_max_lse) / safe_scale_sum, 0.0)
        offsets = (
            split * TAIL_ROWS * h * d
            + (compact_idx[:, None] * h + head_idx) * d
            + col[None, :]
        )
        partial = tl.load(
            out_splits_ptr + offsets,
            mask=valid_i[:, None] & (col[None, :] < d),
            other=0.0,
        )
        out += weight[:, None] * partial
    out_offsets = (
        q_idx[:, None] * o_row_stride + head_idx * o_head_stride + col[None, :]
    )
    tl.store(out_ptr + out_offsets, out, mask=row_mask[:, None] & (col[None, :] < d))
    merged_lse = tl.where(
        has_valid_lse, tl.log(safe_scale_sum) + safe_max_lse, float("-inf")
    )
    tl.store(lse_ptr + head_idx * total_q + q_idx, merged_lse, mask=row_mask)


@libentry()
@triton.jit
def fused_boundary_merge_kernel(
    out_ptr,
    lse_ptr,
    tail_out_ptr,
    tail_lse_ptr,
    prefix_out_ptr,
    prefix_lse_ptr,
    o_row_stride,
    o_head_stride,
    total_q: tl.constexpr,
    h: tl.constexpr,
    cu_q_ptr,
    cu_k_ptr,
    used_k_ptr,
    IS_SEQUSED_K: tl.constexpr,
    TAIL_ROWS: tl.constexpr,
    PREFIX_ROWS: tl.constexpr,
    PREFIX_SPLITS: tl.constexpr,
    TAIL_SPLITS: tl.constexpr,
):
    # Each CTA uses the original split count and merge arithmetic for its region.
    if tl.program_id(0) >= 32:
        _merge_region(
            out_ptr,
            lse_ptr,
            prefix_out_ptr,
            prefix_lse_ptr,
            o_row_stride,
            o_head_stride,
            total_q,
            h,
            256,
            PREFIX_SPLITS,
            16,
            256,
            cu_q_ptr,
            cu_k_ptr,
            used_k_ptr,
            IS_SEQUSED_K,
            PREFIX_ROWS,
            False,
            True,
            16,
            128,
        )
    else:
        _merge_region(
            out_ptr,
            lse_ptr,
            tail_out_ptr,
            tail_lse_ptr,
            o_row_stride,
            o_head_stride,
            total_q,
            h,
            256,
            TAIL_SPLITS,
            4,
            256,
            cu_q_ptr,
            cu_k_ptr,
            used_k_ptr,
            IS_SEQUSED_K,
            TAIL_ROWS,
            True,
            False,
            16,
            0,
        )


def launch_boundary_merge(params, *, batch_size, num_heads, total_q, tail, prefix):
    """Merge S4 aligned tails and S16 prefixes after both producers finish."""
    assert params.block_size == 16 and params.h_hk_ratio == 4
    return fused_boundary_merge_kernel[(36, num_heads, batch_size)](
        params.o_ptr,
        params.softmax_lse_ptr,
        tail[1],
        tail[2],
        prefix[1],
        prefix[2],
        params.o_row_stride,
        params.o_head_stride,
        total_q,
        num_heads,
        params.cu_seqlens_q_ptr,
        params.cu_seqlens_k_ptr,
        params.seqused_k_ptr,
        params.is_seqused_k,
        batch_size * 128,
        batch_size * 16,
        prefix[1].shape[0],
        tail[1].shape[0],
        num_warps=1,
        num_stages=1,
    )

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

"""Ascend910-optimized Qwen4 compress+norm+mrope+store fused kernel.

[KernelGen] Auto-generated and tuned for Ascend910.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_norm_mrope_store_qsa_groups_kernel(
    raw_cache_ptr,
    rope_cache_ptr,
    raw_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    norm_weight_ptr,
    cos_sin_cache_ptr,
    compressed_cache_ptr,
    stride_raw_block,
    stride_raw_token,
    stride_raw_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_table_req,
    stride_table_page,
    stride_cos_row,
    stride_cos_dim,
    stride_compressed_block,
    stride_compressed_token,
    stride_compressed_dim,
    num_rows,
    num_raw_blocks,
    num_rope_blocks,
    num_compressed_blocks,
    num_requests,
    num_cos_rows,
    norm_eps,
    RAW_PAGE_SIZE: tl.constexpr,
    RAW_TABLE_WIDTH: tl.constexpr,
    ROPE_PAGE_SIZE: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MROPE_SECTION_T: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
    MROPE_INTERLEAVED: tl.constexpr,
    LOAD_MROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
        & (compressed_slot < num_compressed_blocks * COMPRESSED_PAGE_SIZE)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if valid_row:
        for group_offset in tl.range(0, COMPRESS_RATIO):
            position = end_position - (COMPRESS_RATIO - 1 - group_offset)
            logical_page = position // RAW_PAGE_SIZE
            page_offset = position % RAW_PAGE_SIZE
            valid = logical_page < RAW_TABLE_WIDTH
            physical_page = tl.load(
                raw_table_ptr
                + request * stride_table_req
                + tl.minimum(logical_page, RAW_TABLE_WIDTH - 1) * stride_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_raw_blocks)
            accumulator += tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_block
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)

    pooled = (accumulator / COMPRESS_RATIO).to(tl.bfloat16)
    pooled_fp32 = pooled.to(tl.float32)
    variance = tl.sum(pooled_fp32 * pooled_fp32, axis=0) / HEAD_DIM
    weight = tl.load(
        norm_weight_ptr + dims,
        mask=dims < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    normalized = (pooled_fp32 * tl.rsqrt(variance + norm_eps) * (weight + 1.0)).to(
        tl.bfloat16
    )

    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_MROPE_POSITIONS:
        rope_page = first_position // ROPE_PAGE_SIZE
        rope_offset = first_position % ROPE_PAGE_SIZE
        valid_rope = valid_row & (rope_page < RAW_TABLE_WIDTH)
        rope_physical_page = tl.load(
            raw_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_requests - 1) * stride_table_req
            + tl.minimum(rope_page, RAW_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid_rope,
            other=-1,
        )
        valid_rope &= (rope_physical_page >= 0) & (rope_physical_page < num_rope_blocks)
        axis_offsets = tl.arange(0, 4)
        axis_positions = tl.load(
            rope_cache_ptr
            + tl.maximum(rope_physical_page, 0).to(tl.int64) * stride_rope_block
            + rope_offset * stride_rope_token
            + axis_offsets * stride_rope_dim,
            mask=valid_rope & (axis_offsets < 3),
            other=0,
        )
        time_position = tl.max(tl.where(axis_offsets == 0, axis_positions, 0))
        height_position = tl.max(tl.where(axis_offsets == 1, axis_positions, 0))
        width_position = tl.max(tl.where(axis_offsets == 2, axis_positions, 0))
    else:
        time_position = first_position
        height_position = first_position
        width_position = first_position

    head_pairs = tl.permute(tl.reshape(normalized, (2, BLOCK_D // 2)), (1, 0))
    rotary_values, pass_values = tl.split(head_pairs)
    rotary_pairs = tl.permute(tl.reshape(rotary_values, (2, ROTARY_DIM // 2)), (1, 0))
    first_half, second_half = tl.split(rotary_pairs)
    frequencies = tl.arange(0, ROTARY_DIM // 2)
    if MROPE_INTERLEAVED:
        use_height = ((frequencies % 3) == 1) & (frequencies < 3 * MROPE_SECTION_H)
        use_width = ((frequencies % 3) == 2) & (frequencies < 3 * MROPE_SECTION_W)
    else:
        height_start = MROPE_SECTION_T
        width_start = height_start + MROPE_SECTION_H
        use_height = (frequencies >= height_start) & (frequencies < width_start)
        use_width = (frequencies >= width_start) & (
            frequencies < width_start + MROPE_SECTION_W
        )
    rope_positions = tl.where(
        use_height,
        height_position,
        tl.where(use_width, width_position, time_position),
    )
    valid_position = (rope_positions >= 0) & (rope_positions < num_cos_rows)
    safe_positions = tl.minimum(tl.maximum(rope_positions, 0), num_cos_rows - 1)
    cos = tl.load(
        cos_sin_cache_ptr
        + safe_positions * stride_cos_row
        + frequencies * stride_cos_dim,
        mask=valid_row & valid_position,
        other=0.0,
    )
    sin = tl.load(
        cos_sin_cache_ptr
        + safe_positions * stride_cos_row
        + (ROTARY_DIM // 2 + frequencies) * stride_cos_dim,
        mask=valid_row & valid_position,
        other=0.0,
    )
    rotated_first = (first_half * cos - second_half * sin).to(tl.bfloat16)
    rotated_second = (second_half * cos + first_half * sin).to(tl.bfloat16)

    compressed_block = tl.maximum(compressed_slot, 0) // COMPRESSED_PAGE_SIZE
    compressed_token = tl.maximum(compressed_slot, 0) % COMPRESSED_PAGE_SIZE
    compressed_base = (
        compressed_cache_ptr
        + compressed_block.to(tl.int64) * stride_compressed_block
        + compressed_token * stride_compressed_token
    )
    tl.store(
        compressed_base + frequencies * stride_compressed_dim,
        rotated_first,
        mask=valid_row & valid_position,
    )
    tl.store(
        compressed_base + (ROTARY_DIM // 2 + frequencies) * stride_compressed_dim,
        rotated_second,
        mask=valid_row & valid_position,
    )
    pass_offsets = tl.arange(0, BLOCK_D // 2)
    tl.store(
        compressed_base + (ROTARY_DIM + pass_offsets) * stride_compressed_dim,
        pass_values,
        mask=valid_row & ((ROTARY_DIM + pass_offsets) < HEAD_DIM),
    )


def qwen4_compress_norm_mrope_store_groups(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compressed_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int = 4,
    norm_eps: float = 1.0e-6,
    rotary_dim: int = 64,
    mrope_section: tuple = (11, 11, 10),
    mrope_interleaved: bool = True,
    rope_cache: torch.Tensor | None = None,
) -> None:
    """Fuse QSA group pooling, Gemma RMSNorm, interleaved MRoPE, and store."""

    tensors = [
        raw_cache,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        compressed_cache,
        norm_weight,
        cos_sin_cache,
    ]
    if rope_cache is not None:
        tensors.append(rope_cache)
    if any(t.device.type in ("cpu", "meta") for t in tensors):
        raise RuntimeError("Qwen4 QSA compression requires a Triton accelerator")

    if (
        raw_cache.ndim != 4
        or raw_cache.shape[2:] != (1, 128)
        or not all(raw_cache.shape)
    ):
        raise ValueError("Qwen4 QSA raw cache must be nonempty [blocks, page, 1, 128]")
    if (
        compressed_cache.ndim != 4
        or compressed_cache.shape[2:] != (1, 128)
        or not all(compressed_cache.shape)
    ):
        raise ValueError("Qwen4 QSA compressed cache must be [blocks, page, 1, 128]")
    if compressed_cache.dtype != raw_cache.dtype:
        raise ValueError("Qwen4 QSA raw and compressed caches must match dtype")
    if raw_block_table.ndim != 2:
        raise ValueError("Qwen4 QSA raw block table must be rank-2")
    if raw_block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA raw block table must use int32 or int64")

    rows = token_to_req.numel()
    if rows and not all(raw_block_table.shape):
        raise ValueError("Qwen4 QSA raw block table must be nonempty for nonempty rows")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("Qwen4 QSA compression metadata must match token rows")
    if (
        token_to_req.dtype not in (torch.int32, torch.int64)
        or logical_positions.dtype not in (torch.int32, torch.int64)
        or compressed_slots.dtype not in (torch.int32, torch.int64)
    ):
        raise TypeError("Qwen4 QSA compression metadata must use int32 or int64")
    if norm_weight.shape != (128,) or norm_weight.dtype != raw_cache.dtype:
        raise ValueError("Qwen4 QSA norm weight must be a same-dtype [128] vector")
    if norm_weight.stride(0) != 1:
        raise ValueError("Qwen4 QSA norm weight must be contiguous")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != rotary_dim:
        raise ValueError("Qwen4 QSA cos/sin cache must be [positions, rotary_dim]")
    if cos_sin_cache.dtype != raw_cache.dtype or not cos_sin_cache.shape[0]:
        raise ValueError(
            "Qwen4 QSA cos/sin cache must be nonempty and match cache dtype"
        )
    if (
        len(mrope_section) != 3
        or any(section < 0 for section in mrope_section)
        or rotary_dim != 64
        or sum(mrope_section) != rotary_dim // 2
    ):
        raise ValueError(
            "Qwen4 QSA MRoPE requires rotary_dim=64 and sections summing to 32"
        )
    if compress_ratio <= 0 or norm_eps <= 0:
        raise ValueError(
            "Qwen4 QSA compression ratio and norm epsilon must be positive"
        )
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != raw_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError(
            "Qwen4 QSA packed MRoPE cache must be [blocks, page, 1, 3] int64"
        )

    rows = token_to_req.numel()
    if rows == 0:
        return
    if rope_cache is None:
        rope_cache = raw_cache
        load_mrope_positions = False
    else:
        load_mrope_positions = True
    _compress_norm_mrope_store_qsa_groups_kernel[(rows,)](
        raw_cache,
        rope_cache,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        norm_weight,
        cos_sin_cache,
        compressed_cache,
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        compressed_cache.stride(3),
        rows,
        raw_cache.shape[0],
        rope_cache.shape[0],
        compressed_cache.shape[0],
        raw_block_table.shape[0],
        cos_sin_cache.shape[0],
        float(norm_eps),
        RAW_PAGE_SIZE=raw_cache.shape[1],
        RAW_TABLE_WIDTH=raw_block_table.shape[1],
        ROPE_PAGE_SIZE=rope_cache.shape[1],
        COMPRESSED_PAGE_SIZE=compressed_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=128,
        ROTARY_DIM=rotary_dim,
        BLOCK_D=128,
        MROPE_SECTION_T=mrope_section[0],
        MROPE_SECTION_H=mrope_section[1],
        MROPE_SECTION_W=mrope_section[2],
        MROPE_INTERLEAVED=mrope_interleaved,
        LOAD_MROPE_POSITIONS=load_mrope_positions,
        num_warps=4,
    )

# SPDX-License-Identifier: Apache-2.0
# Vendor-derived sections retain copyright of contributors to the vLLM project.
"""Qwen4 QSA kernels and their model-vendor comparison baselines.

The three ``qwen4_*`` fast paths are self-developed kernels extracted from the
Qwen4 production implementation.  The three ``qwen4_vendor_*`` entry points
are comparison baselines copied from the original model-vendor implementation:

* repository: ``flagos-ai/vllm-plugin-FL``;
* base commit: ``fadbba0ea59bbaa46c77b06d465321ab88b44643``;
* source: ``vllm_fl/models/qwen3_8_flash_next/gpu/ops/qsa.py``;
* source snapshot SHA256:
  ``40a3190b04af28a1bc5944e29a334b041384ee6cc12bdd8c8a7382e23b207342``.

The vendor kernels remain vendor-owned and are not counted as self-developed
FlagGems-vllm kernels.

GCU adaptation (enflame backend copy): torch_gcu silently narrows device-side
int64 tensors to 4-byte physical slots ("GCU not support Long use Int
replace"), so kernels loading them at the declared int64 width straddle two
slots per element and read garbage (compiler_issues/triton_i64_narrow_load.md;
scalar i64 address chains are equally poisoned). Each integer index/position
pointer is therefore bitcast to an int32 pointer at kernel entry — a scalar
pointer cast, the proven-safe form on this backend (qsa_prepare precedent) —
so every load/store reads the correct 4-byte slot at zero host cost. Slot
strides are numerically identical to the int64 element strides, so the launch
passes the original tensors and strides unchanged. Values stay in the i32 safe
domain by the production contract (positions/slots < 2**31); the
TORCH_GCU_ENABLE_INT64_AND_UINT64=1 true-8-byte layout is not supported (the
upstream torch reference itself aborts under it). Kernel memory access
patterns otherwise follow the generic implementation, except where the
toolchain's memref_ext.slice_start bug forces the restructures documented on
the affected kernels.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

QWEN4_VENDOR_QSA_SOURCE = {
    "repository": "https://github.com/flagos-ai/vllm-plugin-FL",
    "base_commit": "fadbba0ea59bbaa46c77b06d465321ab88b44643",
    "path": "vllm_fl/models/qwen3_8_flash_next/gpu/ops/qsa.py",
    "sha256": "40a3190b04af28a1bc5944e29a334b041384ee6cc12bdd8c8a7382e23b207342",
}


def _is_triton_device(*tensors: torch.Tensor) -> bool:
    """Return true for same-device accelerator tensors accepted by Triton."""

    return bool(
        tensors
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.device.type not in ("cpu", "meta") for tensor in tensors)
    )


# Model-vendor baseline. Source provenance is recorded in the module docstring.
@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    query_positions_ptr = query_positions_ptr.to(tl.pointer_type(tl.int32))
    sequence_lengths_ptr = sequence_lengths_ptr.to(tl.pointer_type(tl.int32))
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (row < num_rows)
        & (columns < num_columns)
        & (columns < visible)
        & (request >= 0)
        & (request < num_requests)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_req
        + safe_logical_page * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)

    score /= score_divisor
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score, -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )


@triton.jit
def _qsa_mqa_paged_dot_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    """GCU adaptation of the tensor-core/MFMA QSA score path.

    The generic ``tl.dot`` form loads a transposed ``(BLOCK_D, BLOCK_N)``
    key tile and a padded 2D query tile; on the 2026-11-07 enflame
    toolchain those freshly-compiled 2D masked loads fail with
    ``memref_ext.slice_start`` operand-segment mismatches (the
    ``slice_transpose_start`` family, see compiler_issues/). This port
    keeps the identical math (per-head ReLU dot reduction) with the
    proven-safe scalar-reduction memory pattern: 1D query loads inside a
    static head loop plus a row-major 2D key tile, all lane-affine.

    Performance retune (2026-11-10, GCU probes; bit-exact vs the previous
    scalar-reduction port): floor probes showed the four redundant per-head
    key loads cost nothing measurable (4x loads hit cache, ``floor1 ==
    floor4``), while the per-tile fixed machinery dominated. Two levers
    landed: (1) hot loads are unmasked — the page-table entry is read at a
    clamped logical index and the physical page is clamped into
    ``[0, num_pages-1]`` so every key address stays in-bounds even for dead
    lanes (invalid pages are still detected from the raw loaded value and
    masked out at the store via ``-inf``, matching the reference
    semantics); (2) the launch config is rows-aware (see
    ``_qsa_dot_launch_config``): wide tiles amortize the fixed cost for
    batch/prefill rows while single-row decode keeps 32-column tiles.
    ``HEAD_DIM != BLOCK_D`` keeps the original masked loads.
    """

    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    query_positions_ptr = query_positions_ptr.to(tl.pointer_type(tl.int32))
    sequence_lengths_ptr = sequence_lengths_ptr.to(tl.pointer_type(tl.int32))
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    page_valid = (
        (request >= 0) & (request < num_requests) & (logical_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page
    )
    valid = (
        (row < num_rows)
        & (columns < num_columns)
        & (columns < visible)
        & page_valid
        & (physical_page >= 0)
        & (physical_page < num_pages)
    )
    safe_page = (
        tl.minimum(tl.maximum(physical_page, 0), num_pages - 1).to(tl.int64)
        * stride_cache_block
    )
    if HEAD_DIM == BLOCK_D:
        keys = tl.load(
            k_cache_ptr
            + safe_page[:, None]
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim
        ).to(tl.float32)
        score = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for head in tl.static_range(0, NUM_HEADS):
            query = tl.load(
                q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim
            ).to(tl.float32)
            dot = tl.sum(keys * query[None, :], axis=1)
            score += tl.maximum(dot, 0.0)
    else:
        keys = tl.load(
            k_cache_ptr
            + safe_page[:, None]
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=(physical_page[:, None] >= 0)
            & (physical_page[:, None] < num_pages)
            & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        score = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for head in tl.static_range(0, NUM_HEADS):
            query = tl.load(
                q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
                mask=dims < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
            dot = tl.sum(keys * query[None, :], axis=1)
            score += tl.maximum(dot, 0.0)
    score /= score_divisor
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score, -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )


def _qsa_dot_launch_config(rows: int, columns: int) -> tuple[int, int]:
    """Pick the GCU (BLOCK_N, num_warps) for the QSA dot kernel.

    Probe-matrix medians (kernel-only, S60, columns=1024): wide tiles
    amortize the per-tile fixed cost as rows grow, while single-row decode
    keeps parallelism with narrow tiles — rows=1: 32/w4 9.3us; rows=2: 64/w4
    11.9us; rows=4: 128/w1 15.9us; rows=8-16: 256/w2 24.5-43.3us;
    rows>=24: 512/w4 48.3-137.8us. The tile is capped to the padded column
    count so short score widths do not launch mostly-dead wide tiles.
    """

    if rows >= 24:
        block_n, num_warps = 512, 4
    elif rows >= 8:
        block_n, num_warps = 256, 2
    elif rows >= 4:
        block_n, num_warps = 128, 1
    elif rows >= 2:
        block_n, num_warps = 64, 4
    else:
        block_n, num_warps = 32, 4
    return min(block_n, max(16, triton.next_power_of_2(columns))), num_warps


# Model-vendor baseline. Source provenance is recorded in the module docstring.
@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slots_ptr = slots_ptr.to(tl.pointer_type(tl.int32))
    slot = tl.load(slots_ptr + row).to(tl.int64)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _store_qsa_kv_rows_kernel(
    k_cache_ptr,
    v_cache_ptr,
    slots_ptr,
    k_rows_ptr,
    v_rows_ptr,
    stride_k_cache_block,
    stride_k_cache_token,
    stride_k_cache_head,
    stride_k_cache_dim,
    stride_v_cache_block,
    stride_v_cache_token,
    stride_v_cache_head,
    stride_v_cache_dim,
    stride_k_rows_row,
    stride_k_rows_head,
    stride_k_rows_dim,
    stride_v_rows_row,
    stride_v_rows_head,
    stride_v_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Store K and V together while preserving arbitrary cache strides."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    slots_ptr = slots_ptr.to(tl.pointer_type(tl.int32))
    slot = tl.load(slots_ptr + row).to(tl.int64)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    k_values = tl.load(
        k_rows_ptr
        + row * stride_k_rows_row
        + head * stride_k_rows_head
        + dims * stride_k_rows_dim,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
        other=0,
    )
    v_values = tl.load(
        v_rows_ptr
        + row * stride_v_rows_row
        + head * stride_v_rows_head
        + dims * stride_v_rows_dim,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
        other=0,
    )
    tl.store(
        k_cache_ptr
        + block * stride_k_cache_block
        + token * stride_k_cache_token
        + head * stride_k_cache_head
        + dims * stride_k_cache_dim,
        k_values,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
    )
    tl.store(
        v_cache_ptr
        + block * stride_v_cache_block
        + token * stride_v_cache_token
        + head * stride_v_cache_head
        + dims * stride_v_cache_dim,
        v_values,
        mask=valid & (head < NUM_HEADS) & (dims < HEAD_DIM),
    )


# Model-vendor baseline. Source provenance is recorded in the module docstring.
@triton.jit
def _compress_qsa_groups_kernel(
    raw_cache_ptr,
    rope_cache_ptr,
    raw_table_ptr,
    rope_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_block,
    stride_raw_token,
    stride_raw_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_raw_table_req,
    stride_raw_table_page,
    stride_rope_table_req,
    stride_rope_table_page,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_raw_blocks,
    num_rope_blocks,
    num_raw_requests,
    num_rope_requests,
    RAW_PAGE_SIZE: tl.constexpr,
    RAW_TABLE_WIDTH: tl.constexpr,
    ROPE_PAGE_SIZE: tl.constexpr,
    ROPE_TABLE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    logical_positions_ptr = logical_positions_ptr.to(tl.pointer_type(tl.int32))
    compressed_slots_ptr = compressed_slots_ptr.to(tl.pointer_type(tl.int32))
    first_positions_ptr = first_positions_ptr.to(tl.pointer_type(tl.int32))
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row).to(tl.int64)
    compressed_slot = tl.load(compressed_slots_ptr + row).to(tl.int64)
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_raw_requests)
        & (request < num_rope_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
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
                + request * stride_raw_table_req
                + tl.minimum(logical_page, RAW_TABLE_WIDTH - 1) * stride_raw_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_raw_blocks)
            values = tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_block
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            accumulator += values

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        rope_cache_ptr = rope_cache_ptr.to(tl.pointer_type(tl.int32))
        rope_logical_page = first_position // ROPE_PAGE_SIZE
        rope_page_offset = first_position % ROPE_PAGE_SIZE
        valid_rope = valid_row & (rope_logical_page < ROPE_TABLE_WIDTH)
        rope_physical_page = tl.load(
            rope_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_rope_requests - 1)
            * stride_rope_table_req
            + tl.minimum(rope_logical_page, ROPE_TABLE_WIDTH - 1)
            * stride_rope_table_page,
            mask=valid_rope,
            other=-1,
        )
        valid_rope &= (rope_physical_page >= 0) & (rope_physical_page < num_rope_blocks)
        rope_values = tl.load(
            rope_cache_ptr
            + tl.maximum(rope_physical_page, 0).to(tl.int64) * stride_rope_block
            + rope_page_offset * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_rope & (position_dims < 3),
            other=0,
        )
        tl.store(
            first_positions_ptr
            + row * stride_positions_row
            + position_dims * stride_positions_dim,
            rope_values,
            mask=(row < num_rows) & (position_dims < 3),
        )
    else:
        first_position = tl.where(valid_row, first_position, 0)
        tl.store(
            first_positions_ptr
            + row * stride_positions_row
            + position_dims * stride_positions_dim,
            first_position.to(tl.int32),
            mask=(row < num_rows) & (position_dims < 3),
        )


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
    """Pool, Gemma-normalize, MRoPE, and store one compressed QSA key.

    GCU adaptation: the generic kernel pools into one ``(BLOCK_D,)``
    accumulator and extracts the NeoX rotation halves via
    ``tl.reshape``/``tl.permute``/``tl.split``; that register-shuffle
    chain fails to compile on the 2026-11-07 enflame toolchain
    (``memref_ext.slice_start`` operand-segment mismatch). This port
    pools three lane-affine segments instead (rotary first half, rotary
    second half, pass-through tail) and rotates them pairwise without
    any register permutation, mirroring the rotary_embedding enflame
    adaptation. Numerics follow the same BF16 materialization steps.
    """

    row = tl.program_id(0)
    first_offsets = tl.arange(0, ROTARY_DIM // 2)
    second_offsets = ROTARY_DIM // 2 + first_offsets
    tail_offsets = ROTARY_DIM + tl.arange(0, BLOCK_D - ROTARY_DIM)
    logical_positions_ptr = logical_positions_ptr.to(tl.pointer_type(tl.int32))
    compressed_slots_ptr = compressed_slots_ptr.to(tl.pointer_type(tl.int32))
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
    acc_first = tl.zeros((ROTARY_DIM // 2,), dtype=tl.float32)
    acc_second = tl.zeros((ROTARY_DIM // 2,), dtype=tl.float32)
    acc_tail = tl.zeros((BLOCK_D - ROTARY_DIM,), dtype=tl.float32)
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
            row_base = (
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_block
                + page_offset * stride_raw_token
            )
            acc_first += tl.load(
                row_base + first_offsets * stride_raw_dim,
                mask=valid & (first_offsets < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            acc_second += tl.load(
                row_base + second_offsets * stride_raw_dim,
                mask=valid & (second_offsets < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            acc_tail += tl.load(
                row_base + tail_offsets * stride_raw_dim,
                mask=valid & (tail_offsets < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)

    # Match the unfused BF16 materialization before Gemma RMSNorm.
    pooled_first = (acc_first / COMPRESS_RATIO).to(tl.bfloat16)
    pooled_second = (acc_second / COMPRESS_RATIO).to(tl.bfloat16)
    pooled_tail = (acc_tail / COMPRESS_RATIO).to(tl.bfloat16)
    pooled_first_fp32 = pooled_first.to(tl.float32)
    pooled_second_fp32 = pooled_second.to(tl.float32)
    pooled_tail_fp32 = pooled_tail.to(tl.float32)
    variance = (
        tl.sum(pooled_first_fp32 * pooled_first_fp32, axis=0)
        + tl.sum(pooled_second_fp32 * pooled_second_fp32, axis=0)
        + tl.sum(pooled_tail_fp32 * pooled_tail_fp32, axis=0)
    ) / HEAD_DIM
    scale = tl.rsqrt(variance + norm_eps)
    weight_first = tl.load(
        norm_weight_ptr + first_offsets,
        mask=first_offsets < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    weight_second = tl.load(
        norm_weight_ptr + second_offsets,
        mask=second_offsets < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    weight_tail = tl.load(
        norm_weight_ptr + tail_offsets,
        mask=tail_offsets < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    normalized_first = (pooled_first_fp32 * scale * (weight_first + 1.0)).to(
        tl.bfloat16
    )
    normalized_second = (pooled_second_fp32 * scale * (weight_second + 1.0)).to(
        tl.bfloat16
    )
    normalized_tail = (pooled_tail_fp32 * scale * (weight_tail + 1.0)).to(tl.bfloat16)

    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_MROPE_POSITIONS:
        rope_cache_ptr = rope_cache_ptr.to(tl.pointer_type(tl.int32))
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

    # HEAD_DIM=128 and ROTARY_DIM=64 in the production checkpoint.  The
    # rotary halves stay in their lane-affine segments (no register
    # permutation): first segment pairs with the second through the
    # per-frequency cos/sin, the tail passes through unchanged.
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
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr
        + safe_positions * stride_cos_row
        + (ROTARY_DIM // 2 + frequencies) * stride_cos_dim,
        mask=valid_row & valid_position,
        other=0.0,
    ).to(tl.float32)
    first_fp32 = normalized_first.to(tl.float32)
    second_fp32 = normalized_second.to(tl.float32)
    rotated_first = (first_fp32 * cos - second_fp32 * sin).to(tl.bfloat16)
    rotated_second = (second_fp32 * cos + first_fp32 * sin).to(tl.bfloat16)

    compressed_block = tl.maximum(compressed_slot, 0) // COMPRESSED_PAGE_SIZE
    compressed_token = tl.maximum(compressed_slot, 0) % COMPRESSED_PAGE_SIZE
    compressed_base = (
        compressed_cache_ptr
        + compressed_block.to(tl.int64) * stride_compressed_block
        + compressed_token * stride_compressed_token
    )
    tl.store(
        compressed_base + first_offsets * stride_compressed_dim,
        rotated_first,
        mask=valid_row & valid_position,
    )
    tl.store(
        compressed_base + second_offsets * stride_compressed_dim,
        rotated_second,
        mask=valid_row & valid_position,
    )
    tl.store(
        compressed_base + tail_offsets * stride_compressed_dim,
        normalized_tail,
        mask=valid_row & (tail_offsets < HEAD_DIM),
    )


def qwen4_vendor_qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int = 4,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model-vendor scalar-reduction QSA paged-MQA kernel."""

    if not _is_triton_device(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths
    ):
        raise RuntimeError("Qwen4 vendor QSA MQA requires a Triton accelerator")
    if q.ndim != 3 or q.shape[1:] != (4, 128) or q.dtype != torch.bfloat16:
        raise ValueError("Qwen4 vendor QSA MQA requires BF16 q shaped [rows, 4, 128]")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        raise ValueError(
            "Qwen4 vendor QSA MQA cache must be [pages, page_size, 1, 128]"
        )
    if k_cache.dtype != q.dtype:
        raise ValueError("Qwen4 vendor QSA query and cache must have the same dtype")
    if page_table.ndim != 2:
        raise ValueError("Qwen4 vendor QSA MQA page table must be rank-2")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 vendor QSA page table must use int32 or int64")
    rows = q.shape[0]
    if rows and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError(
            "Qwen4 vendor QSA cache and page table must be nonempty for nonempty q"
        )
    if token_to_req.shape != (rows,) or query_positions.shape != (rows,):
        raise ValueError("Qwen4 vendor QSA request metadata must match query rows")
    if token_to_req.dtype not in (
        torch.int32,
        torch.int64,
    ) or query_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("Qwen4 vendor QSA request metadata must use int32 or int64")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError(
            "Qwen4 vendor QSA sequence lengths must match page-table requests"
        )
    if sequence_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 vendor QSA sequence lengths must use int32 or int64")
    if compress_ratio <= 0:
        raise ValueError("Qwen4 vendor QSA compression ratio must be positive")
    divisor = math.sqrt(128) if score_scale is None else float(score_scale)
    if divisor <= 0:
        raise ValueError("Qwen4 vendor QSA score scale must be positive")
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else int(num_columns)
    if columns < 0 or columns > capacity:
        raise ValueError(
            "Qwen4 vendor QSA score width must be in [0, page-table capacity]"
        )
    logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty((rows,), dtype=torch.int32, device=q.device)
    if rows == 0 or columns == 0:
        return logits, visible_blocks
    block_n = 32
    _qsa_mqa_paged_kernel[(rows, triton.cdiv(columns, block_n))](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        rows,
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        divisor,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=block_n,
        BLOCK_D=128,
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
    )
    return logits, visible_blocks


def qwen4_qsa_mqa_paged_dot(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int = 4,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA indexer scores directly from a paged compressed-key cache.

    This entry point intentionally selects the self-developed ``tl.dot``
    kernel.  The Qwen4 production contract is BF16, four query heads, and
    head dimension 128; callers with another shape must use a model/vendor
    implementation instead of silently falling back to Torch.
    """

    if not _is_triton_device(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths
    ):
        raise RuntimeError("Qwen4 QSA MQA dot requires a Triton accelerator")
    if q.ndim != 3 or q.shape[1:] != (4, 128) or q.dtype != torch.bfloat16:
        raise ValueError("Qwen4 QSA MQA dot requires BF16 q shaped [rows, 4, 128]")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        raise ValueError("Qwen4 QSA MQA cache must be [pages, page_size, 1, 128]")
    if k_cache.dtype != q.dtype:
        raise ValueError("Qwen4 QSA query and cache must have the same dtype")
    if page_table.ndim != 2:
        raise ValueError("Qwen4 QSA MQA page table must be rank-2")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA page table must use int32 or int64")
    rows = q.shape[0]
    if rows and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError(
            "Qwen4 QSA MQA cache and page table must be nonempty for nonempty q"
        )
    if token_to_req.shape != (rows,) or query_positions.shape != (rows,):
        raise ValueError("Qwen4 QSA request metadata must match query rows")
    if token_to_req.dtype not in (
        torch.int32,
        torch.int64,
    ) or query_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("Qwen4 QSA request metadata must use int32 or int64")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("Qwen4 QSA sequence lengths must match page-table requests")
    if sequence_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA sequence lengths must use int32 or int64")
    if compress_ratio <= 0:
        raise ValueError("Qwen4 QSA compression ratio must be positive")
    divisor = math.sqrt(128) if score_scale is None else float(score_scale)
    if divisor <= 0:
        raise ValueError("Qwen4 QSA score scale must be positive")
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else int(num_columns)
    if columns < 0 or columns > capacity:
        raise ValueError("Qwen4 QSA score width must be in [0, page-table capacity]")
    logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty((rows,), dtype=torch.int32, device=q.device)
    if rows == 0 or columns == 0:
        return logits, visible_blocks
    block_n, num_warps = _qsa_dot_launch_config(rows, columns)
    _qsa_mqa_paged_dot_kernel[(rows, triton.cdiv(columns, block_n))](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        rows,
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        divisor,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_H=16,
        BLOCK_N=block_n,
        BLOCK_D=128,
        COMPRESS_RATIO=compress_ratio,
        num_warps=num_warps,
    )
    return logits, visible_blocks


def qwen4_vendor_store_qsa_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Run the model-vendor single-cache-row QSA store kernel."""

    if not _is_triton_device(cache, slot_mapping, rows):
        raise RuntimeError("Qwen4 vendor QSA store requires a Triton accelerator")
    if cache.ndim != 4 or cache.shape[2] != 1 or not all(cache.shape):
        raise ValueError(
            "Qwen4 vendor QSA cache must be nonempty [pages, page_size, 1, width]"
        )
    if slot_mapping.ndim != 1 or slot_mapping.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("Qwen4 vendor QSA slots must be a rank-1 int32/int64 tensor")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("Qwen4 vendor QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("Qwen4 vendor QSA cache rows and slots are incompatible")
    if rows.dtype != cache.dtype:
        raise ValueError("Qwen4 vendor QSA rows and cache must have matching dtypes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qwen4_store_qsa_kv_rows(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Store paired QSA K/V rows in one stride-aware Triton launch.

    The production fast-path precondition is that valid slots are unique.  An
    invalid slot is safely ignored by the kernel; duplicate slots are outside
    the parallel write contract and must be removed by the caller.
    """

    if not _is_triton_device(k_cache, v_cache, slot_mapping, key, value):
        raise RuntimeError("Qwen4 QSA K/V store requires a Triton accelerator")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("Qwen4 QSA K/V caches must be [blocks, page, heads, dim]")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("Qwen4 QSA K/V rows must be [rows, heads, dim]")
    if key.shape != (slot_mapping.numel(), k_cache.shape[2], k_cache.shape[3]):
        raise ValueError("Qwen4 QSA K/V rows and slot mapping have incompatible shapes")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 QSA slot mapping must use int32 or int64")
    if key.dtype != k_cache.dtype or value.dtype != v_cache.dtype:
        raise ValueError("Qwen4 QSA K/V rows and caches must have matching dtypes")
    if not all(k_cache.shape):
        raise ValueError("Qwen4 QSA K/V caches must be nonempty")
    if not key.shape[0]:
        return
    _store_qsa_kv_rows_kernel[(key.shape[0], key.shape[1])](
        k_cache,
        v_cache,
        slot_mapping,
        key,
        value,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key.shape[0],
        k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=key.shape[1],
        HEAD_DIM=key.shape[2],
        BLOCK_D=triton.next_power_of_2(key.shape[2]),
        num_warps=4,
    )


def qwen4_vendor_compress_qsa_groups(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int = 4,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model-vendor QSA group-pooling and position-load kernel."""

    tensors = [
        raw_cache,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
    ]
    if rope_cache is not None:
        tensors.append(rope_cache)
    if not _is_triton_device(*tensors):
        raise RuntimeError("Qwen4 vendor QSA compression requires a Triton accelerator")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("Qwen4 vendor QSA compression ratio must be positive")
    if (
        raw_cache.ndim != 4
        or raw_cache.shape[2:] != (1, 128)
        or raw_cache.dtype != torch.bfloat16
    ):
        raise ValueError(
            "Qwen4 vendor QSA raw cache must be BF16 [blocks, page, 1, 128]"
        )
    if raw_block_table.ndim != 2 or raw_block_table.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("Qwen4 vendor QSA block table must be rank-2 int32/int64")
    if rows and (not all(raw_cache.shape) or not all(raw_block_table.shape)):
        raise ValueError("Qwen4 vendor QSA raw cache and block table must be nonempty")
    if token_to_req.dtype not in (
        torch.int32,
        torch.int64,
    ) or logical_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("Qwen4 vendor QSA request metadata must use int32 or int64")
    if compressed_slots.dtype not in (torch.int32, torch.int64):
        raise TypeError("Qwen4 vendor QSA compressed slots must use int32 or int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("Qwen4 vendor QSA compression metadata must match rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != raw_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError(
            "Qwen4 vendor QSA packed position view must be [blocks, page, 1, 3] int64"
        )
    pooled = torch.empty(
        (rows, 1, raw_cache.shape[3]),
        dtype=raw_cache.dtype,
        device=raw_cache.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_cache.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = raw_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_cache,
        rope_cache,
        raw_block_table,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        raw_cache.shape[0],
        rope_cache.shape[0],
        raw_block_table.shape[0],
        raw_block_table.shape[0],
        RAW_PAGE_SIZE=raw_cache.shape[1],
        RAW_TABLE_WIDTH=raw_block_table.shape[1],
        ROPE_PAGE_SIZE=rope_cache.shape[1],
        ROPE_TABLE_WIDTH=raw_block_table.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_cache.shape[3],
        BLOCK_D=triton.next_power_of_2(raw_cache.shape[3]),
        LOAD_ROPE_POSITIONS=load_rope_positions,
        num_warps=4,
    )
    return pooled, first_positions


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
    mrope_section: tuple[int, int, int] = (11, 11, 10),
    mrope_interleaved: bool = True,
    rope_cache: torch.Tensor | None = None,
) -> None:
    """Fuse QSA group pooling, Gemma RMSNorm, interleaved MRoPE, and store.

    For the Qwen4 cache, ``mrope_section=(11, 11, 10)`` describes counts of
    frequency lanes.  With ``mrope_interleaved=True`` the actual lane selector
    is ``frequency % 3`` (T/H/W), not three contiguous slices.
    """

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
    if not _is_triton_device(*tensors):
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
        or logical_positions.dtype
        not in (
            torch.int32,
            torch.int64,
        )
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


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def _use_qsa_mqa_dot(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """Gate the portable dot path where it wins without hurting single-row decode."""

    return (
        q.shape[0] >= 8
        and q.shape[1:] == (4, 128)
        and q.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
    )


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache."""

    _validate_mqa(q)
    if not _is_triton_device(q):
        raise NotImplementedError(
            "paged QSA scoring requires an accelerator Triton backend"
        )
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if score_divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0:
        raise ValueError("QSA score width must be non-negative")
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    use_dot = _use_qsa_mqa_dot(q, k_cache)
    if use_dot:
        block_n, num_warps = _qsa_dot_launch_config(q.shape[0], columns)
        kernel = _qsa_mqa_paged_dot_kernel
        extra_kwargs = {"BLOCK_H": 16}
    else:
        block_n, num_warps = 32, 4
        kernel = _qsa_mqa_paged_kernel
        extra_kwargs = {}
    kernel[(q.shape[0], triton.cdiv(columns, block_n))](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        COMPRESS_RATIO=compress_ratio,
        num_warps=num_warps,
        **extra_kwargs,
    )
    return logits, visible_blocks


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not _is_triton_device(cache):
        raise NotImplementedError(
            "QSA cache stores require an accelerator Triton backend"
        )
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_store_kv_cache_rows(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Store paired K/V rows with one cross-vendor Triton launch."""

    if not _is_triton_device(k_cache):
        raise NotImplementedError(
            "QSA K/V cache stores require an accelerator Triton backend"
        )
    if (
        k_cache.ndim != 4
        or v_cache.shape != k_cache.shape
        or key.ndim != 3
        or value.shape != key.shape
    ):
        raise ValueError("QSA K/V cache store received invalid tensor shapes")
    if key.shape != (slot_mapping.numel(), k_cache.shape[2], k_cache.shape[3]):
        raise ValueError("QSA K/V rows and cache geometry are incompatible")
    if key.dtype != k_cache.dtype or value.dtype != v_cache.dtype:
        raise ValueError("QSA K/V rows and caches must have matching dtypes")
    if not all(k_cache.shape):
        raise ValueError("QSA K/V cache dimensions must be nonzero")
    if not key.shape[0]:
        return
    _store_qsa_kv_rows_kernel[(key.shape[0], key.shape[1])](
        k_cache,
        v_cache,
        slot_mapping,
        key,
        value,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key.shape[0],
        k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=key.shape[1],
        HEAD_DIM=key.shape[2],
        BLOCK_D=triton.next_power_of_2(key.shape[2]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool raw-key groups and load their packed or derived positions."""

    if not _is_triton_device(raw_cache):
        raise NotImplementedError(
            "QSA compression requires an accelerator Triton backend"
        )
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if raw_cache.ndim != 4 or raw_cache.shape[2] != 1:
        raise ValueError("QSA raw cache has an invalid shape")
    if raw_block_table.ndim != 2:
        raise ValueError("QSA raw compression block table must be rank two")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != raw_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (not all(raw_cache.shape) or not all(raw_block_table.shape)):
        raise ValueError("QSA raw cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_cache.shape[3]), dtype=raw_cache.dtype, device=raw_cache.device
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_cache.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = raw_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_cache,
        rope_cache,
        raw_block_table,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        raw_cache.shape[0],
        rope_cache.shape[0],
        raw_block_table.shape[0],
        raw_block_table.shape[0],
        RAW_PAGE_SIZE=raw_cache.shape[1],
        RAW_TABLE_WIDTH=raw_block_table.shape[1],
        ROPE_PAGE_SIZE=rope_cache.shape[1],
        ROPE_TABLE_WIDTH=raw_block_table.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_cache.shape[3],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_cache.shape[3]),
        num_warps=4,
    )
    return pooled, first_positions


def qsa_compress_norm_mrope_store_groups(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compressed_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
    norm_eps: float,
    rotary_dim: int,
    mrope_section: tuple[int, int, int],
    mrope_interleaved: bool,
    rope_cache: torch.Tensor | None = None,
) -> None:
    """Fuse QSA compression, Gemma norm, Neox MRoPE, and cache insert."""

    if not _is_triton_device(raw_cache):
        raise NotImplementedError("fused QSA compression requires accelerator Triton")
    rows = token_to_req.numel()
    if raw_cache.ndim != 4 or raw_cache.shape[2] != 1 or not all(raw_cache.shape):
        raise ValueError("fused QSA compression received an invalid raw cache")
    head_dim = raw_cache.shape[3]
    if head_dim != 128 or rotary_dim != 64:
        raise ValueError("fused QSA compression requires head_dim=128, rotary_dim=64")
    if (
        compressed_cache.ndim != 4
        or compressed_cache.shape[2:] != (1, head_dim)
        or not all(compressed_cache.shape)
        or compressed_cache.device != raw_cache.device
        or compressed_cache.dtype != raw_cache.dtype
    ):
        raise ValueError("fused QSA compressed cache must match the raw key width")
    if raw_block_table.ndim != 2 or not all(raw_block_table.shape):
        raise ValueError("fused QSA compression requires a nonempty block table")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("fused QSA compression metadata must match token rows")
    if (
        norm_weight.shape != (head_dim,)
        or norm_weight.dtype != raw_cache.dtype
        or norm_weight.device != raw_cache.device
        or norm_weight.stride(0) != 1
    ):
        raise ValueError("fused QSA compression norm weight is incompatible")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or not cos_sin_cache.shape[0]
        or cos_sin_cache.device != raw_cache.device
        or cos_sin_cache.dtype != raw_cache.dtype
    ):
        raise ValueError("fused QSA compression received an invalid RoPE cache")
    if sum(mrope_section) != rotary_dim // 2:
        raise ValueError("fused QSA MRoPE sections must cover half the rotary dim")
    if compress_ratio <= 0 or norm_eps <= 0:
        raise ValueError("fused QSA compression ratio and epsilon must be positive")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != raw_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
        or rope_cache.device != raw_cache.device
    ):
        raise ValueError("fused QSA packed MRoPE cache is invalid")
    if not rows:
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
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        MROPE_SECTION_T=mrope_section[0],
        MROPE_SECTION_H=mrope_section[1],
        MROPE_SECTION_W=mrope_section[2],
        MROPE_INTERLEAVED=mrope_interleaved,
        LOAD_MROPE_POSITIONS=load_mrope_positions,
        num_warps=4,
    )


_PUBLIC_OPS = (
    "qwen4_vendor_qsa_mqa_paged",
    "qwen4_qsa_mqa_paged_dot",
    "qwen4_vendor_store_qsa_rows",
    "qwen4_store_qsa_kv_rows",
    "qwen4_vendor_compress_qsa_groups",
    "qwen4_compress_norm_mrope_store_groups",
    "qsa_mqa_paged",
    "qsa_store_cache_rows",
    "qsa_store_kv_cache_rows",
    "qsa_compress_groups_with_ratio",
    "qsa_compress_norm_mrope_store_groups",
)


def _install() -> None:
    """Swap the GCU-adapted entry points into the generic module.

    Tests and benchmarks import ``from flaggems_vllm.ops.qwen4.qsa import
    ...``, which resolves against the already-imported generic module, so
    the SpecOpRegistrar top-level namespace swap alone cannot reach them.
    Replace the attributes on that module in place, mirroring the kpool
    backend override precedent: only a pristine generic function is
    swapped, so explicit user or test patches always win.
    """
    import sys

    mod = sys.modules.get("flaggems_vllm.ops.qwen4.qsa")
    if mod is None:
        return
    for name in _PUBLIC_OPS:
        current = getattr(mod, name, None)
        if current is not None and getattr(current, "__module__", None) == (
            "flaggems_vllm.ops.qwen4.qsa"
        ):
            setattr(mod, name, globals()[name])


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
__all__ = [
    "QWEN4_VENDOR_QSA_SOURCE",
    "_compress_qsa_groups_kernel",
    "_compress_norm_mrope_store_qsa_groups_kernel",
    "_qsa_mqa_paged_kernel",
    "_qsa_mqa_paged_dot_kernel",
    "_store_qsa_rows_kernel",
    "_store_qsa_kv_rows_kernel",
    "qwen4_compress_norm_mrope_store_groups",
    "qwen4_qsa_mqa_paged_dot",
    "qwen4_store_qsa_kv_rows",
    "qwen4_vendor_compress_qsa_groups",
    "qwen4_vendor_qsa_mqa_paged",
    "qwen4_vendor_store_qsa_rows",
    "qsa_mqa_paged",
    "qsa_store_cache_rows",
    "qsa_store_kv_cache_rows",
    "qsa_compress_groups_with_ratio",
    "qsa_compress_norm_mrope_store_groups",
]

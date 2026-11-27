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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
)

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store


@triton.jit
def _generic_cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,
    kv_cache_scale_ptr,
    k_fp8_ptr,
    k_scale_ptr,
    block_table_ptr,
    cu_seqlen_ptr,
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    k_fp8_stride,
    num_quant_blocks,
    batch_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    BATCH_SCAN_SIZE: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    tid = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    quant_block_id = tl.program_id(1)

    if batch_size <= 16:
        batch_offsets = tl.arange(0, BATCH_SCAN_SIZE)
        batch_mask = batch_offsets < batch_size
        seq_starts = tl.load(cu_seqlen_ptr + batch_offsets, mask=batch_mask, other=0)
        seq_ends = tl.load(cu_seqlen_ptr + batch_offsets + 1, mask=batch_mask, other=0)
        in_batch = (
            (tid[:, None] >= seq_starts[None, :])
            & (tid[:, None] < seq_ends[None, :])
            & batch_mask[None, :]
        )
        batch_id = tl.max(tl.where(in_batch, batch_offsets[None, :], -1), axis=1)
    else:
        left = tl.full((TOKEN_BLOCK,), 0, dtype=tl.int32)
        right = tl.full((TOKEN_BLOCK,), batch_size + 1, dtype=tl.int32)
        for _ in tl.static_range(0, SEARCH_STEPS):
            mid = (left + right) // 2
            seq_start = tl.load(
                cu_seqlen_ptr + mid,
                mask=mid <= batch_size,
                other=2147483647,
            )
            seq_start_before_token = seq_start <= tid
            left = tl.where(seq_start_before_token, mid + 1, left)
            right = tl.where(seq_start_before_token, right, mid)
        batch_id = left - 1
    valid_batch = (batch_id >= 0) & (batch_id < batch_size)
    safe_batch_id = tl.minimum(tl.maximum(batch_id, 0), batch_size - 1)
    batch_start = tl.load(cu_seqlen_ptr + safe_batch_id, mask=valid_batch, other=0)
    batch_end = tl.load(cu_seqlen_ptr + safe_batch_id + 1, mask=valid_batch, other=0)
    valid_tokens = valid_batch & (tid >= batch_start) & (tid < batch_end)
    batch_offset = tid - batch_start
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = safe_batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset, mask=valid_tokens, other=0)

    offsets = quant_block_id * QUANT_BLOCK_SIZE + tl.arange(0, QUANT_BLOCK_SIZE)
    mask = valid_tokens[:, None]
    src_cache_offset = (
        block_id[:, None].to(tl.int64) * kv_cache_stride
        + block_offset[:, None].to(tl.int64) * HEAD_DIM
    )
    src_scale_offset = (
        block_id * kv_cache_scale_stride
        + block_offset * num_quant_blocks
        + quant_block_id
    )
    dst_offset = tid[:, None].to(tl.int64) * k_fp8_stride

    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset

    scale_val = tl.load(
        src_scale_ptr,
        mask=valid_tokens,
        other=0.0,
    )
    tl.store(
        k_scale_ptr + tid * num_quant_blocks + quant_block_id,
        scale_val,
        mask=valid_tokens,
    )
    val = tl.load(src_cache_ptr + offsets[None, :], mask=mask)
    tl.store(dst_k_ptr + offsets[None, :], val, mask=mask)


def _generic_cp_gather(
    k_cache,
    k_fp8,
    k_fp8_scale,
    block_table,
    cu_seqlen,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    quant_block_size = head_dim * 4 // k_fp8_scale.size(1)
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size

    k_cache_flat = k_cache.view(num_blocks, -1)
    k_cache_value = k_cache_flat[:, : block_size * head_dim]
    k_cache_scale = k_cache_flat[:, block_size * head_dim :].view(torch.float32)
    k_fp8 = k_fp8.view(torch.uint8)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    batch_size = block_table.shape[0]
    if num_tokens < 32:
        token_block = 1
    elif num_tokens < 64:
        token_block = 2
    elif num_tokens < 128:
        token_block = 4
    elif num_tokens < 256:
        token_block = 8
    elif num_tokens < 512:
        token_block = 16
    else:
        token_block = 32
    if batch_size <= 16:
        batch_scan_size = triton.next_power_of_2(batch_size)
    else:
        batch_scan_size = 1
        token_block = max(token_block, 2)
    search_steps = (batch_size + 1).bit_length()

    grid = (triton.cdiv(num_tokens, token_block), num_quant_blocks)
    _generic_cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        k_fp8.stride(0),
        num_quant_blocks,
        batch_size,
        head_dim,
        quant_block_size,
        token_block,
        batch_scan_size,
        search_steps,
    )


# Fallback when the device property query is unavailable (e.g. CPU unit tests
# under the Triton interpreter). Ascend 910B exposes 40+ vector cores; a grid
# slightly below the real count only costs a little tail imbalance.
_DEFAULT_VECTOR_CORES = 40
_MAX_TILE_WORDS = 2048

_npu_vector_cores = None


def _get_vector_core_count():
    """Return the device's vector-core count, cached."""
    global _npu_vector_cores
    if _npu_vector_cores is not None:
        return _npu_vector_cores
    try:
        import torch_npu
        import triton.runtime.driver as driver

        props = driver.active.utils.get_device_properties(
            torch_npu.npu.current_device()
        )
        _npu_vector_cores = int(props["num_vectorcore"])
    except Exception:  # pragma: no cover - depends on the runtime
        _npu_vector_cores = _DEFAULT_VECTOR_CORES
    if _npu_vector_cores < 1:
        _npu_vector_cores = _DEFAULT_VECTOR_CORES
    return _npu_vector_cores


@triton.jit(
    do_not_specialize=[
        "num_tasks",
        "max_blocks",
        "block_table_stride",
        "cache_block_words",
    ]
)
def _cp_gather_indexer_k_quant_cache_kernel(
    cache_ptr,  # int32 view of the paged cache, flat per block
    dst_k_ptr,  # int32 view of [num_tokens, head_dim]
    dst_scale_ptr,  # int32 view of [num_tokens, num_quant_blocks * 4]
    block_table_ptr,
    cu_seqlen_ptr,
    num_tasks,
    max_blocks,
    block_table_stride,
    cache_block_words,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # tokens per cache block
    VAL_WORDS: tl.constexpr,  # head_dim // 4
    NQB: tl.constexpr,  # scale words per token
    SCALE_BASE_WORDS: tl.constexpr,  # BLOCK_SIZE * VAL_WORDS
    VAL_TILE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    val_offs = tl.arange(0, VAL_TILE)
    scale_offs = tl.arange(0, SCALE_TILE)

    for task in range(pid, num_tasks, NUM_PROGRAMS):
        batch_id = task // max_blocks
        block_pos = task - batch_id * max_blocks

        seq_start = tl.load(cu_seqlen_ptr + batch_id)
        seq_end = tl.load(cu_seqlen_ptr + batch_id + 1)
        first_token = seq_start + block_pos * BLOCK_SIZE

        # Blocks past the end of this sequence (and empty sequences) drop out
        # here, so the block table is never read outside its valid entries.
        if first_token < seq_end:
            num_valid = tl.minimum(BLOCK_SIZE, seq_end - first_token)
            block_id = tl.load(
                block_table_ptr + batch_id * block_table_stride + block_pos
            )
            # Scalar int64 bases; the offset vectors stay int32 so the address
            # tiles do not widen.
            src = cache_ptr + block_id.to(tl.int64) * cache_block_words
            dst_val = dst_k_ptr + first_token.to(tl.int64) * VAL_WORDS
            dst_scale = dst_scale_ptr + first_token.to(tl.int64) * NQB

            valid_val_words = num_valid * VAL_WORDS
            for base in tl.static_range(0, SCALE_BASE_WORDS, VAL_TILE):
                offs = base + val_offs
                mask = offs < valid_val_words
                tl.store(dst_val + offs, tl.load(src + offs, mask=mask), mask=mask)

            scale_mask = scale_offs < num_valid * NQB
            tl.store(
                dst_scale + scale_offs,
                tl.load(src + SCALE_BASE_WORDS + scale_offs, mask=scale_mask),
                mask=scale_mask,
            )


def _as_int32(tensor):
    """Reinterpret a contiguous byte-addressable tensor as int32 words."""
    return tensor.view(torch.int32)


def cp_gather_indexer_k_quant_cache(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
):
    args = (k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen)
    if HAS_TENSORVIEW and _cp_tv_eligible(*args):
        return tensorview_call(
            "CP_GATHER_INDEXER_K_QUANT_CACHE",
            _cp_gather_tensorview,
            _cp_gather_generic,
            *args,
        )
    return _cp_gather_generic(*args)


def _cp_gather_generic(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
):
    logger.debug("GEMS_ASCEND CP_GATHER_INDEXER_K_QUANT_CACHE")
    num_tokens = k_fp8.size(0)
    head_dim = k_fp8.size(-1)
    num_blocks = k_cache.shape[0]
    block_size = k_cache.shape[1]
    num_scale_bytes = k_fp8_scale.size(1)
    quant_block_size = head_dim * 4 // num_scale_bytes
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")

    if num_tokens == 0 or num_blocks == 0:
        return

    # Each block's payload is flat: value bytes then scale bytes.
    block_bytes = k_cache.stride(0)
    word_aligned = (
        block_bytes % 4 == 0
        and head_dim % 4 == 0
        and num_scale_bytes % 4 == 0
        and (block_size * head_dim) % 4 == 0
    )
    contiguous = (
        k_cache.is_contiguous()
        and k_fp8.is_contiguous()
        and k_fp8_scale.is_contiguous()
        and k_fp8.stride(0) == head_dim
        and k_fp8_scale.stride(0) == num_scale_bytes
    )
    if not (word_aligned and contiguous):
        # Still a Triton path, never torch compute.
        return _generic_cp_gather(k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen)

    val_words = head_dim // 4
    num_quant_blocks = num_scale_bytes // 4
    scale_base_words = block_size * val_words

    cache_words = _as_int32(k_cache.view(num_blocks, block_bytes))
    dst_k_words = _as_int32(k_fp8.view(num_tokens, head_dim))
    dst_scale_words = _as_int32(k_fp8_scale)

    batch_size, max_blocks = block_table.shape
    num_tasks = batch_size * max_blocks
    if num_tasks == 0:
        return

    val_tile = min(triton.next_power_of_2(scale_base_words), _MAX_TILE_WORDS)
    scale_tile = triton.next_power_of_2(block_size * num_quant_blocks)
    num_programs = _get_vector_core_count()

    _cp_gather_indexer_k_quant_cache_kernel[(num_programs,)](
        cache_words,
        dst_k_words,
        dst_scale_words,
        block_table,
        cu_seqlen,
        num_tasks,
        max_blocks,
        block_table.stride(0),
        block_bytes // 4,
        NUM_PROGRAMS=num_programs,
        BLOCK_SIZE=block_size,
        VAL_WORDS=val_words,
        NQB=num_quant_blocks,
        SCALE_BASE_WORDS=scale_base_words,
        VAL_TILE=val_tile,
        SCALE_TILE=scale_tile,
        num_warps=8,
    )


if HAS_TENSORVIEW:

    @triton.jit
    def _cp_gather_tv_kernel(
        kv_cache,
        kv_cache_scale,
        k_fp8,
        k_scale,
        block_table,
        cu_seqlen,
        num_tokens,
        cache_tiles,
        block_size,
        block_table_stride,
        kv_cache_scale_stride,
        num_quant_blocks,
        batch_size,
        HEAD_DIM: tl.constexpr,
        ROW_TILES: tl.constexpr,
        QUANT_SLOTS: tl.constexpr,
        BATCH_SCAN_SIZE: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        SMALL_BATCH: tl.constexpr,
        TOKEN_BLOCK: tl.constexpr,
    ):
        token_block = tl.program_id(0)
        tid = token_block * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        in_range = tid < num_tokens
        if SMALL_BATCH:
            batch_offsets = tl.arange(0, BATCH_SCAN_SIZE)
            batch_mask = batch_offsets < batch_size
            seq_starts = tl.load(cu_seqlen + batch_offsets, mask=batch_mask, other=0)
            seq_ends = tl.load(cu_seqlen + batch_offsets + 1, mask=batch_mask, other=0)
            in_batch = (
                (tid[:, None] >= seq_starts[None, :])
                & (tid[:, None] < seq_ends[None, :])
                & batch_mask[None, :]
            )
            batch_id = tl.max(tl.where(in_batch, batch_offsets[None, :], -1), axis=1)
        else:
            left = tl.zeros((TOKEN_BLOCK,), dtype=tl.int32)
            right = tl.full((TOKEN_BLOCK,), batch_size + 1, dtype=tl.int32)
            for _ in tl.static_range(0, SEARCH_STEPS):
                mid = (left + right) // 2
                seq_start = tl.load(
                    cu_seqlen + mid, mask=mid <= batch_size, other=2147483647
                )
                before = seq_start <= tid
                left = tl.where(before, mid + 1, left)
                right = tl.where(before, right, mid)
            batch_id = left - 1

        valid_batch = (batch_id >= 0) & (batch_id < batch_size)
        safe_batch_id = tl.minimum(tl.maximum(batch_id, 0), batch_size - 1)
        batch_start = tl.load(cu_seqlen + safe_batch_id, mask=valid_batch, other=0)
        batch_end = tl.load(cu_seqlen + safe_batch_id + 1, mask=valid_batch, other=0)
        valid = in_range & valid_batch & (tid >= batch_start) & (tid < batch_end)
        batch_offset = tid - batch_start
        block_id = tl.load(
            block_table
            + safe_batch_id * block_table_stride
            + batch_offset // block_size,
            mask=valid,
            other=0,
        )
        block_offset = batch_offset % block_size
        # The full cache allocation includes inline scales; tile row indices
        # must step over the scale bytes between adjacent physical blocks.
        rows = tl.where(valid, block_id * ROW_TILES + block_offset, -1).to(tl.int32)
        cache_view = tle_exp.make_gather_scatter_view(
            kv_cache,
            shape=[cache_tiles, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            tile=[TOKEN_BLOCK, HEAD_DIM],
            sparse_dim=[0],
            bounds_check=True,
        )
        gathered = tile_load(
            cache_view, index=(rows, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )
        out_view = tle_exp.make_partition_view(
            k_fp8,
            shape=[tl.minimum(tl.load(cu_seqlen + batch_size), num_tokens), HEAD_DIM],
            strides=[HEAD_DIM, 1],
            tile=[TOKEN_BLOCK, HEAD_DIM],
        )
        tile_store(
            out_view,
            value=gathered,
            index=(token_block, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

        quant = tl.arange(0, QUANT_SLOTS)
        scale_mask = valid[:, None] & (quant < num_quant_blocks)[None, :]
        scale_val = tl.load(
            kv_cache_scale
            + block_id[:, None] * kv_cache_scale_stride
            + block_offset[:, None] * num_quant_blocks
            + quant[None, :],
            mask=scale_mask,
            other=0.0,
        )
        tl.store(
            k_scale + tid[:, None] * num_quant_blocks + quant[None, :],
            scale_val,
            mask=scale_mask,
        )

    def _cp_tv_eligible(k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen):
        if k_cache.ndim < 2 or k_fp8.ndim != 2 or k_fp8_scale.ndim != 2:
            return False
        if block_table.ndim != 2 or cu_seqlen.ndim != 1:
            return False
        if k_cache.numel() == 0 or k_fp8.numel() == 0 or block_table.numel() == 0:
            return False
        if not (
            k_cache.is_contiguous()
            and k_fp8.is_contiguous()
            and k_fp8_scale.is_contiguous()
        ):
            return False
        if any(
            t.device != k_cache.device
            for t in (k_fp8, k_fp8_scale, block_table, cu_seqlen)
        ):
            return False
        if k_cache.dtype != torch.uint8 or k_fp8.element_size() != 1:
            return False
        if k_fp8_scale.dtype != torch.float32:
            return False
        head_dim = k_fp8.shape[-1]
        scale_bytes = k_fp8_scale.shape[1]
        if head_dim == 0 or head_dim != triton.next_power_of_2(head_dim):
            return False
        if scale_bytes == 0 or scale_bytes % 4 != 0:
            return False
        if k_fp8.stride(0) != head_dim or k_fp8.stride(1) != 1:
            return False
        quant_block_size = head_dim * 4 // scale_bytes
        if quant_block_size == 0 or head_dim % quant_block_size != 0:
            return False
        row_stride = k_cache.shape[1] * (head_dim + scale_bytes)
        if k_cache.stride(0) != row_stride or row_stride % head_dim != 0:
            return False
        if k_cache.numel() % row_stride != 0 or block_table.stride(1) != 1:
            return False
        if triton.next_power_of_2(scale_bytes // 4) * 4 > 8192:
            return False
        return block_table.dtype in (torch.int32, torch.int64) and cu_seqlen.dtype in (
            torch.int32,
            torch.int64,
        )

    def _cp_gather_tensorview(k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen):
        num_tokens = k_fp8.shape[0]
        block_size = k_cache.shape[1]
        head_dim = k_fp8.shape[-1]
        num_blocks = k_cache.shape[0]
        num_quant_blocks = k_fp8_scale.shape[1] // 4
        k_cache_flat = k_cache.view(num_blocks, -1)
        k_cache_scale = k_cache_flat[:, block_size * head_dim :].view(torch.float32)
        k_fp8_u8 = k_fp8.view(torch.uint8)
        k_fp8_scale_f32 = k_fp8_scale.view(torch.float32)
        batch_size = block_table.shape[0]
        small_batch = batch_size <= 16
        token_block = 32
        with torch_device_fn.device(k_cache.device):
            _cp_gather_tv_kernel[(triton.cdiv(num_tokens, token_block),)](
                k_cache_flat,
                k_cache_scale,
                k_fp8_u8,
                k_fp8_scale_f32,
                block_table,
                cu_seqlen,
                num_tokens,
                num_blocks * (k_cache_flat.stride(0) // head_dim),
                block_size,
                block_table.stride(0),
                k_cache_scale.stride(0),
                num_quant_blocks,
                batch_size,
                HEAD_DIM=head_dim,
                ROW_TILES=k_cache_flat.stride(0) // head_dim,
                QUANT_SLOTS=triton.next_power_of_2(num_quant_blocks),
                BATCH_SCAN_SIZE=(
                    triton.next_power_of_2(batch_size) if small_batch else 1
                ),
                SEARCH_STEPS=(batch_size + 1).bit_length(),
                SMALL_BATCH=small_batch,
                TOKEN_BLOCK=token_block,
            )
        return None

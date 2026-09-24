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

"""Ascend NPU gather of the FP8 indexer K cache.

The generic kernel parallelises over tokens: every program resolves which
sequence its token block belongs to (a ``tl.max`` reduction over a
``[TOKEN_BLOCK, BATCH_SCAN]`` tile, or a binary search), then issues a
``[TOKEN_BLOCK, QUANT_BLOCK_SIZE]`` gather whose addresses are a wide int64
tile.  On the AI vector cores that costs ~40x the torch baseline: the token
grid runs to thousands of programs at ~70ns scheduling each, the 2D int64
address tiles do not lower to wide vector moves, and ``block_size`` is a
runtime value so ``//`` and ``%`` stay as divisions.

This implementation parallelises over *cache blocks* instead.  Within one
block the cache holds ``block_size * head_dim`` value bytes followed by
``block_size * num_quant_blocks * 4`` scale bytes, all contiguous, and the
destination rows for those same tokens are contiguous as well.  One task is
therefore a pair of flat contiguous copies (8 KB at the shapes vLLM uses),
addressed as a scalar int64 base plus a 1D int32 offset vector.  The grid is
pinned to the vector-core count and each program walks its tasks with a
strided loop, so program scheduling cost no longer scales with token count,
and ``block_size`` / ``head_dim`` enter as constexpr so the offset math folds
into shifts.

The sequence lookup disappears entirely: the task index carries the batch id,
so no per-token search is needed.  Tokens past ``cu_seqlen[-1]`` are never
written, matching the C++ kernel, which sizes the gather from the device-side
``cu_seqlen`` rather than from ``dst_k``'s allocated row count.

Layouts that cannot be reinterpreted as int32 words (non-contiguous inputs, or
byte counts not divisible by 4) fall back to the generic Triton kernel; there
is no torch compute path.

Measured on 910B4 (64 GB), ``benchmark/test_cp_gather_indexer_k_quant_cache.py
--level comprehensive``, latency in us against the torch baseline::

    tokens x bytes  batch   torch   generic   this   SpeedUp
      1024 x 132        4    15.9      93.9    4.6      3.50
      4096 x 132        8    33.1     340.8    7.9      4.18
     16384 x 528       16   112.6    4536.1   13.1      8.61
     32768 x 528       32   209.6    4525.4   19.3     10.85
      8192 x 132        1    66.0     597.9    6.3     10.51
     32768 x 132        8   233.2    2597.2    9.9     23.54
     65536 x 132       64   429.7     558.3   13.4     32.06

The two smallest shapes sit on a ~4.5us floor set by launch and task-loop
overhead rather than by bandwidth. Note that the benchmark reuses one set of
input tensors per shape, so the larger shapes' apparent GB/s benefits from
cache residency and should not be read as sustained HBM throughput.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _generic_cp_gather(*args):
    """Generic Triton kernel, imported lazily.

    The vendor ops package is imported from ``flaggems_vllm/__init__.py``
    while that module is still executing, so importing the generic op at
    module scope would add a cycle for no benefit: this path only runs for
    layouts the word-based kernel cannot address.
    """
    from flaggems_vllm.ops.cp_gather_indexer_k_quant_cache import (
        cp_gather_indexer_k_quant_cache as generic,
    )

    return generic(*args)


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

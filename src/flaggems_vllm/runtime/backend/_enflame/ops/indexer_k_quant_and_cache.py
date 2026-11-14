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

# Adapted from vLLM v0.20.2:
# csrc/cache_kernels.cu::indexer_k_quant_and_cache_kernel
#
# GCU (gcu400) adaptation notes:
# - The generic one-token-per-program grid pays the full serial dependent
#   memory latency chain (slot load -> row load -> scattered store) per
#   token, capping effective bandwidth at ~25 GB/s (~117x off roofline).
#   gcu400 memory-path rules (see knowledge base): plain `base + tl.arange`
#   flat ops and 2D affine tiles are the only fast shapes; per-lane
#   computed gather offsets (e.g. 3D per-row bases) are ~18x slower, and
#   per-token scalar walks still serialize at ~1.15us/token.
# - The fast kernel below batches U consecutive tokens per program: one
#   flat contiguous load of U*head_dim elements, one int32 slot-word
#   vector load, register reshape for the per-quant-block amax, then one
#   2D per-row-base scatter store for the fp8 row plus one for the scale
#   row. Reshaped tensors are capped at 4096 elements (compiler_issues
#   P16 safe bound); num_warps=1 (nw>1 measured catastrophic on the
#   scatter store). The fast path requires head_dim == HD_TILE (which
#   implies QB_PAD == num_quant_blocks, so no lane masks are needed);
#   only the last grid program takes the masked tail arm, and negative
#   slots are dropped by a row mask with where-clamped safe bases.
# - int64 appears only as scalar base folds in the ttok/legacy fallbacks
#   (caches beyond 2**31 elements route there); the fast kernel keeps all
#   lane arithmetic int32 with a host-side numel guard. The zero-copy
#   int32 slot-word view with env-aware word stride is preserved
#   (torch_gcu keeps ATen int64 values as compact int32 words unless
#   TORCH_GCU_ENABLE_INT64_AND_UINT64=1).
# - Non-power-of-two quant_block_size (never produced by vLLM indexer
#   configs) falls back to the original two-axis kernel; every path is
#   Triton, no torch compute.

import os

import torch
import triton
import triton.language as tl


def _get_fp8_dtype() -> torch.dtype:
    try:
        from vllm.platforms import current_platform

        return current_platform.fp8_dtype()
    except ImportError:
        pass

    if getattr(torch.version, "hip", None) is not None and hasattr(
        torch, "float8_e4m3fnuz"
    ):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    raise RuntimeError("float8_e4m3fn is required for indexer_k_quant_and_cache")


def _is_fp8_fnuz(dtype: torch.dtype) -> bool:
    return hasattr(torch, "float8_e4m3fnuz") and dtype == torch.float8_e4m3fnuz


def _slot_word_stride() -> int:
    # torch_gcu without TORCH_GCU_ENABLE_INT64_AND_UINT64 keeps ATen int64
    # values as compact int32 words (element j lives at word j); with the gate
    # on, values sit in the low word of each 8-byte slot (word 2*j). Either
    # way, a Triton int64 load at the 8-byte dtype stride reads
    # neighbor-joined garbage, so slots are always read through a zero-copy
    # int32 view with the matching word stride.
    return 2 if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1" else 1


def _slot_words(slot_mapping: torch.Tensor):
    if slot_mapping.dtype == torch.int32:
        return slot_mapping, 1
    if slot_mapping.dtype != torch.int64:
        raise NotImplementedError(
            "indexer_k_quant_and_cache supports int32/int64 slot_mapping, got "
            f"{slot_mapping.dtype}"
        )
    return slot_mapping.contiguous().view(torch.int32), _slot_word_stride()


# Enflame GCU hardware limit for grid.x; extra pages fold into grid.y
# (see scaled_int8_quant / cp_gather_indexer_k_quant_cache for the same
# adaptation).
GCU_GRID_X_MAX = 65535

# Compiler-side safe upper bound for flat+reshape tensors (P16); the fast
# kernel keeps U * HD_TILE at or below it.
_FLAT_RESHAPE_ELEMS_MAX = 4096


@triton.jit
def _indexer_k_quant_cache_tile_kernel(
    k_ptr,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    slot_words_ptr,
    slot_word_stride,
    kv_cache_scale_stride,
    kv_cache_value_stride,
    num_tokens,
    num_programs,
    GRID_X: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QBS: tl.constexpr,
    NQB: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    U: tl.constexpr,
):
    # Fast path: head_dim == U*HD_TILE layout is exact, so no lane masks.
    pid = tl.program_id(0) + tl.program_id(1) * GRID_X
    t0 = pid * U
    tv = t0 + tl.arange(0, U)
    offs = tl.arange(0, U * HEAD_DIM)

    if pid == num_programs - 1:
        # Ragged tail: mask token rows past num_tokens (only this program).
        tmask = tv < num_tokens
        slot = tl.load(slot_words_ptr + tv * slot_word_stride, mask=tmask, other=-1)
        lmask = tl.reshape(
            tl.broadcast_to(tmask[:, None], (U, HEAD_DIM)), (U * HEAD_DIM,)
        )
        v = tl.load(k_ptr + t0 * HEAD_DIM + offs, mask=lmask, other=0.0)
    else:
        slot = tl.load(slot_words_ptr + tv * slot_word_stride)
        v = tl.load(k_ptr + t0 * HEAD_DIM + offs)

    valid = slot >= 0
    v3 = tl.reshape(v, (U, NQB, QBS))
    amax = tl.max(tl.abs(v3).to(tl.float32), axis=2)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    fp8 = tl.reshape(
        (v3.to(tl.float32) / scale[:, :, None]).to(kv_cache_ptr.type.element_ty),
        (U, HEAD_DIM),
    )
    # where-clamped safe bases (slot 0) for invalid rows; the row mask on
    # the stores drops them. Lane arithmetic stays int32 (host numel guard).
    safe_slot = tl.where(valid, slot, 0)
    block_id = safe_slot // BLOCK_SIZE
    block_off = safe_slot - block_id * BLOCK_SIZE
    vh = tl.arange(0, HEAD_DIM)
    dst = (
        block_id[:, None] * kv_cache_value_stride
        + block_off[:, None] * HEAD_DIM
        + vh[None, :]
    )
    tl.store(kv_cache_ptr + dst, fp8, mask=valid[:, None])
    soffs = tl.arange(0, NQB)
    sdst = (
        block_id[:, None] * kv_cache_scale_stride
        + block_off[:, None] * NQB
        + soffs[None, :]
    )
    tl.store(kv_cache_scale_ptr + sdst, scale, mask=valid[:, None])


@triton.jit
def _indexer_k_quant_cache_ttok_kernel(
    k_ptr,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    slot_words_ptr,
    slot_word_stride,
    kv_cache_scale_stride,
    kv_cache_value_stride,
    num_tokens,
    GRID_X: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QBS: tl.constexpr,
    NQB: tl.constexpr,
    QB_PAD: tl.constexpr,
    HD_TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    TB: tl.constexpr,
    VEXACT: tl.constexpr,
    SEXACT: tl.constexpr,
):
    # Fallback for non-exact head dims / oversized caches: per-token scalar
    # walk with flat ops and i64 scalar base folds.
    pid = tl.program_id(0) + tl.program_id(1) * GRID_X
    t_begin = pid * TB
    t_end = tl.minimum(t_begin + TB, num_tokens)
    voffs = tl.arange(0, HD_TILE)
    soffs = tl.arange(0, QB_PAD)

    t = t_begin
    while t < t_end:
        slot_id = tl.load(slot_words_ptr + t * slot_word_stride)
        if slot_id >= 0:
            block_id = slot_id // BLOCK_SIZE
            block_offset = slot_id - block_id * BLOCK_SIZE

            if VEXACT:
                v = tl.load(k_ptr + t * HEAD_DIM + voffs)
            else:
                v = tl.load(
                    k_ptr + t * HEAD_DIM + voffs,
                    mask=voffs < HEAD_DIM,
                    other=0.0,
                )
            a2 = tl.reshape(v, (QB_PAD, QBS))
            amax = tl.max(tl.abs(a2).to(tl.float32), axis=1)
            if IS_FNUZ:
                scale = tl.maximum(1e-4, amax) / 224.0
            else:
                scale = tl.maximum(1e-4, amax) / 448.0
            if USE_UE8M0:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))
            fp8 = tl.reshape(
                (a2.to(tl.float32) / scale[:, None]).to(kv_cache_ptr.type.element_ty),
                (HD_TILE,),
            )
            # Fold every scalar offset term into one i64 expression: i32
            # addptr chains truncate element offsets >= 2**31, and kv
            # caches beyond 2**31 elements are routine for the indexer.
            cache_base = (
                block_id.to(tl.int64) * kv_cache_value_stride
                + block_offset.to(tl.int64) * HEAD_DIM
            )
            if VEXACT:
                tl.store(kv_cache_ptr + cache_base + voffs, fp8)
            else:
                tl.store(
                    kv_cache_ptr + cache_base + voffs,
                    fp8,
                    mask=voffs < HEAD_DIM,
                )
            scale_base = (
                block_id.to(tl.int64) * kv_cache_scale_stride
                + block_offset.to(tl.int64) * NQB
            )
            if SEXACT:
                tl.store(kv_cache_scale_ptr + scale_base + soffs, scale)
            else:
                tl.store(
                    kv_cache_scale_ptr + scale_base + soffs,
                    scale,
                    mask=soffs < NQB,
                )
        t = t + 1


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["QUANT_BLOCK_SIZE", "NUM_TOKENS_PAD"],
)
@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    slot_words_ptr,
    slot_word_stride,
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_quant_blocks,
    num_tokens,
    grid_x,
    quant_grids,
    head_dim: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    NUM_TOKENS_PAD: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    token_page = pid1 // quant_grids
    quant_page = pid1 % quant_grids
    tid = token_page * grid_x + pid0
    if tid >= num_tokens:
        return
    quant_block_id = quant_page * 4
    quant_block_offsets = tl.arange(0, 4)
    head_offsets = tl.arange(0, QUANT_BLOCK_SIZE)
    offsets = (
        quant_block_id + quant_block_offsets[:, None]
    ) * QUANT_BLOCK_SIZE + head_offsets[None, :]
    mask = offsets < head_dim

    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_words_ptr + tid * slot_word_stride)
    if slot_id < 0:
        return

    block_id = slot_id // block_size
    block_offset = slot_id % block_size

    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    amax = tl.max(tl.abs(val).to(tl.float32), axis=1)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale[:, None]).to(kv_cache_ptr.type.element_ty)
    cache_base = (
        block_id.to(tl.int64) * kv_cache_value_stride
        + block_offset.to(tl.int64) * head_dim
    )
    dst_ptr = kv_cache_ptr + cache_base
    tl.store(dst_ptr + offsets, fp8_val, mask=mask)

    scale_base = (
        block_id.to(tl.int64) * kv_cache_scale_stride
        + block_offset.to(tl.int64) * num_quant_blocks
        + quant_block_id
    )
    dst_scale_ptr = kv_cache_scale_ptr + scale_base
    scale_mask = quant_block_id + quant_block_offsets < num_quant_blocks
    tl.store(dst_scale_ptr + quant_block_offsets, scale, mask=scale_mask)


def _launch_legacy(
    k,
    kv_cache_value,
    kv_cache_scale,
    slot_words,
    slot_word_stride,
    num_blocks,
    head_dim,
    num_tokens,
    block_size,
    quant_block_size,
    num_quant_blocks,
    fp8_dtype,
    scale_fmt,
):
    num_tokens_pad = triton.next_power_of_2(num_tokens)
    grid_x = min(num_tokens, GCU_GRID_X_MAX)
    quant_grids = triton.cdiv(num_quant_blocks, 4)
    grid = (grid_x, triton.cdiv(num_tokens, grid_x) * quant_grids)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_words,
        slot_word_stride,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_quant_blocks,
        num_tokens,
        grid_x,
        quant_grids,
        head_dim,
        quant_block_size,
        IS_FNUZ=_is_fp8_fnuz(fp8_dtype),
        USE_UE8M0=scale_fmt == "ue8m0",
        NUM_TOKENS_PAD=num_tokens_pad,
    )


def _launch_ttok(
    k,
    kv_cache_value,
    kv_cache_scale,
    slot_words,
    slot_word_stride,
    head_dim,
    num_tokens,
    block_size,
    quant_block_size,
    num_quant_blocks,
    qb_pad,
    hd_tile,
    fp8_dtype,
    scale_fmt,
):
    if num_tokens <= 2048:
        token_block = 64
    elif head_dim >= 256:
        token_block = 128
    else:
        token_block = 96
    num_programs = triton.cdiv(num_tokens, token_block)
    grid_x = min(num_programs, GCU_GRID_X_MAX)
    grid_y = triton.cdiv(num_programs, grid_x)
    _indexer_k_quant_cache_ttok_kernel[(grid_x, grid_y)](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_words,
        slot_word_stride,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        num_tokens,
        grid_x,
        head_dim,
        quant_block_size,
        num_quant_blocks,
        qb_pad,
        hd_tile,
        block_size,
        IS_FNUZ=_is_fp8_fnuz(fp8_dtype),
        USE_UE8M0=scale_fmt == "ue8m0",
        TB=token_block,
        VEXACT=hd_tile == head_dim,
        SEXACT=qb_pad == num_quant_blocks,
        num_warps=1,
    )


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size
    if num_tokens == 0 or num_quant_blocks == 0:
        return

    kv_cache_flat = kv_cache.view(num_blocks, -1)
    fp8_dtype = _get_fp8_dtype()
    kv_cache_value = kv_cache_flat[:, : block_size * head_dim].view(fp8_dtype)
    kv_cache_scale = kv_cache_flat[:, block_size * head_dim :].view(torch.float32)
    slot_words, slot_word_stride = _slot_words(slot_mapping)

    quant_block_pow2 = (quant_block_size & (quant_block_size - 1)) == 0
    qb_pad = triton.next_power_of_2(num_quant_blocks)
    hd_tile = qb_pad * quant_block_size
    # Tiny batches keep the original one-token-per-program parallelism:
    # below ~64 tokens the launch floor plus a single serial tile program
    # loses to token-level parallelism (measured crossover on gcu400).
    if num_tokens < 64 or not quant_block_pow2:
        _launch_legacy(
            k,
            kv_cache_value,
            kv_cache_scale,
            slot_words,
            slot_word_stride,
            num_blocks,
            head_dim,
            num_tokens,
            block_size,
            quant_block_size,
            num_quant_blocks,
            fp8_dtype,
            scale_fmt,
        )
        return

    # Fast tile path: exact padded layout (implies no lane masks) and all
    # int32 lane arithmetic provably in range.
    if (
        hd_tile == head_dim
        and kv_cache_flat.numel() < 2**31
        and num_tokens * head_dim < 2**31
    ):
        u = max(1, min(_FLAT_RESHAPE_ELEMS_MAX // hd_tile, 32))
        num_programs = triton.cdiv(num_tokens, u)
        grid_x = min(num_programs, GCU_GRID_X_MAX)
        grid_y = triton.cdiv(num_programs, grid_x)
        _indexer_k_quant_cache_tile_kernel[(grid_x, grid_y)](
            k,
            kv_cache_value,
            kv_cache_scale,
            slot_words,
            slot_word_stride,
            kv_cache_scale.stride(0),
            kv_cache_value.stride(0),
            num_tokens,
            num_programs,
            grid_x,
            head_dim,
            quant_block_size,
            num_quant_blocks,
            block_size,
            IS_FNUZ=_is_fp8_fnuz(fp8_dtype),
            USE_UE8M0=scale_fmt == "ue8m0",
            U=u,
            num_warps=1,
        )
        return

    _launch_ttok(
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_words,
        slot_word_stride,
        head_dim,
        num_tokens,
        block_size,
        quant_block_size,
        num_quant_blocks,
        qb_pad,
        hd_tile,
        fp8_dtype,
        scale_fmt,
    )

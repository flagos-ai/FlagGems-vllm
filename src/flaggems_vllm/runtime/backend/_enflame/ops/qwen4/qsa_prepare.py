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

"""Fused QSA prepare kernel for Qwen4Exp (Enflame GCU adaptation).

Semantics are identical to the generic implementation (single launch:
indexer Q norm/RoPE + K compression + circular raw-state commit + main
attention). The GCU port restructures the memory access patterns only:

- Row-scalar + 1D-arange addressing for the K-compression and ring-commit
  segments (int64 scalar row bases are the proven-safe addressing form on
  this backend; int64 *vector* addressing is avoided).
- 2D row-batch tiles for the three streaming segments (indexer Q, main Q,
  main K/V). One work item covers a [TILE_T, cols] tile of one head (main)
  or of all indexer Q heads via an unrolled head loop. Row bases are i32
  and in-row columns are contiguous affine, the load/store shapes that
  vectorize on this backend; the NeoX layout chains of the seed are not
  used - rotated sub-segments are re-loaded from memory instead (the
  "flatten + re-load" recipe validated on this op family). The host
  dispatches batches below 64 tokens to the original 1-D per-item arms
  (STREAMING_TILED=False): a 2-D tile has a flat per-item cost, so token
  counts that cannot fill the fixed grid with tiles are cheaper unbatched.
  The tile width climbs to 64 rows (bf16) / 32 rows (any fp8 destination;
  wider fp8 scatter tiles cliff, measured 3.4x at 64) and halves until the
  eight main items per token tile fill the 48-program grid.
- The main K/V scatter store uses per-row slot bases with i32 lane
  arithmetic; a host guard falls back to per-(token, head) 1D items when
  the paged cache exceeds the i32 safe domain (2**31 elements).
- torch_gcu stores int64 tensors either as a 4-byte payload
  (TORCH_GCU_ENABLE_INT64_AND_UINT64 unset: "GCU not support Long use Int
  replace") or as true 8-byte slots (the env set). The host hands the
  kernel int32 word views in the 4-byte mode and the int64 tensors
  themselves in the 8-byte mode; the kernel normalizes loaded index
  scalars to i32 (the fast arithmetic domain on this backend) and widens
  only the scalar row bases to int64, so both layouts read the correct
  values. Values stay in the i32 safe domain (positions/slots < 2**31) by
  production contract.
- The K-compression source select between the current chunk and the
  circular state ring uses two mutually exclusive masked loads summed
  per segment; a `tl.where` on pointer tiles lowers to an illegal
  arith.select on this backend.
- FP8 destinations are written through an fp32 intermediate; a direct
  bf16->fp8 store cast trips a linker R_DTU_ADDR16_LO_ICALL relocation.
- Persistent grid-stride launch (Fixed Grid 48): the upstream grid of
  one program per work item exceeds the 65535 grid.x hardware limit for
  large batches (a 16k-token prefill needs ~200k work items).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Libra L600 / GCU400: 48 concurrent CTAs is the tuned persistent-grid width.
GCU_NUM_GRID = 48

# Paged-cache element ceiling for the i32-lane scatter path (main K/V).
_SCATTER_I32_MAX_NUMEL = 2**31


def _int64_env_on() -> bool:
    return os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "") in ("1", "true")


def _device_is_gcu(device) -> bool:
    if device is None:
        return False
    return str(device).split(":")[0] in ("cuda", "gcu")


def _install_randperm_guard() -> None:
    """Route int64 ``torch.randperm`` on GCU through CPU while the env is on.

    torch_gcu's topsatenRandperm aborts the whole process when
    ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` is set (true 8-byte slots, see P21);
    the crash fires in callers outside this operator (test input construction,
    data loaders). torch_gcu wraps ``torch.randperm`` at the Python level, so
    the guard wraps the current callable the same way: an env-on int64
    permutation for a GCU device is computed on the CPU and transferred back
    (value-equivalent, construction-time only); every other call - other
    dtypes, explicit generators, ``out=`` and every non-GCU device - keeps the
    native path unchanged.
    """
    current = torch.randperm
    if getattr(current, "__module__", None) == __name__:
        return

    def _randperm_guarded(n, *args, **kwargs):
        dtype = kwargs.get("dtype")
        device = kwargs.get("device")
        if (
            _int64_env_on()
            and kwargs.get("generator") is None
            and kwargs.get("out") is None
            and _device_is_gcu(device)
            and (dtype is None or dtype == torch.int64)
        ):
            cpu_kwargs = dict(kwargs)
            cpu_kwargs["device"] = "cpu"
            return current(n, *args, **cpu_kwargs).to(device)
        return current(n, *args, **kwargs)

    torch.randperm = _randperm_guarded


_install_randperm_guard()


@triton.jit
def _cos_sin(
    pos_t,
    pos_h,
    pos_w,
    cos_sin_ptr,
    cos_sin_stride,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    QUARTER: tl.constexpr,
):
    """Row-gather cos/sin for one token from the position-major table.

    ``pos_*`` are scalar positions; the three MRoPE axis masks are
    mutually exclusive, so the masked-load accumulation (upstream form,
    ``other=0``) is equivalent to a per-pair row selection. Segment
    bounds follow the upstream ``_norm_rope`` convention
    (``pairs <= 3 * MROPE_*``); the main-attention path keeps its own
    stricter ``rot < 3 * MROPE_*`` bounds via ``_cos_sin_tile``.
    """
    pairs = tl.arange(0, QUARTER)
    if IS_MROPE:
        h_mask = ((pairs % 3) == 1) & (pairs <= 3 * MROPE_H)
        w_mask = ((pairs % 3) == 2) & (pairs <= 3 * MROPE_W)
        t_mask = ~(h_mask | w_mask)
        cos = tl.load(
            cos_sin_ptr + pos_t * cos_sin_stride + pairs, mask=t_mask, other=0
        )
        sin = tl.load(
            cos_sin_ptr + pos_t * cos_sin_stride + QUARTER + pairs,
            mask=t_mask,
            other=0,
        )
        cos += tl.load(
            cos_sin_ptr + pos_h * cos_sin_stride + pairs, mask=h_mask, other=0
        )
        sin += tl.load(
            cos_sin_ptr + pos_h * cos_sin_stride + QUARTER + pairs,
            mask=h_mask,
            other=0,
        )
        cos += tl.load(
            cos_sin_ptr + pos_w * cos_sin_stride + pairs, mask=w_mask, other=0
        )
        sin += tl.load(
            cos_sin_ptr + pos_w * cos_sin_stride + QUARTER + pairs,
            mask=w_mask,
            other=0,
        )
    else:
        cos = tl.load(cos_sin_ptr + pos_t * cos_sin_stride + pairs)
        sin = tl.load(cos_sin_ptr + pos_t * cos_sin_stride + QUARTER + pairs)
    return cos, sin


@triton.jit
def _cos_sin_tile(
    pos_t,
    pos_h,
    pos_w,
    row_valid,
    cos_sin_ptr,
    cos_sin_stride,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    QUARTER: tl.constexpr,
    STRICT_BOUNDS: tl.constexpr,
):
    """Per-token cos/sin tile, [TILE_T, QUARTER], for the batched segments.

    ``pos_*`` are [TILE_T] i32 row positions. The three MRoPE axis masks
    are mutually exclusive column masks, so the masked-load accumulation
    reproduces the per-pair axis selection of the scalar form exactly
    (``other=0`` adds are exact in fp32). ``STRICT_BOUNDS`` selects the
    main-attention ``rot < 3 * MROPE_*`` convention.
    """
    pairs = tl.arange(0, QUARTER)
    if IS_MROPE:
        if STRICT_BOUNDS:
            h_mask = ((pairs % 3) == 1) & (pairs < 3 * MROPE_H)
            w_mask = ((pairs % 3) == 2) & (pairs < 3 * MROPE_W)
        else:
            h_mask = ((pairs % 3) == 1) & (pairs <= 3 * MROPE_H)
            w_mask = ((pairs % 3) == 2) & (pairs <= 3 * MROPE_W)
        t_mask = ~(h_mask | w_mask)
        base_t = cos_sin_ptr + pos_t[:, None] * cos_sin_stride
        base_h = cos_sin_ptr + pos_h[:, None] * cos_sin_stride
        base_w = cos_sin_ptr + pos_w[:, None] * cos_sin_stride
        cos = tl.load(base_t + pairs[None, :], mask=t_mask[None, :], other=0)
        sin = tl.load(base_t + QUARTER + pairs[None, :], mask=t_mask[None, :], other=0)
        cos += tl.load(base_h + pairs[None, :], mask=h_mask[None, :], other=0)
        sin += tl.load(base_h + QUARTER + pairs[None, :], mask=h_mask[None, :], other=0)
        cos += tl.load(base_w + pairs[None, :], mask=w_mask[None, :], other=0)
        sin += tl.load(base_w + QUARTER + pairs[None, :], mask=w_mask[None, :], other=0)
    else:
        m = row_valid[:, None]
        cos = tl.load(
            cos_sin_ptr + pos_t[:, None] * cos_sin_stride + pairs[None, :],
            mask=m,
            other=0,
        )
        sin = tl.load(
            cos_sin_ptr + pos_t[:, None] * cos_sin_stride + QUARTER + pairs[None, :],
            mask=m,
            other=0,
        )
    return cos, sin


@triton.jit
def _to_dst_dtype(x, dst, scale, DST_FP8: tl.constexpr):
    """Round to BF16 like the unfused path, then scale for an FP8 destination.

    ``DST_FP8`` is computed on the host from the destination dtype; the
    final cast leaves through fp32 so the bf16->fp8 store never uses the
    intrinsic that trips the linker relocation.
    """
    out_ty = dst.dtype.element_ty
    x = x.to(tl.bfloat16)
    if DST_FP8:
        x = x.to(tl.float32) / scale
    return x.to(out_ty)


@triton.jit
def _store_rotated(dst, y, o1, o2, scale, DST_FP8: tl.constexpr):
    """Store a normalized head whose first ``2 * len(o1)`` dims are rotated."""
    HALF: tl.constexpr = o1.shape[0]
    dims = tl.arange(0, y.shape[0])
    rot = tl.arange(0, HALF)
    tl.store(dst + dims, _to_dst_dtype(y, dst, scale, DST_FP8), mask=dims >= 2 * HALF)
    tl.store(dst + rot, _to_dst_dtype(o1, dst, scale, DST_FP8))
    tl.store(dst + HALF + rot, _to_dst_dtype(o2, dst, scale, DST_FP8))


@triton.jit
def _store_rotated_2d(row_base, m, y, o1, o2, scale, DST_FP8: tl.constexpr):
    """Store a tile of normalized heads, rows selected by ``m`` [TILE_T, 1].

    ``row_base`` is a [TILE_T, 1] pointer tile; the first ``2 * HALF``
    columns of each row are rotated, the rest pass through.
    """
    HALF: tl.constexpr = o1.shape[1]
    dims = tl.arange(0, y.shape[1])
    rot = tl.arange(0, HALF)
    tl.store(
        row_base + dims[None, :],
        _to_dst_dtype(y, row_base, scale, DST_FP8),
        mask=m & (dims[None, :] >= 2 * HALF),
    )
    tl.store(
        row_base + rot[None, :], _to_dst_dtype(o1, row_base, scale, DST_FP8), mask=m
    )
    tl.store(
        row_base + HALF + rot[None, :],
        _to_dst_dtype(o2, row_base, scale, DST_FP8),
        mask=m,
    )


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_state_blocks",
        "num_compressed_blocks",
        "num_k_work",
        "num_q_tiles",
        "num_main_tiles",
        "num_main_q_items",
        "num_main_kv_items",
        "total_work",
    ]
)
def _qsa_prepare_kernel(
    q_ptr,
    q_stride_token,
    k_ptr,
    k_stride_token,
    pos_ptr,
    pos_stride_axis,
    pos_stride_token,
    cos_sin_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    eps,
    q_out_ptr,
    q_out_stride_token,
    q_out_stride_head,
    state_cache_ptr,
    state_cache_stride_block,
    state_cache_stride_token,
    state_slots_ptr,
    state_table_ptr,
    state_table_stride_req,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    k_work_metadata_ptr,
    compressed_cache_ptr,
    compressed_cache_stride_block,
    compressed_cache_stride_token,
    num_tokens,
    num_state_blocks,
    num_compressed_blocks,
    num_k_work,
    num_q_tiles,
    num_main_tiles,
    num_main_q_items,
    num_main_kv_items,
    total_work,
    HQ: tl.constexpr,
    D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    COMP_PAGE_SIZE: tl.constexpr,
    IS_2D_POSITIONS: tl.constexpr,
    IS_K_MROPE: tl.constexpr,
    CACHE_HAS_ROPE_POS: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    main_qkv_ptr,
    main_qkv_stride_token,
    main_q_norm_weight_ptr,
    main_k_norm_weight_ptr,
    main_eps,
    main_q_out_ptr,
    main_gate_out_ptr,
    main_cache_ptr,
    main_cache_stride_block,
    main_cache_stride_token,
    main_cache_stride_head,
    main_slots_ptr,
    main_k_scale,
    main_v_scale,
    MAIN_HQ: tl.constexpr,
    MAIN_HK: tl.constexpr,
    MAIN_D: tl.constexpr,
    MAIN_PAGE_SIZE: tl.constexpr,
    MAIN_DST_FP8: tl.constexpr,
    MAIN_KV_TILED: tl.constexpr,
    STREAMING_TILED: tl.constexpr,
    DECODE_ONLY: tl.constexpr,
    TILE_T: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    pid = tl.program_id(0)
    QUARTER: tl.constexpr = D // 4
    HALF: tl.constexpr = D // 2
    qcols = tl.arange(0, QUARTER)
    pcols = tl.arange(0, HALF)
    mcols = tl.arange(0, MAIN_D)
    mrot = tl.arange(0, QUARTER)
    # Work layout: [0, num_k_work) K compression + ring commit (1D items),
    # then the streaming items - 2-D row tiles in tiled mode (indexer-Q
    # token tiles, main-Q (head, tile) items, main-KV items), or the
    # original 1-D items in untiled mode. One persistent grid-stride loop
    # covers them all in a single launch.
    if STREAMING_TILED:
        num_index_work = num_k_work + num_q_tiles
        num_main_q_end = num_index_work + num_main_q_items
    else:
        num_index_work = num_k_work + num_tokens
        num_main_q_end = num_index_work
    for wid in range(pid, total_work, NUM_SPC):
        if wid < num_k_work:
            # One work item owns one completed compression group. Work item
            # zero additionally commits the request's current raw-K suffix
            # below.
            request = tl.load(k_work_metadata_ptr + wid * 2).to(tl.int32)
            work_in_request = tl.load(k_work_metadata_ptr + wid * 2 + 1).to(tl.int32)
            if request >= 0:
                query_start = tl.load(query_start_loc_ptr + request).to(tl.int32)
                query_end = tl.load(query_start_loc_ptr + request + 1).to(tl.int32)
                query_len = query_end - query_start
                chunk_end = tl.load(logical_positions_ptr + query_end - 1).to(tl.int32)
                chunk_start = chunk_end - query_len + 1
                num_groups = (
                    chunk_end + 1
                ) // COMPRESS_RATIO - chunk_start // COMPRESS_RATIO
                dims = tl.arange(0, D)

                if work_in_request < num_groups:
                    first_boundary = (
                        (chunk_start + COMPRESS_RATIO) // COMPRESS_RATIO
                    ) * COMPRESS_RATIO - 1
                    end_position = first_boundary + work_in_request * COMPRESS_RATIO
                    boundary_token = query_start + end_position - chunk_start
                    valid_token = (
                        (boundary_token >= query_start)
                        & (boundary_token < query_end)
                        & (boundary_token < num_tokens)
                    )
                    compressed_slot = tl.load(
                        compressed_slots_ptr + boundary_token,
                        mask=valid_token,
                        other=-1,
                    ).to(tl.int32)
                    valid = (
                        valid_token
                        & (compressed_slot >= 0)
                        & (compressed_slot < num_compressed_blocks * COMP_PAGE_SIZE)
                    )
                    state_block = tl.load(
                        state_table_ptr + request * state_table_stride_req
                    ).to(tl.int32)
                    state_block_valid = (state_block >= 0) & (
                        state_block < num_state_blocks
                    )
                    safe_state_block = tl.maximum(state_block, 0).to(tl.int64)
                    # Pool the group row by row: each row selects between
                    # the current chunk and the circular state ring with two
                    # mutually exclusive masked loads summed per column
                    # segment (a tl.where on pointer tiles lowers to an
                    # illegal arith.select on this backend). Rows pool in
                    # position order, matching the unfused reference.
                    acc0 = tl.zeros((QUARTER,), dtype=tl.float32)
                    acc1 = tl.zeros((QUARTER,), dtype=tl.float32)
                    accp = tl.zeros((HALF,), dtype=tl.float32)
                    for group_offset in tl.static_range(COMPRESS_RATIO):
                        position = end_position - (COMPRESS_RATIO - 1) + group_offset
                        source_in_chunk = position >= chunk_start
                        source_token = query_start + position - chunk_start
                        source_token_valid = (
                            (source_token >= query_start)
                            & (source_token < query_end)
                            & (source_token < num_tokens)
                        )
                        current_base = (
                            k_ptr
                            + tl.maximum(source_token, 0).to(tl.int64) * k_stride_token
                        )
                        # Clamp before the modulo: a negative position would
                        # wrap under the ring base and read out of bounds.
                        ring_idx = tl.maximum(position, 0) % STATE_SIZE
                        cached_base = (
                            state_cache_ptr
                            + safe_state_block * state_cache_stride_block
                            + ring_idx * state_cache_stride_token
                        )
                        cur_ok = valid & source_in_chunk & source_token_valid
                        cach_ok = valid & (~source_in_chunk) & state_block_valid
                        c0 = tl.load(current_base + qcols, mask=cur_ok, other=0.0)
                        h0 = tl.load(cached_base + qcols, mask=cach_ok, other=0.0)
                        acc0 += tl.where(source_in_chunk, c0, h0).to(tl.float32)
                        c1 = tl.load(
                            current_base + QUARTER + qcols, mask=cur_ok, other=0.0
                        )
                        h1 = tl.load(
                            cached_base + QUARTER + qcols, mask=cach_ok, other=0.0
                        )
                        acc1 += tl.where(source_in_chunk, c1, h1).to(tl.float32)
                        cp = tl.load(
                            current_base + HALF + pcols, mask=cur_ok, other=0.0
                        )
                        hp = tl.load(
                            cached_base + HALF + pcols, mask=cach_ok, other=0.0
                        )
                        accp += tl.where(source_in_chunk, cp, hp).to(tl.float32)
                    # Match the unfused path's BF16 pooled tensor before RMSNorm.
                    pooled0 = (acc0 / COMPRESS_RATIO).to(tl.bfloat16).to(tl.float32)
                    pooled1 = (acc1 / COMPRESS_RATIO).to(tl.bfloat16).to(tl.float32)
                    pooledp = (accp / COMPRESS_RATIO).to(tl.bfloat16).to(tl.float32)

                    first_position = end_position - (COMPRESS_RATIO - 1)
                    if CACHE_HAS_ROPE_POS:
                        # RoPE uses the first token in the pooled group. Its
                        # exact MRoPE coordinates may live in this chunk or
                        # in the raw-state ring tail.
                        first_in_chunk = first_position >= chunk_start
                        first_token = query_start + first_position - chunk_start
                        first_token_valid = (
                            (first_token >= query_start)
                            & (first_token < query_end)
                            & (first_token < num_tokens)
                        )
                        safe_first_token = tl.maximum(first_token, 0).to(tl.int64)
                        load_current_position = first_in_chunk & first_token_valid
                        first_pos_t = tl.load(
                            pos_ptr + safe_first_token * pos_stride_token,
                            mask=load_current_position,
                            other=0,
                        )
                        if IS_2D_POSITIONS:
                            first_pos_h = tl.load(
                                pos_ptr
                                + pos_stride_axis
                                + safe_first_token * pos_stride_token,
                                mask=load_current_position,
                                other=0,
                            )
                            first_pos_w = tl.load(
                                pos_ptr
                                + 2 * pos_stride_axis
                                + safe_first_token * pos_stride_token,
                                mask=load_current_position,
                                other=0,
                            )
                        else:
                            first_pos_h = first_pos_t
                            first_pos_w = first_pos_t
                        tail = (
                            state_cache_ptr
                            + safe_state_block * state_cache_stride_block
                            + (tl.maximum(first_position, 0) % STATE_SIZE)
                            * state_cache_stride_token
                            + D
                        ).to(tl.pointer_type(tl.int64))
                        load_cached_position = ~first_in_chunk & state_block_valid
                        cached_pos_t = tl.load(tail, mask=load_cached_position, other=0)
                        cached_pos_h = tl.load(
                            tail + 1, mask=load_cached_position, other=0
                        )
                        cached_pos_w = tl.load(
                            tail + 2, mask=load_cached_position, other=0
                        )
                        pos_t = tl.where(
                            first_in_chunk, first_pos_t.to(tl.int64), cached_pos_t
                        )
                        pos_h = tl.where(
                            first_in_chunk, first_pos_h.to(tl.int64), cached_pos_h
                        )
                        pos_w = tl.where(
                            first_in_chunk, first_pos_w.to(tl.int64), cached_pos_w
                        )
                    else:
                        pos_t = first_position
                        pos_h = first_position
                        pos_w = first_position
                    cos, sin = _cos_sin(
                        pos_t.to(tl.int32),
                        pos_h.to(tl.int32),
                        pos_w.to(tl.int32),
                        cos_sin_ptr,
                        D // 2,
                        IS_K_MROPE,
                        MROPE_H,
                        MROPE_W,
                        QUARTER,
                    )
                    rrms = tl.rsqrt(
                        (
                            tl.sum(pooled0 * pooled0)
                            + tl.sum(pooled1 * pooled1)
                            + tl.sum(pooledp * pooledp)
                        )
                        / D
                        + eps
                    )
                    kw0 = tl.load(k_norm_weight_ptr + qcols).to(tl.float32) + 1.0
                    kw1 = (
                        tl.load(k_norm_weight_ptr + QUARTER + qcols).to(tl.float32)
                        + 1.0
                    )
                    kwp = tl.load(k_norm_weight_ptr + HALF + pcols).to(tl.float32) + 1.0
                    r0 = (pooled0 * rrms * kw0).to(cos.dtype)
                    r1 = (pooled1 * rrms * kw1).to(cos.dtype)
                    rp = (pooledp * rrms * kwp).to(cos.dtype)
                    o0 = r0 * cos - r1 * sin
                    o1 = r1 * cos + r0 * sin
                    compressed_block = (compressed_slot // COMP_PAGE_SIZE).to(tl.int64)
                    compressed_row = compressed_slot % COMP_PAGE_SIZE
                    cbase = (
                        compressed_cache_ptr
                        + compressed_block * compressed_cache_stride_block
                        + compressed_row * compressed_cache_stride_token
                    )
                    tl.store(cbase + qcols, o0, mask=valid)
                    tl.store(cbase + QUARTER + qcols, o1, mask=valid)
                    tl.store(cbase + HALF + pcols, rp, mask=valid)

                if work_in_request == 0:
                    # This CTA may have just read historical rows from the
                    # circular buffer. Keep every lane past those loads
                    # before overwriting the same ring.
                    tl.debug_barrier()
                    if DECODE_ONLY:
                        # Every request contributes exactly one token here,
                        # so the commit is a single row.
                        token = (query_end - 1).to(tl.int64)
                        valid_token = (
                            (token >= query_start)
                            & (token < query_end)
                            & (token < num_tokens)
                        )
                        slot = tl.load(
                            state_slots_ptr + token, mask=valid_token, other=-1
                        ).to(tl.int32)
                        valid_slot = (
                            valid_token
                            & (slot >= 0)
                            & (slot < num_state_blocks * STATE_SIZE)
                        )
                        safe_slot = tl.maximum(slot, 0).to(tl.int64)
                        block_idx = safe_slot // STATE_SIZE
                        ring_idx = safe_slot % STATE_SIZE
                        state_row = (
                            state_cache_ptr
                            + block_idx * state_cache_stride_block
                            + ring_idx * state_cache_stride_token
                        )
                        k = tl.load(
                            k_ptr + token * k_stride_token + dims,
                            mask=valid_slot,
                            other=0.0,
                        )
                        tl.store(state_row + dims, k, mask=valid_slot)
                        if CACHE_HAS_ROPE_POS:
                            pos_t = tl.load(
                                pos_ptr + token * pos_stride_token,
                                mask=valid_slot,
                                other=0,
                            )
                            if IS_2D_POSITIONS:
                                pos_h = tl.load(
                                    pos_ptr
                                    + pos_stride_axis
                                    + token * pos_stride_token,
                                    mask=valid_slot,
                                    other=0,
                                )
                                pos_w = tl.load(
                                    pos_ptr
                                    + 2 * pos_stride_axis
                                    + token * pos_stride_token,
                                    mask=valid_slot,
                                    other=0,
                                )
                            else:
                                pos_h = pos_t
                                pos_w = pos_t
                            tail = (state_row + D).to(tl.pointer_type(tl.int64))
                            tl.store(tail, pos_t.to(tl.int64), mask=valid_slot)
                            tl.store(tail + 1, pos_h.to(tl.int64), mask=valid_slot)
                            tl.store(tail + 2, pos_w.to(tl.int64), mask=valid_slot)
                    else:
                        num_state_rows = tl.minimum(query_len, STATE_SIZE)
                        for state_offset in tl.range(0, num_state_rows):
                            token = (query_end - num_state_rows + state_offset).to(
                                tl.int64
                            )
                            valid_token = (
                                (token >= query_start)
                                & (token < query_end)
                                & (token < num_tokens)
                            )
                            slot = tl.load(
                                state_slots_ptr + token,
                                mask=valid_token,
                                other=-1,
                            ).to(tl.int32)
                            valid_slot = (
                                valid_token
                                & (slot >= 0)
                                & (slot < num_state_blocks * STATE_SIZE)
                            )
                            safe_slot = tl.maximum(slot, 0).to(tl.int64)
                            block_idx = safe_slot // STATE_SIZE
                            ring_idx = safe_slot % STATE_SIZE
                            state_row = (
                                state_cache_ptr
                                + block_idx * state_cache_stride_block
                                + ring_idx * state_cache_stride_token
                            )
                            k = tl.load(
                                k_ptr + token * k_stride_token + dims,
                                mask=valid_slot,
                                other=0.0,
                            )
                            tl.store(state_row + dims, k, mask=valid_slot)
                            if CACHE_HAS_ROPE_POS:
                                pos_t = tl.load(
                                    pos_ptr + token * pos_stride_token,
                                    mask=valid_slot,
                                    other=0,
                                )
                                if IS_2D_POSITIONS:
                                    pos_h = tl.load(
                                        pos_ptr
                                        + pos_stride_axis
                                        + token * pos_stride_token,
                                        mask=valid_slot,
                                        other=0,
                                    )
                                    pos_w = tl.load(
                                        pos_ptr
                                        + 2 * pos_stride_axis
                                        + token * pos_stride_token,
                                        mask=valid_slot,
                                        other=0,
                                    )
                                else:
                                    pos_h = pos_t
                                    pos_w = pos_t
                                tail = (state_row + D).to(tl.pointer_type(tl.int64))
                                tl.store(tail, pos_t.to(tl.int64), mask=valid_slot)
                                tl.store(tail + 1, pos_h.to(tl.int64), mask=valid_slot)
                                tl.store(tail + 2, pos_w.to(tl.int64), mask=valid_slot)

        if STREAMING_TILED:
            # Batched 2-D row-tile arms (large batches).
            if wid >= num_k_work:
                # Batched 2-D row-tile arms (large batches).
                if wid < num_index_work:
                    # Indexer Q: one token tile per item; the head loop is unrolled
                    # so the position loads, the rope table rows and the norm
                    # weights are shared across all Q heads of the tile.
                    tile = wid - num_k_work
                    tokens = tile * TILE_T + tl.arange(0, TILE_T)
                    row_valid = tokens < num_tokens
                    pos_t = tl.load(
                        pos_ptr + tokens * pos_stride_token, mask=row_valid, other=0
                    ).to(tl.int32)
                    if IS_2D_POSITIONS:
                        pos_h = tl.load(
                            pos_ptr + pos_stride_axis + tokens * pos_stride_token,
                            mask=row_valid,
                            other=0,
                        ).to(tl.int32)
                        pos_w = tl.load(
                            pos_ptr + 2 * pos_stride_axis + tokens * pos_stride_token,
                            mask=row_valid,
                            other=0,
                        ).to(tl.int32)
                    else:
                        pos_h = pos_t
                        pos_w = pos_t
                    cos, sin = _cos_sin_tile(
                        pos_t,
                        pos_h,
                        pos_w,
                        row_valid,
                        cos_sin_ptr,
                        D // 2,
                        IS_2D_POSITIONS,
                        MROPE_H,
                        MROPE_W,
                        QUARTER,
                        False,
                    )
                    # Weights load with a 2-D shape on purpose: a 1-D load
                    # broadcast as a 2-D-tile multiply operand hits a known silent
                    # miscompile on this backend.
                    w0 = (
                        tl.load(q_norm_weight_ptr + qcols[None, :]).to(tl.float32) + 1.0
                    )
                    w1 = (
                        tl.load(q_norm_weight_ptr + QUARTER + qcols[None, :]).to(
                            tl.float32
                        )
                        + 1.0
                    )
                    wp = (
                        tl.load(q_norm_weight_ptr + HALF + pcols[None, :]).to(
                            tl.float32
                        )
                        + 1.0
                    )
                    for q_head in tl.static_range(HQ):
                        qbase = q_ptr + tokens[:, None] * q_stride_token + q_head * D
                        m = row_valid[:, None]
                        x0 = tl.load(qbase + qcols[None, :], mask=m, other=0.0).to(
                            tl.float32
                        )
                        x1 = tl.load(
                            qbase + QUARTER + qcols[None, :], mask=m, other=0.0
                        ).to(tl.float32)
                        xp = tl.load(
                            qbase + HALF + pcols[None, :], mask=m, other=0.0
                        ).to(tl.float32)
                        rrms = tl.rsqrt(
                            (
                                tl.sum(x0 * x0, axis=1)
                                + tl.sum(x1 * x1, axis=1)
                                + tl.sum(xp * xp, axis=1)
                            )
                            / D
                            + eps
                        )
                        r0 = (x0 * rrms[:, None] * w0).to(cos.dtype)
                        r1 = (x1 * rrms[:, None] * w1).to(cos.dtype)
                        rp = (xp * rrms[:, None] * wp).to(cos.dtype)
                        o0 = r0 * cos - r1 * sin
                        o1 = r1 * cos + r0 * sin
                        obase = (
                            q_out_ptr
                            + tokens[:, None] * q_out_stride_token
                            + q_head * q_out_stride_head
                        )
                        tl.store(obase + qcols[None, :], o0, mask=m)
                        tl.store(obase + QUARTER + qcols[None, :], o1, mask=m)
                        tl.store(obase + HALF + pcols[None, :], rp, mask=m)

                elif wid < num_main_q_end:
                    # Main attention Q heads: one (head, token-tile) item. RoPE
                    # covers the first D // 2 dims, the width of the shared
                    # cos/sin table.
                    mq_item = wid - num_index_work
                    head = mq_item // num_main_tiles
                    tile = mq_item % num_main_tiles
                    tokens = tile * TILE_T + tl.arange(0, TILE_T)
                    row_valid = tokens < num_tokens
                    m = row_valid[:, None]
                    src = (
                        main_qkv_ptr
                        + tokens[:, None] * main_qkv_stride_token
                        + 2 * head * MAIN_D
                    )
                    x = tl.load(src + mcols[None, :], mask=m, other=0.0).to(tl.float32)
                    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / MAIN_D + main_eps)
                    w = (
                        tl.load(main_q_norm_weight_ptr + mcols[None, :]).to(tl.float32)
                        + 1.0
                    )
                    y = x * inv_rms[:, None] * w
                    x1 = tl.load(src + mrot[None, :], mask=m, other=0.0).to(tl.float32)
                    x2 = tl.load(src + QUARTER + mrot[None, :], mask=m, other=0.0).to(
                        tl.float32
                    )
                    w1 = (
                        tl.load(main_q_norm_weight_ptr + mrot[None, :]).to(tl.float32)
                        + 1.0
                    )
                    w2 = (
                        tl.load(main_q_norm_weight_ptr + QUARTER + mrot[None, :]).to(
                            tl.float32
                        )
                        + 1.0
                    )
                    x1 = (x1 * inv_rms[:, None] * w1).to(tl.bfloat16).to(tl.float32)
                    x2 = (x2 * inv_rms[:, None] * w2).to(tl.bfloat16).to(tl.float32)
                    pos_t = tl.load(
                        pos_ptr + tokens * pos_stride_token, mask=row_valid, other=0
                    ).to(tl.int32)
                    if IS_2D_POSITIONS:
                        pos_h = tl.load(
                            pos_ptr + pos_stride_axis + tokens * pos_stride_token,
                            mask=row_valid,
                            other=0,
                        ).to(tl.int32)
                        pos_w = tl.load(
                            pos_ptr + 2 * pos_stride_axis + tokens * pos_stride_token,
                            mask=row_valid,
                            other=0,
                        ).to(tl.int32)
                    else:
                        pos_h = pos_t
                        pos_w = pos_t
                    cos, sin = _cos_sin_tile(
                        pos_t,
                        pos_h,
                        pos_w,
                        row_valid,
                        cos_sin_ptr,
                        D // 2,
                        IS_2D_POSITIONS,
                        MROPE_H,
                        MROPE_W,
                        QUARTER,
                        True,
                    )
                    o1 = x1 * cos - x2 * sin
                    o2 = x2 * cos + x1 * sin
                    out_base = (
                        main_q_out_ptr + (tokens[:, None] * MAIN_HQ + head) * MAIN_D
                    )
                    _store_rotated_2d(out_base, m, y, o1, o2, 1.0, False)
                    gate = tl.load(src + MAIN_D + mcols[None, :], mask=m, other=0.0)
                    tl.store(
                        main_gate_out_ptr
                        + (tokens[:, None] * MAIN_HQ + head) * MAIN_D
                        + mcols[None, :],
                        gate,
                        mask=m,
                    )

                else:
                    mkv_item = wid - num_main_q_end
                    if MAIN_KV_TILED:
                        # Main attention K/V heads: one (kv head, token-tile) item.
                        # The paged-cache store is a per-row slot scatter with i32
                        # lane arithmetic (the host guards the cache size).
                        kv_head = mkv_item // num_main_tiles
                        tile = mkv_item % num_main_tiles
                        tokens = tile * TILE_T + tl.arange(0, TILE_T)
                        row_valid = tokens < num_tokens
                        m = row_valid[:, None]
                        k_src = (2 * MAIN_HQ + kv_head) * MAIN_D
                        v_src = (2 * MAIN_HQ + MAIN_HK + kv_head) * MAIN_D
                        src = (
                            main_qkv_ptr
                            + tokens[:, None] * main_qkv_stride_token
                            + k_src
                        )
                        x = tl.load(src + mcols[None, :], mask=m, other=0.0).to(
                            tl.float32
                        )
                        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / MAIN_D + main_eps)
                        w = tl.load(main_k_norm_weight_ptr + mcols[None, :]).to(
                            tl.float32
                        )
                        w += 1.0
                        y = x * inv_rms[:, None] * w
                        x1 = tl.load(src + mrot[None, :], mask=m, other=0.0).to(
                            tl.float32
                        )
                        x2 = tl.load(
                            src + QUARTER + mrot[None, :], mask=m, other=0.0
                        ).to(tl.float32)
                        w1 = (
                            tl.load(main_k_norm_weight_ptr + mrot[None, :]).to(
                                tl.float32
                            )
                            + 1.0
                        )
                        w2 = (
                            tl.load(
                                main_k_norm_weight_ptr + QUARTER + mrot[None, :]
                            ).to(tl.float32)
                            + 1.0
                        )
                        x1 = (x1 * inv_rms[:, None] * w1).to(tl.bfloat16).to(tl.float32)
                        x2 = (x2 * inv_rms[:, None] * w2).to(tl.bfloat16).to(tl.float32)
                        pos_t = tl.load(
                            pos_ptr + tokens * pos_stride_token, mask=row_valid, other=0
                        ).to(tl.int32)
                        if IS_2D_POSITIONS:
                            pos_h = tl.load(
                                pos_ptr + pos_stride_axis + tokens * pos_stride_token,
                                mask=row_valid,
                                other=0,
                            ).to(tl.int32)
                            pos_w = tl.load(
                                pos_ptr
                                + 2 * pos_stride_axis
                                + tokens * pos_stride_token,
                                mask=row_valid,
                                other=0,
                            ).to(tl.int32)
                        else:
                            pos_h = pos_t
                            pos_w = pos_t
                        cos, sin = _cos_sin_tile(
                            pos_t,
                            pos_h,
                            pos_w,
                            row_valid,
                            cos_sin_ptr,
                            D // 2,
                            IS_2D_POSITIONS,
                            MROPE_H,
                            MROPE_W,
                            QUARTER,
                            True,
                        )
                        o1 = x1 * cos - x2 * sin
                        o2 = x2 * cos + x1 * sin
                        v = tl.load(
                            main_qkv_ptr
                            + tokens[:, None] * main_qkv_stride_token
                            + v_src
                            + mcols[None, :],
                            mask=m,
                            other=0.0,
                        )
                        slot = tl.load(
                            main_slots_ptr + tokens, mask=row_valid, other=-1
                        ).to(tl.int32)
                        slot_valid = row_valid & (slot >= 0)
                        safe_slot = tl.maximum(slot, 0)
                        block_id = safe_slot // MAIN_PAGE_SIZE
                        row_id = safe_slot % MAIN_PAGE_SIZE
                        dst = (
                            main_cache_ptr
                            + block_id[:, None] * main_cache_stride_block
                            + row_id[:, None] * main_cache_stride_token
                            + kv_head * main_cache_stride_head
                        )
                        m2 = slot_valid[:, None]
                        _store_rotated_2d(
                            dst, m2, y, o1, o2, main_k_scale, MAIN_DST_FP8
                        )
                        tl.store(
                            dst + MAIN_D + mcols[None, :],
                            _to_dst_dtype(v, dst, main_v_scale, MAIN_DST_FP8),
                            mask=m2,
                        )
                    else:
                        # i32-unsafe caches: per-(token, head) items with int64
                        # scalar row bases.
                        token = (mkv_item // MAIN_HK).to(tl.int64)
                        kv_head = mkv_item % MAIN_HK
                        dims = tl.arange(0, MAIN_D)
                        rot = tl.arange(0, QUARTER)
                        row = main_qkv_ptr + token * main_qkv_stride_token
                        src = row + (2 * MAIN_HQ + kv_head) * MAIN_D
                        weight_ptr = main_k_norm_weight_ptr
                        x = tl.load(src + dims).to(tl.float32)
                        inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / MAIN_D + main_eps)
                        w = tl.load(weight_ptr + dims).to(tl.float32) + 1.0
                        y = x * inv_rms * w
                        x1 = tl.load(src + rot).to(tl.float32)
                        x2 = tl.load(src + QUARTER + rot).to(tl.float32)
                        w1 = tl.load(weight_ptr + rot).to(tl.float32) + 1.0
                        w2 = tl.load(weight_ptr + QUARTER + rot).to(tl.float32) + 1.0
                        x1 = (x1 * inv_rms * w1).to(tl.bfloat16).to(tl.float32)
                        x2 = (x2 * inv_rms * w2).to(tl.bfloat16).to(tl.float32)
                        pos = tl.load(pos_ptr + token * pos_stride_token).to(tl.int64)
                        if IS_2D_POSITIONS:
                            pos_h = tl.load(
                                pos_ptr + pos_stride_axis + token * pos_stride_token
                            )
                            pos_w = tl.load(
                                pos_ptr + 2 * pos_stride_axis + token * pos_stride_token
                            )
                            is_h = (rot % 3 == 1) & (rot < 3 * MROPE_H)
                            is_w = (rot % 3 == 2) & (rot < 3 * MROPE_W)
                            pos = tl.where(
                                is_h,
                                pos_h.to(tl.int64),
                                tl.where(is_w, pos_w.to(tl.int64), pos),
                            )
                        cos = tl.load(cos_sin_ptr + pos * (D // 2) + rot).to(tl.float32)
                        sin = tl.load(cos_sin_ptr + pos * (D // 2) + QUARTER + rot).to(
                            tl.float32
                        )
                        o1 = x1 * cos - x2 * sin
                        o2 = x2 * cos + x1 * sin
                        slot = tl.load(main_slots_ptr + token).to(tl.int64)
                        if slot >= 0:
                            dst = (
                                main_cache_ptr
                                + (slot // MAIN_PAGE_SIZE) * main_cache_stride_block
                                + (slot % MAIN_PAGE_SIZE) * main_cache_stride_token
                                + kv_head * main_cache_stride_head
                            )
                            _store_rotated(dst, y, o1, o2, main_k_scale, MAIN_DST_FP8)
                            v = tl.load(src + MAIN_HK * MAIN_D + dims)
                            tl.store(
                                dst + MAIN_D + dims,
                                _to_dst_dtype(v, dst, main_v_scale, MAIN_DST_FP8),
                            )

        else:
            # Small batches: the original 1-D per-item arms. A 2-D tile
            # has a flat per-item cost, so token counts that cannot fill
            # the fixed grid with tiles are cheaper as per-token /
            # per-(token, head) items.
            if wid >= num_k_work:
                # Small batches: the original 1-D per-item arms. A 2-D tile has
                # a flat per-item cost, so token counts that cannot fill the
                # fixed grid with tiles are cheaper as per-token / per-(token,
                # head) items.
                if wid < num_index_work:
                    # Indexer Q: one work item per token; the head loop is
                    # unrolled so the position loads and the rope table row
                    # are shared across all Q heads of the token.
                    token = (wid - num_k_work).to(tl.int64)
                    pos_t = tl.load(pos_ptr + token * pos_stride_token)
                    if IS_2D_POSITIONS:
                        pos_h = tl.load(
                            pos_ptr + pos_stride_axis + token * pos_stride_token
                        )
                        pos_w = tl.load(
                            pos_ptr + 2 * pos_stride_axis + token * pos_stride_token
                        )
                    else:
                        pos_h = pos_t
                        pos_w = pos_t
                    # Normalize at the call boundary so both int64 layouts
                    # (int32 word views / direct int64) share one i32 form.
                    cos, sin = _cos_sin(
                        pos_t.to(tl.int32),
                        pos_h.to(tl.int32),
                        pos_w.to(tl.int32),
                        cos_sin_ptr,
                        D // 2,
                        IS_2D_POSITIONS,
                        MROPE_H,
                        MROPE_W,
                        QUARTER,
                    )
                    w0 = tl.load(q_norm_weight_ptr + qcols).to(tl.float32) + 1.0
                    w1 = (
                        tl.load(q_norm_weight_ptr + QUARTER + qcols).to(tl.float32)
                        + 1.0
                    )
                    wp = tl.load(q_norm_weight_ptr + HALF + pcols).to(tl.float32) + 1.0
                    for q_head in tl.static_range(HQ):
                        qbase = q_ptr + token * q_stride_token + q_head * D
                        x0 = tl.load(qbase + qcols).to(tl.float32)
                        x1 = tl.load(qbase + QUARTER + qcols).to(tl.float32)
                        xp = tl.load(qbase + HALF + pcols).to(tl.float32)
                        rrms = tl.rsqrt(
                            (tl.sum(x0 * x0) + tl.sum(x1 * x1) + tl.sum(xp * xp)) / D
                            + eps
                        )
                        r0 = (x0 * rrms * w0).to(cos.dtype)
                        r1 = (x1 * rrms * w1).to(cos.dtype)
                        rp = (xp * rrms * wp).to(cos.dtype)
                        o0 = r0 * cos - r1 * sin
                        o1 = r1 * cos + r0 * sin
                        obase = (
                            q_out_ptr
                            + token * q_out_stride_token
                            + q_head * q_out_stride_head
                        )
                        tl.store(obase + qcols, o0)
                        tl.store(obase + QUARTER + qcols, o1)
                        tl.store(obase + HALF + pcols, rp)
                else:
                    # Main attention: one program per (token, Q or KV head).
                    # RoPE covers the first D // 2 dims, the width of the shared
                    # cos/sin table.
                    main_pid = wid - num_index_work
                    token = (main_pid // (MAIN_HQ + MAIN_HK)).to(tl.int64)
                    head = main_pid % (MAIN_HQ + MAIN_HK)
                    dims = tl.arange(0, MAIN_D)
                    rot = tl.arange(0, QUARTER)
                    row = main_qkv_ptr + token * main_qkv_stride_token
                    is_k = head >= MAIN_HQ
                    kv_head = head - MAIN_HQ
                    if is_k:
                        src = row + (2 * MAIN_HQ + kv_head) * MAIN_D
                        weight_ptr = main_k_norm_weight_ptr
                    else:
                        src = row + 2 * head * MAIN_D
                        weight_ptr = main_q_norm_weight_ptr
                    x = tl.load(src + dims).to(tl.float32)
                    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / MAIN_D + main_eps)
                    w = tl.load(weight_ptr + dims).to(tl.float32) + 1.0
                    y = x * inv_rms * w
                    x1 = tl.load(src + rot).to(tl.float32)
                    x2 = tl.load(src + QUARTER + rot).to(tl.float32)
                    w1 = tl.load(weight_ptr + rot).to(tl.float32) + 1.0
                    w2 = tl.load(weight_ptr + QUARTER + rot).to(tl.float32) + 1.0
                    x1 = (x1 * inv_rms * w1).to(tl.bfloat16).to(tl.float32)
                    x2 = (x2 * inv_rms * w2).to(tl.bfloat16).to(tl.float32)
                    pos = tl.load(pos_ptr + token * pos_stride_token).to(tl.int64)
                    if IS_2D_POSITIONS:
                        pos_h = tl.load(
                            pos_ptr + pos_stride_axis + token * pos_stride_token
                        )
                        pos_w = tl.load(
                            pos_ptr + 2 * pos_stride_axis + token * pos_stride_token
                        )
                        is_h = (rot % 3 == 1) & (rot < 3 * MROPE_H)
                        is_w = (rot % 3 == 2) & (rot < 3 * MROPE_W)
                        pos = tl.where(
                            is_h,
                            pos_h.to(tl.int64),
                            tl.where(is_w, pos_w.to(tl.int64), pos),
                        )
                    cos = tl.load(cos_sin_ptr + pos * (D // 2) + rot).to(tl.float32)
                    sin = tl.load(cos_sin_ptr + pos * (D // 2) + QUARTER + rot).to(
                        tl.float32
                    )
                    o1 = x1 * cos - x2 * sin
                    o2 = x2 * cos + x1 * sin
                    if is_k:
                        slot = tl.load(main_slots_ptr + token).to(tl.int64)
                        if slot >= 0:
                            dst = (
                                main_cache_ptr
                                + (slot // MAIN_PAGE_SIZE) * main_cache_stride_block
                                + (slot % MAIN_PAGE_SIZE) * main_cache_stride_token
                                + kv_head * main_cache_stride_head
                            )
                            _store_rotated(dst, y, o1, o2, main_k_scale, MAIN_DST_FP8)
                            v = tl.load(src + MAIN_HK * MAIN_D + dims)
                            tl.store(
                                dst + MAIN_D + dims,
                                _to_dst_dtype(v, dst, main_v_scale, MAIN_DST_FP8),
                            )
                    else:
                        out = (token * MAIN_HQ + head) * MAIN_D
                        _store_rotated(main_q_out_ptr + out, y, o1, o2, 1.0, False)
                        gate = tl.load(src + MAIN_D + dims)
                        tl.store(main_gate_out_ptr + out + dims, gate)


def _gcu_index_payload(t: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize integer index tensors for both torch_gcu int64 layouts.

    torch_gcu stores int64 tensors either as a 4-byte physical payload
    (``TORCH_GCU_ENABLE_INT64_AND_UINT64`` unset: one ``GCU not support
    Long use Int replace`` warning per process) or as true 8-byte slots
    (the env set). In the 4-byte mode a kernel reading the tensor at its
    declared int64 width straddles two payload words per element, so the
    host re-exposes the payload as an int32 word view: the view reuses
    the same storage with the original shape, stride, and storage offset
    (slot i holds element i). ``Tensor.view(torch.int32)`` must not be
    used because it doubles the storage offset and misaligns sliced
    inputs. In the 8-byte mode the tensor is handed to the kernel as-is;
    the kernel widens loaded index scalars on demand, so both layouts
    read the correct values. Values stay within the i32 safe domain
    (positions/slots < 2**31) by production contract. Non-int64 tensors
    pass through unchanged.
    """
    if t.dtype is not torch.int64:
        return t
    if _int64_env_on():
        return t
    if int(t.numel()) == 0:
        return torch.empty(
            tuple(int(x) for x in t.shape), dtype=torch.int32, device=t.device
        )
    if int(t.stride(-1)) != 1:
        raise ValueError(
            f"qsa_prepare: {name} must be innermost-contiguous to reinterpret "
            f"its int64 slots as int32 (GCU narrows int64 tensors to 4-byte "
            f"slots); got shape={tuple(t.shape)} stride={tuple(t.stride())}"
        )
    words = torch.empty(0, dtype=torch.int32, device=t.device)
    return words.set_(
        t.untyped_storage(),
        int(t.storage_offset()),
        tuple(int(x) for x in t.shape),
        tuple(int(x) for x in t.stride()),
    )


def _tile_tokens(num_tokens: int, fp8_dst: bool) -> int:
    """Tokens per streaming tile for the batched arms.

    Wide tiles amortize the flat per-item cost of the 2-D row batches, so
    the tile climbs to the dtype ceiling (fp8 destinations cliff above 32
    rows - measured 3.4x on the 4k fp8 shape - while bf16 keeps improving
    to 64). The ladder then halves the tile until the eight main items per
    token tile can fill the 48-program grid.
    """
    tile = 32 if fp8_dst else 64
    while tile > 8 and triton.cdiv(num_tokens, tile) * 8 < GCU_NUM_GRID:
        tile //= 2
    return tile


def qsa_prepare(
    q,
    k,
    positions,
    cos_sin_cache,
    q_norm_weight,
    k_norm_weight,
    eps,
    q_out,
    state_cache,
    state_slots,
    state_block_table,
    query_start_loc,
    logical_positions,
    compressed_cache,
    compressed_slots,
    k_work_metadata,
    *,
    compress_ratio,
    mrope_section,
    rope_pos_offset,
    main_qkv,
    main_q_norm_weight,
    main_k_norm_weight,
    main_eps,
    main_kv_cache,
    main_slot_mapping,
    main_k_scale,
    main_v_scale,
):
    """Normalize Q, compress K, update the circular raw state, and prepare
    the main attention (QK-norm/RoPE, gate copy, K/V cache write) in one
    launch; returns the main attention's Q and gate."""
    num_tokens = q.shape[0]
    main_head_dim = main_kv_cache.shape[-1] // 2
    num_main_kv_heads = main_kv_cache.shape[2]
    num_main_q_heads = main_qkv.shape[1] // (2 * main_head_dim) - num_main_kv_heads
    main_q_out = main_qkv.new_empty(num_tokens, num_main_q_heads, main_head_dim)
    main_gate_out = torch.empty_like(main_q_out)
    if num_tokens == 0:
        return main_q_out, main_gate_out
    num_q_heads, head_dim = q_out.shape[1:]
    assert cos_sin_cache.shape[-1] * 2 == head_dim
    assert q.shape == (num_tokens, num_q_heads * head_dim)
    assert k.shape == (num_tokens, head_dim)
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert q_out.stride(-1) == 1
    assert cos_sin_cache.is_contiguous()
    assert state_cache.stride(-1) == 1
    assert compressed_cache.stride(-1) == 1
    assert k_work_metadata.ndim == 2 and k_work_metadata.shape[1] == 2
    is_2d_positions = positions.ndim == 2
    is_k_mrope = bool(mrope_section)
    cache_has_rope_pos = rope_pos_offset is not None
    assert rope_pos_offset is None or rope_pos_offset == head_dim
    if is_2d_positions:
        assert positions.shape == (3, num_tokens)
        assert is_k_mrope
        pos_stride_axis, pos_stride_token = positions.stride()
    else:
        assert positions.shape == (num_tokens,)
        pos_stride_axis, pos_stride_token = 0, positions.stride(0)
    section = mrope_section if mrope_section is not None else (0, 0, 0)
    assert len(section) == 3
    qkv_width = 2 * (num_main_q_heads + num_main_kv_heads) * main_head_dim
    assert main_qkv.shape == (num_tokens, qkv_width)
    assert main_qkv.stride(-1) == 1
    assert main_slot_mapping.shape == (num_tokens,)
    # The batched segments address outputs and caches with i32 lane
    # arithmetic; guard the element counts of the widest tensors.
    assert main_q_out.numel() < 2**31
    assert state_cache.numel() < 2**31
    assert compressed_cache.numel() < 2**31
    main_kv_tiled = int(main_kv_cache.numel()) < _SCATTER_I32_MAX_NUMEL

    pos_arg = _gcu_index_payload(positions, "positions")
    lp_arg = _gcu_index_payload(logical_positions, "logical_positions")
    sslots_arg = _gcu_index_payload(state_slots, "state_slots")
    cslots_arg = _gcu_index_payload(compressed_slots, "compressed_slots")
    mslots_arg = _gcu_index_payload(main_slot_mapping, "main_slot_mapping")
    qsl_arg = _gcu_index_payload(query_start_loc, "query_start_loc")
    kwm_arg = _gcu_index_payload(k_work_metadata, "k_work_metadata")
    sbt_arg = _gcu_index_payload(state_block_table, "state_block_table")

    num_k_work = k_work_metadata.shape[0]
    # 2-D row-tile items have a flat per-item cost; batches too small to
    # fill the fixed grid with tiles run faster as the original 1-D items
    # (measured crossover at 64 tokens: at 54 tokens the 1-D arm wins
    # ~15-20%, at 64 they tie and above that the tiles win up to 2.5x).
    streaming_tiled = num_tokens >= 64
    if streaming_tiled:
        fp8_dst = (
            main_kv_cache.dtype == torch.float8_e4m3fn
            or q_out.dtype == torch.float8_e4m3fn
        )
        tile_t = _tile_tokens(num_tokens, fp8_dst)
        num_main_tiles = triton.cdiv(num_tokens, tile_t)
        num_q_tiles = num_main_tiles
        num_main_q_items = num_main_q_heads * num_main_tiles
        if main_kv_tiled:
            num_main_kv_items = num_main_kv_heads * num_main_tiles
        else:
            num_main_kv_items = num_tokens * num_main_kv_heads
    else:
        tile_t = 1
        num_main_tiles = num_tokens
        num_q_tiles = num_tokens
        num_main_q_items = num_tokens * (num_main_q_heads + num_main_kv_heads)
        num_main_kv_items = 0
    # A 16k-token prefill needs ~200k seed work items, far beyond the
    # 65535 grid.x limit, so the launch is persistent with a grid-stride
    # loop sized to the tuned CTA count.
    total_work = num_k_work + num_q_tiles + num_main_q_items + num_main_kv_items
    grid0 = min(GCU_NUM_GRID, total_work)
    # Sum(query_lens) == num_tokens with one query per request means
    # every query_len is 1, so each request's ring commit is one row.
    decode_only = num_tokens == query_start_loc.shape[0] - 1
    _qsa_prepare_kernel[(grid0,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        pos_arg,
        pos_stride_axis,
        pos_stride_token,
        cos_sin_cache,
        q_norm_weight,
        k_norm_weight,
        eps,
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        sslots_arg,
        sbt_arg,
        state_block_table.stride(0),
        qsl_arg,
        lp_arg,
        cslots_arg,
        kwm_arg,
        compressed_cache,
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        num_tokens,
        state_cache.shape[0],
        compressed_cache.shape[0],
        num_k_work,
        num_q_tiles,
        num_main_tiles,
        num_main_q_items,
        num_main_kv_items,
        total_work,
        HQ=num_q_heads,
        D=head_dim,
        COMPRESS_RATIO=compress_ratio,
        STATE_SIZE=state_cache.shape[1],
        COMP_PAGE_SIZE=compressed_cache.shape[1],
        IS_2D_POSITIONS=is_2d_positions,
        IS_K_MROPE=is_k_mrope,
        CACHE_HAS_ROPE_POS=cache_has_rope_pos,
        MROPE_H=section[1],
        MROPE_W=section[2],
        main_qkv_ptr=main_qkv,
        main_qkv_stride_token=main_qkv.stride(0),
        main_q_norm_weight_ptr=main_q_norm_weight,
        main_k_norm_weight_ptr=main_k_norm_weight,
        main_eps=main_eps,
        main_q_out_ptr=main_q_out,
        main_gate_out_ptr=main_gate_out,
        main_cache_ptr=main_kv_cache,
        main_cache_stride_block=main_kv_cache.stride(0),
        main_cache_stride_token=main_kv_cache.stride(1),
        main_cache_stride_head=main_kv_cache.stride(2),
        main_slots_ptr=mslots_arg,
        main_k_scale=main_k_scale,
        main_v_scale=main_v_scale,
        MAIN_HQ=num_main_q_heads,
        MAIN_HK=num_main_kv_heads,
        MAIN_D=main_head_dim,
        MAIN_PAGE_SIZE=main_kv_cache.shape[1],
        MAIN_DST_FP8=main_kv_cache.dtype == torch.float8_e4m3fn,
        MAIN_KV_TILED=main_kv_tiled,
        STREAMING_TILED=streaming_tiled,
        DECODE_ONLY=decode_only,
        TILE_T=tile_t,
        NUM_SPC=grid0,
        num_warps=1,
    )
    return main_q_out, main_gate_out


__all__ = ["qsa_prepare"]

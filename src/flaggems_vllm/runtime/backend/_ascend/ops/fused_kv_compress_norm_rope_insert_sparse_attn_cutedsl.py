# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ascend packed C4 compression and group-wise cache insertion."""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime.backend._ascend.ops.per_token_group_quant_fp8 import (
    _f32_to_fp8_e4m3fn,
)

logger = logging.getLogger(__name__)


@triton.jit
def _compress_tiles(
    State,
    Request,
    Position,
    Slot,
    Table,
    Destination,
    Work,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
    N_REQ: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    STATE_PAGES: tl.constexpr,
    STATE_PAGE: tl.constexpr,
    STATE_ROW: tl.constexpr,
    STATE_COL: tl.constexpr,
    ROTARY_LEN: tl.constexpr,
    KV_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    tiles: tl.constexpr = 512 // TILE
    for item in range(tl.program_id(0).to(tl.int64), N * tiles, tl.num_programs(0)):
        token = item // tiles
        tile = (item % tiles).to(tl.int32)
        col = tile * TILE + tl.arange(0, TILE)
        slot = tl.load(Slot + token * SLOT_STRIDE)
        pos = tl.load(Position + token * POSITION_STRIDE)
        dst = tl.load(Destination + token * DEST_STRIDE)
        req = tl.load(Request + token * REQUEST_STRIDE).to(tl.int64)
        valid = (slot >= 0) & (pos >= 3) & ((pos & 3) == 3) & (pos - 3 < ROTARY_LEN)
        valid &= (dst >= 0) & (dst < KV_PAGES * PAGE_SIZE)
        valid &= (req >= 0) & (req < N_REQ) & (pos // BLOCK_SIZE < TABLE_WIDTH)
        if valid:
            peak = tl.full((TILE,), -float("inf"), tl.float32)
            mass = tl.full((TILE,), 0.0, tl.float32)
            weighted = tl.full((TILE,), 0.0, tl.float32)
            for j in range(8):
                loc = pos - 7 + j
                if loc >= 0:
                    page = tl.load(
                        Table + req * TABLE_ROW + (loc // BLOCK_SIZE) * TABLE_COL
                    )
                    in_bounds = (page >= 0) & (page < STATE_PAGES)
                    valid &= in_bounds
                    offset = (
                        page.to(tl.int64) * STATE_PAGE + (loc % BLOCK_SIZE) * STATE_ROW
                    )
                    ptrs = (
                        State + offset + ((j // 4) * 512 + col).to(tl.int64) * STATE_COL
                    )
                    score = tl.load(ptrs + 1024 * STATE_COL, in_bounds, 0.0)
                    value = tl.load(ptrs, in_bounds, 0.0)
                    next_peak = tl.maximum(peak, score)
                    old_scale = tl.exp(peak - next_peak)
                    new_scale = tl.exp(score - next_peak)
                    mass = mass * old_scale + new_scale
                    weighted = weighted * old_scale + value * new_scale
                    peak = next_peak
            tl.store(Work + token * 512 + col, weighted / mass)
        if tile == 0:
            tl.store(Work + N * 512 + token, valid.to(tl.float32))


@triton.jit
def _encode_group(x, FP8_MAX: tl.constexpr):
    if FP8_MAX < 448.0:
        x = tl.minimum(tl.maximum(x, 0.0 - FP8_MAX), FP8_MAX)
    else:
        x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    return _f32_to_fp8_e4m3fn(x)


@triton.jit
def _write_bf16(ptr, index, values):
    bits = values.to(tl.float32).to(tl.int32, bitcast=True)
    tl.store(ptr + index * 2, ((bits >> 16) & 255).to(tl.uint8))
    tl.store(ptr + index * 2 + 1, ((bits >> 24) & 255).to(tl.uint8))


@triton.jit
def _store_packed_groups(
    Work,
    Position,
    Weight,
    Rotary,
    Cache,
    KvSlot,
    eps,
    N: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    ROTARY_ROW: tl.constexpr,
    ROTARY_COL: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    for item in range(tl.program_id(0).to(tl.int64), N * 8, tl.num_programs(0)):
        token = item // 8
        group = (item % 8).to(tl.int32)
        if tl.load(Work + N * 512 + token) != 0.0:
            rotary_pos = tl.load(Position + token * POS_STRIDE) - 3
            dst = tl.load(KvSlot + token * KV_STRIDE)
            work = Work + token * 512
            row = tl.load(work + tl.arange(0, 512).to(tl.int64))
            rrms = tl.rsqrt(tl.sum(row * row, 0) * (1.0 / 512.0) + eps)
            dst_row = dst % PAGE_SIZE
            page = Cache + (dst // PAGE_SIZE) * PAGE_BYTES
            payload = page + dst_row * ROW_BYTES
            scale_row = page + PAGE_SIZE * ROW_BYTES + dst_row * SCALE_DIM
            if group == 7:
                pair = tl.arange(0, 32)
                even_col = (448 + pair * 2).to(tl.int64)
                odd_col = even_col + 1
                even = tl.load(work + even_col) * rrms
                odd = tl.load(work + odd_col) * rrms
                even *= tl.load(Weight + even_col * W_STRIDE).to(tl.float32)
                odd *= tl.load(Weight + odd_col * W_STRIDE).to(tl.float32)
                cs = Rotary + rotary_pos * ROTARY_ROW + pair.to(tl.int64) * ROTARY_COL
                cosine = tl.load(cs)
                sine = tl.load(cs + 32 * ROTARY_COL)
                real = (even * cosine - odd * sine).to(tl.bfloat16)
                imag = (even * sine + odd * cosine).to(tl.bfloat16)
                tail = payload + 448
                tl.store(scale_row + 7, 0)
                _write_bf16(tail, pair * 2, real)
                _write_bf16(tail, pair * 2 + 1, imag)
            else:
                col = group * 64 + tl.arange(0, 64)
                x = tl.load(work + col.to(tl.int64)) * rrms
                x *= tl.load(Weight + col.to(tl.int64) * W_STRIDE).to(tl.float32)
                rounded = x.to(tl.bfloat16)
                group_f32 = rounded.to(tl.float32)
                raw_scale = tl.maximum(tl.max(tl.abs(group_f32), 0) / FP8_MAX, 1e-4)
                bits = raw_scale.to(tl.int32, bitcast=True)
                code = ((bits + 0x7FFFFF) >> 23) & 255
                inv = ((254 - code) << 23).to(tl.float32, bitcast=True)
                scaled = group_f32 * inv
                tl.store(scale_row + group, code.to(tl.uint8))
                tl.store(payload + col, _encode_group(scaled, FP8_MAX))


def _launch_packed(
    state_cache,
    token_to_req_indices,
    positions,
    slot_mapping,
    block_table,
    block_size,
    rms_norm_weight,
    rms_norm_eps,
    cos_sin_cache,
    k_cache,
    kv_slot_mapping,
    kv_cache_block_size,
    n,
    page_bytes,
    row_bytes,
    scale_dim,
    fp8_max,
):
    # The fused packed epilogue exceeds 910B1's 192 KiB unified buffer.
    # The first 512*n floats hold complete rows. The trailing n flags carry
    # metadata and physical-page validity across the two kernels.
    device = state_cache.device
    grid = min(n * 8, 65535)
    with runtime.torch_device_fn.device(device):
        work = torch.empty((n * 513,), dtype=torch.float32, device=device)
        _compress_tiles[(grid,)](
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            kv_slot_mapping,
            work,
            n,
            block_size,
            token_to_req_indices.stride(0),
            positions.stride(0),
            slot_mapping.stride(0),
            kv_slot_mapping.stride(0),
            *block_table.shape,
            *block_table.stride(),
            state_cache.shape[0],
            *state_cache.stride(),
            cos_sin_cache.shape[0],
            k_cache.shape[0],
            kv_cache_block_size,
            64,
            num_warps=4,
            num_stages=1,
        )
        _store_packed_groups[(grid,)](
            work,
            positions,
            rms_norm_weight,
            cos_sin_cache,
            k_cache.view(torch.uint8),
            kv_slot_mapping,
            rms_norm_eps,
            n,
            positions.stride(0),
            rms_norm_weight.stride(0),
            *cos_sin_cache.stride(),
            kv_slot_mapping.stride(0),
            kv_cache_block_size,
            page_bytes,
            row_bytes,
            scale_dim,
            fp8_max,
            num_warps=4,
            num_stages=1,
        )


def fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    kv_block_stride: int,
    head_size: int = 512,
    state_width: int = 1024,
    rope_head_dim: int = 64,
    fp8_max: float = 448.0,
    quant_block: int = 64,
    token_stride: int = 576,
    scale_dim: int = 8,
    compress_ratio: int = 4,
    overlap: bool = True,
    store_full_kv: bool = False,
    store_full_fp8: bool = False,
    fp8_scale: torch.Tensor | None = None,
) -> None:
    """Use tiled packed insertion and the fused full-cache kernel on Ascend."""
    logger.debug("GEMS_ASCEND FUSED_KV_COMPRESS_NORM_ROPE_INSERT_SPARSE_ATTN_CUTEDSL")
    from flaggems_vllm.ops.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl import (
        _dispatch_c4,
    )

    _dispatch_c4(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        k_cache,
        kv_slot_mapping,
        kv_cache_block_size,
        head_size,
        state_width,
        rope_head_dim,
        fp8_max,
        quant_block,
        token_stride,
        scale_dim,
        compress_ratio,
        overlap,
        store_full_kv,
        store_full_fp8,
        fp8_scale,
        packed_launcher=_launch_packed,
    )


__all__ = ["fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl"]

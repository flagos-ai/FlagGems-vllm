# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from __future__ import annotations

import logging
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime

logger = logging.getLogger(__name__)
_ASCEND = runtime.device.vendor_name == "ascend"

if _ASCEND:
    from flaggems_vllm.runtime.backend._ascend.ops.per_token_group_quant_fp8 import (
        _f32_to_fp8_e4m3fn as _encode,
    )
else:
    from flaggems_vllm.ops.fp8_storage import encode_e4m3fn as _encode


@triton.jit
def _compress_window(
    State,
    table,
    position,
    col,
    BLOCK_SIZE: tl.constexpr,
    TABLE_COL: tl.constexpr,
    STATE_PAGES: tl.constexpr,
    STATE_PAGE: tl.constexpr,
    STATE_ROW: tl.constexpr,
    STATE_COL: tl.constexpr,
):
    # Each SIMD lane keeps just the running maximum, mass and weighted value.
    maximum = tl.full((512,), -float("inf"), tl.float32)
    mass = tl.full((512,), 0.0, tl.float32)
    weighted = tl.full((512,), 0.0, tl.float32)
    valid_window = tl.full((), True, tl.int1)
    for bank in tl.static_range(2):
        first = position - 7 + bank * 4
        if BLOCK_SIZE % 4 == 0:
            # Four aligned positions share their physical page.
            page = tl.load(table + (first // BLOCK_SIZE) * TABLE_COL, first >= 0, 0)
        for row in tl.static_range(4):
            loc = first + row
            if BLOCK_SIZE % 4 != 0:
                page = tl.load(table + (loc // BLOCK_SIZE) * TABLE_COL, loc >= 0, 0)
            present = loc >= 0
            valid_page = (page >= 0) & (page < STATE_PAGES)
            valid_window &= ~present | valid_page
            offset = page.to(tl.int64) * STATE_PAGE + (loc % BLOCK_SIZE) * STATE_ROW
            ptr = State + offset + (bank * 512 + col) * STATE_COL
            load = present & valid_page
            score = tl.load(ptr + 1024 * STATE_COL, load, 0.0)
            value = tl.load(ptr, load, 0.0)
            peak = tl.maximum(maximum, score)
            old_scale = tl.exp(maximum - peak)
            new_scale = tl.exp(score - peak)
            mass = tl.where(load, mass * old_scale + new_scale, mass)
            weighted = tl.where(
                load, weighted * old_scale + value * new_scale, weighted
            )
            maximum = tl.where(load, peak, maximum)
    return weighted / mass, valid_window


@triton.jit
def _fp8_bytes(x, LIMIT: tl.constexpr):
    # SATFINITE E4M3FN also saturates when a caller supplies a limit above 448.
    if LIMIT < 448.0:
        x = tl.minimum(tl.maximum(x, 0.0 - LIMIT), LIMIT)
    else:
        x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    return _encode(x)


@triton.jit
def _finish_row(
    x,
    Weight,
    rotary,
    page,
    row,
    Scale,
    eps,
    WEIGHT_COL: tl.constexpr,
    ROTARY_COL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    FULL: tl.constexpr,
    FP8: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    col = tl.arange(0, 512)
    rrms = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 512.0) + eps)
    x = x * rrms * tl.load(Weight + col * WEIGHT_COL).to(tl.float32)

    even, odd = tl.split(tl.reshape(x, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cosine = tl.load(rotary + pair * ROTARY_COL, pair >= 0, 1.0)
    sine = tl.load(rotary + (pair + 32) * ROTARY_COL, pair >= 0, 0.0)
    real = tl.where(pair >= 0, even * cosine - odd * sine, even)
    imag = tl.where(pair >= 0, even * sine + odd * cosine, odd)
    rounded = tl.interleave(real, imag).to(tl.bfloat16).to(tl.float32)
    payload = page + row * ROW_BYTES
    if FULL and FP8:
        scaled = rounded
        if HAS_SCALE:
            scaled = rounded * (1.0 / tl.load(Scale))
        tl.store(payload + col, _fp8_bytes(scaled, FP8_MAX))
    else:
        # Write the two bytes of each BF16 without imposing extra alignment on
        # a byte view. Packed rows contain BF16 only in the rotated tail.
        bits = rounded.to(tl.int32, bitcast=True)
        if FULL:
            bf16 = payload + 2 * col
            tl.store(bf16, ((bits >> 16) & 255).to(tl.uint8))
            tl.store(bf16 + 1, ((bits >> 24) & 255).to(tl.uint8))
        else:
            bf16 = payload + 448 + (col - 448) * 2
            tl.store(bf16, ((bits >> 16) & 255).to(tl.uint8), col >= 448)
            tl.store(bf16 + 1, ((bits >> 24) & 255).to(tl.uint8), col >= 448)
            groups = tl.reshape(rounded, (8, 64))
            amax = tl.max(tl.abs(groups), 1)
            raw_scale = tl.maximum(amax / FP8_MAX, 1.0e-4)
            scale_bits = raw_scale.to(tl.int32, bitcast=True)
            exponent = ((scale_bits + 0x7FFFFF) >> 23) & 255
            inv = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
            scaled = tl.reshape(groups * inv[:, None], (512,))
            tl.store(payload + col, _fp8_bytes(scaled, FP8_MAX), col < 448)
            group = tl.arange(0, 8)
            scales = page + PAGE_SIZE * ROW_BYTES + row * SCALE_DIM
            tl.store(scales + group, tl.where(group < 7, exponent, 0).to(tl.uint8))


@triton.jit
def _fused_c4_kernel(
    State,
    Request,
    Position,
    Slot,
    Table,
    Weight,
    Rotary,
    Cache,
    Destination,
    Scale,
    eps,
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
    WEIGHT_COL: tl.constexpr,
    ROTARY_LEN: tl.constexpr,
    ROTARY_ROW: tl.constexpr,
    ROTARY_COL: tl.constexpr,
    KV_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    FULL: tl.constexpr,
    FP8: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    for token in range(tl.program_id(0).to(tl.int64), N, tl.num_programs(0)):
        slot = tl.load(Slot + token * SLOT_STRIDE)
        position = tl.load(Position + token * POSITION_STRIDE)
        dst = tl.load(Destination + token * DEST_STRIDE)
        req = tl.load(Request + token * REQUEST_STRIDE).to(tl.int64)
        rotary_pos = position - 3
        active = (slot >= 0) & (position >= 3) & ((position & 3) == 3)
        active &= (dst >= 0) & (dst < KV_PAGES * PAGE_SIZE)
        active &= (req >= 0) & (req < N_REQ) & (position // BLOCK_SIZE < TABLE_WIDTH)
        if active & (rotary_pos < ROTARY_LEN):
            col = tl.arange(0, 512).to(tl.int64)
            x, valid = _compress_window(
                State,
                Table + req * TABLE_ROW,
                position,
                col,
                BLOCK_SIZE,
                TABLE_COL,
                STATE_PAGES,
                STATE_PAGE,
                STATE_ROW,
                STATE_COL,
            )
            if valid:
                page = Cache + (dst // PAGE_SIZE) * PAGE_BYTES
                _finish_row(
                    x,
                    Weight,
                    Rotary + rotary_pos * ROTARY_ROW,
                    page,
                    dst % PAGE_SIZE,
                    Scale,
                    eps,
                    WEIGHT_COL,
                    ROTARY_COL,
                    PAGE_SIZE,
                    ROW_BYTES,
                    SCALE_DIM,
                    FULL,
                    FP8,
                    HAS_SCALE,
                    FP8_MAX,
                )


_fused_c4 = _fused_c4_kernel


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
    """Compress overlapping C4 windows and insert the boundary rows into k_cache.

    State rows hold two 512-element value banks followed by their logit banks.
    Packed pages have a separate scale plane. Full pages use the supplied
    tensor's byte strides. The upstream ABI retains kv_block_stride in both modes.
    """
    logger.debug("GEMS FUSED_KV_COMPRESS_NORM_ROPE_INSERT_SPARSE_ATTN_CUTEDSL")
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
    )


def _dispatch_c4(
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
    *,
    packed_launcher=None,
    tuned_kernel=None,
):
    if positions.numel() == 0:
        return
    has_scale = store_full_fp8 and fp8_scale is not None
    if (compress_ratio, overlap, head_size, state_width) != (4, True, 512, 1024):
        raise ValueError(
            "C4 requires compress_ratio=4, overlap=True, head_size=512 and state_width=1024"
        )
    if (quant_block, rope_head_dim) != (64, 64):
        raise ValueError("C4 requires quant_block=rope_head_dim=64")
    if store_full_fp8 and not store_full_kv:
        raise ValueError("store_full_fp8 requires store_full_kv")
    if (
        state_cache.dtype != torch.float32
        or state_cache.ndim != 3
        or state_cache.shape[2] != 2048
    ):
        raise ValueError("state_cache must be FP32 [pages, rows, 2048]")
    if block_size <= 0 or block_size != state_cache.shape[1]:
        raise ValueError("block_size must match the positive state page size")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be an INT32 matrix")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.dtype != torch.float32
    ):
        raise ValueError("cos_sin_cache must be FP32 [positions, 64]")
    if rms_norm_weight.shape != (512,) or rms_norm_weight.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("rms_norm_weight must be a BF16 or FP32 vector of length 512")
    mappings = (positions, slot_mapping, kv_slot_mapping)
    if any(t.ndim != 1 or t.dtype != torch.int64 for t in mappings):
        raise ValueError(
            "positions, slot_mapping and kv_slot_mapping must be INT64 vectors"
        )
    if token_to_req_indices.ndim != 1 or token_to_req_indices.dtype != torch.int32:
        raise ValueError("token_to_req_indices must be an INT32 vector")
    if (
        k_cache.ndim != 3
        or k_cache.stride(2) != 1
        or kv_cache_block_size != k_cache.shape[1]
        or kv_cache_block_size <= 0
    ):
        raise ValueError(
            "k_cache must be 3D with a contiguous inner dimension and a matching positive kv_cache_block_size"
        )

    if store_full_kv:
        logical_dtype = (
            getattr(torch, "float8_e4m3fn", None) if store_full_fp8 else torch.bfloat16
        )
        if k_cache.dtype not in (logical_dtype, torch.uint8):
            raise ValueError(
                "full k_cache must have the selected BF16/E4M3FN dtype or its uint8 byte view"
            )
    elif k_cache.dtype != torch.uint8:
        raise ValueError("packed k_cache must be uint8")
    if (not store_full_kv or store_full_fp8) and (
        not math.isfinite(fp8_max) or fp8_max <= 0
    ):
        raise ValueError("fp8_max must be finite and positive")

    item_size = k_cache.element_size()
    page_bytes = k_cache.stride(0) * item_size
    row_bytes = k_cache.stride(1) * item_size
    width_bytes = k_cache.shape[2] * item_size
    page_span = (k_cache.shape[1] - 1) * row_bytes + width_bytes
    if store_full_kv:
        payload = 512 if store_full_fp8 else 1024
        if width_bytes < payload or (kv_cache_block_size > 1 and row_bytes < payload):
            raise ValueError(
                "full-cache rows must hold the complete 512-element payload"
            )
        required = (kv_cache_block_size - 1) * row_bytes + payload
    else:
        if token_stride < 576 or scale_dim < 8:
            raise ValueError(
                "packed C4 requires token_stride >= 576 and scale_dim >= 8"
            )
        row_bytes = token_stride
        required = (
            kv_cache_block_size * token_stride
            + (kv_cache_block_size - 1) * scale_dim
            + 8
        )
    if required > page_span or (k_cache.shape[0] > 1 and required > page_bytes):
        raise ValueError("cache page is too small for the selected layout")

    device = state_cache.device
    inputs = (
        *mappings,
        token_to_req_indices,
        block_table,
        rms_norm_weight,
        cos_sin_cache,
        k_cache,
    )
    if any(t.device != device for t in inputs):
        raise ValueError("all C4 tensors must be on the same device")
    if has_scale and (
        fp8_scale.device != device
        or fp8_scale.dtype != torch.float32
        or fp8_scale.numel() != 1
    ):
        raise ValueError("fp8_scale must contain one FP32 value on the cache device")

    n = min(
        positions.numel(),
        slot_mapping.numel(),
        token_to_req_indices.numel(),
        kv_slot_mapping.numel(),
    )
    if (
        not n
        or not block_table.numel()
        or not state_cache.shape[0]
        or not k_cache.shape[0]
        or not cos_sin_cache.shape[0]
    ):
        return
    if packed_launcher is not None and not store_full_kv:
        packed_launcher(
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
        )
        return

    grid = min(n, 65535) if _ASCEND else n
    kernel = tuned_kernel if tuned_kernel is not None else _fused_c4
    opts = {} if tuned_kernel is not None else {"num_warps": 4, "num_stages": 1}
    kernel_args = (
        rms_norm_eps,
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
        rms_norm_weight.stride(0),
        cos_sin_cache.shape[0],
        *cos_sin_cache.stride(),
        k_cache.shape[0],
        kv_cache_block_size,
        page_bytes,
        row_bytes,
        scale_dim if not store_full_kv else 0,
        store_full_kv,
        store_full_fp8,
        has_scale,
        fp8_max,
    )
    with runtime.torch_device_fn.device(device):
        launch = kernel[(grid,)](
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            rms_norm_weight,
            cos_sin_cache,
            k_cache.view(torch.uint8),
            kv_slot_mapping,
            fp8_scale if has_scale else state_cache,
            *kernel_args,
            **opts,
        )
    if tuned_kernel is not None:
        return launch, grid, kernel_args


__all__ = ["fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl"]

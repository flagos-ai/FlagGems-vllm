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
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _rotate_tail(src, dst, valid, cos, sin, scale, NORM: tl.constexpr):
    pairs = tl.arange(0, 32)
    offs = 448 + 2 * pairs
    even = tl.load(src + offs[None, :], valid, other=0).to(tl.float32)
    odd = tl.load(src + offs[None, :] + 1, valid, other=0).to(tl.float32)
    if NORM:
        even = even * scale
        odd = odd * scale
    # Match the fused multiply-add order of the FP32 reference.
    out_even = tl.fma(even, cos, -(odd * sin))
    out_odd = tl.fma(even, sin, odd * cos)
    tl.store(dst + offs[None, :], out_even, valid)
    tl.store(dst + offs[None, :] + 1, out_odd, valid)


@triton.jit
def _query_tile(tile, q, positions, table, eps, H: tl.constexpr, NORM: tl.constexpr):
    # One group of heads from the same token shares a single RoPE table row.
    groups: tl.constexpr = triton.cdiv(H, 8)
    token = tile // groups
    head = (tile % groups) * 8 + tl.arange(0, 8)
    row = q + ((token * H + head) * 512)[:, None]
    valid = (head < H)[:, None]
    scale = 1.0
    if NORM:
        col = tl.arange(0, 512)[None, :]
        x = tl.load(row + col, valid, other=0).to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / 512) + eps)[:, None]
        tl.store(row + col, x * scale, valid & (col < 448))
    pos = tl.load(positions + token)
    freq = table + pos * 64 + tl.arange(0, 32)
    _rotate_tail(row, row, valid, tl.load(freq), tl.load(freq + 32), scale, NORM)


@triton.jit
def _copy_kv_prefix(
    tile,
    kv,
    cache,
    slots,
    positions,
    table,
    insert_count,
    page_stride,
    row_stride,
    S: tl.constexpr,
    ROWS: tl.constexpr = 8,
):
    token = tile * ROWS + tl.arange(0, ROWS)
    slot = tl.load(slots + token, token < insert_count, other=-1)
    active = slot >= 0
    safe_slot = tl.where(active, slot, 0)
    dst = (
        cache + ((safe_slot // S) * page_stride + (safe_slot % S) * row_stride)[:, None]
    )
    src = kv + (token * 512)[:, None]
    col = tl.arange(0, 512)[None, :]
    prefix = active[:, None] & (col < 448)
    tl.store(dst + col, tl.load(src + col, prefix, other=0), prefix)
    pos = tl.load(positions + token, active, other=0)
    freq = table + (pos * 64)[:, None] + tl.arange(0, 32)[None, :]
    cos = tl.load(freq, active[:, None], other=0)
    sin = tl.load(freq + 32, active[:, None], other=0)
    return src, dst, active, cos, sin


@triton.jit
def _insert_tile(
    tile,
    kv,
    cache,
    slots,
    positions,
    table,
    insert_count,
    page_stride,
    row_stride,
    S: tl.constexpr,
):
    src, dst, active, cos, sin = _copy_kv_prefix(
        tile,
        kv,
        cache,
        slots,
        positions,
        table,
        insert_count,
        page_stride,
        row_stride,
        S,
    )
    _rotate_tail(src, dst, active[:, None], cos, sin, 1.0, False)


@libentry()
@triton.jit
def _full_cache_kernel(
    q,
    kv,
    cache,
    slots,
    positions,
    table,
    eps,
    query_tiles,
    insert_count,
    page_stride,
    row_stride,
    H: tl.constexpr,
    S: tl.constexpr,
    NORM: tl.constexpr,
):
    # Widen before deriving row, token and byte addresses, including on sparse pages.
    tile = tl.program_id(0).to(tl.int64)
    if H > 0:
        if tile < query_tiles:
            _query_tile(tile, q, positions, table, eps, H, NORM)
            return
    _insert_tile(
        tile - query_tiles,
        kv,
        cache,
        slots,
        positions,
        table,
        insert_count,
        page_stride,
        row_stride,
        S,
    )


@libentry()
@triton.jit
def _large_qnorm_full_cache_kernel(
    q,
    kv,
    cache,
    slots,
    positions,
    table,
    eps,
    q_rows,
    insert_count,
    page_stride,
    row_stride,
    H: tl.constexpr,
    S: tl.constexpr,
    WIDE_ROWS: tl.constexpr,
):
    """Eight consecutive query heads per CTA, including tiles spanning two tokens."""
    tile = tl.program_id(0)
    if WIDE_ROWS:
        tile = tile.to(tl.int64)
        q_rows = q_rows.to(tl.int64)
    q_tiles = tl.cdiv(q_rows, 8)
    if tile < q_tiles:
        row_id = tile * 8 + tl.arange(0, 8)
        valid = row_id < q_rows
        row = q + (row_id.to(tl.int64) * 512)[:, None]
        pos = tl.load(positions + row_id // H, valid, other=0)
        col = tl.arange(0, 512)[None, :]
        x = tl.load(row + col, valid[:, None], other=0).to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / 512) + eps)[:, None]
        x = x * scale
        pair = tl.arange(0, 256)[None, :]
        tail = pair >= 224
        coeff = table + (pos * 64)[:, None] + (pair - 224)
        cos = tl.load(coeff, valid[:, None] & tail, other=0.0)
        sin = tl.load(coeff + 32, valid[:, None] & tail, other=0.0)
        even, odd = tl.split(tl.reshape(x, (8, 256, 2)))
        y_even = tl.where(tail, tl.fma(even, cos, -(odd * sin)), even)
        y_odd = tl.where(tail, tl.fma(even, sin, odd * cos), odd)
        y = tl.reshape(tl.join(y_even, y_odd), (8, 512))
        tl.store(row + col, y, valid[:, None])
        return
    _insert_tile(
        (tile - q_tiles).to(tl.int64),
        kv,
        cache,
        slots,
        positions,
        table,
        insert_count,
        page_stride,
        row_stride,
        S,
    )


def _check_args(q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, block_size):
    tensors = (q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError(
            "q, kv, k_cache, slot_mapping, position_ids and cos_sin_cache must be tensors"
        )
    if any(x.device != q.device for x in tensors[1:]):
        raise ValueError("all tensors must be on the same device")
    if not q.dtype == kv.dtype == k_cache.dtype == torch.bfloat16:
        raise TypeError("q, kv and k_cache must be bfloat16")
    if not slot_mapping.dtype == position_ids.dtype == torch.int64:
        raise TypeError("slot_mapping and position_ids must be int64")
    if cos_sin_cache.dtype != torch.float32:
        raise TypeError("cos_sin_cache must be float32")
    if q.ndim != 3 or q.shape[2] != 512:
        raise ValueError("q must have shape [N, H, 512]")
    n = q.shape[0]
    if kv.shape != (n, 512) or position_ids.shape != (n,):
        raise ValueError("expected kv [N, 512] and position_ids [N]")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] > n:
        raise ValueError("slot_mapping must have shape [I] with I <= N")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 64:
        raise ValueError("cos_sin_cache must have shape [max_pos, 64]")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("cache_block_size must be a positive integer")
    if k_cache.ndim != 3 or k_cache.shape[1:] != (block_size, 512):
        raise ValueError("k_cache must have shape [P, cache_block_size, 512]")
    if k_cache.stride(-1) != 1:
        raise ValueError("k_cache must have unit last stride")
    if not all(
        x.is_contiguous() for x in (q, kv, slot_mapping, position_ids, cos_sin_cache)
    ):
        raise ValueError(
            "q, kv, slot_mapping, position_ids and cos_sin_cache must be contiguous"
        )
    if q.device.type == "cpu":
        raise NotImplementedError("an accelerator device is required")


def _run(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps,
    cache_block_size,
    apply_q_norm,
    kernel,
    rows_per_insert_tile=8,
):
    _check_args(
        q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, cache_block_size
    )
    n, h, _ = q.shape
    insert_count = slot_mapping.numel()
    query_tiles = n * triton.cdiv(h, 8)
    grid = (query_tiles + triton.cdiv(insert_count, rows_per_insert_tile),)
    if grid[0] == 0:
        return
    with torch_device_fn.device(q.device):
        kernel[grid](
            q,
            kv,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            float(eps),
            query_tiles,
            insert_count,
            k_cache.stride(0),
            k_cache.stride(1),
            h,
            cache_block_size,
            bool(apply_q_norm),
        )


def fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
) -> None:
    """Update all Q heads in place and insert the selected KV rows into a paged cache."""
    logger.debug("GEMS FUSED_DEEPSEEK_V4_QNORM_ROPE_KV_ROPE_FULL_CACHE_BF16_INSERT")
    _run(
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        cache_block_size,
        apply_q_norm,
        _full_cache_kernel,
    )

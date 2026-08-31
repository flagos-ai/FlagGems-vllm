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

"""Iluvatar-optimized Qwen4 QSA K/V cache store.

[KernelGen] Auto-generated and tuned for Iluvatar BI/MR series.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
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


def qwen4_store_qsa_kv_rows(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if any(
        t.device.type in ("cpu", "meta")
        for t in (k_cache, v_cache, slot_mapping, key, value)
    ):
        raise RuntimeError("Qwen4 QSA K/V store requires a Triton accelerator")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("Qwen4 QSA K/V caches must be [blocks, page, heads, dim]")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("Qwen4 QSA K/V rows must be [rows, heads, dim]")
    if key.shape != (slot_mapping.numel(), k_cache.shape[2], k_cache.shape[3]):
        raise ValueError("Qwen4 QSA K/V rows and slot_mapping shapes are incompatible")
    if k_cache.dtype != key.dtype or v_cache.dtype != value.dtype:
        raise TypeError("Qwen4 QSA K/V caches and rows must have matching dtypes")
    if not all(k_cache.shape):
        raise ValueError("Qwen4 QSA K/V caches must be nonempty")
    if not key.shape[0]:
        return

    rows, heads, dim = key.shape
    _store_qsa_kv_rows_kernel[(rows, heads)](
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
        rows,
        k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=heads,
        HEAD_DIM=dim,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=1,
    )

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Stride-aware writes into unquantized paged K/V views on Sunrise."""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _write_kv_cache(
    key,
    value,
    key_cache,
    value_cache,
    slots,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    kc0: tl.constexpr,
    kc1: tl.constexpr,
    kc2: tl.constexpr,
    kc3: tl.constexpr,
    vc0: tl.constexpr,
    vc1: tl.constexpr,
    vc2: tl.constexpr,
    vc3: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, WIDTH)
    slot = tl.load(slots + token)
    mask = (slot >= 0) & (dim < HEAD_SIZE)
    block = slot // BLOCK_SIZE
    offset = slot % BLOCK_SIZE
    k = tl.load(key + token * ks0 + head * ks1 + dim * ks2, mask=mask, other=0)
    v = tl.load(value + token * vs0 + head * vs1 + dim * vs2, mask=mask, other=0)
    tl.store(key_cache + block * kc0 + offset * kc1 + head * kc2 + dim * kc3, k, mask)
    tl.store(value_cache + block * vc0 + offset * vc1 + head * vc2 + dim * vc3, v, mask)


def reshape_and_cache_flash_strided(
    key, value, key_cache, value_cache, slot_mapping
) -> None:
    """Write [tokens, heads, dim] inputs to [blocks, slots, heads, dim] views.

    This PTPU operator supports independent source and cache strides, including
    combined K/V storage. The current device must own the tensors. Each
    nonnegative slot must be unique and in range, and its output elements must
    not overlap any other destination or input. Negative slots are padding.
    Only the token rows covered by the contiguous slot mapping are read.
    """
    if key.device.type != "ptpu":
        raise NotImplementedError("Strided Flash KV writes require Sunrise PTPU")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("Key and value must have equal [tokens, heads, dim] shapes")
    if (
        key_cache.ndim != 4
        or key_cache.shape != value_cache.shape
        or key.shape[1:] != key_cache.shape[2:]
    ):
        raise ValueError("K/V cache shapes must match the input head dimensions")
    if key.dtype not in _FLOAT_DTYPES or any(
        tensor.dtype != key.dtype for tensor in (value, key_cache, value_cache)
    ):
        raise ValueError("K/V inputs and caches must share an unquantized float dtype")
    if any(
        tensor.device != key.device
        for tensor in (value, key_cache, value_cache, slot_mapping)
    ):
        raise ValueError("K/V inputs, caches and slots must be on the same device")
    if (
        slot_mapping.ndim != 1
        or not slot_mapping.is_contiguous()
        or slot_mapping.dtype not in (torch.int32, torch.int64)
        or slot_mapping.numel() > key.shape[0]
    ):
        raise ValueError(
            "Slot mapping must be a contiguous integer vector of token slots"
        )
    if slot_mapping.numel() == 0:
        return

    logger.debug("GEMS RESHAPE_AND_CACHE_FLASH_STRIDED")
    # Retain the fixed PTPU launch used by the native graph path.
    _write_kv_cache[(slot_mapping.numel(), key.shape[1])](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        *key.stride(),
        *value.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        HEAD_SIZE=key.shape[2],
        BLOCK_SIZE=key_cache.shape[1],
        WIDTH=triton.next_power_of_2(key.shape[2]),
        num_warps=4,
    )

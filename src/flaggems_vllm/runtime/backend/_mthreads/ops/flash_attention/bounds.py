# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fresh paged-key norm bounds for the BF16 grouped Short-Q Split-KV path."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import launch
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.kv_cache import (
    cache_extent,
)


@triton.jit
def norm_keys(
    K,
    TABLE,
    USED,
    NORM,
    HK: tl.constexpr,
    NT: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
):
    (tile, bh) = (tl.program_id(0), tl.program_id(1))
    (b, h) = (bh // HK, bh % HK)
    n = tile * 32 + tl.arange(0, 32)
    d = tl.arange(0, 256)
    length = tl.load(USED + b)
    page = tl.load(TABLE + b * TS + n // PAGE, n < length, 0)
    key = tl.load(
        K
        + page[:, None].to(tl.int64) * KP
        + n[:, None] % PAGE * KS
        + h * KH
        + d[None, :],
        n[:, None] < length,
        0,
    ).to(tl.float32)
    n2 = tl.max(tl.sum(key * key, 1), 0)
    tl.store(NORM + bh * NT + tile, n2)


def prepare_key_norms(k, table, used, maxk=None):
    tiles = triton.cdiv(cache_extent(k, table, maxk, 32), 32)
    heads = table.shape[0] * k.shape[2]
    norms = torch.empty((heads, tiles), device=k.device, dtype=torch.float32)
    launch(
        norm_keys,
        ((tiles, heads)),
        k,
        table,
        used,
        norms,
        k.shape[2],
        tiles,
        *k.stride()[:3],
        table.stride(0),
        k.shape[1],
        num_warps=4,
        num_stages=1,
    )
    return norms, tiles

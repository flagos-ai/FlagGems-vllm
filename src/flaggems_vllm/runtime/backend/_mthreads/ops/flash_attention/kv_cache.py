# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared strided KV staging, bounded by the caller's host length maximum."""
import triton
import triton.language as tl


def cache_extent(k, table, maxk=None, tile=64):
    capacity = table.shape[1] * k.shape[1]
    upper = capacity if maxk is None else min(capacity, maxk)
    return max(tile, triton.cdiv(upper, tile) * tile) if capacity else 0


@triton.jit
def gather_cache_kernel(
    K,
    V,
    KC,
    VC,
    TABLE,
    USEDK,
    HK: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    TS: tl.constexpr,
    NK: tl.constexpr,
    PERMUTE: tl.constexpr = False,
    KVTILE: tl.constexpr = 64,
    BN: tl.constexpr = 32,
    BD: tl.constexpr = 64,
    NORM=None,
):
    block, bh, db = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = bh // HK, bh % HK
    length = tl.load(USEDK + batch)
    if NORM is not None:
        tl.static_assert(BD == D)
        # Classifiers read all norm slots, including those beyond seqused_k.
        tl.store(NORM + bh * tl.cdiv(NK, BN) + block, 0.0)
    # The final MMA tile must contain finite zeros in all masked K/V lanes:
    # a zero probability multiplied by uninitialized NaN V still produces NaN.
    if block * BN < tl.cdiv(length, KVTILE) * KVTILE:
        cols = block * BN + tl.arange(0, BN)
        vcols = cols
        if PERMUTE:
            vcols = (
                cols // KVTILE * KVTILE + cols % 8 * (KVTILE // 8) + cols % KVTILE // 8
            )
        dims = db * BD + tl.arange(0, BD)
        page = tl.load(TABLE + batch * TS + cols // PAGE, cols < length, 0)
        ko = page.to(tl.int64) * KP + cols % PAGE * KS + head * KH
        vo = page.to(tl.int64) * VP + cols % PAGE * VS + head * VH
        mask = (cols[:, None] < length) & (dims[None, :] < D)
        key = tl.load(K + ko[:, None] + dims[None, :], mask, 0)
        value = tl.load(V + vo[:, None] + dims[None, :], mask, 0)
        if NORM is not None:
            fp32 = key.to(tl.float32)
            norm = tl.max(tl.sum(fp32 * fp32, 1), 0)
            tl.store(NORM + bh * tl.cdiv(NK, BN) + block, norm)
        output_mask = (cols[:, None] < NK) & (dims[None, :] < D)
        tl.store(
            KC + (bh * D + dims[None, :]).to(tl.int64) * NK + cols[:, None],
            key,
            output_mask,
        )
        tl.store(
            VC + (bh * NK + vcols[:, None]).to(tl.int64) * D + dims[None, :],
            value,
            output_mask,
        )


# Existing imports retain their names while using one implementation.
gather_norm_kernel = gather_cache_kernel

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Triton KV compaction and a descriptor-based S5000 prefill path."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.capabilities import (
    supports_tle,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.direct import (
    _BACKEND_OPT,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
    current_config,
    launch,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.kv_cache import (
    cache_extent,
    gather_cache_kernel,
)

try:
    from triton.tools.experimental_descriptor import create_2d_tma_descriptor
except ImportError:
    create_2d_tma_descriptor = None

# TLE is optional so the original Triton 3.2 baseline environment can import us.
try:
    if not supports_tle():
        raise ImportError("MThreads TLE primitives are unavailable")
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.tle_prefill import (
        launch_prefill,
    )
except ImportError:
    launch_prefill = None

# Bound K/V workspace without falling off the cached route just beyond 64K.
# The S5000 full-shape workload needs up to 520 MiB for the compacted K/V.
_MAX_CACHE_BYTES = 1024 * 1024 * 1024


def supports_cached(q, k, v, table, maxq, window, softcap, alibi, plan, maxk=None):
    if (create_2d_tma_descriptor is None and launch_prefill is None) or table is None:
        return False
    if q.shape[-1] != 256 or q.shape[0] < 512 or maxq <= 1:
        return False
    if window != (-1, -1) or softcap or alibi is not None or plan.num_splits != 1:
        return False
    # Only Q is read directly through a contiguous TensorDescriptor. The
    # gather kernels honor independent K/V page, token and head strides before
    # building descriptors over fresh contiguous workspaces.
    if not q.is_contiguous():
        return False
    nk = cache_extent(k, table, maxk)
    return 0 < nk and table.shape[0] * k.shape[2] * 256 * nk * 4 <= _MAX_CACHE_BYTES


@triton.jit
def cached_attention_kernel(
    QDESC,
    KDESC,
    VDESC,
    OUT,
    LSE,
    CUQ,
    USEDK,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    NK: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 64,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    klen = tl.load(USEDK + batch)
    row_base = tile * BM
    if row_base >= qlen:
        return
    rows = row_base + tl.arange(0, BM)
    dims = tl.arange(0, 256)
    kh = head // (HQ // HK)
    q = tl._experimental_descriptor_load(
        QDESC, [qstart + row_base, head * 256], [BM, 256], OUT.dtype.element_ty
    )
    high = klen
    if CAUSAL:
        high = tl.minimum(high, tl.minimum(row_base + BM, qlen) + klen - qlen)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 256), tl.float32)
    for start in range(0, high, BN):
        key = tl._experimental_descriptor_load(
            KDESC, [(batch * HK + kh) * 256, start], [256, BN], OUT.dtype.element_ty
        )
        scores = tl.dot(q, key) * (SCALE * 1.4426950408889634)
        fully_visible = start + BN <= klen
        if CAUSAL:
            fully_visible &= start + BN <= row_base + klen - qlen + 1
        if not fully_visible:
            cols = start + tl.arange(0, BN)
            mask = (rows[:, None] < qlen) & (cols[None, :] < klen)
            if CAUSAL:
                mask &= cols[None, :] <= rows[:, None] + klen - qlen
            scores = tl.where(mask, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        probs = tl.exp2(scores - safe_maximum[:, None])
        denominator = denominator * alpha + tl.sum(probs, 1)
        acc *= alpha[:, None]
        value = tl._experimental_descriptor_load(
            VDESC, [(batch * HK + kh) * NK + start, 0], [BN, 256], OUT.dtype.element_ty
        )
        acc = tl.dot(probs.to(value.dtype), value, acc)
        maximum = next_maximum
    result = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    lse = tl.where(
        denominator > 0,
        (maximum + tl.log2(denominator)) * 0.6931471805599453,
        float("inf"),
    )
    tl.store(
        OUT + ((qstart + rows[:, None]).to(tl.int64) * HQ + head) * 256 + dims[None, :],
        result,
        rows[:, None] < qlen,
    )
    tl.store(LSE + head * TQ + qstart + rows, lse, rows < qlen)


def launch_cached(q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal, maxk=None):
    config = current_config()
    route = config.route if config is not None else "auto"
    if launch_prefill is not None and route != "descriptor":
        if route == "bounded":
            from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.bounded_prefill import (
                launch_bounded_prefill,
            )

            return launch_bounded_prefill(
                q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal, maxk
            )
        return launch_prefill(
            q, k, v, out, lse, cuq, usedk, table, maxq, scale, causal, maxk
        )
    batch, hk, dim = table.shape[0], k.shape[2], q.shape[2]
    nk = cache_extent(k, table, maxk)
    kc = torch.empty((batch * hk * dim, nk), device=q.device, dtype=q.dtype)
    vc = torch.empty((batch * hk * nk, dim), device=q.device, dtype=q.dtype)
    gn = config.gather_tile if config is not None and config.gather_tile else 32
    gw = config.gather_warps if config is not None and config.gather_warps else 4
    launch(
        gather_cache_kernel,
        ((triton.cdiv(nk, gn), batch * hk, 4)),
        k,
        v,
        kc,
        vc,
        table,
        usedk,
        hk,
        dim,
        k.shape[1],
        *k.stride()[:3],
        *v.stride()[:3],
        table.stride(0),
        nk,
        BN=gn,
        num_warps=gw,
        num_stages=1,
    )
    qdesc = create_2d_tma_descriptor(
        q.data_ptr(), q.shape[0], q.shape[1] * dim, 32, dim, q.element_size()
    )
    kdesc = create_2d_tma_descriptor(
        kc.data_ptr(), kc.shape[0], nk, dim, 64, q.element_size()
    )
    vdesc = create_2d_tma_descriptor(
        vc.data_ptr(), vc.shape[0], dim, 64, dim, q.element_size()
    )
    return launch(
        cached_attention_kernel,
        ((triton.cdiv(maxq, 32), batch, q.shape[1])),
        qdesc,
        kdesc,
        vdesc,
        out,
        lse,
        cuq,
        usedk,
        q.shape[0],
        q.shape[1],
        hk,
        nk,
        scale,
        causal,
        num_warps=4,
        num_stages=1,
        **{_BACKEND_OPT: True},
    )


def cached_routes(q, scale):
    """Offer only paths compiled by the installed, unmodified Triton backend."""
    routes = []
    default = None
    if launch_prefill is not None:
        routes.append("tle")
        default = "tle"
        if q.dtype == torch.bfloat16 and scale > 0:
            routes.append("bounded")
            default = "bounded"
    if create_2d_tma_descriptor is not None:
        routes.append("descriptor")
        default = default or "descriptor"
    return default, routes

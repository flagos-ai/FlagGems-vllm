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

"""flash_mla_with_kvcache for T-Head PPU: sparse decode with an FP8 KV cache.

This is the backend's registered flash_mla_with_kvcache. It serves the two
sparse FP8 cache layouts with their own kernels and hands everything else --
dense decode, a cache neither kernel handles, a build without the TLE
primitives they need -- to the generic implementation, whose argument checks
and output allocation it reuses so the operator's contract lives in one place.

The two layouts are different enough to be separate kernels and the caller does
not say which one it has, so this picks by ``head_dim_k``:
``flash_mla_with_kvcache_model1`` for 512 (584 bytes per token) and
``flash_mla_with_kvcache_v32`` for 576 (656 bytes per token, the V3.2 layout).
"""
import functools

import triton.experimental.tle as tle_api
import triton.experimental.tle.language as tle

from flaggems_vllm.ops.flash_mla_with_kvcache import _sparse_decode_dispatch
from flaggems_vllm.ops.flash_mla_with_kvcache import (
    flash_mla_with_kvcache as flash_mla_with_kvcache_generic,
)
from flaggems_vllm.ops.flash_mla_with_kvcache import prepare_call, validate_sparse_args
from flaggems_vllm.runtime.backend._thead.ops.flash_mla_with_kvcache_model1 import (
    can_use_model1_thead,
    sparse_decode_model1_thead,
)
from flaggems_vllm.runtime.backend._thead.ops.flash_mla_with_kvcache_v32 import (
    can_use_v32_thead,
    sparse_decode_v32_thead,
)

_PRIMITIVES = (tle.gpu.alloc, tle.gpu.local_ptr)


@functools.lru_cache(maxsize=1)
def _usable():
    """A T-Head device whose backend has the TLE primitives both kernels need.

    The rest of what they build shared memory out of -- smem, the swizzled
    layouts -- is plain tle.language, which TLE does not gate per backend.
    """
    try:
        from flaggems_vllm import runtime

        if runtime.device.vendor_name != "thead":
            return False
        return tle_api.is_tle_supported(_PRIMITIVES)
    except Exception:
        return False


def flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata,
    num_splits=None,
    softmax_scale=None,
    causal=False,
    is_fp8_kvcache=False,
    indices=None,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    topk_length=None,
    extra_topk_length=None,
    out=None,
    scale_fmt=None,
):
    """Sparse decode on PPU, falling back to the generic operator otherwise.

    scale_fmt is the format the cache scales were quantized in, as passed to
    indexer_k_quant_and_cache. "ue8m0" promises every scale is a power of two,
    which lets the V32 kernel fold the scale into the FP8 exponent and skip a
    multiply per tile; any other value, None included, takes the path that
    reads the scales as written. MODEL1 does not look at it: its scales are
    ue8m0 exponent bytes by layout and never anything else.
    """
    if indices is None or not _usable():
        return flash_mla_with_kvcache_generic(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            head_dim_v,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale,
            causal,
            is_fp8_kvcache,
            indices,
            attn_sink,
            extra_k_cache,
            extra_indices_in_kvcache,
            topk_length,
            extra_topk_length,
            out,
            scale_fmt,
        )

    sched_meta = tile_scheduler_metadata
    out, lse, softmax_scale, topk = prepare_call(
        q,
        k_cache,
        head_dim_v,
        sched_meta,
        num_splits,
        softmax_scale,
        causal,
        is_fp8_kvcache,
        indices,
        extra_k_cache,
        extra_indices_in_kvcache,
        out,
    )
    validate_sparse_args(
        q,
        k_cache,
        indices,
        attn_sink,
        topk_length,
        extra_k_cache,
        extra_indices_in_kvcache,
        extra_topk_length,
        head_dim_v,
        topk,
        causal,
        is_fp8_kvcache,
    )

    head_dim_k = q.shape[3]
    if head_dim_k == 512 and can_use_model1_thead(
        q,
        k_cache,
        indices,
        out,
        lse,
        extra_k_cache,
        extra_indices_in_kvcache,
        topk_length,
        extra_topk_length,
    ):
        # The tile schedule is kept in sched_meta and reused on later calls.
        sparse_decode_model1_thead(
            q,
            k_cache,
            indices,
            attn_sink=attn_sink,
            topk_length=topk_length,
            extra_kv=extra_k_cache,
            extra_indices=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
            out=out,
            lse=lse,
            sm_scale=softmax_scale,
            sched_meta=sched_meta,
        )
        return out, lse
    if head_dim_k == 576 and can_use_v32_thead(
        q, k_cache, indices, out, lse, topk_length
    ):
        sparse_decode_v32_thead(
            q,
            k_cache,
            indices,
            attn_sink=attn_sink,
            out=out,
            lse=lse,
            sm_scale=softmax_scale,
            sched_meta=sched_meta,
            pow2_kv_scales=scale_fmt == "ue8m0",
        )
        return out, lse

    # A sparse layout neither kernel handles; the portable one reads it.
    _sparse_decode_dispatch(
        q,
        k_cache,
        indices,
        out,
        lse,
        attn_sink,
        topk_length,
        extra_k_cache,
        extra_indices_in_kvcache,
        extra_topk_length,
        q.shape[0],
        q.shape[1],
        q.shape[2],
        head_dim_k,
        head_dim_v,
        topk,
        k_cache.shape[1],
        softmax_scale,
        is_fp8_kvcache,
        sched_meta,
    )
    return out, lse

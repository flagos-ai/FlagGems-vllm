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

"""Sparse decode with an FP8 KV cache for T-Head PPU, one entry for both layouts.

The two FP8 cache layouts are different enough to be separate kernels and the
caller does not say which one it has, so this picks by ``head_dim_k``:
``flash_mla_with_kvcache_model1`` for 512 (584 bytes per token) and
``flash_mla_with_kvcache_v32`` for 576 (656 bytes per token, the V3.2 layout).
"""
import functools

import triton
import triton.experimental.tle as tle_api
import triton.experimental.tle.language as tle

from flaggems_vllm.runtime.backend._thead.ops.flash_mla_with_kvcache_model1 import (
    can_use_model1_thead,
    sparse_decode_model1_thead,
)
from flaggems_vllm.runtime.backend._thead.ops.flash_mla_with_kvcache_v32 import (
    can_use_v32_thead,
    sparse_decode_v32_thead,
)


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
        # GPUTarget.backend reads "cuda" on this device; the whitelist is keyed
        # by the backend package the active driver comes from.
        backend = type(triton.runtime.driver.active).__module__.split(".")[2]
        return all(
            tle_api.is_primitive_supported(backend, primitive)
            for primitive in (tle.gpu.alloc, tle.gpu.local_ptr)
        )
    except Exception:
        return False


def sparse_decode_thead(
    q,
    kv,
    indices,
    *,
    attn_sink=None,
    topk_length=None,
    extra_kv=None,
    extra_indices=None,
    extra_topk_length=None,
    out,
    lse,
    sm_scale,
    sched_meta=None,
    pow2_kv_scales=False,
):
    """Runs the kernel for whichever layout q and kv are in and returns True.

    Returns False, having written nothing, when this path cannot serve the call
    -- another device, another dtype, a layout neither kernel handles -- so the
    caller falls back to its portable kernel. Writes out [B, 1, 64, 512] and
    lse [B, 64, 1].

    pow2_kv_scales is the V32 layout's optional promise that every cache scale
    is a power of two, which buys 2.7% there; MODEL1 ignores it, since its
    scales are ue8m0 exponent bytes by layout and never anything else.
    """
    if not _usable():
        return False
    head_dim_k = q.shape[-1]
    if head_dim_k == 512:
        if not can_use_model1_thead(
            q,
            kv,
            indices,
            out,
            lse,
            extra_kv,
            extra_indices,
            topk_length,
            extra_topk_length,
        ):
            return False
        sparse_decode_model1_thead(
            q,
            kv,
            indices,
            attn_sink=attn_sink,
            topk_length=topk_length,
            extra_kv=extra_kv,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            out=out,
            lse=lse,
            sm_scale=sm_scale,
            sched_meta=sched_meta,
        )
        return True
    if head_dim_k == 576:
        if not can_use_v32_thead(q, kv, indices, out, lse, topk_length):
            return False
        sparse_decode_v32_thead(
            q,
            kv,
            indices,
            attn_sink=attn_sink,
            out=out,
            lse=lse,
            sm_scale=sm_scale,
            sched_meta=sched_meta,
            pow2_kv_scales=pow2_kv_scales,
        )
        return True
    return False

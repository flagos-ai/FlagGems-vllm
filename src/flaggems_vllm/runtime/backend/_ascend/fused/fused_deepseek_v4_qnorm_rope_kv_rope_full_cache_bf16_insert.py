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

from flaggems_vllm.ops.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert import (
    _check_args,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.fused.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
    launch_group_size,
)
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _two_sum(a, b):
    """FP32 sum and its exact rounding residual (Knuth TwoSum)."""
    s = a + b
    b_part = s - a
    return s, (a - (s - b_part)) + (b - b_part)


@triton.jit
def _bf16_fma(x, y, z):
    # 910B1 tl.fma rounds the product before adding. Split the FP32
    # coefficient so both products with the 8-significand-bit KV are exact.
    y_hi = (y.to(tl.int32, bitcast=True) & -256).to(tl.float32, bitcast=True)
    hi = x * y_hi
    lo = x * (y - y_hi)
    product = hi + lo
    product_error = (hi - product) + lo

    total, sum_error = _two_sum(product, z)
    correction, residual = _two_sum(product_error, sum_error)
    # Round the correction to odd, preserving which side of a final FP32
    # midpoint the exact sum lies on even when the exponents are far apart.
    bits = correction.to(tl.int32, bitcast=True)
    step = tl.where((correction < 0) == (residual < 0), 1, -1)
    bits = tl.where(((bits & 1) == 0) & (residual != 0), bits + step, bits)
    corrected = total + bits.to(tl.float32, bitcast=True)
    # Non-finite inputs can make the split itself undefined (e.g. inf - inf).
    return tl.where(tl.abs(total) < float("inf"), corrected, tl.fma(x, y, z))


@triton.jit
def _query_block(
    tile, q, positions, table, eps, H: tl.constexpr, B: tl.constexpr, NORM: tl.constexpr
):
    groups: tl.constexpr = triton.cdiv(H, B)
    token = tile // groups
    head_base = (tile % groups) * B
    if H < 2**22:
        head_base = head_base.to(tl.int32)
    heads = head_base + tl.arange(0, B)
    valid = (heads < H)[:, None]
    row = q + token * (H * 512) + (heads * 512)[:, None]
    if NORM:
        dims = tl.arange(0, 512)[None, :]
        x = tl.load(row + dims, valid, other=0).to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / 512.0) + eps)[:, None]
        tl.store(row + dims, x * scale, valid & (dims < 448))

    # A two-lane pair axis issues contiguous RoPE loads and stores on 910B.
    pair_col = 448 + 2 * tl.arange(0, 32)[:, None] + tl.arange(0, 2)[None, :]
    pair_ptr = row[:, :, None] + pair_col[None, :, :]
    pair = tl.load(pair_ptr, valid[:, :, None], other=0).to(tl.float32)
    even, odd = tl.split(pair)
    if NORM:
        even = even * scale
        odd = odd * scale
    freq = table + tl.load(positions + token) * 64 + tl.arange(0, 32)
    cos = tl.load(freq)[None, :]
    sin = tl.load(freq + 32)[None, :]
    re = tl.fma(even, cos, -(odd * sin))
    im = tl.fma(even, sin, odd * cos)
    tl.store(pair_ptr, tl.join(re, im), valid[:, :, None])


@triton.jit
def _insert_row(
    token, kv, cache, slots, positions, table, page_stride, row_stride, S: tl.constexpr
):
    slot = tl.load(slots + token)
    if slot >= 0:
        src = kv + token * 512
        dst = cache + (slot // S) * page_stride + (slot % S) * row_stride
        col = tl.arange(0, 512)
        prefix = col < 448
        tl.store(dst + col, tl.load(src + col, prefix, other=0), prefix)
        freq = table + tl.load(positions + token) * 64 + tl.arange(0, 32)
        cos = tl.load(freq)
        sin = tl.load(freq + 32)
        pair_col = 448 + 2 * tl.arange(0, 32)[:, None] + tl.arange(0, 2)[None, :]
        even, odd = tl.split(tl.load(src + pair_col).to(tl.float32))
        re = _bf16_fma(even, cos, -(odd * sin))
        im = _bf16_fma(even, sin, odd * cos)
        tl.store(dst + pair_col, tl.join(re, im))


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
    B: tl.constexpr,
    S: tl.constexpr,
    NORM: tl.constexpr,
    PERSIST: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if PERSIST:
        # Each vector core walks both independent work lists, amortizing setup.
        if H:
            for tile in range(pid, query_tiles, tl.num_programs(0)):
                _query_block(tile, q, positions, table, eps, H, B, NORM)
        for token in range(pid, insert_count, tl.num_programs(0)):
            _insert_row(
                token, kv, cache, slots, positions, table, page_stride, row_stride, S
            )
    else:
        if pid < query_tiles:
            if H:
                _query_block(pid, q, positions, table, eps, H, B, NORM)
        else:
            _insert_row(
                pid - query_tiles,
                kv,
                cache,
                slots,
                positions,
                table,
                page_stride,
                row_stride,
                S,
            )


_last_launch = None


def _launch(query, kv, cache, slots, positions, table, eps, block_size, norm):
    global _last_launch
    n, h, _ = query.shape
    head_block = 8
    q_tiles = n * triton.cdiv(h, head_block)
    i = slots.numel()
    if not q_tiles + i:
        return
    device_index = query.get_device()
    cores = launch_group_size(device_index)
    persistent = q_tiles + i > cores
    programs = min(cores, max(q_tiles, i)) if persistent else q_tiles + i
    grid = (programs,)
    scalars = (
        float(eps),
        q_tiles,
        i,
        cache.stride(0),
        cache.stride(1),
        h,
        head_block,
        block_size,
        bool(norm),
        persistent,
    )
    tensors = (query, kv, cache, slots, positions, table)
    args = tensors + scalars
    # The compiled runner selects the calling thread's current stream at each
    # invocation. Keep neither a tensor address nor a stream in the cache.
    aligned = all(t.data_ptr() % 16 == 0 for t in tensors)
    if aligned:
        key = (device_index, programs, scalars)
        previous = _last_launch
        if previous is not None and key == previous[0]:
            previous[1](*args)
            return
    compiled, _ = _full_cache_kernel[grid](*args, multibuffer=False)
    if aligned:
        _last_launch = (key, compiled[(programs, 1, 1)])


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
    logger.debug(
        "GEMS_ASCEND FUSED_DEEPSEEK_V4_QNORM_ROPE_KV_ROPE_FULL_CACHE_BF16_INSERT"
    )
    _check_args(
        q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, cache_block_size
    )
    args = (
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        cache_block_size,
        apply_q_norm,
    )
    if torch_device_fn.current_device() == q.get_device():
        _launch(*args)
    else:
        with torch_device_fn.device(q.device):
            _launch(*args)

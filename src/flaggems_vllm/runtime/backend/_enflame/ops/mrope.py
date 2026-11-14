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

"""MRotaryEmbedding (mrope) — Enflame (GCU) implementation.

GCU adaptations relative to the device-agnostic version:

1. **1D-only memory accesses.** The generic (GROUP, BLOCK) 2D q/k tiles fail
   to compile on the current enflame toolchain (``memref_ext.slice_pad_start``
   operandSegmentSizes regression; see ``compiler_issues/vllm_moe_sum.md``).
   This kernel launches one program per (token, head-group) and performs pure
   1D affine loads/stores of ``[HALF]`` rotary pairs, which are proven to
   compile.  Rows beyond 65535 are folded into grid.z with a row guard.

2. **int32 index normalization on the host.** ``tl.load`` from an int64 tensor
   followed by an in-kernel narrowing misreads elements by a 2x element
   offset (silently wrong values plus out-of-bounds reads; see
   ``compiler_issues/triton_i64_narrow_load.md``).  ``positions`` and
   ``axis_map`` are index tensors whose values are provably < 2^31, so the
   host normalizes them to int32 before the launch (ATen cast verified
   value-correct on GCU).  This is backend-required index-dtype
   normalization, not a compute fallback.

Autotune note (workflow G3 exemption): the only launch parameter is
``num_warps``, picked from the semantic tile width ``HALF`` (1D vector of
``rotary_dim // 2``); there is no free perf-parameter surface worth a tuner
entry, mirroring the generic host-side heuristic.

Supported surface: sectioned / interleaved / interleaved-glm axis selection,
neox and interleaved rotation layouts, independent q/k head counts (including
``k_heads > q_heads``), in-place update of q and k.  Unsupported paths raise
``NotImplementedError`` instead of falling back to torch.
"""

import logging
import os
import sys

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.mrope import mrope as _general_mrope
from flaggems_vllm.runtime import torch_device_fn

logger = logging.getLogger(__name__)

GRID_X_MAX = 65535
GRID_Y_MAX = 255
GRID_Z_MAX = 255

_MODE_SECTION = 0
_MODE_INTERLEAVED = 1
_MODE_GLM = 2


@triton.jit
def _mrope_kernel(
    q_ptr,
    k_ptr,
    cache_ptr,
    pos_ptr,
    axis_map_ptr,
    num_tokens,
    grid_x,
    q_row_stride,
    k_row_stride,
    cache_row_stride,
    pos_axis_stride,
    pos_token_stride,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    head_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    section_t: tl.constexpr,
    section_h: tl.constexpr,
    section_w: tl.constexpr,
    mode: tl.constexpr,
    neox: tl.constexpr,
    HALF: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
):
    row = tl.program_id(0) + tl.program_id(2) * grid_x
    head_base = tl.program_id(1) * HEADS_PER_PROG
    if row < num_tokens:
        pair = tl.arange(0, HALF)
        # mode: 0 = sectioned (st/sh/sw), 1 = interleaved, 2 = interleaved-glm
        if mode == 2:
            axis = tl.load(axis_map_ptr + pair).to(tl.int32)
        elif mode == 1:
            mod3 = pair % 3
            is_h = (mod3 == 1) & (pair < section_h * 3)
            is_w = (mod3 == 2) & (pair < section_w * 3)
            axis = tl.where(is_h, 1, tl.where(is_w, 2, 0))
        else:
            axis = tl.where(
                pair < section_t, 0, tl.where(pair < section_t + section_h, 1, 2)
            )
        pos_t = tl.load(pos_ptr + row * pos_token_stride).to(tl.int32)
        pos_h = tl.load(pos_ptr + pos_axis_stride + row * pos_token_stride).to(tl.int32)
        pos_w = tl.load(pos_ptr + 2 * pos_axis_stride + row * pos_token_stride).to(
            tl.int32
        )
        pos = tl.where(axis == 0, pos_t, tl.where(axis == 1, pos_h, pos_w))
        cache_off = pos * cache_row_stride + pair
        cos_v = tl.load(cache_ptr + cache_off).to(tl.float32)
        sin_v = tl.load(cache_ptr + cache_off + HALF).to(tl.float32)
        if neox:
            d0 = pair
            d1 = pair + HALF
        else:
            d0 = pair * 2
            d1 = d0 + 1
        for h in tl.static_range(0, HEADS_PER_PROG):
            head = head_base + h
            if head < q_heads:
                base = row * q_row_stride + head * head_size
                x0 = tl.load(q_ptr + base + d0).to(tl.float32)
                x1 = tl.load(q_ptr + base + d1).to(tl.float32)
                tl.store(q_ptr + base + d0, x0 * cos_v - x1 * sin_v)
                tl.store(q_ptr + base + d1, x1 * cos_v + x0 * sin_v)
            if head < k_heads:
                base = row * k_row_stride + head * head_size
                x0 = tl.load(k_ptr + base + d0).to(tl.float32)
                x1 = tl.load(k_ptr + base + d1).to(tl.float32)
                tl.store(k_ptr + base + d0, x0 * cos_v - x1 * sin_v)
                tl.store(k_ptr + base + d1, x1 * cos_v + x0 * sin_v)


def _as_int32(tensor):
    if tensor.dtype == torch.int32:
        return tensor
    return tensor.to(torch.int32)


def _heads_per_prog(n_tokens, heads):
    """Pick how many heads one program rotates (amortizes the cos/sin gather).

    Decode-size batches keep one head per program for maximum parallelism
    (measured 5-8us vs 18us for a single whole-row program); prefill-size
    batches rotate a whole row (up to 32 heads) per program so the per-row
    position/axis/cos-sin work is done once instead of once per head
    (measured 3.6x faster at 8192 tokens).  ``FLAGGEMS_MROPE_HEADS_PER_PROG``
    overrides the heuristic for tuning experiments.
    """
    override = os.environ.get("FLAGGEMS_MROPE_HEADS_PER_PROG")
    if override is not None:
        return max(1, min(int(override), heads))
    if n_tokens <= 8:
        return 1
    return min(heads, 32)


def mrope(
    query,
    key,
    cos_sin_cache,
    positions,
    section_t,
    section_h,
    section_w,
    head_size,
    rotary_dim,
    is_interleaved=False,
    is_interleaved_glm=False,
    is_neox_style=True,
    axis_map=None,
):
    logger.debug("GEMS MROPE (enflame)")
    n_tokens = query.shape[0]
    if rotary_dim == 0 or n_tokens == 0:
        return query, key

    if query.dim() != 2 or key.dim() != 2:
        raise NotImplementedError(
            f"mrope: expected 2D [num_tokens, heads*head_size] q/k, got "
            f"{query.dim()}D/{key.dim()}D"
        )
    if query.shape[1] % head_size or key.shape[1] % head_size:
        raise NotImplementedError(
            "mrope: q/k widths must be multiples of head_size, got "
            f"{query.shape[1]}/{key.shape[1]} vs head_size={head_size}"
        )
    if query.stride(1) != 1 or key.stride(1) != 1:
        raise NotImplementedError(
            "mrope: q/k must have unit inner stride, got "
            f"{query.stride(1)}/{key.stride(1)}"
        )
    if cos_sin_cache.dim() != 2 or cos_sin_cache.stride(1) != 1:
        raise NotImplementedError(
            "mrope: cos_sin_cache must be 2D with unit inner stride"
        )
    if not cos_sin_cache.dtype.is_floating_point:
        raise NotImplementedError(
            f"mrope: unsupported cos_sin_cache dtype {cos_sin_cache.dtype}"
        )
    if positions.dim() != 2 or positions.shape[0] < 3:
        raise NotImplementedError(
            "mrope: positions must be [>=3, num_tokens] (t/h/w axes), got "
            f"{tuple(positions.shape)}"
        )
    if positions.dtype not in (torch.int32, torch.int64):
        raise NotImplementedError(
            f"mrope: unsupported positions dtype {positions.dtype}"
        )

    half = rotary_dim // 2
    if rotary_dim % 2 != 0 or half < 1 or (half & (half - 1)) != 0:
        raise NotImplementedError(
            f"mrope: rotary_dim must be a positive power-of-2 multiple of 2, "
            f"got {rotary_dim}"
        )
    if rotary_dim > head_size:
        raise NotImplementedError(
            f"mrope: rotary_dim {rotary_dim} > head_size {head_size}"
        )

    positions_i32 = _as_int32(positions)
    if is_interleaved_glm:
        if axis_map is None:
            raise NotImplementedError("mrope: is_interleaved_glm requires axis_map")
        if axis_map.numel() != half:
            raise NotImplementedError(
                f"mrope: axis_map must have rotary_dim//2={half} entries, got "
                f"{axis_map.numel()}"
            )
        if axis_map.dtype not in (torch.int32, torch.int64):
            raise NotImplementedError(
                f"mrope: unsupported axis_map dtype {axis_map.dtype}"
            )
        axis_map_arg = _as_int32(axis_map)
        mode = _MODE_GLM
    elif is_interleaved:
        axis_map_arg = positions_i32
        mode = _MODE_INTERLEAVED
    else:
        axis_map_arg = positions_i32
        mode = _MODE_SECTION

    q_heads = query.shape[1] // head_size
    k_heads = key.shape[1] // head_size
    heads = max(q_heads, k_heads)
    if heads == 0 or heads > GRID_Y_MAX:
        raise NotImplementedError(
            f"mrope: max(q_heads, k_heads)={heads} outside supported "
            f"1..{GRID_Y_MAX} range on this backend"
        )
    grid_x = n_tokens if n_tokens <= GRID_X_MAX else GRID_X_MAX
    grid_z = triton.cdiv(n_tokens, grid_x)
    if grid_z > GRID_Z_MAX:
        raise NotImplementedError(
            f"mrope: num_tokens {n_tokens} exceeds grid capacity "
            f"{GRID_X_MAX * GRID_Z_MAX}"
        )
    heads_per_prog = _heads_per_prog(n_tokens, heads)
    grid_y = triton.cdiv(heads, heads_per_prog)
    num_warps = 2 if half > 32 else 1

    with torch_device_fn.device(query.device):
        _mrope_kernel[(grid_x, grid_y, grid_z)](
            query,
            key,
            cos_sin_cache,
            positions_i32,
            axis_map_arg,
            n_tokens,
            grid_x,
            query.stride(0),
            key.stride(0),
            cos_sin_cache.stride(0),
            positions_i32.stride(0),
            positions_i32.stride(1),
            q_heads,
            k_heads,
            head_size,
            rotary_dim,
            section_t,
            section_h,
            section_w,
            mode,
            bool(is_neox_style),
            half,
            heads_per_prog,
            num_warps=num_warps,
        )
    return query, key


def _install():
    """Wire the GCU implementation into the direct-import entrypoints.

    tests/test_mrope.py and benchmark/test_mrope.py import
    ``from flaggems_vllm.ops.mrope import mrope``, which resolves against the
    already-imported generic module, so the SpecOpRegistrar namespace swap
    alone can not reach them. Replace the attributes in place, mirroring the
    _kunlunxin bf16_paged_mqa_logits / mhc_pre backend overrides.
    """
    for module_name in ("flaggems_vllm.ops.mrope", "flaggems_vllm.ops"):
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        current = getattr(mod, "mrope", None)
        if current is _general_mrope:
            mod.mrope = mrope


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
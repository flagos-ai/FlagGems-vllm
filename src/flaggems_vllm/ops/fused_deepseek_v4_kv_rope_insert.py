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
from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_CACHE_BF16 = tl.constexpr(0)
_CACHE_FP8 = tl.constexpr(1)
_CACHE_PACKED_V4 = tl.constexpr(2)
_CACHE_PACKED_MXFP8 = 3


@triton.jit
def _quantize_ue8m0(values, NUM_BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    tiles = tl.reshape(values, (NUM_BLOCKS, BLOCK_SIZE))
    absmax = tl.maximum(tl.max(tl.abs(tiles), axis=1), 1e-4)
    exponent = tl.ceil(tl.log2(absmax / 448.0))
    inv_scale = tl.exp2(-exponent)
    scaled = tiles * tl.reshape(inv_scale, (NUM_BLOCKS, 1))
    quantized = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    return quantized, exponent


@triton.jit
def _fused_deepseek_v4_kv_rope_insert_kernel(
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    fp8_scale,
    cache_block_size: tl.constexpr,
    cache_block_stride,
    cache_token_stride,
    CACHE_KIND: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    slot_id = tl.load(slot_mapping + token_idx)
    position = tl.load(position_ids + token_idx)
    cs_base = cos_sin_cache + position * 64
    pair_idx = tl.arange(0, 32)
    cos = tl.load(cs_base + pair_idx)
    sin = tl.load(cs_base + 32 + pair_idx)
    if slot_id < 0:
        return

    if CACHE_KIND == _CACHE_PACKED_V4:
        dims = tl.arange(0, 512)
        nope_values = tl.load(
            kv + token_idx * 512 + dims, mask=dims < 448, other=0.0
        ).to(tl.float32)
        rope_dims = tl.arange(0, 64)
        rope_values = tl.load(kv + token_idx * 512 + 448 + rope_dims).to(
            tl.float32
        )

        quantized, exponent = _quantize_ue8m0(nope_values, 8, 64)
        quantized = tl.reshape(quantized, (512,)).to(tl.uint8, bitcast=True)
        even, odd = tl.split(tl.reshape(rope_values, (32, 2)))
        rotated_rope = tl.interleave(
            even * cos - odd * sin, tl.fma(even, sin, odd * cos)
        )

        block_idx = slot_id // cache_block_size
        pos_in_block = slot_id % cache_block_size
        block_base = k_cache + block_idx * cache_block_stride
        token_base = block_base + pos_in_block * 576
        tl.store(token_base + dims, quantized, mask=dims < 448)
        rope_base = (token_base + 448).to(tl.pointer_type(tl.bfloat16))
        tl.store(rope_base + rope_dims, rotated_rope.to(tl.bfloat16))

        scale_base = block_base + cache_block_size * 576 + pos_in_block * 8
        scale_dims = tl.arange(0, 8)
        encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
        tl.store(scale_base + scale_dims, encoded.to(tl.uint8), mask=scale_dims < 7)
        tl.store(scale_base + 7, tl.zeros((), dtype=tl.uint8))
        return

    dims = tl.arange(0, 512)
    values = tl.load(kv + token_idx * 512 + dims).to(tl.float32)
    rope_dims = tl.arange(0, 64)
    rope_values = tl.load(kv + token_idx * 512 + 448 + rope_dims).to(tl.float32)
    even, odd = tl.split(tl.reshape(rope_values, (32, 2)))
    rotated_rope = tl.interleave(
        even * cos - odd * sin, tl.fma(even, sin, odd * cos)
    )
    tiles = tl.reshape(values, (8, 64))
    rotated = tl.reshape(
        tl.where(tl.arange(0, 8)[:, None] == 7, rotated_rope[None, :], tiles),
        (512,),
    )

    if CACHE_KIND >= _CACHE_PACKED_V4:
        rotated = rotated.to(tl.bfloat16).to(tl.float32)

    block_idx = slot_id // cache_block_size
    pos_in_block = slot_id % cache_block_size
    block_base = k_cache + block_idx * cache_block_stride

    if CACHE_KIND == _CACHE_BF16:
        dst = block_base + pos_in_block * cache_token_stride + dims
        tl.store(dst, rotated.to(tl.bfloat16))
    elif CACHE_KIND == _CACHE_FP8:
        dst = block_base + pos_in_block * cache_token_stride + dims
        scaled = rotated * (1.0 / tl.load(fp8_scale))
        tl.store(dst, tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv))
    else:
        token_base = block_base + pos_in_block * 512
        quantized, exponent = _quantize_ue8m0(rotated, 16, 32)
        quantized = tl.reshape(quantized, (512,)).to(tl.uint8, bitcast=True)
        tl.store(token_base + dims, quantized)

        scale_base = block_base + cache_block_size * 512 + pos_in_block * 16
        scale_dims = tl.arange(0, 16)
        encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
        tl.store(scale_base + scale_dims, encoded.to(tl.uint8))


def fused_deepseek_v4_kv_rope_insert(
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_block_size: int,
    fp8_scale: Optional[torch.Tensor] = None,
    kv_mxfp8: bool = False,
) -> None:
    """Apply GPT-J RoPE to KV's last 64 dims and insert into a paged cache.

    Mirrors ``torch.ops._C.fused_deepseek_v4_kv_rope_insert`` in vLLM's
    ``csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu``.
    ``kv`` is read-only contiguous BF16 [tokens, 512]. Only ``k_cache`` is
    mutated; negative slots and rows omitted by ``slot_mapping`` are skipped.
    Positions must index valid rows of the FP32 cos||sin cache [max_pos, 64].
    Live slots must be unique and within the cache allocation.

    Cache formats [num_blocks, cache_block_size, width]:
      * BF16 / E4M3: width 512; E4M3 requires a scalar FP32 ``fp8_scale``.
      * uint8: width 584 (V4) or 528 (MXFP8, ``kv_mxfp8=True``).
        Packed blocks store token data followed by scales, rather than
        independent token records; only the block stride is used.

    Each Triton program handles one token with one warp, matching the CUDA
    implementation's per-warp work assignment. This fixed semantic mapping
    exposes no autotune parameters.
    """
    logger.debug("GEMS FUSED_DEEPSEEK_V4_KV_ROPE_INSERT")
    if not (
        kv.is_contiguous()
        and kv.ndim == 2
        and kv.shape[1] == 512
        and kv.dtype == torch.bfloat16
    ):
        raise RuntimeError("kv must be contiguous bfloat16 [tokens, 512]")
    if not (
        k_cache.device == kv.device
        and k_cache.ndim == 3
        and cache_block_size > 0
        and k_cache.shape[1] == cache_block_size
        and k_cache.stride(2) == 1
    ):
        raise RuntimeError("k_cache must be a paged cache on the KV device")
    for tensor in (slot_mapping, position_ids):
        if not (
            tensor.device == kv.device
            and tensor.dtype == torch.int64
            and tensor.ndim == 1
            and tensor.is_contiguous()
        ):
            raise RuntimeError(
                "slots and positions must be contiguous int64 on the KV device"
            )
    if not (
        position_ids.shape[0] == kv.shape[0] and slot_mapping.shape[0] <= kv.shape[0]
    ):
        raise RuntimeError("positions must match KV rows; slots may omit padded rows")
    if not (
        cos_sin_cache.device == kv.device
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[1] == 64
        and cos_sin_cache.is_contiguous()
    ):
        raise RuntimeError("cos_sin_cache must be contiguous float32 [max_pos, 64]")
    packed = k_cache.dtype == torch.uint8
    fp8 = k_cache.dtype == torch.float8_e4m3fn
    if not (packed or fp8 or k_cache.dtype == torch.bfloat16):
        raise RuntimeError("cache must be uint8, bfloat16 or float8_e4m3fn")
    cache_width = (528 if kv_mxfp8 else 584) if packed else 512
    if k_cache.shape[2] != cache_width:
        raise RuntimeError("cache row width does not match its dtype")
    if fp8 and not (
        fp8_scale is not None
        and fp8_scale.device == kv.device
        and fp8_scale.dtype == torch.float32
        and fp8_scale.numel() == 1
    ):
        raise RuntimeError("fp8 cache requires one float32 KV scale on the KV device")
    if slot_mapping.shape[0] == 0:
        return

    if packed:
        cache_kind = _CACHE_PACKED_MXFP8 if kv_mxfp8 else _CACHE_PACKED_V4
    elif fp8:
        cache_kind = _CACHE_FP8
    else:
        cache_kind = _CACHE_BF16

    with torch_device_fn.device(kv.device):
        _fused_deepseek_v4_kv_rope_insert_kernel[(slot_mapping.shape[0],)](
            kv,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            fp8_scale,
            cache_block_size,
            k_cache.stride(0),
            k_cache.stride(1),
            CACHE_KIND=cache_kind,
            num_warps=1,
        )

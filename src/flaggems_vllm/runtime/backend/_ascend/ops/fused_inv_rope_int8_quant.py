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
from typing import Tuple

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)

_INT8_ABS_MAX = 127.0
_VECTOR_CORE_NUM = None


def _vector_core_num() -> int:
    global _VECTOR_CORE_NUM
    if _VECTOR_CORE_NUM is None:
        device = torch.npu.current_device()
        _VECTOR_CORE_NUM = int(torch.npu.get_device_limit(device)["vector_core_num"])
    return _VECTOR_CORE_NUM


def _is_positive_pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _group_major_view(stored: torch.Tensor) -> torch.Tensor:
    groups, tokens, inner = stored.shape
    return stored.as_strided((tokens, groups, inner), (inner, tokens * inner, 1))


@triton.jit
def _fused_inv_rope_int8_quant_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_ptr,
    scale_ptr,
    num_tokens,
    num_heads,
    heads_per_group: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    q_stride_group,
    q_stride_token,
    scale_stride_group,
    scale_stride_token,
    scale_stride_k,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    ROPE_WIDTH: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    INT8_ABS_MAX: tl.constexpr,
):
    # cos/sin are per token; reloading them on every head does not pipeline.
    # extract_slice stride 2 replaces a tensor slice, which this Triton cannot index.
    tl.static_assert(ROPE_START + ROPE_WIDTH == QUANT_GROUP_SIZE)
    tl.static_assert(ROPE_WIDTH == 2 * HALF_ROPE)
    pid = tl.program_id(0)
    head_offs = tl.arange(0, CHUNKS_PER_HEAD * QUANT_GROUP_SIZE)
    chunk_ids = tl.arange(0, CHUNKS_PER_HEAD)
    half_i = tl.arange(0, HALF_ROPE)
    row = chunk_ids[:, None]
    col = tl.arange(0, QUANT_GROUP_SIZE)[None, :]
    int8_min = -INT8_ABS_MAX - 1.0

    for step in range(BLOCK_TOKENS):
        token_i = pid * BLOCK_TOKENS + step
        if token_i < num_tokens:
            token = token_i.to(tl.int64)
            cache_base = (
                cos_sin_cache_ptr
                + tl.load(positions_ptr + token).to(tl.int64) * cache_stride_pos
            )
            cos_b = tl.reshape(tl.load(cache_base + half_i), (1, HALF_ROPE))
            sin_b = tl.reshape(tl.load(cache_base + HALF_ROPE + half_i), (1, HALF_ROPE))
            for head_i in tl.range(0, num_heads):
                local_head = head_i.to(tl.int64)
                group = local_head // heads_per_group
                head_in_group = local_head % heads_per_group
                input_base = o_ptr + token * o_stride_token + local_head * o_stride_head
                values = tl.reshape(
                    tl.load(input_base + head_offs).to(tl.float32),
                    (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
                )
                last = tle.dsa.extract_slice(
                    values,
                    offsets=(CHUNKS_PER_HEAD - 1, 0),
                    sizes=(1, QUANT_GROUP_SIZE),
                    strides=(1, 1),
                )
                rope = tle.dsa.extract_slice(
                    last,
                    offsets=(0, ROPE_START),
                    sizes=(1, ROPE_WIDTH),
                    strides=(1, 1),
                )
                even = tle.dsa.extract_slice(
                    rope, offsets=(0, 0), sizes=(1, HALF_ROPE), strides=(1, 2)
                )
                odd = tle.dsa.extract_slice(
                    rope, offsets=(0, 1), sizes=(1, HALF_ROPE), strides=(1, 2)
                )
                rot_even = even * cos_b + odd * sin_b
                rot_odd = odd * cos_b - even * sin_b
                rope = tle.dsa.insert_slice(
                    rope,
                    rot_even,
                    offsets=(0, 0),
                    sizes=(1, HALF_ROPE),
                    strides=(1, 2),
                )
                rope = tle.dsa.insert_slice(
                    rope,
                    rot_odd,
                    offsets=(0, 1),
                    sizes=(1, HALF_ROPE),
                    strides=(1, 2),
                )
                last = tle.dsa.insert_slice(
                    last,
                    rope,
                    offsets=(0, ROPE_START),
                    sizes=(1, ROPE_WIDTH),
                    strides=(1, 1),
                )
                values = tle.dsa.insert_slice(
                    values,
                    last,
                    offsets=(CHUNKS_PER_HEAD - 1, 0),
                    sizes=(1, QUANT_GROUP_SIZE),
                    strides=(1, 1),
                )
                absmax = tl.max(tl.abs(values), axis=1)
                scale = absmax * (1.0 / INT8_ABS_MAX)
                safe = tl.where(absmax == 0, 1.0, scale)
                scaled = tl.where(absmax[:, None] == 0, 0.0, values / safe[:, None])
                codes = tl.floor(scaled + 0.5)
                codes = tl.minimum(tl.maximum(codes, int8_min), INT8_ABS_MAX).to(
                    tl.int8
                )
                q_base = (
                    q_ptr
                    + group * q_stride_group
                    + token * q_stride_token
                    + head_in_group * CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
                )
                # hivm.hir.copy requires the pointer rank to match the tile.
                code_buf = tle.dsa.to_buffer(codes, space=tle.dsa.ascend.UB)
                with tle.dsa.hint(inter_no_alias=True):
                    tle.dsa.copy(
                        code_buf,
                        q_base + row * QUANT_GROUP_SIZE + col,
                        [CHUNKS_PER_HEAD, QUANT_GROUP_SIZE],
                    )
                scale_base = (
                    scale_ptr
                    + group * scale_stride_group
                    + token * scale_stride_token
                    + head_in_group * CHUNKS_PER_HEAD * scale_stride_k
                )
                tl.store(scale_base + chunk_ids * scale_stride_k, scale)


def fused_inv_rope_int8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse interleaved RoPE, then symmetric per-group int8 quantization.

    Each group uses scale = absmax / 127 and round-half-up into [-128, 127].
    """
    logger.debug("GEMS ASCEND FUSED INV ROPE INT8 QUANT")

    if o.ndim != 3:
        raise ValueError("`o` must be [num_tokens, num_heads, head_dim]")
    if positions.ndim != 1:
        raise ValueError("`positions` must be 1D")
    if cos_sin_cache.ndim != 2:
        raise ValueError("`cos_sin_cache` must be 2D")
    if o.stride(-1) != 1 or o.stride(1) != o.shape[-1]:
        raise ValueError("`o` must be contiguous in the head and head_dim dimensions")
    if positions.shape[0] != o.shape[0]:
        raise ValueError("positions and o token count mismatch")
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("`cos_sin_cache` must be float32")

    num_tokens, num_heads, head_dim = o.shape
    if num_heads != n_groups * heads_per_group:
        raise ValueError("num_heads must equal n_groups * heads_per_group")
    if head_dim != nope_dim + rope_dim:
        raise ValueError("head_dim must equal nope_dim + rope_dim")
    if head_dim % quant_group_size != 0 or rope_dim % 2 != 0:
        raise ValueError("head_dim and rope_dim must match the quant group")
    if nope_dim % quant_group_size != quant_group_size - rope_dim:
        raise ValueError("rope must sit at the tail of the last quant group")
    if cos_sin_cache.shape[-1] != rope_dim:
        raise ValueError("`cos_sin_cache` width must equal rope_dim")

    chunks_per_head = head_dim // quant_group_size
    rope_start = nope_dim % quant_group_size
    rope_width = quant_group_size - rope_start
    if (
        chunks_per_head < 2
        or not _is_positive_pow2(quant_group_size)
        or not _is_positive_pow2(rope_start)
        or not _is_positive_pow2(rope_width)
    ):
        raise NotImplementedError(
            "ascend fused_inv_rope_int8_quant requires power-of-two quant tiles "
            "and rope boundaries inside the last quant chunk"
        )

    group_width = heads_per_group * head_dim
    num_scale_blocks = group_width // quant_group_size
    quantized = torch.empty(
        (n_groups, num_tokens, group_width), dtype=torch.int8, device=o.device
    )
    scale = torch.empty(
        (n_groups, num_tokens, num_scale_blocks),
        dtype=torch.float32,
        device=o.device,
    )

    if num_tokens and num_heads:
        num_cores = min(_vector_core_num(), num_tokens)
        _fused_inv_rope_int8_quant_kernel[(num_cores,)](
            o,
            positions,
            cos_sin_cache,
            quantized,
            scale,
            num_tokens,
            num_heads,
            heads_per_group=heads_per_group,
            BLOCK_TOKENS=triton.cdiv(num_tokens, num_cores),
            o_stride_token=o.stride(0),
            o_stride_head=o.stride(1),
            cache_stride_pos=cos_sin_cache.stride(0),
            q_stride_group=quantized.stride(0),
            q_stride_token=quantized.stride(1),
            scale_stride_group=scale.stride(0),
            scale_stride_token=scale.stride(1),
            scale_stride_k=scale.stride(2),
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=rope_start,
            ROPE_WIDTH=rope_width,
            HALF_ROPE=rope_dim // 2,
            INT8_ABS_MAX=_INT8_ABS_MAX,
            multibuffer=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        )

    return _group_major_view(quantized), _group_major_view(scale)

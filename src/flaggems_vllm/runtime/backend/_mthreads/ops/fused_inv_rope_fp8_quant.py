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

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._mthreads.ops.per_token_group_quant_fp8 import (
    _quant_groups,
)


@triton.jit
def _fused_inv_rope_fp8_quant_heads(
    O,
    Positions,
    Cache,
    Q,
    S,
    tokens,
    heads,
    heads_per_group: tl.constexpr,
    o_token_stride,
    o_head_stride,
    cache_stride,
    aligned_tokens,
    eps,
    HEADS: tl.constexpr,
    PACKED: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, HEADS * 4)
    head = tl.program_id(1) * HEADS + rows // 4
    chunk = rows % 4
    group = head // heads_per_group
    local_head = head % heads_per_group
    valid_head = head < heads
    scale_col = local_head if PACKED else local_head * 4 + chunk
    scale_cols: tl.constexpr = heads_per_group if PACKED else heads_per_group * 4
    scale_offsets = (
        group.to(tl.int64) * scale_cols * aligned_tokens
        + scale_col * aligned_tokens
        + token
    )
    if token >= tokens:
        if PACKED:
            tl.store(S + scale_offsets, 0, valid_head & (chunk == 0))
        else:
            tl.store(S + scale_offsets, 0.0, valid_head)
        return

    cols = tl.arange(0, 128)
    source = (
        token * o_token_stride
        + head.to(tl.int64)[:, None] * o_head_stride
        + chunk[:, None] * 128
        + cols[None, :]
    )
    x = tl.load(O + source, valid_head[:, None], other=0.0).to(tl.float32)
    is_rope = (chunk[:, None] == 3) & (cols[None, :] >= 64)
    partner = tl.load(
        O + (source - cols[None, :] + (cols[None, :] ^ 1)),
        valid_head[:, None] & is_rope,
        other=0.0,
    ).to(tl.float32)

    # Every head for this token uses the same 32 cosine and sine entries.
    position = tl.load(Positions + token)
    cache_idx = tl.maximum((cols - 64) >> 1, 0)
    cosine = tl.load(Cache + position * cache_stride + cache_idx, cols >= 64, other=1.0)
    sine = tl.load(
        Cache + position * cache_stride + 32 + cache_idx, cols >= 64, other=0.0
    )
    x_add = x * cosine[None, :] + partner * sine[None, :]
    x_sub = x * cosine[None, :] - partner * sine[None, :]
    rotated = tl.where((cols[None, :] & 1) == 0, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)
    quantized, scales = _quant_groups(x, eps, -448.0, 448.0, 1.0 / 448.0, PACKED, False)
    target = (
        group.to(tl.int64)[:, None] * tokens * heads_per_group * 512
        + token * heads_per_group * 512
        + local_head[:, None] * 512
        + chunk[:, None] * 128
        + cols[None, :]
    )
    tl.store(Q + target, quantized.to(Q.dtype.element_ty), valid_head[:, None])
    if PACKED:
        scale_bits = scales.to(tl.int32, bitcast=True)
        exponents = ((scale_bits >> 23) & 0xFF) << (chunk * 8)
        packed = tl.sum(tl.reshape(exponents, (HEADS, 4)), axis=1)
        packed_head = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
        packed_offsets = (
            (packed_head // heads_per_group).to(tl.int64)
            * heads_per_group
            * aligned_tokens
            + (packed_head % heads_per_group) * aligned_tokens
            + token
        )
        tl.store(S + packed_offsets, packed, packed_head < heads)
    else:
        tl.store(S + scale_offsets, scales, valid_head)


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    tma_aligned_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse interleaved RoPE and group FP8 quantization on MUSA."""
    if o.dtype != torch.bfloat16:
        raise NotImplementedError("MUSA inverse RoPE quantization requires BF16 input")
    if dtype not in (None, torch.float8_e4m3fn):
        raise NotImplementedError("only torch.float8_e4m3fn output is supported")
    if o.ndim != 3 or positions.ndim != 1 or cos_sin_cache.ndim != 2:
        raise ValueError("expected o[T,H,D], positions[T], and cos_sin_cache[P,R]")
    if (
        o.device.type != "musa"
        or positions.device != o.device
        or cos_sin_cache.device != o.device
    ):
        raise ValueError("all inputs must be on the same MUSA device")
    if positions.dtype != torch.int64 or cos_sin_cache.dtype != torch.float32:
        raise ValueError("positions must be int64 and cos_sin_cache must be float32")
    if o.stride(-1) != 1 or positions.stride(0) != 1 or cos_sin_cache.stride(1) != 1:
        raise NotImplementedError("input innermost dimensions must be contiguous")
    if (nope_dim, rope_dim, quant_group_size) != (448, 64, 128):
        raise NotImplementedError(
            "MUSA inverse RoPE currently supports dimensions 448+64 and groups of 128"
        )
    if n_groups <= 0 or heads_per_group <= 0 or eps <= 0:
        raise ValueError("group counts and eps must be positive")
    if (
        o.shape[1:] != (n_groups * heads_per_group, 512)
        or positions.shape[0] != o.shape[0]
    ):
        raise ValueError("input shape does not match the group and position counts")
    if cos_sin_cache.shape[1] != rope_dim:
        raise ValueError("cos_sin_cache width must equal rope_dim")
    tokens, heads, _ = o.shape
    aligned_tokens = triton.cdiv(tokens, 4) * 4
    scale_cols = heads_per_group if tma_aligned_scales else heads_per_group * 4
    output = torch.empty(
        (n_groups, tokens, heads_per_group * 512),
        device=o.device,
        dtype=torch.float8_e4m3fn,
    )
    scales = torch.empty(
        n_groups * scale_cols * aligned_tokens,
        device=o.device,
        dtype=torch.int32 if tma_aligned_scales else torch.float32,
    ).as_strided(
        (n_groups, tokens, scale_cols),
        (scale_cols * aligned_tokens, 1, aligned_tokens),
    )
    if tokens == 0:
        return output.transpose(0, 1), scales.transpose(0, 1)

    # S5000 measurements favor separate heads for short batches and eight-head
    # tiles for prefill, where the smaller grid and shared cache loads help.
    heads_per_program = 8 if tokens >= 128 else 1
    _fused_inv_rope_fp8_quant_heads[
        (aligned_tokens, triton.cdiv(heads, heads_per_program))
    ](
        o,
        positions,
        cos_sin_cache,
        output,
        scales,
        tokens,
        heads,
        heads_per_group,
        o.stride(0),
        o.stride(1),
        cos_sin_cache.stride(0),
        aligned_tokens,
        eps,
        HEADS=heads_per_program,
        PACKED=tma_aligned_scales,
        num_warps=4 if heads_per_program == 8 else 1,
        num_stages=1,
    )
    return output.transpose(0, 1), scales.transpose(0, 1)

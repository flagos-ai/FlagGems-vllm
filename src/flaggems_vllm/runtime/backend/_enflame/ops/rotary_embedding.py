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
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _position_ids_as_int32_payload(position_ids):
    # torch_gcu (without TORCH_GCU_ENABLE_INT64_AND_UINT64=1) stores int64
    # tensors as an int32 payload ("GCU not support Long use Int replace"):
    # the value of element i lives in int32 word i. A scalar i64 tl.load in
    # the rotary kernels is lowered to an 8-byte access at byte offset 8*i,
    # which reads value[2i] | (value[2i+1] << 32) instead of value[i]; the
    # bogus position id then drives cos/sin far out of bounds (device abort,
    # see compiler_issues/int64_scalar_load_addressing.md). Re-expose the
    # int64 payload as int32 words so the kernel loads the real values.
    if position_ids.dtype != torch.int64:
        return position_ids
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "") in ("1", "true"):
        return position_ids
    if not position_ids.is_contiguous():
        position_ids = position_ids.contiguous()
    return position_ids.view(torch.int32)


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _prev_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


# Tuning knobs for the contiguous neox fast path below: tokens per program
# block and launch num_warps. Module constants so a benchmark run can pin
# one configuration per subprocess (libentry caches are keyed by constexpr
# args only, so num_warps must not vary within a process).
_NEOX_TPP = 16
_NEOX_NUM_WARPS = 2


@triton.jit
def _rope_neox_tblock(
    src_ptr,
    dst_ptr,
    cos_ptr,
    sin_ptr,
    base,  # scalar element offset of the first token owned by this program
    cos_row0,  # scalar element offset of the first cos/sin row
    LOG_HALF: tl.constexpr,
    LOG_H: tl.constexpr,
    HALF: tl.constexpr,
    NLANE: tl.constexpr,  # TPP * H * HALF, power of two
):
    # Enumerate (token, head, pair) triples of one token block as a flat
    # [NLANE] lane vector: m = t * (H * HALF) + h * HALF + p.
    # x offsets stay affine in m (m + th * HALF == t * RS + h * D + p), and
    # cos/sin rows are re-read per head with duplicated addresses (gather
    # form with shift/AND index arithmetic only -- the lowering survives the
    # current toolchain, unlike div/mod index forms).
    m = tl.arange(0, NLANE)
    th = m >> LOG_HALF  # t * H + h
    t = th >> LOG_H
    xoff = base + m + th * HALF
    coff = cos_row0 + t * HALF + (m & (HALF - 1))
    c = tl.load(cos_ptr + coff).to(tl.float32)
    s = tl.load(sin_ptr + coff).to(tl.float32)
    x1 = tl.load(src_ptr + xoff)
    x2 = tl.load(src_ptr + xoff + HALF)
    tl.store(dst_ptr + xoff, x1 * c - x2 * s)
    tl.store(dst_ptr + xoff + HALF, x2 * c + x1 * s)


@libentry()
@triton.jit
def apply_rotary_pos_emb_neox_kernel(
    q_ptr,  # (n_tokens, q_heads, head_dim), contiguous
    k_ptr,  # (n_tokens, k_heads, head_dim), contiguous
    oq_ptr,
    ok_ptr,
    cos_ptr,  # (max_seq_len, head_dim // 2), contiguous
    sin_ptr,  # (max_seq_len, head_dim // 2), contiguous
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    LOG_HALF: tl.constexpr,
    LOG_Q_H: tl.constexpr,
    LOG_K_H: tl.constexpr,
    TPP: tl.constexpr,  # tokens per program, power of two, <= seq_len
):
    sb = ext.program_id(0)
    b = ext.program_id(1)
    # Anchor the last block so every program owns a full in-range token
    # window; overlapping rows are recomputed and written idempotently.
    s0 = tl.minimum(sb * TPP, seq_len - TPP)
    token0 = b * seq_len + s0

    _rope_neox_tblock(
        q_ptr,
        oq_ptr,
        cos_ptr,
        sin_ptr,
        token0 * (NUM_Q_HEADS * 2 * HALF),
        s0 * HALF,
        LOG_HALF,
        LOG_Q_H,
        HALF,
        TPP * NUM_Q_HEADS * HALF,
    )
    _rope_neox_tblock(
        k_ptr,
        ok_ptr,
        cos_ptr,
        sin_ptr,
        token0 * (NUM_K_HEADS * 2 * HALF),
        s0 * HALF,
        LOG_HALF,
        LOG_K_H,
        HALF,
        TPP * NUM_K_HEADS * HALF,
    )


@libentry()
@triton.jit
def apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = ext.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM // 2)
    mask = ordered_block < HEAD_DIM // 2
    cos = tl.load(cos_ptr + ordered_block, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + ordered_block, mask=mask, other=0.0).to(tl.float32)

    oq_ptr += s_id * oq_stride_s
    q_ptr += s_id * q_stride_s

    for off_h in range(0, NUM_Q_HEADS):
        base_q = off_h * q_stride_h
        base_oq = off_h * oq_stride_h
        if ROTARY_INTERLEAVED:
            even_cols = base_q + (2 * ordered_block) * q_stride_d
            odd_cols = base_q + (2 * ordered_block + 1) * q_stride_d
            even_out = base_oq + (2 * ordered_block) * oq_stride_d
            odd_out = base_oq + (2 * ordered_block + 1) * oq_stride_d
            x_even = tl.load(q_ptr + even_cols, mask=mask, other=0.0)
            x_odd = tl.load(q_ptr + odd_cols, mask=mask, other=0.0)
            y_even = x_even * cos - x_odd * sin
            y_odd = x_odd * cos + x_even * sin
            tl.store(oq_ptr + even_out, y_even, mask=mask)
            tl.store(oq_ptr + odd_out, y_odd, mask=mask)
        else:
            first_cols = base_q + ordered_block * q_stride_d
            second_cols = base_q + (ordered_block + HEAD_DIM // 2) * q_stride_d
            first_out = base_oq + ordered_block * oq_stride_d
            second_out = base_oq + (ordered_block + HEAD_DIM // 2) * oq_stride_d
            x_first = tl.load(q_ptr + first_cols, mask=mask, other=0.0)
            x_second = tl.load(q_ptr + second_cols, mask=mask, other=0.0)
            y_first = x_first * cos - x_second * sin
            y_second = x_second * cos + x_first * sin
            tl.store(oq_ptr + first_out, y_first, mask=mask)
            tl.store(oq_ptr + second_out, y_second, mask=mask)

    ok_ptr += s_id * ok_stride_s
    k_ptr += s_id * k_stride_s

    for off_h in range(0, NUM_K_HEADS):
        base_k = off_h * k_stride_h
        base_ok = off_h * ok_stride_h
        if ROTARY_INTERLEAVED:
            even_cols = base_k + (2 * ordered_block) * k_stride_d
            odd_cols = base_k + (2 * ordered_block + 1) * k_stride_d
            even_out = base_ok + (2 * ordered_block) * ok_stride_d
            odd_out = base_ok + (2 * ordered_block + 1) * ok_stride_d
            x_even = tl.load(k_ptr + even_cols, mask=mask, other=0.0)
            x_odd = tl.load(k_ptr + odd_cols, mask=mask, other=0.0)
            y_even = x_even * cos - x_odd * sin
            y_odd = x_odd * cos + x_even * sin
            tl.store(ok_ptr + even_out, y_even, mask=mask)
            tl.store(ok_ptr + odd_out, y_odd, mask=mask)
        else:
            first_cols = base_k + ordered_block * k_stride_d
            second_cols = base_k + (ordered_block + HEAD_DIM // 2) * k_stride_d
            first_out = base_ok + ordered_block * ok_stride_d
            second_out = base_ok + (ordered_block + HEAD_DIM // 2) * ok_stride_d
            x_first = tl.load(k_ptr + first_cols, mask=mask, other=0.0)
            x_second = tl.load(k_ptr + second_cols, mask=mask, other=0.0)
            y_first = x_first * cos - x_second * sin
            y_second = x_second * cos + x_first * sin
            tl.store(ok_ptr + first_out, y_first, mask=mask)
            tl.store(ok_ptr + second_out, y_second, mask=mask)


@libentry()
@triton.jit
def apply_rotary_pos_emb_inplace_kernel(
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = ext.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM // 2)
    mask = ordered_block < HEAD_DIM // 2
    cos = tl.load(cos_ptr + ordered_block, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + ordered_block, mask=mask, other=0.0).to(tl.float32)

    q_ptr += s_id * q_stride_s

    for off_h in range(0, NUM_Q_HEADS):
        base = off_h * q_stride_h
        if ROTARY_INTERLEAVED:
            even_cols = base + (2 * ordered_block) * q_stride_d
            odd_cols = base + (2 * ordered_block + 1) * q_stride_d
            x_even = tl.load(q_ptr + even_cols, mask=mask, other=0.0)
            x_odd = tl.load(q_ptr + odd_cols, mask=mask, other=0.0)
            y_even = x_even * cos - x_odd * sin
            y_odd = x_odd * cos + x_even * sin
            tl.store(q_ptr + even_cols, y_even, mask=mask)
            tl.store(q_ptr + odd_cols, y_odd, mask=mask)
        else:
            first_cols = base + ordered_block * q_stride_d
            second_cols = base + (ordered_block + HEAD_DIM // 2) * q_stride_d
            x_first = tl.load(q_ptr + first_cols, mask=mask, other=0.0)
            x_second = tl.load(q_ptr + second_cols, mask=mask, other=0.0)
            y_first = x_first * cos - x_second * sin
            y_second = x_second * cos + x_first * sin
            tl.store(q_ptr + first_cols, y_first, mask=mask)
            tl.store(q_ptr + second_cols, y_second, mask=mask)

    k_ptr += s_id * k_stride_s

    for off_h in range(0, NUM_K_HEADS):
        base = off_h * k_stride_h
        if ROTARY_INTERLEAVED:
            even_cols = base + (2 * ordered_block) * k_stride_d
            odd_cols = base + (2 * ordered_block + 1) * k_stride_d
            x_even = tl.load(k_ptr + even_cols, mask=mask, other=0.0)
            x_odd = tl.load(k_ptr + odd_cols, mask=mask, other=0.0)
            y_even = x_even * cos - x_odd * sin
            y_odd = x_odd * cos + x_even * sin
            tl.store(k_ptr + even_cols, y_even, mask=mask)
            tl.store(k_ptr + odd_cols, y_odd, mask=mask)
        else:
            first_cols = base + ordered_block * k_stride_d
            second_cols = base + (ordered_block + HEAD_DIM // 2) * k_stride_d
            x_first = tl.load(k_ptr + first_cols, mask=mask, other=0.0)
            x_second = tl.load(k_ptr + second_cols, mask=mask, other=0.0)
            y_first = x_first * cos - x_second * sin
            y_second = x_second * cos + x_first * sin
            tl.store(k_ptr + first_cols, y_first, mask=mask)
            tl.store(k_ptr + second_cols, y_second, mask=mask)


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
):
    """
    Apply rotary position embedding to q and k

    Args:
        q: (*, q_heads, head_dim)
        k: (*, k_heads, head_dim)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
        position_ids: (*, ), optional, position ids for each token
        rotary_interleaved: whether the head_dim is rotated in an interleaved way

    Returns:
        q_embed: (*, q_heads, head_dim)
        k_embed: (*, k_heads, head_dim)
    """
    logger.debug("GEMS ROTARY_POS_EMBEDDING")
    assert (
        k.shape[-1] == q.shape[-1]
    ), f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
    assert (
        cos.shape[-1] == sin.shape[-1]
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == q.shape[-1]
    ), f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    q_shape = q.shape
    k_shape = k.shape

    assert (
        q.shape[:-2] == k.shape[:-2]
    ), f"q and k must have the same length, got {q.shape[:-2]} and {k.shape[:-2]}"
    if position_ids is None:
        assert (
            len(q.shape) == 4
        ), f"q must have 4 dimensions if position_ids is not provided, got {q.shape}"
        seq_len = q.shape[-3]
    else:
        assert (
            position_ids.shape == q.shape[:-2]
        ), f"position_ids must have the same length as q, got {position_ids.shape} and {q.shape[:-2]}"

        position_ids = position_ids.view(-1)
        position_ids = _position_ids_as_int32_payload(position_ids)
        seq_len = None

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    n_tokens, q_heads, head_dim = q.shape

    # The block size must be the next power of two, sometimes we need to pad it.
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    if inplace:
        grid = (n_tokens,)
        with torch_device_fn.device(q.device):
            apply_rotary_pos_emb_inplace_kernel[grid](
                q,
                k,
                cos,
                sin,
                position_ids,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                position_ids.stride(0) if position_ids is not None else 0,
                cos.stride(0),
                sin.stride(0),
                seq_len,
                q.shape[-2],
                k.shape[-2],
                head_dim,
                padded_head_dim,
                rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
            )
        return q.view(q_shape), k.view(k_shape)
    # If not inplace, we need to create new tensors for q_embed and k_embed
    else:
        q_embed = torch.empty_like(q)
        k_embed = torch.empty_like(k)

        if (
            position_ids is None
            and not rotary_interleaved
            and _is_pow2(head_dim // 2)
            and _is_pow2(q_heads)
            and _is_pow2(k.shape[-2])
            and q.is_contiguous()
            and k.is_contiguous()
            and cos.is_contiguous()
            and sin.is_contiguous()
        ):
            # Fast path: (seq-block, batch) grid keeps the cos/sin row index
            # affine (pos == s, no modulo), so the whole rotation is 1D
            # computed-index loads/stores without masks or broadcasts.
            half = head_dim // 2
            k_heads = k.shape[-2]
            tpp = min(_NEOX_TPP, _prev_pow2(seq_len))
            num_blocks = (seq_len + tpp - 1) // tpp
            grid = (num_blocks, q_shape[0])
            with torch_device_fn.device(q_embed.device):
                apply_rotary_pos_emb_neox_kernel[grid](
                    q,
                    k,
                    q_embed,
                    k_embed,
                    cos,
                    sin,
                    seq_len,
                    NUM_Q_HEADS=q_heads,
                    NUM_K_HEADS=k_heads,
                    HALF=half,
                    LOG_HALF=half.bit_length() - 1,
                    LOG_Q_H=q_heads.bit_length() - 1,
                    LOG_K_H=k_heads.bit_length() - 1,
                    TPP=tpp,
                    num_warps=_NEOX_NUM_WARPS,
                )
            return q_embed.view(q_shape), k_embed.view(k_shape)

        grid = (n_tokens,)
        with torch_device_fn.device(q_embed.device):
            apply_rotary_pos_emb_kernel[grid](
                q_embed,
                k_embed,
                q,
                k,
                cos,
                sin,
                position_ids,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                q_embed.stride(0),
                q_embed.stride(1),
                q_embed.stride(2),
                k_embed.stride(0),
                k_embed.stride(1),
                k_embed.stride(2),
                position_ids.stride(0) if position_ids is not None else 0,
                cos.stride(0),
                sin.stride(0),
                seq_len,
                q.shape[-2],
                k.shape[-2],
                head_dim,
                padded_head_dim,
                rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
            )
        q_embed = q_embed.view(q_shape)
        k_embed = k_embed.view(k_shape)
        return q_embed, k_embed

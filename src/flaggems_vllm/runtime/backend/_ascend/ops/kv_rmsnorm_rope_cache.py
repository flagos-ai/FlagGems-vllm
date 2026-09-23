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

"""Fused KV RMSNorm + RoPE + paged-KV cache writeback for Ascend.

``kv_rmsnorm_rope_cache`` splits a fused KV projection of shape
``[batch, 1, seq_len, rms_size + rope_size]`` (RMS part first, RoPE part
second), applies RMSNorm to the RMS half and rotary position embedding to the
RoPE half, and scatters both results into the paged ``k_cache`` / ``ckv_cache``
through the per-token slot ``index``.
"""

import logging

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)

RMS_SIZE = 512
ROPE_SIZE = 64

CACHE_MODE_NORM = tl.constexpr(0)
CACHE_MODE_PA = tl.constexpr(1)
CACHE_MODE_PA_BNSD = tl.constexpr(1)
CACHE_MODE_PA_NZ = tl.constexpr(2)
CACHE_MODE_PA_BLK_BNSD = tl.constexpr(3)
CACHE_MODE_PA_BLK_NZ = tl.constexpr(4)
cache_mode_map = {
    "Norm": 0,
    "PA_BNSD": 1,
    "PA": 1,
    "PA_NZ": 2,
    "PA_BLK_BNSD": 3,
    "PA_BLK_NZ": 4,
}


@triton.jit
def _apply_rotary_pos_emb_kernel(
    q_embed_ptr,
    k_cache_ptr,
    index,  # [batch_size * seq_len]
    q_ptr,  # (batch, 1, seq_len, rms_size + rope_size)
    cos_ptr,  # (batch, 1, seq_len, rope_size)
    sin_ptr,  # (batch, 1, seq_len, rope_size)
    q_last_dim,
    RMS_SIZE_C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_PADDED_HEAD_DIM: tl.constexpr,
    CACHE_MODE: tl.constexpr,
    token_num,
    BLOCK_SIZE_TOKEN: tl.constexpr,
    IS_ALIGNED: tl.constexpr,
):
    s_id_batch = tl.program_id(0)

    s_id = s_id_batch * BLOCK_SIZE_TOKEN + tl.arange(0, BLOCK_SIZE_TOKEN)

    mask_batch = s_id[:, None] < token_num
    cos_or_sin_offset = s_id[:, None] * HEAD_DIM
    cos_ptr += cos_or_sin_offset
    sin_ptr += cos_or_sin_offset

    offsets_all = tl.arange(0, HEAD_DIM)[None, :]

    # The RoPE part of the fused KV row starts after the RMS part.
    x_offset = s_id[:, None] * q_last_dim + RMS_SIZE_C

    # Load the first and second halves of the RoPE input.
    x = tl.load(q_ptr + x_offset + offsets_all, mask=mask_batch, other=0.0)
    x1 = tle.dsa.extract_slice(
        x, (0, 0), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 2)
    )
    x2 = tle.dsa.extract_slice(
        x, (0, 1), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 2)
    )

    # cos/sin layout: [BLOCK_SIZE_TOKEN, HEAD_DIM], stride [HEAD_DIM, 1]
    cos_tile = tl.load(cos_ptr + offsets_all, mask=mask_batch, other=0.0).to(tl.float32)
    sin_tile = tl.load(sin_ptr + offsets_all, mask=mask_batch, other=0.0).to(tl.float32)

    # cos_1/sin_1: first half [0:half_dim], sequential access
    cos_1 = tle.dsa.extract_slice(
        cos_tile, (0, 0), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 1)
    )
    sin_1 = tle.dsa.extract_slice(
        sin_tile, (0, 0), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 1)
    )

    # cos_2/sin_2: second half [half_dim:HEAD_DIM], sequential access
    cos_2 = tle.dsa.extract_slice(
        cos_tile,
        (0, HALF_PADDED_HEAD_DIM),
        (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM),
        (1, 1),
    )
    sin_2 = tle.dsa.extract_slice(
        sin_tile,
        (0, HALF_PADDED_HEAD_DIM),
        (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM),
        (1, 1),
    )

    # First half output: x1*cos_1 - x2*sin_1
    first_half = (x1 * cos_1 - x2 * sin_1).to(q_embed_ptr.dtype.element_ty)

    # Second half output: x2*cos_2 + x1*sin_2
    second_half = (x2 * cos_2 + x1 * sin_2).to(q_embed_ptr.dtype.element_ty)

    result = tl.zeros((BLOCK_SIZE_TOKEN, HEAD_DIM), dtype=q_embed_ptr.dtype.element_ty)
    result = tle.dsa.insert_slice(
        result,
        first_half,
        offsets=(0, 0),
        sizes=(BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM),
        strides=(1, 1),
    )
    result = tle.dsa.insert_slice(
        result,
        second_half,
        offsets=(0, HALF_PADDED_HEAD_DIM),
        sizes=(BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM),
        strides=(1, 1),
    )

    # Store the RoPE output.
    tl.store(q_embed_ptr + cos_or_sin_offset + offsets_all, result)

    index_value = tl.load(index + s_id)

    row_ids = tl.arange(0, HEAD_DIM)
    if CACHE_MODE == CACHE_MODE_PA:
        for i in tle.dsa.parallel(BLOCK_SIZE_TOKEN):
            offset_i = s_id_batch * BLOCK_SIZE_TOKEN + i
            if IS_ALIGNED:
                reload_result = tle.dsa.extract_slice(
                    result, (i, 0), (1, HEAD_DIM), (1, 1)
                )
                reload_result = tl.reshape(reload_result, (HEAD_DIM))

                k_cache_offset = tle.dsa.extract_element(index_value, (i,)) * HEAD_DIM

                tl.store(k_cache_ptr + k_cache_offset + row_ids, reload_result)
            else:
                if offset_i < token_num:
                    reload_result = tle.dsa.extract_slice(
                        result, (i, 0), (1, HEAD_DIM), (1, 1)
                    )
                    reload_result = tl.reshape(reload_result, (HEAD_DIM))

                    k_cache_offset = (
                        tle.dsa.extract_element(index_value, (i,)) * HEAD_DIM
                    )

                    tl.store(k_cache_ptr + k_cache_offset + row_ids, reload_result)


def apply_rotary_pos_emb(q, cos, sin, cache_mode, index, k_cache):
    """Apply rotary position embedding to the RoPE part and update k_cache."""
    assert (
        cos.shape[-1] == sin.shape[-1]
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    batch, _, sequence_len, head_dim = cos.shape

    # The block size must be the next power of two, sometimes we need to pad it.
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    q_embed = torch.empty_like(cos)

    n_tokens = batch * sequence_len

    BLOCK_SIZE_TOKEN = 64
    grid = (triton.cdiv(n_tokens, BLOCK_SIZE_TOKEN),)

    IS_ALIGNED = n_tokens % BLOCK_SIZE_TOKEN == 0

    _apply_rotary_pos_emb_kernel[grid](
        q_embed,
        k_cache,
        index,
        q,
        cos,
        sin,
        q.shape[-1],
        RMS_SIZE,
        head_dim,
        padded_head_dim // 2,
        cache_mode_map[cache_mode],
        n_tokens,
        BLOCK_SIZE_TOKEN,
        IS_ALIGNED,
    )
    return q_embed


@triton.jit(do_not_specialize=["eps"])
def _rms_norm_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input
    w_ptr,  # pointer to the weights
    index_ptr,
    kv_cache_ptr,
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    CACHE_MODE: tl.constexpr,
    token_num,
    BLOCK_SIZE_TOKEN: tl.constexpr,
    BLOCK_SIZE_TOKEN_PER: tl.constexpr,
    BLOCK_LOOP_SIZE: tl.constexpr,
    IS_ALIGNED: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)

    pid_offset_ori = pid * BLOCK_SIZE_TOKEN + tl.arange(0, BLOCK_SIZE_TOKEN_PER)
    for j in range(BLOCK_LOOP_SIZE):
        pid_offset_1 = pid_offset_ori + j * BLOCK_SIZE_TOKEN_PER
        pid_offset = pid_offset_1[:, None]
        out_ptr_1 = out_ptr + pid_offset * y_stride_r

        mask = pid_offset < token_num
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(
            in_ptr + pid_offset * x_stride_r + cols * x_stride_c, mask, other=0.0
        ).to(cdtype)

        var = tl.sum(x * x, axis=1) * (1 / N)
        rrms = tl.sqrt(var + eps)

        w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE)[None, :])
        y = (x / rrms[:, None] * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr_1 + cols * y_stride_c, y)

        index_value = tl.load(index_ptr + pid_offset_1)

        if CACHE_MODE == CACHE_MODE_PA:
            for i in tle.dsa.parallel(BLOCK_SIZE_TOKEN_PER):
                offset_i = pid * BLOCK_SIZE_TOKEN + j * BLOCK_SIZE_TOKEN_PER + i
                if IS_ALIGNED:
                    # Get the global token position in the KV cache.
                    value_reload = tle.dsa.extract_slice(
                        y, (i, 0), (1, BLOCK_SIZE), (1, 1)
                    )
                    tl.compile_hint(value_reload, "disable_bubble_up")
                    value_reload = tl.reshape(value_reload, (BLOCK_SIZE))

                    offset_kv_cache = tle.dsa.extract_element(index_value, (i,))

                    k_cache_offset = offset_kv_cache * N + cols
                    tl.store(kv_cache_ptr + k_cache_offset, value_reload)
                else:
                    if offset_i < token_num:
                        value_reload = tle.dsa.extract_slice(
                            y, (i, 0), (1, BLOCK_SIZE), (1, 1)
                        )
                        tl.compile_hint(value_reload, "disable_bubble_up")
                        value_reload = tl.reshape(value_reload, (BLOCK_SIZE))

                        offset_kv_cache = tle.dsa.extract_element(index_value, (i,))

                        k_cache_offset = offset_kv_cache * N + cols
                        tl.store(kv_cache_ptr + k_cache_offset, value_reload)


def rms_norm(x, weight, index, ckv_cache, cache_mode, eps=1e-5):
    """RMSNorm the RMS part of the fused KV row and update ckv_cache."""
    N = RMS_SIZE
    BLOCK_SIZE = N

    batch, _, sequence_len, x_last_dim = x.shape
    y = torch.zeros([batch, 1, sequence_len, N], dtype=x.dtype).to(x.device)

    token_num = batch * sequence_len
    BLOCK_SIZE_TOKEN = 128
    BLOCK_SIZE_TOKEN_PER = 16
    BLOCK_LOOP_SIZE = 8

    IS_ALIGNED = token_num % BLOCK_SIZE_TOKEN == 0

    grid = (triton.cdiv(token_num, BLOCK_SIZE_TOKEN),)

    _rms_norm_kernel[grid](
        y,
        x,
        weight,
        index,
        ckv_cache,
        N,
        1,
        x_last_dim,
        1,
        N,
        eps,
        BLOCK_SIZE,
        cache_mode_map[cache_mode],
        token_num,
        BLOCK_SIZE_TOKEN,
        BLOCK_SIZE_TOKEN_PER,
        BLOCK_LOOP_SIZE,
        IS_ALIGNED,
    )

    return y


def kv_rmsnorm_rope_cache(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    k_rope_scale,
    c_kv_scale,
    k_rope_offset,
    c_kv_offset,
    epsilon,
    cache_mode,
    is_output_kv,
):
    """Apply RMSNorm + RoPE to a fused KV projection and update the caches.

    Args:
        kv: [batch_size, 1, seq_len, rms_size + rope_size]; the RMS part
            occupies [0:rms_size] and the RoPE part [rms_size:].
        gamma: [rms_size] RMSNorm weight.
        cos: [batch_size, 1, seq_len, rope_size].
        sin: [batch_size, 1, seq_len, rope_size].
        index: slot ids into k_cache / ckv_cache, length batch_size * seq_len.
        k_cache: paged cache, usually (page_num, page_size, 1, rope_size).
        ckv_cache: paged cache, usually (page_num, page_size, 1, rms_size).
        k_rope_scale: [rope_size] or None, rope scaling.
        c_kv_scale: [rms_size] or None, rms scaling.
        k_rope_offset: [rope_size] or None, rope offset.
        c_kv_offset: [rms_size] or None, rms offset.
        epsilon: RMSNorm epsilon, default 1e-5.
        cache_mode: paged-cache layout, see ``cache_mode_map``.
        is_output_kv: whether to also return the normed / rope-ed activations.
    """
    logger.debug("GEMS_ASCEND KV_RMSNORM_ROPE_CACHE")
    if kv.shape[-1] != RMS_SIZE + ROPE_SIZE:
        raise ValueError(
            f"kv last dimension must be {RMS_SIZE + ROPE_SIZE}, got {kv.shape[-1]}"
        )
    if cache_mode not in cache_mode_map:
        raise ValueError(f"unsupported cache_mode {cache_mode!r}")
    if cache_mode_map[cache_mode] != CACHE_MODE_PA:
        raise NotImplementedError("only the PA / PA_BNSD cache mode is implemented")

    rope_embedding_value = apply_rotary_pos_emb(
        kv, cos, sin, cache_mode, index, k_cache
    )

    rms_norm_out = rms_norm(kv, gamma, index, ckv_cache, cache_mode, epsilon)
    if is_output_kv:
        return k_cache, ckv_cache, rope_embedding_value, rms_norm_out
    else:
        return k_cache, ckv_cache

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

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_fp8_quant(
    kv_quant_blk,  # [64], fp32
    out_data_ptrs,  # [64]
    out_scale_ptr,  # scalar
):
    FP8_MAX: tl.constexpr = 448.0
    block_max = tl.max(tl.abs(kv_quant_blk), axis=-1)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(out_data_ptrs, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(out_scale_ptr, encoded_scale.to(tl.uint8))


@triton.jit
def _kv_fp8_quant_2d(
    kv_quant_blk,  # [2, 32], fp32
    out_data_ptrs,  # [2, 32]
    out_scale_ptrs,  # [2, 1]
):
    FP8_MAX: tl.constexpr = 448.0
    block_max = tl.max(tl.abs(kv_quant_blk), axis=-1, keep_dims=True)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(out_data_ptrs, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(out_scale_ptrs, encoded_scale.to(tl.uint8))


@triton.jit
def _kv_fp8_quant_3d(
    kv_quant_blk,  # [TPP, 2, 32], fp32
    out_data_ptrs,  # [TPP, 2, 32]
    out_scale_ptrs,  # [TPP, 2, 1]
    ins_ok,  # [TPP, 1, 1]
):
    FP8_MAX: tl.constexpr = 448.0
    block_max = tl.max(tl.abs(kv_quant_blk), axis=-1, keep_dims=True)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(out_data_ptrs, x_uint8, mask=ins_ok)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(out_scale_ptrs, encoded_scale.to(tl.uint8), mask=ins_ok)


@triton.jit
def _deepseek_v4_kv_quant(
    qkv_blk_rope,  # [ROPE_DIM], bf16
    kv_base,
    block_base,
    pos_in_block,
    cache_block_size,
):
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    QUANT_BLOCK: tl.constexpr = 64
    NUM_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK  # 7
    SCALE_BYTES_PER_TOKEN: tl.constexpr = NUM_QUANT_BLOCKS + 1  # 8 (7 real + 1 pad)
    TOKEN_DATA_BYTES: tl.constexpr = NOPE_DIM + 2 * ROPE_DIM  # 576

    offset_rope = tl.arange(0, ROPE_DIM)
    offset_quant = tl.arange(0, QUANT_BLOCK)
    token_fp8_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES
    token_bf16_ptr = token_fp8_ptr + NOPE_DIM
    token_bf16_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    token_scale_ptr = (
        block_base
        + cache_block_size * TOKEN_DATA_BYTES
        + pos_in_block * SCALE_BYTES_PER_TOKEN
    )
    # store kv rope
    tl.store(token_bf16_ptr + offset_rope, qkv_blk_rope)  # [ROPE_DIM]
    # quantization of kv nope
    # unroll the quantization loop and co-issue loads for better performance
    kv_quant_blk0 = tl.load(kv_base + offset_quant)
    kv_quant_blk1 = tl.load(kv_base + QUANT_BLOCK + offset_quant)
    kv_quant_blk2 = tl.load(kv_base + 2 * QUANT_BLOCK + offset_quant)
    kv_quant_blk3 = tl.load(kv_base + 3 * QUANT_BLOCK + offset_quant)
    kv_quant_blk4 = tl.load(kv_base + 4 * QUANT_BLOCK + offset_quant)
    kv_quant_blk5 = tl.load(kv_base + 5 * QUANT_BLOCK + offset_quant)
    kv_quant_blk6 = tl.load(kv_base + 6 * QUANT_BLOCK + offset_quant)
    # quant group 0
    qblock_idx = 0
    _kv_fp8_quant(
        kv_quant_blk0.to(tl.float32),
        token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant,
        token_scale_ptr + qblock_idx,
    )
    # quant group 1
    qblock_idx = 1
    _kv_fp8_quant(
        kv_quant_blk1.to(tl.float32),
        token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant,
        token_scale_ptr + qblock_idx,
    )
    # quant group 2
    qblock_idx = 2
    _kv_fp8_quant(
        kv_quant_blk2.to(tl.float32),
        token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant,
        token_scale_ptr + qblock_idx,
    )
    # quant group 3
    qblock_idx = 3
    _kv_fp8_quant(
        kv_quant_blk3.to(tl.float32),
        token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant,
        token_scale_ptr + qblock_idx,
    )
    # quant group 4
    qblock_idx = 4
    _kv_fp8_quant(
        kv_quant_blk4.to(tl.float32),
        token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant,
        token_scale_ptr + qblock_idx,
    )
    # quant group 5
    qblock_idx = 5
    _kv_fp8_quant(
        kv_quant_blk5.to(tl.float32),
        token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant,
        token_scale_ptr + qblock_idx,
    )
    # quant group 6
    qblock_idx = 6
    _kv_fp8_quant(
        kv_quant_blk6.to(tl.float32),
        token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant,
        token_scale_ptr + qblock_idx,
    )
    # padding of scale
    tl.store(token_scale_ptr + NUM_QUANT_BLOCKS, tl.zeros((), dtype=tl.uint8))


@triton.jit
def _deepseek_v41_kv_quant(
    qkv_blk_rope,  # [ROPE_DIM], bf16
    kv_base,
    block_base,
    pos_in_block,
    cache_block_size,
):
    """
    V4.1 (528 B/token, kv_mxfp8=true) -- RoPE is quantized as well, so the
    whole row is MXFP8 (fp8 e4m3 with one UE8M0 scale per 32 dims)
    [0,      bs*512):          token data, 512 fp8 each
    [bs*512, bs*512 + bs*16):  UE8M0 scales, 16 per token
    """
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    QUANT_BLOCK: tl.constexpr = 32
    TOKEN_DATA_BYTES: tl.constexpr = 512
    SCALE_BYTES_PER_TOKEN: tl.constexpr = 16
    NOPE_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK  # 14
    ROPE_QUANT_BLOCKS: tl.constexpr = ROPE_DIM // QUANT_BLOCK  # 2

    repeats = tl.arange(0, ROPE_QUANT_BLOCKS)
    offset_quant = tl.arange(0, QUANT_BLOCK)
    token_data_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES
    token_scale_ptr = (
        block_base
        + cache_block_size * TOKEN_DATA_BYTES
        + pos_in_block * SCALE_BYTES_PER_TOKEN
    )
    # quant rope
    _kv_fp8_quant_2d(
        qkv_blk_rope.reshape(ROPE_QUANT_BLOCKS, QUANT_BLOCK).to(tl.float32),
        token_data_ptr
        + NOPE_DIM
        + repeats[:, None] * QUANT_BLOCK
        + offset_quant[None, :],
        token_scale_ptr + NOPE_QUANT_BLOCKS + repeats[:, None],
    )
    # quant nope
    for i in range(NOPE_QUANT_BLOCKS // ROPE_QUANT_BLOCKS):
        chunk_offs = i * ROPE_QUANT_BLOCKS + repeats
        kv_quant_blk = tl.load(
            kv_base + chunk_offs[:, None] * QUANT_BLOCK + offset_quant[None, :]
        )
        _kv_fp8_quant_2d(
            kv_quant_blk.to(tl.float32),
            token_data_ptr + chunk_offs[:, None] * QUANT_BLOCK + offset_quant[None, :],
            token_scale_ptr + chunk_offs[:, None],
        )


@triton.jit
def _tiled_deepseek_v41_kv_quant(
    qkv_blk_rope,  # [TPP, ROPE_DIM], bf16
    kv_base,  # [TPP]
    block_base,  # [TPP]
    pos_in_block,  # [TPP]
    cache_block_size,
    TPP: tl.constexpr,  # tokens per program
    ins_ok,  # [TPP]
):
    """
    V4.1 (528 B/token, kv_mxfp8=true) -- RoPE is quantized as well, so the
    whole row is MXFP8 (fp8 e4m3 with one UE8M0 scale per 32 dims)
    [0,      bs*512):          token data, 512 fp8 each
    [bs*512, bs*512 + bs*16):  UE8M0 scales, 16 per token
    """
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    QUANT_BLOCK: tl.constexpr = 32
    TOKEN_DATA_BYTES: tl.constexpr = 512
    SCALE_BYTES_PER_TOKEN: tl.constexpr = 16
    NOPE_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK  # 14
    ROPE_QUANT_BLOCKS: tl.constexpr = ROPE_DIM // QUANT_BLOCK  # 2

    repeats = tl.arange(0, ROPE_QUANT_BLOCKS)
    offset_quant = tl.arange(0, QUANT_BLOCK)
    token_data_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES
    token_scale_ptr = (
        block_base
        + cache_block_size * TOKEN_DATA_BYTES
        + pos_in_block * SCALE_BYTES_PER_TOKEN
    )
    # quant rope
    _kv_fp8_quant_3d(
        qkv_blk_rope.reshape(TPP, ROPE_QUANT_BLOCKS, QUANT_BLOCK).to(tl.float32),
        token_data_ptr[:, None, None]
        + NOPE_DIM
        + repeats[None, :, None] * QUANT_BLOCK
        + offset_quant[None, None, :],
        token_scale_ptr[:, None, None] + NOPE_QUANT_BLOCKS + repeats[None, :, None],
        ins_ok[:, None, None],
    )
    # quant nope
    for i in range(NOPE_QUANT_BLOCKS // ROPE_QUANT_BLOCKS):
        chunk_offs = i * ROPE_QUANT_BLOCKS + repeats
        kv_quant_blk = tl.load(
            kv_base[:, None, None]
            + chunk_offs[None, :, None] * QUANT_BLOCK
            + offset_quant[None, None, :]
        )
        _kv_fp8_quant_3d(
            kv_quant_blk.to(tl.float32),
            token_data_ptr[:, None, None]
            + chunk_offs[:, None] * QUANT_BLOCK
            + offset_quant[None, :],
            token_scale_ptr[:, None, None] + chunk_offs[None, :, None],
            ins_ok[:, None, None],
        )


@triton.jit
def fused_qnorm_rope_kv_insert_kernel(
    q,
    q_out,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    q_head_padded: tl.constexpr,
    eps,
    cache_block_size: tl.constexpr,
    apply_q_norm: tl.constexpr,
    kv_mxfp8: tl.constexpr,
    apply_q_rope: tl.constexpr,
    is_q_interleaved: tl.constexpr,
    num_tokens,
    num_heads: tl.constexpr,
    kv_block_stride,
    num_tokens_insert,
):
    HEAD_DIM: tl.constexpr = 512
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    HALF_ROPE_DIM: tl.constexpr = 32
    WARP_SIZE: tl.constexpr = 32
    ELEMS_PER_LANE: tl.constexpr = 16
    offset = tl.arange(0, HEAD_DIM)
    lane_offs = tl.arange(0, WARP_SIZE)
    elem_offs = tl.arange(0, ELEMS_PER_LANE)

    pid = tl.program_id(0).to(tl.int64)  # grid = (num_tokens * (q_head_padded + 1),)
    blocks_per_token = q_head_padded + 1
    token_idx = pid // blocks_per_token
    if token_idx >= num_tokens:
        return
    slot_idx = pid % blocks_per_token
    is_kv = slot_idx == q_head_padded
    if is_kv and token_idx >= num_tokens_insert:  # no need to insert
        return
    is_pad_q = (not is_kv) and (slot_idx >= num_heads)
    if is_pad_q:
        if not is_q_interleaved:
            tl.store(
                q_out + (token_idx * q_head_padded + slot_idx) * HEAD_DIM + offset, 0
            )
        else:
            tl.store(
                q_out
                + token_idx * q_head_padded * HEAD_DIM
                + (lane_offs[:, None] * q_head_padded + slot_idx) * ELEMS_PER_LANE
                + elem_offs[None, :],
                0,
            )
        return

    q_base = q + (token_idx * num_heads + slot_idx) * HEAD_DIM
    kv_base = kv + token_idx * HEAD_DIM
    mask_nope = offset < NOPE_DIM
    offset_rope = tl.arange(0, ROPE_DIM)
    offset_half_rope = tl.arange(0, HALF_ROPE_DIM)
    if not is_kv:
        # load q
        q_blk = tl.load(q_base + offset).to(tl.float32)
        q_blk_rope = tl.load(q_base + NOPE_DIM + offset_rope).to(tl.float32)
        # RMSNorm with no weight
        if apply_q_norm:
            variance = tl.sum(q_blk * q_blk) / HEAD_DIM
            rsqrt = tl.rsqrt(variance + eps)
            q_blk = q_blk * rsqrt
        # store q or q nope
        if not is_q_interleaved:
            tl.store(
                q_out + (token_idx * q_head_padded + slot_idx) * HEAD_DIM + offset,
                q_blk.to(tl.bfloat16),
                mask=True if not apply_q_rope else mask_nope,
            )
        else:
            new_lane_offs = (slot_idx * WARP_SIZE + lane_offs) // num_heads
            new_slot_offs = (slot_idx * WARP_SIZE + lane_offs) % num_heads
            tl.store(
                q_out
                + token_idx * q_head_padded * HEAD_DIM
                + (new_lane_offs * q_head_padded + new_slot_offs)[:, None]
                * ELEMS_PER_LANE
                + elem_offs[None, :],
                q_blk.reshape(WARP_SIZE, ELEMS_PER_LANE).to(tl.bfloat16),
                mask=(
                    True
                    if not apply_q_rope
                    else lane_offs[:, None] < (NOPE_DIM // ELEMS_PER_LANE)
                ),
            )

        if not apply_q_rope:
            return
        if apply_q_norm:
            qkv_blk_rope = q_blk_rope * rsqrt
        else:
            qkv_blk_rope = q_blk_rope
    else:
        # load kv rope
        qkv_blk_rope = tl.load(kv_base + NOPE_DIM + offset_rope).to(tl.float32)

    if is_kv or apply_q_rope:
        # load cos/sin
        position_id = tl.load(position_ids + token_idx)  # i64
        cs_base = cos_sin_cache + position_id * ROPE_DIM
        cos_blk = tl.load(cs_base + offset_half_rope)  # [HALF_ROPE_DIM], f32
        sin_blk = tl.load(
            cs_base + offset_half_rope + HALF_ROPE_DIM
        )  # [HALF_ROPE_DIM], f32
        # ROPE
        qkv_blk_rope_2d = tl.reshape(qkv_blk_rope, HALF_ROPE_DIM, 2)
        even_blk, odd_blk = tl.split(qkv_blk_rope_2d)  # [HALF_ROPE_DIM], f32
        new_even_blk = even_blk * cos_blk - odd_blk * sin_blk
        new_odd_blk = even_blk * sin_blk + odd_blk * cos_blk
        qkv_blk_rope = tl.reshape(tl.join(new_even_blk, new_odd_blk), ROPE_DIM)
        if not is_kv:
            # store q rope
            if not is_q_interleaved:
                tl.store(
                    q_out
                    + (token_idx * q_head_padded + slot_idx) * HEAD_DIM
                    + NOPE_DIM
                    + offset_rope,
                    qkv_blk_rope.to(tl.bfloat16),
                )  # [ROPE_DIM]
            else:
                rope_lane_offs = NOPE_DIM // ELEMS_PER_LANE + tl.arange(
                    0, ROPE_DIM // ELEMS_PER_LANE
                )
                new_lane_offs = (slot_idx * WARP_SIZE + rope_lane_offs) // num_heads
                new_slot_offs = (slot_idx * WARP_SIZE + rope_lane_offs) % num_heads
                tl.store(
                    q_out
                    + token_idx * q_head_padded * HEAD_DIM
                    + (new_lane_offs * q_head_padded + new_slot_offs)[:, None]
                    * ELEMS_PER_LANE
                    + elem_offs[None, :],
                    qkv_blk_rope.reshape(ROPE_DIM // ELEMS_PER_LANE, ELEMS_PER_LANE).to(
                        tl.bfloat16
                    ),
                )
            return
    else:
        # unreached branch
        return
    # load slot
    slot_id = tl.load(slot_mapping + token_idx)  # i64
    if slot_id < 0:
        return
    qkv_blk_rope = qkv_blk_rope.to(tl.bfloat16)
    block_idx = slot_id // cache_block_size
    pos_in_block = slot_id % cache_block_size
    block_base = k_cache + block_idx * kv_block_stride
    if kv_mxfp8:
        _deepseek_v41_kv_quant(
            qkv_blk_rope, kv_base, block_base, pos_in_block, cache_block_size
        )
    else:
        _deepseek_v4_kv_quant(
            qkv_blk_rope, kv_base, block_base, pos_in_block, cache_block_size
        )


@triton.jit
def _tiled_kernel(
    q,
    q_out,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    q_head_padded: tl.constexpr,
    eps,
    cache_block_size: tl.constexpr,
    apply_q_norm: tl.constexpr,
    kv_mxfp8: tl.constexpr,
    apply_q_rope: tl.constexpr,
    is_q_interleaved: tl.constexpr,
    num_tokens,
    num_heads: tl.constexpr,
    kv_block_stride,
    num_tokens_insert,
    TPP: tl.constexpr,  # tokens per program
):
    HEAD_DIM: tl.constexpr = 512
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    HALF_ROPE_DIM: tl.constexpr = 32
    FP8_MAX: tl.constexpr = 448.0
    WARP_SIZE: tl.constexpr = 32
    ELEMS_PER_LANE: tl.constexpr = 16
    off = tl.arange(0, HEAD_DIM)
    lane_offs = tl.arange(0, WARP_SIZE)
    elem_offs = tl.arange(0, ELEMS_PER_LANE)

    pid = tl.program_id(0).to(tl.int64)
    blocks_per_token: tl.constexpr = q_head_padded + 1

    # grid = cdiv(num_tokens, TPP) * blocks_per_token
    tile = pid // blocks_per_token
    slot_idx = pid % blocks_per_token
    is_kv = slot_idx == q_head_padded
    is_pad_q = (not is_kv) and (slot_idx >= num_heads)

    tok = tile * TPP + tl.arange(0, TPP).to(tl.int64)  # [TPP]
    tok_ok = tok < num_tokens
    if is_pad_q:
        if not is_q_interleaved:
            tl.store(
                q_out
                + (tok[:, None] * q_head_padded + slot_idx) * HEAD_DIM
                + off[None, :],
                0,
                mask=tok_ok[:, None],
            )
        else:
            tl.store(
                q_out
                + tok[:, None, None] * q_head_padded * HEAD_DIM
                + (lane_offs[None, :, None] * q_head_padded + slot_idx) * ELEMS_PER_LANE
                + off[None, None, :],
                0,
                mask=tok_ok[:, None, None],
            )
        return

    off_rope = tl.arange(0, ROPE_DIM)
    off_half = tl.arange(0, HALF_ROPE_DIM)

    # cos/sin are needed by both paths
    pos = tl.load(position_ids + tok, mask=tok_ok, other=0)  # [TPP]
    cs_base = cos_sin_cache + pos[:, None] * ROPE_DIM
    cos_blk = tl.load(cs_base + off_half[None, :], mask=tok_ok[:, None], other=0.0)
    sin_blk = tl.load(
        cs_base + (off_half + HALF_ROPE_DIM)[None, :], mask=tok_ok[:, None], other=0.0
    )

    if not is_kv:
        # ── Q: per-head RMSNorm (no weight) + GPT-J RoPE, in place ──
        q_base = q + (tok * num_heads + slot_idx) * HEAD_DIM  # [TPP]
        q_blk = tl.load(
            q_base[:, None] + off[None, :], mask=tok_ok[:, None], other=0.0
        ).to(tl.float32)
        if apply_q_norm:
            variance = tl.sum(q_blk * q_blk, axis=1) / HEAD_DIM  # [TPP]
            rsqrt = tl.rsqrt(variance + eps)
            q_blk = q_blk * rsqrt[:, None]
        if not is_q_interleaved:
            tl.store(
                q_out
                + (tok[:, None] * q_head_padded + slot_idx) * HEAD_DIM
                + off[None, :],
                q_blk.to(tl.bfloat16),
                mask=tok_ok[:, None] & (off[None, :] < NOPE_DIM),
            )
        else:
            new_lane_offs = (slot_idx * WARP_SIZE + lane_offs) // num_heads
            new_slot_offs = (slot_idx * WARP_SIZE + lane_offs) % num_heads
            tl.store(
                q_out
                + tok[:, None, None] * q_head_padded * HEAD_DIM
                + (new_lane_offs * q_head_padded + new_slot_offs)[None, :, None]
                * ELEMS_PER_LANE
                + elem_offs[None, None, :],
                q_blk.reshape(TPP, WARP_SIZE, ELEMS_PER_LANE).to(tl.bfloat16),
                mask=tok_ok[:, None, None]
                & (
                    True
                    if not apply_q_rope
                    else (lane_offs[None, :, None] < (NOPE_DIM // ELEMS_PER_LANE))
                ),
            )
        if not apply_q_rope:
            return
        rope = tl.load(
            q_base[:, None] + NOPE_DIM + off_rope[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        ).to(tl.float32)
        if apply_q_norm:
            rope *= rsqrt[:, None]
    else:
        kv_base = kv + tok * HEAD_DIM
        rope = tl.load(
            kv_base[:, None] + NOPE_DIM + off_rope[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        ).to(tl.float32)

    # ── GPT-J interleaved RoPE on the trailing ROPE_DIM ──
    rope = tl.reshape(rope, TPP, HALF_ROPE_DIM, 2)
    even, odd = tl.split(rope)  # [TPP, HALF_ROPE_DIM]
    new_even = even * cos_blk - odd * sin_blk
    new_odd = even * sin_blk + odd * cos_blk
    rope = tl.reshape(tl.join(new_even, new_odd), TPP, ROPE_DIM).to(tl.bfloat16)

    if not is_kv:
        if not is_q_interleaved:
            tl.store(
                q_out
                + (tok[:, None] * q_head_padded + slot_idx) * HEAD_DIM
                + NOPE_DIM
                + off_rope[None, :],
                rope,
                mask=tok_ok[:, None],
            )
        else:
            rope_lane_offs = NOPE_DIM // ELEMS_PER_LANE + tl.arange(
                0, ROPE_DIM // ELEMS_PER_LANE
            )
            new_lane_offs = (slot_idx * WARP_SIZE + rope_lane_offs) // num_heads
            new_slot_offs = (slot_idx * WARP_SIZE + rope_lane_offs) % num_heads
            tl.store(
                q_out
                + tok[:, None, None] * q_head_padded * HEAD_DIM
                + (new_lane_offs * q_head_padded + new_slot_offs)[None, :, None]
                * ELEMS_PER_LANE
                + elem_offs[None, None, :],
                rope.reshape(TPP, ROPE_DIM // ELEMS_PER_LANE, ELEMS_PER_LANE),
                mask=tok_ok[:, None, None],
            )
        return

    # ── KV: RoPE already applied; UE8M0 FP8 quant + paged cache insert ──
    kv_base = kv + tok * HEAD_DIM
    ins_ok = tok_ok & (tok < num_tokens_insert)
    slot_id = tl.load(slot_mapping + tok, mask=ins_ok, other=-1)  # [TPP]
    ins_ok = ins_ok & (slot_id >= 0)

    block_idx = slot_id // cache_block_size
    pos_in_block = slot_id % cache_block_size
    block_base = block_idx * kv_block_stride
    if not kv_mxfp8:
        QUANT_BLOCK: tl.constexpr = 64
        NUM_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK  # 7
        SCALE_BYTES_PER_TOKEN: tl.constexpr = NUM_QUANT_BLOCKS + 1  # 8
        TOKEN_DATA_BYTES: tl.constexpr = NOPE_DIM + 2 * ROPE_DIM  # 576
        off_quant = tl.arange(0, QUANT_BLOCK)

        token_fp8 = block_base + pos_in_block * TOKEN_DATA_BYTES  # [TPP] byte offset
        token_scale = (
            block_base
            + cache_block_size * TOKEN_DATA_BYTES
            + pos_in_block * SCALE_BYTES_PER_TOKEN
        )

        bf16_ptr = (k_cache + token_fp8 + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        tl.store(bf16_ptr[:, None] + off_rope[None, :], rope, mask=ins_ok[:, None])

        for b in tl.static_range(NUM_QUANT_BLOCKS):
            blk = tl.load(
                kv_base[:, None] + b * QUANT_BLOCK + off_quant[None, :],
                mask=ins_ok[:, None],
                other=0.0,
            ).to(tl.float32)
            block_max = tl.maximum(tl.max(tl.abs(blk), axis=1), 1e-4)  # [TPP]
            exponent = tl.ceil(tl.log2(block_max / FP8_MAX))
            scale = tl.exp2(exponent)
            x = tl.clamp(blk / scale[:, None], -FP8_MAX, FP8_MAX)
            tl.store(
                k_cache + token_fp8[:, None] + b * QUANT_BLOCK + off_quant[None, :],
                x.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
                mask=ins_ok[:, None],
            )
            enc = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0)
            tl.store(k_cache + token_scale + b, enc.to(tl.uint8), mask=ins_ok)

        tl.store(
            k_cache + token_scale + NUM_QUANT_BLOCKS,
            tl.zeros((TPP,), dtype=tl.uint8),
            mask=ins_ok,
        )
    else:
        _tiled_deepseek_v41_kv_quant(
            rope, kv_base, block_base, pos_in_block, cache_block_size, TPP, ins_ok
        )


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_head_padded: int,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
    kv_mxfp8: bool = False,
    apply_q_rope: bool = True,
    is_q_interleaved: bool = False,
):
    """
    Horizontally-fused DeepseekV4-MLA:
      - Q side:  optional per-head RMSNorm (apply_q_norm) and GPT-J RoPE on the
                 last ROPE_DIM (apply_q_rope), plus the zero-pad to
                 q_head_padded, in the head-major layout or (is_q_interleaved)
                 FlashMLA's mega-attention chunk-interleaved one
      - KV side: GPT-J RoPE on last ROPE_DIM + UE8M0 FP8 quant on NoPE + paged
                 cache insert

    Assumptions (hard-coded for DeepseekV4 attention):
      HEAD_DIM  = 512
      ROPE_DIM  = 64   (RoPE applied to dims [NOPE_DIM, HEAD_DIM))
      NOPE_DIM  = 448
      QUANT_BLOCK = 64 (UE8M0 FP8 quant block)
      FP8_MAX   = 224.0f on ROCm FNUZ / 448.0f on OCP
      is_neox=false (GPT-J interleaved pairs)
      cos_sin_cache layout [max_pos, rope_dim] = cos || sin (cos first, sin
        second along last dim; each half is rope_dim/2 = 32 values)

    Cache layout per paged-cache block (block_size tokens), selected by the
    `kv_mxfp8` argument:
      V4 (584 B/token, kv_mxfp8=false)
        [0,      bs*576):          token data, 448 fp8 + 128 bf16 each
        [bs*576, bs*576 + bs*8):   UE8M0 scales, 7 real + 1 pad per token
      V4.1 (528 B/token, kv_mxfp8=true) -- RoPE is quantized as well, so the
      whole row is MXFP8 (fp8 e4m3 with one UE8M0 scale per 32 dims)
        [0,      bs*512):          token data, 512 fp8 each
        [bs*512, bs*512 + bs*16):  UE8M0 scales, 16 per token

    Args:
        q: [N, num_heads_q, 512], bf16
        kv: [N, 512], bf16, read-only
        k_cache: [num_blocks, block_bytes], uint8
        slot_mapping: [num_tokens_insert], i64
        position_ids: [N], i64
        cos_sin_cache: [max_pos, 64], fp32
        q_head_padded: i32
        eps: used in RMSNorm, fp32
        cache_block_size: tokens per paged-cache block, i32
        apply_q_norm: optional RMSNorm for q, bool, default True
        kv_mxfp8: True for V4.1 all-dims MXFP8 record, bool, default False
        apply_q_rope: rotate Q too, not just KV, bool, default True
        is_q_interleaved: is FlashMLA's mega-attention layout, bool, default False

    Return:
        q_out: [N, q_head_padded, 512] or [0], bf16
    """
    assert q.is_contiguous() and kv.is_contiguous()
    num_tokens, num_heads, head_dims = q.shape
    assert head_dims == 512
    assert kv.shape == (num_tokens, 512)
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert k_cache.dtype == torch.uint8
    assert slot_mapping.dim() == 1
    num_tokens_insert = slot_mapping.shape[0]
    assert num_tokens_insert <= num_tokens
    assert slot_mapping.dtype == torch.int64
    assert position_ids.shape == (num_tokens,)
    assert position_ids.dtype == torch.int64
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64
    assert cos_sin_cache.dtype == torch.float32
    assert q_head_padded in (0, 8, 16, 32, 64, 128)

    if q_head_padded == 0:
        q_out = q.new_empty([0])
    else:
        q_out = q.new_empty([num_tokens, q_head_padded, 512])

    grid = num_tokens * (q_head_padded + 1)
    fused_qnorm_rope_kv_insert_kernel[(grid,)](
        q,
        q_out,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        q_head_padded,
        eps,
        cache_block_size,
        apply_q_norm,
        kv_mxfp8,
        apply_q_rope,
        is_q_interleaved,
        num_tokens,
        num_heads,
        k_cache.stride(0),
        num_tokens_insert,
        num_warps=1,
        num_stages=2,
    )
    return q_out


def _fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_impl(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_head_padded: int,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
    kv_mxfp8: bool = False,
    apply_q_rope: bool = True,
    is_q_interleaved: bool = False,
    TPP: int = 8,
    num_warps: int = 4,
):
    """Tiled version shared for hygon, metax, mthread"""
    assert q.is_contiguous() and kv.is_contiguous()
    num_tokens, num_heads, head_dims = q.shape
    assert head_dims == 512
    assert kv.shape == (num_tokens, 512)
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert k_cache.dtype == torch.uint8
    assert slot_mapping.dim() == 1
    num_tokens_insert = slot_mapping.shape[0]
    assert num_tokens_insert <= num_tokens
    assert slot_mapping.dtype == torch.int64
    assert position_ids.shape == (num_tokens,)
    assert position_ids.dtype == torch.int64
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64
    assert cos_sin_cache.dtype == torch.float32
    assert q_head_padded in (0, 8, 16, 32, 64, 128)

    if q_head_padded == 0:
        q_out = q.new_empty([0])
    else:
        q_out = q.new_empty([num_tokens, q_head_padded, 512])

    grid = triton.cdiv(num_tokens, TPP) * (q_head_padded + 1)
    _tiled_kernel[(grid,)](
        q,
        q_out,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        q_head_padded,
        eps,
        cache_block_size,
        apply_q_norm,
        kv_mxfp8,
        apply_q_rope,
        is_q_interleaved,
        num_tokens,
        num_heads,
        k_cache.stride(0),
        num_tokens_insert,
        TPP=TPP,
        num_warps=num_warps,
        num_stages=2,
    )
    return q_out

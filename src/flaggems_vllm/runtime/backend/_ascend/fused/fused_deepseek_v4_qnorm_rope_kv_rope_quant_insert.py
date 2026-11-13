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
"""Ascend 910B override for the fused DeepSeek-V4 qnorm/RoPE/quant/insert kernel.

The arithmetic is the generic implementation's. What differs, and why:

1. FP8 conversion goes through `_f32_to_e4m3_bits`: BiShengIR has no `f8E4M3FN`.
2. The UE8M0 scale is read off the exponent bits by `_ue8m0_scale`: AICore has no
   scalar float-to-integer conversion.
3. The bf16 half of each cache token is reached through a bfloat16 view passed in
   by the wrapper: Triton here cannot bitcast a pointer's element type.
4. RoPE pairs are addressed with a 2-D offset: `[ROPE_DIM] -> [HALF_ROPE_DIM, 2]`
   does not survive the shape pipeline.
5. Q and KV share one kernel and one launch, since launches are costly here.
6. Every launch is at most one core group or a whole number of them; see
   `launch_group_size`.

Points 2-4 fail to compile, with misleading errors, if written the generic way.
"""

import functools

import torch
import triton
import triton.language as tl

# The runtime rejects a launch whose program count exceeds this, reporting it as
# an invalid `coreDim`. It is a launch-API limit, not a hardware occupancy one.
MAX_PROGRAMS_PER_LAUNCH = 65535


@functools.lru_cache(maxsize=None)
def launch_group_size(device_index: int) -> int:
    """Programs per launch must be at most this, or a multiple of it.

    Otherwise the runtime executes program 0 more than once, and this operator
    writes q in place.
    """
    import torch_npu  # noqa: F401

    # get_device_limit needs an initialised device context, which entering
    # torch.npu.device does not create when the index is already current.
    with torch.npu.device(device_index):
        torch.npu.set_device(device_index)
        return int(torch.npu.get_device_limit(device_index)["vector_core_num"])


@functools.lru_cache(maxsize=None)
def launch_geometry(device_index: int) -> tuple:
    """(group size, chunk step): the step is the largest whole number of groups
    within MAX_PROGRAMS_PER_LAUNCH. Cached, since the wrapper runs per call."""
    group = launch_group_size(device_index)
    return group, MAX_PROGRAMS_PER_LAUNCH // group * group


# Most heads of one token that a Q program may take, as an [H, HEAD_DIM] tile.
# 64 overflows the Unified Buffer and does not compile.
Q_MAX_HEADS_PER_PROGRAM = 32


@triton.jit
def _f32_to_e4m3_bits(x):
    """float32 -> OCP E4M3 (float8_e4m3fn) bit pattern, round-to-nearest-even.

    **Requires |x| <= 448**: there is no saturation branch, so a larger input
    silently yields a wrong byte. The caller guarantees it by construction; restore
    saturation before using this anywhere else. Subnormals are handled.
    """
    b = x.to(tl.int32, bitcast=True)
    sign = (b >> 24) & 0x80
    mag = b & 0x7FFFFFFF
    e = (mag >> 23) - 120

    m_n = (mag >> 20) & 0x7
    round_n = (mag >> 19) & 1
    sticky_n = (mag & 0x7FFFF) != 0
    m_n = m_n + tl.where((round_n == 1) & (sticky_n | ((m_n & 1) == 1)), 1, 0)
    e_n = e + tl.where(m_n > 7, 1, 0)
    m_n = tl.where(m_n > 7, 0, m_n)

    # A subnormal is m * 2^-9 for m in 1..7, so its mantissa is |x| * 512 rounded
    # to nearest even; adding and removing 2^23 forces that rounding without a
    # per-lane shift, which this backend scalarises.
    magic: tl.constexpr = 8388608.0  # 2^23
    m_s = ((tl.abs(x) * 512.0 + magic) - magic).to(tl.int32)

    v = tl.where(e >= 1, (e_n << 3) | m_n, m_s)
    return (sign | v).to(tl.uint8)


@triton.jit
def _ue8m0_scale(raw_scale):
    """UE8M0 scale 2^ceil(log2(raw_scale)) and its encoded byte, read off the exponent
    bits: AICore's scalar unit has no float-to-integer conversion, so `.to(tl.uint8)`
    on the per-block scale does not compile. raw_scale is always positive here.
    """
    bits = raw_scale.to(tl.int32, bitcast=True)
    biased_exp = (bits >> 23) & 0xFF
    # Ceil rather than floor: any set mantissa bit means the next power of two.
    code = biased_exp + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    scale = (code << 23).to(tl.float32, bitcast=True)
    stored = tl.minimum(tl.maximum(code, 0), 255)
    return scale, stored


def q_heads_per_program(num_heads: int, is_q_interleaved: bool) -> int:
    """Heads of one token per program: the largest power of two, at most 32, that
    divides num_heads. Every tile is then full, so no mask is needed; this backend
    faults on an out-of-bounds address rather than honouring a mask.
    """
    h = 1
    cap = min(
        (
            Q_MAX_HEADS_PER_PROGRAM
            if not is_q_interleaved
            else Q_MAX_HEADS_PER_PROGRAM // 2
        ),
        num_heads,
    )
    while h * 2 <= cap and num_heads % (h * 2) == 0:
        h *= 2
    return h


@triton.jit
def _kv_fp8_quant_2d(
    kv_quant_blk,  # [2, 32], fp32
    out_data_ptrs,  # [2, 32]
    out_scale_ptrs,  # [2, 1]
):
    FP8_MAX: tl.constexpr = 448.0
    block_max = tl.maximum(tl.max(tl.abs(kv_quant_blk), axis=-1, keep_dims=True), 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX)), off the exponent bits
    scale, scale_code = _ue8m0_scale(block_max / FP8_MAX)
    # No clamp: scale is the smallest power of two with block_max / scale <=
    # FP8_MAX, so |x| <= FP8_MAX holds by construction.
    x_scaled = kv_quant_blk / scale
    x_uint8 = _f32_to_e4m3_bits(x_scaled)
    tl.store(out_data_ptrs, x_uint8)
    tl.store(out_scale_ptrs, scale_code.to(tl.uint8))


@triton.jit
def _deepseek_v41_kv_quant(
    qkv_blk_rope,  # [HALF_ROPE_DIM, 2], bf16
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
    for i in tl.static_range(NOPE_QUANT_BLOCKS // ROPE_QUANT_BLOCKS):
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
def fused_qnorm_rope_kv_insert_kernel(
    q,
    q_out,
    kv,
    k_cache,
    k_cache_bf16,
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
    num_heads,
    kv_block_stride,
    pid_offset,
    q_programs,
    total_programs,
    tiles_per_token,
    H: tl.constexpr,
    HAS_PADDING: tl.constexpr,
):
    """Both halves of the operator in one launch, selected per program.

    Programs [0, q_programs) each normalise and rotate H heads of one token; the
    rest each handle one token's KV. A chunk may straddle the boundary, since every
    program classifies itself from its global id.

    Two compiler constraints shape the code:
    - no early `return` inside either arm of the two-armed `if`; it aborts the
      compiler, so guards are written as conditions.
    - the arms share no local name: Triton unifies a name assigned in both arms into
      one value of one type, so KV locals are prefixed `kv_`.
    """
    HEAD_DIM: tl.constexpr = 512
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    HALF_ROPE_DIM: tl.constexpr = 32
    QUANT_BLOCK: tl.constexpr = 64
    NUM_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK  # 7
    SCALE_BYTES_PER_TOKEN: tl.constexpr = NUM_QUANT_BLOCKS + 1  # 8 (7 real + 1 pad)
    TOKEN_DATA_BYTES: tl.constexpr = NOPE_DIM + 2 * ROPE_DIM  # 576
    FP8_MAX: tl.constexpr = 448.0
    WARP_SIZE: tl.constexpr = 32
    ELEMS_PER_LANE: tl.constexpr = 16
    lane_offs = tl.arange(0, WARP_SIZE)
    elem_offs = tl.arange(0, ELEMS_PER_LANE)

    # The last chunk is padded to a whole number of core groups; programs at or past
    # total_programs must do nothing, hence the `elif` guard below.
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    if pid < q_programs:
        # ---- Q: RMSNorm without weight, then GPT-J RoPE, for H heads of ONE token.
        # Tiling heads within a token keeps position_id, cos and sin scalar. No loop,
        # device function or early return: each aborts ttir_to_linalg here.
        token_idx = pid // tiles_per_token
        head_base = (pid % tiles_per_token) * H
        slot_idx = head_base + tl.arange(0, H)
        rows = token_idx * num_heads + slot_idx

        col = tl.arange(0, HEAD_DIM)
        if not HAS_PADDING:
            blk = tl.load(q + rows[:, None] * HEAD_DIM + col[None, :]).to(tl.float32)
        else:
            is_q_pad = slot_idx >= num_heads
            blk = tl.load(
                q + rows[:, None] * HEAD_DIM + col[None, :],
                mask=(not is_q_pad)[:, None],
                other=0,
            ).to(tl.float32)

        if apply_q_norm:
            variance = tl.sum(blk * blk, axis=1) / HEAD_DIM
            rsqrt = tl.rsqrt(variance + eps)
            blk = blk * rsqrt[:, None]
        if not is_q_interleaved:
            if not HAS_PADDING:
                tl.store(
                    q_out + rows[:, None] * HEAD_DIM + col[None, :],
                    blk.to(tl.bfloat16),
                    mask=True if not apply_q_rope else (col[None, :] < NOPE_DIM),
                )
            else:
                tl.store(
                    q_out
                    + (token_idx * q_head_padded + slot_idx[:, None]) * HEAD_DIM
                    + col[None, :],
                    blk.to(tl.bfloat16),
                    mask=True if not apply_q_rope else (col[None, :] < NOPE_DIM),
                )
        else:
            if not HAS_PADDING:
                new_lane_offs = (
                    slot_idx[:, None] * WARP_SIZE + lane_offs[None, :]
                ) // num_heads
                new_slot_offs = (
                    slot_idx[:, None] * WARP_SIZE + lane_offs[None, :]
                ) % num_heads
            else:
                new_lane_offs1 = (
                    slot_idx[:, None] * WARP_SIZE + lane_offs[None, :]
                ) // num_heads
                new_slot_offs1 = (
                    slot_idx[:, None] * WARP_SIZE + lane_offs[None, :]
                ) % num_heads
                new_lane_offs2 = (
                    slot_idx[:, None] * WARP_SIZE
                    + lane_offs[None, :]
                    - WARP_SIZE * num_heads
                ) // (q_head_padded - num_heads)
                new_slot_offs2 = (
                    slot_idx[:, None] * WARP_SIZE
                    + lane_offs[None, :]
                    - WARP_SIZE * num_heads
                ) % (q_head_padded - num_heads) + num_heads
                new_lane_offs = tl.where(
                    not is_q_pad[:, None], new_lane_offs1, new_lane_offs2
                )
                new_slot_offs = tl.where(
                    not is_q_pad[:, None], new_slot_offs1, new_slot_offs2
                )
            tl.store(
                q_out
                + token_idx * q_head_padded * HEAD_DIM
                + (new_lane_offs * q_head_padded + new_slot_offs)[:, :, None]
                * ELEMS_PER_LANE
                + elem_offs[None, None, :],
                blk.reshape(H, WARP_SIZE, ELEMS_PER_LANE).to(tl.bfloat16),
                mask=(
                    True
                    if not apply_q_rope
                    else (lane_offs[None, :, None] < (NOPE_DIM // ELEMS_PER_LANE))
                ),
            )
        if apply_q_rope:
            position_id = tl.load(position_ids + token_idx)  # scalar: one per token
            half = tl.arange(0, HALF_ROPE_DIM)
            cos_blk = tl.load(cos_sin_cache + position_id * ROPE_DIM + half)
            sin_blk = tl.load(
                cos_sin_cache + position_id * ROPE_DIM + HALF_ROPE_DIM + half
            )

            if not is_q_interleaved:
                # [H, HALF_ROPE_DIM, 2]: the pair axis is broadcast into existence, as
                # on the KV path, so no reshape is needed.
                if not HAS_PADDING:
                    pair_off = (
                        rows[:, None, None] * HEAD_DIM
                        + NOPE_DIM
                        + half[None, :, None] * 2
                        + tl.arange(0, 2)[None, None, :]
                    )
                    pair = tl.load(q + pair_off).to(tl.float32)
                else:
                    pair = tl.load(
                        q
                        + rows[:, None, None] * HEAD_DIM
                        + NOPE_DIM
                        + half[None, :, None] * 2
                        + tl.arange(0, 2)[None, None, :],
                        mask=(not is_q_pad)[:, None, None],
                        other=0,
                    ).to(tl.float32)
                even_blk, odd_blk = tl.split(pair)
                if apply_q_norm:
                    even_blk = even_blk * rsqrt[:, None]
                    odd_blk = odd_blk * rsqrt[:, None]
                new_even_blk = even_blk * cos_blk[None, :] - odd_blk * sin_blk[None, :]
                new_odd_blk = even_blk * sin_blk[None, :] + odd_blk * cos_blk[None, :]
                pair = tl.join(new_even_blk, new_odd_blk).to(tl.bfloat16)
                if not HAS_PADDING:
                    tl.store(q_out + pair_off, pair)
                else:
                    tl.store(
                        q_out
                        + (token_idx * q_head_padded + slot_idx[:, None, None])
                        * HEAD_DIM
                        + NOPE_DIM
                        + half[None, :, None] * 2
                        + tl.arange(0, 2)[None, None, :],
                        pair,
                    )
            else:
                pair = tl.load(
                    q
                    + rows[:, None] * HEAD_DIM
                    + NOPE_DIM
                    + tl.arange(0, ROPE_DIM)[None, None, :],
                    mask=True if not HAS_PADDING else (not is_q_pad)[:, None],
                    other=0,
                )
                pair = pair.reshape(H, HALF_ROPE_DIM, 2)
                even_blk, odd_blk = tl.split(pair.to(tl.float32))
                if apply_q_norm:
                    even_blk = even_blk * rsqrt[:, None]
                    odd_blk = odd_blk * rsqrt[:, None]
                new_even_blk = even_blk * cos_blk[None, :] - odd_blk * sin_blk[None, :]
                new_odd_blk = even_blk * sin_blk[None, :] + odd_blk * cos_blk[None, :]
                pair = tl.join(new_even_blk, new_odd_blk).to(tl.bfloat16)

                rope_lane_offs = NOPE_DIM // ELEMS_PER_LANE + tl.arange(
                    0, ROPE_DIM // ELEMS_PER_LANE
                )
                if not HAS_PADDING:
                    new_rope_lane_offs = (
                        slot_idx[:, None] * WARP_SIZE + rope_lane_offs[None, :]
                    ) // num_heads
                    new_rope_slot_offs = (
                        slot_idx[:, None] * WARP_SIZE + rope_lane_offs[None, :]
                    ) % num_heads
                else:
                    new_rope_lane_offs1 = (
                        slot_idx[:, None] * WARP_SIZE + rope_lane_offs[None, :]
                    ) // num_heads
                    new_rope_slot_offs1 = (
                        slot_idx[:, None] * WARP_SIZE + rope_lane_offs[None, :]
                    ) % num_heads
                    new_rope_lane_offs2 = (
                        slot_idx[:, None] * WARP_SIZE
                        + rope_lane_offs[None, :]
                        - WARP_SIZE * num_heads
                    ) // (q_head_padded - num_heads)
                    new_rope_slot_offs2 = (
                        slot_idx[:, None] * WARP_SIZE
                        + rope_lane_offs[None, :]
                        - WARP_SIZE * num_heads
                    ) % (q_head_padded - num_heads) + num_heads
                    new_rope_lane_offs = tl.where(
                        not is_q_pad[:, None], new_rope_lane_offs1, new_rope_lane_offs2
                    )
                    new_rope_slot_offs = tl.where(
                        not is_q_pad[:, None], new_rope_slot_offs1, new_rope_slot_offs2
                    )
                tl.store(
                    q_out
                    + token_idx * q_head_padded * HEAD_DIM
                    + (new_rope_lane_offs * q_head_padded + new_rope_slot_offs)[
                        :, :, None
                    ]
                    * ELEMS_PER_LANE
                    + elem_offs[None, None, :],
                    pair.reshape(H, ROPE_DIM // ELEMS_PER_LANE, ELEMS_PER_LANE),
                )
    elif pid < total_programs:
        # ---- KV: GPT-J RoPE on the last 64, then UE8M0 FP8 quantisation of the
        # NoPE region and the paged-cache insert.
        kv_token = pid - q_programs
        kv_base = kv + kv_token * HEAD_DIM
        offset_half_rope = tl.arange(0, HALF_ROPE_DIM)
        offset_quant = tl.arange(0, QUANT_BLOCK)
        # RoPE pairs via a 2-D offset: reshaping [ROPE_DIM] to [HALF_ROPE_DIM, 2] and
        # 1-D stride-2 offsets both fail to compile here.
        offset_pair = offset_half_rope[:, None] * 2 + tl.arange(0, 2)[None, :]

        qkv_blk_rope = tl.load(kv_base + NOPE_DIM + offset_pair).to(tl.float32)
        kv_position = tl.load(position_ids + kv_token)  # i64
        cs_base = cos_sin_cache + kv_position * ROPE_DIM
        kv_cos = tl.load(cs_base + offset_half_rope)  # [HALF_ROPE_DIM], f32
        kv_sin = tl.load(cs_base + offset_half_rope + HALF_ROPE_DIM)
        kv_even, kv_odd = tl.split(qkv_blk_rope)  # [HALF_ROPE_DIM], f32
        kv_new_even = kv_even * kv_cos - kv_odd * kv_sin
        kv_new_odd = kv_even * kv_sin + kv_odd * kv_cos
        qkv_blk_rope = tl.join(kv_new_even, kv_new_odd).to(tl.bfloat16)

        kv_slot = tl.load(slot_mapping + kv_token)  # i64
        if kv_slot >= 0:  # a negative slot is padding
            block_idx = kv_slot // cache_block_size
            pos_in_block = kv_slot % cache_block_size
            block_base = k_cache + block_idx * kv_block_stride
            if not kv_mxfp8:
                token_fp8_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES
                # Pointers cannot change element width here, so the RoPE region is reached
                # through a bf16 view; every byte offset is even.
                token_bf16_idx = (
                    block_idx * kv_block_stride
                    + pos_in_block * TOKEN_DATA_BYTES
                    + NOPE_DIM
                ) // 2
                token_scale_ptr = (
                    block_base
                    + cache_block_size * TOKEN_DATA_BYTES
                    + pos_in_block * SCALE_BYTES_PER_TOKEN
                )
                tl.store(
                    k_cache_bf16 + token_bf16_idx + offset_pair, qkv_blk_rope
                )  # [HALF_ROPE_DIM, 2]
                # Quantisation of the KV NoPE region: all seven groups as one [8, QUANT_BLOCK]
                # tile. The eighth row is the token's RoPE segment, computed and discarded; both
                # stores mask it off.
                gidx = tl.arange(0, 8)
                keep_group = gidx < NUM_QUANT_BLOCKS
                kv_quant_blk = tl.load(
                    kv_base + gidx[:, None] * QUANT_BLOCK + offset_quant[None, :]
                ).to(tl.float32)
                block_max = tl.maximum(tl.max(tl.abs(kv_quant_blk), axis=1), 1e-4)
                # scale = 2^ceil(log2(block_max / FP8_MAX)), off the exponent bits
                scale, scale_code = _ue8m0_scale(block_max / FP8_MAX)
                # No clamp: scale is the smallest power of two with block_max / scale <=
                # FP8_MAX, so |x| <= FP8_MAX holds by construction.
                x_scaled = kv_quant_blk / scale[:, None]
                x_uint8 = _f32_to_e4m3_bits(x_scaled)
                tl.store(
                    token_fp8_ptr + gidx[:, None] * QUANT_BLOCK + offset_quant[None, :],
                    x_uint8,
                    mask=keep_group[:, None],
                )
                # store scale: stored_value = exponent + 127 (bias)
                tl.store(
                    token_scale_ptr + gidx, scale_code.to(tl.uint8), mask=keep_group
                )
                tl.store(
                    token_scale_ptr + NUM_QUANT_BLOCKS, tl.zeros((), dtype=tl.uint8)
                )
            else:
                _deepseek_v41_kv_quant(
                    qkv_blk_rope, kv_base, block_base, pos_in_block, cache_block_size
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
    """See the generic implementation for the layout contract."""
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

    assert k_cache.is_contiguous()
    k_cache_bf16 = k_cache.view(torch.bfloat16)

    # One grid: Q tiles first, then one program per inserted token. It is issued in
    # chunks within MAX_PROGRAMS_PER_LAUNCH, each program adding its chunk's offset,
    # and every launch is a whole number of core groups.

    heads_per_program = q_heads_per_program(q_head_padded, is_q_interleaved)
    tiles_per_token = q_head_padded // heads_per_program
    q_programs = num_tokens * tiles_per_token
    total_programs = q_programs + num_tokens_insert
    device_index = q.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    group, step = launch_geometry(device_index)
    for pid_offset in range(0, total_programs, step):
        grid = min(step, total_programs - pid_offset)
        if grid > group:
            # Integer arithmetic: triton.cdiv is slow on this host.
            grid = -(-grid // group) * group
        fused_qnorm_rope_kv_insert_kernel[(grid,)](
            q,
            q_out,
            kv,
            k_cache,
            k_cache_bf16,
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
            num_heads,
            k_cache.stride(0),
            pid_offset,
            q_programs,
            total_programs,
            tiles_per_token,
            heads_per_program,
            HAS_PADDING=q_head_padded > num_heads,
            num_warps=1,
            num_stages=1,
        )
    return q_out

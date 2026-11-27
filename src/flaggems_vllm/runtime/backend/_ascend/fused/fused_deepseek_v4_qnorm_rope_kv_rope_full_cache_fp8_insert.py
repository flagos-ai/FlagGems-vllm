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

import functools

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

MAX_PROGRAMS_PER_LAUNCH = 65535
Q_MAX_HEADS_PER_PROGRAM = 16
Q_PROGRAMS_PER_VECTOR_CORE = 8


if has_triton_tle(3, 5, 0):
    try:
        import triton.experimental.tle as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False


@functools.lru_cache(maxsize=None)
def launch_group_size(device_index: int) -> int:
    """Return the vector-core group size required by the launch runtime."""
    import torch_npu  # noqa: F401

    with torch.npu.device(device_index):
        torch.npu.set_device(device_index)
        return int(torch.npu.get_device_limit(device_index)["vector_core_num"])


@functools.lru_cache(maxsize=None)
def launch_geometry(device_index: int) -> tuple:
    """Return (group size, maximum whole-group programs per launch)."""
    group = launch_group_size(device_index)
    return group, MAX_PROGRAMS_PER_LAUNCH // group * group


def q_heads_per_program(num_heads: int) -> int:
    """Largest power-of-two divisor of num_heads, capped for UB usage."""
    h = 1
    cap = min(Q_MAX_HEADS_PER_PROGRAM, num_heads)
    while h * 2 <= cap and num_heads % (h * 2) == 0:
        h *= 2
    return h


@triton.jit
def _f32_to_e4m3_bits(x):
    """Encode a non-NaN FP32 value as saturated OCP E4M3 with RNE."""
    FP8_MAX_F32_BITS: tl.constexpr = 0x43E00000
    NORMAL_ROUNDING_BIAS: tl.constexpr = 0x0007FFFF - 0x3C000000

    b = x.to(tl.int32, bitcast=True)
    a = tl.minimum(b & 0x7FFFFFFF, FP8_MAX_F32_BITS)
    t = a + NORMAL_ROUNDING_BIAS
    t += (a >> 20) & 1
    r_norm = t >> 20
    r_sub = (a.to(tl.float32, bitcast=True) * 512.0 + 8388608.0).to(
        tl.int32, bitcast=True
    ) - 0x4B000000
    r = tl.maximum(r_norm, tl.minimum(r_sub, 8))
    return (r | ((b >> 24) & 0x80)).to(tl.uint8)


if HAS_TLE:

    @triton.jit
    def ascend_fused_deepseek_v4_qnorm_rope_full_cache_fp8_kernel_q(
        q,
        q_fp8,
        position_ids,
        cos_sin_cache,
        q_fp8_scale_inv,
        eps,
        num_heads,
        num_tokens,
        H: tl.constexpr,
        APPLY_Q_NORM: tl.constexpr,
    ):
        """Persistent token/head loops with one RoPE table load per token."""
        HEAD_DIM: tl.constexpr = 512
        NOPE_DIM: tl.constexpr = 448
        NOPE_BLOCK: tl.constexpr = 256
        SECOND_NOPE_DIM: tl.constexpr = NOPE_DIM - NOPE_BLOCK
        ROPE_DIM: tl.constexpr = 64
        HALF_ROPE_DIM: tl.constexpr = 32

        pid = tl.program_id(0).to(tl.int64)
        row_block_size = tl.num_programs(0)
        q_scale_inv = tl.load(q_fp8_scale_inv).to(tl.float32)
        q_half = tl.arange(0, HALF_ROPE_DIM)
        for q_token in tl.range(pid, num_tokens, row_block_size):
            q_position = tl.load(position_ids + q_token)
            q_cs_base = cos_sin_cache + q_position * ROPE_DIM
            q_cos = tl.load(q_cs_base + q_half)
            q_sin = tl.load(q_cs_base + q_half + HALF_ROPE_DIM)

            for q_head_base in tl.range(0, num_heads, H):
                q_rows = q_token * num_heads + q_head_base + tl.arange(0, H)
                q_col = tl.arange(0, NOPE_BLOCK)
                q_offsets0 = q_rows[:, None] * HEAD_DIM + q_col[None, :]
                q_offsets1 = q_offsets0 + NOPE_BLOCK

                # Keep the shared input tiles in BF16/FP16 UB storage. Convert to
                # FP32 only after the read-only tensor views are returned.
                q_ub0 = tle.dsa.alloc(
                    [H, NOPE_BLOCK],
                    dtype=q.dtype.element_ty,
                    mem_addr_space=tle.dsa.ascend.UB,
                )
                q_ub1 = tle.dsa.alloc(
                    [H, NOPE_BLOCK],
                    dtype=q.dtype.element_ty,
                    mem_addr_space=tle.dsa.ascend.UB,
                )
                tle.dsa.copy(
                    q + q_offsets0,
                    q_ub0,
                    [H, NOPE_BLOCK],
                    inter_no_alias=True,
                )
                tle.dsa.copy(
                    q + q_offsets1,
                    q_ub1,
                    [H, NOPE_BLOCK],
                    inter_no_alias=True,
                )
                q_blk0 = tle.dsa.to_tensor(q_ub0, writable=False).to(tl.float32)
                q_blk1 = tle.dsa.to_tensor(q_ub1, writable=False).to(tl.float32)

                if APPLY_Q_NORM:
                    q_square_sum = tl.sum(q_blk0 * q_blk0, axis=1) + tl.sum(
                        q_blk1 * q_blk1, axis=1
                    )
                    q_rstd = tl.rsqrt(q_square_sum / HEAD_DIM + eps)
                else:
                    q_rstd = tl.full((H,), 1.0, tl.float32)
                q_combined_scale = q_rstd * q_scale_inv

                q_nope0_f32 = q_blk0 * q_combined_scale[:, None]
                q_nope0_u8 = _f32_to_e4m3_bits(q_nope0_f32)
                tl.store(q_fp8 + q_offsets0, q_nope0_u8)

                q_nope1_f32 = q_blk1 * q_combined_scale[:, None]
                q_nope1_u8 = _f32_to_e4m3_bits(q_nope1_f32)
                tl.store(
                    q_fp8 + q_offsets1,
                    q_nope1_u8,
                    mask=q_col[None, :] < SECOND_NOPE_DIM,
                )

                q_pair_off = (
                    q_rows[:, None, None] * HEAD_DIM
                    + NOPE_DIM
                    + q_half[None, :, None] * 2
                    + tl.arange(0, 2)[None, None, :]
                )
                q_pair = tl.load(q + q_pair_off).to(tl.float32)
                q_even = tl.reshape(
                    tle.dsa.extract_slice(
                        q_pair,
                        (0, 0, 0),
                        (H, HALF_ROPE_DIM, 1),
                        (1, 1, 1),
                    ),
                    (H, HALF_ROPE_DIM),
                )
                q_odd = tl.reshape(
                    tle.dsa.extract_slice(
                        q_pair,
                        (0, 0, 1),
                        (H, HALF_ROPE_DIM, 1),
                        (1, 1, 1),
                    ),
                    (H, HALF_ROPE_DIM),
                )
                q_new_even = q_even * q_cos[None, :] - q_odd * q_sin[None, :]
                q_new_odd = q_even * q_sin[None, :] + q_odd * q_cos[None, :]

                q_rope_even_u8 = _f32_to_e4m3_bits(
                    q_new_even * q_combined_scale[:, None]
                )
                q_rope_odd_u8 = _f32_to_e4m3_bits(q_new_odd * q_combined_scale[:, None])
                q_rope_u8 = tl.zeros((H, HALF_ROPE_DIM, 2), dtype=tl.uint8)
                q_rope_u8 = tle.dsa.insert_slice(
                    q_rope_u8,
                    tl.reshape(q_rope_even_u8, (H, HALF_ROPE_DIM, 1)),
                    (0, 0, 0),
                    (H, HALF_ROPE_DIM, 1),
                    (1, 1, 1),
                )
                q_rope_u8 = tle.dsa.insert_slice(
                    q_rope_u8,
                    tl.reshape(q_rope_odd_u8, (H, HALF_ROPE_DIM, 1)),
                    (0, 0, 1),
                    (H, HALF_ROPE_DIM, 1),
                    (1, 1, 1),
                )
                tl.store(q_fp8 + q_pair_off, q_rope_u8)

else:

    @triton.jit
    def ascend_fused_deepseek_v4_qnorm_rope_full_cache_fp8_kernel_q(
        q,
        q_fp8,
        position_ids,
        cos_sin_cache,
        q_fp8_scale_inv,
        eps,
        num_heads,
        num_tokens,
        H: tl.constexpr,
        APPLY_Q_NORM: tl.constexpr,
    ):
        """Non-TLE persistent Q kernel with the same launch signature."""
        HEAD_DIM: tl.constexpr = 512
        NOPE_DIM: tl.constexpr = 448
        NOPE_BLOCK: tl.constexpr = 256
        SECOND_NOPE_DIM: tl.constexpr = NOPE_DIM - NOPE_BLOCK
        ROPE_DIM: tl.constexpr = 64
        HALF_ROPE_DIM: tl.constexpr = 32

        pid = tl.program_id(0).to(tl.int64)
        row_block_size = tl.num_programs(0)
        q_scale_inv = tl.load(q_fp8_scale_inv).to(tl.float32)
        q_half = tl.arange(0, HALF_ROPE_DIM)
        for q_token in tl.range(pid, num_tokens, row_block_size):
            q_position = tl.load(position_ids + q_token)
            q_cs_base = cos_sin_cache + q_position * ROPE_DIM
            q_cos = tl.load(q_cs_base + q_half)
            q_sin = tl.load(q_cs_base + q_half + HALF_ROPE_DIM)

            for q_head_base in tl.range(0, num_heads, H):
                q_rows = q_token * num_heads + q_head_base + tl.arange(0, H)
                q_col = tl.arange(0, NOPE_BLOCK)
                q_offsets0 = q_rows[:, None] * HEAD_DIM + q_col[None, :]
                q_offsets1 = q_offsets0 + NOPE_BLOCK

                # Without TLE, load both input tiles directly into tensors.
                q_blk0 = tl.load(q + q_offsets0).to(tl.float32)
                q_blk1 = tl.load(q + q_offsets1).to(tl.float32)

                if APPLY_Q_NORM:
                    q_square_sum = tl.sum(q_blk0 * q_blk0, axis=1) + tl.sum(
                        q_blk1 * q_blk1, axis=1
                    )
                    q_rstd = tl.rsqrt(q_square_sum / HEAD_DIM + eps)
                else:
                    q_rstd = tl.full((H,), 1.0, tl.float32)
                q_combined_scale = q_rstd * q_scale_inv

                q_nope0_f32 = q_blk0 * q_combined_scale[:, None]
                q_nope0_u8 = _f32_to_e4m3_bits(q_nope0_f32)
                tl.store(q_fp8 + q_offsets0, q_nope0_u8)

                q_nope1_f32 = q_blk1 * q_combined_scale[:, None]
                q_nope1_u8 = _f32_to_e4m3_bits(q_nope1_f32)
                tl.store(
                    q_fp8 + q_offsets1,
                    q_nope1_u8,
                    mask=q_col[None, :] < SECOND_NOPE_DIM,
                )

                q_pair_off = (
                    q_rows[:, None, None] * HEAD_DIM
                    + NOPE_DIM
                    + q_half[None, :, None] * 2
                    + tl.arange(0, 2)[None, None, :]
                )
                q_pair = tl.load(q + q_pair_off).to(tl.float32)
                q_even, q_odd = tl.split(q_pair)
                q_new_even = q_even * q_cos[None, :] - q_odd * q_sin[None, :]
                q_new_odd = q_even * q_sin[None, :] + q_odd * q_cos[None, :]

                q_rope_even_u8 = _f32_to_e4m3_bits(
                    q_new_even * q_combined_scale[:, None]
                )
                q_rope_odd_u8 = _f32_to_e4m3_bits(q_new_odd * q_combined_scale[:, None])
                q_rope_u8 = tl.join(q_rope_even_u8, q_rope_odd_u8)
                tl.store(q_fp8 + q_pair_off, q_rope_u8)


@triton.jit
def ascend_fused_deepseek_v4_kv_rope_full_cache_fp8_insert_kernel_kv(
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    fp8_scale,
    cache_block_size: tl.constexpr,
    k_cache_stride0,
    pid_offset,
    total_programs,
):
    """Rotate, scale, encode, and insert one token's 512-byte KV record."""
    HEAD_DIM: tl.constexpr = 512
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    HALF_ROPE_DIM: tl.constexpr = 32
    QUANT_BLOCK: tl.constexpr = 64
    NUM_NOPE_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK

    kv_token = tl.program_id(0).to(tl.int64) + pid_offset
    if kv_token < total_programs:
        kv_slot = tl.load(slot_mapping + kv_token)
        if kv_slot >= 0:
            kv_base = kv + kv_token * HEAD_DIM
            kv_block = kv_slot // cache_block_size
            kv_in_block = kv_slot % cache_block_size
            kv_cache_token = (
                k_cache + kv_block * k_cache_stride0 + kv_in_block * HEAD_DIM
            )
            kv_scale = 1.0 / tl.load(fp8_scale).to(tl.float32)

            kv_group = tl.arange(0, 8)
            kv_quant_col = tl.arange(0, QUANT_BLOCK)
            kv_quant_off = kv_group[:, None] * QUANT_BLOCK + kv_quant_col[None, :]
            kv_nope = tl.load(kv_base + kv_quant_off).to(tl.float32)
            kv_nope_f32 = kv_nope * kv_scale
            kv_nope_u8 = _f32_to_e4m3_bits(kv_nope_f32)
            tl.store(
                kv_cache_token + kv_quant_off,
                kv_nope_u8,
                mask=(kv_group < NUM_NOPE_BLOCKS)[:, None],
            )

            kv_half = tl.arange(0, HALF_ROPE_DIM)
            kv_pair_off = kv_half[:, None] * 2 + tl.arange(0, 2)[None, :]
            kv_pair = tl.load(kv_base + NOPE_DIM + kv_pair_off).to(tl.float32)
            kv_position = tl.load(position_ids + kv_token)
            kv_cs_base = cos_sin_cache + kv_position * ROPE_DIM
            kv_cos = tl.load(kv_cs_base + kv_half)
            kv_sin = tl.load(kv_cs_base + kv_half + HALF_ROPE_DIM)
            kv_even, kv_odd = tl.split(kv_pair)
            kv_new_even = kv_even * kv_cos - kv_odd * kv_sin
            kv_new_odd = kv_even * kv_sin + kv_odd * kv_cos
            kv_even_u8 = _f32_to_e4m3_bits(kv_new_even * kv_scale)
            kv_odd_u8 = _f32_to_e4m3_bits(kv_new_odd * kv_scale)
            tl.store(
                kv_cache_token + NOPE_DIM + kv_pair_off,
                tl.join(kv_even_u8, kv_odd_u8),
            )


def fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_fp8: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    fp8_scale: torch.Tensor,
    q_fp8_scale_inv: torch.Tensor,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
):
    """Apply Q norm/RoPE and KV RoPE, then store external-scale E4M3 bytes."""
    assert q.dim() == 3 and q.shape[2] == 512
    num_tokens, num_heads, _ = q.shape
    assert q.is_contiguous() and kv.is_contiguous() and q_fp8.is_contiguous()
    assert q.dtype == kv.dtype and q.dtype in (torch.float16, torch.bfloat16)
    assert kv.shape == (num_tokens, 512) and q_fp8.shape == q.shape
    assert q_fp8.dtype == k_cache.dtype == torch.uint8
    assert k_cache.is_contiguous() and k_cache.dim() == 3
    assert k_cache.shape[1:] == (cache_block_size, 512)
    assert slot_mapping.dim() == 1 and slot_mapping.is_contiguous()
    assert slot_mapping.dtype == torch.int64
    assert slot_mapping.numel() <= num_tokens
    assert position_ids.shape == (num_tokens,) and position_ids.is_contiguous()
    assert position_ids.dtype == torch.int64
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.is_contiguous()
    assert fp8_scale.dtype == torch.float32 and fp8_scale.numel() == 1
    assert q_fp8_scale_inv.dtype == torch.float32 and q_fp8_scale_inv.numel() == 1

    if num_tokens == 0:
        return

    heads_per_program = q_heads_per_program(num_heads) if num_heads else 1
    tiles_per_token = num_heads // heads_per_program if num_heads else 0
    q_programs = num_tokens * tiles_per_token
    num_tokens_insert = slot_mapping.numel()
    if q_programs == 0 and num_tokens_insert == 0:
        return

    device_index = q.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    group, step = launch_geometry(device_index)

    if q_programs > 0:
        q_grid = min(num_tokens, group * Q_PROGRAMS_PER_VECTOR_CORE)
        ascend_fused_deepseek_v4_qnorm_rope_full_cache_fp8_kernel_q[(q_grid,)](
            q,
            q_fp8,
            position_ids,
            cos_sin_cache,
            q_fp8_scale_inv,
            float(eps),
            num_heads,
            num_tokens,
            heads_per_program,
            bool(apply_q_norm),
            num_stages=1,
        )

    for pid_offset in range(0, num_tokens_insert, step):
        grid = min(step, num_tokens_insert - pid_offset)
        if grid > group:
            grid = -(-grid // group) * group
        ascend_fused_deepseek_v4_kv_rope_full_cache_fp8_insert_kernel_kv[(grid,)](
            kv,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            fp8_scale,
            cache_block_size,
            k_cache.stride(0),
            pid_offset,
            num_tokens_insert,
            num_stages=1,
        )

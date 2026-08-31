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
"""Hygon override: token-tiled fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.

The generic kernel runs one program per (token, head) slot at num_warps=1 -- 64
threads for 512 elements, which on BW1000's 64-lane warp is 8 elements per lane.
That leaves more than half the bandwidth unused: measured 604.8 GB/s at
32768x64 against a 1340.3 GB/s ceiling (512 MiB device-to-device copy), i.e.
45.1%.

Giving each program TPP tokens of ONE slot raises elements per lane to 16 and the
block to 256 threads:

    shape           generic            tiled          of ceiling
    32768 x 64    604.8 GB/s      1182.9 GB/s     45.1% -> 88.3%
    32768 x 128   609.2 GB/s      1201.4 GB/s     45.5% -> 89.6%

Two axes matter and neither is visible on its own. A full TPP x num_warps sweep
puts every optimum at TPP/num_warps = 2, which is two tokens per warp and so 16
elements per lane; bandwidth by elements per lane is 237 / 408 / 604 / 906 /
**1183** / 1077 / 1050 for 1 / 2 / 4 / 8 / 16 / 32 / 64. But elements per lane
does not explain everything: TPP=1/warps=1 and TPP=2/warps=2 are both 8 elements
per lane and differ by 50% (604 vs 906 GB/s), because the second has a wider
program. Sweeping num_warps alone at TPP=1 shows 8 and 4 elements per lane tied,
which invites the wrong conclusion that access width does not matter -- at TPP=1
the block is only 512 elements and there is nothing to widen into. Do not tune
these two parameters separately.

TPP=8 with num_warps=4 is one of the optimal points and is also what the MetaX
C550 override uses; both parts have 64-lane warps, so the tuning transfers.

Below a threshold the generic kernel wins, because a 256-thread block costs about
10 us more to launch here (measured at one token: 80.5 us versus 90.2 us) and
TPP=8 masks off most of every program when there are fewer than 8 tokens to fill
it. The crossover was measured at 256 tokens for 64 heads and 128 tokens for 128
heads -- which are 16640 and 16512 programs respectively, so the real quantity is
the program count, not the token count. Dispatching on the grid size covers both
head counts; a flat token threshold would forfeit the 1.16x-1.32x available
between 128 and 256 tokens at 128 heads.

Output is bit-identical to the generic kernel: the FP8 cache matches byte for
byte and q matches exactly, including at token counts that are not multiples of
TPP.
"""

import torch
import triton
import triton.language as tl

# Measured on Hygon BW1000; see the module docstring. The threshold is a program
# count -- num_tokens * (num_heads + 1) -- because that is what the measured
# crossovers agree on across head counts.
_TILED_MIN_PROGRAMS = 16384
_TPP = 8
_NUM_WARPS = 4


@triton.jit
def _tiled_kernel(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps,
    cache_block_size: tl.constexpr,
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
    QUANT_BLOCK: tl.constexpr = 64
    NUM_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK  # 7
    SCALE_BYTES_PER_TOKEN: tl.constexpr = NUM_QUANT_BLOCKS + 1  # 8
    TOKEN_DATA_BYTES: tl.constexpr = NOPE_DIM + 2 * ROPE_DIM  # 576
    FP8_MAX: tl.constexpr = 448.0

    pid = tl.program_id(0).to(tl.int64)
    blocks_per_token: tl.constexpr = num_heads + 1

    # grid = cdiv(num_tokens, TPP) * blocks_per_token
    tile = pid // blocks_per_token
    slot_idx = pid % blocks_per_token
    is_kv = slot_idx == num_heads

    tok = tile * TPP + tl.arange(0, TPP).to(tl.int64)  # [TPP]
    tok_ok = tok < num_tokens

    off = tl.arange(0, HEAD_DIM)  # [HEAD_DIM]
    off_rope = tl.arange(0, ROPE_DIM)
    off_half = tl.arange(0, HALF_ROPE_DIM)
    off_quant = tl.arange(0, QUANT_BLOCK)

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
        variance = tl.sum(q_blk * q_blk, axis=1) / HEAD_DIM  # [TPP]
        rsqrt = tl.rsqrt(variance + eps)
        q_blk = q_blk * rsqrt[:, None]
        tl.store(
            q_base[:, None] + off[None, :],
            q_blk.to(tl.bfloat16),
            mask=tok_ok[:, None] & (off[None, :] < NOPE_DIM),
        )
        rope = (
            tl.load(
                q_base[:, None] + NOPE_DIM + off_rope[None, :],
                mask=tok_ok[:, None],
                other=0.0,
            ).to(tl.float32)
            * rsqrt[:, None]
        )
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
        q_base = q + (tok * num_heads + slot_idx) * HEAD_DIM
        tl.store(
            q_base[:, None] + NOPE_DIM + off_rope[None, :],
            rope,
            mask=tok_ok[:, None],
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


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
):
    """See the generic implementation for the layout contract."""
    assert q.is_contiguous() and kv.is_contiguous()
    num_tokens, num_heads, head_dims = q.shape

    if num_tokens * (num_heads + 1) < _TILED_MIN_PROGRAMS:
        # Small shapes are launch-bound and the narrower generic kernel wins.
        from flaggems_vllm.ops.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
            fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert as _generic,
        )

        return _generic(
            q,
            kv,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            eps,
            cache_block_size,
        )

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

    grid = triton.cdiv(num_tokens, _TPP) * (num_heads + 1)
    _tiled_kernel[(grid,)](
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        cache_block_size,
        num_tokens,
        num_heads,
        k_cache.stride(0),
        num_tokens_insert,
        TPP=_TPP,
        num_warps=_NUM_WARPS,
        num_stages=2,
    )

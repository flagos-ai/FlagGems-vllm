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
"""Ascend override for the fused DeepSeek-V4 qnorm/RoPE/quant/insert kernel.

The arithmetic is the generic implementation's throughout. Five things are
expressed differently, the first four because the toolchain rejects the generic
form and the fifth because of what this backend charges for a launch:

1. FP8 conversion goes through `_f32_to_e4m3_bits`, not `.to(tl.float8e4nv)`.
2. The UE8M0 scale is read off the exponent bits by `_ue8m0_scale`, not built
   with log2/ceil/exp2 and a float-to-integer conversion.
3. The bf16 half of each cache token is reached through a bfloat16 view passed
   in by the wrapper, not by casting a pointer's element type.
4. RoPE pairs are addressed with a 2-D offset, so they need no reshape.
5. The Q and KV halves share one kernel and one launch, branching on the program
   id, because a launch costs ~450 us here and two of them put a ~0.95 ms floor
   under every shape below a few hundred tokens.

Each is explained at its definition or use site. Points 2, 3 and 4 also each
surfaced as a compiler failure that reports the wrong thing -- a missing output
file, or nothing at all -- so read those notes before reformulating any of them.
Point 5 has its own trap, recorded in the kernel's docstring: a `return` nested
inside an arm of the branch aborts the compiler with an MLIR use-list assertion
that says nothing about control flow.

`tl.float8e4nv` cannot be compiled for this card, but for a different reason than
on T-Head. There is no capability gate to argue with — BiShengIR does not know the
type at all:

    error: 'hivm.hir.vcast' op currently don't support cast float_to_UNKNOWN_rintmode
    error: unrecognized float type: 'f8E4M3FN'

So the conversion is done with integer operations, which the backend compiles
without complaint. Verified bit-identical to `torch.float8_e4m3fn` on the card at
block widths 256 through 2048; the operator uses 512.

Ascend's Unified Buffer is the constraint to watch when changing this: the encoder
keeps roughly fifteen live intermediates, so a 4096-wide block asks for 240 KB
against the 192 KB available and fails to compile with `ub overflow, requires
1966080 bits while 1572864 bits available`. That is a loud compile-time failure
rather than a silent one, but it caps how much this kernel can be widened.

This file can go away once BiShengIR lowers `f8E4M3FN`, AICore can select a
scalar float-to-integer conversion, Triton can bitcast a pointer's element type
here, and `[ROPE_DIM] -> [HALF_ROPE_DIM, 2]` survives the shape pipeline.
"""

import torch
import triton
import triton.language as tl

# The runtime rejects a launch whose program count exceeds this, reporting it as
# an invalid `coreDim`. It is a launch-API limit, not a hardware occupancy one.
MAX_PROGRAMS_PER_LAUNCH = 65535

# Most heads of one token that a Q program may take, as an [H, HEAD_DIM] tile.
# A single head moves only a kilobyte, far too little to cover this backend's
# per-program cost. Measured at 4096 tokens, ns per (token, head) unit:
#
#     H = 16 -> 5.24 (391 GB/s)      H = 32 -> 4.15 (493 GB/s)   at 64 heads
#     H = 16 -> 5.14 (398 GB/s)      H = 32 -> 3.99 (513 GB/s)   at 128 heads
#
# 64 will not compile:
#
#     ub overflow, requires 3215360 bits while 1572864 bits available
#
# that is 392.5 KB wanted against 192 KB of Unified Buffer -- three times what
# the tile itself occupies, [64, HEAD_DIM] in float32 being 128 KB, because
# several intermediates are live at once and multi-buffering asks for more
# again, as the message says. Size a tile by measuring it, not by multiplying
# out its dimensions.
Q_MAX_HEADS_PER_PROGRAM = 32


@triton.jit
def _f32_to_e4m3_bits(x):
    """float32 -> OCP E4M3 (float8_e4m3fn) bit pattern, round-to-nearest-even.

    **Requires |x| <= 448. This is NOT a general E4M3 encoder** -- it has no
    saturation branch, so a larger input silently produces a wrong byte rather
    than 448. The one caller satisfies the precondition by construction: it
    divides by the smallest power of two with block_max / scale <= 448, and
    dividing by a power of two is exact on this card. Reinstate the saturation
    branch before using this anywhere else.

    Handles subnormals (m * 2^-9 for m in 1..7). Two branches the generic
    encoder carries are gone, each unreachable here and worth measuring:
    saturation, since |x| <= 448 means e_n reaches 15 only with mantissa 6,
    never the 15/7 NaN encoding; and the x == 0 case, which takes the subnormal
    path where |x| * 512 rounds to zero anyway. Removing both took the
    quantisation from 0.679 to 0.470 us per token.
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

    # A subnormal is m * 2^-9 for m in 1..7, so its mantissa is |x| * 512
    # rounded to nearest even. Adding 2^23 and taking it away again forces that
    # rounding with no intrinsic and, more to the point, no shifting.
    #
    # This replaces three shifts by a PER-LANE amount, which is what the
    # arithmetic form of this used to need. Those three were half of the whole
    # operator's KV time on this backend -- evidently scalarised, there being no
    # vector variable-shift instruction. Measured: 1.276 -> 0.627 us per token
    # for the quantisation, which is the same as deleting the subnormal path
    # altogether, at no cost in accuracy.
    magic: tl.constexpr = 8388608.0  # 2^23
    m_s = ((tl.abs(x) * 512.0 + magic) - magic).to(tl.int32)

    v = tl.where(e >= 1, (e_n << 3) | m_n, m_s)
    return (sign | v).to(tl.uint8)


@triton.jit
def _ue8m0_scale(raw_scale):
    """UE8M0 scale for a positive float: 2^ceil(log2(raw_scale)), plus the byte
    that encodes it -- which is exactly that power of two's biased exponent.

    Read straight off the exponent bits rather than going through
    log2 / ceil / exp2 and a float-to-integer conversion. The conversion is the
    reason this exists: the value is a per-block scalar, and AICore's scalar
    unit has no float-to-integer instruction, so `encoded_scale.to(tl.uint8)`
    fails instruction selection in bisheng with

        fatal error: error in backend: Cannot select: i64 = fp_to_uint

    (wrapped in the NaN and lower-bound selects Triton emits for a saturating
    conversion). The failure is reported as a missing kernel.o, because hivmc
    exits 0 after bisheng fails.

    raw_scale is always positive here -- block_max is floored at 1e-4 -- so the
    sign bit needs no handling.
    """
    bits = raw_scale.to(tl.int32, bitcast=True)
    biased_exp = (bits >> 23) & 0xFF
    # Ceil rather than floor: any set mantissa bit means the next power of two.
    code = biased_exp + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    scale = (code << 23).to(tl.float32, bitcast=True)
    stored = tl.minimum(tl.maximum(code, 0), 255)
    return scale, stored


def q_heads_per_program(num_heads: int) -> int:
    """Heads of one token per program: the largest power of two, at most 32,
    that divides num_heads.

    Dividing exactly means every tile is full, so the kernel needs no masks and
    can never address past the end of q -- which this backend faults on rather
    than honouring a mask.
    """
    h = 1
    cap = min(Q_MAX_HEADS_PER_PROGRAM, num_heads)
    while h * 2 <= cap and num_heads % (h * 2) == 0:
        h *= 2
    return h


@triton.jit
def fused_qnorm_rope_kv_insert_kernel(
    q,
    kv,
    k_cache,
    k_cache_bf16,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps,
    cache_block_size: tl.constexpr,
    num_heads,
    kv_block_stride,
    pid_offset,
    q_programs,
    tiles_per_token,
    H: tl.constexpr,
):
    """Both halves of the operator in ONE launch, selected per program.

    Programs [0, q_programs) each normalise and rotate H heads of one token; the
    rest each handle one token's KV. A chunk may straddle the boundary, since
    every program decides for itself from its global id -- so this needs no
    "one kernel below 65535 programs, two above" dispatch rule.

    WHY ONE LAUNCH. A launch costs ~450 us here, which is not overhead worth
    chasing when there is work to hide it behind, but below roughly 256 tokens
    there is not: two launches put a ~0.95 ms floor under every shape, and the
    kernels themselves finish long before it. Measured against the same code in
    two launches, bit-identical output either way:

        1 token,    64 heads   0.990 -> 0.510 ms   1.94x
        64 tokens,  64 heads   0.971 -> 0.495      1.96x
        256 tokens, 64 heads   0.983 -> 0.510      1.93x
        1024 tokens, 128 heads 1.046 -> 1.028      1.02x
        2048 tokens, 64 heads  1.546 -> 1.522      1.02x

    The gain stops once the work outgrows the dispatch, because dispatch and
    execution overlap: what one launch removes is the part that could not be
    hidden. Predicting a fixed 0.45 ms saving at every size would have been
    wrong by a factor of eight at 1024 tokens.

    **The KV arm must not use an early `return`.** `if kv_slot < 0: return`
    inside this `else` aborts the compiler outright --

        UseDefLists.h:198: ~IRObjectWithUseList() [OperandType = BlockOperand]:
        Assertion `use_empty() && "Cannot destroy a value that still has uses!"'

    -- with no mention of control flow, at every H and every num_stages. Plain
    early returns are fine and this file used to be full of them; what breaks is
    a `return` nested inside an arm of a two-armed `if`. The positive form below
    compiles and is bit-exact.

    **The two arms share no variable name.** Triton folds a name assigned in
    both arms into one SSA value and demands a single type at the join, so
    `even_blk` as [H, 32] here and [32] there is `Mismatched type for even_blk`.
    Every local in the KV arm is prefixed, including the ones whose shapes agree
    today -- relying on that agreement means relying on the two arms never being
    edited apart.
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

    # The work is issued in chunks of at most MAX_PROGRAMS_PER_LAUNCH, hence the
    # offset. The grid is exact, so neither arm needs a bounds guard.
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    if pid < q_programs:
        # ---- Q: RMSNorm without weight, then GPT-J RoPE, for H heads of ONE
        # token.
        #
        # Tiling heads within a token rather than walking a flat (token, head)
        # index is what makes `position_id` -- and with it cos and sin --
        # SCALAR. Every head of a token shares one position, so the flat form
        # gathered the same 256 bytes of cos/sin once per head, 64 or 128 times
        # over, on the unstructured pointer path. Measured at 4096 tokens:
        # 11.43 -> 4.15 ns per unit at 64 heads and 16.81 -> 3.99 at 128, i.e.
        # 179 -> 493 GB/s and 122 -> 513. It also explains why the flat form was
        # markedly slower at 128 heads than at 64 -- more heads, more repeats of
        # the same gather -- and that gap is gone.
        #
        # Built from nothing but wider vectors: no loop, no device function, no
        # early return. All three abort ttir_to_linalg on this backend with no
        # message. H divides num_heads, so every tile is full and no mask is
        # needed anywhere.
        token_idx = pid // tiles_per_token
        head_base = (pid % tiles_per_token) * H
        rows = token_idx * num_heads + head_base + tl.arange(0, H)

        col = tl.arange(0, HEAD_DIM)
        blk = tl.load(q + rows[:, None] * HEAD_DIM + col[None, :]).to(tl.float32)

        variance = tl.sum(blk * blk, axis=1) / HEAD_DIM
        rsqrt = tl.rsqrt(variance + eps)
        blk = blk * rsqrt[:, None]
        tl.store(
            q + rows[:, None] * HEAD_DIM + col[None, :],
            blk.to(tl.bfloat16),
            mask=col[None, :] < NOPE_DIM,
        )

        position_id = tl.load(position_ids + token_idx)  # scalar: one per token
        half = tl.arange(0, HALF_ROPE_DIM)
        cos_blk = tl.load(cos_sin_cache + position_id * ROPE_DIM + half)
        sin_blk = tl.load(
            cos_sin_cache + position_id * ROPE_DIM + HALF_ROPE_DIM + half
        )

        # [H, HALF_ROPE_DIM, 2]: the pair axis is broadcast into existence, as
        # on the KV path, so no reshape is needed.
        pair_off = (
            rows[:, None, None] * HEAD_DIM
            + NOPE_DIM
            + half[None, :, None] * 2
            + tl.arange(0, 2)[None, None, :]
        )
        pair = tl.load(q + pair_off).to(tl.float32)
        even_blk, odd_blk = tl.split(pair)
        even_blk = even_blk * rsqrt[:, None]
        odd_blk = odd_blk * rsqrt[:, None]
        new_even_blk = even_blk * cos_blk[None, :] - odd_blk * sin_blk[None, :]
        new_odd_blk = even_blk * sin_blk[None, :] + odd_blk * cos_blk[None, :]
        tl.store(q + pair_off, tl.join(new_even_blk, new_odd_blk).to(tl.bfloat16))
    else:
        # ---- KV: GPT-J RoPE on the last 64, then UE8M0 FP8 quantisation of the
        # NoPE region and the paged-cache insert.
        kv_token = pid - q_programs
        kv_base = kv + kv_token * HEAD_DIM
        offset_half_rope = tl.arange(0, HALF_ROPE_DIM)
        offset_quant = tl.arange(0, QUANT_BLOCK)
        # The RoPE pairs are addressed with a 2-D offset, so they arrive already
        # shaped [HALF_ROPE_DIM, 2] and need no reshape. BiShengIR rejects
        # turning [ROPE_DIM] into [HALF_ROPE_DIM, 2] here (`cannot align 0 axis`
        # on the expand_shape, `collapsing non-contiguous dims` on the way
        # back), and taking the pairs with 1-D stride-2 offsets instead aborts
        # the compiler in InterleaveOptimization.cpp. This form needs neither.
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
        if kv_slot >= 0:  # a negative slot is padding; see the docstring on why
            block_idx = kv_slot // cache_block_size
            pos_in_block = kv_slot % cache_block_size
            block_base = k_cache + block_idx * kv_block_stride
            token_fp8_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES
            # Ascend's Triton rejects casting a pointer between element widths
            # -- `Casting pointers with unmatched bitwidth!` -- so the RoPE
            # region is reached through a bf16 view passed in from the host.
            # Every byte offset here is even (block stride 37376, 576 per token,
            # 448 NoPE), so halving them is exact.
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
            # Quantisation of the KV NoPE region: all seven groups as ONE tile.
            #
            # The generic implementation unrolls seven 64-wide groups, its
            # comment citing load co-issue -- an NVIDIA consideration. On this
            # vector unit 64 lanes do not fill the machine and 448 do: one
            # [8, QUANT_BLOCK] tile, one axis=1 reduction giving all seven
            # scales at once, one encoder pass over 448 lanes. Measured 0.688 vs
            # 0.923 us per token on its own, and 0.470 once the dead branches
            # below go too.
            #
            # Triton wants a power-of-two block, hence eight rows. The eighth
            # covers the token's RoPE segment -- a token is 512 elements, 448
            # quantised plus 64 -- so it is real memory, not an overrun. It is
            # computed and discarded. Both stores mask it off with
            # `gidx < NUM_QUANT_BLOCKS`: an upper bound over an affine address,
            # which is the safe side of this backend's mask defect (a LOWER
            # bound over a non-affine address is what corrupts).
            gidx = tl.arange(0, 8)
            keep_group = gidx < NUM_QUANT_BLOCKS
            kv_quant_blk = tl.load(
                kv_base + gidx[:, None] * QUANT_BLOCK + offset_quant[None, :]
            ).to(tl.float32)
            block_max = tl.maximum(tl.max(tl.abs(kv_quant_blk), axis=1), 1e-4)
            # scale = 2^ceil(log2(block_max / FP8_MAX)), off the exponent bits
            scale, scale_code = _ue8m0_scale(block_max / FP8_MAX)
            # No clamp: scale is the smallest power of two with
            # block_max / scale <= FP8_MAX, and dividing by a power of two is
            # exact on this card, so |x| <= FP8_MAX holds by construction.
            # Worth 1.05x on its own.
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
            tl.store(token_scale_ptr + NUM_QUANT_BLOCKS, tl.zeros((), dtype=tl.uint8))


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
    """
    Horizontally-fused DeepseekV4-MLA: per-head RMSNorm + GPT-J RoPE for Q, and
    GPT-J RoPE + UE8M0 FP8 quant + paged cache insert for KV, all in one kernel
    launch.
    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576

    Args:
        q: [num_tokens, num_heads, 512], bfloat16, in place
        kv: [num_tokens, 512], bfloat16, read-only
        k_cache: [num_blocks, block_bytes], uint8
        slot_mapping: [num_tokens_insert], i64
        position_ids: [num_tokens], i64
        cos_sin_cache: [max_pos, 64], fp32
        eps: used in RMSNorm
        cache_block_size: tokens per paged-cache block
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

    assert k_cache.is_contiguous()
    k_cache_bf16 = k_cache.view(torch.bfloat16)

    # ONE grid over both kinds of work: Q tiles first, then one program per
    # inserted token. H heads of one token per Q program, and H divides
    # num_heads, so every tile is full -- no masks, and nothing can address past
    # the end of q, which this backend faults on rather than honouring a mask.
    #
    # The runtime refuses a launch wider than MAX_PROGRAMS_PER_LAUNCH:
    #
    #   KernelLaunch failed because value 532480 for parameter coreDim is
    #   invalid. Expected value: less than or equal to 65535.
    #
    # which 8192 tokens at 64 heads already exceeds eightfold, so the work is
    # issued in chunks and each program adds its chunk's offset to its id. A
    # chunk boundary may fall inside either region -- each program classifies
    # itself from its global id, so nothing here has to align to it.
    #
    # num_stages=1 is what was measured. The two-launch form used 2 for KV and
    # the default for Q, and a merged kernel can only have one value; 2 has not
    # been timed against 1 for this kernel.
    heads_per_program = q_heads_per_program(num_heads)
    tiles_per_token = num_heads // heads_per_program
    q_programs = num_tokens * tiles_per_token
    total_programs = q_programs + num_tokens_insert
    for pid_offset in range(0, total_programs, MAX_PROGRAMS_PER_LAUNCH):
        grid = min(MAX_PROGRAMS_PER_LAUNCH, total_programs - pid_offset)
        fused_qnorm_rope_kv_insert_kernel[(grid,)](
            q,
            kv,
            k_cache,
            k_cache_bf16,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            eps,
            cache_block_size,
            num_heads,
            k_cache.stride(0),
            pid_offset,
            q_programs,
            tiles_per_token,
            heads_per_program,
            num_warps=1,
            num_stages=1,
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused RoPE + quantize Q for the sparse indexer (enflame GCU adaptation).

Differences from the generic implementation (all documented GCU adaptation
recipes with in-repo precedents; every item below is measured and
bit-exact against the torch reference -- see the A/B lab in the 2026-11-11
perf case):

1. **int32 index normalization on the host.** ``tl.load`` from an int64
   tensor followed by address arithmetic misreads elements by a 2x element
   offset (silently wrong values plus out-of-bounds reads that abort the
   device; see ``compiler_issues/triton_i64_narrow_load.md``).  Token
   positions are provably < 2^31.  When torch_gcu's packed int64 layout
   probe passes, ``positions.view(torch.int32)[:numel]`` aliases the same
   bytes with zero kernel launches (bincount recipe); ``.to(torch.int32)``
   remains the fallback.  This removes one cast-kernel launch per call
   (~1.3us at num_tokens=1, official do_bench protocol).

2. **Token-dim grid folding.**  GCU caps grid.x at 65535, so tokens beyond
   that are spread over grid.z pages with an in-bounds guard (same recipe
   as mrope).  Shapes with num_tokens <= 65535 launch the identical
   (num_tokens, num_heads, 1) grid and address bits.

3. **Word-pair IO for the rotated tail (PAIR_WORD fast path).**  Each
   even/odd bf16 pair of the GPT-J interleaved tail is loaded as one int32
   word (low half = even element, little-endian) and the quantized fp8 pair
   is packed ``even | odd << 8`` and stored as one uint16 word -- replacing
   the stride-2 gathers on both the load and store sides (rotary_embedding
   pair-chain precedent).  Host guards (even strides + even nope dim) fall
   back to the original strided path.

4. **Tile arm for large token counts.**  The per-(token, head) grid is
   program-dispatch bound above ~256 programs (~10 programs/us measured on
   this part): at num_tokens >= 512 the tile kernel folds TILE_T tokens x
   ALL heads per program (runtime loops, cos/sin loaded once per token and
   reused across heads).  Measured (do_bench median, w1000/r100):
   512x8: 401->247us, 4096x8: 3189->1796us, 32768x8: 25645->14017us,
   4096x64: 25463->13712us.  num_tokens < 512 keeps the per-(token, head)
   grid, where the tile kernel's serialized row latency would regress
   (1x8: 6->22us).  Per-row math is identical in both kernels, so the arm
   choice never changes results.

Autotune note (workflow G3 exemption): the only launch parameter is
``num_warps``; a {1, 2, 4} sweep on the shipped kernels showed 1 optimal on
every measured shape (2/4 regress 10-15%), so it stays a compile-time
constant and the kernel has no free perf-parameter surface worth a tuner.

The MXFP4 path (``use_fp4=True``) relies on NVIDIA PTX inline asm
(``cvt.rn.satfinite.e2m1x2.f32``) and stays CUDA-only exactly like the
generic implementation; on GCU the host gate raises before any launch.
"""

import torch
import triton
import triton.language as tl

# MXFP4: 32 elements per block, packed 2 nibbles per byte, ue8m0 block scale.
MXFP4_BLOCK_SIZE = 32

# Enflame GCU hardware limit for grid.x; tokens beyond it are spread over
# grid.z pages (see mrope / indexer_k_quant_and_cache for the same pattern).
GCU_GRID_X_MAX = 65535

# Token count at (and above) which the dispatch-bound per-(token, head) grid
# switches to the tile kernel (TILE_T tokens x all heads per program).
# Measured crossover: the tile arm wins from 512x8 (401->247us) upward and
# loses below ~128 tokens (1x8: 6->22us, 32x8: 29->80us), so 512 is the
# conservative all-measured threshold.
TILE_MIN_TOKENS = 512
TILE_T = 4

_INT64_ALIAS_MODE = None


def _int32_view_of_int64(tensor_contig):
    """Zero-copy int32 alias of a GCU int64 tensor (bincount recipe).

    torch_gcu stores int64 tensors in a packed 4-byte-per-element device
    layout (the low 32 bits of each value, contiguously), so the first
    ``numel`` words of ``view(torch.int32)`` are exactly the int32 values --
    identical to ``.to(torch.int32)`` without launching a cast kernel.  A
    one-time per-process probe verifies the layout and falls back to the
    casting path if a future stack switches to a true 8-byte layout.
    """
    global _INT64_ALIAS_MODE
    if _INT64_ALIAS_MODE is None:
        try:
            probe = torch.arange(3, dtype=torch.int64, device=tensor_contig.device)
            words = probe.view(torch.int32)[:3]
            _INT64_ALIAS_MODE = (
                "packed" if words.cpu().tolist() == [0, 1, 2] else "wide"
            )
        except Exception:
            _INT64_ALIAS_MODE = "wide"
    if _INT64_ALIAS_MODE == "packed":
        return tensor_contig.view(torch.int32)[: tensor_contig.numel()]
    return tensor_contig.to(torch.int32)


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi):
    """Quantize a block of MXFP4_BLOCK_SIZE fp32 values given as two
    interleaved halves (x_lo = values at even positions in the block,
    x_hi = values at odd positions). Returns:
        - packed : uint8[BLOCK/2]  (low nibble = quant(x_lo), high = quant(x_hi))
        - ue8m0  : scalar uint8    (block scale = 2^(ue8m0 - 127))
    """
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    # 6 * 2^-126 is from https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py#L163
    amax = tl.maximum(amax, 6.0 * (2**-126))
    # ue8m0 block scale: 2^ceil(log2(amax/6.0)).
    log2_ratio = tl.math.ceil(tl.math.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    scale = tl.math.exp2(log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)

    inv_scale = 1.0 / scale
    packed = _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale)
    return packed, ue8m0


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    pos_ptr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # Index Q Quantize
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    num_tokens,
    grid_x,
    # GCU word-pair IO fast path eligibility (host-guarded; False keeps the
    # generic strided even/odd accesses bit-for-bit).
    PAIR_WORD: tl.constexpr,
):
    # Layout matches the unfused reference (DeepseekV4ScalingRotaryEmbedding
    # + per_token_group_quant_fp8): GPT-J interleaved RoPE applied to the
    # LAST rope_dim dims of each head; the leading [0, NOPE_DIM) is passed
    # through unchanged.
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0) + tl.program_id(2) * grid_x
    head_idx = tl.program_id(1)
    if tok_idx >= num_tokens:
        return

    pos = tl.load(pos_ptr + tok_idx)
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    cos = tl.load(
        index_q_cos_sin_ptr + pos * index_q_cos_sin_stride + half_offset
    ).to(tl.float32)
    sin = tl.load(
        index_q_cos_sin_ptr
        + pos * index_q_cos_sin_stride
        + half_offset
        + INDEX_Q_HALF_ROT_DIM
    ).to(tl.float32)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    # Interleaved (GPT-J) RoPE on dims [NOPE_DIM, HEAD_DIM):
    #   even = q[NOPE_DIM + 2*i],  odd = q[NOPE_DIM + 2*i + 1]
    # PAIR_WORD loads each (even, odd) bf16 pair as one int32 word
    # (little-endian: low half = even element) instead of two stride-2
    # gathers; identical values by construction.
    if PAIR_WORD:
        rot_words = tl.load(
            base_ptr.to(tl.pointer_type(tl.int32))
            + INDEX_Q_NOPE_DIM // 2
            + half_offset
        )
        x_even = (rot_words & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        x_odd = (rot_words >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    else:
        rot_base = base_ptr + INDEX_Q_NOPE_DIM
        x_even = tl.load(rot_base + half_offset * 2)
        x_odd = tl.load(rot_base + half_offset * 2 + 1)

    r_even = x_even.to(tl.float32) * cos - x_odd.to(tl.float32) * sin
    r_odd = x_odd.to(tl.float32) * cos + x_even.to(tl.float32) * sin

    # Match reference numerics: fp32 → bf16 → fp32 before the ue8m0 absmax.
    # Same pattern as the K-side compressor kernel (fused_compress_quant_cache.py).
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
    index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
    index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))

    # Store quantized values to index_q_fp8.  PAIR_WORD packs the quantized
    # (even, odd) fp8 pair into one uint16 word (low byte = even element),
    # turning the two stride-2 byte stores into one contiguous 64-byte store.
    fp8_base_ptr = (
        index_q_fp8_ptr + tok_idx * index_q_fp8_stride0 + head_idx * index_q_fp8_stride1
    )
    if INDEX_Q_NOPE_DIM > 0:
        tl.store(
            fp8_base_ptr + nope_offset,
            tl.div_rn(x_nope, index_q_scale).to(tl.float8e4nv),
        )
    if PAIR_WORD:
        re8 = tl.div_rn(r_even, index_q_scale).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        ro8 = tl.div_rn(r_odd, index_q_scale).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        pair = re8.to(tl.uint16) | (ro8.to(tl.uint16) << 8)
        tl.store(
            fp8_base_ptr.to(tl.pointer_type(tl.uint16))
            + INDEX_Q_NOPE_DIM // 2
            + half_offset,
            pair,
        )
    else:
        fp8_rot_base = fp8_base_ptr + INDEX_Q_NOPE_DIM
        tl.store(
            fp8_rot_base + half_offset * 2,
            tl.div_rn(r_even, index_q_scale).to(tl.float8e4nv),
        )
        tl.store(
            fp8_rot_base + half_offset * 2 + 1,
            tl.div_rn(r_odd, index_q_scale).to(tl.float8e4nv),
        )

    # FP8 weight-fold contract:
    #   index_weights_out = index_weights * q_scale * softmax_scale * head_scale
    # The per-token-per-head q_scale (fp32) IS folded into the output weights
    # here because FP8 Q is stored WITHOUT a companion scale tensor — the
    # downstream fp8_fp4_mqa_logits/fp8_fp4_paged_mqa_logits kernels use `weights` to
    # apply per-token Q scale inline. See the MXFP4 kernel below for the
    # contrasting convention (scales live with the Q values, weights are NOT
    # q-scaled).
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    )
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


@triton.jit
def _fused_indexer_q_rope_tile_kernel(
    pos_ptr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # Index Q Quantize
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    num_tokens,
    num_heads,
    # Tokens per program (host-chosen, >= TILE_T when the grid.x cap forces
    # larger tiles) and the same word-pair eligibility flag as the row kernel.
    TILE_T: tl.constexpr,
    PAIR_WORD: tl.constexpr,
):
    """Large-token-count arm of the fp8 path (host-dispatched at
    num_tokens >= TILE_MIN_TOKENS).

    One program owns TILE_T consecutive tokens x ALL heads.  The per-row
    math is IDENTICAL to ``_fused_indexer_q_rope_quant_kernel`` (bit-exact
    by construction and verified against both the torch reference and the
    row kernel), but the row loop amortizes the ~10 programs/us dispatch
    cap that dominates the per-(token, head) grid above ~256 programs.
    cos/sin are loaded once per token and reused across heads.
    """
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tile = tl.program_id(0)
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)

    for i in range(0, TILE_T):
        tok = tile * TILE_T + i
        if tok < num_tokens:
            pos = tl.load(pos_ptr + tok)
            cos = tl.load(
                index_q_cos_sin_ptr + pos * index_q_cos_sin_stride + half_offset
            ).to(tl.float32)
            sin = tl.load(
                index_q_cos_sin_ptr
                + pos * index_q_cos_sin_stride
                + half_offset
                + INDEX_Q_HALF_ROT_DIM
            ).to(tl.float32)
            for h in range(0, num_heads):
                base_ptr = index_q_ptr + tok * index_q_stride0 + h * index_q_stride1

                if PAIR_WORD:
                    rot_words = tl.load(
                        base_ptr.to(tl.pointer_type(tl.int32))
                        + INDEX_Q_NOPE_DIM // 2
                        + half_offset
                    )
                    x_even = (rot_words & 0xFFFF).to(tl.uint16).to(
                        tl.bfloat16, bitcast=True
                    )
                    x_odd = (rot_words >> 16).to(tl.uint16).to(
                        tl.bfloat16, bitcast=True
                    )
                else:
                    rot_base = base_ptr + INDEX_Q_NOPE_DIM
                    x_even = tl.load(rot_base + half_offset * 2)
                    x_odd = tl.load(rot_base + half_offset * 2 + 1)

                r_even = x_even.to(tl.float32) * cos - x_odd.to(tl.float32) * sin
                r_odd = x_odd.to(tl.float32) * cos + x_even.to(tl.float32) * sin
                r_even = r_even.to(tl.bfloat16).to(tl.float32)
                r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

                amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
                if INDEX_Q_NOPE_DIM > 0:
                    x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
                    amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
                index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
                index_q_scale = tl.math.exp2(
                    tl.math.ceil(tl.math.log2(index_q_scale))
                )

                fp8_base_ptr = (
                    index_q_fp8_ptr
                    + tok * index_q_fp8_stride0
                    + h * index_q_fp8_stride1
                )
                if INDEX_Q_NOPE_DIM > 0:
                    tl.store(
                        fp8_base_ptr + nope_offset,
                        tl.div_rn(x_nope, index_q_scale).to(tl.float8e4nv),
                    )
                if PAIR_WORD:
                    re8 = tl.div_rn(r_even, index_q_scale).to(tl.float8e4nv).to(
                        tl.uint8, bitcast=True
                    )
                    ro8 = tl.div_rn(r_odd, index_q_scale).to(tl.float8e4nv).to(
                        tl.uint8, bitcast=True
                    )
                    pair = re8.to(tl.uint16) | (ro8.to(tl.uint16) << 8)
                    tl.store(
                        fp8_base_ptr.to(tl.pointer_type(tl.uint16))
                        + INDEX_Q_NOPE_DIM // 2
                        + half_offset,
                        pair,
                    )
                else:
                    fp8_rot_base = fp8_base_ptr + INDEX_Q_NOPE_DIM
                    tl.store(
                        fp8_rot_base + half_offset * 2,
                        tl.div_rn(r_even, index_q_scale).to(tl.float8e4nv),
                    )
                    tl.store(
                        fp8_rot_base + half_offset * 2 + 1,
                        tl.div_rn(r_odd, index_q_scale).to(tl.float8e4nv),
                    )

                index_weights = tl.load(
                    index_weights_ptr + tok * index_weights_stride + h
                ).to(tl.float32)
                index_weights *= index_q_scale
                index_weights *= index_weights_softmax_scale
                index_weights *= index_weights_head_scale
                tl.store(
                    index_weights_out_ptr + tok * index_weights_out_stride + h,
                    index_weights,
                )


@triton.jit
def _fused_indexer_q_rope_mxfp4_kernel(
    pos_ptr,
    # Index Q RoPE input (fp/bf16)
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # MXFP4 Q outputs
    index_q_mxfp4_ptr,  # uint8, (T, H, HEAD_DIM // 2)
    index_q_mxfp4_stride0,
    index_q_mxfp4_stride1,
    index_q_scale_ptr,  # uint8 ue8m0, (T, H, HEAD_DIM // BLOCK)
    index_q_scale_stride0,
    index_q_scale_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    MXFP4_BLOCK: tl.constexpr,
    # Weights (NO per-token q_scale fold for MXFP4; per-block scales stay
    # with the Q values in the output scale tensor).
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    NUM_NOPE_BLOCKS: tl.constexpr = INDEX_Q_NOPE_DIM // MXFP4_BLOCK
    NUM_ROPE_BLOCKS: tl.constexpr = INDEX_Q_ROT_DIM // MXFP4_BLOCK
    HALF_BLOCK: tl.constexpr = MXFP4_BLOCK // 2
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)
    tl.static_assert(INDEX_Q_NOPE_DIM % MXFP4_BLOCK == 0)
    tl.static_assert(INDEX_Q_ROT_DIM % MXFP4_BLOCK == 0)
    tl.static_assert(MXFP4_BLOCK % 2 == 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)

    q_base = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    out_base = (
        index_q_mxfp4_ptr
        + tok_idx * index_q_mxfp4_stride0
        + head_idx * index_q_mxfp4_stride1
    )
    scale_base = (
        index_q_scale_ptr
        + tok_idx * index_q_scale_stride0
        + head_idx * index_q_scale_stride1
    )

    half_off = tl.arange(0, HALF_BLOCK)

    # ---- NoPE blocks: direct load, pair as (even-index, odd-index) values ----
    for b in tl.static_range(NUM_NOPE_BLOCKS):
        base = b * MXFP4_BLOCK
        x_lo = tl.load(q_base + base + half_off * 2).to(tl.float32)
        x_hi = tl.load(q_base + base + half_off * 2 + 1).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(x_lo, x_hi)
        tl.store(out_base + base // 2 + half_off, packed)
        tl.store(scale_base + b, ue8m0)

    # ---- RoPE blocks: apply GPT-J interleaved RoPE to the block's 16 pairs,
    # then quantize. Each block covers HALF_BLOCK (=16) cos/sin pairs. ----
    rot_q_base = q_base + INDEX_Q_NOPE_DIM
    for b in tl.static_range(NUM_ROPE_BLOCKS):
        pair_off = b * HALF_BLOCK + half_off  # indices in [0, HALF_ROT_DIM)
        cos_b = tl.load(
            index_q_cos_sin_ptr + pos * index_q_cos_sin_stride + pair_off
        ).to(tl.float32)
        sin_b = tl.load(
            index_q_cos_sin_ptr
            + pos * index_q_cos_sin_stride
            + pair_off
            + INDEX_Q_HALF_ROT_DIM
        ).to(tl.float32)
        x_even = tl.load(rot_q_base + pair_off * 2).to(tl.float32)
        x_odd = tl.load(rot_q_base + pair_off * 2 + 1).to(tl.float32)
        r_even = x_even * cos_b - x_odd * sin_b
        r_odd = x_odd * cos_b + x_even * sin_b
        # bf16 roundtrip for parity with the FP8 kernel / reference numerics.
        r_even = r_even.to(tl.bfloat16).to(tl.float32)
        r_odd = r_odd.to(tl.bfloat16).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd)
        rope_byte_off = (INDEX_Q_NOPE_DIM + b * MXFP4_BLOCK) // 2
        tl.store(out_base + rope_byte_off + half_off, packed)
        tl.store(scale_base + NUM_NOPE_BLOCKS + b, ue8m0)

    # MXFP4 weight-fold contract:
    #   index_weights_out = index_weights * softmax_scale * head_scale
    # NOTE: q_scale is NOT folded here (contrast with the FP8 kernel above).
    # MXFP4 Q emits a separate ue8m0 scale tensor of shape
    # (T, H, HEAD_DIM // MXFP4_BLOCK) alongside the packed values, so each
    # per-block scale is applied by the downstream MXFP4 logits kernel when
    # dequantizing Q — there is no per-token scalar to fold into `weights`.
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    ).to(tl.float32)
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    # Index weights
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    """Fused RoPE + quantize Q for the sparse indexer.

    Weight-fold semantics (important — the two paths differ):

    FP8 path (use_fp4=False, default):
        q_fp8      : (T, H, HEAD_DIM) float8_e4m3fn, per-token-per-head
                     scalar scale (NOT stored — folded into weights below)
        weights_out = weights * q_scale * softmax_scale * head_scale
        Rationale: a single per-token q_scale is a scalar the downstream FP8
        logits kernel would otherwise multiply in. Folding it into `weights`
        avoids emitting a separate tensor and is free for the logits kernel.

    MXFP4 path (use_fp4=True):
        q_packed   : (T, H, HEAD_DIM // 2) uint8 (2 E2M1 nibbles per byte)
        q_scale    : (T, H, HEAD_DIM // MXFP4_BLOCK_SIZE) uint8 ue8m0 bytes
        weights_out = weights * softmax_scale * head_scale
        Rationale: MXFP4 has PER-BLOCK (32-element) scales that live with
        the Q values — they cannot be folded into a per-token weight
        scalar, so `weights` carries only the softmax and head scales.

    Returns (q_quant, weights_out) where q_quant is either a Tensor (FP8) or
    a (values, scales) tuple (MXFP4). This matches the union type accepted
    by `SparseAttnIndexer.forward_*`.
    """
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]

    # GCU: int64 index loads poison the scalar load + address arithmetic
    # (triton_i64_narrow_load); token positions are provably < 2^31, so
    # normalize to int32 on the host.  The zero-copy packed-layout view
    # avoids the cast kernel launch; .to() remains the fallback.  No-op for
    # int32 inputs.
    if positions.dtype == torch.int64:
        if positions.is_contiguous():
            positions = _int32_view_of_int64(positions)
        else:
            positions = positions.to(torch.int32)

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    if use_fp4:
        if not index_q.is_cuda:
            raise RuntimeError("MXFP4 fused_indexer_q_rope_quant requires CUDA")
        major, _ = torch.cuda.get_device_capability(index_q.device)
        if major < 10:
            raise RuntimeError(
                "MXFP4 fused_indexer_q_rope_quant requires sm100 or newer; "
                f"got sm{major}x"
            )
        assert index_q_head_dim % MXFP4_BLOCK_SIZE == 0, (
            f"head_dim={index_q_head_dim} must be a multiple of MXFP4 block "
            f"size {MXFP4_BLOCK_SIZE}"
        )
        num_scale_blocks = index_q_head_dim // MXFP4_BLOCK_SIZE
        index_q_packed = torch.empty(
            (num_tokens, num_index_q_heads, index_q_head_dim // 2),
            dtype=torch.uint8,
            device=index_q.device,
        )
        index_q_scale = torch.empty(
            (num_tokens, num_index_q_heads, num_scale_blocks),
            dtype=torch.uint8,
            device=index_q.device,
        )
        _fused_indexer_q_rope_mxfp4_kernel[(num_tokens, num_index_q_heads)](
            positions,
            index_q,
            index_q.stride(0),
            index_q.stride(1),
            index_q_cos_sin_cache,
            index_q_cos_sin_cache.stride(0),
            index_q_cos_sin_cache.shape[-1] // 2,
            index_q_packed,
            index_q_packed.stride(0),
            index_q_packed.stride(1),
            index_q_scale,
            index_q_scale.stride(0),
            index_q_scale.stride(1),
            index_q_head_dim,
            MXFP4_BLOCK_SIZE,
            index_weights,
            index_weights.stride(0),
            index_weights_softmax_scale,
            index_weights_head_scale,
            index_weights_out,
            index_weights_out.stride(0),
            num_warps=1,  # TODO: Tune this
        )

        # Values stay uint8 (2 E2M1 nibbles per byte). Scales are 4 ue8m0
        # bytes per (token, head) reinterpreted as one int32, then squeezed
        # from (T, H, 1) to (T, H) to match DeepGEMM's expected q_sf rank
        # (prefill wants 2-D (seq_len, num_heads); decode reshapes this to
        # 3-D (batch, next_n, num_heads)).
        return (
            index_q_packed,
            index_q_scale.view(torch.int32).squeeze(-1),
        ), index_weights_out

    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    # GCU word-pair IO eligibility: the int32/uint16 word views of the
    # rotated tail need 2-byte element alignment of every addressed row
    # base (even strides) and an even nope dim; anything else keeps the
    # generic strided accesses (bit-for-bit identical results).
    index_q_half_rot_dim = index_q_cos_sin_cache.shape[-1] // 2
    pair_word = (
        index_q_head_dim % 2 == 0
        and (index_q_head_dim - 2 * index_q_half_rot_dim) % 2 == 0
        and index_q.stride(0) % 2 == 0
        and index_q.stride(1) % 2 == 0
        and index_q_fp8.stride(0) % 2 == 0
        and index_q_fp8.stride(1) % 2 == 0
    )
    if num_tokens >= TILE_MIN_TOKENS:
        # Dispatch-bound regime: fold TILE_T tokens x all heads per program.
        # The tile grows past TILE_T only to respect the grid.x hardware cap.
        tile_t = max(TILE_T, triton.cdiv(num_tokens, GCU_GRID_X_MAX))
        _fused_indexer_q_rope_tile_kernel[(triton.cdiv(num_tokens, tile_t),)](
            positions,
            index_q,
            index_q.stride(0),
            index_q.stride(1),
            index_q_cos_sin_cache,
            index_q_cos_sin_cache.stride(0),
            index_q_half_rot_dim,
            index_q_fp8,
            index_q_fp8.stride(0),
            index_q_fp8.stride(1),
            index_q_head_dim,
            index_weights,
            index_weights.stride(0),
            index_weights_softmax_scale,
            index_weights_head_scale,
            index_weights_out,
            index_weights_out.stride(0),
            num_tokens,
            num_index_q_heads,
            TILE_T=tile_t,
            PAIR_WORD=pair_word,
            num_warps=1,
        )
        return index_q_fp8, index_weights_out
    # GCU: fold tokens beyond the grid.x hardware limit into grid.z pages;
    # shapes with num_tokens <= 65535 launch the identical 2-D grid.
    grid_x = min(num_tokens, GCU_GRID_X_MAX)
    grid_z = triton.cdiv(num_tokens, grid_x) if grid_x > 0 else 1
    _fused_indexer_q_rope_quant_kernel[(grid_x, num_index_q_heads, grid_z)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_half_rot_dim,
        index_q_fp8,
        index_q_fp8.stride(0),
        index_q_fp8.stride(1),
        index_q_head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        num_tokens,
        grid_x,
        PAIR_WORD=pair_word,
        num_warps=1,
    )
    return index_q_fp8, index_weights_out

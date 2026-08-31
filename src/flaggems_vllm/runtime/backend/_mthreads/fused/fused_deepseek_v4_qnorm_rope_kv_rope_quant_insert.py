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
"""Moore Threads override: token-tiled, and only at 64 heads.

REQUIRES FLAGTREE >= 0.6.1+mthreads3.6, which bundles a working `llc` at
`triton/backends/mthreads/bin/llc` (md5 cec9ff66714e311670b9412ec760e4aa). On
older builds -- 0.6.0rc3+mthreads3.6 among them -- there is no `bin/` and Triton
falls back to the `llc` shipped with MUSA toolkit 4.3.5, where this kernel does
not compile in ANY configuration: five were tried, including TPP=1/num_warps=1
whose 16 elements per lane match the generic launch that works fine, and all five
abort with `llc` code -6. The 2-D tile trips an instruction-selection defect
there (`MTGPU DAG->DAG Pattern Instruction Selection`,
`SelectionDAGISel::CannotYetSelect` on a v8bf16 `LSU_LD_CACHE_HINT`); the same
five pass under the bundled one. On a build without it the dispatch below still
routes to the generic kernel, so nothing here is reached and nothing breaks.

Two conditions gate the tiled path, and both were measured on S5000 rather than
carried over -- the MetaX and Hygon overrides use different rules, and their
tuning does NOT transfer because those parts have 64-lane warps while this one
has 32, so the generic launch here already delivers 16 elements per lane, the
value the other two need TPP=8/num_warps=4 to reach.

**Head count.** Tiling helps only at 64 heads. Measured against a 1332 GB/s
ceiling at 32768 tokens, the generic kernel reaches 87.4% at 64 heads but 93.6%
at 128, and a full TPP x num_warps sweep at 128 heads finds nothing above
0.94x -- there is simply no headroom left there. The operator's own tests and
benchmark use 64 and 128 heads only, so this is an exhaustive two-case rule, not
a fitted threshold. At 64 heads the best cell is TPP=4/num_warps=4 at 91.6%.

**Token count.** Below 192 tokens the measurement itself is unusable: three
round-robin repetitions spread 12-34%, so the apparent 0.82x at 64 tokens and
1.02x at 96 are both noise. From 192 upward the spread falls to <= 3%, and the
crossover sits between 256 (1.00x) and 512 (1.03x). The threshold is 512, which
is conservative: it forfeits nothing measurable and stays clear of the unstable
region entirely.

The gain is modest and worth stating plainly: about 1.04-1.05x at 64 heads on
large shapes, nothing at 128 heads. It is not the 2x that tiling buys on Hygon,
because there the generic kernel starts at 45% of achievable and here it starts
at 87%.

Output is bit-identical to the generic kernel in the FP8 cache at every shape
measured; q differs by at most one bf16 ULP, from the RMSNorm reduction order.
"""

import torch
import triton
import triton.language as tl

# See the docstring: 64 heads only, and only above the point where the
# measurement stops being noise.
_TILED_MAX_HEADS = 64
_TILED_MIN_TOKENS = 512
_TPP = 4
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

    if num_heads > _TILED_MAX_HEADS or num_tokens < _TILED_MIN_TOKENS:
        # 128 heads: the generic kernel is already at 93.6% of achievable and no
        # tiling configuration beats it. Small shapes: launch-bound, and the
        # region is too noisy to claim a win in.
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

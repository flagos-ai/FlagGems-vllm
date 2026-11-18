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

import triton
import triton.language as tl

from flaggems_vllm.ops.mhc.mhc_pre import _mhc_recip


@triton.jit
def mega_mhc_kernel(
    X,
    Residual,
    Post,
    Comb,
    Fn,
    Scale,
    Base,
    Norm,
    NewResidual,
    NewPost,
    NewComb,
    Y,
    comb_stride_t,
    comb_stride_src,
    comb_stride_dst,
    rms_eps,
    pre_eps,
    sinkhorn_eps,
    post_multiplier,
    sinkhorn_repeat,
    norm_eps,
    H: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_NORM: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    residual_base = token * R * H
    input_base = token * H
    comb_base = token * comb_stride_t
    k = tl.arange(0, BLOCK_K)
    p = tl.arange(0, BLOCK_P)
    projection = tl.zeros((BLOCK_P, BLOCK_K), tl.float32)
    square_sum = tl.zeros((BLOCK_K,), tl.float32)

    # These coefficients are reused for every hidden tile.
    post_values = ()
    comb_values = ()
    for dst in tl.static_range(R):
        post_values += (tl.load(Post + token * R + dst),)
        for src in tl.static_range(R):
            comb_values += (
                tl.load(
                    Comb + comb_base + src * comb_stride_src + dst * comb_stride_dst
                ),
            )

    # Keep all projection heads and the row norm in the same CTA as the
    # residual writer. The projection uses the newly rounded BF16 residual.
    for start in range(0, H, BLOCK_K):
        h = start + k
        valid = h < H
        x = tl.load(X + input_base + h, valid, other=0).to(tl.float32)
        # Load each residual source once per tile and reuse it for all outputs.
        sources = ()
        for src in tl.static_range(R):
            sources += (
                tl.load(Residual + residual_base + src * H + h, valid, other=0).to(
                    tl.float32
                ),
            )
        for dst in tl.static_range(R):
            post = post_values[dst]
            mixed = tl.zeros((BLOCK_K,), tl.float32)
            for src in tl.static_range(R):
                coef = comb_values[dst * R + src]
                stream = sources[src]
                mixed = tl.fma(coef, stream, mixed)
            bf16 = (mixed + post * x).to(tl.bfloat16)
            tl.store(NewResidual + residual_base + dst * H + h, bf16, valid)
            value = bf16.to(tl.float32)
            # Wide K tiles benefit from fused FP32 square accumulation on PPU.
            if BLOCK_K >= 256:
                square_sum = tl.fma(value, value, square_sum)
            else:
                square_sum += value * value
            weight = tl.load(
                Fn + p[:, None] * (R * H) + dst * H + h[None, :],
                (p < P)[:, None] & valid[None, :],
                other=0,
            )
            projection = tl.fma(weight, value[None, :], projection)

    logits = tl.sum(projection, 1) * tl.rsqrt(tl.sum(square_sum, 0) / (R * H) + rms_eps)
    idx = tl.arange(0, R)
    pre = (
        tl.sigmoid(tl.gather(logits, idx, 0) * tl.load(Scale) + tl.load(Base + idx))
        + pre_eps
    )
    post = tl.sigmoid(
        tl.gather(logits, R + idx, 0) * tl.load(Scale + 1) + tl.load(Base + R + idx)
    )
    tl.store(NewPost + token * R + idx, post * post_multiplier)

    matrix_index = tl.arange(0, R * R)
    comb = tl.gather(logits, 2 * R + matrix_index, 0)
    comb = (comb * tl.load(Scale + 2) + tl.load(Base + 2 * R + matrix_index)).reshape(
        (R, R)
    )
    comb = tl.exp(comb - tl.max(comb, 1)[:, None])
    comb = comb * _mhc_recip(tl.sum(comb, 1))[:, None] + sinkhorn_eps
    comb = comb * _mhc_recip(tl.sum(comb, 0) + sinkhorn_eps)[None, :]
    for _ in tl.range(1, sinkhorn_repeat, loop_unroll_factor=4):
        comb = comb * _mhc_recip(tl.sum(comb, 1) + sinkhorn_eps)[:, None]
        comb = comb * _mhc_recip(tl.sum(comb, 0) + sinkhorn_eps)[None, :]
    tl.store(NewComb + token * R * R + matrix_index, comb.reshape((R * R,)))

    # The layouts of the producer vectors and this full-row load can differ.
    tl.debug_barrier()
    h = tl.arange(0, BLOCK_H)
    valid = h < H
    collapsed = tl.zeros((BLOCK_H,), tl.float32)
    for src in tl.static_range(R):
        weight = tl.sum(tl.where(idx == src, pre, 0), 0)
        stream = tl.load(NewResidual + residual_base + src * H + h, valid, other=0).to(
            tl.float32
        )
        collapsed = tl.fma(weight, stream, collapsed)
    result = collapsed.to(tl.bfloat16).to(tl.float32)
    if HAS_NORM:
        weight = tl.load(Norm + h, valid, other=0).to(tl.float32)
        result = result * tl.rsqrt(tl.sum(result * result, 0) / H + norm_eps) * weight
    tl.store(Y + input_base + h, result, valid)

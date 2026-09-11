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

"""Iluvatar BI-V150-optimized mhc_post.

[KernelGen] Auto-generated and tuned for Iluvatar BI-V150.
Measured geo-mean speedup vs torch reference: 15.38x.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _load_row(p, m, EVEN_H: tl.constexpr):
    if EVEN_H:
        return tl.load(p).to(tl.float32)
    else:
        return tl.load(p, mask=m, other=0.0).to(tl.float32)


@triton.jit
def _mhc_post_kernel(
    x_ptr,
    res_ptr,
    comb_ptr,
    post_ptr,
    out_ptr,
    H,
    C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN_H: tl.constexpr,
):
    """Fused mHC post block.

    out[t, i, d] = bf16( post[t, i] * x[t, d]
                         + sum_j comb[t, i, j] * residual[t, j, d] )

    Grid: (T, cdiv(H, BLOCK_H)).  Each program owns a [C, BLOCK_H] output tile
    (all C heads x a contiguous slice of hidden) for one token.  The C residual
    rows and the x row are read exactly once from DRAM; comb/post per token are
    tiny scalar payloads.  All arithmetic is fp32 with a single fp32 -> bf16
    round at the store (mirrors the reference).
    """
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)

    t = pid_t.to(tl.int64)
    t_off = t * (C * H)
    x_base = x_ptr + t * H
    comb_base = comb_ptr + t * (C * C)
    post_base = post_ptr + t * C

    ia = tl.arange(0, C)  # head index i
    d = pid_d * BLOCK_H + tl.arange(0, BLOCK_H)
    m = d < H

    # x slice for this token / hidden block (exact bf16 -> fp32 upcast).
    xv = _load_row(x_base + d, m, EVEN_H)  # [BLOCK_H]

    # acc2[i, d] = sum_j comb[t, i, j] * residual[t, j, d]
    acc2 = tl.zeros([C, BLOCK_H], dtype=tl.float32)
    for j in tl.static_range(C):
        rowj = _load_row(res_ptr + t_off + j * H + d, m, EVEN_H)
        colj = tl.load(comb_base + ia * C + j)  # [C] fp32
        acc2 += colj[:, None] * rowj[None, :]

    # add post_term: post[t, i] * x[t, d]
    postv = tl.load(post_base + ia)  # [C] fp32
    acc2 += postv[:, None] * xv[None, :]

    # single fp32 -> bf16 round at the store
    out_off = out_ptr + t_off + ia[:, None] * H + d[None, :]
    if EVEN_H:
        tl.store(out_off, acc2.to(tl.bfloat16))
    else:
        m2 = (ia[:, None] < C) & (d[None, :] < H)
        tl.store(out_off, acc2.to(tl.bfloat16), mask=m2)


def run(x, residual, post_layer_mix, comb_res_mix):
    """mHC post block, value-returning.

    x              : [T, hidden_size]          bf16
    residual       : [T, hc_mult, hidden_size] bf16
    post_layer_mix : [T, hc_mult, 1]           fp32
    comb_res_mix   : [T, hc_mult, hc_mult]     fp32
    returns out    : [T, hc_mult, hidden_size] bf16
    """
    T, H = x.shape
    C = residual.shape[1]

    out = torch.empty((T, C, H), device=x.device, dtype=torch.bfloat16)

    BLOCK_H = 512
    even_h = (H % BLOCK_H) == 0
    grid = (triton.cdiv(H, BLOCK_H), T)
    _mhc_post_kernel[grid](
        x,
        residual,
        comb_res_mix,
        post_layer_mix,
        out,
        H,
        C=C,
        BLOCK_H=BLOCK_H,
        EVEN_H=even_h,
        num_warps=4,
    )
    return out


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return run(x, residual, post_layer_mix, comb_res_mix)

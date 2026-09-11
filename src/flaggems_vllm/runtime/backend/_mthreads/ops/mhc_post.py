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

"""Moore Threads MTT S5000-optimized mhc_post.

[KernelGen] Auto-generated and tuned for Moore Threads MTT S5000.
Measured geo-mean speedup vs torch reference: 13.17x.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    H,
    M: tl.constexpr,  # hc_mult: out rows and contraction length
    M_P: tl.constexpr,  # next_pow2(M) for tl.arange
    BH: tl.constexpr,  # hidden tile width
    NEED_H: tl.constexpr,  # True when H % BH != 0 (tail block masking)
    NEED_I: tl.constexpr,  # True when M is not a power of two
):
    # Each program computes out[t, :, h0:h0+BH] for one (t, h-block).
    hb = tl.program_id(0)
    t = tl.program_id(1)

    i = tl.arange(0, M_P)
    h = hb * BH + tl.arange(0, BH)

    tH = t * H
    # x[t, h] -> fp32 [BH], reused across all M output rows
    if NEED_H:
        hmask = h < H
        x_row = tl.load(x_ptr + tH + h, mask=hmask, other=0.0).to(tl.float32)
    else:
        x_row = tl.load(x_ptr + tH + h, eviction_policy="evict_first").to(tl.float32)

    # fp32 accumulator [M_P, BH]
    acc = tl.zeros((M_P, BH), dtype=tl.float32)

    res_base = t * (M * H)
    comb_base = t * (M * M)
    # acc += comb_col_j (outer) residual_row_j over contraction axis j
    for j in tl.static_range(M):
        if NEED_I:
            comb_col = tl.load(
                comb_ptr + comb_base + i * M + j, mask=i < M, other=0.0
            ).to(tl.float32)
        else:
            comb_col = tl.load(comb_ptr + comb_base + i * M + j).to(tl.float32)
        if NEED_H:
            res_row = tl.load(
                residual_ptr + res_base + j * H + h, mask=hmask, other=0.0
            ).to(tl.float32)
        else:
            res_row = tl.load(
                residual_ptr + res_base + j * H + h, eviction_policy="evict_first"
            ).to(tl.float32)
        acc += comb_col[:, None] * res_row[None, :]

    # post_layer_mix[t, i] broadcast against x
    if NEED_I:
        post = tl.load(post_ptr + t * M + i, mask=i < M, other=0.0).to(tl.float32)
    else:
        post = tl.load(post_ptr + t * M + i).to(tl.float32)
    acc += post[:, None] * x_row[None, :]

    out_off = t * (M * H) + i[:, None] * H + h[None, :]
    out_val = acc.to(tl.bfloat16)
    if NEED_I and NEED_H:
        tl.store(out_ptr + out_off, out_val, mask=(i[:, None] < M) & (h[None, :] < H))
    elif NEED_I:
        tl.store(out_ptr + out_off, out_val, mask=i[:, None] < M)
    elif NEED_H:
        tl.store(out_ptr + out_off, out_val, mask=h[None, :] < H)
    else:
        tl.store(out_ptr + out_off, out_val, eviction_policy="evict_first")


def run(x, residual, post_layer_mix, comb_res_mix):
    T, H = x.shape
    M = residual.shape[1]
    out = torch.empty((T, M, H), dtype=residual.dtype, device=residual.device)
    M_P = triton.next_power_of_2(M)
    # Tiny-T workloads are launch-latency bound: small BH keeps per-CTA setup
    # light (measured best at T<=8). Larger T is bandwidth bound: BH=512 with
    # 128 threads gives 4 elements/thread along h (64-bit vector memory ops).
    if T <= 8:
        BH, nw = 128, 4
    else:
        BH, nw = 512, 4
    nhb = triton.cdiv(H, BH)
    grid = (nhb, T)
    _mhc_post_kernel[grid](
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        out,
        H,
        M=M,
        M_P=M_P,
        BH=BH,
        NEED_H=(H % BH != 0),
        NEED_I=(M != M_P),
        num_warps=nw,
    )
    return out


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return run(x, residual, post_layer_mix, comb_res_mix)

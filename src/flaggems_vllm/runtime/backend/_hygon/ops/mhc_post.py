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

"""Hygon DCU-optimized mhc_post.

[KernelGen] Auto-generated and tuned for Hygon DCU.
Measured geo-mean speedup vs torch reference: 12.12x.
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
    M: tl.constexpr,  # hc_mult (runtime value, specialized)
    BM: tl.constexpr,  # next_pow2(M)
    BN: tl.constexpr,  # hidden block size
    FULL: tl.constexpr,  # True: H % BN == 0 and BM == M -> no masking
):
    pid_h = tl.program_id(0)  # hidden-block index
    pid_t = tl.program_id(1)  # token index

    i = tl.arange(0, BM)
    d = tl.arange(0, BN)
    col = pid_h * BN + d

    x_base = x_ptr + pid_t * H
    res_base = residual_ptr + pid_t * (M * H)
    comb_base = comb_ptr + pid_t * (M * M)
    post_base = post_ptr + pid_t * M
    out_base = out_ptr + pid_t * (M * H)

    acc = tl.zeros([BM, BN], dtype=tl.float32)

    if FULL:
        for j in range(M):
            c = tl.load(comb_base + i * M + j)  # [BM] fp32 column j
            r = tl.load(res_base + j * H + col)  # [BN] bf16 row j
            acc += c[:, None] * r[None, :].to(tl.float32)
        xrow = tl.load(x_base + col)  # [BN] bf16
        p = tl.load(post_base + i)  # [BM] fp32
        val = acc + p[:, None] * xrow[None, :].to(tl.float32)
        tl.store(out_base + i[:, None] * H + col[None, :], val.to(tl.bfloat16))
    else:
        m_d = col < H
        for j in range(M):
            c = tl.load(comb_base + i * M + j, mask=i < M, other=0.0)
            r = tl.load(res_base + j * H + col, mask=m_d, other=0.0)
            acc += c[:, None] * r[None, :].to(tl.float32)
        xrow = tl.load(x_base + col, mask=m_d, other=0.0)
        p = tl.load(post_base + i, mask=i < M, other=0.0)
        val = acc + p[:, None] * xrow[None, :].to(tl.float32)
        tl.store(
            out_base + i[:, None] * H + col[None, :],
            val.to(tl.bfloat16),
            mask=(i[:, None] < M) & m_d[None, :],
        )


def run(x, residual, post_layer_mix, comb_res_mix):
    T, H = x.shape
    M = residual.shape[1]
    out = torch.empty((T, M, H), device=x.device, dtype=residual.dtype)

    BM = triton.next_power_of_2(M)
    # Bandwidth-dominated large workloads (many tokens x wide hidden) prefer wide
    # 512-element rows with few warps; small/medium and latency-bound shapes do
    # better with 256-element rows. Dispatch on total input elements T*H.
    if T * H >= 800_000:
        BN = 512
        num_warps = 4
    else:
        BN = 256
        num_warps = 4
    FULL = (H % BN == 0) and (BM == M)
    grid = (triton.cdiv(H, BN), T)
    _mhc_post_kernel[grid](
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        out,
        H,
        M=M,
        BM=BM,
        BN=BN,
        FULL=FULL,
        num_warps=num_warps,
    )
    return out


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return run(x, residual, post_layer_mix, comb_res_mix)

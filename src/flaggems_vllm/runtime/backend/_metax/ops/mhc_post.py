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

"""MetaX C550-optimized mhc_post.

[KernelGen] Auto-generated and tuned for MetaX C550.
Measured geo-mean speedup vs torch reference: 19.79x.
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
    P: tl.constexpr,
    P_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """mHC post block fused kernel.

    out[t, i, d] = bf16( post_layer_mix[t, i] * x[t, d]
                         + sum_j comb_res_mix[t, i, j] * residual[t, j, d] )

    One program handles one token t and one contiguous hidden chunk
    [d0, d0+BLOCK_D).  The j-loop loads each residual row once and
    broadcasts it in registers across all P output planes (acc tile is
    [P_PAD, BLOCK_D] fp32).  All arithmetic is fp32; a single explicit
    bf16 cast happens right before the store (mirrors the reference
    rounding boundary).
    """
    t = tl.program_id(1)
    chunk = tl.program_id(0)
    d0 = chunk * BLOCK_D
    d = d0 + tl.arange(0, BLOCK_D)
    d_mask = d < H

    i = tl.arange(0, P_PAD)
    i_mask = i < P

    # x row chunk [BLOCK_D] (bf16 -> fp32, lossless)
    x_row = tl.load(x_ptr + t * H + d, mask=d_mask, other=0.0).to(tl.float32)
    # post column [P_PAD] fp32
    post_col = tl.load(post_ptr + t * P + i, mask=i_mask, other=0.0).to(tl.float32)

    # acc[i, d] = post_i * x_d   (fp32)
    acc = post_col[:, None] * x_row[None, :]

    res_base = t * P * H
    comb_base = t * P * P
    for j in range(P):
        # residual row j chunk [BLOCK_D], loaded once and reused across planes
        res_row = tl.load(
            residual_ptr + res_base + j * H + d, mask=d_mask, other=0.0
        ).to(tl.float32)
        # comb column j [P_PAD] fp32 (gather stride P, stays in cache)
        comb_col = tl.load(comb_ptr + comb_base + i * P + j, mask=i_mask, other=0.0).to(
            tl.float32
        )
        acc += comb_col[:, None] * res_row[None, :]

    out_val = acc.to(tl.bfloat16)
    out_off = (t * P + i)[:, None] * H + d[None, :]
    tl.store(out_ptr + out_off, out_val, mask=i_mask[:, None] & d_mask[None, :])


def run(x, residual, post_layer_mix, comb_res_mix):
    """mHC post block (vllm/model_executor/kernels/mhc/torch.py) in Triton.

    x              : [T, H]            bfloat16
    residual       : [T, P, H]         bfloat16
    post_layer_mix : [T, P, 1]         float32
    comb_res_mix   : [T, P, P]         float32
    returns out    : [T, P, H]         bfloat16
    """
    T, H = x.shape
    P = residual.shape[1]
    out = torch.empty((T, P, H), device=x.device, dtype=residual.dtype)
    if T == 0 or H == 0:
        return out

    # Launch config (C550-measured; same algorithm is valid on C500 but these
    # BLOCK/num_warps values are per-device tuning evidence, not transferable
    # guarantees).  T==1 decode is latency/launch-dominated and prefers many
    # small blocks (BLOCK_D=256/w4).  Larger T is DRAM-streaming bound: chunk
    # = program_id(0) so consecutive blocks stream one token's contiguous
    # rows, and BLOCK_D=256 with num_warps=1 keeps 8 elements per thread in
    # single-warp blocks (measured ~1.46 TB/s on C550, ~97% of the linear-copy
    # ceiling; 512/w2 -> 1.40 TB/s).
    if T == 1:
        BLOCK_D = 256
        num_warps = 4
    else:
        BLOCK_D = 256
        num_warps = 1
    P_PAD = triton.next_power_of_2(P)
    grid = (triton.cdiv(H, BLOCK_D), T)
    _mhc_post_kernel[grid](
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        out,
        H,
        P=P,
        P_PAD=P_PAD,
        BLOCK_D=BLOCK_D,
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

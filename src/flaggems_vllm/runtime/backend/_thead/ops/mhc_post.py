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

"""PPU (_thead) shape for ``mhc_post``: one CTA per token.

The fat-CTA kernel in ``flaggems_vllm.ops.mhc.mhc_post`` tiles the hidden axis
across programs; this one keeps a single long-lived CTA per token and streams H
in pipelined tiles, which is the shape the vendor tilelang operator uses.

This module is intentionally *not* re-exported from ``_thead.ops.__init__``:
SpecOpRegistrar would then replace the public ``flaggems_vllm.mhc_post`` with
this private launcher.  ``flaggems_vllm.ops.mhc.mhc_post`` imports
``mhc_post_hc4_tokloop`` lazily instead.
"""

import triton
import triton.language as tl

_TOKLOOP_AUTOTUNE_CONFIG = [
    triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_H": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_H": 2048}, num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=_TOKLOOP_AUTOTUNE_CONFIG,
    key=["H"],
)
@triton.jit
def mhc_post_kernel_hc_mult_4_tokloop(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4),   float32
    d_ptr,  # x            : (N, H),    bfloat16
    out_ptr,  # output      : (N, 4, H), bfloat16
    a_stride_n,
    a_stride_j,
    a_stride_i,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Grid: (N,). One long-lived CTA per token; H streamed in BLOCK_H
    tiles with a two-stage pipelined loop (mirrors vllm's tilelang
    mhc_post kernel, which is the fastest shape for this axis)."""
    pid_n = tl.program_id(0)
    a_base = pid_n * a_stride_n
    c_base = pid_n * 4
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    o_base = pid_n * 4 * H

    a00 = tl.load(a_ptr + a_base + 0 * a_stride_j + 0 * a_stride_i)
    a01 = tl.load(a_ptr + a_base + 0 * a_stride_j + 1 * a_stride_i)
    a02 = tl.load(a_ptr + a_base + 0 * a_stride_j + 2 * a_stride_i)
    a03 = tl.load(a_ptr + a_base + 0 * a_stride_j + 3 * a_stride_i)
    a10 = tl.load(a_ptr + a_base + 1 * a_stride_j + 0 * a_stride_i)
    a11 = tl.load(a_ptr + a_base + 1 * a_stride_j + 1 * a_stride_i)
    a12 = tl.load(a_ptr + a_base + 1 * a_stride_j + 2 * a_stride_i)
    a13 = tl.load(a_ptr + a_base + 1 * a_stride_j + 3 * a_stride_i)
    a20 = tl.load(a_ptr + a_base + 2 * a_stride_j + 0 * a_stride_i)
    a21 = tl.load(a_ptr + a_base + 2 * a_stride_j + 1 * a_stride_i)
    a22 = tl.load(a_ptr + a_base + 2 * a_stride_j + 2 * a_stride_i)
    a23 = tl.load(a_ptr + a_base + 2 * a_stride_j + 3 * a_stride_i)
    a30 = tl.load(a_ptr + a_base + 3 * a_stride_j + 0 * a_stride_i)
    a31 = tl.load(a_ptr + a_base + 3 * a_stride_j + 1 * a_stride_i)
    a32 = tl.load(a_ptr + a_base + 3 * a_stride_j + 2 * a_stride_i)
    a33 = tl.load(a_ptr + a_base + 3 * a_stride_j + 3 * a_stride_i)
    p0 = tl.load(c_ptr + c_base + 0)
    p1 = tl.load(c_ptr + c_base + 1)
    p2 = tl.load(c_ptr + c_base + 2)
    p3 = tl.load(c_ptr + c_base + 3)

    for h0 in tl.range(0, H, BLOCK_H, num_stages=2):
        hs = h0 + tl.arange(0, BLOCK_H)
        msk = hs < H
        xv = tl.load(
            d_ptr + d_base + hs,
            mask=msk,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        b0 = tl.load(b_ptr + b_base + 0 * H + hs, mask=msk, other=0.0).to(tl.float32)
        b1 = tl.load(b_ptr + b_base + 1 * H + hs, mask=msk, other=0.0).to(tl.float32)
        b2 = tl.load(b_ptr + b_base + 2 * H + hs, mask=msk, other=0.0).to(tl.float32)
        b3 = tl.load(b_ptr + b_base + 3 * H + hs, mask=msk, other=0.0).to(tl.float32)
        o0 = tl.fma(a30, b3, tl.fma(a20, b2, tl.fma(a10, b1, tl.fma(a00, b0, p0 * xv))))
        o1 = tl.fma(a31, b3, tl.fma(a21, b2, tl.fma(a11, b1, tl.fma(a01, b0, p1 * xv))))
        o2 = tl.fma(a32, b3, tl.fma(a22, b2, tl.fma(a12, b1, tl.fma(a02, b0, p2 * xv))))
        o3 = tl.fma(a33, b3, tl.fma(a23, b2, tl.fma(a13, b1, tl.fma(a03, b0, p3 * xv))))
        tl.store(
            out_ptr + o_base + 0 * H + hs,
            o0.to(tl.bfloat16),
            mask=msk,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + o_base + 1 * H + hs,
            o1.to(tl.bfloat16),
            mask=msk,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + o_base + 2 * H + hs,
            o2.to(tl.bfloat16),
            mask=msk,
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + o_base + 3 * H + hs,
            o3.to(tl.bfloat16),
            mask=msk,
            eviction_policy="evict_first",
        )


def mhc_post_hc4_tokloop(a, b, c, d, out, a_stride_n, a_stride_j, a_stride_i, H):
    """Launch the token loop: grid (N,), one CTA per token, H streamed."""
    mhc_post_kernel_hc_mult_4_tokloop[(a.shape[0],)](
        a, b, c, d, out, a_stride_n, a_stride_j, a_stride_i, H=H
    )

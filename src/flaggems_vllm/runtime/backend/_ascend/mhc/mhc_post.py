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

"""Modified Triton mhc_post matching NPU semantics.

This is a copy of the original since mhc_post already produces outputs
identical to the NPU implementation:

    output[t,i,d] = sum_j h_res[t,j,i] * x[t,j,d] + h_post[t,i] * h_out[t,d]

No semantic changes needed.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HC_MULT=4 fast path
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_kernel_hc4(
    x_ptr,
    h_res_ptr,
    h_out_ptr,
    h_post_ptr,
    out_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    x_base = pid_t * 4 * D
    out_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    hp0 = tl.load(h_post_ptr + hpost_base + 0)
    hp1 = tl.load(h_post_ptr + hpost_base + 1)
    hp2 = tl.load(h_post_ptr + hpost_base + 2)
    hp3 = tl.load(h_post_ptr + hpost_base + 3)

    r00 = tl.load(h_res_ptr + hres_base + 0)
    r01 = tl.load(h_res_ptr + hres_base + 1)
    r02 = tl.load(h_res_ptr + hres_base + 2)
    r03 = tl.load(h_res_ptr + hres_base + 3)
    r10 = tl.load(h_res_ptr + hres_base + 4)
    r11 = tl.load(h_res_ptr + hres_base + 5)
    r12 = tl.load(h_res_ptr + hres_base + 6)
    r13 = tl.load(h_res_ptr + hres_base + 7)
    r20 = tl.load(h_res_ptr + hres_base + 8)
    r21 = tl.load(h_res_ptr + hres_base + 9)
    r22 = tl.load(h_res_ptr + hres_base + 10)
    r23 = tl.load(h_res_ptr + hres_base + 11)
    r30 = tl.load(h_res_ptr + hres_base + 12)
    r31 = tl.load(h_res_ptr + hres_base + 13)
    r32 = tl.load(h_res_ptr + hres_base + 14)
    r33 = tl.load(h_res_ptr + hres_base + 15)

    x0 = tl.load(x_ptr + x_base + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x_base + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + x_base + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + x_base + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    ho = tl.load(h_out_ptr + hout_base + d_off, mask=d_mask, other=0.0).to(tl.float32)

    y0 = hp0 * ho + r00 * x0 + r10 * x1 + r20 * x2 + r30 * x3
    y1 = hp1 * ho + r01 * x0 + r11 * x1 + r21 * x2 + r31 * x3
    y2 = hp2 * ho + r02 * x0 + r12 * x1 + r22 * x2 + r32 * x3
    y3 = hp3 * ho + r03 * x0 + r13 * x1 + r23 * x2 + r33 * x3

    dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + out_base + 0 * D + d_off, y0.to(dtype), mask=d_mask)
    tl.store(out_ptr + out_base + 1 * D + d_off, y1.to(dtype), mask=d_mask)
    tl.store(out_ptr + out_base + 2 * D + d_off, y2.to(dtype), mask=d_mask)
    tl.store(out_ptr + out_base + 3 * D + d_off, y3.to(dtype), mask=d_mask)


# ---------------------------------------------------------------------------
# LoopD + manual 2-stage pipeline: overlaps GM→UB load of d-block (db+1)
# with compute of d-block (db). Mimics AscendC DOUBLE_BUFFER_DEPTH=2.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 896}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1792}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=4, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_kernel_hc4_loopD_pipe(
    x_ptr,
    h_res_ptr,
    h_out_ptr,
    h_post_ptr,
    out_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    # Derived in-kernel so every autotuned BLOCK_D config still covers all D.
    NUM_D_BLOCKS: tl.constexpr = (D + BLOCK_D - 1) // BLOCK_D

    x_base = pid_t * 4 * D
    out_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    hp0 = tl.load(h_post_ptr + hpost_base + 0)
    hp1 = tl.load(h_post_ptr + hpost_base + 1)
    hp2 = tl.load(h_post_ptr + hpost_base + 2)
    hp3 = tl.load(h_post_ptr + hpost_base + 3)

    r00 = tl.load(h_res_ptr + hres_base + 0)
    r01 = tl.load(h_res_ptr + hres_base + 1)
    r02 = tl.load(h_res_ptr + hres_base + 2)
    r03 = tl.load(h_res_ptr + hres_base + 3)
    r10 = tl.load(h_res_ptr + hres_base + 4)
    r11 = tl.load(h_res_ptr + hres_base + 5)
    r12 = tl.load(h_res_ptr + hres_base + 6)
    r13 = tl.load(h_res_ptr + hres_base + 7)
    r20 = tl.load(h_res_ptr + hres_base + 8)
    r21 = tl.load(h_res_ptr + hres_base + 9)
    r22 = tl.load(h_res_ptr + hres_base + 10)
    r23 = tl.load(h_res_ptr + hres_base + 11)
    r30 = tl.load(h_res_ptr + hres_base + 12)
    r31 = tl.load(h_res_ptr + hres_base + 13)
    r32 = tl.load(h_res_ptr + hres_base + 14)
    r33 = tl.load(h_res_ptr + hres_base + 15)

    # ---- Prologue: load d-block 0 ----
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    x0_cur = tl.load(x_ptr + x_base + 0 * D + d_off, mask=d_mask, other=0.0)
    x1_cur = tl.load(x_ptr + x_base + 1 * D + d_off, mask=d_mask, other=0.0)
    x2_cur = tl.load(x_ptr + x_base + 2 * D + d_off, mask=d_mask, other=0.0)
    x3_cur = tl.load(x_ptr + x_base + 3 * D + d_off, mask=d_mask, other=0.0)
    ho_cur = tl.load(h_out_ptr + hout_base + d_off, mask=d_mask, other=0.0)

    for db in range(NUM_D_BLOCKS):
        # ---- Issue loads for d-block (db+1) FIRST (async with compute below) ----
        # Loop guard makes NUM_D_BLOCKS >= 1, so (db+1) % NUM_D_BLOCKS is safe.
        nxt = (db + 1) % NUM_D_BLOCKS
        d_off_nxt = nxt * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask_nxt = d_off_nxt < D
        x0_nxt = tl.load(x_ptr + x_base + 0 * D + d_off_nxt, mask=d_mask_nxt, other=0.0)
        x1_nxt = tl.load(x_ptr + x_base + 1 * D + d_off_nxt, mask=d_mask_nxt, other=0.0)
        x2_nxt = tl.load(x_ptr + x_base + 2 * D + d_off_nxt, mask=d_mask_nxt, other=0.0)
        x3_nxt = tl.load(x_ptr + x_base + 3 * D + d_off_nxt, mask=d_mask_nxt, other=0.0)
        ho_nxt = tl.load(h_out_ptr + hout_base + d_off_nxt, mask=d_mask_nxt, other=0.0)

        # ---- Compute on current d-block (overlaps with next loads) ----
        x0_f = x0_cur.to(tl.float32)
        x1_f = x1_cur.to(tl.float32)
        x2_f = x2_cur.to(tl.float32)
        x3_f = x3_cur.to(tl.float32)
        ho_f = ho_cur.to(tl.float32)

        y0 = hp0 * ho_f + r00 * x0_f + r10 * x1_f + r20 * x2_f + r30 * x3_f
        y1 = hp1 * ho_f + r01 * x0_f + r11 * x1_f + r21 * x2_f + r31 * x3_f
        y2 = hp2 * ho_f + r02 * x0_f + r12 * x1_f + r22 * x2_f + r32 * x3_f
        y3 = hp3 * ho_f + r03 * x0_f + r13 * x1_f + r23 * x2_f + r33 * x3_f

        # ---- Store current (overlaps with next compute) ----
        d_off_cur = db * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask_cur = d_off_cur < D
        dtype = out_ptr.dtype.element_ty
        tl.store(out_ptr + out_base + 0 * D + d_off_cur, y0.to(dtype), mask=d_mask_cur)
        tl.store(out_ptr + out_base + 1 * D + d_off_cur, y1.to(dtype), mask=d_mask_cur)
        tl.store(out_ptr + out_base + 2 * D + d_off_cur, y2.to(dtype), mask=d_mask_cur)
        tl.store(out_ptr + out_base + 3 * D + d_off_cur, y3.to(dtype), mask=d_mask_cur)

        # ---- Rotate buffers (modulo loads make this unconditional) ----
        x0_cur = x0_nxt
        x1_cur = x1_nxt
        x2_cur = x2_nxt
        x3_cur = x3_nxt
        ho_cur = ho_nxt


# ---------------------------------------------------------------------------
# BLOCK_T kernel: process BLOCK_T tokens per program (grid=(cdiv(T,BLOCK_T),)).
# Backward taught us fewer/fatter programs win (grid=(T,) beat the 2D grid).
# The forward does little work per token, so launching T programs leaves each
# program overhead-dominated; batching tokens fattens each program. BLOCK_D=D
# (single 3584 block) matches the measured-best forward config.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 8}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_T": 16}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_kernel_hc4_blockT(
    x_ptr,
    h_res_ptr,
    h_out_ptr,
    h_post_ptr,
    out_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    d_off = tl.arange(0, D)
    dtype = out_ptr.dtype.element_ty

    for tt in range(BLOCK_T):
        t = pid * BLOCK_T + tt
        if t < T:
            x_base = t * 4 * D
            hout_base = t * D
            hres_base = t * 16
            hpost_base = t * 4

            hp0 = tl.load(h_post_ptr + hpost_base + 0)
            hp1 = tl.load(h_post_ptr + hpost_base + 1)
            hp2 = tl.load(h_post_ptr + hpost_base + 2)
            hp3 = tl.load(h_post_ptr + hpost_base + 3)

            r00 = tl.load(h_res_ptr + hres_base + 0)
            r01 = tl.load(h_res_ptr + hres_base + 1)
            r02 = tl.load(h_res_ptr + hres_base + 2)
            r03 = tl.load(h_res_ptr + hres_base + 3)
            r10 = tl.load(h_res_ptr + hres_base + 4)
            r11 = tl.load(h_res_ptr + hres_base + 5)
            r12 = tl.load(h_res_ptr + hres_base + 6)
            r13 = tl.load(h_res_ptr + hres_base + 7)
            r20 = tl.load(h_res_ptr + hres_base + 8)
            r21 = tl.load(h_res_ptr + hres_base + 9)
            r22 = tl.load(h_res_ptr + hres_base + 10)
            r23 = tl.load(h_res_ptr + hres_base + 11)
            r30 = tl.load(h_res_ptr + hres_base + 12)
            r31 = tl.load(h_res_ptr + hres_base + 13)
            r32 = tl.load(h_res_ptr + hres_base + 14)
            r33 = tl.load(h_res_ptr + hres_base + 15)

            x0 = tl.load(x_ptr + x_base + 0 * D + d_off).to(tl.float32)
            x1 = tl.load(x_ptr + x_base + 1 * D + d_off).to(tl.float32)
            x2 = tl.load(x_ptr + x_base + 2 * D + d_off).to(tl.float32)
            x3 = tl.load(x_ptr + x_base + 3 * D + d_off).to(tl.float32)
            ho = tl.load(h_out_ptr + hout_base + d_off).to(tl.float32)

            y0 = hp0 * ho + r00 * x0 + r10 * x1 + r20 * x2 + r30 * x3
            y1 = hp1 * ho + r01 * x0 + r11 * x1 + r21 * x2 + r31 * x3
            y2 = hp2 * ho + r02 * x0 + r12 * x1 + r22 * x2 + r32 * x3
            y3 = hp3 * ho + r03 * x0 + r13 * x1 + r23 * x2 + r33 * x3

            tl.store(out_ptr + x_base + 0 * D + d_off, y0.to(dtype))
            tl.store(out_ptr + x_base + 1 * D + d_off, y1.to(dtype))
            tl.store(out_ptr + x_base + 2 * D + d_off, y2.to(dtype))
            tl.store(out_ptr + x_base + 3 * D + d_off, y3.to(dtype))


def _flatten_bsn(x, h_res, h_out, h_post):
    if x.dim() == 4:
        B, S, N, D = x.shape
        T = B * S
        x = x.reshape(T, N, D).contiguous()
        h_res = h_res.reshape(T, N, N).contiguous()
        h_out = h_out.reshape(T, D).contiguous()
        h_post = h_post.reshape(T, N).contiguous()
        return x, h_res, h_out, h_post, (B, S, N, D)
    if x.dim() == 3:
        T, N, D = x.shape
        return (
            x.contiguous(),
            h_res.reshape(T, N, N).contiguous(),
            h_out.reshape(T, D).contiguous(),
            h_post.reshape(T, N).contiguous(),
            (T, N, D),
        )
    raise ValueError(f"unsupported x.dim()={x.dim()}, expect 3 or 4")


def mhc_post(x, h_res, h_out, h_post):
    # Fused fast-path: fold _flatten_bsn into the kernel launch.
    #
    # The blockT kernel does pure FLAT pointer arithmetic (x_base = t*4*D,
    # hres_base = t*16, ...), so it only cares that the underlying buffer is
    # laid out as [T][4][D] contiguously -- the logical ndim (3D or 4D) is
    # irrelevant. For contiguous input, _flatten_bsn's reshapes are free views
    # and its .contiguous() calls are no-ops, yet they still cost ~18us of aten
    # dispatch that is fully exposed at small T (T<=512). We fold the flatten
    # into the launch: pass the raw tensor, derive T = numel // (4*D), and the
    # output keeps the input's shape (no output reshape needed).
    if (
        x.shape[-2] == 4
        and x.is_contiguous()
        and h_res.is_contiguous()
        and h_out.is_contiguous()
        and h_post.is_contiguous()
    ):
        D = x.shape[-1]
        T = x.numel() // (4 * D)
        if T >= 64:
            out = torch.empty_like(x)
            grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]),)
            _mhc_post_kernel_hc4_blockT[grid](x, h_res, h_out, h_post, out, T=T, D=D)
            return out

    xf, hres, hout, hpost, shape = _flatten_bsn(x, h_res, h_out, h_post)
    T, N, D = xf.shape
    out = torch.empty_like(xf)

    if N == 4:
        if T >= 64:
            grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]),)
            # BLOCK_T kernel: batch tokens per program to amortize per-program
            # overhead (backward-inspired: fewer/fatter programs).
            _mhc_post_kernel_hc4_blockT[grid](
                xf,
                hres,
                hout,
                hpost,
                out,
                T=T,
                D=D,
            )
        else:
            grid = lambda META: (T, triton.cdiv(D, META["BLOCK_D"]))
            _mhc_post_kernel_hc4[grid](xf, hres, hout, hpost, out, D=D)
    else:
        raise NotImplementedError("Only N=4 supported")

    return out.reshape(shape)


def mhc_post_ref(x, h_res, h_out, h_post):
    orig_dtype = x.dtype
    xf, hres, hout, hpost, shape = _flatten_bsn(x, h_res, h_out, h_post)
    x_f = xf.float()
    hout_f = hout.float()
    mix = torch.einsum("tji,tjd->tid", hres, x_f)
    outer = hpost.unsqueeze(-1) * hout_f.unsqueeze(1)
    y = (mix + outer).to(orig_dtype)
    return y.reshape(shape)

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

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

# Block/warp configuration per shape family (selected by measurement).
_BR_BIG = 64  # rows (of NUM_HEADS lanes) per device program, multi-row family
_WARPS_BIG = 2  # num_warps, multi-row family
_WARPS_SMALL = 1  # num_warps, single-row (tiny) family


@triton.jit
def _gated_softplus(x, beta, THRESH: tl.constexpr, BETA_ONE: tl.constexpr):
    """Reference softplus_beta expression, bit-exactly folded when beta == 1.0."""
    if BETA_ONE:
        return tl.where(x <= THRESH, tl.log(1.0 + tl.exp(x)), x)
    bx = beta * x
    return tl.where(bx <= THRESH, (1.0 / beta) * tl.log(1.0 + tl.exp(bx)), x)


@triton.jit
def _gdn_gating_fast(
    g_ptr,
    bo_ptr,
    A_ptr,
    a_ptr,
    b_ptr,
    dt_ptr,
    NUM_HEADS: tl.constexpr,
    BR: tl.constexpr,
    THRESH: tl.constexpr,
):
    """Fully covered block, beta == 1.0: no mask, no runtime scalars."""
    pid = tl.program_id(0)
    offs = pid * (BR * NUM_HEADS) + tl.arange(0, BR * NUM_HEADS)
    offs = tl.max_contiguous(tl.multiple_of(offs, BR * NUM_HEADS), BR * NUM_HEADS)

    hs = tl.arange(0, NUM_HEADS)
    dt_par = tl.load(dt_ptr + hs).to(tl.float32)
    neg_exp_A_par = -tl.exp(tl.load(A_ptr + hs).to(tl.float32))
    dt_v = tl.reshape(
        tl.broadcast_to(dt_par[None, :], (BR, NUM_HEADS)), (BR * NUM_HEADS,)
    )
    neg_exp_A = tl.reshape(
        tl.broadcast_to(neg_exp_A_par[None, :], (BR, NUM_HEADS)), (BR * NUM_HEADS,)
    )

    a_v = tl.load(a_ptr + offs).to(tl.float32)
    b_v = tl.load(b_ptr + offs).to(tl.float32)
    x = a_v + dt_v
    sp = tl.where(x <= THRESH, tl.log(1.0 + tl.exp(x)), x)
    tl.store(g_ptr + offs, neg_exp_A * sp)
    tl.store(bo_ptr + offs, tl.sigmoid(b_v).to(bo_ptr.dtype.element_ty))


@triton.jit
def _gdn_gating_row(
    g_ptr,
    bo_ptr,
    A_ptr,
    a_ptr,
    b_ptr,
    dt_ptr,
    HS_A: tl.constexpr,
    HS_G: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    THRESH: tl.constexpr,
    UNIT_STRIDE: tl.constexpr,
):
    """Exactly one valid row: the whole region is one contiguous lane vector.

    No row index, no BLK arithmetic, no broadcast/reshape of the parameter
    vectors and no mask.  When both last-dim strides are 1 (the measured
    layouts) the UNIT_STRIDE branch uses bare tl.arange offsets so no stride
    multiplier reaches the address computation at all.
    """
    hs = tl.arange(0, NUM_HEADS)
    if UNIT_STRIDE:
        a_v = tl.load(a_ptr + hs)
        b_v = tl.load(b_ptr + hs)
    else:
        a_v = tl.load(a_ptr + hs * HS_A)
        b_v = tl.load(b_ptr + hs * HS_A)
    dt_v = tl.load(dt_ptr + hs).to(tl.float32)
    A_v = tl.load(A_ptr + hs).to(tl.float32)
    if UNIT_STRIDE:
        tl.store(
            bo_ptr + hs,
            tl.sigmoid(b_v.to(tl.float32)).to(bo_ptr.dtype.element_ty),
        )
    else:
        tl.store(
            bo_ptr + hs * HS_G,
            tl.sigmoid(b_v.to(tl.float32)).to(bo_ptr.dtype.element_ty),
        )
    neg_exp_A = -tl.exp(A_v)
    x = a_v.to(tl.float32) + dt_v
    sp = tl.where(x <= THRESH, tl.log(1.0 + tl.exp(x)), x)
    if UNIT_STRIDE:
        tl.store(g_ptr + hs, neg_exp_A * sp)
    else:
        tl.store(g_ptr + hs * HS_G, neg_exp_A * sp)


@triton.jit
def _gdn_gating_kernel(
    g_ptr,
    bo_ptr,
    A_ptr,
    a_ptr,
    b_ptr,
    dt_ptr,
    n_elems,
    beta,
    NUM_HEADS: tl.constexpr,
    BR: tl.constexpr,
    EVEN: tl.constexpr,
    THRESH: tl.constexpr,
    BETA_ONE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * (BR * NUM_HEADS) + tl.arange(0, BR * NUM_HEADS)
    offs = tl.max_contiguous(tl.multiple_of(offs, BR * NUM_HEADS), BR * NUM_HEADS)

    hs = tl.arange(0, NUM_HEADS)
    dt_par = tl.load(dt_ptr + hs).to(tl.float32)
    neg_exp_A_par = -tl.exp(tl.load(A_ptr + hs).to(tl.float32))
    dt_v = tl.reshape(
        tl.broadcast_to(dt_par[None, :], (BR, NUM_HEADS)), (BR * NUM_HEADS,)
    )
    neg_exp_A = tl.reshape(
        tl.broadcast_to(neg_exp_A_par[None, :], (BR, NUM_HEADS)), (BR * NUM_HEADS,)
    )

    if EVEN:
        a_v = tl.load(a_ptr + offs).to(tl.float32)
        b_v = tl.load(b_ptr + offs).to(tl.float32)
        sp = _gated_softplus(a_v + dt_v, beta, THRESH, BETA_ONE)
        tl.store(g_ptr + offs, neg_exp_A * sp)
        tl.store(bo_ptr + offs, tl.sigmoid(b_v).to(bo_ptr.dtype.element_ty))
    else:
        mask = offs < n_elems
        a_v = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b_v = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sp = _gated_softplus(a_v + dt_v, beta, THRESH, BETA_ONE)
        tl.store(g_ptr + offs, neg_exp_A * sp, mask=mask)
        tl.store(bo_ptr + offs, tl.sigmoid(b_v).to(bo_ptr.dtype.element_ty), mask=mask)


def _plan(n_elems, num_heads):
    nrows = (n_elems + num_heads - 1) // num_heads
    if nrows <= 1:
        br = 1
        warps = _WARPS_SMALL
    else:
        br = _BR_BIG
        if br > nrows:
            br = nrows
        warps = _WARPS_BIG
    nblk = (nrows + br - 1) // br
    even = (nblk * br * num_heads) == n_elems
    return br, nblk, even, warps


@torch.inference_mode()
def _launch(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS,
    NUM_BATCHES,
    beta,
    threshold,
    BLK_HEADS,
    BLK_BATCHES,
    ROW_ITER,
    *,
    grid=None,
    warmup=None,
):
    nh = int(NUM_HEADS)
    n = int(a.numel())
    br, nblk, even, warps = _plan(n, nh)
    beta = float(beta)
    if beta == 1.0 and n == nh and nh > 0 and (nh & (nh - 1)) == 0:
        hs_a = int(a.stride(-1))
        hs_g = int(g.stride(-1))
        if (
            hs_a > 0
            and hs_g > 0
            and hs_a == int(b.stride(-1))
            and hs_g == int(beta_output.stride(-1))
        ):
            _gdn_gating_row[(1,)](
                g,
                beta_output,
                A_log,
                a,
                b,
                dt_bias,
                HS_A=hs_a,
                HS_G=hs_g,
                NUM_HEADS=nh,
                THRESH=float(threshold),
                UNIT_STRIDE=(hs_a == 1 and hs_g == 1),
                num_warps=_WARPS_SMALL,
                num_stages=1,
            )
            return None
    if even and beta == 1.0:
        _gdn_gating_fast[(nblk,)](
            g,
            beta_output,
            A_log,
            a,
            b,
            dt_bias,
            NUM_HEADS=nh,
            BR=br,
            THRESH=float(threshold),
            num_warps=warps,
            num_stages=1,
        )
    else:
        _gdn_gating_kernel[(nblk,)](
            g,
            beta_output,
            A_log,
            a,
            b,
            dt_bias,
            n,
            beta,
            NUM_HEADS=nh,
            BR=br,
            EVEN=even,
            THRESH=float(threshold),
            BETA_ONE=(beta == 1.0),
            num_warps=warps,
        )
    return None


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """Return the FP32 decay and BF16 gate as [1, batch, heads]."""
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("expected matching [batch, heads] activation tensors")
    batch, heads = a.shape
    if heads < 1 or heads > 64 or heads & (heads - 1):
        raise ValueError("heads must be a power of two in [1, 64]")
    if A_log.shape != (heads,) or dt_bias.shape != (heads,):
        raise ValueError("decay and bias must have one entry per head")
    if (
        a.dtype != torch.bfloat16
        or b.dtype != a.dtype
        or A_log.dtype != torch.float32
        or dt_bias.dtype != torch.bfloat16
    ):
        raise ValueError("expected BF16 activations and bias, FP32 A_log")
    if (
        not all(
            t.is_contiguous() and t.device == a.device for t in (a, b, A_log, dt_bias)
        )
        or a.device.type != "npu"
    ):
        raise ValueError("inputs must be contiguous on one NPU")
    if (
        not isinstance(beta, (int, float))
        or not math.isfinite(beta)
        or beta <= 0
        or not math.isfinite(threshold)
    ):
        raise ValueError("beta must be finite and positive and threshold finite")
    g = torch.empty((1, batch, heads), dtype=torch.float32, device=a.device)
    gate = torch.empty_like(g, dtype=b.dtype)
    if batch:
        with torch_device_fn.device(a.device):
            _launch(
                g,
                gate,
                A_log,
                a,
                b,
                dt_bias,
                1,
                heads,
                batch,
                beta,
                threshold,
                8,
                64,
                1,
            )
    return g, gate

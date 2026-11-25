# SPDX-License-Identifier: Apache-2.0
"""Generic PLE gate implementation (fallback)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ple_gate_kernel(
    key_ptr,
    value_ptr,
    hidden_ptr,
    nk_ptr,
    nq_ptr,
    ncw_ptr,
    gated_ptr,
    normed_ptr,
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    mask = lanes < H
    offs = s * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    k = tl.load(key_ptr + t * key_rs + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(hidden_ptr + t * HC * H + offs, mask=mask, other=0.0).to(tl.float32)
    nk = tl.load(nk_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    nq = tl.load(nq_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    k_n = (k * tl.rsqrt(tl.sum(k * k) / H + eps) * (1.0 + nk)).to(dtype)
    q_n = (q * tl.rsqrt(tl.sum(q * q) / H + eps) * (1.0 + nq)).to(dtype)
    products = (k_n.to(tl.float32) * q_n.to(tl.float32)).to(dtype)
    dot = tl.sum(products.to(tl.float32)).to(dtype).to(tl.float32)
    d = dot / tl.sqrt(float(H))
    d = d.to(dtype).to(tl.float32)
    sign = tl.where(d < 0, -1.0, 0.0)
    sign = tl.where(d > 0, 1.0, sign)
    magnitude = tl.sqrt(tl.maximum(tl.abs(d), 1e-6)).to(dtype)
    g = tl.sigmoid(sign * magnitude).to(dtype).to(tl.float32)

    v = tl.load(value_ptr + t * value_rs + lanes, mask=mask, other=0.0).to(tl.float32)
    gated = (g * v).to(dtype)
    gf = gated.to(tl.float32)
    ncw = tl.load(ncw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = gf * tl.rsqrt(tl.sum(gf * gf) / H + eps) * (1.0 + ncw)

    tl.store(gated_ptr + t * HC * H + offs, gated, mask=mask)
    tl.store(normed_ptr + t * HC * H + offs, normed, mask=mask)


def ple_gate(key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps):
    num_tokens = hidden.shape[0]
    h = value.shape[-1]
    hc = hidden.shape[-1] // h
    gated = torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    _ple_gate_kernel[(num_tokens, hc)](
        key,
        value,
        hidden,
        norm_key_w,
        norm_query_w,
        norm_conv_w,
        gated,
        normed,
        key.stride(0),
        value.stride(0),
        eps,
        H=h,
        HC=hc,
        BLOCK_H=triton.next_power_of_2(h),
        num_warps=4,
    )
    return gated, normed

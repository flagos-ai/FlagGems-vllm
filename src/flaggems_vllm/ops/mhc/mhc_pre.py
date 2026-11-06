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

"""
Triton implementation of mHC Pre operator (optimized v2).

Key optimizations:
- GEMM: our own Triton projection kernel (fp32 out, masked K)
- sqrsum + norm + mix + sinkhorn + weighted sum: single fused Triton kernel
  Two passes over residual: pass 1 computes sqrsum, pass 2 does weighted sum
"""

import importlib
import os
import warnings
import weakref

import torch
import triton
import triton.language as tl

_MHC_PRE_MIX_PAD = tl.constexpr(32)

_VENDOR = None

_HYGON = None


_MHC_FN_CACHE: dict[int, tuple[weakref.ref, int, int, dict]] = {}


def _mhc_fn_cache(fn: torch.Tensor) -> dict:
    """Store for the weight copies this op derives, validated by identity plus
    _version and data_ptr so that an in-place edit invalidates them.
    """
    key = id(fn)
    version = fn._version
    ptr = fn.data_ptr()
    entry = _MHC_FN_CACHE.get(key)
    if entry is not None:
        ref, cached_version, cached_ptr, store = entry
        if ref() is fn and cached_version == version and cached_ptr == ptr:
            return store

    def expire(ref, key=key):
        cached = _MHC_FN_CACHE.get(key)
        if cached is not None and cached[0] is ref:
            _MHC_FN_CACHE.pop(key)

    if len(_MHC_FN_CACHE) > 64:
        _MHC_FN_CACHE.clear()
    store: dict[str, torch.Tensor] = {}
    _MHC_FN_CACHE[key] = (weakref.ref(fn, expire), version, ptr, store)
    return store


def _get_fn_bf16_cached(fn: torch.Tensor):
    """bf16 copy of the weights and its transposed view."""
    if fn.requires_grad:
        fn_bf16 = _mhc_copy_as(fn, torch.bfloat16)
        return fn_bf16, fn_bf16.t()
    store = _mhc_fn_cache(fn)
    fn_bf16 = store.get("bf16")
    if fn_bf16 is None:
        fn_bf16 = _mhc_copy_as(fn, torch.bfloat16)
        store["bf16"] = fn_bf16
        store["bf16_t"] = fn_bf16.t()
    return fn_bf16, store["bf16_t"]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["num_tokens", "hc_hidden_size", "HC_MULT3"],
)
@triton.jit
def mhc_pre_proj_gemm_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, HC_MULT3) bfloat16, fn^T possibly zero padded
    out_ptr,  # (num_tokens, HC_MULT3) float32
    num_tokens,
    hc_hidden_size,
    a_stride_n,
    bt_stride_k,
    bt_stride_j,
    HC_MULT3: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``x @ fn^T`` for the structures that keep the projection outside the
    epilogue kernel.  K is masked, so no alignment is required of hc_hidden_size.
    """
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens
    n_mask = offs_n < HC_MULT3
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hc_hidden_size
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            bt_ptr + offs_k[:, None] * bt_stride_k + offs_n[None, :] * bt_stride_j,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(
        out_ptr + offs_m[:, None] * HC_MULT3 + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _mhc_project(x_flat, fn, num_tokens, hc_hidden_size, hc_mult3, device):
    """`x_flat @ fn^T` as float32 (num_tokens, hc_mult3), without a torch GEMM."""
    out = torch.empty(num_tokens, hc_mult3, dtype=torch.float32, device=device)
    if fn.shape[0] <= _MHC_PRE_MIX_PAD.value:
        bt = _mhc_pre_fn_packed_cached(fn)  # (hc_hidden, 32), rows contiguous
    else:
        bt = _get_fn_bf16_cached(fn)[0].t()  # wider mix row: no packed form
    block_n = 16
    while block_n < hc_mult3:
        block_n *= 2
    mhc_pre_proj_gemm_kernel[lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_M"]),)](
        x_flat,
        bt,
        out,
        num_tokens,
        hc_hidden_size,
        x_flat.stride(0),
        bt.stride(0),
        bt.stride(1),
        HC_MULT3=hc_mult3,
        BLOCK_N=block_n,
    )
    return out


@triton.jit
def _mhc_epilogue_token(
    m0,
    m1,
    m2,
    m3,
    m4,
    m5,
    m6,
    m7,
    m8,
    m9,
    m10,
    m11,
    m12,
    m13,
    m14,
    m15,
    m16,
    m17,
    m18,
    m19,
    m20,
    m21,
    m22,
    m23,
    rms_inv,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    pid_n,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """mixes at indices 0..23 -> pre/post mix, Sinkhorn and the weighted sum.

    Shared by the two one-token-per-program structures.
    """
    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)
    res_base = pid_n * res_stride_n
    pre_mix_0 = (
        tl.sigmoid(m0 * rms_inv * scale_0 + tl.load(hc_base_ptr + 0)) + hc_pre_eps
    )
    pre_mix_1 = (
        tl.sigmoid(m1 * rms_inv * scale_0 + tl.load(hc_base_ptr + 1)) + hc_pre_eps
    )
    pre_mix_2 = (
        tl.sigmoid(m2 * rms_inv * scale_0 + tl.load(hc_base_ptr + 2)) + hc_pre_eps
    )
    pre_mix_3 = (
        tl.sigmoid(m3 * rms_inv * scale_0 + tl.load(hc_base_ptr + 3)) + hc_pre_eps
    )

    post_0 = (
        tl.sigmoid(m4 * rms_inv * scale_1 + tl.load(hc_base_ptr + 4))
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 0, post_0)
    post_1 = (
        tl.sigmoid(m5 * rms_inv * scale_1 + tl.load(hc_base_ptr + 5))
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 1, post_1)
    post_2 = (
        tl.sigmoid(m6 * rms_inv * scale_1 + tl.load(hc_base_ptr + 6))
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 2, post_2)
    post_3 = (
        tl.sigmoid(m7 * rms_inv * scale_1 + tl.load(hc_base_ptr + 7))
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + pid_n * 4 + 3, post_3)

    cb = 8
    cm_00 = m8 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 0)
    cm_01 = m9 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 1)
    cm_02 = m10 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 2)
    cm_03 = m11 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 3)
    cm_10 = m12 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 4)
    cm_11 = m13 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 5)
    cm_12 = m14 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 6)
    cm_13 = m15 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 7)
    cm_20 = m16 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 8)
    cm_21 = m17 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 9)
    cm_22 = m18 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 10)
    cm_23 = m19 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 11)
    cm_30 = m20 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 12)
    cm_31 = m21 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 13)
    cm_32 = m22 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 14)
    cm_33 = m23 * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 15)

    (
        cm_00,
        cm_01,
        cm_02,
        cm_03,
        cm_10,
        cm_11,
        cm_12,
        cm_13,
        cm_20,
        cm_21,
        cm_22,
        cm_23,
        cm_30,
        cm_31,
        cm_32,
        cm_33,
    ) = _sinkhorn_iters(
        cm_00,
        cm_01,
        cm_02,
        cm_03,
        cm_10,
        cm_11,
        cm_12,
        cm_13,
        cm_20,
        cm_21,
        cm_22,
        cm_23,
        cm_30,
        cm_31,
        cm_32,
        cm_33,
        hc_sinkhorn_eps,
        sinkhorn_repeat,
    )

    co = pid_n * 16
    tl.store(comb_mix_ptr + co + 0, cm_00)
    tl.store(comb_mix_ptr + co + 1, cm_01)
    tl.store(comb_mix_ptr + co + 2, cm_02)
    tl.store(comb_mix_ptr + co + 3, cm_03)
    tl.store(comb_mix_ptr + co + 4, cm_10)
    tl.store(comb_mix_ptr + co + 5, cm_11)
    tl.store(comb_mix_ptr + co + 6, cm_12)
    tl.store(comb_mix_ptr + co + 7, cm_13)
    tl.store(comb_mix_ptr + co + 8, cm_20)
    tl.store(comb_mix_ptr + co + 9, cm_21)
    tl.store(comb_mix_ptr + co + 10, cm_22)
    tl.store(comb_mix_ptr + co + 11, cm_23)
    tl.store(comb_mix_ptr + co + 12, cm_30)
    tl.store(comb_mix_ptr + co + 13, cm_31)
    tl.store(comb_mix_ptr + co + 14, cm_32)
    tl.store(comb_mix_ptr + co + 15, cm_33)

    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        r0 = tl.load(
            residual_ptr + res_base + 0 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        r1 = tl.load(
            residual_ptr + res_base + 1 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        acc = pre_mix_0 * r0 + pre_mix_1 * r1
        r2 = tl.load(
            residual_ptr + res_base + 2 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        r3 = tl.load(
            residual_ptr + res_base + 3 * res_stride_i + h_offsets * res_stride_h,
            mask=h_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        acc += pre_mix_2 * r2 + pre_mix_3 * r3
        tl.store(
            layer_input_ptr + pid_n * li_stride_n + h_offsets * li_stride_h,
            acc.to(tl.bfloat16),
            mask=h_mask,
        )


@triton.jit
def _mhc_mix_at(mix_ptr, offs, tmask, MASKED: tl.constexpr):
    """One mix row entry. The split structures read a padded (n, 32) scratch,
    the mm structures a (n, hc_mult3) projection, so masking is per call site.
    """
    if MASKED:
        v = tl.load(mix_ptr + offs, mask=tmask, other=0.0)
    else:
        v = tl.load(mix_ptr + offs)
    return v.to(tl.float32)


@triton.jit
def _mhc_epilogue_tile(
    mix_ptr,
    rms_inv,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    offs_t,
    tmask,
    res_row,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    MIX_STRIDE: tl.constexpr,
    MIX_MASKED: tl.constexpr,
    H_STRIDE_ONE: tl.constexpr,
    H_EXACT: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Same epilogue with TOKEN_TILE tokens per program, so the serial
    Sinkhorn chain runs as [T]-vector math.

    MIX_STRIDE / MIX_MASKED describe how the mix rows are laid out,
    H_STRIDE_ONE / H_EXACT how the residual is addressed, so that each caller
    gets the code its own layout wants instead of a general form.
    """
    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    pre_mix_0 = (
        tl.sigmoid(
            _mhc_mix_at(mix_ptr, offs_t * MIX_STRIDE + 0, tmask, MIX_MASKED)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 0)
        )
        + hc_pre_eps
    )
    pre_mix_1 = (
        tl.sigmoid(
            _mhc_mix_at(mix_ptr, offs_t * MIX_STRIDE + 1, tmask, MIX_MASKED)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 1)
        )
        + hc_pre_eps
    )
    pre_mix_2 = (
        tl.sigmoid(
            _mhc_mix_at(mix_ptr, offs_t * MIX_STRIDE + 2, tmask, MIX_MASKED)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 2)
        )
        + hc_pre_eps
    )
    pre_mix_3 = (
        tl.sigmoid(
            _mhc_mix_at(mix_ptr, offs_t * MIX_STRIDE + 3, tmask, MIX_MASKED)
            * rms_inv
            * scale_0
            + tl.load(hc_base_ptr + 3)
        )
        + hc_pre_eps
    )

    post_0 = (
        tl.sigmoid(
            _mhc_mix_at(mix_ptr, offs_t * MIX_STRIDE + 4, tmask, MIX_MASKED)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 4)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + offs_t * 4 + 0, post_0, mask=tmask)
    post_1 = (
        tl.sigmoid(
            _mhc_mix_at(mix_ptr, offs_t * MIX_STRIDE + 5, tmask, MIX_MASKED)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 5)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + offs_t * 4 + 1, post_1, mask=tmask)
    post_2 = (
        tl.sigmoid(
            _mhc_mix_at(mix_ptr, offs_t * MIX_STRIDE + 6, tmask, MIX_MASKED)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 6)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + offs_t * 4 + 2, post_2, mask=tmask)
    post_3 = (
        tl.sigmoid(
            _mhc_mix_at(mix_ptr, offs_t * MIX_STRIDE + 7, tmask, MIX_MASKED)
            * rms_inv
            * scale_1
            + tl.load(hc_base_ptr + 7)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + offs_t * 4 + 3, post_3, mask=tmask)

    cb = 8
    cm_00 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 8, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 0)
    cm_01 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 9, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 1)
    cm_02 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 10, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 2)
    cm_03 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 11, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 3)
    cm_10 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 12, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 4)
    cm_11 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 13, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 5)
    cm_12 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 14, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 6)
    cm_13 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 15, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 7)
    cm_20 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 16, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 8)
    cm_21 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 17, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 9)
    cm_22 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 18, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 10)
    cm_23 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 19, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 11)
    cm_30 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 20, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 12)
    cm_31 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 21, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 13)
    cm_32 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 22, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 14)
    cm_33 = _mhc_mix_at(
        mix_ptr, offs_t * MIX_STRIDE + 23, tmask, MIX_MASKED
    ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + 15)

    (
        cm_00,
        cm_01,
        cm_02,
        cm_03,
        cm_10,
        cm_11,
        cm_12,
        cm_13,
        cm_20,
        cm_21,
        cm_22,
        cm_23,
        cm_30,
        cm_31,
        cm_32,
        cm_33,
    ) = _sinkhorn_iters(
        cm_00,
        cm_01,
        cm_02,
        cm_03,
        cm_10,
        cm_11,
        cm_12,
        cm_13,
        cm_20,
        cm_21,
        cm_22,
        cm_23,
        cm_30,
        cm_31,
        cm_32,
        cm_33,
        hc_sinkhorn_eps,
        sinkhorn_repeat,
    )

    co = offs_t * 16
    tl.store(comb_mix_ptr + co + 0, cm_00, mask=tmask)
    tl.store(comb_mix_ptr + co + 1, cm_01, mask=tmask)
    tl.store(comb_mix_ptr + co + 2, cm_02, mask=tmask)
    tl.store(comb_mix_ptr + co + 3, cm_03, mask=tmask)
    tl.store(comb_mix_ptr + co + 4, cm_10, mask=tmask)
    tl.store(comb_mix_ptr + co + 5, cm_11, mask=tmask)
    tl.store(comb_mix_ptr + co + 6, cm_12, mask=tmask)
    tl.store(comb_mix_ptr + co + 7, cm_13, mask=tmask)
    tl.store(comb_mix_ptr + co + 8, cm_20, mask=tmask)
    tl.store(comb_mix_ptr + co + 9, cm_21, mask=tmask)
    tl.store(comb_mix_ptr + co + 10, cm_22, mask=tmask)
    tl.store(comb_mix_ptr + co + 11, cm_23, mask=tmask)
    tl.store(comb_mix_ptr + co + 12, cm_30, mask=tmask)
    tl.store(comb_mix_ptr + co + 13, cm_31, mask=tmask)
    tl.store(comb_mix_ptr + co + 14, cm_32, mask=tmask)
    tl.store(comb_mix_ptr + co + 15, cm_33, mask=tmask)

    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        if H_EXACT:
            hmask = tmask[:, None]
        else:
            hmask = tmask[:, None] & (h_offsets < hidden_size)[None, :]
        if H_STRIDE_ONE:
            rbase = residual_ptr + res_row[:, None] + h_offsets[None, :]
        else:
            rbase = residual_ptr + res_row[:, None] + h_offsets[None, :] * res_stride_h
        r0 = tl.load(rbase, mask=hmask, other=0.0).to(tl.float32)
        r1 = tl.load(rbase + res_stride_i, mask=hmask, other=0.0).to(tl.float32)
        acc_li = pre_mix_0[:, None] * r0 + pre_mix_1[:, None] * r1
        r2 = tl.load(rbase + 2 * res_stride_i, mask=hmask, other=0.0).to(tl.float32)
        r3 = tl.load(rbase + 3 * res_stride_i, mask=hmask, other=0.0).to(tl.float32)
        acc_li += pre_mix_2[:, None] * r2 + pre_mix_3[:, None] * r3
        tl.store(
            layer_input_ptr
            + (offs_t * li_stride_n)[:, None]
            + h_offsets[None, :] * li_stride_h,
            acc_li.to(tl.bfloat16),
            mask=hmask,
        )


@triton.jit
def _mm_fuse_body(
    gemm_out_ptr,  # (num_tokens, hc_mult3), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, hc_mult, hidden_size), bfloat16
    post_mix_ptr,  # (num_tokens, hc_mult), float32
    comb_mix_ptr,  # (num_tokens, hc_mult*hc_mult), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fully fused: sqrsum + RMS norm + sigmoid + Sinkhorn + weighted sum. One token per program."""
    pid_n = tl.program_id(0)
    if pid_n >= num_tokens:
        return

    # ══ Pass 1: compute sqrsum over all 4 heads ══
    sq = 0.0
    res_base = pid_n * res_stride_n
    for k in tl.static_range(4):
        head_base = res_base + k * res_stride_i
        for h_start in range(0, hidden_size, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < hidden_size
            v = tl.load(
                residual_ptr + head_base + h_offsets * res_stride_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            sq += tl.sum(v * v)

    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    go_base = pid_n * HC_MULT3
    m0 = tl.load(gemm_out_ptr + go_base + 0).to(tl.float32)
    m1 = tl.load(gemm_out_ptr + go_base + 1).to(tl.float32)
    m2 = tl.load(gemm_out_ptr + go_base + 2).to(tl.float32)
    m3 = tl.load(gemm_out_ptr + go_base + 3).to(tl.float32)
    m4 = tl.load(gemm_out_ptr + go_base + 4).to(tl.float32)
    m5 = tl.load(gemm_out_ptr + go_base + 5).to(tl.float32)
    m6 = tl.load(gemm_out_ptr + go_base + 6).to(tl.float32)
    m7 = tl.load(gemm_out_ptr + go_base + 7).to(tl.float32)
    m8 = tl.load(gemm_out_ptr + go_base + 8).to(tl.float32)
    m9 = tl.load(gemm_out_ptr + go_base + 9).to(tl.float32)
    m10 = tl.load(gemm_out_ptr + go_base + 10).to(tl.float32)
    m11 = tl.load(gemm_out_ptr + go_base + 11).to(tl.float32)
    m12 = tl.load(gemm_out_ptr + go_base + 12).to(tl.float32)
    m13 = tl.load(gemm_out_ptr + go_base + 13).to(tl.float32)
    m14 = tl.load(gemm_out_ptr + go_base + 14).to(tl.float32)
    m15 = tl.load(gemm_out_ptr + go_base + 15).to(tl.float32)
    m16 = tl.load(gemm_out_ptr + go_base + 16).to(tl.float32)
    m17 = tl.load(gemm_out_ptr + go_base + 17).to(tl.float32)
    m18 = tl.load(gemm_out_ptr + go_base + 18).to(tl.float32)
    m19 = tl.load(gemm_out_ptr + go_base + 19).to(tl.float32)
    m20 = tl.load(gemm_out_ptr + go_base + 20).to(tl.float32)
    m21 = tl.load(gemm_out_ptr + go_base + 21).to(tl.float32)
    m22 = tl.load(gemm_out_ptr + go_base + 22).to(tl.float32)
    m23 = tl.load(gemm_out_ptr + go_base + 23).to(tl.float32)

    _mhc_epilogue_token(
        m0,
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        m7,
        m8,
        m9,
        m10,
        m11,
        m12,
        m13,
        m14,
        m15,
        m16,
        m17,
        m18,
        m19,
        m20,
        m21,
        m22,
        m23,
        rms_inv,
        hc_scale_ptr,
        hc_base_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        layer_input_ptr,
        pid_n,
        res_stride_n,
        res_stride_i,
        res_stride_h,
        li_stride_n,
        li_stride_h,
        hidden_size,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        BLOCK_H=BLOCK_H,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=2),
    ],
    key=["hidden_size", "num_tokens_bucket"],
)
@triton.jit
def mhc_pre_mm_fuse_kernel(
    gemm_out_ptr,  # (num_tokens, hc_mult3), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, hc_mult, hidden_size), bfloat16
    post_mix_ptr,  # (num_tokens, hc_mult), float32
    comb_mix_ptr,  # (num_tokens, hc_mult*hc_mult), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    _mm_fuse_body(
        gemm_out_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        layer_input_ptr,
        num_tokens,
        num_tokens_bucket,
        res_stride_n,
        res_stride_i,
        res_stride_h,
        li_stride_n,
        li_stride_h,
        hidden_size,
        hc_hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        HC_MULT3,
        BLOCK_H,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    ],
    key=["hidden_size", "num_tokens_bucket", "HC"],
)
@triton.jit
def mhc_pre_generic_kernel(
    gemm_out_ptr,  # (num_tokens, hc_mult3), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, HC, hidden_size), bfloat16
    post_mix_ptr,  # (num_tokens, HC), float32
    comb_mix_ptr,  # (num_tokens, HC*HC), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= num_tokens:
        return

    res_base = pid_n * res_stride_n
    go_base = pid_n * (HC * 2 + HC * HC)
    comb_base = pid_n * (HC * HC)

    sq = 0.0
    for k in tl.static_range(HC):
        head_base = res_base + k * res_stride_i
        for h_start in range(0, hidden_size, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < hidden_size
            v = tl.load(
                residual_ptr + head_base + h_offsets * res_stride_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            sq += tl.sum(v * v)

    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    for i in tl.static_range(HC):
        post_i = (
            tl.sigmoid(
                tl.load(gemm_out_ptr + go_base + HC + i) * rms_inv * scale_1
                + tl.load(hc_base_ptr + HC + i)
            )
            * hc_post_mult_value
        )
        tl.store(post_mix_ptr + pid_n * HC + i, post_i)

    cb = 2 * HC
    for i in tl.static_range(HC):
        for j in tl.static_range(HC):
            idx = i * HC + j
            v = tl.load(
                gemm_out_ptr + go_base + cb + idx
            ) * rms_inv * scale_2 + tl.load(hc_base_ptr + cb + idx)
            tl.store(comb_mix_ptr + comb_base + idx, v)

    for i in tl.static_range(HC):
        row_max = tl.load(comb_mix_ptr + comb_base + i * HC + 0)
        for j in tl.static_range(1, HC):
            row_max = tl.maximum(
                row_max, tl.load(comb_mix_ptr + comb_base + i * HC + j)
            )

        row_sum = 0.0
        for j in tl.static_range(HC):
            e = tl.exp(tl.load(comb_mix_ptr + comb_base + i * HC + j) - row_max)
            tl.store(comb_mix_ptr + comb_base + i * HC + j, e)
            row_sum += e

        inv_row_sum = 1.0 / row_sum
        for j in tl.static_range(HC):
            v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
            tl.store(
                comb_mix_ptr + comb_base + i * HC + j, v * inv_row_sum + hc_sinkhorn_eps
            )

    for j in tl.static_range(HC):
        col_sum = 0.0
        for i in tl.static_range(HC):
            col_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
        inv_col_sum = 1.0 / (col_sum + hc_sinkhorn_eps)
        for i in tl.static_range(HC):
            v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
            tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_col_sum)

    for _ in tl.static_range(sinkhorn_repeat - 1):
        for i in tl.static_range(HC):
            row_sum = 0.0
            for j in tl.static_range(HC):
                row_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
            inv_row_sum = 1.0 / (row_sum + hc_sinkhorn_eps)
            for j in tl.static_range(HC):
                v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
                tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_row_sum)

        for j in tl.static_range(HC):
            col_sum = 0.0
            for i in tl.static_range(HC):
                col_sum += tl.load(comb_mix_ptr + comb_base + i * HC + j)
            inv_col_sum = 1.0 / (col_sum + hc_sinkhorn_eps)
            for i in tl.static_range(HC):
                v = tl.load(comb_mix_ptr + comb_base + i * HC + j)
                tl.store(comb_mix_ptr + comb_base + i * HC + j, v * inv_col_sum)

    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k in tl.static_range(HC):
            pre_k = (
                tl.sigmoid(
                    tl.load(gemm_out_ptr + go_base + k) * rms_inv * scale_0
                    + tl.load(hc_base_ptr + k)
                )
                + hc_pre_eps
            )
            rk = tl.load(
                residual_ptr + res_base + k * res_stride_i + h_offsets * res_stride_h,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            acc += pre_k * rk

        tl.store(
            layer_input_ptr + pid_n * li_stride_n + h_offsets * li_stride_h,
            acc.to(tl.bfloat16),
            mask=h_mask,
        )


@triton.jit
def _mhc_row_get(row, idx: tl.constexpr, WIDTH: tl.constexpr):
    """Element idx of a (WIDTH,) row as a scalar (integer subscripts are
    rejected by some backends, hence the one-hot reduction)."""
    return tl.sum(tl.where(tl.arange(0, WIDTH) == idx, row, 0.0))


@triton.jit
def _mhc_residual_mix(ptr, pid_n, j, num_tokens, N_SPLITS: tl.constexpr):

    acc = 0.0
    for i_split in range(0, N_SPLITS):
        acc += tl.load(
            ptr
            + i_split * (num_tokens * _MHC_PRE_MIX_PAD)
            + pid_n * _MHC_PRE_MIX_PAD
            + j
        )
    return acc


@triton.jit
def _mhc_cast_kernel(src_ptr, out_ptr, n0, n1, s0, s1, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = n0 * n1
    mask = i < total
    i1 = i % n1
    i0 = i // n1
    v = tl.load(src_ptr + i0 * s0 + i1 * s1, mask=mask, other=0.0).to(
        out_ptr.dtype.element_ty
    )
    tl.store(out_ptr + i, v, mask=mask)


@triton.jit
def _mhc_fill_kernel(out_ptr, N, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + i, VALUE, mask=i < N)


def _mhc_copy_as(x: torch.Tensor, dtype) -> torch.Tensor:
    """Contiguous 2D copy of ``x`` cast to ``dtype``, strides kept out of torch."""
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    n0, n1 = x.shape
    if x.numel():
        _mhc_cast_kernel[lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)](
            x, out, n0, n1, x.stride(0), x.stride(1), BLOCK=1024
        )
    return out


@triton.jit
def _mhc_copy_nd_kernel(
    src_ptr, out_ptr, n0, n1, n2, n3, s0, s1, s2, s3, BLOCK: tl.constexpr
):
    """Contiguous copy of a strided 3D/4D tensor; unused trailing dims are
    passed as size 1 / stride 0 so one kernel covers both ranks.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = n0 * n1 * n2 * n3
    mask = i < total
    i3 = i % n3
    rest = i // n3
    i2 = rest % n2
    rest = rest // n2
    i1 = rest % n1
    i0 = rest // n1
    v = tl.load(src_ptr + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3, mask=mask, other=0.0)
    tl.store(out_ptr + i, v, mask=mask)


def _mhc_copy_to_3d(x: torch.Tensor, hc_mult: int, hidden_size: int) -> torch.Tensor:
    """Contiguous (lead, hc_mult, hidden_size) copy of a strided x.

    The strides stay here rather than in a torch reshape, so a non-contiguous
    input never costs a torch copy kernel.
    """
    out = torch.empty(
        (x.numel() // (hc_mult * hidden_size), hc_mult, hidden_size),
        dtype=x.dtype,
        device=x.device,
    )
    if x.numel():
        dims = x.dim()
        pad = 4 - dims
        sizes = tuple(x.size(d) for d in range(dims)) + (1,) * pad
        strides = tuple(x.stride(d) for d in range(dims)) + (0,) * pad
        _mhc_copy_nd_kernel[lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)](
            x, out, *sizes, *strides, BLOCK=1024
        )
    return out


def _mhc_zero_(x: torch.Tensor) -> torch.Tensor:
    if x.numel():
        _mhc_fill_kernel[lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)](
            x, x.numel(), 0, BLOCK=1024
        )
    return x


@triton.jit
def _mhc_pack_kernel(
    src_ptr, out_ptr, K, NZ: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k = idx // W
    j = idx % W
    v = tl.load(src_ptr + j * K + k, mask=(idx < K * W) & (j < NZ), other=0.0)
    tl.store(out_ptr + idx, v, mask=idx < K * W)


def _mhc_pack_fn(fn: torch.Tensor) -> torch.Tensor:
    """fn (24, K) as (K, 32) bf16, rows 0..23 being fn^T and 24..31 zero."""
    assert fn.shape[0] <= _MHC_PRE_MIX_PAD.value, (
        f"the dot-based structures pack the mix row into {_MHC_PRE_MIX_PAD.value} "
        f"columns, but this weight has {fn.shape[0]} rows"
    )
    src = _mhc_copy_as(fn, torch.bfloat16)
    packed = torch.empty(
        fn.shape[1], _MHC_PRE_MIX_PAD, dtype=torch.bfloat16, device=fn.device
    )
    _mhc_pack_kernel[lambda meta: (triton.cdiv(packed.numel(), meta["BLOCK"]),)](
        src,
        packed,
        fn.shape[1],
        NZ=fn.shape[0],
        W=_MHC_PRE_MIX_PAD,
        BLOCK=1024,
    )
    return packed


def _mhc_pre_fn_packed_cached(fn):
    """Packed weights for the dot-based structures, cached per tensor."""
    if fn.requires_grad:
        return _mhc_pack_fn(fn)
    store = _mhc_fn_cache(fn)
    packed = store.get("packed")
    if packed is None:
        packed = _mhc_pack_fn(fn)
        store["packed"] = packed
    return packed


def _mhc_pre_reader_s4(num_tokens, block_m, block_k, hc_hidden_size, target_ctas=None):
    """Split-K count for the packed reader.

    Splitting buys occupancy but multiplies the partials the fuse gathers, so it
    depends on how far the shape already is from being bandwidth bound.
    """
    ctas_m = -(-num_tokens // block_m)

    def _capped(target, k_floor, cap):
        s = max(1, min(cap, int(round(target / max(ctas_m, 1)))))
        if k_floor:
            s = min(s, max(1, hc_hidden_size // k_floor))
        while s > 1 and (hc_hidden_size // s) % block_k != 0:
            s -= 1
        while s > 1 and hc_hidden_size % (s * block_k) != 0:
            s -= 1
        return s

    if target_ctas is not None:
        return _capped(target_ctas, 0, 8)

    if num_tokens <= 512:

        s = _capped(2048, 320, 16)
    else:
        s = max(_capped(180, 0, 8), _capped(480, 4096, 8))

    _bytes, _from_n, _upto_n, _cap = 0, 0, 0, 8
    if _VENDOR == "mthreads":
        _bytes, _upto_n, _cap = 1024, 1024, 20
    elif _VENDOR == "thead":
        _bytes, _from_n, _cap = 2048, 8192, 10
    if _bytes and num_tokens >= _from_n and (_upto_n == 0 or num_tokens <= _upto_n):
        s = max(s, min(-(-(hc_hidden_size * 2) // _bytes), _cap))
        while s > 1 and (
            (hc_hidden_size // s) % block_k or hc_hidden_size % (s * block_k)
        ):
            s -= 1
    return s


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["M", "stride_am"],
)
@triton.jit
def mhc_pre_packed_reader_kernel(
    A_ptr,  # (m, k) bfloat16
    BT_ptr,  # (k, 32) bfloat16, rows 0..23 = fn^T, 24..31 zero
    D_ptr,  # (n_splits, m, _MHC_PRE_MIX_PAD) float32, 32-wide rows
    S_ptr,  # (n_splits, m) float32
    M,
    stride_am,
    K_SPLIT_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The loads below carry no mask, so BLOCK_M has to divide num_tokens. The
    # reader gate only guarantees that for the configs up to 128.
    tl.static_assert(BLOCK_M <= 128)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(2)
    k0 = pid_k * K_SPLIT_LEN
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq2 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for koff in tl.range(0, K_SPLIT_LEN, BLOCK_K):
        offs_k = k0 + koff + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :])
        b = tl.load(BT_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
        acc += tl.dot(a, b)
        af = a.to(tl.float32)
        sq2 += af * af
    sq = tl.sum(sq2, axis=1)
    tl.store(
        D_ptr
        + pid_k * M * _MHC_PRE_MIX_PAD
        + offs_m[:, None] * _MHC_PRE_MIX_PAD
        + offs_n[None, :],
        acc,
    )
    tl.store(S_ptr + pid_k * M + offs_m, sq)


@triton.jit
def _mm_fuse_tiled_body(
    gemm_out_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
):
    """TOKEN_TILE tokens per program, so the serial Sinkhorn chain runs as
    [T]-vector math.
    """
    pid_t = tl.program_id(0)
    offs_t = pid_t * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
    tmask = offs_t < num_tokens
    res_row = offs_t * res_stride_n

    sq = tl.zeros((TOKEN_TILE,), dtype=tl.float32)
    for k in tl.static_range(4):
        for h_start in range(0, hidden_size, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            v = tl.load(
                residual_ptr
                + res_row[:, None]
                + k * res_stride_i
                + h_offsets[None, :] * res_stride_h,
                mask=tmask[:, None] & (h_offsets < hidden_size)[None, :],
                other=0.0,
            )
            vf = v.to(tl.float32)
            sq += tl.sum(vf * vf, axis=1)

    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    _mhc_epilogue_tile(
        gemm_out_ptr,
        rms_inv,
        hc_scale_ptr,
        hc_base_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        layer_input_ptr,
        offs_t,
        tmask,
        res_row,
        res_stride_i,
        res_stride_h,
        li_stride_n,
        li_stride_h,
        hidden_size,
        MIX_STRIDE=HC_MULT3,
        MIX_MASKED=True,
        H_STRIDE_ONE=False,
        H_EXACT=False,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        BLOCK_H=BLOCK_H,
    )


@triton.autotune(
    configs=[
        triton.Config({"TOKEN_TILE": t, "BLOCK_H": bh}, num_warps=8, num_stages=1)
        for t in (2, 4, 8)
        for bh in (256, 512, 1024)
    ],
    key=["hidden_size", "num_tokens_bucket"],
)
@triton.jit
def mhc_pre_mm_fuse_tiled_kernel(
    gemm_out_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
):
    _mm_fuse_tiled_body(
        gemm_out_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        layer_input_ptr,
        num_tokens,
        num_tokens_bucket,
        res_stride_n,
        res_stride_i,
        res_stride_h,
        li_stride_n,
        li_stride_h,
        hidden_size,
        hc_hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        HC_MULT3,
        BLOCK_H,
        TOKEN_TILE,
    )


try:
    from triton.language.extra.libdevice import fast_dividef as _mhc_fast_div
except Exception:  # backend ships no libdevice
    _mhc_fast_div = None


@triton.jit
def _mhc_recip_exact(x):
    return 1.0 / x


@triton.jit
def _mhc_recip_fast(x):
    return _mhc_fast_div(1.0, x)


_mhc_recip = _mhc_recip_exact


@triton.jit
def _sinkhorn_iters(
    cm_00,
    cm_01,
    cm_02,
    cm_03,
    cm_10,
    cm_11,
    cm_12,
    cm_13,
    cm_20,
    cm_21,
    cm_22,
    cm_23,
    cm_30,
    cm_31,
    cm_32,
    cm_33,
    hc_sinkhorn_eps: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
):
    """Sinkhorn fixpoint chain shared by every hc4 fused kernel."""
    rm = tl.maximum(tl.maximum(cm_00, cm_01), tl.maximum(cm_02, cm_03))
    cm_00 = tl.exp(cm_00 - rm)
    cm_01 = tl.exp(cm_01 - rm)
    cm_02 = tl.exp(cm_02 - rm)
    cm_03 = tl.exp(cm_03 - rm)
    rs = cm_00 + cm_01 + cm_02 + cm_03
    inv_rs = 1.0 / rs
    cm_00 = cm_00 * inv_rs + hc_sinkhorn_eps
    cm_01 = cm_01 * inv_rs + hc_sinkhorn_eps
    cm_02 = cm_02 * inv_rs + hc_sinkhorn_eps
    cm_03 = cm_03 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_10, cm_11), tl.maximum(cm_12, cm_13))
    cm_10 = tl.exp(cm_10 - rm)
    cm_11 = tl.exp(cm_11 - rm)
    cm_12 = tl.exp(cm_12 - rm)
    cm_13 = tl.exp(cm_13 - rm)
    rs = cm_10 + cm_11 + cm_12 + cm_13
    inv_rs = 1.0 / rs
    cm_10 = cm_10 * inv_rs + hc_sinkhorn_eps
    cm_11 = cm_11 * inv_rs + hc_sinkhorn_eps
    cm_12 = cm_12 * inv_rs + hc_sinkhorn_eps
    cm_13 = cm_13 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_20, cm_21), tl.maximum(cm_22, cm_23))
    cm_20 = tl.exp(cm_20 - rm)
    cm_21 = tl.exp(cm_21 - rm)
    cm_22 = tl.exp(cm_22 - rm)
    cm_23 = tl.exp(cm_23 - rm)
    rs = cm_20 + cm_21 + cm_22 + cm_23
    inv_rs = 1.0 / rs
    cm_20 = cm_20 * inv_rs + hc_sinkhorn_eps
    cm_21 = cm_21 * inv_rs + hc_sinkhorn_eps
    cm_22 = cm_22 * inv_rs + hc_sinkhorn_eps
    cm_23 = cm_23 * inv_rs + hc_sinkhorn_eps

    rm = tl.maximum(tl.maximum(cm_30, cm_31), tl.maximum(cm_32, cm_33))
    cm_30 = tl.exp(cm_30 - rm)
    cm_31 = tl.exp(cm_31 - rm)
    cm_32 = tl.exp(cm_32 - rm)
    cm_33 = tl.exp(cm_33 - rm)
    rs = cm_30 + cm_31 + cm_32 + cm_33
    inv_rs = 1.0 / rs
    cm_30 = cm_30 * inv_rs + hc_sinkhorn_eps
    cm_31 = cm_31 * inv_rs + hc_sinkhorn_eps
    cm_32 = cm_32 * inv_rs + hc_sinkhorn_eps
    cm_33 = cm_33 * inv_rs + hc_sinkhorn_eps

    cs0 = cm_00 + cm_10 + cm_20 + cm_30
    cs1 = cm_01 + cm_11 + cm_21 + cm_31
    cs2 = cm_02 + cm_12 + cm_22 + cm_32
    cs3 = cm_03 + cm_13 + cm_23 + cm_33
    inv_cs0 = _mhc_recip(cs0 + hc_sinkhorn_eps)
    inv_cs1 = _mhc_recip(cs1 + hc_sinkhorn_eps)
    inv_cs2 = _mhc_recip(cs2 + hc_sinkhorn_eps)
    inv_cs3 = _mhc_recip(cs3 + hc_sinkhorn_eps)
    cm_00 *= inv_cs0
    cm_10 *= inv_cs0
    cm_20 *= inv_cs0
    cm_30 *= inv_cs0
    cm_01 *= inv_cs1
    cm_11 *= inv_cs1
    cm_21 *= inv_cs1
    cm_31 *= inv_cs1
    cm_02 *= inv_cs2
    cm_12 *= inv_cs2
    cm_22 *= inv_cs2
    cm_32 *= inv_cs2
    cm_03 *= inv_cs3
    cm_13 *= inv_cs3
    cm_23 *= inv_cs3
    cm_33 *= inv_cs3

    for _ in tl.static_range(sinkhorn_repeat - 1):
        rs0 = cm_00 + cm_01 + cm_02 + cm_03
        rs1 = cm_10 + cm_11 + cm_12 + cm_13
        rs2 = cm_20 + cm_21 + cm_22 + cm_23
        rs3 = cm_30 + cm_31 + cm_32 + cm_33
        inv_rs0 = _mhc_recip(rs0 + hc_sinkhorn_eps)
        inv_rs1 = _mhc_recip(rs1 + hc_sinkhorn_eps)
        inv_rs2 = _mhc_recip(rs2 + hc_sinkhorn_eps)
        inv_rs3 = _mhc_recip(rs3 + hc_sinkhorn_eps)
        cm_00 *= inv_rs0
        cm_01 *= inv_rs0
        cm_02 *= inv_rs0
        cm_03 *= inv_rs0
        cm_10 *= inv_rs1
        cm_11 *= inv_rs1
        cm_12 *= inv_rs1
        cm_13 *= inv_rs1
        cm_20 *= inv_rs2
        cm_21 *= inv_rs2
        cm_22 *= inv_rs2
        cm_23 *= inv_rs2
        cm_30 *= inv_rs3
        cm_31 *= inv_rs3
        cm_32 *= inv_rs3
        cm_33 *= inv_rs3
        cs0 = cm_00 + cm_10 + cm_20 + cm_30
        cs1 = cm_01 + cm_11 + cm_21 + cm_31
        cs2 = cm_02 + cm_12 + cm_22 + cm_32
        cs3 = cm_03 + cm_13 + cm_23 + cm_33
        inv_cs0 = _mhc_recip(cs0 + hc_sinkhorn_eps)
        inv_cs1 = _mhc_recip(cs1 + hc_sinkhorn_eps)
        inv_cs2 = _mhc_recip(cs2 + hc_sinkhorn_eps)
        inv_cs3 = _mhc_recip(cs3 + hc_sinkhorn_eps)
        cm_00 *= inv_cs0
        cm_01 *= inv_cs1
        cm_02 *= inv_cs2
        cm_03 *= inv_cs3
        cm_10 *= inv_cs0
        cm_11 *= inv_cs1
        cm_12 *= inv_cs2
        cm_13 *= inv_cs3
        cm_20 *= inv_cs0
        cm_21 *= inv_cs1
        cm_22 *= inv_cs2
        cm_23 *= inv_cs3
        cm_30 *= inv_cs0
        cm_31 *= inv_cs1
        cm_32 *= inv_cs2
        cm_33 *= inv_cs3

    return (
        cm_00,
        cm_01,
        cm_02,
        cm_03,
        cm_10,
        cm_11,
        cm_12,
        cm_13,
        cm_20,
        cm_21,
        cm_22,
        cm_23,
        cm_30,
        cm_31,
        cm_32,
        cm_33,
    )


@triton.jit
def _sinkhorn_fuse_body(
    gemm_mul_ptr,  # (n_splits, num_tokens, hc_mult3), float32
    gemm_sq_ptr,  # (n_splits, num_tokens), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, hc_mult, hidden_size), bfloat16
    post_mix_ptr,  # (num_tokens, hc_mult), float32
    comb_mix_ptr,  # (num_tokens, hc_mult*hc_mult), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    N_SPLITS: tl.constexpr,
    MIX_VEC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fully fused: sqrsum + RMS norm + sigmoid + Sinkhorn + weighted sum. One token per program."""
    pid_n = tl.program_id(0)
    if pid_n >= num_tokens:
        return

    # ══ sqrsum + mixes arrive from the reader above ══

    sq = 0.0
    for i_split in range(0, N_SPLITS):
        sq += tl.load(gemm_sq_ptr + i_split * num_tokens + pid_n)

    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    if MIX_VEC:

        mix_offs = tl.arange(0, 32)
        mix_row = tl.zeros((32,), dtype=tl.float32)
        for i_split in range(0, N_SPLITS):
            mix_row += tl.load(
                gemm_mul_ptr
                + i_split * (num_tokens * _MHC_PRE_MIX_PAD)
                + pid_n * _MHC_PRE_MIX_PAD
                + mix_offs,
                mask=mix_offs < HC_MULT3,
                other=0.0,
            )
        m0 = _mhc_row_get(mix_row, 0, 32)
        m1 = _mhc_row_get(mix_row, 1, 32)
        m2 = _mhc_row_get(mix_row, 2, 32)
        m3 = _mhc_row_get(mix_row, 3, 32)
        m4 = _mhc_row_get(mix_row, 4, 32)
        m5 = _mhc_row_get(mix_row, 5, 32)
        m6 = _mhc_row_get(mix_row, 6, 32)
        m7 = _mhc_row_get(mix_row, 7, 32)
        m8 = _mhc_row_get(mix_row, 8, 32)
        m9 = _mhc_row_get(mix_row, 9, 32)
        m10 = _mhc_row_get(mix_row, 10, 32)
        m11 = _mhc_row_get(mix_row, 11, 32)
        m12 = _mhc_row_get(mix_row, 12, 32)
        m13 = _mhc_row_get(mix_row, 13, 32)
        m14 = _mhc_row_get(mix_row, 14, 32)
        m15 = _mhc_row_get(mix_row, 15, 32)
        m16 = _mhc_row_get(mix_row, 16, 32)
        m17 = _mhc_row_get(mix_row, 17, 32)
        m18 = _mhc_row_get(mix_row, 18, 32)
        m19 = _mhc_row_get(mix_row, 19, 32)
        m20 = _mhc_row_get(mix_row, 20, 32)
        m21 = _mhc_row_get(mix_row, 21, 32)
        m22 = _mhc_row_get(mix_row, 22, 32)
        m23 = _mhc_row_get(mix_row, 23, 32)
    else:
        m0 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 0, num_tokens, N_SPLITS)
        m1 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 1, num_tokens, N_SPLITS)
        m2 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 2, num_tokens, N_SPLITS)
        m3 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 3, num_tokens, N_SPLITS)
        m4 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 4, num_tokens, N_SPLITS)
        m5 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 5, num_tokens, N_SPLITS)
        m6 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 6, num_tokens, N_SPLITS)
        m7 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 7, num_tokens, N_SPLITS)
        m8 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 8, num_tokens, N_SPLITS)
        m9 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 9, num_tokens, N_SPLITS)
        m10 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 10, num_tokens, N_SPLITS)
        m11 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 11, num_tokens, N_SPLITS)
        m12 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 12, num_tokens, N_SPLITS)
        m13 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 13, num_tokens, N_SPLITS)
        m14 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 14, num_tokens, N_SPLITS)
        m15 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 15, num_tokens, N_SPLITS)
        m16 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 16, num_tokens, N_SPLITS)
        m17 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 17, num_tokens, N_SPLITS)
        m18 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 18, num_tokens, N_SPLITS)
        m19 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 19, num_tokens, N_SPLITS)
        m20 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 20, num_tokens, N_SPLITS)
        m21 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 21, num_tokens, N_SPLITS)
        m22 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 22, num_tokens, N_SPLITS)
        m23 = _mhc_residual_mix(gemm_mul_ptr, pid_n, 23, num_tokens, N_SPLITS)

    _mhc_epilogue_token(
        m0,
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        m7,
        m8,
        m9,
        m10,
        m11,
        m12,
        m13,
        m14,
        m15,
        m16,
        m17,
        m18,
        m19,
        m20,
        m21,
        m22,
        m23,
        rms_inv,
        hc_scale_ptr,
        hc_base_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        layer_input_ptr,
        pid_n,
        res_stride_n,
        res_stride_i,
        res_stride_h,
        li_stride_n,
        li_stride_h,
        hidden_size,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        BLOCK_H=BLOCK_H,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256, "MIX_VEC": True}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256, "MIX_VEC": False}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256, "MIX_VEC": True}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 256, "MIX_VEC": False}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512, "MIX_VEC": True}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512, "MIX_VEC": False}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512, "MIX_VEC": True}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512, "MIX_VEC": False}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024, "MIX_VEC": True}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024, "MIX_VEC": False}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024, "MIX_VEC": True}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024, "MIX_VEC": False}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512, "MIX_VEC": True}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_H": 1024, "MIX_VEC": True}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_H": 2048, "MIX_VEC": True}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_H": 512, "MIX_VEC": True}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_H": 1024, "MIX_VEC": True}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_H": 2048, "MIX_VEC": True}, num_warps=1, num_stages=1),
    ],
    key=["hidden_size", "num_tokens_bucket"],
)
@triton.jit
def mhc_pre_sinkhorn_fuse_kernel(
    gemm_mul_ptr,  # (n_splits, num_tokens, hc_mult3), float32
    gemm_sq_ptr,  # (n_splits, num_tokens), float32
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, hc_mult, hidden_size), bfloat16
    post_mix_ptr,  # (num_tokens, hc_mult), float32
    comb_mix_ptr,  # (num_tokens, hc_mult*hc_mult), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    N_SPLITS: tl.constexpr,
    MIX_VEC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    _sinkhorn_fuse_body(
        gemm_mul_ptr,
        gemm_sq_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        layer_input_ptr,
        num_tokens,
        num_tokens_bucket,
        res_stride_n,
        res_stride_i,
        res_stride_h,
        li_stride_n,
        li_stride_h,
        hidden_size,
        hc_hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        HC_MULT3,
        N_SPLITS,
        MIX_VEC,
        BLOCK_H,
    )


def _mhc_reader_divides(num_tokens, hidden_size):
    """Whether the packed reader can split K evenly: the condition its launch
    asserts, evaluated before committing to it.
    """
    hc_hidden = 4 * hidden_size
    bm21 = 64 if num_tokens >= 256 else 32
    s4 = _mhc_pre_reader_s4(num_tokens, bm21, 128, hc_hidden)
    return hc_hidden % (s4 * 128) == 0


_VARIANT_MM_FUSE = 0
_VARIANT_SPLIT_PASS = 7
_VARIANT_MM_FUSE_TILED = 10
_VARIANT_PACKED_READER = 21
_SPLIT_MIN_K = 1280
_MHC_PRE_VARIANT_CACHE: dict[tuple[int, int, int], int] = {}


def _mhc_pre_default_variant(num_tokens, hidden_size, hc_hidden_size):
    """Structure to run, as a threshold decision on the inputs."""
    key = (num_tokens, hidden_size, hc_hidden_size)
    cached = _MHC_PRE_VARIANT_CACHE.get(key)
    if cached is not None:
        return cached

    if _HYGON is not None and hidden_size % 256 == 0:
        if hc_hidden_size <= 16384:
            _MHC_PRE_VARIANT_CACHE[key] = _VARIANT_SPLIT_PASS
            return _VARIANT_SPLIT_PASS

    reader_ok = (
        num_tokens % 128 == 0
        and hidden_size % 32 == 0
        and _mhc_reader_divides(num_tokens, hidden_size)
    )

    _reader_from = 2048 if _VENDOR == "hygon" else 0
    if reader_ok and num_tokens >= _reader_from:
        variant = _VARIANT_PACKED_READER
    elif num_tokens > 512 and num_tokens % 2 == 0:
        variant = _VARIANT_MM_FUSE_TILED
    else:
        variant = _VARIANT_MM_FUSE
    _MHC_PRE_VARIANT_CACHE[key] = variant
    return variant


_MHC_PRE_SETUP_DONE = False


def _mhc_pre_setup():
    """Resolve the vendor once, then do what that vendor needs done once."""
    global _VENDOR, _MHC_PRE_SETUP_DONE, _mhc_recip
    if _MHC_PRE_SETUP_DONE:
        return
    try:
        import flaggems_vllm

        _VENDOR = flaggems_vllm.vendor_name
    except Exception:  # noqa: BLE001 - stay generic, retry on the next call
        return
    if _VENDOR == "mthreads":
        os.environ.setdefault("TRITON_MUSA_ENABLE_LLC_OPT", "1")
    elif _VENDOR == "thead":
        if _mhc_fast_div is not None:
            _mhc_recip = _mhc_recip_fast
    elif _VENDOR == "hygon":
        global _HYGON
        try:
            _HYGON = importlib.import_module(
                "flaggems_vllm.runtime.backend._hygon.ops.mhc_pre"
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the generic structures
            _HYGON = None
            _MHC_PRE_SETUP_DONE = True
            warnings.warn(
                f"mhc_pre: hygon fast path unavailable ({exc!r}); "
                "running the generic structures without fast launch",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        _HYGON.install_fastlaunch(globals(), _HYGON.KERNELS)
    _MHC_PRE_SETUP_DONE = True


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized mHC pre block.

    - hc_mult == 4: specialized fused Triton kernel
    - hc_mult != 4: generic Triton kernel aligned to reference math
    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32 and hc_base.dtype == torch.float32
    assert hc_scale.is_contiguous() and hc_base.is_contiguous()
    assert hc_scale.device == residual.device and hc_base.device == residual.device
    assert hc_scale.numel() == 3, f"hc_scale must hold 3 values, got {hc_scale.numel()}"
    if n_splits != 1:
        warnings.warn(
            f"mhc_pre: n_splits={n_splits} is ignored; the split count is chosen "
            "from the shape",
            RuntimeWarning,
            stacklevel=2,
        )
    _mhc_pre_setup()

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_base.numel() == hc_mult3, f"hc_base must hold {hc_mult3} values"

    outer_shape = residual.shape[:-2]
    if residual.dim() == 3 and residual.is_contiguous():

        residual_flat = residual
    else:
        residual_flat = _mhc_copy_to_3d(residual, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    device = residual.device
    if num_tokens == 0:

        return (
            torch.empty((*outer_shape, hc_mult, 1), dtype=torch.float32, device=device),
            torch.empty(
                (*outer_shape, hc_mult, hc_mult), dtype=torch.float32, device=device
            ),
            torch.empty(
                (*outer_shape, hidden_size), dtype=torch.bfloat16, device=device
            ),
        )
    if num_tokens <= 512:
        num_tokens_bucket = 1
    elif num_tokens <= 1024:
        num_tokens_bucket = 2
    elif num_tokens <= 2048:
        num_tokens_bucket = 3
    elif num_tokens <= 4096:
        num_tokens_bucket = 4
    else:
        num_tokens_bucket = 5

    x_flat = residual_flat.view(num_tokens, hc_hidden_size)
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    def _dispatch(variant):
        if variant == _VARIANT_PACKED_READER:

            bm21 = 64 if num_tokens >= 256 else 32
            s4 = _mhc_pre_reader_s4(num_tokens, bm21, 128, hc_hidden_size)
            if num_tokens % 128:
                raise ValueError(
                    f"packed reader requires n % 128 == 0, got {num_tokens}"
                )
            if hc_hidden_size % (s4 * 128):
                raise ValueError(
                    f"packed reader requires 4H % (s4*128) == 0, got {hc_hidden_size}, s4={s4}"
                )

            d_p = torch.empty(s4, num_tokens, 32, dtype=torch.float32, device=device)
            s_p = torch.empty(s4, num_tokens, dtype=torch.float32, device=device)
            _bt = _mhc_pre_fn_packed_cached(fn)
            mhc_pre_packed_reader_kernel[
                lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_M"]), 1, s4)
            ](
                x_flat,
                _bt,
                d_p,
                s_p,
                num_tokens,
                hc_hidden_size,
                K_SPLIT_LEN=hc_hidden_size // s4,
                BLOCK_N=32,
            )
            mhc_pre_sinkhorn_fuse_kernel[(num_tokens,)](
                d_p,
                s_p,
                hc_scale,
                hc_base,
                residual_flat,
                post_mix,
                comb_mix,
                layer_input,
                num_tokens,
                num_tokens_bucket,
                residual_flat.stride(0),
                residual_flat.stride(1),
                residual_flat.stride(2),
                layer_input.stride(0),
                layer_input.stride(1),
                hidden_size,
                hc_hidden_size,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                HC_MULT3=hc_mult3,
                N_SPLITS=s4,
            )
        elif variant == _VARIANT_SPLIT_PASS:

            if hc_mult != 4 or hidden_size % 256:
                raise ValueError(
                    f"split-pass requires hc4 and H%256==0, got {hc_mult}, {hidden_size}"
                )
            if residual_flat.stride(2) != 1 or residual_flat.stride(1) != hidden_size:
                raise ValueError("split-pass requires contiguous residual_flat")
            tile7 = 32
            bh7 = 256
            ctas_m = triton.cdiv(num_tokens, tile7)
            s7 = 1
            while (
                s7 < 8
                and s7 * ctas_m < 128
                and hc_hidden_size // (2 * s7) >= _SPLIT_MIN_K
            ):
                s7 *= 2
            if hc_hidden_size % (s7 * 128):
                raise ValueError(
                    f"split-pass requires 4H % (s7*128) == 0, "
                    f"got {hc_hidden_size}, s7={s7}"
                )
            part7, part_sq7, scratch7, cnt7 = _HYGON.split_buffers(
                num_tokens, hidden_size, s7, tile7, residual_flat.device
            )
            _HYGON.mhc_pre_split_pass_kernel[(ctas_m, s7)](
                _mhc_pre_fn_packed_cached(fn),
                hc_scale,
                hc_base,
                residual_flat,
                post_mix,
                comb_mix,
                layer_input,
                part7,
                part_sq7,
                scratch7,
                cnt7,
                num_tokens,
                num_tokens_bucket,
                residual_flat.stride(0),
                residual_flat.stride(1),
                layer_input.stride(0),
                layer_input.stride(1),
                hidden_size,
                hc_hidden_size,
                K_LEN=hc_hidden_size,
                N_SPLITS=s7,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                HC_MULT3=hc_mult3,
                TOKEN_TILE=tile7,
                BLOCK_H=bh7,
            )
        elif variant == _VARIANT_MM_FUSE_TILED:

            gemm_out = _mhc_project(
                x_flat, fn, num_tokens, hc_hidden_size, hc_mult3, device
            )
            mhc_pre_mm_fuse_tiled_kernel[
                lambda meta: (triton.cdiv(num_tokens, meta["TOKEN_TILE"]),)
            ](
                gemm_out,
                hc_scale,
                hc_base,
                residual_flat,
                post_mix,
                comb_mix,
                layer_input,
                num_tokens,
                num_tokens_bucket,
                residual_flat.stride(0),
                residual_flat.stride(1),
                residual_flat.stride(2),
                layer_input.stride(0),
                layer_input.stride(1),
                hidden_size,
                hc_hidden_size,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                HC_MULT3=hc_mult3,
            )
        else:

            gemm_out = _mhc_project(
                x_flat, fn, num_tokens, hc_hidden_size, hc_mult3, device
            )
            mhc_pre_mm_fuse_kernel[(num_tokens,)](
                gemm_out,
                hc_scale,
                hc_base,
                residual_flat,
                post_mix,
                comb_mix,
                layer_input,
                num_tokens,
                num_tokens_bucket,
                residual_flat.stride(0),
                residual_flat.stride(1),
                residual_flat.stride(2),
                layer_input.stride(0),
                layer_input.stride(1),
                hidden_size,
                hc_hidden_size,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                HC_MULT3=hc_mult3,
            )

    if hc_mult != 4:
        gemm_out = _mhc_project(
            x_flat, fn, num_tokens, hc_hidden_size, hc_mult3, device
        )
        mhc_pre_generic_kernel[(num_tokens,)](
            gemm_out,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            num_tokens,
            num_tokens_bucket,
            residual_flat.stride(0),
            residual_flat.stride(1),
            residual_flat.stride(2),
            layer_input.stride(0),
            layer_input.stride(1),
            hidden_size,
            hc_hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            HC=hc_mult,
        )

    else:
        _dispatch(_mhc_pre_default_variant(num_tokens, hidden_size, hc_hidden_size))

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference."""
    hc_mult = residual.shape[-2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )
    hc_scale_expanded = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * hc_scale_expanded + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    res_mix = sinkhorn_normalize_ref(
        res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
    )
    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    return post_mix, res_mix, layer_input

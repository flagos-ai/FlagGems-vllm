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

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_attn_varlen_func_w8a8_fp8 import (
    _flash_attn_varlen_func_w8a8_fp8,
    _load_fp8_block_descales,
    apply_alibi,
    apply_mask,
    apply_softcap,
    fwd_params,
    mha_varlan_fwd,
    softmax_rescale,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils.triton_version_utils import has_triton_tle_attrs

HAS_TLE = has_triton_tle_attrs(
    ("gpu.alloc", "gpu.copy", "gpu.local_ptr", "gpu.wgmma", "gpu.wgmma_wait"),
    3,
    6,
    0,
)
if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None


_FWD_PARAM_INDEX = {name: i for i, name in enumerate(fwd_params.__slots__)}


@triton.jit
def _fp8_pv_dot(
    P,
    V,
    acc,
    v_descale,
    fp8_p_max: tl.constexpr,
    fp8_dtype: tl.constexpr,
    precise_p: tl.constexpr = False,
):
    if fp8_p_max == 448.0:
        # The scaled row sum cancels the factor of 256 during normalization.
        P_scaled = P
        p_descale = 1.0
        p_dtype: tl.constexpr = fp8_dtype
    else:
        P_scaled = P * 256.0
        p_descale = 1.0 / 256.0
        # MUSA lowers same-format E5M2 products to native matrix instructions.
        # Compensate its shorter mantissa with the residual product below.
        p_dtype: tl.constexpr = tl.float8e5
    P_fp8 = P_scaled.to(p_dtype)
    pv = tl.dot(P_fp8, V, out_dtype=tl.float32)
    # Long-KV loops with non-unit block descales also need residual compensation.
    P_residual = ((P_scaled - P_fp8.to(tl.float32)) * 32.0).to(p_dtype)
    pv += tl.dot(P_residual, V, out_dtype=tl.float32) * (1.0 / 32.0)
    return acc + pv * (p_descale * v_descale)


@libentry()
@triton.jit
def _pad_fp8_qkv_bytes(
    Q,
    K,
    V,
    PQ,
    PK,
    PV,
    Q_ROWS: tl.constexpr,
    K_ROWS: tl.constexpr,
    V_ROWS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    V_HEADS: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    D: tl.constexpr,
    PADDED_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    which = tl.program_id(1)
    col = x % PADDED_D
    if which == 0:
        valid = x < Q_ROWS * Q_HEADS * PADDED_D
        row = x // (Q_HEADS * PADDED_D)
        head = x // PADDED_D % Q_HEADS
        value = tl.load(
            Q.to(tl.pointer_type(tl.uint8)) + row * QS0 + head * QS1 + col,
            valid & (col < D),
            other=0,
        )
        tl.store(PQ.to(tl.pointer_type(tl.uint8)) + x, value, valid)
    elif which == 1:
        valid = x < K_ROWS * K_HEADS * PADDED_D
        row = x // (K_HEADS * PADDED_D)
        head = x // PADDED_D % K_HEADS
        value = tl.load(
            K.to(tl.pointer_type(tl.uint8)) + row * KS0 + head * KS1 + col,
            valid & (col < D),
            other=0,
        )
        tl.store(PK.to(tl.pointer_type(tl.uint8)) + x, value, valid)
    else:
        valid = x < V_ROWS * V_HEADS * PADDED_D
        row = x // (V_HEADS * PADDED_D)
        head = x // PADDED_D % V_HEADS
        value = tl.load(
            V.to(tl.pointer_type(tl.uint8)) + row * VS0 + head * VS1 + col,
            valid & (col < D),
            other=0,
        )
        tl.store(PV.to(tl.pointer_type(tl.uint8)) + x, value, valid)


def _pad_fp8_qkv(q, k, v):
    """Preserve validated packed FP8 tensors with zero-filled 16-byte head tails."""
    d = q.shape[-1]
    padded_d = triton.cdiv(d, 16) * 16
    sources = (q, k, v)
    backing = tuple(
        torch.empty(
            (source.shape[0], source.shape[1], padded_d),
            device=source.device,
            dtype=source.dtype,
        )
        for source in sources
    )
    # Allocate every physical tail, including the final head of the final row.
    # These views retain the backing storage while exposing the original shape.
    views = tuple(
        target.as_strided(source.shape, (source.shape[1] * padded_d, padded_d, 1))
        for source, target in zip(sources, backing)
    )
    total = max(target.numel() for target in backing)
    if total:
        _pad_fp8_qkv_bytes[(triton.cdiv(total, 256), 3)](
            q,
            k,
            v,
            *backing,
            q.shape[0],
            k.shape[0],
            v.shape[0],
            q.shape[1],
            k.shape[1],
            v.shape[1],
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            d,
            padded_d,
            256,
            num_warps=4,
            num_stages=1,
        )
    return views


@triton.jit
def _quantize_probability(P):
    P_fp8 = P.to(tl.float8e4nv)
    P_residual = ((P - P_fp8.to(tl.float32)) * 32.0).to(tl.float8e4nv)
    return (P_fp8, P_residual)


@libentry()
@triton.jit(
    do_not_specialize=[
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "b",
        "bk",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "total_q",
    ]
)
def flash_varlen_fwd_kernel_tle(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q: tl.constexpr,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    b,
    bk,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    use_fa3_fp8_scales: tl.constexpr,
    fp8_p_max: tl.constexpr,
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    k_page_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_D: tl.constexpr,
    PRECISE_SHORT_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    PV_DOT: tl.constexpr = None,
    PADDED_D: tl.constexpr = 0,
):
    tl.static_assert(
        PADDED_D == 0 or (PADDED_D >= d and PADDED_D % 16 == 0 and PADDED_D <= BLOCK_K)
    )
    LOAD_D: tl.constexpr = PADDED_D if PADDED_D else d
    tl.static_assert(is_cu_seqlens_q)
    tl.static_assert(not is_seqused_k)
    tl.static_assert(not is_paged)
    tl.static_assert(is_cu_seqlens_k)
    tl.static_assert(not seqlenq_ngroups_swapped)
    tl.static_assert(not is_dropout)
    tl.static_assert(fp8_p_max == 448.0)
    tl.static_assert(q_ptr.dtype.element_ty == tl.float8e4nv)
    tl.static_assert(k_ptr.dtype.element_ty == tl.float8e4nv)
    tl.static_assert(v_ptr.dtype.element_ty == tl.float8e4nv)
    tl.static_assert(BLOCK_M >= 16 and BLOCK_M % 16 == 0)
    tl.static_assert(BLOCK_N == 32 or BLOCK_N == 64 or BLOCK_N == 128)
    tl.static_assert(
        BLOCK_K == 32 or BLOCK_K == 64 or BLOCK_K == 128 or (BLOCK_K == 256)
    )
    tl.static_assert(
        BLOCK_D == 32 or BLOCK_D == 64 or BLOCK_D == 128 or (BLOCK_D == 256)
    )
    tl.static_assert(128 % BLOCK_M == 0 and 128 % BLOCK_N == 0)
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    hd = tl.program_id(2)
    hid = hd % h
    d_split = hd // h
    d_start = d_split * BLOCK_D
    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
    q_len = q_eos - q_bos
    q_offset = q_bos * q_row_stride
    o_offset = q_bos * o_row_stride
    lse_offset = q_bos * 1
    k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
    k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
    k_len_cache = k_eos - k_bos
    k_len = k_len_cache
    if (k_len <= 128) != PRECISE_SHORT_K:
        return
    if m_block * BLOCK_M >= q_len:
        return
    is_even_mn: tl.constexpr = False
    if is_local:
        n_block_min = max(
            0, (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N
        )
    else:
        n_block_min = 0
    n_block_max = tl.cdiv(k_len, BLOCK_N)
    if is_causal or is_local:
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + k_len - q_len + window_size_right, BLOCK_N
            ),
        )
    q_row_offset = hid * q_head_stride
    kv_hid = hid // h_hk_ratio
    k_row_offset = kv_hid * k_head_stride
    k_ptr_base = k_ptr + k_row_offset
    v_ptr_base = v_ptr + k_row_offset
    # Keep shared-memory copies and SQMMA in this scope: the TLE layout pass
    # does not propagate shared-memory type changes across copy helper calls.
    sQ = tle.gpu.alloc(
        (BLOCK_M, BLOCK_K), tl.float8e4nv, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    sK = tle.gpu.alloc(
        (BLOCK_N, BLOCK_K), tl.float8e4nv, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    sV = tle.gpu.alloc(
        (BLOCK_N, BLOCK_D), tl.float8e4nv, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    sP = tle.gpu.alloc(
        (BLOCK_M, BLOCK_N), tl.float8e4nv, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    sPr = tle.gpu.alloc(
        (BLOCK_M, BLOCK_N), tl.float8e4nv, scope=tle.gpu.smem, nv_mma_shared_layout=True
    )
    q_rows = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q_cols = tl.arange(0, BLOCK_K)
    # Masked pointer copies write zero to every masked shared-memory element.
    tle.gpu.copy(
        q_ptr
        + q_offset
        + q_row_offset
        + q_rows[:, None] * q_row_stride
        + q_cols[None, :],
        sQ,
        (BLOCK_M, BLOCK_K),
        mask=(q_rows[:, None] < q_len) & (q_cols[None, :] < LOAD_D),
    )
    acc_ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)
    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0
    if not is_causal and (not is_local):
        n_masking_steps = 1
    else:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N) + 1
    n_masking_steps = min(n_block_max - n_block_min, n_masking_steps)
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_block = n_block_max - 1
    # Preserve the generic kernel's reverse traversal and separate boundary
    # iterations so online softmax follows the same accumulation order.
    for step in tl.range(0, n_masking_steps):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        start_n = n_block * BLOCK_N
        k_ptr_seq = k_ptr_base + k_bos * k_row_stride
        v_ptr_seq = v_ptr_base + k_bos * k_row_stride
        kv_rows = start_n + tl.arange(0, BLOCK_N)
        k_cols = tl.arange(0, BLOCK_K)
        v_cols = d_start + tl.arange(0, BLOCK_D)
        tle.gpu.copy(
            k_ptr_seq + kv_rows[:, None] * k_row_stride + k_cols[None, :],
            sK,
            (BLOCK_N, BLOCK_K),
            mask=(kv_rows[:, None] < k_len) & (k_cols[None, :] < LOAD_D),
        )
        tle.gpu.copy(
            v_ptr_seq + kv_rows[:, None] * k_row_stride + v_cols[None, :],
            sV,
            (BLOCK_N, BLOCK_D),
            mask=(kv_rows[:, None] < k_len) & (v_cols[None, :] < LOAD_D),
        )
        (q_descale, k_descale, v_descale) = _load_fp8_block_descales(
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            q_descale_batch_stride,
            q_descale_head_stride,
            q_descale_block_stride,
            k_descale_batch_stride,
            k_descale_head_stride,
            k_descale_block_stride,
            v_descale_batch_stride,
            v_descale_head_stride,
            v_descale_block_stride,
            bid,
            hid,
            kv_hid,
            m_block * BLOCK_M,
            n_block * BLOCK_N,
        )
        tl.debug_barrier()
        S = tle.gpu.wgmma(sQ, sK, trans_b=True)
        S = tle.gpu.wgmma_wait(0, S)
        S *= q_descale * k_descale
        S = apply_softcap(S, softcap, is_softcap)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            is_even_mn=is_even_mn,
            is_causal=is_causal,
            is_local=is_local,
        )
        (acc_, P, rowmax_, rowsum_) = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=True,
            fp8_p_max=fp8_p_max,
        )
        (P_fp8, P_residual) = _quantize_probability(P)
        tl.store(tle.gpu.local_ptr(sP), P_fp8)
        tl.store(tle.gpu.local_ptr(sPr), P_residual)
        tl.debug_barrier()
        pv_main = tle.gpu.wgmma(sP, sV)
        pv_main = tle.gpu.wgmma_wait(0, pv_main)
        pv_residual = tle.gpu.wgmma(sPr, sV)
        pv_residual = tle.gpu.wgmma_wait(0, pv_residual)
        pv_main += pv_residual * (1.0 / 32.0)
        acc_ += pv_main * v_descale
        tl.debug_barrier()
        n_block -= 1
    for n_block in tl.range(
        n_block_max - n_masking_steps - 1,
        n_block_min - 1,
        step=-1,
        num_stages=num_stages,
    ):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        start_n = n_block * BLOCK_N
        k_ptr_seq = k_ptr_base + k_bos * k_row_stride
        v_ptr_seq = v_ptr_base + k_bos * k_row_stride
        kv_rows = start_n + tl.arange(0, BLOCK_N)
        k_cols = tl.arange(0, BLOCK_K)
        v_cols = d_start + tl.arange(0, BLOCK_D)
        tle.gpu.copy(
            k_ptr_seq + kv_rows[:, None] * k_row_stride + k_cols[None, :],
            sK,
            (BLOCK_N, BLOCK_K),
            mask=(kv_rows[:, None] < k_len) & (k_cols[None, :] < LOAD_D),
        )
        tle.gpu.copy(
            v_ptr_seq + kv_rows[:, None] * k_row_stride + v_cols[None, :],
            sV,
            (BLOCK_N, BLOCK_D),
            mask=(kv_rows[:, None] < k_len) & (v_cols[None, :] < LOAD_D),
        )
        (q_descale, k_descale, v_descale) = _load_fp8_block_descales(
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            q_descale_batch_stride,
            q_descale_head_stride,
            q_descale_block_stride,
            k_descale_batch_stride,
            k_descale_head_stride,
            k_descale_block_stride,
            v_descale_batch_stride,
            v_descale_head_stride,
            v_descale_block_stride,
            bid,
            hid,
            kv_hid,
            m_block * BLOCK_M,
            n_block * BLOCK_N,
        )
        tl.debug_barrier()
        S = tle.gpu.wgmma(sQ, sK, trans_b=True)
        S = tle.gpu.wgmma_wait(0, S)
        S *= q_descale * k_descale
        S = apply_softcap(S, softcap, is_softcap)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
        )
        (acc_, P, rowmax_, rowsum_) = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=is_local,
            fp8_p_max=fp8_p_max,
        )
        (P_fp8, P_residual) = _quantize_probability(P)
        tl.store(tle.gpu.local_ptr(sP), P_fp8)
        tl.store(tle.gpu.local_ptr(sPr), P_residual)
        tl.debug_barrier()
        pv_main = tle.gpu.wgmma(sP, sV)
        pv_main = tle.gpu.wgmma_wait(0, pv_main)
        pv_residual = tle.gpu.wgmma(sPr, sV)
        pv_residual = tle.gpu.wgmma_wait(0, pv_residual)
        pv_main += pv_residual * (1.0 / 32.0)
        acc_ += pv_main * v_descale
        tl.debug_barrier()
    lse = tl.where(
        (rowsum_ == 0) | (rowsum_ != rowsum_),
        float("inf"),
        rowmax_ * scale_softmax
        + tl.log(rowsum_ * (1.0 / 256.0 if fp8_p_max == 448.0 else 1.0)),
    )
    inv_sum = tl.where((rowsum_ == 0) | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)
    acc_ *= inv_sum[:, None]
    out = acc_.to(o_ptr.type.element_ty)
    o_row_offset = hid * o_head_stride
    gO = tl.make_block_ptr(
        base=o_ptr + o_offset + o_row_offset,
        shape=(q_len, d),
        strides=(o_row_stride, 1),
        offsets=(0, d_start),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(gO.advance([m_block * BLOCK_M, 0]), out, boundary_check=(0, 1))
    softmax_lse_ptr += hid * total_q
    lse_row_offset = lse_offset + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(
        softmax_lse_ptr + lse_row_offset,
        lse,
        mask=(lse_row_offset < lse_offset + q_len) & (d_split == 0),
    )


def _normalize_varlen_window(
    max_seqlen_q,
    max_seqlen_k,
    alibi_slopes,
    is_causal,
    window_size_left,
    window_size_right,
):
    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    unlimited_right = max(max_seqlen_q, max_seqlen_k)
    if window_size_right >= unlimited_right:
        window_size_right = -1

    is_local = window_size_left >= 0
    is_local = is_local or (window_size_right >= 0 and not is_causal)
    if is_local:
        # A negative side means unbounded, not an offset of minus one.
        window_size_left = max_seqlen_k if window_size_left < 0 else window_size_left
        window_size_right = (
            unlimited_right if window_size_right < 0 else window_size_right
        )
    return is_causal, is_local, window_size_left, window_size_right


def _fits_varlen_tle_indexing(args, head_size):
    # Copy lengths, scalar strides and element offsets must fit signed int32.
    # Padding may exceed that range even when the original input fits.
    index_max = (1 << 31) - 1
    padded_d = triton.cdiv(head_size, 16) * 16
    if any(
        tensor.shape[0] * tensor.shape[1] * padded_d > index_max for tensor in args[:3]
    ):
        return False
    for tensor in args[:4]:
        strides = tensor.stride()
        if max(strides[:2]) > index_max:
            return False
        if (
            tensor.numel()
            and sum((size - 1) * stride for size, stride in zip(tensor.shape, strides))
            > index_max
        ):
            return False
    return True


def _get_varlen_fwd_config(
    args,
    head_size,
    use_varlen_split_d,
    is_paged,
    is_causal,
    window_size_left,
    window_size_right,
    is_alibi,
    is_softcap,
    is_dropout,
    *_,
):
    cfg = runtime.get_heuristic_config("mha_block_32")
    cfg_params = {
        "BLOCK_M": cfg["BLOCK_M"](args),
        "BLOCK_N": cfg["BLOCK_N"](args),
        "BLOCK_K": max(32, triton.next_power_of_2(head_size)),
        "BLOCK_D": (
            64 if use_varlen_split_d else max(32, triton.next_power_of_2(head_size))
        ),
        "SPLIT_D": use_varlen_split_d,
        "num_warps": cfg["num_warps"](args),
        "num_stages": 1 if not is_paged else cfg["num_stages"](args),
    }
    if head_size not in (64, 128) and not is_paged:
        cfg_params.update(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)
    if (
        HAS_TLE
        and not is_paged
        and not is_dropout
        and not is_alibi
        and not is_softcap
        and args[0].dtype == torch.float8_e4m3fn
        and torch_device_fn.get_device_capability(args[0].device) == (3, 1)
        and _fits_varlen_tle_indexing(args, head_size)
    ):
        # Stage FP8 operands directly in shared memory for native SQMMA.
        # E5M2 and paged KV retain the common Triton implementation.
        cfg_params["kernel"] = flash_varlen_fwd_kernel_tle
        cfg_params.update(BLOCK_M=32, BLOCK_N=64, num_warps=4, num_stages=1)
        if head_size >= 128:
            cfg_params["BLOCK_D"] = 128
        if head_size > 128:
            cfg_params.update(BLOCK_M=64, num_warps=8)
        if head_size % 16:
            # A byte-copy kernel pads the physical head stride and zeros its
            # tail. Logical dimensions, descales and the output stay intact.
            padded = _pad_fp8_qkv(*args[:3])
            prepared = list(args)
            for name, tensor in zip(("q", "k", "v"), padded):
                prepared[_FWD_PARAM_INDEX[f"{name}_ptr"]] = tensor
                prepared[_FWD_PARAM_INDEX[f"{name}_row_stride"]] = tensor.stride(0)
                prepared[_FWD_PARAM_INDEX[f"{name}_head_stride"]] = tensor.stride(1)
            cfg_params["args"] = tuple(prepared)
            cfg_params["PADDED_D"] = triton.cdiv(head_size, 16) * 16
        cfg_params["SPLIT_D"] = head_size > cfg_params["BLOCK_D"]
    return cfg_params


def _mha_varlan_fwd(*args, **kwargs):
    return mha_varlan_fwd(
        *args,
        **kwargs,
        window_normalizer=_normalize_varlen_window,
        get_config=_get_varlen_fwd_config,
        pv_dot=_fp8_pv_dot,
        allow_group_swap=False,
    )


def flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # Compatibility arguments from the shared FlashAttention API.
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """Run block-scaled FP8 FlashAttention-2 on MUSA, including MQA and GQA.

    Q/K/V use E4M3FN or E5M2 with descales per logical 128-token block.
    The output uses ``out.dtype`` when provided, or BF16 otherwise.
    """
    if torch_device_fn.get_device_capability()[0] < 3:
        raise NotImplementedError(
            "W8A8 FP8 attention requires MUSA capability 3 or newer"
        )
    return _flash_attn_varlen_func_w8a8_fp8(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        q_v=q_v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        block_table=block_table,
        return_softmax_lse=return_softmax_lse,
        out=out,
        scheduler_metadata=scheduler_metadata,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=s_aux,
        num_splits=num_splits,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
        fa_version=fa_version,
        fp8_dtypes=(torch.float8_e4m3fn, torch.float8_e5m2),
        allow_gqa=True,
        use_dense=False,
        varlen_fwd=_mha_varlan_fwd,
    )

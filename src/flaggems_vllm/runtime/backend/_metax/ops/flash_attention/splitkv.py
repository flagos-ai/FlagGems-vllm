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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.flash_kernel import (
    apply_alibi,
    apply_dropout,
    apply_mask,
    apply_softcap,
    load_from_kvcache,
    softmax_rescale,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.common import (
    tn_compile_scenario,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.scheduling import (
    MMA_MIN_BLOCK_M,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.tn_direct import (
    flash_varlen_fwd_d256_tn_kernel,
)
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


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
def flash_varlen_splitkv_kernel(
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
    # sizes
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
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    k_page_stride,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    Q_TILES: tl.constexpr,
):
    split_id = tl.program_id(0)
    batch_q_pid = tl.program_id(1)
    head_pid = tl.program_id(2)
    bid = batch_q_pid // Q_TILES
    m_block = batch_q_pid % Q_TILES
    o_ptr += split_id * total_q * h * d
    softmax_lse_ptr += split_id * h * total_q
    # num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        # Current request's start offset in the batched Q
        q_offset = q_bos * q_row_stride
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos * 1
    else:
        q_len = seqlen_q
        q_offset = bid * q_batch_stride
        o_offset = bid * o_batch_stride
        lse_offset = bid * seqlen_q

    if is_cu_seqlens_k:
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len_cache = k_eos - k_bos
        # k_offset = k_bos * k_row_stride
    else:
        k_len_cache = seqlen_k
        # k_offset = bid * k_batch_stride

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_len = k_len_cache

    packed_row = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_idx = packed_row
    query_head = head_pid
    kv_head = head_pid // h_hk_ratio
    effective_q_len = q_len

    # Noop CTA
    if m_block * BLOCK_M >= effective_q_len:
        return

    # is_even_mn = (q_len % BLOCK_M == 0) and (k_len % BLOCK_N == 0)
    is_even_mn: tl.constexpr = False

    if is_local:
        n_block_min = max(
            0,
            (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N,
        )
    else:
        n_block_min = 0

    n_block_max = tl.cdiv(k_len, BLOCK_N)
    if is_causal or is_local:
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + k_len - q_len + window_size_right,
                BLOCK_N,
            ),
        )

    span_blocks = max(n_block_max - n_block_min, 0)
    blocks_per_split = tl.cdiv(span_blocks, NUM_SPLITS)
    split_block_min = min(n_block_max, n_block_min + split_id * blocks_per_split)
    split_block_max = min(n_block_max, split_block_min + blocks_per_split)
    n_block_min = split_block_min
    n_block_max = split_block_max

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)

    # Locate the page table entry for the current batch element
    if is_paged:
        page_table_ptr += bid * page_table_batch_stride
    # Calculate the starting offset of q for the current head
    # Calculate the starting offset of k and v for the current head
    k_row_offset = kv_head * k_head_stride
    # Shift the k, v pointers to align with the current head
    k_ptr_base = k_ptr + k_row_offset
    v_ptr_base = v_ptr + k_row_offset

    q_row_offset = query_head * q_head_stride
    gQ = tl.make_block_ptr(
        base=q_ptr + q_offset + q_row_offset,
        shape=(q_len, d),
        strides=(q_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    bQ = tl.load(gQ.advance([m_block * BLOCK_M, 0]), boundary_check=(0, 1))

    acc_ = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + query_head
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    if not is_causal and not is_local:
        n_masking_steps = 1
    elif is_even_mn:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N) + 1

    n_masking_steps = max(0, min(n_block_max - n_block_min, n_masking_steps))

    n_block = n_block_max - 1
    for step in tl.range(0, n_masking_steps):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        if is_paged:
            bK, bV = load_from_kvcache(
                col_idx,
                k_len,
                page_table_ptr,
                k_ptr_base,
                v_ptr_base,
                block_size,
                d,
                k_row_stride,
                BLOCK_K=BLOCK_K,
                k_page_stride=k_page_stride,
                boundary_check=True,
            )
        else:
            start_n = n_block * BLOCK_N
            k_ptr_seq = k_ptr_base + k_bos * k_row_stride
            v_ptr_seq = v_ptr_base + k_bos * k_row_stride
            gK = tl.make_block_ptr(
                base=k_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            gV = tl.make_block_ptr(
                base=v_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            bK = tl.load(gK, boundary_check=(0, 1))
            bK = tl.trans(bK)
            bV = tl.load(gV, boundary_check=(0, 1))
        S = tl.dot(bQ, bK, out_dtype=tl.float32)
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

        acc_, P, rowmax_, rowsum_ = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=True,
        )
        P = P.to(v_ptr.type.element_ty)

        if is_dropout:
            P = apply_dropout(
                P,
                n_block * BLOCK_N,
                m_block * BLOCK_M,
                k_len,
                bid,
                head_pid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        acc_ = tl.dot(P, bV, acc_)
        n_block -= 1

    for n_block in tl.range(
        n_block_max - n_masking_steps - 1, n_block_min - 1, step=-1
    ):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        if is_paged:
            bK, bV = load_from_kvcache(
                col_idx,
                k_len,
                page_table_ptr,
                k_ptr_base,
                v_ptr_base,
                block_size,
                d,
                k_row_stride,
                BLOCK_K=BLOCK_K,
                k_page_stride=k_page_stride,
            )
        else:
            start_n = n_block * BLOCK_N
            k_ptr_seq = k_ptr_base + k_bos * k_row_stride
            v_ptr_seq = v_ptr_base + k_bos * k_row_stride
            gK = tl.make_block_ptr(
                base=k_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            gV = tl.make_block_ptr(
                base=v_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            bK = tl.load(gK)
            bK = tl.trans(bK)
            bV = tl.load(gV)
        S = tl.dot(bQ, bK, out_dtype=tl.float32)
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

        acc_, P, rowmax_, rowsum_ = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=is_local,
        )
        P = P.to(v_ptr.type.element_ty)

        if is_dropout:
            P = apply_dropout(
                P,
                m_block * BLOCK_M,
                n_block * BLOCK_N,
                k_len,
                bid,
                head_pid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        acc_ = tl.dot(P, bV, acc_)

    # LSE
    valid_rowsum = (
        (rowsum_ > 0.0)
        & (rowsum_ < float("inf"))
        & (rowmax_ > float("-inf"))
        & (rowmax_ < float("inf"))
    )
    safe_rowsum = tl.where(valid_rowsum, rowsum_, 1.0)
    safe_rowmax = tl.where(valid_rowsum, rowmax_, 0.0)
    lse = tl.where(
        valid_rowsum,
        safe_rowmax * scale_softmax + tl.log(safe_rowsum),
        float("-inf"),
    )
    inv_sum = tl.where(valid_rowsum, 1.0 / safe_rowsum, 0.0)

    acc_ *= inv_sum[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    # Write back output
    o_row_offset = query_head * o_head_stride
    gO = tl.make_block_ptr(
        base=o_ptr + o_offset + o_row_offset,
        shape=(q_len, d),
        strides=(o_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    tl.store(gO.advance([m_block * BLOCK_M, 0]), out, boundary_check=(0, 1))

    # Write back lse
    # lse shape: [h, total_q]
    softmax_lse_ptr += query_head * total_q
    lse_row_offset = lse_offset + packed_row
    tl.store(
        softmax_lse_ptr + lse_row_offset,
        lse,
        mask=lse_row_offset < (lse_offset + q_len),
    )


@libentry()
@triton.jit
def flash_varlen_splitkv_merge_kernel(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    o_row_stride,
    o_head_stride,
    total_q: tl.constexpr,
    h: tl.constexpr,
    d: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MAX_N_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SELECTION_PTR=None,
    SKIP_SELECTED: tl.constexpr = False,
):
    if SKIP_SELECTED:
        if tl.load(SELECTION_PTR) >= 0:
            return
    q_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.arange(0, MAX_N_SPLITS)
    split_mask = split_idx < NUM_SPLITS

    lse_offsets = split_idx * h * total_q + head_idx * total_q + q_idx
    split_lse = tl.load(
        lse_splits_ptr + lse_offsets,
        mask=split_mask,
        other=float("-inf"),
    )
    valid_lse = split_mask & (split_lse > float("-inf")) & (split_lse < float("inf"))
    max_lse = tl.max(tl.where(valid_lse, split_lse, float("-inf")), axis=0)
    has_valid_lse = max_lse > float("-inf")
    safe_max_lse = tl.where(has_valid_lse, max_lse, 0.0)
    split_scale = tl.where(valid_lse, tl.exp(split_lse - safe_max_lse), 0.0)
    scale_sum = tl.sum(split_scale, axis=0)
    safe_scale_sum = tl.where(has_valid_lse, scale_sum, 1.0)
    weights = split_scale / safe_scale_sum

    col = tl.arange(0, BLOCK_K)
    split_out_offsets = (
        split_idx[:, None] * total_q * h * d + (q_idx * h + head_idx) * d + col[None, :]
    )
    split_out = tl.load(
        out_splits_ptr + split_out_offsets,
        mask=split_mask[:, None] & (col[None, :] < d),
        other=0.0,
    )
    out = tl.sum(weights[:, None] * split_out, axis=0)
    out_offsets = q_idx * o_row_stride + head_idx * o_head_stride + col
    tl.store(out_ptr + out_offsets, out, mask=col < d)
    merged_lse = tl.where(
        has_valid_lse,
        tl.log(safe_scale_sum) + safe_max_lse,
        float("-inf"),
    )
    tl.store(lse_ptr + head_idx * total_q + q_idx, merged_lse)


def launch_splitkv(
    params,
    *,
    max_seqlen_q,
    batch_size,
    num_heads,
    total_q,
    head_size,
    num_splits,
    block_m,
    block_n,
    q_tiles,
):
    """Launch split partial attention followed by an FP32 softmax merge."""

    if not 2 <= num_splits <= 32:
        raise RuntimeError("MetaX Split-KV requires 2 to 32 splits")
    if (block_m, block_n) != (MMA_MIN_BLOCK_M, 16):
        raise RuntimeError("MetaX Split-KV received an invalid tile")
    expected_q_tiles = (max_seqlen_q + block_m - 1) // block_m
    if q_tiles != expected_q_tiles:
        raise RuntimeError("MetaX Split-KV received an invalid q_tiles value")

    original_out = params.o_ptr
    original_lse = params.softmax_lse_ptr
    out_splits = torch.empty(
        (num_splits, total_q, num_heads, head_size),
        dtype=torch.float32,
        device=original_out.device,
    )
    lse_splits = torch.empty(
        (num_splits, num_heads, total_q),
        dtype=torch.float32,
        device=original_out.device,
    )

    # Partial results use a new contiguous FP32 layout. In particular,
    # single-request GQA head/sequence swapping can leave the caller's output
    # as a strided view; its strides must not index these temporary buffers.
    # Keep the original output strides for the merge and avoid mutating params.
    partial_args = {
        "o_ptr": out_splits,
        "softmax_lse_ptr": lse_splits,
        "o_row_stride": num_heads * head_size,
        "o_head_stride": head_size,
        "o_batch_stride": max_seqlen_q * num_heads * head_size,
    }
    args = tuple(
        partial_args[key] if key in partial_args else getattr(params, key)
        for key in params.__slots__
    )

    cfg_params = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": triton.next_power_of_2(head_size),
        "num_warps": 1,
        "num_stages": 1,
        "NUM_SPLITS": num_splits,
        "Q_TILES": q_tiles,
    }
    logger.debug("Running split-KV partial kernel with config: %s", cfg_params)
    partial = flash_varlen_splitkv_kernel[(num_splits, batch_size * q_tiles, num_heads)]
    partial(*args, **cfg_params)

    block_k = triton.next_power_of_2(head_size)
    merge = flash_varlen_splitkv_merge_kernel[(total_q, num_heads)]
    merge(
        original_out,
        original_lse,
        out_splits,
        lse_splits,
        params.o_row_stride,
        params.o_head_stride,
        total_q,
        num_heads,
        head_size,
        num_splits,
        triton.next_power_of_2(num_splits),
        block_k,
        num_warps=1,
        num_stages=1,
    )
    return partial


def launch_page16_decode(params, *, batch_size):
    """Run BM16/N16 MMA without partial buffers or merge."""
    args = tuple(getattr(params, key) for key in params.__slots__)
    cfg = {
        "BLOCK_M": MMA_MIN_BLOCK_M,
        "BLOCK_N": 16,
        "BLOCK_K": 256,
        "num_warps": 1,
        "num_stages": 1,
        "NUM_SPLITS": 1,
        "Q_TILES": 1,
    }
    return flash_varlen_splitkv_kernel[(1, batch_size, 1)](*args, **cfg)


def launch_d256_tn_splitkv(
    params,
    *,
    max_seqlen_q,
    batch_size,
    num_heads,
    total_q,
    num_splits,
    block_m=32,
    block_n=128,
    _skip_selection_desc=None,
):
    """Run packed partials and FP32 merge with a validated private tile."""
    skip_desc = _skip_selection_desc
    packed_decode = (
        params.seqlenq_ngroups_swapped
        and max_seqlen_q in (4, 8)
        and params.h_hk_ratio == 1
        and num_heads == 1
    )
    if params.q_ptr.dtype != torch.bfloat16:
        raise ValueError("Async-TN Split-KV requires BF16 Q/K/V")
    page16_short_q = (
        params.block_size == 16
        and not packed_decode
        and num_heads == params.h == params.h_hk_ratio == 4
        and params.hk == 1
        and params.is_cu_seqlens_q
        and params.is_seqused_k
        and 4 < max_seqlen_q <= 132
        and batch_size <= 64
        and total_q <= max_seqlen_q + batch_size - 1
    )
    if (
        not params.is_paged
        or params.d != 256
        or not (
            params.block_size == 32
            or (params.block_size == 16 and packed_decode and max_seqlen_q == 4)
            or page16_short_q
        )
    ):
        raise ValueError(
            "Async-TN Split-KV requires paged D256; page16 requires packed or bounded-short GQA4"
        )
    if (not params.is_causal and not packed_decode) or params.is_local:
        raise ValueError("Async-TN Split-KV only supports causal global attention")
    if params.is_dropout or params.is_alibi or params.is_softcap:
        raise ValueError(
            "Async-TN Split-KV does not support optional attention features"
        )
    if params.cu_seqlens_q_ptr is None and not packed_decode:
        raise ValueError("Async-TN Split-KV requires varlen Q prefix sums")
    if params.cu_seqlens_k_ptr is None and params.seqused_k_ptr is None:
        raise ValueError("Async-TN Split-KV requires KV lengths")
    split_limit = 128 if packed_decode else 32
    if not 2 <= num_splits <= split_limit:
        raise ValueError(f"Async-TN Split-KV requires 2 to {split_limit} splits")
    if num_heads % params.h_hk_ratio != 0:
        raise ValueError("Async-TN Split-KV received an invalid GQA ratio")
    if (block_m, block_n) != (32, 128) and not (
        (block_m, block_n) == (16, 64)
        and packed_decode
        and max_seqlen_q == 8
        and params.block_size == 32
        and 49 <= batch_size <= 64
    ):
        raise ValueError("Unsupported packed-decode tile")
    block_k = 128
    q_tiles = triton.cdiv(max_seqlen_q * params.h_hk_ratio, block_m)
    head_programs = num_heads // params.h_hk_ratio
    original_out = params.o_ptr
    original_lse = params.softmax_lse_ptr
    out_splits = torch.empty(
        (num_splits, total_q, num_heads, 256),
        dtype=torch.float32,
        device=original_out.device,
    )
    lse_splits = torch.empty(
        (num_splits, num_heads, total_q),
        dtype=torch.float32,
        device=original_out.device,
    )
    grid = (head_programs, batch_size, num_splits * q_tiles)
    adaptive_page16 = packed_decode and max_seqlen_q == 4 and params.block_size == 16
    cfg = {
        "COMPACT_WORKLIST": False,
        "SKIP_SELECTED": skip_desc is not None,
        "PAGE_SIZE": params.block_size,
        "PAGE16_ADAPTIVE_DECODE": adaptive_page16,
        "PACKED_DECODE_Q": max_seqlen_q if packed_decode else 0,
        "SPLIT_KV": True,
        "NUM_SPLITS": num_splits,
        "Q_TILES": q_tiles,
        "GRID_ORDER": 0 if block_n == 64 else 1,
        "REVERSE_Q_TILES": False,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "QK_BLOCK_K": 64,
        "FAST_CAUSAL_ALIGNED": False,
        "num_warps": 4,
        "num_stages": 4,
        "pipeline": "cpasync",
        "scenario": tn_compile_scenario(
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            out_splits,
            lse_splits,
            params.page_table_ptr,
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
        ),
        "pipeline_load_num": -1,
        "inner_stages": (0, 0),
    }
    # Specialize LibEntry on the same scenario used for compilation.
    cfg["TN_SCENARIO"] = cfg["scenario"]
    logger.debug("Running D256 async-TN Split-KV with config: %s", cfg)
    # A BM64 full split and its two complementary BM32 tiles share the same
    # KV partition. Keep noncontiguous and wide-address inputs on the old path.
    full_rows = (
        not packed_decode
        and max_seqlen_q >= 128
        and num_heads == params.h_hk_ratio == 8
        and params.q_row_stride == 2048
        and params.q_head_stride == 256
        and "noaddropt" not in cfg["scenario"]
    )
    for region in [1, 2] if full_rows else [0]:
        cfg["SPLIT_REGION"] = region
        cfg["BLOCK_M"] = 64 if region == 1 else block_m
        cfg["Q_TILES"] = triton.cdiv(max_seqlen_q * params.h_hk_ratio, cfg["BLOCK_M"])
        if block_n == 64:
            # Adjacent CTAs process the same request's KV partitions. This
            # also avoids interleaving padded empty requests between splits.
            grid = (num_splits * cfg["Q_TILES"], batch_size, head_programs)
        else:
            grid = (head_programs, batch_size, num_splits * cfg["Q_TILES"])
        partial = flash_varlen_fwd_d256_tn_kernel[grid](
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            out_splits,
            lse_splits,
            params.q_row_stride,
            params.k_row_stride,
            params.v_row_stride,
            params.q_head_stride,
            params.k_head_stride,
            params.v_head_stride,
            out_splits.stride(1),
            out_splits.stride(2),
            params.scale_softmax,
            params.scale_softmax_log2,
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
            params.is_cu_seqlens_k,
            params.is_seqused_k,
            num_heads,
            params.h_hk_ratio,
            total_q,
            params.page_table_ptr,
            params.page_table_batch_stride,
            params.k_page_stride,
            params.v_ptr.stride(0),
            skip_desc if skip_desc is not None else params.page_table_ptr,
            **cfg,
        )
    if adaptive_page16:
        finalize = (
            _page16_decode_batch_finalize_kernel
            if batch_size > 128
            else _page16_decode_finalize_kernel
        )
        grid = (batch_size,) if batch_size > 128 else (total_q,)
        finalize[grid](
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            original_out,
            original_lse,
            out_splits,
            lse_splits,
            params.seqused_k_ptr,
            params.page_table_ptr,
            params.q_row_stride,
            params.q_head_stride,
            params.k_row_stride,
            params.v_row_stride,
            params.k_page_stride,
            params.v_ptr.stride(0),
            params.page_table_batch_stride,
            params.o_row_stride,
            params.o_head_stride,
            total_q,
            num_splits,
            triton.next_power_of_2(num_splits),
            params.scale_softmax,
            num_warps=4,
            num_stages=1,
        )
        return partial
    merge = flash_varlen_splitkv_merge_kernel[total_q, num_heads]
    merge(
        original_out,
        original_lse,
        out_splits,
        lse_splits,
        params.o_row_stride,
        params.o_head_stride,
        total_q,
        num_heads,
        256,
        num_splits,
        triton.next_power_of_2(num_splits),
        256,
        SELECTION_PTR=skip_desc,
        SKIP_SELECTED=skip_desc is not None,
        num_warps=4 if num_splits > 32 else 1,
        num_stages=1,
    )
    return partial


@libentry()
@triton.jit
def _page16_decode_finalize_kernel(
    Q,
    K,
    V,
    O,
    L,
    PO,
    PL,
    USED,
    T,
    QR: tl.constexpr,
    QH: tl.constexpr,
    KR: tl.constexpr,
    VR: tl.constexpr,
    KP: tl.constexpr,
    VP: tl.constexpr,
    TS: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL: tl.constexpr,
    S: tl.constexpr,
    RS: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    bid = row // 4
    length = tl.load(USED + bid)
    d = tl.arange(0, 256)
    if length <= 32:
        if length == 0:
            result = tl.full((256,), 0.0, tl.float32)
            lse = float("-inf")
        elif length == 1:
            page_single = tl.load(T + bid * TS).to(tl.int64)
            q = tl.load(Q + row * QR + d).to(tl.float32)
            k = tl.load(K + page_single * KP + d).to(tl.float32)
            result = tl.load(V + page_single * VP + d).to(tl.float32)
            lse = tl.sum(q * k, 0) * SCALE
        else:
            q = tl.load(Q + row * QR + d).to(tl.float32)
            n = tl.arange(0, 32)
            page = tl.load(T + bid * TS + n // 16, n < length, other=0).to(tl.int64)
            keys = tl.load(
                K + page[:, None] * KP + n[:, None] % 16 * KR + d[None, :],
                n[:, None] < length,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], 1) * SCALE
            scores = tl.where(n < length, scores, float("-inf"))
            maximum = tl.max(scores, 0)
            prob = tl.exp(scores - maximum)
            den = tl.sum(prob, 0)
            values = tl.load(
                V + page[:, None] * VP + n[:, None] % 16 * VR + d[None, :],
                n[:, None] < length,
                other=0.0,
            ).to(tl.float32)
            result = tl.sum(values * prob[:, None], 0) / den
            lse = maximum + tl.log(den)
    else:
        s = tl.arange(0, RS)
        lses = tl.load(PL + s * TOTAL + row, s < S, other=float("-inf"))
        maximum = tl.max(lses, 0)
        maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
        weights = tl.exp(lses - maximum)
        denominator = tl.sum(weights, 0)
        partial = tl.load(
            PO + s[:, None] * TOTAL * 256 + row * 256 + d[None, :],
            s[:, None] < S,
            other=0.0,
        )
        result = tl.sum(partial * weights[:, None], 0) / denominator
        lse = maximum + tl.log(denominator)
    tl.store(O + row * OR + d, result.to(O.dtype.element_ty))
    tl.store(L + row, lse)


@libentry()
@triton.jit
def _page16_decode_batch_finalize_kernel(
    Q,
    K,
    V,
    O,
    L,
    PO,
    PL,
    USED,
    T,
    QR: tl.constexpr,
    QH: tl.constexpr,
    KR: tl.constexpr,
    VR: tl.constexpr,
    KP: tl.constexpr,
    VP: tl.constexpr,
    TS: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL: tl.constexpr,
    S: tl.constexpr,
    RS: tl.constexpr,
    SCALE: tl.constexpr,
):
    bid = tl.program_id(0)
    length = tl.load(USED + bid)
    d = tl.arange(0, 256)
    if length <= 32:
        h = tl.arange(0, 16)
        if length == 0:
            short_o = tl.full((16, 256), 0.0, tl.float32)
            short_l = tl.full((16,), float("-inf"), tl.float32)
        elif length == 1:
            page_one = tl.load(T + bid * TS).to(tl.int64)
            q_one = tl.load(
                Q + (bid * 4 + h[:, None]) * QR + d[None, :], h[:, None] < 4, other=0.0
            ).to(tl.float32)
            k_one = tl.load(K + page_one * KP + d).to(tl.float32)
            v_one = tl.load(V + page_one * VP + d).to(tl.float32)
            short_o = tl.broadcast_to(v_one[None, :], (16, 256))
            short_l = tl.sum(q_one * k_one[None, :], 1) * SCALE
        else:
            n = tl.arange(0, 32)
            page_many = tl.load(T + bid * TS + n // 16, n < length, other=0).to(
                tl.int64
            )
            q_many = tl.load(
                Q + (bid * 4 + h[:, None]) * QR + d[None, :], h[:, None] < 4, other=0.0
            )
            k_many = tl.load(
                K + page_many[None, :] * KP + n[None, :] % 16 * KR + d[:, None],
                n[None, :] < length,
                other=0.0,
            )
            score_many = tl.dot(q_many, k_many) * SCALE
            score_many = tl.where(n[None, :] < length, score_many, float("-inf"))
            maximum_many = tl.max(score_many, 1)
            prob_many = tl.exp(score_many - maximum_many[:, None])
            denominator_many = tl.sum(prob_many, 1)
            v_many = tl.load(
                V + page_many[:, None] * VP + n[:, None] % 16 * VR + d[None, :],
                n[:, None] < length,
                other=0.0,
            )
            short_o = (
                tl.dot(prob_many.to(v_many.dtype), v_many) / denominator_many[:, None]
            )
            short_l = maximum_many + tl.log(denominator_many)
        tl.store(
            O + (bid * 4 + h[:, None]) * OR + d[None, :],
            short_o.to(O.dtype.element_ty),
            h[:, None] < 4,
        )
        tl.store(L + bid * 4 + h, short_l, h < 4)
    else:
        for head in range(4):
            row = bid * 4 + head
            s = tl.arange(0, RS)
            lses = tl.load(PL + s * TOTAL + row, s < S, other=float("-inf"))
            maximum = tl.max(lses, 0)
            maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
            weights = tl.exp(lses - maximum)
            denominator = tl.sum(weights, 0)
            partial = tl.load(
                PO + s[:, None] * TOTAL * 256 + row * 256 + d[None, :],
                s[:, None] < S,
                other=0.0,
            )
            result = tl.sum(partial * weights[:, None], 0) / denominator
            lse = maximum + tl.log(denominator)
            tl.store(O + row * OR + d, result.to(O.dtype.element_ty))
            tl.store(L + row, lse)


@triton.jit(do_not_specialize=["B"])
def _residual_worklist(
    CU,
    WL,
    B,
    R: tl.constexpr,
    UPPER: tl.constexpr,
    RATIO: tl.constexpr,
    BM: tl.constexpr,
):
    bid = tl.program_id(0)
    b = tl.arange(0, R)
    start = tl.load(CU + b, b <= B, other=0)
    end = tl.load(CU + b + 1, b < B, other=0)
    tiles = tl.where(b < B, tl.cdiv(tl.maximum(end - start, 0) * RATIO, BM), 0)
    base = tl.sum(tl.where(b < bid, tiles, 0), 0)
    count = tl.sum(tl.where(b == bid, tiles, 0), 0)
    total = tl.sum(tiles, 0)
    row = tl.arange(0, 128)
    for offset in tl.range(0, count, 128):
        task = offset + row
        tl.store(WL + (base + task) * 2, bid, task < count)
        tl.store(WL + (base + task) * 2 + 1, task, task < count)
    if bid == 0:
        for offset in tl.range(total, UPPER, 128):
            task = offset + row
            tl.store(WL + task * 2, -1, task < UPPER)
            tl.store(WL + task * 2 + 1, 0, task < UPPER)


def launch_d256_tn_compact_splitkv(
    params,
    *,
    max_seqlen_q,
    batch_size,
    num_heads,
    total_q,
    num_splits,
    block_m=32,
    block_n=128,
):
    """Run packed partials and FP32 merge with a validated private tile."""
    assert (
        params.block_size == 32
        and params.h == params.h_hk_ratio == num_heads == 8
        and (not params.seqlenq_ngroups_swapped)
    )
    packed_decode = (
        params.seqlenq_ngroups_swapped
        and max_seqlen_q in (4, 8)
        and (params.h_hk_ratio == 1)
        and (num_heads == 1)
    )
    if params.q_ptr.dtype != torch.bfloat16:
        raise ValueError("Async-TN Split-KV requires BF16 Q/K/V")
    page16_short_q = (
        params.block_size == 16
        and (not packed_decode)
        and (num_heads == params.h == params.h_hk_ratio == 4)
        and (params.hk == 1)
        and params.is_cu_seqlens_q
        and params.is_seqused_k
        and (4 < max_seqlen_q <= 132)
        and (batch_size <= 64)
        and (total_q <= max_seqlen_q + batch_size - 1)
    )
    if (
        not params.is_paged
        or params.d != 256
        or (
            not (
                params.block_size == 32
                or (params.block_size == 16 and packed_decode and (max_seqlen_q == 4))
                or page16_short_q
            )
        )
    ):
        raise ValueError(
            "Async-TN Split-KV requires paged D256; page16 requires packed or bounded-short GQA4"
        )
    if not params.is_causal and (not packed_decode) or params.is_local:
        raise ValueError("Async-TN Split-KV only supports causal global attention")
    if params.is_dropout or params.is_alibi or params.is_softcap:
        raise ValueError(
            "Async-TN Split-KV does not support optional attention features"
        )
    if params.cu_seqlens_q_ptr is None and (not packed_decode):
        raise ValueError("Async-TN Split-KV requires varlen Q prefix sums")
    if params.cu_seqlens_k_ptr is None and params.seqused_k_ptr is None:
        raise ValueError("Async-TN Split-KV requires KV lengths")
    split_limit = 128 if packed_decode else 32
    if not 2 <= num_splits <= split_limit:
        raise ValueError(f"Async-TN Split-KV requires 2 to {split_limit} splits")
    if num_heads % params.h_hk_ratio != 0:
        raise ValueError("Async-TN Split-KV received an invalid GQA ratio")
    if (block_m, block_n) != (32, 128) and (
        not (
            (block_m, block_n) == (16, 64)
            and packed_decode
            and (max_seqlen_q == 8)
            and (params.block_size == 32)
            and (49 <= batch_size <= 64)
        )
    ):
        raise ValueError("Unsupported packed-decode tile")
    block_k = 128
    q_tiles = triton.cdiv(max_seqlen_q * params.h_hk_ratio, block_m)
    head_programs = num_heads // params.h_hk_ratio
    original_out = params.o_ptr
    original_lse = params.softmax_lse_ptr
    out_splits = torch.empty(
        (num_splits, total_q, num_heads, 256),
        dtype=torch.float32,
        device=original_out.device,
    )
    lse_splits = torch.empty(
        (num_splits, num_heads, total_q),
        dtype=torch.float32,
        device=original_out.device,
    )
    grid = (head_programs, batch_size, num_splits * q_tiles)
    adaptive_page16 = packed_decode and max_seqlen_q == 4 and (params.block_size == 16)
    cfg = {
        "COMPACT_WORKLIST": False,
        "PAGE_SIZE": params.block_size,
        "PAGE16_ADAPTIVE_DECODE": adaptive_page16,
        "PACKED_DECODE_Q": max_seqlen_q if packed_decode else 0,
        "SPLIT_KV": True,
        "NUM_SPLITS": num_splits,
        "Q_TILES": q_tiles,
        "GRID_ORDER": 0 if block_n == 64 else 1,
        "REVERSE_Q_TILES": False,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "QK_BLOCK_K": 64,
        "FAST_CAUSAL_ALIGNED": False,
        "num_warps": 4,
        "num_stages": 4,
        "pipeline": "cpasync",
        "scenario": tn_compile_scenario(
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            out_splits,
            lse_splits,
            params.page_table_ptr,
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
        ),
        "pipeline_load_num": -1,
        "inner_stages": (0, 0),
    }
    cfg["TN_SCENARIO"] = cfg["scenario"]
    logger.debug("Running D256 async-TN Split-KV with config: %s", cfg)
    full_rows = (
        not packed_decode
        and max_seqlen_q >= 128
        and (num_heads == params.h_hk_ratio == 8)
        and (params.q_row_stride == 2048)
        and (params.q_head_stride == 256)
        and ("noaddropt" not in cfg["scenario"])
    )
    for region in [1, 2] if full_rows else [0]:
        cfg["SPLIT_REGION"] = region
        cfg["BLOCK_M"] = 64 if region == 1 else block_m
        cfg["Q_TILES"] = triton.cdiv(max_seqlen_q * params.h_hk_ratio, cfg["BLOCK_M"])
        if block_n == 64:
            grid = (num_splits * cfg["Q_TILES"], batch_size, head_programs)
        else:
            grid = (head_programs, batch_size, num_splits * cfg["Q_TILES"])
        upper = triton.next_power_of_2(
            triton.cdiv(total_q * params.h_hk_ratio, cfg["BLOCK_M"]) + batch_size
        )
        work = torch.empty((upper, 2), dtype=torch.int32, device=params.q_ptr.device)
        _residual_worklist[batch_size,](
            params.cu_seqlens_q_ptr,
            work,
            batch_size,
            triton.next_power_of_2(batch_size + 1),
            upper,
            params.h_hk_ratio,
            cfg["BLOCK_M"],
            num_warps=1,
        )
        cfg["COMPACT_WORKLIST"] = True
        cfg["Q_TILES"] = upper
        grid = (head_programs, 1, num_splits * upper)
        partial = flash_varlen_fwd_d256_tn_kernel[grid](
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            out_splits,
            lse_splits,
            params.q_row_stride,
            params.k_row_stride,
            params.v_row_stride,
            params.q_head_stride,
            params.k_head_stride,
            params.v_head_stride,
            out_splits.stride(1),
            out_splits.stride(2),
            params.scale_softmax,
            params.scale_softmax_log2,
            params.cu_seqlens_q_ptr,
            params.cu_seqlens_k_ptr,
            params.seqused_k_ptr,
            params.is_cu_seqlens_k,
            params.is_seqused_k,
            num_heads,
            params.h_hk_ratio,
            total_q,
            params.page_table_ptr,
            params.page_table_batch_stride,
            params.k_page_stride,
            params.v_ptr.stride(0),
            work,
            **cfg,
        )
    merge = _d256_tn_compact_splitkv_merge_kernel[total_q, num_heads]
    merge(
        original_out,
        original_lse,
        out_splits,
        lse_splits,
        params.o_row_stride,
        params.o_head_stride,
        total_q,
        num_heads,
        256,
        num_splits,
        triton.next_power_of_2(num_splits),
        256,
        num_warps=4 if num_splits > 32 else 1,
        num_stages=1,
    )
    return partial


@libentry()
@triton.jit(do_not_specialize=["total_q"])
def _d256_tn_compact_splitkv_merge_kernel(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    o_row_stride,
    o_head_stride,
    total_q,
    h: tl.constexpr,
    d: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MAX_N_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    q_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.arange(0, MAX_N_SPLITS)
    split_mask = split_idx < NUM_SPLITS
    lse_offsets = split_idx * h * total_q + head_idx * total_q + q_idx
    split_lse = tl.load(
        lse_splits_ptr + lse_offsets, mask=split_mask, other=float("-inf")
    )
    valid_lse = split_mask & (split_lse > float("-inf")) & (split_lse < float("inf"))
    max_lse = tl.max(tl.where(valid_lse, split_lse, float("-inf")), axis=0)
    has_valid_lse = max_lse > float("-inf")
    safe_max_lse = tl.where(has_valid_lse, max_lse, 0.0)
    split_scale = tl.where(valid_lse, tl.exp(split_lse - safe_max_lse), 0.0)
    scale_sum = tl.sum(split_scale, axis=0)
    safe_scale_sum = tl.where(has_valid_lse, scale_sum, 1.0)
    weights = split_scale / safe_scale_sum
    col = tl.arange(0, BLOCK_K)
    split_out_offsets = (
        split_idx[:, None] * total_q * h * d + (q_idx * h + head_idx) * d + col[None, :]
    )
    split_out = tl.load(
        out_splits_ptr + split_out_offsets,
        mask=split_mask[:, None] & (col[None, :] < d),
        other=0.0,
    )
    out = tl.sum(weights[:, None] * split_out, axis=0)
    out_offsets = q_idx * o_row_stride + head_idx * o_head_stride + col
    tl.store(out_ptr + out_offsets, out, mask=col < d)
    merged_lse = tl.where(
        has_valid_lse, tl.log(safe_scale_sum) + safe_max_lse, float("-inf")
    )
    tl.store(lse_ptr + head_idx * total_q + q_idx, merged_lse)

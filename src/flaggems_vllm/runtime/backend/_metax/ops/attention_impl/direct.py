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

"""Base-equivalent direct varlen FlashAttention kernel and launcher."""

import logging

import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

from .common import (
    apply_alibi,
    apply_dropout,
    apply_mask,
    apply_softcap,
    load_from_kvcache,
    softmax_rescale,
)

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
def flash_varlen_fwd_kernel(
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
    worklist_ptr,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPACT_WORKLIST: tl.constexpr,
    GRID_ORDER: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    if GRID_ORDER == 1:
        task_pid = tl.program_id(2)
        rectangular_bid = tl.program_id(1)
        hid = tl.program_id(0)
    else:
        task_pid = tl.program_id(0)
        rectangular_bid = tl.program_id(1)
        hid = tl.program_id(2)

    if COMPACT_WORKLIST:
        descriptor_offset = task_pid * 2
        bid = tl.load(worklist_ptr + descriptor_offset).to(tl.int32)
        if bid < 0:
            return
        m_block = tl.load(worklist_ptr + descriptor_offset + 1).to(tl.int32)
    else:
        m_block = task_pid
        bid = rectangular_bid
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
    query_head = hid
    kv_head = hid // h_hk_ratio
    effective_q_len = q_len
    q_block_start = m_block * BLOCK_M
    q_block_end = (m_block + 1) * BLOCK_M

    # Noop CTA
    if m_block * BLOCK_M > effective_q_len:
        return

    # is_even_mn = (q_len % BLOCK_M == 0) and (k_len % BLOCK_N == 0)
    is_even_mn: tl.constexpr = False

    if is_local:
        n_block_min = max(
            0, (q_block_start + k_len - q_len - window_size_left) // BLOCK_N
        )
    else:
        n_block_min = 0

    n_block_max = tl.cdiv(k_len, BLOCK_N)
    if is_causal or is_local:
        n_block_max = min(
            n_block_max,
            tl.cdiv(q_block_end + k_len - q_len + window_size_right, BLOCK_N),
        )

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)

    # Locate the page table entry for the current batch element
    if is_paged:
        page_table_ptr += bid * page_table_batch_stride
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
        alibi_offset = bid * alibi_slopes_batch_stride + hid
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

    n_masking_steps = min(n_block_max - n_block_min, n_masking_steps)

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
                hid,
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
                hid,
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
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("-inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

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
    lse_row_offset = lse_offset + row_idx
    tl.store(
        softmax_lse_ptr + lse_row_offset,
        lse,
        mask=lse_row_offset < (lse_offset + q_len),
    )


def launch_direct(
    params,
    *,
    max_seqlen_q,
    batch_size,
    num_heads,
    total_q,
    head_size,
    is_paged,
    compact_worklist=None,
    worklist_block_m=0,
    grid_order="row_batch_head",
):
    """Launch Direct with rectangular or compact-worklist mapping."""

    use_compact_worklist = compact_worklist is not None
    grid_order_value = getattr(grid_order, "value", grid_order)
    grid_order_code = {
        "row_batch_head": 0,
        "head_batch_row": 1,
    }[grid_order_value]
    effective_num_heads = num_heads
    effective_max_q = max_seqlen_q

    def grid(args):
        row_tiles = (
            triton.cdiv(total_q, args["BLOCK_M"]) + batch_size - 1
            if use_compact_worklist
            else triton.cdiv(effective_max_q, args["BLOCK_M"])
        )
        if grid_order_code == 1:
            return (
                effective_num_heads,
                1 if use_compact_worklist else batch_size,
                row_tiles,
            )
        return (
            row_tiles,
            1 if use_compact_worklist else batch_size,
            effective_num_heads,
        )

    kernel = flash_varlen_fwd_kernel[grid]
    args = tuple(getattr(params, k) for k in params.__slots__)

    # We assess which phase the requests are likely to be in and set the config accordingly.
    total_rows = total_q * num_heads
    num_sms = torch_device_fn.get_device_properties(
        flaggems_vllm.device
    ).multi_processor_count
    avg_rows_per_sm = total_rows / num_sms
    avg_rows_per_batch = total_q / batch_size
    avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
    # Heuristic: if avg_rows_per_sm >= 128, we are likely in prefill phase.
    # This is a rough heuristic and may not be accurate for all scenarios.
    if avg_rows_per_cta > 64:
        varlen_fwd_config_str = "mha_block_128"
    elif avg_rows_per_cta > 32:
        varlen_fwd_config_str = "mha_block_64"
    elif avg_rows_per_cta > 16:
        varlen_fwd_config_str = "mha_block_32"
    else:
        varlen_fwd_config_str = "mha_block_16"
    if flaggems_vllm.vendor_name == "mthreads":
        varlen_fwd_config_str = "mha_block_32"

    cfg = runtime.get_heuristic_config(varlen_fwd_config_str)
    use_c550_paged_d128_block128 = (
        is_paged and head_size == 128 and varlen_fwd_config_str == "mha_block_128"
    )
    # Paged D256 needs 100 KiB with the block-64/128 three-stage configs.
    # Keep the C550 launch below its 64 KiB per-CTA shared-memory limit.
    use_c550_paged_d256_block64 = (
        is_paged
        and head_size == 256
        and varlen_fwd_config_str in ("mha_block_64", "mha_block_128")
    )
    cfg_params = {
        "BLOCK_M": (
            worklist_block_m
            if use_compact_worklist
            else (64 if use_c550_paged_d256_block64 else cfg["BLOCK_M"](args))
        ),
        "BLOCK_N": (16 if use_c550_paged_d256_block64 else cfg["BLOCK_N"](args)),
        "BLOCK_K": triton.next_power_of_2(head_size),
        "COMPACT_WORKLIST": use_compact_worklist,
        "GRID_ORDER": grid_order_code,
        "num_warps": (8 if use_c550_paged_d128_block128 else cfg["num_warps"](args)),
        "num_stages": (
            1
            if not is_paged or use_c550_paged_d256_block64
            else cfg["num_stages"](args)
        ),
    }

    logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)
    worklist_ptr = compact_worklist if use_compact_worklist else params.page_table_ptr
    return kernel(*args, worklist_ptr, **cfg_params)


__all__ = ["flash_varlen_fwd_kernel", "launch_direct"]

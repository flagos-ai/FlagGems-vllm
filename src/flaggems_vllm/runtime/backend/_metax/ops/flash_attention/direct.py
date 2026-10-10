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

import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_kernel import (
    apply_alibi,
    apply_dropout,
    apply_mask,
    apply_softcap,
    load_from_kvcache,
    softmax_rescale,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.common import (
    compact_ragged_tile_coords,
    online_softmax_stats,
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

    total_rows = total_q * num_heads
    num_sms = torch_device_fn.get_device_properties(
        flaggems_vllm.device
    ).multi_processor_count
    avg_rows_per_sm = total_rows / num_sms
    avg_rows_per_batch = total_q / batch_size
    avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
    # Choose the Q tile from the average rows per request and processor.
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


@triton.jit
def flash_varlen_int8_fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USED,
    TABLE,
    QS,
    KS,
    VS,
    ALIBI,
    sq: tl.constexpr,
    hq: tl.constexpr,
    sk: tl.constexpr,
    hk: tl.constexpr,
    pk: tl.constexpr,
    sv: tl.constexpr,
    hv: tl.constexpr,
    pv: tl.constexpr,
    so: tl.constexpr,
    ho: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    table_stride: tl.constexpr,
    alibi_stride: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    CAUSAL: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    CAP: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    OUT_SPLIT_STRIDE: tl.constexpr,
    LSE_SPLIT_STRIDE: tl.constexpr,
    LOGICAL_KV: tl.constexpr,
    BATCH: tl.constexpr,
    COMPACT: tl.constexpr,
    FOLD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    program_tile = tl.program_id(0)
    (tile, batch, head) = (
        program_tile // KV_SPLITS,
        tl.program_id(1),
        tl.program_id(2),
    )
    split_id = program_tile % KV_SPLITS
    kv_head = head if FOLD else head // GROUP
    query_tile: tl.constexpr = BM // GROUP if FOLD else BM
    active = tl.full((), True, tl.int1)
    if COMPACT:
        tile, batch, active = compact_ragged_tile_coords(tile, CUQ, BATCH, query_tile)
    q_start = tl.load(CUQ + batch)
    nq = tl.load(CUQ + batch + 1) - q_start
    if PAGED:
        k_start = 0
        nk = tl.load(USED + batch)
    else:
        k_start = tl.load(CUK + batch)
        nk = tl.load(CUK + batch + 1) - k_start
    rows = tile * BM + tl.arange(0, BM)
    m = rows // GROUP if FOLD else rows
    h = kv_head * GROUP + rows % GROUP if FOLD else tl.full((BM,), head, tl.int32)
    d = tl.arange(0, D)
    if active & (tile * query_tile < nq):
        q = tl.load(
            Q + (q_start + m[:, None]) * sq + h[:, None] * hq + d[None, :],
            (m[:, None] < nq) & (d[None, :] < HEAD_DIM),
            0,
        )
        q_scale = tl.load(QS + batch * qs0 + h * qs1 + m // 128 * qs2, m < nq, 0)
        if not COMPACT and CAP <= 0 and (not HAS_ALIBI):
            q_scale = q_scale * (SCALE * 1.4426950408889634)
        maximum = tl.full((BM,), float("-inf"), tl.float32)
        denom = tl.full((BM,), 0, tl.float32)
        acc = tl.full((BM, D), 0, tl.float32)
        if HAS_ALIBI:
            slope = tl.load(ALIBI + batch * alibi_stride + h)
        first = 0
        end = nk
        if LEFT >= 0:
            first = tl.maximum(0, tile * query_tile + nk - nq - LEFT) // BN
        if CAUSAL:
            end = tl.minimum(end, (tile + 1) * query_tile + nk - nq)
        if RIGHT >= 0:
            end = tl.minimum(end, (tile + 1) * query_tile + nk - nq + RIGHT)
        use_dense: tl.constexpr = LEFT < 0 and RIGHT < 0
        end_block = tl.cdiv(tl.maximum(end, 0), BN)
        if use_dense:
            full_keys = nk
            if CAUSAL:
                full_keys = tl.minimum(nk, tile * query_tile + nk - nq + 1)
            full_hi = tl.maximum(full_keys, 0) // BN
        for mask_phase in tl.static_range(2 if use_dense else 1):
            if use_dense:
                begin_block = first if mask_phase == 0 else full_hi
                stop_block = full_hi if mask_phase == 0 else end_block
            else:
                (begin_block, stop_block) = (first, end_block)
            if KV_SPLITS > 1:
                blocks_per_split = tl.cdiv(tl.cdiv(nk, BN), KV_SPLITS)
                begin_block = tl.maximum(begin_block, split_id * blocks_per_split)
                stop_block = tl.minimum(stop_block, (split_id + 1) * blocks_per_split)
            for start in range(begin_block, stop_block):
                n = start * BN + tl.arange(0, BN)
                if LOGICAL_KV:
                    k_row = batch * pk + n * sk
                    v_row = batch * pv + n * sv
                elif PAGED:
                    if not use_dense or mask_phase == 1:
                        page = tl.load(
                            TABLE + batch * table_stride + n // PAGE, n < nk, 0
                        )
                    else:
                        page = tl.load(TABLE + batch * table_stride + n // PAGE)
                    k_row = page * pk + n % PAGE * sk
                    v_row = page * pv + n % PAGE * sv
                else:
                    k_row = (k_start + n) * sk
                    v_row = (k_start + n) * sv
                if not use_dense or mask_phase == 1:
                    k = tl.load(
                        K + k_row[None, :] + kv_head * hk + d[:, None],
                        (n[None, :] < nk) & (d[:, None] < HEAD_DIM),
                        0,
                    )
                else:
                    k = tl.load(
                        K + k_row[None, :] + kv_head * hk + d[:, None],
                        d[:, None] < HEAD_DIM,
                        0,
                    )
                descale_block = start * BN // 128
                ks = tl.load(KS + batch * ks0 + kv_head * ks1 + descale_block * ks2)
                vs = tl.load(VS + batch * vs0 + kv_head * vs1 + descale_block * vs2)
                scores = tl.trans(
                    tl.dot(tl.trans(k), tl.trans(q), out_dtype=tl.int32)
                ).to(tl.float32)
                if not COMPACT and CAP <= 0 and (not HAS_ALIBI):
                    scores = scores * (q_scale * ks)[:, None]
                else:
                    scores = scores * (q_scale * ks * SCALE)[:, None]
                if CAP > 0:
                    scores = CAP * (2 / (1 + tl.exp(2 * (-scores / CAP))) - 1)
                position = m + nk - nq
                if HAS_ALIBI:
                    scores -= slope[:, None] * tl.abs(position[:, None] - n[None, :])
                if not use_dense or mask_phase == 1:
                    valid = n[None, :] < nk
                    if CAUSAL:
                        valid &= n[None, :] <= position[:, None]
                    if LEFT >= 0:
                        valid &= n[None, :] >= position[:, None] - LEFT
                    if RIGHT >= 0:
                        valid &= n[None, :] <= position[:, None] + RIGHT
                if not (not COMPACT and CAP <= 0 and (not HAS_ALIBI)):
                    scores = scores * 1.4426950408889634
                if not use_dense or mask_phase == 1:
                    scores = tl.where(valid, scores, float("-inf"))
                alpha, probabilities, maximum, denom = online_softmax_stats(
                    scores, maximum, denom, 1.0, IS_BORDER=True
                )
                if not use_dense or mask_phase == 1:
                    v = tl.load(
                        V + v_row[:, None] + kv_head * hv + d[None, :],
                        (n[:, None] < nk) & (d[None, :] < HEAD_DIM),
                        0,
                    )
                else:
                    v = tl.load(
                        V + v_row[:, None] + kv_head * hv + d[None, :],
                        d[None, :] < HEAD_DIM,
                        0,
                    )
                if vs2 == 0:
                    acc = tl.trans(
                        tl.dot(
                            tl.trans(v.to(tl.float16)),
                            tl.trans(probabilities.to(tl.float16)),
                            tl.trans(acc * alpha[:, None]),
                        )
                    )
                else:
                    partial = tl.trans(
                        tl.dot(
                            tl.trans(v.to(tl.float16)),
                            tl.trans(probabilities.to(tl.float16)),
                            out_dtype=tl.float32,
                        )
                    )
                    acc = acc * alpha[:, None] + partial * vs
        if KV_SPLITS > 1:
            result = acc
        else:
            result = acc / tl.where(denom > 0, denom, 1)[:, None]
        if vs2 == 0:
            v_scale = tl.load(VS + batch * vs0 + kv_head * vs1, nk > 0, 0)
            result *= v_scale
        tl.store(
            O
            + split_id * OUT_SPLIT_STRIDE
            + (q_start + m[:, None]) * so
            + h[:, None] * ho
            + d[None, :],
            result,
            (m[:, None] < nq) & (d[None, :] < HEAD_DIM),
        )
        if WRITE_LSE:
            if KV_SPLITS > 1:
                lse_address = (
                    LSE + split_id * LSE_SPLIT_STRIDE + h * TOTAL_Q + q_start + m
                )
                tl.store(lse_address, maximum, m < nq)
                tl.store(lse_address + LSE_SPLIT_STRIDE // 2, denom, m < nq)
            else:
                lse = tl.where(
                    denom > 0,
                    maximum * 0.6931471805599453 + tl.log(denom),
                    float("inf"),
                )
                tl.store(LSE + h * TOTAL_Q + q_start + m, lse, m < nq)

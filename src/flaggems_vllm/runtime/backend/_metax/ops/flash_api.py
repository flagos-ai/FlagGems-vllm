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
import math

import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

from .attention_impl import MetaXAttentionScheduler, launch_metax_attention
from .attention_impl.tensor_utils import (
    copy_tensor,
    fill_tensor,
    reshape_view_or_copy,
    view_tensor,
)

logger = logging.getLogger(__name__)


def CHECK_DEVICE(x):
    assert x.device.type == flaggems_vllm.device


class fwd_params:
    __slots__ = (
        # pointers and strides
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "o_ptr",
        "p_ptr",
        "softmax_lse_ptr",
        "q_row_stride",
        "k_row_stride",
        "v_row_stride",
        "q_head_stride",
        "k_head_stride",
        "v_head_stride",
        "o_row_stride",
        "o_head_stride",
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "is_cu_seqlens_q",
        "cu_seqlens_q_ptr",
        "is_cu_seqlens_k",
        "cu_seqlens_k_ptr",
        "is_seqused_k",
        "seqused_k_ptr",
        # sizes
        "b",
        "bk",
        "h",
        "hk",
        "h_hk_ratio",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "d",
        "d_rounded",
        # scaling factors
        "is_softcap",
        "softcap",
        "scale_softmax",
        "scale_softmax_log2",
        # dropout
        "is_dropout",
        "p_dropout",
        "rp_dropout",
        "p_dropout_in_uint8_t",
        "philox_args",
        "return_softmax",
        # masking
        "is_causal",
        "is_local",
        "window_size_left",
        "window_size_right",
        "seqlenq_ngroups_swapped",
        "is_paged",
        # alibi
        "is_alibi",
        "alibi_slopes_ptr",
        "alibi_slopes_batch_stride",
        # block table
        "total_q",
        "page_table_ptr",
        "page_table_batch_stride",
        "block_size",
        "k_page_stride",
    )

    def __init__(
        self,
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
        is_cu_seqlens_q,
        cu_seqlens_q_ptr,
        is_cu_seqlens_k,
        cu_seqlens_k_ptr,
        is_seqused_k,
        seqused_k_ptr,
        # sizes
        b,
        bk,
        h,
        hk,
        h_hk_ratio,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        d,
        d_rounded,
        # scaling factors
        is_softcap,
        softcap,
        scale_softmax,
        scale_softmax_log2,
        # dropout
        is_dropout,
        p_dropout,
        rp_dropout,
        p_dropout_in_uint8_t,
        philox_args,
        return_softmax,
        # masking
        is_causal,
        is_local,
        window_size_left,
        window_size_right,
        seqlenq_ngroups_swapped,
        is_paged,
        # alibi
        is_alibi,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride,
        # block table
        total_q,
        page_table_ptr,
        page_table_batch_stride,
        block_size,
        k_page_stride,
    ):
        self.q_ptr = q_ptr
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr
        self.o_ptr = o_ptr
        self.p_ptr = p_ptr
        self.softmax_lse_ptr = softmax_lse_ptr
        self.q_row_stride = q_row_stride
        self.k_row_stride = k_row_stride
        self.v_row_stride = v_row_stride
        self.q_head_stride = q_head_stride
        self.k_head_stride = k_head_stride
        self.v_head_stride = v_head_stride
        self.o_row_stride = o_row_stride
        self.o_head_stride = o_head_stride
        self.q_batch_stride = q_batch_stride
        self.k_batch_stride = k_batch_stride
        self.v_batch_stride = v_batch_stride
        self.o_batch_stride = o_batch_stride
        self.is_cu_seqlens_q = is_cu_seqlens_q
        self.cu_seqlens_q_ptr = cu_seqlens_q_ptr
        self.is_cu_seqlens_k = is_cu_seqlens_k
        self.cu_seqlens_k_ptr = cu_seqlens_k_ptr
        self.is_seqused_k = is_seqused_k
        self.seqused_k_ptr = seqused_k_ptr
        # sizes
        self.b = b
        self.bk = bk
        self.h = h
        self.hk = hk
        self.h_hk_ratio = h_hk_ratio
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.seqlen_q_rounded = seqlen_q_rounded
        self.seqlen_k_rounded = seqlen_k_rounded
        self.d = d
        self.d_rounded = d_rounded
        # scaling factors
        self.is_softcap = is_softcap
        self.softcap = softcap
        self.scale_softmax = scale_softmax
        self.scale_softmax_log2 = scale_softmax_log2
        # dropout
        self.is_dropout = is_dropout
        self.p_dropout = p_dropout
        self.rp_dropout = rp_dropout
        self.p_dropout_in_uint8_t = p_dropout_in_uint8_t
        self.philox_args = philox_args
        self.return_softmax = return_softmax
        # masking
        self.is_causal = is_causal
        self.is_local = is_local
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped
        self.is_paged = is_paged
        # alibi
        self.is_alibi = is_alibi
        self.alibi_slopes_ptr = alibi_slopes_ptr
        self.alibi_slopes_batch_stride = alibi_slopes_batch_stride
        # block table
        self.total_q = total_q
        self.page_table_ptr = page_table_ptr
        self.page_table_batch_stride = page_table_batch_stride
        self.block_size = block_size
        self.k_page_stride = k_page_stride

    def args(self):
        return tuple(getattr(self, k) for k in self.__slots__)


def mha_varlan_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
    num_splits=0,
):
    if p_dropout != 0 or return_softmax:
        raise NotImplementedError(
            "MetaX varlen attention supports inference without dropout"
        )
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    q_dtype = q.dtype
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FlashAttention only support fp16 and bf16 data type"
    assert q_dtype == k.dtype
    assert q_dtype == v.dtype
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    # q shape: [total_q_tokens, num_heads, head_size]
    # k shape:
    #   paged_kv: [num_pages, block_size, num_heads_k, head_size]
    # batch_size, number of sentences
    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    # max_num_pages_per_seq = page_table.size(1)
    page_table_batch_stride = page_table.stride(0)
    k_batch_stride = k.stride(0)
    v_batch_stride = v.stride(0)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        assert out.dtype == q.dtype
        assert out.size() == (total_q, num_heads, head_size)

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_q:
        window_size_right = -1

    is_local = window_size_left >= 0 or (window_size_right >= 0 and not is_causal)
    if is_local:
        # Normalize an unbounded side for the local-mask arithmetic.
        if window_size_left < 0:
            window_size_left = max_seqlen_k
        if window_size_right < 0:
            window_size_right = max_seqlen_q

    # Swap query-group and sequence dimensions for dense single-query batches.
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and total_q == batch_size
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
        logger.debug("Swapping query groups and sequence dimensions")
        q = reshape_view_or_copy(
            reshape_view_or_copy(
                q, (batch_size, num_heads_k, q_groups, head_size)
            ).transpose(1, 2),
            (batch_size * q_groups, num_heads_k, head_size),
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None
        q_batch_stride = q.stride(0) * max_seqlen_q
        k_batch_stride = k.stride(0)
        v_batch_stride = v.stride(0)
        # o_batch_stride = out.stride(0) * max_seqlen_q
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0

    total_q = q.size(0)

    assert leftpad_k is None, "leftpad_k is not supported."
    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"

    assert q.shape == (total_q, num_heads, head_size)
    if is_paged:
        assert k.shape == (num_pages, block_size, num_heads_k, head_size)
        assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    if softcap > 0.0:
        assert p_dropout == 0, "dropout is not supported if softcap is used."

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    M_LOG2E = 1.4426950408889634074
    if softcap > 0.0:
        is_softcap = True
        adjusted_scale_softmax = softcap
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax_log2e = softcap * M_LOG2E
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

    # Set alibi params
    if alibi_slopes is not None:
        assert alibi_slopes.device == q_device
        assert alibi_slopes.dtype in (torch.float,)
        assert alibi_slopes.stride(-1) == 1
        assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
            batch_size,
            num_heads,
        )
        alibi_slopes_batch_stride = (
            alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
        )
        is_alibi = True
    else:
        alibi_slopes_batch_stride = 0
        is_alibi = False

    # Prepare params to kernel
    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q, dtype=v.dtype)
        else:
            out_ = None
            out = torch.empty_like(q, dtype=v.dtype)

        if seqlenq_ngroups_swapped:
            o_batch_stride = out.stride(0) * max_seqlen_q

        lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)

        is_dropout = False
        philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        p = torch.empty((), device=q_device)

        if zero_tensors:
            fill_tensor(out, 0.0)
            fill_tensor(lse, float("-inf"))

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q_batch_stride,  # q_batch_stride,
            k_batch_stride,  # k_batch_stride,
            v_batch_stride,  # v_batch_stride,
            o_batch_stride,  # o_batch_stride,
            cu_seqlens_q is not None,  # is_cu_seqlens_q,
            cu_seqlens_q,  # cu_seqlens_q_ptr,
            seqused_k is None,  # is_cu_seqlens_k,
            cu_seqlens_k,  # cu_seqlens_k_ptr,
            seqused_k is not None,  # is_seqused_k,
            seqused_k,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            k_batch_size,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            max_seqlen_q,  # seqlen_q,
            max_seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            total_q,  # total_q,
            page_table,  # page_table_ptr,
            page_table_batch_stride,  # page_table_batch_stride,
            block_size,  # block_size,
            k.stride(0) if is_paged else 0,  # k_page_stride,
        )

        logger.debug("kernel: flash_varlen_fwd")
        plan = MetaXAttentionScheduler.build(
            is_bfloat16=q_dtype is torch.bfloat16,
            is_paged=is_paged,
            block_size=block_size,
            is_cu_seqlens_q=cu_seqlens_q is not None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            total_q=total_q,
            batch_size=batch_size,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            head_size=head_size,
            is_causal=is_causal,
            is_local=is_local,
            is_dropout=is_dropout,
            is_alibi=is_alibi,
            is_softcap=is_softcap,
            seqlenq_ngroups_swapped=seqlenq_ngroups_swapped,
            num_splits=num_splits,
        )
        launch_metax_attention(
            plan,
            params,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            batch_size=batch_size,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            total_q=total_q,
            head_size=head_size,
            is_paged=is_paged,
        )

        if seqlenq_ngroups_swapped:
            out = reshape_view_or_copy(
                out, (batch_size, max_seqlen_q, num_heads_k, head_size)
            ).transpose(1, 2)
            if out_ is not None:
                copy_tensor(
                    out,
                    view_tensor(
                        out_, (batch_size, num_heads_k, max_seqlen_q, head_size)
                    ),
                )
                out = out_
            else:
                out = reshape_view_or_copy(
                    out, (batch_size, num_heads_k * max_seqlen_q, head_size)
                )
            lse = reshape_view_or_copy(
                view_tensor(lse, (num_heads_k, batch_size, max_seqlen_q)).permute(
                    0, 2, 1
                ),
                (num_heads_k * max_seqlen_q, batch_size),
            )

    return out, lse

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

"""D256 async-TN Split-KV adapter for short-Q paged attention."""
import logging

import torch
import triton

from .splitkv import flash_varlen_splitkv_merge_kernel
from .tn_addressing import tn_compile_scenario
from .tn_direct import flash_varlen_fwd_d256_tn_kernel

logger = logging.getLogger(__name__)


def launch_d256_tn_splitkv(
    params, *, max_seqlen_q, batch_size, num_heads, total_q, num_splits
):
    """Run BM32xBN128 PackGQA partials and FP32 merge."""
    if params.q_ptr.dtype != torch.bfloat16:
        raise ValueError("Async-TN Split-KV requires BF16 Q/K/V")
    if not params.is_paged or params.block_size != 32 or params.d != 256:
        raise ValueError("Async-TN Split-KV requires paged D256 with page size 32")
    if not params.is_causal or params.is_local:
        raise ValueError("Async-TN Split-KV only supports causal global attention")
    if params.is_dropout or params.is_alibi or params.is_softcap:
        raise ValueError(
            "Async-TN Split-KV does not support optional attention features"
        )
    if params.cu_seqlens_q_ptr is None:
        raise ValueError("Async-TN Split-KV requires varlen Q prefix sums")
    if params.cu_seqlens_k_ptr is None and params.seqused_k_ptr is None:
        raise ValueError("Async-TN Split-KV requires KV lengths")
    if not 2 <= num_splits <= 32:
        raise ValueError("Async-TN Split-KV requires 2 to 32 splits")
    if num_heads % params.h_hk_ratio != 0:
        raise ValueError("Async-TN Split-KV received an invalid GQA ratio")
    block_m = 32
    block_n = 128
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
    cfg = {
        "COMPACT_WORKLIST": False,
        "SPLIT_KV": True,
        "NUM_SPLITS": num_splits,
        "Q_TILES": q_tiles,
        "GRID_ORDER": 1,
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
    logger.debug("Running D256 async-TN Split-KV with config: %s", cfg)
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
        params.page_table_ptr,
        **cfg,
    )
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
        num_warps=1,
        num_stages=1,
    )
    return partial


__all__ = ["launch_d256_tn_splitkv"]

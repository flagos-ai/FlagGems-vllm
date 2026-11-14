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

from typing import Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

_SPARSE_PREFILL_TOPK_ALIGNMENT = 128

# GCU (gcu410) tuning for this kernel, measured on the core benchmark shapes
# (benchmark/test_deepseek_v4_attention_combine_topk_swa_indices.py, do_bench
# median, kernel mode): the machine exposes 24 SIPs, so at most 24 workers per
# request are useful; each worker processes a [TB, COMBINED_TOPK] row tile per
# iteration which amortizes the per-token load/store issue latency that made
# the one-token-per-iteration strided loop serial-latency bound.
_TOKEN_BLOCK = 16
_MAX_WORKERS_PER_REQ = 24
# <= 24 total programs (one per SIP) run best with num_warps=4; grids of ~48
# programs (two per SIP) prefer num_warps=2 (measured, see case log).
_FULL_MACHINE_PROGRAMS = 24


def _next_power_of_2_or_1(x: int) -> int:
    return 1 if x <= 1 else triton.next_power_of_2(x)


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_ptr,
    combined_stride,
    lens_ptr,
    topk_ptr,
    topk_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    COMBINED_TOPK: tl.constexpr,
    PADDED_COMBINED_TOPK: tl.constexpr,
    TB: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - (query_end - query_start)
    gather_start = seq_len - gather_len

    tok = tl.arange(0, TB)
    col = tl.arange(0, PADDED_COMBINED_TOPK)
    for blk in range(query_start + worker_idx * TB, query_end, num_workers * TB):
        token_idx = blk + tok
        m_row = token_idx < query_end
        token_in_query = token_idx - query_start
        pos = start_pos + token_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)
        valid_len = topk_len + swa_len

        # One fused store per row tile: topk values (masked load, only the
        # first topk_len columns fetch), sliding-window indices and the -1
        # padding tail are selected with a where chain, so the whole row is
        # written exactly once and the caller can allocate with torch.empty.
        row = combined_ptr + token_idx[:, None] * combined_stride
        m_topk = m_row[:, None] & (col[None, :] < topk_len[:, None])
        topk_vals = tl.load(
            topk_ptr + token_idx[:, None] * topk_stride + col[None, :],
            mask=m_topk,
            other=0,
        )
        swa_val = (
            M * batch_idx
            + N
            + (col[None, :] - topk_len[:, None])
            + pos[:, None]
            - swa_len[:, None]
            + 1
            - gather_start
        )
        val = tl.where(
            col[None, :] < topk_len[:, None],
            topk_vals + M * batch_idx,
            tl.where(col[None, :] < valid_len[:, None], swa_val, -1),
        )
        if PADDED_COMBINED_TOPK == COMBINED_TOPK:
            tl.store(row + col[None, :], val, mask=m_row[:, None])
        else:
            tl.store(
                row + col[None, :],
                val,
                mask=m_row[:, None] & (col[None, :] < COMBINED_TOPK),
            )
        tl.store(lens_ptr + token_idx, valid_len, mask=m_row)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.ndim == 2
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    # Allocate uninitialized: the kernel writes the -1 padding tail itself, so
    # torch.full's separate full-buffer memset (mostly overwritten) is avoided.
    combined = torch.empty(
        (num_tokens, combined_topk), device=topk_indices.device, dtype=torch.int32
    )
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)
    # Worker count from shape metadata only (no device readback): enough
    # workers to cover the average per-request token count in single tiles,
    # capped at the machine width. Skewed request lengths are handled by the
    # in-kernel grid-stride loop (long requests simply run more iterations).
    avg_tokens = (num_tokens + num_reqs - 1) // num_reqs if num_reqs > 0 else 0
    workers = (avg_tokens + _TOKEN_BLOCK - 1) // _TOKEN_BLOCK
    workers = max(1, min(workers, _MAX_WORKERS_PER_REQ))
    num_warps = 4 if num_reqs * workers <= _FULL_MACHINE_PROGRAMS else 2
    with torch_device_fn.device(topk_indices.device):
        _combine_topk_swa_indices_kernel[(num_reqs, workers)](
            combined,
            combined.stride(0),
            lens,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            M,
            N,
            TOP_K=topk,
            COMPRESS_RATIO=compress_ratio,
            WINDOW_SIZE=window_size,
            COMBINED_TOPK=combined_topk,
            PADDED_COMBINED_TOPK=_next_power_of_2_or_1(combined_topk),
            TB=_TOKEN_BLOCK,
            num_warps=num_warps,
        )
    return combined, lens


__all__ = ["combine_topk_swa_indices"]

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

"""Ascend-optimized deepseek_v4_attention_combine_topk_swa_indices."""

from typing import Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    MIN_TILE_ROWS,
    tensorview_call,
    tile_rows_for_ub,
)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store

_SPARSE_PREFILL_TOPK_ALIGNMENT = 128

_MAX_WORKERS = 128
_FILL_BLOCK = 4096


def _next_power_of_2_or_1(x: int) -> int:
    return 1 if x <= 1 else triton.next_power_of_2(x)


def _select_num_workers(num_tokens: int, num_reqs: int) -> int:
    # Metadata only: reading query_start_loc back to the host would sync the device.
    avg = -(-int(num_tokens) // max(int(num_reqs), 1))
    return max(1, min(_MAX_WORKERS, _next_power_of_2_or_1(avg)))


@triton.jit
def _fill_neg_one_kernel(ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, -1, mask=offs < numel)


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
    PADDED_TOP_K: tl.constexpr,
    PADDED_WINDOW_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_idx, query_end, num_workers):
        token_in_query = token_idx - query_start
        pos = start_pos + token_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        # The -1 padding tail was written by _fill_neg_one_kernel, so only
        # [0, valid_len) is stored here.
        offs = tl.arange(0, PADDED_TOP_K)
        mask = offs < topk_len
        topk_vals = tl.load(
            topk_ptr + token_idx * topk_stride + offs, mask=mask, other=-1
        )
        tl.store(
            combined_ptr + token_idx * combined_stride + offs,
            topk_vals + M * batch_idx,
            mask=mask,
        )

        swa_offs = tl.arange(0, PADDED_WINDOW_SIZE)
        tl.store(
            combined_ptr + token_idx * combined_stride + topk_len + swa_offs,
            M * batch_idx + N + swa_offs + pos - swa_len + 1 - gather_start,
            mask=(swa_offs < swa_len) & (swa_offs < WINDOW_SIZE),
        )
        tl.store(lens_ptr + token_idx, topk_len + swa_len)


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
    # Uninitialized: the fill kernel below writes every column the stores do not.
    combined = torch.empty(
        (num_tokens, combined_topk), device=topk_indices.device, dtype=torch.int32
    )
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)

    num_workers = _select_num_workers(num_tokens, num_reqs)
    with torch_device_fn.device(topk_indices.device):
        numel = num_tokens * combined_topk
        _fill_neg_one_kernel[(triton.cdiv(numel, _FILL_BLOCK),)](
            combined, numel, BLOCK=_FILL_BLOCK
        )
        _combine_topk_swa_indices_kernel[(num_reqs, num_workers)](
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
            PADDED_TOP_K=_next_power_of_2_or_1(topk_indices.shape[-1]),
            PADDED_WINDOW_SIZE=_next_power_of_2_or_1(window_size),
        )
    return combined, lens


_combine_topk_generic = combine_topk_swa_indices

if HAS_TENSORVIEW:
    _TV_MAX_ROWS = 32
    _TV_WORKERS = 128

    @triton.jit
    def _combine_topk_swa_tv_kernel(
        combined,
        lens,
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        num_tokens,
        offset_m,
        offset_n,
        TOPK: tl.constexpr,
        TOPK_COUNT: tl.constexpr,
        COMBINED_TOPK: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        WINDOW_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        worker = tl.program_id(1)
        num_workers = tl.num_programs(1)
        base = tl.load(query_start_loc)
        query_start = tl.load(query_start_loc + batch) - base
        query_end = tl.load(query_start_loc + batch + 1) - base
        seq_len = tl.load(seq_lens + batch)
        gather_len = tl.load(gather_lens + batch)
        start_pos = seq_len - (query_end - query_start)
        gather_start = seq_len - gather_len

        topk_view = tle_exp.make_strided_view(
            topk_indices,
            shape=[num_tokens, TOPK],
            strides=[TOPK, 1],
            tile=[BLOCK_T, BLOCK_C],
            traversal_strides=[1, BLOCK_C],
        )
        combined_view = tle_exp.make_strided_view(
            combined,
            shape=[query_end, COMBINED_TOPK],
            strides=[COMBINED_TOPK, 1],
            tile=[BLOCK_T, BLOCK_C],
            traversal_strides=[1, BLOCK_C],
        )
        lens_view = tle_exp.make_strided_view(
            lens,
            shape=[query_end],
            strides=[1],
            tile=[BLOCK_T],
            traversal_strides=[1],
        )
        col = tl.arange(0, BLOCK_C)[None, :]
        tiles = tl.cdiv(query_end - query_start, BLOCK_T)
        for tile in range(worker, tiles, num_workers):
            first_token = query_start + tile * BLOCK_T
            topk_vals = tile_load(
                topk_view,
                index=(first_token, 0),
                dst_space=tle_exp.language.dsa.ascend.UB,
            )
            pos = start_pos + tile * BLOCK_T + tl.arange(0, BLOCK_T)
            topk_count = tl.minimum((pos + 1) // COMPRESS_RATIO, TOPK_COUNT)
            swa_count = tl.minimum(pos + 1, WINDOW_SIZE)
            topk_len = topk_count[:, None]
            swa_len = swa_count[:, None]
            window_vals = (
                batch * offset_m
                + offset_n
                + (col - topk_len)
                + pos[:, None]
                - swa_len
                + 1
                - gather_start
            )
            row = tl.where(
                col < topk_len,
                topk_vals + batch * offset_m,
                tl.where(col < topk_len + swa_len, window_vals, -1),
            )
            tile_store(
                combined_view,
                value=row,
                index=(first_token, 0),
                src_space=tle_exp.language.dsa.ascend.UB,
            )
            tile_store(
                lens_view,
                value=topk_count + swa_count,
                index=(first_token,),
                src_space=tle_exp.language.dsa.ascend.UB,
            )

    def _combine_tv_width(topk, window_size):
        return (
            triton.cdiv(topk + window_size, _SPARSE_PREFILL_TOPK_ALIGNMENT)
            * _SPARSE_PREFILL_TOPK_ALIGNMENT
        )

    def _combine_tv_eligible(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
    ):
        if topk_indices.ndim != 2 or topk_indices.numel() == 0:
            return False
        if topk_indices.dtype != torch.int32 or not topk_indices.is_contiguous():
            return False
        if query_start_loc.ndim != 1 or seq_lens.ndim != 1 or gather_lens.ndim != 1:
            return False
        if not (
            query_start_loc.is_contiguous()
            and seq_lens.is_contiguous()
            and gather_lens.is_contiguous()
        ):
            return False
        num_reqs = seq_lens.numel()
        if num_reqs == 0 or gather_lens.numel() != num_reqs:
            return False
        if query_start_loc.numel() != num_reqs + 1:
            return False
        if topk_indices.shape[1] < topk or min(window_size, compress_ratio, topk) <= 0:
            return False
        if any(
            t.device != topk_indices.device
            for t in (query_start_loc, seq_lens, gather_lens)
        ):
            return False
        combined_topk = _combine_tv_width(topk, window_size)
        if combined_topk < topk_indices.shape[1]:
            return False
        block_c = triton.next_power_of_2(combined_topk)
        return tile_rows_for_ub(block_c * 4, 2, _TV_MAX_ROWS) >= MIN_TILE_ROWS

    def _combine_topk_swa_tensorview(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    ):
        num_tokens, topk_width = topk_indices.shape
        num_reqs = seq_lens.numel()
        combined_topk = _combine_tv_width(topk, window_size)
        block_c = triton.next_power_of_2(combined_topk)
        block_t = tile_rows_for_ub(block_c * 4, 2, _TV_MAX_ROWS)
        combined = torch.empty(
            (num_tokens, combined_topk), device=topk_indices.device, dtype=torch.int32
        )
        lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)
        with torch_device_fn.device(topk_indices.device):
            _combine_topk_swa_tv_kernel[(num_reqs, _TV_WORKERS)](
                combined,
                lens,
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                num_tokens,
                M,
                N,
                TOPK=topk_width,
                TOPK_COUNT=topk,
                COMBINED_TOPK=combined_topk,
                COMPRESS_RATIO=compress_ratio,
                WINDOW_SIZE=window_size,
                BLOCK_T=block_t,
                BLOCK_C=block_c,
            )
        return combined, lens

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
        args = (
            topk_indices,
            query_start_loc,
            seq_lens,
            gather_lens,
            window_size,
            compress_ratio,
            topk,
            M,
            N,
        )
        if not _combine_tv_eligible(*args[:7]):
            return _combine_topk_generic(*args)
        return tensorview_call(
            "COMBINE_TOPK_SWA_INDICES",
            _combine_topk_swa_tensorview,
            _combine_topk_generic,
            *args,
        )


__all__ = ["combine_topk_swa_indices"]

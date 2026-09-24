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

"""Hygon BW1000 top_k_per_row_prefill, routed per call.

The radix routes run this file's copy of the generic kernel, specialized by
constexprs; the two sampled routes run kernels of their own. Routes, their
gates and the environment switch are documented on `top_k_per_row_prefill`
at the bottom of this file.
"""

import functools
import os
import threading
from collections import OrderedDict
from importlib import import_module

import torch
import triton
import triton.language as tl

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

# FLAGGEMS_HYGON_TOPK_PREFILL=0 turns the override off: every call goes to the
# generic operator.
_ENABLED = os.environ.get("FLAGGEMS_HYGON_TOPK_PREFILL", "1").strip().lower() not in (
    "0",
    "false",
    "off",
    "no",
)

# Dense iff vocab_size <= DENSE_VOCAB_PER_TOPK * top_k, i.e. density >= 10%. A
# prefix sum over the take mask beats one atomic per selected element above a
# measured ~9.4% (48 of 512).
DENSE_VOCAB_PER_TOPK = 10


# ---------------------------------------------------------------------------
# The radix kernel, copied from the generic operator for the non-TLE,
# one-program-per-row case, with four changes selected by constexprs:
#
#   one threshold scan  each step clears its histogram in one store and finds
#                       the threshold bin with one RADIX_SIZE-wide cumsum,
#                       kept in registers, instead of a carried chain of
#                       rounds through scratch memory (geomean 1.09x).
#   DENSE               slots for the definitely-in elements come from a prefix
#                       sum over the take mask with the counter carried in a
#                       register through the step, not from one atomic per
#                       element.
#   VEC                 the vector width of the bulk loads; the dense route
#                       reads 2 floats per lane, the generic route 4.
#   SHORT               512 STEP-0 bins instead of 2048.
#   SKIP                a program returns at once unless its row is flagged:
#                       the one-read route's retry.
#
# The generic route (DENSE False) calls the generic operator's own
# _process_bins, so only the scan differs from upstream there.

_generic_process_bins = _generic._process_bins


@triton.jit
def _extract_bin_idx(x, in_range, pattern, STEP: tl.constexpr, SHORT: tl.constexpr):
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        if SHORT:
            bin_idx = (mapped >> 7).to(tl.uint32)
        else:
            bin_idx = (mapped >> 5).to(tl.uint32)
    else:
        bits = _key32(x)
        if STEP == 1:
            bin_idx = bits >> 21
        elif STEP == 2:
            bin_idx = (bits >> 10) & 0x7FF
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = bits & 0x3FF
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


@triton.jit
def _distribute_to_bins(
    logits,
    in_range,
    ones,
    logit_pattern,
    s_histogram_ptr,
    STEP: tl.constexpr,
    SHORT: tl.constexpr,
):
    bin_idx, is_partial_match = _extract_bin_idx(
        logits, in_range, logit_pattern, STEP=STEP, SHORT=SHORT
    )
    tl.atomic_add(
        s_histogram_ptr + bin_idx,
        ones,
        mask=is_partial_match,
        sem="relaxed",
        scope="cta",
    )


@triton.jit
def _dense_bins(
    logits,
    in_range,
    ones,
    offs,
    final_cnt_ptrs,
    logit_pattern,
    threshold_bin_idx,
    write_directly,
    use_final,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_out_indices_ptr,
    slot_base,
    STEP: tl.constexpr,
    TOPK: tl.constexpr,
    SHORT: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048

    bin_idx, is_partial_match = _extract_bin_idx(
        logits, in_range, logit_pattern, STEP=STEP, SHORT=SHORT
    )
    take_lt = is_partial_match & (bin_idx < threshold_bin_idx) & write_directly
    take_int = take_lt.to(tl.int32)
    flat_take = tl.reshape(take_int, (take_int.numel,))
    offsets = tl.cumsum(flat_take, axis=0) - flat_take
    out_pos_lt = slot_base + tl.reshape(offsets, take_int.shape)
    slot_base += tl.sum(flat_take, axis=0)
    tl.store(s_out_indices_ptr + out_pos_lt, offs.to(tl.int32), mask=take_lt)

    if STEP < 3:
        if use_final:
            take_eq_final = is_partial_match & (bin_idx == threshold_bin_idx)
            final_pos = tl.atomic_add(
                final_cnt_ptrs,
                ones,
                mask=take_eq_final,
                sem="relaxed",
                scope="cta",
            )
            keep = take_eq_final & (final_pos < NUM_FINAL_ITEMS)
            tl.store(s_final_logits_ptr + final_pos, logits, mask=keep)
            # s_histogram_ptr holds the indices for the final sort
            tl.store(s_histogram_ptr + final_pos, offs.to(tl.int32), mask=keep)
    else:
        take_eq = is_partial_match & (bin_idx == threshold_bin_idx)
        # s_histogram_ptr holds the exclusive prefix sum
        out_pos_eq = tl.atomic_add(
            s_histogram_ptr + bin_idx,
            ones,
            mask=take_eq,
            sem="relaxed",
            scope="cta",
        )
        tl.store(
            s_out_indices_ptr + out_pos_eq,
            offs.to(tl.int32),
            mask=take_eq & (out_pos_eq < TOPK),
        )
    return slot_base


@triton.jit
def _bins(
    logits,
    in_range,
    ones,
    offs,
    found_ptrs,
    final_cnt_ptrs,
    logit_pattern,
    threshold_bin_idx,
    write_directly,
    use_final,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_out_indices_ptr,
    slot_base,
    STEP: tl.constexpr,
    TOPK: tl.constexpr,
    DENSE: tl.constexpr,
    SHORT: tl.constexpr,
):
    if DENSE:
        slot_base = _dense_bins(
            logits,
            in_range,
            ones,
            offs,
            final_cnt_ptrs,
            logit_pattern,
            threshold_bin_idx,
            write_directly,
            use_final,
            s_histogram_ptr,
            s_final_logits_ptr,
            s_out_indices_ptr,
            slot_base,
            STEP=STEP,
            TOPK=TOPK,
            SHORT=SHORT,
        )
    else:
        _generic_process_bins(
            logits,
            in_range,
            ones,
            offs,
            found_ptrs,
            final_cnt_ptrs,
            logit_pattern,
            threshold_bin_idx,
            write_directly,
            use_final,
            0,
            None,
            s_histogram_ptr,
            s_final_logits_ptr,
            s_out_indices_ptr,
            None,
            STEP=STEP,
            TOPK=TOPK,
            MULTIPLE_BLOCKS_PER_ROW=False,
            MERGE_BLOCKS=False,
        )
    return slot_base


@triton.jit
def _histogram_step(
    logits_ptr,
    row_start,
    row_end,
    stride1,
    vocab_size,
    skip_elems,
    logit_pattern,
    threshold_bin_idx,
    assume_aligned,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_out_indices_ptr,
    STEP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
    DENSE: tl.constexpr,
    SHORT: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX_SIZE: tl.constexpr = (
        1024 if STEP == 3 else ((512 if STEP == 0 else 2048) if SHORT else 2048)
    )

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)

    radix_bins = tl.arange(0, RADIX_SIZE)
    tl.store(s_histogram_ptr + radix_bins, tl.zeros([RADIX_SIZE], tl.int32))
    tl.debug_barrier()

    if STEP == 2:
        logit_pattern = (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 21
    elif STEP == 3:
        logit_pattern |= (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 10

    if assume_aligned:
        n_vec_full = vocab_size // (BLOCK_SIZE * VEC)
        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(logits_ptr + offs)
            _distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
                SHORT=SHORT,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(logits_ptr + offs)
            _distribute_to_bins(
                x, True, ones, logit_pattern, s_histogram_ptr, STEP=STEP, SHORT=SHORT
            )
    elif stride1 == 1:
        aligned_row_ptr = tl.multiple_of(logits_ptr + row_start + skip_elems, VEC * 4)
        row_len = row_end - row_start - skip_elems
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(aligned_row_ptr + offs)
            _distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
                SHORT=SHORT,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(aligned_row_ptr + offs)
            _distribute_to_bins(
                x, True, ones, logit_pattern, s_histogram_ptr, STEP=STEP, SHORT=SHORT
            )
        if skip_elems > 0:
            in_range = lane < skip_elems
            x = tl.load(
                logits_ptr + row_start + lane, mask=in_range, other=float("-inf")
            )
            _distribute_to_bins(
                x,
                in_range,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
                SHORT=SHORT,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(aligned_row_ptr + offs, mask=in_range, other=float("-inf"))
            _distribute_to_bins(
                x,
                in_range,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
                SHORT=SHORT,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                logits_ptr + row_start + offs * stride1,
                mask=in_range,
                other=float("-inf"),
            )
            _distribute_to_bins(
                x,
                in_range,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
                SHORT=SHORT,
            )
    last_value = tl.load(s_found_topk_values_ptr)
    tl.debug_barrier()

    # The threshold bin: the one where the running count crosses TOPK.
    counts = tl.load(s_histogram_ptr + radix_bins)
    incl = last_value + tl.cumsum(counts, axis=0)
    prefix_sum = incl - counts
    threshold_mask = (prefix_sum < TOPK) & (incl >= TOPK)
    threshold_bin_idx = tl.min(
        tl.where(threshold_mask, radix_bins, RADIX_SIZE), axis=0
    ).to(tl.int32)
    final_bin_size = tl.max(tl.where(threshold_mask, counts, 0), axis=0)
    if STEP == 3:
        tl.store(s_histogram_ptr + radix_bins, prefix_sum)
        tl.debug_barrier()
    use_final = final_bin_size <= NUM_FINAL_ITEMS
    write_directly = ((STEP == 0) & (final_bin_size <= NUM_FINAL_ITEMS)) | (STEP >= 1)

    found_ptrs = s_found_topk_values_ptr + zeros
    final_cnt_ptrs = s_final_cnt_ptr + zeros
    found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
    final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
    if DENSE:
        slot_base = tl.load(s_found_topk_values_ptr)
    else:
        slot_base = tl.zeros((), dtype=tl.int32)
    if assume_aligned:
        n_vec_full = vocab_size // (BLOCK_SIZE * VEC)
        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(logits_ptr + offs)
            slot_base = _bins(
                x_vec,
                True,
                ones_vec_2d,
                offs,
                found_ptrs_vec_2d,
                final_cnt_ptrs_vec_2d,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                slot_base,
                STEP=STEP,
                TOPK=TOPK,
                DENSE=DENSE,
                SHORT=SHORT,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(logits_ptr + offs)
            slot_base = _bins(
                x,
                True,
                ones,
                offs,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                slot_base,
                STEP=STEP,
                TOPK=TOPK,
                DENSE=DENSE,
                SHORT=SHORT,
            )
    elif stride1 == 1:
        aligned_row_ptr = tl.multiple_of(logits_ptr + row_start + skip_elems, VEC * 4)
        row_len = row_end - row_start - skip_elems
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(aligned_row_ptr + offs)
            slot_base = _bins(
                x_vec,
                True,
                ones_vec_2d,
                offs + skip_elems,
                found_ptrs_vec_2d,
                final_cnt_ptrs_vec_2d,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                slot_base,
                STEP=STEP,
                TOPK=TOPK,
                DENSE=DENSE,
                SHORT=SHORT,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(aligned_row_ptr + offs)
            slot_base = _bins(
                x,
                True,
                ones,
                offs + skip_elems,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                slot_base,
                STEP=STEP,
                TOPK=TOPK,
                DENSE=DENSE,
                SHORT=SHORT,
            )
        if skip_elems > 0:
            in_range = lane < skip_elems
            x = tl.load(
                logits_ptr + row_start + lane, mask=in_range, other=float("-inf")
            )
            slot_base = _bins(
                x,
                in_range,
                ones,
                lane,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                slot_base,
                STEP=STEP,
                TOPK=TOPK,
                DENSE=DENSE,
                SHORT=SHORT,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(aligned_row_ptr + offs, mask=in_range, other=float("-inf"))
            slot_base = _bins(
                x,
                in_range,
                ones,
                offs + skip_elems,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                slot_base,
                STEP=STEP,
                TOPK=TOPK,
                DENSE=DENSE,
                SHORT=SHORT,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                logits_ptr + row_start + offs * stride1,
                mask=in_range,
                other=float("-inf"),
            )
            slot_base = _bins(
                x,
                in_range,
                ones,
                offs,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                slot_base,
                STEP=STEP,
                TOPK=TOPK,
                DENSE=DENSE,
                SHORT=SHORT,
            )
    if DENSE:
        tl.store(s_found_topk_values_ptr, slot_base)
    tl.debug_barrier()
    return final_bin_size > NUM_FINAL_ITEMS, logit_pattern, threshold_bin_idx


@triton.jit
def _radix_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    vocab_size,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    skip_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
    DENSE: tl.constexpr,
    SHORT: tl.constexpr,
    SKIP: tl.constexpr,
):
    NUM_BINS: tl.constexpr = 2048
    NUM_FINAL_ITEMS: tl.constexpr = 2048

    row_id = tl.program_id(0)
    if SKIP:
        if tl.load(skip_ptr + row_id) == 0:
            return
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    logits_ptr += row_id * stride0
    # float4 align
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
    out_indices_ptr += row_id * TOPK
    s_histogram_ptr += row_id * NUM_BINS
    s_final_logits_ptr += row_id * NUM_FINAL_ITEMS
    s_final_cnt_ptr += row_id
    s_found_topk_values_ptr += row_id

    assume_aligned = (
        (row_start == 0)
        & (row_end == vocab_size)
        & (stride1 == 1)
        & ((vocab_size % BLOCK_SIZE) == 0)
    )
    if assume_aligned:
        tl.assume(row_start == 0)
        tl.assume(row_end == vocab_size)
        tl.assume(stride1 == 1)
        vocab_size = tl.multiple_of(vocab_size, BLOCK_SIZE)
    elif stride1 == 1:
        tl.assume(stride1 == 1)

    lane = tl.arange(0, BLOCK_SIZE)
    row_len = row_end - row_start
    if row_len <= TOPK:
        chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for chunk_idx in tl.range(0, chunks):
            pos = chunk_idx * BLOCK_SIZE + lane
            tl.store(out_indices_ptr + pos, pos.to(tl.int32), mask=pos < row_len)
            tl.store(out_indices_ptr + pos, -1, mask=(pos >= row_len) & (pos < TOPK))
        return
    tl.store(s_final_cnt_ptr, 0)
    tl.store(s_found_topk_values_ptr, 0)
    tl.debug_barrier()
    logit_pattern = tl.zeros((), dtype=tl.uint32)
    continue_to_next_step = tl.full((), True, dtype=tl.int1)
    threshold_bin_idx = tl.full((), -1, dtype=tl.int32)
    for step_idx in tl.static_range(0, 4):
        if continue_to_next_step:
            (
                continue_to_next_step,
                logit_pattern,
                threshold_bin_idx,
            ) = _histogram_step(
                logits_ptr,
                row_start,
                row_end,
                stride1,
                vocab_size,
                skip_elems,
                logit_pattern,
                threshold_bin_idx,
                assume_aligned,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                out_indices_ptr,
                STEP=step_idx,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                VEC=VEC,
                DENSE=DENSE,
                SHORT=SHORT,
            )

    if not continue_to_next_step:
        base_idx = tl.load(s_found_topk_values_ptr)
        final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), NUM_FINAL_ITEMS)
        sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
        for sort_chunk in tl.range(0, sort_chunks):
            pos = sort_chunk * BLOCK_SIZE + lane
            valid = pos < final_cnt
            logit_i = tl.load(s_final_logits_ptr + pos, mask=valid, other=0)
            out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            for j in tl.range(0, final_cnt):
                logit_j = tl.load(s_final_logits_ptr + j)
                better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                out_rank = out_rank + (valid & better).to(tl.int32)
            dst_pos = base_idx + out_rank
            take = valid & (dst_pos < TOPK)
            idx_i = tl.load(s_histogram_ptr + pos, mask=take, other=0)
            tl.store(out_indices_ptr + dst_pos, idx_i, mask=take)
        tl.debug_barrier()


# ---------------------------------------------------------------------------
# Launch geometry by occupancy: past one row per SM the grid is the
# parallelism, so many rows want narrower programs -- up to 1.95x at about 200
# rows per SM. num_warps=1 returned wrong answers on the generic kernel, so
# this never picks it.

SHORT_ROW_MAX = 8192
# 512 STEP-0 bins beat 2048 for top_k 512 on rows up to 1536 long; at 1792
# they were already neutral, and rows of 4095 and 5115 regressed.
SHORT_BINS_TOPK = 512
SHORT_BINS_MAX_VOCAB = 1536


@functools.lru_cache(maxsize=1)
def _sm_count():
    try:
        import torch

        props = torch.cuda.get_device_properties(0)
        return int(getattr(props, "multi_processor_count", 0)) or 80
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return 80


def _geometry(num_rows, row_len):
    """(BLOCK_SIZE, num_warps) for this call; below 4 rows per SM, generic's."""
    sms = _sm_count()
    if num_rows < 4 * sms:
        block = _generic.NUM_THREADS_PER_BLOCK
        return block, _generic._num_warps(block)
    if num_rows < 32 * sms:
        return 512, 4
    return (256, 2) if row_len <= SHORT_ROW_MAX else (256, 4)


# The generic host wrapper allocates its scratch tensors on every call. On
# BW1000, reusing one plan per module/device/row-count removes 4-52% of wall
# time on the benchmark's small-row and four-row shapes. Keep only the most
# recently used row-count for each loaded route and cap live storage so a
# serving process cannot accumulate one full scratch set for every request
# shape.
_SCRATCH_LOCK = threading.Lock()
_SCRATCH_CACHE = OrderedDict()
_SCRATCH_CACHE_BYTES = 0
_SCRATCH_CACHE_LIMIT = 512 * 1024 * 1024


def _scratch_buffers(route, device, num_rows):
    """Return the scratch set, reusing one active shape per route. The caller
    holds _SCRATCH_LOCK."""
    global _SCRATCH_CACHE_BYTES

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = torch.cuda.current_stream(device).cuda_stream
    key = (route, device.type, device_index, stream_id)
    num_bins = int(_generic.NUM_BINS)
    num_final_items = int(_generic.NUM_FILNAL_ITEMS)
    cached = _SCRATCH_CACHE.get(key)
    if cached is not None:
        cached_rows, cached_bins, cached_final, _, buffers = cached
        if (
            cached_rows == num_rows
            and cached_bins == num_bins
            and cached_final == num_final_items
        ):
            _SCRATCH_CACHE.move_to_end(key)
            return buffers
        del _SCRATCH_CACHE[key]
        _SCRATCH_CACHE_BYTES -= cached[3]

    allocation_bytes = (
        num_rows * num_bins * 4 + num_rows * num_final_items * 4 + num_rows * 2 * 4
    )
    while (
        _SCRATCH_CACHE
        and _SCRATCH_CACHE_BYTES + allocation_bytes > _SCRATCH_CACHE_LIMIT
    ):
        _, old = _SCRATCH_CACHE.popitem(last=False)
        _SCRATCH_CACHE_BYTES -= old[3]

    buffers = (
        torch.empty((num_rows, num_bins), device=device, dtype=torch.int32),
        torch.empty((num_rows, num_final_items), device=device, dtype=torch.float32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
    )
    if allocation_bytes <= _SCRATCH_CACHE_LIMIT:
        _SCRATCH_CACHE[key] = (
            num_rows,
            num_bins,
            num_final_items,
            allocation_bytes,
            buffers,
        )
        _SCRATCH_CACHE_BYTES += allocation_bytes
    return buffers


# (VEC, DENSE, SHORT) of the radix kernel for each route.
_GENERIC_ROUTE = (4, False, False)
_DENSE_ROUTE = (2, True, False)
_SHORT_ROUTE = (2, True, True)


def _route(vocab, top_k):
    if vocab <= DENSE_VOCAB_PER_TOPK * top_k:
        if top_k == SHORT_BINS_TOPK and vocab <= SHORT_BINS_MAX_VOCAB:
            return _SHORT_ROUTE
        return _DENSE_ROUTE
    return _GENERIC_ROUTE


def _launch_radix(
    logits,
    row_starts,
    row_ends,
    indices,
    num_rows,
    stride0,
    stride1,
    top_k,
    route,
    skip=None,
):
    """The radix kernel for one route. With `skip`, only rows whose flag is
    set run."""
    block, warps = _geometry(num_rows, logits.shape[1])
    vec, dense, short = route
    with _SCRATCH_LOCK:
        scratch = _scratch_buffers((route, skip is not None), logits.device, num_rows)
        _radix_prefill[(num_rows,)](
            logits,
            indices,
            row_starts,
            row_ends,
            stride0,
            stride1,
            logits.shape[1],
            *scratch,
            row_starts if skip is None else skip,
            TOPK=top_k,
            BLOCK_SIZE=block,
            VEC=vec,
            DENSE=dense,
            SHORT=short,
            SKIP=skip is not None,
            num_warps=warps,
        )
    return indices


# ---------------------------------------------------------------------------
# A sampled threshold for very sparse rows. The generic step reads each row
# twice, the first time only to find the threshold; here that pass reads every
# SSTRIDE-th tile and aims at TARGET_MULT * top_k, so one pass collects a
# superset that _s_finish ranks exactly. Below TARGET_MULT 1.25 the estimate
# falls short of top_k often enough (28% of rows at 1.0) that the redo
# dominates; above it the larger candidate set costs more than it saves.
SAMPLED_MIN_VOCAB_PER_TOPK = 64
SSTRIDE = 16
TARGET_MULT = 1.25
CAP_MULT = 4  # candidate buffer; the acceptance window is [top_k, CAP]
SBLOCK = 512
SWARPS = 8
SRADIX = 256
# Programs per row in the collect pass, each with its own counter and segment:
# a partially masked atomic to one address costs ~12 ns per taken lane on this
# card, serialized, so one counter shared by the row queued all its programs.
# 4 measured best of 2 to 16. A power of two, because prepare zeroes a row's
# counters with one arange.
SSPLIT = 4
_MAX_CAND_ELEMS = 1 << 24


def _s_geometry(vocab, top_k):
    """(CAP, CHUNK, SEG) for a sampled plan. A segment holds min(CAP, CHUNK),
    so it can overflow only when the row already exceeds CAP; CHUNK's 2048
    granularity can leave programs idle, which CAP // SSPLIT did not survive."""
    cap = max(SBLOCK, triton.next_power_of_2(top_k * CAP_MULT))
    chunk = triton.cdiv(triton.cdiv(vocab, SSPLIT), SBLOCK * 4) * SBLOCK * 4
    return cap, chunk, min(cap, chunk)


# The sample and the collect key off the operator's own 11-bit STEP-0 key; the
# retry in _s_finish keys off the full 32-bit one, because the 11-bit key
# collapses on a narrow band and the retry is what has to be exact.
_key11 = _generic._convert_to_trt_uint16_hi11
_key32 = _generic._convert_to_uint32


@triton.jit
def _s_scan(base, target, NB: tl.constexpr, BLOCK: tl.constexpr):
    """Lowest bin whose inclusive prefix reaches `target`, and that bin's
    exclusive prefix. Bin 0 holds the largest values."""
    lane = tl.arange(0, BLOCK)
    carry = tl.zeros([], tl.int32)
    tb = tl.full([], NB - 1, tl.int32)
    lt = tl.zeros([], tl.int32)
    found = tl.full([], False, tl.int1)
    for t in tl.static_range(NB // BLOCK):
        bins = t * BLOCK + lane
        c = tl.load(base + bins)
        pre = carry + tl.cumsum(c, axis=0) - c
        hit = (pre < target) & (pre + c >= target) & (not found)
        cand = tl.min(tl.where(hit, bins, NB - 1), axis=0)
        candlt = tl.max(tl.where(hit, pre, 0), axis=0)
        if (not found) & (tl.max(hit.to(tl.int32), axis=0) > 0):
            tb = cand
            lt = candlt
            found = tl.full([], True, tl.int1)
        carry += tl.sum(c, axis=0)
    return tb, lt


@triton.jit
def _s_hist(
    logits_ptr, base, row, stride0, s, e, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    """Histogram every STRIDE-th TILE of [s, e): whole tiles read 1/STRIDE of
    the bytes, where strided elements would touch every cache line. The sample
    is unbiased only for rows without spatial structure; _s_finish's exact
    retry covers the rest."""
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(e - s, BLOCK * STRIDE)):
        i = s + t * BLOCK * STRIDE + lane
        m = i < e
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        tl.atomic_add(
            base + _key11(x),
            tl.full([BLOCK], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="cta",
        )


@triton.jit
def _s_prepare(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    thr_ptr,
    cnt_ptr,
    stride0,
    TARGET: tl.constexpr,
    NB: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Zero, sample and threshold, one program per row -- so the histogram is
    this program's alone and a barrier is all the ordering needed."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.store(cnt_ptr + row * SPLIT + tl.arange(0, SPLIT), tl.zeros([SPLIT], tl.int32))
    tl.debug_barrier()
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    _s_hist(logits_ptr, base, row, stride0, s, e, STRIDE, BLOCK)
    tl.debug_barrier()
    # Take the WHOLE boundary bin (+1, exclusive): coarser estimates should
    # over-collect, since falling short of top_k forces the exact retry while
    # overshooting only costs a slightly larger ranking.
    tb, _ = _s_scan(base, tl.cdiv(TARGET, STRIDE), NB, BLOCK)
    tl.store(thr_ptr + row, tb + 1)


@triton.jit
def _s_collect(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    thr_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
    SPLIT: tl.constexpr,
    CHUNK: tl.constexpr,
    SEG: tl.constexpr,
):
    """Append every element strictly better than the threshold bin, as an
    index relative to row_start; _s_finish re-reads the values.

    The bulk loop is unmasked and the remainder handled separately, as in the
    generic passes: a mask on every load cost this pass more than twice its
    modeled time. SPLIT programs divide the row, each appending through its
    own counter into its own SEG-long segment. CHUNK is a multiple of
    BLOCK * VEC so the bulk loop stays unmasked; the last part runs to the end
    of the row regardless.
    """
    pid = tl.program_id(0)
    row = pid // SPLIT
    part = pid % SPLIT
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    thr = tl.load(thr_ptr + row)
    base = logits_ptr + row * stride0 + s
    lane = tl.arange(0, BLOCK)
    off = lane[:, None] * VEC + tl.arange(0, VEC)[None, :]
    ones2 = tl.full([BLOCK, VEC], 1, tl.int32)
    ones1 = tl.full([BLOCK], 1, tl.int32)
    cnt2 = cnt_ptr + pid + tl.zeros([BLOCK, VEC], tl.int32)
    cnt1 = cnt_ptr + pid + tl.zeros([BLOCK], tl.int32)

    start = part * CHUNK
    stop = tl.minimum(start + CHUNK, span)
    stop = tl.where(part == SPLIT - 1, span, stop)
    have = tl.maximum(stop - start, 0)

    n_vec = have // (BLOCK * VEC)
    # Two stages hide load latency inside a wave; deeper ones cost registers,
    # and so occupancy.
    for t in tl.range(0, n_vec, num_stages=2):
        i = start + t * BLOCK * VEC + off
        x = tl.load(base + i)
        # Cast explicitly: the key is uint32 and thr int32, and leaving that
        # promotion implicit selects every element (the MTT override records
        # the same bug).
        take = _key11(x).to(tl.int32) < thr
        pos = tl.atomic_add(cnt2, ones2, mask=take, sem="relaxed", scope="cta")
        keep = take & (pos >= 0) & (pos < SEG)
        tl.store(cand_idx_ptr + pid * SEG + pos, i.to(tl.int32), mask=keep)

    tail = start + n_vec * BLOCK * VEC
    for t in tl.range(0, tl.cdiv(tl.maximum(stop - tail, 0), BLOCK)):
        i = tail + t * BLOCK + lane
        m = i < stop
        x = tl.load(base + i, mask=m, other=0.0)
        take = m & (_key11(x).to(tl.int32) < thr)
        pos = tl.atomic_add(cnt1, ones1, mask=take, sem="relaxed", scope="cta")
        keep = take & (pos >= 0) & (pos < SEG)
        tl.store(cand_idx_ptr + pid * SEG + pos, i.to(tl.int32), mask=keep)


@triton.jit
def _s_finish(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    cidx_ptr,
    out_ptr,
    counts_ptr,
    slot_ptr,
    stride0,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    CAP: tl.constexpr,
    RADIX: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLIT: tl.constexpr,
    SEG: tl.constexpr,
):
    """The retry decision and the exact answer, one program per row.

    A row outside [TOPK, CAP], or with a full segment, is redone over the full
    32-bit ordered key; the 11-bit key the sample and the collect use can
    collapse on a narrow band. Rows inside take the exact top-k of their
    candidates: four 8-bit radix rounds over the same 32-bit key.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX)
    ones = tl.full([BLOCK], 1, tl.int32)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    c = tl.zeros((), tl.int32)
    over = tl.zeros((), tl.int32)
    for sg in tl.static_range(SPLIT):
        craw = tl.load(cnt_ptr + row * SPLIT + sg)
        c += tl.minimum(craw, SEG)
        over += (craw > SEG).to(tl.int32)
    if (c < tl.minimum(TOPK, span)) | (c > CAP) | (over > 0):
        # The 11-bit fp16 key resolves magnitude/32, so a row in a narrow band
        # away from zero collapses into one or two bins and an overflow can
        # drop the true top-k. This path has no STEP 1-3 to refine through, so
        # the redo ranks the full 32-bit ordered key, which is injective on
        # distinct floats. Same fix as top_k_per_row_decode's.
        obase_r = out_ptr + row * TOPK
        cbase_r = counts_ptr + row * RADIX
        rbase = logits_ptr + row * stride0 + s
        rdesired = tl.zeros((), dtype=tl.uint32)
        rmask = tl.zeros((), dtype=tl.uint32)
        r_to_find = TOPK + 1
        row_tiles = tl.cdiv(span, BLOCK)
        for rdpos in tl.static_range(24, -1, -8):
            if r_to_find > 1:
                tl.store(cbase_r + bins, tl.zeros([RADIX], tl.int32))
                tl.debug_barrier()
                for rt in tl.range(0, row_tiles):
                    ri = rt * BLOCK + lane
                    rvalid = ri < span
                    rkey = _key32(tl.load(rbase + ri, mask=rvalid, other=0.0))
                    rdigit = ((rkey >> rdpos) & (RADIX - 1)).to(tl.int32)
                    tl.atomic_add(
                        cbase_r + rdigit,
                        ones,
                        mask=rvalid & ((rkey & rmask) == rdesired),
                        sem="relaxed",
                        scope="cta",
                    )
                tl.debug_barrier()
                rcounts = tl.load(cbase_r + bins)
                rprefix = tl.cumsum(rcounts, axis=0) - rcounts
                rhit = (rprefix < r_to_find) & (rprefix + rcounts >= r_to_find)
                rb0 = tl.min(tl.where(rhit, bins, RADIX), axis=0).to(tl.int32)
                rb0 = tl.where(rb0 == RADIX, RADIX - 1, rb0)
                rlt = tl.max(tl.where(bins == rb0, rprefix, 0), axis=0).to(tl.int32)
                rdesired = rdesired | (rb0.to(tl.uint32) << rdpos)
                rmask = rmask | (tl.full((), RADIX - 1, tl.uint32) << rdpos)
                r_to_find = r_to_find - rlt
        rthr = rdesired
        tl.store(slot_ptr + row, 0)
        tl.debug_barrier()
        rslots = slot_ptr + row + tl.zeros([BLOCK], tl.int32)
        # strictly better than the k-th, then its exact ties
        for req in tl.static_range(2):
            for rt2 in tl.range(0, row_tiles):
                ri2 = rt2 * BLOCK + lane
                rvalid2 = ri2 < span
                rkey2 = _key32(tl.load(rbase + ri2, mask=rvalid2, other=0.0))
                if req == 0:
                    rtake = rvalid2 & (rkey2 < rthr)
                else:
                    rtake = rvalid2 & (rkey2 == rthr)
                rq = tl.atomic_add(rslots, ones, mask=rtake, sem="relaxed", scope="cta")
                tl.store(obase_r + rq, ri2.to(tl.int32), mask=rtake & (rq < TOPK))
            tl.debug_barrier()
        # a row shorter than TOPK leaves the rest of the output padded
        rfilled = tl.load(slot_ptr + row)
        for rp in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            rj = rp * BLOCK + lane
            tl.store(obase_r + rj, -1, mask=(rj >= rfilled) & (rj < TOPK))
        return

    ibase = cidx_ptr + row * CAP
    vbase = cand_val_ptr + row * CAP
    obase = out_ptr + row * TOPK
    cbase = counts_ptr + row * RADIX

    # _s_collect stores indices only, in per-program segments: compact them and
    # gather each candidate's value on the way. The barrier is required; this
    # program reads vbase and ibase right after storing them.
    row_base = logits_ptr + row * stride0 + s
    n = tl.zeros((), tl.int32)
    for sg in tl.static_range(SPLIT):
        cseg = tl.minimum(tl.load(cnt_ptr + row * SPLIT + sg), SEG)
        sbase = cand_idx_ptr + (row * SPLIT + sg) * SEG
        for t in tl.range(0, tl.cdiv(cseg, BLOCK)):
            p = t * BLOCK + lane
            pv = p < cseg
            ci = tl.load(sbase + p, mask=pv, other=0)
            tl.store(ibase + n + p, ci, mask=pv)
            tl.store(vbase + n + p, tl.load(row_base + ci, mask=pv, other=0.0), mask=pv)
        n += cseg
    tiles = tl.cdiv(n, BLOCK)
    tl.debug_barrier()

    if n <= TOPK:
        for ft in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            j = ft * BLOCK + lane
            idx = tl.load(ibase + j, mask=j < n, other=-1)
            tl.store(obase + j, tl.where(j < n, idx, -1), mask=j < TOPK)
        return

    desired = tl.zeros((), tl.uint32)
    desired_mask = tl.zeros((), tl.uint32)
    k_to_find = TOPK + 1
    for digit_pos in tl.static_range(24, -1, -8):
        if k_to_find > 1:
            tl.store(cbase + bins, tl.zeros([RADIX], tl.int32))
            tl.debug_barrier()
            for t in tl.range(0, tiles):
                pos = t * BLOCK + lane
                valid = pos < n
                key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                tl.atomic_add(
                    cbase + digit,
                    ones,
                    mask=valid & ((key & desired_mask) == desired),
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()
            cnts = tl.load(cbase + bins)
            prefix = tl.cumsum(cnts, axis=0) - cnts
            hit = (prefix < k_to_find) & (prefix + cnts >= k_to_find)
            rb = tl.min(tl.where(hit, bins, RADIX), axis=0).to(tl.int32)
            rb = tl.where(rb == RADIX, RADIX - 1, rb)
            lt = tl.max(tl.where(bins == rb, prefix, 0), axis=0).to(tl.int32)
            desired = desired | (rb.to(tl.uint32) << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX - 1, tl.uint32) << digit_pos
            )
            k_to_find = k_to_find - lt

    thr_key = desired
    # One program owns the row, so output positions need no atomic: a running
    # offset plus an exclusive prefix over the take mask. Allocating the slots
    # with an atomic took nearly half of this kernel's time.
    filled = tl.zeros((), tl.int32)
    for equal in tl.static_range(2):
        for t in tl.range(0, tiles):
            pos = t * BLOCK + lane
            valid = pos < n
            key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            ti = take.to(tl.int32)
            q = filled + tl.cumsum(ti, axis=0) - ti
            filled += tl.sum(ti, axis=0)
            idx = tl.load(ibase + pos, mask=take, other=-1)
            tl.store(obase + q, idx, mask=take & (q < TOPK))


class _SLaunch:
    """One kernel: JIT on first use, direct afterwards. Same recipe as the
    decode override's; copied so the two operators stay independent."""

    __slots__ = ("jit", "grid", "grid3", "constexprs", "num_warps", "runner")

    def __init__(self, jit, grid, constexprs, num_warps):
        self.jit = jit
        self.grid = grid
        self.grid3 = tuple(grid) + (1,) * (3 - len(grid))
        self.constexprs = constexprs
        self.num_warps = num_warps
        self.runner = None

    def __call__(self, *args):
        if self.runner is not None:
            self.runner(*args, *self.constexprs.values())
            return
        ck = self.jit.run(
            *args,
            **self.constexprs,
            num_warps=self.num_warps,
            grid=self.grid,
            warmup=False,
        )
        if ck is not None:
            self.runner = ck[self.grid3]


class _SPlan:
    """Buffers and the three launches for one sampled shape."""

    def __init__(self, dev, dtype, num_rows, vocab, top_k):
        import torch

        cap, schunk, seg = _s_geometry(vocab, top_k)
        self.cap = cap
        nb = _generic.NUM_BINS
        self.hist = torch.empty((num_rows, nb), dtype=torch.int32, device=dev)
        self.thr = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cnt = torch.empty((num_rows * SSPLIT,), dtype=torch.int32, device=dev)
        self.cand_idx = torch.empty(
            (num_rows, SSPLIT * seg), dtype=torch.int32, device=dev
        )
        self.cand_val = torch.empty((num_rows, cap), dtype=dtype, device=dev)
        self.cidx = torch.empty((num_rows, cap), dtype=torch.int32, device=dev)
        self.counts = torch.empty((num_rows, SRADIX), dtype=torch.int32, device=dev)
        self.slot = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.prepare = _SLaunch(
            _s_prepare,
            (num_rows,),
            {
                "TARGET": int(top_k * TARGET_MULT),
                "NB": nb,
                "STRIDE": SSTRIDE,
                "BLOCK": SBLOCK,
                "SPLIT": SSPLIT,
            },
            SWARPS,
        )
        self.collect = _SLaunch(
            _s_collect,
            (num_rows * SSPLIT,),
            {
                "CAP": cap,
                "BLOCK": SBLOCK,
                "VEC": 4,
                "SPLIT": SSPLIT,
                "CHUNK": schunk,
                "SEG": seg,
            },
            SWARPS,
        )
        self.finish = _SLaunch(
            _s_finish,
            (num_rows,),
            {
                "TOPK": top_k,
                "NB": nb,
                "CAP": cap,
                "RADIX": SRADIX,
                "BLOCK": SBLOCK,
                "SPLIT": SSPLIT,
                "SEG": seg,
            },
            SWARPS,
        )

    def run(self, logits, starts, ends, indices, stride0):
        self.prepare(logits, starts, ends, self.hist, self.thr, self.cnt, stride0)
        self.collect(
            logits,
            starts,
            ends,
            self.thr,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            stride0,
        )
        self.finish(
            logits,
            starts,
            ends,
            self.hist,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            self.cidx,
            indices,
            self.counts,
            self.slot,
            stride0,
        )


_SPLANS = {}
_SPLANS_MAX = 8
_SPLAN_LOCK = threading.Lock()


def _s_aligned(t):
    return t.data_ptr() % 16 == 0


def _can_sample(logits, row_starts, row_ends, num_rows, stride0, stride1, top_k):
    import torch

    vocab = logits.shape[1]
    return (
        vocab >= SAMPLED_MIN_VOCAB_PER_TOPK * top_k
        and stride1 == 1
        and num_rows > 0
        and num_rows == logits.shape[0]
        and logits.dtype == torch.float32
        and row_starts.dtype == torch.int32
        and row_ends.dtype == torch.int32
        and not getattr(_generic, "HAS_TLE", False)
        and num_rows * SSPLIT * _s_geometry(vocab, top_k)[2] <= _MAX_CAND_ELEMS
        and num_rows * _s_geometry(vocab, top_k)[0] <= _MAX_CAND_ELEMS
    )


# ---------------------------------------------------------------------------
# One read for the large dense shapes. The dense route reads each row twice and
# fires one global atomic per element for its histogram; on 12961x4100 the
# atomics are about half its time, bound by the distinct addresses they touch,
# and the second read, which L2 does not serve, a fifth. This route reads each
# row once and issues no per-element atomic (_d_sampled). A row it cannot
# answer is flagged -- about 1% of standard-normal rows, every row of a narrow
# band -- and the dense route, launched after it for every row, returns at once
# unless its row was flagged, so every answer is exact. On standard-normal rows
# the route is 1.6-1.8x the dense route alone; with every row flagged it costs
# 2-6% more.
DS_BLOCK = 512
DS_NS = 512
DS_BCAP = 512
DS_HI = 75  # percent of top_k expected above T_hi: surely in
DS_LO = 135  # percent of top_k expected above T_lo: the band's end


@triton.jit
def _d_kth(keys, valid, r):
    """The r-th smallest 11-bit key (1-based) among the valid lanes."""
    res = tl.zeros((), tl.int32)
    for b in tl.static_range(10, -1, -1):
        probe = res | (1 << b)
        cnt = tl.sum((valid & (keys < probe)).to(tl.int32), axis=0)
        res = tl.where(cnt < r, probe, res)
    return res


@triton.jit
def _d_classify(
    x, i, m, t_hi, t_lo, S, B, obase, kb, ib, TOPK: tl.constexpr, BCAP: tl.constexpr
):
    k = _key11(x).to(tl.int32)
    sure = m & (k < t_hi)
    band = m & (k >= t_hi) & (k < t_lo)
    # both slot positions from one scan: the sure count in the low 16 bits
    packed = sure.to(tl.int32) + (band.to(tl.int32) << 16)
    cs = tl.cumsum(packed, axis=0) - packed
    tot = tl.sum(packed, axis=0)
    ps = S + (cs & 0xFFFF)
    pb = B + (cs >> 16)
    tl.store(obase + ps, i, mask=sure & (ps < TOPK))
    keep = band & (pb < BCAP)
    tl.store(kb + pb, _key32(x).to(tl.int32, bitcast=True), mask=keep)
    tl.store(ib + pb, i, mask=keep)
    return S + (tot & 0xFFFF), B + (tot >> 16)


@triton.jit
def _d_sampled(
    x_ptr,
    starts_ptr,
    ends_ptr,
    out_ptr,
    bkey_ptr,
    bidx_ptr,
    flag_ptr,
    stride0,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    NS: tl.constexpr,
    BCAP: tl.constexpr,
    HI: tl.constexpr,
    LO: tl.constexpr,
):
    """One program per row: the row's top-k and a flag of 0, or a flag of 1
    and the row left to the dense route.

    Two thresholds come from an NS-element sample by bitwise lifting on the
    11-bit key: about HI% of top_k lies above T_hi and is surely in, about LO%
    above T_lo. One pass writes the sure set straight to the output and keeps
    the band [T_hi, T_lo) in BCAP slots; the missing top_k - S then come from
    the band by lifting on the full 32-bit ordered key. The row is flagged when
    its sure set passes top_k, its band falls short, or the band overflows. At
    one warp every reduction and scan stays inside a wave.
    """
    row = tl.program_id(0)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    base = x_ptr + row * stride0 + s

    CH: tl.constexpr = NS // 8
    sl = tl.arange(0, NS)
    si = (sl // CH) * (span // 8) + (sl % CH)
    sv = si < span
    sk = _key11(tl.load(base + si, mask=sv, other=float("-inf"))).to(tl.int32)
    ns = tl.sum(sv.to(tl.int32), axis=0)
    expect = TOPK * ns.to(tl.float32) / tl.maximum(span, 1).to(tl.float32)
    t_hi = _d_kth(sk, sv, (expect * HI / 100).to(tl.int32))
    t_lo = _d_kth(sk, sv, (expect * LO / 100).to(tl.int32) + 1) + 1

    lane = tl.arange(0, BLOCK)
    obase = out_ptr + row * TOPK
    kb = bkey_ptr + row * BCAP
    ib = bidx_ptr + row * BCAP
    S = tl.zeros((), tl.int32)
    B = tl.zeros((), tl.int32)
    n_full = span // BLOCK
    for t in tl.range(0, n_full):
        i = t * BLOCK + lane
        S, B = _d_classify(
            tl.load(base + i), i, i >= 0, t_hi, t_lo, S, B, obase, kb, ib, TOPK, BCAP
        )
    i = n_full * BLOCK + lane
    m = i < span
    x = tl.load(base + i, mask=m, other=float("-inf"))
    S, B = _d_classify(x, i, m, t_hi, t_lo, S, B, obase, kb, ib, TOPK, BCAP)

    need = TOPK - S
    good = (S <= TOPK) & (need <= B) & (B <= BCAP)
    tl.store(flag_ptr + row, 1 - good.to(tl.int32))
    # the band select reads back what this program just stored
    tl.debug_barrier()
    if good:
        q = tl.arange(0, BCAP)
        bv = q < B
        bk = tl.load(kb + q, mask=bv, other=0).to(tl.uint32, bitcast=True)
        # Lift only the bits where the band's keys differ. The highest one is
        # the float exponent of min ^ max, which can round up, never down.
        one = tl.full((), 1, tl.uint32)
        kmin = tl.min(tl.where(bv, bk, tl.full([BCAP], 0xFFFFFFFF, tl.uint32)), axis=0)
        kmax = tl.max(tl.where(bv, bk, tl.zeros([BCAP], tl.uint32)), axis=0)
        hb = ((kmin ^ kmax).to(tl.float32).to(tl.int32, bitcast=True) >> 23) - 127
        hb = tl.minimum(tl.maximum(hb, 0), 31)
        nb = hb + 1
        low = tl.where(
            nb >= 32,
            tl.full((), 0xFFFFFFFF, tl.uint32),
            (one << nb.to(tl.uint32)) - one,
        )
        kth = kmin & ~low
        for j in tl.range(0, nb):
            probe = kth | (one << (hb - j).to(tl.uint32))
            cnt = tl.sum((bv & (bk < probe)).to(tl.int32), axis=0)
            kth = tl.where(cnt < need, probe, kth)
        idx = tl.load(ib + q, mask=bv, other=0)
        lt = bv & (bk < kth)
        lti = lt.to(tl.int32)
        nlt = tl.sum(lti, axis=0)
        tl.store(obase + S + tl.cumsum(lti, axis=0) - lti, idx, mask=lt)
        eq = bv & (bk == kth)
        eqi = eq.to(tl.int32)
        pe = S + nlt + tl.cumsum(eqi, axis=0) - eqi
        tl.store(obase + pe, idx, mask=eq & (pe < TOPK))


class _DPlan:
    """Buffers and the sampled launch for one dense shape."""

    def __init__(self, dev, num_rows, top_k):
        self.bkey = torch.empty((num_rows, DS_BCAP), dtype=torch.int32, device=dev)
        self.bidx = torch.empty((num_rows, DS_BCAP), dtype=torch.int32, device=dev)
        self.flags = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.launch = _SLaunch(
            _d_sampled,
            (num_rows,),
            {
                "TOPK": top_k,
                "BLOCK": DS_BLOCK,
                "NS": DS_NS,
                "BCAP": DS_BCAP,
                "HI": DS_HI,
                "LO": DS_LO,
            },
            1,
        )


_DPLANS = {}
_DPLAN_LOCK = threading.Lock()


def _can_dense_sample(logits, row_starts, row_ends, num_rows, stride1, top_k):
    vocab = logits.shape[1]
    return (
        top_k == 512
        and num_rows >= 8192
        and 2048 <= vocab <= 5120
        and stride1 == 1
        and num_rows == logits.shape[0]
        and logits.dtype == torch.float32
        and row_starts.dtype == torch.int32
        and row_ends.dtype == torch.int32
        and not getattr(_generic, "HAS_TLE", False)
        and num_rows * DS_BCAP <= _MAX_CAND_ELEMS
    )


def _dense_sampled(logits, row_starts, row_ends, indices, num_rows, stride0, top_k):
    dev = logits.device
    key = (
        dev,
        torch.cuda.current_stream(dev).cuda_stream,
        num_rows,
        logits.shape[1],
        top_k,
        stride0,
        _s_aligned(logits),
        _s_aligned(row_starts),
        _s_aligned(row_ends),
        _s_aligned(indices),
    )
    with _DPLAN_LOCK:
        plan = _DPLANS.get(key)
        if plan is None:
            if len(_DPLANS) >= _SPLANS_MAX:
                _DPLANS.pop(next(iter(_DPLANS)))
            plan = _DPLANS[key] = _DPlan(dev, num_rows, top_k)
        plan.launch(
            logits,
            row_starts,
            row_ends,
            indices,
            plan.bkey,
            plan.bidx,
            plan.flags,
            stride0,
        )
        _launch_radix(
            logits,
            row_starts,
            row_ends,
            indices,
            num_rows,
            stride0,
            1,
            top_k,
            _DENSE_ROUTE,
            skip=plan.flags,
        )
    return indices


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for DeepSeek V4 prefill, routed per call.

    Same contract as the generic operator. Routes, first match wins:

      sampled   vocab >= SAMPLED_MIN_VOCAB_PER_TOPK * top_k. prepare samples
                every SSTRIDE-th tile for a threshold, collect makes one pass
                split over SSPLIT programs, finish ranks the candidates exactly
                or redoes the row on the full 32-bit key.
      one-read  top_k 512, num_rows >= 8192, 2048 <= vocab <= 5120. One
                program per row, with the dense route as its retry.
      dense     vocab <= DENSE_VOCAB_PER_TOPK * top_k. The radix kernel with a
                prefix-sum slot allocator and VEC=2; 512 STEP-0 bins for
                top_k 512 on rows up to 1536.
      generic   everything else: the radix kernel with the generic operator's
                own bin pass.

    The sampled and one-read routes need stride1 == 1, float32 logits, int32
    row bounds, no TLE and buffers within _MAX_CAND_ELEMS; a call that misses
    a gate falls through to the next route. The dense and generic routes run
    at a launch geometry chosen by rows per SM and reuse their scratch buffers.

    FLAGGEMS_HYGON_TOPK_PREFILL=0 disables the override, and so does a Triton
    with TLE; every call then goes to the generic operator.
    """
    if not _ENABLED or getattr(_generic, "HAS_TLE", False):
        return _generic.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )
    if _can_sample(logits, row_starts, row_ends, num_rows, stride0, stride1, top_k):
        skey = (
            logits.device,
            num_rows,
            logits.shape[1],
            top_k,
            stride0,
            _s_aligned(logits),
            _s_aligned(row_starts),
            _s_aligned(row_ends),
            _s_aligned(indices),
        )
        with _SPLAN_LOCK:
            plan = _SPLANS.get(skey)
            if plan is None:
                if len(_SPLANS) >= _SPLANS_MAX:
                    _SPLANS.pop(next(iter(_SPLANS)))
                plan = _SPLANS[skey] = _SPlan(
                    logits.device, logits.dtype, num_rows, logits.shape[1], top_k
                )
            plan.run(logits, row_starts, row_ends, indices, stride0)
        return indices

    if _can_dense_sample(logits, row_starts, row_ends, num_rows, stride1, top_k):
        return _dense_sampled(
            logits, row_starts, row_ends, indices, num_rows, stride0, top_k
        )

    return _launch_radix(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
        _route(logits.shape[1], top_k),
    )

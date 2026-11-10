# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""MetaX implementation of the common per-row decode Top-k contract."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._metax.fused.top_k_per_row_decode import (
    top_k_per_row_decode as _general_top_k_per_row_decode,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.common import fill_tensor

logger = logging.getLogger(__name__)


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Mutate indices and return None, as the common NVIDIA-facing API does.

    The existing stable tree is used only for its validated one-query K512
    domain. All other domains retain the previous MetaX public implementation.
    Selection happens before either implementation writes or launches work.
    """
    logger.debug("GEMS_METAX TOP_K_PER_ROW_DECODE")
    if (
        next_n == 1
        and top_k == 512
        and logits.ndim == 2
        and logits.dtype == torch.float32
        and logits.stride(1) == 1
        and num_rows == logits.shape[0]
        and (stride0, stride1) == logits.stride()
        and seq_lens.shape == (num_rows,)
        and seq_lens.dtype in (torch.int32, torch.int64)
        and indices.ndim == 2
        and indices.shape[0] == num_rows
        and indices.shape[1] >= top_k
        and indices.dtype == torch.int32
        and seq_lens.device == logits.device == indices.device
    ):
        _decode_topk_tree(logits, seq_lens, indices, k=top_k)
        return None
    return _general_top_k_per_row_decode(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
    )


@triton.jit
def _row_local_topk(
    logits_ptr,
    valid_ptr,
    cand_vals_ptr,
    cand_idx_ptr,
    stride_logits_row,
    NB_MAX: tl.constexpr,
    K: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(1)
    blk = tl.program_id(0)
    valid_len = tl.load(valid_ptr + row)
    # Early-exit (ml-static-waste fix): when the whole block lies beyond
    # valid_len, `chosen` is all-false in the original kernel (no stores), and
    # the caller pre-fills cand buffers with -inf/-1 every call, so returning
    # early is semantically identical. Removes the tl.topk on padding blocks
    # whose count scaled with static max_model_len. Grid stays static ->
    # graph-replay safe.
    if blk * BC >= valid_len:
        return
    offs = tl.arange(0, BC)
    g = blk * BC + offs
    vals = tl.load(
        logits_ptr + row * stride_logits_row + g,
        mask=g < valid_len,
        other=-float("inf"),
    )
    tv = tl.topk(vals, K)
    thresh = tl.min(tv, axis=0)
    valid = g < valid_len
    above = valid & (vals > thresh)
    equal = valid & (vals == thresh)
    # Reserve all strictly higher scores before taking the lowest-index ties.
    # Truncating vals>=threshold by position can discard a later higher score.
    remaining = K - tl.sum(above.to(tl.int32), 0)
    tie_pos = tl.cumsum(equal.to(tl.int32), 0)
    chosen = above | (equal & (tie_pos <= remaining))
    pos = tl.cumsum(chosen.to(tl.int32), 0) - 1
    base = (row * NB_MAX + blk) * K
    tl.store(cand_vals_ptr + base + pos, vals, mask=chosen & (pos < K))
    tl.store(cand_idx_ptr + base + pos, g.to(tl.int32), mask=chosen & (pos < K))


@triton.jit
def _row_reduce_pair(
    in_vals_ptr,
    in_idx_ptr,
    out_vals_ptr,
    out_idx_ptr,
    NB_IN: tl.constexpr,
    NB_OUT: tl.constexpr,
    K: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(1)
    pid = tl.program_id(0)
    offs = tl.arange(0, BC)
    in_base = (row * NB_IN + pid * (BC // K)) * K + offs
    valid = (pid * BC + offs) < NB_IN * K
    vals = tl.load(in_vals_ptr + in_base, mask=valid, other=-float("inf"))
    idx = tl.load(in_idx_ptr + in_base, mask=valid, other=-1)
    tv = tl.topk(vals, K)
    thresh = tl.min(tv, axis=0)
    valid = valid & (idx >= 0)
    above = valid & (vals > thresh)
    equal = valid & (vals == thresh)
    remaining = K - tl.sum(above.to(tl.int32), 0)
    tie_pos = tl.cumsum(equal.to(tl.int32), 0)
    chosen = above | (equal & (tie_pos <= remaining))
    pos = tl.cumsum(chosen.to(tl.int32), 0) - 1
    out_base = (row * NB_OUT + pid) * K
    tl.store(out_vals_ptr + out_base + pos, vals, mask=chosen & (pos < K))
    tl.store(out_idx_ptr + out_base + pos, idx, mask=chosen & (pos < K))


def _levels(nb_max):
    """Buffer block counts per level: nb_max, ceil(/2), ..., 1."""
    levels = [nb_max]
    while levels[-1] > 1:
        levels.append(triton.cdiv(levels[-1], 2))
    return levels


def _decode_topk_tree(logits, valid_lens, out_indices, k=512, block=1024):
    """Exact per-row top-k (or select-all when valid_len <= k).

    logits:      [rows, N] fp32, row-stride arbitrary; columns >= valid_lens[r]
                 are ignored (callers must ensure padded cols are not relied on).
    valid_lens:  [rows] int32/int64 on the same device.
    out_indices: [rows, k] int32; entries not selected stay -1 (fresh buffers
                 are initialized to -1 here each call).
    """
    logger.debug("GEMS_METAX DECODE_TOPK_TREE")
    if (
        logits.ndim != 2
        or valid_lens.shape != (logits.shape[0],)
        or out_indices.ndim != 2
        or out_indices.shape[0] != logits.shape[0]
    ):
        raise ValueError("Invalid Top-k row shapes")
    if k != 512 or block != 1024:
        raise NotImplementedError("This tree variant supports K512/block1024")
    if (
        out_indices.dtype != torch.int32
        or valid_lens.dtype not in (torch.int32, torch.int64)
        or logits.stride(1) != 1
    ):
        raise TypeError("Invalid Top-k dtype/layout")
    if valid_lens.device != logits.device or out_indices.device != logits.device:
        raise ValueError("Top-k inputs must share a device")
    if not (logits.is_cuda and valid_lens.is_cuda and out_indices.is_cuda):
        raise ValueError(
            "Invalid operator argument: logits.is_cuda and valid_lens.is_cuda and out_indices.is_cuda"
        )
    if not (logits.dtype == torch.float32):
        raise ValueError("Invalid operator argument: logits.dtype == torch.float32")
    if not (out_indices.shape[1] >= k):
        raise ValueError("Invalid operator argument: out_indices.shape[1] >= k")
    if logits.shape[0] == 0:
        return out_indices
    if logits.shape[1] == 0:
        fill_tensor(out_indices[:, :k], -1)
        return out_indices

    rows = logits.shape[0]
    nb_max = triton.cdiv(logits.shape[1], block)
    levels = _levels(nb_max)
    dev = logits.device

    vals_buf = [
        fill_tensor(
            torch.empty((rows, lv * k), dtype=torch.float32, device=dev), -float("inf")
        )
        for lv in levels
    ]
    idx_buf = [
        fill_tensor(torch.empty((rows, lv * k), dtype=torch.int32, device=dev), -1)
        for lv in levels
    ]

    _row_local_topk[(nb_max, rows)](
        logits,
        valid_lens,
        vals_buf[0],
        idx_buf[0],
        logits.stride(0),
        NB_MAX=nb_max,
        K=k,
        BC=block,
    )
    nb = nb_max
    lvl = 1
    while nb > 1:
        nb_out = levels[lvl]
        _row_reduce_pair[(nb_out, rows)](
            vals_buf[lvl - 1],
            idx_buf[lvl - 1],
            vals_buf[lvl],
            idx_buf[lvl],
            NB_IN=nb,
            NB_OUT=nb_out,
            K=k,
            BC=2 * k,
        )
        nb = nb_out
        lvl += 1
    out_indices[:, :k] = idx_buf[-1][:, :k]
    return out_indices

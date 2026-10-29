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

"""
Apply per-row top-k / top-p filtering on Ascend NPU (Triton).

#
# Pipeline (two stages, same as the reference "sort first, then custom kernel"):
#
#   Stage 1 (sort): produce the per-row sorted top-L ``(value, index)`` pair
#       in DESCENDING order (largest first). ``L`` is the sort width.
#   Stage 2 (kernel): apply per-row top-k/top-p and scatter to vocab order.
#
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops.topk import topk

try:
    import triton.language.extra.cann.extension as al

    HAS_CANN_EXT = True
except (ImportError, AttributeError):
    al = None
    HAS_CANN_EXT = False

_NEG_INF = float("-inf")

# UB-friendly chunk size (fp32).
_BLOCK_N = 4 * 1024

_npu_core_counts = None


def _get_aiv_num():
    """Return (vectorcore_num, aicore_num) for current device, cached."""
    global _npu_core_counts
    if _npu_core_counts is not None:
        return _npu_core_counts
    import torch_npu
    import triton.runtime.driver as driver

    props = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
    _npu_core_counts = props["num_vectorcore"]
    return _npu_core_counts


@triton.jit
def _apply_top_k_top_p_kernel(
    sorted_values_ptr,  # [B, L] DESCENDING sorted top-L values (largest first)
    sorted_indices_ptr,  # [B, L] int32 original vocab indices of the sorted values
    k_ptr,  # [B] int32 per-row top-k counts, or dummy
    p_ptr,  # [B] float top-p thresholds, or dummy
    out_ptr,  # [B, V], initialized from logits
    vocab_size,  # V, output width / scatter stride
    topk_len,  # L, input width (= max(k) if HAS_K else V)
    HAS_K: tl.constexpr,
    HAS_P: tl.constexpr,
    FULL_VOCAB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    in_start = row * topk_len
    out_start = row * vocab_size

    # per-row top-k limit within the sorted top-L block (descending => first k)
    kk = topk_len
    if HAS_K:
        kk = tl.load(k_ptr + row)

    # ---- fused single-tile fast path -----------------------------------------
    # When the whole sorted top-L block fits in one tile (the common top-k case
    # where L = max(k) is small), load ``(value, index)`` once and reuse it for
    # the softmax denominator, cumulative mass and scatter. This halves the loads
    # versus the two-pass multi-chunk path below.
    if NUM_CHUNKS == 1:
        offs = tl.arange(0, BLOCK_N)
        valid = offs < topk_len
        sv = tl.load(sorted_values_ptr + in_start + offs, mask=valid, other=NEG_INF).to(
            tl.float32
        )
        si = tl.load(sorted_indices_ptr + in_start + offs, mask=valid, other=0)
        keep = valid & (offs < kk)  # per-row top-k
        if HAS_P:
            pval = tl.load(p_ptr + row).to(tl.float32)
            # largest element is at position 0 (descending) and is always kept
            row_max = tl.load(sorted_values_ptr + in_start).to(tl.float32)
            e = tl.where(keep, tl.exp(sv - row_max), 0.0)
            inv_denom = 1.0 / tl.sum(e, axis=0)
            probs = e * inv_denom
            incl = tl.cumsum(probs, axis=0)
            excl = incl - probs  # cumulative mass of strictly-larger elements
            # nucleus: keep while larger-mass <= p; always keep the largest
            keep = keep & ((excl <= pval) | (offs == 0))
        write_num = tl.sum(keep.to(tl.int32), axis=0)
        if FULL_VOCAB:
            write_neg_inf = write_num > vocab_size // 2
            if write_neg_inf:
                tl.store(
                    out_ptr + out_start + si.to(tl.int64),
                    NEG_INF,
                    mask=valid & (offs >= write_num),
                )
            else:
                for c in tl.range(0, tl.cdiv(vocab_size, BLOCK_N)):
                    vocab_offs = c * BLOCK_N + tl.arange(0, BLOCK_N)
                    tl.store(
                        out_ptr + out_start + vocab_offs,
                        NEG_INF,
                        mask=vocab_offs < vocab_size,
                    )
                tl.debug_barrier()
                tl.store(
                    out_ptr + out_start + si.to(tl.int64),
                    sv.to(out_ptr.dtype.element_ty),
                    mask=valid & (offs < write_num),
                )
        else:
            tl.store(
                out_ptr + out_start + si.to(tl.int64),
                sv.to(out_ptr.dtype.element_ty),
                mask=valid & (offs < write_num),
            )
        return

    # ---- top-p pass 1: softmax denominator over the top-k-kept region --------
    inv_denom = 0.0
    pval = 0.0
    row_max = 0.0
    if HAS_P:
        pval = tl.load(p_ptr + row).to(tl.float32)
        # largest element is at position 0 (descending) and is always kept
        row_max = tl.load(sorted_values_ptr + in_start).to(tl.float32)
        denom = 0.0
        for c in tl.range(0, NUM_CHUNKS):
            offs = c * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = offs < topk_len
            keep_k = valid & (offs < kk)
            sv = tl.load(
                sorted_values_ptr + in_start + offs, mask=valid, other=NEG_INF
            ).to(tl.float32)
            e = tl.where(keep_k, tl.exp(sv - row_max), 0.0)
            denom += tl.sum(e, axis=0)
        inv_denom = 1.0 / denom

    # ---- pass 2: determine the final kept prefix length -----------------------
    write_num = kk
    if HAS_P:
        write_num = 0
        cum_prev = 0.0
        for c in tl.range(0, NUM_CHUNKS):
            offs = c * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = offs < topk_len
            sv = tl.load(
                sorted_values_ptr + in_start + offs,
                mask=valid,
                other=NEG_INF,
            ).to(tl.float32)
            keep = valid & (offs < kk)
            probs = tl.where(keep, tl.exp(sv - row_max) * inv_denom, 0.0)
            incl = tl.cumsum(probs, axis=0) + cum_prev
            excl = incl - probs
            cum_prev += tl.sum(probs, axis=0)
            keep = keep & ((excl <= pval) | (offs == 0))
            write_num += tl.sum(keep.to(tl.int32), axis=0)

    # ---- pass 3: write the selected side of the final output ------------------
    # The kept set is the prefix [0, write_num); the removed set is the suffix
    # [write_num, topk_len). Trim each write loop to the chunks that actually
    # overlap write_num instead of scanning all NUM_CHUNKS with a mask:
    #   - dense (write -inf into the suffix): skip whole chunks before write_num,
    #     start from the chunk that contains it (boundary chunk stays masked).
    #   - sparse (scatter the prefix): stop after the chunk that contains
    #     write_num (boundary chunk stays masked); skip the rest.
    c_start = write_num // BLOCK_N  # first chunk touching the suffix
    c_end = (write_num + BLOCK_N - 1) // BLOCK_N  # chunks covering the prefix
    if FULL_VOCAB:
        write_neg_inf = write_num > vocab_size // 2
        if write_neg_inf:
            for c in tl.range(c_start, NUM_CHUNKS):
                offs = c * BLOCK_N + tl.arange(0, BLOCK_N)
                valid = offs < topk_len
                si = tl.load(
                    sorted_indices_ptr + in_start + offs,
                    mask=valid,
                    other=0,
                )
                tl.store(
                    out_ptr + out_start + si.to(tl.int64),
                    NEG_INF,
                    mask=valid & (offs >= write_num),
                )
        else:
            for c in tl.range(0, tl.cdiv(vocab_size, BLOCK_N)):
                vocab_offs = c * BLOCK_N + tl.arange(0, BLOCK_N)
                tl.store(
                    out_ptr + out_start + vocab_offs,
                    NEG_INF,
                    mask=vocab_offs < vocab_size,
                )
            tl.debug_barrier()
            for c in tl.range(0, c_end):
                offs = c * BLOCK_N + tl.arange(0, BLOCK_N)
                valid = offs < topk_len
                sv = tl.load(
                    sorted_values_ptr + in_start + offs,
                    mask=valid,
                    other=NEG_INF,
                ).to(tl.float32)
                si = tl.load(
                    sorted_indices_ptr + in_start + offs,
                    mask=valid,
                    other=0,
                )
                tl.store(
                    out_ptr + out_start + si.to(tl.int64),
                    sv.to(out_ptr.dtype.element_ty),
                    mask=valid & (offs < write_num),
                )
    else:
        for c in tl.range(0, c_end):
            offs = c * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = offs < topk_len
            sv = tl.load(
                sorted_values_ptr + in_start + offs,
                mask=valid,
                other=NEG_INF,
            ).to(tl.float32)
            si = tl.load(
                sorted_indices_ptr + in_start + offs,
                mask=valid,
                other=0,
            )
            tl.store(
                out_ptr + out_start + si.to(tl.int64),
                sv.to(out_ptr.dtype.element_ty),
                mask=valid & (offs < write_num),
            )


# =============================================================================
# Multi-core scatter path (small-batch optimization)
#
# Mirrors the two-phase design of the CANN operator
#   ops-nn/index/apply_top_k_top_p_with_sorted
# where the compute stage only records a per-row cutoff index into a workspace,
# and a separate scatter stage spreads the surviving/removed region of each row
# across many cores (see ProcessScatter / ProcessResScatter). Here:
#
#   Phase 1 ``_compute_write_num_kernel``: grid=(batch,), one program per row,
#       computes ``write_num`` (the kept-prefix length after top-k + top-p) and
#       stores it to the ``write_nums`` workspace. No scatter.
#
#   Phase 2 ``_scatter_kernel``: grid=(batch, SCATTER_CORES). The SCATTER_CORES
#       programs of a row cooperatively scatter that row's region:
#         - sparse mode   : region = kept prefix [0, write_num), write values
#         - full-vocab mode: region = removed suffix [write_num, topk_len),
#                            knock those original positions to -inf (survivors
#                            keep their cloned logit values).
#       Chunks are assigned to cores by ``chunk_index % SCATTER_CORES`` so every
#       chunk is owned by exactly one core (no double writes, no races: the
#       sorted indices touched by different chunks are disjoint).
# =============================================================================


@triton.jit
def _compute_p_only_partials_kernel(
    sorted_values_ptr,
    partial_sums_ptr,
    topk_len,
    SHARDS: tl.constexpr,
    SHARD_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SHARD_TILES: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    shard = tl.program_id(1)
    in_start = row * topk_len
    shard_start = shard * SHARD_LEN
    shard_end = tl.minimum(shard_start + SHARD_LEN, topk_len)
    row_max = tl.load(sorted_values_ptr + in_start).to(tl.float32)

    partial = 0.0
    for tile in tl.range(0, SHARD_TILES):
        offs = shard_start + tile * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = offs < shard_end
        values = tl.load(
            sorted_values_ptr + in_start + offs, mask=valid, other=NEG_INF
        ).to(tl.float32)
        partial += tl.sum(tl.exp(values - row_max), axis=0)
    tl.store(partial_sums_ptr + row * SHARDS + shard, partial)


@triton.jit
def _scan_finalize_p_only_write_num_kernel(
    partial_sums_ptr,
    p_ptr,
    sorted_values_ptr,
    write_nums_ptr,
    topk_len,
    SHARDS: tl.constexpr,
    SHARD_LEN: tl.constexpr,
    SCAN_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SHARD_TILES: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    in_start = row * topk_len
    offs = tl.arange(0, SCAN_N)
    valid = offs < SHARDS
    partials = tl.load(
        partial_sums_ptr + row * SHARDS + offs,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    denom = tl.sum(partials, axis=0)
    inv_denom = 1.0 / denom
    masses = partials * inv_denom
    inclusive = tl.cumsum(masses, axis=0)
    pval = tl.load(p_ptr + row).to(tl.float32)
    crossing_mask = valid & (inclusive > pval)
    crossing = tl.min(tl.where(crossing_mask, offs, SCAN_N), axis=0)
    all_kept = crossing == SCAN_N
    crossing = tl.where(all_kept, SHARDS - 1, crossing)
    prefix_before = tl.sum(tl.where(valid & (offs < crossing), masses, 0.0), axis=0)

    if all_kept or pval >= 1.0:
        tl.store(write_nums_ptr + row, topk_len)
    else:
        shard_start = crossing * SHARD_LEN
        shard_end = tl.minimum(shard_start + SHARD_LEN, topk_len)
        row_max = tl.load(sorted_values_ptr + in_start).to(tl.float32)
        cumulative = prefix_before
        write_num = shard_start

        for tile in tl.range(0, SHARD_TILES):
            if cumulative <= pval:
                tile_offs = shard_start + tile * BLOCK_N + tl.arange(0, BLOCK_N)
                tile_valid = tile_offs < shard_end
                values = tl.load(
                    sorted_values_ptr + in_start + tile_offs,
                    mask=tile_valid,
                    other=NEG_INF,
                ).to(tl.float32)
                probs = tl.where(tile_valid, tl.exp(values - row_max) * inv_denom, 0.0)
                tile_inclusive = tl.cumsum(probs, axis=0) + cumulative
                tile_cumulative = al.get_element(tile_inclusive, (BLOCK_N - 1,))
                if tile_cumulative <= pval:
                    write_num += tl.sum(tile_valid.to(tl.int32), axis=0)
                else:
                    keep = tile_valid & (
                        ((tile_inclusive - probs) <= pval) | (tile_offs == 0)
                    )
                    write_num += tl.sum(keep.to(tl.int32), axis=0)
                cumulative = tile_cumulative
        tl.store(write_nums_ptr + row, write_num.to(tl.int32))


@triton.jit
def _compute_write_num_kernel(
    sorted_values_ptr,  # [B, L] DESCENDING sorted top-L values
    k_ptr,  # [B] int32 per-row top-k counts, or dummy
    p_ptr,  # [B] float top-p thresholds, or dummy
    write_nums_ptr,  # [B] int32 output: kept-prefix length per row
    topk_len,  # L
    HAS_K: tl.constexpr,
    HAS_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    in_start = row * topk_len

    kk = topk_len
    if HAS_K:
        kk = tl.load(k_ptr + row)

    # top-k only: kept prefix is exactly the first kk sorted positions.
    if not HAS_P:
        tl.store(write_nums_ptr + row, kk.to(tl.int32))
        return

    pval = tl.load(p_ptr + row).to(tl.float32)
    # largest element is at position 0 (descending) and is always kept.
    row_max = tl.load(sorted_values_ptr + in_start).to(tl.float32)

    # pass 1: full chunks are in bounds; only the final chunk needs a load mask.
    denom = 0.0
    for c in tl.range(0, NUM_CHUNKS - 1):
        offs = c * BLOCK_N + tl.arange(0, BLOCK_N)
        values = tl.load(sorted_values_ptr + in_start + offs).to(tl.float32)
        keep_k = offs.to(tl.float32) < kk.to(tl.float32)
        denom += tl.sum(tl.where(keep_k, tl.exp(values - row_max), 0.0), axis=0)

    offs = (NUM_CHUNKS - 1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < topk_len
    values = tl.load(
        sorted_values_ptr + in_start + offs,
        mask=valid,
        other=NEG_INF,
    ).to(tl.float32)
    keep_k = offs.to(tl.float32) < kk.to(tl.float32)
    denom += tl.sum(tl.where(keep_k, tl.exp(values - row_max), 0.0), axis=0)
    inv_denom = 1.0 / denom

    # pass 2: cumulative mass -> kept-prefix length.
    write_num = 0
    cumulative = 0.0
    for c in tl.range(0, NUM_CHUNKS - 1):
        offs = c * BLOCK_N + tl.arange(0, BLOCK_N)
        values = tl.load(sorted_values_ptr + in_start + offs).to(tl.float32)
        keep = offs.to(tl.float32) < kk.to(tl.float32)
        probs = tl.where(keep, tl.exp(values - row_max) * inv_denom, 0.0)
        inclusive = tl.cumsum(probs, axis=0) + cumulative
        cumulative = al.get_element(inclusive, (BLOCK_N - 1,))
        keep &= ((inclusive - probs) <= pval) | (offs == 0)
        write_num += tl.sum(keep.to(tl.int32), axis=0)

    offs = (NUM_CHUNKS - 1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < topk_len
    values = tl.load(
        sorted_values_ptr + in_start + offs,
        mask=valid,
        other=NEG_INF,
    ).to(tl.float32)
    keep = offs.to(tl.float32) < kk.to(tl.float32)
    probs = tl.where(keep, tl.exp(values - row_max) * inv_denom, 0.0)
    inclusive = tl.cumsum(probs, axis=0) + cumulative
    keep &= ((inclusive - probs) <= pval) | (offs == 0)
    write_num += tl.sum(keep.to(tl.int32), axis=0)
    tl.store(write_nums_ptr + row, write_num.to(tl.int32))


@triton.jit
def _scatter_kernel(
    sorted_values_ptr,  # [B, L] DESCENDING sorted top-L values
    sorted_indices_ptr,  # [B, L] int32 original vocab indices
    write_nums_ptr,  # [B] int32 kept-prefix length per row
    out_ptr,  # [B, V] pre-filled (clone for full-vocab, -inf for sparse)
    vocab_size,  # V
    topk_len,  # L
    FULL_VOCAB: tl.constexpr,
    SCATTER_CORES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    # The output is initialized to -inf before this launch. Split the kept
    # sorted prefix [0, write_num) across scatter cores so every kept element is
    # loaded and written exactly once; no cross-program synchronization is needed.
    row = tl.program_id(0).to(tl.int64)
    sc = tl.program_id(1)
    in_start = row * topk_len
    out_start = row * vocab_size
    write_num = tl.load(write_nums_ptr + row)

    per_core = (write_num + SCATTER_CORES - 1) // SCATTER_CORES
    core_start = sc * per_core
    core_end = core_start + per_core
    if core_end > write_num:
        core_end = write_num
    if core_start < core_end:
        n_tiles = (core_end - core_start + BLOCK_N - 1) // BLOCK_N
        for t in tl.range(0, n_tiles):
            offs = core_start + t * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = offs < core_end
            si = tl.load(sorted_indices_ptr + in_start + offs, mask=valid, other=0)
            sv = tl.load(
                sorted_values_ptr + in_start + offs,
                mask=valid,
                other=NEG_INF,
            ).to(tl.float32)
            tl.store(
                out_ptr + out_start + si.to(tl.int64),
                sv.to(out_ptr.dtype.element_ty),
                mask=valid,
            )


def _p_is_one(p: torch.Tensor) -> bool:
    return bool(torch.all(p == 1).item())


def top_k_top_p(
    logits: torch.Tensor,
    p: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    *,
    topk_len_override: int | None = None,
    p_is_one_override: bool | None = None,
) -> torch.Tensor:
    """Triton-Ascend ``npu_apply_top_k_top_p``.

    Args:
        logits: ``[batch, vocab]`` float16/bfloat16/float32 tensor.
        p: optional ``[batch]`` top-p thresholds (cast to float internally).
        k: optional ``[batch]`` int32 top-k counts.

    Returns:
        ``[batch, vocab]`` tensor with masked positions set to ``-inf`` and kept
        positions holding their original logit values, in original vocab order.
    """
    if not HAS_CANN_EXT:
        raise NotImplementedError(
            "ascend top_k_top_p requires the triton CANN extension "
            "(triton.language.extra.cann.extension)"
        )
    if p is None and k is None:
        raise ValueError("p and k cannot both be None")
    if logits.dim() != 2:
        raise ValueError("logits must be 2D [batch, vocab]")

    batch, vocab = logits.shape

    # Fast path: nothing to filter when the vocab dimension is 1.
    if vocab == 1:
        return logits.clone()

    has_p = p is not None
    has_k = k is not None

    if has_k:
        if k.shape != (batch,):
            raise ValueError("k must have shape [batch]")
        k = k.contiguous().to(torch.int32)
        if topk_len_override is None:
            valid_sparse_k = (k >= 1) & (k <= min(1024, vocab))
            valid_full_k = k == vocab
            if not bool((valid_sparse_k | valid_full_k).all()):
                raise ValueError("each k must be in [1, 1024] or equal vocab_size")
            topk_len = int(k.max().item())  # host sync
            if not has_p and bool((k == vocab).all()):
                return logits.clone()
        else:
            topk_len = int(topk_len_override)
            if topk_len > vocab:
                raise ValueError("topk_len_override must be <= vocab_size")
            if topk_len != vocab and not 1 <= topk_len <= 1024:
                raise ValueError(
                    "topk_len_override must be in [1, 1024] or equal vocab_size"
                )
    else:
        if topk_len_override is not None and int(topk_len_override) != vocab:
            raise ValueError(
                "p-only filtering requires topk_len_override == vocab_size"
            )
        topk_len = vocab

    if has_p:
        if p.shape != (batch,):
            raise ValueError("p must have shape [batch]")
        p = p.contiguous().to(torch.float32)
        p_is_one = (
            bool(p_is_one_override) if p_is_one_override is not None else _p_is_one(p)
        )
        if not has_k and p_is_one:
            return logits.clone()

    full_vocab = topk_len == vocab
    sorted_values, sorted_indices = topk(
        logits.contiguous().to(torch.float32), topk_len
    )

    if not has_p:
        p = torch.empty(1, dtype=torch.float32, device=logits.device)  # dummy
    if not has_k:
        k = torch.empty(1, dtype=torch.int32, device=logits.device)  # dummy

    block_n = min(_BLOCK_N, triton.next_power_of_2(topk_len))
    num_chunks = triton.cdiv(topk_len, block_n)

    # Small-batch scatter fan-out: when the batch does not fill the cores, a
    # single-program-per-row launch leaves most cores idle while the scatter
    # (the memory-bound part) runs serially per row. In that regime, mirror the
    # CANN operator's two-phase design and split each row's scatter region
    # across ``scatter_cores`` cores. For batch >= core count the original fused
    # single-kernel path is already core-saturated, so keep it unchanged.
    scatter_cores = 1
    vector_core_num = _get_aiv_num()
    if batch < vector_core_num:
        scatter_cores = max(1, vector_core_num // batch)

    if scatter_cores == 1:
        # Single-kernel path: full-vocab starts from logits (clone) and knocks
        # out the removed side; sparse starts fully masked.
        out = logits.clone() if full_vocab else torch.full_like(logits, _NEG_INF)
        grid = (batch,)
        _apply_top_k_top_p_kernel[grid](
            sorted_values,
            sorted_indices,
            k,
            p,
            out,
            vocab,
            topk_len,
            HAS_K=has_k,
            HAS_P=has_p,
            FULL_VOCAB=full_vocab,
            BLOCK_N=block_n,
            NUM_CHUNKS=num_chunks,
            NEG_INF=_NEG_INF,
        )
        return out

    write_nums = torch.empty(batch, dtype=torch.int32, device=logits.device)
    p_only_parallel = has_p and not has_k and full_vocab
    if p_only_parallel:
        compute_shards = max(1, vector_core_num // batch)
        compute_shards = min(compute_shards, num_chunks)
        shard_len = triton.cdiv(topk_len, compute_shards)
        shard_tiles = triton.cdiv(shard_len, block_n)
        scan_n = triton.next_power_of_2(compute_shards)
        partial_sums = torch.empty(
            (batch, compute_shards), dtype=torch.float32, device=logits.device
        )
        _compute_p_only_partials_kernel[(batch, compute_shards)](
            sorted_values,
            partial_sums,
            topk_len,
            SHARDS=compute_shards,
            SHARD_LEN=shard_len,
            BLOCK_N=block_n,
            SHARD_TILES=shard_tiles,
            NEG_INF=_NEG_INF,
        )
        _scan_finalize_p_only_write_num_kernel[(batch,)](
            partial_sums,
            p,
            sorted_values,
            write_nums,
            topk_len,
            SHARDS=compute_shards,
            SHARD_LEN=shard_len,
            SCAN_N=scan_n,
            BLOCK_N=block_n,
            SHARD_TILES=shard_tiles,
            NEG_INF=_NEG_INF,
        )
    else:
        _compute_write_num_kernel[(batch,)](
            sorted_values,
            k,
            p,
            write_nums,
            topk_len,
            HAS_K=has_k,
            HAS_P=has_p,
            BLOCK_N=block_n,
            NUM_CHUNKS=num_chunks,
            NEG_INF=_NEG_INF,
        )

    # The multi-core scatter always writes the kept prefix directly, so initialize
    # the complete output to -inf for both sparse and full-vocab cases.
    out = torch.full_like(logits, _NEG_INF)

    # Phase 2: ``scatter_cores`` cores per row cooperatively scatter the chosen
    # region. 2D grid: axis 0 == batch (row), axis 1 == scatter core (sc). Work
    # is split at element granularity (ceil(region_len / scatter_cores) per core),
    # looping over each core's span in BLOCK_N tiles.
    _scatter_kernel[(batch, scatter_cores)](
        sorted_values,
        sorted_indices,
        write_nums,
        out,
        vocab,
        topk_len,
        FULL_VOCAB=full_vocab,
        SCATTER_CORES=scatter_cores,
        BLOCK_N=block_n,
        NEG_INF=_NEG_INF,
    )
    return out

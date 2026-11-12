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

"""Candidate-block selection for DeepSeek-V4.1 decode rows.

Triton port of vLLM's
``vllm/models/deepseek_v41/nvidia/specialized_decode/attention/candidate_topk.py``
(``select_candidate_blocks_cooperative``). The vLLM version runs three steps:

1. ``_block_scores_kernel`` -- per-block max of the logits row (NaN
   propagating), pinning the block of the newest token to ``+inf``, writing
   each row's valid block count and pre-filling ``out`` with ``-1``.
2. ``torch.ops._C.cooperative_topk`` -- CUDA top-k over the valid prefix.
3. ``_finalize_candidates_kernel`` -- drop out-of-range or ``-inf`` ids. Only
   needed for ``cooperative_topk`` (which may leave stale/overflow slots);
   skipped for ``persistent_topk``, which already emits only valid ids and
   ``-1`` padding.

Steps 1 and 3 are ported directly. Step 2 calls this repo's Triton top-k:
``flaggems_vllm.persistent_topk`` (preferred -- currently the faster of the two
Triton backends on NVIDIA), falling back to ``flaggems_vllm.cooperative_topk``.
vLLM's own indexer treats the two as interchangeable backends (same arguments,
same "top-k of the first ``lengths[i]`` entries, ``-1`` padded" contract), so
the selected set matches vLLM up to ties at the selection boundary, which the
vLLM docstring already declares approximate. The backend is resolved at call
time so vendor backends that override either op pick up their own version.

Output rows are unsorted and ``-1`` padding may sit anywhere; consumers treat
each row as a set.
"""

import torch
import triton
import triton.language as tl

# vLLM: vllm/model_executor/layers/indexer_topk.py
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
AUTO_COOPERATIVE_MAX_ROWS = 64
SUPPORTED_K = (512, 1024, 2048)

_FINALIZE_TILE = 1024


def _pick_score_tile(nblocks: int):
    """Fixed launch config for the (memory-bound) score kernel, by block count.

    A cheap ``nblocks`` heuristic reproduces the autotuned configs' GPU time
    while removing the per-call autotune key/lookup host cost -- which, on the
    eager decode path, is a measurable share of the op's launch overhead.
    """
    if nblocks >= 32768:
        return 1024, 8
    if nblocks >= 4096:
        return 512, 8
    if nblocks >= 512:
        return 256, 8
    return 128, 4


# Per-(device, rows, nblocks) scratch for block scores and valid-block counts,
# reused across calls to drop two ``torch.empty`` allocations per call on the hot
# decode path. The score kernel fully overwrites the ``[0, limit)`` prefix each
# call and the top-k only reads that prefix, so reuse never exposes stale data.
_SCRATCH_CACHE: dict = {}


def _get_scratch(rows: int, nblocks: int, device: torch.device):
    key = (device, rows, nblocks)
    buf = _SCRATCH_CACHE.get(key)
    if buf is None:
        scores = torch.empty(
            (rows, max(nblocks, 1)), dtype=torch.float32, device=device
        )
        lengths = torch.empty(rows, dtype=torch.int32, device=device)
        buf = (scores, lengths)
        _SCRATCH_CACHE[key] = buf
    return buf


def _resolve_block_topk():
    """The row-prefix top-k backend, resolved at call time.

    vLLM's ``select_candidate_blocks_cooperative`` names ``cooperative_topk``,
    but vLLM's own indexer treats ``cooperative_topk`` and ``persistent_topk``
    as interchangeable backends (same signature, same "top-k of each row's
    first ``lengths[i]`` scores, ``-1`` padded" contract). This repo's Triton
    ``persistent_topk`` is currently several times faster than its Triton
    ``cooperative_topk`` on NVIDIA, so it is preferred here, with
    ``cooperative_topk`` as the fallback for backends that only ship that one.
    Resolving the top-level attribute late lets vendor backends override either.
    """
    import flaggems_vllm

    topk = getattr(flaggems_vllm, "persistent_topk", None)
    if topk is not None:
        # persistent_topk only emits in-range indices and pads short rows with
        # -1 itself, so the finalize pass is a proven no-op for it and is
        # skipped. cooperative_topk may leave stale/overflow slots, so it keeps
        # finalize.
        return topk, False
    return flaggems_vllm.cooperative_topk, True


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    ends,
    scores,
    lengths,
    out,
    stride_row,
    stride_col,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    end = tl.load(ends + row)
    # Blocks from cdiv(end, BLOCK_SIZE) on score -inf and are never pinned, so
    # the top-k only reads (and we only store) this prefix of the row.
    limit = tl.minimum((tl.maximum(end, 0) + BLOCK_SIZE - 1) // BLOCK_SIZE, nblocks)
    if tl.program_id(1) == 0:
        tl.store(lengths + row, limit.to(tl.int32))
        # The top-k may leave slots unwritten; they must read -1, never a
        # stale block id.
        for tile in range(0, K, TILE):
            slots = tile + tl.arange(0, TILE)
            tl.store(out + row * K + slots, -1, slots < K)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols < end)
        & (cols < width),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    # Pin the block holding the newest token.
    reduced = tl.where(
        (end > 0) & (blocks == (end - 1) // BLOCK_SIZE), float("inf"), reduced
    )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < limit)


@triton.jit(do_not_specialize=["nblocks"])
def _finalize_candidates_kernel(
    out,
    scores,
    lengths,
    nblocks,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    index = tl.load(out + row * K + cols, cols < K, other=-1)
    valid = (index >= 0) & (index < tl.load(lengths + row))
    value = tl.load(
        scores + row * nblocks + index, (cols < K) & valid, other=-float("inf")
    )
    # Only -inf blocks become padding, as in vLLM's select_candidate_blocks.
    tl.store(
        out + row * K + cols,
        tl.where(valid & (value != -float("inf")), index, -1),
        cols < K,
    )


def supports_cooperative_candidates(num_rows: int, num_blocks: int, k: int) -> bool:
    """Whether ``select_candidate_blocks_cooperative`` takes ``num_rows`` rows of
    ``num_blocks`` blocks and ``k`` candidates (same bounds as vLLM)."""
    return (
        0 < num_rows <= AUTO_COOPERATIVE_MAX_ROWS
        and k in SUPPORTED_K
        # Score rows must stay 16-byte aligned for the vectorized top-k loads.
        and num_blocks % 4 == 0
    )


def _check_inputs(logits, row_ends, block_size, out):
    if logits.dim() != 2:
        raise ValueError("select_candidate_blocks_cooperative: logits must be 2D")
    if row_ends.dim() != 1 or not row_ends.is_contiguous():
        raise ValueError(
            "select_candidate_blocks_cooperative: row_ends must be contiguous 1D"
        )
    if out.dim() != 2 or not out.is_contiguous():
        raise ValueError(
            "select_candidate_blocks_cooperative: out must be contiguous 2D"
        )
    if not (logits.device == row_ends.device == out.device):
        raise ValueError(
            "select_candidate_blocks_cooperative: tensors must share one device"
        )
    if logits.dtype != torch.float32:
        raise NotImplementedError(
            f"select_candidate_blocks_cooperative: logits must be float32, got {logits.dtype}"
        )
    if row_ends.dtype != torch.int32:
        raise NotImplementedError(
            "select_candidate_blocks_cooperative: row_ends must be int32"
        )
    if out.dtype != torch.int32:
        raise NotImplementedError(
            "select_candidate_blocks_cooperative: out must be int32"
        )
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(
            "select_candidate_blocks_cooperative: block_size must be a positive int"
        )
    rows = logits.shape[0]
    if row_ends.shape[0] != rows or out.shape[0] != rows:
        raise ValueError(
            "select_candidate_blocks_cooperative: row counts of logits/row_ends/out differ"
        )
    if logits.stride(0) < 0 or logits.stride(1) < 0:
        raise NotImplementedError(
            "select_candidate_blocks_cooperative: negative logits strides"
        )


def select_candidate_blocks_cooperative(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    block_size: int,
    out: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> None:
    """vLLM-compatible ``select_candidate_blocks_cooperative`` for decode rows.

    Args:
        logits: [rows, width] fp32 logits; row ``i`` is valid below
            ``row_ends[i]``. Any non-negative strides.
        row_ends: Contiguous [rows] int32 row lengths in logits columns.
        block_size: Columns per candidate block.
        out: Contiguous [rows, k] int32 candidate block ids, ``-1`` padded,
            unsorted; see ``supports_cooperative_candidates`` for the shapes.
        workspace: Optional uint8 top-k scratch buffer of at least
            ``RADIX_TOPK_WORKSPACE_SIZE`` bytes (vLLM passes the workspace
            manager's buffer). Allocated when omitted.
    """
    _check_inputs(logits, row_ends, block_size, out)
    rows, width = logits.shape
    k = out.shape[1]
    nblocks = triton.cdiv(width, block_size)
    if not supports_cooperative_candidates(rows, nblocks, k):
        raise NotImplementedError(
            "select_candidate_blocks_cooperative supports 0 < rows <= "
            f"{AUTO_COOPERATIVE_MAX_ROWS}, k in {SUPPORTED_K} and nblocks % 4 == 0; "
            f"got rows={rows}, nblocks={nblocks}, k={k}"
        )
    if workspace is not None and (
        workspace.dtype != torch.uint8
        or workspace.numel() < RADIX_TOPK_WORKSPACE_SIZE
        or workspace.device != logits.device
    ):
        raise ValueError(
            "select_candidate_blocks_cooperative: workspace must be a uint8 buffer of "
            f">= {RADIX_TOPK_WORKSPACE_SIZE} bytes on the logits device"
        )

    scores, lengths = _get_scratch(rows, nblocks, logits.device)

    tile, num_warps = _pick_score_tile(nblocks)
    grid = (rows, max(1, triton.cdiv(nblocks, tile)))
    _block_scores_kernel[grid](
        logits,
        row_ends,
        scores,
        lengths,
        out,
        logits.stride(0),
        logits.stride(1),
        width,
        nblocks,
        BLOCK_SIZE=block_size,
        K=k,
        TILE=tile,
        num_warps=num_warps,
    )
    if nblocks == 0:
        # Empty rows: out is already all -1.
        return

    if workspace is None:
        workspace = torch.empty(
            RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=logits.device
        )
    # Rows of at most k blocks take all of them and pad with -1. nblocks bounds
    # every row's length (lengths <= nblocks by construction).
    topk, needs_finalize = _resolve_block_topk()
    topk(scores, lengths, out, workspace, k, nblocks)

    if needs_finalize:
        _finalize_candidates_kernel[(rows, triton.cdiv(k, _FINALIZE_TILE))](
            out, scores, lengths, nblocks, k, _FINALIZE_TILE
        )

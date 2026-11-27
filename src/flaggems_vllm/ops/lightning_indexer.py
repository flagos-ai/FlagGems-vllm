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

"""Pure Triton TND LightningIndexer and helpers shared by backend implementations."""

import os
from itertools import product

import torch
import triton
import triton.language as tl
from triton.language.standard import _log2

_INT64_MAX = 9223372036854775807


@triton.jit(do_not_specialize=["OFFSET"])
def _copy_strided(
    src,
    dst,
    OFFSET: tl.int64,
    N: tl.int64,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    S0: tl.int64,
    S1: tl.int64,
    S2: tl.int64,
    S3: tl.int64,
    BLOCK: tl.constexpr,
):
    """Copy an arbitrary up-to-4D strided view into contiguous storage."""
    ix = OFFSET + tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    i3 = ix % D3
    q = ix // D3
    i2 = q % D2
    q = q // D2
    i1 = q % D1
    i0 = q // D1
    values = tl.load(src + i0 * S0 + i1 * S1 + i2 * S2 + i3 * S3, ix < N, 0)
    tl.store(dst + ix, values, ix < N)


def _contiguous_tensor(x):
    """Return a contiguous view of ``x`` using a pure-Triton copy if needed."""
    if x.is_contiguous():
        return x
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    total = x.numel()
    if total:
        shape = (1,) * (4 - x.ndim) + tuple(x.shape)
        stride = (0,) * (4 - x.ndim) + tuple(x.stride())
        _copy_strided[(triton.cdiv(total, 1024),)](
            x, out, 0, total, *shape[1:], *stride, BLOCK=1024
        )
    return out


def launch_plain_tasks(kernel, axes, args):
    """Launch a 1D/2D kernel in grid chunks staying within 65535 total blocks."""
    remaining = 65535
    sizes = []
    for extent in reversed(axes):
        size = min(extent, remaining)
        sizes.append(size)
        remaining //= size
    sizes.reverse()
    starts = (range(0, extent, size) for extent, size in zip(axes, sizes))
    for offsets in product(*starts):
        grid = tuple(
            min(size, extent - offset)
            for extent, size, offset in zip(axes, sizes, offsets)
        )
        kernel[grid](*args, *offsets)


@triton.jit(do_not_specialize=["row_base"])
def _row_bounds(
    QEnd,
    KLen,
    Starts,
    Ends,
    B: tl.constexpr,
    MAXK: tl.constexpr,
    SPARSE_MODE: tl.constexpr = 3,
    row_base=0,
):
    """Compute per-row visible key count under causal (mode 3) or full (mode 0) masking."""
    row = tl.program_id(0) + row_base
    req = 0
    end = tl.load(QEnd)
    while req < B - 1 and row >= end:
        req += 1
        end = tl.load(QEnd + req)
    k_len = tl.load(KLen + req)
    if SPARSE_MODE == 3:
        raw_valid = k_len - (end - row) + 1
    else:
        raw_valid = k_len
    valid = tl.maximum(0, tl.minimum(MAXK, raw_valid))
    tl.store(Starts + row, 0)
    tl.store(Ends + row, valid)


def compute_row_bounds(ends, lengths, starts, row_ends, b, max_keys, sparse_mode=3):
    """Populate per-row ``starts`` (0) and ``row_ends`` (visible key counts)."""
    rows = starts.numel()
    if not rows:
        return
    launch_plain_tasks(
        _row_bounds,
        (rows,),
        (ends, lengths, starts, row_ends, b, max_keys, sparse_mode),
    )


@triton.jit
def _direct_indices(
    QEnd,
    KLen,
    Out,
    B: tl.constexpr,
    SPARSE_COUNT: tl.constexpr = 512,
    SPARSE_MODE: tl.constexpr = 3,
    CHUNK: tl.constexpr = 512,
):
    """Emit sequential indices ``0..valid-1`` padded with ``-1`` for short rows."""
    row = tl.program_id(0)
    req = 0
    end = tl.load(QEnd)
    while req < B - 1 and row >= end:
        req += 1
        end = tl.load(QEnd + req)
    k_len = tl.load(KLen + req)
    if SPARSE_MODE == 3:
        valid = tl.maximum(0, k_len - (end - row) + 1)
    else:
        valid = tl.maximum(0, k_len)
    row_out = Out + row.to(tl.int64) * SPARSE_COUNT
    for chunk_idx in tl.static_range((SPARSE_COUNT + CHUNK - 1) // CHUNK):
        ix = chunk_idx * CHUNK + tl.arange(0, CHUNK)
        tl.store(
            row_out + ix,
            tl.where(ix < valid, ix, -1),
            mask=ix < SPARSE_COUNT,
        )


@triton.jit
def _fill_empty_outputs(
    OutIndices,
    OutValues,
    TOTAL_ELEMS: tl.int64,
    RETURN_VALUE: tl.constexpr,
    BLOCK: tl.constexpr = 1024,
):
    """Fill outputs with ``-1`` indices (and ``-inf`` values when requested)."""
    ix = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = ix < TOTAL_ELEMS
    tl.store(OutIndices + ix, -1, mask=mask)
    if RETURN_VALUE:
        neg_inf = tl.full([BLOCK], -float("inf"), dtype=tl.float32).to(
            OutValues.dtype.element_ty
        )
        tl.store(OutValues + ix, neg_inf, mask=mask)


def fill_empty_outputs(out_indices, out_values, return_value):
    """Fill all output positions when no active keys exist."""
    total = out_indices.numel()
    if not total:
        return
    val_ptr = out_values if return_value else out_indices
    _fill_empty_outputs[(triton.cdiv(total, 1024),)](
        out_indices, val_ptr, total, RETURN_VALUE=return_value, BLOCK=1024
    )


@triton.jit
def _better(a, ia, b, ib):
    """Total descending order on ``(score, index)`` with NaN first and smaller index winning ties."""
    an = (a.to(tl.int32, bitcast=True) & 0x7FFFFFFF) > 0x7F800000
    bn = (b.to(tl.int32, bitcast=True) & 0x7FFFFFFF) > 0x7F800000
    return (an & ~bn) | ((an == bn) & ((a > b) | (((a == b) | (an & bn)) & (ia < ib))))


@triton.jit(do_not_specialize=["WIDTH", "GROUPS", "row_base", "group_base"])
def _sort_score_tiles(
    Scores,
    Ends,
    Values,
    Indices,
    WIDTH: tl.int64,
    GROUPS,
    TILE: tl.constexpr,
    row_base,
    group_base,
):
    """Sort 64-key score tiles in descending order via pairwise rank counting."""
    row = tl.program_id(0) + row_base
    group = tl.program_id(1) + group_base
    valid = tl.load(Ends + row)
    col = group * TILE + tl.arange(0, TILE)
    active = col < valid
    score = tl.load(
        Scores + row.to(tl.int64) * WIDTH + col, active, other=-float("inf")
    )
    before = _better(score[None, :], col[None, :], score[:, None], col[:, None])
    rank = tl.sum((before & active[None, :]).to(tl.int32), 1)
    offset = (row.to(tl.int64) * GROUPS + group) * TILE + rank
    tl.store(Values + offset, score, active)
    tl.store(Indices + offset, col, active)


@triton.jit(
    do_not_specialize=["GROUPS", "NEXT_GROUPS", "SPAN", "row_base", "task_base"]
)
def _merge_score_tiles(
    V,
    I,
    OutV,
    OutI,
    Ends,
    GROUPS,
    NEXT_GROUPS,
    SPAN,
    KEEP: tl.constexpr,
    NEXT_KEEP: tl.constexpr,
    TILE: tl.constexpr,
    SKIP_OUT_VALUES: tl.constexpr,
    row_base,
    task_base,
):
    """Two-way binary-search merge of sorted score/index lists, keeping top ``NEXT_KEEP``."""
    row = tl.program_id(0) + row_base
    task = tl.program_id(1) + task_base
    chunks: tl.constexpr = KEEP // TILE
    group = task // chunks
    lane = task % chunks * TILE + tl.arange(0, TILE)
    valid = tl.load(Ends + row)
    row_base_offset = row.to(tl.int64) * GROUPS
    count = tl.minimum(KEEP, tl.maximum(0, valid - group * SPAN))
    active = lane < count
    offset = (row_base_offset + group) * KEEP + lane
    score = tl.load(V + offset, active, other=-float("inf"))
    index = tl.load(I + offset, active, other=2147483647)
    mate = group ^ 1
    has_mate = mate < GROUPS
    safe_mate = tl.where(has_mate, mate, 0)
    mate_count = tl.where(
        has_mate, tl.minimum(KEEP, tl.maximum(0, valid - mate * SPAN)), 0
    )
    mate_base = (row_base_offset + safe_mate) * KEEP
    lo = tl.full((TILE,), 0, tl.int32)
    hi = tl.full((TILE,), mate_count, tl.int32)
    for _ in range(_log2(KEEP) + 1):
        mid = (lo + hi) // 2
        search = active & (lo < hi)
        other_score = tl.load(V + mate_base + mid, search, other=-float("inf"))
        other_index = tl.load(I + mate_base + mid, search, other=2147483647)
        before = _better(other_score, other_index, score, index)
        lo = tl.where(search & before, mid + 1, lo)
        hi = tl.where(search & ~before, mid, hi)
    rank = lane + lo
    output = (row.to(tl.int64) * NEXT_GROUPS + group // 2) * NEXT_KEEP + rank
    tl.store(OutI + output, index, active & (rank < NEXT_KEEP))
    if not SKIP_OUT_VALUES:
        tl.store(OutV + output, score, active & (rank < NEXT_KEEP))


@triton.jit(do_not_specialize=["row_base"])
def _finish_selected(
    Indices,
    Values,
    Ends,
    OutIndices,
    OutValues,
    KEEP: tl.constexpr,
    SPARSE_COUNT: tl.constexpr,
    CHUNK: tl.constexpr,
    RETURN_VALUE: tl.constexpr,
    row_base,
):
    """Copy top ``SPARSE_COUNT`` entries into output buffers with ``-1`` / ``-inf`` padding."""
    row = tl.program_id(0) + row_base
    valid = tl.load(Ends + row)
    row_in = row.to(tl.int64) * KEEP
    row_out = row.to(tl.int64) * SPARSE_COUNT
    for chunk_idx in tl.static_range((SPARSE_COUNT + CHUNK - 1) // CHUNK):
        pos = chunk_idx * CHUNK + tl.arange(0, CHUNK)
        in_mask = (pos < KEEP) & (pos < valid) & (pos < SPARSE_COUNT)
        out_mask = pos < SPARSE_COUNT
        selected_idx = tl.load(Indices + row_in + pos, mask=in_mask, other=-1)
        tl.store(OutIndices + row_out + pos, selected_idx, mask=out_mask)
        if RETURN_VALUE:
            selected_val = tl.load(
                Values + row_in + pos, mask=in_mask, other=-float("inf")
            ).to(OutValues.dtype.element_ty)
            tl.store(OutValues + row_out + pos, selected_val, mask=out_mask)


def select_topk_sorted(
    scores, row_ends, out_indices, out_values, *, sparse_count, return_value
):
    """Pure-Triton exact descending TopK selection via tile sort and binary merge tree."""
    rows, width = scores.shape
    tile = 64
    keep_target = max(tile, triton.next_power_of_2(sparse_count))
    span = keep = tile
    groups = triton.cdiv(width, tile)
    device = scores.device
    values = torch.empty((rows, groups, keep), dtype=torch.float32, device=device)
    indices = torch.empty((rows, groups, keep), dtype=torch.int32, device=device)
    launch_plain_tasks(
        _sort_score_tiles,
        (rows, groups),
        (scores, row_ends, values, indices, width, groups, tile),
    )
    while groups > 1:
        next_groups = triton.cdiv(groups, 2)
        next_keep = min(keep_target, 2 * keep)
        skip_out_values = (next_groups == 1) and (not return_value)
        shape = (rows, next_groups, next_keep)
        next_indices = torch.empty(shape, dtype=torch.int32, device=device)
        next_values = (
            values
            if skip_out_values
            else torch.empty(shape, dtype=torch.float32, device=device)
        )
        launch_plain_tasks(
            _merge_score_tiles,
            (rows, groups * (keep // tile)),
            (
                values,
                indices,
                next_values,
                next_indices,
                row_ends,
                groups,
                next_groups,
                span,
                keep,
                next_keep,
                tile,
                skip_out_values,
            ),
        )
        groups, span, keep = next_groups, 2 * span, next_keep
        values, indices = next_values, next_indices

    chunk = min(512, max(16, triton.next_power_of_2(sparse_count)))
    val_out_ptr = out_values if return_value else out_indices
    launch_plain_tasks(
        _finish_selected,
        (rows,),
        (
            indices,
            values,
            row_ends,
            out_indices,
            val_out_ptr,
            keep,
            sparse_count,
            chunk,
            return_value,
        ),
    )


def _validate_lightning_indexer_inputs(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3,
    pre_tokens=_INT64_MAX,
    next_tokens=_INT64_MAX,
    return_value=False,
):
    """Validate TND LightningIndexer arguments and normalize contiguous metadata."""
    version = os.environ.get("GET_FLAGGEMS_VERSION", "v1")
    if version not in {"v0", "v1", "v2"}:
        raise ValueError("GET_FLAGGEMS_VERSION must be v0, v1 or v2")
    if layout_query != "TND":
        raise NotImplementedError("layout_query only supports TND")
    if layout_key not in {"PA_BSND", "TND"}:
        raise NotImplementedError("layout_key only supports PA_BSND or TND")

    is_paged = layout_key == "PA_BSND"
    if (
        query is None
        or key is None
        or weights is None
        or actual_seq_lengths_query is None
        or actual_seq_lengths_key is None
        or (is_paged and block_table is None)
    ):
        raise ValueError(
            "query/key/weights, sequence lengths and block_table are required"
        )
    if not is_paged and block_table is not None:
        raise ValueError("block_table must be None when layout_key is TND")

    dev = query.device
    if dev.type != "cuda":
        raise NotImplementedError(
            "the generic LightningIndexer requires a CUDA-compatible GPU"
        )
    if (
        key.device != dev
        or weights.device != dev
        or actual_seq_lengths_query.device != dev
        or actual_seq_lengths_key.device != dev
        or (is_paged and block_table.device != dev)
    ):
        raise ValueError("all inputs must be on the same device")

    q_dtype = query.dtype
    if (q_dtype != torch.float16 and q_dtype != torch.bfloat16) or key.dtype != q_dtype:
        raise TypeError("query and key must have matching FP16 or BF16 dtype")
    if weights.dtype != q_dtype and weights.dtype != torch.float32:
        raise TypeError("weights must match query dtype or be FP32")
    if (
        actual_seq_lengths_query.dtype != torch.int32
        or actual_seq_lengths_key.dtype != torch.int32
        or (is_paged and block_table.dtype != torch.int32)
    ):
        raise TypeError("endpoints, lengths and block_table must be int32")

    if query.ndim != 3 or weights.ndim != 2:
        raise ValueError("expected Q[T1,N1,D] and W[T1,N1]")
    t, h, d = query.shape
    if weights.shape != (t, h):
        raise ValueError("incompatible query/weights dimensions")
    if not (1 <= h <= 64 and d == 128):
        raise ValueError("expected 1 <= query_heads <= 64 and head_dim == 128")

    if is_paged:
        if key.ndim != 4:
            raise ValueError("expected K[BlockCount,BlockSize,1,D] for PA_BSND")
        block_count, block_size, k_heads, k_dim = key.shape
        if k_heads != 1 or k_dim != d:
            raise ValueError("incompatible query/key dimensions")
        if (
            block_count <= 0
            or block_size <= 0
            or block_size > 1024
            or block_size % 16 != 0
        ):
            raise ValueError(
                "PA_BSND key requires block_count > 0 and block_size in (0, 1024] divisible by 16"
            )
    else:
        if key.ndim != 3:
            raise ValueError("expected K[T2,1,D] for TND")
        if key.shape[1] != 1 or key.shape[2] != d:
            raise ValueError("incompatible query/key dimensions")

    if (
        actual_seq_lengths_query.ndim != 1
        or actual_seq_lengths_key.ndim != 1
        or (is_paged and block_table.ndim != 2)
    ):
        raise ValueError("invalid metadata rank")
    if actual_seq_lengths_query.shape != actual_seq_lengths_key.shape or (
        is_paged and block_table.shape[0] != actual_seq_lengths_key.numel()
    ):
        raise ValueError("metadata batch sizes differ")
    if t and not actual_seq_lengths_key.numel():
        raise ValueError("nonempty query requires at least one request")

    if not isinstance(return_value, bool):
        raise TypeError("return_value must be bool")
    if return_value and is_paged:
        raise NotImplementedError(
            "return_value=True is not supported when layout_key='PA_BSND'"
        )
    if (
        isinstance(sparse_mode, bool)
        or not isinstance(sparse_mode, int)
        or sparse_mode not in {0, 3}
    ):
        raise NotImplementedError("sparse_mode must be 0 or 3")
    if pre_tokens != _INT64_MAX or next_tokens != _INT64_MAX:
        raise NotImplementedError("finite windows are not supported")
    if isinstance(sparse_count, bool) or not isinstance(sparse_count, int):
        raise TypeError("sparse_count must be int")
    if sparse_count <= 0:
        raise ValueError("sparse_count must be positive")
    if sparse_count > 8192 or (sparse_count > 2048 and sparse_count % 1024 != 0):
        raise NotImplementedError(
            "sparse_count must be in [1, 2048] or a multiple of 1024 in (2048, 8192]"
        )

    # Validate sequence lengths and page capacity before launching any kernel.
    host_ends = actual_seq_lengths_query.cpu().tolist()
    raw_key_meta = actual_seq_lengths_key.cpu().tolist()
    previous_q = 0
    active_max = 0
    if is_paged:
        capacity = key.shape[1] * block_table.shape[1]
        for end, length in zip(host_ends, raw_key_meta):
            if end < previous_q or length < 0 or length > capacity:
                raise ValueError(
                    "invalid sequence endpoints, lengths or page-table capacity"
                )
            if end > previous_q and length > active_max:
                active_max = length
            previous_q = end
        host_lengths = raw_key_meta
    else:
        t2 = key.shape[0]
        previous_k = 0
        host_lengths = []
        for end, k_end in zip(host_ends, raw_key_meta):
            if end < previous_q or k_end < previous_k or k_end > t2:
                raise ValueError("invalid TND cumulative sequence endpoints")
            length = k_end - previous_k
            host_lengths.append(length)
            if end > previous_q and length > active_max:
                active_max = length
            previous_q = end
            previous_k = k_end
    if previous_q != t:
        raise ValueError("final cumulative query endpoint must equal T")

    query = _contiguous_tensor(query)
    key = _contiguous_tensor(key)
    weights = _contiguous_tensor(weights)
    ends = _contiguous_tensor(actual_seq_lengths_query)
    key_meta = _contiguous_tensor(actual_seq_lengths_key)
    if is_paged:
        block_table = _contiguous_tensor(block_table)
    return (
        version,
        query,
        key,
        weights,
        ends,
        key_meta,
        block_table,
        active_max,
        host_ends,
        host_lengths,
    )


@triton.jit
def _generic_scores_kernel(
    Q,
    K,
    W,
    QEnd,
    KMeta,
    BT,
    Scores,
    H: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    BTW: tl.constexpr,
    MAXK: tl.constexpr,
    B: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr,
    SPARSE_MODE: tl.constexpr,
    IS_PAGED: tl.constexpr,
):
    """Generic pure-Triton paged/ragged QK-ReLU-weighted scorer."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    req = 0
    end = tl.load(QEnd)
    while req < B - 1 and row >= end:
        req += 1
        end = tl.load(QEnd + req)
    if IS_PAGED:
        k_start = 0
        k_len = tl.load(KMeta + req)
    else:
        k_end = tl.load(KMeta + req)
        k_start = tl.load(KMeta + req - 1) if req > 0 else 0
        k_len = k_end - k_start
    valid = tl.maximum(
        0,
        tl.minimum(MAXK, (k_len - (end - row) + 1) if SPARSE_MODE == 3 else k_len),
    )
    if tile * BK < valid:
        ks = tile * BK + tl.arange(0, BK)
        ds = tl.arange(0, BD)
        hs = tl.arange(0, BH)
        q = tl.load(
            Q + row.to(tl.int64) * H * D + hs[:, None] * D + ds[None, :],
            mask=(hs[:, None] < H) & (ds[None, :] < D),
            other=0,
        )
        if IS_PAGED:
            pages = tl.load(BT + req * BTW + ks // PS, mask=ks < valid, other=0)
            k_rows = pages.to(tl.int64) * PS + (ks % PS)
        else:
            k_rows = k_start.to(tl.int64) + ks
        k = tl.load(
            K + k_rows[None, :] * D + ds[:, None],
            mask=(ks[None, :] < valid) & (ds[:, None] < D),
            other=0,
        )
        qk = tl.dot(q, k)
        w = tl.load(W + row.to(tl.int64) * H + hs, mask=hs < H, other=0).to(tl.float32)
        score = tl.sum(tl.maximum(qk, 0.0) * w[:, None], 0)
        tl.store(
            Scores + row.to(tl.int64) * MAXK + ks,
            tl.where(ks < valid, score, -float("inf")),
            mask=ks < MAXK,
        )


def lightning_indexer(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3,
    pre_tokens=_INT64_MAX,
    next_tokens=_INT64_MAX,
    return_value=False,
):
    """Platform-agnostic pure-Triton TND LightningIndexer (overridden by backend ops)."""
    (
        _version,
        query,
        key,
        weights,
        ends,
        key_meta,
        block_table,
        active_max,
        _host_ends,
        host_lengths,
    ) = _validate_lightning_indexer_inputs(
        query,
        key,
        weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        return_value=return_value,
    )
    t, h, d = query.shape
    out = torch.empty((t, 1, sparse_count), dtype=torch.int32, device=query.device)
    values = (
        torch.empty((t, 1, sparse_count), dtype=query.dtype, device=query.device)
        if return_value
        else torch.empty((0,), dtype=query.dtype, device=query.device)
    )
    if not t:
        return out, values
    if not active_max:
        fill_empty_outputs(out, values, return_value)
        return out, values

    is_paged = layout_key == "PA_BSND"
    b = key_meta.numel()
    lengths = (
        key_meta
        if is_paged
        else torch.tensor(host_lengths, dtype=torch.int32, device=query.device)
    )
    if not return_value and active_max <= sparse_count:
        chunk = min(512, max(16, triton.next_power_of_2(sparse_count)))
        _direct_indices[(t,)](
            ends,
            lengths,
            out,
            b,
            SPARSE_COUNT=sparse_count,
            SPARSE_MODE=sparse_mode,
            CHUNK=chunk,
        )
        return out, values

    tile_k = 128
    tiles = triton.cdiv(active_max, tile_k)
    scores = torch.empty((t, active_max), dtype=torch.float32, device=query.device)
    _generic_scores_kernel[(t, tiles)](
        query,
        key,
        weights,
        ends,
        key_meta,
        block_table if is_paged else ends,
        scores,
        h,
        d,
        key.shape[1] if is_paged else 1,
        block_table.shape[1] if is_paged else 1,
        active_max,
        b,
        max(16, triton.next_power_of_2(h)),
        max(32, triton.next_power_of_2(d)),
        tile_k,
        SPARSE_MODE=sparse_mode,
        IS_PAGED=is_paged,
    )
    starts = torch.empty((t,), dtype=torch.int32, device=query.device)
    row_ends = torch.empty((t,), dtype=torch.int32, device=query.device)
    compute_row_bounds(ends, lengths, starts, row_ends, b, active_max, sparse_mode)
    select_topk_sorted(
        scores,
        row_ends,
        out,
        values,
        sparse_count=sparse_count,
        return_value=return_value,
    )
    return out, values

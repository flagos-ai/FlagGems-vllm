# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import triton
import triton.language as tl
from triton.experimental.tle.language.dsa import insert_slice

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

from .scheduling import _classify_long_requests, classify_mixed_requests

# ---- common ----


@triton.jit
def load_paged_tile(
    X,
    TABLE,
    batch,
    head,
    begin,
    length,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    BN: tl.constexpr,
):
    """Gather affine page segments with 64-bit physical cache addresses."""
    SEG: tl.constexpr = BN if BN < PAGE else PAGE
    tile = tl.zeros((BN, 256), X.dtype.element_ty)
    for start in tl.static_range(0, BN, SEG):
        first = begin + start
        # Clamp even masked page addresses: scalar loads can be speculated.
        page_index = tl.minimum(first // PAGE, tl.maximum(tl.cdiv(length, PAGE) - 1, 0))
        page = tl.load(TABLE + batch.to(tl.int64) * TABLE_STRIDE + page_index)
        base = X + page.to(tl.int64) * PAGE_STRIDE
        base += (first % PAGE) * ROW_STRIDE + head * HEAD_STRIDE
        rows = tl.arange(0, SEG)
        values = tl.load(
            base + rows[:, None] * ROW_STRIDE + tl.arange(0, 256)[None, :],
            first + rows[:, None] < length,
            0,
        )
        tile = insert_slice(tile, values, (start, 0), (SEG, 256), (1, 1))
    return tile


# ---- common ----


@libentry()
@triton.jit
def _copy_or_fill(
    source,
    destination,
    count,
    SHAPE: tl.constexpr,
    SOURCE_STRIDES: tl.constexpr,
    DESTINATION_STRIDES: tl.constexpr,
    FILL: tl.constexpr,
    VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Widen before multiplying by strides: paged tensors may span above 4 GiB.
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    remaining = index
    source_offset = tl.full((BLOCK,), 0, tl.int64)
    destination_offset = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = remaining % SHAPE[axis]
        remaining = remaining // SHAPE[axis]
        destination_offset += coordinate * DESTINATION_STRIDES[axis]
        if not FILL:
            source_offset += coordinate * SOURCE_STRIDES[axis]
    if FILL:
        value = tl.full((BLOCK,), VALUE, tl.float32)
    else:
        value = tl.load(source + source_offset, index < count, other=0)
    tl.store(destination + destination_offset, value, index < count)


# ---- common ----


def _launch(source, destination, *, fill=False, value=0.0):
    count = destination.numel()
    if count == 0:
        return destination
    with torch_device_fn.device(destination.device):
        # Memory-only layout helper; no matrix multiplication or staging search.
        _copy_or_fill[(triton.cdiv(count, 1024),)](
            source,
            destination,
            count,
            tuple(destination.shape),
            tuple(source.stride()),
            tuple(destination.stride()),
            fill,
            value,
            1024,
            num_warps=4,
            num_stages=1,
        )
    return destination


# ---- common ----


def copy_tensor(source, destination):
    assert source.shape == destination.shape
    assert source.dtype == destination.dtype and source.device == destination.device
    return _launch(source, destination)


# ---- common ----


def fill_tensor(destination, value):
    return _launch(destination, destination, fill=True, value=value)


# ---- common ----


def _contiguous_copy(source):
    destination = torch.empty(source.shape, dtype=source.dtype, device=source.device)
    return copy_tensor(source, destination)


# ---- common ----


def ensure_last_dim_contiguous(source):
    if source.stride(-1) == 1:
        return source
    return _contiguous_copy(source)


# ---- common ----


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * max(shape[axis + 1], 1)
    return tuple(strides)


# ---- common ----


def _view_strides(source, shape):
    """Partition contiguous stride chunks using metadata only."""
    assert all(size >= 0 for size in shape)
    assert math.prod(shape) == source.numel()
    sizes, strides = tuple(source.shape), source.stride()
    if source.numel() == 0:
        return strides if sizes == shape else _contiguous_strides(shape)
    if not sizes:
        return (1,) * len(shape)
    result = [0] * len(shape)
    view_axis = len(shape) - 1
    chunk_stride = strides[-1]
    chunk_elements = view_elements = 1
    for axis in range(len(sizes) - 1, -1, -1):
        chunk_elements *= sizes[axis]
        if axis == 0 or (
            sizes[axis - 1] != 1 and strides[axis - 1] != chunk_elements * chunk_stride
        ):
            while view_axis >= 0 and (
                view_elements < chunk_elements or shape[view_axis] == 1
            ):
                result[view_axis] = view_elements * chunk_stride
                view_elements *= shape[view_axis]
                view_axis -= 1
            if view_elements != chunk_elements:
                return None
            if axis > 0:
                chunk_stride = strides[axis - 1]
                chunk_elements = view_elements = 1
    return tuple(result) if view_axis == -1 else None


# ---- common ----


def view_tensor(source, shape):
    shape = tuple(shape)
    strides = _view_strides(source, shape)
    if strides is None:
        raise ValueError("output layout cannot be represented as a no-copy view")
    return source.as_strided(shape, strides)


# ---- common ----


def reshape_view_or_copy(source, shape):
    shape = tuple(shape)
    strides = _view_strides(source, shape)
    if strides is not None:
        return source.as_strided(shape, strides)
    # Decide whether a view is possible before issuing an explicit Triton copy.
    return _contiguous_copy(source).as_strided(shape, _contiguous_strides(shape))


# ---- scaled_k_pack ----


def use_packed_k_prescale(params):
    # Fixed D256 power-of-two scaling; no length values leave the device.
    return (
        params.q_ptr.dtype == torch.bfloat16
        and params.k_ptr.dtype == torch.bfloat16
        and params.is_cu_seqlens_q
        and params.is_paged
        and params.is_causal
        and params.d == 256
        and params.block_size in (16, 32)
        and (
            (
                params.b == 1
                and params.total_q == params.seqlen_q == params.seqlen_k
                and (512 <= params.seqlen_q <= 1152 or 2304 < params.seqlen_q <= 4608)
                and params.h // params.hk in (4, 8)
            )
            or (
                params.b > 1
                and params.h // params.hk in (4, 8)
                and 8192 <= params.total_q
                and 128 * params.b < params.total_q <= 4608 * params.b
                and params.total_q < params.b * params.seqlen_q
                and 2304 < params.seqlen_q <= 4608
                and 2304 < params.seqlen_k <= 4608
            )
            or (
                params.h // params.hk in (4, 8)
                and params.b > 1
                and 128 * params.b < params.total_q
                and params.total_q < params.b * params.seqlen_q
                and 512 <= params.seqlen_q <= 1152
                and 512 <= params.seqlen_k <= 1536
            )
            or (
                params.h // params.hk == 8
                and 1 < params.b < 32
                and params.total_q >= 8192
                and 128 * params.b < params.total_q
                and params.total_q < params.b * params.seqlen_q
                and 8192 < params.seqlen_q <= 32768
                and 32768 <= params.seqlen_k <= 73728
            )
            or (
                params.h // params.hk in (4, 8)
                and params.b >= 32
                and params.seqlen_q + params.b - 1 < params.total_q
                and params.total_q <= 128 * params.b
                and params.total_q < params.b * params.seqlen_q
                and 512 <= params.seqlen_q <= 1152
                and 512
                <= params.seqlen_k
                <= (2304 if params.h // params.hk == 8 else 1536)
            )
        )
        and params.scale_softmax == 0.0625
    )


# ---- scaled_k_pack ----


@triton.jit
def pack_kv_prescaled(
    K,
    V,
    TABLE,
    USED,
    KO,
    VO,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    TS: tl.constexpr,
    LENGTH: tl.constexpr,
    PAGE: tl.constexpr,
    HK: tl.constexpr,
    PROGRAMS: tl.constexpr,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
    PACK_Q: tl.constexpr = False,
    Q=None,
    CUQ=None,
    QP=None,
    H: tl.constexpr = 0,
    MAX_TASKS: tl.constexpr = 0,
    QR: tl.constexpr = 0,
    QH: tl.constexpr = 0,
    FUSE_MIXED_LISTS: tl.constexpr = False,
    DECODE_IDS=None,
    OTHER_IDS=None,
    COUNTS=None,
    B: tl.constexpr = 0,
    PAGES_PER_TILE: tl.constexpr = 1,
):
    if FUSE_MIXED_LISTS:
        if (tl.program_id(0) == 0) & (tl.program_id(1) == 0) & (tl.program_id(2) == 0):
            classify_mixed_requests(
                CUQ, DECODE_IDS, OTHER_IDS, COUNTS, B, triton.next_power_of_2(B)
            )
    count = B if FUSE_MIXED_LISTS else tl.num_programs(1)
    if REQUEST_LIST:
        count = tl.load(REQUEST_COUNT)
    first = tl.program_id(1)
    stride = tl.num_programs(1)
    if FUSE_MIXED_LISTS:
        first = tl.where(first == 0, B, first - 1)
        stride -= 1
    for request in range(first, count, stride):
        batch = request
        if REQUEST_LIST:
            batch = tl.load(REQUEST_IDS + request)
        keep_request = True
        if FUSE_MIXED_LISTS:
            keep_request = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch) > 1
        if keep_request:
            if PACK_Q:
                # Retain original request-list Q packing; every slice has one owner.
                q_start = tl.load(CUQ + batch)
                q_len = tl.load(CUQ + batch + 1) - q_start
                for q_linear in range(
                    tl.program_id(0) * HK + tl.program_id(2), H * 4, PROGRAMS * HK
                ):
                    q_head_tile = q_linear // 4
                    q_group: tl.constexpr = H // HK
                    q_head_rows = tl.minimum(
                        q_group, tl.maximum(1, (128 // tl.maximum(q_len, 1)) // 2 * 2)
                    )
                    q_head_tiles = tl.cdiv(q_group, q_head_rows)
                    if (q_head_tile < q_head_tiles * HK) & (q_len % 128 != 0):
                        q_rows = (q_linear % 4) * 32 + tl.arange(0, 32)
                        q_tile = q_len // 128
                        q_dims = tl.arange(0, 256)
                        q_hs = (q_head_tile % q_head_tiles) * q_head_rows
                        q_kh = q_head_tile // q_head_tiles
                        q_qs = tl.maximum(1, tl.minimum(q_len, 128))
                        if q_head_rows == 1:
                            q_actual = q_tile * 128 + q_rows
                            q_affine_head = q_kh * q_group + q_hs
                            q_values = tl.load(
                                Q
                                + (q_start + q_actual).to(tl.int64)[:, None] * QR
                                + q_affine_head * QH
                                + q_dims[None, :],
                                q_actual[:, None] < q_len,
                                0,
                            )
                        else:
                            # Preserve the original pack_query mapping under dynamic CUQ.
                            q_actual = q_tile * 128 + q_rows % q_qs
                            q_local_head = q_hs + q_rows // q_qs
                            q_group_head = tl.minimum(
                                q_kh * q_group + q_local_head, H - 1
                            )
                            q_valid = (
                                (q_rows < q_qs * q_head_rows)
                                & (q_local_head < q_group)
                                & (q_actual < q_len)
                            )
                            q_values = tl.load(
                                Q
                                + (q_start + q_actual).to(tl.int64)[:, None] * QR
                                + q_group_head[:, None] * QH
                                + q_dims[None, :],
                                q_valid[:, None],
                                0,
                            )
                        q_task = q_tile * q_head_tiles * HK + q_head_tile
                        tl.store(
                            QP
                            + (
                                (batch.to(tl.int64) * MAX_TASKS + q_task) * 128
                                + q_rows[:, None]
                            )
                            * 256
                            + q_dims[None, :],
                            q_values,
                        )
            head = tl.program_id(2)
            length = tl.load(USED + batch)
            if PAGES_PER_TILE > 1:
                tile_rows: tl.constexpr = PAGE * PAGES_PER_TILE
                tiles = tl.cdiv(length, 512) * (512 // tile_rows)
                steps = tl.cdiv(tiles, PROGRAMS)
                for step in range(steps):
                    tile = tl.program_id(0) + step * PROGRAMS
                    if tile < tiles:
                        logical_rows = tile * tile_rows + tl.arange(0, tile_rows)
                        logical_pages = logical_rows // PAGE
                        safe_pages = tl.minimum(
                            logical_pages, tl.maximum(tl.cdiv(length, PAGE) - 1, 0)
                        )
                        physical = tl.load(
                            TABLE + batch.to(tl.int64) * TS + safe_pages
                        ).to(tl.int64)
                        dims = tl.arange(0, 256)
                        valid = logical_rows[:, None] < length
                        local_rows = logical_rows % PAGE
                        k = tl.load(
                            K
                            + physical[:, None] * KS0
                            + local_rows[:, None] * KS1
                            + head * KS2
                            + dims[None, :],
                            valid,
                            0,
                        )
                        v = tl.load(
                            V
                            + physical[:, None] * VS0
                            + local_rows[:, None] * VS1
                            + head * VS2
                            + dims[None, :],
                            valid,
                            0,
                        )
                        dst = (
                            (batch.to(tl.int64) * HK + head) * LENGTH
                            + logical_rows[:, None]
                        ) * 256 + dims[None, :]
                        tl.store(
                            KO + dst,
                            (k.to(tl.float32) * 0.0625).to(KO.dtype.element_ty),
                        )
                        tl.store(VO + dst, v)
            else:
                pages = tl.cdiv(length, 512) * (512 // PAGE)
                if PROGRAMS >= LENGTH // PAGE:
                    steps = 1
                else:
                    steps = tl.cdiv(pages, PROGRAMS)
                for step in range(steps):
                    page = tl.program_id(0) + step * PROGRAMS
                    if page < pages:
                        safe_page = tl.minimum(
                            page, tl.maximum(tl.cdiv(length, PAGE) - 1, 0)
                        )
                        physical = tl.load(
                            TABLE + batch.to(tl.int64) * TS + safe_page
                        ).to(tl.int64)
                        rows = tl.arange(0, PAGE)[:, None]
                        dims = tl.arange(0, 256)[None, :]
                        valid = page * PAGE + rows < length
                        k = tl.load(
                            K + physical * KS0 + rows * KS1 + head * KS2 + dims,
                            valid,
                            0,
                        )
                        v = tl.load(
                            V + physical * VS0 + rows * VS1 + head * VS2 + dims,
                            valid,
                            0,
                        )
                        dst = (
                            (batch.to(tl.int64) * HK + head) * LENGTH
                            + page * PAGE
                            + rows
                        ) * 256 + dims
                        tl.store(
                            KO + dst,
                            (k.to(tl.float32) * 0.0625).to(KO.dtype.element_ty),
                        )
                        tl.store(VO + dst, v)


# ---- repack ----


@triton.jit
def repack_head_major(
    K,
    V,
    TABLE,
    USED,
    KO,
    VO,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    TS: tl.constexpr,
    LENGTH: tl.constexpr,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    batch = tl.program_id(1)
    head = tl.program_id(2)
    length = tl.load(USED + batch)
    if PROGRAMS >= LENGTH // PAGE:
        steps = 1
    else:
        steps = tl.cdiv(tl.cdiv(length, PAGE), PROGRAMS)
    for step in range(steps):
        page = tl.program_id(0) + step * PROGRAMS
        if page < tl.cdiv(length, PAGE):
            physical = tl.load(TABLE + batch.to(tl.int64) * TS + page).to(tl.int64)
            rows = tl.arange(0, PAGE)[:, None]
            dims = tl.arange(0, 256)[None, :]
            valid = page * PAGE + rows < length
            k = tl.load(K + physical * KS0 + rows * KS1 + head * KS2 + dims, valid, 0)
            v = tl.load(V + physical * VS0 + rows * VS1 + head * VS2 + dims, valid, 0)
            dst = (
                (batch.to(tl.int64) * HEADS + head) * LENGTH + page * PAGE + rows
            ) * 256 + dims
            tl.store(KO + dst, k, valid)
            tl.store(VO + dst, v, valid)


# ---- repack ----


@triton.jit
def load_kv_tile(
    X,
    TABLE,
    batch,
    head,
    begin,
    length,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    BN: tl.constexpr,
    CONTIG_K: tl.constexpr,
    HK: tl.constexpr,
):
    if CONTIG_K:
        rows = begin + tl.arange(0, BN)
        dims = tl.arange(0, 256)
        offset = (
            (batch.to(tl.int64) * HK + head) * CONTIG_K + rows[:, None]
        ) * 256 + dims[None, :]
        values = tl.load(X + offset, rows[:, None] < length, 0)
    else:
        values = load_paged_tile(
            X,
            TABLE,
            batch,
            head,
            begin,
            length,
            PAGE_STRIDE,
            ROW_STRIDE,
            HEAD_STRIDE,
            TABLE_STRIDE,
            PAGE,
            BN,
        )
    return values


# ---- repack ----


@triton.jit
def pack_query(
    Q,
    CUQ,
    QP,
    H: tl.constexpr,
    HK: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    PROGRAMS: tl.constexpr,
    Q_DIRECT: tl.constexpr = False,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
):
    count = tl.num_programs(1)
    if REQUEST_LIST:
        count = tl.load(REQUEST_COUNT)
    for request in range(tl.program_id(1), count, tl.num_programs(1)):
        batch = request
        if REQUEST_LIST:
            batch = tl.load(REQUEST_IDS + request)
        qstart = tl.load(CUQ + batch)
        qlen = tl.load(CUQ + batch + 1) - qstart
        group: tl.constexpr = H // HK
        head_rows = tl.minimum(
            group, tl.maximum(1, (128 // tl.maximum(qlen, 1)) // 2 * 2)
        )
        head_tiles = tl.cdiv(group, head_rows)
        ntiles = tl.cdiv(qlen, 128) * head_tiles * HK
        if PROGRAMS >= MAX_TASKS:
            steps = 1
        else:
            steps = tl.cdiv(ntiles, PROGRAMS)
        for step in range(steps):
            tile = tl.program_id(0) + step * PROGRAMS
            if tile < ntiles:
                qt = tile // (head_tiles * HK)
                ht = tile % (head_tiles * HK)
                kh = ht // head_tiles
                hs = (ht % head_tiles) * head_rows
                qs = tl.maximum(1, tl.minimum(qlen, 128))
                direct = (head_rows == 1) & (qt * 128 + 128 <= qlen)
                if not Q_DIRECT or not direct:
                    rr = tl.program_id(2) * 32 + tl.arange(0, 32)
                    row = qt * 128 + rr % qs
                    hl = hs + rr // qs
                    head = kh * group + hl
                    valid = (rr < qs * head_rows) & (hl < group) & (row < qlen)
                    dims = tl.arange(0, 256)
                    if head_rows == 1:
                        affine_row = qt * 128 + rr
                        affine_head = kh * group + hs
                        q = tl.load(
                            Q
                            + (qstart + affine_row).to(tl.int64)[:, None] * QR
                            + affine_head * QH
                            + dims[None, :],
                            affine_row[:, None] < qlen,
                            0,
                        )
                    else:
                        # Keep masked heads inside the Q allocation too.
                        safe_head = tl.minimum(head, H - 1)
                        q = tl.load(
                            Q
                            + (qstart + row).to(tl.int64)[:, None] * QR
                            + safe_head[:, None] * QH
                            + dims[None, :],
                            valid[:, None],
                            0,
                        )
                    dst = (
                        (batch.to(tl.int64) * MAX_TASKS + tile) * 128 + rr[:, None]
                    ) * 256 + dims[None, :]
                    tl.store(QP + dst, q)


# ---- repack ----


@triton.jit
def pack_kv(
    K,
    V,
    TABLE,
    USED,
    KO,
    VO,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    TS: tl.constexpr,
    LENGTH: tl.constexpr,
    PAGE: tl.constexpr,
    HK: tl.constexpr,
    PROGRAMS: tl.constexpr,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
    PACK_Q: tl.constexpr = False,
    Q=None,
    CUQ=None,
    QP=None,
    H: tl.constexpr = 0,
    MAX_TASKS: tl.constexpr = 0,
    QR: tl.constexpr = 0,
    QH: tl.constexpr = 0,
    FUSE_MIXED_LISTS: tl.constexpr = False,
    DECODE_IDS=None,
    OTHER_IDS=None,
    COUNTS=None,
    B: tl.constexpr = 0,
):
    if FUSE_MIXED_LISTS:
        if (tl.program_id(0) == 0) & (tl.program_id(1) == 0) & (tl.program_id(2) == 0):
            classify_mixed_requests(
                CUQ, DECODE_IDS, OTHER_IDS, COUNTS, B, triton.next_power_of_2(B)
            )
    count = B if FUSE_MIXED_LISTS else tl.num_programs(1)
    if REQUEST_LIST:
        count = tl.load(REQUEST_COUNT)
    first = tl.program_id(1)
    stride = tl.num_programs(1)
    if FUSE_MIXED_LISTS:
        first = tl.where(first == 0, B, first - 1)
        stride -= 1
    for request in range(first, count, stride):
        batch = request
        if REQUEST_LIST:
            batch = tl.load(REQUEST_IDS + request)
        keep_request = True
        if FUSE_MIXED_LISTS:
            keep_request = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch) > 1
        if keep_request:
            if PACK_Q:
                # The host gate guarantees B1/direct-Q. One 32-row slice has one owner.
                q_start = tl.load(CUQ + batch)
                q_len = tl.load(CUQ + batch + 1) - q_start
                for q_linear in range(
                    tl.program_id(0) * HK + tl.program_id(2), H * 4, PROGRAMS * HK
                ):
                    q_head_tile = q_linear // 4
                    q_group: tl.constexpr = H // HK
                    q_head_rows = tl.minimum(
                        q_group, tl.maximum(1, (128 // tl.maximum(q_len, 1)) // 2 * 2)
                    )
                    q_head_tiles = tl.cdiv(q_group, q_head_rows)
                    if (q_head_tile < q_head_tiles * HK) & (q_len % 128 != 0):
                        q_rows = (q_linear % 4) * 32 + tl.arange(0, 32)
                        q_tile = q_len // 128
                        q_dims = tl.arange(0, 256)
                        q_hs = (q_head_tile % q_head_tiles) * q_head_rows
                        q_kh = q_head_tile // q_head_tiles
                        q_qs = tl.maximum(1, tl.minimum(q_len, 128))
                        if q_head_rows == 1:
                            q_actual = q_tile * 128 + q_rows
                            q_affine_head = q_kh * q_group + q_hs
                            q_values = tl.load(
                                Q
                                + (q_start + q_actual).to(tl.int64)[:, None] * QR
                                + q_affine_head * QH
                                + q_dims[None, :],
                                q_actual[:, None] < q_len,
                                0,
                            )
                        else:
                            # Preserve the original pack_query mapping under dynamic CUQ.
                            q_actual = q_tile * 128 + q_rows % q_qs
                            q_local_head = q_hs + q_rows // q_qs
                            q_group_head = tl.minimum(
                                q_kh * q_group + q_local_head, H - 1
                            )
                            q_valid = (
                                (q_rows < q_qs * q_head_rows)
                                & (q_local_head < q_group)
                                & (q_actual < q_len)
                            )
                            q_values = tl.load(
                                Q
                                + (q_start + q_actual).to(tl.int64)[:, None] * QR
                                + q_group_head[:, None] * QH
                                + q_dims[None, :],
                                q_valid[:, None],
                                0,
                            )
                        q_task = q_tile * q_head_tiles * HK + q_head_tile
                        tl.store(
                            QP
                            + (
                                (batch.to(tl.int64) * MAX_TASKS + q_task) * 128
                                + q_rows[:, None]
                            )
                            * 256
                            + q_dims[None, :],
                            q_values,
                        )
            head = tl.program_id(2)
            length = tl.load(USED + batch)
            pages = tl.cdiv(length, 512) * (512 // PAGE)
            if PROGRAMS >= LENGTH // PAGE:
                steps = 1
            else:
                steps = tl.cdiv(pages, PROGRAMS)
            for step in range(steps):
                page = tl.program_id(0) + step * PROGRAMS
                if page < pages:
                    safe_page = tl.minimum(
                        page, tl.maximum(tl.cdiv(length, PAGE) - 1, 0)
                    )
                    physical = tl.load(TABLE + batch.to(tl.int64) * TS + safe_page).to(
                        tl.int64
                    )
                    rows = tl.arange(0, PAGE)[:, None]
                    dims = tl.arange(0, 256)[None, :]
                    valid = page * PAGE + rows < length
                    k = tl.load(
                        K + physical * KS0 + rows * KS1 + head * KS2 + dims, valid, 0
                    )
                    v = tl.load(
                        V + physical * VS0 + rows * VS1 + head * VS2 + dims, valid, 0
                    )
                    dst = (
                        (batch.to(tl.int64) * HK + head) * LENGTH + page * PAGE + rows
                    ) * 256 + dims
                    tl.store(KO + dst, k)
                    tl.store(VO + dst, v)


# ---- repack ----


@triton.jit
def ascend_flash_attention_decode_pack(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    USED,
    TABLE,
    QP,
    KO,
    VO,
    scale,
    QR: tl.constexpr,
    QH: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TS: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    LENGTH: tl.constexpr,
    PAGE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    REQUEST_IDS=None,
    REQUEST_COUNT=None,
    REQUEST_LIST: tl.constexpr = False,
):
    lane = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    qstart = tl.load(CUQ + batch)
    qlen = tl.load(CUQ + batch + 1) - qstart
    length = tl.load(USED + batch)
    group: tl.constexpr = H // HK
    if qlen == 1:
        if length <= 32:
            if lane == 0:
                _decode_actual_task(
                    Q,
                    K,
                    V,
                    O,
                    LSE,
                    CUQ,
                    None,
                    USED,
                    TABLE,
                    scale,
                    QR,
                    QH,
                    KS0,
                    KS1,
                    KS2,
                    VS0,
                    VS1,
                    VS2,
                    OR,
                    OH,
                    TS,
                    TOTAL_Q,
                    group,
                    256,
                    PAGE,
                    True,
                    32,
                    256,
                    group,
                    batch,
                    head,
                    0,
                    1,
                )
        else:
            if lane == 0:
                # Q1 has one tile per KV head; keep all 128 packed rows written.
                for part in range(4):
                    rows = part * 32 + tl.arange(0, 32)
                    dims = tl.arange(0, 256)
                    safe_head = tl.minimum(head * group + rows, H - 1)
                    q = tl.load(
                        Q
                        + qstart.to(tl.int64) * QR
                        + safe_head[:, None] * QH
                        + dims[None, :],
                        rows[:, None] < group,
                        0,
                    )
                    dst = (
                        (batch.to(tl.int64) * MAX_TASKS + head) * 128 + rows[:, None]
                    ) * 256 + dims[None, :]
                    tl.store(QP + dst, q)
            pages = tl.cdiv(length, 512) * (512 // PAGE)
            steps = tl.cdiv(pages, PROGRAMS)
            for step in range(steps):
                page = lane + step * PROGRAMS
                if page < pages:
                    safe_page = tl.minimum(
                        page, tl.maximum(tl.cdiv(length, PAGE) - 1, 0)
                    )
                    physical = tl.load(TABLE + batch.to(tl.int64) * TS + safe_page).to(
                        tl.int64
                    )
                    rows = tl.arange(0, PAGE)[:, None]
                    dims = tl.arange(0, 256)[None, :]
                    valid = page * PAGE + rows < length
                    k = tl.load(
                        K + physical * KS0 + rows * KS1 + head * KS2 + dims, valid, 0
                    )
                    v = tl.load(
                        V + physical * VS0 + rows * VS1 + head * VS2 + dims, valid, 0
                    )
                    dst = (
                        (batch.to(tl.int64) * HK + head) * LENGTH + page * PAGE + rows
                    ) * 256 + dims
                    tl.store(KO + dst, k)
                    tl.store(VO + dst, v)
    # Sibling programs never read this list: only the next kernel consumes it.
    if REQUEST_LIST:
        if (lane == 0) & (batch == 0) & (head == 0):
            _classify_long_requests(
                CUQ, USED, REQUEST_IDS, REQUEST_COUNT, B, BLOCK_BATCH
            )


# ---- repack ----


def prepare_packed_inputs(
    params,
    workers,
    request_ids=None,
    request_count=None,
    *,
    fuse_short_decode=False,
    stage_workers=0,
    fused_mixed_lists=None,
):
    """Pack Q/K/V once so the pipelined kernel can use affine L1 loads."""
    capacity = triton.cdiv(params.seqlen_k, 512) * 512
    tasks = triton.cdiv(params.seqlen_q, 128) * params.h
    elements = params.b * params.hk * capacity * params.d
    bytes_needed = elements * params.k_ptr.element_size() * 2
    if (
        not params.is_cu_seqlens_q
        or not params.is_paged
        or not capacity
        or bytes_needed > (1 << 30)
    ):
        return None
    q_direct = (
        params.q_ptr.stride(-1) == 1
        and params.q_row_stride >= 256
        and params.q_head_stride >= 256
        and params.q_row_stride % 16 == 0
        and params.q_head_stride % 16 == 0
        and params.q_ptr.data_ptr() % 32 == 0
    )
    qp = torch.empty(
        (
            (params.b * tasks + stage_workers, 128, 256)
            if stage_workers
            else (params.b, tasks, 128, 256)
        ),
        dtype=params.q_ptr.dtype,
        device=params.q_ptr.device,
    )
    kp = torch.empty(
        (
            (params.b * params.hk + stage_workers, capacity, 256)
            if stage_workers
            else (params.b, params.hk, capacity, 256)
        ),
        dtype=params.k_ptr.dtype,
        device=params.k_ptr.device,
    )
    vp = torch.empty_like(kp)
    if fuse_short_decode:
        programs = min(
            capacity // params.block_size,
            max(1, workers * 2 // (params.b * params.hk)),
        )
        ascend_flash_attention_decode_pack[(programs, params.b, params.hk)](
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            params.o_ptr,
            params.softmax_lse_ptr,
            params.cu_seqlens_q_ptr,
            params.seqused_k_ptr,
            params.page_table_ptr,
            qp,
            kp,
            vp,
            params.scale_softmax,
            params.q_row_stride,
            params.q_head_stride,
            *params.k_ptr.stride()[:3],
            *params.v_ptr.stride()[:3],
            params.o_row_stride,
            params.o_head_stride,
            params.page_table_batch_stride,
            params.total_q,
            params.b,
            params.h,
            params.hk,
            tasks,
            capacity,
            params.block_size,
            programs,
            triton.next_power_of_2(params.b),
            REQUEST_IDS=request_ids,
            REQUEST_COUNT=request_count,
            REQUEST_LIST=request_ids is not None,
            enable_fp_fusion=False,
        )
        return qp, kp, vp, tasks, capacity, q_direct
    # A bounded wave visits only actual tiles/pages, irrespective of capacity.
    query_programs = min(tasks, max(1, workers * 8 // (params.b * 4)))
    kv_programs = min(
        capacity // params.block_size, max(1, workers * 8 // (params.b * params.hk))
    )
    batches = min(params.b, workers) if request_ids is not None else params.b
    fuse_prefill_q = (
        params.q_ptr.dtype == torch.bfloat16
        and params.is_causal
        and params.b == 1
        and params.total_q == params.seqlen_q == params.seqlen_k
        and 512 <= params.seqlen_q <= 16384
        and params.h // params.hk in (4, 8)
        and q_direct
        and request_ids is None
        and stage_workers == 0
        and kv_programs * params.hk >= 4 * params.h
    )
    fuse_mixed_q = (
        params.q_ptr.dtype == torch.bfloat16
        and params.is_causal
        and params.b > 1
        and params.b < params.total_q < params.b * params.seqlen_q
        and 512 <= params.seqlen_k <= 4608
        and params.h // params.hk in (4, 8)
        and q_direct
    )
    fuse_query = fuse_prefill_q or fuse_mixed_q
    if fuse_mixed_q:
        kv_programs = max(kv_programs, 4)
    if not fuse_query:
        pack_query[(query_programs, batches, 4)](
            params.q_ptr,
            params.cu_seqlens_q_ptr,
            qp,
            params.h,
            params.hk,
            tasks,
            params.q_row_stride,
            params.q_head_stride,
            query_programs,
            Q_DIRECT=q_direct,
            REQUEST_IDS=request_ids,
            REQUEST_COUNT=request_count,
            REQUEST_LIST=request_ids is not None,
        )
    from .common import pack_kv_prescaled, use_packed_k_prescale

    pack_kernel = pack_kv_prescaled if use_packed_k_prescale(params) else pack_kv
    multipage_prefill_pack = (
        pack_kernel is pack_kv_prescaled
        and params.b == 1
        and params.h == 4
        and params.hk == 1
        and params.total_q == params.seqlen_q == params.seqlen_k
        and 2304 < params.seqlen_q <= 4608
        and q_direct
        and fuse_prefill_q
        and request_ids is None
        and stage_workers == 0
    )
    if multipage_prefill_pack:
        kv_programs = min(capacity // (4 * params.block_size), workers * 2)

    fused_list_options = {}
    if fused_mixed_lists is not None:
        assert fuse_query and request_ids is not None
        fused_list_options = dict(
            FUSE_MIXED_LISTS=True,
            DECODE_IDS=fused_mixed_lists[0],
            OTHER_IDS=fused_mixed_lists[1],
            COUNTS=fused_mixed_lists[2],
            B=params.b,
        )
    pack_kernel[(kv_programs, batches + (fused_mixed_lists is not None), params.hk)](
        params.k_ptr,
        params.v_ptr,
        params.page_table_ptr,
        params.seqused_k_ptr,
        kp,
        vp,
        *params.k_ptr.stride()[:3],
        *params.v_ptr.stride()[:3],
        params.page_table_batch_stride,
        capacity,
        params.block_size,
        params.hk,
        kv_programs,
        **(dict(PAGES_PER_TILE=4, multibuffer=False) if multipage_prefill_pack else {}),
        REQUEST_IDS=request_ids,
        REQUEST_COUNT=request_count,
        REQUEST_LIST=request_ids is not None and fused_mixed_lists is None,
        **fused_list_options,
        **(
            dict(
                PACK_Q=True,
                Q=params.q_ptr,
                CUQ=params.cu_seqlens_q_ptr,
                QP=qp,
                H=params.h,
                MAX_TASKS=tasks,
                QR=params.q_row_stride,
                QH=params.q_head_stride,
            )
            if fuse_query
            else {}
        ),
    )
    return qp, kp, vp, tasks, capacity, q_direct


# ---- repack ----


def prepare_contiguous_kv(params, workers):
    length = triton.cdiv(params.seqlen_k, params.block_size) * params.block_size
    elements = params.b * params.hk * length * params.d
    if (
        not params.is_cu_seqlens_q
        or params.seqlen_q < 128
        or params.total_q < 1024
        or not length
        or elements * params.k_ptr.element_size() * 2 > (1 << 30)
    ):
        return params.k_ptr, params.v_ptr, 0
    shape = (params.b, params.hk, length, params.d)
    k = torch.empty(shape, dtype=params.k_ptr.dtype, device=params.k_ptr.device)
    v = torch.empty(shape, dtype=params.v_ptr.dtype, device=params.v_ptr.device)
    programs = min(
        length // params.block_size, max(1, workers * 8 // (params.b * params.hk))
    )
    repack_head_major[(programs, params.b, params.hk)](
        params.k_ptr,
        params.v_ptr,
        params.page_table_ptr,
        params.seqused_k_ptr,
        k,
        v,
        *params.k_ptr.stride()[:3],
        *params.v_ptr.stride()[:3],
        params.page_table_batch_stride,
        length,
        params.block_size,
        params.hk,
        programs,
    )
    return k, v, length


# ---- decode ----


@triton.jit
def _decode_actual_task(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USED,
    TABLE,
    scale,
    sq0: tl.constexpr,
    sq1: tl.constexpr,
    sk0: tl.constexpr,
    sk1: tl.constexpr,
    sk2: tl.constexpr,
    sv0: tl.constexpr,
    sv1: tl.constexpr,
    sv2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    st0: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    BG: tl.constexpr,
    batch,
    kh,
    head_part,
    HEAD_PARTS: tl.constexpr = 1,
):
    row = tl.load(CUQ + batch)
    end = tl.load(CUQ + batch + 1)
    if end - row == 1:
        if PAGED:
            length = tl.load(USED + batch)
            first = 0
        else:
            first = tl.load(CUK + batch)
            length = tl.load(CUK + batch + 1) - first
        cols, dims = tl.arange(0, BN), tl.arange(0, BD)
        if length == 1:
            physical = tl.load(TABLE + batch.to(tl.int64) * st0).to(tl.int64)
            g = head_part * (GROUP // HEAD_PARTS) + tl.arange(0, BG)
            dims = tl.arange(0, BD)
            heads = kh * GROUP + g
            valid = (g[:, None] < GROUP) & (dims[None, :] < D)
            q = tl.load(
                Q + row.to(tl.int64) * sq0 + heads[:, None] * sq1 + dims[None, :],
                valid,
                0,
            ).to(tl.float32)
            k = tl.load(K + physical * sk0 + kh * sk2 + dims, dims < D, 0).to(
                tl.float32
            )
            v = tl.load(V + physical * sv0 + kh * sv2 + dims, dims < D, 0)
            scores = tl.sum(q * k[None, :], 1) * scale
            tl.store(
                O + row.to(tl.int64) * so0 + heads[:, None] * so1 + dims[None, :],
                v[None, :],
                valid,
            )
            tl.store(LSE + heads * TOTAL_Q + row, scores, g < GROUP)
        elif (length > 1) & (length <= BN):
            if PAGED:
                k = load_paged_tile(
                    K, TABLE, batch, kh, 0, length, sk0, sk1, sk2, st0, PAGE, BN
                ).to(tl.float32)
            else:
                ko = (first + cols).to(tl.int64) * sk0 + kh * sk1
                k = tl.load(
                    K + ko[:, None] + dims[None, :],
                    (cols[:, None] < length) & (dims[None, :] < D),
                    0,
                ).to(tl.float32)
            if PAGED:
                v = load_paged_tile(
                    V, TABLE, batch, kh, 0, length, sv0, sv1, sv2, st0, PAGE, BN
                ).to(tl.float32)
            else:
                vo = (first + cols).to(tl.int64) * sv0 + kh * sv1
                v = tl.load(
                    V + vo[:, None] + dims[None, :],
                    (cols[:, None] < length) & (dims[None, :] < D),
                    0,
                ).to(tl.float32)
            for gh in range(GROUP // HEAD_PARTS):
                head = kh * GROUP + head_part * (GROUP // HEAD_PARTS) + gh
                q = tl.load(
                    Q + row.to(tl.int64) * sq0 + head * sq1 + dims, dims < D, 0
                ).to(tl.float32)
                scores = tl.sum(k * q[None, :], 1) * scale
                scores = tl.where(cols < length, scores, -float("inf"))
                maximum = tl.max(scores, 0)
                safe_max = tl.where(length > 0, maximum, 0.0)
                weights = tl.exp(scores - safe_max)
                normalizer = tl.sum(weights, 0)
                denom = tl.where(length > 0, normalizer, 1.0)
                p = weights.to(Q.dtype.element_ty).to(tl.float32)
                result = tl.sum(p[:, None] * v, 0) / denom
                tl.store(
                    O + row.to(tl.int64) * so0 + head * so1 + dims, result, dims < D
                )
                lse = tl.where(length > 0, maximum + tl.log(denom), -float("inf"))
                tl.store(LSE + head * TOTAL_Q + row, lse)
        elif length == 0:
            for empty_gh in range(GROUP // HEAD_PARTS):
                empty_head = kh * GROUP + head_part * (GROUP // HEAD_PARTS) + empty_gh
                tl.store(
                    O + row.to(tl.int64) * so0 + empty_head * so1 + dims, 0.0, dims < D
                )
                tl.store(LSE + empty_head * TOTAL_Q + row, -float("inf"))

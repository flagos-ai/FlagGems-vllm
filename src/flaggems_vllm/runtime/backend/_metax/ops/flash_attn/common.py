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

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, tl_extra_shim


@triton.jit
def compact_ragged_tile_coords(
    tile_idx,
    cu_seqlens_q_ptr,
    batch_size,
    BLOCK_M: tl.constexpr,
):
    """Map one compact Q-tile id to request-local ``(m_block, bid)``."""

    # tl.cumsum benefits from a power-of-two vector.  Use 31 requests per
    # group so bids + 1 remains a naturally masked cu_seqlens_q load.
    lane = tl.arange(0, 32)
    group_bid = 0
    group_start = 0
    found = False
    selected_bid = 0
    selected_m_block = 0

    while (group_bid < batch_size) & (~found):
        bids = group_bid + lane
        valid = (lane < 31) & (bids < batch_size)
        q_bos = tl.load(cu_seqlens_q_ptr + bids, mask=valid, other=0).to(tl.int32)
        q_eos = tl.load(cu_seqlens_q_ptr + bids + 1, mask=valid, other=0).to(tl.int32)
        m_blocks = tl.cdiv(q_eos - q_bos, BLOCK_M)
        group_prefix = tl.cumsum(tl.where(valid, m_blocks, 0), axis=0)
        work_before = group_prefix - m_blocks
        group_work = tl.sum(tl.where(valid, m_blocks, 0), axis=0)
        local_idx = tile_idx - group_start
        in_group = (local_idx >= 0) & (local_idx < group_work)
        hit = valid & (local_idx >= work_before) & (local_idx < group_prefix)

        selected_bid = tl.where(
            in_group, tl.sum(tl.where(hit, bids, 0), axis=0), selected_bid
        )
        selected_m_block = tl.where(
            in_group,
            tl.sum(tl.where(hit, local_idx - work_before, 0), axis=0),
            selected_m_block,
        )
        found = found | in_group
        group_start += group_work
        group_bid += 31

    return selected_m_block, selected_bid, found


@triton.jit
def u64_to_lohi(x):
    return (x >> 32).to(tl.uint32), (x & 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def u64_from_lohi(lo, hi):
    # Pack the low and high 32-bit words into one 64-bit value.
    return (hi.to(tl.uint64) << 32) + lo.to(tl.uint64)


@triton.jit
def philox_(seed, subsequence, offset):
    kPhilox10A: tl.constexpr = 0x9E3779B9
    kPhilox10B: tl.constexpr = 0xBB67AE85
    k0, k1 = u64_to_lohi(seed.to(tl.uint64))
    c0, c1 = u64_to_lohi(offset.to(tl.uint64))
    c2, c3 = u64_to_lohi(subsequence.to(tl.uint64))

    # pragma unroll
    kPhiloxSA: tl.constexpr = 0xD2511F53
    kPhiloxSB: tl.constexpr = 0xCD9E8D57
    for _ in tl.static_range(6):
        res0 = kPhiloxSA * c0.to(tl.uint64)
        res1 = kPhiloxSB * c2.to(tl.uint64)
        res0_x, res0_y = u64_to_lohi(res0)
        res1_x, res1_y = u64_to_lohi(res1)
        c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x
        k0 += kPhilox10A
        k1 += kPhilox10B

    res0 = kPhiloxSA * c0.to(tl.uint64)
    res1 = kPhiloxSB * c2.to(tl.uint64)
    res0_x, res0_y = u64_to_lohi(res0)
    res1_x, res1_y = u64_to_lohi(res1)
    c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x

    return c0, c1, c2, c3


@triton.jit
def apply_dropout_mask(
    P,
    mask,
    encode_dropout_in_sign_bit: tl.constexpr,
):
    if encode_dropout_in_sign_bit:
        P = tl.where(mask, -P, P)
    else:
        P = tl.where(mask, (P * 0).to(P.dtype), P)
    return P


@triton.jit
def apply_dropout(
    P,
    row_start,
    col_start,
    n_cols,
    bid,
    hid,
    philox_seed,
    philox_offset,
    p_dropout_uint8: tl.constexpr,
    is_dropout: tl.constexpr,
    encode_dropout_in_sign_bit: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if is_dropout:
        row_start = tl.multiple_of(row_start, BLOCK_M)
        col_start = tl.multiple_of(col_start, BLOCK_N)
        row = row_start + tl.arange(0, BLOCK_M)[:, None]
        # Down scale col_idx by 4
        col = col_start // 4 + tl.arange(0, BLOCK_N // 4)[None, :]

        subsequence = row.to(tl.uint64) * n_cols + col.to(tl.uint64)

        offset = philox_offset + bid * NUM_HEADS + hid
        offset += subsequence * 0
        r0, r1, r2, r3 = philox_(philox_seed, subsequence, offset)

        r = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(BLOCK_M, BLOCK_N)

        mask = (r & 0xFF) >= p_dropout_uint8

        P = apply_dropout_mask(
            P, mask, encode_dropout_in_sign_bit=encode_dropout_in_sign_bit
        )
    return P


@triton.jit
def apply_alibi(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    is_causal: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slope: tl.constexpr = None,
):
    if is_alibi:
        if is_causal:
            # The row independent alibi bias renders the same attention output
            # as with the standard alibi because softmax is shift invariant, i.e.,
            # softmax(A + bias + const) = softamx(A + bias). The following two
            # biases are no different if causal is true.
            # bias_1 = [
            #   -4, -3, -2,  X, X,
            #   -4, -3, -2, -1, X,
            #   -4, -3, -2, -1, 0,
            # ]
            # bias_2 = [
            #   -2, -1, 0,  X,  X,
            #   -3, -2, -1, 0,  X,
            #   -4, -3, -2, -1, 0,
            # ]
            bias = alibi_slope * (-max_seqlen_k + 1 + col_idx[None, :]).to(tl.float32)
            S += bias
        else:
            bias = -alibi_slope * tl.abs(
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias

    return S


@triton.jit
def apply_mask(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    is_even_mn: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
):
    need_mask = is_causal | is_local | (not is_even_mn)
    # need_mask: tl.constexpr = is_causal | is_local
    if need_mask:
        # Extra care should be taken to void one-off errors: both col_lb and col_rb are inclusive!
        col_lb = max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left)
        col_rb = min(
            max_seqlen_k - 1, row_idx + max_seqlen_k - max_seqlen_q + window_size_right
        )

        if is_causal:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)

        if is_local:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )

        if (not is_local) & (not is_causal) & (not is_even_mn):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)

    return S


@triton.jit
def softmax_rescale(
    O_acc,
    S,
    row_max,
    row_sum,
    softmax_scale_log2e: tl.constexpr,
    is_border: tl.constexpr,
    # is_init: tl.constexpr
):
    prev_max = row_max
    row_max = tl.maximum(row_max, tl.max(S, 1))

    if is_border:
        cur_max = tl.where(row_max == float("-inf"), 0, row_max)
    else:
        cur_max = row_max

    p_scale = tl.math.exp2((prev_max - cur_max) * softmax_scale_log2e)
    row_sum *= p_scale
    O_acc *= p_scale[:, None]

    max_scaled = tl.where(row_max == float("-inf"), 0, row_max * softmax_scale_log2e)
    P = tl.math.exp2(S * softmax_scale_log2e - max_scaled[:, None])
    row_sum = row_sum + tl.sum(P, 1)
    return O_acc, P, row_max, row_sum


@triton.jit
def apply_softcap(S, softcap, is_softcap: tl.constexpr):
    if is_softcap:
        S = tl_extra_shim.tanh(S * softcap)

    return S


@triton.jit
def virtual_to_cache_offset(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    block_size,
    k_row_stride,
    k_page_stride,
    boundary_check: tl.constexpr = False,
):
    # virtual_index is the kv sequence index in the current batch element
    # page_table_ptr is already pointed at current batch element's block table entry
    # block_size is the size of each block in the page table
    virtual_page_index = virtual_index // block_size
    page_offset = virtual_index % block_size
    if boundary_check:
        page_block_index = tl.load(
            page_table_ptr + virtual_page_index,
            mask=virtual_index < max_virtual_index,
            other=0,
        ).to(tl.int64)
    else:
        page_block_index = tl.load(page_table_ptr + virtual_page_index).to(tl.int64)
    return page_block_index * k_page_stride + page_offset * k_row_stride


@triton.jit
def load_from_kvcache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    k_ptr_base,
    v_ptr_base,
    block_size,
    d: tl.constexpr,
    k_row_stride,
    BLOCK_K: tl.constexpr,
    k_page_stride=0,
    boundary_check: tl.constexpr = False,
):
    cache_offset = virtual_to_cache_offset(
        virtual_index,
        max_virtual_index,
        page_table_ptr,
        block_size,
        k_row_stride,
        k_page_stride,
        boundary_check,
    )
    k_offset = tl.arange(0, BLOCK_K)[:, None] + cache_offset[None, :]
    v_offset = tl.arange(0, BLOCK_K)[None, :] + cache_offset[:, None]
    if d == BLOCK_K:
        bK_mask = virtual_index[None, :] < max_virtual_index[None, :]
        bV_mask = virtual_index[:, None] < max_virtual_index[:, None]
        bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
        bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
    else:
        bK_mask = (tl.arange(0, BLOCK_K)[:, None] < d) & (
            virtual_index[None, :] < max_virtual_index[None, :]
        )
        bV_mask = (tl.arange(0, BLOCK_K)[None, :] < d) & (
            virtual_index[:, None] < max_virtual_index[:, None]
        )
        bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
        bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
    return bK, bV


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


def _launch(source, destination, *, fill=False, value=0.0):
    count = destination.numel()
    if count == 0:
        return destination
    with torch_device_fn.device(destination.device):
        # A memory-only helper: 256 C550 threads, no MMA/staging or tuning search.
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


def copy_tensor(source, destination):
    assert source.shape == destination.shape
    assert source.dtype == destination.dtype and source.device == destination.device
    return _launch(source, destination)


def fill_tensor(destination, value):
    return _launch(destination, destination, fill=True, value=value)


def _contiguous_copy(source):
    destination = torch.empty(source.shape, dtype=source.dtype, device=source.device)
    return copy_tensor(source, destination)


def ensure_last_dim_contiguous(source):
    if source.stride(-1) == 1:
        return source
    return _contiguous_copy(source)


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * max(shape[axis + 1], 1)
    return tuple(strides)


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


def view_tensor(source, shape):
    shape = tuple(shape)
    strides = _view_strides(source, shape)
    if strides is None:
        raise ValueError("output layout cannot be represented as a no-copy view")
    return source.as_strided(shape, strides)


def reshape_view_or_copy(source, shape):
    shape = tuple(shape)
    strides = _view_strides(source, shape)
    if strides is not None:
        return source.as_strided(shape, strides)
    # mcPyTorch may silently copy inside Tensor.view. Decide from strides first.
    return _contiguous_copy(source).as_strided(shape, _contiguous_strides(shape))


def tn_compile_scenario(*tensors):
    """Enable the 4 GiB address optimization only for bounded byte spans."""
    for tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        # numel() misses holes in noncontiguous caches. The origin is the
        # pointer passed to the kernel, so storage_offset is not added again.
        last_element = sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        if (last_element + 1) * tensor.element_size() > (1 << 32):
            # cpasync otherwise enables metaxgpu-aggressive-4g-addr-opt,
            # which truncates large K/V byte offsets even with int64 source.
            return "storeCoalesce;noaddropt"
    return "storeCoalesce"

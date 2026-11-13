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

"""Triton scatter for paged token caches.

Replaces the torch ``where/zeros_like/sum/div/remainder/stack/scatter_nd``
combos (plus the ACLGraph row-zero sentinel chains they need to stay
fixed-shape) with a single kernel::

    slot = slots[n]
    if 0 <= slot < num_blocks * block_size:
        cache[slot // bs, slot % bs, :] = values[n, :]

Invalid slots are skipped inside the kernel, so the sentinel bookkeeping
disappears and the kernel is trivially ACLGraph-capturable (fixed grid, no
host sync).  ``block_size=1`` covers plain row scatters.  The cache is
addressed through explicit block/token strides, so payload-prefix
``as_strided`` views of larger physical pages work as long as the inner
dims are contiguous; a separate specialization supports strided rows.

The original kernel divided an int64 slot by a runtime
``block_size`` (software-emulated int64 division on the vector cores) and
used a fixed BLOCK=2048 against inner sizes of 256/512 (75-87% dead lanes)
— together ~600us on prefill shapes.  Now ``block_size`` is a constexpr
(only 128/4/1 occur in practice; division lowers to a shift), the slot
math runs in int32 and ``BLOCK_INNER`` is next_pow2(inner) capped at 512.
A 2D [rows, inner] tile variant was measured and rejected: wide int64
address tiles collapse to ~50x slower on the vector cores.
"""

import logging
from dataclasses import dataclass
from math import prod

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "num_slots",
        "inner_numel",
        "stride_block",
        "stride_off",
        "value_stride",
    ]
)
def _paged_scatter_kernel(
    cache_ptr,
    slots_ptr,
    values_ptr,
    num_rows,
    num_slots,
    inner_numel,
    stride_block,
    stride_off,
    value_stride,
    slot_stride,
    cache_inner_stride,
    value_inner_stride,
    STRIDED: tl.constexpr,
    BLOCK_SZ: tl.constexpr,  # page size (128/4/1); constexpr -> // and % lower to shifts
    ROWS: tl.constexpr,  # rows scattered per program (amortises per-program overhead)
    BLOCK_INNER: tl.constexpr,  # next_pow2(inner_numel) capped at 512
):
    pid = tl.program_id(0)
    blk = tl.program_id(1)
    o = blk * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    m_col = o < inner_numel
    for i in tl.static_range(ROWS):
        n = pid * ROWS + i
        if n < num_rows:
            raw_slot = tl.load(slots_ptr + n.to(tl.int64) * slot_stride)
            if (raw_slot >= 0) & (raw_slot < num_slots):
                slot = raw_slot.to(tl.int32)
                row = slot // BLOCK_SZ
                off = slot - row * BLOCK_SZ
                if STRIDED:
                    source_col = o.to(tl.int64) * value_inner_stride
                    target_col = o.to(tl.int64) * cache_inner_stride
                else:
                    source_col = o
                    target_col = o
                v = tl.load(
                    values_ptr + n.to(tl.int64) * value_stride + source_col,
                    mask=m_col,
                    other=0,
                )
                tl.store(
                    cache_ptr
                    + row.to(tl.int64) * stride_block
                    + off.to(tl.int64) * stride_off
                    + target_col,
                    v,
                    mask=m_col,
                )


_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.int8)
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class _ScatterPlan:
    num_slots: int
    inner: int
    stride_block: int
    stride_off: int
    cache_inner: int
    value_inner: int
    strided: bool
    copy_values: bool = False


def _contiguous_inner(shape, strides):
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if size > 1 and stride != expected:
            return False
        expected *= size
    return True


def _disjoint_rows(pages, block, stride0, stride1, span):
    # A conservative span proof also protects holes in strided payloads.
    if (pages <= 1 or (block - 1) * stride1 + span <= stride0) and (
        block <= 1 or stride1 >= span
    ):
        return True
    # Search the shorter axis. For each displacement, the two nearest
    # multiples on the other axis suffice to detect overlapping row spans.
    # This keeps transposed page layouts bounded by the page size rather
    # than the potentially large number of allocated pages.
    if pages > block:
        pages, block, stride0, stride1 = block, pages, stride1, stride0
    for page_delta in range(1 - pages, pages):
        quotient = -(page_delta * stride0) // stride1
        for offset_delta in (quotient, quotient + 1):
            if abs(offset_delta) >= block:
                continue
            if page_delta == 0 and offset_delta == 0:
                continue
            if abs(page_delta * stride0 + offset_delta * stride1) < span:
                return False
    return True


def _plan(cache, slots, values, block_size):
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise ValueError("block_size must be an integer")
    if block_size <= 0:
        return None
    if cache.ndim < (1 if block_size == 1 else 2):
        raise ValueError("Cache rank does not match paged layout")
    if block_size != 1 and cache.shape[1] != block_size:
        raise ValueError("Cache block dimension must match block_size")
    if slots.ndim != 1 or values.ndim < 1:
        raise ValueError("Expected slots [tokens] and values [tokens, ...payload]")
    inner_shape = cache.shape[1:] if block_size == 1 else cache.shape[2:]
    inner = prod(inner_shape)
    if values.shape[0] != slots.shape[0] or prod(values.shape[1:]) != inner:
        raise ValueError("Values and slots must match cache payload and token count")
    if cache.device != slots.device or cache.device != values.device:
        raise ValueError("Cache, slots and values must use the same device")
    if cache.device.type != "npu" or cache.layout != torch.strided:
        return None
    if slots.layout != torch.strided or values.layout != torch.strided:
        return None
    if cache.dtype not in _DTYPES or slots.dtype not in (torch.int32, torch.int64):
        return None
    if values.dtype != cache.dtype and not (
        values.dtype == torch.float32 and cache.dtype in (torch.bfloat16, torch.float16)
    ):
        return None
    pages = cache.shape[0]
    block = 1 if block_size == 1 else block_size
    num_slots = pages * block
    if num_slots > _INT32_MAX:
        return None
    stride0 = cache.stride(0)
    stride1 = 0 if block_size == 1 else cache.stride(1)
    # Empty calls never index tensor rows or inspect absent storage.
    if slots.shape[0] == 0 or inner == 0:
        return _ScatterPlan(num_slots, inner, stride0, stride1, 1, 1, False)
    if num_slots == 0:
        raise ValueError("Cannot write nonempty values to an empty cache")
    if any(s <= 0 for s in cache.stride() + values.stride() + slots.stride()):
        return None
    cache_strides = cache.stride()[1:] if block_size == 1 else cache.stride()[2:]
    cache_contiguous = _contiguous_inner(inner_shape, cache_strides)
    value_contiguous = _contiguous_inner(values.shape[1:], values.stride()[1:])
    # Preserve the original reshape-copy path for multi-dimensional values.
    # Capability checks only describe it; copying happens in the writer.
    copy_values = cache_contiguous and not value_contiguous and values.ndim > 2
    strided = not (cache_contiguous and (value_contiguous or copy_values))
    strided_cache_rank = cache.ndim == (2 if block_size == 1 else 3)
    if block_size == 1 and cache.ndim == 3 and cache.shape[1] == 1:
        strided_cache_rank = True
    if strided and not (strided_cache_rank and values.ndim == 2):
        return None
    cache_inner = cache.stride(-1) if strided else 1
    value_inner = values.stride(-1) if strided else 1
    span = (inner - 1) * cache_inner + 1
    # Treat block_size=1 as a one-token page for the same disjointness proof.
    if not _disjoint_rows(pages, block, stride0, stride1 or span, span):
        return None
    try:
        storage = cache.untyped_storage()
        limit = storage.nbytes() // cache.element_size() - cache.storage_offset()
        max_offset = (pages - 1) * stride0 + (block - 1) * stride1 + span
        if max_offset > limit or max_offset > 2**63 - 1:
            return None
        if any(
            storage.data_ptr() == t.untyped_storage().data_ptr()
            for t in (values, slots)
        ):
            return None
    except (RuntimeError, NotImplementedError):
        return None
    return _ScatterPlan(
        num_slots,
        inner,
        stride0,
        stride1,
        cache_inner,
        value_inner,
        strided,
        copy_values,
    )


def paged_scatter_supports(cache, slots, values, block_size) -> bool:
    """Check metadata without launching, copying, casting or reading tensor data.

    Uses the writer's validation. Malformed shapes/devices raise ValueError;
    unsupported dtype/layout returns False. Unique valid slots are a caller
    precondition, not inspected here. No cross-call tensor state is retained.
    """
    logger.debug("GEMS_ASCEND PAGED_SCATTER_SUPPORTS")
    return _plan(cache, slots, values, block_size) is not None


def paged_scatter_triton(cache, slots, values, block_size) -> bool:
    """Mutate cache at unique valid slots; skip negative/out-of-range slots.

    Retains the four-argument boolean API. False means unsupported before any
    write; launch errors propagate and must never trigger backend retry.
    block_size=1 uses [rows, ...payload], otherwise [pages, block, ...payload].
    Contiguous inner payloads and positive disjoint 3-D strided payloads are
    supported, including storage offsets. Slots and values may have positive
    row strides. Cache/source shared storage is conservatively unsupported.
    Noncontiguous multi-dimensional values may be copied to flatten their
    payload for a contiguous-inner cache, preserving the original reshape path.

    Cache BF16/FP16/FP32/INT8 supports same-dtype values, plus FP32 values to
    BF16/FP16. Other conversions are explicitly unsupported. Floating stores
    preserve NaN and signed zero semantics; no reduction or slot-zero sentinel
    is used. Empty tokens/payload are no-ops. No autograd guarantee is made.
    """
    logger.debug("GEMS_ASCEND PAGED_SCATTER_TRITON")
    plan = _plan(cache, slots, values, block_size)
    if plan is None:
        return False
    n = slots.shape[0]
    if n == 0 or plan.inner == 0:
        return True
    if plan.copy_values:
        values = values.contiguous().view(n, plan.inner)
    block_inner = min(triton.next_power_of_2(plan.inner), 512)
    rows_per_prog = 8 if n > 8 else triton.next_power_of_2(max(n, 1))
    if plan.strided and n <= 64:
        # Small strided writes need independent programs for memory parallelism.
        rows_per_prog = 1
    _paged_scatter_kernel[
        (triton.cdiv(n, rows_per_prog), triton.cdiv(plan.inner, block_inner))
    ](
        cache,
        slots,
        values,
        n,
        plan.num_slots,
        plan.inner,
        plan.stride_block,
        plan.stride_off,
        values.stride(0),
        slots.stride(0),
        plan.cache_inner,
        plan.value_inner,
        STRIDED=plan.strided,
        BLOCK_SZ=block_size,
        ROWS=rows_per_prog,
        BLOCK_INNER=block_inner,
        num_warps=8,
    )
    return True

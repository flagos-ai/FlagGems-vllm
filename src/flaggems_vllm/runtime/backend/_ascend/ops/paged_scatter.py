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
dims are contiguous.

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
            slot = tl.load(slots_ptr + n).to(tl.int32)
            if (slot >= 0) & (slot < num_slots):
                row = slot // BLOCK_SZ
                off = slot - row * BLOCK_SZ
                v = tl.load(
                    values_ptr + n.to(tl.int64) * value_stride + o, mask=m_col, other=0
                )
                tl.store(
                    cache_ptr
                    + row.to(tl.int64) * stride_block
                    + off.to(tl.int64) * stride_off
                    + o,
                    v,
                    mask=m_col,
                )


def paged_scatter_triton(
    cache: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
) -> bool:
    """Scatter ``values`` rows into ``cache`` pages; returns False when the
    layout is unsupported and the caller must fall back to torch.

    ``block_size=1`` treats ``cache`` as ``[num_rows, ...inner]`` and scatters
    rows directly; otherwise ``cache`` must be ``[num_blocks, block_size,
    ...inner]``.
    """
    logger.debug("GEMS_ASCEND PAGED_SCATTER")
    if block_size <= 0:
        return False
    if cache.device.type != "npu":
        # CPU unit tests exercise the callers with fallback tensors.
        return False
    if block_size == 1:
        num_slots = cache.shape[0]
        inner_shape = cache.shape[1:]
        stride_block, stride_off = cache.stride(0), 0
    else:
        if cache.ndim < 2 or cache.shape[1] != block_size:
            return False
        num_slots = cache.shape[0] * block_size
        inner_shape = cache.shape[2:]
        stride_block, stride_off = cache.stride(0), cache.stride(1)

    inner_numel = 1
    for s in inner_shape:
        inner_numel *= s
    if inner_numel == 0:
        return True
    values = values.reshape(values.shape[0], *inner_shape)
    inner_ref = cache[0] if block_size == 1 else cache[0, 0]
    if not inner_ref.is_contiguous() or not values[0].is_contiguous():
        return False

    n = slots.shape[0]
    if n == 0:
        return True
    block_inner = min(triton.next_power_of_2(inner_numel), 512)
    # 8 rows per program amortises the ~70ns/program AIV scheduling cost that
    # dominated the one-row-per-program variant (600us -> ~80us at 8192 rows).
    rows_per_prog = 8 if n > 8 else triton.next_power_of_2(max(n, 1))
    _paged_scatter_kernel[
        (triton.cdiv(n, rows_per_prog), triton.cdiv(inner_numel, block_inner))
    ](
        cache,
        slots,
        values,
        n,
        num_slots,
        inner_numel,
        stride_block,
        stride_off,
        values.stride(0),
        BLOCK_SZ=block_size,
        ROWS=rows_per_prog,
        BLOCK_INNER=block_inner,
        num_warps=8,
    )
    return True

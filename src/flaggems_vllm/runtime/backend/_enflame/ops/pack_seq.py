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

from __future__ import annotations

import logging
import math
import struct

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.data_movement import contiguous_copy

logger = logging.getLogger(__name__)

# L600 (GCU400): 48 SPs; a program whose DSM footprint stays <= ~448KB is
# co-resident as dual program per SP, so a fixed 48-program grid with a
# strided chunk loop keeps full 48-way parallelism for every input size.
_NUM_SPC = 48
# Wide flat copy op width in lanes (blessed 1D streaming form, mirrors the
# unpack_seq kernel): rescan on the decode shapes picked 16384 lanes over
# 32768 for every dtype (-3.5%..-12% kernel-only); 16384 lanes of fp32 keep
# the per-program DSM peak (load + store staging) at 128KB, far below the
# 448KB dual-program budget.
_COPY_BLOCK = 16384
_NUM_STAGES = 2

# torch fp8 dtype -> (mant_bits, exp_bits, bias, max_e_field, max_m_at_top)
# for the pad encoder; max_* delimit the largest finite encoding.
_FP8_ELEMENT_FORMATS: dict = {}
for _dt, _fmt in (
    (getattr(torch, "float8_e4m3fn", None), (3, 4, 7, 15, 6)),
    (getattr(torch, "float8_e5m2", None), (2, 5, 15, 30, 3)),
):
    if _dt is not None:
        _FP8_ELEMENT_FORMATS[_dt] = _fmt
del _dt, _fmt


def _fp8_bits_from_value(
    value: float,
    mant_bits: int,
    exp_bits: int,
    bias: int,
    max_e_field: int,
    max_m_at_top: int,
) -> int:
    # GCU kernels cannot compile fp8-typed loads or fp32->fp8 casts, so the
    # pad byte is encoded on the host (pure bit math, no torch ops) and the
    # payload is moved through uint8 views. Saturating round-to-nearest-even
    # matches the generic CUDA path (cvt.rn.satfinite): -inf -> -448 (e4m3fn)
    # / -57344 (e5m2) instead of NaN, overflow clamps to max finite.
    try:
        raw = struct.pack("<f", float(value))
    except (OverflowError, ValueError):
        raw = struct.pack("<f", math.inf if value > 0 else -math.inf)
    fbits = struct.unpack("<I", raw)[0]
    sign = (fbits >> 24) & 0x80
    f32 = struct.unpack("<f", raw)[0]
    if math.isnan(f32):
        return 0x7F if exp_bits == 4 else 0x7E
    a = abs(f32)
    if a == 0.0:
        return sign
    if math.isinf(a):
        e_field, m_field = max_e_field, max_m_at_top
        return sign | (e_field << mant_bits) | m_field
    min_norm = 2.0 ** (1 - bias)
    if a < min_norm:
        # subnormal range: uniform grid of 2^(1-bias-mant_bits)
        ri = round(a / 2.0 ** (1 - bias - mant_bits))
        if ri >= (1 << mant_bits):
            e_field, m_field = 1, 0
        else:
            e_field, m_field = 0, ri
    else:
        # normal range: quantize the mantissa at the local exponent spacing
        e = max(math.floor(math.log2(a)), 1 - bias)
        q = round(a / 2.0 ** (e - mant_bits))
        if q == (1 << (mant_bits + 1)):
            q >>= 1
            e += 1
        e_field = e + bias
        m_field = q - (1 << mant_bits)
        if e_field > max_e_field or (e_field == max_e_field and m_field > max_m_at_top):
            e_field, m_field = max_e_field, max_m_at_top
    return sign | (e_field << mant_bits) | m_field


@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D] (uint8 view for 1-byte element paths)
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    n_chunks,
    rows_per_chunk,
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    PAD_IS_UINT8: tl.constexpr,
    BLOCK_B: tl.constexpr,  # rounded-up batch entries for prefix sum
    BLOCK: tl.constexpr,  # flat elements per maskless load/store
    NUM_SPC: tl.constexpr,
    WIDE: tl.constexpr,
):
    # Fixed-grid walk over the flat [B*Lmax, D] output row space (the 2D
    # [BLOCK_T, BLOCK_D] masked tile form and any fp8-typed access fail to
    # compile on the enflame toolchain; flat 1D accesses are the blessed
    # form). Each program strides over row chunks; within a chunk, batch
    # rows are moved as flat runs:
    #   - valid rows  [t, min(len_b, Lmax)): src and dst are both contiguous
    #     -> maskless full blocks + a single masked tail block
    #   - pad rows    [len_b, Lmax): constant fill, no loads at all
    #   - a full batch (len_b == Lmax) is source-contiguous with every
    #     following full batch, so consecutive full batches merge into one
    #     long flat run (bounded by the chunk's row budget)
    pid = tl.program_id(0)
    TR = B * Lmax
    off_b = tl.arange(0, BLOCK_B)
    if PAD_IS_UINT8:
        pad_vec = tl.full((BLOCK,), PAD_VALUE, tl.uint8)
    else:
        pad_vec = tl.full((BLOCK,), PAD_VALUE, tl.float32)
    for chunk in range(pid, n_chunks, NUM_SPC):
        # A chunk covers a contiguous output-row range [r0, r_end); one
        # masked prefix load yields the packed source row of the chunk start.
        r0 = chunk * rows_per_chunk
        r_end = tl.minimum(r0 + rows_per_chunk, TR)
        b = r0 // Lmax
        t = r0 - b * Lmax
        prev = tl.load(lengths_ptr + off_b, mask=off_b < b, other=0)
        in_row = tl.sum(prev, axis=0) + t
        r = r0
        while r < r_end:
            len_b = tl.load(lengths_ptr + b)
            copy_rows = tl.minimum(len_b, Lmax) - t
            if copy_rows > 0:
                run = copy_rows
                if len_b == Lmax:
                    # Batch b is completely full, so its remaining rows are
                    # source-contiguous with every following full batch:
                    # merge them into one flat run. The peek is bounded by
                    # the chunk's row budget — runs are clamped to the chunk
                    # anyway, so scanning past it is wasted work.
                    nb = b + 1
                    go = 1
                    budget = r_end - r
                    while (
                        (nb < B) & (go == 1) & (copy_rows + (nb - b) * Lmax <= budget)
                    ):
                        len_nb = tl.load(lengths_ptr + nb)
                        go = tl.where(len_nb == Lmax, 1, 0)
                        nb += go
                    run = copy_rows + (nb - b - 1) * Lmax
                run = tl.minimum(run, r_end - r)
                if WIDE:
                    n_elem = run.to(tl.int64) * D
                    src = in_row.to(tl.int64) * D
                    dst = r.to(tl.int64) * D
                else:
                    n_elem = run * D
                    src = in_row * D
                    dst = r * D
                # Block count always fits i32: >= 2^31 blocks would need a
                # > 2^46-element tensor, impossible on this hardware.
                n_blocks = (n_elem // BLOCK).to(tl.int32)
                for j in range(0, n_blocks):
                    if WIDE:
                        offs = j.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                    else:
                        offs = j * BLOCK + tl.arange(0, BLOCK)
                    tl.store(out_ptr + (dst + offs), tl.load(x_ptr + (src + offs)))
                if WIDE:
                    base_full = n_blocks.to(tl.int64) * BLOCK
                else:
                    base_full = n_blocks * BLOCK
                rem = n_elem - base_full
                if rem > 0:
                    offs = base_full + tl.arange(0, BLOCK)
                    m = offs < n_elem
                    tl.store(
                        out_ptr + (dst + offs),
                        tl.load(x_ptr + (src + offs), mask=m),
                        mask=m,
                    )
                r += run
                in_row += run
            else:
                # Rows [t, Lmax) of batch b are pure padding: constant fill.
                # A chunk may start inside the pad region (t > len_b), so
                # clamp the tracked source row back to the batch boundary.
                in_row += len_b - t
                fill_rows = tl.minimum(Lmax - t, r_end - r)
                if WIDE:
                    n_elem = fill_rows.to(tl.int64) * D
                    dst = r.to(tl.int64) * D
                else:
                    n_elem = fill_rows * D
                    dst = r * D
                n_blocks = (n_elem // BLOCK).to(tl.int32)
                for j in range(0, n_blocks):
                    if WIDE:
                        offs = j.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                    else:
                        offs = j * BLOCK + tl.arange(0, BLOCK)
                    tl.store(out_ptr + (dst + offs), pad_vec)
                if WIDE:
                    base_full = n_blocks.to(tl.int64) * BLOCK
                else:
                    base_full = n_blocks * BLOCK
                rem = n_elem - base_full
                if rem > 0:
                    offs = base_full + tl.arange(0, BLOCK)
                    m = offs < n_elem
                    tl.store(out_ptr + (dst + offs), pad_vec, mask=m)
                r += fill_rows
            b_next = r // Lmax
            if b_next > b:
                # Left batch b: skip its source rows clipped past Lmax
                # (explicit max_length smaller than len_b).
                in_row += tl.maximum(len_b - Lmax, 0)
            b = b_next
            t = r - b * Lmax


def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
    *,
    max_length: int | None = None,
) -> torch.Tensor:
    logger.debug("GEMS PACK_SEQ_TRITON")
    is_uint8 = x.dtype == torch.uint8
    fp8_format = None if is_uint8 else _FP8_ELEMENT_FORMATS.get(x.dtype)
    use_byte_path = is_uint8 or fp8_format is not None
    if is_uint8:
        assert (
            isinstance(pad_value, int) and 0 <= pad_value <= 255
        ), f"uint8 pack requires an integer pad in [0, 255], got {pad_value!r}"
        pad_constexpr: int | float = int(pad_value)
    elif fp8_format is not None:
        pad_constexpr = _fp8_bits_from_value(float(pad_value), *fp8_format)
    else:
        pad_constexpr = float(pad_value)

    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x

    B = lengths.numel()
    Lmax = int(lengths.max().item()) if max_length is None else max_length

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    if fp8_format is not None:
        if not x_reshaped.is_contiguous():
            x_reshaped = x_reshaped.contiguous()
        x_kernel = x_reshaped.view(torch.uint8)
        out_kernel = out.view(torch.uint8)
    else:
        x_kernel = x_reshaped
        out_kernel = out

    total_rows = B * Lmax
    if total_rows == 0 or D == 0:
        if len(original_shape) > 2:
            return out.reshape((B, Lmax) + original_shape[1:])
        return out

    # Fixed-grid chunking: 48 strided programs, each walking contiguous
    # output-row chunks of ~TR/48 rows; one O(B) prefix per chunk.
    rows_per_chunk = max(1, total_rows // _NUM_SPC)
    n_chunks = (total_rows + rows_per_chunk - 1) // rows_per_chunk
    # Element offsets can exceed i32 for very large tensors; widen the whole
    # address chain (single addptr per trap-note) instead of wrapping.
    wide = 1 if (total_rows * D > 2**31 - 1) or (N * D > 2**31 - 1) else 0

    # Rescan on the decode shapes: 4 warps win for <=2-byte elements, 2 warps
    # for 4-byte elements (fp32 nw4 regressed +14%).
    num_warps = 4 if x_kernel.element_size() <= 2 else 2

    _pack_seq_kernel[(_NUM_SPC,)](
        x_kernel,
        out_kernel,
        contiguous_copy(lengths, torch.int32),
        n_chunks,
        rows_per_chunk,
        B=B,
        Lmax=Lmax,
        D=D,
        PAD_VALUE=pad_constexpr,
        PAD_IS_UINT8=use_byte_path,
        BLOCK_B=triton.next_power_of_2(B),
        BLOCK=_COPY_BLOCK,
        NUM_SPC=_NUM_SPC,
        WIDE=wide,
        num_warps=num_warps,
        num_stages=_NUM_STAGES,
    )

    if len(original_shape) > 2:
        out = out.reshape((B, Lmax) + original_shape[1:])

    return out

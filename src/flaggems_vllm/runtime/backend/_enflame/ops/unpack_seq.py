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

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.data_movement import contiguous_copy

logger = logging.getLogger(__name__)

# L600 (GCU400): 48 SPs; a program whose DSM footprint stays <= ~448KB is
# co-resident as dual program per SP, so a fixed 48-program grid with a
# strided chunk loop keeps full 48-way parallelism for every input size.
_NUM_SPC = 48
# Wide flat copy op width in lanes (blessed 1D streaming form: BLOCK_ELEMS
# ~= 32768, grid=48 strided). 32768 lanes of fp16/bf16/fp8/int8 keep the
# per-program DSM peak (load + store staging) far below the 448KB budget.
_COPY_BLOCK = 32768
_NUM_WARPS = 2
_NUM_STAGES = 2


@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,  # [B, Lmax, D] contiguous
    out_ptr,  # [N, D] contiguous
    lengths_ptr,  # *i32, [B]
    n_chunks,
    rows_per_chunk,
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SPC: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    TP = B * Lmax
    off_b = tl.arange(0, BLOCK_B)
    for chunk in range(pid, n_chunks, NUM_SPC):
        # A chunk covers a contiguous packed-row range [r0, r_end); batches
        # crossed by the range are contiguous, so one masked prefix load per
        # chunk yields the output row of the chunk start.
        r0 = chunk * rows_per_chunk
        r_end = tl.minimum(r0 + rows_per_chunk, TP)
        b = r0 // Lmax
        t = r0 - b * Lmax
        prev = tl.load(lengths_ptr + off_b, mask=off_b < b, other=0)
        out_row = tl.sum(prev, axis=0) + t
        r = r0
        while r < r_end:
            len_b = tl.load(lengths_ptr + b)
            valid_b = len_b - t
            if valid_b > 0:
                run = valid_b
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
                        (nb < B) & (go == 1) & (valid_b + (nb - b) * Lmax <= budget)
                    ):
                        len_nb = tl.load(lengths_ptr + nb)
                        go = tl.where(len_nb == Lmax, 1, 0)
                        nb += go
                    run = valid_b + (nb - b - 1) * Lmax
                run = tl.minimum(run, r_end - r)
                if WIDE:
                    n_elem = run.to(tl.int64) * D
                    src = (b * Lmax + t).to(tl.int64) * D
                    dst = out_row.to(tl.int64) * D
                else:
                    n_elem = run * D
                    src = (b * Lmax + t) * D
                    dst = out_row * D
                # Block count always fits i32: >= 2^31 blocks would need a
                # > 2^46-element tensor, impossible on this hardware.
                n_blocks = (n_elem // BLOCK).to(tl.int32)
                for j in range(0, n_blocks):
                    if WIDE:
                        offs = j.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                    else:
                        offs = j * BLOCK + tl.arange(0, BLOCK)
                    tl.store(
                        out_ptr + (dst + offs), tl.load(packed_ptr + (src + offs))
                    )
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
                        tl.load(packed_ptr + (src + offs), mask=m),
                        mask=m,
                    )
                r += run
                out_row += run
            else:
                # Chunk starts inside batch b's padding: nothing to copy for
                # b, resume at the next batch's first row.
                out_row += len_b - t
                r = (b + 1) * Lmax
            b = r // Lmax
            t = r - b * Lmax


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
    *,
    total_tokens: int | None = None,
) -> torch.Tensor:
    logger.debug("GEMS UNPACK_SEQ_TRITON")
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    N = int(lengths.sum().item()) if total_tokens is None else total_tokens

    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)
    total_packed_rows = B * Lmax
    if N == 0 or total_packed_rows == 0 or D == 0:
        if len(original_shape) > 3:
            return out.reshape((N,) + original_shape[2:])
        return out

    rows_per_chunk = max(1, total_packed_rows // _NUM_SPC)
    n_chunks = (total_packed_rows + rows_per_chunk - 1) // rows_per_chunk
    # Element offsets can exceed i32 for very large tensors; widen the whole
    # address chain (single addptr per trap-note) instead of wrapping.
    wide = 1 if (total_packed_rows * D > 2**31 - 1) or (N * D > 2**31 - 1) else 0

    _unpack_seq_triton_kernel[(_NUM_SPC,)](
        packed_reshaped,
        out,
        contiguous_copy(lengths, torch.int32),
        n_chunks,
        rows_per_chunk,
        B=B,
        Lmax=Lmax,
        D=D,
        BLOCK_B=triton.next_power_of_2(B),
        BLOCK=_COPY_BLOCK,
        NUM_SPC=_NUM_SPC,
        WIDE=wide,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out

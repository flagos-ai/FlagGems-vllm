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

"""Ascend-optimized Qwen4 PLE state scatter.

[KernelGen] Auto-generated and tuned for Ascend910.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(
    state_ptr,
    indices_ptr,
    rows_ptr,
    write_mask_ptr,
    indices_stride,
    write_mask_stride,
    state_stride0,
    state_stride1,
    state_stride2,
    rows_stride0,
    rows_stride1,
    rows_stride2,
    num_cache_rows,
    hidden_size,
    state_width,
    HIDDEN_FASTEST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_elements = hidden_size * state_width
    wm = tl.load(write_mask_ptr + row * write_mask_stride).to(tl.int1)
    index = tl.load(indices_ptr + row * indices_stride).to(tl.int64)
    row_ok = wm & (index >= 0) & (index < num_cache_rows)
    valid = (offsets < row_elements) & row_ok
    safe_index = tl.minimum(tl.maximum(index, 0), num_cache_rows - 1)
    if HIDDEN_FASTEST:
        hidden = offsets % hidden_size
        width = offsets // hidden_size
    else:
        hidden = offsets // state_width
        width = offsets % state_width
    source = (
        rows_ptr + row * rows_stride0 + hidden * rows_stride1 + width * rows_stride2
    )
    destination = (
        state_ptr
        + safe_index * state_stride0
        + hidden * state_stride1
        + width * state_stride2
    )
    values = tl.load(source, mask=valid, other=0)
    tl.store(destination, values, mask=valid)


def _validate(state, name):
    if state.ndim != 3:
        raise ValueError(f"PLE {name} must be a rank-3 [rows, hidden, width] tensor")
    if state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"PLE {name} must use float16, bfloat16, or float32; got {state.dtype}"
        )


def ple_state_scatter_(
    state, indices, rows, *, write_mask=None, indices_are_safe=False
):
    del indices_are_safe
    _validate(state, "state")
    _validate(rows, "state rows")
    if state.device.type in ("cpu", "meta"):
        raise RuntimeError("Qwen4 PLE scatter requires a Triton accelerator")
    if indices.ndim != 1 or indices.device != state.device:
        raise ValueError(
            "PLE scatter indices must be a one-dimensional same-device tensor"
        )
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("PLE scatter indices must be int32 or int64")
    if state.shape[0] == 0:
        raise ValueError("PLE scatter requires at least one cache row")
    if rows.shape[0] != indices.numel() or tuple(rows.shape[1:]) != tuple(
        state.shape[1:]
    ):
        raise ValueError(
            "PLE state scatter rows and indices have incompatible shapes: "
            f"rows={tuple(rows.shape)}, indices={indices.numel()}, "
            f"state={tuple(state.shape)}"
        )
    if rows.device != state.device or rows.dtype != state.dtype:
        raise ValueError("PLE state scatter rows must match state device and dtype")
    if write_mask is None:
        raise NotImplementedError(
            "Qwen4 PLE scatter requires an explicit write_mask; "
            "derive duplicate/null semantics in the caller"
        )
    if write_mask.ndim != 1 or write_mask.numel() != indices.numel():
        raise ValueError("PLE state scatter write_mask must match indices")
    if write_mask.device != state.device or write_mask.dtype != torch.bool:
        raise ValueError("PLE state scatter write_mask must be same-device bool")

    n = indices.numel()
    if not n or state.shape[1] == 0 or state.shape[2] == 0:
        return state

    row_elements = state.shape[1] * state.shape[2]
    BLOCK_SIZE = 256
    _scatter_kernel[(n, triton.cdiv(row_elements, BLOCK_SIZE))](
        state,
        indices,
        rows,
        write_mask,
        indices.stride(0),
        write_mask.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        rows.stride(0),
        rows.stride(1),
        rows.stride(2),
        state.shape[0],
        state.shape[1],
        state.shape[2],
        HIDDEN_FASTEST=(state.stride(1) <= state.stride(2)),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return state

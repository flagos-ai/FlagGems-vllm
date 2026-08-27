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

"""MetaX-optimized Qwen4 PLE state scatter.

[KernelGen] Auto-generated and tuned for MetaX C-series.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(
    dst_ptr,
    indices_ptr,
    rows_ptr,
    wm_ptr,
    num_rows,
    row_elems,
    out_s0,
    out_s1,
    out_s2,
    rows_s0,
    rows_s1,
    rows_s2,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    h = offs // WIDTH
    w = offs % WIDTH
    elem_ok = offs < row_elems
    wm = tl.load(wm_ptr + i)
    idx = tl.load(indices_ptr + i).to(tl.int32)
    ok = wm & (idx >= 0) & (idx < num_rows)
    safe = tl.minimum(tl.maximum(idx, 0), num_rows - 1)
    valid = elem_ok & ok
    src = rows_ptr + i * rows_s0 + h * rows_s1 + w * rows_s2
    dst = dst_ptr + safe * out_s0 + h * out_s1 + w * out_s2
    v = tl.load(src, mask=valid, other=0)
    tl.store(dst, v, mask=valid)


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
        raise ValueError("PLE state scatter rows and indices have incompatible shapes")
    if rows.device != state.device or rows.dtype != state.dtype:
        raise ValueError("PLE state scatter rows must match state device and dtype")
    if write_mask is None:
        raise NotImplementedError("Qwen4 PLE scatter requires an explicit write_mask")
    if write_mask.ndim != 1 or write_mask.numel() != indices.numel():
        raise ValueError("PLE state scatter write_mask must match indices")
    if write_mask.device != state.device or write_mask.dtype != torch.bool:
        raise ValueError("PLE state scatter write_mask must be same-device bool")

    n = indices.numel()
    R, H, W = state.shape
    if not n or H == 0 or W == 0:
        return state

    row_elems = H * W
    BLOCK = 2048
    _scatter_kernel[(n,)](
        state,
        indices,
        rows,
        write_mask,
        R,
        row_elems,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        rows.stride(0),
        rows.stride(1),
        rows.stride(2),
        WIDTH=W,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return state

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

"""Mthreads-optimized Qwen4 PLE state scatter.

[KernelGen] Auto-generated and tuned for Mthreads S-series.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(
    dst,
    indices,
    rows,
    wmask,
    cache_rows,
    s0,
    s1,
    s2,
    HIDDEN: tl.constexpr,
    WIDTH: tl.constexpr,
    HP: tl.constexpr,
    BW: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.load(indices + pid).to(tl.int32)
    m = tl.load(wmask + pid)
    ok = (idx >= 0) & (idx < cache_rows) & m
    if ok:
        h = tl.arange(0, HP)
        w = tl.arange(0, BW)
        hmask = h < HIDDEN
        wmask2 = w < WIDTH
        src_offs = pid * (HIDDEN * WIDTH) + h[:, None] * WIDTH + w[None, :]
        dst_offs = idx * s0 + h[:, None] * s1 + w[None, :] * s2
        mm = hmask[:, None] & wmask2[None, :]
        v = tl.load(rows + src_offs, mask=mm, other=0.0)
        tl.store(dst + dst_offs, v, mask=mm)


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
    cache_rows, hidden, width = state.shape
    if not n or hidden == 0 or width == 0:
        return state

    s0, s1, s2 = state.stride()
    HP = triton.next_power_of_2(hidden)
    BW = triton.next_power_of_2(width)
    _scatter_kernel[(n,)](
        state,
        indices,
        rows,
        write_mask,
        cache_rows,
        s0,
        s1,
        s2,
        HIDDEN=hidden,
        WIDTH=width,
        HP=HP,
        BW=BW,
    )
    return state

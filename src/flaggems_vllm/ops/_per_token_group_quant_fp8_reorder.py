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

"""Independent group tiles for the optional FlagTree load-reorder path."""

import inspect
from functools import lru_cache

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None


def _supports_load_reorder():
    if tle is None:
        return False
    try:
        return "reorder" in inspect.signature(tle.range.__init__).parameters
    except (AttributeError, TypeError, ValueError):
        return False


_HAS_LOAD_REORDER = _supports_load_reorder()


@lru_cache(maxsize=None)
def _is_h20(device_index):
    return torch.cuda.get_device_name(device_index) in ("NVIDIA H20", "NVIDIA H20-3e")


def can_use_group_reorder(
    x, group_size, fp8_dtype, eps, column_major_scales, scale_ue8m0
):
    # This fixed schedule is qualified for these H20 shapes. Retain the existing
    # implementation for other layouts, devices, dtypes and semantic options.
    return (
        _HAS_LOAD_REORDER
        and x.device.type == "cuda"
        and x.dtype == torch.bfloat16
        and fp8_dtype == torch.float8_e4m3fn
        and x.ndim == 2
        and x.shape[0] in (2048, 4096, 8192)
        and x.shape[1] == 13824
        and group_size == 512
        and eps == 1e-10
        and not column_major_scales
        and not scale_ue8m0
        and x.is_contiguous()
        and _is_h20(x.device.index)
    )


@triton.jit
def _clamp_like_upstream(value):
    # A symmetric constant clip can map NaNs to the upper bound. Upstream's
    # runtime-bound max/min instead maps them to the lower bound; retain that
    # sequence explicitly, including for Inf/Inf during normalization.
    return tl.inline_asm_elementwise(
        "max.f32 $0, $1, 0fC3E00000; min.f32 $0, $0, 0f43E00000;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _quantize_group_tiles(
    X,
    Q,
    S,
    TOTAL: tl.constexpr,
    M: tl.constexpr,
    NGROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    EPS: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr,
    UE8M0: tl.constexpr,
    REORDER: tl.constexpr,
):
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    base = tl.program_id(0) * (4 * ROWS)
    # Rotate the results through four accumulators. Stores remain outside the
    # unrolled loop so they cannot block load motion between independent tiles.
    q0 = tl.full((ROWS, BLOCK), 0, tl.uint8)
    q1 = tl.full((ROWS, BLOCK), 1, tl.uint8)
    q2 = tl.full((ROWS, BLOCK), 2, tl.uint8)
    q3 = tl.full((ROWS, BLOCK), 3, tl.uint8)
    s0 = tl.full((ROWS,), 0, tl.float32)
    s1 = tl.full((ROWS,), 1, tl.float32)
    s2 = tl.full((ROWS,), 2, tl.float32)
    s3 = tl.full((ROWS,), 3, tl.float32)
    for step in tle.range(0, 4, loop_unroll_factor=4, reorder=REORDER):
        group = base + step * ROWS + rows
        mask = (group[:, None] < TOTAL) & (cols[None, :] < GROUP_SIZE)
        values = tl.load(
            X + group[:, None] * GROUP_SIZE + cols[None, :], mask, other=0.0
        ).to(tl.float32)
        maximum = tl.maximum(tl.max(tl.abs(values), axis=1), EPS)
        scale = maximum * (1.0 / 448.0)
        if UE8M0:
            scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1e-10))))
        quantized = _clamp_like_upstream(values / scale[:, None])
        quantized = quantized.to(Q.dtype.element_ty).to(tl.uint8, bitcast=True)
        q0, q1, q2, q3 = q1, q2, q3, quantized
        s0, s1, s2, s3 = s1, s2, s3, scale
    for step in tl.static_range(4):
        group = base + step * ROWS + rows
        if step == 0:
            quantized, scale = q0, s0
        elif step == 1:
            quantized, scale = q1, s1
        elif step == 2:
            quantized, scale = q2, s2
        else:
            quantized, scale = q3, s3
        mask = (group[:, None] < TOTAL) & (cols[None, :] < GROUP_SIZE)
        tl.store(
            Q + group[:, None] * GROUP_SIZE + cols[None, :],
            quantized.to(Q.dtype.element_ty, bitcast=True),
            mask,
        )
        if COLUMN_MAJOR:
            scale_offset = (group % NGROUPS) * M + group // NGROUPS
        else:
            scale_offset = group
        tl.store(S + scale_offset, scale, group < TOTAL)


def quantize_group_tiles(
    x,
    group_size,
    *,
    eps=1e-10,
    column_major_scales=False,
    scale_ue8m0=False,
    rows=8,
    num_warps=4,
    reorder=True,
):
    """Fixed four-step schedule; the public guard owns its qualified scope.

    No online autotuning is used: ROWS=8 and num_warps=4 bound the register
    footprint of the fully-unrolled loop on the admitted H20 shapes. The private
    options retain the same-config OFF/ON and tail regression controls.
    """
    groups_per_row = x.shape[1] // group_size
    total = x.numel() // group_size
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    if column_major_scales:
        s = torch.empty(
            (groups_per_row, x.shape[0]), device=x.device, dtype=torch.float32
        ).permute(1, 0)
    else:
        s = torch.empty(
            (x.shape[0], groups_per_row), device=x.device, dtype=torch.float32
        )
    _quantize_group_tiles[(triton.cdiv(total, rows * 4),)](
        x,
        q,
        s,
        TOTAL=total,
        M=x.shape[0],
        NGROUPS=groups_per_row,
        GROUP_SIZE=group_size,
        BLOCK=triton.next_power_of_2(group_size),
        ROWS=rows,
        EPS=eps,
        COLUMN_MAJOR=column_major_scales,
        UE8M0=scale_ue8m0,
        REORDER=reorder,
        num_warps=num_warps,
        num_stages=1,
    )
    return q, s

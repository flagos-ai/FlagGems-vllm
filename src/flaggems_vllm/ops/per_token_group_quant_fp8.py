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

import inspect
import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils.device_info import kernel_supports_fp8_e4m3

if kernel_supports_fp8_e4m3():
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = tl


def _supports_load_reorder():
    if tle is None:
        return False
    try:
        return "reorder" in inspect.signature(tle.range.__init__).parameters
    except (AttributeError, TypeError, ValueError):
        return False


_HAS_LOAD_REORDER = _supports_load_reorder()


def _contiguous_group_config(num_groups, group_size, scale_ue8m0):
    # Bound live values using the group width and available parallel work.
    # Fixed schedules keep same-configuration OFF/ON comparisons reproducible;
    # this is an explicit no-autotune exemption, independent of GPU model.
    if num_groups < 128:
        groups, steps, warps = 1, 2, 1
    elif group_size <= 128:
        groups, steps, warps = 1, 4, 1
    elif group_size == 256:
        groups, steps, warps = (16, 1, 4) if scale_ue8m0 else (8, 2, 4)
    else:
        groups, steps, warps = 8, 2 if scale_ue8m0 else 4, 4
    return groups, steps if _HAS_LOAD_REORDER else 1, warps


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += row * y_row_stride + row_g_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax * (1.0 / fp8_max)

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    group_id = g_id % groups_per_row

    y_ptr += row * y_row_stride + group_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += group_id * y_s_col_stride + row

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax * (1.0 / fp8_max)

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size: tl.constexpr,
    y_num_columns: tl.constexpr,
    y_row_stride: tl.constexpr,
    eps: tl.constexpr,
    fp8_min,
    fp8_max,
    scale_ue8m0: tl.constexpr,
    BLOCK: tl.constexpr,
    NGROUPS: tl.constexpr,
    TOTAL_GROUPS: tl.constexpr = 0,
    UNROLL: tl.constexpr = 1,
    REORDER: tl.constexpr = False,
    INV_FP8_MAX: tl.constexpr = 0.0,
):
    if TOTAL_GROUPS:
        group_ids = tl.arange(0, NGROUPS)
        cols = tl.arange(0, BLOCK)
        base = tl.program_id(0) * (UNROLL * NGROUPS)
        if UNROLL == 1:
            group = base + group_ids
            offsets = group[:, None] * group_size + cols[None, :]
            mask = (group[:, None] < TOTAL_GROUPS) & (cols[None, :] < group_size)
            y = tl.load(y_ptr + offsets, mask, other=0.0).to(tl.float32)
            maximum = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
            scale = maximum * INV_FP8_MAX
            if scale_ue8m0:
                scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1e-10))))
            q = tl.clamp(y / scale[:, None], fp8_min, fp8_max).to(
                y_q_ptr.dtype.element_ty
            )
            tl.store(y_q_ptr + offsets, q, mask)
            tl.store(y_s_ptr + group, scale, group < TOTAL_GROUPS)
        else:
            q0 = tl.full((NGROUPS, BLOCK), 0, tl.uint8)
            q1 = tl.full((NGROUPS, BLOCK), 1, tl.uint8)
            q2 = tl.full((NGROUPS, BLOCK), 2, tl.uint8)
            q3 = tl.full((NGROUPS, BLOCK), 3, tl.uint8)
            s0 = tl.full((NGROUPS,), 0, tl.float32)
            s1 = tl.full((NGROUPS,), 1, tl.float32)
            s2 = tl.full((NGROUPS,), 2, tl.float32)
            s3 = tl.full((NGROUPS,), 3, tl.float32)
            for step in tle.range(
                0, UNROLL, loop_unroll_factor=UNROLL, reorder=REORDER
            ):
                group = base + step * NGROUPS + group_ids
                mask = (group[:, None] < TOTAL_GROUPS) & (cols[None, :] < group_size)
                y = tl.load(
                    y_ptr + group[:, None] * group_size + cols[None, :], mask, other=0.0
                ).to(tl.float32)
                maximum = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
                scale = maximum * INV_FP8_MAX
                if scale_ue8m0:
                    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1e-10))))
                q = (
                    tl.clamp(y / scale[:, None], fp8_min, fp8_max)
                    .to(y_q_ptr.dtype.element_ty)
                    .to(tl.uint8, bitcast=True)
                )
                if UNROLL == 2:
                    q0, q1 = q1, q
                    s0, s1 = s1, scale
                else:
                    q0, q1, q2, q3 = q1, q2, q3, q
                    s0, s1, s2, s3 = s1, s2, s3, scale
            # Keep stores after the load/reduction loop to permit load motion
            # between independent groups without crossing visible writes.
            for step in tl.static_range(UNROLL):
                group = base + step * NGROUPS + group_ids
                if step == 0:
                    q, scale = q0, s0
                elif step == 1:
                    q, scale = q1, s1
                elif step == 2:
                    q, scale = q2, s2
                else:
                    q, scale = q3, s3
                mask = (group[:, None] < TOTAL_GROUPS) & (cols[None, :] < group_size)
                tl.store(
                    y_q_ptr + group[:, None] * group_size + cols[None, :],
                    q.to(y_q_ptr.dtype.element_ty, bitcast=True),
                    mask,
                )
                tl.store(y_s_ptr + group, scale, group < TOTAL_GROUPS)
        return

    groups_per_row = y_num_columns // group_size
    programs_per_row = groups_per_row // NGROUPS

    pid = tl.program_id(0)
    row = pid // programs_per_row
    program_id = pid % programs_per_row

    start_group = program_id * NGROUPS
    start_gid = row * groups_per_row + start_group

    group_ids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (
        row * y_row_stride
        + start_group * group_size
        + group_ids[:, None] * group_size
        + cols[None, :]
    )
    mask = cols[None, :] < group_size

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _absmax * (1.0 / fp8_max)

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    output_offsets = (
        start_gid * group_size + group_ids[:, None] * group_size + cols[None, :]
    )

    tl.store(y_q_ptr + output_offsets, y_q, mask=mask)
    tl.store(y_s_ptr + start_gid + group_ids, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    programs_per_row = groups_per_row // NGROUPS

    pid = tl.program_id(0)
    row = pid // programs_per_row
    program_id = pid % programs_per_row

    start_group = program_id * NGROUPS
    start_gid = row * groups_per_row + start_group

    group_ids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (
        row * y_row_stride
        + start_group * group_size
        + group_ids[:, None] * group_size
        + cols[None, :]
    )
    mask = cols[None, :] < group_size

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _absmax * (1.0 / fp8_max)

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    output_offsets = (
        start_gid * group_size + group_ids[:, None] * group_size + cols[None, :]
    )
    scale_offsets = (start_group + group_ids) * y_s_col_stride + row

    tl.store(y_q_ptr + output_offsets, y_q, mask=mask)
    tl.store(y_s_ptr + scale_offsets, y_s)


def _groups_per_program(x: torch.Tensor, group_size: int) -> int:
    groups_per_row = x.shape[-1] // group_size
    for groups in (8, 4, 2):
        if groups_per_row % groups == 0:
            return groups
    return 1


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS PER TOKEN GROUP QUANT FP8")
    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    num_groups = x.numel() // group_size

    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    if num_groups == 0:
        return x_q, x_s

    if (
        x.dtype == torch.bfloat16
        and fp8_dtype == torch.float8_e4m3fn
        and x.ndim == 2
        and x.is_contiguous()
        and not column_major_scales
        and group_size in (64, 128, 256, 512)
        and isinstance(scale_ue8m0, bool)
    ):
        groups, steps, warps = _contiguous_group_config(
            num_groups, group_size, scale_ue8m0
        )
        _per_token_group_quant_fp8_vec[(triton.cdiv(num_groups, groups * steps),)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            BLOCK=triton.next_power_of_2(group_size),
            NGROUPS=groups,
            TOTAL_GROUPS=num_groups,
            UNROLL=steps,
            # Power-of-two scale rounding increases live arithmetic; retain
            # unrolling but avoid the measured reorder regressions in this mode.
            REORDER=_HAS_LOAD_REORDER and not scale_ue8m0,
            INV_FP8_MAX=1.0 / fp8_max,
            num_warps=warps,
            num_stages=1,
        )
        return x_q, x_s

    block = triton.next_power_of_2(group_size)
    num_warps = min(max(block // 256, 1), 8)
    groups_per_program = _groups_per_program(x, group_size)
    grid = (num_groups // groups_per_program,)

    if column_major_scales:
        if groups_per_program > 1:
            kernel = _per_token_group_quant_fp8_colmajor_vec
            kernel[grid](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=block,
                NGROUPS=groups_per_program,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            kernel = _per_token_group_quant_fp8_colmajor
            kernel[grid](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=block,
                num_warps=num_warps,
                num_stages=1,
            )
    elif groups_per_program > 1:
        kernel = _per_token_group_quant_fp8_vec
        kernel[grid](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            BLOCK=block,
            NGROUPS=groups_per_program,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        kernel = _per_token_group_quant_fp8
        kernel[grid](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )

    return x_q, x_s

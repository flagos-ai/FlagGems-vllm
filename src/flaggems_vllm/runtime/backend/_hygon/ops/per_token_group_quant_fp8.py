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

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Retain the software FP8 encoder used for gfx936, where generic conversion
# expands to a longer emulated sequence. Sharing it preserves that optimization;
# its benefit is separate from the load-reorder window below.
from flaggems_vllm.ops.per_token_group_quant_fp8 import (
    _HAS_LOAD_REORDER,
    _f32_to_fp8_e4m3fn,
    _quant_groups,
    _vendor_ngroups_per_program,
    _vendor_reorder_launch_config,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils.device_info import get_device_capability, get_sm_count

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = tl


if torch_device_fn.is_available() and get_device_capability() >= (9, 0):
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


@triton.jit
def _per_token_group_quant_fp8_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_row_stride,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    inv_fp8_max: tl.constexpr,
    scale_ue8m0: tl.constexpr,
    subnormal_scale: tl.constexpr,
    fast_cvt: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    NGROUPS: tl.constexpr,
    TOTAL_GROUPS: tl.constexpr = 0,
    UNROLL: tl.constexpr = 1,
    REORDER: tl.constexpr = False,
):
    if UNROLL == 1:
        pid = tl.program_id(0)
        programs_per_row = GROUPS_PER_ROW // NGROUPS
        row = pid // programs_per_row
        pg = pid % programs_per_row

        start_gid = row * GROUPS_PER_ROW + pg * NGROUPS

        gids = tl.arange(0, NGROUPS)
        cols = tl.arange(0, BLOCK)
        offsets = (
            row.to(tl.int64) * y_row_stride
            + (pg.to(tl.int64) * NGROUPS + gids[:, None]) * GROUP_SIZE
            + cols[None, :]
        )
        if BLOCK == GROUP_SIZE:
            y = tl.load(y_ptr + offsets).to(tl.float32)
        else:
            y = tl.load(y_ptr + offsets, mask=cols[None, :] < GROUP_SIZE, other=0.0).to(
                tl.float32
            )

        y_q, y_s = _quant_groups(
            y, eps, fp8_min, fp8_max, inv_fp8_max, scale_ue8m0, subnormal_scale
        )
        if fast_cvt:
            y_q = _f32_to_fp8_e4m3fn(y_q).to(y_q_ptr.dtype.element_ty, bitcast=True)
        else:
            y_q = y_q.to(y_q_ptr.dtype.element_ty)

        out_offsets = (
            start_gid.to(tl.int64) * GROUP_SIZE
            + gids[:, None] * GROUP_SIZE
            + cols[None, :]
        )
        if BLOCK == GROUP_SIZE:
            tl.store(y_q_ptr + out_offsets, y_q)
        else:
            tl.store(y_q_ptr + out_offsets, y_q, mask=cols[None, :] < GROUP_SIZE)
        tl.store(y_s_ptr + start_gid + gids, y_s)

    else:
        # The same portable independent-group loop on every backend. Reuse
        # the backend's existing quantization and FP8 conversion helpers.
        pid = tl.program_id(0)
        gids = tl.arange(0, NGROUPS)
        cols = tl.arange(0, BLOCK)
        q0 = tl.full((NGROUPS, BLOCK), 0, y_q_ptr.dtype.element_ty)
        q1 = tl.full((NGROUPS, BLOCK), 0, y_q_ptr.dtype.element_ty)
        s0 = tl.full((NGROUPS,), 0, tl.float32)
        s1 = tl.full((NGROUPS,), 0, tl.float32)
        for step in tle.range(0, UNROLL, loop_unroll_factor=UNROLL, reorder=REORDER):
            group = pid * NGROUPS * UNROLL + step * NGROUPS + gids
            if GROUP_SIZE == 128:
                offset = group[:, None].to(tl.int64) * GROUP_SIZE + cols[None, :]
            else:
                row = group // GROUPS_PER_ROW
                row_group = group % GROUPS_PER_ROW
                offset = (
                    row[:, None].to(tl.int64) * y_row_stride
                    + row_group[:, None].to(tl.int64) * GROUP_SIZE
                    + cols[None, :]
                )
            mask = (group[:, None] < TOTAL_GROUPS) & (cols[None, :] < GROUP_SIZE)
            y = tl.load(y_ptr + offset, mask, other=0.0).to(tl.float32)
            q, scale = _quant_groups(
                y, eps, fp8_min, fp8_max, inv_fp8_max, scale_ue8m0, subnormal_scale
            )
            if fast_cvt:
                q = _f32_to_fp8_e4m3fn(q).to(y_q_ptr.dtype.element_ty, bitcast=True)
            else:
                q = q.to(y_q_ptr.dtype.element_ty)
            if step == 0:
                q0, s0 = q, scale
            else:
                q1, s1 = q, scale
        for step in tl.static_range(0, UNROLL):
            group = pid * NGROUPS * UNROLL + step * NGROUPS + gids
            q = q0 if step == 0 else q1
            scale = s0 if step == 0 else s1
            mask = (group[:, None] < TOTAL_GROUPS) & (cols[None, :] < GROUP_SIZE)
            tl.store(
                y_q_ptr + group[:, None].to(tl.int64) * GROUP_SIZE + cols[None, :],
                q,
                mask,
            )
            tl.store(y_s_ptr + group, scale, group < TOTAL_GROUPS)


@triton.jit
def _per_token_group_quant_fp8_colmajor_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    inv_fp8_max: tl.constexpr,
    scale_ue8m0: tl.constexpr,
    subnormal_scale: tl.constexpr,
    fast_cvt: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    programs_per_row = GROUPS_PER_ROW // NGROUPS
    row = pid // programs_per_row
    pg = pid % programs_per_row

    start_gid = row * GROUPS_PER_ROW + pg * NGROUPS

    gids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (
        row.to(tl.int64) * y_row_stride
        + (pg.to(tl.int64) * NGROUPS + gids[:, None]) * GROUP_SIZE
        + cols[None, :]
    )
    if BLOCK == GROUP_SIZE:
        y = tl.load(y_ptr + offsets).to(tl.float32)
    else:
        y = tl.load(y_ptr + offsets, mask=cols[None, :] < GROUP_SIZE, other=0.0).to(
            tl.float32
        )

    y_q, y_s = _quant_groups(
        y, eps, fp8_min, fp8_max, inv_fp8_max, scale_ue8m0, subnormal_scale
    )
    if fast_cvt:
        y_q = _f32_to_fp8_e4m3fn(y_q).to(y_q_ptr.dtype.element_ty, bitcast=True)
    else:
        y_q = y_q.to(y_q_ptr.dtype.element_ty)

    out_offsets = (
        start_gid.to(tl.int64) * GROUP_SIZE + gids[:, None] * GROUP_SIZE + cols[None, :]
    )
    scale_offsets = (pg * NGROUPS + gids) * y_s_col_stride + row
    if BLOCK == GROUP_SIZE:
        tl.store(y_q_ptr + out_offsets, y_q)
    else:
        tl.store(y_q_ptr + out_offsets, y_q, mask=cols[None, :] < GROUP_SIZE)
    tl.store(y_s_ptr + scale_offsets, y_s)


def _ngroups_per_program(
    total_groups: int, groups_per_row: int, group_size: int
) -> int:
    return _vendor_ngroups_per_program(
        total_groups, groups_per_row, group_size, get_sm_count()
    )


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
    # The kernels flatten the leading dims into rows addressed by one uniform
    # row stride; reject layouts that do not collapse that way instead of
    # silently reading wrong addresses.
    assert (
        x.dim() <= 2 or x.is_contiguous()
    ), "`x` with more than 2 dims must be contiguous"
    row_stride = x.stride(-2) if x.dim() >= 2 else 0

    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    inv_fp8_max = 1.0 / fp8_max
    # When `fp8_max` is huge (e.g. the fp32 fallback), `inv_fp8_max` is a
    # subnormal fp32 and so is `y_s = _absmax * inv_fp8_max`; some backends
    # flush subnormal fp32 divisors to zero, which turns `y / y_s` into inf.
    subnormal_scale = inv_fp8_max < torch.finfo(torch.float32).tiny
    fast_cvt = fp8_dtype == torch.float8_e4m3fn

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    num_groups = x.numel() // group_size
    groups_per_row = x.shape[-1] // group_size

    if column_major_scales:
        assert x.dim() == 2, "column_major_scales only supports a 2-dim `x`"
        shape = (groups_per_row,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (groups_per_row,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    block = triton.next_power_of_2(group_size)
    ngroups = _ngroups_per_program(num_groups, groups_per_row, group_size)
    num_warps = min(max((ngroups * block) // 1024, 1), 4)
    grid = (num_groups // ngroups,)
    unroll = 1
    window_config = _vendor_reorder_launch_config(
        x,
        fp8_dtype,
        column_major_scales,
        group_size in (256, 512) or (group_size == 128 and scale_ue8m0),
        num_groups,
        block,
        _HAS_LOAD_REORDER,
    )
    if window_config is not None:
        ngroups, unroll, num_warps, grid = window_config

    if column_major_scales:
        _per_token_group_quant_fp8_colmajor_vec[grid](
            x,
            x_q,
            x_s,
            row_stride,
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            inv_fp8_max=inv_fp8_max,
            scale_ue8m0=scale_ue8m0,
            subnormal_scale=subnormal_scale,
            fast_cvt=fast_cvt,
            GROUP_SIZE=group_size,
            BLOCK=block,
            GROUPS_PER_ROW=groups_per_row,
            NGROUPS=ngroups,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _per_token_group_quant_fp8_vec[grid](
            x,
            x_q,
            x_s,
            row_stride,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            inv_fp8_max=inv_fp8_max,
            scale_ue8m0=scale_ue8m0,
            subnormal_scale=subnormal_scale,
            fast_cvt=fast_cvt,
            GROUP_SIZE=group_size,
            BLOCK=block,
            GROUPS_PER_ROW=groups_per_row,
            NGROUPS=ngroups,
            TOTAL_GROUPS=num_groups,
            UNROLL=unroll,
            REORDER=_HAS_LOAD_REORDER,
            num_warps=num_warps,
            num_stages=1,
        )

    return x_q, x_s

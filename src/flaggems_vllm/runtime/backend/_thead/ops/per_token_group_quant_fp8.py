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

from flaggems_vllm.runtime import torch_device_fn

if torch_device_fn.is_available():
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


@triton.jit
def _float_to_e4m3fn_bits(x):
    """Convert pre-clamped (|x| <= 448) f32 values to e4m3fn bits (0..255).

    FlagTree on PPU has no working native fp8e4m3fn conversion, so the byte
    is built with integer ops directly from the f32 bit pattern. Rounding is
    round-to-nearest-even with saturation, matching torch's float8_e4m3fn
    cast bit-for-bit (single rounding, no f16 intermediate; verified by an
    exhaustive probe over all 65536 f16 patterns and a wide f32 sweep).
    NaN inputs are not handled: production values are finite after clamping.
    """
    xb = x.to(tl.int32, bitcast=True)
    s = (xb >> 31) & 1
    e8 = (xb >> 23) & 0xFF
    m23 = xb & 0x7FFFFF
    # fp8-normal region: f32 unbiased exponent >= -6 (e8 >= 121). The rebased
    # field carries mantissa overflow into the exponent automatically.
    c = ((e8 << 23) | m23) - (120 << 23)
    q_norm = (c + 0x7FFFF + ((c >> 20) & 1)) >> 20
    q_norm = tl.minimum(q_norm, 0x7E)
    # fp8-subnormal region: 117 <= e8 <= 120, RNE shift with implicit 1 bit.
    sh = tl.maximum(141 - e8, 1)
    n = (1 << 23) | m23
    q_sub = (n + (1 << (sh - 1)) - 1 + ((n >> sh) & 1)) >> sh
    # e8 <= 116 (incl. f32 subnormals): magnitude < 2^-10 rounds to +-0.
    q = tl.where(e8 >= 121, q_norm, tl.where(e8 >= 117, q_sub, 0))
    return (s << 7) | (q & 0x7F)


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
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = _float_to_e4m3fn_bits(tl.clamp(y / y_s, fp8_min, fp8_max)).to(tl.uint8)

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
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = _float_to_e4m3fn_bits(tl.clamp(y / y_s, fp8_min, fp8_max)).to(tl.uint8)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_vec(
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
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = _float_to_e4m3fn_bits(tl.clamp(y / y_s[:, None], fp8_min, fp8_max)).to(
        tl.uint8
    )
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
    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = _float_to_e4m3fn_bits(tl.clamp(y / y_s[:, None], fp8_min, fp8_max)).to(
        tl.uint8
    )
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
    logger.debug("GEMS THEAD PER TOKEN GROUP QUANT FP8")
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
    # FlagTree cannot take a float8_e4m3fn pointer; store through a uint8 view.
    x_q_arg = x_q.view(torch.uint8)
    num_groups = x.numel() // group_size

    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    block = triton.next_power_of_2(group_size)
    num_warps = min(max(block // 256, 1), 8)
    groups_per_program = _groups_per_program(x, group_size)
    grid = (num_groups // groups_per_program,)

    if column_major_scales:
        if groups_per_program > 1:
            kernel = _per_token_group_quant_fp8_colmajor_vec
            kernel[grid](
                x,
                x_q_arg,
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
                x_q_arg,
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
            x_q_arg,
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
            x_q_arg,
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

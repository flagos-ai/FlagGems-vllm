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

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.device_info import kernel_supports_fp8_e4m3

if kernel_supports_fp8_e4m3():
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("per_token_group_quant_fp8_rows"),
    key=["M", "N", "DTYPE", "USE_UE8M0", "COLUMN_MAJOR"],
)
@triton.jit
def _per_token_group_quant_fp8_rows(
    X,
    Q,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    eps,
    fp8_min,
    fp8_max,
    DTYPE: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr,
    BROWS: tl.constexpr,
):
    # The host restricts this path to contiguous group-128 inputs with fewer
    # than 2**31 elements. Every element and scale offset fits signed int32.
    rows = tl.program_id(0) * BROWS + tl.arange(0, BROWS)
    group = tl.program_id(1)
    cols = group * 128 + tl.arange(0, 128)
    values = tl.load(X + rows[:, None] * N + cols[None, :]).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), eps)
    scales = amax * (1.0 / fp8_max)
    if USE_UE8M0:
        scales = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scales), 1e-10))))
    quantized = tl.clamp(values / scales[:, None], fp8_min, fp8_max)
    tl.store(
        Q + rows[:, None] * N + cols[None, :],
        quantized.to(Q.dtype.element_ty),
    )
    if COLUMN_MAJOR:
        tl.store(S + group * M + rows, scales)
    else:
        tl.store(S + rows * (N // 128) + group, scales)


def _prune_quant_tiles(configs, named_args, **kwargs):
    return [
        config
        for config in configs
        if named_args["N"] % (128 * config.kwargs["BG"]) == 0
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("per_token_group_quant_fp8_tile"),
    key=["M", "N", "DTYPE", "COL", "UE"],
    prune_configs_by={"early_config_prune": _prune_quant_tiles},
    use_cuda_graph=True,
)
@triton.jit
def _per_token_group_quant_fp8_tile(
    X,
    Q,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS,
    MIN,
    MAX,
    COL: tl.constexpr,
    UE: tl.constexpr,
    BR: tl.constexpr,
    BG: tl.constexpr,
    DTYPE: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    groups = tl.program_id(1) * BG + tl.arange(0, BG)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    offs = rows[:, None, None] * N + cols[None, :, :]
    y = tl.load(X + offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), axis=2), EPS)
    scale = amax * (1.0 / MAX)
    if UE:
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1e-10))))
    quant = tl.clamp(y / scale[:, :, None], MIN, MAX).to(Q.dtype.element_ty)
    tl.store(Q + offs, quant)
    so = (
        groups[None, :] * M + rows[:, None]
        if COL
        else rows[:, None] * (N // 128) + groups[None, :]
    )
    tl.store(S + so, scale)


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

    block = triton.next_power_of_2(group_size)
    num_warps = min(max(block // 256, 1), 8)
    groups_per_program = _groups_per_program(x, group_size)
    grid = (num_groups // groups_per_program,)

    native_vector_input = (
        runtime.device.vendor_name == "nvidia"
        and group_size == 128
        and x.ndim == 2
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16)
        and fp8_dtype == torch.float8_e4m3fn
        and SUPPORTED_FP8_DTYPE == torch.float8_e4m3fn
    )
    # Narrow group counts lack column batching. Eight rows share one program
    # and write adjacent column-major scales, reducing program count fourfold.
    if (
        native_vector_input
        and groups_per_program == 2
        and x.shape[0] >= 512
        and x.shape[0] % 8 == 0
        and x.numel() < 2**31
        and isinstance(scale_ue8m0, bool)
        and isinstance(column_major_scales, bool)
        and torch.cuda.get_device_capability(x.device) == (9, 0)
    ):
        _per_token_group_quant_fp8_rows[
            lambda meta: (x.shape[0] // meta["BROWS"], x.shape[1] // 128)
        ](
            x,
            x_q,
            x_s,
            x.shape[0],
            x.shape[1],
            eps,
            fp8_min,
            fp8_max,
            DTYPE=str(x.dtype),
            USE_UE8M0=scale_ue8m0,
            COLUMN_MAJOR=column_major_scales,
        )
        return x_q, x_s

    if (
        native_vector_input
        and x.shape[0] >= 4096
        and x.shape[0] % 8 == 0
        and x.shape[1] >= 1024
        and groups_per_program >= 4
        and x.numel() < 2**31
        and isinstance(scale_ue8m0, bool)
        and isinstance(column_major_scales, bool)
        and torch.cuda.get_device_capability(x.device) == (9, 0)
    ):
        _per_token_group_quant_fp8_tile[
            lambda meta: (x.shape[0] // meta["BR"], x.shape[1] // (128 * meta["BG"]))
        ](
            x,
            x_q,
            x_s,
            x.shape[0],
            x.shape[1],
            eps,
            fp8_min,
            fp8_max,
            COL=column_major_scales,
            UE=scale_ue8m0,
            DTYPE=str(x.dtype),
        )
        return x_q, x_s

    specialize_metadata = (
        native_vector_input and x.shape[0] >= 128 and x.shape[1] >= 1024
    )
    # Two-group row-major programs regress with constexpr indexing once M
    # exceeds 512, despite lower register usage. Retain their original kernel
    # specialization, and keep the medium-row fast path for larger groups.
    # The remaining column-batched paths keep their existing launch configs.
    specialize_medium = (
        specialize_metadata
        and x.shape[0] <= 2048
        and groups_per_program > 1
        and (column_major_scales or groups_per_program >= 4 or x.shape[0] <= 512)
    )
    quant_scale_mode = (
        tl.constexpr(scale_ue8m0)
        if specialize_medium and isinstance(scale_ue8m0, bool)
        else scale_ue8m0
    )

    if column_major_scales:
        if groups_per_program > 1:
            kernel = _per_token_group_quant_fp8_colmajor_vec
            metadata = (group_size, x.shape[1], x.stride(0), x_s.stride(1))
            # Preserve the previous wide column-major specialization.
            if specialize_metadata:
                metadata = tuple(tl.constexpr(value) for value in metadata)
            kernel[grid](
                x,
                x_q,
                x_s,
                *metadata,
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=quant_scale_mode,
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
        metadata = (group_size, x.shape[1], x.stride(0))
        if specialize_medium:
            metadata = tuple(tl.constexpr(value) for value in metadata)
        kernel[grid](
            x,
            x_q,
            x_s,
            *metadata,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=quant_scale_mode,
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

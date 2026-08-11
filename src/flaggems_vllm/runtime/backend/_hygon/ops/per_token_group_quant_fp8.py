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
from flaggems_vllm.utils.device_info import get_device_capability, get_sm_count

if torch_device_fn.is_available() and get_device_capability() >= (9, 0):
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


@triton.jit
def _f32_to_fp8_e4m3fn(y):
    """Bit-exact f32 -> e4m3fn conversion (RNE); `y` must be finite and
    pre-clamped to [-448, 448].

    gfx936 has no native fp8 conversion instruction, so Triton lowers
    ``.to(float8_e4m3fn)`` to a long emulated sequence (~20 ops/element)
    that dominates this memory-bound kernel.  This branchless sequence needs
    only a few integer ops per element:
      * normals (|y| >= 2^-6): rebias the exponent (127 -> 7) and RNE-round
        the mantissa from 23 to 3 bits via ``t += 0x7FFFF + lsb_of_result``;
      * subnormals: ``k = RNE(|y| / 2**-9)`` with the magic-number add
        ``|y| * 512 + 2**23`` (an f32 add rounds to-nearest-even, leaving k
        in the low mantissa bits).
    """
    b = y.to(tl.uint32, bitcast=True)
    a = b & 0x7FFFFFFF
    t = a - 0x3C000000
    t += 0x0007FFFF + ((t >> 20) & 1)
    r_norm = t >> 20
    r_sub = (a.to(tl.float32, bitcast=True) * 512.0 + 8388608.0).to(
        tl.uint32, bitcast=True
    ) - 0x4B000000
    r = tl.where(a >= 0x3C800000, r_norm, r_sub)
    return (r | ((b >> 24) & 0x80)).to(tl.uint8)


@triton.jit
def _quant_groups(
    y,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    inv_fp8_max: tl.constexpr,
    scale_ue8m0: tl.constexpr,
    subnormal_scale: tl.constexpr,
):
    """Per-row (axis=1) abs-max scale + quantize, in f32."""
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _absmax * inv_fp8_max

    if scale_ue8m0:
        e = tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10)))
        y_s = tl.exp2(e)
        # y_s is a power of two, so multiplying by 2^-e is a bit-exact
        # division and avoids one f32 divide per element.
        y_q = tl.clamp(y * tl.exp2(-e)[:, None], fp8_min, fp8_max)
    elif subnormal_scale:
        # `y_s` is subnormal when `fp8_max` is huge (e.g. fp32), and some
        # backends flush subnormal fp32 divisors to zero; divide by the always
        # normal `_absmax` instead and scale at the end.
        y_q = tl.clamp((y / _absmax[:, None]) * fp8_max, fp8_min, fp8_max)
    else:
        # one reciprocal per group instead of one divide per element
        y_q = tl.clamp(y * (1.0 / y_s)[:, None], fp8_min, fp8_max)
    return y_q, y_s


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
    if BLOCK == GROUP_SIZE:
        tl.store(y_q_ptr + out_offsets, y_q)
    else:
        tl.store(y_q_ptr + out_offsets, y_q, mask=cols[None, :] < GROUP_SIZE)
    tl.store(y_s_ptr + start_gid + gids, y_s)


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
    # Fuse neighbouring groups into one ~1024-element tile per program; on
    # gfx936 a single wavefront per program with 16B vectorized accesses
    # saturates memory bandwidth best.
    cap = max(1, 1024 // group_size)
    ngroups = 1
    while ngroups < cap and groups_per_row % (ngroups * 2) == 0:
        ngroups *= 2
    # keep enough programs in flight when the input is small
    cu_count = get_sm_count()
    while ngroups > 1 and total_groups // ngroups < 4 * cu_count:
        ngroups //= 2
    return ngroups


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
            num_warps=num_warps,
            num_stages=1,
        )

    return x_q, x_s

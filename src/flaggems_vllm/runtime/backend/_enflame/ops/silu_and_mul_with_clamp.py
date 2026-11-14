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
import math
import struct
from functools import lru_cache
from numbers import Real

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.type_utils import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    type_promotion,
)

logger = logging.getLogger(__name__)


_FULL_LINEAR_MIN_ELEMENTS = 1 << 20
_LINEAR_BLOCK_SIZES = (128, 256, 512, 1024, 2048, 4096)

# GCU (Libra/L600, GCU400) tile ladder. The NVIDIA yaml tops out at
# BLOCK_SIZE 4096, but the stream cost is dominated by the per-program fixed
# overhead: measured latency scales ~1/BLOCK_SIZE from 128 through 4096
# (fp16 1G elements: 473ms -> 14.8ms). Extend the ladder to the DSM budget
# (S_token ~= n_live_tiles * BLOCK * elem_bytes <= 448KB keeps the dual
# program per SP concurrency; 2B dtype -> BLOCK 65536 == 384KB, 4B -> 32768).
_GCU_BLOCK_LADDER = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
_GCU_NUM_WARPS = (1, 2, 4)
# Live tile count per program: x/y/out for the forward kernels, plus
# grad/dx/dy for the backward ones.
_DSM_BUDGET_BYTES = 448 * 1024


def _gcu_block_cap(itemsize, n_tiles):
    block = _DSM_BUDGET_BYTES // (n_tiles * itemsize)
    capped = 1
    for candidate in _GCU_BLOCK_LADDER:
        if candidate <= block:
            capped = candidate
    return capped


def _gcu_local_configs(n_tiles, unmasked=False):
    # Configs must be generated per-tile-count because the DSM cap depends on
    # how many live tiles a kernel keeps; the GCU backend has no entry for
    # this op in tune_configs.yaml, so the ladder lives here.
    configs = []
    for block in _GCU_BLOCK_LADDER:
        for num_warps in _GCU_NUM_WARPS:
            if block < 32 * num_warps:
                continue
            meta = {"BLOCK_SIZE": block}
            if unmasked:
                meta["UNMASKED"] = False
            configs.append(triton.Config(meta, num_warps=num_warps, num_stages=1))
    return configs


# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_YZ_MAX = 255


def _flat_block_grid(total_blocks):
    """Spread ``total_blocks`` logical blocks over a 3-D grid within the GCU
    launch limits. The kernels recombine the axes into a flat block id via
    ``_flat_program_id``, so any decomposition with
    ``grid_x * grid_y * grid_z >= total_blocks`` is correct; overshoot blocks
    from the ceil-divisions return early in-kernel before touching memory.
    """
    if total_blocks <= _GRID_X_MAX:
        return (total_blocks, 1, 1)
    grid_y = triton.cdiv(total_blocks, _GRID_X_MAX)
    if grid_y <= _GRID_YZ_MAX:
        return (triton.cdiv(total_blocks, grid_y), grid_y, 1)
    # total_blocks > 65535 * 255: spill into grid.z as well. The flat pid stays
    # within int32 as long as total_blocks < 2**31, i.e. within the same
    # int32-offset domain the elementwise kernels already assume.
    grid_z = triton.cdiv(total_blocks, _GRID_X_MAX * _GRID_YZ_MAX)
    grid_y = triton.cdiv(total_blocks, _GRID_X_MAX * grid_z)
    return (triton.cdiv(total_blocks, grid_y * grid_z), grid_y, grid_z)


def _silu_clamp_tuning_key(value):
    # Keep composite dtype/shape/stride keys SQL-compatible without changing
    # the original constexpr arguments used by the kernels.
    if isinstance(value, (tuple, list)):
        return repr(value)
    return value


def _use_full_linear_kernel(n_elements):
    return n_elements >= _FULL_LINEAR_MIN_ELEMENTS and any(
        n_elements % block_size == 0 for block_size in _LINEAR_BLOCK_SIZES
    )


def _silu_clamp_gcu_prune(n_tiles, require_even=False):
    """Prune configs by the GCU DSM budget.

    ``S_token ~= n_tiles * BLOCK_SIZE * elem_bytes`` must stay within
    448KB so each SP keeps two concurrent programs (48-way); beyond that the
    scheduler degrades to one program per SP (24-way, measured slower).
    ``require_even`` keeps only blocks that divide the element count exactly
    (the unmasked full kernel cannot mask a tail).
    """

    def prune(configs, named_args, **kwargs):
        args = {**named_args, **kwargs}
        n = args["n_elements"]
        dt = args.get("dtype")
        if isinstance(dt, (tuple, list)):
            dt = dt[0] if dt else None
        itemsize = dt.itemsize if hasattr(dt, "itemsize") else 2
        cap = _gcu_block_cap(itemsize, n_tiles)
        kept = []
        for config in configs:
            block = config.kwargs["BLOCK_SIZE"]
            if block > cap or block < 32 * config.num_warps:
                continue
            if require_even or config.kwargs.get("UNMASKED", False):
                if n % block != 0:
                    continue
            kept.append(config)
        return kept

    return prune


@triton.jit
def _strided_offset(linear, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offset = tl.full(linear.shape, 0, tl.int64)
    for dim in tl.static_range(len(SHAPE) - 1, 0, -1):
        index = linear % SHAPE[dim]
        linear = linear // SHAPE[dim]
        offset += index.to(tl.int64) * STRIDES[dim]
    if len(SHAPE) > 0:
        offset += linear.to(tl.int64) * STRIDES[0]
    return offset


@triton.jit
def _flat_program_id():
    # Recombine the 3-D launch grid into a flat block id. grid.x is capped at
    # 65535 (grid.y/grid.z at 255) on the GCU, so large element counts are
    # spread over the remaining axes by _flat_block_grid() on the host side.
    grid_x = tl.num_programs(0)
    grid_xy = grid_x * tl.num_programs(1)
    return tl.program_id(0) + tl.program_id(1) * grid_x + tl.program_id(2) * grid_xy


@triton.jit
def _fmin_nan_propagating(a, b):
    # ``x != x`` is the portable Triton spelling; some vendor Triton
    # backends (including the Mthreads one) do not expose ``tl.isnan``.
    a_nan = a != a
    b_nan = b != b
    # ``torch.clamp`` preserves NaNs in both the input and the bound.  This is
    # deliberately different from IEEE ``fmin``, which selects the non-NaN
    # operand when only one argument is NaN.
    return tl.where(a_nan, a, tl.where(b_nan, b, tl.minimum(a, b)))


@triton.jit
def _fmax_nan_propagating(a, b):
    a_nan = a != a
    b_nan = b != b
    # Match ``torch.clamp(..., min=...)`` for NaN values; do not use the
    # backend's potentially different ``maximum`` NaN behavior directly.
    return tl.where(a_nan, a, tl.where(b_nan, b, tl.maximum(a, b)))


@triton.jit
def _silu_clamp_value(x, y, limit, LIMIT_IS_NAN: tl.constexpr):
    # The reference uses torch.clamp.  Triton's minimum/maximum NaN behavior
    # is backend-dependent, so spell out the clamp semantics explicitly.
    # Keeping the bounds in this order also preserves
    # torch's behavior for limit < 0 (where min > max and the second clamp
    # collapses to the upper bound).
    if LIMIT_IS_NAN:
        gate = _fmin_nan_propagating(x, limit)
        up = _fmin_nan_propagating(_fmax_nan_propagating(y, -limit), limit)
    else:
        # limit is a finite scalar: "value > limit" is false for a NaN value,
        # so the comparison form below preserves torch.clamp's NaN
        # propagation for NaN inputs while replacing the select-heavy NaN
        # helper chain (6 wide selects + 3 min/max + 5 compares) with 3
        # compares + 3 selects.  The nested min-then-max order keeps the
        # limit < 0 collapse to the upper bound of torch.clamp.
        gate = tl.where(x > limit, limit, x)
        up_lo = tl.where(y < -limit, -limit, y)
        up = tl.where(up_lo > limit, limit, up_lo)
    return tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up


@triton.jit
def _silu_clamp_grad_values(
    x,
    y,
    grad,
    limit,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
    LIMIT_IS_NAN: tl.constexpr,
):
    if LIMIT_IS_NAN:
        gate = _fmin_nan_propagating(x, limit)
        up = _fmin_nan_propagating(_fmax_nan_propagating(y, -limit), limit)
    else:
        gate = tl.where(x > limit, limit, x)
        up_lo = tl.where(y < -limit, -limit, y)
        up = tl.where(up_lo > limit, limit, up_lo)
    sig = 1 / (1 + tl.exp(-gate))
    dx = tl.full(x.shape, 0, tl.float32)
    dy = tl.full(x.shape, 0, tl.float32)
    if NEED_DX:
        derivative = sig * (1 + gate * (1 - sig))
        dx = grad * up * derivative * (x <= limit).to(tl.float32)
    if NEED_DY:
        gate_silu = gate * sig
        dy = grad * gate_silu * ((y >= -limit) & (y <= limit)).to(tl.float32)
    return dx, dy


@libentry()
@libtuner(
    configs=_gcu_local_configs(3, unmasked=True),
    prune_configs_by={"early_config_prune": _silu_clamp_gcu_prune(3)},
    key=["n_elements", "dtype", "LIMIT_IS_NAN"],
    strategy=_silu_clamp_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_linear_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    limit,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    LIMIT_IS_NAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    pid = _flat_program_id()
    if pid * BLOCK_SIZE >= n_elements:
        return
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if UNMASKED and n_elements % BLOCK_SIZE == 0:
        mask = tl.full((BLOCK_SIZE,), True, tl.int1)
    else:
        mask = offsets < n_elements
    xo = offsets
    yo = offsets
    oo = offsets
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    result = _silu_clamp_value(x, y, limit, LIMIT_IS_NAN)
    tl.store(out_ptr + oo, result, mask=mask)


@libentry()
@libtuner(
    configs=_gcu_local_configs(3),
    prune_configs_by={
        "early_config_prune": _silu_clamp_gcu_prune(3, require_even=True)
    },
    key=["n_elements", "dtype", "LIMIT_IS_NAN"],
    strategy=_silu_clamp_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_linear_full_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    limit,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    LIMIT_IS_NAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = _flat_program_id()
    # Guard the overshoot blocks produced by the 3-D grid split: this kernel
    # loads/stores without a mask, so any block past the element count must
    # not touch memory.
    if pid * BLOCK_SIZE >= n_elements:
        return
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    y = tl.load(y_ptr + offsets).to(tl.float32)
    result = _silu_clamp_value(x, y, limit, LIMIT_IS_NAN)
    tl.store(out_ptr + offsets, result)


@libentry()
@libtuner(
    configs=_gcu_local_configs(3),
    prune_configs_by={"early_config_prune": _silu_clamp_gcu_prune(3)},
    key=[
        "n_elements",
        "dtype",
        "n_cols",
        "X_STRIDES",
        "Y_STRIDES",
        "OUT_STRIDES",
        "LIMIT_IS_NAN",
    ],
    strategy=_silu_clamp_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_2d_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    limit,
    n_elements,
    dtype: tl.constexpr,
    n_cols: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    LIMIT_IS_NAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = _flat_program_id()
    if pid * BLOCK_SIZE >= n_elements:
        return
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = (offsets // n_cols).to(tl.int64)
    col = offsets % n_cols
    xo = row * X_STRIDES[0] + col * X_STRIDES[1]
    yo = row * Y_STRIDES[0] + col * Y_STRIDES[1]
    oo = row * OUT_STRIDES[0] + col * OUT_STRIDES[1]
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    result = _silu_clamp_value(x, y, limit, LIMIT_IS_NAN)
    tl.store(out_ptr + oo, result, mask=mask)


@libentry()
@libtuner(
    configs=_gcu_local_configs(3),
    prune_configs_by={"early_config_prune": _silu_clamp_gcu_prune(3)},
    key=[
        "n_elements",
        "dtype",
        "SHAPE",
        "X_STRIDES",
        "Y_STRIDES",
        "OUT_STRIDES",
        "LIMIT_IS_NAN",
    ],
    strategy=_silu_clamp_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_nd_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    limit,
    n_elements,
    dtype: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    LIMIT_IS_NAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = _flat_program_id()
    if pid * BLOCK_SIZE >= n_elements:
        return
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    xo = _strided_offset(offsets, SHAPE, X_STRIDES)
    yo = _strided_offset(offsets, SHAPE, Y_STRIDES)
    oo = _strided_offset(offsets, SHAPE, OUT_STRIDES)
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    result = _silu_clamp_value(x, y, limit, LIMIT_IS_NAN)
    tl.store(out_ptr + oo, result, mask=mask)


@libentry()
@libtuner(
    configs=_gcu_local_configs(5, unmasked=True),
    prune_configs_by={"early_config_prune": _silu_clamp_gcu_prune(5)},
    key=["n_elements", "dtype", "NEED_DX", "NEED_DY", "LIMIT_IS_NAN"],
    strategy=_silu_clamp_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_backward_linear_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    limit,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
    LIMIT_IS_NAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    pid = _flat_program_id()
    if pid * BLOCK_SIZE >= n_elements:
        return
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if UNMASKED and n_elements % BLOCK_SIZE == 0:
        mask = tl.full((BLOCK_SIZE,), True, tl.int1)
    else:
        mask = offsets < n_elements
    xo = offsets
    yo = offsets
    go = offsets
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    grad = tl.load(grad_ptr + go, mask=mask, other=0).to(tl.float32)
    dx, dy = _silu_clamp_grad_values(x, y, grad, limit, NEED_DX, NEED_DY, LIMIT_IS_NAN)
    if NEED_DX:
        tl.store(dx_ptr + offsets, dx, mask=mask)
    if NEED_DY:
        tl.store(dy_ptr + offsets, dy, mask=mask)


@libentry()
@libtuner(
    configs=_gcu_local_configs(5),
    prune_configs_by={"early_config_prune": _silu_clamp_gcu_prune(5)},
    key=[
        "n_elements",
        "dtype",
        "n_cols",
        "X_STRIDES",
        "Y_STRIDES",
        "G_STRIDES",
        "NEED_DX",
        "NEED_DY",
        "LIMIT_IS_NAN",
    ],
    strategy=_silu_clamp_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_backward_2d_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    limit,
    n_elements,
    dtype: tl.constexpr,
    n_cols: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    G_STRIDES: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
    LIMIT_IS_NAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = _flat_program_id()
    if pid * BLOCK_SIZE >= n_elements:
        return
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = (offsets // n_cols).to(tl.int64)
    col = offsets % n_cols
    xo = row * X_STRIDES[0] + col * X_STRIDES[1]
    yo = row * Y_STRIDES[0] + col * Y_STRIDES[1]
    go = row * G_STRIDES[0] + col * G_STRIDES[1]
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    grad = tl.load(grad_ptr + go, mask=mask, other=0).to(tl.float32)
    dx, dy = _silu_clamp_grad_values(x, y, grad, limit, NEED_DX, NEED_DY, LIMIT_IS_NAN)
    if NEED_DX:
        tl.store(dx_ptr + offsets, dx, mask=mask)
    if NEED_DY:
        tl.store(dy_ptr + offsets, dy, mask=mask)


@libentry()
@libtuner(
    configs=_gcu_local_configs(5),
    prune_configs_by={"early_config_prune": _silu_clamp_gcu_prune(5)},
    key=[
        "n_elements",
        "dtype",
        "SHAPE",
        "X_STRIDES",
        "Y_STRIDES",
        "G_STRIDES",
        "NEED_DX",
        "NEED_DY",
        "LIMIT_IS_NAN",
    ],
    strategy=_silu_clamp_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_with_clamp_backward_nd_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    limit,
    n_elements,
    dtype: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    G_STRIDES: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
    LIMIT_IS_NAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = _flat_program_id()
    if pid * BLOCK_SIZE >= n_elements:
        return
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    xo = _strided_offset(offsets, SHAPE, X_STRIDES)
    yo = _strided_offset(offsets, SHAPE, Y_STRIDES)
    go = _strided_offset(offsets, SHAPE, G_STRIDES)
    x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + yo, mask=mask, other=0).to(tl.float32)
    grad = tl.load(grad_ptr + go, mask=mask, other=0).to(tl.float32)
    dx, dy = _silu_clamp_grad_values(x, y, grad, limit, NEED_DX, NEED_DY, LIMIT_IS_NAN)
    if NEED_DX:
        tl.store(dx_ptr + offsets, dx, mask=mask)
    if NEED_DY:
        tl.store(dy_ptr + offsets, dy, mask=mask)


def _is_nan_limit(limit):
    # Host-side scalar metadata test (the limit is a host-provided scalar, not
    # device data): finite and infinite limits take the comparison-based fast
    # clamp arm, a NaN limit keeps the exact NaN-propagating reference arm.
    try:
        return math.isnan(float(limit))
    except (TypeError, ValueError):
        return False


def _launch_forward(x, y, out, limit, shape, strides, linear):
    n = out.numel()
    args = (x, y, out, limit, n, (x.dtype, y.dtype, out.dtype))
    if linear:
        kernel = (
            silu_and_mul_with_clamp_linear_full_kernel
            if _use_full_linear_kernel(n)
            else silu_and_mul_with_clamp_linear_kernel
        )
    elif len(shape) == 2:
        kernel = silu_and_mul_with_clamp_2d_kernel
        args += (shape[1], *strides)
    else:
        kernel = silu_and_mul_with_clamp_nd_kernel
        args += (shape, *strides)
    args += (_is_nan_limit(limit),)
    with torch_device_fn.device(x.device):
        kernel[lambda meta: _flat_block_grid(triton.cdiv(n, meta["BLOCK_SIZE"]))](*args)


def _launch_backward(x, y, grad, dx, dy, limit, shape, strides, linear):
    n = grad.numel()
    dtype = dx.dtype if dx is not None else dy.dtype
    args = (x, y, grad, dx, dy, limit, n, (x.dtype, y.dtype, grad.dtype, dtype))
    if linear:
        kernel = silu_and_mul_with_clamp_backward_linear_kernel
    elif len(shape) == 2:
        kernel = silu_and_mul_with_clamp_backward_2d_kernel
        args += (shape[1], *strides)
    else:
        kernel = silu_and_mul_with_clamp_backward_nd_kernel
        args += (shape, *strides)
    args += (dx is not None, dy is not None, _is_nan_limit(limit))
    with torch_device_fn.device(x.device):
        kernel[lambda meta: _flat_block_grid(triton.cdiv(n, meta["BLOCK_SIZE"]))](*args)


@lru_cache(maxsize=128)
def _rounded_limit(limit, dtype):
    # Preserve Python -> input dtype -> FP32 without creating a tensor or
    # synchronizing the accelerator.
    value = float(limit)
    if dtype == torch.float32:
        return struct.unpack("f", struct.pack("f", value))[0]
    if dtype == torch.float16:
        try:
            return struct.unpack("e", struct.pack("e", value))[0]
        except OverflowError:
            return math.copysign(float("inf"), value)
    if dtype == torch.bfloat16:
        bits = struct.unpack("I", struct.pack("f", value))[0]
        exponent = bits & 0x7F800000
        mantissa = bits & 0x007FFFFF
        if exponent == 0x7F800000 and mantissa:
            rounded = (bits >> 16) | 0x0040
        else:
            rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        return struct.unpack("f", struct.pack("I", rounded << 16))[0]
    return value


def _normalize_limit(x, limit):
    if not isinstance(limit, Real):
        raise TypeError("silu_and_mul_with_clamp expects a real scalar limit")
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return _rounded_limit(limit, x.dtype)
    return limit


def _result_dtype(*args):
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    if (
        all(t.dtype == tensors[0].dtype for t in tensors)
        and tensors[0].is_floating_point()
    ):
        return tensors[0].dtype
    return type_promotion(
        *args, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )[1]


def _broadcast_strides(tensor, shape):
    padding = len(shape) - tensor.ndim
    return (0,) * padding + tuple(
        0 if size == 1 else stride
        for size, stride in zip(tensor.shape, tensor.stride())
    )


def _broadcast_shape(lhs, rhs):
    result = []
    for left, right in zip(reversed(lhs), reversed(rhs)):
        if left == 1:
            result.append(right)
        elif right == 1 or left == right:
            result.append(left)
        else:
            raise ValueError(
                f"shapes {tuple(lhs)} and {tuple(rhs)} are not broadcastable"
            )
    longer = lhs if len(lhs) > len(rhs) else rhs
    result.extend(reversed(longer[: abs(len(lhs) - len(rhs))]))
    return tuple(reversed(result))


def _is_non_overlapping_and_dense(tensor):
    expected_stride = 1
    for stride, size in sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 1
    ):
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _storage_span(tensor):
    if tensor.numel() == 0:
        return None
    start = tensor.storage_offset()
    end = start + sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 0
    )
    return start, end


def _overlaps(lhs, rhs):
    if lhs.untyped_storage().data_ptr() != rhs.untyped_storage().data_ptr():
        return False
    lhs_span = _storage_span(lhs)
    rhs_span = _storage_span(rhs)
    if lhs_span is None or rhs_span is None:
        return False
    return lhs_span[0] <= rhs_span[1] and rhs_span[0] <= lhs_span[1]


def _coalesce_layout(shape, strides):
    # Merge only adjacent logical dimensions that are contiguous in *every*
    # operand. Zero strides merge only with zero strides; packed rows keep gaps.
    sizes = []
    merged = [[] for _ in strides]
    for dim, size in enumerate(shape):
        if size == 1:
            continue
        if sizes and all(
            row[-1] == size * stride[dim] for row, stride in zip(merged, strides)
        ):
            sizes[-1] *= size
            for row, stride in zip(merged, strides):
                row[-1] = stride[dim]
        else:
            sizes.append(size)
            for row, stride in zip(merged, strides):
                row.append(stride[dim])
    if not sizes:
        return (1,), tuple((0,) for _ in strides)
    return tuple(sizes), tuple(tuple(row) for row in merged)


def _layout(shape, tensors, allow_dense=False):
    if all(t.shape == shape and t.is_contiguous() for t in tensors):
        # Canonicalize metadata so equal-numel contiguous shapes share tuning.
        return (tensors[0].numel(),), ((1,),) * len(tensors), True
    if (
        allow_dense
        and all(t.shape == shape and t.stride() == tensors[0].stride() for t in tensors)
        and _is_non_overlapping_and_dense(tensors[0])
    ):
        # Matching dense transposes/channels-last tensors have the same physical
        # element order. Flatten their storage and preserve that layout in outputs.
        return (tensors[0].numel(),), ((1,),) * len(tensors), True
    task_shape, strides = _coalesce_layout(
        shape, tuple(_broadcast_strides(t, shape) for t in tensors)
    )
    return task_shape, strides, False


def _forward(x, y, limit, out=None):
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    shape = x.shape if x.shape == y.shape else _broadcast_shape(x.shape, y.shape)
    if out is None:
        dtype = _result_dtype(x, y, limit)
        out = (
            torch.empty_like(x, dtype=dtype)
            if x.shape == shape
            else torch.empty(shape, dtype=dtype, device=x.device)
        )
    elif out.device != x.device or out.shape != shape:
        raise ValueError("out must have the broadcast output shape and input device")
    if out.numel() == 0:
        return out
    # Autotuning launches candidates repeatedly, so an output sharing input
    # storage would mutate later measurements.
    if _overlaps(out, x) or _overlaps(out, y):
        raise NotImplementedError("out must not overlap x or y")
    task_shape, strides, linear = _layout(shape, (x, y, out), allow_dense=True)
    _launch_forward(x, y, out, limit, task_shape, strides, linear)
    return out


class SiluAndMulWithClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, limit):
        ctx.save_for_backward(x, y)
        ctx.limit = _normalize_limit(x, limit)
        logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_FORWARD")
        return _forward(x, y, ctx.limit)

    @staticmethod
    def backward(ctx, dgrad):
        x, y = ctx.saved_tensors
        need_dx, need_dy = ctx.needs_input_grad[:2]
        logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_BACKWARD")
        dtype = _result_dtype(x, y, dgrad, ctx.limit)
        shape, strides, linear = _layout(dgrad.shape, (x, y, dgrad), allow_dense=True)

        def allocate_gradient():
            if linear:
                return torch.empty_like(dgrad, dtype=dtype)
            return torch.empty(dgrad.shape, dtype=dtype, device=x.device)

        dx = allocate_gradient() if need_dx else None
        dy = allocate_gradient() if need_dy else None
        n = dgrad.numel()
        if n:
            _launch_backward(x, y, dgrad, dx, dy, ctx.limit, shape, strides, linear)
        # Autograd reduces broadcast dimensions to the original input shapes.
        return dx, dy, None


def silu_and_mul_with_clamp(x, y, limit):
    if not torch.is_grad_enabled() or not (x.requires_grad or y.requires_grad):
        logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_INFERENCE_FORWARD")
        return _forward(x, y, _normalize_limit(x, limit))
    return SiluAndMulWithClamp.apply(x, y, limit)


def silu_and_mul_with_clamp_out(x, y, out, limit):
    logger.debug("GEMS SILU_AND_MUL_WITH_CLAMP_OUT")
    return _forward(x, y, _normalize_limit(x, limit), out)

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

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_YZ_MAX = 255

# 1D load/store slices beyond these sizes trip the memref_ext.slice_(pad_)start
# attribute-conversion bug in gcu-compiler-opt (operandSegmentSizes 8 vs 7); see
# compiler_issues/vllm_silu_flat_slice_start.md. _silu_prune() keeps every
# autotuning candidate inside both bounds, which caps BLOCK_SIZE at 16384
# elements for 2-byte dtypes and 8192 for 4-byte dtypes.
_SLICE_BYTES_PER_CTA = 32 * 1024
_SLICE_BYTES_PER_WARP = 8 * 1024

_PERSISTENT_CTAS = 48


def _persistent_ctas():
    """CTA count that fills the machine for the grid-stride linear kernels.

    One CTA per SP loops over BLOCK_SIZE tiles (probe-verified faster than
    monolithic one-tile-per-CTA grids at every benchmark shape: 2.2x+ on the
    1G-element shapes). The GCU exposes 48 simultaneously resident programs
    ("Fixed Grid 48"); torch_device_fn.get_device_properties() reports
    multi_processor_count=4 on this backend, so the SP count cannot be queried
    and stays a constant.
    """
    return _PERSISTENT_CTAS


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


def _silu_tuning_key(value):
    # Keep composite dtype/shape/stride keys SQL-compatible without changing
    # the original constexpr arguments used by the kernels.
    if isinstance(value, (tuple, list)):
        return repr(value)
    return value


def _silu_prune(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    n_elements = int(args["n_elements"])
    dtypes = args["dtype"]
    itemsize = max(getattr(dtype, "itemsize", 4) for dtype in dtypes)
    block_cap = max(1024, triton.next_power_of_2(max(n_elements, 1)))
    keep = []
    for config in configs:
        block = config.kwargs["BLOCK_SIZE"]
        if block > block_cap:
            continue
        if block * itemsize > _SLICE_BYTES_PER_CTA:
            continue
        if (block // config.num_warps) * itemsize > _SLICE_BYTES_PER_WARP:
            continue
        keep.append(config)
    return keep or [configs[0]]


def _silu_configs():
    # num_stages=3 matches the backend default the probe measurements used;
    # stage count is irrelevant for these pure streaming kernels.
    configs = []
    for block in (1024, 2048, 4096, 8192, 16384):
        for num_warps in (1, 2, 4, 8):
            configs.append(
                triton.Config({"BLOCK_SIZE": block}, num_warps=num_warps, num_stages=3)
            )
    return configs


_SILU_CONFIGS = _silu_configs()


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
def _silu_mul_value(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


@triton.jit
def _silu_mul_grad_values(x, y, grad, NEED_DX: tl.constexpr, NEED_DY: tl.constexpr):
    sig = 1 / (1 + tl.exp(-x))
    dx = tl.full(x.shape, 0, tl.float32)
    dy = tl.full(x.shape, 0, tl.float32)
    if NEED_DX:
        d_x_silu = sig * (1 + x * (1 - sig))
        dx = d_x_silu * grad * y
    if NEED_DY:
        x_silu = x * sig
        dy = grad * x_silu
    return dx, dy


@libentry()
@libtuner(
    configs=_SILU_CONFIGS,
    prune_configs_by={"early_config_prune": _silu_prune},
    key=["n_elements", "dtype"],
    strategy=_silu_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_linear_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Persistent grid-stride form: grid is clamped to the SP count on the host
    # side and every CTA loops over BLOCK_SIZE tiles, which amortizes the
    # per-CTA prologue that dominates the one-tile-per-CTA shape.
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for start in range(pid * BLOCK_SIZE, n_elements, num_ctas * BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        tl.store(
            out_ptr + offsets,
            _silu_mul_value(x, y).to(out_ptr.type.element_ty),
            mask=mask,
        )


@libentry()
@libtuner(
    configs=_SILU_CONFIGS,
    prune_configs_by={"early_config_prune": _silu_prune},
    key=["n_elements", "dtype", "n_cols", "X_STRIDES", "Y_STRIDES", "OUT_STRIDES"],
    strategy=_silu_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_2d_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    dtype: tl.constexpr,
    n_cols: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
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
    tl.store(out_ptr + oo, _silu_mul_value(x, y).to(out_ptr.type.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=_SILU_CONFIGS,
    prune_configs_by={"early_config_prune": _silu_prune},
    key=["n_elements", "dtype", "SHAPE", "X_STRIDES", "Y_STRIDES", "OUT_STRIDES"],
    strategy=_silu_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_nd_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    dtype: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
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
    tl.store(out_ptr + oo, _silu_mul_value(x, y).to(out_ptr.type.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=_SILU_CONFIGS,
    prune_configs_by={"early_config_prune": _silu_prune},
    key=["n_elements", "dtype", "NEED_DX", "NEED_DY"],
    strategy=_silu_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_backward_linear_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for start in range(pid * BLOCK_SIZE, n_elements, num_ctas * BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0).to(tl.float32)
        dx, dy = _silu_mul_grad_values(x, y, grad, NEED_DX, NEED_DY)
        if NEED_DX:
            tl.store(dx_ptr + offsets, dx.to(dx_ptr.type.element_ty), mask=mask)
        if NEED_DY:
            tl.store(dy_ptr + offsets, dy.to(dy_ptr.type.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=_SILU_CONFIGS,
    prune_configs_by={"early_config_prune": _silu_prune},
    key=[
        "n_elements",
        "dtype",
        "n_cols",
        "X_STRIDES",
        "Y_STRIDES",
        "G_STRIDES",
        "NEED_DX",
        "NEED_DY",
    ],
    strategy=_silu_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_backward_2d_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    n_elements,
    dtype: tl.constexpr,
    n_cols: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    G_STRIDES: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
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
    dx, dy = _silu_mul_grad_values(x, y, grad, NEED_DX, NEED_DY)
    if NEED_DX:
        tl.store(dx_ptr + offsets, dx.to(dx_ptr.type.element_ty), mask=mask)
    if NEED_DY:
        tl.store(dy_ptr + offsets, dy.to(dy_ptr.type.element_ty), mask=mask)


def _launch_forward(x, y, out, shape, strides, linear):
    n = out.numel()
    args = (x, y, out, n, (x.dtype, y.dtype, out.dtype))
    if linear:
        kernel = silu_and_mul_linear_kernel
    elif len(shape) == 2:
        kernel = silu_and_mul_2d_kernel
        args += (shape[1], *strides)
    else:
        kernel = silu_and_mul_nd_kernel
        args += (shape, *strides)
    with torch_device_fn.device(x.device):
        if linear:
            kernel[
                lambda meta: (
                    min(_persistent_ctas(), triton.cdiv(n, meta["BLOCK_SIZE"])),
                )
            ](*args)
        else:
            kernel[lambda meta: _flat_block_grid(triton.cdiv(n, meta["BLOCK_SIZE"]))](
                *args
            )


def _launch_backward(x, y, grad, dx, dy, shape, strides, linear):
    n = grad.numel()
    dtype = dx.dtype if dx is not None else dy.dtype
    # A missing gradient arm never dereferences its pointer (the stores are
    # gated by NEED_DX/NEED_DY constexprs), but Triton still needs a valid
    # tensor argument; pass the gradient as a stand-in.
    dx_arg = dx if dx is not None else grad
    dy_arg = dy if dy is not None else grad
    args = (x, y, grad, dx_arg, dy_arg, n, (x.dtype, y.dtype, grad.dtype, dtype))
    if linear:
        kernel = silu_and_mul_backward_linear_kernel
    elif len(shape) == 2:
        kernel = silu_and_mul_backward_2d_kernel
        args += (shape[1], *strides)
    else:
        kernel = silu_and_mul_backward_nd_kernel
        args += (shape, *strides)
    args += (dx is not None, dy is not None)
    with torch_device_fn.device(x.device):
        if linear:
            kernel[
                lambda meta: (
                    min(_persistent_ctas(), triton.cdiv(n, meta["BLOCK_SIZE"])),
                )
            ](*args)
        else:
            kernel[lambda meta: _flat_block_grid(triton.cdiv(n, meta["BLOCK_SIZE"]))](
                *args
            )


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


def _forward(x, y, out=None):
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    shape = x.shape if x.shape == y.shape else _broadcast_shape(x.shape, y.shape)
    if out is None:
        dtype = _result_dtype(x, y)
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
    _launch_forward(x, y, out, task_shape, strides, linear)
    return out


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        logger.debug("GEMS SILU AND MUL FORWARD")
        return _forward(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        need_dx, need_dy = ctx.needs_input_grad[:2]
        logger.debug("GEMS SILU AND MUL BACKWARD")
        dtype = _result_dtype(x, y, grad_output)
        shape, strides, linear = _layout(
            grad_output.shape, (x, y, grad_output), allow_dense=True
        )

        def allocate_gradient():
            if linear:
                return torch.empty_like(grad_output, dtype=dtype)
            return torch.empty(grad_output.shape, dtype=dtype, device=x.device)

        dx = allocate_gradient() if need_dx else None
        dy = allocate_gradient() if need_dy else None
        n = grad_output.numel()
        if n:
            _launch_backward(x, y, grad_output, dx, dy, shape, strides, linear)
        # Autograd reduces broadcast dimensions to the original input shapes.
        return dx, dy


def silu_and_mul(A, B):
    if not torch.is_grad_enabled() or not (A.requires_grad or B.requires_grad):
        logger.debug("GEMS SILU_AND_MUL_INFERENCE_FORWARD")
        return _forward(A, B)
    return SiluAndMul.apply(A, B)


def silu_and_mul_out(A, B, out):
    logger.debug("GEMS SILU_AND_MUL_OUT")
    return _forward(A, B, out=out)


@libentry()
@libtuner(
    configs=_SILU_CONFIGS,
    prune_configs_by={"early_config_prune": _silu_prune},
    key=[
        "n_elements",
        "dtype",
        "SHAPE",
        "X_STRIDES",
        "Y_STRIDES",
        "G_STRIDES",
        "NEED_DX",
        "NEED_DY",
    ],
    strategy=_silu_tuning_key,
    policy="default",
    warmup=25,
    rep=100,
)
@triton.jit
def silu_and_mul_backward_nd_kernel(
    x_ptr,
    y_ptr,
    grad_ptr,
    dx_ptr,
    dy_ptr,
    n_elements,
    dtype: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    G_STRIDES: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DY: tl.constexpr,
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
    dx, dy = _silu_mul_grad_values(x, y, grad, NEED_DX, NEED_DY)
    if NEED_DX:
        tl.store(dx_ptr + offsets, dx.to(dx_ptr.type.element_ty), mask=mask)
    if NEED_DY:
        tl.store(dy_ptr + offsets, dy.to(dy_ptr.type.element_ty), mask=mask)

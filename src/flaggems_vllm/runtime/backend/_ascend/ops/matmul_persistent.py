# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""Batch-invariant FP16/BF16 matmul for the Ascend backend.

Testing to date has been limited to Ascend 910 NPUs (Ascend 910B4-1).
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils.shape_utils import broadcast

logger = logging.getLogger(__name__)

_ASCEND_AFFINE_BATCH_RANK = 4
_ASCEND_AFFINE_BLOCK_M = 64
_ASCEND_AFFINE_BLOCK_N = 32
_ASCEND_AFFINE_BLOCK_K = 128
_ASCEND_PACK_ROWS = 8
_ASCEND_PACK_COLUMNS = 32


@triton.jit
def _ascend_matmul_accumulate(
    a_ptr,
    b_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    start_m,
    start_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    # Bound the source rectangle before copying it. With a mask on an array
    # of pointers, Ascend's lowering can copy a full K tile and crop afterward.
    a_order: tl.constexpr = (1, 0) if A_ROW_MAJOR else (0, 1)
    b_order: tl.constexpr = (1, 0) if B_ROW_MAJOR else (0, 1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        start_k = ki * BK
        a_block = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(start_m, start_k),
            block_shape=(BM, BK),
            order=a_order,
        )
        b_block = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(start_k, start_n),
            block_shape=(BK, BN),
            order=b_order,
        )
        a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc)
    return acc


@triton.jit
def _ascend_affine_bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    IS_ASCEND: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    A3: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    start_m = (tile // tiles_n) * BLOCK_SIZE_M
    start_n = (tile % tiles_n) * BLOCK_SIZE_N
    q3 = batch % D3
    q2 = (batch // D3) % D2
    q1 = (batch // (D3 * D2)) % D1
    q0 = batch // (D3 * D2 * D1)
    a_offset = q0 * A0 + q1 * A1 + q2 * A2 + q3 * A3
    b_offset = q0 * B0 + q1 * B1 + q2 * B2 + q3 * B3
    acc = _ascend_matmul_accumulate(
        a_ptr + a_offset,
        b_ptr + b_offset,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        start_m,
        start_n,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        A_ROW_MAJOR,
        B_ROW_MAJOR,
    )
    output = tl.make_block_ptr(
        base=c_ptr + batch * stride_cb,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(start_m, start_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(output, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _ascend_pack_matrix_rows(
    source,
    destination,
    M: tl.constexpr,
    N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STEP: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    col_tiles = tl.cdiv(N, COLS)
    tiles = tl.cdiv(M, ROWS) * col_tiles
    batch = program // tiles
    tile = program % tiles
    row = (tile // col_tiles) * ROWS
    column = (tile % col_tiles) * COLS
    q3 = batch % D3
    q2 = (batch // D3) % D2
    q1 = (batch // (D3 * D2)) % D1
    q0 = batch // (D3 * D2 * D1)
    base = q0 * S0 + q1 * S1 + q2 * S2 + q3 * S3
    src = tl.make_block_ptr(
        base=source + base,
        shape=(M, (N - 1) * STEP + 1),
        strides=(STRIDE_M, 1),
        offsets=(row.to(tl.int32), (column * STEP).to(tl.int32)),
        block_shape=(ROWS, COLS * STEP),
        order=(1, 0),
    )
    values = tl.load(src, boundary_check=(0, 1), padding_option="zero")
    if STEP == 2:
        values, _ = tl.split(tl.reshape(values, (ROWS, COLS, 2)))
    dst = tl.make_block_ptr(
        base=destination + batch * M * N,
        shape=(M, N),
        strides=(N, 1),
        offsets=(row.to(tl.int32), column.to(tl.int32)),
        block_shape=(ROWS, COLS),
        order=(1, 0),
    )
    tl.store(dst, values, boundary_check=(0, 1))


def _ascend_matmul_config(a, b, k, config):
    return dict(
        config,
        # The short K=16 and K=32 smoke cases fit in a single full tile.
        # The block bounds also cover an arbitrary final K tail.
        BLOCK_SIZE_K=min(config["BLOCK_SIZE_K"], triton.next_power_of_2(max(16, k))),
        # This FlagTree frontend expects scalar constexpr arguments across JIT calls.
        IS_ASCEND=True,
        A_ROW_MAJOR=a.stride(-1) <= a.stride(-2),
        B_ROW_MAJOR=b.stride(-1) <= b.stride(-2),
    )


def _ascend_dense(shape, strides):
    expected = 1
    for size, step in zip(reversed(shape), reversed(strides)):
        if size != 1 and step != expected:
            return False
        expected *= size
    return True


def _ascend_canonical_view(tensor):
    if not _ascend_dense(tensor.shape, tensor.stride()):
        return tensor
    strides = list(tensor.stride())
    expected = 1
    for axis in range(tensor.ndim - 1, -1, -1):
        if tensor.shape[axis] == 1:
            strides[axis] = expected
        expected *= tensor.shape[axis]
    if tuple(strides) == tensor.stride():
        return tensor
    return tensor.as_strided(
        tensor.shape, tuple(strides), storage_offset=tensor.storage_offset()
    )


def _ascend_requires_pack(tensor, role):
    if _ascend_dense(tensor.shape, tensor.stride()):
        return False
    if role == "b" and tensor.storage_offset() == 0:
        shape = (*tensor.shape[:-2], tensor.shape[-1], tensor.shape[-2])
        strides = (*tensor.stride()[:-2], tensor.stride(-1), tensor.stride(-2))
        if _ascend_dense(shape, strides):
            return False
    return True


def _ascend_pack_input(tensor):
    m, n = tensor.shape[-2:]
    columns = triton.next_power_of_2(min(n, _ASCEND_PACK_COLUMNS))
    batches = math.prod(tensor.shape[:-2])
    programs = batches * triton.cdiv(m, _ASCEND_PACK_ROWS) * triton.cdiv(n, columns)
    output = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    dimensions = (1,) * (_ASCEND_AFFINE_BATCH_RANK + 2 - tensor.ndim) + tuple(
        tensor.shape[:-2]
    )
    strides = (0,) * (_ASCEND_AFFINE_BATCH_RANK + 2 - tensor.ndim) + tuple(
        tensor.stride()[:-2]
    )
    _ascend_pack_matrix_rows[(programs,)](
        tensor,
        output,
        m,
        n,
        tensor.stride(-2),
        tensor.stride(-1),
        *dimensions,
        *strides,
        ROWS=_ASCEND_PACK_ROWS,
        COLS=columns,
    )
    return output


def _ascend_batch_strides(tensor, batch_shape):
    rank = len(batch_shape)
    pad = rank - tensor.ndim + 2
    shape = (1,) * pad + tuple(tensor.shape[:-2])
    strides = (0,) * pad + tuple(tensor.stride()[:-2])
    return (0,) * (_ASCEND_AFFINE_BATCH_RANK - rank) + tuple(
        0 if size == 1 else step for size, step in zip(shape, strides)
    )


def _ascend_affine_matmul(a, b, plan, out=None):
    a, b = _ascend_canonical_view(a), _ascend_canonical_view(b)
    if _ascend_requires_pack(a, "a"):
        a = _ascend_pack_input(a)
    if _ascend_requires_pack(b, "b"):
        b = _ascend_pack_input(b)
    if out is None:
        out = torch.empty(plan["output_shape"], dtype=a.dtype, device=a.device)
    batch_shape = plan["batch_shape"]
    dimensions = (1,) * (_ASCEND_AFFINE_BATCH_RANK - len(batch_shape)) + batch_shape
    a_strides = _ascend_batch_strides(a, batch_shape)
    b_strides = _ascend_batch_strides(b, batch_shape)
    meta = {
        prefix + str(index): value
        for prefix, values in (("D", dimensions), ("A", a_strides), ("B", b_strides))
        for index, value in enumerate(values)
    }
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    config = _ascend_matmul_config(
        a,
        b,
        k,
        dict(
            BLOCK_SIZE_M=_ASCEND_AFFINE_BLOCK_M,
            BLOCK_SIZE_N=_ASCEND_AFFINE_BLOCK_N,
            BLOCK_SIZE_K=_ASCEND_AFFINE_BLOCK_K,
            num_stages=3,
            num_warps=8,
        ),
    )
    grid = (
        math.prod(batch_shape),
        triton.cdiv(m, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"]),
    )
    _ascend_affine_bmm_kernel[grid](
        a,
        b,
        out,
        math.prod(batch_shape),
        m,
        n,
        k,
        a.stride(-3) if a.ndim > 2 else 0,
        a.stride(-2),
        a.stride(-1),
        b.stride(-3) if b.ndim > 2 else 0,
        b.stride(-2),
        b.stride(-1),
        m * n,
        n,
        1,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=out.numel() > 2**31,
        **config,
        **meta,
    )
    return out


def _validate(a, b, out):
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("Both inputs must have rank at least two")
    if max(a.ndim, b.ndim) > _ASCEND_AFFINE_BATCH_RANK + 2:
        raise NotImplementedError("Ascend supports input ranks 2 through 6")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("Inputs must be on the same device and have identical dtypes")
    if a.device.type != "npu":
        raise NotImplementedError("This port supports Ascend NPU only")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("Ascend supports FP16 and BF16")
    if a.layout != torch.strided or b.layout != torch.strided:
        raise NotImplementedError("Only strided tensors are supported")
    if a.requires_grad or b.requires_grad:
        raise NotImplementedError("This operator supports forward inference only")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("Incompatible inner dimensions")

    try:
        batch_shape = broadcast(tuple(a.shape[:-2]), tuple(b.shape[:-2]))
    except ValueError as exc:
        raise ValueError("Incompatible batch dimensions") from exc
    shape = (*batch_shape, a.shape[-2], b.shape[-1])
    if out is not None:
        if tuple(out.shape) != shape or out.dtype != a.dtype or out.device != a.device:
            raise ValueError("out must match output shape, dtype and device")
        if out.layout != torch.strided or out.requires_grad or not out.is_contiguous():
            raise ValueError("out must be a contiguous strided inference tensor")
        if out.numel() and any(
            out.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            for value in (a, b)
        ):
            raise ValueError("out must not share input storage")
    return batch_shape, shape


def matmul_persistent(a, b, *, out=None):
    """Compute vLLM batch-invariant matmul using the Ascend specialization."""
    logger.debug("GEMS_ASCEND MATMUL_PERSISTENT")
    batch_shape, output_shape = _validate(a, b, out)
    if math.prod(output_shape) == 0:
        if out is not None:
            return out
        return torch.empty(output_shape, dtype=a.dtype, device=a.device)
    if a.shape[-1] == 0:
        result = torch.zeros(output_shape, dtype=a.dtype, device=a.device)
        if out is not None:
            out.copy_(result)
            return out
        return result

    for role, value in (("a", a), ("b", b)):
        if min(value.stride()[-2:]) <= 0 or max(value.shape[-2:]) >= 2**31:
            raise NotImplementedError("Unsupported Ascend matrix shape or strides")
        if _ascend_requires_pack(value, role):
            m, n = value.shape[-2:]
            step = value.stride(-1)
            if step not in (1, 2) or max(m - 1, (n - 1) * step) >= 2**31:
                raise NotImplementedError("Unsupported Ascend matrix packing strides")

    plan = dict(batch_shape=batch_shape, output_shape=output_shape)
    with torch.npu.device(a.device):
        return _ascend_affine_matmul(a, b, plan, out=out)

# SPDX-License-Identifier: Apache-2.0
"""Data movement used by model fused operators; no Torch copy/cast launches."""
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(configs=runtime.get_tuned_config("glm_data_copy"), key=["N"])
@triton.jit
def _copy_kernel(
    X,
    Y,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = tl.full((BLOCK,), 0, tl.int64)
    remaining = i
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (remaining % SHAPE[d]) * STRIDES[d]
        remaining //= SHAPE[d]
    value = tl.load(X + offset, i < N, other=0)
    tl.store(Y + i, value, i < N)


def contiguous_copy(x, dtype=None):
    dtype = dtype or x.dtype
    if x.is_contiguous() and x.dtype == dtype:
        return x
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    if x.numel():
        _copy_kernel[lambda m: (triton.cdiv(x.numel(), m["BLOCK"]),)](
            x, out, x.numel(), tuple(x.shape), tuple(x.stride())
        )
    return out


@libentry()
@libtuner(configs=runtime.get_tuned_config("glm_data_copy"), key=["N"])
@triton.jit
def _fill_kernel(Y, N: tl.constexpr, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, VALUE, i < N)


def fill(out, value):
    if not out.is_contiguous():
        raise ValueError("fill requires contiguous output")
    if out.numel():
        _fill_kernel[lambda m: (triton.cdiv(out.numel(), m["BLOCK"]),)](
            out, out.numel(), value
        )
    return out


@libentry()
@libtuner(configs=runtime.get_tuned_config("glm_data_copy"), key=["N", "WIDTH"])
@triton.jit
def _gather_kernel(
    X,
    IDX,
    Y,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    STRIDE_ROW: tl.constexpr,
    STRIDE_COL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    src_row = tl.load(IDX + i // WIDTH, i < N, other=0)
    value = tl.load(X + src_row * STRIDE_ROW + (i % WIDTH) * STRIDE_COL, i < N, other=0)
    tl.store(Y + i, value, i < N)


def gather_rows(x, indices):
    if x.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("gather_rows requires matrix and integer indices")
    if indices.device != x.device:
        raise ValueError("gather indices must share input device")
    indices = contiguous_copy(indices)
    out = torch.empty((*indices.shape, x.shape[1]), dtype=x.dtype, device=x.device)
    if out.numel():
        _gather_kernel[lambda m: (triton.cdiv(out.numel(), m["BLOCK"]),)](
            x, indices, out, out.numel(), x.shape[1], x.stride(0), x.stride(1)
        )
    return out


@libentry()
@triton.jit
def _scatter_kernel(
    X,
    REQ,
    POS,
    Y,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    LMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // WIDTH
    req = tl.load(REQ + row, i < N, other=0)
    pos = tl.load(POS + row, i < N, other=0)
    value = tl.load(X + i, i < N, other=0)
    tl.store(Y + (req * LMAX + pos) * WIDTH + i % WIDTH, value, i < N)


def scatter_decode_tokens(tokens, pad_value, num_requests, lmax, scatter_indices):
    req, pos = scatter_indices
    if req.dtype not in (torch.int32, torch.int64) or pos.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("scatter indices must have integer dtype")
    if num_requests < 0 or lmax < 0:
        raise ValueError("scatter output dimensions must be nonnegative")
    if req.shape != (tokens.shape[0],) or pos.shape != req.shape:
        raise ValueError("scatter indices must have one row per token")
    if req.device != tokens.device or pos.device != tokens.device:
        raise ValueError("scatter indices must share token device")
    tokens, req, pos = map(contiguous_copy, (tokens, req, pos))
    out = torch.empty(
        (num_requests, lmax, *tokens.shape[1:]),
        dtype=tokens.dtype,
        device=tokens.device,
    )
    fill(out, pad_value)
    width = math.prod(tokens.shape[1:])
    if tokens.numel():
        _scatter_kernel[(triton.cdiv(tokens.numel(), 256),)](
            tokens, req, pos, out, tokens.numel(), width, lmax, BLOCK=256
        )
    return out


@libentry()
@triton.jit
def _state_rows_kernel(
    STATE,
    VALUES,
    IDX,
    OUT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    SLOTS: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    slot = tl.load(IDX + row, i < N, other=-1)
    valid = (i < N) & (slot >= 0) & (slot < SLOTS)
    if MODE == "gather":
        value = tl.load(STATE + slot * STATE_STRIDE + col, valid, other=0)
        tl.store(OUT + i, value, i < N)
    elif MODE == "scatter":
        value = tl.load(VALUES + i, i < N, other=0)
        tl.store(STATE + slot * STATE_STRIDE + col, value, valid)
    else:
        tl.store(STATE + slot * STATE_STRIDE + col, 0, valid)


@libentry()
@libtuner(configs=runtime.get_tuned_config("glm_data_copy"), key=["N", "WIDTH"])
@triton.jit
def _gather_state_rows_kernel(
    STATE,
    IDX,
    OUT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    SLOTS: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    slot = tl.load(IDX + row, i < N, other=-1)
    valid = (i < N) & (slot >= 0) & (slot < SLOTS)
    value = tl.load(STATE + slot * STATE_STRIDE + col, valid, other=0)
    tl.store(OUT + i, value, i < N)


def _state_rows(state, indices, values, mode):
    if state.ndim < 2:
        raise ValueError("State rows require rank >=2")
    width = math.prod(state.shape[1:])
    dense_stride = 1
    for size, stride in zip(reversed(state.shape[1:]), reversed(state.stride()[1:])):
        if size > 1 and stride != dense_stride:
            raise ValueError("State rows require a dense feature layout")
        dense_stride *= size
    if state.stride(0) < width:
        raise ValueError("State slots must not overlap")
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("State rows require an integer slot vector")
    if indices.device != state.device:
        raise ValueError("State indices must share the cache device")
    if values is not None and (
        values.shape != (indices.numel(), *state.shape[1:])
        or values.dtype != state.dtype
        or values.device != state.device
    ):
        raise ValueError(
            "State row values must match the cache shape, dtype and device"
        )
    indices = contiguous_copy(indices)
    values = contiguous_copy(values) if values is not None else None
    count = indices.numel() * width
    out = (
        torch.empty(
            (indices.numel(), *state.shape[1:]), dtype=state.dtype, device=state.device
        )
        if mode == "gather"
        else None
    )
    if count and mode == "gather":
        _gather_state_rows_kernel[lambda m: (triton.cdiv(count, m["BLOCK"]),)](
            state, indices, out, count, width, state.shape[0], state.stride(0)
        )
    elif count:
        # Cache mutation has one fixed launch; repeated autotune writes are unsafe.
        _state_rows_kernel[(triton.cdiv(count, 256),)](
            state,
            values,
            indices,
            out,
            count,
            width,
            state.shape[0],
            state.stride(0),
            mode,
            BLOCK=256,
        )
    return out


def gather_state_rows(state, indices):
    """Gather request state; sentinel slots yield zeros without reading a cache row."""
    return _state_rows(state, indices, None, "gather")


def scatter_state_rows(state, indices, values):
    """Write each valid request slot once; slot uniqueness is owned by metadata."""
    _state_rows(state, indices, values, "scatter")


def zero_state_rows(state, indices):
    """Initialize valid request slots without changing padded request state."""
    _state_rows(state, indices, None, "zero")


def copy_to(x, out):
    if (
        x.shape != out.shape
        or x.dtype != out.dtype
        or x.device != out.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "copy_to requires matching shape/dtype/device and contiguous output"
        )
    if x.numel():
        # Pure overwrite: autotuning may repeat identical writes to this output.
        _copy_kernel[lambda m: (triton.cdiv(x.numel(), m["BLOCK"]),)](
            x, out, x.numel(), tuple(x.shape), tuple(x.stride())
        )
    return out


@libentry()
@libtuner(configs=runtime.get_tuned_config("glm_data_copy"), key=["N", "DA", "DB"])
@triton.jit
def _concat_query_kernel(
    A, B, Y, N: tl.constexpr, DA: tl.constexpr, DB: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // (DA + DB), i % (DA + DB)
    left = tl.load(A + row * DA + col, (i < N) & (col < DA), other=0)
    right = tl.load(B + row * DB + col - DA, (i < N) & (col >= DA), other=0)
    tl.store(Y + i, tl.where(col < DA, left, right), i < N)


def concat_mla_q(ql_nope, q_pe, q_out):
    if ql_nope.shape[:-1] != q_pe.shape[:-1] or q_out.shape != (
        *ql_nope.shape[:-1],
        ql_nope.shape[-1] + q_pe.shape[-1],
    ):
        raise ValueError("MLA query concatenation shape mismatch")
    if (
        ql_nope.dtype != q_pe.dtype
        or q_out.dtype != ql_nope.dtype
        or ql_nope.device != q_pe.device
        or q_out.device != ql_nope.device
    ):
        raise ValueError("MLA query tensors must share dtype/device")
    if not q_out.is_contiguous():
        raise ValueError("MLA query output must be contiguous")
    a, b = map(contiguous_copy, (ql_nope, q_pe))
    if q_out.numel():
        _concat_query_kernel[lambda m: (triton.cdiv(q_out.numel(), m["BLOCK"]),)](
            a, b, q_out, q_out.numel(), a.shape[-1], b.shape[-1]
        )
    return None


def concat_query(ql_nope, q_pe):
    out = torch.empty(
        (*ql_nope.shape[:-1], ql_nope.shape[-1] + q_pe.shape[-1]),
        dtype=ql_nope.dtype,
        device=ql_nope.device,
    )
    concat_mla_q(ql_nope, q_pe, out)
    return out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("glm_data_copy"), key=["N", "H", "PAD_H", "D"]
)
@triton.jit
def _pad_heads_kernel(
    X,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    PAD_H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = (i // D) % PAD_H
    src = (i // (PAD_H * D) * H + head) * D + i % D
    value = tl.load(X + src, (i < N) & (head < H), other=0)
    tl.store(Y + i, value, i < N)


def pad_attention_heads(q, heads):
    if q.ndim != 3 or q.shape[1] > heads:
        raise ValueError("Attention head padding expects [T,H,D] and heads>=H")
    if q.shape[1] == heads:
        return contiguous_copy(q)
    q = contiguous_copy(q)
    out = torch.empty((q.shape[0], heads, q.shape[2]), dtype=q.dtype, device=q.device)
    if out.numel():
        _pad_heads_kernel[lambda m: (triton.cdiv(out.numel(), m["BLOCK"]),)](
            q, out, out.numel(), q.shape[1], heads, q.shape[2]
        )
    return out

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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def _ln_fwd_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    FULL_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    if FULL_COLS:
        w = tl.load(W + cols).to(tl.float32)
    else:
        col_mask = cols < N
        w = tl.load(W + cols, mask=col_mask).to(tl.float32)

    x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :] + group * N
    if HAS_Z:
        z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :] + group * N

    if FULL_COLS:
        if EVEN_ROWS:
            x = tl.load(x_ptrs).to(tl.float32)
            if HAS_Z:
                z = tl.load(z_ptrs).to(tl.float32)
        else:
            row_mask = rows < M
            x = tl.load(x_ptrs, mask=row_mask[:, None]).to(tl.float32)
            if HAS_Z:
                z = tl.load(z_ptrs, mask=row_mask[:, None]).to(tl.float32)
    else:
        if EVEN_ROWS:
            col_mask = cols < N
            x = tl.load(x_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
            if HAS_Z:
                z = tl.load(z_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
        else:
            row_mask = rows < M
            col_mask = cols < N
            msk = row_mask[:, None] & col_mask[None, :]
            x = tl.load(x_ptrs, mask=msk, other=0.0).to(tl.float32)
            if HAS_Z:
                z = tl.load(z_ptrs, mask=msk, other=0.0).to(tl.float32)

    if HAS_BIAS:
        if FULL_COLS:
            b = tl.load(B + cols).to(tl.float32)
        else:
            b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    if HAS_Z:
        if not NORM_BEFORE_GATE:
            x *= z * tl.sigmoid(z)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N
        if FULL_COLS:
            xbar = x - mean[:, None]
        else:
            xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        if EVEN_ROWS:
            tl.store(Mean + rows, mean)
        else:
            tl.store(Mean + rows, mean, mask=row_mask)
    else:
        if FULL_COLS:
            xbar = x
        else:
            xbar = tl.where(col_mask[None, :], x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N

    rstd = 1.0 / tl.sqrt(var + eps)
    if EVEN_ROWS:
        tl.store(Rstd + rows, rstd)
    else:
        tl.store(Rstd + rows, rstd, mask=row_mask)

    y = xbar * rstd[:, None]
    y = y * w[None, :]
    if HAS_BIAS:
        y += b[None, :]
    if HAS_Z and NORM_BEFORE_GATE:
        y *= z * tl.sigmoid(z)

    y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :] + group * N
    if EVEN_ROWS and FULL_COLS:
        tl.store(y_ptrs, y)
    elif EVEN_ROWS:
        tl.store(y_ptrs, y, mask=col_mask[None, :])
    elif FULL_COLS:
        tl.store(y_ptrs, y, mask=row_mask[:, None])
    else:
        tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


# ---------------------------------------------------------------------------
# Exact-tile fast path: identical per-row arithmetic, but every runtime address
# scalar is gone from both the launch packet and the address math. The row
# length and row stride are compile-time constants equal to BLOCK_N, there is no
# group offset (only launched when ngroups == 1), the grid is 1-D and the tile
# starts at a flat contiguous block offset, so the tile pointer lowers to one
# contiguous descriptor instead of a dynamic-stride memref. Only launched when
# ngroups == 1, group_size == N == BLOCK_N, all row strides equal group_size and
# M % BLOCK_M == 0. Division by BLOCK_N is exact (BLOCK_N is a power of two), so
# the numerics stay bit-identical.
# ---------------------------------------------------------------------------


@triton.jit
def _ln_fwd_tile_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    pid = tl.program_id(0)

    cols = tl.arange(0, BLOCK_N)

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    off = rows[:, None] * BLOCK_N + cols[None, :]

    # ORDER CELL (x, w, z): the small [BLOCK_N] weight fetch sits BETWEEN the
    # two large tile DMAs instead of after both. Values, operations and their
    # order are unchanged; only the issue position of the weight load moves, and
    # it is consumed far downstream in the epilogue. Bit-exact by construction.
    x = tl.load(X + off).to(tl.float32)

    w = tl.load(W + cols).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols).to(tl.float32)

    if HAS_Z:
        z = tl.load(Z + off).to(tl.float32)
        if not NORM_BEFORE_GATE:
            x *= z * tl.sigmoid(z)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / BLOCK_N
        xbar = x - mean[:, None]
        var = tl.sum(xbar * xbar, axis=1) / BLOCK_N
        tl.store(Mean + rows, mean)
    else:
        xbar = x
        var = tl.sum(xbar * xbar, axis=1) / BLOCK_N

    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd)

    y = xbar * rstd[:, None]
    y = y * w[None, :]
    if HAS_BIAS:
        y += b[None, :]
    if HAS_Z and NORM_BEFORE_GATE:
        # Folded gate: one divide of the product instead of a separate
        # [BLOCK_M,BLOCK_N] reciprocal followed by two multiplies.
        y = (y * z) / (1.0 + tl.exp(-z))

    tl.store(Y + off, y)


def _launch(
    x, weight, bias, eps, z, group_size, norm_before_gate, is_rms_norm, out=None
):
    M, N = x.shape
    gs = N if group_size is None else group_size
    if not gs:
        gs = N
    ngroups = N // gs

    if out is None:
        out = torch.empty_like(x)
    if not is_rms_norm:
        mean = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    else:
        mean = None
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    if M == 0:
        return out, mean, rstd

    BLOCK_N = min(65536 // x.element_size(), triton.next_power_of_2(gs))
    BLOCK_M = min(128, max(16, triton.next_power_of_2(M)))

    s_z = z.stride(0) if z is not None else gs
    exact = (
        ngroups == 1
        and BLOCK_N == gs
        and gs == N
        and x.stride(0) == gs
        and out.stride(0) == gs
        and s_z == gs
        and M % BLOCK_M == 0
    )

    if exact:
        _ln_fwd_tile_kernel[(M // BLOCK_M,)](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            eps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HAS_BIAS=bias is not None,
            HAS_Z=z is not None,
            NORM_BEFORE_GATE=bool(norm_before_gate),
            IS_RMS_NORM=bool(is_rms_norm),
        )
        return out, mean, rstd

    # --- UB-fit guard, generic fallback only -------------------------------
    # The tuned exact-tile fast path above is left byte-identical (BLOCK_M is
    # its own value). The masked 2-D kernel is the path that serves every other
    # geometry, and it materialises several full [BLOCK_M, BLOCK_N] tiles at
    # once: x and z and the promoted output in fp32, i.e. 4 bytes per element
    # regardless of the input dtype, plus a byte-per-element predicate tile in
    # the masked branch, plus the weight/bias vectors. At BLOCK_M=128 with
    # BLOCK_N=128 that is ~264 KB against the 253,952-byte per-core UB, so an
    # unseen geometry could fail capacity instead of computing. Clamp the
    # generic tile height to what the budget admits; the row tile height is
    # numerically irrelevant here (each row is reduced over BLOCK_N alone), so
    # this can only change program count, never a value.
    _UB_BUDGET = 253952
    _BYTES_PER_ELEM = 5 * 4 + 2 * 1  # x/z/y fp32 + 2 predicate tiles
    BM_G = BLOCK_M
    while BM_G > 1 and BM_G * BLOCK_N * _BYTES_PER_ELEM + 8192 > _UB_BUDGET:
        BM_G //= 2

    _ln_fwd_kernel[(triton.cdiv(M, BM_G), ngroups)](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        gs,
        eps,
        BLOCK_M=BM_G,
        BLOCK_N=BLOCK_N,
        HAS_BIAS=bias is not None,
        HAS_Z=z is not None,
        NORM_BEFORE_GATE=bool(norm_before_gate),
        IS_RMS_NORM=bool(is_rms_norm),
        EVEN_ROWS=(M % BM_G == 0),
        FULL_COLS=(BLOCK_N == gs),
    )
    return out, mean, rstd


def layer_norm_fwd_npu(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    """Return normalized activations, optional group means and FP32 reciprocal std."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise ValueError("expected a 2D BF16 NPU input")
    m, n = x.shape
    if n < 1 or n > 128:
        raise ValueError("supported width is [1, 128]")
    group_size = n if group_size is None else group_size
    if group_size < 1 or n % group_size:
        raise ValueError("group size must divide the normalization width")
    if weight.shape != (n,) or bias is not None and bias.shape != (n,):
        raise ValueError("weight and bias must have shape [width]")
    if z is not None and z.shape != x.shape or out is not None and out.shape != x.shape:
        raise ValueError("gate and output must match the activation shape")
    for t in (x, weight, bias, z, out):
        if t is not None and (
            t.device != x.device or t.dtype != x.dtype or not t.is_contiguous()
        ):
            raise ValueError("expected contiguous BF16 inputs on one NPU")
    if out is not None and any(
        torch._C._is_alias_of(out, t) for t in (x, weight, bias, z) if t is not None
    ):
        raise ValueError("preallocated output must not overlap an input")
    with torch_device_fn.device(x.device):
        return _launch(
            x, weight, bias, eps, z, group_size, norm_before_gate, is_rms_norm, out
        )

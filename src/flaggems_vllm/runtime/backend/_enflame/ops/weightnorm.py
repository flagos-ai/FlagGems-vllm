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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)

# GCU (24-SIP) shape-derived tile heuristics, probe-tuned on
# benchmark/test_weight_norm_interface.py + test_weight_norm.py core shapes:
# num_warps=1 wins on every reduction-chain form; dim-0 tiles target ~32K
# elements with the whole row resident when N fits one block; dim-last tiles
# pick BLOCK_COL so grid ~= 32 programs, an elementwise [COL, ROW] accumulator
# for narrow column blocks and a per-chunk tree reduction for wide ones.
_DSM_TILE_BUDGET = 393216
_GRID_X_LIMIT = 65535


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_first(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    rvec = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
    if WIDE:
        rvec = rvec.to(tl.int64)
    rows = rvec[:, None]
    rmask = rvec < M
    if N <= BLOCK_COL_SIZE:
        # whole row resident: one load feeds both the norm and the scale
        cols = tl.arange(0, BLOCK_COL_SIZE)[None, :]
        m = rmask[:, None] & (cols < N)
        x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
        nrm = tl.sqrt(tl.sum(x * x, axis=1) + eps)
        tl.store(norm + rvec, nrm, mask=rmask)
        gv = tl.load(g + rvec, mask=rmask, other=0.0).to(tl.float32)
        tl.store(output + rows * N + cols, x * (gv / nrm)[:, None], mask=m)
    else:
        acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
        for base in range(0, N, BLOCK_COL_SIZE):
            cols = base + tl.arange(0, BLOCK_COL_SIZE)[None, :]
            m = rmask[:, None] & (cols < N)
            x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            acc += x * x
        nrm = tl.sqrt(tl.sum(acc, axis=1) + eps)
        tl.store(norm + rvec, nrm, mask=rmask)
        gv = tl.load(g + rvec, mask=rmask, other=0.0).to(tl.float32)
        scale = (gv / nrm)[:, None]
        for base in range(0, N, BLOCK_COL_SIZE):
            cols = base + tl.arange(0, BLOCK_COL_SIZE)[None, :]
            m = rmask[:, None] & (cols < N)
            x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            tl.store(output + rows * N + cols, x * scale, mask=m)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_last(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    cvec = pid * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    cols = cvec[None, :]
    cmask = cvec < N
    rvec = tl.arange(0, BLOCK_ROW_SIZE)
    if WIDE:
        rvec = rvec.to(tl.int64)
    if M <= BLOCK_ROW_SIZE:
        # whole column block resident: one load feeds norm and scale
        rows = rvec[:, None]
        m = (rows < M) & cmask[None, :]
        x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
        nrm = tl.sqrt(tl.sum(x * x, axis=0) + eps)
        tl.store(norm + cvec, nrm, mask=cmask)
        gv = tl.load(g + cvec, mask=cmask, other=0.0).to(tl.float32)
        tl.store(output + rows * N + cols, x * (gv / nrm)[None, :], mask=m)
    else:
        vacc = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
        for base in range(0, M, BLOCK_ROW_SIZE):
            rows = base + rvec[:, None]
            m = (rows < M) & cmask[None, :]
            x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            vacc += tl.sum(x * x, axis=0)
        nrm = tl.sqrt(vacc + eps)
        tl.store(norm + cvec, nrm, mask=cmask)
        gv = tl.load(g + cvec, mask=cmask, other=0.0).to(tl.float32)
        scale = gv / nrm
        for base in range(0, M, BLOCK_ROW_SIZE):
            rows = base + rvec[:, None]
            m = (rows < M) & cmask[None, :]
            x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            tl.store(output + rows * N + cols, x * scale[None, :], mask=m)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_last_acc(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    WIDE: tl.constexpr,
):
    # narrow column blocks with deep row loops: elementwise [COL, ROW]
    # accumulator beats a per-chunk reduction tree (probe: 2.6x on 1M rows)
    pid = tl.program_id(0)
    cvec = pid * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    cols = cvec[:, None]
    cmask = cvec < N
    rvec = tl.arange(0, BLOCK_ROW_SIZE)
    if WIDE:
        rvec = rvec.to(tl.int64)
    acc = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        rows = base + rvec[None, :]
        m = cmask[:, None] & (rows < M)
        x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
        acc += x * x
    nrm = tl.sqrt(tl.sum(acc, axis=1) + eps)
    tl.store(norm + cvec, nrm, mask=cmask)
    gv = tl.load(g + cvec, mask=cmask, other=0.0).to(tl.float32)
    scale = gv / nrm
    # scale pass uses the natural [ROW, COL] layout: masked stores in the
    # transposed [COL, ROW] form silently corrupt the tail chunk (P15 family)
    cols_t = cvec[None, :]
    for base in range(0, M, BLOCK_ROW_SIZE):
        rows = base + rvec[:, None]
        m = (rows < M) & cmask[None, :]
        x = tl.load(v + rows * N + cols_t, mask=m, other=0.0).to(tl.float32)
        tl.store(output + rows * N + cols_t, x * scale[None, :], mask=m)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_first(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    rvec = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
    if WIDE:
        rvec = rvec.to(tl.int64)
    rows = rvec[:, None]
    rmask = rvec < M
    n1 = 1.0 / (tl.load(norm + rvec, mask=rmask, other=1.0).to(tl.float32) + eps)
    gv = tl.load(g + rvec, mask=rmask, other=0.0).to(tl.float32)
    n3 = n1 * n1 * n1
    if N <= BLOCK_COL_SIZE:
        cols = tl.arange(0, BLOCK_COL_SIZE)[None, :]
        m = rmask[:, None] & (cols < N)
        x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(w + rows * N + cols, mask=m, other=0.0).to(tl.float32)
        vw = tl.sum(x * wv, axis=1)
        tl.store(g_grad + rvec, vw * n1, mask=rmask)
        a = (gv * n1)[:, None]
        b = (gv * n3 * vw)[:, None]
        tl.store(v_grad + rows * N + cols, a * wv - b * x, mask=m)
    else:
        acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
        for base in range(0, N, BLOCK_COL_SIZE):
            cols = base + tl.arange(0, BLOCK_COL_SIZE)[None, :]
            m = rmask[:, None] & (cols < N)
            x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            acc += x * wv
        vw = tl.sum(acc, axis=1)
        tl.store(g_grad + rvec, vw * n1, mask=rmask)
        a = (gv * n1)[:, None]
        b = (gv * n3 * vw)[:, None]
        for base in range(0, N, BLOCK_COL_SIZE):
            cols = base + tl.arange(0, BLOCK_COL_SIZE)[None, :]
            m = rmask[:, None] & (cols < N)
            x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            tl.store(v_grad + rows * N + cols, a * wv - b * x, mask=m)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_last(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    cvec = pid * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    cols = cvec[None, :]
    cmask = cvec < N
    rvec = tl.arange(0, BLOCK_ROW_SIZE)
    if WIDE:
        rvec = rvec.to(tl.int64)
    n1 = 1.0 / (tl.load(norm + cvec, mask=cmask, other=1.0).to(tl.float32) + eps)
    gv = tl.load(g + cvec, mask=cmask, other=0.0).to(tl.float32)
    n3 = n1 * n1 * n1
    if M <= BLOCK_ROW_SIZE:
        rows = rvec[:, None]
        m = (rows < M) & cmask[None, :]
        x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(w + rows * N + cols, mask=m, other=0.0).to(tl.float32)
        vw = tl.sum(x * wv, axis=0)
        tl.store(g_grad + cvec, vw * n1, mask=cmask)
        tl.store(
            v_grad + rows * N + cols,
            (gv * n1)[None, :] * wv - (gv * n3 * vw)[None, :] * x,
            mask=m,
        )
    else:
        vacc = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)
        for base in range(0, M, BLOCK_ROW_SIZE):
            rows = base + rvec[:, None]
            m = (rows < M) & cmask[None, :]
            x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            vacc += tl.sum(x * wv, axis=0)
        tl.store(g_grad + cvec, vacc * n1, mask=cmask)
        a = gv * n1
        b = gv * n3 * vacc
        for base in range(0, M, BLOCK_ROW_SIZE):
            rows = base + rvec[:, None]
            m = (rows < M) & cmask[None, :]
            x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + rows * N + cols, mask=m, other=0.0).to(tl.float32)
            tl.store(v_grad + rows * N + cols, a[None, :] * wv - b[None, :] * x, mask=m)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_last_acc(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    WIDE: tl.constexpr,
):
    # see weight_norm_kernel_last_acc
    pid = tl.program_id(0)
    cvec = pid * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    cols = cvec[:, None]
    cmask = cvec < N
    rvec = tl.arange(0, BLOCK_ROW_SIZE)
    if WIDE:
        rvec = rvec.to(tl.int64)
    n1 = 1.0 / (tl.load(norm + cvec, mask=cmask, other=1.0).to(tl.float32) + eps)
    gv = tl.load(g + cvec, mask=cmask, other=0.0).to(tl.float32)
    n3 = n1 * n1 * n1
    acc = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        rows = base + rvec[None, :]
        m = cmask[:, None] & (rows < M)
        x = tl.load(v + rows * N + cols, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(w + rows * N + cols, mask=m, other=0.0).to(tl.float32)
        acc += x * wv
    vw = tl.sum(acc, axis=1)
    tl.store(g_grad + cvec, vw * n1, mask=cmask)
    a = gv * n1
    b = gv * n3 * vw
    # scale pass in natural [ROW, COL] layout (see weight_norm_kernel_last_acc)
    cols_t = cvec[None, :]
    for base in range(0, M, BLOCK_ROW_SIZE):
        rows = base + rvec[:, None]
        m = (rows < M) & cmask[None, :]
        x = tl.load(v + rows * N + cols_t, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(w + rows * N + cols_t, mask=m, other=0.0).to(tl.float32)
        tl.store(v_grad + rows * N + cols_t, a[None, :] * wv - b[None, :] * x, mask=m)


def _next_pow2(value):
    return int(triton.next_power_of_2(max(1, int(value))))


def _floor_pow2(value):
    return 1 << (max(1, int(value)).bit_length() - 1)


def _pick_first_tile(M, N, elem_size, budget):
    block_col = _next_pow2(N) if N <= 8192 else 8192
    block_row = _floor_pow2(min(budget // block_col, 32))
    if M >= 64:
        block_row = min(block_row, _floor_pow2(max(1, M // 32)))
    else:
        block_row = max(1, min(block_row, _next_pow2(M)))
    if triton.cdiv(M, block_row) > _GRID_X_LIMIT:
        cap = _floor_pow2(
            max(1, _DSM_TILE_BUDGET // (block_col * (4 + elem_size)))
        )
        block_row = min(
            max(block_row, _next_pow2(triton.cdiv(M, _GRID_X_LIMIT))), cap
        )
    return block_row, block_col


def _pick_last_tile(M, N, elem_size):
    block_col = max(1, min(256, _next_pow2(max(1, N // 32))))
    if M <= 1024 and _next_pow2(M) * block_col * (4 + elem_size) <= _DSM_TILE_BUDGET:
        return "resident", _next_pow2(M), block_col
    rmax = max(1, _DSM_TILE_BUDGET // (block_col * (4 + elem_size)))
    if block_col <= 32:
        return "acc", min(1024 if elem_size == 2 else 512, _floor_pow2(rmax)), block_col
    return "partial", min(512 if elem_size == 2 else 256, _floor_pow2(rmax)), block_col


def weight_norm_interface(v, g, dim=0):
    logger.debug("GEMS WEIGHT NORM INTERFACE FORWARD")
    v = v.contiguous()
    g = g.contiguous()
    output = torch.empty_like(v)
    norm = torch.empty_like(g)
    eps = torch.finfo(torch.float32).tiny
    wide = 1 if v.numel() >= 2**31 else 0
    elem_size = v.element_size()
    if dim == 0:
        M = v.shape[0]
        N = math.prod(v.shape[1:])
        block_row, block_col = _pick_first_tile(M, N, elem_size, 32768)
        grid = (triton.cdiv(M, block_row),)
        with torch_device_fn.device(v.device):
            weight_norm_kernel_first[grid](
                output,
                norm,
                v,
                g,
                M,
                N,
                eps=eps,
                BLOCK_ROW_SIZE=block_row,
                BLOCK_COL_SIZE=block_col,
                WIDE=wide,
                num_warps=1,
            )
    elif dim == v.ndim - 1:
        M = math.prod(v.shape[:-1])
        N = v.shape[dim]
        mode, block_row, block_col = _pick_last_tile(M, N, elem_size)
        grid = (triton.cdiv(N, block_col),)
        kernel = (
            weight_norm_kernel_last_acc if mode == "acc" else weight_norm_kernel_last
        )
        with torch_device_fn.device(v.device):
            kernel[grid](
                output,
                norm,
                v,
                g,
                M,
                N,
                eps=eps,
                BLOCK_ROW_SIZE=block_row,
                BLOCK_COL_SIZE=block_col,
                WIDE=wide,
                num_warps=1,
            )
    return output, norm


def weight_norm_interface_backward(w_grad, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS WEIGHT NORM INTERFACE BACKWARD")
    w_grad = w_grad.contiguous()
    saved_v = saved_v.contiguous()
    saved_g = saved_g.contiguous()
    saved_norms = saved_norms.contiguous()
    v_grad = torch.empty_like(saved_v)
    g_grad = torch.empty_like(saved_g)
    eps = torch.finfo(torch.float32).tiny
    wide = 1 if saved_v.numel() >= 2**31 else 0
    elem_size = saved_v.element_size()

    if dim == 0:
        M = saved_v.shape[0]
        N = math.prod(saved_v.shape[1:])
        block_row, block_col = _pick_first_tile(
            M, N, elem_size, 32768 if elem_size == 2 else 16384
        )
        grid = (triton.cdiv(M, block_row),)
        with torch_device_fn.device(saved_v.device):
            weight_norm_bwd_kernel_first[grid](
                v_grad,
                g_grad,
                w_grad,
                saved_v,
                saved_g,
                saved_norms,
                M,
                N,
                eps=eps,
                BLOCK_ROW_SIZE=block_row,
                BLOCK_COL_SIZE=block_col,
                WIDE=wide,
                num_warps=1,
            )
    elif dim == saved_v.ndim - 1:
        M = math.prod(saved_v.shape[:dim])
        N = saved_v.shape[dim]
        mode, block_row, block_col = _pick_last_tile(M, N, elem_size)
        grid = (triton.cdiv(N, block_col),)
        kernel = (
            weight_norm_bwd_kernel_last_acc
            if mode == "acc"
            else weight_norm_bwd_kernel_last
        )
        with torch_device_fn.device(saved_v.device):
            kernel[grid](
                v_grad,
                g_grad,
                w_grad,
                saved_v,
                saved_g,
                saved_norms,
                M,
                N,
                eps=eps,
                BLOCK_ROW_SIZE=block_row,
                BLOCK_COL_SIZE=block_col,
                WIDE=wide,
                num_warps=1,
            )
    return v_grad, g_grad

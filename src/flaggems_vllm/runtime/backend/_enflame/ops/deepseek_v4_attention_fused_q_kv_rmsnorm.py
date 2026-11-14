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

from typing import Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

_MAX_TILE_ELEMS = 32768
_MAX_CHUNK_TILE_ELEMS = 16384
_WIDE_SINGLE_MAX = 2048
_CHUNK_WIDTHS = (4096, 2048, 1024)
_MAX_CHUNKS = 16
_WIDE_CHUNK_MIN_TOKENS = 96
_GRID_X_LIMIT = 65535


@triton.jit
def _rmsnorm_rows_1d(
    in_ptr,
    out_ptr,
    w_ptr,
    row0,
    in_stride,
    out_stride,
    eps,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_in = in_ptr + row0.to(tl.int64) * in_stride
    row_out = out_ptr + row0.to(tl.int64) * out_stride
    offs = tl.arange(0, BLOCK)
    mask = offs < SIZE
    x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_rows_2d(
    in_ptr,
    out_ptr,
    w_ptr,
    row0,
    in_stride,
    out_stride,
    eps,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    RTILE: tl.constexpr,
):
    rows = row0 + tl.arange(0, RTILE)
    rows64 = rows.to(tl.int64)
    in_base = in_ptr + rows64[:, None] * in_stride
    out_base = out_ptr + rows64[:, None] * out_stride
    offs = tl.arange(0, BLOCK)
    if SIZE == BLOCK:
        x = tl.load(in_base + offs[None, :]).to(tl.float32)
        var = tl.sum(x * x, axis=1) / SIZE
        rrms = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + offs).to(tl.float32)
        y = x * rrms[:, None] * w[None, :]
        tl.store(out_base + offs[None, :], y.to(out_ptr.dtype.element_ty))
    else:
        m = (offs < SIZE)[None, :]
        x = tl.load(in_base + offs[None, :], mask=m, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / SIZE
        rrms = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + offs, mask=offs < SIZE, other=0.0).to(tl.float32)
        y = x * rrms[:, None] * w[None, :]
        tl.store(
            out_base + offs[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=m,
        )


@triton.jit
def _rmsnorm_rows_2d_chunked(
    in_ptr,
    out_ptr,
    w_ptr,
    row0,
    in_stride,
    out_stride,
    eps,
    SIZE: tl.constexpr,
    CHUNK: tl.constexpr,
    RTILE: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    rows = row0 + tl.arange(0, RTILE)
    rows64 = rows.to(tl.int64)
    in_base = in_ptr + rows64[:, None] * in_stride
    out_base = out_ptr + rows64[:, None] * out_stride
    acc = tl.zeros([RTILE, CHUNK], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        offs = c * CHUNK + tl.arange(0, CHUNK)
        x = tl.load(in_base + offs[None, :]).to(tl.float32)
        acc += x * x
    var = tl.sum(acc, axis=1) / SIZE
    rrms = tl.rsqrt(var + eps)
    for c in tl.static_range(NCHUNK):
        offs = c * CHUNK + tl.arange(0, CHUNK)
        x = tl.load(in_base + offs[None, :]).to(tl.float32)
        w = tl.load(w_ptr + offs).to(tl.float32)
        y = x * rrms[:, None] * w[None, :]
        tl.store(out_base + offs[None, :], y.to(out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["eps", "num_tokens", "nq_blocks"])
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    num_tokens,
    nq_blocks,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    Q_MODE: tl.constexpr,
    KV_MODE: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    Q_RTILE: tl.constexpr,
    KV_RTILE: tl.constexpr,
    Q_NCHUNK: tl.constexpr,
    KV_NCHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < nq_blocks:
        row0 = pid * Q_RTILE
        row0 = tl.minimum(row0, num_tokens - Q_RTILE)
        if Q_MODE == 0:
            _rmsnorm_rows_1d(
                q_ptr,
                q_out_ptr,
                q_weight_ptr,
                row0,
                q_in_stride,
                q_out_stride,
                eps,
                Q_SIZE,
                Q_BLOCK,
            )
        elif Q_MODE == 1:
            _rmsnorm_rows_2d(
                q_ptr,
                q_out_ptr,
                q_weight_ptr,
                row0,
                q_in_stride,
                q_out_stride,
                eps,
                Q_SIZE,
                Q_BLOCK,
                Q_RTILE,
            )
        else:
            _rmsnorm_rows_2d_chunked(
                q_ptr,
                q_out_ptr,
                q_weight_ptr,
                row0,
                q_in_stride,
                q_out_stride,
                eps,
                Q_SIZE,
                Q_BLOCK,
                Q_RTILE,
                Q_NCHUNK,
            )
    else:
        row0 = (pid - nq_blocks) * KV_RTILE
        row0 = tl.minimum(row0, num_tokens - KV_RTILE)
        if KV_MODE == 0:
            _rmsnorm_rows_1d(
                kv_ptr,
                kv_out_ptr,
                kv_weight_ptr,
                row0,
                kv_in_stride,
                kv_out_stride,
                eps,
                KV_SIZE,
                KV_BLOCK,
            )
        elif KV_MODE == 1:
            _rmsnorm_rows_2d(
                kv_ptr,
                kv_out_ptr,
                kv_weight_ptr,
                row0,
                kv_in_stride,
                kv_out_stride,
                eps,
                KV_SIZE,
                KV_BLOCK,
                KV_RTILE,
            )
        else:
            _rmsnorm_rows_2d_chunked(
                kv_ptr,
                kv_out_ptr,
                kv_weight_ptr,
                row0,
                kv_in_stride,
                kv_out_stride,
                eps,
                KV_SIZE,
                KV_BLOCK,
                KV_RTILE,
                KV_NCHUNK,
            )


def _largest_pow2_leq(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def _narrow_rtile(num_tokens: int, block: int) -> int:
    if block >= _WIDE_SINGLE_MAX:
        if num_tokens <= 160:
            rtile = 4
        elif num_tokens <= 384:
            rtile = 8
        else:
            rtile = 16
    else:
        rtile = 8 if num_tokens <= 160 else 16
    rtile = min(
        rtile,
        _largest_pow2_leq(num_tokens),
        max(1, _MAX_TILE_ELEMS // block),
    )
    return max(rtile, 1)


def _task_config(num_tokens: int, size: int) -> Tuple[int, int, int, int]:
    block = triton.next_power_of_2(size)
    if block <= _WIDE_SINGLE_MAX:
        rtile = _narrow_rtile(num_tokens, block)
        if rtile == 1:
            return 0, block, 1, 1
        return 1, block, rtile, 1
    if num_tokens >= _WIDE_CHUNK_MIN_TOKENS:
        for chunk in _CHUNK_WIDTHS:
            if size % chunk == 0:
                nchunk = size // chunk
                if 1 <= nchunk <= _MAX_CHUNKS:
                    rtile = min(
                        4,
                        _largest_pow2_leq(num_tokens),
                        max(1, _MAX_CHUNK_TILE_ELEMS // chunk),
                    )
                    if rtile >= 2:
                        return 2, chunk, rtile, nchunk
                break
    return 0, block, 1, 1


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out

    q_mode, q_block, q_rtile, q_nchunk = _task_config(num_tokens, q_size)
    kv_mode, kv_block, kv_rtile, kv_nchunk = _task_config(num_tokens, kv_size)

    nq = triton.cdiv(num_tokens, q_rtile)
    nkv = triton.cdiv(num_tokens, kv_rtile)
    while nq + nkv > _GRID_X_LIMIT and q_mode == 1 and kv_mode == 1:
        bumped = False
        if q_rtile < max(1, _MAX_TILE_ELEMS // q_block):
            q_rtile *= 2
            bumped = True
        if kv_rtile < max(1, _MAX_TILE_ELEMS // kv_block):
            kv_rtile *= 2
            bumped = True
        if not bumped:
            break
        nq = triton.cdiv(num_tokens, q_rtile)
        nkv = triton.cdiv(num_tokens, kv_rtile)

    with torch_device_fn.device(qr.device):
        _fused_q_kv_rmsnorm_kernel[(nq + nkv,)](
            qr,
            qr_out,
            q_weight,
            qr.stride(0),
            qr_out.stride(0),
            kv,
            kv_out,
            kv_weight,
            kv.stride(0),
            kv_out.stride(0),
            num_tokens,
            nq,
            eps,
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            Q_MODE=q_mode,
            KV_MODE=kv_mode,
            Q_BLOCK=q_block,
            KV_BLOCK=kv_block,
            Q_RTILE=q_rtile,
            KV_RTILE=kv_rtile,
            Q_NCHUNK=q_nchunk,
            KV_NCHUNK=kv_nchunk,
            num_warps=1,
        )
    return qr_out, kv_out


__all__ = ["fused_q_kv_rmsnorm"]

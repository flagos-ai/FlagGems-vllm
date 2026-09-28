#!/usr/bin/env python3
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

_TILE_M = 8
_NUM_WARPS = 4

_LARGE_N_THRESHOLD = 4096
_LARGE_N_BLOCK_N = 4096

_MUSA_MEDIUM_N_MIN = 512
_MUSA_MEDIUM_N_MAX = 2048

_MUSA_MEDIUM_N_CONFIGS = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
]
_MUSA_LARGE_FUSED_CONFIGS = [
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
]
_TILE_CONFIGS = [
    triton.Config({"TILE_M": 1, "N_BLOCK": 128}, num_warps=1),
    triton.Config({"TILE_M": 1, "N_BLOCK": 256}, num_warps=2),
    triton.Config({"TILE_M": 2, "N_BLOCK": 256}, num_warps=2),
    triton.Config({"TILE_M": 4, "N_BLOCK": 256}, num_warps=2),
    triton.Config({"TILE_M": 4, "N_BLOCK": 512}, num_warps=2),
    triton.Config({"TILE_M": 8, "N_BLOCK": 512}, num_warps=4),
    triton.Config({"TILE_M": 4, "N_BLOCK": 1024}, num_warps=2),
    triton.Config({"TILE_M": 8, "N_BLOCK": 1024}, num_warps=4),
    triton.Config({"TILE_M": 2, "N_BLOCK": 2048}, num_warps=2),
    triton.Config({"TILE_M": 4, "N_BLOCK": 2048}, num_warps=2),
    triton.Config({"TILE_M": 8, "N_BLOCK": 2048}, num_warps=4),
]


def _prune_tile_configs(configs, named_args, **_kwargs):
    max_size = max(_kwargs["Q_SIZE"], _kwargs["KV_SIZE"])
    return [c for c in configs if c.kwargs["N_BLOCK"] >= max_size]


def _check_inputs(qr, kv, q_weight, kv_weight):
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()
    assert q_weight.ndim == 1 and kv_weight.ndim == 1
    assert q_weight.numel() == qr.shape[1]
    assert kv_weight.numel() == kv.shape[1]
    assert qr.device == kv.device == q_weight.device == kv_weight.device


@triton.autotune(
    configs=_TILE_CONFIGS,
    key=["num_tokens", "Q_SIZE", "KV_SIZE"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
)
@triton.jit(do_not_specialize=["eps"])
def _fused_q_kv_rmsnorm_tile_kernel(
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
    eps,
    num_tokens,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    N_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)

    if task == 0:
        size, in_ptr, out_ptr = Q_SIZE, q_ptr, q_out_ptr
        weight_ptr, in_stride, out_stride = q_weight_ptr, q_in_stride, q_out_stride
    else:
        size, in_ptr, out_ptr = KV_SIZE, kv_ptr, kv_out_ptr
        weight_ptr, in_stride, out_stride = kv_weight_ptr, kv_in_stride, kv_out_stride

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = tl.arange(0, N_BLOCK)
    mask = (offs_m < num_tokens)[:, None] & (offs_n < size)[None, :]

    x = tl.load(
        in_ptr + offs_m[:, None] * in_stride + offs_n[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=1) / size + eps)
    w = tl.load(weight_ptr + offs_n, mask=offs_n < size, other=0.0).to(tl.float32)
    y = x * rrms[:, None] * w[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * out_stride + offs_n[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_q_kv_rmsnorm_tile(qr, kv, q_weight, kv_weight, eps):
    _check_inputs(qr, kv, q_weight, kv_weight)
    num_tokens = qr.shape[0]
    qr_out, kv_out = torch.empty_like(qr), torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out
    grid = lambda meta: (triton.cdiv(num_tokens, meta["TILE_M"]), 2)
    with torch_device_fn.device(qr.device):
        _fused_q_kv_rmsnorm_tile_kernel[grid](
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
            eps,
            num_tokens,
            Q_SIZE=qr.shape[1],
            KV_SIZE=kv.shape[1],
        )
    return qr_out, kv_out


@triton.autotune(
    configs=_MUSA_MEDIUM_N_CONFIGS,
    key=["num_rows", "Q_SIZE", "KV_SIZE"],
)
@triton.jit(do_not_specialize=["eps"])
def _fused_q_kv_rmsnorm_medium_kernel(
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
    eps,
    num_rows,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)

    if task == 0:
        size = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        row_out = q_out_ptr + token_idx * q_out_stride
        weight_ptr = q_weight_ptr
    else:
        size = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        row_out = kv_out_ptr + token_idx * kv_out_stride
        weight_ptr = kv_weight_ptr

    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < size
    x = tl.load(row_in + offs_n, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / size + eps)
    w = tl.load(weight_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
    tl.store(row_out + offs_n, (x * rrms * w).to(row_out.dtype.element_ty), mask=mask)


def fused_q_kv_rmsnorm_medium(qr, kv, q_weight, kv_weight, eps):
    _check_inputs(qr, kv, q_weight, kv_weight)
    num_tokens = qr.shape[0]
    q_out, kv_out = torch.empty_like(qr), torch.empty_like(kv)
    if num_tokens == 0:
        return q_out, kv_out
    block_n = triton.next_power_of_2(max(qr.shape[1], kv.shape[1]))
    with torch_device_fn.device(qr.device):
        _fused_q_kv_rmsnorm_medium_kernel[(num_tokens, 2)](
            qr,
            q_out,
            q_weight,
            qr.stride(0),
            q_out.stride(0),
            kv,
            kv_out,
            kv_weight,
            kv.stride(0),
            kv_out.stride(0),
            eps,
            num_tokens,
            Q_SIZE=qr.shape[1],
            KV_SIZE=kv.shape[1],
            BLOCK_N=block_n,
        )
    return q_out, kv_out


@triton.autotune(
    configs=_MUSA_LARGE_FUSED_CONFIGS,
    key=["num_rows", "Q_SIZE", "KV_SIZE"],
)
@triton.jit(do_not_specialize=["eps"])
def _fused_q_kv_rmsnorm_large_kernel(
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
    eps,
    num_rows,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)

    if task == 0:
        size = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        row_out = q_out_ptr + token_idx * q_out_stride
        weight_ptr = q_weight_ptr
    else:
        size = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        row_out = kv_out_ptr + token_idx * kv_out_stride
        weight_ptr = kv_weight_ptr

    offs_n = tl.arange(0, BLOCK_N)

    sum_sq = 0.0
    for tile_idx in tl.static_range(0, MAX_TILES):
        cols = tile_idx * BLOCK_N + offs_n
        mask = cols < size
        x = tl.load(row_in + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    rrms = tl.rsqrt(sum_sq / size + eps)

    for tile_idx in tl.static_range(0, MAX_TILES):
        cols = tile_idx * BLOCK_N + offs_n
        mask = cols < size
        x = tl.load(row_in + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(row_out + cols, (x * rrms * w).to(row_out.dtype.element_ty), mask=mask)


def _fused_q_kv_rmsnorm_large(qr, kv, q_weight, kv_weight, eps):
    num_tokens = qr.shape[0]
    q_out, kv_out = torch.empty_like(qr), torch.empty_like(kv)
    if num_tokens == 0:
        return q_out, kv_out
    max_tiles = triton.cdiv(max(qr.shape[1], kv.shape[1]), _LARGE_N_BLOCK_N)
    with torch_device_fn.device(qr.device):
        _fused_q_kv_rmsnorm_large_kernel[(num_tokens, 2)](
            qr,
            q_out,
            q_weight,
            qr.stride(0),
            q_out.stride(0),
            kv,
            kv_out,
            kv_weight,
            kv.stride(0),
            kv_out.stride(0),
            eps,
            num_tokens,
            Q_SIZE=qr.shape[1],
            KV_SIZE=kv.shape[1],
            BLOCK_N=_LARGE_N_BLOCK_N,
            MAX_TILES=max_tiles,
        )
    return q_out, kv_out


def _is_musa_medium_n(size: int) -> bool:
    return _MUSA_MEDIUM_N_MIN < size <= _MUSA_MEDIUM_N_MAX


def fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
    _check_inputs(qr, kv, q_weight, kv_weight)
    q_size, kv_size = qr.shape[1], kv.shape[1]
    has_large = q_size >= _LARGE_N_THRESHOLD or kv_size >= _LARGE_N_THRESHOLD
    has_medium = _is_musa_medium_n(q_size) or _is_musa_medium_n(kv_size)

    if has_large:
        return _fused_q_kv_rmsnorm_large(qr, kv, q_weight, kv_weight, eps)
    elif has_medium:
        return fused_q_kv_rmsnorm_medium(qr, kv, q_weight, kv_weight, eps)
    else:
        return fused_q_kv_rmsnorm_tile(qr, kv, q_weight, kv_weight, eps)


__all__ = [
    "fused_q_kv_rmsnorm",
]

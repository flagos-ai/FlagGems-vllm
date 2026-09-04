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

_LARGE_N_THRESHOLD = 4096

_HYGON_TILE_CONFIGS = [
    triton.Config({"TILE_M": 1, "N_BLOCK": 128}, num_warps=1),
    triton.Config({"TILE_M": 2, "N_BLOCK": 256}, num_warps=2),
    triton.Config({"TILE_M": 2, "N_BLOCK": 512}, num_warps=2),
    triton.Config({"TILE_M": 2, "N_BLOCK": 1024}, num_warps=4),
    triton.Config({"TILE_M": 1, "N_BLOCK": 2048}, num_warps=4),
    triton.Config({"TILE_M": 1, "N_BLOCK": 4096}, num_warps=4),
]

_HYGON_LARGE_CONFIGS = [
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
]


def _prune_tile_configs(configs, named_args, **kwargs):
    max_size = max(
        kwargs["Q_SIZE"],
        kwargs["KV_SIZE"],
    )
    return [config for config in configs if config.kwargs["N_BLOCK"] >= max_size]


def _check_inputs(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
):
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == 1
    assert kv.stride(-1) == 1
    assert q_weight.is_contiguous()
    assert kv_weight.is_contiguous()
    assert q_weight.ndim == 1
    assert kv_weight.ndim == 1
    assert q_weight.numel() == qr.shape[1]
    assert kv_weight.numel() == kv.shape[1]
    assert qr.device == kv.device == q_weight.device == kv_weight.device


@triton.autotune(
    configs=_HYGON_TILE_CONFIGS,
    key=[
        "num_tokens",
        "Q_SIZE",
        "KV_SIZE",
    ],
    prune_configs_by={
        "early_config_prune": _prune_tile_configs,
    },
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

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = tl.arange(0, N_BLOCK)

    token_mask = (offs_m < num_tokens)[:, None]

    if task == 0:
        mask = token_mask & (offs_n < Q_SIZE)[None, :]

        x = tl.load(
            q_ptr + offs_m[:, None] * q_in_stride + offs_n[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        sum_sq = tl.sum(
            x * x,
            axis=1,
        )

        rrms = tl.rsqrt(sum_sq / Q_SIZE + eps)

        w = tl.load(
            q_weight_ptr + offs_n,
            mask=offs_n < Q_SIZE,
            other=0.0,
        ).to(tl.float32)

        y = x * rrms[:, None] * w[None, :]

        tl.store(
            q_out_ptr + offs_m[:, None] * q_out_stride + offs_n[None, :],
            y.to(q_out_ptr.dtype.element_ty),
            mask=mask,
        )

    else:
        mask = token_mask & (offs_n < KV_SIZE)[None, :]

        x = tl.load(
            kv_ptr + offs_m[:, None] * kv_in_stride + offs_n[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        sum_sq = tl.sum(
            x * x,
            axis=1,
        )

        rrms = tl.rsqrt(sum_sq / KV_SIZE + eps)

        w = tl.load(
            kv_weight_ptr + offs_n,
            mask=offs_n < KV_SIZE,
            other=0.0,
        ).to(tl.float32)

        y = x * rrms[:, None] * w[None, :]

        tl.store(
            kv_out_ptr + offs_m[:, None] * kv_out_stride + offs_n[None, :],
            y.to(kv_out_ptr.dtype.element_ty),
            mask=mask,
        )


def _fused_q_kv_rmsnorm_tile(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = qr.shape[0]

    q_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)

    if num_tokens == 0:
        return q_out, kv_out

    grid = lambda meta: (
        triton.cdiv(
            num_tokens,
            meta["TILE_M"],
        ),
        2,
    )

    with torch_device_fn.device(qr.device):
        _fused_q_kv_rmsnorm_tile_kernel[grid](
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
        )

    return q_out, kv_out


@triton.autotune(
    configs=_HYGON_LARGE_CONFIGS,
    key=[
        "num_rows",
        "Q_SIZE",
        "KV_SIZE",
    ],
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
    Q_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)

    if task == 0:
        q_offs = tl.arange(0, Q_BLOCK)
        q_mask = q_offs < Q_SIZE

        q_row_in = q_ptr + token_idx * q_in_stride

        q_x = tl.load(
            q_row_in + q_offs,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)

        q_sum_sq = tl.sum(
            q_x * q_x,
            axis=0,
        )

        q_rrms = tl.rsqrt(q_sum_sq / Q_SIZE + eps)

        q_w = tl.load(
            q_weight_ptr + q_offs,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)

        q_y = q_x * q_rrms * q_w

        q_row_out = q_out_ptr + token_idx * q_out_stride

        tl.store(
            q_row_out + q_offs,
            q_y.to(q_out_ptr.dtype.element_ty),
            mask=q_mask,
        )

    else:
        kv_offs = tl.arange(0, KV_BLOCK)
        kv_mask = kv_offs < KV_SIZE

        kv_row_in = kv_ptr + token_idx * kv_in_stride

        kv_x = tl.load(
            kv_row_in + kv_offs,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)

        kv_sum_sq = tl.sum(
            kv_x * kv_x,
            axis=0,
        )

        kv_rrms = tl.rsqrt(kv_sum_sq / KV_SIZE + eps)

        kv_w = tl.load(
            kv_weight_ptr + kv_offs,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)

        kv_y = kv_x * kv_rrms * kv_w

        kv_row_out = kv_out_ptr + token_idx * kv_out_stride

        tl.store(
            kv_row_out + kv_offs,
            kv_y.to(kv_out_ptr.dtype.element_ty),
            mask=kv_mask,
        )


def _fused_q_kv_rmsnorm_large(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = qr.shape[0]

    q_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)

    if num_tokens == 0:
        return q_out, kv_out

    q_block = triton.next_power_of_2(qr.shape[1])

    kv_block = triton.next_power_of_2(kv.shape[1])

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
            Q_BLOCK=q_block,
            KV_BLOCK=kv_block,
        )

    return q_out, kv_out


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_inputs(
        qr,
        kv,
        q_weight,
        kv_weight,
    )

    q_size = qr.shape[1]
    kv_size = kv.shape[1]

    if q_size >= _LARGE_N_THRESHOLD or kv_size >= _LARGE_N_THRESHOLD:
        return _fused_q_kv_rmsnorm_large(
            qr,
            kv,
            q_weight,
            kv_weight,
            eps,
        )

    return _fused_q_kv_rmsnorm_tile(
        qr,
        kv,
        q_weight,
        kv_weight,
        eps,
    )


__all__ = ["fused_q_kv_rmsnorm"]

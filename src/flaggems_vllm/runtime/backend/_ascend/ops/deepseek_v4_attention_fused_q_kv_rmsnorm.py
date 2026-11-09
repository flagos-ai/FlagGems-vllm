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
import triton.experimental.tle as tle
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

# Best measured combination on Ascend910B4 (do_bench_npu device time, all 7
# benchmark shapes >= 1.15x vs torch_npu.npu_rms_norm x2):
# - narrow rows (whole-tile single pass fits UB), tokens <= 1024: ONE launch
#   with q/kv on the task axis (uni_launch); single cast per element, weights
#   DMA'd into UB once per program.
# - otherwise: sep_launch per tensor, each routed by its own width --
#   narrow -> whole-tile single pass (sp); wide -> GM-streaming two-pass
#   (sep_2p_wub, pass 2 re-reads L2-resident rows; multibuffered TILE_N loop).
#   Mirrors CANN's own per-tensor template selection (MERGE_N / SPLIT_D).
# Tile sizes come from measured sweeps instead of runtime autotune
# (one config per shape key -> one compile per shape).
_SP_TILE_BUDGET_BYTES = 128 * 1024
_SP_TASK_MAX_TOKENS = 1024

_CORE_NUM = None


def _core_num() -> int:
    global _CORE_NUM
    if _CORE_NUM is None:
        try:
            import torch_npu

            device = torch.npu.current_device()
            torch.npu.set_device(device)
            _CORE_NUM = torch_npu.npu.get_device_limit(device)["vector_core_num"]
        except Exception:
            _CORE_NUM = 24
    return _CORE_NUM


def _sp_feasible(n_pad2: int, esz: int) -> bool:
    # can one [TILE_M, n_pad2] whole tile live in the per-core UB budget?
    return n_pad2 * (4 + esz) <= _SP_TILE_BUDGET_BYTES


def _sp_budget_tile(n_pad2: int, esz: int) -> int:
    # TILE_M from the UB budget alone (caller guarantees _sp_feasible)
    row = n_pad2 * (4 + esz)
    tile = 1
    while tile * 2 * row <= _SP_TILE_BUDGET_BYTES:
        tile *= 2
    return tile


def _sep_sp_wub_tile_m(num_tokens: int, n_pad2: int, esz: int) -> int:
    # TILE_M for the sep sp route: UB budget, capped so the grid still
    # floods the vector cores
    flood = triton.next_power_of_2(max(1, triton.cdiv(num_tokens, _core_num())))
    return max(
        1,
        min(
            _sp_budget_tile(n_pad2, esz),
            flood,
            triton.next_power_of_2(max(num_tokens, 1)),
        ),
    )


def _uni_feasible(q_pad2: int, kv_pad2: int, q_esz: int, kv_esz: int) -> bool:
    return _sp_feasible(q_pad2, q_esz) and _sp_feasible(kv_pad2, kv_esz)


def _uni_sp_wub_tile_m(
    num_tokens: int, q_pad2: int, kv_pad2: int, q_esz: int, kv_esz: int
) -> int:
    # TILE_M for the uni launch: min of both tensors' budgets, capped only
    # by the token count (no flood cap -- measured better for the combined kernel)
    return max(
        1,
        min(
            _sp_budget_tile(q_pad2, q_esz),
            _sp_budget_tile(kv_pad2, kv_esz),
            triton.next_power_of_2(max(num_tokens, 1)),
        ),
    )


def _sep_2p_tiles(num_tokens: int) -> Tuple[int, int]:
    # largest TILE_M whose grid still fills ~3/4 of the vector cores;
    # largest TILE_N whose per-element staging (f32 cast 4B + y f32 4B +
    # y bf16 2B) stays within ~160KB
    min_programs = max(8, _core_num() * 3 // 4)
    tile_m = 1
    while tile_m < 16 and triton.cdiv(num_tokens, tile_m * 2) >= min_programs:
        tile_m *= 2
    tile_n = 12288
    while tile_m * tile_n * 10 > 160 * 1024 and tile_n > 256:
        tile_n //= 2
    return tile_m, tile_n


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


@triton.jit(do_not_specialize=["eps"])
def _rmsnorm_uni_sp_wub_kernel(
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
    Q_PAD2: tl.constexpr,
    KV_PAD2: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)

    offs_m = pid * TILE_M + tl.arange(0, TILE_M)
    token_mask = offs_m < num_tokens

    # branch-local names must differ (triton if/else type-merge on shapes)
    if task == 0:
        q_w_ub = tle.dsa.alloc(
            [Q_PAD2],
            dtype=q_weight_ptr.dtype.element_ty,
            mem_addr_space=tle.dsa.ascend.UB,
        )
        q_wcols = tl.arange(0, Q_PAD2)
        tle.dsa.copy(q_weight_ptr + q_wcols, q_w_ub, [Q_SIZE])
        q_w = tle.dsa.to_tensor(q_w_ub, writable=False).to(tl.float32)

        q_cols = tl.arange(0, Q_PAD2)
        q_mask = token_mask[:, None] & (q_cols < Q_SIZE)[None, :]
        q_x = tl.load(
            q_ptr + offs_m[:, None] * q_in_stride + q_cols[None, :],
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        q_ssq = tl.sum(q_x * q_x, axis=1)
        q_rrms = tl.rsqrt(q_ssq / Q_SIZE + eps)
        q_y = q_x * q_rrms[:, None] * q_w[None, :]
        tl.store(
            q_out_ptr + offs_m[:, None] * q_out_stride + q_cols[None, :],
            q_y.to(q_out_ptr.dtype.element_ty),
            mask=q_mask,
        )
    else:
        kv_w_ub = tle.dsa.alloc(
            [KV_PAD2],
            dtype=kv_weight_ptr.dtype.element_ty,
            mem_addr_space=tle.dsa.ascend.UB,
        )
        kv_wcols = tl.arange(0, KV_PAD2)
        tle.dsa.copy(kv_weight_ptr + kv_wcols, kv_w_ub, [KV_SIZE])
        kv_w = tle.dsa.to_tensor(kv_w_ub, writable=False).to(tl.float32)

        kv_cols = tl.arange(0, KV_PAD2)
        kv_mask = token_mask[:, None] & (kv_cols < KV_SIZE)[None, :]
        kv_x = tl.load(
            kv_ptr + offs_m[:, None] * kv_in_stride + kv_cols[None, :],
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        kv_ssq = tl.sum(kv_x * kv_x, axis=1)
        kv_rrms = tl.rsqrt(kv_ssq / KV_SIZE + eps)
        kv_y = kv_x * kv_rrms[:, None] * kv_w[None, :]
        tl.store(
            kv_out_ptr + offs_m[:, None] * kv_out_stride + kv_cols[None, :],
            kv_y.to(kv_out_ptr.dtype.element_ty),
            mask=kv_mask,
        )


@triton.jit(do_not_specialize=["eps"])
def _rmsnorm_sep_sp_wub_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    x_in_stride,
    out_stride,
    eps,
    num_tokens,
    N_SIZE: tl.constexpr,
    N_PAD2: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * TILE_M + tl.arange(0, TILE_M)
    token_mask = offs_m < num_tokens

    w_ub = tle.dsa.alloc(
        [N_PAD2],
        dtype=w_ptr.dtype.element_ty,
        mem_addr_space=tle.dsa.ascend.UB,
    )
    wcols = tl.arange(0, N_PAD2)
    tle.dsa.copy(w_ptr + wcols, w_ub, [N_SIZE])
    w = tle.dsa.to_tensor(w_ub, writable=False).to(tl.float32)

    cols = tl.arange(0, N_PAD2)
    mask = token_mask[:, None] & (cols < N_SIZE)[None, :]
    x = tl.load(
        x_ptr + offs_m[:, None] * x_in_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    ssq = tl.sum(x * x, axis=1)
    rrms = tl.rsqrt(ssq / N_SIZE + eps)
    y = x * rrms[:, None] * w[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * out_stride + cols[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=["eps"])
def _rmsnorm_sep_2p_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    x_in_stride,
    out_stride,
    eps,
    num_tokens,
    N_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * TILE_M + tl.arange(0, TILE_M)
    token_mask = offs_m < num_tokens
    cols = tl.arange(0, TILE_N)

    ssq = tl.zeros((TILE_M,), tl.float32)
    for nb in tl.range(0, N_SIZE, TILE_N):
        offs = nb + cols
        m = token_mask[:, None] & (offs < N_SIZE)[None, :]
        x = tl.load(
            x_ptr + offs_m[:, None] * x_in_stride + offs[None, :],
            mask=m,
            other=0.0,
        ).to(tl.float32)
        ssq += tl.sum(x * x, axis=1)
    rrms = tl.rsqrt(ssq / N_SIZE + eps)

    for nb in tl.range(0, N_SIZE, TILE_N):
        offs = nb + cols
        m = token_mask[:, None] & (offs < N_SIZE)[None, :]
        x = tl.load(
            x_ptr + offs_m[:, None] * x_in_stride + offs[None, :],
            mask=m,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=offs < N_SIZE, other=0.0).to(tl.float32)
        y = x * rrms[:, None] * w[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * out_stride + offs[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=m,
        )


def _uni_launch(qr, kv, q_weight, kv_weight, eps, q_out, kv_out):
    num_tokens = qr.shape[0]
    tile_m = _uni_sp_wub_tile_m(
        num_tokens,
        triton.next_power_of_2(qr.shape[1]),
        triton.next_power_of_2(kv.shape[1]),
        qr.element_size(),
        kv.element_size(),
    )
    num_tile_m = triton.cdiv(num_tokens, tile_m)
    grid = (num_tile_m, 2)
    with torch_device_fn.device(qr.device):
        _rmsnorm_uni_sp_wub_kernel[grid](
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
            Q_PAD2=triton.next_power_of_2(qr.shape[1]),
            KV_PAD2=triton.next_power_of_2(kv.shape[1]),
            TILE_M=tile_m,
            num_warps=8,
            multibuffer=False,
        )


def _sep_launch(qr, kv, q_weight, kv_weight, eps, q_out, kv_out):
    # one sep entry: q and kv each get their own launch, routed by width
    for x, w, out in ((qr, q_weight, q_out), (kv, kv_weight, kv_out)):
        num_tokens = x.shape[0]
        n_pad2 = triton.next_power_of_2(x.shape[1])
        use_sp = _sp_feasible(n_pad2, x.element_size())

        with torch_device_fn.device(x.device):
            if use_sp:
                tile_m = _sep_sp_wub_tile_m(num_tokens, n_pad2, x.element_size())
                num_tile_m = triton.cdiv(num_tokens, tile_m)
                grid = (num_tile_m,)
                _rmsnorm_sep_sp_wub_kernel[grid](
                    x,
                    out,
                    w,
                    x.stride(0),
                    out.stride(0),
                    eps,
                    num_tokens,
                    N_SIZE=x.shape[1],
                    N_PAD2=n_pad2,
                    TILE_M=tile_m,
                    num_warps=8,
                    multibuffer=False,
                )
            else:
                tile_m, tile_n = _sep_2p_tiles(num_tokens)
                num_tile_m = triton.cdiv(num_tokens, tile_m)
                grid = (num_tile_m,)
                _rmsnorm_sep_2p_kernel[grid](
                    x,
                    out,
                    w,
                    x.stride(0),
                    out.stride(0),
                    eps,
                    num_tokens,
                    N_SIZE=x.shape[1],
                    TILE_M=tile_m,
                    TILE_N=tile_n,
                    num_warps=8,
                    multibuffer=True,
                )


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_inputs(qr, kv, q_weight, kv_weight)

    num_tokens = qr.shape[0]
    q_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return q_out, kv_out

    use_uni = (
        _uni_feasible(
            triton.next_power_of_2(qr.shape[1]),
            triton.next_power_of_2(kv.shape[1]),
            qr.element_size(),
            kv.element_size(),
        )
        and num_tokens <= _SP_TASK_MAX_TOKENS
    )
    if use_uni:
        _uni_launch(qr, kv, q_weight, kv_weight, eps, q_out, kv_out)
        return q_out, kv_out
    else:
        _sep_launch(qr, kv, q_weight, kv_weight, eps, q_out, kv_out)
        return q_out, kv_out


__all__ = ["fused_q_kv_rmsnorm"]

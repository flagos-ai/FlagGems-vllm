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

import inspect
import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

logger = logging.getLogger(__name__)

# FlagTree reorder-loop-loads: both H loops are latency bound at small token
# counts (one program per token, 8..16 loads per block). The autotune space
# below keeps the five original configs and adds, for each of them, the loop
# unrolled by 2, and unrolled by 2 / 4 with `tle.range(..., reorder=1)`, which
# issues the next block's loads before the current block's FMAs. Autotune
# picks per (H, HC), so a shape that gains nothing keeps its original config.
# Upstream Triton sees only the original configs.
try:
    import triton.experimental.tle.language as tle

    _HAS_REORDER_LOOP_LOADS = (
        has_triton_tle(3, 6, 0)
        and "reorder" in inspect.signature(getattr(tle, "range").__init__).parameters
    )
except (ImportError, AttributeError, ValueError):
    tle = None
    _HAS_REORDER_LOOP_LOADS = False


def _hc_configs():
    base = [
        (512, 4, 2),
        (1024, 4, 2),
        (1024, 8, 2),
        (2048, 8, 2),
        (2048, 8, 3),
    ]
    schedules = [(1, False)]
    if _HAS_REORDER_LOOP_LOADS:
        schedules += [(2, False), (2, True), (4, True)]
    return [
        triton.Config(
            {"BLOCK_H": block_h, "LOOP_UNROLL": unroll, "REORDER": reorder},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_h, num_warps, num_stages in base
        for unroll, reorder in schedules
    ]


@triton.jit
def _hc_accumulate(
    residual_ptr,
    fn_ptr,
    x_base,
    h_start,
    H: tl.constexpr,
    fn_stride_m,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    sqr_acc,
    mix_acc0,
    mix_acc1,
    mix_acc2,
    mix_acc3,
):
    h_off = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    r0 = tl.load(residual_ptr + x_base + 0 * H + h_off, mask=h_mask, other=0.0).to(
        tl.float32
    )
    r1 = tl.load(residual_ptr + x_base + 1 * H + h_off, mask=h_mask, other=0.0).to(
        tl.float32
    )
    sqr_acc += r0 * r0 + r1 * r1

    fn00 = tl.load(fn_ptr + 0 * fn_stride_m + 0 * H + h_off, mask=h_mask, other=0.0)
    fn01 = tl.load(fn_ptr + 0 * fn_stride_m + 1 * H + h_off, mask=h_mask, other=0.0)
    mix_acc0 += r0 * fn00 + r1 * fn01

    fn10 = tl.load(fn_ptr + 1 * fn_stride_m + 0 * H + h_off, mask=h_mask, other=0.0)
    fn11 = tl.load(fn_ptr + 1 * fn_stride_m + 1 * H + h_off, mask=h_mask, other=0.0)
    mix_acc1 += r0 * fn10 + r1 * fn11

    if HC > 2:
        r2 = tl.load(residual_ptr + x_base + 2 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        r3 = tl.load(residual_ptr + x_base + 3 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        sqr_acc += r2 * r2 + r3 * r3

        mix_acc0 += r2 * tl.load(
            fn_ptr + 0 * fn_stride_m + 2 * H + h_off, mask=h_mask, other=0.0
        )
        mix_acc0 += r3 * tl.load(
            fn_ptr + 0 * fn_stride_m + 3 * H + h_off, mask=h_mask, other=0.0
        )

        mix_acc1 += r2 * tl.load(
            fn_ptr + 1 * fn_stride_m + 2 * H + h_off, mask=h_mask, other=0.0
        )
        mix_acc1 += r3 * tl.load(
            fn_ptr + 1 * fn_stride_m + 3 * H + h_off, mask=h_mask, other=0.0
        )

        fn20 = tl.load(fn_ptr + 2 * fn_stride_m + 0 * H + h_off, mask=h_mask, other=0.0)
        fn21 = tl.load(fn_ptr + 2 * fn_stride_m + 1 * H + h_off, mask=h_mask, other=0.0)
        fn22 = tl.load(fn_ptr + 2 * fn_stride_m + 2 * H + h_off, mask=h_mask, other=0.0)
        fn23 = tl.load(fn_ptr + 2 * fn_stride_m + 3 * H + h_off, mask=h_mask, other=0.0)
        mix_acc2 += r0 * fn20 + r1 * fn21 + r2 * fn22 + r3 * fn23

        fn30 = tl.load(fn_ptr + 3 * fn_stride_m + 0 * H + h_off, mask=h_mask, other=0.0)
        fn31 = tl.load(fn_ptr + 3 * fn_stride_m + 1 * H + h_off, mask=h_mask, other=0.0)
        fn32 = tl.load(fn_ptr + 3 * fn_stride_m + 2 * H + h_off, mask=h_mask, other=0.0)
        fn33 = tl.load(fn_ptr + 3 * fn_stride_m + 3 * H + h_off, mask=h_mask, other=0.0)
        mix_acc3 += r0 * fn30 + r1 * fn31 + r2 * fn32 + r3 * fn33
    return sqr_acc, mix_acc0, mix_acc1, mix_acc2, mix_acc3


@triton.jit
def _hc_weighted_store(
    residual_ptr,
    out_ptr,
    x_base,
    out_base,
    h_start,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    pre_mix0,
    pre_mix1,
    pre_mix2,
    pre_mix3,
):
    h_off = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_off < H
    r0 = tl.load(residual_ptr + x_base + 0 * H + h_off, mask=h_mask, other=0.0).to(
        tl.float32
    )
    r1 = tl.load(residual_ptr + x_base + 1 * H + h_off, mask=h_mask, other=0.0).to(
        tl.float32
    )
    acc = pre_mix0 * r0 + pre_mix1 * r1
    if HC > 2:
        r2 = tl.load(residual_ptr + x_base + 2 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        r3 = tl.load(residual_ptr + x_base + 3 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        acc += pre_mix2 * r2 + pre_mix3 * r3
    tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16), mask=h_mask)


@triton.autotune(
    configs=_hc_configs(),
    key=["H", "HC"],
)
@triton.jit
def _hc_head_fused_kernel(
    residual_ptr,
    fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    out_ptr,
    T,
    H: tl.constexpr,
    rms_eps,
    hc_eps,
    residual_stride_t,
    fn_stride_m,
    out_stride_t,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    LOOP_UNROLL: tl.constexpr,
    REORDER: tl.constexpr,
):
    pid_t = tl.program_id(0)
    if pid_t >= T:
        return

    x_base = pid_t * residual_stride_t

    # Pass 1: iterate over H blocks to compute sqrsum and mixes
    sqr_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix_acc0 = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix_acc1 = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix_acc2 = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix_acc3 = tl.zeros([BLOCK_H], dtype=tl.float32)

    if REORDER:
        for h_start in tle.range(
            0, H, BLOCK_H, loop_unroll_factor=LOOP_UNROLL, reorder=1
        ):
            sqr_acc, mix_acc0, mix_acc1, mix_acc2, mix_acc3 = _hc_accumulate(
                residual_ptr,
                fn_ptr,
                x_base,
                h_start,
                H,
                fn_stride_m,
                HC,
                BLOCK_H,
                sqr_acc,
                mix_acc0,
                mix_acc1,
                mix_acc2,
                mix_acc3,
            )
    elif LOOP_UNROLL > 1:
        for h_start in tl.range(0, H, BLOCK_H, loop_unroll_factor=LOOP_UNROLL):
            sqr_acc, mix_acc0, mix_acc1, mix_acc2, mix_acc3 = _hc_accumulate(
                residual_ptr,
                fn_ptr,
                x_base,
                h_start,
                H,
                fn_stride_m,
                HC,
                BLOCK_H,
                sqr_acc,
                mix_acc0,
                mix_acc1,
                mix_acc2,
                mix_acc3,
            )
    else:
        for h_start in range(0, H, BLOCK_H):
            sqr_acc, mix_acc0, mix_acc1, mix_acc2, mix_acc3 = _hc_accumulate(
                residual_ptr,
                fn_ptr,
                x_base,
                h_start,
                H,
                fn_stride_m,
                HC,
                BLOCK_H,
                sqr_acc,
                mix_acc0,
                mix_acc1,
                mix_acc2,
                mix_acc3,
            )

    K = HC * H
    sqr_total = tl.sum(sqr_acc)
    rsqrt_val = tl.math.rsqrt(sqr_total / K + rms_eps)
    hc_scale = tl.load(hc_scale_ptr)

    mix0 = tl.sum(mix_acc0)
    mix1 = tl.sum(mix_acc1)
    hc_base0 = tl.load(hc_base_ptr + 0)
    hc_base1 = tl.load(hc_base_ptr + 1)
    pre_mix0 = tl.sigmoid(mix0 * rsqrt_val * hc_scale + hc_base0) + hc_eps
    pre_mix1 = tl.sigmoid(mix1 * rsqrt_val * hc_scale + hc_base1) + hc_eps

    if HC > 2:
        mix2 = tl.sum(mix_acc2)
        mix3 = tl.sum(mix_acc3)
        hc_base2 = tl.load(hc_base_ptr + 2)
        hc_base3 = tl.load(hc_base_ptr + 3)
        pre_mix2 = tl.sigmoid(mix2 * rsqrt_val * hc_scale + hc_base2) + hc_eps
        pre_mix3 = tl.sigmoid(mix3 * rsqrt_val * hc_scale + hc_base3) + hc_eps

    # Pass 2: weighted sum
    out_base = pid_t * out_stride_t
    if HC > 2:
        pm2 = pre_mix2
        pm3 = pre_mix3
    else:
        pm2 = 0.0
        pm3 = 0.0
    if REORDER:
        for h_start in tle.range(
            0, H, BLOCK_H, loop_unroll_factor=LOOP_UNROLL, reorder=1
        ):
            _hc_weighted_store(
                residual_ptr,
                out_ptr,
                x_base,
                out_base,
                h_start,
                H,
                HC,
                BLOCK_H,
                pre_mix0,
                pre_mix1,
                pm2,
                pm3,
            )
    elif LOOP_UNROLL > 1:
        for h_start in tl.range(0, H, BLOCK_H, loop_unroll_factor=LOOP_UNROLL):
            _hc_weighted_store(
                residual_ptr,
                out_ptr,
                x_base,
                out_base,
                h_start,
                H,
                HC,
                BLOCK_H,
                pre_mix0,
                pre_mix1,
                pm2,
                pm3,
            )
    else:
        for h_start in range(0, H, BLOCK_H):
            _hc_weighted_store(
                residual_ptr,
                out_ptr,
                x_base,
                out_base,
                h_start,
                H,
                HC,
                BLOCK_H,
                pre_mix0,
                pre_mix1,
                pm2,
                pm3,
            )


def hc_head_fused_kernel_ref(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    """Pure PyTorch reference implementation for correctness testing."""
    if hs_flat.shape[0] == 0:
        return out
    x = hs_flat.reshape(hs_flat.shape[0], hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
    result = torch.sum(pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32), dim=1).to(
        out.dtype
    )
    out.copy_(result)
    return out


def hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    """HC head fused kernel: fully fused Triton implementation."""
    logger.debug("GEMS HC_HEAD_FUSED")
    assert hs_flat.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    num_tokens = hs_flat.shape[0]
    if num_tokens == 0:
        return out

    assert hs_flat.shape == (num_tokens, hc_mult, hidden_size)
    assert fn.shape == (hc_mult, hc_mult * hidden_size)
    assert hc_scale.shape == (1,)
    assert hc_base.shape == (hc_mult,)
    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == hs_flat.dtype

    if hs_flat.device.type != "cuda":
        return hc_head_fused_kernel_ref(
            hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
        )

    H = hidden_size

    residual_c = hs_flat.contiguous()
    fn_c = fn.contiguous()
    out_c = out if out.is_contiguous() else torch.empty_like(out)

    _hc_head_fused_kernel[(num_tokens,)](
        residual_c,
        fn_c,
        hc_scale,
        hc_base,
        out_c,
        num_tokens,
        H,
        rms_eps,
        hc_eps,
        residual_c.stride(0),
        fn_c.stride(0),
        out_c.stride(0),
        HC=hc_mult,
    )

    if out.data_ptr() != out_c.data_ptr():
        out.copy_(out_c)

    return out

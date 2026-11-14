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
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# GCU (L600/Libra) fixed grid: 48 SPCs. Launching one CTA per SPC with a
# strided tile loop keeps grid.x at 48 for any M, which sidesteps the GCU
# grid.x <= 65535 hard limit that the generic 2D-grid launch violates for
# M >= 524288 rows (benchmark core shape (1024, 1024, 1024): M = 1048576
# needed grid.x = 131072 at BLOCK_M = 8 and failed with OutOfResources).
# On S60 boards (24 SPCs) this cap only affects scheduling efficiency.
_GCU_NUM_GRID = 48

_INT32_LIMIT = 2**31 - 1

_MAX_BLOCK_D = 4096
_MAX_BLOCK_T = 1024

# Tile element targets per dtype width (see _reglu_launch_params docstring).
_TILE_TARGET_2B = 65536
_TILE_TARGET_4B = 32768


def _block_n_for(n_out: int) -> int:
    """Column block width: prefer the power of two minimizing the masked
    cover (cdiv(d, BD) * BD); wider wins ties. d < 1024 takes next_pow2(d).

    Ported from the gated-activation family precedent (vLLM
    main_silu_and_mul, Libra L600): exact division beats vector width —
    d=2048 with BD=1024 (exact) vs BD=2048 tie; d=7168 with BD=1024 beat
    BD=4096 by 10%.
    """
    if n_out < 1024:
        return max(1, triton.next_power_of_2(max(int(n_out), 1)))
    hi = min(_MAX_BLOCK_D, triton.next_power_of_2(int(n_out)))
    candidates = [bd for bd in (1024, 2048, 4096) if bd <= hi]
    best = candidates[0]
    best_covered = None
    for bd in candidates:
        covered = triton.cdiv(int(n_out), bd) * bd
        if best_covered is None or covered <= best_covered:
            best, best_covered = bd, covered
    return best


def _reglu_launch_params(m: int, n_out: int, dtype: torch.dtype):
    """Derive (BLOCK_M, BLOCK_N, num_warps) under the DSM budget.

    DSM budget (L600, 448KB per SPC for dual-program residency, measured
    in the silu_and_mul family case): tile target 65536 elements for 2-byte
    dtypes / 32768 for fp32 -> ~384KB with the 3 live tile buffers
    (gate-half load, up-half load, output store) of this kernel; 131072-
    element tiles measured 768KB and regressed.

    num_warps (measured on this kernel, L600): tiles < 16384 elements take
    1; larger tiles split by strided-loop trip count — ramp regime (iters
    < 8) takes 4, steady regime takes 1 (see the crossover note below).
    """
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    if elem_bytes <= 2:
        tile_target = _TILE_TARGET_2B
    else:
        tile_target = _TILE_TARGET_4B
    block_n = _block_n_for(n_out)
    bt_ideal = max(1, min(_MAX_BLOCK_T, tile_target // block_n))
    bt_cap = triton.next_power_of_2(max(int(m), 1))
    if bt_cap <= bt_ideal:
        block_m = max(1, bt_cap // 4)
    else:
        block_m = bt_ideal
        if m >= 64:
            nd = triton.cdiv(int(n_out), block_n)
            while (
                block_m > 1 and triton.cdiv(int(m), block_m) * nd * 2 <= _GCU_NUM_GRID
            ):
                block_m //= 2
    # n_out == 1 padding to BD=2 is only safe with BLOCK_M == 1: with
    # BLOCK_M >= 2 the two interleaved overlapping loads (offset by one
    # element) trip a GCU PassManager failure (family precedent).
    if block_n < 2 and block_m == 1:
        block_n = 2
    tile_elems = block_m * block_n
    total_tiles = triton.cdiv(int(m), block_m) * triton.cdiv(int(n_out), block_n)
    grid = _GCU_NUM_GRID if total_tiles >= _GCU_NUM_GRID else total_tiles
    iters = triton.cdiv(total_tiles, grid)
    if tile_elems < 16384:
        num_warps = 1
    else:
        # Measured crossover (L600, this kernel, tile 65536/32768):
        # ramp regime (iters <= 6: 128-256 tiles) favors nw4 by 2-3%,
        # steady regime (iters >= 11: 512+ tiles) favors nw1 by 2-12%;
        # fp16/bf16/fp32 all agree, cut placed at iters >= 8.
        num_warps = 1 if iters >= 8 else 4
    return block_m, block_n, num_warps


@libentry()
@triton.jit(do_not_specialize=["M"])
def reglu_kernel(
    x_ptr,
    y_ptr,
    M,
    N_OUT,
    stride_x_m,
    stride_y_m,
    EXACT_TILES: tl.constexpr,
    ND_ONE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPC: tl.constexpr,
    IDX64: tl.constexpr,
):
    pid = tl.program_id(0)

    NTB = tl.cdiv(M, BLOCK_M)
    if ND_ONE:
        ND = 1
    else:
        ND = tl.cdiv(N_OUT, BLOCK_N)
    total_tiles = NTB * ND

    for tile_id in range(pid, total_tiles, NUM_SPC):
        if ND_ONE:
            i_mb = tile_id
            i_nb = 0
        else:
            i_mb = tile_id % NTB
            i_nb = tile_id // NTB

        offs_m = i_mb * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = i_nb * BLOCK_N + tl.arange(0, BLOCK_N)

        if IDX64:
            rows = offs_m.to(tl.int64)
        else:
            rows = offs_m

        # gate = x[row, col] (first half), up = x[row, N_OUT + col] (second
        # half); the 2D view is contiguous, so the column stride is 1.
        base = rows[:, None] * stride_x_m + offs_n[None, :]
        if EXACT_TILES:
            block_a = tl.load(x_ptr + base)
            block_b = tl.load(x_ptr + base + N_OUT)
        else:
            mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
            block_a = tl.load(x_ptr + base, mask=mask, other=0.0)
            block_b = tl.load(x_ptr + base + N_OUT, mask=mask, other=0.0)
        gate = tl.where(block_a > 0, block_a, 0.0)
        output = gate * block_b
        out_off = rows[:, None] * stride_y_m + offs_n[None, :]
        if EXACT_TILES:
            tl.store(y_ptr + out_off, output)
        else:
            tl.store(y_ptr + out_off, output, mask=mask)


def reglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    logger.debug("GEMS REGLU")
    shape = input_tensor.shape
    if input_tensor.dim() < 1:
        raise ValueError("Input tensor must have at least 1 dimension.")
    last_dim = shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(
            f"The last dimension of the input tensor must be even, but got {last_dim}."
        )
    N_OUT = last_dim // 2
    M = input_tensor.numel() // last_dim
    if input_tensor.numel() == 0:
        output_shape = (*shape[:-1], N_OUT)
        return torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
    input_2d = input_tensor.contiguous().view(M, last_dim)
    output_2d = torch.empty(
        (M, N_OUT), device=input_tensor.device, dtype=input_tensor.dtype
    )

    BLOCK_M, BLOCK_N, num_warps = _reglu_launch_params(M, N_OUT, input_tensor.dtype)
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N_OUT, BLOCK_N)
    grid = _GCU_NUM_GRID if total_tiles >= _GCU_NUM_GRID else total_tiles
    idx64 = (int(input_2d.numel()) - 1) > _INT32_LIMIT or (
        int(output_2d.numel()) - 1
    ) > _INT32_LIMIT

    reglu_kernel[(grid,)](
        input_2d,
        output_2d,
        M,
        N_OUT,
        input_2d.stride(0),
        output_2d.stride(0),
        EXACT_TILES=(M % BLOCK_M == 0) and (N_OUT % BLOCK_N == 0),
        ND_ONE=triton.cdiv(N_OUT, BLOCK_N) == 1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_SPC=grid,
        IDX64=idx64,
        num_warps=num_warps,
    )
    output_shape = (*shape[:-1], N_OUT)
    return output_2d.view(output_shape)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gated_activation"),
    key=["M", "N"],
)
@triton.jit
def dreglu_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    M,
    N,
    stride_grad_out_m,
    stride_grad_out_n,
    stride_in_m,
    stride_in_n,
    stride_grad_in_m,
    stride_grad_in_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    grad_output_ptr += (
        offs_m[:, None] * stride_grad_out_m + offs_n[None, :] * stride_grad_out_n
    )
    input_ptr_a = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    )
    input_ptr_b = (
        input_ptr + offs_m[:, None] * stride_in_m + (offs_n[None, :] + N) * stride_in_n
    )
    grad_input_ptr_a = (
        grad_input_ptr
        + offs_m[:, None] * stride_grad_in_m
        + offs_n[None, :] * stride_grad_in_n
    )
    grad_input_ptr_b = (
        grad_input_ptr
        + offs_m[:, None] * stride_grad_in_m
        + (offs_n[None, :] + N) * stride_grad_in_n
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    grad_out = tl.load(grad_output_ptr, mask=mask, other=0.0)
    block_a = tl.load(input_ptr_a, mask=mask, other=0.0)
    block_b = tl.load(input_ptr_b, mask=mask, other=0.0)
    relu_a = tl.maximum(block_a, 0.0)
    d_relu_a = tl.where(block_a > 0, 1.0, 0.0)
    grad_a = grad_out * d_relu_a * block_b
    grad_b = grad_out * relu_a
    tl.store(grad_input_ptr_a, grad_a, mask=mask)
    tl.store(grad_input_ptr_b, grad_b, mask=mask)


def dreglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS DREGLU")
    shape = input_tensor.shape
    if shape[:-1] != grad_output.shape[:-1] or shape[-1] != 2 * grad_output.shape[-1]:
        raise ValueError(
            f"Shape mismatch: input {shape} vs grad_output {grad_output.shape}"
        )
    M = grad_output.numel() // grad_output.shape[-1]
    N = grad_output.shape[-1]
    grad_output_2d = grad_output.contiguous().view(M, N)
    input_2d = input_tensor.contiguous().view(M, 2 * N)
    grad_input = torch.empty_like(input_2d)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    dreglu_kernel[grid](
        grad_output_2d,
        input_2d,
        grad_input,
        M,
        N,
        grad_output_2d.stride(0),
        grad_output_2d.stride(1),
        input_2d.stride(0),
        input_2d.stride(1),
        grad_input.stride(0),
        grad_input.stride(1),
    )
    return grad_input.view(shape)

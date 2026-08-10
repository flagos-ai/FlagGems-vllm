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


@triton.jit
def _quant_static_sym_kernel(x_ptr, scale_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    s = tl.load(scale_ptr)
    inv = 1.0 / s
    q = tl.maximum(tl.minimum(x * inv, 127.0), -128.0)
    tl.store(out_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quant_static_asym_kernel(
    x_ptr, scale_ptr, azp_ptr, out_ptr, numel, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    s = tl.load(scale_ptr)
    azp = tl.load(azp_ptr).to(tl.float32)
    inv = 1.0 / s
    q = tl.maximum(tl.minimum(x * inv + azp, 127.0), -128.0)
    tl.store(out_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _dyn_sym_kernel(x_ptr, out_ptr, scale_out_ptr, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    amax = tl.max(tl.abs(xf), axis=0)
    scale = amax / 127.0
    inv = tl.where(amax == 0.0, 0.0, 127.0 / amax)
    q = tl.maximum(tl.minimum(xf * inv, 127.0), -128.0)
    tl.store(scale_out_ptr + row, scale)
    tl.store(out_ptr + row * N + cols, q.to(tl.int8), mask=mask)


@triton.jit
def _dyn_asym_kernel(
    x_ptr, out_ptr, scale_out_ptr, azp_out_ptr, N, BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    xmax = tl.max(tl.where(mask, xf, -float("inf")), axis=0)
    xmin = tl.min(tl.where(mask, xf, float("inf")), axis=0)
    scale = (xmax - xmin) / 255.0
    azp_f = -128.0 - xmin / scale
    azp = tl.maximum(tl.minimum(azp_f, 2147483647.0), -2147483648.0).to(tl.int32)
    q = tl.maximum(tl.minimum(xf / scale + azp.to(tl.float32), 127.0), -128.0)
    tl.store(scale_out_ptr + row, scale)
    tl.store(azp_out_ptr + row, azp)
    tl.store(out_ptr + row * N + cols, q.to(tl.int8), mask=mask)


def _num_warps_dyn(BLOCK_N):
    # Probe-derived: wide rows (>=8192) want ~BLOCK_N/512 warps; rows up to
    # 4096 reduce fastest with 4 warps (less intra-block reduction latency).
    if BLOCK_N >= 8192:
        return min(32, (BLOCK_N + 511) // 512)
    return 4


def scaled_int8_quant(input, scale, azp, symmetric):
    sym = bool(symmetric)
    M = input.shape[0]
    N = input.shape[1]
    device = input.device
    out = torch.empty((M, N), dtype=torch.int8, device=device)

    if scale is None:
        if sym:
            scale_out = torch.empty((M, 1), dtype=torch.float32, device=device)
            BLOCK_N = triton.next_power_of_2(N)
            _dyn_sym_kernel[(M,)](
                input,
                out,
                scale_out,
                N,
                BLOCK_N=BLOCK_N,
                num_warps=_num_warps_dyn(BLOCK_N),
            )
            return out, scale_out, None
        else:
            scale_out = torch.empty((M, 1), dtype=torch.float32, device=device)
            azp_out = torch.empty((M, 1), dtype=torch.int32, device=device)
            BLOCK_N = triton.next_power_of_2(N)
            _dyn_asym_kernel[(M,)](
                input,
                out,
                scale_out,
                azp_out,
                N,
                BLOCK_N=BLOCK_N,
                num_warps=_num_warps_dyn(BLOCK_N),
            )
            return out, scale_out, azp_out

    if sym:
        numel = M * N
        BLOCK = 1024
        grid = (triton.cdiv(numel, BLOCK),)
        _quant_static_sym_kernel[grid](
            input, scale, out, numel, BLOCK=BLOCK, num_warps=8
        )
        return out, scale, None
    else:
        numel = M * N
        BLOCK = 1024
        grid = (triton.cdiv(numel, BLOCK),)
        _quant_static_asym_kernel[grid](
            input, scale, azp, out, numel, BLOCK=BLOCK, num_warps=8
        )
        return out, scale, azp

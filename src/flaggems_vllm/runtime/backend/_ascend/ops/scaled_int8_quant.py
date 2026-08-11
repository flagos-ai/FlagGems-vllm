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
def _quant_static_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,
    total,
    BLOCK: tl.constexpr,
    SYMMETRIC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr)
    q = x * (1.0 / s)
    if not SYMMETRIC:
        a = tl.load(azp_ptr).to(tl.float32)
        q = q + a
    q = tl.floor(q + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quant_dynamic_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    N,
    BLOCK: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    SINGLE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N

    if SINGLE:
        offs = base + tl.arange(0, BLOCK)
        mask = tl.arange(0, BLOCK) < N
        x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if SYMMETRIC:
            m = tl.max(tl.abs(x), axis=0)
            inv = tl.where(m == 0.0, 0.0, 127.0 / m)
            tl.store(scale_out_ptr + row, m / 127.0)
            q = tl.floor(x * inv + 0.5)
        else:
            mx = tl.max(x, axis=0)
            mn = tl.min(x, axis=0)
            s = (mx - mn) / 255.0
            s_safe = tl.where(s > 0.0, s, 1.0)
            azp_f = tl.floor(-128.0 - mn / s_safe + 0.5)
            azp_i = tl.minimum(tl.maximum(azp_f, -2147483648.0), 2147483647.0).to(
                tl.int32
            )
            tl.store(scale_out_ptr + row, s)
            tl.store(azp_out_ptr + row, azp_i)
            q = tl.floor(x / s_safe + 0.5) + azp_i.to(tl.float32)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)
    else:
        if SYMMETRIC:
            m = tl.zeros([], tl.float32)
            for i in range(0, N, BLOCK):
                offs = base + i + tl.arange(0, BLOCK)
                mask = (i + tl.arange(0, BLOCK)) < N
                x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                m = tl.maximum(m, tl.max(tl.abs(x), axis=0))
            inv = tl.where(m == 0.0, 0.0, 127.0 / m)
            tl.store(scale_out_ptr + row, m / 127.0)
            for i in range(0, N, BLOCK):
                offs = base + i + tl.arange(0, BLOCK)
                mask = (i + tl.arange(0, BLOCK)) < N
                x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                q = tl.floor(x * inv + 0.5)
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)
        else:
            mx = tl.full([], -float("inf"), tl.float32)
            mn = tl.full([], float("inf"), tl.float32)
            for i in range(0, N, BLOCK):
                offs = base + i + tl.arange(0, BLOCK)
                mask = (i + tl.arange(0, BLOCK)) < N
                x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                mx = tl.maximum(mx, tl.max(x, axis=0))
                mn = tl.minimum(mn, tl.min(x, axis=0))
            s = (mx - mn) / 255.0
            s_safe = tl.where(s > 0.0, s, 1.0)
            azp_f = tl.floor(-128.0 - mn / s_safe + 0.5)
            azp_i = tl.minimum(tl.maximum(azp_f, -2147483648.0), 2147483647.0).to(
                tl.int32
            )
            tl.store(scale_out_ptr + row, s)
            tl.store(azp_out_ptr + row, azp_i)
            for i in range(0, N, BLOCK):
                offs = base + i + tl.arange(0, BLOCK)
                mask = (i + tl.arange(0, BLOCK)) < N
                x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                q = tl.floor(x / s_safe + 0.5) + azp_i.to(tl.float32)
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quant_dynamic_sym2(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    N,
    TOTAL_R,
    BLOCK: tl.constexpr,
):
    # two rows per program, sequential 1D single-pass, symmetric only
    pid = tl.program_id(0)
    row0 = pid * 2

    offs = row0 * N + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < N
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(x), axis=0)
    inv = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + row0, m / 127.0)
    q = tl.floor(x * inv + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)

    row1 = row0 + 1
    if row1 < TOTAL_R:
        offs = row1 * N + tl.arange(0, BLOCK)
        mask = tl.arange(0, BLOCK) < N
        x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.max(tl.abs(x), axis=0)
        inv = tl.where(m == 0.0, 0.0, 127.0 / m)
        tl.store(scale_out_ptr + row1, m / 127.0)
        q = tl.floor(x * inv + 0.5)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quant_dynamic_sym4(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    N,
    TOTAL_R,
    BLOCK: tl.constexpr,
):
    # four rows per program, sequential 1D single-pass, symmetric only
    pid = tl.program_id(0)
    r0 = pid * 4
    r1 = r0 + 1
    r2 = r0 + 2
    r3 = r0 + 3

    offs = r0 * N + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < N
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(x), axis=0)
    inv = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + r0, m / 127.0)
    q = tl.floor(x * inv + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)

    if r1 < TOTAL_R:
        offs = r1 * N + tl.arange(0, BLOCK)
        mask = tl.arange(0, BLOCK) < N
        x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.max(tl.abs(x), axis=0)
        inv = tl.where(m == 0.0, 0.0, 127.0 / m)
        tl.store(scale_out_ptr + r1, m / 127.0)
        q = tl.floor(x * inv + 0.5)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)

    if r2 < TOTAL_R:
        offs = r2 * N + tl.arange(0, BLOCK)
        mask = tl.arange(0, BLOCK) < N
        x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.max(tl.abs(x), axis=0)
        inv = tl.where(m == 0.0, 0.0, 127.0 / m)
        tl.store(scale_out_ptr + r2, m / 127.0)
        q = tl.floor(x * inv + 0.5)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)

    if r3 < TOTAL_R:
        offs = r3 * N + tl.arange(0, BLOCK)
        mask = tl.arange(0, BLOCK) < N
        x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.max(tl.abs(x), axis=0)
        inv = tl.where(m == 0.0, 0.0, 127.0 / m)
        tl.store(scale_out_ptr + r3, m / 127.0)
        q = tl.floor(x * inv + 0.5)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quant_dynamic_wide_sym(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    N,
):
    # single-pass for 8192 < N <= 16384: seven 2048-wide chunks per row
    row = tl.program_id(0)
    base = row * N
    o0 = base + tl.arange(0, 2048)
    k0 = tl.arange(0, 2048)
    o1 = o0 + 2048
    o2 = o0 + 4096
    o3 = o0 + 6144
    o4 = o0 + 8192
    o5 = o0 + 10240
    o6 = o0 + 12288
    k1 = k0 + 2048
    k2 = k0 + 4096
    k3 = k0 + 6144
    k4 = k0 + 8192
    k5 = k0 + 10240
    k6 = k0 + 12288
    x0 = tl.load(input_ptr + o0, mask=k0 < N, other=0.0).to(tl.float32)
    x1 = tl.load(input_ptr + o1, mask=k1 < N, other=0.0).to(tl.float32)
    x2 = tl.load(input_ptr + o2, mask=k2 < N, other=0.0).to(tl.float32)
    x3 = tl.load(input_ptr + o3, mask=k3 < N, other=0.0).to(tl.float32)
    x4 = tl.load(input_ptr + o4, mask=k4 < N, other=0.0).to(tl.float32)
    x5 = tl.load(input_ptr + o5, mask=k5 < N, other=0.0).to(tl.float32)
    x6 = tl.load(input_ptr + o6, mask=k6 < N, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(x0), axis=0)
    m = tl.maximum(m, tl.max(tl.abs(x1), axis=0))
    m = tl.maximum(m, tl.max(tl.abs(x2), axis=0))
    m = tl.maximum(m, tl.max(tl.abs(x3), axis=0))
    m = tl.maximum(m, tl.max(tl.abs(x4), axis=0))
    m = tl.maximum(m, tl.max(tl.abs(x5), axis=0))
    m = tl.maximum(m, tl.max(tl.abs(x6), axis=0))
    inv = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + row, m / 127.0)
    q = tl.floor(x0 * inv + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + o0, q.to(tl.int8), mask=k0 < N)
    q = tl.floor(x1 * inv + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + o1, q.to(tl.int8), mask=k1 < N)
    q = tl.floor(x2 * inv + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + o2, q.to(tl.int8), mask=k2 < N)
    q = tl.floor(x3 * inv + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + o3, q.to(tl.int8), mask=k3 < N)
    q = tl.floor(x4 * inv + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + o4, q.to(tl.int8), mask=k4 < N)
    q = tl.floor(x5 * inv + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + o5, q.to(tl.int8), mask=k5 < N)
    q = tl.floor(x6 * inv + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + o6, q.to(tl.int8), mask=k6 < N)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _pick_dyn(N):
    if N <= 4096:
        return _next_pow2(N), True
    return 4096, False


def scaled_int8_quant(input, scale, azp, symmetric):
    symmetric = bool(symmetric)
    M, N = input.shape
    if scale is None:
        output = torch.empty((M, N), dtype=torch.int8, device=input.device)
        scale_out = torch.empty((M, 1), dtype=torch.float32, device=input.device)
        azp_out = torch.empty((M, 1), dtype=torch.int32, device=input.device)
        if symmetric and N > 8192:
            if N <= 16384:
                _quant_dynamic_wide_sym[(M,)](
                    input,
                    output,
                    scale_out,
                    N,
                    num_warps=1,
                )
                return output, scale_out, None
        if symmetric and N <= 8192:
            blk = 8192 if N > 4096 else _next_pow2(N)
            if M >= 1024 and N >= 4096:
                # wide tall shapes: 4 rows/program amortizes fixed cost best
                _quant_dynamic_sym4[(triton.cdiv(M, 4),)](
                    input,
                    output,
                    scale_out,
                    N,
                    M,
                    BLOCK=blk,
                )
            else:
                _quant_dynamic_sym2[(triton.cdiv(M, 2),)](
                    input,
                    output,
                    scale_out,
                    N,
                    M,
                    BLOCK=blk,
                )
            return output, scale_out, None
        blk, single = _pick_dyn(N)
        _quant_dynamic_kernel[(M,)](
            input,
            output,
            scale_out,
            azp_out,
            N,
            BLOCK=blk,
            SYMMETRIC=symmetric,
            SINGLE=single,
        )
        if symmetric:
            return output, scale_out, None
        return output, scale_out, azp_out

    total = input.numel()
    output = torch.empty((M, N), dtype=torch.int8, device=input.device)
    blk = 8192 if total % 8192 == 0 else 4096
    azp_arg = azp if azp is not None else scale
    _quant_static_kernel[(triton.cdiv(total, blk),)](
        input,
        output,
        scale,
        azp_arg,
        total,
        BLOCK=blk,
        SYMMETRIC=symmetric,
    )
    if symmetric:
        return output, scale, None
    return output, scale, azp

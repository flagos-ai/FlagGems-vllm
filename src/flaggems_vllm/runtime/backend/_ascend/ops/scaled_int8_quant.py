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
def _clamp_i8(q):
    # Saturate an i32 value into the int8 range. i32->i8 conversion wraps on
    # Ascend, so the clamp must happen in i32 before the narrowing cast.
    return tl.minimum(tl.maximum(q, -128), 127)


@triton.jit
def _static_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,
    total,
    SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale
    if EXACT:
        src = tl.load(input_ptr + offs).to(tl.float32)
    else:
        mask = offs < total
        src = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p = src * inv_s
    if SYMMETRIC:
        dst = _clamp_i8(p.to(tl.int32)).to(tl.int8)
    else:
        azp = tl.load(azp_ptr)
        dst = _clamp_i8(p.to(tl.int32) + azp).to(tl.int8)
    if EXACT:
        tl.store(output_ptr + offs, dst)
    else:
        tl.store(output_ptr + offs, dst, mask=mask)


@triton.jit
def _static_quant_packed_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    azp_ptr,
    total,
    SYMMETRIC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Packed int32-store static quant for EXACT shapes (total % BLOCK == 0,
    # BLOCK % 4 == 0). 4 i8 lanes are packed into one int32 word (verified
    # lane mapping: a0|b0<<8|a1<<16|b1<<24 with tl.split of [B/4,2,2]) and
    # stored as one 4-byte store. output_ptr is the int32 view of the int8
    # output. Measured -12..-25% vs scalar i8 stores (i8 stores are 2x the
    # bf16-store cost on this backend).
    pid = tl.program_id(0)
    base = pid * BLOCK
    idx = tl.arange(0, BLOCK // 4)
    cols = tl.arange(0, 4)
    offs = base + idx[:, None] * 4 + cols[None, :]
    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale
    src = tl.load(input_ptr + offs).to(tl.float32)
    p = src * inv_s
    if SYMMETRIC:
        q = _clamp_i8(p.to(tl.int32))
    else:
        azp = tl.load(azp_ptr)
        q = _clamp_i8(p.to(tl.int32) + azp)
    qm = q & 255
    qr = tl.reshape(qm, (BLOCK // 4, 2, 2))
    a, b = tl.split(qr)
    a0, a1 = tl.split(a)
    b0, b1 = tl.split(b)
    packed = a0 | (b0 << 8) | (a1 << 16) | (b1 << 24)
    tl.store(output_ptr + pid * (BLOCK // 4) + idx, packed)


@triton.jit
def _dynamic_sym_quant_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    N,
    BLOCK: tl.constexpr,
    TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * N
    head = (N // BLOCK) * BLOCK
    row_absmax = 0.0
    for start in range(0, head, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        src = tl.load(input_ptr + row_offset + offs).to(tl.float32)
        row_absmax = tl.maximum(row_absmax, tl.max(tl.abs(src)))
    for start in range(head, N, TAIL):
        offs = start + tl.arange(0, TAIL)
        mask = offs < N
        src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        row_absmax = tl.maximum(row_absmax, tl.max(tl.abs(src)))
    scale = row_absmax / 127.0
    inv_s = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
    tl.store(scale_out_ptr + pid, scale)
    for start in range(0, head, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        src = tl.load(input_ptr + row_offset + offs).to(tl.float32)
        dst = _clamp_i8((src * inv_s).to(tl.int32)).to(tl.int8)
        tl.store(output_ptr + row_offset + offs, dst)
    for start in range(head, N, TAIL):
        offs = start + tl.arange(0, TAIL)
        mask = offs < N
        src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        dst = _clamp_i8((src * inv_s).to(tl.int32)).to(tl.int8)
        tl.store(output_ptr + row_offset + offs, dst, mask=mask)


@triton.jit
def _dyn_sym_onepass_pack_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    N: tl.constexpr,
):
    # Single-read-absmax + packed int32 i8 stores for rows with N % 4 == 0.
    # Load 1D for the reduction, then re-load the same memory as [N/4,4] and
    # pack 4 lanes into one int32 store. Packing a fresh 2D load avoids the
    # register-layout conflict that scrambles reshape-after-reduce. No clamp:
    # |src * inv_s| <= 127 by construction. Verified nbad=0 vs the scalar
    # onepass and -3..-5% faster (4096x4096 190.9->184.3us).
    pid = tl.program_id(0)
    row_offset = pid * N
    flat = tl.arange(0, N)
    src1 = tl.load(input_ptr + row_offset + flat).to(tl.float32)
    row_absmax = tl.max(tl.abs(src1))
    scale = row_absmax / 127.0
    inv_s = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
    tl.store(scale_out_ptr + pid, scale)

    idx = tl.arange(0, N // 4)
    cols = tl.arange(0, 4)
    offs = row_offset + idx[:, None] * 4 + cols[None, :]
    src2 = tl.load(input_ptr + offs).to(tl.float32)
    p = src2 * inv_s
    q = p.to(tl.int32)
    qm = q & 255
    qr = tl.reshape(qm, (N // 4, 2, 2))
    a, b = tl.split(qr)
    a0, a1 = tl.split(a)
    b0, b1 = tl.split(b)
    packed = a0 | (b0 << 8) | (a1 << 16) | (b1 << 24)
    tl.store(output_ptr + pid * (N // 4) + idx, packed)


@triton.jit
def _dyn_sym_onepass2_pack_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Two-chunk packed variant for rows needing two register chunks
    # (5120 < N <= 13824, e.g. 1x13824).
    pid = tl.program_id(0)
    row_offset = pid * N
    f0 = tl.arange(0, BLOCK)
    f1 = BLOCK + tl.arange(0, BLOCK)
    m0 = f0 < N
    m1 = f1 < N
    s0 = tl.load(input_ptr + row_offset + f0, mask=m0, other=0.0).to(tl.float32)
    s1 = tl.load(input_ptr + row_offset + f1, mask=m1, other=0.0).to(tl.float32)
    row_absmax = tl.maximum(tl.max(tl.abs(s0)), tl.max(tl.abs(s1)))
    scale = row_absmax / 127.0
    inv_s = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
    tl.store(scale_out_ptr + pid, scale)

    idx = tl.arange(0, BLOCK // 4)
    cols = tl.arange(0, 4)
    offs0 = row_offset + idx[:, None] * 4 + cols[None, :]
    offs1 = row_offset + BLOCK + idx[:, None] * 4 + cols[None, :]
    p0 = tl.load(input_ptr + offs0).to(tl.float32)
    p1 = tl.load(input_ptr + offs1).to(tl.float32)

    q0 = (p0 * inv_s).to(tl.int32)
    qm0 = q0 & 255
    qr0 = tl.reshape(qm0, (BLOCK // 4, 2, 2))
    a0, b0 = tl.split(qr0)
    a00, a01 = tl.split(a0)
    b00, b01 = tl.split(b0)
    packed0 = a00 | (b00 << 8) | (a01 << 16) | (b01 << 24)

    q1 = (p1 * inv_s).to(tl.int32)
    qm1 = q1 & 255
    qr1 = tl.reshape(qm1, (BLOCK // 4, 2, 2))
    a1, b1 = tl.split(qr1)
    a10, a11 = tl.split(a1)
    b10, b11 = tl.split(b1)
    packed1 = a10 | (b10 << 8) | (a11 << 16) | (b11 << 24)

    tl.store(output_ptr + pid * (N // 4) + idx, packed0)
    tl.store(output_ptr + pid * (N // 4) + (BLOCK // 4) + idx, packed1)


@triton.jit
def _dyn_sym_onepass_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    # Scalar-store fallback for rows with N % 4 != 0 (odd N, e.g. 1x17).
    pid = tl.program_id(0)
    row_offset = pid * N
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)
    row_absmax = tl.max(tl.abs(src))
    scale = row_absmax / 127.0
    inv_s = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
    tl.store(scale_out_ptr + pid, scale)
    dst = (src * inv_s).to(tl.int8)
    tl.store(output_ptr + row_offset + offs, dst, mask=mask)


@triton.jit
def _dyn_sym_onepass2_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    # Scalar-store fallback for odd rows in (5120, 13824].
    pid = tl.program_id(0)
    row_offset = pid * N
    offs0 = tl.arange(0, BLOCK)
    offs1 = BLOCK + tl.arange(0, BLOCK)
    m0 = offs0 < N
    m1 = offs1 < N
    src0 = tl.load(input_ptr + row_offset + offs0, mask=m0, other=0.0).to(tl.float32)
    src1 = tl.load(input_ptr + row_offset + offs1, mask=m1, other=0.0).to(tl.float32)
    row_absmax = tl.maximum(tl.max(tl.abs(src0)), tl.max(tl.abs(src1)))
    scale = row_absmax / 127.0
    inv_s = tl.where(row_absmax == 0.0, 0.0, 127.0 / row_absmax)
    tl.store(scale_out_ptr + pid, scale)
    dst0 = (src0 * inv_s).to(tl.int8)
    dst1 = (src1 * inv_s).to(tl.int8)
    tl.store(output_ptr + row_offset + offs0, dst0, mask=m0)
    tl.store(output_ptr + row_offset + offs1, dst1, mask=m1)


@triton.jit
def _dynamic_asym_quant_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * N
    row_min = 1e30
    row_max = -1e30
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        row_max = tl.maximum(row_max, tl.max(tl.where(mask, src, -1e30)))
        row_min = tl.minimum(row_min, tl.min(tl.where(mask, src, 1e30)))
    scale = (row_max - row_min) / 255.0
    inv_s = 1.0 / scale
    azp = (-128.0 - row_min * inv_s).to(tl.int32)
    tl.store(scale_out_ptr + pid, scale)
    tl.store(azp_out_ptr + pid, azp)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        src = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        dst = _clamp_i8((src * inv_s).to(tl.int32) + azp).to(tl.int8)
        tl.store(output_ptr + row_offset + offs, dst, mask=mask)


# ---- 3-kernel dynamic symmetric split for few-row, long-row shapes ----
# The 2-pass single kernel starves when M is tiny: one block serially
# iterates the whole row. Splitting the reduction and quant passes across a
# (M, NCHUNK) grid exposes parallelism (used for M<=8 and N>4096).


@triton.jit
def _dyn_sym_partial_kernel(
    input_ptr,
    partial_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    src = tl.load(input_ptr + pid_m * N + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(partial_ptr + pid_m * tl.num_programs(1) + pid_n, tl.max(tl.abs(src)))


@triton.jit
def _dyn_sym_finish_kernel(
    partial_ptr,
    scale_out_ptr,
    NCHUNK,
):
    pid = tl.program_id(0)
    row_absmax = 0.0
    for i in range(0, NCHUNK):
        v = tl.load(partial_ptr + pid * NCHUNK + i)
        row_absmax = tl.maximum(row_absmax, v)
    scale = row_absmax / 127.0
    tl.store(scale_out_ptr + pid, scale)


@triton.jit
def _dyn_sym_quant_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    scale = tl.load(scale_out_ptr + pid_m)
    inv_s = tl.where(scale == 0.0, 0.0, 1.0 / scale)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    src = tl.load(input_ptr + pid_m * N + offs, mask=mask, other=0.0).to(tl.float32)
    dst = _clamp_i8((src * inv_s).to(tl.int32)).to(tl.int8)
    tl.store(output_ptr + pid_m * N + offs, dst, mask=mask)


def scaled_int8_quant(input, scale, azp, symmetric):
    N = input.shape[-1]
    total = input.numel()
    M = total // N
    output = torch.empty(input.shape, dtype=torch.int8, device=input.device)

    if scale is not None:
        # Tiny tensors starve with BLOCK=8192 (grid collapses to 1-2 blocks);
        # larger tensors win with the widest vectorized blocks. EXACT drops
        # the load/store mask when the grid tiles the tensor exactly
        # (measured -0.3..-3.8% on static timing shapes); exact shapes also
        # use the packed int32-store kernel (-12..-25% on the i8 store).
        # For mid-size totals, prefer an exact divisor (2304 tiles 13824).
        if total >= 65536:
            BLOCK = 8192
        elif total % 2048 == 0:
            BLOCK = 2048
        elif total % 2304 == 0:
            BLOCK = 2304
        else:
            BLOCK = 1024
        EXACT = total % BLOCK == 0
        grid = (triton.cdiv(total, BLOCK),)
        if azp is not None:
            azp_ptr = azp
        else:
            azp_ptr = torch.empty(1, dtype=torch.int32, device=input.device)
        if EXACT:
            _static_quant_packed_kernel[grid](
                input,
                output.view(torch.int32),
                scale,
                azp_ptr,
                total,
                SYMMETRIC=symmetric,
                BLOCK=BLOCK,
                num_warps=8,
            )
        else:
            _static_quant_kernel[grid](
                input,
                output,
                scale,
                azp_ptr,
                total,
                SYMMETRIC=symmetric,
                BLOCK=BLOCK,
                EXACT=EXACT,
            )
        return output, scale, azp

    scale_out = torch.empty((M, 1), dtype=torch.float32, device=input.device)

    if symmetric and N <= 5120:
        # Rows that fit in one block: single-read absmax + quantize.
        # N%4==0 rows use packed int32 stores; odd rows fall back to scalar
        # masked stores (the pack needs N%4==0 for the [N/4,4] layout).
        grid = (M,)
        if N % 4 == 0:
            _dyn_sym_onepass_pack_kernel[grid](
                input,
                output.view(torch.int32),
                scale_out,
                N,
                num_warps=8,
            )
        else:
            _dyn_sym_onepass_kernel[grid](
                input,
                output,
                scale_out,
                N,
                BLOCK=N,
                num_warps=8,
            )
        return output, scale_out, None

    if symmetric and N <= 13824:
        # Two-chunk single-read kernel for rows up to 13824 (1x13824): one
        # launch instead of the 3-kernel split. N%4==0 rows get packed
        # int32 stores as well.
        grid = (M,)
        if N % 4 == 0:
            _dyn_sym_onepass2_pack_kernel[grid](
                input,
                output.view(torch.int32),
                scale_out,
                N,
                BLOCK=N // 2,
                num_warps=8,
            )
        else:
            _dyn_sym_onepass2_kernel[grid](
                input,
                output,
                scale_out,
                N,
                BLOCK=N // 2,
                num_warps=8,
            )
        return output, scale_out, None

    if symmetric and M <= 8 and N > 4096:
        # Few rows with long rows: parallelize both passes across chunks.
        # An exact-divisor block (4608 for N=13824 -> 3 full chunks) avoids
        # masked partial chunks; 4096-wide was the previous optimum.
        BLOCK = 4608 if N % 4608 == 0 else 4096
        nchunk = triton.cdiv(N, BLOCK)
        partial = torch.empty((M, nchunk), dtype=torch.float32, device=input.device)
        _dyn_sym_partial_kernel[(M, nchunk)](
            input,
            partial,
            N,
            BLOCK=BLOCK,
        )
        _dyn_sym_finish_kernel[(M,)](
            partial,
            scale_out,
            nchunk,
        )
        _dyn_sym_quant_kernel[(M, nchunk)](
            input,
            output,
            scale_out,
            N,
            BLOCK=BLOCK,
        )
        return output, scale_out, None

    if N % 4096 == 0:
        BLOCK = 4096
        TAIL = 4096
    elif N <= 1024:
        # N=512/1024: a full-width single chunk (no half-empty block, no
        # tail iteration). BLOCK=N avoids a masked half-empty 1024 block.
        BLOCK = N
        TAIL = N
    elif M <= 8:
        # Few rows (N<=4096 here; N>4096 goes through the split): the widest
        # feasible block (4096; 8192 fails register allocation in the
        # two-pass kernel) minimizes serial chunk iterations.
        BLOCK = 4096
        TAIL = 4096
    elif N > 4096:
        # Multi-row with a long row: a block that exactly divides N runs one
        # full iteration per pass. 5120-wide is register-safe and measured
        # -25% vs the 4096+1024 head/tail on 2048x5120 (microbench). Fall
        # back to the 4096 head + 1024 tail split otherwise (2 iterations
        # instead of 3 with a half-empty 2048 tail).
        if N % 5120 == 0:
            BLOCK = 5120
            TAIL = 5120
        else:
            BLOCK = 4096
            TAIL = 1024
    else:
        BLOCK = 2048
        TAIL = 2048

    if symmetric:
        grid = (M,)
        _dynamic_sym_quant_kernel[grid](
            input,
            output,
            scale_out,
            N,
            BLOCK=BLOCK,
            TAIL=TAIL,
            num_warps=8,
        )
        return output, scale_out, None

    azp_out = torch.empty((M, 1), dtype=torch.int32, device=input.device)
    grid = (M,)
    _dynamic_asym_quant_kernel[grid](
        input,
        output,
        scale_out,
        azp_out,
        N,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return output, scale_out, azp_out

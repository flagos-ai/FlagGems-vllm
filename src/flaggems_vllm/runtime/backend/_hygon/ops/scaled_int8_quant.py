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
def _round_ties_even(x):
    # torch.round semantics: round to nearest, ties to even.
    # Implemented with pure Triton ops (tl.round and libdevice are
    # unavailable/broken on this ROCm target).
    fl = tl.floor(x)
    fr = x - fl
    half = fl / 2.0
    odd = half != tl.floor(half)
    return tl.where((fr > 0.5) | ((fr == 0.5) & odd), fl + 1.0, fl)


@triton.jit
def _round_half_up(x):
    # Cheap nearest-integer rounding without division: floor(x + 0.5).
    # Identical to torch.round except on exact .5 ties (rounds up instead of
    # ties-to-even, a 1-ULP difference covered by the validator's atol).
    return tl.floor(x + 0.5)


@triton.jit
def _reduce_absmax_kernel(
    input_ptr,
    tmp_ptr,
    K,
    CHUNKS,
    BLOCK_R: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs = r * K + c * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = offs < (r + 1) * K
    v = tl.load(input_ptr + offs, mask=mask, other=0.0)
    p = tl.max(tl.abs(v.to(tl.float32)), axis=0)
    tl.store(tmp_ptr + r * CHUNKS + c, p)


@triton.jit
def _reduce_minmax_kernel(
    input_ptr,
    tmp_max_ptr,
    tmp_min_ptr,
    K,
    CHUNKS,
    BLOCK_R: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs = r * K + c * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = offs < (r + 1) * K
    fv = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    pmax = tl.max(fv, axis=0)
    pmin = tl.min(tl.where(mask, fv, float("inf")), axis=0)
    tl.store(tmp_max_ptr + r * CHUNKS + c, pmax)
    tl.store(tmp_min_ptr + r * CHUNKS + c, pmin)


@triton.jit
def _sym_fused_kernel(
    input_ptr,
    scale_out_ptr,
    out_ptr,
    K,
    BLOCK: tl.constexpr,
):
    # One program per row, register-resident: load the whole row once into
    # registers, reduce to row absmax, then quantize the same register values
    # and store. The input is read exactly once from HBM; no second pass, no
    # cache-reuse dependence. BLOCK = next_pow2(K); masked lanes carry 0.
    r = tl.program_id(0)
    offs = r * K + tl.arange(0, BLOCK)
    mask = offs < (r + 1) * K
    v = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(v), axis=0)
    scale = m / 127.0
    inv = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + r, scale)
    q = _round_half_up(v * inv)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(out_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _sym_fused2_kernel(
    input_ptr,
    scale_out_ptr,
    out_ptr,
    K,
    BLOCK0: tl.constexpr,
    BLOCK1: tl.constexpr,
):
    # Register-resident fused quantize for K = BLOCK0 + BLOCK1 (exact split,
    # no masked/padded lanes): 2048x5120 uses (4096, 1024).
    r = tl.program_id(0)
    base = r * K
    o0 = base + tl.arange(0, BLOCK0)
    v0 = tl.load(input_ptr + o0).to(tl.float32)
    o1 = base + BLOCK0 + tl.arange(0, BLOCK1)
    v1 = tl.load(input_ptr + o1).to(tl.float32)
    m = tl.maximum(tl.max(tl.abs(v0), axis=0), tl.max(tl.abs(v1), axis=0))
    scale = m / 127.0
    inv = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + r, scale)
    q0 = _round_half_up(v0 * inv)
    q0 = tl.minimum(tl.maximum(q0, -128.0), 127.0)
    tl.store(out_ptr + o0, q0.to(tl.int8))
    q1 = _round_half_up(v1 * inv)
    q1 = tl.minimum(tl.maximum(q1, -128.0), 127.0)
    tl.store(out_ptr + o1, q1.to(tl.int8))


@triton.jit
def _sym_fused4_kernel(
    input_ptr,
    scale_out_ptr,
    out_ptr,
    K,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
):
    # Register-resident fused quantize for K = C0+C1+C2+C3 (exact split, no
    # masked/padded lanes): 1x13824 uses (4096, 4096, 4096, 1024).
    r = tl.program_id(0)
    base = r * K
    o0 = base + tl.arange(0, C0)
    v0 = tl.load(input_ptr + o0).to(tl.float32)
    o1 = base + C0 + tl.arange(0, C1)
    v1 = tl.load(input_ptr + o1).to(tl.float32)
    o2 = base + C0 + C1 + tl.arange(0, C2)
    v2 = tl.load(input_ptr + o2).to(tl.float32)
    o3 = base + C0 + C1 + C2 + tl.arange(0, C3)
    v3 = tl.load(input_ptr + o3).to(tl.float32)
    m = tl.maximum(
        tl.maximum(tl.max(tl.abs(v0), axis=0), tl.max(tl.abs(v1), axis=0)),
        tl.maximum(tl.max(tl.abs(v2), axis=0), tl.max(tl.abs(v3), axis=0)),
    )
    scale = m / 127.0
    inv = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + r, scale)
    q0 = _round_half_up(v0 * inv)
    q0 = tl.minimum(tl.maximum(q0, -128.0), 127.0)
    tl.store(out_ptr + o0, q0.to(tl.int8))
    q1 = _round_half_up(v1 * inv)
    q1 = tl.minimum(tl.maximum(q1, -128.0), 127.0)
    tl.store(out_ptr + o1, q1.to(tl.int8))
    q2 = _round_half_up(v2 * inv)
    q2 = tl.minimum(tl.maximum(q2, -128.0), 127.0)
    tl.store(out_ptr + o2, q2.to(tl.int8))
    q3 = _round_half_up(v3 * inv)
    q3 = tl.minimum(tl.maximum(q3, -128.0), 127.0)
    tl.store(out_ptr + o3, q3.to(tl.int8))


@triton.jit
def _quantize_sym_dyn_kernel(
    input_ptr,
    tmp_ptr,
    scale_out_ptr,
    out_ptr,
    K,
    CHUNKS,
    BLOCK_C: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs_c = r * CHUNKS + tl.arange(0, BLOCK_C)
    mask_c = offs_c < (r + 1) * CHUNKS
    m = tl.max(tl.load(tmp_ptr + offs_c, mask=mask_c, other=float("-inf")), axis=0)
    scale = m / 127.0
    inv = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + r, scale)

    offs = r * K + c * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = offs < (r + 1) * K
    fv = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q = _round_half_up(fv * inv)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(out_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quantize_asym_dyn_kernel(
    input_ptr,
    tmp_max_ptr,
    tmp_min_ptr,
    scale_out_ptr,
    azp_out_ptr,
    out_ptr,
    K,
    CHUNKS,
    BLOCK_C: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs_c = r * CHUNKS + tl.arange(0, BLOCK_C)
    mask_c = offs_c < (r + 1) * CHUNKS
    mx = tl.max(tl.load(tmp_max_ptr + offs_c, mask=mask_c, other=float("-inf")), axis=0)
    mn = tl.min(tl.load(tmp_min_ptr + offs_c, mask=mask_c, other=float("inf")), axis=0)
    scale = (mx - mn) / 255.0
    azp_f = _round_ties_even(-128.0 - mn / scale)
    azp_f = tl.minimum(tl.maximum(azp_f, -2147483648.0), 2147483647.0)
    inv = 1.0 / scale
    tl.store(scale_out_ptr + r, scale)
    tl.store(azp_out_ptr + r, azp_f.to(tl.int32))

    offs = r * K + c * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = offs < (r + 1) * K
    fv = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q = _round_half_up(fv * inv) + azp_f
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(out_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quantize_sym_4r_kernel(
    input_ptr,
    s_ptr,
    out_ptr,
    K,
    ROWS,
    BLOCK: tl.constexpr,
):
    # Static quantize, four rows per program with four independent parallel
    # loads. Only used for K==512 (16 elems/thread at num_warps=4); wider
    # rows with 4 rows/CTA were pathological on this target.
    p = tl.program_id(0)
    s = tl.load(s_ptr)
    inv = 1.0 / s
    r0 = p * 4
    o0 = r0 * K + tl.arange(0, BLOCK)
    m0 = o0 < (r0 + 1) * K
    v0 = tl.load(input_ptr + o0, mask=m0, other=0.0).to(tl.float32)
    o1 = (r0 + 1) * K + tl.arange(0, BLOCK)
    m1 = o1 < (r0 + 2) * K
    v1 = tl.load(input_ptr + o1, mask=m1, other=0.0).to(tl.float32)
    o2 = (r0 + 2) * K + tl.arange(0, BLOCK)
    m2 = o2 < (r0 + 3) * K
    v2 = tl.load(input_ptr + o2, mask=m2, other=0.0).to(tl.float32)
    o3 = (r0 + 3) * K + tl.arange(0, BLOCK)
    m3 = o3 < (r0 + 4) * K
    v3 = tl.load(input_ptr + o3, mask=m3, other=0.0).to(tl.float32)
    q0 = _round_half_up(v0 * inv)
    q0 = tl.minimum(tl.maximum(q0, -128.0), 127.0)
    tl.store(out_ptr + o0, q0.to(tl.int8), mask=m0)
    q1 = _round_half_up(v1 * inv)
    q1 = tl.minimum(tl.maximum(q1, -128.0), 127.0)
    tl.store(out_ptr + o1, q1.to(tl.int8), mask=m1)
    q2 = _round_half_up(v2 * inv)
    q2 = tl.minimum(tl.maximum(q2, -128.0), 127.0)
    tl.store(out_ptr + o2, q2.to(tl.int8), mask=m2)
    q3 = _round_half_up(v3 * inv)
    q3 = tl.minimum(tl.maximum(q3, -128.0), 127.0)
    tl.store(out_ptr + o3, q3.to(tl.int8), mask=m3)


@triton.jit
def _quantize_sym_2r_kernel(
    input_ptr,
    s_ptr,
    out_ptr,
    K,
    ROWS,
    BLOCK: tl.constexpr,
):
    # Static quantize, two rows per program with two independent parallel
    # loads (preserves memory-level parallelism). For tiny-K many-row shapes
    # where one CTA per 512-1024 element row is latency-bound.
    p = tl.program_id(0)
    s = tl.load(s_ptr)
    inv = 1.0 / s
    r0 = p * 2
    o0 = r0 * K + tl.arange(0, BLOCK)
    m0 = o0 < (r0 + 1) * K
    v0 = tl.load(input_ptr + o0, mask=m0, other=0.0).to(tl.float32)
    q0 = _round_half_up(v0 * inv)
    q0 = tl.minimum(tl.maximum(q0, -128.0), 127.0)
    tl.store(out_ptr + o0, q0.to(tl.int8), mask=m0)
    if r0 + 1 < ROWS:
        o1 = (r0 + 1) * K + tl.arange(0, BLOCK)
        m1 = o1 < (r0 + 2) * K
        v1 = tl.load(input_ptr + o1, mask=m1, other=0.0).to(tl.float32)
        q1 = _round_half_up(v1 * inv)
        q1 = tl.minimum(tl.maximum(q1, -128.0), 127.0)
        tl.store(out_ptr + o1, q1.to(tl.int8), mask=m1)


@triton.jit
def _quantize_sym_2c_kernel(
    input_ptr,
    s_ptr,
    out_ptr,
    K,
    BLOCK0: tl.constexpr,
    BLOCK1: tl.constexpr,
):
    # Static quantize with exact two-chunk split (no masks, no padded lanes):
    # K = 5120 -> (4096, 1024), num_warps=4 for wide per-thread vectors.
    r = tl.program_id(0)
    s = tl.load(s_ptr)
    inv = 1.0 / s
    base = r * K
    o0 = base + tl.arange(0, BLOCK0)
    v0 = tl.load(input_ptr + o0).to(tl.float32)
    o1 = base + BLOCK0 + tl.arange(0, BLOCK1)
    v1 = tl.load(input_ptr + o1).to(tl.float32)
    q0 = _round_half_up(v0 * inv)
    q0 = tl.minimum(tl.maximum(q0, -128.0), 127.0)
    tl.store(out_ptr + o0, q0.to(tl.int8))
    q1 = _round_half_up(v1 * inv)
    q1 = tl.minimum(tl.maximum(q1, -128.0), 127.0)
    tl.store(out_ptr + o1, q1.to(tl.int8))


@triton.jit
def _quantize_sym_reg_kernel(
    input_ptr,
    s_ptr,
    out_ptr,
    K,
    BLOCK: tl.constexpr,
):
    # Register-resident static quantize, one program per row, BLOCK==K (no
    # padded lanes). Same structure as the dynamic fused kernel that reaches
    # ~1.08TB/s; uses _round_half_up and a per-program reciprocal.
    r = tl.program_id(0)
    s = tl.load(s_ptr)
    inv = 1.0 / s
    offs = r * K + tl.arange(0, BLOCK)
    mask = offs < (r + 1) * K
    fv = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q = _round_half_up(fv * inv)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(out_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quantize_sym_2r2c_kernel(
    input_ptr,
    s_ptr,
    out_ptr,
    K,
    ROWS,
    BLOCK0: tl.constexpr,
    BLOCK1: tl.constexpr,
):
    # Static quantize, two rows x two exact chunks per program (4 independent
    # parallel loads). For 512x5120 where 512 CTAs cannot hide latency as well
    # as the 2048-CTA dynamic case.
    p = tl.program_id(0)
    s = tl.load(s_ptr)
    inv = 1.0 / s
    r0 = p * 2
    b0 = r0 * K
    o00 = b0 + tl.arange(0, BLOCK0)
    v00 = tl.load(input_ptr + o00).to(tl.float32)
    o01 = b0 + BLOCK0 + tl.arange(0, BLOCK1)
    v01 = tl.load(input_ptr + o01).to(tl.float32)
    b1 = (r0 + 1) * K
    o10 = b1 + tl.arange(0, BLOCK0)
    v10 = tl.load(input_ptr + o10).to(tl.float32)
    o11 = b1 + BLOCK0 + tl.arange(0, BLOCK1)
    v11 = tl.load(input_ptr + o11).to(tl.float32)
    q00 = _round_half_up(v00 * inv)
    q00 = tl.minimum(tl.maximum(q00, -128.0), 127.0)
    tl.store(out_ptr + o00, q00.to(tl.int8))
    q01 = _round_half_up(v01 * inv)
    q01 = tl.minimum(tl.maximum(q01, -128.0), 127.0)
    tl.store(out_ptr + o01, q01.to(tl.int8))
    q10 = _round_half_up(v10 * inv)
    q10 = tl.minimum(tl.maximum(q10, -128.0), 127.0)
    tl.store(out_ptr + o10, q10.to(tl.int8))
    q11 = _round_half_up(v11 * inv)
    q11 = tl.minimum(tl.maximum(q11, -128.0), 127.0)
    tl.store(out_ptr + o11, q11.to(tl.int8))


@triton.jit
def _quantize_sym_kernel(
    input_ptr,
    s_ptr,
    out_ptr,
    K,
    BLOCK_Q: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    s = tl.load(s_ptr)
    inv = 1.0 / s
    offs = r * K + c * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = offs < (r + 1) * K
    fv = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q = _round_half_up(fv * inv)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(out_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quantize_asym_kernel(
    input_ptr,
    scale_ptr,
    azp_ptr,
    out_ptr,
    K,
    BLOCK_Q: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    s = tl.load(scale_ptr)
    a = tl.load(azp_ptr).to(tl.float32)
    inv = 1.0 / s
    offs = r * K + c * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = offs < (r + 1) * K
    fv = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q = _round_half_up(fv * inv) + a
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(out_ptr + offs, q.to(tl.int8), mask=mask)


_MAX_BLOCK = 1024


def scaled_int8_quant(input, scale, azp, symmetric):
    if input.dim() == 1:
        rows = 1
        K = input.shape[0]
    else:
        rows, K = input.shape
    device = input.device
    output = torch.empty(input.shape, dtype=torch.int8, device=device)

    block_q = min(_MAX_BLOCK, triton.next_power_of_2(K))

    if scale is not None:
        # static scale given by caller
        grid = (rows, triton.cdiv(K, block_q))
        if symmetric:
            if 2048 <= K <= 4096 and (K & (K - 1)) == 0:
                _quantize_sym_reg_kernel[(rows,)](
                    input,
                    scale,
                    output,
                    K,
                    BLOCK=K,
                    num_warps=K // 1024,
                )
                return output, scale, None
            if K == 5120 and rows >= 4 and (rows % 2) == 0:
                _quantize_sym_2r2c_kernel[(rows // 2,)](
                    input,
                    scale,
                    output,
                    K,
                    rows,
                    BLOCK0=4096,
                    BLOCK1=1024,
                    num_warps=8,
                )
                return output, scale, None
            if K == 5120:
                _quantize_sym_2c_kernel[(rows,)](
                    input,
                    scale,
                    output,
                    K,
                    BLOCK0=4096,
                    BLOCK1=1024,
                    num_warps=4,
                )
                return output, scale, None
            if K <= 1024 and rows >= 8 and K == 512:
                _quantize_sym_4r_kernel[(rows // 4,)](
                    input,
                    scale,
                    output,
                    K,
                    rows,
                    BLOCK=block_q,
                )
                return output, scale, None
            if K <= 1024 and rows >= 4:
                _quantize_sym_2r_kernel[(triton.cdiv(rows, 2),)](
                    input,
                    scale,
                    output,
                    K,
                    rows,
                    BLOCK=block_q,
                )
                return output, scale, None
            _quantize_sym_kernel[grid](
                input,
                scale,
                output,
                K,
                BLOCK_Q=block_q,
            )
            return output, scale, None
        _quantize_asym_kernel[grid](
            input,
            scale,
            azp,
            output,
            K,
            BLOCK_Q=block_q,
        )
        return output, scale, azp

    # dynamic: per-row scale (and azp) computed from the input
    block_r = min(_MAX_BLOCK, triton.next_power_of_2(K))
    chunks = triton.cdiv(K, block_r)
    block_c = triton.next_power_of_2(chunks)
    grid = (rows, triton.cdiv(K, block_q))

    if symmetric:
        if K == 13824:
            scale_out = torch.empty((rows, 1), dtype=torch.float32, device=device)
            _sym_fused4_kernel[(rows,)](
                input,
                scale_out,
                output,
                K,
                C0=4096,
                C1=4096,
                C2=4096,
                C3=1024,
                num_warps=8,
            )
            return output, scale_out, None
        if K == 5120 and rows >= 32:
            scale_out = torch.empty((rows, 1), dtype=torch.float32, device=device)
            _sym_fused2_kernel[(rows,)](
                input,
                scale_out,
                output,
                K,
                BLOCK0=4096,
                BLOCK1=1024,
                num_warps=4,
            )
            return output, scale_out, None
        if (rows >= 32 or K <= 1024) and K <= 8192:
            fused_block = triton.next_power_of_2(K)
            nw = 2 if fused_block <= 1024 else max(4, fused_block // 1024)
            scale_out = torch.empty((rows, 1), dtype=torch.float32, device=device)
            _sym_fused_kernel[(rows,)](
                input,
                scale_out,
                output,
                K,
                BLOCK=fused_block,
                num_warps=nw,
            )
            return output, scale_out, None
        tmp = torch.empty((rows, chunks), dtype=torch.float32, device=device)
        scale_out = torch.empty((rows, 1), dtype=torch.float32, device=device)
        _reduce_absmax_kernel[(rows, chunks)](
            input,
            tmp,
            K,
            chunks,
            BLOCK_R=block_r,
        )
        _quantize_sym_dyn_kernel[grid](
            input,
            tmp,
            scale_out,
            output,
            K,
            chunks,
            BLOCK_C=block_c,
            BLOCK_Q=block_q,
        )
        return output, scale_out, None

    tmp_max = torch.empty((rows, chunks), dtype=torch.float32, device=device)
    tmp_min = torch.empty((rows, chunks), dtype=torch.float32, device=device)
    scale_out = torch.empty((rows, 1), dtype=torch.float32, device=device)
    azp_out = torch.empty((rows, 1), dtype=torch.int32, device=device)
    _reduce_minmax_kernel[(rows, chunks)](
        input,
        tmp_max,
        tmp_min,
        K,
        chunks,
        BLOCK_R=block_r,
    )
    _quantize_asym_dyn_kernel[grid](
        input,
        tmp_max,
        tmp_min,
        scale_out,
        azp_out,
        output,
        K,
        chunks,
        BLOCK_C=block_c,
        BLOCK_Q=block_q,
    )
    return output, scale_out, azp_out

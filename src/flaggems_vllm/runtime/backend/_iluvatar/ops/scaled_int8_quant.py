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


# ---------------------------------------------------------------------------
# Round-half-to-even (matches torch.round / reference .round()) using only
# floor, so it works on backends without tl.round.  All arithmetic in f32:
#   r = floor(x + 0.5)  (round-half-up candidate)
#   if |x - r| == 0.5 and r is odd: result = r - 1
# ---------------------------------------------------------------------------
@triton.jit
def _rnd(x):
    r = tl.floor(x + 0.5)
    half = tl.abs(x - r) == 0.5
    odd = tl.abs(r - 2.0 * tl.floor(r / 2.0)) == 1.0
    adj = tl.where(half, tl.where(odd, 1.0, 0.0), 0.0)
    return r - adj


# ---------------------------------------------------------------------------
# Static scale: pure pointwise quantization.
#   symmetric:  out = clamp(round(x / scale), -128, 127) as int8
#   asymmetric: out = clamp(round(x / scale) + azp, -128, 127) as int8
# ---------------------------------------------------------------------------
@triton.jit
def _quant_static_sym_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr).to(tl.float32)
    q = _rnd(x / s)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _quant_static_asym_kernel(
    input_ptr,
    scale_ptr,
    azp_ptr,
    output_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr).to(tl.float32)
    a = tl.load(azp_ptr).to(tl.float32)
    q = _rnd(x / s) + a
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)


# ---------------------------------------------------------------------------
# Dynamic symmetric: per-row absmax reduction, then quantize.
# One CTA per row; pass 1 reduces, pass 2 re-reads (L2-resident) and
# quantizes.  HAS_MASK removes all masking when cols % BLOCK_N == 0 so the
# timing shapes (all multiples of 512/1024) run unpredicated.
# Matches reference arithmetic exactly:
#   scale = absmax / 127 ; inv = where(absmax==0, 0, 127/absmax)
#   out   = clamp(round(x * inv), -128, 127)
# ---------------------------------------------------------------------------
@triton.jit
def _quant_dynamic_sym_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    cols,
    HAS_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * cols
    n_chunks = tl.cdiv(cols, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_N)
    acc = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    for i in tl.range(0, n_chunks):
        offs = i * BLOCK_N + offs_c
        if HAS_MASK:
            mask = offs < cols
            x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(x))
        else:
            x = tl.load(input_ptr + base + offs).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(x))
    absmax = tl.max(acc, axis=0)
    inv = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    tl.store(scale_out_ptr + row, absmax / 127.0)
    for i in tl.range(0, n_chunks):
        offs = i * BLOCK_N + offs_c
        if HAS_MASK:
            mask = offs < cols
            x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            q = _rnd(x * inv)
            q = tl.minimum(tl.maximum(q, -128.0), 127.0)
            tl.store(output_ptr + base + offs, q.to(tl.int8), mask=mask)
        else:
            x = tl.load(input_ptr + base + offs).to(tl.float32)
            q = _rnd(x * inv)
            q = tl.minimum(tl.maximum(q, -128.0), 127.0)
            tl.store(output_ptr + base + offs, q.to(tl.int8))


# ---------------------------------------------------------------------------
# Dynamic asymmetric: per-row max/min reduction, then quantize with the
# computed scale and zero point.  Matches reference arithmetic exactly:
#   scale = (max - min) / 255
#   azp   = clamp(round(-128 - min/scale), -2^31, 2^31-1) as int32
#   out   = clamp(round(x / scale) + azp, -128, 127)
# ---------------------------------------------------------------------------
@triton.jit
def _quant_dynamic_asym_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    cols,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * cols
    n_chunks = tl.cdiv(cols, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_N)
    acc_max = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    acc_min = tl.full((BLOCK_N,), float("inf"), tl.float32)
    for i in tl.range(0, n_chunks):
        offs = i * BLOCK_N + offs_c
        mask = offs < cols
        x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc_max = tl.maximum(acc_max, tl.where(mask, x, float("-inf")))
        acc_min = tl.minimum(acc_min, tl.where(mask, x, float("inf")))
    row_max = tl.max(acc_max, axis=0)
    row_min = tl.min(acc_min, axis=0)
    out_scale = (row_max - row_min) / 255.0
    azp_f = _rnd(-128.0 - row_min / out_scale)
    azp_f = tl.minimum(tl.maximum(azp_f, -2147483648.0), 2147483647.0)
    tl.store(scale_out_ptr + row, out_scale)
    tl.store(azp_out_ptr + row, azp_f.to(tl.int32))
    for i in tl.range(0, n_chunks):
        offs = i * BLOCK_N + offs_c
        mask = offs < cols
        x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        q = _rnd(x / out_scale) + azp_f
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + base + offs, q.to(tl.int8), mask=mask)


def _dyn_cfg(rows, cols):
    """Shape-specialized (BLOCK_N, num_warps) for the dynamic row kernels.

    Measured best configs on the target (Iluvatar BI-V150, 16 SMs):
      1-row rows want big blocks and many warps (latency bound);
      tall matrices want small blocks and few warps (occupancy bound).
    """
    if rows == 1:
        if cols <= 512:
            return 512, 4
        if cols <= 2048:
            return 1024, 8
        return 4096, 16
    if cols <= 1024:
        return 1024, 8
    if rows >= 4096:
        return 1024, 4
    if cols <= 4096:
        return 2048, 16
    return 1024, 8


def _static_cfg(numel):
    if numel < 65536:
        return 2048, 8
    return 512, 4


def scaled_int8_quant(input, scale, azp, symmetric):
    if isinstance(symmetric, torch.Tensor):
        symmetric = bool(symmetric.item())
    else:
        symmetric = bool(symmetric)

    rows, cols = input.shape
    device = input.device
    output = torch.empty_like(input, dtype=torch.int8)

    if scale is None:
        # dynamic: compute per-row scale (and zero point) on device
        scale_out = torch.empty((rows, 1), dtype=torch.float32, device=device)
        if symmetric:
            bn, warps = _dyn_cfg(rows, cols)
            has_mask = (cols % bn) != 0
            grid = (rows,)
            _quant_dynamic_sym_kernel[grid](
                input,
                output,
                scale_out,
                cols,
                HAS_MASK=has_mask,
                BLOCK_N=bn,
                num_warps=warps,
            )
            return output, scale_out, None
        azp_out = torch.empty((rows, 1), dtype=torch.int32, device=device)
        bn, warps = _dyn_cfg(rows, cols)
        grid = (rows,)
        _quant_dynamic_asym_kernel[grid](
            input,
            output,
            scale_out,
            azp_out,
            cols,
            BLOCK_N=bn,
            num_warps=warps,
        )
        return output, scale_out, azp_out

    # static: scale (and azp) are given; output only
    numel = input.numel()
    BLOCK, warps = _static_cfg(numel)
    grid = (triton.cdiv(numel, BLOCK),)
    if azp is None:
        _quant_static_sym_kernel[grid](
            input,
            scale,
            output,
            numel,
            BLOCK=BLOCK,
            num_warps=warps,
        )
        return output, scale, None
    _quant_static_asym_kernel[grid](
        input,
        scale,
        azp,
        output,
        numel,
        BLOCK=BLOCK,
        num_warps=warps,
    )
    return output, scale, azp

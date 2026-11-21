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

import math

import torch
import triton
import triton.language as tl

# Round-half-to-even for the int8 quantize step via the classic 2^23
# magic-add: for v already clamped to [-128, 127], bitcast(v + 1.5*2^23)
# minus the magic's own bit pattern yields rne(v) exactly (the sum stays in
# the ulp == 1 binade). This Triton build exposes no rint intrinsic and the
# fmod+floor emulation below lowers to expensive extern libm calls on GCU
# (it dominated the original kernel: ~12GB/s effective bandwidth).
_RNE_MAGIC = tl.constexpr(12582912.0)
_RNE_MAGIC_BITS = tl.constexpr(1262485504)  # 0x4B400000


@triton.jit
def _round_even(x):
    # torch.round semantics: round half to even, valid on the full fp32
    # domain. Used only for the per-row azp, whose operand can leave the
    # magic-add domain (|v| >= 2^22); the per-element paths use _quant_i8.
    f = x % 1.0  # fractional part, sign follows x
    u = x - f  # integer part (trunc)
    is_half = (f == 0.5) | (f == -0.5)
    up = f > 0.5
    dn = f < -0.5
    r = u + tl.where(up, 1.0, tl.where(dn, -1.0, 0.0))
    half_u = u * 0.5
    u_odd = half_u != tl.math.floor(half_u)
    sign = tl.where(f > 0.0, 1.0, -1.0)
    r = tl.where(is_half & u_odd, u + sign, r)
    return r


@triton.jit
def _quant_i8(x):
    # clamp(round_half_even(x), -128, 127) -> int8, exact for any fp32 x:
    # clamping before the rounding is identical to rounding before the
    # clamp because both bounds are integers.
    t = tl.minimum(tl.maximum(x, -128.0), 127.0) + _RNE_MAGIC
    return (t.to(tl.int32, bitcast=True) - _RNE_MAGIC_BITS).to(tl.int8)


# ---------------------------------------------------------------------------
# Dynamic symmetric quantization (scale computed per row).
#
# Per-program row overhead (absmax reduction tree + dispatch) dominates the
# one-row-per-program layout on GCU, so the main path batches R rows per
# program into 2D tiles with an axis=1 absmax reduction. Power-of-two row
# widths use a single full-width [R, N] tile (128K-element budget, 4 warps at
# >=128K elements / 2 below, grid floor of 16 programs). Non-power-of-two
# widths use the exact-tile kernel: a masked 2D tile pays the full padded
# lane width on GCU, so the row is split into up to four exact power-of-two
# column tiles (only the last may be masked). Shapes with too few rows to
# batch (few tokens) and very wide rows keep dedicated single-row kernels;
# rows wider than the single-block ceiling use the two-pass chunked loop.
# ---------------------------------------------------------------------------


@triton.jit
def _dyn_sym_rowbatch_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    ROWS,
    ROW_SIZE: tl.constexpr,
    R: tl.constexpr,
    LANES: tl.constexpr,
):
    # The tile is kept in the input dtype (no fp32 cast on load): abs/max are
    # pure comparisons and the bf16->fp32 conversion of the reduced amax is
    # exact, so the amax/scale/quant values are bit-identical to a cast-first
    # pipeline while the resident tile stays 2 bytes/elem for 16-bit inputs.
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    rows_offs = pid * R + tl.arange(0, R)
    rmask = rows_offs < ROWS
    base = rows_offs.to(tl.int64)[:, None] * ROW_SIZE
    offs = tl.arange(0, LANES)[None, :]
    mask = rmask[:, None] & (offs < ROW_SIZE)
    x = tl.load(input_ptr + base + offs, mask=mask, other=0.0)
    absmax = tl.max(tl.abs(x), axis=1).to(tl.float32)
    scale = absmax / 127.0
    inv = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    tl.store(scale_out_ptr + rows_offs, scale, mask=rmask)
    q = _quant_i8(x.to(tl.float32) * inv[:, None])
    tl.store(output_ptr + base + offs, q, mask=mask)


@triton.jit
def _dyn_sym_tiles_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    ROWS,
    ROW_SIZE: tl.constexpr,
    R: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    W3: tl.constexpr,
    TAIL_END: tl.constexpr,
):
    # Exact power-of-two column decomposition: ROW_SIZE is covered by tiles
    # of widths W0..W3 (each a power of two, trailing zeros allowed) where
    # only the last tile may be masked down to TAIL_END valid columns. On
    # GCU a masked 2D tile pays the full padded lane width (measured:
    # (4096,5120) on an [8,8192] tile runs at the same time as (4096,8192)),
    # so splitting a non-power-of-two row into exact tiles removes the
    # padding waste. The amax reduction is split per tile and merged with
    # tl.maximum; max is associative for finite values, so the result is
    # bit-identical to the single-tile reduction.
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    rows_offs = pid * R + tl.arange(0, R)
    rmask = rows_offs < ROWS
    m = rmask[:, None]
    base = rows_offs.to(tl.int64)[:, None] * ROW_SIZE
    x0 = tl.load(input_ptr + base + tl.arange(0, W0)[None, :], mask=m, other=0.0)
    absmax = tl.max(tl.abs(x0), axis=1)
    if W1 > 0:
        x1 = tl.load(
            input_ptr + base + W0 + tl.arange(0, W1)[None, :], mask=m, other=0.0
        )
        absmax = tl.maximum(absmax, tl.max(tl.abs(x1), axis=1))
    if W2 > 0:
        x2 = tl.load(
            input_ptr + base + W0 + W1 + tl.arange(0, W2)[None, :], mask=m, other=0.0
        )
        absmax = tl.maximum(absmax, tl.max(tl.abs(x2), axis=1))
    if W3 > 0:
        cols3 = W0 + W1 + W2 + tl.arange(0, W3)
        m3 = m & (cols3 < TAIL_END)[None, :]
        x3 = tl.load(input_ptr + base + cols3[None, :], mask=m3, other=0.0)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x3), axis=1))
    absmax = absmax.to(tl.float32)
    scale = absmax / 127.0
    inv = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    tl.store(scale_out_ptr + rows_offs, scale, mask=rmask)
    tl.store(
        output_ptr + base + tl.arange(0, W0)[None, :],
        _quant_i8(x0.to(tl.float32) * inv[:, None]),
        mask=m,
    )
    if W1 > 0:
        tl.store(
            output_ptr + base + W0 + tl.arange(0, W1)[None, :],
            _quant_i8(x1.to(tl.float32) * inv[:, None]),
            mask=m,
        )
    if W2 > 0:
        tl.store(
            output_ptr + base + W0 + W1 + tl.arange(0, W2)[None, :],
            _quant_i8(x2.to(tl.float32) * inv[:, None]),
            mask=m,
        )
    if W3 > 0:
        tl.store(
            output_ptr + base + cols3[None, :],
            _quant_i8(x3.to(tl.float32) * inv[:, None]),
            mask=m3,
        )


@triton.jit
def _dyn_sym_single_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    base = row.to(tl.int64) * ROW_SIZE
    offs = tl.arange(0, BLOCK)
    mask = offs < ROW_SIZE
    x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = absmax / 127.0
    inv = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    tl.store(scale_out_ptr + row, scale)
    tl.store(output_ptr + base + offs, _quant_i8(x * inv), mask=mask)


@triton.jit
def _dyn_sym_loop_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    ROW_SIZE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    base = row.to(tl.int64) * ROW_SIZE
    offs = tl.arange(0, CHUNK)
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for off in range(0, ROW_SIZE, CHUNK):
        m = off + offs < ROW_SIZE
        x = tl.load(input_ptr + base + off + offs, mask=m, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    absmax = tl.max(acc, axis=0)
    scale = absmax / 127.0
    inv = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    tl.store(scale_out_ptr + row, scale)
    for off in range(0, ROW_SIZE, CHUNK):
        m = off + offs < ROW_SIZE
        x = tl.load(input_ptr + base + off + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(output_ptr + base + off + offs, _quant_i8(x * inv), mask=m)


# ---------------------------------------------------------------------------
# Dynamic asymmetric quantization (scale + azp computed per row). Kept on the
# one-row-per-program layout: azp needs a wide-domain rounding (see
# _round_even) and the asymmetric path is not throughput-critical.
# ---------------------------------------------------------------------------


@triton.jit
def _dyn_asym_single_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    base = row.to(tl.int64) * ROW_SIZE
    offs = tl.arange(0, BLOCK)
    mask = offs < ROW_SIZE
    x = tl.load(input_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    xmax = tl.where(mask, x, float("-inf"))
    xmin = tl.where(mask, x, float("inf"))
    row_max = tl.max(xmax, axis=0)
    row_min = tl.min(xmin, axis=0)
    scale = (row_max - row_min) / 255.0
    safe = tl.where(scale == 0.0, 1.0, scale)
    azp = _round_even(-128.0 - row_min / safe)
    azp = tl.minimum(tl.maximum(azp, -2147483648.0), 2147483647.0)
    azp_i = azp.to(tl.int32)
    tl.store(scale_out_ptr + row, scale)
    tl.store(azp_out_ptr + row, azp_i)
    # round(x / safe) + azp then clamp == clamp-round(x / safe + azp):
    # azp is an integer, so folding it below the rounding is exact.
    q = _quant_i8(x / safe + azp_i.to(tl.float32))
    tl.store(output_ptr + base + offs, q, mask=mask)


@triton.jit
def _dyn_asym_loop_kernel(
    input_ptr,
    output_ptr,
    scale_out_ptr,
    azp_out_ptr,
    ROW_SIZE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    base = row.to(tl.int64) * ROW_SIZE
    offs = tl.arange(0, CHUNK)
    acc_max = tl.full([CHUNK], float("-inf"), tl.float32)
    acc_min = tl.full([CHUNK], float("inf"), tl.float32)
    for off in range(0, ROW_SIZE, CHUNK):
        m = off + offs < ROW_SIZE
        x = tl.load(input_ptr + base + off + offs, mask=m, other=0.0).to(tl.float32)
        acc_max = tl.maximum(acc_max, tl.where(m, x, float("-inf")))
        acc_min = tl.minimum(acc_min, tl.where(m, x, float("inf")))
    row_max = tl.max(acc_max, axis=0)
    row_min = tl.min(acc_min, axis=0)
    scale = (row_max - row_min) / 255.0
    safe = tl.where(scale == 0.0, 1.0, scale)
    azp = _round_even(-128.0 - row_min / safe)
    azp = tl.minimum(tl.maximum(azp, -2147483648.0), 2147483647.0)
    azp_i = azp.to(tl.int32)
    tl.store(scale_out_ptr + row, scale)
    tl.store(azp_out_ptr + row, azp_i)
    azp_f = azp_i.to(tl.float32)
    for off in range(0, ROW_SIZE, CHUNK):
        m = off + offs < ROW_SIZE
        x = tl.load(input_ptr + base + off + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(output_ptr + base + off + offs, _quant_i8(x / safe + azp_f), mask=m)


# ---------------------------------------------------------------------------
# Static quantization: pointwise with a per-tensor scale (and azp). Flat 1-D
# stream over the whole tensor; the block follows a sqrt(numel) law clamped
# to [1024, 65536] (on-target sweep: per-CTA dispatch dominates below ~16K
# elements, while 65536 is the 2B-dtype tile ceiling before the DSM
# double-program line).
# ---------------------------------------------------------------------------


@triton.jit
def _static_sym_kernel(input_ptr, scale_ptr, output_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv_s = 1.0 / tl.load(scale_ptr)
    tl.store(output_ptr + offs, _quant_i8(x * inv_s), mask=mask)


@triton.jit
def _static_asym_kernel(
    input_ptr, scale_ptr, azp_ptr, output_ptr, numel, BLOCK: tl.constexpr
):
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr)
    a = tl.load(azp_ptr).to(tl.float32)
    # round(x / s) + azp then clamp == clamp-round(x / s + azp): azp is an
    # integer, so folding it below the rounding is exact.
    q = _quant_i8(x / s + a)
    tl.store(output_ptr + offs, q, mask=mask)


# ---------------------------------------------------------------------------
# Host dispatch.
# ---------------------------------------------------------------------------

_GRID_X_CAP = 65535  # grid.x hardware limit on Enflame GCU
_DYN_FULL_TILE_BYTES = 262144  # full-width row-batch tile budget (bytes)
_DYN_TILES_TILE_BYTES = 458752  # exact-tile row-batch tile budget (bytes)
_DYN_NW4_ELEMS = 131072  # tile size (elements) at/above which 4 warps win


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _prev_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _grid_for(num_blocks):
    # Spread blocks across a 2-D grid so the block size is unaffected by the
    # tensor size; the kernels recombine (pid_x, pid_y) into a flat block id.
    grid_x = min(num_blocks, _GRID_X_CAP)
    return (grid_x, triton.cdiv(num_blocks, grid_x))


def _row_tiles(n):
    """Column decomposition for the exact-tile row-batch kernel.

    Up to three exact power-of-two tiles plus one tail tile (always in the
    last slot) whose width is the next power of two of the remainder, masked
    down to the valid columns; an exact remainder leaves the mask all-true
    and folded. Widths below 2 are avoided (single-lane vectors are a known
    slow path on GCU): a remainder of 1 is padded to a masked width-2 tail.
    Returns (w0, w1, w2, w3, tail_end) or None when the shape does not fit
    (row wider than 32768 or no full-width leading tile).
    """
    if n > 32768:
        return None
    widths = []
    v = n
    while v > 0 and len(widths) < 3:
        w = 1 << (v.bit_length() - 1)
        if w < 2:
            break
        widths.append(w)
        v -= w
    tail_end = 0
    if v > 0:
        w = max(2, _next_pow2(v))
        if w > 16384:
            return None
        while len(widths) < 3:
            widths.append(0)
        widths.append(w)
        tail_end = n
    while len(widths) < 4:
        widths.append(0)
    if widths[0] < 2:
        return None
    return widths[0], widths[1], widths[2], widths[3], tail_end


def _dyn_batch_rows(rows, row_size, budget_elems, min_tile):
    """Rows per program for the dynamic symmetric row-batch kernels.

    Capped by the tile budget and by the machine fill (grid >= 16 programs:
    below that the dispatch/launch floor dominates and smaller tiles win).
    R is additionally capped at 64: every on-target-validated configuration
    of both row-batch kernels lives at R <= 64 (the perf sweet spots are all
    R <= 32), while an [R=256, W=16] axis=1 reduction produced silently
    wrong amax values on GCU (same narrow-tile multi-warp row-reduce
    miscompile family as compiler_issues/sglang_fused_eh_norm_...md).

    For small row counts (<= 128) the single-row kernel keeps a healthy
    grid, so tiny row-batch tiles fall back to it (on-target: (64,512)
    regressed 16% on an [4,512] tile, (64,13824) regressed 28% on a 4-tile
    batch). Larger row counts always keep the batched form: falling to the
    single-row kernel there would explode the grid.
    """
    r_cap = _prev_pow2(max(1, budget_elems // row_size))
    r_cap = min(r_cap, 64)
    r = min(r_cap, _prev_pow2(max(1, rows // 16)))
    if rows <= 128:
        while r > 1 and r * row_size < min_tile:
            r //= 2
    return r


def _static_block(numel):
    target = 32 * math.isqrt(numel)
    return min(65536, _next_pow2(max(1024, target)))


def _run_dynamic(input, sym, out=None, scale_out=None, azp_out=None):
    row_size = input.shape[-1]
    rows = input.numel() // row_size
    device = input.device

    if out is None:
        out = torch.empty(input.shape, dtype=torch.int8, device=device)
    if scale_out is None:
        scale_out = torch.empty((rows, 1), dtype=torch.float32, device=device)
    if rows == 0:
        if not sym and azp_out is None:
            azp_out = torch.empty((rows, 1), dtype=torch.int32, device=device)
        return out, scale_out, None if sym else azp_out

    lanes = _next_pow2(row_size)
    if sym:
        # Row-batch fast paths (16/32-bit-friendly element sizes only; wider
        # dtypes keep the cast-first single/loop kernels below, matching the
        # fp32-cast semantics of the reference). On-target laws (GCU400):
        # tile budget in elements, 4 warps at >=128K-element tiles, 2 below,
        # and a grid floor of 16 programs.
        esize = input.element_size()
        if esize <= 4 and row_size >= 2:
            if row_size & (row_size - 1) == 0 and lanes <= 16384:
                # Power-of-two row width: one full-width tile per program
                # (the column mask is all-true and folds away).
                budget = _DYN_FULL_TILE_BYTES // esize
                r = _dyn_batch_rows(rows, row_size, budget, 4096)
                if r >= 2:
                    _dyn_sym_rowbatch_kernel[_grid_for(triton.cdiv(rows, r))](
                        input,
                        out,
                        scale_out,
                        rows,
                        ROW_SIZE=row_size,
                        R=r,
                        LANES=lanes,
                        num_warps=4 if r * lanes >= _DYN_NW4_ELEMS else 2,
                    )
                    return out, scale_out, None
            else:
                tiles = _row_tiles(row_size)
                if tiles is not None:
                    budget = _DYN_TILES_TILE_BYTES // esize
                    w0, w1, w2, w3, tail_end = tiles
                    # More tiles per program means a longer serial op chain,
                    # so small batches need a bigger tile to stay worth it.
                    n_tiles = (w0 > 0) + (w1 > 0) + (w2 > 0) + (w3 > 0)
                    min_tile = 4096 if n_tiles <= 2 else 65536
                    r = _dyn_batch_rows(rows, row_size, budget, min_tile)
                    if r >= 2:
                        _dyn_sym_tiles_kernel[_grid_for(triton.cdiv(rows, r))](
                            input,
                            out,
                            scale_out,
                            rows,
                            ROW_SIZE=row_size,
                            R=r,
                            W0=w0,
                            W1=w1,
                            W2=w2,
                            W3=w3,
                            TAIL_END=tail_end,
                            num_warps=4 if r * row_size >= _DYN_NW4_ELEMS else 2,
                        )
                        return out, scale_out, None
        if row_size <= 32768:
            _dyn_sym_single_kernel[_grid_for(rows)](
                input,
                out,
                scale_out,
                ROW_SIZE=row_size,
                BLOCK=lanes,
                num_warps=8 if lanes <= 1024 else 4,
            )
            return out, scale_out, None
        _dyn_sym_loop_kernel[_grid_for(rows)](
            input,
            out,
            scale_out,
            ROW_SIZE=row_size,
            CHUNK=4096,
            num_warps=8,
        )
        return out, scale_out, None

    if azp_out is None:
        azp_out = torch.empty((rows, 1), dtype=torch.int32, device=device)
    if row_size <= 16384:
        _dyn_asym_single_kernel[_grid_for(rows)](
            input,
            out,
            scale_out,
            azp_out,
            ROW_SIZE=row_size,
            BLOCK=lanes,
            num_warps=8 if lanes <= 1024 else 4,
        )
        return out, scale_out, azp_out
    _dyn_asym_loop_kernel[_grid_for(rows)](
        input,
        out,
        scale_out,
        azp_out,
        ROW_SIZE=row_size,
        CHUNK=4096,
        num_warps=8,
    )
    return out, scale_out, azp_out


def _run_static(input, scale, azp, sym, out=None):
    numel = input.numel()
    device = input.device
    if out is None:
        out = torch.empty(input.shape, dtype=torch.int8, device=device)
    if numel == 0:
        return out, scale, None if sym else azp
    block = _static_block(numel)
    # 16384 is the one block size preferring 4 warps (on-target sweep);
    # everything else is fastest with 2.
    nw = 4 if block == 16384 else 2
    grid = _grid_for(triton.cdiv(numel, block))
    if sym:
        _static_sym_kernel[grid](input, scale, out, numel, BLOCK=block, num_warps=nw)
        return out, scale, None
    _static_asym_kernel[grid](input, scale, azp, out, numel, BLOCK=block, num_warps=nw)
    return out, scale, azp


def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
):
    if hasattr(symmetric, "item"):
        sym = bool(symmetric.item())
    else:
        sym = bool(symmetric)

    if scale is None:
        return _run_dynamic(input, sym)
    return _run_static(input, scale, azp, sym)


# ---------------------------------------------------------------------------
# vLLM torch.ops registration.
#
# On the GCU there is no CUDA driver, so vLLM's `_C` extension never loads and
# `torch.ops._C.dynamic_scaled_int8_quant` / `torch.ops._C.static_scaled_int8_quant`
# stay unregistered: vLLM's `scaled_int8_quant` wrapper (used by this repo's
# vs-vLLM benchmarks and by any model path that quantizes activations) then
# fails with `AttributeError: '_OpNamespace' '_C' object has no attribute
# '...'`. Register this backend's Triton implementation under the same
# qualified names so the wrapper resolves on GCU. Hosts where the native CUDA
# ops are already registered (real CUDA boxes) skip the override.
# ---------------------------------------------------------------------------


def _c_dynamic_scaled_int8_quant_impl(out, input, scales, azp):
    _run_dynamic(input, azp is None, out=out, scale_out=scales, azp_out=azp)


def _c_static_scaled_int8_quant_impl(out, input, scales, azp):
    _run_static(input, scales, azp, azp is None, out=out)


def _install_vllm_c_ops():
    # The Library object must stay referenced for the lifetime of the process:
    # torch.library.Library unregisters its ops when garbage collected, so the
    # handle is kept in a module-level global (apply_repetition_penalties.py
    # keeps `_c_lib` alive for the same reason).
    global _C_OP_LIB
    if hasattr(torch.ops._C, "dynamic_scaled_int8_quant") and hasattr(
        torch.ops._C, "static_scaled_int8_quant"
    ):
        return
    _C_OP_LIB = torch.library.Library("_C", "FRAGMENT")
    if not hasattr(torch.ops._C, "dynamic_scaled_int8_quant"):
        # dynamic_scaled_int8_quant(out!, input, scales!, azp!) -> ()
        _C_OP_LIB.define(
            "dynamic_scaled_int8_quant(Tensor(a!) out, Tensor input, "
            "Tensor(b!) scales, Tensor(c!)? azp) -> ()"
        )
        _C_OP_LIB.impl("dynamic_scaled_int8_quant", _c_dynamic_scaled_int8_quant_impl)
    if not hasattr(torch.ops._C, "static_scaled_int8_quant"):
        # static_scaled_int8_quant(out!, input, scales, azp) -> ()
        _C_OP_LIB.define(
            "static_scaled_int8_quant(Tensor(a!) out, Tensor input, "
            "Tensor scales, Tensor? azp) -> ()"
        )
        _C_OP_LIB.impl("static_scaled_int8_quant", _c_static_scaled_int8_quant_impl)


_C_OP_LIB = None
# Re-enabled 2026-11-11 (E-2 bench unblock): this install is a pure
# torch.library.Library("_C", "FRAGMENT") op registration routing vLLM's
# scaled_int8_quant wrapper to this backend's Triton kernels on GCU — it is
# NOT a GCUIR/gcu_compiler_opt patch and never touched the compiler
# downgrade-retry path whose regression (commit b333be6) motivated the 48fd644
# blanket disable; that commit listed this file among "private GCUIR shims"
# but the only install call here was this torch-compat registration (same
# non-GCUIR class as the library-compat shims 48fd644 explicitly preserved,
# and as topk_softplus_sqrt.py / apply_repetition_penalties.py, whose module
# level _moe_C/_C registrations stayed enabled and delivered). With it
# disabled, benchmark/test_scaled_int8_quant.py fails wholesale with
# "'_OpNamespace' '_C' object has no attribute 'static_scaled_int8_quant'".
# Hosts where the native ops are registered (real CUDA boxes) still skip.
_install_vllm_c_ops()

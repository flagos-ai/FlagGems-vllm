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
# Rounding without a slow float->int conversion: this Kunlunxin Triton fork
# has no tl.round and no linkable libdevice, and a bitcast of a COMPUTED
# register value costs ~2.4ms per 16.8M elements (~7 Gelem/s wall), while a
# bitcast folded into a global LOAD is free.  The magic-number trick
# (f + 1.5*2^23 rounds to nearest-even in float; the bit pattern of the sum,
# minus 0x4B400000 = 1262485504, is the IEEE round-half-to-even integer for
# |f| <= 2^22) is therefore split across two kernels on large tensors:
#   stage A:  clamp(x*inv) + 1.5*2^23  -> f32 temp   (pure float, fast)
#   stage B:  load temp, fold bitcast, sub, clamp, int8 store (load-fold is
#             free, so the expensive register bitcast never executes)
# Small tensors stay in one fused kernel (launch-bound).
#
# Other backend quirks discovered empirically:
#  * a kernel containing BOTH a tl.max and a tl.min reduction miscompiles the
#    min result, so every kernel uses at most one reduction kind;
#  * an inner `for c in range(NCHUNKS)` loop over 2D masked tiles is ~10x
#    slower than a single-chunk tile, so wide rows use a simple 2D grid;
#  * 1D grids with runtime divmod are pathological; use clean 2D grids;
#  * redundant element masks (offs < C when the tile exactly covers the
#    tensor, or offs < N when N % BLOCK == 0) cost 12-24% on this backend,
#    so "nomask" kernel variants are used whenever the access is provably
#    in-bounds (C == BLOCK and R % RT == 0, or N % BLOCK == 0).
# ---------------------------------------------------------------------------


@triton.jit
def _magic_i32(xf):
    f = tl.minimum(tl.maximum(xf, -4194304.0), 4194304.0)
    s = f + 12582912.0
    return s.to(tl.int32, bitcast=True) - 1262485504


@triton.jit
def _quant_mul_i8(xf, inv):
    i = _magic_i32(xf * inv)
    i = tl.minimum(tl.maximum(i, -128), 127)
    return i.to(tl.int8)


@triton.jit
def _quant_div_i8(xf, s):
    i = _magic_i32(xf / s)
    i = tl.minimum(tl.maximum(i, -128), 127)
    return i.to(tl.int8)


@triton.jit
def _quant_div_azp_i8(xf, s, azp):
    i = _magic_i32(xf / s) + azp
    i = tl.minimum(tl.maximum(i, -128), 127)
    return i.to(tl.int8)


@triton.jit
def _round_even_any(x):
    # Floor-based round-half-to-even, exact at any magnitude (per-row use only).
    xp = x + 0.5
    r = tl.floor(xp)
    tie = xp == r
    odd = (r - 2.0 * tl.floor(r * 0.5)) != 0.0
    return tl.where(tie, tl.where(odd, r - 1.0, r), r)


# ---------------------------------------------------------------------------
# Partial per-row reductions (stage 1).  Single reduction kind per kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _partial_absmax_tile(
    in_ptr,
    part_ptr,
    R,
    C,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * RT + tl.arange(0, RT)
    rmask = rows < R
    offs = tl.arange(0, BLOCK)
    mmask = rmask[:, None] & (offs < C)[None, :]
    ptrs = in_ptr + rows[:, None] * C + offs[None, :]
    t = tl.load(ptrs, mask=mmask, other=0.0)
    m = tl.max(tl.abs(t.to(tl.float32)), axis=1)
    tl.store(part_ptr + rows, m, mask=rmask)


@triton.jit
def _partial_absmax_tile_nomask(
    in_ptr,
    part_ptr,
    R,
    C,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Safe only when C == BLOCK and R % RT == 0 (tile exactly covers input).
    pid = tl.program_id(0)
    rows = pid * RT + tl.arange(0, RT)
    rmask = rows < R
    offs = tl.arange(0, BLOCK)
    ptrs = in_ptr + rows[:, None] * C + offs[None, :]
    t = tl.load(ptrs)
    m = tl.max(tl.abs(t.to(tl.float32)), axis=1)
    tl.store(part_ptr + rows, m, mask=rmask)


@triton.jit
def _partial_absmax_inv_tile(
    in_ptr,
    scale_out_ptr,
    inv_ptr,
    R,
    C,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # k2 folded into k1: computes the final row absmax and writes scale/inv
    # directly, eliminating the k2 launch.  For small R the k2 grid was
    # 25-75% idle-lane waste, so this fold wins there.
    pid = tl.program_id(0)
    rows = pid * RT + tl.arange(0, RT)
    rmask = rows < R
    offs = tl.arange(0, BLOCK)
    mmask = rmask[:, None] & (offs < C)[None, :]
    ptrs = in_ptr + rows[:, None] * C + offs[None, :]
    t = tl.load(ptrs, mask=mmask, other=0.0)
    m = tl.max(tl.abs(t.to(tl.float32)), axis=1)
    scale = m / 127.0
    inv = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + rows, scale, mask=rmask)
    tl.store(inv_ptr + rows, inv, mask=rmask)


@triton.jit
def _partial_absmax_inv_tile_nomask(
    in_ptr,
    scale_out_ptr,
    inv_ptr,
    R,
    C,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Safe only when C == BLOCK and R % RT == 0.
    pid = tl.program_id(0)
    rows = pid * RT + tl.arange(0, RT)
    rmask = rows < R
    offs = tl.arange(0, BLOCK)
    ptrs = in_ptr + rows[:, None] * C + offs[None, :]
    t = tl.load(ptrs)
    m = tl.max(tl.abs(t.to(tl.float32)), axis=1)
    scale = m / 127.0
    inv = tl.where(m == 0.0, 0.0, 127.0 / m)
    tl.store(scale_out_ptr + rows, scale, mask=rmask)
    tl.store(inv_ptr + rows, inv, mask=rmask)


@triton.jit
def _partial_max_tile(
    in_ptr,
    part_ptr,
    R,
    C,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * RT + tl.arange(0, RT)
    rmask = rows < R
    offs = tl.arange(0, BLOCK)
    mmask = rmask[:, None] & (offs < C)[None, :]
    ptrs = in_ptr + rows[:, None] * C + offs[None, :]
    t = tl.load(ptrs, mask=mmask, other=0.0)
    mx = tl.max(tl.where(mmask, t.to(tl.float32), float("-inf")), axis=1)
    tl.store(part_ptr + rows, mx, mask=rmask)


@triton.jit
def _partial_min_tile(
    in_ptr,
    part_ptr,
    R,
    C,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * RT + tl.arange(0, RT)
    rmask = rows < R
    offs = tl.arange(0, BLOCK)
    mmask = rmask[:, None] & (offs < C)[None, :]
    ptrs = in_ptr + rows[:, None] * C + offs[None, :]
    t = tl.load(ptrs, mask=mmask, other=0.0)
    mn = tl.min(tl.where(mmask, t.to(tl.float32), float("inf")), axis=1)
    tl.store(part_ptr + rows, mn, mask=rmask)


@triton.jit
def _partial_absmax_2d(in_ptr, part_ptr, C, nchunks, BLOCK: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    x = tl.load(in_ptr + pid_r * C + offs, mask=mask, other=0.0)
    m = tl.max(tl.abs(x.to(tl.float32)), axis=0)
    tl.store(part_ptr + pid_r * nchunks + pid_c, m)


@triton.jit
def _partial_absmax_head_tile(
    in_ptr,
    part_ptr,
    R,
    C,
    COL: tl.constexpr,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # cols [0, BLOCK) of each row (row stride C).  Safe when BLOCK <= C and
    # BLOCK is a power of 2 (the 8192-lane masked reduction tree is ~3x
    # slower per element than this 4096-lane one).  Writes partial into
    # part[row, COL] of a (R, 2) buffer.
    pid = tl.program_id(0)
    rows = pid * RT + tl.arange(0, RT)
    rmask = rows < R
    offs = tl.arange(0, BLOCK)
    ptrs = in_ptr + rows[:, None] * C + offs[None, :]
    t = tl.load(ptrs)
    m = tl.max(tl.abs(t.to(tl.float32)), axis=1)
    tl.store(part_ptr + rows * 2 + COL, m, mask=rmask)


@triton.jit
def _partial_absmax_tail_tile(
    in_ptr,
    part_ptr,
    R,
    C,
    BASE: tl.constexpr,
    COL: tl.constexpr,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # cols [BASE, BASE+BLOCK) of each row.  Safe when BASE+BLOCK <= C and
    # BLOCK is a power of 2.  Writes partial into part[row, COL].
    pid = tl.program_id(0)
    rows = pid * RT + tl.arange(0, RT)
    rmask = rows < R
    offs = BASE + tl.arange(0, BLOCK)
    ptrs = in_ptr + rows[:, None] * C + offs[None, :]
    t = tl.load(ptrs)
    m = tl.max(tl.abs(t.to(tl.float32)), axis=1)
    tl.store(part_ptr + rows * 2 + COL, m, mask=rmask)


@triton.jit
def _partial_max_2d(in_ptr, part_ptr, C, nchunks, BLOCK: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    x = tl.load(in_ptr + pid_r * C + offs, mask=mask, other=0.0)
    mx = tl.max(tl.where(mask, x.to(tl.float32), float("-inf")), axis=0)
    tl.store(part_ptr + pid_r * nchunks + pid_c, mx)


@triton.jit
def _partial_min_2d(in_ptr, part_ptr, C, nchunks, BLOCK: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    x = tl.load(in_ptr + pid_r * C + offs, mask=mask, other=0.0)
    mn = tl.min(tl.where(mask, x.to(tl.float32), float("inf")), axis=0)
    tl.store(part_ptr + pid_r * nchunks + pid_c, mn)


# ---------------------------------------------------------------------------
# Row reduce (stage 2): scale / inverse / azp.  Max-only reductions.
# ---------------------------------------------------------------------------


@triton.jit
def _reduce_sym_tile(
    part_ptr,
    scale_out_ptr,
    inv_ptr,
    R,
    nchunks,
    RPR: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * RPR + tl.arange(0, RPR)
    rmask = rows < R
    offs = tl.arange(0, BLOCK_R)
    cmask = offs < nchunks
    p = tl.load(
        part_ptr + rows[:, None] * nchunks + offs[None, :],
        mask=rmask[:, None] & cmask[None, :],
        other=float("-inf"),
    )
    amax = tl.max(p, axis=1)
    scale = amax / 127.0
    inv = tl.where(amax == 0.0, 0.0, 127.0 / amax)
    tl.store(scale_out_ptr + rows, scale, mask=rmask)
    tl.store(inv_ptr + rows, inv, mask=rmask)


@triton.jit
def _reduce_asym_tile(
    part_max_ptr,
    part_min_ptr,
    scale_out_ptr,
    azp_out_ptr,
    R,
    nchunks,
    RPR: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * RPR + tl.arange(0, RPR)
    rmask = rows < R
    offs = tl.arange(0, BLOCK_R)
    cmask = offs < nchunks
    mm = rmask[:, None] & cmask[None, :]
    pm = tl.load(
        part_max_ptr + rows[:, None] * nchunks + offs[None, :],
        mask=mm,
        other=float("-inf"),
    )
    pn = tl.load(
        part_min_ptr + rows[:, None] * nchunks + offs[None, :],
        mask=mm,
        other=float("inf"),
    )
    rmax = tl.max(pm, axis=1)
    rmin = -tl.max(-pn, axis=1)
    scale = (rmax - rmin) / 255.0
    azpf = _round_even_any(-128.0 - rmin / scale)
    azpf = tl.minimum(tl.maximum(azpf, -2147483648.0), 2147483647.0)
    tl.store(scale_out_ptr + rows, scale, mask=rmask)
    tl.store(azp_out_ptr + rows, azpf.to(tl.int32), mask=rmask)


# ---------------------------------------------------------------------------
# Quantize (stage 3): single fused kernel (small tensors) or two-pass staged
# kernels (large tensors) that dodge the expensive register bitcast.
# ---------------------------------------------------------------------------


@triton.jit
def _quant_sym_kernel(
    in_ptr,
    out_ptr,
    inv_ptr,
    C,
    nchunks,
    BLOCK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    inv = tl.load(inv_ptr + pid_r)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    x = tl.load(in_ptr + pid_r * C + offs, mask=mask, other=0.0)
    q = _quant_mul_i8(x.to(tl.float32), inv)
    tl.store(out_ptr + pid_r * C + offs, q, mask=mask)


@triton.jit
def _quant_sym_kernel_nomask(
    in_ptr,
    out_ptr,
    inv_ptr,
    C,
    nchunks,
    BLOCK: tl.constexpr,
):
    # Safe only when C == BLOCK (single chunk, grid covers exactly R rows).
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    inv = tl.load(inv_ptr + pid_r)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + pid_r * C + offs)
    q = _quant_mul_i8(x.to(tl.float32), inv)
    tl.store(out_ptr + pid_r * C + offs, q)


@triton.jit
def _quant_asym_kernel(
    in_ptr,
    out_ptr,
    scale_out_ptr,
    azp_out_ptr,
    C,
    nchunks,
    BLOCK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    s = tl.load(scale_out_ptr + pid_r)
    azp = tl.load(azp_out_ptr + pid_r)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    x = tl.load(in_ptr + pid_r * C + offs, mask=mask, other=0.0)
    q = _quant_div_azp_i8(x.to(tl.float32), s, azp)
    tl.store(out_ptr + pid_r * C + offs, q, mask=mask)


@triton.jit
def _quant_a_mul(
    in_ptr,
    t_ptr,
    inv_ptr,
    C,
    nchunks,
    BLOCK: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    inv = tl.load(inv_ptr + pid_r)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    x = tl.load(in_ptr + pid_r * C + offs, mask=mask, other=0.0)
    # Tight pre-clamp: round(clamp(f, -128.49, 127.49)) == clamp(round(f)),
    # so the magic-sum range keeps stage B inside [-128,127] and stage B can
    # drop its int clamp entirely.
    f = tl.minimum(tl.maximum(x.to(tl.float32) * inv, -128.49), 127.49)
    tl.store(t_ptr + pid_r * C + offs, f + 12582912.0, mask=mask)


@triton.jit
def _quant_a_mul_nomask(
    in_ptr,
    t_ptr,
    inv_ptr,
    C,
    nchunks,
    BLOCK: tl.constexpr,
):
    # Safe only when C == BLOCK (grid covers exactly R rows).
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    inv = tl.load(inv_ptr + pid_r)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + pid_r * C + offs)
    f = tl.minimum(tl.maximum(x.to(tl.float32) * inv, -128.49), 127.49)
    tl.store(t_ptr + pid_r * C + offs, f + 12582912.0)


@triton.jit
def _quant_a_div(
    in_ptr,
    t_ptr,
    scale_out_ptr,
    C,
    nchunks,
    BLOCK: tl.constexpr,
):
    # Loose clamp: the azp shift in stage B_azp moves the result out of the
    # int8 range again, so the wide clamp (magic-exact for |f|<=2^22) is
    # required here; stage B_azp keeps its own int clamp.
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    s = tl.load(scale_out_ptr + pid_r)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    x = tl.load(in_ptr + pid_r * C + offs, mask=mask, other=0.0)
    f = tl.minimum(tl.maximum(x.to(tl.float32) / s, -4194304.0), 4194304.0)
    tl.store(t_ptr + pid_r * C + offs, f + 12582912.0, mask=mask)


@triton.jit
def _quant_b(t_ptr, out_ptr, C, nchunks, BLOCK: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    s = tl.load(t_ptr + pid_r * C + offs, mask=mask, other=0.0)
    i = s.to(tl.int32, bitcast=True) - 1262485504
    tl.store(out_ptr + pid_r * C + offs, i.to(tl.int8), mask=mask)


@triton.jit
def _quant_b_nomask(t_ptr, out_ptr, C, nchunks, BLOCK: tl.constexpr):
    # Safe only when C == BLOCK (grid covers exactly R rows).  Stage A's
    # tight clamp guarantees i is already inside [-128, 127].
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(t_ptr + pid_r * C + offs)
    i = s.to(tl.int32, bitcast=True) - 1262485504
    tl.store(out_ptr + pid_r * C + offs, i.to(tl.int8))


@triton.jit
def _quant_b_azp(t_ptr, out_ptr, azp_out_ptr, C, nchunks, BLOCK: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    azp = tl.load(azp_out_ptr + pid_r)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    s = tl.load(t_ptr + pid_r * C + offs, mask=mask, other=0.0)
    i = s.to(tl.int32, bitcast=True) - 1262485504 + azp
    i = tl.minimum(tl.maximum(i, -128), 127)
    tl.store(out_ptr + pid_r * C + offs, i.to(tl.int8), mask=mask)


# ---------------------------------------------------------------------------
# Static (scale given): pure pointwise quantization.
# ---------------------------------------------------------------------------


@triton.jit
def _static_sym_kernel(
    in_ptr,
    out_ptr,
    scale_ptr,
    scale_out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    s = tl.load(scale_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    q = _quant_div_i8(x.to(tl.float32), s)
    tl.store(out_ptr + offs, q, mask=mask)
    if pid == 0:
        tl.store(scale_out_ptr, s)


@triton.jit
def _static_sym_kernel_nomask(
    in_ptr,
    out_ptr,
    scale_ptr,
    scale_out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    # Safe only when N % BLOCK == 0.
    pid = tl.program_id(0)
    s = tl.load(scale_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    q = _quant_div_i8(x.to(tl.float32), s)
    tl.store(out_ptr + offs, q)
    if pid == 0:
        tl.store(scale_out_ptr, s)


@triton.jit
def _static_asym_kernel(
    in_ptr,
    out_ptr,
    scale_ptr,
    azp_ptr,
    scale_out_ptr,
    azp_out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    s = tl.load(scale_ptr)
    azp = tl.load(azp_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    q = _quant_div_azp_i8(x.to(tl.float32), s, azp)
    tl.store(out_ptr + offs, q, mask=mask)
    if pid == 0:
        tl.store(scale_out_ptr, s)
        tl.store(azp_out_ptr, azp)


@triton.jit
def _static_a_sym(in_ptr, t_ptr, scale_ptr, N, BLOCK: tl.constexpr):
    # Tight clamp for the symmetric path: stage B can then drop its int
    # clamp (round(clamp(f,-128.49,127.49)) == clamp(round(f))).
    pid = tl.program_id(0)
    s = tl.load(scale_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    f = tl.minimum(tl.maximum(x.to(tl.float32) / s, -128.49), 127.49)
    tl.store(t_ptr + offs, f + 12582912.0, mask=mask)


@triton.jit
def _static_a_sym_nomask(in_ptr, t_ptr, scale_ptr, N, BLOCK: tl.constexpr):
    # Safe only when N % BLOCK == 0.
    pid = tl.program_id(0)
    s = tl.load(scale_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    f = tl.minimum(tl.maximum(x.to(tl.float32) / s, -128.49), 127.49)
    tl.store(t_ptr + offs, f + 12582912.0)


@triton.jit
def _static_a(in_ptr, t_ptr, scale_ptr, N, BLOCK: tl.constexpr):
    # Loose clamp for the asymmetric path (azp shifts the result out of the
    # int8 range again); stage B_azp keeps its own int clamp.
    pid = tl.program_id(0)
    s = tl.load(scale_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    f = tl.minimum(tl.maximum(x.to(tl.float32) / s, -4194304.0), 4194304.0)
    tl.store(t_ptr + offs, f + 12582912.0, mask=mask)


@triton.jit
def _static_a_nomask(in_ptr, t_ptr, scale_ptr, N, BLOCK: tl.constexpr):
    # Safe only when N % BLOCK == 0.
    pid = tl.program_id(0)
    s = tl.load(scale_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    f = tl.minimum(tl.maximum(x.to(tl.float32) / s, -4194304.0), 4194304.0)
    tl.store(t_ptr + offs, f + 12582912.0)


@triton.jit
def _static_b(t_ptr, out_ptr, scale_ptr, scale_out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    s = tl.load(t_ptr + offs, mask=mask, other=0.0)
    i = s.to(tl.int32, bitcast=True) - 1262485504
    tl.store(out_ptr + offs, i.to(tl.int8), mask=mask)
    if pid == 0:
        tl.store(scale_out_ptr, tl.load(scale_ptr))


@triton.jit
def _static_b_nomask(t_ptr, out_ptr, scale_ptr, scale_out_ptr, N, BLOCK: tl.constexpr):
    # Safe only when N % BLOCK == 0.  Stage A's tight clamp guarantees the
    # magic result is already inside [-128, 127].
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(t_ptr + offs)
    i = s.to(tl.int32, bitcast=True) - 1262485504
    tl.store(out_ptr + offs, i.to(tl.int8))
    if pid == 0:
        tl.store(scale_out_ptr, tl.load(scale_ptr))


@triton.jit
def _static_b_azp(
    t_ptr,
    out_ptr,
    scale_ptr,
    azp_ptr,
    scale_out_ptr,
    azp_out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    azp = tl.load(azp_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    s = tl.load(t_ptr + offs, mask=mask, other=0.0)
    i = s.to(tl.int32, bitcast=True) - 1262485504 + azp
    i = tl.minimum(tl.maximum(i, -128), 127)
    tl.store(out_ptr + offs, i.to(tl.int8), mask=mask)
    if pid == 0:
        tl.store(scale_out_ptr, tl.load(scale_ptr))
        tl.store(azp_out_ptr, azp)


def _next_pow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


_RPR = 256
_K1_BLOCK_CAP = 8192
_K2_BLOCK_R = 16
_2PASS_N_DYN = 512 * 1024
# Static two-pass wins even at small N: the staged sa+sb kernels dodge the
# expensive register bitcast entirely, and two light launches (measured
# 0.0143ms at N=13824 vs 0.0192ms single-pass) beat one heavy fused kernel.
_2PASS_N_STAT = 4096


def _stage1_plan(R, C):
    """k1 (per-row absmax) plan.  Returns (kind, nchunks, params):
      - ('tile', 1, (RT, BLOCK)): single-chunk (RT, BLOCK) tile
      - ('split', 2, (RT, TAIL)): 4096-lane head + exact power-of-2 tail
      - ('grid2d', nchunks, (None, BLOCK)): simple 2D grid for wide rows
    The 8192-lane masked reduction tree is ~3x slower per element than the
    4096-lane one, so rows with 4096 < C < 8192 are split instead of masked.
    """
    BLOCK = min(_K1_BLOCK_CAP, _next_pow2(max(C, 1)))
    nchunks = (C + BLOCK - 1) // BLOCK
    if nchunks == 1 and BLOCK <= 4096:
        RT = 32 if R >= 2048 else 16
        RT = min(RT, _next_pow2(max(R, 1)))
        return ("tile", 1, (RT, BLOCK))
    if nchunks == 1 and BLOCK == 8192 and C > 4096:
        tail = C - 4096
        if tail & (tail - 1) == 0:  # power of 2 -> exact unmasked tail
            return ("split", 2, (16, tail))
    return ("grid2d", nchunks, (None, BLOCK))


def _quant_block3(C):
    p = _next_pow2(max(C, 1))
    return 2048 if p > 4096 else min(4096, p)


def _quant_block3_2p(C):
    # memory-bound two-pass stages: widest single-chunk block (4096 lanes for
    # C<=4096, 8192 for wider rows measured fastest)
    return min(8192, _next_pow2(max(C, 1)))


def scaled_int8_quant(input, scale, azp, symmetric):
    dev = input.device
    shape = input.shape
    C = shape[-1]
    R = 1 if len(shape) == 1 else input.numel() // C
    N = R * C
    out = torch.empty(shape, dtype=torch.int8, device=dev)

    if scale is None:
        # dynamic: per-row statistics
        kind, nchunks, p = _stage1_plan(R, C)
        grid2 = ((R + _RPR - 1) // _RPR,)
        BLOCK3 = _quant_block3_2p(C) if N >= _2PASS_N_DYN else _quant_block3(C)
        nchunks3 = (C + BLOCK3 - 1) // BLOCK3
        two_pass = N >= _2PASS_N_DYN
        if symmetric:
            inv = torch.empty((R,), dtype=torch.float32, device=dev)
            scale_out = torch.empty((R, 1), dtype=torch.float32, device=dev)
            if kind == "tile" and R < 1024:
                # Small-R k2 grids waste 25-75% of their lanes (RPR=256 tile
                # on few rows); folding k2 into k1 removes the launch and the
                # waste (measured: 512x4096 0.126->0.109ms, 64x1024
                # 0.030->0.018ms).  Identical amax/scale/inv chain.
                RT, BLOCK1 = p
                grid1 = ((R + RT - 1) // RT,)
                if C == BLOCK1 and R % RT == 0:
                    _partial_absmax_inv_tile_nomask[grid1](
                        input, scale_out, inv, R, C, RT=RT, BLOCK=BLOCK1
                    )
                else:
                    _partial_absmax_inv_tile[grid1](
                        input, scale_out, inv, R, C, RT=RT, BLOCK=BLOCK1
                    )
            else:
                part = torch.empty((R, nchunks), dtype=torch.float32, device=dev)
                if kind == "tile":
                    RT, BLOCK1 = p
                    grid1 = ((R + RT - 1) // RT,)
                    if C == BLOCK1 and R % RT == 0:
                        _partial_absmax_tile_nomask[grid1](
                            input, part, R, C, RT=RT, BLOCK=BLOCK1
                        )
                    else:
                        _partial_absmax_tile[grid1](
                            input, part, R, C, RT=RT, BLOCK=BLOCK1
                        )
                elif kind == "split":
                    RT, TAIL = p
                    grid1 = ((R + RT - 1) // RT,)
                    _partial_absmax_head_tile[grid1](
                        input, part, R, C, COL=0, RT=RT, BLOCK=4096
                    )
                    _partial_absmax_tail_tile[grid1](
                        input, part, R, C, BASE=4096, COL=1, RT=RT, BLOCK=TAIL
                    )
                else:
                    BLOCK1 = p[1]
                    _partial_absmax_2d[(R, nchunks)](
                        input, part, C, nchunks, BLOCK=BLOCK1
                    )
                _reduce_sym_tile[grid2](
                    part, scale_out, inv, R, nchunks, RPR=_RPR, BLOCK_R=_K2_BLOCK_R
                )
            if two_pass:
                tmp = torch.empty(shape, dtype=torch.float32, device=dev)
                if C == BLOCK3:
                    _quant_a_mul_nomask[(R, nchunks3)](
                        input, tmp, inv, C, nchunks3, BLOCK=BLOCK3
                    )
                    _quant_b_nomask[(R, nchunks3)](tmp, out, C, nchunks3, BLOCK=BLOCK3)
                else:
                    _quant_a_mul[(R, nchunks3)](
                        input, tmp, inv, C, nchunks3, BLOCK=BLOCK3
                    )
                    _quant_b[(R, nchunks3)](tmp, out, C, nchunks3, BLOCK=BLOCK3)
            else:
                if C == BLOCK3:
                    _quant_sym_kernel_nomask[(R, nchunks3)](
                        input, out, inv, C, nchunks3, BLOCK=BLOCK3
                    )
                else:
                    _quant_sym_kernel[(R, nchunks3)](
                        input, out, inv, C, nchunks3, BLOCK=BLOCK3
                    )
            return out, scale_out, None
        else:
            # asymmetric (correctness-only workloads): single-reduction
            # max/min partial kernels + max-only merge reduce.
            if kind == "tile":
                RT, BLOCK1 = p
                grid1 = ((R + RT - 1) // RT,)
                use_tile = True
            else:
                BLOCK1 = 8192 if kind == "split" else p[1]
                grid1 = (R, nchunks if kind == "grid2d" else 1)
                use_tile = False
            part_max = torch.empty((R, nchunks), dtype=torch.float32, device=dev)
            part_min = torch.empty((R, nchunks), dtype=torch.float32, device=dev)
            scale_out = torch.empty((R, 1), dtype=torch.float32, device=dev)
            azp_out = torch.empty((R, 1), dtype=torch.int32, device=dev)
            if use_tile:
                _partial_max_tile[grid1](input, part_max, R, C, RT=RT, BLOCK=BLOCK1)
                _partial_min_tile[grid1](input, part_min, R, C, RT=RT, BLOCK=BLOCK1)
            else:
                _partial_max_2d[grid1](input, part_max, C, nchunks, BLOCK=BLOCK1)
                _partial_min_2d[grid1](input, part_min, C, nchunks, BLOCK=BLOCK1)
            _reduce_asym_tile[grid2](
                part_max,
                part_min,
                scale_out,
                azp_out,
                R,
                nchunks,
                RPR=_RPR,
                BLOCK_R=_K2_BLOCK_R,
            )
            if two_pass:
                tmp = torch.empty(shape, dtype=torch.float32, device=dev)
                _quant_a_div[(R, nchunks3)](
                    input, tmp, scale_out, C, nchunks3, BLOCK=BLOCK3
                )
                _quant_b_azp[(R, nchunks3)](
                    tmp, out, azp_out, C, nchunks3, BLOCK=BLOCK3
                )
            else:
                _quant_asym_kernel[(R, nchunks3)](
                    input, out, scale_out, azp_out, C, nchunks3, BLOCK=BLOCK3
                )
            return out, scale_out, azp_out

    # static: scale given
    if N < _2PASS_N_STAT:
        if N <= 65536:
            cap = 2048
        elif N <= 1048576:
            cap = 4096
        else:
            cap = 8192
        BLOCK = min(cap, _next_pow2(max(N, 1)))
        grid = ((N + BLOCK - 1) // BLOCK,)
        scale_out = torch.empty((1,), dtype=torch.float32, device=dev)
        if symmetric:
            if N % BLOCK == 0:
                _static_sym_kernel_nomask[grid](
                    input, out, scale, scale_out, N, BLOCK=BLOCK
                )
            else:
                _static_sym_kernel[grid](input, out, scale, scale_out, N, BLOCK=BLOCK)
            return out, scale_out, None
        else:
            azp_out = torch.empty((1,), dtype=torch.int32, device=dev)
            _static_asym_kernel[grid](
                input, out, scale, azp, scale_out, azp_out, N, BLOCK=BLOCK
            )
            return out, scale_out, azp_out

    # large static: two-pass staged quantization.  8192-lane blocks stream
    # the f32 temp faster for every two-pass size (sa/sb: 0.056->0.043ms at
    # N=524288, 0.215->0.139ms at N=2M).
    BLOCK = min(8192, _next_pow2(max(N, 1)))
    grid = ((N + BLOCK - 1) // BLOCK,)
    tmp = torch.empty(N, dtype=torch.float32, device=dev)
    scale_out = torch.empty((1,), dtype=torch.float32, device=dev)
    if N % BLOCK == 0:
        if symmetric:
            _static_a_sym_nomask[grid](input, tmp, scale, N, BLOCK=BLOCK)
        else:
            _static_a_nomask[grid](input, tmp, scale, N, BLOCK=BLOCK)
    else:
        if symmetric:
            _static_a_sym[grid](input, tmp, scale, N, BLOCK=BLOCK)
        else:
            _static_a[grid](input, tmp, scale, N, BLOCK=BLOCK)
    if symmetric:
        if N % BLOCK == 0:
            _static_b_nomask[grid](tmp, out, scale, scale_out, N, BLOCK=BLOCK)
        else:
            _static_b[grid](tmp, out, scale, scale_out, N, BLOCK=BLOCK)
        return out, scale_out, None
    else:
        azp_out = torch.empty((1,), dtype=torch.int32, device=dev)
        _static_b_azp[grid](tmp, out, scale, azp, scale_out, azp_out, N, BLOCK=BLOCK)
        return out, scale_out, azp_out

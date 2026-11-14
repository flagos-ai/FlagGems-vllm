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
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255


def _row_grid(m):
    """Spread ``m`` logical rows over a 3-D grid within GCU launch limits.

    The kernel recombines (pid_x, pid_y, pid_z) into a flat row id via
    ``GRID_X``/``GRID_XY`` runtime arguments, so any decomposition with
    ``grid_x * grid_y * grid_z >= m`` is correct; rows beyond ``m`` are
    masked out by the in-kernel ``pid < M`` guard.
    """
    if m <= _GRID_X_MAX:
        return (m, 1, 1)

    grid_y = triton.cdiv(m, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (triton.cdiv(m, grid_y), grid_y, 1)

    # m > 65535 * 255: spill into grid.z as well (supports up to
    # 65535 * 255 * 255 ~= 4.26e9 rows; beyond that the launcher's own
    # OutOfResources check fires, same failure mode as before this split).
    grid_z = triton.cdiv(m, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = triton.cdiv(m, _GRID_X_MAX * grid_z)
    return (triton.cdiv(m, grid_y * grid_z), grid_y, grid_z)


# Row-batch (2D [ROWS_PER_CTA, BLOCK_SIZE] tile) budget.  Two hard caps:
# - tile element budget: keeps the fp32 live set (input + residual + a couple
#   of intermediates) well inside the GCU per-program DSM budget (448KB on
#   Libra; ~3 live fp32 tiles of R*B*4B plus the [BLOCK] weight/bias vectors).
# - rows cap: R >= 32 combined with a 32768-element tile hits a triton_gcu
#   silent-miscompile zone (deterministic wrong values, all num_warps; see
#   compiler_issues/vllm_fused_q_kv_rmsnorm_rowbatch_bcast_miscompile.md,
#   2026-11-08 append).  tile <= 16384 with R <= 32 was verified correct
#   across 225 (shape, dtype, R, num_warps) configurations.
_ROW_TILE_ELEM_MAX = 16384
_ROWS_PER_CTA_CAP = 32
_ROWS_PER_CTA_MIN = 4
# Aim for roughly this many row-batch tiles in flight on the persistent grid
# (L600 has 48 SPs).  With the persistent kernel the launch grid is
# min(_PERSISTENT_CTAS, cdiv(M, R)): every program walks tiles with grid
# stride, so large M no longer scales the CTA count (per-CTA dispatch costs
# ~103-160ns on this backend; 2026-11-11 probe: g48 == g96 == g192 within
# noise on the core shapes, g6-g24 starve DRAM).
_PERSISTENT_CTAS = 48
# The row-batch kernel must launch with num_warps=1: at num_warps >= 2 the
# in-branch 1-D tail body degrades the whole 2-D hot path by 1.3-1.9x
# (measured 2026-11-08; same pathology as the fused_q_kv rmsnorm row-batch
# archive entry, where nw1 was likewise the only correct-and-fast setting).
_ROWS_NUM_WARPS = 1


def _rows_per_cta(m, n):
    """Pick the row-batch factor R for the [R, BLOCK] 2D-tile launch.

    Returns 1 (one row per CTA, the legacy kernel) for inputs whose row
    width already fills the tile budget, and otherwise the largest power of
    two that keeps ~_PERSISTENT_CTAS tiles in flight for small inputs,
    bounded by the verified-safe tile/rows caps.  Large inputs launch only
    _PERSISTENT_CTAS persistent programs regardless, so the tile/rows caps
    dominate there.  Partial tail tiles are handled in-kernel by the 1-D
    row body (correct at every BLOCK width).
    """
    block = triton.next_power_of_2(n)
    tile_cap = _ROW_TILE_ELEM_MAX // max(block, 1)
    if tile_cap < _ROWS_PER_CTA_MIN:
        return 1
    rows = m // _PERSISTENT_CTAS
    rows = 1 << max(0, rows.bit_length() - 1) if rows >= 1 else 1
    rows = max(rows, _ROWS_PER_CTA_MIN)
    if m < rows:
        # Keep at most one partial tail tile in flight.
        rows = 1 << max(0, m.bit_length() - 1)
    return max(1, min(_ROWS_PER_CTA_CAP, tile_cap, rows))


@triton.jit
def _skip_layer_norm_row(
    out_ptr,
    in_ptr,
    res_ptr,
    w_ptr,
    b_ptr,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    r_stride_r,
    r_stride_c,
    row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,  # N == BLOCK_SIZE: no column masking needed
):
    """Normalize a single row (1-D form, proven-correct on this backend)."""
    out_ptr += row * y_stride_r
    in_ptr += row * x_stride_r
    res_ptr += row * r_stride_r
    if EXACT:
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_ptr + cols * x_stride_c).to(tl.float32)
        r = tl.load(res_ptr + cols * r_stride_c).to(tl.float32)
        x += r
        mean = tl.sum(x, axis=0) / N
        _var = x - mean
        _var = _var * _var
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols).to(tl.float32)
        b = tl.load(b_ptr + cols).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        tl.store(
            out_ptr + cols * y_stride_c, y.to(out_ptr.dtype.element_ty)
        )
    else:
        mask = tl.arange(0, BLOCK_SIZE) < N
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        mean = tl.sum(x, axis=0) / N
        _var = tl.where(mask, x - mean, 0.0)
        _var = _var * _var
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        w = tl.load(
            w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0
        ).to(tl.float32)
        b = tl.load(
            b_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0
        ).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        tl.store(
            out_ptr + cols * y_stride_c, y.to(out_ptr.dtype.element_ty), mask=mask
        )


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    M,  # number of rows in X
    GRID_X,
    GRID_XY,
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Legacy one-row-per-CTA path, used when the row width already fills the
    # row-batch tile budget (wide N).  Behavior is identical to the original
    # implementation, including the 3-D grid recombination for large M.
    pid = (
        ext.program_id(0)
        + ext.program_id(1) * GRID_X
        + ext.program_id(2) * GRID_XY
    )

    if pid < M:
        _skip_layer_norm_row(
            Y,
            X,
            R,
            W,
            B,
            y_stride_r,
            y_stride_c,
            x_stride_r,
            x_stride_c,
            r_stride_r,
            r_stride_c,
            pid,
            N,
            eps,
            BLOCK_SIZE,
            False,
        )


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_rows_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    M,  # number of rows in X
    NUM_TILES,  # number of [ROWS_PER_CTA, BLOCK_SIZE] tiles
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    EXACT: tl.constexpr,  # N == BLOCK_SIZE: no column masking needed
    NUM_PROGS: tl.constexpr,  # persistent grid width (grid = NUM_PROGS)
):
    # Persistent grid-stride kernel: each of the NUM_PROGS programs walks the
    # [ROWS_PER_CTA, BLOCK_SIZE] tiles with stride NUM_PROGS, so the launch
    # grid stays at <= 48 CTAs no matter how large M is.  Per-CTA dispatch on
    # this backend costs ~103-160ns (measured 2026-11-11 on GCU400/L600, same
    # "micro-CTA dispatch floor" as the fused_deepseek_v4_qnorm_rope /
    # qkv_lora_b cases), which made the previous one-tile-per-CTA launch pay
    # 65536 * ~103ns ~= 6.8ms of pure dispatch on the (1024,1024,1024) core
    # shape.  The loop also lets the loop-invariant weight/bias vectors be
    # loaded once per program instead of once per CTA.
    pid = ext.program_id(0)

    # Loop-invariant hoist: W/B are shared by every tile this program
    # processes.  Loading them before the loop removes one L2 load pair per
    # tile from the steady-state iteration (probe-verified 2026-11-11: same
    # outputs, -1.2% latency on the large shapes).
    if EXACT:
        wcols = tl.arange(0, BLOCK_SIZE)
        w = tl.load(W + wcols).to(tl.float32)
        b = tl.load(B + wcols).to(tl.float32)
    else:
        wmask = tl.arange(0, BLOCK_SIZE) < N
        wcols = tl.arange(0, BLOCK_SIZE)
        w = tl.load(W + wcols, mask=wmask, other=0.0).to(tl.float32)
        b = tl.load(B + wcols, mask=wmask, other=0.0).to(tl.float32)

    for tile in range(pid, NUM_TILES, NUM_PROGS):
        row0 = tile * ROWS_PER_CTA
        # Full-tile row-batch body.  Partial tail tiles are handled by the 1-D
        # row body below: every 2-D masked/anchored tail variant is a
        # silent-miscompile zone at BLOCK 512/1024 on this backend, while the
        # 1-D form is correct at every width - but it must be launched with
        # num_warps=1, or the 2-D hot path above degrades 1.3-1.9x (see
        # compiler_issues/vllm_fused_q_kv_rmsnorm_rowbatch_bcast_miscompile.md).
        if row0 + ROWS_PER_CTA <= M:
            rows = row0 + tl.arange(0, ROWS_PER_CTA)
            cols = tl.arange(0, BLOCK_SIZE)
            x_ptrs = X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c
            r_ptrs = R + rows[:, None] * r_stride_r + cols[None, :] * r_stride_c
            y_ptrs = Y + rows[:, None] * y_stride_r + cols[None, :] * y_stride_c
            if EXACT:
                x = tl.load(x_ptrs).to(tl.float32)
                r = tl.load(r_ptrs).to(tl.float32)
                x += r
                mean = tl.sum(x, axis=1) / N
                _var = x - mean[:, None]
                _var = _var * _var
                var = tl.sum(_var, axis=1) / N
                rstd = 1 / tl.sqrt(var + eps)
                x_hat = (x - mean[:, None]) * rstd[:, None]
                y = w[None, :] * x_hat + b[None, :]
                tl.store(y_ptrs, y.to(Y.dtype.element_ty))
            else:
                col_mask = cols < N
                x = tl.load(x_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
                r = tl.load(r_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
                x += r
                mean = tl.sum(x, axis=1) / N
                _var = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
                _var = _var * _var
                var = tl.sum(_var, axis=1) / N
                rstd = 1 / tl.sqrt(var + eps)
                x_hat = (x - mean[:, None]) * rstd[:, None]
                y = w[None, :] * x_hat + b[None, :]
                tl.store(
                    y_ptrs, y.to(Y.dtype.element_ty), mask=col_mask[None, :]
                )
        else:
            # Tail tile (partial rows / grid overshoot): normalize the leftover
            # rows one at a time with the 1-D form (correct at every BLOCK
            # width; the 2-D masked-tile forms are miscompile zones at BLOCK
            # 512/1024 on this backend).
            if row0 < M:
                for i in tl.static_range(ROWS_PER_CTA):
                    row = row0 + i
                    if row < M:
                        _skip_layer_norm_row(
                            Y,
                            X,
                            R,
                            W,
                            B,
                            y_stride_r,
                            y_stride_c,
                            x_stride_r,
                            x_stride_c,
                            r_stride_r,
                            r_stride_c,
                            row,
                            N,
                            eps,
                            BLOCK_SIZE,
                            EXACT,
                        )


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logger.debug("GEMS SKIP LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = triton.next_power_of_2(N)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        rows_per_cta = _rows_per_cta(M, N)

        with torch_device_fn.device(x.device):
            if rows_per_cta > 1:
                num_tiles = triton.cdiv(M, rows_per_cta)
                # Persistent grid-stride launch: at most _PERSISTENT_CTAS
                # programs regardless of M, so the CTA count (and its
                # ~103-160ns/CTA dispatch cost) no longer scales with M.
                # grid.x <= 48 is far inside the GCU launch limits, so no 3-D
                # grid splitting is needed on this path (the legacy kernel
                # below keeps _row_grid for its own large-M handling).
                num_progs = min(_PERSISTENT_CTAS, num_tiles)
                grid = (num_progs, 1, 1)
                skip_layer_norm_rows_kernel[grid](
                    y,
                    x,
                    residual,
                    weight,
                    bias,
                    N,
                    1,
                    N,
                    1,
                    N,
                    1,
                    N,
                    M,
                    num_tiles,
                    eps,
                    BLOCK_SIZE,
                    ROWS_PER_CTA=rows_per_cta,
                    EXACT=(N == BLOCK_SIZE),
                    NUM_PROGS=num_progs,
                    num_warps=_ROWS_NUM_WARPS,
                )
            else:
                grid_x, grid_y, grid_z = _row_grid(M)
                grid_xy = grid_x * grid_y
                grid = (grid_x, grid_y, grid_z)
                skip_layer_norm_kernel[grid](
                    y,
                    x,
                    residual,
                    weight,
                    bias,
                    N,
                    1,
                    N,
                    1,
                    N,
                    1,
                    N,
                    M,
                    grid_x,
                    grid_xy,
                    eps,
                    BLOCK_SIZE,
                )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)

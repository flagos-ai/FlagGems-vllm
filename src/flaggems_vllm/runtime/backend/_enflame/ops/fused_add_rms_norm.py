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

import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tuned constants
# -----------------------------------------------------------------------------
_WHOLE_ROW_SAFE_N = 16384
_LARGE_N_THRESHOLD = 32768

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255

# Persistent row-batch path (L600/48SP, measured 2026-11-08, retuned 2026-11-12):
#   - fixed grid of 48 programs (one per SP), num_warps=2, 2-stage software
#     pipeline over the block loop;
#   - sweet-spot tile is dtype-dependent (measured 2026-11-12): fp16/bf16
#     programs reach 1.5-1.9TB/s with a 32768-element tile (R * BLOCK_N) at
#     320KB DSM, comfortably inside the 448KB dual-program line; fp32 already
#     sits at the 448KB line with a 16384-element tile, so 16384 stays the fp32
#     tile (32768 would push fp32 to 544KB and drop dual-program concurrency,
#     measured +14~21% on the fp32 core shapes);
#   - the one-row-per-CTA whole-row kernel only reaches 66-132GB/s on
#     large-M shapes (per-CTA dispatch overhead ~3us dominates), hence this
#     persistent arm for every shape with enough total work.
#   - R is additionally shrunk (down to 2) so that at least 8 blocks exist,
#     keeping enough programs busy for small M; below 64Ki total elements the
#     launch-latency-bound whole-row kernel is kept instead.
#   - R is capped at 32 rows to stay inside the empirically-validated safe
#     envelope of the 2D-tile broadcast-chain miscompile family (see
#     compiler_issues/vllm_fused_q_kv_rmsnorm_rowbatch_bcast_miscompile.md:
#     tile<=16384 with R<=32 verified green across sibling configs, while
#     R>=32 at tile=32768 miscompiles there).  No bench shape needs R>32.
_PERSIST_TILE_ELEMS = 16384
_PERSIST_TILE_ELEMS_WIDE = 32768
_PERSIST_MAX_ROWS = 32
_PERSIST_NUM_PROGRAMS = 48
_PERSIST_NUM_WARPS = 2
_PERSIST_MIN_BLOCKS = 8
_PERSIST_MIN_WORK_ELEMS = 65536


def _row_grid(m):
    """Spread ``m`` logical rows over a 3-D grid within GCU launch limits.

    The kernels recombine (pid_x, pid_y, pid_z) into a flat row id via
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


# -----------------------------------------------------------------------------
# 1) Original whole-row baseline
# -----------------------------------------------------------------------------
@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    N,
    M,
    GRID_X,
    GRID_XY,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    # Recombine the 3-D launch grid into a flat row id.  grid.x is capped at
    # 65535 (grid.y/grid.z at 255) on the GCU, so large M is spread across
    # the remaining axes by _row_grid() on the host side.
    pid = ext.program_id(0) + ext.program_id(1) * GRID_X + ext.program_id(2) * GRID_XY

    if pid < M:
        input_ptr += pid * in_stride_r
        residual_ptr += pid * r_stride_r

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(
            input_ptr + cols * in_stride_c,
            mask=mask,
            other=0.0,
        ).to(cdtype)

        r = tl.load(
            residual_ptr + cols * r_stride_c,
            mask=mask,
            other=0.0,
        ).to(cdtype)

        x += r

        tl.store(
            residual_ptr + cols * r_stride_c,
            x,
            mask=mask,
        )

        var = tl.sum(x * x / N, axis=0)
        rrms = 1.0 / tl.sqrt(var + eps)

        w = tl.load(
            w_ptr + cols,
            mask=mask,
            other=0.0,
        )

        y = (x * rrms * w).to(cdtype)

        tl.store(
            input_ptr + cols * in_stride_c,
            y,
            mask=mask,
        )


# -----------------------------------------------------------------------------
# 2) Single-CTA tiled two-pass path with Triton autotune
# -----------------------------------------------------------------------------
def _get_tiled_autotune_configs():
    # Tune TILE_SIZE, num_warps and num_stages jointly.
    #
    # LOOP_NUM_STAGES mirrors Config.num_stages into tl.range(...).  This is
    # intentional: tl.range(num_stages=...) controls software pipelining for
    # loads in this non-dot reduction loop, while Config.num_stages is also
    # supplied as the normal Triton compilation option.
    return [
        triton.Config(
            {
                "TILE_SIZE": tile_size,
                "LOOP_NUM_STAGES": num_stages,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for tile_size in (256, 512, 1024, 2048)
        for num_warps in (2, 4, 8)
        for num_stages in (1, 2, 3)
    ]


@libentry()
@triton.autotune(
    configs=_get_tiled_autotune_configs(),
    key=["M", "N"],
    # This kernel updates input_ptr and residual_ptr in-place.  Autotune runs
    # every candidate multiple times, so restore the original tensors before
    # benchmarking each candidate to keep all configs semantically equivalent.
    restore_value=["input_ptr", "residual_ptr"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_tiled_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    M,
    N,
    GRID_X,
    GRID_XY,
    eps,
    TILE_SIZE: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        compute_dtype = tl.float32
    else:
        compute_dtype = input_ptr.dtype.element_ty

    # Recombine the 3-D launch grid into a flat row id (see
    # fused_add_rms_norm_kernel).
    pid = ext.program_id(0) + ext.program_id(1) * GRID_X + ext.program_id(2) * GRID_XY

    if pid < M:
        input_row = input_ptr + pid * in_stride_r
        residual_row = residual_ptr + pid * r_stride_r

        offsets = tl.arange(0, TILE_SIZE)
        sum_sq = tl.zeros([1], dtype=compute_dtype)

        # Pass 1: compute sum of squares and immediately materialize
        # added = input + residual into residual.  This removes the
        # second-pass input load and the repeated addition for all
        # supported dtypes.
        for tile_start in tl.range(
            0,
            N,
            TILE_SIZE,
            num_stages=LOOP_NUM_STAGES,
        ):
            cols = tile_start + offsets
            mask = cols < N

            x = tl.load(
                input_row + cols * in_stride_c,
                mask=mask,
                other=0.0,
            ).to(compute_dtype)

            residual = tl.load(
                residual_row + cols * r_stride_c,
                mask=mask,
                other=0.0,
            ).to(compute_dtype)

            added = x + residual
            sum_sq += tl.sum(added * added, axis=0)

            tl.store(
                residual_row + cols * r_stride_c,
                added,
                mask=mask,
            )

        variance = sum_sq / N
        rrms = 1.0 / tl.sqrt(variance + eps)

        # Pass 2: read the already-updated residual and produce only the
        # output.
        for tile_start in tl.range(
            0,
            N,
            TILE_SIZE,
            num_stages=LOOP_NUM_STAGES,
        ):
            cols = tile_start + offsets
            mask = cols < N

            added = tl.load(
                residual_row + cols * r_stride_c,
                mask=mask,
                other=0.0,
            ).to(compute_dtype)

            weight = tl.load(
                w_ptr + cols,
                mask=mask,
                other=0.0,
            )

            output = (added * rrms * weight).to(compute_dtype)

            tl.store(
                input_row + cols * in_stride_c,
                output,
                mask=mask,
            )


def _prefer_whole_row(N):
    """Empirical whole-row selector without the removed Cluster paths."""
    if N <= _WHOLE_ROW_SAFE_N:
        return True

    # Just above 16384 is a severe next_power_of_2 cliff.
    if N < 24576:
        return False

    # BLOCK_SIZE=32768 remains efficient in this region.
    if N <= _LARGE_N_THRESHOLD:
        return True

    # Once next_power_of_2(N) is 65536, use the tiled path.
    return False


# -----------------------------------------------------------------------------
# 3) Persistent row-batched path for large-M whole-row shapes
# -----------------------------------------------------------------------------
@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_persist_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    r_stride_r,
    M,
    N,
    NUM_BLOCKS,
    eps,
    BLOCK_N: tl.constexpr,
    R: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EXACT_COLS: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    # The weight is loaded with a 2-D shape ([1, BLOCK_N]) on purpose: a 1-D
    # fp32 load broadcast as a 2D-tile multiply operand hits a known silent
    # miscompile (compiler_issues/triton_fp32_1d_bcast_miscompile.md, sibling
    # add_rms_norm row-batch kernel).  The 2-D-shaped load dodges the bad
    # layout path at zero extra cost and is correct for every dtype.
    if EXACT_COLS:
        w = tl.load(w_ptr + cols[None, :])
    else:
        w = tl.load(
            w_ptr + cols[None, :],
            mask=(cols < N)[None, :],
            other=0.0,
        )

    # Strided persistent loop over [R, BLOCK_N] row tiles.  Different blocks
    # touch disjoint rows, so the 2-stage software pipeline can overlap the
    # next block's loads with the current block's compute/stores safely.
    for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=2):
        row0 = blk * R
        rows = row0 + tl.arange(0, R)
        in_off = rows[:, None] * in_stride_r + cols[None, :]
        r_off = rows[:, None] * r_stride_r + cols[None, :]

        if EXACT_COLS and EVEN_ROWS:
            x = tl.load(input_ptr + in_off).to(cdtype)
            r = tl.load(residual_ptr + r_off).to(cdtype)
            added = x + r
            tl.store(residual_ptr + r_off, added)
            var = tl.sum(added * added, axis=1) / N
            rrms = 1.0 / tl.sqrt(var + eps)
            y = (added * rrms[:, None] * w).to(cdtype)
            tl.store(input_ptr + in_off, y)
        else:
            if EVEN_ROWS:
                mask = (cols < N)[None, :]
            elif EXACT_COLS:
                mask = (rows < M)[:, None]
            else:
                mask = (rows < M)[:, None] & (cols < N)[None, :]
            x = tl.load(input_ptr + in_off, mask=mask, other=0.0).to(cdtype)
            r = tl.load(residual_ptr + r_off, mask=mask, other=0.0).to(cdtype)
            added = x + r
            tl.store(residual_ptr + r_off, added, mask=mask)
            var = tl.sum(added * added, axis=1) / N
            rrms = 1.0 / tl.sqrt(var + eps)
            y = (added * rrms[:, None] * w).to(cdtype)
            tl.store(input_ptr + in_off, y, mask=mask)


def _persist_rows_per_cta(M, block_n, tile_elems=_PERSIST_TILE_ELEMS):
    """Rows per persistent block, from the dtype-dependent tile sweet spot.

    The tile size is additionally shrunk (down to 2 rows) when M is too small
    to produce at least 8 blocks, so enough of the 48 programs stay busy, and
    capped at 32 rows to stay inside the validated broadcast-chain envelope.
    """
    rows_per_cta = min(tile_elems // block_n, _PERSIST_MAX_ROWS)
    if rows_per_cta < 2:
        return 1
    while rows_per_cta > 2 and triton.cdiv(M, rows_per_cta) < _PERSIST_MIN_BLOCKS:
        rows_per_cta //= 2
    return rows_per_cta


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """Fused residual addition + RMSNorm, in-place.

    Dispatch:
      A) N <= 16384 and enough total work
           -> persistent row-batched kernel: fixed 48-program grid, each
              program strides over [R, BLOCK_N] row tiles.  The tile holds
              32768 elements for fp16/bf16 (320KB DSM, dual-program kept)
              and 16384 for fp32 (at the 448KB dual-program line), so
              R >= 2 requires BLOCK_N <= 16384 (fp16/bf16) / 8192 (fp32).

      B) N <= 16384, small work (or BLOCK_N so large that R would be 1:
             > 16384 for fp16/bf16, > 8192 for fp32)
           -> whole-row baseline (one row per CTA).

      C) 16384 < N < 24576
           -> Single-CTA tiled.

      D) 24576 <= N <= 32768
           -> whole-row baseline.

      E) N > 32768
           -> Single-CTA tiled.

    Rows are spread over a 3-D grid (grid.x <= 65535, grid.y/z <= 255) when
    M exceeds the GCU grid.x limit; the kernels recombine the axes into a
    flat row id.
    """
    logger.debug(
        "GEMS FUSED_ADD_RMS_NORM FORWARD, "
        "[input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    prefer_whole = _prefer_whole_row(N)

    with torch_device_fn.device(x.device):
        block_n = triton.next_power_of_2(N)
        # DType-dependent tile sweet spot (measured 2026-11-12): fp16/bf16
        # programs at a 32768-element tile use 320KB DSM, inside the 448KB
        # dual-program line, and are 11-26% faster than the 16384-element tile
        # on the core shapes; fp32 already sits at the 448KB line with 16384
        # elements and would lose dual-program concurrency at 32768 (544KB,
        # measured +14~21%), so fp32 keeps the smaller tile.  The tile math is
        # only paid on the candidate persist path so the whole-row/tiled arms
        # keep their original host overhead.
        use_persist = prefer_whole and M * N >= _PERSIST_MIN_WORK_ELEMS
        rows_per_cta = 1
        if use_persist:
            tile_elems = (
                _PERSIST_TILE_ELEMS_WIDE
                if x.element_size() < 4
                else _PERSIST_TILE_ELEMS
            )
            rows_per_cta = _persist_rows_per_cta(M, block_n, tile_elems)
            use_persist = rows_per_cta >= 2

        if use_persist:
            fused_add_rms_norm_persist_kernel[(_PERSIST_NUM_PROGRAMS, 1, 1)](
                x,
                residual,
                weight,
                N,
                N,
                M,
                N,
                triton.cdiv(M, rows_per_cta),
                eps,
                BLOCK_N=block_n,
                R=rows_per_cta,
                NPROG=_PERSIST_NUM_PROGRAMS,
                EVEN_ROWS=(M % rows_per_cta == 0),
                EXACT_COLS=(block_n == N),
                num_warps=_PERSIST_NUM_WARPS,
            )

        elif prefer_whole:
            # Keep grid.x within the GCU 65535 limit (grid.y/z within 255) for
            # large M by spreading rows over a 3-D grid; the kernels recombine
            # the axes into a flat row id.
            grid_x, grid_y, grid_z = _row_grid(M)
            grid_xy = grid_x * grid_y
            grid = (grid_x, grid_y, grid_z)

            block_size = triton.next_power_of_2(N)

            fused_add_rms_norm_kernel[grid](
                x,
                residual,
                weight,
                N,
                1,
                N,
                1,
                N,
                M,
                grid_x,
                grid_xy,
                eps,
                BLOCK_SIZE=block_size,
            )

        else:
            grid_x, grid_y, grid_z = _row_grid(M)
            grid_xy = grid_x * grid_y
            grid = (grid_x, grid_y, grid_z)

            fused_add_rms_norm_tiled_kernel[grid](
                x,
                residual,
                weight,
                N,
                1,
                N,
                1,
                M,
                N,
                grid_x,
                grid_xy,
                eps,
            )

    return x, residual

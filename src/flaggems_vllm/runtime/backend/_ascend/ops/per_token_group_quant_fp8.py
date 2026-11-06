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
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils.device_info import get_sm_count

if torch_device_fn.is_available():
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


@triton.jit
def _f32_to_fp8_e4m3fn(y):
    """Bit-exact f32 -> e4m3fn conversion (RNE); `y` must be finite and
    pre-clamped to [-448, 448].

    Ascend has no native fp8 conversion instruction and BiShengHIR cannot
    lower ``.to(float8_e4m3fn)`` at all, so this branchless sequence needs
    only a few integer ops per element:
      * normals (|y| >= 2^-6): rebias the exponent (127 -> 7) and RNE-round
        the mantissa from 23 to 3 bits via ``t += 0x7FFFF + lsb_of_result``;
      * subnormals: ``k = RNE(|y| / 2**-9)`` with the magic-number add
        ``|y| * 512 + 2**23`` (an f32 add rounds to-nearest-even, leaving k
        in the low mantissa bits).

    int32 stands in for uint32 because the Ascend compiler rejects
    ``tl.where`` on uint32 (it would widen it to uint64 with rint mode).
    The result is identical: ``a`` is always non-negative so signed
    compares match unsigned ones, ``r_norm`` is only selected where ``t``
    is positive (arithmetic shift == logical shift there), and the
    ``& 1`` / ``& 0x80`` bit extractions read the same two's-complement
    bits under either shift semantics.
    """
    b = y.to(tl.int32, bitcast=True)
    a = b & 0x7FFFFFFF
    t = a - 0x3C000000
    t += 0x0007FFFF + ((t >> 20) & 1)
    r_norm = t >> 20
    r_sub = (a.to(tl.float32, bitcast=True) * 512.0 + 8388608.0).to(
        tl.int32, bitcast=True
    ) - 0x4B000000
    r = tl.where(a >= 0x3C800000, r_norm, r_sub)
    return (r | ((b >> 24) & 0x80)).to(tl.uint8)


@triton.jit
def _quant_groups(
    y,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    inv_fp8_max: tl.constexpr,
    scale_ue8m0: tl.constexpr,
    subnormal_scale: tl.constexpr,
):
    """Per-row (axis=1) abs-max scale + quantize, in f32."""
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _absmax * inv_fp8_max

    if scale_ue8m0:
        # Round the scale up to a power of two via exponent-field arithmetic
        # instead of exp2(ceil(log2(x))): on Ascend the libdevice log2/exp2
        # sequence lands one ulp below the exact power of two.  `y_s` is
        # positive and normal (>= eps * inv_fp8_max), so adding all-ones to
        # the mantissa field carries into the exponent exactly when the
        # mantissa is non-zero, i.e. computes 2**ceil(log2(y_s)) bit-exactly.
        s_bits = (y_s.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000
        y_s = s_bits.to(tl.float32, bitcast=True)
        # The inverse of 2**(E-127) is 2**(127-E), whose biased exponent
        # field is 254 - E; exact unless the result would be subnormal.
        inv_s = (0x7F000000 - s_bits).to(tl.float32, bitcast=True)
        # y_s is a power of two, so multiplying by inv_s is a bit-exact
        # division and avoids one f32 divide per element.
        y_q = tl.clamp(y * inv_s[:, None], fp8_min, fp8_max)
    elif subnormal_scale:
        # `y_s` is subnormal when `fp8_max` is huge (e.g. fp32), and some
        # backends flush subnormal fp32 divisors to zero; divide by the always
        # normal `_absmax` instead and scale at the end.
        y_q = tl.clamp((y / _absmax[:, None]) * fp8_max, fp8_min, fp8_max)
    else:
        # one reciprocal per group instead of one divide per element
        y_q = tl.clamp(y * (1.0 / y_s)[:, None], fp8_min, fp8_max)
    return y_q, y_s


@triton.jit
def _per_token_group_quant_fp8_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_row_stride,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    inv_fp8_max: tl.constexpr,
    scale_ue8m0: tl.constexpr,
    subnormal_scale: tl.constexpr,
    fast_cvt: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    programs_per_row = GROUPS_PER_ROW // NGROUPS
    row = pid // programs_per_row
    pg = pid % programs_per_row

    start_gid = row * GROUPS_PER_ROW + pg * NGROUPS

    gids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (
        row.to(tl.int64) * y_row_stride
        + (pg.to(tl.int64) * NGROUPS + gids[:, None]) * GROUP_SIZE
        + cols[None, :]
    )
    if BLOCK == GROUP_SIZE:
        y = tl.load(y_ptr + offsets).to(tl.float32)
    else:
        y = tl.load(y_ptr + offsets, mask=cols[None, :] < GROUP_SIZE, other=0.0).to(
            tl.float32
        )

    y_q, y_s = _quant_groups(
        y, eps, fp8_min, fp8_max, inv_fp8_max, scale_ue8m0, subnormal_scale
    )
    if fast_cvt:
        # raw uint8 bit patterns; the Ascend compiler has no fp8-typed
        # store, so the host passes `y_q_ptr` as a uint8 view of the fp8
        # output tensor.  Its i32 -> u8 cast lowering also inflates the
        # local buffer past the UB budget on 2D tiles, so the conversion
        # runs on the flattened 1D tile.
        y_q = tl.reshape(
            _f32_to_fp8_e4m3fn(tl.reshape(y_q, (NGROUPS * BLOCK,))),
            (NGROUPS, BLOCK),
        )
    else:
        y_q = y_q.to(y_q_ptr.dtype.element_ty)

    out_offsets = (
        start_gid.to(tl.int64) * GROUP_SIZE + gids[:, None] * GROUP_SIZE + cols[None, :]
    )
    if BLOCK == GROUP_SIZE:
        tl.store(y_q_ptr + out_offsets, y_q)
    else:
        tl.store(y_q_ptr + out_offsets, y_q, mask=cols[None, :] < GROUP_SIZE)
    tl.store(y_s_ptr + start_gid + gids, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    inv_fp8_max: tl.constexpr,
    scale_ue8m0: tl.constexpr,
    subnormal_scale: tl.constexpr,
    fast_cvt: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    programs_per_row = GROUPS_PER_ROW // NGROUPS
    row = pid // programs_per_row
    pg = pid % programs_per_row

    start_gid = row * GROUPS_PER_ROW + pg * NGROUPS

    gids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (
        row.to(tl.int64) * y_row_stride
        + (pg.to(tl.int64) * NGROUPS + gids[:, None]) * GROUP_SIZE
        + cols[None, :]
    )
    if BLOCK == GROUP_SIZE:
        y = tl.load(y_ptr + offsets).to(tl.float32)
    else:
        y = tl.load(y_ptr + offsets, mask=cols[None, :] < GROUP_SIZE, other=0.0).to(
            tl.float32
        )

    y_q, y_s = _quant_groups(
        y, eps, fp8_min, fp8_max, inv_fp8_max, scale_ue8m0, subnormal_scale
    )
    if fast_cvt:
        # raw uint8 bit patterns; the Ascend compiler has no fp8-typed
        # store, so the host passes `y_q_ptr` as a uint8 view of the fp8
        # output tensor.  Its i32 -> u8 cast lowering also inflates the
        # local buffer past the UB budget on 2D tiles, so the conversion
        # runs on the flattened 1D tile.
        y_q = tl.reshape(
            _f32_to_fp8_e4m3fn(tl.reshape(y_q, (NGROUPS * BLOCK,))),
            (NGROUPS, BLOCK),
        )
    else:
        y_q = y_q.to(y_q_ptr.dtype.element_ty)

    out_offsets = (
        start_gid.to(tl.int64) * GROUP_SIZE + gids[:, None] * GROUP_SIZE + cols[None, :]
    )
    scale_offsets = (pg * NGROUPS + gids) * y_s_col_stride + row
    if BLOCK == GROUP_SIZE:
        tl.store(y_q_ptr + out_offsets, y_q)
    else:
        tl.store(y_q_ptr + out_offsets, y_q, mask=cols[None, :] < GROUP_SIZE)
    tl.store(y_s_ptr + scale_offsets, y_s)


@triton.jit
def _per_token_group_quant_fp8_persistent(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    n_groups,
    n_tasks,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    inv_fp8_max: tl.constexpr,
    scale_ue8m0: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    NGROUPS: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    """Persistent flat kernel for contiguous inputs: groups are laid out
    linearly in memory, so a task is one tile of NGROUPS consecutive groups
    with pure linear addressing (no row decomposition).  A fixed grid of
    workers strides over all tasks, amortizing program launch over the whole
    tensor instead of paying it per tile.

    Only the e4m3 fast-conversion path reaches this kernel; the host gates
    on ``fast_cvt``.
    """
    pid = tl.program_id(0)
    n_workers = tl.num_programs(0)
    tasks_per_worker = tl.cdiv(n_tasks, n_workers)

    gids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    inner = gids[:, None] * GROUP_SIZE + cols[None, :]
    TILE: tl.constexpr = NGROUPS * GROUP_SIZE

    for i in range(tasks_per_worker):
        task_id = pid + i * n_workers
        if task_id < n_tasks:
            base = task_id.to(tl.int64) * TILE
            offsets = base + inner
            sg = task_id * NGROUPS + gids

            if BLOCK == GROUP_SIZE and not HAS_TAIL:
                y = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(
                    tl.float32
                )
            else:
                m = cols[None, :] < GROUP_SIZE
                if HAS_TAIL:
                    m = m & (sg < n_groups)[:, None]
                y = tl.load(
                    y_ptr + offsets, mask=m, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)

            y_q, y_s = _quant_groups(
                y, eps, fp8_min, fp8_max, inv_fp8_max, scale_ue8m0, False
            )
            # raw uint8 bit patterns through the flattened 1D tile; see
            # _per_token_group_quant_fp8_vec
            y_q8 = tl.reshape(
                _f32_to_fp8_e4m3fn(tl.reshape(y_q, (TILE,))),
                (NGROUPS, BLOCK),
            )

            if BLOCK == GROUP_SIZE and not HAS_TAIL:
                tl.store(y_q_ptr + offsets, y_q8, eviction_policy="evict_first")
                tl.store(y_s_ptr + sg, y_s)
            else:
                tl.store(y_q_ptr + offsets, y_q8, mask=m)
                tl.store(y_s_ptr + sg, y_s, mask=sg < n_groups)


@triton.jit
def _per_token_group_quant_fp8_slice_persistent(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    n_groups,
    n_tasks,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    inv_fp8_max: tl.constexpr,
    scale_ue8m0: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT: tl.constexpr,
    SUB: tl.constexpr,
    NGROUPS: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    """Persistent wide-group path (GROUP_SIZE >= 512, multiple of SUB=32):
    each group is SPLIT rows of SUB elements inside a (NGROUPS*SPLIT, SUB)
    tile, so the group abs-max is a two-stage 2D reduction and no tile
    dimension is wide.  Wide single-row tiles (e.g. (1, 512)) carry a fixed
    UB cost proportional to the row width on Ascend, which caps such tiles
    at one group per program; slicing keeps the per-element UB cost of
    narrow tiles.  Workers stride over tasks as in the flat kernel.
    """
    pid = tl.program_id(0)
    n_workers = tl.num_programs(0)
    tasks_per_worker = tl.cdiv(n_tasks, n_workers)

    ROWS: tl.constexpr = NGROUPS * SPLIT
    TILE: tl.constexpr = NGROUPS * GROUP_SIZE
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, SUB)
    inner = rows[:, None] * SUB + cols[None, :]
    gids = tl.arange(0, NGROUPS)

    for i in range(tasks_per_worker):
        task_id = pid + i * n_workers
        if task_id < n_tasks:
            base = task_id.to(tl.int64) * TILE
            offsets = base + inner
            sg = task_id * NGROUPS + gids

            if HAS_TAIL:
                rmask = tl.reshape(
                    tl.broadcast_to((sg < n_groups)[:, None], (NGROUPS, SPLIT)),
                    (ROWS,),
                )
                y = tl.load(
                    y_ptr + offsets,
                    mask=rmask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                y = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(
                    tl.float32
                )

            row_max = tl.max(tl.abs(y), axis=1)  # (ROWS,)
            g_absmax = tl.maximum(
                tl.max(tl.reshape(row_max, (NGROUPS, SPLIT)), axis=1), eps
            )
            y_s = g_absmax * inv_fp8_max
            if scale_ue8m0:
                # bit-exact 2**ceil(log2(y_s)); see _quant_groups
                s_bits = (y_s.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000
                y_s = s_bits.to(tl.float32, bitcast=True)
                inv_s = (0x7F000000 - s_bits).to(tl.float32, bitcast=True)
            else:
                inv_s = 1.0 / y_s
            inv_rows = tl.reshape(
                tl.broadcast_to(inv_s[:, None], (NGROUPS, SPLIT)), (ROWS,)
            )
            y_q = tl.clamp(y * inv_rows[:, None], fp8_min, fp8_max)

            # raw uint8 bit patterns through the flattened 1D tile; see
            # _per_token_group_quant_fp8_vec
            y_q8 = tl.reshape(
                _f32_to_fp8_e4m3fn(tl.reshape(y_q, (TILE,))),
                (ROWS, SUB),
            )
            if HAS_TAIL:
                tl.store(y_q_ptr + offsets, y_q8, mask=rmask[:, None])
                tl.store(y_s_ptr + sg, y_s, mask=sg < n_groups)
            else:
                tl.store(y_q_ptr + offsets, y_q8, eviction_policy="evict_first")
                tl.store(y_s_ptr + sg, y_s)


def _ngroups_per_program(
    total_groups: int, groups_per_row: int, group_size: int
) -> int:
    # Fuse neighbouring groups into one ~512-element tile per program: wider
    # tiles blow past the Ascend UB budget (192KB) once the bit-trick
    # conversion's temporaries are accounted for (a 2x512 tile needs >200KB).
    cap = max(1, 512 // group_size)
    ngroups = 1
    while ngroups < cap and groups_per_row % (ngroups * 2) == 0:
        ngroups *= 2
    # keep enough programs in flight when the input is small
    core_count = _vector_core_num()
    while ngroups > 1 and total_groups // ngroups < 4 * core_count:
        ngroups //= 2
    return ngroups


# UB-safe tile sizes for the persistent flat kernel, measured on Ascend910B4
# (fp8 fast-conversion path): the bit-trick fp8 conversion needs ~42B of UB
# per element for narrow tiles and ~190B/elem at BLOCK=512, so the per-tile
# element cap shrinks as the row width grows (wider tiles raise
# MLIRCompilationError: ub overflow against the 192KB UB budget).
def _flat_tile_elems_cap(block: int) -> int:
    if block <= 128:
        return 4096
    if block == 256:
        return 2048
    return 512


# Tile geometry for the wide-group slice kernel, measured on Ascend910B4:
# the UB cost of a tile is driven by its row width, not its element count,
# so narrow rows admit much larger tiles.  A (128, 32) tile (NGROUPS=8 at
# GROUP_SIZE=512) measures ~1.3x over the widest UB-safe (NGROUPS*4, 128)
# tile; wider rows (SUB=64/128) or more rows (NGROUPS=16) raise
# MLIRCompilationError: ub overflow against the 192KB UB budget.
_SLICE_TILE_ELEMS_CAP = 4096
_SLICE_SUB = 32

# Persistent grid size as a multiple of the vector core count.  Measured on
# Ascend910B4: 8 workers per core keeps every core busy while leaving
# enough independent programs per core to hide memory latency.
_WORKERS_PER_CORE = 8

_VECTOR_CORE_NUM = None


def _vector_core_num() -> int:
    # Every kernel in this file is vector-only (compiled as mix_mode "aiv"),
    # so programs are scheduled on the AI Vector cores, not the AI Cube
    # cores; on 910 the vector core count is 2x the cube core count.  Grid
    # and tiling heuristics must therefore use the vector core count.
    global _VECTOR_CORE_NUM
    if _VECTOR_CORE_NUM is None:
        try:
            device = torch.npu.current_device()
            _VECTOR_CORE_NUM = torch.npu.get_device_limit(device)["vector_core_num"]
        except (AttributeError, KeyError, TypeError, RuntimeError):
            try:
                utils = triton.runtime.driver.active.utils
                _VECTOR_CORE_NUM = utils.get_aivector_core_num()
            except (AttributeError, RuntimeError):
                _VECTOR_CORE_NUM = get_sm_count()
    return _VECTOR_CORE_NUM


def _slice_supported(group_size: int) -> bool:
    split = group_size // _SLICE_SUB
    return (
        512 <= group_size <= _SLICE_TILE_ELEMS_CAP
        and group_size % _SLICE_SUB == 0
        and split & (split - 1) == 0
    )


def _ngroups_persistent(n_groups: int, tile_elems_cap: int, group_size: int) -> int:
    # Largest power-of-two group count per tile, bounded by the UB-safe tile
    # size; shrunk again when a wider tile would leave persistent workers
    # without tasks.  Flat addressing only needs masks for the tail tile,
    # not divisibility into each row.
    workers = _vector_core_num() * _WORKERS_PER_CORE
    cap = max(1, tile_elems_cap // group_size)
    ngroups = 1
    while ngroups * 2 <= cap:
        if -(-n_groups // (ngroups * 2)) < workers:
            break
        ngroups *= 2
    return ngroups


def _persistent_grid(n_tasks: int) -> int:
    workers = _vector_core_num() * _WORKERS_PER_CORE
    return max(1, min(n_tasks, workers))


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS PER TOKEN GROUP QUANT FP8")
    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"
    # The fallback kernels flatten the leading dims into rows addressed by
    # one uniform row stride; reject layouts that do not collapse that way
    # instead of silently reading wrong addresses.
    assert (
        x.dim() <= 2 or x.is_contiguous()
    ), "`x` with more than 2 dims must be contiguous"
    row_stride = x.stride(-2) if x.dim() >= 2 else 0

    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    inv_fp8_max = 1.0 / fp8_max
    # When `fp8_max` is huge (e.g. the fp32 fallback), `inv_fp8_max` is a
    # subnormal fp32 and so is `y_s = _absmax * inv_fp8_max`; some backends
    # flush subnormal fp32 divisors to zero, which turns `y / y_s` into inf.
    subnormal_scale = inv_fp8_max < torch.finfo(torch.float32).tiny
    fast_cvt = fp8_dtype == torch.float8_e4m3fn

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    num_groups = x.numel() // group_size
    groups_per_row = x.shape[-1] // group_size

    if column_major_scales:
        assert x.dim() == 2, "column_major_scales only supports a 2-dim `x`"
        shape = (groups_per_row,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (groups_per_row,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    if num_groups == 0:
        return x_q, x_s

    block = triton.next_power_of_2(group_size)
    # The Ascend compiler rejects fp8-typed stores, so in the fast_cvt path
    # the kernels write the raw fp8 bit patterns through a uint8 view.
    x_q_arg = x_q.view(torch.uint8) if fast_cvt else x_q

    if fast_cvt and not column_major_scales and x.is_contiguous():
        if _slice_supported(group_size):
            ngroups = _ngroups_persistent(num_groups, _SLICE_TILE_ELEMS_CAP, group_size)
            n_tasks = -(-num_groups // ngroups)
            _per_token_group_quant_fp8_slice_persistent[(_persistent_grid(n_tasks),)](
                x,
                x_q_arg,
                x_s,
                num_groups,
                n_tasks,
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                inv_fp8_max=inv_fp8_max,
                scale_ue8m0=scale_ue8m0,
                GROUP_SIZE=group_size,
                SPLIT=group_size // _SLICE_SUB,
                SUB=_SLICE_SUB,
                NGROUPS=ngroups,
                HAS_TAIL=num_groups % ngroups != 0,
                num_stages=1,
            )
            return x_q, x_s

        ngroups = _ngroups_persistent(num_groups, _flat_tile_elems_cap(block), block)
        n_tasks = -(-num_groups // ngroups)
        _per_token_group_quant_fp8_persistent[(_persistent_grid(n_tasks),)](
            x,
            x_q_arg,
            x_s,
            num_groups,
            n_tasks,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            inv_fp8_max=inv_fp8_max,
            scale_ue8m0=scale_ue8m0,
            GROUP_SIZE=group_size,
            BLOCK=block,
            NGROUPS=ngroups,
            HAS_TAIL=num_groups % ngroups != 0,
            num_stages=1,
        )
        return x_q, x_s

    ngroups = _ngroups_per_program(num_groups, groups_per_row, group_size)
    grid = (num_groups // ngroups,)

    if column_major_scales:
        _per_token_group_quant_fp8_colmajor_vec[grid](
            x,
            x_q_arg,
            x_s,
            row_stride,
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            inv_fp8_max=inv_fp8_max,
            scale_ue8m0=scale_ue8m0,
            subnormal_scale=subnormal_scale,
            fast_cvt=fast_cvt,
            GROUP_SIZE=group_size,
            BLOCK=block,
            GROUPS_PER_ROW=groups_per_row,
            NGROUPS=ngroups,
            num_stages=1,
        )
    else:
        _per_token_group_quant_fp8_vec[grid](
            x,
            x_q_arg,
            x_s,
            row_stride,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            inv_fp8_max=inv_fp8_max,
            scale_ue8m0=scale_ue8m0,
            subnormal_scale=subnormal_scale,
            fast_cvt=fast_cvt,
            GROUP_SIZE=group_size,
            BLOCK=block,
            GROUPS_PER_ROW=groups_per_row,
            NGROUPS=ngroups,
            num_stages=1,
        )

    return x_q, x_s

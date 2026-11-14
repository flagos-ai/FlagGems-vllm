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
import re
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# GCU Triton casts and stores float8e4nv bit-exactly (verified on L600; the
# generic kernel_supports_fp8_e4m3() vendor whitelist predates that verification
# and does not know enflame — see the package-level gate correction in
# runtime/backend/_enflame/ops/__init__.py).  Declare real fp8 output like the
# other enflame fp8-quant ops instead of the fp32-pseudo-fp8 mode: the output
# matches vLLM's own per_token_group_quant_fp8 contract, halves store traffic
# and skips the _quant_scale denormal emulation entirely.
SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn


# ===========================================================================
# GCU compiler-toolchain dialect compatibility shim
# ===========================================================================
# The installed FlagTree wheel emits MLIR where several `memref_ext.*` /
# `gcu.matmul` ops carry a trailing zero operand segment, while the external
# `gcu-compiler-opt` tool shipped in /opt/triton_gcu still registers the older
# operand layouts. The mismatch makes every kernel with 2-D tensor accesses
# (this file's `[NGROUPS, BLOCK]` vec tiles included) fail to compile with
# "operandSegmentSizes ... size mismatch". The shim below rewrites the
# generated MLIR text to the layout the external tool accepts:
#   * `memref_ext.memset_start` must carry no properties at all;
#   * other `memref_ext.*` / `gcu.matmul` ops drop their trailing zero
#     segments (pure metadata, no operand rebinding);
#   * `memref.reinterpret_cast` / `memref.subview` need explicit segment
#     sizes derived from their INT64_MIN sentinel counts when missing.
# It is idempotent and leaves already-compatible IR untouched. Remove once
# the FlagTree wheel and the external gcu-compiler-opt agree on the dialect
# (see compiler_issues/vllm_moe_sum.md).

_GCU_SENTINEL = "-9223372036854775808"


def _fix_gcu_attr(attr_match, line, op_name):
    inner = attr_match.group(1)
    seg_match = re.search(r"operandSegmentSizes = array<i32: ([^>]*)>", inner)
    if op_name == "memref_ext.memset_start":
        return ""
    if op_name.startswith("memref_ext.") or op_name == "gcu.matmul":
        if seg_match:
            segs = [s.strip() for s in seg_match.group(1).split(",")]
            while len(segs) > 1 and segs[-1] == "0":
                segs.pop()
            new_seg = "array<i32: " + ", ".join(segs) + ">"
            rest = re.sub(r"operandSegmentSizes = array<i32:[^>]*>", "", inner)
            rest = rest.strip().strip(",").strip()
            if rest:
                return f" <{{operandSegmentSizes = {new_seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {new_seg}}}>"
        return attr_match.group(0)
    if op_name in ("memref.reinterpret_cast", "memref.subview") and not seg_match:
        opers = re.search(r"\(([^()]*)\)\s*<\{", line)
        n_ops = len([x for x in opers.group(1).split(",") if x.strip()]) if opers else 1
        offs = re.search(r"static_offsets = array<i64: ([^>]*)>", line)
        sizes = re.search(r"static_sizes = array<i64: ([^>]*)>", line)
        strides = re.search(r"static_strides = array<i64: ([^>]*)>", line)
        n_off = offs.group(1).count(_GCU_SENTINEL) if offs else 0
        n_size = sizes.group(1).count(_GCU_SENTINEL) if sizes else 0
        n_stride = strides.group(1).count(_GCU_SENTINEL) if strides else 0
        if 1 + n_off + n_size + n_stride == n_ops:
            seg = f"array<i32: 1, {n_off}, {n_size}, {n_stride}"
            rest = inner.strip()
            if rest:
                return f" <{{operandSegmentSizes = {seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {seg}}}>"
    return attr_match.group(0)


def _fix_gcu_dialect_line(line):
    op_match = re.search(r'"([a-z0-9_.]+)"\(', line)
    if not op_match:
        return line
    op_name = op_match.group(1)
    if not (
        op_name.startswith("memref_ext.")
        or op_name.startswith("gcu.")
        or op_name in ("memref.reinterpret_cast", "memref.subview")
    ):
        return line
    return re.sub(
        r" <\{([^}]*)\}>",
        lambda m: _fix_gcu_attr(m, line, op_name),
        line,
        count=1,
    )


def _patch_gcu_compiler_opt():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(_toolkit.gcu_compiler_opt, "_flaggems_dialect_shim", False):
        return
    _orig = _toolkit.gcu_compiler_opt

    def _shim(content, *args):
        if isinstance(content, str) and "operandSegmentSizes" in content:
            content = "\n".join(_fix_gcu_dialect_line(l) for l in content.split("\n"))
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


logger = logging.getLogger(__name__)


@triton.jit
def _quant_scale(absmax, fp8_max, FP32_AS_FP8: tl.constexpr):
    """Group scale ``absmax * (1.0 / fp8_max)`` without GCU denormal flushing.

    On GCU every fp32 arithmetic op is FTZ: in the fp32-pseudo-fp8 mode
    (``fp8_max == flt_max``, ``fl32(1.0 / fp8_max) == 2**-128``) the product
    is denormal for ``absmax < 4`` and would flush to zero, so synthesize the
    IEEE-RNE product from the bit pattern instead (exact match with torch).
    """
    if FP32_AS_FP8:
        bits = absmax.to(tl.int32, bitcast=True)
        exp = bits >> 23
        exp_c = tl.minimum(tl.maximum(exp, 1), 254)
        sig = (bits & 0x007FFFFF) + 0x00800000
        shift = tl.maximum(129 - exp_c, 1)
        sh = tl.minimum(shift, 24)
        quot = sig >> sh
        rem = sig & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        round_up = (rem > half) | ((rem == half) & ((quot & 1) != 0))
        mant = tl.where(shift <= 24, quot + round_up.to(tl.int32), 0)
        scaled = tl.where(exp_c >= 129, bits - 0x40000000, mant)
        scale = scaled.to(tl.float32, bitcast=True)
        return tl.where(exp >= 255, absmax, scale)
    else:
        return absmax * (1.0 / fp8_max)


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    FP32_AS_FP8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += row * y_row_stride + row_g_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _quant_scale(_absmax, fp8_max, FP32_AS_FP8)

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max)
    elif FP32_AS_FP8:
        # GCU fp32 division flushes denormal reciprocals: the scale is
        # denormal while absmax < 4, while the reciprocal of a huge absmax
        # would flush instead, so always divide by the normal operand.
        inv = tl.where(_absmax >= 4.0, y_s, _absmax)
        mul = tl.where(_absmax >= 4.0, 1.0, fp8_max)
        y_q = tl.clamp((y / inv) * mul, fp8_min, fp8_max)
    else:
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max)
    y_q = y_q.to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    FP32_AS_FP8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    group_id = g_id % groups_per_row

    y_ptr += row * y_row_stride + group_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += group_id * y_s_col_stride + row

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _quant_scale(_absmax, fp8_max, FP32_AS_FP8)

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max)
    elif FP32_AS_FP8:
        # GCU fp32 division flushes denormal reciprocals: the scale is
        # denormal while absmax < 4, while the reciprocal of a huge absmax
        # would flush instead, so always divide by the normal operand.
        inv = tl.where(_absmax >= 4.0, y_s, _absmax)
        mul = tl.where(_absmax >= 4.0, 1.0, fp8_max)
        y_q = tl.clamp((y / inv) * mul, fp8_min, fp8_max)
    else:
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max)
    y_q = y_q.to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    FP32_AS_FP8: tl.constexpr,
    BLOCK: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    programs_per_row = groups_per_row // NGROUPS

    pid = tl.program_id(0)
    row = pid // programs_per_row
    program_id = pid % programs_per_row

    start_group = program_id * NGROUPS
    start_gid = row * groups_per_row + start_group

    group_ids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (
        row * y_row_stride
        + start_group * group_size
        + group_ids[:, None] * group_size
        + cols[None, :]
    )
    mask = cols[None, :] < group_size

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _quant_scale(_absmax, fp8_max, FP32_AS_FP8)

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max)
    elif FP32_AS_FP8:
        # GCU fp32 division flushes denormal reciprocals: the scale is
        # denormal while absmax < 4, while the reciprocal of a huge absmax
        # would flush instead, so always divide by the normal operand.
        inv = tl.where(_absmax >= 4.0, y_s, _absmax)
        mul = tl.where(_absmax >= 4.0, 1.0, fp8_max)
        y_q = tl.clamp((y / inv[:, None]) * mul[:, None], fp8_min, fp8_max)
    else:
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max)
    y_q = y_q.to(y_q_ptr.dtype.element_ty)
    output_offsets = (
        start_gid * group_size + group_ids[:, None] * group_size + cols[None, :]
    )

    tl.store(y_q_ptr + output_offsets, y_q, mask=mask)
    tl.store(y_s_ptr + start_gid + group_ids, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor_vec(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    FP32_AS_FP8: tl.constexpr,
    BLOCK: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    programs_per_row = groups_per_row // NGROUPS

    pid = tl.program_id(0)
    row = pid // programs_per_row
    program_id = pid % programs_per_row

    start_group = program_id * NGROUPS
    start_gid = row * groups_per_row + start_group

    group_ids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (
        row * y_row_stride
        + start_group * group_size
        + group_ids[:, None] * group_size
        + cols[None, :]
    )
    mask = cols[None, :] < group_size

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _quant_scale(_absmax, fp8_max, FP32_AS_FP8)

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max)
    elif FP32_AS_FP8:
        # GCU fp32 division flushes denormal reciprocals: the scale is
        # denormal while absmax < 4, while the reciprocal of a huge absmax
        # would flush instead, so always divide by the normal operand.
        inv = tl.where(_absmax >= 4.0, y_s, _absmax)
        mul = tl.where(_absmax >= 4.0, 1.0, fp8_max)
        y_q = tl.clamp((y / inv[:, None]) * mul[:, None], fp8_min, fp8_max)
    else:
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max)
    y_q = y_q.to(y_q_ptr.dtype.element_ty)
    output_offsets = (
        start_gid * group_size + group_ids[:, None] * group_size + cols[None, :]
    )
    scale_offsets = (start_group + group_ids) * y_s_col_stride + row

    tl.store(y_q_ptr + output_offsets, y_q, mask=mask)
    tl.store(y_s_ptr + scale_offsets, y_s)


@triton.jit
def _per_token_group_quant_fp8_flat(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    FP32_AS_FP8: tl.constexpr,
    BLOCK: tl.constexpr,
    NGROUPS: tl.constexpr,
):
    # Flat-contiguous variant: groups are dense in memory, so each program
    # handles NGROUPS consecutive groups with a [NGROUPS, BLOCK] tile and no
    # row/group div-mod prologue.  Amortizes the per-program cost that dominates
    # this kernel family on GCU at large group counts.
    pid = tl.program_id(0)
    start_gid = pid * NGROUPS
    group_ids = tl.arange(0, NGROUPS)
    cols = tl.arange(0, BLOCK)
    offsets = (start_gid + group_ids[:, None]) * group_size + cols[None, :]
    mask = cols[None, :] < group_size

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _quant_scale(_absmax, fp8_max, FP32_AS_FP8)

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max)
    elif FP32_AS_FP8:
        # GCU fp32 division flushes denormal reciprocals: the scale is
        # denormal while absmax < 4, while the reciprocal of a huge absmax
        # would flush instead, so always divide by the normal operand.
        inv = tl.where(_absmax >= 4.0, y_s, _absmax)
        mul = tl.where(_absmax >= 4.0, 1.0, fp8_max)
        y_q = tl.clamp((y / inv[:, None]) * mul[:, None], fp8_min, fp8_max)
    else:
        y_q = tl.clamp(y / y_s[:, None], fp8_min, fp8_max)
    y_q = y_q.to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + offsets, y_q, mask=mask)
    tl.store(y_s_ptr + start_gid + group_ids, y_s)


def _groups_per_program(x: torch.Tensor, group_size: int) -> int:
    groups_per_row = x.shape[-1] // group_size
    for groups in (8, 4, 2):
        if groups_per_row % groups == 0:
            return groups
    return 1


# Flat-tile budget: NGROUPS * BLOCK elements live in fp32 during the tile, plus
# the quantized tile; 32768 elements (128KB fp32) measured safe on GCU400/L600
# ([64, 512] winner) while keeping DSM well under the 448KB program budget.
_FLAT_MAX_TILE_ELEMS = 32768


def _flat_groups_per_program(num_groups: int, block: int) -> int:
    # Largest divisor wins at every scale: large tensors need the per-program
    # amortization, and tiny ones sit on the launch floor where fewer programs
    # is also better.  (A "fill 48 SPs" guard was tried and measurably regressed
    # the 112-group shape: it forced NG2/56 programs instead of NG16/7, +27%.)
    max_groups = max(1, _FLAT_MAX_TILE_ELEMS // block)
    for groups in (64, 32, 16, 8, 4, 2):
        if groups <= max_groups and num_groups % groups == 0:
            return groups
    return 1


def _flat_num_warps(ngroups: int, block: int) -> int:
    tile_elems = ngroups * block
    if tile_elems > 16384:
        return 4
    if tile_elems > 1024:
        return 2
    return 1


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

    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    # fp32-pseudo-fp8 mode: fp8_max == flt_max makes every group scale
    # denormal, which GCU fp32 arithmetic (FTZ) cannot produce or divide by.
    fp32_as_fp8 = fp8_dtype == torch.float32

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    num_groups = x.numel() // group_size

    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    block = triton.next_power_of_2(group_size)
    num_warps = min(max(block // 256, 1), 8)
    groups_per_program = _groups_per_program(x, group_size)
    flat_groups_per_program = (
        _flat_groups_per_program(num_groups, block)
        if not column_major_scales and x.is_contiguous()
        else 1
    )
    grid = (num_groups // groups_per_program,)

    if column_major_scales:
        if groups_per_program > 1:
            kernel = _per_token_group_quant_fp8_colmajor_vec
            kernel[grid](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=scale_ue8m0,
                FP32_AS_FP8=fp32_as_fp8,
                BLOCK=block,
                NGROUPS=groups_per_program,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            kernel = _per_token_group_quant_fp8_colmajor
            kernel[grid](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=scale_ue8m0,
                FP32_AS_FP8=fp32_as_fp8,
                BLOCK=block,
                num_warps=num_warps,
                num_stages=1,
            )
    elif flat_groups_per_program > 1:
        kernel = _per_token_group_quant_fp8_flat
        grid = (num_groups // flat_groups_per_program,)
        kernel[grid](
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            FP32_AS_FP8=fp32_as_fp8,
            BLOCK=block,
            NGROUPS=flat_groups_per_program,
            num_warps=_flat_num_warps(flat_groups_per_program, block),
            num_stages=1,
        )
    elif groups_per_program > 1:
        kernel = _per_token_group_quant_fp8_vec
        kernel[grid](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            FP32_AS_FP8=fp32_as_fp8,
            BLOCK=block,
            NGROUPS=groups_per_program,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        kernel = _per_token_group_quant_fp8
        kernel[grid](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
            FP32_AS_FP8=fp32_as_fp8,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )

    return x_q, x_s

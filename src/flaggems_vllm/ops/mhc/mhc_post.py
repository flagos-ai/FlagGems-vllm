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

"""
Triton implementation of mHC Post operator (optimized v3).

Computes:
    out[n, i, h] = post_layer_mix[n, i] * x[n, h]
                 + sum_j(comb_res_mix[n, j, i] * residual[n, j, h])

Key optimizations (v3):
- 2D grid = (N, cdiv(H, BLOCK_H)): high program count for latency hiding.
- @triton.autotune over BLOCK_H / num_warps / num_stages.
- Contiguous layout: stride math removed, enabling LDG.128.
- All 4 accumulators computed then stored (better ILP).
- BLOCK_H chosen to evenly divide H when possible (256 divides all targets).

"""

from __future__ import annotations

import enum
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_HC_MULT4_AUTOTUNE_CONFIG = [
    # Small BLOCK_H: many programs, good for latency hiding
    triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=2),
    # Medium BLOCK_H
    triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=2),
    # Large BLOCK_H
    triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
]

_GENERIC_AUTOTUNE_CONFIG = [
    triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=_HC_MULT4_AUTOTUNE_CONFIG,
    key=["H"],
)
@triton.jit
def mhc_post_kernel_hc_mult_4(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 — a[n, j, i]
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4),   float32
    d_ptr,  # x            : (N, H),    bfloat16
    out_ptr,  # output       : (N, 4, H), bfloat16
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Grid: (N, cdiv(H, BLOCK_H)).
    Each program handles one token × one h-tile × all 4 hc streams.
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    # ── pointer bases (contiguous layout) ──
    a_base = pid_n * 16  # (N, 4, 4) → stride_n = 16
    c_base = pid_n * 4  # (N, 4)    → stride_n = 4
    b_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H
    d_base = pid_n * H  # (N, H)    → stride_n = H
    out_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H

    # ── load 20 scalars (L1 cached across h-tiles) ──
    c0 = tl.load(c_ptr + c_base + 0).to(tl.float32)
    c1 = tl.load(c_ptr + c_base + 1).to(tl.float32)
    c2 = tl.load(c_ptr + c_base + 2).to(tl.float32)
    c3 = tl.load(c_ptr + c_base + 3).to(tl.float32)

    a00 = tl.load(a_ptr + a_base + 0).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 1).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 2).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 3).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 4).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 5).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 6).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 7).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 8).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 9).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 10).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 11).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 12).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 13).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 14).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 15).to(tl.float32)

    # ── load vectors (bf16 → f32) ──
    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + b_base + 0 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + b_base + 1 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b2 = tl.load(b_ptr + b_base + 2 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b3 = tl.load(b_ptr + b_base + 3 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)

    # ── compute all 4 output streams ──
    acc0 = c0 * d_vals + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    acc1 = c1 * d_vals + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    acc2 = c2 * d_vals + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    acc3 = c3 * d_vals + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

    # ── store all 4 outputs ──
    tl.store(out_ptr + out_base + 0 * H + h_off, acc0.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_off, acc1.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_off, acc2.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_off, acc3.to(tl.bfloat16), mask=h_mask)


@triton.autotune(
    configs=_GENERIC_AUTOTUNE_CONFIG,
    key=["H", "HC"],
)
@triton.jit
def mhc_post_kernel_generic(
    a_ptr,  # comb_res_mix : (N, HC, HC), float32
    b_ptr,  # residual     : (N, HC, H), bfloat16
    c_ptr,  # post_layer_mix: (N, HC), float32
    d_ptr,  # x            : (N, H), bfloat16
    out_ptr,  # output      : (N, HC, H), bfloat16
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Generic mHC post kernel for arbitrary HC.

    Grid: (N, HC, cdiv(H, BLOCK_H)).
    Each program handles one token × one output-stream(i) × one h-tile.
    """
    pid_n = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    a_base = pid_n * HC * HC
    b_base = pid_n * HC * H
    c_base = pid_n * HC
    d_base = pid_n * H
    out_base = pid_n * HC * H + pid_i * H

    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    c_i = tl.load(c_ptr + c_base + pid_i).to(tl.float32)

    acc = c_i * d_vals
    for j in tl.static_range(0, HC):
        a_ji = tl.load(a_ptr + a_base + j * HC + pid_i).to(tl.float32)
        b_j = tl.load(b_ptr + b_base + j * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        acc += a_ji * b_j

    tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16), mask=h_mask)


@triton.autotune(
    configs=_HC_MULT4_AUTOTUNE_CONFIG,
    key=["H"],
)
@triton.jit
def mhc_post_kernel_hc_mult_4_strided(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 — a[n, j, i]
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4),   float32
    d_ptr,  # x            : (N, H),    bfloat16
    out_ptr,  # output       : (N, 4, H), bfloat16
    a_stride_n,  # comb_res_mix stride over N
    a_stride_j,  # comb_res_mix stride over residual-stream dim j
    a_stride_i,  # comb_res_mix stride over output-stream dim i
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Grid: (N, cdiv(H, BLOCK_H)).
    Each program handles one token × one h-tile × all 4 hc streams.
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    # ── pointer bases (mix strided; b/c/d/out contiguous) ──
    a_base = pid_n * a_stride_n
    c_base = pid_n * 4  # (N, 4)    → stride_n = 4
    b_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H
    d_base = pid_n * H  # (N, H)    → stride_n = H
    out_base = pid_n * 4 * H  # (N, 4, H) → stride_n = 4*H

    # ── load 20 scalars (L1 cached across h-tiles) ──
    c0 = tl.load(c_ptr + c_base + 0).to(tl.float32)
    c1 = tl.load(c_ptr + c_base + 1).to(tl.float32)
    c2 = tl.load(c_ptr + c_base + 2).to(tl.float32)
    c3 = tl.load(c_ptr + c_base + 3).to(tl.float32)

    a00 = tl.load(a_ptr + a_base).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 1 * a_stride_i).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 2 * a_stride_i).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 3 * a_stride_i).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 1 * a_stride_j).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 1 * a_stride_j + 1 * a_stride_i).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 1 * a_stride_j + 2 * a_stride_i).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 1 * a_stride_j + 3 * a_stride_i).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 2 * a_stride_j).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 2 * a_stride_j + 1 * a_stride_i).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 2 * a_stride_j + 2 * a_stride_i).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 2 * a_stride_j + 3 * a_stride_i).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 3 * a_stride_j).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 3 * a_stride_j + 1 * a_stride_i).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 3 * a_stride_j + 2 * a_stride_i).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 3 * a_stride_j + 3 * a_stride_i).to(tl.float32)

    # ── load vectors (bf16 → f32) ──
    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + b_base + 0 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + b_base + 1 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b2 = tl.load(b_ptr + b_base + 2 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b3 = tl.load(b_ptr + b_base + 3 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)

    # ── compute all 4 output streams ──
    acc0 = c0 * d_vals + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    acc1 = c1 * d_vals + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    acc2 = c2 * d_vals + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    acc3 = c3 * d_vals + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

    # ── store all 4 outputs ──
    tl.store(out_ptr + out_base + 0 * H + h_off, acc0.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_off, acc1.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_off, acc2.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_off, acc3.to(tl.bfloat16), mask=h_mask)


@triton.autotune(
    configs=_GENERIC_AUTOTUNE_CONFIG,
    key=["H", "HC"],
)
@triton.jit
def mhc_post_kernel_generic_strided(
    a_ptr,  # comb_res_mix : (N, HC, HC), float32
    b_ptr,  # residual     : (N, HC, H), bfloat16
    c_ptr,  # post_layer_mix: (N, HC), float32
    d_ptr,  # x            : (N, H), bfloat16
    out_ptr,  # output      : (N, HC, H), bfloat16
    a_stride_n,  # comb_res_mix stride over N
    a_stride_j,  # comb_res_mix stride over residual-stream dim j
    a_stride_i,  # comb_res_mix stride over output-stream dim i
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Generic mHC post kernel for arbitrary HC.

    Grid: (N, HC, cdiv(H, BLOCK_H)).
    Each program handles one token × one output-stream(i) × one h-tile.
    """
    pid_n = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    a_base = pid_n * a_stride_n
    b_base = pid_n * HC * H
    c_base = pid_n * HC
    d_base = pid_n * H
    out_base = pid_n * HC * H + pid_i * H

    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    c_i = tl.load(c_ptr + c_base + pid_i).to(tl.float32)

    acc = c_i * d_vals
    for j in tl.static_range(0, HC):
        a_ji = tl.load(a_ptr + a_base + j * a_stride_j + pid_i * a_stride_i).to(
            tl.float32
        )
        b_j = tl.load(b_ptr + b_base + j * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        acc += a_ji * b_j

    tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16), mask=h_mask)


class MixLayout(enum.Enum):
    """Logical layout of ``comb_res_mix`` accepted by :func:`mhc_post`.

    Contract (i/o are output/residual stream indices, h the hidden index):

    * ``NATURAL`` — ``mix[t, i, j]``, i.e.
      ``out[t, i, h] = post[t, i] * x[t, h] + Σ_j mix[t, i, j] * residual[t, j, h]``
      which equals ``bmm(mix, residual)``. This is the layout ``mhc_pre``
      produces — Sinkhorn's row-major ``[output_stream, input_stream]`` —
      the layout upstream calls "the natural layout" (XC4-MUSA-PERF-002);
      consumed copy-free straight from any view.
    * ``TRANSPOSED`` — ``mix[t, j, i]``, i.e. ``out = bmm(mix.mT, residual)``;
      the legacy convention where callers passed a caller-side
      ``mix.transpose(-1, -2).contiguous()`` (default).
    """

    NATURAL = "natural"
    TRANSPOSED = "transposed"


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    mix_layout: MixLayout = MixLayout.TRANSPOSED,
) -> torch.Tensor:
    """
    mHC post-processing operator (zero-copy on ``comb_res_mix``).

    Computes ``out[t, i, h] = post[t, i] * x[t, h] + Σ_j mix[t, i, j] *
    residual[t, j, h]`` for ``mix_layout=NATURAL`` (see :class:`MixLayout`).

    Args:
        x: (N, H), bfloat16 — layer output; must be contiguous
        residual: (N, hc_mult, H), bfloat16 — multi-head residual; contiguous
        post_layer_mix: (N, hc_mult, 1) or (N, hc_mult), float32; contiguous
        comb_res_mix: (N, hc_mult, hc_mult), float32 — combination matrix;
            read **in place** in the layout given by ``mix_layout``; a
            non-contiguous view is allowed (no transpose/copy is made)
        mix_layout: layout of ``comb_res_mix`` (default ``TRANSPOSED``)

    Raises:
        TypeError: ``mix_layout`` is not a :class:`MixLayout` member
        NotImplementedError: ``x`` / ``residual`` / ``post_layer_mix`` are
            non-contiguous (a silent copy here would defeat the zero-copy
            contract)

    Returns:
        out: (N, hc_mult, H), bfloat16
    """
    logger.debug(
        "GEMS MHC_POST FORWARD, x=%s, residual=%s, post_layer_mix=%s, comb_res_mix=%s",
        x.shape,
        residual.shape,
        post_layer_mix.shape,
        comb_res_mix.shape,
    )

    if not isinstance(mix_layout, MixLayout):
        raise TypeError(
            "mix_layout must be a MixLayout enum member, "
            f"got {type(mix_layout).__name__}"
        )
    if residual.dtype != torch.bfloat16:
        raise NotImplementedError("mHC post residual must use bfloat16")
    if residual.device.type not in ("cuda", "musa"):
        raise NotImplementedError("mHC post requires CUDA or MUSA tensors")

    N, hc, H = residual.shape
    if x.shape != (N, H):
        raise ValueError(f"x must have shape ({N}, {H}), got {tuple(x.shape)}")
    if post_layer_mix.shape not in ((N, hc, 1), (N, hc)):
        raise ValueError(
            f"post_layer_mix must have shape ({N}, {hc}, 1), got {tuple(post_layer_mix.shape)}"
        )
    if comb_res_mix.shape != (N, hc, hc):
        raise ValueError(
            f"comb_res_mix must have shape ({N}, {hc}, {hc}), got {tuple(comb_res_mix.shape)}"
        )

    out = torch.empty_like(residual)

    c = post_layer_mix.squeeze(-1)  # (N, hc); view, input validated contiguous
    b = residual
    d = x
    a = comb_res_mix  # read in place — never copied

    # TRANSPOSED (the legacy transpose-copy layout) + contiguous keeps the
    # original hard-coded offsets (byte-identical fast path); any other
    # layout/memory form goes through the strided variants.
    use_strided_mix = not (
        mix_layout is MixLayout.TRANSPOSED and comb_res_mix.is_contiguous()
    )
    if use_strided_mix:
        st = comb_res_mix.stride()
        a_sn = st[0]
        # Map stored mix onto the kernel's (j, i) coefficient indices:
        # TRANSPOSED stores mix[t, j, i]; NATURAL stores mix[t, i, j].
        a_sj, a_si = (
            (st[2], st[1]) if mix_layout is MixLayout.NATURAL else (st[1], st[2])
        )

    if hc == 4:

        def grid_specialized(META):
            return (N, triton.cdiv(H, META["BLOCK_H"]))

        if use_strided_mix:
            mhc_post_kernel_hc_mult_4_strided[grid_specialized](
                a, b, c, d, out, a_sn, a_sj, a_si, H=H
            )
        else:
            kernel = mhc_post_kernel_hc_mult_4
            kernel[grid_specialized](a, b, c, d, out, H=H)
    else:

        def grid_generic(META):
            return (N, hc, triton.cdiv(H, META["BLOCK_H"]))

        if use_strided_mix:
            mhc_post_kernel_generic_strided[grid_generic](
                a, b, c, d, out, a_sn, a_sj, a_si, H=H, HC=hc
            )
        else:
            kernel = mhc_post_kernel_generic
            kernel[grid_generic](a, b, c, d, out, H=H, HC=hc)
    return out


__all__ = ["MixLayout", "mhc_post", "mhc_post_ref"]


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    mix_layout: MixLayout = MixLayout.TRANSPOSED,
) -> torch.Tensor:
    """PyTorch reference implementation (mirrors :func:`mhc_post` layouts)."""
    if not isinstance(mix_layout, MixLayout):
        raise TypeError(
            "mix_layout must be a MixLayout enum member, "
            f"got {type(mix_layout).__name__}"
        )
    mix = comb_res_mix.mT if mix_layout is MixLayout.TRANSPOSED else comb_res_mix
    y = x.unsqueeze(-2) * post_layer_mix + torch.bmm(mix, residual.float())
    return y.type_as(x)

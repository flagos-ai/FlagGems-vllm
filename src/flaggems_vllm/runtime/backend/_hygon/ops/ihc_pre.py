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

"""Hygon schedules for iHC projection and gated channel reduction."""
import logging

import triton
import triton.language as tl

from flaggems_vllm.ops.ihc.ihc_pre import (
    _decode_block_k,
    _ihc_pre,
    _ihc_pre_stage1,
    _ihc_pre_stage2,
    _launch_ihc_pre,
)
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _short_row(
    X,
    W,
    S,
    B,
    Y,
    P,
    D: tl.constexpr,
    H: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SD: tl.constexpr,
    SS: tl.constexpr,
    SB: tl.constexpr,
    MAG: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BK)
    d = tl.arange(0, BD)
    x = tl.load(X + n * SN + (k // D) * SC + (k % D) * SD, k < H * D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / (H * D) + NORM)
    s = tl.load(S)
    ps = tl.load(S + SS)
    y = tl.full((BD,), 0, tl.float32)
    for c in tl.static_range(H):
        w = tl.load(W + c * H * D + k, k < H * D, 0)
        wp = tl.load(W + (H + c) * H * D + k, k < H * D, 0)
        pre = tl.sigmoid(tl.sum(w * x, 0) * inv * s + tl.load(B + c * SB)) + EPS
        post = (
            MAG * tl.sigmoid(tl.sum(wp * x, 0) * inv * ps + tl.load(B + (H + c) * SB))
            + EPS
        )
        xc = tl.load(X + n * SN + c * SC + d * SD, d < D, 0).to(tl.float32)
        y += pre * xc
        tl.store(P + n * H + c, post)
    tl.store(Y + n * D + d, y, d < D)


@triton.jit
def _decode_projection(
    X,
    W,
    Q,
    D: tl.constexpr,
    H: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SD: tl.constexpr,
    NS: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    z = tl.program_id(1)
    k = z * BK + tl.arange(0, BK)
    c = tl.arange(0, BC)
    if SC == D * SD:
        idx = k * SD
    else:
        idx = k // D * SC + k % D * SD
    x = tl.load(X + n * SN + idx, k < H * D, 0).to(tl.float32)
    w = tl.load(
        W + c[:, None] * (H * D) + k[None, :],
        (c[:, None] < 2 * H) & (k[None, :] < H * D),
        0,
    )
    slot = Q + (n * NS + z) * (2 * H + 1)
    tl.store(slot, tl.sum(x * x, 0))
    tl.store(slot + 1 + c, tl.sum(w * x[None, :], 1), c < 2 * H)


@triton.jit
def _decode_output(
    X,
    Q,
    S,
    B,
    Y,
    P,
    D: tl.constexpr,
    H: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SD: tl.constexpr,
    SS: tl.constexpr,
    SB: tl.constexpr,
    NS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    MAG: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    z = tl.arange(0, BS)
    q = Q + (n * NS + z) * (2 * H + 1)
    sq = tl.sum(tl.load(q, z < NS, 0), 0)
    inv = tl.rsqrt(sq / (D * H) + NORM)
    s = tl.load(S)
    d = b * BD + tl.arange(0, BD)
    x0 = X + n * SN + d * SD
    y = tl.full((BD,), 0, tl.float32)
    for c in tl.static_range(H):
        v = tl.sum(tl.load(q + 1 + c, z < NS, 0), 0)
        pre = tl.sigmoid(v * inv * s + tl.load(B + c * SB)) + EPS
        x = tl.load(x0 + c * SC, d < D, 0).to(tl.float32)
        y += pre * x
        if b == 0:
            v = tl.sum(tl.load(q + 1 + H + c, z < NS, 0), 0)
            post = (
                MAG * tl.sigmoid(v * inv * tl.load(S + SS) + tl.load(B + (H + c) * SB))
                + EPS
            )
            tl.store(P + n * H + c, post)
    tl.store(Y + n * D + d, y, d < D)


_pointer_key = libentry()(_ihc_pre_stage1).key
_runners = {}


def _launch(x, weight, scale, base, magnitude, hc_eps, norm_eps, pipeline):
    n, hc, d = x.shape
    k = hc * d
    if not n or hc > 4 or x.numel() > 2**31 - 1 or (k > 1024 and (hc != 4 or k % 1024)):
        return _launch_ihc_pre(
            x, weight, scale, base, magnitude, hc_eps, norm_eps, pipeline
        )
    y = x.new_empty((n, d))
    post = weight.new_empty((n, hc))
    if k <= 1024:
        _short_row[(n,)](
            x,
            weight,
            scale,
            base,
            y,
            post,
            d,
            hc,
            *x.stride(),
            scale.stride(0),
            base.stride(0),
            magnitude,
            hc_eps,
            norm_eps,
            triton.next_power_of_2(k),
            triton.next_power_of_2(d),
            enable_fp_fusion=False,
        )
        return y, post

    # Contiguous inputs amortize the tiled projection at fewer tokens.
    if n < 64 or (n < 256 and not x.is_contiguous()):
        # Split the reduction over K and consume input strides in both stages.
        bk = min(_decode_block_k(n), 256)
        splits = triton.cdiv(k, bk)
        partial = weight.new_empty((n, splits, 2 * hc + 1))
        project, finish = _decode_projection, _decode_output
        proj_grid, final_grid = (n, splits, 1), (n, triton.cdiv(d, 512), 1)
        proj_args = (
            x,
            weight,
            partial,
            d,
            hc,
            *x.stride(),
            splits,
            bk,
            triton.next_power_of_2(2 * hc),
        )
        final_args = (
            x,
            partial,
            scale,
            base,
            y,
            post,
            d,
            hc,
            *x.stride(),
            scale.stride(0),
            base.stride(0),
            splits,
            triton.next_power_of_2(splits),
            512,
            magnitude,
            hc_eps,
            norm_eps,
        )
        pointers = (x, weight, partial, scale, base, y, post)
        warps = 1
    else:
        x, scale, base = x.contiguous(), scale.contiguous(), base.contiguous()
        # Four wavefronts cover a 512-element K tile. Private split slots keep
        # the final reduction deterministic while shortening the serial K loop.
        bm, bk, warps = 8, 512, 4
        splits = 1
        while splits < 8 and k % (2 * splits * bk) == 0:
            splits *= 2
        squares = weight.new_empty((splits, n))
        mix = weight.new_empty((splits, n, 2 * hc))
        project, finish = _ihc_pre_stage1, _ihc_pre_stage2
        proj_grid, final_grid = (triton.cdiv(n, bm), splits, 1), (
            n,
            triton.cdiv(d, 1024),
            1,
        )
        proj_args = (
            x,
            weight,
            squares,
            mix,
            n,
            k,
            k // splits,
            2 * hc,
            bm,
            bk,
            splits,
            pipeline,
        )
        final_args = (
            x,
            mix,
            squares,
            scale,
            base,
            y,
            post,
            n,
            d,
            k,
            hc,
            triton.next_power_of_2(hc),
            2 * hc,
            splits,
            triton.next_power_of_2(splits),
            1024,
            magnitude,
            norm_eps,
            hc_eps,
        )
        pointers = (x, weight, squares, mix, scale, base, y, post)
    # Resolve both launchers before submitting the first kernel. FlagTree HCU
    # specializes on storage range as well as dtype and pointer alignment.
    key = None
    if triton.__version__.startswith("3.6."):
        key = (
            x.device.index,
            x.shape,
            x.stride(),
            scale.stride(0),
            base.stride(0),
            magnitude.hex(),
            hc_eps.hex(),
            norm_eps.hex(),
            pipeline,
            _pointer_key(pointers, (), []),
        )
        runners = _runners.get(key)
        if runners is not None:
            project_run, finish_run = runners
            project_run(*proj_args)
            finish_run(*final_args)
            return y, post
    first = project[proj_grid](*proj_args, num_warps=warps, enable_fp_fusion=False)
    second = finish[final_grid](*final_args, num_warps=4, enable_fp_fusion=False)
    if key is not None:
        if len(_runners) >= 256:
            _runners.clear()
        _runners[key] = (first[proj_grid], second[final_grid])
    return y, post


def triton_ihc_pre(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    logger.debug("GEMS_HYGON TRITON_IHC_PRE")
    return _ihc_pre(
        x, weight, scale, base, magnitude, hc_eps, norm_eps, _launcher=_launch
    )


def ihc_pre(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    logger.debug("GEMS_HYGON IHC_PRE")
    return _ihc_pre(
        x, weight, scale, base, magnitude, hc_eps, norm_eps, _launcher=_launch
    )

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

"""iHC pre schedules for Ascend vector and cube cores."""

import logging
from functools import partial

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.ihc.ihc_pre import _ihc_pre, _launch_ihc_pre

logger = logging.getLogger(__name__)


@triton.jit
def _short_projection(
    X,
    W,
    S,
    B,
    Y,
    P,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    MAG: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    d = tl.arange(0, BD)
    x = tl.load(
        X + m[:, None] * H * D + k[None, :], (m[:, None] < N) & (k[None, :] < H * D), 0
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / (H * D) + NORM)
    sc = tl.load(S)
    ps = tl.load(S + SCALE_STRIDE)
    y = tl.full((BM, BD), 0, tl.float32)
    for j in tl.static_range(H):
        w = tl.load(W + j * H * D + k, k < H * D, 0)
        wp = tl.load(W + (H + j) * H * D + k, k < H * D, 0)
        gate = (
            tl.sigmoid(
                tl.sum(x * w[None, :], 1) * inv * sc + tl.load(B + j * BASE_STRIDE)
            )
            + EPS
        )
        post = (
            MAG
            * tl.sigmoid(
                tl.sum(x * wp[None, :], 1) * inv * ps
                + tl.load(B + (H + j) * BASE_STRIDE)
            )
            + EPS
        )
        xx = tl.load(
            X + (m[:, None] * H + j) * D + d[None, :],
            (m[:, None] < N) & (d[None, :] < D),
            0,
        ).to(tl.float32)
        y += gate[:, None] * xx
        tl.store(P + m * H + j, post, m < N)
    tl.store(Y + m[:, None] * D + d[None, :], y, (m[:, None] < N) & (d[None, :] < D))


@triton.jit
def _load_stride_two(
    row, col, valid_rows, D: tl.constexpr, BM: tl.constexpr, WIDTH: tl.constexpr
):
    # Load contiguous pairs and discard the unused lane in local memory.
    # The last pair may end at the storage boundary of the original view.
    offset = 2 * col + tl.arange(0, 2 * WIDTH)
    lanes = tl.load(
        row + offset[None, :], valid_rows[:, None] & (offset[None, :] < 2 * D - 1), 0
    )
    even, _ = tl.split(tl.reshape(lanes, (BM, WIDTH, 2)))
    return even.to(tl.float32)


@triton.jit
def _project_and_gate(
    X,
    W,
    S,
    B,
    Y,
    P,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    MAG: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    c = tl.arange(0, 16)
    # Keep the eight FP32 projections on the cube cores and reuse each input
    # tile for the row norm. Only the final outputs go to global memory.
    acc = tl.full((BM, 16), 0, tl.float32)
    sq = tl.full((BM,), 0, tl.float32)
    for i in range(triton.cdiv(H * D, BK)):
        off = i * BK + k
        if CHANNEL_STRIDE:
            channel = i // (D // BK)
            col = (i - channel * (D // BK)) * BK
            row = X + m[:, None] * TOKEN_STRIDE + channel * CHANNEL_STRIDE
            x = _load_stride_two(row, col, m < N, D, BM, BK)
        else:
            x = tl.load(
                X + m[:, None] * (H * D) + off[None, :],
                (m[:, None] < N) & (off[None, :] < H * D),
                0,
            ).to(tl.float32)
        w = tl.load(
            W + c[:, None] * (H * D) + off[None, :],
            (c[:, None] < 2 * H) & (off[None, :] < H * D),
            0,
        )
        acc += tl.dot(x, tl.trans(w), input_precision="ieee")
        sq += tl.sum(x * x, 1)
    inv = tl.rsqrt(sq / (H * D) + NORM)
    sc = tl.load(S)
    ps = tl.load(S + SCALE_STRIDE)
    pre = (
        tl.sigmoid(
            acc * inv[:, None] * sc + tl.load(B + c * BASE_STRIDE, c < H, 0)[None, :]
        )
        + EPS
    )
    post = (
        MAG
        * tl.sigmoid(
            acc * inv[:, None] * ps
            + tl.load(B + c * BASE_STRIDE, c < 2 * H, 0)[None, :]
        )
        + EPS
    )
    tl.store(
        P + m[:, None] * H + c[None, :] - H,
        post,
        (m[:, None] < N) & (c[None, :] >= H) & (c[None, :] < 2 * H),
    )
    gates = ()
    for j in tl.static_range(H):
        gates += (tl.sum(tl.where(c[None, :] == j, pre, 0.0), 1),)
    for i in range(triton.cdiv(D, BD)):
        off = i * BD + tl.arange(0, BD)
        mask = (m[:, None] < N) & (off[None, :] < D)
        y = tl.full((BM, BD), 0, tl.float32)
        for j in tl.static_range(H):
            if CHANNEL_STRIDE:
                row = X + m[:, None] * TOKEN_STRIDE + j * CHANNEL_STRIDE
                x = _load_stride_two(row, i * BD, m < N, D, BM, BD)
            else:
                x = tl.load(X + (m[:, None] * H + j) * D + off[None, :], mask, 0).to(
                    tl.float32
                )
            y += gates[j][:, None] * x
        tl.store(Y + m[:, None] * D + off[None, :], y, mask)


class _IhcKernel:
    """Bind a compiled launch without retaining tensors or a device stream."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.runners = {}

    def bind(self, grid, pointers, parameters, warps=4):
        options = dict(enable_fp_fusion=False, num_warps=warps)
        if not triton.__version__.startswith("3.5."):

            return partial(self.kernel[grid], **options)
        # Keep pointer alignment and the exact scalar specialization in the key.
        scalars = tuple(v.hex() if isinstance(v, float) else v for v in parameters)
        key = (
            pointers[0].device.index,
            grid,
            pointers[0].dtype,
            tuple(t.data_ptr() % 16 == 0 for t in pointers),
            scalars,
            warps,
        )
        launch = self.runners.get(key)
        if launch is None:
            compiled = self.kernel.warmup(*pointers, *parameters, grid=grid, **options)
            launch = compiled[grid]
            if len(self.runners) >= 256:
                self.runners.clear()
            self.runners[key] = launch
        return launch


_short_launch = _IhcKernel(_short_projection)
_fused_launch = _IhcKernel(_project_and_gate)


def _launch(x, weight, scale, base, magnitude, hc_eps, norm_eps, pipeline):
    n, h, d = x.shape
    k = h * d
    if (
        not n
        or h > 4
        or x.numel() > 2**31 - 1
        or (k > 1024 and (h != 4 or d % 1024 or d > 8192))
    ):
        return _launch_ihc_pre(
            x, weight, scale, base, magnitude, hc_eps, norm_eps, pipeline
        )
    token_stride = channel_stride = 0
    if not x.is_contiguous():
        sn, sc, sd = x.stride()
        if k > 1024 and sd == 2 and sc > 0 and n * sn + h * sc + 2 * d < 2**31:
            token_stride, channel_stride = sn, sc
        else:
            x = x.contiguous()
    y = x.new_empty((n, d))
    post = weight.new_empty((n, h))
    if k <= 1024:
        bm, bk, bd = 1, triton.next_power_of_2(k), triton.next_power_of_2(d)
        launch = _short_launch
    else:
        # A wider projection tile amortizes the K loop on prefill.
        bm, bk, bd = (
            16,
            (1024 if n >= 64 or channel_stride else 512),
            (1024 if n >= 512 else 512),
        )
        launch = _fused_launch
    pointers = (x, weight, scale, base, y, post)
    parameters = (
        n,
        h,
        d,
        bm,
        bk,
        bd,
        scale.stride(0),
        base.stride(0),
        magnitude,
        hc_eps,
        norm_eps,
    )
    if k > 1024:
        parameters += (token_stride, channel_stride)
    grid = (triton.cdiv(n, bm), 1, 1)
    launch.bind(grid, pointers, parameters)(*pointers, *parameters)
    return y, post


def triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce iHC channels using FP32 projections and gates."""
    logger.debug("GEMS_ASCEND TRITON_IHC_PRE")
    return _ihc_pre(
        x,
        weight,
        scale,
        base,
        magnitude,
        hc_eps,
        norm_eps,
        _pipeline_stages=1,
        _launcher=_launch,
    )


def ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility entry for :func:`triton_ihc_pre`."""
    logger.debug("GEMS_ASCEND IHC_PRE")
    return _ihc_pre(
        x,
        weight,
        scale,
        base,
        magnitude,
        hc_eps,
        norm_eps,
        _pipeline_stages=1,
        _launcher=_launch,
    )

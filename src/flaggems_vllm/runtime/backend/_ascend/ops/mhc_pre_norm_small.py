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

"""Three-vector-kernel decode path with FP32 projection and BF16 premix."""
import math
from numbers import Integral, Real
from operator import index

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _project(
    X, W, P, K: tl.constexpr, SPLITS: tl.constexpr, BK: tl.constexpr, BG: tl.constexpr
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    part = tl.program_id(2)
    k = part * BK + tl.arange(0, BK)
    g = group * BG + tl.arange(0, BG)
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    w = tl.load(
        W + g[:, None] * K + k[None, :], (g[:, None] < 24) & (k[None, :] < K), 0
    )
    dot = tl.sum(w * x[None, :], 1)
    tl.store(P + (row * 25 + g) * SPLITS + part, dot, g < 24)
    if group == 0:
        tl.store(P + (row * 25 + 24) * SPLITS + part, tl.sum(x * x, 0))


@triton.jit
def _mix(
    P,
    SCALE,
    BASE,
    PRE,
    POST,
    COMB,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.arange(0, 32)
    s = tl.arange(0, BS)
    partial = tl.load(
        P + (row * 25 + g[:, None]) * SPLITS + s[None, :],
        (g[:, None] < 25) & (s[None, :] < SPLITS),
        0,
    )
    dots = tl.sum(partial, 1)
    square = tl.sum(tl.where(g == 24, dots, 0), 0)
    inv = 1.0 / tl.sqrt(square / K + EPS)
    n = tl.arange(0, 4)
    # Preserve native operation order: dot * inv, then scale, then bias.
    pre_dot = tl.sum(tl.where(g[None, :] == n[:, None], dots[None, :], 0), 1)
    post_dot = tl.sum(tl.where(g[None, :] == (n[:, None] + 4), dots[None, :], 0), 1)
    pre_z = (pre_dot * inv) * tl.load(SCALE) + tl.load(BASE + n)
    post_z = (post_dot * inv) * tl.load(SCALE + 1) + tl.load(BASE + n + 4)
    pre = 1.0 / (1.0 + tl.exp(-pre_z)) + PRE_EPS
    post = (1.0 / (1.0 + tl.exp(-post_z))) * POST_MULT
    tl.store(PRE + row * 4 + n, pre)
    tl.store(POST + row * 4 + n, post)
    gate = 8 + n[:, None] * 4 + n[None, :]
    # Direct reduction of the 16 comb gates avoids a dynamic gather on Ascend.
    p = tl.load(
        P + (row * 25 + gate[:, :, None]) * SPLITS + s[None, None, :],
        s[None, None, :] < SPLITS,
        0,
    )
    z = (tl.sum(p, 2) * inv) * tl.load(SCALE + 2) + tl.load(BASE + gate)
    c = tl.exp(z - tl.max(z, 1)[:, None])
    c = c / tl.sum(c, 1)[:, None] + SINKHORN_EPS
    c = c / (tl.sum(c, 0)[None, :] + SINKHORN_EPS)
    for _ in range(ITERS - 1):
        c = c / (tl.sum(c, 1)[:, None] + SINKHORN_EPS)
        c = c / (tl.sum(c, 0)[None, :] + SINKHORN_EPS)
    tl.store(COMB + row * 16 + n[:, None] * 4 + n[None, :], c)


@triton.jit
def _premix_norm(
    X, PRE, GAMMA, OUT, D: tl.constexpr, BD: tl.constexpr, NORM_EPS: tl.constexpr
):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    p0 = tl.load(PRE + row * 4)
    y = tl.load(X + row * 4 * D + d, d < D, 0).to(tl.float32) * p0
    for n in tl.static_range(1, 4):
        x = tl.load(X + (row * 4 + n) * D + d, d < D, 0).to(tl.float32)
        y = tl.fma(x, tl.load(PRE + row * 4 + n), y)
    # Native HcPre materializes BF16 before the subsequent native RMSNorm.
    y = y.to(OUT.dtype.element_ty).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(y * y, 0) / D + NORM_EPS)
    w = tl.load(GAMMA + d, d < D, 0).to(tl.float32)
    tl.store(OUT + row * D + d, (y * inv) * w, d < D)


def mhc_pre_norm_small(
    x,
    fn,
    scale,
    base,
    gamma,
    *,
    rms_eps=1e-5,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    norm_eps=1e-5,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
):
    if x.ndim != 3 or x.shape[1:] != (4, 4096):
        raise ValueError("residual must be [N,4,4096]")
    b = x.shape[0]
    if b > 4:
        raise NotImplementedError("small-row path requires N <= 4")
    if x.device.type != "npu":
        raise NotImplementedError("only Ascend is supported")
    specs = (
        (x, x.shape, torch.bfloat16),
        (fn, (24, 16384), torch.float32),
        (scale, (3,), torch.float32),
        (base, (24,), torch.float32),
        (gamma, (4096,), torch.bfloat16),
    )
    for tensor, shape, dtype in specs:
        if tensor.shape != shape or tensor.device != x.device:
            raise ValueError("mHC input shape/device mismatch")
        if tensor.dtype != dtype or not tensor.is_contiguous():
            raise NotImplementedError(
                "mHC requires contiguous BF16 residual/norm and FP32 coefficients"
            )
        if tensor.requires_grad:
            raise NotImplementedError("only forward inference is supported")
        if torch_npu.get_npu_format(tensor) not in (0, 2):
            raise NotImplementedError("mHC requires standard ND/NCHW storage")
    if not isinstance(sinkhorn_repeat, Integral):
        raise ValueError("sinkhorn_repeat must be a host integer")
    iters = index(sinkhorn_repeat)
    if iters < 1:
        raise ValueError("sinkhorn_repeat must be positive")
    params = (rms_eps, hc_pre_eps, hc_sinkhorn_eps, norm_eps, hc_post_mult_value)
    if not all(isinstance(v, Real) and math.isfinite(v) and v >= 0 for v in params):
        raise ValueError("epsilons and post multiplier must be finite and nonnegative")
    rms_eps, hc_pre_eps, hc_sinkhorn_eps, norm_eps, hc_post_mult_value = map(
        float, params
    )
    post = torch.empty((b, 4), device=x.device, dtype=torch.float32)
    comb = torch.empty((b, 4, 4), device=x.device, dtype=torch.float32)
    out = torch.empty((b, 4096), device=x.device, dtype=torch.bfloat16)
    if b:
        # Ascend-only fixed configuration; see the documented autotune exemption.
        k, bk, bg, splits = 16384, 1024, 4, 16
        partial = torch.empty((b, 25, splits), device=x.device, dtype=torch.float32)
        pre = torch.empty_like(post)
        _project[(b, 6, splits)](
            x, fn, partial, k, splits, bk, bg, enable_fp_fusion=False
        )
        _mix[(b,)](
            partial,
            scale,
            base,
            pre,
            post,
            comb,
            k,
            splits,
            splits,
            iters,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            enable_fp_fusion=False,
        )
        _premix_norm[(b,)](
            x, pre, gamma, out, 4096, 4096, norm_eps, enable_fp_fusion=False
        )
    return post.view(b, 4, 1), comb, out

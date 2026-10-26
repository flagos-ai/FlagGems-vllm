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

"""Hygon (gfx93x DCU) parts of ``ops/mhc/mhc_pre.py``."""

import triton
import triton.language as tl

from flaggems_vllm.ops.mhc.mhc_pre import _sinkhorn_iters

KERNELS = (
    "mhc_pre_mm_fuse_kernel",
    "mhc_pre_mm_fuse_tiled_kernel",
    "mhc_pre_generic_kernel",
    "mhc_pre_packed_reader_kernel",
    "mhc_pre_sinkhorn_fuse_kernel",
    "mhc_pre_single_pass_kernel",
)


class FastLaunchKernel:
    """``kernel[grid](...args)`` with the autotuner wrapper removed."""

    __slots__ = ("_inner", "_cache")

    def __init__(self, inner):
        self._inner = inner
        self._cache = {}

    def __getattr__(self, name):

        return getattr(self._inner, name)

    @staticmethod
    def _proxy_key(grid, args, kwargs):
        if callable(grid):
            return None
        return (
            grid,
            tuple(getattr(a, "dtype", a) for a in args),
            tuple((k, getattr(v, "dtype", v)) for k, v in kwargs.items()),
        )

    def __getitem__(self, grid):
        inner = self._inner
        cache = self._cache
        proxy_key = self._proxy_key

        def launcher(*args, **kwargs):
            key = proxy_key(grid, args, kwargs)
            cfg = None if key is None else cache.get(key)
            if cfg is None:
                inner[grid](*args, **kwargs)
                best = getattr(inner, "best_config", None)
                if (
                    key is not None
                    and best is not None
                    and getattr(best, "pre_hook", None) is None
                ):
                    cache[key] = dict(
                        num_warps=best.num_warps,
                        num_stages=best.num_stages,
                        **best.kwargs,
                    )
                return None
            return inner.fn[grid](*args, **kwargs, **cfg)

        return launcher


def _wrap(namespace, names):
    wrapped = 0
    for name in names:
        kernel = namespace.get(name)
        if (
            kernel is not None
            and getattr(kernel, "configs", None)
            and not isinstance(kernel, FastLaunchKernel)
        ):
            namespace[name] = FastLaunchKernel(kernel)
            wrapped += 1
    return wrapped


def install_fastlaunch(namespace, names):
    """Launch the hot kernels through the JITFunction, skipping the autotuner,
    which rebuilds several dicts per call.
    """

    return _wrap(namespace, names) + _wrap(globals(), names)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["hidden_size"],
)
@triton.jit
def mhc_pre_single_pass_kernel(
    BT_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    scratch_ptr,
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    K_LEN: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    _single_pass_body(
        BT_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        layer_input_ptr,
        scratch_ptr,
        num_tokens,
        num_tokens_bucket,
        res_stride_n,
        res_stride_i,
        li_stride_n,
        li_stride_h,
        hidden_size,
        hc_hidden_size,
        K_LEN,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        HC_MULT3,
        TOKEN_TILE,
        BLOCK_K,
        BLOCK_H,
    )


@triton.jit
def _single_pass_body(
    BT_ptr,  # (hc_hidden_size, 32) bf16 packed fn^T, rows 0..23 used
    hc_scale_ptr,  # (3,), float32
    hc_base_ptr,  # (hc_mult3,), float32
    residual_ptr,  # (num_tokens, hc_hidden) bf16, contiguous
    post_mix_ptr,  # (num_tokens, hc_mult), float32
    comb_mix_ptr,  # (num_tokens, hc_mult*hc_mult), float32
    layer_input_ptr,  # (num_tokens, hidden_size), bfloat16
    scratch_ptr,  # (num_tokens, 32) float32 dot-fragment spill
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    K_LEN: tl.constexpr,  # == hc_hidden_size, makes the K-loop static
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One launch: GEMM, sqrsum, mixes and weighted sum over one tile."""
    pid_n = tl.program_id(0)
    offs_t = pid_n * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
    tmask = offs_t < num_tokens
    res_row = offs_t * res_stride_n

    offs_n = tl.arange(0, 32)
    acc = tl.zeros((TOKEN_TILE, 32), dtype=tl.float32)
    sq2 = tl.zeros((TOKEN_TILE, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K_LEN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            residual_ptr + res_row[:, None] + offs_k[None, :],
            mask=tmask[:, None],
            other=0.0,
        )
        b = tl.load(BT_ptr + offs_k[:, None] * 32 + offs_n[None, :])
        acc += tl.dot(a, b)
        af = a.to(tl.float32)
        sq2 += af * af
    sq = tl.sum(sq2, axis=1)

    rms_inv = tl.rsqrt(sq / hc_hidden_size + rms_eps)

    acc = acc.to(tl.bfloat16).to(tl.float32)

    tl.store(
        scratch_ptr + offs_t[:, None] * 32 + tl.arange(0, 32)[None, :],
        acc,
        mask=tmask[:, None],
    )
    tl.debug_barrier()

    # ══ Load scales ══
    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)

    # ══ pre_mix: indices 0..3 ══
    pre_mix_0 = (
        tl.sigmoid(
            tl.load(scratch_ptr + offs_t * 32 + 0) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 0)
        )
        + hc_pre_eps
    )
    pre_mix_1 = (
        tl.sigmoid(
            tl.load(scratch_ptr + offs_t * 32 + 1) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 1)
        )
        + hc_pre_eps
    )
    pre_mix_2 = (
        tl.sigmoid(
            tl.load(scratch_ptr + offs_t * 32 + 2) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 2)
        )
        + hc_pre_eps
    )
    pre_mix_3 = (
        tl.sigmoid(
            tl.load(scratch_ptr + offs_t * 32 + 3) * rms_inv * scale_0
            + tl.load(hc_base_ptr + 3)
        )
        + hc_pre_eps
    )

    # ══ post_mix: indices 4..7 ══
    post_0 = (
        tl.sigmoid(
            tl.load(scratch_ptr + offs_t * 32 + 4) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 4)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + offs_t * 4 + 0, post_0, mask=tmask)
    post_1 = (
        tl.sigmoid(
            tl.load(scratch_ptr + offs_t * 32 + 5) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 5)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + offs_t * 4 + 1, post_1, mask=tmask)
    post_2 = (
        tl.sigmoid(
            tl.load(scratch_ptr + offs_t * 32 + 6) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 6)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + offs_t * 4 + 2, post_2, mask=tmask)
    post_3 = (
        tl.sigmoid(
            tl.load(scratch_ptr + offs_t * 32 + 7) * rms_inv * scale_1
            + tl.load(hc_base_ptr + 7)
        )
        * hc_post_mult_value
    )
    tl.store(post_mix_ptr + offs_t * 4 + 3, post_3, mask=tmask)

    # ══ comb_mix: indices 8..23 → 4x4 Sinkhorn ══
    cb = 8
    cm_00 = tl.load(scratch_ptr + offs_t * 32 + cb + 0) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 0
    )
    cm_01 = tl.load(scratch_ptr + offs_t * 32 + cb + 1) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 1
    )
    cm_02 = tl.load(scratch_ptr + offs_t * 32 + cb + 2) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 2
    )
    cm_03 = tl.load(scratch_ptr + offs_t * 32 + cb + 3) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 3
    )
    cm_10 = tl.load(scratch_ptr + offs_t * 32 + cb + 4) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 4
    )
    cm_11 = tl.load(scratch_ptr + offs_t * 32 + cb + 5) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 5
    )
    cm_12 = tl.load(scratch_ptr + offs_t * 32 + cb + 6) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 6
    )
    cm_13 = tl.load(scratch_ptr + offs_t * 32 + cb + 7) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 7
    )
    cm_20 = tl.load(scratch_ptr + offs_t * 32 + cb + 8) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 8
    )
    cm_21 = tl.load(scratch_ptr + offs_t * 32 + cb + 9) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 9
    )
    cm_22 = tl.load(scratch_ptr + offs_t * 32 + cb + 10) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 10
    )
    cm_23 = tl.load(scratch_ptr + offs_t * 32 + cb + 11) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 11
    )
    cm_30 = tl.load(scratch_ptr + offs_t * 32 + cb + 12) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 12
    )
    cm_31 = tl.load(scratch_ptr + offs_t * 32 + cb + 13) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 13
    )
    cm_32 = tl.load(scratch_ptr + offs_t * 32 + cb + 14) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 14
    )
    cm_33 = tl.load(scratch_ptr + offs_t * 32 + cb + 15) * rms_inv * scale_2 + tl.load(
        hc_base_ptr + cb + 15
    )

    (
        cm_00,
        cm_01,
        cm_02,
        cm_03,
        cm_10,
        cm_11,
        cm_12,
        cm_13,
        cm_20,
        cm_21,
        cm_22,
        cm_23,
        cm_30,
        cm_31,
        cm_32,
        cm_33,
    ) = _sinkhorn_iters(
        cm_00,
        cm_01,
        cm_02,
        cm_03,
        cm_10,
        cm_11,
        cm_12,
        cm_13,
        cm_20,
        cm_21,
        cm_22,
        cm_23,
        cm_30,
        cm_31,
        cm_32,
        cm_33,
        hc_sinkhorn_eps,
        sinkhorn_repeat,
    )

    co = offs_t * 16
    tl.store(comb_mix_ptr + co + 0, cm_00, mask=tmask)
    tl.store(comb_mix_ptr + co + 1, cm_01, mask=tmask)
    tl.store(comb_mix_ptr + co + 2, cm_02, mask=tmask)
    tl.store(comb_mix_ptr + co + 3, cm_03, mask=tmask)
    tl.store(comb_mix_ptr + co + 4, cm_10, mask=tmask)
    tl.store(comb_mix_ptr + co + 5, cm_11, mask=tmask)
    tl.store(comb_mix_ptr + co + 6, cm_12, mask=tmask)
    tl.store(comb_mix_ptr + co + 7, cm_13, mask=tmask)
    tl.store(comb_mix_ptr + co + 8, cm_20, mask=tmask)
    tl.store(comb_mix_ptr + co + 9, cm_21, mask=tmask)
    tl.store(comb_mix_ptr + co + 10, cm_22, mask=tmask)
    tl.store(comb_mix_ptr + co + 11, cm_23, mask=tmask)
    tl.store(comb_mix_ptr + co + 12, cm_30, mask=tmask)
    tl.store(comb_mix_ptr + co + 13, cm_31, mask=tmask)
    tl.store(comb_mix_ptr + co + 14, cm_32, mask=tmask)
    tl.store(comb_mix_ptr + co + 15, cm_33, mask=tmask)

    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        r0 = tl.load(
            residual_ptr + res_row[:, None] + h_offsets[None, :],
            mask=tmask[:, None],
            other=0.0,
        ).to(tl.float32)
        r1 = tl.load(
            residual_ptr + res_row[:, None] + res_stride_i + h_offsets[None, :],
            mask=tmask[:, None],
            other=0.0,
        ).to(tl.float32)
        r2 = tl.load(
            residual_ptr + res_row[:, None] + 2 * res_stride_i + h_offsets[None, :],
            mask=tmask[:, None],
            other=0.0,
        ).to(tl.float32)
        r3 = tl.load(
            residual_ptr + res_row[:, None] + 3 * res_stride_i + h_offsets[None, :],
            mask=tmask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc_li = pre_mix_0[:, None] * r0 + pre_mix_1[:, None] * r1
        acc_li += pre_mix_2[:, None] * r2 + pre_mix_3[:, None] * r3
        tl.store(
            layer_input_ptr
            + (offs_t * li_stride_n)[:, None]
            + h_offsets[None, :] * li_stride_h,
            acc_li.to(tl.bfloat16),
            mask=tmask[:, None],
        )

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

"""MetaX C550 specialization of the shifted mHC post -> pre boundary.

Two round trips dominate the generic path on the DSV4.1 boundary shapes:

* the post kernel materialises ``residual_cur`` and the collapse kernel reads
  it back, and
* the project kernel folds the row sum of squares into the pipelined K loop,
  which pins ``num_stages`` to 1 and serialises a single CTA for decode-sized
  batches.

This module keeps the post arithmetic verbatim (the post output has to stay
bitwise equal to ``mhc_post``) and changes the dataflow:

1. the post kernel computes the collapse and the per-token sum of squares while
   the just-rounded bf16 values are still in registers;
2. the project kernel becomes a pure skinny GEMM, with the ``fn`` weight packed
   once into a contiguous ``(hc*H, mix_size)`` bf16 tensor and the K dimension
   split across CTAs for small token counts.

The split-K and row-sum partials are combined by fixed-order reduction
kernels, so repeated calls stay bitwise deterministic even though the project
stage is parallelised over K.
"""

import weakref

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_FN_PACK_CACHE: dict = {}


@triton.jit
def _pack_fn_bf16_kernel(
    src_ptr,  # (mix_size, K) float32, contiguous
    dst_ptr,  # (K, mix_size) bfloat16, contiguous
    numel,
    K,
    mix_size,
    BLOCK: tl.constexpr,
):
    """One-time ``fn`` transpose-cast for the project kernel's B operands."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    n = offs // K
    k = offs % K
    value = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + k * mix_size + n, value.to(tl.bfloat16), mask=mask)


def _fn_bf16_packed(fn: torch.Tensor) -> torch.Tensor:
    """``fn`` as a contiguous ``(hc*H, mix_size)`` bfloat16 tensor, cached."""
    key = id(fn)
    version = None if fn.is_inference() else fn._version
    ptr = fn.data_ptr()
    entry = _FN_PACK_CACHE.get(key)
    if entry is not None:
        ref, cached_version, cached_ptr, cached = entry
        if ref() is fn and cached_version == version and cached_ptr == ptr:
            return cached

    mix_size, K = fn.shape
    out = torch.empty(K, mix_size, dtype=torch.bfloat16, device=fn.device)
    if fn.numel():
        _pack_fn_bf16_kernel[lambda meta: (triton.cdiv(fn.numel(), meta["BLOCK"]),)](
            fn, out, fn.numel(), K, mix_size, BLOCK=1024
        )
    _FN_PACK_CACHE[key] = (
        weakref.ref(fn, lambda ref, key=key: _FN_PACK_CACHE.pop(key, None)),
        version,
        ptr,
        out,
    )
    return out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_project_metax"),
    key=["num_tokens", "hc_hidden_size", "split_k"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_project_metax_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, mix_size) bfloat16, contiguous
    out_ptr,  # (split_k, num_tokens, mix_size) float32 partials
    num_tokens,
    hc_hidden_size,
    mix_size,
    a_stride_n,
    split_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Pure ``a @ fn^T`` with the K dimension optionally split across CTAs."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens
    n_mask = offs_n < mix_size

    k_per_split = tl.cdiv(hc_hidden_size, split_k)
    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, hc_hidden_size)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_end
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            bt_ptr + offs_k[:, None] * mix_size + offs_n[None, :],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)

    # Write the split-K partials; a separate fixed-order reduction makes the
    # result bitwise deterministic (the epsilon-wiring test compares two calls
    # with rtol=0, atol=0).
    tl.store(
        out_ptr + (pid_k * num_tokens + offs_m[:, None]) * mix_size + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _mhc_shifted_split_k_reduce_kernel(
    part_ptr,  # (split_k, numel) float32
    out_ptr,  # (numel,) float32
    numel,
    split_k,
    BLOCK: tl.constexpr,
):
    """Sum the split-K partials in a fixed split order."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(split_k):
        acc += tl.load(part_ptr + k * numel + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_post_collapse"),
    key=["H"],
    warmup=5,
    rep=10,
    # The tuner benchmarks configs with different BLOCK_H, hence different
    # tile counts, against the same partials buffer.  A short config must not
    # leave a long config's slots behind for the ssq reduction.
    reset_to_zero=["part_ptr"],
)
@triton.jit
def _mhc_shifted_post_collapse_kernel(
    a_ptr,  # comb_res_mix : (N, 4, 4), float32 -- vLLM (j, i) orientation
    b_ptr,  # residual     : (N, 4, H), bfloat16
    c_ptr,  # post_layer_mix: (N, 4), float32
    d_ptr,  # x            : (N, H), bfloat16
    pre_ptr,  # pre_mix    : (N, 4), float32, or a dummy when USE_PRE is off
    out_ptr,  # residual_cur: (N, 4, H), bfloat16
    layer_ptr,  # layer_input: (N, H), bfloat16
    aux_ptr,  # aux         : (N, H), bfloat16, or a dummy when WRITE_AUX is off
    part_ptr,  # (N, max_tiles) float32, zero-initialised partial sums
    a_stride_n,
    a_stride_j,
    a_stride_i,
    max_tiles,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
):
    """Fused hc=4 post + collapse + row sum of squares."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    a_base = pid_n * a_stride_n
    c_base = pid_n * 4
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    out_base = pid_n * 4 * H

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

    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + b_base + 0 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + b_base + 1 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b2 = tl.load(b_ptr + b_base + 2 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b3 = tl.load(b_ptr + b_base + 3 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)

    # Identical fma chain to mhc_post_kernel_hc_mult_4: the post output must
    # stay bitwise equal to mhc_post.
    acc0 = tl.fma(
        a30, b3, tl.fma(a20, b2, tl.fma(a10, b1, tl.fma(a00, b0, c0 * d_vals)))
    )
    acc1 = tl.fma(
        a31, b3, tl.fma(a21, b2, tl.fma(a11, b1, tl.fma(a01, b0, c1 * d_vals)))
    )
    acc2 = tl.fma(
        a32, b3, tl.fma(a22, b2, tl.fma(a12, b1, tl.fma(a02, b0, c2 * d_vals)))
    )
    acc3 = tl.fma(
        a33, b3, tl.fma(a23, b2, tl.fma(a13, b1, tl.fma(a03, b0, c3 * d_vals)))
    )

    v0 = acc0.to(tl.bfloat16)
    v1 = acc1.to(tl.bfloat16)
    v2 = acc2.to(tl.bfloat16)
    v3 = acc3.to(tl.bfloat16)
    tl.store(out_ptr + out_base + 0 * H + h_off, v0, mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_off, v1, mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_off, v2, mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_off, v3, mask=h_mask)

    # The generic collapse loads the materialised bf16 residual_cur; converting
    # the stored values back to fp32 keeps the same rounding boundary.
    f0 = v0.to(tl.float32)
    f1 = v1.to(tl.float32)
    f2 = v2.to(tl.float32)
    f3 = v3.to(tl.float32)
    if USE_PRE:
        p0 = tl.load(pre_ptr + pid_n * 4 + 0)
        p1 = tl.load(pre_ptr + pid_n * 4 + 1)
        p2 = tl.load(pre_ptr + pid_n * 4 + 2)
        p3 = tl.load(pre_ptr + pid_n * 4 + 3)
        col = f0 * p0
        col += f1 * p1
        col += f2 * p2
        col += f3 * p3
    else:
        col = f0
    tl.store(layer_ptr + pid_n * H + h_off, col.to(tl.bfloat16), mask=h_mask)
    if WRITE_AUX:
        total = f0 + f1
        total += f2
        total += f3
        total = total / 4.0
        tl.store(aux_ptr + pid_n * H + h_off, total.to(tl.bfloat16), mask=h_mask)

    # Per-(token, h-tile) row sum of squares; a separate fixed-order reduction
    # combines the tiles deterministically.
    ssq = f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3
    tl.store(part_ptr + pid_n * max_tiles + pid_h, tl.sum(ssq))


@triton.jit
def _mhc_shifted_ssq_reduce_kernel(
    part_ptr,  # (num_tokens, max_tiles) float32
    ssq_ptr,  # (num_tokens,) float32
    max_tiles,
):
    """Sum the per-tile partials in a fixed tile order."""
    n = tl.program_id(0)
    acc = 0.0
    for t in range(max_tiles):
        acc += tl.load(part_ptr + n * max_tiles + t)
    tl.store(ssq_ptr + n, acc)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_norm_metax"),
    key=["D"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mhc_shifted_norm_metax_kernel(
    X,  # (num_tokens, D) bfloat16
    W,  # (D,) bfloat16
    Y,  # (num_tokens, D) bfloat16
    D,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Single-pass RMSNorm: load the row once, reduce, normalise and store."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    value = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ssq = tl.sum(value * value, axis=0)
    inv = tl.rsqrt(ssq / D + EPS)
    result = value * inv * tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * D + offs, result.to(tl.bfloat16), mask=mask)


def mhc_shifted_norm_metax(x, weight, eps):
    """One-pass RMSNorm when the row fits in one CTA (D <= 8192)."""
    D = x.shape[1]
    out = torch.empty_like(x)
    block_d = triton.next_power_of_2(D)
    # BLOCK_D must cover the whole row; aabs must not rewrite it to a divisor.
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_norm_metax_kernel[(x.shape[0],)](
            x, weight, out, D, EPS=eps, BLOCK_D=block_d
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs
    return out


def _split_k_for(num_tokens, hc_hidden_size):
    """Split K until the grid covers the C550 CUs, without shattering K."""
    est_m = max(1, triton.cdiv(num_tokens, 32))
    occupancy = 104 // est_m
    # One M wave leaves most of the 104 CUs idle (e.g. the 2048-token
    # boundary); fill the machine with K splits instead.  Larger batches
    # already saturate the memory system, so they keep split_k=1.
    if occupancy == 1 and num_tokens < 4096:
        occupancy = 8
    split = max(1, min(8, occupancy))
    return min(split, max(1, hc_hidden_size // 256))


def mhc_shifted_post_collapse(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    pre_mix,
    num_tokens,
    hidden_size,
    capture_aux,
):
    """Return ``(residual_cur, layer_input, aux, ssq)`` with post fused."""
    device = residual.device
    residual_cur = torch.empty_like(residual)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    aux = (
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        if capture_aux
        else residual_cur.new_empty(0, hidden_size)
    )
    # 128 is the smallest BLOCK_H in the tuned space, so every tuned tile
    # partition has at most max_tiles slots; the zeros cover unused slots.
    max_tiles = triton.cdiv(hidden_size, 128)
    partials = torch.zeros(num_tokens, max_tiles, dtype=torch.float32, device=device)
    ssq = torch.empty(num_tokens, dtype=torch.float32, device=device)

    st = comb_res_mix.stride()
    a_sn = st[0]
    # vLLM orientation (MixLayout.TRANSPOSED): stored mix is [j, i].
    a_sj, a_si = st[1], st[2]

    # FlagTree aabs rewrites the tile constexprs during autotuning; the sweep
    # below is what we actually want to compare, so keep it off for the launch
    # (and therefore for the tuner's benchmark runs) and restore it afterwards.
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_post_collapse_kernel[
            lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_H"]))
        ](
            comb_res_mix,
            residual,
            post_layer_mix,
            x,
            pre_mix if pre_mix is not None else post_layer_mix,
            residual_cur,
            layer_input,
            aux if capture_aux else layer_input,
            partials,
            a_sn,
            a_sj,
            a_si,
            max_tiles,
            H=hidden_size,
            USE_PRE=pre_mix is not None,
            WRITE_AUX=capture_aux,
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs

    _mhc_shifted_ssq_reduce_kernel[(num_tokens,)](partials, ssq, max_tiles)
    return residual_cur, layer_input, aux, ssq


def mhc_shifted_project_metax(a_flat, fn_packed, num_tokens, mix_size):
    """Return ``a @ fn^T`` for the validated contiguous bf16 ``a_flat``."""
    split_k = _split_k_for(num_tokens, a_flat.shape[1])
    partials = torch.empty(
        split_k, num_tokens, mix_size, dtype=torch.float32, device=a_flat.device
    )
    # See mhc_shifted_post_collapse: keep aabs from rewriting the tuned tiles.
    aabs = triton.knobs.autotuning.adjust_block_size
    triton.knobs.autotuning.adjust_block_size = False
    try:
        _mhc_shifted_project_metax_kernel[
            lambda meta: (
                triton.cdiv(num_tokens, meta["BLOCK_M"]),
                triton.cdiv(mix_size, meta["BLOCK_N"]),
                split_k,
            )
        ](
            a_flat,
            fn_packed,
            partials,
            num_tokens,
            a_flat.shape[1],
            mix_size,
            a_flat.stride(0),
            split_k,
        )
    finally:
        triton.knobs.autotuning.adjust_block_size = aabs

    if split_k == 1:
        return partials[0]

    out = torch.empty(num_tokens, mix_size, dtype=torch.float32, device=a_flat.device)
    numel = num_tokens * mix_size
    _mhc_shifted_split_k_reduce_kernel[(triton.cdiv(numel, 1024),)](
        partials, out, numel, split_k, BLOCK=1024
    )
    return out

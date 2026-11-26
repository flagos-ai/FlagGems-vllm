# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Hygon BW1000-optimized Triton implementation of the vLLM Mamba-1 mixer.

Adaptations and optimizations for Hygon DCU (gfx936, warp_size=64):
  * FP32 GEMM uses IEEE precision (``input_precision="ieee"``) instead of
    ``tf32x3`` (unsupported by the Hygon HCU Triton backend).
  * Small-M (M <= 4) linear projections use dedicated GEMV / deterministic
    Split-K kernels rather than padded 16xN ``tl.dot`` tiles, keeping all 80 CUs
    busy on narrow projections such as ``x_proj``.
  * Larger-M linear projections use BW1000-tuned ``tl.dot`` tiles with L2
    swizzling (``GROUP_M=8``) while preserving compile-time specialization on
    contiguous unit strides so Triton emits vectorized 128-bit loads/stores.
  * Prefill Conv1D is parallelized across both token tiles (``BLOCK_T=32``)
    and channel tiles (``BLOCK_I=64``), with a separate race-free cache-state
    update kernel.
  * Prefill Selective Scan (>64 tokens) uses a 3-pass chunked parallel
    associative scan (``CHUNK_SIZE=64``) with coalesced ``tl.trans`` I/O and
    the time axis placed innermost (``axis=2``) so each 64-lane wavefront
    performs intra-warp register shuffles instead of cross-warp LDS barriers.
  * Optional 3-way RMSNorm (dt, B, C) is fused into a single kernel launch.
  * Autotune exemption: tile configurations are fixed from BW1000 sweeps and
    compiled kernels are cached and launched directly via ``_COMPILED``,
    eliminating ``@libentry`` / ``@libtuner`` CPU dispatch overhead and
    avoiding stateful cache replay.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False

_COMPILED: dict = {}

_CONV_BLOCK_I_DECODE = 64
_CONV_BLOCK_T_PREFILL = 32
_CONV_BLOCK_I_PREFILL = 64

_SCAN_BLOCK_I_SMALL = 16
_SCAN_CHUNK_SIZE = 64
_SCAN_CHUNK_BLOCK_I = 4
_SCAN_INTER_BLOCK_I = 16


# ---------------------------------------------------------------------------
# Linear kernels (GEMV / Split-K for small M, swizzled GEMM for larger M)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["M", "N", "K", "sx_m", "sw_n", "sy_m"])
def _hygon_gemv_kernel(
    X,
    W,
    BIAS,
    Y,
    M,
    N,
    K,
    sx_m,
    sx_k,
    sw_n,
    sw_k,
    sy_m,
    sy_n,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    x_row = X + pid_m * sx_m
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        k_mask = k < K
        xv = tl.load(x_row + k * sx_k, mask=k_mask, other=0.0).to(tl.float32)
        wv = tl.load(
            W + offs_n[:, None] * sw_n + k[None, :] * sw_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(wv * xv[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(
        Y + pid_m * sy_m + offs_n * sy_n,
        acc.to(Y.dtype.element_ty),
        mask=n_mask,
    )


@triton.jit(do_not_specialize=["M", "N", "K", "sx_m", "sw_n"])
def _hygon_gemv_splitk_kernel(
    X,
    W,
    Y_PART,
    M,
    N,
    K,
    sx_m,
    sx_k,
    sw_n,
    sw_k,
    SPLIT_K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BN,), dtype=tl.float32)
    x_row = X + pid_m * sx_m
    for k0 in range(k_start, k_end, BK):
        k = k0 + tl.arange(0, BK)
        k_mask = k < k_end
        xv = tl.load(x_row + k * sx_k, mask=k_mask, other=0.0).to(tl.float32)
        wv = tl.load(
            W + offs_n[:, None] * sw_n + k[None, :] * sw_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(wv * xv[None, :], axis=1)
    tl.store(Y_PART + (pid_k * M + pid_m) * N + offs_n, acc, mask=n_mask)


@triton.jit(do_not_specialize=["M", "N", "sy_m"])
def _hygon_reduce_splitk_kernel(
    Y_PART,
    BIAS,
    Y,
    M,
    N,
    sy_m,
    sy_n,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for sk in tl.static_range(SPLIT_K):
        acc += tl.load(Y_PART + (sk * M + pid_m) * N + offs_n, mask=n_mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(
        Y + pid_m * sy_m + offs_n * sy_n,
        acc.to(Y.dtype.element_ty),
        mask=n_mask,
    )


@triton.jit(do_not_specialize=["M", "N", "K", "sx_m", "sw_n", "sy_m"])
def _hygon_gemm_kernel(
    X,
    W,
    BIAS,
    Y,
    M,
    N,
    K,
    sx_m,
    sx_k,
    sw_n,
    sw_k,
    sy_m,
    sy_n,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Y = X @ W.T + bias, with W stored as [N, K]."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + offs_k
        x = tl.load(
            X + offs_m[:, None] * sx_m + k[None, :] * sx_k,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            W + offs_n[None, :] * sw_n + k[:, None] * sw_k,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        if X.dtype.element_ty == tl.float32:
            acc += tl.dot(x, w, out_dtype=tl.float32, input_precision="ieee")
        else:
            acc += tl.dot(x, w, out_dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    tl.store(
        Y + offs_m[:, None] * sy_m + offs_n[None, :] * sy_n,
        acc.to(Y.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _linear_hygon(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    dev_idx: int,
    dtype: torch.dtype,
    x_align: bool,
) -> None:
    m, k = x.shape
    n = weight.shape[0]
    has_bias = bias is not None
    sx_m, sx_k = x.stride(0), x.stride(1)
    sw_n, sw_k = weight.stride(0), weight.stride(1)
    sy_m, sy_n = out.stride(0), out.stride(1)

    if m <= 4:
        if n <= 256 and k >= 512:
            split_k = 8 if m == 1 else 4
            bn, bk = 16, 128
            y_part = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
            grid1 = ((n + bn - 1) // bn, split_k, m)
            key1 = ("gemv_sk", dev_idx, dtype, x_align, split_k, bn, bk)
            comp1 = _COMPILED.get(key1)
            if comp1 is None:
                comp1 = _hygon_gemv_splitk_kernel[grid1](
                    x,
                    weight,
                    y_part,
                    m,
                    n,
                    k,
                    sx_m,
                    sx_k,
                    sw_n,
                    sw_k,
                    SPLIT_K=split_k,
                    BN=bn,
                    BK=bk,
                    num_warps=4,
                    num_stages=2,
                )
                _COMPILED[key1] = comp1
            else:
                comp1[grid1](
                    x, weight, y_part, m, n, k, sx_m, sx_k, sw_n, sw_k, split_k, bn, bk
                )

            red_bn = 64
            grid2 = (m, (n + red_bn - 1) // red_bn, 1)
            key2 = ("red_sk", dev_idx, dtype, has_bias, split_k, red_bn)
            comp2 = _COMPILED.get(key2)
            if comp2 is None:
                comp2 = _hygon_reduce_splitk_kernel[grid2](
                    y_part,
                    bias,
                    out,
                    m,
                    n,
                    sy_m,
                    sy_n,
                    HAS_BIAS=has_bias,
                    SPLIT_K=split_k,
                    BN=red_bn,
                    num_warps=4,
                    num_stages=2,
                )
                _COMPILED[key2] = comp2
            else:
                comp2[grid2](
                    y_part, bias, out, m, n, sy_m, sy_n, has_bias, split_k, red_bn
                )
            return

        bn, bk = (64, 64) if k <= 128 else (16, 128)
        grid = ((n + bn - 1) // bn, m, 1)
        key = ("gemv", dev_idx, dtype, x_align, has_bias, bn, bk)
        comp = _COMPILED.get(key)
        if comp is None:
            comp = _hygon_gemv_kernel[grid](
                x,
                weight,
                bias,
                out,
                m,
                n,
                k,
                sx_m,
                sx_k,
                sw_n,
                sw_k,
                sy_m,
                sy_n,
                HAS_BIAS=has_bias,
                BN=bn,
                BK=bk,
                num_warps=4,
                num_stages=2,
            )
            _COMPILED[key] = comp
        else:
            comp[grid](
                x,
                weight,
                bias,
                out,
                m,
                n,
                k,
                sx_m,
                sx_k,
                sw_n,
                sw_k,
                sy_m,
                sy_n,
                has_bias,
                bn,
                bk,
            )
        return

    if m <= 16:
        bm, bn, bk, nw = 16, 64, 64, 4
    elif m <= 32 or n <= 256:
        bm, bn, bk, nw = 32, 64, 64, 4
    elif k <= 128:
        bm, bn, bk, nw = 128, 64, 32, 4
    elif m <= 512:
        bm, bn, bk, nw = 64, 128, 32, 8
    else:
        bm, bn, bk, nw = 128, 128, 32, 8

    grid = (((m + bm - 1) // bm) * ((n + bn - 1) // bn), 1, 1)
    key = ("gemm", dev_idx, dtype, x_align, has_bias, bm, bn, bk, nw)
    comp = _COMPILED.get(key)
    if comp is None:
        comp = _hygon_gemm_kernel[grid](
            x,
            weight,
            bias,
            out,
            m,
            n,
            k,
            sx_m,
            sx_k,
            sw_n,
            sw_k,
            sy_m,
            sy_n,
            HAS_BIAS=has_bias,
            BM=bm,
            BN=bn,
            BK=bk,
            GROUP_M=8,
            num_warps=nw,
            num_stages=2,
        )
        _COMPILED[key] = comp
    else:
        comp[grid](
            x,
            weight,
            bias,
            out,
            m,
            n,
            k,
            sx_m,
            sx_k,
            sw_n,
            sw_k,
            sy_m,
            sy_n,
            has_bias,
            bm,
            bn,
            bk,
            8,
        )


# ---------------------------------------------------------------------------
# Conv1D kernels (Decode single-kernel, Small varlen, Parallel 2D prefill)
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "total_tokens",
        "intermediate_size",
        "num_sequences",
        "stride_w_i",
        "stride_w_k",
        "stride_state_slot",
        "stride_state_i",
        "stride_state_k",
    ]
)
def _hygon_conv1d_varlen_kernel(
    PROJ,
    WEIGHT,
    BIAS,
    CONV_STATE,
    QUERY_START_LOC,
    CACHE_INDICES,
    HAS_INITIAL_STATE,
    CONV_OUT,
    total_tokens,
    intermediate_size,
    num_sequences,
    stride_w_i,
    stride_w_k,
    stride_state_slot,
    stride_state_i,
    stride_state_k,
    HAS_BIAS: tl.constexpr,
    HAS_CONV_STATE: tl.constexpr,
    HAS_QUERY_START_LOC: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE_PTR: tl.constexpr,
    INIT_ALL_TRUE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Depthwise causal Conv1D for decode or small varlen prefill."""
    seq_id = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    channel_mask = channel < intermediate_size

    if IS_DECODE:
        seq_start = seq_id
        seq_end = seq_id + 1
    elif HAS_QUERY_START_LOC:
        seq_start = tl.load(QUERY_START_LOC + seq_id).to(tl.int32)
        seq_end = tl.load(QUERY_START_LOC + seq_id + 1).to(tl.int32)
    else:
        seq_start = 0
        seq_end = total_tokens

    slot = seq_id
    if HAS_CACHE_INDICES:
        slot = tl.load(CACHE_INDICES + seq_id).to(tl.int32)
    slot_valid = (slot != NULL_BLOCK_ID) & (slot >= 0)

    init_state = INIT_ALL_TRUE
    if HAS_INITIAL_STATE_PTR:
        init_state = tl.load(HAS_INITIAL_STATE + seq_id).to(tl.int1)

    for token in range(seq_start, seq_end):
        acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(BIAS + channel, mask=channel_mask, other=0.0).to(tl.float32)

        for tap in tl.static_range(KERNEL_SIZE):
            source_token = token - (KERNEL_SIZE - 1 - tap)
            if source_token >= seq_start:
                value = tl.load(
                    PROJ + source_token * (2 * intermediate_size) + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                value = tl.full((BLOCK_I,), 0.0, dtype=tl.float32)
                if HAS_CONV_STATE:
                    if slot_valid and init_state:
                        history_idx = source_token - seq_start + KERNEL_SIZE - 1
                        value = tl.load(
                            CONV_STATE
                            + slot * stride_state_slot
                            + channel * stride_state_i
                            + history_idx * stride_state_k,
                            mask=channel_mask,
                            other=0.0,
                        ).to(tl.float32)

            weight = tl.load(
                WEIGHT + channel * stride_w_i + tap * stride_w_k,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            acc += value * weight

        if APPLY_SILU:
            acc = acc / (1.0 + tl.exp(-acc))

        tl.store(
            CONV_OUT + token * intermediate_size + channel,
            acc.to(CONV_OUT.dtype.element_ty),
            mask=channel_mask,
        )

    if HAS_CONV_STATE:
        if slot_valid and (seq_end > seq_start):
            for j in tl.static_range(KERNEL_SIZE - 1):
                source_token = seq_end - (KERNEL_SIZE - 1) + j
                if source_token >= seq_start:
                    value = tl.load(
                        PROJ + source_token * (2 * intermediate_size) + channel,
                        mask=channel_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    value = tl.full((BLOCK_I,), 0.0, dtype=tl.float32)
                    if init_state:
                        history_idx = source_token - seq_start + KERNEL_SIZE - 1
                        value = tl.load(
                            CONV_STATE
                            + slot * stride_state_slot
                            + channel * stride_state_i
                            + history_idx * stride_state_k,
                            mask=channel_mask,
                            other=0.0,
                        ).to(tl.float32)
                tl.store(
                    CONV_STATE
                    + slot * stride_state_slot
                    + channel * stride_state_i
                    + j * stride_state_k,
                    value.to(CONV_STATE.dtype.element_ty),
                    mask=channel_mask,
                )


@triton.jit(
    do_not_specialize=[
        "total_tokens",
        "intermediate_size",
        "num_sequences",
        "stride_w_i",
        "stride_w_k",
        "stride_state_slot",
        "stride_state_i",
        "stride_state_k",
    ]
)
def _hygon_conv1d_prefill_kernel(
    PROJ,
    WEIGHT,
    BIAS,
    CONV_STATE,
    QUERY_START_LOC,
    CACHE_INDICES,
    HAS_INITIAL_STATE,
    CONV_OUT,
    total_tokens,
    intermediate_size,
    num_sequences,
    stride_w_i,
    stride_w_k,
    stride_state_slot,
    stride_state_i,
    stride_state_k,
    HAS_BIAS: tl.constexpr,
    HAS_CONV_STATE: tl.constexpr,
    HAS_QUERY_START_LOC: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE_PTR: tl.constexpr,
    INIT_ALL_TRUE: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_I: tl.constexpr,
    LOG2_SEQS: tl.constexpr,
    HAS_TLE: tl.constexpr,
):
    """2D-tiled parallel causal Conv1D for prefill (reads CONV_STATE read-only)."""
    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)
    t_local = tl.arange(0, BLOCK_T)
    i_local = tl.arange(0, BLOCK_I)
    tok = pid_t * BLOCK_T + t_local
    channel = pid_i * BLOCK_I + i_local
    tok_mask = tok < total_tokens
    channel_mask = channel < intermediate_size

    if HAS_QUERY_START_LOC:
        lo = tl.zeros((BLOCK_T,), dtype=tl.int32)
        hi = tl.full((BLOCK_T,), num_sequences, dtype=tl.int32)
        for _ in tl.static_range(LOG2_SEQS):
            mid = (lo + hi + 1) >> 1
            qval = tl.load(
                QUERY_START_LOC + mid,
                mask=mid <= num_sequences,
                other=total_tokens + 1,
            ).to(tl.int32)
            cond = qval <= tok
            lo = tl.where(cond, mid, lo)
            hi = tl.where(cond, hi, mid - 1)
        seq_id = lo
        seq_start = tl.load(QUERY_START_LOC + seq_id, mask=tok_mask, other=0).to(
            tl.int32
        )
    else:
        seq_id = tl.zeros((BLOCK_T,), dtype=tl.int32)
        seq_start = tl.zeros((BLOCK_T,), dtype=tl.int32)

    if HAS_CONV_STATE:
        if HAS_CACHE_INDICES:
            slot = tl.load(
                CACHE_INDICES + seq_id, mask=tok_mask, other=NULL_BLOCK_ID
            ).to(tl.int32)
        else:
            slot = seq_id
        slot_valid = (slot != NULL_BLOCK_ID) & (slot >= 0)
        if HAS_INITIAL_STATE_PTR:
            init_state = tl.load(
                HAS_INITIAL_STATE + seq_id, mask=tok_mask, other=False
            ).to(tl.int1)
        else:
            init_state = tl.full((BLOCK_T,), INIT_ALL_TRUE, dtype=tl.int1)
        use_hist = slot_valid & init_state
    else:
        slot = tl.zeros((BLOCK_T,), dtype=tl.int32)
        use_hist = tl.zeros((BLOCK_T,), dtype=tl.int1)

    acc = tl.zeros((BLOCK_T, BLOCK_I), dtype=tl.float32)
    if HAS_BIAS:
        if HAS_TLE:
            bias_smem = tle.gpu.alloc([BLOCK_I], dtype=tl.float32, scope=tle.gpu.smem)
            bp = tle.gpu.local_ptr(bias_smem, (i_local,))
            tl.store(
                bp, tl.load(BIAS + channel, mask=channel_mask, other=0.0).to(tl.float32)
            )
            acc += tl.load(bp)[None, :]
        else:
            bias = tl.load(BIAS + channel, mask=channel_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]

    if HAS_TLE:
        w_smem = tle.gpu.alloc([BLOCK_I], dtype=tl.float32, scope=tle.gpu.smem)
        wp = tle.gpu.local_ptr(w_smem, (i_local,))

    for tap in tl.static_range(KERNEL_SIZE):
        source_token = tok - (KERNEL_SIZE - 1 - tap)
        from_proj = tok_mask & (source_token >= seq_start)
        val = tl.load(
            PROJ + source_token[:, None] * (2 * intermediate_size) + channel[None, :],
            mask=from_proj[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        if HAS_CONV_STATE:
            from_hist = tok_mask & (source_token < seq_start) & use_hist
            history_idx = source_token - seq_start + (KERNEL_SIZE - 1)
            hist_val = tl.load(
                CONV_STATE
                + slot[:, None] * stride_state_slot
                + channel[None, :] * stride_state_i
                + history_idx[:, None] * stride_state_k,
                mask=from_hist[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            val = tl.where(from_hist[:, None], hist_val, val)

        w_g = tl.load(
            WEIGHT + channel * stride_w_i + tap * stride_w_k,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_TLE:
            tl.store(wp, w_g)
            w = tl.load(wp)
        else:
            w = w_g
        acc += val * w[None, :]

    if APPLY_SILU:
        acc = acc / (1.0 + tl.exp(-acc))

    tl.store(
        CONV_OUT + tok[:, None] * intermediate_size + channel[None, :],
        acc.to(CONV_OUT.dtype.element_ty),
        mask=tok_mask[:, None] & channel_mask[None, :],
    )


@triton.jit(
    do_not_specialize=[
        "total_tokens",
        "intermediate_size",
        "stride_state_slot",
        "stride_state_i",
        "stride_state_k",
    ]
)
def _hygon_conv1d_state_update_kernel(
    PROJ,
    CONV_STATE,
    QUERY_START_LOC,
    CACHE_INDICES,
    HAS_INITIAL_STATE,
    total_tokens,
    intermediate_size,
    stride_state_slot,
    stride_state_i,
    stride_state_k,
    HAS_QUERY_START_LOC: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE_PTR: tl.constexpr,
    INIT_ALL_TRUE: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Update Conv cache with the last K-1 input samples per sequence."""
    seq_id = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    channel_mask = channel < intermediate_size

    if HAS_QUERY_START_LOC:
        seq_start = tl.load(QUERY_START_LOC + seq_id).to(tl.int32)
        seq_end = tl.load(QUERY_START_LOC + seq_id + 1).to(tl.int32)
    else:
        seq_start = 0
        seq_end = total_tokens

    slot = seq_id
    if HAS_CACHE_INDICES:
        slot = tl.load(CACHE_INDICES + seq_id).to(tl.int32)
    slot_valid = (slot != NULL_BLOCK_ID) & (slot >= 0)

    init_state = INIT_ALL_TRUE
    if HAS_INITIAL_STATE_PTR:
        init_state = tl.load(HAS_INITIAL_STATE + seq_id).to(tl.int1)

    if slot_valid and (seq_end > seq_start):
        for j in tl.static_range(KERNEL_SIZE - 1):
            source_token = seq_end - (KERNEL_SIZE - 1) + j
            if source_token >= seq_start:
                value = tl.load(
                    PROJ + source_token * (2 * intermediate_size) + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                value = tl.full((BLOCK_I,), 0.0, dtype=tl.float32)
                if init_state:
                    history_idx = source_token - seq_start + KERNEL_SIZE - 1
                    value = tl.load(
                        CONV_STATE
                        + slot * stride_state_slot
                        + channel * stride_state_i
                        + history_idx * stride_state_k,
                        mask=channel_mask,
                        other=0.0,
                    ).to(tl.float32)
            tl.store(
                CONV_STATE
                + slot * stride_state_slot
                + channel * stride_state_i
                + j * stride_state_k,
                value.to(CONV_STATE.dtype.element_ty),
                mask=channel_mask,
            )


# ---------------------------------------------------------------------------
# Fused 3-way RMSNorm kernel
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "stride_row",
        "eps",
        "w_dt",
        "w_bc",
        "off_b",
        "off_c",
    ]
)
def _hygon_rms_norm_3way_kernel(
    X,
    DT_W,
    B_W,
    C_W,
    stride_row,
    stride_col,
    eps,
    w_dt,
    w_bc,
    off_b,
    off_c,
    HAS_DT_W: tl.constexpr,
    HAS_B_W: tl.constexpr,
    HAS_C_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused in-place RMSNorm for dt, B, and C segments across (tokens, 3)."""
    row = tl.program_id(0)
    seg = tl.program_id(1)
    col = tl.arange(0, BLOCK)

    if seg == 0:
        width = w_dt
        offset = 0
        has_w = HAS_DT_W
        w_ptr = DT_W
    elif seg == 1:
        width = w_bc
        offset = off_b
        has_w = HAS_B_W
        w_ptr = B_W
    else:
        width = w_bc
        offset = off_c
        has_w = HAS_C_W
        w_ptr = C_W

    mask = col < width
    ptr = X + row * stride_row + (offset + col) * stride_col
    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / width + eps)
    if has_w:
        gamma = tl.load(w_ptr + col, mask=mask, other=1.0).to(tl.float32)
    else:
        gamma = tl.full((BLOCK,), 1.0, dtype=tl.float32)
    tl.store(ptr, (x * inv_rms * gamma).to(X.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Selective Scan kernels (Small/Decode sequential + Large Chunked Parallel)
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "total_tokens",
        "intermediate_size",
        "num_sequences",
        "time_step_rank",
        "params_width",
        "stride_a_i",
        "stride_a_s",
        "stride_state_slot",
        "stride_state_i",
        "stride_state_s",
    ]
)
def _hygon_selective_scan_varlen_kernel(
    PROJ,
    CONV_OUT,
    SSM_PARAMS,
    DT,
    A,
    D,
    DT_BIAS,
    SSM_STATE,
    QUERY_START_LOC,
    CACHE_INDICES,
    HAS_INITIAL_STATE,
    SCAN_OUT,
    total_tokens,
    intermediate_size,
    num_sequences,
    time_step_rank,
    params_width,
    stride_a_i,
    stride_a_s,
    stride_state_slot,
    stride_state_i,
    stride_state_s,
    HAS_DT_BIAS: tl.constexpr,
    HAS_SSM_STATE: tl.constexpr,
    HAS_QUERY_START_LOC: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE_PTR: tl.constexpr,
    INIT_ALL_TRUE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Sequential recurrence per (sequence, channel tile) for decode & small prefill."""
    seq_id = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    state_idx = tl.arange(0, BLOCK_S)
    channel_mask = channel < intermediate_size
    state_mask = state_idx < STATE_SIZE

    if IS_DECODE:
        seq_start = seq_id
        seq_end = seq_id + 1
    elif HAS_QUERY_START_LOC:
        seq_start = tl.load(QUERY_START_LOC + seq_id).to(tl.int32)
        seq_end = tl.load(QUERY_START_LOC + seq_id + 1).to(tl.int32)
    else:
        seq_start = 0
        seq_end = total_tokens

    slot = seq_id
    if HAS_CACHE_INDICES:
        slot = tl.load(CACHE_INDICES + seq_id).to(tl.int32)
    slot_valid = (slot != NULL_BLOCK_ID) & (slot >= 0)

    init_state = INIT_ALL_TRUE
    if HAS_INITIAL_STATE_PTR:
        init_state = tl.load(HAS_INITIAL_STATE + seq_id).to(tl.int1)

    state = tl.zeros((BLOCK_I, BLOCK_S), dtype=tl.float32)
    if HAS_SSM_STATE:
        if slot_valid and init_state:
            state = tl.load(
                SSM_STATE
                + slot * stride_state_slot
                + channel[:, None] * stride_state_i
                + state_idx[None, :] * stride_state_s,
                mask=channel_mask[:, None] & state_mask[None, :],
                other=0.0,
            ).to(tl.float32)

    a = tl.load(
        A + channel[:, None] * stride_a_i + state_idx[None, :] * stride_a_s,
        mask=channel_mask[:, None] & state_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    d = tl.load(D + channel, mask=channel_mask, other=0.0).to(tl.float32)

    for token in range(seq_start, seq_end):
        u = tl.load(
            CONV_OUT + token * intermediate_size + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(
            DT + token * intermediate_size + channel, mask=channel_mask, other=0.0
        ).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(DT_BIAS + channel, mask=channel_mask, other=0.0).to(
                tl.float32
            )
        dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)

        b = tl.load(
            SSM_PARAMS + token * params_width + time_step_rank + state_idx,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        c = tl.load(
            SSM_PARAMS + token * params_width + time_step_rank + STATE_SIZE + state_idx,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)

        d_a = tl.exp(dt[:, None] * a)
        d_b = dt[:, None] * b[None, :]
        state = state * d_a + d_b * u[:, None]
        y = tl.sum(state * c[None, :], axis=1) + d * u

        gate = tl.load(
            PROJ + token * (2 * intermediate_size) + intermediate_size + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        y = y * (gate / (1.0 + tl.exp(-gate)))
        tl.store(
            SCAN_OUT + token * intermediate_size + channel,
            y.to(SCAN_OUT.dtype.element_ty),
            mask=channel_mask,
        )

    if HAS_SSM_STATE:
        if slot_valid and (seq_end > seq_start):
            tl.store(
                SSM_STATE
                + slot * stride_state_slot
                + channel[:, None] * stride_state_i
                + state_idx[None, :] * stride_state_s,
                state.to(SSM_STATE.dtype.element_ty),
                mask=channel_mask[:, None] & state_mask[None, :],
            )


@triton.jit
def _ssm_scan_combine(a_l, b_l, a_r, b_r):
    return a_r * a_l, a_r * b_l + b_r


@triton.jit(do_not_specialize=["num_sequences", "max_chunks"])
def _hygon_build_chunks_kernel(
    QUERY_START_LOC,
    CHUNK_SEQ_ID,
    CHUNK_START,
    CHUNK_END,
    SEQ_CHUNK_OFF,
    num_sequences,
    max_chunks,
    CHUNK_SIZE: tl.constexpr,
):
    """Build per-sequence chunk ranges on GPU without CPU-GPU sync."""
    cid = 0
    for s in range(num_sequences):
        tl.store(SEQ_CHUNK_OFF + s, cid)
        s_start = tl.load(QUERY_START_LOC + s).to(tl.int32)
        s_end = tl.load(QUERY_START_LOC + s + 1).to(tl.int32)
        t = s_start
        while t < s_end:
            t_next = tl.minimum(t + CHUNK_SIZE, s_end)
            tl.store(CHUNK_SEQ_ID + cid, s)
            tl.store(CHUNK_START + cid, t)
            tl.store(CHUNK_END + cid, t_next)
            cid += 1
            t = t_next
    tl.store(SEQ_CHUNK_OFF + num_sequences, cid)
    for k in range(cid, max_chunks):
        tl.store(CHUNK_SEQ_ID + k, -1)


@triton.jit(
    do_not_specialize=[
        "total_tokens",
        "intermediate_size",
        "time_step_rank",
        "params_width",
        "stride_a_i",
        "stride_a_s",
    ]
)
def _hygon_scan_chunk_pass1_kernel(
    CONV_OUT,
    SSM_PARAMS,
    DT,
    A,
    DT_BIAS,
    CHUNK_SEQ_ID,
    CHUNK_START,
    CHUNK_END,
    CHUNK_A,
    CHUNK_H,
    total_tokens,
    intermediate_size,
    time_step_rank,
    params_width,
    stride_a_i,
    stride_a_s,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CHUNK_META: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HAS_TLE: tl.constexpr,
):
    """Pass 1: coalesced loads + intra-chunk warp-shuffle associative scan."""
    cid = tl.program_id(0)
    if HAS_CHUNK_META:
        seq_id = tl.load(CHUNK_SEQ_ID + cid).to(tl.int32)
        if seq_id < 0:
            return
        tok_start = tl.load(CHUNK_START + cid).to(tl.int32)
        tok_end = tl.load(CHUNK_END + cid).to(tl.int32)
    else:
        tok_start = cid * CHUNK_SIZE
        tok_end = tl.minimum(tok_start + CHUNK_SIZE, total_tokens)

    i_local = tl.arange(0, BLOCK_I)
    channel = tl.program_id(1) * BLOCK_I + i_local
    state_idx = tl.arange(0, BLOCK_S)
    t_local = tl.arange(0, CHUNK_SIZE)
    tok = tok_start + t_local

    channel_mask = channel < intermediate_size
    state_mask = state_idx < STATE_SIZE
    t_mask = tok < tok_end

    a_g = tl.load(
        A + channel[:, None] * stride_a_i + state_idx[None, :] * stride_a_s,
        mask=channel_mask[:, None] & state_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_TLE:
        a_smem = tle.gpu.alloc([BLOCK_I, BLOCK_S], dtype=tl.float32, scope=tle.gpu.smem)
        ar = tl.broadcast_to(i_local[:, None], (BLOCK_I, BLOCK_S))
        ac = tl.broadcast_to(state_idx[None, :], (BLOCK_I, BLOCK_S))
        ap = tle.gpu.local_ptr(a_smem, (ar, ac))
        tl.store(ap, a_g)
        a = tl.load(ap)
    else:
        a = a_g

    u = tl.trans(
        tl.load(
            CONV_OUT + tok[:, None] * intermediate_size + channel[None, :],
            mask=t_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    )
    dt = tl.trans(
        tl.load(
            DT + tok[:, None] * intermediate_size + channel[None, :],
            mask=t_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    )
    b = tl.trans(
        tl.load(
            SSM_PARAMS
            + tok[:, None] * params_width
            + time_step_rank
            + state_idx[None, :],
            mask=t_mask[:, None] & state_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    )

    if HAS_DT_BIAS:
        dt += tl.load(DT_BIAS + channel, mask=channel_mask, other=0.0).to(tl.float32)[
            :, None
        ]
    dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)

    d_a = tl.exp(dt[:, None, :] * a[:, :, None])
    b_u = (dt * u)[:, None, :] * b[None, :, :]

    d_a = tl.where(t_mask[None, None, :], d_a, 1.0)
    b_u = tl.where(t_mask[None, None, :], b_u, 0.0)

    cum_a, local_h = tl.associative_scan(
        (d_a, b_u), axis=2, combine_fn=_ssm_scan_combine
    )

    is_last = (t_local == (CHUNK_SIZE - 1))[None, None, :]
    end_a = tl.sum(tl.where(is_last, cum_a, 0.0), axis=2)
    end_h = tl.sum(tl.where(is_last, local_h, 0.0), axis=2)

    cs_stride = intermediate_size * BLOCK_S
    out_offs = cid * cs_stride + channel[:, None] * BLOCK_S + state_idx[None, :]
    tl.store(
        CHUNK_A + out_offs, end_a, mask=channel_mask[:, None] & state_mask[None, :]
    )
    tl.store(
        CHUNK_H + out_offs, end_h, mask=channel_mask[:, None] & state_mask[None, :]
    )


@triton.jit(
    do_not_specialize=[
        "intermediate_size",
        "num_chunks_total",
        "stride_state_slot",
        "stride_state_i",
        "stride_state_s",
    ]
)
def _hygon_scan_inter_chunk_kernel(
    CHUNK_A,
    CHUNK_H,
    CHUNK_INIT_H,
    SSM_STATE,
    SEQ_CHUNK_OFF,
    CACHE_INDICES,
    HAS_INITIAL_STATE,
    intermediate_size,
    num_chunks_total,
    stride_state_slot,
    stride_state_i,
    stride_state_s,
    HAS_SSM_STATE: tl.constexpr,
    HAS_CHUNK_META: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE_PTR: tl.constexpr,
    INIT_ALL_TRUE: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Pass 2: propagate chunk-boundary states sequentially per sequence."""
    seq_id = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    state_idx = tl.arange(0, BLOCK_S)
    channel_mask = channel < intermediate_size
    state_mask = state_idx < STATE_SIZE
    mask_is = channel_mask[:, None] & state_mask[None, :]

    if HAS_CHUNK_META:
        c_start = tl.load(SEQ_CHUNK_OFF + seq_id).to(tl.int32)
        c_end = tl.load(SEQ_CHUNK_OFF + seq_id + 1).to(tl.int32)
    else:
        c_start = 0
        c_end = num_chunks_total

    slot = seq_id
    if HAS_CACHE_INDICES:
        slot = tl.load(CACHE_INDICES + seq_id).to(tl.int32)
    slot_valid = (slot != NULL_BLOCK_ID) & (slot >= 0)

    init_state = INIT_ALL_TRUE
    if HAS_INITIAL_STATE_PTR:
        init_state = tl.load(HAS_INITIAL_STATE + seq_id).to(tl.int1)

    state = tl.zeros((BLOCK_I, BLOCK_S), dtype=tl.float32)
    if HAS_SSM_STATE:
        if slot_valid and init_state:
            state = tl.load(
                SSM_STATE
                + slot * stride_state_slot
                + channel[:, None] * stride_state_i
                + state_idx[None, :] * stride_state_s,
                mask=mask_is,
                other=0.0,
            ).to(tl.float32)

    cs_stride = intermediate_size * BLOCK_S
    for cid in range(c_start, c_end):
        offs = cid * cs_stride + channel[:, None] * BLOCK_S + state_idx[None, :]
        tl.store(CHUNK_INIT_H + offs, state, mask=mask_is)
        ca = tl.load(CHUNK_A + offs, mask=mask_is, other=1.0)
        ch = tl.load(CHUNK_H + offs, mask=mask_is, other=0.0)
        state = state * ca + ch

    if HAS_SSM_STATE:
        if slot_valid and (c_end > c_start):
            tl.store(
                SSM_STATE
                + slot * stride_state_slot
                + channel[:, None] * stride_state_i
                + state_idx[None, :] * stride_state_s,
                state.to(SSM_STATE.dtype.element_ty),
                mask=mask_is,
            )


@triton.jit(
    do_not_specialize=[
        "total_tokens",
        "intermediate_size",
        "time_step_rank",
        "params_width",
        "stride_a_i",
        "stride_a_s",
    ]
)
def _hygon_scan_chunk_pass3_kernel(
    PROJ,
    CONV_OUT,
    SSM_PARAMS,
    DT,
    A,
    D,
    DT_BIAS,
    CHUNK_SEQ_ID,
    CHUNK_START,
    CHUNK_END,
    CHUNK_INIT_H,
    SCAN_OUT,
    total_tokens,
    intermediate_size,
    time_step_rank,
    params_width,
    stride_a_i,
    stride_a_s,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CHUNK_META: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HAS_TLE: tl.constexpr,
):
    """Pass 3: coalesced I/O + intra-chunk scan with initial state + C reduction + gate."""
    cid = tl.program_id(0)
    if HAS_CHUNK_META:
        seq_id = tl.load(CHUNK_SEQ_ID + cid).to(tl.int32)
        if seq_id < 0:
            return
        tok_start = tl.load(CHUNK_START + cid).to(tl.int32)
        tok_end = tl.load(CHUNK_END + cid).to(tl.int32)
    else:
        tok_start = cid * CHUNK_SIZE
        tok_end = tl.minimum(tok_start + CHUNK_SIZE, total_tokens)

    i_local = tl.arange(0, BLOCK_I)
    channel = tl.program_id(1) * BLOCK_I + i_local
    state_idx = tl.arange(0, BLOCK_S)
    t_local = tl.arange(0, CHUNK_SIZE)
    tok = tok_start + t_local

    channel_mask = channel < intermediate_size
    state_mask = state_idx < STATE_SIZE
    t_mask = tok < tok_end

    cs_stride = intermediate_size * BLOCK_S
    h_init = tl.load(
        CHUNK_INIT_H
        + cid * cs_stride
        + channel[:, None] * BLOCK_S
        + state_idx[None, :],
        mask=channel_mask[:, None] & state_mask[None, :],
        other=0.0,
    )

    a_g = tl.load(
        A + channel[:, None] * stride_a_i + state_idx[None, :] * stride_a_s,
        mask=channel_mask[:, None] & state_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_TLE:
        a_smem = tle.gpu.alloc([BLOCK_I, BLOCK_S], dtype=tl.float32, scope=tle.gpu.smem)
        ar = tl.broadcast_to(i_local[:, None], (BLOCK_I, BLOCK_S))
        ac = tl.broadcast_to(state_idx[None, :], (BLOCK_I, BLOCK_S))
        ap = tle.gpu.local_ptr(a_smem, (ar, ac))
        tl.store(ap, a_g)
        a = tl.load(ap)
    else:
        a = a_g
    d = tl.load(D + channel, mask=channel_mask, other=0.0).to(tl.float32)

    u = tl.trans(
        tl.load(
            CONV_OUT + tok[:, None] * intermediate_size + channel[None, :],
            mask=t_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    )
    dt = tl.trans(
        tl.load(
            DT + tok[:, None] * intermediate_size + channel[None, :],
            mask=t_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    )
    b = tl.trans(
        tl.load(
            SSM_PARAMS
            + tok[:, None] * params_width
            + time_step_rank
            + state_idx[None, :],
            mask=t_mask[:, None] & state_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    )
    c = tl.trans(
        tl.load(
            SSM_PARAMS
            + tok[:, None] * params_width
            + time_step_rank
            + STATE_SIZE
            + state_idx[None, :],
            mask=t_mask[:, None] & state_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    )
    gate = tl.trans(
        tl.load(
            PROJ
            + tok[:, None] * (2 * intermediate_size)
            + intermediate_size
            + channel[None, :],
            mask=t_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    )

    if HAS_DT_BIAS:
        dt += tl.load(DT_BIAS + channel, mask=channel_mask, other=0.0).to(tl.float32)[
            :, None
        ]
    dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)

    d_a = tl.exp(dt[:, None, :] * a[:, :, None])
    b_u = (dt * u)[:, None, :] * b[None, :, :]

    is_first = (t_local == 0)[None, None, :]
    b_u = tl.where(is_first, d_a * h_init[:, :, None] + b_u, b_u)
    d_a = tl.where(t_mask[None, None, :], d_a, 1.0)
    b_u = tl.where(t_mask[None, None, :], b_u, 0.0)

    _, h = tl.associative_scan((d_a, b_u), axis=2, combine_fn=_ssm_scan_combine)

    y = tl.sum(h * c[None, :, :], axis=1) + d[:, None] * u
    y = y * (gate / (1.0 + tl.exp(-gate)))

    tl.store(
        SCAN_OUT + tok[:, None] * intermediate_size + channel[None, :],
        tl.trans(y).to(SCAN_OUT.dtype.element_ty),
        mask=t_mask[:, None] & channel_mask[None, :],
    )


def _launch_one(
    key: tuple,
    kernel,
    grid: tuple[int, int, int],
    args: tuple,
    kwargs: dict,
    cvals: tuple,
    num_warps: int,
    num_stages: int = 2,
) -> None:
    comp = _COMPILED.get(key)
    if comp is None:
        comp = kernel[grid](*args, **kwargs, num_warps=num_warps, num_stages=num_stages)
        _COMPILED[key] = comp
    else:
        comp[grid](*args, *cvals)


def mamba_mixer(
    hidden_states: torch.Tensor,
    in_proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    x_proj_weight: torch.Tensor,
    dt_proj_weight: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    out_proj_weight: torch.Tensor,
    *,
    in_proj_bias: torch.Tensor | None = None,
    conv1d_bias: torch.Tensor | None = None,
    dt_proj_bias: torch.Tensor | None = None,
    out_proj_bias: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    ssm_state: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    dt_norm_weight: torch.Tensor | None = None,
    b_norm_weight: torch.Tensor | None = None,
    c_norm_weight: torch.Tensor | None = None,
    use_rms_norm: bool = False,
    rms_norm_eps: float = 1e-5,
    activation: str | None = "silu",
    is_decode: bool = False,
    output: torch.Tensor | None = None,
    num_actual_tokens: int | None = None,
    null_block_id: int = 0,
) -> torch.Tensor:
    """Run the Mamba-1 mixer on Hygon DCU with optimized Triton kernels."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [total_tokens, hidden_size]")
    dtype = hidden_states.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "mamba_mixer supports FP16, BF16 and FP32 activations only"
        )
    if activation not in (None, "silu", "swish", True):
        raise NotImplementedError(f"unsupported activation: {activation}")
    if is_decode and query_start_loc is not None:
        raise ValueError("query_start_loc must be omitted when is_decode=True")

    padded_tokens, hidden_size = hidden_states.shape
    if num_actual_tokens is None:
        total_tokens = padded_tokens
    else:
        if isinstance(num_actual_tokens, bool) or not isinstance(
            num_actual_tokens, int
        ):
            raise TypeError("num_actual_tokens must be an int or None")
        if not 0 <= num_actual_tokens <= padded_tokens:
            raise ValueError(
                f"num_actual_tokens must be in [0, {padded_tokens}], "
                f"got {num_actual_tokens}"
            )
        total_tokens = num_actual_tokens

    dev = hidden_states.device
    if output is not None:
        if output.ndim != 2 or output.shape[1] != hidden_size:
            raise ValueError(
                "output must have shape [>=num_actual_tokens, hidden_size]"
            )
        if output.shape[0] < total_tokens:
            raise ValueError("output has fewer rows than num_actual_tokens")
        if output.dtype != dtype:
            raise ValueError("output dtype must match hidden_states dtype")
        if output.device != dev:
            raise ValueError("output and hidden_states must be on the same device")

    if total_tokens == 0:
        if output is not None:
            return output
        return torch.empty((0, hidden_size), device=dev, dtype=dtype)

    if total_tokens < padded_tokens:
        hidden_states = hidden_states[:total_tokens]

    if (
        in_proj_weight.ndim != 2
        or in_proj_weight.shape[1] != hidden_size
        or in_proj_weight.shape[0] % 2
    ):
        raise ValueError(
            "in_proj_weight must have shape [2 * intermediate_size, hidden_size]"
        )
    intermediate_size = in_proj_weight.shape[0] // 2

    if conv1d_weight.ndim == 3:
        if conv1d_weight.shape[1] != 1:
            raise ValueError("3D conv1d_weight must have shape [I, 1, K]")
        kernel_size = conv1d_weight.shape[2]
        conv_w_stride_i = conv1d_weight.stride(0)
        conv_w_stride_k = conv1d_weight.stride(2)
    elif conv1d_weight.ndim == 2:
        kernel_size = conv1d_weight.shape[1]
        conv_w_stride_i = conv1d_weight.stride(0)
        conv_w_stride_k = conv1d_weight.stride(1)
    else:
        raise ValueError("conv1d_weight must be [I, K] or [I, 1, K]")
    if conv1d_weight.shape[0] != intermediate_size:
        raise ValueError("conv1d_weight dimensions do not match intermediate_size")
    if not 2 <= kernel_size <= 7:
        raise NotImplementedError(
            "vLLM-compatible causal Conv1D currently requires kernel_size in [2, 7]"
        )

    if x_proj_weight.ndim != 2 or x_proj_weight.shape[1] != intermediate_size:
        raise ValueError("x_proj_weight must have shape [R + 2*S, intermediate_size]")
    if dt_proj_weight.ndim != 2 or dt_proj_weight.shape[0] != intermediate_size:
        raise ValueError("dt_proj_weight must have shape [intermediate_size, R]")
    time_step_rank = dt_proj_weight.shape[1]
    if (x_proj_weight.shape[0] - time_step_rank) % 2:
        raise ValueError("x_proj_weight output dimension must equal R + 2*S")
    state_size = (x_proj_weight.shape[0] - time_step_rank) // 2
    if state_size < 1 or state_size > 64:
        raise NotImplementedError(
            "state_size must be in [1, 64] for this first Triton path"
        )
    if A.shape != (intermediate_size, state_size) or D.shape != (intermediate_size,):
        raise ValueError("A or D shape does not match intermediate_size/state_size")
    if out_proj_weight.shape != (hidden_size, intermediate_size):
        raise ValueError(
            "out_proj_weight must have shape [hidden_size, intermediate_size]"
        )

    if in_proj_bias is not None and in_proj_bias.shape != (2 * intermediate_size,):
        raise ValueError(f"in_proj_bias must have shape {(2 * intermediate_size,)}")
    if conv1d_bias is not None and conv1d_bias.shape != (intermediate_size,):
        raise ValueError(f"conv1d_bias must have shape {(intermediate_size,)}")
    if dt_proj_bias is not None:
        if dt_proj_bias.shape != (intermediate_size,):
            raise ValueError(f"dt_proj_bias must have shape {(intermediate_size,)}")
        if dt_proj_bias.dtype != torch.float32:
            raise ValueError("dt_proj_bias must be FP32, matching vLLM scan bias path")
    if out_proj_bias is not None and out_proj_bias.shape != (hidden_size,):
        raise ValueError(f"out_proj_bias must have shape {(hidden_size,)}")
    if A.dtype != torch.float32 or D.dtype != torch.float32:
        raise ValueError("A and D must be FP32")

    if is_decode:
        num_sequences = total_tokens
    elif query_start_loc is None:
        num_sequences = 1
    else:
        if query_start_loc.ndim != 1 or query_start_loc.dtype != torch.int32:
            raise ValueError("query_start_loc must be a 1D int32 tensor")
        num_sequences = query_start_loc.shape[0] - 1
        if num_sequences < 1:
            raise ValueError("query_start_loc must contain at least two entries")

    has_cache = conv_state is not None or ssm_state is not None
    if (conv_state is None) != (ssm_state is None):
        raise ValueError(
            "conv_state and ssm_state must either both be provided or both be omitted"
        )
    if has_cache:
        if conv_state.ndim != 3 or conv_state.shape[1:] != (
            intermediate_size,
            kernel_size - 1,
        ):
            raise ValueError("conv_state must have shape [cache_slots, I, K-1]")
        if ssm_state.ndim != 3 or ssm_state.shape[1:] != (
            intermediate_size,
            state_size,
        ):
            raise ValueError("ssm_state must have shape [cache_slots, I, S]")
        if conv_state.shape[0] != ssm_state.shape[0]:
            raise ValueError(
                "conv_state and ssm_state must have the same number of cache slots"
            )
        allowed_state_dtypes = (torch.float16, torch.bfloat16, torch.float32)
        if (
            conv_state.dtype not in allowed_state_dtypes
            or ssm_state.dtype not in allowed_state_dtypes
        ):
            raise NotImplementedError(
                "conv_state and ssm_state must be FP16, BF16, or FP32"
            )
        if conv_state.dtype not in (dtype, torch.float32):
            raise NotImplementedError(
                "conv_state dtype must match hidden_states or be FP32"
            )
        if cache_indices is None and null_block_id == 0:
            raise ValueError(
                "cache_indices must be provided when using vLLM NULL_BLOCK_ID=0"
            )
        if conv_state.device != dev or ssm_state.device != dev:
            raise ValueError(
                "conv_state/ssm_state and hidden_states must be on the same device"
            )
    elif cache_indices is not None or has_initial_state is not None:
        raise ValueError(
            "cache_indices/has_initial_state require conv_state and ssm_state"
        )

    if cache_indices is not None:
        if (
            cache_indices.shape != (num_sequences,)
            or cache_indices.dtype != torch.int32
        ):
            raise ValueError("cache_indices must be int32 with shape [num_sequences]")
        if cache_indices.device != dev or not cache_indices.is_contiguous():
            raise ValueError("cache_indices must be contiguous on the same device")
    if has_initial_state is not None:
        if (
            has_initial_state.shape != (num_sequences,)
            or has_initial_state.dtype != torch.bool
        ):
            raise ValueError(
                "has_initial_state must be bool with shape [num_sequences]"
            )
        if has_initial_state.device != dev or not has_initial_state.is_contiguous():
            raise ValueError("has_initial_state must be contiguous on the same device")
    if query_start_loc is not None:
        if query_start_loc.device != dev or not query_start_loc.is_contiguous():
            raise ValueError("query_start_loc must be contiguous on the same device")

    if not (
        hidden_states.is_contiguous()
        and in_proj_weight.is_contiguous()
        and conv1d_weight.is_contiguous()
        and x_proj_weight.is_contiguous()
        and dt_proj_weight.is_contiguous()
        and A.is_contiguous()
        and D.is_contiguous()
        and out_proj_weight.is_contiguous()
    ):
        raise ValueError(
            "input/weight tensors must be contiguous in this implementation"
        )

    if (
        in_proj_weight.device != dev
        or conv1d_weight.device != dev
        or x_proj_weight.device != dev
        or dt_proj_weight.device != dev
        or A.device != dev
        or D.device != dev
        or out_proj_weight.device != dev
    ):
        raise ValueError("weights and hidden_states must be on the same device")

    if (
        in_proj_weight.dtype != dtype
        or conv1d_weight.dtype != dtype
        or x_proj_weight.dtype != dtype
        or dt_proj_weight.dtype != dtype
        or out_proj_weight.dtype != dtype
    ):
        raise ValueError("weight dtypes must match hidden_states dtype")

    for name, bias_t in (
        ("in_proj_bias", in_proj_bias),
        ("conv1d_bias", conv1d_bias),
        ("dt_proj_bias", dt_proj_bias),
        ("out_proj_bias", out_proj_bias),
    ):
        if bias_t is not None:
            if bias_t.device != dev or not bias_t.is_contiguous():
                raise ValueError(f"{name} must be contiguous on the same device")

    if use_rms_norm:
        if (
            (
                dt_norm_weight is not None
                and (
                    dt_norm_weight.numel() != time_step_rank
                    or not dt_norm_weight.is_contiguous()
                    or dt_norm_weight.device != dev
                )
            )
            or (
                b_norm_weight is not None
                and (
                    b_norm_weight.numel() != state_size
                    or not b_norm_weight.is_contiguous()
                    or b_norm_weight.device != dev
                )
            )
            or (
                c_norm_weight is not None
                and (
                    c_norm_weight.numel() != state_size
                    or not c_norm_weight.is_contiguous()
                    or c_norm_weight.device != dev
                )
            )
        ):
            raise ValueError(
                "RMSNorm weights must be contiguous on the same device with valid width"
            )
    elif (
        dt_norm_weight is not None
        or b_norm_weight is not None
        or c_norm_weight is not None
    ):
        raise ValueError("norm weights were supplied but use_rms_norm=False")

    dev_idx = dev.index if dev.index is not None else torch_device_fn.current_device()
    x_align = (hidden_states.data_ptr() & 15) == 0

    projected = torch.empty(
        (total_tokens, 2 * intermediate_size), device=dev, dtype=dtype
    )
    conv_out = torch.empty((total_tokens, intermediate_size), device=dev, dtype=dtype)
    params_width = time_step_rank + 2 * state_size
    ssm_params = torch.empty((total_tokens, params_width), device=dev, dtype=dtype)
    dt = torch.empty((total_tokens, intermediate_size), device=dev, dtype=dtype)
    scan_out = torch.empty((total_tokens, intermediate_size), device=dev, dtype=dtype)
    if output is None:
        output = torch.empty((total_tokens, hidden_size), device=dev, dtype=dtype)

    has_qsl = query_start_loc is not None
    has_cache_indices = cache_indices is not None
    has_init_ptr = has_initial_state is not None
    init_all_true = has_cache and is_decode and has_initial_state is None
    apply_silu = activation in ("silu", "swish", True)
    block_s = triton.next_power_of_2(state_size)
    cs_dtype = conv_state.dtype if has_cache else None
    ss_dtype = ssm_state.dtype if has_cache else None
    cs_align = ((conv_state.data_ptr() & 15) == 0) if has_cache else False
    ss_align = ((ssm_state.data_ptr() & 15) == 0) if has_cache else False
    has_conv_bias = conv1d_bias is not None
    has_dt_bias = dt_proj_bias is not None

    cs_s0 = conv_state.stride(0) if has_cache else 0
    cs_s1 = conv_state.stride(1) if has_cache else 0
    cs_s2 = conv_state.stride(2) if has_cache else 0
    ss_s0 = ssm_state.stride(0) if has_cache else 0
    ss_s1 = ssm_state.stride(1) if has_cache else 0
    ss_s2 = ssm_state.stride(2) if has_cache else 0

    # 1. in_proj
    _linear_hygon(
        hidden_states, in_proj_weight, in_proj_bias, projected, dev_idx, dtype, x_align
    )

    # 2. Causal Conv1D
    if is_decode or total_tokens <= 64:
        cvals = (
            has_conv_bias,
            has_cache,
            has_qsl,
            has_cache_indices,
            has_init_ptr,
            init_all_true,
            is_decode,
            apply_silu,
            null_block_id,
            kernel_size,
            _CONV_BLOCK_I_DECODE,
        )
        key = ("conv1d_varlen", dev_idx, dtype, cs_dtype, cs_align, cvals)
        grid = (
            num_sequences,
            (intermediate_size + _CONV_BLOCK_I_DECODE - 1) // _CONV_BLOCK_I_DECODE,
            1,
        )
        args = (
            projected,
            conv1d_weight,
            conv1d_bias,
            conv_state,
            query_start_loc,
            cache_indices,
            has_initial_state,
            conv_out,
            total_tokens,
            intermediate_size,
            num_sequences,
            conv_w_stride_i,
            conv_w_stride_k,
            cs_s0,
            cs_s1,
            cs_s2,
        )
        kwargs = {
            "HAS_BIAS": has_conv_bias,
            "HAS_CONV_STATE": has_cache,
            "HAS_QUERY_START_LOC": has_qsl,
            "HAS_CACHE_INDICES": has_cache_indices,
            "HAS_INITIAL_STATE_PTR": has_init_ptr,
            "INIT_ALL_TRUE": init_all_true,
            "IS_DECODE": is_decode,
            "APPLY_SILU": apply_silu,
            "NULL_BLOCK_ID": null_block_id,
            "KERNEL_SIZE": kernel_size,
            "BLOCK_I": _CONV_BLOCK_I_DECODE,
        }
        _launch_one(key, _hygon_conv1d_varlen_kernel, grid, args, kwargs, cvals, 4)
    else:
        log2_seqs = max(1, num_sequences.bit_length())
        cvals = (
            has_conv_bias,
            has_cache,
            has_qsl,
            has_cache_indices,
            has_init_ptr,
            init_all_true,
            apply_silu,
            null_block_id,
            kernel_size,
            _CONV_BLOCK_T_PREFILL,
            _CONV_BLOCK_I_PREFILL,
            log2_seqs,
            HAS_TLE,
        )
        key = ("conv1d_prefill", dev_idx, dtype, cs_dtype, cs_align, cvals)
        grid = (
            (total_tokens + _CONV_BLOCK_T_PREFILL - 1) // _CONV_BLOCK_T_PREFILL,
            (intermediate_size + _CONV_BLOCK_I_PREFILL - 1) // _CONV_BLOCK_I_PREFILL,
            1,
        )
        args = (
            projected,
            conv1d_weight,
            conv1d_bias,
            conv_state,
            query_start_loc,
            cache_indices,
            has_initial_state,
            conv_out,
            total_tokens,
            intermediate_size,
            num_sequences,
            conv_w_stride_i,
            conv_w_stride_k,
            cs_s0,
            cs_s1,
            cs_s2,
        )
        kwargs = {
            "HAS_BIAS": has_conv_bias,
            "HAS_CONV_STATE": has_cache,
            "HAS_QUERY_START_LOC": has_qsl,
            "HAS_CACHE_INDICES": has_cache_indices,
            "HAS_INITIAL_STATE_PTR": has_init_ptr,
            "INIT_ALL_TRUE": init_all_true,
            "APPLY_SILU": apply_silu,
            "NULL_BLOCK_ID": null_block_id,
            "KERNEL_SIZE": kernel_size,
            "BLOCK_T": _CONV_BLOCK_T_PREFILL,
            "BLOCK_I": _CONV_BLOCK_I_PREFILL,
            "LOG2_SEQS": log2_seqs,
            "HAS_TLE": HAS_TLE,
        }
        _launch_one(key, _hygon_conv1d_prefill_kernel, grid, args, kwargs, cvals, 4)
        if has_cache:
            cvals_u = (
                has_qsl,
                has_cache_indices,
                has_init_ptr,
                init_all_true,
                null_block_id,
                kernel_size,
                _CONV_BLOCK_I_PREFILL,
            )
            key_u = ("conv1d_state_upd", dev_idx, dtype, cs_dtype, cs_align, cvals_u)
            grid_u = (
                num_sequences,
                (intermediate_size + _CONV_BLOCK_I_PREFILL - 1)
                // _CONV_BLOCK_I_PREFILL,
                1,
            )
            args_u = (
                projected,
                conv_state,
                query_start_loc,
                cache_indices,
                has_initial_state,
                total_tokens,
                intermediate_size,
                cs_s0,
                cs_s1,
                cs_s2,
            )
            kwargs_u = {
                "HAS_QUERY_START_LOC": has_qsl,
                "HAS_CACHE_INDICES": has_cache_indices,
                "HAS_INITIAL_STATE_PTR": has_init_ptr,
                "INIT_ALL_TRUE": init_all_true,
                "NULL_BLOCK_ID": null_block_id,
                "KERNEL_SIZE": kernel_size,
                "BLOCK_I": _CONV_BLOCK_I_PREFILL,
            }
            _launch_one(
                key_u,
                _hygon_conv1d_state_update_kernel,
                grid_u,
                args_u,
                kwargs_u,
                cvals_u,
                4,
            )

    # 3. x_proj
    _linear_hygon(conv_out, x_proj_weight, None, ssm_params, dev_idx, dtype, True)

    # 4. Optional fused 3-way RMSNorm
    if use_rms_norm:
        norm_block = triton.next_power_of_2(max(time_step_rank, state_size))
        cvals_n = (
            dt_norm_weight is not None,
            b_norm_weight is not None,
            c_norm_weight is not None,
            norm_block,
        )
        key_n = ("rms_norm_3way", dev_idx, dtype, cvals_n)
        args_n = (
            ssm_params,
            dt_norm_weight,
            b_norm_weight,
            c_norm_weight,
            ssm_params.stride(0),
            ssm_params.stride(1),
            rms_norm_eps,
            time_step_rank,
            state_size,
            time_step_rank,
            time_step_rank + state_size,
        )
        kwargs_n = {
            "HAS_DT_W": cvals_n[0],
            "HAS_B_W": cvals_n[1],
            "HAS_C_W": cvals_n[2],
            "BLOCK": norm_block,
        }
        _launch_one(
            key_n,
            _hygon_rms_norm_3way_kernel,
            (total_tokens, 3, 1),
            args_n,
            kwargs_n,
            cvals_n,
            4,
        )

    # 5. dt_proj
    dt_lowrank = ssm_params[:, :time_step_rank]
    _linear_hygon(dt_lowrank, dt_proj_weight, None, dt, dev_idx, dtype, True)

    # 6. Selective Scan
    if is_decode or total_tokens <= 64:
        cvals_s = (
            has_dt_bias,
            has_cache,
            has_qsl,
            has_cache_indices,
            has_init_ptr,
            init_all_true,
            is_decode,
            null_block_id,
            state_size,
            block_s,
            _SCAN_BLOCK_I_SMALL,
        )
        key_s = ("scan_varlen", dev_idx, dtype, ss_dtype, ss_align, cvals_s)
        grid_s = (
            num_sequences,
            (intermediate_size + _SCAN_BLOCK_I_SMALL - 1) // _SCAN_BLOCK_I_SMALL,
            1,
        )
        args_s = (
            projected,
            conv_out,
            ssm_params,
            dt,
            A,
            D,
            dt_proj_bias,
            ssm_state,
            query_start_loc,
            cache_indices,
            has_initial_state,
            scan_out,
            total_tokens,
            intermediate_size,
            num_sequences,
            time_step_rank,
            params_width,
            A.stride(0),
            A.stride(1),
            ss_s0,
            ss_s1,
            ss_s2,
        )
        kwargs_s = {
            "HAS_DT_BIAS": has_dt_bias,
            "HAS_SSM_STATE": has_cache,
            "HAS_QUERY_START_LOC": has_qsl,
            "HAS_CACHE_INDICES": has_cache_indices,
            "HAS_INITIAL_STATE_PTR": has_init_ptr,
            "INIT_ALL_TRUE": init_all_true,
            "IS_DECODE": is_decode,
            "NULL_BLOCK_ID": null_block_id,
            "STATE_SIZE": state_size,
            "BLOCK_S": block_s,
            "BLOCK_I": _SCAN_BLOCK_I_SMALL,
        }
        _launch_one(
            key_s,
            _hygon_selective_scan_varlen_kernel,
            grid_s,
            args_s,
            kwargs_s,
            cvals_s,
            4,
        )
    else:
        has_chunk_meta = has_qsl and (num_sequences > 1)
        if has_chunk_meta:
            max_chunks = (
                total_tokens + _SCAN_CHUNK_SIZE - 1
            ) // _SCAN_CHUNK_SIZE + num_sequences
            chunk_seq_id = torch.empty((max_chunks,), device=dev, dtype=torch.int32)
            chunk_start = torch.empty((max_chunks,), device=dev, dtype=torch.int32)
            chunk_end = torch.empty((max_chunks,), device=dev, dtype=torch.int32)
            seq_chunk_off = torch.empty(
                (num_sequences + 1,), device=dev, dtype=torch.int32
            )
            _launch_one(
                ("build_chunks", dev_idx, _SCAN_CHUNK_SIZE),
                _hygon_build_chunks_kernel,
                (1, 1, 1),
                (
                    query_start_loc,
                    chunk_seq_id,
                    chunk_start,
                    chunk_end,
                    seq_chunk_off,
                    num_sequences,
                    max_chunks,
                ),
                {"CHUNK_SIZE": _SCAN_CHUNK_SIZE},
                (_SCAN_CHUNK_SIZE,),
                1,
            )
        else:
            max_chunks = (total_tokens + _SCAN_CHUNK_SIZE - 1) // _SCAN_CHUNK_SIZE
            chunk_seq_id = None
            chunk_start = None
            chunk_end = None
            seq_chunk_off = None

        chunk_a = torch.empty(
            (max_chunks, intermediate_size, block_s), device=dev, dtype=torch.float32
        )
        chunk_h = torch.empty(
            (max_chunks, intermediate_size, block_s), device=dev, dtype=torch.float32
        )
        chunk_init_h = torch.empty(
            (max_chunks, intermediate_size, block_s), device=dev, dtype=torch.float32
        )

        cvals_p13 = (
            has_dt_bias,
            has_chunk_meta,
            state_size,
            block_s,
            _SCAN_CHUNK_BLOCK_I,
            _SCAN_CHUNK_SIZE,
            HAS_TLE,
        )
        grid_p13 = (
            max_chunks,
            (intermediate_size + _SCAN_CHUNK_BLOCK_I - 1) // _SCAN_CHUNK_BLOCK_I,
            1,
        )
        _launch_one(
            ("scan_pass1", dev_idx, dtype, cvals_p13),
            _hygon_scan_chunk_pass1_kernel,
            grid_p13,
            (
                conv_out,
                ssm_params,
                dt,
                A,
                dt_proj_bias,
                chunk_seq_id,
                chunk_start,
                chunk_end,
                chunk_a,
                chunk_h,
                total_tokens,
                intermediate_size,
                time_step_rank,
                params_width,
                A.stride(0),
                A.stride(1),
            ),
            {
                "HAS_DT_BIAS": has_dt_bias,
                "HAS_CHUNK_META": has_chunk_meta,
                "STATE_SIZE": state_size,
                "BLOCK_S": block_s,
                "BLOCK_I": _SCAN_CHUNK_BLOCK_I,
                "CHUNK_SIZE": _SCAN_CHUNK_SIZE,
                "HAS_TLE": HAS_TLE,
            },
            cvals_p13,
            2,
        )

        cvals_p2 = (
            has_cache,
            has_chunk_meta,
            has_cache_indices,
            has_init_ptr,
            init_all_true,
            null_block_id,
            state_size,
            block_s,
            _SCAN_INTER_BLOCK_I,
        )
        grid_p2 = (
            num_sequences,
            (intermediate_size + _SCAN_INTER_BLOCK_I - 1) // _SCAN_INTER_BLOCK_I,
            1,
        )
        _launch_one(
            ("scan_pass2", dev_idx, ss_dtype, ss_align, cvals_p2),
            _hygon_scan_inter_chunk_kernel,
            grid_p2,
            (
                chunk_a,
                chunk_h,
                chunk_init_h,
                ssm_state,
                seq_chunk_off,
                cache_indices,
                has_initial_state,
                intermediate_size,
                max_chunks,
                ss_s0,
                ss_s1,
                ss_s2,
            ),
            {
                "HAS_SSM_STATE": has_cache,
                "HAS_CHUNK_META": has_chunk_meta,
                "HAS_CACHE_INDICES": has_cache_indices,
                "HAS_INITIAL_STATE_PTR": has_init_ptr,
                "INIT_ALL_TRUE": init_all_true,
                "NULL_BLOCK_ID": null_block_id,
                "STATE_SIZE": state_size,
                "BLOCK_S": block_s,
                "BLOCK_I": _SCAN_INTER_BLOCK_I,
            },
            cvals_p2,
            4,
        )

        _launch_one(
            ("scan_pass3", dev_idx, dtype, cvals_p13),
            _hygon_scan_chunk_pass3_kernel,
            grid_p13,
            (
                projected,
                conv_out,
                ssm_params,
                dt,
                A,
                D,
                dt_proj_bias,
                chunk_seq_id,
                chunk_start,
                chunk_end,
                chunk_init_h,
                scan_out,
                total_tokens,
                intermediate_size,
                time_step_rank,
                params_width,
                A.stride(0),
                A.stride(1),
            ),
            {
                "HAS_DT_BIAS": has_dt_bias,
                "HAS_CHUNK_META": has_chunk_meta,
                "STATE_SIZE": state_size,
                "BLOCK_S": block_s,
                "BLOCK_I": _SCAN_CHUNK_BLOCK_I,
                "CHUNK_SIZE": _SCAN_CHUNK_SIZE,
                "HAS_TLE": HAS_TLE,
            },
            cvals_p13,
            2,
        )

    # 7. out_proj
    _linear_hygon(
        scan_out, out_proj_weight, out_proj_bias, output, dev_idx, dtype, True
    )

    return output


__all__ = ["mamba_mixer"]

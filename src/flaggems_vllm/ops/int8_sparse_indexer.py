# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Symmetric INT8 sparse-indexer primitives for NVIDIA devices without FP8.

Q scales are folded into the head weights by the framework, as on the FP8 path.
Each cache page contains all INT8 key bytes followed by FP32 per-token scales.
Inputs must be finite. This forward-only API does not change checkpoint weights.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def _quantize(values, EPS: tl.constexpr, UE8M0: tl.constexpr):
    values = values.to(tl.float32)
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(values), -1), EPS), 127.0)
    if UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    x = tl.minimum(
        tl.maximum(tl.div_rn(values, tl.expand_dims(scale, -1)), -127.0), 127.0
    )
    lower = tl.floor(x)
    fraction = x - lower
    # Round to nearest, ties to even; no vendor-specific libdevice dependency.
    up = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    return (lower + up.to(tl.float32)).to(tl.int8), scale


@triton.jit
def _quantize_query(X, Q, S, stride_x, GROUPS: tl.constexpr, UE8M0: tl.constexpr):
    group = tl.program_id(0)
    row, column = group // GROUPS, (group % GROUPS) * 128
    d = tl.arange(0, 128)
    values = tl.load(X + row * stride_x + column + d)
    quantized, scale = _quantize(values, 1e-10, UE8M0)
    tl.store(Q + group * 128 + d, quantized)
    tl.store(S + group, scale)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("per_token_group_quant_int8_batched"),
    key=["GROUP_COUNT", "GROUPS"],
)
@triton.jit
def _quantize_query_batched(
    X,
    Q,
    S,
    stride_x,
    GROUP_COUNT,
    GROUPS: tl.constexpr,
    UE8M0: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row, column = group // GROUPS, (group % GROUPS) * 128
    d = tl.arange(0, 128)
    values = tl.load(
        X + row[:, None] * stride_x + column[:, None] + d[None, :],
        group[:, None] < GROUP_COUNT,
        other=0,
    )
    quantized, scale = _quantize(values, 1e-10, UE8M0)
    tl.store(
        Q + group[:, None] * 128 + d[None, :], quantized, group[:, None] < GROUP_COUNT
    )
    tl.store(S + group, scale, group < GROUP_COUNT)


@triton.jit
def _quantize_cache(
    K,
    Cache,
    Scales,
    Slots,
    stride_k,
    stride_cache,
    PAGE: tl.constexpr,
    SLOTS: tl.constexpr,
    UE8M0: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Slots + row).to(tl.int64)
    if slot < 0 or slot >= SLOTS:
        return
    page, offset = slot // PAGE, slot % PAGE
    d = tl.arange(0, 128)
    quantized, scale = _quantize(tl.load(K + row * stride_k + d), 1e-4, UE8M0)
    tl.store(
        Cache + page * stride_cache + offset * 128 + d,
        quantized.to(tl.uint8, bitcast=True),
    )
    tl.store(Scales + page * (stride_cache // 4) + PAGE * 32 + offset, scale)


def _cuda(*tensors):
    if (
        runtime.device.vendor_name != "nvidia"
        or not tensors
        or tensors[0].device.type != "cuda"
    ):
        raise NotImplementedError("INT8 indexer requires CUDA tensors")
    if any(t.device != tensors[0].device for t in tensors):
        raise ValueError("INT8 indexer tensors must share a device")


def per_token_group_quant_int8(x, group_size, *, use_ue8m0=True):
    """Quantize 2D FP16/BF16/FP32 input in groups of 128, with FP32 scales."""
    _cuda(x)
    if (
        x.ndim != 2
        or group_size != 128
        or x.shape[-1] % 128
        or x.stride(-1) != 1
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise NotImplementedError(
            "INT8 indexer query requires float [M,128*G], feature stride 1"
        )
    groups = x.shape[-1] // 128
    if groups == 0:
        raise ValueError("INT8 indexer query width must be positive")
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scales = torch.empty((x.shape[0], groups), device=x.device, dtype=torch.float32)
    count = x.shape[0] * groups
    if count >= 1024:
        _quantize_query_batched[lambda meta: (triton.cdiv(count, meta["BLOCK_ROWS"]),)](
            x, q, scales, x.stride(0), count, groups, use_ue8m0
        )
    elif x.shape[0]:
        # Fixed 128-element reduction: no schedule/performance search dimensions.
        _quantize_query[(x.shape[0] * groups,)](
            x, q, scales, x.stride(0), groups, use_ue8m0, num_warps=4
        )
    return q, scales


def indexer_k_quant_and_cache_int8(k, cache, slots, quant_block_size, scale_fmt):
    """Write INT8 keys/scales, ignoring negative or out-of-capacity slots."""
    _cuda(k, cache, slots)
    if (
        k.ndim != 2
        or k.shape[1] != 128
        or k.stride(-1) != 1
        or k.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or quant_block_size != 128
        or scale_fmt not in (None, "ue8m0")
    ):
        raise NotImplementedError("INT8 indexer cache requires float [T,128] keys")
    if (
        cache.ndim != 3
        or cache.shape[-1] != 132
        or cache.dtype != torch.uint8
        or not cache.is_contiguous()
        or cache.shape[1] <= 0
    ):
        raise NotImplementedError("INT8 cache requires contiguous uint8 [P,B,132]")
    if (
        slots.ndim != 1
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.is_contiguous()
        or slots.numel() > k.shape[0]
    ):
        raise ValueError("INT8 cache slots must be contiguous int32/int64 [T]")
    if slots.numel():
        # Fixed 128-element reduction, one independent token per program.
        _quantize_cache[(slots.numel(),)](
            k,
            cache,
            cache.view(torch.float32),
            slots,
            k.stride(0),
            cache.stride(0),
            cache.shape[1],
            cache.shape[0] * cache.shape[1],
            scale_fmt == "ue8m0",
            num_warps=4,
        )


@libentry()
@libtuner(configs=runtime.get_tuned_config("int8_mqa_logits"), key=["M", "N", "H"])
@triton.jit
def _prefill(
    Q,
    K,
    KS,
    W,
    Starts,
    Ends,
    Out,
    M,
    N,
    H: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_wm,
    CLEAN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, 128)
    keys = tl.load(K + n[:, None] * stride_kn + d[None, :], n[:, None] < N, other=0)
    scales = tl.load(KS + n, n < N, other=0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for base in range(0, H, HEAD_BLOCK):
        h = base + tl.arange(0, HEAD_BLOCK)
        q = tl.load(
            Q
            + m[:, None, None] * stride_qm
            + h[None, :, None] * stride_qh
            + d[None, None, :],
            m[:, None, None] < M,
            other=0,
        )
        dots = tl.dot(
            tl.reshape(q, (BLOCK_M * HEAD_BLOCK, 128)),
            tl.trans(keys),
            out_dtype=tl.int32,
        )
        dots = tl.reshape(dots, (BLOCK_M, HEAD_BLOCK, BLOCK_N)).to(tl.float32)
        weights = tl.load(
            W + m[:, None] * stride_wm + h[None, :], m[:, None] < M, other=0
        )
        acc += tl.sum(
            tl.maximum(dots * scales[None, None, :], 0.0) * weights[:, :, None], 1
        )
    start = tl.load(Starts + m, m < M, other=0)
    end = tl.load(Ends + m, m < M, other=0)
    valid = (n[None, :] >= start[:, None]) & (n[None, :] < end[:, None])
    value = tl.where(valid, acc, float("-inf") if CLEAN else 0.0)
    tl.store(
        Out + m[:, None] * N + n[None, :], value, (m[:, None] < M) & (n[None, :] < N)
    )


def _query(q, rank):
    if (
        not isinstance(q, tuple)
        or len(q) != 2
        or q[1] is not None
        or q[0].ndim != rank
        or q[0].dtype != torch.int8
        or q[0].shape[-1] != 128
        or q[0].shape[-2] not in (16, 32, 64)
        or q[0].stride(-1) != 1
    ):
        raise NotImplementedError(
            "INT8 MQA requires (int8 [...,H,128], None), H=16/32/64"
        )
    return q[0]


@libentry()
@libtuner(configs=runtime.get_tuned_config("int8_mqa_logits_rows"), key=["M", "N", "H"])
@triton.jit
def _prefill_rows(
    Q,
    K,
    KS,
    W,
    Starts,
    Ends,
    Out,
    M,
    N,
    H: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_wm,
    CLEAN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    h, d = tl.arange(0, H), tl.arange(0, 128)
    q = tl.load(Q + row * stride_qm + h[:, None] * stride_qh + d[None, :])
    keys = tl.load(K + n[:, None] * stride_kn + d[None, :], n[:, None] < N, other=0)
    dots = tl.dot(q, tl.trans(keys), out_dtype=tl.int32).to(tl.float32)
    scales = tl.load(KS + n, n < N, other=0)
    weights = tl.load(W + row * stride_wm + h)
    logits = tl.sum(tl.maximum(dots * scales[None, :], 0.0) * weights[:, None], 0)
    start, end = tl.load(Starts + row), tl.load(Ends + row)
    valid = (n >= start) & (n < end)
    value = tl.where(valid, logits, float("-inf") if CLEAN else 0.0)
    tl.store(Out + row * N + n, value, n < N)


def int8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    """Weighted ReLU MQA from INT8 Q/K and per-key FP32 scales."""
    query = _query(q, 3)
    keys, scales = kv
    _cuda(query, keys, scales, weights, cu_seqlen_ks, cu_seqlen_ke)
    m, heads, _ = query.shape
    if (
        keys.ndim != 2
        or keys.shape[1] != 128
        or keys.dtype != torch.int8
        or keys.stride(1) != 1
    ):
        raise NotImplementedError(
            "INT8 MQA keys must be int8 [N,128], feature stride 1"
        )
    n = keys.shape[0]
    if (
        scales.shape != (n,)
        or scales.dtype != torch.float32
        or not scales.is_contiguous()
    ):
        raise ValueError("INT8 MQA scales must be contiguous float32 [N]")
    if (
        weights.shape != (m, heads)
        or weights.dtype != torch.float32
        or weights.stride(1) != 1
    ):
        raise ValueError("INT8 MQA weights must be float32 [M,H], head stride 1")
    for limits in (cu_seqlen_ks, cu_seqlen_ke):
        if (
            limits.shape != (m,)
            or limits.dtype != torch.int32
            or not limits.is_contiguous()
        ):
            raise ValueError("INT8 MQA valid ranges must be contiguous int32 [M]")
    out = torch.empty((m, n), device=query.device, dtype=torch.float32)
    if m and n:
        # Small M cannot amortize padding an M tile and looping over head pairs.
        # One row/head matrix per program keeps every MMA row useful.
        kernel = _prefill_rows if m <= 16 else _prefill
        kernel[
            lambda meta: (
                m if m <= 16 else triton.cdiv(m, meta["BLOCK_M"]),
                triton.cdiv(n, meta["BLOCK_N"]),
            )
        ](
            query,
            keys,
            scales,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            out,
            m,
            n,
            heads,
            query.stride(0),
            query.stride(1),
            keys.stride(0),
            weights.stride(0),
            clean_logits,
        )
    return out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("int8_paged_mqa_logits"),
    key=["ROWS", "CAPACITY", "H"],
)
@triton.jit
def _decode(
    Q,
    Cache,
    Scales,
    W,
    Lengths,
    Tables,
    Out,
    ROWS,
    CAPACITY,
    H: tl.constexpr,
    NEXT: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES,
    TABLE_WIDTH,
    stride_cache,
    stride_w,
    stride_bt,
    stride_btn,
    stride_len,
    stride_lenn,
    CLEAN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    batch, token = row // NEXT, row % NEXT
    length = tl.load(Lengths + batch * stride_len + token * stride_lenn)
    logical = pos // PAGE
    valid = (pos < CAPACITY) & (pos < length) & (logical < TABLE_WIDTH)
    physical = tl.load(
        Tables + batch * stride_bt + token * stride_btn + logical, valid, other=0
    )
    valid &= (physical >= 0) & (physical < PAGES)
    h, d = tl.arange(0, H), tl.arange(0, 128)
    query = tl.load(Q + row * H * 128 + h[:, None] * 128 + d[None, :])
    offsets = (
        physical[:, None].to(tl.int64) * stride_cache
        + (pos % PAGE)[:, None] * 128
        + d[None, :]
    )
    keys = tl.load(Cache + offsets, valid[:, None], other=0).to(tl.int8, bitcast=True)
    dots = tl.dot(query, tl.trans(keys), out_dtype=tl.int32).to(tl.float32)
    scales = tl.load(
        Scales + physical * (stride_cache // 4) + PAGE * 32 + pos % PAGE, valid, other=0
    )
    weights = tl.load(W + row * stride_w + h)
    logits = tl.sum(tl.maximum(dots * scales[None, :], 0.0) * weights[:, None], 0)
    logits = tl.where(valid, logits, float("-inf") if CLEAN else 0.0)
    tl.store(Out + row * CAPACITY + pos, logits, pos < CAPACITY)


def int8_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
):
    """Static-grid paged INT8 logits; lengths and mappings refresh on graph replay."""
    if isinstance(q, tuple) and q[0].ndim == 3:
        q = (q[0].unsqueeze(1), q[1])
    query = _query(q, 4)
    _cuda(query, kv_cache, weights, context_lens, block_tables)
    batch, next_n, heads, _ = query.shape
    rows = batch * next_n
    if not query.is_contiguous():
        raise NotImplementedError("Paged INT8 MQA requires contiguous Q")
    if (
        kv_cache.ndim != 4
        or kv_cache.shape[2:] != (1, 132)
        or kv_cache.dtype != torch.uint8
        or not kv_cache.is_contiguous()
        or kv_cache.shape[1] <= 0
    ):
        raise NotImplementedError(
            "Paged INT8 cache requires contiguous uint8 [P,B,1,132]"
        )
    if (
        weights.shape != (rows, heads)
        or weights.dtype != torch.float32
        or weights.stride(1) != 1
    ):
        raise ValueError("Paged INT8 MQA weights require float32 [B*N,H]")
    if (
        context_lens.dtype != torch.int32
        or context_lens.shape not in ((batch,), (batch, next_n))
        or block_tables.dtype != torch.int32
        or block_tables.ndim not in (2, 3)
        or block_tables.shape[0] != batch
        or block_tables.stride(-1) != 1
        or (block_tables.ndim == 3 and block_tables.shape[1] != next_n)
    ):
        raise ValueError("Paged INT8 MQA requires int32 lengths and page tables")
    if max_model_len < 0:
        raise ValueError("max_model_len must be nonnegative")
    out = torch.empty((rows, max_model_len), device=query.device, dtype=torch.float32)
    if rows and max_model_len:
        _decode[lambda meta: (rows, triton.cdiv(max_model_len, meta["BLOCK_N"]))](
            query,
            kv_cache,
            kv_cache.view(torch.float32),
            weights,
            context_lens,
            block_tables,
            out,
            rows,
            max_model_len,
            heads,
            next_n,
            kv_cache.shape[1],
            kv_cache.shape[0],
            block_tables.shape[-1],
            kv_cache.stride(0),
            weights.stride(0),
            block_tables.stride(0),
            block_tables.stride(1) if block_tables.ndim == 3 else 0,
            context_lens.stride(0),
            context_lens.stride(1) if context_lens.ndim == 2 else 0,
            clean_logits,
        )
    return out


int8_paged_mqa_logits.supports_cuda_graph = True

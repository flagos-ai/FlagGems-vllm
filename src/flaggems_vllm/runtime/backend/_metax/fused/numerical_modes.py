# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX numerical modes; public validation and shared kernel bodies stay in ops."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops._memory_overlap import tensor_ranges_overlap
from flaggems_vllm.ops.fused_minimax_m3_qknorm_rope_kv_insert import (
    _fused_minimax_m3_body,
)
from flaggems_vllm.ops.moe_sum import _moe_sum_general_kernel
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def _swigluoai_division_flat_kernel(
    X,
    Y,
    N: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    TOTAL: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row = i // N
    col = i % N
    gate = tl.minimum(tl.load(X + row * XS + col, i < TOTAL, 0).to(tl.float32), LIMIT)
    up = tl.minimum(
        tl.maximum(
            tl.load(X + row * XS + N + col, i < TOTAL, 0).to(tl.float32), -LIMIT
        ),
        LIMIT,
    )
    value = gate / (1.0 + tl.exp(-ALPHA * gate)) * (up + BETA)
    tl.store(Y + row * YS + col, value, i < TOTAL)


def division_swiglu(input, output, alpha, beta, limit):
    """Preserve the accepted division evaluation and flat launch."""
    if input.device.type != "cuda" or input.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise NotImplementedError("Division mode requires CUDA BF16/FP16/FP32")
    if input.stride(-1) != 1 or output.stride(-1) != 1:
        raise NotImplementedError("Division mode requires unit column strides")
    if tensor_ranges_overlap(input, output):
        raise ValueError("Input and output must not alias")
    if output.numel() >= 2**31:
        raise NotImplementedError("Division mode index range exceeded")
    _swigluoai_division_flat_kernel[(triton.cdiv(output.numel(), 256),)](
        input,
        output,
        output.size(1),
        input.stride(0),
        output.stride(0),
        output.numel(),
        alpha,
        beta,
        limit,
        256,
        num_warps=4,
    )
    return output


@triton.jit
def _staged_normalize(x, weights, EPS: tl.constexpr, DTYPE: tl.constexpr):
    rms = tl.rsqrt(tl.sum(x * x, 0) / 128 + EPS)
    normalized = (x * rms) * (1.0 + weights)
    return normalized.to(DTYPE).to(tl.float32)


@triton.jit
def _staged_rotate(normalized, partner, c, s, d, ROT: tl.constexpr):
    rotated = normalized * c + tl.where(d < ROT // 2, -partner, partner) * s
    return tl.where(d < ROT, rotated, normalized)


@triton.jit
def _strided_index_offset(
    slot,
    BLOCK_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    return (slot // BLOCK_SIZE) * BLOCK_STRIDE + (slot % BLOCK_SIZE) * TOKEN_STRIDE


@libentry()
@triton.jit
def _staged_minimax_m3_kernel(
    qkv,
    q_weight,
    k_weight,
    iq_weight,
    ik_weight,
    cos_sin,
    positions,
    slot_mapping,
    index_slot_mapping,
    kv_cache,
    index_cache,
    q_out,
    iq_out,
    KV_BLOCK_STRIDE: tl.constexpr,
    KV_KIND_STRIDE: tl.constexpr,
    KV_TOKEN_STRIDE: tl.constexpr,
    KV_HEAD_STRIDE: tl.constexpr,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    NIQ: tl.constexpr,
    ROT: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_INDEX: tl.constexpr,
    INSERT_KV: tl.constexpr,
    HAS_Q_OUT: tl.constexpr,
    HAS_IQ_OUT: tl.constexpr,
    FP8_KV: tl.constexpr,
    INDEX_BLOCK_STRIDE: tl.constexpr,
    INDEX_TOKEN_STRIDE: tl.constexpr,
    INDEX_BLOCK_SIZE: tl.constexpr,
    HEAD_CLASS: tl.constexpr,
):
    _fused_minimax_m3_body(
        qkv,
        q_weight,
        k_weight,
        iq_weight,
        ik_weight,
        cos_sin,
        positions,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        q_out,
        iq_out,
        KV_BLOCK_STRIDE,
        KV_KIND_STRIDE,
        KV_TOKEN_STRIDE,
        KV_HEAD_STRIDE,
        NQ,
        NKV,
        NIQ,
        ROT,
        EPS,
        BLOCK_SIZE,
        HAS_INDEX,
        INSERT_KV,
        HAS_Q_OUT,
        HAS_IQ_OUT,
        FP8_KV,
        NORMALIZE=_staged_normalize,
        ROTATE=_staged_rotate,
        INDEX_OFFSET=_strided_index_offset,
        INDEX_BLOCK_STRIDE=INDEX_BLOCK_STRIDE,
        INDEX_TOKEN_STRIDE=INDEX_TOKEN_STRIDE,
        INDEX_BLOCK_SIZE=INDEX_BLOCK_SIZE,
        HEAD_CLASS=HEAD_CLASS,
    )


def launch_staged_qkv(
    qkv,
    q_weight,
    k_weight,
    iq_weight,
    ik_weight,
    cos_sin,
    positions,
    slot_mapping,
    index_slot_mapping,
    kv_cache,
    index_cache,
    q_out,
    iq_out,
    cache_strides,
    nq,
    nkv,
    niq,
    rotary_dim,
    eps,
    block_size,
    has_index,
    insert_kv,
    fp8_kv,
):
    """Use the original four-warp tree and grouped norm-only launches."""
    packed_heads = nq + 2 * nkv + (niq + 1 if has_index else 0)
    if not insert_kv:
        launches = [(nq, 1), (nkv, 2)]
        if has_index:
            launches += [(niq, 3), (1, 4)]
    else:
        launches = [(packed_heads, 0)]
    for head_count, head_class in launches:
        _staged_minimax_m3_kernel[(qkv.shape[0], head_count)](
            qkv,
            q_weight,
            k_weight,
            iq_weight,
            ik_weight,
            cos_sin,
            positions,
            slot_mapping,
            index_slot_mapping,
            kv_cache,
            index_cache,
            q_out,
            iq_out,
            *cache_strides,
            nq,
            nkv,
            niq,
            rotary_dim,
            eps,
            block_size,
            has_index,
            insert_kv,
            q_out is not None,
            iq_out is not None,
            fp8_kv,
            HEAD_CLASS=head_class,
            INDEX_BLOCK_STRIDE=index_cache.stride(0) if index_cache is not None else 0,
            INDEX_TOKEN_STRIDE=index_cache.stride(1) if index_cache is not None else 0,
            INDEX_BLOCK_SIZE=index_cache.shape[1] if index_cache is not None else 1,
            num_warps=4,
            enable_fp_fusion=False,
        )


def sequential_moe_sum(input, output):
    """Select the existing sequential kernel with the accepted launch arguments."""
    num_tokens, topk, hidden_size = input.shape
    input_stride = input.stride()
    output_stride = output.stride()
    elem_size = input.element_size()
    # Keep the accepted narrow indexing only when every logical offset fits.
    # Other callers retain the shared kernel's safe int64 default.
    max_input_offset = (
        (num_tokens - 1) * input_stride[0]
        + (topk - 1) * input_stride[1]
        + (hidden_size - 1) * input_stride[2]
    )
    max_output_offset = (num_tokens - 1) * output_stride[0] + (
        hidden_size - 1
    ) * output_stride[1]
    use_int64_index = (
        max(num_tokens - 1, max_input_offset, max_output_offset) >= 2**31
        or min(*input_stride, *output_stride) < 0
    )
    grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
    _moe_sum_general_kernel[grid](
        input,
        output,
        num_tokens,
        topk,
        hidden_size,
        input_stride[0],
        input_stride[1],
        input_stride[2],
        output_stride[0],
        output_stride[1],
        ELEM_SIZE=elem_size,
        USE_INT64_INDEX=use_int64_index,
    )

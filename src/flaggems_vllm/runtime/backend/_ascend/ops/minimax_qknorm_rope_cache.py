# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MiniMax-M3 QK normalization, NeoX RoPE and paged cache insertion.

Packed or separate main/index projections produce Q and index Q. Programs
process tokens cyclically and reuse FP32 Gemma normalization scales. All
input rows produce queries; only actual tokens with nonnegative slots
write K/V or index K directly into their independent paged caches.
"""

from functools import lru_cache

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.experimental.tle.language.dsa import tile_store

try:
    from triton.language.extra.cann.extension import extract_slice, insert_slice
except ImportError:
    from triton.language import extract_slice, insert_slice


@lru_cache(None)
def _vector_core_count(device):
    properties = triton.runtime.driver.active.utils.get_device_properties(device)
    return int(properties["num_vectorcore"])


def get_vectorcore_num():
    return _vector_core_count(torch.npu.current_device())


def init_device_properties_triton():
    get_vectorcore_num()


FP8_E4M3_MAX = 448.0


# Request-dependent bounds must not specialize on their value or alignment.
# Head dimensions and layout strides below are fixed for a warmed deployment.
@triton.jit(do_not_specialize=["NUM_SLOTS", "NUM_TOKENS"])
def _minimax_qknorm_rope_cache_kernel(
    packed,
    index_packed,
    cos_sin,
    positions,
    q_weight,
    k_weight,
    iq_weight,
    ik_weight,
    query,
    index_query,
    key_cache,
    value_cache,
    index_cache,
    slots,
    index_slots,
    NUM_SLOTS,
    NUM_TOKENS,
    KCAP,
    ICAP,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    IQ_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    INDEX_INPUT_STRIDE: tl.constexpr,
    INDEX_OFFSET: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    COS_SIN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_BLOCK: tl.constexpr,
    V_TOKEN: tl.constexpr,
    V_HEAD: tl.constexpr,
    I_BLOCK: tl.constexpr,
    I_TOKEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INDEX_BLOCK_SIZE: tl.constexpr,
    CACHE_FP8: tl.constexpr,
    INDEX_FP8: tl.constexpr,
    FP8_E4M3_MAX: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    programs = tl.num_programs(0).to(tl.int64)
    weight_dim = tl.arange(0, HEAD_DIM)
    q_weight_value = tle.load(q_weight + weight_dim).to(tl.float32)
    k_weight_value = tle.load(k_weight + weight_dim).to(tl.float32)
    iq_weight_value = tle.load(iq_weight + weight_dim).to(tl.float32)
    ik_weight_value = tle.load(ik_weight + weight_dim).to(tl.float32)
    MAIN: tl.constexpr = Q_HEADS + KV_HEADS
    HEADS: tl.constexpr = MAIN + IQ_HEADS + 1
    h = tl.arange(0, HEADS)
    w = tl.where(
        h[:, None] < Q_HEADS,
        q_weight_value[None, :],
        tl.where(
            h[:, None] < MAIN,
            k_weight_value[None, :],
            tl.where(
                h[:, None] < MAIN + IQ_HEADS,
                iq_weight_value[None, :],
                ik_weight_value[None, :],
            ),
        ),
    )
    scale = 1.0 + w
    # Flatten contiguous paged caches into rows. Use original kernel pointers
    # as view bases; slot (in elements of rows) is the only sparse index.
    kc = tle.make_gather_scatter_view(
        key_cache,
        shape=[KCAP, KV_HEADS * HEAD_DIM],
        strides=[KV_HEADS * HEAD_DIM, 1],
        tile=[1, KV_HEADS * HEAD_DIM],
        sparse_dim=[0],
    )
    vc = tle.make_gather_scatter_view(
        value_cache,
        shape=[KCAP, KV_HEADS * HEAD_DIM],
        strides=[KV_HEADS * HEAD_DIM, 1],
        tile=[1, KV_HEADS * HEAD_DIM],
        sparse_dim=[0],
    )
    ic = tle.make_gather_scatter_view(
        index_cache,
        shape=[ICAP, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        tile=[1, HEAD_DIM],
        sparse_dim=[0],
    )
    for token in range(program, NUM_TOKENS, programs):
        dim = tl.arange(0, HEAD_DIM)
        row = packed + token * INPUT_STRIDE
        irow = index_packed + token * INDEX_INPUT_STRIDE + INDEX_OFFSET
        # Two contiguous reads are assembled before normalization.
        main = tle.load(row + tl.arange(0, MAIN)[:, None] * HEAD_DIM + dim[None, :])
        idx = tle.load(
            irow + tl.arange(0, IQ_HEADS + 1)[:, None] * HEAD_DIM + dim[None, :]
        )
        joined = tl.full((HEADS, HEAD_DIM), 0, packed.dtype.element_ty)
        joined = insert_slice(joined, main, (0, 0), (MAIN, HEAD_DIM), (1, 1))
        joined = insert_slice(joined, idx, (MAIN, 0), (IQ_HEADS + 1, HEAD_DIM), (1, 1))
        position = tle.load(positions + token * POSITION_STRIDE).to(tl.int64)
        cs = cos_sin + position * COS_SIN_STRIDE
        x = joined.to(tl.float32)
        x = x * tl.rsqrt(tl.sum(x * x, 1) / HEAD_DIM + EPS)[:, None] * scale
        rounded = x.to(packed.dtype.element_ty)
        ub = tle.dsa.tile_alloc(
            [HEADS, HEAD_DIM], packed.dtype.element_ty, tle.dsa.ascend.UB
        )
        bound = tle.dsa.to_buffer(rounded, space=tle.dsa.ascend.UB, bind_buffer=ub)
        bf = tle.dsa.tile_to_tensor(bound)
        half: tl.constexpr = ROTARY_DIM // 2
        c = tle.load(cs + tl.arange(0, half)).to(tl.float32)
        s = tle.load(cs + half + tl.arange(0, half)).to(tl.float32)
        a = extract_slice(bf, (0, 0), (HEADS, half), (1, 1)).to(tl.float32)
        b = extract_slice(bf, (0, half), (HEADS, half), (1, 1)).to(tl.float32)
        y = insert_slice(
            bf,
            (a * c[None, :] - b * s[None, :]).to(packed.dtype.element_ty),
            (0, 0),
            (HEADS, half),
            (1, 1),
        )
        y = insert_slice(
            y,
            (b * c[None, :] + a * s[None, :]).to(packed.dtype.element_ty),
            (0, half),
            (HEADS, half),
            (1, 1),
        )
        q = extract_slice(y, (0, 0), (Q_HEADS, HEAD_DIM), (1, 1))
        iq = extract_slice(y, (MAIN, 0), (IQ_HEADS, HEAD_DIM), (1, 1))
        tl.store(
            query
            + token * Q_HEADS * HEAD_DIM
            + tl.arange(0, Q_HEADS)[:, None] * HEAD_DIM
            + dim[None, :],
            q,
        )
        tl.store(
            index_query
            + token * IQ_HEADS * HEAD_DIM
            + tl.arange(0, IQ_HEADS)[:, None] * HEAD_DIM
            + dim[None, :],
            iq,
        )
        slot = tle.load(slots + token, token < NUM_SLOTS, other=-1).to(tl.int64)
        if (slot >= 0) & (slot < KCAP):
            k = extract_slice(y, (Q_HEADS, 0), (KV_HEADS, HEAD_DIM), (1, 1))
            v = tle.load(
                row
                + MAIN * HEAD_DIM
                + tl.arange(0, KV_HEADS)[:, None] * HEAD_DIM
                + dim[None, :]
            )
            ids = tl.full((1,), 0, tl.int64) + slot
            tile_store(
                kc,
                value=tl.reshape(k, (1, KV_HEADS * HEAD_DIM)),
                index=(ids, 0),
                src_space=tle.dsa.ascend.UB,
            )
            tile_store(
                vc,
                value=tl.reshape(v, (1, KV_HEADS * HEAD_DIM)),
                index=(ids, 0),
                src_space=tle.dsa.ascend.UB,
            )
        slot = tle.load(index_slots + token, token < NUM_SLOTS, other=-1).to(tl.int64)
        if (slot >= 0) & (slot < ICAP):
            ik = extract_slice(y, (MAIN + IQ_HEADS, 0), (1, HEAD_DIM), (1, 1))
            ids = tl.full((1,), 0, tl.int64) + slot
            tile_store(ic, value=ik, index=(ids, 0), src_space=tle.dsa.ascend.UB)


def minimax_qknorm_rope_cache(
    packed: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    num_actual_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
    index_packed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Q/index Q and insert K/V/index K, with no intermediate KV tensors.

    Caches use [block, token, head, dim] and [block, token, dim] layouts.
    Head dimensions must match (the MiniMax-M3 128-wide main/index heads).
    The caller owns cache lifetime and orders cache readers after this call.
    """
    if (
        (num_q_heads not in (8, 16) or num_kv_heads != 1 or num_index_heads != 1)
        or packed.dtype != torch.bfloat16
        or any(
            t.dtype != torch.bfloat16 or not t.is_contiguous()
            for t in (key_cache, value_cache, index_cache)
        )
    ):
        raise NotImplementedError(
            "Ascend MiniMax requires BF16 input/cache and Q8 or Q16, KV1, index-Q1"
        )
    weights = (q_weight, k_weight, index_q_weight, index_k_weight)
    slots = (slot_mapping, index_slot_mapping)
    inputs = (
        *weights,
        *slots,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        index_cache,
    )
    if index_packed is not None:
        inputs = (*inputs, index_packed)
    if any(t.device != packed.device for t in inputs):
        raise ValueError("all inputs and caches must be on the packed input device")
    if packed.requires_grad or any(t.requires_grad for t in inputs):
        raise NotImplementedError("Ascend MiniMax cache insertion is inference-only")
    head_dim, rotary_dim = 128, 64
    if any(w.shape != (head_dim,) or not w.is_contiguous() for w in weights):
        raise ValueError("expected four contiguous D128 weight vectors")
    if any(w.dtype not in (torch.bfloat16, torch.float32) for w in weights):
        raise NotImplementedError("weights must be BF16 or FP32")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or cos_sin_cache.stride(1) != 1
        or cos_sin_cache.dtype != packed.dtype
    ):
        raise ValueError(
            "expected a BF16 [P,64] cos/sin table with contiguous last dimension"
        )
    if packed.ndim != 2 or packed.stride(1) != 1:
        raise ValueError("expected a packed matrix with contiguous last dimension")
    main_width = (num_q_heads + 2 * num_kv_heads) * head_dim
    index_width = (num_index_heads + 1) * head_dim
    if index_packed is None:
        if packed.shape[1] != main_width + index_width:
            raise ValueError("packed width must include the main and index projections")
        index_packed, index_offset = packed, main_width
    else:
        if (
            packed.shape[1] != main_width
            or index_packed.shape != (packed.shape[0], index_width)
            or index_packed.stride(1) != 1
            or index_packed.dtype != packed.dtype
        ):
            raise ValueError("invalid separate main/index projections")
        index_offset = 0
    if (
        key_cache.ndim != 4
        or key_cache.shape != value_cache.shape
        or key_cache.shape[2:] != (num_kv_heads, head_dim)
        or index_cache.ndim != 3
        or index_cache.shape[-1] != head_dim
        or min(key_cache.shape[:2] + index_cache.shape[:2]) < 1
    ):
        raise ValueError("expected positive-capacity paged K/V and index-K caches")
    if not 0 <= num_actual_tokens <= packed.shape[0]:
        raise ValueError(
            "num_actual_tokens must be between zero and the packed row count"
        )
    if any(
        s.ndim != 1
        or s.dtype != torch.int64
        or not s.is_contiguous()
        or s.numel() < num_actual_tokens
        for s in slots
    ):
        raise ValueError(
            "slot mappings must be contiguous int64 vectors covering all actual tokens"
        )
    if positions.shape != (packed.shape[0],) or positions.dtype != torch.int64:
        raise ValueError("positions must be an int64 vector covering all input rows")
    num_tokens = packed.shape[0]
    query = torch.empty(
        (num_tokens, num_q_heads * head_dim), device=packed.device, dtype=packed.dtype
    )
    index_query = torch.empty(
        (num_tokens, num_index_heads * head_dim),
        device=packed.device,
        dtype=index_cache.dtype,
    )
    if num_tokens == 0:
        return query, index_query
    with torch.npu.device(packed.device):
        # Each program owns tokens congruent to its id modulo this grid.
        num_programs = min(num_tokens, get_vectorcore_num())
        _minimax_qknorm_rope_cache_kernel[(num_programs,)](
            packed,
            index_packed,
            cos_sin_cache,
            positions,
            q_weight,
            k_weight,
            index_q_weight,
            index_k_weight,
            query,
            index_query,
            key_cache,
            value_cache,
            index_cache,
            slot_mapping,
            index_slot_mapping,
            num_actual_tokens,
            num_tokens,
            key_cache.shape[0] * key_cache.shape[1],
            index_cache.shape[0] * index_cache.shape[1],
            num_q_heads,
            num_kv_heads,
            num_index_heads,
            head_dim,
            rotary_dim,
            packed.stride(0),
            index_packed.stride(0),
            index_offset,
            positions.stride(0),
            cos_sin_cache.stride(0),
            eps,
            *key_cache.stride()[:3],
            *value_cache.stride()[:3],
            *index_cache.stride()[:2],
            key_cache.shape[1],
            index_cache.shape[1],
            False,
            False,
            FP8_E4M3_MAX=FP8_E4M3_MAX,
            enable_fp_fusion=False,
        )
    return query, index_query

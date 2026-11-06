# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagGems/Triton MiniMax M3 preprocessing and paged KV insertion.

The main FP8 cache has identity scale, matching MiniMax M3 sparse attention's
reader. Norm and RoPE remain FP32 until the final BF16/FP8 store. This module
uses no vLLM compiled extension and does not quantize query tensors.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@triton.jit
def _native_warp_sumsq(x):
    """Mirror the CUDA reference's four values per lane and XOR reduction."""
    matrix = tl.reshape(x, (32, 4))
    lanes = tl.arange(0, 32)
    x0 = tl.reshape(tl.gather(matrix, tl.full((32, 1), 0, tl.int32), 1), (32,))
    x1 = tl.reshape(tl.gather(matrix, tl.full((32, 1), 1, tl.int32), 1), (32,))
    x2 = tl.reshape(tl.gather(matrix, tl.full((32, 1), 2, tl.int32), 1), (32,))
    x3 = tl.reshape(tl.gather(matrix, tl.full((32, 1), 3, tl.int32), 1), (32,))
    sumsq = x0 * x0
    sumsq = tl.fma(x1, x1, sumsq)
    sumsq = tl.fma(x2, x2, sumsq)
    sumsq = tl.fma(x3, x3, sumsq)
    for shift in tl.static_range(5):
        sumsq += tl.gather(sumsq, lanes ^ (16 >> shift), 0)
    return tl.sum(tl.where(lanes == 0, sumsq, 0.0), 0)


@libentry()
@triton.jit
def _fused_minimax_m3_kernel(
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
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    packed_heads: tl.constexpr = NQ + 2 * NKV + (NIQ + 1 if HAS_INDEX else 0)
    src = qkv + token.to(tl.int64) * packed_heads * 128 + head * 128 + d
    x = tl.load(src).to(tl.float32)
    is_v = (head >= NQ + NKV) & (head < NQ + 2 * NKV)

    if is_v:
        result = x
    else:
        if head < NQ:
            weight_ptr = q_weight
        elif head < NQ + NKV:
            weight_ptr = k_weight
        elif HAS_INDEX and head < NQ + 2 * NKV + NIQ:
            weight_ptr = iq_weight
        else:
            weight_ptr = ik_weight
        weights = tl.load(weight_ptr + d).to(tl.float32)
        rms = tl.rsqrt(_native_warp_sumsq(x) / 128.0 + EPS)
        normalized = (x * rms) * (1.0 + weights)

        # NeoX pairs the two halves of the rotary prefix. The remaining
        # dimensions retain their RMSNorm values.
        partner_d = tl.where(d < ROT, d ^ (ROT // 2), d)
        partner = tl.gather(normalized, partner_d, 0)
        pos = tl.load(positions + token)
        phase_d = d % (ROT // 2)
        c = tl.load(cos_sin + pos * ROT + phase_d, mask=d < ROT, other=1).to(tl.float32)
        s = tl.load(cos_sin + pos * ROT + ROT // 2 + phase_d, mask=d < ROT, other=0).to(
            tl.float32
        )
        # Explicit FMA preserves the native fused-RoPE rounding order.
        first = tl.fma(normalized, c, -partner * s)
        second = tl.fma(normalized, c, partner * s)
        result = tl.where(d < ROT, tl.where(d < ROT // 2, first, second), normalized)

        if HAS_Q_OUT and head < NQ:
            tl.store(q_out + token.to(tl.int64) * NQ * 128 + head * 128 + d, result)
        elif HAS_IQ_OUT and head >= NQ + 2 * NKV and head < NQ + 2 * NKV + NIQ:
            index_head = head - NQ - 2 * NKV
            tl.store(
                iq_out + token.to(tl.int64) * NIQ * 128 + index_head * 128 + d, result
            )
        else:
            tl.store(src, result)

    if INSERT_KV:
        if head >= NQ and head < NQ + 2 * NKV:
            slot = tl.load(slot_mapping + token).to(tl.int64)
            if slot >= 0:
                is_key = head < NQ + NKV
                kv_head = tl.where(is_key, head - NQ, head - NQ - NKV)
                kind = tl.where(is_key, 0, 1)
                target = (
                    (slot // BLOCK_SIZE) * KV_BLOCK_STRIDE
                    + kind * KV_KIND_STRIDE
                    + (slot % BLOCK_SIZE) * KV_TOKEN_STRIDE
                    + kv_head * KV_HEAD_STRIDE
                    + d
                )
                if FP8_KV:
                    encoded = result.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
                    tl.store(kv_cache + target, encoded)
                else:
                    tl.store(kv_cache + target, result)
        elif HAS_INDEX and head == NQ + 2 * NKV + NIQ:
            index_slot = tl.load(index_slot_mapping + token).to(tl.int64)
            if index_slot >= 0:
                tl.store(index_cache + index_slot * 128 + d, result)


@torch.library.custom_op(
    "flaggems_vllm::fused_minimax_m3_qknorm_rope_kv_insert",
    mutates_args=("qkv", "kv_cache", "index_cache", "q_out", "index_q_out"),
)
def _fused_minimax_m3_op(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None,
    index_k_norm_weight: torch.Tensor | None,
    num_index_heads: int,
    slot_mapping: torch.Tensor | None,
    index_slot_mapping: torch.Tensor | None,
    kv_cache: torch.Tensor | None,
    index_cache: torch.Tensor | None,
    block_size: int,
    q_out: torch.Tensor | None,
    index_q_out: torch.Tensor | None,
    kv_cache_dtype: str,
) -> None:
    """Same mutation/gather contract as vLLM's MiniMax M3 fused wrapper."""
    if qkv.device.type != "cuda" or qkv.dtype != torch.bfloat16:
        raise NotImplementedError("M3 cache writer requires CUDA BF16 QKV")
    assert num_heads > 0 and num_kv_heads > 0 and num_index_heads >= 0
    assert eps >= 0
    assert qkv.is_contiguous()
    assert positions.dtype == torch.int64 and positions.device == qkv.device
    assert positions.is_contiguous()
    assert positions.numel() >= qkv.shape[0]
    assert cos_sin_cache.dtype == qkv.dtype and cos_sin_cache.is_contiguous()
    assert cos_sin_cache.device == qkv.device
    assert rotary_dim in (8, 16, 32, 64, 128)
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] == rotary_dim
    has_index = num_index_heads > 0
    insert_kv = kv_cache is not None
    packed_heads = (
        num_heads + 2 * num_kv_heads + (num_index_heads + 1 if has_index else 0)
    )
    assert qkv.ndim == 2 and qkv.shape[1] == packed_heads * 128
    for weight in (q_norm_weight, k_norm_weight):
        assert (
            weight.dtype == qkv.dtype
            and weight.numel() == 128
            and weight.is_contiguous()
        )
        assert weight.device == qkv.device
    if has_index:
        for weight in (index_q_norm_weight, index_k_norm_weight):
            assert (
                weight is not None
                and weight.dtype == qkv.dtype
                and weight.numel() == 128
            )
            assert weight.is_contiguous() and weight.device == qkv.device
    fp8_kv = kv_cache_dtype in ("fp8", "fp8_e4m3")
    if kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        raise NotImplementedError(
            "MiniMax M3 FlagGems supports BF16 or FP8 E4M3 main KV only"
        )
    if insert_kv:
        assert block_size > 0
        assert has_index and slot_mapping is not None and index_cache is not None
        assert slot_mapping.dtype == torch.int64 and slot_mapping.device == qkv.device
        assert slot_mapping.numel() >= qkv.shape[0] and slot_mapping.is_contiguous()
        assert kv_cache.ndim == 5 and kv_cache.shape[1] == 2
        assert (
            kv_cache.shape[2:] == (block_size, num_kv_heads, 128)
            and kv_cache.stride(-1) == 1
        )
        assert kv_cache.dtype == (torch.uint8 if fp8_kv else qkv.dtype)
        assert kv_cache.device == index_cache.device == qkv.device
        assert index_cache.dtype == qkv.dtype and index_cache.is_contiguous()
        assert index_cache.ndim >= 2 and index_cache.shape[-1] == 128
        if index_slot_mapping is None:
            index_slot_mapping = slot_mapping
        assert (
            index_slot_mapping.dtype == torch.int64
            and index_slot_mapping.device == qkv.device
        )
        assert (
            index_slot_mapping.numel() >= qkv.shape[0]
            and index_slot_mapping.is_contiguous()
        )
        cache_strides = kv_cache.stride()[:4]
    else:
        cache_strides = (0, 0, 0, 0)
    for output, heads in ((q_out, num_heads), (index_q_out, num_index_heads)):
        if output is not None:
            assert output.dtype == qkv.dtype and output.is_contiguous()
            assert (
                output.device == qkv.device
                and output.numel() == qkv.shape[0] * heads * 128
            )
    if index_q_out is not None:
        assert has_index
    if qkv.shape[0] == 0:
        return
    with torch_device_fn.device(qkv.device):
        _fused_minimax_m3_kernel[(qkv.shape[0], packed_heads)](
            qkv,
            q_norm_weight,
            k_norm_weight,
            index_q_norm_weight if has_index else q_norm_weight,
            index_k_norm_weight if has_index else k_norm_weight,
            cos_sin_cache,
            positions,
            slot_mapping,
            index_slot_mapping,
            kv_cache,
            index_cache,
            q_out,
            index_q_out,
            *cache_strides,
            num_heads,
            num_kv_heads,
            num_index_heads,
            rotary_dim,
            float(eps),
            block_size,
            has_index,
            insert_kv,
            q_out is not None,
            index_q_out is not None,
            fp8_kv,
            # One warp owns the native 32-lane, four-values-per-lane reduction.
            num_warps=1,
            enable_fp_fusion=False,
        )


def fused_minimax_m3_qknorm_rope_kv_insert(
    qkv,
    q_norm_weight,
    k_norm_weight,
    cos_sin_cache,
    positions,
    num_heads,
    num_kv_heads,
    rotary_dim,
    eps,
    index_q_norm_weight=None,
    index_k_norm_weight=None,
    num_index_heads=0,
    slot_mapping=None,
    index_slot_mapping=None,
    kv_cache=None,
    index_cache=None,
    block_size=0,
    q_out=None,
    index_q_out=None,
    kv_cache_dtype="auto",
):
    # Supply all arguments to the internal op. Optional mutable tensors with
    # schema defaults otherwise hit a PyTorch 2.11 version-counter bug when
    # trailing None/default arguments are omitted by dispatcher canonicalization.
    return _fused_minimax_m3_op(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        kv_cache_dtype,
    )


__all__ = ["fused_minimax_m3_qknorm_rope_kv_insert"]


@_fused_minimax_m3_op.register_fake
def _fake_fused_minimax_m3(*args, **kwargs):
    # All results are mutations of existing buffers, including optional
    # gathered Q/index-Q. This keeps the same compile boundary as native.
    return None

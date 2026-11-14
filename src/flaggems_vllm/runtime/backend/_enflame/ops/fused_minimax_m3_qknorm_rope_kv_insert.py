# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagGems/Triton MiniMax M3 preprocessing and paged KV insertion.

The main FP8 cache has identity scale, matching MiniMax M3 sparse attention's
reader. Norm and RoPE remain FP32 until the final BF16/FP8 store. This module
uses no vLLM compiled extension and does not quantize query tensors.
"""

import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


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
    n_tokens,
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
    TOKENS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    packed_heads: tl.constexpr = NQ + 2 * NKV + (NIQ + 1 if HAS_INDEX else 0)
    # GCU: keep the addressing chain in the i32 safe domain (token/head counts
    # stay far below the 2**31 element limit of real deployments).
    is_v = (head >= NQ + NKV) & (head < NQ + 2 * NKV)
    if head < NQ:
        weight_ptr = q_weight
    elif head < NQ + NKV:
        weight_ptr = k_weight
    elif HAS_INDEX and head < NQ + 2 * NKV + NIQ:
        weight_ptr = iq_weight
    else:
        weight_ptr = ik_weight
    weights = tl.load(weight_ptr + d).to(tl.float32)

    # Token-invariant rotary helpers. GCU: tl.gather is not implemented by the
    # backend, so the warp-shuffle sum and the in-register partner lookup are
    # replaced by a plain tree reduction and masked affine reloads.
    half = ROT // 2
    up = d < half
    down = (d >= half) & (d < ROT)
    w_up = tl.load(weight_ptr + d + half, mask=up, other=0.0).to(tl.float32)
    w_down = tl.load(weight_ptr + d - half, mask=down, other=0.0).to(tl.float32)
    phase_d = d % half
    rot_lane = d < ROT

    # Each program walks TOKENS_PER_PROG rows of one head; the host picks the
    # block factor so the grid stays in the latency-hiding sweet spot.
    start = pid * TOKENS_PER_PROG
    count = tl.minimum(n_tokens - start, TOKENS_PER_PROG)
    for i in range(0, count):
        token = start + i
        row = qkv + (token * packed_heads + head) * 128
        src = row + d
        x = tl.load(src).to(tl.float32)

        if is_v:
            result = x
        else:
            rms = tl.rsqrt(tl.sum(x * x, 0) / 128.0 + EPS)
            normalized = (x * rms) * (1.0 + weights)

            # NeoX pairs the two halves of the rotary prefix; the remaining
            # dimensions retain their RMSNorm values.
            x_up = tl.load(row + d + half, mask=up, other=0.0).to(tl.float32)
            x_down = tl.load(row + d - half, mask=down, other=0.0).to(tl.float32)
            partner = tl.where(
                up, (x_up * rms) * (1.0 + w_up), (x_down * rms) * (1.0 + w_down)
            )
            pos = tl.load(positions + token).to(tl.int32)
            c = tl.load(cos_sin + pos * ROT + phase_d, mask=rot_lane, other=1).to(
                tl.float32
            )
            s = tl.load(
                cos_sin + pos * ROT + half + phase_d, mask=rot_lane, other=0
            ).to(tl.float32)
            # Explicit FMA preserves the native fused-RoPE rounding order.
            first = tl.fma(normalized, c, -partner * s)
            second = tl.fma(normalized, c, partner * s)
            result = tl.where(rot_lane, tl.where(up, first, second), normalized)

            if HAS_Q_OUT and head < NQ:
                tl.store(q_out + (token * NQ + head) * 128 + d, result)
            elif HAS_IQ_OUT and head >= NQ + 2 * NKV and head < NQ + 2 * NKV + NIQ:
                index_head = head - NQ - 2 * NKV
                tl.store(iq_out + (token * NIQ + index_head) * 128 + d, result)
            else:
                tl.store(src, result)

        if INSERT_KV:
            if head >= NQ and head < NQ + 2 * NKV:
                slot = tl.load(slot_mapping + token).to(tl.int32)
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
                index_slot = tl.load(index_slot_mapping + token).to(tl.int32)
                if index_slot >= 0:
                    tl.store(index_cache + index_slot * 128 + d, result)


@libentry()
@triton.jit
def _fused_minimax_m3_tile_kernel(
    qkv,
    q_weight,
    k_weight,
    iq_weight,
    ik_weight,
    cos_sin,
    positions,
    scratch,
    q_out,
    iq_out,
    n_tokens,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    NIQ: tl.constexpr,
    ROT: tl.constexpr,
    EPS: tl.constexpr,
    HAS_INDEX: tl.constexpr,
    HAS_Q_OUT: tl.constexpr,
    HAS_IQ_OUT: tl.constexpr,
    TPP: tl.constexpr,
    H_PAD: tl.constexpr,
):
    """No-insert fast arm: per token-block program over all normed heads.

    The NeoX row is consumed as column slices (up half / down half / tail),
    so the RoPE partner is the opposite slice and no cross-lane shuffle,
    128-lane non-uniform bool mask, or in-register layout op is needed
    (P03). Positions are loaded per token as scalars and the cos/sin rows
    are staged through a small global scratch so every head consumes them
    as affine (TPP, half) tiles; a direct data-dependent 2D gather of the
    cos_sin rows lowers to per-element addressing and costs far more than
    the whole kernel.
    """
    pid = tl.program_id(0)
    packed_heads: tl.constexpr = NQ + 2 * NKV + (NIQ + 1 if HAS_INDEX else 0)
    half: tl.constexpr = ROT // 2
    ah = tl.arange(0, half)
    at = tl.arange(0, 64)
    start = pid * TPP
    toks = start + tl.arange(0, TPP)
    tmask = toks < n_tokens
    toks_c = tl.minimum(toks, n_tokens - 1)

    # Stage this block's cos/sin rows: scalar position loads plus 1D
    # scalar-base row reads (all fast, affine lowering), stored once and
    # re-read as affine tiles shared by every head below.
    for t in tl.static_range(TPP):
        tokc = tl.minimum(start + t, n_tokens - 1)
        pos = tl.load(positions + tokc).to(tl.int32)
        cv = tl.load(cos_sin + pos * ROT + ah).to(tl.float32)
        sv = tl.load(cos_sin + pos * ROT + half + ah).to(tl.float32)
        tl.store(scratch + tokc * ROT + ah, cv.to(scratch.dtype.element_ty))
        tl.store(scratch + tokc * ROT + half + ah, sv.to(scratch.dtype.element_ty))
    c = tl.load(scratch + toks_c[:, None] * ROT + ah[None, :]).to(tl.float32)
    s = tl.load(scratch + toks_c[:, None] * ROT + (half + ah)[None, :]).to(tl.float32)

    for j in tl.static_range(H_PAD):
        if (j < packed_heads) and not ((j >= NQ + NKV) and (j < NQ + 2 * NKV)):
            if j < NQ:
                weight_ptr = q_weight
            elif j < NQ + NKV:
                weight_ptr = k_weight
            elif HAS_INDEX and j < NQ + 2 * NKV + NIQ:
                weight_ptr = iq_weight
            else:
                weight_ptr = ik_weight
            head = j
            row = qkv + (toks_c * packed_heads + head) * 128
            A = tl.load(row[:, None] + ah[None, :]).to(tl.float32)
            B = tl.load(row[:, None] + (half + ah)[None, :]).to(tl.float32)
            ss = tl.sum(A * A, 1) + tl.sum(B * B, 1)
            if ROT == 64:
                T = tl.load(row[:, None] + (ROT + at)[None, :]).to(tl.float32)
                ss += tl.sum(T * T, 1)
            rms = tl.rsqrt(ss / 128.0 + EPS)
            wA = tl.load(weight_ptr + ah).to(tl.float32)
            wB = tl.load(weight_ptr + half + ah).to(tl.float32)
            nA = (A * rms[:, None]) * (1.0 + wA[None, :])
            nB = (B * rms[:, None]) * (1.0 + wB[None, :])
            outA = tl.fma(nA, c, -nB * s)
            outB = tl.fma(nB, c, nA * s)

            if HAS_Q_OUT and head < NQ:
                obase = q_out + (toks_c * NQ + head) * 128
            elif (
                HAS_IQ_OUT
                and HAS_INDEX
                and head >= NQ + 2 * NKV
                and head < NQ + 2 * NKV + NIQ
            ):
                index_head = head - NQ - 2 * NKV
                obase = iq_out + (toks_c * NIQ + index_head) * 128
            else:
                obase = row
            tl.store(obase[:, None] + ah[None, :], outA, mask=tmask[:, None])
            tl.store(obase[:, None] + (half + ah)[None, :], outB, mask=tmask[:, None])
            if ROT == 64:
                wT = tl.load(weight_ptr + ROT + at).to(tl.float32)
                nT = (T * rms[:, None]) * (1.0 + wT[None, :])
                tl.store(obase[:, None] + (ROT + at)[None, :], nT, mask=tmask[:, None])


def _int32_index_view(tensor):
    """Env-aware int32 access path over a GCU int64 index tensor.

    Without TORCH_GCU_ENABLE_INT64_AND_UINT64 torch_gcu materializes int64
    tensors as compact int32 payload words ("GCU not support Long use Int
    replace"), so a zero-copy int32 view lets the kernel load the logical
    values while dodging the in-kernel int64 mis-narrowing load
    (compiler issue triton_i64_narrow_load) and the ATen cast fallback kernel.
    With the env gate ON torch_gcu stores real 8-byte int64 slots; the same
    word view then reads neighbor-joined garbage ([lo0, hi0, lo1, ...]), so
    the tensor is passed through unchanged and the kernel loads int64 values
    natively (narrowed to int32 right after each load; token/slot values are
    far below the 2**31 element safe domain).
    """
    if tensor.dtype != torch.int64:
        return tensor
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "") in ("1", "true"):
        return tensor
    view = torch.empty(0, device=tensor.device, dtype=torch.int32)
    view.set_(
        tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride()
    )
    return view


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
    # GCU: torch_gcu remaps device="cuda" to the "gcu" device type.
    if qkv.device.type not in ("cuda", "gcu") or qkv.dtype != torch.bfloat16:
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
    # GCU: env-aware index access (see _int32_index_view). Without the
    # int64 gate the backend stores int64 payloads as compact int32 words
    # (in-kernel int64 loads mis-narrow, compiler issue
    # triton_i64_narrow_load), so the zero-copy int32 view is used; with
    # the gate ON the int64 tensor is passed through and the kernels load
    # the native 8-byte values (narrowed to int32 right after each load).
    if positions.dtype != torch.int32:
        if qkv.device.type == "gcu":
            positions = _int32_index_view(positions)
        else:
            positions = positions.to(torch.int32)
    if slot_mapping is not None and slot_mapping.dtype != torch.int32:
        if qkv.device.type == "gcu":
            slot_mapping = _int32_index_view(slot_mapping)
        else:
            slot_mapping = slot_mapping.to(torch.int32)
    if index_slot_mapping is not None and index_slot_mapping.dtype != torch.int32:
        if qkv.device.type == "gcu":
            index_slot_mapping = _int32_index_view(index_slot_mapping)
        else:
            index_slot_mapping = index_slot_mapping.to(torch.int32)
    # GCU: per-program fixed overhead dominates small grids, while long
    # serial walks starve latency hiding; ~512 programs is the sweet spot.
    # The tile fast arm (no KV insert, RoPE dims 64/128, enough tokens to
    # fill the grid) rewrites the same math as column-slice tiles shared
    # across heads with scratch-staged cos/sin; measured dispatch line is
    # legacy below 128 tokens and a TPP ladder keyed on >=24 programs above.
    n_tokens = qkv.shape[0]
    use_tile_arm = (
        qkv.device.type == "gcu"
        and not insert_kv
        and rotary_dim in (64, 128)
        and n_tokens >= 128
    )
    if use_tile_arm:
        if n_tokens >= 1536:
            tokens_per_prog = 64
        elif n_tokens >= 768:
            tokens_per_prog = 32
        elif n_tokens >= 384:
            tokens_per_prog = 16
        else:
            tokens_per_prog = 8
        scratch = torch.empty(
            (n_tokens, rotary_dim), device=qkv.device, dtype=qkv.dtype
        )
        h_pad = triton.next_power_of_2(packed_heads)
        grid = (triton.cdiv(n_tokens, tokens_per_prog),)
        with torch_device_fn.device(qkv.device):
            _fused_minimax_m3_tile_kernel[grid](
                qkv,
                q_norm_weight,
                k_norm_weight,
                index_q_norm_weight if has_index else q_norm_weight,
                index_k_norm_weight if has_index else k_norm_weight,
                cos_sin_cache,
                positions,
                scratch,
                q_out,
                index_q_out,
                n_tokens,
                num_heads,
                num_kv_heads,
                num_index_heads,
                rotary_dim,
                float(eps),
                has_index,
                q_out is not None,
                index_q_out is not None,
                tokens_per_prog,
                h_pad,
                num_warps=1,
                enable_fp_fusion=False,
            )
        return
    tokens_per_prog = max(1, min(256, (qkv.shape[0] * packed_heads) // 512))
    tokens_per_prog = triton.next_power_of_2(tokens_per_prog)
    grid = (triton.cdiv(qkv.shape[0], tokens_per_prog), packed_heads)
    with torch_device_fn.device(qkv.device):
        _fused_minimax_m3_kernel[grid](
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
            qkv.shape[0],
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
            tokens_per_prog,
            # A single warp keeps the 128-wide row reduction co-resident.
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

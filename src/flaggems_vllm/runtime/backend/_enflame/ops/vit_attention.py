# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Noncausal ViT attention through FlagGems, without vLLM extensions.

Import this module before tracing/capture so its custom ops and fake kernels
are registered. Sequence boundaries remain device inputs on graph replay.
"""

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

# Flash-tile hyperparameters (GCU400, 2026-11-11 sweep): one program per
# (query tile, head); BLOCK_N keys per inner-loop step. nw4 beats nw1/nw2 on
# every core shape here (0.014/0.076/0.055 ms vs 0.040/0.101/0.062 ms at nw2)
# — the sglang context_attention nw2 sweet spot does not transfer to this
# bf16/fp16 short-sequence shape family (warp sweet spots are per-variant).
BLOCK_M = 64
BLOCK_N = 128
NUM_WARPS = 4


@libentry()
@triton.jit
def _pack_kernel(
    X,
    Y,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SB: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SD: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = i // (H * D)
    offset = token // S * SB + token % S * ST + (i // D % H) * SH + i % D * SD
    if MODE == 0:
        value = tl.full((BLOCK,), 0.0, tl.float32)
    else:
        value = tl.load(X + offset, mask=i < N, other=0)
        if MODE == -1:
            value = -value
    tl.store(Y + i, value, mask=i < N)


def _pack(x, mode=1):
    tokens = math.prod(x.shape[:-2])
    output_shape = (tokens, x.shape[-2], x.shape[-1])
    if x.is_contiguous() and mode == 1:
        return x.view(output_shape)
    y = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if tokens:
        if x.ndim == 3:
            sequence, sb, st, sh, sd = x.shape[0], 0, *x.stride()
        else:
            sequence = x.shape[1]
            sb, st, sh, sd = x.stride()
        with torch_device_fn.device(x.device):
            _pack_kernel[(triton.cdiv(y.numel(), 1024),)](
                x,
                y,
                y.numel(),
                sequence,
                x.shape[-2],
                x.shape[-1],
                sb,
                st,
                sh,
                sd,
                mode,
                BLOCK=1024,
            )
    return y


def supports_vit_attention(head_size, dtype):
    """Pure metadata gate for the dimensions validated by this adapter."""
    return head_size in (64, 80) and dtype in (torch.float16, torch.bfloat16)


def _check_qkv(q, k, v, dimensions):
    # torch_gcu reports GCU tensors as is_cuda=True, but device.type is the
    # vendor name ("gcu"); gate on the active backend device instead of "cuda".
    if q.device.type != runtime.device.name:
        raise NotImplementedError("ViT attention requires a CUDA device")
    if q.ndim != dimensions or k.ndim != dimensions or v.ndim != dimensions:
        raise ValueError(f"Q/K/V must all have {dimensions} dimensions")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("ViT FlagGems attention requires FP16 or BF16 Q/K/V")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V devices must match")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("K/V shapes and Q/K head dimensions must match")
    if not supports_vit_attention(q.shape[-1], q.dtype):
        raise ValueError("ViT adapter supports validated head dimensions 64/80")
    if k.shape[-2] <= 0 or q.shape[-2] % k.shape[-2]:
        raise ValueError("KV heads must divide Q heads")


def _check_cu(cu_q, cu_k, device):
    if cu_q.ndim != 1 or cu_k.ndim != 1 or cu_q.shape != cu_k.shape:
        raise ValueError("Q/K cumulative lengths must be equal-sized vectors")
    if cu_q.numel() < 2:
        raise ValueError("Cumulative lengths must include a start and end")
    if cu_q.dtype != torch.int32 or cu_k.dtype != torch.int32:
        raise ValueError("Cumulative lengths must be int32")
    if cu_q.device != device or cu_k.device != device:
        raise ValueError("Cumulative lengths must be on the Q/K/V device")
    if not cu_q.is_contiguous() or not cu_k.is_contiguous():
        raise ValueError("Cumulative lengths must be contiguous")


@libentry()
@triton.jit
def _vit_attn_flash_kernel(
    Q,
    K,
    V,
    O,
    CU_Q,
    CU_K,
    sm_scale,
    T,
    q_len,
    k_len,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT_D: tl.constexpr,
    UNIFORM: tl.constexpr,
):
    # Flash-attention tile kernel: one program per (query tile, head). Q stays
    # resident while the inner loop scans K/V in BLOCK_N tiles through
    # tl.dot; this is the sglang context_attention recipe (VPRE load of K and
    # V before the QK dot, arithmetic masking, acc-fused dot rescale, one
    # canonical tile loop) adapted to ViT semantics: bf16/fp16, GQA,
    # device-resident cu_seqlens (graph-replay safe), per-row K ranges, and
    # empty-K rows produce zero output. Replaces the earlier pure-1D
    # row-streaming kernel whose per-token scalar online-softmax serial chain
    # and strided K/V row reads cost 150-1000x vs this tile form.
    m_tile = tl.program_id(0)
    head = tl.program_id(1)
    m0 = m_tile * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mmask = offs_m < T

    # Sequence lookup. UNIFORM covers the batched 4-D entry without packed
    # cu_seqlens: every sequence has the host-static per-row length, so the
    # boundaries derive arithmetically (graph-replay safe, and no separate
    # boundary-fill kernel launch on the host path). Otherwise the scan reads
    # the device-resident boundaries and stays correct when cu_seqlens values
    # change on CUDA graph replay. NUM_SEQS is constexpr so the scan unrolls;
    # tokens past T clamp to the last sequence so the CU_K gathers stay in
    # bounds (their rows are masked at the store).
    if UNIFORM:
        seq_vec = tl.minimum(offs_m // q_len, NUM_SEQS - 1)
        k_start_vec = seq_vec * k_len
        k_end_vec = k_start_vec + k_len
    else:
        seq_vec = tl.zeros([BLOCK_M], dtype=tl.int32)
        for s in range(1, NUM_SEQS + 1):
            bound = tl.load(CU_Q + s)
            seq_vec += (bound <= offs_m).to(tl.int32)
        seq_vec = tl.minimum(seq_vec, NUM_SEQS - 1)
        k_start_vec = tl.load(CU_K + seq_vec)
        k_end_vec = tl.load(CU_K + seq_vec + 1)

    # Tile-wide K scan bounds: the union of the per-row ranges. Rows outside
    # their own range are masked arithmetically inside the loop, so a tile
    # spanning a sequence boundary stays exact (extra keys underflow to 0).
    k_lo = tl.min(tl.where(mmask, k_start_vec, 1 << 30))
    k_hi = tl.max(tl.where(mmask, k_end_vec, -(1 << 30)))

    kv_head = head // GROUPS
    q_ptrs = Q + offs_m[:, None] * (H_Q * D) + head * D + offs_d[None, :]
    if EXACT_D:
        q_tile = tl.load(q_ptrs, mask=mmask[:, None], other=0.0)
    else:
        dmask = offs_d < D
        q_tile = tl.load(q_ptrs, mask=mmask[:, None] & dmask[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], -1e9, tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for n_start in range(k_lo, k_hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        nmask = offs_n < k_hi
        kv_off = offs_n[:, None] * (H_KV * D) + kv_head * D
        if EXACT_D:
            # K and V are loaded together before the QK dot: the dot covers
            # both DMA latencies (context_attention VPRE, +6-9% there).
            k_tile = tl.load(
                K + kv_off + offs_d[None, :], mask=nmask[:, None], other=0.0
            )
            v_tile = tl.load(
                V + kv_off + offs_d[None, :], mask=nmask[:, None], other=0.0
            )
        else:
            k_tile = tl.load(
                K + kv_off + offs_d[None, :],
                mask=nmask[:, None] & dmask[None, :],
                other=0.0,
            )
            v_tile = tl.load(
                V + kv_off + offs_d[None, :],
                mask=nmask[:, None] & dmask[None, :],
                other=0.0,
            )

        qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale

        # Per-row key-range mask, applied arithmetically: valid lanes keep
        # their score (x*1 + 0), invalid lanes become exactly -1e9. The
        # select form between the dot and the softmax reduction costs ~40%
        # of the kernel on this backend (context_attention measurement).
        okf = (
            (offs_n[None, :] >= k_start_vec[:, None])
            & (offs_n[None, :] < k_end_vec[:, None])
        ).to(tl.float32)
        qk = qk * okf + (okf - 1.0) * 1e9

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp(m_i - m_new)
        p = tl.math.exp(qk - m_new[:, None])
        # Rows whose whole range is empty keep m_new == m_i == -1e9, for
        # which exp(0) would be 1; scale by the mask so they stay 0.
        p = p * okf
        # The rescale feeds the dot accumulator so the add fuses into the
        # MMA epilogue (serial-recurrence acc-fusion pattern).
        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc=acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Empty K ranges yield zero outputs (l_i == 0 there), matching the
    # packed-varlen contract.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    o_ptrs = O + offs_m[:, None] * (H_Q * D) + head * D + offs_d[None, :]
    if EXACT_D:
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=mmask[:, None])
    else:
        tl.store(
            o_ptrs, out.to(O.dtype.element_ty), mask=mmask[:, None] & dmask[None, :]
        )


def _flash_attn(q, k, v, cu_q, cu_k, scale, uniform_q_len=0, uniform_k_len=0):
    # q: [total_q, H_Q, D] and k/v: [total_k, H_KV, D], contiguous packed rows.
    # uniform_q_len > 0 selects the equal-length batch layout: the K boundaries
    # are (token // q_len) * k_len arithmetic, so cu_q/cu_k are unused pointer
    # placeholders and no boundary-fill kernel is needed.
    total_q = q.shape[0]
    if total_q == 0:
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)
    num_heads, head_dim = q.shape[1], q.shape[2]
    total_k = k.shape[0]
    # 32-bit affine offsets are the fast addressing path on this backend;
    # larger token volumes are outside the adapter's envelope.
    if (
        total_q * num_heads * head_dim >= 2**31
        or total_k * k.shape[1] * head_dim >= 2**31
    ):
        raise NotImplementedError(
            "ViT flash attention supports at most 2^31 elements per packed "
            "Q/KV tensor on 32-bit addressing"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_d = triton.next_power_of_2(head_dim)
    uniform = uniform_q_len > 0
    _vit_attn_flash_kernel[(triton.cdiv(total_q, BLOCK_M), num_heads)](
        q,
        k,
        v,
        out,
        q if uniform else cu_q,
        k if uniform else cu_k,
        scale,
        total_q,
        uniform_q_len,
        uniform_k_len,
        H_Q=num_heads,
        H_KV=k.shape[1],
        GROUPS=num_heads // k.shape[1],
        D=head_dim,
        BLOCK_D=block_d,
        NUM_SEQS=cu_q.numel() - 1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        EXACT_D=(block_d == head_dim),
        UNIFORM=uniform,
        num_warps=NUM_WARPS,
    )
    return out


def _varlen_impl(q, k, v, cu_q, cu_k, scale, uniform_q_len=0, uniform_k_len=0):
    _check_qkv(q, k, v, 3)
    if not uniform_q_len:
        _check_cu(cu_q, cu_k, q.device)
    if scale is not None and not math.isfinite(scale):
        raise ValueError("Attention scale must be finite")
    if q.shape[0] == 0:
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)

    # Normalize to contiguous packed [tokens, heads, dim] rows. Strided inputs
    # go through the pure-1D _pack_kernel; contiguous ones are free views.
    q, k, v = _pack(q), _pack(k), _pack(v)
    # The flash-tile kernel applies the scale before the softmax exponent,
    # so zero and negative finite scales need no input transformation. An
    # empty K range yields zero outputs, matching the packed-varlen contract.
    return _flash_attn(q, k, v, cu_q, cu_k, scale, uniform_q_len, uniform_k_len)


@torch.library.custom_op("flaggems_vllm::vision_flash_attn_varlen", mutates_args=())
def _vision_flash_attn_varlen_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float | None,
) -> torch.Tensor:
    return _varlen_impl(q, k, v, cu_seqlens_q, cu_seqlens_k, scale)


@_vision_flash_attn_varlen_op.register_fake
def _vision_flash_attn_varlen_fake(q, k, v, cu_seqlens_q, cu_seqlens_k, scale):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def vision_flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, scale=None):
    """Packed [tokens, heads, dim] attention with distinct Q/K boundaries."""
    return _vision_flash_attn_varlen_op(q, k, v, cu_seqlens_q, cu_seqlens_k, scale)


@torch.library.custom_op("flaggems_vllm::vit_flash_attn_wrapper", mutates_args=())
def _vit_flash_attn_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: torch.Tensor | None,
) -> torch.Tensor:
    _check_qkv(q, k, v, 4)
    if q.shape[0] != batch_size or k.shape[0] != batch_size:
        raise ValueError("Q/K batch dimensions must equal batch_size")
    if max_seqlen is not None and max_seqlen.numel() != 1:
        raise ValueError("max_seqlen must be a scalar tensor")

    total_q = q.shape[0] * q.shape[1]
    total_k = k.shape[0] * k.shape[1]
    if total_q == 0:
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if cu_seqlens is None:
        # Equal-length batch rows: the flash kernel derives the sequence
        # boundaries from the host-static per-row lengths, so no boundary
        # tensors are materialized and no fill kernel is launched.
        cu_q = cu_k = q
        uniform_q_len = q.shape[1]
        uniform_k_len = k.shape[1]
    else:
        if total_q != total_k:
            raise ValueError(
                "Shared cu_seqlens requires equal total Q/K tokens; use "
                "vision_flash_attn_varlen for packed cross attention"
            )
        cu_q = cu_k = cu_seqlens
        uniform_q_len = uniform_k_len = 0

    q_flat, k_flat, v_flat = _pack(q), _pack(k), _pack(v)
    output = _varlen_impl(
        q_flat, k_flat, v_flat, cu_q, cu_k, scale, uniform_q_len, uniform_k_len
    )
    # is_rocm_aiter/fa_version select upstream compiled libraries. This
    # callable explicitly selects the existing FlagGems Triton implementation.
    return output.view(q.shape)


@_vit_flash_attn_op.register_fake
def _vit_flash_attn_fake(
    q, k, v, batch_size, is_rocm_aiter, fa_version, scale, cu_seqlens, max_seqlen
):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def vit_flash_attn_wrapper(
    q,
    k,
    v,
    batch_size,
    is_rocm_aiter=False,
    fa_version=None,
    scale=None,
    cu_seqlens=None,
    max_seqlen=None,
):
    """vLLM-compatible [batch, tokens, heads, dim] ViT attention callable.

    The optional shared cumulative lengths describe packed self attention.
    Without them, each batch row is independent, including different Q/K
    sequence lengths. max_seqlen is accepted for API compatibility; the launch
    uses safe shape upper bounds, so it never reads a tensor scalar on the host.
    """
    return _vit_flash_attn_op(
        q,
        k,
        v,
        batch_size,
        is_rocm_aiter,
        fa_version,
        scale,
        cu_seqlens,
        max_seqlen,
    )

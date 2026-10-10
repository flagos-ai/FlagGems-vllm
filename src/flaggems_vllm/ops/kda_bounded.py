# SPDX-License-Identifier: Apache-2.0
"""Bounded KDA with original 64-token prefill and vLLM V,K decode state."""
import math
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.data_movement import contiguous_copy, fill
from flaggems_vllm.ops.kda64 import (
    _prefill_core,
    _recurrent_core,
    prefill_kda64,
    recurrent_kda_vk,
)
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(configs=runtime.get_tuned_config("kda_safe_gate"), key=["T", "H", "D"])
@triton.jit
def _gate_kernel(
    G,
    A,
    BIAS,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    LOWER: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < T * H * D
    h = (i // D) % H
    g = tl.load(G + i, mask, other=0).to(tl.float32)
    if HAS_BIAS:
        g += tl.load(BIAS + i % (H * D), mask, other=0).to(tl.float32)
    a = tl.exp(tl.load(A + h, mask, other=0).to(tl.float32))
    value = LOWER / (1.0 + tl.exp(-(a * g)))
    tl.store(OUT + i, value, mask)


def safe_kda_gate(g, A_log, head_k_dim, g_bias=None, lower_bound=-5.0):
    heads = A_log.numel()
    if (
        g.ndim < 1
        or heads < 1
        or head_k_dim < 1
        or g.shape[-1] != heads * head_k_dim
        or not math.isfinite(lower_bound)
        or lower_bound >= 0
    ):
        raise ValueError("Invalid KDA gate hidden dimension or lower bound")
    if g_bias is not None and g_bias.numel() != heads * head_k_dim:
        raise ValueError("KDA gate bias must cover every head/key dimension")
    if A_log.device != g.device or (g_bias is not None and g_bias.device != g.device):
        raise ValueError("KDA gate parameters must share input device")
    if any(
        x is not None and x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        for x in (g, A_log, g_bias)
    ):
        raise ValueError("KDA gate parameters must be floating tensors")
    g, A_log = map(contiguous_copy, (g, A_log))
    if g_bias is not None:
        g_bias = contiguous_copy(g_bias)
    out = torch.empty(
        (*g.shape[:-1], heads, head_k_dim), dtype=torch.float32, device=g.device
    )
    if out.numel():
        _gate_kernel[lambda m: (triton.cdiv(out.numel(), m["BLOCK"]),)](
            g,
            A_log,
            g_bias,
            out,
            g.numel() // (heads * head_k_dim),
            heads,
            head_k_dim,
            lower_bound,
            g_bias is not None,
        )
    return out


@libentry()
@triton.jit
def _recurrent_kernel(
    Q,
    K,
    V,
    G,
    BETA,
    H0,
    HT,
    OUT,
    CU,
    SLOTS,
    T: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_H0: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_SLOTS: tl.constexpr,
    INPLACE: tl.constexpr,
    NORM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    seq, head = tl.program_id(0), tl.program_id(1)
    start = tl.load(CU + seq) if HAS_CU else seq * T
    end = tl.load(CU + seq + 1) if HAS_CU else start + T
    slot = tl.load(SLOTS + seq) if HAS_SLOTS else seq
    ik, iv = tl.arange(0, BK), tl.arange(0, BV)
    mask = (ik[:, None] < DK) & (iv[None, :] < DV)
    if HAS_H0:
        state = tl.load(
            H0 + (slot * H + head) * DK * DV + ik[:, None] * DV + iv[None, :],
            mask & (slot >= 0),
            other=0,
        ).to(tl.float32)
    else:
        state = tl.zeros((BK, BV), tl.float32)
    for token in range(start, end):
        offset = (token * H + head) * DK + ik
        q = tl.load(Q + offset, ik < DK, other=0).to(tl.float32)
        k = tl.load(K + offset, ik < DK, other=0).to(tl.float32)
        if NORM:
            q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6)
            k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
            q = q.to(Q.dtype.element_ty).to(tl.float32)
            k = k.to(K.dtype.element_ty).to(tl.float32)
        g = tl.load(G + offset, ik < DK, other=0).to(tl.float32)
        v = tl.load(V + (token * H + head) * DV + iv, iv < DV, other=0).to(tl.float32)
        beta = tl.load(BETA + token * H + head).to(tl.float32)
        state *= tl.exp(g)[:, None]
        residual = v - tl.sum(k[:, None] * state, 0)
        state += (beta * k)[:, None] * residual[None, :]
        output = tl.sum((q * SCALE)[:, None] * state, 0)
        tl.store(
            OUT + (token * H + head) * DV + iv, tl.where(slot >= 0, output, 0), iv < DV
        )
    target = slot if INPLACE else seq
    tl.store(
        HT + (target * H + head) * DK * DV + ik[:, None] * DV + iv[None, :],
        state,
        mask & (slot >= 0),
    )


def recurrent_kda(
    q,
    k,
    v,
    gate,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=True,
    inplace_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    ssm_state_indices=None,
):
    """Portable sequential helper with explicit [N,H,K,V] state layout."""
    if q.ndim != 4 or k.shape != q.shape or gate.shape != q.shape:
        raise ValueError("KDA q/k/g must share [B,T,H,K] shape")
    b, t, h, dk = q.shape
    if v.ndim != 4 or v.shape[:3] != (b, t, h) or beta.shape != (b, t, h):
        raise ValueError("KDA value/beta shape mismatch")
    dv = v.shape[-1]
    if dk < 1 or dv < 1 or dk > 256 or dv > 256:
        raise NotImplementedError("KDA recurrent supports positive K,V <=256")
    if cu_seqlens is not None and (b != 1 or cu_seqlens.ndim != 1):
        raise ValueError("KDA varlen expects flattened batch of one")
    n = b if cu_seqlens is None else cu_seqlens.numel() - 1
    if initial_state is not None and (
        initial_state.ndim != 4
        or initial_state.shape[1:] != (h, dk, dv)
        or initial_state.dtype != torch.float32
    ):
        raise ValueError("KDA state must be float32 [slots,H,K,V]")
    if ssm_state_indices is not None and (
        ssm_state_indices.shape != (n,) or initial_state is None
    ):
        raise NotImplementedError("KDA supports one state slot per request")
    all_tensors = [q, k, v, gate, beta, initial_state, cu_seqlens, ssm_state_indices]
    if any(x is not None and x.device != q.device for x in all_tensors):
        raise ValueError("KDA inputs and state must share device")
    q, k, v, gate, beta = map(contiguous_copy, (q, k, v, gate, beta))
    if initial_state is not None and not initial_state.is_contiguous():
        raise ValueError("KDA state must be contiguous")
    inplace = inplace_final_state and initial_state is not None
    final = (
        initial_state
        if inplace
        else torch.empty((n, h, dk, dv), dtype=torch.float32, device=q.device)
    )
    out = torch.empty(v.shape, dtype=v.dtype, device=q.device)
    if n:
        _recurrent_kernel[(n, h)](
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            final,
            out,
            cu_seqlens,
            ssm_state_indices,
            t,
            h,
            dk,
            dv,
            dk**-0.5 if scale is None else scale,
            initial_state is not None,
            cu_seqlens is not None,
            ssm_state_indices is not None,
            inplace,
            use_qk_l2norm_in_kernel,
            BK=triton.next_power_of_2(dk),
            BV=triton.next_power_of_2(dv),
            num_warps=4,
        )
    return out, final if output_final_state else None


def _validate_kda(
    q,
    k,
    v,
    gate,
    beta,
    initial_state,
    cu_seqlens,
    ssm_state_indices=None,
    prefill=False,
):
    if q.ndim != 4 or k.shape != q.shape or gate.shape != q.shape:
        raise ValueError("KDA q/k/g must share [B,T,H,K] shape")
    b, t, h, dk = q.shape
    if v.ndim != 4 or v.shape[:3] != (b, t, h) or beta.shape != (b, t, h):
        raise ValueError("KDA value/beta shape mismatch")
    dv = v.shape[-1]
    if h < 1 or not 1 <= dk <= 256 or not 1 <= dv <= 256:
        raise NotImplementedError("KDA supports positive heads and K,V <=256")
    floating = (torch.float16, torch.bfloat16, torch.float32)
    if q.dtype not in floating or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("KDA q/k/v require a shared fp16/bf16/fp32 dtype")
    if gate.dtype not in floating or beta.dtype not in floating:
        raise ValueError("KDA gate and beta must be floating tensors")
    if cu_seqlens is not None and (
        b != 1
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 1
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("KDA varlen requires contiguous integer CU and batch one")
    n = b if cu_seqlens is None else cu_seqlens.numel() - 1
    if initial_state is not None and (
        initial_state.ndim != 4
        or initial_state.shape[1:] != (h, dv, dk)
        or initial_state.dtype != torch.float32
        or (prefill and initial_state.shape[0] != n)
    ):
        raise ValueError("KDA state must be float32 [slots,H,V,K]")
    if (
        initial_state is not None
        and not prefill
        and (
            initial_state.stride()[1:] != (dv * dk, dk, 1)
            or initial_state.stride(0) < h * dv * dk
        )
    ):
        raise ValueError(
            "KDA decode state requires dense H,V,K and nonoverlapping slots"
        )
    if ssm_state_indices is not None:
        if (
            initial_state is None
            or ssm_state_indices.dtype not in (torch.int32, torch.int64)
            or not ssm_state_indices.is_contiguous()
            or ssm_state_indices.ndim not in (1, 2)
            or ssm_state_indices.shape[0] != n
        ):
            raise ValueError(
                "KDA decode requires contiguous integer state slots per request"
            )
        if ssm_state_indices.ndim == 1:
            # The original kernel reads an index for every token. The caller
            # owns CU values and must provide one token per request for 1D
            # slots. T == N is necessary, but does not inspect/prove CU values.
            if (cu_seqlens is None and t > 1) or (cu_seqlens is not None and t != n):
                raise ValueError(
                    "Multi-token KDA requires [requests,max_tokens] state slots"
                )
        elif ssm_state_indices.shape[1] < (
            t if cu_seqlens is None else triton.cdiv(t, max(n, 1))
        ):
            raise ValueError("KDA state-slot columns cannot cover the supplied tokens")
        # For 2D varlen slots the caller additionally guarantees that every CU
        # interval fits max_tokens and that indices refer to valid cache slots.
    if (
        initial_state is not None
        and ssm_state_indices is None
        and initial_state.shape[0] < n
    ):
        raise ValueError("KDA initial state does not cover every request")
    tensors = (q, k, v, gate, beta, initial_state, cu_seqlens, ssm_state_indices)
    if any(x is not None and x.device != q.device for x in tensors):
        raise ValueError("KDA inputs, metadata, and state must share device")
    return n


def fused_recurrent_kda(
    q,
    k,
    v,
    g,
    beta=None,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    **kwargs,
):
    """vLLM-compatible decode with live float32 [slots,H,V,K] state."""
    if beta is None:
        raise ValueError("KDA beta tensor is required")
    if kwargs:
        raise NotImplementedError(f"Unsupported KDA decode arguments: {tuple(kwargs)}")
    n = _validate_kda(q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices)
    if initial_state is None:
        raise ValueError("vLLM KDA decode requires an initial-state buffer")
    if q.shape[1] == 0 or n == 0:
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        final = (
            initial_state
            if inplace_final_state
            else torch.empty(
                (q.shape[1], *initial_state.shape[1:]),
                dtype=initial_state.dtype,
                device=q.device,
            )
        )
        return out, final
    if inplace_final_state and ssm_state_indices is None:
        raise ValueError("Inplace vLLM KDA decode requires an explicit state-slot map")
    if not inplace_final_state and q.shape[0] != 1:
        raise NotImplementedError(
            "Non-inplace vLLM KDA state requires flattened batch one"
        )
    kernel = _recurrent_core()
    q, k, v, g, beta = map(contiguous_copy, (q, k, v, g, beta))
    return recurrent_kda_vk(
        q,
        k,
        v,
        g,
        beta,
        q.shape[-1] ** -0.5 if scale is None else scale,
        initial_state,
        inplace_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        ssm_state_indices,
        kernel,
    )


def chunk_kda_with_safe_gate(
    q,
    k,
    v,
    raw_g,
    beta,
    A_log,
    g_bias=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    lower_bound=-5.0,
    chunk_indices=None,
    chunk_offsets=None,
):
    """Original bounded-gate prefill: fixed 64-token chunks and V,K state.

    Beta is already activated. The existing 16-token fused KDA implementation
    has different intermediate materializations and is not used here.
    """
    n = _validate_kda(q, k, v, raw_g, beta, initial_state, cu_seqlens, prefill=True)
    _validate_gate(raw_g, A_log, g_bias, lower_bound)
    if cu_seqlens is not None:
        if chunk_indices is None or chunk_offsets is None:
            raise ValueError("Varlen KDA requires framework-prepared 64-token metadata")
        if (
            chunk_indices.ndim != 2
            or chunk_indices.shape[1] != 2
            or chunk_offsets.shape != (n + 1,)
            or any(
                x.device != q.device
                or x.dtype not in (torch.int32, torch.int64)
                or not x.is_contiguous()
                for x in (chunk_indices, chunk_offsets)
            )
        ):
            raise ValueError(
                "KDA chunk metadata must be contiguous integers on input device"
            )
    if q.shape[1] == 0 or n == 0:
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        if not output_final_state:
            return out, None
        state = (
            initial_state
            if initial_state is not None
            else fill(
                torch.empty(
                    (n, q.shape[2], v.shape[3], q.shape[3]),
                    dtype=torch.float32,
                    device=q.device,
                ),
                0,
            )
        )
        return out, state
    core = _prefill_core()
    q, k, v, raw_g, beta = map(contiguous_copy, (q, k, v, raw_g, beta))
    if use_qk_l2norm_in_kernel:
        q, k = core[1](q), core[1](k)
    gate = safe_kda_gate_chunk_cumsum(
        raw_g,
        A_log,
        g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=64,
        lower_bound=lower_bound,
    )
    if initial_state is not None:
        # Match the original prefill host's contiguous state materialization;
        # decode instead preserves padded cache-slot stride and aliases.
        initial_state = contiguous_copy(initial_state)
    return prefill_kda64(
        q,
        k,
        v,
        gate,
        beta,
        q.shape[-1] ** -0.5 if scale is None else scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        core,
    )


@libentry()
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=runtime.get_tuned_config("kda_safe_gate_cumsum"),
    key=["H", "D", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def _safe_gate_cumsum_kernel(
    g,
    A,
    y,
    g_bias,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    lower_bound: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    p_y = tl.make_block_ptr(
        y + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )

    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        o_d = i_d * BD + tl.arange(0, BD)
        b_bias = tl.load(g_bias + i_h * D + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        b_g += b_bias[None, :]

    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))
    b_gate = lower_bound / (1.0 + tl.exp(-(b_a * b_g)))

    # Chunk-local inclusive cumsum, stored in log2 units for the downstream
    # exp2-based KDA core.
    o_t = tl.arange(0, BT)
    m_cumsum = tl.where(o_t[:, None] >= o_t[None, :], 1.0, 0.0)
    b_y = tl.dot(m_cumsum, b_gate, allow_tf32=False) * cumsum_scale
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def _validate_gate(raw_g, A_log, g_bias, lower_bound):
    if raw_g.ndim != 4 or raw_g.shape[2] < 1 or raw_g.shape[3] < 1:
        raise ValueError("KDA cumulative gate requires [B,T,H,D] with positive H,D")
    h, d = raw_g.shape[2:]
    floating = (torch.float16, torch.bfloat16, torch.float32)
    if (
        A_log.numel() != h
        or (g_bias is not None and g_bias.numel() != h * d)
        or not math.isfinite(lower_bound)
        or lower_bound >= 0
    ):
        raise ValueError("KDA gate parameter shape or lower bound is invalid")
    if any(
        x is not None and (x.device != raw_g.device or x.dtype not in floating)
        for x in (raw_g, A_log, g_bias)
    ):
        raise ValueError("KDA gate parameters require floating tensors on input device")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("kda_safe_gate_cumsum"),
    key=["T", "H", "D", "CHUNK", "VARLEN"],
)
@triton.jit
def _gate_sequential_cumsum_kernel(
    G,
    A,
    BIAS,
    CU,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    LOWER: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    VARLEN: tl.constexpr,
    BD: tl.constexpr,
):
    seq, head, dblock = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(CU + seq) if VARLEN else seq * T
    end = tl.load(CU + seq + 1) if VARLEN else start + T
    d = dblock * BD + tl.arange(0, BD)
    a = tl.exp(tl.load(A + head).to(tl.float32))
    bias = (
        tl.load(BIAS + head * D + d, d < D, other=0).to(tl.float32)
        if HAS_BIAS
        else tl.full((BD,), 0, tl.float32)
    )
    for chunk_start in range(start, end, CHUNK):
        accum = tl.full((BD,), 0, tl.float32)
        for token in range(chunk_start, tl.minimum(chunk_start + CHUNK, end)):
            offset = (token * H + head) * D + d
            raw = tl.load(G + offset, d < D, other=0).to(tl.float32)
            gate = LOWER / (1.0 + tl.exp(-(a * (raw + bias))))
            accum += gate
            tl.store(OUT + offset, accum * 1.4426950408889634, d < D)


def safe_kda_gate_chunk_cumsum(
    raw_g,
    A_log,
    g_bias=None,
    cu_seqlens=None,
    chunk_size=64,
    output_dtype=torch.float32,
    lower_bound=-5.0,
    *,
    chunk_indices=None,
):
    """Inclusive log2 cumsum; GLM 64-token chunks retain original tl.dot order."""
    _validate_gate(raw_g, A_log, g_bias, lower_bound)
    if chunk_size < 1:
        raise ValueError("KDA cumulative gate requires a positive chunk size")
    dot_cumsum = chunk_size in (16, 32, 64, 128)
    if output_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("KDA cumulative gate output must be floating")
    b, t, h, d = raw_g.shape
    if cu_seqlens is not None and (
        b != 1
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 1
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or cu_seqlens.device != raw_g.device
        or not cu_seqlens.is_contiguous()
        or (
            dot_cumsum
            and (
                chunk_indices is None
                or chunk_indices.ndim != 2
                or chunk_indices.shape[1] != 2
                or chunk_indices.dtype not in (torch.int32, torch.int64)
                or chunk_indices.device != raw_g.device
                or not chunk_indices.is_contiguous()
            )
        )
    ):
        raise ValueError(
            "KDA cumulative gate requires prepared integer varlen metadata"
        )
    n = b if cu_seqlens is None else cu_seqlens.numel() - 1
    nt = (
        triton.cdiv(t, chunk_size)
        if cu_seqlens is None
        else (chunk_indices.shape[0] if dot_cumsum else n)
    )
    out = torch.empty(
        raw_g.shape, dtype=output_dtype or raw_g.dtype, device=raw_g.device
    )
    if out.numel() and nt:
        raw_g, A_log = map(contiguous_copy, (raw_g, A_log))
        if g_bias is not None:
            g_bias = contiguous_copy(g_bias)
        if not dot_cumsum:
            _gate_sequential_cumsum_kernel[
                lambda meta: (n, h, triton.cdiv(d, meta["BD"]))
            ](
                raw_g,
                A_log,
                g_bias,
                cu_seqlens,
                out,
                t,
                h,
                d,
                chunk_size,
                lower_bound,
                g_bias is not None,
                cu_seqlens is not None,
            )
            return out
        _safe_gate_cumsum_kernel[lambda meta: (triton.cdiv(d, meta["BD"]), nt, b * h)](
            g=raw_g,
            A=A_log,
            y=out,
            g_bias=g_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            cumsum_scale=1.4426950408889634,
            lower_bound=lower_bound,
            T=t,
            H=h,
            D=d,
            BT=chunk_size,
        )
    return out


@lru_cache(maxsize=1)
def _prefill_available():
    try:
        _prefill_core()
        return True
    except (ImportError, AttributeError, NotImplementedError, OSError):
        return False


@lru_cache(maxsize=1)
def _recurrent_available():
    try:
        _recurrent_core()
        return True
    except (ImportError, AttributeError, NotImplementedError, OSError):
        return False


# Public capability hooks run only the cached ABI resolvers. They do not
# submit a kernel, inspect real model state, or catch execution failures.
chunk_kda_with_safe_gate._is_available = _prefill_available
fused_recurrent_kda._is_available = _recurrent_available

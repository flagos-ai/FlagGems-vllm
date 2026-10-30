# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Retains the recurrent formulation from flash-linear-attention (MIT):
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""Stride-aware recurrent GDN with vLLM's value-major state cache."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent_gdn_kernel(
    Q,
    Kp,
    Vp,
    G,
    Beta,
    State,
    Out,
    Final,
    Cu,
    Indices,
    Accepted,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    POOL: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    GS: tl.constexpr,
    BS: tl.constexpr,
    SS: tl.constexpr,
    IX: tl.constexpr,
    WIDTH: tl.constexpr,
    VARLEN: tl.constexpr,
    INDEXED: tl.constexpr,
    TABLE: tl.constexpr,
    SPEC: tl.constexpr,
    INITIAL: tl.constexpr,
    INPLACE: tl.constexpr,
    VECTOR_BETA: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_G: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    iv = pid % nv
    nh = pid // nv
    n, hv = nh // HV, nh % HV
    h = hv // (HV // H)
    if VARLEN:
        start = tl.load(Cu + n).to(tl.int64)
        end = tl.load(Cu + n + 1).to(tl.int64)
        batch = 0
    else:
        start, end = 0, T
        batch = n
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mask = (rv[:, None] < V) & (rk[None, :] < K)
    slot = n.to(tl.int64)
    valid = True
    if INDEXED:
        col = 0
        if SPEC:
            col = tl.load(Accepted + n).to(tl.int64) - 1
        valid = (col >= 0) & (col < WIDTH)
        slot = tl.load(Indices + n * IX[0] + col * IX[1], mask=valid, other=0).to(
            tl.int64
        )
        valid = valid & (slot > 0) & (slot < POOL)
    state_offsets = hv * SS[1] + rv[:, None] * SS[2] + rk[None, :] * SS[3]
    acc = tl.full((BV, BK), 0, tl.float32)
    if INITIAL:
        acc = tl.load(
            State + slot * SS[0] + state_offsets, mask=mask & valid, other=0
        ).to(tl.float32)
    for t in range(start, end):
        q = tl.load(
            Q + batch * QS[0] + t * QS[1] + h * QS[2] + rk * QS[3], rk < K, 0
        ).to(tl.float32)
        k = tl.load(
            Kp + batch * KS[0] + t * KS[1] + h * KS[2] + rk * KS[3], rk < K, 0
        ).to(tl.float32)
        v = tl.load(
            Vp + batch * VS[0] + t * VS[1] + hv * VS[2] + rv * VS[3], rv < V, 0
        ).to(tl.float32)
        if NORMALIZE:
            q = q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)
            k = k / tl.sqrt(tl.sum(k * k, 0) + 1.0e-6)
        q *= scale
        if HAS_G:
            g = tl.load(G + batch * GS[0] + t * GS[1] + hv * GS[2]).to(tl.float32)
            acc *= tl.exp(g)
        delta = v - tl.sum(acc * k[None, :], 1)
        if HAS_BETA:
            bp = Beta + batch * BS[0] + t * BS[1] + hv * BS[2]
            if VECTOR_BETA:
                beta = tl.load(bp + rv * BS[3], rv < V, 0).to(tl.float32)
            else:
                beta = tl.load(bp).to(tl.float32)
            delta *= beta
        acc += delta[:, None] * k[None, :]
        out = tl.sum(acc * q[None, :], 1)
        token = batch * T + t
        tl.store(Out + (token * HV + hv) * V + rv, tl.where(valid, out, 0), rv < V)
        if INPLACE:
            dest = slot
            store = valid
            if TABLE:
                final_col = t - start
                dest = tl.load(
                    Indices + n * IX[0] + final_col * IX[1], final_col < WIDTH, 0
                ).to(tl.int64)
                store = store & (dest > 0) & (dest < POOL) & (final_col < WIDTH)
            else:
                store = store & (t == end - 1)
            tl.store(Final + dest * SS[0] + state_offsets, acc, mask & store)
        else:
            offsets = ((token * HV + hv) * V + rv[:, None]) * K + rk[None, :]
            tl.store(Final + offsets, tl.where(valid, acc, 0), mask)


def fused_recurrent_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    """Return output and state in [pool, HV, V, K] layout.

    A 1D index maps each sequence to one cache slot (last-state writeback).
    A 2D index table stores each token's intermediate state, as required by
    speculative decoding; accepted counts select the initial table column.
    Initial slots <=0 or >=pool and invalid accepted columns produce zeros
    without updating the cache. Invalid destination slots suppress that write.
    Writable slots across sequences must be disjoint.
    Without indices, cache slots are sequence-major. Out-of-place mode returns
    every token state as [B*T, HV, V, K] and never modifies initial_state.
    cu_seqlens must be monotone, start at zero and end at T. Index-table width
    must cover each sequence. These device metadata are caller preconditions.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q/k must have identical [B,T,H,K] shapes; v must be rank 4")
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    if v.shape[:2] != (B, T) or min(H, HV, K, V) <= 0 or HV % H:
        raise ValueError("Invalid GVA dimensions")
    if K > 256:
        raise NotImplementedError("Recurrent GDN supports K <= 256")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("Recurrent GDN supports FP16, BF16 and FP32")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must have the same dtype")
    if scale is None:
        scale = K**-0.5
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
            raise ValueError("Packed varlen requires B=1 and cu_seqlens[N+1]")
        if not cu_seqlens.is_contiguous() or cu_seqlens.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("cu_seqlens must be contiguous int32/int64")
        N = cu_seqlens.numel() - 1
    else:
        N = B
    if inplace_final_state and initial_state is None:
        raise ValueError("In-place mode requires initial_state")
    if initial_state is not None:
        if initial_state.ndim != 4 or initial_state.shape[1:] != (HV, V, K):
            raise ValueError("initial_state must use [pool, HV, V, K] layout")
        if initial_state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Unsupported state dtype")
        if any(s <= 0 for s in initial_state.stride()):
            raise ValueError("State must have positive non-overlapping strides")
        ordered = sorted(zip(initial_state.stride(), initial_state.shape))
        extent = 1
        for stride, size in ordered:
            if size > 1 and stride < extent:
                raise ValueError("Overlapping state is unsupported")
            extent = max(extent, stride * size)
        pool, ss = initial_state.shape[0], initial_state.stride()
        if ssm_state_indices is None and pool < N:
            raise ValueError("State pool is smaller than the sequence count")
    else:
        pool, ss = N, (HV * V * K, V * K, K, 1)
    ix, width = (0, 0), 1
    if ssm_state_indices is not None:
        if initial_state is None:
            raise ValueError("Indexed mode requires an initial state pool")
        if ssm_state_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("State indices must be int32/int64")
        if ssm_state_indices.ndim not in (1, 2) or ssm_state_indices.shape[0] != N:
            raise ValueError("State indices must have shape [N] or [N,width]")
        if ssm_state_indices.ndim == 2:
            width = ssm_state_indices.shape[1]
            if width == 0:
                raise ValueError("State table cannot have zero width")
            ix = ssm_state_indices.stride()
        else:
            ix = (ssm_state_indices.stride(0), 0)
    if num_accepted_tokens is not None:
        if ssm_state_indices is None or ssm_state_indices.ndim != 2:
            raise ValueError("Accepted-token counts require a 2D state table")
        if (
            num_accepted_tokens.shape != (N,)
            or not num_accepted_tokens.is_contiguous()
            or num_accepted_tokens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Accepted-token counts must be contiguous int32/int64 [N]")
    if g is not None and g.shape != (B, T, HV):
        raise ValueError("g must have shape [B,T,HV]")
    if beta is not None and beta.shape not in ((B, T, HV), (B, T, HV, V)):
        raise ValueError("beta must have shape [B,T,HV] or [B,T,HV,V]")
    for tensor in (
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
    ):
        if tensor is not None and tensor.device != q.device:
            raise ValueError("All inputs must be on the same device")
    for tensor in (g, beta):
        if tensor is not None and tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("g/beta must be floating point")
    o = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    final = (
        initial_state
        if inplace_final_state
        else torch.empty(
            (B * T, HV, V, K),
            dtype=initial_state.dtype if initial_state is not None else torch.float32,
            device=q.device,
        )
    )
    if B * T == 0:
        return o, final
    if N == 0:
        raise ValueError("Nonempty input requires at least one sequence")
    # No-autotune exemption: replaying this kernel mutates its state pool.
    # Tile experiments must restore state outside timing, never in production.
    bv = 16
    bs = beta.stride() if beta is not None else (0, 0, 0)
    if len(bs) == 3:
        bs = (*bs, 0)
    total = N * HV * triton.cdiv(V, bv)
    grid, kernel = (total,), _recurrent_gdn_kernel
    kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        o,
        final,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        scale,
        T,
        H,
        HV,
        K,
        V,
        pool,
        q.stride(),
        k.stride(),
        v.stride(),
        g.stride() if g is not None else (0, 0, 0),
        bs,
        ss,
        ix,
        width,
        cu_seqlens is not None,
        ssm_state_indices is not None,
        ssm_state_indices is not None and ssm_state_indices.ndim == 2,
        num_accepted_tokens is not None,
        initial_state is not None,
        inplace_final_state,
        beta is not None and beta.ndim == 4,
        use_qk_l2norm_in_kernel,
        g is not None,
        beta is not None,
        triton.next_power_of_2(K),
        bv,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return o, final

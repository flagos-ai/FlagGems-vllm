# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime


@triton.jit
def gdn_w8a16_kernel(
    Q,
    Kp,
    Vp,
    G,
    Beta,
    State,
    Cu,
    O,
    Final,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    SCALE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SINGLE: tl.constexpr,
    WIDE: tl.constexpr,
):
    seq, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, end = tl.load(Cu + seq), tl.load(Cu + seq + 1)
    if WIDE:
        seq = seq.to(tl.int64)
        start, end = start.to(tl.int64), end.to(tl.int64)
    key_idx = tl.arange(0, BK)
    value_idx = tile * BV + tl.arange(0, BV)
    mask = (value_idx[:, None] < DV) & (key_idx[None, :] < DK)
    offset = ((seq * HV + head) * DV + value_idx[:, None]) * DK + key_idx[None, :]
    state = tl.load(State + offset, mask, other=0).to(tl.float32)
    qhead = head // (HV // H)
    if SINGLE and end - start == 1:
        token = start
        query = tl.load(
            Q + token * QS + qhead * DK + key_idx, key_idx < DK, other=0
        ).to(tl.float32)
        key = tl.load(Kp + token * KS + qhead * DK + key_idx, key_idx < DK, other=0).to(
            tl.float32
        )
        value = tl.load(
            Vp + token * VS + head * DV + value_idx, value_idx < DV, other=0
        ).to(tl.float32)
        gate = tl.load(G + token * HV + head).to(tl.float32)
        beta = tl.load(Beta + token * HV + head).to(tl.float32)
        state *= tl.exp(gate)
        delta = (value - tl.sum(state * key[None, :], 1)) * beta
        state += delta[:, None] * key[None, :]
        out = tl.sum(state * query[None, :], 1) * SCALE
        tl.store(O + (token * HV + head) * DV + value_idx, out, value_idx < DV)
    else:
        for token in range(start, end):
            query = tl.load(
                Q + token * QS + qhead * DK + key_idx, key_idx < DK, other=0
            ).to(tl.float32)
            key = tl.load(
                Kp + token * KS + qhead * DK + key_idx, key_idx < DK, other=0
            ).to(tl.float32)
            value = tl.load(
                Vp + token * VS + head * DV + value_idx, value_idx < DV, other=0
            ).to(tl.float32)
            gate = tl.load(G + token * HV + head).to(tl.float32)
            beta = tl.load(Beta + token * HV + head).to(tl.float32)
            state *= tl.exp(gate)
            delta = (value - tl.sum(state * key[None, :], 1)) * beta
            state += delta[:, None] * key[None, :]
            out = tl.sum(state * query[None, :], 1) * SCALE
            tl.store(O + (token * HV + head) * DV + value_idx, out, value_idx < DV)
    tl.store(Final + offset, state, mask)


def gdn_w8a16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GDN core forward with precomputed log-decay and sigmoid beta.

    q/k: [1,T,H,K], v: [1,T,HV,V], g/beta: [1,T,HV].
    State is FP32 [num_sequences,HV,V,K]. Packed QKV views are supported.
    W8 refers to the surrounding model weights; no projection is performed here.
    This forward-only implementation leaves input state unchanged. MetaX uses
    a 64-token chunk path for long sequences with 128-dimensional heads;
    other shapes retain the recurrence. Chunk workspace is capped at 2 GiB.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Expected rank-four q/k/v and matching q/k shapes")
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if (
        batch != 1
        or v.shape[:2] != (1, tokens)
        or heads < 1
        or value_heads < 1
        or value_heads % heads
    ):
        raise ValueError("Expected packed batch=1 and HV divisible by H")
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise NotImplementedError("Only BF16/FP16 activations are supported")
    if (
        any(t.device != q.device for t in (k, v, g, beta, initial_state, cu_seqlens))
        or q.device.type != "cuda"
    ):
        raise ValueError("All inputs must be on the same GPU")
    if any(t.stride(-1) != 1 or t.stride(-2) != t.shape[-1] for t in (q, k, v)):
        raise NotImplementedError("Only contiguous head dimensions are supported")
    if any(
        t.shape != (1, tokens, value_heads) or not t.is_contiguous() for t in (g, beta)
    ):
        raise ValueError("g/beta must be contiguous [1,T,HV]")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise NotImplementedError("g/beta must be FP32")
    if (
        cu_seqlens.ndim != 1
        or not cu_seqlens.is_contiguous()
        or cu_seqlens.numel() < 2
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("cu_seqlens must be a contiguous integer vector")
    sequences = cu_seqlens.numel() - 1
    if (
        initial_state.shape != (sequences, value_heads, value_dim, key_dim)
        or initial_state.dtype != torch.float32
        or not initial_state.is_contiguous()
    ):
        raise ValueError("State must be contiguous FP32 [N,HV,V,K]")
    if key_dim > 256 or value_dim > 256 or key_dim < 1 or value_dim < 1:
        raise NotImplementedError("Head dimensions must be in [1,256]")
    if any(t.requires_grad for t in (q, k, v, g, beta, initial_state)):
        raise NotImplementedError("Only inference forward is supported")
    # cu_seqlens values follow the caller's packed-sequence contract: monotone,
    # starting at zero and ending at T; reading them here would synchronize.
    # MetaX measurements select chunking for long, 128-dimensional heads.
    # Keep other backends, dimensions, and short sequences on the recurrence.
    if (
        runtime.device.vendor_name == "metax"
        and key_dim == value_dim == 128
        and tokens >= 128 * sequences
        and sequences <= 64
    ):
        from flaggems_vllm.runtime.backend._metax.gdn_w8a16_chunk import (
            _workspace_plan,
            gdn_w8a16_chunk,
        )

        if _workspace_plan(tokens, sequences, value_heads)[1]:
            return gdn_w8a16_chunk(
                (q, k, v, g, beta, initial_state, cu_seqlens),
                key_dim**-0.5 if scale is None else scale,
            )
    single = tokens == sequences
    value_block = 32 if single and sequences >= 3 else 16
    output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    final_state = torch.empty(initial_state.shape, device=q.device, dtype=torch.float32)
    gdn_w8a16_kernel[(sequences, value_heads, triton.cdiv(value_dim, value_block))](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        output,
        final_state,
        tokens,
        heads,
        value_heads,
        key_dim,
        value_dim,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        key_dim**-0.5 if scale is None else scale,
        triton.next_power_of_2(key_dim),
        value_block,
        single,
        max(
            tokens * max(q.stride(1), k.stride(1), v.stride(1)),
            initial_state.numel(),
            v.numel(),
        )
        >= 2**31,
        num_warps=4,
        num_stages=1,
    )
    return output, final_state

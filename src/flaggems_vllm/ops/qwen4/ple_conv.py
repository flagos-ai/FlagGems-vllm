# SPDX-License-Identifier: Apache-2.0
"""PLE causal depthwise convolution with decode and speculative state updates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _request(token, starts, stride, requests: tl.constexpr):
    low = tl.full(token.shape, 0, tl.int32)
    high = tl.full(token.shape, requests, tl.int32)
    while tl.sum((low < high).to(tl.int32), 0) > 0:
        mid = (low + high) // 2
        end = tl.load(starts + (mid + 1) * stride, mask=mid < requests, other=0)
        advance = (low < high) & (token >= end)
        high = tl.where((low < high) & ~advance, mid, high)
        low = tl.where(advance, mid + 1, low)
    return low


@triton.jit
def _ple_conv_kernel(
    x,
    out,
    state,
    weight,
    indices,
    starts,
    initial,
    accepted,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    requests: tl.constexpr,
    state_rows: tl.constexpr,
    state_len: tl.constexpr,
    width: tl.constexpr,
    dilation: tl.constexpr,
    max_query: tl.constexpr,
    null_id: tl.constexpr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    sw0: tl.constexpr,
    sw1: tl.constexpr,
    si: tl.constexpr,
    sq: tl.constexpr,
    sh: tl.constexpr,
    sa: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    SPEC: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    if MODE == 2:
        req = _request(t, starts, sq, requests)
        begin = tl.load(starts + req * sq, mask=req < requests, other=0)
        end = tl.load(starts + (req + 1) * sq, mask=req < requests, other=0)
    elif MODE == 1:
        req = t
        begin, end = t, t + 1
    else:
        req = tl.full((BT,), 0, tl.int32)
        begin, end = tl.full((BT,), 0, tl.int32), tokens
    valid = (t < tokens) & (req < requests) & (t < end)
    row = tl.load(indices + req * si, mask=valid & (MODE != 0), other=0).to(tl.int64)
    valid_state = (row != null_id) & (row >= 0) & (row < state_rows)
    if MODE == 0:
        valid_state = tl.full((BT,), True, tl.int1)
    has_history = valid_state & (MODE != 0)
    if HAS_INITIAL:
        has_history &= tl.load(initial + req * sh, mask=valid, other=0).to(tl.int1)
    rollback = tl.full((BT,), 0, tl.int32)
    if SPEC:
        rollback = tl.load(accepted + req * sa, mask=valid, other=0) - 1
        has_history &= (rollback >= 0) & (rollback < max_query)
    acc = tl.zeros((BT, BC), tl.float32)
    for tap in tl.static_range(width):
        pos = t - begin - (width - 1 - tap) * dilation
        values = tl.load(
            x + (begin + pos)[:, None] * sx0 + c[None, :] * sx1,
            mask=valid[:, None] & (pos[:, None] >= 0) & (c[None, :] < channels),
            other=0,
        ).to(tl.float32)
        if state_len > 0:
            old_pos = rollback + state_len + pos
            old = (
                tl.load(
                    state
                    + row[:, None] * ss0
                    + c[None, :] * ss1
                    + old_pos[:, None] * ss2,
                    mask=valid[:, None]
                    & has_history[:, None]
                    & (pos[:, None] < 0)
                    & (old_pos[:, None] >= 0)
                    & (c[None, :] < channels),
                    other=0,
                )
                .to(x.dtype.element_ty)
                .to(tl.float32)
            )
            values += old
        w = tl.load(weight + c * sw0 + tap * sw1, mask=c < channels, other=0).to(
            tl.float32
        )
        acc += values * w[None, :]
    # Native conv stores in x.dtype before SiLU. Keep that rounding boundary.
    conv = acc.to(x.dtype.element_ty).to(tl.float32)
    result = conv * tl.sigmoid(conv)
    result = tl.where(valid[:, None] & valid_state[:, None], result, 0)
    tl.store(
        out + t[:, None] * so0 + c[None, :] * so1,
        result,
        mask=(t[:, None] < tokens) & (c[None, :] < channels),
    )


@triton.jit
def _ple_conv_state_kernel(
    x,
    state,
    indices,
    starts,
    initial,
    accepted,
    channels: tl.constexpr,
    state_rows: tl.constexpr,
    state_len: tl.constexpr,
    max_query: tl.constexpr,
    null_id: tl.constexpr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    si: tl.constexpr,
    sq: tl.constexpr,
    sh: tl.constexpr,
    sa: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    SPEC: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
):
    req = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s = tl.arange(0, BS)
    if MODE == 2:
        begin = tl.load(starts + req * sq)
        end = tl.load(starts + (req + 1) * sq)
    else:
        begin, end = req, req + 1
    length = end - begin
    row = tl.load(indices + req * si)
    valid = (row != null_id) & (row >= 0) & (row < state_rows) & (length > 0)
    has_history = valid
    if HAS_INITIAL:
        has_history &= tl.load(initial + req * sh).to(tl.int1)
    rollback = 0
    if SPEC:
        rollback = tl.load(accepted + req * sa) - 1
        has_history &= (rollback >= 0) & (rollback < max_query)
        # Keep all candidate tokens, preserving the unused extended tail.
        pos = s + 1 - state_len
        write = s < state_len + length - 1
    else:
        pos = s + length - state_len
        write = s < state_len
    mask = valid & write[None, :] & (c[:, None] < channels)
    values = tl.load(
        x + (begin + pos)[None, :] * sx0 + c[:, None] * sx1,
        mask=mask & (pos[None, :] >= 0) & (pos[None, :] < length),
        other=0,
    )
    old_pos = rollback + state_len + pos
    old = tl.load(
        state + row * ss0 + c[:, None] * ss1 + old_pos[None, :] * ss2,
        mask=mask & has_history & (pos[None, :] < 0) & (old_pos[None, :] >= 0),
        other=0,
    )
    # Every CTA reads its complete old channel slice before writing it; no
    # other CTA updates those channels. Output readers finished in prior launch.
    result = (values.to(x.dtype.element_ty) + old.to(x.dtype.element_ty)).to(
        state.dtype.element_ty
    )
    tl.store(state + row * ss0 + c[:, None] * ss1 + s[None, :] * ss2, result, mask=mask)


def ple_short_conv_(
    x: torch.Tensor,
    output: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    state_indices: torch.Tensor | None,
    *,
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    accepted_tokens: torch.Tensor | None = None,
    max_query_len: int = 1,
    null_block_id: int = -1,
) -> None:
    """Write causal conv+SiLU and PLE state without packing/copying the cache.

    Without offsets, each cache index corresponds to one decode token. With
    offsets, requests may be ragged or empty; graph-padded tokens are zeroed.
    accepted_tokens selects the previous speculative window at accepted-1.
    With no indices, run a zero-history profiling convolution without writes.
    Speculative query lengths must not exceed max_query_len.
    Valid nonempty requests must have distinct cache indices.
    Output storage must not alias input or cache storage. Forward inference only.
    """
    if x.ndim != 2 or conv_weight.ndim != 2 or conv_weight.shape[0] != x.shape[1]:
        raise ValueError("PLE expects x[T,C] and weight[C,K]")
    if output.shape != x.shape or output.dtype != x.dtype:
        raise ValueError("PLE output must match input shape and dtype")
    if conv_state.ndim != 3 or conv_state.shape[1] != x.shape[1]:
        raise ValueError("PLE state must have shape [rows,C,S]")
    width = conv_weight.shape[1]
    state_len = (width - 1) * dilation
    spec = accepted_tokens is not None
    capacity = state_len + (max_query_len - 1 if spec else 0)
    if dilation <= 0 or width < 1 or width > 8 or max_query_len < 1:
        raise ValueError("Invalid PLE convolution geometry")
    if state_indices is not None and conv_state.shape[2] < capacity:
        raise ValueError("PLE state cannot retain the required history")
    mode = 0 if state_indices is None else (2 if query_start_loc is not None else 1)
    requests = 1 if mode == 0 else state_indices.numel()
    if state_indices is not None and (
        state_indices.ndim != 1 or state_indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("PLE cache indices must be an integer vector")
    if mode == 1 and requests != x.shape[0]:
        raise ValueError("PLE decode needs one state index per token")
    if query_start_loc is not None and (
        query_start_loc.ndim != 1
        or query_start_loc.numel() != requests + 1
        or query_start_loc.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("PLE query offsets must cover every request")
    if spec and mode != 2:
        raise ValueError("PLE speculative decode requires query offsets")
    for name, tensor in (
        ("initial-state flags", has_initial_state),
        ("accepted counts", accepted_tokens),
    ):
        if tensor is not None and (tensor.ndim != 1 or tensor.numel() < requests):
            raise ValueError(f"PLE {name} must cover every request")
    if has_initial_state is not None and has_initial_state.dtype != torch.bool:
        raise TypeError("PLE initial-state flags must be boolean")
    if accepted_tokens is not None and accepted_tokens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("PLE accepted counts must be integers")
    floating = (x, output, conv_state, conv_weight)
    if any(
        t.dtype not in (torch.float16, torch.bfloat16, torch.float32) for t in floating
    ):
        raise NotImplementedError("PLE supports FP16, BF16 and FP32")
    if conv_weight.dtype != x.dtype:
        raise TypeError("PLE weights must match input dtype")
    tensors = (
        *floating,
        state_indices,
        query_start_loc,
        has_initial_state,
        accepted_tokens,
    )
    if x.device.type in ("cpu", "meta") or any(
        t is not None and t.device != x.device for t in tensors
    ):
        raise NotImplementedError("PLE requires tensors on one Triton accelerator")
    if not x.numel():
        return
    # Absent pointer arguments are compile-time dead. Reuse x as a dummy,
    # allocating no flags/indices and submitting no initialization kernels.
    dummy = x
    _ple_conv_kernel[(triton.cdiv(x.shape[0], 8), triton.cdiv(x.shape[1], 128))](
        x,
        output,
        conv_state,
        conv_weight,
        state_indices if state_indices is not None else dummy,
        query_start_loc if query_start_loc is not None else dummy,
        has_initial_state if has_initial_state is not None else dummy,
        accepted_tokens if accepted_tokens is not None else dummy,
        x.shape[0],
        x.shape[1],
        requests,
        conv_state.shape[0],
        state_len,
        width,
        dilation,
        max_query_len,
        null_block_id,
        *x.stride(),
        *output.stride(),
        *conv_state.stride(),
        *conv_weight.stride(),
        state_indices.stride(0) if state_indices is not None else 0,
        query_start_loc.stride(0) if query_start_loc is not None else 0,
        has_initial_state.stride(0) if has_initial_state is not None else 0,
        accepted_tokens.stride(0) if accepted_tokens is not None else 0,
        MODE=mode,
        HAS_INITIAL=has_initial_state is not None,
        SPEC=spec,
        BT=8,
        BC=128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if mode and requests and state_len:
        _ple_conv_state_kernel[(requests, triton.cdiv(x.shape[1], 128))](
            x,
            conv_state,
            state_indices,
            query_start_loc if query_start_loc is not None else dummy,
            has_initial_state if has_initial_state is not None else dummy,
            accepted_tokens if accepted_tokens is not None else dummy,
            x.shape[1],
            conv_state.shape[0],
            state_len,
            max_query_len,
            null_block_id,
            *x.stride(),
            *conv_state.stride(),
            state_indices.stride(0),
            query_start_loc.stride(0) if query_start_loc is not None else 0,
            has_initial_state.stride(0) if has_initial_state is not None else 0,
            accepted_tokens.stride(0) if accepted_tokens is not None else 0,
            MODE=mode,
            HAS_INITIAL=has_initial_state is not None,
            SPEC=spec,
            BS=triton.next_power_of_2(capacity),
            BC=128,
            num_warps=4,
        )


__all__ = ["ple_short_conv_"]

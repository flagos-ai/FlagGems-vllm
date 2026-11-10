# SPDX-License-Identifier: Apache-2.0
"""PLE causal depthwise convolution with decode and speculative state updates."""

from __future__ import annotations

from typing import Literal

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
def _ple_short_conv_kernel(
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
def _ple_short_conv_state_kernel(
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
    # Promote before multiplying the cache row stride, which can exceed int32.
    row = tl.load(indices + req * si).to(tl.int64)
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
    result = (values.to(x.dtype.element_ty) + old.to(x.dtype.element_ty)).to(
        state.dtype.element_ty
    )
    # History spans warps: finish every old-state load before any in-place write.
    tl.debug_barrier()
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
    _ple_short_conv_kernel[(triton.cdiv(x.shape[0], 8), triton.cdiv(x.shape[1], 128))](
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
        _ple_short_conv_state_kernel[(requests, triton.cdiv(x.shape[1], 128))](
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



# Match vLLM's reserved null cache slot for the upstream-compatible API.
NULL_BLOCK_ID = 0

try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    def _arch_support_pdl() -> bool:
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 9

except ImportError:
    gdc_launch_dependents = None  # type: ignore[assignment]
    gdc_wait = None  # type: ignore[assignment]

    def _arch_support_pdl() -> bool:
        return False


# Dilated short convolution
@triton.jit(do_not_specialize=["num_reqs", "bs_iters", "has_token_map"])
def _ple_conv_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    residual_ptr,
    outer_residual_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    token_idx_ptr,
    has_token_map,
    num_reqs,
    bs_iters,
    state_idx_stride,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DILATION: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    FUSE_RESIDUAL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    if launch_pdl:
        gdc_wait()

    # Mode-specific launches use logical row indices. Mixed batches map them
    # back to the original token order without gathering inputs or residuals.
    output_t = tl.load(token_idx_ptr + t, mask=has_token_map, other=t).to(tl.int64)

    if MODE == "decode":
        r = t
        q_start = t
        j = tl.full([], 0, tl.int32)
        slot_off = tl.full([], 0, tl.int32)
        in_range = True
    else:
        # Locate the request containing this token.
        lo = tl.full([], 1, tl.int32)
        hi = tl.full([], num_reqs + 1, tl.int32)
        for _ in range(bs_iters):
            mid = (lo + hi) // 2
            qmid = tl.load(qsl_ptr + mid, mask=mid <= num_reqs, other=0)
            pred = qmid <= t
            lo = tl.where(pred, mid + 1, lo)
            hi = tl.where(pred, hi, mid)
        r = tl.minimum(lo - 1, num_reqs - 1)
        q_start = tl.load(qsl_ptr + r)
        j = (t - q_start).to(tl.int32)
        total_real = tl.load(qsl_ptr + num_reqs)
        in_range = t < total_real
        if MODE == "spec":
            num_acc = tl.load(num_acc_ptr + r)
            # Roll back draft state to the last accepted token.
            slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(
                tl.int32
            )
        else:
            slot_off = tl.full([], 0, tl.int32)

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    sid_safe = tl.where(state_ok, sid, 0)
    if HAS_INIT:
        has_init = tl.load(has_init_ptr + r, mask=state_ok, other=0) != 0
    else:
        has_init = state_ok
    if MODE == "spec":
        # A new spec chunk can produce output from its own tokens without a
        # prior state; null slots in other modes denote inactive rows.
        read_state = state_ok
        out_ok = in_range
    else:
        read_state = state_ok & has_init
        out_ok = in_range & state_ok

    # state[sid, w, c] may be a strided view into a shared pool.
    base_state = state_ptr + sid_safe * state_bs
    acc = tl.zeros([BLOCK_C], tl.float32)
    for k in tl.static_range(0, KERNEL_SIZE):
        h = j + DILATION * k
        from_state = h <= STATE_LEN - 1
        state_tap = tl.load(
            base_state + (h + slot_off) * state_ws + c_offs * state_cs,
            mask=c_mask & read_state & from_state,
            other=0.0,
        )
        input_t = q_start + h - STATE_LEN
        input_t = tl.load(
            token_idx_ptr + input_t,
            mask=has_token_map & out_ok & (~from_state),
            other=input_t,
        ).to(tl.int64)
        input_tap = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & out_ok & (~from_state),
            other=0.0,
        )
        tap = tl.where(from_state, state_tap, input_tap).to(tl.float32)
        weight = tl.load(
            w_ptr + c_offs * KERNEL_SIZE + k,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        acc += weight * tap

    # F.conv1d materializes its output dtype before SiLU.
    conv = acc.to(residual_ptr.dtype.element_ty).to(tl.float32)
    y = conv * tl.sigmoid(conv)
    conv_output = tl.where(out_ok, y, 0.0).to(residual_ptr.dtype.element_ty)
    ple_output = conv_output
    if FUSE_RESIDUAL:
        residual = tl.load(
            residual_ptr + output_t * C + c_offs,
            mask=c_mask,
            other=0.0,
        )
        # Preserve the original eager operation boundaries: short convolution is
        # first accumulated into the BF16/FP16 PLE output, then the outer residual
        # is added to that rounded value.
        ple_output = (residual + conv_output).to(residual_ptr.dtype.element_ty)
        outer_residual = tl.load(
            outer_residual_ptr + output_t * C + c_offs,
            mask=c_mask,
            other=0.0,
        )
        ple_output = outer_residual.to(tl.float32) + ple_output.to(tl.float32)
    if launch_pdl:
        gdc_launch_dependents()
    tl.store(
        residual_ptr + output_t * C + c_offs,
        ple_output,
        mask=c_mask,
    )

    if MODE == "decode":
        # Decode has one program per request and channel block, so no peer can
        # still be reading this state slice when it is updated.
        decode_input = tl.load(
            x_ptr + output_t * C + c_offs,
            mask=c_mask & state_ok,
            other=0.0,
        )
        for i in tl.static_range(0, STATE_LEN):
            if i < STATE_LEN - 1:
                next_state = tl.load(
                    base_state + (i + 1) * state_ws + c_offs * state_cs,
                    mask=c_mask & state_ok & has_init,
                    other=0.0,
                )
            else:
                next_state = decode_input
            tl.store(
                base_state + i * state_ws + c_offs * state_cs,
                next_state,
                mask=c_mask & state_ok,
            )


# Prefill and spec write back separately so every token reads the old state.
@triton.jit(do_not_specialize=["has_token_map"])
def _ple_conv_writeback_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    token_idx_ptr,
    has_token_map,
    state_idx_stride,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
):
    r = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    if not state_ok:
        return

    q_start = tl.load(qsl_ptr + r)
    q_end = tl.load(qsl_ptr + r + 1)
    qlen = (q_end - q_start).to(tl.int32)
    if MODE == "spec":
        num_acc = tl.load(num_acc_ptr + r)
        # Use the same rollback window as the output kernel.
        slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(
            tl.int32
        )
        shift = 1
    else:
        slot_off = tl.full([], 0, tl.int32)
        shift = qlen
        if qlen <= 0:
            return

    has_init = tl.load(has_init_ptr + r) != 0 if HAS_INIT else True
    src_ok = state_ok if MODE == "spec" else state_ok & has_init

    WRITE_W: tl.constexpr = STATE_WIDTH if MODE == "spec" else STATE_LEN
    base_state = state_ptr + sid * state_bs
    for i in tl.static_range(0, WRITE_W):
        m = shift + i
        from_state = m <= STATE_LEN - 1
        # Short or graph-padded spec chunks preserve the unused state tail.
        do_write = i < STATE_LEN + qlen - 1 if MODE == "spec" else True
        state_value = tl.load(
            base_state + (slot_off + m) * state_ws + c_offs * state_cs,
            mask=c_mask & from_state & src_ok,
            other=0.0,
        )
        input_t = q_start + m - STATE_LEN
        input_t = tl.load(
            token_idx_ptr + input_t,
            mask=has_token_map & (~from_state) & do_write,
            other=input_t,
        ).to(tl.int64)
        input_value = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & (~from_state) & do_write,
            other=0.0,
        )
        value = tl.where(from_state, state_value, input_value)
        tl.store(
            base_state + i * state_ws + c_offs * state_cs,
            value,
            mask=c_mask & do_write,
        )


def ple_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    outer_residual: torch.Tensor | None,
    *,
    mode: Literal["decode", "spec", "prefill"],
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    spec_query_len: int = 1,
    token_indices: torch.Tensor | None = None,
) -> None:
    """Update state and write activated convolution into residual.
    A non-None outer_residual enables both fused residual additions.
    """
    BLOCK_C = 512
    kernel_spec_query_len = spec_query_len if mode == "spec" else 1
    T, C = inputs.shape
    K = conv_weights.shape[1]
    state_len = (K - 1) * dilation
    state_width = state_len + kernel_spec_query_len - 1
    if token_indices is not None:
        T = token_indices.numel()
    if conv_state.shape[1] != C or conv_state.shape[2] < state_width:
        raise ValueError(
            "conv_state must have shape [slots, channels, window], with "
            f"channels={C} and window >= {state_width}"
        )
    state_bs, state_cs, state_ws = conv_state.stride()

    if mode == "decode":
        num_reqs = T
        binary_search_iters = 1
        has_initial_states_arg = has_initial_states is not None
    elif mode == "spec":
        if query_start_loc is None or num_accepted_tokens is None:
            raise ValueError(
                "query_start_loc and num_accepted_tokens are required for spec decode"
            )
        num_reqs = state_indices.numel()
        binary_search_iters = max(num_reqs, 1).bit_length()
        has_initial_states_arg = False
    elif mode == "prefill":
        if query_start_loc is None or has_initial_states is None:
            raise ValueError(
                "query_start_loc and has_initial_states are required for prefill"
            )
        num_reqs = state_indices.numel()
        binary_search_iters = max(num_reqs, 1).bit_length()
        has_initial_states_arg = True
    else:
        raise ValueError(f"Unsupported short-conv mode: {mode}")

    num_warps = 4 if mode == "prefill" else 8
    launch_pdl = _arch_support_pdl()
    # Pure-prefill indices can be a strided block-table column view.
    state_idx_stride = state_indices.stride(0)

    # Constexpr flags eliminate accesses to optional None arguments. Absent
    # pointers reuse a device-resident dummy so Triton always receives an address.
    dummy = inputs
    _ple_conv_kernel[(T, triton.cdiv(C, BLOCK_C))](
        inputs,
        conv_state,
        conv_weights,
        residual,
        outer_residual if outer_residual is not None else dummy,
        state_indices,
        query_start_loc if query_start_loc is not None else dummy,
        num_accepted_tokens if num_accepted_tokens is not None else dummy,
        has_initial_states if has_initial_states is not None else dummy,
        token_indices if token_indices is not None else state_indices,
        token_indices is not None,
        num_reqs,
        binary_search_iters,
        state_idx_stride,
        state_bs,
        state_ws,
        state_cs,
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        DILATION=dilation,
        KERNEL_SIZE=K,
        SPEC_QUERY_LEN=kernel_spec_query_len,
        MODE=mode,
        HAS_INIT=has_initial_states_arg,
        FUSE_RESIDUAL=outer_residual is not None,
        NULL_STATE_ID=NULL_BLOCK_ID,
        launch_pdl=launch_pdl,
        num_warps=num_warps,
    )
    # conv state update is fused with the kernel above for decode
    if mode != "decode":
        _ple_conv_writeback_kernel[(num_reqs, triton.cdiv(C, BLOCK_C))](
            inputs,
            conv_state,
            state_indices,
            query_start_loc if query_start_loc is not None else dummy,
            num_accepted_tokens if num_accepted_tokens is not None else dummy,
            has_initial_states if has_initial_states is not None else dummy,
            token_indices if token_indices is not None else state_indices,
            token_indices is not None,
            state_idx_stride,
            state_bs,
            state_ws,
            state_cs,
            C=C,
            BLOCK_C=BLOCK_C,
            STATE_LEN=state_len,
            SPEC_QUERY_LEN=kernel_spec_query_len,
            STATE_WIDTH=state_width,
            MODE=mode,
            HAS_INIT=has_initial_states_arg,
            NULL_STATE_ID=NULL_BLOCK_ID,
            num_warps=num_warps,
        )


__all__ = ["ple_short_conv_", "ple_conv", "NULL_BLOCK_ID"]

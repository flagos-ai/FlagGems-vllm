# SPDX-License-Identifier: Apache-2.0
"""PLE causal depthwise convolution with decode and speculative state updates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    request_pad: tl.constexpr,
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
    # GCU backend: row-streamed form — channels are the only vector dimension and
    # every memory op is 1D. 2D tiles abort in MaskAnalysis (vec-vs-vec cmpi in a
    # mask) or fail memref_ext.slice_* lowering in make_llir.
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    in_ch = c < channels
    ws = ()
    for tap in tl.static_range(width):
        ws = ws + (
            tl.load(weight + c * sw0 + tap * sw1, mask=in_ch, other=0).to(tl.float32),
        )
    if MODE == 2:
        # Request of token t = #{k in [1, requests]: starts[k] <= t}. Replaces the
        # data-dependent binary search; when req < requests, t < starts[req + 1]
        # holds by construction, so the old `t < end` validity clause is implied.
        lane = tl.arange(0, request_pad)
        bounds = tl.load(starts + (lane + 1) * sq, mask=lane < requests, other=1 << 30)
    for i in tl.static_range(BT):
        t = tl.program_id(0) * BT + i
        if MODE == 2:
            req = tl.sum((bounds <= t).to(tl.int32), 0)
            begin = tl.load(starts + req * sq, mask=req < requests, other=0)
        elif MODE == 1:
            req = t
            begin = t
        else:
            req = 0
            begin = 0
        valid = (t < tokens) & (req < requests)
        row = tl.load(indices + req * si, mask=valid & (MODE != 0), other=0).to(
            tl.int64
        )
        if MODE == 0:
            valid_state = True
        else:
            valid_state = (row != null_id) & (row >= 0) & (row < state_rows)
        has_history = valid_state & (MODE != 0)
        if HAS_INITIAL:
            has_history &= tl.load(initial + req * sh, mask=valid, other=0).to(tl.int1)
        rollback = 0
        if SPEC:
            rollback = tl.load(accepted + req * sa, mask=valid, other=0) - 1
            has_history &= (rollback >= 0) & (rollback < max_query)
        acc = tl.zeros((BC,), tl.float32)
        for tap in tl.static_range(width):
            pos = t - begin - (width - 1 - tap) * dilation
            values = tl.load(
                x + (begin + pos) * sx0 + c * sx1,
                mask=valid & (pos >= 0) & in_ch,
                other=0,
            ).to(tl.float32)
            if state_len > 0:
                old_pos = rollback + state_len + pos
                old = (
                    tl.load(
                        state + (row * ss0 + old_pos.to(tl.int64) * ss2) + c * ss1,
                        mask=valid & has_history & (pos < 0) & (old_pos >= 0) & in_ch,
                        other=0,
                    )
                    .to(x.dtype.element_ty)
                    .to(tl.float32)
                )
                values += old
            w = ws[tap]
            acc += values * w
        # Native conv stores in x.dtype before SiLU. Keep that rounding boundary.
        conv = acc.to(x.dtype.element_ty).to(tl.float32)
        result = conv * tl.sigmoid(conv)
        result = tl.where(valid & valid_state, result, 0)
        tl.store(
            out + t * so0 + c * so1,
            result,
            mask=(t < tokens) & in_ch,
        )


@triton.jit
def _ple_conv_staged_kernel(
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
    request_pad: tl.constexpr,
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
    # Sliding-window row staging for decode/prefill/profile (non-SPEC): load
    # BT + (width-1)*dilation consecutive rows of x once per program, all
    # staged before the token loop, so taps of neighbouring tokens share the
    # loads (width=4, dilation=3 stages 17 rows instead of 32 masked loads)
    # and the loads are no longer interleaved with the accumulate chain.
    # Row for token i, tap t lives at staged index i + t*dilation. state reads
    # stay per-tap because old_pos depends on each token's begin/rollback;
    # where(pos >= 0, staged, old) keeps the pos < 0 semantics: the staged row
    # may belong to a previous request, so it must not leak into tokens whose
    # window starts after it.
    t0 = tl.program_id(0) * BT
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    in_ch = c < channels
    ws = ()
    for tap in tl.static_range(width):
        ws = ws + (
            tl.load(weight + c * sw0 + tap * sw1, mask=in_ch, other=0).to(tl.float32),
        )
    rows = ()
    for j in tl.static_range(BT + (width - 1) * dilation):
        r = t0 - (width - 1) * dilation + j
        rows = rows + (
            tl.load(
                x + r * sx0 + c * sx1, mask=(r >= 0) & (r < tokens) & in_ch, other=0
            ).to(tl.float32),
        )
    if MODE == 2:
        lane = tl.arange(0, request_pad)
        bounds = tl.load(starts + (lane + 1) * sq, mask=lane < requests, other=1 << 30)
    for i in tl.static_range(BT):
        t = t0 + i
        if MODE == 2:
            req = tl.sum((bounds <= t).to(tl.int32), 0)
            begin = tl.load(starts + req * sq, mask=req < requests, other=0)
        elif MODE == 1:
            req = t
            begin = t
        else:
            req = 0
            begin = 0
        valid = (t < tokens) & (req < requests)
        row = tl.load(indices + req * si, mask=valid & (MODE != 0), other=0).to(
            tl.int64
        )
        if MODE == 0:
            valid_state = True
        else:
            valid_state = (row != null_id) & (row >= 0) & (row < state_rows)
        has_history = valid_state & (MODE != 0)
        if HAS_INITIAL:
            has_history &= tl.load(initial + req * sh, mask=valid, other=0).to(tl.int1)
        rollback = 0
        if SPEC:
            rollback = tl.load(accepted + req * sa, mask=valid, other=0) - 1
            has_history &= (rollback >= 0) & (rollback < max_query)
        acc = tl.zeros((BC,), tl.float32)
        for tap in tl.static_range(width):
            off = (width - 1 - tap) * dilation
            pos = t - begin - off
            values = rows[i + tap * dilation]
            if state_len > 0:
                old_pos = rollback + state_len + pos
                old = (
                    tl.load(
                        state + (row * ss0 + old_pos.to(tl.int64) * ss2) + c * ss1,
                        mask=valid
                        & has_history
                        & (pos < 0)
                        & (old_pos >= 0)
                        & in_ch,
                        other=0,
                    )
                    .to(x.dtype.element_ty)
                    .to(tl.float32)
                )
                values = tl.where(pos >= 0, values, old)
            w = ws[tap]
            acc += values * w
        # Native conv stores in x.dtype before SiLU. Keep that rounding boundary.
        conv = acc.to(x.dtype.element_ty).to(tl.float32)
        result = conv * tl.sigmoid(conv)
        result = tl.where(valid & valid_state, result, 0)
        tl.store(
            out + t * so0 + c * so1,
            result,
            mask=(t < tokens) & in_ch,
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
    # GCU backend: slots unrolled as scalars; channels stay the 1D vector axis.
    # Each channel belongs to exactly one lane for loads and stores, so per-slot
    # load-before-store program order replaces the old cross-warp barrier.
    req = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    in_ch = c < channels
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
    for slot in tl.static_range(BS):
        if SPEC:
            # Keep all candidate tokens, preserving the unused extended tail.
            pos = slot + 1 - state_len
            write = slot < state_len + length - 1
        else:
            pos = slot + length - state_len
            write = slot < state_len
        mask = valid & write & in_ch
        values = tl.load(
            x + (begin + pos) * sx0 + c * sx1,
            mask=mask & (pos >= 0) & (pos < length),
            other=0,
        )
        old_pos = rollback + state_len + pos
        old = tl.load(
            state + (row * ss0 + old_pos.to(tl.int64) * ss2) + c * ss1,
            mask=mask & has_history & (pos < 0) & (old_pos >= 0),
            other=0,
        )
        result = (values.to(x.dtype.element_ty) + old.to(x.dtype.element_ty)).to(
            state.dtype.element_ty
        )
        tl.store(state + (row * ss0 + slot * ss2) + c * ss1, result, mask=mask)


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
    # GCU backend: kernel-side int64 loads poison downstream arithmetic (each
    # lane reads the 32-bit word at element 2k), so normalize integer metadata
    # to int32 here where the ATen cast is exact. Index values stay far below
    # 2^31; in-kernel `.to(tl.int64)` widening for row strides is safe.
    if state_indices is not None and state_indices.dtype == torch.int64:
        state_indices = state_indices.to(torch.int32)
    if query_start_loc is not None and query_start_loc.dtype == torch.int64:
        query_start_loc = query_start_loc.to(torch.int32)
    if accepted_tokens is not None and accepted_tokens.dtype == torch.int64:
        accepted_tokens = accepted_tokens.to(torch.int32)
    # Absent pointer arguments are compile-time dead. Reuse x as a dummy,
    # allocating no flags/indices and submitting no initialization kernels.
    dummy = x
    request_pad = max(triton.next_power_of_2(requests), 1)
    # GCU tuning, reswept on gcu400/L600 after the staged-kernel change
    # (sweep: tmp/probe_sweep.py, do_bench median, 2 fresh rounds):
    # wide channels want BC=2048 tiles (decode 64x10240 149->54us, spec
    # 302->194us); narrow channels want nw=2 (1024x33 65->52.6us); long
    # prefills prefer BT=4 (1024x10240 456->325us, 33x10240 39->20us).
    if x.shape[1] <= 128:
        block_c, conv_warps, conv_bt = 128, 2, 8
    elif x.shape[0] <= 8:
        block_c, conv_warps, conv_bt = 512, 2, 8
    elif spec:
        block_c, conv_warps, conv_bt = 2048, 1, 8
    elif mode == 1:
        block_c, conv_warps, conv_bt = 2048, 2, 8
    else:
        block_c, conv_warps, conv_bt = 2048, 2, 4
    if x.shape[1] <= 128:
        state_bc, state_warps = 128, 4
    elif requests > 8:
        state_bc, state_warps = 2048, 2
    else:
        state_bc, state_warps = 1024, 2
    # SPEC keeps the tap-by-tap kernel: its requests are 4-token windows, so
    # staged rows are never shared across taps of neighbouring tokens and the
    # extra staging loads plus the where select only add cost (measured on
    # gcu400/L600 core shapes). Every non-SPEC mode stages.
    conv_kernel = _ple_conv_kernel if spec else _ple_conv_staged_kernel
    conv_kernel[(triton.cdiv(x.shape[0], conv_bt), triton.cdiv(x.shape[1], block_c))](
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
        request_pad,
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
        BT=conv_bt,
        BC=block_c,
        num_warps=conv_warps,
        enable_fp_fusion=False,
    )
    if mode and requests and state_len:
        _ple_conv_state_kernel[(requests, triton.cdiv(x.shape[1], state_bc))](
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
            # Exact capacity instead of next_power_of_2: static_range unrolls
            # any bound, so the padded slots (up to 7 of 16 at decode, 4 of 16
            # at spec) stop emitting dead load/store pairs.
            BS=capacity,
            BC=state_bc,
            num_warps=state_warps,
        )


__all__ = ["ple_short_conv_"]

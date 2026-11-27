# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MetaX backend for vLLM ``ple_conv``.

This is the function vLLM calls: ``inputs, residual, conv_state,
conv_weights, state_indices, outer_residual`` plus ``mode`` and ``dilation``.
SiLU(conv) is written into ``residual``. A non-None ``outer_residual`` adds
the PLE residual in the activation dtype, then adds the outer residual in
fp32. Decode updates the conv cache in the same kernel; prefill and spec
write it back after every token has read the old cache. Slot 0 is the null
cache id. Programmatic dependent launch stays off on this backend.

When FlagTree TLE is available, channel-weight tiles are staged once in
shared memory via ``tle.gpu.alloc`` / ``local_ptr`` so dilation taps reuse
them instead of re-gathering from global memory. That replaces the earlier
host-side tile/warp-only specialization as the primary optimization.
"""

from __future__ import annotations

import types
from typing import Literal

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import has_triton_tle

NULL_BLOCK_ID = 0

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as _tle
    except ImportError:
        _tle = None  # type: ignore[assignment]
else:
    _tle = None  # type: ignore[assignment]


def _build_metax_tle():
    """MetaX needs a local_ptr shim: the plugin widens shared loads from
    pointer alignment and asserts below 4 elements/thread. Adding
    ``pid >> 31`` (always 0) keeps alignment unprovable so the load is safe.
    A builtin is required because a @jit function cannot take a TLE buffer.
    """
    if _tle is None:
        return None
    real = _tle.gpu

    @tl.core.builtin
    def local_ptr(buffer, indices=None, _semantic=None, _generator=None):
        p = real.local_ptr(buffer, indices, _semantic=_semantic, _generator=_generator)
        pid = tl.program_id(0, _semantic=_semantic)
        return p.__add__(pid.__rshift__(31, _semantic=_semantic), _semantic=_semantic)

    gpu = types.ModuleType("flaggems_metax_ple_conv_tle_gpu")
    for name, value in vars(real).items():
        if not name.startswith("__"):
            setattr(gpu, name, value)
    gpu.local_ptr = local_ptr
    shim = types.ModuleType("flaggems_metax_ple_conv_tle")
    shim.gpu = gpu
    return shim


tle = _build_metax_tle()
HAS_TLE = tle is not None

try:
    # Names stay defined so the constexpr-false PDL branches compile. They
    # are not called: this backend always passes launch_pdl=False.
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
except ImportError:
    gdc_launch_dependents = None  # type: ignore[assignment]
    gdc_wait = None  # type: ignore[assignment]


# Dilated short convolution
@triton.jit(do_not_specialize=["num_reqs", "bs_iters", "has_token_map"])
def _ple_conv_kernel_baseline(
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
    KERNEL_SIZE_SMEM: tl.constexpr,
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


if HAS_TLE:

    @triton.jit(do_not_specialize=["num_reqs", "bs_iters", "has_token_map"])
    def _ple_conv_kernel_tle(
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
        KERNEL_SIZE_SMEM: tl.constexpr,
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

        # Stage the channel x kernel weight tile in shared memory once so every
        # dilation tap reuses the same smem row instead of re-gathering weights.
        w_smem = tle.gpu.alloc(
            [BLOCK_C, KERNEL_SIZE_SMEM],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        c_local = tl.arange(0, BLOCK_C)
        for k0 in tl.static_range(0, KERNEL_SIZE):
            w0 = tl.load(
                w_ptr + c_offs * KERNEL_SIZE + k0,
                mask=c_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                tle.gpu.local_ptr(w_smem, (c_local, tl.full([BLOCK_C], k0, tl.int32))),
                w0,
                mask=c_mask,
            )

        if launch_pdl:
            gdc_wait()

        output_t = tl.load(token_idx_ptr + t, mask=has_token_map, other=t).to(tl.int64)

        if MODE == "decode":
            r = t
            q_start = t
            j = tl.full([], 0, tl.int32)
            slot_off = tl.full([], 0, tl.int32)
            in_range = True
        else:
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
                slot_off = tl.minimum(
                    tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1
                ).to(tl.int32)
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
            read_state = state_ok
            out_ok = in_range
        else:
            read_state = state_ok & has_init
            out_ok = in_range & state_ok

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
                tle.gpu.local_ptr(w_smem, (c_local, tl.full([BLOCK_C], k, tl.int32))),
                mask=c_mask,
                other=0.0,
            )
            acc += weight * tap

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


if HAS_TLE:
    _ple_conv_kernel = _ple_conv_kernel_tle
else:
    _ple_conv_kernel = _ple_conv_kernel_baseline


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
    # Spec still uses a narrower channel tile: state is [slot, channel, window]
    # so consecutive channels are strided by the window. Decode/prefill keep
    # the wider tile. Perf now comes from TLE weight smem when available.
    if mode == "spec":
        BLOCK_C = 128
        num_warps = 4
    else:
        BLOCK_C = 512
        num_warps = 4 if mode == "prefill" else 8
    kernel_spec_query_len = spec_query_len if mode == "spec" else 1
    T, C = inputs.shape
    K = conv_weights.shape[1]
    # Pad the smem weight row so TLE alloc accepts non-pow2 KERNEL_SIZE.
    kernel_size_smem = max(triton.next_power_of_2(K), 16)
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

    # C-series reports capability 8 and has no programmatic dependent launch.
    launch_pdl = False
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
        KERNEL_SIZE_SMEM=kernel_size_smem,
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


__all__ = ["NULL_BLOCK_ID", "ple_conv"]

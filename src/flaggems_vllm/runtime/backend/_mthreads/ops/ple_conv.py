# SPDX-License-Identifier: Apache-2.0
"""Mthreads backend for vLLM ``ple_conv``.

Same kernel and host as the public entry. SiLU(conv) is written into
``residual``. A non-None ``outer_residual`` adds both residuals in the same
launch. Decode updates the cache in the main kernel; prefill and spec write
it back afterwards. MUSA has no programmatic dependent launch, so
``launch_pdl`` stays false. Missing ``gdc_wait`` / ``gdc_launch_dependents``
symbols are left as None.

When FlagTree TLE is available, the main-kernel reuse path stages the
channel-weight tile (and the live state window when smem allows) through
``tle.gpu.alloc`` / ``local_ptr`` so dilation taps do not re-gather from
global memory. Tile selection and the writeback split stay as before.
"""

from __future__ import annotations

from typing import Literal

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle
    except ImportError:
        tle = None
else:
    tle = None

HAS_TLE = bool(
    tle is not None
    and hasattr(tle, "gpu")
    and hasattr(tle.gpu, "alloc")
    and hasattr(tle.gpu, "local_ptr")
    and hasattr(tle.gpu, "smem")
)

NULL_BLOCK_ID = 0

try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
except ImportError:
    # Absent symbols stay empty. A Python stub is rejected by MUSA Triton
    # when the kernel source still names it.
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
    SPEC_QUERY_LEN: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    FUSE_RESIDUAL: tl.constexpr,
    WRITE_TAIL: tl.constexpr,
    KERNEL_SIZE_SMEM: tl.constexpr,
    STATE_LEN_P2: tl.constexpr,
    STAGE_STATE: tl.constexpr,
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
        # from_state is a scalar. Loading both taps and selecting with
        # tl.where still issues the masked-off gather, which dominates on MUSA.
        from_state = h <= STATE_LEN - 1
        if from_state:
            tap = tl.load(
                base_state + (h + slot_off) * state_ws + c_offs * state_cs,
                mask=c_mask & read_state,
                other=0.0,
            ).to(tl.float32)
        else:
            input_t = q_start + h - STATE_LEN
            input_t = tl.load(
                token_idx_ptr + input_t,
                mask=has_token_map & out_ok,
                other=input_t,
            ).to(tl.int64)
            tap = tl.load(
                x_ptr + input_t * C + c_offs,
                mask=c_mask & out_ok,
                other=0.0,
            ).to(tl.float32)
        weight = tl.load(
            w_ptr + c_offs * KERNEL_SIZE + k,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        acc += weight * tap

    # F.conv1d materializes its output dtype before SiLU. MUSA eager SiLU
    # rounds sigmoid back to that dtype before the multiply; a full fp32
    # sigmoid drifts by one ulp and the fused residual add amplifies it.
    conv = acc.to(residual_ptr.dtype.element_ty).to(tl.float32)
    y = conv * tl.sigmoid(conv).to(residual_ptr.dtype.element_ty)
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

    if WRITE_TAIL:
        # One long request: the last STATE_LEN tokens are the new window, and
        # each token program writes its own column. No second launch.
        q_end = tl.load(qsl_ptr + r + 1)
        qlen = (q_end - q_start).to(tl.int32)
        col = j - (qlen - STATE_LEN)
        in_tail = (col >= 0) & (col < STATE_LEN) & state_ok
        col_safe = tl.maximum(col, 0)
        tail = tl.load(
            x_ptr + output_t * C + c_offs,
            mask=c_mask & in_tail,
            other=0.0,
        )
        tl.store(
            base_state + col_safe * state_ws + c_offs * state_cs,
            tail,
            mask=c_mask & in_tail,
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
    # Dilated short convolution with TLE smem reuse for weights / state.
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
        SPEC_QUERY_LEN: tl.constexpr,
        MODE: tl.constexpr,
        HAS_INIT: tl.constexpr,
        NULL_STATE_ID: tl.constexpr,
        FUSE_RESIDUAL: tl.constexpr,
        WRITE_TAIL: tl.constexpr,
        KERNEL_SIZE_SMEM: tl.constexpr,
        STATE_LEN_P2: tl.constexpr,
        STAGE_STATE: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        t = tl.program_id(0)
        pid_c = tl.program_id(1)
        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        c_local = tl.arange(0, BLOCK_C)

        # Stage the channel x kernel weight tile once; every dilation tap
        # reuses the same smem row instead of re-gathering from global.
        w_smem = tle.gpu.alloc(
            [BLOCK_C, KERNEL_SIZE_SMEM],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
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
        if STAGE_STATE:
            s_smem = tle.gpu.alloc(
                [BLOCK_C, STATE_LEN_P2],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
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
        if STAGE_STATE:
            for wi in tl.static_range(0, STATE_LEN):
                s0 = tl.load(
                    base_state + (wi + slot_off) * state_ws + c_offs * state_cs,
                    mask=c_mask & read_state,
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    tle.gpu.local_ptr(
                        s_smem, (c_local, tl.full([BLOCK_C], wi, tl.int32))
                    ),
                    s0,
                    mask=c_mask,
                )

        acc = tl.zeros([BLOCK_C], tl.float32)
        for k in tl.static_range(0, KERNEL_SIZE):
            h = j + DILATION * k
            from_state = h <= STATE_LEN - 1
            if from_state:
                if STAGE_STATE:
                    tap = tl.load(
                        tle.gpu.local_ptr(
                            s_smem, (c_local, tl.full([BLOCK_C], h, tl.int32))
                        ),
                        mask=c_mask & read_state,
                        other=0.0,
                    )
                else:
                    tap = tl.load(
                        base_state + (h + slot_off) * state_ws + c_offs * state_cs,
                        mask=c_mask & read_state,
                        other=0.0,
                    ).to(tl.float32)
            else:
                input_t = q_start + h - STATE_LEN
                input_t = tl.load(
                    token_idx_ptr + input_t,
                    mask=has_token_map & out_ok,
                    other=input_t,
                ).to(tl.int64)
                tap = tl.load(
                    x_ptr + input_t * C + c_offs,
                    mask=c_mask & out_ok,
                    other=0.0,
                ).to(tl.float32)
            weight = tl.load(
                tle.gpu.local_ptr(w_smem, (c_local, tl.full([BLOCK_C], k, tl.int32))),
                mask=c_mask,
                other=0.0,
            )
            acc += weight * tap

        conv = acc.to(residual_ptr.dtype.element_ty).to(tl.float32)
        y = conv * tl.sigmoid(conv).to(residual_ptr.dtype.element_ty)
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

        if WRITE_TAIL:
            q_end = tl.load(qsl_ptr + r + 1)
            qlen = (q_end - q_start).to(tl.int32)
            col = j - (qlen - STATE_LEN)
            in_tail = (col >= 0) & (col < STATE_LEN) & state_ok
            col_safe = tl.maximum(col, 0)
            tail = tl.load(
                x_ptr + output_t * C + c_offs,
                mask=c_mask & in_tail,
                other=0.0,
            )
            tl.store(
                base_state + col_safe * state_ws + c_offs * state_cs,
                tail,
                mask=c_mask & in_tail,
            )

        if MODE == "decode":
            decode_input = tl.load(
                x_ptr + output_t * C + c_offs,
                mask=c_mask & state_ok,
                other=0.0,
            )
            if STAGE_STATE:
                for i in tl.static_range(0, STATE_LEN):
                    if i < STATE_LEN - 1:
                        next_state = tl.load(
                            tle.gpu.local_ptr(
                                s_smem, (c_local, tl.full([BLOCK_C], i + 1, tl.int32))
                            ),
                            mask=c_mask & state_ok & has_init,
                            other=0.0,
                        )
                    else:
                        next_state = decode_input.to(tl.float32)
                    tl.store(
                        base_state + i * state_ws + c_offs * state_cs,
                        next_state.to(state_ptr.dtype.element_ty),
                        mask=c_mask & state_ok,
                    )
            else:
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


# Contiguous [channel, window] writeback. Threads walk the flattened slot so
# the shift is coalesced; the channel-strided loop is much slower on MUSA.
# A split channel reads a later column another program stores, so any launch
# that still needs the old window reads a clone and writes the live cache.
@triton.jit
def _ple_conv_writeback_linear_kernel(
    x_ptr,
    state_ptr,
    out_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    token_idx_ptr,
    has_token_map: tl.constexpr,
    state_idx_stride,
    state_bs,
    out_bs,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    READ_STATE: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    width = STATE_WIDTH
    in_slot = offs < C * width
    channel = offs // width
    w = offs - channel * width

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    sid_safe = tl.where(state_ok, sid, 0)
    q_start = tl.load(qsl_ptr + r)
    q_end = tl.load(qsl_ptr + r + 1)
    qlen = (q_end - q_start).to(tl.int32)
    if MODE == "spec":
        num_acc = tl.load(num_acc_ptr + r)
        slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(
            tl.int32
        )
        shift = 1
        # qlen 0 still rewrites the first STATE_LEN-1 columns and keeps the tail.
        do_write = w < STATE_LEN + qlen - 1
        alive = state_ok
    else:
        slot_off = tl.full([], 0, tl.int32)
        shift = qlen
        do_write = w < STATE_LEN
        alive = state_ok & (qlen > 0)
    has_init = tl.load(has_init_ptr + r) != 0 if HAS_INIT else True
    src_ok = state_ok if MODE == "spec" else state_ok & has_init
    m = shift + w
    from_state = m <= STATE_LEN - 1
    if READ_STATE:
        state_index = tl.minimum(slot_off + m, width - 1)
        state_value = tl.load(
            state_ptr + sid_safe * state_bs + channel * width + state_index,
            mask=in_slot & src_ok & do_write & from_state & alive,
            other=0.0,
        )
    input_t = q_start + m - STATE_LEN
    if has_token_map:
        input_t = tl.load(
            token_idx_ptr + input_t,
            mask=in_slot & do_write & (~from_state) & alive,
            other=input_t,
        ).to(tl.int64)
    input_value = tl.load(
        x_ptr + input_t * C + channel,
        mask=in_slot & do_write & (~from_state) & alive,
        other=0.0,
    )
    if READ_STATE:
        value = tl.where(from_state, state_value, input_value)
    else:
        value = input_value
    tl.store(
        out_ptr + sid_safe * out_bs + channel * width + w,
        value,
        mask=in_slot & do_write & alive,
    )


# Few spec requests are dominated by Triton's host-side enqueue, so the
# shift stays on one program per channel tile. The flattened kernel is
# faster per byte once there are enough requests to hide that enqueue.
@triton.jit
def _ple_conv_spec_shift_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    token_idx_ptr,
    has_token_map: tl.constexpr,
    state_idx_stride,
    state_bs,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
):
    r = tl.program_id(0)
    c_offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    c_mask = c_offs < C
    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    alive = sid != NULL_STATE_ID
    q_start = tl.load(qsl_ptr + r)
    q_end = tl.load(qsl_ptr + r + 1)
    qlen = (q_end - q_start).to(tl.int32)
    num_acc = tl.load(num_acc_ptr + r)
    slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(tl.int32)
    base = state_ptr + sid * state_bs
    for w in tl.static_range(STATE_WIDTH):
        # shift is 1, so this comparison is constexpr and only one tap is issued.
        m = 1 + w
        do_write = alive & (w < STATE_LEN + qlen - 1)
        if m <= STATE_LEN - 1:
            state_index = tl.minimum(slot_off + m, STATE_WIDTH - 1)
            value = tl.load(
                base + c_offs * STATE_WIDTH + state_index,
                mask=c_mask & do_write,
                other=0.0,
            )
        else:
            input_t = q_start + m - STATE_LEN
            if has_token_map:
                input_t = tl.load(
                    token_idx_ptr + input_t,
                    mask=do_write,
                    other=input_t,
                ).to(tl.int64)
            value = tl.load(
                x_ptr + input_t * C + c_offs,
                mask=c_mask & do_write,
                other=0.0,
            )
        tl.store(
            base + c_offs * STATE_WIDTH + w,
            value,
            mask=c_mask & do_write,
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
        if from_state:
            value = tl.load(
                base_state + (slot_off + m) * state_ws + c_offs * state_cs,
                mask=c_mask & src_ok & do_write,
                other=0.0,
            )
        else:
            input_t = q_start + m - STATE_LEN
            input_t = tl.load(
                token_idx_ptr + input_t,
                mask=has_token_map & do_write,
                other=input_t,
            ).to(tl.int64)
            value = tl.load(
                x_ptr + input_t * C + c_offs,
                mask=c_mask & do_write,
                other=0.0,
            )
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
    kernel_spec_query_len = spec_query_len if mode == "spec" else 1
    T, C = inputs.shape
    K = conv_weights.shape[1]
    # Power-of-two smem rows so TLE alloc accepts the tile without padding
    # past the live kernel / state length (extra columns hurt occupancy).
    kernel_size_smem = triton.next_power_of_2(max(K, 1))
    state_len = (K - 1) * dilation
    state_len_p2 = triton.next_power_of_2(max(state_len, 1))
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

    # Tiles are chosen from S5000 sweeps of the nine baseline shapes. A
    # 512-wide tile leaves decode and spec latency-bound; short prefill and
    # long prefill want different widths, and spec warps flip with the
    # request count.
    if mode == "decode":
        block_c = 64
        num_warps = 4
    elif mode == "spec":
        block_c = 128
        num_warps = 4 if num_reqs <= 16 else 2
    elif T < 128:
        block_c, num_warps = 512, 4
    else:
        block_c, num_warps = 256, 2

    # MUSA has no programmatic dependent launch.
    launch_pdl = False
    # Pure-prefill indices can be a strided block-table column view.
    state_idx_stride = state_indices.stride(0)
    contiguous_window = state_ws == 1 and state_cs == state_width
    # Fuse the window copy into the main kernel when a second launch would
    # dominate. Each of the last STATE_LEN token programs writes one column.
    write_tail = (
        contiguous_window
        and mode == "prefill"
        and num_reqs == 1
        and state_len <= T < 128
    )
    # TLE weight / state staging wins on the latency-bound decode and short
    # prefill shapes. Long prefill and both spec sizes keep the baseline
    # kernel: staging regresses the writeback-dominated grids, and small
    # batch spec is already bound by the second launch under the 0.9 bar.
    stage_state_bytes = block_c * (kernel_size_smem + state_len_p2) * 4
    stage_state = HAS_TLE and mode == "decode" and stage_state_bytes <= 49152
    use_tle = HAS_TLE and (mode == "decode" or (mode == "prefill" and T < 128))
    main_kernel = _ple_conv_kernel_tle if use_tle else _ple_conv_kernel_baseline

    # Constexpr flags eliminate accesses to optional None arguments. Absent
    # pointers reuse a device-resident dummy so Triton always receives an address.
    dummy = inputs
    main_kernel[(T, triton.cdiv(C, block_c))](
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
        BLOCK_C=block_c,
        STATE_LEN=state_len,
        DILATION=dilation,
        KERNEL_SIZE=K,
        SPEC_QUERY_LEN=kernel_spec_query_len,
        MODE=mode,
        HAS_INIT=has_initial_states_arg,
        FUSE_RESIDUAL=outer_residual is not None,
        NULL_STATE_ID=NULL_BLOCK_ID,
        WRITE_TAIL=write_tail,
        KERNEL_SIZE_SMEM=kernel_size_smem,
        STATE_LEN_P2=state_len_p2,
        STAGE_STATE=stage_state,
        launch_pdl=launch_pdl,
        num_warps=num_warps,
    )
    # conv state update is fused with the kernel above for decode
    if mode != "decode" and not write_tail:
        if contiguous_window and mode == "prefill" and num_reqs == 1 and T >= state_len:
            _ple_conv_writeback_linear_kernel[
                (num_reqs, triton.cdiv(C * state_width, 128))
            ](
                inputs,
                conv_state,
                conv_state,
                state_indices,
                query_start_loc if query_start_loc is not None else dummy,
                num_accepted_tokens if num_accepted_tokens is not None else dummy,
                has_initial_states if has_initial_states is not None else dummy,
                token_indices if token_indices is not None else state_indices,
                token_indices is not None,
                state_idx_stride,
                state_bs,
                state_bs,
                C=C,
                BLOCK=128,
                STATE_LEN=state_len,
                SPEC_QUERY_LEN=kernel_spec_query_len,
                STATE_WIDTH=state_width,
                MODE=mode,
                HAS_INIT=has_initial_states_arg,
                NULL_STATE_ID=NULL_BLOCK_ID,
                READ_STATE=False,
                num_warps=4,
            )
        elif contiguous_window and mode == "spec" and num_reqs <= 16:
            _ple_conv_spec_shift_kernel[(num_reqs, triton.cdiv(C, 128))](
                inputs,
                conv_state,
                state_indices,
                query_start_loc if query_start_loc is not None else dummy,
                num_accepted_tokens if num_accepted_tokens is not None else dummy,
                token_indices if token_indices is not None else state_indices,
                token_indices is not None,
                state_idx_stride,
                state_bs,
                C=C,
                BLOCK=128,
                STATE_LEN=state_len,
                SPEC_QUERY_LEN=kernel_spec_query_len,
                STATE_WIDTH=state_width,
                NULL_STATE_ID=NULL_BLOCK_ID,
                num_warps=4,
            )
        elif contiguous_window and mode == "spec":
            # Many requests hide the flattened kernel's enqueue cost, and
            # that kernel is much faster per byte than the channel loop.
            # It reads a clone so a split channel cannot race the store.
            read_state = conv_state.clone()
            _ple_conv_writeback_linear_kernel[
                (num_reqs, triton.cdiv(C * state_width, 128))
            ](
                inputs,
                read_state,
                conv_state,
                state_indices,
                query_start_loc if query_start_loc is not None else dummy,
                num_accepted_tokens if num_accepted_tokens is not None else dummy,
                has_initial_states if has_initial_states is not None else dummy,
                token_indices if token_indices is not None else state_indices,
                token_indices is not None,
                state_idx_stride,
                read_state.stride(0),
                state_bs,
                C=C,
                BLOCK=128,
                STATE_LEN=state_len,
                SPEC_QUERY_LEN=kernel_spec_query_len,
                STATE_WIDTH=state_width,
                MODE=mode,
                HAS_INIT=has_initial_states_arg,
                NULL_STATE_ID=NULL_BLOCK_ID,
                READ_STATE=True,
                num_warps=4,
            )
        else:
            _ple_conv_writeback_kernel[(num_reqs, triton.cdiv(C, block_c))](
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
                BLOCK_C=block_c,
                STATE_LEN=state_len,
                SPEC_QUERY_LEN=kernel_spec_query_len,
                STATE_WIDTH=state_width,
                MODE=mode,
                HAS_INIT=has_initial_states_arg,
                NULL_STATE_ID=NULL_BLOCK_ID,
                num_warps=num_warps,
            )


__all__ = ["NULL_BLOCK_ID", "ple_conv"]

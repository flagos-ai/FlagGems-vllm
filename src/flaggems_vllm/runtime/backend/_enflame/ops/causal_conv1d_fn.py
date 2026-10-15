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

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    cs_ptr,
    qsl_ptr,
    ci_ptr,
    hi_ptr,
    dim,
    stride_xd,
    stride_xt,
    stride_wd,
    stride_wk,
    stride_od,
    stride_ot,
    stride_cs0,
    stride_cs1,
    stride_cs2,
    pad_slot_id,
    W: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
    HAS_CI: tl.constexpr,
    HAS_HI: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Derive per-sequence metadata on-device from the raw inputs, avoiding
    # host-side CPU syncs and extra H2D copies.
    s = tl.load(qsl_ptr + pid_s)
    e = tl.load(qsl_ptr + pid_s + 1)
    L = e - s
    if HAS_CI:
        slot = tl.load(ci_ptr + pid_s)
    else:
        slot = pid_s
    if HAS_HI:
        hasinit = tl.load(hi_ptr + pid_s).to(tl.int32)
    else:
        hasinit = 0
    is_pad = slot == pad_slot_id
    safe_slot = tl.where(is_pad, 0, slot)

    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < dim

    if HAS_BIAS:
        b = tl.load(b_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    else:
        b = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Hoist the W=4 weight taps once (each [BLOCK_D]); reused in bulk and tail.
    w0 = tl.load(w_ptr + d * stride_wd + 0 * stride_wk, mask=d_mask, other=0.0).to(
        tl.float32
    )
    w1 = tl.load(w_ptr + d * stride_wd + 1 * stride_wk, mask=d_mask, other=0.0).to(
        tl.float32
    )
    w2 = tl.load(w_ptr + d * stride_wd + 2 * stride_wk, mask=d_mask, other=0.0).to(
        tl.float32
    )
    w3 = tl.load(w_ptr + d * stride_wd + 3 * stride_wk, mask=d_mask, other=0.0).to(
        tl.float32
    )

    # ==== BULK: outputs t in [W-1, L) via non-negative forward index ====
    n_out = L - (W - 1)
    for u0 in range(0, n_out, BLOCK_T):
        u = u0 + tl.arange(0, BLOCK_T)
        um = u < n_out
        dt = d_mask[:, None] & um[None, :]
        tcol = s + u + (W - 1)
        if is_pad:
            # padding slot: output is a straight copy of x[t], no convolution
            xcur = tl.load(
                x_ptr + d[:, None] * stride_xd + tcol[None, :] * stride_xt,
                mask=dt,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                out_ptr + d[:, None] * stride_od + tcol[None, :] * stride_ot,
                xcur.to(out_ptr.dtype.element_ty),
                mask=dt,
            )
        elif W == 4:
            # Shared base pointer for the 4 taps; hoisted weights (w0..w3).
            # Sequential accumulation keeps only ONE tap tile live at a time,
            # cutting register/local spill pressure vs holding all 4 at once.
            base = x_ptr + d[:, None] * stride_xd + (s + u)[None, :] * stride_xt
            acc = w0[:, None] * tl.load(base + 0 * stride_xt, mask=dt, other=0.0).to(
                tl.float32
            )
            acc += w1[:, None] * tl.load(base + 1 * stride_xt, mask=dt, other=0.0).to(
                tl.float32
            )
            acc += w2[:, None] * tl.load(base + 2 * stride_xt, mask=dt, other=0.0).to(
                tl.float32
            )
            acc += w3[:, None] * tl.load(base + 3 * stride_xt, mask=dt, other=0.0).to(
                tl.float32
            )
            acc += b[:, None]
            if ACT:
                acc = acc * tl.sigmoid(acc)
            tl.store(
                out_ptr + d[:, None] * stride_od + tcol[None, :] * stride_ot,
                acc.to(out_ptr.dtype.element_ty),
                mask=dt,
            )
        else:
            acc = tl.zeros((BLOCK_D, BLOCK_T), dtype=tl.float32)
            for k in tl.static_range(W):
                wk = tl.load(
                    w_ptr + d * stride_wd + k * stride_wk, mask=d_mask, other=0.0
                ).to(tl.float32)
                col = s + u + k
                xv = tl.load(
                    x_ptr + d[:, None] * stride_xd + col[None, :] * stride_xt,
                    mask=dt,
                    other=0.0,
                ).to(tl.float32)
                acc += wk[:, None] * xv
            acc += b[:, None]
            if ACT:
                acc = acc * tl.sigmoid(acc)
            tl.store(
                out_ptr + d[:, None] * stride_od + tcol[None, :] * stride_ot,
                acc.to(out_ptr.dtype.element_ty),
                mask=dt,
            )

    # ==== TAIL: W-1 boundary outputs + conv_states update ====
    hmask = d_mask & (hasinit != 0) & (~is_pad)

    if W == 4:
        # ---- Fast path: weights already hoisted above ----
        c0 = tl.load(
            cs_ptr + safe_slot * stride_cs0 + d * stride_cs1 + 0 * stride_cs2,
            mask=hmask,
            other=0.0,
        ).to(tl.float32)
        c1 = tl.load(
            cs_ptr + safe_slot * stride_cs0 + d * stride_cs1 + 1 * stride_cs2,
            mask=hmask,
            other=0.0,
        ).to(tl.float32)
        c2 = tl.load(
            cs_ptr + safe_slot * stride_cs0 + d * stride_cs1 + 2 * stride_cs2,
            mask=hmask,
            other=0.0,
        ).to(tl.float32)

        xh0 = tl.load(
            x_ptr + d * stride_xd + (s + 0) * stride_xt,
            mask=d_mask & (0 < L),
            other=0.0,
        ).to(tl.float32)
        xh1 = tl.load(
            x_ptr + d * stride_xd + (s + 1) * stride_xt,
            mask=d_mask & (1 < L),
            other=0.0,
        ).to(tl.float32)
        xh2 = tl.load(
            x_ptr + d * stride_xd + (s + 2) * stride_xt,
            mask=d_mask & (2 < L),
            other=0.0,
        ).to(tl.float32)
        # extended row ex = [c0, c1, c2, xh0, xh1, xh2]; boundary tt in {0,1,2}
        a0 = w0 * c0 + w1 * c1 + w2 * c2 + w3 * xh0 + b
        a1 = w0 * c1 + w1 * c2 + w2 * xh0 + w3 * xh1 + b
        a2 = w0 * c2 + w1 * xh0 + w2 * xh1 + w3 * xh2 + b
        if ACT:
            a0 = a0 * tl.sigmoid(a0)
            a1 = a1 * tl.sigmoid(a1)
            a2 = a2 * tl.sigmoid(a2)
        r0 = tl.where(is_pad, xh0, a0)
        r1 = tl.where(is_pad, xh1, a1)
        r2 = tl.where(is_pad, xh2, a2)
        tl.store(
            out_ptr + d * stride_od + (s + 0) * stride_ot,
            r0.to(out_ptr.dtype.element_ty),
            mask=d_mask & (0 < L),
        )
        tl.store(
            out_ptr + d * stride_od + (s + 1) * stride_ot,
            r1.to(out_ptr.dtype.element_ty),
            mask=d_mask & (1 < L),
        )
        tl.store(
            out_ptr + d * stride_od + (s + 2) * stride_ot,
            r2.to(out_ptr.dtype.element_ty),
            mask=d_mask & (2 < L),
        )
    else:
        # ---- General fallback for W != 4 ----
        for tt in range(W - 1):
            acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for k in tl.static_range(W):
                m = tt + k  # compile-time extended position
                wk = tl.load(
                    w_ptr + d * stride_wd + k * stride_wk, mask=d_mask, other=0.0
                ).to(tl.float32)
                if m < W - 1:
                    ev = tl.load(
                        cs_ptr
                        + safe_slot * stride_cs0
                        + d * stride_cs1
                        + m * stride_cs2,
                        mask=hmask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    n = m - (W - 1)
                    ev = tl.load(
                        x_ptr + d * stride_xd + (s + n) * stride_xt,
                        mask=d_mask & (n < L),
                        other=0.0,
                    ).to(tl.float32)
                acc += wk * ev
            acc += b
            if ACT:
                acc = acc * tl.sigmoid(acc)
            xpass = tl.load(
                x_ptr + d * stride_xd + (s + tt) * stride_xt,
                mask=d_mask & (tt < L),
                other=0.0,
            ).to(tl.float32)
            res = tl.where(is_pad, xpass, acc)
            tl.store(
                out_ptr + d * stride_od + (s + tt) * stride_ot,
                res.to(out_ptr.dtype.element_ty),
                mask=d_mask & (tt < L),
            )

    # state update: conv_states[slot][:, j] = e_x[L + j], j in [0, W-2]
    # When L >= W-1, p = L + j - (W-1) >= 0 for every j, so the whole new state
    # comes from x and the conv_states reload/select is dead work. Branch on
    # that common case at runtime to drop the csval load + tl.where.
    if L >= W - 1:
        for j in tl.static_range(W - 1):
            p = L + j - (W - 1)
            col = s + p
            xval = tl.load(
                x_ptr + d * stride_xd + col * stride_xt, mask=d_mask, other=0.0
            ).to(tl.float32)
            outptr = cs_ptr + slot * stride_cs0 + d * stride_cs1 + j * stride_cs2
            tl.store(outptr, xval.to(cs_ptr.dtype.element_ty), mask=d_mask & (~is_pad))
    else:
        for j in tl.static_range(W - 1):
            p = L + j - (W - 1)
            col = s + p
            xval = tl.load(
                x_ptr + d * stride_xd + col * stride_xt,
                mask=d_mask & (p >= 0),
                other=0.0,
            ).to(tl.float32)
            cs_col = (W - 1) + p
            csptr = (
                cs_ptr + safe_slot * stride_cs0 + d * stride_cs1 + cs_col * stride_cs2
            )
            csval = tl.load(
                csptr, mask=d_mask & (p < 0) & (hasinit != 0), other=0.0
            ).to(tl.float32)
            val = tl.where(p >= 0, xval, csval)
            outptr = cs_ptr + slot * stride_cs0 + d * stride_cs1 + j * stride_cs2
            tl.store(outptr, val.to(cs_ptr.dtype.element_ty), mask=d_mask & (~is_pad))


def causal_conv1d_fn(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=-1,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    num_computed_tokens=None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, 'silu', or 'swish'")

    dim, total_seq = x.shape
    width = weight.shape[1]
    out = torch.empty_like(x)

    # Fully async launch: derive everything from tensor shapes, no CPU sync.
    num_seq = query_start_loc.shape[0] - 1
    if num_seq <= 0:
        return out

    # Sync-free tile selector: average sequence length (total tokens / num_seq).
    # cache_indices / has_initial_state are passed straight to the kernel and
    # indexed on-device, avoiding syncs and H2D copies per call.
    max_len = total_seq // num_seq

    qsl_i32 = query_start_loc.to(torch.int32)
    HAS_CI = cache_indices is not None
    HAS_HI = has_initial_state is not None
    ci_arg = cache_indices if HAS_CI else query_start_loc
    hi_arg = has_initial_state if HAS_HI else query_start_loc

    HAS_BIAS = bias is not None
    b_ptr = bias if bias is not None else weight
    ACT = 1 if activation in ("silu", "swish") else 0

    # Sequence-length-specialized tiling. On-device do_bench sweeps on
    # ZIXIAOC200 showed num_warps=1 dominates for this tiny-arithmetic
    # depthwise conv: short seqs favor 256x256, long seqs 128x512.
    if max_len <= 512:
        BLOCK_D = 256
        BLOCK_T = 256
    else:
        BLOCK_D = 128
        BLOCK_T = 512

    grid = (num_seq, triton.cdiv(dim, BLOCK_D))
    _fused_kernel[grid](
        x,
        weight,
        b_ptr,
        out,
        conv_states,
        qsl_i32,
        ci_arg,
        hi_arg,
        dim,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        pad_slot_id,
        W=width,
        BLOCK_D=BLOCK_D,
        BLOCK_T=BLOCK_T,
        HAS_BIAS=HAS_BIAS,
        ACT=ACT,
        HAS_CI=HAS_CI,
        HAS_HI=HAS_HI,
        num_warps=1,
    )

    return out

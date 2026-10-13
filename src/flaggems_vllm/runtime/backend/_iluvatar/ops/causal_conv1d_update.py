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
def _causal_conv1d_update_kernel(
    x_ptr,
    cs_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B,
    D,
    sx_b,
    sx_d,
    sx_s,
    scs_b,
    scs_d,
    scs_l,
    sw_d,
    sw_k,
    sout_b,
    sout_d,
    sout_s,
    STATE_LEN: tl.constexpr,
    SEQLEN: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATE: tl.constexpr,
):
    # Swap grid dims: pid_d first for potentially better cache reuse in D dim.
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    cs_base = cs_ptr + pid_b * scs_b + d_offs * scs_d
    x_base = x_ptr + pid_b * sx_b + d_offs * sx_d
    w_base = w_ptr + d_offs * sw_d
    out_base = out_ptr + pid_b * sout_b + d_offs * sout_d

    # --- Pre-load weight taps (BLOCK_D per k) as fp32, once per program ---
    w0 = (
        tl.load(w_base + 0 * sw_k, mask=d_mask, other=0.0).to(tl.float32)
        if WIDTH > 0
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )
    w1 = (
        tl.load(w_base + 1 * sw_k, mask=d_mask, other=0.0).to(tl.float32)
        if WIDTH > 1
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )
    w2 = (
        tl.load(w_base + 2 * sw_k, mask=d_mask, other=0.0).to(tl.float32)
        if WIDTH > 2
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )
    w3 = (
        tl.load(w_base + 3 * sw_k, mask=d_mask, other=0.0).to(tl.float32)
        if WIDTH > 3
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )

    # --- Pre-load state values as fp32 ---
    s0 = (
        tl.load(cs_base + 0 * scs_l, mask=d_mask, other=0.0).to(tl.float32)
        if STATE_LEN > 0
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )
    s1 = (
        tl.load(cs_base + 1 * scs_l, mask=d_mask, other=0.0).to(tl.float32)
        if STATE_LEN > 1
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )
    s2 = (
        tl.load(cs_base + 2 * scs_l, mask=d_mask, other=0.0).to(tl.float32)
        if STATE_LEN > 2
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )

    # --- Pre-load x values as fp32 ---
    x0 = (
        tl.load(x_base + 0 * sx_s, mask=d_mask, other=0.0).to(tl.float32)
        if SEQLEN > 0
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )
    x1 = (
        tl.load(x_base + 1 * sx_s, mask=d_mask, other=0.0).to(tl.float32)
        if SEQLEN > 1
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )
    x2 = (
        tl.load(x_base + 2 * sx_s, mask=d_mask, other=0.0).to(tl.float32)
        if SEQLEN > 2
        else tl.zeros((BLOCK_D,), dtype=tl.float32)
    )

    # --- Bias ---
    if HAS_BIAS:
        bias_v = tl.load(b_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    else:
        bias_v = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # ---- Phase A: compute output for each t ----
    for t in tl.static_range(SEQLEN):
        acc = bias_v
        for k in tl.static_range(WIDTH):
            p = STATE_LEN - WIDTH + 1 + t + k
            wv = w0
            if k == 1:
                wv = w1
            elif k == 2:
                wv = w2
            elif k == 3:
                wv = w3

            if p < 0:
                val = tl.zeros((BLOCK_D,), dtype=tl.float32)
            elif p == 0:
                val = s0
            elif p == 1:
                val = s1
            elif p == 2:
                val = s2
            elif p == 3:
                if STATE_LEN > 3:
                    val = tl.zeros((BLOCK_D,), dtype=tl.float32)
                else:
                    val = x0
            elif p == 4:
                if STATE_LEN > 4:
                    val = tl.zeros((BLOCK_D,), dtype=tl.float32)
                elif STATE_LEN == 4:
                    val = tl.zeros((BLOCK_D,), dtype=tl.float32)
                else:
                    val = x1
            else:  # p == 5
                val = x2

            acc = acc + wv * val

        if ACTIVATE:
            acc = acc * tl.sigmoid(acc)

        tl.store(out_base + t * sout_s, acc.to(out_ptr.dtype.element_ty), mask=d_mask)

    # ---- Phase B: state update ----
    for j in tl.static_range(STATE_LEN):
        p = SEQLEN + j
        if p == 0:
            v = s0
        elif p == 1:
            v = s1
        elif p == 2:
            v = s2
        elif p == 3:
            if STATE_LEN > 3:
                v = tl.zeros((BLOCK_D,), dtype=tl.float32)
            else:
                v = x0
        elif p == 4:
            if STATE_LEN > 4:
                v = tl.zeros((BLOCK_D,), dtype=tl.float32)
            else:
                v = x1
        else:  # p == 5
            v = x2
        tl.store(cs_base + j * scs_l, v.to(cs_ptr.dtype.element_ty), mask=d_mask)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _pick_block_d(dim: int) -> int:
    if dim >= 256:
        return 256
    return max(_next_pow2(dim), 1)


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1,
    pad_slot_id=-1,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    validate_data=False,
):
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, 'silu', or 'swish'")

    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    x_local = x.unsqueeze(-1) if unsqueeze else x
    batch, dim, seqlen = x_local.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]

    out = torch.empty((batch, dim, seqlen), dtype=dtype_in, device=x.device)

    has_bias = bias is not None
    b_ptr = bias if has_bias else x_local  # dummy tensor when unused

    activate = 1 if activation is not None else 0

    BLOCK_D = _pick_block_d(dim)
    # Grid: (d_tiles, batch) so consecutive programs share batch for L2 reuse of weight.
    grid = (triton.cdiv(dim, BLOCK_D), batch)

    _causal_conv1d_update_kernel[grid](
        x_local,
        conv_state,
        weight,
        b_ptr,
        out,
        batch,
        dim,
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        STATE_LEN=state_len,
        SEQLEN=seqlen,
        WIDTH=width,
        BLOCK_D=BLOCK_D,
        HAS_BIAS=has_bias,
        ACTIVATE=activate,
        num_warps=4,
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out

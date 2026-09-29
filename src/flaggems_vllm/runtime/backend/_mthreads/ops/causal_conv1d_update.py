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
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    # x: (B, D, S)
    stride_x_b,
    stride_x_d,
    stride_x_s,
    # conv_state: (B, D, L)
    stride_cs_b,
    stride_cs_d,
    stride_cs_l,
    # weight: (D, W)
    stride_w_d,
    stride_w_k,
    # out: (B, D, S)
    stride_out_b,
    stride_out_d,
    stride_out_s,
    D,
    BLOCK_D: tl.constexpr,
    S: tl.constexpr,
    L: tl.constexpr,
    W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_dtile = tl.program_id(1)

    d_off = pid_dtile * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    # Base pointers into (b, d)
    x_bd = x_ptr + pid_b * stride_x_b + d_off * stride_x_d
    cs_bd = conv_state_ptr + pid_b * stride_cs_b + d_off * stride_cs_d
    w_bd = weight_ptr + d_off * stride_w_d
    out_bd = out_ptr + pid_b * stride_out_b + d_off * stride_out_d

    # Load weight[d, 0:W] -> shape (BLOCK_D, W) as fp32; we keep in scalars for unrolling
    # Load conv_state[b, d, 0:L] into L scalars (each BLOCK_D wide)
    # Load x[b, d, 0:S] into S scalars (each BLOCK_D wide)

    # We'll rely on Python-level unrolling by indexing constexpr positions.
    # Build x_new list: length L + S. Then compute out[t] = bias + sum_k w[k] * x_new[t+k]
    # Then update conv_state: state[i] = x_new[i + S] for i in [0, L)

    # Load bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Load conv_state[b, d, i] for i in [0, L)
    state_0 = tl.load(cs_bd + 0 * stride_cs_l, mask=d_mask, other=0.0).to(tl.float32)
    if L >= 2:
        state_1 = tl.load(cs_bd + 1 * stride_cs_l, mask=d_mask, other=0.0).to(
            tl.float32
        )
    else:
        state_1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if L >= 3:
        state_2 = tl.load(cs_bd + 2 * stride_cs_l, mask=d_mask, other=0.0).to(
            tl.float32
        )
    else:
        state_2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if L >= 4:
        state_3 = tl.load(cs_bd + 3 * stride_cs_l, mask=d_mask, other=0.0).to(
            tl.float32
        )
    else:
        state_3 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Load x[b, d, t] for t in [0, S)
    x_0 = tl.load(x_bd + 0 * stride_x_s, mask=d_mask, other=0.0).to(tl.float32)
    if S >= 2:
        x_1 = tl.load(x_bd + 1 * stride_x_s, mask=d_mask, other=0.0).to(tl.float32)
    else:
        x_1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if S >= 3:
        x_2 = tl.load(x_bd + 2 * stride_x_s, mask=d_mask, other=0.0).to(tl.float32)
    else:
        x_2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if S >= 4:
        x_3 = tl.load(x_bd + 3 * stride_x_s, mask=d_mask, other=0.0).to(tl.float32)
    else:
        x_3 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Load weight[d, k] for k in [0, W)
    w_0 = tl.load(w_bd + 0 * stride_w_k, mask=d_mask, other=0.0).to(tl.float32)
    if W >= 2:
        w_1 = tl.load(w_bd + 1 * stride_w_k, mask=d_mask, other=0.0).to(tl.float32)
    else:
        w_1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if W >= 3:
        w_2 = tl.load(w_bd + 2 * stride_w_k, mask=d_mask, other=0.0).to(tl.float32)
    else:
        w_2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if W >= 4:
        w_3 = tl.load(w_bd + 3 * stride_w_k, mask=d_mask, other=0.0).to(tl.float32)
    else:
        w_3 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # x_new indexed by i in [0, L+S):
    # i in [0, L): state[i]; i in [L, L+S): x[i - L]
    # We only need x_new[t + k] for t in [0, S), k in [0, W).
    # With W=4, S<=3, L<=3, indexes t+k in [0, S+W-1] = [0, S+3].
    # For state_len == W-1 == 3, valid indices land inside [0, L+S).

    # Compute output for each t (constexpr).
    # out[t] = bias + sum_k w[k] * x_new[t + k]. All comparisons are constexpr,
    # so branches collapse at compile time and reduce to straight-line code.

    # t = 0
    if S >= 1:
        acc0 = bias
        # k = 0: idx = 0
        acc0 += w_0 * (
            state_0
            if 0 < L
            else (x_0 if 0 - L == 0 else tl.zeros((BLOCK_D,), dtype=tl.float32))
        )
        if W >= 2:
            # k = 1: idx = 1
            if 1 < L:
                acc0 += w_1 * state_1
            else:
                acc0 += w_1 * (
                    x_0
                    if (1 - L) == 0
                    else (
                        x_1 if (1 - L) == 1 else tl.zeros((BLOCK_D,), dtype=tl.float32)
                    )
                )
        if W >= 3:
            # k = 2: idx = 2
            if 2 < L:
                acc0 += w_2 * state_2
            else:
                if (2 - L) == 0:
                    acc0 += w_2 * x_0
                elif (2 - L) == 1:
                    acc0 += w_2 * x_1
                elif (2 - L) == 2:
                    acc0 += w_2 * x_2
        if W >= 4:
            # k = 3: idx = 3
            if 3 < L:
                acc0 += w_3 * state_3
            else:
                if (3 - L) == 0:
                    acc0 += w_3 * x_0
                elif (3 - L) == 1:
                    acc0 += w_3 * x_1
                elif (3 - L) == 2:
                    acc0 += w_3 * x_2
                elif (3 - L) == 3:
                    acc0 += w_3 * x_3
        if ACTIVATION:
            acc0 = acc0 * tl.sigmoid(acc0)
        tl.store(out_bd + 0 * stride_out_s, acc0, mask=d_mask)

    if S >= 2:
        acc1 = bias
        # k = 0: idx = 1
        if 1 < L:
            acc1 += w_0 * state_1
        else:
            if (1 - L) == 0:
                acc1 += w_0 * x_0
        if W >= 2:
            # k = 1: idx = 2
            if 2 < L:
                acc1 += w_1 * state_2
            else:
                if (2 - L) == 0:
                    acc1 += w_1 * x_0
                elif (2 - L) == 1:
                    acc1 += w_1 * x_1
        if W >= 3:
            # k = 2: idx = 3
            if 3 < L:
                acc1 += w_2 * state_3
            else:
                if (3 - L) == 0:
                    acc1 += w_2 * x_0
                elif (3 - L) == 1:
                    acc1 += w_2 * x_1
                elif (3 - L) == 2:
                    acc1 += w_2 * x_2
        if W >= 4:
            # k = 3: idx = 4
            if 4 < L:
                pass  # not possible for L<=4
            else:
                if (4 - L) == 0:
                    acc1 += w_3 * x_0
                elif (4 - L) == 1:
                    acc1 += w_3 * x_1
                elif (4 - L) == 2:
                    acc1 += w_3 * x_2
                elif (4 - L) == 3:
                    acc1 += w_3 * x_3
        if ACTIVATION:
            acc1 = acc1 * tl.sigmoid(acc1)
        tl.store(out_bd + 1 * stride_out_s, acc1, mask=d_mask)

    if S >= 3:
        acc2 = bias
        # k = 0: idx = 2
        if 2 < L:
            acc2 += w_0 * state_2
        else:
            if (2 - L) == 0:
                acc2 += w_0 * x_0
            elif (2 - L) == 1:
                acc2 += w_0 * x_1
        if W >= 2:
            # k = 1: idx = 3
            if 3 < L:
                acc2 += w_1 * state_3
            else:
                if (3 - L) == 0:
                    acc2 += w_1 * x_0
                elif (3 - L) == 1:
                    acc2 += w_1 * x_1
                elif (3 - L) == 2:
                    acc2 += w_1 * x_2
        if W >= 3:
            # k = 2: idx = 4
            if (4 - L) == 0:
                acc2 += w_2 * x_0
            elif (4 - L) == 1:
                acc2 += w_2 * x_1
            elif (4 - L) == 2:
                acc2 += w_2 * x_2
            elif (4 - L) == 3:
                acc2 += w_2 * x_3
        if W >= 4:
            # k = 3: idx = 5
            if (5 - L) == 0:
                acc2 += w_3 * x_0
            elif (5 - L) == 1:
                acc2 += w_3 * x_1
            elif (5 - L) == 2:
                acc2 += w_3 * x_2
            elif (5 - L) == 3:
                acc2 += w_3 * x_3
        if ACTIVATION:
            acc2 = acc2 * tl.sigmoid(acc2)
        tl.store(out_bd + 2 * stride_out_s, acc2, mask=d_mask)

    if S >= 4:
        acc3 = bias
        # k=0: idx=3
        if 3 < L:
            acc3 += w_0 * state_3
        else:
            if (3 - L) == 0:
                acc3 += w_0 * x_0
            elif (3 - L) == 1:
                acc3 += w_0 * x_1
            elif (3 - L) == 2:
                acc3 += w_0 * x_2
        if W >= 2:
            # k=1: idx=4
            if (4 - L) == 0:
                acc3 += w_1 * x_0
            elif (4 - L) == 1:
                acc3 += w_1 * x_1
            elif (4 - L) == 2:
                acc3 += w_1 * x_2
            elif (4 - L) == 3:
                acc3 += w_1 * x_3
        if W >= 3:
            # k=2: idx=5
            if (5 - L) == 0:
                acc3 += w_2 * x_0
            elif (5 - L) == 1:
                acc3 += w_2 * x_1
            elif (5 - L) == 2:
                acc3 += w_2 * x_2
            elif (5 - L) == 3:
                acc3 += w_2 * x_3
        if W >= 4:
            # k=3: idx=6
            if (6 - L) == 0:
                acc3 += w_3 * x_0
            elif (6 - L) == 1:
                acc3 += w_3 * x_1
            elif (6 - L) == 2:
                acc3 += w_3 * x_2
            elif (6 - L) == 3:
                acc3 += w_3 * x_3
        if ACTIVATION:
            acc3 = acc3 * tl.sigmoid(acc3)
        tl.store(out_bd + 3 * stride_out_s, acc3, mask=d_mask)

    # State update: new_state[i] = x_new[i + S] for i in [0, L).
    # Choose the source register based on constexpr indices; initialize `v` to
    # zero first so Triton's IR builder sees a definition on every path.
    if L >= 1:
        i = 0
        idx = i + S
        v = tl.zeros((BLOCK_D,), dtype=tl.float32)
        if idx == 0:
            v = state_0
        elif idx == 1:
            v = state_1 if 1 < L else x_0
        elif idx == 2:
            v = state_2 if 2 < L else (x_0 if (2 - L) == 0 else x_1)
        elif idx == 3:
            if 3 < L:
                v = state_3
            else:
                if (3 - L) == 0:
                    v = x_0
                elif (3 - L) == 1:
                    v = x_1
                elif (3 - L) == 2:
                    v = x_2
                else:
                    v = x_3
        elif idx == 4:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 5:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 6:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        tl.store(cs_bd + 0 * stride_cs_l, v, mask=d_mask)

    if L >= 2:
        i = 1
        idx = i + S
        v = tl.zeros((BLOCK_D,), dtype=tl.float32)
        if idx == 1:
            v = state_1
        elif idx == 2:
            v = state_2 if 2 < L else x_0
        elif idx == 3:
            if 3 < L:
                v = state_3
            else:
                if (3 - L) == 0:
                    v = x_0
                elif (3 - L) == 1:
                    v = x_1
                elif (3 - L) == 2:
                    v = x_2
                else:
                    v = x_3
        elif idx == 4:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 5:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 6:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 7:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        tl.store(cs_bd + 1 * stride_cs_l, v, mask=d_mask)

    if L >= 3:
        i = 2
        idx = i + S
        v = tl.zeros((BLOCK_D,), dtype=tl.float32)
        if idx == 2:
            v = state_2
        elif idx == 3:
            v = state_3 if 3 < L else x_0
        elif idx == 4:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 5:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 6:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 7:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 8:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        tl.store(cs_bd + 2 * stride_cs_l, v, mask=d_mask)

    if L >= 4:
        i = 3
        idx = i + S
        v = tl.zeros((BLOCK_D,), dtype=tl.float32)
        if idx == 3:
            v = state_3
        elif idx == 4:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 5:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 6:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        elif idx == 7:
            xi = idx - L
            if xi == 0:
                v = x_0
            elif xi == 1:
                v = x_1
            elif xi == 2:
                v = x_2
            else:
                v = x_3
        tl.store(cs_bd + 3 * stride_cs_l, v, mask=d_mask)


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
    del (
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        pad_slot_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
        validate_data,
    )
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, 'silu', or 'swish'")

    unsqueeze = x.dim() == 2
    x_local = x.unsqueeze(-1) if unsqueeze else x
    B, D, S = x_local.shape
    W = weight.shape[1]
    L = conv_state.shape[-1]

    out = torch.empty_like(x_local)

    # BLOCK_D heuristic
    def _next_pow2(n):
        p = 1
        while p < n:
            p <<= 1
        return p

    BLOCK_D = min(_next_pow2(D), 128)
    if BLOCK_D < 16:
        BLOCK_D = 16

    grid = (B, triton.cdiv(D, BLOCK_D))

    _causal_conv1d_update_kernel[grid](
        x_local,
        conv_state,
        weight,
        bias if bias is not None else x_local,  # dummy pointer if no bias
        out,
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
        D,
        BLOCK_D=BLOCK_D,
        S=S,
        L=L,
        W=W,
        HAS_BIAS=(bias is not None),
        ACTIVATION=(activation is not None),
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out

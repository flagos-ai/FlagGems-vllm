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
def _conv1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    cs_ptr,
    out_ptr,
    qsl_ptr,
    ci_ptr,
    hi_ptr,
    dim,
    total_seq,
    stride_x0,
    stride_x1,
    stride_w0,
    stride_w1,
    stride_cs0,
    stride_cs1,
    stride_cs2,
    stride_o0,
    stride_o1,
    W: tl.constexpr,
    PAD_SLOT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    seq_id = tl.program_id(2)
    c_id = tl.program_id(1)
    t_id = tl.program_id(0)

    s = tl.load(qsl_ptr + seq_id)
    e = tl.load(qsl_ptr + seq_id + 1)
    L = e - s

    t_start = t_id * BLOCK_T
    if t_start >= L:
        return

    slot = tl.load(ci_ptr + seq_id)

    if HAS_INIT:
        has_init_i = tl.load(hi_ptr + seq_id).to(tl.int1)
    else:
        has_init_i = False

    offs_t = t_start + tl.arange(0, BLOCK_T)
    offs_c = c_id * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < dim
    mask_t = offs_t < L

    if HAS_BIAS:
        bias_v = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    if HAS_PAD:
        slot_safe = tl.maximum(slot, 0)
        is_pad_scalar = slot == PAD_SLOT
    else:
        slot_safe = slot
        is_pad_scalar = False

    # Preload all W weight vectors into registers
    acc = tl.zeros((BLOCK_C, BLOCK_T), dtype=tl.float32)

    # Iterate over W taps
    for k in tl.static_range(W):
        # Load weight column k as (BLOCK_C,)
        w_k_offs = offs_c * stride_w0 + k * stride_w1
        w_k = tl.load(w_ptr + w_k_offs, mask=mask_c, other=0.0).to(tl.float32)

        # position within sequence for tap k contributing to output p:
        # y[p] = sum_k w[k] * ctx[p + k - (W-1)]
        pos = offs_t + k - (W - 1)  # (BLOCK_T,), int32 sequence-relative
        # x branch: pos in [0, L)
        x_ok = (pos >= 0) & (pos < L)
        x_col = s + pos  # int32
        x_offs = offs_c[:, None] * stride_x0 + x_col[None, :] * stride_x1
        x_load_mask = mask_c[:, None] & x_ok[None, :]
        x_val = tl.load(x_ptr + x_offs, mask=x_load_mask, other=0.0).to(tl.float32)

        # initial state branch: only relevant when t_start == 0 (interior tiles never hit it)
        if HAS_INIT:
            if t_start < (W - 1):
                state_col = (W - 1) + pos  # in [0, W-2] when pos < 0
                state_ok = (pos < 0) & has_init_i
                if HAS_PAD:
                    state_ok = state_ok & (slot != PAD_SLOT)
                state_offs = (
                    slot_safe * stride_cs0
                    + offs_c[:, None] * stride_cs1
                    + state_col[None, :] * stride_cs2
                )
                state_mask = mask_c[:, None] & state_ok[None, :]
                state_val = tl.load(cs_ptr + state_offs, mask=state_mask, other=0.0).to(
                    tl.float32
                )
                x_val = x_val + state_val

        acc = acc + w_k[:, None] * x_val

    if HAS_BIAS:
        acc = acc + bias_v[:, None]

    if ACT == 1:
        acc = acc * (1.0 / (1.0 + tl.exp(-acc)))

    # Pad-slot passthrough (identity copy of x, no bias/activation)
    if HAS_PAD:
        x_raw_offs = offs_c[:, None] * stride_x0 + (s + offs_t)[None, :] * stride_x1
        raw_mask = mask_c[:, None] & mask_t[None, :]
        x_raw = tl.load(x_ptr + x_raw_offs, mask=raw_mask, other=0.0).to(tl.float32)
        acc = tl.where(is_pad_scalar, x_raw, acc)

    # Store output
    out_mask = mask_c[:, None] & mask_t[None, :]
    out_offs = offs_c[:, None] * stride_o0 + (s + offs_t)[None, :] * stride_o1
    tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _state_update_kernel(
    x_ptr,
    cs_ptr,
    qsl_ptr,
    ci_ptr,
    hi_ptr,
    dim,
    stride_x0,
    stride_x1,
    stride_cs0,
    stride_cs1,
    stride_cs2,
    W: tl.constexpr,
    PAD_SLOT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    seq_id = tl.program_id(0)
    c_id = tl.program_id(1)

    slot = tl.load(ci_ptr + seq_id)
    if HAS_PAD:
        if slot == PAD_SLOT:
            return

    s = tl.load(qsl_ptr + seq_id)
    e = tl.load(qsl_ptr + seq_id + 1)
    L = e - s

    if HAS_INIT:
        has_init_i = tl.load(hi_ptr + seq_id).to(tl.int1)
    else:
        has_init_i = False

    offs_c = c_id * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < dim

    W_MINUS_1: tl.constexpr = W - 1
    for j in tl.static_range(W_MINUS_1):
        # target column j in [0, W-2] of conv_states[slot]
        # source semantics: last (W-1) elements of cat([init if has_init else empty, x[:,s:e]])
        # x_col relative to sequence: L - (W-1) + j
        x_col = L - (W - 1) + j  # int32 scalar
        x_ok = x_col >= 0

        # x path
        x_offs = offs_c * stride_x0 + (s + x_col) * stride_x1
        x_mask = mask_c & x_ok
        x_val = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        # init path: only when x_col < 0, and has_init
        if HAS_INIT:
            init_col = L + j  # only valid when x_col < 0
            init_ok = (x_col < 0) & has_init_i
            init_offs = slot * stride_cs0 + offs_c * stride_cs1 + init_col * stride_cs2
            init_mask = mask_c & init_ok
            init_val = tl.load(cs_ptr + init_offs, mask=init_mask, other=0.0)
        else:
            init_val = tl.zeros_like(x_val)

        val = x_val + init_val

        state_offs = slot * stride_cs0 + offs_c * stride_cs1 + j * stride_cs2
        tl.store(cs_ptr + state_offs, val, mask=mask_c)


def _activation_code(activation):
    if activation is None:
        return 0
    if activation in ("silu", "swish"):
        return 1
    raise NotImplementedError("activation must be None, 'silu', or 'swish'")


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
    del (
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        block_size_to_align,
        metadata,
        validate_data,
    )

    act_code = _activation_code(activation)
    dim, total_seq = x.shape
    W = int(weight.shape[1])
    device = x.device

    # boundaries: small tensor, host-side transfer is cheap
    qsl_cpu = query_start_loc.detach().to(torch.int64).cpu().tolist()
    num_seq = len(qsl_cpu) - 1
    if num_seq <= 0:
        return torch.empty((dim, 0), dtype=x.dtype, device=device)

    max_seqlen = 0
    for i in range(num_seq):
        L = qsl_cpu[i + 1] - qsl_cpu[i]
        if L > max_seqlen:
            max_seqlen = L

    # Materialize defaults for optional tensors
    if cache_indices is None:
        cache_indices_t = torch.arange(num_seq, dtype=torch.int32, device=device)
    else:
        cache_indices_t = cache_indices

    HAS_INIT = has_initial_state is not None
    if has_initial_state is not None:
        hi_t = has_initial_state
    else:
        hi_t = torch.empty((num_seq,), dtype=torch.bool, device=device)

    HAS_BIAS = bias is not None
    if bias is not None:
        bias_t = bias
    else:
        bias_t = torch.empty((1,), dtype=x.dtype, device=device)

    # Pad-slot detection: quickly check on host via small transfer
    ci_cpu = cache_indices_t.detach().to(torch.int64).cpu().tolist()
    HAS_PAD = any(sl == pad_slot_id for sl in ci_cpu)

    out = torch.empty_like(x)

    if total_seq == 0 or max_seqlen == 0:
        return out

    # Grid config -- pick tile shape by max sequence length
    if max_seqlen >= 512:
        BLOCK_C = 2
        BLOCK_T = 1024
        num_warps = 4
    elif max_seqlen >= 128:
        BLOCK_C = 4
        BLOCK_T = 256
        num_warps = 4
    else:
        BLOCK_C = 32
        BLOCK_T = 64
        num_warps = 2

    grid = (
        (max_seqlen + BLOCK_T - 1) // BLOCK_T,
        (dim + BLOCK_C - 1) // BLOCK_C,
        num_seq,
    )

    _conv1d_kernel[grid](
        x,
        weight,
        bias_t,
        conv_states,
        out,
        query_start_loc,
        cache_indices_t,
        hi_t,
        dim,
        total_seq,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        out.stride(0),
        out.stride(1),
        W=W,
        PAD_SLOT=pad_slot_id,
        BLOCK_C=BLOCK_C,
        BLOCK_T=BLOCK_T,
        HAS_BIAS=HAS_BIAS,
        ACT=act_code,
        HAS_INIT=HAS_INIT,
        HAS_PAD=HAS_PAD,
        num_warps=num_warps,
    )

    # State update
    BLOCK_C_STATE = 128
    grid_state = (num_seq, (dim + BLOCK_C_STATE - 1) // BLOCK_C_STATE)
    _state_update_kernel[grid_state](
        x,
        conv_states,
        query_start_loc,
        cache_indices_t,
        hi_t,
        dim,
        x.stride(0),
        x.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        W=W,
        PAD_SLOT=pad_slot_id,
        BLOCK_C=BLOCK_C_STATE,
        HAS_INIT=HAS_INIT,
        HAS_PAD=HAS_PAD,
        num_warps=4,
    )

    return out

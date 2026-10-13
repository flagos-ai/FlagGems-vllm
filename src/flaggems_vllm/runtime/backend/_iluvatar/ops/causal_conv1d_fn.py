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
def _causal_conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    conv_states_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    stride_x_d,
    stride_x_t,
    stride_w_d,
    stride_w_k,
    stride_out_d,
    stride_out_t,
    stride_cs_slot,
    stride_cs_d,
    stride_cs_k,
    dim,
    pad_slot_id,
    HAS_BIAS: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE_TENSOR: tl.constexpr,
    APPLY_ACTIVATION: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_t = tl.program_id(2)

    start = tl.load(query_start_loc_ptr + pid_seq).to(tl.int64)
    end = tl.load(query_start_loc_ptr + pid_seq + 1).to(tl.int64)
    seq_len = end - start

    if seq_len <= 0:
        return

    if HAS_CACHE_INDICES:
        slot = tl.load(cache_indices_ptr + pid_seq).to(tl.int64)
    else:
        slot = pid_seq.to(tl.int64)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_mask = d_off < dim
    t_mask = t_off < seq_len

    is_pad = slot == pad_slot_id

    if is_pad:
        # Pad-slot: bit-exact copy of x, no activation, no state update.
        x_vals = tl.load(
            x_ptr + d_off[:, None] * stride_x_d + (start + t_off[None, :]) * stride_x_t,
            mask=d_mask[:, None] & t_mask[None, :],
            other=0.0,
        )
        tl.store(
            out_ptr
            + d_off[:, None] * stride_out_d
            + (start + t_off[None, :]) * stride_out_t,
            x_vals,
            mask=d_mask[:, None] & t_mask[None, :],
        )
        return

    has_init = 0
    if HAS_INITIAL_STATE_TENSOR:
        has_init = tl.load(has_initial_state_ptr + pid_seq).to(tl.int32)

    acc = tl.zeros((BLOCK_D, BLOCK_T), dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
        acc = acc + bias[:, None]

    # Unroll over k. For pid_t > 0, all g >= 0 (assuming BLOCK_T >= W-1) so no
    # init-state overlay is needed; only the first t-tile touches negative g.
    is_first_t_tile = pid_t == 0
    for k in tl.static_range(0, WIDTH):
        # Load per-k weight slice: (BLOCK_D,)
        w_k = tl.load(
            weight_ptr + d_off * stride_w_d + k * stride_w_k,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

        g = t_off + k - (WIDTH - 1)  # position relative to x_seq start
        g_valid_x = (g >= 0) & t_mask

        x_val = tl.load(
            x_ptr + d_off[:, None] * stride_x_d + (start + g[None, :]) * stride_x_t,
            mask=d_mask[:, None] & g_valid_x[None, :],
            other=0.0,
        ).to(tl.float32)

        if HAS_INITIAL_STATE_TENSOR:
            if is_first_t_tile:
                init_pos = g + (WIDTH - 1)  # in [0, W-1)
                g_valid_init = (g < 0) & t_mask & (has_init != 0)
                init_val = tl.load(
                    conv_states_ptr
                    + slot * stride_cs_slot
                    + d_off[:, None] * stride_cs_d
                    + init_pos[None, :] * stride_cs_k,
                    mask=d_mask[:, None] & g_valid_init[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += w_k[:, None] * (x_val + init_val)
            else:
                acc += w_k[:, None] * x_val
        else:
            acc += w_k[:, None] * x_val

    if APPLY_ACTIVATION:
        acc = acc * (1.0 / (1.0 + tl.exp(-acc)))

    tl.store(
        out_ptr
        + d_off[:, None] * stride_out_d
        + (start + t_off[None, :]) * stride_out_t,
        acc,
        mask=d_mask[:, None] & t_mask[None, :],
    )

    # Fused state update: only pid_t==0 writes conv_states. This is the same
    # program that reads the initial state, so read-then-write ordering is
    # guaranteed within the program (no cross-program hazard).
    if pid_t == 0:
        for j in tl.static_range(0, WIDTH - 1):
            p = seq_len + j - (WIDTH - 1)
            use_x = p >= 0
            p_init = seq_len + j
            use_init = (p < 0) & (has_init != 0)

            x_val_st = tl.load(
                x_ptr + d_off * stride_x_d + (start + p) * stride_x_t,
                mask=d_mask & use_x,
                other=0.0,
            )
            init_val_st = tl.load(
                conv_states_ptr
                + slot * stride_cs_slot
                + d_off * stride_cs_d
                + p_init * stride_cs_k,
                mask=d_mask & use_init,
                other=0.0,
            )
            val = tl.where(use_x, x_val_st, init_val_st)

            tl.store(
                conv_states_ptr
                + slot * stride_cs_slot
                + d_off * stride_cs_d
                + j * stride_cs_k,
                val,
                mask=d_mask,
            )


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
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, 'silu', or 'swish'")

    dim, total_seq = x.shape
    width = weight.shape[1]
    num_seq = query_start_loc.numel() - 1

    out = torch.empty_like(x)

    if num_seq <= 0 or total_seq == 0:
        return out

    # Read the small (num_seq+1,) boundary tensor to host to compute max seqlen.
    boundaries = query_start_loc.tolist()
    max_seq_len = 0
    for i in range(num_seq):
        s = boundaries[i + 1] - boundaries[i]
        if s > max_seq_len:
            max_seq_len = s
    if max_seq_len == 0:
        return out

    BLOCK_D = 128
    BLOCK_T = 64

    # Dummy placeholder for None-tensor arguments (Triton needs a valid pointer).
    dummy = torch.empty(1, dtype=torch.int32, device=x.device)

    grid = (num_seq, triton.cdiv(dim, BLOCK_D), triton.cdiv(max_seq_len, BLOCK_T))
    _causal_conv1d_kernel[grid](
        x,
        weight,
        bias if bias is not None else dummy,
        out,
        conv_states,
        query_start_loc,
        cache_indices if cache_indices is not None else dummy,
        has_initial_state if has_initial_state is not None else dummy,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        dim,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        HAS_CACHE_INDICES=cache_indices is not None,
        HAS_INITIAL_STATE_TENSOR=has_initial_state is not None,
        APPLY_ACTIVATION=activation in ("silu", "swish"),
        WIDTH=width,
        BLOCK_D=BLOCK_D,
        BLOCK_T=BLOCK_T,
        num_warps=16,
    )

    return out

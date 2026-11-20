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

from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def _conv_update_w4_l1(
    x_ptr,
    w_ptr,
    st_ptr,
    idx_ptr,
    qsl_ptr,
    out_ptr,
    D,
    x_st,
    x_sd,
    st_s0,
    st_s1,
    st_s2,
    w_s0,
    w_s1,
    o_st,
    o_sd,
    NT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN: tl.constexpr,
    SINGLE: tl.constexpr,
    POOL: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NT
    tile = pid - b * NT
    offs = tile * BLOCK_D + tl.arange(0, BLOCK_D)

    d64 = offs.to(tl.int64)

    # The four weight loads depend only on the channel offset, and the x load only
    # on query_start_loc. Issuing them without the (idx >= 0) predicate keeps them
    # off the scalar conv_state_indices dependency chain, so they are already in
    # flight while the state-row addresses are still being resolved.
    woff = d64 * w_s1
    if EVEN:
        w0 = tl.load(w_ptr + woff).to(tl.float32)
        w1 = tl.load(w_ptr + w_s0 + woff).to(tl.float32)
        w2 = tl.load(w_ptr + 2 * w_s0 + woff).to(tl.float32)
        w3 = tl.load(w_ptr + 3 * w_s0 + woff).to(tl.float32)
    else:
        dm = offs < D
        w0 = tl.load(w_ptr + woff, mask=dm, other=0.0).to(tl.float32)
        w1 = tl.load(w_ptr + w_s0 + woff, mask=dm, other=0.0).to(tl.float32)
        w2 = tl.load(w_ptr + 2 * w_s0 + woff, mask=dm, other=0.0).to(tl.float32)
        w3 = tl.load(w_ptr + 3 * w_s0 + woff, mask=dm, other=0.0).to(tl.float32)

    idx = tl.load(idx_ptr + b).to(tl.int64)
    if SINGLE:
        # query_start_loc is a cumulative-length array and therefore starts at 0;
        # with exactly one sequence and one token its only entry is 0, so the

        qs = 0
    else:
        qs = tl.load(qsl_ptr + b).to(tl.int64)

    if idx < 0 or idx >= POOL or qs < 0 or qs >= NUM_TOKENS:
        return

    xoff = qs * x_st + d64 * x_sd
    ooff = qs * o_st + d64 * o_sd

    if EVEN:
        xb = tl.load(x_ptr + xoff)
    else:
        xb = tl.load(x_ptr + xoff, mask=dm, other=0.0)

    m = (offs < D) & (idx >= 0)

    so = idx * st_s0 + d64 * st_s2
    s0b = tl.load(st_ptr + so, mask=m, other=0.0)
    s1b = tl.load(st_ptr + so + st_s1, mask=m, other=0.0)
    s2b = tl.load(st_ptr + so + 2 * st_s1, mask=m, other=0.0)

    # Start the dependent multiply-add chain immediately so the latency-critical
    # fp32 compute no longer waits behind store issue, then let the three
    # accumulator-independent state stores overlap it.
    acc = w0 * s0b.to(tl.float32)
    acc += w1 * s1b.to(tl.float32)
    acc += w2 * s2b.to(tl.float32)
    acc += w3 * xb.to(tl.float32)

    # in-place state shift: row k <- window[k+1] of the last window.
    # None of these values depend on the accumulator.
    tl.store(st_ptr + so, s1b, mask=m)
    tl.store(st_ptr + so + st_s1, s2b, mask=m)
    tl.store(st_ptr + so + 2 * st_s1, xb, mask=m)

    y = acc * tl.sigmoid(acc)
    tl.store(out_ptr + ooff, y.to(out_ptr.dtype.element_ty), mask=m)


def fla_causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    *,
    activation="silu",
    conv_state_indices,
    query_start_loc,
    null_block_id=0,
    out,
):
    """Update one four-tap FLA state and return the caller-supplied output.

    Each token belongs to one independent active sequence with its own valid
    state slot. query_start_loc is the unit-step prefix-sum array for that batch.
    """
    state, cu, indices = conv_state, query_start_loc, conv_state_indices
    if (
        x.device.type != "npu"
        or x.ndim != 2
        or x.dtype != torch.bfloat16
        or x.stride(-1) != 1
        or not x.numel()
    ):
        raise ValueError("expected nonempty token-major BF16 activations on NPU")
    t, d = x.shape
    if (
        d != 4096
        or weight.shape != (4, d)
        or weight.dtype != x.dtype
        or not weight.is_contiguous()
    ):
        raise ValueError("expected a contiguous four-tap weight with 4096 channels")
    if (
        state.ndim != 3
        or state.shape[1:] != (3, d)
        or state.dtype != x.dtype
        or state.stride(-1) != 1
        or state.stride(1) < d
        or state.stride(0) < (2 * state.stride(1) + d)
        or state.shape[0] == 0
    ):
        raise ValueError("expected a non-overlapping [slots,3,4096] BF16 state")
    if (
        any(a.device != x.device for a in (state, weight, cu, indices))
        or cu.dtype != torch.int32
        or indices.dtype != torch.int32
        or not cu.is_contiguous()
        or not indices.is_contiguous()
    ):
        raise ValueError(
            "expected contiguous int32 lengths and indices on the same NPU"
        )

    if bias is not None or activation != "silu" or null_block_id != 0:
        raise ValueError("expected bias-free SiLU and null-block ID zero")
    if cu.shape != (t + 1,) or indices.shape != (t,):
        raise ValueError("expected one token and one state index per sequence")
    if (
        out.shape != x.shape
        or out.dtype != x.dtype
        or out.device != x.device
        or out.stride(-1) != 1
        or out.stride(0) < d
        and t > 1
    ):
        raise ValueError(
            "out must be a writable channel-last BF16 [tokens,channels] tensor"
        )
    with torch_device_fn.device(x.device):
        _conv_update_w4_l1[(t * (d // 512),)](
            x,
            weight,
            state,
            indices,
            cu,
            out,
            d,
            x.stride(0),
            x.stride(1),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            NT=d // 512,
            BLOCK_D=512,
            EVEN=True,
            SINGLE=t == 1,
            POOL=state.shape[0],
            NUM_TOKENS=t,
            num_warps=1,
        )
    return out

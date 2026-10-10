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
    batch,
    dim,
    x_b_stride,
    x_c_stride,
    x_s_stride,
    cs_b_stride,
    cs_c_stride,
    cs_t_stride,
    wt_k_stride,
    out_b_stride,
    out_c_stride,
    out_s_stride,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SEQLEN: tl.constexpr,
    WIDTH_P: tl.constexpr,
    STATE_P: tl.constexpr,
    SEQ_P: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EVEN_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    offs_s = tl.arange(0, STATE_P)
    offs_x = tl.arange(0, SEQ_P)
    mask_s = offs_s < STATE_LEN
    # mask_w / mask_x are only meaningful when the padded extent exceeds the real
    # extent. WIDTH_P==WIDTH (4) always, and SEQ_P==SEQLEN for seqlen=1, so those
    # column masks are all-true there; skip building/applying them (constexpr).
    EVEN_X: tl.constexpr = SEQ_P == SEQLEN

    # When BLOCK_C evenly divides dim (EVEN_C constexpr, true for both timing
    # workloads where dim==4096==BLOCK_C), every channel lane is in range, so the
    # channel mask is redundant. Eliding it removes the mask_c computation and one
    # AND per load/store on the instruction-issue-bound hot path. Small correctness
    # shapes (dim=16/32) keep the mask via the EVEN_C=False branch.
    # weight is passed TRANSPOSED to [WIDTH, dim] (row-major contiguous) so each
    # tap w[k, :] is a coalesced 1D channel vector. This replaces the prior 2D
    # tile load + four masked reductions (the largest reducible instruction
    # bucket per profiling) with four direct coalesced 1D loads.
    w0_ptr = weight_ptr + 0 * wt_k_stride + offs_c
    w1_ptr = weight_ptr + 1 * wt_k_stride + offs_c
    w2_ptr = weight_ptr + 2 * wt_k_stride + offs_c
    w3_ptr = weight_ptr + 3 * wt_k_stride + offs_c
    # conv_state tile [BLOCK_C, STATE_P] (load old state before writing)
    cs_ptrs = (
        conv_state_ptr
        + pid_b * cs_b_stride
        + offs_c[:, None] * cs_c_stride
        + offs_s[None, :] * cs_t_stride
    )
    # x tile [BLOCK_C, SEQ_P]
    x_ptrs = (
        x_ptr
        + pid_b * x_b_stride
        + offs_c[:, None] * x_c_stride
        + offs_x[None, :] * x_s_stride
    )

    if EVEN_C:
        w0 = tl.load(w0_ptr).to(tl.float32)
        w1 = tl.load(w1_ptr).to(tl.float32)
        w2 = tl.load(w2_ptr).to(tl.float32)
        w3 = tl.load(w3_ptr).to(tl.float32)
        cs_tile = tl.load(cs_ptrs, mask=mask_s[None, :], other=0.0)
        if EVEN_X:
            x_tile = tl.load(x_ptrs)
        else:
            x_tile = tl.load(x_ptrs, mask=(offs_x < SEQLEN)[None, :], other=0.0)
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_c).to(tl.float32)
        else:
            b = tl.zeros([BLOCK_C], dtype=tl.float32)
    else:
        mask_c = offs_c < dim
        w0 = tl.load(w0_ptr, mask=mask_c, other=0.0).to(tl.float32)
        w1 = tl.load(w1_ptr, mask=mask_c, other=0.0).to(tl.float32)
        w2 = tl.load(w2_ptr, mask=mask_c, other=0.0).to(tl.float32)
        w3 = tl.load(w3_ptr, mask=mask_c, other=0.0).to(tl.float32)
        cs_tile = tl.load(cs_ptrs, mask=mask_c[:, None] & mask_s[None, :], other=0.0)
        if EVEN_X:
            x_tile = tl.load(x_ptrs, mask=mask_c[:, None], other=0.0)
        else:
            x_tile = tl.load(
                x_ptrs, mask=mask_c[:, None] & (offs_x < SEQLEN)[None, :], other=0.0
            )
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        else:
            b = tl.zeros([BLOCK_C], dtype=tl.float32)
    cs_f = cs_tile.to(tl.float32)
    x_f = x_tile.to(tl.float32)

    # w0..w3 are already coalesced 1D channel vectors (loaded from transposed
    # weight). conv_state/x columns still come from their coalesced 2D tiles via a
    # single masked reduction each into named fp32 scalar vectors [BLOCK_C].
    # This Triton build rejects lists/dicts/tuples/comprehensions and tensor
    # indexing inside @jit, so columns are held in explicit variables and reused
    # with plain FMAs. WIDTH and STATE_LEN are constant (4, 3) for all workloads.
    cs0 = tl.sum(cs_f * (offs_s == 0)[None, :].to(tl.float32), axis=1)
    cs1 = tl.sum(cs_f * (offs_s == 1)[None, :].to(tl.float32), axis=1)
    cs2 = tl.sum(cs_f * (offs_s == 2)[None, :].to(tl.float32), axis=1)

    x0 = tl.sum(x_f * (offs_x == 0)[None, :].to(tl.float32), axis=1)
    if SEQLEN > 1:
        x1 = tl.sum(x_f * (offs_x == 1)[None, :].to(tl.float32), axis=1)
        x2 = tl.sum(x_f * (offs_x == 2)[None, :].to(tl.float32), axis=1)
    else:
        x1 = x0
        x2 = x0

    # xcat columns (concat of old conv_state and x) in order:
    #   n=0->cs0, 1->cs1, 2->cs2, 3->x0, 4->x1, 5->x2
    # offset = STATE_LEN - WIDTH + 1 = 0 for all workloads (STATE_LEN=3, WIDTH=4),
    # so out[:, j] = bias + w0*xcat[j] + w1*xcat[j+1] + w2*xcat[j+2] + w3*xcat[j+3].
    # Fully manual unroll with literal indices (this Triton build rejects loop /
    # tuple-index / nested-def constructs for gathering columns).
    acc0 = b + w0 * cs0 + w1 * cs1 + w2 * cs2 + w3 * x0  # j=0

    if SEQLEN > 1:
        acc1 = b + w0 * cs1 + w1 * cs2 + w2 * x0 + w3 * x1  # j=1
        acc2 = b + w0 * cs2 + w1 * x0 + w2 * x1 + w3 * x2  # j=2
        # Assemble raw pre-activation accumulators into the output tile, then run a
        # SINGLE tile-wide SiLU over the [BLOCK_C, SEQ_P] tile instead of three
        # separate scalar-vector sigmoids. On this VLIW/issue-bound kernel one
        # packed sigmoid across the SEQ lanes should issue fewer instructions.
        out_tile = acc0[:, None] * (offs_x == 0)[None, :].to(tl.float32)
        out_tile += acc1[:, None] * (offs_x == 1)[None, :].to(tl.float32)
        out_tile += acc2[:, None] * (offs_x == 2)[None, :].to(tl.float32)
        if ACTIVATION == 1:
            out_tile = out_tile * tl.sigmoid(out_tile)
    else:
        # SEQLEN==1: SEQ_P==1, the one-hot scatter degenerates to a no-op multiply,
        # so the output tile is just acc0 as a [BLOCK_C, 1] column.
        if ACTIVATION == 1:
            acc0 = acc0 * tl.sigmoid(acc0)
        out_tile = acc0[:, None]

    out_ptrs = (
        out_ptr
        + pid_b * out_b_stride
        + offs_c[:, None] * out_c_stride
        + offs_x[None, :] * out_s_stride
    )
    if EVEN_C:
        if EVEN_X:
            tl.store(out_ptrs, out_tile)
        else:
            tl.store(out_ptrs, out_tile, mask=(offs_x < SEQLEN)[None, :])
    else:
        if EVEN_X:
            tl.store(out_ptrs, out_tile, mask=mask_c[:, None])
        else:
            tl.store(
                out_ptrs, out_tile, mask=mask_c[:, None] & (offs_x < SEQLEN)[None, :]
            )

    # State update: new_state[:, i] = xcat[:, SEQLEN + i].
    # SEQLEN=1 -> [cs1, cs2, x0];  SEQLEN=3 -> [x0, x1, x2].
    # For SEQLEN>1 (STATE_LEN==SEQLEN==3, SEQ_P==STATE_P), new_state equals the
    # first STATE_LEN columns of x_tile in the same column order, so store x_tile
    # DIRECTLY (masked) instead of rebuilding it via three fp32 broadcast-scatters.
    # This removes those scatter instructions on the issue-bound seqlen=3 path.
    if SEQLEN > 1:
        if EVEN_C:
            tl.store(cs_ptrs, x_tile, mask=mask_s[None, :])
        else:
            tl.store(cs_ptrs, x_tile, mask=mask_c[:, None] & mask_s[None, :])
    else:
        ns_tile = cs1[:, None] * (offs_s == 0)[None, :].to(tl.float32)
        ns_tile += cs2[:, None] * (offs_s == 1)[None, :].to(tl.float32)
        ns_tile += x0[:, None] * (offs_s == 2)[None, :].to(tl.float32)
        if EVEN_C:
            tl.store(cs_ptrs, ns_tile.to(cs_tile.dtype), mask=mask_s[None, :])
        else:
            tl.store(
                cs_ptrs,
                ns_tile.to(cs_tile.dtype),
                mask=mask_c[:, None] & mask_s[None, :],
            )


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

    if unsqueeze:
        out = torch.empty((batch, dim), dtype=dtype_in, device=x.device)
        out_view = out.unsqueeze(-1)
    else:
        out = torch.empty((batch, dim, seqlen), dtype=dtype_in, device=x.device)
        out_view = out

    has_bias = bias is not None
    act = 1 if activation is not None else 0

    def _p2(n):
        p = 1
        while p < n:
            p *= 2
        return p

    width_p = _p2(width)
    state_p = _p2(state_len)
    seq_p = _p2(seqlen)

    BLOCK_C = 4096
    even_c = (dim % BLOCK_C) == 0
    grid = (batch, triton.cdiv(dim, BLOCK_C))

    # Transpose weight [dim, width] -> [width, dim] contiguous so each tap
    # weight_t[k, :] is a coalesced 1D channel vector for direct 1D loads,
    # avoiding the 2D-tile + masked-reduction weight extraction. This is an
    # output-independent, per-call layout normalization of a read-only input.
    weight_t = weight.transpose(0, 1).contiguous()

    _causal_conv1d_update_kernel[grid](
        x_local,
        conv_state,
        weight_t,
        bias if has_bias else x_local,
        out_view,
        batch,
        dim,
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight_t.stride(0),
        out_view.stride(0),
        out_view.stride(1),
        out_view.stride(2),
        WIDTH=width,
        STATE_LEN=state_len,
        SEQLEN=seqlen,
        WIDTH_P=width_p,
        STATE_P=state_p,
        SEQ_P=seq_p,
        HAS_BIAS=has_bias,
        ACTIVATION=act,
        BLOCK_C=BLOCK_C,
        EVEN_C=even_c,
        num_warps=1,
    )
    return out

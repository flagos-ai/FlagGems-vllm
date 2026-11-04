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

"""Triton implementation of iHC (independent Hyper-Connections) pre operator.

Baseline port of the vLLM HY V4 implementation
(vllm/models/hy_v4/nvidia/triton_ihc.py), adapted to FlagGems-vllm
conventions. iHC is the *independent* variant of Hyper-Connections: unlike mHC
there is no Sinkhorn and no n x n cross-stream mixing matrix -- each residual
channel gets a single scalar sigmoid gate.

Computed, for input x of shape [num_tokens, hc_mult, hidden_size]:
    x_flat = x.flatten(1)                                  # [N, hc*d]
    r      = rsqrt(mean(x_flat^2, -1) + norm_eps)          # [N, 1]
    mixes  = (x_flat @ weight.T) * r                       # [N, 2*hc]
    H_pre  = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_eps
    H_post = magnitude * sigmoid(mixes[:, hc:] * scale[1] + base[hc:]) + hc_eps
    y[n,:] = sum_i H_pre[n, i] * x[n, i, :]                # [N, d]
Returns (y, H_post): the pre-gated channel reduction and the post gates that
HYV4HCPostLayer consumes.

Two-stage design:
  stage1: per-token fp32 reduction over hc*d producing mix[N, 2*hc] (w-projection)
          and sqrsum[N]. M-blocked (one program owns BLOCK_M tokens, x read once
          per K-chunk and reused for sqrsum + all 2*hc FMAs) with a TLE-pipelined
          K loop; a K-split grid (per-split slots, reduced in stage2) kicks in
          for small N.
  stage2: mix + sqrsum -> rms, pre/post sigmoid gates, post store, and the
          pre-gated channel sum y[n, :] = sum_i pre[n, i] * x[n, i, :].
"""

import torch
import triton
import triton.language as tl

# stage1 (w-projection + sqrsum) tile; stage2 (gates + channel sum) tile.
_BLOCK_K = 128
_BLOCK_D = 1024
_BLOCK_M = 8
_NUM_WARPS = 1
_NUM_STAGES = 3
# Fill the H20 (~132 SMs): only K-split stage1 when token-blocks are too few.
_MIN_PROGRAMS = 512

# Token count below which the decode-shaped path is used. The M-blocked stage1
# amortizes the weight rows across BLOCK_M tokens, which only pays off once
# there are enough token-blocks to fill the GPU; below that it starves (measured
# up to 2.5x slower than vLLM at a single token). The decode path instead splits
# K finely, loads the 2*hc weight rows as one 2D tile, and folds the partial
# reduction into stage2 so the whole op costs two kernel launches instead of
# three -- at decode sizes a launch is ~4.5us, i.e. 20-40% of the total.
_DECODE_N_THRESHOLD = 256
# BLOCK_K for the decode stage1, by token count: tiny batches need K cut finely
# to expose enough programs, larger ones prefer fewer, wider chunks.
_DECODE_BLOCK_K = ((2, 128), (16, 256), (64, 512), (None, 1024))


def _decode_block_k(num_tokens: int) -> int:
    for limit, block_k in _DECODE_BLOCK_K:
        if limit is None or num_tokens < limit:
            return block_k
    return _DECODE_BLOCK_K[-1][1]


# stage1 computes, in fp32, the w-projection mix[N, 2*hc] and the row sqrsum[N].
# One program owns BLOCK_M tokens; x is loaded once per K-chunk and reused for
# sqrsum plus all 2*hc weight-row FMAs (register-resident), removing the eager /
# baseline 2*hc-fold L1 re-read of x -- the measured L1TEX bottleneck. The K loop
# uses tl.range software pipelining (num_stages) to prefetch the next x chunk.
# fp32 accumulation keeps the post gate bit-accurate (no tensor cores); tl.dot in
# bf16/tf32 would blow the 2e-5 gate tolerance at K = hc*hidden.
#
# One grid axis for the token blocks, one (NSPLIT-wide) for a K-split that kicks
# in when the token-blocks alone cannot fill the GPU. Every program writes its
# own (split, token) slot and stage2 sums the slots in index order, so the
# reduction order is fixed no matter how the blocks interleave.
#
# The slots replace an earlier fp32 tl.atomic_add into mix / sqrsum. The atomic
# was not an accuracy problem (the post gates stayed ~2e-7 against fp64, well
# inside their 2e-5 tolerance) but it made the result depend on which block
# arrived first, so repeated launches were not bitwise identical -- measured
# spread ~6e-7 on the post gates at 256..2048 tokens, which is what disqualifies
# an op from a batch-invariant path. Private slots are also faster here, since
# they drop both the atomic contention and the pre-zeroing the accumulate
# required: 1.26x at 256 tokens / NSPLIT=16, tapering to 1.03x at NSPLIT=2.
#
# X_STRIDE is the full row length and K_CHUNK is this program's share of it.
# They have to be separate: deriving the row stride from the reduction length
# (as a single K_TOTAL did) is what forced the atomic, because then a program
# could not cover part of K while still addressing whole rows.
@triton.jit
def _ihc_pre_stage1(
    x_ptr,
    weight_ptr,
    sumsq_slot_ptr,
    mix_slot_ptr,
    N,
    X_STRIDE: tl.constexpr,
    K_CHUNK: tl.constexpr,
    TWO_HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NSPLIT: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    token_block_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)

    m_off = token_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_off < N

    k_start = split_idx * K_CHUNK

    # One accumulator per projection row, plus one for the row sqrsum. Triton
    # has no mutable list, so the running values are carried in a tuple that is
    # rebuilt each iteration.
    sum_squares = tl.zeros((BLOCK_M,), dtype=tl.float32)
    mixes = ()
    for _ in tl.static_range(TWO_HC):
        mixes = mixes + (tl.zeros((BLOCK_M,), dtype=tl.float32),)

    x_base = x_ptr + m_off[:, None] * X_STRIDE
    for k_offset in tl.range(0, K_CHUNK, BLOCK_K, num_stages=NUM_STAGES):
        k_off = k_start + k_offset + tl.arange(0, BLOCK_K)
        k_mask = k_off < X_STRIDE

        x = tl.load(
            x_base + k_off[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        sum_squares += tl.sum(x * x, axis=1)

        # Reuse the x tile just loaded for every projection row.
        next_mixes = ()
        for j in tl.static_range(TWO_HC):
            weight_row = tl.load(
                weight_ptr + j * X_STRIDE + k_off, mask=k_mask, other=0.0
            )
            contribution = tl.sum(x * weight_row[None, :], axis=1)
            next_mixes = next_mixes + (mixes[j] + contribution,)
        mixes = next_mixes

    # Private slot per (split, token): a plain store either way, so NSPLIT == 1
    # needs no special case.
    tl.store(sumsq_slot_ptr + split_idx * N + m_off, sum_squares, mask=m_mask)
    mix_slot = mix_slot_ptr + split_idx * N * TWO_HC + m_off * TWO_HC
    for j in tl.static_range(TWO_HC):
        tl.store(mix_slot + j, mixes[j], mask=m_mask)


# Decode-shaped stage1: one program per (token, K-chunk). With few tokens there
# is no weight reuse to win, so the goal is simply to expose enough programs and
# keep each one cheap. The 2*hc weight rows are fetched as a single
# [2*hc, BLOCK_K] tile and reduced along K in one vectorized step, rather than
# looping the rows one at a time. Each program owns one partial slot, so no
# atomics are needed; _ihc_pre_stage2_decode folds the slot reduction in.
@triton.jit
def _ihc_pre_stage1_decode(
    x_ptr,
    weight_ptr,
    partial_ptr,
    K_TOTAL: tl.constexpr,
    TWO_HC: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    k_off = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K_TOTAL

    x = tl.load(
        x_ptr + token_idx * K_TOTAL + k_off,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    sum_squares = tl.sum(x * x, axis=0)

    # All 2*hc weight rows as one tile, so the projection is a single reduction
    # along K instead of one pass per row.
    mix_idx = tl.arange(0, TWO_HC)
    weight = tl.load(
        weight_ptr + mix_idx[:, None] * K_TOTAL + k_off[None, :],
        mask=k_mask[None, :],
        other=0.0,
    )
    mixes = tl.sum(weight * x[None, :], axis=1)

    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, sum_squares)
    tl.store(partial + 1 + mix_idx, mixes)


# stage2 reduces the per-split slots stage1 wrote, forms the pre / post sigmoid
# gates, stores the post gate, and emits the pre-gated channel sum
# y[n, :] = sum_i pre[n, i] * x[n, i, :].
#
# Folding the slot reduction in here keeps the op at two launches, the same
# trick _ihc_pre_stage2_decode uses; a separate reduction kernel would add a
# third. The sum runs over the split axis in index order, which is what makes
# the result independent of block completion order.
@triton.jit
def _ihc_pre_stage2(
    x_ptr,
    mix_slot_ptr,
    sumsq_slot_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    N,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    TWO_HC: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    split_idx = tl.arange(0, BLOCK_S)
    split_mask = split_idx < NSPLIT

    sum_squares = tl.sum(
        tl.load(sumsq_slot_ptr + split_idx * N + token_idx, mask=split_mask, other=0.0),
        axis=0,
    )

    mix_idx = tl.arange(0, TWO_HC)
    mix_parts = tl.load(
        mix_slot_ptr
        + split_idx[:, None] * N * TWO_HC
        + token_idx * TWO_HC
        + mix_idx[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    mixes = tl.sum(mix_parts, axis=0)

    # Split the reduced row into the pre gates' logits and the post gates'.
    is_pre = mix_idx[None, :] == hc_idx[:, None]
    is_post = mix_idx[None, :] == (hc_idx[:, None] + HC_MULT)
    pre_mix = tl.sum(tl.where(is_pre, mixes[None, :], 0.0), axis=1)
    post_mix = tl.sum(tl.where(is_post, mixes[None, :], 0.0), axis=1)

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS

    if hidden_block_idx == 0:
        post = (
            MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
            + HC_EPS
        )
        tl.store(
            post_ptr + token_idx * HC_MULT + hc_idx,
            post,
            mask=hc_mask,
        )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel

    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


# Decode stage2: same gating and channel sum as above, but it reduces the
# partial slots written by _ihc_pre_stage1_decode itself instead of reading a
# pre-reduced mix / sqrsum. That keeps the decode path at two kernel launches;
# a separate reduction kernel would add a third, which costs ~4.5us and is a
# large fraction of the total at these sizes.
@triton.jit
def _ihc_pre_stage2_decode(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    TWO_HC: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    split_idx = tl.arange(0, BLOCK_S)
    split_mask = split_idx < NUM_SPLITS
    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    slot_ptrs = partial_row + split_idx * PARTIAL_STRIDE

    # Slot layout is [sqrsum, mix[0 .. 2*hc-1]]; reduce over the split axis.
    sqrsum_parts = tl.load(slot_ptrs, mask=split_mask, other=0.0)
    sum_squares = tl.sum(sqrsum_parts, axis=0)

    mix_idx = tl.arange(0, TWO_HC)
    mix_ptrs = slot_ptrs[:, None] + 1 + mix_idx[None, :]
    mix_parts = tl.load(mix_ptrs, mask=split_mask[:, None], other=0.0)
    mixes = tl.sum(mix_parts, axis=0)

    # Split the reduced row into the pre gates' logits (first hc entries) and
    # the post gates' (next hc).
    is_pre = mix_idx[None, :] == hc_idx[:, None]
    is_post = mix_idx[None, :] == (hc_idx[:, None] + HC_MULT)
    pre_mix = tl.sum(tl.where(is_pre, mixes[None, :], 0.0), axis=1)
    post_mix = tl.sum(tl.where(is_post, mixes[None, :], 0.0), axis=1)

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS

    if hidden_block_idx == 0:
        post = (
            MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
            + HC_EPS
        )
        tl.store(post_ptr + token_idx * HC_MULT + hc_idx, post, mask=hc_mask)

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel

    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


def ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iHC pre: reduce the hc channels and produce the post gates.

    Args:
        x: [num_tokens, hc_mult, hidden_size], float16/bfloat16.
        weight: [2*hc_mult, hc_mult*hidden_size], float32 (kept out of fp8).
        scale: [2], float32 -- (pre_scale, post_scale).
        base: [2*hc_mult], float32 -- pre bias then post bias.
        magnitude: scalar multiplier on the post gate.
        hc_eps: epsilon added to both gates.
        norm_eps: epsilon inside the rsqrt.

    Returns:
        (y, post): y is [num_tokens, hidden_size] (x.dtype), post is
        [num_tokens, hc_mult] (float32).
    """
    assert x.dim() == 3, "x must be [num_tokens, hc_mult, hidden_size]"
    assert x.is_cuda and weight.is_cuda and scale.is_cuda and base.is_cuda
    assert weight.dtype == torch.float32
    assert scale.dtype == torch.float32 and base.dtype == torch.float32

    x = x.contiguous()
    assert weight.is_contiguous()
    scale = scale.contiguous()
    base = base.contiguous()
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    assert weight.shape == (2 * hc_mult, k_total)
    assert scale.shape == (2,)
    assert base.shape == (2 * hc_mult,)

    if num_tokens == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, hc_mult), dtype=torch.float32, device=x.device),
        )

    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    hc_pow2 = triton.next_power_of_2(hc_mult)
    two_hc = 2 * hc_mult
    x_flat = x.reshape(num_tokens, k_total)

    if num_tokens < _DECODE_N_THRESHOLD:
        # Decode shapes: fine K-split + fused reduction, two launches total.
        block_k = _decode_block_k(num_tokens)
        num_splits = triton.cdiv(k_total, block_k)
        partial_stride = 1 + two_hc
        partial = torch.empty(
            (num_tokens, num_splits, partial_stride),
            dtype=torch.float32,
            device=x.device,
        )
        _ihc_pre_stage1_decode[(num_tokens, num_splits)](
            x_flat,
            weight,
            partial,
            K_TOTAL=k_total,
            TWO_HC=two_hc,
            NUM_SPLITS=num_splits,
            BLOCK_K=block_k,
            PARTIAL_STRIDE=partial_stride,
            num_warps=1,
            enable_fp_fusion=False,
        )
        _ihc_pre_stage2_decode[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
            x,
            partial,
            scale,
            base,
            output,
            post,
            HIDDEN_SIZE=hidden_size,
            K_TOTAL=k_total,
            HC_MULT=hc_mult,
            HC_POW2=hc_pow2,
            TWO_HC=two_hc,
            NUM_SPLITS=num_splits,
            PARTIAL_STRIDE=partial_stride,
            BLOCK_S=triton.next_power_of_2(num_splits),
            BLOCK_D=_BLOCK_D,
            MAGNITUDE=magnitude,
            NORM_EPS=norm_eps,
            HC_EPS=hc_eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return output, post

    # stage1: fp32 mix / sqrsum slots, one per (split, token). K-split only when
    # token-blocks alone cannot fill the GPU; otherwise the single-pass M-block
    # grid already has enough programs and splitting just adds slots to reduce.
    #
    # The split is not merely about program count: it also shortens each
    # program's serial K chain (24576 -> 1536 at NSPLIT=16). Lowering BLOCK_M
    # raises the program count too but leaves that chain intact, and measures
    # 2.3x slower at 256 tokens even with 256 programs on 78 SMs.
    grid_m = triton.cdiv(num_tokens, _BLOCK_M)
    nsplit = 1
    while (
        grid_m * nsplit * 2 <= _MIN_PROGRAMS and k_total % (nsplit * 2 * _BLOCK_K) == 0
    ):
        nsplit *= 2

    # One slot per (split, token); stage2 reduces them. torch.empty is enough
    # because nothing accumulates into these -- the atomic variant this replaces
    # needed torch.zeros.
    sumsq = torch.empty(nsplit * num_tokens, dtype=torch.float32, device=x.device)
    mix = torch.empty(
        nsplit * num_tokens * two_hc, dtype=torch.float32, device=x.device
    )
    _ihc_pre_stage1[(grid_m, nsplit)](
        x_flat,
        weight,
        sumsq,
        mix,
        num_tokens,
        X_STRIDE=k_total,
        K_CHUNK=k_total // nsplit,
        TWO_HC=two_hc,
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        NSPLIT=nsplit,
        NUM_STAGES=_NUM_STAGES,
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )

    _ihc_pre_stage2[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        x,
        mix,
        sumsq,
        scale,
        base,
        output,
        post,
        num_tokens,
        HIDDEN_SIZE=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        TWO_HC=two_hc,
        NSPLIT=nsplit,
        BLOCK_S=triton.next_power_of_2(nsplit),
        BLOCK_D=_BLOCK_D,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, post


def ihc_pre_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference (test/benchmark only; never imported by production)."""
    import torch.nn.functional as F

    num_tokens, hc_mult, hidden_size = x.shape
    x_flat = x.flatten(1).float()
    reciprocal_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, weight) * reciprocal_rms
    pre = torch.sigmoid(mixes[:, :hc_mult] * scale[0] + base[:hc_mult])
    post = torch.sigmoid(mixes[:, hc_mult:] * scale[1] + base[hc_mult:])
    pre = pre + hc_eps
    post = magnitude * post + hc_eps
    output = torch.sum(pre.unsqueeze(-1) * x.float(), dim=1)
    return output.to(x.dtype).reshape(num_tokens, hidden_size), post

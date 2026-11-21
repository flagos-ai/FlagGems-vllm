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

GCU (enflame) adaptation of the device-agnostic port of the vLLM HY V4
implementation (vllm/models/hy_v4/nvidia/triton_ihc.py). iHC is the
*independent* variant of Hyper-Connections: unlike mHC there is no Sinkhorn
and no n x n cross-stream mixing matrix -- each residual channel gets a
single scalar sigmoid gate.

Computed, for input x of shape [num_tokens, hc_mult, hidden_size]:
    x_flat = x.flatten(1)                                  # [N, hc*d]
    r      = rsqrt(mean(x_flat^2, -1) + norm_eps)          # [N, 1]
    mixes  = (x_flat @ weight.T) * r                       # [N, 2*hc]
    H_pre  = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_eps
    H_post = magnitude * sigmoid(mixes[:, hc:] * scale[1] + base[hc:]) + hc_eps
    y[n,:] = sum_i H_pre[n, i] * x[n, i, :]                # [N, d]
Returns (y, H_post).

Two launches, same structure as the generic two-stage design:

  stage1: per (token-block, K-split) work item, one fp32 [BLOCK_K] vector
          accumulator per projection row plus one for the row sqrsum. The
          products x*w_j are FMA-folded into the accumulators inside the K
          loop (no per-chunk reductions); each accumulator is reduced once
          per work item and written to a private (token, split) slot.
  stage2: per token, reduces the split slots in index order (deterministic),
          forms the pre/post sigmoid gates once, stores the post gate, and
          emits the pre-gated channel sum y[n, :] over hidden blocks.

GCU-specific adaptations (2026-11-07 toolchain, perf retune 2026-11-11):

* Fully 1-D affine accesses everywhere: the toolchain fails to compile any
  2-D tile load/store (`memref_ext.slice_(pad_)start` operand-segment
  mismatch, see compiler_issues/vllm_moe_sum.md). The M-blocked 2-D x tile of
  the generic design becomes per-token 1-D rows; the weight rows are 1-D
  loads shared across the token block.
* Capped persistent grids (48 programs = one per GCU400 SP, the
  `_enflame/ops/moe_sum.py` idiom): work items are strided by program id
  instead of one program per item. num_tokens on grid.y (limit 255) and the
  ~52ns/CTA dispatch floor at fine granularity are both avoided.
* `x_m * x_m + 0.0`: the backend's mul+add fusion mangles the square of a
  bf16-promoted value (measured up to 0.18 absolute error per 256-element
  slot vs fp64, breaking the 2e-5 post-gate tolerance through the shared
  reciprocal rms). The +0.0 barrier breaks the pattern; the square of a
  bf16 value is exact in fp32 either way. The x*w_j products are unaffected
  (weight is fp32) and stay fused -- FMA there only helps.
* num_warps: stage1 decode runs one warp (small work items: 512-lane chunks
  and 9 intra-warp reduction trees; nw>=2 measured 2.3x slower at N=64),
  while stage1 prefill runs eight warps -- 18 accumulators x 1024 lanes /
  256 threads = 72 regs/thread, so the [BLOCK_K] vector accumulators stay
  register-resident instead of spilling to DSM. The +0.0 barrier keeps the
  self-square exact at any warp count (the pre-barrier "nw=4 drifts ~2.3"
  observation was this fusion bug, whose magnitude is warp-count sensitive).
* stage2 prefill vectorized gates: scale/base are loop-invariant and are
  loaded once per program (not per token); the post gates are formed as one
  [hc] vector sigmoid + one vector store, the mixes as [hc] vector loads;
  the pre gates stay scalar (consumed as FMA scalars by the channel-outer
  y stream). reciprocal rms uses `sqrsum * (1/K)` (mulrec) -- within the
  2e-5 post-gate tolerance. This halves stage2 latency at large N.
* Masked loads are kept even where statically all-true: K_EXACT/M_EXACT
  mask-free arms measured 8% *slower* on this backend (2026-11-11 sweep,
  both stage1 and stage2), unlike the predicate-tax wins seen elsewhere.
"""

import torch
import triton
import triton.language as tl

# Persistent-grid program count: one program per GCU400 SP (moe_sum idiom).
_GRID_TARGET = 48

# Decode shapes (num_tokens < _DECODE_N_THRESHOLD): one token per work item
# and a fine K split -- there is no weight reuse to win with few tokens, so
# the goal is enough cheap work items to cover the persistent programs.
# BLOCK_K by token count: tiny batches need K cut finely to expose enough
# programs, larger ones prefer fewer, wider chunks.
_DECODE_N_THRESHOLD = 256
_DECODE_BLOCK_K = ((2, 128), (16, 256), (64, 512), (None, 1024))

# Prefill shapes: 2026-11-11 sweep over (BLOCK_M, BLOCK_K, num_warps) found a
# sharp U-valley at BLOCK_M=2 / BLOCK_K=1024 / nw=8 (N=8192 stage1 3045 ->
# 2216us, -27%; N=1024 434 -> 288us, -34%). Larger BLOCK_M re-reads the
# weight less but loses more to scheduling/register pressure (BM=4/BK=512 is
# 2.1x slower); BLOCK_M=1 has no x-row sharing. num_stages pipelining and
# grid 96/192 are both neutral; mask-free exact arms are 8% slower.
_PREFILL_BLOCK_M = 2
_PREFILL_BLOCK_K = 1024

# stage1 warps: decode work items are 512-lane chunks reduced by intra-warp
# trees (nw=1 wins); prefill register residency wants nw=8 (see docstring).
_STAGE1_DECODE_WARPS = 1
_STAGE1_PREFILL_WARPS = 8
# stage2 warps (vector-gated kernel; nw=4 == nw=8 measured).
_STAGE2_WARPS = 4
# stage2 hidden-slice width: 2048 beats 1024 (-9% prod / -13% vec kernel at
# N=8192); channel-outer streaming, see the kernel comment.
_STAGE2_BLOCK_D = 2048


def _decode_block_k(num_tokens: int) -> int:
    for limit, block_k in _DECODE_BLOCK_K:
        if limit is None or num_tokens < limit:
            return block_k
    return _DECODE_BLOCK_K[-1][1]


@triton.jit
def _ihc_pre_stage1(
    x_ptr,
    weight_ptr,
    partial_ptr,
    N,
    K_TOTAL: tl.constexpr,
    K_CHUNK: tl.constexpr,
    TWO_HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    # One work item = (token block, K split). Slot layout per (token, split):
    # [sqrsum, mix[0 .. 2*hc-1]]; stage2 reduces the split axis.
    ACC: tl.constexpr = TWO_HC + 1
    ACC_LEN: tl.constexpr = BLOCK_M * ACC
    num_blocks = (N + BLOCK_M - 1) // BLOCK_M
    total = num_blocks * NUM_SPLITS
    for work in range(tl.program_id(0), total, tl.num_programs(0)):
        block = work // NUM_SPLITS
        split = work % NUM_SPLITS
        k_start = split * K_CHUNK
        acc = ()
        for i in tl.static_range(ACC_LEN):
            acc = acc + (tl.zeros((BLOCK_K,), dtype=tl.float32),)
        for k_off in range(0, K_CHUNK, BLOCK_K):
            k = k_start + k_off + tl.arange(0, BLOCK_K)
            kmask = k < K_TOTAL
            # Weight rows are loaded once per chunk and shared by the whole
            # token block; the products FMA-fold into the accumulators.
            w_rows = ()
            for j in tl.static_range(TWO_HC):
                w_rows = w_rows + (
                    tl.load(weight_ptr + j * K_TOTAL + k, mask=kmask, other=0.0),
                )
            deltas = ()
            for m in tl.static_range(BLOCK_M):
                token = block * BLOCK_M + m
                x_m = tl.load(
                    x_ptr + token * K_TOTAL + k,
                    mask=kmask & (token < N),
                    other=0.0,
                ).to(tl.float32)
                for j in tl.static_range(TWO_HC):
                    deltas = deltas + (x_m * w_rows[j],)
                # +0.0 breaks the backend's acc + x*x fusion, which mangles
                # the square of a bf16-promoted value (see module docstring).
                deltas = deltas + (x_m * x_m + 0.0,)
            new_acc = ()
            for i in tl.static_range(ACC_LEN):
                new_acc = new_acc + (acc[i] + deltas[i],)
            acc = new_acc
        for m in tl.static_range(BLOCK_M):
            token = block * BLOCK_M + m
            slot = partial_ptr + (token * NUM_SPLITS + split) * PARTIAL_STRIDE
            tl.store(slot, tl.sum(acc[m * ACC + TWO_HC]), mask=token < N)
            for j in tl.static_range(TWO_HC):
                tl.store(slot + 1 + j, tl.sum(acc[m * ACC + j]), mask=token < N)


# ---- STAGE2_APPEND_MARKER ----


@triton.jit
def _ihc_pre_stage2(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    num_tokens,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    TWO_HC: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    INV_K: tl.constexpr,
):
    # Vectorized-gate stage2 (2026-11-11): scale/base are loop-invariant and
    # are loaded once per program; the post gates are formed as one [HC]
    # vector sigmoid + one vector store and the post mixes as a [HC] vector
    # load; the pre gates stay scalar (consumed as FMA scalars by the y
    # stream, 4 independent sigmoids). This removed 10 redundant scalar
    # loads + 4 scalar sigmoids per token (stage2 -50% at N=8192).
    NUM_HB: tl.constexpr = (HIDDEN_SIZE + BLOCK_D - 1) // BLOCK_D
    j_offs = tl.arange(0, HC_MULT)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    base_pre = ()
    base_post_j = ()
    for j in tl.static_range(HC_MULT):
        base_pre = base_pre + (tl.load(base_ptr + j),)
        base_post_j = base_post_j + (tl.load(base_ptr + HC_MULT + j),)
    base_post = tl.load(base_ptr + HC_MULT + j_offs)
    for token in range(tl.program_id(0), num_tokens, tl.num_programs(0)):
        slot_base = partial_ptr + token * NUM_SPLITS * PARTIAL_STRIDE
        if NUM_SPLITS == 1:
            sqrsum = tl.load(slot_base)
            mixes_post = tl.load(slot_base + 1 + HC_MULT + j_offs)
            # Reduce the split slots in index order (fixed reduction order,
            # no atomics); prefill has a single split, read it directly.
            reciprocal_rms = tl.rsqrt(sqrsum * INV_K + NORM_EPS)
            post_gates = (
                MAGNITUDE
                * tl.sigmoid(mixes_post * (reciprocal_rms * post_scale) + base_post)
                + HC_EPS
            )
            tl.store(post_ptr + token * HC_MULT + j_offs, post_gates)
            x_row = x_ptr + token * K_TOTAL
            outputs = ()
            for hb in tl.static_range(NUM_HB):
                outputs = outputs + (tl.zeros((BLOCK_D,), dtype=tl.float32),)
            for j in tl.static_range(HC_MULT):
                mix_j = tl.load(slot_base + 1 + j)
                # Gates are formed once per token (they only depend on the
                # mixes); the +0.0-free product below feeds an FMA whose
                # operands are all fp32, which the backend fuses correctly.
                gate_j = (
                    tl.sigmoid(mix_j * (reciprocal_rms * pre_scale) + base_pre[j])
                    + HC_EPS
                )
                for hb in tl.static_range(NUM_HB):
                    hidden_offsets = hb * BLOCK_D + tl.arange(0, BLOCK_D)
                    channel = tl.load(
                        x_row + j * HIDDEN_SIZE + hidden_offsets,
                        mask=hidden_offsets < HIDDEN_SIZE,
                        other=0.0,
                    ).to(tl.float32)
                    new_outputs = ()
                    for hb2 in tl.static_range(NUM_HB):
                        if hb2 == hb:
                            new_outputs = new_outputs + (
                                outputs[hb2] + gate_j * channel,
                            )
                        else:
                            new_outputs = new_outputs + (outputs[hb2],)
                    outputs = new_outputs
        else:
            split_idx = tl.arange(0, BLOCK_S)
            split_mask = split_idx < NUM_SPLITS
            sqrsum = tl.sum(
                tl.load(
                    slot_base + split_idx * PARTIAL_STRIDE,
                    mask=split_mask,
                    other=0.0,
                )
            )
            mixes = ()
            for j in tl.static_range(TWO_HC):
                mixes = mixes + (
                    tl.sum(
                        tl.load(
                            slot_base + split_idx * PARTIAL_STRIDE + 1 + j,
                            mask=split_mask,
                            other=0.0,
                        )
                    ),
                )
            reciprocal_rms = tl.rsqrt(sqrsum * INV_K + NORM_EPS)
            pre_gates = ()
            for j in tl.static_range(HC_MULT):
                pre_gates = pre_gates + (
                    tl.sigmoid(mixes[j] * (reciprocal_rms * pre_scale) + base_pre[j])
                    + HC_EPS,
                )
                post_gate = (
                    MAGNITUDE
                    * tl.sigmoid(
                        mixes[HC_MULT + j] * (reciprocal_rms * post_scale)
                        + base_post_j[j]
                    )
                    + HC_EPS
                )
                tl.store(post_ptr + token * HC_MULT + j, post_gate)
            x_row = x_ptr + token * K_TOTAL
            outputs = ()
            for hb in tl.static_range(NUM_HB):
                outputs = outputs + (tl.zeros((BLOCK_D,), dtype=tl.float32),)
            for j in tl.static_range(HC_MULT):
                for hb in tl.static_range(NUM_HB):
                    hidden_offsets = hb * BLOCK_D + tl.arange(0, BLOCK_D)
                    channel = tl.load(
                        x_row + j * HIDDEN_SIZE + hidden_offsets,
                        mask=hidden_offsets < HIDDEN_SIZE,
                        other=0.0,
                    ).to(tl.float32)
                    new_outputs = ()
                    for hb2 in tl.static_range(NUM_HB):
                        if hb2 == hb:
                            new_outputs = new_outputs + (
                                outputs[hb2] + pre_gates[j] * channel,
                            )
                        else:
                            new_outputs = new_outputs + (outputs[hb2],)
                    outputs = new_outputs
        # Pre-gated channel sum over hidden blocks: y[n, :] = sum_j
        # H_pre[n, j] * x[n, j, :]. Channel-outer so each channel's hidden
        # row streams contiguously; the per-element operation order is the
        # same as the hb-outer form.
        for hb in tl.static_range(NUM_HB):
            hidden_offsets = hb * BLOCK_D + tl.arange(0, BLOCK_D)
            tl.store(
                output_ptr + token * HIDDEN_SIZE + hidden_offsets,
                outputs[hb].to(output_ptr.dtype.element_ty),
                mask=hidden_offsets < HIDDEN_SIZE,
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
    two_hc = 2 * hc_mult
    x_flat = x.reshape(num_tokens, k_total)

    if num_tokens < _DECODE_N_THRESHOLD:
        block_m = 1
        block_k = _decode_block_k(num_tokens)
        num_splits = triton.cdiv(k_total, block_k)
        stage1_warps = _STAGE1_DECODE_WARPS
    else:
        block_m = _PREFILL_BLOCK_M
        block_k = min(_PREFILL_BLOCK_K, triton.next_power_of_2(k_total))
        num_splits = 1
        stage1_warps = _STAGE1_PREFILL_WARPS
    k_chunk = block_k if num_tokens < _DECODE_N_THRESHOLD else k_total
    partial_stride = 1 + two_hc
    partial = torch.empty(
        (num_tokens, num_splits, partial_stride),
        dtype=torch.float32,
        device=x.device,
    )
    total_work = triton.cdiv(num_tokens, block_m) * num_splits
    _ihc_pre_stage1[(min(total_work, _GRID_TARGET),)](
        x_flat,
        weight,
        partial,
        num_tokens,
        K_TOTAL=k_total,
        K_CHUNK=k_chunk,
        TWO_HC=two_hc,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        PARTIAL_STRIDE=partial_stride,
        num_warps=stage1_warps,
    )
    _ihc_pre_stage2[(min(num_tokens, _GRID_TARGET),)](
        x,
        partial,
        scale,
        base,
        output,
        post,
        num_tokens,
        HIDDEN_SIZE=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        TWO_HC=two_hc,
        NUM_SPLITS=num_splits,
        PARTIAL_STRIDE=partial_stride,
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=_STAGE2_BLOCK_D,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        INV_K=1.0 / k_total,
        num_warps=_STAGE2_WARPS,
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

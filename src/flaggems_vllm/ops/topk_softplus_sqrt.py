# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from the vLLM project (https://github.com/vllm-project/vllm).
# Source: vllm/model_executor/layers/fused_moe/topk_softplus_sqrt_kernels.cu
#
# This Triton implementation is based on the CUDA kernel from vLLM 0.20.0.
# The kernel fuses softplus, sqrt, top-k selection, and optional renormalization
# for MoE gating in models like DeepSeek-V3/V4.

"""TopK Softplus-Sqrt gating kernel in Triton.

Optimized v27: num_warps=1 + all v19-v26 wins.
Key insight: For 256 experts, CUDA uses exactly 1 warp (32 threads) per row,
with each thread holding 8 elements. Using num_warps=4 adds warp scheduling
overhead without helping the 256-element reduction. Combining num_warps=1
(matching CUDA's single-warp-per-row) with tensor caching (v19), score-arithmetic
weight extraction (v20), and the max+compare index recovery (v26) should
minimize overhead.

Eliminates the store-load-store pattern for renormalization by storing weights
during the loop and re-reading with scale at the end.

TLE variant
-----------
When Triton's experimental TLE (Triton Language Extensions) API is available
(Triton >= 3.6.0, see `has_triton_tle` in flaggems_vllm.utils), an additional
pair of kernels (`_fused_topk_kernel_tle` / `_hash_kernel_tle`) is compiled.
These are algorithmically identical to the baseline kernels above; the only
difference is that intermediate topk-loop state is staged in explicit shared
memory (via `tle.gpu.alloc` / `tle.gpu.local_ptr`) instead of round-tripping
through global memory:

  - Dense path (`_fused_topk_kernel_tle`): the topk loop's
    store -> (later) load -> store pattern used to implement the two-pass
    renormalization is replaced with a single shared-memory scratch buffer,
    so only one global store happens at the end.
  - Hash path (`_hash_kernel_tle`): the O(BLOCK_E) masked-reduction gather
    used to fetch each selected expert's weight is replaced by staging the
    post-transform scores into shared memory once, then doing O(1) indexed
    loads per topk slot.

The TLE kernels are dispatched automatically when available and beneficial;
callers do not need to opt in explicitly.
"""

import logging

import triton
import triton.language as tl

from flaggems_vllm.utils import has_triton_tle

logger = logging.getLogger(__name__)

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False


@triton.jit
def _fused_topk_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts

    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
        tl.float32
    )

    # Fused softplus + sqrt
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    raw = tl.sqrt(x)

    # Scores for top-k selection (with optional bias)
    if HAS_BIAS:
        bias = tl.load(
            e_score_correction_bias_ptr + expert_offsets, mask=emask, other=0.0
        ).to(tl.float32)
        scores = raw + bias
    else:
        scores = raw
    scores = tl.where(emask, scores, -float("inf"))

    out_base = pid * topk
    weight_sum = 0.0

    for k_idx in tl.static_range(topk):
        max_score = tl.max(scores, axis=0)
        is_max = scores == max_score
        match_priority = tl.where(is_max, BLOCK_E - expert_offsets, 0)
        best_slot = BLOCK_E - tl.max(match_priority, axis=0)
        eidx = best_slot.to(tl.int32)

        if HAS_BIAS:
            bias_at_eidx = tl.load(e_score_correction_bias_ptr + eidx)
            w = max_score - bias_at_eidx
        else:
            w = max_score

        weight_sum += w
        tl.store(topk_weights_ptr + out_base + k_idx, w)
        tl.store(topk_indices_ptr + out_base + k_idx, eidx)
        tl.store(
            token_expert_indices_ptr + out_base + k_idx,
            (pid * topk + k_idx).to(tl.int32),
        )

        # Zero out winner
        scores = tl.where(expert_offsets == eidx, -float("inf"), scores)

    # Renormalize: re-read weights and apply scale
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor

    for k_idx in tl.static_range(topk):
        w = tl.load(topk_weights_ptr + out_base + k_idx)
        tl.store(topk_weights_ptr + out_base + k_idx, w * scale)


@triton.jit
def _hash_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    input_tokens_ptr,
    hash_indices_table_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Hash mode: expert indices come from lookup table."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts

    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
        tl.float32
    )

    # Fused softplus + sqrt
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    x = tl.sqrt(x)

    # Get expert indices from lookup table
    token_id = tl.load(input_tokens_ptr + pid)
    k_offsets = tl.arange(0, BLOCK_K)
    kmask = k_offsets < topk
    expert_ids = tl.load(
        hash_indices_table_ptr + token_id * topk + k_offsets, mask=kmask, other=0
    )

    # Gather weights for each selected expert
    weight_sum = 0.0
    weights = tl.zeros([BLOCK_K], dtype=tl.float32)

    for k_idx in tl.static_range(topk):
        eidx = tl.sum(tl.where(k_offsets == k_idx, expert_ids, 0))
        w = tl.sum(tl.where(expert_offsets == eidx, x, 0.0))
        weight_sum += w
        weights = tl.where(k_offsets == k_idx, w, weights)

    # Apply renormalization + scaling
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor
    weights = weights * scale

    # Single burst store
    out_base = pid * topk
    tl.store(topk_weights_ptr + out_base + k_offsets, weights, mask=kmask)
    tl.store(topk_indices_ptr + out_base + k_offsets, expert_ids, mask=kmask)
    tei = (pid * topk + k_offsets).to(tl.int32)
    tl.store(token_expert_indices_ptr + out_base + k_offsets, tei, mask=kmask)


# ---------------------------------------------------------------------------
# TLE variants (Triton experimental TLE API, Hopper+)
#
# Identical math/output contract to the baseline kernels above. The only
# difference is that per-token scratch state that the baseline round-trips
# through global memory is instead staged in explicit shared memory via
# `tle.gpu.alloc` / `tle.gpu.local_ptr`:
#
#   - `_fused_topk_kernel_tle`: stages the topk-loop weights in shared memory
#     instead of doing store -> load -> store through global memory for the
#     two-pass renormalization.
#   - `_hash_kernel_tle`: stages the post-transform scores in shared memory
#     once, turning each topk-slot gather into an O(1) indexed load instead
#     of an O(BLOCK_E) masked reduction over the whole expert row.
# ---------------------------------------------------------------------------

if HAS_TLE:

    @triton.jit
    def _fused_topk_kernel_tle(
        gating_ptr,
        topk_weights_ptr,
        topk_indices_ptr,
        token_expert_indices_ptr,
        e_score_correction_bias_ptr,
        num_tokens,
        num_experts: tl.constexpr,
        topk: tl.constexpr,
        renormalize: tl.constexpr,
        routed_scaling_factor,
        HAS_BIAS: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_tokens:
            return

        expert_offsets = tl.arange(0, BLOCK_E)
        emask = expert_offsets < num_experts

        row_base = pid * num_experts
        x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
            tl.float32
        )

        if HAS_BIAS:
            bias_full = tl.load(
                e_score_correction_bias_ptr + expert_offsets, mask=emask, other=0.0
            ).to(tl.float32)
        else:
            bias_full = None

        x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        raw = tl.sqrt(x)

        if HAS_BIAS:
            scores = raw + bias_full
        else:
            scores = raw
        scores = tl.where(emask, scores, -float("inf"))

        # Shared-memory scratch for the topk-loop weights. Replaces the
        # global-memory store -> (later) load -> store pattern used to
        # implement the two-pass renormalization in the baseline kernel.
        w_smem = tle.gpu.alloc(
            [BLOCK_TOPK],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        k_range = tl.arange(0, BLOCK_TOPK)
        w_smem_ptrs = tle.gpu.local_ptr(w_smem, (k_range,))

        out_base = pid * topk
        weight_sum = 0.0

        for k_idx in tl.static_range(topk):
            max_score = tl.max(scores, axis=0)
            is_max = scores == max_score
            match_priority = tl.where(is_max, BLOCK_E - expert_offsets, 0)
            best_slot = BLOCK_E - tl.max(match_priority, axis=0)
            eidx = best_slot.to(tl.int32)

            if HAS_BIAS:
                bias_at_eidx = tl.load(e_score_correction_bias_ptr + eidx)
                w = max_score - bias_at_eidx
            else:
                w = max_score

            weight_sum += w
            # Shared memory instead of global memory: single-element store
            # via local_ptr, addressed with a k_idx-selecting index so the
            # loop stays branch-free / vectorizable the same way as the
            # baseline.
            tl.store(
                tle.gpu.local_ptr(w_smem, (tl.full([1], k_idx, tl.int32),)),
                tl.full([1], 0.0, tl.float32) + w,
            )
            tl.store(topk_indices_ptr + out_base + k_idx, eidx)
            tl.store(
                token_expert_indices_ptr + out_base + k_idx,
                (pid * topk + k_idx).to(tl.int32),
            )

            scores = tl.where(expert_offsets == eidx, -float("inf"), scores)

        if renormalize:
            scale = routed_scaling_factor / tl.where(
                weight_sum > 0.0, weight_sum, 1.0
            )
        else:
            scale = routed_scaling_factor

        # Single read-back from shared memory (not global memory) + single
        # global-memory store of the final scaled weights.
        kmask = k_range < topk
        w_all = tl.load(w_smem_ptrs, mask=kmask, other=0.0)
        tl.store(topk_weights_ptr + out_base + k_range, w_all * scale, mask=kmask)

    @triton.jit
    def _hash_kernel_tle(
        gating_ptr,
        topk_weights_ptr,
        topk_indices_ptr,
        token_expert_indices_ptr,
        e_score_correction_bias_ptr,
        input_tokens_ptr,
        hash_indices_table_ptr,
        num_tokens,
        num_experts: tl.constexpr,
        topk: tl.constexpr,
        renormalize: tl.constexpr,
        routed_scaling_factor,
        HAS_BIAS: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_tokens:
            return

        expert_offsets = tl.arange(0, BLOCK_E)
        emask = expert_offsets < num_experts

        row_base = pid * num_experts
        x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
            tl.float32
        )

        x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        x = tl.sqrt(x)

        # Stage the post-transform scores into shared memory once, so each
        # of the `topk` gathers below becomes an O(1) indexed load instead
        # of an O(BLOCK_E) masked reduction over the whole row.
        x_smem = tle.gpu.alloc(
            [BLOCK_E],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        tl.store(tle.gpu.local_ptr(x_smem, (expert_offsets,)), x, mask=emask)

        token_id = tl.load(input_tokens_ptr + pid)
        k_offsets = tl.arange(0, BLOCK_K)
        kmask = k_offsets < topk
        expert_ids = tl.load(
            hash_indices_table_ptr + token_id * topk + k_offsets,
            mask=kmask,
            other=0,
        )

        weight_sum = 0.0
        weights = tl.zeros([BLOCK_K], dtype=tl.float32)

        for k_idx in tl.static_range(topk):
            eidx = tl.sum(tl.where(k_offsets == k_idx, expert_ids, 0))
            # O(1) shared-memory gather instead of an O(BLOCK_E) masked sum.
            eidx_vec = tl.full([1], 0, tl.int32) + eidx
            w = tl.sum(tl.load(tle.gpu.local_ptr(x_smem, (eidx_vec,))))
            weight_sum += w
            weights = tl.where(k_offsets == k_idx, w, weights)

        if renormalize:
            scale = routed_scaling_factor / tl.where(
                weight_sum > 0.0, weight_sum, 1.0
            )
        else:
            scale = routed_scaling_factor
        weights = weights * scale

        out_base = pid * topk
        tl.store(topk_weights_ptr + out_base + k_offsets, weights, mask=kmask)
        tl.store(topk_indices_ptr + out_base + k_offsets, expert_ids, mask=kmask)
        tei = (pid * topk + k_offsets).to(tl.int32)
        tl.store(token_expert_indices_ptr + out_base + k_offsets, tei, mask=kmask)

else:
    _fused_topk_kernel_tle = None
    _hash_kernel_tle = None


def _topk_tle_available(gating_output) -> bool:
    """Whether the TLE shared-memory kernels can/should be used.

    Mirrors the dispatch guard used elsewhere in this repo (e.g.
    moe_align_block_size.py): requires the TLE API to be importable and a
    CUDA tensor. No additional compute-capability gate is applied here,
    matching the convention used by sibling ops in this file's directory.
    """
    return HAS_TLE and gating_output.device.type == "cuda"


def _topk_softplus_sqrt_impl(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias,
    input_ids,
    tid2eid,
    use_tle,
):
    """Shared dispatch body. `use_tle` selects baseline vs TLE kernels."""
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]

    if num_tokens == 0:
        return

    BLOCK_E = triton.next_power_of_2(num_experts)

    if input_ids is not None and tid2eid is not None:
        BLOCK_K = triton.next_power_of_2(topk)
        grid = (num_tokens,)
        kernel = _hash_kernel_tle if use_tle else _hash_kernel
        kernel[grid](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            correction_bias if correction_bias is not None else gating_output,
            input_ids,
            tid2eid,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=correction_bias is not None,
            BLOCK_E=BLOCK_E,
            BLOCK_K=BLOCK_K,
            num_warps=1,
            num_stages=1,
        )
        return

    grid = (num_tokens,)
    if use_tle:
        BLOCK_TOPK = triton.next_power_of_2(topk)
        _fused_topk_kernel_tle[grid](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            correction_bias if correction_bias is not None else gating_output,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=correction_bias is not None,
            BLOCK_E=BLOCK_E,
            BLOCK_TOPK=BLOCK_TOPK,
            num_warps=1,
            num_stages=1,
        )
    else:
        _fused_topk_kernel[grid](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            correction_bias if correction_bias is not None else gating_output,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=correction_bias is not None,
            BLOCK_E=BLOCK_E,
            num_warps=1,
            num_stages=1,
        )


def topk_softplus_sqrt(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    """Fused topk + softplus + sqrt kernel for MoE gating.

    Interface aligned with vLLM CUDA operator:
        void topk_softplus_sqrt(Tensor& topk_weights, Tensor& topk_indices,
            Tensor& token_expert_indices, Tensor& gating_output,
            bool renormalize, double routed_scaling_factor,
            const c10::optional<Tensor>& correction_bias,
            const c10::optional<Tensor>& input_ids,
            const c10::optional<Tensor>& tid2eid);

    Args:
        topk_weights: Output tensor [num_tokens, topk], dtype float32
        topk_indices: Output tensor [num_tokens, topk], dtype int32
        token_expert_indices: Output tensor [num_tokens, topk], dtype int32
        gating_output: Gating logits [num_tokens, num_experts]
        renormalize: Whether to renormalize weights
        routed_scaling_factor: Scaling factor for final weights
        correction_bias: Optional bias for expert scores [num_experts]
        input_ids: Token IDs for hash mode [num_tokens]
        tid2eid: Hash table mapping tokens to expert indices

    When Triton's TLE API is available and the input is a CUDA tensor, a
    shared-memory-staged TLE variant of the kernel is used automatically
    (see module docstring); the output is equivalent to the baseline
    kernel, only the memory access pattern differs. Use
    `topk_softplus_sqrt_baseline` / `topk_softplus_sqrt_tle` instead of this
    function to force a specific variant (e.g. for benchmarking/testing).
    """
    logger.debug("GEMS TOPK_SOFTPLUS_SQRT")
    use_tle = _topk_tle_available(gating_output)
    _topk_softplus_sqrt_impl(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
        use_tle=use_tle,
    )


def topk_softplus_sqrt_baseline(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    """Same as `topk_softplus_sqrt` but always uses the global-memory
    baseline kernel, regardless of TLE availability. Intended for
    benchmarking/testing the TLE speedup, not for general use.
    """
    logger.debug("GEMS TOPK_SOFTPLUS_SQRT (forced baseline)")
    _topk_softplus_sqrt_impl(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
        use_tle=False,
    )


def topk_softplus_sqrt_tle(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    """Same as `topk_softplus_sqrt` but always uses the shared-memory TLE
    kernel. Raises if TLE is unavailable in the current environment (no
    silent fallback), so benchmarks/tests fail loudly instead of quietly
    measuring the baseline.
    """
    logger.debug("GEMS TOPK_SOFTPLUS_SQRT (forced TLE)")
    if not (HAS_TLE and gating_output.device.type == "cuda"):
        raise RuntimeError(
            "topk_softplus_sqrt_tle requires Triton TLE support "
            "(triton.experimental.tle) and a CUDA tensor; "
            f"HAS_TLE={HAS_TLE}, device={gating_output.device}"
        )
    _topk_softplus_sqrt_impl(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
        use_tle=True,
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from the vLLM project (https://github.com/vllm-project/vllm).
# Source: vllm/model_executor/layers/fused_moe/topk_softplus_sqrt_kernels.cu
#
# This Triton implementation is based on the CUDA kernel from vLLM 0.20.0.
# The kernel fuses softplus, sqrt, top-k selection, and optional renormalization
# for MoE gating in models like DeepSeek-V3/V4.

"""TopK Softplus-Sqrt gating kernel in Triton (Ascend).

The original implementation launched one program per token and ran the whole
softplus/sqrt/top-k pipeline on a single row.  On Ascend that leaves the vector
units almost idle -- a 256-expert row is only 1 KiB -- so the per-token block
scheduling and reduction cost dominate: 4096 tokens took ~250 us while the
kernel below takes ~55 us for the same input.

This version follows the structure of TileKernels' ``topk_gate`` Ascend kernel:

* **Row tiling**: each program owns ``BLOCK_M`` tokens, so a whole tile of
  logits is loaded with one wide ``tl.load`` and ``softplus``/``sqrt``/bias are
  applied vector-wise over the ``[BLOCK_M, BLOCK_E]`` tile.
* **Persistent grid**: the grid is capped at the number of AI vector cores and
  each program walks a strided slice of the row blocks, exactly like the
  TileKernels persistent kernel.  Launch/scheduling cost no longer scales with
  the token count.
* **Two store strategies**: the top-k loop produces one ``[BLOCK_M]`` vector per
  rank.  When the row tile is wide (small expert counts) writing those vectors
  straight to HBM is cheapest (``PER_K_STORE``); when the tile is narrow (large
  expert counts) assembling a ``[BLOCK_M, BLOCK_K]`` tile and doing one
  contiguous store per output tensor wins.
* **Index recovery**: the winner index is recovered with a masked integer
  reduction (``min`` over the experts that hit the maximum), which keeps the
  original "smallest expert id wins a tie" semantics.
"""

import logging

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle as tle

    HAS_TLE = True
except ImportError:
    tle = None
    HAS_TLE = False

logger = logging.getLogger(__name__)

# Maximum number of fp32 elements in the ``[BLOCK_M, BLOCK_E]`` score tile.
# The Ascend lowering rejects single tensors larger than 32 KiB (a [32, 512] fp32
# tile fails to compile while [32, 256] works), so stay at 32 KiB of scores.
_MAX_TILE_ELEMS = 8192

# A row tile at least this wide makes the per-rank HBM stores cheaper than
# assembling a [BLOCK_M, BLOCK_K] tile first.
_PER_K_STORE_MIN_BLOCK_M = 32

# Fallback when the runtime cannot report the vector core count (910B class).
_DEFAULT_VECTOR_CORES = 48

_CACHED_VECTOR_CORE_NUM = None


def _next_pow2(value):
    """`triton.next_power_of_2` without the ~1.2 us python call per use."""
    return 1 << (value - 1).bit_length()


def _cdiv(a, b):
    """`triton.cdiv` without the ~1.3 us python call per use."""
    return -(-a // b)


def _get_vector_core_num():
    """Number of AI vector cores available to the current device."""
    global _CACHED_VECTOR_CORE_NUM
    if _CACHED_VECTOR_CORE_NUM is None:
        try:
            import torch_npu  # noqa: F401

            device_index = torch.npu.current_device()
            limits = torch.npu.get_device_limit(device_index)
            _CACHED_VECTOR_CORE_NUM = int(limits["vector_core_num"])
        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError):
            _CACHED_VECTOR_CORE_NUM = 0
    return _CACHED_VECTOR_CORE_NUM or _DEFAULT_VECTOR_CORES


def _plan_launch(num_tokens, num_experts):
    """Pick the row tile, the number of programs and the store strategy."""
    if num_experts * 2 > _MAX_TILE_ELEMS:
        # Even a two-row tile would no longer fit the Ascend UB budget.  No
        # vLLM MoE model routes to more than 512 experts; fail loudly instead
        # of silently producing a kernel that cannot be compiled.
        raise NotImplementedError(
            "topk_softplus_sqrt (Ascend) supports at most "
            f"{_MAX_TILE_ELEMS // 2} experts, got {num_experts}"
        )
    block_e = _next_pow2(num_experts) if num_experts > 1 else 1
    max_block_m = max(1, _MAX_TILE_ELEMS // block_e)
    num_cores = _get_vector_core_num()

    # Spread the rows over the vector cores; a tile needs at least two rows
    # because the biased weight path gathers the correction bias per winner and
    # that gather does not lower for a one-row tile.
    rows_per_core = _cdiv(num_tokens, num_cores)
    block_m = min(max(2, _next_pow2(rows_per_core)), max_block_m)
    block_m = max(2, block_m)

    num_blocks = _cdiv(num_tokens, block_m)
    grid = (min(num_cores, num_blocks),)
    per_k_store = block_m >= _PER_K_STORE_MIN_BLOCK_M
    return block_e, block_m, grid, per_k_store


@triton.jit
def _fused_topk_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    num_tokens,
    routed_scaling_factor,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    PER_K_STORE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(num_tokens, BLOCK_M)

    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts
    k_offsets = tl.arange(0, BLOCK_K)
    kmask = k_offsets < topk

    if HAS_BIAS:
        bias = tl.load(
            e_score_correction_bias_ptr + expert_offsets, mask=emask, other=0.0
        ).to(tl.float32)

    # Persistent walk over the row blocks: one program per vector core.
    for block_id in range(pid, num_blocks, NUM_CORES):
        rows = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < num_tokens
        tile_mask = row_mask[:, None] & emask[None, :]

        offsets = rows[:, None] * num_experts + expert_offsets[None, :]
        x = tl.load(gating_ptr + offsets, mask=tile_mask, other=0.0).to(tl.float32)

        # Fused softplus + sqrt.
        x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        scores = tl.sqrt(x)
        # Bias only shifts the ranking; the weight below is the raw score.
        if HAS_BIAS:
            scores = scores + bias[None, :]
        scores = tl.where(tile_mask, scores, -float("inf"))

        out_offsets = rows[:, None] * topk + k_offsets[None, :]
        out_mask = row_mask[:, None] & kmask[None, :]

        weight_sum = tl.zeros([BLOCK_M], tl.float32)
        if PER_K_STORE:
            # token_expert_indices is a plain range, so it can be written as a
            # single contiguous tile instead of one strided store per rank.
            tl.store(
                token_expert_indices_ptr + out_offsets,
                out_offsets.to(tl.int32),
                mask=out_mask,
            )
            for k in tl.static_range(topk):
                max_score = tl.max(scores, axis=1)
                is_max = scores == max_score[:, None]
                eidx = tl.min(
                    tl.where(is_max, expert_offsets[None, :], BLOCK_E), axis=1
                )
                if HAS_BIAS:
                    w = max_score - tl.load(e_score_correction_bias_ptr + eidx)
                else:
                    w = max_score
                weight_sum += w
                rank_offset = rows * topk + k
                if renormalize:
                    # Weights are rescaled once the whole row is known.
                    tl.store(topk_weights_ptr + rank_offset, w, mask=row_mask)
                else:
                    tl.store(
                        topk_weights_ptr + rank_offset,
                        w * routed_scaling_factor,
                        mask=row_mask,
                    )
                tl.store(topk_indices_ptr + rank_offset, eidx, mask=row_mask)
                scores = tl.where(
                    expert_offsets[None, :] == eidx[:, None], -float("inf"), scores
                )
            if renormalize:
                scale = routed_scaling_factor / tl.where(
                    weight_sum > 0.0, weight_sum, 1.0
                )
                for k in tl.static_range(topk):
                    rank_offset = rows * topk + k
                    w = tl.load(topk_weights_ptr + rank_offset, mask=row_mask)
                    tl.store(topk_weights_ptr + rank_offset, w * scale, mask=row_mask)
        else:
            weights = tl.zeros([BLOCK_M, BLOCK_K], tl.float32)
            indices = tl.zeros([BLOCK_M, BLOCK_K], tl.int32)
            for k in tl.static_range(topk):
                max_score = tl.max(scores, axis=1)
                is_max = scores == max_score[:, None]
                eidx = tl.min(
                    tl.where(is_max, expert_offsets[None, :], BLOCK_E), axis=1
                )
                if HAS_BIAS:
                    w = max_score - tl.load(e_score_correction_bias_ptr + eidx)
                else:
                    w = max_score
                weight_sum += w
                weights = tl.where(k_offsets[None, :] == k, w[:, None], weights)
                indices = tl.where(k_offsets[None, :] == k, eidx[:, None], indices)
                scores = tl.where(
                    expert_offsets[None, :] == eidx[:, None], -float("inf"), scores
                )
            if renormalize:
                scale = routed_scaling_factor / tl.where(
                    weight_sum > 0.0, weight_sum, 1.0
                )
                weights = weights * scale[:, None]
            else:
                weights = weights * routed_scaling_factor
            tl.store(topk_weights_ptr + out_offsets, weights, mask=out_mask)
            tl.store(topk_indices_ptr + out_offsets, indices, mask=out_mask)
            tl.store(
                token_expert_indices_ptr + out_offsets,
                out_offsets.to(tl.int32),
                mask=out_mask,
            )


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


if HAS_TLE:

    @triton.jit
    def _fused_topk_kernel_ascend_tle(
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

        w_tensor = tl.full([BLOCK_K], 0.0, tl.float32)

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

            # Write w into the running tensor at position k_idx (replaces
            # the HBM store of the baseline's first loop).
            w_tensor = tle.dsa.insert_slice(
                w_tensor,
                tl.full([1], 0.0, tl.float32) + w,
                offsets=[k_idx],
                sizes=[1],
                strides=[1],
            )

            tl.store(topk_indices_ptr + out_base + k_idx, eidx)
            tl.store(
                token_expert_indices_ptr + out_base + k_idx,
                (pid * topk + k_idx).to(tl.int32),
            )

            # Zero out winner
            scores = tl.where(expert_offsets == eidx, -float("inf"), scores)

        # Renormalize directly on the tensor -- no HBM round trip.
        if renormalize:
            scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
        else:
            scale = routed_scaling_factor

        w_tensor = w_tensor * scale

        k_offsets = tl.arange(0, BLOCK_K)
        kmask = k_offsets < topk
        tl.store(topk_weights_ptr + out_base + k_offsets, w_tensor, mask=kmask)

else:
    _fused_topk_kernel_ascend_tle = None


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
    use_ascend_tle=False,
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
        use_ascend_tle: If True (and the Ascend TLE kernel is available),
            use the TLE kernel for the non-hash path instead of the tiled
            persistent kernel. Ignored for the hash-mode path, which always
            uses `_hash_kernel`. Has no effect if `triton.experimental.tle`
            is not importable (falls back to the tiled kernel, e.g. on
            non-Ascend backends).
    """
    logger.debug("GEMS TOPK_SOFTPLUS_SQRT")
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]

    if num_tokens == 0:
        return

    BLOCK_E = _next_pow2(num_experts) if num_experts > 1 else 1

    if input_ids is not None and tid2eid is not None:
        BLOCK_K = _next_pow2(topk) if topk > 1 else 1
        grid = (num_tokens,)
        _hash_kernel[grid](
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

    if use_ascend_tle and HAS_TLE:
        BLOCK_K = _next_pow2(topk) if topk > 1 else 1
        grid = (num_tokens,)
        _fused_topk_kernel_ascend_tle[grid](
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
            BLOCK_K=BLOCK_K,
            num_warps=1,
            num_stages=1,
        )
        return

    BLOCK_E, BLOCK_M, grid, per_k_store = _plan_launch(num_tokens, num_experts)
    BLOCK_K = _next_pow2(topk) if topk > 1 else 1
    _fused_topk_kernel[grid](
        gating_output,
        topk_weights,
        topk_indices,
        token_expert_indices,
        correction_bias if correction_bias is not None else gating_output,
        num_tokens=num_tokens,
        routed_scaling_factor=routed_scaling_factor,
        num_experts=num_experts,
        topk=topk,
        renormalize=renormalize,
        HAS_BIAS=correction_bias is not None,
        BLOCK_E=BLOCK_E,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        NUM_CORES=grid[0],
        PER_K_STORE=per_k_store,
        num_warps=1,
        num_stages=1,
    )

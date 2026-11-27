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

Two implementations share this module:

1. **Hardware-sort path (default on 910B, needs TLE)** — ``_softplus_sqrt_score_kernel``
   materialises ``sqrt(softplus(x)) [+ bias]`` into an fp32 scratch buffer with
   one wide vector pass, then ``_tle_sort_topk_kernel`` hands each row to the
   Ascend sort engine (``tle.dsa`` ``sort_1d_pack``/``unpack_sort``).  The sort
   engine returns the exact top-k values and indices, so the baseline's
   per-rank max/mask/index-recovery reductions disappear entirely.  On 910B
   (bf16, 256 experts, topk=6) this is ~4-6x faster than the reduction kernel
   for 1024-4096 tokens and 1.1-1.3x faster than vllm-ascend's CANN
   ``moe_gating_top_k_hash`` kernel for 128-4096 tokens.  It is also the only
   path that compiles for the common shapes on the current BiShengHIR
   toolchain.

2. **Reduction fallback** — the original fused persistent-tiled kernel.  Each
   program owns ``BLOCK_M`` rows: one wide ``tl.load`` plus vector softplus/
   sqrt/bias over ``[BLOCK_M, BLOCK_E]``, then a per-rank max + masked integer
   ``min`` to recover the winner index ("smallest expert id wins a tie").  It
   is used when TLE is unavailable or the shape is outside the sort engine's
   comfort zone (very large expert count or topk).

Both paths keep the vLLM semantics: bias only shifts the ranking, routed
weights come from the unbiased score, and renormalization/scaling happen once
the top-k set is known.
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

# The current BiShengHIR toolchain overflows the 192 KiB UB budget for a
# [32, 256] fp32 tile once the softplus/sqrt/reduction temporaries are
# materialised, so the tiled fallback never uses more than 16 rows per tile
# even though the 32 KiB score tile itself would fit.
_MAX_TILE_ROWS = 16

# Hardware-sort path (``tle.dsa`` custom ops).  Each row is sorted by the
# Ascend sort engine, so this path only makes sense while the row fits the
# sort scratch budget and topk is small enough to keep the proposal buffer
# in UB.
_TLE_SORT_TILE_ELEMS = 8192
_TLE_SORT_MAX_EXPERTS = 1024
_TLE_SORT_MAX_TOPK = 32
_TLE_SORT_ROWS = 8


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
    block_m = min(max(2, _next_pow2(rows_per_core)), max_block_m, _MAX_TILE_ROWS)
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


@triton.jit
def _softplus_sqrt_score_kernel(
    gating_ptr,
    e_score_correction_bias_ptr,
    score_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Materialise ``sqrt(softplus(x)) [+ bias]`` as fp32 scores.

    The score buffer is row-padded to ``BLOCK_E`` experts so that the sort
    kernel can treat every row as a contiguous power-of-two segment.  Padding
    experts are filled with ``-inf`` so they can never win a rank.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    eo = tl.arange(0, BLOCK_E)
    rm = rows < num_tokens
    em = eo < num_experts
    mask = rm[:, None] & em[None, :]

    x = tl.load(
        gating_ptr + rows[:, None] * num_experts + eo[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    s = tl.sqrt(x)
    if HAS_BIAS:
        bias = tl.load(eo + e_score_correction_bias_ptr, mask=em, other=0.0).to(
            tl.float32
        )
        s = s + bias[None, :]
    s = tl.where(em[None, :], s, -float("inf"))
    tl.store(score_ptr + rows[:, None] * BLOCK_E + eo[None, :], s, mask=rm[:, None])


if HAS_TLE:

    @triton.jit
    def _tle_sort_topk_kernel(
        e_score_correction_bias_ptr,
        score_ptr,
        topk_weights_ptr,
        topk_indices_ptr,
        token_expert_indices_ptr,
        num_tokens,
        routed_scaling_factor,
        num_experts: tl.constexpr,
        topk: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        NUM_CORES: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        renormalize: tl.constexpr,
    ):
        """Per-row hardware sort (``tle.dsa`` sort engine) + weight assembly.

        Each program owns ``BLOCK_M`` rows: one contiguous DMA copy brings the
        tile from HBM into UB, then every row is sorted by ``sort_1d_pack`` and
        decoded by ``unpack_sort``.  ``tl.static_range`` keeps the per-row
        sub-view offsets constexpr while still sharing the tile copy.

        The bias only shifts the ranking, so the routed weight is recovered
        from the *unbiased* logit as ``value - bias[winner]``.  ``value`` is
        the exact fp32 score carried by the proposal and the bias of the
        selected experts is gathered from the UB-resident bias vector, so the
        packed proposal layout stays intact and no second score buffer is
        needed.
        """
        pid = tl.program_id(0)
        kk = tl.arange(0, BLOCK_K)
        kmask = kk < topk
        tile = tl.arange(0, BLOCK_M * BLOCK_E)

        # BASE sort path scratch: 4 * segment length floats.
        tmp = tl.zeros([BLOCK_E * 4], dtype=tl.float32)
        buf_ub = tle.dsa.alloc(
            [BLOCK_M * BLOCK_E], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
        )
        seg_ub = tle.dsa.alloc(
            [2 * BLOCK_K], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
        )
        dval = tl.zeros([BLOCK_K], dtype=tl.float32)
        didx = tl.zeros([BLOCK_K], dtype=tl.int32)
        if HAS_BIAS:
            # Bias is tiny ([E]) and reused by every row: keep it in UB and
            # gather from there instead of issuing a scattered HBM load per
            # selected expert.
            bias_vec = tl.load(
                e_score_correction_bias_ptr + tl.arange(0, BLOCK_E),
                mask=tl.arange(0, BLOCK_E) < num_experts,
                other=0.0,
            ).to(tl.float32)

        num_blocks = tl.cdiv(num_tokens, BLOCK_M)
        for blk in range(pid, num_blocks, NUM_CORES):
            row0 = blk * BLOCK_M
            tle.dsa.copy(score_ptr + row0 * BLOCK_E + tile, buf_ub, [BLOCK_M * BLOCK_E])
            # static_range keeps the sub-view offsets constexpr (required by
            # tle.dsa.subview) while the single tile copy above is amortised
            # over all BLOCK_M rows.
            for r in tl.static_range(BLOCK_M):
                row = row0 + r
                if row < num_tokens:
                    row_ub = tle.dsa.subview(buf_ub, [r * BLOCK_E], [BLOCK_E], [1])
                    tle.dsa.ascend.raw(
                        "sort_1d_pack",
                        tle.dsa.to_tensor(row_ub),
                        tmp,
                        True,
                        BLOCK_K,
                        0,
                        0,
                        out=tle.dsa.to_tensor(seg_ub),
                    )
                    dval, didx = tle.dsa.ascend.raw(
                        "unpack_sort",
                        tle.dsa.to_tensor(seg_ub),
                        BLOCK_K,
                        out=[dval, didx],
                    )

                    if HAS_BIAS:
                        # The proposal value is the exact biased score and the
                        # bias only shifted the ranking, so the routed weight
                        # is the unbiased score: value - bias[winner].  This
                        # avoids re-running softplus/sqrt per selected expert.
                        bg = tl.gather(bias_vec, didx, 0)
                        w = dval - bg
                    else:
                        w = dval

                    if renormalize:
                        wsum = tl.sum(tl.where(kmask, w, 0.0))
                        scale = routed_scaling_factor / tl.where(wsum > 0.0, wsum, 1.0)
                    else:
                        scale = routed_scaling_factor

                    out = row * topk + kk
                    tl.store(topk_weights_ptr + out, w * scale, mask=kmask)
                    tl.store(topk_indices_ptr + out, didx, mask=kmask)
                    tl.store(
                        token_expert_indices_ptr + out,
                        (row * topk + kk).to(tl.int32),
                        mask=kmask,
                    )

else:
    _tle_sort_topk_kernel = None


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


def _can_use_tle_sort(num_experts, topk):
    """Whether the hardware-sort path can handle this expert/topk shape."""
    if not HAS_TLE:
        return False
    if num_experts < 1 or num_experts > _TLE_SORT_MAX_EXPERTS:
        return False
    if topk < 1 or topk > _TLE_SORT_MAX_TOPK:
        return False
    return True


def _tle_sort_path(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias,
):
    """Two-stage softplus/sqrt + hardware-sort path (see the kernel docs)."""
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]

    block_e = _next_pow2(num_experts) if num_experts > 1 else 1
    block_k = _next_pow2(topk) if topk > 1 else 1
    max_rows = max(1, _TLE_SORT_TILE_ELEMS // block_e)
    sort_rows = min(_TLE_SORT_ROWS, max_rows)
    score_rows = min(32, max_rows)
    num_cores = _get_vector_core_num()

    # Row-pad the score buffer so the sort kernel's tile DMA never reads past
    # the last token.  Padding rows are written by the score kernel with
    # ``-inf`` and skipped by the sort kernel.
    n_pad = _cdiv(num_tokens, sort_rows) * sort_rows
    scores = torch.empty(
        (n_pad, block_e), dtype=torch.float32, device=gating_output.device
    )
    has_bias = correction_bias is not None
    bias_arg = correction_bias if has_bias else gating_output

    _softplus_sqrt_score_kernel[(_cdiv(num_tokens, score_rows),)](
        gating_output,
        bias_arg,
        scores,
        num_tokens,
        num_experts=num_experts,
        BLOCK_E=block_e,
        BLOCK_M=score_rows,
        HAS_BIAS=has_bias,
        num_warps=1,
        num_stages=1,
    )

    num_blocks = _cdiv(num_tokens, sort_rows)
    grid = (min(num_cores, num_blocks),)
    _tle_sort_topk_kernel[grid](
        bias_arg,
        scores,
        topk_weights,
        topk_indices,
        token_expert_indices,
        num_tokens,
        routed_scaling_factor,
        num_experts=num_experts,
        topk=topk,
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        BLOCK_M=sort_rows,
        NUM_CORES=grid[0],
        HAS_BIAS=has_bias,
        renormalize=renormalize,
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
            force the original per-token TLE kernel for the non-hash path.
            The default path is the faster two-stage hardware-sort
            implementation, which shares the same TLE dependency. Ignored for
            the hash-mode path, which always uses `_hash_kernel`. Has no
            effect if `triton.experimental.tle` is not importable (falls back
            to the tiled kernel, e.g. on non-Ascend backends).
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

    # Default path: the Ascend sort engine is several times faster than the
    # reduction-based tiled kernel on 910B (a 256-expert row sorted by
    # ``sort_1d_pack`` vs. 12 full-width reductions per row).
    if _can_use_tle_sort(num_experts, topk):
        _tle_sort_path(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            routed_scaling_factor,
            correction_bias,
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""fused_indexer_weight_scale: the GLM-5-next sparse indexer weight scaling.

Aligns with the vLLM leaf ``_fused_indexer_weight_scale`` in
``vllm/models/glm5next/common/attention.py`` (present from v0.31.0, unchanged
through v0.32.0 and main)::

    out[t, h] = (weights[t, h] * q_scale[t, h, 0]) * scale      # fp32 out

The multiplication order is load-bearing: ``(w * s) * scale`` and
``w * (s * scale)`` differ in the last fp32 ulp, and the accuracy gate is
atol=1e-6 against the native result. On NVIDIA the native leaf is compiled
(``current_platform.simple_compile_backend == "inductor"``), so the baseline is
itself a Triton elementwise kernel; the arithmetic below is identical to the
one Inductor emits and the outputs come out bit-exact.

Two paths cover the contract:

* ``_flat_kernel`` for the fast path -- both inputs contiguous fp32 -- which
  needs no index arithmetic;
* ``_strided_kernel`` for everything else: strided views, independently strided
  inputs, non-zero storage offsets, and fp16/bf16 weights promoted to fp32.

No TLE primitive is used. The op is elementwise with no reduction and no
matmul, and the trace shapes are tiny (all ``[T, 32]``, T <= 76, so at most
2432 elements and 29 KB of traffic), which makes it launch-bound rather than
bandwidth-bound; ``triton.experimental.tle``'s NVIDIA ``gpu.*`` primitives
express warp specialization and async pipelining, and there is no stage here
for them to overlap.
"""

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# The flat path's launch configuration, size-independent on purpose.
#
# NOTE (autotune exemption, workflow.md Step 5): this op deliberately does not
# go through ``runtime.get_tuned_config`` + ``@libtuner``. Calibrated on an
# H800 by timing native and all 16 combinations of BLOCK in {128, 256, 512,
# 1024} x num_warps in {1, 2, 4, 8} inside shared measurement rounds, so every
# ratio carries the machine's clock drift identically: (128, 2) is at or below
# parity with native on all ten trace shapes (0.975-1.001), and the configs are
# within ~2% of each other everywhere except BLOCK >= 1024 (1.24-1.36). With
# the spread that small and one config dominating, a tuner would only add a
# re-tune on every newly seen token count -- which in decode is every step.
# (An earlier per-size table fitted in *unpaired* measurement suggested a 5%
# spread; paired measurement showed that was noise.)
#
# The loads stay masked. A mask-free variant specialized on numel % BLOCK == 0
# was measured and dropped: median gain 0.9991 over the four trace shapes it
# applies to, against control shapes -- where both sides run the same kernel --
# scattering over 0.986-1.008. Below the noise floor, so it would have bought a
# second compiled variant for nothing.
FLAT_BLOCK = 128
FLAT_NUM_WARPS = 2
# Tile budget for the strided path: BLOCK_T * BLOCK_H elements per program.
_STRIDED_TILE = 2048
_STRIDED_MIN_BLOCK_H = 16
_STRIDED_NUM_WARPS = 4


@triton.jit
def _flat_kernel(w_ptr, s_ptr, o_ptr, scale, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    w = tl.load(w_ptr + offsets, mask=mask)
    s = tl.load(s_ptr + offsets, mask=mask)
    tl.store(o_ptr + offsets, (w * s) * scale, mask=mask)


@triton.jit
def _strided_kernel(
    w_ptr,
    s_ptr,
    o_ptr,
    scale,
    tokens,
    heads,
    w_stride_t,
    w_stride_h,
    s_stride_t,
    s_stride_h,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = (tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T))[:, None]
    h = tl.arange(0, BLOCK_H)[None, :]
    mask = (t < tokens) & (h < heads)
    # Promote before multiplying: the native expression upcasts through the
    # fp32 q_scale, so a half-precision product would round differently.
    w = tl.load(w_ptr + t * w_stride_t + h * w_stride_h, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + t * s_stride_t + h * s_stride_h, mask=mask).to(tl.float32)
    # The output is freshly allocated and contiguous, so its strides are known.
    tl.store(o_ptr + t * heads + h, (w * s) * scale, mask=mask)


def _strided_tiling(tokens, heads):
    """Tile shape for the strided path.

    Both extents feed ``tl.arange``, which takes powers of two only, so neither
    can be the raw extent. ``_STRIDED_TILE // block_h`` stays a power of two
    because ``block_h`` is one, and the ``max`` keeps it >= 1 when a row alone
    exceeds the tile budget.
    """
    block_h = max(_STRIDED_MIN_BLOCK_H, triton.next_power_of_2(heads))
    block_t = max(
        1, min(triton.next_power_of_2(tokens), max(1, _STRIDED_TILE // block_h))
    )
    return block_t, block_h


def fused_indexer_weight_scale(
    weights: torch.Tensor, q_scale: torch.Tensor, scale: float
) -> torch.Tensor:
    """Scale per-token, per-head indexer weights by their query scales.

    Args:
        weights: ``[tokens, heads]``, fp32 / fp16 / bf16 (promoted to fp32).
        q_scale: ``[tokens, heads, 1]``, fp32 / fp16 / bf16. In production this
            is a ue8m0 block scale, i.e. exact powers of two.
        scale: scalar folding ``softmax_scale`` and ``n_head ** -0.5``.

    Returns:
        A new ``[tokens, heads]`` fp32 tensor. The inputs are not modified.

    Raises:
        NotImplementedError: for layouts or dtypes outside the contract.
    """
    if weights.ndim != 2:
        raise NotImplementedError(
            f"weights must be [tokens, heads], got {tuple(weights.shape)}"
        )
    if q_scale.ndim != 3 or q_scale.shape[-1] != 1:
        raise NotImplementedError(
            f"q_scale must be [tokens, heads, 1], got {tuple(q_scale.shape)}"
        )
    if q_scale.shape[:2] != weights.shape:
        raise NotImplementedError(
            f"shape mismatch: weights {tuple(weights.shape)} vs "
            f"q_scale {tuple(q_scale.shape)}"
        )
    if weights.dtype not in _SUPPORTED_DTYPES or q_scale.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"unsupported dtypes: weights {weights.dtype}, q_scale {q_scale.dtype}"
        )
    if q_scale.device != weights.device:
        raise NotImplementedError(
            f"inputs on different devices: {weights.device} and {q_scale.device}"
        )

    tokens, heads = weights.shape
    out = torch.empty((tokens, heads), dtype=torch.float32, device=weights.device)
    numel = tokens * heads
    if numel == 0:
        # An empty grid dimension is rejected by the launcher, so the degenerate
        # shapes -- (0, heads) and (tokens, 0) -- return before any launch.
        return out

    # The trailing extent is 1, so its stride cannot affect an address:
    # dropping it makes q_scale a [tokens, heads] view shaped like weights.
    scales = q_scale.squeeze(-1)
    if (
        weights.dtype == torch.float32
        and scales.dtype == torch.float32
        and weights.is_contiguous()
        and scales.is_contiguous()
    ):
        _flat_kernel[(triton.cdiv(numel, FLAT_BLOCK),)](
            weights,
            scales,
            out,
            scale,
            numel,
            BLOCK=FLAT_BLOCK,
            num_warps=FLAT_NUM_WARPS,
        )
        return out

    block_t, block_h = _strided_tiling(tokens, heads)
    _strided_kernel[(triton.cdiv(tokens, block_t),)](
        weights,
        scales,
        out,
        scale,
        tokens,
        heads,
        weights.stride(0),
        weights.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        num_warps=_STRIDED_NUM_WARPS,
    )
    return out

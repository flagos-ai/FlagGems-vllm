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
#
# This implementation is adapted from the flash-linear-attention recurrent KDA
# kernel, originally licensed under MIT.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Opt-in Triton recurrent KDA kernel for one-token serving decode.

The implementation is intentionally narrow.  It is a latency-oriented SM90
decode path for the fixed serving shape documented by ``_validate_inputs``;
callers must opt in explicitly and unsupported inputs are rejected instead of
silently changing the behavior of the generic recurrent operator.
"""

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.FLA.triton_ops_helper import exp


@triton.jit
def fused_recurrent_kda_decode_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    state,
    cu_seqlens,
    state_indices,
    scale: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_g_head: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_beta_head: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_o_head: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_value: tl.constexpr,
    stride_state_key: tl.constexpr,
    stride_cu_seqlens: tl.constexpr,
    stride_state_indices: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    GROUP_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    TOKEN_COUNT: tl.constexpr,
    PREFETCH_STATE: tl.constexpr,
):
    """Update the FP32 V-first state cache for one token per active sequence."""
    i_v_group, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV

    o_k = tl.arange(0, K)

    # vLLM supplies readable metadata for every graph sequence, including
    # padded sequences.  Defer every state access until after the sequence and
    # NULL_BLOCK_ID checks below.
    state_idx = tl.load(state_indices + i_n * stride_state_indices).to(tl.int64)

    if USE_CU_SEQLENS:
        bos = tl.load(cu_seqlens + i_n * stride_cu_seqlens).to(tl.int64)
        eos = tl.load(cu_seqlens + (i_n + 1) * stride_cu_seqlens).to(tl.int64)
        sequence_length = eos - bos

        # CUDA graphs may reserve trailing q/out rows.  Define those rows as
        # zero even when fewer tokens are packed in the replay.
        num_sequences = tl.num_programs(1) // HV
        packed_tokens = tl.load(cu_seqlens + num_sequences * stride_cu_seqlens)
        if i_n >= packed_tokens and i_n < TOKEN_COUNT:
            for i_tile in tl.range(0, GROUP_V, loop_unroll_factor=1):
                o_v = (i_v_group * GROUP_V + i_tile) * BV + tl.arange(0, BV)
                p_o = o + i_n * stride_o_token + i_hv * stride_o_head + o_v
                tl.store(p_o, 0.0, mask=o_v < V)

        if sequence_length == 0:
            return
        if sequence_length != 1:
            # This opt-in kernel is decode-only.  In particular, never mutate
            # state if device-side graph metadata unexpectedly describes a
            # prefill/speculative sequence.
            return
        i_token = bos
    else:
        i_token = i_n

    # vLLM 0.24 reserves slot zero as NULL_BLOCK_ID.  Negative indices are
    # padding too.  Neither may read or write the cache, and their output is
    # explicitly zeroed.
    if state_idx <= 0:
        for i_tile in tl.range(0, GROUP_V, loop_unroll_factor=1):
            o_v = (i_v_group * GROUP_V + i_tile) * BV + tl.arange(0, BV)
            p_o = o + i_token * stride_o_token + i_hv * stride_o_head + o_v
            tl.store(p_o, 0.0, mask=o_v < V)
        return

    if PREFETCH_STATE:
        first_v = i_v_group * GROUP_V * BV + tl.arange(0, BV)
        first_ptr = (
            state
            + state_idx * stride_state_token
            + i_hv * stride_state_head
            + first_v[:, None] * stride_state_value
            + o_k[None, :] * stride_state_key
        )
        pending_state = tl.load(
            first_ptr,
            mask=first_v[:, None] < V,
            other=0.0,
        ).to(tl.float32)
        pending_state_second = tl.load(
            first_ptr + BV * stride_state_value,
            mask=(GROUP_V > 1) & (first_v[:, None] + BV < V),
            other=0.0,
        ).to(tl.float32)

    p_q = q + i_token * stride_q_token + i_hv * stride_q_head + o_k
    p_k = k + i_token * stride_k_token + i_hv * stride_k_head + o_k
    b_q = tl.load(p_q, eviction_policy="evict_last").to(tl.float32)
    b_k = tl.load(p_k, eviction_policy="evict_last").to(tl.float32)
    b_q *= tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k *= tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    p_g = g + i_token * stride_g_token + i_hv * stride_g_head + o_k
    b_decay = exp(tl.load(p_g, eviction_policy="evict_last").to(tl.float32))
    p_beta = beta + i_token * stride_beta_token + i_hv * stride_beta_head
    b_beta = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)

    # A small serial V group reuses q/k/g and their norms while retaining
    # enough independent CTAs at the target decode batch sizes.
    for i_tile in tl.range(0, GROUP_V, loop_unroll_factor=1):
        o_v = (i_v_group * GROUP_V + i_tile) * BV + tl.arange(0, BV)
        mask_v = o_v < V
        p_state = (
            state
            + state_idx * stride_state_token
            + i_hv * stride_state_head
            + o_v[:, None] * stride_state_value
            + o_k[None, :] * stride_state_key
        )
        if PREFETCH_STATE:
            b_state = pending_state
            next_v = o_v + 2 * BV
            next_state = tl.load(
                p_state + 2 * BV * stride_state_value,
                mask=(i_tile + 2 < GROUP_V) & (next_v[:, None] < V),
                other=0.0,
            ).to(tl.float32)
            pending_state = pending_state_second
            pending_state_second = next_state
        else:
            b_state = tl.load(
                p_state,
                mask=mask_v[:, None],
                other=0.0,
            ).to(tl.float32)

        p_v = v + i_token * stride_v_token + i_hv * stride_v_head + o_v
        b_v = tl.load(
            p_v,
            mask=mask_v,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        b_state *= b_decay[None, :]
        b_v -= tl.sum(b_state * b_k[None, :], axis=1)
        b_v *= b_beta
        b_state += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_state * b_q[None, :], axis=1)

        p_o = o + i_token * stride_o_token + i_hv * stride_o_head + o_v
        tl.store(
            p_o,
            b_o.to(p_o.dtype.element_ty),
            mask=mask_v,
            eviction_policy="evict_first",
        )
        tl.store(
            p_state,
            b_state,
            mask=mask_v[:, None],
            eviction_policy="evict_first",
        )


def _require_enabled(enable_decode_optimization: bool) -> None:
    if enable_decode_optimization is not True:
        raise NotImplementedError(
            "The specialized recurrent KDA decode path is disabled. Pass "
            "`enable_decode_optimization=True` explicitly to opt in."
        )


def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Conservatively reject aliases using host-side storage metadata only."""
    return (
        lhs.device == rhs.device
        and lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()
    )


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    out: torch.Tensor | None,
    scale: float,
) -> int:
    """Validate the exact no-fallback serving contract and return graph batch."""
    if q.ndim != 4:
        raise ValueError("`q` must have shape [1, N, 4, 128].")
    B, token_capacity, H, K = q.shape
    if B != 1 or not 1 <= token_capacity <= 128 or H != 4 or K != 128:
        raise ValueError(
            "The optimized decode path requires q shape [1, N, 4, 128] "
            "with 1 <= N <= 128."
        )
    expected_qkv_shape = (1, token_capacity, 4, 128)
    if k.shape != expected_qkv_shape or v.shape != expected_qkv_shape:
        raise ValueError("`k` and `v` must match q shape [1, N, 4, 128].")
    if g.shape != expected_qkv_shape:
        raise ValueError("`g` must have shape [1, N, 4, 128].")
    if beta.shape != (1, token_capacity, 4):
        raise ValueError("`beta` must have shape [1, N, 4].")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("`q`, `k`, and `v` must all have bfloat16 dtype.")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("Preprocessed `g` and `beta` must have float32 dtype.")
    if (
        initial_state.ndim != 4
        or initial_state.shape[0] < 2
        or initial_state.shape[1:] != (4, 128, 128)
    ):
        raise ValueError(
            "`initial_state` must use V-first layout [num_slots, 4, 128, 128] "
            "and reserve slot zero."
        )
    if initial_state.dtype != torch.float32:
        raise ValueError("`initial_state` must have float32 dtype.")

    dense_tensors = (q, k, v, g, beta, initial_state)
    if any(not tensor.is_contiguous() for tensor in dense_tensors):
        raise ValueError("All dense inputs must be contiguous.")
    if ssm_state_indices.ndim != 1 or not ssm_state_indices.is_contiguous():
        raise ValueError("`ssm_state_indices` must be contiguous and one-dimensional.")
    if ssm_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("`ssm_state_indices` must have int32 or int64 dtype.")

    num_sequences = token_capacity
    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("`cu_seqlens` must have shape [num_sequences + 1].")
        if not cu_seqlens.is_contiguous():
            raise ValueError("`cu_seqlens` must be contiguous.")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("`cu_seqlens` must have int32 or int64 dtype.")
        num_sequences = cu_seqlens.numel() - 1
        if not token_capacity <= num_sequences <= 128:
            raise ValueError("Decode requires token_capacity <= num_sequences <= 128.")
    if ssm_state_indices.numel() != num_sequences:
        raise ValueError(
            "`ssm_state_indices` must contain one slot index per graph sequence."
        )

    tensors = dense_tensors + (ssm_state_indices,)
    if cu_seqlens is not None:
        tensors += (cu_seqlens,)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("All inputs must be on the same device.")
    if any(tensor.requires_grad for tensor in dense_tensors):
        raise ValueError("The optimized decode path is inference-only.")

    if out is not None:
        if out.shape != v.shape or out.dtype != v.dtype or out.device != v.device:
            raise ValueError("`out` must match `v` in shape, dtype, and device.")
        if not out.is_contiguous():
            raise ValueError("`out` must be contiguous.")
        if out.requires_grad:
            raise ValueError("The optimized decode path is inference-only.")
        if any(_shares_storage(out, tensor) for tensor in tensors):
            raise ValueError("`out` must not alias any optimized decode input.")

    state_readers = (q, k, v, g, beta, ssm_state_indices)
    if cu_seqlens is not None:
        state_readers += (cu_seqlens,)
    if any(_shares_storage(initial_state, tensor) for tensor in state_readers):
        raise ValueError("`initial_state` must not alias any other decode input.")

    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("`scale` must be a finite positive Python scalar.")

    if runtime.device.vendor_name.lower() != "nvidia" or not q.is_cuda:
        raise ValueError("The optimized decode path requires an NVIDIA CUDA tensor.")
    try:
        capability = runtime.torch_device_fn.get_device_capability(q.device)
    except (AttributeError, RuntimeError, TypeError) as error:
        raise ValueError("Unable to query the CUDA device capability.") from error
    if capability is None or len(capability) < 1 or capability[0] != 9:
        raise ValueError("The optimized decode path requires NVIDIA SM90.")

    # Values are intentionally not copied back to the host: the API remains
    # CUDA-graph-safe.  Active sequences must contain exactly one token and
    # use unique in-range state slots > 0.  The Triton kernel independently
    # guards empty/non-decode sequences and every slot <= 0 before state IO.
    return num_sequences


def fused_recurrent_kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scale: float | None = None,
    *,
    enable_decode_optimization: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the explicit one-token KDA decode optimization.

    ``g`` and ``beta`` are the already-activated FP32 values consumed by the
    vLLM 0.24 recurrent KDA ABI.  Active state indices are a caller contract:
    they must be unique, in range, and greater than zero.  Slot zero is the
    vLLM ``NULL_BLOCK_ID`` and produces zero output without touching state.
    """
    _require_enabled(enable_decode_optimization)
    if scale is None:
        scale = 128**-0.5
    num_sequences = _validate_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        ssm_state_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        out=out,
        scale=scale,
    )
    if out is None:
        out = torch.empty_like(v)

    # Fixed-shape exemption from autotuning: these three schedules are the
    # H20/SM90 latency winners measured by PR #6015 for H=V=K=128 decode.
    # Keeping them deterministic also avoids autotune work during graph capture.
    if num_sequences == 1:
        block_v, group_v, num_warps, num_stages = 16, 1, 2, 1
    elif num_sequences * 4 <= 256:
        block_v, group_v, num_warps, num_stages = 8, 1, 1, 1
    else:
        block_v, group_v, num_warps, num_stages = 4, 8, 1, 2

    grid = (triton.cdiv(128, block_v * group_v), num_sequences * 4)
    fused_recurrent_kda_decode_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=out,
        state=initial_state,
        cu_seqlens=cu_seqlens,
        state_indices=ssm_state_indices,
        scale=float(scale),
        stride_q_token=q.stride(1),
        stride_q_head=q.stride(2),
        stride_k_token=k.stride(1),
        stride_k_head=k.stride(2),
        stride_v_token=v.stride(1),
        stride_v_head=v.stride(2),
        stride_g_token=g.stride(1),
        stride_g_head=g.stride(2),
        stride_beta_token=beta.stride(1),
        stride_beta_head=beta.stride(2),
        stride_o_token=out.stride(1),
        stride_o_head=out.stride(2),
        stride_state_token=initial_state.stride(0),
        stride_state_head=initial_state.stride(1),
        stride_state_value=initial_state.stride(2),
        stride_state_key=initial_state.stride(3),
        stride_cu_seqlens=cu_seqlens.stride(0) if cu_seqlens is not None else 0,
        stride_state_indices=ssm_state_indices.stride(0),
        HV=4,
        K=128,
        V=128,
        BV=block_v,
        GROUP_V=group_v,
        USE_CU_SEQLENS=cu_seqlens is not None,
        TOKEN_COUNT=q.shape[1],
        PREFETCH_STATE=(block_v == 4 and group_v == 8 and num_sequences * 4 <= 384),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    enable_decode_optimization: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM-compatible opt-in entry for preprocessed recurrent KDA decode."""
    _require_enabled(enable_decode_optimization)
    if not inplace_final_state:
        raise ValueError("The optimized path requires `inplace_final_state=True`.")
    if num_accepted_tokens is not None:
        raise ValueError("Speculative decode is not supported by this path.")
    if not use_qk_l2norm_in_kernel:
        raise ValueError("The optimized path requires in-kernel q/k L2 normalization.")
    if ssm_state_indices is None:
        raise ValueError("`ssm_state_indices` is required for serving decode.")
    return fused_recurrent_kda_decode(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        enable_decode_optimization=True,
        cu_seqlens=cu_seqlens,
        out=out,
    )


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    *,
    enable_decode_optimization: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opt-in drop-in wrapper for the preprocessed serving KDA call."""
    _require_enabled(enable_decode_optimization)
    if beta is None:
        raise ValueError("`beta` is required by the optimized decode path.")
    if initial_state is None:
        raise ValueError("`initial_state` is required by the optimized decode path.")
    if scale is None:
        scale = 128**-0.5

    allowed_kwargs = {
        "num_accepted_tokens",
        "out",
        # These vLLM policy arguments describe preprocessing already applied
        # before this recurrent call; they do not select another implementation.
        "safe_gate",
        "lower_bound",
        "output_final_state",
    }
    unknown_kwargs = set(kwargs) - allowed_kwargs
    if unknown_kwargs:
        names = ", ".join(sorted(unknown_kwargs))
        raise TypeError(f"Unsupported optimized KDA keyword argument(s): {names}")

    return fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=kwargs.get("num_accepted_tokens"),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        enable_decode_optimization=True,
        out=kwargs.get("out"),
    )


__all__ = [
    "fused_recurrent_kda",
    "fused_recurrent_kda_decode",
    "fused_recurrent_kda_fwd",
]

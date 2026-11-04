# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Hygon KDA launcher using Triton's current tensor specialization directly."""

import torch
import triton

from flaggems_vllm.ops.FLA.fused_recurrent_kda import (
    _launch_config,
    _recurrent_kda_kernel,
    _validate_recurrent_kda,
)
from flaggems_vllm.runtime import torch_device_fn

# LibEntry repeats HIP tensor-specialization queries on every call. Measured
# dispatch overhead dominates small decode workloads on the Hygon BW stack.
# Reuse the identical JIT body through Triton's own specialization/cache; keep
# all input validation, device guards and YAML launch configuration unchanged.
# No online autotune: replay would mutate the live recurrent state.
_hygon_recurrent_kda_kernel = _recurrent_kda_kernel.jit_function


def fused_recurrent_kda(
    q,
    k,
    v,
    g,
    beta=None,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    **kwargs,
):
    """Return output and FP32 [pool,H,V,K] state, without calling native vLLM.

    A 1D slot map is single-token decode; a 2D map stores every token state.
    Each sequence starts at its map's first slot. Writable slots must be
    disjoint across sequences, and mutated state must not alias Q/K/V/g/beta.
    cu_seqlens must be monotone, start at zero and end at T; each sequence
    must fit the slot-table width. These device metadata are caller preconditions.
    Initial slots <= 0 or >= pool produce zero output and suppress writeback.
    Invalid destination slots suppress only that write. Out-of-place mode
    returns [T,H,V,K] token states and leaves initial_state unchanged. Without
    indices its initial state is token-major, matching vLLM's KDA convention.
    """
    scale = (
        q.shape[-1] ** -0.5
        if scale is None and q.ndim == 4 and q.shape[-1] > 0
        else scale
    )
    n = _validate_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        kwargs,
    )
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    final = (
        initial_state
        if inplace_final_state
        else torch.empty((t, h, dv, dk), dtype=torch.float32, device=q.device)
    )
    if b * t == 0:
        return out, final
    bv, warps, stages = _launch_config()
    ix, width = (0, 0), 1
    if ssm_state_indices is not None:
        ix = (ssm_state_indices.stride(0), 0)
        if ssm_state_indices.ndim == 2:
            ix, width = ssm_state_indices.stride(), ssm_state_indices.shape[1]
    bs = beta.stride() if beta.ndim == 4 else (*beta.stride(), 0)
    with torch_device_fn.device(q.device):
        _hygon_recurrent_kda_kernel[(n * h * triton.cdiv(dv, bv),)](
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            out,
            final,
            cu_seqlens,
            ssm_state_indices,
            scale,
            (
                t,
                h,
                dk,
                dv,
                initial_state.shape[0],
                q.stride(),
                k.stride(),
                v.stride(),
                g.stride(),
                bs,
                initial_state.stride(),
                ix,
                width,
                cu_seqlens is not None,
                ssm_state_indices is not None,
                ssm_state_indices is not None and ssm_state_indices.ndim == 2,
                inplace_final_state,
                beta.ndim == 4,
                use_qk_l2norm_in_kernel,
                triton.next_power_of_2(dk),
                bv,
            ),
            num_warps=warps,
            num_stages=stages,
        )
    return out, final


__all__ = ["fused_recurrent_kda"]

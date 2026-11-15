# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Recurrent formulation from flash-linear-attention (MIT):
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""KDA decode with key-wise decay and value-major recurrent state."""

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@lru_cache(maxsize=1)
def _launch_config():
    # No online autotune: replay would update live state repeatedly. The YAML
    # launch policy is selected offline with the entire pool reset per sample.
    config = runtime.get_tuned_config("fused_recurrent_kda")[0]
    return config.kwargs["BV"], config.num_warps, config.num_stages


@libentry()
@triton.jit
def _recurrent_kda_kernel(
    Q,
    Kp,
    Vp,
    G,
    Beta,
    State,
    Out,
    Final,
    Cu,
    Indices,
    scale,
    META: tl.constexpr,
):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    V: tl.constexpr = META[3]
    POOL: tl.constexpr = META[4]
    QS: tl.constexpr = META[5]
    KS: tl.constexpr = META[6]
    VS: tl.constexpr = META[7]
    GS: tl.constexpr = META[8]
    BS: tl.constexpr = META[9]
    SS: tl.constexpr = META[10]
    IX: tl.constexpr = META[11]
    WIDTH: tl.constexpr = META[12]
    VARLEN: tl.constexpr = META[13]
    INDEXED: tl.constexpr = META[14]
    TABLE: tl.constexpr = META[15]
    INPLACE: tl.constexpr = META[16]
    VECTOR_BETA: tl.constexpr = META[17]
    NORMALIZE: tl.constexpr = META[18]
    BK: tl.constexpr = META[19]
    BV: tl.constexpr = META[20]
    pid = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    iv = pid % nv
    nh = pid // nv
    n, h = nh // H, nh % H
    if VARLEN:
        start = tl.load(Cu + n).to(tl.int64)
        end = tl.load(Cu + n + 1).to(tl.int64)
        batch = 0
    else:
        start, end, batch = 0, T, n
    # Unindexed native KDA state is token-major: the first token selects h0.
    slot = (batch * T + start).to(tl.int64)
    valid = slot < POOL
    if INDEXED:
        slot = tl.load(Indices + n * IX[0]).to(tl.int64)
        valid = (slot > 0) & (slot < POOL)
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    linear = tl.arange(0, BV * BK)
    flat_v, flat_k = iv * BV + linear // BK, linear % BK
    state_offsets = h * SS[1] + flat_v * SS[2] + flat_k * SS[3]
    state_mask = (flat_v < V) & (flat_k < K)
    acc = (
        tl.load(State + slot * SS[0] + state_offsets, state_mask & valid, 0)
        .to(tl.float32)
        .reshape((BV, BK))
    )
    for t in range(start, end):
        q = tl.load(
            Q + batch * QS[0] + t * QS[1] + h * QS[2] + rk * QS[3], rk < K, 0
        ).to(tl.float32)
        k = tl.load(
            Kp + batch * KS[0] + t * KS[1] + h * KS[2] + rk * KS[3], rk < K, 0
        ).to(tl.float32)
        if NORMALIZE:
            q /= tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)
            k /= tl.sqrt(tl.sum(k * k, 0) + 1.0e-6)
        q *= scale
        gate = tl.load(
            G + batch * GS[0] + t * GS[1] + h * GS[2] + rk * GS[3], rk < K, 0
        ).to(tl.float32)
        acc *= tl.exp(gate)[None, :]
        value = tl.load(
            Vp + batch * VS[0] + t * VS[1] + h * VS[2] + rv * VS[3], rv < V, 0
        ).to(tl.float32)
        bp = Beta + batch * BS[0] + t * BS[1] + h * BS[2]
        if VECTOR_BETA:
            beta = tl.load(bp + rv * BS[3], rv < V, 0).to(tl.float32)
        else:
            beta = tl.load(bp).to(tl.float32)
        delta = (value - tl.sum(acc * k[None, :], 1)) * beta
        acc += delta[:, None] * k[None, :]
        output = tl.sum(acc * q[None, :], 1)
        token = batch * T + t
        tl.store(Out + (token * H + h) * V + rv, tl.where(valid, output, 0), rv < V)
        if INPLACE:
            dest, store = slot, valid
            if TABLE:
                column = t - start
                dest = tl.load(
                    Indices + n * IX[0] + column * IX[1], column < WIDTH, 0
                ).to(tl.int64)
                store = store & (column < WIDTH) & (dest > 0) & (dest < POOL)
            tl.store(
                Final + dest * SS[0] + state_offsets,
                acc.reshape((BV * BK,)),
                state_mask & store,
            )
        else:
            offsets = ((token * H + h) * V + flat_v) * K + flat_k
            tl.store(
                Final + offsets, tl.where(valid, acc.reshape((BV * BK,)), 0), state_mask
            )


@lru_cache(maxsize=128)
def _validate_metadata(metadata, state_strides, cu_contiguous, scale, inplace):
    # Keys contain immutable shape/dtype/device/grad metadata, never tensors,
    # state values, data pointers or device-side sequence/index contents.
    q, k, v, g, beta, state, cu, indices = metadata
    if len(q[0]) != 4 or k[0] != q[0] or g[0] != q[0]:
        raise ValueError("q/k/g must share [B,T,H,K] shape")
    b, t, h, dk = q[0]
    if len(v[0]) != 4 or v[0][:3] != (b, t, h):
        raise ValueError("v must have shape [B,T,H,V]")
    dv = v[0][-1]
    if h < 1 or not 1 <= dk <= 256 or not 1 <= dv <= 256:
        raise NotImplementedError("KDA supports positive heads and K,V <= 256")
    floating = (torch.float16, torch.bfloat16, torch.float32)
    if q[1] not in floating or k[1] != q[1] or v[1] != q[1]:
        raise ValueError("q/k/v must share FP16, BF16 or FP32 dtype")
    if g[1] not in floating or beta[1] not in floating:
        raise ValueError("g/beta must be floating tensors")
    if beta[0] not in ((b, t, h), (b, t, h, dv)):
        raise ValueError("beta must have shape [B,T,H] or [B,T,H,V]")
    if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    n = b
    if cu is not None:
        if (
            b != 1
            or len(cu[0]) != 1
            or cu[0][0] < 1
            or cu[1] not in (torch.int32, torch.int64)
            or not cu_contiguous
        ):
            raise ValueError(
                "Varlen requires B=1 and contiguous integer cu_seqlens[N+1]"
            )
        n = cu[0][0] - 1
    if state is None:
        raise ValueError("KDA decode requires an initial-state buffer")
    if len(state[0]) != 4 or state[0][1:] != (h, dv, dk) or state[1] != torch.float32:
        raise ValueError("initial_state must be FP32 [pool,H,V,K]")
    extent = 1
    for stride, size in sorted(zip(state_strides, state[0])):
        if stride <= 0 or (size > 1 and stride < extent):
            raise ValueError("State must have positive non-overlapping strides")
        extent = max(extent, stride * size)
    if indices is None:
        if inplace and b * t:
            raise ValueError("In-place KDA requires an explicit state-slot map")
        if state[0][0] < b * t:
            raise ValueError("Unindexed KDA requires token-major initial state")
    else:
        if (
            indices[1] not in (torch.int32, torch.int64)
            or len(indices[0]) not in (1, 2)
            or indices[0][0] != n
        ):
            raise ValueError("State indices must be integer [N] or [N,width]")
        if len(indices[0]) == 1:
            if b * t and ((cu is None and t > 1) or (cu is not None and t != n)):
                raise ValueError("Multi-token KDA requires a 2D state-slot table")
        elif indices[0][1] < max(1, triton.cdiv(t, max(n, 1))):
            raise ValueError("State-slot table is too narrow")
    if not inplace and b != 1:
        raise NotImplementedError("Non-inplace KDA requires flattened batch one")
    if any(x is not None and x[2] != q[2] for x in metadata):
        raise ValueError("All KDA tensors must share the input device")
    if any(x is not None and x[3] for x in metadata):
        raise NotImplementedError("Recurrent KDA is inference-only")
    if b * t and n == 0:
        raise ValueError("Nonempty input requires at least one sequence")
    return n


def _validate_recurrent_kda(
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
):
    if kwargs:
        raise NotImplementedError(f"Unsupported KDA decode arguments: {tuple(kwargs)}")
    if beta is None:
        raise ValueError("KDA beta tensor is required")
    metadata = tuple(
        (x.shape, x.dtype, x.device, x.requires_grad) if x is not None else None
        for x in (q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices)
    )
    return _validate_metadata(
        metadata,
        initial_state.stride() if initial_state is not None else (),
        cu_seqlens.is_contiguous() if cu_seqlens is not None else True,
        scale,
        inplace_final_state,
    )


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
        _recurrent_kda_kernel[(n * h * triton.cdiv(dv, bv),)](
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

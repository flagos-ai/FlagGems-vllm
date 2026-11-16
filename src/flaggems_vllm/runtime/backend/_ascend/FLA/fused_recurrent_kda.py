# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Recurrent formulation from flash-linear-attention (MIT):
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""Ascend KDA: affine aligned tiles and conservative irregular transfers."""

import logging
from contextlib import nullcontext
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.backends.ascend.driver import NPUUtils

from flaggems_vllm.ops.FLA.fused_recurrent_kda import (
    _launch_config,
    _validate_recurrent_kda,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _vector_core_count(device_index):
    return NPUUtils().get_aivector_core_num()


@triton.jit
def _recurrent_kda_body(
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
    POOL,
    BLOCKS,
    EXTRA,
    NORMALIZE_INPUT,
    META: tl.constexpr,
):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    V: tl.constexpr = META[3]
    QS: tl.constexpr = META[4]
    KS: tl.constexpr = META[5]
    VS: tl.constexpr = META[6]
    GS: tl.constexpr = META[7]
    BS: tl.constexpr = META[8]
    SS: tl.constexpr = META[9]
    IX: tl.constexpr = META[10]
    WIDTH: tl.constexpr = META[11]
    VARLEN: tl.constexpr = META[12]
    INDEXED: tl.constexpr = META[13]
    TABLE: tl.constexpr = META[14]
    INPLACE: tl.constexpr = META[15]
    VECTOR_BETA: tl.constexpr = META[16]
    NORMALIZE: tl.constexpr = META[17]
    do_normalize = NORMALIZE_INPUT != 0 if NORMALIZE is None else NORMALIZE
    BK: tl.constexpr = META[18]
    BV: tl.constexpr = META[19]
    FAST: tl.constexpr = META[20]
    core = tl.program_id(0)
    first = core * BLOCKS + tl.minimum(core, EXTRA)
    last = first + BLOCKS + (core.to(tl.float32) < EXTRA).to(tl.int32)
    for pid in range(first, last):
        nv = tl.cdiv(V, BV)
        iv = pid - (pid // nv) * nv
        nh = pid // nv
        n, h = nh // H, nh - (nh // H) * H
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
        flat_v, flat_k = iv * BV + linear // BK, linear - (linear // BK) * BK
        if FAST:
            # Expose a unit-stride 1D transfer directly to the NPU compiler.
            # The decomposed V/K quotient-remainder expression may obscure DMA.
            state_offsets = h * SS[1] + iv * BV * K + linear
            if V % BV == 0:
                state_mask = tl.full((BV * BK,), True, tl.int1)
            else:
                state_mask = linear.to(tl.float32) < (V - iv * BV) * K
        else:
            state_offsets = h * SS[1] + flat_v * SS[2] + flat_k * SS[3]
            state_mask = (flat_v < V) & (flat_k < K)
        acc = (
            tl.load(State + slot * SS[0] + state_offsets, state_mask & valid, 0)
            .to(tl.float32)
            .reshape((BV, BK))
        )
        if FAST:
            key_mask = rk.to(tl.float32) < K
            if V % BV == 0:
                value_mask = tl.full((BV,), True, tl.int1)
            else:
                value_mask = rv.to(tl.float32) < V
        else:
            key_mask = rk < K
            value_mask = rv < V
        for t in range(start, end):
            q = tl.load(
                Q + batch * QS[0] + t * QS[1] + h * QS[2] + rk * QS[3], key_mask, 0
            ).to(tl.float32)
            k = tl.load(
                Kp + batch * KS[0] + t * KS[1] + h * KS[2] + rk * KS[3], key_mask, 0
            ).to(tl.float32)
            if do_normalize:
                q /= tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)
                k /= tl.sqrt(tl.sum(k * k, 0) + 1.0e-6)
            q *= scale
            gate = tl.load(
                G + batch * GS[0] + t * GS[1] + h * GS[2] + rk * GS[3], key_mask, 0
            ).to(tl.float32)
            acc *= tl.exp(gate)[None, :]
            value = tl.load(
                Vp + batch * VS[0] + t * VS[1] + h * VS[2] + rv * VS[3], value_mask, 0
            ).to(tl.float32)
            bp = Beta + batch * BS[0] + t * BS[1] + h * BS[2]
            if VECTOR_BETA:
                beta = tl.load(bp + rv * BS[3], value_mask, 0).to(tl.float32)
            else:
                beta = tl.load(bp).to(tl.float32)
            delta = (value - tl.sum(acc * k[None, :], 1)) * beta
            acc += delta[:, None] * k[None, :]
            output = tl.sum(acc * q[None, :], 1)
            token = batch * T + t
            tl.store(
                Out + (token * H + h) * V + rv, tl.where(valid, output, 0), value_mask
            )
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
                if FAST:
                    offsets = ((token * H + h) * V + iv * BV) * K + linear
                else:
                    offsets = ((token * H + h) * V + flat_v) * K + flat_k
                tl.store(
                    Final + offsets,
                    tl.where(valid, acc.reshape((BV * BK,)), 0),
                    state_mask,
                )


@libentry()
@triton.jit(do_not_specialize=["POOL", "BLOCKS", "EXTRA", "NORMALIZE_INPUT"])
def _ascend_recurrent_kda_kernel(
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
    POOL,
    BLOCKS,
    EXTRA,
    NORMALIZE_INPUT,
    META: tl.constexpr,
):
    _recurrent_kda_body(
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
        POOL,
        BLOCKS,
        EXTRA,
        NORMALIZE_INPUT,
        META,
    )


@libentry()
@triton.jit
def _ascend_recurrent_kda_specialized(
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
    pool: tl.constexpr = META[21]
    blocks: tl.constexpr = META[22]
    extra: tl.constexpr = META[23]
    _recurrent_kda_body(
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
        pool,
        blocks,
        extra,
        1,
        META,
    )


def _input_strides(tensor, varlen):
    strides = tensor.stride()
    # Packed inputs are always addressed in batch zero. Omit the stride that
    # otherwise changes with the total number of packed tokens.
    return (0, *strides[1:]) if varlen else strides


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
    logger.debug("GEMS_ASCEND FUSED_RECURRENT_KDA")
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
    # Keep the verified small-tile/scalar-address lowering for irregular views.
    # Larger affine tiles were numerically unstable on these layouts.
    ss = initial_state.stride()
    fast = (
        dk == triton.next_power_of_2(dk)
        and ss[2:] == (dk, 1)
        and initial_state.data_ptr() % 64 == 0
        and ss[0] % 16 == 0
        and ss[1] % 16 == 0
        and dv * v.element_size() % 64 == 0
    )
    bv = min(bv if fast else 8, triton.next_power_of_2(dv))
    ix, width = (0, 0), 1
    if ssm_state_indices is not None:
        ix = (ssm_state_indices.stride(0), 0)
        if ssm_state_indices.ndim == 2:
            ix, width = ssm_state_indices.stride(), ssm_state_indices.shape[1]
    varlen = cu_seqlens is not None
    normalize = bool(use_qk_l2norm_in_kernel)
    beta_strides = _input_strides(beta, varlen)
    if beta.ndim == 3:
        beta_strides = (*beta_strides, 0)
    meta = (
        0 if varlen else t,
        h,
        dk,
        dv,
        _input_strides(q, varlen),
        _input_strides(k, varlen),
        _input_strides(v, varlen),
        _input_strides(g, varlen),
        beta_strides,
        ss,
        ix,
        width,
        varlen,
        ssm_state_indices is not None,
        ssm_state_indices is not None and ssm_state_indices.ndim == 2,
        inplace_final_state,
        beta.ndim == 4,
        normalize if fast else None,
        triton.next_power_of_2(dk),
        bv,
        fast,
    )
    args = (
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
    )
    pool = initial_state.shape[0]
    device = q.device
    device_context = (
        nullcontext()
        if torch_device_fn.current_device() == device.index
        else torch_device_fn.device(device)
    )
    with device_context:
        total = n * h * triton.cdiv(dv, bv)
        cores = min(total, _vector_core_count(device.index))
        blocks, extra = divmod(total, cores)
        grid = (cores,)
        if fast and normalize:
            _ascend_recurrent_kda_specialized[grid](
                *args,
                (*meta, pool, blocks, extra),
                num_warps=warps,
                num_stages=stages,
                enable_fp_fusion=False,
            )
        else:
            _ascend_recurrent_kda_kernel[grid](
                *args,
                pool,
                blocks,
                extra,
                int(normalize),
                meta,
                num_warps=warps,
                num_stages=stages,
                enable_fp_fusion=False,
            )
    return out, final


__all__ = ["fused_recurrent_kda"]

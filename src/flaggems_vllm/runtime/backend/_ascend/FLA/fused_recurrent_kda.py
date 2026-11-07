# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Recurrent formulation from flash-linear-attention (MIT):
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""Ascend KDA: affine aligned tiles and conservative irregular transfers."""

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


@lru_cache(maxsize=None)
def _vector_core_count(device_index):
    return NPUUtils().get_aivector_core_num()


@libentry()
@triton.jit
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
    A_log,
    Dt_bias,
    Accepted,
    scale,
    lower_bound,
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
    TOTAL: tl.constexpr = META[21]
    FAST: tl.constexpr = META[22]
    # Kimi K3 passes the raw gate projection and expects the bounded gate plus
    # beta's sigmoid to be fused, matching AscendC recurrent_kda's
    # use_gate_in_kernel / use_beta_sigmoid_in_kernel / safe_gate flags.
    SAFE_GATE: tl.constexpr = META[23]
    BETA_SIGMOID: tl.constexpr = META[24]
    SPEC: tl.constexpr = META[25]
    DENSE: tl.constexpr = META[26]
    programs = tl.num_programs(0)
    core = tl.program_id(0)
    blocks = TOTAL // programs
    extra = TOTAL - blocks * programs
    first = core * blocks + tl.minimum(core, extra)
    last = first + blocks + (core.to(tl.float32) < extra).to(tl.int32)
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
            # Speculative decode seeds from the last accepted token's column
            # rather than column zero.
            column = 0
            if SPEC:
                column = tl.load(Accepted + n).to(tl.int64) - 1
            inrange = (column >= 0) & (column < WIDTH)
            slot = tl.load(Indices + n * IX[0] + column * IX[1], inrange, 0).to(
                tl.int64
            )
            valid = inrange & (slot > 0) & (slot < POOL)
        rk = tl.arange(0, BK)
        rv = iv * BV + tl.arange(0, BV)
        if DENSE:
            # A dense [V, K] page addresses as a 2D tile, so the state never
            # needs a BV*BK index vector or a reshape. That keeps live UB low
            # enough for one v-tile per (sequence, head); the flat forms below
            # cap BV at 32, which splits V and multiplies per-core iterations.
            state_offsets = h * SS[1] + rv[:, None] * SS[2] + rk[None, :]
            state_mask = (rv[:, None] < V) & (rk[None, :] < K)
            acc = tl.load(
                State + slot * SS[0] + state_offsets, state_mask & valid, 0
            ).to(tl.float32)
        else:
            linear = tl.arange(0, BV * BK)
            flat_v, flat_k = iv * BV + linear // BK, linear - (linear // BK) * BK
            if FAST:
                # Expose a unit-stride 1D transfer directly to the NPU compiler.
                # The decomposed V/K quotient-remainder expression may obscure DMA.
                state_offsets = h * SS[1] + iv * BV * K + linear
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
            value_mask = rv.to(tl.float32) < V
        else:
            key_mask = rk < K
            value_mask = rv < V
        if SAFE_GATE:
            # Hoisted out of the token loop: both are indexed only by
            # (head, key), so re-loading them per token adds a scalar fetch and
            # a [BK] transfer to every iteration of a serial recurrence.
            gate_decay = tl.exp(tl.load(A_log + h).to(tl.float32))
            gate_bias = tl.load(Dt_bias + h * K + rk, key_mask, 0.0).to(tl.float32)
        for t in range(start, end):
            q = tl.load(
                Q + batch * QS[0] + t * QS[1] + h * QS[2] + rk * QS[3], key_mask, 0
            ).to(tl.float32)
            k = tl.load(
                Kp + batch * KS[0] + t * KS[1] + h * KS[2] + rk * KS[3], key_mask, 0
            ).to(tl.float32)
            if NORMALIZE:
                q /= tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)
                k /= tl.sqrt(tl.sum(k * k, 0) + 1.0e-6)
            q *= scale
            gate = tl.load(
                G + batch * GS[0] + t * GS[1] + h * GS[2] + rk * GS[3], key_mask, 0
            ).to(tl.float32)
            if SAFE_GATE:
                # gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)),
                # with exp(A_log) and dt_bias hoisted above the loop.
                gate = lower_bound * tl.sigmoid(gate_decay * (gate + gate_bias))
            decay = tl.exp(gate)
            value = tl.load(
                Vp + batch * VS[0] + t * VS[1] + h * VS[2] + rv * VS[3], value_mask, 0
            ).to(tl.float32)
            bp = Beta + batch * BS[0] + t * BS[1] + h * BS[2]
            if VECTOR_BETA:
                beta = tl.load(bp + rv * BS[3], value_mask, 0).to(tl.float32)
            else:
                beta = tl.load(bp).to(tl.float32)
            if BETA_SIGMOID:
                beta = tl.sigmoid(beta)
            # The decayed state only ever appears inside the delta reduction
            # and the rank-1 update, so fold `decay` into both rather than
            # applying it as its own `acc *= decay[None, :]` pass. Same
            # arithmetic (decay is applied to K lanes either way), but the
            # [V, K] fp32 tile -- 64KB at K=V=128 -- is traversed twice per
            # token instead of three times. Measured 121.7 us -> 94.7 us on the
            # bs=16 k=7 speculative case.
            kd = k * decay
            delta = (value - tl.sum(acc * kd[None, :], 1)) * beta
            acc = acc * decay[None, :] + delta[:, None] * k[None, :]
            output = tl.sum(acc * q[None, :], 1)
            token = batch * T + t
            tl.store(
                Out + (token * H + h) * V + rv, tl.where(valid, output, 0), value_mask
            )
            if INPLACE:
                dest, store = slot, valid
                if TABLE:
                    # Name this separately from the int64 `column` computed for
                    # the seed slot above: reusing that name re-assigns an
                    # int64 loop-carried variable from an int32 expression,
                    # which the Ascend backend rejects outright
                    # ("Loop-carried variable column has initial type int32 but
                    # is re-assigned to int64 in loop").
                    token_column = (t - start).to(tl.int64)
                    dest = tl.load(
                        Indices + n * IX[0] + token_column * IX[1],
                        token_column < WIDTH,
                        0,
                    ).to(tl.int64)
                    store = store & (token_column < WIDTH) & (dest > 0) & (dest < POOL)
                if DENSE:
                    tl.store(
                        Final + dest * SS[0] + state_offsets,
                        acc,
                        state_mask & store,
                    )
                else:
                    tl.store(
                        Final + dest * SS[0] + state_offsets,
                        acc.reshape((BV * BK,)),
                        state_mask & store,
                    )
            else:
                if DENSE:
                    offsets = ((token * H + h) * V + rv[:, None]) * K + rk[None, :]
                    tl.store(Final + offsets, tl.where(valid, acc, 0), state_mask)
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
    A_log=None,
    dt_bias=None,
    safe_gate=False,
    lower_bound=-5.0,
    use_beta_sigmoid_in_kernel=False,
    num_accepted_tokens=None,
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

    Kimi K3 additionally needs the activations AscendC ``recurrent_kda`` fuses,
    so ``g``/``beta`` may be passed as raw projections:

    * ``safe_gate`` with ``A_log``/``dt_bias`` applies the bounded gate
      ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` in-kernel;
    * ``use_beta_sigmoid_in_kernel`` folds in beta's sigmoid;
    * ``num_accepted_tokens`` selects each sequence's seed column from a 2D
      slot table, as speculative decode requires.

    Omitting all four keeps the pre-activated-gate contract unchanged.
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
    # A dense [V, K] page lets the kernel address state as a 2D tile, which
    # holds little enough live UB to take one v-tile per (sequence, head).
    # The recurrence is serial over tokens, so splitting V adds no parallelism
    # and only multiplies each core's grid-stride iterations.
    dense = ss[2:] == (dk, 1) and dv == triton.next_power_of_2(dv)
    bv = min(dv if dense else (bv if fast else 8), triton.next_power_of_2(dv))
    ix, width = (0, 0), 1
    if ssm_state_indices is not None:
        ix = (ssm_state_indices.stride(0), 0)
        if ssm_state_indices.ndim == 2:
            ix, width = ssm_state_indices.stride(), ssm_state_indices.shape[1]
    bs = beta.stride() if beta.ndim == 4 else (*beta.stride(), 0)
    if safe_gate and (A_log is None or dt_bias is None):
        raise ValueError("The in-kernel safe gate needs both A_log and dt_bias")
    if safe_gate and (A_log.numel() != h or dt_bias.numel() != h * dk):
        raise ValueError("A_log must hold H entries and dt_bias H*K entries")
    if num_accepted_tokens is not None:
        if ssm_state_indices is None or ssm_state_indices.ndim != 2:
            raise ValueError("Accepted-token counts require a 2D state table")
        if num_accepted_tokens.shape != (n,):
            raise ValueError("Accepted-token counts must be a [N] vector")
    with torch_device_fn.device(q.device):
        total = n * h * triton.cdiv(dv, bv)
        grid = (min(total, _vector_core_count(q.device.index)),)
        _ascend_recurrent_kda_kernel[grid](
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
            A_log,
            dt_bias,
            num_accepted_tokens,
            scale,
            lower_bound,
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
                total,
                fast,
                safe_gate,
                use_beta_sigmoid_in_kernel,
                num_accepted_tokens is not None,
                dense,
            ),
            num_warps=warps,
            num_stages=stages,
            enable_fp_fusion=False,
        )
    return out, final


__all__ = ["fused_recurrent_kda"]

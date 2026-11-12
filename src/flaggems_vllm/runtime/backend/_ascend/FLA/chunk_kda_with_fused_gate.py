# SPDX-License-Identifier: Apache-2.0
"""Ascend KDA prefill with up to eight statically unrolled state chunks per launch.

CANN 9/FlagTree produced incorrect values with a dynamic loop-carried Cube
state tile. Static unrolling removes that loop backedge; kernel boundaries
separate groups. Distinct input/output state buffers also keep stage tuning
from consuming its own previous output.
"""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.data_movement import contiguous_copy, copy_to, fill
from flaggems_vllm.ops.FLA.chunk_kda_with_fused_gate import (
    _CHUNK,
    _RCP_LN2,
    _SUBCHUNK,
    _chunk_layout_args,
    _ckfg_gate,
    _ckfg_inter,
    _ckfg_intra,
    _ckfg_inverse,
    _ckfg_output,
    _ckfg_recompute,
    _gate_bias_args,
    _metadata,
    _norm_rows,
    _stage_bench,
    _validate,
)
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _ckfg_zero_matrices_ascend(Akk, Aqk, Ai, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    tl.store(Akk + i, 0, mask)
    tl.store(Aqk + i, 0, mask)
    tl.store(Ai + i, 0, mask)


@libentry()
@triton.jit(do_not_specialize=["N", "K"])
def _ckfg_norm_ascend(X, Y, N, K, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    _norm_rows(X, Y, N, K, BLOCK, ROWS)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ckfg_state_ascend"),
    key=["T", "H", "K", "V", "BT"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit(do_not_specialize=["chunk", "T", "K", "V", "INITIAL", "FINAL"])
def _ckfg_state_ascend(
    k,
    v,
    w,
    v_new,
    gk,
    h,
    h0,
    ht,
    S,
    cu_seqlens,
    chunk_offsets,
    chunk,
    T,
    H: tl.constexpr,
    K,
    V,
    INITIAL,
    FINAL,
    BT: tl.constexpr,
    CHUNKS: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    VARLEN,
):
    iv, nh = tl.program_id(0), tl.program_id(1)
    seq, head = nh // H, nh % H
    if VARLEN:
        bos = tl.load(cu_seqlens + seq).to(tl.int32)
        end = tl.load(cu_seqlens + seq + 1).to(tl.int32)
        length = end - bos
        first = tl.load(chunk_offsets + seq).to(tl.int32)
    else:
        bos = seq * T
        length = T
        first = seq * tl.cdiv(T, BT)
    if (chunk * BT < length) | (chunk == 0):
        rv = iv * BV + tl.arange(0, BV)
        rk = tl.arange(0, BK)
        rt = tl.arange(0, BT)
        mask_h = (rv[:, None] < V) & (rk[None, :] < K)
        state = tl.full((BV, BK), 0, tl.float32)
        if INITIAL:
            state = tl.load(
                h0 + nh * V * K + rv[:, None] * K + rk[None, :], mask_h, other=0
            ).to(tl.float32)
        for step in tl.static_range(CHUNKS):
            index = chunk + step
            if index * BT < length:
                tl.store(
                    h
                    + ((first + index) * H + head) * V * K
                    + rv[:, None] * K
                    + rk[None, :],
                    state,
                    mask_h,
                )
                pos = index * BT + rt
                wk = tl.load(
                    w + ((bos + pos[:, None]) * H + head) * K + rk[None, :],
                    (pos[:, None] < length) & (rk[None, :] < K),
                    other=0,
                )
                vu = tl.load(
                    v + ((bos + pos[:, None]) * H + head) * V + rv[None, :],
                    (pos[:, None] < length) & (rv[None, :] < V),
                    other=0,
                ).to(tl.float32)
                residual = vu
                if INITIAL or step > 0:
                    residual = vu - tl.dot(
                        wk, tl.trans(state).to(wk.dtype), input_precision="ieee"
                    )
                residual = tl.where(
                    (pos[:, None] < length) & (rv[None, :] < V), residual, 0
                ).to(v.dtype.element_ty)
                tl.store(
                    v_new + ((bos + pos[:, None]) * H + head) * V + rv[None, :],
                    residual,
                    (pos[:, None] < length) & (rv[None, :] < V),
                )
                last = tl.minimum((index + 1) * BT, length) - 1
                decay = tl.load(
                    gk + ((bos + last) * H + head) * K + rk, rk < K, other=0
                )
                kg = tl.load(
                    k + ((bos + pos[:, None]) * H + head) * K + rk[None, :],
                    (pos[:, None] < length) & (rk[None, :] < K),
                    other=0,
                )
                state = state * tl.exp2(decay)[None, :] + tl.trans(
                    tl.dot(tl.trans(kg), residual, input_precision="ieee")
                )
        tl.store(S + nh * V * K + rv[:, None] * K + rk[None, :], state, mask_h)
        if FINAL:
            if (chunk + CHUNKS) * BT >= length:
                tl.store(ht + nh * V * K + rv[:, None] * K + rk[None, :], state, mask_h)


def chunk_kda_with_fused_gate(
    q,
    k,
    v,
    raw_g,
    beta,
    A_log,
    g_bias,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    **kwargs,
):
    """Return output and optional [N,H,V,K] FP32 final state.

    Beta is already activated. Packed cu_seqlens must be monotone, start at 0
    and end at T. Like vLLM 0.24.0, extra kwargs do not select a safe gate.
    """
    logger.debug("GEMS_ASCEND CHUNK_KDA_WITH_FUSED_GATE")
    n = _validate(q, k, v, raw_g, beta, A_log, g_bias, initial_state, cu_seqlens, scale)
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    scale = dk ** (-0.5) if scale is None else scale
    q, k, v, raw_g, beta = map(contiguous_copy, (q, k, v, raw_g, beta))
    a = contiguous_copy(A_log).view(-1)
    bias = contiguous_copy(g_bias).view(-1) if g_bias is not None else None
    state = contiguous_copy(initial_state) if initial_state is not None else None
    final = (
        torch.empty((n, h, dv, dk), dtype=torch.float32, device=q.device)
        if output_final_state
        else None
    )
    if t == 0 or b == 0:
        if final is not None:
            if state is None:
                fill(final, 0)
            else:
                copy_to(contiguous_copy(state, dtype=torch.float32), final)
        return (v, final)
    if use_qk_l2norm_in_kernel:
        qn, kn = (torch.empty_like(q), torch.empty_like(k))
        rows = 32
        for x, y in ((q, qn), (k, kn)):
            _ckfg_norm_ascend[(triton.cdiv(b * t * h, rows),)](
                x,
                y,
                b * t * h,
                dk,
                triton.next_power_of_2(dk),
                rows,
                num_warps=4,
                num_stages=2,
            )
        q, k = (qn, kn)
    cu = contiguous_copy(cu_seqlens) if cu_seqlens is not None else None
    if cu is None:
        indices, offsets = (None, None)
        nt = triton.cdiv(t, _CHUNK)
    else:
        indices, offsets = _metadata(cu, t, q.device)
        nt = indices.shape[0]
    packing = _chunk_layout_args(cu, indices, q.device)
    gate = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    _ckfg_gate[lambda meta: (triton.cdiv(dk, meta["BD"]), nt, b * h)](
        g=raw_g,
        A=a,
        y=gate,
        **_gate_bias_args(bias, a),
        **packing,
        cumsum_scale=_RCP_LN2,
        beta=1.0,
        threshold=20.0,
        T=t,
        H=h,
        D=dk,
        BT=_CHUNK,
    )
    matrix_shape = (b, t, h, _CHUNK)
    akk = torch.empty(matrix_shape, dtype=torch.float32, device=q.device)
    aqk = torch.empty_like(akk)
    inverse = torch.empty(matrix_shape, dtype=k.dtype, device=q.device)
    _ckfg_zero_matrices_ascend[(triton.cdiv(akk.numel(), 512),)](
        akk, aqk, inverse, akk.numel(), 512, num_warps=4, num_stages=2
    )
    common = dict(
        q=q,
        k=k,
        g=gate,
        beta=beta,
        A=akk,
        Aqk=aqk,
        scale=scale,
        **packing,
        T=t,
        H=h,
        K=dk,
        BT=_CHUNK,
        BC=_SUBCHUNK,
    )
    _ckfg_inter[nt, 16, b * h](**common, NC=4)
    _ckfg_intra[nt, 4, b * h](**common, BK=max(triton.next_power_of_2(dk), 16))
    _ckfg_inverse[nt, b * h](
        A=akk,
        Ai=inverse,
        **packing,
        T=t,
        H=h,
        BT=_CHUNK,
        USE_TMA=False,
        DOT_PRECISION="ieee",
    )
    w, u, kg = (torch.empty_like(k), torch.empty_like(v), torch.empty_like(k))
    _ckfg_recompute[nt, b * h](
        q=None,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=inverse,
        gk=gate,
        **packing,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=_CHUNK,
        BK=64,
        BV=64,
        DOT_PRECISION="ieee",
    )
    history = torch.empty((b, nt, h, dv, dk), dtype=k.dtype, device=q.device)
    v_new = torch.empty_like(u)
    state_args = dict(
        k=kg,
        v=u,
        w=w,
        v_new=v_new,
        gk=gate,
        h=history,
        cu_seqlens=packing["cu_seqlens"],
        chunk_offsets=packing["chunk_indices"] if offsets is None else offsets,
        VARLEN=cu is not None,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=_CHUNK,
    )
    state_grid = lambda meta: (triton.cdiv(dv, meta["BV"]), n * h)
    buffers = tuple(
        (
            torch.empty((n, h, dv, dk), dtype=torch.float32, device=q.device)
            for _ in range(2)
        )
    )
    total_chunks = triton.cdiv(t, _CHUNK)
    # Reuse static unrolls across adjacent chunk counts. The kernel masks the tail.
    state_chunks = min(8, max(2, triton.next_power_of_2(total_chunks)))
    for chunk in range(0, total_chunks, state_chunks):
        group = chunk // state_chunks
        current = buffers[group % 2]
        previous = state if group == 0 else buffers[(group - 1) % 2]
        _ckfg_state_ascend[state_grid](
            **state_args,
            # These pointers are only dereferenced under their respective flags.
            h0=current if previous is None else previous,
            ht=current if final is None else final,
            S=current,
            INITIAL=previous is not None,
            FINAL=final is not None,
            chunk=chunk,
            CHUNKS=state_chunks,
            BK=max(16, triton.next_power_of_2(dk)),
        )
    _ckfg_output[lambda meta: (triton.cdiv(dv, meta["BV"]), nt, b * h)](
        q=q,
        v=v_new,
        g=gate,
        h=history,
        o=v,
        A=aqk,
        **packing,
        scale=scale,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=_CHUNK,
    )
    return (v, final)

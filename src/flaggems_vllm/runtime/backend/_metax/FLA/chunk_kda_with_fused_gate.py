# SPDX-License-Identifier: Apache-2.0
"""MetaX KDA prefill with FP32 token recurrence for wide-key precision.

K=256 without q/k normalization can amplify chunk intermediate FP16 error.
This path retains FP32 state and residuals; it writes an independent output
buffer so stage tuning never consumes an overwritten value input.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.data_movement import contiguous_copy, copy_to, fill
from flaggems_vllm.ops.FLA.chunk_kda_with_fused_gate import (
    _CHUNK,
    _RCP_LN2,
    _SUBCHUNK,
    _ckfg_gate,
    _ckfg_inter,
    _ckfg_intra,
    _ckfg_inverse,
    _ckfg_output,
    _ckfg_recompute,
    _ckfg_state,
    _metadata,
    _norm_kernel,
    _stage_bench,
    _validate,
)
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@triton.heuristics(
    {
        "VARLEN": lambda a: a["CU"] is not None,
        "INITIAL": lambda a: a["H0"] is not None,
        "FINAL": lambda a: a["HT"] is not None,
        "HAS_BIAS": lambda a: a["Bias"] is not None,
    }
)
@libtuner(
    configs=runtime.get_tuned_config("ckfg_recurrent_metax"),
    key=["H", "K", "V"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit(do_not_specialize=["T"])
def _ckfg_recurrent_metax(
    Q,
    Kptr,
    Vptr,
    G,
    Beta,
    A,
    Bias,
    H0,
    O,
    HT,
    CU,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    VARLEN: tl.constexpr,
    INITIAL: tl.constexpr,
    FINAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    tile, nh = (tl.program_id(0), tl.program_id(1))
    seq, head = (nh // H, nh % H)
    if VARLEN:
        bos = tl.load(CU + seq).to(tl.int32)
        end = tl.load(CU + seq + 1).to(tl.int32)
        length = end - bos
    else:
        bos = seq * T
        length = T
    rk = tl.arange(0, BK)
    rv = tile * BV + tl.arange(0, BV)
    state = tl.full((BV, BK), 0, tl.float32)
    if INITIAL:
        state = tl.load(
            H0 + nh * V * K + rv[:, None] * K + rk[None, :],
            (rv[:, None] < V) & (rk[None, :] < K),
            other=0,
        ).to(tl.float32)
    a = -tl.exp(tl.load(A + head).to(tl.float32))
    bias = tl.full((BK,), 0, tl.float32)
    if HAS_BIAS:
        bias = tl.load(Bias + head * K + rk, rk < K, other=0).to(tl.float32)
    for pos in range(length):
        offset = (bos + pos) * H + head
        q = tl.load(Q + offset * K + rk, rk < K, other=0).to(tl.float32)
        k = tl.load(Kptr + offset * K + rk, rk < K, other=0).to(tl.float32)
        raw = tl.load(G + offset * K + rk, rk < K, other=0).to(tl.float32) + bias
        soft = tl.where(raw > 20, raw, tl.log(1 + tl.exp(raw)))
        state = state * tl.exp(a * soft)[None, :]
        v = tl.load(Vptr + offset * V + rv, rv < V, other=0).to(tl.float32)
        beta = tl.load(Beta + offset).to(tl.float32)
        residual = (v - tl.sum(state * k[None, :], 1)) * beta
        state = state + residual[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], 1) * scale
        tl.store(O + offset * V + rv, output, rv < V)
    if FINAL:
        tl.store(
            HT + nh * V * K + rv[:, None] * K + rk[None, :],
            state,
            (rv[:, None] < V) & (rk[None, :] < K),
        )


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
        for x, y in ((q, qn), (k, kn)):
            _norm_kernel[lambda meta: (triton.cdiv(b * t * h, meta["ROWS"]),)](
                x, y, b * t * h, dk, triton.next_power_of_2(dk)
            )
        q, k = (qn, kn)
    cu = contiguous_copy(cu_seqlens) if cu_seqlens is not None else None
    if dk == 256:
        precise = torch.empty_like(v)
        _ckfg_recurrent_metax[lambda meta: (triton.cdiv(dv, meta["BV"]), n * h)](
            Q=q,
            Kptr=k,
            Vptr=v,
            G=raw_g,
            Beta=beta,
            A=a,
            Bias=bias,
            H0=state,
            O=precise,
            HT=final,
            CU=cu,
            scale=scale,
            T=t,
            H=h,
            K=dk,
            V=dv,
            BK=triton.next_power_of_2(dk),
        )
        copy_to(precise, v)
        return (v, final)
    if cu is None:
        indices, offsets = (None, None)
        nt = triton.cdiv(t, _CHUNK)
    else:
        indices, offsets = _metadata(cu, t, q.device)
        nt = indices.shape[0]
    gate = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    _ckfg_gate[lambda meta: (triton.cdiv(dk, meta["BD"]), nt, b * h)](
        g=raw_g,
        A=a,
        y=gate,
        g_bias=bias,
        cu_seqlens=cu,
        chunk_indices=indices,
        cumsum_scale=_RCP_LN2,
        beta=1.0,
        threshold=20.0,
        T=t,
        H=h,
        D=dk,
        BT=_CHUNK,
    )
    matrix_shape = (b, t, h, _CHUNK)
    akk = fill(torch.empty(matrix_shape, dtype=torch.float32, device=q.device), 0)
    aqk = fill(torch.empty_like(akk), 0)
    common = dict(
        q=q,
        k=k,
        g=gate,
        beta=beta,
        A=akk,
        Aqk=aqk,
        scale=scale,
        cu_seqlens=cu,
        chunk_indices=indices,
        T=t,
        H=h,
        K=dk,
        BT=_CHUNK,
        BC=_SUBCHUNK,
    )
    _ckfg_inter[nt, 16, b * h](**common, NC=4)
    _ckfg_intra[nt, 4, b * h](**common, BK=max(triton.next_power_of_2(dk), 16))
    inverse = fill(torch.empty(matrix_shape, dtype=k.dtype, device=q.device), 0)
    _ckfg_inverse[nt, b * h](
        A=akk,
        Ai=inverse,
        cu_seqlens=cu,
        chunk_indices=indices,
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
        cu_seqlens=cu,
        chunk_indices=indices,
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
        h0=state,
        ht=final,
        cu_seqlens=cu,
        chunk_offsets=offsets,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=_CHUNK,
    )
    state_grid = lambda meta: (triton.cdiv(dv, meta["BV"]), n * h)
    _ckfg_state[state_grid](**state_args, g=None, Hg=h, USE_EXP2=True)
    _ckfg_output[lambda meta: (triton.cdiv(dv, meta["BV"]), nt, b * h)](
        q=q,
        v=v_new,
        g=gate,
        h=history,
        o=v,
        A=aqk,
        cu_seqlens=cu,
        chunk_indices=indices,
        scale=scale,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=_CHUNK,
    )
    return (v, final)

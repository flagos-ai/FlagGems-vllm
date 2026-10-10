# SPDX-License-Identifier: Apache-2.0
"""Allocation-only adapters for the existing vLLM 64-token KDA kernels.

The launch sequence and materialization dtypes follow vLLM's FLA KDA core.
Request indices and offsets are supplied by the framework. The adapters never
invoke the core hosts that initialize workspaces with Torch compute kernels.
"""

from functools import lru_cache
from importlib import import_module

import torch
import triton

from flaggems_vllm.ops.data_movement import fill


def _kernel(module, name, required):
    kernel = getattr(module, name, None)
    original = kernel
    while kernel is not None and not hasattr(kernel, "arg_names"):
        kernel = getattr(kernel, "fn", None)
    if (
        kernel is None
        or not hasattr(original, "__getitem__")
        or not set(required).issubset(kernel.arg_names)
    ):
        raise NotImplementedError(f"The installed KDA core lacks {name} ABI")
    return original


@lru_cache(maxsize=1)
def _prefill_core():
    """Resolve stable kernel capabilities before submitting any computation."""
    kda = import_module("vllm.model_executor.layers.fla.ops.kda")
    utils = import_module("vllm.model_executor.layers.fla.ops.solve_tril")
    delta_h = import_module("vllm.model_executor.layers.fla.ops.chunk_delta_h")
    kernels = (
        _kernel(
            kda,
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter",
            ("q", "k", "g", "beta", "A", "Aqk", "BT", "BC", "NC"),
        ),
        _kernel(
            kda,
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra",
            ("q", "k", "g", "beta", "A", "Aqk", "BT", "BC", "BK"),
        ),
        _kernel(
            utils,
            "merge_16x16_to_64x64_inverse_kernel",
            ("A", "Ai", "BT", "USE_TMA", "DOT_PRECISION"),
        ),
        _kernel(
            kda,
            "recompute_w_u_fwd_kernel",
            ("q", "k", "kg", "v", "beta", "w", "u", "A", "gk", "BT"),
        ),
        _kernel(
            delta_h,
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            ("k", "v", "w", "v_new", "gk", "h", "h0", "ht", "BT", "USE_EXP2"),
        ),
        _kernel(
            kda, "chunk_gla_fwd_kernel_o", ("q", "v", "g", "h", "o", "A", "BT", "scale")
        ),
    )
    if not callable(getattr(kda, "l2norm_fwd", None)):
        raise NotImplementedError("The installed KDA core lacks l2norm_fwd")
    return kernels, kda.l2norm_fwd, utils.is_tma_supported, utils.FLA_TRIL_PRECISION


@lru_cache(maxsize=1)
def _recurrent_core():
    kda = import_module("vllm.model_executor.layers.fla.ops.kda")
    return _kernel(
        kda,
        "fused_recurrent_gated_delta_rule_fwd_kernel",
        (
            "q",
            "k",
            "v",
            "g",
            "beta",
            "o",
            "h0",
            "ht",
            "cu_seqlens",
            "ssm_state_indices",
            "IS_KDA",
            "INPLACE_FINAL_STATE",
        ),
    )


def prefill_kda64(
    q,
    k,
    v,
    gate,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    core,
):
    kernels, _, use_tma, precision = core
    inter, intra, inverse, recompute, state, output = kernels
    b, t, h, dk = q.shape
    dv, bt = v.shape[-1], 64
    n = b if cu_seqlens is None else cu_seqlens.numel() - 1
    nt = triton.cdiv(t, bt) if cu_seqlens is None else chunk_indices.shape[0]
    a = fill(torch.empty((b, t, h, bt), dtype=torch.float32, device=q.device), 0)
    aqk = fill(torch.empty_like(a), 0)
    common = dict(
        q=q,
        k=k,
        g=gate,
        beta=beta,
        A=a,
        Aqk=aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=t,
        H=h,
        K=dk,
        BT=bt,
        BC=16,
    )
    inter[(nt, 16, b * h)](**common, NC=4)
    intra[(nt, 4, b * h)](**common, BK=max(triton.next_power_of_2(dk), 16))
    ai = fill(torch.empty(a.shape, dtype=k.dtype, device=k.device), 0)
    inverse[(nt, b * h)](
        A=a,
        Ai=ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=t,
        H=h,
        BT=bt,
        USE_TMA=use_tma,
        DOT_PRECISION=precision,
    )
    w, u, kg = torch.empty_like(k), torch.empty_like(v), torch.empty_like(k)
    recompute[(nt, b * h)](
        q=None,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=ai,
        gk=gate,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=bt,
        BK=64,
        BV=64,
        DOT_PRECISION="ieee",
    )
    history = torch.empty((b, nt, h, dv, dk), dtype=k.dtype, device=k.device)
    final = (
        torch.empty((n, h, dv, dk), dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )
    v_new = torch.empty_like(u)
    state[lambda meta: (triton.cdiv(dv, meta["BV"]), n * h)](
        k=kg,
        v=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=gate,
        h=history,
        h0=initial_state,
        ht=final,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=t,
        H=h,
        Hg=h,
        K=dk,
        V=dv,
        BT=bt,
        USE_EXP2=True,
    )
    output[lambda meta: (triton.cdiv(dv, meta["BV"]), nt, b * h)](
        q=q,
        v=v_new,
        g=gate,
        h=history,
        o=v,
        A=aqk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=bt,
    )
    return v, final


def recurrent_kda_vk(
    q,
    k,
    v,
    gate,
    beta,
    scale,
    initial_state,
    inplace_final_state,
    use_qk_l2norm_in_kernel,
    cu_seqlens,
    ssm_state_indices,
    kernel,
):
    """Preserve vLLM's [slots,H,V,K] state and per-token final-state ABI."""
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    n = b if cu_seqlens is None else cu_seqlens.numel() - 1
    bk, bv = triton.next_power_of_2(dk), min(triton.next_power_of_2(dv), 8)
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    final = (
        initial_state
        if inplace_final_state
        else torch.empty((t, h, dv, dk), dtype=initial_state.dtype, device=q.device)
    )
    stride_seq = 1 if ssm_state_indices is None else ssm_state_indices.stride(0)
    stride_tok = (
        1
        if ssm_state_indices is None or ssm_state_indices.ndim == 1
        else ssm_state_indices.stride(1)
    )
    # Mutation is launched once at vLLM's fixed 1-warp/3-stage configuration;
    # autotuning this launch would repeatedly update live recurrent state.
    kernel[(1, triton.cdiv(dv, bv), n * h)](
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        o=out,
        h0=initial_state,
        ht=final,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        scale=scale,
        N=n,
        T=t,
        B=b,
        H=h,
        HV=h,
        K=dk,
        V=dv,
        BK=bk,
        BV=bv,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=final.stride(0),
        stride_indices_seq=stride_seq,
        stride_indices_tok=stride_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=True,
        num_warps=1,
        num_stages=3,
    )
    return out, final

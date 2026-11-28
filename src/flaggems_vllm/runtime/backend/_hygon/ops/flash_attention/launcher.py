# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from flaggems_vllm.runtime.backend._hygon.ops.flash_attention import direct, prefill
from flaggems_vllm.runtime.backend._hygon.ops.flash_attention.common import (
    validate_inputs,
)
from flaggems_vllm.runtime.backend._hygon.ops.flash_attention.decode import launch_split
from flaggems_vllm.runtime.backend._hygon.ops.flash_attention.direct import (
    launch_direct,
)
from flaggems_vllm.runtime.backend._hygon.ops.flash_attention.scheduling import (
    select_plan,
)
from flaggems_vllm.runtime.backend._hygon.ops.flash_attention.tuning import (
    select_splits,
)


def launch_attention(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits=0,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    fa_version=2,
):
    if dropout_p != 0 or return_attn_probs:
        raise NotImplementedError("Dropout and attention probabilities are unsupported")
    if any(
        x is not None
        for x in (
            q_v,
            scheduler_metadata,
            q_descale,
            k_descale,
            v_descale,
            s_aux,
            cp_tot_seqused_k,
        )
    ):
        raise NotImplementedError("FA3 auxiliary inputs are unsupported")
    if num_splits != 0 or cp_world_size != 1 or cp_rank != 0 or fa_version != 2:
        raise NotImplementedError(
            "Explicit splits, context parallel and FA3 are unsupported"
        )
    validate_inputs(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        block_table,
        out,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
    )
    window = (-1, -1) if window_size is None else tuple(window_size)
    if len(window) != 2 or any(not isinstance(x, int) or x < -1 for x in window):
        raise ValueError("window_size must contain two integers >= -1")
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    softcap = float(softcap)
    if not math.isfinite(scale) or not math.isfinite(softcap) or softcap < 0:
        raise ValueError("Scale must be finite and softcap finite/nonnegative")
    plan = select_plan(
        max_seqlen_q,
        q.shape[1] // k.shape[-2],
        page=k.shape[1] if block_table is not None else 0,
        dim=q.shape[-1],
        total_q=q.shape[0],
        max_k=max_seqlen_k,
        num_seqs=cu_seqlens_q.numel() - 1,
        kv_heads=k.shape[-2],
        plain=window == (-1, -1) and softcap == 0 and alibi_slopes is None,
        bf16=q.dtype == torch.bfloat16,
    )
    with torch.cuda.device(q.device):
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse = torch.empty(
            (q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device
        )
        if q.shape[0]:
            launch_args = (
                q,
                k,
                v,
                out,
                lse,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                scale,
                causal,
                window,
                softcap,
            )
            splits = plan.splits
            if plan.family in ("split", "hybrid"):
                splits = select_splits(launch_args, plan)
            _execute_plan(launch_args, plan, splits)
    return (out, lse) if return_softmax_lse else out


def _execute_plan(arguments, plan, splits):
    q, k, _, _, _, _, _, usedk, table, _, maxq, maxk, _, causal, _, _ = arguments
    if plan.family == "split":
        launch_split(*arguments, splits=splits)
    elif plan.family in ("paged", "hybrid", "tp2_prefill"):
        short_limit = 128 if plan.family == "hybrid" else 0
        kernel = direct.paged_attention_kernel
        options = dict(SHORT_LIMIT=short_limit)
        if q.dtype == torch.bfloat16 and maxq >= 4096:
            kernel = prefill.attention_kernel
            options["KNORM"] = prefill.prepare_norms(k, table, usedk, maxk)
        elif (
            q.dtype == torch.float16
            and causal
            and maxq >= 4096
            and q.shape[1] // k.shape[-2] in (4, 8)
        ):
            kernel = prefill.attention_kernel
            options["KNORM"] = None
        launch_direct(*arguments, kernel=kernel, **options)
        if short_limit:
            launch_split(*arguments, splits=splits, short_limit=short_limit)
    else:
        launch_direct(*arguments, packed=plan.family == "packed_decode")

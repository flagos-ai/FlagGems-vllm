# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.autotune import (
    exact_key,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.autotune import (
    run as run_tuned,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.cached import (
    cached_routes,
    launch_cached,
    supports_cached,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.capabilities import (
    supports_tle,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.common import (
    validate_inputs,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.decode_tile import (
    launch_decode_tile,
    supports_decode_tile,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.direct import (
    launch_direct,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
    current_config,
    record_call,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.policy import (
    Candidate,
    Problem,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.scheduling import (
    AttentionPlan,
    select_plan,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.single_kv import (
    launch_single_kv,
    supports_single_kv,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.splitkv import (
    launch_splitkv,
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
    """Inference-only MUSA frontend; no data conversion or torch compute path."""
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
        max_seqlen_k=max_seqlen_k,
        total_q=q.shape[0],
        num_heads=q.shape[1],
        batch_size=cu_seqlens_q.numel() - 1,
        head_size=q.shape[2],
    )
    with torch.musa.device(q.device):
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse = torch.empty(
            (q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device
        )
        if not q.shape[0]:
            return (out, lse) if return_softmax_lse else out
        route, routes = "generic", ["generic"]
        if supports_single_kv(
            q, k, block_table, max_seqlen_q, max_seqlen_k, window, softcap, alibi_slopes
        ):
            route, routes = "single_kv", ["single_kv"]
            plan = AttentionPlan("packed_decode", 0, 1)
        elif supports_decode_tile(
            q, k, block_table, max_seqlen_q, window, softcap, alibi_slopes, plan
        ):
            route = "decode_tile"
            routes.append(route)
            if supports_tle():
                routes.append("tle_decode")
        elif supports_cached(
            q,
            k,
            v,
            block_table,
            max_seqlen_q,
            window,
            softcap,
            alibi_slopes,
            plan,
            max_seqlen_k,
        ):
            route, cached_options = cached_routes(q, scale)
            routes.extend(cached_options)
        problem = Problem(
            max_seqlen_q,
            max_seqlen_k,
            q.shape[0],
            q.shape[1],
            q.shape[1] // k.shape[-2],
            q.shape[2],
            cu_seqlens_q.numel() - 1,
            tuple(routes),
        )
        default = Candidate(route, plan)
        key = exact_key(
            (
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k,
                block_table,
                alibi_slopes,
            ),
            (
                max_seqlen_q,
                max_seqlen_k,
                scale,
                causal,
                window,
                softcap,
                deterministic,
            ),
            q.device.index,
        )

        record_call(problem, default, key)

        def pipeline():
            config = current_config()
            selected_plan = config.plan
            if config.route == "single_kv":
                return launch_single_kv(
                    q, k, v, out, lse, cu_seqlens_q, seqused_k, block_table, scale
                )
            if config.route == "decode_tile":
                return launch_decode_tile(
                    q, k, v, out, lse, cu_seqlens_q, seqused_k, block_table, scale
                )
            if config.route == "tle_decode":
                from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.tle_decode import (
                    launch_tle_decode_tile,
                )

                return launch_tle_decode_tile(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    scale,
                    selected_plan.block_m,
                    selected_plan.num_warps,
                )
            if config.route != "generic":
                return launch_cached(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    scale,
                    causal,
                    max_seqlen_k,
                )
            launch = launch_splitkv if selected_plan.num_splits > 1 else launch_direct
            return launch(
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
                scale,
                causal,
                window,
                softcap,
                selected_plan,
                maxk=max_seqlen_k,
            )

        run_tuned(pipeline, key, problem, default)
    return (out, lse) if return_softmax_lse else out

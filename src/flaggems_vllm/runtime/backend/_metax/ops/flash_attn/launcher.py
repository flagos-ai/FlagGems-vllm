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

import torch

from .direct import launch_direct
from .ragged import launch_compact_worklist, launch_d256_tn_compact_worklist
from .scheduling import (
    D256_TN_BLOCK_M,
    D256_TN_BLOCK_N,
    MetaXAttentionPlan,
    MetaXAttentionScheduler,
    MetaXKernelFamily,
    MetaXTaskMapper,
    validate_metax_attention_plan,
)
from .splitkv import launch_d256_tn_splitkv, launch_page16_decode, launch_splitkv
from .tn_direct import launch_d256_tn_direct, launch_d256_tn_direct_fast

D256_TN_SPLITKV_MIN_K = 32768


D256_TN_SPLITKV_MAX_Q = 132


D256_TN_SPLITKV_MAX_TOTAL_Q = 139


def launch_metax_attention(
    plan: MetaXAttentionPlan,
    params,
    *,
    max_seqlen_q,
    max_seqlen_k,
    batch_size,
    num_heads,
    num_heads_k,
    total_q,
    head_size,
    is_paged,
):
    """Validate a plan and dispatch through the selected family adapter."""
    validate_metax_attention_plan(plan)
    if total_q == 0:
        return None
    if plan.family is MetaXKernelFamily.PAGE16_DECODE:
        return launch_page16_decode(params, batch_size=batch_size)
    if plan.family is MetaXKernelFamily.SPLIT_KV:
        use_d256_tn_splitkv = (
            is_paged
            and params.q_ptr.dtype == torch.bfloat16
            and (head_size == 256)
            and (params.block_size == 32)
            and (params.h_hk_ratio == 8)
            and (1 < max_seqlen_q <= D256_TN_SPLITKV_MAX_Q)
            and (total_q <= D256_TN_SPLITKV_MAX_TOTAL_Q)
            and (max_seqlen_k >= D256_TN_SPLITKV_MIN_K)
            and params.is_causal
            and (not params.is_local)
            and (not params.is_dropout)
            and (not params.is_alibi)
            and (not params.is_softcap)
        )
        if use_d256_tn_splitkv:
            return launch_d256_tn_splitkv(
                params,
                max_seqlen_q=max_seqlen_q,
                batch_size=batch_size,
                num_heads=num_heads,
                total_q=total_q,
                num_splits=plan.max_splits,
            )
        return launch_splitkv(
            params,
            max_seqlen_q=max_seqlen_q,
            batch_size=batch_size,
            num_heads=num_heads,
            total_q=total_q,
            head_size=head_size,
            num_splits=plan.max_splits,
            block_m=plan.split_kv_block_m,
            block_n=plan.split_kv_block_n,
            q_tiles=plan.split_kv_q_tiles,
        )
    if plan.family is MetaXKernelFamily.LEGACY:
        return launch_direct(
            params,
            max_seqlen_q=max_seqlen_q,
            batch_size=batch_size,
            num_heads=num_heads,
            total_q=total_q,
            head_size=head_size,
            is_paged=is_paged,
            grid_order=plan.grid_order,
        )
    if plan.family is MetaXKernelFamily.D256_TN_DIRECT:
        from .ragged import maybe_repack

        if max_seqlen_k < 32768 or batch_size == 1:
            params = maybe_repack(
                params,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                total_q=total_q,
            )
        if plan.task_mapper is MetaXTaskMapper.COMPACT_WORKLIST:
            return launch_d256_tn_compact_worklist(
                params,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                num_heads=num_heads,
                total_q=total_q,
                grid_order=plan.grid_order,
                task_upper=plan.worklist_task_upper,
                block_m=plan.worklist_block_m,
                block_n=D256_TN_BLOCK_N,
                allow_split_kv=plan.allow_split_kv,
            )
        fast_causal_aligned = (
            batch_size == 1
            and total_q == max_seqlen_q
            and (max_seqlen_k >= max_seqlen_q)
            and ((max_seqlen_k - max_seqlen_q) % D256_TN_BLOCK_M == 0)
        )
        if fast_causal_aligned:
            return launch_d256_tn_direct_fast(
                params,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                num_heads=num_heads,
                total_q=total_q,
                block_m=D256_TN_BLOCK_M,
                block_n=D256_TN_BLOCK_N,
                grid_order=plan.grid_order,
            )
        return launch_d256_tn_direct(
            params,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            batch_size=batch_size,
            num_heads=num_heads,
            total_q=total_q,
            block_m=D256_TN_BLOCK_M,
            block_n=D256_TN_BLOCK_N,
            grid_order=plan.grid_order,
            allow_split_kv=plan.allow_split_kv,
        )
    if plan.family is MetaXKernelFamily.COMPACT_WORKLIST:
        return launch_compact_worklist(
            params,
            max_seqlen_q=max_seqlen_q,
            batch_size=batch_size,
            num_heads=num_heads,
            total_q=total_q,
            head_size=head_size,
            is_paged=is_paged,
            grid_order=plan.grid_order,
            task_upper=plan.worklist_task_upper,
            block_m=plan.worklist_block_m,
        )
    raise RuntimeError(f"unsupported MetaX attention family: {plan.family.value}")


def launch_attention(params, *, num_splits=0):
    """Plan and launch C550 attention after common API input preparation."""
    plan = MetaXAttentionScheduler.build(
        is_bfloat16=params.q_ptr.dtype is torch.bfloat16,
        is_paged=params.is_paged,
        block_size=params.block_size,
        is_cu_seqlens_q=params.is_cu_seqlens_q,
        is_seqused_k=params.is_seqused_k,
        max_seqlen_q=params.seqlen_q,
        max_seqlen_k=params.seqlen_k,
        total_q=params.total_q,
        batch_size=params.b,
        num_heads=params.h,
        num_heads_k=params.hk,
        head_size=params.d,
        is_causal=params.is_causal,
        is_local=params.is_local,
        is_dropout=params.is_dropout,
        is_alibi=params.is_alibi,
        is_softcap=params.is_softcap,
        seqlenq_ngroups_swapped=params.seqlenq_ngroups_swapped,
        num_splits=num_splits,
    )
    return launch_metax_attention(
        plan,
        params,
        max_seqlen_q=params.seqlen_q,
        max_seqlen_k=params.seqlen_k,
        batch_size=params.b,
        num_heads=params.h,
        num_heads_k=params.hk,
        total_q=params.total_q,
        head_size=params.d,
        is_paged=params.is_paged,
    )

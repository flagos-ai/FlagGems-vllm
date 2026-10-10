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

import math

import torch
import triton

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.common import fill_tensor
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.direct import (
    flash_varlen_int8_fwd_kernel,
    launch_direct,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.ragged import (
    launch_compact_worklist,
    launch_d256_tn_compact_worklist,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.scheduling import (
    D256_TN_BLOCK_M,
    D256_TN_BLOCK_N,
    MetaXAttentionPlan,
    MetaXAttentionScheduler,
    MetaXKernelFamily,
    MetaXTaskMapper,
    validate_metax_attention_plan,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.splitkv import (
    launch_d256_tn_splitkv,
    launch_page16_decode,
    launch_splitkv,
)
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.tn_direct import (
    launch_d256_tn_direct,
    launch_d256_tn_direct_fast,
)

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
        from flaggems_vllm.runtime.backend._metax.ops.flash_attention.ragged import (
            maybe_repack,
        )

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


def normalize_int8_descale(descale, batch_size, num_heads, nblocks, device, name):
    if descale is None:
        raise ValueError(f"{name} is required for W8A8-INT8 attention")
    if descale.device != device:
        raise ValueError(f"{name} must be on the same device as q")
    if descale.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if descale.ndim == 0:
        descale = descale.reshape(1, 1, 1).expand(batch_size, num_heads, nblocks)
    elif descale.ndim == 1:
        if descale.numel() == 1:
            descale = descale.reshape(1, 1, 1).expand(batch_size, num_heads, nblocks)
        else:
            if descale.numel() != num_heads:
                raise ValueError(f"{name} 1D scale must have H elements")
            descale = descale.reshape(1, num_heads, 1).expand(
                batch_size, num_heads, nblocks
            )
    elif descale.ndim == 2:
        if descale.shape != (batch_size, num_heads):
            raise ValueError(f"{name} 2D scale must be [B, H]")
        descale = descale[:, :, None].expand(batch_size, num_heads, nblocks)
    else:
        if descale.ndim != 3 or descale.shape != (
            batch_size,
            num_heads,
            nblocks,
        ):
            raise ValueError(f"{name} must be [B, H, nblocks]")
    return descale


def flash_attn_varlen_func_w8a8_int8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # Compatibility arguments from the shared FlashAttention API.
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """MetaX INT8 attention with FP32 softmax, for inference only.

    Paged KV and GQA use FP16 PV and support head dimensions 64 and 128.
    Descales index logical 128-token blocks, independently of physical cache pages. The
    output defaults to BF16 or uses a supplied FP16/BF16 buffer. Forward
    inference only; unsupported training and scheduling modes raise errors.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not supported by this inference path")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs is not supported")
    if q_v is not None:
        raise NotImplementedError("q_v is not supported")
    if scheduler_metadata is not None or s_aux is not None:
        raise NotImplementedError("scheduler arguments are not supported")
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("context parallel attention is not supported")
    if runtime.device.vendor_name != "metax":
        raise NotImplementedError(
            "This W8A8 INT8 attention implementation only supports MetaX"
        )
    if fa_version != 2:
        raise NotImplementedError("Only FA2 is implemented.")
    if num_splits != 0:
        raise NotImplementedError("Explicit num_splits is not implemented.")
    assert (
        cu_seqlens_k is not None or seqused_k is not None
    ), "cu_seqlens_k or seqused_k must be provided"
    assert (
        cu_seqlens_k is None or seqused_k is None
    ), "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert (
        block_table is None or seqused_k is not None
    ), "seqused_k must be provided if block_table is provided"
    if seqused_k is not None and block_table is None:
        raise NotImplementedError("seqused_k without a paged KV cache is not supported")

    if not isinstance(max_seqlen_q, int) or not isinstance(max_seqlen_k, int):
        raise TypeError("max_seqlen_q and max_seqlen_k must be Python integers")
    if max_seqlen_q < 0 or max_seqlen_k < 0:
        raise ValueError("max_seqlen_q and max_seqlen_k must be nonnegative")
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, heads, head_dim]")
    if max_seqlen_q == 0 and q.shape[0] != 0:
        raise ValueError("max_seqlen_q must be positive when q is nonempty")
    expected_kv_ndim = 4 if block_table is not None else 3
    if k.ndim != expected_kv_ndim or v.ndim != expected_kv_ndim:
        raise ValueError("k and v rank does not match the selected cache layout")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise TypeError("q, k, and v must have dtype torch.int8 for W8A8-INT8")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise NotImplementedError("q, k, and v must be contiguous in head_dim")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    head_size = q.shape[-1]
    if not 8 <= head_size <= 256 or head_size % 8 != 0:
        raise NotImplementedError("head_dim must be a multiple of 8 between 8 and 256")
    if k.shape[-1] != head_size:
        raise ValueError("q, k, and v must have the same head_dim")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.device != q.device:
        raise ValueError("cu_seqlens_q must be an int32 tensor on q.device")
    if cu_seqlens_k is not None and (
        cu_seqlens_k.dtype != torch.int32 or cu_seqlens_k.device != q.device
    ):
        raise ValueError("cu_seqlens_k must be an int32 tensor on q.device")
    if seqused_k is not None and (
        seqused_k.dtype != torch.int32 or seqused_k.device != q.device
    ):
        raise ValueError("seqused_k must be an int32 tensor on q.device")
    if block_table is not None and (
        block_table.dtype != torch.int32 or block_table.device != q.device
    ):
        raise ValueError("block_table must be an int32 tensor on q.device")
    if out is not None:
        if out.shape != q.shape or out.device != q.device:
            raise ValueError("out must match q shape and device")
        if out.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("out must have dtype torch.float16 or torch.bfloat16")
        if out.stride(-1) != 1:
            raise NotImplementedError("out must be contiguous in head_dim")
        if out.numel() and out.data_ptr() in (
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
        ):
            raise ValueError("out must not alias q, k, or v")

    num_heads_k = k.shape[2] if block_table is not None else k.shape[1]
    if num_heads_k <= 0 or q.shape[1] <= 0 or q.shape[1] % num_heads_k != 0:
        raise ValueError("The number of KV heads must divide the number of query heads")
    if q.shape[1] != num_heads_k and head_size not in (64, 128):
        raise NotImplementedError("GQA supports head dimensions 64 and 128")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])

    batch_size = cu_seqlens_q.numel() - 1
    assert batch_size > 0, "batch_size must be positive"
    if cu_seqlens_q.ndim != 1 or not cu_seqlens_q.is_contiguous():
        raise ValueError("cu_seqlens_q must be a contiguous 1D tensor")
    if cu_seqlens_k is not None and (
        cu_seqlens_k.ndim != 1
        or cu_seqlens_k.numel() != batch_size + 1
        or not cu_seqlens_k.is_contiguous()
    ):
        raise ValueError("cu_seqlens_k must be contiguous with shape [batch + 1]")
    if seqused_k is not None and (
        seqused_k.ndim != 1
        or seqused_k.numel() != batch_size
        or not seqused_k.is_contiguous()
    ):
        raise ValueError("seqused_k must be contiguous with shape [batch]")
    if block_table is not None and (
        block_table.ndim != 2
        or block_table.shape[0] != batch_size
        or block_table.stride(-1) != 1
    ):
        raise ValueError(
            "block_table must be contiguous in its last dimension with shape [batch, pages]"
        )
    if alibi_slopes is not None and (
        alibi_slopes.device != q.device
        or alibi_slopes.dtype != torch.float32
        or alibi_slopes.stride(-1) != 1
        or alibi_slopes.shape not in ((q.shape[1],), (batch_size, q.shape[1]))
    ):
        raise ValueError(
            "alibi_slopes must be FP32 [heads] or [batch, heads] on q.device"
        )
    if q.shape[0] == 0 or max_seqlen_k == 0:
        if out is None:
            out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        lse = (
            torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
            if return_softmax_lse
            else None
        )
        if q.shape[0] != 0:
            fill_tensor(out, 0.0)
            if return_softmax_lse:
                fill_tensor(lse, float("inf"))
        return (out, lse) if return_softmax_lse else out
    from flaggems_vllm.runtime.backend._thead.fused.attention import (
        PACK_TILE,
        _flash_int8_merge_splits,
        _flash_int8_pack_kv,
    )

    normalized_descales = []
    for descale, heads, max_length, name in (
        (q_descale, q.shape[1], max_seqlen_q, "q_descale"),
        (k_descale, num_heads_k, max_seqlen_k, "k_descale"),
        (v_descale, num_heads_k, max_seqlen_k, "v_descale"),
    ):
        # Padded maximum lengths need not allocate unused logical scale blocks.
        blocks = (
            descale.shape[2]
            if descale is not None and descale.ndim == 3
            else triton.cdiv(max_length, 128)
        )
        normalized_descales.append(
            normalize_int8_descale(descale, batch_size, heads, blocks, q.device, name)
        )
    q_descale, k_descale, v_descale = normalized_descales
    batch = cu_seqlens_q.numel() - 1
    total, heads, dim = q.shape
    kv_heads = k.shape[-2]
    group = heads // kv_heads
    paged = block_table is not None
    page_size = k.shape[1] if paged else 0
    # Bound packing by the physical cache size, including shared-page layouts.
    logical_kv = (
        paged
        and dim in (64, 128)
        and max_seqlen_q >= 128
        and max_seqlen_k > 0
        and k.shape[0] > 0
        and batch * max_seqlen_k <= k.shape[0] * page_size
    )
    left, right = real_window_size
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = (
        torch.empty((heads, total), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )
    # Folding short queries reuses each KV tile across its grouped query heads.
    fold = max_seqlen_q <= 16 and 1 < group <= 16 and group & (group - 1) == 0
    block_m = 32 if fold and max_seqlen_q > 4 else 16
    if max_seqlen_q >= 128:
        block_m = 64
    block_n = 128 if max_seqlen_k >= 512 else 64
    num_warps = 4
    if dim > 128:
        block_m, block_n = 16, 64
    if dim == 128 and logical_kv:
        block_m, block_n, num_warps = 32, 64, 2
    query_tile = block_m // group if fold else block_m
    grid_heads = kv_heads if fold else heads
    parallel_heads = batch * grid_heads
    if left < 0 and right < 0 and max_seqlen_q <= 4 and max_seqlen_k >= 512:
        splits = 4 if parallel_heads <= 128 else 2
    elif left < 0 and right < 0 and 4 < max_seqlen_q <= 16 and max_seqlen_k >= 2048:
        splits = 2
    else:
        splits = 1
    if dim not in (64, 128):
        splits = 1
    split_kv = splits > 1
    # The compact mapper reads CUQ on device, including empty and padded requests.
    compact = max_seqlen_q > 16 and total * 2 < batch * max_seqlen_q
    tiles = (
        triton.cdiv(total, query_tile) + batch - 1
        if compact
        else triton.cdiv(max_seqlen_q, query_tile)
    )
    grid = (tiles * splits, 1 if compact else batch, grid_heads)
    with torch_device_fn.device(q.device):
        if logical_kv:
            packed_shape = (batch, max_seqlen_k, kv_heads, dim)
            packed_stride = (
                kv_heads * max_seqlen_k * dim,
                dim,
                max_seqlen_k * dim,
                1,
            )
            packed_key = torch.empty_strided(
                packed_shape, packed_stride, dtype=torch.int8, device=k.device
            )
            packed_value = torch.empty_strided(
                packed_shape, packed_stride, dtype=torch.float16, device=v.device
            )
            _flash_int8_pack_kv[
                (triton.cdiv(max_seqlen_k, PACK_TILE.value), batch, kv_heads)
            ](
                k,
                v,
                packed_key,
                packed_value,
                block_table,
                seqused_k,
                cu_seqlens_q,
                max_seqlen_k,
                dim,
                kv_heads,
                page_size,
                block_table.stride(0),
                *k.stride()[:3],
                *v.stride()[:3],
                LOGICAL=True,
                SKIP_SHORT=False,
            )
            k, v = packed_key, packed_value
        if split_kv:
            kernel_out = torch.empty(
                (splits, total, heads, dim), dtype=torch.float32, device=q.device
            )
            kernel_stats = torch.empty(
                (splits, 2, heads, total), dtype=torch.float32, device=q.device
            )
        else:
            kernel_out, kernel_stats = out, lse
        flash_varlen_int8_fwd_kernel[grid](
            q,
            k,
            v,
            kernel_out,
            kernel_stats,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            block_table,
            q_descale,
            k_descale,
            v_descale,
            alibi_slopes,
            q.stride(0),
            q.stride(1),
            k.stride(-3),
            k.stride(-2),
            k.stride(0) if paged else 0,
            v.stride(-3),
            v.stride(-2),
            v.stride(0) if paged else 0,
            kernel_out.stride(-3),
            kernel_out.stride(-2),
            *q_descale.stride(),
            *k_descale.stride(),
            *v_descale.stride(),
            block_table.stride(0) if paged else 0,
            (
                alibi_slopes.stride(0)
                if alibi_slopes is not None and alibi_slopes.ndim == 2
                else 0
            ),
            total,
            group,
            dim,
            triton.next_power_of_2(max(dim, 64)),
            page_size,
            paged,
            causal,
            left,
            right,
            softcap,
            softmax_scale,
            alibi_slopes is not None,
            return_softmax_lse or split_kv,
            KV_SPLITS=splits,
            OUT_SPLIT_STRIDE=kernel_out.stride(0) if split_kv else 0,
            LSE_SPLIT_STRIDE=kernel_stats.stride(0) if split_kv else 0,
            LOGICAL_KV=logical_kv,
            BATCH=batch,
            COMPACT=compact,
            FOLD=fold,
            BM=block_m,
            BN=block_n,
            num_warps=num_warps,
            num_stages=1,
            pipeline="basic",
        )
        if split_kv:
            _flash_int8_merge_splits[(total, heads)](
                kernel_out,
                kernel_stats,
                out,
                lse,
                cu_seqlens_q,
                total,
                heads,
                dim,
                batch,
                splits,
                out.stride(0),
                out.stride(1),
                return_softmax_lse,
                num_warps=1,
            )
    return (out, lse) if return_softmax_lse else out

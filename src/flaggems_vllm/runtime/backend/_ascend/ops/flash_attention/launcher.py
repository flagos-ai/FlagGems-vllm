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

from dataclasses import replace
from math import gcd

import torch
import triton

from flaggems_vllm.runtime import torch_device_fn

from .common import prepare_contiguous_kv, prepare_packed_inputs, use_packed_k_prescale
from .decode import (
    ascend_attention_short_kv_staged,
    launch_decode,
    launch_decode256_staged,
    launch_decode_actual,
    launch_paged_decode,
)
from .direct import ascend_attention_persistent, ascend_attention_persistent_pipelined
from .mixed import (
    ascend_attention_short_kv,
    launch_grouped_tail,
    launch_split_full_body,
)
from .scheduling import (
    allocate_long_request_list,
    build_mixed_decode_plan,
    build_mixed_request_lists,
    build_plan,
    merge_splits,
)

# ---- launcher ----


def _launch_pipelined(
    params,
    plan,
    workers,
    packed,
    skip_short_kv=0,
    request_ids=None,
    request_count=None,
    task_m=None,
    compact_q1=False,
    defer_merge=False,
    merge_q1=None,
    stage_q1=False,
    compact_long_tail=False,
    body_task_pipeline=False,
    native_split_q1=False,
    local_body_stream=False,
    short_split_q1=False,
    merge_min_split_k=32768,
    residual_merge=None,
):
    q, k, v, qcap, contig_k, q_direct = packed
    pipeline_bm = 128
    scores = torch.empty(
        (workers, 3, pipeline_bm, plan.kv_stack),
        dtype=torch.float32,
        device=params.q_ptr.device,
    )
    probs = torch.empty(
        (workers, 3, pipeline_bm, plan.kv_stack),
        dtype=params.q_ptr.dtype,
        device=params.q_ptr.device,
    )
    pvwork = torch.empty(
        (workers, 3, pipeline_bm, params.d),
        dtype=torch.float32,
        device=params.q_ptr.device,
    )
    accwork = torch.empty(
        (workers, pipeline_bm, params.d),
        dtype=torch.float32,
        device=params.q_ptr.device,
    )
    partial_q = params.b if compact_q1 else params.total_q
    kernel_o, kernel_lse = params.o_ptr, params.softmax_lse_ptr
    row_stride, head_stride = params.o_row_stride, params.o_head_stride
    if plan.kv_splits > 1:
        kernel_o = torch.empty(
            (plan.kv_splits, partial_q, params.h, params.d),
            dtype=torch.float32,
            device=params.q_ptr.device,
        )
        kernel_lse = torch.empty(
            (plan.kv_splits, params.h, partial_q),
            dtype=torch.float32,
            device=params.q_ptr.device,
        )
        row_stride, head_stride = params.h * params.d, params.d
    # One request has a host-known Q length. Use a fixed small M only when
    # every head group fits the same bucket, retaining a single kernel launch.
    automatic_task_m = task_m is None
    if automatic_task_m:
        task_m = 0
    if (
        automatic_task_m
        and params.b == 1
        and params.seqlen_q <= 64
        and params.seqlen_k >= 32768
    ):
        qlen = params.total_q
        group = params.h // params.hk
        head_rows = min(group, max(1, (128 // max(qlen, 1)) // 2 * 2))
        full_m = max(16, triton.next_power_of_2(qlen * head_rows))
        tail_heads = group % head_rows
        tail_m = (
            max(16, triton.next_power_of_2(qlen * tail_heads)) if tail_heads else full_m
        )
        if full_m == tail_m and full_m < 128:
            task_m = full_m
    # KV4K mixed requests have full-M128 body plus at most16-row tails.
    # Reuse verified D1 QK-only lookahead and the native compact-tail stream.
    # Preserve FP32 O for low-batch GQA4; only the M128 body changes pipeline.
    fp32_smallbatch_4k = (
        params.q_ptr.dtype == torch.bfloat16
        and params.is_cu_seqlens_q
        and params.is_paged
        and params.is_causal
        and params.d == 256
        and params.block_size in (16, 32)
        and params.h // params.hk == 4
        and 1 < params.b < 8
        and params.total_q >= 8192
        and params.total_q > 128 * params.b
        and params.total_q < params.b * params.seqlen_q
        and 2304 < params.seqlen_q <= 4608
        and 2304 < params.seqlen_k <= 4608
        and plan.kv_splits == 1
        and task_m == 0
        and request_ids is not None
        and not compact_q1
        and not defer_merge
        and merge_q1 is None
        and not stage_q1
        and not body_task_pipeline
    )
    cross_4k = (
        params.q_ptr.dtype == torch.bfloat16
        and params.is_cu_seqlens_q
        and params.is_paged
        and params.is_causal
        and params.d == 256
        and params.block_size in (16, 32)
        and (params.h // params.hk == 8 or local_body_stream or fp32_smallbatch_4k)
        and params.b > 1
        and params.total_q >= 8192
        and params.b < params.total_q
        and params.total_q < params.b * params.seqlen_q
        and 512 <= params.seqlen_q <= 4608
        and 2304 < params.seqlen_k <= 4608
        and contig_k > 0
        and plan.kv_splits == 1
        and task_m in (0, 16)
        and request_ids is not None
        and merge_q1 is None
        and not compact_q1
        and not defer_merge
    )
    # The existing D1 stream can seed the next Q tile for standalone 4K
    # prefill too; retain its 512-key online softmax and original packing.
    prefill_4k_stream = (
        params.q_ptr.dtype == torch.bfloat16
        and params.is_cu_seqlens_q
        and params.is_paged
        and params.is_causal
        and params.d == 256
        and params.block_size in (16, 32)
        and params.h // params.hk in (4, 8)
        and params.b == 1
        and params.total_q == params.seqlen_q
        and 2304 < params.seqlen_q <= 4608
        and 2304 < params.seqlen_k <= 4608
        and contig_k > 0
        and plan.kv_splits == 1
        and task_m == 0
        and request_ids is None
        and merge_q1 is None
        and not compact_q1
        and not defer_merge
    )
    cross_4k = cross_4k or prefill_4k_stream
    # Short-KV prefill needs enough query-head work to amortize specialization.
    single_head_index = (
        params.is_cu_seqlens_q
        and params.is_causal
        and plan.kv_splits == 1
        and 128 <= params.seqlen_q <= 2048
        and 128
        <= params.seqlen_k
        <= (
            2304
            if (
                task_m == 0
                and params.total_q * params.h >= 8192
                and params.q_ptr.dtype == torch.bfloat16
                and params.is_paged
                and params.h // params.hk == 8
            )
            else 2048
        )
        and params.total_q * params.h >= 8192
    )
    single_head_index = single_head_index or cross_4k
    acc_ub_rows = 0 if task_m else 32
    if task_m == 16 and request_ids is not None:
        # Mixed Q1 M16 tasks have eight rows per Vector, all fitting in UB.
        acc_ub_rows = 8
    short_kv = (
        params.q_ptr.dtype == torch.bfloat16
        and params.is_cu_seqlens_q
        and params.is_paged
        and params.is_causal
        and params.d == 256
        and params.block_size in (16, 32)
        and params.h // params.hk in (4, 8)
        and params.b > 1
        and params.b < params.total_q < params.b * params.seqlen_q
        and 512 <= params.seqlen_q <= (4608 if stage_q1 else 1152)
        and 512 <= params.seqlen_k <= (4608 if stage_q1 else 2304)
        and contig_k > 0
        and plan.kv_splits == 1
        and task_m in (0, 16)
        and request_ids is not None
        and merge_q1 is None
        and not compact_q1
        and not defer_merge
    )
    short_kv = short_kv or cross_4k or compact_long_tail
    cross_q_task = short_kv and task_m == 0 and single_head_index
    if cross_q_task:
        acc_ub_rows = 32
        short_kv = False
    pipeline_kernel = ascend_attention_persistent_pipelined
    if short_kv:
        pipeline_kernel = (
            ascend_attention_short_kv_staged if stage_q1 else ascend_attention_short_kv
        )
    task_options = {}
    if fp32_smallbatch_4k:
        # Existing Q1-only launch does not own these tails: retain M128 writers.
        task_options["INCLUDE_COMPACT_TAIL"] = True
    if prefill_4k_stream:
        task_options["COMPACT_BODY_O"] = True
    inline_prefill_tail = prefill_4k_stream and 0 < params.total_q % 128 <= 16
    if inline_prefill_tail:
        # A single fixed M128 stream owns the final masked rows as well.
        # Avoid a separate compact-tail launch and its sparsely filled wave.
        task_options["INCLUDE_COMPACT_TAIL"] = True
    if cross_q_task and body_task_pipeline:
        task_scores = torch.empty(
            (workers, 3, 128, 1536), dtype=torch.float32, device=params.q_ptr.device
        )
        task_probs = torch.empty(
            (workers, 3, 128, 1536),
            dtype=params.q_ptr.dtype,
            device=params.q_ptr.device,
        )
        task_options = dict(
            BODY_TASK_PIPELINE=True, TASK_SCORES=task_scores, TASK_PROBS=task_probs
        )
        if params.b >= 32 and params.seqlen_k <= 1536:
            task_options["PARTITION_SHORT_BODY"] = True
    if (
        cross_q_task
        and request_ids is not None
        and not body_task_pipeline
        and params.q_ptr.dtype == torch.bfloat16
        and params.h // params.hk in (4, 8)
        and (params.b >= 8 or (params.h // params.hk == 8 and params.b > 1))
        and params.total_q >= 8192
        and params.total_q > 128 * params.b
        and params.total_q < params.b * params.seqlen_q
        and 2304 < params.seqlen_q <= 4608
        and 2304 < params.seqlen_k <= 4608
    ):
        task_options["COMPACT_BODY_O"] = True
    long_mixed_stream = (
        params.q_ptr.dtype == torch.bfloat16
        and params.is_cu_seqlens_q
        and params.is_paged
        and params.is_causal
        and params.d == 256
        and params.block_size in (16, 32)
        and params.h // params.hk == 8
        and 1 < params.b < 32
        and params.total_q >= 8192
        and params.total_q > 128 * params.b
        and params.total_q < params.b * params.seqlen_q
        and 8192 < params.seqlen_q <= 32768
        and 32768 <= params.seqlen_k <= 73728
        and contig_k > 0
        and q_direct
        and plan.kv_splits == 1
        and task_m == 0
        and request_ids is not None
        and not compact_q1
        and not defer_merge
        and pipeline_kernel is ascend_attention_persistent_pipelined
    )
    if long_mixed_stream:
        task_options["LONG_MIXED_STREAM"] = True
        # Keep all O rows in UB, with BF16 storage between FP32 updates.
        task_options["COMPACT_BODY_O"] = True
        single_head_index = True
    prefill_task_queue = (
        params.q_ptr.dtype == torch.bfloat16
        and params.is_cu_seqlens_q
        and params.is_paged
        and params.is_causal
        and params.d == 256
        and params.b == 1
        and params.block_size in (16, 32)
        and params.h // params.hk in (4, 8)
        and 512 <= params.seqlen_q <= 1152
        and 512 <= params.seqlen_k <= 1536
        and params.total_q > 64
        and plan.kv_splits == 1
        and contig_k > 0
        and task_m == 0
        and request_ids is None
        and merge_q1 is None
        and not compact_q1
        and not defer_merge
    )
    if prefill_task_queue:
        from .prefill import ascend_attention_prefill_task_queue

        pipeline_kernel = ascend_attention_prefill_task_queue
        if (
            params.h == 4
            and params.total_q == params.seqlen_q == params.seqlen_k
            and 1024 < params.total_q <= 1152
        ):
            from .prefill import ascend_attention_prefill_clipped

            pipeline_kernel = ascend_attention_prefill_clipped
        if (
            params.h == 8
            and params.hk == 1
            and params.total_q == params.seqlen_q == params.seqlen_k
            and 1024 < params.total_q <= 1152
        ):
            from .prefill import ascend_attention_prefill_balanced

            pipeline_kernel = ascend_attention_prefill_balanced
        task_scores = torch.empty(
            (workers, 3, 128, 1536), dtype=torch.float32, device=params.q_ptr.device
        )
        task_probs = torch.empty(
            (workers, 3, 128, 1536),
            dtype=params.q_ptr.dtype,
            device=params.q_ptr.device,
        )
        task_options = dict(TASK_SCORES=task_scores, TASK_PROBS=task_probs)
    long_prefill_stream = (
        params.q_ptr.dtype == torch.bfloat16
        and params.is_cu_seqlens_q
        and params.is_paged
        and params.is_causal
        and params.d == 256
        and params.b == 1
        and params.block_size in (16, 32)
        and params.h // params.hk in (4, 8)
        and params.total_q == params.seqlen_q == params.seqlen_k
        and 4608 < params.total_q <= 65536
        and params.total_q % 128 == 0
        and plan.kv_splits == 1
        and contig_k > 0
        and task_m == 0
        and request_ids is None
        and merge_q1 is None
        and not compact_q1
        and not defer_merge
    )
    if long_prefill_stream or prefill_4k_stream:
        from .prefill import launch_long_prefill

        launch_long_prefill(
            params,
            workers,
            packed,
            scores,
            probs,
            pvwork,
            accwork,
            scale=1.0 if use_packed_k_prescale(params) else params.scale_softmax,
        )
        return
    stage_options = {}
    if native_split_q1:
        assert stage_q1 and task_m == 16 and plan.kv_splits in (3, 4)
        assert compact_q1 and request_ids is not None
        pipeline_kernel = ascend_attention_short_kv_staged
        stage_options["NATIVE_SPLIT_Q1"] = True
        stage_options["MIN_SPLIT_K"] = 2048 if short_split_q1 else 32768
    if stage_q1:
        if use_packed_k_prescale(params):
            stage_options["QUERY_SCALE"] = params.scale_softmax
        if not native_split_q1:
            assert short_kv and task_m == 16 and plan.kv_splits == 1
        stage_options.update(
            dict(
                STAGED_Q1=True,
                STREAM_Q1=True,
                RAW_PAGE_PAIR=(
                    native_split_q1
                    or params.h // params.hk == 8
                    or (params.h // params.hk == 4 and params.seqlen_k <= 2304)
                ),
                RAW_K=params.k_ptr,
                RAW_V=params.v_ptr,
            )
        )
    # Keep the fused Q1/compact-tail stream; specialize only actual Q1 tasks.
    if (
        stage_q1
        and not native_split_q1
        and task_m == 16
        and params.q_ptr.dtype == torch.bfloat16
        and (
            (params.h // params.hk == 8 and 1536 < params.seqlen_k <= 2304)
            or (
                params.h // params.hk == 4
                and params.b >= 32
                and params.b < params.total_q <= 2048 * params.b
                and params.total_q < params.b * params.seqlen_q
                and 512 <= params.seqlen_q <= 1152
                and 512 <= params.seqlen_k <= 1536
            )
        )
        and not compact_q1
        and not defer_merge
        and merge_q1 is None
    ):
        from .decode import ascend_attention_q1_coreloops_staged

        pipeline_kernel = ascend_attention_q1_coreloops_staged
    # Let native sparse-4K Q1 Cube/Vector loops advance independently.
    if (
        stage_q1
        and native_split_q1
        and short_split_q1
        and task_m == 16
        and plan.kv_splits == 4
        and params.q_ptr.dtype == torch.bfloat16
        and params.h // params.hk == 4
        and compact_q1
        and merge_q1 is None
    ):
        from .decode import ascend_attention_q1_native_coreloops_staged

        pipeline_kernel = ascend_attention_q1_native_coreloops_staged
    residual_wave = (
        stage_q1
        and not native_split_q1
        and task_m == 16
        and params.q_ptr.dtype == torch.bfloat16
        and params.h // params.hk == 8
        and params.b >= 32
        and params.b < params.total_q <= params.seqlen_q + params.b - 1
        and 512 <= params.seqlen_q <= 1152
        and 1536 < params.seqlen_k <= 2304
    )
    if residual_wave:
        wave_partial = torch.empty(
            (4, params.total_q, params.h, 256),
            dtype=torch.float32,
            device=params.q_ptr.device,
        )
        wave_lse = torch.empty(
            (4, params.h, params.total_q),
            dtype=torch.float32,
            device=params.q_ptr.device,
        )
        stage_options.update(
            RESIDUAL_WAVE=True, WAVE_PARTIAL=wave_partial, WAVE_LSE=wave_lse
        )
    merge_o = merge_lse = merge_ids = merge_count = None
    merge_splits_count = 0
    if merge_q1 is not None:
        merge_o, merge_lse, merge_ids, merge_count, merge_splits_count = merge_q1
        if merge_min_split_k != 32768:
            assert pipeline_kernel is ascend_attention_persistent_pipelined
            task_options["MERGE_MIN_SPLIT_K"] = merge_min_split_k
    if (
        cross_q_task
        and (task_options.get("COMPACT_BODY_O", False) or fp32_smallbatch_4k)
        and not body_task_pipeline
        and not stage_q1
        and pipeline_kernel is ascend_attention_persistent_pipelined
        and params.h // params.hk in (4, 8)
        and (
            params.b >= 8
            or (params.h // params.hk == 8 and params.b > 1)
            or fp32_smallbatch_4k
        )
        and params.total_q >= 8192
        and params.total_q > 128 * params.b
        and 2304 < params.seqlen_q <= 4608
        and 2304 < params.seqlen_k <= 4608
    ):
        from .mixed import ascend_attention_mixed_coreloops

        pipeline_kernel = ascend_attention_mixed_coreloops
        if params.h // params.hk == 8:
            from .mixed import ascend_attention_mixed_prob_stack

            pipeline_kernel = ascend_attention_mixed_prob_stack
    if long_mixed_stream:
        from .mixed import ascend_attention_mixed_coreloops

        pipeline_kernel = ascend_attention_mixed_coreloops
        if (
            (
                params.b < 8
                or (
                    params.b == 8
                    and params.total_q <= params.seqlen_q + params.b - 1 + 127
                )
            )
            and workers % params.h != 0
            and params.total_q > params.seqlen_q + params.b - 1
        ):
            from .mixed import ascend_attention_long_short_main, launch_long_short_split

            launch_long_short_split(
                params,
                workers,
                packed,
                scores,
                probs,
                pvwork,
                accwork,
                request_ids,
                request_count,
            )
            pipeline_kernel = ascend_attention_long_short_main
    if residual_merge is not None:
        assert pipeline_kernel is ascend_attention_persistent_pipelined
        assert task_m == 0 and plan.kv_splits == 1 and merge_q1 is None
        wave_o, wave_lse, wave_ids, wave_count = residual_merge
        task_options.update(
            RESIDUAL_MERGE=True,
            WAVE_PARTIAL=wave_o,
            WAVE_LSE=wave_lse,
            WAVE_IDS=wave_ids,
            WAVE_COUNT=wave_count,
        )
    pipeline_kernel[(workers,)](
        q,
        params.q_ptr,
        k,
        v,
        kernel_o,
        kernel_lse,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
        scores,
        probs,
        pvwork,
        accwork,
        params.b,
        params.h,
        params.hk,
        params.total_q,
        params.seqlen_q,
        1.0 if use_packed_k_prescale(params) else params.scale_softmax,
        params.q_row_stride,
        params.q_head_stride,
        params.q_batch_stride,
        row_stride,
        head_stride,
        params.o_batch_stride,
        params.k_page_stride,
        params.k_row_stride,
        params.k_head_stride,
        params.v_ptr.stride(0),
        params.v_row_stride,
        params.v_head_stride,
        params.page_table_batch_stride,
        params.block_size,
        params.is_cu_seqlens_q,
        params.is_causal,
        pipeline_bm,
        256,
        plan.kv_stack,
        plan.kv_splits,
        contig_k,
        QCAP=qcap,
        Q_DIRECT=q_direct,
        SINGLE_HEAD_INDEX=single_head_index,
        ACC_UB_ROWS=acc_ub_rows,
        TASK_M=task_m,
        SKIP_SHORT_KV=skip_short_kv,
        REQUEST_IDS=request_ids,
        REQUEST_COUNT=request_count,
        REQUEST_LIST=request_ids is not None
        and (compact_long_tail or not (short_kv and task_m == 16)),
        COMPACT_Q1=compact_q1,
        FINAL_O=params.o_ptr,
        FINAL_LSE=params.softmax_lse_ptr,
        FINAL_OR=params.o_row_stride,
        FINAL_OH=params.o_head_stride,
        FINAL_TOTAL_Q=params.total_q,
        BYPASS_SHORT=compact_q1,
        MERGE_O=merge_o,
        MERGE_LSE=merge_lse,
        MERGE_IDS=merge_ids,
        MERGE_COUNT=merge_count,
        MERGE_SPLITS=merge_splits_count,
        disable_auto_inject_block_sync=True,
        enable_fp_fusion=False,
        multibuffer=False,
        **(
            {"TAIL_ONLY": task_m == 16}
            if short_kv or native_split_q1
            else {"CROSS_Q_TASK": cross_q_task}
        ),
        **task_options,
        **stage_options,
    )
    if residual_wave:
        # The mandatory body launch consumes these partials; no extra merge launch.
        return wave_partial, wave_lse, request_ids, request_count
    if prefill_4k_stream and not inline_prefill_tail and 0 < params.total_q % 128 <= 16:
        # The stream omits compact tails, just as the mixed body path does.
        # Reuse the same packed inputs and explicitly finish those final rows.
        _launch_pipelined(
            params,
            plan,
            workers,
            packed,
            task_m=16,
            compact_long_tail=True,
        )
    if defer_merge:
        return kernel_o, kernel_lse, request_ids, request_count, plan.kv_splits
    if plan.kv_splits > 1:
        row_parallel_merge = (
            params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.is_causal
            and params.d == 256
            and params.block_size in (16, 32)
            and params.h // params.hk in (4, 8)
            and 1 < params.b <= 8
            and 1 < params.seqlen_q <= 256
            and 32768 <= params.seqlen_k <= 73728
            and not compact_q1
        )
        merge_splits[
            (
                (workers * 2,)
                if row_parallel_merge
                else (
                    (min(params.b * params.h, workers * 2),)
                    if request_ids is not None
                    else (params.b, params.h)
                )
            )
        ](
            kernel_o,
            kernel_lse,
            params.o_ptr,
            params.softmax_lse_ptr,
            params.cu_seqlens_q_ptr,
            params.seqused_k_ptr,
            params.o_row_stride,
            params.o_head_stride,
            params.total_q,
            params.h,
            params.d,
            plan.kv_splits,
            triton.next_power_of_2(plan.kv_splits),
            plan.kv_stack,
            SKIP_SHORT_KV=skip_short_kv,
            REQUEST_IDS=request_ids,
            REQUEST_COUNT=request_count,
            REQUEST_LIST=request_ids is not None,
            PARTIAL_Q=partial_q if compact_q1 else 0,
            BYPASS_SHORT=compact_q1,
            MIN_SPLIT_K=2048 if short_split_q1 else 32768,
            PARALLEL_Q=row_parallel_merge,
            B=params.b,
            enable_fp_fusion=False,
        )


# ---- launcher ----


def _launch_dimension_direct(params, plan):
    """Use dimension-masked paged tiles without D256-specific packing."""
    workers = torch.npu.get_device_limit(params.q_ptr.device.index)["cube_core_num"]
    block_d = triton.next_power_of_2(params.d)
    # Leave UB space for padding/masking D192 columns in the direct path.
    kv_stack = 128
    scores = torch.empty(
        (workers, plan.block_m, kv_stack),
        dtype=torch.float32,
        device=params.q_ptr.device,
    )
    probs = torch.empty_like(scores, dtype=params.q_ptr.dtype)
    pvwork = torch.empty(
        (workers, plan.block_m, block_d),
        dtype=torch.float32,
        device=params.q_ptr.device,
    )
    ascend_attention_persistent[(workers,)](
        params.q_ptr,
        params.k_ptr,
        params.v_ptr,
        params.o_ptr,
        params.softmax_lse_ptr,
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
        scores,
        probs,
        pvwork,
        params.b,
        params.h,
        params.hk,
        params.total_q,
        params.seqlen_q,
        params.scale_softmax,
        params.q_row_stride,
        params.q_head_stride,
        params.q_batch_stride,
        params.o_row_stride,
        params.o_head_stride,
        params.o_batch_stride,
        params.k_page_stride,
        params.k_row_stride,
        params.k_head_stride,
        params.v_ptr.stride(0),
        params.v_row_stride,
        params.v_head_stride,
        params.page_table_batch_stride,
        params.block_size,
        params.is_cu_seqlens_q,
        params.is_causal,
        plan.block_m,
        plan.block_n,
        kv_stack,
        D=params.d,
        BD=block_d,
        enable_fp_fusion=False,
        multibuffer=False,
    )


def launch_attention(params, *, num_splits=0):
    """Launch the Ascend forward path with a packed pipelined fast path."""
    plan = build_plan(params, num_splits)
    if params.q_ptr.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("Ascend attention supports FP16 and BF16")
    if not params.q_ptr.numel():
        return
    if any(
        params.o_ptr.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        for x in (
            params.q_ptr,
            params.k_ptr,
            params.v_ptr,
            params.cu_seqlens_q_ptr,
            params.seqused_k_ptr,
            params.page_table_ptr,
        )
        if x is not None and x.numel()
    ):
        raise ValueError("attention output must not alias inputs or metadata")
    with torch_device_fn.device(params.q_ptr.device):
        if params.d in (128, 192):
            _launch_dimension_direct(params, plan)
            return
        # Keep dynamic actual lengths on device, including graph replay.
        if (
            params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.seqlen_q == 1
            and params.h // params.hk in (4, 8)
            and params.b <= 8
            and params.b * params.hk <= 8
            and params.seqlen_k >= 32768
            and num_splits == 0
            and plan.kv_splits > 1
            and params.b
            * params.hk
            * triton.cdiv(params.seqlen_k, 512)
            * 512
            * params.d
            * params.k_ptr.element_size()
            * 2
            <= (1 << 30)
        ):
            from .decode import launch_decode_split_staged

            workers = torch.npu.get_device_limit(params.q_ptr.device.index)[
                "cube_core_num"
            ]
            launch_decode_split_staged(params, plan, workers)
            return
        if (
            params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.d == 256
            and params.block_size in (16, 32)
            and params.seqlen_q == 1
            and params.hk == 1
            and params.h // params.hk == 8
            and 32 <= params.b <= 128
            and params.seqlen_k > 32
            and params.k_ptr.shape[0] * params.block_size <= params.b * 4096
        ):
            launch_decode256_staged(params)
            return
        if (
            params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.seqlen_q == 1
            and params.h // params.hk in (4, 8)
            and params.b <= 8
            and params.b * params.hk <= 8
            and params.seqlen_k > 256
            and num_splits == 0
        ):
            fused_ids, fused_count = None, None
            if params.seqlen_k >= 32768:
                fused_ids, fused_count = allocate_long_request_list(params)
            workers = torch.npu.get_device_limit(params.q_ptr.device.index)[
                "cube_core_num"
            ]
            packed = prepare_packed_inputs(
                params, workers, fused_ids, fused_count, fuse_short_decode=True
            )
            if packed is not None:
                _launch_pipelined(
                    params, plan, workers, packed, 32, fused_ids, fused_count
                )
                return
        if (
            params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.seqlen_q == 1
            and params.h // params.hk in (4, 8)
            and 32 < params.seqlen_k <= 256
            and 16 <= params.b <= 256
        ):
            launch_paged_decode(params)
            return
        qdirect_cube = (
            params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.seqlen_q == 1
            and params.d == 256
            and params.block_size == 32
            and params.h == 16
            and params.hk == 2
            and params.b >= 512
            and params.b % 4 == 0
            and params.seqlen_k > 1
            and params.k_ptr.shape[0] * params.block_size <= params.b * 2048
            and params.q_head_stride == 256
            and params.q_ptr.stride(-1) == 1
            and params.k_ptr.stride()[1:] == (512, 256, 1)
            and params.v_ptr.stride()[1:] == (512, 256, 1)
        )
        if plan.short_decode and not qdirect_cube:
            launch_decode(
                params.q_ptr,
                params.k_ptr,
                params.v_ptr,
                params.o_ptr,
                params.softmax_lse_ptr,
                params.cu_seqlens_q_ptr,
                None,
                params.seqused_k_ptr,
                params.page_table_ptr,
                params.scale_softmax,
                params.seqlen_k,
            )
            return
        request_ids, request_count = None, None
        skip_short_kv = 0
        if (
            params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.seqlen_q == 1
            and params.h // params.hk <= 8
        ):
            if params.seqlen_k >= 32768:
                request_ids, request_count = allocate_long_request_list(params)
            if qdirect_cube:
                from .decode import launch_qdirect_decode

                launch_qdirect_decode(params, request_ids, request_count)
            else:
                launch_decode_actual(params, request_ids, request_count)
            if params.seqlen_k <= 32:
                return
            skip_short_kv = 32
        workers = torch.npu.get_device_limit(params.q_ptr.device.index)["cube_core_num"]
        local_q1 = (
            params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.is_causal
            and params.d == 256
            and params.block_size in (16, 32)
            and params.h // params.hk in (4, 8)
            and params.b
            >= (
                8
                if params.h // params.hk in (4, 8)
                and params.total_q >= 8192
                and (
                    2304 < params.seqlen_k <= 4608
                    or (
                        512 <= params.seqlen_k <= 1536
                        and 4 * params.total_q <= 3 * params.b * params.seqlen_q
                    )
                )
                else 32
            )
            and params.b
            < params.total_q
            <= (
                2048
                if (params.h // params.hk == 8 and 512 <= params.seqlen_k <= 4608)
                or (params.h // params.hk == 4 and params.seqlen_k <= 1536)
                or (
                    params.h // params.hk == 4
                    and 2304 < params.seqlen_k <= 4608
                    and params.total_q >= 8192
                )
                else 128
            )
            * params.b
            and params.total_q < params.b * params.seqlen_q
            and 512
            <= params.seqlen_q
            <= (
                4608
                if (params.h // params.hk == 8 or params.total_q >= 8192)
                and 2304 < params.seqlen_k <= 4608
                else 1152
            )
            and (
                512 <= params.seqlen_k <= 2304
                or (
                    num_splits == 0
                    and params.h // params.hk in (4, 8)
                    and 2304 < params.seqlen_k <= 4608
                )
            )
            and plan.kv_splits == 1
            and params.b * params.hk * triton.cdiv(params.seqlen_k, 512) * 512 * 256 * 4
            <= (1 << 30)
        )
        if local_q1:
            fuse_classification = (
                (
                    (
                        params.h // params.hk == 4
                        and params.b >= 32
                        and params.seqlen_q + params.b - 1
                        < params.total_q
                        <= 128 * params.b
                        and 512 <= params.seqlen_q <= 1152
                        and 512 <= params.seqlen_k <= 1536
                        and use_packed_k_prescale(params)
                    )
                    or (
                        params.h // params.hk == 8
                        and params.b >= 32
                        and params.b < params.total_q <= 128 * params.b
                        and 512 <= params.seqlen_q <= 1152
                        and 1536 < params.seqlen_k <= 2304
                    )
                    or (
                        params.h // params.hk == 4
                        and params.b >= 32
                        and params.b < params.total_q <= params.seqlen_q + params.b - 1
                        and 512 <= params.seqlen_q <= 1152
                        and 2304 < params.seqlen_k <= 4608
                    )
                )
                and params.q_ptr.stride(-1) == 1
                and params.q_row_stride >= 256
                and params.q_head_stride >= 256
                and params.q_row_stride % 16 == 0
                and params.q_head_stride % 16 == 0
                and params.q_ptr.data_ptr() % 32 == 0
            )
            decode_ids, other_ids, counts = build_mixed_request_lists(
                params, defer=fuse_classification
            )
            packed = prepare_packed_inputs(
                params,
                workers,
                other_ids,
                counts[1:],
                stage_workers=workers,
                fused_mixed_lists=(
                    (decode_ids, other_ids, counts) if fuse_classification else None
                ),
            )
            assert packed is not None
            # Split the raw-paged Q1 stream; keep the body on its original D1 path.
            dense_native_q1_4k = (
                params.h // params.hk == 4
                and params.b >= 32
                and 2304 < params.seqlen_q <= 4608
                and 2304 < params.seqlen_k <= 4608
                and params.total_q >= 8192
                and params.total_q > params.seqlen_q + params.b - 1
            )
            split_q1_4k = (
                params.h // params.hk == 4
                and 2304 < params.seqlen_k <= 4608
                and params.total_q <= params.seqlen_q + params.b - 1
            )
            split_q1_4k = split_q1_4k or dense_native_q1_4k
            q1_plan = replace(plan, kv_splits=4) if split_q1_4k else plan
            q1_partials = _launch_pipelined(
                params,
                q1_plan,
                workers,
                packed,
                request_ids=decode_ids,
                request_count=counts,
                task_m=16,
                compact_q1=split_q1_4k,
                stage_q1=True,
                native_split_q1=split_q1_4k,
                short_split_q1=split_q1_4k,
                defer_merge=split_q1_4k and not dense_native_q1_4k,
            )
            _launch_pipelined(
                params,
                plan,
                workers,
                packed,
                request_ids=other_ids,
                request_count=counts[1:],
                task_m=0,
                local_body_stream=True,
                merge_q1=(
                    q1_partials if split_q1_4k and not dense_native_q1_4k else None
                ),
                merge_min_split_k=2048 if split_q1_4k else 32768,
                residual_merge=None if split_q1_4k else q1_partials,
                body_task_pipeline=(
                    (params.h // params.hk == 8 and params.seqlen_k <= 2304)
                    or (params.h // params.hk == 4 and params.seqlen_k <= 1536)
                ),
            )
            if split_q1_4k:
                # The old unsplit Q1 stream also owns tiny Q>1 tails.
                # Native Split4 visits only decode_ids; finish those tails here.
                _launch_pipelined(
                    params,
                    plan,
                    workers,
                    packed,
                    request_ids=other_ids,
                    request_count=counts[1:],
                    task_m=16,
                    compact_long_tail=True,
                )
            return
        native_split_q1 = (
            num_splits == 0
            and params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.is_causal
            and params.d == 256
            and params.block_size in (16, 32)
            and params.h // params.hk == 8
            and 1 < params.b <= 8
            and 1 < params.seqlen_q <= 64
            and params.b < params.total_q < params.b * params.seqlen_q
            and 32768 <= params.seqlen_k <= 73728
            and plan.kv_splits == 4
            and params.b * params.hk * triton.cdiv(params.seqlen_k, 512) * 512 * 256 * 4
            <= (1 << 30)
        )
        if native_split_q1:
            decode_ids, other_ids, counts = build_mixed_request_lists(params)
            packed = prepare_packed_inputs(
                params, workers, other_ids, counts[1:], stage_workers=workers
            )
            assert packed is not None
            # TOTAL_Q > B guarantees at least one non-Q1 request. Bound
            # Q1 work by (B - 1) * HK and avoid a partly occupied second wave.
            q1_splits = workers // ((params.b - 1) * params.hk)
            q1_plan = replace(plan, kv_splits=3) if q1_splits == 3 else plan
            merge_q1 = _launch_pipelined(
                params,
                q1_plan,
                workers,
                packed,
                request_ids=decode_ids,
                request_count=counts,
                task_m=16,
                compact_q1=True,
                defer_merge=True,
                stage_q1=True,
                native_split_q1=True,
            )
            _launch_pipelined(
                params,
                plan,
                workers,
                packed,
                request_ids=other_ids,
                request_count=counts[1:],
                task_m=0,
                merge_q1=merge_q1,
            )
            return
        # Long-context mixed batches need contiguous KV only for Q>1.
        # Q1 reuses the existing raw-page S4 kernel and its fused merge.
        native_long_body_q1 = (
            num_splits == 0
            and params.q_ptr.dtype == torch.bfloat16
            and params.is_cu_seqlens_q
            and params.is_paged
            and params.is_causal
            and params.d == 256
            and params.block_size in (16, 32)
            and params.h // params.hk in (4, 8)
            and (
                (
                    1 < params.b <= 8
                    and params.b < params.total_q <= params.seqlen_q + params.b - 1
                    and 64 < params.seqlen_q <= 144
                )
                or (
                    params.h // params.hk == 8
                    and 1 < params.b < 8
                    and params.total_q >= 8192
                    and params.total_q > 128 * params.b
                    and params.total_q < params.b * params.seqlen_q
                    and 8192 < params.seqlen_q <= 32768
                )
                or (
                    params.h // params.hk == 8
                    and 8 <= params.b < 32
                    and 128 * params.b < params.total_q <= 2048 * params.b
                    and params.total_q < params.b * params.seqlen_q
                    and params.seqlen_q > 144
                )
            )
            and params.b * params.hk < workers
            and 32768 <= params.seqlen_k <= 73728
            and plan.kv_splits == 1
            and params.b * params.hk * triton.cdiv(params.seqlen_k, 512) * 512 * 256 * 4
            <= (1 << 30)
        )
        if native_long_body_q1:
            decode_ids, other_ids, counts = build_mixed_request_lists(params)
            packed = prepare_packed_inputs(
                params, workers, other_ids, counts[1:], stage_workers=workers
            )
        else:
            packed = prepare_packed_inputs(params, workers, request_ids, request_count)
        if packed is not None:
            mixed_decode = (
                params.q_ptr.dtype == torch.bfloat16
                and params.is_cu_seqlens_q
                and params.is_paged
                and params.b > 1
                and params.seqlen_q > 1
                and params.b < params.total_q < params.b * params.seqlen_q
                and params.h // params.hk in (4, 8)
                and plan.kv_splits == 1
            )
            if mixed_decode:
                if not native_long_body_q1:
                    decode_ids, other_ids, counts = build_mixed_request_lists(params)
                decode_plan = build_mixed_decode_plan(params, plan, num_splits, workers)
                # A padded M128 tail creates a second worker wave for H>workers/2.
                # Reuse separate fixed-layout M128/M16 kernels; retain unsplit KV.
                compact_long_tail = (
                    params.is_causal
                    and params.d == 256
                    and params.block_size in (16, 32)
                    and 1 < params.b <= 8
                    and params.b < params.total_q <= params.seqlen_q + params.b - 1
                    and 128 < params.seqlen_q <= 144
                    and 32768 <= params.seqlen_k <= 73728
                    and workers // 2 < params.h <= workers
                    and plan.kv_splits == 1
                )
                # A sparse Q>1 body otherwise leaves most Cube workers idle.
                # Fill whole head waves without shrinking the matrix tiles.
                body_split_long = (
                    num_splits == 0
                    and params.q_ptr.dtype == torch.bfloat16
                    and params.is_cu_seqlens_q
                    and params.is_paged
                    and params.is_causal
                    and params.d == 256
                    and params.block_size in (16, 32)
                    and params.h // params.hk in (4, 8)
                    and 1 < params.b <= 8
                    and params.b < params.total_q <= params.seqlen_q + params.b - 1
                    and 64 < params.seqlen_q <= 256
                    and 32768 <= params.seqlen_k <= 73728
                    and params.h <= workers // 4
                    and plan.kv_splits == 1
                )
                body_plan = (
                    replace(plan, kv_splits=min(16, workers // gcd(workers, params.h)))
                    if body_split_long
                    else plan
                )
                merge_q1 = _launch_pipelined(
                    params,
                    decode_plan,
                    workers,
                    packed,
                    request_ids=decode_ids,
                    request_count=counts,
                    task_m=16,
                    compact_q1=decode_plan.kv_splits > 1,
                    defer_merge=decode_plan.kv_splits > 1,
                    stage_q1=native_long_body_q1,
                    native_split_q1=native_long_body_q1,
                )
                split_full_body = (
                    num_splits == 0
                    and params.is_causal
                    and params.d == 256
                    and params.block_size in (16, 32)
                    and params.h // params.hk in (4, 8)
                    and 1 < params.b <= 8
                    and params.b < params.total_q <= params.seqlen_q + params.b - 1
                    and 128 < params.seqlen_q <= 132
                    and 32768 <= params.seqlen_k <= 73728
                    and (
                        workers // 4 < params.h <= workers // 2
                        or (params.h // params.hk == 4 and params.h <= workers // 4)
                    )
                )
                if split_full_body:
                    launch_split_full_body(
                        params,
                        plan,
                        workers,
                        packed,
                        other_ids,
                        counts[1:],
                        merge_q1,
                        body_splits=body_plan.kv_splits if body_split_long else 2,
                    )
                    launch_grouped_tail(
                        params, plan, workers, packed, other_ids, counts[1:]
                    )
                    return
                dense_body_pipeline = (
                    num_splits == 0
                    and params.q_ptr.dtype == torch.bfloat16
                    and params.is_cu_seqlens_q
                    and params.is_paged
                    and params.is_causal
                    and params.d == 256
                    and params.block_size in (16, 32)
                    and params.h // params.hk in (4, 8)
                    and (
                        params.b >= 32
                        or (
                            params.h // params.hk in (4, 8)
                            and 1 < params.b < 32
                            and params.seqlen_k <= 1536
                        )
                    )
                    and params.total_q > 128 * params.b
                    and 512 <= params.seqlen_q <= 1152
                    and 512 <= params.seqlen_k <= 2304
                    and plan.kv_splits == 1
                    and merge_q1 is None
                    and not compact_long_tail
                )
                _launch_pipelined(
                    params,
                    body_plan,
                    workers,
                    packed,
                    request_ids=other_ids,
                    request_count=counts[1:],
                    task_m=0,
                    merge_q1=merge_q1,
                    compact_long_tail=compact_long_tail,
                    body_task_pipeline=dense_body_pipeline,
                )
                if compact_long_tail:
                    grouped_tail = (
                        num_splits == 0
                        and params.q_ptr.dtype == torch.bfloat16
                        and params.h // params.hk == 8
                        and params.seqlen_q <= 132
                    )
                    if grouped_tail:
                        launch_grouped_tail(
                            params, plan, workers, packed, other_ids, counts[1:]
                        )
                    else:
                        _launch_pipelined(
                            params,
                            plan,
                            workers,
                            packed,
                            request_ids=other_ids,
                            request_count=counts[1:],
                            task_m=16,
                            compact_long_tail=True,
                        )
                return
            _launch_pipelined(
                params, plan, workers, packed, skip_short_kv, request_ids, request_count
            )
            return

        scores = torch.empty(
            (workers, plan.block_m, plan.kv_stack),
            dtype=torch.float32,
            device=params.q_ptr.device,
        )
        probs = torch.empty_like(scores, dtype=params.q_ptr.dtype)
        pvwork = torch.empty(
            (workers, plan.block_m, params.d),
            dtype=torch.float32,
            device=params.q_ptr.device,
        )
        kernel_o, kernel_lse = params.o_ptr, params.softmax_lse_ptr
        row_stride, head_stride = params.o_row_stride, params.o_head_stride
        if plan.kv_splits > 1:
            kernel_o = torch.empty(
                (plan.kv_splits, params.total_q, params.h, params.d),
                dtype=torch.float32,
                device=params.q_ptr.device,
            )
            kernel_lse = torch.empty(
                (plan.kv_splits, params.h, params.total_q),
                dtype=torch.float32,
                device=params.q_ptr.device,
            )
            row_stride, head_stride = params.h * params.d, params.d
        k, v, contiguous_k = prepare_contiguous_kv(params, workers)
        ascend_attention_persistent[(workers,)](
            params.q_ptr,
            k,
            v,
            kernel_o,
            kernel_lse,
            params.cu_seqlens_q_ptr,
            params.seqused_k_ptr,
            params.page_table_ptr,
            scores,
            probs,
            pvwork,
            params.b,
            params.h,
            params.hk,
            params.total_q,
            params.seqlen_q,
            params.scale_softmax,
            params.q_row_stride,
            params.q_head_stride,
            params.q_batch_stride,
            row_stride,
            head_stride,
            params.o_batch_stride,
            params.k_page_stride,
            params.k_row_stride,
            params.k_head_stride,
            params.v_ptr.stride(0),
            params.v_row_stride,
            params.v_head_stride,
            params.page_table_batch_stride,
            params.block_size,
            params.is_cu_seqlens_q,
            params.is_causal,
            plan.block_m,
            256 if contiguous_k else plan.block_n,
            plan.kv_stack,
            plan.kv_splits,
            contiguous_k,
            SKIP_SHORT_KV=skip_short_kv,
            REQUEST_IDS=request_ids,
            REQUEST_COUNT=request_count,
            REQUEST_LIST=request_ids is not None,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        if plan.kv_splits > 1:
            merge_splits[
                (
                    (min(params.b * params.h, workers * 2),)
                    if request_ids is not None
                    else (params.b, params.h)
                )
            ](
                kernel_o,
                kernel_lse,
                params.o_ptr,
                params.softmax_lse_ptr,
                params.cu_seqlens_q_ptr,
                params.seqused_k_ptr,
                params.o_row_stride,
                params.o_head_stride,
                params.total_q,
                params.h,
                params.d,
                plan.kv_splits,
                triton.next_power_of_2(plan.kv_splits),
                plan.kv_stack,
                SKIP_SHORT_KV=skip_short_kv,
                REQUEST_IDS=request_ids,
                REQUEST_COUNT=request_count,
                REQUEST_LIST=request_ids is not None,
                enable_fp_fusion=False,
            )
